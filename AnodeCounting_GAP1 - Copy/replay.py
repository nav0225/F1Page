"""
Offline replay of recorded footage through the production counting logic.

Every frame of the file is processed in order (no frame dropping), so results
are deterministic and comparable between code versions. Nothing is written to
SQL Server.

Usage:
    python replay.py --video Video1.mp4 --out runs/replay_video1 --device cpu
    python replay.py --video clip.mp4 --env .env --save-video
    python replay.py --video Video1.mp4 --out runs/legacy --evidence frame --conf 0.35
    python replay.py --video Video1.mp4 --out runs/haze --haze 0.6,6,3

--haze ALPHA[,PERIOD_S,DURATION_S] overlays a synthetic grey veil (plus blur)
to stress-test counting under reduced visibility: ALPHA is the veil strength
(0-1); with PERIOD_S/DURATION_S the veil is present for DURATION_S out of
every PERIOD_S seconds. It is a controlled stand-in for fumes, not a
substitute for real fume footage.

Outputs in --out:
    holes.csv     one row per counted hole crossing
    rejected.csv  crossings without enough confident detections (not counted)
    anodes.csv    one row per anode emitted by the grouper
    summary.json  totals, settings used and per-stage timing percentiles
    annotated.mp4 (only with --save-video)
"""
import argparse
import csv
import dataclasses
import json
import os
import sys
import time

import cv2
from dotenv import dotenv_values

from config import load_settings
from counting import ModuloGrouper
from overlay import draw_overlay
from pipeline import FrameProcessor, build_counter


def percentiles(values, ps=(50, 95, 99)):
    if not values:
        return {}
    ordered = sorted(values)
    out = {}
    for p in ps:
        k = min(len(ordered) - 1, max(0, int(round(p / 100.0 * (len(ordered) - 1)))))
        out[f"p{p}"] = round(ordered[k], 2)
    out["max"] = round(ordered[-1], 2)
    out["mean"] = round(sum(ordered) / len(ordered), 2)
    return out


def build_settings(args):
    env = dict(dotenv_values(args.env)) if args.env else {}
    env.setdefault("COUNT_DIRECTION", args.direction or "up")
    env["SOURCE"] = "file"
    env["VIDEO_PATH"] = args.video
    settings = load_settings(env, require_sql=False)
    overrides = {}
    if args.model:
        overrides["model_path"] = args.model
    if args.device:
        overrides["device"] = args.device
    if args.line_y is not None:
        overrides["line_y"] = args.line_y
    if args.direction:
        overrides["count_direction"] = args.direction
    if getattr(args, "evidence", None):
        overrides["count_evidence"] = args.evidence
    if getattr(args, "conf", None) is not None:
        overrides["conf_thres"] = args.conf
    if getattr(args, "count_min_conf", None) is not None:
        overrides["count_min_conf"] = args.count_min_conf
    if getattr(args, "min_hits", None) is not None:
        overrides["count_min_hits"] = args.min_hits
    return dataclasses.replace(settings, **overrides)


def parse_haze(spec):
    """'ALPHA[,PERIOD_S,DURATION_S]' -> (alpha, period_s or None, duration_s or None)"""
    if not spec:
        return None
    parts = [float(p) for p in spec.split(",")]
    if len(parts) == 1:
        return parts[0], None, None
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    raise SystemExit("--haze expects ALPHA or ALPHA,PERIOD_S,DURATION_S")


def apply_haze(frame, alpha):
    """Blend towards uniform grey and blur: reduces contrast like a fume veil."""
    grey = frame.mean()
    hazed = cv2.addWeighted(frame, 1.0 - alpha, frame * 0 + int(grey), alpha, 0)
    return cv2.GaussianBlur(hazed, (0, 0), 1.0 + 3.0 * alpha)


def run(args, detector_factory=None):
    settings = build_settings(args)
    os.makedirs(args.out, exist_ok=True)

    if detector_factory is None:
        from detector import YoloTracker
        detector = YoloTracker(settings.model_path, settings.conf_thres, settings.iou_thres, settings.device)
    else:
        detector = detector_factory(settings)

    counter = build_counter(settings)
    grouper = ModuloGrouper(settings.holes_per_anode)
    processor = FrameProcessor(detector, counter, grouper)
    haze = parse_haze(getattr(args, "haze", None))
    hazed_frames = 0

    cap = cv2.VideoCapture(settings.video_path)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {settings.video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.save_video:
        writer = cv2.VideoWriter(os.path.join(args.out, "annotated.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps or 20.0, (width, height))

    hole_rows, anode_rows, rejected_rows = [], [], []
    timings = {"decode": [], "detect": [], "count": [], "total": []}
    line = (settings.line_x1, settings.line_y, settings.line_x2)
    frame_id = 0
    wall_start = time.perf_counter()

    while True:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        t_decode = (time.perf_counter() - t0) * 1000.0
        if not ok or frame is None:
            break
        frame_id += 1
        if args.max_frames and frame_id > args.max_frames:
            break
        video_t = (frame_id - 1) / fps if fps else None

        if haze is not None:
            alpha, period, duration = haze
            if period is None or (video_t or 0.0) % period < duration:
                frame = apply_haze(frame, alpha)
                hazed_frames += 1

        result = processor.process(frame_id, frame)
        timings["decode"].append(t_decode)
        timings["detect"].append(result.timings_ms["detect"])
        timings["count"].append(result.timings_ms["count"])
        timings["total"].append(t_decode + result.timings_ms["detect"] + result.timings_ms["count"])

        for h in result.holes:
            hole_rows.append({"hole_seq": h.seq, "frame_id": h.frame_id, "video_time_s": video_t,
                              "track_id": h.track_id, "cx": round(h.cx, 1), "cy": round(h.cy, 1),
                              "conf": None if h.conf is None else round(h.conf, 4)})
        for r in result.rejected:
            rejected_rows.append({"frame_id": r.frame_id, "video_time_s": (r.frame_id - 1) / fps if fps else None,
                                  "track_id": r.track_id, "cx": round(r.cx, 1), "cy": round(r.cy, 1),
                                  "max_conf": None if r.max_conf is None else round(r.max_conf, 4),
                                  "strong_hits": r.strong_hits, "reason": r.reason})
        for g in result.anodes:
            anode_rows.append({"anode_seq": g.seq, "frame_id": result.frame_id, "video_time_s": video_t,
                               "hole_start": g.hole_start, "hole_end": g.hole_end,
                               "hole_track_ids": " ".join(str(h.track_id) for h in g.holes)})

        if writer is not None:
            writer.write(draw_overlay(frame, result, counter, line, counter.hole_count, grouper.anode_count, fps))

    cap.release()
    if writer is not None:
        writer.release()
    # let tracks still in view expire so pending crossings are resolved
    for extra in range(settings.track_stale_frames + 2):
        counter.update(frame_id + 1 + extra, [])
        for r in counter.drain_rejected():
            rejected_rows.append({"frame_id": r.frame_id, "video_time_s": (r.frame_id - 1) / fps if fps else None,
                                  "track_id": r.track_id, "cx": round(r.cx, 1), "cy": round(r.cy, 1),
                                  "max_conf": None if r.max_conf is None else round(r.max_conf, 4),
                                  "strong_hits": r.strong_hits, "reason": r.reason})
    wall_s = time.perf_counter() - wall_start

    for name, rows, fields in (("holes.csv", hole_rows, ["hole_seq", "frame_id", "video_time_s", "track_id", "cx", "cy", "conf"]),
                               ("rejected.csv", rejected_rows, ["frame_id", "video_time_s", "track_id", "cx", "cy",
                                                                "max_conf", "strong_hits", "reason"]),
                               ("anodes.csv", anode_rows, ["anode_seq", "frame_id", "video_time_s", "hole_start", "hole_end", "hole_track_ids"])):
        with open(os.path.join(args.out, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    summary = {
        "video": settings.video_path,
        "frames_processed": frame_id if not args.max_frames else min(frame_id, args.max_frames),
        "video_fps": fps,
        "resolution": [width, height],
        "holes_counted": counter.hole_count,
        "anodes_counted": grouper.anode_count,
        "crossings_rejected": len(rejected_rows),
        "crossings_outside_roi": counter.outside_roi,
        "hazed_frames": hazed_frames,
        "settings": {k: v for k, v in dataclasses.asdict(settings).items()
                     if not k.startswith("sql_") and k != "rtsp_url"},
        "timing_ms": {k: percentiles(v) for k, v in timings.items()},
        "wall_time_s": round(wall_s, 2),
        "processing_fps": round(len(timings["total"]) / wall_s, 2) if wall_s > 0 else None,
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--env", help="optional .env file to take line/model settings from")
    p.add_argument("--model")
    p.add_argument("--device", help="'cpu' or a CUDA index such as '0'")
    p.add_argument("--line-y", type=int)
    p.add_argument("--direction", choices=["up", "down"])
    p.add_argument("--evidence", choices=["track", "frame"], help="counting evidence mode (frame = legacy)")
    p.add_argument("--conf", type=float, help="detector confidence passed to the tracker (CONF_THRES)")
    p.add_argument("--count-min-conf", type=float)
    p.add_argument("--min-hits", type=int)
    p.add_argument("--haze", help="synthetic visibility loss: ALPHA[,PERIOD_S,DURATION_S]")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--save-video", action="store_true")
    args = p.parse_args(argv)
    summary = run(args)
    print(json.dumps({k: summary[k] for k in ("frames_processed", "holes_counted", "anodes_counted",
                                               "crossings_rejected", "hazed_frames", "processing_fps")}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
