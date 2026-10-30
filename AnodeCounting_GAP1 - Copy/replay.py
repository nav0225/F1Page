"""
Offline replay of recorded footage through the production counting logic.

Every frame of the file is processed in order (no frame dropping), so results
are deterministic and comparable between code versions. Nothing is written to
SQL Server.

Usage:
    python replay.py --video Video1.mp4 --out runs/replay_video1 --device cpu
    python replay.py --video clip.mp4 --env .env --save-video

Outputs in --out:
    holes.csv     one row per counted hole crossing
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
from counting import LineCrossingCounter, ModuloGrouper
from overlay import draw_overlay
from pipeline import FrameProcessor


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
    return dataclasses.replace(settings, **overrides)


def run(args, detector_factory=None):
    settings = build_settings(args)
    os.makedirs(args.out, exist_ok=True)

    if detector_factory is None:
        from detector import YoloTracker
        detector = YoloTracker(settings.model_path, settings.conf_thres, settings.iou_thres, settings.device)
    else:
        detector = detector_factory(settings)

    counter = LineCrossingCounter(settings.line_y, settings.count_direction,
                                  settings.count_min_conf, settings.track_stale_frames)
    grouper = ModuloGrouper(settings.holes_per_anode)
    processor = FrameProcessor(detector, counter, grouper)

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

    hole_rows, anode_rows = [], []
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

        result = processor.process(frame_id, frame)
        timings["decode"].append(t_decode)
        timings["detect"].append(result.timings_ms["detect"])
        timings["count"].append(result.timings_ms["count"])
        timings["total"].append(t_decode + result.timings_ms["detect"] + result.timings_ms["count"])

        for h in result.holes:
            hole_rows.append({"hole_seq": h.seq, "frame_id": h.frame_id, "video_time_s": video_t,
                              "track_id": h.track_id, "cx": round(h.cx, 1), "cy": round(h.cy, 1),
                              "conf": None if h.conf is None else round(h.conf, 4)})
        for g in result.anodes:
            anode_rows.append({"anode_seq": g.seq, "frame_id": result.frame_id, "video_time_s": video_t,
                               "hole_start": g.hole_start, "hole_end": g.hole_end,
                               "hole_track_ids": " ".join(str(h.track_id) for h in g.holes)})

        if writer is not None:
            writer.write(draw_overlay(frame, result, counter, line, counter.hole_count, grouper.anode_count, fps))

    cap.release()
    if writer is not None:
        writer.release()
    wall_s = time.perf_counter() - wall_start

    for name, rows, fields in (("holes.csv", hole_rows, ["hole_seq", "frame_id", "video_time_s", "track_id", "cx", "cy", "conf"]),
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
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--save-video", action="store_true")
    args = p.parse_args(argv)
    summary = run(args)
    print(json.dumps({k: summary[k] for k in ("frames_processed", "holes_counted", "anodes_counted",
                                               "processing_fps", "timing_ms")}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
