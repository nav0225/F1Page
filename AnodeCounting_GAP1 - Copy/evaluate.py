"""
Score a replay (replay.py output) against ground truth.

Ground truth CSV columns:
    event         'hole' (a hole crossing the counting line) or 'boundary'
                  (the gap between two anodes crossing the line)
    video_time_s  time in the video
    anode         index of the physical anode the hole belongs to (holes only)

Metrics:
    holes   one-to-one matching of counted holes to true holes within
            --tolerance seconds: recall, precision, misses, extras, timing error.
    anodes  grouping purity: an emitted anode is 'pure' if all of its matched
            holes belong to one physical anode, 'mixed' otherwise. Mixed anodes
            are records that do not correspond to a real anode.

Usage:
    python evaluate.py --replay runs/replay_video1 --truth validation/video1_ground_truth.csv
    python evaluate.py --replay runs/replay_video1 --truth ... --anodes anodes_shadow.csv
"""
import argparse
import csv
import json
import os
import sys


def load_truth(path):
    holes, boundaries = [], []
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(line for line in f if not line.startswith("#"))]
    for r in rows:
        t = float(r["video_time_s"])
        if r["event"] == "hole":
            holes.append((t, int(r["anode"])))
        elif r["event"] == "boundary":
            boundaries.append(t)
    return sorted(holes), sorted(boundaries)


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def match_holes(counted, truth, tolerance):
    """Greedy one-to-one matching by time. counted: [(time, seq)], truth: [(time, anode)]."""
    pairs = sorted(
        (abs(c[0] - t[0]), ci, ti)
        for ci, c in enumerate(counted) for ti, t in enumerate(truth)
        if abs(c[0] - t[0]) <= tolerance
    )
    used_c, used_t, matches = set(), set(), {}
    for _, ci, ti in pairs:
        if ci in used_c or ti in used_t:
            continue
        used_c.add(ci)
        used_t.add(ti)
        matches[ci] = ti
    return matches


def evaluate(replay_dir, truth_path, tolerance=0.5, anodes_file="anodes.csv"):
    truth_holes, _ = load_truth(truth_path)
    holes = load_csv(os.path.join(replay_dir, "holes.csv"))
    anodes = load_csv(os.path.join(replay_dir, anodes_file))
    counted = [(float(h["video_time_s"]), int(h["hole_seq"])) for h in holes]

    matches = match_holes(counted, truth_holes, tolerance)
    timing = [counted[ci][0] - truth_holes[ti][0] for ci, ti in matches.items()]
    hole_metrics = {
        "true": len(truth_holes),
        "counted": len(counted),
        "matched": len(matches),
        "missed": len(truth_holes) - len(matches),
        "extra": len(counted) - len(matches),
        "recall": round(len(matches) / len(truth_holes), 4) if truth_holes else None,
        "precision": round(len(matches) / len(counted), 4) if counted else None,
        "mean_abs_timing_error_s": round(sum(abs(x) for x in timing) / len(timing), 3) if timing else None,
    }

    anode_of_seq = {counted[ci][1]: truth_holes[ti][1] for ci, ti in matches.items()}
    pure = mixed = 0
    details = []
    for a in anodes:
        seqs = range(int(a["hole_start"]), int(a["hole_end"]) + 1)
        physical = sorted({anode_of_seq[s] for s in seqs if s in anode_of_seq})
        is_pure = len(physical) == 1
        pure += is_pure
        mixed += not is_pure
        details.append({"anode_seq": int(a["anode_seq"]), "holes": f"{a['hole_start']}-{a['hole_end']}",
                        "physical_anodes": physical, "pure": is_pure,
                        "status": a.get("status") or None, "slots": a.get("slots") or None})
    anode_metrics = {
        "physical_anodes_with_holes": len({a for _, a in truth_holes}),
        "emitted": len(anodes),
        "pure": pure,
        "mixed": mixed,
        "details": details,
    }
    return {"replay": replay_dir, "anodes_file": anodes_file, "tolerance_s": tolerance,
            "holes": hole_metrics, "anodes": anode_metrics}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--replay", required=True, help="output folder of replay.py")
    p.add_argument("--truth", required=True)
    p.add_argument("--tolerance", type=float, default=0.5, help="seconds")
    p.add_argument("--anodes", default="anodes.csv", help="anodes.csv (counted) or anodes_shadow.csv")
    args = p.parse_args(argv)
    print(json.dumps(evaluate(args.replay, args.truth, args.tolerance, args.anodes), indent=2))


if __name__ == "__main__":
    sys.exit(main())
