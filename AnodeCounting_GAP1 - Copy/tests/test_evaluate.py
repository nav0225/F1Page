import csv
import os

from evaluate import evaluate, match_holes

TRUTH = """# test truth: 2 physical anodes, the first only partly in view
event,video_time_s,anode
hole,1.0,0
hole,2.0,0
boundary,2.5,
hole,3.0,1
hole,4.0,1
hole,5.0,1
hole,6.0,1
"""


def write(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def test_counts_misses_extras_and_mixed_anodes(tmp_path):
    truth = tmp_path / "truth.csv"
    truth.write_text(TRUTH)
    replay = tmp_path / "replay"
    os.makedirs(replay)
    # hole at 4.0 missed; spurious hole at 7.9; timing slightly off
    holes = [(1, 1.1), (2, 2.2), (3, 3.0), (4, 5.1), (5, 5.9), (6, 7.9)]
    write(replay / "holes.csv", [{"hole_seq": s, "video_time_s": t} for s, t in holes],
          ["hole_seq", "video_time_s"])
    # modulo-4 grouping: holes 1-4 mix anode 0 and anode 1
    write(replay / "anodes.csv", [{"anode_seq": 1, "hole_start": 1, "hole_end": 4}],
          ["anode_seq", "hole_start", "hole_end"])

    r = evaluate(str(replay), str(truth), tolerance=0.5)
    assert r["holes"]["true"] == 6 and r["holes"]["counted"] == 6
    assert r["holes"]["matched"] == 5 and r["holes"]["missed"] == 1 and r["holes"]["extra"] == 1
    assert r["anodes"]["emitted"] == 1 and r["anodes"]["mixed"] == 1
    assert r["anodes"]["details"][0]["physical_anodes"] == [0, 1]


def test_matching_maximises_matches_before_timing_error():
    # closest pair first (0.9-0.5) would leave 0.0 and 1.4 unmatched
    counted = [(0.0, 1), (0.9, 2)]
    truth = [(0.5, 0), (1.4, 0)]
    assert match_holes(counted, truth, tolerance=0.6) == {0: 0, 1: 1}
    # among equally many matches the smaller timing error wins
    assert match_holes([(1.0, 1)], [(0.7, 0), (1.1, 0)], tolerance=0.5) == {0: 1}


def test_anode_without_matched_holes_is_unassessed_not_mixed(tmp_path):
    truth = tmp_path / "truth.csv"
    truth.write_text(TRUTH)
    replay = tmp_path / "replay"
    os.makedirs(replay)
    holes = [(1, 1.0), (2, 2.0), (3, 9.0), (4, 9.5)]   # holes 3-4: nothing true near them
    write(replay / "holes.csv", [{"hole_seq": s, "video_time_s": t} for s, t in holes],
          ["hole_seq", "video_time_s"])
    write(replay / "anodes.csv", [{"anode_seq": 1, "hole_start": 1, "hole_end": 2},
                                  {"anode_seq": 2, "hole_start": 3, "hole_end": 4}],
          ["anode_seq", "hole_start", "hole_end"])
    r = evaluate(str(replay), str(truth), tolerance=0.5)
    assert (r["anodes"]["pure"], r["anodes"]["mixed"], r["anodes"]["unassessed"]) == (1, 0, 1)
    assert r["anodes"]["details"][1]["pure"] is None
