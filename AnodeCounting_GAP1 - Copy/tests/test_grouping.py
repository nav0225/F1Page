import random

import pytest

from counting import LineCrossingCounter, ModuloGrouper
from grouping import SpacingClassifier, StructuralGrouper
from tests.sim import simulate

LINE_Y = 520


def run(frames, start_frame=0, repeat_frame=None, repeat=0):
    """Counter + structural grouper (+ legacy grouper) over simulated tracker output."""
    counter = LineCrossingCounter(LINE_Y, "up", 0.80, 30, evidence="track", min_hits=2)
    grouper = StructuralGrouper(LINE_Y, "up", 4, 1, SpacingClassifier(d1=60, d2=120))
    legacy = ModuloGrouper(4)
    records, legacy_groups = [], []
    seq = list(enumerate(frames))[start_frame:]
    if repeat_frame is not None:  # belt stop: the same picture for `repeat` frames
        i = next(k for k, (f, _) in enumerate(seq) if f == repeat_frame)
        seq = seq[:i] + [seq[i]] * repeat + seq[i:]
    for frame_id, (_, dets) in enumerate(seq, start=1):
        grouper.observe(frame_id, dets)
        for h in counter.update(frame_id, dets):
            records += grouper.add(h)
            legacy_groups += legacy.add(h)
    records += grouper.flush()
    return records, legacy_groups


def physical(record_or_group, truth):
    return sorted({truth[h.track_id][0] for h in record_or_group.holes})


def test_clean_train_gives_one_confirmed_record_per_anode():
    frames, truth = simulate(n_anodes=5, speed=6.0, return_tracks=True)
    records, _ = run(frames)
    assert [r.status for r in records] == ["CONFIRMED"] * 5
    assert [physical(r, truth) for r in records] == [[0], [1], [2], [3], [4]]
    assert all(r.slots == (0, 1, 2, 3) for r in records)


def test_start_mid_anode_resynchronises_unlike_modulo_grouping():
    """Video1.mp4 situation: counting starts after the first hole has crossed."""
    frames, truth = simulate(n_anodes=4, speed=6.0, return_tracks=True)
    records, legacy = run(frames, start_frame=35)   # hole (0,0) crossed at frame 30
    assert [physical(r, truth) for r in records] == [[0], [1], [2], [3]]
    assert records[0].status == "PARTIAL" and records[0].slots == (1, 2, 3)
    assert [r.status for r in records[1:]] == ["CONFIRMED"] * 3
    # legacy: every group mixes two physical anodes
    assert all(len(physical(g, truth)) == 2 for g in legacy)


@pytest.mark.parametrize("missing,slots", [((2, 1), (0, 2, 3)), ((2, 0), (1, 2, 3)), ((2, 3), (0, 1, 2))])
def test_one_missed_hole_affects_only_its_own_anode(missing, slots):
    frames, truth = simulate(n_anodes=5, speed=6.0, missing_holes={missing}, return_tracks=True)
    records, legacy = run(frames)
    assert [physical(r, truth) for r in records] == [[0], [1], [2], [3], [4]]
    assert records[2].status == "PARTIAL" and records[2].slots == slots
    assert [r.status for i, r in enumerate(records) if i != 2] == ["CONFIRMED"] * 4
    assert len(legacy) == 4                          # legacy: 19 holes -> 4 "anodes"
    assert sum(len(physical(g, truth)) > 1 for g in legacy) >= 2


def test_belt_stop_mid_anode_does_not_split_it():
    frames, truth = simulate(n_anodes=3, speed=6.0, return_tracks=True)
    records, _ = run(frames, repeat_frame=95, repeat=2000)   # stop while anode 1 is crossing
    assert [physical(r, truth) for r in records] == [[0], [1], [2]]
    assert [r.status for r in records] == ["CONFIRMED"] * 3


@pytest.mark.parametrize("seed", range(20))
def test_flickering_detections_never_merge_anodes(seed):
    rng = random.Random(seed)
    n = rng.randint(3, 8)
    frames, truth = simulate(n_anodes=n, speed=rng.uniform(4.0, 9.0), miss_prob=rng.uniform(0.0, 0.25),
                             jitter=rng.uniform(0.0, 2.0), seed=seed, return_tracks=True)
    records, _ = run(frames)
    assert all(len(physical(r, truth)) == 1 for r in records)   # every record is one physical anode
    counted_anodes = {truth[h.track_id][0] for r in records for h in r.holes}
    assert len(records) == len(counted_anodes)                  # no anode split into two records


def test_projective_spacing_model_matches_video1_geometry():
    c = SpacingClassifier(d1=65, d2=110)
    assert [round(c.pitches(d)) for d in (65, 110, 142, 167)] == [1, 2, 3, 4]
    assert c.pitches(142) == pytest.approx(3.0, abs=0.1)        # measured 142 px for 3 pitches


def test_spacing_learned_from_crossings():
    c = SpacingClassifier()
    assert not c.calibrated
    for _ in range(6):
        for d in (64, 66, 65, 110):        # 3 holes 1 pitch ahead, 1 hole after the gap
            c.observe(d)
    assert c.d1 == pytest.approx(65, abs=1.5) and c.d2 == pytest.approx(110, abs=1.5)


def test_uncalibrated_spacing_holds_crossings_instead_of_guessing():
    """Cold start (no HOLE_SPACING_PX configured): spacing is learned from the first crossings."""
    frames, truth = simulate(n_anodes=5, speed=6.0, return_tracks=True)
    counter = LineCrossingCounter(LINE_Y, "up", 0.80, 30, evidence="track", min_hits=2)
    grouper = StructuralGrouper(LINE_Y, "up", 4, 1, SpacingClassifier())   # nothing configured
    records = []
    for frame_id, dets in enumerate(frames, start=1):
        grouper.observe(frame_id, dets)
        for h in counter.update(frame_id, dets):
            records += grouper.add(h)
    records += grouper.flush()
    assert grouper.classifier.d1 == pytest.approx(60, abs=2)
    assert [physical(r, truth) for r in records] == [[0], [1], [2], [3], [4]]
    assert [r.status for r in records] == ["CONFIRMED"] * 5
