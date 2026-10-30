import random

import pytest

from counting import Detection, LineCrossingCounter, ModuloGrouper, crossed
from tests.sim import simulate

LINE_Y = 520


def det(tid, cy, conf=0.9, cx=950.0, h=40.0):
    return Detection(tid, cx - 50, cy - h / 2, cx + 50, cy + h / 2, conf)


# ----------------------------------------------------------------------------
# Reference: the counting loop body exactly as it was in service_app.py at
# commit 77fb086 (before extraction). Used to prove the refactor preserved
# behaviour.
# ----------------------------------------------------------------------------
def legacy_reference(frames, line_y, direction, stale_frames=60):
    def legacy_crossed(prev_y, curr_y, line_y):
        if direction == "down":
            return prev_y < line_y <= curr_y
        else:
            return prev_y > line_y >= curr_y

    track_last_y, track_last_seen, track_counted = {}, {}, set()
    hole_count = anode_count = 0
    holes, anodes = [], []
    for frame_id, dets in enumerate(frames, start=1):
        for d in dets:
            tid = int(d.track_id)
            cx = (d.x1 + d.x2) / 2.0
            cy = (d.y1 + d.y2) / 2.0
            conf = float(d.conf) if d.conf is not None else None
            if conf is None or conf < 0.80:
                continue
            prev_y = track_last_y.get(tid, cy)
            track_last_y[tid] = cy
            track_last_seen[tid] = frame_id
            if tid not in track_counted and legacy_crossed(prev_y, cy, line_y):
                hole_count += 1
                track_counted.add(tid)
                holes.append((hole_count, tid, frame_id))
                if hole_count % 4 == 0:
                    anode_count += 1
                    anodes.append((anode_count, hole_count - 3, hole_count, frame_id))
        stale = [tid for tid, last in track_last_seen.items() if frame_id - last > stale_frames]
        for tid in stale:
            track_last_seen.pop(tid, None)
            track_last_y.pop(tid, None)
    return holes, anodes


def run_new(frames, line_y, direction, stale_frames=60):
    counter = LineCrossingCounter(line_y, direction, 0.80, stale_frames)
    grouper = ModuloGrouper(4)
    holes, anodes = [], []
    for frame_id, dets in enumerate(frames, start=1):
        for h in counter.update(frame_id, dets):
            holes.append((h.seq, h.track_id, h.frame_id))
            g = grouper.add(h)
            if g is not None:
                anodes.append((g.seq, g.hole_start, g.hole_end, h.frame_id))
    return holes, anodes


@pytest.mark.parametrize("seed", range(40))
def test_refactor_matches_legacy_loop_on_randomized_tracks(seed):
    rng = random.Random(seed)
    direction = rng.choice(["up", "down"])
    line_y = LINE_Y if direction == "up" else 1080 - LINE_Y
    frames = simulate(
        n_anodes=rng.randint(1, 12),
        speed=rng.uniform(2.0, 30.0),
        direction=direction,
        miss_prob=rng.uniform(0.0, 0.3),
        low_conf_prob=rng.uniform(0.0, 0.4),
        id_switch_prob=rng.uniform(0.0, 0.05),
        jitter=rng.uniform(0.0, 8.0),
        extra_tracks=rng.randint(0, 3),
        seed=seed,
    )
    stale = rng.choice([5, 30, 60])
    assert run_new(frames, line_y, direction, stale) == legacy_reference(frames, line_y, direction, stale)


def test_clean_train_counts_every_hole_and_anode():
    frames = simulate(n_anodes=7, speed=5.0)
    holes, anodes = run_new(frames, LINE_Y, "up")
    assert len(holes) == 28
    assert len(anodes) == 7


def test_crossed_direction():
    assert crossed(530, 520, 520, "up")
    assert not crossed(520, 510, 520, "up")  # already on/after the line
    assert crossed(510, 520, 520, "down")
    assert not crossed(530, 520, 520, "down")


def test_track_counted_once_despite_jitter_around_line():
    c = LineCrossingCounter(LINE_Y, "up")
    ys = [560, 530, 515, 525, 510, 530, 505]
    total = sum(len(c.update(i, [det(1, y)])) for i, y in enumerate(ys, start=1))
    assert total == 1


def test_track_first_seen_past_line_is_not_counted():
    c = LineCrossingCounter(LINE_Y, "up")
    assert c.update(1, [det(1, 500)]) == []
    assert c.update(2, [det(1, 480)]) == []


def test_low_confidence_detection_neither_moves_nor_counts_track():
    c = LineCrossingCounter(LINE_Y, "up", min_conf=0.80)
    assert c.update(1, [det(1, 540, conf=0.9)]) == []
    assert c.update(2, [det(1, 510, conf=0.5)]) == []  # ignored
    # next accepted detection is compared with the last accepted one (540)
    assert len(c.update(3, [det(1, 500, conf=0.9)])) == 1


def test_stale_track_is_forgotten_but_never_recounted():
    c = LineCrossingCounter(LINE_Y, "up", stale_frames=5)
    c.update(1, [det(7, 530)])
    assert len(c.update(2, [det(7, 510)])) == 1
    for f in range(3, 20):
        c.update(f, [])
    c.update(20, [det(7, 530)])
    assert c.update(21, [det(7, 500)]) == []


def test_modulo_grouping_shifts_after_one_missed_hole():
    """Documents the legacy defect (D4): one missed hole misaligns every later anode."""
    frames = simulate(n_anodes=3, speed=5.0, missing_holes={(0, 1)})
    counter = LineCrossingCounter(LINE_Y, "up")
    grouper = ModuloGrouper(4)
    groups = []
    for i, dets in enumerate(frames, start=1):
        for h in counter.update(i, dets):
            g = grouper.add(h)
            if g:
                groups.append(g)
    # 11 holes seen -> 2 anodes reported, and the second "anode" mixes holes
    # from physical anodes 2 and 3.
    assert counter.hole_count == 11
    assert len(groups) == 2
