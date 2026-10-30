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
            for g in grouper.add(h):
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
            groups += grouper.add(h)
    # 11 holes seen -> 2 anodes reported, and the second "anode" mixes holes
    # from physical anodes 2 and 3.
    assert counter.hole_count == 11
    assert len(groups) == 2


# ----------------------------------------------------------------------------
# Track evidence mode (D5, D13, D20)
# ----------------------------------------------------------------------------
def track_counter(**kw):
    kw.setdefault("x_range", (0, 1920))
    return LineCrossingCounter(LINE_Y, "up", min_conf=0.80, stale_frames=kw.pop("stale_frames", 10),
                               evidence="track", min_hits=kw.pop("min_hits", 2), **kw)


def feed(counter, frames):
    holes, rejected = [], []
    for i, dets in enumerate(frames, start=1):
        holes += counter.update(i, dets)
        rejected += counter.drain_rejected()
    for j in range(len(frames) + 1, len(frames) + counter.stale_frames + 3):  # let tracks expire
        holes += counter.update(j, [])
        rejected += counter.drain_rejected()
    return holes, rejected


def test_confidence_dip_at_line_is_counted_in_track_mode_but_missed_in_legacy():
    # confident far below the line, weak (fumes) while crossing and afterwards
    seq = [det(1, 600, 0.92), det(1, 580, 0.90), det(1, 560, 0.55), det(1, 535, 0.45),
           det(1, 515, 0.40), det(1, 495, 0.42)]
    frames = [[d] for d in seq]
    legacy = LineCrossingCounter(LINE_Y, "up", 0.80, 10)
    assert sum(len(legacy.update(i, f)) for i, f in enumerate(frames, start=1)) == 0
    holes, rejected = feed(track_counter(), frames)
    assert len(holes) == 1 and rejected == []
    assert holes[0].frame_id == 5            # frame in which the line was crossed
    assert holes[0].conf == pytest.approx(0.92)


def test_weak_track_is_rejected_not_counted():
    frames = [[det(1, y, 0.5)] for y in (560, 540, 515, 495)]
    holes, rejected = feed(track_counter(), frames)
    assert holes == []
    assert len(rejected) == 1
    r = rejected[0]
    assert (r.track_id, r.frame_id, r.strong_hits, r.reason) == (1, 3, 0, "LOW_CONFIDENCE")
    assert r.max_conf == pytest.approx(0.5)


def test_crossing_confirmed_later_keeps_crossing_frame():
    # weak while crossing, then becomes confident above the line (e.g. fume clears)
    seq = [det(1, 540, 0.5), det(1, 515, 0.5), det(1, 495, 0.9), det(1, 475, 0.9)]
    c = track_counter()
    out = [c.update(i, [d]) for i, d in enumerate(seq, start=1)]
    assert [len(o) for o in out] == [0, 0, 0, 1]   # counted once 2 confident hits exist
    assert out[3][0].frame_id == 2


def test_crossing_outside_roi_never_counts():
    c = track_counter(x_range=(400, 1000))
    frames = [[det(1, y, 0.9, cx=1500)] for y in (540, 530, 515, 500)]
    holes, rejected = feed(c, frames)
    assert holes == [] and rejected == [] and c.outside_roi == 1


def test_track_state_released_when_tracks_end():
    c = track_counter(stale_frames=5)
    frames = simulate(n_anodes=30, speed=8.0)
    feed(c, frames)
    assert c.hole_count == 120
    assert len(c._tracks) == 0


def test_track_mode_matches_legacy_on_clean_footage():
    frames = simulate(n_anodes=7, speed=5.0)
    holes, _ = feed(track_counter(), frames)
    assert len(holes) == 28


@pytest.mark.parametrize("seed", range(30))
def test_track_mode_recovers_weak_crossings_without_overcounting(seed):
    rng = random.Random(1000 + seed)
    n = rng.randint(2, 10)
    frames = simulate(n_anodes=n, speed=rng.uniform(3.0, 12.0), miss_prob=rng.uniform(0.0, 0.2),
                      low_conf_prob=rng.uniform(0.2, 0.6), jitter=rng.uniform(0.0, 3.0), seed=seed,
                      weak_after_line=rng.uniform(0.1, 0.5))
    legacy_holes, _ = legacy_reference(frames, LINE_Y, "up", 30)
    holes, rejected = feed(track_counter(stale_frames=30), frames)
    assert len(holes) >= len(legacy_holes)          # never worse than the legacy gate
    assert len(holes) <= 4 * n                      # never more holes than physically passed
    assert len(set(h.track_id for h in holes)) == len(holes)
