from datetime import datetime, timedelta, timezone

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from counting import Detection
from visibility import VisibilityMonitor

T0 = datetime(2026, 10, 30, 10, 0, tzinfo=timezone.utc)


def image(contrast, h=720, w=1280, seed=0):
    """
    Checkerboard of 64 px blocks whose grey-level standard deviation is
    `contrast`. Coarse structure (like a real scene) survives the monitor's
    downscaling; per-pixel noise would be averaged away.
    """
    yy, xx = np.mgrid[0:h, 0:w]
    sign = np.where(((yy // 64) + (xx // 64)) % 2 == 0, 1.0, -1.0)
    img = np.clip(128 + sign * contrast, 0, 255).astype(np.uint8)
    return np.dstack([img, img, img])


HOLE = [Detection(1, 900, 500, 1000, 540, 0.93)]
WEAK = [Detection(1, 900, 500, 1000, 540, 0.30)]


def run(monitor, steps):
    """steps: list of (seconds, contrast, detections). Returns intervals and states."""
    intervals, states = [], []
    for t, c, dets in steps:
        out = monitor.update(image(c, seed=int(t * 10)), dets, t, T0 + timedelta(seconds=t))
        states.append(monitor.state)
        if out:
            intervals.append(out)
    return intervals, states


def monitor():
    return VisibilityMonitor(line_y=520, baseline=40.0, min_duration_s=2.0)


def at(t0, t1, contrast, dets, step=0.1):
    return [(round(t0 + i * step, 2), contrast, dets) for i in range(int(round((t1 - t0) / step)))]


def test_clear_view_and_empty_belt_are_ok():
    intervals, states = run(monitor(), at(0, 5, 40, HOLE) + at(5, 10, 40, []))
    assert intervals == [] and set(states) == {"OK"}


def test_reduced_contrast_with_confident_holes_is_not_a_gap():
    intervals, states = run(monitor(), at(0, 10, 10, HOLE))
    assert intervals == [] and states[-1] == "REDUCED"


def test_blind_period_becomes_one_low_visibility_gap():
    steps = at(0, 3, 40, HOLE) + at(3, 9, 4, WEAK) + at(9, 13, 40, HOLE)
    intervals, states = run(monitor(), steps)
    assert "LOW" in states and states[-1] == "OK"
    assert len(intervals) == 1
    gap = intervals[0]
    assert (gap.start_utc - T0).total_seconds() == pytest.approx(3.0)
    assert (gap.end_utc - T0).total_seconds() == pytest.approx(9.0)
    assert gap.min_ratio < 0.2


def test_short_blip_is_ignored():
    intervals, states = run(monitor(), at(0, 3, 40, HOLE) + at(3, 4.5, 4, []) + at(4.5, 8, 40, HOLE))
    assert intervals == [] and "LOW" not in states


def test_open_period_is_closed_at_stop():
    m = monitor()
    run(m, at(0, 5, 4, []))
    gap = m.close(T0 + timedelta(seconds=5))
    assert gap is not None and (gap.start_utc - T0).total_seconds() == pytest.approx(0.0)


def test_baseline_is_learned_but_not_from_blind_periods():
    m = VisibilityMonitor(line_y=520, sample_every_s=0.5, min_samples=10, min_duration_s=1.0)
    run(m, at(0, 10, 40, HOLE))
    learned = m.baseline
    assert learned == pytest.approx(40, rel=0.1)
    run(m, at(10, 60, 4, []))                       # long fume: must not drag the baseline down
    assert m.state == "LOW" and m.baseline == learned


def test_non_image_frames_are_skipped():
    m = monitor()
    assert m.update(object(), HOLE, 0.0, T0) is None and m.contrast is None
