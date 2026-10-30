import csv
import json
import os
from types import SimpleNamespace

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

import replay
from tests.sim import simulate


class ScriptedDetector:
    """Returns pre-computed detections, one list per call (i.e. per frame)."""

    def __init__(self, frames):
        self.frames = frames
        self.i = 0

    def __call__(self, frame):
        dets = self.frames[self.i] if self.i < len(self.frames) else []
        self.i += 1
        return dets


def write_video(path, n_frames, size=(64, 48), fps=20.0):
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(n_frames):
        w.write(np.full((size[1], size[0], 3), i % 255, dtype=np.uint8))
    w.release()


def test_replay_writes_holes_anodes_and_summary(tmp_path):
    frames = simulate(n_anodes=3, speed=6.0)
    video = str(tmp_path / "clip.mp4")
    write_video(video, len(frames))
    out = str(tmp_path / "out")
    args = SimpleNamespace(video=video, out=out, env=None, model=None, device="cpu",
                           line_y=520, direction="up", max_frames=0, save_video=False)

    summary = replay.run(args, detector_factory=lambda s: ScriptedDetector(frames))

    assert summary["holes_counted"] == 12
    assert summary["anodes_counted"] == 3
    with open(os.path.join(out, "anodes.csv")) as f:
        anodes = list(csv.DictReader(f))
    assert [(a["hole_start"], a["hole_end"]) for a in anodes] == [("1", "4"), ("5", "8"), ("9", "12")]
    with open(os.path.join(out, "holes.csv")) as f:
        assert len(list(csv.DictReader(f))) == 12
    with open(os.path.join(out, "summary.json")) as f:
        saved = json.load(f)
    assert "rtsp_url" not in saved["settings"]
    assert saved["timing_ms"]["total"]["p95"] >= 0
