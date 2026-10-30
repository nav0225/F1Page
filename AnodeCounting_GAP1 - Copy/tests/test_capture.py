import os
import subprocess
import sys
import time

import pytest

from capture import FrameGrabber, ffmpeg_capture_options, open_rtsp

APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class CountingSource:
    """Fake VideoCapture: frame images are increasing integers; can fail after N reads."""

    opened = 0

    def __init__(self, fail_after=None, delay_s=0.0):
        CountingSource.opened += 1
        self.fail_after = fail_after
        self.delay_s = delay_s
        self.n = 0

    def isOpened(self):
        return True

    def read(self):
        if self.delay_s:
            time.sleep(self.delay_s)
        self.n += 1
        if self.fail_after is not None and self.n > self.fail_after:
            return False, None
        return True, self.n

    def release(self):
        pass


def take(grabber, count, timeout=5.0):
    frames, deadline = [], time.time() + timeout
    while len(frames) < count and time.time() < deadline:
        f = grabber.get(timeout=0.2)
        if f is not None:
            frames.append(f)
    return frames


def test_file_mode_delivers_every_frame_in_order():
    g = FrameGrabber(lambda: CountingSource(), drop_stale=False)
    g.start()
    frames = take(g, 50)
    g.stop()
    assert [f.image for f in frames] == list(range(1, 51))
    assert [f.seq for f in frames] == list(range(1, 51))
    assert g.frames_dropped == 0
    assert g.realtime is False


def test_live_mode_bounds_backlog_and_counts_drops():
    g = FrameGrabber(lambda: CountingSource(delay_s=0.002), drop_stale=True, buffer_frames=5)
    g.start()
    seen, buffered = [], []
    for _ in range(10):
        time.sleep(0.05)                 # consumer much slower than the camera
        f = g.get(timeout=1.0)
        seen.append(f.seq)
        buffered.append(g.stats()["buffered"])
    g.stop()
    assert seen == sorted(seen) and len(set(seen)) == len(seen)   # in order, never repeated
    assert g.frames_dropped > 0
    assert seen[-1] - seen[0] > len(seen)                          # intermediate frames skipped
    assert max(buffered) <= 5                                      # backlog never exceeds the buffer
    # every frame read was delivered, dropped, or is still buffered
    assert g.frames_read == len(seen) + g.frames_dropped + g.stats()["buffered"]


def test_live_mode_absorbs_short_hiccup_without_dropping():
    g = FrameGrabber(lambda: CountingSource(delay_s=0.01), drop_stale=True, buffer_frames=10)
    g.start()
    frames = take(g, 5)
    time.sleep(0.05)                     # brief stall: ~5 frames arrive meanwhile
    frames += take(g, 15)
    g.stop()
    assert [f.seq for f in frames] == list(range(1, 21))
    assert g.frames_dropped == 0


def test_read_failure_reconnects_with_backoff():
    CountingSource.opened = 0
    g = FrameGrabber(lambda: CountingSource(fail_after=5), drop_stale=False,
                     reconnect_min_s=0.01, reconnect_max_s=0.02)
    g.start()
    frames = take(g, 12)
    g.stop()
    assert len(frames) == 12
    assert CountingSource.opened >= 3
    assert g.reconnects >= 2 and g.failures >= 2
    assert [f.seq for f in frames] == list(range(1, 13))   # sequence continues across reconnects


def test_open_failure_is_retried_and_reported():
    class Closed:
        def isOpened(self):
            return False

        def release(self):
            pass

    g = FrameGrabber(lambda: Closed(), reconnect_min_s=0.01, reconnect_max_s=0.01)
    g.start()
    time.sleep(0.1)
    g.stop()
    stats = g.stats()
    assert stats["connected"] is False and stats["failures"] >= 3
    assert "could not open" in stats["last_error"]


def test_ffmpeg_options_do_not_drop_frames_at_connect():
    assert ffmpeg_capture_options("tcp") == "rtsp_transport;tcp"
    assert "nobuffer" not in ffmpeg_capture_options("udp")


DECODE = "import cv2,sys; c=cv2.VideoCapture(sys.argv[1]); n=0\nwhile c.read()[0]: n+=1\nprint(n)"


@pytest.mark.skipif(not os.path.exists(os.path.join(APP_DIR, "Video1.mp4")), reason="sample video not present")
def test_service_ffmpeg_options_decode_every_frame_of_sample_video():
    pytest.importorskip("cv2")
    video = os.path.join(APP_DIR, "Video1.mp4")

    def decoded(options):
        env = dict(os.environ, OPENCV_FFMPEG_CAPTURE_OPTIONS=options)
        out = subprocess.run([sys.executable, "-c", DECODE, video], env=env, capture_output=True, text=True, check=True)
        return int(out.stdout.strip())

    assert decoded(ffmpeg_capture_options("tcp")) == 579
    # the option removed by this change loses everything before the 2nd keyframe
    assert decoded("rtsp_transport;tcp|fflags;nobuffer|flags;low_delay") == 579 - 158


@pytest.mark.parametrize("has_timeouts", [True, False])
def test_open_rtsp_uses_timeouts_only_where_opencv_has_them(monkeypatch, has_timeouts):
    import types
    calls = []
    cv2 = types.ModuleType("cv2")
    cv2.CAP_FFMPEG = 1900
    cv2.VideoCapture = lambda *args: calls.append(args) or "capture"
    if has_timeouts:   # OpenCV 4.5.4+
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, cv2.CAP_PROP_READ_TIMEOUT_MSEC = 53, 54
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    assert open_rtsp("rtsp://camera/stream", 10000, 5000) == "capture"
    expected = ("rtsp://camera/stream", 1900, [53, 10000, 54, 5000]) if has_timeouts else ("rtsp://camera/stream", 1900)
    assert calls == [expected]
