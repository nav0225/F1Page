"""
Video capture on its own thread.

The FrameGrabber owns the camera connection: it reads frames continuously,
stamps each one with its capture time, and reconnects with backoff when the
stream fails. The counting loop takes frames from it.

Two hand-over modes:
- drop_stale=True (live camera): frames wait in a small FIFO buffer
  (buffer_frames). Short processing hiccups are absorbed without losing a
  frame; if counting stays slower than the camera, the oldest frames are
  discarded instead of queueing up, so counts can never fall more than
  buffer_frames behind the belt. Discarded frames are counted, and the
  counting loop records a gap whenever the interval between processed
  frames exceeds the configured maximum.
- drop_stale=False (video files): every frame is handed over in order.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional

log = logging.getLogger("anode_counter.capture")


def ffmpeg_capture_options(rtsp_transport: str) -> str:
    """
    Value for the OPENCV_FFMPEG_CAPTURE_OPTIONS environment variable.

    Deliberately does NOT include "fflags;nobuffer": with it, FFmpeg discards
    the packets read while probing the stream at every (re)connect and decoding
    restarts at the next keyframe (measured: the first 158 frames / 5.3 s of
    Video1.mp4 were lost). "flags;low_delay" is omitted because OpenCV passes
    these options to the demuxer only, where it has no effect.
    """
    return f"rtsp_transport;{rtsp_transport}"


def open_rtsp(url: str, open_timeout_ms: int, read_timeout_ms: int):
    import cv2

    open_prop = getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", None)
    read_prop = getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None)
    if open_prop is None or read_prop is None:
        # OpenCV without the timeout properties (older than 4.5.4): FFmpeg defaults apply
        return cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    try:
        return cv2.VideoCapture(url, cv2.CAP_FFMPEG, [open_prop, open_timeout_ms, read_prop, read_timeout_ms])
    except TypeError:  # OpenCV older than 4.5.2: no open parameters
        return cv2.VideoCapture(url, cv2.CAP_FFMPEG)


def open_file(path: str):
    import cv2

    return cv2.VideoCapture(path)


@dataclass(frozen=True)
class CapturedFrame:
    seq: int                # increases across reconnects
    image: Any
    t_capture_mono: float   # time.monotonic() when the frame was read
    t_capture_utc: datetime


class FrameGrabber:
    def __init__(self, open_source: Callable[[], Any], drop_stale: bool = True, buffer_frames: int = 10,
                 reconnect_min_s: float = 1.0, reconnect_max_s: float = 30.0,
                 clock: Optional[Callable[[], datetime]] = None):
        self.open_source = open_source
        self.drop_stale = drop_stale
        self.buffer_frames = max(1, buffer_frames) if drop_stale else 1
        # Capture timestamps are real time only when frames are read as fast as
        # the source delivers them (live camera mode).
        self.realtime = drop_stale
        self.reconnect_min_s = reconnect_min_s
        self.reconnect_max_s = reconnect_max_s
        self.clock = clock or (lambda: datetime.now(timezone.utc))

        self._cond = threading.Condition()
        self._buffer = deque()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._seq = 0

        self.connected = False
        self.frames_read = 0
        self.frames_dropped = 0
        self.reconnects = 0           # successful re-opens after a failure
        self.failures = 0             # failed opens or reads
        self.last_error: Optional[str] = None
        self._read_times = []         # recent read timestamps for capture fps

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="capture", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def get(self, timeout: float = 1.0) -> Optional[CapturedFrame]:
        """Oldest buffered frame, waiting up to timeout seconds. None if nothing arrived."""
        with self._cond:
            if not self._buffer:
                self._cond.wait(timeout)
            frame = self._buffer.popleft() if self._buffer else None
            self._cond.notify_all()
            return frame

    def stats(self) -> dict:
        now = time.monotonic()
        recent = [t for t in self._read_times if now - t <= 5.0]
        fps = (len(recent) - 1) / (recent[-1] - recent[0]) if len(recent) > 1 and recent[-1] > recent[0] else 0.0
        return {
            "connected": self.connected,
            "frames_read": self.frames_read,
            "frames_dropped": self.frames_dropped,
            "buffered": len(self._buffer),
            "reconnects": self.reconnects,
            "failures": self.failures,
            "capture_fps": round(fps, 2),
            "last_error": self.last_error,
        }

    def _run(self) -> None:
        backoff = self.reconnect_min_s
        cap = None
        had_failure = False
        while not self._stop.is_set():
            try:
                if cap is None:
                    cap = self.open_source()
                    if cap is None or (hasattr(cap, "isOpened") and not cap.isOpened()):
                        raise IOError("could not open video source")
                ok, image = cap.read()
                if not ok or image is None:
                    raise IOError("video read failed")
            except Exception as e:
                self.connected = False
                self.failures += 1
                self.last_error = str(e)
                had_failure = True
                if self.failures in (1, 10) or self.failures % 100 == 0:
                    log.warning("Capture failure #%d (%s); retrying in %.0fs", self.failures, e, backoff)
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap = None
                self._stop.wait(backoff)
                backoff = min(self.reconnect_max_s, backoff * 2)
                continue

            if had_failure:
                self.reconnects += 1
                had_failure = False
                log.info("Video source reconnected (#%d)", self.reconnects)
            backoff = self.reconnect_min_s
            self.connected = True
            self._publish(image)

        if cap is not None:
            cap.release()
        self.connected = False

    def _publish(self, image) -> None:
        self._seq += 1
        self.frames_read += 1
        now_mono = time.monotonic()
        self._read_times.append(now_mono)
        if len(self._read_times) > 300:
            del self._read_times[:150]
        frame = CapturedFrame(self._seq, image, now_mono, self.clock())
        with self._cond:
            if self.drop_stale:
                if len(self._buffer) >= self.buffer_frames:
                    self._buffer.popleft()
                    self.frames_dropped += 1
            else:
                while self._buffer and not self._stop.is_set():
                    self._cond.wait(0.1)
            self._buffer.append(frame)
            self._cond.notify_all()
