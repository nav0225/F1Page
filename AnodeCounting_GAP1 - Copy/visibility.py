"""
Visibility monitoring at the counting line.

Why: under a heavy fume veil the detector does not degrade gradually - it
stops detecting holes altogether. Processing continues normally, so without
this monitor such a period looks exactly like an empty belt and the missing
anodes leave no trace.

How: two cheap signals per frame.
- Contrast: standard deviation of the grey levels in a band around the
  counting line (downscaled crop, well under a millisecond), relative to a
  rolling "clear view" baseline (90th percentile over VISIBILITY_WINDOW_S).
- Detector activity: whether any detection in the counting zone reaches the
  counting confidence.

States:
  OK       contrast normal (an empty belt is OK: low detections, normal contrast)
  REDUCED  contrast low but holes are still detected confidently - reported
           in /health only, counting is still working
  LOW      contrast low and no confident detection for min_duration_s - the
           belt cannot be observed; the interval is recorded as a
           LOW_VISIBILITY counting gap

Measured on Video1.mp4 with a synthetic grey veil: contrast ratio 0.96 clear,
0.36 / 0.22 / 0.13 at veil 0.6 / 0.75 / 0.85 (all holes still counted), 0.09
at 0.90 (no hole detected). Edge-based sharpness was rejected: it collapses at
veil 0.6 while counting still works.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class LowVisibilityInterval:
    start_utc: datetime
    end_utc: datetime
    min_ratio: float


class VisibilityMonitor:
    def __init__(self, line_y: int, band_px: int = 120, x_range: tuple = (0, 100000), min_conf: float = 0.80,
                 low_ratio: float = 0.5, min_duration_s: float = 2.0, window_s: float = 7200.0,
                 sample_every_s: float = 1.0, min_samples: int = 60, baseline: Optional[float] = None):
        self.line_y = line_y
        self.band_px = band_px
        self.x_range = x_range
        self.min_conf = min_conf
        self.low_ratio = low_ratio
        self.min_duration_s = min_duration_s
        self.sample_every_s = sample_every_s
        self.min_samples = min_samples
        self.fixed_baseline = baseline
        self._samples = deque(maxlen=max(10, int(window_s / sample_every_s)))
        self._last_sample = None
        self._baseline = baseline

        self.state = "OK"
        self.contrast = None
        self.ratio = None
        self._candidate_since = None   # (mono, utc) when the LOW condition started
        self._clear_since = None       # mono when it stopped (while LOW)
        self._clear_utc = None
        self._low_start = None
        self._min_ratio = None

    # ----------------------------
    # Measurement
    # ----------------------------
    def measure(self, image) -> Optional[float]:
        import cv2

        if getattr(image, "ndim", 0) < 2:
            return None  # not an image (e.g. test doubles): nothing to measure
        h, w = image.shape[:2]
        y0, y1 = max(0, self.line_y - self.band_px), min(h, self.line_y + self.band_px)
        x0, x1 = max(0, self.x_range[0]), min(w, self.x_range[1])
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            return 0.0
        if crop.ndim == 3:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if crop.shape[1] > 320:
            crop = cv2.resize(crop, (320, max(1, crop.shape[0] * 320 // crop.shape[1])), interpolation=cv2.INTER_AREA)
        return float(crop.std())

    @property
    def baseline(self) -> Optional[float]:
        return self._baseline

    def _update_baseline(self, contrast, t_mono):
        if self.fixed_baseline is not None or self.state == "LOW":
            return
        if self._last_sample is not None and t_mono - self._last_sample < self.sample_every_s:
            return
        self._last_sample = t_mono
        self._samples.append(contrast)
        if len(self._samples) >= self.min_samples:
            ordered = sorted(self._samples)
            self._baseline = ordered[int(0.9 * (len(ordered) - 1))]

    # ----------------------------
    # Per-frame update
    # ----------------------------
    def update(self, image, detections, t_mono: float, t_utc: datetime) -> Optional[LowVisibilityInterval]:
        """Returns a LowVisibilityInterval when a LOW period ends."""
        self.contrast = self.measure(image)
        if self.contrast is None:
            return None
        self._update_baseline(self.contrast, t_mono)
        if not self._baseline:
            self.ratio = None
            return None
        self.ratio = self.contrast / self._baseline
        low_contrast = self.ratio < self.low_ratio
        confident = any(d.conf is not None and d.conf >= self.min_conf
                        and self.x_range[0] <= d.cx <= self.x_range[1] for d in detections)
        blind = low_contrast and not confident

        if self.state != "LOW":
            self.state = "REDUCED" if low_contrast else "OK"
            if blind:
                if self._candidate_since is None:
                    self._candidate_since = (t_mono, t_utc)
                    self._min_ratio = self.ratio
                self._min_ratio = min(self._min_ratio, self.ratio)
                if t_mono - self._candidate_since[0] >= self.min_duration_s:
                    self.state = "LOW"
                    self._low_start = self._candidate_since[1]
                    self._clear_since = None
            else:
                self._candidate_since = None
            return None

        # in LOW: wait for a sustained recovery
        if blind:
            self._min_ratio = min(self._min_ratio, self.ratio)
            self._clear_since = None
            return None
        if self._clear_since is None:
            self._clear_since, self._clear_utc = t_mono, t_utc
        if t_mono - self._clear_since >= self.min_duration_s:
            return self._end(self._clear_utc)
        return None

    def close(self, t_utc: datetime) -> Optional[LowVisibilityInterval]:
        """End an open LOW period (service stopping, end of a replay)."""
        if self.state != "LOW":
            return None
        return self._end(t_utc)

    def _end(self, end_utc) -> LowVisibilityInterval:
        interval = LowVisibilityInterval(self._low_start, end_utc, round(self._min_ratio, 3))
        self.state = "OK"
        self._candidate_since = self._clear_since = self._clear_utc = self._low_start = None
        return interval

    def status(self) -> dict:
        return {
            "state": self.state,
            "contrast": None if self.contrast is None else round(self.contrast, 2),
            "baseline": None if self._baseline is None else round(self._baseline, 2),
            "ratio": None if self.ratio is None else round(self.ratio, 3),
            "low_since_utc": self._low_start.isoformat() if self.state == "LOW" and self._low_start else None,
        }
