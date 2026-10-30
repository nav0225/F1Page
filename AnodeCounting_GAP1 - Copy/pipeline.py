"""
Per-frame processing shared by the live service and the offline replay.

Keeping this in one place guarantees that a replay of recorded footage runs
exactly the same counting logic as production.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from counting import LineCrossingCounter, ModuloGrouper


@dataclass
class FrameResult:
    frame_id: int
    detections: list
    holes: list
    anodes: list
    timings_ms: dict = field(default_factory=dict)
    rejected: list = field(default_factory=list)  # crossings without enough confident detections


def build_counter(settings) -> LineCrossingCounter:
    """The line-crossing counter configured from Settings (same for service and replay)."""
    return LineCrossingCounter(
        settings.line_y, settings.count_direction, settings.count_min_conf, settings.track_stale_frames,
        evidence=settings.count_evidence, min_hits=settings.count_min_hits,
        x_range=(settings.line_x1, settings.line_x2),
    )


class FrameProcessor:
    def __init__(self, detector, counter: LineCrossingCounter, grouper: ModuloGrouper):
        self.detector = detector
        self.counter = counter
        self.grouper = grouper

    def process(self, frame_id: int, frame) -> FrameResult:
        t0 = time.perf_counter()
        detections = self.detector(frame)
        t1 = time.perf_counter()

        holes = self.counter.update(frame_id, detections)
        rejected = self.counter.drain_rejected()
        anodes = []
        for hole in holes:
            group = self.grouper.add(hole)
            if group is not None:
                anodes.append(group)
        t2 = time.perf_counter()

        return FrameResult(
            frame_id, detections, holes, anodes,
            timings_ms={"detect": (t1 - t0) * 1000.0, "count": (t2 - t1) * 1000.0},
            rejected=rejected,
        )
