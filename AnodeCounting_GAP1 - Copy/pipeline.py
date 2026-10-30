"""
Per-frame processing shared by the live service and the offline replay.

Keeping this in one place guarantees that a replay of recorded footage runs
exactly the same counting logic as production.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from counting import LineCrossingCounter, ModuloGrouper
from grouping import SpacingClassifier, StructuralGrouper


@dataclass
class FrameResult:
    frame_id: int
    detections: list
    holes: list
    anodes: list
    timings_ms: dict = field(default_factory=dict)
    rejected: list = field(default_factory=list)       # crossings without enough confident detections
    shadow_anodes: list = field(default_factory=list)  # anodes from the comparison grouper (not counted)


def build_counter(settings) -> LineCrossingCounter:
    """The line-crossing counter configured from Settings (same for service and replay)."""
    return LineCrossingCounter(
        settings.line_y, settings.count_direction, settings.count_min_conf, settings.track_stale_frames,
        evidence=settings.count_evidence, min_hits=settings.count_min_hits,
        x_range=(settings.line_x1, settings.line_x2),
    )


def build_groupers(settings):
    """
    (counted grouper, shadow grouper). ANODE_GROUPING selects which one
    produces the counted anodes; the other runs alongside for comparison.
    """
    modulo = ModuloGrouper(settings.holes_per_anode)
    structural = StructuralGrouper(
        settings.line_y, settings.count_direction, settings.holes_per_anode, settings.anode_gap_slots,
        SpacingClassifier(settings.hole_spacing_px, settings.hole_spacing_2_px),
        lane_half_width=settings.lane_half_width_px,
    )
    if settings.anode_grouping == "structural":
        return structural, modulo
    return modulo, structural


class FrameProcessor:
    def __init__(self, detector, counter: LineCrossingCounter, grouper, shadow=None):
        self.detector = detector
        self.counter = counter
        self.grouper = grouper
        self.shadow = shadow

    def process(self, frame_id: int, frame) -> FrameResult:
        t0 = time.perf_counter()
        detections = self.detector(frame)
        t1 = time.perf_counter()

        holes = self.counter.update(frame_id, detections)
        rejected = self.counter.drain_rejected()
        anodes, shadow_anodes = [], []
        self.grouper.observe(frame_id, detections)
        if self.shadow is not None:
            self.shadow.observe(frame_id, detections)
        for hole in holes:
            anodes += self.grouper.add(hole)
            if self.shadow is not None:
                shadow_anodes += self.shadow.add(hole)
        t2 = time.perf_counter()

        return FrameResult(
            frame_id, detections, holes, anodes,
            timings_ms={"detect": (t1 - t0) * 1000.0, "count": (t2 - t1) * 1000.0},
            rejected=rejected, shadow_anodes=shadow_anodes,
        )
