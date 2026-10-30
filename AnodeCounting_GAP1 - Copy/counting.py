"""
Counting logic for the anode handover counter.

This module has no OpenCV, torch or SQL dependencies so that the exact same
code runs in the live service, in offline replays and in unit tests.

Flow per processed frame:
    tracker detections -> LineCrossingCounter -> HoleCrossing events
    HoleCrossing events -> grouper              -> AnodeGroup events
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


# ----------------------------
# Data passed between stages
# ----------------------------
@dataclass(frozen=True)
class Detection:
    """One tracked box from the detector for a single frame."""
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    conf: Optional[float]

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0


@dataclass(frozen=True)
class HoleCrossing:
    """A hole whose track crossed the counting line."""
    seq: int  # running hole number assigned by the counter
    track_id: int
    frame_id: int
    cx: float
    cy: float
    conf: Optional[float]


@dataclass(frozen=True)
class AnodeGroup:
    """Holes that were grouped into one anode."""
    seq: int  # running anode number assigned by the grouper
    holes: tuple

    @property
    def hole_start(self) -> int:
        return self.holes[0].seq

    @property
    def hole_end(self) -> int:
        return self.holes[-1].seq


# ----------------------------
# Line crossing
# ----------------------------
def crossed(prev_y: float, curr_y: float, line_y: float, direction: str) -> bool:
    """True when a point moved across line_y in the counting direction."""
    if direction == "down":
        return prev_y < line_y <= curr_y
    return prev_y > line_y >= curr_y


class LineCrossingCounter:
    """
    Counts each track at most once when its centre crosses the line.

    Detections below min_conf are ignored entirely (they neither move a track
    nor count). A track is forgotten after stale_frames frames without an
    accepted detection, but its id stays in the counted set so it can never
    be counted twice.
    """

    def __init__(self, line_y: float, direction: str, min_conf: float = 0.80,
                 stale_frames: int = 60):
        if direction not in ("up", "down"):
            raise ValueError("direction must be 'up' or 'down'")
        self.line_y = line_y
        self.direction = direction
        self.min_conf = min_conf
        self.stale_frames = stale_frames
        self.hole_count = 0
        self._last_y = {}      # track_id -> last centre y
        self._last_seen = {}   # track_id -> last frame_id
        self._counted = set()  # track_ids already counted

    def accepts(self, det: Detection) -> bool:
        return det.conf is not None and det.conf >= self.min_conf

    def reset_tracks(self) -> None:
        """
        Forget all per-track state. Call whenever the tracker is re-created:
        a new tracker may reuse ids that were already counted, which would
        otherwise suppress genuine new holes. Totals are kept.
        """
        self._last_y.clear()
        self._last_seen.clear()
        self._counted.clear()

    def update(self, frame_id: int, detections: Iterable[Detection]) -> list:
        crossings = []
        for det in detections:
            if not self.accepts(det):
                continue
            tid = det.track_id
            cy = det.cy
            prev_y = self._last_y.get(tid, cy)
            self._last_y[tid] = cy
            self._last_seen[tid] = frame_id

            if tid not in self._counted and crossed(prev_y, cy, self.line_y, self.direction):
                self.hole_count += 1
                self._counted.add(tid)
                crossings.append(HoleCrossing(self.hole_count, tid, frame_id, det.cx, cy, det.conf))

        stale = [tid for tid, last in self._last_seen.items() if frame_id - last > self.stale_frames]
        for tid in stale:
            self._last_seen.pop(tid, None)
            self._last_y.pop(tid, None)
        return crossings


# ----------------------------
# Daily totals
# ----------------------------
class DailyTally:
    """
    Hole and anode totals per plant-local date.

    Counts are keyed by the date of each event, so they roll over at midnight
    without a reset, and can be restored at startup from stored events.
    Hole numbers (HoleStartNo/HoleEndNo) are sequence numbers within the day.
    """

    def __init__(self, keep_days: int = 7):
        self.keep_days = keep_days
        self._days = {}  # date -> {"holes", "anodes", "last_event_time"}

    def _day(self, day):
        if day not in self._days:
            self._days[day] = {"holes": 0, "anodes": 0, "last_event_time": None}
            for old in sorted(self._days)[:-self.keep_days]:
                del self._days[old]
        return self._days[day]

    def restore(self, day, holes: int, anodes: int, last_event_time=None) -> None:
        self._days[day] = {"holes": holes, "anodes": anodes, "last_event_time": last_event_time}

    def add_hole(self, day, when) -> int:
        d = self._day(day)
        d["holes"] += 1
        d["last_event_time"] = when
        return d["holes"]

    def add_anode(self, day, when) -> int:
        d = self._day(day)
        d["anodes"] += 1
        d["last_event_time"] = when
        return d["anodes"]

    def get(self, day) -> dict:
        d = self._days.get(day)
        return dict(d) if d else {"holes": 0, "anodes": 0, "last_event_time": None}


# ----------------------------
# Hole -> anode grouping
# ----------------------------
class ModuloGrouper:
    """
    Legacy grouping: every holes_per_anode-th counted hole closes one anode.

    This has no notion of which holes physically belong together, so a single
    missed or extra hole shifts every later group.
    """

    def __init__(self, holes_per_anode: int = 4):
        self.holes_per_anode = holes_per_anode
        self.anode_count = 0
        self._pending = []

    def add(self, hole: HoleCrossing) -> Optional[AnodeGroup]:
        self._pending.append(hole)
        if len(self._pending) < self.holes_per_anode:
            return None
        self.anode_count += 1
        group = AnodeGroup(self.anode_count, tuple(self._pending))
        self._pending = []
        return group
