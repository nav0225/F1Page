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
    frame_id: int  # frame in which the line was crossed
    cx: float
    cy: float
    conf: Optional[float]  # 'frame' evidence: confidence at crossing; 'track': best confidence of the track


@dataclass(frozen=True)
class RejectedCrossing:
    """A track that crossed the line but never had enough confident detections to be counted."""
    track_id: int
    frame_id: int
    cx: float
    cy: float
    max_conf: Optional[float]
    strong_hits: int
    reason: str = "LOW_CONFIDENCE"


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


@dataclass
class _TrackState:
    last_y: float
    last_seen: int
    max_conf: float = 0.0
    strong_hits: int = 0
    counted: bool = False
    pending: Optional[tuple] = None  # (frame_id, cx, cy) where the line was crossed


class LineCrossingCounter:
    """
    Counts each track at most once when its centre crosses the line.

    Two evidence modes:

    'frame' (legacy behaviour): detections below min_conf are ignored
        entirely, so a crossing only counts if confident detections exist on
        both sides of the line. A hole whose confidence dips while crossing
        (fumes, partial occlusion) is missed.

    'track': every tracked detection moves the track, and a crossing counts
        once the track has min_hits detections at or above min_conf at any
        point of its life. Crossings that never reach that evidence are
        reported as RejectedCrossing (see drain_rejected) instead of silently
        disappearing. Crossings outside x_range do not count. Per-track state
        is released when the track ends. Holes are numbered in the order they
        crossed the line: a qualified crossing is held while a track that
        crossed earlier is still pending (at most stale_frames frames).

    A track is forgotten after stale_frames frames without a detection. Keep
    stale_frames above the tracker's own buffer (ByteTrack: 30) so a track id
    is never revived after the counter forgot it.
    """

    def __init__(self, line_y: float, direction: str, min_conf: float = 0.80,
                 stale_frames: int = 60, evidence: str = "frame", min_hits: int = 2,
                 x_range: Optional[tuple] = None):
        if direction not in ("up", "down"):
            raise ValueError("direction must be 'up' or 'down'")
        if evidence not in ("frame", "track"):
            raise ValueError("evidence must be 'frame' or 'track'")
        self.line_y = line_y
        self.direction = direction
        self.min_conf = min_conf
        self.stale_frames = stale_frames
        self.evidence = evidence
        self.min_hits = min_hits
        self.x_range = x_range
        self.hole_count = 0
        self.outside_roi = 0
        self.rejected_count = 0
        # 'frame' mode state
        self._last_y = {}      # track_id -> last centre y
        self._last_seen = {}   # track_id -> last frame_id
        self._counted = set()  # track_ids already counted
        # 'track' mode state
        self._tracks = {}      # track_id -> _TrackState
        self._rejected = []
        self._ready = []       # qualified crossings not yet numbered: (frame, track_id, cx, cy, conf)

    def accepts(self, det: Detection) -> bool:
        """Whether the detection is used for counting at all."""
        if self.evidence == "track":
            return True
        return det.conf is not None and det.conf >= self.min_conf

    def is_strong(self, det: Detection) -> bool:
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
        self._tracks.clear()

    def drain_rejected(self) -> list:
        rejected, self._rejected = self._rejected, []
        return rejected

    def update(self, frame_id: int, detections: Iterable[Detection]) -> list:
        if self.evidence == "track":
            return self._update_track(frame_id, detections)
        return self._update_frame(frame_id, detections)

    def _update_frame(self, frame_id, detections):
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
            # NOTE: keep tid in _counted to avoid recount if it reappears.
        return crossings

    def _update_track(self, frame_id, detections):
        for det in detections:
            cy = det.cy
            st = self._tracks.get(det.track_id)
            if st is None:
                st = self._tracks[det.track_id] = _TrackState(last_y=cy, last_seen=frame_id)
            prev_y = st.last_y
            st.last_y = cy
            st.last_seen = frame_id
            if det.conf is not None:
                st.max_conf = max(st.max_conf, det.conf)
                if det.conf >= self.min_conf:
                    st.strong_hits += 1

            if not st.counted and st.pending is None and crossed(prev_y, cy, self.line_y, self.direction):
                if self.x_range is not None and not (self.x_range[0] <= det.cx <= self.x_range[1]):
                    st.counted = True  # crossed outside the counting zone; never count this track
                    self.outside_roi += 1
                else:
                    st.pending = (frame_id, det.cx, cy)

            if st.pending is not None and not st.counted and st.strong_hits >= self.min_hits:
                crossing_frame, cx, crossing_y = st.pending
                st.counted = True
                self._ready.append((crossing_frame, det.track_id, cx, crossing_y, st.max_conf))

        for tid in [t for t, st in self._tracks.items() if frame_id - st.last_seen > self.stale_frames]:
            st = self._tracks.pop(tid)
            if st.pending is not None and not st.counted:
                self.rejected_count += 1
                self._rejected.append(RejectedCrossing(tid, st.pending[0], st.pending[1], st.pending[2],
                                                       st.max_conf or None, st.strong_hits))
        return self._release()

    def _release(self):
        """
        Number qualified crossings in the order they crossed the line. Evidence
        can complete in a different order (a track that crossed first may reach
        min_hits only after the next hole has qualified); numbering by crossing
        keeps hole numbers, and the anode grouping built on them, in belt order.
        """
        if not self._ready:
            return []
        waiting = [st.pending[0] for st in self._tracks.values() if st.pending is not None and not st.counted]
        limit = min(waiting) if waiting else None
        self._ready.sort()
        crossings = []
        while self._ready and (limit is None or self._ready[0][0] <= limit):
            frame_id, tid, cx, cy, conf = self._ready.pop(0)
            self.hole_count += 1
            crossings.append(HoleCrossing(self.hole_count, tid, frame_id, cx, cy, conf))
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
