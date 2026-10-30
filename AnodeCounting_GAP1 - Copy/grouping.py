"""
Grouping hole crossings into anodes by belt geometry.

Why: the legacy grouper turns every 4th counted hole into an anode. Starting
mid-anode, or one missed or extra hole, shifts every later group, so anode
records stop corresponding to physical anodes. On Video1.mp4 all 3 emitted
anodes mixed holes of two physical anodes.

How: on the belt each anode carries `holes_per_anode` holes at a fixed
pitch, and consecutive anodes are separated by `gap_slots` empty pitches
(GAP1: the last hole of one anode is about two pitches from the first hole
of the next, i.e. one empty slot). When a hole is on the counting line, the
holes ahead of it sit at whole-pitch offsets, and image distances map to
pitch offsets the same way at every crossing (the crossing hole is always at
the same image position), independent of belt speed or stops.

The slot of each crossing hole within its anode is decided afresh at every
crossing, so an error cannot propagate to later anodes:
1. neighbour: the previously counted hole is visible exactly one pitch
   ahead, so this hole is the next slot of the same anode (anodes are always
   separated by a gap slot);
2. pattern: otherwise, the slot whose expected arrangement of holes and gaps
   ahead best matches the holes detected ahead;
3. step: if the pattern is ambiguous, the step from the previous hole; the
   anode is then flagged phase_assumed (status UNCERTAIN).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median
from typing import Optional


@dataclass(frozen=True)
class AnodeRecord:
    seq: int
    holes: tuple                 # HoleCrossing objects seen for this anode
    slots: tuple                 # slot (0..holes_per_anode-1) of each seen hole
    holes_per_anode: int
    phase_assumed: bool          # at least one slot could not be read from the belt pattern
    close_reason: str            # COMPLETE | NEXT_ANODE | FLUSH

    @property
    def holes_seen(self) -> int:
        return len(self.holes)

    @property
    def status(self) -> str:
        """
        CONFIRMED  every hole seen, slots read from the belt pattern
        PARTIAL    slots read from the pattern, but some holes were not counted
        UNCERTAIN  some slots had to be inferred (pattern not readable)
        """
        if self.phase_assumed:
            return "UNCERTAIN"
        return "CONFIRMED" if self.holes_seen == self.holes_per_anode else "PARTIAL"

    @property
    def hole_start(self) -> int:
        return self.holes[0].seq

    @property
    def hole_end(self) -> int:
        return self.holes[-1].seq


class SpacingClassifier:
    """
    Converts the image distance from the counting line to a hole ahead into a
    number of pitches along the belt.

    For points on a flat belt, the image distance from the counting line of a
    point D pitches ahead follows d(D) = A*D / (B + D) (projective mapping of
    a line; B -> infinity for a camera looking straight down). The 1-pitch and
    2-pitch distances d1 and d2 fix A and B. On Video1.mp4 d1 ~ 65 px and
    d2 ~ 110 px predict d(3) = 143 px; measured 142 px.

    d1/d2 can be configured from a calibration replay; otherwise they are
    learned online from the nearest hole ahead at each crossing.
    """

    DEFAULT_D2_OVER_D1 = 1.7   # Video1.mp4 at the counting line

    def __init__(self, d1: Optional[float] = None, d2: Optional[float] = None, min_samples: int = 8):
        self.fixed = d1 is not None
        self.d1 = self.d2 = None
        self.min_samples = min_samples
        self._samples = []
        if d1 is not None:
            self._set(d1, d2 if d2 is not None else d1 * self.DEFAULT_D2_OVER_D1)

    @property
    def calibrated(self) -> bool:
        return self.d1 is not None

    def _set(self, d1, d2):
        r = d2 / d1
        if not 1.05 < r <= 2.0:
            r, d2 = self.DEFAULT_D2_OVER_D1, d1 * self.DEFAULT_D2_OVER_D1
        self.d1, self.d2 = d1, d2
        if r > 1.97:   # practically no perspective: distances are linear in pitches
            self._a, self._b = math.inf, math.inf
        else:
            self._b = 2.0 * (r - 1.0) / (2.0 - r)
            self._a = d1 * (self._b + 1.0)

    def observe(self, nearest_ahead: float) -> None:
        """Learn d1/d2 from the distance to the nearest hole ahead at each crossing."""
        if self.fixed:
            return
        self._samples.append(nearest_ahead)
        if len(self._samples) > 400:
            del self._samples[:200]
        if len(self._samples) >= self.min_samples:
            self._learn()

    def calibrate_now(self) -> bool:
        """Calibrate from whatever samples exist (end of a replay). Returns calibrated."""
        if not self.calibrated and self._samples:
            self._learn()
        return self.calibrated

    def _learn(self) -> None:
        ordered = sorted(self._samples)
        d1 = median(ordered[: max(1, int(len(ordered) * 0.6))])
        twos = [d for d in ordered if 1.35 * d1 < d < 2.05 * d1]   # 2.0 = no perspective
        self._set(d1, median(twos) if len(twos) >= 2 else d1 * self.DEFAULT_D2_OVER_D1)

    def pitches(self, distance: float) -> float:
        """Belt position in pitches of a point `distance` px ahead of the line (negative = behind)."""
        if math.isinf(self._a):
            return distance / self.d1
        if distance >= self._a:
            return math.inf
        return self._b * distance / (self._a - distance)


class StructuralGrouper:
    """
    Call observe(frame_id, detections) for every processed frame, then
    add(hole) for each hole counted in that frame. Returns AnodeRecords as
    anodes close.

    An anode closes when its last slot is filled from the pattern (COMPLETE)
    or when a hole of the next anode crosses (NEXT_ANODE). A partially seen
    anode therefore closes when the next hole arrives, never on a timer, so a
    belt stop cannot split an anode. Slots read from the pattern override a
    previous hole's inferred slot instead of splitting the anode.
    """

    MEMORY_FRAMES = 5   # use holes missing from the current frame if seen this recently
    method = "STRUCTURAL"

    def __init__(self, line_y: float, direction: str, holes_per_anode: int = 4, gap_slots: int = 1,
                 classifier: Optional[SpacingClassifier] = None, lane_half_width: float = 80.0):
        self.line_y = line_y
        self.direction = direction
        self.n = holes_per_anode
        self.gap = gap_slots
        self.period = holes_per_anode + gap_slots
        self.classifier = classifier or SpacingClassifier()
        self.lane_half_width = lane_half_width
        self.anode_count = 0
        self.last_decision = None   # diagnostics for the latest crossing
        self._frame = []
        self._recent = {}           # track_id -> (frame_id, cx, cy)
        self._holes, self._slots = [], []
        self._prev_slot = None      # slot of the most recent hole (kept across anode closes)
        self._phase_assumed = False
        self._last_inferred = False  # slot of the most recent hole was inferred, not read
        self._prev_track = None      # track id of the most recently counted hole
        self._waiting = []           # crossings held until the spacing is calibrated

    # ----------------------------
    # Inputs
    # ----------------------------
    def observe(self, frame_id: int, detections) -> None:
        self._frame = list(detections)
        for d in self._frame:
            self._recent[d.track_id] = (frame_id, d.cx, d.cy)
        for tid in [t for t, (f, _, _) in self._recent.items() if frame_id - f > self.MEMORY_FRAMES]:
            del self._recent[tid]

    def add(self, hole) -> list:
        ref, ahead = self._positions_ahead(hole)
        if ahead and abs(ref) < 10.0:   # learn spacing only when the hole is right on the line
            self.classifier.observe(ahead[0][1] - ref)
        if not self.classifier.calibrated:
            # distances cannot be read as pitches yet: hold the crossing with its
            # measurements and decide it in order once the spacing is known
            self._waiting.append((hole, ref, ahead))
            return []
        records = []
        for waiting in self._waiting:
            records += self._decide(*waiting)
        self._waiting = []
        return records + self._decide(hole, ref, ahead)

    @property
    def waiting(self) -> int:
        return len(self._waiting)

    def _decide(self, hole, ref, ahead) -> list:
        offsets = sorted({o for o in (self._offset(ref, d) for _, d in ahead) if o is not None})
        prev_offset = next((self._offset(ref, d) for tid, d in ahead if tid == self._prev_track), None)
        slot, how = None, None
        if prev_offset == 1 and self._prev_slot is not None and self._prev_slot < self.n - 1:
            slot, how = self._prev_slot + 1, "neighbour"
        if slot is None:
            slot, how = self._slot_from_pattern(offsets)
        if slot is None:
            slot, how = self._slot_from_step(offsets, prev_offset)
        self.last_decision = {"track_id": hole.track_id, "reference_px": round(ref, 1),
                              "offsets": offsets[:6], "previous_hole_offset": prev_offset,
                              "slot": slot, "method": how}
        self._prev_track = hole.track_id
        return self._place(hole, slot, assumed=how not in ("pattern", "neighbour"))

    def flush(self) -> list:
        records = []
        if self._waiting and self.classifier.calibrate_now():
            for waiting in self._waiting:
                records += self._decide(*waiting)
        elif self._waiting:
            # never calibrated (no hole was ever seen ahead of another): slots unknown
            for hole, _, _ in self._waiting:
                records += self._place(hole, 0, assumed=True)
        self._waiting = []
        records += self._close("FLUSH")
        self._prev_slot = None
        return records

    def reset_tracks(self) -> None:
        """The tracker was re-created: forget track-id based memory, keep the open anode."""
        self._frame, self._recent, self._prev_track = [], {}, None

    # ----------------------------
    # Slot of the crossing hole
    # ----------------------------
    def _ahead(self, y: float) -> float:
        return self.line_y - y if self.direction == "up" else y - self.line_y

    def _positions_ahead(self, hole):
        """
        (reference, [(track_id, distance), ...]): image distances ahead of the
        line for the crossing hole itself (its current position; a crossing can
        be registered a few frames late) and for the holes ahead of it in its lane.
        """
        current_ids = {d.track_id for d in self._frame}
        points = [(d.track_id, d.cx, d.cy) for d in self._frame]
        points += [(tid, cx, cy) for tid, (_, cx, cy) in self._recent.items() if tid not in current_ids]
        ref_y = next((cy for tid, _, cy in points if tid == hole.track_id), hole.cy)
        ref = self._ahead(ref_y)
        out = []
        for tid, cx, cy in points:
            if tid == hole.track_id or abs(cx - hole.cx) > self.lane_half_width:
                continue
            d = self._ahead(cy)
            if d - ref > 5.0:  # ignore boxes overlapping the crossing hole
                out.append((tid, d))
        return ref, sorted(out, key=lambda item: item[1])

    def _offset(self, ref, distance):
        """Whole pitches between the crossing hole (at `ref`) and a hole at `distance`."""
        if not self.classifier.calibrated:
            return None
        p = self.classifier.pitches(distance) - self.classifier.pitches(ref)
        return None if math.isinf(p) else max(1, int(round(p)))

    # Pattern scoring: a detected hole where the pattern has a gap is strong
    # evidence against a candidate slot (holes cannot be in gaps); an expected
    # hole that was not detected is weak evidence (missed detections happen).
    HOLE_IN_GAP_PENALTY = 3
    MISSING_HOLE_PENALTY = 1
    MIN_MARGIN = 2

    def _expects_hole(self, slot, offset):
        """Whether the pattern puts a hole `offset` pitches ahead of a hole in `slot`."""
        return (slot - offset) % self.period < self.n

    def _slot_from_pattern(self, offsets):
        """
        Choose the slot whose expected pattern of holes ahead best matches the
        detected holes ahead (up to the farthest detected one).
        """
        if not offsets:
            return 0, "pattern"                    # front of the train: first hole of an anode
        present = set(offsets)
        run = 0
        while run + 1 in present:
            run += 1
        if run > self.n - 1:
            return None, "inconsistent"            # more holes in a row than an anode has
        if not any(o > run + 1 for o in offsets):
            return run, "pattern"                  # front of the train, partly passed

        horizon = min(max(offsets), 2 * self.period)
        scores = []
        for slot in range(self.n):
            score = 0
            for offset in range(1, horizon + 1):
                hole_expected = self._expects_hole(slot, offset)
                if offset in present and not hole_expected:
                    score += self.HOLE_IN_GAP_PENALTY
                elif hole_expected and offset not in present:
                    score += self.MISSING_HOLE_PENALTY
            scores.append((score, slot))
        scores.sort()
        (best, slot), (second, _) = scores[0], scores[1]
        if second - best < self.MIN_MARGIN:
            return None, "ambiguous"
        return slot, "pattern"

    def _slot_from_step(self, offsets, prev_offset=None):
        """Fallback: one pitch per step from the previous hole."""
        if self._prev_slot is None or not offsets:
            return 0, "assumed"
        step = prev_offset or offsets[0]
        candidate = self._prev_slot + step
        if candidate <= self.n - 1:
            return candidate, "step"
        slot = candidate - self.period
        return (slot, "step") if 0 <= slot <= self.n - 1 else (0, "assumed")

    # ----------------------------
    # Anode assembly
    # ----------------------------
    def _place(self, hole, slot, assumed) -> list:
        closed = []
        if self._holes and slot <= self._slots[-1]:
            earlier = self._slots[-2] if len(self._slots) > 1 else -1
            if self._last_inferred and not assumed and earlier < slot - 1:
                # the previous hole's slot was a guess; the pattern read now says
                # it must be slot-1, so correct it rather than split the anode
                self._slots[-1] = slot - 1
            else:
                closed += self._close("NEXT_ANODE")  # slot did not advance: next anode
        self._holes.append(hole)
        self._slots.append(slot)
        self._phase_assumed = self._phase_assumed or assumed
        self._last_inferred = assumed
        self._prev_slot = slot
        if slot == self.n - 1 and not assumed:
            closed += self._close("COMPLETE")
        return closed

    def _close(self, reason) -> list:
        if not self._holes:
            return []
        self.anode_count += 1
        record = AnodeRecord(self.anode_count, tuple(self._holes), tuple(self._slots), self.n,
                             self._phase_assumed, reason)
        self._holes, self._slots, self._phase_assumed = [], [], False
        return [record]
