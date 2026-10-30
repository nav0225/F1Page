"""
The counting service: reads frames, counts holes/anodes and records events.

Runs in one background thread under a supervisor. Any exception (camera,
model, disk) is logged, the pipeline is rebuilt after a short backoff, and
counting resumes with totals intact. The thread never talks to SQL Server;
events go to the local outbox and are delivered by the EventWriter.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone

from counting import DailyTally, LineCrossingCounter, ModuloGrouper
from persistence import new_uid
from pipeline import FrameProcessor

log = logging.getLogger("anode_counter.service")


class PendingEvents(deque):
    """
    Events the local outbox could not store yet (e.g. disk full), oldest first.
    Bounded so that a long outage cannot exhaust memory. An event pushed out
    because the queue is full is counted in `lost` and reported in /health,
    never dropped silently.
    """

    def __init__(self, maxlen: int):
        super().__init__(maxlen=maxlen)
        self.lost = 0

    def append(self, item) -> None:
        if len(self) == self.maxlen:
            self.lost += 1
            if self.lost in (1, 10) or self.lost % 1000 == 0:
                log.critical("Local outbox unwritable with %d events waiting: %d events lost",
                             self.maxlen, self.lost)
        super().append(item)


class CounterService:
    # A pipeline error stays visible in /health until this many frames have
    # been processed successfully after the restart.
    CLEAR_ERROR_AFTER_FRAMES = 600
    RESTART_BACKOFF_S = 1.0       # first wait after a pipeline failure (doubles, max 30 s)
    RECONNECT_WAIT_S = 1.0        # wait before reopening a failed video source

    def __init__(self, settings, outbox, writer, detector_factory, source_factory,
                 encode_preview=None, clock=None, local_tz=None):
        """
        detector_factory() -> callable(frame) -> list[Detection]
        source_factory()   -> object with read() -> (ok, frame) and release()
        encode_preview(frame, result, counter, totals, fps) -> JPEG bytes or None
        clock()            -> timezone-aware UTC datetime (default: system clock)
        local_tz           -> plant time zone (default: the server's local time zone)
        """
        self.settings = settings
        self.outbox = outbox
        self.writer = writer
        self.detector_factory = detector_factory
        self.source_factory = source_factory
        self.encode_preview = encode_preview
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.local_tz = local_tz

        self.counter = LineCrossingCounter(settings.line_y, settings.count_direction,
                                           settings.count_min_conf, settings.track_stale_frames)
        self.grouper = ModuloGrouper(settings.holes_per_anode)
        self.tally = DailyTally()
        self.totals_source = "new"   # how today's totals were initialised at startup
        self._day_seq = {}            # counter hole seq -> hole number within its day

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._unpersisted = PendingEvents(maxlen=100_000)  # events the outbox could not store yet

        self.started_mono = time.monotonic()
        self.latest_jpeg = None
        self.last_frame_mono = None
        self.frames_processed = 0
        self.fps = 0.0
        self.loop_restarts = 0
        self.reconnects = 0
        self.last_error = None
        self.persist_errors = 0

    # ----------------------------
    # Time and daily totals
    # ----------------------------
    def to_local(self, utc_dt):
        """Plant-local wall time (naive), the convention of the existing SQL columns."""
        return utc_dt.astimezone(self.local_tz).replace(tzinfo=None)

    def local_today(self):
        return self.to_local(self.clock()).date()

    def restore_today(self, fetch_sql_counts=None):
        """
        Initialise today's totals before counting starts, so a restart does not
        reset them. True total = rows already in SQL + local events not yet
        delivered. If SQL is unreachable, fall back to the local outbox alone
        (which lacks events recorded before this service version).
        """
        today = self.local_today()
        unsent = self.outbox.count_by_date(today.isoformat(), unsent_only=True)
        if fetch_sql_counts is not None:
            try:
                holes, anodes, last = fetch_sql_counts(today)
                self.tally.restore(today, holes + unsent.get("hole", 0), anodes + unsent.get("anode", 0), last)
                self.totals_source = "sql+outbox"
                log.info("Restored today's totals from SQL: %s", self.tally.get(today))
                return
            except Exception as e:
                log.warning("Could not read today's totals from SQL (%s); using local outbox only", e)
        local = self.outbox.count_by_date(today.isoformat())
        self.tally.restore(today, local.get("hole", 0), local.get("anode", 0))
        self.totals_source = "outbox_only"
        log.info("Restored today's totals from local outbox: %s", self.tally.get(today))

    # ----------------------------
    # Lifecycle
    # ----------------------------
    def start(self):
        self._thread = threading.Thread(target=self._run_supervised, name="counter", daemon=True)
        self._thread.start()

    def stop(self, timeout=10.0):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def _run_supervised(self):
        backoff = self.RESTART_BACKOFF_S
        while not self._stop.is_set():
            try:
                self._run_pipeline()
                return  # only returns when stop was requested
            except Exception as e:
                self.loop_restarts += 1
                self.last_error = f"{type(e).__name__}: {e}"
                log.exception("Counting pipeline failed (restart #%d in %.0fs)", self.loop_restarts, backoff)
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 30.0)

    # ----------------------------
    # Pipeline
    # ----------------------------
    def _run_pipeline(self):
        detector = self.detector_factory()
        self.counter.reset_tracks()  # new tracker instance -> old track ids are meaningless
        processor = FrameProcessor(detector, self.counter, self.grouper)
        cap = self.source_factory()

        t0 = time.time()
        frames = 0
        healthy_frames = 0
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    self.reconnects += 1
                    log.warning("Video read failed; reconnecting (#%d)", self.reconnects)
                    cap.release()
                    self._stop.wait(self.RECONNECT_WAIT_S)
                    cap = self.source_factory()
                    continue

                self.frames_processed += 1
                frames += 1
                dt = time.time() - t0
                if dt >= 1.0:
                    self.fps = frames / dt
                    frames = 0
                    t0 = time.time()

                result = processor.process(self.frames_processed, frame)
                self._record_events(result)

                jpeg = None
                if self.encode_preview is not None:
                    totals = self.tally.get(self.local_today())
                    jpeg = self.encode_preview(frame, result, self.counter, totals, self.fps)
                with self._lock:
                    if jpeg is not None:
                        self.latest_jpeg = jpeg
                    self.last_frame_mono = time.monotonic()

                healthy_frames += 1
                if self.last_error and healthy_frames >= self.CLEAR_ERROR_AFTER_FRAMES:
                    self.last_error = None
        finally:
            cap.release()

    def _record_events(self, result):
        if not result.holes and not self._unpersisted:
            return
        now_utc = self.clock()
        now_local = self.to_local(now_utc)
        day = now_local.date()
        camera = self.settings.camera_name

        with self._lock:
            for hole in result.holes:
                day_seq = self.tally.add_hole(day, now_local)
                self._day_seq[hole.seq] = day_seq
                self._unpersisted.append(("hole", day, {
                    "uid": new_uid(), "event_time": now_local.isoformat(),
                    "event_time_utc": now_utc.replace(tzinfo=None).isoformat(), "camera": camera,
                    "track_id": hole.track_id, "cx": hole.cx, "cy": hole.cy, "conf": hole.conf,
                    "frame_id": hole.frame_id,
                }))
            for group in result.anodes:
                self.tally.add_anode(day, now_local)
                self._unpersisted.append(("anode", day, {
                    "uid": new_uid(), "anode_time": now_local.isoformat(),
                    "anode_time_utc": now_utc.replace(tzinfo=None).isoformat(), "camera": camera,
                    "hole_start": self._day_seq.get(group.hole_start, group.hole_start),
                    "hole_end": self._day_seq.get(group.hole_end, group.hole_end),
                }))
            for seq in [s for s in self._day_seq if s <= self.counter.hole_count - 64]:
                del self._day_seq[seq]

        while self._unpersisted:
            kind, day, payload = self._unpersisted[0]
            try:
                self.outbox.append(kind, payload, local_date=day.isoformat())
            except Exception as e:  # e.g. disk full: keep counting, retry next frame
                self.persist_errors += 1
                if self.persist_errors in (1, 10) or self.persist_errors % 1000 == 0:
                    log.error("Cannot write event to local outbox (%d errors, %d waiting): %s",
                              self.persist_errors, len(self._unpersisted), e)
                break
            self._unpersisted.popleft()
        self.writer.notify()

    # ----------------------------
    # Read-only views for the API
    # ----------------------------
    def daily_snapshot(self):
        today = self.local_today()
        with self._lock:
            totals = self.tally.get(today)
        last = totals["last_event_time"]
        return {
            "date": today.isoformat(),
            "hole_count": totals["holes"],
            "anode_count": totals["anodes"],
            "last_event_time": last.isoformat() if hasattr(last, "isoformat") else last,
        }

    def stats(self):
        snap = self.daily_snapshot()
        return {"camera": self.settings.camera_name, **snap, "fps": float(self.fps),
                "totals_source": self.totals_source}

    def preview_jpeg(self):
        with self._lock:
            return self.latest_jpeg

    def health(self):
        now = time.monotonic()
        with self._lock:
            last_frame = self.last_frame_mono
        frame_age = None if last_frame is None else now - last_frame
        uptime = now - self.started_mono
        writer = self.writer.stats()
        problems, status = [], "ok"

        def down(msg):
            nonlocal status
            status = "down"
            problems.append(msg)

        def degraded(msg):
            nonlocal status
            if status == "ok":
                status = "degraded"
            problems.append(msg)

        if not self.is_running():
            down("counting thread is not running")
        elif frame_age is None:
            if uptime > self.settings.startup_grace_s:
                down(f"no frame processed since start ({uptime:.0f}s)")
        elif frame_age > self.settings.health_max_frame_age_s:
            down(f"no frame processed for {frame_age:.0f}s")
        if not self.writer.is_alive():
            down("SQL writer thread is not running")
        if writer["consecutive_failures"]:
            degraded(f"SQL delivery failing: {writer['last_error']}")
        if writer["dead_letters"]:
            degraded(f"{writer['dead_letters']} events could not be written to SQL (dead letters)")
        if self._unpersisted:
            degraded(f"{len(self._unpersisted)} events not yet stored in local outbox")
        if self._unpersisted.lost:
            degraded(f"{self._unpersisted.lost} events LOST: the local outbox could not be written "
                     f"while {self._unpersisted.maxlen} events were waiting")
        if self.last_error:
            degraded(f"last pipeline error: {self.last_error}")

        return {
            "status": status,
            "problems": problems,
            "camera": self.settings.camera_name,
            "totals_source": self.totals_source,
            "uptime_s": round(uptime, 1),
            "frame_age_s": None if frame_age is None else round(frame_age, 2),
            "frames_processed": self.frames_processed,
            "processing_fps": round(self.fps, 2),
            "pipeline_restarts": self.loop_restarts,
            "reconnects": self.reconnects,
            "outbox_write_errors": self.persist_errors,
            "events_lost": self._unpersisted.lost,
            "sql": writer,
        }
