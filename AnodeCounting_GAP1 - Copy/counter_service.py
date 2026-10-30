"""
The counting service: takes frames from the capture thread, counts
holes/anodes and records events.

Runs in one background thread under a supervisor. Any exception (model,
disk, unexpected data) is logged, the pipeline is rebuilt after a short
backoff, and counting resumes with totals intact. The thread never talks to
SQL Server; events go to the local outbox and are delivered by the
EventWriter.

Every interval in which frames were not processed often enough to count
reliably (camera reconnect, slow processing, pipeline restart, service
down) is recorded as a counting gap, so missed anodes are explainable.
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


def _percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(p / 100.0 * (len(ordered) - 1))))]


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
    RESTART_BACKOFF_S = 1.0          # first wait after a pipeline failure (doubles, max 30 s)
    STATE_SAVE_EVERY_S = 5.0         # how often the last processed frame time is persisted

    def __init__(self, settings, outbox, writer, detector_factory, grabber,
                 encode_preview=None, clock=None, local_tz=None):
        """
        detector_factory() -> callable(frame image) -> list[Detection]
        grabber            -> capture.FrameGrabber (or compatible: start/stop/get/stats)
        encode_preview(image, result, counter, totals, fps) -> JPEG bytes or None
        clock()            -> timezone-aware UTC datetime (default: system clock)
        local_tz           -> plant time zone (default: the server's local time zone)
        """
        self.settings = settings
        self.outbox = outbox
        self.writer = writer
        self.detector_factory = detector_factory
        self.grabber = grabber
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
        self.last_error = None
        self.persist_errors = 0
        self.gaps_recorded = 0
        self.last_gap = None
        self._lags_ms = deque(maxlen=600)   # capture -> processed, recent frames
        self._prev = None                   # last processed frame: (utc, mono, capture failures, restarts)
        self._down_since = None             # last frame time from the previous run (SERVICE_DOWN gap)
        self._last_state_save = 0.0

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
        last = self.outbox.get_state("last_frame_utc")
        if last:
            self._down_since = datetime.fromisoformat(last)
        self.grabber.start()
        self._thread = threading.Thread(target=self._run_supervised, name="counter", daemon=True)
        self._thread.start()

    def stop(self, timeout=10.0):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
        self.grabber.stop()
        self._save_state(force=True)

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

        t0 = time.time()
        frames = 0
        healthy_frames = 0
        while not self._stop.is_set():
            frame = self.grabber.get(timeout=0.5)
            if frame is None:
                continue  # /health reports the growing frame age

            self._check_gap(frame)
            self.frames_processed += 1
            frames += 1
            dt = time.time() - t0
            if dt >= 1.0:
                self.fps = frames / dt
                frames = 0
                t0 = time.time()

            result = processor.process(frame.seq, frame.image)
            lag_ms = (time.monotonic() - frame.t_capture_mono) * 1000.0
            self._lags_ms.append(lag_ms)
            self._record_events(result, frame, lag_ms)

            jpeg = None
            if self.encode_preview is not None:
                totals = self.tally.get(self.local_today())
                jpeg = self.encode_preview(frame.image, result, self.counter, totals, self.fps)
            with self._lock:
                if jpeg is not None:
                    self.latest_jpeg = jpeg
                self.last_frame_mono = time.monotonic()
            self._save_state()

            healthy_frames += 1
            if self.last_error and healthy_frames >= self.CLEAR_ERROR_AFTER_FRAMES:
                self.last_error = None

    # ----------------------------
    # Counting gaps
    # ----------------------------
    def _check_gap(self, frame):
        # Only meaningful when capture timestamps are real time (live camera);
        # a video file is read at whatever pace processing allows.
        if not getattr(self.grabber, "realtime", True):
            return
        failures = self.grabber.stats()["failures"]
        prev = self._prev
        self._prev = (frame.t_capture_utc, frame.t_capture_mono, failures, self.loop_restarts, frame.seq)

        if prev is None:
            # first frame of this run: was the service down for a while?
            if self._down_since is not None:
                gap_ms = (frame.t_capture_utc - self._down_since).total_seconds() * 1000.0
                if gap_ms > self.settings.max_frame_gap_ms:
                    self._record_gap(self._down_since, frame.t_capture_utc, "SERVICE_DOWN",
                                     "counter service was not running")
            return

        prev_utc, prev_mono, prev_failures, prev_restarts, prev_seq = prev
        gap_ms = (frame.t_capture_mono - prev_mono) * 1000.0
        if gap_ms <= self.settings.max_frame_gap_ms:
            return
        dropped = frame.seq - prev_seq - 1
        if failures != prev_failures:
            reason, details = "SOURCE_DISCONNECTED", f"{failures - prev_failures} capture failures"
        elif self.loop_restarts != prev_restarts:
            reason, details = "PIPELINE_RESTART", self.last_error or ""
        elif dropped > 0:
            reason, details = "PROCESSING_GAP", f"counting fell behind; {dropped} frames skipped"
        else:
            reason, details = "SOURCE_STALL", "camera delivered no frames"
        self._record_gap(prev_utc, frame.t_capture_utc, reason, details)

    def _record_gap(self, start_utc, end_utc, reason, details):
        duration_ms = int((end_utc - start_utc).total_seconds() * 1000.0)
        start_local, end_local = self.to_local(start_utc), self.to_local(end_utc)
        self.gaps_recorded += 1
        self.last_gap = {"reason": reason, "start": start_local.isoformat(), "duration_ms": duration_ms}
        log.warning("Counting gap %s: %s for %.1fs (%s)", reason, start_local.isoformat(),
                    duration_ms / 1000.0, details)
        self._unpersisted.append(("gap", start_local.date(), {
            "uid": new_uid(), "camera": self.settings.camera_name,
            "start_time": start_local.isoformat(), "end_time": end_local.isoformat(),
            "start_time_utc": start_utc.replace(tzinfo=None).isoformat(),
            "end_time_utc": end_utc.replace(tzinfo=None).isoformat(),
            "duration_ms": duration_ms, "reason": reason, "details": details,
        }))
        self._flush_unpersisted()

    def _save_state(self, force=False):
        now = time.monotonic()
        if self._prev is None or (not force and now - self._last_state_save < self.STATE_SAVE_EVERY_S):
            return
        self._last_state_save = now
        try:
            self.outbox.set_state("last_frame_utc", self._prev[0].isoformat())
        except Exception as e:
            log.warning("Could not save last frame time: %s", e)

    # ----------------------------
    # Events
    # ----------------------------
    def _record_events(self, result, frame, lag_ms):
        if not result.holes and not self._unpersisted:
            return
        t_utc = frame.t_capture_utc
        t_local = self.to_local(t_utc)
        day = t_local.date()
        camera = self.settings.camera_name
        lag = round(lag_ms, 1)

        with self._lock:
            for hole in result.holes:
                day_seq = self.tally.add_hole(day, t_local)
                self._day_seq[hole.seq] = day_seq
                self._unpersisted.append(("hole", day, {
                    "uid": new_uid(), "event_time": t_local.isoformat(),
                    "event_time_utc": t_utc.replace(tzinfo=None).isoformat(), "camera": camera,
                    "track_id": hole.track_id, "cx": hole.cx, "cy": hole.cy, "conf": hole.conf,
                    "frame_id": hole.frame_id, "lag_ms": lag,
                }))
            for group in result.anodes:
                self.tally.add_anode(day, t_local)
                self._unpersisted.append(("anode", day, {
                    "uid": new_uid(), "anode_time": t_local.isoformat(),
                    "anode_time_utc": t_utc.replace(tzinfo=None).isoformat(), "camera": camera,
                    "hole_start": self._day_seq.get(group.hole_start, group.hole_start),
                    "hole_end": self._day_seq.get(group.hole_end, group.hole_end),
                    "lag_ms": lag,
                }))
            for seq in [s for s in self._day_seq if s <= self.counter.hole_count - 64]:
                del self._day_seq[seq]

        self._flush_unpersisted()

    def _flush_unpersisted(self):
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
        capture = self.grabber.stats()
        lags = list(self._lags_ms)
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
        if not self.grabber.is_alive():
            down("capture thread is not running")
        elif not capture["connected"] and capture["failures"]:
            degraded(f"video source disconnected: {capture['last_error']}")
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
            "lag_ms": {"p50": _percentile(lags, 50), "p95": _percentile(lags, 95), "max": max(lags) if lags else None},
            "pipeline_restarts": self.loop_restarts,
            "gaps_recorded": self.gaps_recorded,
            "last_gap": self.last_gap,
            "outbox_write_errors": self.persist_errors,
            "capture": capture,
            "events_lost": self._unpersisted.lost,
            "sql": writer,
        }
