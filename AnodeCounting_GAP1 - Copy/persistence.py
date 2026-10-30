"""
Durable event delivery from the counting loop to SQL Server.

The counting loop never talks to SQL Server directly. It appends each event
to a local SQLite outbox (a few milliseconds, survives process crashes) and a
background EventWriter delivers the outbox to SQL Server, retrying until it
succeeds. Every event carries a unique id, so a retry after an ambiguous
failure cannot create a duplicate row.

    counting loop --append--> Outbox (SQLite) --EventWriter--> SQL Server
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

log = logging.getLogger("anode_counter.persistence")


def new_uid() -> str:
    return str(uuid.uuid4())


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class PermanentEventError(Exception):
    """The event itself can never be written (bad data). Retrying will not help."""


@dataclass(frozen=True)
class OutboxEvent:
    uid: str
    kind: str
    payload: dict
    attempts: int


# ----------------------------
# Outbox
# ----------------------------
class Outbox:
    """
    Append-only local store of events with a 'sent' marker.

    Events stay in the file after delivery (until pruned) so they can be
    inspected or re-sent if the central database is ever restored from backup.
    """

    def __init__(self, path: str, max_attempts_permanent: int = 5):
        self.path = path
        self.max_attempts_permanent = max_attempts_permanent
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS outbox (
                uid TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_utc TEXT NOT NULL,
                sent_utc TEXT,
                dead INTEGER NOT NULL DEFAULT 0,
                attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                local_date TEXT
            )""")
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(outbox)")}
        if "local_date" not in columns:  # outbox file created by an earlier version
            self._conn.execute("ALTER TABLE outbox ADD COLUMN local_date TEXT")
        self._conn.execute("CREATE INDEX IF NOT EXISTS ix_outbox_pending ON outbox(sent_utc, dead)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS ix_outbox_local_date ON outbox(local_date, kind)")

    def append(self, kind: str, payload: dict, uid: Optional[str] = None,
               local_date: Optional[str] = None) -> str:
        """local_date: plant-local date of the event (YYYY-MM-DD), used to restore daily totals."""
        uid = uid or payload.get("uid") or new_uid()
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO outbox(uid, kind, payload, created_utc, local_date) VALUES (?, ?, ?, ?, ?)",
                (uid, kind, json.dumps(payload, default=str), utc_now_iso(), local_date))
        return uid

    def count_by_date(self, local_date: str, unsent_only: bool = False) -> dict:
        """Number of events per kind for one plant-local date."""
        sql = "SELECT kind, COUNT(*) FROM outbox WHERE local_date = ?"
        if unsent_only:
            sql += " AND sent_utc IS NULL"
        with self._lock:
            rows = self._conn.execute(sql + " GROUP BY kind", (local_date,)).fetchall()
        return dict(rows)

    def pending(self, limit: int = 100) -> list:
        with self._lock:
            rows = self._conn.execute(
                "SELECT uid, kind, payload, attempts FROM outbox "
                "WHERE sent_utc IS NULL AND dead = 0 ORDER BY rowid LIMIT ?", (limit,)).fetchall()
        return [OutboxEvent(uid, kind, json.loads(payload), attempts) for uid, kind, payload, attempts in rows]

    def mark_sent(self, uid: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE outbox SET sent_utc = ?, last_error = NULL WHERE uid = ?",
                               (utc_now_iso(), uid))

    def mark_failed(self, uid: str, error: str, permanent: bool = False) -> bool:
        """Record a failed attempt. Returns True if the event was moved to dead letters."""
        with self._lock:
            self._conn.execute("UPDATE outbox SET attempts = attempts + 1, last_error = ? WHERE uid = ?",
                               (error[:1000], uid))
            if permanent:
                attempts = self._conn.execute("SELECT attempts FROM outbox WHERE uid = ?", (uid,)).fetchone()[0]
                if attempts >= self.max_attempts_permanent:
                    self._conn.execute("UPDATE outbox SET dead = 1 WHERE uid = ?", (uid,))
                    return True
        return False

    def backlog(self) -> int:
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM outbox WHERE sent_utc IS NULL AND dead = 0").fetchone()[0]

    def dead_count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM outbox WHERE dead = 1").fetchone()[0]

    def prune_sent(self, older_than_days: int) -> int:
        cutoff = datetime.now(timezone.utc).timestamp() - older_than_days * 86400
        cutoff_iso = datetime.fromtimestamp(cutoff, timezone.utc).isoformat(timespec="milliseconds")
        with self._lock:
            cur = self._conn.execute("DELETE FROM outbox WHERE sent_utc IS NOT NULL AND sent_utc < ?",
                                     (cutoff_iso,))
            return cur.rowcount

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ----------------------------
# Writer
# ----------------------------
class EventWriter(threading.Thread):
    """
    Delivers outbox events to a sink (SQL Server) in order.

    Transient failures (connection lost, timeout, missing schema) block the
    queue and are retried with exponential backoff, so nothing is skipped.
    Permanent data errors are retried a few times and then set aside as dead
    letters, so one bad row cannot stall all later events; dead letters are
    reported by /health.
    """

    def __init__(self, outbox: Outbox, sink, batch_size: int = 50, idle_wait_s: float = 0.5,
                 min_backoff_s: float = 1.0, max_backoff_s: float = 60.0,
                 periodic: Optional[Callable[[], None]] = None, periodic_every_s: float = 5.0,
                 prune_days: int = 30):
        super().__init__(name="event-writer", daemon=True)
        self.outbox = outbox
        self.sink = sink
        self.batch_size = batch_size
        self.idle_wait_s = idle_wait_s
        self.min_backoff_s = min_backoff_s
        self.max_backoff_s = max_backoff_s
        self.periodic = periodic
        self.periodic_every_s = periodic_every_s
        self.prune_days = prune_days

        self._wake = threading.Event()
        self._stop = threading.Event()
        self._backoff = 0.0
        self._last_periodic = 0.0
        self._last_prune = 0.0

        self.written_total = 0
        self.consecutive_failures = 0
        self.last_success_utc = None
        self.last_error = None
        self.last_write_ms = None

    # called from the counting thread after appending events
    def notify(self) -> None:
        self._wake.set()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._wake.set()
        if self.is_alive():
            self.join(timeout)

    def stats(self) -> dict:
        return {
            "backlog": self.outbox.backlog(),
            "dead_letters": self.outbox.dead_count(),
            "written_total": self.written_total,
            "consecutive_failures": self.consecutive_failures,
            "last_success_utc": self.last_success_utc,
            "last_error": self.last_error,
            "last_write_ms": self.last_write_ms,
        }

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                delivered_any = self.drain_once()
                self._run_periodic()
            except Exception as e:  # never let the writer thread die
                log.exception("Event writer loop error")
                self._record_failure(f"writer loop: {e}")
                delivered_any = False

            if self._backoff > 0:
                self._stop.wait(self._backoff)
            elif not delivered_any:
                self._wake.wait(self.idle_wait_s)
                self._wake.clear()

    def drain_once(self) -> bool:
        """Try to deliver one batch. Returns True if at least one event was delivered."""
        batch = self.outbox.pending(self.batch_size)
        delivered = False
        for event in batch:
            if self._stop.is_set():
                break
            t0 = time.perf_counter()
            try:
                self.sink.write(event.kind, event.payload)
            except PermanentEventError as e:
                dead = self.outbox.mark_failed(event.uid, str(e), permanent=True)
                if dead:
                    log.error("Event %s (%s) moved to dead letters: %s", event.uid, event.kind, e)
                    continue  # skip it and keep delivering later events
                self._record_failure(str(e))
                return delivered
            except Exception as e:
                self.outbox.mark_failed(event.uid, str(e))
                self._record_failure(str(e))
                return delivered
            self.outbox.mark_sent(event.uid)
            self.last_write_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            self._record_success()
            delivered = True
        return delivered

    def _run_periodic(self) -> None:
        now = time.monotonic()
        if self.periodic is not None and self._backoff == 0 and now - self._last_periodic >= self.periodic_every_s:
            self._last_periodic = now
            try:
                self.periodic()
            except Exception as e:
                log.warning("Periodic SQL task failed: %s", e)
                self._record_failure(f"periodic: {e}")
        if now - self._last_prune >= 3600:
            self._last_prune = now
            removed = self.outbox.prune_sent(self.prune_days)
            if removed:
                log.info("Pruned %d delivered events older than %d days from outbox", removed, self.prune_days)

    def _record_success(self) -> None:
        self.written_total += 1
        self.consecutive_failures = 0
        self.last_success_utc = utc_now_iso()
        self._backoff = 0.0

    def _record_failure(self, error: str) -> None:
        self.consecutive_failures += 1
        self.last_error = error[:500]
        self._backoff = min(self.max_backoff_s,
                            max(self.min_backoff_s, self._backoff * 2 if self._backoff else self.min_backoff_s))
        if self.consecutive_failures in (1, 10) or self.consecutive_failures % 100 == 0:
            log.warning("SQL delivery failing (%d consecutive): %s; retrying in %.0fs",
                        self.consecutive_failures, error, self._backoff)
