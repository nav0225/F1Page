"""
SQL Server access for the anode counter.

Only the EventWriter thread uses this client, so a single connection is kept
open and re-created after any failure. Event inserts are idempotent: each row
carries the event's EventUid and is skipped if that id already exists.
"""
from dataclasses import dataclass, field
from datetime import datetime, date

from persistence import PermanentEventError

# SQL Server error numbers for duplicate keys (unique index / primary key)
_DUPLICATE_KEY_ERRORS = ("2601", "2627")


@dataclass
class SqlConfig:
    driver: str
    server: str
    database: str
    user: str
    password: str = field(repr=False)
    trusted: bool = False
    login_timeout_s: int = 5
    query_timeout_s: int = 15


def _as_datetime(value):
    if value is None or isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


def _as_date(value):
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


class SqlClient:
    def __init__(self, cfg: SqlConfig):
        self.cfg = cfg
        self._connection = None
        self._schema_checked = False

    def _conn_str(self):
        if self.cfg.trusted:
            return (
                f"DRIVER={{{self.cfg.driver}}};"
                f"SERVER={self.cfg.server};"
                f"DATABASE={self.cfg.database};"
                "Trusted_Connection=yes;"
                "TrustServerCertificate=yes;"
            )
        return (
            f"DRIVER={{{self.cfg.driver}}};"
            f"SERVER={self.cfg.server};"
            f"DATABASE={self.cfg.database};"
            f"UID={self.cfg.user};"
            f"PWD={self.cfg.password};"
            "TrustServerCertificate=yes;"
        )

    def _conn(self):
        if self._connection is None:
            import pyodbc

            conn = pyodbc.connect(self._conn_str(), autocommit=True, timeout=self.cfg.login_timeout_s)
            conn.timeout = self.cfg.query_timeout_s
            self._connection = conn
            if not self._schema_checked:
                missing = self.missing_schema_objects()
                if missing:
                    self.close()
                    raise RuntimeError("SQL schema is out of date, missing: " + ", ".join(missing) +
                                       ". Apply the scripts in sql_migrations/ in order.")
                self._schema_checked = True
        return self._connection

    def close(self):
        if self._connection is not None:
            try:
                self._connection.close()
            except Exception:
                pass
            self._connection = None

    def _execute(self, sql, *params):
        cur = self._conn().cursor()
        try:
            cur.execute(sql, *params)
        finally:
            cur.close()

    def missing_schema_objects(self):
        cur = self._connection.cursor()
        try:
            row = cur.execute(
                "SELECT COL_LENGTH('dbo.HoleCrossingEvents', 'EventUid'), "
                "OBJECT_ID('dbo.AnodeEvents'), COL_LENGTH('dbo.AnodeEvents', 'EventUid'), "
                "OBJECT_ID('dbo.spUpsertAnodeDaily')").fetchone()
        finally:
            cur.close()
        names = ["dbo.HoleCrossingEvents.EventUid", "dbo.AnodeEvents",
                 "dbo.AnodeEvents.EventUid", "dbo.spUpsertAnodeDaily"]
        return [name for name, value in zip(names, row) if value is None]

    # ----------------------------
    # Writes
    # ----------------------------
    def upsert_daily(self, count_date, camera_name, hole_count, anode_count, last_event_time):
        self._execute(
            "EXEC dbo.spUpsertAnodeDaily ?, ?, ?, ?, ?",
            _as_date(count_date), camera_name, int(hole_count), int(anode_count), _as_datetime(last_event_time)
        )

    def insert_hole_event(self, e: dict):
        self._execute(
            """
            INSERT INTO dbo.HoleCrossingEvents(EventUid, EventTime, CameraName, TrackId, Cx, Cy, Confidence, FrameId)
            SELECT ?, ?, ?, ?, ?, ?, ?, ?
            WHERE NOT EXISTS (SELECT 1 FROM dbo.HoleCrossingEvents WHERE EventUid = ?)
            """,
            e["uid"], _as_datetime(e["event_time"]), e["camera"], int(e["track_id"]),
            float(e["cx"]), float(e["cy"]),
            None if e.get("conf") is None else float(e["conf"]),
            None if e.get("frame_id") is None else int(e["frame_id"]),
            e["uid"],
        )

    def insert_anode_event(self, e: dict):
        self._execute(
            """
            INSERT INTO dbo.AnodeEvents(EventUid, AnodeTime, CameraName, HoleStartNo, HoleEndNo)
            SELECT ?, ?, ?, ?, ?
            WHERE NOT EXISTS (SELECT 1 FROM dbo.AnodeEvents WHERE EventUid = ?)
            """,
            e["uid"], _as_datetime(e["anode_time"]), e["camera"], int(e["hole_start"]), int(e["hole_end"]),
            e["uid"],
        )


class SqlEventSink:
    """Adapter used by EventWriter: routes outbox events to SqlClient and classifies errors."""

    def __init__(self, client: SqlClient):
        self.client = client
        self._handlers = {
            "hole": client.insert_hole_event,
            "anode": client.insert_anode_event,
        }

    def write(self, kind: str, payload: dict):
        handler = self._handlers.get(kind)
        if handler is None:
            raise PermanentEventError(f"unknown event kind {kind!r}")
        try:
            handler(payload)
        except (KeyError, TypeError, ValueError) as e:
            raise PermanentEventError(f"malformed {kind} event: {e}") from e
        except Exception as e:
            if _is_duplicate_key(e):
                return  # already written by an earlier attempt
            if _is_data_error(e):
                raise PermanentEventError(str(e)) from e
            self.client.close()  # force a fresh connection on the next attempt
            raise


def _is_duplicate_key(exc) -> bool:
    text = " ".join(str(a) for a in getattr(exc, "args", ()))
    return type(exc).__name__ == "IntegrityError" and any(f"({n})" in text for n in _DUPLICATE_KEY_ERRORS)


def _is_data_error(exc) -> bool:
    # pyodbc.DataError: value out of range, conversion failures, etc.
    # IntegrityError other than duplicate key: NOT NULL / CHECK violations.
    return type(exc).__name__ in ("DataError", "IntegrityError")
