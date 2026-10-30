import sys
import types

import pytest

from persistence import PermanentEventError


class FakeCursor:
    def __init__(self, conn):
        self.conn = conn
        self._row = None

    def execute(self, sql, *params):
        self.conn.executed.append((" ".join(sql.split()), params))
        if self.conn.raise_on_execute is not None:
            exc = self.conn.raise_on_execute
            self.conn.raise_on_execute = None
            raise exc
        if "COL_LENGTH" in sql:
            self._row = self.conn.schema_row
        elif "COUNT(*)" in sql:
            self._row = self.conn.day_counts
        return self

    def fetchone(self):
        return self._row

    def close(self):
        pass


class FakeConnection:
    def __init__(self, schema_row):
        self.schema_row = schema_row
        self.executed = []
        self.raise_on_execute = None
        self.day_counts = (0, 0, None)
        self.closed = False
        self.timeout = None

    def cursor(self):
        return FakeCursor(self)

    def close(self):
        self.closed = True


def make_fake_pyodbc(schema_row=(16, 1, 16, 8, 8, 1, 1, 4, 4, 1, 32, 1)):
    mod = types.ModuleType("pyodbc")
    mod.connections = []

    class Error(Exception):
        pass

    class IntegrityError(Error):
        pass

    class DataError(Error):
        pass

    class OperationalError(Error):
        pass

    mod.Error, mod.IntegrityError, mod.DataError, mod.OperationalError = Error, IntegrityError, DataError, OperationalError

    def connect(conn_str, autocommit=False, timeout=0):
        conn = FakeConnection(schema_row)
        conn.autocommit, conn.login_timeout = autocommit, timeout
        mod.connections.append(conn)
        return conn

    mod.connect = connect
    return mod


@pytest.fixture
def fake_pyodbc(monkeypatch):
    mod = make_fake_pyodbc()
    monkeypatch.setitem(sys.modules, "pyodbc", mod)
    return mod


def make_client():
    from sql_client import SqlClient, SqlConfig
    return SqlClient(SqlConfig("ODBC Driver 17 for SQL Server", "srv", "db", "u", "p"))


HOLE = {"uid": "6f1c1b0e-1111-4c4c-8888-000000000001", "event_time": "2026-10-30T10:15:00.123456",
        "event_time_utc": "2026-10-30T04:45:00.123456",
        "camera": "Camera_01", "track_id": 7, "cx": 1.5, "cy": 2.5, "conf": 0.91, "frame_id": 42}
ANODE = {"uid": "6f1c1b0e-1111-4c4c-8888-000000000002", "anode_time": "2026-10-30T23:59:59.900000",
         "anode_time_utc": "2026-10-30T18:29:59.900000", "camera": "Camera_01", "hole_start": 1, "hole_end": 4}


def test_hole_insert_is_idempotent_and_reuses_connection(fake_pyodbc):
    client = make_client()
    client.insert_hole_event(HOLE)
    client.insert_hole_event(HOLE)
    assert len(fake_pyodbc.connections) == 1
    conn = fake_pyodbc.connections[0]
    assert conn.autocommit is True and conn.login_timeout == 5 and conn.timeout == 15
    sql, params = conn.executed[-1]
    assert "WHERE NOT EXISTS" in sql and "EventUid = ?" in sql
    assert params[0] == HOLE["uid"] and params[-1] == HOLE["uid"]
    assert params[1].isoformat() == HOLE["event_time"]
    assert params[2].isoformat() == HOLE["event_time_utc"]


def test_missing_migration_is_reported_clearly(monkeypatch):
    mod = make_fake_pyodbc(schema_row=(16, 1, 16, None, None, None, 1, 4, 4, 1, 32, 1))
    monkeypatch.setitem(sys.modules, "pyodbc", mod)
    client = make_client()
    with pytest.raises(RuntimeError, match="sql_migrations") as e:
        client.insert_hole_event(HOLE)
    assert "EventTimeUtc" in str(e.value) and "spRefreshAnodeDaily" in str(e.value)
    assert "EventUid" not in str(e.value)
    assert mod.connections[0].closed


def test_sink_treats_duplicate_key_as_success(fake_pyodbc):
    from sql_client import SqlEventSink
    client = make_client()
    sink = SqlEventSink(client, "Camera_01")
    client._conn().raise_on_execute = fake_pyodbc.IntegrityError(
        "23000", "Cannot insert duplicate key row ... (2601)")
    sink.write("hole", HOLE)  # no exception


def test_sink_classifies_data_error_as_permanent(fake_pyodbc):
    from sql_client import SqlEventSink
    client = make_client()
    sink = SqlEventSink(client, "Camera_01")
    client._conn().raise_on_execute = fake_pyodbc.DataError("22003", "out of range")
    with pytest.raises(PermanentEventError):
        sink.write("hole", HOLE)


def test_sink_malformed_payload_is_permanent(fake_pyodbc):
    from sql_client import SqlEventSink
    sink = SqlEventSink(make_client(), "Camera_01")
    with pytest.raises(PermanentEventError):
        sink.write("hole", {"uid": "x"})
    with pytest.raises(PermanentEventError):
        sink.write("nonsense", HOLE)


def test_sink_transient_error_propagates_and_resets_connection(fake_pyodbc):
    from sql_client import SqlEventSink
    client = make_client()
    sink = SqlEventSink(client, "Camera_01")
    first = client._conn()
    first.raise_on_execute = fake_pyodbc.OperationalError("08S01", "Communication link failure")
    with pytest.raises(fake_pyodbc.OperationalError):
        sink.write("hole", HOLE)
    assert first.closed
    sink.write("hole", HOLE)  # reconnects
    assert len(fake_pyodbc.connections) == 2


def test_sink_refreshes_daily_rows_for_every_day_written(fake_pyodbc):
    from sql_client import SqlEventSink
    client = make_client()
    sink = SqlEventSink(client, "Camera_01")
    sink.write("hole", HOLE)
    sink.write("anode", ANODE)
    late = dict(HOLE, uid="6f1c1b0e-1111-4c4c-8888-000000000003", event_time="2026-10-29T23:58:00")
    sink.write("hole", late)  # delivered late after an outage: belongs to the previous day
    sink.refresh_dirty_days()
    calls = [(sql, params) for sql, params in fake_pyodbc.connections[0].executed if sql.startswith("EXEC dbo.spRefreshAnodeDaily")]
    assert [(p[0].isoformat(), p[1]) for _, p in calls] == [("2026-10-29", "Camera_01"), ("2026-10-30", "Camera_01")]
    sink.refresh_dirty_days()  # nothing new -> no further calls
    assert len([1 for sql, _ in fake_pyodbc.connections[0].executed if sql.startswith("EXEC dbo.spRefreshAnodeDaily")]) == 2


def test_duplicate_write_still_marks_day_for_refresh(fake_pyodbc):
    from sql_client import SqlEventSink
    client = make_client()
    sink = SqlEventSink(client, "Camera_01")
    client._conn().raise_on_execute = fake_pyodbc.IntegrityError("23000", "duplicate key (2627)")
    sink.write("hole", HOLE)
    sink.refresh_dirty_days()
    assert any(sql.startswith("EXEC dbo.spRefreshAnodeDaily") for sql, _ in fake_pyodbc.connections[0].executed)


def test_failed_refresh_keeps_day_dirty(fake_pyodbc):
    from sql_client import SqlEventSink
    client = make_client()
    sink = SqlEventSink(client, "Camera_01")
    sink.write("hole", HOLE)
    client._conn().raise_on_execute = fake_pyodbc.OperationalError("08S01", "link failure")
    with pytest.raises(fake_pyodbc.OperationalError):
        sink.refresh_dirty_days()
    sink.refresh_dirty_days()  # retried on the next periodic run with a new connection
    assert any(sql.startswith("EXEC dbo.spRefreshAnodeDaily") for sql, _ in fake_pyodbc.connections[-1].executed)


def test_fetch_day_counts(fake_pyodbc):
    import datetime as dt
    client = make_client()
    client._conn().day_counts = (41, 10, dt.datetime(2026, 10, 30, 13, 5))
    holes, anodes, last = client.fetch_day_counts("2026-10-30", "Camera_01")
    assert (holes, anodes, last.hour) == (41, 10, 13)
    sql, params = fake_pyodbc.connections[0].executed[-1]
    assert params[0] == "Camera_01" and params[1] == dt.date(2026, 10, 30)


def test_anode_insert_carries_grouping_evidence_and_shadow_goes_to_own_table(fake_pyodbc):
    from sql_client import SqlEventSink
    client = make_client()
    sink = SqlEventSink(client, "Camera_01")
    sink.write("anode", dict(ANODE, grouping_method="STRUCTURAL", holes_seen=3, count_status="PARTIAL"))
    sql, params = fake_pyodbc.connections[0].executed[-1]
    assert "INSERT INTO dbo.AnodeEvents" in sql
    assert params[7:10] == ("STRUCTURAL", 3, "PARTIAL")
    sink.write("anode_shadow", dict(ANODE, uid="6f1c1b0e-1111-4c4c-8888-000000000009",
                                    grouping_method="MODULO", holes_seen=4, count_status=None,
                                    slots="", close_reason=None))
    sql, params = fake_pyodbc.connections[0].executed[-1]
    assert "INSERT INTO dbo.AnodeGroupingShadow" in sql and params[4] == "MODULO"
