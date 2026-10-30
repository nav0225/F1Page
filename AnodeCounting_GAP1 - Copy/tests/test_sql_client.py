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
        self.closed = False
        self.timeout = None

    def cursor(self):
        return FakeCursor(self)

    def close(self):
        self.closed = True


def make_fake_pyodbc(schema_row=(16, 1, 16, 1)):
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
        "camera": "Camera_01", "track_id": 7, "cx": 1.5, "cy": 2.5, "conf": 0.91, "frame_id": 42}


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


def test_missing_migration_is_reported_clearly(monkeypatch):
    mod = make_fake_pyodbc(schema_row=(None, 1, None, 1))
    monkeypatch.setitem(sys.modules, "pyodbc", mod)
    client = make_client()
    with pytest.raises(RuntimeError, match="sql_migrations"):
        client.insert_hole_event(HOLE)
    assert mod.connections[0].closed


def test_sink_treats_duplicate_key_as_success(fake_pyodbc):
    from sql_client import SqlEventSink
    client = make_client()
    sink = SqlEventSink(client)
    client._conn().raise_on_execute = fake_pyodbc.IntegrityError(
        "23000", "Cannot insert duplicate key row ... (2601)")
    sink.write("hole", HOLE)  # no exception


def test_sink_classifies_data_error_as_permanent(fake_pyodbc):
    from sql_client import SqlEventSink
    client = make_client()
    sink = SqlEventSink(client)
    client._conn().raise_on_execute = fake_pyodbc.DataError("22003", "out of range")
    with pytest.raises(PermanentEventError):
        sink.write("hole", HOLE)


def test_sink_malformed_payload_is_permanent(fake_pyodbc):
    from sql_client import SqlEventSink
    sink = SqlEventSink(make_client())
    with pytest.raises(PermanentEventError):
        sink.write("hole", {"uid": "x"})
    with pytest.raises(PermanentEventError):
        sink.write("nonsense", HOLE)


def test_sink_transient_error_propagates_and_resets_connection(fake_pyodbc):
    from sql_client import SqlEventSink
    client = make_client()
    sink = SqlEventSink(client)
    first = client._conn()
    first.raise_on_execute = fake_pyodbc.OperationalError("08S01", "Communication link failure")
    with pytest.raises(fake_pyodbc.OperationalError):
        sink.write("hole", HOLE)
    assert first.closed
    sink.write("hole", HOLE)  # reconnects
    assert len(fake_pyodbc.connections) == 2
