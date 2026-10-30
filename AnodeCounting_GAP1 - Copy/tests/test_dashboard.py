import os
import sys
import types
from datetime import date, datetime

import pytest

pd = pytest.importorskip("pandas")

from dashboard_data import DashboardData, hourly, unobserved_seconds

APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DAY = date(2026, 10, 30)
NEW_OBJECTS = {"dbo.CountingGaps", "dbo.RejectedHoleCrossings", "dbo.AnodeGroupingShadow"}
NEW_COLUMNS = {("dbo.AnodeEvents", c) for c in ("GroupingMethod", "HolesSeen", "CountStatus")}


class FakeCursor:
    def __init__(self, db):
        self.db = db
        self.description = None
        self._rows = []

    def execute(self, sql, *params):
        self.db.executed.append((sql, params))
        cols, rows = self.db.answer(sql, params)
        self.description = [(c,) for c in cols]
        self._rows = rows
        return self

    def fetchall(self):
        return self._rows

    def close(self):
        pass


class FakeDb:
    """Answers the dashboard's queries from in-memory tables."""

    def __init__(self, migrated=True):
        self.migrated = migrated
        self.executed = []
        self.anodes = [  # AnodeId, AnodeTime, HoleStartNo, HoleEndNo, GroupingMethod, HolesSeen, CountStatus
            (1, datetime(2026, 10, 29, 23, 59), 1, 4, "MODULO", 4, None),      # previous day
            (2, datetime(2026, 10, 30, 6, 10), 1, 4, "STRUCTURAL", 4, "CONFIRMED"),
            (3, datetime(2026, 10, 30, 6, 20), 5, 7, "STRUCTURAL", 3, "PARTIAL"),
            (4, datetime(2026, 10, 30, 14, 5), 8, 11, "STRUCTURAL", 4, "CONFIRMED"),
        ]
        self.gaps = [
            (datetime(2026, 10, 29, 23, 50), datetime(2026, 10, 30, 0, 10), 1_200_000, "SERVICE_DOWN", "x"),
            (datetime(2026, 10, 30, 9, 0), datetime(2026, 10, 30, 9, 5), 300_000, "LOW_VISIBILITY", "y"),
        ]

    def cursor(self):
        return FakeCursor(self)

    def close(self):
        pass

    def answer(self, sql, params):
        if sql.startswith("SELECT OBJECT_ID"):
            return ["v"], [(1 if self.migrated and params[0] in NEW_OBJECTS else None,)]
        if sql.startswith("SELECT COL_LENGTH"):
            return ["v"], [(8 if self.migrated and tuple(params) in NEW_COLUMNS else None,)]
        if "FROM dbo.AnodeEvents" in sql:
            cols = sql.split("SELECT ")[1].split(" FROM")[0].split(", ")
            _, start, end = params
            rows = [r[:len(cols)] for r in self.anodes if start <= r[1] < end]
            return cols, rows
        if "FROM dbo.HoleCrossingEvents" in sql:
            return ["n"], [(11,)]
        if "FROM dbo.AnodeCountDaily" in sql:
            return ["HoleCount", "AnodeCount", "LastEventTime", "UpdatedAt"], [(11, 3, None, None)]
        if "FROM dbo.CountingGaps" in sql:
            _, end, start = params
            return ["StartTime", "EndTime", "DurationMs", "Reason", "Details"], \
                [g for g in self.gaps if g[0] < end and g[1] >= start]
        if "FROM dbo.RejectedHoleCrossings" in sql:
            return ["n"], [(2,)]
        if "FROM dbo.AnodeGroupingShadow" in sql:
            return ["GroupingMethod", "CountStatus", "Anodes"], [("MODULO", "NOT ASSESSED", 3)]
        raise AssertionError(f"unexpected query: {sql}")


def test_day_queries_on_migrated_schema():
    data = DashboardData(FakeDb(), "Camera_01")
    anodes = data.anodes(DAY)
    assert list(anodes["AnodeId"]) == [2, 3, 4]                     # only the selected day
    assert list(anodes["CountStatus"]) == ["CONFIRMED", "PARTIAL", "CONFIRMED"]
    assert data.hole_count(DAY) == 11 and data.rejected_count(DAY) == 2
    gaps = data.gaps(DAY)
    assert list(gaps["Reason"]) == ["SERVICE_DOWN", "LOW_VISIBILITY"]
    assert unobserved_seconds(gaps, DAY) == 600 + 300                # gap across midnight clipped to 10 min
    assert data.shadow_summary(DAY)["Anodes"].sum() == 3
    by_hour = hourly(anodes)
    assert len(by_hour) == 24 and by_hour.loc[6, "Anodes"] == 2 and by_hour.loc[14, "Anodes"] == 1


def test_day_queries_before_migrations():
    data = DashboardData(FakeDb(migrated=False), "Camera_01")
    anodes = data.anodes(DAY)
    assert list(anodes.columns) == ["AnodeId", "AnodeTime", "HoleStartNo", "HoleEndNo"]
    assert data.gaps(DAY) is None and data.rejected_count(DAY) is None and data.shadow_summary(DAY) is None


@pytest.mark.parametrize("migrated", [True, False])
def test_dashboard_page_renders(monkeypatch, migrated):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    fake = types.ModuleType("pyodbc")
    fake.connect = lambda *a, **k: FakeDb(migrated)
    monkeypatch.setitem(sys.modules, "pyodbc", fake)
    monkeypatch.setenv("SERVICE_URL", "http://127.0.0.1:9")          # nothing listening
    monkeypatch.chdir(APP_DIR)
    monkeypatch.syspath_prepend(APP_DIR)

    at = AppTest.from_file(os.path.join(APP_DIR, "dashboard_app.py"), default_timeout=30)
    at.run()                                   # widgets exist after the first run
    at.sidebar.date_input[0].set_value(DAY)
    at.run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Anodes"] == "3" and metrics["Holes"] == "11"
    assert metrics["Belt not observable"] == ("15.0 min" if migrated else "n/a")
    assert any("not reachable" in e.value for e in at.error)        # service down is shown, page still works


def test_anodes_without_assessment_are_shown_as_not_assessed(monkeypatch):
    """Default modulo grouping: every CountStatus is NULL, shown as NOT ASSESSED."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    db = FakeDb()
    db.anodes = [r[:4] + ("MODULO", 4, None) for r in db.anodes]
    fake = types.ModuleType("pyodbc")
    fake.connect = lambda *a, **k: db
    monkeypatch.setitem(sys.modules, "pyodbc", fake)
    monkeypatch.setenv("SERVICE_URL", "http://127.0.0.1:9")
    monkeypatch.chdir(APP_DIR)
    monkeypatch.syspath_prepend(APP_DIR)

    at = AppTest.from_file(os.path.join(APP_DIR, "dashboard_app.py"), default_timeout=30)
    at.run()
    at.sidebar.date_input[0].set_value(DAY)
    at.run()
    assert not at.exception
    assert any("NOT ASSESSED: 3" in c.value for c in at.caption)
