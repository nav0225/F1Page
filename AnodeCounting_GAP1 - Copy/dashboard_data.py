"""
Queries for the operator dashboard.

All anode numbers come from dbo.AnodeEvents, the table the Power BI report
counts, so the dashboard and the report cannot disagree. Objects added by
sql_migrations/ are used only when they exist, so the dashboard works before
and after the migrations are applied.
"""
from datetime import datetime, timedelta

import pandas as pd


def day_range(day):
    start = datetime(day.year, day.month, day.day)
    return start, start + timedelta(days=1)


class DashboardData:
    def __init__(self, conn, camera: str):
        self.conn = conn
        self.camera = camera
        self._exists = {}

    # ----------------------------
    # Helpers
    # ----------------------------
    def _rows(self, sql, *params):
        cur = self.conn.cursor()
        try:
            cur.execute(sql, *params)
            columns = [c[0] for c in cur.description] if cur.description else []
            return columns, [tuple(r) for r in cur.fetchall()]
        finally:
            cur.close()

    def _frame(self, sql, *params) -> pd.DataFrame:
        columns, rows = self._rows(sql, *params)
        return pd.DataFrame.from_records(rows, columns=columns)

    def _scalar(self, sql, *params):
        _, rows = self._rows(sql, *params)
        return rows[0][0] if rows else None

    def exists(self, obj: str, column: str = None) -> bool:
        key = (obj, column)
        if key not in self._exists:
            if column is None:
                value = self._scalar("SELECT OBJECT_ID(?)", obj)
            else:
                value = self._scalar("SELECT COL_LENGTH(?, ?)", obj, column)
            self._exists[key] = value is not None
        return self._exists[key]

    # ----------------------------
    # Day queries
    # ----------------------------
    def anodes(self, day) -> pd.DataFrame:
        start, end = day_range(day)
        extra = [c for c in ("GroupingMethod", "HolesSeen", "CountStatus") if self.exists("dbo.AnodeEvents", c)]
        cols = ", ".join(["AnodeId", "AnodeTime", "HoleStartNo", "HoleEndNo"] + extra)
        return self._frame(
            f"SELECT {cols} FROM dbo.AnodeEvents WHERE CameraName = ? AND AnodeTime >= ? AND AnodeTime < ? "
            "ORDER BY AnodeTime", self.camera, start, end)

    def hole_count(self, day) -> int:
        return int(self._scalar(
            "SELECT COUNT(*) FROM dbo.HoleCrossingEvents WHERE CameraName = ? AND CountDate = ?",
            self.camera, day) or 0)

    def daily_row(self, day):
        df = self._frame(
            "SELECT HoleCount, AnodeCount, LastEventTime, UpdatedAt FROM dbo.AnodeCountDaily "
            "WHERE CountDate = ? AND CameraName = ?", day, self.camera)
        return None if df.empty else df.iloc[0].to_dict()

    def gaps(self, day):
        """Counting gaps overlapping the day, or None before migration 003."""
        if not self.exists("dbo.CountingGaps"):
            return None
        start, end = day_range(day)
        return self._frame(
            "SELECT StartTime, EndTime, DurationMs, Reason, Details FROM dbo.CountingGaps "
            "WHERE CameraName = ? AND StartTime < ? AND EndTime >= ? ORDER BY StartTime",
            self.camera, end, start)

    def rejected_count(self, day):
        if not self.exists("dbo.RejectedHoleCrossings"):
            return None
        start, end = day_range(day)
        return int(self._scalar(
            "SELECT COUNT(*) FROM dbo.RejectedHoleCrossings WHERE CameraName = ? AND EventTime >= ? AND EventTime < ?",
            self.camera, start, end) or 0)

    def shadow_summary(self, day):
        """Anodes by method and status from the comparison grouping, or None before migration 005."""
        if not self.exists("dbo.AnodeGroupingShadow"):
            return None
        start, end = day_range(day)
        return self._frame(
            "SELECT GroupingMethod, ISNULL(CountStatus, 'NOT ASSESSED') AS CountStatus, COUNT(*) AS Anodes "
            "FROM dbo.AnodeGroupingShadow WHERE CameraName = ? AND AnodeTime >= ? AND AnodeTime < ? "
            "GROUP BY GroupingMethod, CountStatus ORDER BY GroupingMethod, CountStatus",
            self.camera, start, end)


def unobserved_seconds(gaps: pd.DataFrame, day) -> float:
    """Total gap time inside the day (gaps crossing midnight are clipped)."""
    if gaps is None or gaps.empty:
        return 0.0
    start, end = day_range(day)
    total = 0.0
    for s, e in zip(pd.to_datetime(gaps["StartTime"]), pd.to_datetime(gaps["EndTime"])):
        total += max(0.0, (min(e, end) - max(s, start)).total_seconds())
    return total


def hourly(anodes: pd.DataFrame) -> pd.DataFrame:
    """Anodes per hour of the day (0-23), including empty hours."""
    hours = pd.to_datetime(anodes["AnodeTime"]).dt.hour if not anodes.empty else pd.Series(dtype=int)
    counts = hours.value_counts().reindex(range(24), fill_value=0).sort_index()
    return pd.DataFrame({"Hour": counts.index, "Anodes": counts.values}).set_index("Hour")
