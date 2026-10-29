import pyodbc
from dataclasses import dataclass


@dataclass
class SqlConfig:
    driver: str
    server: str
    database: str
    user: str
    password: str
    trusted: bool = False


class SqlClient:
    def __init__(self, cfg: SqlConfig):
        self.cfg = cfg

    def _conn(self):
        if self.cfg.trusted:
            conn_str = (
                f"DRIVER={{{self.cfg.driver}}};"
                f"SERVER={self.cfg.server};"
                f"DATABASE={self.cfg.database};"
                "Trusted_Connection=yes;"
                "TrustServerCertificate=yes;"
            )
        else:
            conn_str = (
                f"DRIVER={{{self.cfg.driver}}};"
                f"SERVER={self.cfg.server};"
                f"DATABASE={self.cfg.database};"
                f"UID={self.cfg.user};"
                f"PWD={self.cfg.password};"
                "TrustServerCertificate=yes;"
            )
        return pyodbc.connect(conn_str, autocommit=True)

    def upsert_daily(self, count_date, camera_name, hole_count, anode_count, last_event_time):
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(
            "EXEC dbo.spUpsertAnodeDaily ?, ?, ?, ?, ?",
            count_date, camera_name, hole_count, anode_count, last_event_time
        )
        cur.close()
        conn.close()

    def insert_event(self, event_time, camera_name, track_id, cx, cy, conf=None, frame_id=None):
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO dbo.HoleCrossingEvents(EventTime, CameraName, TrackId, Cx, Cy, Confidence, FrameId)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            event_time, camera_name, int(track_id), float(cx), float(cy),
            None if conf is None else float(conf),
            None if frame_id is None else int(frame_id)
        )
        cur.close()
        conn.close()

    def insert_anode_event(self, anode_time, camera_name, hole_start, hole_end):
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(
        """
        INSERT INTO dbo.AnodeEvents
        (AnodeTime, CameraName, HoleStartNo, HoleEndNo)
        VALUES (?, ?, ?, ?)
        """,
        anode_time,
        camera_name,
        hole_start,
        hole_end
        )
        cur.close()
        conn.close()
  


    