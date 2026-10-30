"""
Validated settings for the anode counter.

All configuration comes from environment variables (normally a .env file).
Settings are checked once at startup so that a typo fails loudly instead of
silently changing how anodes are counted.
"""
import os
from dataclasses import dataclass, field
from typing import Mapping, Optional


class ConfigError(ValueError):
    """Raised when one or more settings are missing or invalid."""


@dataclass(frozen=True)
class Settings:
    # Model / tracker
    model_path: str = "best.pt"
    conf_thres: float = 0.10       # detector threshold for tracker input (Ultralytics tracking default)
    iou_thres: float = 0.5
    count_min_conf: float = 0.80
    count_evidence: str = "track"  # track | frame (legacy per-frame gate)
    count_min_hits: int = 2
    device: str = "0"

    # Video source
    camera_name: str = "Camera_01"
    source: str = "rtsp"  # rtsp | file
    rtsp_url: str = field(default="", repr=False)  # may embed credentials
    video_path: str = ""
    rtsp_transport: str = "tcp"
    rtsp_open_timeout_ms: int = 10000
    rtsp_read_timeout_ms: int = 5000
    reconnect_max_s: float = 30.0
    # Frames buffered between capture and counting (absorbs short hiccups;
    # bounds how far counting can fall behind: 10 frames = 0.5 s at 20 fps)
    capture_buffer_frames: int = 10
    # Longest acceptable interval between processed frames. Above it holes can
    # pass the line unobserved, so the interval is recorded as a counting gap.
    # Derived from Video1.mp4 belt speed; re-check at maximum belt speed.
    max_frame_gap_ms: int = 300

    # Counting line
    line_y: int = 520
    line_x1: int = 0
    line_x2: int = 1920
    count_direction: str = "up"
    holes_per_anode: int = 4
    track_stale_frames: int = 60

    # API / health
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    health_max_frame_age_s: float = 10.0
    startup_grace_s: float = 120.0

    # Local event outbox (survives SQL Server outages and process restarts)
    outbox_path: str = "data/outbox.sqlite3"
    outbox_retention_days: int = 30

    # SQL Server
    sql_driver: str = "ODBC Driver 17 for SQL Server"
    sql_server: str = ""
    sql_db: str = ""
    sql_user: str = ""
    sql_pass: str = field(default="", repr=False)
    sql_trusted: bool = False


def _parse_bool(raw: str) -> bool:
    value = raw.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"expected true/false, got {raw!r}")


def load_settings(env: Optional[Mapping[str, str]] = None, require_sql: bool = True) -> Settings:
    """
    Build Settings from environment variables and validate them.

    Every problem is collected and reported together so an operator can fix a
    .env file in one pass.
    """
    env = os.environ if env is None else env
    errors = []
    defaults = Settings()

    def get_str(key, default):
        value = env.get(key)
        return default if value is None or value.strip() == "" else value.strip()

    def get_num(key, default, cast, lo=None, hi=None):
        raw = env.get(key)
        if raw is None or raw.strip() == "":
            return default
        try:
            value = cast(raw.strip())
        except ValueError:
            errors.append(f"{key}={raw!r} is not a valid {cast.__name__}")
            return default
        if (lo is not None and value < lo) or (hi is not None and value > hi):
            errors.append(f"{key}={value} must be between {lo} and {hi}")
        return value

    def get_bool(key, default):
        raw = env.get(key)
        if raw is None or raw.strip() == "":
            return default
        try:
            return _parse_bool(raw)
        except ValueError as e:
            errors.append(f"{key}: {e}")
            return default

    # COUNT_DIRECTION has no safe default: counting the wrong direction
    # silently produces zero counts, so it must be set explicitly.
    direction = get_str("COUNT_DIRECTION", "").lower()
    if direction not in ("up", "down"):
        errors.append("COUNT_DIRECTION must be set to 'up' or 'down' "
                      "(up = objects move from bottom to top of the image)")

    evidence = get_str("COUNT_EVIDENCE", defaults.count_evidence).lower()
    if evidence not in ("track", "frame"):
        errors.append(f"COUNT_EVIDENCE={evidence!r} must be 'track' or 'frame'")

    source = get_str("SOURCE", defaults.source).lower()
    if source not in ("rtsp", "file"):
        errors.append(f"SOURCE={source!r} must be 'rtsp' or 'file'")

    settings = Settings(
        model_path=get_str("MODEL_PATH", defaults.model_path),
        conf_thres=get_num("CONF_THRES", defaults.conf_thres, float, 0.0, 1.0),
        iou_thres=get_num("IOU_THRES", defaults.iou_thres, float, 0.0, 1.0),
        count_min_conf=get_num("COUNT_MIN_CONF", defaults.count_min_conf, float, 0.0, 1.0),
        count_evidence=evidence or defaults.count_evidence,
        count_min_hits=get_num("COUNT_MIN_HITS", defaults.count_min_hits, int, 1, 100),
        device=get_str("DEVICE", defaults.device),
        camera_name=get_str("CAMERA_NAME", defaults.camera_name),
        source=source,
        rtsp_url=get_str("RTSP_URL", ""),
        video_path=get_str("VIDEO_PATH", ""),
        rtsp_transport=get_str("RTSP_TRANSPORT", defaults.rtsp_transport).lower(),
        rtsp_open_timeout_ms=get_num("RTSP_OPEN_TIMEOUT_MS", defaults.rtsp_open_timeout_ms, int, 1000),
        rtsp_read_timeout_ms=get_num("RTSP_READ_TIMEOUT_MS", defaults.rtsp_read_timeout_ms, int, 500),
        reconnect_max_s=get_num("RECONNECT_MAX_S", defaults.reconnect_max_s, float, 1.0),
        capture_buffer_frames=get_num("CAPTURE_BUFFER_FRAMES", defaults.capture_buffer_frames, int, 1, 200),
        max_frame_gap_ms=get_num("MAX_FRAME_GAP_MS", defaults.max_frame_gap_ms, int, 50),
        line_y=get_num("LINE_Y", defaults.line_y, int, 0),
        line_x1=get_num("LINE_X1", defaults.line_x1, int, 0),
        line_x2=get_num("LINE_X2", defaults.line_x2, int, 0),
        count_direction=direction or defaults.count_direction,
        holes_per_anode=get_num("HOLES_PER_ANODE", defaults.holes_per_anode, int, 1, 16),
        track_stale_frames=get_num("TRACK_STALE_FRAMES", defaults.track_stale_frames, int, 1),
        api_host=get_str("API_HOST", defaults.api_host),
        api_port=get_num("API_PORT", defaults.api_port, int, 1, 65535),
        health_max_frame_age_s=get_num("HEALTH_MAX_FRAME_AGE_S", defaults.health_max_frame_age_s, float, 1.0),
        startup_grace_s=get_num("STARTUP_GRACE_S", defaults.startup_grace_s, float, 0.0),
        outbox_path=get_str("OUTBOX_PATH", defaults.outbox_path),
        outbox_retention_days=get_num("OUTBOX_RETENTION_DAYS", defaults.outbox_retention_days, int, 1),
        sql_driver=get_str("SQL_DRIVER", defaults.sql_driver),
        sql_server=get_str("SQL_SERVER", ""),
        sql_db=get_str("SQL_DB", ""),
        sql_user=get_str("SQL_USER", ""),
        sql_pass=env.get("SQL_PASS", "") or "",
        sql_trusted=get_bool("SQL_TRUSTED", defaults.sql_trusted),
    )

    if settings.rtsp_transport not in ("tcp", "udp"):
        errors.append(f"RTSP_TRANSPORT={settings.rtsp_transport!r} must be 'tcp' or 'udp'")
    if settings.line_x1 >= settings.line_x2:
        errors.append("LINE_X1 must be smaller than LINE_X2")
    if source == "rtsp" and not settings.rtsp_url:
        errors.append("RTSP_URL is required when SOURCE=rtsp")
    if source == "file" and not settings.video_path:
        errors.append("VIDEO_PATH is required when SOURCE=file")
    if require_sql:
        if not settings.sql_server or not settings.sql_db:
            errors.append("SQL_SERVER and SQL_DB are required")
        if not settings.sql_trusted and not settings.sql_user:
            errors.append("SQL_USER is required unless SQL_TRUSTED=true")

    if errors:
        raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(errors))
    return settings
