"""
Entry point of the anode counter service.

Run with exactly one worker process (the counter owns the camera stream and
the counting state):
    python service_app.py
or
    uvicorn service_app:app --host 0.0.0.0 --port 8000
"""
import logging
import os

import cv2
from dotenv import load_dotenv

from api import create_app
from capture import FrameGrabber, ffmpeg_capture_options, open_file, open_rtsp
from config import load_settings
from counter_service import CounterService
from instance_lock import InstanceLock
from overlay import draw_overlay
from persistence import EventWriter, Outbox
from sql_client import SqlClient, SqlConfig, SqlEventSink

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("anode_counter")

load_dotenv()

# ----------------------------
# Config
# ----------------------------
settings = load_settings()

# FFmpeg options for RTSP. "fflags;nobuffer" was removed on purpose: it made
# FFmpeg drop the frames read while probing at every (re)connect (see capture.py).
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = ffmpeg_capture_options(settings.rtsp_transport)


def log_device():
    try:
        import torch
        if torch.cuda.is_available():
            log.info("CUDA devices: %d, using %s (%s)", torch.cuda.device_count(),
                     settings.device, torch.cuda.get_device_name(0))
        else:
            log.warning("CUDA not available; DEVICE=%s", settings.device)
    except Exception as e:
        log.warning("Could not query torch device info: %s", e)


def make_detector():
    from detector import YoloTracker
    log_device()
    return YoloTracker(settings.model_path, settings.conf_thres, settings.iou_thres, settings.device)


def make_grabber():
    if settings.source == "file":
        log.info("Using VIDEO file: %s", settings.video_path)
        return FrameGrabber(lambda: open_file(settings.video_path), drop_stale=False,
                            reconnect_max_s=settings.reconnect_max_s)
    log.info("Using RTSP stream")
    return FrameGrabber(lambda: open_rtsp(settings.rtsp_url, settings.rtsp_open_timeout_ms,
                                          settings.rtsp_read_timeout_ms),
                        drop_stale=True, buffer_frames=settings.capture_buffer_frames,
                        reconnect_max_s=settings.reconnect_max_s)


def encode_preview(frame, result, counter, totals, fps):
    line = (settings.line_x1, settings.line_y, settings.line_x2)
    annotated = draw_overlay(frame, result, counter, line, totals["holes"], totals["anodes"], fps)
    h, w = annotated.shape[:2]
    if w > settings.preview_width:
        annotated = cv2.resize(annotated, (settings.preview_width, h * settings.preview_width // w),
                               interpolation=cv2.INTER_AREA)
    ok, jpg = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), settings.preview_jpeg_quality])
    return jpg.tobytes() if ok else None


# ----------------------------
# Wiring
# ----------------------------
os.makedirs(os.path.dirname(os.path.abspath(settings.outbox_path)), exist_ok=True)
outbox = Outbox(settings.outbox_path)
sql = SqlClient(SqlConfig(
    driver=settings.sql_driver,
    server=settings.sql_server,
    database=settings.sql_db,
    user=settings.sql_user,
    password=settings.sql_pass,
    trusted=settings.sql_trusted,
))
sink = SqlEventSink(sql, settings.camera_name)
# The writer recomputes the AnodeCountDaily row of every day that received events.
writer = EventWriter(outbox, sink, periodic=sink.refresh_dirty_days,
                     prune_days=settings.outbox_retention_days)
service = CounterService(settings, outbox, writer, make_detector, make_grabber(), encode_preview)


instance_lock = InstanceLock(os.path.join(os.path.dirname(os.path.abspath(settings.outbox_path)), "counter.lock"))


def before_start():
    instance_lock.acquire()   # refuse to start a second counter for the same outbox/camera
    service.restore_today(lambda day: sql.fetch_day_counts(day, settings.camera_name))


app = create_app(service, writer, before_start=before_start)


def main():
    import uvicorn
    uvicorn.run(app, host=settings.api_host, port=settings.api_port, workers=1)


if __name__ == "__main__":
    main()
