import logging
import os
import time
import threading
from datetime import datetime, date

import cv2
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse

from config import load_settings
from counting import LineCrossingCounter, ModuloGrouper
from detector import YoloTracker
from overlay import draw_overlay
from pipeline import FrameProcessor
from sql_client import SqlClient, SqlConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("anode_counter")

load_dotenv()

# ----------------------------
# Config
# ----------------------------
settings = load_settings()

sql = SqlClient(SqlConfig(
    driver=settings.sql_driver,
    server=settings.sql_server,
    database=settings.sql_db,
    user=settings.sql_user,
    password=settings.sql_pass,
    trusted=settings.sql_trusted,
))

# Low latency hints for FFmpeg backend
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{settings.rtsp_transport}|fflags;nobuffer|flags;low_delay"


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


def open_source():
    if settings.source == "file":
        log.info("Using VIDEO file: %s", settings.video_path)
        return cv2.VideoCapture(settings.video_path)
    log.info("Using RTSP stream")
    return cv2.VideoCapture(settings.rtsp_url, cv2.CAP_FFMPEG)


# ----------------------------
# Shared State
# ----------------------------
state_lock = threading.Lock()
latest_jpeg = None
latest_stats = {
    "camera": settings.camera_name,
    "date": date.today().isoformat(),
    "hole_count": 0,
    "anode_count": 0,
    "last_event_time": None,
    "fps": 0.0,
}


# ----------------------------
# Inference loop
# ----------------------------
def inference_loop():
    global latest_jpeg, latest_stats

    log_device()
    detector = YoloTracker(settings.model_path, settings.conf_thres, settings.iou_thres, settings.device)
    counter = LineCrossingCounter(settings.line_y, settings.count_direction,
                                  settings.count_min_conf, settings.track_stale_frames)
    grouper = ModuloGrouper(settings.holes_per_anode)
    processor = FrameProcessor(detector, counter, grouper)
    line = (settings.line_x1, settings.line_y, settings.line_x2)

    cap = open_source()

    frame_id = 0
    last_sql_push = 0.0

    t0 = time.time()
    frames = 0
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            # Reconnect
            cap.release()
            time.sleep(1)
            cap = open_source()
            continue

        frame_id += 1
        frames += 1

        # FPS calc
        dt = time.time() - t0
        if dt >= 1.0:
            fps = frames / dt
            frames = 0
            t0 = time.time()

        result = processor.process(frame_id, frame)

        now = datetime.now()
        last_event_time = None

        for hole in result.holes:
            last_event_time = now
            sql.insert_event(now, settings.camera_name, hole.track_id, hole.cx, hole.cy, conf=hole.conf)

        for group in result.anodes:
            sql.insert_anode_event(
                anode_time=now,
                camera_name=settings.camera_name,
                hole_start=group.hole_start,
                hole_end=group.hole_end,
            )
            sql.upsert_daily(date.today(), settings.camera_name, counter.hole_count,
                             grouper.anode_count, last_event_time)

        annotated = draw_overlay(frame, result, counter, line, counter.hole_count, grouper.anode_count, fps)

        # Update shared state
        ok2, jpg = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok2:
            with state_lock:
                latest_jpeg = jpg.tobytes()
                latest_stats = {
                    "camera": settings.camera_name,
                    "date": date.today().isoformat(),
                    "hole_count": counter.hole_count,
                    "anode_count": grouper.anode_count,
                    "last_event_time": last_event_time.isoformat() if last_event_time else latest_stats.get("last_event_time"),
                    "fps": float(fps),
                }

        # Upsert daily totals periodically
        if (time.time() - last_sql_push > 5) or (last_event_time and time.time() - last_sql_push > 1):
            try:
                sql.upsert_daily(date.today(), settings.camera_name, counter.hole_count,
                                 grouper.anode_count, last_event_time)
                last_sql_push = time.time()
            except Exception as e:
                log.error("SQL upsert_daily failed: %s", e)


# ----------------------------
# FastAPI
# ----------------------------
app = FastAPI(title="Anode Counter Service")


@app.on_event("startup")
def startup():
    th = threading.Thread(target=inference_loop, daemon=True)
    th.start()


def mjpeg_generator():
    while True:
        with state_lock:
            frame = latest_jpeg
        if frame is None:
            time.sleep(0.05)
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.03)


@app.get("/mjpeg")
def mjpeg():
    return StreamingResponse(mjpeg_generator(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/stats")
def stats():
    with state_lock:
        return JSONResponse(latest_stats)


@app.get("/")
def root():
    return {"message": "Anode Counter Service running", "endpoints": ["/mjpeg", "/stats"]}
