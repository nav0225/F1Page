import os
import time
import threading
from datetime import datetime, date

import cv2
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from ultralytics import YOLO

from sql_client import SqlClient, SqlConfig

import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

load_dotenv()

# ----------------------------
# Config
# ----------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")
CONF_THRES = float(os.getenv("CONF_THRES", "0.35"))
IOU_THRES = float(os.getenv("IOU_THRES", "0.5"))

CAMERA_NAME = os.getenv("CAMERA_NAME", "Camera_01")
RTSP_URL = os.getenv("RTSP_URL", "")

LINE_Y = int(os.getenv("LINE_Y", "520"))
LINE_X1 = int(os.getenv("LINE_X1", "0"))
LINE_X2 = int(os.getenv("LINE_X2", "1920"))
COUNT_DIRECTION = os.getenv("COUNT_DIRECTION", "down").lower()  # down or up

RTSP_TRANSPORT = os.getenv("RTSP_TRANSPORT", "tcp")
OPENCV_BUFFERSIZE = int(os.getenv("OPENCV_BUFFERSIZE", "1"))

TRACK_STALE_FRAMES = int(os.getenv("TRACK_STALE_FRAMES", "60"))

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# SQL
SQL_TRUSTED = os.getenv("SQL_TRUSTED", "false").lower() == "true"
sql_cfg = SqlConfig(
    driver=os.getenv("SQL_DRIVER", "ODBC Driver 17 for SQL Server"),
    server=os.getenv("SQL_SERVER", ""),
    database=os.getenv("SQL_DB", ""),
    user=os.getenv("SQL_USER", ""),
    password=os.getenv("SQL_PASS", ""),
    trusted=SQL_TRUSTED,
)
sql = SqlClient(sql_cfg)

# Low latency hints for FFmpeg backend (often helps)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{RTSP_TRANSPORT}|fflags;nobuffer|flags;low_delay"

# ----------------------------
# Helpers
# ----------------------------
def crossed(prev_y, curr_y, line_y):
    if COUNT_DIRECTION == "down":
        return prev_y < line_y <= curr_y
    else:
        return prev_y > line_y >= curr_y


# ----------------------------
# Shared State
# ----------------------------
state_lock = threading.Lock()
latest_jpeg = None
latest_stats = {
    "camera": CAMERA_NAME,
    "date": date.today().isoformat(),
    "hole_count": 0,
    "anode_count": 0,
    "last_event_time": None,
    "fps": 0.0,
}

track_last_y = {}        # track_id -> last y
track_last_seen = {}     # track_id -> last frame_id
track_counted = set()    # track_id already counted

# ----------------------------
# Inference loop
# ----------------------------
def inference_loop():
    global latest_jpeg, latest_stats

    if not RTSP_URL:
        raise RuntimeError("RTSP_URL is missing in .env")
    
    torch.cuda.empty_cache()
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, OPENCV_BUFFERSIZE)
    
    VIDEO_PATH = os.getenv("VIDEO_PATH")

    # if VIDEO_PATH and os.path.exists(VIDEO_PATH):
    #  print(f"[INFO] Using VIDEO file: {VIDEO_PATH}")
    #  cap = cv2.VideoCapture(VIDEO_PATH)
    # else:
    #  print(f"[INFO] Using RTSP stream")
    #  cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    #  cap.set(cv2.CAP_PROP_BUFFERSIZE, OPENCV_BUFFERSIZE)

    

    hole_count = 0
    anode_count = 0

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
            cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, OPENCV_BUFFERSIZE)
            continue

        frame_id += 1
        frames += 1

        # FPS calc
        dt = time.time() - t0
        if dt >= 1.0:
            fps = frames / dt
            frames = 0
            t0 = time.time()

        # YOLO tracking with persist IDs using ByteTrack [1](https://vedanta-my.sharepoint.com/personal/11020436_vedanta_co_in/_layouts/15/Doc.aspx?sourcedoc=%7B65C35DB6-DAE6-45C1-B92B-334AEC2F77F9%7D&file=Digital_MMR_13th_Sept.pptx&action=edit&mobileredirect=true&DefaultItemOpen=1)
        results = model.track(
            frame,
            persist=True,
            conf=CONF_THRES,
            iou=IOU_THRES,
            tracker="bytetrack.yaml",
            verbose=False,
            device=0
        )

        annotated = frame.copy()
        cv2.line(annotated, (LINE_X1, LINE_Y), (LINE_X2, LINE_Y), (0, 0, 255), 3)

        now = datetime.now()
        last_event_time = None

        r0 = results[0]
        if r0.boxes is not None and r0.boxes.id is not None:
            boxes = r0.boxes.xyxy.cpu().numpy()
            ids = r0.boxes.id.int().cpu().numpy()
            confs = r0.boxes.conf.cpu().numpy() if r0.boxes.conf is not None else None

            for i, (x1, y1, x2, y2) in enumerate(boxes):
                tid = int(ids[i])
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                conf = float(confs[i]) if confs is not None else None
                # Only consider detections with confidence >= 80%
                if conf is None or conf < 0.80:
                  continue

                # Draw detection
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(annotated, f"ID:{tid}", (int(x1), int(y1) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                prev_y = track_last_y.get(tid, cy)
                track_last_y[tid] = cy
                track_last_seen[tid] = frame_id

                # Count only once per TrackID when crossing line
                # if tid not in track_counted and crossed(prev_y, cy, LINE_Y):
                #     hole_count += 1
                #     track_counted.add(tid)
                #     last_event_time = now

                #     # 4 holes = 1 anode
                #     if hole_count % 4 == 0:
                #         anode_count += 1
                
                if tid not in track_counted and crossed(prev_y, cy, LINE_Y):
                    hole_count += 1
                    track_counted.add(tid)
                    last_event_time = now

                    # ✅ Insert hole event (already present)
                    sql.insert_event(
                    now,
                    CAMERA_NAME,
                    tid,
                    cx,
                    cy,
                    conf=conf
                    )

                    # ✅ Every 4 holes → create 1 anode event

                    if hole_count % 4 == 0:
                        hole_end = hole_count
                        hole_start = hole_count - 3
                        anode_count += 1
 
                        # ✅ INSERT new anode row
                        sql.insert_anode_event(
                        anode_time=now,
                        camera_name=CAMERA_NAME,
                        hole_start=hole_start,
                        hole_end=hole_end
                        )
 
                        # ✅ Immediately update SQL for anode
                        sql.upsert_daily(
                          date.today(),
                          CAMERA_NAME,
                          hole_count,
                          anode_count,
                          last_event_time
                        )

                   

        # Cleanup stale tracks
        stale = [tid for tid, last in track_last_seen.items() if frame_id - last > TRACK_STALE_FRAMES]
        for tid in stale:
            track_last_seen.pop(tid, None)
            track_last_y.pop(tid, None)
            # NOTE: keep tid in track_counted to avoid recount if it reappears.

        # Overlay counters
        cv2.putText(annotated, f"Holes: {hole_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(annotated, f"Anodes: {anode_count}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(annotated, f"FPS: {fps:.1f}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Update shared state
        ok2, jpg = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok2:
            with state_lock:
                latest_jpeg = jpg.tobytes()
                latest_stats = {
                    "camera": CAMERA_NAME,
                    "date": date.today().isoformat(),
                    "hole_count": hole_count,
                    "anode_count": anode_count,
                    "last_event_time": last_event_time.isoformat() if last_event_time else latest_stats.get("last_event_time"),
                    "fps": float(fps),
                }

        # Upsert daily totals periodically
        if (time.time() - last_sql_push > 5) or (last_event_time and time.time() - last_sql_push > 1):
            try:
                sql.upsert_daily(date.today(), CAMERA_NAME, hole_count, anode_count, last_event_time)
                last_sql_push = time.time()
            except Exception as e:
                print("SQL upsert_daily failed:", e)


# ----------------------------
# FastAPI
# ----------------------------
app = FastAPI(title="Anode Counter Service")

@app.on_event("startup")
def startup():
    th = threading.Thread(target=inference_loop, daemon=True)
    th.start()

def mjpeg_generator():
    # Standard MJPEG streaming generator [2](https://outlook.office365.com/owa/?ItemID=AAMkAGE3YzdkYzgwLTA4OGYtNDZjMy1hNDgzLTJlMjRmMDRhYjA3MABGAAAAAADhivuuYma5R4yybI5YfWmjBwB2Kz0H1B6ATpY2rZYxtDoZAAAAAAEMAAB2Kz0H1B6ATpY2rZYxtDoZAAFgwaQVAAA%3d&exvsurl=1&viewmodel=ReadMessageItem)
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
