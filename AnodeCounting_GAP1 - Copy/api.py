"""HTTP endpoints of the counter service."""
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse


def create_app(service, writer, before_start=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        if before_start is not None:
            before_start()
        writer.start()
        service.start()
        yield
        service.stop()
        writer.stop()

    app = FastAPI(title="Anode Counter Service", lifespan=lifespan)

    def mjpeg_generator():
        # Frames are only encoded while at least one viewer is connected, at
        # PREVIEW_MAX_FPS, and each image is sent once.
        service.preview_client(+1)
        last_seq = -1
        try:
            while True:
                seq, frame = service.preview_jpeg()
                if frame is None or seq == last_seq:
                    time.sleep(0.05)
                    continue
                last_seq = seq
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        finally:
            service.preview_client(-1)

    @app.get("/mjpeg")
    def mjpeg():
        return StreamingResponse(mjpeg_generator(),
                                 media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/stats")
    def stats():
        return JSONResponse(service.stats())

    @app.get("/health")
    def health():
        report = service.health()
        return JSONResponse(report, status_code=503 if report["status"] == "down" else 200)

    @app.get("/")
    def root():
        return {"message": "Anode Counter Service running", "endpoints": ["/mjpeg", "/stats", "/health"]}

    return app
