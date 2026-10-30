"""HTTP endpoints of the counter service."""
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse


def create_app(service, writer) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        writer.start()
        service.start()
        yield
        service.stop()
        writer.stop()

    app = FastAPI(title="Anode Counter Service", lifespan=lifespan)

    def mjpeg_generator():
        while True:
            frame = service.preview_jpeg()
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
        return JSONResponse(service.stats())

    @app.get("/health")
    def health():
        report = service.health()
        return JSONResponse(report, status_code=503 if report["status"] == "down" else 200)

    @app.get("/")
    def root():
        return {"message": "Anode Counter Service running", "endpoints": ["/mjpeg", "/stats", "/health"]}

    return app
