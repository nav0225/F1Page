"""
YOLO + ByteTrack wrapper.

Converts Ultralytics results into plain Detection objects so the counting
code never touches torch tensors. Ultralytics is imported lazily so tests and
tools that do not run the model do not need torch installed.
"""
from __future__ import annotations

from counting import Detection


class YoloTracker:
    def __init__(self, model_path: str, conf: float, iou: float, device: str,
                 tracker: str = "bytetrack.yaml"):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.device = int(device) if device.isdigit() else device
        self.tracker = tracker

    @property
    def class_names(self) -> dict:
        return dict(self.model.names)

    def __call__(self, frame) -> list:
        results = self.model.track(
            frame,
            persist=True,
            conf=self.conf,
            iou=self.iou,
            tracker=self.tracker,
            verbose=False,
            device=self.device,
        )
        r0 = results[0]
        if r0.boxes is None or r0.boxes.id is None:
            return []
        boxes = r0.boxes.xyxy.cpu().numpy()
        ids = r0.boxes.id.int().cpu().numpy()
        confs = r0.boxes.conf.cpu().numpy() if r0.boxes.conf is not None else None
        return [
            Detection(int(ids[i]), float(x1), float(y1), float(x2), float(y2),
                      float(confs[i]) if confs is not None else None)
            for i, (x1, y1, x2, y2) in enumerate(boxes)
        ]
