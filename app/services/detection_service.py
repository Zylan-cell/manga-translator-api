from typing import List, Dict, Any
from ultralytics import YOLO

from app.config import settings
from app.logger import app_logger
from app.utils.images import b64_to_pil

class DetectionService:
    def __init__(self):
        self.yolo_model = None

    async def initialize(self):
        if self.yolo_model is None:
            app_logger.info(f"Loading Bubble YOLO model from {settings.YOLO_MODEL_PATH}...")
            self.yolo_model = YOLO(settings.YOLO_MODEL_PATH)
            app_logger.info("Bubble YOLO model loaded.")

    def detect(self, image_b64: str) -> List[Dict[str, Any]]:
        if self.yolo_model is None:
            raise RuntimeError("YOLO service not initialized.")
        image = b64_to_pil(image_b64)
        results = self.yolo_model(image, verbose=False)

        boxes_data: List[Dict[str, Any]] = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                coords = box.xyxy[0].tolist()
                boxes_data.append({
                    "x1": int(coords[0]),
                    "y1": int(coords[1]),
                    "x2": int(coords[2]),
                    "y2": int(coords[3]),
                    "confidence": float(box.conf[0]),
                })
        return boxes_data