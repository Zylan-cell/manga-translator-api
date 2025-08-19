import base64
from io import BytesIO
import numpy as np
from typing import List, Dict, Any
from ultralytics import YOLO
from PIL import Image  # <-- ДОБАВЛЕН ЭТОТ ИМПОРТ

from app.config import YOLO_MODEL_PATH
from app.logger import app_logger

class DetectionService:
    def __init__(self):
        self.yolo_model = None

    async def initialize(self):
        if self.yolo_model is None:
            app_logger.info(f"Loading Bubble YOLO model from {YOLO_MODEL_PATH}...")
            self.yolo_model = YOLO(YOLO_MODEL_PATH)
            app_logger.info("Bubble YOLO model loaded.")

    def detect(self, image_b64: str) -> List[Dict[str, Any]]:
        if self.yolo_model is None:
            raise Exception("YOLO service not initialized.")
        
        image_data = base64.b64decode(image_b64.split(',')[-1])
        image = Image.open(BytesIO(image_data)).convert("RGB")
        
        results = self.yolo_model(image, verbose=False)
        
        boxes_data = []
        if results and results[0].boxes:
            for box in results[0].boxes:
                coords = box.xyxy[0].tolist()
                boxes_data.append({
                    "x1": int(coords[0]), "y1": int(coords[1]),
                    "x2": int(coords[2]), "y2": int(coords[3]),
                    "confidence": float(box.conf[0])
                })
        return boxes_data