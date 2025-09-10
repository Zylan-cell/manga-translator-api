from typing import List, Dict, Any, Optional
from ultralytics import YOLO

from app.config import settings
from app.logger import app_logger
from app.utils.images import b64_to_pil

class DetectionService:
    def __init__(self):
        # ИЗМЕНЕНО: Храним несколько моделей в словаре
        self.models: Dict[str, YOLO] = {}

    async def initialize(self):
        # ИЗМЕНЕНО: Загружаем все модели, указанные в конфиге
        if not self.models:
            app_logger.info("Loading detection models...")
            for name, path in settings.DETECTION_MODELS.items():
                app_logger.info(f" -> Loading model '{name}' from {path}...")
                self.models[name] = YOLO(path)
            app_logger.info("All detection models loaded.")

    def detect(self, image_b64: str, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        # ИЗМЕНЕНО: Выбираем модель по имени
        if not self.models:
            raise RuntimeError("Detection service not initialized.")

        # Используем модель по умолчанию, если имя не указано или не найдено
        active_model_name = model_name if model_name in self.models else settings.DEFAULT_DETECTION_MODEL
        model = self.models.get(active_model_name)
        
        if model is None:
            # Этого не должно произойти, если конфиг правильный
            raise RuntimeError(f"Default detection model '{settings.DEFAULT_DETECTION_MODEL}' not found.")

        app_logger.info(f"Running detection with model: '{active_model_name}'")
        
        image = b64_to_pil(image_b64)
        results = model(image, verbose=False)

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