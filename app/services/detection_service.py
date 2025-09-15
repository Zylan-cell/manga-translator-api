from typing import List, Dict, Any, Optional
from ultralytics import YOLO

from app.config import settings
from app.logger import app_logger
from app.utils.images import b64_to_pil
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

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

    def detect_batch(self, images: List[Image.Image], model_name: Optional[str] = None) -> List[List[Dict[str, Any]]]:
        """Run detection for a batch of PIL images. Returns list of boxes per image.

        If the underlying YOLO model supports batch inference, we call it with the list of images.
        Otherwise, we fallback to parallel per-image detection using ThreadPoolExecutor.
        """
        if not self.models:
            raise RuntimeError("Detection service not initialized.")

        active_model_name = model_name if model_name in self.models else settings.DEFAULT_DETECTION_MODEL
        model = self.models.get(active_model_name)
        if model is None:
            raise RuntimeError(f"Default detection model '{settings.DEFAULT_DETECTION_MODEL}' not found.")

        max_workers = getattr(settings, "BUBBLE_DETECT_MAX_WORKERS", 1)

        # Attempt native batch inference: model accepts a list of PIL images
        try:
            app_logger.info(f"Running batch detection on {len(images)} images with model: '{active_model_name}'")
            results = model(images, verbose=False)  # type: ignore[arg-type]
            batch_boxes: List[List[Dict[str, Any]]] = []
            for res in results:
                boxes_data: List[Dict[str, Any]] = []
                if res and getattr(res, 'boxes', None) is not None:
                    for box in res.boxes:
                        coords = box.xyxy[0].tolist()
                        boxes_data.append({
                            "x1": int(coords[0]),
                            "y1": int(coords[1]),
                            "x2": int(coords[2]),
                            "y2": int(coords[3]),
                            "confidence": float(box.conf[0]),
                        })
                batch_boxes.append(boxes_data)
            return batch_boxes
        except Exception:
            # If batch invocation fails, fallback to per-image detection (possible on CPU-only setups)
            app_logger.info("Batch inference failed, falling back to parallel per-image detection.")

        # Fallback: parallel per-image detection
        if not images:
            return []

        if max_workers is None or max_workers <= 1:
            # sequential
            return [self.detect_image_obj(img, model) for img in images]

        # Parallel per-image detection
        results_list: List[List[Dict[str, Any]]] = [[] for _ in images]
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(self.detect_image_obj, img, model): i for i, img in enumerate(images)}
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results_list[idx] = fut.result()
                except Exception as e:
                    app_logger.error(f"[Detection BATCH] Error processing image index {idx}: {e}", exc_info=True)
                    results_list[idx] = []

        return results_list

    def detect_image_obj(self, image: Image.Image, model: YOLO) -> List[Dict[str, Any]]:
        """Helper to run detection for a single PIL image using specified model instance."""
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