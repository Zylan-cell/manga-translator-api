from __future__ import annotations
from typing import List, Optional
import re
import os

from PIL import Image
from manga_ocr import MangaOcr

from app.logger import app_logger as log
from app.utils.images import b64_to_pil
from app.config import settings

class OcrService:
    def __init__(self) -> None:
        self.ocr_manga: Optional[MangaOcr] = None

    async def initialize(self) -> None:
        # MangaOCR — ТОЛЬКО из локальной папки weights
        local_path = settings.MANGA_OCR_PATH
        
        # Проверяем существование локальной модели
        if not os.path.exists(local_path):
            raise RuntimeError(
                f"Local MangaOCR model not found at {local_path}. "
                f"Please download the model to weights/manga-ocr-base/ folder."
            )
        
        # Проверяем что это папка с необходимыми файлами
        config_file = os.path.join(local_path, "config.json")
        if not os.path.exists(config_file):
            raise RuntimeError(
                f"Invalid MangaOCR model folder at {local_path}. "
                f"Missing config.json. Please ensure the complete model is downloaded."
            )
            
        try:
            if self.ocr_manga is None:
                # Принудительно используем только локальную модель
                self.ocr_manga = MangaOcr(pretrained_model_name_or_path=local_path)
                log.info(f"✅ MangaOCR initialized successfully from LOCAL model: {local_path}")
                log.info(f"🎯 Using LOCAL MangaOCR model at: {os.path.abspath(local_path)}")
        except Exception as e:
            log.error(f"❌ Failed to initialize MangaOCR from local path {local_path}: {e}", exc_info=True)
            raise RuntimeError(
                f"Cannot initialize MangaOCR from local model at {local_path}. "
                f"Error: {e}. Please check the model files."
            )

    def _ensure_manga(self):
        if not self.ocr_manga:
            raise RuntimeError("MangaOCR not initialized")
    
    def get_model_info(self) -> dict:
        """Returns information about the loaded MangaOCR model to verify it's local."""
        if not self.ocr_manga:
            return {"status": "not_initialized"}
        
        # Получаем информацию о модели
        model_info = {
            "status": "initialized",
            "local_path": settings.MANGA_OCR_PATH,
            "local_path_exists": os.path.exists(settings.MANGA_OCR_PATH),
            "config_exists": os.path.exists(os.path.join(settings.MANGA_OCR_PATH, "config.json")),
        }
        
        # Проверяем атрибуты модели
        try:
            if hasattr(self.ocr_manga, 'model'):
                if hasattr(self.ocr_manga.model, 'config'):
                    config = self.ocr_manga.model.config
                    if hasattr(config, '_name_or_path'):
                        model_info["model_name_or_path"] = config._name_or_path
                    if hasattr(config, 'model_type'):
                        model_info["model_type"] = config.model_type
        except Exception as e:
            model_info["model_info_error"] = str(e)
        
        return model_info

    def _manga_recognize(self, img: Image.Image) -> str:
        self._ensure_manga()
        # Логируем подтверждение что используется локальная модель
        log.debug(f"🎯 Processing OCR with LOCAL MangaOCR from: {settings.MANGA_OCR_PATH}")
        return self.ocr_manga(img)  # type: ignore

    def recognize(
        self,
        image_b64: str,
        engine: str = "manga",         # Only "manga" supported now
        langs: Optional[List[str]] = None,  # Unused, kept for API compatibility
        auto_rotate: bool = False  # Unused, kept for API compatibility
    ) -> str:
        """Recognize text using MangaOCR only."""
        img = b64_to_pil(image_b64)
        return self._manga_recognize(img)

    def recognize_batch(
        self,
        images_b64: List[str],
        engine: str = "manga",
        langs: Optional[List[str]] = None,
        auto_rotate: bool = False
    ) -> List[str]:
        """Batch recognition using MangaOCR only."""
        return [self.recognize(b, engine=engine, langs=langs, auto_rotate=auto_rotate) for b in images_b64]