from __future__ import annotations
from typing import List, Optional
import os

from PIL import Image
from manga_ocr import MangaOcr

from app.logger import app_logger as log
from app.utils.images import b64_to_pil
from app.config import settings

class OcrService:
    def __init__(self) -> None:
        self.ocr: Optional[MangaOcr] = None

    async def initialize(self) -> None:
        local_path = settings.MANGA_OCR_PATH
        if os.path.isdir(local_path):
            log.info(f"Initializing MangaOCR from local path: {local_path}")
            self.ocr = MangaOcr(pretrained_model_name_or_path=local_path)
        else:
            log.warning(
                f"Local MangaOCR model not found at {local_path}. Falling back to default (may try to download)."
            )
            self.ocr = MangaOcr()
        log.info("MangaOCR initialized")

    def recognize(self, image_b64: str) -> str:
        if not self.ocr:
            raise RuntimeError("OCR not initialized")
        img: Image.Image = b64_to_pil(image_b64)
        return self.ocr(img)

    def recognize_batch(self, images_b64: List[str]) -> List[str]:
        if not self.ocr:
            raise RuntimeError("OCR not initialized")
        return [self.recognize(b) for b in images_b64]