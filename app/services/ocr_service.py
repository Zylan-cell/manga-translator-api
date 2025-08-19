from __future__ import annotations
import string
from typing import List, Optional
import base64
from io import BytesIO

from PIL import Image
from manga_ocr import MangaOcr

from app.logger import app_logger as log


def b64_to_pil(b64: string) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")


class OcrService:
    def __init__(self) -> None:
        self.ocr: Optional[MangaOcr] = None

    async def initialize(self) -> None:
        # Torch 2.8 разрешает .bin, поэтому достаточно простой инициализации:
        self.ocr = MangaOcr()
        log.info("MangaOCR initialized")

    def recognize(self, image_b64: str) -> str:
        if not self.ocr:
            raise RuntimeError("OCR not initialized")
        img = b64_to_pil(image_b64)
        return self.ocr(img)

    def recognize_batch(self, images_b64: List[str]) -> List[str]:
        if not self.ocr:
            raise RuntimeError("OCR not initialized")
        return [self.recognize(b) for b in images_b64]