from __future__ import annotations
import base64
from io import BytesIO
from typing import List, Optional, Tuple

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter
import torch
from simple_lama_inpainting import SimpleLama

from app.logger import app_logger as log

def b64_to_pil(b64: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")

def pil_to_b64_png(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

class InpaintService:
    def __init__(self) -> None:
        self.lama: Optional[SimpleLama] = None
        # CRAFT больше не нужен
        # self.craft: Optional[Craft] = None

    async def initialize(self) -> None:
        if self.lama is None:
            self.lama = SimpleLama()
            log.info("SimpleLaMa initialized")

    def _ensure(self) -> None:
        if self.lama is None:
            raise RuntimeError("InpaintService not initialized")

    def inpaint_with_mask(self, image_b64: str, mask_b64: str) -> str:
        self._ensure()
        img = b64_to_pil(image_b64)
        mask = Image.open(BytesIO(base64.b64decode(mask_b64))).convert("L")
        result = self.lama(img, mask)
        return pil_to_b64_png(result)

    def _draw_boxes_to_mask(
        self, size: Tuple[int, int], boxes: List[List[int]], dilate: int = 2
    ) -> Image.Image:
        """Рисует прямоугольные рамки на маске."""
        W, H = size
        mask = Image.new("L", (W, H), 0)
        drw = ImageDraw.Draw(mask)
        for (x1, y1, x2, y2) in boxes:
            drw.rectangle([x1, y1, x2, y2], fill=255)
        
        if dilate > 0:
            # Дилатация, чтобы маска была чуть больше рамок
            k = max(1, dilate * 2 + 1)
            mask = mask.filter(ImageFilter.MaxFilter(size=k))
        return mask

    def inpaint_auto_text(
        self,
        image_b64: str,
        boxes: Optional[List[List[int]]] = None,
        dilate: int = 2, 
        return_mask: bool = False
    ) -> Tuple[str, Optional[str]]:
        """
        Упрощенная функция инпеинтинга. Использует предоставленные
        прямоугольные рамки (от YOLO) для создания маски.
        """
        self._ensure()
        img = b64_to_pil(image_b64)
        W, H = img.size

        if not boxes:
            log.warning("Inpaint auto called with no boxes, returning original image.")
            return image_b64, None

        # Создаем маску напрямую из рамок пузырей
        mask = self._draw_boxes_to_mask((W, H), boxes, dilate=dilate)
        
        # Закрашиваем
        result = self.lama(img, mask)
        
        img_out = pil_to_b64_png(result)
        mask_out = pil_to_b64_png(mask) if return_mask else None
        return img_out, mask_out