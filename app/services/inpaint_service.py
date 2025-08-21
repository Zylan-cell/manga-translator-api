from __future__ import annotations
from typing import List, Optional, Tuple
import io

from PIL import Image, ImageDraw, ImageFilter
from simple_lama_inpainting import SimpleLama

from app.logger import app_logger as log
from app.utils.images import b64_to_pil, b64_to_bytes, pil_to_b64_png

class InpaintService:
    def __init__(self) -> None:
        self.lama: Optional[SimpleLama] = None

    async def initialize(self) -> None:
        if self.lama is None:
            try:
                self.lama = SimpleLama()
                log.info("SimpleLaMa initialized")
            except Exception as e:
                log.error("Failed to initialize SimpleLaMa: %s", e, exc_info=True)
                self.lama = None  # явно

    def _ensure(self) -> None:
        if self.lama is None:
            raise RuntimeError("InpaintService not initialized")

    def inpaint_with_mask(self, image_b64: str, mask_b64: str) -> str:
        self._ensure()
        img = b64_to_pil(image_b64)
        mask = Image.open(io.BytesIO(b64_to_bytes(mask_b64))).convert("L")
        result = self.lama(img, mask)
        return pil_to_b64_png(result)

    def _draw_boxes_to_mask(self, size: Tuple[int, int], boxes: List[List[int]], dilate: int = 2) -> Image.Image:
        W, H = size
        mask = Image.new("L", (W, H), 0)
        drw = ImageDraw.Draw(mask)
        for (x1, y1, x2, y2) in boxes:
            drw.rectangle([x1, y1, x2, y2], fill=255)
        if dilate > 0:
            k = max(1, dilate * 2 + 1)
            mask = mask.filter(ImageFilter.MaxFilter(size=k))
        return mask

    def inpaint_auto_text(
        self,
        image_b64: str,
        boxes: Optional[List[List[int]]] = None,
        dilate: int = 2,
        return_mask: bool = False,
    ) -> Tuple[str, Optional[str]]:
        self._ensure()
        img = b64_to_pil(image_b64)
        W, H = img.size

        if not boxes:
            log.warning("Inpaint auto called with no boxes, returning original image.")
            return pil_to_b64_png(img), None

        mask = self._draw_boxes_to_mask((W, H), boxes, dilate=dilate)
        result = self.lama(img, mask)

        img_out = pil_to_b64_png(result)
        mask_out = pil_to_b64_png(mask) if return_mask else None
        return img_out, mask_out