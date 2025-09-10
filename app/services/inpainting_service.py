# app/services/inpainting_service.py
from __future__ import annotations

import os
import io
import base64
from typing import Optional, List, Tuple, Any

import numpy as np
from PIL import Image, ImageDraw
import cv2
import torch

# Реальная LaMa (умеет .ckpt)
try:
    from lama_cleaner.model.lama import LaMa
    from lama_cleaner.schema import Config as LamaConfig, HDStrategy as LamaHDStrategy
except Exception:
    LaMa = None            # type: ignore[assignment]
    LamaConfig = None      # type: ignore[assignment]
    LamaHDStrategy = None  # type: ignore[assignment]

from app.logger import app_logger as app_logger

# Кросс-совместимый ресемплинг для Pillow
try:
    from PIL.Image import Resampling as PILResampling  # Pillow >= 9.1
    RESAMPLE_NEAREST = PILResampling.NEAREST
except Exception:
    # Pillow < 9.1 — fallback
    RESAMPLE_NEAREST = getattr(Image, "NEAREST", 0)


class InpaintingService:
    """
    LaMa-инпейнт с:
      - ROI-кропом вокруг маски (PAD),
      - защитой краёв (EDGE_BAND),
      - апскейлом ROI (UPSCALE),
      - мягким смешиванием по границе (FEATHER).
    Использует вес weights/lama_large_512px.ckpt (если найден).
    """

    def __init__(self):
        self.device: Optional[str] = None
        self.model_path: Optional[str] = None
        self.lama: Any = None  # избегаем жёсткой типизации для совместимости с Pylance

        # Тюнинг по умолчанию (можно вынести в конфиг)
        self.PAD: int = 16
        self.EDGE_BAND: int = 0 # временно выключим защиту краёв
        self.UPSCALE: float = 2.0
        self.FEATHER: float = 1.2
        self.DILATE_BEFORE: int = 2 # дилатация маски перед LaMa (px)
        self.MIN_DIFF_TRIGGER: float = 0.5 # порог “почти нет изменений” в среднем по ROI

    async def initialize(self):
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            app_logger.info(f"🎨 Inpainting service using device: {self.device}")

            # Ищем приоритетно lama_large_512px.ckpt
            weights_dir = "weights"
            candidates = [
                "lama_large_512px.ckpt",
                "lama_large.ckpt",
                "lama.ckpt",
            ]
            for name in candidates:
                p = os.path.join(weights_dir, name)
                if os.path.exists(p):
                    self.model_path = p
                    app_logger.info(f"🎨 Found LaMa ckpt: {os.path.abspath(p)}")
                    break

            if not self.model_path or LaMa is None:
                app_logger.error("🎨 LaMa not available (no ckpt or no lama-cleaner). Inpainting unavailable.")
                self.lama = None
                return

            # Загружаем LaMa через lama-cleaner
            self.lama = LaMa(device=self.device, model_path=self.model_path)
            app_logger.info("🎨 LaMa loaded (lama-cleaner) with provided ckpt")
        except Exception as e:
            app_logger.error(f"🎨 Failed to initialize LaMa: {e}", exc_info=True)
            self.lama = None

    def is_available(self) -> bool:
        return self.lama is not None

    # ----------------- Вспомогательные -----------------

    def _decode_base64_image(self, image_data: str) -> Image.Image:
        try:
            if "," in image_data:
                image_data = image_data.split(",", 1)[1]
            raw = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(raw))
            return img.convert("RGB")
        except Exception as e:
            app_logger.error(f"🎨 Failed to decode image: {e}")
            raise ValueError(f"Invalid image data: {e}")

    def _encode_image_to_base64(self, image: Image.Image) -> str:
        try:
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            buf.seek(0)
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as e:
            app_logger.error(f"🎨 Failed to encode image: {e}")
            raise ValueError(f"Failed to encode image: {e}")

    @staticmethod
    def _binarize_mask(mask_img: Image.Image) -> np.ndarray:
        # ч/б 0/255; белое = область инпейнта
        m = np.array(mask_img.convert("L"))
        _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
        return m

    @staticmethod
    def _bbox_from_mask(m: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        ys, xs = np.where(m > 0)
        if len(xs) == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

    @staticmethod
    def _protect_edges(rgb_roi: np.ndarray, mask_roi: np.ndarray, edge_band: int) -> np.ndarray:
        # Убираем маску возле контуров (Canny + dilate), чтобы не поплыл контур бабла
        gray = cv2.cvtColor(rgb_roi, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        if edge_band > 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_band, edge_band))
            edges = cv2.dilate(edges, k, iterations=1)
        safe = mask_roi.copy()
        safe[edges > 0] = 0
        return safe

    @staticmethod
    def _soft_alpha(mask_roi: np.ndarray, feather: float) -> np.ndarray:
        # Из 0/255 делаем мягкую альфу (GaussianBlur). HxWx1 float32 [0..1]
        a = (mask_roi.astype(np.float32) / 255.0)
        if feather > 0:
            a = cv2.GaussianBlur(a, (0, 0), sigmaX=feather, sigmaY=feather)
        return np.clip(a, 0.0, 1.0)[..., None]

    def _lama_infer(self, rgb_up: np.ndarray, mask_up: np.ndarray, invert_mask: bool = False) -> np.ndarray:
        lama = self.lama
        if lama is None:
            raise RuntimeError("LaMa model is not loaded")
        img_np = np.asarray(rgb_up, dtype=np.uint8)
        if img_np.ndim != 3 or img_np.shape[2] != 3:
            raise ValueError(f"Invalid image array shape: {img_np.shape} (expected HxWx3)")

        m_np = np.asarray(mask_up, dtype=np.uint8)
        if m_np.ndim == 3:
            m_np = cv2.cvtColor(m_np, cv2.COLOR_RGB2GRAY)
        _, m_np = cv2.threshold(m_np, 127, 255, cv2.THRESH_BINARY)

        if invert_mask:
            m_np = 255 - m_np

        # Конфиг под твою версию lama-cleaner (обязательные поля у Pydantic)
        if LamaConfig is not None and LamaHDStrategy is not None:
            infer_cfg = LamaConfig(  # type: ignore[call-arg]
                hd_strategy=LamaHDStrategy.CROP,        # ← CROP вместо ORIGINAL
                ldm_steps=20,
                hd_strategy_crop_margin=32,
                hd_strategy_crop_trigger_size=320,      # триггерим кроп даже на средних ROI
                hd_strategy_resize_limit=1536,          # лимит ресайза
            )
        else:
            infer_cfg = {
                "hd_strategy": "crop",
                "ldm_steps": 20,
                "hd_strategy_crop_margin": 32,
                "hd_strategy_crop_trigger_size": 320,
                "hd_strategy_resize_limit": 1536,
            }

        res_up = lama(img_np, m_np, config=infer_cfg)
        if isinstance(res_up, Image.Image):
            res_up = np.array(res_up)
        if not isinstance(res_up, np.ndarray) or res_up.ndim != 3:
            raise ValueError(f"Unexpected LaMa output type/shape: {type(res_up)} {getattr(res_up, 'shape', None)}")

        return res_up.astype(np.uint8)

    # ----------------- Основная логика -----------------

    async def _run_inpainting(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """
        ROI + (опц.) защита краёв + (опц.) дилатация маски + апскейл ROI + LaMa + (опц.) dering + мягкое смешивание.
        Белое на маске = область инпейнта.
        """
        # 0) В numpy
        img = np.array(image)  # RGB, uint8
        m = self._binarize_mask(mask)  # 0/255, uint8

        H, W = m.shape
        bbox = self._bbox_from_mask(m)
        if not bbox:
            app_logger.info("🎨 Mask empty — return original")
            return image

        # 1) ROI с паддингом
        x1, y1, x2, y2 = bbox
        pad = int(getattr(self, "PAD", 16))
        x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
        x2 = min(W, x2 + pad); y2 = min(H, y2 + pad)

        if x2 <= x1 or y2 <= y1:
            app_logger.info("🎨 Degenerate ROI — return original")
            return image

        rgb_roi = img[y1:y2, x1:x2].copy()
        mask_roi = m[y1:y2, x1:x2].copy()

        # 2) Защита краёв (границ) — убираем маску вдоль контуров (если включена)
        edge_band = int(getattr(self, "EDGE_BAND", 0))
        mask_safe = self._protect_edges(rgb_roi, mask_roi, edge_band)
        coverage = float(mask_safe.sum()) / (255.0 * mask_safe.size + 1e-6)
        app_logger.info(f"🎨 Mask coverage after edge-protect (band={edge_band}): {coverage*100:.2f}%")

        if mask_safe.max() == 0:
            app_logger.info("🎨 Mask became empty after edge protect — return original")
            return image

        # 3) (Опционально) Небольшая дилатация маски, чтобы гарантировать влияние
        dil = int(getattr(self, "DILATE_BEFORE", 2))
        if dil > 0:
            ksz = max(1, 2 * dil + 1)  # радиус → нечётный размер ядра
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
            mask_safe = cv2.dilate(mask_safe, kernel, iterations=1)
            coverage2 = float(mask_safe.sum()) / (255.0 * mask_safe.size + 1e-6)
            app_logger.info(f"🎨 Mask coverage after dilate({dil}): {coverage2*100:.2f}%")

        # 4) Апскейл ROI (для лучшего качества восстановления деталей)
        scale = float(getattr(self, "UPSCALE", 2.0) or 1.0)
        if scale > 1.0:
            nh, nw = max(1, int((y2 - y1) * scale)), max(1, int((x2 - x1) * scale))
            rgb_up = cv2.resize(rgb_roi, (nw, nh), interpolation=cv2.INTER_CUBIC)
            mask_up = cv2.resize(mask_safe, (nw, nh), interpolation=cv2.INTER_NEAREST)
        else:
            rgb_up, mask_up = rgb_roi, mask_safe

        # 5) Первый прогон LaMa (как есть)
        res_up = self._lama_infer(rgb_up, mask_up, invert_mask=False)

        # Привести к размеру ROI
        if res_up.shape[:2] != rgb_roi.shape[:2]:
            res_roi = cv2.resize(res_up, (rgb_roi.shape[1], rgb_roi.shape[0]), interpolation=cv2.INTER_CUBIC)
            mask_safe = cv2.resize(mask_safe, (rgb_roi.shape[1], rgb_roi.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            res_roi = res_up

        # 6) (Опционально) лёгкий dering внутри ROI, чтобы сгладить полосы
        DERING = True
        if DERING:
            # bilateral лучше median сохраняет локальные структуры
            res_roi = cv2.bilateralFilter(res_roi, d=5, sigmaColor=12, sigmaSpace=12)

        # Оценка изменения в ROI — средняя |разница|
        delta1 = float(np.mean(np.abs(res_roi.astype(np.float32) - rgb_roi.astype(np.float32))))
        app_logger.info(f"🎨 Delta after first pass: {delta1:.3f}")

        # 7) При «почти нулевом» эффекте — один раз пробуем инвертировать маску
        min_diff = float(getattr(self, "MIN_DIFF_TRIGGER", 0.5))
        coverage_thr = 0.0005  # минимальное покрытие, чтобы имело смысл переезжать
        cur_coverage = float(mask_safe.sum()) / (255.0 * mask_safe.size + 1e-6)

        if delta1 < min_diff and cur_coverage > coverage_thr:
            app_logger.info("🎨 Low delta — try inverted mask once")
            res_up_inv = self._lama_infer(rgb_up, mask_up, invert_mask=True)
            if res_up_inv.shape[:2] != rgb_roi.shape[:2]:
                res_roi_inv = cv2.resize(res_up_inv, (rgb_roi.shape[1], rgb_roi.shape[0]), interpolation=cv2.INTER_CUBIC)
            else:
                res_roi_inv = res_up_inv

            if DERING:
                res_roi_inv = cv2.bilateralFilter(res_roi_inv, d=5, sigmaColor=12, sigmaSpace=12)

            delta2 = float(np.mean(np.abs(res_roi_inv.astype(np.float32) - rgb_roi.astype(np.float32))))
            app_logger.info(f"🎨 Delta after inverted pass: {delta2:.3f}")
            # Берём тот результат, который больше изменил область (эвристика)
            if delta2 > delta1:
                res_roi = res_roi_inv

        # 8) Мягкое смешивание только по маске (перо)
        feather = float(getattr(self, "FEATHER", 1.2))
        alpha = self._soft_alpha(mask_safe, feather)  # HxWx1 float32 [0..1]

        base = img.copy().astype(np.float32)
        region = base[y1:y2, x1:x2]
        blended = (alpha * res_roi.astype(np.float32) + (1.0 - alpha) * region).astype(np.uint8)
        base[y1:y2, x1:x2] = blended

        return Image.fromarray(base.astype(np.uint8), mode="RGB")

    # ----------------- Публичные методы API -----------------

    async def inpaint_with_mask(
        self,
        image_data: str,
        mask_data: str,
        model: Optional[str] = None
    ) -> str:
        if not self.is_available():
            raise RuntimeError("Inpainting service not available")
        try:
            app_logger.info("🎨 LaMa inpainting (custom mask)")
            image = self._decode_base64_image(image_data)
            mask = self._decode_base64_image(mask_data)

            # Маску к размеру изображения (без размывания)
            if image.size != mask.size:
                mask = mask.resize(image.size, RESAMPLE_NEAREST)

            result = await self._run_inpainting(image, mask)
            return self._encode_image_to_base64(result)
        except Exception as e:
            app_logger.error(f"🎨 Inpainting failed: {e}", exc_info=True)
            raise

    async def inpaint_auto_text(
        self,
        image_data: str,
        boxes: List[List[int]],
        dilate: int = 2
    ) -> str:
        if not self.is_available():
            raise RuntimeError("Inpainting service not available")
        try:
            app_logger.info(f"🎨 Auto text inpainting ({len(boxes)} boxes, dilate={dilate})")
            image = self._decode_base64_image(image_data)
            W, H = image.size

            # Маска из боксов (белое = inpaint)
            mask_img = Image.new("L", (W, H), 0)
            draw = ImageDraw.Draw(mask_img)
            for b in boxes:
                x1, y1, x2, y2 = map(int, b)
                x1 = max(0, x1 - dilate); y1 = max(0, y1 - dilate)
                x2 = min(W, x2 + dilate); y2 = min(H, y2 + dilate)
                draw.rectangle([x1, y1, x2, y2], fill=255)

            result = await self._run_inpainting(image, mask_img)
            return self._encode_image_to_base64(result)
        except Exception as e:
            app_logger.error(f"🎨 Auto text inpainting failed: {e}", exc_info=True)
            raise

    def get_model_info(self) -> dict:
        return {
            "available": self.is_available(),
            "device": self.device,
            "ckpt": self.model_path,
            "engine": "lama-cleaner/saicinpainting" if self.lama else "none",
            "pad": self.PAD,
            "edge_band": self.EDGE_BAND,
            "upscale": self.UPSCALE,
            "feather": self.FEATHER,
        }