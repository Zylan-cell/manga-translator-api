from __future__ import annotations
from typing import List, Optional, Tuple
import re
import os

import numpy as np
import onnxruntime as ort
from PIL import Image
from manga_ocr import MangaOcr

from app.logger import app_logger as log
from app.utils.images import b64_to_pil
from app.config import settings

class OcrService:
    def __init__(self) -> None:
        self.ocr_manga: Optional[MangaOcr] = None
        self.easy_reader = None       # lazy
        self.rapid_engine = None      # lazy

    async def initialize(self) -> None:
        # MangaOCR — из локальной папки, если есть
        local_path = settings.MANGA_OCR_PATH
        try:
            if self.ocr_manga is None:
                self.ocr_manga = MangaOcr(pretrained_model_name_or_path=local_path)
                log.info("MangaOCR initialized")
        except Exception as e:
            log.error("Failed to init MangaOCR from %s: %s", local_path, e, exc_info=True)
            self.ocr_manga = MangaOcr()
            log.info("MangaOCR initialized (fallback)")

    def _ensure_manga(self):
        if not self.ocr_manga:
            raise RuntimeError("MangaOCR not initialized")

    def _ensure_easy(self, langs: Optional[List[str]]):
        if self.easy_reader is None:
            try:
                import easyocr  # type: ignore
            except Exception as e:
                raise RuntimeError(f"easyocr not installed: {e}")
            lang_list = langs or ["ja", "en"]
            model_dir = settings.EASYOCR_MODEL_DIR
            self.easy_reader = easyocr.Reader(
                lang_list,
                gpu=True,
                model_storage_directory=model_dir,
                download_enabled=True,
            )
            log.info("EasyOCR initialized with langs=%s, dir=%s", lang_list, model_dir)

    def _ensure_rapid(self):
        """
        Инициализация RapidOCR (EN-only).
        Требуемые локальные файлы:
        - ch_PP-OCRv3_det_infer.onnx                 (детектор)
        - ch_ppocr_mobile_v2.0_cls_infer.onnx        (классификатор)
        - en_PP-OCRv3_rec_infer.onnx                 (EN распознаватель, у него высота 48)
        Если файлов нет/перепутаны — fallback: RapidOCR() сам скачает в свой кэш.
        """
        if self.rapid_engine is not None:
            return

        import os
        import numpy as np
        import onnxruntime as ort
        from app.config import settings
        from app.logger import app_logger as log

        det = settings.RAPID_DET_MODEL_PATH
        cls = settings.RAPID_CLS_MODEL_PATH
        rec = settings.RAPID_REC_MODEL_PATH

        def file_ok(p: str) -> bool:
            return bool(p) and os.path.isfile(p)

        def input_shape(path: str):
            try:
                sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
                shp = sess.get_inputs()[0].shape
                return [int(x) if isinstance(x, (int, np.integer)) else str(x) for x in shp]
            except Exception:
                return None

        def has_48(shape) -> bool:
            if not shape:
                return False
            return 48 in [x for x in shape if isinstance(x, int)]

        need_fallback = False
        reason = ""

        # наличие файлов
        if not (file_ok(det) and file_ok(cls) and file_ok(rec)):
            need_fallback = True
            reason = "missing local weights"

        # проверяем формы
        det_shape = input_shape(det) if not need_fallback else None
        rec_shape = input_shape(rec) if not need_fallback else None

        # у rec должен быть 48, у det — не должен
        if not need_fallback:
            if det_shape and has_48(det_shape):
                need_fallback = True
                reason = f"RAPID_DET_MODEL_PATH points to a recognition model (has 48 in shape {det_shape})."
            elif rec_shape and not has_48(rec_shape):
                log.warning(
                    "RAPID_REC_MODEL_PATH doesn't look like a recognition model (no 48 in input shape %s). "
                    "Make sure to use en_PP-OCRv3_rec_infer.onnx",
                    rec_shape,
                )

        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore
            if need_fallback:
                log.warning(
                    "RapidOCR local weights %s. Falling back to builtin downloader/cache.",
                    reason or "(invalid configuration)"
                )
                self.rapid_engine = RapidOCR()  # скачает в свой кэш пользователя
                return

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.rapid_engine = RapidOCR(
                det_model_path=det,
                cls_model_path=cls,
                rec_model_path=rec,
                providers=providers,
            )
            log.info("RapidOCR initialized (EN rec). det=%s, rec=%s", det, rec)
        except Exception as e:
            log.error("RapidOCR init error (%s). Falling back to builtin downloader.", e, exc_info=True)
            from rapidocr_onnxruntime import RapidOCR  # type: ignore
            self.rapid_engine = RapidOCR()



    @staticmethod
    def _rotate(img: Image.Image, k: int) -> Image.Image:
        if k == 1:
            return img.transpose(Image.ROTATE_90)
        if k == 2:
            return img.transpose(Image.ROTATE_270)
        return img

    @staticmethod
    def _score_text(s: str) -> int:
        # Японские + латиница/цифры — выше вес
        jp = len(re.findall(r'[\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uFF66-\uFF9D]', s))
        alnum = len(re.findall(r'[A-Za-z0-9]', s))
        return jp * 2 + alnum

    def _manga_recognize(self, img: Image.Image) -> str:
        self._ensure_manga()
        return self.ocr_manga(img)  # type: ignore

    def _easy_recognize(self, img: Image.Image, langs: Optional[List[str]]) -> str:
        self._ensure_easy(langs)
        arr = np.array(img)
        lines = self.easy_reader.readtext(arr, detail=0, paragraph=False)  # type: ignore
        return "\n".join(map(str, lines)) if isinstance(lines, list) else str(lines)

    def _rapid_recognize(self, img: Image.Image) -> str:
        """
        Возвращаем «одну строку» (для латиницы/EN в манге удобнее),
        т.е. объединяем строковые фрагменты через пробел.
        """
        self._ensure_rapid()
        import numpy as np

        # RapidOCR ждёт BGR
        arr = np.array(img)[:, :, ::-1].copy()
        try:
            result, _ = self.rapid_engine(arr)  # type: ignore
        except Exception as e:
            raise RuntimeError(f"RapidOCR inference failed: {e}")

        if not result:
            return ""

        # сортируем сверху-вниз, слева-направо
        def key_fn(item):
            box = item[0]
            ys = [pt[1] for pt in box]
            xs = [pt[0] for pt in box]
            return (min(ys), min(xs))

        result.sort(key=key_fn)
        # «одной строкой» — схлопываем переносы
        tokens = [x[1] for x in result if isinstance(x, (list, tuple)) and len(x) >= 2]
        return " ".join(tokens)

    def recognize(
        self,
        image_b64: str,
        engine: str = "manga",         # "manga" | "easy" | "rapid" | "auto"
        langs: Optional[List[str]] = None,
        auto_rotate: bool = True
    ) -> str:
        img = b64_to_pil(image_b64)
        rotations = [0, 1, 2] if auto_rotate else [0]

        candidates: List[Tuple[str, int]] = []
        if engine == "easy":
            for k in rotations:
                s = self._easy_recognize(self._rotate(img, k), langs)
                candidates.append((s, self._score_text(s)))
        elif engine == "rapid":
            for k in rotations:
                s = self._rapid_recognize(self._rotate(img, k))
                candidates.append((s, self._score_text(s)))
        elif engine == "auto":
            # Соревнуются MangaOCR и RapidOCR
            for k in rotations:
                s1 = self._manga_recognize(self._rotate(img, k))
                candidates.append((s1, self._score_text(s1)))
                s2 = self._rapid_recognize(self._rotate(img, k))
                candidates.append((s2, self._score_text(s2)))
        else:
            for k in rotations:
                s = self._manga_recognize(self._rotate(img, k))
                candidates.append((s, self._score_text(s)))

        best = max(candidates, key=lambda t: (t[1], len(t[0]))) if candidates else ("", 0)
        return best[0]

    def recognize_batch(
        self,
        images_b64: List[str],
        engine: str = "manga",
        langs: Optional[List[str]] = None,
        auto_rotate: bool = True
    ) -> List[str]:
        return [self.recognize(b, engine=engine, langs=langs, auto_rotate=auto_rotate) for b in images_b64]