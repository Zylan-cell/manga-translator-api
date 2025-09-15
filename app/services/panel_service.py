import os
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

from app.config import settings
from app.logger import app_logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

class PanelOrderDetector:
    def __init__(self):
        if not os.path.isdir(settings.PANEL_MODEL_PATH):
            raise FileNotFoundError(f"Model directory not found: {settings.PANEL_MODEL_PATH}")

        app_logger.info("Initializing Panel Order Detector (Florence-2)...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        app_logger.info(f"Using device: {self.device} (dtype={dtype})")

        self.model = AutoModelForCausalLM.from_pretrained(
            settings.PANEL_MODEL_PATH,
            torch_dtype=dtype,
            trust_remote_code=True
        ).to(self.device).eval()

        self.processor = AutoProcessor.from_pretrained(
            settings.PANEL_MODEL_PATH,
            trust_remote_code=True
        )
        app_logger.info("Panel Detector is ready.")

    @torch.no_grad()
    def get_sorted_panels(self, image: Image.Image) -> list:
        image_rgb = image.convert("RGB")
        detections = self.model.predict_detections_and_associations([image_rgb], self.processor)
        if isinstance(detections, list) and detections and 'panels' in detections[0]:
            return [list(map(int, p)) for p in detections[0]['panels']]
        return []

    @torch.no_grad()
    def get_sorted_panels_batch(self, images: list) -> list:
        """Accepts a list of PIL.Image and returns a list of panel lists for each image.

        The underlying model supports batch prediction via predict_detections_and_associations,
        so this method just forwards the list to the model and formats results.
        """
        # Ensure images are RGB
        images_rgb = [img.convert("RGB") for img in images]
        max_workers = getattr(settings, "PANEL_MAX_WORKERS", 1)

        # Try native batch inference first
        try:
            detections = self.model.predict_detections_and_associations(images_rgb, self.processor)
            results = []
            if isinstance(detections, list):
                for det in detections:
                    if det and 'panels' in det:
                        results.append([list(map(int, p)) for p in det['panels']])
                    else:
                        results.append([])
            return results
        except Exception:
            app_logger.info("Panel batch inference failed, falling back to per-image processing.")

        # Fallback: parallel per-image processing
        if not images_rgb:
            return []

        if max_workers is None or max_workers <= 1:
            return [self.get_sorted_panels(img) for img in images_rgb]

        results: list = [None] * len(images_rgb)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(self.get_sorted_panels, img): i for i, img in enumerate(images_rgb)}
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:
                    app_logger.error(f"[Panel BATCH] Error processing image index {idx}: {e}", exc_info=True)
                    results[idx] = []

        return results