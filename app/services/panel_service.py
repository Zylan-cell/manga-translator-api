import os
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

from app.config import settings
from app.logger import app_logger

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