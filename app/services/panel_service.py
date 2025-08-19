import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import os
import warnings

from app.config import PANEL_MODEL_PATH
from app.logger import app_logger

warnings.filterwarnings("ignore")

class PanelOrderDetector:
    def __init__(self):
        # Путь к модели теперь берется из конфига
        if not os.path.isdir(PANEL_MODEL_PATH):
            raise FileNotFoundError(f"Model directory not found: {PANEL_MODEL_PATH}")

        app_logger.info("Initializing Panel Order Detector (Florence-2)...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        app_logger.info(f"Using device: {self.device}")

        self.model = AutoModelForCausalLM.from_pretrained(
            PANEL_MODEL_PATH,
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).to(self.device).eval()

        self.processor = AutoProcessor.from_pretrained(
            PANEL_MODEL_PATH,
            trust_remote_code=True
        )
        app_logger.info("Panel Detector is ready.")

    @torch.no_grad()
    def get_sorted_panels(self, image: Image.Image) -> list:
        image_rgb = image.convert("RGB")
        detections = self.model.predict_detections_and_associations([image_rgb], self.processor)
        
        if detections and 'panels' in detections[0]:
            return detections[0]['panels']
        return []