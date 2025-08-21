import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = BASE_DIR / "weights"

class Settings:
    # Пути к моделям (все из ./weights)
    YOLO_MODEL_PATH: str = os.getenv("YOLO_MODEL_PATH", str(WEIGHTS_DIR / "bubbles_yolo.pt"))
    PANEL_MODEL_PATH: str = os.getenv("PANEL_MODEL_PATH", str(WEIGHTS_DIR / "magiv3"))
    # Локальная папка модели MangaOCR (скачайте с HF и положите сюда)
    MANGA_OCR_PATH: str = os.getenv("MANGA_OCR_PATH", str(WEIGHTS_DIR / "manga-ocr-base"))

    # Настройки сервера
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", 8000))

    # URL для LM Studio (OpenAI-совместимый сервер)
    LM_STUDIO_URL: str = os.getenv("LM_STUDIO_URL", "http://localhost:1234")

settings = Settings()