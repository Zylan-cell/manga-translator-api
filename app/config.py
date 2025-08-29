import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = BASE_DIR / "weights"

class Settings:
    # Пути к моделям (все из ./weights)
    YOLO_MODEL_PATH: str = os.getenv("YOLO_MODEL_PATH", str(WEIGHTS_DIR / "bubbles_yolo.pt"))
    PANEL_MODEL_PATH: str = os.getenv("PANEL_MODEL_PATH", str(WEIGHTS_DIR / "magiv3"))
    # Локальная папка модели MangaOCR (ОБЯЗАТЕЛЬНО должна существовать в weights/)
    # Структура: weights/manga-ocr-base/config.json, model.safetensors, tokenizer.json и т.д.
    MANGA_OCR_PATH: str = os.getenv("MANGA_OCR_PATH", str(WEIGHTS_DIR / "manga-ocr-base"))

    # Настройки сервера
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", 8000))

    # URL для LM Studio (OpenAI-совместимый сервер)
    LM_STUDIO_URL: str = os.getenv("LM_STUDIO_URL", "http://localhost:1234")

settings = Settings()

# Валидация на старте: проверяем что локальная модель существует
def validate_local_model():
    """Validates that the local MangaOCR model exists and is properly configured."""
    manga_path = settings.MANGA_OCR_PATH
    
    if not os.path.exists(manga_path):
        raise RuntimeError(
            f"❌ Local MangaOCR model not found at: {manga_path}\n"
            f"Please download the model and place it in the weights/manga-ocr-base/ folder.\n"
            f"Expected structure:\n"
            f"  weights/manga-ocr-base/\n"
            f"    ├── config.json\n"
            f"    ├── model.safetensors (or pytorch_model.bin)\n"
            f"    ├── tokenizer.json\n"
            f"    └── ... (other files)"
        )
    
    config_path = os.path.join(manga_path, "config.json")
    if not os.path.exists(config_path):
        raise RuntimeError(
            f"❌ Invalid MangaOCR model at: {manga_path}\n"
            f"Missing config.json file. Please ensure you have the complete model."
        )
    
    print(f"✅ Local MangaOCR model validated at: {os.path.abspath(manga_path)}")
    return True

# Выполняем валидацию при импорте модуля
validate_local_model()