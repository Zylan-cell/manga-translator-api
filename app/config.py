import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = BASE_DIR / "weights"

class Settings:
    # ИЗМЕНЕНО: Словарь с доступными моделями детекции
    # Ключ - это ID, который будет использоваться в API и UI
    # Значение - путь к файлу модели
    DETECTION_MODELS = {
        "bubbles_yolo": str(WEIGHTS_DIR / "bubbles_yolo.pt"),
        "comic_text_segmenter": str(WEIGHTS_DIR / "comic-text-segmenter.pt"),
    }
    DEFAULT_DETECTION_MODEL = "bubbles_yolo"

    # Пути к остальным моделям
    PANEL_MODEL_PATH: str = os.getenv("PANEL_MODEL_PATH", str(WEIGHTS_DIR / "magiv3"))
    MANGA_OCR_PATH: str = os.getenv("MANGA_OCR_PATH", str(WEIGHTS_DIR / "manga-ocr-base"))

    # Настройки сервера
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", 8000))

    # URL для LM Studio (OpenAI-совместимый сервер)
    LM_STUDIO_URL: str = os.getenv("LM_STUDIO_URL", "http://localhost:1234")

settings = Settings()

def validate_models():
    """Проверяет, что все необходимые файлы моделей существуют."""
    # Проверка MangaOCR
    manga_path = settings.MANGA_OCR_PATH
    if not os.path.exists(manga_path) or not os.path.exists(os.path.join(manga_path, "config.json")):
        raise RuntimeError(f"❌ Модель MangaOCR не найдена или неполная по пути: {manga_path}")
    print(f"✅ Модель MangaOCR найдена: {os.path.abspath(manga_path)}")

    # Проверка моделей детекции
    for name, path in settings.DETECTION_MODELS.items():
        if not os.path.exists(path):
            raise RuntimeError(f"❌ Модель детекции '{name}' не найдена по пути: {path}")
        print(f"✅ Модель детекции '{name}' найдена: {os.path.abspath(path)}")
    
    return True

# Выполняем валидацию при импорте модуля
validate_models()