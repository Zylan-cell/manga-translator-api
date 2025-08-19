import os

# Пути к моделям
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "weights/bubbles_yolo.pt")
PANEL_MODEL_PATH = os.getenv("PANEL_MODEL_PATH", "weights/magiv3")

# Настройки сервера
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))