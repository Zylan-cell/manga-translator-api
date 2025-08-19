# app/logger.py
import logging
import sys

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Настраивает и возвращает именованный логгер."""
    # Получаем логгер для нашего приложения
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Предотвращаем добавление обработчиков несколько раз, если модуль импортируется повторно
    if not logger.handlers:
        # Создаем обработчик, который выводит логи в консоль (sys.stdout)
        handler = logging.StreamHandler(sys.stdout)
        
        # Устанавливаем уровень для обработчика
        handler.setLevel(level)
        
        # Создаем форматтер для логов
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Применяем форматтер к обработчику
        handler.setFormatter(formatter)
        
        # Добавляем обработчик к логгеру
        logger.addHandler(handler)
        
    # Выключаем распространение логов к корневому логгеру,
    # чтобы избежать дублирования сообщений (например, от uvicorn)
    logger.propagate = False
    
    return logger

# Создаем основной логгер для всего приложения
app_logger = setup_logger("manga-api")