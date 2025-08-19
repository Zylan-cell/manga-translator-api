# app/services/translate_service.py

from __future__ import annotations
import time
import uuid
import requests
from typing import Dict, Any, Generator

from app.config import settings
from app.logger import app_logger as log

class TranslateService:
    def __init__(self) -> None:
        self.base = settings.LM_STUDIO_URL.rstrip("/")

    async def initialize(self) -> None:
        pass

    def models(self) -> Dict[str, Any]:
        url = f"{self.base}/v1/models"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.json()

    def _chat_stream(self, payload: Dict[str, Any]) -> Generator[bytes, None, None]:
        """Приватный метод для потоковой передачи."""
        url = f"{self.base}/v1/chat/completions"
        # Убедимся, что stream=True для запроса к LM Studio
        payload['stream'] = True
        try:
            with requests.post(url, json=payload, timeout=180, stream=True) as r:
                if not r.ok:
                    log.error("LM Studio stream error: %s %s", r.status_code, r.text[:500])
                    r.raise_for_status()
                
                # Просто проксируем строки (чанки) ответа как есть
                for line in r.iter_lines():
                    if line:
                        yield line + b'\n'
        except Exception as e:
            log.error("Exception during LM Studio stream: %s", e)
            # В случае ошибки можно отправить событие с ошибкой, если ваш фронтенд это поддерживает
            # Например: yield b'data: {"error": "stream failed"}\n\n'
            pass


    def _chat_blocking(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Приватный метод для обычных, блокирующих запросов."""
        url = f"{self.base}/v1/chat/completions"
        payload['stream'] = False # Явно указываем
        r = requests.post(url, json=payload, timeout=180)
        if not r.ok:
            log.error("LM Studio error: %s %s", r.status_code, r.text[:500])
            r.raise_for_status()
        
        data = r.json()
        # унифицируем поля
        if "id" not in data: data["id"] = f"chatcmpl-{uuid.uuid4().hex}"
        if "created" not in data: data["created"] = int(time.time())
        if "object" not in data: data["object"] = "chat.completion"
        if "usage" not in data: data["usage"] = {}
        return data

    def chat(self, payload: Dict[str, Any]) -> Any:
        """
        Основной метод. Выбирает потоковый или блокирующий режим
        в зависимости от параметра 'stream' в запросе.
        """
        if payload.get("stream", False):
            return self._chat_stream(payload)
        return self._chat_blocking(payload)