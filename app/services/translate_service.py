from __future__ import annotations
from typing import Any, Dict, AsyncGenerator
import time
import uuid
import httpx

from app.config import settings
from app.logger import app_logger as log

class TranslateService:
    def __init__(self) -> None:
        self.base = settings.LM_STUDIO_URL.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    async def initialize(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=180.0)

    async def models(self):
        assert self._client is not None
        url = f"{self.base}/v1/models"
        r = await self._client.get(url)
        r.raise_for_status()
        return r.json()

    async def chat_blocking(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        assert self._client is not None
        url = f"{self.base}/v1/chat/completions"
        payload = dict(payload)
        payload["stream"] = False
        r = await self._client.post(url, json=payload)
        if not r.is_success:
            log.error("LM Studio error: %s %s", r.status_code, r.text[:500])
            r.raise_for_status()
        data = r.json()
        data.setdefault("id", f"chatcmpl-{uuid.uuid4().hex}")
        data.setdefault("created", int(time.time()))
        data.setdefault("object", "chat.completion")
        data.setdefault("usage", {})
        return data

    async def chat_stream(self, payload: Dict[str, Any]) -> AsyncGenerator[bytes, None]:
        assert self._client is not None
        url = f"{self.base}/v1/chat/completions"
        payload = dict(payload)
        payload["stream"] = True
        async with self._client.stream("POST", url, json=payload) as r:
            if not r.is_success:
                body = await r.aread()
                log.error("LM Studio stream error: %s %s", r.status_code, body[:500])
                r.raise_for_status()
            async for line in r.aiter_lines():
                if line:
                    yield (line + "\n").encode("utf-8")

    async def chat(self, payload: Dict[str, Any]) -> Any:
        if payload.get("stream"):
            return self.chat_stream(payload)
        return await self.chat_blocking(payload)