# app/routers/translate.py

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse # <-- ИМПОРТИРУЕМ
from typing import Generator # <-- ИМПОРТИРУЕМ

from ..models.dto import ChatCompletionRequest
from ..services.translate_service import TranslateService

router = APIRouter(prefix="/v1")

def get_service(request: Request) -> TranslateService:
    return request.app.state.services["translate"]

@router.get("/models")
async def list_models(svc: TranslateService = Depends(get_service)):
    try:
        return svc.models()
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

@router.post("/chat/completions")
async def chat(req: ChatCompletionRequest, svc: TranslateService = Depends(get_service)):
    try:
        payload = req.model_dump()
        result = svc.chat(payload)
        
        # Если сервис вернул генератор, значит это стрим
        if isinstance(result, Generator):
            return StreamingResponse(result, media_type="text/event-stream")
        
        # Иначе это обычный JSON ответ
        return result
        
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))