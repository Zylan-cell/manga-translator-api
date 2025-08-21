from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.models.dto import ChatCompletionRequest
from app.services.translate_service import TranslateService

router = APIRouter(prefix="/v1")

def get_service(request: Request) -> TranslateService:
    return request.app.state.services["translate"]

@router.get("/models")
async def list_models(svc: TranslateService = Depends(get_service)):
    try:
        return await svc.models()
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

@router.post("/chat/completions")
async def chat(req: ChatCompletionRequest, svc: TranslateService = Depends(get_service)):
    try:
        payload = req.model_dump()
        if payload.get("stream"):
            gen = svc.chat_stream(payload)
            return StreamingResponse(gen, media_type="text/event-stream")
        data = await svc.chat_blocking(payload)
        return data
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))