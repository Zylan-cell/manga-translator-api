from fastapi import APIRouter, HTTPException, Depends, Request
from app.services.ocr_service import OcrService
from app.logger import app_logger
from app.models.dto import (
    ImageRecognitionRequest,
    BatchRecognitionRequest,
    BatchRecognitionResponse,
)

router = APIRouter()

def get_service(request: Request) -> OcrService:
    return request.app.state.services["ocr"]

@router.get("/model_info")
async def get_model_info(svc: OcrService = Depends(get_service)):
    """Returns information about the loaded MangaOCR model to verify it's using local model."""
    try:
        model_info = svc.get_model_info()
        return {
            "success": True,
            "model_info": model_info,
            "message": "Local model verification" if model_info.get("local_path_exists") else "Local model not found"
        }
    except Exception as e:
        app_logger.error(f"[OCR MODEL INFO] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recognize_image")
async def recognize_image(request: ImageRecognitionRequest, svc: OcrService = Depends(get_service)):
    try:
        text = svc.recognize(
            request.image_data,
            engine=request.engine or "manga",
            langs=request.langs,
            auto_rotate=request.auto_rotate if request.auto_rotate is not None else True,
        )
        return {"full_text": text}
    except Exception as e:
        app_logger.error(f"[OCR API] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recognize_images_batch", response_model=BatchRecognitionResponse)
async def recognize_images_batch(request: BatchRecognitionRequest, svc: OcrService = Depends(get_service)):
    try:
        if not request.images_data:
            return BatchRecognitionResponse(results=[])
        texts = svc.recognize_batch(
            request.images_data,
            engine=request.engine or "manga",
            langs=request.langs,
            auto_rotate=request.auto_rotate if request.auto_rotate is not None else True,
        )
        return BatchRecognitionResponse(results=texts)
    except Exception as e:
        app_logger.error(f"[OCR API BATCH] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))