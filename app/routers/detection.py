from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel
from typing import List

from app.services.detection_service import DetectionService
from app.logger import app_logger

router = APIRouter()

def get_service(request: Request) -> DetectionService:
    # Правильный способ получить сервис из состояния приложения
    return request.app.state.services["detection"]

class ImageRecognitionRequest(BaseModel):
    image_data: str

class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

class YoloDetectionResult(BaseModel):
    boxes: List[BoundingBox]

@router.post("/detect_text_areas", response_model=YoloDetectionResult)
async def detect_text_areas(
    request: ImageRecognitionRequest,
    svc: DetectionService = Depends(get_service) # Используем Depends
):
    try:
        boxes_data = svc.detect(request.image_data)
        boxes = [BoundingBox(**box) for box in boxes_data]
        return YoloDetectionResult(boxes=boxes)
    except Exception as e:
        app_logger.error(f"[Detection API] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))