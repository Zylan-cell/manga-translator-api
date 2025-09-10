from fastapi import APIRouter, HTTPException, Depends, Request

from app.services.detection_service import DetectionService
from app.logger import app_logger
from app.models.dto import ImageRecognitionRequest, YoloDetectionResult, BoundingBox

router = APIRouter()

def get_service(request: Request) -> DetectionService:
    return request.app.state.services["detection"]

@router.post("/detect_text_areas", response_model=YoloDetectionResult)
async def detect_text_areas(
    request: ImageRecognitionRequest,
    svc: DetectionService = Depends(get_service),
):
    try:
        # ИЗМЕНЕНО: Передаем имя модели в сервис
        boxes_data = svc.detect(request.image_data, model_name=request.detection_model)
        boxes = [BoundingBox(**box) for box in boxes_data]
        return YoloDetectionResult(boxes=boxes)
    except Exception as e:
        app_logger.error(f"[Detection API] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))