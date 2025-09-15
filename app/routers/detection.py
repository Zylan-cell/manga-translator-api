from fastapi import APIRouter, HTTPException, Depends, Request

from app.services.detection_service import DetectionService
from app.logger import app_logger
from app.models.dto import ImageRecognitionRequest, YoloDetectionResult, BoundingBox
from app.models.dto import BatchRecognitionRequest, BatchRecognitionResponse
from app.utils.images import b64_to_pil

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


@router.post("/detect_text_areas_batch", response_model=BatchRecognitionResponse)
async def detect_text_areas_batch(
    request: BatchRecognitionRequest,
    svc: DetectionService = Depends(get_service),
):
    try:
        if not request.images_data:
            return BatchRecognitionResponse(results=[])
        images = [b64_to_pil(b) for b in request.images_data]
        batch_results = svc.detect_batch(images, model_name=None)
        # batch_results is a list of lists of boxes; convert to JSON-friendly
        texts = []
        for boxes in batch_results:
            # For compatibility with BatchRecognitionResponse, encode concatenated text or JSON string
            # We'll return a JSON string of boxes for now (frontend should parse)
            import json
            texts.append(json.dumps(boxes))
        return BatchRecognitionResponse(results=texts)
    except Exception as e:
        app_logger.error(f"[Detection API BATCH] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))