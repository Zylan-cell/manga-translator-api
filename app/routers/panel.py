from fastapi import APIRouter, HTTPException, Depends, Request

from app.services.panel_service import PanelOrderDetector
from app.logger import app_logger
from app.models.dto import ImageRecognitionRequest, PanelDetectionResult
from app.utils.images import b64_to_pil
from app.models.dto import BatchRecognitionRequest

router = APIRouter()

def get_service(request: Request) -> PanelOrderDetector:
    return request.app.state.services["panel"]

@router.post("/detect_panels", response_model=PanelDetectionResult)
async def detect_panels(
    request: ImageRecognitionRequest,
    svc: PanelOrderDetector = Depends(get_service),
):
    try:
        image = b64_to_pil(request.image_data)
        sorted_panel_bboxes = svc.get_sorted_panels(image)
        return PanelDetectionResult(panels=sorted_panel_bboxes)
    except Exception as e:
        app_logger.error(f"[Panel API] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect_panels_batch")
async def detect_panels_batch(
    request: BatchRecognitionRequest,
    svc: PanelOrderDetector = Depends(get_service),
):
    try:
        if not request.images_data:
            return {"results": []}
        images = [b64_to_pil(b) for b in request.images_data]
        # svc already supports batch inference via model.predict_detections_and_associations
        results = svc.get_sorted_panels_batch(images)
        return {"results": results}
    except Exception as e:
        app_logger.error(f"[Panel API BATCH] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))