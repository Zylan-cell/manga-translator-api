from fastapi import APIRouter, HTTPException, Depends, Request

from app.services.panel_service import PanelOrderDetector
from app.logger import app_logger
from app.models.dto import ImageRecognitionRequest, PanelDetectionResult
from app.utils.images import b64_to_pil

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