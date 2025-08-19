from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel
from PIL import Image
import base64
from io import BytesIO
from typing import List

from app.services.panel_service import PanelOrderDetector
from app.logger import app_logger

router = APIRouter()

def get_service(request: Request) -> PanelOrderDetector:
    return request.app.state.services["panel"]

class ImageRequest(BaseModel):
    image_data: str

class PanelResponse(BaseModel):
    panels: List[List[float]]

@router.post("/detect_panels", response_model=PanelResponse)
async def detect_panels(
    request: ImageRequest,
    svc: PanelOrderDetector = Depends(get_service)
):
    try:
        image_data = base64.b64decode(request.image_data.split(',')[-1])
        image = Image.open(BytesIO(image_data))
        
        sorted_panel_bboxes = svc.get_sorted_panels(image)
        
        # Округляем до int
        panels_as_int = [list(map(int, p)) for p in sorted_panel_bboxes]

        return PanelResponse(panels=panels_as_int)
    except Exception as e:
        app_logger.error(f"[Panel API] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))