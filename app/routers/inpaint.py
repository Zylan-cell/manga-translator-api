from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel
from typing import List, Optional

from app.services.inpaint_service import InpaintService
from app.logger import app_logger

router = APIRouter()

def get_service(request: Request) -> InpaintService:
    return request.app.state.services["inpaint"]

class InpaintRequest(BaseModel):
    image_data: str
    mask_data: str

class InpaintResponse(BaseModel):
    image_data: str
    mask_data: Optional[str] = None
    
class InpaintAutoTextRequest(BaseModel):
    image_data: str
    boxes: Optional[List[List[int]]] = None
    dilate: Optional[int] = 2
    return_mask: Optional[bool] = False

@router.post("/inpaint", response_model=InpaintResponse)
async def inpaint(req: InpaintRequest, svc: InpaintService = Depends(get_service)):
    try:
        out = svc.inpaint_with_mask(req.image_data, req.mask_data)
        return InpaintResponse(image_data=out)
    except Exception as e:
        app_logger.error(f"[INPAINT /inpaint] {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/inpaint_auto_text", response_model=InpaintResponse)
async def inpaint_auto_text(req: InpaintAutoTextRequest, svc: InpaintService = Depends(get_service)):
    try:
        img_b64, mask_b64 = svc.inpaint_auto_text(
            req.image_data,
            boxes=req.boxes,
            dilate=req.dilate,
            return_mask=req.return_mask,
        )
        return InpaintResponse(image_data=img_b64, mask_data=mask_b64)
    except Exception as e:
        app_logger.error(f"[INPAINT /inpaint_auto_text] {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))