from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional

from app.logger import app_logger


class InpaintRequest(BaseModel):
    image_data: str
    mask_data: str
    model: Optional[str] = None


class InpaintAutoTextRequest(BaseModel):
    image_data: str
    boxes: List[List[int]]
    dilate: Optional[int] = 2


class InpaintResponse(BaseModel):
    image_data: str


router = APIRouter(prefix="", tags=["Inpainting"])


@router.post("/inpaint", response_model=InpaintResponse)
async def inpaint_basic(request: InpaintRequest, app_request: Request):
    """Basic inpainting endpoint (legacy compatibility)."""
    try:
        inpainting_service = app_request.app.state.services["inpainting"]
        
        if not inpainting_service.is_available():
            raise HTTPException(status_code=503, detail="Inpainting service not available")
        
        result = await inpainting_service.inpaint_with_mask(
            image_data=request.image_data,
            mask_data=request.mask_data,
            model=request.model
        )
        
        return InpaintResponse(image_data=result)
        
    except Exception as e:
        app_logger.error(f"Inpainting failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/inpaint_lama", response_model=InpaintResponse)
async def inpaint_lama(request: InpaintRequest, app_request: Request):
    """Inpaint using LAMA model with custom mask."""
    try:
        inpainting_service = app_request.app.state.services["inpainting"]
        
        if not inpainting_service.is_available():
            raise HTTPException(status_code=503, detail="LAMA inpainting service not available")
        
        app_logger.info(f"🎨 LAMA inpainting request (model: {request.model or 'default'})")
        
        result = await inpainting_service.inpaint_with_mask(
            image_data=request.image_data,
            mask_data=request.mask_data,
            model=request.model or "lama_large_512px"
        )
        
        return InpaintResponse(image_data=result)
        
    except Exception as e:
        app_logger.error(f"LAMA inpainting failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/inpaint_manual", response_model=InpaintResponse)
async def inpaint_manual(request: InpaintRequest, app_request: Request):
    """Manual inpainting with custom mask."""
    try:
        inpainting_service = app_request.app.state.services["inpainting"]
        
        if not inpainting_service.is_available():
            raise HTTPException(status_code=503, detail="Manual inpainting service not available")
        
        app_logger.info(f"🎨 Manual inpainting request (model: {request.model or 'default'})")
        
        result = await inpainting_service.inpaint_with_mask(
            image_data=request.image_data,
            mask_data=request.mask_data,
            model=request.model or "lama_large_512px"
        )
        
        return InpaintResponse(image_data=result)
        
    except Exception as e:
        app_logger.error(f"Manual inpainting failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/inpaint_auto_text", response_model=InpaintResponse)
async def inpaint_auto_text(request: InpaintAutoTextRequest, app_request: Request):
    """Automatically inpaint text areas based on bounding boxes."""
    try:
        inpainting_service = app_request.app.state.services["inpainting"]
        
        if not inpainting_service.is_available():
            raise HTTPException(status_code=503, detail="Auto text inpainting service not available")
        
        app_logger.info(f"🎨 Auto text inpainting request ({len(request.boxes)} boxes, dilate: {request.dilate})")
        
        result = await inpainting_service.inpaint_auto_text(
            image_data=request.image_data,
            boxes=request.boxes,
            dilate=request.dilate
        )
        
        return InpaintResponse(image_data=result)
        
    except Exception as e:
        app_logger.error(f"Auto text inpainting failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/inpaint/status")
async def inpaint_status(app_request: Request):
    """Get inpainting service status."""
    try:
        inpainting_service = app_request.app.state.services["inpainting"]
        model_info = inpainting_service.get_model_info()
        
        return {
            "status": "available" if inpainting_service.is_available() else "unavailable",
            "model_info": model_info
        }
        
    except Exception as e:
        app_logger.error(f"Failed to get inpaint status: {e}")
        return {
            "status": "error",
            "error": str(e)
        }