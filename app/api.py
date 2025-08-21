from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.logger import app_logger
from app.services.detection_service import DetectionService
from app.services.ocr_service import OcrService
from app.services.inpaint_service import InpaintService
from app.services.panel_service import PanelOrderDetector
from app.services.translate_service import TranslateService

from app.routers import detection, ocr, inpaint, panel, translate

@asynccontextmanager
async def lifespan(app: FastAPI):
    app_logger.info("--- Service Initialization ---")
    app.state.services = {}

    app.state.services["detection"] = DetectionService()
    await app.state.services["detection"].initialize()

    app.state.services["ocr"] = OcrService()
    await app.state.services["ocr"].initialize()

    app.state.services["inpaint"] = InpaintService()
    await app.state.services["inpaint"].initialize()

    app.state.services["panel"] = PanelOrderDetector()

    app.state.services["translate"] = TranslateService()
    await app.state.services["translate"].initialize()

    app_logger.info("--- All services initialized successfully ---")
    try:
        yield
    finally:
        app_logger.info("--- Service Shutdown ---")
        app.state.services.clear()

def create_app() -> FastAPI:
    app = FastAPI(title="Manga Processing API", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(detection.router)
    app.include_router(ocr.router)
    app.include_router(inpaint.router)
    app.include_router(panel.router)
    app.include_router(translate.router)

    @app.get("/", tags=["Root"])
    async def root():
        return {"message": "Manga Processing API is running."}

    return app