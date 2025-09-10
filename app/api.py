from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.logger import app_logger
from app.services.detection_service import DetectionService
from app.services.ocr_service import OcrService
from app.services.panel_service import PanelOrderDetector
from app.services.translate_service import TranslateService
from app.services.inpainting_service import InpaintingService

from app.routers import detection, ocr, panel, translate, inpainting

@asynccontextmanager
async def lifespan(app: FastAPI):
    app_logger.info("--- Service Initialization ---")
    app.state.services = {}

    app.state.services["detection"] = DetectionService()
    await app.state.services["detection"].initialize()

    app.state.services["ocr"] = OcrService()
    await app.state.services["ocr"].initialize()
    
    # Проверяем что MangaOCR использует локальную модель
    ocr_model_info = app.state.services["ocr"].get_model_info()
    app_logger.info(f"🔍 MangaOCR Model Verification: {ocr_model_info}")

    app.state.services["panel"] = PanelOrderDetector()

    app.state.services["translate"] = TranslateService()
    await app.state.services["translate"].initialize()

    app.state.services["inpainting"] = InpaintingService()
    await app.state.services["inpainting"].initialize()

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
    app.include_router(panel.router)
    app.include_router(translate.router)
    app.include_router(inpainting.router)

    @app.get("/", tags=["Root"])
    async def root():
        return {"message": "Manga Processing API is running."}
    
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Health check with model verification."""
        try:
            ocr_service = app.state.services.get("ocr")
            inpainting_service = app.state.services.get("inpainting")
            
            ocr_info = {}
            inpainting_info = {}
            
            if ocr_service:
                model_info = ocr_service.get_model_info()
                ocr_info = {
                    "ocr_model": model_info,
                    "using_local_model": model_info.get("local_path_exists", False) and model_info.get("config_exists", False)
                }
            
            if inpainting_service:
                inpainting_info = {
                    "inpainting_available": inpainting_service.is_available(),
                    "inpainting_model": inpainting_service.get_model_info()
                }
            
                return {
                    "status": "healthy",
                    "message": "API is running",
                    **ocr_info,
                    **inpainting_info
                }
            else:
                return {"status": "initializing", "message": "Services not ready"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    return app