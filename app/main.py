import socket
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.logger import app_logger
from app.config import settings
from app.routers import detection, ocr, inpaint, panel, translate
from app.services.detection_service import DetectionService
from app.services.ocr_service import OcrService
from app.services.inpaint_service import InpaintService
from app.services.panel_service import PanelOrderDetector
from app.services.translate_service import TranslateService

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

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
    
    yield
    
    app_logger.info("--- Service Shutdown ---")
    app.state.services.clear()

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

# --- ВОЗВРАЩАЕМ ЭТОТ БЛОК ---
if __name__ == "__main__":
    local_ip = get_local_ip()
    print("---" * 10)
    app_logger.info(f"API Server starting up...")
    app_logger.info(f"To connect from your phone, use this URL in the app settings:")
    app_logger.info(f"==> http://{local_ip}:{settings.API_PORT} <==")
    print("---" * 10)
    
    # Запускаем Uvicorn программно, это решит проблему с --reload
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    )