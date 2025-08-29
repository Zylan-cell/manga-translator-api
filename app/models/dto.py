from typing import List, Optional
from pydantic import BaseModel

# Общие DTO
class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float = 1.0

class YoloDetectionResult(BaseModel):
    boxes: List[BoundingBox]

# НОВОЕ: DTO для панелей
class PanelDetectionResult(BaseModel):
    panels: List[List[int]]

class ImageRecognitionRequest(BaseModel):
    image_data: str
    engine: Optional[str] = "manga"          # "manga" | "easy"
    langs: Optional[List[str]] = None        # для easyocr, напр. ["ja","en"]
    auto_rotate: Optional[bool] = True       # автоповорот 0/90/270

class BatchRecognitionRequest(BaseModel):
    images_data: List[str]
    engine: Optional[str] = "manga"
    langs: Optional[List[str]] = None
    auto_rotate: Optional[bool] = True

class BatchRecognitionResponse(BaseModel):
    results: List[str]

# LM Studio (OpenAI-like)
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = 1500

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: dict