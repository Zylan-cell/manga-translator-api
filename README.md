# Manga Processing API (Python)

FastAPI service for manga processing:
- Text bubble detection (YOLO)
- OCR (MangaOCR)
- Text inpainting (SimpleLaMa)
- Manga panel detection and ordering (Florence‑2 `magiv3`)
- Proxy for OpenAI‑compatible chat (e.g., LM Studio)

Tested on: Python 3.11.0 (Windows)

## Requirements
- Windows, Python 3.11.x (recommended)
- NVIDIA GPU for acceleration (optional). CPU works but is slower.

## Install & Run (uv recommended)

1) Install uv (Windows PowerShell)
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

2) Create and activate a virtual environment
```powershell
python -m venv .env
.env\scripts\activate
```

3) Install PyTorch
- CUDA 12.9 (GPU):
```powershell
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
```
- CPU only:
```powershell
uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
```

4) Install the rest of the dependencies
```powershell
uv pip install -r requirements.txt
```

5) Place your models under `weights/` (Download from Release):

```
weights/
  bubbles_yolo.pt             # YOLO model for bubble detection
  manga-ocr-base/             # local MangaOCR model folder (from HF)
    config.json
    preprocessor_config.json
    tokenizer.json
    model.safetensors         # recommended (avoid .bin)
    ... (other tokenizer files)
  magiv3/                     # Florence-2 panel model folder (from HF)
    config.json
    model.safetensors
    tokenizer.json
    ... (other assets)
```

6) Run the server
```powershell
python main.py
# Dev mode with autoreload:
python main.py
```
- Default address: http://localhost:8000

Notes:
- Defaults are set in `app/config.py`. You can override via env vars (see below).
- Using `safetensors` is recommended. If you only have `pytorch_model.bin`, you’ll need `torch>=2.6` due to a security restriction in `torch.load` (CVE-2025-32434).

## Configuration (env vars)

- API_HOST — default `0.0.0.0`
- API_PORT — default `8000`
- YOLO_MODEL_PATH — default `weights/bubbles_yolo.pt`
- PANEL_MODEL_PATH — default `weights/magiv3`
- MANGA_OCR_PATH — default `weights/manga-ocr-base`
- LM_STUDIO_URL — default `http://localhost:1234`

Optional (for fully offline usage):
- TRANSFORMERS_OFFLINE=1
- HF_HUB_DISABLE_TELEMETRY=1

## Endpoints

- GET `/` — health message: `{"message": "Manga Processing API is running."}`

Bubble detection (YOLO)
- POST `/detect_text_areas`
  - body: `{ "image_data": "<base64 or data:image/...;base64,...>" }`
  - response: `{ "boxes": [{ "x1": int, "y1": int, "x2": int, "y2": int, "confidence": float }, ...] }`

OCR (MangaOCR)
- POST `/recognize_image`
  - body: `{ "image_data": "<base64>" }`
  - response: `{ "full_text": "..." }`
- POST `/recognize_images_batch`
  - body: `{ "images_data": ["<base64>", "<base64>", ...] }`
  - response: `{ "results": ["...", "...", ...] }`

Inpainting (SimpleLaMa)
- POST `/inpaint`
  - body: `{ "image_data": "<base64>", "mask_data": "<base64 PNG mask>" }`
  - response: `{ "image_data": "<base64 PNG result>" }`
- POST `/inpaint_auto_text`
  - body: `{ "image_data": "<base64>", "boxes": [[x1,y1,x2,y2], ...], "dilate": 2, "return_mask": false }`
  - response: `{ "image_data": "<base64 PNG>", "mask_data": "<base64 PNG or null>" }`

Panel detection (Florence‑2 magiv3)
- POST `/detect_panels`
  - body: `{ "image_data": "<base64>" }`
  - response: `{ "panels": [[x1,y1,x2,y2], ...] }`

Translation proxy (OpenAI-like, e.g., LM Studio)
- GET `/v1/models`
- POST `/v1/chat/completions` (supports `stream: true`)

## Troubleshooting

Torch (CUDA/CPU)
- CUDA 12.9:
```powershell
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
```
- CPU:
```powershell
uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
```

MangaOCR `.bin` restriction
- If you see: “upgrade torch to at least v2.6 …” — that’s due to `pytorch_model.bin`.
  - Use safetensors instead, or
  - Upgrade torch to >= 2.6 (not always available for all CUDA combos).

Panel model missing deps
```powershell
uv pip install einops pytorch_metric_learning timm shapely
```

Transformers >= 4.50 and custom models
- Newer Transformers dropped `GenerationMixin` from `PreTrainedModel`, some custom models may break on `.generate`.
  - Easiest fix: pin Transformers `< 4.50`:
```powershell
uv pip install "transformers<4.50" -U
```
  - Or patch the custom model (advanced).
