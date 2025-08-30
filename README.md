# Manga Processing API (Python)

FastAPI service for manga processing:

- Text bubble detection (YOLO)
- OCR (MangaOCR - Japanese text recognition only)
- Manga panel detection and ordering (Florence‑2 `magiv3`)
- Proxy for OpenAI‑compatible chat (e.g., LM Studio)

Tested on: Python 3.11.0 (Windows)

## Requirements

- Windows, Python 3.11.x (recommended)
- NVIDIA GPU for acceleration (optional). CPU works but is slower.

## Install & Run

### Option A: Fast installation (recommended)

1. Download [manga-translator-api.7z.001](https://github.com/Zylan-cell/manga-translator-api/releases/download/release/manga-translator-api.7z.001), [manga-translator-api.7z.002](https://github.com/Zylan-cell/manga-translator-api/releases/download/release/manga-translator-api.7z.002), [manga-translator-api.7z.003](https://github.com/Zylan-cell/manga-translator-api/releases/download/release/manga-translator-api.7z.003), [manga-translator-api.7z.004](https://github.com/Zylan-cell/manga-translator-api/releases/download/release/manga-translator-api.7z.004) from the Release section.
2. Extract `manga-translator-api.7z.001`. It will automatically merge both parts into the full project folder.
3. Start ``run.bat``

This is simpler and much faster than installing all dependencies manually.

---

### Option B: Manual installation

1. Install uv (Windows PowerShell)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

2. Create and activate a virtual environment

```powershell
python -m venv .env
.env\scripts\activate
```

3. Install PyTorch

- CUDA 12.9 (GPU):

```powershell
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
```

- CPU only:

```powershell
uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
```

4. Install the rest of the dependencies

```powershell
uv pip install -r requirements.txt
```

5. Place your models under `weights/` (Download from Release):

```
weights/
  bubbles_yolo.pt             # YOLO model for bubble detection
  manga-ocr-base/             # LOCAL MangaOCR model folder (REQUIRED)
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

6. Run the server

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

OCR (MangaOCR - Japanese text only)

- GET `/model_info` — Verify which MangaOCR model is loaded (local vs downloaded)
- POST `/recognize_image`
  - body: `{ "image_data": "<base64>" }`
  - response: `{ "full_text": "..." }`
- POST `/recognize_images_batch`
  - body: `{ "images_data": ["<base64>", "<base64>", ...] }`
  - response: `{ "results": ["...", "...", ...] }`

Panel detection (Florence‑2 magiv3)

- POST `/detect_panels`
  - body: `{ "image_data": "<base64>" }`
  - response: `{ "panels": [[x1,y1,x2,y2], ...] }`

Translation proxy (OpenAI-like, e.g., LM Studio)

- GET `/v1/models`
- POST `/v1/chat/completions` (supports `stream: true`)
