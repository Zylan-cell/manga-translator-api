# Manga Translator - Backend API

This is the backend server for the Manga Translator application.  
It handles heavy-lifting tasks like object detection, optical character recognition (OCR), inpainting, and text translation.  
The API is built with [FastAPI](https://fastapi.tiangolo.com/).

---

## ✨ Features

- **Panel Detection:** Detects comic panels to determine reading order.  
- **Bubble Detection:** Uses a YOLO model to find speech bubbles in an image.  
- **Text Recognition (OCR):** Extracts Japanese text from detected bubbles using Manga-OCR.  
- **Inpainting:** Removes the original text from the bubbles, preparing them for translation.  
- **Translation:** Connects to an LM Studio instance (or any OpenAI-compatible API) to translate text.  

---

## 🛠️ Setup and Installation

### Prerequisites

- Python 3.10+  
- `pip` and `virtualenv`  
- Git  
- (Optional but recommended) NVIDIA GPU with CUDA installed for performance  

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/manga-translator-api.git
   cd manga-translator-api
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv .env

   # On Windows
   .\.env\Scripts\activate

   # On macOS/Linux
   source .env/bin/activate
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Models**  
   Place the required models into the `weights/` directory:
   - `bubbles_yolo.pt` — speech bubble detection model  
   - `magiv3/` — panel detection model directory  
   - Manga-OCR and inpainting models will be downloaded automatically on first run  

---

## 🚀 How to Run

### Windows
```bash
run_server.bat
```

### macOS/Linux (example `run_server.sh`)
```sh
#!/bin/bash
source .env/bin/activate
export HF_HOME="./weights/hf"
# ... set other environment variables ...
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The server will start on `http://0.0.0.0:8000`.  
When it starts, it prints the local network IP to connect from your phone:

```
==> http://192.168.1.5:8000 <==
```

---

## 📡 API Endpoints

- `POST /detect_panels` — Detects comic panels  
- `POST /detect_text_areas` — Detects speech bubbles  
- `POST /recognize_images_batch` — Performs OCR on a batch of cropped bubble images  
- `POST /inpaint_auto_text` — Automatically inpaints text within specified bounding boxes  
- `POST /v1/chat/completions` — Proxies translation requests to an OpenAI-compatible API  
