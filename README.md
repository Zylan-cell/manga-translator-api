\# Manga Translator - Backend API



This is the backend server for the Manga Translator application. It handles heavy-lifting tasks like object detection, optical character recognition (OCR), inpainting, and text translation. The API is built with FastAPI.



---



\## ✨ Features



\- \*\*Panel Detection:\*\* Detects comic panels to determine reading order.  

\- \*\*Bubble Detection:\*\* Uses a YOLO model to find speech bubbles in an image.  

\- \*\*Text Recognition (OCR):\*\* Extracts Japanese text from detected bubbles using Manga-OCR.  # Manga Translator - Backend API



This is the backend server for the Manga Translator application. It handles heavy-lifting tasks like object detection, optical character recognition (OCR), inpainting, and text translation. The API is built with FastAPI.



---



\## ✨ Features



\- \*\*Panel Detection:\*\* Detects comic panels to determine reading order.  

\- \*\*Bubble Detection:\*\* Uses a YOLO model to find speech bubbles in an image.  

\- \*\*Text Recognition (OCR):\*\* Extracts Japanese text from detected bubbles using Manga-OCR.  

\- \*\*Inpainting:\*\* Removes the original text from the bubbles, preparing them for translation.  

\- \*\*Translation:\*\* Connects to an LM Studio instance (or any OpenAI-compatible API) to translate text.  



---



\## 🛠️ Setup and Installation



\### Prerequisites



\- Python 3.10+  

\- `pip` and `virtualenv`  

\- Git  

\- (Optional but recommended) NVIDIA GPU with CUDA installed for performance.  



\### Installation Steps



1\. \*\*Clone the repository\*\*

&nbsp;  ```bash

&nbsp;  git clone https://github.com/YOUR\_USERNAME/manga-translator-api.git

&nbsp;  cd manga-translator-api

&nbsp;  ```



2\. \*\*Create and activate a virtual environment\*\*

&nbsp;  ```bash

&nbsp;  python -m venv .env



&nbsp;  # On Windows

&nbsp;  .\\.env\\Scripts\\activate



&nbsp;  # On macOS/Linux

&nbsp;  source .env/bin/activate

&nbsp;  ```



3\. \*\*Install Python dependencies\*\*

&nbsp;  ```bash

&nbsp;  pip install -r requirements.txt

&nbsp;  ```



4\. \*\*Download Models\*\*  

&nbsp;  The required models are not included in this repository. You need to download them and place them in the `weights/` directory:

&nbsp;  - `bubbles\_yolo.pt` — speech bubble detection model  

&nbsp;  - `magiv3/` — directory for the panel detection model  

&nbsp;  - Manga-OCR and inpainting models will be downloaded automatically on first run to their respective cache folders  



---



\## 🚀 How to Run



1\. \*\*Run the server\*\*



&nbsp;  On \*\*Windows\*\*:

&nbsp;  ```bash

&nbsp;  run\_server.bat

&nbsp;  ```



&nbsp;  On \*\*macOS/Linux\*\* (create a `run\_server.sh`):

&nbsp;  ```sh

&nbsp;  #!/bin/bash

&nbsp;  source .env/bin/activate

&nbsp;  export HF\_HOME="./weights/hf"

&nbsp;  # ... set other environment variables ...

&nbsp;  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

&nbsp;  ```



2\. The server will start on `http://0.0.0.0:8000`.  

&nbsp;  When it starts, it will print the local network IP address you should use in the Tauri application to connect from your phone.



&nbsp;  Example:

&nbsp;  ```

&nbsp;  ==> http://192.168.1.5:8000 <==

&nbsp;  ```



---



\## 📡 API Endpoints



\- `POST /detect\_panels` — Detects comic panels  

\- `POST /detect\_text\_areas` — Detects speech bubbles  

\- `POST /recognize\_images\_batch` — Performs OCR on a batch of cropped bubble images  

\- `POST /inpaint\_auto\_text` — Automatically inpaints text within specified bounding boxes  

\- `POST /v1/chat/completions` — Proxies translation requests to an OpenAI-compatible API  



\- \*\*Inpainting:\*\* Removes the original text from the bubbles, preparing them for translation.  

\- \*\*Translation:\*\* Connects to an LM Studio instance (or any OpenAI-compatible API) to translate text.  



---



\## 🛠️ Setup and Installation



\### Prerequisites



\- Python 3.10+  

\- `pip` and `virtualenv`  

\- Git  

\- (Optional but recommended) NVIDIA GPU with CUDA installed for performance.  



\### Installation Steps



1\. \*\*Clone the repository\*\*

&nbsp;  ```bash

&nbsp;  git clone https://github.com/YOUR\_USERNAME/manga-translator-api.git

&nbsp;  cd manga-translator-api

&nbsp;  ```



2\. \*\*Create and activate a virtual environment\*\*

&nbsp;  ```bash

&nbsp;  python -m venv .env



&nbsp;  # On Windows

&nbsp;  .\\.env\\Scripts\\activate



&nbsp;  # On macOS/Linux

&nbsp;  source .env/bin/activate

&nbsp;  ```



3\. \*\*Install Python dependencies\*\*

&nbsp;  ```bash

&nbsp;  pip install -r requirements.txt

&nbsp;  ```



4\. \*\*Download Models\*\*  

&nbsp;  The required models are not included in this repository. You need to download them and place them in the `weights/` directory:

&nbsp;  - `bubbles\_yolo.pt` — speech bubble detection model  

&nbsp;  - `magiv3/` — directory for the panel detection model  

&nbsp;  - Manga-OCR and inpainting models will be downloaded automatically on first run to their respective cache folders  



---



\## 🚀 How to Run



1\. \*\*Run the server\*\*



&nbsp;  On \*\*Windows\*\*:

&nbsp;  ```bash

&nbsp;  run\_server.bat

&nbsp;  ```



&nbsp;  On \*\*macOS/Linux\*\* (create a `run\_server.sh`):

&nbsp;  ```sh

&nbsp;  #!/bin/bash

&nbsp;  source .env/bin/activate

&nbsp;  export HF\_HOME="./weights/hf"

&nbsp;  # ... set other environment variables ...

&nbsp;  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

&nbsp;  ```



2\. The server will start on `http://0.0.0.0:8000`.  

&nbsp;  When it starts, it will print the local network IP address you should use in the Tauri application to connect from your phone.



&nbsp;  Example:

&nbsp;  ```

&nbsp;  ==> http://192.168.1.5:8000 <==

&nbsp;  ```



---



\## 📡 API Endpoints



\- `POST /detect\_panels` — Detects comic panels  

\- `POST /detect\_text\_areas` — Detects speech bubbles  

\- `POST /recognize\_images\_batch` — Performs OCR on a batch of cropped bubble images  

\- `POST /inpaint\_auto\_text` — Automatically inpaints text within specified bounding boxes  

\- `POST /v1/chat/completions` — Proxies translation requests to an OpenAI-compatible API  



