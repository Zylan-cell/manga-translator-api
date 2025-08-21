import base64
import io
from io import BytesIO
from PIL import Image, ImageOps

def b64_to_bytes(s: str) -> bytes:
    if not s:
        return b""
    if "," in s:
        s = s.split(",", 1)[1]
    return base64.b64decode(s)

def b64_to_pil(s: str) -> Image.Image:
    img = Image.open(BytesIO(b64_to_bytes(s)))
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    return img.convert("RGB")

def pil_to_b64_png(img: Image.Image, with_data_uri: bool = False) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    raw = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{raw}" if with_data_uri else raw