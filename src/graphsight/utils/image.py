import base64
from pathlib import Path

def encode_image_to_base64(image_path: str) -> str:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
        
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

