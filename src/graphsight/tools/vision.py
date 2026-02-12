import cv2
import uuid
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

class VisionTools:
    """
    OpenCV-based vision tools for the GraphSight agent.
    Handles image loading, cropping, and basic analysis.
    """

    def __init__(self, working_dir: Path = Path(".graphsight_temp")):
        self.working_dir = working_dir
        self.working_dir.mkdir(exist_ok=True, parents=True)

    def _load(self, image_path: str) -> np.ndarray:
        """Internal helper to load image safely."""
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to decode image at {image_path}")
        return img

    def _save_temp(self, img: np.ndarray, prefix: str = "crop") -> str:
        """Saves an image to a temporary path and returns the absolute string path."""
        filename = f"{prefix}_{uuid.uuid4().hex[:8]}.png"
        path = self.working_dir / filename
        cv2.imwrite(str(path), img)
        return str(path.absolute())

    def get_image_info(self, image_path: str) -> str:
        """
        Returns dimensions of the image.
        Useful for checking boundaries before cropping.
        """
        try:
            img = self._load(image_path)
            h, w, c = img.shape
            return f"Width: {w}, Height: {h}, Channels: {c}"
        except Exception as e:
            return f"Error getting image info: {str(e)}"

    def crop_image(self, image_path: str, x: int, y: int, w: int, h: int) -> str:
        """
        Crops a specific region of the image.
        
        Args:
            image_path: Path to the source image.
            x: Top-left X coordinate.
            y: Top-left Y coordinate.
            w: Width of the crop.
            h: Height of the crop.
            
        Returns:
            Path to the saved cropped image file.
        """
        try:
            img = self._load(image_path)
            img_h, img_w = img.shape[:2]

            # Validation (Explicit Error approach)
            if x < 0 or y < 0 or w <= 0 or h <= 0:
                 raise ValueError(f"Invalid dimensions: x={x}, y={y}, w={w}, h={h}")
            
            if (x + w) > img_w or (y + h) > img_h:
                raise ValueError(
                    f"Crop region out of bounds. Image size is {img_w}x{img_h}, "
                    f"but requested region ends at {x+w}x{y+h}."
                )

            # Execution
            crop = img[y:y+h, x:x+w]
            saved_path = self._save_temp(crop, prefix="crop")
            return f"Cropped image saved to: {saved_path}"

        except Exception as e:
            return f"Error cropping image: {str(e)}"

    def resize_image(self, image_path: str, scale: float) -> str:
        """
        Resizes the image by a scale factor. 
        Useful if the image is too large for processing or too small to see details.
        """
        try:
            img = self._load(image_path)
            new_w = int(img.shape[1] * scale)
            new_h = int(img.shape[0] * scale)
            
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            saved_path = self._save_temp(resized, prefix="resize")
            return f"Resized image (scale={scale}) saved to: {saved_path}"
            
        except Exception as e:
            return f"Error resizing image: {str(e)}"

