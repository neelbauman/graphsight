import cv2
import numpy as np
from pathlib import Path
from langchain_core.tools import tool
from loguru import logger
from typing import Optional

class ImageProcessor:
    @staticmethod
    def _load_image(image_path: str):
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        # 日本語パス対応
        n = np.fromfile(str(path), np.uint8)
        img = cv2.imdecode(n, cv2.IMREAD_COLOR)
        return img, str(path)

    @staticmethod
    def _process_array(img: np.ndarray, method: str) -> np.ndarray:
        """OpenCV画像配列に対して画像処理を適用する共通メソッド"""
        
        # グレースケール変換（共通前処理）
        if len(img.shape) == 3 and method in ["edge_enhancement", "binarize"]:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        if method == "edge_enhancement":
            # コントラスト制限付き適応的ヒストグラム平坦化 (CLAHE)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            return enhanced
            
        elif method == "binarize":
            # 1. 二値化 (大津の二値化)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 2. Erode (収縮 = 黒領域の膨張)
            # 文字や線を太くして、かすれを修復する
            kernel = np.ones((2,2), np.uint8)
            eroded = cv2.erode(binary, kernel, iterations=1)
            
            # 3. ノイズ低減 (Median Blur)
            # Erodeで強調されたスパイクノイズを除去し、エッジを滑らかにする
            # ksize=3 は 3x3 の領域の中央値を採用する（文字を潰しすぎないサイズ）
            denoised = cv2.medianBlur(eroded, 3)
            
            return denoised

        elif method == "high_contrast":
             # コントラスト強調 (カラー保持)
            if len(img.shape) == 3:
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                cl = clahe.apply(l)
                limg = cv2.merge((cl,a,b))
                processed = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
                return processed
            else:
                return img

        return img

    @tool
    def get_image_info(image_path: str):
        """
        Get the dimensions (width, height) of the image at the given path.
        Use this to understand the scale before cropping.
        """
        try:
            img, _ = ImageProcessor._load_image(image_path)
            h, w, _ = img.shape
            return f"Image Size: {w}x{h}"
        except Exception as e:
            return f"Error: {e}"

    @tool
    def crop_region(image_path: str, x: int, y: int, w: int, h: int, preprocess: Optional[str] = None):
        """
        Crop a specific rectangular region of the image to see details (text, arrows).
        Arguments:
            x, y: Top-left coordinates.
            w, h: Width and height of the crop.
            preprocess: Optional. 'edge_enhancement' or 'binarize' to apply filters immediately.
        Returns:
            The file path of the cropped (and optionally processed) image.
        """
        try:
            img, original_path = ImageProcessor._load_image(image_path)
            logger.info(f"Tool: crop_region {x},{y} {w}x{h} (preprocess={preprocess})")
            
            # 範囲制限
            H, W, _ = img.shape
            x, y = max(0, int(x)), max(0, int(y))
            w, h = min(int(w), W - x), min(int(h), H - y)

            crop = img[y:y+h, x:x+w]
            
            # 前処理の適用
            suffix = ""
            if preprocess:
                crop = ImageProcessor._process_array(crop, preprocess)
                suffix = f"_{preprocess}"

            # 保存
            output_dir = Path(original_path).parent / "crops"
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"crop_{x}_{y}_{w}x{h}{suffix}.jpg"
            
            cv2.imwrite(str(output_path), crop)
            return str(output_path)
        except Exception as e:
            return f"Error: {e}"

    @tool
    def preprocess_image(image_path: str, method: str = "edge_enhancement"):
        """
        Process the image to enhance lines or text clarity.
        Arguments:
            method: 'edge_enhancement' (CLAHE), 'binarize' (Threshold+Erode+Denoise).
        Returns:
            The file path of the processed image.
        """
        try:
            img, original_path = ImageProcessor._load_image(image_path)
            logger.info(f"Tool: preprocess {method}")
            
            processed = ImageProcessor._process_array(img, method)
            
            output_dir = Path(original_path).parent / "processed"
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"proc_{Path(original_path).stem}_{method}.jpg"
            
            cv2.imwrite(str(output_path), processed)
            return str(output_path)
        except Exception as e:
            return f"Error: {e}"

ALL_TOOLS = [ImageProcessor.get_image_info, ImageProcessor.crop_region, ImageProcessor.preprocess_image]

