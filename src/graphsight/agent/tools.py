import cv2
import numpy as np
from pathlib import Path
from langchain_core.tools import tool
from loguru import logger

class ImageProcessor:
    @staticmethod
    def apply_filter(image_path: str, filter_type: str) -> str:
        """指定されたフィルタを適用して一時保存し、パスを返す"""
        img = cv2.imread(image_path)
        
        if filter_type == "edge_enhancement":
            # エッジ強調（カーネル適用）
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            processed = cv2.filter2D(img, -1, kernel)
            
        elif filter_type == "binarize":
            # 二値化（白黒はっきりさせる）
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 大津の二値化
            _, processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

        elif filter_type == "high_contrast":
            # コントラストを上げる（CLAHE: 適応的ヒストグラム平坦化）
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl,a,b))
            processed = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        elif filter_type == "invert":
            # 色反転（黒背景・白文字の場合などに有効）
            processed = cv2.bitwise_not(img)

        else:
            return image_path # 何もしない

        # 保存処理（省略：一時ファイル名を生成して保存）
        output_path = image_path.replace(".png", f"_{filter_type}.png")
        cv2.imwrite(output_path, processed)
        return output_path

    @staticmethod
    def _load_image(image_path: str):
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        # 日本語パス対応
        n = np.fromfile(str(path), np.uint8)
        img = cv2.imdecode(n, cv2.IMREAD_COLOR)
        return img, str(path)

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
    def crop_region(image_path: str, x: int, y: int, w: int, h: int):
        """
        Crop a specific rectangular region of the image to see details (text, arrows).
        Arguments:
            x, y: Top-left coordinates.
            w, h: Width and height of the crop.
        Returns:
            The file path of the cropped image.
        """
        try:
            img, original_path = ImageProcessor._load_image(image_path)
            logger.info(f"Tool: crop_region {x},{y} {w}x{h}")
            
            # 範囲制限（画像外へのアクセス防止）
            H, W, _ = img.shape
            x, y = max(0, int(x)), max(0, int(y))
            w, h = min(int(w), W - x), min(int(h), H - y)

            crop = img[y:y+h, x:x+w]
            
            # クロップ画像を保存 (ファイル名は座標情報を含む)
            output_dir = Path(original_path).parent / "crops"
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"crop_{x}_{y}_{w}x{h}.jpg"
            
            cv2.imwrite(str(output_path), crop)
            
            # パスを返す (Agent Coreがこれを検知して画像をロードする)
            return str(output_path)
        except Exception as e:
            return f"Error: {e}"

    @tool
    def preprocess_image(image_path: str, method: str = "edge_enhancement"):
        """
        Process the image to enhance lines or text clarity.
        Arguments:
            method: 'edge_enhancement' (for faint lines/arrows), 'binarize' (for text reading).
        Returns:
            The file path of the processed image.
        """
        try:
            img, original_path = ImageProcessor._load_image(image_path)
            logger.info(f"Tool: preprocess {method}")
            
            output_dir = Path(original_path).parent / "processed"
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"proc_{Path(original_path).stem}_{method}.jpg"
            
            if method == "edge_enhancement":
                # エッジ強調処理
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                enhanced = clahe.apply(gray)
                # 白背景の黒線を太くするには erode (収縮) を使う
                kernel = np.ones((2,2), np.uint8)
                processed = cv2.erode(enhanced, kernel, iterations=1)
            elif method == "binarize":
                # 二値化処理
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                return "Error: Unknown method. Use 'edge_enhancement' or 'binarize'."

            cv2.imwrite(str(output_path), processed)
            return str(output_path)
        except Exception as e:
            return f"Error: {e}"

ALL_TOOLS = [ImageProcessor.get_image_info, ImageProcessor.crop_region, ImageProcessor.preprocess_image]

