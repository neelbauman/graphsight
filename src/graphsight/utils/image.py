import base64
from pathlib import Path
import math
import string
import os
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple

def encode_image_to_base64(image_path: str) -> str:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
        
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def add_grid_overlay(image_path: str, min_cell_size: int = 150) -> Tuple[str, int, int]:
    """
    画像にグリッドとラベル(A1, B2...)を焼き込み、一時ファイルのパスを返す。
    ラベルはセルの中央に配置し、サイズを控えめにする。
    """
    with Image.open(image_path) as img:
        img = img.convert("RGBA")
        width, height = img.size
        
        cols = max(1, width // min_cell_size)
        rows = max(1, height // min_cell_size)
        
        cell_w = width / cols
        cell_h = height / rows
        
        overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        
        # フォント設定: 少し小さく (24 -> 16)
        font_size = 16
        try:
            # Linux/Docker/Mac/Win それぞれの代表的なフォントパス
            font_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/System/Library/Fonts/Helvetica.ttc",
                "arial.ttf",
                "C:\\Windows\\Fonts\\arial.ttf"
            ]
            font = None
            for path in font_paths:
                try:
                    font = ImageFont.truetype(path, font_size)
                    break
                except IOError:
                    continue
            if font is None:
                font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

        for r in range(rows):
            for c in range(cols):
                x = c * cell_w
                y = r * cell_h
                
                # Excel風カラム名
                col_label = ""
                temp_c = c
                while temp_c >= 0:
                    col_label = string.ascii_uppercase[temp_c % 26] + col_label
                    temp_c = (temp_c // 26) - 1
                    if temp_c < 0: break
                
                label = f"{col_label}{r + 1}"
                
                # グリッド線 (赤色、半透明)
                draw.rectangle([x, y, x + cell_w, y + cell_h], outline=(255, 0, 0, 80), width=1)
                
                # --- 中央配置の計算 ---
                # テキストのサイズを取得
                if hasattr(draw, "textbbox"):
                    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
                    text_w = right - left
                    text_h = bottom - top
                else:
                    text_w, text_h = draw.textsize(label, font=font)

                # セルの中央 - テキストの半分のサイズ
                text_x = x + (cell_w - text_w) / 2
                text_y = y + (cell_h - text_h) / 2
                
                # 背景矩形 (視認性向上)
                pad = 4
                bg_rect = [text_x - pad, text_y - pad, text_x + text_w + pad, text_y + text_h + pad]
                draw.rectangle(bg_rect, fill=(255, 255, 255, 200))
                
                # テキスト描画
                draw.text((text_x, text_y), label, fill=(200, 0, 0, 255), font=font)

        out = Image.alpha_composite(img, overlay)
        
        output_path = f"{os.path.splitext(image_path)[0]}.grid.png"
        out.save(output_path)
        
        return output_path, rows, cols

