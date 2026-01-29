import base64
import math
import string
import os
from pathlib import Path
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
    画像にグリッドを焼き込む。
    【改善版】実線によるノイズを避けるため、交点マーカー(+)とラベルのみを描画する。
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
        
        # フォント設定 (小さめ、かつ視認性重視)
        font_size = 14
        try:
            font_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
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

        # マーカーとラベルの色設定 (シアン系: 図面で赤や黒はよく使われるため避ける)
        marker_color = (0, 150, 255, 180) # 青緑系、半透明
        text_color = (0, 100, 200, 200)
        bg_color = (255, 255, 255, 160)   # 白背景、かなり薄く

        for r in range(rows):
            for c in range(cols):
                # セル左上の座標
                x = c * cell_w
                y = r * cell_h
                
                # --- 1. 交点マーカー (+) の描画 ---
                # 各セルの四隅を描画するとうるさいので、左上だけ描画して網羅する
                # マーカーサイズ
                m_size = 10 
                # 横棒
                draw.line([(x - m_size, y), (x + m_size, y)], fill=marker_color, width=2)
                # 縦棒
                draw.line([(x, y - m_size), (x, y + m_size)], fill=marker_color, width=2)

                # --- 2. ラベルの生成 ---
                col_label = ""
                temp_c = c
                while temp_c >= 0:
                    col_label = string.ascii_uppercase[temp_c % 26] + col_label
                    temp_c = (temp_c // 26) - 1
                    if temp_c < 0: break
                
                label = f"{col_label}{r + 1}"
                
                # --- 3. ラベルを中央に配置 ---
                if hasattr(draw, "textbbox"):
                    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
                    text_w = right - left
                    text_h = bottom - top
                else:
                    text_w, text_h = draw.textsize(label, font=font)

                text_x = x + (cell_w - text_w) / 2
                text_y = y + (cell_h - text_h) / 2
                
                # テキスト背景 (最小限に)
                pad = 2
                draw.rectangle(
                    [text_x - pad, text_y - pad, text_x + text_w + pad, text_y + text_h + pad], 
                    fill=bg_color
                )
                draw.text((text_x, text_y), label, fill=text_color, font=font)

        # 最後の右端・下端の線のためにマーカーを追加描画するループは省略（視認性には影響小）

        out = Image.alpha_composite(img, overlay)
        
        output_path = f"{os.path.splitext(image_path)[0]}.grid.png"
        out.save(output_path)
        
        return output_path, rows, cols

