import base64
import math
import string
import os
import tempfile
import re


from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple


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


def crop_connection_area(image_path: str, bbox_a: List[int], bbox_b: List[int], padding: int = 50) -> str:
    """
    2つのノード(BBox [ymin, xmin, ymax, xmax]) を含む矩形領域を切り出し、
    一時ファイルのパスを返す。
    """
    try:
        with Image.open(image_path) as img:
            w, h = img.size
            
            # 0-1000 scale to pixels
            def to_px(bbox):
                return [
                    int(bbox[1] * w / 1000), # xmin
                    int(bbox[0] * h / 1000), # ymin
                    int(bbox[3] * w / 1000), # xmax
                    int(bbox[2] * h / 1000)  # ymax
                ]
            
            b1 = to_px(bbox_a)
            b2 = to_px(bbox_b)
            
            # Union Rectangle
            x_min = max(0, min(b1[0], b2[0]) - padding)
            y_min = max(0, min(b1[1], b2[1]) - padding)
            x_max = min(w, max(b1[2], b2[2]) + padding)
            y_max = min(h, max(b1[3], b2[3]) + padding)
            
            # Crop
            cropped = img.crop((x_min, y_min, x_max, y_max))
            
            # --- FIX: RGBA -> RGB Conversion ---
            # JPEGは透明度(Alpha)をサポートしていないため、RGBに変換する
            if cropped.mode in ("RGBA", "P"):
                cropped = cropped.convert("RGB")
            
            # Save temp
            tf = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            cropped.save(tf.name, quality=95)
            return tf.name

    except Exception as e:
        # 失敗時は元の画像を返す（フォールバック）
        # logger等があればログ出力推奨
        print(f"Crop failed: {e}")
        return image_path


def parse_grid_ref(ref: str) -> Tuple[int, int]:
    """
    "A1" -> (0, 0), "B2" -> (1, 1)
    Returns (row_index, col_index)
    """
    match = re.match(r"([A-Z]+)(\d+)", ref.upper())
    if not match:
        return (0, 0)
    
    row_str, col_str = match.groups()
    
    row_idx = 0
    for char in row_str:
        row_idx = row_idx * 26 + (ord(char) - ord('A') + 1)
    row_idx -= 1
    
    col_idx = int(col_str) - 1
    
    return row_idx, col_idx

def crop_grid_area(
    image_path: str, 
    grid_refs_a: List[str], 
    grid_refs_b: List[str], 
    total_rows: int, 
    total_cols: int,
    margin_cells: int = 1  # <-- NEW: 周囲何マス分広げるか (デフォルト1=周辺8マスを含む)
) -> str:
    """
    Grid参照リストに基づき、2つのノードを含む領域＋周囲のグリッドセルをマージンとして切り出す。
    """
    try:
        with Image.open(image_path) as img:
            w, h = img.size
            
            # 1セルあたりのピクセルサイズ
            cell_w = w / total_cols
            cell_h = h / total_rows
            
            # 全ての関連グリッドを集める
            all_refs = (grid_refs_a or []) + (grid_refs_b or [])
            
            if not all_refs:
                # 参照がない場合は画像全体を返すなどの安全策
                return image_path
            
            # 最小/最大インデックスを計算
            rows = []
            cols = []
            for ref in all_refs:
                r, c = parse_grid_ref(ref)
                rows.append(r)
                cols.append(c)
            
            min_r, max_r = min(rows), max(rows)
            min_c, max_c = min(cols), max(cols)
            
            # --- Grid Margin Expansion ---
            # 指定されたセル数分だけインデックスを広げる（画像の範囲内に収める）
            min_r = max(0, min_r - margin_cells)
            max_r = min(total_rows - 1, max_r + margin_cells)
            
            min_c = max(0, min_c - margin_cells)
            max_c = min(total_cols - 1, max_c + margin_cells)
            # -----------------------------
            
            # ピクセル座標へ変換
            # x_max, y_max は「次のセルの開始位置」なので +1 していることに注意
            x_min = int(min_c * cell_w)
            y_min = int(min_r * cell_h)
            x_max = int((max_c + 1) * cell_w)
            y_max = int((max_r + 1) * cell_h)
            
            # クロップ実行
            cropped = img.crop((x_min, y_min, x_max, y_max))
            
            # JPEG互換処理
            if cropped.mode in ("RGBA", "P"):
                cropped = cropped.convert("RGB")
                
            tf = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            cropped.save(tf.name, quality=95)
            return tf.name

    except Exception as e:
        print(f"Grid Crop failed: {e}")
        return image_path

