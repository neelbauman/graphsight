import sys
import os
import typer
from pathlib import Path

# srcディレクトリをPythonパスに追加して、graphsightパッケージを直接import可能にする
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

try:
    from graphsight.utils.image import add_grid_overlay
except ImportError:
    print("Error: Could not import 'graphsight'. Make sure you are in the project root.")
    sys.exit(1)

def main(
    image_path: str = typer.Argument(..., help="Path to the input image file"),
    cell_size: int = typer.Option(150, "--size", "-s", help="Minimum grid cell size in pixels")
):
    """
    Generate and preview the SoM (Set-of-Mark) grid overlay for a given image.
    This helps in tuning the cell_size for better LLM visibility.
    """
    file_path = Path(image_path)
    if not file_path.exists():
        typer.secho(f"❌ Error: File not found: {image_path}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.echo(f"🔍 Processing '{image_path}' with min_cell_size={cell_size}px...")

    try:
        # グリッド生成処理の実行
        output_path, rows, cols = add_grid_overlay(str(file_path), min_cell_size=cell_size)

        typer.secho(f"✅ Grid Generated Successfully!", fg=typer.colors.GREEN)
        typer.echo(f"   - Grid Dimensions: {rows} rows x {cols} cols")
        typer.echo(f"   - Total Cells: {rows * cols}")
        typer.echo(f"   - Saved to: {output_path}")

        # OSに合わせて画像を開く
        typer.echo("📂 Opening preview...")
        if sys.platform == "darwin":  # macOS
            os.system(f"open '{output_path}'")
        elif sys.platform == "win32": # Windows
            os.startfile(output_path)
        else: # Linux (Ubuntu/Debian etc)
            os.system(f"xdg-open '{output_path}'")

    except Exception as e:
        typer.secho(f"❌ Error generating grid: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

if __name__ == "__main__":
    typer.run(main)

