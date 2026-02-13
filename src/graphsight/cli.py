import typer
from enum import Enum
from pathlib import Path
from typing import Optional

# Pipelines
from .pipelines.stable.draft_refine import DraftRefinePipeline
from .pipelines.experimental.agentic import AgenticPipeline
from .pipelines.experimental.ensemble import EnsemblePipeline
# from .pipelines.experimental.crawling import CrawlingPipeline # 必要ならimport

app = typer.Typer()

class PipelineType(str, Enum):
    # Stable
    STANDARD = "standard"  # Default (Draft-Refine)
    
    # Experimental
    EXP_AGENT = "exp-agent"
    EXP_ENSEMBLE = "exp-ensemble"
    EXP_CRAWL = "exp-crawl"

@app.command()
def parse(
    image_path: str = typer.Argument(..., help="Path to the flowchart image"),
    output: str = typer.Option(None, "--output", "-o", help="Output file path (.mmd)"),
    model: str = typer.Option("gpt-4o", help="OpenAI Model to use"),
    pipeline: PipelineType = typer.Option(
        PipelineType.STANDARD, 
        "--pipeline", "-p",
        help="Select processing logic. 'exp-' options are experimental."
    ),
):
    """
    Interpret a flowchart image and convert it to Mermaid.js code.
    """
    if not Path(image_path).exists():
        typer.secho(f"Error: Image file not found at {image_path}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # 実験的機能の警告
    if pipeline.value.startswith("exp-"):
        typer.secho(f"⚠️  Using EXPERIMENTAL pipeline: {pipeline.value}", fg=typer.colors.YELLOW)

    # Pipeline Factory
    try:
        if pipeline == PipelineType.STANDARD:
            runner = DraftRefinePipeline(model=model)
        elif pipeline == PipelineType.EXP_AGENT:
            runner = AgenticPipeline(model=model)
        elif pipeline == PipelineType.EXP_ENSEMBLE:
            runner = EnsemblePipeline(model=model)
        elif pipeline == PipelineType.EXP_CRAWL:
            # runner = CrawlingPipeline(model=model)
            typer.secho("Experimental Crawl pipeline is under maintenance.", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        
        typer.echo(f"🤖 Analyzing {image_path}...")
        mermaid_code = runner.run(image_path)
        
        # 結果出力
        typer.echo(f"\n✨ --- Generated Mermaid Code ---")
        typer.echo(mermaid_code)
        
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(mermaid_code, encoding="utf-8")
            typer.echo(f"\n💾 Saved result to {output_path}")

    except Exception as e:
        typer.secho(f"\n❌ Processing failed: {e}", fg=typer.colors.RED)
        # raise e # デバッグ時はコメントアウトを外す
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()

