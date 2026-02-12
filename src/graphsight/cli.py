import typer
from pathlib import Path
from .agent.core import GraphSightAgent

app = typer.Typer()

@app.command()
def parse(
    image_path: str = typer.Argument(..., help="Path to the flowchart image"),
    output: str = typer.Option(None, "--output", "-o", help="Output file path (.mmd)"),
    model: str = typer.Option("gpt-4o", help="OpenAI Model to use"),
):
    """
    Interpret a flowchart image and convert it to Mermaid.js code.
    Uses an autonomous agent with computer vision capabilities (GraphSight Agent).
    """
    if not Path(image_path).exists():
        typer.secho(f"Error: Image file not found at {image_path}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    try:
        # エージェントの初期化
        typer.echo(f"🤖 Initializing GraphSight Agent with {model}...")
        agent = GraphSightAgent(model=model)
        
        typer.echo(f"👀 Analyzing {image_path}...")
        typer.echo("   (The agent is planning and inspecting the diagram...)")
        
        # エージェント実行 (Plan -> Think -> Tool -> Finish)
        mermaid_code = agent.run(image_path)
        
        # 結果出力
        typer.echo(f"\n✨ --- Generated Mermaid Code ---")
        typer.echo(mermaid_code)
        
        # ファイル保存
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(mermaid_code, encoding="utf-8")
            typer.echo(f"\n💾 Saved result to {output_path}")
            
    except Exception as e:
        typer.secho(f"\n❌ Processing failed: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()

