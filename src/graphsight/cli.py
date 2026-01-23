import typer
from .api import GraphSight

app = typer.Typer()

@app.command()
def parse(
    image_path: str, 
    output: str = typer.Option(None, help="Output file path"),
    format: str = typer.Option("mermaid", help="'mermaid' or 'natural_language'"),
    model: str = typer.Option("gpt-4o", help="Model to use (gpt-4o, gpt-4o-mini, gpt-5-preview...)")
):
    try:
        sight = GraphSight(model=model)
        result = sight.interpret(image_path, format=format)
        
        typer.echo(f"\n✨ --- AI Refined Result ({result.output_format.value}) ---")
        typer.echo(result.content)

        typer.echo(f"\n🔧 --- Raw Mechanical Result ---")
        typer.echo(result.raw_content)
        
        typer.echo("\n📊 --- Usage & Cost ---")
        typer.echo(f"Model: {result.model_name}")
        typer.echo(f"Total Tokens: {result.usage.total_tokens:,} (In: {result.usage.input_tokens:,}, Out: {result.usage.output_tokens:,})")
        typer.echo(f"Estimated Cost: ${result.cost_usd:.4f}")
        
        if output:
            with open(output, "w") as f:
                f.write(result.content)
            typer.echo(f"\nSaved refined result to {output}")
            
    except Exception as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED)

if __name__ == "__main__":
    app()
