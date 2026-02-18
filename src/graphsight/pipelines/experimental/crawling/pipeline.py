from graphsight.pipelines.base import BasePipeline
from graphsight.llm.openai_client import OpenAIVLM

# Crawling Pipeline内のモジュール
from .engine import GraphInterpreter
from .strategies.flowchart import FlowchartStrategy

class CrawlingPipeline(BasePipeline):
    def __init__(self, model: str = "gpt-4o"):
        """
        Initialize the Crawling Pipeline.
        
        Args:
            model (str): The name of the LLM model to use (e.g., 'gpt-4o').
        """
        self.model_name = model

    def run(self, image_path: str) -> str:
        """
        Execute the crawling process on the flowchart image.

        Args:
            image_path (str): Path to the input image file.

        Returns:
            str: Generated Mermaid code.
        """
        # 1. Initialize VLM (Vision Language Model) Wrapper
        # プロジェクト共通のOpenAIクライアントを使用
        vlm = OpenAIVLM(model=self.model_name)
        
        # 2. Initialize Core Engine & Strategy
        # FlowchartStrategyはデフォルトでMermaid出力を指向します
        strategy = FlowchartStrategy(use_grid=False) 
        
        # EngineにVLMを注入
        engine = GraphInterpreter(vlm)
        
        # 3. Process the Image
        # Engine returns a DiagramResult object
        result = engine.process(image_path, strategy)
        
        # ログ出力（CLI標準出力への表示はcli.py側で行うが、デバッグ用に）
        # typer.echo(f"   💰 Estimated Cost: ${result.cost_usd:.4f}")
        
        # 4. Return just the content (Mermaid code)
        return result.content

