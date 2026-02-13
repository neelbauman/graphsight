import os
from pathlib import Path
from typing import Optional, Literal

from dotenv import load_dotenv
from loguru import logger

# Experimental Pipelines (必要に応じてコメントアウトを外して使えるように準備)
# from .pipelines.experimental.agentic import AgenticPipeline
# from .pipelines.experimental.ensemble import EnsemblePipeline

load_dotenv()

class GraphSight:
    """
    GraphSight API Client.
    
    This class provides a high-level interface to the GraphSight pipelines.
    By default, it uses the stable 'Draft -> Refine' architecture.
    """

    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None):
        """
        Initialize GraphSight client.

        Args:
            model (str): The OpenAI model to use (default: "gpt-4o").
            api_key (Optional[str]): OpenAI API Key. If None, uses OPENAI_API_KEY env var.
        """
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        
        if not os.getenv("OPENAI_API_KEY"):
            logger.warning("⚠️ OPENAI_API_KEY is not set. API calls may fail.")

        self.model = model
        # デフォルトでStableパイプラインを初期化
        self.pipeline = DraftRefinePipeline(model=model)

    def interpret(
        self, 
        image_path: str, 
        pipeline: Literal["standard"] = "standard"
    ) -> str:
        """
        Interpret a flowchart image and convert it to Mermaid code.

        Args:
            image_path (str): Path to the image file.
            pipeline (str): Pipeline strategy to use. Currently only "standard" is fully supported via API.

        Returns:
            str: Generated Mermaid diagram code.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # 将来的な拡張性: pipeline引数で実験的パイプラインに切り替え可能にする余地を残す
        if pipeline != "standard":
            logger.warning(f"Pipeline '{pipeline}' is not currently exposed via standard API. Using standard.")

        # 実行
        logger.info(f"🚀 GraphSight processing: {path.name} (Model: {self.model})")
        mermaid_code = self.pipeline.run(str(path))
        
        return mermaid_code

# 関数ベースで手軽に使いたい場合のためのショートカット
def interpret(image_path: str, model: str = "gpt-4o") -> str:
    client = GraphSight(model=model)
    return client.interpret(image_path)

