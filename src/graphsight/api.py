import os
from loguru import logger
from dotenv import load_dotenv
from .llm.openai_client import OpenAIVLM
from .core.engine import GraphInterpreter
from .strategies.flowchart import FlowchartStrategy
from .strategies.fast_flowchart import FastFlowchartStrategy
from .strategies.structured import StructuredFlowchartStrategy
from .classifier.detector import DiagramDetector
from .models import OutputFormat, DiagramResult

load_dotenv()

class GraphSight:
    def __init__(self, api_key: str | None = None, model: str = "gpt-4o"):
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("API Key is missing.")
        
        self.vlm = OpenAIVLM(api_key=key, model=model)
        self.engine = GraphInterpreter(self.vlm)
        self.detector = DiagramDetector(self.vlm)

    def interpret(
        self, 
        image_path: str, 
        format: str = "mermaid", 
        experimental_grid: bool = False, 
        strategy_mode: str = "standard",
        traversal_mode: str = "dfs"
    ) -> DiagramResult:
        
        try:
            output_fmt = OutputFormat(format)
        except ValueError:
            output_fmt = OutputFormat.MERMAID

        detected_type, detector_usage = self.detector.detect(image_path)
        
        # Strategy Selection
        strategy = None
        if strategy_mode == "fast":
            logger.info("🐇 Using FastFlowchartStrategy (Few-Shot Mode)")
            strategy = FastFlowchartStrategy(output_format=output_fmt, use_grid=experimental_grid)
        elif strategy_mode == "structured":
            logger.info("🏗️ Using StructuredFlowchartStrategy (JSON Extraction Mode)")
            # StructuredもGridを受け取るようにする（BaseStrategyにはないが、Duck typingで渡す、あるいはコンストラクタで受ける）
            # 今回のStructuredFlowchartStrategy実装では use_grid を明示的に属性セットする
            strategy = StructuredFlowchartStrategy(output_format=output_fmt)
            strategy.use_grid = experimental_grid # 属性注入
        else:
            logger.info("🐢 Using Standard FlowchartStrategy (Reasoning Mode)")
            strategy = FlowchartStrategy(output_format=output_fmt, use_grid=experimental_grid)
        
        return self.engine.process(
            image_path, 
            strategy, 
            initial_usage=detector_usage,
            traversal_mode=traversal_mode
        )

