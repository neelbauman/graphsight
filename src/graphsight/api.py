import os
from loguru import logger
from dotenv import load_dotenv
from .llm.openai_client import OpenAIVLM
from .core.engine import GraphInterpreter
from .strategies.flowchart import FlowchartStrategy
from .classifier.detector import DiagramDetector
from .models import OutputFormat, DiagramResult, DiagramType

load_dotenv()

class GraphSight:
    def __init__(self, api_key: str | None = None, model: str = "gpt-4o"):
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("API Key is missing.")
        
        self.vlm = OpenAIVLM(api_key=key, model=model)
        self.engine = GraphInterpreter(self.vlm)
        self.detector = DiagramDetector(self.vlm)

    def interpret(self, image_path: str, format: str = "mermaid") -> DiagramResult:
        try:
            output_fmt = OutputFormat(format)
        except ValueError:
            output_fmt = OutputFormat.MERMAID

        detected_type, detector_usage = self.detector.detect(image_path)
        
        # Strategy selection
        strategy = FlowchartStrategy(output_format=output_fmt)
        
        return self.engine.process(image_path, strategy, initial_usage=detector_usage)
