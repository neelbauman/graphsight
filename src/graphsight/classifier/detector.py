from typing import Tuple
from pydantic import BaseModel, Field
from loguru import logger
from ..llm.base import BaseVLM
from ..models import DiagramType, TokenUsage

class ClassificationResult(BaseModel):
    diagram_type: DiagramType = Field(..., description="判定されたMermaidの図版タイプ")
    reasoning: str = Field(..., description="なぜそのタイプだと判断したかの理由")

class DiagramDetector:
    def __init__(self, vlm: BaseVLM):
        self.vlm = vlm

    def detect(self, image_path: str) -> Tuple[DiagramType, TokenUsage]:
        logger.info("🕵️  Detecting diagram type...")
        
        prompt = """
        Analyze this image and classify the diagram type (flowchart, sequenceDiagram, etc).
        If uncertain, choose 'flowchart'.
        """
        
        try:
            result, usage = self.vlm.query_structured(prompt, image_path, ClassificationResult)
            
            logger.info(f"✅ Type Detected: {result.diagram_type.value.upper()}")
            logger.debug(f"   Reason: {result.reasoning}")
            
            return result.diagram_type, usage

        except Exception as e:
            logger.warning(f"⚠️ Detection failed: {e}")
            return DiagramType.FLOWCHART, TokenUsage()
