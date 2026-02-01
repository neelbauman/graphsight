from typing import Tuple
from loguru import logger
from ..llm.base import BaseVLM
from ..models import DiagramType, TokenUsage, ClassificationResult

class DiagramDetector:
    def __init__(self, vlm: BaseVLM):
        self.vlm = vlm

    def detect(self, image_path: str) -> Tuple[DiagramType, TokenUsage]:
        logger.info("🕵️  Detecting diagram type...")
        
        prompt = """
        Analyze the provided image and classify the diagram type.
        Options: flowchart, sequenceDiagram, classDiagram, erDiagram, unknown.
        Return the detected type and a brief reasoning.
        """
        
        try:
            # 同期呼び出し
            result, usage = self.vlm.query_structured(prompt, image_path, ClassificationResult)
            
            logger.info(f"✅ Type Detected: {result.diagram_type.name}")
            logger.debug(f"   Reason: {result.reasoning}")
            
            return result.diagram_type, usage

        except Exception as e:
            logger.warning(f"Detection failed or timed out: {e}. Defaulting to FLOWCHART.")
            return DiagramType.FLOWCHART, TokenUsage()

