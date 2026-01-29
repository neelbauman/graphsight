from abc import ABC, abstractmethod
from typing import List, Tuple
from ..llm.base import BaseVLM
from ..models import Focus, StepInterpretation, OutputFormat, TokenUsage

class BaseStrategy(ABC):
    def __init__(self, output_format: OutputFormat = OutputFormat.MERMAID):
        self.output_format = output_format

    @property
    @abstractmethod
    def mermaid_type(self) -> str:
        pass

    @abstractmethod
    def find_initial_focus(self, vlm: BaseVLM, image_path: str) -> Tuple[List[Focus], TokenUsage]:
        pass

    @abstractmethod
    def interpret_step(self, vlm: BaseVLM, image_path: str, current_focus: Focus, context_history: List[str]) -> Tuple[StepInterpretation, TokenUsage]:
        pass
    
    @abstractmethod
    def synthesize(self, vlm: BaseVLM, image_path: str, extracted_texts: List[str], step_history: List[StepInterpretation]) -> Tuple[str, str, TokenUsage]:
        pass

