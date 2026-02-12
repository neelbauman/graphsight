"""
graphsight.strategies.base
~~~~~~~~~~~~~~~~~~~~~~~~~~
Abstract Base Class for all interpretation strategies.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from ..llm.base import BaseVLM
from ..models import Focus, StepInterpretation, TokenUsage, DiagramResult, OutputFormat

class BaseStrategy(ABC):
    def __init__(self, output_format: OutputFormat = OutputFormat.MERMAID):
        self.output_format = output_format
        # Grid dimensions (Engineからセットされる)
        self.grid_rows: int = 0
        self.grid_cols: int = 0
    
    def set_grid_dimensions(self, rows: int, cols: int):
        self.grid_rows = rows
        self.grid_cols = cols

    @property
    @abstractmethod
    def mermaid_type(self) -> str:
        """Returns the Mermaid diagram type string (e.g. 'flowchart', 'sequenceDiagram')."""
        pass

    @abstractmethod
    def find_initial_focus(self, vlm: BaseVLM, image_path: str) -> Tuple[List[Focus], TokenUsage]:
        """Identifies the starting points of the diagram."""
        pass

    @abstractmethod
    def interpret_step(
        self, 
        vlm: BaseVLM, 
        image_path: str, 
        current_focus: Focus, 
        context_history: List[StepInterpretation]
    ) -> Tuple[StepInterpretation, TokenUsage]:
        """Analyzes a single node and finds outgoing connections."""
        pass

    def reexamine_step(
        self,
        vlm: BaseVLM,
        image_path: str,
        current_focus: Focus,
        context_history: List[StepInterpretation],
        previous_result: StepInterpretation,
        warnings: List[str]
    ) -> Tuple[StepInterpretation, TokenUsage]:
        """
        Re-analyzes a step based on validation warnings.
        Default implementation just calls interpret_step, but strategies should override this.
        """
        return self.interpret_step(vlm, image_path, current_focus, context_history)

    def audit_node(
        self,
        vlm: BaseVLM,
        image_path: str,
        current_focus: Focus,
        context_history: List[StepInterpretation],
        proposed_incoming: List[str],  # 今のグラフで「入ってくる」ことになっているIDリスト
        proposed_outgoing: List[str]   # 今のグラフで「出ていく」ことになっているIDリスト
    ) -> Tuple[StepInterpretation, TokenUsage]:
        """
        Global Integrity Check:
        提示された In/Out リストが、画像の実態と合っているか最終確認する。
        """
        # デフォルトはスルー（何もしない）だが、FlowchartStrategyで実装する
        return self.interpret_step(vlm, image_path, current_focus, context_history)

    @abstractmethod
    def synthesize(
        self, 
        vlm: BaseVLM, 
        image_path: str, 
        extracted_texts: List[str], 
        step_history: List[StepInterpretation]
    ) -> Tuple[str, str, TokenUsage]:
        """Synthesizes the final output from the interpretation history."""
        pass

