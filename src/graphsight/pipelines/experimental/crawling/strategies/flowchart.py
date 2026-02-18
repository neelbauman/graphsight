"""
graphsight.strategies.flowchart
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Standard strategy for Flowcharts: Simple DFS Approach.
"""

import os
from typing import List, Tuple
from pydantic import BaseModel
from loguru import logger
from .base import BaseStrategy, OutputFormat
from ..llm.base import BaseVLM
from ..llm.config import get_model_config, ModelType
from ..models import Focus, StepInterpretation, TokenUsage, InitialFocusList
from graphsight.utils.geometry import calculate_relative_direction

if 'InitialFocusList' not in globals():
    class InitialFocusList(BaseModel):
        start_nodes: List[Focus]

class FlowchartStrategy(BaseStrategy):
    def __init__(self, output_format: OutputFormat = OutputFormat.MERMAID, use_grid: bool = False):
        super().__init__(output_format)
        self.use_grid = use_grid

    @property
    def mermaid_type(self) -> str:
        return "flowchart"

    def find_initial_focus(self, vlm: BaseVLM, image_path: str) -> Tuple[List[Focus], TokenUsage]:
        # シンプルに「スタートはどこ？」と聞く
        prompt = (
            "Look at this flowchart.\n"
            "**Question**: What is the **Start Node** of this process?\n\n"
            "Instructions:\n"
            "1. Find the node that starts the flow (usually 'Start', or a top/left node without incoming arrows).\n"
            "2. Provide a unique ID (e.g. `node_Start`) and its Bounding Box.\n"
            "3. Return it in the list."
        )
        result, usage = vlm.query_structured(prompt, image_path, InitialFocusList)
        return result.start_nodes, usage

    def interpret_step(
        self, 
        vlm: BaseVLM, 
        image_path: str, 
        current_focus: Focus, 
        context_history: List[StepInterpretation]
    ) -> Tuple[StepInterpretation, TokenUsage]:
        
        # 履歴をテキスト化してコンテキストとして渡す
        history_text = self._build_history_text(context_history)
        
        # シンプルに「次は？」と聞く
        prompt = f"""
        You are traversing a flowchart step-by-step.
        
        # Current Location
        - We are currently at Node: **"{current_focus.suggested_id}"**
        - Description: "{current_focus.description}"
        - Location (BBox): {current_focus.bbox}
        
        # Task
        **Question**: From THIS node, where do the lines go next?
        
        1. **Trace**: Follow the outgoing lines from the current node visually.
        2. **Identify**: List all immediate next nodes connected by arrows.
        3. **Describe**: For each next node, provide a suggested ID and BBox.
        4. **End**: If there are no outgoing lines (end of flow), return an empty list for outgoing edges.

        # Context (Already visited)
        {history_text}
        """

        return vlm.query_structured(prompt, image_path, StepInterpretation)

    def _build_history_text(self, history: List[StepInterpretation]) -> str:
        if not history:
            return "(No steps taken yet. This is the first node.)"
        
        lines = []
        # 直近の履歴のみを表示してコンテキスト溢れを防ぐ（必要に応じて調整）
        recent_steps = history[-10:] 
        for step in recent_steps:
            src = step.source_id or "Unknown"
            connections = []
            for edge in step.outgoing_edges:
                label = f"|{edge.edge_label}|" if edge.edge_label else ""
                connections.append(f"--{label}--> {edge.target_id}")
            
            if connections:
                lines.append(f"{src} {' '.join(connections)}")
            else:
                lines.append(f"{src} (End)")
                
        return "\n".join(lines)

    # 既存のsynthesizeメソッド等はそのまま利用（または必要に応じて簡略化）
    def synthesize(
        self, 
        vlm: BaseVLM, 
        image_path: str, 
        extracted_texts: List[str],
        step_history: List[StepInterpretation]
    ) -> Tuple[str, str, TokenUsage]:
        
        # 単純に履歴からGraphを構築
        unique_lines = set()
        
        for step in step_history:
            src = step.source_id or "Unknown"
            for edge in step.outgoing_edges:
                arrow = "-->"
                if edge.edge_label:
                    arrow = f"-->|{edge.edge_label}|"
                line = f"{src} {arrow} {edge.target_id}"
                unique_lines.add(line)

        raw_content = "graph TD\n    " + "\n    ".join(sorted(unique_lines))
        
        # 仕上げのLLM呼び出し（整形用）
        prompt = f"""
        Convert this raw connection list into clean Mermaid.js code.
        Fix IDs to be alphanumeric. Keep labels.
        
        {raw_content}
        """
        refined_content, usage = vlm.query_text(prompt, image_path=None) # 画像なしで高速化
        
        return refined_content, raw_content, usage

