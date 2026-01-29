from typing import List, Tuple
from pydantic import BaseModel
from .base import BaseStrategy, OutputFormat
from ..llm.base import BaseVLM
from ..llm.config import get_model_config, ModelType
from ..models import Focus, StepInterpretation, TokenUsage

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
        if self.use_grid:
            loc_instr = "3. **Location**: Provide BOTH `grid_refs` (all overlapping cells) AND `bbox` (0-1000)."
        else:
            loc_instr = "3. **Location**: Provide `bbox` [ymin, xmin, ymax, xmax] (0-1000)."
        
        prompt = (
            "Analyze this flowchart to find start nodes.\n"
            "1. Scan for 'Start' terminators.\n"
            "2. Identify them visually.\n"
            "3. **ID Generation**: Create a descriptive ID (e.g. `node_Start_Process`). DO NOT use generic numbers like `node_1`.\n"
            f"{loc_instr}\n"
            "5. Return list."
        )
        result, usage = vlm.query_structured(prompt, image_path, InitialFocusList)
        return result.start_nodes, usage

    def interpret_step(self, vlm: BaseVLM, image_path: str, current_focus: Focus, context_history: List[str]) -> Tuple[StepInterpretation, TokenUsage]:
        history_text = "\n".join(context_history[-15:])
        config = get_model_config(vlm.model_name)
        
        # コンテキストの読み方ガイダンス
        if self.use_grid:
            loc_str = f"Location: Grid={current_focus.grid_refs}"
            rules = self._build_hybrid_rules()
            context_note = "(Note: '%% Grid: ...' in Context indicates spatial location. Use this to detect loops to existing nodes.)"
        else:
            loc_str = f"Location: BBox={current_focus.bbox}"
            rules = self._build_bbox_rules()
            context_note = "(Note: Check '%% BBox: ...' in Context to identify if we are looping back to a visited node.)"

        if config.model_type == ModelType.REASONING:
            prompt = self._build_reasoning_prompt(current_focus, history_text, loc_str, rules, context_note)
        else:
            prompt = self._build_recognition_prompt(current_focus, history_text, loc_str, rules, context_note)

        return vlm.query_structured(prompt, image_path, StepInterpretation)

    def _build_bbox_rules(self) -> str:
        return """
        - **Spatial Info**: Provide `bbox` [ymin, xmin, ymax, xmax] (0-1000) for EVERY connected node.
        """
    
    def _build_hybrid_rules(self) -> str:
        return """
        - **Dual Spatial Info**:
          1. **grid_refs**: List **ALL** grid labels overlapping the NEXT node.
          2. **bbox**: Also provide estimated [ymin, xmin, ymax, xmax] (0-1000).
        """

    def _build_recognition_prompt(self, current_focus: Focus, history_text: str, loc_str: str, rules: str, context_note: str) -> str:
        return f"""
        You are an expert Flowchart Crawler.
        # Current Focus
        - Description: "{current_focus.description}"
        - ID: "{current_focus.suggested_id}"
        - {loc_str}
        
        # Context
        {history_text}
        {context_note}

        # INSTRUCTIONS
        ## Step 1: Visual Observation
        - Describe SHAPE and TEXT.
        - If Diamond, find ALL branches.
        
        ## Step 2: Trace Arrows
        - Trace lines physically.
        
        ## Step 3: Extract Connections
        - List directly connected nodes.
        - **ID Naming**: Use descriptive IDs based on node text (e.g. `node_Check_Stock`). 
          - **FORBIDDEN**: Do NOT use numeric IDs (node_1, node_2) unless the text is literally a number.
          - Consistency: If a node seems to be in the Context (same text & location), reuse that ID.
        {rules}
        """

    def _build_reasoning_prompt(self, current_focus: Focus, history_text: str, loc_str: str, rules: str, context_note: str) -> str:
        return f"""
        Analyze flowchart and extract connections.
        # Target
        - ID: "{current_focus.suggested_id}"
        - {loc_str}
        # Context
        {history_text}
        {context_note}
        # Goal
        Identify every connected node.
        # Requirements
        1. Completeness (all branches).
        2. Descriptive IDs (e.g. `node_Submit`). No sequential numbers.
        3. Spatial Accuracy.
        {rules}
        Output strictly matching schema.
        """

    def synthesize(self, vlm: BaseVLM, extracted_texts: List[str], step_history: List[StepInterpretation]) -> Tuple[str, str, TokenUsage]:
        seen = set()
        unique_lines = []
        for line in extracted_texts:
            clean = line.strip()
            if clean and clean not in seen:
                unique_lines.append(clean)
                seen.add(clean)

        raw_content = "graph TD\n    " + "\n    ".join(unique_lines) if self.output_format == OutputFormat.MERMAID else "\n".join(unique_lines)

        investigation_log = ""
        for i, step in enumerate(step_history):
            investigation_log += f"Step {i+1}:\n"
            investigation_log += f"  Obs: {step.visual_observation}\n"
            for edge in step.outgoing_edges:
                 loc_info = f"Grid: {edge.grid_refs}" if self.use_grid else f"BBox: {edge.bbox}"
                 investigation_log += f"    -> {edge.target_id} [{edge.edge_label}] ({loc_info})\n"
            investigation_log += "\n"

        prompt = f"""
        Refine the flowchart data into valid Mermaid code.
        # Log
        {investigation_log}
        # Raw Data
        {raw_content}
        # Instructions
        1. Ensure logical flow.
        2. Remove '%%' comments.
        3. Fix syntax.
        4. Output ONLY Mermaid code.
        """
        refined_content, usage = vlm.query_text(prompt, image_path=None)
        return refined_content, raw_content, usage

