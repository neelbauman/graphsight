from typing import List, Tuple
from pydantic import BaseModel
from .base import BaseStrategy, OutputFormat
from ..llm.base import BaseVLM
from ..llm.config import get_model_config, ModelType
from ..models import Focus, StepInterpretation, TokenUsage, InitialFocusList

# InitialFocusListは models.py に定義がない場合ここで定義
# ある場合は import して使う
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
        # グリッドモードかどうかに応じて要求する位置情報を変える
        if self.use_grid:
            loc_instr = "3. **Location**: Provide BOTH `grid_refs` (all overlapping cells e.g. ['A1', 'B1']) AND `bbox` (0-1000)."
        else:
            loc_instr = "3. **Location**: Provide `bbox` [ymin, xmin, ymax, xmax] (0-1000)."
        
        prompt = (
            "Analyze this flowchart image to find the starting points.\n"
            "1. **Scan** the top or left side for 'Start' terminators or initial process blocks.\n"
            "2. **Identify** them visually (shape, text).\n"
            "3. **ID Generation**: Create a descriptive ID (e.g. `node_Start_Process`). **DO NOT** use generic numbers like `node_1`.\n"
            f"{loc_instr}\n"
            "5. Return the list of all detected start nodes."
        )
        result, usage = vlm.query_structured(prompt, image_path, InitialFocusList)
        return result.start_nodes, usage

    def interpret_step(self, vlm: BaseVLM, image_path: str, current_focus: Focus, context_history: List[str]) -> Tuple[StepInterpretation, TokenUsage]:
        # 直近の履歴を取得（文脈理解用）
        history_text = "\n".join(context_history[-15:])
        config = get_model_config(vlm.model_name)
        
        # モードに応じたロケーション情報の文字列化とルール選択
        if self.use_grid:
            loc_str = f"Location: Grid={current_focus.grid_refs}"
            rules = self._build_hybrid_rules()
            # Contextに含まれるグリッド情報の読み方をAIに教える
            context_note = "(Note: '%% Grid: ...' in Context indicates the spatial location of past nodes. Use this to distinguish identical nodes in different places.)"
        else:
            loc_str = f"Location: BBox={current_focus.bbox}"
            rules = self._build_bbox_rules()
            context_note = "(Note: Check '%% BBox: ...' in Context to identify if we are looping back to a visited node.)"

        # モデルタイプ（推論系 vs 認識系）に応じたプロンプト分岐
        if config.model_type == ModelType.REASONING:
            prompt = self._build_reasoning_prompt(current_focus, history_text, loc_str, rules, context_note)
        else:
            prompt = self._build_recognition_prompt(current_focus, history_text, loc_str, rules, context_note)

        return vlm.query_structured(prompt, image_path, StepInterpretation)

    # --- Prompt Building Helpers ---

    def _build_bbox_rules(self) -> str:
        return """
        - **Spatial Info**: Provide `bbox` [ymin, xmin, ymax, xmax] (0-1000) for EVERY connected node.
        - Accuracy is vital for distinguishing identical text nodes in different locations.
        """
    
    def _build_hybrid_rules(self) -> str:
        return """
        - **Dual Spatial Info (CRITICAL)**:
          1. **grid_refs**: List **ALL** grid labels (e.g. ["C3", "C4", "D3"]) that overlap with the NEXT node's body.
          2. **bbox**: Also provide the estimated bounding box [ymin, xmin, ymax, xmax] (0-1000).
        - We use BOTH criteria to distinguish nodes. Be precise.
        """

    def _build_recognition_prompt(self, current_focus: Focus, history_text: str, loc_str: str, rules: str, context_note: str) -> str:
        # GPT-4o系: 視覚的な思考プロセス(Visual CoT)を強制し、意味的バイアスを防ぐ
        return f"""
        You are an expert Flowchart Crawler.
        
        # Current Focus Area
        - Description: "{current_focus.description}"
        - ID: "{current_focus.suggested_id}"
        - {loc_str}
        
        # Context (Path History)
        {history_text}
        {context_note}

        # INSTRUCTIONS (Visual Chain-of-Thought)
        
        ## Step 1: Visual Observation
        - Explicitly describe the node's **SHAPE** (Diamond, Rectangle, etc.) and **TEXT**.
        - If it is a **Diamond (Decision)**, look for ALL outgoing branches (Yes/No).
        
        ## Step 2: Trace Arrows (CRITICAL)
        - Trace each outgoing line physically with your eyes to the **EXACT** destination.
        - **WARNING: DO NOT ASSUME based on text.**
          - Even if the current node has the same text as a previous node (e.g. "Error"), the connections might be completely different because it is in a different location.
          - Follow the line pixel-by-pixel. Does it go Up? Down? Cross other lines?
          - Verify where the arrow head actually lands.

        ## Step 3: Extract Connections
        - List ALL directly connected nodes.
        - **ID Naming**: Use descriptive IDs (e.g. `node_Check_Stock`).
          - **FORBIDDEN**: Do NOT use sequential numbers like `node_1`, `node_2`.
          - **Consistency**: If a path loops back to a node that explicitly exists in "Context" (check ID and Location), reuse that ID.
        {rules}
        """

    def _build_reasoning_prompt(self, current_focus: Focus, history_text: str, loc_str: str, rules: str, context_note: str) -> str:
        # o1系: ゴールと制約条件を明確にする
        return f"""
        Analyze the flowchart and extract all outgoing connections from the target node.

        # Target Node
        - ID: "{current_focus.suggested_id}"
        - Description: "{current_focus.description}"
        - {loc_str}

        # Context
        {history_text}
        {context_note}

        # Goal
        Identify every node that is directly connected via an outgoing arrow from the Target Node.
        
        # Critical Requirements
        1. **Physical Tracing**: Ignore semantic expectations. Nodes with same text can have different logic. Follow lines strictly.
        2. **Completeness**: If this is a Decision node, you MUST find all branches (Yes/No).
        3. **Descriptive IDs**: Use meaningful IDs (e.g. `node_Submit`). No generic `node_1`.
        4. **Spatial Accuracy**: The location info must match the DESTINATION node.
        {rules}
        
        Output strictly matching the schema.
        """

    def synthesize(self, vlm: BaseVLM, image_path: str, extracted_texts: List[str], step_history: List[StepInterpretation]) -> Tuple[str, str, TokenUsage]:
        # 1. Mechanical Synthesis (Raw)
        # engine.py で生成された Mermaid コード（グリッドコメント付き）を整理
        seen = set()
        unique_lines = []
        for line in extracted_texts:
            clean = line.strip()
            if clean and clean not in seen:
                unique_lines.append(clean)
                seen.add(clean)

        raw_content = ""
        if self.output_format == OutputFormat.MERMAID:
            raw_content = "graph TD\n    " + "\n    ".join(unique_lines)
        else:
            raw_content = "\n".join(unique_lines)

        # 2. Build Investigation Log
        # AIがたどった物理的な軌跡をログとして構築
        investigation_log = ""
        for i, step in enumerate(step_history):
            investigation_log += f"Step {i+1}:\n"
            investigation_log += f"  - Observation: {step.visual_observation}\n"
            investigation_log += f"  - Tracing: {step.arrow_tracing}\n" # トレース過程も重要
            investigation_log += "  - Connections Found:\n"
            for edge in step.outgoing_edges:
                 # ログにもグリッド情報を残す（Refinementのヒントになる）
                 loc_info = f"Grid: {edge.grid_refs}" if self.use_grid else f"BBox: {edge.bbox}"
                 investigation_log += f"    -> {edge.target_id} [{edge.edge_label or 'flow'}] ({loc_info})\n"
            investigation_log += "\n"

        # 3. AI Refinement
        prompt = f"""
        Refine the fragmented flowchart data into a single, valid, and clean Mermaid flowchart.

        # Input Data
        1. **The Image**: Use the provided image (with Grid overlay) as the ULTIMATE GROUND TRUTH.
        2. **Investigation Log**: The step-by-step tracing history.
        3. **Raw Data**: The mechanically assembled graph (may contain fragments).
        
        # Investigation Log (The Physical Truth)
        This log records what was actually seen and traced step-by-step. Trust this over your semantic assumptions.
        {investigation_log}
        
        # Raw Mechanical Output (Contains '%%' location comments)
        {raw_content}
        
        # Instructions
        1. **Reconstruct Logic**: Ensure the flow makes logical sense based on the Log.
        2. **Clean Up**: Remove all '%% Grid: ...' or '%% BBox: ...' comments from the final output.
        3. **Fix Syntax**: Ensure valid Mermaid syntax.
        4. **Merge/Split**: If the log shows two nodes with same text but different locations/logic, keep them separate (or use distinct IDs). If they are truly the same (loop back), merge them.
        5. Output ONLY the Mermaid code block.
        """
        
        refined_content, usage = vlm.query_text(prompt, image_path=image_path)
        
        return refined_content, raw_content, usage

