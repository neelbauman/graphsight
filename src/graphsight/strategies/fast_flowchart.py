from typing import List, Tuple
from .flowchart import FlowchartStrategy
from ..llm.base import BaseVLM
from ..models import Focus, StepInterpretation, TokenUsage

class FastFlowchartStrategy(FlowchartStrategy):
    """
    GPT-4o / GPT-4o-mini 向けに最適化された高速・低コスト戦略。
    - Instructionを極限まで減らし、Few-Shot（具体例）で誘導する。
    - Reasoning（思考ログ）を省略させる。
    - Grid情報が有効な場合、座標情報をプロンプトに注入して精度を上げる。
    - AIによる最終清書をスキップし、Rawデータをそのまま返す。
    """

    def interpret_step(self, vlm: BaseVLM, image_path: str, current_focus: Focus, context_history: List[str]) -> Tuple[StepInterpretation, TokenUsage]:
        # コンテキストは直近5件に絞ってトークン節約
        history_text = "\n".join(context_history[-5:])
        
        # Gridモード判定とプロンプト注入
        if self.use_grid:
            loc_str = f"Location: Grid={current_focus.grid_refs} (Look at these cells)"
            # Gridモード用の追加ルール: 次のノードのグリッド位置を強く要求
            spatial_rule = "4. **Spatial**: For next nodes, you MUST provide `grid_refs` (e.g. ['C3', 'D3']). This is vital for loops."
        else:
            loc_str = f"Location: BBox={current_focus.bbox}"
            spatial_rule = ""

        # Few-Shot Examples (Gridの有無で少し例示を変えるとさらに良いが、ここでは汎用的に定義)
        examples = """
        Example 1:
        Input Focus: "Decision Node 'Is Valid?'"
        Output: {
            "extracted_text": "node_Valid{Is Valid?} -->|Yes| node_ProcessA[Process A]\\nnode_Valid{Is Valid?} -->|No| node_End[End]",
            "outgoing_edges": [
                {"description": "Process A node on the right", "suggested_id": "node_ProcessA", "grid_refs": ["E4"]},
                {"description": "End node at the bottom", "suggested_id": "node_End", "grid_refs": ["D6"]}
            ],
            "reasoning": ""
        }
        """

        prompt = f"""
        Analyze the flowchart focused at: "{current_focus.description}".
        Return the Mermaid connection syntax and next nodes.
        
        # Current Target
        - ID: "{current_focus.suggested_id}"
        - {loc_str}
        
        # Rules
        1. Use ID format: `node_SanitizedText`
        2. If node exists in Context, reuse ID.
        3. `reasoning` field can be null/empty.
        {spatial_rule}

        # Context (Recent Path)
        {history_text}

        # Examples
        {examples}
        """
        
        return vlm.query_structured(prompt, image_path, StepInterpretation)

    def synthesize(self, vlm: BaseVLM, image_path: str, extracted_texts: List[str], step_history: List[StepInterpretation]) -> Tuple[str, str, TokenUsage]:
        # Fastモードでは「AIによる清書（Refinement）」をスキップし、
        # 機械的に結合したものをそのまま返す（究極の高速化）
        
        # 重複排除のみ行う
        seen = set()
        unique_lines = []
        for line in extracted_texts:
            for sub in line.split('\n'):
                clean = sub.strip()
                if clean and clean not in seen:
                    unique_lines.append(clean)
                    seen.add(clean)

        raw_content = "graph TD\n    " + "\n    ".join(unique_lines)
        
        # Refinementなし。RawをそのままFinalとして返す。
        return raw_content, raw_content, TokenUsage()

