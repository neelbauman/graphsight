from typing import List, Tuple
from .flowchart import FlowchartStrategy
from ..llm.base import BaseVLM
from ..models import Focus, StepInterpretation, TokenUsage

class FastFlowchartStrategy(FlowchartStrategy):
    """
    GPT-4o-mini 向けに最適化された高速・低コスト戦略。
    - Instructionを極限まで減らし、Few-Shot（具体例）で誘導する。
    - Reasoning（思考ログ）を省略させる。
    - AIによる最終清書をスキップし、Rawデータをそのまま返す。
    """

    def interpret_step(self, vlm: BaseVLM, image_path: str, current_focus: Focus, context_history: List[str]) -> Tuple[StepInterpretation, TokenUsage]:
        # コンテキストは直近5件に絞ってトークン節約
        history_text = "\n".join(context_history[-5:])
        
        # Few-Shot Examples: 言葉で説明するより、例を見せる方が安いモデルには効く
        examples = """
        Example 1:
        Input Focus: "Decision Node 'Is Valid?'"
        Output: {
            "extracted_text": "node_Valid{Is Valid?} -->|Yes| node_ProcessA[Process A]\\nnode_Valid{Is Valid?} -->|No| node_End[End]",
            "next_focus_candidates": [
                {"description": "Process A node on the right", "suggested_id": "node_ProcessA"},
                {"description": "End node at the bottom", "suggested_id": "node_End"}
            ],
            "is_done": false
        }
        """

        prompt = f"""
        Analyze the flowchart focused at: "{current_focus.description}".
        Return the Mermaid connection syntax and next nodes.
        
        # Rules
        1. Use ID format: `node_SanitizedText`
        2. If node exists in Context, reuse ID.
        3. reasoning field can be null/empty.

        # Context
        {history_text}

        # Examples
        {examples}
        """
        
        return vlm.query_structured(prompt, image_path, StepInterpretation)

    def synthesize(self, vlm: BaseVLM, extracted_texts: List[str], step_history: List[StepInterpretation]) -> Tuple[str, str, TokenUsage]:
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

