from typing import List, Tuple
from .base import BaseStrategy, OutputFormat
from ..llm.base import BaseVLM
from ..models import Focus, StepInterpretation, TokenUsage, ConnectedNode, InitialFocusList

class StructuredFlowchartStrategy(BaseStrategy):
    """
    Structured Mode:
    LLMには「データの構造化（JSON抽出）」だけを依頼し、コード生成はPython側で行う戦略。
    Mermaid記法に関する指示を排除し、純粋な視覚認識能力を引き出す。
    """

    @property
    def mermaid_type(self) -> str:
        return "flowchart"

    def find_initial_focus(self, vlm: BaseVLM, image_path: str) -> Tuple[List[Focus], TokenUsage]:
        prompt = "Identify all 'Start' nodes or initial trigger points in this flowchart. Return their text and location."
        result, usage = vlm.query_structured(prompt, image_path, InitialFocusList)
        return result.start_nodes, usage

    def interpret_step(
        self, 
        vlm: BaseVLM, 
        image_path: str, 
        current_focus: Focus, 
        context_history: List[StepInterpretation]
    ) -> Tuple[StepInterpretation, TokenUsage]:
        
        # 1. ロケーションヒントの生成
        loc_hint = ""
        if current_focus.grid_refs:
            loc_hint = f"(Grid Cells: {current_focus.grid_refs})"
        elif current_focus.bbox:
            loc_hint = f"(BBox: {current_focus.bbox})"

        # 2. コンテキストから既知のIDリストを抽出 (重複排除とループ検知のため)
        known_ids = set()
        for step in context_history:
            # エンジンによって注入された source_id を参照
            if step.source_id:
                known_ids.add(step.source_id)
            # 検出されたターゲットIDも参照
            for edge in step.outgoing_edges:
                known_ids.add(edge.target_id)
        
        known_ids_list = sorted(list(known_ids))
        # トークン節約のため、直近または重要そうなIDのみ渡す調整も可能だが、
        # Structuredモードは精度重視のため全リスト（多すぎなければ）を渡す方針とする。
        known_ids_str = ", ".join(known_ids_list)

        # 3. プロンプト構築
        # LLMへの指示は「Pydanticモデル（StepInterpretation）を埋めろ」という暗黙の指示だけで機能するが、
        # IDの一貫性を保つために Context 情報を明示する。
        prompt = f"""
        Focus Node: "{current_focus.description}" {loc_hint}
        
        Context (Known IDs): [{known_ids_str if known_ids_str else "None"}]
        
        Task: Identify all direct OUTGOING connections from this node.
        
        For each connection, extract:
        1. target_id: Create a unique ID (e.g., node_Submit). Reuse IDs from 'Context' if the connection loops back to an existing node.
        2. description: The text inside the target node.
        3. edge_label: Text on the arrow (e.g., "Yes", "No", "Error"), or null.
        4. grid_refs: The grid location of the target node (Crucial for distinguishing identical nodes).
        """
        
        return vlm.query_structured(prompt, image_path, StepInterpretation)

    def synthesize(self, vlm: BaseVLM, image_path: str, extracted_texts: List[str], step_history: List[StepInterpretation]) -> Tuple[str, str, TokenUsage]:
        # Python側で Mermaid を組み立てる
        # Engine がループ中に extracted_texts に "A --> B" の形式で溜め込んでいるため、
        # ここではそれを整理・結合するだけでよい。
        
        # 重複排除とソート（決定論的な出力を保証するため）
        unique_lines = sorted(list(set(line.strip() for line in extracted_texts if line.strip())))
        
        # プログラムによる確実なフォーマット
        raw_content = "graph TD\n    " + "\n    ".join(unique_lines)
        
        return raw_content, raw_content, TokenUsage()

