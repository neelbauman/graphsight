from loguru import logger
from typing import List
from ..strategies.base import BaseStrategy
from ..llm.base import BaseVLM
from ..models import DiagramResult, Focus, TokenUsage, StepInterpretation

class GraphInterpreter:
    def __init__(self, vlm_client: BaseVLM):
        self.vlm = vlm_client

    def process(self, image_path: str, strategy: BaseStrategy, initial_usage: TokenUsage | None = None) -> DiagramResult:
        logger.info(f"Starting interpretation: {strategy.mermaid_type} (Mode: {strategy.output_format.value})")

        extracted_data: List[str] = []
        step_history: List[StepInterpretation] = []
        frontier_queue: List[Focus] = []
        
        # IDベースでの訪問済み管理 (Set of strings)
        # description ではなく、AIが正規化した ID (例: "node_Check_Stock") で管理する
        visited_ids: set = set()
        
        # トークン集計初期化
        total_usage = initial_usage if initial_usage else TokenUsage()

        # 1. 初期探索 (Startノードの発見)
        initial_focus_list, usage = strategy.find_initial_focus(self.vlm, image_path)
        total_usage += usage
        frontier_queue.extend(initial_focus_list)
        
        step_count = 0
        max_steps = 30  # 複雑な分岐や合流を考慮してステップ数を確保

        while frontier_queue and step_count < max_steps:
            current_focus = frontier_queue.pop(0)

            # --- 訪問済みチェック (Exploration Filter) ---
            # ここで弾くのは「そのノードを起点とした探索」であり、
            # そのノードへの「接続（矢印）」自体は前のステップで記録済みである点に注意。
            
            # IDが提案されていればIDで、なければDescriptionでチェック
            check_key = current_focus.suggested_id if current_focus.suggested_id else current_focus.description
            
            if check_key in visited_ids:
                logger.debug(f"Skipping exploration of visited node: {check_key}")
                continue
            
            # 訪問済みに登録
            visited_ids.add(check_key)
            
            logger.debug(f"Step {step_count + 1}: Focusing on -> {current_focus.description} (ID: {current_focus.suggested_id})")
            
            # --- ステップ実行 (AIによる解釈) ---
            result, usage = strategy.interpret_step(self.vlm, image_path, current_focus, extracted_data)
            total_usage += usage
            
            # 履歴の保存 (AI Refinement用)
            step_history.append(result)

            # --- データ収集 ---
            # テキスト（Mermaidの接続コード等）は無条件で採用する
            if result.extracted_text:
                extracted_data.append(result.extracted_text)
                logger.info(f"  Found: {result.extracted_text}")

            # --- 次の探索候補のキューイング ---
            for candidate in result.next_focus_candidates:
                # 候補にIDがある場合、既に訪問済みならキューに追加しない
                # これにより「合流（Merge）」や「ループ（Loop）」の場合に、
                # 線は引かれるが、無限探索にはならない（閉路ができる）
                cand_id = candidate.suggested_id
                
                if cand_id and cand_id in visited_ids:
                    logger.debug(f"  -> Path to '{cand_id}' exists, but already visited. Connection recorded, skipping enqueue.")
                    continue
                
                # 未訪問ならキューに追加
                frontier_queue.append(candidate)

            step_count += 1

        # --- 最終合成 (Refinement) ---
        # 機械的な結合だけでなく、全ステップの思考ログ(step_history)を渡してAIに清書させる
        final_content, raw_content, synth_usage = strategy.synthesize(self.vlm, extracted_data, step_history)
        total_usage += synth_usage
        
        # コスト計算 (Model Configに基づく単価計算)
        total_cost = self.vlm.calculate_cost(total_usage)
        
        return DiagramResult(
            diagram_type=strategy.mermaid_type,
            output_format=strategy.output_format,
            content=final_content,     # AI Refined Result
            raw_content=raw_content,   # Mechanical Raw Result
            full_description=f"Interpreted in {step_count} steps.",
            usage=total_usage,
            cost_usd=total_cost,
            model_name=self.vlm.model_name
        )

