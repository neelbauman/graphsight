"""
graphsight.core.engine
~~~~~~~~~~~~~~~~~~~~~~
Core graph interpretation engine.
Refactored for Simple BFS: "Find Start -> Ask Next -> Queue Next".
"""

import os
from collections import deque  # 追加
from beautyspot import Spot
from loguru import logger
from typing import List, Dict, Set, Optional, Tuple, NamedTuple
from .strategies.base import BaseStrategy
from .llm.base import BaseVLM
from .models import DiagramResult, Focus, TokenUsage, StepInterpretation, ConnectedNode
from .utils.image import add_grid_overlay

class NodeRegistry:
    # ... (変更なし) ...
    def __init__(self, mode: str = "bbox", spatial_threshold: float = 100.0):
        self.nodes: Dict[str, List[Focus]] = {}
        self.mode = mode
        self.threshold = spatial_threshold

    def resolve_id(self, focus: Focus) -> str:
        base_id = focus.suggested_id if focus.suggested_id else "node_Unknown"
        if base_id not in self.nodes:
            self.nodes[base_id] = [focus]
            return base_id
        candidates = self.nodes[base_id]
        for i, candidate in enumerate(candidates):
            if focus.is_same_location_hybrid(candidate, self.threshold):
                return base_id if i == 0 else f"{base_id}_{i + 1}"
        self.nodes[base_id].append(focus)
        return f"{base_id}_{len(self.nodes[base_id])}"

class GraphInterpreter:
    def __init__(self, vlm: BaseVLM):
        self.vlm = vlm

    def process(self, image_path: str, strategy: BaseStrategy, initial_usage: Optional[TokenUsage] = None) -> DiagramResult:
        # ログメッセージをBFSに変更
        logger.info(f"🚀 Starting Simple BFS exploration: {strategy.mermaid_type}")

        # --- 0. Setup ---
        target_image_path = image_path
        
        registry = NodeRegistry(mode="bbox", spatial_threshold=150.0)
        total_usage = initial_usage if initial_usage else TokenUsage()

        # --- Phase 1: Iterative BFS ---
        logger.info("Phase 1: 🌊 Starting Layer-by-Layer BFS...")
        
        # _crawl_dfs -> _crawl_bfs に変更
        step_history, usage = self._crawl_bfs(
            target_image_path, strategy, registry
        )
        total_usage += usage
        
        logger.info(f"   📊 Exploration complete. Visited {len(step_history)} nodes.")

        # --- Phase 2: Synthesis ---
        logger.info("Phase 2: 📝 Synthesizing final graph...")
        final_content, raw_content, synth_usage = strategy.synthesize(
            self.vlm, target_image_path, [], step_history
        )
        total_usage += synth_usage

        return DiagramResult(
            diagram_type=strategy.mermaid_type,
            output_format=strategy.output_format,
            content=final_content,
            raw_content=raw_content,
            full_description=f"Interpreted in {len(step_history)} steps via BFS.",
            usage=total_usage,
            cost_usd=self.vlm.calculate_cost(total_usage),
            model_name=self.vlm.model_name
        )

    def _crawl_bfs(
        self,
        image_path: str,
        strategy: BaseStrategy,
        registry: NodeRegistry
    ) -> Tuple[List[StepInterpretation], TokenUsage]:
        
        step_history: List[StepInterpretation] = []
        
        # Stack([]) ではなく Deque() を使用
        queue: deque[Focus] = deque()
        
        visited_ids: Set[str] = set()
        usage = TokenUsage()

        # 1. Ask: "What is the start node?"
        logger.info("   ❓ Asking: 'What is the start node?'")
        start_nodes, u = strategy.find_initial_focus(self.vlm, image_path)
        usage += u
        
        for focus in start_nodes:
            unique_id = registry.resolve_id(focus)
            focus.suggested_id = unique_id
            
            # スタートノードをキューに追加
            queue.append(focus)
            # BFSではキュー追加時にvisited判定をした方が無駄な重複追加を防ぎやすいですが、
            # ここでは厳密な探索順序を守るため取り出し時にチェックします
            logger.debug(f"   🚩 Found Start: {unique_id}")

        step_count = 0
        max_steps = 30

        while queue and step_count < max_steps:
            # Pop from LEFT (FIFO) -> BFS behavior
            current = queue.popleft()
            
            if current.suggested_id in visited_ids:
                continue
            visited_ids.add(current.suggested_id)

            logger.info(f"   📍 Visiting: {current.suggested_id}")
            
            # 2. Ask: "What is next from here?"
            context = list(step_history)
            result, u = strategy.interpret_step(self.vlm, image_path, current, context)
            usage += u

            # メタデータの付与
            result.source_id = current.suggested_id
            result.source_bbox = current.bbox
            step_history.append(result)

            # 3. Queue next nodes
            for edge in result.outgoing_edges:
                next_focus = Focus(
                    description=edge.description,
                    bbox=edge.bbox,
                    suggested_id=edge.target_id
                )
                resolved_id = registry.resolve_id(next_focus)
                next_focus.suggested_id = resolved_id
                edge.target_id = resolved_id
                
                if resolved_id not in visited_ids:
                    # 重複してキューに入れないよう、簡単なチェックを入れても良い
                    # (厳密には queueの中身もチェックすべきだが、簡易実装として)
                    logger.info(f"      👉 Enqueue: {resolved_id}")
                    queue.append(next_focus)
                else:
                    logger.debug(f"      🔄 Already visited: {resolved_id}")

            step_count += 1
            
        return step_history, usage

