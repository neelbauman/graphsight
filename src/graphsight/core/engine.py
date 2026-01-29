import math
import os
from loguru import logger
from typing import List, Dict, Set, Optional
from ..strategies.base import BaseStrategy
from ..llm.base import BaseVLM
from ..models import DiagramResult, Focus, TokenUsage, StepInterpretation
from ..utils.image import add_grid_overlay

class NodeRegistry:
    def __init__(self, mode: str = "bbox", spatial_threshold: float = 100.0):
        self.nodes: Dict[str, List[Focus]] = {}
        self.mode = mode
        self.threshold = spatial_threshold

    def resolve_id(self, focus: Focus) -> str:
        base_id = focus.suggested_id if focus.suggested_id else "node_Unknown"
        
        if base_id not in self.nodes:
            self.nodes[base_id] = [focus]
            logger.debug(f"🆕 Registry: New ID '{base_id}' created.")
            return base_id

        candidates = self.nodes[base_id]
        
        for i, candidate in enumerate(candidates):
            # Hybrid判定 (OR条件)
            # どちらか一方でも一致すればマージする
            if focus.is_same_location_hybrid(candidate, self.threshold):
                unique_id = base_id if i == 0 else f"{base_id}_{i + 1}"
                logger.debug(f"🔗 Registry: Merged '{base_id}' -> '{unique_id}' (Location Match)")
                return unique_id

        # どちらも一致しなければ新規 (Spatial Split)
        self.nodes[base_id].append(focus)
        new_suffix = len(self.nodes[base_id])
        unique_id = f"{base_id}_{new_suffix}"
        logger.info(f"🔱 Registry: Split '{base_id}' -> '{unique_id}' (Different Grid AND Far BBox)")
        return unique_id

class GraphInterpreter:
    def __init__(self, vlm_client: BaseVLM):
        self.vlm = vlm_client

    def _format_loc(self, focus: Focus, use_grid: bool) -> str:
        if use_grid:
            refs = focus.grid_refs if focus.grid_refs else "NoGrid"
            return f"Grid: {refs}"
        else:
            return f"BBox: {focus.bbox}"

    def process(self, image_path: str, strategy: BaseStrategy, initial_usage: TokenUsage | None = None) -> DiagramResult:
        logger.info(f"🚀 Starting interpretation: {strategy.mermaid_type}")

        use_grid = getattr(strategy, "use_grid", False)
        target_image_path = image_path
        if use_grid:
            logger.info("Applying Grid SoM Overlay...")
            try:
                grid_path, r, c = add_grid_overlay(image_path, min_cell_size=150)
                target_image_path = grid_path
                logger.info(f"✅ Grid applied: {r}x{c} cells. Using temporary file: {target_image_path}")
            except Exception as e:
                logger.error(f"❌ Failed grid overlay: {e}. Using original.")
                use_grid = False

        registry = NodeRegistry(mode="grid" if use_grid else "bbox", spatial_threshold=150.0)
        
        extracted_data: List[str] = []
        step_history: List[StepInterpretation] = []
        frontier_queue: List[Focus] = []
        visited_unique_ids: Set[str] = set()
        
        total_usage = initial_usage if initial_usage else TokenUsage()

        # 1. 初期探索
        logger.info("🔍 Finding initial nodes...")
        initial_focus_list, usage = strategy.find_initial_focus(self.vlm, target_image_path)
        total_usage += usage
        
        for focus in initial_focus_list:
            unique_id = registry.resolve_id(focus)
            focus.suggested_id = unique_id
            frontier_queue.append(focus)
            
            loc_info = self._format_loc(focus, use_grid)
            logger.info(f"   Found Start Node: {unique_id} ({loc_info}) - {focus.description}")
        
        step_count = 0
        max_steps = 30

        while frontier_queue and step_count < max_steps:
            current_focus = frontier_queue.pop(0)
            
            if current_focus.suggested_id in visited_unique_ids:
                continue
            visited_unique_ids.add(current_focus.suggested_id)
            
            loc_info = self._format_loc(current_focus, use_grid)
            logger.info(f"📌 Step {step_count+1}: Analyzing '{current_focus.suggested_id}' ({loc_info})")

            # AI実行
            context_snapshot = list(extracted_data)
            step_result, usage = strategy.interpret_step(self.vlm, target_image_path, current_focus, context_snapshot)
            total_usage += usage
            step_history.append(step_result)

            if not step_result.outgoing_edges:
                logger.info(f"   🛑 Terminal node.")

            # Late Binding & Context Enrichment
            for edge_info in step_result.outgoing_edges:
                next_focus = Focus(
                    description=edge_info.description,
                    bbox=edge_info.bbox,
                    grid_refs=edge_info.grid_refs,
                    suggested_id=edge_info.target_id
                )

                # 1. ID解決
                resolved_target_id = registry.resolve_id(next_focus)
                next_focus.suggested_id = resolved_target_id

                # 2. Mermaidコード生成
                arrow = "-->"
                if edge_info.edge_label:
                    arrow = f"-->|{edge_info.edge_label}|"
                
                # 3. メタデータ埋め込み
                meta_comment = ""
                if use_grid:
                    src_g = current_focus.grid_refs or []
                    dst_g = next_focus.grid_refs or []
                    meta_comment = f" %% Grid: {src_g} -> {dst_g}"
                
                mermaid_line = f"{current_focus.suggested_id} {arrow} {resolved_target_id}{meta_comment}"
                extracted_data.append(mermaid_line)
                
                next_loc = self._format_loc(next_focus, use_grid)
                logger.info(f"   ➡️  Edge: {mermaid_line}")

                if resolved_target_id not in visited_unique_ids:
                    frontier_queue.append(next_focus)
                else:
                    logger.debug(f"   (Link to existing: {resolved_target_id})")

            step_count += 1

        if use_grid and target_image_path != image_path and os.path.exists(target_image_path):
            try:
                os.remove(target_image_path)
            except OSError: pass

        logger.info("📝 Synthesizing...")

        final_content, raw_content, synth_usage = strategy.synthesize(self.vlm, target_image_path, extracted_data, step_history)
        total_usage += synth_usage

        if use_grid and target_image_path != image_path and os.path.exists(target_image_path):
            try:
                os.remove(target_image_path)
                logger.debug("🧹 Cleaned up temporary grid image.")
            except OSError:
                pass
        
        return DiagramResult(
            diagram_type=strategy.mermaid_type,
            output_format=strategy.output_format,
            content=final_content,
            raw_content=raw_content,
            full_description=f"Interpreted in {step_count} steps.",
            usage=total_usage,
            cost_usd=self.vlm.calculate_cost(total_usage),
            model_name=self.vlm.model_name
        )

