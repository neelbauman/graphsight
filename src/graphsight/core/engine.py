"""
graphsight.core.engine
~~~~~~~~~~~~~~~~~~~~~~
Core graph interpretation engine.
Implements the "Crawl -> Initial Audit -> Consistency Loop" workflow with verbose logging.
"""

import math
import os
from beautyspot import Spot
from loguru import logger
from typing import List, Dict, Set, Optional, Tuple, NamedTuple
from ..strategies.base import BaseStrategy
from ..llm.base import BaseVLM
from ..models import DiagramResult, Focus, TokenUsage, StepInterpretation, ConnectedNode
from ..utils.image import add_grid_overlay

# 監査タスクを管理する構造体
class AuditTask(NamedTuple):
    index: int
    step: StepInterpretation
    proposed_in: List[str]
    proposed_out: List[str]
    reasons: List[str]

class NodeRegistry:
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

    def _format_loc(self, focus: Focus, use_grid: bool) -> str:
        if use_grid:
            refs = focus.grid_refs if focus.grid_refs else "NoGrid"
            return f"Grid: {refs}"
        else:
            return f"BBox: {focus.bbox}"

    def process(self, image_path: str, strategy: BaseStrategy, initial_usage: Optional[TokenUsage] = None, traversal_mode: str = "dfs") -> DiagramResult:
        logger.info(f"🚀 Starting interpretation flow: {strategy.mermaid_type}")

        # --- 0. Setup ---
        use_grid = False
        if hasattr(strategy, "use_grid"):
             use_grid = strategy.use_grid

        target_image_path = image_path
        if use_grid:
            try:
                grid_path, r, c = add_grid_overlay(image_path, min_cell_size=150)
                target_image_path = grid_path
                strategy.set_grid_dimensions(r, c)

                logger.info(f"✅ Grid applied: {target_image_path}")
            except Exception:
                use_grid = False

        registry = NodeRegistry(mode="grid" if use_grid else "bbox", spatial_threshold=150.0)
        total_usage = initial_usage if initial_usage else TokenUsage()

        # --- Phase 1: Crawl (仮説構築) ---
        logger.info("Phase 1: 🕷️ Crawling nodes to build initial hypothesis...")
        step_history, usage = self._crawl(
            target_image_path, strategy, registry, traversal_mode, use_grid
        )
        total_usage += usage
        logger.info(f"   📊 Crawl complete. Found {len(step_history)} nodes.")

        # --- Phase 2: Initial Audit (初回一斉監査) ---
        logger.info("Phase 2: 📸 Running Initial Audit on all nodes (Outgoing check)...")
        usage = self._run_initial_audit(
            target_image_path, strategy, step_history
        )
        total_usage += usage

        # --- Phase 3: Consistency Loop (整合性収束) ---
        logger.info("Phase 3: ⚖️ Starting Global Consistency Loop (In/Out Check)...")
        usage = self._run_consistency_loop(
            target_image_path, strategy, step_history, max_attempts=10
        )
        total_usage += usage

        # --- 4. Synthesis ---
        logger.info("Phase 4: 📝 Synthesizing final graph...")
        final_content, raw_content, synth_usage = strategy.synthesize(
            self.vlm, target_image_path, [], step_history
        )
        total_usage += synth_usage

        # Cleanup
        if use_grid and target_image_path != image_path and os.path.exists(target_image_path):
            try: os.remove(target_image_path)
            except OSError: pass
        
        return DiagramResult(
            diagram_type=strategy.mermaid_type,
            output_format=strategy.output_format,
            content=final_content,
            raw_content=raw_content,
            full_description=f"Interpreted in {len(step_history)} steps.",
            usage=total_usage,
            cost_usd=self.vlm.calculate_cost(total_usage),
            model_name=self.vlm.model_name
        )

    # =========================================================================
    # Phase 1: Crawl
    # =========================================================================
    def _crawl(
        self,
        image_path: str,
        strategy: BaseStrategy,
        registry: NodeRegistry,
        mode: str,
        use_grid: bool
    ) -> Tuple[List[StepInterpretation], TokenUsage]:
        
        step_history: List[StepInterpretation] = []
        frontier_queue: List[Focus] = []
        visited_ids: Set[str] = set()
        usage = TokenUsage()

        # Initial Focus
        start_nodes, u = strategy.find_initial_focus(self.vlm, image_path)
        usage += u
        
        for focus in start_nodes:
            unique_id = registry.resolve_id(focus)
            focus.suggested_id = unique_id
            frontier_queue.append(focus)
            logger.debug(f"   🚩 Start Node: {unique_id}")

        step_count = 0
        while frontier_queue and step_count < 30:
            if mode.lower() == "dfs":
                current = frontier_queue.pop(-1)
            else:
                current = frontier_queue.pop(0)
            
            if current.suggested_id in visited_ids: continue
            visited_ids.add(current.suggested_id)

            logger.info(f"   📍 Exploring: {current.suggested_id}")
            
            context = list(step_history)
            result, u = strategy.interpret_step(self.vlm, image_path, current, context)
            usage += u

            result.source_id = current.suggested_id
            result.source_grid_refs = current.grid_refs
            result.source_bbox = current.bbox
            step_history.append(result)

            # Queue Next
            for edge in result.outgoing_edges:
                next_focus = Focus(
                    description=edge.description,
                    bbox=edge.bbox,
                    grid_refs=edge.grid_refs,
                    suggested_id=edge.target_id
                )
                resolved_id = registry.resolve_id(next_focus)
                next_focus.suggested_id = resolved_id
                edge.target_id = resolved_id
                
                if resolved_id not in visited_ids:
                    frontier_queue.append(next_focus)
            step_count += 1
            
        return step_history, usage

    # =========================================================================
    # Phase 2: Initial Audit
    # =========================================================================
    def _run_initial_audit(
        self,
        image_path: str,
        strategy: BaseStrategy,
        step_history: List[StepInterpretation]
    ) -> TokenUsage:
        """
        全てのノードについて、現在のOutgoing接続が画像と合っているかを確認する。
        """
        total_usage = TokenUsage()

        for step in step_history:
            node_id = step.source_id
            proposed_out = [e.target_id for e in step.outgoing_edges]
            
            reconstruct_focus = Focus(
                description=step.visual_observation or "Audit target",
                suggested_id=node_id,
                bbox=step.source_bbox,
                grid_refs=step.source_grid_refs
            )

            # 監査実行 (Incomingは空)
            audit_result, u = strategy.audit_node(
                self.vlm,
                image_path,
                reconstruct_focus,
                step_history,
                [], 
                proposed_out
            )
            total_usage += u

            # 結果反映
            if audit_result.audit_confirmed_outgoing is not None:
                confirmed_set = set(audit_result.audit_confirmed_outgoing)
                new_edges = []
                
                # 既存のエッジ選別
                for edge in step.outgoing_edges:
                    if edge.target_id in confirmed_set:
                        new_edges.append(edge)
                        confirmed_set.remove(edge.target_id)
                
                # 新規エッジ (見落とし)
                for new_tgt in confirmed_set:
                    logger.info(f"      ➕ [InitAudit] Adding missed edge: {node_id} --> {new_tgt}")
                    new_edges.append(ConnectedNode(target_id=new_tgt, description="(Audit Added)", edge_label=None))
                
                step.outgoing_edges = new_edges
                step.audit_confirmed_outgoing = audit_result.audit_confirmed_outgoing
                step.audit_notes = audit_result.audit_notes

        return total_usage

    # =========================================================================
    # Phase 3: Consistency Loop
    # =========================================================================
    def _run_consistency_loop(
        self,
        image_path: str,
        strategy: BaseStrategy,
        step_history: List[StepInterpretation],
        max_attempts: int
    ) -> TokenUsage:
        """
        不整合チェック -> 修正(Audit) -> 履歴書き換え -> 繰り返し
        """
        total_usage = TokenUsage()

        for attempt in range(max_attempts):
            logger.info(f"🔄 Consistency Iteration {attempt + 1}/{max_attempts}")

            # 1. Check
            tasks = self._find_inconsistencies(step_history)

            if not tasks:
                logger.info("✅ Graph Converged! No inconsistencies found.")
                break
            
            logger.info(f"   ⚠️ Found {len(tasks)} inconsistencies. Fixing...")

            # 2. Fix
            changes_made, u = self._execute_fix_batch(
                image_path, strategy, step_history, tasks
            )
            total_usage += u

            if not changes_made:
                logger.info("   🛑 Loop finished (No structural changes detected).")
                break
        
        return total_usage

    def _find_inconsistencies(self, step_history: List[StepInterpretation]) -> List[AuditTask]:
        # A. Logic Graph Construction
        graph_map = {} 
        for step in step_history:
            if step.source_id: graph_map[step.source_id] = {"in": set(), "out": set()}
        
        for step in step_history:
            src = step.source_id
            if not src: continue
            for edge in step.outgoing_edges:
                dst = edge.target_id
                graph_map[src]["out"].add(dst)
                if dst not in graph_map: graph_map[dst] = {"in": set(), "out": set()}
                graph_map[dst]["in"].add(src)

        # B. Compare
        tasks = []
        for i, step in enumerate(step_history):
            node_id = step.source_id
            if not node_id: continue

            logic_in = graph_map.get(node_id, {}).get("in", set())
            logic_out = graph_map.get(node_id, {}).get("out", set())
            reasons = []

            # Check Incoming
            if step.audit_confirmed_incoming is not None:
                visual_in_set = set(step.audit_confirmed_incoming)
                if logic_in != visual_in_set:
                    reasons.append(f"Incoming Logic({len(logic_in)}) != Confirmed({len(visual_in_set)})")
            else:
                visual_count = len(step.incoming_edges)
                if len(logic_in) != visual_count:
                    reasons.append(f"Incoming Logic({len(logic_in)}) != VisualCount({visual_count})")

            # Check Outgoing
            current_out = set([e.target_id for e in step.outgoing_edges])
            if logic_out != current_out:
                reasons.append("Outgoing Sync Error")

            if reasons:
                tasks.append(AuditTask(
                    index=i, step=step, proposed_in=list(logic_in), proposed_out=list(logic_out), reasons=reasons
                ))
        return tasks

    def _execute_fix_batch(
        self,
        image_path: str,
        strategy: BaseStrategy,
        step_history: List[StepInterpretation],
        tasks: List[AuditTask]
    ) -> Tuple[bool, TokenUsage]:
        """
        監査タスクを実行し、Forward Patching（自分の出力修正）と
        Reverse Patching（相手の出力修正、または新規ノード作成）を行う。
        """
        total_usage = TokenUsage()
        changes = False
        
        # ※ ID検索用マップは動的にノードが増えるため、ここでは作らず
        #    都度 _find_matching_node で検索する戦略をとる。

        for task in tasks:
            step = task.step
            node_id = step.source_id
            
            reconstruct_focus = Focus(
                description=step.visual_observation or "Audit",
                suggested_id=node_id,
                bbox=step.source_bbox,
                grid_refs=step.source_grid_refs
            )

            logger.info(f"   ⚖️ Re-Auditing '{node_id}': {task.reasons}")
            
            # 1. 再監査 (Re-Audit)
            audit_result, u = strategy.audit_node(
                self.vlm,
                image_path,
                reconstruct_focus,
                step_history,
                task.proposed_in,
                task.proposed_out
            )
            total_usage += u

            # 2. Metadata Update Detection
            prev_in = step.audit_confirmed_incoming
            prev_out = step.audit_confirmed_outgoing
            
            step.audit_confirmed_incoming = audit_result.audit_confirmed_incoming
            step.audit_confirmed_outgoing = audit_result.audit_confirmed_outgoing
            step.audit_notes = audit_result.audit_notes
            
            # リストの内容が変わったかチェック (順不同比較)
            if set(prev_in or []) != set(step.audit_confirmed_incoming or []):
                logger.info(f"      🔄 Incoming Meta updated: {prev_in} -> {step.audit_confirmed_incoming}")
                changes = True
            
            # Outgoing Metaの変更は、下のForward Patchingで構造変更として検知されるため、ここではログのみでも良いが
            # 安全のためフラグを立てておく
            if set(prev_out or []) != set(step.audit_confirmed_outgoing or []):
                changes = True

            # 3. Forward Patching (自分のOutgoingを修正)
            if audit_result.audit_confirmed_outgoing is not None:
                confirmed = set(audit_result.audit_confirmed_outgoing)
                current_edges = step.outgoing_edges
                new_edges = []
                
                # 既存エッジの維持判定
                for edge in current_edges:
                    if edge.target_id in confirmed:
                        new_edges.append(edge)
                        confirmed.remove(edge.target_id)
                    else:
                        logger.info(f"      ✂️ Removing edge: {node_id} --> {edge.target_id}")
                        changes = True # 削除発生
                
                # 新規エッジの追加
                for new_tgt in confirmed:
                    logger.info(f"      ➕ [Fix] Adding edge: {node_id} --> {new_tgt}")
                    new_edges.append(ConnectedNode(
                        target_id=new_tgt,
                        description="(Fix Added)",
                        edge_label=None
                    ))
                    changes = True # 追加発生
                
                step.outgoing_edges = new_edges

            # 4. Reverse Patching (相手のOutgoingを強制修正 & 新規作成)
            # 「私(B)へのIncomingはAだ」と確定したら、AのOutgoingにBを強制追加する
            if audit_result.audit_confirmed_incoming is not None:
                for src_id_raw in audit_result.audit_confirmed_incoming:
                    
                    # A. 既存ノードから検索 (Fuzzy Match)
                    matched_step = self._find_matching_node(src_id_raw, step_history)
                    
                    if matched_step:
                        # 既存ノードが見つかった場合 -> 接続を追加
                        src_id = matched_step.source_id
                        already_connected = any(e.target_id == node_id for e in matched_step.outgoing_edges)
                        
                        if not already_connected:
                            logger.info(f"      🔗 [Reverse Patch] Forcing {src_id} --> {node_id} (Matched from '{src_id_raw}')")
                            matched_step.outgoing_edges.append(ConnectedNode(
                                target_id=node_id,
                                description="(Reverse Patched)",
                                edge_label=None
                            ))
                            # Cache update (相手のAudit結果も更新して整合性を保つ)
                            if matched_step.audit_confirmed_outgoing is not None:
                                if node_id not in matched_step.audit_confirmed_outgoing:
                                    matched_step.audit_confirmed_outgoing.append(node_id)
                            changes = True
                    else:
                        # B. 見つからない場合 -> 新規ノード作成 (Missing Node Creation)
                        # AIが「そこから来ている」と言った以上、そのノードは存在する。探索漏れとして追加する。
                        logger.info(f"      🆕 Creating MISSING node found in audit: '{src_id_raw}'")
                        
                        new_step = StepInterpretation(
                            source_id=src_id_raw, # AIが言った名前をそのままIDにする
                            visual_observation="Discovered during Audit (Reverse Patch)",
                            outgoing_edges=[
                                ConnectedNode(
                                    target_id=node_id,
                                    description="(Reverse Patched)",
                                    edge_label=None
                                )
                            ],
                            incoming_edges=[], # 新規ノードのIncomingは不明
                            is_done=True
                        )
                        
                        step_history.append(new_step)
                        changes = True

        return changes, total_usage

    def _find_matching_node(self, target_id: str, history: List[StepInterpretation]) -> Optional[StepInterpretation]:
        """
        IDの表記ゆれを吸収して、履歴から該当するノードを探すヘルパーメソッド。
        """
        if not target_id: return None
        target_clean = target_id.lower().replace("node_", "").replace("_", " ").strip()
        
        # 1. 完全一致 (Exact)
        for step in history:
            if step.source_id == target_id:
                return step
        
        # 2. 正規化一致 (Normalized)
        for step in history:
            if not step.source_id: continue
            src_clean = step.source_id.lower().replace("node_", "").replace("_", " ").strip()
            if src_clean == target_clean:
                return step
                
        # 3. 部分一致 (Substring) - 慎重に適用
        # 短すぎるIDでの誤爆を防ぐため、ある程度の長さがある場合のみ許可
        if len(target_clean) >= 4:
            for step in history:
                if not step.source_id: continue
                src_clean = step.source_id.lower().replace("node_", "").replace("_", " ").strip()
                
                # 双方向の部分一致 ("Is there opportunity" in "node_Is_there_an_opportunity")
                if len(src_clean) >= 4:
                    if target_clean in src_clean or src_clean in target_clean:
                        return step
                    
        return None

