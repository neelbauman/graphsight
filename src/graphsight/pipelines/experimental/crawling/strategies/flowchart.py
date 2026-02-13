"""
graphsight.strategies.flowchart
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Standard strategy for Flowcharts with Visual Chain-of-Thought and Bidirectional Verification.
"""

import os
from typing import List, Tuple, Dict
from pydantic import BaseModel
from loguru import logger
from .base import BaseStrategy, OutputFormat
from ..llm.base import BaseVLM
from ..llm.config import get_model_config, ModelType
from ..models import Focus, StepInterpretation, TokenUsage, InitialFocusList, ConnectionVerificationResult
from ..utils.geometry import calculate_relative_direction
from ..utils.image import crop_connection_area, crop_grid_area

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

    def interpret_step(
        self, 
        vlm: BaseVLM, 
        image_path: str, 
        current_focus: Focus, 
        context_history: List[StepInterpretation]
    ) -> Tuple[StepInterpretation, TokenUsage]:
        
        history_text = self._build_history_text(context_history)
        config = get_model_config(vlm.model_name)
        
        if self.use_grid:
            loc_str = f"Location: Grid={current_focus.grid_refs}"
            rules = self._build_hybrid_rules()
            context_note = "(Note: '%% Grid: ...' in Context indicates the spatial location of past nodes.)"
        else:
            loc_str = f"Location: BBox={current_focus.bbox}"
            rules = self._build_bbox_rules()
            context_note = "(Note: Check '%% BBox: ...' in Context for locations.)"

        if config.model_type == ModelType.REASONING:
            prompt = self._build_reasoning_prompt(current_focus, history_text, loc_str, rules, context_note)
        else:
            prompt = self._build_recognition_prompt(current_focus, history_text, loc_str, rules, context_note)

        return vlm.query_structured(prompt, image_path, StepInterpretation)

    def reexamine_step(
        self,
        vlm: BaseVLM,
        image_path: str,
        current_focus: Focus,
        context_history: List[StepInterpretation],
        previous_result: StepInterpretation,
        warnings: List[str]
    ) -> Tuple[StepInterpretation, TokenUsage]:
        # (This is mostly for local check, audit_node is now the main driver, but keeping for compatibility)
        return self.audit_node(vlm, image_path, current_focus, context_history, [], [])

    # --- Prompt Building Helpers ---

    def _build_history_text(self, history: List[StepInterpretation]) -> str:
        lines = []
        recent_steps = history[-15:]
        for step in recent_steps:
            src = step.source_id or "Unknown"
            for edge in step.outgoing_edges:
                tgt = edge.target_id
                label = f"|{edge.edge_label}|" if edge.edge_label else ""
                meta = ""
                if self.use_grid:
                    src_g = step.source_grid_refs if step.source_grid_refs else []
                    dst_g = edge.grid_refs if edge.grid_refs else []
                    if src_g or dst_g:
                        meta = f" %% Grid: {src_g} -> {dst_g}"
                else:
                    src_b = step.source_bbox
                    dst_b = edge.bbox
                    if src_b or dst_b:
                        meta = f" %% BBox: {src_b} -> {dst_b}"
                line = f"{src} -->{label} {tgt}{meta}"
                lines.append(line)
        return "\n".join(lines)

    def _build_bbox_rules(self) -> str:
        return """
        - **Spatial Info**: Provide `bbox` [ymin, xmin, ymax, xmax] (0-1000) for EVERY connected node.
        """
    
    def _build_hybrid_rules(self) -> str:
        return """
        - **Dual Spatial Info (CRITICAL)**:
          1. **grid_refs**: List **ALL** grid labels (e.g. ["C3", "C4"]).
          2. **bbox**: Also provide the estimated bounding box.
        """

    def _build_recognition_prompt(self, current_focus: Focus, history_text: str, loc_str: str, rules: str, context_note: str) -> str:
        return f"""
        You are a **Pixel-Perfect Line Tracing Engine**. 
        Your ONLY job is to trace visible black pixels (lines) from a starting point to an ending point.
        
        # 🚫 STRICT PROHIBITIONS (Anti-Hallucination Rules)
        1. **IGNORE TEXT MEANING**: Do not guess connections based on what the text says. Even if a node says "Go to End", if there is no line, IT DOES NOT CONNECT.
        2. **PROXIMITY ≠ CONNECTION**: Just because two nodes are close does NOT mean they connect. There must be a continuous line.
        3. **NO GUESSING**: If a line fades out, disappears, or is ambiguous, report NO CONNECTION.
        
        # Current Focus Node
        - ID: "{current_focus.suggested_id}"
        - {loc_str}
        - Description: "{current_focus.description}"
        
        # Context
        {history_text}
        
        # INSTRUCTIONS (Step-by-Step Physical Tracing)
        
        ## Step 1: Analyze Incoming (Arrivals)
        - Scan the perimeter of this node. Do you see arrowheads touching the border?
        - **Constraint**: If an arrow points *near* the node but doesn't touch, it is NOT an incoming connection.
        
        ## Step 2: Trace Outgoing (Departures) - CRITICAL
        - Find lines starting from this node's border.
        - **Trace the Path**: Follow the line visually. 
          - "It goes down, turns right, crosses over a vertical line..."
          - **Crossing vs Junction**: If a line crosses another without a dot/arrow, it is a bridge (NOT a connection). Keep tracing.
        - **Identify Target**: Where does the arrowhead explicitly land?
        
        ## Step 3: Extract Data
        - List strictly visible connections.
        {rules}
        """

    def audit_node(
        self,
        vlm: BaseVLM,
        image_path: str,
        current_focus: Focus,
        context_history: List[StepInterpretation],
        proposed_incoming: List[str],
        proposed_outgoing: List[str]
    ) -> Tuple[StepInterpretation, TokenUsage]:
        """
        Global Integrity Check with Two-Stage Verification:
        1. Macro Audit: Full image context check.
        2. Micro Verification: Cropped zoom-in check for outgoing edges.
        """
        total_usage = TokenUsage()
        
        # --- Stage 1: Macro Audit (全体画像での監査) ---
        
        loc_str = f"Location: Grid={current_focus.grid_refs}" if self.use_grid else f"Location: BBox={current_focus.bbox}"
        in_str = ", ".join(sorted(proposed_incoming)) if proposed_incoming else "(None)"
        out_str = ", ".join(sorted(proposed_outgoing)) if proposed_outgoing else "(None)"

        macro_prompt = f"""
        You are a **Forensic Graph Auditor**.
        Your goal is to detect and remove "Phantom Connections" (Hallucinations) and find missed "Long-distance Connections".
        
        # Target Node
        - ID: "{current_focus.suggested_id}"
        - {loc_str}
        - Description: "{current_focus.description}"
        
        # HYPOTHESIS (Current Data)
        - Claimed Incoming: [{in_str}]
        - Claimed Outgoing: [{out_str}]
        
        # TASK: Visual Verification on Full Image
        1. **Verify Incoming**:
           - **Action**: REMOVE if no visible line connects.
           - **Action**: ADD if you see a clear arrow from an unlisted node.
        
        2. **Verify Outgoing**:
           - **Action**: REMOVE if the line fades out, crosses over, or stops short.
           - **Action**: ADD if you trace a line to a valid target (even if far away).
           
        3. **Critical Rules**:
           - **Ignore Proximity**: Two nodes being close does NOT mean they connect. Look for the LINE.
           - **Trace Carefully**: Follow lines through crosses and turns.
        
        # Output Requirement
        Return a `StepInterpretation` with `audit_confirmed_incoming`, `audit_confirmed_outgoing`, and `audit_notes`.
        """
        
        base_audit, u = vlm.query_structured(macro_prompt, image_path, StepInterpretation)
        total_usage += u
        
        # --- Stage 2: Micro Verification (クロップ検証) ---
        # "Outgoing" として確定しかけたリストに対し、本当に繋がっているか「虫眼鏡」で確認する。
        # (Incomingの検証は、相手側のOutgoing検証で行われるため、ここではOutgoingに集中する)
        
        final_outgoing_confirmed = []
        step_map = {s.source_id: s for s in context_history if s.source_id}
        
        # Macro Auditで「繋がっている」と判定された候補
        candidates = base_audit.audit_confirmed_outgoing or []
        
        if not candidates:
            # 候補がゼロなら検証不要
            return base_audit, total_usage

        if not current_focus.bbox:
            # 自分の位置が不明ならクロップできないので、Macroの結果をそのまま返す
            return base_audit, total_usage

        logger.info(f"      🔎 Micro-Verifying {len(candidates)} outgoing candidates for '{current_focus.suggested_id}'...")

        for target_id in candidates:
            target_step = step_map.get(target_id)

            if target_step is None:
                logger.warning(f"      ⚠️ Target '{target_id}' not found in history. Skipping micro-verification.")
                # 検証できないので、Macro Auditの結果を維持して次へ
                final_outgoing_confirmed.append(target_id)
                continue

            crop_path = None

            if self.use_grid and self.grid_rows > 0:
                # === GRID BASED CROP ===
                # 両者の grid_refs を使用
                src_refs = current_focus.grid_refs or []
                tgt_refs = target_step.source_grid_refs or []
                
                if src_refs or tgt_refs:
                    crop_path = crop_grid_area(
                        image_path, 
                        src_refs, 
                        tgt_refs, 
                        self.grid_rows, 
                        self.grid_cols
                    )
            
            # Gridで失敗した、またはGridモードでない場合はBBoxを使用
            if not crop_path and current_focus.bbox and target_step and target_step.source_bbox:
                # === BBOX BASED CROP ===
                crop_path = crop_connection_area(image_path, current_focus.bbox, target_step.source_bbox)
            
            if not crop_path:
                final_outgoing_confirmed.append(target_id)
                continue
            # ターゲットの位置情報がない場合は、クロップできないのでMacro判定を信じる
            if not target_step or not target_step.source_bbox:
                final_outgoing_confirmed.append(target_id)
                continue
            
            # クロップ画像の生成
            crop_path = crop_connection_area(image_path, current_focus.bbox, target_step.source_bbox)
            
            micro_prompt = f"""
            You are a **Connectivity Verifier**.
            I have cropped the image to show ONLY the area between two nodes.
            
            1. Source: "{current_focus.suggested_id}" ({current_focus.description})
            2. Target: "{target_id}"
            
            # TASK
            Look at this zoomed-in image.
            **Is there a CONTINUOUS, UNBROKEN LINE connecting the Source to the Target?**
            
            - **NO** if the line breaks.
            - **NO** if the line is just passing by (crossing) without a dot.
            - **NO** if the arrow points to a different node.
            - **YES** only if pixels physically connect them.
            
            Return `is_connected` (bool) and a brief `reason`.
            """
            
            try:
                # 軽量な構造化抽出を使用
                verify_res, u = vlm.query_structured(micro_prompt, crop_path, ConnectionVerificationResult)
                total_usage += u
                
                if verify_res.is_connected:
                    final_outgoing_confirmed.append(target_id)
                    # logger.debug(f"         ✅ Verified: -> {target_id}")
                else:
                    logger.info(f"         ✂️ REJECTED by Micro-View: {current_focus.suggested_id} -> {target_id} ({verify_res.reason})")
                    if base_audit.audit_notes:
                        base_audit.audit_notes += f" [Rejected {target_id}: {verify_res.reason}]"
                    else:
                        base_audit.audit_notes = f"[Rejected {target_id}: {verify_res.reason}]"

            except Exception as e:
                logger.error(f"         ❌ Micro-Verify Error: {e}")
                # エラー時は安全側に倒して（あるいは元の判定を信じて）残す
                final_outgoing_confirmed.append(target_id)
            finally:
                # 一時ファイルの削除
                if os.path.exists(crop_path):
                    os.remove(crop_path)
        
        # 最終結果の上書き
        base_audit.audit_confirmed_outgoing = final_outgoing_confirmed
        
        return base_audit, total_usage

    def _build_reasoning_prompt(self, current_focus: Focus, history_text: str, loc_str: str, rules: str, context_note: str) -> str:
        return f"""
        Analyze the flowchart.
        Target: "{current_focus.suggested_id}" ({loc_str})
        Context: {history_text}

        Goal: Identify INCOMING and OUTGOING connections.
        Requirements:
        1. Physical Tracing: Ignore semantic expectations.
        2. Incoming Check: Verify incoming arrow COUNT and direction.
        3. Reality First: Do not force branches (e.g. Yes/No) if not visible. Only report actual lines.
        4. Global Awareness: Check for long lines connecting from distant parts of the diagram.
        5. Spatial Accuracy: Location info must match.
        {rules}
        """

    def synthesize(
        self, 
        vlm: BaseVLM, 
        image_path: str, 
        extracted_texts: List[str],
        step_history: List[StepInterpretation]
    ) -> Tuple[str, str, TokenUsage]:
        
        # 1. Mechanical Synthesis (Raw) - REBUILT from History
        seen = set()
        unique_lines = []
        
        for step in step_history:
            src = step.source_id or "Unknown"
            
            for edge in step.outgoing_edges:
                arrow = "-->"
                if edge.edge_label:
                    arrow = f"-->|{edge.edge_label}|"
                meta = ""
                if self.use_grid:
                    src_g = step.source_grid_refs or []
                    dst_g = edge.grid_refs or []
                    meta = f" %% Grid: {src_g} -> {dst_g}"
                
                line = f"{src} {arrow} {edge.target_id}{meta}"
                if line not in seen:
                    unique_lines.append(line)
                    seen.add(line)

        raw_content = "graph TD\n    " + "\n    ".join(sorted(unique_lines))

        # 2. Build Investigation Log - from Corrected History
        investigation_log = ""
        for i, step in enumerate(step_history):
            investigation_log += f"Step {i+1} [Node: {step.source_id}]:\n"
            investigation_log += f"  - Observation: {step.visual_observation}\n"
            investigation_log += f"  - Tracing: {step.arrow_tracing}\n"
            
            if step.incoming_edges:
                dirs = [inc.direction for inc in step.incoming_edges]
                investigation_log += f"  - Incoming Observed: {len(dirs)} arrows from {dirs}\n"
            
            if step.visual_override_reason:
                investigation_log += f"  - NOTE: Visual Override active: {step.visual_override_reason}\n"
            
            if step.audit_notes:
                investigation_log += f"  - AUDIT FIX: {step.audit_notes}\n"
                
            investigation_log += "  - Connections Found:\n"
            for edge in step.outgoing_edges:
                 loc_info = f"Grid: {edge.grid_refs}" if self.use_grid else f"BBox: {edge.bbox}"
                 investigation_log += f"    -> {edge.target_id} [{edge.edge_label or ''}] ({loc_info})\n"
            investigation_log += "\n"

        # 3. AI Refinement
        prompt = f"""
        Refine the fragmented flowchart data into a single, valid, and clean Mermaid flowchart.

        # Input Data
        1. **The Image**: The ULTIMATE GROUND TRUTH.
        2. **Investigation Log**: The CORRECTED tracing history (Audited).
        3. **Raw Data**: The mechanically assembled graph from the corrected history.
        
        # Investigation Log
        {investigation_log}
        
        # Raw Mechanical Output
        {raw_content}
        
        # Instructions
        1. **Reconstruct**: Build the final Mermaid code based on the Log.
        2. **Truth Source**: The 'Investigation Log' contains the audited truths. Prioritize it.
        3. **Clean Up**: Remove grid comments.
        """
        
        refined_content, usage = vlm.query_text(prompt, image_path=image_path)
        
        return refined_content, raw_content, usage

