# neelbauman/graphsight/graphsight-refact-1/src/graphsight/pipelines/stable/draft_refine/draft_refine.py

from typing import List, Optional
from pathlib import Path

from loguru import logger
from langchain_core.messages import HumanMessage, SystemMessage 
# Note: ImageProcessor is likely a LangChain tool, so we keep using it as is, 
# or we could migrate it to a standalone function if desired.
from .tools import ImageProcessor 
from .agent import InspectorAgent

from graphsight.pipelines.base import BasePipeline
from graphsight.llm.openai_client import OpenAIVLM  # Use our wrapper
from graphsight.models import TokenUsage

from .models import (
    GraphStructure,
    DraftOutputStructured,
    UncertainPoint,
    CheckResult,    # Need to add to models.py
    CorrectionPlan, # Need to add to models.py
    RefineVerdict,
)

class DraftRefinePipeline(BasePipeline):
    """
    GraphSight Agent v6 — Draft → Refine Architecture (Structured Output Edition)
    
    Architecture:
      1. Draft: Generate GraphSchema (Nodes/Edges) directly via Structured Output.
      2. Refine: Check uncertain points using crops.
      3. Finalize: Apply structural corrections via Structured Output.
    """

    MAX_REFINE_CHECKS = 20
    CROP_MARGIN_RATIO = 0.5

    def __init__(self, model: str = "gpt-4o"):
        super().__init__(model_name=model)
        # Use OpenAIVLM wrapper for consistent token usage & config
        self.vlm = OpenAIVLM(model=model)

    def run(self, image_path: str) -> str:
        logger.info(f"🚀 Starting Draft→Refine (Structured) for: {image_path}")

        # 画像サイズ取得 (ImageProcessorは既存のToolを使用)
        info = ImageProcessor.get_image_info.invoke({"image_path": image_path})
        parts = info.replace("Image Size: ", "").split("x")
        img_w, img_h = int(parts[0]), int(parts[1])
        logger.info(f"📐 Image: {img_w}x{img_h}")

        # ===== Phase 1: Draft (Structured) =====
        draft_output, usage = self._phase_draft(image_path, img_w, img_h)
        
        # 内部表現に変換
        graph = GraphStructure.from_schema(draft_output.graph)
        
        logger.info(f"📝 Draft: {draft_output.confidence:.0%} confidence, "
                     f"{len(draft_output.uncertain_points)} uncertain points")
        logger.info(f"   Structure: {len(graph.nodes)} nodes, {len(graph.edges)} edges")

        # 確信度が十分高ければRefineスキップ
        if draft_output.confidence >= 0.95 and not draft_output.uncertain_points:
            logger.info("✅ High confidence — skipping refine")
            return graph.to_mermaid()

        # ===== Phase 2: Refine (Structural Diff) =====
        refined_graph = self._phase_refine(
            image_path, img_w, img_h, draft_output.uncertain_points, graph
        )
        
        # 最終的なMermaidを生成
        final_mermaid = refined_graph.to_mermaid()
        logger.info(f"✅ Final Mermaid: {len(final_mermaid)} chars")
        return final_mermaid

    # -----------------------------------------------------------------
    # Phase 1: Draft — Structured Output
    # -----------------------------------------------------------------
    # src/graphsight/pipelines/stable/draft_refine/draft_refine.py の _phase_draft メソッドのみ抜粋

    def _phase_draft(self, image_path: str, img_w: int, img_h: int) -> tuple[DraftOutputStructured, TokenUsage]:
        logger.info("=" * 50 + " Phase 1: DRAFT (Structured)")

        # 元のプロンプトの精神を継承しつつ、Structured Outputに最適化
        prompt = f"""You are an expert at converting flowchart images to structured graph data.

Image size: {img_w}x{img_h} pixels.

**TASK:**
1. Analyze the flowchart structure as if you were writing a Mermaid diagram.
2. Extract all Nodes and Edges into the structured format.
3. Honestly assess your uncertainty fairly, But be Confident your work.

**CRITICAL RULES for Nodes:**
- **Transcribe the text inside the node EXACTLY.**
- **Do NOT use the node ID as the label.** (e.g. If a node contains "Select Option", ID="A", Label="Select Option". NOT Label="A", NOT Label="SO").
- Use generic shapes (rect, diamond, etc.) that match the visual.

**CRITICAL RULES for Actor/Role Extraction:**
- **Swimlanes:** If the diagram has swimlanes (columns/rows), identify the header text. Assign that header to the 'actor' field for all nodes in that lane.
- **Colors:** If nodes are color-coded by role (e.g., Blue=User, Green=System), infer the actor name if a legend exists or from context.
- **Icons:** If an icon accompanies the node (e.g., Person, DB), use it to infer the actor.
- If no specific actor is indicated, leave 'actor' as null.

**CRITICAL RULES for Edges:**
- Ensure source and destination IDs strictly match the Nodes you defined.
- Capture edge labels (e.g. "Yes", "No") if present.

**Rules for Uncertain Points:**
- List ONLY genuinely uncertain items (unclear text, faint lines, ambiguous overlaps).
- Provide crop coordinates (x, y, w, h) to zoom in on the issue.
- Coordinates must be within {img_w}x{img_h}.

**Confidence Scoring:**
- 0.95+ = Perfect visibility, no doubts.
- 0.8-0.9 = Mostly clear, minor text ambiguity.
- <0.7 = Significant parts are unreadable or complex.
"""

        result, usage = self.vlm.query_structured(
            prompt=prompt,
            image_path=image_path,
            response_model=DraftOutputStructured,
        )
        
        logger.info(f"   Tokens used: {usage.total_tokens}")
        # (以下、不確実点の座標調整ログなど既存コードと同じ)
        
        return result, usage

    # -----------------------------------------------------------------
    # Phase 2: Refine
    # -----------------------------------------------------------------

    def _phase_refine(self, image_path: str, img_w: int, img_h: int,
                      uncertain_points: List[UncertainPoint], 
                      graph: GraphStructure) -> GraphStructure:
        logger.info("=" * 50 + " Phase 2: REFINE")

        if not uncertain_points:
            return graph

        # Limit checks
        to_check = uncertain_points[:self.MAX_REFINE_CHECKS]

        inspector = InspectorAgent(model_name=self.model_name)

        # 1. Visual Verification (Loop)
        for u in to_check:
            logger.info(f"   🔍 Checking {u.id}: {u.description}")
            try:
                # エージェントに調査を委譲
                u.resolution = inspector.verify_point(image_path, u, graph)
                logger.info(f"      ✅ Resolution: {u.resolution}")
            except Exception as e:
                logger.error(f"      ❌ Agent failed: {e}")
                u.resolution = "Agent error (skipping)"

            logger.info(f"      ✅ {u.resolution}")

        # 2. Filter corrections
        corrections = [u for u in to_check 
                       if u.resolution and "Correction:" in u.resolution]

        if not corrections:
            logger.info("   No corrections needed.")
            return graph

        # 3. Apply Corrections (Structural)
        return self._apply_structural_corrections(graph, corrections)

    def _apply_structural_corrections(
        self, graph: GraphStructure, corrections: List[UncertainPoint]
    ) -> GraphStructure:
        """修正指示に基づいてグラフ構造を更新する"""
        
        import copy
        new_graph = copy.deepcopy(graph)

        corrections_text = "\n".join(
            f"- {u.id}: {u.resolution}" for u in corrections
        )
        
        # 現在のグラフ構造を簡易JSON化してコンテキストとして渡す
        # (全部渡すとトークンが多いので、nodes/edgesのサマリでも良いが、
        #  構造変更にはIDが必要なのである程度詳細が必要)
        current_state = {
            "nodes": {nid: n.label for nid, n in graph.nodes.items()},
            "edges": [f"{e.src}->{e.dst}" for e in graph.edges]
        }


        prompt = f"""You have a graph structure and a list of corrections derived from visual verification.

Current Graph Summary:
{current_state}

Visual Corrections to Apply:
{corrections_text}

**TASK:** Generate a list of atomic operations to fix the graph.

**Rules:**
- **ONLY output operations that are explicitly justified by the corrections.**
- If a correction says "draft was correct", do NOT output any operation for it.
- Do NOT change anything that is not mentioned in the corrections.
- Do NOT hallucinate new changes.
"""

        # Structured Output for Plan
        plan, _ = self.vlm.query_structured(
            prompt=prompt,
            image_path=None, # Text only task
            response_model=CorrectionPlan
        )

        applied_count = 0
        for op in plan.operations:
            try:
                if op.op == "relabel":
                    if op.node_id in new_graph.nodes:
                        old = new_graph.nodes[op.node_id].label
                        new_graph.nodes[op.node_id].label = op.new_label
                        logger.info(f"      ✏️ Relabel {op.node_id}: '{old}' -> '{op.new_label}'")
                        applied_count += 1

                elif op.op == "reshape":
                    if op.node_id in new_graph.nodes:
                        new_graph.nodes[op.node_id].shape = op.new_shape
                        applied_count += 1
                
                elif op.op == "add_edge":
                    # 重複チェック等はGraphStructure側で行うのが理想だがここで簡易チェック
                    if not any(e.src == op.src and e.dst == op.dst for e in new_graph.edges):
                        # Edgeモデルの生成 (import from models)
                        from .models import Edge
                        new_graph.edges.append(Edge(
                            src=op.src, dst=op.dst, 
                            label=op.label or "", 
                            style=op.style or "-->"
                        ))
                        logger.info(f"      ➕ Add Edge: {op.src} -> {op.dst}")
                        applied_count += 1

                elif op.op == "remove_edge":
                    before = len(new_graph.edges)
                    new_graph.edges = [e for e in new_graph.edges 
                                       if not (e.src == op.src and e.dst == op.dst)]
                    if len(new_graph.edges) < before:
                        logger.info(f"      ➖ Remove Edge: {op.src} -> {op.dst}")
                        applied_count += 1

                elif op.op == "add_node":
                    if op.node_id not in new_graph.nodes:
                        from .models import Node
                        new_graph.nodes[op.node_id] = Node(
                            id=op.node_id, 
                            label=op.label or op.node_id, 
                            shape=op.new_shape or "rect"
                        )
                        logger.info(f"      ➕ Add Node: {op.node_id}")
                        applied_count += 1

                elif op.op == "remove_node":
                    if op.node_id in new_graph.nodes:
                        del new_graph.nodes[op.node_id]
                        new_graph.edges = [e for e in new_graph.edges 
                                           if e.src != op.node_id and e.dst != op.node_id]
                        logger.info(f"      ➖ Remove Node: {op.node_id}")
                        applied_count += 1

                elif op.op == "relabel_edge":
                    found = False
                    for e in new_graph.edges:
                        if e.src == op.src and e.dst == op.dst:
                            e.label = op.label
                            found = True
                    if found:
                        logger.info(f"      ✏️ Relabel Edge: {op.src}->{op.dst}")
                        applied_count += 1

            except Exception as e:
                logger.warning(f"      ⚠️ Failed op {op}: {e}")

        logger.info(f"   Applied {applied_count} operations.")
        return new_graph

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _check_uncertain_point(self, image_path: str, img_w: int, img_h: int, point: UncertainPoint, graph: GraphStructure) -> str:
        """Crop and verify using Structured Output"""
        
        # Calculate Crop
        margin_x = int(point.crop_w * self.CROP_MARGIN_RATIO)
        margin_y = int(point.crop_h * self.CROP_MARGIN_RATIO)
        crop_x = max(0, point.crop_x - margin_x)
        crop_y = max(0, point.crop_y - margin_y)
        crop_w = min(point.crop_w + margin_x * 2, img_w - crop_x)
        crop_h = min(point.crop_h + margin_y * 2, img_h - crop_y)

        # Use ImageProcessor Tool
        crop_path = ImageProcessor.crop_region.invoke({
            "image_path": image_path,
            "x": crop_x, "y": crop_y,
            "w": crop_w, "h": crop_h,
            "preprocess": "binarize",
        })
        
        if isinstance(crop_path, str) and crop_path.startswith("Error"):
            return f"Crop failed: {crop_path}"

        context_nodes = ", ".join([f"{n.id}[{n.label}]" for n in graph.nodes.values()])

        # Verify
        prompt = f"""You are an Objective Auditor validating a flowchart digitization.

**Context (Draft Interpretation):**
The draft currently sees: {context_nodes}

**Focus Area:**
"{point.description}"

**TASK:**
Compare the crop image against the draft interpretation.

**GUIDELINES FOR AUDIT:**
1. **Presume Innocence:** The draft is likely correct. Only mark it as 'False' if you see **undeniable evidence** of an error.
2. **Ambiguity:** If the image is blurry or ambiguous, trust the draft (set draft_matches_image=True). Do NOT guess.
3. **Minor Diff:** Ignore minor stylistic differences or trivial punctuation changes unless they alter meaning.
4. **Correction:** Provide a correction ONLY if you are 100% certain.

Output your audit result honestly.
"""

        try:
            result, _ = self.vlm.query_structured(
                prompt=prompt,
                image_path=crop_path,
                response_model=CheckResult
            )

            # Logic: Only accept correction if Verdict is INCORRECT
            if result.verdict == RefineVerdict.INCORRECT and result.correction_value:
                return f"{result.observation} → Correction: {result.correction_value}"
            elif result.verdict == RefineVerdict.UNCLEAR:
                return f"{result.observation} (Unclear, keeping draft)"
            else:
                return f"{result.observation} (Verified Correct)"
                
        except Exception as e:
            return f"Verification error: {e}"

    @staticmethod
    def _clamp(val, lo, hi):
        return max(lo, min(int(val), hi))

