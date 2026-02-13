"""
GraphSight Agent v6 — Draft → Refine Architecture (Structural Diff Edition)

- GraphStructure / Node / Edge / GraphDiff データモデル追加
- MermaidParser: LLM出力のMermaidをグラフ構造にパース
- Refineフェーズを構造的差分適用に変更:
  LLMに「Mermaidを書き直して」ではなく「グラフ操作コマンド」を出力させ、
  プログラム的に適用する。修正対象以外のノード・エッジは一切触らない。
- crop座標にマージン追加（見切れ防止）

設計思想:
  LLMは全体画像を見て一発でMermaidを書くのが一番精度が高い。
  しかし細部（小さいラベル、薄い矢印）を見落とすことがある。

  そこで:
  1. Draft: 全体画像からMermaidを一発生成（LLMの全体把握力を最大活用）
  2. Self-Review: LLM自身にドラフトの不確実な箇所を挙げさせる
  3. Refine: 不確実な箇所だけをcrop/enhanceで確認し、修正
  4. Finalize: 修正をグラフ操作として適用（正しい箇所を壊さない）

  ┌───────────────────────────────────────┐
  │ Draft: 全体画像 → Mermaid一発生成    │ ← LLMの強みを活かす
  └──────────────┬────────────────────────┘
                 ▼
  ┌───────────────────────────────────────┐
  │ Self-Review: 不確実な箇所をリスト化   │ ← 構造化された疑問点
  └──────────────┬────────────────────────┘
                 ▼
  ┌───────────────────────────────────────┐
  │ Refine: 疑問点をcrop/enhanceで確認   │ ← ツール使用は的確・最小限
  │         (最大N個の疑問点を検証)       │    crop座標にマージン追加
  └──────────────┬────────────────────────┘
                 ▼
  ┌───────────────────────────────────────┐
  │ Finalize: グラフ操作コマンドで修正    │ ← 構造的差分適用
  │           正しい箇所は一切触らない     │
  └───────────────────────────────────────┘
"""

import base64
import json
import re
from typing import List, Optional
from dataclasses import dataclass, field
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from .tools import ImageProcessor
from ...base import BasePipeline
from .mermaid import MermaidParser
from .models import (
    Node,
    Edge,
    GraphStructure,
    UncertainPoint,
    DraftOutput,
)
from graphsight.bridge.client import NodeMermaidBridge


class DraftRefinePipeline(BasePipeline):

    MAX_REFINE_CHECKS = 20   # Refineフェーズで確認する疑問点の上限
    CROP_MARGIN_RATIO = 0.5  # crop座標に追加するマージン比率

    def __init__(self, model: str = "gpt-5.2"):
        try:
            self.llm = ChatOpenAI(model=model, temperature=0, reasoning_effort="high")
        except Exception:
            self.llm = ChatOpenAI(model=model, temperature=0)

    def run(self, image_path: str) -> str:
        logger.info(f"🚀 Starting Draft→Refine for: {image_path}")

        # 画像サイズ取得
        info = ImageProcessor.get_image_info.invoke({"image_path": image_path})
        parts = info.replace("Image Size: ", "").split("x")
        img_w, img_h = int(parts[0]), int(parts[1])
        logger.info(f"📐 Image: {img_w}x{img_h}")

        # ===== Phase 1: Draft =====
        draft = self._phase_draft(image_path, img_w, img_h)
        logger.info(f"📝 Draft: {draft.confidence:.0%} confidence, "
                     f"{len(draft.uncertain_points)} uncertain points")

        # ドラフトをグラフ構造にパース（検証用）
        try:
            parsed_data = NodeMermaidBridge.parse(draft.mermaid_code)
            
            # JSONデータをGraphStructureモデルに変換
            draft_graph = GraphStructure(
                direction=parsed_data.get("direction", "TD"),
                nodes={n["id"]: Node(**n) for n in parsed_data["nodes"]},
                edges=[Edge(**e) for e in parsed_data["edges"]]
            )
            logger.info(f"   Parsed: {len(draft_graph.nodes)} nodes, "
                         f"{len(draft_graph.edges)} edges")
        except Exception as e:
            logger.error(f"Failed to parse Mermaid with Node bridge: {e}")
            # フォールバック処理を入れるか、エラーで落とすか
            # ここではエラーとして返す
            return f"Error parsing draft: {e}"

        # 確信度が十分高ければRefineスキップ
        if draft.confidence >= 0.95 and not draft.uncertain_points:
            logger.info("✅ High confidence — skipping refine")
            # パーサーで正規化してから返す（LLM生出力の構文崩壊を防ぐ）
            return draft_graph.to_mermaid()

        # ===== Phase 2: Refine (構造的差分適用) =====
        refined_code = self._phase_refine(
            image_path, img_w, img_h, draft, draft_graph
        )
        logger.info(f"✅ Final: {len(refined_code)} chars")

        return refined_code

    # -----------------------------------------------------------------
    # Phase 1: Draft — 全体画像からMermaid一発生成 + 自己レビュー
    # -----------------------------------------------------------------

    def _phase_draft(self, image_path: str, img_w: int, img_h: int) -> DraftOutput:
        logger.info("=" * 50 + " Phase 1: DRAFT")

        image_content = self._load_image(image_path)

        response = self.llm.invoke([
            SystemMessage(content=f"""You are an expert at converting flowchart images to Mermaid diagrams.

Image size: {img_w}x{img_h} pixels.

**TASK:** Generate a Mermaid diagram AND honestly assess your uncertainty.

Output ONLY this JSON (no other text):
{{
  "mermaid": "graph TD\\n    A[Start] --> B[Process]\\n    ...",
  "confidence": 0.85,
  "uncertain_points": [
    {{
      "id": "U1",
      "description": "Unsure how may arrows go from Node C, in Mermaid I suggest 2 lines.",
      "location": "center-right area",
      "crop_x": 400, "crop_y": 300, "crop_w": 200, "crop_h": 150
    }},
    {{
      "id": "U2",
      "description": "Faint arrow — unsure if node E connects to F or G",
      "location": "bottom-left",
      "crop_x": 50, "crop_y": 500, "crop_w": 250, "crop_h": 200
    }}
  ]
}}

**Rules for the mermaid field:**
- Use actual newlines (\\n) to separate lines.
- Reproduce the flowchart structure as accurately as possible.
- For unclear labels, write your best guess.
- Use double quote for label. (e.g. Write A["Some node (extra)"], NOT A[Any Node])
- Double quote IN (not arround) label must be written as `&quot;`. (e.g. A["Some node about &quot;Bob&quot;"])
- **CRITICAL: Transcribe the text inside the node exactly. Do NOT use the node ID as the label. (e.g. Write A["Select Option"], NOT A[A], NOT A[SO])**

**Rules for uncertain_points:**
- List ONLY genuinely uncertain items (unclear if Edge label or Node, complex lines, faint lines, ambiguous connections).
- Do NOT Miss list things.
- For each point, provide crop coordinates (x, y, w, h) in the original image where you'd want to zoom in.
- Coordinates must be within image bounds ({img_w}x{img_h}).
- Max 20 Items.

**Rules for confidence:**
- 0.9+ = all labels readable, all connections clear
- 0.7-0.9 = mostly clear, a few uncertain labels or connections
- Below 0.7 = significant parts unclear
"""),
            HumanMessage(content=image_content)
        ])

        try:
            data = self._parse_json(response.content)
        except Exception as e:
            logger.warning(f"Draft JSON parse failed: {e}")
            # JSONパース失敗 → Mermaid直接抽出を試みる
            mermaid = self._extract_mermaid(response.content)
            return DraftOutput(mermaid_code=mermaid, confidence=0.5)

        mermaid_raw = data.get("mermaid", "")
        # エスケープされた改行を実際の改行に変換
        # mermaid_code = mermaid_raw.replace("\\n", "\n")
        mermaid_code = mermaid_raw

        uncertain_points = []
        for u in data.get("uncertain_points", [])[:self.MAX_REFINE_CHECKS]:
            uncertain_points.append(UncertainPoint(
                id=u.get("id", "U?"),
                description=u.get("description", ""),
                location=u.get("location", ""),
                crop_x=self._clamp(u.get("crop_x", 0), 0, img_w - 10),
                crop_y=self._clamp(u.get("crop_y", 0), 0, img_h - 10),
                crop_w=min(u.get("crop_w", 200), img_w),
                crop_h=min(u.get("crop_h", 200), img_h),
            ))

        confidence = float(data.get("confidence", 0.5))

        logger.info(f"   Mermaid lines: {mermaid_code.count(chr(10)) + 1}")
        logger.info(f"   Confidence: {confidence:.0%}")
        for u in uncertain_points:
            logger.info(f"   ❓ {u.id}: {u.description}")

        return DraftOutput(
            mermaid_code=mermaid_code,
            confidence=confidence,
            uncertain_points=uncertain_points
        )

    # -----------------------------------------------------------------
    # Phase 2: Refine — 構造的差分適用
    # -----------------------------------------------------------------

    def _phase_refine(self, image_path: str, img_w: int, img_h: int,
                      draft: DraftOutput, draft_graph: GraphStructure) -> str:
        logger.info("=" * 50 + " Phase 2: REFINE")

        if not draft.uncertain_points:
            logger.info("   No uncertain points — returning normalized draft")
            return draft_graph.to_mermaid()

        # 各疑問点をcropで確認
        for u in draft.uncertain_points:
            logger.info(f"   🔍 Checking {u.id}: {u.description}")
            u.resolution = self._check_uncertain_point(
                image_path, img_w, img_h, u, draft.mermaid_code
            )
            logger.info(f"      ✅ {u.resolution[:100]}")

        # 修正が必要な箇所を抽出
        corrections = [u for u in draft.uncertain_points
                       if u.resolution and "Correction:" in u.resolution]

        if not corrections:
            logger.info("   No corrections needed — returning normalized draft")
            return draft_graph.to_mermaid()

        # グラフ操作コマンドとして修正を適用
        corrected_graph = self._apply_structural_corrections(
            draft_graph, corrections, image_path
        )

        # 修正前後の差分をログ出力
        diff = draft_graph.diff(corrected_graph)
        if not diff.is_empty:
            logger.info(f"   Structural diff:\n{diff.summary()}")

        result = corrected_graph.to_mermaid()
        logger.info(f"   Applied {len(corrections)} corrections structurally")
        return result

    def _apply_structural_corrections(
        self, graph: GraphStructure, corrections: list[UncertainPoint],
        image_path: str
    ) -> GraphStructure:
        """修正をグラフ操作コマンドとして適用する"""

        import copy
        graph = copy.deepcopy(graph)  # 元のグラフを壊さない

        corrections_text = "\n".join(
            f"- {u.id}: {u.resolution}" for u in corrections
        )

        current_structure = json.dumps({
            "direction": graph.direction,
            "nodes": {nid: {"label": n.label, "shape": n.shape}
                      for nid, n in graph.nodes.items()},
            "edges": [{"src": e.src, "dst": e.dst, "label": e.label, "style": e.style}
                      for e in graph.edges]
        }, ensure_ascii=False, indent=2)

        response = self.llm.invoke([
            SystemMessage(content=f"""You have a graph structure and a list of corrections to apply.

Current graph structure:
{current_structure}

Corrections from visual verification:
{corrections_text}

Output ONLY a JSON object with an "operations" array. Each operation must be one of:

{{
  "operations": [
    {{"op": "relabel", "node_id": "B", "new_label": "Validate Input"}},
    {{"op": "reshape", "node_id": "C", "new_shape": "diamond"}},
    {{"op": "add_edge", "src": "E", "dst": "F", "label": "yes", "style": "-->"}},
    {{"op": "remove_edge", "src": "E", "dst": "G"}},
    {{"op": "add_node", "node_id": "X", "label": "New Step", "shape": "rect"}},
    {{"op": "remove_node", "node_id": "Z"}},
    {{"op": "relabel_edge", "src": "A", "dst": "B", "new_label": "OK"}}
  ]
}}

Rules:
- ONLY output operations that are justified by the corrections above.
- Do NOT change anything that is not mentioned in the corrections.
- Valid shapes: rect, diamond, round, stadium, hex, circle
- Valid styles: -->, ---, -.->, ==>, ===
- If a correction says "draft was correct", do NOT output any operation for it.
"""),
            HumanMessage(content=[{"type": "text", "text": "Apply the corrections."}])
        ])

        try:
            data = self._parse_json(response.content)
        except Exception as e:
            logger.warning(f"Structural correction parse failed: {e}")
            logger.warning(f"Falling back to draft graph (no corrections applied)")
            return graph

        # 操作を1つずつ適用
        applied = 0
        for op_data in data.get("operations", []):
            op = op_data.get("op")
            try:
                if op == "relabel":
                    nid = op_data["node_id"]
                    if nid in graph.nodes:
                        old = graph.nodes[nid].label
                        graph.nodes[nid].label = op_data["new_label"]
                        logger.info(f"      ✏️  relabel {nid}: '{old}' → '{op_data['new_label']}'")
                        applied += 1
                    else:
                        logger.warning(f"      ⚠️  relabel: node '{nid}' not found")

                elif op == "reshape":
                    nid = op_data["node_id"]
                    if nid in graph.nodes:
                        old = graph.nodes[nid].shape
                        graph.nodes[nid].shape = op_data["new_shape"]
                        logger.info(f"      ✏️  reshape {nid}: {old} → {op_data['new_shape']}")
                        applied += 1
                    else:
                        logger.warning(f"      ⚠️  reshape: node '{nid}' not found")

                elif op == "add_edge":
                    src, dst = op_data["src"], op_data["dst"]
                    # 重複チェック
                    if not any(e.src == src and e.dst == dst for e in graph.edges):
                        graph.edges.append(Edge(
                            src=src, dst=dst,
                            label=op_data.get("label", ""),
                            style=op_data.get("style", "-->")
                        ))
                        logger.info(f"      ➕ add_edge: {src} → {dst}")
                        applied += 1
                    else:
                        logger.info(f"      ⏭️  add_edge: {src} → {dst} already exists")

                elif op == "remove_edge":
                    src, dst = op_data["src"], op_data["dst"]
                    before_count = len(graph.edges)
                    graph.edges = [
                        e for e in graph.edges
                        if not (e.src == src and e.dst == dst)
                    ]
                    if len(graph.edges) < before_count:
                        logger.info(f"      ➖ remove_edge: {src} → {dst}")
                        applied += 1
                    else:
                        logger.warning(f"      ⚠️  remove_edge: {src} → {dst} not found")

                elif op == "add_node":
                    nid = op_data["node_id"]
                    if nid not in graph.nodes:
                        graph.nodes[nid] = Node(
                            id=nid,
                            label=op_data["label"],
                            shape=op_data.get("shape", "rect")
                        )
                        logger.info(f"      ➕ add_node: {nid}[{op_data['label']}]")
                        applied += 1
                    else:
                        logger.info(f"      ⏭️  add_node: {nid} already exists")

                elif op == "remove_node":
                    nid = op_data["node_id"]
                    if nid in graph.nodes:
                        graph.nodes.pop(nid)
                        # 関連エッジも除去
                        graph.edges = [e for e in graph.edges
                                       if e.src != nid and e.dst != nid]
                        logger.info(f"      ➖ remove_node: {nid}")
                        applied += 1
                    else:
                        logger.warning(f"      ⚠️  remove_node: '{nid}' not found")

                elif op == "relabel_edge":
                    src, dst = op_data["src"], op_data["dst"]
                    for e in graph.edges:
                        if e.src == src and e.dst == dst:
                            old = e.label
                            e.label = op_data.get("new_label", "")
                            logger.info(f"      ✏️  relabel_edge {src}→{dst}: "
                                        f"'{old}' → '{e.label}'")
                            applied += 1
                            break
                    else:
                        logger.warning(f"      ⚠️  relabel_edge: edge {src}→{dst} not found")

                else:
                    logger.warning(f"      ⚠️  Unknown operation: {op}")

            except (KeyError, TypeError) as e:
                logger.warning(f"      ⚠️  Skipped invalid op: {op_data} ({e})")

        logger.info(f"   Total operations applied: {applied}/{len(data.get('operations', []))}")
        return graph

    # -----------------------------------------------------------------
    # 疑問点の確認（crop + enhance）
    # -----------------------------------------------------------------

    def _check_uncertain_point(
        self, image_path: str, img_w: int, img_h: int,
        point: UncertainPoint, current_mermaid: str
    ) -> str:
        """1つの不確実箇所をcropで確認し、結論を返す"""

        # マージン追加（対象が見切れるリスクを低減）
        margin_x = int(point.crop_w * self.CROP_MARGIN_RATIO)
        margin_y = int(point.crop_h * self.CROP_MARGIN_RATIO)
        crop_x = max(0, point.crop_x - margin_x)
        crop_y = max(0, point.crop_y - margin_y)
        crop_w = min(point.crop_w + margin_x * 2, img_w - crop_x)
        crop_h = min(point.crop_h + margin_y * 2, img_h - crop_y)

        # Step 1: 通常cropで確認
        crop_path = ImageProcessor.crop_region.invoke({
            "image_path": image_path,
            "x": crop_x, "y": crop_y,
            "w": crop_w, "h": crop_h
        })

        if isinstance(crop_path, str) and crop_path.startswith("Error"):
            return f"Could not crop: {crop_path}"

        crop_content = self._load_image(crop_path)

        response = self.llm.invoke([
            SystemMessage(content=f"""You are verifying a specific part of a flowchart.

**Question:** {point.description}
**Location:** {point.location}
**Current assumption in the diagram:** (see the mermaid code below for context)

```mermaid
{current_mermaid}
```

Look at this zoomed-in crop and answer:
1. What does the text/label actually say?
2. Where do the arrows/connections actually go?
3. What correction (if any) is needed to the Mermaid code?

Output ONLY JSON:
{{
  "readable": true,
  "finding": "<what you can now see clearly>",
  "correction": "<specific change needed, or 'none' if draft was correct>"
}}
"""),
            HumanMessage(content=crop_content)
        ])

        try:
            data = self._parse_json(response.content)
            readable = data.get("readable", False)
            finding = data.get("finding", "")
            correction = data.get("correction", "none")

            # 読めなかった場合 → enhance して再トライ
            if not readable:
                return self._check_with_enhancement(
                    crop_path, point, current_mermaid
                )

            if correction and correction.lower() != "none":
                return f"{finding} → Correction: {correction}"
            else:
                return f"{finding} (draft was correct)"

        except Exception as e:
            return f"Parse error: {e}. Raw: {response.content[:200]}"

    def _check_with_enhancement(
        self, crop_path: str, point: UncertainPoint,
        current_mermaid: str
    ) -> str:
        """通常cropで読めなかった場合にenhanceして再トライ"""
        logger.info(f"      🔧 Enhancing for better readability...")

        # edge_enhancementを試す
        enhanced_path = ImageProcessor.preprocess_image.invoke({
            "image_path": crop_path,
            "method": "edge_enhancement"
        })

        if isinstance(enhanced_path, str) and enhanced_path.startswith("Error"):
            return f"Enhancement failed: {enhanced_path}"

        enhanced_content = self._load_image(enhanced_path)

        response = self.llm.invoke([
            SystemMessage(content=f"""This is an ENHANCED version of a flowchart crop.
Lines and text have been thickened for readability.

**Question:** {point.description}

Look carefully and answer:
{{
  "finding": "<what you can see>",
  "correction": "<change needed, or 'none'>"
}}
"""),
            HumanMessage(content=enhanced_content)
        ])

        try:
            data = self._parse_json(response.content)
            finding = data.get("finding", "unclear even after enhancement")
            correction = data.get("correction", "none")
            if correction and correction.lower() != "none":
                return f"(after enhancement) {finding} → Correction: {correction}"
            return f"(after enhancement) {finding}"
        except Exception:
            return f"(after enhancement) {response.content[:200]}"

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _load_image(self, path: str) -> list:
        try:
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            ext = Path(path).suffix.lower()
            mime = "image/png" if ext == ".png" else "image/jpeg"
            return [{"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}]
        except Exception as e:
            return [{"type": "text", "text": f"[Error: {e}]"}]

    def _extract_mermaid(self, text: str) -> str:
        if "```mermaid" in text:
            return text.split("```mermaid")[1].split("```")[0].strip()
        if "```" in text:
            parts = text.split("```")
            if len(parts) >= 2:
                return parts[1].strip()
        return text

    def _parse_json(self, text: str):
        text = text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        return json.loads(text.strip())

    @staticmethod
    def _clamp(val, lo, hi):
        return max(lo, min(int(val), hi))

