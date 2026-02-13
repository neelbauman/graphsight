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

from graphsight.agent.tools import ImageProcessor


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class Node:
    """グラフのノード"""
    id: str
    label: str
    shape: str = "rect"  # rect, diamond, round, stadium, hex, circle


@dataclass
class Edge:
    """グラフのエッジ"""
    src: str
    dst: str
    label: str = ""
    style: str = "-->"  # -->, ---, -.->, ==>, ===


@dataclass
class GraphStructure:
    """正規化されたグラフ構造"""
    direction: str = "TD"  # TD, TB, LR, RL, BT
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)

    def diff(self, other: "GraphStructure") -> "GraphDiff":
        """2つのグラフの構造差分を返す"""
        d = GraphDiff()

        self_ids = set(self.nodes.keys())
        other_ids = set(other.nodes.keys())
        d.added_nodes = {nid: other.nodes[nid] for nid in other_ids - self_ids}
        d.removed_nodes = {nid: self.nodes[nid] for nid in self_ids - other_ids}

        for nid in self_ids & other_ids:
            if self.nodes[nid].label != other.nodes[nid].label:
                d.changed_labels[nid] = (self.nodes[nid].label, other.nodes[nid].label)
            if self.nodes[nid].shape != other.nodes[nid].shape:
                d.changed_shapes[nid] = (self.nodes[nid].shape, other.nodes[nid].shape)

        self_edge_set = {(e.src, e.dst) for e in self.edges}
        other_edge_set = {(e.src, e.dst) for e in other.edges}
        d.added_edges = [e for e in other.edges
                         if (e.src, e.dst) in other_edge_set - self_edge_set]
        d.removed_edges = [e for e in self.edges
                           if (e.src, e.dst) in self_edge_set - other_edge_set]

        return d

    def to_mermaid(self) -> str:
        """GraphStructureからMermaidコードを再生成

        ノード宣言とエッジ定義を分離する。
        - ノード: 全ノードを先に1回だけ宣言 (例: A[ラベル])
        - エッジ: IDのみで参照 (例: A --> B)
        これにより重複宣言によるMermaid構文の崩壊を防ぐ。
        """
        lines = [f"graph {self.direction}"]

        # 全ノードを宣言（1回だけ）
        for nid, node in self.nodes.items():
            lines.append(f"    {self._node_str(node)}")

        # エッジはIDのみで参照
        for edge in self.edges:
            if edge.label:
                lines.append(f"    {edge.src} {edge.style}|{edge.label}| {edge.dst}")
            else:
                lines.append(f"    {edge.src} {edge.style} {edge.dst}")

        return "\n".join(lines)

    @staticmethod
    def _node_str(node: Node) -> str:
        brackets = {
            "rect":    ("[", "]"),
            "round":   ("(", ")"),
            "diamond": ("{", "}"),
            "stadium": ("([", "])"),
            "hex":     ("{{", "}}"),
            "circle":  ("((", "))"),
        }
        l, r = brackets.get(node.shape, ("[", "]"))
        return f"{node.id}{l}{node.label}{r}"


@dataclass
class GraphDiff:
    """2つのグラフ間の構造差分"""
    added_nodes: dict[str, Node] = field(default_factory=dict)
    removed_nodes: dict[str, Node] = field(default_factory=dict)
    changed_labels: dict[str, tuple[str, str]] = field(default_factory=dict)
    changed_shapes: dict[str, tuple[str, str]] = field(default_factory=dict)
    added_edges: list[Edge] = field(default_factory=list)
    removed_edges: list[Edge] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not any([
            self.added_nodes, self.removed_nodes,
            self.changed_labels, self.changed_shapes,
            self.added_edges, self.removed_edges
        ])

    def summary(self) -> str:
        parts = []
        for nid, n in self.added_nodes.items():
            parts.append(f"ADD node {nid}[{n.label}]")
        for nid, n in self.removed_nodes.items():
            parts.append(f"REMOVE node {nid}[{n.label}]")
        for nid, (old, new) in self.changed_labels.items():
            parts.append(f"RELABEL {nid}: '{old}' → '{new}'")
        for nid, (old, new) in self.changed_shapes.items():
            parts.append(f"RESHAPE {nid}: {old} → {new}")
        for e in self.added_edges:
            parts.append(f"ADD edge {e.src} → {e.dst}")
        for e in self.removed_edges:
            parts.append(f"REMOVE edge {e.src} → {e.dst}")
        return "\n".join(parts) if parts else "(no changes)"


@dataclass
class UncertainPoint:
    """ドラフト内の不確実な箇所"""
    id: str                 # "U1", "U2", ...
    description: str        # 何が不確実か
    location: str           # "top-left", "center", 座標ヒントなど
    crop_x: int = 0        # 確認用crop座標
    crop_y: int = 0
    crop_w: int = 200
    crop_h: int = 200
    resolution: str = ""    # 確認後の結論


@dataclass
class DraftResult:
    """ドラフトフェーズの出力"""
    mermaid_code: str
    confidence: float           # 全体の確信度 (0-1)
    uncertain_points: List[UncertainPoint] = field(default_factory=list)


# =============================================================================
# Mermaid Parser
# =============================================================================

class MermaidParser:
    """LLMが出力する典型的なMermaid構文をパースしてGraphStructureに変換する"""

    # ノード形状の検出パターン（マッチ順序が重要：長いパターンを先に）
    SHAPE_PATTERNS = [
        (r'\(\[(.+?)\]\)', 'stadium'),
        (r'\(\((.+?)\)\)', 'circle'),
        (r'\{\{(.+?)\}\}', 'hex'),
        (r'\{(.+?)\}',     'diamond'),
        (r'\[(.+?)\]',     'rect'),
        (r'\((.+?)\)',     'round'),
    ]

    # 矢印パターン（マッチ順序が重要：長いパターンを先に）
    ARROW_PATTERNS = [
        (r'-\.->',  '-.->'),
        (r'===',    '==='),
        (r'==>',    '==>'),
        (r'-->',    '-->'),
        (r'---',    '---'),
    ]

    @classmethod
    def _preprocess_line(cls, line: str) -> str:
        """LLMが出力する非標準なエッジ構文を標準形に正規化する。

        LLMはしばしば以下のような非標準構文を出力する:
          D --|開示も求める| --> E    (ラベルが矢印の前にパイプで囲まれている)
          D --|Yes| E               (矢印なしのパイプラベル)

        これらを標準的なMermaid構文に変換する:
          D -->|開示も求める| E
          D ---|Yes| E
        """
        # --|label| --> を -->|label| に変換
        line = re.sub(r'\s*--\|(.+?)\|\s*-->', r' -->|\1|', line)
        # --|label| --- を ---|label| に変換
        line = re.sub(r'\s*--\|(.+?)\|\s*---', r' ---|\1|', line)
        # --|label| -.-> を -.->|label| に変換
        line = re.sub(r'\s*--\|(.+?)\|\s*-\.->',r' -.->|\1|', line)
        # --|label| ==> を ==>|label| に変換
        line = re.sub(r'\s*--\|(.+?)\|\s*==>', r' ==>|\1|', line)
        # --|label| (矢印なし、次がノード) を ---|label| に変換
        line = re.sub(r'\s*--\|(.+?)\|\s+(?!-->|---|-\.->|==>|===)', r' ---|\1| ', line)
        return line

    @classmethod
    def parse(cls, code: str) -> GraphStructure:
        """Mermaidコード文字列をGraphStructureにパースする"""
        graph = GraphStructure()
        lines = code.strip().splitlines()

        for line in lines:
            stripped = cls._preprocess_line(line.strip())

            # 空行・コメント行をスキップ
            if not stripped or stripped.startswith("%%"):
                continue

            # graph direction
            m = re.match(r'^graph\s+(TD|TB|LR|RL|BT)', stripped)
            if m:
                graph.direction = m.group(1)
                continue

            # flowchart direction (flowchart キーワードもサポート)
            m = re.match(r'^flowchart\s+(TD|TB|LR|RL|BT)', stripped)
            if m:
                graph.direction = m.group(1)
                continue

            # style / classDef 行はスキップ（構造には影響しない）
            if stripped.startswith("style ") or stripped.startswith("classDef "):
                continue

            # subgraph / end はスキップ（将来拡張ポイント）
            if stripped.startswith("subgraph ") or stripped == "end":
                continue

            # エッジ行を試す
            edge_parsed = cls._try_parse_edge(stripped, graph)
            if edge_parsed:
                continue

            # 単独ノード宣言を試す
            cls._try_parse_standalone_node(stripped, graph)

        return graph

    # Mermaidのインラインラベル構文: "A -- text --> B", "A -- text --- B"
    # "--" の後にラベルテキストがあり、その後に矢印本体が来る
    INLINE_LABEL_PATTERNS = [
        # -- text --> (arrow)
        (r'^(.+?)\s+--\s+(.+?)\s+-->\s+(.+)$',  '-->'),
        # -- text --- (line)
        (r'^(.+?)\s+--\s+(.+?)\s+---\s+(.+)$',   '---'),
        # -- text -.-> (dotted arrow)
        (r'^(.+?)\s+--\s+(.+?)\s+-\.->+\s+(.+)$', '-.->'),
        # -- text ==> (thick arrow)
        (r'^(.+?)\s+--\s+(.+?)\s+==>\s+(.+)$',   '==>'),
    ]

    @classmethod
    def _try_parse_edge(cls, line: str, graph: GraphStructure) -> bool:
        """エッジ行をパースする。3つの構文をサポート:
        1. A -->|label| B      (パイプ構文)
        2. A -- label --> B    (インラインラベル構文)
        3. A --> B             (ラベルなし)
        """

        # --- 1. インラインラベル構文を最優先で試す ---
        # "A -- text --> B" を先にマッチしないと、
        # "-->" だけが矢印として認識され "A -- text" がノード化してしまう
        for pattern, arrow_style in cls.INLINE_LABEL_PATTERNS:
            m = re.match(pattern, line)
            if m:
                src = cls._parse_node_ref(m.group(1).strip(), graph)
                edge_label = m.group(2).strip()
                dst = cls._parse_node_ref(m.group(3).strip(), graph)
                graph.edges.append(Edge(
                    src=src, dst=dst, label=edge_label, style=arrow_style
                ))
                return True

        # --- 2. パイプ構文: A -->|label| B ---
        for arrow_re, arrow_style in cls.ARROW_PATTERNS:
            pattern = rf'^(.+?)\s*{arrow_re}\s*\|(.+?)\|\s*(.+)$'
            m = re.match(pattern, line)
            if m:
                src = cls._parse_node_ref(m.group(1).strip(), graph)
                edge_label = m.group(2).strip()
                dst = cls._parse_node_ref(m.group(3).strip(), graph)
                graph.edges.append(Edge(
                    src=src, dst=dst, label=edge_label, style=arrow_style
                ))
                return True

        # --- 3. ラベルなし: A --> B ---
        for arrow_re, arrow_style in cls.ARROW_PATTERNS:
            pattern = rf'^(.+?)\s*{arrow_re}\s*(.+)$'
            m = re.match(pattern, line)
            if m:
                src_text = m.group(1).strip()
                dst_text = m.group(2).strip()
                # src OR dst にまだ矢印が含まれている場合はチェーン行
                if cls._contains_arrow(src_text) or cls._contains_arrow(dst_text):
                    return cls._parse_chained_edges(line, graph)
                src = cls._parse_node_ref(src_text, graph)
                dst = cls._parse_node_ref(dst_text, graph)
                graph.edges.append(Edge(src=src, dst=dst, style=arrow_style))
                return True

        return False

    @classmethod
    def _contains_arrow(cls, text: str) -> bool:
        """テキスト内に矢印パターンが含まれているか"""
        for arrow_re, _ in cls.ARROW_PATTERNS:
            if re.search(arrow_re, text):
                return True
        return False

    @classmethod
    def _parse_chained_edges(cls, line: str, graph: GraphStructure) -> bool:
        """A --> B --> C のようなチェーンを複数エッジに分解する"""
        # 矢印で分割
        parts = []
        arrows = []
        remaining = line
        while remaining:
            matched = False
            for arrow_re, arrow_style in cls.ARROW_PATTERNS:
                m = re.search(rf'\s*{arrow_re}\s*', remaining)
                if m:
                    part = remaining[:m.start()].strip()
                    if part:
                        parts.append(part)
                        arrows.append(arrow_style)
                    remaining = remaining[m.end():]
                    matched = True
                    break
            if not matched:
                if remaining.strip():
                    parts.append(remaining.strip())
                break

        if len(parts) < 2:
            return False

        # 連続するノードペアをエッジとして登録
        node_ids = [cls._parse_node_ref(p, graph) for p in parts]
        for i in range(len(node_ids) - 1):
            style = arrows[i] if i < len(arrows) else "-->"
            graph.edges.append(Edge(src=node_ids[i], dst=node_ids[i + 1], style=style))

        return True

    @classmethod
    def _parse_node_ref(cls, text: str, graph: GraphStructure) -> str:
        """'A[Some Label]' → ノード登録してIDを返す。'A' だけなら既存参照。"""
        for pattern, shape in cls.SHAPE_PATTERNS:
            # ID + shape: "A[Label]"
            m = re.match(rf'^([A-Za-z_]\w*)\s*' + pattern + r'$', text)
            if m:
                nid = m.group(1)
                label = m.group(2).strip()
                # 初出時のみ登録（最初のラベルを正とする）
                if nid not in graph.nodes:
                    graph.nodes[nid] = Node(id=nid, label=label, shape=shape)
                return nid

        # IDのみ（形状なし）
        m = re.match(r'^([A-Za-z_]\w*)$', text.strip())
        if m:
            nid = m.group(1)
            if nid not in graph.nodes:
                graph.nodes[nid] = Node(id=nid, label=nid, shape="rect")
            return nid

        # テキストがエッジラベル残骸を含んでいる場合
        # (例: "E -- 任意開示確実" "D --|開示も求める|")
        # 先頭のIDだけを抽出する
        m = re.match(r'^([A-Za-z_]\w*)\s*--', text)
        if m:
            nid = m.group(1)
            if nid not in graph.nodes:
                graph.nodes[nid] = Node(id=nid, label=nid, shape="rect")
            return nid

        # パースできない場合 → テキスト自体をサニタイズしてIDにする
        safe_id = re.sub(r'[^A-Za-z0-9_]', '_', text)[:20]
        if not safe_id or safe_id[0].isdigit():
            safe_id = "N_" + safe_id
        if safe_id not in graph.nodes:
            graph.nodes[safe_id] = Node(id=safe_id, label=text, shape="rect")
        return safe_id

    @classmethod
    def _try_parse_standalone_node(cls, line: str, graph: GraphStructure):
        """単独のノード宣言行をパース"""
        for pattern, shape in cls.SHAPE_PATTERNS:
            m = re.match(rf'^([A-Za-z_]\w*)\s*' + pattern + r'$', line)
            if m:
                nid = m.group(1)
                label = m.group(2).strip()
                if nid not in graph.nodes:
                    graph.nodes[nid] = Node(id=nid, label=label, shape=shape)
                return


# =============================================================================
# Agent
# =============================================================================

class GraphSightAgent:

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
        draft_graph = MermaidParser.parse(draft.mermaid_code)
        logger.info(f"   Parsed: {len(draft_graph.nodes)} nodes, "
                     f"{len(draft_graph.edges)} edges")

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

    def _phase_draft(self, image_path: str, img_w: int, img_h: int) -> DraftResult:
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

**Rules for uncertain_points:**
- List ONLY genuinely uncertain items (unclear if Label or Node, complex lines, faint lines, ambiguous connections).
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
            return DraftResult(mermaid_code=mermaid, confidence=0.5)

        mermaid_raw = data.get("mermaid", "")
        # エスケープされた改行を実際の改行に変換
        mermaid_code = mermaid_raw.replace("\\n", "\n")

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

        return DraftResult(
            mermaid_code=mermaid_code,
            confidence=confidence,
            uncertain_points=uncertain_points
        )

    # -----------------------------------------------------------------
    # Phase 2: Refine — 構造的差分適用
    # -----------------------------------------------------------------

    def _phase_refine(self, image_path: str, img_w: int, img_h: int,
                      draft: DraftResult, draft_graph: GraphStructure) -> str:
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


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python core.py <image_path>")
        sys.exit(1)

    agent = GraphSightAgent()
    result = agent.run(sys.argv[1])
    print(f"\n{'='*60}")
    print(f"```mermaid\n{result}\n```")

