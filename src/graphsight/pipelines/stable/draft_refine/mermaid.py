import re
from .models import (
    GraphStructure,
    Node,
    Edge,
)


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


