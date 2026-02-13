import re
from loguru import logger
from .models import (
    GraphStructure,
    Node,
    Edge,
)


class MermaidParser:
    """LLMが出力する典型的なMermaid構文をパースしてGraphStructureに変換する"""

    # ノード形状の検出パターン（マッチ順序が重要：長いパターンを先に）
    SHAPE_PATTERNS = [
        (r'\(\[((?:.|\\n)+?)\]\)', 'stadium'), # ([...])
        (r'\(\(((?:.|\\n)+?)\)\)', 'circle'),  # ((...))
        (r'\{\{((?:.|\\n)+?)\}\}', 'hex'),     # {{...}}
        (r'\{((?:.|\\n)+?)\}',     'diamond'), # {...}
        (r'\[((?:.|\\n)+?)\]',     'rect'),    # [...]
        (r'\(((?:.|\\n)+?)\)',     'round'),   # (...)
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

        fallback_events = []

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
            edge_parsed = cls._try_parse_edge(stripped, graph, fallback_events)
            if edge_parsed:
                continue

            # 単独ノード宣言を試す
            cls._try_parse_standalone_node(stripped, graph)

        if fallback_events:
            logger.warning(f"⚠️  MermaidParser triggered fallback for {len(fallback_events)} items:")
            for text in fallback_events:
                logger.warning(f"   - Fallback input: '{text}'")

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
    def _try_parse_edge(cls, line: str, graph: GraphStructure, fallback_events: list) -> bool:
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
                src = cls._parse_node_ref(m.group(1).strip(), graph, fallback_events)
                edge_label = m.group(2).strip()
                dst = cls._parse_node_ref(m.group(3).strip(), graph, fallback_events)
                graph.edges.append(Edge(
                    src=src, dst=dst, label=edge_label, style=arrow_style
                ))
                return True

        # --- 2. パイプ構文: A -->|label| B ---
        for arrow_re, arrow_style in cls.ARROW_PATTERNS:
            pattern = rf'^(.+?)\s*{arrow_re}\s*\|(.+?)\|\s*(.+)$'
            m = re.match(pattern, line)
            if m:
                src = cls._parse_node_ref(m.group(1).strip(), graph, fallback_events)
                edge_label = m.group(2).strip()
                dst = cls._parse_node_ref(m.group(3).strip(), graph, fallback_events)
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
                    return cls._parse_chained_edges(line, graph, fallback_events)
                src = cls._parse_node_ref(src_text, graph, fallback_events)
                dst = cls._parse_node_ref(dst_text, graph, fallback_events)
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
    def _parse_chained_edges(cls, line: str, graph: GraphStructure, fallback_events: list | None = None) -> bool:
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
        node_ids = [cls._parse_node_ref(p, graph, fallback_events) for p in parts]
        for i in range(len(node_ids) - 1):
            style = arrows[i] if i < len(arrows) else "-->"
            graph.edges.append(Edge(src=node_ids[i], dst=node_ids[i + 1], style=style))

        return True

    @classmethod
    def _parse_node_ref(cls, text: str, graph: GraphStructure, fallback_events: list = None) -> str:
        """'A[Some Label]' → ノード登録してIDを返す。'A' だけなら既存参照。"""
        
        # 1. Strict Parsing (厳密な正規表現: 閉じカッコあり)
        for pattern, shape in cls.SHAPE_PATTERNS:
            # 改行またぎ対応の正規表現 ((?:.|\\n)+?) を使用
            m = re.match(rf'^([A-Za-z_]\w*)\s*' + pattern + r'$', text)
            if m:
                nid = m.group(1)
                raw_label = m.group(2).strip()
                # クォート除去 ("label" -> label)
                if (raw_label.startswith('"') and raw_label.endswith('"')) or \
                   (raw_label.startswith("'") and raw_label.endswith("'")):
                    label = raw_label[1:-1]
                else:
                    label = raw_label
                
                if nid not in graph.nodes:
                    graph.nodes[nid] = Node(id=nid, label=label, shape=shape)
                return nid

        # 2. Heuristic Parsing (救済措置: 閉じカッコ欠損/改行分割への対応)
        # 例: "R[電話会社に" (ここで改行されて切れている)
        # 開始カッコのパターン: ([Or (( Or {{ Or { Or [ Or (
        heuristic_match = re.match(r'^([A-Za-z_]\w*)\s*(\(\[|\(\(|\{\{|\{|\[|\()((?:.|\\n)*)', text)
        if heuristic_match:
            nid = heuristic_match.group(1)
            bracket = heuristic_match.group(2)
            raw_content = heuristic_match.group(3).strip()
            
            # 末尾のゴミ（閉じカッコの断片など）があれば除去
            label = re.sub(r'(\]\)|\]|\)\)|\}|\}\})$', '', raw_content)

            # クォート除去
            if (label.startswith('"') and label.endswith('"')) or \
               (label.startswith("'") and label.endswith("'")):
                label = label[1:-1]

            # 開始カッコから形状を決定
            shape_map = {
                "([": "stadium", "((": "circle", "{{": "hex", 
                "{": "diamond", "[": "rect", "(": "round"
            }
            shape = shape_map.get(bracket, "rect")
            
            if nid not in graph.nodes:
                # ログで救済を通知（デバッグ用）
                # logger.debug(f"🔧 Heuristically parsed node: {nid}[{label}...] (incomplete line)")
                graph.nodes[nid] = Node(id=nid, label=label, shape=shape)
            return nid

        # 3. IDのみ (形状なし)
        m = re.match(r'^([A-Za-z_]\w*)$', text.strip())
        if m:
            nid = m.group(1)
            if nid not in graph.nodes:
                graph.nodes[nid] = Node(id=nid, label=nid, shape="rect")
            return nid

        # エッジラベル残骸処理 (例: "E -- text")
        m = re.match(r'^([A-Za-z_]\w*)\s*--', text)
        if m:
            nid = m.group(1)
            if nid not in graph.nodes:
                graph.nodes[nid] = Node(id=nid, label=nid, shape="rect")
            return nid

        # 4. Fallback (最終手段: 強制ID化)
        if fallback_events is not None:
            fallback_events.append(text)

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


