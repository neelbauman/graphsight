from typing import List
from dataclasses import dataclass, field

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

