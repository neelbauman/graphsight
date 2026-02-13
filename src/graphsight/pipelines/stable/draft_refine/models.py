from typing import List, Literal, Dict, Optional, Tuple, Set
from pydantic import BaseModel, Field, model_validator


# --- 基本要素モデル ---

class Node(BaseModel):
    """グラフのノード"""
    id: str = Field(..., description="Unique identifier for the node (alphanumeric).")
    label: str = Field(..., description="Text content explicitly written inside the node.")
    shape: Literal["rect", "diamond", "round", "stadium", "hex", "circle"] = Field(
        "rect", description="Visual shape of the node."
    )
    actor: Optional[str] = Field(
        None, 
        description="The actor, role, or swimlane responsible for this node (e.g. 'User', 'System')."
    )

class Edge(BaseModel):
    """グラフのエッジ"""
    src: str = Field(..., description="Source node ID.")
    dst: str = Field(..., description="Destination node ID.")
    label: str = Field("", description="Text label usually written on or near the arrow.")
    style: Literal["-->", "---", "-.->", "==>", "==="] = Field(
        "-->", description="Style of the connection."
    )

# --- LLM出力用モデル (Listベース) ---

class GraphSchema(BaseModel):
    """LLMがStructured Outputとして生成するためのスキーマ"""
    direction: Literal["TD", "TB", "LR", "RL", "BT"] = Field("TD", description="Flow direction.")
    nodes: List[Node] = Field(..., description="List of all nodes found in the image.")
    edges: List[Edge] = Field(..., description="List of all connections found.")

# --- アプリケーション内部用モデル (Dictベース, ロジック保持) ---

class GraphStructure(BaseModel):
    """正規化されたグラフ構造 (アプリケーション内部用)"""
    direction: str = "TD"
    nodes: Dict[str, Node] = Field(default_factory=dict)
    edges: List[Edge] = Field(default_factory=list)

    @classmethod
    def from_schema(cls, schema: GraphSchema) -> "GraphStructure":
        """LLM出力スキーマから内部構造へ変換"""
        return cls(
            direction=schema.direction,
            nodes={n.id: n for n in schema.nodes},
            edges=schema.edges
        )

    def diff(self, other: "GraphStructure") -> "GraphDiff":
        """2つのグラフの構造差分を返す"""
        d = GraphDiff()

        self_ids = set(self.nodes.keys())
        other_ids = set(other.nodes.keys())
        
        # Node differences
        d.added_nodes = {nid: other.nodes[nid] for nid in other_ids - self_ids}
        d.removed_nodes = {nid: self.nodes[nid] for nid in self_ids - other_ids}

        for nid in self_ids & other_ids:
            n1 = self.nodes[nid]
            n2 = other.nodes[nid]
            
            if n1.label != n2.label:
                d.changed_labels[nid] = (n1.label, n2.label)
            if n1.shape != n2.shape:
                d.changed_shapes[nid] = (n1.shape, n2.shape)
            # --- 追加: Actorの変更検知 ---
            if n1.actor != n2.actor:
                d.changed_actors[nid] = (n1.actor, n2.actor)

        # Edge differences
        # EdgeはIDを持たないため、(src, dst) のペアで同一性を判定（多重エッジは簡易比較）
        self_edge_set = {(e.src, e.dst) for e in self.edges}
        other_edge_set = {(e.src, e.dst) for e in other.edges}
        
        # シンプル化のため、属性違いは無視して存在有無のみチェック
        d.added_edges = [e for e in other.edges 
                         if (e.src, e.dst) not in self_edge_set]
        d.removed_edges = [e for e in self.edges 
                           if (e.src, e.dst) not in other_edge_set]
        
        return d

    def to_mermaid(self) -> str:
        """GraphStructureからMermaidコードを再生成 (Swimlane対応)"""
        lines = [f"graph {self.direction}"]
        
        # --- Actorごとのグルーピング処理 ---
        nodes_by_actor: Dict[str, List[Node]] = {}
        no_actor_nodes: List[Node] = []
        
        for node in self.nodes.values():
            if node.actor:
                if node.actor not in nodes_by_actor:
                    nodes_by_actor[node.actor] = []
                nodes_by_actor[node.actor].append(node)
            else:
                no_actor_nodes.append(node)
        
        # Subgraphs (Swimlanes) の生成
        # 一意なIDを振るためにカウンターを使用
        sg_counter = 1
        for actor_name, nodes in nodes_by_actor.items():
            # Actor名のエスケープ（簡易的）
            safe_actor_name = actor_name.replace('"', "'")
            lines.append(f"    subgraph sg_{sg_counter} [\"{safe_actor_name}\"]")
            for node in nodes:
                lines.append(f"        {self._node_str(node)}")
            lines.append("    end")
            sg_counter += 1
            
        # Actorなしノードの定義
        for node in no_actor_nodes:
            lines.append(f"    {self._node_str(node)}")
            
        # エッジ定義
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
        # Mermaidのエスケープ処理が必要ならここに追加
        safe_label = node.label.replace('"', "'") 
        return f'{node.id}{l}"{safe_label}"{r}'

# --- その他 ---

class GraphDiff(BaseModel):
    """2つのグラフ間の構造差分"""
    added_nodes: Dict[str, Node] = Field(default_factory=dict)
    removed_nodes: Dict[str, Node] = Field(default_factory=dict)
    changed_labels: Dict[str, Tuple[str, str]] = Field(default_factory=dict)
    changed_shapes: Dict[str, Tuple[str, str]] = Field(default_factory=dict)
    changed_actors: Dict[str, Tuple[Optional[str], Optional[str]]] = Field(default_factory=dict)
    added_edges: List[Edge] = Field(default_factory=list)
    removed_edges: List[Edge] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not any([
            self.added_nodes, self.removed_nodes,
            self.changed_labels, self.changed_shapes,
            self.changed_actors,
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
        for nid, (old, new) in self.changed_actors.items():
            parts.append(f"REASSIGN {nid}: Actor '{old}' → '{new}'")
        for e in self.added_edges:
            parts.append(f"ADD edge {e.src} → {e.dst}")
        for e in self.removed_edges:
            parts.append(f"REMOVE edge {e.src} → {e.dst}")
        return "\n".join(parts) if parts else "(no changes)"


class UncertainPoint(BaseModel):
    """ドラフト内の不確実な箇所"""
    id: str = Field(..., description="ID like U1, U2")
    description: str = Field(..., description="Description of uncertainty")
    location: str = Field(..., description="Rough location like 'top-left'")
    crop_x: int = 0
    crop_y: int = 0
    crop_w: int = 200
    crop_h: int = 200
    resolution: str = "" # 確認後の結論

# DraftResult も Output構造に統合するか、別途定義するか
# Structured Outputでは1つのルートモデルしか返せないので、これらをまとめるラッパーが必要

class DraftOutput(BaseModel):
    """DraftフェーズのLLM最終出力"""
    mermaid_code: str = Field(..., description="The complete Mermaid flowchart code.")
    confidence: float = Field(..., description="Overall confidence (0.0-1.0)")
    uncertain_points: List[UncertainPoint] = Field(default_factory=list)

class DraftOutputStructured(BaseModel):
    """DraftフェーズのLLM最終出力"""
    graph: GraphSchema
    confidence: float = Field(..., description="Overall confidence (0.0-1.0)")
    uncertain_points: List[UncertainPoint] = Field(default_factory=list)


class CheckResult(BaseModel):
    """Refineフェーズ: 個別の疑問点確認結果"""
    readable: bool = Field(..., description="Is the text/connection clearly visible?")
    finding: str = Field(..., description="What you actually see in the crop.")
    correction: str = Field(..., description="Specific correction needed, or 'none' if draft was correct.")


class GraphOperation(BaseModel):
    """Refineフェーズ: 1つの修正操作"""
    op: Literal["relabel", "reshape", "add_edge", "remove_edge", "add_node", "remove_node", "relabel_edge"]
    node_id: Optional[str] = None
    new_label: Optional[str] = None
    new_shape: Optional[str] = None
    src: Optional[str] = None
    dst: Optional[str] = None
    label: Optional[str] = None
    style: Optional[str] = None

class CorrectionPlan(BaseModel):
    """Refineフェーズ: 修正計画全体"""
    operations: List[GraphOperation]

