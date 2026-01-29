#!/usr/bin/env python3
"""
Mermaid記法からグラフ構造（nodes/edges）をJSON形式で出力するツール
"""

import re
import json
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Node:
    id: str
    label: str
    desc: str = ""
    tools: list = field(default_factory=list)


@dataclass
class Edge:
    source: str
    target: str


class MermaidParser:
    """Mermaidのグラフ記法をパースするクラス"""
    
    # ノード形状のパターン（括弧の種類で形状が決まる）
    NODE_SHAPES = {
        r'\(\[([^\]]*)\]\)': 'stadium',    # ([label]) - stadium/pill shape
        r'\[\[([^\]]*)\]\]': 'subroutine', # [[label]]
        r'\(\(([^\)]*)\)\)': 'circle',     # ((label))
        r'\[([^\]]*)\]': 'rectangle',      # [label]
        r'\(([^\)]*)\)': 'rounded',        # (label)
        r'\{([^\}]*)\}': 'diamond',        # {label}
        r'\[/([^\]]*)/\]': 'parallelogram', # [/label/]
        r'>([^\]]*)\]': 'asymmetric',      # >label]
    }
    
    # エッジのパターン
    EDGE_PATTERNS = [
        r'-->',      # 矢印
        r'---',      # 線のみ
        r'-\.->',    # 点線矢印
        r'==>',      # 太い矢印
        r'--[^-]',   # ラベル付きエッジの開始
    ]
    
    def __init__(self):
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
    
    def parse(self, mermaid_text: str) -> tuple[list[dict], list[dict]]:
        """
        Mermaidテキストをパースしてnodes, edgesを返す
        """
        self.nodes = {}
        self.edges = []
        
        lines = mermaid_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            
            # コメントや空行をスキップ
            if not line or line.startswith('%%'):
                continue
            
            # グラフ定義行をスキップ
            if re.match(r'^(graph|flowchart|subgraph|end)\s*', line, re.IGNORECASE):
                # subgraphの場合はラベルを抽出することも可能
                continue
            
            # エッジを含む行を処理
            self._parse_line(line)
        
        # 結果を辞書形式で返す
        nodes_list = [asdict(node) for node in self.nodes.values()]
        edges_list = [asdict(edge) for edge in self.edges]
        
        return nodes_list, edges_list
    
    def _parse_line(self, line: str):
        """1行をパースしてノードとエッジを抽出"""
        
        # エッジパターン（ラベル付きエッジも含む）
        # -->|label|, -->|label|>, --|label|-->, -->, ---,-.->, ==>
        edge_patterns = [
            r'-->\|[^|]*\|>?',   # -->|Yes|  or  -->|Yes|>
            r'-->\|[^|]*\|',     # -->|label|
            r'-\.->\|[^|]*\|',   # -.->|label|
            r'==>\|[^|]*\|',     # ==>|label|
            r'--\|[^|]*\|-->',   # --|label|-->
            r'-->',              # -->
            r'---',              # ---
            r'-\.->',            # -.->
            r'==>',              # ==>
        ]
        
        # 結合したパターン
        edge_pattern = '(' + '|'.join(edge_patterns) + ')'
        
        parts = re.split(edge_pattern, line)
        
        if len(parts) == 1:
            # エッジがない場合、ノード定義のみ
            self._extract_node(parts[0].strip())
        else:
            # エッジがある場合
            prev_node_id = None
            
            for i, part in enumerate(parts):
                part = part.strip()
                if not part:
                    continue
                
                # エッジパターンかどうかチェック
                if re.match(edge_pattern, part):
                    continue
                
                # ノードを抽出
                node_id = self._extract_node(part)
                
                # 前のノードがあればエッジを作成
                if prev_node_id and node_id:
                    self.edges.append(Edge(source=prev_node_id, target=node_id))
                
                prev_node_id = node_id
    
    def _extract_node(self, node_str: str) -> Optional[str]:
        """ノード文字列からID、ラベルを抽出"""
        node_str = node_str.strip()
        if not node_str:
            return None
        
        # 各形状パターンを試す
        for pattern, shape in self.NODE_SHAPES.items():
            # ID + 形状パターン
            full_pattern = rf'^([A-Za-z_][A-Za-z0-9_]*)\s*{pattern}$'
            match = re.match(full_pattern, node_str)
            
            if match:
                node_id = match.group(1)
                label = match.group(2).strip()
                
                if node_id not in self.nodes:
                    self.nodes[node_id] = Node(id=node_id, label=label)
                elif not self.nodes[node_id].label and label:
                    self.nodes[node_id].label = label
                
                return node_id
        
        # 形状なし（IDのみ）
        id_only_match = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)$', node_str)
        if id_only_match:
            node_id = id_only_match.group(1)
            if node_id not in self.nodes:
                self.nodes[node_id] = Node(id=node_id, label=node_id)
            return node_id
        
        return None
    
    def set_node_description(self, node_id: str, desc: str):
        """ノードに説明を設定"""
        if node_id in self.nodes:
            self.nodes[node_id].desc = desc
    
    def set_node_tools(self, node_id: str, tools: list):
        """ノードにツールを設定"""
        if node_id in self.nodes:
            self.nodes[node_id].tools = tools


def mermaid_to_graph(mermaid_text: str, 
                     descriptions: dict[str, str] = None,
                     tools: dict[str, list] = None) -> dict:
    """
    Mermaidテキストをグラフ構造に変換するメイン関数
    
    Args:
        mermaid_text: Mermaid形式のグラフ定義
        descriptions: ノードIDをキーとした説明文の辞書（オプション）
        tools: ノードIDをキーとしたツールリストの辞書（オプション）
    
    Returns:
        {"nodes": [...], "edges": [...]} 形式の辞書
    """
    parser = MermaidParser()
    nodes, edges = parser.parse(mermaid_text)
    
    # 説明を追加
    if descriptions:
        for node in nodes:
            if node['id'] in descriptions:
                node['desc'] = descriptions[node['id']]
    
    # ツールを追加
    if tools:
        for node in nodes:
            if node['id'] in tools:
                node['tools'] = tools[node['id']]
    
    return {"nodes": nodes, "edges": edges}


def generate_python_code(graph: dict, 
                         nodes_var: str = "nodes",
                         edges_var: str = "edges") -> str:
    """
    グラフ構造からPythonコードを生成
    """
    lines = []
    
    # ノード定義
    lines.append(f"{nodes_var} = [")
    for node in graph['nodes']:
        tools_str = node.get('tools', [])
        if isinstance(tools_str, list) and tools_str:
            tools_repr = f"[{', '.join(str(t) for t in tools_str)}]"
        else:
            tools_repr = "[]"
        
        lines.append("    {")
        lines.append(f'        "id": "{node["id"]}",')
        lines.append(f'        "label": "{node["label"]}",')
        lines.append(f'        "desc": "{node.get("desc", "")}",')
        lines.append(f'        "tools": {tools_repr}')
        lines.append("    },")
    lines.append("]")
    
    lines.append("")
    
    # エッジ定義
    lines.append(f"{edges_var} = [")
    for edge in graph['edges']:
        lines.append(f'    {{"source": "{edge["source"]}", "target": "{edge["target"]}"}},')
    lines.append("]")
    
    return '\n'.join(lines)


# ========== サンプル使用例 ==========

if __name__ == "__main__":
    # サンプル1: 基本的なフローチャート
    sample_mermaid_1 = """
    graph TD
        triage[トリアージルーム] --> web_layer[Webサーバー層]
        web_layer --> app_layer[アプリサーバー層]
        web_layer --> triage
        app_layer --> db_layer[DBサーバー層]
        app_layer --> web_layer
        db_layer --> app_layer
        db_layer --> escalation[エスカレーション]
    """
    
    # サンプル2: 条件分岐付きフローチャート（ラベル付きエッジ）
    sample_mermaid_2 = """
    graph TD
        start([Start]) --> define_job[Define job description]
        define_job --> request_hr[Request personnel to HR]
        request_hr --> classify_hiring[Classify job hiring]
        classify_hiring --> regular_process{Is it a regular process?}
        regular_process -->|Yes| hiring_plan[Create hiring plan]
        regular_process -->|No| special_hiring[Special hiring process]
        special_hiring --> review_prev[Review previous candidate]
        review_prev --> candidate_suitable{Is the candidate suitable for the job?}
        candidate_suitable -->|Yes| setup_meeting[Setup meeting]
        candidate_suitable -->|No| create_posting[Create job posting]
        create_posting --> found_suitable{Found a suitable candidate?}
        found_suitable -->|Yes| setup_meeting
        found_suitable -->|No| hiring_plan
        hiring_plan --> post_online[Create job posting online]
        post_online --> collect_resumes[Collect submitted resumes]
        collect_resumes --> setup_meeting
        setup_meeting --> initial_interview[Conduct initial Interview]
        initial_interview --> list_questions[List all interview questions]
        list_questions --> final_interview[Conduct final Interview]
        final_interview --> reference_checked{Reference checked?}
        reference_checked -->|Yes| shortlist[Shortlist candidates]
        reference_checked -->|No| final_interview
        shortlist --> finalize_rate[Finalize job rate]
        finalize_rate --> send_offer[Send job offer]
        send_offer --> candidate_accept{Candidate accept the offer?}
        candidate_accept -->|Yes| end_success([End - Success])
        candidate_accept -->|No| post_online
    """
    
    print("=" * 60)
    print("サンプル1: 基本的なフローチャート")
    print("=" * 60)
    
    # ノードの説明
    descriptions = {
        "triage": "障害対応の起点。まずはHTTPステータスを確認し、問題の切り分けを行う場所。",
        "web_layer": "NginxなどのWebサーバーの状態を確認する場所。",
        "app_layer": "アプリケーションのログを確認し、エラー原因を特定する場所。DBエラーならDB層へ移動する。",
        "db_layer": "データベースのプロセス確認や再起動を行う場所。",
        "escalation": "自動復旧できない場合の最終通知場所。"
    }
    
    # ツール定義（文字列として。実際は関数参照）
    tools = {
        "triage": ["check_http_status"],
        "web_layer": ["check_nginx_status"],
        "app_layer": ["read_app_logs", "check_db_connection"],
        "db_layer": ["check_db_connection", "execute_db_restart"],
        "escalation": ["escalate_to_human"]
    }
    
    result1 = mermaid_to_graph(sample_mermaid_1, descriptions, tools)
    print(json.dumps(result1, ensure_ascii=False, indent=2))
    
    print("\n" + "=" * 60)
    print("サンプル2: 条件分岐付きフローチャート")
    print("=" * 60)
    
    result2 = mermaid_to_graph(sample_mermaid_2)
    print(json.dumps(result2, ensure_ascii=False, indent=2))
    
    # 条件分岐ノードからのエッジを確認
    print("\n" + "=" * 60)
    print("条件分岐ノードからのエッジ確認:")
    print("=" * 60)
    decision_nodes = ["regular_process", "candidate_suitable", "found_suitable", 
                      "reference_checked", "candidate_accept"]
    for node_id in decision_nodes:
        outgoing = [e for e in result2['edges'] if e['source'] == node_id]
        print(f"{node_id}: {len(outgoing)} outgoing edges -> {[e['target'] for e in outgoing]}")


if __name__ == "__main__":
    with open("samples/sample-7.mmd", "r", encoding='utf-8') as f:
        mermaid = f.read()

    result = mermaid_to_graph(mermaid)

    with open("samples/sample-7.json", "w", encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

