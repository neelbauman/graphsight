import re
from typing import Dict, List, Tuple

class MermaidParser:
    """Mermaidコードを簡易解析し、接続関係を抽出するヘルパー"""
    
    # ノード定義の正規表現 (例: A[ラベル], B(ラベル), C{ラベル})
    NODE_PATTERN = re.compile(r'([a-zA-Z0-9_]+)\s*(?:\[| \(|\{\{?|\(\(|>)(.*?)(?:\]| \)|\}\}?|\)\))')
    
    # エッジ定義の正規表現 (例: A --> B, A -->|Yes| B)
    # Group 1: From, Group 2: Arrow, Group 3: Label(opt), Group 4: To
    EDGE_PATTERN = re.compile(r'([a-zA-Z0-9_]+)\s*(-+\.|={2,}|-+)>\s*(?:\|(.*?)\|\s*)?([a-zA-Z0-9_]+)')

    @staticmethod
    def parse_structure(code: str) -> Dict[str, Any]:
        nodes = {}
        edges = []
        
        # コードを行ごとに処理
        for line in code.split('\n'):
            line = line.strip()
            # コメントやグラフ宣言をスキップ
            if not line or line.startswith('%%') or line.startswith(('graph', 'flowchart')):
                continue
                
            # ノードラベルの抽出
            n_match = MermaidParser.NODE_PATTERN.search(line)
            if n_match:
                node_id, label = n_match.groups()
                nodes[node_id] = label.strip('"\'')
            
            # 接続の抽出 (1行に複数の接続がある場合も考慮して findall 的な処理が必要だが簡略化)
            e_match = MermaidParser.EDGE_PATTERN.search(line)
            if e_match:
                u, arrow, label, v = e_match.groups()
                edges.append({
                    "from": u,
                    "to": v,
                    "label": label if label else None,
                    "raw": line
                })
                # ノードリストに未登録ならIDだけ登録
                if u not in nodes: nodes[u] = "Unknown Label"
                if v not in nodes: nodes[v] = "Unknown Label"

        return {"nodes": nodes, "edges": edges}

    @staticmethod
    def get_outgoing(code: str, node_id: str) -> List[str]:
        """指定されたノードから出るエッジのリストを返す"""
        structure = MermaidParser.parse_structure(code)
        outgoing = []
        for e in structure["edges"]:
            if e["from"] == node_id:
                to_label = structure["nodes"].get(e["to"], "N/A")
                label_info = f" --[{e['label']}]--> " if e['label'] else " --> "
                outgoing.append(f"{label_info} {e['to']}[{to_label}]")
        return outgoing

