"""
GraphSight Agent v9 — Union Strategy (High Recall Edition)

Concept:
  1. Generate 3 drafts (Optimist, Pessimist, Structuralist).
  2. Normalize IDs using LLM (map everyone to Structuralist's semantic IDs).
  3. UNION MERGE:
     - If Node X is found in ANY draft, add it.
     - If Edge A->B is found in ANY draft, add it.
     - Priority for Labels: Structuralist > Optimist > Pessimist.

Dependencies:
  - langchain_openai
  - loguru
"""

import base64
import json
import re
import concurrent.futures
from typing import List, Dict, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
import sys

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

# =============================================================================
# Data Models
# =============================================================================

@dataclass
class Node:
    id: str
    label: str
    shape: str = "rect"

@dataclass
class Edge:
    src: str
    dst: str
    label: str = ""
    style: str = "-->"

@dataclass
class GraphStructure:
    direction: str = "TD"
    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)

    def to_mermaid(self) -> str:
        lines = [f"graph {self.direction}"]
        
        # Sort for consistency
        sorted_nodes = sorted(self.nodes.values(), key=lambda n: n.id)
        for node in sorted_nodes:
            lines.append(f"    {self._node_str(node)}")

        sorted_edges = sorted(self.edges, key=lambda e: (e.src, e.dst))
        for edge in sorted_edges:
            # Ensure safe output even if node is missing (though union logic prevents this)
            if edge.src in self.nodes and edge.dst in self.nodes:
                if edge.label:
                    lines.append(f"    {edge.src} {edge.style}|{edge.label}| {edge.dst}")
                else:
                    lines.append(f"    {edge.src} {edge.style} {edge.dst}")
        return "\n".join(lines)

    @staticmethod
    def _node_str(node: Node) -> str:
        brackets = {
            "rect": ("[", "]"), "round": ("(", ")"), "diamond": ("{", "}"),
            "stadium": ("([", "])"), "hex": ("{{", "}}"), "circle": ("((", "))"),
        }
        l, r = brackets.get(node.shape, ("[", "]"))
        safe_label = node.label.replace('"', '').replace('\n', ' ')
        return f'{node.id}{l}"{safe_label}"{r}'


# =============================================================================
# Mermaid Parser (Standard)
# =============================================================================

class MermaidParser:
    SHAPE_PATTERNS = [
        (r'\(\[(.+?)\]\)', 'stadium'), (r'\(\((.+?)\)\)', 'circle'),
        (r'\{\{(.+?)\}\}', 'hex'), (r'\{(.+?)\}', 'diamond'),
        (r'\[(.+?)\]', 'rect'), (r'\((.+?)\)', 'round'),
    ]
    ARROW_PATTERNS = [
        (r'-\.->', '-.->'), (r'===', '==='), (r'==>', '==>'),
        (r'-->', '-->'), (r'---', '---'),
    ]

    @classmethod
    def parse(cls, code: str) -> GraphStructure:
        graph = GraphStructure()
        lines = code.strip().splitlines()
        for line in lines:
            line = cls._preprocess(line.strip())
            if not line or line.startswith("%%") or line.startswith("classDef"): continue
            
            # Direction
            if re.match(r'^(graph|flowchart)\s+', line):
                m = re.match(r'^(graph|flowchart)\s+(.+)', line)
                if m: graph.direction = m.group(2)
                continue
            
            # Edges & Nodes
            if not cls._parse_edge(line, graph):
                cls._parse_node(line, graph)
        return graph

    @classmethod
    def _preprocess(cls, line: str) -> str:
        line = re.sub(r'\s*--\|(.+?)\|\s*-->', r' -->|\1|', line)
        return line

    @classmethod
    def _parse_edge(cls, line: str, graph: GraphStructure) -> bool:
        # Inline label check
        m = re.match(r'^(.+?)\s+--\s+(.+?)\s+-->\s+(.+)$', line)
        if m:
            s, l, d = m.groups()
            graph.edges.append(Edge(cls._get_nid(s, graph), cls._get_nid(d, graph), l.strip(), "-->"))
            return True
            
        # Standard check
        for pat, style in cls.ARROW_PATTERNS:
            # With label
            m = re.match(rf'^(.+?)\s*{pat}\s*\|(.+?)\|\s*(.+)$', line)
            if m:
                s, l, d = m.groups()
                graph.edges.append(Edge(cls._get_nid(s, graph), cls._get_nid(d, graph), l.strip(), style))
                return True
            # Without label
            m = re.match(rf'^(.+?)\s*{pat}\s*(.+)$', line)
            if m:
                s, d = m.groups()
                # Chain check A --> B --> C
                if cls._contains_arrow(s): return cls._parse_chain(line, graph)
                graph.edges.append(Edge(cls._get_nid(s, graph), cls._get_nid(d, graph), "", style))
                return True
        return False

    @classmethod
    def _parse_chain(cls, line: str, graph: GraphStructure) -> bool:
        parts = re.split(r'\s*(?:-->|---|==>|-\.->)\s*', line)
        if len(parts) < 2: return False
        nodes = [cls._get_nid(p.strip(), graph) for p in parts]
        for i in range(len(nodes)-1):
            graph.edges.append(Edge(nodes[i], nodes[i+1], ""))
        return True

    @classmethod
    def _contains_arrow(cls, text: str) -> bool:
        return any(re.search(p, text) for p, _ in cls.ARROW_PATTERNS)

    @classmethod
    def _parse_node(cls, line: str, graph: GraphStructure):
        cls._get_nid(line, graph)

    @classmethod
    def _get_nid(cls, text: str, graph: GraphStructure) -> str:
        for pat, shape in cls.SHAPE_PATTERNS:
            m = re.match(rf'^([A-Za-z0-9_]+)\s*' + pat + r'$', text)
            if m:
                nid, lbl = m.groups()
                if nid not in graph.nodes: graph.nodes[nid] = Node(nid, lbl.strip(), shape)
                return nid
        safe_id = re.sub(r'[^A-Za-z0-9_]', '', text)
        if not safe_id: safe_id = "UNKNOWN"
        if safe_id not in graph.nodes: graph.nodes[safe_id] = Node(safe_id, text, "rect")
        return safe_id


# =============================================================================
# GraphSight Agent v9 (Union Logic)
# =============================================================================

class GraphSightAgent:
    
    def __init__(self, model: str = "gpt-5"):
        # Temperature=0 for deterministic steps, but drafts have built-in persona bias
        self.llm = ChatOpenAI(model=model, temperature=0)

    def run(self, image_path: str) -> str:
        logger.info(f"🚀 [GraphSight v9 Union] Processing: {image_path}")
        image_content = self._load_image(image_path)
        
        # 1. Draft
        drafts = self._generate_drafts(image_content)
        
        # 2. Normalize (Critical Step)
        normalized_drafts = self._normalize_ids(drafts)
        
        # 3. Union Merge (Adopt if ANY present)
        final_graph = self._merge_union(normalized_drafts)
        
        logger.info(f"✅ Final Graph: {len(final_graph.nodes)} nodes, {len(final_graph.edges)} edges")
        return final_graph.to_mermaid()

    def _generate_drafts(self, image_content) -> Dict[str, GraphStructure]:
        perspectives = {
            "Structuralist": "Focus on Semantic IDs (e.g. DeleteRequest) and logical flow.",
            "Optimist": "High Recall. Include EVERYTHING. Even faint lines.",
            "Pessimist": "High Precision. Only clear text/lines."
        }
        results = {}
        
        def _run(name, role):
            prompt = f"""You are the {name}. {role}
Output JSON: {{ "mermaid": "graph TD\\n..." }}"""
            try:
                resp = self.llm.invoke([SystemMessage(content=prompt), HumanMessage(content=image_content)])
                mermaid = self._parse_json(resp.content).get("mermaid", "")
                return name, MermaidParser.parse(mermaid)
            except Exception as e:
                logger.error(f"{name} failed: {e}")
                return name, GraphStructure()

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
            futures = [ex.submit(_run, n, p) for n, p in perspectives.items()]
            for f in concurrent.futures.as_completed(futures):
                n, g = f.result()
                if g.nodes: results[n] = g
        return results

    def _normalize_ids(self, drafts: Dict[str, GraphStructure]) -> Dict[str, GraphStructure]:
        if "Structuralist" not in drafts: return drafts
        
        anchor = drafts["Structuralist"]
        anchor_txt = "\n".join([f"- {n.id}: {n.label}" for n in anchor.nodes.values()])
        mappings = {}

        for name, graph in drafts.items():
            if name == "Structuralist": continue
            target_txt = "\n".join([f"- {n.id}: {n.label}" for n in graph.nodes.values()])
            
            prompt = f"""
Map IDs from List B to List A (Anchor) based on meaning.
If unique, keep original ID.
Output JSON: {{ "mapping": {{ "B_ID": "A_ID" }} }}

List A (Anchor):
{anchor_txt}

List B:
{target_txt}
"""
            try:
                resp = self.llm.invoke([SystemMessage(content=prompt)])
                mappings[name] = self._parse_json(resp.content).get("mapping", {})
            except: mappings[name] = {}

        # Apply
        normalized = {"Structuralist": anchor}
        for name, graph in drafts.items():
            if name == "Structuralist": continue
            mapping = mappings.get(name, {})
            new_nodes = {}
            id_map = {} # Local ID remap
            
            for nid, node in graph.nodes.items():
                new_id = mapping.get(nid, nid)
                id_map[nid] = new_id
                # Keep original label/shape for now
                new_nodes[new_id] = Node(new_id, node.label, node.shape)
            
            new_edges = []
            for e in graph.edges:
                s, d = id_map.get(e.src, e.src), id_map.get(e.dst, e.dst)
                new_edges.append(Edge(s, d, e.label, e.style))
                
            normalized[name] = GraphStructure(graph.direction, new_nodes, new_edges)
        
        return normalized

    def _merge_union(self, drafts: Dict[str, GraphStructure]) -> GraphStructure:
        """
        UNION STRATEGY:
        Iterate through drafts in priority order (Structuralist -> Optimist -> Pessimist).
        Add ANY node/edge that hasn't been added yet.
        """
        final = GraphStructure()
        
        # Priority order determines whose LABEL/SHAPE wins, but EXISTENCE is purely additive.
        priority = ["Structuralist", "Optimist", "Pessimist"]
        
        # 1. Merge Nodes
        for name in priority:
            if name not in drafts: continue
            graph = drafts[name]
            
            if not final.direction and graph.direction:
                final.direction = graph.direction
            
            for nid, node in graph.nodes.items():
                if nid not in final.nodes:
                    # New node found! Add it.
                    final.nodes[nid] = node
                else:
                    # Node exists. 
                    # Optionally: Update label if current is "better" (longer)?
                    # For now, stick to Priority Order (Structuralist label wins).
                    pass

        # 2. Merge Edges
        # Use a set to track unique edges (src, dst) to avoid duplication
        existing_edges = set()
        
        for name in priority:
            if name not in drafts: continue
            graph = drafts[name]
            
            for edge in graph.edges:
                # Union Logic: Add if (src, dst) not present
                pair = (edge.src, edge.dst)
                if pair not in existing_edges:
                    # Safeguard: Ensure nodes exist (Logic: Union implies nodes are already added above)
                    # But if an edge references a missing node (rare but possible), create a dummy node.
                    if edge.src not in final.nodes:
                        final.nodes[edge.src] = Node(edge.src, edge.src)
                    if edge.dst not in final.nodes:
                        final.nodes[edge.dst] = Node(edge.dst, edge.dst)
                        
                    final.edges.append(edge)
                    existing_edges.add(pair)

        return final

    def _load_image(self, path):
        with open(path, "rb") as f: b64 = base64.b64encode(f.read()).decode()
        return [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]

    def _parse_json(self, t):
        t = t.strip()
        if "```" in t: t = t.split("json")[-1].split("```")[0]
        return json.loads(t.strip())

if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit(1)
    print(GraphSightAgent().run(sys.argv[1]))

