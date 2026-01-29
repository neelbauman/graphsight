from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple

class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    
    def __add__(self, other):
        if not isinstance(other, TokenUsage):
            return self
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens
        )
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

class DiagramType(str, Enum):
    FLOWCHART = "flowchart"
    SEQUENCE = "sequenceDiagram"
    CLASS = "classDiagram"
    ER = "erDiagram"
    UNKNOWN = "unknown"

class OutputFormat(str, Enum):
    MERMAID = "mermaid"
    NATURAL_LANGUAGE = "natural_language"

class ConnectedNode(BaseModel):
    target_id: str = Field(..., description="Suggested ID for the next node.")
    description: str = Field(..., description="Brief visual description.")
    edge_label: Optional[str] = Field(None, description="Label on the arrow (e.g., Yes, No).")
    
    bbox: Optional[List[int]] = Field(None, description="[ymin, xmin, ymax, xmax] (0-1000).")
    grid_refs: Optional[List[str]] = Field(None, description="List of ALL overlapping grid labels.")

class Focus(BaseModel):
    description: str = Field(..., description="視覚的な説明")
    suggested_id: Optional[str] = Field(None, description="推測されるノードID")
    
    bbox: Optional[List[int]] = Field(None)
    grid_refs: Optional[List[str]] = Field(None)

    def centroid(self) -> Tuple[float, float]:
        if not self.bbox or len(self.bbox) != 4:
            return (0.0, 0.0)
        y_center = (self.bbox[0] + self.bbox[2]) / 2
        x_center = (self.bbox[1] + self.bbox[3]) / 2
        return (y_center, x_center)
    
    def is_same_location_hybrid(self, other: 'Focus', spatial_threshold: float = 100.0) -> bool:
        """
        [Hybrid Identity Check: OR Logic]
        Grid または BBox のどちらか一方が一致していれば「同一ノード」とみなす。
        """
        # 1. Grid Check
        grid_match = False
        if self.grid_refs and other.grid_refs:
            if not set(self.grid_refs).isdisjoint(set(other.grid_refs)):
                grid_match = True
        
        # 2. BBox Check (Secondary)
        bbox_match = False
        c1 = self.centroid()
        c2 = other.centroid()
        
        if c1 != (0.0, 0.0) and c2 != (0.0, 0.0):
            dist_sq = (c1[0] - c2[0])**2 + (c1[1] - c2[1])**2
            if dist_sq < spatial_threshold**2:
                bbox_match = True
        
        return grid_match or bbox_match

# ▼▼▼ 追加: InitialFocusList ▼▼▼
class InitialFocusList(BaseModel):
    start_nodes: List[Focus]

class StepInterpretation(BaseModel):
    visual_observation: str = Field(..., description="Step 1: Visual observation.")
    arrow_tracing: str = Field(..., description="Step 2: Trace lines.")
    outgoing_edges: List[ConnectedNode] = Field(..., description="Step 3: Identified connections.")
    is_done: bool = Field(False)

class DiagramResult(BaseModel):
    diagram_type: str
    output_format: OutputFormat
    content: str
    raw_content: str
    full_description: str
    usage: TokenUsage
    cost_usd: float
    model_name: str

