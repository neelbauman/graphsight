"""
graphsight.models
~~~~~~~~~~~~~~~~~
Pydantic models defining the data structure for flowchart interpretation.
Includes Spot/Beautyspot registration for structured output.
"""

from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
from .llm.config import spot

@spot.register(
    code=1,
    encoder=lambda x: x.model_dump_json().encode(),
    decoder=lambda x: TokenUsage.model_validate_json(x),
)
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


@spot.register(
    code=2,
    encoder=lambda x: x.model_dump_json().encode(),
    decoder=lambda x: ClassificationResult.model_validate_json(x),
)
class ClassificationResult(BaseModel):
    diagram_type: DiagramType
    reasoning: str = Field(..., description="The reason for this classification based on visual features.")


class OutputFormat(str, Enum):
    MERMAID = "mermaid"
    NATURAL_LANGUAGE = "natural_language"


@spot.register(
    code=3,
    encoder=lambda x: x.model_dump_json().encode(),
    decoder=lambda x: ConnectedNode.model_validate_json(x),
)
class ConnectedNode(BaseModel):
    target_id: str = Field(..., description="Suggested ID for the next node.")
    description: str = Field(..., description="Brief visual description.")
    edge_label: Optional[str] = Field(None, description="Label on the arrow (e.g., Yes, No).")
    
    bbox: Optional[List[int]] = Field(None, description="[ymin, xmin, ymax, xmax] (0-1000).")
    grid_refs: Optional[List[str]] = Field(None, description="List of ALL overlapping grid labels.")


@spot.register(
    code=4,
    encoder=lambda x: x.model_dump_json().encode(),
    decoder=lambda x: Focus.model_validate_json(x),
)
class Focus(BaseModel):
    description: str = Field(..., description="Visual description of the node.")
    suggested_id: Optional[str] = Field(None, description="Suggested unique identifier.")
    
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
        Grid or BBox match implies identity.
        """
        # 1. Grid Check
        grid_match = False
        if self.grid_refs and other.grid_refs:
            if not set(self.grid_refs).isdisjoint(set(other.grid_refs)):
                grid_match = True
        
        # 2. BBox Check
        bbox_match = False
        c1 = self.centroid()
        c2 = other.centroid()
        
        if c1 != (0.0, 0.0) and c2 != (0.0, 0.0):
            dist_sq = (c1[0] - c2[0])**2 + (c1[1] - c2[1])**2
            if dist_sq < spatial_threshold**2:
                bbox_match = True
        
        return grid_match or bbox_match


@spot.register(
    code=5,
    encoder=lambda x: x.model_dump_json().encode(),
    decoder=lambda x: InitialFocusList.model_validate_json(x),
)
class InitialFocusList(BaseModel):
    start_nodes: List[Focus]


@spot.register(
    code=8,
    encoder=lambda x: x.model_dump_json().encode(),
    decoder=lambda x: IncomingConnection.model_validate_json(x),
)
class IncomingConnection(BaseModel):
    """
    Visual observation of incoming arrows.
    Used for bidirectional verification.
    """
    direction: str = Field(..., description="Arrow direction relative to this node (Top, Bottom, Left, Right).")
    style: Optional[str] = Field(None, description="Line style (Solid, Dotted, etc).")
    bbox: Optional[List[int]] = Field(None, description="Approx location of the arrow head.")


@spot.register(
    code=6,
    encoder=lambda x: x.model_dump_json().encode(),
    decoder=lambda x: StepInterpretation.model_validate_json(x),
)
class StepInterpretation(BaseModel):
    # Step 1 & 2: Thinking Process (CoT)
    visual_observation: Optional[str] = Field(None, description="Step 1: Visual observation.")
    arrow_tracing: Optional[str] = Field(None, description="Step 2: Trace lines.")
    
    # Step 3: Connections
    outgoing_edges: List[ConnectedNode] = Field(..., description="Step 3: Identified outgoing connections.")
    
    # Validation Fields
    incoming_edges: List[IncomingConnection] = Field(default_factory=list, description="Step 0: Arrows pointing INTO this node.")
    
    # --- Override Reason for Validation Loop ---
    # 論理と視覚が一致しない正当な理由 (例: 矢印が消えかかっている、Phantom edgeなど)
    visual_override_reason: Optional[str] = Field(None, description="Explanation if visual observation contradicts logical geometry.")

    # 最終監査で「確定」された接続情報 (Re-wiring用)
    audit_confirmed_incoming: Optional[List[str]] = Field(None, description="List of source_ids CONFIRMED by global audit.")
    audit_confirmed_outgoing: Optional[List[str]] = Field(None, description="List of target_ids CONFIRMED by global audit.")
    audit_notes: Optional[str] = Field(None, description="Notes from the audit (e.g. 'Removed phantom edge from A').")

    is_done: bool = Field(False)

    # --- Context Meta (Engine injected) ---
    source_id: Optional[str] = Field(None, description="The ID of the node analyzed.")
    source_grid_refs: Optional[List[str]] = Field(None)
    source_bbox: Optional[List[int]] = Field(None)

    # --- Validation Meta ---
    validation_warnings: List[str] = Field(default_factory=list, description="Inconsistencies detected during validation.")


@spot.register(
    code=7,
    encoder=lambda x: x.model_dump_json(),
    decoder=lambda x: DiagramResult.model_validate_json(x),
)
class DiagramResult(BaseModel):
    diagram_type: str
    output_format: OutputFormat
    content: str
    raw_content: str
    full_description: str
    usage: TokenUsage
    cost_usd: float
    model_name: str

@spot.register(
    code=99,
    encoder=lambda x: x.model_dump_json(),
    decoder=lambda x: ConnectionVerificationResult.model_validate_json(x),
)
class ConnectionVerificationResult(BaseModel):
    is_connected: bool = Field(..., description="True only if a visible line connects the two nodes.")
    reason: str = Field(..., description="Why is it connected or not? (e.g. 'Line fades out', 'Clear arrow seen')")

