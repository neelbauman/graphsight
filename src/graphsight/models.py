from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional

# ▼ コスト集計用
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

    # 単価を受け取って計算する（モデル設定に依存するためここには単価を持たない）
    def calculate_cost(self, input_price: float, output_price: float) -> float:
        input_cost = (self.input_tokens / 1_000_000) * input_price
        output_cost = (self.output_tokens / 1_000_000) * output_price
        return input_cost + output_cost

class DiagramType(str, Enum):
    FLOWCHART = "flowchart"
    SEQUENCE = "sequenceDiagram"
    CLASS = "classDiagram"
    ER = "erDiagram"
    UNKNOWN = "unknown"

class OutputFormat(str, Enum):
    MERMAID = "mermaid"
    NATURAL_LANGUAGE = "natural_language"

class Focus(BaseModel):
    description: str = Field(..., description="視覚的な説明")
    bbox: Optional[List[int]] = Field(None, description="[x1, y1, x2, y2]")
    # ▼ 追加: IDによる管理
    suggested_id: Optional[str] = Field(None, description="推測されるノードID (例: node_General_Inquiries)")

class StepInterpretation(BaseModel):
    extracted_text: str = Field(..., description="The interpreted content.")
    next_focus_candidates: List[Focus] = Field(..., description="List of ALL immediate next nodes.")
    is_done: bool = Field(False, description="Is path finished")
    reasoning: str = Field(..., description="Thinking process")

class DiagramResult(BaseModel):
    diagram_type: str
    output_format: OutputFormat
    
    content: str      # AIによって洗練された最終結果 (Refined)
    raw_content: str  # 機械的に結合された生の結果 (Raw)
    
    full_description: str
    usage: TokenUsage
    cost_usd: float
    model_name: str

