from typing import List, Literal, Optional
from pydantic import BaseModel, Field

class PlanStep(BaseModel):
    """計画の個々のステップ"""
    id: int = Field(..., description="Step ID")
    description: str = Field(..., description="実行すべき具体的なアクション（例: '左上のノード群をクロップして確認する'）")
    status: Literal['pending', 'in_progress', 'completed', 'failed'] = "pending"
    reasoning: Optional[str] = Field(None, description="なぜこのステップが必要かの理由")

class ExecutionPlan(BaseModel):
    """エージェントの行動計画"""
    goal: str = Field(..., description="この計画の最終目標")
    steps: List[PlanStep] = Field(..., description="実行すべきステップのリスト")
    
class EvaluationResult(BaseModel):
    """Evaluatorの判定結果"""
    status: Literal['continue', 'replan', 'finished'] = Field(..., description="次のアクション")
    feedback: str = Field(..., description="PlannerまたはExecutorへのフィードバック")
    mermaid_code: Optional[str] = Field(None, description="完成した場合のMermaidコード")

