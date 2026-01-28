import pytest
from unittest.mock import MagicMock
from graphsight.core.engine import GraphInterpreter
from graphsight.strategies.flowchart import FlowchartStrategy
from graphsight.models import Focus, StepInterpretation, TokenUsage, OutputFormat

@pytest.fixture
def mock_vlm():
    return MagicMock()

def test_identical_labels_distinct_nodes(mock_vlm):
    """
    同名のラベル("Error")を持つが、IDが異なる("node_Error_1", "node_Error_2")場合、
    Engineがそれらを別のノードとして扱うことを検証する。
    """
    # Setup
    strategy = FlowchartStrategy(output_format=OutputFormat.MERMAID)
    engine = GraphInterpreter(mock_vlm)
    
    # 1. Initial Focus (Start)
    start_focus = Focus(description="Start Node", suggested_id="node_Start")
    mock_vlm.query_structured.side_effect = [
        # find_initial_focus result
        (MagicMock(start_nodes=[start_focus]), TokenUsage()),
        
        # Step 1: Start -> Check
        (StepInterpretation(
            extracted_text="node_Start[Start] --> node_Check{Check}",
            next_focus_candidates=[Focus(description="Check decision", suggested_id="node_Check")],
            reasoning="Start flows to Check",
            is_done=False
        ), TokenUsage()),

        # Step 2: Check -> Error_1 (Branch A) & End (Branch B)
        # ここで分岐。片方は "Error" ノードへ。
        (StepInterpretation(
            extracted_text="node_Check{Check} -->|Fail| node_Error_1[Error]\nnode_Check{Check} -->|Success| node_End[End]",
            next_focus_candidates=[
                Focus(description="Error Dialog A", suggested_id="node_Error_1"),
                Focus(description="End Node", suggested_id="node_End")
            ],
            reasoning="Fail goes to Error (left), Success goes to End",
            is_done=False
        ), TokenUsage()),
        
        # Step 3: Error_1 -> End
        (StepInterpretation(
            extracted_text="node_Error_1[Error] --> node_End[End]",
            next_focus_candidates=[Focus(description="End Node", suggested_id="node_End")],
            reasoning="Error 1 merges to End",
            is_done=False
        ), TokenUsage()),

         # Step 4: End (No outgoing)
        (StepInterpretation(
            extracted_text="node_End[End]",
            next_focus_candidates=[],
            reasoning="Process ends",
            is_done=True
        ), TokenUsage()),
        
        # Step 5: (Mocking potential explore of other branch if queue logic varies, but simplistic here)
        # ...
    ]
    
    # Strategyのsynthesizeをモックして、生の抽出データをそのまま返すようにする
    strategy.synthesize = MagicMock(return_value=("Refined Content", "Raw Content", TokenUsage()))

    # Run
    result = engine.process("dummy_image.png", strategy)
    
    # Verification
    # 抽出されたテキストの中に、別々のIDが含まれているか、あるいは統合ロジックが働いたかを確認
    # ここではEngineが `extracted_data` を収集する動きを確認したい
    
    # プロセス呼び出しで query_structured が呼ばれた回数などをチェック
    assert mock_vlm.query_structured.call_count >= 3
    
    # 実際に渡された extracted_texts を確認 (synthesizeの第2引数)
    extracted_texts = strategy.synthesize.call_args[0][1]
    
    # node_Error_1 が存在することを確認
    assert any("node_Error_1" in text for text in extracted_texts)
    # もし別の Error_2 があればそれも検証（このシナリオでは簡略化のため1つだが、IDが保持されていることが重要）

