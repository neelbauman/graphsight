from ..base import BasePipeline
from ...core.engine import GraphInterpreter
from ...strategies.flowchart import FlowchartStrategy
from ...llm.base import BaseVLM # 仮: VLMの初期化が必要なら適宜調整

class CrawlingPipeline(BasePipeline):
    def run(self, image_path: str) -> str:
        # 旧ロジックの組み立て
        # Note: VLMやStrategyの初期化ロジックは既存のCLI実装から移植する
        vlm = ... # Initialize VLM wrapper needed for GraphInterpreter
        engine = GraphInterpreter(vlm)
        strategy = FlowchartStrategy()
        
        result = engine.process(image_path, strategy)
        return result.content
