# src/graphsight/pipelines/stable/draft_refine/agent.py

from typing import List
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI  # 追加: LangChain用のクライアント

from graphsight.llm.openai_client import OpenAIVLM
from .models import GraphStructure, UncertainPoint, CheckResult, RefineVerdict
from .tools import ALL_TOOLS

class InspectorAgent:
    """
    1つの不安点(UncertainPoint)に対し、ツールを駆使して自律的に検証を行うエージェント。
    """
    def __init__(self, model_name: str = "gpt-4o"):
        # 1. エージェント駆動用: LangChainのChatOpenAIを使用
        # (bind_toolsなどのメソッドを持っているのはこちら)
        self.agent_model = ChatOpenAI(model=model_name, temperature=0)
        
        # 2. 構造化出力用: 既存のOpenAIVLMラッパーを使用 (プロジェクト内の統一感のため)
        self.vlm = OpenAIVLM(model=model_name)
        
        # ツール定義
        self.tools = ALL_TOOLS

        # LangGraph で ReAct エージェントを作成
        self.agent_executor = create_react_agent(self.agent_model, self.tools)

    def verify_point(self, image_path: str, point: UncertainPoint, graph: GraphStructure) -> str:
        """
        エージェントを実行し、検証結果を文字列（CheckResultの要約）で返す。
        """
        # コンテキストの準備
        context_nodes = ", ".join([f"{n.id}[{n.label}]" for n in graph.nodes.values()])
        
        # システムプロンプト
        system_prompt = f"""You are a "Graph Inspector" agent.
Your mission is to verify a specific uncertain point in a flowchart image.

**Target Image Path:** `{image_path}`
(Always pass this path exactly to tools.)

**Current Draft Context:**
{context_nodes}

**The Uncertainty:**
ID: {point.id}
Question: "{point.description}"
Location Hint: {point.location}
Crop Coordinates: x={point.crop_x}, y={point.crop_y}, w={point.crop_w}, h={point.crop_h}

**Your Workflow:**
1. **Investigate**: Use `crop_region` to see the target area.
   - If the text is hard to read, use `crop_region` again with `preprocess="binarize"` or `preprocess="edge_enhancement"`.
2. **Analyze**: Compare what you see with the draft context.
3. **Conclude**: Decide if the draft is correct, incorrect, or unclear.

**Rules:**
- Do NOT guess. If you can't see it, try a filter. If still unclear, admit it.
- Trust your eyes over the draft.
"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Start investigation. Use tools to verify the point.")
        ]
        
        # エージェント実行
        result_state = self.agent_executor.invoke(
            {"messages": messages},
            config={"recursion_limit": 10}
        )
        
        # 実行後の全会話履歴を取得
        final_history = result_state["messages"]

        # 履歴を元に、最終的な判定を Structured Output で抽出
        final_verdict = self._summarize_verdict(final_history, point)
        
        return self._format_result(final_verdict)

    def _summarize_verdict(self, messages: List, point: UncertainPoint) -> CheckResult:
        """会話履歴から最終的な CheckResult を生成する"""
        
        prompt = f"""Review the investigation history above for point {point.id}.
Based on the tool outputs and the agent's findings, verify the draft.

Uncertainty Question: "{point.description}"

Output the final audit result (CheckResult).
"""
        # メッセージ履歴をテキスト化
        history_text = "\n".join([f"{m.type}: {m.content}" for m in messages])
        full_prompt = f"{history_text}\n\n{prompt}"

        # OpenAIVLMを使って型安全に抽出
        result, _ = self.vlm.query_structured(
            prompt=full_prompt,
            image_path=None, 
            response_model=CheckResult
        )
        return result

    def _format_result(self, result: CheckResult) -> str:
        """CheckResultをパイプラインが期待する文字列形式に変換"""
        if result.verdict == RefineVerdict.INCORRECT and result.correction_value:
            return f"{result.observation} → Correction: {result.correction_value}"
        elif result.verdict == RefineVerdict.UNCLEAR:
            return f"{result.observation} (Unclear, keeping draft)"
        else:
            return f"{result.observation} (Verified Correct)"

