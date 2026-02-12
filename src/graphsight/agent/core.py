import base64
import json
from typing import TypedDict, Annotated, List, Literal
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from loguru import logger

try:
    from .tools import ALL_TOOLS
except ImportError:
    from agent.tools import ALL_TOOLS


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    step_count: int


class GraphSightAgent:
    MAX_TOOL_STEPS = 30
    FORCE_THOUGHT_INTERVAL = 3  # 5→3 に短縮: より頻繁に思考させる
    SOFT_LIMIT_STEP = 20

    # フェーズ境界ステップ (Overview→NodeExtraction→EdgeTracing→Validation)
    PHASE_BOUNDARIES = (2, 8, 15)

    def __init__(self, model: str = "gpt-4o"):
        try:
            self.llm = ChatOpenAI(model=model, temperature=0, reasoning_effort="high")
        except Exception:
            self.llm = ChatOpenAI(model=model, temperature=0)

        self.llm_with_tools = self.llm.bind_tools(ALL_TOOLS)
        self.tools_by_name = {tool.name: tool for tool in ALL_TOOLS}

        # ===================================================================
        # 改善: 3フェーズを明確に分離したシステムプロンプト
        #   Phase 2a: ノードだけを全て列挙
        #   Phase 2b: 矢印/線の接続点を個別にcropして追跡
        #   Phase 2c: 不確実なエッジの検証
        # ===================================================================
        self.system_prompt = """You are an expert AI for generating Mermaid diagrams from images.
Your Goal: Reproduce the flowchart in the image as Mermaid code.

**WORKFLOW (follow these phases strictly):**

## Phase 1: Overview (1-2 tool calls)
- Use `get_image_info` to get dimensions.
- Look at the full image to understand the overall structure.
- Describe the global layout in text: how many nodes, rough arrangement, any branching.

## Phase 2a: Node Extraction (3-6 tool calls)
- Crop and list ALL nodes first.
- For each node, record: a temporary ID, exact label text, shape (rectangle, diamond, rounded, etc.), and approximate position (e.g. "top-left", "center-right").
- Do NOT trace connections yet. Focus only on reading every node label accurately.

## Phase 2b: Edge Tracing (3-8 tool calls)
- Now trace each arrow/line individually.
- IMPORTANT: Do NOT guess connections from node proximity alone.
- For each edge, crop the ARROW/CONNECTION POINT (the line between nodes, not the node itself) to verify where the arrow starts and where it ends.
- Pay special attention to:
  - Crossing lines: two arrows that visually cross but connect to different nodes.
  - Curved or bent arrows: trace the full path, not just endpoints.
  - Long-distance connections: crop the intermediate path to confirm the route.
  - Arrows with labels (Yes/No, condition text): read the label on the arrow.

## Phase 2c: Validation (1-2 tool calls, only if needed)
- Review any edges you are uncertain about.
- Crop the specific ambiguous connection points to confirm.
- If two nodes are far apart, crop the intermediate path to verify.
- Skip this phase if all edges are high-confidence.

## Phase 3: Output (NO tool calls)
- Once you have all node labels and connections, STOP using tools immediately.
- Output the complete Mermaid diagram in a ```mermaid code block.
- Do NOT do "one more check". Output what you have.

**CRITICAL RULES:**
- You MUST produce a ```mermaid code block as your final answer.
- Separate node discovery from edge discovery. Do NOT mix them.
- When tracing edges, crop the ARROW itself, not just the nodes it connects.
- Prefer outputting an approximate diagram over endless exploration.
- If a label is hard to read, use your best guess rather than re-cropping endlessly.
- Total tool budget: ~20 calls. Use them wisely.
"""
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", self.call_model)
        workflow.add_node("tools", self.call_tools)

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            self.should_continue,
            {"continue": "tools", "end": END}
        )
        workflow.add_edge("tools", "agent")

        return workflow.compile(checkpointer=MemorySaver())

    # --- フェーズ遷移の検出 ---

    def _is_phase_boundary(self, step_count: int) -> bool:
        """フェーズ境界ステップで強制思考を発動"""
        return step_count in self.PHASE_BOUNDARIES

    # --- フェーズに応じた観察プロンプトを生成 ---

    def _get_observation_prompt(self, step_count: int) -> SystemMessage:
        """ステップ数に応じて、フェーズ適切な観察プロンプトを返す"""

        if step_count <= 3:
            # Phase 1→2a 境界: ノード抽出の準備
            return SystemMessage(content="""
[OBSERVATION CHECKPOINT — Tools disabled for this response]
Respond with TEXT ONLY. Do NOT call any tools.

You are transitioning from Overview to Node Extraction.
Summarize:
1. Overall layout: [describe structure, direction, branching]
2. Estimated node count: [number]
3. Regions to crop for nodes: [list areas to examine]
""")
        elif step_count <= 10:
            # Phase 2a→2b 境界: ノード完了→エッジ追跡開始
            return SystemMessage(content="""
[OBSERVATION CHECKPOINT — Tools disabled for this response]
Respond with TEXT ONLY. Do NOT call any tools.

You are transitioning from Node Extraction to Edge Tracing.
Summarize:
1. Nodes confirmed: [list each with ID, label, shape]
2. Any nodes you might have missed? [check carefully]
3. Arrows/connections to trace next: [list visible arrows to investigate]
4. Any crossing or overlapping arrows noticed? [yes/no, details]
""")
        else:
            # Phase 2b→2c/3 境界: エッジ検証→出力準備
            return SystemMessage(content="""
[OBSERVATION CHECKPOINT — Tools disabled for this response]
Respond with TEXT ONLY. Do NOT call any tools.

You are preparing for final output.
Summarize:
1. Nodes confirmed: [list each with ID, label, shape]
2. Edges confirmed (HIGH confidence): [list source→target with any labels]
3. Edges uncertain (LOW confidence): [list — these NEED one more crop to verify]
4. Are there crossing/overlapping arrows? [yes/no, which ones]
5. Ready to output mermaid? [yes ONLY if no uncertain edges remain, otherwise no]
""")

    # --- Node: Call Model (改善版: フェーズ遷移対応の強制思考) ---

    def call_model(self, state: AgentState):
        messages = state["messages"]
        step_count = state.get("step_count", 0)
        last_message = messages[-1]

        if not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=self.system_prompt)] + messages

        messages_for_llm = self._smart_prune_images(messages, max_recent=15)

        # --- ハードリミット ---
        if step_count >= self.MAX_TOOL_STEPS:
            logger.warning(f"🛑 HARD LIMIT ({step_count} steps). Forcing final output.")
            force_msg = SystemMessage(content=f"""
[SYSTEM: HARD LIMIT — {step_count} tool calls used. Tools DISABLED.]
Output your ```mermaid code block NOW with everything you have.
Do NOT call any tools. Just output the diagram.
""")
            messages_for_llm.append(force_msg)
            response = self.llm.invoke(messages_for_llm)
            return {"messages": [response], "step_count": step_count}

        # --- 連続ツール使用カウント ---
        consecutive_tool_turns = 0
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                if msg.content and str(msg.content).strip() and not msg.tool_calls:
                    break
                consecutive_tool_turns += 1
            elif isinstance(msg, ToolMessage):
                continue
            else:
                break

        # --- 強制思考の判定 (改善: インターバル短縮 + フェーズ境界) ---
        should_force_thought = (
            isinstance(last_message, ToolMessage)
            and (
                consecutive_tool_turns >= self.FORCE_THOUGHT_INTERVAL
                or self._is_phase_boundary(step_count)
            )
        )

        # --- ソフトリミット ---
        if step_count >= self.SOFT_LIMIT_STEP:
            messages_for_llm.append(SystemMessage(content=f"""
[NOTICE: {step_count}/{self.MAX_TOOL_STEPS} tool calls used. Wrap up soon.]
If you have enough information, output the ```mermaid diagram now.
"""))

        # =================================================================
        # 強制思考: 2段階呼び出し (改善: フェーズ適応型プロンプト)
        # =================================================================
        if should_force_thought:
            logger.info(
                f"🔒 Forcing observation (consecutive: {consecutive_tool_turns}, "
                f"step: {step_count}, phase_boundary: {self._is_phase_boundary(step_count)})"
            )

            # ---- Phase 1: ツールなしで観察を強制出力 ----
            observation_prompt = self._get_observation_prompt(step_count)
            observation_response = self.llm.invoke(
                messages_for_llm + [observation_prompt]
            )
            observation_text = observation_response.content
            logger.info(f"📝 Observation: {observation_text[:300]}...")

            # ---- Phase 2: 観察を踏まえて次のアクションを判断 ----
            phase2_messages = messages_for_llm + [
                observation_prompt,
                AIMessage(content=observation_text),
                HumanMessage(content=(
                    "Based on your observation above, decide your next action:\n"
                    "- If you listed uncertain edges → crop the ARROW CONNECTION POINT (not the node) to verify.\n"
                    "- If all nodes and edges are confirmed → output the ```mermaid diagram now.\n"
                    "- If nodes are incomplete → crop the missing region.\n"
                    "Make at most ONE targeted tool call, or output the final diagram."
                ))
            ]
            response = self.llm_with_tools.invoke(phase2_messages)

            return {
                "messages": [observation_response, response],
                "step_count": step_count
            }
        else:
            # --- 通常モード ---
            response = self.llm_with_tools.invoke(messages_for_llm)
            return {"messages": [response], "step_count": step_count}

    # --- Node: Call Tools (改善: スキップ理由を明示) ---

    def call_tools(self, state: AgentState):
        last_message = state["messages"][-1]
        step_count = state.get("step_count", 0)
        tool_results = []

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            for i, tool_call in enumerate(last_message.tool_calls):
                tool_call_id = tool_call['id']
                tool_name = tool_call['name']

                if i == 0:
                    tool_args = tool_call['args']
                    logger.info(f"🛠️  [{step_count+1}/{self.MAX_TOOL_STEPS}] {tool_name} args={tool_args}")

                    content_to_return = ""
                    if tool_name in self.tools_by_name:
                        try:
                            output = self.tools_by_name[tool_name].invoke(tool_args)
                            if isinstance(output, str) and (
                                output.lower().endswith(".jpg") or output.lower().endswith(".png")
                            ):
                                content_to_return = self._create_image_content(
                                    output,
                                    f"Result from {tool_name}. [Step {step_count+1}/{self.MAX_TOOL_STEPS}]"
                                )
                            else:
                                content_to_return = str(output)
                        except Exception as e:
                            content_to_return = f"Error: {e}"
                    else:
                        content_to_return = f"Error: Tool {tool_name} not found."

                    tool_results.append(ToolMessage(
                        tool_call_id=tool_call_id, content=content_to_return, name=tool_name
                    ))
                    step_count += 1
                else:
                    # 改善: スキップ理由を明示し、リトライを促す
                    logger.info(f"   (Skipping parallel: {tool_name})")
                    tool_results.append(ToolMessage(
                        tool_call_id=tool_call_id,
                        content=(
                            f"Skipped: parallel tool calls are not supported. "
                            f"You requested '{tool_name}' but only the first tool call was executed. "
                            f"Please retry this call separately if you still need it."
                        ),
                        name=tool_name
                    ))

        return {"messages": tool_results, "step_count": step_count}

    def should_continue(self, state: AgentState):
        last_message = state["messages"][-1]

        # AIMessage でかつ tool_calls がある場合のみ続行
        if isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"

        return "end"

    # --- Context Management ---

    def _smart_prune_images(self, messages: List[BaseMessage], max_recent: int = 15) -> List[BaseMessage]:
        image_indices = []
        for i, msg in enumerate(messages):
            if self._has_image(msg):
                image_indices.append(i)

        if len(image_indices) <= max_recent + 1:
            return messages

        keep_indices = {image_indices[0]} | set(image_indices[-max_recent:])
        prune_indices = set(image_indices) - keep_indices

        new_messages = []
        for i, msg in enumerate(messages):
            if i in prune_indices:
                new_messages.append(self._remove_image_content(msg))
            else:
                new_messages.append(msg)
        return new_messages

    def _has_image(self, msg: BaseMessage) -> bool:
        if isinstance(msg.content, list):
            for item in msg.content:
                if isinstance(item, dict) and (item.get("type") == "image_url" or "image" in item):
                    return True
        return False

    def _remove_image_content(self, msg: BaseMessage) -> BaseMessage:
        if not isinstance(msg.content, list):
            return msg
        new_content = []
        for item in msg.content:
            if isinstance(item, dict) and item.get("type") == "text":
                new_content.append(item)
        new_content.append({"type": "text", "text": "\n[Old crop image removed]\n"})
        msg_copy = msg.model_copy()
        msg_copy.content = new_content
        return msg_copy

    def _create_image_content(self, image_path: str, text: str):
        try:
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            return [
                {"type": "text", "text": f"{text}\n(File: {image_path})"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]
        except Exception as e:
            return f"Error loading image: {e}"

    def run(self, image_path: str) -> str:
        logger.info(f"Agent starting run for: {image_path}")

        initial_content = self._create_image_content(
            image_path,
            f"Analyze the flowchart. File path: '{image_path}'.\nThis is the full overview image."
        )
        initial_message = HumanMessage(content=initial_content)

        inputs = {"messages": [initial_message], "step_count": 0}
        config = {
            "configurable": {"thread_id": "session_v6"},
            "recursion_limit": 80
        }

        final_response = ""

        try:
            for event in self.graph.stream(inputs, config=config, stream_mode="values"):
                if "messages" in event:
                    msg = event["messages"][-1]
                    if isinstance(msg, AIMessage):
                        if msg.tool_calls:
                            target = msg.tool_calls[0]
                            logger.info(f"🤖 Plan: {target['name']} args={target['args']}")
                        else:
                            logger.debug(f"🤖 Output: {msg.content[:200]}...")
                            final_response = msg.content
        except Exception as e:
            logger.error(f"Execution Error: {e}")
            raise e

        if "```mermaid" in final_response:
            return final_response.split("```mermaid")[1].split("```")[0].strip()
        if "```" in final_response:
            parts = final_response.split("```")
            if len(parts) >= 2:
                return parts[1].strip()

        logger.warning("⚠️ No mermaid block found in final response.")
        return final_response

