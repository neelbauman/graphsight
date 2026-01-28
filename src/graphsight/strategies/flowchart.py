from typing import List, Tuple
from pydantic import BaseModel
from .base import BaseStrategy, OutputFormat
from ..llm.base import BaseVLM
from ..models import Focus, StepInterpretation, TokenUsage

class InitialFocusList(BaseModel):
    start_nodes: List[Focus]

class FlowchartStrategy(BaseStrategy):
    @property
    def mermaid_type(self) -> str:
        return "flowchart"

    def find_initial_focus(self, vlm: BaseVLM, image_path: str) -> Tuple[List[Focus], TokenUsage]:
        prompt = (
            "Analyze this flowchart image. "
            "Identify the 'Start' nodes or top-most nodes that initiate the process. "
            "Return a list of these nodes with a visual description and a suggested ID (e.g., node_Start)."
        )
        result, usage = vlm.query_structured(prompt, image_path, InitialFocusList)
        return result.start_nodes, usage

    def interpret_step(self, vlm: BaseVLM, image_path: str, current_focus: Focus, context_history: List[str]) -> Tuple[StepInterpretation, TokenUsage]:
        # コンテキスト履歴（直近15件）を作成し、IDの一貫性を保つヒントにする
        history_text = "\n".join(context_history[-15:])
        
        if self.output_format == OutputFormat.MERMAID:
            task_instruction = """
            # TASK: Output Mermaid Code
            
            ## Step 1: Analyze Current Node
            - Identify text: e.g. "Check Stock"
            - **Identify SHAPE**: Is it a Diamond (Decision)? Rectangle (Process)? Circle (Start/End)?
            - **DETERMINE ID (Crucial)**:
              - **Rule A (Re-visit)**: If this is the EXACT SAME node (spatially) accessed before (e.g., a loop back), reuse the existing ID found in "Context".
              - **Rule B (Distinct)**: If this node has the same text as a previous node but is a DIFFERENT node in the diagram (different position/branch), you MUST append a unique suffix (e.g., `node_Check_Stock_2`).
              - **Default**: `node_SanitizedText` (remove spaces/symbols).

            ## Step 2: Identify ALL Outgoing Connections (CRITICAL)
            - If Shape is **Diamond**: You MUST find at least 2 outgoing arrows (Yes/No, True/False).
            - Look carefully for lines exiting from bottom, right, left, or top.
            - Trace the line to the IMMEDIATE next node.

            ## Step 3: Construct `extracted_text`
            - Format: `CURRENT_ID[Text] -->|Label| NEXT_ID[Text]`
            - Output ONE LINE per outgoing arrow.
            - Example:
              node_Check[Check] -->|Yes| node_Ship[Ship Item]
              node_Check[Check] -->|No| node_Error_1[Show Error]
            - If it is an End node with no outgoing arrows, output just the node definition: `node_End[End]`.

            ## Step 4: List `next_focus_candidates`
            - Create a Focus item for EACH destination node found in Step 3.
            - **Important**: You MUST fill `suggested_id` with the `NEXT_ID` you used above. This is used for loop detection.
            """
        else:
            # 自然言語モード
            task_instruction = f"""
            # TASK: Output Natural Language (Japanese)
            
            1. Look at the focus area: "{current_focus.description}".
            2. Read the node text and Identify the shape (Decision/Process).
            3. **construct `extracted_text`**:
               - Describe the logic flow in Japanese.
               - IMPORTANT: Incorporate "how we got here" (e.g., "Yesの場合は...").
               - If it is a Decision node, mention that a branching occurs.
            4. `next_focus_candidates`: Describe visual location of NEXT nodes.
            """

        prompt = f"""
        You are an expert at interpreting Flowcharts.
        
        # Current Focus Area
        Target: "{current_focus.description}"
        (Look around this area in the image)

        # Instructions
        {task_instruction}

        # Context (Previously extracted code)
        {history_text}
        
        # Constraints
        - STRICTLY follow the `node_Text` naming convention, but add suffixes `_2`, `_3` if nodes are distinct.
        - Exhaustively list ALL branches (Yes, No, Else). Do NOT miss any path.
        - If a branch goes to a node that seems to exist in Context, assume it connects to that existing node ONLY IF it makes sense spatially (loop). Otherwise, treat it as a new instance.
        """
        return vlm.query_structured(prompt, image_path, StepInterpretation)

    def synthesize(self, vlm: BaseVLM, extracted_texts: List[str], step_history: List[StepInterpretation]) -> Tuple[str, str, TokenUsage]:
        # 1. Mechanical Synthesis (Raw) - extracted_textsを使用
        seen = set()
        unique_lines = []
        for line in extracted_texts:
            # 複数行のレスポンスを分解
            sub_lines = line.split('\n')
            for sub_line in sub_lines:
                clean_line = sub_line.strip()
                if clean_line and clean_line not in seen:
                    unique_lines.append(clean_line)
                    seen.add(clean_line)

        # Raw Contentの作成
        raw_content = ""
        if self.output_format == OutputFormat.MERMAID:
            raw_content = "graph TD\n    " + "\n    ".join(unique_lines)
        else:
            raw_content = "\n".join([f"- {line}" for line in unique_lines])

        if not unique_lines:
            return "", "", TokenUsage()

        # 2. Build Investigation Log (AI Reasoning History)
        # 最終的なRefinementのために、AIが辿った思考プロセスをログ化する
        investigation_log = ""
        for i, step in enumerate(step_history):
            investigation_log += f"Step {i+1}:\n"
            investigation_log += f"  - Reasoning: {step.reasoning}\n"
            investigation_log += f"  - Extracted: {step.extracted_text}\n"
            if step.is_done:
                investigation_log += "  - Status: Reached End/Termination\n"
            investigation_log += "\n"

        # 3. AI Synthesis (Refined)
        if self.output_format == OutputFormat.MERMAID:
            prompt = f"""
            Refine the fragmented diagram information into a single, valid, and clean Mermaid flowchart.
            
            # Investigation Log (Context)
            You have walked through the diagram step-by-step. Here is the log of your findings:
            {investigation_log}
            
            # Instructions
            1. Use the "Reasoning" in the log to understand the true logic and flow.
            2. **Merge duplicate nodes**: Ensure nodes with same text (e.g. "End", "Start") share the same ID.
            3. Fix any syntax errors in the Mermaid code.
            4. Ensure standard top-down (graph TD) orientation.
            5. Output ONLY the Mermaid code block.
            """
        else:
            prompt = f"""
            Based on the detailed investigation log below, summarize the entire business flow into a coherent, natural Japanese explanation.
            
            # Investigation Log (Context)
            You have walked through the diagram step-by-step. Here is the log of your findings:
            {investigation_log}
            
            # Instructions
            1. Read the "Reasoning" to capture the nuance of conditions (Yes/No branches).
            2. Reconstruct the story of the process flow.
            3. Use natural conjunctions (e.g., "その後", "もし〜なら", "この時点で終了します").
            4. Output as a clean, easy-to-read text (Markdown allowed).
            """

        # 画像なし(None)でテキストのみの処理としてリクエスト
        refined_content, usage = vlm.query_text(prompt, image_path=None)
        
        return refined_content, raw_content, usage

