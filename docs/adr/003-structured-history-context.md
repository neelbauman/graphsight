# 3. Structured History Context Passing

* Status: Proposed
* Date: 2026-02-05

## Context
現在、`GraphInterpreter` (Engine) は、探索の履歴 (`context_history`) を「生成済みのMermaid文字列のリスト (`List[str]`)」として `Strategy` に渡している。

この文字列には、ループ検知やノード同定のために必要な空間情報（Grid座標やBBox）が `%% Grid: ...` というコメント形式で埋め込まれている。これには以下の問題がある：

1.  **出力汚染**: 最終的なMermaid出力にデバッグ用のコメントが混入し、可読性を損なう。また、これを除去する処理がAd-Hocになりがちである。
2.  **決定論的動作の阻害**: 空間情報の微細な揺らぎ（例: `['A1']` vs `['A1', 'A2']`）により、同一エッジが異なる文字列として扱われ、重複排除（Dedup）が機能しない。
3.  **責務の漏洩**: Engineが特定の出力フォーマット（Mermaidのコメント記法）を知りすぎており、Strategyの自律性を侵害している。

## Decision
EngineとStrategy間の履歴データの受け渡しを、**文字列ベースから構造化オブジェクトベースに変更する**。

1.  **データ構造**: `interpret_step` メソッドは、`List[str]` ではなく `List[StepInterpretation]` を受け取る。
2.  **メタデータ注入**: Engineは、`StepInterpretation` オブジェクトに「どのノードからの遷移か」を示す `source_id` を注入して履歴に保存する。
3.  **動的プロンプト生成**: 各 `Strategy` は、受け取ったオブジェクトリストから、LLMに見せるためのコンテキストテキスト（必要に応じてGrid情報などを含む）を動的に生成する責務を持つ。
4.  **クリーンな出力**: Engineが生成・蓄積する `extracted_data` (文字列リスト) には、空間情報などのメタデータを含めず、純粋なグラフ構造のみを保持する。

## Consequences
* **Positive**: 最終出力されるMermaidコードが常にクリーンになる。重複排除ロジックが文字列の一致ではなく、IDや構造に基づき堅牢になる。Strategyがプロンプトへの情報注入を完全に制御できる。
* **Negative**: `BaseStrategy` のインターフェース変更に伴い、すべてのStrategyクラス（Flowchart, Structured, etc.）の実装修正が必要になる。

