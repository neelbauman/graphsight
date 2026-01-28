# 2. Decouple Node Identity from Node Label

Date: 2026-01-28

## Status
Accepted

## Context
現在の `FlowchartStrategy` は、ノードのテキスト内容（Label）を正規化してノードID（Identity）として使用している（例: "Error" -> `node_Error`）。

しかし、フローチャートにおいては「ラベルは同じだが、論理的・空間的に異なるノード」が頻繁に登場する。
典型例：
- 複数の条件分岐の先にある、それぞれの「処理終了」ノード
- 異なるプロセスで発生する、同名の「エラー表示」処理

現状のロジックでは、これらが全て単一のノードID（`node_Error`）にマージされてしまうため、グラフの構造が不正になり（意図しない合流が発生する）、正確なトレースができない。

## Decision
ノードIDの生成ロジックを変更し、テキストの一意性に依存しない形にする。
VLM（Vision-Language Model）へのプロンプト指示を修正し、以下のルールでIDを生成させる。

1.  **Context-Aware Identity**:
    - 過去に訪問したノードと同じ場所（再訪問/ループ）であれば、既存のIDを再利用する。
    - テキストが同じでも、異なる場所にある新しいノードであれば、ユニークなサフィックスを付与する（例: `node_Error_1`, `node_Error_2`）。
2.  **Explicit Instruction**:
    - プロンプトにて「同名別ノード」の可能性を明示的に示唆する。

## Consequences
- **Positive**: 同名の異なるノードが正しく区別され、Mermaidおよび自然言語出力の構造的正確性が向上する。
- **Negative**: LLMへの指示が複雑化するため、プロンプト消費トークンが若干増加する可能性がある。また、LLMが過剰にIDを分割してしまう（同じノードなのに別IDにしてしまう）リスクがあるため、"Consistency Check" の指示も併せて強化する必要がある。

