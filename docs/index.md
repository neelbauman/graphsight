# Welcome to GraphSight

**GraphSight** は、複雑な図版（フローチャート等）を AI (VLM) が視覚的にトレースし、**正確な Mermaid コード** や **自然言語によるロジック解説** に変換する Python ライブラリ & CLI ツールです。

単なる画像認識やOCRとは異なり、AIが図の中を「指差し確認」しながら論理構造（ノードのつながり）を順次追跡するため、複雑な条件分岐や長いフローでも高精度な解釈が可能です。

## ✨ 主な機能

* **Logic Tracing Engine** 🧠
    * Start から End まで手順を追って図を解釈する独自エンジン。
    * AIの「思考プロセス（Reasoning）」をログとして記録し、最終出力の精度を高めます。
* **Dual Output Modes** 🔄
    * 📊 **Mermaid Mode:** 作図ツールで編集可能なコードを出力。
    * 📝 **Natural Language Mode:** 非エンジニアにも分かりやすい、日本語での業務フロー解説を出力。
* **Cost & Token Transparency** 💰
    * 実行にかかったトークン数と概算コスト（USD）を即座に表示。
* **Multi-Model Support** 🤖
    * `gpt-4o`, `gpt-4o-mini`, `o1-preview` (想定) など、用途と予算に合わせてモデルを切り替え可能。

## デモ

=== "Mermaid出力例"

    ```mermaid
    graph TD
        node_Start[Start] --> node_Login[Login]
        node_Login -->|Success| node_Home[Home Screen]
        node_Login -->|Fail| node_Error[Error Dialog]
    ```

=== "自然言語出力例"

    > 処理は **Start** から開始されます。まず **Login** 画面へ遷移し、ユーザー認証を行います。認証が **Success** の場合は **Home Screen** へ移動しますが、**Fail** の場合は **Error Dialog** を表示して終了します。
