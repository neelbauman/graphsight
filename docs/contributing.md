# Contributing

GraphSight は新しい図版タイプへの対応を歓迎しています！

## 新しい Strategy の追加方法

新しい図版（例: シーケンス図）に対応するには、`src/graphsight/strategies/` 下に新しいクラスを作成します。

1.  `BaseStrategy` を継承したクラス（例: `SequenceStrategy`）を作成します。
2.  以下のメソッドを実装します。
    * `find_initial_focus`: 最初の着目点（Actorなど）を探す。
    * `interpret_step`: 1ステップごとの解釈ロジック（メッセージの送受信など）。
    * `synthesize`: 結果の統合ロジック。
3.  `api.py` の分岐ロジックに新しい Strategy を登録します。

## 開発環境のセットアップ

```bash
# 依存関係のインストール
uv sync

# テストの実行 (実装予定)
uv run pytest

```
