# graphsight

Author: neelbauman <exp.noymann@gmail.com>
Target Python: 3.14

---

# 👁️ GraphSight

**GraphSight** は、複雑な図版（フローチャート等）を AI (VLM) が視覚的にトレースし、**正確な Mermaid コード** や **自然言語によるロジック解説** に変換する Python ライブラリ & CLI ツールです。

単なる OCR や一括画像変換と異なり、AI が図の中を「指差し確認」しながら論理構造（ノードのつながり）を順次追跡するため、複雑な図版でも高精度な解釈が可能です。

## ✨ 特徴

* **Logic Tracing Engine:** 独自のグラフ探索エンジンにより、Start から End まで手順を追って図を解釈します。
* **Dual Output Modes:**
* 📊 **Mermaid Mode:** 作図ツールで編集可能なコードを出力。
* 📝 **Natural Language Mode:** 非エンジニアにも分かりやすい、日本語でのロジック・業務フロー解説を出力。


* **Smart Detection:** 入力された図版のタイプ（フローチャート等）を自動判定します。
* **Extensible Architecture:** 将来的にシーケンス図や ER 図など、新しい図版タイプを追加しやすい設計（Strategy パターン）を採用。
* **Built with uv:** 高速なパッケージ管理ツール `uv` で構築されています。

## 📦 インストール

このプロジェクトは `uv` を使用して管理されています。

```bash
# リポジトリのクローン
git clone https://github.com/your-username/graphsight.git
cd graphsight

# 依存関係のインストール
uv sync

```

## ⚙️ セットアップ

OpenAI の API キーが必要です（現在は GPT-4o を推奨）。
プロジェクトルートに `.env` ファイルを作成してください。

```ini
# .env
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

```

## 🚀 使い方 (CLI)

コマンドラインから画像を解析し、結果を出力できます。

### 1. Mermaid コードとして出力（デフォルト）

図の構造を解析し、Mermaid 記法で出力します。

```bash
uv run graphsight ./path/to/image.png

```

### 2. 自然言語（日本語）でロジックを解説

図の流れを読み解き、業務フローや条件分岐を日本語で説明させます。

```bash
uv run graphsight ./path/to/image.png --format natural_language

```

### 3. ファイルへの保存

`--output` オプションで結果をファイルに保存できます。

```bash
uv run graphsight ./path/to/image.png --output result.mmd

```

## 💻 使い方 (Python Library)

Python スクリプトからライブラリとして利用することも可能です。

```python
from graphsight import GraphSight

# クライアントの初期化
sight = GraphSight()

# 画像の解析 (デフォルトは Mermaid)
image_path = "./samples/login_flow.png"
result = sight.interpret(image_path)

print(f"Type: {result.diagram_type}")
print(result.content)

# 自然言語モードでの解析
result_nl = sight.interpret(image_path, format="natural_language")
print(result_nl.content)

```

## 🏗️ アーキテクチャ

GraphSight は **"Engine + Strategy"** パターンを採用しています。

1. **Detector:** 画像を見て、それが「フローチャート」なのか「シーケンス図」なのかを分類します。
2. **Strategy:** 図の種類に応じた「読み解き方（視点の動かし方）」を決定します。
3. **Engine (The Loop):** * AI に「現在の注目点」と「これまでの文脈」を与えます。
* AI は「そこにある情報」と「次に見るべき場所（接続先）」を返します。
* これを探索すべき箇所がなくなるまで繰り返します。



これにより、一度に全体を読ませると見落としがちな細かい矢印や条件分岐も、確実に拾い上げることができます。

## 🤝 Contributing

新しい図版タイプ（例: シーケンス図）への対応は `src/graphsight/strategies/` に新しい Strategy クラスを追加することで実現できます。PR は歓迎です！

## 📄 License

MIT License

