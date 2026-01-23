# Usage

GraphSight は **CLI (コマンドライン)** または **Python ライブラリ** として利用できます。

## CLI (Command Line Interface)

基本的な使い方は、`parse` コマンドに画像パスを渡すだけです。

### 基本的な解析 (Mermaid)

デフォルトでは Mermaid 形式で出力されます。

```bash
uv run graphsight parse ./samples/flowchart.png

```

### 自然言語での解説

`--format` オプションで `natural_language` を指定すると、日本語の解説文を出力します。

```bash
uv run graphsight parse ./samples/flowchart.png --format natural_language

```

### モデルの切り替え

`--model` オプションで、使用する OpenAI モデルを指定できます。コストを抑えたい場合は `gpt-4o-mini` が便利です。

```bash
uv run graphsight parse ./samples/flowchart.png --model gpt-4o-mini

```

### ファイルへの保存

`--output` オプションで結果をファイルに保存します。

```bash
uv run graphsight parse ./samples/flowchart.png --output result.mmd

```

---

## Python API

Python スクリプト内に組み込んで使用することも可能です。

```python
from graphsight import GraphSight

# クライアントの初期化 (モデル指定も可能)
sight = GraphSight(model="gpt-4o")

# 画像のパス
image_path = "./samples/complex_chart.png"

# 1. 解析実行 (デフォルト: Mermaid)
result = sight.interpret(image_path)

print(f"--- {result.diagram_type} ---")
print(result.content)      # AIにより洗練された結果
print(result.raw_content)  # 機械的に結合された生の結果

# 2. 自然言語モードで実行
result_nl = sight.interpret(image_path, format="natural_language")
print(result_nl.content)

# 3. コスト情報の確認
print(f"Cost: ${result.cost_usd:.4f}")

```

