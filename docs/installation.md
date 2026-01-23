# Installation

GraphSight は Python 3.10 以上で動作します。パッケージ管理には高速な [uv](https://github.com/astral-sh/uv) を推奨しています。

## 前提条件

* Python >= 3.10
* `uv` (推奨) または `pip`
* OpenAI API Key

## インストール手順

リポジトリをクローンし、依存関係をインストールします。

```bash
# リポジトリのクローン
git clone [https://github.com/your-username/graphsight.git](https://github.com/your-username/graphsight.git)
cd graphsight

# 依存関係のインストール (uvの場合)
uv sync

```

## APIキーの設定

GraphSight は OpenAI の Vision-Language Model を使用します。
プロジェクトルートに `.env` ファイルを作成し、APIキーを設定してください。

```ini title=".env"
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

```

!!! warning "API利用料について"
画像解析はトークン消費量が多いため、利用時には OpenAI の料金体系にご注意ください。GraphSight は実行ごとに概算コストを表示します。

