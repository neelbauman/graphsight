import sys
from pathlib import Path

# プロジェクトルートをパスに追加して src を参照できるようにする
sys.path.append(str(Path(__file__).parent.parent / "src"))

from graphsight.agent.core import GraphSightAgent

def main():
    print("🎨 Generating graph visualization...")
    
    # エージェントのインスタンス化（モデル指定はダミーでOK）
    agent = GraphSightAgent(model="gpt-4o")
    
    # グラフオブジェクトの取得
    graph = agent.graph.get_graph()
    
    # 1. Mermaid形式で出力 (コンソール)
    print("\n--- Mermaid Code ---")
    mermaid_code = graph.draw_mermaid()
    print(mermaid_code)
    print("--------------------\n")
    
    # 2. 画像として保存 (PNG)
    # Note: これには 'graphviz' などが必要な場合があります。
    # 環境によっては draw_mermaid_png() が動作しないことがあるため、
    # 失敗時はMermaidコードの使用を推奨するメッセージを出します。
    try:
        output_path = "agent_graph.png"
        graph.draw_mermaid_png(output_file_path=output_path)
        print(f"✅ Graph image saved to: {output_path}")
    except Exception as e:
        print(f"⚠️ Could not save PNG image directly (requires graphviz).")
        print(f"   Error: {e}")
        print("   👉 Copy the Mermaid code above and paste it into https://mermaid.live")

if __name__ == "__main__":
    main()
