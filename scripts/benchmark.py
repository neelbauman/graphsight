import time
import sys
from pathlib import Path
# パスを通す（プロジェクトルートで実行想定）
sys.path.append(str(Path(__file__).parent.parent / "src"))

from graphsight import GraphSight

def run_benchmark(image_path: str):
    print(f"🚀 Benchmarking: {image_path}")
    
    # 1. Baseline (GPT-4o)
    print("\n--- Running GPT-4o (Baseline) ---")
    start = time.time()
    sight_4o = GraphSight(model="gpt-5.2")
    res_4o = sight_4o.interpret(image_path)
    duration_4o = time.time() - start
    print(f"Time: {duration_4o:.2f}s | Cost: ${res_4o.cost_usd:.4f} | Steps: {res_4o.full_description}")

    # 2. Challenger (GPT-4o-mini)
    # ※ ここで FastFlowchartStrategy が使われるように api.py を調整済みと仮定
    print("\n--- Running GPT-4o-mini (Optimized) ---")
    start = time.time()
    sight_mini = GraphSight(model="gpt-4o-mini")
    res_mini = sight_mini.interpret(image_path)
    duration_mini = time.time() - start
    print(f"Time: {duration_mini:.2f}s | Cost: ${res_mini.cost_usd:.4f} | Steps: {res_mini.full_description}")
    
    # Summary
    print("\n📊 Impact Summary")
    cost_reduction = (1 - res_mini.cost_usd / res_4o.cost_usd) * 100
    time_reduction = (1 - duration_mini / duration_4o) * 100
    print(f"💰 Cost Reduced: {cost_reduction:.1f}%")
    print(f"⚡ Time Reduced: {time_reduction:.1f}%")

if __name__ == "__main__":
    target_image = "./samples/sample-6.png" # 存在する画像パスを指定
    run_benchmark(target_image)

