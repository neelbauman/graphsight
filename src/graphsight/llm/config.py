from pydantic import BaseModel
from typing import Dict, Any

class ModelConfig(BaseModel):
    name: str
    input_price_per_m: float
    output_price_per_m: float
    default_params: Dict[str, Any] = {}
    excluded_params: set = set()

MODEL_REGISTRY = {
    "gpt-4o": ModelConfig(
        name="gpt-4o-2024-08-06",
        input_price_per_m=2.50,
        output_price_per_m=10.00,
        default_params={"temperature": 0.0, "max_tokens": 4096}
    ),
    "gpt-4o-mini": ModelConfig(
        name="gpt-4o-mini",
        input_price_per_m=0.15,
        output_price_per_m=0.60,
        default_params={"temperature": 0.0}
    ),
    # 将来的なモデルの例 (temperature不可、reasoning_effort指定)
    "gpt-5": ModelConfig(
        name="gpt-5", # 仮
        input_price_per_m=15.00,
        output_price_per_m=60.00,
        default_params={"reasoning_effort": "medium"},
        excluded_params={"temperature", "max_tokens"}
    ),
}

def get_model_config(model_name: str) -> ModelConfig:
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name]
    # エイリアス解決などが必要ならここに追加
    return MODEL_REGISTRY["gpt-4o"]

