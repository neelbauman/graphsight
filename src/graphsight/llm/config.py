from pydantic import BaseModel
from loguru import logger
from typing import Dict, Any, Set
from enum import Enum


class ModelType(str, Enum):
    RECOGNITION = "recognition"  # GPT-4o系: Visual CoTが必要
    REASONING = "reasoning" # 推論モデル: CoT不要

class ModelConfig(BaseModel):
    name: str
    model_type: ModelType
    input_price_per_m: float
    output_price_per_m: float
    default_params: Dict[str, Any] = {}
    excluded_params: set = set()

MODEL_REGISTRY = {
    "gpt-4o": ModelConfig(
        name="gpt-4o-2024-08-06",
        model_type=ModelType.RECOGNITION,
        input_price_per_m=2.50,
        output_price_per_m=10.00,
        default_params={"temperature": 0.0, "max_tokens": 4096}
    ),
    "gpt-4o-mini": ModelConfig(
        name="gpt-4o-mini",
        model_type=ModelType.RECOGNITION,
        input_price_per_m=0.15,
        output_price_per_m=0.60,
        default_params={"temperature": 0.0}
    ),
    # 将来的なモデルの例 (temperature不可、reasoning_effort指定)
    "gpt-5": ModelConfig(
        name="gpt-5", # 仮
        model_type=ModelType.REASONING,
        input_price_per_m=15.00,
        output_price_per_m=60.00,
        default_params={
            "reasoning_effort": "high",
        },
        excluded_params={"temperature", "max_tokens"}
    ),
    "gpt-5.2": ModelConfig(
        name="gpt-5.2", # 仮
        model_type=ModelType.REASONING,
        input_price_per_m=1.75,
        output_price_per_m=14.00,
        default_params={
            "reasoning_effort": "high",
        },
        excluded_params={"temperature", "max_tokens"}
    ),
}

def get_model_config(model_name: str) -> ModelConfig:
    if model_name in MODEL_REGISTRY:
        # logger.info(f"Got model config of {model_name} successfully")
        return MODEL_REGISTRY[model_name]
    # エイリアス解決などが必要ならここに追加
    logger.info(f"{model_name} DOES NOT EXIST in MODEL_REGISTRY. Use gpt-4o as default.")
    return MODEL_REGISTRY["gpt-4o"]

