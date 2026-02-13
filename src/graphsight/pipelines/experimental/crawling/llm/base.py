from abc import ABC, abstractmethod
from typing import Tuple, Type, TypeVar
from ..models import TokenUsage

T = TypeVar("T")

class BaseVLM(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str:
        """モデル名を返すプロパティ"""
        pass

    @abstractmethod
    def query_structured(self, prompt: str, image_path: str, response_model: Type[T]) -> Tuple[T, TokenUsage]:
        """構造化データ(Pydanticモデル)を同期で問い合わせる"""
        pass

    @abstractmethod
    def query_text(self, prompt: str, image_path: str | None = None) -> Tuple[str, TokenUsage]:
        """テキスト(str)を同期で問い合わせる"""
        pass
    
    @abstractmethod
    def calculate_cost(self, usage: TokenUsage) -> float:
        """トークン使用量からコスト(USD)を計算する"""
        pass

