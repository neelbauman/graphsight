from abc import ABC, abstractmethod
from typing import Type, TypeVar, Tuple
from pydantic import BaseModel
from ..models import TokenUsage

T = TypeVar("T", bound=BaseModel)

class BaseVLM(ABC):
    model_name: str

    @abstractmethod
    def query_structured(self, prompt: str, image_path: str, response_model: Type[T]) -> Tuple[T, TokenUsage]:
        pass
    
    @abstractmethod
    def query_text(self, prompt: str, image_path: str | None = None) -> Tuple[str, TokenUsage]:
        pass
        
    @abstractmethod
    def calculate_cost(self, usage: TokenUsage) -> float:
        pass

