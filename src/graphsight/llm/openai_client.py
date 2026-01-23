import os
from typing import Type, TypeVar, Tuple, Any, Dict
from pydantic import BaseModel
from openai import OpenAI
from .base import BaseVLM
from .config import get_model_config
from ..utils.image import encode_image_to_base64
from ..models import TokenUsage

T = TypeVar("T", bound=BaseModel)

class OpenAIVLM(BaseVLM):
    def __init__(self, model: str = "gpt-4o", api_key: str | None = None):
        self.client = OpenAI(api_key=api_key)
        self.config = get_model_config(model)
        self.model_name = self.config.name

    def _prepare_messages(self, prompt: str, image_path: str | None = None):
        content = [{"type": "text", "text": prompt}]
        if image_path:
            base64_image = encode_image_to_base64(image_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            })
        return [{"role": "user", "content": content}]

    def _extract_usage(self, response) -> TokenUsage:
        if response.usage:
            return TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens
            )
        return TokenUsage()

    def _build_request_params(self, messages: list, **kwargs) -> Dict[str, Any]:
        params = {
            "model": self.model_name,
            "messages": messages,
        }
        combined_args = {**self.config.default_params, **kwargs}
        for key in self.config.excluded_params:
            if key in combined_args:
                combined_args.pop(key)
        params.update(combined_args)
        return params

    def query_structured(self, prompt: str, image_path: str, response_model: Type[T]) -> Tuple[T, TokenUsage]:
        messages = self._prepare_messages(prompt, image_path)
        request_kwargs = self._build_request_params(messages=messages, response_format=response_model)
        completion = self.client.beta.chat.completions.parse(**request_kwargs)
        return completion.choices[0].message.parsed, self._extract_usage(completion)

    def query_text(self, prompt: str, image_path: str | None = None) -> Tuple[str, TokenUsage]:
        messages = self._prepare_messages(prompt, image_path)
        request_kwargs = self._build_request_params(messages=messages)
        completion = self.client.chat.completions.create(**request_kwargs)
        return completion.choices[0].message.content, self._extract_usage(completion)

    def calculate_cost(self, usage: TokenUsage) -> float:
        return usage.calculate_cost(self.config.input_price_per_m, self.config.output_price_per_m)

