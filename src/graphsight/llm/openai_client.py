import base64
from typing import List, Type, TypeVar, Tuple
from openai import OpenAI  # Sync Client
from ..models import TokenUsage
from .base import BaseVLM
from .config import get_model_config, ModelType

T = TypeVar("T")

class OpenAIVLM(BaseVLM):
    def __init__(self, model: str = "gpt-4o", api_key: str | None = None):
        self.client = OpenAI(api_key=api_key) # Sync Client
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    def _encode_image(self, image_path: str) -> dict:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"}
        }

    def _extract_usage(self, completion) -> TokenUsage:
        if not completion.usage:
            return TokenUsage()
        return TokenUsage(
            input_tokens=completion.usage.prompt_tokens,
            output_tokens=completion.usage.completion_tokens
        )

    def _prepare_messages(self, prompt: str, image_path: str | None) -> List[dict]:
        content = [{"type": "text", "text": prompt}]
        if image_path:
            content.append(self._encode_image(image_path))
        
        config = get_model_config(self.model_name)
        if config.model_type == ModelType.REASONING:
            return [{"role": "user", "content": content}]
        else:
            return [
                {"role": "system", "content": "You are a helpful assistant specialized in diagram analysis."},
                {"role": "user", "content": content}
            ]

    def _build_request_params(self, messages, response_format=None):
        config = get_model_config(self.model_name)
        params = {
            "model": config.name,
            "messages": messages,
        }
        for k, v in config.default_params.items():
            params[k] = v
        if response_format:
            params["response_format"] = response_format
        for excl in config.excluded_params:
            if excl in params:
                del params[excl]
        return params

    def query_structured(self, prompt: str, image_path: str, response_model: Type[T]) -> Tuple[T, TokenUsage]:
        messages = self._prepare_messages(prompt, image_path)
        request_kwargs = self._build_request_params(messages=messages, response_format=response_model)
        
        # 同期呼び出し
        completion = self.client.beta.chat.completions.parse(**request_kwargs)
        return completion.choices[0].message.parsed, self._extract_usage(completion)

    def query_text(self, prompt: str, image_path: str | None = None) -> Tuple[str, TokenUsage]:
        messages = self._prepare_messages(prompt, image_path)
        request_kwargs = self._build_request_params(messages=messages)
        
        # 同期呼び出し
        completion = self.client.chat.completions.create(**request_kwargs)
        return completion.choices[0].message.content, self._extract_usage(completion)

    def calculate_cost(self, usage: TokenUsage) -> float:
        config = get_model_config(self.model_name)
        input_cost = (usage.input_tokens / 1_000_000) * config.input_price_per_m
        output_cost = (usage.output_tokens / 1_000_000) * config.output_price_per_m
        return input_cost + output_cost

