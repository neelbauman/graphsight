import base64
import json
from typing import List, Type, TypeVar, Tuple
from openai import OpenAI, APITimeoutError
from beautyspot import KeyGen
import pickle
from loguru import logger
from ..models import TokenUsage, spot
from .base import BaseVLM
from .config import get_model_config, ModelType

T = TypeVar("T")

class OpenAIVLM(BaseVLM):
    # --- 変更点: デフォルトタイムアウトを 600秒 (10分) に延長 ---
    def __init__(self, model: str = "gpt-4o", api_key: str | None = None, timeout: float = 600.0):
        # クライアント初期化時にタイムアウトを設定
        self.client = OpenAI(api_key=api_key, timeout=timeout)
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    def _encode_image(self, image_path: str) -> dict:
        try:
            with open(image_path, "rb") as image_file:
                data = image_file.read()
                encoded_string = base64.b64encode(data).decode("utf-8")
            logger.debug(f"✅ Encoded image: {len(encoded_string)} chars")
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_string}",
                    "detail": "high",
                }
            }
        except Exception as e:
            logger.error(f"❌ Image encoding failed: {e}")
            raise

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
            logger.debug(f"📎 Encoding image: {image_path}")
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
        
        # response_formatがある場合は優先設定
        if response_format:
            params["response_format"] = response_format
            
        for excl in config.excluded_params:
            if excl in params:
                del params[excl]
        return params

    @spot.mark(
        keygen=KeyGen.map(
            self=KeyGen.IGNORE,
        ),
        retention="1h",
    )
    def query_structured(self, prompt: str, image_path: str, response_model: Type[T]) -> Tuple[T, TokenUsage]:
        logger.info(f"📤 Sending Structured Request to {self.model_name} (JSON Mode, Timeout={self.client.timeout}s)...")
        
        # 1. スキーマ情報をプロンプトに追加 (JSON Mode用)
        try:
            schema_json = json.dumps(response_model.model_json_schema(), indent=2)
            prompt_with_schema = (
                f"{prompt}\n\n"
                "IMPORTANT: Output must be a valid JSON object conforming to this schema:\n"
                f"```json\n{schema_json}\n```"
            )
        except Exception as e:
            logger.warning(f"⚠️ Failed to generate JSON schema: {e}")
            prompt_with_schema = prompt

        logger.debug("   Preparing messages...")
        messages = self._prepare_messages(prompt_with_schema, image_path)
        logger.debug(f"   Messages prepared. (Payload size ~{sum(len(str(m)) for m in messages)} bytes)")

        # 2. JSON Mode を指定 ("type": "json_object")
        request_kwargs = self._build_request_params(
            messages=messages, 
            response_format={"type": "json_object"}
        )
        
        try:
            # 3. 標準の create メソッドを使用
            logger.info("   Calling OpenAI API (chat.completions.create)...")
            completion = self.client.chat.completions.create(**request_kwargs)
            
            logger.info("📥 Received response from OpenAI.")
            
            content = completion.choices[0].message.content
            if not content:
                raise ValueError("Empty response content from LLM.")

            # 4. Pydanticで手動パース・検証
            parsed = response_model.model_validate_json(content)
            
            return parsed, self._extract_usage(completion)
            
        except APITimeoutError:
            logger.error(f"❌ OpenAI API Timeout after {self.client.timeout}s.")
            raise
        except Exception as e:
            logger.error(f"❌ OpenAI API Error: {e}")
            raise

    @spot.mark(keygen=KeyGen.map(self=KeyGen.IGNORE))
    def query_text(self, prompt: str, image_path: str | None = None) -> Tuple[str, TokenUsage]:
        logger.info(f"📤 Sending Text Request to {self.model_name}...")
        
        messages = self._prepare_messages(prompt, image_path)
        request_kwargs = self._build_request_params(messages=messages)
        
        try:
            completion = self.client.chat.completions.create(**request_kwargs)
            logger.info("📥 Received text response.")
            return completion.choices[0].message.content, self._extract_usage(completion)
        except Exception as e:
            logger.error(f"❌ OpenAI API Error: {e}")
            raise

    def calculate_cost(self, usage: TokenUsage) -> float:
        config = get_model_config(self.model_name)
        input_cost = (usage.input_tokens / 1_000_000) * config.input_price_per_m
        output_cost = (usage.output_tokens / 1_000_000) * config.output_price_per_m
        return input_cost + output_cost

