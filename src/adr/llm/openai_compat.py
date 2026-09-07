from __future__ import annotations

import os
from typing import Any

from openai import AsyncOpenAI

from adr.core.types import TokenUsage
from adr.llm.base import LLMResponse, Message


class OpenAICompatLLM:
    """Chat client for OpenAI, OpenRouter, vLLM, Ollama, Together, etc."""

    name = "openai_compat"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        default_temperature: float = 0.0,
        default_max_tokens: int = 2048,
    ) -> None:
        self.model = model
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        key = api_key or os.environ.get(api_key_env)
        if key is None and api_key_env == "OPENAI_API_KEY":
            key = os.environ.get("OPENAI_API_KEY")
        key = key or "dummy"
        kwargs: dict[str, Any] = {"api_key": key}
        if base_url:
            kwargs["base_url"] = base_url.rstrip("/")
        self._client = AsyncOpenAI(**kwargs)

    async def complete(
        self,
        messages: list[Message] | list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> LLMResponse:
        payload = [
            {"role": m.role, "content": m.content} if isinstance(m, Message) else dict(m)
            for m in messages
        ]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": payload,
            "temperature": self.default_temperature if temperature is None else temperature,
            "max_tokens": self.default_max_tokens if max_tokens is None else max_tokens,
        }
        if extra:
            kwargs.update(extra)
        response = await self._client.chat.completions.create(**kwargs)
        choice = response.choices[0].message
        text = choice.content or ""
        usage = TokenUsage()
        if response.usage:
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens or 0,
                completion_tokens=response.usage.completion_tokens or 0,
                total_tokens=response.usage.total_tokens or 0,
            )
        return LLMResponse(text=text, usage=usage, raw=response.model_dump())
