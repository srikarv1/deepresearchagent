from __future__ import annotations

import os
from typing import Any

from openai import AsyncOpenAI

from adr.core.types import TokenUsage
from adr.llm.base import LLMResponse, Message

# gpt-5 / o-series reject `max_tokens` (want max_completion_tokens) and often
# reject temperature other than the default. Used by the BCP judge stand-in.
_REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def is_reasoning_model(model: str) -> bool:
    return (model or "").lower().startswith(_REASONING_PREFIXES)


def chat_token_kwargs(
    model: str, max_tokens: int, temperature: float | None
) -> dict[str, Any]:
    """Chat Completions extra kwargs that Azure Foundry / gpt-5 will accept."""
    if is_reasoning_model(model):
        out: dict[str, Any] = {"max_completion_tokens": int(max_tokens)}
        if temperature in (1, 1.0):
            out["temperature"] = temperature
        return out
    out = {"max_tokens": int(max_tokens)}
    if temperature is not None:
        out["temperature"] = temperature
    return out


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
        max_out = self.default_max_tokens if max_tokens is None else max_tokens
        temp = self.default_temperature if temperature is None else temperature
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": payload,
            **chat_token_kwargs(self.model, max_out, temp),
        }
        if extra:
            kwargs.update(extra)
        try:
            response = await self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            msg = str(exc).lower()
            retry = False
            if "max_completion_tokens" in msg and "max_tokens" in kwargs:
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
                retry = True
            if "temperature" in msg and "temperature" in kwargs:
                kwargs.pop("temperature", None)
                retry = True
            if not retry:
                raise
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
