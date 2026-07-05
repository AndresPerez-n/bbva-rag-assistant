"""LLM providers behind a factory.

Design pattern: **Factory Method**. :class:`LLMFactory` builds the configured
provider; callers depend only on the abstract :class:`LLMClient` (``generate`` /
``stream``). Claude (Anthropic) is the default; OpenAI is supported as an
alternative. Switching providers/models is a configuration change, not a code
change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterator, List

from src.config import Settings, get_settings


class LLMClient(ABC):
    """Provider-agnostic chat interface."""

    @abstractmethod
    def generate(self, system: str, messages: List[Dict[str, str]]) -> str:
        ...

    @abstractmethod
    def stream(self, system: str, messages: List[Dict[str, str]]) -> Iterator[str]:
        ...


class AnthropicLLM(LLMClient):
    """Claude via the Anthropic Messages API."""

    def __init__(self, api_key: str, model: str, max_tokens: int) -> None:
        import anthropic

        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, system: str, messages: List[Dict[str, str]]) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
        )
        return "".join(block.text for block in resp.content if block.type == "text")

    def stream(self, system: str, messages: List[Dict[str, str]]) -> Iterator[str]:
        with self._client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                yield text


class OpenAILLM(LLMClient):
    """OpenAI Chat Completions (optional alternative provider)."""

    def __init__(self, api_key: str, model: str, max_tokens: int) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError("LLM_PROVIDER=openai requires `pip install openai`") from exc
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        self._client = OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def _payload(self, system: str, messages: List[Dict[str, str]]):
        return [{"role": "system", "content": system}, *messages]

    def generate(self, system: str, messages: List[Dict[str, str]]) -> str:
        resp = self._client.chat.completions.create(
            model=self.model, max_tokens=self.max_tokens, messages=self._payload(system, messages)
        )
        return resp.choices[0].message.content or ""

    def stream(self, system: str, messages: List[Dict[str, str]]) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=self._payload(system, messages),
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


class LLMFactory:
    """Instantiate the configured provider. Add new providers here only."""

    @staticmethod
    def create(settings: Settings | None = None) -> LLMClient:
        settings = settings or get_settings()
        provider = (settings.llm_provider or "anthropic").lower()
        if provider == "anthropic":
            return AnthropicLLM(settings.anthropic_api_key, settings.llm_model, settings.llm_max_tokens)
        if provider == "openai":
            return OpenAILLM(settings.openai_api_key, settings.llm_model, settings.llm_max_tokens)
        raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider!r}")
