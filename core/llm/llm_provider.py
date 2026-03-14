#!/usr/bin/env python3
"""
core/llm/llm_provider.py — LLM Provider Abstraction

Abstract base class for LLM providers. All inference runs locally
via Ollama (Qwen 3.5 Uncensored 4b/9b).

All providers expose an identical interface so GPTManager and callers
don't need to know implementation details.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ariaska.llm.provider")


@dataclass
class LLMResponse:
    """Unified response from any LLM provider."""

    text: str
    model: str
    provider: str  # "local"
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    cached: bool = False
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and bool(self.text)


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    All providers must implement these methods to be usable
    as drop-in replacements for the OpenAI client.
    """

    @abstractmethod
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: int = 150,
        temperature: float = 0.3,
        response_format: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> LLMResponse:
        """Send a chat completion request.

        Args:
            messages: Chat messages [{"role": "system/user/assistant", "content": "..."}]
            model: Model name (provider-specific or alias)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            response_format: Optional format constraint (e.g. {"type": "json_object"})
            timeout: Request timeout in seconds

        Returns:
            LLMResponse with generated text and metadata.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is ready to serve requests."""
        ...

    @abstractmethod
    def get_model_name(self) -> str:
        """Get the primary model name for this provider."""
        ...

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return provider identifier (e.g. 'openai', 'local', 'huggingface')."""
        ...

    def get_cost_per_1k_tokens(self, model: Optional[str] = None) -> float:
        """Get cost per 1K tokens for the given model. Default: 0.0 (free/local)."""
        return 0.0

    def shutdown(self) -> None:
        """Cleanup resources. Override if provider manages server processes."""
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get provider-specific runtime statistics."""
        return {
            "provider": self.get_provider_name(),
            "model": self.get_model_name(),
            "available": self.is_available(),
        }


class LocalServerProvider(LLMProvider):
    """Provider wrapping a local LLM server (llama-cpp-python or vLLM).

    Uses the existing LocalLLMProvider infrastructure from Phase 43.
    Connects via OpenAI-compatible API on localhost.
    """

    def __init__(self) -> None:
        self._llm_provider: Any = None
        self._client: Any = None
        self._stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "errors": 0,
        }

    def _ensure_provider(self) -> Any:
        """Lazy-initialize the local LLM provider."""
        if self._llm_provider is None:
            from core.llm.local_llm_provider import get_local_llm_provider

            self._llm_provider = get_local_llm_provider()
        return self._llm_provider

    def _get_client(self) -> Any:
        """Get or create the OpenAI-compatible client."""
        if self._client is None:
            provider = self._ensure_provider()
            self._client = provider.get_client()
        return self._client

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: int = 150,
        temperature: float = 0.3,
        response_format: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> LLMResponse:
        import time

        start = time.time()
        try:
            client = self._get_client()
            provider = self._ensure_provider()
            local_model = provider.get_model_name()

            request_params: Dict[str, Any] = {
                "model": local_model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if response_format:
                request_params["response_format"] = response_format

            resp = client.chat.completions.create(**request_params)
            text = resp.choices[0].message.content or ""

            # Estimate tokens
            input_tokens = sum(len(m.get("content", "").split()) for m in messages)
            output_tokens = len(text.split())
            total_tokens = input_tokens + output_tokens
            latency = (time.time() - start) * 1000

            self._stats["total_requests"] += 1
            self._stats["total_tokens"] += total_tokens

            # Record in provider stats too
            provider.stats.record_request(tokens=total_tokens, latency_ms=latency)

            return LLMResponse(
                text=text.strip(),
                model=local_model,
                provider="local",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                latency_ms=latency,
                cost_usd=0.0,  # Local = free
            )
        except Exception as e:
            self._stats["errors"] += 1
            latency = (time.time() - start) * 1000
            logger.error(f"Local LLM request failed: {e}")
            return LLMResponse(
                text="",
                model=model,
                provider="local",
                latency_ms=latency,
                error=str(e),
            )

    def is_available(self) -> bool:
        try:
            provider = self._ensure_provider()
            return provider.is_available()
        except Exception:
            return False

    def get_model_name(self) -> str:
        try:
            return self._ensure_provider().get_model_name()
        except Exception:
            return "local-model"

    def get_provider_name(self) -> str:
        return "local"

    def shutdown(self) -> None:
        if self._llm_provider is not None:
            self._llm_provider.stop_server()
            self._llm_provider = None
            self._client = None

    def get_stats(self) -> Dict[str, Any]:
        base = super().get_stats()
        base.update(self._stats)
        try:
            provider_stats = self._ensure_provider().get_stats()
            base.update(provider_stats)
        except Exception:
            pass
        return base
