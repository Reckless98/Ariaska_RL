#!/usr/bin/env python3
"""
core/llm/llm_provider.py — Phase 44: LLM Provider Abstraction

Abstract base class for LLM providers. Enables transparent switching
between OpenAI, local LLM (llama-cpp/vLLM), and HuggingFace providers.

All providers expose an identical interface so GPTManager and callers
don't need to know *where* inference happens.

Author: Phase 44 — OpenAI Removal & Local LLM Migration
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
    provider: str  # "openai", "local", "huggingface"
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


class OpenAIProvider(LLMProvider):
    """Provider wrapping the OpenAI API (GPT-5.x, codex, etc.).

    Uses the official openai Python SDK. Supports both Chat Completions
    and the Responses API for codex models.
    """

    # Cost per 1K tokens (USD) — blended input+output estimate
    COST_MAP: Dict[str, float] = {
        "local-llm": 0.00010,
        "qwen2.5:7b": 0.00,
        "qwen2.5-coder:3b": 0.00,
        "local": 0.00,
    }

    def __init__(self, api_key: Optional[str] = None) -> None:
        import os

        self._api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._client: Any = None
        self._async_client: Any = None
        self._stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "errors": 0,
        }

    def _get_client(self) -> Any:
        """Lazy-initialize the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI

                if not self._api_key:
                    raise RuntimeError("OPENAI_API_KEY not set")
                self._client = OpenAI(api_key=self._api_key)
            except ImportError:
                raise RuntimeError(
                    "openai package not installed. Install with: pip install openai"
                )
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
            uses_responses_api = "codex" in model
            uses_new_api = any(x in model for x in ["gpt-5", "o1-", "o3-"])

            if uses_responses_api:
                # Codex models use Responses API
                system_msg = next(
                    (m["content"] for m in messages if m["role"] == "system"), ""
                )
                user_msg = next(
                    (m["content"] for m in messages if m["role"] == "user"), ""
                )
                codex_budget = max(max_tokens * 15, 2000)
                resp = client.responses.create(
                    model=model,
                    instructions=system_msg,
                    input=user_msg,
                    max_output_tokens=codex_budget,
                )
                # Extract text from Responses API
                text = ""
                if hasattr(resp, "output"):
                    for block in resp.output:
                        if hasattr(block, "content"):
                            for part in block.content:
                                if hasattr(part, "text"):
                                    text += part.text
                elif hasattr(resp, "output_text"):
                    text = resp.output_text

                input_tokens = getattr(
                    getattr(resp, "usage", None), "input_tokens", 0
                )
                output_tokens = getattr(
                    getattr(resp, "usage", None), "output_tokens", 0
                )
            else:
                # Standard Chat Completions
                token_param = (
                    "max_completion_tokens" if uses_new_api else "max_tokens"
                )
                request_params: Dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    token_param: max_tokens,
                }
                if not uses_new_api:
                    request_params["temperature"] = temperature
                if response_format:
                    request_params["response_format"] = response_format
                if timeout:
                    request_params["timeout"] = min(timeout, 600.0)

                resp = client.chat.completions.create(**request_params)
                text = resp.choices[0].message.content or ""
                input_tokens = getattr(resp.usage, "prompt_tokens", 0) if resp.usage else 0
                output_tokens = getattr(resp.usage, "completion_tokens", 0) if resp.usage else 0

            total_tokens = input_tokens + output_tokens
            cost = total_tokens * self.get_cost_per_1k_tokens(model) / 1000.0
            latency = (time.time() - start) * 1000

            self._stats["total_requests"] += 1
            self._stats["total_tokens"] += total_tokens
            self._stats["total_cost_usd"] += cost

            return LLMResponse(
                text=text.strip(),
                model=model,
                provider="openai",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                latency_ms=latency,
                cost_usd=cost,
            )
        except Exception as e:
            self._stats["errors"] += 1
            latency = (time.time() - start) * 1000
            logger.error(f"OpenAI request failed: {e}")
            return LLMResponse(
                text="",
                model=model,
                provider="openai",
                latency_ms=latency,
                error=str(e),
            )

    def is_available(self) -> bool:
        return bool(self._api_key)

    def get_model_name(self) -> str:
        return "local-llm"

    def get_provider_name(self) -> str:
        return "openai"

    def get_cost_per_1k_tokens(self, model: Optional[str] = None) -> float:
        return self.COST_MAP.get(model or "", 0.01)

    def get_stats(self) -> Dict[str, Any]:
        base = super().get_stats()
        base.update(self._stats)
        return base


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
