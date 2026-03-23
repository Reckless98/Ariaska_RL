"""Phase 56: Cloud Reasoning Provider — High-quality strategic LLM via free APIs.

Provides deep reasoning capability for the Strategic Advisor and other
high-stakes decisions where PPO/DDQN need expert-level guidance.

Supported backends (all have free tiers):
  - Groq:      300+ tok/s, Llama 3.3 70B, 30 req/min free
  - OpenRouter: Multi-provider routing, some free models
  - Together:   Fast inference, free tier available

Multi-provider failover: if primary (Groq) rate-limits or errors,
automatically retries on next available provider (OpenRouter, Together).
This doubles/triples monthly free capacity.

Architecture:
    Cloud LLM (70B) produces expert strategic plans
    → PPO/DDQN execute plans, receive reward signals
    → RL agents internalize reasoning patterns over time
    = Teacher-student knowledge distillation via environment interaction
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

from core.llm.llm_provider import LLMProvider, LLMResponse

logger = logging.getLogger("ariaska.llm.cloud_reasoning")

# ── Default models per provider ──────────────────────────────────────────────

_PROVIDER_DEFAULTS: Dict[str, Dict[str, str]] = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
        "env_key": "GROQ_API_KEY",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "meta-llama/llama-3.3-70b-instruct",
        "env_key": "OPENROUTER_API_KEY",
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "env_key": "TOGETHER_API_KEY",
    },
}


class _SingleCloudBackend:
    """One cloud API backend (Groq, OpenRouter, or Together)."""

    def __init__(self, name: str, api_key: str, base_url: str, model: str) -> None:
        self.name = name
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self._client: Any = None
        self.consecutive_errors = 0
        self.cooldown_until: float = 0.0

    def get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    @property
    def is_cooled_down(self) -> bool:
        return time.time() < self.cooldown_until

    def mark_success(self) -> None:
        self.consecutive_errors = 0

    def mark_failure(self) -> None:
        self.consecutive_errors += 1
        # Exponential backoff: 30s, 60s, 120s, 240s max
        backoff = min(30 * (2 ** (self.consecutive_errors - 1)), 240)
        self.cooldown_until = time.time() + backoff
        logger.info(
            f"[CLOUD] {self.name} cooldown {backoff}s after {self.consecutive_errors} errors"
        )


class CloudReasoningProvider(LLMProvider):
    """Cloud LLM provider with multi-provider failover.

    Tries providers in priority order (Groq → OpenRouter → Together).
    On failure, automatically retries on the next available provider.
    Failed providers enter exponential cooldown before being retried.

    This maximizes free tier capacity across multiple providers.
    """

    def __init__(self, backends: List[_SingleCloudBackend]) -> None:
        self._backends = backends
        self._active_idx = 0
        self._stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "errors": 0,
            "failovers": 0,
            "avg_latency_ms": 0.0,
            "total_cost_usd": 0.0,
            "requests_by_provider": {},
        }
        self._latencies: list[float] = []

        names = [b.name for b in backends]
        logger.info(f"[CLOUD] Multi-provider init: {names} (primary: {names[0]})")

    @classmethod
    def auto_detect(cls) -> Optional["CloudReasoningProvider"]:
        """Create provider from all available API keys.

        Builds a failover chain from every configured provider.
        Priority: GROQ > OPENROUTER > TOGETHER.
        """
        backends: List[_SingleCloudBackend] = []
        for name, config in _PROVIDER_DEFAULTS.items():
            api_key = os.getenv(config["env_key"], "")
            if api_key:
                model = os.getenv("ARIASKA_CLOUD_MODEL", config["model"])
                backends.append(_SingleCloudBackend(
                    name=name,
                    api_key=api_key,
                    base_url=config["base_url"],
                    model=model,
                ))
                logger.info(f"[CLOUD] Detected {name} provider ({model})")

        # Check generic cloud config
        generic_key = os.getenv("ARIASKA_CLOUD_API_KEY", "")
        generic_url = os.getenv("ARIASKA_CLOUD_BASE_URL", "")
        generic_model = os.getenv("ARIASKA_CLOUD_MODEL", "")
        if generic_key and generic_url and generic_model:
            backends.append(_SingleCloudBackend(
                name="custom",
                api_key=generic_key,
                base_url=generic_url,
                model=generic_model,
            ))

        if not backends:
            return None

        return cls(backends)

    def _pick_backend(self) -> Optional[_SingleCloudBackend]:
        """Pick next available backend, skipping cooled-down ones."""
        for backend in self._backends:
            if not backend.is_cooled_down:
                return backend
        # All backends in cooldown — use the one with earliest cooldown end
        return min(self._backends, key=lambda b: b.cooldown_until)

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: int = 150,
        temperature: float = 0.3,
        response_format: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> LLMResponse:
        last_error = ""
        for attempt in range(len(self._backends)):
            backend = self._pick_backend()
            if backend is None:
                break

            start = time.time()
            try:
                client = backend.get_client()
                params: Dict[str, Any] = {
                    "model": backend.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                if response_format:
                    params["response_format"] = response_format
                if timeout:
                    params["timeout"] = timeout

                resp = client.chat.completions.create(**params)
                text = resp.choices[0].message.content or ""

                input_tokens = getattr(resp.usage, "prompt_tokens", 0) if resp.usage else 0
                output_tokens = getattr(resp.usage, "completion_tokens", 0) if resp.usage else 0
                total_tokens = input_tokens + output_tokens
                latency_ms = (time.time() - start) * 1000

                # Update stats
                backend.mark_success()
                self._stats["total_requests"] += 1
                self._stats["total_tokens"] += total_tokens
                self._stats["total_input_tokens"] += input_tokens
                self._stats["total_output_tokens"] += output_tokens
                self._latencies.append(latency_ms)
                if len(self._latencies) > 50:
                    self._latencies = self._latencies[-50:]
                self._stats["avg_latency_ms"] = sum(self._latencies) / len(self._latencies)
                prov_key = backend.name
                self._stats["requests_by_provider"][prov_key] = (
                    self._stats["requests_by_provider"].get(prov_key, 0) + 1
                )
                if attempt > 0:
                    self._stats["failovers"] += 1

                logger.debug(
                    f"[CLOUD] {backend.name} | {latency_ms:.0f}ms | "
                    f"{total_tokens} tokens | {text[:80]}..."
                )

                return LLMResponse(
                    text=text.strip(),
                    model=backend.model,
                    provider=backend.name,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    latency_ms=latency_ms,
                    cost_usd=0.0,
                )

            except Exception as e:
                latency_ms = (time.time() - start) * 1000
                backend.mark_failure()
                last_error = str(e)
                self._stats["errors"] += 1
                logger.warning(
                    f"[CLOUD] {backend.name} failed (attempt {attempt + 1}): {e}"
                )
                if attempt < len(self._backends) - 1:
                    logger.info(f"[CLOUD] Failing over to next provider...")
                continue

        # All backends failed
        return LLMResponse(
            text="",
            model=self._backends[0].model if self._backends else "unknown",
            provider="cloud-all-failed",
            latency_ms=0.0,
            error=f"All cloud providers failed: {last_error}",
        )

    def is_available(self) -> bool:
        return len(self._backends) > 0

    def get_model_name(self) -> str:
        if self._backends:
            return self._backends[0].model
        return "unknown"

    def get_provider_name(self) -> str:
        names = [b.name for b in self._backends]
        return "+".join(names) if names else "cloud"

    def get_stats(self) -> Dict[str, Any]:
        base = super().get_stats()
        base.update(self._stats)
        base["backends"] = [
            {
                "name": b.name,
                "model": b.model,
                "consecutive_errors": b.consecutive_errors,
                "cooled_down": b.is_cooled_down,
            }
            for b in self._backends
        ]
        return base
