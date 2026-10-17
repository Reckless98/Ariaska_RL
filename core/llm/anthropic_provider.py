#!/usr/bin/env python3
"""
core/llm/anthropic_provider.py — Optional Anthropic cloud mentor provider.

Architecture note
-----------------
Ariaska's intelligence is its five RL agents (PPO Actor-Critic + decision
systems). An LLM is only ever a *mentor*. This provider lets that mentor be a
hosted Claude model, interchangeable with the local Ollama mentor or with no
mentor at all. Nothing in the agents, PPO, rewards or checkpoints knows which
model family answers: every call still flows through GPTManager (invariant #1).

Optional dependency
-------------------
The ``anthropic`` SDK is imported lazily and only when ``ANTHROPIC_API_KEY``
is present. Without the key or without the SDK, ``auto_detect()`` returns
``None`` and behaviour is exactly what it was before (local or mentor-free).

Environment
-----------
    ANTHROPIC_API_KEY             activates the provider
    ANTHROPIC_WORKSPACE_ID        optional; sent as ``anthropic-workspace-id``
                                  (required for user-scoped ``sk-ant-usr-`` keys)
    ARIASKA_ANTHROPIC_MODEL       default ``claude-opus-5-5``
    ARIASKA_ANTHROPIC_EFFORT      low | medium | high | xhigh | max (default medium)
    ARIASKA_ANTHROPIC_FALLBACKS   ``1`` (default) -> server-side refusal fallbacks
    ARIASKA_ANTHROPIC_TIMEOUT     per-request seconds (default 60)
    ARIASKA_ANTHROPIC_MIN_OUTPUT  floor for ``max_tokens`` so adaptive thinking
                                  cannot starve the visible answer (default 2048)

Behavioural notes
-----------------
* Sampling knobs (``temperature``/``top_p``) are NOT sent: current Claude
  models reject non-default values. GPTManager's temperature hint is ignored.
* ``response_format={"type": "json_object"}`` becomes a system instruction;
  callers already parse JSON defensively.
* ``stop_reason == "refusal"`` is returned as an *error* LLMResponse with empty
  text so GPTManager falls back to the RL policy. It never raises.
* Auth/permission errors disable the provider for the process; rate limits and
  connection errors trigger a short cooldown so training never stalls on it.
"""

from __future__ import annotations

import importlib
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from core.llm.llm_provider import LLMProvider, LLMResponse

logger = logging.getLogger("ariaska.llm.anthropic")

DEFAULT_MODEL = "claude-opus-5-5"
_FALLBACK_BETA = "server-side-fallback-2026-07-01"
_VALID_EFFORT: Tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")

# USD per million tokens (input, output). Unknown models cost 0.0 but are still
# tracked by token count so observability never silently loses a call.
_PRICE_PER_MTOK: Dict[str, Tuple[float, float]] = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-fable-5-1": (10.0, 50.0),
}

_JSON_INSTRUCTION = (
    "Respond with a single valid JSON object and nothing else: no prose, "
    "no markdown code fences, no explanation outside the JSON."
)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _wants_json(response_format: Any) -> bool:
    if isinstance(response_format, dict):
        return response_format.get("type") == "json_object"
    return isinstance(response_format, str) and response_format == "json_object"


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost for a call from the static price table (0.0 if unknown)."""
    prices = _PRICE_PER_MTOK.get(model or "")
    if prices is None:
        return 0.0
    in_price, out_price = prices
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000.0


class AnthropicProvider(LLMProvider):
    """Claude Messages API behind Ariaska's provider-neutral LLMProvider contract."""

    # Used by GPTManager to cap the outer executor timeout for cloud calls.
    default_timeout: float

    _sdk_ok_cache: Optional[bool] = None

    def __init__(
        self,
        api_key: str,
        model: Optional[str] = None,
        effort: Optional[str] = None,
        workspace_id: Optional[str] = None,
        use_fallbacks: Optional[bool] = None,
        timeout: Optional[float] = None,
        min_output_tokens: Optional[int] = None,
        client: Any = None,
    ) -> None:
        self.api_key = api_key
        self.model = (model or os.getenv("ARIASKA_ANTHROPIC_MODEL") or DEFAULT_MODEL).strip()
        eff = (effort or os.getenv("ARIASKA_ANTHROPIC_EFFORT") or "medium").strip().lower()
        if eff not in _VALID_EFFORT:
            logger.warning(f"[ANTHROPIC] Invalid effort {eff!r}; using 'medium'")
            eff = "medium"
        self.effort = eff
        ws = workspace_id if workspace_id is not None else os.getenv("ANTHROPIC_WORKSPACE_ID", "")
        self.workspace_id = (ws or "").strip()
        self.use_fallbacks = (
            _env_bool("ARIASKA_ANTHROPIC_FALLBACKS", True) if use_fallbacks is None else bool(use_fallbacks)
        )
        self.default_timeout = float(
            timeout if timeout is not None else os.getenv("ARIASKA_ANTHROPIC_TIMEOUT", "60")
        )
        self.min_output_tokens = int(
            min_output_tokens if min_output_tokens is not None else os.getenv("ARIASKA_ANTHROPIC_MIN_OUTPUT", "2048")
        )
        self._client = client  # injectable for tests; otherwise lazily built
        self._disabled_reason = ""
        self._cooldown_until = 0.0
        self._consecutive_errors = 0
        self._latencies: List[float] = []
        self._stats: Dict[str, Any] = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost_usd": 0.0,
            "errors": 0,
            "refusals": 0,
            "fallback_disabled": False,
            "avg_latency_ms": 0.0,
        }

    # ── Detection ────────────────────────────────────────────────────────

    @classmethod
    def sdk_available(cls) -> bool:
        """True if the optional ``anthropic`` SDK can be imported (cached)."""
        if cls._sdk_ok_cache is None:
            try:
                importlib.import_module("anthropic")
                cls._sdk_ok_cache = True
            except Exception:
                cls._sdk_ok_cache = False
        return bool(cls._sdk_ok_cache)

    @classmethod
    def auto_detect(cls) -> Optional["AnthropicProvider"]:
        """Return a provider when ``ANTHROPIC_API_KEY`` is set and the SDK exists."""
        api_key = (os.getenv("ANTHROPIC_API_KEY") or "").strip()
        if not api_key:
            return None
        if not cls.sdk_available():
            logger.warning(
                "[ANTHROPIC] ANTHROPIC_API_KEY is set but the optional `anthropic` SDK is "
                "not installed (pip install anthropic). Cloud mentor disabled."
            )
            return None
        provider = cls(api_key=api_key)
        logger.info(
            f"[ANTHROPIC] Cloud mentor configured: model={provider.model} effort={provider.effort} "
            f"workspace_scoped={'yes' if provider.workspace_id else 'no'} fallbacks={provider.use_fallbacks}"
        )
        return provider

    # ── Request translation (pure, testable) ─────────────────────────────

    @staticmethod
    def build_request(
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: int,
        effort: str,
        response_format: Any = None,
        min_output_tokens: int = 2048,
    ) -> Dict[str, Any]:
        """Translate OpenAI-style chat messages into Messages API parameters.

        * ``system`` messages are concatenated into the top-level ``system``.
        * JSON mode becomes an explicit system instruction.
        * ``max_tokens`` is floored so adaptive thinking cannot starve output.
        * No sampling parameters are emitted.
        """
        system_parts: List[str] = []
        chat: List[Dict[str, str]] = []
        for m in messages or []:
            role = m.get("role")
            content = str(m.get("content", "") or "")
            if role == "system":
                system_parts.append(content)
            elif role in ("user", "assistant"):
                chat.append({"role": role, "content": content})
        if _wants_json(response_format):
            system_parts.append(_JSON_INSTRUCTION)
        if not chat or chat[0]["role"] != "user":
            chat.insert(0, {"role": "user", "content": "Begin."})

        params: Dict[str, Any] = {
            "model": model,
            "max_tokens": max(int(max_tokens or 0), int(min_output_tokens)),
            "messages": chat,
            "output_config": {"effort": effort},
        }
        system = "\n\n".join(p for p in system_parts if p)
        if system:
            params["system"] = system
        return params

    # ── LLMProvider contract ─────────────────────────────────────────────

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: int = 150,
        temperature: float = 0.3,  # intentionally unused — see module docstring
        response_format: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> LLMResponse:
        start = time.time()
        if self._disabled_reason:
            return self._fail(f"anthropic provider disabled: {self._disabled_reason}", start)
        if time.time() < self._cooldown_until:
            return self._fail("anthropic provider cooling down", start, count_error=False)

        target_model = model if (model or "").startswith("claude-") else self.model
        params = self.build_request(
            messages, target_model, max_tokens, self.effort, response_format, self.min_output_tokens
        )
        try:
            resp = self._create(params, timeout)
        except Exception as exc:  # noqa: BLE001 — classified below, never propagates
            if self.use_fallbacks and self._looks_like_fallback_rejection(exc):
                logger.warning(
                    "[ANTHROPIC] Server-side fallbacks rejected by the API; disabling them for "
                    "this process and retrying once without."
                )
                self.use_fallbacks = False
                self._stats["fallback_disabled"] = True
                try:
                    resp = self._create(params, timeout)
                except Exception as exc2:  # noqa: BLE001
                    return self._handle_exception(exc2, target_model, start)
            else:
                return self._handle_exception(exc, target_model, start)
        return self._to_response(resp, target_model, start)

    def is_available(self) -> bool:
        return bool(self.api_key) and not self._disabled_reason and self.sdk_available()

    def get_model_name(self) -> str:
        return self.model

    def get_provider_name(self) -> str:
        return "anthropic"

    def get_cost_per_1k_tokens(self, model: Optional[str] = None) -> float:
        prices = _PRICE_PER_MTOK.get(model or self.model)
        return (prices[0] / 1000.0) if prices else 0.0

    def get_stats(self) -> Dict[str, Any]:
        base = super().get_stats()
        base.update(self._stats)
        base.update({
            "effort": self.effort,
            "workspace_scoped": bool(self.workspace_id),
            "fallbacks_enabled": self.use_fallbacks,
            "disabled_reason": self._disabled_reason,
            "cooling_down": time.time() < self._cooldown_until,
        })
        return base

    # ── Internals ────────────────────────────────────────────────────────

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic  # lazy: optional dependency

            headers = {"anthropic-workspace-id": self.workspace_id} if self.workspace_id else None
            self._client = anthropic.Anthropic(
                api_key=self.api_key,
                default_headers=headers,
                timeout=self.default_timeout,
                max_retries=1,
            )
        return self._client

    def _create(self, params: Dict[str, Any], timeout: Optional[float]) -> Any:
        client = self._get_client()
        if timeout:
            client = client.with_options(timeout=float(timeout))
        if self.use_fallbacks:
            return client.beta.messages.create(betas=[_FALLBACK_BETA], fallbacks="default", **params)
        return client.messages.create(**params)

    def _record(self, in_tok: int, out_tok: int, cost: float, latency_ms: float) -> None:
        self._stats["total_requests"] += 1
        self._stats["total_tokens"] += in_tok + out_tok
        self._stats["total_input_tokens"] += in_tok
        self._stats["total_output_tokens"] += out_tok
        self._stats["total_cost_usd"] = round(self._stats["total_cost_usd"] + cost, 6)
        self._latencies.append(latency_ms)
        if len(self._latencies) > 50:
            self._latencies = self._latencies[-50:]
        self._stats["avg_latency_ms"] = sum(self._latencies) / len(self._latencies)

    def _to_response(self, resp: Any, model: str, start: float) -> LLMResponse:
        latency = (time.time() - start) * 1000
        blocks = getattr(resp, "content", None) or []
        text = "".join(
            str(getattr(b, "text", "") or "") for b in blocks if getattr(b, "type", "") == "text"
        ).strip()
        stop = getattr(resp, "stop_reason", None)
        usage = getattr(resp, "usage", None)
        in_tok = int(getattr(usage, "input_tokens", 0) or 0)
        out_tok = int(getattr(usage, "output_tokens", 0) or 0)
        served_model = str(getattr(resp, "model", None) or model)
        cost = estimate_cost_usd(served_model, in_tok, out_tok)
        self._record(in_tok, out_tok, cost, latency)

        common = dict(
            model=served_model, provider="anthropic", input_tokens=in_tok,
            output_tokens=out_tok, total_tokens=in_tok + out_tok,
            latency_ms=latency, cost_usd=cost,
        )
        if stop == "refusal":
            details = getattr(resp, "stop_details", None)
            category = getattr(details, "category", None) or "unknown"
            self._stats["refusals"] += 1
            logger.warning(
                f"[ANTHROPIC] Mentor refused (category={category}); RL policy continues without mentor"
            )
            return LLMResponse(text="", error=f"refusal:{category}", **common)
        if not text:
            self._stats["errors"] += 1
            return LLMResponse(text="", error=f"empty response (stop_reason={stop})", **common)
        self._consecutive_errors = 0
        return LLMResponse(text=text, **common)

    def _fail(self, error: str, start: float, model: Optional[str] = None, count_error: bool = True) -> LLMResponse:
        if count_error:
            self._stats["errors"] += 1
        return LLMResponse(
            text="", model=model or self.model, provider="anthropic",
            latency_ms=(time.time() - start) * 1000, error=error,
        )

    def _handle_exception(self, exc: BaseException, model: str, start: float) -> LLMResponse:
        """Classify SDK errors by name/status so no SDK import is needed here."""
        name = type(exc).__name__
        status = getattr(exc, "status_code", None)
        self._consecutive_errors += 1
        if name in ("AuthenticationError", "PermissionDeniedError") or status in (401, 403):
            self._disabled_reason = f"{name}: {str(exc)[:160]}"
            logger.error(f"[ANTHROPIC] {name} — cloud mentor disabled for this process")
        elif name == "RateLimitError" or status == 429:
            backoff = min(30 * (2 ** (self._consecutive_errors - 1)), 240)
            self._cooldown_until = time.time() + backoff
            logger.warning(f"[ANTHROPIC] Rate limited; cooling down {backoff}s")
        elif name in ("APIConnectionError", "APITimeoutError") or (isinstance(status, int) and status >= 500):
            self._cooldown_until = time.time() + 10.0
            logger.warning(f"[ANTHROPIC] {name}; cooling down 10s")
        else:
            logger.warning(f"[ANTHROPIC] Request failed: {name}: {str(exc)[:200]}")
        return self._fail(f"{name}: {str(exc)[:200]}", start, model=model)

    @staticmethod
    def _looks_like_fallback_rejection(exc: BaseException) -> bool:
        msg = str(exc).lower()
        status = getattr(exc, "status_code", None)
        is_bad_request = status == 400 or type(exc).__name__ == "BadRequestError"
        return bool(is_bad_request and ("fallback" in msg or "beta" in msg))


__all__ = ["AnthropicProvider", "DEFAULT_MODEL", "estimate_cost_usd"]
