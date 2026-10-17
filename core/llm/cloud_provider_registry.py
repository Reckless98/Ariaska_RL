#!/usr/bin/env python3
"""
core/llm/cloud_provider_registry.py — Pluggable cloud mentor providers.

GPTManager asks this registry for *one* cloud mentor provider at startup. Each
factory is lazy: no SDK is imported and no network call is made unless that
provider's credentials are present in the environment. Adding a new cloud
mentor is one factory + one registration — the agents, PPO and GPTManager
routing do not change.

Operator control:
    ARIASKA_CLOUD_PROVIDER = auto | anthropic | openai-compatible
        auto               try providers in CLOUD_PROVIDER_ORDER, first available wins
        anthropic          only the Anthropic provider (ANTHROPIC_API_KEY)
        openai-compatible  only Groq / OpenRouter / Together / custom base URL
                           (aliases: groq, openrouter, together, custom)
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Dict, List, Optional

from core.llm.llm_provider import LLMProvider

logger = logging.getLogger("ariaska.llm.cloud_registry")

ENV_VAR = "ARIASKA_CLOUD_PROVIDER"
CLOUD_PROVIDER_ORDER: tuple = ("anthropic", "openai-compatible")

_ALIASES: Dict[str, str] = {
    "claude": "anthropic",
    "groq": "openai-compatible",
    "openrouter": "openai-compatible",
    "together": "openai-compatible",
    "custom": "openai-compatible",
    "openai_compatible": "openai-compatible",
    "openai": "openai-compatible",
}

ProviderFactory = Callable[[], Optional[LLMProvider]]


def _anthropic_factory() -> Optional[LLMProvider]:
    from core.llm.anthropic_provider import AnthropicProvider

    return AnthropicProvider.auto_detect()


def _openai_compatible_factory() -> Optional[LLMProvider]:
    from core.llm.cloud_reasoning_provider import CloudReasoningProvider

    return CloudReasoningProvider.auto_detect()


_FACTORIES: Dict[str, ProviderFactory] = {
    "anthropic": _anthropic_factory,
    "openai-compatible": _openai_compatible_factory,
}


def register_cloud_provider(name: str, factory: ProviderFactory) -> None:
    """Register (or override) a cloud provider factory. Extension point for tests/plugins."""
    _FACTORIES[name.strip().lower()] = factory


def normalize_preference(preference: Optional[str]) -> str:
    raw = (preference if preference is not None else os.getenv(ENV_VAR, "auto")) or "auto"
    raw = raw.strip().lower()
    raw = _ALIASES.get(raw, raw)
    if raw != "auto" and raw not in _FACTORIES:
        logger.warning(
            f"[CLOUD] Unknown {ENV_VAR}={raw!r}; expected auto or one of {sorted(_FACTORIES)}. Using 'auto'."
        )
        return "auto"
    return raw


def detection_order(preference: Optional[str] = None) -> List[str]:
    pref = normalize_preference(preference)
    if pref == "auto":
        ordered = [n for n in CLOUD_PROVIDER_ORDER if n in _FACTORIES]
        extras = [n for n in _FACTORIES if n not in ordered]
        return ordered + extras
    return [pref]


def detect_cloud_provider(preference: Optional[str] = None) -> Optional[LLMProvider]:
    """Return the first available cloud provider per preference, or ``None``.

    Never raises: a misbehaving factory is logged and skipped so a broken
    optional dependency can never take the mentor-free RL path down with it.
    """
    for name in detection_order(preference):
        factory = _FACTORIES.get(name)
        if factory is None:
            continue
        try:
            provider = factory()
        except Exception as exc:  # noqa: BLE001 — optional providers must never break startup
            logger.debug(f"[CLOUD] Provider factory {name!r} failed: {exc}")
            continue
        if provider is not None and provider.is_available():
            logger.info(
                f"[CLOUD] Mentor provider selected: {name} "
                f"({provider.get_provider_name()} / {provider.get_model_name()})"
            )
            return provider
    return None


__all__ = [
    "ENV_VAR",
    "CLOUD_PROVIDER_ORDER",
    "register_cloud_provider",
    "normalize_preference",
    "detection_order",
    "detect_cloud_provider",
]
