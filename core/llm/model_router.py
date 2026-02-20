#!/usr/bin/env python3
"""
core/llm/model_router.py — Phase 43: Intelligent Model Router

Routes LLM requests to the optimal provider based on tier:
  - nano/mini → Local GPU model (fast, free, ~15 tok/s on RTX 3090)
  - codex/full → OpenAI API (deep reasoning, expensive)

When local LLM is unavailable, ALL tiers route to OpenAI (graceful degradation).

The router intercepts model names at the GPTManager level and returns
a routing decision: {"provider": "local"|"openai", "model": "<model_name>"}.

Feature-flag gated: FF_LOCAL_LLM + FF_LOCAL_LLM_OFFLOAD_NANO/MINI.

Author: Phase 43 — GPU Acceleration Layer
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("ariaska.llm.model_router")


@dataclass(frozen=True)
class RoutingDecision:
    """Result of a model routing decision."""
    provider: str       # "local" or "openai"
    model: str          # Model name to use with that provider
    tier: str           # Budget tier: "nano", "mini", "codex", "full", "local"
    reason: str = ""    # Why this routing was chosen


# ── Tier classification ─────────────────────────────────────────────────────

# Models that belong to the nano tier (offloadable to local)
_NANO_MODELS = frozenset({
    "gpt-5-nano", "gpt-5.2-nano",
})

# Models that belong to the mini tier (offloadable to local)
_MINI_MODELS = frozenset({
    "gpt-5-mini", "gpt-5.2-mini",
})

# Models that stay on OpenAI (reasoning tier)
_CODEX_MODELS = frozenset({
    "gpt-5.2-codex", "gpt-5.1-codex", "gpt-5.1-codex-mini",
})

_FULL_MODELS = frozenset({
    "gpt-5.2",
})


def classify_tier(model: str) -> str:
    """Classify a model name into a budget tier."""
    if model in _NANO_MODELS:
        return "nano"
    if model in _MINI_MODELS:
        return "mini"
    if model in _CODEX_MODELS:
        return "codex"
    if model in _FULL_MODELS:
        return "full"
    # Unknown model — check for keywords
    ml = model.lower()
    if "nano" in ml:
        return "nano"
    if "mini" in ml:
        return "mini"
    if "codex" in ml:
        return "codex"
    return "full"


class ModelRouter:
    """
    Routes model requests to local GPU or OpenAI based on tier and availability.
    
    Offload strategy:
      - nano  → local (classification, lightweight checks)
      - mini  → local (parsing, structured extraction)
      - codex → OpenAI (deep reasoning, strategic)
      - full  → OpenAI (verification, validation)
    
    When local is unavailable, everything goes to OpenAI (no degradation).
    """
    
    def __init__(
        self,
        local_available: bool = False,
        local_model_name: str = "",
        offload_nano: bool = True,
        offload_mini: bool = True,
    ):
        self._local_available = local_available
        self._local_model_name = local_model_name
        self._offload_nano = offload_nano
        self._offload_mini = offload_mini
        self._routed_local = 0
        self._routed_openai = 0
    
    @classmethod
    def from_flags(cls) -> "ModelRouter":
        """Create router from feature flags and runtime state."""
        try:
            from core.feature_flags import get_feature_flags
            ff = get_feature_flags()
            local_llm = getattr(ff, "local_llm", False)
            offload_nano = getattr(ff, "local_llm_offload_nano", True)
            offload_mini = getattr(ff, "local_llm_offload_mini", True)
        except Exception:
            local_llm = os.getenv("FF_LOCAL_LLM", "0").lower() in ("1", "true", "yes", "on")
            offload_nano = True
            offload_mini = True
        
        local_model_name = ""
        if local_llm:
            try:
                from core.llm.local_llm_provider import get_local_llm_provider
                provider = get_local_llm_provider()
                local_model_name = provider.get_model_name()
                local_llm = provider.is_available()
            except Exception:
                local_llm = False
        
        return cls(
            local_available=local_llm,
            local_model_name=local_model_name,
            offload_nano=offload_nano,
            offload_mini=offload_mini,
        )
    
    def route(self, model: str, task_type: Optional[str] = None) -> RoutingDecision:
        """
        Route a model request to the optimal provider.
        
        Args:
            model: The requested model name (e.g. "gpt-5-nano")
            task_type: Optional task type for context-aware routing
            
        Returns:
            RoutingDecision with provider, model, and tier.
        """
        tier = classify_tier(model)
        
        # If local LLM is not available, everything goes to OpenAI
        if not self._local_available:
            self._routed_openai += 1
            return RoutingDecision(
                provider="openai", model=model, tier=tier,
                reason="local LLM unavailable"
            )
        
        # Nano → local (if flag enabled)
        if tier == "nano" and self._offload_nano:
            self._routed_local += 1
            return RoutingDecision(
                provider="local", model=self._local_model_name, tier="local",
                reason="nano offloaded to local GPU"
            )
        
        # Mini → local (if flag enabled)
        if tier == "mini" and self._offload_mini:
            self._routed_local += 1
            return RoutingDecision(
                provider="local", model=self._local_model_name, tier="local",
                reason="mini offloaded to local GPU"
            )
        
        # Codex/Full → always OpenAI
        self._routed_openai += 1
        return RoutingDecision(
            provider="openai", model=model, tier=tier,
            reason=f"{tier} stays on OpenAI"
        )
    
    def get_stats(self) -> dict:
        """Get routing statistics."""
        total = self._routed_local + self._routed_openai
        return {
            "local_available": self._local_available,
            "local_model": self._local_model_name,
            "routed_local": self._routed_local,
            "routed_openai": self._routed_openai,
            "local_pct": round(
                100 * self._routed_local / max(1, total), 1
            ),
            "offload_nano": self._offload_nano,
            "offload_mini": self._offload_mini,
        }
    
    def set_local_available(self, available: bool, model_name: str = "") -> None:
        """Update local availability at runtime."""
        self._local_available = available
        if model_name:
            self._local_model_name = model_name
