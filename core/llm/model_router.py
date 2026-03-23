#!/usr/bin/env python3
"""
core/llm/model_router.py — Phase 57: Hybrid Cloud+Local Model Router

Routes LLM requests to the optimal provider based on task complexity:
  - System 1 (local 4B): classification, parsing, command selection — fast reflexive
  - System 2 (local 4B): recon, defensive, general — balanced reasoning
  - System 3: strategic planning, postmortem — deep reasoning
      Priority: local-strategic 9B > local 4B (cloud removed)

Phase 58 — Local-first routing (cloud removed from chain):
  1. Local 4B (ariaska-cybersec4) for System 1+2 fast/tactical tasks.
  2. Local 9B (ariaska-strategic-9b) for System 3 when available —
     fine-tuned on Ariaska schemas, zero latency, no API limits.
  3. Local 4B as System 3 fallback when 9B not available.
  Cloud removed: free tiers hit daily rate limits, and cloud models
  hallucinate wrong target IPs (192.168.x.x instead of actual target).

Feature-flag gated: FF_LOCAL_LLM + FF_LOCAL_LLM_OFFLOAD_ALL.

Author: Phase 43/44/54/56/57 — Hybrid Cloud+Local
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
    provider: str       # "local" or "cloud"
    model: str          # Model name for the provider
    tier: str           # Budget tier: "local" or "cloud"
    reason: str = ""    # Why this routing was chosen


# ── Task-type to cognitive tier mapping ─────────────────────────────────────

# System 1: Fast reflexive tasks → local 4B (low latency, structured output)
_SYSTEM1_TASKS = frozenset({
    "classification", "diversify", "parsing", "playbook",
    "command_selection", "output_parse",
})

# System 2: Medium reasoning → local 4B (same model, different task pool)
_SYSTEM2_TASKS = frozenset({
    "reconnaissance", "defensive", "general", "embedding",
    "tactical", "learning", "reflection",
})

# System 3: Deep reasoning → local 9B or local 4B (strategic planning, analysis)
# P58: Cloud removed — free tiers rate-limit and hallucinate wrong IPs.
_SYSTEM3_TASKS = frozenset({
    "strategic", "postmortem", "analysis", "reasoning",
})

# ── Local model names (Ollama) ──────────────────────────────────────────────

# Phase 57: Local 4B for fast tasks (System 1+2), local 9B for strategic (System 3).
# The fine-tuned ariaska-cybersec3 retains cybersec vocabulary for structured output.
# ariaska-strategic-9b is trained specifically for multi-step attack planning.
_LOCAL_DEFAULT = "ariaska-cybersec4"
FAST_MODEL = os.getenv("ARIASKA_FAST_MODEL", _LOCAL_DEFAULT)
MEDIUM_MODEL = os.getenv("ARIASKA_MEDIUM_MODEL", _LOCAL_DEFAULT)
# REASONING_MODEL: used as System 3 fallback when neither strategic-9b nor cloud is available
REASONING_MODEL = os.getenv("ARIASKA_REASONING_MODEL", _LOCAL_DEFAULT)
# STRATEGIC_MODEL: local 9B fine-tuned for System 3 (replaces cloud when available)
STRATEGIC_MODEL = os.getenv("ARIASKA_STRATEGIC_MODEL", "ariaska-strategic-9b")

# Runtime availability flags — set by GPTManager during init
_cloud_available: bool = False
_local_strategic_available: bool = False


def set_cloud_available(available: bool) -> None:
    """Called by GPTManager when cloud provider is detected."""
    global _cloud_available
    _cloud_available = available
    if available:
        logger.info("[ROUTER] Cloud reasoning provider available — System 3 cloud fallback ready")


def set_local_strategic_available(available: bool, model_name: str = "") -> None:
    """Called by GPTManager when local strategic 9B is detected in Ollama."""
    global _local_strategic_available, STRATEGIC_MODEL
    _local_strategic_available = available
    if model_name:
        STRATEGIC_MODEL = model_name
    if available:
        logger.info(f"[ROUTER] Local strategic model available — System 3 routes to {STRATEGIC_MODEL}")


def classify_tier(model: str) -> str:
    """Classify a model name into a budget tier."""
    if model in ("cloud", "cloud-reasoning"):
        return "cloud"
    return "local"


def _task_to_model(task_type: Optional[str]) -> tuple[str, str, str]:
    """Map task_type to (model_name, system_label, provider).

    Returns the optimal model, a human-readable label, and provider name.

    System 3 priority chain (P58: cloud removed):
      1. Local strategic 9B (fine-tuned, zero latency, no API limits)
      2. Local 4B fallback (always available, no rate limits)
    """
    if task_type in _SYSTEM1_TASKS:
        return FAST_MODEL, "System1-fast", "local"

    if task_type in _SYSTEM3_TASKS:
        if _local_strategic_available:
            return STRATEGIC_MODEL, "System3-local-strategic", "local-strategic"
        return REASONING_MODEL, "System3-local", "local"

    if task_type in _SYSTEM2_TASKS:
        return MEDIUM_MODEL, "System2-balanced", "local"

    # Default: local for unknown task types
    return MEDIUM_MODEL, "System2-default", "local"


class ModelRouter:
    """
    Routes model requests to local or cloud models based on task complexity.

    Hybrid strategy (Phase 56):
      - System 1 (local 4B): classification, diversify, parsing — fast reflexive
      - System 2 (local 4B): recon, defensive, general — balanced
      - System 3 (local 9B/4B): strategic, analysis, postmortem — deep reasoning

    Falls back to local for System 3 if no cloud API key is configured.
    """

    def __init__(
        self,
        local_available: bool = False,
        local_model_name: str = "",
        offload_nano: bool = True,
        offload_mini: bool = True,
        offload_all: bool = True,
    ):
        self._local_available = local_available
        self._local_model_name = local_model_name or FAST_MODEL
        self._offload_nano = offload_nano
        self._offload_mini = offload_mini
        self._offload_all = offload_all
        self._routed_local = 0
        self._routed_cloud = 0
        self._routed_by_system: dict[str, int] = {"s1": 0, "s2": 0, "s3": 0}

    @classmethod
    def from_flags(cls) -> "ModelRouter":
        """Create router from feature flags and runtime state."""
        try:
            from core.feature_flags import get_feature_flags
            ff = get_feature_flags()
            local_llm = getattr(ff, "local_llm", False)
            offload_nano = getattr(ff, "local_llm_offload_nano", True)
            offload_mini = getattr(ff, "local_llm_offload_mini", True)
            offload_all = getattr(ff, "local_llm_offload_all", False)
        except Exception:
            _truthy = ("1", "true", "yes", "on")
            local_llm = os.getenv("FF_LOCAL_LLM", "0").lower() in _truthy
            offload_nano = True
            offload_mini = True
            offload_all = os.getenv("FF_LOCAL_LLM_OFFLOAD_ALL", "0").lower() in _truthy

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
            offload_all=offload_all,
        )

    @classmethod
    def for_full_local(cls) -> "ModelRouter":
        """Create a router that sends ALL tiers to local models.

        Convenience factory for fully offline operation.
        """
        router = cls.from_flags()
        router._offload_all = True
        return router

    def route(self, model: str, task_type: Optional[str] = None) -> RoutingDecision:
        """
        Route a model request to the optimal provider by task complexity.

        Args:
            model: The requested model name (e.g. "local-llm")
            task_type: Task type for cognitive-tier routing

        Returns:
            RoutingDecision with provider, model, and tier.
        """
        # If no LLM is available at all, error
        if not self._local_available and not _cloud_available and not _local_strategic_available:
            return RoutingDecision(
                provider="local", model=model, tier="local",
                reason="no LLM available — check Ollama or set cloud API key"
            )

        # Task-aware hybrid routing
        target_model, system_label, provider = _task_to_model(task_type)

        # Fallback chains when preferred provider is unavailable:
        # P58: local-strategic → local 4B (cloud removed from chain)
        if provider == "local-strategic" and not _local_strategic_available:
            if self._local_available:
                target_model = REASONING_MODEL
                system_label = f"{system_label}→local-fallback"
                provider = "local"

        # local → cloud fallback
        if provider == "local" and not self._local_available and _cloud_available:
            target_model = "cloud-reasoning"
            system_label = f"{system_label}→cloud-fallback"
            provider = "cloud"

        # Normalize provider for stats and downstream: local-strategic uses local Ollama
        effective_provider = "local" if provider == "local-strategic" else provider

        # Track stats
        if effective_provider == "cloud":
            self._routed_cloud += 1
        else:
            self._routed_local += 1

        if system_label.startswith("System1"):
            self._routed_by_system["s1"] += 1
        elif system_label.startswith("System3"):
            self._routed_by_system["s3"] += 1
        else:
            self._routed_by_system["s2"] += 1

        return RoutingDecision(
            provider=effective_provider, model=target_model, tier=provider,
            reason=f"{task_type or 'unknown'} → {system_label} ({target_model})"
        )

    def get_stats(self) -> dict:
        """Get routing statistics."""
        total = self._routed_local + self._routed_cloud
        return {
            "local_available": self._local_available,
            "cloud_available": _cloud_available,
            "local_strategic_available": _local_strategic_available,
            "local_model": self._local_model_name,
            "strategic_model": STRATEGIC_MODEL if _local_strategic_available else None,
            "routed_local": self._routed_local,
            "routed_cloud": self._routed_cloud,
            "cloud_pct": round(self._routed_cloud / max(1, total) * 100, 1),
            "system_routing": self._routed_by_system,
        }

    def set_local_available(self, available: bool, model_name: str = "") -> None:
        """Update local availability at runtime."""
        self._local_available = available
        if model_name:
            self._local_model_name = model_name
