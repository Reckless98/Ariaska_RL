#!/usr/bin/env python3
"""
core/llm/model_router.py — Phase 43 / 44 / 54: Intelligent Model Router

Routes LLM requests to the optimal local model based on task complexity:
  - Fast tasks (classification, diversify, parsing) → 4B model (System 1: fast, reflexive)
  - Medium tasks (recon, defensive, general) → 4B model (System 2: same physical model)
  - Heavy tasks (tactical, analysis, strategic, postmortem) → 9B model (System 3: deep reasoning)

Phase 55 — Qwen 3.5 Uncensored tri-model local routing:
  All tiers route to local Ollama models. Task type determines which model
  handles the request, matching cognitive load to model capability:
    System 1 (jaahas/qwen3.5-uncensored:4b): ~12-15 tok/s CPU — MicroChain, classification, parsing
    System 2 (jaahas/qwen3.5-uncensored:4b): ~12-15 tok/s CPU — recon, defensive, general
    System 3 (jaahas/qwen3.5-uncensored:9b): ~5-8 tok/s CPU — tactical, strategic, postmortem

Feature-flag gated: FF_LOCAL_LLM + FF_LOCAL_LLM_OFFLOAD_ALL.

Author: Phase 43/44/54 — Full Local LLM Operation
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
    provider: str       # "local"
    model: str          # Local model name for Ollama
    tier: str           # Budget tier: "local"
    reason: str = ""    # Why this routing was chosen


# ── Task-type to cognitive tier mapping ─────────────────────────────────────

# System 1: Fast reflexive tasks → 4B model (low latency, simple tasks)
_SYSTEM1_TASKS = frozenset({
    "classification", "diversify", "parsing", "playbook",
    "command_selection", "output_parse",
})

# System 2: Medium reasoning → 4B model (same model as S1, different task pool)
# On CPU, favour 4b for speed. Tactical/analysis moved here from S3.
_SYSTEM2_TASKS = frozenset({
    "reconnaissance", "defensive", "general", "embedding",
    "tactical", "analysis", "reasoning", "learning", "reflection",
})

# System 3: Deep reasoning → 9B model (only truly critical slow tasks)
_SYSTEM3_TASKS = frozenset({
    "strategic", "postmortem",
})

# ── Local model names (Ollama) ──────────────────────────────────────────────

# Post-fine-tune: single model for all tiers — eliminates Ollama cold-load penalty.
# Override via env vars if you want to test a different model on a specific tier.
_DEFAULT_MODEL = "ariaska-cybersec"
FAST_MODEL = os.getenv("ARIASKA_FAST_MODEL", _DEFAULT_MODEL)
MEDIUM_MODEL = os.getenv("ARIASKA_MEDIUM_MODEL", _DEFAULT_MODEL)
REASONING_MODEL = os.getenv("ARIASKA_REASONING_MODEL", _DEFAULT_MODEL)


def classify_tier(model: str) -> str:
    """Classify a model name into a budget tier. All models are local."""
    return "local"


def _task_to_model(task_type: Optional[str]) -> tuple[str, str]:
    """Map task_type to (model_name, system_label).
    
    Returns the optimal local model and a human-readable label.
    """
    if task_type in _SYSTEM1_TASKS:
        return FAST_MODEL, "System1-fast"
    if task_type in _SYSTEM3_TASKS:
        return REASONING_MODEL, "System3-reasoning"
    if task_type in _SYSTEM2_TASKS:
        return MEDIUM_MODEL, "System2-balanced"
    # Default: medium model for unknown task types
    return MEDIUM_MODEL, "System2-default"


class ModelRouter:
    """
    Routes model requests to local Ollama models based on task complexity.

    Tri-model strategy:
      - System 1 (4B): classification, diversify, parsing — fast reflexive
      - System 2 (4B): recon, defensive, general — balanced
      - System 3 (9B): strategic, postmortem — deep reasoning

    All inference is local. No external API fallback.
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

        Convenience factory for fully offline operation (Phase 44/54).
        """
        router = cls.from_flags()
        router._offload_all = True
        return router
    
    def route(self, model: str, task_type: Optional[str] = None) -> RoutingDecision:
        """
        Route a model request to the optimal local model by task complexity.
        
        Args:
            model: The requested model name (e.g. "local-llm")
            task_type: Task type for cognitive-tier routing
            
        Returns:
            RoutingDecision with provider, model, and tier.
        """
        # If local LLM is not available, return error decision
        if not self._local_available:
            return RoutingDecision(
                provider="local", model=model, tier="local",
                reason="local LLM unavailable — check Ollama"
            )

        # Task-aware tri-model local routing
        target_model, system_label = _task_to_model(task_type)
        self._routed_local += 1
        # Track per-system stats
        if system_label.startswith("System1"):
            self._routed_by_system["s1"] += 1
        elif system_label.startswith("System3"):
            self._routed_by_system["s3"] += 1
        else:
            self._routed_by_system["s2"] += 1
        return RoutingDecision(
            provider="local", model=target_model, tier="local",
            reason=f"{task_type or 'unknown'} → {system_label} ({target_model})"
        )
    
    def get_stats(self) -> dict:
        """Get routing statistics."""
        return {
            "local_available": self._local_available,
            "local_model": self._local_model_name,
            "routed_local": self._routed_local,
            "local_pct": 100.0,
            "system_routing": self._routed_by_system,
        }
    
    def set_local_available(self, available: bool, model_name: str = "") -> None:
        """Update local availability at runtime."""
        self._local_available = available
        if model_name:
            self._local_model_name = model_name
