#!/usr/bin/env python3
"""
core/llm/budget_manager.py — Phase 15.0: BudgetManagerV2

Per-episode and per-model-tier budget management with ROI tags.
Every LLM call must provide an roi_tag and pass budget + cache gates.

Model tiers:
  Tier 1 (codex): gpt-5.2-codex — architecture, complex reasoning (highest cost)
  Tier 2 (full):  gpt-5.2 — verification, validation
  Tier 3 (mini):  gpt-5.2-mini / gpt-5-mini — structured extraction
  Tier 4 (nano):  gpt-5.2-nano / gpt-5-nano — classification only

Budget: 1.5x Phase 14 baseline = 877,500 tokens/episode total.

Feature-flag gated: FF_BUDGET_MANAGER_V2.

Author: Phase 15.0 Contract
"""

from __future__ import annotations

import logging
import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("ariaska.llm.budget_manager")

# ── Constants ───────────────────────────────────────────────────────────────

_TOTAL_BUDGET = 877_500  # 1.5x of 585K

# Per-tier budgets (must sum to _TOTAL_BUDGET)
_TIER_BUDGETS: Dict[str, int] = {
    "codex": 263_250,    # ~30% — Orion plans, mentor reasoning, postmortem synthesis
    "full": 175_500,     # ~20% — plan verification, invariant checks, JSON validity
    "mini": 263_250,     # ~30% — hypothesis ranking, verification, lesson compression
    "nano": 175_500,     # ~20% — micro classification, cache-key summaries
}

# Model → tier mapping
_MODEL_TIER: Dict[str, str] = {
    "gpt-5.2-codex": "codex",
    "gpt-5.2": "full",
    "gpt-5-mini": "mini",
    "gpt-5.2-mini": "mini",
    "gpt-5-nano": "nano",
    "gpt-5.2-nano": "nano",
    "gpt-4o-mini": "mini",
    "gpt-4o": "full",
}

# Valid ROI tags
VALID_ROI_TAGS: Set[str] = {
    "improves_hypothesis_accuracy",
    "reduces_steps_to_foothold",
    "reduces_steps_to_root",
    "reduces_mentor_reliance",
    "increases_chain_coherence",
    "classification",
    "verification",
    "consolidation",
    "strategy_plan",
    "postmortem",
    "mentor_teacher",
    "reflex_microtask",
    "parsing",
    "tactical_advice",
    "reward_shaping",
}

# ROI tags that allow cross-episode caching
STABLE_ROI_TAGS: Set[str] = {
    "classification",
    "verification",
}


# ── Schemas ─────────────────────────────────────────────────────────────────

@dataclass
class BudgetAllocation:
    """Per-tier budget tracking."""
    tier: str = ""
    budget: int = 0
    used: int = 0
    denied: int = 0     # requests denied due to budget
    cached: int = 0     # requests served from cache

    @property
    def remaining(self) -> int:
        return max(0, self.budget - self.used)

    @property
    def utilization(self) -> float:
        return self.used / max(1, self.budget)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["remaining"] = self.remaining
        d["utilization"] = round(self.utilization, 4)
        return d


@dataclass
class BudgetDecision:
    """Result of a budget check."""
    allowed: bool = True
    tier: str = ""
    tokens_requested: int = 0
    tokens_remaining: int = 0
    roi_tag: str = ""
    reason: str = ""
    cache_hit: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ROICounter:
    """Track calls and outcomes per ROI tag."""
    tag: str = ""
    calls: int = 0
    tokens_used: int = 0
    cache_hits: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Budget Manager ──────────────────────────────────────────────────────────

class BudgetManagerV2:
    """
    Per-episode budget manager with model-tier caps and ROI tracking.

    Thread-safe. Resets per episode.
    """

    def __init__(
        self,
        total_budget: int = _TOTAL_BUDGET,
        tier_budgets: Optional[Dict[str, int]] = None,
    ) -> None:
        self._total_budget = total_budget
        self._tier_budgets = tier_budgets or dict(_TIER_BUDGETS)
        self._lock = threading.Lock()
        self._allocations: Dict[str, BudgetAllocation] = {}
        self._roi_counters: Dict[str, ROICounter] = {}
        self._total_used = 0
        self._total_denied = 0
        self._episode_id = ""
        self._init_allocations()

    def _init_allocations(self) -> None:
        """Initialize per-tier allocations."""
        for tier, budget in self._tier_budgets.items():
            self._allocations[tier] = BudgetAllocation(tier=tier, budget=budget)

    def reset_episode(self, episode_id: str = "") -> None:
        """Reset all budgets for a new episode."""
        with self._lock:
            self._episode_id = episode_id
            self._total_used = 0
            self._total_denied = 0
            self._init_allocations()
            self._roi_counters.clear()

    def check_budget(
        self,
        model: str,
        estimated_tokens: int,
        roi_tag: str,
    ) -> BudgetDecision:
        """
        Check if a request is within budget.

        Returns BudgetDecision indicating whether to proceed.
        """
        tier = _MODEL_TIER.get(model, "mini")

        with self._lock:
            alloc = self._allocations.get(tier)
            if alloc is None:
                return BudgetDecision(
                    allowed=False,
                    tier=tier,
                    tokens_requested=estimated_tokens,
                    roi_tag=roi_tag,
                    reason=f"unknown_tier_{tier}",
                )

            if estimated_tokens > alloc.remaining:
                alloc.denied += 1
                self._total_denied += 1
                return BudgetDecision(
                    allowed=False,
                    tier=tier,
                    tokens_requested=estimated_tokens,
                    tokens_remaining=alloc.remaining,
                    roi_tag=roi_tag,
                    reason="budget_exceeded",
                )

            return BudgetDecision(
                allowed=True,
                tier=tier,
                tokens_requested=estimated_tokens,
                tokens_remaining=alloc.remaining,
                roi_tag=roi_tag,
            )

    def record_spend(
        self,
        model: str,
        tokens_used: int,
        roi_tag: str,
        cache_hit: bool = False,
    ) -> None:
        """Record actual token spend after a call completes."""
        tier = _MODEL_TIER.get(model, "mini")

        with self._lock:
            alloc = self._allocations.get(tier)
            if alloc:
                if cache_hit:
                    alloc.cached += 1
                else:
                    alloc.used += tokens_used
                    self._total_used += tokens_used

            # ROI counter
            if roi_tag not in self._roi_counters:
                self._roi_counters[roi_tag] = ROICounter(tag=roi_tag)
            counter = self._roi_counters[roi_tag]
            counter.calls += 1
            counter.tokens_used += tokens_used if not cache_hit else 0
            if cache_hit:
                counter.cache_hits += 1

    def is_stable_roi(self, roi_tag: str) -> bool:
        """Check if an ROI tag allows cross-episode caching."""
        return roi_tag in STABLE_ROI_TAGS

    def get_tier_for_model(self, model: str) -> str:
        """Map model name to tier."""
        return _MODEL_TIER.get(model, "mini")

    def get_stats(self) -> Dict[str, Any]:
        """Return complete budget stats."""
        with self._lock:
            return {
                "episode_id": self._episode_id,
                "total_budget": self._total_budget,
                "total_used": self._total_used,
                "total_denied": self._total_denied,
                "utilization": round(
                    self._total_used / max(1, self._total_budget), 4
                ),
                "tiers": {
                    k: v.to_dict() for k, v in self._allocations.items()
                },
                "roi": {
                    k: v.to_dict() for k, v in self._roi_counters.items()
                },
            }

    def get_tier_stats(self) -> Dict[str, Dict[str, Any]]:
        """Return per-tier budget stats."""
        with self._lock:
            return {k: v.to_dict() for k, v in self._allocations.items()}

    def get_spend_per_tier(self) -> Dict[str, int]:
        """Return tokens used per tier."""
        with self._lock:
            return {k: v.used for k, v in self._allocations.items()}

    def get_roi_summary(self) -> Dict[str, Any]:
        """Return ROI metrics: per-tag calls, tokens, cache efficiency."""
        with self._lock:
            summary: Dict[str, Any] = {}
            for tag, counter in self._roi_counters.items():
                summary[tag] = {
                    "calls": counter.calls,
                    "tokens": counter.tokens_used,
                    "cache_hits": counter.cache_hits,
                    "cache_rate": round(
                        counter.cache_hits / max(1, counter.calls), 4
                    ),
                }
            return summary

    def get_budget_pressure(self) -> float:
        """Return overall budget pressure [0, 1]."""
        with self._lock:
            return self._total_used / max(1, self._total_budget)
