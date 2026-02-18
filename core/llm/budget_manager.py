#!/usr/bin/env python3
"""
core/llm/budget_manager.py — Phase 17: BudgetManagerV2 with Dynamic Scaling

Per-episode and per-model-tier budget management with ROI tags and
adaptive cost control.

Phase 17 Dynamic Budget:
  Episode budget scales from $0.50 to $3.33 based on learning maturity.
  Early episodes: full $3.33 budget (999K tokens) — heavy GPT steering.
  As success_rate rises and skill library grows: scales down to minimum
  $0.50 budget (~146K tokens) — agent knows what to do, mentor is luxury.

  Formula: budget_scale = max(MIN_SCALE, 1.0 - maturity_signal)
  Where maturity_signal = 0.4 * avg_success_rate + 0.3 * skill_coverage
                        + 0.2 * discovery_efficiency + 0.1 * (1 - stagnation_rate)

Model tiers:
  Tier 1 (codex): gpt-5.2-codex — architecture, complex reasoning (highest cost)
  Tier 2 (full):  gpt-5.2 — verification, validation
  Tier 3 (mini):  gpt-5.2-mini / gpt-5-mini — structured extraction
  Tier 4 (nano):  gpt-5.2-nano / gpt-5-nano — classification only

Feature-flag gated: FF_BUDGET_MANAGER_V2.

Author: Phase 15.0 + Phase 17 Dynamic Budget
"""

from __future__ import annotations

import logging
import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("ariaska.llm.budget_manager")

# ── Constants ───────────────────────────────────────────────────────────────

# Phase 36: Hard caps — +23% GPT uplift for live HTB engagements.
_TOTAL_BUDGET = 999_000   # Phase 36: +23% from 811K — ~$3.33/ep ceiling for codex-primary routing
_MIN_BUDGET = 499_500     # Phase 36: 50% floor
_MIN_SCALE = _MIN_BUDGET / _TOTAL_BUDGET  # 0.50

# Phase 36: Tier budgets — codex-primary routing gets larger share.
#   codex *= 1.50  → 199_800  (all reasoning: tactical, strategic, postmortem, analysis)
#   full  *= 1.20  → 199_800  (parsing, interpretation, verification)
#   mini  *= 1.10  → 299_700  (playbook selection, structured extraction)
#   nano  *= 1.10  → 299_700  (classification, micro-chain Stages 1+3, reserve)
#   Sum = 999,000 ✓
_TIER_BUDGETS: Dict[str, int] = {
    "codex":  199_800,  # 20.0% — all reasoning (tactical+strategic+postmortem)
    "full":   199_800,  # 20.0% — parsing, interpretation, verification
    "mini":   299_700,  # 30.0% — playbook selection, structured extraction
    "nano":   299_700,  # 30.0% — classification, micro-chain, reserve
}

# Phase 33.2: Dynamic Burst Pool
_BURST_POOL_RATIO = 0.12     # 12% of max budget as burst reserve
_BURST_STEP_CAP_RATIO = 0.03 # 3% of max budget per-step burst limit
_BURST_COOLDOWN_STEPS = 5    # Minimum steps between bursts
_BURST_TIERS = frozenset({"mini", "codex"})  # Only mini+codex get bursts

# Model → tier mapping
_MODEL_TIER: Dict[str, str] = {
    "gpt-5.2-codex": "codex",
    "gpt-5.2": "full",
    "gpt-5-mini": "mini",
    "gpt-5.2-mini": "mini",
    "gpt-5-nano": "nano",
    "gpt-5.2-nano": "nano",
    # Phase 23: no gpt-4 models — all routing is gpt-5.x
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

# P36.1: Learning-acceleration ROI tags that get doubled budget throughput.
# These are the "learning support" categories: value shaping, reward
# decomposition, distillation, phase scoring, parser feedback.
LEARNING_ACCEL_ROI_TAGS: Set[str] = {
    "value_target_shaping",
    "advantage_sanity",
    "reward_decomposition",
    "distillation_quality",
    "phase_decision_scoring",
    "parser_feedback_loop",
    "micro_chain_scoring",
    "micro_chain_generation",
    "micro_chain_classification",
    "tactical_advice",
    "reward_shaping",
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
        self._max_budget = _TOTAL_BUDGET
        self._total_budget = total_budget
        self._tier_budgets = tier_budgets or dict(_TIER_BUDGETS)
        self._tier_ratios = {k: v / self._max_budget for k, v in self._tier_budgets.items()}
        self._lock = threading.Lock()
        self._allocations: Dict[str, BudgetAllocation] = {}
        self._roi_counters: Dict[str, ROICounter] = {}
        self._total_used = 0
        self._total_denied = 0
        self._episode_id = ""
        self._budget_scale = 1.0  # Phase 17: current dynamic scale factor
        self._maturity_signal = 0.0  # Phase 17: learning maturity [0,1]
        # Phase 32: per-tier call counters (requests, not just tokens)
        self._tier_calls: Dict[str, int] = {}
        # Phase 33.2: Burst pool state
        self._burst_pool = int(_TOTAL_BUDGET * _BURST_POOL_RATIO)  # 79,200
        self._burst_step_cap = int(_TOTAL_BUDGET * _BURST_STEP_CAP_RATIO)  # 19,800
        self._burst_used = 0
        self._burst_last_step = -_BURST_COOLDOWN_STEPS  # allow first burst immediately
        self._burst_count = 0
        self._init_allocations()
        # P36.1: Learning acceleration boost factor (from governor)
        self._learn_boost_factor: float = 1.0

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
            self._tier_calls = {}
            # Phase 33.2: Reset burst pool
            self._burst_pool = int(self._max_budget * _BURST_POOL_RATIO)
            self._burst_used = 0
            self._burst_last_step = -_BURST_COOLDOWN_STEPS
            self._burst_count = 0
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

    def is_learning_accel_tag(self, roi_tag: str) -> bool:
        """P36.1: Check if ROI tag qualifies for doubled learning-accel budget."""
        return roi_tag in LEARNING_ACCEL_ROI_TAGS

    def get_effective_tokens(self, estimated_tokens: int, roi_tag: str) -> int:
        """P36.1: Apply learning boost to token budget for accel tags.

        Learning-acceleration ROI tags get `_learn_boost_factor` applied
        (default 1.0, max 2.0). This means the budget check is more
        lenient for these calls — effectively doubling their throughput.

        Args:
            estimated_tokens: Base token estimate for the call.
            roi_tag: ROI tag for this call.

        Returns:
            Effective token count (reduced for budget check, so more
            calls are allowed within the same budget).
        """
        if roi_tag in LEARNING_ACCEL_ROI_TAGS and self._learn_boost_factor > 1.0:
            # Lower the effective cost so more calls fit in budget
            return max(1, int(estimated_tokens / self._learn_boost_factor))
        return estimated_tokens

    def set_learn_boost_factor(self, factor: float) -> None:
        """P36.1: Set the governor's learning boost factor [0.5, 2.0]."""
        self._learn_boost_factor = max(0.5, min(2.0, factor))

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

            # Phase 32: per-tier call counter
            self._tier_calls[tier] = self._tier_calls.get(tier, 0) + 1

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

    # ── Phase 17: Dynamic Budget Scaling ────────────────────────────────────

    def compute_dynamic_budget(
        self,
        avg_success_rate: float = 0.0,
        skill_count: int = 0,
        max_skills: int = 50,
        discovery_efficiency: float = 0.0,
        stagnation_rate: float = 0.0,
        episode: int = 0,
    ) -> float:
        """
        Compute dynamic budget scale factor based on learning maturity.

        Returns scale factor [_MIN_SCALE, 1.0] where:
          1.0   = full budget ($3.00/ep, 877K tokens) — learning phase
          _MIN_SCALE = minimum budget ($0.50/ep, 146K tokens) — mature agent

        The maturity signal is a weighted combination of:
          40% — average success_rate across agents (0-1)
          30% — skill library coverage (unique SkillCards / max_skills)
          20% — discovery_efficiency (unique discoveries per GPT call)
          10% — inverse stagnation (1 - stagnation_rate)

        First 5 episodes always get full budget (learning warmup).

        Args:
            avg_success_rate: Mean success rate across all agents [0,1]
            skill_count: Number of SkillCards in the library
            max_skills: Normalization constant for skill coverage
            discovery_efficiency: Discoveries per GPT call this episode [0,1]
            stagnation_rate: Fraction of steps that were stagnant [0,1]
            episode: Current episode number

        Returns:
            Scale factor [_MIN_SCALE, 1.0]
        """
        # Warmup: full budget for first 5 episodes
        if episode < 5:
            self._budget_scale = 1.0
            self._maturity_signal = 0.0
            return 1.0

        # Compute maturity signal
        skill_coverage = min(1.0, skill_count / max(1, max_skills))
        maturity = (
            0.40 * min(1.0, avg_success_rate)
            + 0.30 * skill_coverage
            + 0.20 * min(1.0, discovery_efficiency)
            + 0.10 * max(0.0, 1.0 - stagnation_rate)
        )
        self._maturity_signal = maturity

        # Scale: inversely proportional to maturity
        # maturity=0 → scale=1.0 (full budget)
        # maturity=1 → scale=_MIN_SCALE (minimum budget)
        scale = max(_MIN_SCALE, 1.0 - maturity * (1.0 - _MIN_SCALE))
        self._budget_scale = scale

        logger.info(
            f"[DYNAMIC-BUDGET] maturity={maturity:.3f} → scale={scale:.3f} "
            f"(${scale * 3.34:.2f}/ep, {int(scale * _TOTAL_BUDGET):,} tokens) "
            f"[success={avg_success_rate:.2f}, skills={skill_count}, "
            f"disc_eff={discovery_efficiency:.2f}, stag={stagnation_rate:.2f}]"
        )
        return scale

    def apply_dynamic_scale(self, scale: Optional[float] = None) -> None:
        """
        Apply dynamic scale factor to per-tier budgets.

        Called at episode start after compute_dynamic_budget().
        Rescales all tier budgets proportionally.

        Args:
            scale: Scale factor [_MIN_SCALE, 1.0]. If None, uses last computed.
        """
        if scale is not None:
            self._budget_scale = scale

        with self._lock:
            self._total_budget = int(self._max_budget * self._budget_scale)
            for tier, ratio in self._tier_ratios.items():
                new_budget = int(self._max_budget * self._budget_scale * ratio)
                if tier in self._allocations:
                    self._allocations[tier].budget = new_budget
                self._tier_budgets[tier] = new_budget

    @property
    def budget_scale(self) -> float:
        """Current dynamic budget scale [_MIN_SCALE, 1.0]."""
        return self._budget_scale

    @property
    def maturity_signal(self) -> float:
        """Current learning maturity [0, 1]."""
        return self._maturity_signal

    @property
    def estimated_cost_usd(self) -> float:
        """Estimated per-episode cost in USD at current scale."""
        return self._budget_scale * 2.20  # Phase 33: 660k → ~$2.20/ep at scale 1.0

    # ── Phase 33.2: Dynamic Burst Pool ──────────────────────────────

    def burst_available(self, step: int, tier: str) -> bool:
        """
        Check if a burst allocation is available.

        Burst is allowed only for mini/codex tiers, when:
          - Pool has remaining tokens
          - Cooldown has elapsed (5 steps since last burst)
          - Tier is in _BURST_TIERS
        """
        if tier not in _BURST_TIERS:
            return False
        with self._lock:
            remaining = self._burst_pool - self._burst_used
            if remaining <= 0:
                return False
            if step - self._burst_last_step < _BURST_COOLDOWN_STEPS:
                return False
            return True

    def burst_request(
        self,
        step: int,
        tier: str,
        tokens_needed: int,
        trigger: str,
    ) -> int:
        """
        Request burst tokens for a step.

        Args:
            step: Current step number in the episode.
            tier: Model tier requesting burst (must be in _BURST_TIERS).
            tokens_needed: Tokens requested.
            trigger: Reason for burst (semantic_stall, strategy_pivot,
                     mc_escalation_high, low_confidence_streak).

        Returns:
            Number of burst tokens granted (0 if denied).
        """
        if tier not in _BURST_TIERS:
            return 0

        with self._lock:
            remaining = self._burst_pool - self._burst_used
            if remaining <= 0:
                return 0
            if step - self._burst_last_step < _BURST_COOLDOWN_STEPS:
                return 0
            granted = min(tokens_needed, self._burst_step_cap, remaining)
            self._burst_used += granted
            self._burst_last_step = step
            self._burst_count += 1
            logger.debug(
                f"[BURST] +{granted} tokens | tier={tier} trigger={trigger} "
                f"step={step} pool_remaining={remaining - granted}"
            )
            return granted

    def get_burst_stats(self) -> Dict[str, Any]:
        """Return burst pool statistics."""
        with self._lock:
            return {
                "burst_pool_total": self._burst_pool,
                "burst_used": self._burst_used,
                "burst_remaining": self._burst_pool - self._burst_used,
                "burst_count": self._burst_count,
                "burst_last_step": self._burst_last_step,
            }

    # ── Phase 32: Episode Cost Summary ──────────────────────────────

    def get_episode_cost_summary(self) -> Dict[str, Any]:
        """
        Return a per-episode cost summary for trace/dashboard.

        Includes calls_by_tier, tokens_by_tier, cost_by_tier, and
        codex_escalation_rate.
        """
        # Cost rates (USD per token, blended input+output average)
        _COST_PER_TOKEN: Dict[str, float] = {
            "codex": 0.00001000,   # $10.00 / 1M tokens
            "full":  0.00001000,   # $10.00 / 1M tokens
            "mini":  0.00000060,   # $0.60  / 1M tokens
            "nano":  0.00000010,   # $0.10  / 1M tokens
        }
        with self._lock:
            calls_by_tier: Dict[str, int] = dict(self._tier_calls)
            tokens_by_tier: Dict[str, int] = {
                k: v.used for k, v in self._allocations.items()
            }
            cost_by_tier: Dict[str, float] = {}
            for tier, tok in tokens_by_tier.items():
                rate = _COST_PER_TOKEN.get(tier, 0.000001)
                cost_by_tier[tier] = round(tok * rate, 6)

            total_calls = sum(calls_by_tier.values()) or 1
            codex_calls = calls_by_tier.get("codex", 0)

            return {
                "calls_by_tier": calls_by_tier,
                "tokens_by_tier": tokens_by_tier,
                "cost_by_tier": cost_by_tier,
                "total_cost_usd": round(sum(cost_by_tier.values()), 6),
                "total_calls": sum(calls_by_tier.values()),
                "total_tokens": self._total_used,
                "budget_remaining": self._total_budget - self._total_used,
                "codex_escalation_rate": round(codex_calls / total_calls, 4),
                "call_distribution": {
                    tier: round(calls_by_tier.get(tier, 0) / total_calls, 4)
                    for tier in ("nano", "mini", "full", "codex")
                },
            }
