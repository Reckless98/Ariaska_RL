#!/usr/bin/env python3
"""
core/training/adaptive_budget.py — Phase 11.0: Adaptive Budget Controller

Manages per-episode token and LLM call budgets with adaptive pacing.
Replaces static budget caps with intelligent spend-rate tracking that
prevents budget exhaustion in early steps while ensuring enough budget
remains for critical late-episode phases (exploitation, exfiltration).

Key concepts:
  - Budget pressure: 0.0 (relaxed) → 1.0 (exhausted)
  - Spend rate: rolling average of calls/step
  - Phase weighting: exploitation/exfiltration phases get higher budget allocation
  - Reserve: 20% of budget held back for late-game phases

Author: Filip Volf — Phase 11.0
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger("ariaska.adaptive_budget")


# Phase importance weights — higher = more budget allocated
PHASE_BUDGET_WEIGHTS = {
    "RECON": 0.6,
    "ENUMERATION": 0.7,
    "EXPLOITATION": 1.0,
    "PRIVILEGE_ESCALATION": 1.0,
    "LATERAL_MOVEMENT": 0.8,
    "POST_EXPLOITATION": 0.9,
    "EXFILTRATION": 1.0,
    "CLOSEOUT": 0.3,
}


@dataclass
class BudgetConfig:
    """Configuration for adaptive budget controller."""
    # LLM mentor call budget
    mentor_budget_total: int = 30  # max mentor calls per episode
    mentor_reserve_pct: float = 0.20  # Hold 20% for late phases

    # Venice parsing budget
    venice_budget_total: int = 15

    # Token budget
    token_budget_total: int = 50_000  # per episode
    token_reserve_pct: float = 0.15

    # Parse LLM budget
    parse_llm_budget_total: int = 20

    # Spend rate tracking
    spend_rate_window: int = 10  # Rolling window for rate calculation

    # Throttle thresholds — Phase 13.0: Relaxed to allow more GPT reasoning per episode
    # Higher thresholds = agents can spend more budget before throttling kicks in
    soft_throttle_pressure: float = 0.70   # Phase 13.0: 0.60→0.70 — delayed soft throttle
    hard_throttle_pressure: float = 0.90   # Phase 13.0: 0.85→0.90 — delayed hard throttle
    emergency_cutoff_pressure: float = 0.97  # Phase 13.0: 0.95→0.97 — near-exhaustion only


class AdaptiveBudgetController:
    """
    Manages LLM call and token budgets with adaptive pacing.

    Usage:
        budget = AdaptiveBudgetController(config=BudgetConfig())
        budget.reset_episode(max_steps=40)

        # Each step, check before making LLM calls
        if budget.can_call_mentor(phase="EXPLOITATION"):
            # ... make mentor call ...
            budget.record_mentor_call(tokens_used=500)

        # Get current pressure for dashboard
        pressure = budget.get_pressure()
        snapshot = budget.get_snapshot()
    """

    def __init__(self, config: Optional[BudgetConfig] = None):
        self.config = config or BudgetConfig()
        self._mentor_calls: int = 0
        self._venice_calls: int = 0
        self._parse_llm_calls: int = 0
        self._tokens_used: int = 0
        self._max_steps: int = 40
        self._current_step: int = 0

        # Rolling spend rate tracking
        self._mentor_rate_window: deque = deque(maxlen=self.config.spend_rate_window)
        self._token_rate_window: deque = deque(maxlen=self.config.spend_rate_window)

    def reset_episode(self, max_steps: int = 40) -> None:
        """Reset all counters for a new episode."""
        self._mentor_calls = 0
        self._venice_calls = 0
        self._parse_llm_calls = 0
        self._tokens_used = 0
        self._max_steps = max_steps
        self._current_step = 0
        self._mentor_rate_window.clear()
        self._token_rate_window.clear()

    def step_tick(self, step: int) -> None:
        """Called at the start of each step to update tracking."""
        self._current_step = step

    def can_call_mentor(self, phase: str = "RECON") -> bool:
        """Check if a mentor call is allowed given current budget state."""
        if self._mentor_calls >= self.config.mentor_budget_total:
            return False

        pressure = self.get_pressure()
        phase_weight = PHASE_BUDGET_WEIGHTS.get(phase.upper(), 0.5)

        # Always allow if pressure is low
        if pressure < self.config.soft_throttle_pressure:
            return True

        # In soft throttle range, allow only for high-importance phases
        if pressure < self.config.hard_throttle_pressure:
            return phase_weight >= 0.8

        # In hard throttle, only exploitation/exfiltration
        if pressure < self.config.emergency_cutoff_pressure:
            return phase_weight >= 1.0

        # Emergency: no calls
        return False

    def can_call_venice(self) -> bool:
        """Check if a Venice parsing call is allowed."""
        return self._venice_calls < self.config.venice_budget_total

    def can_call_parse_llm(self) -> bool:
        """Check if a parse LLM call is allowed."""
        return self._parse_llm_calls < self.config.parse_llm_budget_total

    def record_mentor_call(self, tokens_used: int = 0) -> None:
        """Record a mentor LLM call."""
        self._mentor_calls += 1
        self._tokens_used += tokens_used
        self._mentor_rate_window.append(1)

    def record_venice_call(self, tokens_used: int = 0) -> None:
        """Record a Venice call."""
        self._venice_calls += 1
        self._tokens_used += tokens_used

    def record_parse_llm_call(self, tokens_used: int = 0) -> None:
        """Record a parse LLM call."""
        self._parse_llm_calls += 1
        self._tokens_used += tokens_used

    def record_no_call(self) -> None:
        """Record a step where no LLM call was made (for rate tracking)."""
        self._mentor_rate_window.append(0)

    def get_pressure(self) -> float:
        """Calculate current budget pressure (0.0 = relaxed, 1.0 = exhausted).

        Combines:
          - Call fraction: mentor_calls / mentor_budget
          - Token fraction: tokens_used / token_budget
          - Step fraction: current_step / max_steps (time pressure)
        """
        call_frac = self._mentor_calls / max(self.config.mentor_budget_total, 1)
        token_frac = self._tokens_used / max(self.config.token_budget_total, 1)
        step_frac = self._current_step / max(self._max_steps, 1)

        # Pressure is highest of resource fractions weighted against time
        # If we've used 80% of budget in 50% of time → pressure is high
        resource_pressure = max(call_frac, token_frac)

        # Compare resource spending against time progress
        # If resource_pressure > step_frac → we're spending too fast
        if step_frac > 0:
            pace_ratio = resource_pressure / step_frac
        else:
            pace_ratio = resource_pressure * 2

        # Normalize to 0-1 range
        pressure = min(1.0, max(0.0, resource_pressure * 0.6 + (pace_ratio - 1.0) * 0.4))

        # Absolute-reserve floor: if ≥90% of ANY resource is consumed,
        # ensure pressure is at least 0.8 regardless of pacing.
        # This prevents the "consistent overspend" blind spot where
        # 29/30 calls at step 39/40 produces pressure 0.59.
        if call_frac >= 0.9 or token_frac >= 0.9:
            pressure = max(pressure, 0.8)

        return pressure

    def get_spend_rate(self) -> float:
        """Get rolling average of mentor calls per step."""
        if not self._mentor_rate_window:
            return 0.0
        return sum(self._mentor_rate_window) / len(self._mentor_rate_window)

    def get_snapshot(self) -> Dict[str, Any]:
        """Get a snapshot dict compatible with BudgetSnapshot dataclass."""
        return {
            "mentor_budget_remaining": max(0, self.config.mentor_budget_total - self._mentor_calls),
            "mentor_budget_total": self.config.mentor_budget_total,
            "mentor_spend_rate": self.get_spend_rate(),
            "venice_budget_remaining": max(0, self.config.venice_budget_total - self._venice_calls),
            "venice_budget_total": self.config.venice_budget_total,
            "parse_llm_remaining": max(0, self.config.parse_llm_budget_total - self._parse_llm_calls),
            "token_budget_remaining": max(0, self.config.token_budget_total - self._tokens_used),
            "token_budget_total": self.config.token_budget_total,
            "budget_pressure": self.get_pressure(),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get end-of-episode summary."""
        return {
            "mentor_calls": self._mentor_calls,
            "mentor_budget": self.config.mentor_budget_total,
            "venice_calls": self._venice_calls,
            "venice_budget": self.config.venice_budget_total,
            "parse_llm_calls": self._parse_llm_calls,
            "parse_llm_budget": self.config.parse_llm_budget_total,
            "tokens_used": self._tokens_used,
            "token_budget": self.config.token_budget_total,
            "final_pressure": self.get_pressure(),
            "avg_spend_rate": self.get_spend_rate(),
        }
