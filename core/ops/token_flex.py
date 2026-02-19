"""
core/ops/token_flex.py — Phase 38.4: Engagement-Progress-Aware Budget Flex

Adjusts the per-step and per-tier token budget allocation dynamically
based on engagement progress signals, complementing the episode-level
maturity scaling already in BudgetManagerV2.

Signals consumed:
  - Phase progress (kill chain position)
  - Discovery stagnation level
  - Flag capture state
  - Exploitation success rate
  - Time pressure (steps remaining vs target)
  - Shell availability

Output: a TokenFlexResult with a scale multiplier [0.50, 1.50] and
per-tier adjustment hints.  SmartCoach / SmartOrchestrator consult this
before making budget-gated LLM calls.

Does NOT modify BudgetManagerV2 internals — purely advisory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger("ariaska.ops.token_flex")

# ── Constants ───────────────────────────────────────────────────────────────

# Scale multiplier bounds — never let flex go below 0.50 or above 1.50
MIN_FLEX_SCALE: float = 0.50
MAX_FLEX_SCALE: float = 1.50

# Stagnation threshold: if stagnation_level >= this, boost budget
STAGNATION_BOOST_THRESHOLD: float = 0.40

# Phase-based base multipliers (engagement-progress-driven)
_PHASE_FLEX: Dict[str, float] = {
    "RECON":                 1.00,   # Normal — heavy LLM guidance expected
    "ENUMERATION":           0.95,   # Slightly less — patterns emerging
    "EXPLOITATION":          1.15,   # Boost — complex reasoning needed
    "PRIVILEGE_ESCALATION":  1.20,   # Boost — creative escalation paths
    "LATERAL_MOVEMENT":      1.10,   # Moderate boost
    "POST_EXPLOITATION":     0.90,   # Winding down
    "EXFILTRATION":          0.80,   # Less LLM needed
    "CLOSEOUT":              0.60,   # Minimal
}


@dataclass
class TokenFlexResult:
    """Engagement-aware budget flex recommendation."""
    scale: float = 1.0
    reason: str = "default"
    tier_hints: Dict[str, float] = field(default_factory=dict)
    signals: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scale": round(self.scale, 4),
            "reason": self.reason,
            "tier_hints": {k: round(v, 4) for k, v in self.tier_hints.items()},
            "signals": {k: round(v, 4) for k, v in self.signals.items()},
        }


class TokenFlexEngine:
    """
    Engagement-progress-aware token budget flexible scaler.

    Produces an advisory scale multiplier based on engagement
    progress signals. This is consumed by the decision pipeline
    (SmartCoach) to adjust willingness to spend tokens.

    Unlike BudgetManagerV2's maturity scaling (which reduces budget
    as the agent matures), TokenFlexEngine can INCREASE budget
    during critical engagement moments (e.g., exploitation phase,
    stagnation, or approaching time limits).

    Usage:
        engine = TokenFlexEngine(max_steps=500)
        result = engine.compute(
            phase="EXPLOITATION",
            step=200,
            stagnation_level=0.6,
            flags_captured=0,
            shells_obtained=1,
            exploit_success_rate=0.3,
        )
        effective_budget = base_budget * result.scale
    """

    def __init__(self, max_steps: int = 500) -> None:
        self._max_steps = max(1, max_steps)
        logger.debug("TokenFlexEngine initialised, max_steps=%d", max_steps)

    def compute(
        self,
        phase: str = "RECON",
        step: int = 0,
        stagnation_level: float = 0.0,
        flags_captured: int = 0,
        shells_obtained: int = 0,
        exploit_success_rate: float = 0.0,
        discovery_rate: float = 0.0,
    ) -> TokenFlexResult:
        """
        Compute engagement-aware flex scale.

        Args:
            phase: Current kill chain phase.
            step: Current step in engagement.
            stagnation_level: Stagnation [0.0, 1.0] from EngagementMetrics.
            flags_captured: Number of flags captured so far.
            shells_obtained: Number of shells obtained.
            exploit_success_rate: Success rate [0.0, 1.0].
            discovery_rate: Discoveries per step [0.0, 1.0+].

        Returns:
            TokenFlexResult with scale, reason, tier hints.
        """
        signals: Dict[str, float] = {}

        # 1. Phase-based baseline
        phase_upper = phase.upper()
        base = _PHASE_FLEX.get(phase_upper, 1.0)
        signals["phase_base"] = base

        # 2. Stagnation boost — if stuck, spend more on LLM
        stagnation_boost = 0.0
        if stagnation_level >= STAGNATION_BOOST_THRESHOLD:
            stagnation_boost = 0.15 * min(1.0, stagnation_level)
        signals["stagnation_boost"] = stagnation_boost

        # 3. Time pressure — boost budget in final 20% of engagement
        time_progress = step / max(1, self._max_steps)
        time_pressure = 0.0
        if time_progress >= 0.80 and flags_captured == 0:
            # Running out of time with no flags — urgency boost
            time_pressure = 0.20 * min(1.0, (time_progress - 0.80) / 0.20)
        signals["time_pressure"] = time_pressure

        # 4. Victory cooldown — if flags already captured, reduce budget
        flag_reduction = 0.0
        if flags_captured >= 2:
            flag_reduction = -0.20  # Both flags → save tokens
        elif flags_captured == 1:
            flag_reduction = -0.05  # One flag → slight reduction
        signals["flag_adjustment"] = flag_reduction

        # 5. Exploitation efficiency — low success rate needs more guidance
        exploit_boost = 0.0
        if phase_upper in ("EXPLOITATION", "PRIVILEGE_ESCALATION"):
            if exploit_success_rate < 0.15 and step > 20:
                exploit_boost = 0.10  # Struggling — spend more
        signals["exploit_boost"] = exploit_boost

        # 6. Discovery drought / surplus
        discovery_adj = 0.0
        if discovery_rate > 0.5:
            discovery_adj = -0.05  # Lots of discoveries, less LLM needed
        elif discovery_rate < 0.05 and step > 30:
            discovery_adj = 0.08  # Very few discoveries, boost
        signals["discovery_adj"] = discovery_adj

        # Combine
        raw_scale = (
            base
            + stagnation_boost
            + time_pressure
            + flag_reduction
            + exploit_boost
            + discovery_adj
        )
        clamped_scale = max(MIN_FLEX_SCALE, min(MAX_FLEX_SCALE, raw_scale))
        signals["raw_scale"] = raw_scale
        signals["clamped_scale"] = clamped_scale

        # Build reason string
        parts = []
        if stagnation_boost > 0:
            parts.append("stagnation")
        if time_pressure > 0:
            parts.append("time_pressure")
        if flag_reduction != 0:
            parts.append("flag_adj")
        if exploit_boost > 0:
            parts.append("exploit_struggle")
        if discovery_adj != 0:
            parts.append("discovery_adj")
        reason = f"phase={phase_upper}"
        if parts:
            reason += "+" + "+".join(parts)

        # Per-tier hints
        tier_hints = self._compute_tier_hints(
            phase_upper, stagnation_level, exploit_success_rate,
        )

        return TokenFlexResult(
            scale=clamped_scale,
            reason=reason,
            tier_hints=tier_hints,
            signals=signals,
        )

    def _compute_tier_hints(
        self,
        phase: str,
        stagnation_level: float,
        exploit_success_rate: float,
    ) -> Dict[str, float]:
        """
        Per-tier allocation hints.

        Returns relative multipliers for each tier (1.0 = normal).
        """
        hints: Dict[str, float] = {
            "codex": 1.0,
            "full": 1.0,
            "mini": 1.0,
            "nano": 1.0,
        }

        if phase in ("EXPLOITATION", "PRIVILEGE_ESCALATION"):
            # Exploitation needs more codex reasoning
            hints["codex"] = 1.30
            hints["nano"] = 0.80
        elif phase == "RECON":
            # Recon is more classification-heavy
            hints["nano"] = 1.20
            hints["codex"] = 0.80
        elif phase in ("EXFILTRATION", "CLOSEOUT"):
            # Winding down — reduce all
            hints["codex"] = 0.70
            hints["full"] = 0.80

        if stagnation_level >= 0.60:
            # When stagnated, escalate to bigger models
            hints["codex"] = min(1.50, hints["codex"] + 0.20)
            hints["mini"] = min(1.30, hints["mini"] + 0.10)

        if exploit_success_rate < 0.10 and phase == "EXPLOITATION":
            # Very low exploit success — need codex help
            hints["codex"] = min(1.50, hints["codex"] + 0.15)

        return hints

    def update_max_steps(self, max_steps: int) -> None:
        """Update the maximum steps for time pressure calculation."""
        self._max_steps = max(1, max_steps)
