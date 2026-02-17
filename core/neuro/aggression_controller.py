#!/usr/bin/env python3
"""
core/neuro/aggression_controller.py — Phase 15.0: Aggression Controller

Computes a bounded aggression_level in [0, 1] per step.
Higher aggression → prefer exploitation over enumeration, shorter timeouts.
Lower aggression → prefer hypothesis testing, broad recon, patience.

Aggression is influenced by neuromodulators (5-HT dampens, DA boosts),
phase (exploitation phases naturally higher), and recent success/failure.

Feature-flag gated: FF_AGGRESSION_CONTROLLER.

Author: Phase 15.0 Contract
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger("ariaska.neuro.aggression")

_CLAMP_MIN = 0.0
_CLAMP_MAX = 1.0
_HISTORY_MAX = 64

# Phase-based aggression baselines (HTB Lab profile)
_PHASE_BASELINE: Dict[str, float] = {
    "RECON": 0.35,
    "ENUMERATION": 0.45,
    "EXPLOITATION": 0.65,
    "PRIVILEGE_ESCALATION": 0.75,
    "LATERAL_MOVEMENT": 0.55,
    "POST_EXPLOITATION": 0.65,
    "EXFILTRATION": 0.45,
}

# Hard ceiling — never exceed this even with all boosts
_AGGRESSION_CEILING = 0.85
# NE spike threshold — clamp aggression when NE exceeds this
_NE_CLAMP_THRESHOLD = 0.75
# Detection risk threshold — force aggression down
_DETECTION_CLAMP_THRESHOLD = 0.7


def _clamp(v: float) -> float:
    return max(_CLAMP_MIN, min(_CLAMP_MAX, v))


# ── Schemas ─────────────────────────────────────────────────────────────────

@dataclass
class AggressionState:
    """Snapshot of aggression level at one step."""
    level: float = 0.3
    reason_codes: List[str] = field(default_factory=list)
    phase: str = ""
    step: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AggressionInputs:
    """Raw signals for aggression computation."""
    phase: str = "RECON"
    da_level: float = 0.5    # Dopamine
    sht_level: float = 0.5   # Serotonin (dampener)
    ne_level: float = 0.3    # Norepinephrine
    recent_successes: int = 0       # successes in last 5 steps
    recent_failures: int = 0        # failures in last 5 steps
    steps_since_progress: int = 0   # stagnation counter
    shell_obtained: bool = False
    detection_risk: float = 0.0


# ── Controller ──────────────────────────────────────────────────────────────

class AggressionController:
    """
    Compute bounded aggression level per step.

    aggression = clamp(phase_baseline + da_boost - sht_dampen + stagnation_push + ...)
    """

    def compute(self, inputs: AggressionInputs) -> AggressionState:
        """Compute AggressionState from inputs."""
        reasons: List[str] = []

        # Phase baseline
        baseline = _PHASE_BASELINE.get(inputs.phase, 0.3)
        reasons.append(f"phase_base={baseline:.2f}")

        # DA boost: high reward signal → more aggressive
        da_boost = (inputs.da_level - 0.5) * 0.3  # [-0.15, +0.15]
        if abs(da_boost) > 0.05:
            reasons.append(f"da_boost={da_boost:+.2f}")

        # 5-HT dampening: high serotonin → less aggressive
        sht_dampen = (inputs.sht_level - 0.5) * 0.25  # [-0.125, +0.125]
        if abs(sht_dampen) > 0.05:
            reasons.append(f"sht_dampen={-sht_dampen:+.2f}")

        # Stagnation push: stuck → increase aggression to try harder things
        stag_push = 0.0
        if inputs.steps_since_progress > 5:
            stag_push = min(0.2, (inputs.steps_since_progress - 5) * 0.04)
            reasons.append(f"stagnation={stag_push:+.2f}")

        # Success momentum
        success_boost = 0.0
        if inputs.recent_successes > 2:
            success_boost = 0.1
            reasons.append("success_momentum")

        # Failure caution
        failure_dampen = 0.0
        if inputs.recent_failures > 3:
            failure_dampen = -0.1
            reasons.append("failure_caution")

        # Detection risk dampening
        risk_dampen = 0.0
        if inputs.detection_risk > 0.5:
            risk_dampen = -(inputs.detection_risk - 0.5) * 0.3
            reasons.append(f"risk_dampen={risk_dampen:+.2f}")

        # Shell obtained → moderate push
        shell_push = 0.0
        if inputs.shell_obtained:
            shell_push = 0.1
            reasons.append("shell_obtained")

        # NE spike guard: high uncertainty → forced aggression reduction
        ne_clamp = 0.0
        if inputs.ne_level > _NE_CLAMP_THRESHOLD:
            ne_clamp = -(inputs.ne_level - _NE_CLAMP_THRESHOLD) * 0.4
            reasons.append(f"ne_clamp={ne_clamp:+.2f}")

        # High detection risk hard clamp
        detection_hard_clamp = 0.0
        if inputs.detection_risk > _DETECTION_CLAMP_THRESHOLD:
            detection_hard_clamp = -(inputs.detection_risk - _DETECTION_CLAMP_THRESHOLD) * 0.5
            reasons.append(f"detection_hard_clamp={detection_hard_clamp:+.2f}")

        raw_level = (
            baseline + da_boost - sht_dampen + stag_push
            + success_boost + failure_dampen + risk_dampen + shell_push
            + ne_clamp + detection_hard_clamp
        )
        level = _clamp(min(raw_level, _AGGRESSION_CEILING))

        return AggressionState(
            level=level,
            reason_codes=reasons,
            phase=inputs.phase,
        )


class AggressionHistory:
    """Bounded per-episode aggression state history."""

    def __init__(self, max_steps: int = _HISTORY_MAX) -> None:
        self._max = max_steps
        self._history: List[AggressionState] = []

    def record(self, state: AggressionState) -> None:
        self._history.append(state)
        if len(self._history) > self._max:
            self._history = self._history[-self._max:]

    def get_mean(self) -> float:
        if not self._history:
            return 0.3
        return sum(s.level for s in self._history) / len(self._history)

    def __len__(self) -> int:
        return len(self._history)

    def clear(self) -> None:
        self._history.clear()
