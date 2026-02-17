#!/usr/bin/env python3
"""
core/neuro/neuromodulators.py — Phase 15.0: Neuromodulator Engine

Four neuromodulators inspired by biological neuroscience:
  DA  (Dopamine)     — reward prediction error → learning rate modulation
  NE  (Norepinephrine) — uncertainty/novelty → exploration boost, mentor re-escalation
  ACh (Acetylcholine) — attention/focus → consolidation priority, hypothesis testing
  5HT (Serotonin)    — stability/patience → entropy damping, aggression reduction

All values are clamped to [0, 1]. The engine is stateless per-call (state is
carried in NeuromodulatorState). Feature-flag gated: FF_NEUROMODULATORS.

Author: Phase 15.0 Contract
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ariaska.neuro.neuromodulators")

# ── Constants ───────────────────────────────────────────────────────────────

_CLAMP_MIN = 0.0
_CLAMP_MAX = 1.0
_HISTORY_MAX = 256  # bounded per-episode storage


def _clamp(v: float) -> float:
    return max(_CLAMP_MIN, min(_CLAMP_MAX, v))


# ── Schemas ─────────────────────────────────────────────────────────────────

@dataclass
class NeuromodulatorState:
    """Snapshot of all four neuromodulator levels at a single step."""
    da: float = 0.5   # Dopamine — reward prediction error
    ne: float = 0.3   # Norepinephrine — uncertainty
    ach: float = 0.4  # Acetylcholine — attention/focus
    sht: float = 0.5  # Serotonin — stability

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    def to_vector(self) -> List[float]:
        """Return [da, ne, ach, sht] as a 4-dim list."""
        return [self.da, self.ne, self.ach, self.sht]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NeuromodulatorState":
        return cls(
            da=_clamp(float(d.get("da", 0.5))),
            ne=_clamp(float(d.get("ne", 0.3))),
            ach=_clamp(float(d.get("ach", 0.4))),
            sht=_clamp(float(d.get("sht", 0.5))),
        )


@dataclass
class NeuromodulatorInputs:
    """Raw signals used to compute neuromodulator levels."""
    # DA inputs
    predicted_value: float = 0.0
    realized_reward: float = 0.0

    # NE inputs
    policy_entropy: float = 0.5         # normalized [0,1]
    confidence_min: float = 1.0         # min parser confidence
    confidence_disagreements: int = 0   # from ParserTeacherOutput
    hypothesis_refuted_rate: float = 0.0

    # ACh inputs
    hypothesis_confirmed_count: int = 0
    hypothesis_tested_count: int = 0
    evidence_delta: int = 0             # new evidence nodes this step

    # 5-HT inputs
    replan_count: int = 0
    steps_since_progress: int = 0
    detection_risk: float = 0.0


# ── Engine ──────────────────────────────────────────────────────────────────

class NeuromodulatorEngine:
    """
    Compute neuromodulator levels from raw signals.

    Stateless per-call — takes NeuromodulatorInputs and previous state,
    returns new NeuromodulatorState. Uses exponential moving average for
    smoothing (alpha=0.3).
    """

    def __init__(self, alpha: float = 0.3) -> None:
        self._alpha = alpha

    def compute(
        self,
        inputs: NeuromodulatorInputs,
        prev: Optional[NeuromodulatorState] = None,
    ) -> NeuromodulatorState:
        """Compute new neuromodulator state from inputs and previous state."""
        if prev is None:
            prev = NeuromodulatorState()

        # ── DA: reward prediction error (RPE) ────────────────────────
        rpe = inputs.realized_reward - inputs.predicted_value
        # Sigmoid-like mapping of RPE to [0, 1]
        raw_da = 1.0 / (1.0 + math.exp(-rpe * 2.0)) if abs(rpe) < 20 else (1.0 if rpe > 0 else 0.0)
        da = _clamp(self._ema(prev.da, raw_da))

        # ── NE: uncertainty signal ───────────────────────────────────
        # High entropy + low confidence + disagreements → high NE
        entropy_signal = inputs.policy_entropy
        confidence_signal = 1.0 - inputs.confidence_min
        disagree_signal = min(1.0, inputs.confidence_disagreements / 3.0)
        refuted_signal = inputs.hypothesis_refuted_rate
        raw_ne = 0.35 * entropy_signal + 0.25 * confidence_signal + 0.2 * disagree_signal + 0.2 * refuted_signal
        ne = _clamp(self._ema(prev.ne, raw_ne))

        # ── ACh: attention/focus ─────────────────────────────────────
        # High when actively testing hypotheses and gaining evidence
        test_signal = min(1.0, inputs.hypothesis_tested_count / 5.0)
        confirm_signal = min(1.0, inputs.hypothesis_confirmed_count / 3.0)
        evidence_signal = min(1.0, inputs.evidence_delta / 5.0)
        raw_ach = 0.3 * test_signal + 0.4 * confirm_signal + 0.3 * evidence_signal
        ach = _clamp(self._ema(prev.ach, raw_ach))

        # ── 5-HT: stability ─────────────────────────────────────────
        # High when stable progress, low when replanning or stuck
        replan_signal = 1.0 - min(1.0, inputs.replan_count / 5.0)
        stuck_signal = 1.0 - min(1.0, inputs.steps_since_progress / 10.0)
        risk_signal = 1.0 - inputs.detection_risk
        raw_sht = 0.35 * replan_signal + 0.35 * stuck_signal + 0.3 * risk_signal
        sht = _clamp(self._ema(prev.sht, raw_sht))

        return NeuromodulatorState(da=da, ne=ne, ach=ach, sht=sht)

    def _ema(self, prev: float, raw: float) -> float:
        """Exponential moving average smoothing."""
        return (1.0 - self._alpha) * prev + self._alpha * raw

    def apply_modulation(
        self, state: NeuromodulatorState
    ) -> Dict[str, float]:
        """
        Convert neuromodulator state to concrete parameter adjustments.

        Returns dict with modulation multipliers/offsets:
          - entropy_coef_mult: multiply PPO entropy coefficient (DA-driven)
          - lr_mult: multiply learning rate (DA-driven)
          - bc_weight_mult: multiply BC loss weight (ACh-driven)
          - exploration_bonus: additive exploration bonus (NE-driven)
          - mentor_gate_threshold_offset: offset for mentor gating (NE-driven)
          - consolidation_priority_mult: multiply consolidation priority (ACh × DA)
          - aggression_offset: additive adjustment to aggression (5-HT dampening)
        """
        return {
            # High DA → higher LR (capitalize on good signal)
            "lr_mult": _clamp(0.5 + state.da),  # [0.5, 1.0]
            # Low DA → more exploration via entropy
            "entropy_coef_mult": _clamp(1.5 - state.da),  # [0.5, 1.5]
            # High ACh → stronger BC learning
            "bc_weight_mult": _clamp(0.5 + state.ach),  # [0.5, 1.0]
            # High NE → exploration bonus
            "exploration_bonus": _clamp(state.ne * 0.3),  # [0, 0.3]
            # High NE → lower mentor gate (easier to call mentor)
            "mentor_gate_threshold_offset": -0.15 * state.ne,  # [-0.15, 0]
            # High DA × High ACh → prioritize consolidation
            "consolidation_priority_mult": _clamp(0.3 + state.da * state.ach),
            # High 5-HT → reduce aggression
            "aggression_offset": -0.2 * state.sht,  # [-0.2, 0]
        }


class NeuromodulatorHistory:
    """Bounded per-episode history of neuromodulator states."""

    def __init__(self, max_steps: int = _HISTORY_MAX) -> None:
        self._max_steps = max_steps
        self._history: List[NeuromodulatorState] = []

    def record(self, state: NeuromodulatorState) -> None:
        """Record a state, evicting oldest if at capacity."""
        self._history.append(state)
        if len(self._history) > self._max_steps:
            self._history = self._history[-self._max_steps:]

    def get_means(self) -> Dict[str, float]:
        """Return mean values across all recorded states."""
        if not self._history:
            return {"da": 0.5, "ne": 0.3, "ach": 0.4, "sht": 0.5}
        n = len(self._history)
        return {
            "da": sum(s.da for s in self._history) / n,
            "ne": sum(s.ne for s in self._history) / n,
            "ach": sum(s.ach for s in self._history) / n,
            "sht": sum(s.sht for s in self._history) / n,
        }

    def __len__(self) -> int:
        return len(self._history)

    def clear(self) -> None:
        self._history.clear()

    def as_list(self) -> List[Dict[str, float]]:
        """Return bounded serializable list."""
        return [s.to_dict() for s in self._history[-self._max_steps:]]
