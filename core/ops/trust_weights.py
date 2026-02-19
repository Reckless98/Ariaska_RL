"""
core/ops/trust_weights.py — Phase 39.2: Trust Weights + Annealing

Per-source trust tracking for mentor/LLM outputs.  Adjusts influence
based on prediction accuracy, contradictions, and training maturity.

Feature flag: FF_TRUST_WEIGHTS (default ON)

Mechanics:
  - scalar trust ∈ [0, 1] per mentor source
  - trust adjusts on validated/failed predictions
  - anneals over training time (early=high, later=fades)
  - converts mentor suggestions → prior vector P over action space
  - all influences are measurable and logged
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ariaska.ops.trust_weights")

# ── Constants ──────────────────────────────────────────────────────────

INITIAL_TRUST: float = 0.75
"""Starting trust for new mentor sources."""

MIN_TRUST: float = 0.05
"""Floor — trust never drops below this."""

MAX_TRUST: float = 0.98
"""Ceiling — trust never exceeds this."""

TRUST_REWARD_VALIDATED: float = 0.05
"""Trust increase on a validated prediction."""

TRUST_PENALTY_FAILED: float = 0.08
"""Trust decrease on a failed prediction."""

TRUST_PENALTY_CONTRADICTION: float = 0.12
"""Trust decrease on a contradicted prediction."""

TRUST_PENALTY_LOW_NOVELTY: float = 0.02
"""Trust decrease for redundant/low-novelty suggestion."""

ANNEAL_TOTAL_STEPS: int = 3000
"""Total steps over which trust influence anneals."""

ANNEAL_INITIAL_WEIGHT: float = 0.60
"""Initial influence weight at step 0."""

ANNEAL_MIN_WEIGHT: float = 0.05
"""Minimum influence weight after full anneal."""


@dataclass
class TrustSnapshot:
    """Point-in-time trust state for a single source."""

    source: str
    trust: float
    predictions_total: int = 0
    predictions_validated: int = 0
    predictions_failed: int = 0
    contradictions: int = 0
    influence_weight: float = ANNEAL_INITIAL_WEIGHT
    last_update_step: int = 0


@dataclass
class TrustInfluenceResult:
    """Result of applying trust-weighted influence."""

    prior_vector: List[float] = field(default_factory=list)
    prior_magnitude: float = 0.0
    trust_score: float = 0.0
    influence_weight: float = 0.0
    source: str = ""
    changed_action: bool = False
    original_action: int = -1
    influenced_action: int = -1


class TrustWeightEngine:
    """
    Per-source trust tracking with evidence-based adjustment and annealing.

    Usage:
        engine = TrustWeightEngine(action_dim=5)

        # Register a mentor source
        engine.register_source("gpt")

        # Record prediction outcome
        engine.record_validated("gpt", step=10)
        engine.record_failed("gpt", step=15)

        # Get trust-weighted prior
        result = engine.compute_prior(
            source="gpt",
            action_preferences=[0.8, 0.1, 0.05, 0.03, 0.02],
            current_step=20,
        )

        # Inject into policy
        biased_logits = logits + result.prior_magnitude * prior_vector
    """

    def __init__(
        self,
        action_dim: int = 5,
        anneal_total_steps: int = ANNEAL_TOTAL_STEPS,
        initial_trust: float = INITIAL_TRUST,
    ) -> None:
        self._action_dim = action_dim
        self._anneal_total = anneal_total_steps
        self._initial_trust = initial_trust

        # Per-source trust state
        self._sources: Dict[str, Dict[str, Any]] = {}

        # Global step counter for annealing
        self._global_step: int = 0

        # Logging buffer
        self._influence_log: List[Dict[str, Any]] = []

    def register_source(self, source: str) -> None:
        """Register a new mentor source with initial trust."""
        if source not in self._sources:
            self._sources[source] = {
                "trust": self._initial_trust,
                "predictions_total": 0,
                "predictions_validated": 0,
                "predictions_failed": 0,
                "contradictions": 0,
                "low_novelty_count": 0,
                "last_update_step": 0,
                "history": [],
            }
            logger.info(
                "Trust source registered: %s (initial=%.2f)",
                source, self._initial_trust,
            )

    def _ensure_source(self, source: str) -> None:
        """Auto-register source if not already known."""
        if source not in self._sources:
            self.register_source(source)

    def get_trust(self, source: str) -> float:
        """Get current trust for a source."""
        self._ensure_source(source)
        return self._sources[source]["trust"]

    def record_validated(self, source: str, step: int = 0) -> float:
        """
        Record a validated prediction (expected_signal observed).

        Returns new trust value.
        """
        self._ensure_source(source)
        s = self._sources[source]
        s["predictions_total"] += 1
        s["predictions_validated"] += 1
        s["trust"] = min(s["trust"] + TRUST_REWARD_VALIDATED, MAX_TRUST)
        s["last_update_step"] = step
        s["history"].append(("validated", step))
        if len(s["history"]) > 100:
            s["history"] = s["history"][-100:]
        return s["trust"]

    def record_failed(self, source: str, step: int = 0) -> float:
        """
        Record a failed prediction.

        Returns new trust value.
        """
        self._ensure_source(source)
        s = self._sources[source]
        s["predictions_total"] += 1
        s["predictions_failed"] += 1
        s["trust"] = max(s["trust"] - TRUST_PENALTY_FAILED, MIN_TRUST)
        s["last_update_step"] = step
        s["history"].append(("failed", step))
        if len(s["history"]) > 100:
            s["history"] = s["history"][-100:]
        return s["trust"]

    def record_contradiction(self, source: str, step: int = 0) -> float:
        """
        Record a contradicted prediction (stronger penalty).

        Returns new trust value.
        """
        self._ensure_source(source)
        s = self._sources[source]
        s["contradictions"] += 1
        s["trust"] = max(s["trust"] - TRUST_PENALTY_CONTRADICTION, MIN_TRUST)
        s["last_update_step"] = step
        s["history"].append(("contradiction", step))
        if len(s["history"]) > 100:
            s["history"] = s["history"][-100:]
        return s["trust"]

    def record_low_novelty(self, source: str, step: int = 0) -> float:
        """
        Record a low-novelty/redundant suggestion.

        Returns new trust value.
        """
        self._ensure_source(source)
        s = self._sources[source]
        s["low_novelty_count"] += 1
        s["trust"] = max(s["trust"] - TRUST_PENALTY_LOW_NOVELTY, MIN_TRUST)
        s["last_update_step"] = step
        return s["trust"]

    def set_global_step(self, step: int) -> None:
        """Update global step counter for anneal computation."""
        self._global_step = step

    def compute_anneal_weight(self, step: Optional[int] = None) -> float:
        """
        Compute anneal weight at given step using cosine schedule.

        Returns weight in [ANNEAL_MIN_WEIGHT, ANNEAL_INITIAL_WEIGHT].
        """
        s = step if step is not None else self._global_step
        if self._anneal_total <= 0:
            return ANNEAL_INITIAL_WEIGHT

        progress = min(s / self._anneal_total, 1.0)
        # Cosine anneal
        weight = (
            ANNEAL_MIN_WEIGHT
            + 0.5 * (ANNEAL_INITIAL_WEIGHT - ANNEAL_MIN_WEIGHT)
            * (1.0 + math.cos(math.pi * progress))
        )
        return max(weight, ANNEAL_MIN_WEIGHT)

    def compute_prior(
        self,
        source: str,
        action_preferences: List[float],
        current_step: Optional[int] = None,
    ) -> TrustInfluenceResult:
        """
        Convert mentor action suggestions to a trust-weighted prior vector.

        Args:
            source: Mentor source name (e.g., "gpt", "heuristic")
            action_preferences: Raw preference scores per action [0..action_dim]
            current_step: Current global step for anneal computation

        Returns:
            TrustInfluenceResult with prior_vector and metadata.
        """
        self._ensure_source(source)
        step = current_step if current_step is not None else self._global_step

        trust = self._sources[source]["trust"]
        anneal = self.compute_anneal_weight(step)

        # Normalize action preferences to probability distribution
        prefs = list(action_preferences)
        if len(prefs) < self._action_dim:
            prefs.extend([0.0] * (self._action_dim - len(prefs)))
        prefs = prefs[: self._action_dim]

        total = sum(max(p, 0.0) for p in prefs)
        if total > 0:
            normed = [max(p, 0.0) / total for p in prefs]
        else:
            normed = [1.0 / self._action_dim] * self._action_dim

        # Scale by trust * anneal weight
        magnitude = trust * anneal
        prior = [p * magnitude for p in normed]

        result = TrustInfluenceResult(
            prior_vector=prior,
            prior_magnitude=magnitude,
            trust_score=trust,
            influence_weight=anneal,
            source=source,
        )

        # Log
        self._influence_log.append({
            "step": step,
            "source": source,
            "trust": trust,
            "anneal": anneal,
            "magnitude": magnitude,
            "prior_max": max(prior),
            "ts": time.time(),
        })
        if len(self._influence_log) > 500:
            self._influence_log = self._influence_log[-500:]

        return result

    def get_snapshot(self, source: str) -> TrustSnapshot:
        """Get current trust snapshot for a source."""
        self._ensure_source(source)
        s = self._sources[source]
        return TrustSnapshot(
            source=source,
            trust=s["trust"],
            predictions_total=s["predictions_total"],
            predictions_validated=s["predictions_validated"],
            predictions_failed=s["predictions_failed"],
            contradictions=s["contradictions"],
            influence_weight=self.compute_anneal_weight(),
            last_update_step=s["last_update_step"],
        )

    def get_all_snapshots(self) -> Dict[str, TrustSnapshot]:
        """Get trust snapshots for all registered sources."""
        return {
            source: self.get_snapshot(source)
            for source in self._sources
        }

    def get_influence_log(self, last_n: int = 20) -> List[Dict[str, Any]]:
        """Get recent influence log entries."""
        return self._influence_log[-last_n:]

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get full diagnostic data for debug instrumentation."""
        snapshots = self.get_all_snapshots()
        return {
            "sources": {
                name: {
                    "trust": snap.trust,
                    "validated": snap.predictions_validated,
                    "failed": snap.predictions_failed,
                    "contradictions": snap.contradictions,
                    "influence": snap.influence_weight,
                }
                for name, snap in snapshots.items()
            },
            "global_step": self._global_step,
            "anneal_weight": self.compute_anneal_weight(),
            "total_influences": len(self._influence_log),
        }

    def reset(self) -> None:
        """Reset all trust state for a new engagement."""
        self._sources.clear()
        self._global_step = 0
        self._influence_log.clear()
