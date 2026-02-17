#!/usr/bin/env python3
"""
core/neurorouter/action_arbitrator.py — Phase 15.0: Action Arbitrator (Midbrain)

Replaces SmartCoach's flat else-block cascade with a structured arbitration
system. Evaluates up to 6 candidates from different sources and selects
the best based on expected value, recency, phase fit, and neuromodulator state.

Candidates:
  1. PPO student action (primary RL output)
  2. Hypothesis action (highest expected value)
  3. SkillLibrary action (semantic match)
  4. Mentor action (only if allowed by AutonomyScheduler + NE rules)
  5. Registry/Playbook verification action
  6. Reflex override (not a candidate — overrides all if triggered)

Feature-flag gated: FF_ACTION_ARBITRATOR.

Author: Phase 15.0 Contract
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ariaska.neurorouter.arbitrator")

_MAX_CANDIDATES = 6


# ── Schemas ─────────────────────────────────────────────────────────────────

@dataclass
class ArbitrationCandidate:
    """A candidate action for arbitration."""
    source: str = ""             # "ppo", "hypothesis", "skill", "mentor", "registry", "playbook"
    command: str = ""
    expected_value: float = 0.0  # estimated reward
    confidence: float = 0.5
    phase_fit: float = 0.5       # how well it fits current phase [0,1]
    recency_penalty: float = 0.0 # penalty for recently tried commands
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def score(self, aggression: float = 0.3) -> float:
        """
        Compute arbitration score.

        Score = expected_value * confidence * phase_fit - recency_penalty
                + aggression_adjustment
        """
        base = self.expected_value * self.confidence * self.phase_fit
        aggression_adj = 0.0
        # Aggressive sources (exploit-oriented) get a boost when aggression is high
        if self.source in ("ppo", "hypothesis") and aggression > 0.5:
            aggression_adj = (aggression - 0.5) * 0.2
        # Conservative sources get a boost when aggression is low
        elif self.source in ("registry", "playbook") and aggression < 0.3:
            aggression_adj = (0.3 - aggression) * 0.15
        return base - self.recency_penalty + aggression_adj

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ArbitrationDecisionLog:
    """Log of an arbitration decision with full reasoning."""
    step: int = 0
    winner_source: str = ""
    winner_command: str = ""
    winner_score: float = 0.0
    candidates_count: int = 0
    all_scores: Dict[str, float] = field(default_factory=dict)
    reason_codes: List[str] = field(default_factory=list)
    aggression_level: float = 0.3
    reflex_override: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def compact(self) -> Dict[str, Any]:
        """Compact representation for CampaignMemory storage."""
        return {
            "step": self.step,
            "src": self.winner_source,
            "cmd": self.winner_command[:60],
            "score": round(self.winner_score, 3),
            "n": self.candidates_count,
            "reasons": self.reason_codes[:3],
        }


# ── Arbitrator ──────────────────────────────────────────────────────────────

class ActionArbitrator:
    """
    Select the best action from multiple candidates.

    Uses a scoring function that accounts for expected value,
    confidence, phase fit, recency, and aggression level.
    """

    def __init__(self) -> None:
        self._decision_count = 0

    def arbitrate(
        self,
        candidates: List[ArbitrationCandidate],
        aggression: float = 0.3,
        step: int = 0,
    ) -> ArbitrationDecisionLog:
        """
        Arbitrate among candidates and return decision log.

        If no candidates, returns empty log.
        Candidates are bounded to MAX_CANDIDATES.
        """
        # Bound candidates
        bounded = candidates[:_MAX_CANDIDATES]

        if not bounded:
            return ArbitrationDecisionLog(
                step=step,
                reason_codes=["no_candidates"],
            )

        # Score all candidates
        scored = [(c, c.score(aggression)) for c in bounded]
        scored.sort(key=lambda x: x[1], reverse=True)

        winner, best_score = scored[0]
        reasons: List[str] = [f"selected_{winner.source}"]

        # Log why we preferred this
        if len(scored) > 1:
            runner_up = scored[1]
            margin = best_score - runner_up[1]
            if margin < 0.05:
                reasons.append("close_margin")
            if winner.source == "ppo":
                reasons.append("rl_primary")

        self._decision_count += 1

        all_scores = {c.source: round(s, 4) for c, s in scored}

        return ArbitrationDecisionLog(
            step=step,
            winner_source=winner.source,
            winner_command=winner.command,
            winner_score=best_score,
            candidates_count=len(bounded),
            all_scores=all_scores,
            reason_codes=reasons,
            aggression_level=aggression,
        )

    def get_stats(self) -> Dict[str, Any]:
        return {"total_decisions": self._decision_count}

    def reset(self) -> None:
        self._decision_count = 0
