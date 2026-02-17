#!/usr/bin/env python3
"""
core/training/autonomy_scheduler.py — Phase 14.0: Autonomy Scheduler

Contract C1.9: Mentor annealing controller with monotonic invariants.
Manages per-agent autonomy scores that start at 0 (full mentoring)
and strictly increase toward 1 (full autonomy).

Formula: score = ALPHA * success_rate + BETA * not_divergence + GAMMA * diversity
Threshold: max(0.2, 0.8 - 0.012 * episode)

Mentor is called when score < threshold.
Re-escalation (lowering autonomy) only for 3 valid reasons.

Author: Phase 14.0 Contract C1.9
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("ariaska.autonomy_scheduler")

# Valid reasons for re-escalation (mentor re-engagement)
VALID_REESCALATION_REASONS = frozenset({
    "uncertainty_spike",
    "hypothesis_failure_streak",
    "guardrail_triggered",
})


@dataclass
class AgentAutonomy:
    """Per-agent autonomy state tracking."""
    agent_id: str
    score: float = 0.0
    episode_scores: List[float] = field(default_factory=list)
    mentor_calls: int = 0
    autonomous_decisions: int = 0
    total_decisions: int = 0
    success_rate: float = 0.0
    divergence_rate: float = 0.0
    diversity_rate: float = 0.0
    reescalation_count: int = 0
    last_reescalation_reason: str = ""

    def mentor_ratio(self) -> float:
        """Fraction of decisions that used mentor."""
        if self.total_decisions == 0:
            return 1.0
        return self.mentor_calls / self.total_decisions

    def to_dict(self) -> Dict:
        return {
            "agent_id": self.agent_id,
            "score": round(self.score, 4),
            "mentor_calls": self.mentor_calls,
            "autonomous_decisions": self.autonomous_decisions,
            "total_decisions": self.total_decisions,
            "success_rate": round(self.success_rate, 4),
            "divergence_rate": round(self.divergence_rate, 4),
            "diversity_rate": round(self.diversity_rate, 4),
            "mentor_ratio": round(self.mentor_ratio(), 4),
            "reescalation_count": self.reescalation_count,
        }


class AutonomyScheduler:
    """
    Manages mentor annealing for all agents.

    Contract C1.9:
    - ALPHA=0.4, BETA=0.3, GAMMA=0.3
    - score = ALPHA * success_rate + BETA * (1 - divergence_rate) + GAMMA * diversity_rate
    - threshold = max(0.2, 0.8 - 0.012 * episode)
    - Monotonic: score can only increase (except for valid re-escalation)
    - Re-escalation only for: uncertainty_spike, hypothesis_failure_streak, guardrail_triggered
    """

    ALPHA: float = 0.4   # Weight on success rate
    BETA: float = 0.3    # Weight on not-divergence
    GAMMA: float = 0.3   # Weight on diversity

    def __init__(self) -> None:
        self._agents: Dict[str, AgentAutonomy] = {}
        self._episode: int = 0

    def register_agent(self, agent_id: str) -> None:
        """Register an agent for autonomy tracking."""
        if agent_id not in self._agents:
            self._agents[agent_id] = AgentAutonomy(agent_id=agent_id)

    def _compute_raw_score(self, agent_id: str) -> float:
        """Compute raw autonomy score from current metrics."""
        if agent_id not in self._agents:
            return 0.0
        agent = self._agents[agent_id]
        score = (
            self.ALPHA * agent.success_rate
            + self.BETA * (1.0 - agent.divergence_rate)
            + self.GAMMA * agent.diversity_rate
        )
        return max(0.0, min(1.0, score))

    def get_score(self, agent_id: str) -> float:
        """
        Return stored autonomy score for agent.

        This is the monotonic-enforced score (may differ from raw computation).
        """
        if agent_id not in self._agents:
            return 0.0
        return self._agents[agent_id].score

    def get_threshold(self, episode: int) -> float:
        """
        Compute mentor threshold for the given episode.

        Formula: max(0.2, 0.8 - 0.012 * episode)
        Decreases linearly from 0.8 to floor of 0.2.
        """
        return max(0.2, 0.8 - 0.012 * episode)

    def should_call_mentor(self, agent_id: str, episode: int) -> Tuple[bool, str]:
        """
        Decide whether mentor should be called for this agent.

        Returns (should_call, reason).
        If score < threshold: mentor needed.
        If score >= threshold: agent is autonomous.
        """
        self._episode = episode
        score = self.get_score(agent_id)
        threshold = self.get_threshold(episode)

        if agent_id not in self._agents:
            return True, "agent_not_registered"

        if score < threshold:
            reason = (
                f"score={score:.3f} < threshold={threshold:.3f} "
                f"(ep={episode})"
            )
            return True, reason

        return False, f"autonomous: score={score:.3f} >= threshold={threshold:.3f}"

    def update(
        self,
        agent_id: str,
        success_rate: float,
        divergence_rate: float,
        diversity_rate: float,
    ) -> float:
        """
        Update agent's autonomy metrics after an episode.

        Enforces monotonic non-decreasing score (except for valid re-escalation).
        Returns new score.
        """
        if agent_id not in self._agents:
            self.register_agent(agent_id)

        agent = self._agents[agent_id]
        agent.success_rate = max(0.0, min(1.0, success_rate))
        agent.divergence_rate = max(0.0, min(1.0, divergence_rate))
        agent.diversity_rate = max(0.0, min(1.0, diversity_rate))

        new_score = self._compute_raw_score(agent_id)

        # Enforce monotonic non-decreasing
        if new_score < agent.score:
            new_score = agent.score

        agent.score = new_score
        agent.episode_scores.append(new_score)

        logger.debug(
            f"[AUTONOMY] {agent_id}: score={new_score:.4f} "
            f"(success={success_rate:.3f}, div={divergence_rate:.3f}, "
            f"diversity={diversity_rate:.3f})"
        )

        return new_score

    def can_reescalate(self, reason: str) -> bool:
        """
        Check if a re-escalation reason is valid.

        Only 3 valid reasons: uncertainty_spike, hypothesis_failure_streak, guardrail_triggered
        """
        return reason in VALID_REESCALATION_REASONS

    def reescalate(self, agent_id: str, reason: str, reduction: float = 0.1) -> bool:
        """
        Temporarily reduce autonomy score for valid re-escalation.

        Returns True if re-escalation was applied, False if reason invalid.
        """
        if not self.can_reescalate(reason):
            logger.warning(
                f"[AUTONOMY] Invalid re-escalation reason: {reason}. "
                f"Valid reasons: {VALID_REESCALATION_REASONS}"
            )
            return False

        if agent_id not in self._agents:
            return False

        agent = self._agents[agent_id]
        old_score = agent.score
        agent.score = max(0.0, agent.score - reduction)
        agent.reescalation_count += 1
        agent.last_reescalation_reason = reason

        logger.info(
            f"[AUTONOMY] Re-escalation for {agent_id}: "
            f"{old_score:.4f} → {agent.score:.4f} (reason: {reason})"
        )
        return True

    def record_decision(self, agent_id: str, used_mentor: bool) -> None:
        """Record whether a decision used mentor or was autonomous."""
        if agent_id not in self._agents:
            self.register_agent(agent_id)

        agent = self._agents[agent_id]
        agent.total_decisions += 1
        if used_mentor:
            agent.mentor_calls += 1
        else:
            agent.autonomous_decisions += 1

    def get_agent_stats(self, agent_id: str) -> Dict:
        """Get autonomy stats for a specific agent."""
        if agent_id not in self._agents:
            return {"agent_id": agent_id, "score": 0.0}
        return self._agents[agent_id].to_dict()

    def get_all_stats(self) -> Dict[str, Dict]:
        """Get autonomy stats for all agents."""
        return {aid: a.to_dict() for aid, a in self._agents.items()}

    def reset_episode(self) -> None:
        """Reset per-episode counters (not cumulative scores)."""
        # Scores persist across episodes (monotonic)
        # Only reset per-episode decision counters
        pass

    def reset(self) -> None:
        """Full reset for new training run."""
        self._agents.clear()
        self._episode = 0
