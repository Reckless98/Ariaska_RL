#!/usr/bin/env python3
"""
core/training/mentor_ledger.py — Mentor outcome ledger (Mentor2Apprentice).

Makes every mentor intervention a MEASURABLE event instead of a fire-and-forget
command substitution. Pure bookkeeping: no torch, no I/O, no decision authority.

Per agent (one ledger per SmartCoach) it records:
  * proposals   — mentor was asked and answered
  * rejections  — the answer was discarded before arbitration, by reason
                  (offline placeholder, role violation, repeat, exception)
  * accepted    — the answer became an Advisory (and a TeacherTrace)
  * won / lost  — whether the mentor's command was the executed action
  * outcomes    — unified step reward and discovery flag on mentor-won steps,
                  plus the same statistics for non-mentor steps, per phase,
                  for observational comparisons only; source selection confounds these means.

The summary feeds SmartCoach.get_stats() / telemetry. Nothing here changes
what PPO learns; it only observes.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

_EMA_ALPHA = 0.1


@dataclass
class _RewardStat:
    """Running mean/EMA of rewards for one (source-class, phase) bucket."""

    count: int = 0
    total: float = 0.0
    ema: Optional[float] = None
    discoveries: int = 0

    def add(self, reward: float, produced_discovery: bool) -> None:
        self.count += 1
        self.total += reward
        self.ema = reward if self.ema is None else (1 - _EMA_ALPHA) * self.ema + _EMA_ALPHA * reward
        if produced_discovery:
            self.discoveries += 1

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "mean_reward": round(self.mean, 4),
            "ema_reward": round(self.ema, 4) if self.ema is not None else None,
            "discovery_rate": round(self.discoveries / self.count, 4) if self.count else 0.0,
        }


@dataclass
class MentorLedger:
    """Counts and reward statistics for mentor interventions of one agent."""

    proposed: int = 0
    accepted: int = 0
    rejected: int = 0
    rejected_by_reason: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    won: int = 0
    lost: int = 0
    correct_when_won: int = 0
    finalized_traces: int = 0
    bc_samples_stored: int = 0
    ppo_pairing_restored: int = 0
    mentor_steps: Dict[str, _RewardStat] = field(default_factory=lambda: defaultdict(_RewardStat))
    policy_steps: Dict[str, _RewardStat] = field(default_factory=lambda: defaultdict(_RewardStat))
    episode_proposed: int = 0
    episode_won: int = 0

    # ── Recording ────────────────────────────────────────────────────────

    def record_rejection(self, reason: str) -> None:
        """Mentor answered but the answer was discarded before arbitration."""
        self.proposed += 1
        self.episode_proposed += 1
        self.rejected += 1
        self.rejected_by_reason[str(reason or "unknown")] += 1

    def record_accepted(self) -> None:
        """Mentor answer survived validation and entered arbitration."""
        self.proposed += 1
        self.episode_proposed += 1
        self.accepted += 1

    def record_outcome(
        self,
        source: str,
        reward: float,
        produced_discovery: bool,
        phase: str = "",
        trace: Any = None,
        bc_stored: bool = False,
    ) -> None:
        """Record the executed step's outcome.

        ``trace`` is the TeacherTrace attached to this step (if the mentor was
        consulted); when present its ``mentor_won`` / ``teacher_was_correct``
        fields are used, otherwise the winner is inferred from ``source``.
        """
        phase_key = str(phase or "ALL")
        mentor_won = bool(getattr(trace, "mentor_won", None)) if trace is not None else (source == "mentor")
        if trace is not None:
            self.finalized_traces += 1
            if trace is not None and getattr(trace, "mentor_won", None) is not None:
                if mentor_won:
                    self.won += 1
                    self.episode_won += 1
                    if getattr(trace, "teacher_was_correct", None):
                        self.correct_when_won += 1
                else:
                    self.lost += 1
        elif mentor_won:
            self.won += 1
            self.episode_won += 1
            if produced_discovery or reward > 0:
                self.correct_when_won += 1
        if bc_stored:
            self.bc_samples_stored += 1
        bucket = self.mentor_steps if mentor_won else self.policy_steps
        bucket[phase_key].add(float(reward), bool(produced_discovery))
        bucket["ALL"].add(float(reward), bool(produced_discovery)) if phase_key != "ALL" else None

    def record_pairing_restored(self) -> None:
        self.ppo_pairing_restored += 1

    def reset_episode(self) -> None:
        self.episode_proposed = 0
        self.episode_won = 0

    # ── Derived metrics ──────────────────────────────────────────────────

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def win_rate(self) -> float:
        decided = self.won + self.lost
        return self.won / decided if decided else 0.0

    @property
    def correct_rate(self) -> float:
        return self.correct_when_won / self.won if self.won else 0.0

    def reward_delta(self, phase: str = "ALL") -> Optional[float]:
        """mean(mentor-won reward) - mean(policy reward) for a phase; None if no data."""
        m = self.mentor_steps.get(phase)
        p = self.policy_steps.get(phase)
        if not m or not p or not m.count or not p.count:
            return None
        d = m.mean - p.mean
        return d if math.isfinite(d) else None

    def summary(self) -> Dict[str, Any]:
        return {
            "observational_only": True,
            "proposed": self.proposed,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "rejected_by_reason": dict(self.rejected_by_reason),
            "acceptance_rate": round(self.acceptance_rate, 4),
            "won": self.won,
            "lost": self.lost,
            "win_rate": round(self.win_rate, 4),
            "correct_when_won": self.correct_when_won,
            "correct_rate": round(self.correct_rate, 4),
            "finalized_traces": self.finalized_traces,
            "bc_samples_stored": self.bc_samples_stored,
            "ppo_pairing_restored": self.ppo_pairing_restored,
            "mentor_vs_policy_reward_delta": (
                round(self.reward_delta("ALL"), 4) if self.reward_delta("ALL") is not None else None
            ),
            "mentor_steps": {k: v.to_dict() for k, v in self.mentor_steps.items()},
            "policy_steps": {k: v.to_dict() for k, v in self.policy_steps.items()},
            "episode": {"proposed": self.episode_proposed, "won": self.episode_won},
        }


__all__ = ["MentorLedger"]
