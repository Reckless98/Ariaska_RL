#!/usr/bin/env python3
"""
core/training/source_win_rate.py — C06: Decision source win-rate EMA tracker

Tracks which decision sources (playbook, ppo, registry, mentor, micro_chain,
phase_guided, cognition, anti_repeat, fallback, dual_mentor) lead to successful
outcomes using exponential moving averages.

Used by SmartCoach to weight decision pipeline stages and by the orchestrator
for telemetry.  No cross-module imports required (pure dataclass + logic).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger("ariaska.source_win_rate")

# Known decision sources from SmartDecisionResult.source
KNOWN_SOURCES = frozenset({
    "playbook",
    "ppo",
    "registry",
    "mentor",
    "dual_mentor",
    "micro_chain",
    "phase_guided",
    "cognition",
    "anti_repeat",
    "fallback",
    "forced",
    "unknown",
})


@dataclass
class SourceStats:
    """Per-source statistics."""
    wins: int = 0
    total: int = 0
    ema_win_rate: float = 0.5  # Prior: 50%
    last_reward: float = 0.0
    ema_reward: float = 0.0

    @property
    def raw_win_rate(self) -> float:
        """Raw (non-smoothed) win rate."""
        return self.wins / max(self.total, 1)


@dataclass
class SourceWinRateTracker:
    """EMA-based win-rate tracker for decision sources.

    Maintains separate EMA win rates and average rewards per source.
    Alpha controls how fast older observations decay (higher = more
    weight on recent).

    Attributes:
        alpha: EMA smoothing factor (default 0.1, ~10-observation window).
        reward_alpha: EMA smoothing for reward tracking (default 0.15).
        sources: Per-source statistics.
    """

    alpha: float = 0.1
    reward_alpha: float = 0.15
    sources: Dict[str, SourceStats] = field(default_factory=dict)

    def record(
        self,
        source: str,
        success: bool,
        reward: float = 0.0,
    ) -> None:
        """Record an outcome for a decision source.

        Args:
            source: Decision source name (e.g. "ppo", "mentor").
            success: Whether the decision led to a positive outcome.
            reward: The reward received for this decision.
        """
        if source not in self.sources:
            self.sources[source] = SourceStats()

        stats = self.sources[source]
        stats.total += 1
        if success:
            stats.wins += 1
        stats.last_reward = reward

        # EMA win rate
        outcome = 1.0 if success else 0.0
        stats.ema_win_rate = (
            self.alpha * outcome + (1.0 - self.alpha) * stats.ema_win_rate
        )

        # EMA reward
        stats.ema_reward = (
            self.reward_alpha * reward + (1.0 - self.reward_alpha) * stats.ema_reward
        )

    def get_win_rate(self, source: str) -> float:
        """Get the EMA win rate for a source (0.5 if unseen)."""
        if source in self.sources:
            return self.sources[source].ema_win_rate
        return 0.5

    def get_reward_ema(self, source: str) -> float:
        """Get the EMA reward for a source (0.0 if unseen)."""
        if source in self.sources:
            return self.sources[source].ema_reward
        return 0.0

    def get_total(self, source: str) -> int:
        """Get total decisions for a source."""
        if source in self.sources:
            return self.sources[source].total
        return 0

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get a summary dict of all tracked sources.

        Returns:
            Dict mapping source name to {ema_win_rate, raw_win_rate,
            ema_reward, total}.
        """
        return {
            name: {
                "ema_win_rate": stats.ema_win_rate,
                "raw_win_rate": stats.raw_win_rate,
                "ema_reward": stats.ema_reward,
                "total": float(stats.total),
            }
            for name, stats in sorted(self.sources.items())
        }

    def get_best_source(self) -> Optional[str]:
        """Return the source with the highest EMA win rate (min 5 samples)."""
        best_name: Optional[str] = None
        best_rate = -1.0
        for name, stats in self.sources.items():
            if stats.total >= 5 and stats.ema_win_rate > best_rate:
                best_rate = stats.ema_win_rate
                best_name = name
        return best_name

    def reset(self) -> None:
        """Clear all source statistics."""
        self.sources.clear()

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        """Alias for get_summary()."""
        return self.get_summary()
