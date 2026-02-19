"""Metrics Tracker — per-step and per-episode metrics collection.

Phase 41: Extracted from SmartCoach for modularity.
"""
from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger("ariaska.coach.metrics_tracker")


@dataclass
class MetricsTrackerConfig:
    """Configuration for metrics tracking."""
    history_window: int = 20
    max_history_multiplier: int = 3


class MetricsTracker:
    """Tracks per-step reasoning and per-episode performance metrics."""

    def __init__(self, config: Optional[MetricsTrackerConfig] = None) -> None:
        self.config = config or MetricsTrackerConfig()
        self._step_reasoning: List[Dict[str, str]] = []
        self._episode_rewards: List[float] = []
        self._episode_discovery_counts: List[int] = []
        self._episode_diversity_ratios: List[float] = []
        self._mentor_calls: int = 0
        self._mentor_overrides: int = 0
        self._total_steps: int = 0
        self._stuck_count: int = 0

    def log_step_reasoning(self, step: int, reasoning: str) -> None:
        self._step_reasoning.append({"step": str(step), "reasoning": reasoning})
        self._total_steps += 1

    def log_mentor_call(self, overrode: bool = False) -> None:
        self._mentor_calls += 1
        if overrode:
            self._mentor_overrides += 1

    def log_stuck(self) -> None:
        self._stuck_count += 1

    def record_episode_performance(self, total_reward: float, discovery_count: int, diversity_ratio: float) -> None:
        self._episode_rewards.append(total_reward)
        self._episode_discovery_counts.append(discovery_count)
        self._episode_diversity_ratios.append(diversity_ratio)
        cap = self.config.history_window * self.config.max_history_multiplier
        if len(self._episode_rewards) > cap:
            self._episode_rewards = self._episode_rewards[-cap:]
            self._episode_discovery_counts = self._episode_discovery_counts[-cap:]
            self._episode_diversity_ratios = self._episode_diversity_ratios[-cap:]

    def get_curriculum_performance(self) -> float:
        if not self._episode_rewards:
            return 0.5
        window = self.config.history_window
        recent_rewards = self._episode_rewards[-window:]
        recent_discoveries = self._episode_discovery_counts[-window:]
        recent_diversity = self._episode_diversity_ratios[-window:]
        scores: List[float] = []
        if recent_discoveries:
            scores.append(min(sum(recent_discoveries) / len(recent_discoveries) / 15.0, 1.0))
        if recent_diversity:
            scores.append(sum(recent_diversity) / len(recent_diversity))
        if len(recent_rewards) >= 3:
            first_half = sum(recent_rewards[:len(recent_rewards) // 2])
            second_half = sum(recent_rewards[len(recent_rewards) // 2:])
            if first_half > 0:
                trend = min(max(second_half / first_half, 0.0), 2.0) / 2.0
            else:
                trend = 0.5 if second_half >= 0 else 0.2
            scores.append(trend)
        return sum(scores) / len(scores) if scores else 0.5

    def get_step_reasoning(self) -> List[Dict[str, str]]:
        return list(self._step_reasoning)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_steps": self._total_steps, "mentor_calls": self._mentor_calls,
            "mentor_overrides": self._mentor_overrides, "stuck_count": self._stuck_count,
            "episodes_tracked": len(self._episode_rewards),
            "curriculum_performance": self.get_curriculum_performance(),
        }

    def reset_step(self) -> None:
        self._step_reasoning.clear()

    def reset_all(self) -> None:
        self._step_reasoning.clear()
        self._episode_rewards.clear()
        self._episode_discovery_counts.clear()
        self._episode_diversity_ratios.clear()
        self._mentor_calls = 0
        self._mentor_overrides = 0
        self._total_steps = 0
        self._stuck_count = 0
