"""Episode Runner — episode/step lifecycle management utilities.

Phase 41: Extracted from SmartOrchestrator for modularity.
Provides helpers for episode timing, stuck detection, and metrics.
"""
from __future__ import annotations

import time
import logging
from typing import Dict, Any, List, Optional, Deque
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger("ariaska.orchestration.episode_runner")


@dataclass
class StepRecord:
    """Record of a single step execution."""
    step: int
    agent_id: str
    command: str
    reward: float
    discoveries: int
    phase: str
    timestamp: float = field(default_factory=time.time)
    source: str = ""
    success: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "agent_id": self.agent_id,
            "command": self.command,
            "reward": self.reward,
            "discoveries": self.discoveries,
            "phase": self.phase,
            "timestamp": self.timestamp,
            "source": self.source,
            "success": self.success,
        }


@dataclass
class EpisodeMetrics:
    """Aggregated metrics for an episode."""
    total_reward: float = 0.0
    total_steps: int = 0
    unique_commands: int = 0
    total_discoveries: int = 0
    phases_reached: List[str] = field(default_factory=list)
    step_at_first_exploit: Optional[int] = None
    completion_bonus: bool = False
    duration_seconds: float = 0.0
    diversity_ratio: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_reward": round(self.total_reward, 2),
            "total_steps": self.total_steps,
            "unique_commands": self.unique_commands,
            "total_discoveries": self.total_discoveries,
            "phases_reached": self.phases_reached,
            "step_at_first_exploit": self.step_at_first_exploit,
            "completion_bonus": self.completion_bonus,
            "duration_seconds": round(self.duration_seconds, 2),
            "diversity_ratio": round(self.diversity_ratio, 3),
        }


class StuckDetector:
    """Detects when the agent is stuck in a non-productive loop."""

    def __init__(self, window: int = 10, threshold: float = 0.5) -> None:
        self.window = window
        self.threshold = threshold
        self._recent_commands: Deque[str] = deque(maxlen=window * 2)
        self._recent_rewards: Deque[float] = deque(maxlen=window)
        self._zero_reward_streak: int = 0

    def record(self, command: str, reward: float) -> None:
        """Record a step result."""
        self._recent_commands.append(command.strip().lower())
        self._recent_rewards.append(reward)
        if reward <= 0:
            self._zero_reward_streak += 1
        else:
            self._zero_reward_streak = 0

    def is_stuck(self) -> bool:
        """Check if the agent appears stuck."""
        if len(self._recent_rewards) < self.window:
            return False
        # Check zero-reward streak
        if self._zero_reward_streak >= self.window:
            return True
        # Check command repetition
        recent = list(self._recent_commands)[-self.window:]
        unique_ratio = len(set(recent)) / max(len(recent), 1)
        if unique_ratio < self.threshold:
            return True
        return False

    def is_repeat_stuck(self) -> bool:
        """Check for exact command repetition."""
        recent = list(self._recent_commands)[-5:]
        if len(recent) < 3:
            return False
        return len(set(recent)) <= 1

    def is_deep_stuck(self) -> bool:
        """Check for deep stagnation (longer window)."""
        if len(self._recent_rewards) < self.window * 2:
            return False
        rewards = list(self._recent_rewards)
        return sum(1 for r in rewards if r <= 0) > len(rewards) * 0.8

    def reset(self) -> None:
        """Reset detector state."""
        self._recent_commands.clear()
        self._recent_rewards.clear()
        self._zero_reward_streak = 0


class EpisodeTracker:
    """Tracks episode-level metrics and step history."""

    def __init__(self) -> None:
        self._steps: List[StepRecord] = []
        self._start_time: float = 0.0
        self._unique_commands: set = set()
        self._phases_seen: List[str] = []
        self._total_reward: float = 0.0
        self._total_discoveries: int = 0
        self._first_exploit_step: Optional[int] = None

    def start_episode(self) -> None:
        """Signal episode start."""
        self._steps.clear()
        self._start_time = time.time()
        self._unique_commands.clear()
        self._phases_seen.clear()
        self._total_reward = 0.0
        self._total_discoveries = 0
        self._first_exploit_step = None

    def record_step(self, record: StepRecord) -> None:
        """Record a step."""
        self._steps.append(record)
        self._unique_commands.add(record.command.strip().lower()[:120])
        self._total_reward += record.reward
        self._total_discoveries += record.discoveries
        if record.phase not in self._phases_seen:
            self._phases_seen.append(record.phase)
        if (
            record.phase in ("EXPLOITATION", "PRIVILEGE_ESCALATION")
            and self._first_exploit_step is None
            and record.success
        ):
            self._first_exploit_step = record.step

    def get_metrics(self) -> EpisodeMetrics:
        """Compute episode metrics."""
        total_steps = len(self._steps)
        return EpisodeMetrics(
            total_reward=self._total_reward,
            total_steps=total_steps,
            unique_commands=len(self._unique_commands),
            total_discoveries=self._total_discoveries,
            phases_reached=list(self._phases_seen),
            step_at_first_exploit=self._first_exploit_step,
            duration_seconds=time.time() - self._start_time if self._start_time else 0.0,
            diversity_ratio=(
                len(self._unique_commands) / max(total_steps, 1)
            ),
        )

    def get_step_history(self) -> List[Dict[str, Any]]:
        """Return step history as dicts."""
        return [s.to_dict() for s in self._steps]
