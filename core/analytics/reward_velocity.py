"""
Reward velocity tracker for Ariaska training optimization.

R67: Tracks per-step reward deltas and momentum to detect stalls early.
- Rolling window of reward deltas
- Velocity = mean of recent deltas (positive = improving, negative = declining)
- Acceleration = velocity change rate (detects momentum shifts)
- Stall detection: sustained low velocity → codex trigger signal
- Peak tracking: records best-ever and best-recent for normalization

Usage:
    from core.analytics.reward_velocity import RewardVelocityTracker
    rv = RewardVelocityTracker()
    rv.record(step_reward=125.0, phase_ord=3)
    print(rv.velocity, rv.is_stalling, rv.peak_ratio)
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

logger = logging.getLogger("ariaska.analytics.reward_velocity")


@dataclass
class RewardVelocityTracker:
    """Tracks reward momentum over a rolling window.

    Provides:
      - velocity: mean reward delta (positive = gaining, negative = losing)
      - acceleration: change in velocity (positive = accelerating)
      - is_stalling: True if velocity near zero for multiple steps
      - peak_ratio: current avg vs best-ever (0..1)
      - phase_velocity: how fast phases are advancing
    """

    window_size: int = 8
    stall_threshold: float = 15.0  # velocity below this = stalling
    stall_steps_needed: int = 4    # consecutive low-velocity steps to trigger

    # Internal state
    _rewards: Deque[float] = field(default_factory=lambda: deque(maxlen=8))
    _deltas: Deque[float] = field(default_factory=lambda: deque(maxlen=8))
    _phase_ords: Deque[int] = field(default_factory=lambda: deque(maxlen=8))
    _velocities: Deque[float] = field(default_factory=lambda: deque(maxlen=8))
    _stall_count: int = 0
    _best_reward: float = 0.0
    _best_recent: float = 0.0
    _total_steps: int = 0

    def __post_init__(self) -> None:
        self._rewards = deque(maxlen=self.window_size)
        self._deltas = deque(maxlen=self.window_size)
        self._phase_ords = deque(maxlen=self.window_size)
        self._velocities = deque(maxlen=self.window_size)
        self._stall_count = 0
        self._best_reward = 0.0
        self._best_recent = 0.0
        self._total_steps = 0

    def record(self, step_reward: float, phase_ord: int = 0) -> None:
        """Record a step's reward and phase ordinal."""
        self._total_steps += 1

        # Compute delta from previous
        if self._rewards:
            delta = step_reward - self._rewards[-1]
        else:
            delta = step_reward  # First step: delta = reward itself

        self._rewards.append(step_reward)
        self._deltas.append(delta)
        self._phase_ords.append(phase_ord)

        # Update velocity
        v = self.velocity
        self._velocities.append(v)

        # Track peaks
        if step_reward > self._best_reward:
            self._best_reward = step_reward
        if step_reward > self._best_recent:
            self._best_recent = step_reward

        # Stall detection
        if abs(v) < self.stall_threshold and self._total_steps >= 3:
            self._stall_count += 1
        else:
            self._stall_count = max(0, self._stall_count - 1)

    @property
    def velocity(self) -> float:
        """Mean reward delta over window. Positive = improving."""
        if not self._deltas:
            return 0.0
        return sum(self._deltas) / len(self._deltas)

    @property
    def acceleration(self) -> float:
        """Change in velocity (second derivative). Positive = accelerating."""
        if len(self._velocities) < 2:
            return 0.0
        return self._velocities[-1] - self._velocities[-2]

    @property
    def is_stalling(self) -> bool:
        """True if velocity has been low for stall_steps_needed steps."""
        return self._stall_count >= self.stall_steps_needed

    @property
    def peak_ratio(self) -> float:
        """Current average vs best-ever reward (0..1). Low = far from peak."""
        if self._best_reward <= 0:
            return 0.5
        current_avg = sum(self._rewards) / len(self._rewards) if self._rewards else 0
        return min(1.0, max(0.0, current_avg / self._best_reward))

    @property
    def phase_velocity(self) -> float:
        """Rate of phase advancement over window. 0 = stuck, 1 = advancing every step."""
        if len(self._phase_ords) < 2:
            return 0.0
        phase_range = max(self._phase_ords) - min(self._phase_ords)
        return min(1.0, phase_range / max(1, len(self._phase_ords) - 1))

    @property
    def trend_emoji(self) -> str:
        """Visual trend indicator for HUD."""
        v = self.velocity
        if v > 30:
            return "🚀"
        elif v > 10:
            return "📈"
        elif v > -10:
            return "➡️"
        elif v > -30:
            return "📉"
        else:
            return "⚠️"

    def get_summary(self) -> dict:
        """Return summary dict for logging."""
        return {
            "velocity": round(self.velocity, 2),
            "acceleration": round(self.acceleration, 2),
            "is_stalling": self.is_stalling,
            "peak_ratio": round(self.peak_ratio, 3),
            "phase_velocity": round(self.phase_velocity, 3),
            "stall_count": self._stall_count,
            "best_reward": round(self._best_reward, 1),
            "total_steps": self._total_steps,
        }

    def reset_episode(self) -> None:
        """Clear all buffers for new episode. Preserves best_reward across episodes."""
        self._rewards.clear()
        self._deltas.clear()
        self._phase_ords.clear()
        self._velocities.clear()
        self._stall_count = 0
        self._best_recent = 0.0
        self._total_steps = 0
        # NOTE: _best_reward persists across episodes for cross-episode normalization
