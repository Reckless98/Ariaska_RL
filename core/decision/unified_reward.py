"""
core/decision/unified_reward.py — Phase 3: Unified Reward Pipeline.

Single reward computation point for ALL learners.
All learners receive the SAME scalar reward.
Per-algorithm internal scaling happens DOWNSTREAM, not upstream.

Structure:
    env_reward
    + intrinsic_reward (RND)
    + shaped_bonus (discovery, phase, efficiency)
    = total_reward

Every learner sees total_reward.  If a learner needs a different scale
(e.g., PPO normalizes, DDQN adds macro-shaping), it does so internally.

Author: Phase 3 — Reward Unification
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ariaska.decision.reward")


@dataclass
class UnifiedReward:
    """Output of the unified reward pipeline.

    All fields are populated once per step.  Every learner reads from
    this object — no per-algorithm upstream modification.
    """

    # Components
    extrinsic: float = 0.0
    """Base reward from RewardCalculator (env + shaped bonuses)."""

    intrinsic_rnd: float = 0.0
    """RND curiosity bonus (scaled by maturity)."""

    # Derived
    total: float = 0.0
    """extrinsic + intrinsic_rnd — the unified scalar ALL learners see."""

    # Metadata
    breakdown: Dict[str, float] = field(default_factory=dict)
    """Itemized reward breakdown from RewardCalculator for logging."""

    rnd_scale: float = 1.0
    """Scale factor applied to RND (from maturity controller)."""

    def to_dict(self) -> Dict[str, float]:
        return {
            "extrinsic": round(self.extrinsic, 4),
            "intrinsic_rnd": round(self.intrinsic_rnd, 4),
            "total": round(self.total, 4),
            "rnd_scale": round(self.rnd_scale, 3),
        }


class UnifiedRewardPipeline:
    """Single reward computation point for all learners.

    Replaces the divergent reward paths where:
      - PPO saw normalized breakdown.total
      - SAC saw raw breakdown.total + RND intrinsic
      - DDQN saw compute_macro_reward() (completely reshaped)

    Now ALL learners see the same UnifiedReward.total:
      total = extrinsic + rnd_scale * intrinsic_rnd

    DDQN still applies macro-shaping BUT starting from the same base
    reward as everyone else.  PPO still normalizes BUT from the same
    base as everyone else.

    Usage:
        pipeline = UnifiedRewardPipeline()
        reward = pipeline.compute(breakdown, rnd_intrinsic, rnd_scale)
        # ALL learners use reward.total
    """

    def __init__(self) -> None:
        self._step_count: int = 0
        self._total_reward: float = 0.0
        self._reward_variance_acc: float = 0.0
        self._reward_mean: float = 0.0
        self._reward_count: int = 0
        self._last_reward: Optional[UnifiedReward] = None

    def compute(
        self,
        breakdown: Any,
        rnd_intrinsic: float = 0.0,
        rnd_scale: float = 1.0,
    ) -> UnifiedReward:
        """Compute unified reward from reward breakdown + RND.

        Args:
            breakdown: RewardBreakdown from SmartRewardCalculator.
                Must have .total attribute.
            rnd_intrinsic: Raw RND intrinsic reward for this step.
            rnd_scale: Scale factor for RND (from maturity controller).

        Returns:
            UnifiedReward with all fields populated.
        """
        self._step_count += 1

        extrinsic = getattr(breakdown, 'total', 0.0) if breakdown else 0.0
        intrinsic = rnd_intrinsic * rnd_scale
        total = extrinsic + intrinsic

        # Build breakdown dict for logging
        breakdown_dict: Dict[str, float] = {}
        if breakdown is not None:
            for attr in [
                'base_reward', 'progress_bonus', 'novelty_bonus',
                'redundancy_penalty', 'discovery_bonus', 'phase_advance_bonus',
                'efficiency_bonus', 'failure_penalty', 'total',
            ]:
                val = getattr(breakdown, attr, None)
                if val is not None:
                    breakdown_dict[attr] = round(float(val), 4)

        reward = UnifiedReward(
            extrinsic=extrinsic,
            intrinsic_rnd=intrinsic,
            total=total,
            breakdown=breakdown_dict,
            rnd_scale=rnd_scale,
        )

        # Update running statistics
        self._total_reward += total
        self._reward_count += 1
        delta = total - self._reward_mean
        self._reward_mean += delta / self._reward_count
        self._reward_variance_acc += delta * (total - self._reward_mean)
        self._last_reward = reward

        return reward

    @property
    def reward_variance(self) -> float:
        """Running reward variance (Welford's algorithm)."""
        if self._reward_count < 2:
            return 0.0
        return self._reward_variance_acc / (self._reward_count - 1)

    @property
    def reward_mean(self) -> float:
        return self._reward_mean

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def last_reward(self) -> Optional[UnifiedReward]:
        return self._last_reward

    def reset_episode(self) -> None:
        """Reset per-episode stats.  Cross-episode variance kept."""
        pass

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_count": self._step_count,
            "reward_mean": round(self._reward_mean, 4),
            "reward_variance": round(self.reward_variance, 4),
            "total_reward_sum": round(self._total_reward, 4),
        }
