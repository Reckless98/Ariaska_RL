"""Meta-RL Adaptive Controller — Algorithm Harmony Phase.

Lightweight meta-learning controller that adapts hyperparameters across
episodes based on performance signals and scenario characteristics.

Unlike full Reptile inner-loop meta-learning (which requires multiple
separate training episodes per scenario), this controller observes
episode outcomes and tunes PPO/SAC/DDQN hyperparameters in real-time.

Adaptation signals:
    - Episode reward trajectory (improving, stalling, declining)
    - Discovery rate (information gain per step)
    - Phase progression speed (how fast the agent advances phases)
    - Algorithm disagreement (PPO-SAC confidence divergence)

Tunable parameters:
    - PPO entropy coefficient (explore vs exploit)
    - PPO learning rate (fast vs stable learning)
    - DDQN epsilon (macro exploration rate)
    - RND reward scale (curiosity pressure)
    - SAC alpha target (entropy temperature)

The controller uses a simple bandit approach: track which parameter
configurations led to good episodes, and bias toward those settings.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ariaska.algorithms.meta_rl")


@dataclass
class MetaRLConfig:
    """Configuration for the Meta-RL adaptive controller."""
    enabled: bool = True
    warmup_episodes: int = 3         # Skip adaptation for first N episodes
    adaptation_rate: float = 0.15    # How aggressively to adjust (0=none, 1=instant)
    performance_window: int = 5      # Rolling window for trend detection
    # Bounds for tunable parameters
    entropy_range: tuple[float, float] = (0.003, 0.05)
    lr_range: tuple[float, float] = (5e-5, 1e-3)
    ddqn_epsilon_range: tuple[float, float] = (0.05, 0.4)
    rnd_scale_range: tuple[float, float] = (0.3, 2.0)


class MetaRLController:
    """Adaptive hyperparameter controller for cross-algorithm harmony.

    Observes episode-level performance and adjusts algorithm parameters
    to maximize learning velocity. Acts as the "conductor" of the
    algorithm ensemble.

    Usage::

        meta_rl = MetaRLController()

        # After each episode:
        adjustments = meta_rl.adapt(
            episode_metrics={
                "total_reward": 1500.0,
                "steps": 45,
                "discoveries": 12,
                "highest_phase": "EXPLOITATION",
                "avg_sac_agreement": 0.6,
            }
        )
        # adjustments = {"entropy_coef": 0.015, "lr_scale": 1.2, ...}
    """

    PHASE_ORDINALS = {
        "RECON": 0, "ENUMERATION": 1, "EXPLOITATION": 2,
        "PRIVILEGE_ESCALATION": 3, "LATERAL_MOVEMENT": 4,
        "POST_EXPLOITATION": 5, "EXFILTRATION": 6, "CLOSEOUT": 7,
    }

    def __init__(self, config: Optional[MetaRLConfig] = None) -> None:
        self.config = config or MetaRLConfig()
        self._episode_history: List[Dict[str, float]] = []
        self._adjustments_history: List[Dict[str, float]] = []
        self._episodes_seen: int = 0

        # Current meta-state: tracks what's working
        self._reward_trend: float = 0.0       # Positive = improving
        self._discovery_efficiency: float = 0.0  # Discoveries per step
        self._phase_velocity: float = 0.0     # Phase ordinal per step
        self._stagnation_streak: int = 0      # Consecutive non-improving episodes

        logger.info(
            "MetaRLController initialized | warmup=%d | adapt_rate=%.2f",
            self.config.warmup_episodes, self.config.adaptation_rate,
        )

    def adapt(self, episode_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Compute hyperparameter adjustments based on episode performance.

        Args:
            episode_metrics: Dict with keys:
                total_reward: Episode cumulative reward.
                steps: Number of steps in episode.
                discoveries: Number of new discoveries.
                highest_phase: Name of highest phase reached.
                avg_sac_agreement: Mean SAC-PPO agreement [0, 1].

        Returns:
            Dict of adjustments to apply:
                entropy_coef: Suggested PPO entropy coefficient.
                lr_scale: Multiplicative LR adjustment (1.0 = no change).
                ddqn_epsilon: Suggested DDQN exploration rate.
                rnd_scale: Suggested RND reward scale.
                meta_signal: Human-readable summary of adaptation.
        """
        self._episodes_seen += 1
        reward = episode_metrics.get("total_reward", 0.0)
        steps = max(1, episode_metrics.get("steps", 1))
        discoveries = episode_metrics.get("discoveries", 0)
        highest_phase = episode_metrics.get("highest_phase", "RECON")
        sac_agreement = episode_metrics.get("avg_sac_agreement", 0.5)

        # Compute derived signals
        reward_per_step = reward / steps
        disc_per_step = discoveries / steps
        phase_ord = self.PHASE_ORDINALS.get(highest_phase, 0)
        phase_vel = phase_ord / steps

        # Store in history
        self._episode_history.append({
            "reward": reward,
            "reward_per_step": reward_per_step,
            "disc_per_step": disc_per_step,
            "phase_ord": phase_ord,
            "phase_vel": phase_vel,
            "sac_agreement": sac_agreement,
        })
        if len(self._episode_history) > 20:
            self._episode_history.pop(0)

        # Warmup: return neutral adjustments
        if self._episodes_seen < self.config.warmup_episodes:
            return self._neutral_adjustments("warmup")

        # Compute trends over performance window
        window = self._episode_history[-self.config.performance_window:]
        self._reward_trend = self._compute_trend([e["reward"] for e in window])
        self._discovery_efficiency = sum(e["disc_per_step"] for e in window) / len(window)
        self._phase_velocity = sum(e["phase_vel"] for e in window) / len(window)

        # Detect stagnation
        if len(window) >= 2 and window[-1]["reward"] <= window[-2]["reward"] * 1.05:
            self._stagnation_streak += 1
        else:
            self._stagnation_streak = 0

        # --- Adaptation Logic ---
        alpha = self.config.adaptation_rate

        # 1. Entropy: increase when stagnating, decrease when converging
        if self._stagnation_streak >= 3:
            # Stuck → boost exploration
            entropy = self._lerp(self.config.entropy_range[1], 0.7, alpha)
            signal = "EXPLORE (stagnation)"
        elif self._reward_trend > 0.1:
            # Improving → reduce entropy to exploit
            entropy = self._lerp(self.config.entropy_range[0], 0.3, alpha)
            signal = "EXPLOIT (improving)"
        elif self._discovery_efficiency < 0.05:
            # Low discovery rate → moderate exploration boost
            entropy = self._lerp(self.config.entropy_range[1], 0.5, alpha)
            signal = "EXPLORE (low discovery)"
        else:
            entropy = self._lerp(0.01, 0.5, alpha)  # Balanced
            signal = "BALANCED"

        # 2. Learning rate: scale up when stable, down when volatile
        reward_variance = self._compute_variance([e["reward"] for e in window])
        if reward_variance < 100:
            lr_scale = 1.0 + 0.3 * alpha  # Stable → learn faster
        elif reward_variance > 1000:
            lr_scale = 1.0 - 0.3 * alpha  # Volatile → learn slower
        else:
            lr_scale = 1.0

        # 3. DDQN epsilon: higher when macro choices seem random
        if sac_agreement < 0.3:
            # High disagreement → macros uncertain → explore more macros
            ddqn_eps = self._lerp(self.config.ddqn_epsilon_range[1], 0.6, alpha)
        elif phase_ord >= 5:
            # Late phases → reduce macro exploration
            ddqn_eps = self._lerp(self.config.ddqn_epsilon_range[0], 0.3, alpha)
        else:
            ddqn_eps = self._lerp(0.15, 0.5, alpha)

        # 4. RND scale: boost early, reduce once discoveries flowing
        if self._discovery_efficiency > 0.15:
            # Plenty of discoveries → don't need as much curiosity
            rnd_scale = self._lerp(self.config.rnd_scale_range[0], 0.3, alpha)
        elif self._stagnation_streak >= 2:
            # Stagnating → boost curiosity
            rnd_scale = self._lerp(self.config.rnd_scale_range[1], 0.7, alpha)
        else:
            rnd_scale = 1.0

        adjustments = {
            "entropy_coef": max(self.config.entropy_range[0],
                               min(self.config.entropy_range[1], entropy)),
            "lr_scale": max(0.5, min(2.0, lr_scale)),
            "ddqn_epsilon": max(self.config.ddqn_epsilon_range[0],
                                min(self.config.ddqn_epsilon_range[1], ddqn_eps)),
            "rnd_scale": max(self.config.rnd_scale_range[0],
                             min(self.config.rnd_scale_range[1], rnd_scale)),
            "meta_signal": signal,
        }

        self._adjustments_history.append(adjustments)
        logger.info(
            "MetaRL adapt ep%d: %s | H=%.4f lr×%.2f ε=%.2f rnd=%.1f | "
            "trend=%.2f disc_eff=%.3f stag=%d",
            self._episodes_seen, signal,
            adjustments["entropy_coef"], adjustments["lr_scale"],
            adjustments["ddqn_epsilon"], adjustments["rnd_scale"],
            self._reward_trend, self._discovery_efficiency,
            self._stagnation_streak,
        )
        return adjustments

    def _neutral_adjustments(self, reason: str) -> Dict[str, float]:
        """Return no-change adjustments."""
        return {
            "entropy_coef": 0.01,
            "lr_scale": 1.0,
            "ddqn_epsilon": 0.15,
            "rnd_scale": 1.0,
            "meta_signal": f"NEUTRAL ({reason})",
        }

    @staticmethod
    def _lerp(target: float, blend: float, alpha: float) -> float:
        """Linear interpolation toward target with adaptation rate."""
        return target * blend * alpha + target * (1.0 - alpha)

    @staticmethod
    def _compute_trend(values: List[float]) -> float:
        """Compute normalized trend (slope) of a value series."""
        if len(values) < 2:
            return 0.0
        n = len(values)
        x_mean = (n - 1) / 2.0
        y_mean = sum(values) / n
        num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        den = sum((i - x_mean) ** 2 for i in range(n))
        if den < 1e-8:
            return 0.0
        slope = num / den
        # Normalize by mean to get relative trend
        return slope / max(abs(y_mean), 1.0)

    @staticmethod
    def _compute_variance(values: List[float]) -> float:
        """Compute variance of a value series."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / (len(values) - 1)

    def get_stats(self) -> Dict[str, Any]:
        """Return meta-RL controller statistics."""
        return {
            "episodes_seen": self._episodes_seen,
            "reward_trend": round(self._reward_trend, 4),
            "discovery_efficiency": round(self._discovery_efficiency, 4),
            "phase_velocity": round(self._phase_velocity, 4),
            "stagnation_streak": self._stagnation_streak,
            "last_adjustment": self._adjustments_history[-1] if self._adjustments_history else None,
        }
