"""
core/decision/harmony_metrics.py — Phase 7: Orchestration Harmony Metrics.

Tracks 8 mandatory metrics for system health monitoring.
All metrics must log to TensorBoard (when available) and to ariaska logs.

Metrics:
  1. decision_source_entropy — Shannon entropy of source hit distribution
  2. arbitration_weight_distribution — Weight applied per source per step
  3. entropy_writer_count — MUST be 1 after consolidation
  4. gradient_norm_per_loss_component — Per-loss-component grad norms
  5. macro_switch_rate — How often DDQN macro switches per episode
  6. mentor_dependence_curve — Mentor-source ratio over training
  7. reward_variance — Running reward variance across steps
  8. kl_drift_per_episode — KL divergence between teacher and student

Author: Phase 7 — Harmony Metrics
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ariaska.decision.harmony")


@dataclass
class HarmonySnapshot:
    """Point-in-time snapshot of all harmony metrics."""

    episode: int = 0
    step: int = 0

    # 1. Decision source entropy (from DecisionCore hit distribution)
    decision_source_entropy: float = 0.0

    # 2. Arbitration weight distribution (per-source weights this step)
    arbitration_weights: Dict[str, float] = field(default_factory=dict)

    # 3. Entropy writer count (MUST be 1)
    entropy_writer_count: int = 0

    # 4. Gradient norm per loss component
    gradient_norms: Dict[str, float] = field(default_factory=dict)

    # 5. Macro switch rate (DDQN macro changes / total steps)
    macro_switch_rate: float = 0.0

    # 6. Mentor dependence (mentor-source decisions / total decisions)
    mentor_dependence: float = 0.0

    # 7. Reward variance (running Welford)
    reward_variance: float = 0.0

    # 8. KL drift (teacher -> student per-episode average)
    kl_drift: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode": self.episode,
            "step": self.step,
            "decision_source_entropy": round(self.decision_source_entropy, 4),
            "arbitration_weights": {
                k: round(v, 4) for k, v in self.arbitration_weights.items()
            },
            "entropy_writer_count": self.entropy_writer_count,
            "gradient_norms": {
                k: round(v, 6) for k, v in self.gradient_norms.items()
            },
            "macro_switch_rate": round(self.macro_switch_rate, 4),
            "mentor_dependence": round(self.mentor_dependence, 4),
            "reward_variance": round(self.reward_variance, 4),
            "kl_drift": round(self.kl_drift, 6),
        }


class HarmonyMetrics:
    """Tracks orchestration harmony metrics across training.

    Usage:
        metrics = HarmonyMetrics()
        # After each step:
        metrics.record_decision(source, arbitration_weights)
        metrics.record_gradient_norms(norms_dict)
        metrics.record_macro_switch(switched)
        metrics.record_reward(reward)
        metrics.record_kl_drift(kl_value)

        # Get snapshot:
        snap = metrics.snapshot(episode, step)

        # TensorBoard:
        metrics.log_to_tensorboard(writer, global_step)
    """

    def __init__(self) -> None:
        # 1. Source hit counts
        self._source_hits: Dict[str, int] = defaultdict(int)
        self._total_decisions: int = 0

        # 2. Latest arbitration weights
        self._latest_weights: Dict[str, float] = {}

        # 3. Entropy writer tracking
        self._entropy_writers: set = set()

        # 4. Gradient norms
        self._latest_grad_norms: Dict[str, float] = {}

        # 5. Macro switches
        self._macro_switches: int = 0
        self._macro_steps: int = 0
        self._last_macro: Optional[str] = None

        # 6. Mentor hits
        self._mentor_decisions: int = 0

        # 7. Reward tracking (Welford)
        self._reward_count: int = 0
        self._reward_mean: float = 0.0
        self._reward_m2: float = 0.0

        # 8. KL drift
        self._kl_values: List[float] = []
        self._episode_kl_sum: float = 0.0
        self._episode_kl_count: int = 0

    # ── 1. Decision source ──────────────────────────────────────────

    def record_decision(
        self,
        source: str,
        arbitration_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """Record a decision from a source."""
        self._source_hits[source] += 1
        self._total_decisions += 1

        if source in ("mentor", "dual_mentor", "codex_meta"):
            self._mentor_decisions += 1

        if arbitration_weights is not None:
            self._latest_weights = dict(arbitration_weights)

    def get_decision_source_entropy(self) -> float:
        """Shannon entropy of source hit distribution."""
        if self._total_decisions == 0:
            return 0.0
        entropy = 0.0
        for count in self._source_hits.values():
            p = count / self._total_decisions
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    def get_source_distribution(self) -> Dict[str, float]:
        """Normalized source hit distribution."""
        if self._total_decisions == 0:
            return {}
        return {
            src: count / self._total_decisions
            for src, count in self._source_hits.items()
        }

    # ── 2. Arbitration weights ──────────────────────────────────────

    @property
    def latest_arbitration_weights(self) -> Dict[str, float]:
        return dict(self._latest_weights)

    # ── 3. Entropy writer count ─────────────────────────────────────

    def register_entropy_writer(self, writer_name: str) -> None:
        """Register an entropy writer.  After consolidation this MUST be 1."""
        self._entropy_writers.add(writer_name)
        if len(self._entropy_writers) > 1:
            logger.warning(
                "HARMONY VIOLATION: %d entropy writers active: %s",
                len(self._entropy_writers),
                self._entropy_writers,
            )

    @property
    def entropy_writer_count(self) -> int:
        return len(self._entropy_writers)

    # ── 4. Gradient norms ───────────────────────────────────────────

    def record_gradient_norms(self, norms: Dict[str, float]) -> None:
        """Record per-loss-component gradient norms from PPO update."""
        self._latest_grad_norms = dict(norms)

    @property
    def latest_gradient_norms(self) -> Dict[str, float]:
        return dict(self._latest_grad_norms)

    # ── 5. Macro switch rate ────────────────────────────────────────

    def record_macro_step(self, current_macro: str) -> None:
        """Track DDQN macro switches."""
        self._macro_steps += 1
        if self._last_macro is not None and current_macro != self._last_macro:
            self._macro_switches += 1
        self._last_macro = current_macro

    @property
    def macro_switch_rate(self) -> float:
        if self._macro_steps < 2:
            return 0.0
        return self._macro_switches / (self._macro_steps - 1)

    # ── 6. Mentor dependence curve ──────────────────────────────────

    @property
    def mentor_dependence(self) -> float:
        if self._total_decisions == 0:
            return 0.0
        return self._mentor_decisions / self._total_decisions

    # ── 7. Reward variance ──────────────────────────────────────────

    def record_reward(self, reward: float) -> None:
        """Welford's online variance."""
        self._reward_count += 1
        delta = reward - self._reward_mean
        self._reward_mean += delta / self._reward_count
        self._reward_m2 += delta * (reward - self._reward_mean)

    @property
    def reward_variance(self) -> float:
        if self._reward_count < 2:
            return 0.0
        return self._reward_m2 / (self._reward_count - 1)

    # ── 8. KL drift per episode ─────────────────────────────────────

    def record_kl_drift(self, kl_value: float) -> None:
        """Record KL divergence between teacher and student."""
        self._episode_kl_sum += kl_value
        self._episode_kl_count += 1

    def end_episode_kl(self) -> float:
        """Get average KL for this episode and reset accumulator."""
        if self._episode_kl_count == 0:
            avg = 0.0
        else:
            avg = self._episode_kl_sum / self._episode_kl_count
        self._kl_values.append(avg)
        self._episode_kl_sum = 0.0
        self._episode_kl_count = 0
        return avg

    @property
    def latest_kl_drift(self) -> float:
        if not self._kl_values:
            return 0.0
        return self._kl_values[-1]

    # ── Snapshot ────────────────────────────────────────────────────

    def snapshot(self, episode: int = 0, step: int = 0) -> HarmonySnapshot:
        """Create point-in-time snapshot of all harmony metrics."""
        return HarmonySnapshot(
            episode=episode,
            step=step,
            decision_source_entropy=self.get_decision_source_entropy(),
            arbitration_weights=dict(self._latest_weights),
            entropy_writer_count=len(self._entropy_writers),
            gradient_norms=dict(self._latest_grad_norms),
            macro_switch_rate=self.macro_switch_rate,
            mentor_dependence=self.mentor_dependence,
            reward_variance=self.reward_variance,
            kl_drift=self.latest_kl_drift,
        )

    # ── TensorBoard ─────────────────────────────────────────────────

    def log_to_tensorboard(
        self,
        writer: Any,
        global_step: int,
    ) -> None:
        """Log all 8 metrics to TensorBoard SummaryWriter."""
        if writer is None:
            return

        try:
            prefix = "harmony"

            # 1. Decision source entropy
            writer.add_scalar(
                f"{prefix}/decision_source_entropy",
                self.get_decision_source_entropy(),
                global_step,
            )

            # 2. Arbitration weights
            for src, w in self._latest_weights.items():
                writer.add_scalar(
                    f"{prefix}/arb_weight/{src}", w, global_step
                )

            # 3. Entropy writer count
            writer.add_scalar(
                f"{prefix}/entropy_writer_count",
                len(self._entropy_writers),
                global_step,
            )

            # 4. Gradient norms per component
            for comp, norm in self._latest_grad_norms.items():
                writer.add_scalar(
                    f"{prefix}/grad_norm/{comp}", norm, global_step
                )

            # 5. Macro switch rate
            writer.add_scalar(
                f"{prefix}/macro_switch_rate",
                self.macro_switch_rate,
                global_step,
            )

            # 6. Mentor dependence
            writer.add_scalar(
                f"{prefix}/mentor_dependence",
                self.mentor_dependence,
                global_step,
            )

            # 7. Reward variance
            writer.add_scalar(
                f"{prefix}/reward_variance",
                self.reward_variance,
                global_step,
            )

            # 8. KL drift
            writer.add_scalar(
                f"{prefix}/kl_drift", self.latest_kl_drift, global_step
            )

            # Source distribution histogram
            dist = self.get_source_distribution()
            for src, pct in dist.items():
                writer.add_scalar(
                    f"{prefix}/source_pct/{src}", pct, global_step
                )

        except Exception as e:
            logger.debug("TensorBoard logging failed: %s", e)

    # ── Reset ────────────────────────────────────────────────────────

    def reset_episode(self) -> None:
        """Reset per-episode accumulators. Cross-episode data kept."""
        self._macro_switches = 0
        self._macro_steps = 0
        self._last_macro = None

    def to_dict(self) -> Dict[str, Any]:
        """Full state for serialization/debugging."""
        return {
            "source_hits": dict(self._source_hits),
            "total_decisions": self._total_decisions,
            "entropy_writers": list(self._entropy_writers),
            "decision_source_entropy": round(
                self.get_decision_source_entropy(), 4
            ),
            "macro_switch_rate": round(self.macro_switch_rate, 4),
            "mentor_dependence": round(self.mentor_dependence, 4),
            "reward_variance": round(self.reward_variance, 4),
            "latest_kl_drift": round(self.latest_kl_drift, 6),
        }
