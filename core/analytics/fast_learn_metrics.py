#!/usr/bin/env python3
"""
core/analytics/fast_learn_metrics.py — Phase 36.1: Fast-Learn Metrics Logger

Tracks per-step learning quality metrics and persists them to JSONL.
Records value predictions, TD errors, advantage stats, distillation
quality, hallucination counts, contradiction counts, micro-chain
success rate, and model call mix every N steps.

Also implements the Budget Governor: adaptive cost control that
targets <= $3 per 40 steps, boosting learning-acceleration budget
when improvement is detected, and tightening when stalled.

Author: Phase 36.1 — Fast-Learn Mode
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger("ariaska.analytics.fast_learn_metrics")

# ── Constants ───────────────────────────────────────────────────────────────

_LOG_EVERY_N = 5                    # Persist metrics every N steps
_BUDGET_WINDOW = 40                 # Budget governor evaluation window
_BUDGET_TARGET_USD = 3.00           # Target max spend per 40 steps
_IMPROVEMENT_BOOST = 0.30           # +30% learning-accel on improvement
_MODEL_MIX_WINDOW = 50              # Rolling window for model mix %
_MC_SUCCESS_WINDOW = 20             # Rolling window for micro-chain success

# Learning component ROI tags that get doubled budget
LEARNING_ROI_TAGS = frozenset({
    "value_target_shaping",
    "advantage_sanity",
    "reward_decomposition",
    "distillation_quality",
    "phase_decision_scoring",
    "parser_feedback_loop",
    "micro_chain_scoring",
    "micro_chain_generation",
    "micro_chain_classification",
})


# ── Dataclasses ─────────────────────────────────────────────────────────────

@dataclass
class FastLearnSnapshot:
    """Per-step learning quality snapshot."""
    step_id: int = 0
    episode: int = 0
    timestamp: float = 0.0
    phase: str = ""
    chosen_template: str = ""

    # PPO learning signals
    value_pred: float = 0.0
    value_target: float = 0.0
    td_error: float = 0.0
    advantage_mean: float = 0.0
    advantage_std: float = 0.0

    # Quality metrics
    distillation_packet_score: float = 0.0
    distillation_reason: str = ""
    hallucination_flags_count: int = 0
    contradictions_count: int = 0

    # Micro-chain
    micro_chain_success: bool = False
    micro_chain_success_rate: float = 0.0  # rolling window

    # Model mix (rolling %)
    model_nano_pct: float = 0.0
    model_mini_pct: float = 0.0
    model_codex_pct: float = 0.0

    # Budget
    step_cost_usd: float = 0.0
    cumulative_cost_usd: float = 0.0
    budget_pressure: float = 0.0

    # Governor state
    governor_learn_boost: float = 1.0  # 1.0 = normal, 1.3 = boosted
    governor_improvement: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BudgetGovernorState:
    """Tracks the budget governor's adaptive state."""
    window_start_step: int = 0
    window_cost_usd: float = 0.0
    learn_boost_factor: float = 1.0

    # Improvement signals (tracked over last window)
    prev_mc_success_rate: float = 0.0
    prev_contradiction_rate: float = 0.0
    prev_hallucination_rate: float = 0.0
    prev_distillation_score: float = 0.0
    prev_stagnation_steps: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Model Call Tracker ──────────────────────────────────────────────────────

class ModelCallTracker:
    """Track model usage mix with rolling percentages."""

    def __init__(self, window: int = _MODEL_MIX_WINDOW) -> None:
        self._window = window
        self._calls: Deque[str] = deque(maxlen=window)
        self._counts: Dict[str, int] = {"nano": 0, "mini": 0, "codex": 0, "full": 0}
        self._total: int = 0
        self._lock = threading.Lock()

    def record(self, tier: str) -> None:
        """Record a model call by tier."""
        tier_key = tier if tier in self._counts else "mini"
        with self._lock:
            self._calls.append(tier_key)
            self._counts[tier_key] = self._counts.get(tier_key, 0) + 1
            self._total += 1

    def get_mix(self) -> Dict[str, float]:
        """Get model mix percentages from rolling window."""
        with self._lock:
            n = len(self._calls)
            if n == 0:
                return {"nano": 0.0, "mini": 0.0, "codex": 0.0}
            counts: Dict[str, int] = {}
            for tier in self._calls:
                counts[tier] = counts.get(tier, 0) + 1
            return {
                "nano": counts.get("nano", 0) / n,
                "mini": counts.get("mini", 0) / n,
                "codex": counts.get("codex", 0) / n,
            }

    def get_total_counts(self) -> Dict[str, int]:
        """Get total call counts by tier."""
        with self._lock:
            return dict(self._counts)

    @property
    def total(self) -> int:
        return self._total


# ── Fast-Learn Metrics Collector ────────────────────────────────────────────

class FastLearnMetrics:
    """
    Collects and persists per-step learning metrics for Fast-Learn Mode.

    Writes to JSONL every N steps. Implements the Budget Governor
    that adaptively boosts or tightens learning-acceleration budget.

    Thread-safe.

    Usage:
        flm = FastLearnMetrics(log_dir="runs")
        flm.reset_episode(episode=0)
        flm.record_step(step=5, phase="RECON", template="nmap_basic",
                        value_pred=0.3, reward=2.5)
        flm.record_model_call("nano")
        snapshot = flm.get_latest_snapshot()
        governor = flm.evaluate_governor(step=40)
    """

    def __init__(
        self,
        log_dir: str = "runs",
        log_every: int = _LOG_EVERY_N,
    ) -> None:
        self.log_dir = log_dir
        self.log_every = log_every
        self._lock = threading.Lock()

        # JSONL file
        os.makedirs(log_dir, exist_ok=True)
        self._log_path = os.path.join(log_dir, "learn_metrics.jsonl")
        self._log_handle: Optional[Any] = None
        try:
            self._log_handle = open(self._log_path, "a")  # noqa: SIM115
        except Exception as e:
            logger.warning(f"Failed to open learn_metrics.jsonl: {e}")

        # Per-episode state
        self._episode: int = 0
        self._snapshots: List[FastLearnSnapshot] = []
        self._model_tracker = ModelCallTracker()
        self._governor = BudgetGovernorState()
        self._cumulative_cost: float = 0.0

        # Rolling micro-chain success
        self._mc_results: Deque[bool] = deque(maxlen=_MC_SUCCESS_WINDOW)

        # Hallucination / contradiction counters
        self._hallucination_total: int = 0
        self._contradiction_total: int = 0
        self._distillation_scores: Deque[float] = deque(maxlen=20)

        # PPO stats cache (updated by orchestrator)
        self._last_value_pred: float = 0.0
        self._last_value_target: float = 0.0
        self._last_td_error: float = 0.0
        self._last_advantage_mean: float = 0.0
        self._last_advantage_std: float = 0.0

    def reset_episode(self, episode: int = 0) -> None:
        """Reset per-episode state."""
        with self._lock:
            self._episode = episode
            self._snapshots.clear()
            self._mc_results.clear()
            self._hallucination_total = 0
            self._contradiction_total = 0
            self._distillation_scores.clear()
            self._cumulative_cost = 0.0
            self._governor = BudgetGovernorState()
            self._model_tracker = ModelCallTracker()
            self._last_value_pred = 0.0
            self._last_value_target = 0.0
            self._last_td_error = 0.0
            self._last_advantage_mean = 0.0
            self._last_advantage_std = 0.0

    # ── Recording ───────────────────────────────────────────────────────

    def record_ppo_stats(
        self,
        value_pred: float = 0.0,
        value_target: float = 0.0,
        advantage_mean: float = 0.0,
        advantage_std: float = 0.0,
    ) -> None:
        """Update cached PPO stats (called by SmartCoach after PPO action)."""
        with self._lock:
            self._last_value_pred = value_pred
            self._last_value_target = value_target
            self._last_td_error = abs(value_target - value_pred)
            self._last_advantage_mean = advantage_mean
            self._last_advantage_std = advantage_std

    def record_model_call(self, tier: str) -> None:
        """Record a model call for mix tracking."""
        self._model_tracker.record(tier)

    def record_micro_chain_result(self, success: bool) -> None:
        """Record micro-chain success/failure."""
        with self._lock:
            self._mc_results.append(success)

    def record_hallucination(self, count: int = 1) -> None:
        """Record hallucination flag(s)."""
        with self._lock:
            self._hallucination_total += count

    def record_contradiction(self, count: int = 1) -> None:
        """Record contradiction(s)."""
        with self._lock:
            self._contradiction_total += count

    def record_distillation_score(self, score: float, reason: str = "") -> None:
        """Record a distillation packet quality score."""
        with self._lock:
            self._distillation_scores.append(max(0.0, min(1.0, score)))

    def record_step_cost(self, cost_usd: float) -> None:
        """Record cost for this step."""
        with self._lock:
            self._cumulative_cost += cost_usd

    # ── Step Recording ──────────────────────────────────────────────────

    def record_step(
        self,
        step: int,
        phase: str = "",
        template: str = "",
        value_pred: Optional[float] = None,
        reward: float = 0.0,
        step_cost: float = 0.0,
        hallucination_flags: int = 0,
        contradictions: int = 0,
        mc_success: Optional[bool] = None,
        distillation_score: float = 0.0,
        distillation_reason: str = "",
        budget_pressure: float = 0.0,
    ) -> Optional[FastLearnSnapshot]:
        """
        Record step-level metrics and auto-persist every N steps.

        Returns FastLearnSnapshot if persisted, else None.
        """
        with self._lock:
            self._cumulative_cost += step_cost
            self._governor.window_cost_usd += step_cost

            if hallucination_flags > 0:
                self._hallucination_total += hallucination_flags

            if contradictions > 0:
                self._contradiction_total += contradictions

            if mc_success is not None:
                self._mc_results.append(mc_success)

            if distillation_score > 0:
                self._distillation_scores.append(
                    max(0.0, min(1.0, distillation_score))
                )

            # Use provided value_pred or cached
            vp = value_pred if value_pred is not None else self._last_value_pred
            vt = self._last_value_target
            td = abs(vt - vp) if vt != 0 else self._last_td_error

            # Model mix
            mix = self._model_tracker.get_mix()

            # MC success rate
            mc_rate = (
                sum(1 for x in self._mc_results if x) / max(len(self._mc_results), 1)
                if self._mc_results else 0.0
            )

            # Avg distillation score
            avg_distill = (
                sum(self._distillation_scores) / max(len(self._distillation_scores), 1)
                if self._distillation_scores else 0.0
            )

            snapshot = FastLearnSnapshot(
                step_id=step,
                episode=self._episode,
                timestamp=time.time(),
                phase=phase,
                chosen_template=template,
                value_pred=round(vp, 4),
                value_target=round(vt, 4),
                td_error=round(td, 4),
                advantage_mean=round(self._last_advantage_mean, 4),
                advantage_std=round(self._last_advantage_std, 4),
                distillation_packet_score=round(avg_distill, 4),
                distillation_reason=distillation_reason[:80] if distillation_reason else "",
                hallucination_flags_count=self._hallucination_total,
                contradictions_count=self._contradiction_total,
                micro_chain_success=mc_success if mc_success is not None else False,
                micro_chain_success_rate=round(mc_rate, 4),
                model_nano_pct=round(mix.get("nano", 0), 4),
                model_mini_pct=round(mix.get("mini", 0), 4),
                model_codex_pct=round(mix.get("codex", 0), 4),
                step_cost_usd=round(step_cost, 6),
                cumulative_cost_usd=round(self._cumulative_cost, 6),
                budget_pressure=round(budget_pressure, 4),
                governor_learn_boost=self._governor.learn_boost_factor,
                governor_improvement=False,
            )

            self._snapshots.append(snapshot)

            # Persist every N steps
            if step > 0 and step % self.log_every == 0:
                self._write_snapshot(snapshot)
                return snapshot

            return None

    def _write_snapshot(self, snapshot: FastLearnSnapshot) -> None:
        """Write snapshot to JSONL (called with lock held)."""
        if not self._log_handle:
            return
        try:
            record = {"type": "fast_learn_step", **snapshot.to_dict()}
            self._log_handle.write(json.dumps(record) + "\n")
            self._log_handle.flush()
        except Exception as e:
            logger.debug(f"Failed to write fast_learn_metrics: {e}")

    # ── Budget Governor ─────────────────────────────────────────────────

    def evaluate_governor(self, step: int) -> BudgetGovernorState:
        """
        Evaluate budget governor at current step.

        Called every _BUDGET_WINDOW steps. Checks if learning metrics
        improved; if so, boosts learning-acceleration budget by +30%.
        If not, reduces codex usage and tightens.

        Returns updated BudgetGovernorState.
        """
        with self._lock:
            steps_in_window = step - self._governor.window_start_step
            if steps_in_window < _BUDGET_WINDOW:
                return self._governor

            # Current metrics
            mc_rate = (
                sum(1 for x in self._mc_results if x) / max(len(self._mc_results), 1)
                if self._mc_results else 0.0
            )
            n_steps = max(len(self._snapshots), 1)
            contradiction_rate = self._contradiction_total / n_steps
            hallucination_rate = self._hallucination_total / n_steps
            distill_score = (
                sum(self._distillation_scores) / max(len(self._distillation_scores), 1)
                if self._distillation_scores else 0.0
            )
            # Stagnation: average from recent snapshots
            recent = self._snapshots[-_BUDGET_WINDOW:] if self._snapshots else []
            # Not directly tracked here; use 0 as baseline

            # Count improvements (at least 2 of 5 must improve)
            improvements = 0
            if mc_rate > self._governor.prev_mc_success_rate + 0.05:
                improvements += 1
            if contradiction_rate < self._governor.prev_contradiction_rate - 0.02:
                improvements += 1
            if hallucination_rate < self._governor.prev_hallucination_rate - 0.02:
                improvements += 1
            if distill_score > self._governor.prev_distillation_score + 0.05:
                improvements += 1
            # Reduced stall: not tracked directly, count as neutral

            improved = improvements >= 2
            cost_ok = self._governor.window_cost_usd <= _BUDGET_TARGET_USD

            if improved:
                # Boost learning-acceleration allowance
                self._governor.learn_boost_factor = min(
                    2.0, self._governor.learn_boost_factor * (1.0 + _IMPROVEMENT_BOOST)
                )
                logger.info(
                    f"[GOVERNOR] Improvement detected ({improvements}/5 criteria). "
                    f"Learn boost → {self._governor.learn_boost_factor:.2f}"
                )
            elif not cost_ok:
                # Over budget without improvement → tighten
                self._governor.learn_boost_factor = max(
                    0.5, self._governor.learn_boost_factor * 0.85
                )
                logger.info(
                    f"[GOVERNOR] Over budget, no improvement. "
                    f"Learn boost → {self._governor.learn_boost_factor:.2f}"
                )

            # Store current as previous
            self._governor.prev_mc_success_rate = mc_rate
            self._governor.prev_contradiction_rate = contradiction_rate
            self._governor.prev_hallucination_rate = hallucination_rate
            self._governor.prev_distillation_score = distill_score
            self._governor.window_start_step = step
            self._governor.window_cost_usd = 0.0

            return self._governor

    # ── Accessors ───────────────────────────────────────────────────────

    def get_latest_snapshot(self) -> Optional[FastLearnSnapshot]:
        """Get the most recent snapshot."""
        with self._lock:
            return self._snapshots[-1] if self._snapshots else None

    def get_dashboard_line(self) -> str:
        """Build a compact dashboard line for CLI display.

        Format: FL: phase=RECON | V=0.32 TD=0.15 | MC=75% | nano:70% mini:25% codex:5% | $0.82/40
        """
        with self._lock:
            if not self._snapshots:
                return "FL: waiting..."
            s = self._snapshots[-1]
            mix = self._model_tracker.get_mix()
            mc_rate = (
                sum(1 for x in self._mc_results if x) / max(len(self._mc_results), 1)
                if self._mc_results else 0.0
            )
            return (
                f"FL: {s.phase[:6]} | V={s.value_pred:.2f} TD={s.td_error:.2f} "
                f"A={s.advantage_mean:+.2f}±{s.advantage_std:.2f} | "
                f"MC={mc_rate:.0%} H={self._hallucination_total} C={self._contradiction_total} "
                f"D={s.distillation_packet_score:.2f} | "
                f"n:{mix.get('nano', 0):.0%} m:{mix.get('mini', 0):.0%} "
                f"c:{mix.get('codex', 0):.0%} | "
                f"${self._cumulative_cost:.2f} "
                f"boost={self._governor.learn_boost_factor:.1f}x"
            )

    def get_model_mix(self) -> Dict[str, float]:
        """Get current model mix percentages."""
        return self._model_tracker.get_mix()

    def get_governor_state(self) -> BudgetGovernorState:
        """Get current governor state."""
        with self._lock:
            return self._governor

    @property
    def learn_boost_factor(self) -> float:
        """Current learning acceleration boost factor."""
        return self._governor.learn_boost_factor

    @property
    def cumulative_cost(self) -> float:
        return self._cumulative_cost

    def close(self) -> None:
        """Close JSONL handle."""
        if self._log_handle:
            try:
                self._log_handle.close()
            except Exception:
                pass
            self._log_handle = None
