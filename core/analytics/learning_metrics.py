#!/usr/bin/env python3
"""
core/analytics/learning_metrics.py — P34-EXT: Learning Process Metrics Collector

Tracks per-step and per-episode learning metrics for live dashboard display
and JSONL persistence. Computes windowed aggregates, milestones, and model mix.

All metrics are designed to be reward-invariant — they measure real learning
progress independent of reward scaling.

Author: P34-EXT
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("ariaska.analytics.learning_metrics")


# ── Dataclasses ─────────────────────────────────────────────────────────────

@dataclass
class MilestoneTracker:
    """Track time-to-milestone in steps."""
    first_port: int = -1
    first_service: int = -1
    first_creds: int = -1
    first_foothold: int = -1   # shell_obtained
    user_flag: int = -1
    root_flag: int = -1

    def record(self, step: int, discovery_type: str) -> Optional[str]:
        """Record a milestone if it's new. Returns milestone name or None."""
        mapping = {
            "port": "first_port",
            "service": "first_service",
            "credential": "first_creds",
            "shell": "first_foothold",
            "user_flag": "user_flag",
            "root_flag": "root_flag",
        }
        attr = mapping.get(discovery_type)
        if attr and getattr(self, attr) == -1:
            setattr(self, attr, step)
            return attr
        return None

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


@dataclass
class ModelMix:
    """Per-model call/token/cost tracking."""
    calls: Dict[str, int] = field(default_factory=lambda: {"codex": 0, "mini": 0, "nano": 0})
    tokens: Dict[str, int] = field(default_factory=lambda: {"codex": 0, "mini": 0, "nano": 0})
    cost: Dict[str, float] = field(default_factory=lambda: {"codex": 0.0, "mini": 0.0, "nano": 0.0})
    cache_hits: int = 0
    cache_misses: int = 0

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / max(total, 1)

    @property
    def total_cost(self) -> float:
        return sum(self.cost.values())

    def record_call(self, tier: str, tokens: int = 0, cost: float = 0.0, cached: bool = False) -> None:
        tier_key = tier if tier in self.calls else "mini"
        if cached:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
            self.calls[tier_key] = self.calls.get(tier_key, 0) + 1
            self.tokens[tier_key] = self.tokens.get(tier_key, 0) + tokens
            self.cost[tier_key] = self.cost.get(tier_key, 0.0) + cost

    def to_dict(self) -> Dict[str, Any]:
        return {
            "calls": dict(self.calls),
            "tokens": dict(self.tokens),
            "cost": {k: round(v, 6) for k, v in self.cost.items()},
            "cache_hit_rate": round(self.cache_hit_rate, 3),
            "total_cost": round(self.total_cost, 6),
        }


@dataclass
class EvidenceGateStats:
    """Track evidence gate pass/reject decisions."""
    passed: int = 0
    log_rejected: int = 0
    enforce_rejected: int = 0

    @property
    def total(self) -> int:
        return self.passed + self.log_rejected + self.enforce_rejected

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


@dataclass
class StepSnapshot:
    """Snapshot of learning metrics at a single step."""
    step: int = 0
    episode: int = 0
    timestamp: float = 0.0

    # Discovery deltas (this step)
    new_ports: int = 0
    new_services: int = 0
    new_paths: int = 0
    new_users: int = 0
    new_creds: int = 0
    new_shells: int = 0

    # Cumulative
    total_ports: int = 0
    total_services: int = 0
    total_paths: int = 0
    total_creds: int = 0
    total_shells: int = 0

    # Learning quality
    novelty_rate: float = 0.0      # unique_templates / total_commands
    anti_repeat_blocks: int = 0     # this step
    stagnation_steps: int = 0       # current consecutive steps without discovery
    phase_name: str = ""
    phase_changes: int = 0          # total phase changes this episode

    # Decision
    decision_source: str = ""
    template_name: str = ""
    command: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WindowMetrics:
    """Aggregated metrics over a sliding window (default 5 steps)."""
    window_size: int = 5
    discoveries_delta: int = 0
    novelty_rate: float = 0.0
    anti_repeat_rate: float = 0.0
    stagnation_avg: float = 0.0
    phase_thrash: int = 0
    cost_per_discovery: float = 0.0
    cost_this_window: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Main Collector ──────────────────────────────────────────────────────────

class LearningMetrics:
    """
    Collects and aggregates learning metrics for live dashboard + JSONL logging.

    Usage:
        metrics = LearningMetrics(log_dir="runs/20260218_soulmate")
        metrics.reset_episode(episode_id=0)
        # ... per step:
        metrics.record_step(step, discovery_board, decision, ...)
        snapshot = metrics.get_step_snapshot(step)
        window = metrics.get_window_metrics()
        # ... end of episode:
        summary = metrics.get_episode_summary()
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        window_size: int = 5,
        print_every: int = 5,
    ) -> None:
        self.log_dir = log_dir
        self.window_size = window_size
        self.print_every = print_every

        # JSONL file handle
        self._log_handle: Optional[Any] = None
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "learning_metrics.jsonl")
            self._log_handle = open(log_path, "a")  # noqa: SIM115

        # Per-episode state
        self._episode: int = 0
        self._step: int = 0
        self._snapshots: List[StepSnapshot] = []
        self._milestones = MilestoneTracker()
        self._model_mix = ModelMix()
        self._evidence_gate = EvidenceGateStats()
        self._mentor_interventions: List[Dict[str, str]] = []

        # Running state
        self._prev_discovery_board: Dict[str, int] = {}
        self._unique_templates: Set[str] = set()
        self._total_commands: int = 0
        self._anti_repeat_total: int = 0
        self._stagnation_counter: int = 0
        self._phase_changes: int = 0
        self._last_phase: str = ""
        self._phase_history: List[str] = []

        # Cost tracking
        self._episode_cost: float = 0.0

    def reset_episode(self, episode_id: int = 0) -> None:
        """Reset all per-episode tracking."""
        self._episode = episode_id
        self._step = 0
        self._snapshots.clear()
        self._milestones = MilestoneTracker()
        self._model_mix = ModelMix()
        self._evidence_gate = EvidenceGateStats()
        self._mentor_interventions.clear()
        self._prev_discovery_board = {}
        self._unique_templates = set()
        self._total_commands = 0
        self._anti_repeat_total = 0
        self._stagnation_counter = 0
        self._phase_changes = 0
        self._last_phase = ""
        self._phase_history.clear()
        self._episode_cost = 0.0

    def record_step(
        self,
        step: int,
        discovery_board: Dict[str, Any],
        decision_source: str = "",
        template_name: str = "",
        command: str = "",
        phase: str = "",
        anti_repeat_blocked: bool = False,
        model_tier: str = "",
        tokens_used: int = 0,
        cost_usd: float = 0.0,
        cached: bool = False,
    ) -> StepSnapshot:
        """Record metrics for a single step."""
        self._step = step
        self._total_commands += 1

        # Track templates
        if template_name:
            self._unique_templates.add(template_name)

        # Anti-repeat
        if anti_repeat_blocked:
            self._anti_repeat_total += 1

        # Phase tracking
        if phase and phase != self._last_phase:
            if self._last_phase:
                self._phase_changes += 1
            self._last_phase = phase
            self._phase_history.append(phase)

        # Compute discovery deltas
        new_ports = 0
        new_services = 0
        new_paths = 0
        new_users = 0
        new_creds = 0
        new_shells = 0

        current_counts = {
            "ports": len(discovery_board.get("ports", set())),
            "services": len(discovery_board.get("services", set())),
            "web_paths": len(discovery_board.get("web_paths", set())),
            "users": len(discovery_board.get("users", set())),
            "credentials": len(discovery_board.get("credentials", set())),
            "shells": len(discovery_board.get("shells", set())),
        }

        if self._prev_discovery_board:
            new_ports = max(0, current_counts["ports"] - self._prev_discovery_board.get("ports", 0))
            new_services = max(0, current_counts["services"] - self._prev_discovery_board.get("services", 0))
            new_paths = max(0, current_counts["web_paths"] - self._prev_discovery_board.get("web_paths", 0))
            new_users = max(0, current_counts["users"] - self._prev_discovery_board.get("users", 0))
            new_creds = max(0, current_counts["credentials"] - self._prev_discovery_board.get("credentials", 0))
            new_shells = max(0, current_counts["shells"] - self._prev_discovery_board.get("shells", 0))

        total_new = new_ports + new_services + new_creds + new_shells + new_users
        self._prev_discovery_board = dict(current_counts)

        # Stagnation
        if total_new > 0:
            self._stagnation_counter = 0
        else:
            self._stagnation_counter += 1

        # Milestone tracking
        if new_ports > 0:
            self._milestones.record(step, "port")
        if new_services > 0:
            self._milestones.record(step, "service")
        if new_creds > 0:
            self._milestones.record(step, "credential")
        if new_shells > 0:
            self._milestones.record(step, "shell")
        # Flags checked via discovery_board flags_set
        flags = discovery_board.get("flags_set", set())
        if "user_flag" in flags or "user_flag_captured" in flags:
            self._milestones.record(step, "user_flag")
        if "root_flag" in flags or "root_flag_captured" in flags:
            self._milestones.record(step, "root_flag")

        # Model mix
        if model_tier:
            self._model_mix.record_call(model_tier, tokens_used, cost_usd, cached)
            self._episode_cost += cost_usd

        # Novelty
        novelty = len(self._unique_templates) / max(self._total_commands, 1)

        snapshot = StepSnapshot(
            step=step,
            episode=self._episode,
            timestamp=time.time(),
            new_ports=new_ports,
            new_services=new_services,
            new_paths=new_paths,
            new_users=new_users,
            new_creds=new_creds,
            new_shells=new_shells,
            total_ports=current_counts["ports"],
            total_services=current_counts["services"],
            total_paths=current_counts["web_paths"],
            total_creds=current_counts["credentials"],
            total_shells=current_counts["shells"],
            novelty_rate=novelty,
            anti_repeat_blocks=1 if anti_repeat_blocked else 0,
            stagnation_steps=self._stagnation_counter,
            phase_name=phase,
            phase_changes=self._phase_changes,
            decision_source=decision_source,
            template_name=template_name,
            command=command[:120],
        )
        self._snapshots.append(snapshot)

        # Write to JSONL
        if self._log_handle:
            try:
                record = {
                    "type": "step_metrics",
                    **snapshot.to_dict(),
                }
                self._log_handle.write(json.dumps(record) + "\n")
                self._log_handle.flush()
            except Exception as e:
                logger.debug(f"Failed to write step metrics: {e}")

        return snapshot

    def record_evidence_gate(self, verdict: str) -> None:
        """Record an evidence gate decision: 'pass', 'log_reject', 'enforce_reject'."""
        if verdict == "pass":
            self._evidence_gate.passed += 1
        elif verdict == "log_reject":
            self._evidence_gate.log_rejected += 1
        elif verdict == "enforce_reject":
            self._evidence_gate.enforce_rejected += 1

    def record_mentor_intervention(self, reason: str, agent: str = "") -> None:
        """Record a mentor intervention with its reason."""
        self._mentor_interventions.append({
            "step": self._step,
            "reason": reason,
            "agent": agent,
            "timestamp": time.time(),
        })

    def get_step_snapshot(self, step: int) -> Optional[StepSnapshot]:
        """Get snapshot for a specific step."""
        for s in reversed(self._snapshots):
            if s.step == step:
                return s
        return None

    def get_window_metrics(self) -> WindowMetrics:
        """Compute windowed metrics over last N steps."""
        window = self._snapshots[-self.window_size:] if self._snapshots else []
        if not window:
            return WindowMetrics(window_size=self.window_size)

        # Discovery delta
        discoveries = sum(
            s.new_ports + s.new_services + s.new_creds + s.new_shells + s.new_users
            for s in window
        )

        # Novelty
        novelty = window[-1].novelty_rate if window else 0.0

        # Anti-repeat rate
        ar_blocks = sum(s.anti_repeat_blocks for s in window)
        ar_rate = ar_blocks / max(len(window), 1)

        # Stagnation avg
        stag_avg = sum(s.stagnation_steps for s in window) / max(len(window), 1)

        # Phase thrash (unique phases in window)
        phases = [s.phase_name for s in window if s.phase_name]
        phase_thrash = 0
        for i in range(1, len(phases)):
            if phases[i] != phases[i - 1]:
                phase_thrash += 1

        # Cost per discovery
        cost_window = self._model_mix.total_cost  # approximate from episode total
        cpd = cost_window / max(discoveries, 1) if discoveries > 0 else 0.0

        return WindowMetrics(
            window_size=len(window),
            discoveries_delta=discoveries,
            novelty_rate=round(novelty, 3),
            anti_repeat_rate=round(ar_rate, 3),
            stagnation_avg=round(stag_avg, 1),
            phase_thrash=phase_thrash,
            cost_per_discovery=round(cpd, 6),
            cost_this_window=round(cost_window, 6),
        )

    def get_episode_summary(self) -> Dict[str, Any]:
        """Get full episode summary metrics."""
        total_discoveries = sum(
            s.new_ports + s.new_services + s.new_creds + s.new_shells + s.new_users
            for s in self._snapshots
        )
        summary = {
            "episode": self._episode,
            "total_steps": len(self._snapshots),
            "total_discoveries": total_discoveries,
            "unique_templates": len(self._unique_templates),
            "novelty_rate": round(len(self._unique_templates) / max(self._total_commands, 1), 3),
            "anti_repeat_total": self._anti_repeat_total,
            "anti_repeat_rate": round(self._anti_repeat_total / max(self._total_commands, 1), 3),
            "max_stagnation": max((s.stagnation_steps for s in self._snapshots), default=0),
            "phase_changes": self._phase_changes,
            "phase_history": self._phase_history,
            "milestones": self._milestones.to_dict(),
            "model_mix": self._model_mix.to_dict(),
            "evidence_gate": self._evidence_gate.to_dict(),
            "mentor_interventions": len(self._mentor_interventions),
            "mentor_reasons": self._summarize_mentor_reasons(),
            "episode_cost": round(self._episode_cost, 6),
            "cost_per_discovery": round(
                self._episode_cost / max(total_discoveries, 1), 6
            ) if total_discoveries > 0 else 0.0,
        }

        # Write to JSONL
        if self._log_handle:
            try:
                record = {"type": "episode_summary", **summary}
                self._log_handle.write(json.dumps(record) + "\n")
                self._log_handle.flush()
            except Exception:
                pass

        return summary

    def _summarize_mentor_reasons(self) -> Dict[str, int]:
        """Summarize mentor intervention reasons."""
        reasons: Dict[str, int] = {}
        for entry in self._mentor_interventions:
            r = entry.get("reason", "unknown")
            reasons[r] = reasons.get(r, 0) + 1
        return reasons

    def should_print_dashboard(self, step: int) -> bool:
        """Whether to print the learning dashboard at this step."""
        return step > 0 and step % self.print_every == 0

    @property
    def stagnation_steps(self) -> int:
        return self._stagnation_counter

    @property
    def milestones(self) -> MilestoneTracker:
        return self._milestones

    @property
    def model_mix(self) -> ModelMix:
        return self._model_mix

    @property
    def evidence_gate(self) -> EvidenceGateStats:
        return self._evidence_gate

    @property
    def anti_repeat_total(self) -> int:
        return self._anti_repeat_total

    @property
    def total_commands(self) -> int:
        return self._total_commands

    @property
    def unique_template_count(self) -> int:
        return len(self._unique_templates)

    @property
    def phase_changes(self) -> int:
        return self._phase_changes

    def close(self) -> None:
        """Close JSONL file handle."""
        if self._log_handle:
            try:
                self._log_handle.close()
            except Exception:
                pass
            self._log_handle = None
