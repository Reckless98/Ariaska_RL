"""
core/ops/engagement_metrics.py — Phase 38.4: Engagement-Level Metrics

Tracks engagement-level metrics that persist across episodes and
provide higher-level progress signals than episode-level metrics.

Combines reward-invariant metrics with engagement-specific tracking:
  - Phase progression velocity
  - Discovery efficiency over time
  - Flag capture timeline
  - Stagnation detection at engagement level
  - Tool usage diversity
  - Exploitation success rate

These metrics feed into TokenFlexEngine for dynamic budget adjustment
and into the LiveDashboard for operator visibility.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("ariaska.ops.engagement_metrics")

# ── Phase ordering for progress tracking ─────────────────────────────────────

_PHASE_ORDER = [
    "RECON", "ENUMERATION", "EXPLOITATION", "PRIVILEGE_ESCALATION",
    "LATERAL_MOVEMENT", "POST_EXPLOITATION", "EXFILTRATION", "CLOSEOUT",
]
_PHASE_INDEX = {p: i for i, p in enumerate(_PHASE_ORDER)}


@dataclass
class EngagementSnapshot:
    """Point-in-time engagement state."""
    step: int = 0
    episode: int = 0
    timestamp: float = 0.0
    phase: str = "RECON"
    total_discoveries: int = 0
    unique_commands: int = 0
    flags_captured: int = 0
    shells_obtained: int = 0
    tokens_used: int = 0


class EngagementMetrics:
    """
    Engagement-level metrics aggregator.

    Tracks progress across the entire engagement (potentially
    multi-episode), providing signals for budget adjustment,
    stagnation detection, and operator dashboards.

    Usage:
        metrics = EngagementMetrics()
        metrics.record_step(step=1, phase="RECON", discoveries=3,
                           command="nmap -sV", tokens=150)
        metrics.record_flag("user_flag", step=45)
        progress = metrics.get_progress()
    """

    def __init__(self) -> None:
        self._start_time: float = time.monotonic()
        self._total_steps: int = 0
        self._total_episodes: int = 0
        self._current_phase: str = "RECON"
        self._highest_phase: str = "RECON"
        self._phase_history: List[Dict[str, Any]] = []
        self._discovery_counts: Dict[str, int] = {}
        self._total_discoveries: int = 0
        self._unique_commands: Set[str] = set()
        self._total_commands: int = 0
        self._flags_captured: Dict[str, int] = {}  # flag_type -> step
        self._shells_obtained: int = 0
        self._tokens_used: int = 0
        self._stagnation_counter: int = 0
        self._last_discovery_step: int = 0
        self._snapshots: List[EngagementSnapshot] = []
        self._exploit_attempts: int = 0
        self._exploit_successes: int = 0
        logger.debug("EngagementMetrics initialised")

    def record_step(
        self,
        step: int,
        phase: str = "",
        discoveries: int = 0,
        command: str = "",
        tokens: int = 0,
        shell_obtained: bool = False,
    ) -> None:
        """
        Record a single step's contribution to engagement metrics.

        Args:
            step: Global step number.
            phase: Current phase name.
            discoveries: Number of new discoveries this step.
            command: Command executed.
            tokens: Tokens consumed.
            shell_obtained: Whether a shell was obtained.
        """
        self._total_steps += 1

        if phase:
            phase_upper = phase.upper()
            if phase_upper != self._current_phase:
                self._phase_history.append({
                    "from": self._current_phase,
                    "to": phase_upper,
                    "step": step,
                })
                self._current_phase = phase_upper

                # Track highest phase
                cur_idx = _PHASE_INDEX.get(phase_upper, 0)
                high_idx = _PHASE_INDEX.get(self._highest_phase, 0)
                if cur_idx > high_idx:
                    self._highest_phase = phase_upper

        if discoveries > 0:
            self._total_discoveries += discoveries
            self._last_discovery_step = step
            self._stagnation_counter = 0
        else:
            self._stagnation_counter += 1

        if command:
            self._unique_commands.add(command)
            self._total_commands += 1

        self._tokens_used += tokens

        if shell_obtained:
            self._shells_obtained += 1

    def record_flag(self, flag_type: str, step: int) -> None:
        """
        Record a flag capture.

        Args:
            flag_type: Type of flag ("user_flag", "root_flag").
            step: Step at which flag was captured.
        """
        if flag_type not in self._flags_captured:
            self._flags_captured[flag_type] = step
            logger.info("Flag captured: %s at step %d", flag_type, step)

    def record_exploit_attempt(self, success: bool) -> None:
        """Record an exploit attempt result."""
        self._exploit_attempts += 1
        if success:
            self._exploit_successes += 1

    def record_episode_end(self, episode: int) -> None:
        """Record end of an episode."""
        self._total_episodes += 1
        self._snapshots.append(EngagementSnapshot(
            step=self._total_steps,
            episode=episode,
            timestamp=time.monotonic() - self._start_time,
            phase=self._current_phase,
            total_discoveries=self._total_discoveries,
            unique_commands=len(self._unique_commands),
            flags_captured=len(self._flags_captured),
            shells_obtained=self._shells_obtained,
            tokens_used=self._tokens_used,
        ))

    def get_progress(self) -> Dict[str, Any]:
        """
        Get engagement progress summary.

        Returns:
            Dict with progress metrics for budget flex and dashboards.
        """
        elapsed = time.monotonic() - self._start_time
        phase_idx = _PHASE_INDEX.get(self._highest_phase, 0)
        phase_progress = phase_idx / max(len(_PHASE_ORDER) - 1, 1)

        discovery_rate = (
            self._total_discoveries / max(self._total_steps, 1)
        )
        command_diversity = (
            len(self._unique_commands) / max(self._total_commands, 1)
        )
        exploit_success_rate = (
            self._exploit_successes / max(self._exploit_attempts, 1)
        )

        return {
            "total_steps": self._total_steps,
            "total_episodes": self._total_episodes,
            "elapsed_seconds": elapsed,
            "current_phase": self._current_phase,
            "highest_phase": self._highest_phase,
            "phase_progress": phase_progress,
            "total_discoveries": self._total_discoveries,
            "discovery_rate": discovery_rate,
            "unique_commands": len(self._unique_commands),
            "command_diversity": command_diversity,
            "flags_captured": dict(self._flags_captured),
            "flags_count": len(self._flags_captured),
            "shells_obtained": self._shells_obtained,
            "tokens_used": self._tokens_used,
            "stagnation_steps": self._stagnation_counter,
            "exploit_success_rate": exploit_success_rate,
            "phase_transitions": len(self._phase_history),
        }

    def get_stagnation_level(self) -> float:
        """
        Get stagnation level [0.0, 1.0].

        0.0 = discovering every step, 1.0 = 20+ steps without discovery.
        """
        return min(1.0, self._stagnation_counter / 20.0)

    def get_phase_velocity(self) -> float:
        """
        Get phase progression velocity (phases per 100 steps).

        Higher = faster progression through kill chain.
        """
        if self._total_steps == 0:
            return 0.0
        transitions = len(self._phase_history)
        return transitions / max(self._total_steps, 1) * 100.0

    def get_token_efficiency(self) -> float:
        """
        Get discoveries per 1000 tokens.

        Higher = more efficient use of LLM budget.
        """
        if self._tokens_used == 0:
            return 0.0
        return self._total_discoveries / (self._tokens_used / 1000.0)

    def get_snapshots(self) -> List[EngagementSnapshot]:
        """Return engagement snapshots (one per episode end)."""
        return list(self._snapshots)

    def reset(self) -> None:
        """Reset all engagement metrics."""
        self._start_time = time.monotonic()
        self._total_steps = 0
        self._total_episodes = 0
        self._current_phase = "RECON"
        self._highest_phase = "RECON"
        self._phase_history.clear()
        self._discovery_counts.clear()
        self._total_discoveries = 0
        self._unique_commands.clear()
        self._total_commands = 0
        self._flags_captured.clear()
        self._shells_obtained = 0
        self._tokens_used = 0
        self._stagnation_counter = 0
        self._last_discovery_step = 0
        self._snapshots.clear()
        self._exploit_attempts = 0
        self._exploit_successes = 0
        logger.debug("EngagementMetrics reset")
