"""core/metrics/ttf_metrics.py — Phase 42: Time-to-First metrics tracker.

Tracks time-to-first-X metrics for key milestones during an episode:
- TTF Port: first open port discovered
- TTF Service: first service identified
- TTF Credential: first credential obtained
- TTF Shell: first shell obtained
- TTF Root: first root/admin access

These metrics measure learning efficiency independent of reward scaling.

Author: Phase 42 Contract
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ariaska.metrics.ttf_metrics")


@dataclass
class TTFMilestone:
    """Record of when a milestone was first achieved."""
    milestone_type: str
    step: int
    wall_time_seconds: float
    value: str = ""  # e.g., "22/tcp", "root:toor"


class TTFTracker:
    """Phase 42: Time-to-First milestone tracker.

    Tracks when key milestones are first achieved during an episode.
    Produces a summary at episode end for dashboard and telemetry.

    Methods:
        record(): Record a milestone event
        get_ttf(): Get time-to-first for a type
        episode_summary(): Get all TTF metrics for the episode
        reset(): Clear for new episode
    """

    _MILESTONE_TYPES = {
        "port", "service", "version", "credential",
        "shell", "root", "flag", "vuln", "user",
    }

    def __init__(self) -> None:
        self._milestones: Dict[str, TTFMilestone] = {}
        self._episode_start: float = time.monotonic()
        self._step_count: int = 0
        self._history: List[Dict[str, Any]] = []  # cross-episode history
        logger.info("TTFTracker initialized")

    def record(
        self,
        milestone_type: str,
        step: int,
        value: str = "",
    ) -> bool:
        """Record a milestone event.

        Only the first occurrence per type per episode is recorded.

        Args:
            milestone_type: The type of milestone (port, service, etc.).
            step: Current step number.
            value: Optional value string.

        Returns:
            True if this was the first occurrence (newly recorded).
        """
        self._step_count = max(self._step_count, step)
        norm_type = milestone_type.lower()
        if norm_type in self._milestones:
            return False  # Already recorded

        elapsed = time.monotonic() - self._episode_start
        self._milestones[norm_type] = TTFMilestone(
            milestone_type=norm_type,
            step=step,
            wall_time_seconds=round(elapsed, 3),
            value=value,
        )
        logger.debug("TTF recorded: %s at step %d (%.1fs)", norm_type, step, elapsed)
        return True

    def get_ttf(self, milestone_type: str) -> Optional[int]:
        """Get time-to-first (in steps) for a milestone type.

        Args:
            milestone_type: The type to query.

        Returns:
            Step number of first occurrence, or None if not yet achieved.
        """
        m = self._milestones.get(milestone_type.lower())
        return m.step if m else None

    def episode_summary(self) -> Dict[str, Any]:
        """Get all TTF metrics for the current episode.

        Returns:
            Dict with TTF values per milestone type and coverage stats.
        """
        summary: Dict[str, Any] = {
            "ttf_by_type": {},
            "achieved_count": len(self._milestones),
            "total_types": len(self._MILESTONE_TYPES),
            "coverage": len(self._milestones) / max(len(self._MILESTONE_TYPES), 1),
            "total_steps": self._step_count,
        }
        for mtype, m in sorted(self._milestones.items()):
            summary["ttf_by_type"][mtype] = {
                "step": m.step,
                "wall_time": m.wall_time_seconds,
                "value": m.value,
            }
        return summary

    def reset(self) -> None:
        """Clear for new episode, preserving history."""
        if self._milestones:
            self._history.append(self.episode_summary())
            # Keep last 100 episodes
            if len(self._history) > 100:
                self._history = self._history[-100:]
        self._milestones.clear()
        self._episode_start = time.monotonic()
        self._step_count = 0
        logger.debug("TTFTracker reset")

    def get_trend(self, milestone_type: str, last_n: int = 10) -> List[Optional[int]]:
        """Get trend of TTF values across recent episodes.

        Args:
            milestone_type: The type to get trend for.
            last_n: Number of recent episodes.

        Returns:
            List of TTF step values (None if not achieved in that episode).
        """
        trend = []
        for ep_summary in self._history[-last_n:]:
            ttf_data = ep_summary.get("ttf_by_type", {}).get(milestone_type.lower())
            trend.append(ttf_data["step"] if ttf_data else None)
        return trend

    def __len__(self) -> int:
        return len(self._milestones)
