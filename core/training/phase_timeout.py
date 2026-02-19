"""core/training/phase_timeout.py — Phase 41: Time-boxed phase escalation.

Monitors step counts per phase and triggers forced advancement or
mentor consultation when step limits are exceeded.
"""
from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger("ariaska.training.phase_timeout")


class TimeoutAction(enum.Enum):
    """Action to take when phase timeout is checked."""
    CONTINUE = "continue"
    WARNING = "warning"
    FORCE_ADVANCE = "force_advance"
    MENTOR_CONSULT = "mentor_consult"


_DEFAULT_LIMITS: Dict[str, int] = {
    "RECON": 20,
    "ENUMERATION": 25,
    "EXPLOITATION": 30,
    "PRIVILEGE_ESCALATION": 25,
    "LATERAL_MOVEMENT": 15,
    "POST_EXPLOITATION": 15,
    "EXFILTRATION": 10,
    "CLOSEOUT": 5,
}


@dataclass
class PhaseTimeoutConfig:
    """Configuration for phase timeout management."""
    enabled: bool = True
    phase_step_limits: Dict[str, int] = field(
        default_factory=lambda: dict(_DEFAULT_LIMITS)
    )
    escalation_strategy: str = "force_advance"
    warning_at_pct: float = 0.75


class PhaseTimeoutManager:
    """Monitors per-phase step budgets and triggers escalation.

    Called every step from SmartCoach/orchestrator.  When the step count
    for a phase exceeds the configured limit, returns an action
    (FORCE_ADVANCE or MENTOR_CONSULT).
    """

    def __init__(self, config: Optional[PhaseTimeoutConfig] = None) -> None:
        self.config = config or PhaseTimeoutConfig()
        self._steps_in_phase: Dict[str, int] = {}
        self._warnings_issued: Dict[str, bool] = {}

    def record_step(self, phase: str) -> None:
        """Record a step in the given phase."""
        self._steps_in_phase[phase] = self._steps_in_phase.get(phase, 0) + 1

    def check_timeout(
        self, current_phase: str, steps_in_phase: Optional[int] = None
    ) -> TimeoutAction:
        """Check if the current phase has timed out.

        Args:
            current_phase: Current attack phase.
            steps_in_phase: Override step count (uses internal tracking if None).

        Returns:
            TimeoutAction indicating what to do.
        """
        if not self.config.enabled:
            return TimeoutAction.CONTINUE

        if steps_in_phase is None:
            steps_in_phase = self._steps_in_phase.get(current_phase, 0)

        limit = self.config.phase_step_limits.get(current_phase, 30)
        if limit <= 0:
            return TimeoutAction.CONTINUE

        # Check for timeout
        if steps_in_phase >= limit:
            strategy = self.config.escalation_strategy
            if strategy == "mentor_consult":
                return TimeoutAction.MENTOR_CONSULT
            return TimeoutAction.FORCE_ADVANCE

        # Check for warning
        warning_threshold = int(limit * self.config.warning_at_pct)
        if (
            steps_in_phase >= warning_threshold
            and not self._warnings_issued.get(current_phase, False)
        ):
            self._warnings_issued[current_phase] = True
            return TimeoutAction.WARNING

        return TimeoutAction.CONTINUE

    def get_escalation_suggestion(
        self, phase: str, steps_spent: int, state: Optional[Dict[str, Any]] = None
    ) -> str:
        """Suggest what to do when force-advancing.

        Args:
            phase: Phase being left.
            steps_spent: Steps spent in the phase.
            state: Optional state context.

        Returns:
            Human-readable suggestion string.
        """
        phase_order = [
            "RECON", "ENUMERATION", "EXPLOITATION",
            "PRIVILEGE_ESCALATION", "LATERAL_MOVEMENT",
            "POST_EXPLOITATION", "EXFILTRATION", "CLOSEOUT",
        ]
        try:
            idx = phase_order.index(phase)
            next_phase = phase_order[min(idx + 1, len(phase_order) - 1)]
        except ValueError:
            next_phase = "ENUMERATION"

        return (
            f"Phase {phase} exceeded {steps_spent} steps. "
            f"Recommend advancing to {next_phase}."
        )

    def reset(self) -> None:
        """Reset all counters."""
        self._steps_in_phase.clear()
        self._warnings_issued.clear()
