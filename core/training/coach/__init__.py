"""
Coach package — modular components extracted from SmartCoach.

Phase 41: Architecture hardening.
Backward compatibility:
    from core.training.coach import SmartCoach          # works
    from core.training.smart_coach import SmartCoach    # still works
"""
from __future__ import annotations

from core.training.smart_coach import (
    SmartCoach,
    SmartDecisionResult,
    SmartStepContext,
    SmartCoachWrapper,
    create_smart_coach,
)

__all__ = [
    "SmartCoach",
    "SmartDecisionResult",
    "SmartStepContext",
    "SmartCoachWrapper",
    "create_smart_coach",
]
