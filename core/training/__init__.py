"""
ARIASKA Training Module

Provides training infrastructure for the multi-agent system.
"""

from core.training.ariaska_trainer import AriaskaTrainer, TrainingConfig
from core.training.apprentice_trainer import (
    ApprenticeTrainer,
    ApprenticeConfig,
    DecisionRecord,
    EpisodeMetrics,
    MentorFeedback,
)
from core.training.apprentice_coach import (
    ApprenticeCoach,
    DecisionResult,
    StepContext,
)
from core.training.mentor_policy import (
    MentorPolicy,
    MentorPolicyConfig,
)

__all__ = [
    # Main trainer
    "AriaskaTrainer",
    "TrainingConfig",
    # Apprentice trainer
    "ApprenticeTrainer",
    "ApprenticeConfig",
    "DecisionRecord",
    "EpisodeMetrics",
    "MentorFeedback",
    # Apprentice coach
    "ApprenticeCoach",
    "DecisionResult",
    "StepContext",
    # Mentor policy
    "MentorPolicy",
    "MentorPolicyConfig",
]
