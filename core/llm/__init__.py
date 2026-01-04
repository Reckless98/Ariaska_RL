"""
LLM Module - Smart GPT prompting and reward calculation.

This module provides intelligent LLM interaction for command generation,
including structured prompting, attack context management, and reward shaping.
"""

from .smart_mentor import (
    SmartMentor,
    AttackContext,
    MentorResponse,
    create_smart_mentor
)

from .reward_calculator import (
    RewardBreakdown,
    SmartRewardCalculator,
    create_reward_calculator
)

__all__ = [
    # Smart Mentor
    "SmartMentor",
    "AttackContext",
    "MentorResponse",
    "create_smart_mentor",
    
    # Reward Calculator
    "RewardBreakdown",
    "SmartRewardCalculator",
    "create_reward_calculator"
]
