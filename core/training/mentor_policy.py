#!/usr/bin/env python3
"""
core/training/mentor_policy.py — ARIASKA Mentor Policy System v1.0

Defines when and how GPT mentor should be consulted for each agent.
Supports multiple scheduling modes: anneal, threshold, always.
"""

import logging
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger("ariaska.mentor_policy")


@dataclass
class MentorPolicyConfig:
    """Configuration for mentor calling policy."""
    
    # Mode: "anneal", "threshold", "always", "never"
    mode: str = "anneal"
    
    # Warmup settings
    warmup_episodes: int = 1  # Force mentor for first N episodes
    warmup_steps_per_episode: int = 3  # Force mentor for first N steps of each episode
    
    # Threshold settings
    confidence_threshold: float = 0.5  # Call mentor if confidence < threshold
    initial_threshold: float = 0.3  # Starting threshold for anneal mode
    final_threshold: float = 0.8  # Final threshold for anneal mode
    anneal_episodes: int = 50  # Episodes over which to anneal
    
    # Rate caps
    min_mentor_rate: float = 0.15  # Minimum mentor call rate to maintain
    max_mentor_rate: float = 1.0  # Maximum mentor call rate allowed
    
    # Cooldown
    cooldown_steps: int = 2  # Minimum steps between mentor calls
    
    # Per-episode limits
    max_calls_per_episode: int = 10


class MentorPolicy:
    """
    Decides when to call the GPT mentor based on policy configuration.
    
    Modes:
    - anneal: Gradually reduce mentor reliance over training
    - threshold: Fixed confidence threshold
    - always: Always call mentor
    - never: Never call mentor (for testing)
    """
    
    def __init__(self, config: Optional[MentorPolicyConfig] = None):
        self.config = config or MentorPolicyConfig()
        
        # Track state
        self.current_episode = 0
        self.current_step = 0
        self.calls_this_episode = 0
        self.steps_since_last_call = float('inf')  # Start high so first call is allowed
        
        # Stats tracking
        self.total_calls = 0
        self.total_decisions = 0
    
    def reset_episode(self, episode: int):
        """Reset state for a new episode."""
        self.current_episode = episode
        self.current_step = 0
        self.calls_this_episode = 0
        self.steps_since_last_call = float('inf')
    
    def step(self):
        """Advance to next step."""
        self.current_step += 1
        self.steps_since_last_call += 1
    
    def should_call_mentor(
        self,
        agent_name: str,
        confidence: float,
        episode: Optional[int] = None,
        step: Optional[int] = None
    ) -> bool:
        """
        Determine if mentor should be called for this decision.
        
        Args:
            agent_name: Name of the agent making decision
            confidence: Agent's confidence in its proposed action (0-1)
            episode: Current episode number (optional, uses internal state)
            step: Current step number (optional, uses internal state)
            
        Returns:
            True if mentor should be called
        """
        episode = episode if episode is not None else self.current_episode
        step = step if step is not None else self.current_step
        
        self.total_decisions += 1
        
        # Check mode
        if self.config.mode == "never":
            return False
        
        if self.config.mode == "always":
            if self._check_limits():
                self._record_call()
                return True
            return False
        
        # Check cooldown
        if self.steps_since_last_call < self.config.cooldown_steps:
            return False
        
        # Check per-episode limit
        if self.calls_this_episode >= self.config.max_calls_per_episode:
            return False
        
        # Warmup: force mentor in early episodes/steps
        in_warmup = (
            episode < self.config.warmup_episodes or
            step < self.config.warmup_steps_per_episode
        )
        if in_warmup:
            self._record_call()
            return True
        
        # Get threshold based on mode
        threshold = self._get_threshold(episode)
        
        # Call mentor if confidence below threshold
        if confidence < threshold:
            self._record_call()
            return True
        
        # Enforce minimum mentor rate
        current_rate = self.total_calls / max(self.total_decisions, 1)
        if current_rate < self.config.min_mentor_rate:
            # Probabilistic call to maintain minimum rate
            import random
            if random.random() < 0.5:  # 50% chance to boost rate
                self._record_call()
                return True
        
        return False
    
    def _get_threshold(self, episode: int) -> float:
        """Get confidence threshold for current episode."""
        if self.config.mode == "threshold":
            return self.config.confidence_threshold
        
        if self.config.mode == "anneal":
            # Linear annealing from initial to final threshold
            if episode < self.config.warmup_episodes:
                return self.config.initial_threshold
            
            progress = min(1.0, (episode - self.config.warmup_episodes) / self.config.anneal_episodes)
            return self.config.initial_threshold + progress * (
                self.config.final_threshold - self.config.initial_threshold
            )
        
        return self.config.confidence_threshold
    
    def _check_limits(self) -> bool:
        """Check if mentor call is within limits."""
        if self.steps_since_last_call < self.config.cooldown_steps:
            return False
        if self.calls_this_episode >= self.config.max_calls_per_episode:
            return False
        return True
    
    def _record_call(self):
        """Record that a mentor call was made."""
        self.total_calls += 1
        self.calls_this_episode += 1
        self.steps_since_last_call = 0
    
    def get_stats(self) -> dict:
        """Get policy statistics."""
        return {
            "total_calls": self.total_calls,
            "total_decisions": self.total_decisions,
            "call_rate": self.total_calls / max(self.total_decisions, 1),
            "current_episode": self.current_episode,
            "calls_this_episode": self.calls_this_episode,
            "mode": self.config.mode,
            "current_threshold": self._get_threshold(self.current_episode),
        }
