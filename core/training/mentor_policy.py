#!/usr/bin/env python3
"""
core/training/mentor_policy.py — ARIASKA Mentor Policy System v2.0

Defines when and how GPT mentor should be consulted for each agent.
Supports DYNAMIC adaptive mode that adjusts based on agent performance.
"""

import logging
import random
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger("ariaska.mentor_policy")


@dataclass
class AgentPerformanceTracker:
    """Track per-agent performance to adapt mentor calling."""
    
    # Rolling windows for performance tracking
    recent_rewards: deque = field(default_factory=lambda: deque(maxlen=20))
    recent_successes: deque = field(default_factory=lambda: deque(maxlen=20))
    recent_mentor_helped: deque = field(default_factory=lambda: deque(maxlen=10))
    
    # Cumulative stats
    total_decisions: int = 0
    total_mentor_calls: int = 0
    total_positive_rewards: int = 0
    total_negative_rewards: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    
    def record_decision(self, reward: float, used_mentor: bool, success: bool):
        """Record the outcome of a decision."""
        self.total_decisions += 1
        self.recent_rewards.append(reward)
        self.recent_successes.append(success)
        
        if used_mentor:
            self.total_mentor_calls += 1
            # Track if mentor helped (positive outcome when mentor was used)
            self.recent_mentor_helped.append(reward > 0)
        
        if reward > 0:
            self.total_positive_rewards += 1
            self.consecutive_failures = 0
            self.consecutive_successes += 1
        else:
            self.total_negative_rewards += 1
            self.consecutive_successes = 0
            self.consecutive_failures += 1
    
    @property
    def success_rate(self) -> float:
        """Recent success rate (0-1)."""
        if not self.recent_successes:
            return 0.5
        return sum(self.recent_successes) / len(self.recent_successes)
    
    @property
    def avg_reward(self) -> float:
        """Average recent reward."""
        if not self.recent_rewards:
            return 0.0
        return sum(self.recent_rewards) / len(self.recent_rewards)
    
    @property
    def mentor_effectiveness(self) -> float:
        """How often mentor calls led to positive outcomes."""
        if not self.recent_mentor_helped:
            return 0.5  # Assume neutral
        return sum(self.recent_mentor_helped) / len(self.recent_mentor_helped)
    
    @property
    def is_struggling(self) -> bool:
        """Check if agent is struggling and needs help."""
        return (
            self.consecutive_failures >= 3 or
            self.avg_reward < -1.0 or
            self.success_rate < 0.3
        )
    
    @property
    def is_performing_well(self) -> bool:
        """Check if agent is performing well independently."""
        return (
            self.consecutive_successes >= 5 or
            self.avg_reward > 2.0 or
            self.success_rate > 0.7
        )


@dataclass
class MentorPolicyConfig:
    """Configuration for mentor calling policy."""
    
    # Mode: "adaptive", "anneal", "threshold", "always", "never"
    mode: str = "adaptive"
    
    # Warmup settings
    warmup_episodes: int = 2  # Force mentor for first N episodes
    warmup_steps_per_episode: int = 5  # Force mentor for first N steps of each episode
    
    # Threshold settings (for threshold/anneal modes)
    confidence_threshold: float = 0.5
    initial_threshold: float = 0.3
    final_threshold: float = 0.8
    anneal_episodes: int = 50
    
    # Adaptive mode settings — Phase 53: Aggressive deweaning from LLM
    # PPO must learn independently. Mentor is a luxury, not a crutch.
    min_adaptive_rate: float = 0.05  # Phase 53: 0.15→0.05 — near-zero floor
    max_adaptive_rate: float = 0.15  # Phase 53: 0.30→0.15 — low ceiling
    struggling_boost: float = 1.20   # Phase 53: 1.50→1.20 — minimal boost
    performing_reduction: float = 0.20  # Phase 53: 0.10→0.20 — wean 2x faster
    # Phase 17: Maturity-driven floor decay
    # At maturity=0: floor = min_adaptive_rate (0.60)
    # At maturity=1: floor = mature_floor (0.08) — agent barely needs mentor
    mature_floor: float = 0.03  # Phase 53: 0.08→0.03 — near-zero for mature agent
    # Phase 11.2: Confidence-gated dynamic bounds — prevents token bonfire
    confidence_gate_low: float = 0.3   # Below this: agent unsure → raise min/max for more learning
    confidence_gate_high: float = 0.7  # Above this: agent confident → lower min/max to save tokens
    
    # Rate caps — Phase 17: Dynamic floor replaces static min
    min_mentor_rate: float = 0.08  # Phase 17: absolute floor (used when mature)
    max_mentor_rate: float = 1.0   # Full saturation allowed during learning phase
    
    # Cooldown
    cooldown_steps: int = 1  # Minimum steps between mentor calls
    
    # Per-episode limits — Phase 53: Halved from Phase 52
    max_calls_per_episode: int = 15  # Phase 53: 30→15 — lean GPT budget


class MentorPolicy:
    """
    Decides when to call the GPT mentor based on policy configuration.
    
    Modes:
    - adaptive: DYNAMIC mentor calling based on agent performance
    - anneal: Gradually reduce mentor reliance over training
    - threshold: Fixed confidence threshold
    - always: Always call mentor
    - never: Never call mentor (for testing)
    
    The ADAPTIVE mode is the recommended default. It:
    1. Tracks each agent's success/failure rate
    2. Calls mentor MORE when an agent is struggling
    3. Calls mentor LESS when an agent is performing well
    4. Learns which agents benefit most from mentor help
    """
    
    def __init__(self, config: Optional[MentorPolicyConfig] = None):
        self.config = config or MentorPolicyConfig()
        
        # Track state
        self.current_episode = 0
        self.current_step = 0
        self.calls_this_episode = 0
        self.steps_since_last_call = float('inf')
        
        # Stats tracking
        self.total_calls = 0
        self.total_decisions = 0
        
        # Per-agent adaptive tracking
        self.agent_trackers: Dict[str, AgentPerformanceTracker] = {}
        
        # Track last call per agent to prevent spam
        self.last_call_step: Dict[str, int] = {}

        # Phase 17: Learning maturity signal [0,1] — fed from BudgetManagerV2
        self._maturity_signal: float = 0.0
    
    def _get_tracker(self, agent_name: str) -> AgentPerformanceTracker:
        """Get or create tracker for agent."""
        if agent_name not in self.agent_trackers:
            self.agent_trackers[agent_name] = AgentPerformanceTracker()
        return self.agent_trackers[agent_name]
    
    def reset_episode(self, episode: int):
        """Reset state for a new episode."""
        self.current_episode = episode
        self.current_step = 0
        self.calls_this_episode = 0
        self.steps_since_last_call = float('inf')
        self.last_call_step = {}
    
    def step(self):
        """Advance to next step."""
        self.current_step += 1
        self.steps_since_last_call += 1
    
    def record_outcome(self, agent_name: str, reward: float, used_mentor: bool):
        """
        Record the outcome of an action to improve future mentor decisions.
        
        Args:
            agent_name: Name of the agent
            reward: Reward received for the action
            used_mentor: Whether mentor was consulted for this action
        """
        tracker = self._get_tracker(agent_name)
        success = reward > 0
        tracker.record_decision(reward, used_mentor, success)
        
        # Log adaptive learning
        if self.config.mode == "adaptive":
            logger.debug(
                f"[MENTOR] {agent_name}: reward={reward:.2f}, "
                f"success_rate={tracker.success_rate:.2f}, "
                f"avg_reward={tracker.avg_reward:.2f}, "
                f"mentor_effectiveness={tracker.mentor_effectiveness:.2f}"
            )
    
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
                self._record_call(agent_name)
                return True
            return False
        
        # Check per-agent cooldown
        last_step = self.last_call_step.get(agent_name, -float('inf'))
        if step - last_step < self.config.cooldown_steps:
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
            self._record_call(agent_name)
            return True
        
        # Adaptive mode: dynamic threshold based on agent performance
        if self.config.mode == "adaptive":
            return self._adaptive_decision(agent_name, confidence, episode)
        
        # Legacy anneal/threshold modes
        threshold = self._get_threshold(episode)
        
        if confidence < threshold:
            self._record_call(agent_name)
            return True
        
        # Enforce minimum mentor rate
        current_rate = self.total_calls / max(self.total_decisions, 1)
        if current_rate < self.config.min_mentor_rate:
            if random.random() < 0.5:
                self._record_call(agent_name)
                return True
        
        return False
    
    def _adaptive_decision(self, agent_name: str, confidence: float, episode: int) -> bool:
        """
        Make mentor decision based on agent's learning performance.
        
        Phase 17: Maturity-driven floor decay.
        The min_adaptive_rate drops from 0.60 → 0.08 as the agent matures.
        This means early episodes get heavy GPT guidance, and as the agent
        learns patterns/skills, it uses mentor less and less — copying how
        GPT thinks and doing it itself.
        """
        tracker = self._get_tracker(agent_name)
        
        # Phase 17: Compute effective floor based on maturity
        _floor = self.config.min_adaptive_rate
        _mature = self.config.mature_floor
        # Interpolate: maturity=0 → floor, maturity=1 → mature_floor
        effective_floor = _floor - self._maturity_signal * (_floor - _mature)
        
        # Base rate decreases as episodes progress (natural annealing)
        episode_factor = max(0.3, 1.0 - (episode / 100))
        base_rate = 0.4 * episode_factor  # 40% initially, decreasing to 12%
        
        # Adjust based on agent performance
        if tracker.is_struggling:
            # Agent is struggling - boost mentor rate significantly
            base_rate += self.config.struggling_boost
            logger.debug(f"[MENTOR] {agent_name} is STRUGGLING - boosting rate to {base_rate:.2f}")
        elif tracker.is_performing_well:
            # Agent is doing well - reduce mentor calls
            base_rate -= self.config.performing_reduction
            logger.debug(f"[MENTOR] {agent_name} is PERFORMING WELL - reducing rate to {base_rate:.2f}")
        else:
            # Neutral - adjust based on success rate
            # Lower success = higher mentor rate
            base_rate += (0.5 - tracker.success_rate) * 0.3
        
        # Phase 11.5: Confidence modulation — +50% factor for ultra-accelerated mentor guidance
        confidence_factor = (1.0 - confidence) * 1.10
        final_rate = base_rate + confidence_factor
        
        # Phase 11.5: Mentor effectiveness bonus
        if tracker.mentor_effectiveness > 0.7:
            final_rate += 0.27  # Mentor has been very helpful
        elif tracker.mentor_effectiveness < 0.3:
            final_rate -= 0.10  # Mentor hasn't been helping much
        
        # Phase 17: Use maturity-driven floor instead of static min
        _min_rate = effective_floor
        _max_rate = self.config.max_adaptive_rate
        if confidence > self.config.confidence_gate_high:
            # Agent is confident — shrink bounds to prevent waste
            _gate_factor = (confidence - self.config.confidence_gate_high) / (1.0 - self.config.confidence_gate_high)
            _min_rate *= max(0.5, 1.0 - _gate_factor * 0.5)
            _max_rate *= max(0.6, 1.0 - _gate_factor * 0.35)
        elif confidence < self.config.confidence_gate_low:
            # Agent is unsure — widen bounds for more learning
            _gate_factor = (self.config.confidence_gate_low - confidence) / self.config.confidence_gate_low
            _min_rate *= (1.0 + _gate_factor * 0.78)
            _max_rate = min(1.0, _max_rate * (1.0 + _gate_factor * 0.39))
        
        # Clamp to dynamic limits
        final_rate = max(_min_rate, min(_max_rate, final_rate))
        
        # Probabilistic decision
        if random.random() < final_rate:
            self._record_call(agent_name)
            logger.debug(
                f"[MENTOR] Calling mentor for {agent_name} "
                f"(rate={final_rate:.2f}, conf={confidence:.2f}, "
                f"maturity={self._maturity_signal:.2f}, floor={effective_floor:.2f})"
            )
            return True
        
        return False
    
    def set_maturity(self, maturity: float) -> None:
        """
        Phase 17: Update the learning maturity signal.

        Called by SmartOrchestrator at episode start after BudgetManagerV2
        computes dynamic budget. This drives the mentor floor decay:
        maturity=0 → floor=0.60, maturity=1 → floor=0.08.

        Args:
            maturity: Learning maturity signal [0, 1]
        """
        self._maturity_signal = max(0.0, min(1.0, maturity))

    def _get_threshold(self, episode: int) -> float:
        """Get confidence threshold for current episode (legacy modes)."""
        if self.config.mode == "threshold":
            return self.config.confidence_threshold
        
        if self.config.mode == "anneal":
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
    
    def _record_call(self, agent_name: str = "unknown"):
        """Record that a mentor call was made."""
        self.total_calls += 1
        self.calls_this_episode += 1
        self.steps_since_last_call = 0
        self.last_call_step[agent_name] = self.current_step
    
    def get_stats(self) -> dict:
        """Get policy statistics."""
        stats = {
            "total_calls": self.total_calls,
            "total_decisions": self.total_decisions,
            "call_rate": self.total_calls / max(self.total_decisions, 1),
            "current_episode": self.current_episode,
            "calls_this_episode": self.calls_this_episode,
            "mode": self.config.mode,
        }
        
        if self.config.mode == "adaptive":
            # Include per-agent stats
            stats["agent_stats"] = {}
            for agent_name, tracker in self.agent_trackers.items():
                stats["agent_stats"][agent_name] = {
                    "success_rate": tracker.success_rate,
                    "avg_reward": tracker.avg_reward,
                    "mentor_effectiveness": tracker.mentor_effectiveness,
                    "is_struggling": tracker.is_struggling,
                    "is_performing_well": tracker.is_performing_well,
                    "total_mentor_calls": tracker.total_mentor_calls,
                    "total_decisions": tracker.total_decisions,
                }
        else:
            stats["current_threshold"] = self._get_threshold(self.current_episode)
        
        return stats
    
    def get_agent_learning_summary(self) -> str:
        """Get a human-readable summary of agent learning progress."""
        lines = ["=== Agent Learning Summary ==="]
        
        for agent_name, tracker in self.agent_trackers.items():
            status = "🔴 STRUGGLING" if tracker.is_struggling else (
                "🟢 EXCELLING" if tracker.is_performing_well else "🟡 LEARNING"
            )
            mentor_ratio = tracker.total_mentor_calls / max(tracker.total_decisions, 1)
            
            lines.append(
                f"{agent_name}: {status} | "
                f"Success: {tracker.success_rate:.1%} | "
                f"Avg Reward: {tracker.avg_reward:+.2f} | "
                f"Mentor Rate: {mentor_ratio:.1%}"
            )
        
        return "\n".join(lines)
