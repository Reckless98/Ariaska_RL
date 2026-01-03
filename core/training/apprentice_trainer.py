#!/usr/bin/env python3
"""
core/training/apprentice_trainer.py — ARIASKA Apprentice-to-Autonomy Training System v1.0

GPT acts as a mentor/teacher for RedAgent, gradually reducing involvement as the agent
learns and becomes more confident in its decisions.

Key Features:
- Confidence-based mentor calling
- Gradual threshold scheduling (more autonomy over time)
- Structured logging of mentor vs. autonomous decisions
- Compatible with existing training infrastructure
"""

import os
import time
import logging
import random
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import torch

logger = logging.getLogger("ariaska.apprentice_trainer")


@dataclass
class ApprenticeConfig:
    """Configuration for apprentice-to-autonomy training."""
    
    # Confidence thresholds
    initial_confidence_threshold: float = 0.3  # Low threshold = more mentor calls early
    final_confidence_threshold: float = 0.8    # High threshold = agent mostly autonomous later
    
    # Threshold scheduling
    warmup_episodes: int = 10          # Episodes before threshold starts increasing
    threshold_schedule_episodes: int = 100  # Episodes over which threshold increases
    
    # Mentor call limits
    max_mentor_calls_per_episode: int = 10
    mentor_call_cooldown_steps: int = 3  # Min steps between mentor calls
    
    # Learning from mentor
    mentor_feedback_weight: float = 0.3  # How much mentor feedback affects reward
    
    # Feature flags
    enable_mentor_override: bool = True   # Allow mentor to override agent action
    log_all_decisions: bool = True        # Log every decision for analysis
    
    def get_threshold_for_episode(self, episode: int) -> float:
        """Get confidence threshold for a given episode."""
        if episode < self.warmup_episodes:
            return self.initial_confidence_threshold
        
        progress = min(1.0, (episode - self.warmup_episodes) / self.threshold_schedule_episodes)
        threshold = self.initial_confidence_threshold + progress * (
            self.final_confidence_threshold - self.initial_confidence_threshold
        )
        return threshold


@dataclass
class MentorFeedback:
    """Structured feedback from GPT mentor."""
    action_quality: float  # 0-1 score
    suggested_action: str
    reasoning: str
    reward_modifier: float  # -0.5 to 0.5
    should_explore: bool
    model_used: str = "gpt-5-mini"


@dataclass
class DecisionRecord:
    """Record of a single decision for logging."""
    episode: int
    step: int
    timestamp: float
    
    # Agent's proposal
    agent_action: str
    agent_confidence: float
    agent_q_values: Optional[List[float]] = None
    
    # Mentor interaction
    mentor_called: bool = False
    mentor_action: Optional[str] = None
    mentor_feedback: Optional[MentorFeedback] = None
    
    # Final decision
    final_action: str = ""
    action_source: str = "agent"  # "agent" or "mentor"
    
    # Outcome
    reward: float = 0.0
    success: bool = False


@dataclass
class EpisodeMetrics:
    """Metrics for an episode."""
    episode: int
    total_steps: int = 0
    total_reward: float = 0.0
    mentor_calls: int = 0
    autonomous_decisions: int = 0
    
    # Confidence tracking
    avg_confidence: float = 0.0
    min_confidence: float = 1.0
    max_confidence: float = 0.0
    confidence_threshold: float = 0.5
    
    # Success tracking
    agent_successes: int = 0
    mentor_successes: int = 0
    
    decisions: List[DecisionRecord] = field(default_factory=list)
    
    def add_decision(self, decision: DecisionRecord):
        """Add a decision record and update metrics."""
        self.decisions.append(decision)
        self.total_steps += 1
        self.total_reward += decision.reward
        
        if decision.mentor_called:
            self.mentor_calls += 1
            if decision.success:
                self.mentor_successes += 1
        else:
            self.autonomous_decisions += 1
            if decision.success:
                self.agent_successes += 1
        
        # Update confidence stats
        conf = decision.agent_confidence
        self.min_confidence = min(self.min_confidence, conf)
        self.max_confidence = max(self.max_confidence, conf)
        # Running average
        self.avg_confidence = (
            (self.avg_confidence * (self.total_steps - 1) + conf) / self.total_steps
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "episode": self.episode,
            "total_steps": self.total_steps,
            "total_reward": self.total_reward,
            "mentor_calls": self.mentor_calls,
            "autonomous_decisions": self.autonomous_decisions,
            "mentor_call_rate": self.mentor_calls / max(self.total_steps, 1),
            "avg_confidence": self.avg_confidence,
            "min_confidence": self.min_confidence,
            "max_confidence": self.max_confidence,
            "confidence_threshold": self.confidence_threshold,
            "agent_success_rate": self.agent_successes / max(self.autonomous_decisions, 1),
            "mentor_success_rate": self.mentor_successes / max(self.mentor_calls, 1),
        }


class ApprenticeTrainer:
    """
    Apprentice-to-Autonomy trainer for RedAgent.
    
    Training Loop:
    1. Agent proposes action based on neural network
    2. Calculate confidence score from Q-value distribution
    3. If confidence < threshold: call GPT mentor for guidance
    4. Execute action (mentor's or agent's based on config)
    5. Update agent with experience
    6. Gradually increase threshold over training
    
    The mentor's role diminishes as the agent becomes more confident,
    leading to autonomous operation.
    """
    
    def __init__(
        self,
        agent,  # RedAgent
        gpt_manager=None,
        config: Optional[ApprenticeConfig] = None,
        trace_writer=None,
        verbosity: str = "standard"
    ):
        from core.gpt_manager import GPTManager
        
        self.agent = agent
        self.gpt_manager = gpt_manager or GPTManager()
        self.config = config or ApprenticeConfig()
        self.trace_writer = trace_writer
        self.verbosity = verbosity
        
        # Training state
        self.current_episode = 0
        self.steps_since_mentor_call = 0
        self.episode_mentor_calls = 0
        
        # Metrics history
        self.episode_metrics: List[EpisodeMetrics] = []
        self.current_metrics: Optional[EpisodeMetrics] = None
        
        # Learning progress
        self.mentor_call_history: List[float] = []  # Rate per episode
        self.confidence_history: List[float] = []   # Avg per episode
        self.reward_history: List[float] = []
        
        logger.info(f"ApprenticeTrainer initialized for {agent.agent_id}")
        logger.info(f"Initial threshold: {self.config.initial_confidence_threshold}")
        logger.info(f"Final threshold: {self.config.final_confidence_threshold}")
    
    def get_current_threshold(self) -> float:
        """Get confidence threshold for current episode."""
        return self.config.get_threshold_for_episode(self.current_episode)
    
    def start_episode(self, episode: int):
        """Start a new episode."""
        self.current_episode = episode
        self.steps_since_mentor_call = 0
        self.episode_mentor_calls = 0
        
        threshold = self.get_current_threshold()
        self.current_metrics = EpisodeMetrics(
            episode=episode,
            confidence_threshold=threshold
        )
        
        if self.verbosity in ("debug", "verbose"):
            logger.info(f"Episode {episode} started | Confidence threshold: {threshold:.3f}")
    
    def decide_action(
        self,
        state: Dict[str, Any],
        phase: str,
        step: int
    ) -> Tuple[str, DecisionRecord]:
        """
        Make a decision using apprentice-to-autonomy policy.
        
        Args:
            state: Current environment state
            phase: Current training phase
            step: Current step in episode
            
        Returns:
            Tuple of (action_command, decision_record)
        """
        timestamp = time.time()
        
        # 1. Get agent's proposal and confidence
        agent_action, agent_reasoning = self.agent.get_smart_command(state, phase)
        confidence = self.agent.get_confidence_score(state)
        
        # Get Q-values for logging if available
        q_values = None
        try:
            state_tensor = self.agent.encode_env_state_static(state, self.agent.device)
            if isinstance(state_tensor, np.ndarray):
                state_tensor = torch.FloatTensor(state_tensor).to(self.agent.device)
            with torch.no_grad():
                q_values = self.agent.policy_net(state_tensor.unsqueeze(0)).squeeze().tolist()
        except Exception:
            pass
        
        # Create decision record
        decision = DecisionRecord(
            episode=self.current_episode,
            step=step,
            timestamp=timestamp,
            agent_action=agent_action,
            agent_confidence=confidence,
            agent_q_values=q_values
        )
        
        # 2. Check if we should call mentor
        threshold = self.get_current_threshold()
        should_call_mentor = (
            confidence < threshold and
            self.steps_since_mentor_call >= self.config.mentor_call_cooldown_steps and
            self.episode_mentor_calls < self.config.max_mentor_calls_per_episode
        )
        
        final_action = agent_action
        action_source = "agent"
        
        if should_call_mentor:
            # 3. Call mentor for guidance
            mentor_feedback = self._get_mentor_feedback(state, phase, agent_action, confidence)
            
            decision.mentor_called = True
            decision.mentor_action = mentor_feedback.suggested_action
            decision.mentor_feedback = mentor_feedback
            
            self.episode_mentor_calls += 1
            self.steps_since_mentor_call = 0
            
            # 4. Decide whether to use mentor's action
            if self.config.enable_mentor_override and mentor_feedback.action_quality > confidence:
                final_action = mentor_feedback.suggested_action
                action_source = "mentor"
                
                if self.verbosity in ("debug", "verbose"):
                    logger.info(
                        f"[Step {step}] Mentor override | "
                        f"Agent conf: {confidence:.3f} vs Mentor: {mentor_feedback.action_quality:.3f}"
                    )
        else:
            self.steps_since_mentor_call += 1
        
        decision.final_action = final_action
        decision.action_source = action_source
        
        if self.verbosity == "debug":
            logger.debug(
                f"[Step {step}] Action: {final_action[:50]} | "
                f"Source: {action_source} | Conf: {confidence:.3f} | Thresh: {threshold:.3f}"
            )
        
        return final_action, decision
    
    def _get_mentor_feedback(
        self,
        state: Dict[str, Any],
        phase: str,
        agent_action: str,
        agent_confidence: float
    ) -> MentorFeedback:
        """Get feedback from GPT mentor."""
        
        prompt = f"""You are an expert cybersecurity mentor evaluating an AI agent's proposed action.

CURRENT SITUATION:
- Phase: {phase}
- Target: {state.get('target_ip', 'unknown')}
- Open Ports: {state.get('open_ports', [])}
- Privilege Level: {state.get('privilege_level', 'none')}
- Detection Risk: {state.get('detection_risk', 0)}

AGENT'S PROPOSED ACTION:
{agent_action}

AGENT CONFIDENCE: {agent_confidence:.2f}

Evaluate this action and respond in this exact JSON format:
{{
    "action_quality": 0.0-1.0,
    "suggested_action": "your recommended command",
    "reasoning": "brief explanation",
    "reward_modifier": -0.5 to 0.5,
    "should_explore": true/false
}}

IMPORTANT: Respond with ONLY the JSON, no additional text."""

        try:
            response = self.gpt_manager.gpt_request(
                prompt,
                task_type="tactical",
                agent_id=self.agent.agent_id,
                max_tokens=200
            )
            
            # Get the model that was used
            model_used = self.gpt_manager.get_model_for_role(
                agent_id=self.agent.agent_id,
                task_type="tactical"
            )
            
            # Parse JSON response
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return MentorFeedback(
                    action_quality=float(data.get("action_quality", 0.5)),
                    suggested_action=data.get("suggested_action", agent_action),
                    reasoning=data.get("reasoning", ""),
                    reward_modifier=float(data.get("reward_modifier", 0.0)),
                    should_explore=data.get("should_explore", False),
                    model_used=model_used
                )
        except Exception as e:
            logger.warning(f"Mentor feedback failed: {e}")
        
        # Default neutral feedback
        return MentorFeedback(
            action_quality=0.5,
            suggested_action=agent_action,
            reasoning="Feedback unavailable",
            reward_modifier=0.0,
            should_explore=True,
            model_used="fallback"
        )
    
    def record_outcome(self, decision: DecisionRecord, reward: float, success: bool):
        """Record the outcome of a decision."""
        decision.reward = reward
        decision.success = success
        
        if self.current_metrics:
            self.current_metrics.add_decision(decision)
        
        # Apply mentor feedback weight if applicable
        if decision.mentor_feedback and self.config.mentor_feedback_weight > 0:
            modified_reward = reward + (
                self.config.mentor_feedback_weight * decision.mentor_feedback.reward_modifier
            )
            return modified_reward
        
        return reward
    
    def end_episode(self) -> EpisodeMetrics:
        """End current episode and return metrics."""
        if not self.current_metrics:
            raise ValueError("No episode in progress")
        
        metrics = self.current_metrics
        self.episode_metrics.append(metrics)
        
        # Update history
        self.mentor_call_history.append(
            metrics.mentor_calls / max(metrics.total_steps, 1)
        )
        self.confidence_history.append(metrics.avg_confidence)
        self.reward_history.append(metrics.total_reward)
        
        if self.verbosity in ("standard", "verbose", "debug"):
            logger.info(
                f"Episode {metrics.episode} ended | "
                f"Reward: {metrics.total_reward:.2f} | "
                f"Mentor calls: {metrics.mentor_calls}/{metrics.total_steps} | "
                f"Avg confidence: {metrics.avg_confidence:.3f}"
            )
        
        self.current_metrics = None
        return metrics
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training progress."""
        if not self.episode_metrics:
            return {"episodes": 0, "message": "No training data yet"}
        
        recent_metrics = self.episode_metrics[-10:]  # Last 10 episodes
        
        return {
            "total_episodes": len(self.episode_metrics),
            "current_threshold": self.get_current_threshold(),
            "avg_reward_recent": np.mean([m.total_reward for m in recent_metrics]),
            "avg_mentor_rate_recent": np.mean([
                m.mentor_calls / max(m.total_steps, 1) for m in recent_metrics
            ]),
            "avg_confidence_recent": np.mean([m.avg_confidence for m in recent_metrics]),
            "mentor_rate_trend": self._calculate_trend(self.mentor_call_history[-20:]),
            "confidence_trend": self._calculate_trend(self.confidence_history[-20:]),
            "reward_trend": self._calculate_trend(self.reward_history[-20:]),
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a list of values."""
        if len(values) < 2:
            return "insufficient_data"
        
        half = len(values) // 2
        first_half = np.mean(values[:half])
        second_half = np.mean(values[half:])
        
        diff = second_half - first_half
        if abs(diff) < 0.01:
            return "stable"
        elif diff > 0:
            return "increasing"
        else:
            return "decreasing"
    
    def should_continue_training(self) -> bool:
        """Check if training should continue based on autonomy metrics."""
        if len(self.episode_metrics) < 20:
            return True  # Need more data
        
        recent = self.episode_metrics[-10:]
        avg_mentor_rate = np.mean([
            m.mentor_calls / max(m.total_steps, 1) for m in recent
        ])
        
        # Continue if mentor is still being called frequently
        return avg_mentor_rate > 0.05  # More than 5% mentor calls


# Convenience function
def create_apprentice_trainer(
    agent,
    config: Optional[ApprenticeConfig] = None,
    **kwargs
) -> ApprenticeTrainer:
    """Create an apprentice trainer for the given agent."""
    return ApprenticeTrainer(agent=agent, config=config, **kwargs)


if __name__ == "__main__":
    # Test the apprentice trainer
    from rich.console import Console
    console = Console()
    
    console.print("[bold cyan]Testing ApprenticeTrainer[/bold cyan]")
    
    # Test config
    config = ApprenticeConfig(
        initial_confidence_threshold=0.3,
        final_confidence_threshold=0.9,
        warmup_episodes=5,
        threshold_schedule_episodes=50
    )
    
    # Test threshold scheduling
    thresholds = [config.get_threshold_for_episode(ep) for ep in range(60)]
    
    console.print(f"Episode 0 threshold: {thresholds[0]:.3f}")
    console.print(f"Episode 5 threshold: {thresholds[5]:.3f}")
    console.print(f"Episode 25 threshold: {thresholds[25]:.3f}")
    console.print(f"Episode 55 threshold: {thresholds[55]:.3f}")
    
    console.print("\n[bold green]✓ ApprenticeTrainer module test passed![/bold green]")
