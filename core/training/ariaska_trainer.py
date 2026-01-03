#!/usr/bin/env python3
"""
core/training/ariaska_trainer.py — ARIASKA Unified Training System v1.0

Single canonical training entry point that integrates all new systems:
- EpisodeTrace JSONL logging
- Apprentice-to-Autonomy policy
- OrionPostmortem end-of-run analysis
- SkillLibrary skill management

This is the recommended way to run training.
"""

import os
import sys
import time
import random
import logging
import argparse
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger("ariaska.trainer")


@dataclass
class TrainingConfig:
    """Configuration for unified training."""
    
    # Basic training
    episodes: int = 100
    max_steps_per_episode: int = 50
    seed: Optional[int] = None
    
    # Apprentice-to-Autonomy
    initial_confidence_threshold: float = 0.3
    final_confidence_threshold: float = 0.8
    warmup_episodes: int = 10
    threshold_schedule_episodes: int = 100
    max_mentor_calls_per_episode: int = 10
    
    # Tracing
    trace_output_dir: str = "traces"
    enable_tracing: bool = True
    
    # Postmortem
    enable_postmortem: bool = True
    enable_gpt_5_2_postmortem: bool = False
    postmortem_output_dir: str = "postmortems"
    
    # Skill Library
    skill_library_path: str = "core/memory/skill_library.json"
    apply_skill_updates: bool = True  # If False, dry-run mode
    
    # Target network
    target_update_freq: int = 1000
    soft_update_tau: float = 0.005
    
    # Verbosity
    verbosity: str = "standard"  # "quiet", "standard", "verbose", "debug"
    
    # Checkpoints
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 10  # Save every N episodes
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


class AriaskaTrainer:
    """
    Unified training system for ARIASKA.
    
    Integrates:
    - RedAgent with proper target network
    - Apprentice-to-Autonomy training policy
    - EpisodeTrace JSONL logging
    - OrionPostmortem end-of-run analysis
    - SkillLibrary for skill management
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        
        # Set seed for reproducibility
        if self.config.seed is not None:
            self._set_seed(self.config.seed)
        
        # Initialize components
        self._init_components()
        
        # Training state
        self.current_episode = 0
        self.total_steps = 0
        self.training_start_time = None
        
        logger.info(f"AriaskaTrainer initialized | Episodes: {self.config.episodes}")
    
    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        logger.info(f"Random seed set to {seed}")
    
    def _init_components(self):
        """Initialize all training components."""
        
        # Import components
        from core.gpt_manager import GPTManager
        from core.environment.cyber_environment import CyberEnvironment
        from core.agents.red_agent import RedAgent
        from core.memory.enhanced_memory_router import EnhancedMemoryRouter
        from core.tracing import TraceWriter, StepTrace
        from core.training.apprentice_trainer import ApprenticeTrainer, ApprenticeConfig
        from core.postmortem import OrionPostmortem, SkillLibrary
        
        # GPT Manager
        self.gpt_manager = GPTManager()
        
        # Memory Router
        self.memory_router = EnhancedMemoryRouter()
        
        # Environment
        self.environment = CyberEnvironment(defer_reset=True)
        
        # Agent
        self.agent = RedAgent(
            agent_id="RedAgent",
            role="CyberOffense",
            device="cuda" if torch.cuda.is_available() else "cpu",
            memory_router=self.memory_router,
            verbosity=self.config.verbosity
        )
        
        # Trace Writer
        self.trace_writer = None
        if self.config.enable_tracing:
            self.trace_writer = TraceWriter(
                output_dir=self.config.trace_output_dir
            )
        
        # Apprentice Trainer
        apprentice_config = ApprenticeConfig(
            initial_confidence_threshold=self.config.initial_confidence_threshold,
            final_confidence_threshold=self.config.final_confidence_threshold,
            warmup_episodes=self.config.warmup_episodes,
            threshold_schedule_episodes=self.config.threshold_schedule_episodes,
            max_mentor_calls_per_episode=self.config.max_mentor_calls_per_episode
        )
        self.apprentice_trainer = ApprenticeTrainer(
            agent=self.agent,
            gpt_manager=self.gpt_manager,
            config=apprentice_config,
            trace_writer=self.trace_writer,
            verbosity=self.config.verbosity
        )
        
        # Postmortem Analyzer
        self.postmortem = None
        if self.config.enable_postmortem:
            self.postmortem = OrionPostmortem(
                gpt_manager=self.gpt_manager,
                output_dir=self.config.postmortem_output_dir,
                enable_gpt_5_2=self.config.enable_gpt_5_2_postmortem
            )
        
        # Skill Library
        self.skill_library = SkillLibrary(
            library_path=self.config.skill_library_path
        )
        
        logger.info("All components initialized successfully")
    
    def train(self) -> Dict[str, Any]:
        """
        Run the complete training loop.
        
        Returns:
            Dict containing training summary and metrics
        """
        self.training_start_time = time.time()
        
        # Start tracing
        if self.trace_writer:
            self.trace_writer.start_run(
                config=self.config.to_dict(),
                seed=self.config.seed
            )
        
        try:
            # Main training loop
            for episode in range(self.config.episodes):
                self.current_episode = episode
                
                episode_result = self._run_episode(episode)
                
                # Log progress
                if episode % 10 == 0 or episode == self.config.episodes - 1:
                    self._log_progress(episode, episode_result)
                
                # Checkpoint
                if episode % self.config.checkpoint_interval == 0 and episode > 0:
                    self._save_checkpoint(episode)
            
            # End training
            training_summary = self._finalize_training()
            
            return training_summary
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            return self._finalize_training()
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def _run_episode(self, episode: int) -> Dict[str, Any]:
        """Run a single training episode."""
        
        # Reset environment
        state = self.environment.reset()
        if not state:
            state = {
                "phase": "recon",
                "target_ip": "10.10.10.10",
                "open_ports": [],
                "detection_risk": 0.0
            }
        
        # Start episode in trackers
        episode_id = None
        if self.trace_writer:
            episode_id = self.trace_writer.start_episode(episode)
        
        self.apprentice_trainer.start_episode(episode)
        self.gpt_manager.reset_token_count()
        
        # Episode metrics
        episode_reward = 0.0
        episode_steps = 0
        success = False
        
        # Run episode steps
        for step in range(self.config.max_steps_per_episode):
            # Get current phase
            phase = state.get("phase", "recon")
            
            # Make decision using apprentice-to-autonomy policy
            action, decision = self.apprentice_trainer.decide_action(
                state=state,
                phase=phase,
                step=step
            )
            
            # Execute action in environment
            try:
                result = self.environment.step(action)
                # Handle both dict and tuple return formats
                if isinstance(result, tuple):
                    # Gym-style: (state, reward, done, info)
                    next_state = result[0] if result[0] else state
                    reward = result[1] if len(result) > 1 else 0.0
                    done = result[2] if len(result) > 2 else False
                    info = result[3] if len(result) > 3 else {}
                else:
                    # Dict-style
                    next_state = result.get("state", state)
                    reward = result.get("reward", 0.0)
                    done = result.get("done", False)
                    info = result.get("info", {})
            except Exception as e:
                logger.warning(f"Environment step failed: {e}")
                next_state = state
                reward = -1.0
                done = False
                info = {"error": str(e)}
            
            # Record outcome
            action_success = reward > 0
            modified_reward = self.apprentice_trainer.record_outcome(
                decision=decision,
                reward=reward,
                success=action_success
            )
            
            # Log step trace
            if self.trace_writer:
                from core.tracing import StepTrace
                step_trace = StepTrace(
                    episode_id=episode_id,
                    step=step,
                    agent=self.agent.agent_id,
                    phase=phase,
                    proposed_action=decision.agent_action,
                    chosen_action=action,
                    reward=reward,
                    done=done,
                    mentor_call=decision.mentor_called,
                    model_used=decision.mentor_feedback.model_used if decision.mentor_feedback else None,
                    confidence=decision.agent_confidence
                )
                # Store timestamp in private field (not part of deterministic trace)
                step_trace._timestamp = time.time()
                self.trace_writer.log_step(step_trace)
            
            # Store experience in replay buffer
            # Convert tensors to lists for JSON serialization
            state_encoded = self.agent.encode_env_state_static(state, self.agent.device)
            next_state_encoded = self.agent.encode_env_state_static(next_state, self.agent.device)
            
            # Handle tensor conversion
            if hasattr(state_encoded, 'tolist'):
                state_encoded = state_encoded.tolist()
            if hasattr(next_state_encoded, 'tolist'):
                next_state_encoded = next_state_encoded.tolist()
            
            self.agent.replay_buffer.add({
                "state": state_encoded,
                "action": 0,  # Action index
                "reward": modified_reward,
                "next_state": next_state_encoded,
                "done": done,
                "priority": abs(reward) + 0.1
            })
            
            # Train if enough experiences
            if len(self.agent.replay_buffer.buffer) > self.agent.batch_size:
                self.agent.train_on_batch()
            
            # Target network update
            self.total_steps += 1
            if self.total_steps % self.config.target_update_freq == 0:
                self.agent.sync_target_network(tau=self.config.soft_update_tau)
            
            # Update state
            episode_reward += reward
            episode_steps += 1
            state = next_state
            
            if done:
                success = True
                break
        
        # End episode
        episode_metrics = self.apprentice_trainer.end_episode()
        
        if self.trace_writer:
            self.trace_writer.end_episode(
                success=success,
                final_phase=state.get("phase", "unknown")
            )
        
        return {
            "episode": episode,
            "reward": episode_reward,
            "steps": episode_steps,
            "success": success,
            "mentor_calls": episode_metrics.mentor_calls,
            "avg_confidence": episode_metrics.avg_confidence
        }
    
    def _log_progress(self, episode: int, result: Dict[str, Any]):
        """Log training progress."""
        summary = self.apprentice_trainer.get_training_summary()
        
        logger.info(
            f"Episode {episode}/{self.config.episodes} | "
            f"Reward: {result['reward']:.2f} | "
            f"Mentor calls: {result['mentor_calls']} | "
            f"Confidence: {result['avg_confidence']:.3f} | "
            f"Threshold: {self.apprentice_trainer.get_current_threshold():.3f}"
        )
        
        # Print trend info periodically
        if episode > 0 and episode % 20 == 0:
            logger.info(
                f"Trends | Mentor rate: {summary.get('mentor_rate_trend', 'N/A')} | "
                f"Confidence: {summary.get('confidence_trend', 'N/A')} | "
                f"Reward: {summary.get('reward_trend', 'N/A')}"
            )
    
    def _save_checkpoint(self, episode: int):
        """Save training checkpoint."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"checkpoint_ep{episode:04d}.pt"
        
        torch.save({
            "episode": episode,
            "total_steps": self.total_steps,
            "policy_net_state": self.agent.policy_net.state_dict(),
            "target_net_state": self.agent.target_net.state_dict(),
            "epsilon": self.agent.epsilon,
            "config": self.config.to_dict()
        }, checkpoint_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _finalize_training(self) -> Dict[str, Any]:
        """Finalize training and run postmortem."""
        training_time = time.time() - self.training_start_time
        
        # End trace run (returns RunTrace object)
        run_trace_obj = None
        if self.trace_writer:
            run_trace_obj = self.trace_writer.end_run()
        
        # Get training summary
        summary = self.apprentice_trainer.get_training_summary()
        
        # Run postmortem analysis
        postmortem_result = None
        if self.postmortem and run_trace_obj:
            logger.info("Running OrionPostmortem analysis...")
            
            # Extract valid event IDs from trace for evidence_refs validation
            valid_event_ids = run_trace_obj.get_all_event_ids()
            
            postmortem_result = self.postmortem.analyze_run(
                run_trace=run_trace_obj.to_dict(),
                dry_run=not self.config.apply_skill_updates,
                valid_event_ids=valid_event_ids
            )
            
            if postmortem_result.validation_passed:
                logger.info(f"Postmortem analysis complete | Skill cards: {len(postmortem_result.skill_cards)}")
                
                # Apply memory operations
                if self.config.apply_skill_updates:
                    results = self.skill_library.apply_memory_ops(
                        ops=postmortem_result.memory_ops,
                        skills=postmortem_result.skill_cards
                    )
                    logger.info(f"Memory ops applied: {results}")
            else:
                logger.warning("Postmortem validation failed")
        
        # Final summary
        final_summary = {
            "training_time_seconds": training_time,
            "total_episodes": self.current_episode + 1,
            "total_steps": self.total_steps,
            **summary,
            "skill_library_size": len(self.skill_library.get_all_skills()),
            "postmortem_ran": postmortem_result is not None,
            "postmortem_passed": postmortem_result.validation_passed if postmortem_result else False
        }
        
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Time: {training_time:.1f}s | Episodes: {final_summary['total_episodes']}")
        logger.info(f"Final reward trend: {summary.get('reward_trend', 'N/A')}")
        logger.info(f"Final mentor rate trend: {summary.get('mentor_rate_trend', 'N/A')}")
        logger.info("=" * 60)
        
        return final_summary


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description="ARIASKA Unified Training")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--max-steps", type=int, default=50, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--verbosity", choices=["quiet", "standard", "verbose", "debug"], 
                        default="standard", help="Verbosity level")
    parser.add_argument("--no-postmortem", action="store_true", help="Disable postmortem analysis")
    parser.add_argument("--dry-run", action="store_true", help="Don't apply skill updates")
    parser.add_argument("--enable-gpt-5-2", action="store_true", help="Enable GPT-5.2 for postmortem")
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        seed=args.seed,
        verbosity=args.verbosity,
        enable_postmortem=not args.no_postmortem,
        apply_skill_updates=not args.dry_run,
        enable_gpt_5_2_postmortem=args.enable_gpt_5_2
    )
    
    trainer = AriaskaTrainer(config=config)
    results = trainer.train()
    
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    for key, value in results.items():
        print(f"  {key}: {value}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
