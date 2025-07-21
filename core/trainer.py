# core/trainer.py — ARIASKA RL Training Loop v12.0
# 🧠 Advanced Stable Training Loop | 📈 Episodic Learning Curves | ⚖️ Multi-Objective Reward Handling

import os
import random
import time
import torch
import numpy as np
import json
from typing import TYPE_CHECKING, Any
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel

from core.environment.cyber_environment import CyberEnvironment
from core.multiagent.agent_manager import AgentManager

# Conditional imports
if TYPE_CHECKING:
    from core.multiagent.memory_router import MemoryRouter
else:
    # Try to import memory router, if not available, create placeholder
    try:
        from core.multiagent.memory_router import MemoryRouter
        MEMORY_ROUTER_AVAILABLE = True
    except ImportError:
        # Fallback MemoryRouter class when module not available
        class MemoryRouter:  # type: ignore
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass
            def sync_global_insights(self) -> None:
                pass
            def optimize_memories(self) -> None:
                pass
            def snapshot_all_memories(self) -> None:
                pass
        MEMORY_ROUTER_AVAILABLE = False

console = Console()

class Trainer:
    """
    Advanced RL training loop for cyber offense simulation.
    
    Features:
    - Stable Gym-style training pattern
    - Warm-up period for experience collection
    - Proper episode reset and progression handling
    - Automatic replay buffer optimization
    - Multi-phase progression tracking
    - Comprehensive training diagnostics
    - Prioritized experience replay integration
    """
    
    def __init__(
        self,
        agent,
        environment=None,
        episodes=100,
        max_steps=50,
        batch_size=64,
        warmup_steps=1000,
        update_frequency=4,
        target_update_freq=1000,
        log_dir="logs",
        memory_router=None,
        agent_manager=None,
        save_dir="models",
        verbosity="standard",
        use_prioritized_replay=True,
        phase_transitions=True,
        live_mode=False
    ):
        self.agent = agent
        self.env = environment or CyberEnvironment(agent_manager=agent_manager)
        self.episodes = episodes
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.update_frequency = update_frequency
        self.target_update_freq = target_update_freq
        self.log_dir = log_dir
        self.save_dir = save_dir
        self.memory_router = memory_router or MemoryRouter()
        self.agent_manager = agent_manager or AgentManager()
        self.verbosity = verbosity
        self.use_prioritized_replay = use_prioritized_replay
        self.phase_transitions = phase_transitions
        self.live_mode = live_mode
        
        # Ensure directories exist
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)
        
        # Track training metrics
        self.episode_rewards = []
        self.episode_steps = []
        self.episode_phases = []
        self.total_steps = 0
        self.train_start_time = None
        
        # Initialize progress tracking
        self.progress = None
        self.task_id = None
        
        # Training stability metrics
        self.agent_losses = []
        self.unique_actions_per_episode = []
        self.phase_transitions_per_episode = []
        self.successful_episodes = 0
        self.early_terminations = 0
        
        # Initialize visualization components
        try:
            from core.visualization.training_visualizer import TrainingVisualizer
            self.visualizer = TrainingVisualizer.get_instance(
                agents=[agent.agent_id],
                max_history=100,
                log_dir=log_dir
            )
        except ImportError:
            self.visualizer = None
            console.print("[yellow]⚠ TrainingVisualizer not available[/yellow]")
            
        # Create event log for detailed training analysis
        self.event_log_file = os.path.join(log_dir, "training_log.jsonl")
        
        console.print(f"[green]✓ Trainer initialized with {episodes} episodes, {max_steps} max steps[/green]")
        if self.use_prioritized_replay:
            console.print("[cyan]Using prioritized experience replay[/cyan]")
        if hasattr(agent, "agent_id"):
            console.print(f"[green]Training agent: {agent.agent_id}[/green]")

    def run(self):
        """
        Main training loop following stable Gym-style pattern:
        1. Reset environment
        2. Loop until done or max_steps:
           - Select action
           - Take step in environment
           - Store transition in replay buffer
           - Learn from replay buffer
        3. Repeat for episodes
        """
        # Start timing
        self.train_start_time = time.time()
        
        console.rule("[bold cyan]🚀 Starting ARIASKA RL Training Loop[/bold cyan]")
        
        # Start visualizer if available
        if self.visualizer and hasattr(self.visualizer, "start_live_display"):
            self.visualizer.start_live_display()
        
        # Setup progress bar
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[cyan]{task.completed}/{task.total}"),
            TimeRemainingColumn()
        ) as self.progress:
            self.task_id = self.progress.add_task("[bold]Training episodes", total=self.episodes)
            
            # Main training loop
            for episode in range(1, self.episodes + 1):
                # Run episode and collect metrics
                episode_reward, steps_taken, phase_changes, unique_actions, losses = self._run_episode(episode)
                
                # Update training statistics
                self.episode_rewards.append(episode_reward)
                self.episode_steps.append(steps_taken)
                self.episode_phases.append(phase_changes)
                self.unique_actions_per_episode.append(len(unique_actions))
                self.phase_transitions_per_episode.append(phase_changes)
                
                # Track agent losses if available
                if losses:
                    self.agent_losses.extend(losses)
                
                # Update progress bar
                self.progress.update(self.task_id, advance=1)
                
                # Log episode results with enhanced positioning
                self._log_episode_results_enhanced(episode, episode_reward, steps_taken, phase_changes, len(unique_actions))
                
                # Save model periodically or on significant improvements
                if (episode % 10 == 0) or (
                    episode > 5 and 
                    episode_reward > max(self.episode_rewards[:-1], default=0)
                ):
                    self._save_model(episode)
                    
                # Periodic memory consolidation
                if episode % 5 == 0:
                    self._consolidate_memory()
                    
                # Update visualizations
                if self.visualizer and hasattr(self.visualizer, "push_alert"):
                    if episode_reward > max(self.episode_rewards[:-1], default=0):
                        self.visualizer.push_alert(f"New best reward: {episode_reward:.2f}", "success")
        
        # Final cleanup and report
        self._final_report()
        
        # Close visualizer if needed
        if self.visualizer and hasattr(self.visualizer, "stop_live_display"):
            self.visualizer.stop_live_display()
        
        return self.episode_rewards, self.episode_steps

    def _run_episode(self, episode):
        """
        Run a single episode with proper phase transitions and metrics tracking.
        Returns:
          - episode_reward: Total reward accumulated in this episode
          - steps_taken: Number of steps taken in this episode
          - phase_changes: Number of phase transitions that occurred
          - unique_actions: Set of unique actions taken
          - losses: List of loss values for learning steps in this episode
        """
        # Begin episode with environment reset - critical for stable training
        state = self.env.reset()
        
        # Validate state format
        if not state:
            console.print("[red]❌ Environment reset returned invalid state[/red]")
            return 0.0, 0, 0, set(), []
        
        # Track episode metrics
        episode_reward = 0.0
        steps_taken = 0
        phase_transitions = 0
        unique_actions = set()
        episode_losses = []  # Track losses during this episode
        done = False
        
        # Extract initial phase from state 
        last_phase = state.get("phase", "recon")
        
        # Generate episode ID for logging
        episode_id = f"ep{episode}_{int(time.time())}"
        
        # Episode progress tracking for verbose mode
        episode_progress = None
        episode_task = None
        
        if self.verbosity in ("verbose", "detailed"):
            with Progress(
                TextColumn("[bold green]{task.description}"),
                BarColumn(),
                TextColumn("[cyan]{task.completed}/{task.total}")
            ) as episode_progress:
                episode_task = episode_progress.add_task(f"[bold]Episode {episode}", total=self.max_steps)
                
                # Run episode steps with progress tracking
                episode_data = self._run_episode_steps(
                    episode, state, done, unique_actions, last_phase, episode_reward, 
                    steps_taken, phase_transitions, episode_losses, episode_id,
                    episode_progress, episode_task
                )
        else:
            # Run episode steps without progress tracking
            episode_data = self._run_episode_steps(
                episode, state, done, unique_actions, last_phase, episode_reward, 
                steps_taken, phase_transitions, episode_losses, episode_id
            )
        
        # Unpack episode results
        episode_reward, steps_taken, phase_transitions, unique_actions, episode_losses = episode_data
        
        # Update adaptive exploration parameters based on episode performance
        self._update_adaptive_parameters(episode, episode_reward, len(unique_actions), phase_transitions)
        
        # Check if episode was successful or timed out
        if steps_taken < self.max_steps and hasattr(self.env, "data_exfiltrated") and self.env.data_exfiltrated:
            self.successful_episodes += 1
        elif steps_taken >= self.max_steps:
            self.early_terminations += 1
        
        return episode_reward, steps_taken, phase_transitions, unique_actions, episode_losses

    def _run_episode_steps(self, episode, state, done, unique_actions, last_phase, 
                           episode_reward, steps_taken, phase_transitions, episode_losses,
                           episode_id, episode_progress=None, episode_task=None):
        """
        Run steps within an episode with proper progress tracking and stable learning.
        Returns: (episode_reward, steps_taken, phase_transitions, unique_actions, episode_losses)
        """
        # Run steps until done or max_steps reached
        for step in range(1, self.max_steps + 1):
            # Select action based on current state
            # Prefer simulate_step for modern agents
            if hasattr(self.agent, "simulate_step"):
                # Modern approach with full state observation
                action_info = self.agent.simulate_step(state=state, episode=episode, step=step, phase=state.get("phase", None))
                action = action_info.get("command", None)
                
                # Extract any action-specific info for visualization
                if self.visualizer and hasattr(self.visualizer, "update"):
                    self.visualizer.update(action_info)
            elif hasattr(self.agent, "select_action"):
                # Traditional approach - encode state if needed
                if hasattr(self.agent, "encode_env_state"):
                    state_tensor = self.agent.encode_env_state(state)
                    action = self.agent.select_action(state_tensor, phase=state.get("phase", None))
                else:
                    action = self.agent.select_action(state, phase=state.get("phase", None))
            else:
                # Fallback to random action if no selection method available
                console.print("[yellow]⚠ Agent has no action selection method, using random action[/yellow]")
                action = random.randint(0, 5)  # Random action as fallback
            
            # Track unique actions
            if action is not None:
                action_key = action if not isinstance(action, str) else action.split()[0]
                unique_actions.add(str(action_key))
            
            # Execute action in environment
            next_state, reward, done, info = self.env.step(action)
            
            # Track total steps for global metrics
            self.total_steps += 1
            
            # Handle phase transitions
            current_phase = next_state.get("phase", last_phase)
            if current_phase != last_phase:
                phase_transitions += 1
                last_phase = current_phase
                
                if self.verbosity != "quiet":
                    console.print(f"[bold magenta]🔄 Phase transition to: {current_phase}[/bold magenta]")
                    
                # Log phase transition
                self._log_event(episode_id, {
                    "type": "phase_transition",
                    "episode": episode,
                    "step": step,
                    "from_phase": last_phase,
                    "to_phase": current_phase
                })
            
            # Store transition in replay buffer
            self._store_experience(state, action, reward, next_state, done, current_phase)
            
            # Perform learning if we have enough samples
            if self.total_steps >= self.warmup_steps and self.total_steps % self.update_frequency == 0:
                loss_info = self._perform_learning()
                if loss_info and "loss" in loss_info:
                    episode_losses.append(loss_info["loss"])
                    
                    # Log learning step
                    if self.verbosity == "verbose":
                        self._log_event(episode_id, {
                            "type": "learning",
                            "episode": episode,
                            "step": step,
                            "loss": loss_info["loss"],
                            "q_mean": loss_info.get("q_mean", 0),
                            "epsilon": loss_info.get("epsilon", 0)
                        })
            
            # Update progress tracking
            if episode_progress and episode_task:
                episode_progress.update(episode_task, advance=1)
            
            # Use memory router if agent has one
            if hasattr(self.agent, "memory_router") and hasattr(self.agent.memory_router, "log_transition"):
                self.agent.memory_router.log_transition(
                    self.agent.agent_id if hasattr(self.agent, "agent_id") else "Agent",
                    state,
                    action,
                    reward, 
                    next_state,
                    priority=abs(reward) + 0.01
                )
            
            # Update cumulative reward and steps
            episode_reward += reward
            steps_taken = step
            
            # Log step info
            if self.verbosity != "quiet" and (step % 10 == 0 or reward != 0):
                action_str = action if isinstance(action, str) else f"Action {action}"
                console.print(f"Step {step}: {action_str}, Reward: {reward:.2f}")
            
            # Log major rewards
            if abs(reward) >= 10:
                console.print(f"[bold {'green' if reward > 0 else 'red'}]{'🎯' if reward > 0 else '❌'} Step {step}: {reward:+.2f}[/bold {'green' if reward > 0 else 'red'}]")
                
                # Log significant reward event
                self._log_event(episode_id, {
                    "type": "significant_reward",
                    "episode": episode,
                    "step": step,
                    "action": action if isinstance(action, int) else action.split()[0] if action else "None",
                    "reward": reward,
                    "phase": current_phase,
                    "info": info
                })
                
            # Update state for next iteration
            state = next_state
            
            # Break loop if done
            if done:
                if self.verbosity != "quiet":
                    console.print(f"[bold green]✅ Episode {episode} completed in {step} steps.[/bold green]")
                
                # Log episode completion
                self._log_event(episode_id, {
                    "type": "episode_complete", 
                    "episode": episode,
                    "steps": step,
                    "reward": episode_reward
                })
                    
                break
        
        # Return episode metrics
        return episode_reward, steps_taken, phase_transitions, unique_actions, episode_losses

    def _store_experience(self, state, action, reward, next_state, done, phase):
        """Store experience in replay buffer with proper formatting."""
        # Check if agent has a replay buffer
        if not hasattr(self.agent, "replay_buffer") or not self.agent.replay_buffer:
            return
            
        # Format state according to agent's needs
        if hasattr(self.agent, "encode_env_state"):
            state_tensor = self.agent.encode_env_state(state)
            next_state_tensor = self.agent.encode_env_state(next_state)
            
            # Format experience tuple
            experience = {
                'state': state_tensor.cpu().numpy() if isinstance(state_tensor, torch.Tensor) else state_tensor,
                'action': action,
                'reward': reward,
                'next_state': next_state_tensor.cpu().numpy() if isinstance(next_state_tensor, torch.Tensor) else next_state_tensor,
                'done': done,
                'phase': phase
            }
        else:
            # Raw state format
            experience = {
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': done,
                'phase': phase
            }
            
        # Add to buffer with prioritization based on reward magnitude
        priority = abs(reward) + 0.01  # Ensure non-zero priority
        if done:
            priority *= 2.0  # Prioritize terminal states
            
        # Store in replay buffer
        if hasattr(self.agent.replay_buffer, "add"):
            self.agent.replay_buffer.add(experience, priority=priority)
        
    def _perform_learning(self):
        """Perform one step of learning using the agent."""
        if not hasattr(self.agent, "replay_buffer") or not self.agent.replay_buffer:
            return None
            
        # Delegate to appropriate learning method
        if hasattr(self.agent, "train_on_batch"):
            # Modern approach with batch training
            return self.agent.train_on_batch(batch_size=self.batch_size)
        elif hasattr(self.agent, "optimize_model"):
            # Traditional approach
            return self.agent.optimize_model(batch_size=self.batch_size)
        else:
            # Fallback for custom learning approaches
            if hasattr(self.agent, "learn"):
                return self.agent.learn()
                
        return None

    def _log_episode_results(self, episode, reward, steps, phase_changes, unique_actions):
        """
        Log episode results with rich formatting.
        """
        # Skip detailed logging in quiet mode
        if self.verbosity == "quiet":
            if episode % 10 == 0:  # Only log every 10 episodes in quiet mode
                console.print(f"Episode {episode}/{self.episodes}: Reward={reward:.2f}, Steps={steps}")
            return
            
        # Create episode summary table
        table = Table(title=f"Episode {episode} Summary", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Reward", f"{reward:.2f}")
        table.add_row("Steps", f"{steps}")
        table.add_row("Phase Transitions", f"{phase_changes}")
        table.add_row("Unique Actions", f"{unique_actions}")
        
        # Add moving average if we have enough episodes
        if len(self.episode_rewards) >= 5:
            avg_reward = sum(self.episode_rewards[-5:]) / 5
            table.add_row("Avg Reward (5 eps)", f"{avg_reward:.2f}")
            
        # Add overall statistics
        if len(self.episode_rewards) > 1:
            max_reward = max(self.episode_rewards)
            table.add_row("Best Reward", f"{max_reward:.2f}")
            
        # Add success rate if applicable
        if episode >= 10:
            success_rate = (self.successful_episodes / episode) * 100
            table.add_row("Success Rate", f"{success_rate:.1f}%")
            
            # Add agent loss statistics if available
            if self.agent_losses:
                avg_loss = sum(self.agent_losses[-100:]) / min(len(self.agent_losses), 100)
                table.add_row("Recent Avg Loss", f"{avg_loss:.4f}")
            
        # Show episode performance metrics
        console.print(Panel(table, border_style="green"))
        
        # Log to file in JSONL format for better analysis
        log_entry = {
            "timestamp": time.time(),
            "episode": episode,
            "reward": reward,
            "steps": steps, 
            "phase_changes": phase_changes,
            "unique_actions": unique_actions,
            "success_rate": (self.successful_episodes / episode) * 100 if episode > 0 else 0
        }
        
        # Add average reward if available
        if len(self.episode_rewards) >= 5:
            log_entry["avg_reward_5"] = sum(self.episode_rewards[-5:]) / 5
            
        # Add learning stats if available
        if self.agent_losses:
            log_entry["avg_loss"] = sum(self.agent_losses[-100:]) / min(len(self.agent_losses), 100)
            
        # Write to JSONL file
        with open(self.event_log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
            
        # Create visualization snapshot periodically
        if hasattr(self.visualizer, "save_visualization_snapshot") and episode % 10 == 0:
            if self.visualizer:
                self.visualizer.save_visualization_snapshot(
                os.path.join(self.log_dir, f"visualization_ep{episode}.txt")
            )

    def _log_episode_results_enhanced(self, episode, reward, steps, phase_changes, unique_actions):
        """
        Enhanced episode logging with better positioning and clear command/output visibility.
        """
        # Skip detailed logging in quiet mode
        if self.verbosity == "quiet":
            if episode % 10 == 0:
                console.print(f"[bold cyan]Episode {episode}/{self.episodes}[/bold cyan]: Reward={reward:.2f}, Steps={steps}")
            return
            
        # Clear screen area for better visibility (optional)
        if self.verbosity == "verbose":
            console.print("\n" * 2)  # Add some space
            
        # Create enhanced episode summary with better positioning
        table = Table(title=f"🎯 Episode {episode}/{self.episodes} Results", show_header=True, width=80)
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Current", style="green", width=15, justify="right")
        table.add_column("Trend", style="yellow", width=15, justify="center")
        table.add_column("Best/Avg", style="magenta", width=15, justify="right")
        table.add_column("Status", style="blue", width=15, justify="center")
        
        # Calculate trends and comparisons
        reward_trend = "📈" if episode > 1 and reward > self.episode_rewards[-2] else "📉" if episode > 1 and reward < self.episode_rewards[-2] else "➡️"
        best_reward = max(self.episode_rewards) if self.episode_rewards else reward
        avg_reward_5 = sum(self.episode_rewards[-5:]) / min(5, len(self.episode_rewards)) if self.episode_rewards else reward
        
        # Determine status
        if reward > best_reward * 0.9:
            status = "🟢 EXCELLENT"
        elif reward > avg_reward_5:
            status = "🟡 GOOD"
        else:
            status = "🔴 BELOW AVG"
            
        table.add_row("Reward", f"{reward:.2f}", reward_trend, f"Best: {best_reward:.2f}", status)
        table.add_row("Steps", f"{steps}", "📊", f"Avg: {sum(self.episode_steps)/max(1,len(self.episode_steps)):.1f}" if self.episode_steps else f"{steps}", "⏱️")
        table.add_row("Phase Changes", f"{phase_changes}", "🔄", f"Total: {sum(self.phase_transitions_per_episode)}" if self.phase_transitions_per_episode else f"{phase_changes}", "📋")
        table.add_row("Unique Actions", f"{unique_actions}", "🎯", f"Avg: {sum(self.unique_actions_per_episode)/max(1,len(self.unique_actions_per_episode)):.1f}" if self.unique_actions_per_episode else f"{unique_actions}", "🔧")
        
        # Add success metrics if available
        if episode >= 10:
            success_rate = (self.successful_episodes / episode) * 100
            success_status = "🟢 HIGH" if success_rate > 70 else "🟡 MED" if success_rate > 40 else "🔴 LOW"
            table.add_row("Success Rate", f"{success_rate:.1f}%", "📈", f"Target: 70%", success_status)
            
        # Display with enhanced visibility
        console.print(Panel(
            table, 
            title="[bold green]🏆 EPISODE PERFORMANCE REPORT[/bold green]",
            border_style="bright_green",
            padding=(1, 2)
        ))
        
        # Add command visibility separator
        if self.verbosity in ["verbose", "debug"]:
            console.print("─" * 100, style="dim")
            console.print(f"[bold blue]Ready for Episode {episode + 1} commands...[/bold blue]")
            console.print("─" * 100, style="dim")
        
        # Enhanced file logging
        log_entry = {
            "timestamp": time.time(),
            "episode": episode,
            "reward": reward,
            "steps": steps,
            "phase_changes": phase_changes,
            "unique_actions": unique_actions,
            "best_reward": best_reward,
            "avg_reward_5ep": avg_reward_5,
            "success_rate": (self.successful_episodes / episode) * 100 if episode >= 10 else 0
        }
        self._log_event(f"episode_enhanced_{episode}", log_entry)

    def _save_model(self, episode):
        """
        Save model checkpoints with proper metadata.
        """
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        # Save agent model with appropriate method
        if hasattr(self.agent, "save_models"):
            # Modern approach with multiple models
            model_prefix = os.path.join(self.save_dir, f"ep{episode}_")
            self.agent.save_models(prefix=model_prefix)
        elif hasattr(self.agent, "policy_net") and hasattr(self.agent.policy_net, "save"):
            # Traditional approach with policy network
            policy_path = os.path.join(self.save_dir, f"ep{episode}_policy.pt")
            self.agent.policy_net.save(policy_path)
            
            # Save target network if available
            if hasattr(self.agent, "target_net") and hasattr(self.agent.target_net, "save"):
                target_path = os.path.join(self.save_dir, f"ep{episode}_target.pt")
                self.agent.target_net.save(target_path)
        elif hasattr(self.agent, "save"):
            # Simple save method
            path = os.path.join(self.save_dir, f"ep{episode}_agent.pt")
            self.agent.save(path)
            
        # Save checkpoint with training metrics
        metrics = {
            "episode": episode,
            "total_steps": self.total_steps,
            "episode_rewards": self.episode_rewards,
            "episode_steps": self.episode_steps,
            "episode_phases": self.episode_phases,
            "unique_actions": self.unique_actions_per_episode,
            "phase_transitions": self.phase_transitions_per_episode,
            "successful_episodes": self.successful_episodes,
            "timestamp": time.time()
        }
        
        # Create metrics file
        metrics_path = os.path.join(self.save_dir, f"ep{episode}_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Save consolidated metrics file with all episodes
        all_metrics_path = os.path.join(self.save_dir, "training_metrics.json")
        with open(all_metrics_path, "w") as f:
            json.dump({
                "episode": episode,
                "total_episodes": self.episodes,
                "episode_rewards": self.episode_rewards,
                "episode_steps": self.episode_steps,
                "episode_phases": self.episode_phases,
                "unique_actions": self.unique_actions_per_episode,
                "phase_transitions": self.phase_transitions_per_episode,
                "successful_episodes": self.successful_episodes
            }, f, indent=2)
        
        if self.verbosity != "quiet":
            console.print(f"[bold blue]💾 Model checkpoint saved at episode {episode}[/bold blue]")

    def _consolidate_memory(self):
        """
        Perform memory consolidation using the memory router.
        """
        if self.memory_router:
            # Log consolidation
            if self.verbosity != "quiet":
                console.print("[yellow]📦 Consolidating memory...[/yellow]")
                
            # Call memory consolidation
            if hasattr(self.memory_router, "sync_global_insights"):
                self.memory_router.sync_global_insights()
            
            # Optimize agent's replay buffer if available
            if hasattr(self.agent, "replay_buffer") and hasattr(self.agent.replay_buffer, "optimize_and_deduplicate"):
                removed = self.agent.replay_buffer.optimize_and_deduplicate()
                if removed > 0 and self.verbosity != "quiet":
                    console.print(f"[yellow]🧹 Removed {removed} redundant experiences[/yellow]")
                
            # Perform memory optimization if available
            if hasattr(self.memory_router, "optimize_memories"):
                self.memory_router.optimize_memories()  # type: ignore
                
            # Generate memory snapshot if needed
            if hasattr(self.memory_router, "snapshot_all_memories"):
                self.memory_router.snapshot_all_memories()

    def _update_adaptive_parameters(self, episode, reward, unique_actions, phase_transitions):
        """
        Update exploration parameters based on episode performance.
        This implements adaptive exploration to help agents overcome stagnation.
        """
        # Skip early episodes while the agent is still learning
        if episode < 5:
            return
            
        # Check for exploration stagnation (too few unique actions)
        if unique_actions < 3 and hasattr(self.agent, "epsilon"):
            # Increase exploration if too few unique actions
            original_epsilon = self.agent.epsilon
            self.agent.epsilon = min(0.8, self.agent.epsilon * 1.2)
            
            if self.verbosity != "quiet" and original_epsilon != self.agent.epsilon:
                console.print(f"[yellow]⚠ Low action diversity. Increasing exploration to ε={self.agent.epsilon:.3f}[/yellow]")
                
        # Check for phase stagnation (no transitions in multiple episodes)
        if phase_transitions == 0 and len(self.phase_transitions_per_episode) >= 3 and sum(self.phase_transitions_per_episode[-3:]) == 0:
            # Agent is stuck in a phase
            if self.verbosity != "quiet":
                console.print("[yellow]⚠ Phase stagnation detected. Applying interventions.[/yellow]")
                
            # Apply interventions if agent has appropriate methods
            if hasattr(self.agent, "epsilon") and hasattr(self.agent, "epsilon_min"):
                original_epsilon = self.agent.epsilon
                self.agent.epsilon = max(self.agent.epsilon * 1.5, self.agent.epsilon_min + 0.1)
                
                if self.verbosity != "quiet" and original_epsilon != self.agent.epsilon:
                    console.print(f"[yellow]⚠ Increased exploration to ε={self.agent.epsilon:.3f} due to phase stagnation[/yellow]")
                
            # Signal OrionAgent if available
            if self.agent_manager and hasattr(self.agent_manager, "orion_agent") and self.agent_manager.orion_agent:
                if hasattr(self.agent_manager.orion_agent, "notify_stagnation"):
                    stagnation_data = {
                        "agent_id": self.agent.agent_id if hasattr(self.agent, "agent_id") else "Agent",
                        "type": "phase",
                        "episode": episode,
                        "phase_transitions": phase_transitions
                    }
                    self.agent_manager.orion_agent.notify_stagnation(
                        self.agent.agent_id if hasattr(self.agent, "agent_id") else "Agent",
                        stagnation_data
                    )
        
        # Adjust learning rate if performance is poor over extended periods
        if episode >= 20 and episode % 10 == 0:
            # Check for sustained poor performance
            recent_rewards = self.episode_rewards[-10:]
            avg_recent = sum(recent_rewards) / 10
            
            # If rewards are declining or consistently low
            if avg_recent < -5:
                # Adjust learning rate if agent has optimizer
                if hasattr(self.agent, "optimizer") and hasattr(self.agent.optimizer, "param_groups"):
                    for param_group in self.agent.optimizer.param_groups:
                        if "lr" in param_group:
                            param_group['lr'] = max(param_group['lr'] * 0.8, 0.00001)  # Reduce learning rate
                            
                            if self.verbosity != "quiet":
                                console.print(f"[yellow]⚠ Reducing learning rate to {param_group['lr']:.6f} due to poor performance[/yellow]")

    def _log_event(self, episode_id, event_data):
        """Log an event to the JSONL event log."""
        if not hasattr(self, "event_log_file"):
            return
            
        # Add common fields
        event_data["episode_id"] = episode_id
        event_data["timestamp"] = time.time()
        
        # Write to log file
        with open(self.event_log_file, "a") as f:
            f.write(json.dumps(event_data) + "\n")

    def _final_report(self):
        """
        Generate final training report with comprehensive statistics.
        """
        # Calculate training time
        training_time = time.time() - (self.train_start_time or 0)
        hours, remainder = divmod(training_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        console.rule("[bold green]🏁 Training Completed[/bold green]")
        
        # Create summary table
        table = Table(title="Training Summary", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        # Add summary metrics
        table.add_row("Episodes", f"{self.episodes}")
        table.add_row("Total Steps", f"{self.total_steps}")
        
        if self.episode_rewards:
            table.add_row("Max Reward", f"{max(self.episode_rewards):.2f}")
            table.add_row("Final Avg Reward (10 eps)", f"{sum(self.episode_rewards[-10:]) / min(10, len(self.episode_rewards)):.2f}")
        
        # Success metrics
        table.add_row("Success Rate", f"{(self.successful_episodes / self.episodes) * 100:.1f}% ({self.successful_episodes}/{self.episodes})")
        table.add_row("Early Terminations", f"{self.early_terminations} ({(self.early_terminations / self.episodes) * 100:.1f}%)")
        table.add_row("Training Time", f"{int(hours)}h {int(minutes)}m {int(seconds)}s")
        
        console.print(Panel(table, border_style="green"))
        
        # Create final visualization
        if self.visualizer and hasattr(self.visualizer, "create_training_report"):
            self.visualizer.create_training_report(self.episodes)
            
        # Save final training report
        report = {
            "episodes": self.episodes,
            "total_steps": self.total_steps,
            "success_rate": (self.successful_episodes / self.episodes) if self.episodes > 0 else 0,
            "early_terminations": self.early_terminations,
            "max_reward": max(self.episode_rewards) if self.episode_rewards else 0,
            "final_avg_reward": sum(self.episode_rewards[-10:]) / min(10, len(self.episode_rewards)) if self.episode_rewards else 0,
            "training_time_seconds": training_time,
            "completion_date": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save report to file
        report_path = os.path.join(self.save_dir, "training_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
            
        # Plot learning curve if matplotlib is available
        try:
            self._plot_learning_curve()
        except ImportError:
            console.print("[yellow]⚠ matplotlib not available, skipping learning curve plot[/yellow]")

    def _plot_learning_curve(self):
        """
        Plot comprehensive learning curves showing reward progression and other metrics.
        """
        try:
            import matplotlib  # type: ignore
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt  # type: ignore
        except ImportError:
            console.print("[yellow]⚠ matplotlib not available, skipping learning curve plot[/yellow]")
            return
            
            # Create figure with multiple subplots
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
            
            # Plot reward curve
            episodes = list(range(1, len(self.episode_rewards) + 1))
            ax1.plot(episodes, self.episode_rewards, 'b-')
            ax1.set_title('Episode Rewards')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.grid(True)
            
            # Add smoothed moving average
            if len(self.episode_rewards) >= 10:
                window_size = 10
                smoothed_rewards = []
                for i in range(len(self.episode_rewards) - window_size + 1):
                    avg = sum(self.episode_rewards[i:i+window_size]) / window_size
                    smoothed_rewards.append(avg)
                ax1.plot(list(range(window_size, len(self.episode_rewards) + 1)), 
                         smoothed_rewards, 'r-', label=f'{window_size}-ep Moving Avg')
                ax1.legend()
            
            # Plot metrics (unique actions & phase transitions)
            ax2.plot(episodes, self.unique_actions_per_episode, 'g-', label='Unique Actions')
            ax2.plot(episodes, self.phase_transitions_per_episode, 'r-', label='Phase Transitions')
            ax2.set_title('Training Metrics')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Count')
            ax2.legend()
            ax2.grid(True)
            
            # Plot loss if available
            if self.agent_losses:
                # Group losses by episode (average per episode)
                avg_losses = []
                loss_x = []
                batch_size = max(1, len(self.agent_losses) // len(self.episode_rewards))
                
                for i in range(0, len(self.agent_losses), batch_size):
                    batch = self.agent_losses[i:i+batch_size]
                    if batch:
                        avg_losses.append(sum(batch) / len(batch))
                        loss_x.append(i // batch_size + 1)
                
                ax3.plot(loss_x, avg_losses, 'c-')
                ax3.set_title('Training Loss')
                ax3.set_xlabel('Episode')
                ax3.set_ylabel('Loss')
                ax3.grid(True)
            else:
                # Plot steps per episode if loss not available
                ax3.plot(episodes, self.episode_steps, 'm-')
                ax3.set_title('Steps per Episode')
                ax3.set_xlabel('Episode')
                ax3.set_ylabel('Steps')
                ax3.grid(True)
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(self.log_dir, 'learning_curve.png'))
            plt.close()
            
            console.print("[green]✓ Learning curves plotted and saved[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠ Error plotting learning curve: {e}[/yellow]")
            
    def load_checkpoint(self, checkpoint_path):
        """Load a training checkpoint to continue training."""
        if not os.path.exists(checkpoint_path):
            console.print(f"[red]❌ Checkpoint not found: {checkpoint_path}[/red]")
            return False
            
        try:
            # Load checkpoint data
            checkpoint = torch.load(checkpoint_path)
            
            if "episode_rewards" in checkpoint:
                self.episode_rewards = checkpoint["episode_rewards"]
            if "episode_steps" in checkpoint:
                self.episode_steps = checkpoint["episode_steps"]
            if "total_steps" in checkpoint:
                self.total_steps = checkpoint["total_steps"]
            
            # Update agent if it has load method
            if hasattr(self.agent, "load_checkpoint"):
                self.agent.load_checkpoint(checkpoint_path)
                
            console.print(f"[green]✓ Loaded checkpoint from {checkpoint_path}[/green]")
            return True
        except Exception as e:
            console.print(f"[red]❌ Error loading checkpoint: {e}[/red]")
            return False

# ─────────────────────────────────────────────
# 🎬 CLI Test Mode
# ─────────────────────────────────────────────
if __name__ == "__main__":
    console.print("[bold magenta]🚀 ARIASKA Training Module Test Mode[/bold magenta]")
    
    # Import necessary components
    from core.agents.red_agent import RedAgent
    from core.environment.cyber_environment import CyberEnvironment
    from core.algorithms.replay_buffer import ReplayBuffer
    
    # Create agent and components
    replay_buffer = ReplayBuffer(capacity=10000, prioritized=True)
    agent = RedAgent(device="cpu")
    env = CyberEnvironment()
    memory_router = MemoryRouter()
    
    # Create trainer
    trainer = Trainer(
        agent=agent,
        environment=env,
        episodes=2,
        max_steps=5,
        memory_router=memory_router,
        verbosity="detailed",
        use_prioritized_replay=True
    )
    
    # Run training
    trainer.run()
