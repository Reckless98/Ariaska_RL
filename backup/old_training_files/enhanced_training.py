#!/usr/bin/env python3
"""
Enhanced Training System for ARIASKA_RL v2.0
🧠 Neural Network + GPT Hybrid Training | 📈 Progressive Learning | 🎯 Real-Time Metrics
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel
from rich.live import Live

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from core.agents.red_agent import RedAgent
from core.agents.blue_agent import BlueAgent
from core.environment.cyber_environment import CyberEnvironment
from core.multiagent.agent_manager import AgentManager
from core.multiagent.memory_router import MemoryRouter

console = Console()

class EnhancedTrainingSystem:
    """
    Enhanced training system that implements the comprehensive enhancement plan.
    """
    
    def __init__(
        self,
        episodes: int = 100,
        max_steps_per_episode: int = 50,
        save_interval: int = 10,
        log_dir: str = "logs/enhanced_training",
        model_dir: str = "models/enhanced"
    ):
        self.episodes = episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.save_interval = save_interval
        self.log_dir = Path(log_dir)
        self.model_dir = Path(model_dir)
        
        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Training metrics
        self.episode_rewards = []
        self.episode_steps = []
        self.gpt_usage_rates = []
        self.neural_confidence_scores = []
        self.learning_curves = []
        self.episode_commands = []  # Track commands executed
        self.episode_outputs = []   # Track command outputs
        
        # Initialize components
        self.memory_router = MemoryRouter()
        self.agent_manager = None
        self.environment = None
        
        console.print("[bold green]🚀 Enhanced Training System Initialized[/bold green]")
    
    def setup_agents(self):
        """Setup agents with enhanced training capabilities."""
        console.print("[cyan]🤖 Initializing Enhanced Agents...[/cyan]")
        
        # Create enhanced cyber environment
        self.environment = CyberEnvironment(defer_reset=False)
        
        # Create agent manager with default verbosity
        self.agent_manager = AgentManager(verbosity="standard")
        
        # Initialize Red Agent with enhanced capabilities
        self.red_agent = RedAgent(
            agent_id="EnhancedRedAgent",
            device="cuda" if sys.platform != "darwin" else "cpu"
        )
        
        # Initialize Blue Agent
        self.blue_agent = BlueAgent(
            agent_id="EnhancedBlueAgent", 
            device="cuda" if sys.platform != "darwin" else "cpu"
        )
        
        # Add agents to manager's agents list
        if hasattr(self.agent_manager, 'agents'):
            self.agent_manager.agents.extend([self.red_agent, self.blue_agent])
        
        console.print("[green]✓ Enhanced agents initialized and registered[/green]")
    
    def run_training(self):
        """Run the enhanced training loop."""
        console.print(f"[bold blue]🎯 Starting Enhanced Training for {self.episodes} episodes[/bold blue]")
        
        # Setup agents
        self.setup_agents()
        
        # Create training dashboard
        with Live(self._create_dashboard(), refresh_per_second=2) as live:
            for episode in range(1, self.episodes + 1):
                episode_start_time = time.time()
                
                # Run episode
                episode_metrics = self._run_episode(episode)
                
                # Update metrics
                self._update_metrics(episode_metrics)
                
                # Update dashboard
                live.update(self._create_dashboard())
                
                # Save models periodically
                if episode % self.save_interval == 0:
                    self._save_models(episode)
                
                # Log episode results
                self._log_episode(episode, episode_metrics, time.time() - episode_start_time)
                
                # Early stopping if performance is excellent
                if len(self.episode_rewards) >= 10:
                    recent_avg = np.mean(self.episode_rewards[-10:])
                    gpt_usage = np.mean(self.gpt_usage_rates[-10:]) if self.gpt_usage_rates else 1.0
                    
                    if recent_avg > 50 and gpt_usage < 0.3:
                        console.print(f"[green]🎉 Early stopping at episode {episode} - Excellent performance achieved![/green]")
                        break
        
        # Final evaluation and summary
        self._final_evaluation()
    
    def _run_episode(self, episode: int) -> Dict[str, Any]:
        """Run a single training episode."""
        # Reset environment - handle None case
        if self.environment is not None and hasattr(self.environment, 'reset'):
            state = self.environment.reset()
        else:
            state = None
            
        if not state:
            state = {
                "phase": "recon",
                "target": "10.10.10.10",
                "step": 0,
                "discovered_services": [],
                "current_access_level": 0
            }
        
        episode_reward = 0.0
        episode_steps = 0
        gpt_calls = 0
        neural_decisions = 0
        total_confidence = 0.0
        learning_loss = 0.0  # Initialize to avoid unbound variable
        episode_commands = []  # Track commands in this episode
        episode_outputs = []   # Track outputs in this episode
        
        for step in range(self.max_steps_per_episode):
            # Red agent action
            action_result = self.red_agent.act(state)
            
            # Track command and store it
            command = action_result.get("command", "")
            episode_commands.append(command)
            
            # Track decision source
            if action_result.get("decision_source") == "gpt":
                gpt_calls += 1
            elif action_result.get("decision_source") == "neural":
                neural_decisions += 1
                total_confidence += action_result.get("confidence", 0.0)
            
            # Execute action in environment
            if self.environment is not None and hasattr(self.environment, 'step'):
                # Real environment step - returns (state, reward, done, info)
                next_state_raw, reward, done, info = self.environment.step(action_result.get("command", ""))
                
                # Convert to expected format if needed
                if not isinstance(next_state_raw, dict):
                    env_result = {
                        "reward": reward if isinstance(reward, (int, float)) else 0.0,
                        "phase": state.get("phase", "recon"),
                        "discovered_services": [],
                        "access_level": 0,
                        "output": str(info.get("output", "")),
                        "done": done
                    }
                else:
                    env_result = next_state_raw
                    env_result["reward"] = reward if isinstance(reward, (int, float)) else 0.0
            else:
                # Simulate environment response
                env_result = {
                    "reward": 1.0,
                    "phase": state.get("phase"),
                    "discovered_services": state.get("discovered_services", []),
                    "access_level": state.get("current_access_level", 0),
                    "output": f"Simulated response to: {action_result.get('command', '')}",
                    "done": step >= self.max_steps_per_episode - 5
                }
            
            # Extract reward and next state
            reward = env_result.get("reward", 0.0)
            output = env_result.get("output", "")
            episode_outputs.append(output)  # Track the output
            
            next_state = {
                "phase": env_result.get("phase", state.get("phase")),
                "target": state.get("target"),
                "step": step + 1,
                "discovered_services": env_result.get("discovered_services", []),
                "current_access_level": env_result.get("access_level", 0),
                "output": output
            }
            
            # Agent learning
            if hasattr(self.red_agent, 'learn'):
                learning_loss = self.red_agent.learn(state, action_result, reward, next_state, False)
                if learning_loss is None:
                    learning_loss = 0.0
            
            # Blue agent response (if applicable) - temporarily disabled
            # TODO: Implement BlueAgent.act() method
            if False and hasattr(self.blue_agent, 'act'):
                # Blue agent detects and responds to red agent action
                blue_state = {"threat_detected": True, "action": action_result, "environment_state": env_result}
                blue_action = self.blue_agent.act(blue_state)
                if blue_action.get("block", False) or blue_action.get("blocked", False):
                    reward -= 3.0  # Penalty for being detected/blocked
            
            # Update metrics
            episode_reward += reward
            episode_steps += 1
            state = next_state
            
            # Check for episode termination
            if env_result.get("done", False) or reward >= 100:
                break
        
        # Calculate episode metrics
        total_decisions = gpt_calls + neural_decisions
        gpt_usage_rate = gpt_calls / total_decisions if total_decisions > 0 else 1.0
        avg_confidence = total_confidence / neural_decisions if neural_decisions > 0 else 0.0
        
        return {
            "episode": episode,
            "reward": episode_reward,
            "steps": episode_steps,
            "gpt_usage_rate": gpt_usage_rate,
            "neural_confidence": avg_confidence,
            "learning_loss": learning_loss,
            "gpt_calls": gpt_calls,
            "neural_decisions": neural_decisions,
            "commands": episode_commands,
            "outputs": episode_outputs
        }
    
    def _update_metrics(self, episode_metrics: Dict[str, Any]):
        """Update training metrics."""
        self.episode_rewards.append(episode_metrics["reward"])
        self.episode_steps.append(episode_metrics["steps"])
        self.gpt_usage_rates.append(episode_metrics["gpt_usage_rate"])
        self.neural_confidence_scores.append(episode_metrics["neural_confidence"])
        self.learning_curves.append(episode_metrics["learning_loss"])
        self.episode_commands.append(episode_metrics.get("commands", []))
        self.episode_outputs.append(episode_metrics.get("outputs", []))
    
    def _create_dashboard(self) -> Panel:
        """Create real-time training dashboard."""
        # Create metrics table
        table = Table(title="Enhanced Training Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Current", style="yellow")
        table.add_column("Average (Last 10)", style="green")
        table.add_column("Best", style="magenta")
        
        if self.episode_rewards:
            current_reward = self.episode_rewards[-1]
            avg_reward = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else np.mean(self.episode_rewards)
            best_reward = max(self.episode_rewards)
            
            table.add_row("Episode Reward", f"{current_reward:.2f}", f"{avg_reward:.2f}", f"{best_reward:.2f}")
        
        if self.gpt_usage_rates:
            current_gpt = self.gpt_usage_rates[-1]
            avg_gpt = np.mean(self.gpt_usage_rates[-10:]) if len(self.gpt_usage_rates) >= 10 else np.mean(self.gpt_usage_rates)
            min_gpt = min(self.gpt_usage_rates)
            
            table.add_row("GPT Usage Rate", f"{current_gpt:.3f}", f"{avg_gpt:.3f}", f"{min_gpt:.3f}")
        
        if self.neural_confidence_scores:
            current_conf = self.neural_confidence_scores[-1]
            avg_conf = np.mean(self.neural_confidence_scores[-10:]) if len(self.neural_confidence_scores) >= 10 else np.mean(self.neural_confidence_scores)
            max_conf = max(self.neural_confidence_scores)
            
            table.add_row("Neural Confidence", f"{current_conf:.3f}", f"{avg_conf:.3f}", f"{max_conf:.3f}")
        
        # Add latest command and output info
        latest_command = "No commands yet"
        latest_output = "No output yet"
        if self.episode_commands and len(self.episode_commands) > 0:
            latest_commands = self.episode_commands[-1]  # Get latest episode commands
            if latest_commands and len(latest_commands) > 0:
                latest_command = latest_commands[-1][:50] + "..." if len(latest_commands[-1]) > 50 else latest_commands[-1]
        
        if self.episode_outputs and len(self.episode_outputs) > 0:
            latest_outputs = self.episode_outputs[-1]  # Get latest episode outputs
            if latest_outputs and len(latest_outputs) > 0:
                latest_output = latest_outputs[-1][:50] + "..." if len(latest_outputs[-1]) > 50 else latest_outputs[-1]
        
        table.add_row("Latest Command", latest_command, "-", "-")
        table.add_row("Latest Output", latest_output, "-", "-")
        
        # Progress information
        episodes_completed = len(self.episode_rewards)
        progress_text = f"Episodes: {episodes_completed}/{self.episodes} | "
        
        if episodes_completed > 0:
            progress_text += f"Success Rate: {sum(1 for r in self.episode_rewards if r > 10) / episodes_completed:.1%}"
        
        # Create a group containing the table and progress text
        from rich.console import Group
        from rich.text import Text
        
        content_group = Group(
            table,
            Text(""),  # Empty line
            Text(progress_text)
        )
        
        return Panel(
            content_group,
            title="🧠 ARIASKA Enhanced Training Dashboard",
            border_style="blue"
        )
    
    def _save_models(self, episode: int):
        """Save model checkpoints."""
        checkpoint_dir = self.model_dir / f"episode_{episode}"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save Red Agent
        try:
            if hasattr(self.red_agent, 'save_checkpoint'):
                import asyncio
                result = asyncio.run(self.red_agent.save_checkpoint(str(checkpoint_dir / "red_agent")))
            elif hasattr(self.red_agent, 'save_state'):
                self.red_agent.save_state(str(checkpoint_dir / "red_agent"))
            elif hasattr(self.red_agent, 'save'):
                self.red_agent.save(str(checkpoint_dir / "red_agent"))
            else:
                # Fallback: save agent state as pickle if no specific save method exists
                import pickle
                agent_state = {
                    'agent_id': getattr(self.red_agent, 'agent_id', 'red_agent'),
                    'state': getattr(self.red_agent, '__dict__', {}),
                    'episode': episode
                }
                with open(checkpoint_dir / "red_agent_state.pkl", "wb") as f:
                    pickle.dump(agent_state, f)
                console.print(f"[yellow]⚠ Used fallback save method for red agent[/yellow]")
        except Exception as e:
            console.print(f"[yellow]⚠ Could not save red agent: {e}[/yellow]")
        
        # Save Blue Agent (if methods available)
        try:
            if hasattr(self.blue_agent, 'save_agent_state'):
                save_method = getattr(self.blue_agent, 'save_agent_state')
                save_method(str(checkpoint_dir / "blue_agent"))
            elif hasattr(self.blue_agent, 'save'):
                save_method = getattr(self.blue_agent, 'save')
                save_method(str(checkpoint_dir / "blue_agent.pkl"))
        except Exception as e:
            console.print(f"[yellow]⚠ Could not save blue agent: {e}[/yellow]")
        
        # Save training metrics
        metrics = {
            "episode": episode,
            "episode_rewards": self.episode_rewards,
            "episode_steps": self.episode_steps,
            "gpt_usage_rates": self.gpt_usage_rates,
            "neural_confidence_scores": self.neural_confidence_scores,
            "learning_curves": self.learning_curves
        }
        
        with open(checkpoint_dir / "training_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        console.print(f"[green]💾 Checkpoint saved at episode {episode}[/green]")
    
    def _log_episode(self, episode: int, metrics: Dict[str, Any], duration: float):
        """Log episode results."""
        log_entry = {
            "timestamp": time.time(),
            "episode": episode,
            "duration": duration,
            **metrics
        }
        
        log_file = self.log_dir / "training_log.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def _final_evaluation(self):
        """Perform final evaluation and generate report."""
        console.print("\n[bold blue]📊 Final Training Evaluation[/bold blue]")
        
        if not self.episode_rewards:
            console.print("[red]No training data available for evaluation[/red]")
            return
        
        # Calculate final metrics
        total_episodes = len(self.episode_rewards)
        avg_reward = np.mean(self.episode_rewards)
        final_gpt_usage = np.mean(self.gpt_usage_rates[-10:]) if len(self.gpt_usage_rates) >= 10 else 1.0
        final_confidence = np.mean(self.neural_confidence_scores[-10:]) if len(self.neural_confidence_scores) >= 10 else 0.0
        
        success_rate = sum(1 for r in self.episode_rewards if r > 10) / total_episodes
        improvement_rate = (self.episode_rewards[-1] - self.episode_rewards[0]) / max(abs(self.episode_rewards[0]), 1) if total_episodes > 1 else 0.0
        
        # Create final report
        report_table = Table(title="Training Summary")
        report_table.add_column("Metric", style="cyan")
        report_table.add_column("Value", style="yellow")
        report_table.add_column("Target", style="green")
        report_table.add_column("Status", style="magenta")
        
        # Add metrics to table
        report_table.add_row("Total Episodes", str(total_episodes), str(self.episodes), "✓" if total_episodes >= self.episodes else "⚠")
        report_table.add_row("Average Reward", f"{avg_reward:.2f}", "50.0", "✓" if avg_reward >= 50 else "⚠")
        report_table.add_row("Success Rate", f"{success_rate:.1%}", "70%", "✓" if success_rate >= 0.7 else "⚠")
        report_table.add_row("GPT Dependency", f"{final_gpt_usage:.1%}", "<30%", "✓" if final_gpt_usage < 0.3 else "⚠")
        report_table.add_row("Neural Confidence", f"{final_confidence:.3f}", ">0.7", "✓" if final_confidence > 0.7 else "⚠")
        report_table.add_row("Improvement Rate", f"{improvement_rate:.1%}", ">50%", "✓" if improvement_rate > 0.5 else "⚠")
        
        console.print(report_table)
        
        # Save final report
        final_report = {
            "training_completed": True,
            "total_episodes": total_episodes,
            "avg_reward": avg_reward,
            "success_rate": success_rate,
            "final_gpt_usage": final_gpt_usage,
            "final_confidence": final_confidence,
            "improvement_rate": improvement_rate,
            "episode_rewards": self.episode_rewards,
            "gpt_usage_rates": self.gpt_usage_rates,
            "neural_confidence_scores": self.neural_confidence_scores
        }
        
        with open(self.log_dir / "final_report.json", "w") as f:
            json.dump(final_report, f, indent=2)
        
        console.print(f"\n[green]🎉 Training completed! Final report saved to {self.log_dir / 'final_report.json'}[/green]")

def main():
    """Main training entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced ARIASKA Training System")
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--max-steps", type=int, default=50, help="Maximum steps per episode")
    parser.add_argument("--save-interval", type=int, default=10, help="Model save interval")
    parser.add_argument("--log-dir", type=str, default="logs/enhanced_training", help="Log directory")
    parser.add_argument("--model-dir", type=str, default="models/enhanced", help="Model directory")
    
    args = parser.parse_args()
    
    # Create and run training system
    training_system = EnhancedTrainingSystem(
        episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        save_interval=args.save_interval,
        log_dir=args.log_dir,
        model_dir=args.model_dir
    )
    
    try:
        training_system.run_training()
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Training failed with error: {e}[/red]")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
