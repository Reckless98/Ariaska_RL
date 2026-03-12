# core/utils/stats_monitor.py — ARIASKA StatsMonitor v14.0 ULTRA-SIMPLE
"""
Ultra-simple StatsMonitor with minimal dependencies
"""
import os
import time
import json
import logging
import threading
from collections import deque
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger(__name__)

# Try Rich import with graceful fallback
RICH_AVAILABLE = False
RichProgress = None
try:
    from rich.console import Console as RichConsole
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.progress import Progress as RichProgress
    from rich.live import Live
    RICH_AVAILABLE = True
    console = RichConsole()
except ImportError:
    # Simple fallback console
    class FallbackConsole:
        def print(self, *args, **kwargs):
            print(*args)
        def rule(self, *args, **kwargs):
            print("=" * 50)
    console = FallbackConsole()

# Unified Progress class that works with or without Rich
class Progress:
    def __init__(self):
        self._rich_progress = None
        if RICH_AVAILABLE and RichProgress is not None:
            try:
                self._rich_progress = RichProgress()
            except Exception:
                pass
    
    def add_task(self, description, total=None):
        if self._rich_progress:
            try:
                return self._rich_progress.add_task(description, total=total)
            except Exception:
                pass
        return 0
    
    def start(self):
        if self._rich_progress:
            try:
                self._rich_progress.start()
            except Exception:
                pass
    
    def update(self, task_id, completed=None):
        if self._rich_progress:
            try:
                self._rich_progress.update(task_id, completed=completed)
            except Exception:
                pass
    
    def stop(self):
        if self._rich_progress:
            try:
                self._rich_progress.stop()
            except Exception:
                pass

class StatsDict:
    """Type-safe stats dictionary"""
    def __init__(self):
        self.commands_executed: int = 0
        self.successful_commands: int = 0
        self.total_reward: float = 0.0
        self.avg_reward: float = 0.0
        self.last_action: str = 'None'
        self.exploration_rate: float = 0.0
        self.learning_rate: float = 0.001
        self.rewards: List[float] = []
        self.steps: int = 0
        self.gpt_calls: int = 0
        self.current_phase: str = 'recon'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'commands_executed': self.commands_executed,
            'successful_commands': self.successful_commands,
            'total_reward': self.total_reward,
            'avg_reward': self.avg_reward,
            'last_action': self.last_action,
            'exploration_rate': self.exploration_rate,
            'learning_rate': self.learning_rate,
            'rewards': self.rewards.copy(),
            'steps': self.steps,
            'gpt_calls': self.gpt_calls,
            'current_phase': self.current_phase
        }

class StatsMonitor:
    """Ultra-simple stats monitor with type safety."""
    
    def __init__(self, verbosity="standard", enable_live=True):
        self.verbosity = verbosity
        self.enable_live = enable_live and RICH_AVAILABLE
        
        # Core metrics with proper typing
        self.agent_stats: Dict[str, StatsDict] = {}
        
        # Episode tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100) 
        self.current_episode = 0
        self.current_step = 0
        self.global_steps = 0
        self.global_episodes = 0
        
        # Performance metrics
        self.training_start_time = time.time()
        self.alerts = []
        self.orion_insight = None
        
        if RICH_AVAILABLE:
            pass  # Suppress init noise — shown in system init table
        else:
            pass  # Silent init
    
    def _get_agent_stats(self, agent_id: str) -> StatsDict:
        """Get or create agent stats."""
        if agent_id not in self.agent_stats:
            self.agent_stats[agent_id] = StatsDict()
        return self.agent_stats[agent_id]
    
    def log_step(self, agent_id: str, action: str = "unknown", reward: float = 0.0, success: bool = True, **kwargs):
        """Log a training step."""
        try:
            stats = self._get_agent_stats(agent_id)
            stats.commands_executed += 1
            if success:
                stats.successful_commands += 1
            stats.total_reward += reward
            if stats.commands_executed > 0:
                stats.avg_reward = stats.total_reward / stats.commands_executed
            stats.last_action = str(action)
            stats.rewards.append(reward)
            stats.steps += 1
            
            # Track additional info
            if 'gpt_calls' in kwargs and isinstance(kwargs['gpt_calls'], int):
                stats.gpt_calls += kwargs['gpt_calls']
            if 'phase' in kwargs and isinstance(kwargs['phase'], str):
                stats.current_phase = kwargs['phase']
            
            self.current_step += 1
            self.global_steps += 1
                    
        except Exception as e:
            logger.warning(f"Failed to log step: {e}")
    
    def record_step(self, agent_id, reward, command=None, phase=None, **kwargs):
        """Record a training step (alias for log_step)."""
        self.log_step(agent_id, command or "unknown", reward, True, phase=phase, **kwargs)
    
    def log_episode(self, episode_reward: float = 0.0, episode_length: int = 0):
        """Log episode completion."""
        try:
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.current_episode += 1
            self.global_episodes += 1
            self.current_step = 0
                    
        except Exception as e:
            logger.warning(f"Failed to log episode: {e}")
    
    def start_episode(self, episode_num):
        """Start tracking a new episode."""
        self.current_episode = episode_num
        self.current_step = 0
    
    def end_episode(self):
        """End the current episode."""
        pass
    
    def add_alert(self, message: str):
        """Add an alert."""
        try:
            self.alerts.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "message": message
            })
            
            # Keep only recent alerts
            if len(self.alerts) > 10:
                self.alerts = self.alerts[-10:]
                
        except Exception as e:
            logger.warning(f"Failed to add alert: {e}")
    
    def set_orion_insight(self, insight: str):
        """Set Orion insight."""
        try:
            self.orion_insight = insight
        except Exception as e:
            logger.warning(f"Failed to set Orion insight: {e}")
    
    def start(self):
        """Start the monitor."""
        pass
    
    def stop(self):
        """Stop the monitor."""
        pass
    
    def show(self):
        """Show current stats."""
        if RICH_AVAILABLE:
            try:
                from rich.table import Table
                table = Table(title="📊 Training Stats")
                table.add_column("Agent", style="cyan")
                table.add_column("Steps", style="yellow")
                table.add_column("Avg. Reward", style="green")
                table.add_column("Commands", style="magenta")
                
                for agent_id, stats in self.agent_stats.items():
                    table.add_row(
                        agent_id,
                        str(stats.steps),
                        f"{stats.avg_reward:.2f}",
                        str(stats.commands_executed)
                    )
                
                console.print(table)
            except Exception:
                self._show_fallback()
        else:
            self._show_fallback()
    
    def _show_fallback(self):
        """Fallback stats display."""
        print(f"\n=== ARIASKA Stats ===")
        print(f"Episode: {self.current_episode}, Step: {self.current_step}")
        for agent_id, stats in self.agent_stats.items():
            print(f"{agent_id}: {stats.commands_executed} commands, {stats.avg_reward:.2f} avg reward")
    
    def print_summary(self):
        """Print comprehensive stats summary."""
        if RICH_AVAILABLE:
            try:
                from rich.table import Table
                table = Table(title="📊 Training Summary")
                table.add_column("Agent", style="cyan")
                table.add_column("Total Reward", style="green")
                table.add_column("Avg Reward", style="yellow")
                table.add_column("Episodes", style="magenta")
                table.add_column("Commands", style="white")

                for agent_id, stats in self.agent_stats.items():
                    table.add_row(
                        agent_id,
                        f"{stats.total_reward:.2f}",
                        f"{stats.avg_reward:.2f}",
                        str(self.global_episodes),
                        str(stats.commands_executed),
                    )

                console.print(table)
            except Exception:
                self._print_summary_fallback()
        else:
            self._print_summary_fallback()
    
    def _print_summary_fallback(self):
        """Fallback summary display."""
        print(f"=== Training Summary ===")
        for agent_id, stats in self.agent_stats.items():
            print(f"{agent_id}: Total: {stats.total_reward:.2f}, "
                  f"Avg: {stats.avg_reward:.2f}, Commands: {stats.commands_executed}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current stats as dictionary."""
        return {
            "episode": self.current_episode,
            "step": self.current_step,
            "agent_stats": {aid: stats.to_dict() for aid, stats in self.agent_stats.items()},
            "global_steps": self.global_steps,
            "global_episodes": self.global_episodes
        }
    
    def render_ascii_summary(self):
        """Render ASCII summary of training statistics."""
        try:
            if RICH_AVAILABLE:
                from rich.table import Table
                table = Table(title="Training Summary")
                table.add_column("Agent", style="cyan")
                table.add_column("Commands", style="green")
                table.add_column("Avg Reward", style="yellow")
                table.add_column("Current Phase", style="magenta")
                
                for agent_id, stats in self.agent_stats.items():
                    table.add_row(
                        agent_id,
                        str(stats.commands_executed),
                        f"{stats.avg_reward:.2f}",
                        stats.current_phase
                    )
                console.print(table)
            else:
                print("=== Training Summary ===")
                for agent_id, stats in self.agent_stats.items():
                    print(f"{agent_id}: Commands={stats.commands_executed}, Avg Reward={stats.avg_reward:.2f}")
        except Exception as e:
            print(f"Error rendering summary: {e}")
    
    def get_avg_reward(self, agent_id: Optional[str] = None) -> float:
        """Get average reward for specific agent or global average."""
        try:
            if agent_id and agent_id in self.agent_stats:
                return self.agent_stats[agent_id].avg_reward
            elif self.agent_stats:
                total_reward = sum(stats.total_reward for stats in self.agent_stats.values())
                total_commands = sum(stats.commands_executed for stats in self.agent_stats.values())
                return total_reward / total_commands if total_commands > 0 else 0.0
            return 0.0
        except Exception:
            return 0.0
    
    @property
    def total_steps(self) -> int:
        """Get total steps across all agents."""
        return self.global_steps
    
    def get_average_reward(self, agent_id: Optional[str] = None) -> float:
        """Get average reward for agent or all agents."""
        if agent_id:
            stats = self._get_agent_stats(agent_id)
            return stats.avg_reward
        else:
            if not self.episode_rewards:
                return 0.0
            return sum(self.episode_rewards) / len(self.episode_rewards)
    
    def get_metrics_history(self, agent_id=None):
        """Retrieve detailed metrics for analysis."""
        if agent_id:
            stats = self._get_agent_stats(agent_id)
            return stats.rewards.copy()
        return {aid: stats.rewards.copy() for aid, stats in self.agent_stats.items()}
    
    def log_gpt_call(self, agent_id, tokens_used=0, model="local-llm"):
        """Log GPT API call for tracking usage."""
        stats = self._get_agent_stats(agent_id)
        stats.gpt_calls += 1
    
    def report_gpt_usage(self, agent_id, tokens):
        """Report GPT usage."""
        self.log_gpt_call(agent_id, tokens)
    
    def warn(self, message):
        """Add warning alert."""
        self.add_alert(f"WARNING: {message}")
    
    def error(self, message):
        """Add error alert."""
        self.add_alert(f"ERROR: {message}")
    
    def info(self, message):
        """Add info alert."""
        self.add_alert(f"INFO: {message}")
    
    def update(self, *args, **kwargs):
        """Update method for compatibility."""
        pass
    
    def reset(self):
        """Reset stats for a fresh simulation cycle."""
        self.global_steps = 0
        self.global_episodes = 0
        self.current_episode = 0
        self.current_step = 0
        self.agent_stats.clear()
        self.episode_rewards.clear()
        self.episode_lengths.clear()
        self.alerts.clear()
        
        if RICH_AVAILABLE:
            console.print("[yellow]🔄 StatsMonitor reset for new session.[/yellow]")
        else:
            print("🔄 StatsMonitor reset for new session.")
    
    def start_live(self):
        """Activate live dashboard (placeholder)."""
        pass
    
    def stop_live(self):
        """Deactivate live dashboard (placeholder)."""
        pass
    
    def flush_logs(self):
        """Ensure logs are persisted."""
        pass
    
    def record_autonomous_interaction(self, interaction: Dict[str, Any]):
        """Record an autonomous agent interaction."""
        try:
            # Store interaction for later analysis
            if not hasattr(self, 'autonomous_interactions'):
                self.autonomous_interactions = []
            self.autonomous_interactions.append(interaction)
            
            # Keep only recent interactions
            if len(self.autonomous_interactions) > 100:
                self.autonomous_interactions = self.autonomous_interactions[-100:]
        except Exception as e:
            logger.warning(f"Failed to record autonomous interaction: {e}")

    def get_alert_rate(self) -> float:
        """Get the current alert rate."""
        try:
            if not hasattr(self, 'alert_history'):
                self.alert_history = []
            
            # Calculate alert rate from recent history
            recent_alerts = [alert for alert in self.alert_history if time.time() - alert.get('timestamp', 0) < 300]
            alert_rate = len(recent_alerts) / 300.0  # alerts per second over 5 minutes
            return min(1.0, alert_rate * 100)  # Convert to percentage, cap at 100%
        except Exception as e:
            logger.warning(f"Failed to get alert rate: {e}")
            return 0.0

    def get_detection_rate(self) -> float:
        """Get the current detection rate."""
        try:
            total_actions = sum(stats.commands_executed for stats in self.agent_stats.values())
            if total_actions == 0:
                return 0.0
            
            # Simple detection rate calculation based on failed commands
            failed_actions = sum(stats.commands_executed - stats.successful_commands for stats in self.agent_stats.values())
            detection_rate = failed_actions / total_actions
            return min(1.0, detection_rate)
        except Exception as e:
            logger.warning(f"Failed to get detection rate: {e}")
            return 0.0

    def visualize_phase_distribution(self):
        """Visualize the distribution of phases."""
        try:
            if RICH_AVAILABLE:
                from rich.table import Table
                table = Table(title="Phase Distribution")
                table.add_column("Agent", style="cyan")
                table.add_column("Current Phase", style="yellow")
                table.add_column("Commands", style="green")
                
                for agent_id, stats in self.agent_stats.items():
                    table.add_row(
                        agent_id,
                        stats.current_phase,
                        str(stats.commands_executed)
                    )
                
                console.print(table)
            else:
                print("=== Phase Distribution ===")
                for agent_id, stats in self.agent_stats.items():
                    print(f"{agent_id}: {stats.current_phase} ({stats.commands_executed} commands)")
        except Exception as e:
            logger.warning(f"Failed to visualize phase distribution: {e}")


# Legacy compatibility functions
def display_training_summary(agent_stats, total_episodes, total_steps):
    """Display training summary with fallback."""
    if RICH_AVAILABLE:
        try:
            from rich.table import Table
            table = Table(title="📊 ARIASKA Training Summary", show_lines=True)
            table.add_column("Agent", style="cyan")
            table.add_column("Commands", justify="right")
            table.add_column("Avg Reward", justify="right")

            for agent_id, stats in agent_stats.items():
                if isinstance(stats, StatsDict):
                    avg_reward = stats.avg_reward
                    commands = stats.commands_executed
                else:
                    avg_reward = stats.get("avg_reward", 0.0)
                    commands = stats.get("commands_executed", 0)
                table.add_row(agent_id, str(commands), f"{avg_reward:.2f}")

            console.print(table)
        except Exception:
            _display_training_summary_fallback(agent_stats, total_episodes, total_steps)
    else:
        _display_training_summary_fallback(agent_stats, total_episodes, total_steps)


def _display_training_summary_fallback(agent_stats, total_episodes, total_steps):
    """Fallback training summary display."""
    print("\n" + "="*60)
    print("📊 ARIASKA TRAINING SUMMARY")
    print("="*60)
    print(f"Episodes: {total_episodes}, Steps: {total_steps}")
    print("-"*60)
    
    for agent_id, stats in agent_stats.items():
        if isinstance(stats, StatsDict):
            avg_reward = stats.avg_reward
            commands = stats.commands_executed
        else:
            avg_reward = stats.get("avg_reward", 0.0)
            commands = stats.get("commands_executed", 0)
        print(f"{agent_id}: {commands} commands, {avg_reward:.2f} avg reward")
    print("="*60)


def display_phase_distribution_table(phase_stats):
    """Display phase distribution with fallback."""
    if RICH_AVAILABLE:
        try:
            from rich.table import Table
            table = Table(title="🎯 Phase Distribution")
            table.add_column("Phase", style="cyan")
            table.add_column("Count", style="green")
            table.add_column("Percentage", style="yellow")

            total = sum(phase_stats.values())
            for phase, count in phase_stats.items():
                percentage = (count / max(total, 1)) * 100
                table.add_row(phase, str(count), f"{percentage:.1f}%")

            console.print(table)
        except Exception:
            _display_phase_distribution_fallback(phase_stats)
    else:
        _display_phase_distribution_fallback(phase_stats)


def _display_phase_distribution_fallback(phase_stats):
    """Fallback phase distribution display."""
    print("\n=== Phase Distribution ===")
    total = sum(phase_stats.values())
    for phase, count in phase_stats.items():
        percentage = (count / max(total, 1)) * 100
        print(f"{phase}: {count} ({percentage:.1f}%)")


def display_training_stats_table(agent_stats):
    """Display training stats with fallback."""
    if RICH_AVAILABLE:
        try:
            from rich.table import Table
            table = Table(title="📊 Training Stats")
            table.add_column("Agent", style="cyan")
            table.add_column("Commands", style="yellow")
            table.add_column("Avg. Reward", style="green")
            
            for agent_id, stats in agent_stats.items():
                if isinstance(stats, StatsDict):
                    avg_reward = stats.avg_reward
                    commands = stats.commands_executed
                else:
                    avg_reward = stats.get("avg_reward", 0.0)
                    commands = stats.get("commands_executed", 0)
                
                table.add_row(agent_id, str(commands), f"{avg_reward:.2f}")
            
            console.print(table)
        except Exception:
            _display_training_stats_fallback(agent_stats)
    else:
        _display_training_stats_fallback(agent_stats)


def _display_training_stats_fallback(agent_stats):
    """Fallback training stats display."""
    print("\n=== Training Stats ===")
    for agent_id, stats in agent_stats.items():
        if isinstance(stats, StatsDict):
            avg_reward = stats.avg_reward
            commands = stats.commands_executed
        else:
            avg_reward = stats.get("avg_reward", 0.0)
            commands = stats.get("commands_executed", 0)
        print(f"{agent_id}: {commands} commands, {avg_reward:.2f} avg reward")


# Progress Tracker (Simplified)
class ProgressTracker:
    def __init__(self, total_steps=100, total_episodes=10):
        self.total_steps = total_steps
        self.total_episodes = total_episodes
        self.current_step = 0
        self.current_episode = 0
        self.progress = None

    def start(self):
        if RICH_AVAILABLE:
            try:
                self.progress = Progress()
                self.task = self.progress.add_task("Training Progress", total=self.total_steps * self.total_episodes)
                self.progress.start()
            except Exception:
                print("Progress tracking (Rich error)")
        else:
            print("Progress tracking started")

    def update(self, step=1, episode=0):
        self.current_step = step
        self.current_episode = episode
        if self.progress and hasattr(self, 'task'):
            try:
                self.progress.update(self.task, completed=(episode * self.total_steps + step))
            except Exception:
                pass

    def stop(self):
        if self.progress:
            try:
                self.progress.stop()
            except Exception:
                pass
