# core/utils/stats_monitor.py — ARIASKA StatsMonitor v12.0 APEX PRIME
"""
Unified StatsMonitor:
• Live Rich Dashboard + Async Logging
• Prioritized Metrics Tracking
• Orion Insights & Alert System
• Lightweight Fallback Mode
"""

import os
import time
import json
import logging
import threading
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, List, Any, Optional

# Optional Rich Integration
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.progress import (
        Progress,
        BarColumn,
        TextColumn,
        TimeElapsedColumn,
        SpinnerColumn,
    )
    from rich.live import Live
    from rich.errors import LiveError

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

console = Console() if RICH_AVAILABLE else None
logger = logging.getLogger(__name__)

# Global flag to track if a live display is active
_LIVE_DISPLAY_ACTIVE = False


# ─────────────────────────────────────────────
# 📈 Progress Tracker
# ─────────────────────────────────────────────
class ProgressTracker:
    def __init__(self, total_steps=100, total_episodes=10):
        self.total_steps = total_steps
        self.total_episodes = total_episodes
        self.current_step = 0
        self.current_episode = 0
        self.progress = None
        self.task = None

    def start(self):
        if not RICH_AVAILABLE:
            return
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[cyan]{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        )
        self.task = self.progress.add_task(
            "Training Progress", total=self.total_steps * self.total_episodes
        )
        self.progress.start()

    def update(self, step=1, episode=0):
        self.current_step = step
        self.current_episode = episode
        if self.progress and self.task is not None:
            self.progress.update(
                self.task, completed=(episode * self.total_steps + step)
            )

    def stop(self):
        if self.progress:
            self.progress.stop()
            self.progress = None


# ─────────────────────────────────────────────
# 📝 Async Data Logger
# ─────────────────────────────────────────────
class DataLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.paths = {
            "step": os.path.join(log_dir, "step_log.jsonl"),
            "episode": os.path.join(log_dir, "episode_log.jsonl"),
            "gpt": os.path.join(log_dir, "gpt_usage.jsonl"),
            "alerts": os.path.join(log_dir, "alerts.jsonl"),
        }
        self.lock = threading.Lock()

    def _write_async(self, path, entry):
        def _write():
            with self.lock:
                with open(path, "a") as f:
                    f.write(json.dumps(entry) + "\n")

        threading.Thread(target=_write, daemon=True).start()

    def log_step(self, agent_id, reward, **info):
        entry = {
            "agent_id": agent_id,
            "reward": reward,
            "timestamp": time.time(),
            **info,
        }
        self._write_async(self.paths["step"], entry)

    def log_episode(self, agent_id, total_reward, **info):
        entry = {
            "agent_id": agent_id,
            "total_reward": total_reward,
            "timestamp": time.time(),
            **info,
        }
        self._write_async(self.paths["episode"], entry)

    def log_gpt_usage(self, agent_id, tokens):
        entry = {"agent_id": agent_id, "tokens": tokens, "timestamp": time.time()}
        self._write_async(self.paths["gpt"], entry)

    def log_alert(self, message, level="warning"):
        entry = {"message": message, "level": level, "timestamp": time.time()}
        self._write_async(self.paths["alerts"], entry)


# ─────────────────────────────────────────────
# 🖥️ Dashboard Renderer (Rich Hybrid)
# ─────────────────────────────────────────────
class DashboardRenderer:
    _instance = None
    _live_display_active = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(DashboardRenderer, cls).__new__(cls)
        return cls._instance

    def __init__(self, agents, max_history=100):
        if not hasattr(self, "initialized"):  # Prevent reinitialization
            self.agents = agents
            self.stats = {
                a: {"rewards": deque(maxlen=max_history), "gpt_calls": 0, "steps": 0}
                for a in agents
            }
            self.alerts = deque(maxlen=5)
            self.orion_insight = ""
            self.last_update = 0
            self.update_interval = 3
            self.live = None  # Don't create Live object until start() is called
            self.initialized = True

    def start(self):
        if not RICH_AVAILABLE:
            return
            
        # Don't start a new live display if one is already active
        if DashboardRenderer._live_display_active:
            return
            
        try:
            # Reset the terminal state before starting a new live display
            console.clear_live()
            
            # Create a new Live object only when starting
            self.live = Live(
                self.render(),
                console=console, 
                refresh_per_second=2,
                auto_refresh=False  # Manual refresh only
            )
            self.live.start()
            DashboardRenderer._live_display_active = True
        except Exception as e:
            console.print(f"[yellow]⚠ Live display error: {e}[/yellow]")

    def stop(self):
        if self.live and DashboardRenderer._live_display_active:
            try:
                self.live.stop()
                
                # Force a final refresh of the console to reset terminal state
                console.clear_live()
                console.print()  # Print an empty line to reset cursor
                
                self.live = None
                DashboardRenderer._live_display_active = False
            except Exception as e:
                console.print(f"[yellow]⚠ Error stopping live display: {e}[/yellow]")

    def update(self, force=False):
        now = time.time()
        if (force or now - self.last_update > self.update_interval) and self.live and DashboardRenderer._live_display_active:
            try:
                self.live.update(self.render())
                self.last_update = now
            except Exception as e:
                console.print(f"[yellow]⚠ Live update error: {e}[/yellow]")
                self.stop()  # Stop on error to prevent further issues

    def render(self):
        layout = Layout()
        layout.split(
            Layout(name="header", size=3),
            Layout(name="main", ratio=2),
            Layout(name="alerts", size=3),
            Layout(name="orion", size=4),
        )
        layout["header"].update(self._render_header())
        layout["main"].update(self._render_stats())
        layout["alerts"].update(self._render_alerts())
        layout["orion"].update(self._render_orion())
        return layout

    def _render_header(self):
        table = Table.grid(expand=True)
        table.add_column("Agent", style="cyan")
        table.add_column("Steps", style="magenta")
        table.add_column("Avg Reward", style="green")
        table.add_column("GPT Calls", style="yellow")
        for agent in self.agents:
            s = self.stats[agent]
            avg_reward = sum(s["rewards"]) / max(1, len(s["rewards"]))
            table.add_row(
                agent, str(s["steps"]), f"{avg_reward:.2f}", str(s["gpt_calls"])
            )
        return Panel(table, title="Training Progress", border_style="bright_blue")

    def _render_stats(self):
        table = Table(title="Agent Stats", show_lines=True)
        table.add_column("Agent", style="cyan")
        table.add_column("Last Reward", style="green")
        table.add_column("Avg Reward", style="magenta")
        table.add_column("GPT Calls", style="yellow")
        for agent in self.agents:
            s = self.stats[agent]
            last_reward = s["rewards"][-1] if s["rewards"] else 0.0
            avg_reward = sum(s["rewards"]) / max(1, len(s["rewards"]))
            table.add_row(
                agent, f"{last_reward:.2f}", f"{avg_reward:.2f}", str(s["gpt_calls"])
            )
        return Panel(table, border_style="green")

    def _render_alerts(self):
        if not self.alerts:
            return Panel(
                "[green]No alerts[/green]", title="Alerts", border_style="green"
            )
        table = Table(title="Alerts", show_lines=True)
        table.add_column("Level", style="red")
        table.add_column("Message", style="yellow")
        for level, msg in self.alerts:
            table.add_row(level, msg)
        return Panel(table, border_style="red")

    def _render_orion(self):
        return Panel(
            self.orion_insight or "[dim]Awaiting Orion Insight...[/dim]",
            title="Orion GPT Oversight",
            border_style="magenta",
        )

    def update_stats(self, agent_id, reward, gpt_calls=0):
        s = self.stats[agent_id]
        s["rewards"].append(reward)
        s["steps"] += 1
        s["gpt_calls"] = gpt_calls
        self.update()

    def log_alert(self, message, level="warning"):
        self.alerts.appendleft((level, message))
        self.update(force=True)

    def set_orion_insight(self, insight):
        self.orion_insight = insight
        self.update(force=True)


# ─────────────────────────────────────────────
# 🚨 Alert System Wrapper
# ─────────────────────────────────────────────
class AlertSystem:
    def __init__(self, dashboard: DashboardRenderer, datalogger: DataLogger):
        self.dashboard = dashboard
        self.datalogger = datalogger

    def warn(self, message):
        self.dashboard.log_alert(message, "warning")
        self.datalogger.log_alert(message, "warning")

    def error(self, message):
        self.dashboard.log_alert(message, "error")
        self.datalogger.log_alert(message, "error")

    def info(self, message):
        self.dashboard.log_alert(message, "info")
        self.datalogger.log_alert(message, "info")


# ─────────────────────────────────────────────
# 🧠 StatsMonitor Core Initialization
# ─────────────────────────────────────────────
class StatsMonitor:
    def __init__(self, agents=None, verbosity="standard", max_history=100):
        self.agents = agents or [
            "RedAgent",
            "BlueAgent",
            "ScoutAgent",
            "ShadowAgent",
            "OrionAgent",
        ]
        self.verbosity = verbosity
        self.progress = None  # Placeholder if needed later
        self.logger = DataLogger()
        self.dashboard = DashboardRenderer(self.agents, max_history=max_history)
        self.alerts = AlertSystem(self.dashboard, self.logger)
        self.metrics = defaultdict(list)
        self.global_steps = 0
        self.global_episodes = 0
        self.agent_stats = {agent: {"rewards": [], "steps": 0, "phases": {}, 
                                   "avg_reward": 0.0, "total_reward": 0.0,
                                   "commands": [], "current_phase": "recon"} 
                           for agent in self.agents}
        console.print(
            f"[cyan]StatsMonitor initialized for agents: {', '.join(self.agents)}[/cyan]"
        )
        
        # Start the dashboard if in verbose mode
        if self.verbosity not in ["quiet", "silent"]:
            self.dashboard.start()

    def log_step(self, agent_id, reward, **info):
        # Ensure agent exists in our tracking
        if agent_id not in self.agents:
            self.agents.append(agent_id)
            self.dashboard.stats[agent_id] = {
                "rewards": deque(maxlen=100),
                "gpt_calls": 0,
                "steps": 0,
            }
            self.agent_stats[agent_id] = {
                "rewards": [], 
                "steps": 0, 
                "phases": {}, 
                "avg_reward": 0.0,
                "total_reward": 0.0,
                "commands": [],
                "current_phase": "recon"
            }

        self.global_steps += 1
        
        # Update dashboard stats
        self.dashboard.update_stats(agent_id, reward, info.get("gpt_calls", 0))
        
        # Log to file
        self.logger.log_step(agent_id, reward, **info)

        # Track detailed metrics
        self.metrics[agent_id].append({
            "step": self.global_steps,
            "reward": reward,
            "gpt_calls": info.get("gpt_calls", 0),
            "timestamp": time.time(),
            "phase": info.get("phase", "unknown"),
            "command": info.get("command", "")
        })
        
        # Update agent's stats
        if agent_id in self.agent_stats:
            stats = self.agent_stats[agent_id]
            stats["rewards"].append(reward)
            stats["steps"] += 1
            stats["total_reward"] += reward
            
            # Calculate rolling average reward
            if len(stats["rewards"]) > 0:
                stats["avg_reward"] = stats["total_reward"] / stats["steps"]
                
            # Track command history
            if "command" in info and info["command"]:
                stats["commands"].append(info["command"])
                
            # Track phases
            if "phase" in info and info["phase"]:
                phase = info["phase"]
                stats["current_phase"] = phase
                if phase not in stats["phases"]:
                    stats["phases"][phase] = 0
                stats["phases"][phase] += 1

        # Update visualization if appropriate verbosity
        if self.verbosity != "quiet":
            self.dashboard.update()

    def log_episode(self, agent_id, total_reward, **info):
        self.global_episodes += 1
        self.logger.log_episode(agent_id, total_reward, **info)
        
        # Reset episode-specific stats while preserving cumulative stats
        if agent_id in self.agent_stats:
            # Keep a history of rewards but reset other episode-specific data
            self.agent_stats[agent_id]["total_reward"] = 0.0
        
        # Update visualization
        if self.verbosity != "quiet":
            self.dashboard.update(force=True)

    def report_gpt_usage(self, agent_id, tokens):
        self.dashboard.stats[agent_id]["gpt_calls"] += tokens
        self.logger.log_gpt_usage(agent_id, tokens)

    def set_orion_insight(self, insight):
        self.dashboard.set_orion_insight(insight)

    def warn(self, message):
        self.alerts.warn(message)

    def error(self, message):
        self.alerts.error(message)

    def info(self, message):
        self.alerts.info(message)

    def display_summary(self):
        table = Table(title="📊 ARIASKA Episode Summary", show_lines=True)
        table.add_column("Agent", style="cyan")
        table.add_column("Steps", justify="right")
        table.add_column("Avg Reward", justify="right")
        table.add_column("GPT Calls", justify="right")

        for agent in self.agents:
            stats = self.dashboard.stats[agent]
            avg_reward = sum(stats["rewards"]) / max(1, len(stats["rewards"]))
            table.add_row(
                agent, str(stats["steps"]), f"{avg_reward:.2f}", str(stats["gpt_calls"])
            )

        console.print(table)

    def reset(self):
        """Reset stats for a fresh simulation cycle."""
        self.global_steps = 0
        self.global_episodes = 0
        self.metrics.clear()
        for agent in self.agents:
            self.agent_stats[agent]["rewards"] = []
            self.agent_stats[agent]["steps"] = 0
            self.agent_stats[agent]["total_reward"] = 0.0
            self.agent_stats[agent]["avg_reward"] = 0.0
            self.agent_stats[agent]["commands"] = []
            
        console.print("[yellow]🔄 StatsMonitor reset for new session.[/yellow]")

    def flush_logs(self):
        """Ensure async logs are persisted."""
        pass  # Handled by async logger

    def start_live(self):
        """Activate live dashboard."""
        self.dashboard.start()

    def stop_live(self):
        """Deactivate live dashboard."""
        self.dashboard.stop()

    def get_metrics_history(self, agent_id=None):
        """Retrieve detailed metrics for analysis."""
        if agent_id:
            return self.metrics.get(agent_id, [])
        return dict(self.metrics)

    def get_average_reward(self, agent_id=None):
        """Get the average reward for a specific agent or all agents."""
        if agent_id and agent_id in self.agent_stats:
            stats = self.agent_stats[agent_id]
            return stats["avg_reward"]
        
        # Fallback to dashboard stats if agent stats not available
        if agent_id and agent_id in self.dashboard.stats:
            rewards = self.dashboard.stats[agent_id]["rewards"]
            return sum(rewards) / max(1, len(rewards))
            
        # Return average across all agents
        all_rewards = []
        for agent_id, stats in self.agent_stats.items():
            all_rewards.extend(stats["rewards"])
        
        if not all_rewards:  # Check if all_rewards is empty
            return 0.0
            
        return sum(all_rewards) / max(1, len(all_rewards))

    def get_steps(self, agent_id=None):
        """Get the number of steps taken by a specific agent."""
        if agent_id and agent_id in self.agent_stats:
            return self.agent_stats[agent_id]["steps"]
        return 0
        
    def get_current_phase(self, agent_id=None):
        """Get the current phase for a specific agent."""
        if agent_id and agent_id in self.agent_stats:
            return self.agent_stats[agent_id]["current_phase"]
        return None

    def get_detection_rate(self, agent_id=None):
        """Get the detection rate for an agent (placeholder)."""
        # This could be calculated from environment alerts
        return 0.0

    def get_redundancy_rate(self, agent_id=None):
        """Get the command redundancy rate."""
        # Placeholder for redundancy tracking
        return 0.0

    def show(self):
        """Display a summary of current stats."""
        if not RICH_AVAILABLE:
            return
            
        table = Table(title="📊 Training Stats", box=True)
        table.add_column("Agent", style="cyan")
        table.add_column("Steps", style="yellow")
        table.add_column("Avg. Reward", style="green")
        table.add_column("Current Phase", style="magenta")
        
        for agent_id, stats in self.agent_stats.items():
            avg_reward = stats["avg_reward"]
            steps = stats["steps"]
            phase = stats["current_phase"]
            
            table.add_row(
                agent_id,
                str(steps),
                f"{avg_reward:.2f}",
                phase
            )
        
        console.print(table)
        
    @property
    def total_steps(self):
        """Get the total number of steps across all agents."""
        return self.global_steps


if __name__ == "__main__":
    monitor = StatsMonitor()
    monitor.start_live()
    import random

    for ep in range(1, 4):
        monitor.start_episode(ep)
        for step in range(30):
            reward = random.uniform(-1, 2)
            monitor.record_step(
                agent_id="RedAgent",
                reward=reward,
                is_exploit=random.random() < 0.1,
                is_alert=random.random() < 0.05,
                gpt_tokens=random.randint(0, 40),
            )
            time.sleep(0.1)
        monitor.end_episode()
    monitor.print_summary()
    monitor.stop_live()
