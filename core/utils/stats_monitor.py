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

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

console = Console() if RICH_AVAILABLE else None
logger = logging.getLogger(__name__)


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
    def __init__(self, agents, max_history=100):
        self.agents = agents
        self.stats = {
            a: {"rewards": deque(maxlen=max_history), "gpt_calls": 0, "steps": 0}
            for a in agents
        }
        self.alerts = deque(maxlen=5)
        self.orion_insight = ""
        self.last_update = 0
        self.update_interval = 3
        self.live = (
            Live(console=console, refresh_per_second=2) if RICH_AVAILABLE else None
        )

    def start(self):
        if self.live:
            self.live.start()

    def stop(self):
        if self.live:
            self.live.stop()

    def update(self, force=False):
        now = time.time()
        if force or now - self.last_update > self.update_interval:
            if self.live:
                self.live.update(self.render())
            self.last_update = now

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
        console.print(
            f"[cyan]StatsMonitor initialized for agents: {', '.join(self.agents)}[/cyan]"
        )

    def log_step(self, agent_id, reward, **info):
        if agent_id not in self.agents:
            self.agents.append(agent_id)
            self.dashboard.stats[agent_id] = {
                "rewards": deque(maxlen=100),
                "gpt_calls": 0,
                "steps": 0,
            }

        self.global_steps += 1
        self.dashboard.update_stats(agent_id, reward, info.get("gpt_calls", 0))
        self.logger.log_step(agent_id, reward, **info)

        # Track metrics
        self.metrics[agent_id].append(
            {
                "step": self.global_steps,
                "reward": reward,
                "gpt_calls": info.get("gpt_calls", 0),
                "timestamp": time.time(),
            }
        )

        if self.verbosity != "quiet":
            self.dashboard.update()

    def log_episode(self, agent_id, total_reward, **info):
        self.global_episodes += 1
        self.logger.log_episode(agent_id, total_reward, **info)
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
        self.dashboard.reset()
        self.console.print("[yellow]🔄 StatsMonitor reset for new session.[/yellow]")

    def flush_logs(self):
        """Ensure async logs are persisted."""
        pass  # Handled by async logger

    def start_live(self):
        """Activate live dashboard."""
        self.dashboard.start_live()

    def stop_live(self):
        """Deactivate live dashboard."""
        self.dashboard.stop_live()

    def get_metrics_history(self, agent_id=None):
        """Retrieve detailed metrics for analysis."""
        if agent_id:
            return self.metrics.get(agent_id, [])
        return dict(self.metrics)

    def get_average_reward(self, agent_id=None):
        if agent_id:
            rewards = self.dashboard.stats[agent_id]["rewards"]
            return sum(rewards) / max(1, len(rewards))
        all_rewards = []
        for stats in self.dashboard.stats.values():
            all_rewards.extend(stats["rewards"])
        return sum(all_rewards) / max(1, len(all_rewards))

    def get_redundancy_rate(self, agent_id=None):
        # Placeholder for redundancy tracking
        return 0.0

    def print_summary(self, recent_episodes: int = None):
        """Print summarized metrics over recent episodes."""
        history = self.dashboard.history
        if not history:
            self.console.print("[dim]No metrics history available.[/dim]")
            return

        episodes = list(history.values())
        if recent_episodes:
            episodes = episodes[-recent_episodes:]

        avg_reward = sum(ep["reward"] for ep in episodes) / len(episodes)
        avg_steps = sum(ep["steps"] for ep in episodes) / len(episodes)
        avg_exploits = sum(ep["exploits"] for ep in episodes) / len(episodes)
        avg_alerts = sum(ep["alerts"] for ep in episodes) / len(episodes)
        avg_tokens = sum(ep["tokens"] for ep in episodes) / len(episodes)

        summary = Table(title="📊 Training Summary", show_lines=True)
        summary.add_column("Metric", style="cyan")
        summary.add_column("Average", justify="right")
        summary.add_column("Total", justify="right")

        summary.add_row(
            "Reward", f"{avg_reward:.2f}", str(sum(ep["reward"] for ep in episodes))
        )
        summary.add_row(
            "Steps", f"{avg_steps:.1f}", str(sum(ep["steps"] for ep in episodes))
        )
        summary.add_row(
            "Exploits",
            f"{avg_exploits:.1f}",
            str(sum(ep["exploits"] for ep in episodes)),
        )
        summary.add_row(
            "Alerts", f"{avg_alerts:.1f}", str(sum(ep["alerts"] for ep in episodes))
        )
        summary.add_row(
            "GPT Tokens", f"{avg_tokens:.1f}", str(sum(ep["tokens"] for ep in episodes))
        )

        self.console.print(summary)

    # ─────────────────────────────────────────────
    # 🚀 Diagnostic CLI Mode
    # ─────────────────────────────────────────────


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
