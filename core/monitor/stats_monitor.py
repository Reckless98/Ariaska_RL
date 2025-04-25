from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, SpinnerColumn
from rich.live import Live
from collections import defaultdict, deque, Counter
import threading
import time
import json
import os

console = Console()

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
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[cyan]{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        )
        self.task = self.progress.add_task("Training Progress", total=self.total_steps * self.total_episodes)
        self.progress.start()

    def update(self, step=1, episode=0):
        self.current_step = step
        self.current_episode = episode
        if self.progress and self.task is not None:
            self.progress.update(self.task, completed=(episode * self.total_steps + step))

    def stop(self):
        if self.progress:
            self.progress.stop()
            self.progress = None

# ─────────────────────────────────────────────
# 📝 Data Logger (Async)
# ─────────────────────────────────────────────
class DataLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.step_log_path = os.path.join(log_dir, "step_log.jsonl")
        self.episode_log_path = os.path.join(log_dir, "episode_log.jsonl")
        self.gpt_log_path = os.path.join(log_dir, "gpt_usage.jsonl")
        self.lock = threading.Lock()

    def log_step(self, agent_id, reward, **info):
        entry = {"agent_id": agent_id, "reward": reward, "timestamp": time.time(), **info}
        self._write_async(self.step_log_path, entry)

    def log_episode(self, agent_id, total_reward, **info):
        entry = {"agent_id": agent_id, "total_reward": total_reward, "timestamp": time.time(), **info}
        self._write_async(self.episode_log_path, entry)

    def log_gpt_usage(self, agent_id, tokens):
        entry = {"agent_id": agent_id, "tokens": tokens, "timestamp": time.time()}
        self._write_async(self.gpt_log_path, entry)

    def log_alert(self, message, level="warning"):
        entry = {"message": message, "level": level, "timestamp": time.time()}
        self._write_async(os.path.join(self.log_dir, "alerts.jsonl"), entry)

    def _write_async(self, path, entry):
        def _write():
            with self.lock:
                with open(path, "a") as f:
                    f.write(json.dumps(entry) + "\n")
        threading.Thread(target=_write, daemon=True).start()

# ─────────────────────────────────────────────
# 🖥️ Dashboard Renderer
# ─────────────────────────────────────────────
class DashboardRenderer:
    def __init__(self, agents, max_history=100, verbosity="standard"):
        self.agents = agents
        self.max_history = max_history
        self.verbosity = verbosity
        self.agent_stats = {a: {"rewards": deque(maxlen=max_history), "gpt_calls": 0, "steps": 0, "redundancy": 0} for a in agents}
        self.live = None
        self.last_update = 0
        self.update_interval = 3  # seconds
        self.alerts = deque(maxlen=5)
        self.orion_insight = ""
        self.lock = threading.Lock()

    def start_live(self):
        if not self.live:
            self.live = Live(self.render(), console=console, refresh_per_second=2)
            self.live.start()

    def stop_live(self):
        if self.live:
            self.live.stop()
            self.live = None

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
        layout["main"].update(self._render_stats_table())
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
            stats = self.agent_stats[agent]
            avg_reward = sum(stats["rewards"]) / max(1, len(stats["rewards"]))
            table.add_row(agent, str(stats["steps"]), f"{avg_reward:.2f}", str(stats["gpt_calls"]))
        return Panel(table, title="Training Progress", border_style="bright_blue")

    def _render_stats_table(self):
        table = Table(title="Agent Stats", show_lines=True)
        table.add_column("Agent", style="cyan")
        table.add_column("Last Reward", style="green")
        table.add_column("Avg Reward", style="magenta")
        table.add_column("Redundancy", style="yellow")
        for agent in self.agents:
            stats = self.agent_stats[agent]
            last_reward = stats["rewards"][-1] if stats["rewards"] else 0.0
            avg_reward = sum(stats["rewards"]) / max(1, len(stats["rewards"]))
            redundancy = stats.get("redundancy", 0)
            table.add_row(agent, f"{last_reward:.2f}", f"{avg_reward:.2f}", str(redundancy))
        return Panel(table, border_style="green")

    def _render_alerts(self):
        if not self.alerts:
            return Panel("[green]No alerts[/green]", title="Alerts", border_style="green")
        table = Table(title="Alerts", show_lines=True)
        table.add_column("Level", style="red")
        table.add_column("Message", style="yellow")
        for level, msg in self.alerts:
            table.add_row(level, msg)
        return Panel(table, border_style="red")

    def _render_orion(self):
        return Panel(self.orion_insight or "[dim]No Orion insight yet[/dim]", title="OrionAgent GPT Insights", border_style="magenta")

    def log_alert(self, message, level="warning"):
        self.alerts.appendleft((level, message))
        self.update(force=True)

    def set_orion_insight(self, insight):
        self.orion_insight = insight
        self.update(force=True)

    def update_stats(self, agent_id, reward, gpt_calls=0, redundancy=0):
        stats = self.agent_stats[agent_id]
        stats["rewards"].append(reward)
        stats["steps"] += 1
        stats["gpt_calls"] = gpt_calls
        stats["redundancy"] = redundancy
        self.update()

    def reset(self):
        for stats in self.agent_stats.values():
            stats["rewards"].clear()
            stats["steps"] = 0
            stats["gpt_calls"] = 0
            stats["redundancy"] = 0
        self.alerts.clear()
        self.orion_insight = ""
        self.update(force=True)

# ─────────────────────────────────────────────
# 🚨 Alert System
# ─────────────────────────────────────────────
class AlertSystem:
    def __init__(self, dashboard: DashboardRenderer, datalogger: DataLogger):
        self.dashboard = dashboard
        self.datalogger = datalogger

    def warn(self, message):
        self.dashboard.log_alert(message, level="warning")
        self.datalogger.log_alert(message, level="warning")

    def error(self, message):
        self.dashboard.log_alert(message, level="error")
        self.datalogger.log_alert(message, level="error")

    def info(self, message):
        self.dashboard.log_alert(message, level="info")
        self.datalogger.log_alert(message, level="info")

# ─────────────────────────────────────────────
# 🧠 StatsMonitor (Event-Driven Observer)
# ─────────────────────────────────────────────
class StatsMonitor:
    """
    Modular, event-driven stats monitor for ARIASKA multi-agent RL.
    - ProgressTracker: step/episode progress bars
    - DataLogger: async event logging
    - DashboardRenderer: periodic CLI stats, reward plots, agent tables
    - AlertSystem: warnings, errors, info
    - Orion integration via plugin/callback
    """
    def __init__(self, agents_list=None, agents=None, max_history=100, verbosity="standard"):
        # Handle agent initialization more robustly
        if agents_list and isinstance(agents_list, list):
            # Use agent IDs from agent objects if available
            self.agents = [a.agent_id if hasattr(a, 'agent_id') else str(a) for a in agents_list]
        elif agents and isinstance(agents, list):
            # Use provided agent ID strings
            self.agents = list(agents)
        else:
            # Default to standard agent types
            self.agents = ["RedAgent", "BlueAgent", "ScoutAgent", "ShadowAgent", "OrionAgent"]
            
        self.verbosity = verbosity
        self.progress = ProgressTracker()
        self.datalogger = DataLogger()
        self.dashboard = DashboardRenderer(self.agents, max_history=max_history, verbosity=verbosity)
        self.alerts = AlertSystem(self.dashboard, self.datalogger)
        self.orion_callback = None  # Set externally for Orion integration
        
        # Initialize agent_stats dictionary to track metrics for each agent
        self.agent_stats = {agent: {"rewards": [], "actions": [], "steps": 0, "gpt_calls": 0, "redundancy": 0} 
                            for agent in self.agents}
        
        # Initialize other metrics trackers
        self.global_stats = {"total_steps": 0, "total_episodes": 0, "convergence_score": 0.0}
        
        console.print(f"[cyan]✓ StatsMonitor initialized with {len(self.agents)} agents: {', '.join(self.agents)}[/cyan]")

    def log_step(self, agent_id, reward, **info):
        """
        Log a step for an agent with robust handling for unknown agents
        """
        # Ensure agent exists in tracking dictionaries
        if agent_id not in self.agent_stats:
            self.add_agent(agent_id)
            
        # Update step counts
        self.agent_stats[agent_id]["steps"] += 1
        self.agent_stats[agent_id]["rewards"].append(reward)
        self.global_stats["total_steps"] += 1
        
        # Track command if provided
        if "command" in info:
            self.agent_stats[agent_id]["actions"].append(info["command"])
        
        # Handle GPT token tracking
        gpt_calls = info.get("gpt_calls", 0)
        if gpt_calls > 0:
            self.agent_stats[agent_id]["gpt_calls"] += gpt_calls
            
        # Handle redundancy tracking  
        redundancy = info.get("redundancy", 0)
        if redundancy > 0:
            self.agent_stats[agent_id]["redundancy"] += redundancy
        
        # Update dashboard and logs
        self.datalogger.log_step(agent_id, reward, **info)
        self.dashboard.update_stats(agent_id, reward, 
                                   self.agent_stats[agent_id]["gpt_calls"], 
                                   self.agent_stats[agent_id]["redundancy"])
        
        if self.verbosity != "quiet":
            self.dashboard.update()

    def add_agent(self, agent_id):
        """
        Add a new agent to tracking if it doesn't exist
        """
        if agent_id not in self.agent_stats:
            self.agent_stats[agent_id] = {
                "rewards": [], 
                "actions": [], 
                "steps": 0, 
                "gpt_calls": 0, 
                "redundancy": 0
            }
            self.agents.append(agent_id)
            console.print(f"[yellow]⚠️ Added new agent to tracking: {agent_id}[/yellow]")
            
            # Make sure dashboard knows about this agent too
            if hasattr(self.dashboard, "agent_stats") and agent_id not in self.dashboard.agent_stats:
                self.dashboard.agent_stats[agent_id] = {
                    "rewards": deque(maxlen=self.dashboard.max_history),
                    "gpt_calls": 0,
                    "steps": 0,
                    "redundancy": 0
                }

    def get_performance_metrics(self):
        """
        Get comprehensive performance metrics for strategy optimization
        
        Returns:
            dict: Performance metrics for all agents
        """
        metrics = {
            "global": {
                "total_steps": self.global_stats["total_steps"],
                "total_episodes": self.global_stats["total_episodes"],
                "convergence_score": self.global_stats["convergence_score"]
            }
        }
        
        for agent_id in self.agent_stats:
            agent_data = self.agent_stats[agent_id]
            metrics[agent_id] = {
                "steps": agent_data["steps"],
                "avg_reward": sum(agent_data["rewards"]) / max(1, len(agent_data["rewards"])),
                "total_reward": sum(agent_data["rewards"]),
                "gpt_calls": agent_data["gpt_calls"],
                "redundancy": agent_data["redundancy"],
                "redundancy_rate": self.get_redundancy_rate(agent_id),
                "unique_actions": len(set(agent_data["actions"]))
            }
            
        return metrics

    def log_episode(self, agent_id, total_reward, **info):
        # Ensure agent exists
        if agent_id not in self.agent_stats:
            self.add_agent(agent_id)
            
        self.global_stats["total_episodes"] += 1
        self.datalogger.log_episode(agent_id, total_reward, **info)
        if self.verbosity != "quiet":
            self.dashboard.update(force=True)

    def report_gpt_usage(self, agent_id, tokens):
        # Ensure agent exists
        if agent_id not in self.agent_stats:
            self.add_agent(agent_id)
            
        self.agent_stats[agent_id]["gpt_calls"] += tokens
        self.datalogger.log_gpt_usage(agent_id, tokens)

    def display_episode_summary(self):
        if self.verbosity != "quiet":
            self.dashboard.update(force=True)
            
            # Print summary table to console
            table = Table(title="Episode Summary", show_lines=True)
            table.add_column("Agent", style="cyan")
            table.add_column("Steps", style="blue")
            table.add_column("Avg Reward", style="green")
            table.add_column("Total Reward", style="yellow")
            table.add_column("GPT Calls", style="magenta")
            
            for agent_id in self.agents:
                if agent_id in self.agent_stats:
                    stats = self.agent_stats[agent_id]
                    avg_reward = sum(stats["rewards"]) / max(1, len(stats["rewards"]))
                    total_reward = sum(stats["rewards"])
                    table.add_row(
                        agent_id, 
                        str(stats["steps"]), 
                        f"{avg_reward:.2f}", 
                        f"{total_reward:.2f}",
                        str(stats["gpt_calls"])
                    )
                    
            console.print(table)

    def set_orion_insight(self, insight):
        self.dashboard.set_orion_insight(insight)

    def warn(self, message):
        self.alerts.warn(message)

    def error(self, message):
        self.alerts.error(message)

    def info(self, message):
        self.alerts.info(message)

    def reset(self):
        """Reset all stats for a new episode"""
        # Don't destroy agent entries, just clear their stats
        for agent_id in self.agent_stats:
            self.agent_stats[agent_id]["rewards"] = []
            self.agent_stats[agent_id]["actions"] = []
            self.agent_stats[agent_id]["steps"] = 0
            self.agent_stats[agent_id]["redundancy"] = 0
            # Don't reset gpt_calls as we want to track total usage across episodes
        
        # Reset dashboard display
        self.dashboard.reset()
        
    def flush_logs(self):
        """Ensure all logs are written to disk"""
        pass  # The async DataLogger handles this automatically

    def start_live(self):
        self.dashboard.start_live()

    def stop_live(self):
        self.dashboard.stop_live()

    def get_average_reward(self, agent_id=None):
        """
        Calculate average reward for an agent or globally.
        Args:
            agent_id (str, optional): The specific agent to get average for.
        Returns:
            float: Average reward or 0.0 if no data.
        """
        if agent_id and agent_id in self.agent_stats:
            rewards = self.agent_stats[agent_id].get("rewards", [])
            return sum(rewards) / max(len(rewards), 1) if rewards else 0.0
        
        # If no agent_id specified or not found, return global average
        all_rewards = []
        for agent in self.agent_stats:
            all_rewards.extend(self.agent_stats[agent].get("rewards", []))
        return sum(all_rewards) / max(len(all_rewards), 1) if all_rewards else 0.0
    
    def get_detection_rate(self, agent_id=None):
        """
        Calculate detection rate (percentage of actions detected).
        Args:
            agent_id (str, optional): The specific agent to get rate for.
        Returns:
            float: Detection rate as ratio (0.0 to 1.0)
        """
        # This is a stub - implement based on your detection tracking logic
        return 0.0  # Default implementation returns 0%
    
    def get_redundancy_rate(self, agent_id=None):
        """
        Calculate redundancy rate (percentage of redundant actions).
        Args:
            agent_id (str, optional): The specific agent to get rate for.
        Returns:
            float: Redundancy rate as ratio (0.0 to 1.0)
        """
        if agent_id and agent_id in self.agent_stats:
            redundancy = self.agent_stats[agent_id].get("redundancy", 0)
            steps = self.agent_stats[agent_id].get("steps", 0)
            return redundancy / max(steps, 1) if steps else 0.0
        return 0.0  # Default implementation returns 0%

# ─────────────────────────────────────────────
# 🚀 Diagnostic Mode
# ─────────────────────────────────────────────
if __name__ == "__main__":
    monitor = StatsMonitor()
    monitor.start_live()
    for i in range(10):
        monitor.log_step("RedAgent", reward=i, gpt_calls=i*10)
        time.sleep(0.5)
    monitor.set_orion_insight("Orion: All systems nominal.")
    monitor.warn("Test warning")
    monitor.error("Test error")
    monitor.info("Test info")
    monitor.display_episode_summary()
    monitor.stop_live()