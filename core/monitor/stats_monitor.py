# core/models/stats_monitor.py — ARIASKA StatsMonitor v12.0 APEX PRIME
# 🎯 Live Multi-Agent Dashboard | 🧠 Orion GPT Insights | ⚡ Event-Driven Alerts | 📈 Advanced Health Metrics

import os
import time
import json
from collections import Counter, defaultdict
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    SpinnerColumn,
)
from rich.panel import Panel
from rich.columns import Columns
from rich import box

console = Console()


class StatsMonitor:
    _active_live_dashboard = None  # Singleton for live dashboard

    def __init__(self, agents_list=None, log_dir="logs", verbosity="standard"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.agents = agents_list or [
            "RedAgent",
            "BlueAgent",
            "ScoutAgent",
            "ShadowAgent",
            "OrionAgent",
        ]
        self.agent_stats = defaultdict(lambda: {
            "rewards": [],
            "gpt_calls": 0,
            "redundancy": 0,
            "phases": [],
            "steps": 0,
            "risk": [],
            "alert": [],
            "tokens": 0,
        })

        self.total_steps = 0
        self.total_episodes = 0
        self.start_time = time.time()
        self.prev_snapshot = {
            agent: {"reward": 0, "gpt_calls": 0, "alerts": 0} for agent in self.agents
        }

        self.logs = {
            "session_replay": os.path.join(log_dir, "session_replay.jsonl"),
            "warnings": os.path.join(log_dir, "warnings.log"),
        }
        for path in self.logs.values():
            open(path, "w").close()

        self.live_dashboard = None
        self.progress = None
        self.episode_task = None
        self.verbosity = verbosity
        console.print(
            "[green]✔ StatsMonitor v12.0 APEX PRIME Initialized — Tactical Dashboard Ready[/green]"
        )

    # ─────────────────────────────────────────────
    # 🎮 Episode Progress Control
    # ─────────────────────────────────────────────
    def start_episode_progress(self, total_steps):
        """Initialize progress tracking safely"""
        # Only create a new progress if we don't have one
        if self.progress is None:
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=30),
                TextColumn("[cyan]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console,
                transient=True,
            )
        
        if self.episode_task is not None:
            # Clean up any existing task
            try:
                self.progress.remove_task(self.episode_task)
            except:
                pass
        
        self.episode_task = self.progress.add_task(f"Episode {self.total_episodes + 1}", total=total_steps)
        
        # Only start if not already started
        try:
            self.progress.start()
        except:
            pass
            
        # Initialize live dashboard only if needed
        if self.live_dashboard is None and StatsMonitor._active_live_dashboard is None:
            self.live_dashboard = Live(console=console, refresh_per_second=2)
            self.live_dashboard.start()
            StatsMonitor._active_live_dashboard = self.live_dashboard

    def update_progress(self):
        if self.episode_task is not None:
            self.progress.update(self.episode_task, advance=1)
            self.render_live_dashboard()

    def stop_progress(self):
        """Safely stop progress tracking"""
        if self.progress is not None and self.episode_task is not None:
            try:
                self.progress.stop()
                self.progress.remove_task(self.episode_task)
            except:
                pass
        
        # Reset for next time
        self.episode_task = None
        
        if self.live_dashboard is not None:
            try:
                self.live_dashboard.stop()
            except:
                pass
            if StatsMonitor._active_live_dashboard is self.live_dashboard:
                StatsMonitor._active_live_dashboard = None
            self.live_dashboard = None

    # ─────────────────────────────────────────────
    # 🧠 GPT Call Logging + Orion Insight Triggers
    # ─────────────────────────────────────────────
    def log_gpt_call(self, agent_name, tokens=0):
        self.agent_stats[agent_name]["gpt_calls"] += 1
        self.agent_stats[agent_name]["tokens"] += tokens
        self._log_replay(
            {"event": "gpt_call", "agent": agent_name, "step": self.total_steps}
        )

        if self.agent_stats[agent_name]["gpt_calls"] % 15 == 0:
            self._trigger_warning(f"{agent_name} exceeded 15 GPT calls.")

    # ─────────────────────────────────────────────
    # 🚨 Alert & Redundancy Logging
    # ─────────────────────────────────────────────
    def log_alert(self, agent_name, alert_level=1):
        self.agent_stats[agent_name]["alerts"] += alert_level
        self._log_replay(
            {
                "event": "alert",
                "agent": agent_name,
                "level": alert_level,
                "step": self.total_steps,
            }
        )

        if self.agent_stats[agent_name]["alerts"] % 10 == 0:
            self._trigger_warning(
                f"{agent_name} accumulated {self.agent_stats[agent_name]['alerts']} alerts."
            )

    def log_step(self, agent_id, reward, command=None, phase=None, redundancy=False, risk=0.0, alert=0.0, tokens=0):
        stats = self.agent_stats[agent_id]
        stats["rewards"].append(reward)
        stats["steps"] += 1
        if phase:
            stats["phases"].append(phase)
        if redundancy:
            stats["redundancy"] += 1
        if risk is not None:
            stats["risk"].append(risk)
        if alert is not None:
            stats["alert"].append(alert)
        if tokens:
            stats["tokens"] += tokens
        self.total_steps += 1
        self.update_progress()

    # ─────────────────────────────────────────────
    # ─────────────────────────────────────────────
    # 🎥 Session Replay Logger─────────────────────
    # ─────────────────────────────────────────────
    def _log_replay(self, event_data):
        with open(self.logs["session_replay"], "a") as f:
            f.write(json.dumps(event_data) + "\n")

    def _trigger_warning(self, message):
        timestamp = time.strftime("%H:%M:%S")
        warning_msg = f"[{timestamp}] ⚠ {message}"
        with open(self.logs["warnings"], "a") as f:
            f.write(warning_msg + "\n")
        console.print(f"[yellow]{warning_msg}[/yellow]")

    # ─────────────────────────────────────────────
    # ─────────────────────────────────────────────
    # 🏁 Episode Summary + Orion AI Strategic Note─
    # ─────────────────────────────────────────────
    def display_episode_summary(self):
        self.total_episodes += 1
        total_rewards = {
            agent: sum(self.agent_stats[agent]["rewards"]) for agent in self.agents
        }
        # Concise dashboard
        table = Table(title=f"Episode {self.total_episodes} Summary", show_lines=True)
        table.add_column("Agent", style="cyan")
        table.add_column("Reward", style="green")
        table.add_column("Alerts", style="magenta")
        table.add_column("Novelty", style="blue")
        table.add_column("Redundancy", style="red")
        for agent in self.agents:
            rewards = total_rewards[agent]
            alerts = self.agent_stats[agent]["alerts"]
            novelty = self.agent_stats[agent].get("novelty", 0)
            redundancy = max(self.agent_stats[agent]["redundancy"].values()) if self.agent_stats[agent]["redundancy"] else 0
            table.add_row(agent, f"{rewards:+.1f}", str(alerts) if alerts else "N/A", str(novelty), str(redundancy))
        gpt_calls = sum(self.agent_stats[a]["gpt_calls"] for a in self.agents)
        snapshots = getattr(self, "snapshots_this_episode", 0)
        fallbacks = getattr(self, "fallbacks_this_episode", 0)
        console.print(table)
        console.print(f"GPT Calls: {gpt_calls} | Snapshots: {snapshots} | Fallbacks: {fallbacks}")
        insight = self._generate_orion_insight(total_rewards)
        console.print(f"[bold blue]🧠 Orion Insight:[/bold blue] {insight}")
        self._save_log(
            f"Episode {self.total_episodes} | Insight: {insight}", "episodes"
        )
        self.stop_progress()
        # Final session health overview
        console.rule("[bold green]📊 Final Session Health Overview")
        for agent in self.agents:
            rewards = self.agent_stats[agent]["rewards"]
            avg_reward = sum(rewards) / len(rewards) if rewards else 0
            bar = "#" * int(avg_reward / 4)
            console.print(f"{agent}: [{bar}] ({avg_reward:.2f})")
        uptime = time.time() - self.start_time
        console.print(
            f"[cyan]⏱ Uptime:[/cyan] {uptime:.1f}s | [yellow]Steps:[/yellow] {self.total_steps} | [magenta]Episodes:[/magenta] {self.total_episodes}"
        )
        self._save_log(
            f"Session completed in {uptime:.1f}s with {self.total_steps} steps.",
            "session",
        )

    def _generate_orion_insight(self, rewards):
        avg_reward = sum(rewards.values()) / len(rewards) if rewards else 0
        if avg_reward > 120:
            return "🚀 Offensive dominance detected. Recommend stealth optimization."
        elif avg_reward < 60:
            return "⚠ Tactical inefficiency noted. Diversify attack vectors."
        return "✅ Balanced performance. Maintain adaptive strategies."
    # ─────────────────────────────────────────────
    # ─────────────────────────────────────────────
    # ─────────────────────────────────────────────
    # ─────────────────────────────────────────────
    # 🛡️ Reset System for Next Simulation─────────
    # ─────────────────────────────────────────────
    def reset(self):
        """Complete reset of the stats monitor"""
        # First stop any active displays
        self.stop_progress()
        # Then reset the data
        self.agent_stats = defaultdict(lambda: {
            "rewards": [],
            "gpt_calls": 0,
            "redundancy": 0,
            "phases": [],
            "steps": 0,
            "risk": [],
            "alert": [],
            "tokens": 0,
        })
        self.total_steps = 0
        self.total_episodes = 0
        self.start_time = time.time()
        console.print("[yellow]🔄 StatsMonitor reset — Ready for next operation.[/yellow]")
    # 📂 Unified Logger────────────────────────────
    # ─────────────────────────────────────────────
    def _save_log(self, content, log_type):
        with open(self.logs.get(log_type, os.path.join(self.log_dir, f"{log_type}.log")), "a") as f:
            f.write(content + "\n")

    def print_agent_actions_and_gpt(self):
        table = Table(title="Agent Actions & GPT Calls", show_lines=True)
        table.add_column("Agent", style="cyan")
        table.add_column("Last Action", style="magenta")
        table.add_column("GPT Calls", style="yellow")
        for agent in self.agents:
            last_cmd = self.agent_stats[agent]["last_command"]
            gpt_calls = self.agent_stats[agent]["gpt_calls"]
            table.add_row(agent, str(last_cmd), str(gpt_calls))
        console.print(table)

    def render_live_dashboard(self):
        """Generate rich dashboard for live training display"""
        if self.live_dashboard is None:
            return
        # Create a table to show live agent performance
        table = Table(title="🚀 Live Agent Performance", show_lines=True)
        table.add_column("Agent", style="cyan")
        table.add_column("Last Cmd", style="yellow", overflow="fold")
        table.add_column("Last Reward", style="green")
        table.add_column("Avg Reward", style="magenta")
        table.add_column("GPT Calls", style="blue")
        table.add_column("Redundancy %", style="red")
        for agent in self.agents:
            stats = self.agent_stats[agent]
            last_cmd = str(stats.get("last_command", "N/A"))
            if len(last_cmd) > 30:
                last_cmd = last_cmd[:27] + "..."
            rewards = stats.get("rewards", [])
            avg_reward = sum(rewards) / max(len(rewards), 1) if rewards else 0
            last_reward = rewards[-1] if rewards else 0
            total_cmds = self.total_steps / len(self.agents) if len(self.agents) > 0 else 1
            redundancy_count = sum(stats.get("redundancy", {}).values())
            redundancy_pct = (redundancy_count / max(total_cmds, 1)) * 100
            # Color indicators for risk
            risk_val = 0
            if hasattr(self, "env") and hasattr(self.env, "detection_risk"):
                risk_val = self.env.detection_risk
            risk_color = "red" if risk_val > 5.0 else "yellow" if risk_val > 2.5 else "green"
            table.add_row(
                agent,
                last_cmd,
                f"{last_reward:.2f}",
                f"{avg_reward:.2f}",
                str(stats.get("gpt_calls", 0)),
                f"[{risk_color}]{redundancy_pct:.1f}%[/{risk_color}]"
            )
        # Create a progress panel with the dashboard
        from rich.panel import Panel
        from rich.layout import Layout
        layout = Layout()
        layout.split_column(
            Layout(table, name="table"),
            Layout(self._create_environment_panel(), name="environment")
        )
        self.live_dashboard.update(layout)

    def _create_environment_panel(self):
        """Create a panel showing environment state"""
        from rich.panel import Panel
        # Get environment state from the first agent's environment if available
        env_state = {}
        for agent_name in self.agents:
            # This is a simple approximate approach - in a real system you'd have a more direct way
            # to access the environment state from the StatsMonitor
            replay_data = self._load_latest_replay()
            if replay_data:
                env_state = replay_data.get("state", {})
                break
        if not env_state:
            return Panel("No environment data available", title="🌍 Environment")
        # Format state data
        state_lines = [
            f"[cyan]Phase:[/cyan] {env_state.get('phase', 'N/A')}",
            f"[red]Alert:[/red] {env_state.get('blue_team_alert', 0):.1f}",
            f"[yellow]Risk:[/yellow] {env_state.get('detection_risk', 0):.1f}",
            f"[green]Privilege:[/green] {env_state.get('privilege_level', 'none')}",
        ]
        if "open_ports" in env_state:
            ports = env_state['open_ports']
            port_display = ", ".join(str(p) for p in ports[:5])
            if len(ports) > 5:
                port_display += f" +{len(ports)-5} more"
            state_lines.append(f"[blue]Ports:[/blue] {port_display}")
        services = env_state.get('services', [])
        if services:
            service_display = ", ".join(services[:5])
            if len(services) > 5:
                service_display += f" +{len(services)-5} more"
            state_lines.append(f"[magenta]Services:[/magenta] {service_display}")
        return Panel("\n".join(state_lines), title="🌍 Environment State")

    def _load_latest_replay(self):
        """Load the most recent replay event"""
        try:
            with open(self.logs["session_replay"], "r") as f:
                lines = f.readlines()
                if lines:
                    return json.loads(lines[-1])
        except Exception:
            pass
        return {}

    def get_average_reward(self, agent_id=None):
        """Get the average reward for a specific agent or all agents"""
        if agent_id and agent_id in self.agent_stats:
            rewards = self.agent_stats[agent_id].get("rewards", [])
            return sum(rewards) / max(len(rewards), 1) if rewards else 0.0
        # If no agent specified, return average across all agents
        all_rewards = []
        for agent in self.agents:
            all_rewards.extend(self.agent_stats.get(agent, {}).get("rewards", []))
        return sum(all_rewards) / max(len(all_rewards), 1) if all_rewards else 0.0

    def get_detection_rate(self):
        # Example: percent of steps with alert > 5
        total, detected = 0, 0
        for stats in self.agent_stats.values():
            alerts = stats.get("alert", [])
            total += len(alerts)
            detected += sum(1 for a in alerts if a > 5)
        return detected / total if total else 0.0

    def get_redundancy_rate(self, agent_id=None):
        """Calculate redundancy rate for an agent"""
        if agent_id and agent_id in self.agent_stats:
            redundancy = sum(self.agent_stats[agent_id].get("redundancy", {}).values())
            total = max(len(self.agent_stats[agent_id].get("rewards", [])), 1)
            return redundancy / total
        # If no agent specified, return average across all agents
        total_redundancy = 0
        total_commands = 0
        for agent in self.agents:
            total_redundancy += sum(self.agent_stats.get(agent, {}).get("redundancy", {}).values())
            total_commands += len(self.agent_stats.get(agent, {}).get("rewards", []))
        return total_redundancy / max(total_commands, 1)

    def get_phase_counts(self, agent_id):
        stats = self.agent_stats[agent_id]
        return dict(Counter(stats["phases"]))

    def visualize_phase_distribution(self, agent_id=None):
        try:
            """Print a live phase distribution chart for the agent or all agents."""
            if agent_id and agent_id in self.agent_stats:
                rewards = self.agent_stats[agent_id].get("rewards", [])
                phases = [a.get("phase", "unknown") for a in getattr(self, "agent_history", {}).get(agent_id, {}).get("phases", [])]
                counts = Counter(phases)
                table = Table(title=f"{agent_id} Phase Distribution", show_lines=True)
                table.add_column("Phase", style="cyan")
                table.add_column("Count", style="magenta")
                for phase, count in counts.items():
                    table.add_row(phase, str(count))
                console.print(Panel(table, title="Phase Distribution", border_style="magenta"))
            else:
                # All agents
                for agent in self.agents:
                    self.visualize_phase_distribution(agent)
            # Downsample to last 100 points
            if len(data) > 100:
                data = data[-100:]
        except Exception as e:
            console.print(f"[yellow]⚠ StatsMonitor visualization error: {e}[/yellow]")

    def show(self, live_plot=True):
        if not live_plot:
            return
        try:
            # Show a dashboard of all tracked stats
            tables = []
            for agent_id, stats in self.agent_stats.items():
                table = Table(title=f"{agent_id} Stats", show_lines=True, box=box.ROUNDED)
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="magenta")
                table.add_row("Steps", str(stats["steps"]))
                table.add_row("Avg Reward", f"{self.get_average_reward(agent_id):.2f}")
                table.add_row("Redundancy Rate", f"{self.get_redundancy_rate(agent_id):.2%}")
                table.add_row("GPT Calls", str(stats["gpt_calls"]))
                table.add_row("Tokens", str(stats["tokens"]))
                table.add_row("Detection Rate", f"{self.get_detection_rate():.2%}")
                table.add_row("Phases", ", ".join(stats["phases"][-5:]))
                table.add_row("Recent Rewards", ", ".join(f"{r:.1f}" for r in stats["rewards"][-5:]))
                table.add_row("Recent Risk", ", ".join(f"{r:.2f}" for r in stats.get("risk", [])[-5:]))
                table.add_row("Recent Alert", ", ".join(f"{a:.2f}" for a in stats.get("alert", [])[-5:]))
                tables.append(Panel(table, border_style="green"))
            console.print(Columns(tables))
        except Exception as e:
            console.print(f"[yellow]⚠ StatsMonitor show error: {e}[/yellow]")