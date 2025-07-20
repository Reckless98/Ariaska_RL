# core/visualization/training_visualizer.py — ARIASKA Training Visualizer v11.0
# 📊 Real-time Training Visualization | 🧠 Agent Performance Metrics | 🎮 Multi-Agent Dashboard

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, SpinnerColumn
from rich.live import Live
from rich.columns import Columns
from rich import box
import time
import os
import json
from collections import deque, Counter

console = Console()

class DisplayManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @staticmethod
    def get_instance():
        if DisplayManager._instance is None:
            DisplayManager._instance = DisplayManager()
        return DisplayManager._instance

    def __init__(self):
        self.live_display = None

    def start(self, live):
        if self.live_display is not None:
            console.print("[yellow]⚠ Only one live display may be active at once. Skipping new live display.[/yellow]")
            return False
        self.live_display = live
        return True

    def stop(self):
        if self.live_display:
            try:
                self.live_display.stop()
            except Exception as e:
                console.print(f"[yellow]⚠ DisplayManager stop error: {e}[/yellow]")
            self.live_display = None

    def safe_print(self, panel):
        try:
            console.print(panel)
        except Exception:
            self.log_minimal_status()

    def log_minimal_status(self):
        console.print("[yellow]⚠ Display error. Minimal status only.[/yellow]")

class TrainingVisualizer:
    # Use class variable to track active instance
    _active_instance = None
    _active_live_display = None
    
    @classmethod
    def get_instance(cls, agents=None, max_history=100, log_dir="logs"):
        """Singleton factory method - ensures only one TrainingVisualizer exists globally"""
        if cls._active_instance is None:
            cls._active_instance = TrainingVisualizer(agents, max_history, log_dir)
        return cls._active_instance
    
    def __init__(self, agents=None, max_history=100, log_dir="logs"):
        # If an instance already exists, prevent creating a new one
        if TrainingVisualizer._active_instance is not None:
            if agents is not None:
                console.print("[yellow]⚠ TrainingVisualizer already exists. Use get_instance() instead.[/yellow]")
            return
            
        self.agents = agents or ["RedAgent", "BlueAgent", "ScoutAgent", "ShadowAgent", "OrionAgent"]
        self.max_history = max_history
        self.log_dir = log_dir
        
        # Create agent history tracking with deque for efficient history management
        self.agent_history = {
            agent: {
                "rewards": deque(maxlen=max_history),
                "actions": deque(maxlen=max_history),
                "phases": deque(maxlen=max_history)
            } for agent in self.agents
        }
        
        # Environment state tracking
        self.env_states = deque(maxlen=max_history)
        self.live_display = None
        self.is_active = False
        
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        
        self.global_gpt_insight = "Awaiting insight..."
        self.coherence_score = 1.0
        self.token_usage = {agent: 0 for agent in self.agents}
        self.token_usage_per_episode = {agent: [] for agent in self.agents}
        self.epsilon_history = {agent: deque(maxlen=max_history) for agent in self.agents}
        self.alerts = deque(maxlen=5)
        self.llm_usage_stats = {
            agent: {"Seneca": 0, "Lily": 0, "GPT": 0, "tokens": 0, "response_times": []}
            for agent in self.agents
        }
        self.last_update_step = 0
        self.update_interval = 5  # Only update visuals every N steps for performance
        
    def start_live_display(self):
        """Start live visualization display (singleton)"""
        # If there's already a live display active in another instance, stop it first
        if TrainingVisualizer._active_live_display is not None and TrainingVisualizer._active_live_display is not self.live_display:
            try:
                # Try to gracefully stop the other instance's display
                if hasattr(TrainingVisualizer._active_instance, 'stop_live_display'):
                    TrainingVisualizer._active_instance.stop_live_display()
                else:
                    TrainingVisualizer._active_live_display.stop()
            except Exception as e:
                console.print(f"[yellow]⚠ Error stopping previous display: {e}[/yellow]")

        # Start our display
        try:
            if self.live_display is None:
                self.live_display = Live(self.render(), console=console, refresh_per_second=4)
                self.live_display.start()
                self.is_active = True
                TrainingVisualizer._active_live_display = self.live_display
                TrainingVisualizer._active_instance = self
                return True
        except Exception as e:
            console.print(f"[red]❌ Error starting live display: {e}[/red]")
            return False
        return self.is_active

    def stop_live_display(self):
        """Stop live visualization display (singleton)"""
        try:
            if self.live_display:
                self.live_display.stop()
                self.live_display = None
                self.is_active = False
                if TrainingVisualizer._active_live_display is self.live_display:
                    TrainingVisualizer._active_live_display = None
                return True
        except Exception as e:
            console.print(f"[yellow]⚠ Error stopping live display: {e}[/yellow]")
            self.live_display = None
            self.is_active = False
            TrainingVisualizer._active_live_display = None
        return not self.is_active
            
    def update(self, agent_data=None, env_state=None, gpt_insight=None, coherence=None, force=False):
        """
        Update visualization with new data and optionally update global GPT insight/coherence.
        Only refresh visuals every self.update_interval steps unless force=True.
        """
        step = agent_data.get("step", 0) if agent_data else 0
        if not force and step and (step - self.last_update_step) < self.update_interval:
            return
        self.last_update_step = step
        if agent_data:
            self._update_agent_data(agent_data)
            # Store detailed info for advanced panels
            agent_id = agent_data.get("agent_id")
            if agent_id:
                if not hasattr(self, "agent_panels"):
                    self.agent_panels = {}
                self.agent_panels[agent_id] = agent_data
                # Track phase for live chart
                if "phase" in agent_data and agent_id in self.agent_history:
                    self.agent_history[agent_id]["phases"].append(agent_data["phase"])
            # Track token usage per step if available
            if agent_id and "gpt_calls" in agent_data:
                self.token_usage[agent_id] = agent_data["gpt_calls"]
            # Track epsilon for live plot
            if agent_id and "epsilon" in agent_data:
                self.epsilon_history[agent_id].append(agent_data["epsilon"])
            # Track LLM usage stats if available
            if agent_id:
                for llm in ["Seneca", "Lily", "GPT"]:
                    if f"{llm.lower()}_calls" in agent_data:
                        self.llm_usage_stats[agent_id][llm] += agent_data.get(f"{llm.lower()}_calls", 0)
                if "gpt_tokens" in agent_data:
                    self.llm_usage_stats[agent_id]["tokens"] += agent_data["gpt_tokens"]
                if "llm_response_time" in agent_data:
                    self.llm_usage_stats[agent_id]["response_times"].append(agent_data["llm_response_time"])
        if env_state:
            self._update_env_state(env_state)
        if gpt_insight:
            self.global_gpt_insight = gpt_insight
        if coherence is not None:
            self.coherence_score = coherence
        if self.is_active and self.live_display:
            self.live_display.update(self._generate_layout())
            
    def set_global_gpt_insight(self, insight):
        self.global_gpt_insight = insight

    def set_coherence_score(self, score):
        self.coherence_score = score
            
    def _update_agent_data(self, agent_data):
        """Update agent history with new data"""
        agent_id = agent_data.get("agent_id")
        if agent_id in self.agent_history:
            if "reward" in agent_data:
                self.agent_history[agent_id]["rewards"].append(agent_data["reward"])
            if "action" in agent_data:
                self.agent_history[agent_id]["actions"].append(agent_data["action"])
            if "phase" in agent_data:
                self.agent_history[agent_id]["phases"].append(agent_data["phase"])
                
    def _update_env_state(self, env_state):
        """Update environment state history"""
        self.env_states.append(env_state)
        
    def _generate_layout(self):
        layout = Layout()
        layout.split(
            Layout(name="alerts", size=3),
            Layout(name="header", size=6),
            Layout(name="main", ratio=4),
            Layout(name="insights", size=12),
        )
        layout["alerts"].update(self._generate_alert_panel())
        layout["header"].update(self._generate_header())
        layout["main"].split_row(
            Layout(self._generate_agent_panels(), name="agent_panels", ratio=3),
            Layout(self._generate_environment_panel(), name="environment", ratio=2)
        )
        layout["insights"].split_row(
            Layout(self._generate_gpt_insight_panel(), name="gpt"),
            Layout(self._generate_coherence_panel(), name="coherence"),
            Layout(self._generate_token_usage_panel(), name="tokens"),
        )
        return layout

    def _generate_alert_panel(self):
        """Show recent alerts/errors at the top of the dashboard."""
        if not self.alerts:
            return Panel("[green]No alerts[/green]", title="Status", border_style="green")
        panels = []
        for msg, color in list(self.alerts):
            panels.append(Panel(f"[{color}]{msg}[/{color}]", border_style=color))
        return Columns(panels)
            
    def _generate_header(self):
        """Generate header with summary metrics"""
        # Calculate summary statistics
        total_steps = sum(len(self.agent_history[a]["rewards"]) for a in self.agents)
        avg_rewards = {
            a: sum(self.agent_history[a]["rewards"]) / max(len(self.agent_history[a]["rewards"]), 1) 
            for a in self.agents
        }
        
        # Create header table
        table = Table.grid(expand=True)
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        table.add_column("Progress", style="yellow", no_wrap=True)
        table.add_column("Value", style="green")
        
        # Add summary metrics
        table.add_row("Total Steps", str(total_steps), "RedAgent Avg", f"{avg_rewards.get('RedAgent', 0):.2f}")
        table.add_row(
            "Active Agents", str(len(self.agents)), 
            "BlueAgent Avg", f"{avg_rewards.get('BlueAgent', 0):.2f}"
        )
        
        return Panel(table, title="🚀 Training Progress", border_style="bright_blue")
        
    def _generate_agent_panels(self):
        """Show per-agent panels with advanced info, phase charts, and live epsilon/reward plots."""
        panels = []
        for agent in self.agents:
            data = getattr(self, "agent_panels", {}).get(agent, {})
            table = Table(title=f"{agent} State", box=box.ROUNDED)
            table.add_column("Field", style="cyan")
            table.add_column("Value", style="magenta")
            # Phase, last action, status
            phase = data.get("phase", "N/A")
            action = data.get("command", "N/A")
            stuck = False
            if "stuck" in data and data["stuck"]:
                stuck = True
            elif "reward" in data and isinstance(data["reward"], (int, float)):
                # Heuristic: reward not improving for N steps
                rewards = list(self.agent_history[agent]["rewards"])
                if len(rewards) >= 4 and max(rewards[-4:]) - min(rewards[-4:]) < 1e-3:
                    stuck = True
            status = "[red]STUCK[/red]" if stuck else "[green]OK[/green]"
            table.add_row("Phase", str(phase))
            table.add_row("Last Action", str(action))
            table.add_row("Status", status)
            # Epsilon and reward plots
            table.add_row("Epsilon", self._plot_line_ascii(self.epsilon_history[agent], "cyan"))
            table.add_row("Reward", self._plot_line_ascii(self.agent_history[agent]["rewards"], "green"))
            panels.append(Panel(table, border_style="blue"))
        return Columns(panels)
        
    def _generate_phase_chart(self, agent):
        """Live phase distribution bar chart for an agent."""
        phases = list(self.agent_history.get(agent, {}).get("phases", []))
        if not phases:
            return Panel("No phase data", title="Phase Chart", border_style="magenta")
        counts = Counter(phases)
        total = sum(counts.values())
        bar = ""
        for phase in ["recon", "enumeration", "exploit", "privesc", "exfiltrate"]:
            n = counts.get(phase, 0)
            width = int((n / total) * 20) if total else 0
            color = {
                "recon": "blue",
                "enumeration": "cyan",
                "exploit": "yellow",
                "privesc": "magenta",
                "exfiltrate": "red"
            }.get(phase, "white")
            bar += f"[{color}]{'█'*width}[/{color}]"
        label = " ".join(f"{k}:{v}" for k, v in counts.items())
        return Panel(f"{bar}\n{label}", title="Phase Distribution", border_style="magenta")

    def _generate_reward_chart(self):
        """Generate ASCII reward chart for agents"""
        chart_width = 40
        chart_height = 10
        
        # Create empty chart matrix
        chart = [[" " for _ in range(chart_width)] for _ in range(chart_height)]
        
        # Get reward history for plotting agents
        red_rewards = list(self.agent_history.get("RedAgent", {}).get("rewards", []))
        blue_rewards = list(self.agent_history.get("BlueAgent", {}).get("rewards", []))
        
        # Skip if not enough data
        if not red_rewards and not blue_rewards:
            return Panel("Not enough data for reward chart", title="📈 Reward Trends")
            
        # Calculate min/max for scaling
        all_rewards = red_rewards + blue_rewards
        min_reward = min(all_rewards) if all_rewards else 0
        max_reward = max(all_rewards) if all_rewards else 0
        
        # Ensure non-zero range for scaling
        reward_range = max(1.0, max_reward - min_reward)
        
        # Plot red agent rewards
        self._plot_line(chart, red_rewards, min_reward, reward_range, chart_height, "R", "red")
        
        # Plot blue agent rewards
        self._plot_line(chart, blue_rewards, min_reward, reward_range, chart_height, "B", "blue")
        
        # Convert chart matrix to string
        chart_str = "\n".join("".join(row) for row in chart)
        
        return Panel(
            chart_str, 
            title="📈 Reward Trends", 
            subtitle="R=Red, B=Blue",
            border_style="bright_green"
        )
        
    def _plot_line(self, chart, values, min_val, val_range, height, marker, color):
        """Plot a line on the ASCII chart"""
        if not values:
            return
            
        width = len(chart[0])
        # Sample values to fit chart width
        step = max(1, len(values) // width)
        sampled = [values[i] for i in range(0, len(values), step)][:width]
        
        # Plot each point
        for x, val in enumerate(sampled):
            if x < width:
                # Scale value to chart height
                y = height - 1 - int((val - min_val) / val_range * (height - 1))
                y = max(0, min(height-1, y))  # Ensure within bounds
                
                # Place marker
                chart[y][x] = f"[{color}]{marker}[/{color}]"
                
    def _generate_environment_panel(self):
        """Generate panel showing environment state"""
        if not self.env_states:
            return Panel("No environment data available", title="🌍 Environment")
            
        # Get most recent environment state
        state = self.env_states[-1]
        
        # Create table with key metrics
        table = Table(show_header=False)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        # Format state data
        state_items = [
            ("Phase", state.get("phase", "unknown")),
            ("Alert Level", f"{state.get('blue_team_alert', 0):.2f}"),
            ("Detection Risk", f"{state.get('detection_risk', 0):.2f}"),
            ("Privilege", state.get("privilege_level", "none")),
            ("Credentials Found", str(state.get("credentials_found", False))),
            ("Data Exfiltrated", str(state.get("data_exfiltrated", False)))
        ]
        
        for key, value in state_items:
            table.add_row(key, str(value))
            
        # Show open ports if available
        if "open_ports" in state:
            ports = state["open_ports"]
            port_display = ", ".join(str(p) for p in ports[:6])
            if len(ports) > 6:
                port_display += f" +{len(ports)-6}"
            table.add_row("Open Ports", port_display)
            
        # Show services if available
        if "services" in state:
            services = state["services"]
            service_display = ", ". join(services[:6])
            if len(services) > 6:
                service_display += f" +{len(services)-6}"
            table.add_row("Services", service_display)
            
        return Panel(table, title="🌍 Environment State", border_style="green")
        
    def _generate_gpt_insight_panel(self):
        """Panel for global GPT insight (can be updated by Orion/agents)."""
        return Panel(
            f"[bold magenta]{self.global_gpt_insight}[/bold magenta]",
            title="👁️ Global GPT Insight",
            border_style="magenta"
        )

    def _generate_coherence_panel(self):
        """Panel for agent coherence score (dummy for now, can be computed by GPT)."""
        score = self.coherence_score
        bar = "█" * int(score * 20)
        return Panel(
            f"[cyan]Coherence Score:[/cyan] {score:.2f}\n[green]{bar}[/green]",
            title="🤝 Agent Coherence",
            border_style="cyan"
        )

    def _generate_token_usage_panel(self):
        """Show LLM call stats for all agents, including Seneca/Lily/GPT, tokens, and response times."""
        table = Table(title="🧠 LLM Usage", box=box.ROUNDED)
        table.add_column("Agent", style="cyan")
        table.add_column("Seneca", style="yellow")
        table.add_column("Lily", style="magenta")
        table.add_column("GPT", style="red")
        table.add_column("Tokens", style="green")
        table.add_column("Avg Resp (s)", style="blue")
        for agent in self.agents:
            stats = self.llm_usage_stats.get(agent, {})
            avg_resp = (
                sum(stats.get("response_times", [])) / len(stats.get("response_times", []))
                if stats.get("response_times") else 0.0
            )
            table.add_row(
                agent,
                str(stats.get("Seneca", 0)),
                str(stats.get("Lily", 0)),
                str(stats.get("GPT", 0)),
                str(stats.get("tokens", 0)),
                f"{avg_resp:.2f}"
            )
        return Panel(table, title="🧠 LLM Usage", border_style="yellow")

    def _plot_line_ascii(self, values, color="green", width=24, height=1):
        """Render a simple ASCII bar for a list of values."""
        if not values:
            return ""
        min_v, max_v = min(values), max(values)
        rng = max(1e-6, max_v - min_v)
        scaled = [int((v - min_v) / rng * (width - 1)) if rng > 0 else 0 for v in values[-width:]]
        bar = [" "] * width
        for idx in scaled:
            if 0 <= idx < width:
                bar[idx] = "█"
        return f"[{color}]{''.join(bar)}[/{color}]"

    def save_visualization_snapshot(self, filename=None):
        """Save current visualization to a file"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.log_dir, f"training_viz_{timestamp}.txt")
            
        # Generate layout
        layout = self._generate_layout()
        
        # Create a string representation
        console_output = Console(width=100, file=open(filename, "w"))
        console_output.print(layout)
        console_output.file.close()
        
        console.print(f"[green]✓ Visualization snapshot saved to {filename}[/green]")
        
    def create_training_report(self, episode, filename=None):
        """Create a detailed training report for an episode"""
        if filename is None:
            filename = os.path.join(self.log_dir, f"training_report_ep{episode}.json")
            
        # Compile report data
        report = {
            "episode": episode,
            "timestamp": time.time(),
            "agents": {
                agent: {
                    "rewards": list(self.agent_history[agent]["rewards"]),
                    "avg_reward": sum(self.agent_history[agent]["rewards"]) / max(len(self.agent_history[agent]["rewards"]), 1),
                    "phases": list(self.agent_history[agent]["phases"])
                } for agent in self.agent_history
            },
            "environment": {
                "states": list(self.env_states)
            }
        }
        
        # Save to file
        with open(filename, "w") as f:
            json.dump(report, f, indent=2)
            
        console.print(f"[green]✓ Training report saved to {filename}[/green]")
        
        # Create text summary as well
        summary_file = os.path.join(self.log_dir, f"training_summary_ep{episode}.txt")
        console_output = Console(width=100, file=open(summary_file, "w"))
        
        # Print episode summary
        console_output.rule(f"Episode {episode} Summary")
        
        for agent, data in report["agents"].items():
            console_output.print(f"[bold]{agent}[/bold]")
            console_output.print(f"  Average Reward: {data['avg_reward']:.2f}")
            console_output.print(f"  Steps: {len(data['rewards'])}")
            
        console_output.file.close()
        
    def push_alert(self, message, level="error"):
        """Push an alert/warning to be displayed at the top of the dashboard."""
        color = "red" if level == "error" else "yellow"
        self.alerts.appendleft((message, color))

    def render(self):
        """
        Generate a rich layout for the live visualization display.
        This serves as the main rendering function for the Live display.
        
        Returns:
            Layout: A rich Layout object with nested panels and tables.
        """
        layout = Layout()
        layout.split(
            Layout(name="alerts", size=3),
            Layout(name="header", size=6),
            Layout(name="main", ratio=4),
            Layout(name="insights", size=12),
        )
        layout["alerts"].update(self._generate_alert_panel())
        layout["header"].update(self._generate_header())
        layout["main"].split_row(
            Layout(self._generate_agent_panels(), name="agent_panels", ratio=3),
            Layout(self._generate_environment_panel(), name="environment", ratio=2)
        )
        layout["insights"].split_row(
            Layout(self._generate_gpt_insight_panel(), name="gpt"),
            Layout(self._generate_coherence_panel(), name="coherence"),
            Layout(self._generate_token_usage_panel(), name="tokens"),
        )
        return layout
