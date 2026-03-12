# core/ui/enhanced_agent_dashboard.py - ARIASKA Professional Agent Dashboard v3.0
# 🎯 Clean Professional UI | 📊 Detailed Agent Monitoring | 🚀 GPU-Accelerated Training

import time
import json
from typing import Dict, List, Any, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich.align import Align
from collections import defaultdict, deque

console = Console()

class ProfessionalAgentDashboard:
    """Professional, clean dashboard for multi-agent training monitoring."""
    
    def __init__(self, update_interval: float = 0.5):
        self.update_interval = update_interval
        self.agent_data: Dict[str, Dict[str, Any]] = {}
        self.system_metrics = {
            "total_episodes": 0,
            "current_episode": 0,
            "total_steps": 0,
            "gpu_utilization": 0.0,
            "memory_usage": 0.0,
            "learning_rate": 0.001,
            "training_active": False
        }
        self.live_dashboard = None
    
    def _init_agent_data(self, agent_id: str):
        """Initialize clean agent data structure."""
        if agent_id not in self.agent_data:
            self.agent_data[agent_id] = {
                "command": "Initializing...",
                "target": "N/A",
                "output": "No output yet",
                "next_command": "Analyzing...",
                "avg_reward": 0.0,
                "total_reward": 0.0,
                "llm_calls": 0,
                "llm_dependency": 0.0,  # Percentage
                "memory_updated": False,
                "learning_active": False,
                "agent_type": "Rule-based",
                "neural_loss": 0.0,
                "confidence": 0.0,
                "step_count": 0,
                "last_update": time.time(),
                "status": "IDLE"
            }
    
    def update_agent_status(self, agent_id: str, data: Dict[str, Any]):
        """Update agent with clean, essential data only."""
        self._init_agent_data(agent_id)
        
        # Calculate metrics
        current_reward = float(data.get("reward", 0.0))
        self.agent_data[agent_id]["total_reward"] += current_reward
        
        step_count = max(1, int(data.get("step_count", 1)))
        avg_reward = self.agent_data[agent_id]["total_reward"] / step_count
        
        # Calculate LLM dependency
        llm_calls = int(data.get("llm_calls", 0))
        total_actions = int(data.get("actions_taken", 0))
        llm_dependency = (llm_calls / total_actions) * 100 if total_actions > 0 else 0.0   
        # Update essential metrics only
        self.agent_data[agent_id].update({
            "command": str(data.get("action", "No action"))[:50],
            "target": str(data.get("target", "N/A"))[:30],
            "output": str(data.get("output", "No output"))[:40],
            "next_command": str(data.get("next_step", "Analyzing..."))[:35],
            "avg_reward": avg_reward,
            "llm_calls": llm_calls,
            "llm_dependency": llm_dependency,
            "memory_updated": bool(data.get("memory_updated", False)),
            "learning_active": bool(data.get("learning_active", False)),
            "agent_type": str(data.get("agent_type", "Rule-based")),
            "neural_loss": float(data.get("neural_loss", 0.0)),
            "confidence": float(data.get("confidence", 0.0)),
            "step_count": step_count,
            "status": "ACTIVE",
            "last_update": time.time()
        })
    
    def update_system_metrics(self, metrics: Dict[str, Any]):
        """Update system-wide metrics."""
        self.system_metrics.update(metrics)
    
    def create_red_agent_table(self) -> Table:
        """Create detailed table for RedAgent - the primary focus."""
        table = Table(title="🔴 RedAgent - Primary Training Focus", 
                     border_style="red", box=None, show_header=True)
        
        table.add_column("Metric", style="bold white", width=15)
        table.add_column("Value", style="white", width=50)
        table.add_column("Status", style="green", width=10)
        
        red_data = self.agent_data.get("RedAgent", {})
        
        # Learning indicator
        learning_icon = "🟢" if red_data.get("learning_active") else "🔴"
        memory_icon = "✅" if red_data.get("memory_updated") else "❌"
        
        table.add_row("Command", red_data.get("command", "N/A"), "")
        table.add_row("Target", red_data.get("target", "N/A"), "")
        table.add_row("Output", red_data.get("output", "N/A"), "")
        table.add_row("Next Command", red_data.get("next_command", "N/A"), "")
        table.add_row("Avg Reward", f"{red_data.get('avg_reward', 0):.2f}", "")
        table.add_row("Total Reward", f"{red_data.get('total_reward', 0):.1f}", "")
        table.add_row("LLM Calls", str(red_data.get("llm_calls", 0)), "")
        table.add_row("LLM Dependency", f"{red_data.get('llm_dependency', 0):.1f}%", "")
        table.add_row("Memory Updated", "Yes" if red_data.get("memory_updated") else "No", memory_icon)
        table.add_row("Learning Active", "Yes" if red_data.get("learning_active") else "No", learning_icon)
        table.add_row("Neural Loss", f"{red_data.get('neural_loss', 0):.4f}", "")
        table.add_row("Confidence", f"{red_data.get('confidence', 0):.2f}", "")
        
        return table
    
    def create_agent_table(self, agent_id: str) -> Table:
        """Create compact table for other agents."""
        agent_data = self.agent_data.get(agent_id, {})
        
        # Color coding by agent
        colors = {
            "BlueAgent": "blue",
            "OrionAgent": "magenta", 
            "ScoutAgent": "cyan",
            "ShadowAgent": "yellow"
        }
        color = colors.get(agent_id, "white")
        
        table = Table(title=f"{agent_id}", border_style=color, box=None, show_header=False)
        table.add_column("", style="bold", width=12)
        table.add_column("", style="white", width=25)
        
        table.add_row("Action", agent_data.get("command", "N/A")[:25])
        table.add_row("Reward", f"{agent_data.get('avg_reward', 0):.1f}")
        table.add_row("LLM Calls", str(agent_data.get("llm_calls", 0)))
        table.add_row("Type", agent_data.get("agent_type", "Rule-based")[:15])
        table.add_row("Status", "🟢" if agent_data.get("status") == "ACTIVE" else "🔴")
        
        return table
    
    def create_system_metrics_table(self) -> Table:
        """Create system-wide metrics table."""
        table = Table(title="📊 System Metrics", border_style="green", box=None, show_header=False)
        table.add_column("Metric", style="bold white", width=18)
        table.add_column("Value", style="white", width=15)
        table.add_column("Status", style="green", width=8)
        
        # Calculate aggregate metrics
        total_agents = len(self.agent_data)
        active_agents = sum(1 for agent in self.agent_data.values() 
                          if time.time() - agent.get("last_update", 0) < 10)
        
        total_rewards = sum(agent.get("total_reward", 0) for agent in self.agent_data.values())
        avg_system_reward = total_rewards / max(1, total_agents)
        
        total_llm_calls = sum(agent.get("llm_calls", 0) for agent in self.agent_data.values())
        avg_llm_dependency = sum(agent.get("llm_dependency", 0) for agent in self.agent_data.values()) / max(1, total_agents)
        
        table.add_row("Episode", f"{self.system_metrics.get('current_episode', 0)}/{self.system_metrics.get('total_episodes', 0)}", "")
        table.add_row("Active Agents", f"{active_agents}/{total_agents}", "✅" if active_agents > 0 else "❌")
        table.add_row("System Reward", f"{total_rewards:.1f}", "")
        table.add_row("Avg Reward", f"{avg_system_reward:.2f}", "")
        table.add_row("Total LLM Calls", str(total_llm_calls), "")
        table.add_row("Avg LLM Depend.", f"{avg_llm_dependency:.1f}%", "")
        table.add_row("GPU Usage", f"{self.system_metrics.get('gpu_utilization', 0):.1f}%", "🚀" if self.system_metrics.get('gpu_utilization', 0) > 0 else "💻")
        table.add_row("Training", "Active" if self.system_metrics.get('training_active') else "Idle", "🟢" if self.system_metrics.get('training_active') else "🔴")
        
        return table
    
    def create_layout(self) -> Layout:
        """Create clean, professional layout."""
        layout = Layout()
        
        # Main layout: Header + Body
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body")
        )
        
        # Header with title
        header_text = Text("🧠 ARIASKA Multi-Agent Training System", style="bold white")
        layout["header"].update(Align.center(Panel(header_text, border_style="blue")))
        
        # Body: RedAgent (left) + Others (middle) + System (right)
        layout["body"].split_row(
            Layout(name="red_agent", ratio=2),
            Layout(name="other_agents", ratio=2), 
            Layout(name="system_metrics", ratio=1)
        )
        
        # RedAgent section
        layout["red_agent"].update(self.create_red_agent_table())
        
        # Other agents in 2x2 grid
        other_agents = ["BlueAgent", "OrionAgent", "ScoutAgent", "ShadowAgent"]
        agent_tables = []
        
        for agent_id in other_agents:
            if agent_id in self.agent_data:
                agent_tables.append(self.create_agent_table(agent_id))
            else:
                # Placeholder for inactive agents
                placeholder = Table(title=f"{agent_id}", border_style="dim", show_header=False)
                placeholder.add_column("", width=12)
                placeholder.add_column("", width=25)
                placeholder.add_row("Status", "Inactive")
                agent_tables.append(placeholder)
        
        # Arrange in 2x2 grid
        if len(agent_tables) >= 4:
            layout["other_agents"].split_column(
                Layout(name="row1"),
                Layout(name="row2")
            )
            layout["other_agents"]["row1"].split_row(
                Layout(name="agent1"),
                Layout(name="agent2")
            )
            layout["other_agents"]["row2"].split_row(
                Layout(name="agent3"),
                Layout(name="agent4")
            )
            
            layout["other_agents"]["row1"]["agent1"].update(agent_tables[0])
            layout["other_agents"]["row1"]["agent2"].update(agent_tables[1])
            layout["other_agents"]["row2"]["agent3"].update(agent_tables[2])
            layout["other_agents"]["row2"]["agent4"].update(agent_tables[3])
        
        # System metrics
        layout["system_metrics"].update(self.create_system_metrics_table())
        
        return layout
    
    def start_live_dashboard(self):
        """Start the professional live dashboard."""
        with Live(
            self.create_layout(),
            console=console,
            screen=False,
            auto_refresh=True,
            refresh_per_second=2
        ) as live:
            self.live_dashboard = live
            try:
                while True:
                    live.update(self.create_layout())
                    time.sleep(self.update_interval)
            except KeyboardInterrupt:
                console.print("\n[yellow]Dashboard terminated by user[/yellow]")
            finally:
                self.live_dashboard = None
    
    def update_display(self):
        """Update the live dashboard."""
        if self.live_dashboard:
            self.live_dashboard.update(self.create_layout())
    
    def display_static(self):
        """Display static snapshot."""
        console.clear()
        console.print(self.create_layout())

# Export the main classes
__all__ = ['ProfessionalAgentDashboard']

def create_professional_dashboard():
    """Create the professional dashboard instance."""
    return ProfessionalAgentDashboard(update_interval=0.5)

if __name__ == "__main__":
    # Test the dashboard
    dashboard = ProfessionalAgentDashboard()
    
    # Simulate RedAgent data
    dashboard.update_agent_status("RedAgent", {
        "action": "nmap -sC -sV 192.168.1.100",
        "target": "192.168.1.100:22,80,443",
        "output": "22/tcp open ssh OpenSSH 8.0",
        "next_step": "exploit ssh vulnerability",
        "reward": 15.5,
        "step_count": 25,
        "llm_calls": 8,
        "actions_taken": 24,
        "success_rate": 65.5,
        "agent_type": "Neural + LLM",
        "neural_loss": 0.0234,
        "confidence": 0.87
    })
    
    # Simulate other agents
    dashboard.update_agent_status("BlueAgent", {
        "action": "monitor network traffic",
        "reward": 8.2,
        "step_count": 22,
        "llm_calls": 3,
        "agent_type": "Deep Q-Network"
    })
    
    dashboard.update_system_metrics({
        "current_episode": 15,
        "total_episodes": 100,
        "gpu_utilization": 85.5,
        "training_active": True
    })
    
    dashboard.display_static()
