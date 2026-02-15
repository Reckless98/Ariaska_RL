#!/usr/bin/env python3
"""
Enhanced UI Helpers for ARIASKA_RL - Premium Agent Visibility
🎨 Real-time Agent Dashboards | 🧠 Neural Network Visualization | 📊 Coordination Matrix
"""

import time
import json
import numpy as np
from typing import Dict, List, Any, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.live import Live
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.columns import Columns
from rich.text import Text
from rich import box
from rich.rule import Rule

console = Console()

class EnhancedAgentDashboard:
    """Premium real-time agent dashboard with neural network insights."""
    
    def __init__(self, agents: Dict[str, Any], update_interval: float = 0.5):
        self.agents = agents
        self.update_interval = update_interval
        self.layout = Layout()
        self.metrics_history = {agent_id: [] for agent_id in agents.keys()}
        self.coordination_scores = np.zeros((len(agents), len(agents)))
        
    def create_redagent_neural_panel(self, redagent) -> Panel:
        """Create detailed RedAgent neural network monitoring panel."""
        content = []
        
        # Neural Network Status
        if hasattr(redagent, 'neural_trainer') and redagent.neural_trainer:
            trainer = redagent.neural_trainer
            
            # Policy Network Stats
            policy_table = Table(title="🧠 Policy Network", box=box.SIMPLE, show_header=True)
            policy_table.add_column("Metric", style="cyan")
            policy_table.add_column("Value", style="green")
            policy_table.add_column("Trend", style="yellow")
            
            # Get recent training metrics
            if hasattr(trainer, 'training_losses') and trainer.training_losses:
                recent_loss = trainer.training_losses[-1] if trainer.training_losses else 0.0
                avg_loss = np.mean(list(trainer.training_losses)[-10:]) if trainer.training_losses else 0.0
                loss_trend = "↓" if len(trainer.training_losses) > 1 and recent_loss < trainer.training_losses[-2] else "↑"
            else:
                recent_loss, avg_loss, loss_trend = 0.0, 0.0, "-"
                
            # GPT Dependency
            gpt_fallback = getattr(trainer, 'gpt_fallback_rate', 0.5)
            confidence = getattr(trainer, 'confidence_threshold', 0.7)
            
            policy_table.add_row("Training Loss", f"{recent_loss:.4f}", loss_trend)
            policy_table.add_row("Avg Loss (10)", f"{avg_loss:.4f}", "-")
            policy_table.add_row("GPT Fallback", f"{gpt_fallback:.2%}", "📉" if gpt_fallback < 0.3 else "🔄")
            policy_table.add_row("Confidence", f"{confidence:.2%}", "🎯")
            policy_table.add_row("Training Steps", f"{getattr(trainer, 'training_steps', 0)}", "↗")
            
            content.append(policy_table)
            
            # Q-Value Distribution
            if hasattr(trainer, 'q_value_history') and trainer.q_value_history:
                q_values = list(trainer.q_value_history)[-10:]
                q_table = Table(title="📊 Q-Value Distribution", box=box.SIMPLE)
                q_table.add_column("Recent Q-Values", style="magenta")
                
                for i, q_val in enumerate(q_values[-5:]):  # Last 5 Q-values
                    q_table.add_row(f"Step {len(q_values)-5+i}: {q_val:.3f}")
                    
                content.append(q_table)
        
        # Learning Progress
        learning_table = Table(title="📈 Learning Progress", box=box.SIMPLE)
        learning_table.add_column("Aspect", style="blue")
        learning_table.add_column("Current", style="green")
        learning_table.add_column("Target", style="yellow")
        
        epsilon = getattr(redagent, 'epsilon', 0.1)
        skill_level = getattr(redagent, 'skill_level', 0.0)
        
        learning_table.add_row("Exploration (ε)", f"{epsilon:.3f}", "0.05")
        learning_table.add_row("Skill Level", f"{skill_level:.2%}", "80%+")
        learning_table.add_row("Memory Size", f"{len(getattr(redagent, 'replay_buffer', []))}", "1000+")
        
        content.append(learning_table)
        
        # Combine all content
        combined_content = "\n\n".join([str(item) for item in content])
        
        return Panel(
            combined_content,
            title="🦂 RedAgent Neural Intelligence",
            border_style="red",
            padding=(1, 2)
        )
    
    def create_orion_oversight_panel(self, orion_agent) -> Panel:
        """Create OrionAgent strategic oversight panel."""
        
        # Strategic Insights
        insights_table = Table(title="👁️ Strategic Oversight", box=box.ROUNDED)
        insights_table.add_column("Agent", style="cyan", width=12)
        insights_table.add_column("Performance", style="green", width=15)
        insights_table.add_column("Status", style="yellow", width=15)
        insights_table.add_column("Directive", style="magenta", width=20)
        
        # Mock agent performance analysis (replace with real data)
        agent_performances = [
            ("RedAgent", "87.2% success", "Learning", "Continue aggressive tactics"),
            ("BlueAgent", "64.1% detection", "Alert", "Increase monitoring"),
            ("ScoutAgent", "92.8% recon", "Optimal", "Expand reconnaissance"),
            ("ShadowAgent", "78.5% stealth", "Good", "Maintain covert ops"),
        ]
        
        for agent, perf, status, directive in agent_performances:
            status_color = "[green]" if status == "Optimal" else "[yellow]" if status == "Good" else "[red]"
            insights_table.add_row(
                agent, 
                perf, 
                f"{status_color}{status}[/{status_color.replace('[', '').replace(']', '')}]",
                directive[:19]
            )
        
        # Coordination Matrix
        coord_matrix = Table(title="🤝 Agent Coordination Matrix", box=box.SIMPLE)
        coord_matrix.add_column("", style="dim", width=8)
        for agent_name in ["Red", "Blue", "Scout", "Shadow"]:
            coord_matrix.add_column(agent_name, width=6, justify="center")
        
        # Mock coordination scores (replace with real data)
        matrix_data = [
            ["Red", "1.0", "0.3", "0.8", "0.7"],
            ["Blue", "0.3", "1.0", "0.4", "0.2"],
            ["Scout", "0.8", "0.4", "1.0", "0.6"],
            ["Shadow", "0.7", "0.2", "0.6", "1.0"]
        ]
        
        for row in matrix_data:
            coord_matrix.add_row(*row)
        
        # Combine content
        content = f"{insights_table}\n\n{coord_matrix}"
        
        return Panel(
            content,
            title="👁️ Orion Strategic Command",
            border_style="blue",
            padding=(1, 2)
        )
    
    def create_agent_memory_panel(self, agent_name: str, agent) -> Panel:
        """Create agent memory and learning visualization."""
        
        memory_table = Table(title=f"🧠 {agent_name} Memory", box=box.SIMPLE)
        memory_table.add_column("Type", style="cyan")
        memory_table.add_column("Count", style="green")
        memory_table.add_column("Recent", style="yellow")
        
        # Memory stats
        replay_buffer_size = len(getattr(agent, 'replay_buffer', []))
        learned_patterns = len(getattr(agent, 'learned_patterns', {}))
        command_history = len(getattr(agent, 'command_history', []))
        
        memory_table.add_row("Replay Buffer", f"{replay_buffer_size}", "Active")
        memory_table.add_row("Learned Patterns", f"{learned_patterns}", "Growing")
        memory_table.add_row("Command History", f"{command_history}", "Updated")
        
        # Recent actions
        recent_table = Table(title="📝 Recent Actions", box=box.SIMPLE)
        recent_table.add_column("Step", width=6)
        recent_table.add_column("Action", width=20)
        recent_table.add_column("Result", width=8)
        
        # Mock recent actions (replace with real data)
        recent_actions = [
            ("157", "nmap -sV 10.10.10.10", "✓"),
            ("158", "enum4linux 10.10.10.10", "✓"),
            ("159", "hydra ssh://10.10.10.10", "✗"),
            ("160", "gobuster dir -u http://10.10.10.10", "✓"),
        ]
        
        for step, action, result in recent_actions[-4:]:
            recent_table.add_row(step, action[:19], result)
        
        content = f"{memory_table}\n\n{recent_table}"
        
        return Panel(
            content,
            title=f"💾 {agent_name} Intelligence",
            border_style="green",
            padding=(1, 2)
        )
    
    def create_training_metrics_panel(self, training_data: Dict) -> Panel:
        """Create comprehensive training metrics panel."""
        
        # Main metrics
        metrics_table = Table(title="📊 Training Metrics", box=box.ROUNDED)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Current", style="green")
        metrics_table.add_column("Best", style="yellow")
        metrics_table.add_column("Trend", style="magenta")
        
        metrics = [
            ("Episode", f"{training_data.get('current_episode', 0)}", f"{training_data.get('total_episodes', 100)}", "↗"),
            ("Avg Reward", f"{training_data.get('avg_reward', 0.0):.2f}", f"{training_data.get('best_reward', 0.0):.2f}", "📈"),
            ("Success Rate", f"{training_data.get('success_rate', 0.0):.1%}", f"{training_data.get('best_success_rate', 0.0):.1%}", "📊"),
            ("Coordination", f"{training_data.get('coordination_score', 0.0):.2f}", "1.00", "🤝"),
            ("GPT Usage", f"{training_data.get('gpt_usage', 0.0):.1%}", "< 30%", "📉"),
        ]
        
        for metric, current, best, trend in metrics:
            metrics_table.add_row(metric, current, best, trend)
        
        # Performance graph (simplified text representation)
        graph_content = "Performance Trend (Last 10 Episodes):\n"
        rewards = training_data.get('recent_rewards', [1.2, 1.5, 1.1, 1.8, 2.1, 1.9, 2.3, 2.0, 2.4, 2.2])
        max_reward = max(rewards) if rewards else 1.0
        
        for i, reward in enumerate(rewards[-10:]):
            bar_length = int((reward / max_reward) * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            graph_content += f"E{i+1:2d}: {bar} {reward:.1f}\n"
        
        content = f"{metrics_table}\n\n{graph_content}"
        
        return Panel(
            content,
            title="📈 Training Performance",
            border_style="magenta",
            padding=(1, 2)
        )
    
    def create_live_dashboard(self, training_data: Dict) -> Layout:
        """Create the complete live dashboard layout."""
        
        # Create main layout
        layout = Layout()
        
        # Split into top and bottom
        layout.split_column(
            Layout(name="top", ratio=7),
            Layout(name="bottom", ratio=3)
        )
        
        # Split top into left and right
        layout["top"].split_row(
            Layout(name="left", ratio=6),
            Layout(name="right", ratio=4)
        )
        
        # Split left into agent panels
        layout["left"].split_column(
            Layout(name="redagent", ratio=5),
            Layout(name="agents", ratio=5)
        )
        
        # Split right into coordination and metrics
        layout["right"].split_column(
            Layout(name="orion", ratio=6),
            Layout(name="metrics", ratio=4)
        )
        
        # Populate panels
        redagent = self.agents.get('RedAgent')
        orion_agent = self.agents.get('OrionAgent')
        
        if redagent:
            layout["redagent"].update(self.create_redagent_neural_panel(redagent))
        
        if orion_agent:
            layout["orion"].update(self.create_orion_oversight_panel(orion_agent))
        
        # Agent memory panels
        agent_panels = []
        for agent_name, agent in list(self.agents.items())[:2]:  # First 2 agents
            if agent_name not in ['RedAgent', 'OrionAgent']:
                agent_panels.append(self.create_agent_memory_panel(agent_name, agent))
        
        if agent_panels:
            if len(agent_panels) == 1:
                layout["agents"].update(agent_panels[0])
            else:
                layout["agents"].split_row(*[Layout(panel) for panel in agent_panels])
        
        # Training metrics
        layout["metrics"].update(self.create_training_metrics_panel(training_data))
        
        # Bottom status bar
        status_content = f"[bold green]🚀 ARIASKA_RL Enhanced Training[/bold green] | "
        status_content += f"Session: {training_data.get('session_id', 'N/A')} | "
        status_content += f"GPU: {training_data.get('device', 'CPU')} | "
        status_content += f"Uptime: {training_data.get('uptime', '0s')}"
        
        layout["bottom"].update(Panel(
            status_content,
            title="System Status",
            border_style="cyan"
        ))
        
        return layout

def display_enhanced_agent_status(agents_data: Dict, episode: int, step: int, 
                                 training_data: Optional[Dict] = None):
    """Display enhanced agent status with neural network insights."""
    
    console.print(Rule(f"[bold blue]Episode {episode} - Step {step}[/bold blue]"))
    
    # Create comprehensive agent tables
    main_table = Table(title="🤖 Agent Status & Neural Activity", 
                      show_header=True, header_style="bold", box=box.ROUNDED)
    main_table.add_column("Agent", style="cyan", width=12)
    main_table.add_column("Action", style="yellow", width=25)
    main_table.add_column("Target", style="blue", width=15)
    main_table.add_column("Neural", style="magenta", width=15)
    main_table.add_column("Reward", style="green", width=8)
    main_table.add_column("Status", style="white", width=8)
    main_table.add_column("Learning", style="red", width=15)
    
    for agent_name, data in agents_data.items():
        action = str(data.get('action', 'No action'))[:24]
        target = str(data.get('target', 'N/A'))[:14]
        reward = f"{data.get('reward', 0.0):.2f}"
        status = "✓" if data.get('success', False) else "✗"
        
        # Neural network status
        neural_status = "Active" if agent_name == "RedAgent" else "Rule-based"
        learning_status = "Adapting" if data.get('reward', 0) > 0 else "Exploring"
        
        main_table.add_row(
            agent_name, action, target, neural_status, 
            reward, status, learning_status
        )
    
    console.print(main_table)
    
    # Additional coordination matrix if training data provided
    if training_data and 'coordination_matrix' in training_data:
        coord_table = Table(title="🤝 Agent Coordination Strength", box=box.SIMPLE)
        coord_table.add_column("Agents", style="cyan")
        coord_table.add_column("Coordination Score", style="green")
        coord_table.add_column("Efficiency", style="yellow")
        
        matrix = training_data['coordination_matrix']
        agent_names = list(agents_data.keys())
        
        for i, agent1 in enumerate(agent_names[:3]):  # Show top 3 coordination pairs
            for j, agent2 in enumerate(agent_names[:3]):
                if i < j:  # Avoid duplicates
                    score = matrix[i][j] if len(matrix) > i and len(matrix[i]) > j else 0.5
                    efficiency = "High" if score > 0.7 else "Medium" if score > 0.4 else "Low"
                    coord_table.add_row(f"{agent1} ↔ {agent2}", f"{score:.2f}", efficiency)
        
        console.print(coord_table)

def display_learning_progress(agents: Dict[str, Any], session_data: Dict):
    """Display learning progress across all agents."""
    
    progress_table = Table(title="🧠 Multi-Agent Learning Progress", 
                          show_header=True, header_style="bold", box=box.DOUBLE)
    progress_table.add_column("Agent", style="cyan", width=12)
    progress_table.add_column("Algorithm", style="magenta", width=15)
    progress_table.add_column("Exploration", style="yellow", width=12)
    progress_table.add_column("Performance", style="green", width=15)
    progress_table.add_column("Memory Usage", style="blue", width=15)
    progress_table.add_column("GPT Integration", style="red", width=15)
    
    for agent_name, agent in agents.items():
        # Determine algorithm type
        if hasattr(agent, 'neural_trainer'):
            algorithm = "Double DQN"
        elif hasattr(agent, 'q_network'):
            algorithm = "Standard DQN"
        else:
            algorithm = "Rule-based"
        
        # Get exploration rate
        epsilon = getattr(agent, 'epsilon', 0.0)
        exploration = f"{epsilon:.3f}"
        
        # Performance metrics
        skill_level = getattr(agent, 'skill_level', 0.0)
        performance = f"{skill_level:.1%}"
        
        # Memory usage
        buffer_size = len(getattr(agent, 'replay_buffer', []))
        memory_usage = f"{buffer_size}/2000"
        
        # GPT integration status
        gpt_dependency = getattr(agent, 'gpt_dependency', 1.0)
        gpt_status = f"{gpt_dependency:.1%} dependency"
        
        progress_table.add_row(
            agent_name, algorithm, exploration, 
            performance, memory_usage, gpt_status
        )
    
    console.print(progress_table)
    
    # Training session summary
    summary_panel = Panel(
        f"[bold]Session Summary[/bold]\n"
        f"Episodes: {session_data.get('episode', 0)}/{session_data.get('total_episodes', 100)}\n"
        f"Total Actions: {session_data.get('total_actions', 0)}\n"
        f"Success Rate: {session_data.get('success_rate', 0.0):.1%}\n"
        f"Runtime: {session_data.get('runtime', 0.0):.1f}s",
        title="📊 Training Session",
        border_style="green"
    )
    
    console.print(summary_panel)
