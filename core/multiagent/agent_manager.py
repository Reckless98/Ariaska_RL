# core/multiagent/agent_manager.py — ARIASKA Agent Orchestrator v11.5 APEX PRIME

from rich.console import Console
from rich.panel import Panel
import subprocess
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, SpinnerColumn
from rich.table import Table
from rich import box
from rich.columns import Columns
import os
import time
import threading
import signal
from typing import Dict, List, Any, Optional

console = Console()


class AgentManager:
    def __init__(self, verbosity="standard"):
        console.rule("[bold cyan]🚀 ARIASKA Multi-Agent Deployment: System Initialization[/bold cyan]")
        # --- Grouped Initialization Logs ---
        agent_init_table = Table(title="Multi-Agent Deployment", box=box.ROUNDED)
        agent_init_table.add_column("Agent", style="cyan")
        agent_init_table.add_column("Mode", style="magenta")
        agent_init_table.add_column("Status", style="green")
        console.rule(
            "[bold cyan]🚀 Initializing ARIASKA Multi-Agent Orchestration v11.5 APEX PRIME"
        )

        # Shared Systems
        self.stats_monitor = self._import_stats_monitor()()
        self.verbosity = verbosity
        self.shared_context = {}
        self.event_log = []  # Shared, append-only event log for all agent/environment events
        
        # Defer agent creation to avoid circular references
        self.red_agent = None
        self.blue_agent = None
        self.scout_agent = None
        self.shadow_agent = None
        self.orion_agent = None
        self.agents = []
        
        # Create memory router first
        self.memory_router = self._import_memory_router()([])
        
        # Initialize agents
        if self.verbosity not in ["quiet", "silent"]:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("[bold cyan]Deploying multi-agent system...", total=100)
                
                # Step 1: Import dependencies
                progress.update(task, advance=15, description="[bold blue]Importing dependencies...")
                self.gpt_manager = self._import_gpt_manager()()
                
                # Step 2: Create agents
                progress.update(task, advance=35, description="[bold blue]Creating agents...")
                self._initialize_agents(agent_init_table)
                
                # Step 3: Initialize memory router with agents
                progress.update(task, advance=25, description="[bold blue]Linking memory subsystems...")
                self.memory_router = self._import_memory_router()(self.agents)
                
                # Step 4: Initialize agent links
                progress.update(task, advance=15, description="[bold blue]Establishing agent connections...")
                self._initialize_agent_links()
                
                # Step 5: Set up environment connections
                progress.update(task, advance=10, description="[bold blue]Configuring environment...")
                self._setup_environments()
                
                progress.update(task, completed=100)

        # Initialize hierarchical structure (OrionAgent as overseer)
        self._initialize_hierarchy()
        
        # Cache flag for context synchronization
        gpt_context_synced = self._sync_gpt_context()
        
        if self.verbosity not in ["quiet", "silent"]:
            self._print_startup_dashboard(agent_init_table, gpt_context_synced)
        
        # Startup signal handlers for clean shutdown
        self._setup_signal_handlers()

    def _import_memory_router(self):
        """Dynamically import MemoryRouter to avoid circular imports."""
        try:
            from core.multiagent.memory_router import MemoryRouter
            return MemoryRouter
        except ImportError:
            console.print("[yellow]⚠ MemoryRouter import failed, using placeholder[/yellow]")
            # Create placeholder class if import fails
            class PlaceholderMemoryRouter:
                def __init__(self, agents=None):
                    pass

                def store_experience(self, *args, **kwargs):
                    pass
                    
                def log_transition(self, *args, **kwargs):
                    pass

            return PlaceholderMemoryRouter

    def _import_gpt_manager(self):
        """Dynamically import GPTManager to avoid circular imports."""
        try:
            from core.gpt_manager import GPTManager
            return GPTManager
        except ImportError:
            console.print("[yellow]⚠ GPTManager import failed, using placeholder[/yellow]")
            # Create placeholder class if import fails
            class PlaceholderGPTManager:
                def __init__(self):
                    pass

                def smart_decision(self, *args, **kwargs):
                    return "Placeholder GPTManager response"

            return PlaceholderGPTManager

    def _import_stats_monitor(self):
        """Dynamically import StatsMonitor to avoid circular imports."""
        try:
            from core.utils.stats_monitor import StatsMonitor
            return StatsMonitor
        except ImportError:
            console.print("[yellow]⚠ StatsMonitor import failed, using placeholder[/yellow]")
            # Create placeholder class if import fails
            class PlaceholderStatsMonitor:
                def __init__(self):
                    self.agent_stats = {}

                def log_step(self, *args, **kwargs):
                    pass

            return PlaceholderStatsMonitor

    def _initialize_agents(self, agent_init_table):
        """Initialize all agents in the system."""
        try:
            from core.agents.red_agent import RedAgent
            from core.agents.blue_agent import BlueAgent
            from core.agents.scout_agent import ScoutAgent
            from core.agents.shadow_agent import ShadowAgent
            from core.agents.orion_agent import OrionAgent

            # Create agents
            self.red_agent = RedAgent(agent_manager=self, memory_router=self.memory_router, verbosity=self.verbosity)
            self.blue_agent = BlueAgent(agent_manager=self, memory_router=self.memory_router, verbosity=self.verbosity)
            self.scout_agent = ScoutAgent(agent_manager=self, memory_router=self.memory_router, verbosity=self.verbosity)
            self.shadow_agent = ShadowAgent(agent_manager=self, memory_router=self.memory_router, verbosity=self.verbosity)
            self.orion_agent = OrionAgent(agent_manager=self, memory_router=self.memory_router, verbosity=self.verbosity)

            # Add to agents list
            self.agents = [
                self.red_agent,
                self.blue_agent,
                self.scout_agent,
                self.shadow_agent,
                self.orion_agent
            ]

            # Update the table with agent information
            if agent_init_table:
                for agent in self.agents:
                    agent_init_table.add_row(
                        agent.agent_id,
                        getattr(agent, "current_mode", "Standard"),
                        "✓ Active"
                    )

        except ImportError as e:
            console.print(f"[red]❌ Agent initialization failed: {e}[/red]")
            raise

    def _initialize_agent_links(self):
        """Initialize links between agents."""
        for agent in self.agents:
            if hasattr(agent, "_init_multiagent_links"):
                agent._init_multiagent_links()

    def _initialize_hierarchy(self):
        """Setup hierarchical structure with OrionAgent as overseer."""
        if self.orion_agent:
            for agent in self.agents:
                if agent.agent_id != "OrionAgent" and hasattr(self.orion_agent, "register_subordinate"):
                    self.orion_agent.register_subordinate(agent)

    def _setup_environments(self):
        """Configure environments for agents that need them."""
        # Import only when needed
        try:
            from core.environment.cyber_environment import CyberEnvironment
            
            # Initialize environments for agents needing them
            if self.red_agent and not hasattr(self.red_agent, "env"):
                self.red_agent.env = CyberEnvironment(agent_manager=self)
                
            if self.blue_agent and not hasattr(self.blue_agent, "env"):
                # Blue agent typically shares red agent's environment
                if hasattr(self.red_agent, "env"):
                    self.blue_agent.env = self.red_agent.env
                else:
                    self.blue_agent.env = CyberEnvironment(agent_manager=self)
                    
        except ImportError as e:
            console.print(f"[yellow]⚠ Environment setup warning: {e}[/yellow]")

    def _sync_gpt_context(self):
        """Synchronize GPT context and configurations across agents."""
        try:
            # Ensure all agents use the same GPTManager instance
            for agent in self.agents:
                if hasattr(agent, "gpt_manager"):
                    agent.gpt_manager = self.gpt_manager
            return True
        except Exception as e:
            console.print(f"[yellow]⚠ GPT context sync warning: {e}[/yellow]")
            return False

    def _setup_signal_handlers(self):
        """Setup signal handlers for clean shutdown."""
        # Handle SIGINT (Ctrl+C) and SIGTERM
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, sig, frame):
        """Handle shutdown signals."""
        console.print("\n[yellow]⚠ Shutdown signal received. Cleaning up...[/yellow]")
        self.shutdown()
        os._exit(0)  # Force exit after cleanup

    def get_agent(self, agent_id):
        """Get an agent by ID."""
        if agent_id == "RedAgent":
            return self.red_agent
        elif agent_id == "BlueAgent":
            return self.blue_agent
        elif agent_id == "ScoutAgent":
            return self.scout_agent
        elif agent_id == "ShadowAgent":
            return self.shadow_agent
        elif agent_id == "OrionAgent":
            return self.orion_agent
        return None

    def all_agents(self):
        """Get all agents."""
        return self.agents

    def broadcast(self, key, value, sender=None):
        """Broadcast a message to all agents."""
        # Log broadcast to event log
        self.event_log.append({
            "event_type": "broadcast",
            "key": key,
            "value": value,
            "sender": sender,
            "timestamp": time.time()
        })
        
        # Update shared context
        self.shared_context[key] = value
        
        # Notify agents with receive_broadcast method
        for agent in self.agents:
            if hasattr(agent, "receive_broadcast"):
                try:
                    agent.receive_broadcast(key, value, sender=sender)
                except Exception as e:
                    console.print(f"[yellow]⚠ Broadcast error for {agent.agent_id}: {e}[/yellow]")

    def query_context(self, key, default=None):
        """Get a value from the shared context."""
        return self.shared_context.get(key, default)

    def simulate_all_agents(self, episode=1, step=1):
        """Run a simulation step for all agents."""
        turn_events = []
        
        # Get current shared context
        shared_state = self._get_latest_shared_state()
        
        # 1. Scout phase determination
        if self.scout_agent:
            scout_result = self.scout_agent.simulate_step(episode, step, shared_state)
            turn_events.append(scout_result)
            # Broadcast phase if available
            if scout_result and isinstance(scout_result, dict) and "phase" in scout_result:
                self.broadcast("ScoutAgent_phase", scout_result["phase"], sender="ScoutAgent")
                
        # 2. OrionAgent strategic oversight
        if self.orion_agent:
            orion_result = self.orion_agent.simulate_step(episode, step, shared_state)
            turn_events.append(orion_result)
            
        # 3. RedAgent and BlueAgent actions
        if self.red_agent:
            red_result = self.red_agent.simulate_step(episode, step, shared_state)
            turn_events.append(red_result)
            
        if self.blue_agent:
            blue_result = self.blue_agent.simulate_step(episode, step, shared_state)
            turn_events.append(blue_result)
            
        # 4. ShadowAgent memory optimization
        if self.shadow_agent:
            shadow_result = self.shadow_agent.simulate_step(episode, step, shared_state)
            turn_events.append(shadow_result)
        
        # Apply actions to environment and get results
        env_events = self._apply_actions_to_environment(turn_events)
        
        # Sync agent memories
        self._sync_all_agent_memories()
        
        # Periodic global optimization
        if step % 5 == 0:
            self._multiagent_sync()
            
        # Return combined events
        return turn_events + env_events

    def _apply_actions_to_environment(self, turn_events):
        env_events = []
        
        red_action = next((e for e in turn_events if e.get("agent_id") == "RedAgent"), None)
        blue_action = next((e for e in turn_events if e.get("agent_id") == "BlueAgent"), None)
        # Apply RedAgent action
        if red_action and hasattr(self.red_agent, "env"):
            try:
                state, reward, done, info = self.red_agent.env.step(red_action.get("command"))
                env_events.append({
                    "event_type": "env_transition",
                    "agent_id": "RedAgent",
                    "state": state,
                    "reward": reward,
                    "done": done,
                    "info": info,
                    "step": red_action.get("step"),
                    "episode": red_action.get("episode")
                })
            except Exception as e:
                env_events.append({
                    "event_type": "env_transition",
                    "agent_id": "RedAgent",
                    "error": str(e),
                    "step": red_action.get("step"),
                    "episode": red_action.get("episode")
                })
        # Apply BlueAgent action (if needed)
        if blue_action and hasattr(self.blue_agent, "env"):
            try:
                # Optionally, BlueAgent may react to RedAgent's action or environment state
                # For now, just log the action as an event
                env_events.append({
                    "event_type": "env_transition",
                    "agent_id": "BlueAgent",
                    "action": blue_action.get("command"),
                    "step": blue_action.get("step"),
                    "episode": blue_action.get("episode")
                })
            except Exception as e:
                env_events.append({
                    "event_type": "env_transition",
                    "agent_id": "BlueAgent",
                    "error": str(e),
                    "step": blue_action.get("step"),
                    "episode": blue_action.get("episode")
                })
        # Add more agent/environment coordination as needed
        return env_events

    def _get_latest_shared_state(self):
        """
        Return the latest shared state for agents to consume.
        This can include environment state, last actions, etc.
        """
        # For simplicity, use RedAgent's environment state as the canonical shared state
        if hasattr(self.red_agent, "env"):
            return self.red_agent.env.get_global_state()
        return {}

    def _sync_all_agent_memories(self):
        """
        Serialize and atomically sync all agent memories after environment update.
        """
        # Import MemorySyncInterface locally to avoid circular import issues
        try:
            from core.multiagent.memory_router import MemorySyncInterface
        except ImportError:
            MemorySyncInterface = type("MemorySyncInterface", (), {})
        for agent in self.agents:
            if isinstance(agent, MemorySyncInterface):
                try:
                    agent.sync_memory()
                except Exception as e:
                    console.print(f"[yellow]⚠ Memory sync failed for {agent.agent_id}: {e}[/yellow]")

    def replay_event_log(self):
        """
        Replay the event log for validation and reproducibility.
        """
        console.rule("[bold cyan]🔁 Replaying Multi-Agent Event Log")
        for event in self.event_log:
            if event.get("event_type") == "action":
                console.print(f"[blue]{event.get('agent_id')} action: {event.get('command')} (step {event.get('step')})[/blue]")
            elif event.get("event_type") == "env_transition":
                console.print(f"[green]Env transition by {event.get('agent_id')}: reward={event.get('reward', '')} done={event.get('done', '')}[/green]")
            elif "error" in event:
                console.print(f"[red]Error: {event['error']}[/red]")

    def _simulate_agent_step(self, agent, episode, step, shared_context=None):
        info = {
            "command": "N/A", 
            "phase": "N/A",
            "reward": 0.0,
            "gpt_calls": 0,
            "output": "N/A"
        }
        try:
            step_info = agent.simulate_step(episode, step, shared_context)
            return step_info if step_info else info
        except Exception as e:
            console.print(f"[red]❌ Error simulating {agent.agent_id}: {e}[/red]")
            return {**info, "error": str(e), "agent_id": agent.agent_id}

    def _print_startup_dashboard(self, agent_init_table, gpt_context_synced):
        # Create dashboards for startup visualization
        agents_summary = Table(box=box.SIMPLE)
        agents_summary.add_column("Agent")
        agents_summary.add_column("Role")
        agents_summary.add_column("Status")
        
        # Add agents to summary
        for agent in self.agents:
            agents_summary.add_row(
                agent.agent_id,
                getattr(agent, "role", "N/A"),
                "[green]Active[/green]"
            )
            
        # Create GPT table
        gpt_table = Table(box=box.SIMPLE)
        gpt_table.add_column("Model")
        gpt_table.add_column("Status")
        
        # Add GPT info
        gpt_table.add_row("GPT-4o-mini", "[green]Ready[/green]" if gpt_context_synced else "[yellow]Pending[/yellow]")
        gpt_table.add_row("GPT-4.1-nano", "[green]Ready[/green]" if gpt_context_synced else "[yellow]Pending[/yellow]")
        
        # Create environment table
        env_table = Table(box=box.SIMPLE)
        env_table.add_column("Setting")
        env_table.add_column("Value")
        
        # Add environment info
        env_mode = "Simulated" if not os.environ.get("ARIASKA_LIVE_MODE") else "Live"
        env_table.add_row("Mode", env_mode)
        env_table.add_row("Difficulty", os.environ.get("ARIASKA_DIFFICULTY", "Standard"))
        
        # Create memory table
        mem_table = Table(box=box.SIMPLE)
        mem_table.add_column("Setting")
        mem_table.add_column("Value")
        
        # Add memory info
        mem_table.add_row("Router", "[green]Active[/green]" if self.memory_router else "[red]Inactive[/red]")
        mem_table.add_row("Cache", "Enabled")
        
        # Create curriculum table
        curriculum_table = Table(box=box.SIMPLE)
        curriculum_table.add_column("Setting")
        curriculum_table.add_column("Value")
        
        # Add curriculum info
        curriculum_table.add_row("Mode", "Progressive")
        curriculum_table.add_row("Current", "Phase 1")
        
        # Combine all tables into a dashboard
        dashboard = Columns([
            Panel(agents_summary, title="Agents Active", border_style="green"),
            Panel(gpt_table, title="LLM Orchestration", border_style="magenta"),
            Panel(env_table, title="Environment Mode", border_style="cyan"),
            Panel(mem_table, title="Memory", border_style="yellow"),
            Panel(curriculum_table, title="Curriculum", border_style="blue"),
        ], equal=True)
        console.rule("[bold green]🧠 ARIASKA Multi-Agent System Ready[/bold green]")
        console.print(dashboard)
        # Phase data placeholder
        phase_data = getattr(self.red_agent.env, "current_phase", None) if hasattr(self.red_agent, "env") else None
        if not phase_data or phase_data == "N/A":
            console.print("[yellow]ℹ️ Awaiting first agent action to generate phase data...[/yellow]")

    def _multiagent_sync(self):
        """Periodic sync: agents can adjust strategies based on group state."""
        # Example: broadcast current phases
        for agent in self.agents:
            if hasattr(agent, "current_mode"):
                self.broadcast(f"{agent.agent_id}_phase", getattr(agent, "current_mode", "N/A"), sender=agent.agent_id)
        # Example: let Orion/Scout/Shadow analyze and suggest adjustments
        if hasattr(self, "orion_agent") and self.orion_agent:
            self.orion_agent.apply_orion_strategic_adjustments(self.agents)
        if hasattr(self, "shadow_agent") and self.shadow_agent:
            self.shadow_agent.optimize_all_agents_memory(self.agents)
            
    def batch_train_all(self, batches=1):
        """Train all agents with batch updates."""
        for _ in range(batches):
            for agent in self.agents:
                if hasattr(agent, "train_on_batch"):
                    try:
                        agent.train_on_batch()
                    except Exception as e:
                        console.print(f"[yellow]⚠ Training error for {agent.agent_id}: {e}[/yellow]")
                        
    def save_all_models(self):
        """Save all agent models."""
        for agent in self.agents:
            if hasattr(agent, "save_model"):
                try:
                    agent.save_model()
                except Exception as e:
                    console.print(f"[yellow]⚠ Model save error for {agent.agent_id}: {e}[/yellow]")
                    
    def snapshot_all(self):
        """Take snapshots of all agent states."""
        if hasattr(self.memory_router, "snapshot_all_memories"):
            try:
                self.memory_router.snapshot_all_memories()
            except Exception as e:
                console.print(f"[yellow]⚠ Memory snapshot error: {e}[/yellow]")
                
    def display_full_status(self):
        """Display comprehensive status of all agents."""
        console.rule("[bold cyan]📊 ARIASKA Multi-Agent System Status[/bold cyan]")
        
        for agent in self.agents:
            if hasattr(agent, "display_advanced_status"):
                agent.display_advanced_status()
            else:
                # Basic status display for agents without custom display
                status_panel = Panel(
                    f"Role: {getattr(agent, 'role', 'Unknown')}\n"
                    f"Mode: {getattr(agent, 'current_mode', 'Standard')}\n"
                    f"Last Action: {getattr(agent, 'last_action', 'None')}",
                    title=f"[bold]{agent.agent_id}[/bold]",
                    border_style="cyan"
                )
                console.print(status_panel)
                
        # Display memory stats if available
        if hasattr(self.memory_router, "get_stats"):
            try:
                mem_stats = self.memory_router.get_stats()
                mem_table = Table(title="Memory Statistics")
                mem_table.add_column("Metric")
                mem_table.add_column("Value")
                
                for key, value in mem_stats.items():
                    if isinstance(value, dict):
                        mem_table.add_row(key, str(len(value)))
                    else:
                        mem_table.add_row(key, str(value))
                        
                console.print(mem_table)
            except Exception as e:
                console.print(f"[yellow]⚠ Memory stats error: {e}[/yellow]")
        
    def shutdown(self):
        """Clean shutdown of all agents and resources."""
        console.print("[cyan]🔄 Shutting down ARIASKA Multi-Agent System...[/cyan]")
        
        # Close memory router first to ensure data persistence
        if hasattr(self.memory_router, "close"):
            try:
                self.memory_router.close()
            except Exception as e:
                console.print(f"[yellow]⚠ Memory router close error: {e}[/yellow]")
        
        # Shutdown all agents
        for agent in self.agents:
            if hasattr(agent, "safe_shutdown"):
                try:
                    agent.safe_shutdown()
                except Exception as e:
                    console.print(f"[yellow]⚠ Agent shutdown error for {agent.agent_id}: {e}[/yellow]")
        
        console.print("[green]✓ ARIASKA Multi-Agent System shutdown complete[/green]")


# ─────────────────────────────────────────────
# 🚀 CLI Diagnostic Mode
# ─────────────────────────────────────────────
if __name__ == "__main__":
    manager = AgentManager()
    manager.simulate_all_agents(episodes=5)
    manager.batch_train_all(batches=3)
    manager.save_all_models()
    manager.snapshot_all()
    manager.display_full_status()
