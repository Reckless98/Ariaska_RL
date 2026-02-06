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
        """
        Initialize the AgentManager.
        
        Args:
            verbosity: "standard" (full output), "quiet" (minimal), "silent" (none)
        
        PHASE 1 FIX: All agent/env initialization ALWAYS runs regardless of verbosity.
        Only print/log output is gated by verbosity.
        """
        self.verbosity = verbosity
        _verbose = verbosity not in ["quiet", "silent"]
        
        # Only print banners in verbose mode
        if _verbose:
            console.rule("[bold cyan]🚀 ARIASKA Multi-Agent Deployment: System Initialization[/bold cyan]")
            console.rule(
                "[bold cyan]🚀 Initializing ARIASKA Multi-Agent Orchestration v11.5 APEX PRIME"
            )
        
        # --- Grouped Initialization Logs (only used in verbose mode) ---
        agent_init_table = Table(title="Multi-Agent Deployment", box=box.ROUNDED)
        agent_init_table.add_column("Agent", style="cyan")
        agent_init_table.add_column("Mode", style="magenta")
        agent_init_table.add_column("Status", style="green")

        # Shared Systems (ALWAYS initialize)
        self.stats_monitor = self._import_stats_monitor()()
        self.shared_context = {}
        self.event_log = []  # Shared, append-only event log for all agent/environment events
        
        # Defer agent creation to avoid circular references
        self.red_agent = None
        self.blue_agent = None
        self.scout_agent = None
        self.shadow_agent = None
        self.orion_agent = None
        self.agents = []
        
        # Create memory router first (ALWAYS)
        self.memory_router = self._import_memory_router()()
        
        # ALWAYS initialize GPT manager
        self.gpt_manager = self._import_gpt_manager()()
        
        # Initialize agents - ALWAYS (with optional progress display)
        if _verbose:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("[bold cyan]Deploying multi-agent system...", total=100)
                
                progress.update(task, advance=35, description="[bold blue]Creating agents...")
                self._initialize_agents(agent_init_table)
                
                progress.update(task, advance=25, description="[bold blue]Linking memory subsystems...")
                self.memory_router = self._import_memory_router()()
                
                progress.update(task, advance=20, description="[bold blue]Establishing agent connections...")
                self._initialize_agent_links()
                
                progress.update(task, advance=20, description="[bold blue]Configuring environment...")
                self._setup_environments()
                
                progress.update(task, completed=100)
        else:
            # Quiet/silent mode: still initialize everything, just no output
            self._initialize_agents(agent_init_table)
            self.memory_router = self._import_memory_router()()
            self._initialize_agent_links()
            self._setup_environments()

        # Initialize hierarchical structure (OrionAgent as overseer) - ALWAYS
        self._initialize_hierarchy()
        
        # Cache flag for context synchronization - ALWAYS
        gpt_context_synced = self._sync_gpt_context()
        
        # Only print dashboard in verbose mode
        if _verbose:
            self._print_startup_dashboard(agent_init_table, gpt_context_synced)
        
        # Startup signal handlers for clean shutdown - ALWAYS
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
                
                def get_stats(self, *args, **kwargs):
                    return {"placeholder": True}
                
                def snapshot_all_memories(self, *args, **kwargs):
                    return "placeholder_snapshot"
                
                def close(self, *args, **kwargs):
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
        """Initialize all agents in the system.
        
        PHASE 0 FIX: Pass the shared gpt_manager to all agents to avoid
        creating multiple GPTManager instances.
        """
        try:
            from core.agents.red_agent import RedAgent
            from core.agents.blue_agent import BlueAgent
            from core.agents.scout_agent import ScoutAgent
            from core.agents.shadow_agent import ShadowAgent
            from core.agents.orion_agent import OrionAgent

            # Create agents with shared GPTManager
            self.red_agent = RedAgent(
                agent_manager=self, 
                memory_router=self.memory_router, 
                verbosity=self.verbosity,
                gpt_manager=self.gpt_manager  # PHASE 0: Inject shared GPTManager
            )
            self.blue_agent = BlueAgent(
                agent_manager=self, 
                memory_router=self.memory_router, 
                verbosity=self.verbosity,
                gpt_manager=self.gpt_manager  # PHASE 0: Inject shared GPTManager
            )
            self.scout_agent = ScoutAgent(
                agent_manager=self, 
                memory_router=self.memory_router, 
                verbosity=self.verbosity
            )
            self.shadow_agent = ShadowAgent(
                agent_manager=self, 
                memory_router=self.memory_router, 
                verbosity=self.verbosity
            )
            self.orion_agent = OrionAgent(
                agent_manager=self, 
                memory_router=self.memory_router, 
                verbosity=self.verbosity
            )

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
        """Configure environments for agents that need them.
        
        PHASE 0 FIX: Ensure Red and Blue share the SAME environment instance.
        We unconditionally assign Blue's env to Red's env to guarantee a single
        shared CyberEnvironment for proper multi-agent interplay.
        """
        try:
            from core.environment.cyber_environment import CyberEnvironment
            
            # Create a single shared environment for RedAgent
            # This is the canonical environment - all agents share this
            if self.red_agent:
                # Force create/replace RedAgent's env to ensure clean state
                if not hasattr(self.red_agent, "env") or self.red_agent.env is None:
                    self.red_agent.env = CyberEnvironment(agent_manager=self)
                    console.print("[green]✓ Created shared CyberEnvironment for RedAgent[/green]")
                
                # CRITICAL: BlueAgent MUST share RedAgent's environment
                # Unconditionally assign to prevent separate env instances
                if self.blue_agent:
                    self.blue_agent.env = self.red_agent.env  # Force shared reference
                    console.print("[green]✓ BlueAgent now shares RedAgent's environment[/green]")
                    
                # Also share with other agents that need environment access
                for agent in [self.scout_agent, self.shadow_agent, self.orion_agent]:
                    if agent and hasattr(agent, "env"):
                        agent.env = self.red_agent.env
                        
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

    def simulate_all_agents(self, episode=1, step=1, episodes=None, max_steps=None):
        """Run a simulation across multiple episodes and steps.
        
        Args:
            episode: Current episode number (legacy parameter)
            step: Current step number within the episode
            episodes: Number of episodes to run (overrides episode if provided)
            max_steps: Maximum steps per episode
        """
        # Handle both episode and episodes parameters for backward compatibility
        if episodes is not None:
            episode = episodes
            
        # Default max_steps if not provided
        if max_steps is None:
            max_steps = 40
            
        all_events = []
        
        # Run for the specified number of episodes
        for ep in range(1, episode + 1):
            console.print(f"[bold cyan]Episode {ep}/{episode}[/bold cyan]")
            
            # Reset environment for each episode if possible
            if self.red_agent and hasattr(self.red_agent, "env"):
                self.red_agent.env.reset()
                
            # Reset agents for the new episode
            for agent in self.agents:
                if hasattr(agent, "reset"):
                    agent.reset()
            
            # Run for the specified number of steps per episode
            for st in range(1, max_steps + 1):
                console.print(f"[cyan]Step {st}/{max_steps}[/cyan]")
                
                # Get current shared context
                shared_state = self._get_latest_shared_state()
                
                turn_events = []
                
                # 1. Scout phase determination
                if self.scout_agent:
                    scout_result = self.scout_agent.simulate_step(ep, st, shared_state)
                    turn_events.append(scout_result)
                    # Broadcast phase if available
                    if scout_result and isinstance(scout_result, dict) and "phase" in scout_result:
                        self.broadcast("ScoutAgent_phase", scout_result["phase"], sender="ScoutAgent")
                        
                # 2. OrionAgent strategic oversight
                if self.orion_agent:
                    orion_result = self.orion_agent.simulate_step(ep, st, shared_state)
                    turn_events.append(orion_result)
                    
                # 3. RedAgent and BlueAgent actions
                if self.red_agent:
                    red_result = self.red_agent.simulate_step(ep, st, shared_state)
                    turn_events.append(red_result)
                    
                if self.blue_agent:
                    blue_result = self.blue_agent.simulate_step(ep, st, shared_state)
                    turn_events.append(blue_result)
                    
                # 4. ShadowAgent memory optimization
                if self.shadow_agent:
                    shadow_result = self.shadow_agent.simulate_step(ep, st, shared_state)
                    turn_events.append(shadow_result)
                
                # Apply actions to environment and get results
                env_events = self._apply_actions_to_environment(turn_events)
                
                # Track events
                all_events.extend(turn_events + env_events)
                
                # Sync agent memories
                self._sync_all_agent_memories()
                
                # Periodic global optimization
                if st % 5 == 0:
                    self._multiagent_sync()
                
                # Check if the environment is done
                env_done = False
                for event in env_events:
                    if event.get("done", False) or (isinstance(event.get("info"), dict) and event.get("info", {}).get("done", False)):
                        env_done = True
                        break
                
                # If environment signals done, break the step loop
                if env_done:
                    console.print("[yellow]⚠ Environment signals episode completion. Moving to next episode.[/yellow]")
                    break
            
            # After completing all steps for this episode
            # Execute post-episode processing
            self._run_episode_analysis(ep)
            
            # Update stats for all agents
            if hasattr(self, 'stats_monitor'):
                for agent in self.agents:
                    if hasattr(agent, 'stats_monitor') and hasattr(agent.stats_monitor, 'get_average_reward'):
                        avg_reward = agent.stats_monitor.get_average_reward()
                        console.print(f"[green]{agent.agent_id} Average Reward: {avg_reward:.2f}[/green]")
                    
        # Return all events from all episodes
        return all_events
        
    def _run_episode_analysis(self, episode_num):
        """Run analysis after each episode completes."""
        # Get Orion to analyze all agents' performance
        if self.orion_agent and hasattr(self.orion_agent, "analyze_training"):
            try:
                analysis = self.orion_agent.analyze_training(self.agents)
                if isinstance(analysis, dict) and 'gpt_insight' in analysis:
                    console.print(f"[blue]👁️ Orion Training Analysis:[/blue] {analysis['gpt_insight']}")
            except Exception as e:
                console.print(f"[yellow]⚠ Orion analysis error: {e}[/yellow]")
                
        # Update global strategy
        if self.orion_agent and hasattr(self.orion_agent, "update_global_strategy"):
            try:
                strategy_insight = self.orion_agent.update_global_strategy(self.agents)
                console.print(f"[blue]👁️ Strategy Update:[/blue] {strategy_insight}")
            except Exception as e:
                console.print(f"[yellow]⚠ Strategy update error: {e}[/yellow]")
                
        # Display per-episode metrics
        self._display_episode_metrics(episode_num)

    def _apply_actions_to_environment(self, turn_events):
        """Apply agent actions to the shared environment.
        
        PHASE 0 FIX: Only step the environment ONCE per turn (for Red's action).
        Blue's defensive actions are processed via env._process_blue_defense()
        without triggering a separate env.step().
        """
        env_events = []
        
        red_action = next((e for e in turn_events if e.get("agent_id") == "RedAgent"), None)
        blue_action = next((e for e in turn_events if e.get("agent_id") == "BlueAgent"), None)
        
        # Step 1: Apply ONLY RedAgent's action via env.step()
        # This is the SINGLE environment step per turn
        if red_action and self.red_agent and hasattr(self.red_agent, "env"):
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
        
        # Step 2: Process BlueAgent's defensive reaction via react_to_action()
        # This does NOT call env.step() - only applies defense via _process_blue_defense()
        if blue_action and self.blue_agent and hasattr(self.blue_agent, "react_to_action"):
            try:
                # Get Blue's defensive response to Red's action
                red_cmd = red_action.get("command", "") if red_action else ""
                defense_result = self.blue_agent.react_to_action(red_cmd)
                
                # Apply defense to environment without stepping
                if defense_result and hasattr(self.red_agent.env, "_process_blue_defense"):
                    self.red_agent.env._process_blue_defense(defense_result)
                    
                    # Log the defensive action
                    env_events.append({
                        "event_type": "blue_defense",
                        "agent_id": "BlueAgent",
                        "defense_result": defense_result,
                        "honeypots_deployed": defense_result.get("honeypots_deployed", []),
                        "credentials_reset": defense_result.get("credentials_reset", False),
                        "step": blue_action.get("step"),
                        "episode": blue_action.get("episode")
                    })
                    
                    if defense_result.get("honeypots_deployed"):
                        console.print(f"[blue]🛡️ BlueAgent deployed honeypots: {defense_result['honeypots_deployed']}[/blue]")
                    if defense_result.get("credentials_reset"):
                        console.print(f"[blue]🔐 BlueAgent reset credentials[/blue]")
                        
            except Exception as e:
                env_events.append({
                    "event_type": "blue_defense",
                    "agent_id": "BlueAgent",
                    "error": str(e),
                    "step": blue_action.get("step"),
                    "episode": blue_action.get("episode")
                })
        
        return env_events

    def _get_latest_shared_state(self):
        """
        Return the latest shared state for agents to consume.
        This can include environment state, last actions, etc.
        """
        # For simplicity, use RedAgent's environment state as the canonical shared state
        if self.red_agent and hasattr(self.red_agent, "env"):
            return self.red_agent.env.get_global_state()
        return {}

    def _sync_all_agent_memories(self):
        """
        Serialize and atomically sync all agent memories after environment update.
        """
        # Sync memory for all agents that have sync_memory method
        for agent in self.agents:
            if hasattr(agent, "sync_memory"):
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
        phase_data = getattr(self.red_agent.env, "current_phase", None) if self.red_agent and hasattr(self.red_agent, "env") else None
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
                
    def get_primary_agent(self):
        """Get the primary agent (RedAgent by default)."""
        return self.red_agent or self.agents[0] if self.agents else None
    
    def train_all_batches(self, batches=1):
        """Train all agents with batch updates - alias for batch_train_all."""
        return self.batch_train_all(batches)
    
    def display_all_status(self):
        """Display status of all agents - alias for display_full_status."""
        return self.display_full_status()

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

    def _display_episode_metrics(self, episode_num):
        """Display metrics after each episode."""
        try:
            # Create a table for all agent metrics
            table = Table(title=f"Episode {episode_num} Results", box=box.ROUNDED)
            table.add_column("Agent", style="cyan")
            table.add_column("Avg Reward", style="green")
            table.add_column("Steps", style="yellow")
            table.add_column("Phase", style="magenta")
            table.add_column("Mode", style="blue")
            
            # Collect metrics from all agents
            for agent in self.agents:
                avg_reward = 0.0
                steps = 0
                phase = "N/A"
                mode = getattr(agent, "current_mode", "Standard")
                
                # Get metrics from stats_monitor if available
                if hasattr(agent, "stats_monitor"):
                    stats_monitor = agent.stats_monitor
                    if hasattr(stats_monitor, "get_average_reward"):
                        avg_reward = stats_monitor.get_average_reward()
                    if hasattr(stats_monitor, "get_steps"):
                        steps = stats_monitor.get_steps(agent.agent_id)
                    if hasattr(stats_monitor, "get_current_phase"):
                        phase = stats_monitor.get_current_phase(agent.agent_id) or "N/A"
                
                # Get environment phase if agent has access to environment
                if hasattr(agent, "env") and phase == "N/A":
                    if hasattr(agent.env, "current_phase"):
                        phase = agent.env.current_phase
                
                table.add_row(
                    agent.agent_id,
                    f"{avg_reward:.2f}",
                    str(steps),
                    phase,
                    mode
                )
            
            # Display metrics table
            console.print(table)
            
            # Add environment state summary if available
            if self.red_agent and hasattr(self.red_agent, "env"):
                env = self.red_agent.env
                if hasattr(env, "get_global_state"):
                    env_state = env.get_global_state()
                    
                    # Filter out large objects for cleaner display
                    filtered_state = {}
                    for key, value in env_state.items():
                        if isinstance(value, list) and len(value) > 5:
                            filtered_state[key] = f"{len(value)} items"
                        else:
                            filtered_state[key] = value
                    
                    env_table = Table(title="Environment Summary", box=box.SIMPLE)
                    env_table.add_column("Key", style="cyan")
                    env_table.add_column("Value", style="green")
                    
                    for key, value in filtered_state.items():
                        env_table.add_row(key, str(value))
                        
                    console.print(env_table)
                    
        except Exception as e:
            import traceback
            console.print(f"[yellow]⚠ Error displaying episode metrics: {e}[/yellow]")
            console.print(traceback.format_exc())


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
