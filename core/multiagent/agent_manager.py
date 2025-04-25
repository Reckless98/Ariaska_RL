# core/multiagent/agent_manager.py — ARIASKA Agent Orchestrator v11.5 APEX PRIME

from rich.console import Console
from rich.panel import Panel
import subprocess
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, SpinnerColumn
from rich.table import Table
from rich import box  # <-- Add this import

console = Console()


class AgentManager:
    def __init__(self, verbosity="standard"):
        console.rule("[bold cyan]🚀 ARIASKA Multi-Agent Deployment: System Initialization[/bold cyan]")
        # --- Grouped Initialization Logs ---
        agent_init_table = Table(title="Multi-Agent Deployment", box=box.ROUNDED)  # <-- fix here
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
        
        # Defer agent creation to avoid circular references
        self.red_agent = None
        self.blue_agent = None
        self.scout_agent = None
        self.shadow_agent = None
        self.orion_agent = None
        self.agents = []
        
        # Create memory router first
        self.memory_router = self._import_memory_router()([])
        
        # Now create all agents
        self._initialize_agents()
        
        # Collect agent status for dashboard
        for agent in self.agents:
            mode = getattr(agent, "current_mode", getattr(agent, "role", "N/A"))
            agent_init_table.add_row(
                getattr(agent, "agent_id", "Unknown"),
                str(mode),
                "[green]✔ Ready[/green]"
            )
        
        # Auto-Sync GPT Strategy
        gpt_context_synced = self._initialize_gpt_context()

        # Print summary panel after all initialization
        self._print_startup_dashboard(agent_init_table, gpt_context_synced)

        console.print(
            "[green]✔ Agents Ready | MemoryRouter Active | GPT Context Synced[/green]"
        )

    def _initialize_agents(self):
        """Initialize agents in the correct order to avoid circular references"""
        # First, create the agents
        agents = self._get_all_agents(agent_manager=self)
        
        # Store individual agent references
        self.red_agent = agents["RedAgent"]
        self.blue_agent = agents["BlueAgent"]
        self.scout_agent = agents["ScoutAgent"]
        self.shadow_agent = agents["ShadowAgent"]
        self.orion_agent = agents["OrionAgent"]
        
        # Update the agents list
        self.agents = [
            self.red_agent,
            self.blue_agent,
            self.scout_agent,
            self.shadow_agent,
            self.orion_agent,
        ]
        
        # Update memory router with agents
        self.memory_router.agents = self.agents
        
        # Assign memory_router to each agent (if they have the attribute)
        for agent in self.agents:
            if hasattr(agent, "memory_router"):
                agent.memory_router = self.memory_router
            if hasattr(agent, "agent_manager"):
                agent.agent_manager = self
                
        # Initialize multi-agent links for agents that have the method
        for agent in self.agents:
            if hasattr(agent, "_init_multiagent_links"):
                agent._init_multiagent_links()
                
        # Pass verbosity to agents
        for agent in self.agents:
            if hasattr(agent, "verbosity"):
                agent.verbosity = self.verbosity
                
        # Initialize dynamic parameters for each agent's environment if present
        for agent in self.agents:
            env = getattr(agent, "env", None)
            if env and hasattr(env, "initialize_dynamic_parameters"):
                env.initialize_dynamic_parameters()
        
        # After all agents are created and registered, reset their environments
        for agent in self.agents:
            if hasattr(agent, "env") and hasattr(agent.env, "reset_environment"):
                agent.env.reset_environment()

    def get_agent(self, agent_id):
        """
        Retrieve an agent by its agent_id string.
        """
        for agent in self.agents:
            if getattr(agent, "agent_id", None) == agent_id:
                return agent
        raise ValueError(f"Agent '{agent_id}' not found in AgentManager.")

    def get_memory_router(self):
        return self.memory_router

    def _import_stats_monitor(self):
        # Import locally to avoid circular import
        from core.monitor.stats_monitor import StatsMonitor
        return StatsMonitor

    def _import_memory_router(self):
        # Import locally to avoid circular import
        from core.multiagent.memory_router import MemoryRouter
        return MemoryRouter

    def _get_all_agents(self, agent_manager=None):
        # Import locally to avoid circular import
        from core.multiagent.agents import get_all_agents
        return get_all_agents(agent_manager=agent_manager)

    def _import_summarize_rule_stats(self):
        # Import locally to avoid circular import
        from core.logic.rule_engine import summarize_rule_stats
        return summarize_rule_stats

    # ─────────────────────────────────────────────
    # 🎮 Orchestration Controls
    # ─────────────────────────────────────────────
    def all_agents(self):
        return self.agents

    def broadcast(self, key, value, sender=None):
        """Agents can broadcast key-value pairs to the shared context."""
        self.shared_context[key] = value
        if sender:
            console.print(f"[cyan]📡 {sender} broadcasted: {key} = {value}[/cyan]")

    def query_context(self, key, default=None):
        """Agents can query the shared context."""
        return self.shared_context.get(key, default)

    def simulate_all_agents(self, episodes=10, max_steps=40):
        try:
            console.rule("[bold magenta]🎮 Multi-Agent Simulation Mode Engaged")
            agents = self.agents
            
            # Reset global monitoring state to avoid duplicate progress bars
            from rich.live import Live
            global_live = None
            
            # Import visualization module
            try:
                from core.visualization.training_visualizer import TrainingVisualizer
                visualizer = TrainingVisualizer(
                    agents=[a.agent_id for a in self.agents],
                    max_history=100
                )
                visualizer.start_live_display()
            except Exception as e:
                console.print(f"[yellow]⚠ Visualization module load failed: {e}, continuing without enhanced visualization[/yellow]")
                visualizer = None
            
            for ep in range(episodes):
                console.print(f"[bright_white]--- Episode {ep+1}/{episodes} ---[/bright_white]")
                
                # Reset all agent stats for this episode
                for agent in agents:
                    if hasattr(agent, "reset"):
                        agent.reset()
                    elif hasattr(agent, "command_history"):
                        agent.command_history.clear()
                    if hasattr(agent, "stats_monitor"):
                        agent.stats_monitor.reset()
                        
                # Use a single progress bar for all agents/steps in this episode
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]{task.description}"),
                    BarColumn(bar_width=30),
                    TextColumn("[cyan]{task.completed}/{task.total}"),
                    TimeElapsedColumn(),
                    console=console,
                    transient=True,
                ) as progress:
                    total_steps = max_steps * len([a for a in agents if hasattr(a, "simulate_step")])
                    task = progress.add_task(f"Episode {ep+1} Progress", total=total_steps)
                    
                    # First, perform ScoutAgent phase advice for coordination
                    scout = self.scout_agent
                    state = self.red_agent.env.get_global_state()
                    if scout:
                        try:
                            # Fixed: Pass all_agents parameter to advise_phase
                            phase = scout.advise_phase(state, self.all_agents())
                            console.print(f"[cyan]🧭 Episode {ep+1} starting phase: {phase}[/cyan]")
                        except Exception as e:
                            console.print(f"[yellow]⚠ ScoutAgent initial phase advice failed: {e}, defaulting to 'recon'[/yellow]")
                            phase = "recon"
                    
                    # Then run all other agents, with improved error handling
                    for step in range(max_steps):
                        try:
                            # New: Periodic sync step for agent coordination
                            if step % 10 == 0:
                                self._multiagent_sync()
                            for agent in agents:
                                if agent.agent_id == "ScoutAgent":
                                    # Skip Scout as it's already run for coordination
                                    continue
                                    
                                if hasattr(agent, "simulate_step"):
                                    # Use the proper simulate_step method
                                    step_info = self._simulate_agent_step(agent, ep+1, step+1, shared_context=self.shared_context)
                                    self._log_agent_step(agent, ep+1, step+1, step_info)
                                    if visualizer:
                                        visualizer.update(agent_data=step_info)
                                    progress.update(task, advance=1)
                                elif hasattr(agent, "simulate_train"):
                                    # For agents that don't have simulate_step
                                    agent.simulate_train(episodes=1, max_steps=1, log_every=1, use_progress_bar=False)
                        except Exception as e:
                            console.print(f"[bold red]❌ Error in agent execution: {e}[/bold red]")
                            import traceback
                            console.print(traceback.format_exc())
                        
                        # Save models/snapshots every 15 steps or at episode end
                        if (step + 1) % 15 == 0 or (step + 1) == max_steps:
                            if self.verbosity != "quiet":
                                console.print(f"[green]💾 Saving models and snapshots at step {step+1}[/green]")
                            self.save_all_models()
                            self.snapshot_all()
                    
                    # Throttle model saves and snapshots
                    if (ep + 1) % 5 == 0 or (ep + 1) == episodes:
                        if self.verbosity != "quiet":
                            console.print(f"[green]💾 Saving models and snapshots at episode {ep+1}[/green]")
                        self.save_all_models()
                        self.snapshot_all()
                    
                    # Print summary table after each episode
                    self._log_multiagent_episode(ep+1)
                
            self._post_simulation_sync()
            if visualizer:
                visualizer.stop_live_display()
        except Exception as e:
            console.print(f"[red]❌ Error in simulate_all_agents: {e}[/red]")
            import traceback
            console.print(traceback.format_exc())

    def _simulate_agent_step(self, agent, episode, step, shared_context=None):
        info = {
            "command": "N/A", 
            "phase": "N/A",
            "reward": 0.0,
            "gpt_calls": 0,
            "output": "N/A"
        }
        try:
            if hasattr(agent, "simulate_step"):
                info = agent.simulate_step(episode=episode, step=step, shared_context=shared_context or self.shared_context)
                if agent.agent_id == "RedAgent" and hasattr(agent, "env"):
                    agent.env.step(info["command"])
            else:
                info = {
                    "command": getattr(agent, "command_history", ["N/A"])[-1] if getattr(agent, "command_history", None) else "N/A",
                    "phase": getattr(agent, "current_mode", "N/A"),
                    "reward": getattr(agent.stats_monitor, "get_average_reward", lambda: 0.0)() if hasattr(agent, "stats_monitor") else 0.0,
                    "gpt_calls": getattr(agent.stats_monitor, "agent_stats", {}).get(agent.agent_id, {}).get("gpt_calls", 0) if hasattr(agent, "stats_monitor") else 0,
                    "output": getattr(agent, "last_output", "N/A")
                }
        except Exception as e:
            console.print(f"[red]⚠ Error in {agent.agent_id} step: {e}[/red]")
            info = {
                "command": "N/A",
                "phase": "N/A",
                "reward": 0.0,
                "gpt_calls": 0,
                "output": "N/A"
            }
        return info

    def _log_agent_step(self, agent, episode, step, info):
        # Enhanced visualization with detailed information
        table = Table(title=f"[bold cyan]{agent.agent_id} Step {step} (Episode {episode})", show_lines=True)
        table.add_column("Field", style="magenta")
        table.add_column("Value", style="green")
        
        # Basic command info with enhanced formatting
        command = info.get("command", "N/A")
        if isinstance(command, dict) and "response" in command:
            command = command["response"]
        table.add_row("Command", f"[bold cyan]{command}[/bold cyan]")
        
        # Phase info with color coding
        phase = info.get("phase", "N/A")
        if isinstance(phase, dict) and "response" in phase:
            phase = phase["response"]
        phase_colors = {
            "recon": "[blue]recon[/blue]",
            "enumeration": "[cyan]enumeration[/cyan]",
            "exploit": "[yellow]exploit[/yellow]",
            "privesc": "[orange]privesc[/orange]",
            "exfiltrate": "[bold red]exfiltrate[/bold red]",
            "unknown": "[dim]unknown[/dim]"
        }
        table.add_row("Phase", phase_colors.get(str(phase).lower(), str(phase)))
        
        # Reward with color coding based on value
        reward = info.get("reward", 0)
        if reward > 20:
            reward_str = f"[bold green]{reward:.2f}[/bold green]"
        elif reward > 0:
            reward_str = f"[green]{reward:.2f}[/green]"
        elif reward < -10:
            reward_str = f"[bold red]{reward:.2f}[/bold red]"
        elif reward < 0:
            reward_str = f"[red]{reward:.2f}[/red]"
        else:
            reward_str = f"[yellow]{reward:.2f}[/yellow]"
        table.add_row("Reward", reward_str)
        
        # Additional stats
        table.add_row("GPT Calls", str(info.get("gpt_calls", 0)))
        
        # Command output with syntax highlighting
        output = str(info.get("output", "N/A"))
        if output and len(output) > 120:
            output = output[:117] + "..."
            
        # Add basic syntax highlighting
        if "error" in output.lower() or "failed" in output.lower():
            output = f"[red]{output}[/red]"
        elif "success" in output.lower() or "completed" in output.lower():
            output = f"[green]{output}[/green]"
            
        table.add_row("Output", output)
        
        # Add reasoning if available
        reasoning = info.get("reasoning", "")
        if reasoning:
            table.add_row("Reasoning", str(reasoning)[:120])
        
        # Environment state if available
        if "environment" in info:
            env_data = info.get("environment", {})
            env_str = ", ".join(f"{k}={v}" for k, v in env_data.items())
            table.add_row("Environment", env_str)
        
        # Add more agent info
        table.add_row("Replay Buffer", str(info.get("replay_buffer", "-")))
        table.add_row("Epsilon", f"{info.get('epsilon', 0):.3f}")
        table.add_row("Entropy", f"{info.get('entropy_beta', 0):.3f}")
        
        console.print(table)

    def _log_multiagent_episode(self, episode_num):
        """Enhanced episode summary with more detailed visualization"""
        from rich.table import Table
        from rich.panel import Panel
        from rich.columns import Columns
        
        # Create a table for each major agent type
        tables = []
        
        # Group agents by type
        agent_groups = {
            "Offensive": [a for a in self.agents if "Red" in a.agent_id],
            "Defensive": [a for a in self.agents if "Blue" in a.agent_id],
            "Intelligence": [a for a in self.agents if a.agent_id in ["ScoutAgent", "ShadowAgent", "OrionAgent"]]
        }
        
        for group_name, agents in agent_groups.items():
            if not agents:
                continue
                
            table = Table(title=f"🧠 {group_name} Agents - Episode {episode_num}", show_lines=True)
            table.add_column("Agent", style="cyan")
            table.add_column("Last Action", style="magenta")
            table.add_column("Phase", style="yellow")
            table.add_column("Reward", style="green")
            table.add_column("GPT Call", style="blue")
            
            for agent in agents:
                last_action = getattr(agent, "command_history", ["N/A"])[-1] if hasattr(agent, "command_history") and agent.command_history else "N/A"
                phase = getattr(agent, "current_mode", "N/A")
                
                # Get reward stats
                reward = getattr(agent, "stats_monitor", None)
                reward_val = reward.get_average_reward() if reward else "N/A"
                
                # Format reward color
                if isinstance(reward_val, (int, float)):
                    if reward_val > 20:
                        reward_str = f"[bold green]{reward_val:.2f}[/bold green]"
                    elif reward_val > 0:
                        reward_str = f"[green]{reward_val:.2f}[/green]"
                    elif reward_val < 0:
                        reward_str = f"[red]{reward_val:.2f}[/red]"
                    else:
                        reward_str = f"[yellow]{reward_val:.2f}[/yellow]"
                else:
                    reward_str = str(reward_val)
                
                gpt = getattr(agent, "last_reasoning", "N/A")
                gpt_str = str(gpt)[:40] + "..." if len(str(gpt)) > 40 else str(gpt)
                
                table.add_row(
                    agent.agent_id, 
                    str(last_action), 
                    str(phase), 
                    reward_str,
                    gpt_str
                )
                
            tables.append(Panel(table))
        
        console.print(Columns(tables))
        
        # Add environment snapshot after the agent tables
        if hasattr(self.red_agent, "env") and self.red_agent.env:
            env_state = self.red_agent.env.get_global_state()
            env_table = Table(title="🌍 Environment State")
            env_table.add_column("Property", style="cyan")
            env_table.add_column("Value", style="green")
            
            # Add key environment variables
            for key, value in env_state.items():
                if isinstance(value, list) and len(value) > 5:
                    value_str = str(value[:5])[:-1] + ", ...]"
                else:
                    value_str = str(value)
                    
                env_table.add_row(key, value_str)
                
            console.print(Panel(env_table))

    def simulate_all(self, episodes=10):
        # Alias for CLI compatibility
        self.simulate_all_agents(episodes=episodes)

    # ─────────────────────────────────────────────
    # 📊 Post-Simulation Sync & Review
    # ─────────────────────────────────────────────
    def _post_simulation_sync(self):
        console.rule("[bold cyan]♻️ Post-Simulation: Memory Sync & Orion Review")
        self.sync_all_memories()
        self.trigger_orion_review()

    def batch_train_all(self, batches=5):
        console.rule("[bold cyan]📈 Batch Training Across All Agents")
        for agent in self.agents:
            if hasattr(agent, "train_on_batch"):
                console.print(
                    f"[cyan]{agent.agent_id}: Training {batches} batches[/cyan]"
                )
                for _ in range(batches):
                    agent.train_on_batch()
        self._post_training_analysis()

    def _post_training_analysis(self):
        console.rule("[bold magenta]👁 Orion: Strategic Post-Training Analysis")
        # Import summarize_rule_stats locally to avoid circular import
        summarize_rule_stats = self._import_summarize_rule_stats()
        self.orion_agent.analyze_training(self.agents)
        for agent in self.agents:
            summarize_rule_stats(agent)

    # ─────────────────────────────────────────────
    # 💾 Model & Memory Management
    # ─────────────────────────────────────────────
    def save_all_models(self):
        console.rule("[bold yellow]💾 Saving Models & Critical States")
        # Only save at logical checkpoints (end of episode or batch)
        for agent in self.agents:
            if hasattr(agent, "save_models"):
                if hasattr(agent, "total_steps") and hasattr(agent, "total_episodes"):
                    # Only save if at end of episode (simulate_all_agents controls this)
                    if getattr(agent, "_is_end_of_episode", False):
                        agent.save_models(prefix=f"models/{agent.agent_id}")
                else:
                    agent.save_models(prefix=f"models/{agent.agent_id}")

    def snapshot_all(self):
        console.rule("[bold magenta]📸 Creating Global Memory Snapshots")
        # Only snapshot at logical checkpoints (end of episode or batch)
        for agent in self.agents:
            if hasattr(agent, "total_steps") and hasattr(agent, "total_episodes"):
                if getattr(agent, "_is_end_of_episode", False):
                    self.memory_router.snapshot_all_memories()
            else:
                self.memory_router.snapshot_all_memories()

    # ─────────────────────────────────────────────
    # ♻️ Dynamic Memory & GPT Intelligence Sync
    # ─────────────────────────────────────────────
    def sync_all_memories(self):
        console.print("[cyan]🔗 Syncing Global Insights & Optimizing GPT Cache[/cyan]")
        self.memory_router.sync_global_insights()
        self.memory_router.consolidate_gpt_cache()
        self.memory_router.optimize_memories()

    # ─────────────────────────────────────────────
    # 👁 Orion Oversight & Auto-Adjustments
    # ─────────────────────────────────────────────
    def trigger_orion_review(self):
        console.print("[bold blue]👁 Initiating OrionAgent Strategic Review[/bold blue]")
        self.orion_agent.apply_orion_strategic_adjustments(self.agents)

    # ─────────────────────────────────────────────
    # 🚨 Agent Health Diagnostics
    # ─────────────────────────────────────────────
    def display_full_status(self):
        console.rule("[bold green]🧩 Multi-Agent Status Dashboard")
        for agent in self.agents:
            if hasattr(agent, "display_advanced_status"):
                agent.display_advanced_status()
        
        # Visualize agent diversity and redundancy metrics
        for agent in self.agents:
            if hasattr(agent, "stats_monitor"):
                stats = agent.stats_monitor
                diversity = len(set(agent.command_history[-20:])) if hasattr(agent, "command_history") else 0
                redundancy = max(agent.repetition_count.values()) if hasattr(agent, "repetition_count") and agent.repetition_count else 0
                console.print(f"[blue]{agent.agent_id} Diversity (last 20): {diversity} | Max Redundancy: {redundancy}[/blue]")

    # ─────────────────────────────────────────────
    # 🔧 Maintenance Utilities
    # ─────────────────────────────────────────────
    def reset_all_agents(self):
        console.print("[yellow]🔄 Resetting All Agents to Baseline States...[/yellow]")
        for agent in self.agents:
            if hasattr(agent, "reset"):
                agent.reset()
        self._initialize_gpt_context()

    def _initialize_gpt_context(self):
        """Prime GPT with current agent configuration and mission context using GPTManager."""
        from core.gpt_manager import GPTManager
        prompt = """
You are ARIASKA's core strategist AI. 
Agents initialized: Red (Offense), Blue (Defense), Scout (Navigator), Shadow (Optimizer), Orion (Overseer).
Ensure optimal synergy, minimize redundancy, and align strategies across offensive and defensive operations.
Acknowledge with 'Strategic Context Loaded'.
"""
        try:
            gpt_manager = GPTManager()
            response = gpt_manager.gpt_request(prompt, task_type="reasoning")
            self.shared_context['gpt_context'] = response
            if isinstance(response, str) and "Strategic Context Loaded" in response:
                console.print("[green]✔ GPT Context Synced[/green]")
                return True
            else:
                console.print(f"[yellow]⚠ GPT context sync incomplete: {response}[/yellow]")
                return False
        except Exception as e:
            console.print(f"[yellow]⚠ GPT context sync failed: {e}[/yellow]")
            return False

    def _print_startup_dashboard(self, agent_init_table, gpt_context_synced):
        from rich.panel import Panel
        from rich.table import Table
        from rich.columns import Columns
        from rich import box  # <-- ensure import here if needed
        # Agents summary
        agents_summary = agent_init_table
        # GPT Models summary
        gpt_table = Table(title="GPT Models Ready", box=box.ROUNDED)  # <-- fix here
        gpt_table.add_column("Model", style="cyan")
        gpt_table.add_column("Purpose", style="magenta")
        gpt_table.add_row("GPT-4.1", "Primary Reasoning")
        gpt_table.add_row("GPT-4o-mini", "Fallback/Lightweight")
        gpt_table.add_row("GPT-4.1-nano", "Embeddings/Light Tasks")
        # Environment summary
        env_table = Table(title="Environment", box=box.ROUNDED)  # <-- fix here
        env_mode = "Simulated"
        try:
            env = self.red_agent.env if hasattr(self.red_agent, "env") else None
            if env and hasattr(env, "training_mode"):
                env_mode = getattr(env, "training_mode", "Simulated")
        except Exception:
            pass
        env_table.add_column("Mode", style="cyan")
        env_table.add_column("Status", style="green")
        env_table.add_row(env_mode, "[green]Ready[/green]")
        # Memory summary
        mem_table = Table(title="Memory Status", box=box.ROUNDED)  # <-- fix here
        mem_table.add_column("Component", style="cyan")
        mem_table.add_column("Status", style="green")
        mem_table.add_row("Replay Buffers", "Initialized")
        mem_table.add_row("GPT Cache", "Active")
        # Curriculum/difficulty
        curriculum_table = Table(title="Curriculum", box=box.ROUNDED)  # <-- fix here
        curriculum_table.add_column("Phase", style="cyan")
        curriculum_table.add_column("Difficulty", style="magenta")
        try:
            diff = getattr(env, "difficulty_level", 1) if env else 1
            phase = getattr(env, "current_phase", "N/A") if env else "N/A"
        except Exception:
            diff = 1
            phase = "N/A"
        curriculum_table.add_row(str(phase), str(diff))
        # Compose dashboard
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
