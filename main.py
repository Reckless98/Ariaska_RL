import asyncio
import os
import signal
import logging
import subprocess
import sys
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from dotenv import load_dotenv

from core.ui_helpers import (
    create_prompt_session,
    display_output,
    display_ai_hint_table,
    display_phase_tables,
    display_redagent_learning_dashboard
)
from core.multiagent.agent_manager import AgentManager
from core.logic.chainbuilder import ChainGenerator
from core.utils.stats_monitor import StatsMonitor
from core.utils.config_loader import get_config
from prompt_toolkit.formatted_text import HTML
from core.gpt_manager import GPTManager

# Setup logging
logger = logging.getLogger("ariaska")
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        logging.FileHandler("logs/ariaska.log"),
        logging.StreamHandler() if os.environ.get("DEBUG", "0") == "1" else logging.NullHandler()
    ]
)

# Initialize main components
console = Console()
session = create_prompt_session()
stats_monitor = StatsMonitor()  # StatsMonitor will track and display agent stats
gpt_manager = GPTManager()

# Create necessary directories
def setup_environment():
    """Load environment variables and create necessary directories"""
    # Load environment variables using ConfigLoader
    config = get_config()
    console.print(f"[green]✓ Configuration loaded: Live Mode = {config.is_live_mode()}[/green]")
    
    if config.is_live_mode():
        console.print(f"[bold yellow]⚠ LIVE MODE ACTIVE - Target IP: {config.get_target_ip()}[/bold yellow]")
        console.print("[bold red]❗ WARNING: Commands will be executed on real targets![/bold red]")
    
    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("core/memories", exist_ok=True)
    
    # Set up specific memory directories
    memory_dirs = [
        "core/memories/redagent_memory",
        "core/memories/blueagent_memory",
        "core/memories/shared",
        "core/memories/vectorstore"
    ]
    for directory in memory_dirs:
        os.makedirs(directory, exist_ok=True)
        
    # Check for API keys
    if not os.environ.get("OPENAI_API_KEY"):
        console.print("[yellow]⚠️ Warning: OPENAI_API_KEY not set in environment variables[/yellow]")
        console.print("[yellow]Some functionality may be limited. Consider creating a .env file from .env.example[/yellow]")

# Initialize agent manager (should happen only once)
def init_agent_manager():
    verbosity = os.environ.get("VERBOSITY", "standard")
    try:
        agent_manager = AgentManager(verbosity=verbosity)
        # Define the primary agent
        primary_agent = agent_manager.get_agent("RedAgent")
        
        if primary_agent is None:
            console.print("[red]❌ Failed to initialize RedAgent as primary agent[/red]")
            return None, None
            
        return agent_manager, primary_agent
    except Exception as e:
        console.print(f"[red]❌ Failed to initialize agent manager: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        return None, None

# ─────────────────────────────────────────────
# Banner
# ─────────────────────────────────────────────
def banner():
    banner_text = """
     █████╗ ██████╗ ██╗ █████╗ ███████╗██╗  ██╗ █████╗          
    ██╔══██╗██╔══██╗██║██╔══██╗██╔════╝██║ ██╔╝██╔══██╗         
    ███████║██████╔╝██║███████║███████╗█████╔╝ ███████║         
    ██╔══██║██╔═══╝ ██║██╔══██║╚════██║██╔═██╗ ██╔══██║         
    ██║  ██║██║     ██║██║  ██║███████║██║  ██╗██║  ██║         
    ╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝         
        ARIASKA_RL | Hybrid Offensive RL AI ⚔
    """
    # Check live mode
    config = get_config()
    mode_text = "[bold red]LIVE MODE[/bold red]" if config.is_live_mode() else "Simulation Mode"
    
    agent_stats = f"[cyan]Agents Active:[/cyan] {len(agent_manager.all_agents()) if agent_manager else 0} | [magenta]Mode:[/magenta] {mode_text}"
    panel = Panel.fit(
        banner_text + f"\n{agent_stats}", style="bold magenta", padding=(1, 2)
    )
    console.print(panel)

# ─────────────────────────────────────────────
# Command Handlers
# ─────────────────────────────────────────────
def show_hint():
    if primary_agent is None:
        console.print("[red]❌ No primary agent available.[/red]")
        return
    try:
        hint_command = primary_agent.generate_hint()
        if hint_command:
            display_ai_hint_table(
                hint_command,
                [
                    {
                        "command": hint_command,
                        "params": "Auto",
                        "why": "Rule Engine Suggestion",
                        "Full Command": hint_command,
                    }
                ],
            )
        else:
            fallback = "nmap -p- -sC -sV TARGET"
            console.print("[yellow]⚠ No hint generated. Using fallback.[/yellow]")
            display_ai_hint_table(
                fallback,
                [
                    {
                        "command": fallback,
                        "params": "-p- -sC -sV",
                        "why": "Fallback (no memory)",
                        "Full Command": fallback,
                    }
                ],
            )
    except Exception as e:
        console.print(f"[yellow]⚠ Error generating hint: {e}[/yellow]")
        fallback = "nmap -sS -sV 10.10.10.10"
        display_ai_hint_table(
            fallback,
            [
                {
                    "command": fallback,
                    "params": "-sS -sV",
                    "why": "Error recovery fallback",
                    "Full Command": fallback,
                }
            ],
        )

def run_replay_training():
    console.print("[cyan]🔁 Running Multi-Agent Replay Trainer...[/cyan]")
    if agent_manager:
        try:
            agent_manager.train_all_batches(batches=5)
            console.print("[green]✓ Replay training completed[/green]")
        except Exception as e:
            console.print(f"[red]❌ Replay training failed: {e}[/red]")
    else:
        console.print("[red]❌ Agent manager not initialized[/red]")

def run_simulated_environment_training(episodes=50):
    if not agent_manager:
        console.print("[red]❌ Agent manager not initialized[/red]")
        return
        
    console.print(
        f"[cyan]🤖 Running Multi-Agent Cyber Simulation for {episodes} episodes...[/cyan]"
    )
    
    # Clear previous visualizer instances if any
    try:
        from core.visualization.training_visualizer import TrainingVisualizer
        if hasattr(TrainingVisualizer, "_active_live_display") and TrainingVisualizer._active_live_display:
            TrainingVisualizer._active_live_display.stop()
    except Exception:
        pass
    
    # Import visualization module
    try:
        from core.visualization.training_visualizer import TrainingVisualizer
        visualizer = TrainingVisualizer(
            agents=[a.agent_id for a in agent_manager.all_agents()],
            max_history=100
        )
        visualizer.start_live_display()
    except Exception as e:
        console.print(f"[yellow]⚠ Visualization module load failed: {e}, continuing without enhanced visualization[/yellow]")
        visualizer = None
    
    try:
        # Run simulation with progress tracking
        for ep in range(episodes):
            console.print(f"[bold cyan]Episode {ep+1}/{episodes}[/bold cyan]")
            
            # Before-episode preparation and strategy visualization
            orion_agent = agent_manager.get_agent("OrionAgent")
            if orion_agent:
                strategy_insight = orion_agent.provide_reasoning("episode_start", {"episode": ep+1})
                console.print(f"[blue]👁️ Episode Strategy: [/blue]{strategy_insight}")
            
            # Run episode simulation
            agent_manager.simulate_all_agents(episodes=1, max_steps=40)
            
            # Update visualization if available
            env_state = primary_agent.env.get_global_state() if hasattr(primary_agent, "env") else None
            if visualizer and env_state:
                visualizer.update(env_state=env_state)
                
                # Save visualization snapshot every 10 episodes
                if (ep + 1) % 10 == 0:
                    visualizer.save_visualization_snapshot()
                    visualizer.create_training_report(ep + 1)
            
            # After-episode analysis
            agent_manager.display_full_status()
            
            # Periodically refresh the UI to avoid overlapping
            if (ep + 1) % 3 == 0:
                console.clear()
                banner()
            
            # Periodic model saving
            if (ep + 1) % 10 == 0:
                agent_manager.save_all_models()
                console.print(f"[green]💾 Saved models at episode {ep+1}[/green]")
        
        # Clean up visualization
        if visualizer:
            visualizer.stop_live_display()
            
        # Final models and memory saving
        agent_manager.save_all_models()
        agent_manager.snapshot_all()
        
    except Exception as e:
        console.print(f"[bold red]❌ Simulation error: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
        console.print("[yellow]⚠ Attempting to recover and display status...[/yellow]")
        try:
            # Clean up visualization if it exists
            if visualizer:
                visualizer.stop_live_display()
                
            agent_manager.display_full_status()
        except Exception as cleanup_err:
            console.print(f"[red]❌ Status display also failed: {cleanup_err}[/red]")

def show_status():
    if agent_manager:
        try:
            agent_manager.display_all_status()
        except Exception as e:
            console.print(f"[red]❌ Error displaying status: {e}[/red]")
            stats_monitor.show()  # Fallback to basic stats
    else:
        console.print("[red]❌ Agent manager not initialized[/red]")
        stats_monitor.show()  # Fallback to basic stats

def render_plots():
    try:
        stats_monitor.render_ascii_summary()
    except Exception as e:
        console.print(f"[red]❌ Error rendering plots: {e}[/red]")
        console.print("[yellow]⚠ Stats may be unavailable or corrupted[/yellow]")

def build_chain():
    if not agent_manager:
        console.print("[red]❌ Agent manager not initialized[/red]")
        return
        
    console.print("[magenta]🔗 Generating attack chains for all agents...[/magenta]")
    # Instantiate ChainBuilder, pass memory_router if available
    memory_router = getattr(agent_manager, "memory_router", None)
    
    try:
        chain_builder = ChainGenerator(memory_router=memory_router)
        # Use the multi-agent chain build method
        chain_builder.build_and_store_chain_multiagent(agent_manager)
        console.print("[green]✓ Chain generation completed[/green]")
    except Exception as e:
        console.print(f"[red]❌ Chain generation failed: {e}[/red]")

def run_distillation():
    console.print("[cyan]🧠 Running GPT distillation module...[/cyan]")
    try:
        result = subprocess.run(
            ["python3", "core/gpt_distiller.py"], 
            capture_output=True, 
            text=True,
            check=True
        )
        console.print(f"[green]✓ Distillation complete: {result.stdout}[/green]")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]❌ Distillation failed: {e}[/red]")
        console.print(e.stderr)

def repair_memory():
    console.print(
        "[blue]🧠 Repairing low-reward memory entries via GPT across agents...[/blue]"
    )
    try:
        result = subprocess.run(
            ["python3", "trainer.py", "--replay-fix"], 
            capture_output=True, 
            text=True,
            check=True
        )
        console.print(f"[green]✓ Memory repair complete: {result.stdout}[/green]")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]❌ Memory repair failed: {e}[/red]")
        console.print(e.stderr)

def display_reward_window():
    if not agent_manager:
        console.print("[red]❌ Agent manager not initialized[/red]")
        return
        
    try:
        avg_rewards = {
            agent.agent_id: agent.stats_monitor.get_avg_reward()
            for agent in agent_manager.all_agents()
            if hasattr(agent, 'stats_monitor') and callable(getattr(agent.stats_monitor, 'get_avg_reward', None))
        }
        table = Table(title="📈 Average Rewards Window")
        table.add_column("Agent", style="cyan")
        table.add_column("Avg Reward", style="green")
        for agent, avg in avg_rewards.items():
            table.add_row(agent, f"{avg:.2f}")
        console.print(table)
    except Exception as e:
        console.print(f"[red]❌ Error displaying rewards: {e}[/red]")
        stats_monitor.show()  # Fallback to basic stats

def display_phase_table():
    try:
        display_phase_tables()
    except Exception as e:
        console.print(f"[red]❌ Error displaying phase table: {e}[/red]")

def show_help():
    commands = [
        ("simulate-train [n]", "Run simulation for n episodes"),
        ("train", "Run replay training batches"),
        ("status", "Show agent statuses"),
        ("hint", "Display tactical hint"),
        ("chain", "Generate attack chains"),
        ("plot", "Render training stats"),
        ("distill", "Run GPT distillation"),
        ("repair", "Fix low-reward memory"),
        ("reward", "Show average rewards"),
        ("phases", "Display phase info"),
        ("refresh", "Clear console and refresh UI"),
        ("exit", "Exit Ariaska CLI"),
    ]
    table = Table(title="🛈 Available Commands")
    table.add_column("Command", style="cyan")
    table.add_column("Description", style="magenta")
    for cmd, desc in commands:
        table.add_row(cmd, desc)
    console.print(table)

# ─────────────────────────────────────────────
# Async CLI Loop
# ─────────────────────────────────────────────
async def main_loop():
    global agent_manager, primary_agent
    
    # Initialize environment
    setup_environment()
    
    # Initialize agent manager
    agent_manager, primary_agent = init_agent_manager()
    
    # Display banner and phase tables
    banner()
    console.print("[cyan]Initializing multi-agent modules...[/cyan]")
    try:
        display_phase_tables()
    except Exception as e:
        console.print(f"[yellow]⚠ Error displaying phase tables: {e}[/yellow]")
    
    if agent_manager and primary_agent:
        console.print("[green]✓ Ariaska RL Ready. Type 'help' for commands.[/green]\n")
    else:
        console.print("[yellow]⚠ Ariaska RL initialized with warnings. Some features may be limited.[/yellow]\n")

    # Colorized prompt with visible arrow
    prompt_text = HTML('<ansicyan>zer0</ansicyan><ansimagenta>@ARIASKA</ansimagenta><ansibright_white> > </ansibright_white>')

    while True:
        try:
            command = await session.prompt_async(prompt_text)

            if not command.strip():
                console.print("[yellow]⚠ No input. Type 'help' for options.[/yellow]")
                continue

            if isinstance(command, str):
                cmd_lower = command.strip().lower()
                args = command.strip().split()
            else:
                console.print("[red]Error: Command input is not a valid string.[/red]")
                continue

            # --- Add typo warning for common mistakes ---
            if cmd_lower.startswith("simualate-train"):
                console.print("[yellow]⚠ Did you mean 'simulate-train'? Try again with the correct spelling.[/yellow]")
                continue

            if cmd_lower in ["exit", "quit"]:
                console.print("[red]Exiting Ariaska RL. Until next battle.[/red]")
                break
            elif cmd_lower.startswith("simulate-train"):
                count = int(args[1]) if len(args) > 1 and args[1].isdigit() else 50
                run_simulated_environment_training(episodes=count)
            elif cmd_lower in ["train", "train-batch"]:
                run_replay_training()
            elif cmd_lower in ["status", "show-status"]:
                show_status()
            elif cmd_lower in ["hint", "suggest"]:
                show_hint()
            elif cmd_lower == "chain":
                build_chain()
            elif cmd_lower in ["plot", "render", "plot-stats"]:
                render_plots()
            elif cmd_lower == "distill":
                run_distillation()
            elif cmd_lower in ["repair", "replay-fix"]:
                repair_memory()
            elif cmd_lower == "reward":
                display_reward_window()
            elif cmd_lower in ["phases", "phase"]:
                display_phase_table()
            elif cmd_lower in ["refresh", "clear"]:
                console.clear()
                banner()
            elif cmd_lower in ["help", "?"]:
                show_help()
            elif cmd_lower == "test-sgpt":
                # Test if GPTManager is working correctly
                console.print("[yellow]Testing GPTManager connectivity...[/yellow]")
                try:
                    response = gpt_manager.gpt_request("Say hello from ARIASKA diagnostic.", agent_id="CLI")
                    console.print(f"[green]✓ GPTManager test successful:[/green] {response}")
                except Exception as e:
                    console.print(f"[red]❌ GPTManager test failed: {e}[/red]")
            else:
                # Live Command Execution via Primary Agent
                if primary_agent is None:
                    console.print("[red]❌ No primary agent available.[/red]")
                    continue
                try:
                    console.print(f"[cyan]Executing: {command}[/cyan]")
                    
                    # Execute with timeout protection
                    result = await asyncio.wait_for(
                        asyncio.to_thread(primary_agent.execute_command, command),
                        timeout=30.0  # 30-second timeout for command execution
                    )
                    
                    # Ensure result is a dictionary
                    if not isinstance(result, dict):
                        console.print("[red]❌ Unexpected result format[/red]")
                        continue

                    output = result.get("output", "[yellow]⚠ No output returned.[/yellow]")
                    recommendations = result.get("recommendations", [])
                    phase = result.get("phase", "unknown")
                    reward = result.get("reward", 0)
                    alert = result.get("alert", 0.0)
                    entropy = result.get("entropy", None)

                    # Make sure stats are tracked
                    if hasattr(primary_agent, "stats_monitor") and primary_agent.stats_monitor:
                        primary_agent.stats_monitor.log_step(
                            primary_agent.agent_id, reward, alert=alert, phase=phase, command=command
                        )
                        
                        if primary_agent.stats_monitor.total_steps % 10 == 0:
                            primary_agent.stats_monitor.show()

                    # Show meaningful output
                    if not output or output == "output" or output == "Error executing command: ":
                        # Generate realistic output if the agent doesn't provide one
                        if hasattr(primary_agent, "env") and hasattr(primary_agent.env, "generate_output"):
                            output = primary_agent.env.generate_output(command)
                        else:
                            output = f"Executed: {command}"
                    
                    # Display output
                    display_output(output)
                    
                    # Only display recommendations if they exist
                    if recommendations and len(recommendations) > 0:
                        display_ai_hint_table(None, recommendations)
                    
                    # Update state visualization
                    if hasattr(primary_agent, "env") and hasattr(primary_agent.env, "_visualize_environment_state"):
                        primary_agent.env._visualize_environment_state()
                    
                except asyncio.TimeoutError:
                    console.print("[red]❌ Command execution timed out after 30 seconds[/red]")
                except Exception as e:
                    console.print(f"[red]❌ Error executing command: {e}[/red]")
                    import traceback
                    console.print(traceback.format_exc())

        except (KeyboardInterrupt, EOFError):
            console.print("\n[red]⚠ Interrupted. Shutting down Ariaska RL.[/red]")
            # Graceful shutdown: stop visualization, save models, log
            try:
                from core.visualization.training_visualizer import TrainingVisualizer
                if hasattr(TrainingVisualizer, "_active_live_display") and TrainingVisualizer._active_live_display:
                    TrainingVisualizer._active_live_display.stop()
            except Exception:
                pass
            try:
                if agent_manager:
                    agent_manager.save_all_models()
                    agent_manager.snapshot_all()
                    console.print("[green]💾 Models and memory saved on shutdown.[/green]")
            except Exception as e:
                console.print(f"[yellow]⚠ Failed to save models/memory: {e}[/yellow]")
            break
        except Exception as e:
            console.print(f"[bold red]❌ Runtime Error: {e}[/bold red]")
            import traceback
            console.print(traceback.format_exc())  # Print the full traceback for debugging
            
def graceful_shutdown(*args):
    logger.warning("SIGINT received. Initiating safe shutdown...")
    try:
        if 'agent_manager' in globals() and agent_manager:
            agent_manager.save_all_models()
            agent_manager.snapshot_all()
        
        # Stop any active visualizers
        try:
            from core.visualization.training_visualizer import DisplayManager
            DisplayManager.get_instance().stop()
        except Exception:
            pass
    except Exception as e:
        logger.error(f"Shutdown error: {e}")
    exit(0)

signal.signal(signal.SIGINT, graceful_shutdown)

# ─────────────────────────────────────────────
# Launch
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Define global variables
    agent_manager = None
    primary_agent = None
    
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt. Initiating safe shutdown...")
        graceful_shutdown()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
