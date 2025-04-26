import asyncio
import os
import signal
import logging
import subprocess
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from core.ui_helpers import (
    create_prompt_session,
    display_output,
    display_ai_hint_table,
    display_phase_tables,
    display_redagent_learning_dashboard
)
from core.multiagent.agent_manager import AgentManager  # Updated import
from core.logic.chainbuilder import ChainGenerator  # Fixed import
from core.utils.stats_monitor import StatsMonitor
from prompt_toolkit.formatted_text import HTML
from core.gpt_manager import GPTManager

logger = logging.getLogger("ariaska")
logging.basicConfig(level=logging.INFO, filename="logs/ariaska.log", filemode="a")

console = Console()
session = create_prompt_session()
stats_monitor = StatsMonitor()  # StatsMonitor will track and display agent stats
agent_manager = AgentManager()  # Use new centralized manager
gpt_manager = GPTManager()

# Define the primary agent using AgentManager
primary_agent = agent_manager.get_agent("RedAgent")  # Use .get_agent()

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
    agent_stats = f"[cyan]Agents Active:[/cyan] {len(agent_manager.all_agents())} | [magenta]Mode:[/magenta] Multi-Agent Ops"
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

def run_replay_training():
    console.print("[cyan]🔁 Running Multi-Agent Replay Trainer...[/cyan]")
    agent_manager.train_all_batches(batches=5)

def run_simulated_environment_training(episodes=50):
    console.print(
        f"[cyan]🤖 Running Multi-Agent Cyber Simulation for {episodes} episodes...[/cyan]"
    )
    
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
        except:
            console.print("[red]❌ Status display also failed.[/red]")

def show_status():
    agent_manager.display_all_status()

def render_plots():
    stats_monitor.render_ascii_summary()

def build_chain():
    console.print("[magenta]🔗 Generating attack chains for all agents...[/magenta]")
    # Instantiate ChainBuilder, pass memory_router if available
    memory_router = getattr(agent_manager, "memory_router", None)
    chain_builder = ChainGenerator(memory_router=memory_router)
    # Use the multi-agent chain build method
    chain_builder.build_and_store_chain_multiagent(agent_manager)

def run_distillation():
    console.print("[cyan]🧠 Running GPT distillation module...[/cyan]")
    os.system("python3 core/gpt_distiller.py")

def repair_memory():
    console.print(
        "[blue]🧠 Repairing low-reward memory entries via GPT across agents...[/blue]"
    )
    os.system("python3 trainer.py --replay-fix")

def display_reward_window():
    avg_rewards = {
        agent.agent_id: agent.stats_monitor.get_avg_reward()
        for agent in agent_manager.all_agents()
    }
    table = Table(title="📈 Average Rewards Window")
    table.add_column("Agent", style="cyan")
    table.add_column("Avg Reward", style="green")
    for agent, avg in avg_rewards.items():
        table.add_row(agent, f"{avg:.2f}")
    console.print(table)

def display_phase_table():
    display_phase_tables()

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
    banner()
    console.print("[cyan]Initializing multi-agent modules...[/cyan]")
    display_phase_tables()
    console.print("[green]✔ Ariaska RL Ready. Type 'help' for commands.[/green]\n")

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
            elif cmd_lower in ["help", "?"]:
                show_help()
            elif cmd_lower == "test-sgpt":
                # Test if GPTManager is working correctly
                console.print("[yellow]Testing GPTManager connectivity...[/yellow]")
                try:
                    response = gpt_manager.gpt_request("Say hello from ARIASKA diagnostic.", agent_id="CLI")
                    console.print(f"[green]✔ GPTManager test successful:[/green] {response}")
                except Exception as e:
                    console.print(f"[red]❌ GPTManager test failed: {e}[/red]")
            else:
                # Live Command Execution via Primary Agent
                if primary_agent is None:
                    console.print("[red]❌ No primary agent available.[/red]")
                    continue
                try:
                    console.print(f"[cyan]Executing: {command}[/cyan]")
                    result = primary_agent.execute_command(command)
                    
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
                    primary_agent.stats_monitor.log_step(
                        primary_agent.agent_id, reward, alert=alert, phase=phase, command=command
                    )

                    if primary_agent.stats_monitor.total_steps % 10 == 0:
                        primary_agent.stats_monitor.show()

                    # Show meaningful output
                    if not output or output == "output":
                        # Generate realistic output if the agent doesn't provide one
                        if hasattr(primary_agent, "env") and hasattr(primary_agent.env, "generate_output"):
                            output = primary_agent.env.generate_output(command)
                        else:
                            output = f"Executed: {command}"
                    
                    display_output(output)
                    display_ai_hint_table(None, recommendations)
                    
                    # Update state visualization
                    if hasattr(primary_agent, "env"):
                        primary_agent.env._visualize_environment_state()
                    
                    show_hint()
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
        agent_manager.save_all_models()
        agent_manager.snapshot_all()
        from core.visualization.training_visualizer import DisplayManager
        DisplayManager.get_instance().stop()
    except Exception as e:
        logger.error(f"Shutdown error: {e}")
    exit(0)

signal.signal(signal.SIGINT, graceful_shutdown)

# ─────────────────────────────────────────────
# Launch
# ─────────────────────────────────────────────
if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt. Initiating safe shutdown...")
        graceful_shutdown()
    except Exception as e:
        logger.error(f"Fatal error: {e}")

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# main.py — ARIASKA_RL Main Entry Point v2.1 APEX
# 🚀 Multi-Agent Cybersecurity Platform | 🧠 Training Controller | 🌐 Deployment Interface

import os
import argparse
import logging
import sys
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich import box
from dotenv import load_dotenv

console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("ariaska")

def setup_environment():
    """Load environment variables and create necessary directories"""
    # Load environment variables
    load_dotenv()
    
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

def display_welcome_banner():
    """Display a welcome banner with system information"""
    console.rule("[bold cyan]🧠 ARIASKA_RL MULTI-AGENT CYBERSECURITY PLATFORM[/bold cyan]")
    
    # Create feature table
    features = Table(title="System Components", box=box.ROUNDED)
    features.add_column("Component", style="cyan")
    features.add_column("Status", style="green")
    features.add_column("Mode", style="yellow")
    
    # Check for component availability
    try:
        import torch
        torch_status = f"[green]Available[/green] (CUDA: {torch.cuda.is_available()})"
    except ImportError:
        torch_status = "[red]Not installed[/red]"
        
    try:
        import openai
        openai_status = "[green]Available[/green]"
    except ImportError:
        openai_status = "[red]Not installed[/red]"
        
    try:
        import chromadb  # type: ignore
        chromadb_status = "[green]Available[/green]"
    except ImportError:
        chromadb_status = "[red]Not installed[/red]"
    
    # Get environment mode
    env_mode = os.environ.get("ARIASKA_MODE", "simulated")
    
    # Build status table
    features.add_row("PyTorch", torch_status, "")
    features.add_row("OpenAI API", openai_status, "")
    features.add_row("ChromaDB", chromadb_status, "")
    features.add_row("Environment", "[green]Ready[/green]", f"[yellow]{env_mode.upper()}[/yellow]")
    features.add_row("GPT Models", "[green]Ready[/green]", os.environ.get("PRIMARY_MODEL", "gpt-4o-mini"))
    
    # Display system info
    info_table = Table(title="System Information", box=box.ROUNDED)
    info_table.add_column("Setting", style="cyan")
    info_table.add_column("Value", style="green")
    
    info_table.add_row("Version", "2.1 APEX")
    info_table.add_row("Mode", env_mode.upper())
    info_table.add_row("Verbosity", os.environ.get("VERBOSITY", "standard"))
    info_table.add_row("Max Episodes", os.environ.get("EPISODES", "100"))
    
    # Display both tables
    console.print(Panel.fit(features))
    console.print(Panel.fit(info_table))
    
    console.rule("[bold green]System Initialized - Ready for Operation[/bold green]")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="ARIASKA_RL - Multi-Agent Cybersecurity Training Platform"
    )
    
    parser.add_argument(
        "--train",
        action="store_true",
        help="Start training mode with all agents"
    )
    
    parser.add_argument(
        "--episodes",
        type=int,
        default=int(os.environ.get("EPISODES", "100")),
        help="Number of episodes for training"
    )
    
    parser.add_argument(
        "--steps",
        type=int,
        default=int(os.environ.get("MAX_STEPS", "50")),
        help="Maximum steps per episode"
    )
    
    parser.add_argument(
        "--mode",
        choices=["simulated", "live"],
        default=os.environ.get("ARIASKA_MODE", "simulated"),
        help="Environment mode: simulated or live"
    )
    
    parser.add_argument(
        "--verbosity",
        choices=["quiet", "standard", "verbose", "detailed"],
        default=os.environ.get("VERBOSITY", "standard"),
        help="Output verbosity level"
    )
    
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Enable visualization UI (if available)"
    )
    
    parser.add_argument(
        "--test-env",
        action="store_true",
        help="Test environment setup without starting training"
    )
    
    return parser.parse_args()

def run_training(args):
    """Run the training loop with agent manager"""
    try:
        from core.multiagent.agent_manager import AgentManager
        
        # Initialize agent manager with verbosity setting
        console.print("[cyan]Initializing AgentManager and multi-agent system...[/cyan]")
        agent_manager = AgentManager(verbosity=args.verbosity)
        
        # Start training with specified episodes and steps
        console.print(f"[green]Starting training with {args.episodes} episodes, {args.steps} max steps per episode[/green]")
        agent_manager.simulate_all_agents(episodes=args.episodes, max_steps=args.steps)
        
        # Batch train all agents after simulation
        console.print("[cyan]Running batch training for all agents...[/cyan]")
        agent_manager.batch_train_all(batches=5)
        
        # Save all models
        console.print("[cyan]Saving trained models and memory snapshots...[/cyan]")
        agent_manager.save_all_models()
        agent_manager.snapshot_all()
        
        # Display final status
        agent_manager.display_full_status()
        
        console.print("[bold green]✓ Training completed successfully![/bold green]")
        return True
        
    except ImportError as e:
        console.print(f"[bold red]Error loading required modules: {e}[/bold red]")
        console.print("[yellow]Please ensure all requirements are installed: pip install -r requirements.txt[/yellow]")
        return False
    except Exception as e:
        console.print(f"[bold red]Error during training: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
        return False

def test_environment():
    """Test the environment setup without full training"""
    try:
        from core.environment.cyber_environment import CyberEnvironment
        from core.environment.environment_context_detector import EnvironmentContextDetector
        
        # Initialize environment
        console.print("[cyan]Initializing CyberEnvironment for testing...[/cyan]")
        env = CyberEnvironment(defer_reset=False)
        
        # Initialize context detector
        console.print("[cyan]Initializing EnvironmentContextDetector...[/cyan]")
        context_detector = EnvironmentContextDetector()
        
        # Get environment state
        state = env.get_global_state()
        
        # Display environment info
        env_table = Table(title="Environment Test", show_header=True)
        env_table.add_column("Property", style="cyan")
        env_table.add_column("Value", style="green")
        
        for key, value in state.items():
            # Limit list length for display
            if isinstance(value, list) and len(value) > 5:
                value_str = str(value[:5])[:-1] + ", ...]"
            else:
                value_str = str(value)
            env_table.add_row(key, value_str)
            
        console.print(env_table)
        
        # Test context detector
        context = context_detector.get_environment_context("10.10.10.10")
        
        # Test randomization
        randomized = context_detector.randomize_domain()
        
        console.print("[bold green]✓ Environment tests completed successfully![/bold green]")
        return True
        
    except ImportError as e:
        console.print(f"[bold red]Error loading required modules: {e}[/bold red]")
        return False
    except Exception as e:
        console.print(f"[bold red]Error testing environment: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
        return False

def launch_visualization():
    """Launch visualization dashboard if streamlit is installed"""
    try:
        # Try to import streamlit but handle the case where it's not installed
        try:
            import streamlit
        except ImportError:
            console.print("[yellow]⚠️ Streamlit not installed. Install with: pip install streamlit[/yellow]")
            return False
        
        # Check if streamlit server is already running
        import subprocess
        import time
        
        console.print("[cyan]Attempting to launch Streamlit visualization...[/cyan]")
        
        # Launch streamlit as a subprocess
        process = subprocess.Popen(
            ["streamlit", "run", "streamlit_app.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait briefly to check if process starts successfully
        time.sleep(2)
        
        if process.poll() is None:
            console.print("[green]✓ Streamlit visualization launched! Open http://localhost:8501 in your browser[/green]")
            return True
        else:
            console.print("[yellow]⚠️ Streamlit visualization failed to start[/yellow]")
            return False
    
    except ImportError:
        console.print("[yellow]⚠️ Streamlit not installed. Install with: pip install streamlit[/yellow]")
        return False

def main():
    """Main entry point for the ARIASKA_RL platform"""
    # Setup environment variables and directories
    setup_environment()
    
    # Display welcome banner
    display_welcome_banner()
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Set verbosity level
    os.environ["VERBOSITY"] = args.verbosity
    
    # Execute based on arguments
    if args.test_env:
        test_environment()
    elif args.train:
        run_training(args)
        
        # Launch visualization if requested
        if args.ui:
            launch_visualization()
    elif args.ui:
        # Just launch visualization
        launch_visualization()
    else:
        # Default: show help
        console.print("[yellow]No action specified. Use --train to start training or --help for more options.[/yellow]")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Operation interrupted by user. Shutting down...[/bold yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[bold red]Unhandled error: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)
