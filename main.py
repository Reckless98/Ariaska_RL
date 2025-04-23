import asyncio
import os
import signal
import logging
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from core.ui_helpers import (
    create_prompt_session,
    display_output,
    display_ai_hint_table,
    display_phase_tables,
)
from core.multiagent.agent_manager import AgentManager
from core.logic.chainbuilder import ChainBuilder  # Fixed import
from core.monitor.stats_monitor import StatsMonitor
from prompt_toolkit.formatted_text import HTML

logger = logging.getLogger("ariaska")
logging.basicConfig(level=logging.INFO, filename="logs/ariaska.log", filemode="a")

console = Console()
session = create_prompt_session()
stats_monitor = StatsMonitor()  # StatsMonitor will track and display agent stats
agent_manager = AgentManager()  # Initialize the agent manager

# Define the primary agent using AgentManager
primary_agent = agent_manager.red_agent  # Or use .get_primary_agent() if implemented

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
            if hasattr(agent_manager, "orion_agent"):
                strategy_insight = agent_manager.orion_agent.provide_reasoning("episode_start", {"episode": ep+1})
                console.print(f"[blue]👁️ Episode Strategy: [/blue]{strategy_insight}")
            
            # Run episode simulation
            agent_manager.simulate_all_agents(episodes=1, max_steps=40)
            
            # Update visualization if available
            if visualizer:
                env_state = agent_manager.red_agent.env.get_global_state()
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
    chain_builder = ChainBuilder(memory_router=memory_router)
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

            cmd_lower = command.strip().lower()  # Fixed typo: was toLowerCase()
            args = command.strip().split()

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
            else:
                # Live Command Execution via Primary Agent
                if primary_agent is None:
                    console.print("[red]❌ No primary agent available.[/red]")
                    continue
                result = primary_agent.execute_command(command)
                if not isinstance(result, dict):
                    console.print("[red]❌ Unexpected result format[/red]")
                    continue

                output = result.get("output", "[yellow]⚠ No output returned.[/yellow]")
                recommendations = result.get("recommendations", [])
                phase = result.get("phase", "unknown")
                reward = result.get("reward", 0)
                alert = result.get("alert", 0.0)
                entropy = result.get("entropy", None)

                primary_agent.stats_monitor.log_step(
                    primary_agent.agent_id, reward, alert=alert, phase=phase
                )

                if primary_agent.stats_monitor.total_steps % 10 == 0:
                    primary_agent.stats_monitor.show()

                display_output(output)
                display_ai_hint_table(None, recommendations)
                show_hint()

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
