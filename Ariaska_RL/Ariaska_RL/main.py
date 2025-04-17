# main.py — ARIASKA CLI Core (v7.1 Reinforced I/O + Command Execution Fix)

import asyncio
from core.ui_helpers import (
    create_prompt_session,
    display_output,
    display_ai_hint_table,
    display_phase_tables
)
from core.monitor import StatsMonitor
from core.rl_agent import RLAgent
from rich.console import Console
from rich.panel import Panel

console = Console()

# ─────────────────────────────────────────────
# Instantiate RL Agent
# ─────────────────────────────────────────────
rl_agent = RLAgent(agent_id="RedAgent", role="Red Team", device="cuda")

# ─────────────────────────────────────────────
# Prompt Session (Interactive Shell)
# ─────────────────────────────────────────────
session = create_prompt_session()

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
    panel = Panel.fit(banner_text, style="bold magenta", padding=(1, 2))
    console.print(panel)

# ─────────────────────────────────────────────
# AI Hint Logic
# ─────────────────────────────────────────────
def show_hint():
    hint_command = rl_agent.generate_hint()
    ai_recommendations = []

    if hint_command:
        ai_recommendations.append({
            "command": hint_command,
            "params": "Auto",
            "why": "Rule Engine Suggestion",
            "Full Command": hint_command
        })
        display_ai_hint_table(hint_command, ai_recommendations)
    else:
        console.print("[yellow]⚠ No hint generated. Using fallback.[/yellow]")
        fallback_hint = "nmap -p- -sC -sV TARGET"
        ai_recommendations.append({
            "command": fallback_hint,
            "params": "-p- -sC -sV",
            "why": "Fallback (no memory)",
            "Full Command": fallback_hint
        })
        display_ai_hint_table(fallback_hint, ai_recommendations)

# ─────────────────────────────────────────────
# Command: Train on Replay
# ─────────────────────────────────────────────
def run_replay_training():
    console.print("[cyan]🔎 Running Replay Trainer...[/cyan]")
    rl_agent.train_on_batch()

# ─────────────────────────────────────────────
# Command: Simulated RL Training
# ─────────────────────────────────────────────
def run_simulated_environment_training():
    console.print("[cyan]🤖 Running Cyber Simulation Training...[/cyan]")
    rl_agent.simulate_train(episodes=50)

# ─────────────────────────────────────────────
# Main Async Shell Loop
# ─────────────────────────────────────────────
async def main_loop():
    banner()

    console.print("[cyan]Initializing... Loading Phase Guides.[/cyan]")
    display_phase_tables()
    console.print("[green]✔ Ariaska RL Ready. Awaiting your command.[/green]\n")

    while True:
        try:
            command = await session.prompt_async("zer0@ARIASKA > ")

            if not command.strip():
                console.print("[yellow]⚠ No input. Try again.[/yellow]")
                continue

            if command.strip().lower() in ["exit", "quit"]:
                console.print("[red]Exiting Ariaska RL. Until next battle.[/red]")
                break

            if command.strip().lower() == "train-batch":
                run_replay_training()
                continue

            if command.strip().lower() == "simulate-train":
                run_simulated_environment_training()
                continue

            # 🔧 Execute user command
            result = rl_agent.execute_command(command)

            if not isinstance(result, dict):
                console.print(f"[red]❌ Invalid return value from execute_command[/red]")
                continue

            display_output(result.get("output", "[yellow]⚠ No output returned.[/yellow]"))
            display_ai_hint_table(hint=None, recommendations=result.get("recommendations", []))
            show_hint()

        except (KeyboardInterrupt, EOFError):
            console.print("\n[red]⚠ User interrupt. Shutting down Ariaska RL.[/red]")
            break
        except Exception as e:
            console.print(f"[bold red]❌ Unexpected error: {e}[/bold red]")

# ─────────────────────────────────────────────
# Launch CLI Loop
# ─────────────────────────────────────────────
if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except Exception as e:
        console.print(f"[red]Fatal error in main loop: {e}[/red]")
