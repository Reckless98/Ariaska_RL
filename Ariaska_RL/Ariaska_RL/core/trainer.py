# core/trainer.py — ARIASKA Tactical Orchestrator v2.0

import argparse
import os
import json
import datetime
from rich.console import Console
from rich.table import Table
from core.rl_agent import RLAgent

console = Console()

def parse_args():
    parser = argparse.ArgumentParser(description="Train the ARIASKA RLAgent with simulations and batch training.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of simulation episodes")
    parser.add_argument("--batches", type=int, default=5, help="Number of training batches")
    parser.add_argument("--save", action="store_true", help="Save models after training")
    parser.add_argument("--show-status", action="store_true", help="Display agent status at end")
    parser.add_argument("--checkpoint-every", type=int, default=0, help="Save checkpoint every N episodes (0 = off)")
    return parser.parse_args()

def run_simulation(agent, episodes=10, checkpoint_every=0):
    console.print(f"[bold cyan]🔬 Running {episodes} simulation episode(s)...[/bold cyan]")
    for ep in range(1, episodes + 1):
        agent.simulate_train(episodes=1)

        if checkpoint_every > 0 and ep % checkpoint_every == 0:
            checkpoint_path = f"models/checkpoint_ep{ep}.pt"
            agent.policy_net.save(checkpoint_path.replace(".pt", "_policy.pt"))
            agent.value_net.save(checkpoint_path.replace(".pt", "_value.pt"))
            console.print(f"[dim]📌 Checkpoint saved: {checkpoint_path}[/dim]")

    console.print(f"[green]✔ Simulation complete. Memory and experience updated.[/green]")

def run_batch_training(agent, batches=1):
    console.print(f"[bold cyan]🔧 Starting {batches} batch training iteration(s)...[/bold cyan]")
    for i in range(batches):
        console.print(f"[dim]🔄 Batch {i + 1}[/dim]")
        agent.train_on_batch()
    console.print(f"[green]✔ Batch training complete. Models updated.[/green]")

def save_all(agent):
    try:
        os.makedirs("models", exist_ok=True)
        agent.policy_net.save("models/policy_net.pt")
        agent.value_net.save("models/value_net.pt")
        console.print("[bold green]📦 Models saved to 'models/' directory[/bold green]")
    except Exception as e:
        console.print(f"[red]⚠ Failed to save models: {e}[/red]")

def display_agent_summary(agent):
    table = Table(title="🧠 ARIASKA Agent Training Summary", show_lines=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Agent ID", agent.agent_id)
    table.add_row("Role", agent.role)
    table.add_row("Total Episodes", str(agent.total_episodes))
    table.add_row("Avg Reward", f"{agent.stats_monitor.get_avg_reward():.2f}")
    table.add_row("Epsilon (exploration)", f"{agent.epsilon:.4f}")
    table.add_row("Difficulty Level", str(agent.env.difficulty_level))
    table.add_row("Checkpoint Time", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    console.print(table)

def main():
    args = parse_args()
    console.print("[bold magenta]🚀 Launching ARIASKA RLAgent Training Orchestrator[/bold magenta]")

    agent = RLAgent()

    try:
        run_simulation(agent, episodes=args.episodes, checkpoint_every=args.checkpoint_every)
        run_batch_training(agent, batches=args.batches)

        if args.save:
            save_all(agent)

        if args.show_status:
            display_agent_summary(agent)

        console.print("[bold green]🏁 RLAgent Training Complete![/bold green]")

    except KeyboardInterrupt:
        console.print("[yellow]Training interrupted by user.[/yellow]")

    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/red]")

if __name__ == "__main__":
    main()
