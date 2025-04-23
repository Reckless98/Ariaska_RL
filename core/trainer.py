# core/trainer.py — ARIASKA RL Trainer v12.0 APEX PRIME
# 🎮 Unified Multi-Agent Control | 🧠 Smarter GPT Repair | 🔗 Chain Orchestration | 💾 Snapshot Intelligence | 🧠 Orion-Aware

import argparse
from rich.console import Console

console = Console()

def parse_args():
    parser = argparse.ArgumentParser(description="ARIASKA RL — Multi-Agent Training Framework v12.0 APEX PRIME")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per agent")
    parser.add_argument("--batches", type=int, default=5, help="Training batches per agent")
    parser.add_argument("--auto", action="store_true", help="Run full autopilot cycles")
    parser.add_argument("--repair", action="store_true", help="Run GPT replay memory repair")
    parser.add_argument("--chain", action="store_true", help="Generate attack chains after training")
    parser.add_argument("--snapshot", action="store_true", help="Snapshot all memories post-training")
    parser.add_argument("--orion", action="store_true", help="Enable Orion agent strategy review")
    parser.add_argument("--cycles", type=int, default=3, help="Autopilot cycles (if --auto)")
    return parser.parse_args()

def main():
    args = parse_args()
    console.rule("[bold magenta]🚀 ARIASKA RL TRAINER v12.0 — Multi-Agent Protocols Engaged")

    # Import dependencies locally to avoid circular imports
    from core.monitor.stats_monitor import StatsMonitor
    from core.multiagent.agent_manager import AgentManager
    from core.multiagent.memory_router import MemoryRouter
    from core.teach.teach import TeachModule
    from core.agents.orion_agent import OrionAgent
    from core.multi_agent_trainer import MultiAgentTrainer

    # Initialize the StatsMonitor and AgentManager for tracking
    stats_monitor = StatsMonitor()
    agent_manager = AgentManager()
    memory_router = MemoryRouter(agent_manager.all_agents())

    # Initialize Orion for pre/post-training insight
    orion_agent = OrionAgent(agent_manager=agent_manager, memory_router=memory_router)

    # Initialize the MultiAgentTrainer
    trainer = MultiAgentTrainer(agent_manager=agent_manager, stats_monitor=stats_monitor, memory_router=memory_router)

    # Pre-training briefing from Orion
    if args.orion:
        orion_agent.trigger_orion_review()

    if args.auto:
        trainer.run_autopilot(cycles=args.cycles)
    else:
        # Orchestrate simulation and batch training
        trainer.orchestrate_simulation(episodes=args.episodes)
        trainer.orchestrate_batch_training(batches=args.batches)

        if args.repair:
            trainer.repair_memories()

        if args.chain:
            trainer.generate_attack_chains()

        if args.snapshot:
            trainer.save_snapshots()

    # Post-training Orion insights and session summary
    if args.orion:
        orion_agent.display_episode_summary()

    console.rule("[bold green]🏁 Training Session Complete — ARIASKA Standing By")

if __name__ == "__main__":
    main()
