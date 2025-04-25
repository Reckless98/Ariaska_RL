# core/trainer.py — ARIASKA RL Trainer v12.0 APEX PRIME
# 🎮 Unified Multi-Agent Control | 🧠 Smarter GPT Repair | 🔗 Chain Orchestration | 💾 Snapshot Intelligence | 🧠 Orion-Aware

import argparse
import logging
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
    parser.add_argument("--steps", type=int, default=40, help="Steps per episode")
    parser.add_argument("--verbosity", type=str, default="standard", help="Verbosity level")
    parser.add_argument("--no-gpt", action="store_true", help="Disable GPT calls")
    parser.add_argument("--max-episodes", type=int, default=None, help="Maximum episodes to run")
    parser.add_argument("--log-level", type=str, default="info", help="Logging level")
    return parser.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    console.rule("[bold magenta]🚀 ARIASKA RL TRAINER v12.0 — Multi-Agent Protocols Engaged")

    # --- Modular Initialization ---
    from core.monitor.stats_monitor import StatsMonitor
    from core.multiagent.agent_manager import AgentManager
    from core.multiagent.memory_router import MemoryRouter
    from core.teach.teach import TeachModule
    from core.agents.orion_agent import OrionAgent
    from core.multi_agent_trainer import MultiAgentTrainer
    from core.ui_helpers import display_redagent_learning_dashboard
    from core.strategy_optimizer import StrategyOptimizer
    from core.utils.llm_orchestrator import LLMRouter

    # Load configuration (could be extended to YAML/JSON)
    episodes = args.episodes
    steps = args.steps
    verbosity = args.verbosity

    # Instantiate shared components
    stats_monitor = StatsMonitor(verbosity=verbosity)
    agent_manager = AgentManager(verbosity=verbosity)
    memory_router = MemoryRouter(agent_manager.all_agents())
    llm_router = LLMRouter()  # Central LLM dispatcher
    strategy_optimizer = StrategyOptimizer(agent_manager=agent_manager, memory_router=memory_router) if not args.no_gpt else None

    # Assign per-agent modules
    for agent in agent_manager.all_agents():
        if hasattr(agent, "memory_manager") and agent.memory_manager is None:
            from core.utils.memory_manager import MemoryManager
            agent.memory_manager = MemoryManager(agent_id=agent.agent_id)
        if hasattr(agent, "memory_router"):
            agent.memory_router = memory_router
        if hasattr(agent, "stats_monitor"):
            agent.stats_monitor = stats_monitor

    # --- Main Episode Loop (Event-Driven Turn-Based) ---
    for episode in range(1, (args.max_episodes or episodes) + 1):
        console.rule(f"[bold cyan]🌐 Episode {episode}/{episodes} — Multi-Agent Training[/bold cyan]")
        # Reset environment and agent states
        for agent in agent_manager.all_agents():
            if hasattr(agent, "reset"):
                agent.reset()
        # Turn Loop: Red and Blue alternate steps
        for step in range(1, steps + 1):
            # --- RedAgent Turn ---
            red = agent_manager.get_agent("RedAgent")
            red_info = red.simulate_step(episode=episode, step=step)
            # --- BlueAgent Turn ---
            blue = agent_manager.get_agent("BlueAgent")
            blue_info = blue.simulate_step(episode=episode, step=step)
            # --- Output Interpretation ---
            from core.logic.output_interpreter import analyze_output
            red_parsed = analyze_output(red_info["command"], red_info["output"])
            blue_parsed = analyze_output(blue_info["command"], blue_info["output"])
            # --- Reward Computation (handled by agent's simulate_step) ---
            # --- Memory Logging ---
            if hasattr(red, "memory_manager"):
                red.memory_manager.save_experience(red_info)
            if hasattr(blue, "memory_manager"):
                blue.memory_manager.save_experience(blue_info)
            
            # Fixed: Use log_transition with proper parameters instead of route_memory with agent IDs
            memory_router.log_transition(
                red.agent_id, 
                red_info["state"], 
                red_info["command"], 
                red_info["reward"], 
                red_info.get("next_state", red_info["state"]), 
                gpt_tokens=red_info.get("gpt_calls", 0)
            )
            memory_router.log_transition(
                blue.agent_id, 
                blue_info["state"], 
                blue_info["command"], 
                blue_info["reward"], 
                blue_info.get("next_state", blue_info["state"]), 
                gpt_tokens=blue_info.get("gpt_calls", 0)
            )
            
            # --- Stats Logging ---
            stats_monitor.log_step(red.agent_id, red_info["reward"], phase=red_info.get("phase"), command=red_info["command"])
            stats_monitor.log_step(blue.agent_id, blue_info["reward"], phase=blue_info.get("phase"), command=blue_info["command"])
            # --- Teaching Update ---
            teach = TeachModule()
            teach.add_action(command=red_info["command"], description=red_info.get("reasoning", ""), phase=red_info.get("phase"), reward=red_info["reward"])
            teach.add_action(command=blue_info["command"], description=blue_info.get("reasoning", ""), phase=blue_info.get("phase"), reward=blue_info["reward"])
            
            # --- Synchronization (global insights) ---
            # Fixed: route_memory expects a dictionary, not agent IDs
            memory_router.on_turn_end()  # Use event-based hook instead of direct routing
        # --- End of Episode ---
        # Snapshot Memory
        for agent in agent_manager.all_agents():
            if hasattr(agent, "memory_manager") and hasattr(agent.memory_manager, "snapshot"):
                agent.memory_manager.snapshot()
        # Global Sync
        memory_router.snapshot_all_memories()
        memory_router.sync_global_insights()
        # Statistics Display
        stats_monitor.display_episode_summary()
        # Optionally show Orion's insights
        orion = agent_manager.get_agent("OrionAgent")
        if hasattr(orion, "display_episode_summary"):
            orion.display_episode_summary()
        # Chain Generation
        if args.chain:
            from core.logic.chainbuilder import build_and_store_chain
            for agent in agent_manager.all_agents():
                build_and_store_chain(agent.memory_manager.memory)
        # Strategy Optimization
        if strategy_optimizer:
            perf_metrics = stats_monitor.get_performance_metrics()
            strategy_report = strategy_optimizer.optimize_strategy(perf_metrics)
            if strategy_report and "insights" in strategy_report:
                console.print(f"[bold magenta]🧠 Strategy Insights:[/bold magenta] {strategy_report['insights']}")
        # DQN Training (policy/value updates)
        for agent in agent_manager.all_agents():
            if hasattr(agent, "train_on_batch"):
                agent.train_on_batch()
        # Save models
        agent_manager.save_all_models()
        # Visualization (optional)
        from core.visualization.training_visualizer import TrainingVisualizer
        visualizer = TrainingVisualizer(agents=[a.agent_id for a in agent_manager.all_agents()])
        visualizer.update(env_state=red.env.get_global_state())
        visualizer.save_visualization_snapshot()
    # --- Cleanup ---
    memory_router.close()
    stats_monitor.flush_logs()
    console.rule("[bold green]🏁 Training Session Complete — ARIASKA Standing By")

# === Monitoring & Observability Stubs ===
def prometheus_metrics_stub():
    """
    Prometheus metrics endpoint stub. Integrate with Prometheus client to expose agent stats, token usage, and episode metrics.
    """
    pass

def streamlit_dashboard_stub():
    """
    Streamlit dashboard integration stub. Use Streamlit to visualize agent learning, GPT usage, and environment state.
    """
    pass

if __name__ == "__main__":
    main()
