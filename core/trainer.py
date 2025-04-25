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

from core.multi_agent_trainer import display_live_training_dashboard, log_phase_transition

def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    console.rule("[bold magenta]🚀 ARIASKA RL TRAINER v12.0 — Multi-Agent Protocols Engaged")

    # Import dependencies locally to avoid circular imports
    from core.monitor.stats_monitor import StatsMonitor
    from core.multiagent.agent_manager import AgentManager
    from core.multiagent.memory_router import MemoryRouter
    from core.teach.teach import TeachModule
    from core.agents.orion_agent import OrionAgent
    from core.multi_agent_trainer import MultiAgentTrainer
    from core.ui_helpers import display_redagent_learning_dashboard

    # Initialize the StatsMonitor and AgentManager for tracking
    stats_monitor = StatsMonitor(verbosity=args.verbosity)
    agent_manager = AgentManager(verbosity=args.verbosity)
    memory_router = MemoryRouter(agent_manager.all_agents())

    # Initialize Orion for pre/post-training insight
    orion_agent = OrionAgent(agent_manager=agent_manager, memory_router=memory_router)

    # Initialize the MultiAgentTrainer
    trainer = MultiAgentTrainer(agent_manager=agent_manager, stats_monitor=stats_monitor, memory_router=memory_router)

    # Pre-training briefing from Orion
    if args.orion:
        orion_agent.trigger_orion_review()

    # --- Enhanced Training Loop ---
    meta_interval = 5  # Meta-learning every N episodes
    rewards_history = []
    llm_usage = {"Seneca": {"calls": 0, "tokens": 0}, "Lily": {"calls": 0, "tokens": 0}, "GPT-4o": {"calls": 0, "tokens": 0}}
    prev_phase = None
    try:
        if args.auto:
            trainer.run_autopilot(cycles=args.cycles)
        else:
            # Smarter per-agent training loop with detailed logging
            for episode in range(1, (args.max_episodes or args.episodes) + 1):
                console.rule(f"[bold cyan]🌐 Episode {episode}/{args.episodes} — Multi-Agent Training[/bold cyan]")
                for agent in agent_manager.all_agents():
                    if hasattr(agent, "reset"):
                        agent.reset()
                for step in range(args.steps):
                    for agent in agent_manager.all_agents():
                        if hasattr(agent, "simulate_step"):
                            info = agent.simulate_step(episode=episode, step=step + 1)
                            # Human-readable UI logging
                            console.print(
                                f"[bold blue]{agent.agent_id}[/bold blue] | "
                                f"Step: {step+1} | "
                                f"Phase: {info.get('phase', 'N/A')} | "
                                f"Action: {info.get('command', 'N/A')} | "
                                f"Reward: {info.get('reward', 0):+.2f} | "
                                f"GPT Calls: {info.get('gpt_calls', 0)}"
                            )
                            if info.get("reasoning"):
                                console.print(f"[dim]🧠 Reasoning: {info['reasoning']}[/dim]")
                            # Prevent repetitive actions: check replay buffer and adapt
                            if hasattr(agent, "replay_buffer") and hasattr(agent, "command_history"):
                                recent_cmds = agent.command_history[-5:]
                                if len(set(recent_cmds)) < len(recent_cmds):
                                    console.print(
                                        f"[yellow]♻️ {agent.agent_id}: Detected repetition in recent actions. Forcing exploration.[/yellow]"
                                    )
                                    agent.epsilon = min(1.0, agent.epsilon * 1.05)
                            # --- RedAgent dashboard after each step ---
                            if getattr(agent, "agent_id", None) == "RedAgent":
                                display_redagent_learning_dashboard(
                                    redagent=agent,
                                    memory_router=agent.memory_router,
                                    redagent_brain=agent.redagent_brain
                                )
                            # CLI live dashboard
                            rewards_history.append(info.get("reward", 0))
                            # Phase transition logging
                            phase = info.get("phase", None)
                            if prev_phase is not None and phase is not None:
                                log_phase_transition(prev_phase, phase)
                            prev_phase = phase
                            # LLM usage tracking (example, adapt as needed)
                            if hasattr(agent, "gpt_manager"):
                                llm_usage["GPT-4o"]["calls"] += 1
                                llm_usage["GPT-4o"]["tokens"] += agent.gpt_manager.get_token_usage(agent.agent_id)
                            # Display live dashboard every 5 steps
                            if (step + 1) % 5 == 0:
                                display_live_training_dashboard(agent_manager.all_agents(), episode, step + 1, rewards_history, llm_usage)
                    # Optional: checkpoint every N steps
                    if (step + 1) % 10 == 0:
                        console.print(f"[green]💾 Checkpointing models and memory at step {step+1}[/green]")
                        agent_manager.save_all_models()
                        agent_manager.snapshot_all()
                # End of episode: batch train and checkpoint
                console.print(f"[magenta]🔁 Batch training after episode {episode}[/magenta]")
                agent_manager.batch_train_all(batches=args.batches)
                agent_manager.save_all_models()
                agent_manager.snapshot_all()
                agent_manager.sync_all_memories()
                # Show episode summary
                agent_manager._log_multiagent_episode(episode)
                # Meta-learning loop: every N episodes, summarize RedAgent evolution and send to GPT for strategy update
                if episode % meta_interval == 0:
                    red_agent = agent_manager.get_agent("RedAgent")
                    if hasattr(red_agent, "memory_router") and hasattr(red_agent.memory_router, "get_redagent_evolution_stats"):
                        stats = red_agent.memory_router.get_redagent_evolution_stats(n=200)
                        summary = f"Top commands: {sorted(stats['commands'], key=stats['commands'].get, reverse=True)[:5]}\n" \
                                  f"Success rates: {sorted(stats['success_rates'].items(), key=lambda x: x[1], reverse=True)[:5]}\n" \
                                  f"Failures: {sorted(stats['failures'].items(), key=lambda x: x[1], reverse=True)[:5]}"
                        prompt = f"""
You are a cybersecurity RL coach. Analyze these RedAgent stats and recommend improvements:
{summary}
- Identify weak strategies and suggest alternatives.
- Recommend new tactics to explore.
Respond in JSON: {{"recommendations": [...], "new_tactics": [...]}}
"""
                        gpt_feedback = red_agent.gpt_manager.gpt_request(prompt, task_type="meta-analysis")
                        # Optionally log or print meta-analysis
                        console.print(f"[bold magenta]🧠 Meta-Learning Feedback:[/bold magenta] {gpt_feedback}")
                # Checkpointing every 5 episodes
                if episode % 5 == 0:
                    agent_manager.save_all_models()
                    agent_manager.snapshot_all()
            # Print LLM usage summary at end
            display_live_training_dashboard(agent_manager.all_agents(), episode, args.steps, rewards_history, llm_usage)
            # Post-training Orion insights and session summary
            if args.orion:
                orion_agent.display_episode_summary()
    except Exception as e:
        logging.error(f"Training loop crashed: {e}")
        agent_manager.save_all_models()
        agent_manager.snapshot_all()
        raise

    console.rule("[bold green]🏁 Training Session Complete — ARIASKA Standing By")

# === Monitoring & Observability Stubs ===
def prometheus_metrics_stub():
    """
    Prometheus metrics endpoint stub. Integrate with Prometheus client to expose agent stats, token usage, and episode metrics.
    """
    pass  # TODO: Implement Prometheus metrics export

def streamlit_dashboard_stub():
    """
    Streamlit dashboard integration stub. Use Streamlit to visualize agent learning, GPT usage, and environment state.
    """
    pass  # TODO: Implement Streamlit dashboard

if __name__ == "__main__":
    main()
