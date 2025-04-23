# core/multiagent/multi_agent_trainer.py — ARIASKA MultiAgentTrainer v11.5 APEX PRIME
# 🎮 Global Orchestration Loop | 👁 Orion Live Strategy | ⚡ Dynamic Agent Sync | ♾️ Smart Cycles

import time
from rich.console import Console
from rich.panel import Panel

from core.multiagent.agent_manager import AgentManager
from core.monitor.stats_monitor import StatsMonitor
from core.logic.chainbuilder import build_and_store_chain_multiagent
from core.teach.teach import TeachModule

console = Console()


class MultiAgentTrainer:
    def __init__(self, agent_manager=None, stats_monitor=None, memory_router=None, verbosity="standard", optimize_mode=False, steps=40):
        self.verbosity = verbosity
        self.steps = steps
        self.agent_manager = agent_manager or AgentManager(verbosity=verbosity)
        self.stats_monitor = stats_monitor or StatsMonitor(self.agent_manager.get_agent_ids())
        self.memory_router = memory_router or self.agent_manager.memory_router
        self.teach = TeachModule()
        self.orion = self.agent_manager.get_agent("OrionAgent")
        self.global_step = 0
        self.sync_interval = 5
        self.strategy_refresh = 10
        self.verbosity = verbosity
        self.optimize_mode = optimize_mode
        self.token_usage = {}
        console.print(
            Panel.fit(
                "[bold cyan]🎮 MultiAgentTrainer v11.5 Initialized — Global Sync Online[/bold cyan]"
            )
        )

    # ─────────────────────────────────────────────
    # 🌐 Unified Simulation-Training Loop
    # ─────────────────────────────────────────────
    def run_global_cycle(self, total_steps=50):
        console.rule("[bold magenta]♾️ ARIASKA Global Orchestration Loop Started")
        agents = self.agent_manager.all_agents()
        red = self.agent_manager.get_agent("RedAgent")
        blue = self.agent_manager.get_agent("BlueAgent")
        scout = self.agent_manager.get_agent("ScoutAgent")
        shadow = self.agent_manager.get_agent("ShadowAgent")

        # Orion pre-briefing phase
        if hasattr(self.orion, "generate_strategic_chain"):
            self.orion.generate_strategic_chain({}, verbosity=self.verbosity)

        while self.global_step < total_steps:
            console.print(
                f"[green]🚀 Global Step {self.global_step + 1}/{total_steps}[/green]"
            )

            # Offensive & Defensive Agents always act
            red.simulate_train(episodes=1)
            blue.simulate_train(episodes=1)

            # Sync Scout & Shadow periodically
            if self.global_step % self.sync_interval == 0:
                scout.advise_phase({}, all_agents=agents)
                shadow.optimize_memory(target_agent_id="RedAgent", all_agents=agents)

            # Orion live strategic adjustments
            if self.global_step % self.strategy_refresh == 0:
                self.orion.apply_orion_strategic_adjustments(agents)

            # Batch train core agents
            red.train_on_batch()
            blue.train_on_batch()

            # Monitor token usage
            for agent in self.agent_manager.all_agents():
                if hasattr(agent, "gpt_calls"):
                    self.token_usage.setdefault(agent.agent_id, {"4o-mini": 0, "4.1-full": 0})
                    for k in self.token_usage[agent.agent_id]:
                        self.token_usage[agent.agent_id][k] += agent.gpt_calls.get(k, 0)

            # Dynamic episode termination
            if hasattr(self.agent_manager.red_agent.env, "detection_risk"):
                risk = self.agent_manager.red_agent.env.detection_risk
                if risk > 7.0:
                    console.log("🔚 Ending episode early due to high risk.")
                    break

            self.global_step += 1
            time.sleep(1)

        console.print(
            "[bold green]🏁 Global Cycle Completed — Proceeding to Post-Processing[/bold green]"
        )
        self.post_cycle_operations()

    # ─────────────────────────────────────────────
    # 🧠 Post-Cycle Intelligence & GPT Sync
    # ─────────────────────────────────────────────
    def post_cycle_operations(self):
        console.rule("[bold cyan]🧠 Post-Cycle Intelligence: Sync | Analyze | Optimize")

        # 1. Consolidate GPT Cache & Global Insights
        self.memory_router.consolidate_gpt_cache()
        self.memory_router.sync_global_insights()

        # 2. Snapshot All Memories
        self.save_snapshots()

        # 3. Generate Updated Attack Chains
        self.generate_attack_chains()

        # 4. Orion Deep Strategic Review
        self.orion.analyze_training(self.agent_manager.all_agents())

        # 5. Apply Orion's Final Adjustments
        self.orion.apply_orion_strategic_adjustments(self.agent_manager.all_agents())

        console.print(
            "[green]✔ Post-cycle operations completed. ARIASKA is optimized and aligned.[/green]"
        )

    # ─────────────────────────────────────────────
    # 🔗 Chain Generation with Orion Oversight
    # ─────────────────────────────────────────────
    def generate_attack_chains(self):
        console.print("[magenta]🔗 Synthesizing Multi-Agent Attack Chains...[/magenta]")
        build_and_store_chain_multiagent(self.agent_manager)

    # ─────────────────────────────────────────────
    # 💾 Snapshot & GPT Cache Management
    # ─────────────────────────────────────────────
    def save_snapshots(self):
        console.print(
            "[cyan]📸 Saving Memory Snapshots & Syncing GPT Intelligence...[/cyan]"
        )
        self.memory_router.snapshot_all_memories()
        self.memory_router.consolidate_gpt_cache()

    # ─────────────────────────────────────────────
    # ♾️ GPT-Enhanced Autopilot Mode
    # ─────────────────────────────────────────────
    def run_autopilot(self, cycles=3):
        console.rule(
            "[bold magenta]♾️ ARIASKA Autopilot — Adaptive Multi-Agent Execution"
        )
        for cycle in range(1, cycles + 1):
            console.print(f"[green]🚀 Starting Cycle {cycle}/{cycles}[/green]")

            # Dynamic GPT-driven adjustments before each cycle
            self.orion.update_global_strategy(
                self.agent_manager.all_agents(), environment="dynamic"
            )

            # Execute Simulation & Training
            self.orchestrate_simulation(episodes=5 + cycle)
            self.orchestrate_batch_training(batches=2 + (cycle // 2))

            # Mid-Cycle Memory Optimization
            self.repair_memories(threshold=15)

            # Post-Cycle Intelligence Sync
            self.post_cycle_operations()

        console.print(
            "[bold green]🏁 Autopilot Complete — ARIASKA Optimized & Standing By[/bold green]"
        )

    # ─────────────────────────────────────────────
    # 🚀 Diagnostic Execution (Smart CLI Mode)
    # ─────────────────────────────────────────────

    def orchestrate_simulation(self, episodes=10):
        """Main simulation loop with advanced dashboard and error handling."""
        from core.visualization.training_visualizer import TrainingVisualizer
        visualizer = TrainingVisualizer(
            agents=[a.agent_id for a in self.agent_manager.all_agents()],
            max_history=100
        )
        visualizer.start_live_display()
        try:
            for ep in range(episodes):
                console.print(f"[bold cyan]Episode {ep+1}/{episodes}[/bold cyan]")
                self.agent_manager.simulate_all_agents(episodes=1, max_steps=40)
                # Update visualization after each episode
                env_state = self.agent_manager.red_agent.env.get_global_state()
                visualizer.update(env_state=env_state)
                if (ep + 1) % 10 == 0:
                    visualizer.save_visualization_snapshot()
                    visualizer.create_training_report(ep + 1)
                self.agent_manager.display_full_status()
                if (ep + 1) % 10 == 0:
                    self.agent_manager.save_all_models()
            visualizer.stop_live_display()
        except Exception as e:
            console.print(f"[bold red]❌ Simulation error: {e}[/bold red]")
            import traceback
            console.print(traceback.format_exc())
            try:
                visualizer.stop_live_display()
                self.agent_manager.display_full_status()
            except:
                console.print("[red]❌ Status display also failed.[/red]")

    def orchestrate_batch_training(self, batches=5):
        """Batch train all agents with advanced error handling."""
        self.agent_manager.batch_train_all(batches=batches)

    def repair_memories(self, threshold=15):
        """Repair low-reward memories using GPT."""
        self.memory_router.optimize_memories(threshold=threshold)


if __name__ == "__main__":
    console.rule("[bold cyan]🧪 ARIASKA MultiAgentTrainer Diagnostic Mode[/bold cyan]")
    trainer = MultiAgentTrainer()
    trainer.run_autopilot(cycles=2)

    # After autopilot, generate strategic report
    trainer.orion.generate_strategic_report(
        trainer.agent_manager.all_agents(), environment="dynamic"
    )

    console.print(
        "[blue]📊 Diagnostic run complete. Review logs for detailed insights.[/blue]"
    )
