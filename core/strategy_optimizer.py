import os
import json
import time
from typing import Dict, Any, Optional
from rich.console import Console
from rich.table import Table
from core.gpt_manager import GPTManager

console = Console()

# ─────────────────────────────────────────────
# 📊 Performance Analyzer
# ─────────────────────────────────────────────
class PerformanceAnalyzer:
    def __init__(self, agent_manager, memory_router, config_path="config/perf_analyzer.json"):
        self.agent_manager = agent_manager
        self.memory_router = memory_router
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self):
        default = {"reward_window": 10}
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    return json.load(f)
            except Exception:
                return default
        return default

    def collect_metrics(self) -> Dict[str, Any]:
        """Aggregate stats from agents and memory."""
        data = {"agents": {}}
        for agent in self.agent_manager.all_agents():
            agent_id = getattr(agent, "agent_id", "unknown")
            stats = getattr(agent, "stats_monitor", None)
            rewards = stats.agent_stats[agent_id]["rewards"] if stats and agent_id in stats.agent_stats else []
            avg_reward = sum(rewards[-self.config.get("reward_window", 10):]) / max(1, len(rewards[-self.config.get("reward_window", 10):]))
            agent_data = {
                "rewards": rewards,
                "avg_reward": avg_reward,
                "success_rate": sum(1 for r in rewards if r > 0) / max(1, len(rewards)),
                "current_params": {
                    "epsilon": getattr(agent, "epsilon", None),
                    "entropy_beta": getattr(agent, "entropy_beta", None),
                }
            }
            data["agents"][agent_id] = agent_data
        # Optionally add global insights
        if self.memory_router and hasattr(self.memory_router, "get_global_insights"):
            data["global"] = {"insights": self.memory_router.get_global_insights()}
        return data

# ─────────────────────────────────────────────
# 🔧 Hyperparameter Tuner
# ─────────────────────────────────────────────
class HyperparamTuner:
    def __init__(self, agent_manager, config_path="config/hyperparam_tuner.json"):
        self.agent_manager = agent_manager
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self):
        default = {
            "epsilon_range": [0.05, 0.9],
            "entropy_beta_range": [0.001, 0.05],
            "reward_thresholds": {"advance": 15.0, "retreat": 5.0}
        }
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    return json.load(f)
            except Exception:
                return default
        return default

    def tune(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust agent hyperparameters based on metrics."""
        results = {"agents": {}, "changes_made": False}
        for agent in self.agent_manager.all_agents():
            agent_id = getattr(agent, "agent_id", "unknown")
            perf = metrics.get("agents", {}).get(agent_id, {})
            changes = {"before": {}, "after": {}, "changed": False}
            # Epsilon tuning
            if hasattr(agent, "epsilon") and perf.get("success_rate") is not None:
                old_eps = agent.epsilon
                if perf["success_rate"] > 0.8:
                    agent.epsilon = max(self.config["epsilon_range"][0], agent.epsilon * 0.9)
                    changes["changed"] = True
                elif perf["success_rate"] < 0.2:
                    agent.epsilon = min(self.config["epsilon_range"][1], agent.epsilon * 1.2)
                    changes["changed"] = True
                changes["before"]["epsilon"] = old_eps
                changes["after"]["epsilon"] = agent.epsilon
            # Entropy beta tuning
            if hasattr(agent, "entropy_beta") and perf.get("rewards"):
                rewards = perf["rewards"]
                if len(rewards) >= 10:
                    import numpy as np
                    variance = np.var(rewards[-10:])
                    old_beta = agent.entropy_beta
                    if variance > 50:
                        agent.entropy_beta = min(self.config["entropy_beta_range"][1], agent.entropy_beta * 1.2)
                        changes["changed"] = True
                    elif variance < 10:
                        agent.entropy_beta = max(self.config["entropy_beta_range"][0], agent.entropy_beta * 0.9)
                        changes["changed"] = True
                    changes["before"]["entropy_beta"] = old_beta
                    changes["after"]["entropy_beta"] = agent.entropy_beta
            results["agents"][agent_id] = changes
            if changes["changed"]:
                results["changes_made"] = True
        return results

# ─────────────────────────────────────────────
# 🧠 GPT Advisor (Optional, Rate-Limited)
# ─────────────────────────────────────────────
class GPTAdvisor:
    def __init__(self, gpt_manager=None, config_path="config/gpt_advisor.json"):
        self.gpt_manager = gpt_manager or GPTManager()
        self.config_path = config_path
        self.config = self._load_config()
        self.last_call = 0

    def _load_config(self):
        default = {"min_interval": 5}
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    return json.load(f)
            except Exception:
                return default
        return default

    def advise(self, metrics: Dict[str, Any], force=False) -> Dict[str, Any]:
        """Generate strategy insights using GPT, rate-limited."""
        now = time.time()
        if not force and now - self.last_call < self.config.get("min_interval", 5):
            return {"recommendations": "GPTAdvisor: Skipped (rate limit)", "success": False}
        self.last_call = now
        try:
            red = metrics.get("agents", {}).get("RedAgent", {})
            blue = metrics.get("agents", {}).get("BlueAgent", {})
            prompt = f"""
You are the strategic optimizer for ARIASKA_RL, a multi-agent cybersecurity system.
Analyze these performance metrics and suggest strategic adjustments:

- RedAgent avg reward: {red.get('avg_reward', 0):.2f}
- BlueAgent avg reward: {blue.get('avg_reward', 0):.2f}
- RedAgent success rate: {red.get('success_rate', 0):.2f}
- BlueAgent success rate: {blue.get('success_rate', 0):.2f}

Provide 3 specific, actionable strategic recommendations to improve performance.
Format as a bullet list.
"""
            gpt_response = self.gpt_manager.gpt_request(
                prompt=prompt,
                task_type="strategic",
                agent_id="StrategyOptimizer"
            )
            return {"recommendations": gpt_response, "success": True}
        except Exception as e:
            console.print(f"[yellow]⚠ GPTAdvisor failed: {e}[/yellow]")
            return {"recommendations": "Error getting GPT insights.", "success": False, "error": str(e)}

# ─────────────────────────────────────────────
# 🏆 Strategy Optimizer (Orchestrator)
# ─────────────────────────────────────────────
class StrategyOptimizer:
    """
    Modular strategy optimizer for ARIASKA_RL.
    - PerformanceAnalyzer: collects stats
    - HyperparamTuner: tunes agent hyperparameters
    - GPTAdvisor: (optional) generates strategy insights
    """
    def __init__(
        self,
        agent_manager=None,
        memory_router=None,
        config_path="config/strategy_optimizer.json",
        log_path="logs/strategy_optimizer.jsonl"
    ):
        self.agent_manager = agent_manager
        self.memory_router = memory_router
        self.config_path = config_path
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.analyzer = PerformanceAnalyzer(agent_manager, memory_router)
        self.tuner = HyperparamTuner(agent_manager)
        self.gpt_advisor = GPTAdvisor()
        self.iteration = 0
        self.last_optimization = 0
        self.optimization_interval = 5
        self.optimization_history = []
        console.print("[green]✓ StrategyOptimizer initialized.[/green]")

    def optimize_strategy(self, episode_num: int = 0, force_gpt=False) -> Dict[str, Any]:
        """
        Run strategy optimization. Returns dict with optimization results.
        """
        self.iteration += 1
        self.last_optimization = episode_num
        console.print(f"[bold magenta]🧠 Strategy optimization #{self.iteration} (Episode {episode_num})[/bold magenta]")
        # 1. Collect metrics
        metrics = self.analyzer.collect_metrics()
        # 2. Tune hyperparameters
        hp_results = self.tuner.tune(metrics)
        # 3. Optionally get GPT advice (rate-limited)
        gpt_results = self.gpt_advisor.advise(metrics, force=force_gpt or (episode_num % 10 == 0))
        # 4. Compile results
        results = {
            "iteration": self.iteration,
            "episode": episode_num,
            "timestamp": time.time(),
            "hyperparameters": hp_results,
            "insights": gpt_results
        }
        self._log_optimization(results)
        self._display_optimization_results(results)
        return results

    def _log_optimization(self, results):
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(results) + "\n")
        except Exception as e:
            console.print(f"[yellow]⚠ Failed to log optimization: {e}[/yellow]")
        self.optimization_history.append(results)
        if len(self.optimization_history) > 20:
            self.optimization_history = self.optimization_history[-20:]

    def _display_optimization_results(self, results):
        from rich.table import Table
        # Hyperparameter table
        hp_changes = results["hyperparameters"]["agents"]
        if hp_changes:
            table = Table(title="Hyperparameter Adjustments")
            table.add_column("Agent", style="cyan")
            table.add_column("Parameter", style="magenta")
            table.add_column("Before", style="yellow")
            table.add_column("After", style="green")
            for agent_id, changes in hp_changes.items():
                if changes["changed"]:
                    for param, before_val in changes["before"].items():
                        after_val = changes["after"][param]
                        table.add_row(agent_id, param, f"{before_val:.4f}", f"{after_val:.4f}")
            if results["hyperparameters"]["changes_made"]:
                console.print(table)
        # GPT insights
        insights = results["insights"]
        if insights.get("success"):
            console.print("[bold magenta]🧠 GPT Strategy Recommendations:[/bold magenta]")
            console.print(insights["recommendations"])

# ─────────────────────────────────────────────
# 🚀 Diagnostic Mode
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Create test optimizer
    from core.multiagent.agent_manager import AgentManager
    from core.multiagent.memory_router import MemoryRouter
    agent_manager = AgentManager()
    memory_router = MemoryRouter(agent_manager.all_agents())
    optimizer = StrategyOptimizer(agent_manager=agent_manager, memory_router=memory_router)
    # Test optimization
    results = optimizer.optimize_strategy(episode_num=5, force_gpt=True)
    console.print(results)
