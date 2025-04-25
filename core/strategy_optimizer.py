# core/strategy_optimizer.py — ARIASKA Strategy Optimizer v1.0
# 🧠 Autonomous Strategy Refinement | 🎯 GPT-Guided Evolution | 📊 Performance-Based Tuning

import os
import json
import time
import random
import numpy as np
from rich.console import Console
from typing import Dict, List, Any, Optional
from core.gpt_manager import GPTManager

console = Console()

class StrategyOptimizer:
    """
    Autonomous Strategy Refinement (ASR) for ARIASKA_RL.
    Uses GPT-guided evolution to optimize agent strategies based on performance data.
    
    Features:
    - Performance-based hyperparameter tuning
    - GPT-guided strategy evolution
    - Autonomous curriculum advancement
    - Cross-agent coordination optimization
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
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        
        # Load or create config
        self.config = self._load_or_create_config()
        
        # Initialize GPT manager
        self.gpt_manager = GPTManager()
        
        # Tracking variables
        self.iteration = 0
        self.last_optimization = 0
        self.optimization_interval = self.config.get("optimization_interval", 5)  # Episodes between optimizations
        self.curriculum_level = self.config.get("curriculum_level", 1)
        
        # Stats tracking
        self.optimization_history = []
        
        console.print("[green]✓ StrategyOptimizer initialized.[/green]")
        
    def _load_or_create_config(self):
        """Load or create the optimizer configuration."""
        default_config = {
            "optimization_interval": 5,  # Episodes between strategy optimizations
            "curriculum_level": 1,       # Starting curriculum level
            "curriculum_thresholds": {   # Reward thresholds for advancing curriculum
                "2": 15.0,
                "3": 30.0,
                "4": 50.0,
                "5": 75.0
            },
            "hyperparameters": {
                "epsilon_range": [0.05, 0.9],
                "entropy_beta_range": [0.001, 0.05],
                "learning_rate_range": [1e-5, 1e-3]
            }
        }
        
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r") as f:
                    return json.load(f)
            else:
                # Create default config
                os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
                with open(self.config_path, "w") as f:
                    json.dump(default_config, f, indent=2)
                console.print(f"[yellow]⚠ Created default strategy optimizer config at {self.config_path}[/yellow]")
                return default_config
        except Exception as e:
            console.print(f"[yellow]⚠ Error loading config: {e}. Using defaults.[/yellow]")
            return default_config

    def optimize_strategy(self, episode_num: int = 0) -> Dict[str, Any]:
        """
        Run strategy optimization if due.
        Returns dict with optimization results.
        
        Args:
            episode_num: Current episode number
        
        Returns:
            dict: Strategy optimization results
        """
        # Check if it's time for optimization
        if episode_num > 0 and episode_num % self.optimization_interval != 0:
            return {"optimized": False, "message": "Not scheduled for optimization"}
        
        self.iteration += 1
        self.last_optimization = episode_num
        
        console.print(f"[bold magenta]🧠 Strategy optimization #{self.iteration} (Episode {episode_num})[/bold magenta]")
        
        # Collect performance data
        perf_data = self._collect_performance_data()
        
        # Check curriculum advancement
        curriculum_results = self._check_curriculum_advancement(perf_data)
        
        # Optimize hyperparameters
        hyperparameter_results = self._optimize_hyperparameters(perf_data)
        
        # GPT strategy insights
        strategy_insights = self._get_gpt_strategy_insights(perf_data)
        
        # Compile results
        results = {
            "iteration": self.iteration,
            "episode": episode_num,
            "timestamp": time.time(),
            "curriculum": curriculum_results,
            "hyperparameters": hyperparameter_results,
            "insights": strategy_insights
        }
        
        # Log results
        self._log_optimization(results)
        
        # Display results
        self._display_optimization_results(results)
        
        return results

    def _collect_performance_data(self) -> Dict[str, Any]:
        """
        Collect performance data from all agents.
        
        Returns:
            dict: Performance data by agent
        """
        data = {"agents": {}}
        
        if not self.agent_manager or not hasattr(self.agent_manager, "all_agents"):
            return data
            
        # Collect data from each agent
        for agent in self.agent_manager.all_agents():
            agent_id = getattr(agent, "agent_id", "unknown")
            agent_data = {
                "rewards": [],
                "avg_reward": 0.0,
                "success_rate": 0.0,
                "current_params": {}
            }
            
            # Get rewards from stats monitor if available
            if hasattr(agent, "stats_monitor") and hasattr(agent.stats_monitor, "agent_stats"):
                stats = agent.stats_monitor.agent_stats.get(agent_id, {})
                agent_data["rewards"] = stats.get("rewards", [])
                agent_data["avg_reward"] = sum(agent_data["rewards"]) / max(len(agent_data["rewards"]), 1)
                
                # Estimate success rate based on positive rewards
                positive_rewards = [r for r in agent_data["rewards"] if r > 0]
                agent_data["success_rate"] = len(positive_rewards) / max(len(agent_data["rewards"]), 1)
            
            # Get current hyperparameters
            if hasattr(agent, "epsilon"):
                agent_data["current_params"]["epsilon"] = agent.epsilon
            if hasattr(agent, "epsilon_min"):
                agent_data["current_params"]["epsilon_min"] = agent.epsilon_min
            if hasattr(agent, "entropy_beta"):
                agent_data["current_params"]["entropy_beta"] = agent.entropy_beta
                
            # Add to collected data
            data["agents"][agent_id] = agent_data
            
        # Collect global stats
        data["global"] = {
            "curriculum_level": self.curriculum_level
        }
        
        # Get global insights from memory router if available
        if self.memory_router and hasattr(self.memory_router, "get_global_insights"):
            data["global"]["insights"] = self.memory_router.get_global_insights()
            
        return data

    def _check_curriculum_advancement(self, perf_data) -> Dict[str, Any]:
        """
        Check if curriculum should be advanced based on performance.
        
        Args:
            perf_data: Performance data dictionary
            
        Returns:
            dict: Curriculum advancement results
        """
        results = {
            "level_before": self.curriculum_level,
            "level_after": self.curriculum_level,
            "advanced": False,
            "reason": "Performance below threshold"
        }
        
        # Get RedAgent performance (primary agent)
        red_agent_data = perf_data.get("agents", {}).get("RedAgent", {})
        avg_reward = red_agent_data.get("avg_reward", 0.0)
        
        # Check if we should advance curriculum
        thresholds = self.config.get("curriculum_thresholds", {})
        next_level = str(self.curriculum_level + 1)
        
        if next_level in thresholds and avg_reward >= thresholds[next_level]:
            # Advance curriculum
            self.curriculum_level += 1
            results["level_after"] = self.curriculum_level
            results["advanced"] = True
            results["reason"] = f"Average reward {avg_reward:.2f} exceeds threshold {thresholds[next_level]}"
            
            # Update environment context detector if available
            if self.agent_manager and hasattr(self.agent_manager, "red_agent"):
                red_agent = self.agent_manager.red_agent
                if hasattr(red_agent, "env") and hasattr(red_agent.env, "context_detector"):
                    red_agent.env.context_detector.advance_curriculum()
                    
            console.print(f"[bold green]📚 Curriculum advanced to level {self.curriculum_level}![/bold green]")
            
        return results

    def _optimize_hyperparameters(self, perf_data) -> Dict[str, Any]:
        """
        Optimize hyperparameters based on performance data.
        
        Args:
            perf_data: Performance data dictionary
            
        Returns:
            dict: Hyperparameter optimization results
        """
        results = {
            "agents": {},
            "changes_made": False
        }
        
        if not self.agent_manager or not hasattr(self.agent_manager, "all_agents"):
            return results
            
        # Get hyperparameter ranges
        hp_ranges = self.config.get("hyperparameters", {})
        epsilon_range = hp_ranges.get("epsilon_range", [0.05, 0.9])
        entropy_beta_range = hp_ranges.get("entropy_beta_range", [0.001, 0.05])
        
        # Optimize for each agent
        for agent in self.agent_manager.all_agents():
            agent_id = getattr(agent, "agent_id", "unknown")
            agent_perf = perf_data.get("agents", {}).get(agent_id, {})
            
            # Skip if no performance data
            if not agent_perf:
                continue
                
            # Initialize changes dictionary
            changes = {"before": {}, "after": {}, "changed": False}
            
            # Adjust epsilon based on success rate
            if hasattr(agent, "epsilon") and agent_perf.get("success_rate") is not None:
                success_rate = agent_perf.get("success_rate", 0.5)
                changes["before"]["epsilon"] = agent.epsilon
                
                # Higher success rate = lower epsilon (less exploration)
                # Lower success rate = higher epsilon (more exploration)
                if success_rate > 0.8:
                    # High success rate - decrease exploration
                    agent.epsilon = max(epsilon_range[0], agent.epsilon * 0.9)
                    changes["changed"] = True
                elif success_rate < 0.2:
                    # Low success rate - increase exploration
                    agent.epsilon = min(epsilon_range[1], agent.epsilon * 1.2)
                    changes["changed"] = True
                    
                changes["after"]["epsilon"] = agent.epsilon
                
            # Adjust entropy beta based on reward stability
            if hasattr(agent, "entropy_beta") and agent_perf.get("rewards"):
                rewards = agent_perf.get("rewards", [])
                if len(rewards) >= 10:
                    # Calculate reward variance
                    variance = np.var(rewards[-10:])
                    changes["before"]["entropy_beta"] = agent.entropy_beta
                    
                    # High variance = increase entropy beta (more exploration)
                    # Low variance = decrease entropy beta (more exploitation)
                    if variance > 50:  # High variance threshold
                        agent.entropy_beta = min(entropy_beta_range[1], agent.entropy_beta * 1.2)
                        changes["changed"] = True
                    elif variance < 10:  # Low variance threshold
                        agent.entropy_beta = max(entropy_beta_range[0], agent.entropy_beta * 0.9)
                        changes["changed"] = True
                        
                    changes["after"]["entropy_beta"] = agent.entropy_beta
            
            # Store changes
            results["agents"][agent_id] = changes
            if changes["changed"]:
                results["changes_made"] = True
                
        return results

    def _get_gpt_strategy_insights(self, perf_data) -> Dict[str, Any]:
        """
        Get GPT-guided strategy insights based on performance data.
        
        Args:
            perf_data: Performance data dictionary
            
        Returns:
            dict: Strategy insights
        """
        # Extract key performance metrics for the prompt
        red_perf = perf_data.get("agents", {}).get("RedAgent", {})
        blue_perf = perf_data.get("agents", {}).get("BlueAgent", {})
        
        red_rewards = red_perf.get("rewards", [])[-10:]  # Last 10 rewards
        blue_rewards = blue_perf.get("rewards", [])[-10:]  # Last 10 rewards
        
        red_avg = sum(red_rewards) / max(len(red_rewards), 1)
        blue_avg = sum(blue_rewards) / max(len(blue_rewards), 1)
        
        # Get most common phase if available
        phases = []
        if self.memory_router:
            red_memory = self.memory_router.get_memory("RedAgent")
            for action in red_memory.get("actions", [])[-20:]:  # Last 20 actions
                if isinstance(action, dict) and isinstance(action.get("state"), dict):
                    phase = action.get("state", {}).get("phase")
                    if phase:
                        phases.append(phase)
        
        most_common_phase = max(set(phases), key=phases.count) if phases else "unknown"
        
        # Create prompt for GPT
        prompt = f"""
You are the strategic optimizer for ARIASKA_RL, a multi-agent cybersecurity system.
Analyze these performance metrics and suggest strategic adjustments:

- RedAgent avg reward: {red_avg:.2f} (recent: {red_rewards})
- BlueAgent avg reward: {blue_avg:.2f} (recent: {blue_rewards})
- Current curriculum level: {self.curriculum_level}
- Most common phase: {most_common_phase}

Provide 3 specific, actionable strategic recommendations to improve performance.
Focus on agent coordination, attack diversity, and phase progression.
Format as a bullet list with concise tactical directives.
"""

        try:
            # Query GPT via GPTManager
            gpt_response = self.gpt_manager.gpt_request(
                prompt=prompt,
                task_type="strategic",
                agent_id="StrategyOptimizer"
            )
            
            return {
                "recommendations": gpt_response,
                "success": True
            }
        except Exception as e:
            console.print(f"[yellow]⚠ Failed to get GPT strategy insights: {e}[/yellow]")
            return {
                "recommendations": "Error getting GPT insights.",
                "success": False,
                "error": str(e)
            }

    def _log_optimization(self, results):
        """
        Log optimization results.
        
        Args:
            results: Optimization results dictionary
        """
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(results) + "\n")
        except Exception as e:
            console.print(f"[yellow]⚠ Failed to log optimization: {e}[/yellow]")
            
        # Add to history
        self.optimization_history.append(results)
        
        # Keep history size reasonable
        if len(self.optimization_history) > 20:
            self.optimization_history = self.optimization_history[-20:]

    def _display_optimization_results(self, results):
        """
        Display optimization results in console.
        
        Args:
            results: Optimization results dictionary
        """
        from rich.table import Table
        
        # Curriculum table
        curriculum = results["curriculum"]
        if curriculum["advanced"]:
            console.print(f"[bold green]📚 Curriculum advanced to level {curriculum['level_after']}![/bold green]")
            console.print(f"[green]Reason: {curriculum['reason']}[/green]")
        
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
        if insights["success"]:
            console.print("[bold magenta]🧠 GPT Strategy Recommendations:[/bold magenta]")
            console.print(insights["recommendations"])

# === Testing & Debug Hook ===
if __name__ == "__main__":
    # Create test optimizer
    optimizer = StrategyOptimizer()
    
    # Test optimization
    results = optimizer.optimize_strategy(episode_num=5)
    console.print(results)
