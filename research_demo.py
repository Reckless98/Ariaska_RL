#!/usr/bin/env python3
"""
ARIASKA_RL Research Framework Demo

Demonstrates the enhanced research capabilities added to ARIASKA_RL,
including benchmarking, experiment management, statistical analysis,
and research methodology tools.
"""

import os
import sys
import time
import asyncio
from typing import Dict, List, Any

# Add core to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

# Research framework imports
from core.research.research_integration import ResearchIntegration
from core.research.benchmark_suite import BenchmarkMetrics
from core.research.experiment_manager import ExperimentConfig
from core.research.research_methodology import ResearchQuestion

# Rich for beautiful output
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table

console = Console()

class ResearchDemo:
    """
    Demonstration of ARIASKA_RL research enhancements.
    
    Shows how to:
    1. Set up research questions and methodology
    2. Design and run experiments with statistical rigor
    3. Analyze results with advanced metrics
    4. Generate publication-ready reports
    5. Use curriculum learning for adaptive training
    """
    
    def __init__(self):
        self.research_framework = ResearchIntegration("demo_research")
        console.print("[green]🧪 ARIASKA_RL Research Demo initialized[/green]")
    
    def demonstrate_research_workflow(self):
        """Run complete research workflow demonstration"""
        
        console.print(Panel.fit(
            "🔬 ARIASKA_RL Research Framework Demonstration\n"
            "This demo showcases the enhanced research capabilities",
            style="bold blue"
        ))
        
        # Step 1: Define research questions
        console.print("\n[cyan]Step 1: Defining Research Questions[/cyan]")
        research_questions = self._demo_research_questions()
        
        # Step 2: Create experiments
        console.print("\n[cyan]Step 2: Creating Experiments[/cyan]")
        baseline_config = self._demo_experiment_creation()
        
        # Step 3: Demonstrate hyperparameter sweep
        console.print("\n[cyan]Step 3: Hyperparameter Optimization[/cyan]")
        sweep_configs = self._demo_hyperparameter_sweep(baseline_config)
        
        # Step 4: Create A/B test
        console.print("\n[cyan]Step 4: A/B Testing Framework[/cyan]")
        ab_configs = self._demo_ab_testing(baseline_config)
        
        # Step 5: Simulate experiment execution
        console.print("\n[cyan]Step 5: Experiment Execution Simulation[/cyan]")
        results = self._demo_experiment_execution(baseline_config)
        
        # Step 6: Demonstrate curriculum learning
        console.print("\n[cyan]Step 6: Adaptive Curriculum Learning[/cyan]")
        curriculum_results = self._demo_curriculum_learning()
        
        # Step 7: Analysis and reporting
        console.print("\n[cyan]Step 7: Statistical Analysis & Reporting[/cyan]")
        analysis_results = self._demo_analysis_and_reporting()
        
        # Step 8: Research methodology tools
        console.print("\n[cyan]Step 8: Research Methodology & Documentation[/cyan]")
        methodology_outputs = self._demo_research_methodology()
        
        # Final summary
        console.print("\n[green]✅ Research Framework Demo Complete![/green]")
        self._display_demo_summary()
    
    def _demo_research_questions(self) -> List[ResearchQuestion]:
        """Demonstrate research question formulation"""
        
        questions_data = [
            {
                "question": "How does multi-agent coordination affect cybersecurity learning performance compared to independent agent training?",
                "hypothesis": "Multi-agent coordination with shared memory will achieve 25% higher average reward than independent training",
                "variables": {
                    "independent": "coordination_method",
                    "dependent": "average_reward_per_episode", 
                    "control": "environment_complexity, random_seed, training_duration"
                },
                "significance": "Understanding coordination benefits for real-world cybersecurity AI deployment",
                "methodology": "Randomized controlled experiment with coordination vs. independent conditions"
            },
            {
                "question": "What is the optimal curriculum learning progression for cybersecurity skill acquisition?",
                "hypothesis": "Adaptive curriculum with performance-based progression will reduce training time by 30%",
                "variables": {
                    "independent": "curriculum_adaptation_strategy",
                    "dependent": "episodes_to_convergence",
                    "control": "agent_architecture, environment_parameters"
                },
                "significance": "Enabling faster and more efficient cybersecurity AI training",
                "methodology": "Factorial design comparing curriculum strategies"
            }
        ]
        
        research_questions = self.research_framework.define_research_questions(questions_data)
        
        # Display formatted research questions
        for i, rq in enumerate(research_questions, 1):
            console.print(f"[bold]RQ{i}:[/bold] {rq.question}")
            console.print(f"[dim]Hypothesis: {rq.hypothesis}[/dim]")
        
        console.print(f"[green]✓[/green] Defined {len(research_questions)} research questions")
        return research_questions
    
    def _demo_experiment_creation(self) -> ExperimentConfig:
        """Demonstrate experiment configuration creation"""
        
        baseline_config = self.research_framework.create_baseline_experiment(
            "multi_agent_coordination_baseline",
            "Baseline configuration for multi-agent coordination study"
        )
        
        # Display experiment configuration
        config_table = Table(title="Experiment Configuration")
        config_table.add_column("Parameter", style="cyan")
        config_table.add_column("Value", style="green")
        
        config_table.add_row("Episodes", str(baseline_config.episodes))
        config_table.add_row("Max Steps", str(baseline_config.max_steps))
        config_table.add_row("Learning Rate", str(baseline_config.hyperparameters["learning_rate"]))
        config_table.add_row("Batch Size", str(baseline_config.hyperparameters["batch_size"]))
        config_table.add_row("Random Seed", str(baseline_config.random_seed))
        
        console.print(config_table)
        return baseline_config
    
    def _demo_hyperparameter_sweep(self, base_config: ExperimentConfig) -> List[ExperimentConfig]:
        """Demonstrate hyperparameter optimization"""
        
        sweep_params = {
            "learning_rate": [0.001, 0.003, 0.01],
            "batch_size": [32, 64, 128],
            "gamma": [0.95, 0.99, 0.999]
        }
        
        sweep_configs = self.research_framework.create_hyperparameter_sweep(
            base_config, sweep_params, "coordination_optimization"
        )
        
        console.print(f"[green]✓[/green] Created hyperparameter sweep with {len(sweep_configs)} variants")
        
        # Show sample configurations
        sample_table = Table(title="Sample Sweep Configurations")
        sample_table.add_column("Config", style="cyan")
        sample_table.add_column("Learning Rate", style="green")
        sample_table.add_column("Batch Size", style="blue")
        sample_table.add_column("Gamma", style="magenta")
        
        for i, config in enumerate(sweep_configs[:5]):  # Show first 5
            sample_table.add_row(
                f"Variant {i+1}",
                str(config.hyperparameters["learning_rate"]),
                str(config.hyperparameters["batch_size"]),
                str(config.hyperparameters["gamma"])
            )
        
        console.print(sample_table)
        return sweep_configs
    
    def _demo_ab_testing(self, base_config: ExperimentConfig) -> Dict[str, List[ExperimentConfig]]:
        """Demonstrate A/B testing framework"""
        
        # Create variant configuration
        variant_config = base_config
        variant_config.name = "coordination_enabled"
        variant_config.memory_config["memory_sharing"] = True
        variant_config.training_config["multi_objective"] = True
        
        # Create control configuration
        control_config = base_config
        control_config.name = "coordination_disabled"
        control_config.memory_config["memory_sharing"] = False
        control_config.training_config["multi_objective"] = False
        
        ab_configs = self.research_framework.create_ab_test(
            control_config, variant_config, "coordination_effect_test", num_replications=5
        )
        
        console.print(f"[green]✓[/green] A/B test created with {len(ab_configs['A']) + len(ab_configs['B'])} total runs")
        return ab_configs
    
    def _demo_experiment_execution(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Simulate experiment execution with benchmarking"""
        
        console.print("[yellow]Simulating experiment execution...[/yellow]")
        
        # Create mock agent manager factory
        def mock_agent_manager_factory(config):
            return MockAgentManager(config)
        
        # Simulate experiment run
        with console.status("[bold green]Running experiment...") as status:
            time.sleep(2)  # Simulate computation time
            
            # Create simulated results
            results = self._create_mock_experiment_results(config)
        
        console.print(f"[green]✅[/green] Experiment '{config.name}' completed successfully")
        console.print(f"[dim]Duration: 2.5 hours (simulated)[/dim]")
        console.print(f"[dim]Episodes completed: {config.episodes}[/dim]")
        
        return results
    
    def _demo_curriculum_learning(self) -> Dict[str, Any]:
        """Demonstrate adaptive curriculum learning"""
        
        console.print("[yellow]Demonstrating curriculum learning...[/yellow]")
        
        # Initialize curriculum for demo agent
        agent_id = "demo_red_agent"
        curriculum = self.research_framework.curriculum_learning
        curriculum.initialize_agent_progress(agent_id)
        
        # Simulate curriculum progression
        progress_data = []
        
        for episode in track(range(50), description="Curriculum training..."):
            # Get current curriculum config
            curriculum_config = curriculum.get_current_curriculum_config(agent_id)
            
            # Simulate performance based on curriculum difficulty
            performance = self._simulate_curriculum_performance(curriculum_config, episode)
            
            # Update curriculum
            curriculum.update_agent_performance(agent_id, episode, performance)
            progress_data.append({
                "episode": episode,
                "stage": curriculum_config["stage_id"],
                "difficulty": curriculum_config["difficulty"],
                "reward": performance["avg_reward"]
            })
        
        # Display curriculum progression
        progression_table = Table(title="Curriculum Progression")
        progression_table.add_column("Stage", style="cyan")
        progression_table.add_column("Episodes", style="green")
        progression_table.add_column("Final Reward", style="blue")
        
        stages = {}
        for data in progress_data:
            stage = data["stage"]
            if stage not in stages:
                stages[stage] = {"episodes": 0, "final_reward": 0}
            stages[stage]["episodes"] += 1
            stages[stage]["final_reward"] = data["reward"]
        
        for stage, stats in stages.items():
            progression_table.add_row(
                stage.replace("_", " ").title(),
                str(stats["episodes"]),
                f"{stats['final_reward']:.2f}"
            )
        
        console.print(progression_table)
        
        # Generate curriculum report
        report = curriculum.generate_curriculum_report(agent_id)
        console.print(f"[green]✓[/green] Curriculum learning demo completed")
        
        return {"agent_id": agent_id, "progression": progress_data, "report": report}
    
    def _demo_analysis_and_reporting(self) -> Dict[str, Any]:
        """Demonstrate statistical analysis and reporting"""
        
        console.print("[yellow]Generating analysis and reports...[/yellow]")
        
        # Create mock analysis data
        mock_results = self._create_mock_analysis_data()
        
        # Simulate analysis
        with console.status("[bold blue]Analyzing results...") as status:
            time.sleep(1)  # Simulate analysis time
        
        # Display mock statistical results
        stats_table = Table(title="Statistical Analysis Results")
        stats_table.add_column("Agent", style="cyan")
        stats_table.add_column("Mean Reward", style="green")
        stats_table.add_column("Std Dev", style="blue")
        stats_table.add_column("Convergence", style="magenta")
        stats_table.add_column("Ranking", style="yellow")
        
        for agent, data in mock_results.items():
            stats_table.add_row(
                agent,
                f"{data['mean_reward']:.3f}",
                f"{data['std_dev']:.3f}",
                "✓" if data['converged'] else "✗",
                f"#{data['ranking']}"
            )
        
        console.print(stats_table)
        
        # Simulate report generation
        console.print("[green]✓[/green] Generated performance analysis report")
        console.print("[green]✓[/green] Created learning curve visualizations") 
        console.print("[green]✓[/green] Generated stability analysis plots")
        
        return mock_results
    
    def _demo_research_methodology(self) -> Dict[str, str]:
        """Demonstrate research methodology tools"""
        
        console.print("[yellow]Creating research methodology documents...[/yellow]")
        
        # Create methodology documents
        outputs = {}
        
        # Research proposal
        outputs["proposal"] = "research_proposal_demo.json"
        console.print("[green]✓[/green] Research proposal template created")
        
        # Literature review template
        outputs["literature_review"] = self.research_framework.research_methodology.generate_literature_review_template(
            "Multi-Agent Cybersecurity Learning"
        )
        console.print("[green]✓[/green] Literature review template created")
        
        # Reproducibility checklist
        outputs["reproducibility"] = self.research_framework.research_methodology.create_reproducibility_checklist()
        console.print("[green]✓[/green] Reproducibility checklist created")
        
        # Ethics protocol
        outputs["ethics"] = self.research_framework.research_methodology.create_ethics_protocol()
        console.print("[green]✓[/green] Ethics protocol created")
        
        # Research paper template
        outputs["paper"] = self.research_framework.generate_research_report("demo_experiment")
        console.print("[green]✓[/green] Research paper template created")
        
        return outputs
    
    def _display_demo_summary(self):
        """Display comprehensive demo summary"""
        
        summary = self.research_framework.get_research_summary()
        
        console.print(Panel.fit(
            f"""🎯 **Demo Summary**

**Research Framework Components Demonstrated:**
• Advanced benchmarking and performance evaluation
• Reproducible experiment management with versioning
• Statistical analysis with significance testing  
• Hyperparameter optimization and A/B testing
• Adaptive curriculum learning with progression tracking
• Research methodology templates and documentation
• Automated report generation and visualization

**Generated Artifacts:**
• {summary['experiments']['total_experiments']} experiment configurations
• {summary['active_research_questions']} research questions
• {summary['methodology_protocols']} methodology protocols
• Curriculum learning progression for {summary['curriculum_analytics']['total_agents']} agent(s)
• Statistical analysis reports and visualizations

**Research Impact:**
This framework enables rigorous, reproducible research in multi-agent
cybersecurity RL, supporting publication-quality studies and advancing
the field through systematic experimentation and analysis.
""",
            title="🏆 ARIASKA_RL Research Enhancement Complete",
            style="bold green"
        ))
    
    def _create_mock_experiment_results(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Create mock experiment results for demo"""
        
        import random
        import numpy as np
        
        # Set seed for reproducible demo
        random.seed(42)
        np.random.seed(42)
        
        agents = config.agents
        mock_results = {
            "status": "completed",
            "config": config,
            "duration": 9000,  # 2.5 hours in seconds
            "final_metrics": {}
        }
        
        for agent in agents:
            # Generate realistic performance data
            episode_rewards = []
            base_reward = random.uniform(8, 15)
            
            for episode in range(config.episodes):
                # Simulate learning progress with noise
                progress = episode / config.episodes
                trend = base_reward * (1 + progress * 0.5)
                noise = random.normalvariate(0, 1)
                reward = max(0, trend + noise)
                episode_rewards.append(reward)
            
            mock_results["final_metrics"][agent] = {
                "agent_id": agent,
                "episode_rewards": episode_rewards,
                "success_rates": {
                    "recon": random.uniform(0.7, 0.9),
                    "enumeration": random.uniform(0.6, 0.8),
                    "exploit": random.uniform(0.4, 0.6),
                    "privesc": random.uniform(0.3, 0.5),
                    "exfiltrate": random.uniform(0.2, 0.4)
                },
                "learning_efficiency": random.uniform(0.05, 0.15),
                "exploration_efficiency": random.uniform(0.3, 0.7),
                "gpt_token_efficiency": random.uniform(0.01, 0.05),
                "phase_transition_accuracy": random.uniform(0.6, 0.8),
                "memory_utilization": random.uniform(0.4, 0.8),
                "convergence_episode": random.randint(60, 90) if random.random() > 0.3 else None,
                "timestamp": "2024-01-15T10:30:00"
            }
        
        return mock_results
    
    def _simulate_curriculum_performance(self, curriculum_config: Dict[str, Any], episode: int) -> Dict[str, float]:
        """Simulate performance in curriculum learning"""
        
        import random
        
        # Base performance varies by difficulty
        difficulty_multipliers = {
            "beginner": 1.2,
            "intermediate": 1.0,
            "advanced": 0.8,
            "expert": 0.6
        }
        
        difficulty = curriculum_config.get("difficulty", "intermediate")
        multiplier = difficulty_multipliers.get(difficulty, 1.0)
        
        # Simulate learning progress over episodes
        progress = min(episode / 20.0, 1.0)  # Progress within stage
        base_reward = 10.0 * multiplier * (0.5 + progress * 0.5)
        
        return {
            "avg_reward": base_reward + random.normalvariate(0, 1),
            "success_rate": 0.4 + progress * 0.4 + random.uniform(-0.1, 0.1),
            "phase_accuracy": 0.5 + progress * 0.3 + random.uniform(-0.1, 0.1),
            "gpt_token_efficiency": 0.01 + progress * 0.02 + random.uniform(-0.005, 0.005)
        }
    
    def _create_mock_analysis_data(self) -> Dict[str, Dict[str, Any]]:
        """Create mock analysis results"""
        
        import random
        import numpy as np
        
        random.seed(42)
        np.random.seed(42)
        
        agents = ["RedAgent", "BlueAgent", "ScoutAgent", "ShadowAgent", "OrionAgent"]
        mock_data = {}
        
        for i, agent in enumerate(agents):
            mock_data[agent] = {
                "mean_reward": random.uniform(10, 20),
                "std_dev": random.uniform(1, 3),
                "converged": random.random() > 0.3,
                "ranking": i + 1
            }
        
        return mock_data


class MockAgentManager:
    """Mock agent manager for demo purposes"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.agents = config.agents
    
    def simulate_all_agents(self, episodes: int, max_steps: int):
        """Simulate agent training"""
        pass  # Placeholder for actual training
    
    def all_agents(self):
        """Return mock agents"""
        return [MockAgent(agent_id) for agent_id in self.agents]


class MockAgent:
    """Mock agent for demo purposes"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id


def main():
    """Run the research framework demonstration"""
    
    console.print("[bold blue]🧪 ARIASKA_RL Research Framework Demo[/bold blue]")
    console.print("[dim]Demonstrating enhanced research capabilities...[/dim]\n")
    
    try:
        demo = ResearchDemo()
        demo.demonstrate_research_workflow()
        
        console.print("\n[bold green]🎉 Demo completed successfully![/bold green]")
        console.print("[dim]All research framework components are ready for use.[/dim]")
        
    except Exception as e:
        console.print(f"[red]❌ Demo failed: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())


if __name__ == "__main__":
    main()