"""
Research Integration Layer for ARIASKA_RL

Integrates advanced research tools with the existing ARIASKA_RL system,
providing enhanced experiment management, analysis, and methodology.
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Import research components
from .benchmark_suite import BenchmarkSuite, ExperimentConfig as BenchmarkExperimentConfig
from .experiment_manager import ExperimentManager, ExperimentConfig
from .metrics_analyzer import MetricsAnalyzer
from .research_methodology import ResearchMethodology, ResearchQuestion
from .curriculum_learning import CurriculumLearning

console = Console()

class ResearchIntegration:
    """
    Main integration layer for ARIASKA_RL research enhancements.
    
    Coordinates between:
    - Benchmark suite for performance evaluation
    - Experiment manager for reproducible research
    - Metrics analyzer for statistical analysis
    - Research methodology for rigorous practices
    - Curriculum learning for adaptive training
    """
    
    def __init__(self, base_dir: str = "research_workspace"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        
        # Initialize components
        self.benchmark_suite = BenchmarkSuite(f"{base_dir}/benchmarks")
        self.experiment_manager = ExperimentManager(f"{base_dir}/experiments") 
        self.metrics_analyzer = MetricsAnalyzer(f"{base_dir}/analysis")
        self.research_methodology = ResearchMethodology(f"{base_dir}/methodology")
        self.curriculum_learning = CurriculumLearning(f"{base_dir}/curriculum")
        
        # Research session state
        self.current_experiment: Optional[str] = None
        self.active_research_questions: List[ResearchQuestion] = []
        
        console.print(f"[green]🧪[/green] ARIASKA_RL Research Integration initialized at {base_dir}")
        self._display_welcome_message()
    
    def _display_welcome_message(self):
        """Display welcome message with available research capabilities"""
        
        welcome_text = """
🧬 **ARIASKA_RL Research Framework**

**Enhanced Capabilities:**
• Advanced benchmarking and performance evaluation
• Reproducible experiment management with versioning  
• Statistical analysis and significance testing
• Research methodology templates and protocols
• Adaptive curriculum learning with difficulty progression
• Automated report generation and visualization

**Quick Start:**
1. Define research questions: `define_research_questions()`
2. Create experiment: `create_baseline_experiment()`
3. Run with benchmarking: `run_experiment_with_analysis()`
4. Generate insights: `analyze_and_report()`

Type `help_research()` for detailed command reference.
"""
        
        panel = Panel(welcome_text, title="Research Framework Ready", style="green")
        console.print(panel)
    
    def help_research(self):
        """Display comprehensive research framework help"""
        
        help_table = Table(title="🔬 Research Framework Commands")
        help_table.add_column("Category", style="cyan")
        help_table.add_column("Command", style="green")
        help_table.add_column("Description", style="white")
        
        commands = [
            ("Planning", "define_research_questions()", "Define structured research questions"),
            ("Planning", "create_research_proposal()", "Generate research proposal document"),
            ("Experiments", "create_baseline_experiment()", "Create baseline configuration"),
            ("Experiments", "create_hyperparameter_sweep()", "Design parameter optimization"),
            ("Experiments", "create_ab_test()", "Setup A/B comparison"),
            ("Execution", "run_experiment_with_analysis()", "Run experiment with full analysis"),
            ("Execution", "run_curriculum_training()", "Run adaptive curriculum training"),
            ("Analysis", "analyze_experiment_results()", "Statistical analysis of results"),
            ("Analysis", "compare_experiments()", "Compare multiple experiments"),
            ("Reporting", "generate_research_report()", "Create comprehensive report"),
            ("Reporting", "create_visualizations()", "Generate research plots"),
            ("Methodology", "create_reproducibility_checklist()", "Ensure reproducible research"),
            ("Methodology", "generate_paper_template()", "Create research paper template")
        ]
        
        for category, command, description in commands:
            help_table.add_row(category, command, description)
        
        console.print(help_table)
    
    def define_research_questions(self, questions_data: List[Dict[str, str]]) -> List[ResearchQuestion]:
        """Define structured research questions for the study"""
        
        research_questions = []
        for i, q_data in enumerate(questions_data):
            rq = ResearchQuestion(
                id=f"RQ{i+1}",
                question=q_data.get("question", ""),
                hypothesis=q_data.get("hypothesis", ""),
                variables=q_data.get("variables", {}),
                significance=q_data.get("significance", ""),
                methodology=q_data.get("methodology", "")
            )
            research_questions.append(rq)
        
        self.active_research_questions = research_questions
        
        # Generate research proposal
        if research_questions:
            title = "ARIASKA_RL_Multi_Agent_Study"
            proposal_path = self.research_methodology.create_research_proposal(title, research_questions)
            console.print(f"[cyan]📋[/cyan] Research proposal created: {proposal_path}")
        
        return research_questions
    
    def create_baseline_experiment(self, name: str, description: str = "") -> ExperimentConfig:
        """Create baseline experiment configuration"""
        
        config = self.experiment_manager.create_baseline_config(name, description)
        config_hash = self.experiment_manager.create_experiment(config)
        
        console.print(f"[green]✅[/green] Baseline experiment created: {name}")
        console.print(f"[dim]Configuration hash: {config_hash}[/dim]")
        
        return config
    
    def create_hyperparameter_sweep(self, base_config: ExperimentConfig, 
                                  sweep_params: Dict[str, List[Any]], 
                                  sweep_name: str) -> List[ExperimentConfig]:
        """Create hyperparameter sweep for optimization"""
        
        sweep_configs = self.experiment_manager.create_hyperparameter_sweep(
            base_config, sweep_params, sweep_name
        )
        
        console.print(f"[magenta]🔄[/magenta] Hyperparameter sweep created: {sweep_name}")
        console.print(f"[dim]Generated {len(sweep_configs)} experiment variants[/dim]")
        
        return sweep_configs
    
    def create_ab_test(self, config_a: ExperimentConfig, config_b: ExperimentConfig,
                      test_name: str, num_replications: int = 5) -> Dict[str, List[ExperimentConfig]]:
        """Create A/B test with statistical power"""
        
        ab_configs = self.experiment_manager.create_ab_test(
            config_a, config_b, test_name, num_replications
        )
        
        console.print(f"[blue]⚖️[/blue] A/B test created: {test_name}")
        console.print(f"[dim]Total configurations: {len(ab_configs['A']) + len(ab_configs['B'])}[/dim]")
        
        return ab_configs
    
    def run_experiment_with_analysis(self, config: ExperimentConfig, 
                                   agent_manager_factory: callable) -> Dict[str, Any]:
        """Run experiment with comprehensive analysis and benchmarking"""
        
        console.print(f"[cyan]🚀[/cyan] Starting research experiment: {config.name}")
        
        # Register experiment configuration for benchmarking
        benchmark_config = BenchmarkExperimentConfig(
            experiment_name=config.name,
            agents=config.agents,
            episodes=config.episodes,
            max_steps=config.max_steps,
            environment_type=config.environment_type,
            learning_rate=config.hyperparameters.get("learning_rate", 0.001),
            batch_size=config.hyperparameters.get("batch_size", 64),
            gamma=config.hyperparameters.get("gamma", 0.99),
            epsilon_decay=config.hyperparameters.get("epsilon_decay", 0.995),
            gpt_models=config.gpt_config,
            random_seed=config.random_seed,
            curriculum_enabled=config.training_config.get("curriculum_learning", False),
            memory_sharing=config.memory_config.get("memory_sharing", True),
            timestamp=datetime.now().isoformat()
        )
        
        experiment_id = self.benchmark_suite.register_experiment(benchmark_config)
        
        # Run experiment with benchmarking
        results = self.experiment_manager.run_experiment(
            config, agent_manager_factory, self.benchmark_suite
        )
        
        # Analyze results if successful
        if results['status'] == 'completed' and 'final_metrics' in results:
            # Convert to BenchmarkMetrics format for analysis
            benchmark_metrics = {}
            for agent_id, metrics_data in results['final_metrics'].items():
                from .benchmark_suite import BenchmarkMetrics
                benchmark_metrics[agent_id] = BenchmarkMetrics(**metrics_data)
            
            # Generate performance analysis
            analysis = self.metrics_analyzer.analyze_learning_curves(results['final_metrics'])
            
            # Create visualizations
            vis_path = self.metrics_analyzer.create_learning_curve_visualization(
                analysis, config.name
            )
            stability_path = self.metrics_analyzer.create_stability_analysis_visualization(
                analysis, config.name  
            )
            
            # Generate comprehensive report
            report_path = self.benchmark_suite.generate_report(config.name, benchmark_metrics)
            analysis_path = self.metrics_analyzer.generate_performance_report(analysis, config.name)
            
            # Display real-time results
            self.benchmark_suite.display_realtime_metrics(benchmark_metrics)
            self.metrics_analyzer.display_realtime_analysis(analysis)
            
            console.print(f"[green]📊[/green] Analysis complete - Reports generated:")
            console.print(f"  • Benchmark report: {report_path}")
            console.print(f"  • Performance analysis: {analysis_path}")
            console.print(f"  • Learning curves: {vis_path}")
            console.print(f"  • Stability analysis: {stability_path}")
        
        self.current_experiment = config.name
        return results
    
    def run_curriculum_training(self, agent_manager, agent_id: str, 
                              total_episodes: int = 200) -> Dict[str, Any]:
        """Run adaptive curriculum learning training"""
        
        console.print(f"[green]📚[/green] Starting curriculum training for {agent_id}")
        
        # Initialize curriculum progress
        self.curriculum_learning.initialize_agent_progress(agent_id)
        
        results = {
            "agent_id": agent_id,
            "total_episodes": total_episodes,
            "curriculum_stages": [],
            "performance_progression": [],
            "adaptations": []
        }
        
        for episode in range(total_episodes):
            # Get current curriculum configuration
            curriculum_config = self.curriculum_learning.get_current_curriculum_config(agent_id)
            
            # Apply curriculum configuration to environment
            # This would integrate with the actual environment configuration
            # For now, we'll simulate the training step
            
            # Simulate training episode with curriculum
            episode_metrics = self._simulate_curriculum_episode(agent_manager, agent_id, curriculum_config)
            
            # Update curriculum based on performance
            self.curriculum_learning.update_agent_performance(agent_id, episode, episode_metrics)
            
            # Track progress
            results["performance_progression"].append(episode_metrics)
            
            # Periodic reporting
            if (episode + 1) % 20 == 0:
                progress_report = self.curriculum_learning.generate_curriculum_report(agent_id)
                console.print(f"[cyan]Episode {episode + 1}[/cyan] - Curriculum Progress:")
                console.print(f"Stage: {curriculum_config['stage_id']}")
                console.print(f"Recent Reward: {episode_metrics.get('avg_reward', 0):.3f}")
        
        # Generate final curriculum report
        final_report = self.curriculum_learning.generate_curriculum_report(agent_id)
        
        console.print(f"[green]✅[/green] Curriculum training completed for {agent_id}")
        
        return results
    
    def _simulate_curriculum_episode(self, agent_manager, agent_id: str, 
                                   curriculum_config: Dict[str, Any]) -> Dict[str, float]:
        """Simulate curriculum training episode (placeholder for actual integration)"""
        
        # This would integrate with actual agent training
        # For now, return simulated metrics based on curriculum difficulty
        
        difficulty_factor = {
            "beginner": 1.2,
            "intermediate": 1.0,
            "advanced": 0.8,
            "expert": 0.6
        }
        
        base_reward = 10.0
        difficulty = curriculum_config.get("difficulty", "intermediate")
        factor = difficulty_factor.get(difficulty, 1.0)
        
        # Simulate performance with some randomness
        import random
        performance = {
            "avg_reward": base_reward * factor * (0.8 + random.random() * 0.4),
            "success_rate": 0.5 + random.random() * 0.4,
            "phase_accuracy": 0.6 + random.random() * 0.3,
            "gpt_token_efficiency": 0.01 + random.random() * 0.02
        }
        
        return performance
    
    def analyze_experiment_results(self, experiment_name: str) -> Dict[str, Any]:
        """Analyze results from completed experiment"""
        
        # Load experiment results
        results_files = [f for f in os.listdir(f"{self.base_dir}/experiments/results") 
                        if experiment_name in f and f.endswith('_results.json')]
        
        if not results_files:
            console.print(f"[red]❌[/red] No results found for experiment: {experiment_name}")
            return {}
        
        # Load and analyze most recent results
        latest_results = sorted(results_files)[-1]
        results_path = f"{self.base_dir}/experiments/results/{latest_results}"
        
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        if 'final_metrics' not in results:
            console.print(f"[yellow]⚠[/yellow] No metrics found in results")
            return {}
        
        # Perform comprehensive analysis
        analysis = self.metrics_analyzer.analyze_learning_curves(results['final_metrics'])
        
        # Generate visualizations
        self.metrics_analyzer.create_learning_curve_visualization(analysis, experiment_name)
        self.metrics_analyzer.create_stability_analysis_visualization(analysis, experiment_name)
        
        # Generate analysis report
        report_path = self.metrics_analyzer.generate_performance_report(analysis, experiment_name)
        
        console.print(f"[green]📊[/green] Analysis completed for {experiment_name}")
        console.print(f"Report generated: {report_path}")
        
        return {"analysis": analysis, "report_path": report_path}
    
    def compare_experiments(self, experiment_names: List[str]) -> Dict[str, Any]:
        """Compare results across multiple experiments"""
        
        comparison = self.benchmark_suite.compare_experiments(experiment_names)
        
        # Create comparison table
        table = Table(title="🔄 Experiment Comparison")
        table.add_column("Experiment", style="cyan")
        table.add_column("Avg Reward", style="green")
        table.add_column("Success Rate", style="blue") 
        table.add_column("Learning Efficiency", style="magenta")
        
        for exp_name in experiment_names:
            metrics = comparison["comparative_metrics"]
            avg_reward = metrics.get("avg_reward", {}).get(exp_name, {}).get("mean", 0)
            success_rate = metrics.get("overall_success_rate", {}).get(exp_name, {}).get("mean", 0)
            learning_eff = metrics.get("learning_efficiency", {}).get(exp_name, {}).get("mean", 0)
            
            table.add_row(
                exp_name,
                f"{avg_reward:.3f}",
                f"{success_rate:.1%}",
                f"{learning_eff:.4f}"
            )
        
        console.print(table)
        
        # Display recommendations
        if comparison.get("recommendations"):
            recs_text = "\n".join([f"• {rec}" for rec in comparison["recommendations"]])
            panel = Panel(recs_text, title="🎯 Recommendations", style="blue")
            console.print(panel)
        
        return comparison
    
    def generate_research_report(self, experiment_name: str) -> str:
        """Generate comprehensive research report"""
        
        if not self.active_research_questions:
            console.print(f"[yellow]⚠[/yellow] No research questions defined. Creating template...")
            template_questions = [
                {
                    "question": "How does multi-agent coordination affect learning performance?",
                    "hypothesis": "Coordinated multi-agent systems will achieve higher performance than individual agents",
                    "variables": {"independent": "coordination_method", "dependent": "learning_performance"},
                    "significance": "Understanding coordination benefits for cybersecurity AI",
                    "methodology": "Controlled experiment with coordination vs. individual training"
                }
            ]
            self.define_research_questions(template_questions)
        
        # Generate research paper template
        if self.active_research_questions:
            # Create experimental design
            experimental_design = self.research_methodology.design_experiment(
                self.active_research_questions[0]
            )
            
            # Generate complete paper template
            paper_path = self.research_methodology.generate_research_paper_template(
                f"ARIASKA_RL Multi-Agent Cybersecurity Learning: {experiment_name}",
                ["Research Team"],
                self.active_research_questions,
                experimental_design
            )
            
            console.print(f"[green]📄[/green] Research paper template generated: {paper_path}")
            return paper_path
        
        return ""
    
    def create_reproducibility_package(self, experiment_name: str) -> str:
        """Create complete reproducibility package"""
        
        package_dir = f"{self.base_dir}/reproducibility_packages/{experiment_name}"
        os.makedirs(package_dir, exist_ok=True)
        
        # Create reproducibility checklist
        checklist_path = self.research_methodology.create_reproducibility_checklist()
        
        # Create ethics protocol
        ethics_path = self.research_methodology.create_ethics_protocol()
        
        # Copy relevant files
        import shutil
        shutil.copy(checklist_path, package_dir)
        shutil.copy(ethics_path, package_dir)
        
        # Create README for package
        readme_content = f"""# Reproducibility Package: {experiment_name}

## Contents
- `reproducibility_checklist.md` - Ensure all requirements are met
- `ethics_protocol.md` - Ethical considerations and approval
- `experiment_config.json` - Complete experimental configuration
- `requirements.txt` - Software dependencies with versions

## Instructions
1. Review and complete the reproducibility checklist
2. Verify ethical approval if required
3. Install dependencies: `pip install -r requirements.txt`
4. Run experiment: Follow instructions in experiment configuration

## Contact
For questions about reproducing this research, contact the research team.
"""
        
        with open(f"{package_dir}/README.md", 'w') as f:
            f.write(readme_content)
        
        console.print(f"[green]📦[/green] Reproducibility package created: {package_dir}")
        return package_dir
    
    def get_research_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of research activities"""
        
        summary = {
            "experiments": self.experiment_manager.get_experiment_summary(),
            "benchmark_history": len(self.benchmark_suite.benchmark_history),
            "active_research_questions": len(self.active_research_questions),
            "current_experiment": self.current_experiment,
            "curriculum_analytics": self.curriculum_learning.get_curriculum_analytics(),
            "methodology_protocols": len(os.listdir(self.research_methodology.protocols_dir)),
            "generated_reports": len([f for f in os.listdir(f"{self.base_dir}/analysis/reports") 
                                    if f.endswith('.md')]) if os.path.exists(f"{self.base_dir}/analysis/reports") else 0
        }
        
        # Display summary
        summary_table = Table(title="🔬 Research Framework Summary")
        summary_table.add_column("Category", style="cyan")
        summary_table.add_column("Count/Status", style="green")
        
        summary_table.add_row("Total Experiments", str(summary["experiments"]["total_experiments"]))
        summary_table.add_row("Benchmark Runs", str(summary["benchmark_history"]))
        summary_table.add_row("Research Questions", str(summary["active_research_questions"]))
        summary_table.add_row("Current Experiment", summary["current_experiment"] or "None")
        summary_table.add_row("Curriculum Agents", str(summary["curriculum_analytics"]["total_agents"]))
        summary_table.add_row("Generated Reports", str(summary["generated_reports"]))
        
        console.print(summary_table)
        
        return summary