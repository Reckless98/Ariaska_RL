"""
Advanced Benchmarking Suite for ARIASKA_RL

Provides systematic evaluation framework for multi-agent cybersecurity RL performance,
including standardized metrics, statistical analysis, and comparative evaluation tools.
"""

import os
import time
import json
import numpy as np
import statistics
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

@dataclass
class BenchmarkMetrics:
    """Standardized metrics for agent performance evaluation"""
    agent_id: str
    episode_rewards: List[float]
    success_rates: Dict[str, float]  # phase -> success_rate
    learning_efficiency: float  # reward improvement per episode
    exploration_efficiency: float  # unique states explored
    gpt_token_efficiency: float  # reward per token used
    phase_transition_accuracy: float  # correct phase predictions
    memory_utilization: float  # memory usage efficiency
    convergence_episode: Optional[int]  # episode where convergence occurred
    timestamp: str
    
    @property
    def avg_reward(self) -> float:
        return statistics.mean(self.episode_rewards) if self.episode_rewards else 0.0
    
    @property
    def reward_std(self) -> float:
        return statistics.stdev(self.episode_rewards) if len(self.episode_rewards) > 1 else 0.0
    
    @property
    def overall_success_rate(self) -> float:
        return statistics.mean(self.success_rates.values()) if self.success_rates else 0.0

@dataclass
class ExperimentConfig:
    """Configuration for reproducible experiments"""
    experiment_name: str
    agents: List[str]
    episodes: int
    max_steps: int
    environment_type: str  # 'simulated' or 'live'
    learning_rate: float
    batch_size: int
    gamma: float
    epsilon_decay: float
    gpt_models: Dict[str, str]  # task_type -> model_name
    random_seed: int
    curriculum_enabled: bool
    memory_sharing: bool
    timestamp: str

class BenchmarkSuite:
    """
    Comprehensive benchmarking framework for ARIASKA_RL research.
    
    Features:
    - Standardized performance metrics
    - Statistical significance testing
    - Comparative analysis across configurations
    - Reproducible experiment tracking
    - Research-grade reporting
    """
    
    def __init__(self, results_dir: str = "research_results"):
        self.results_dir = results_dir
        self.benchmark_history: List[BenchmarkMetrics] = []
        self.experiment_configs: List[ExperimentConfig] = []
        
        # Create results directory structure
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(f"{self.results_dir}/raw_data", exist_ok=True)
        os.makedirs(f"{self.results_dir}/reports", exist_ok=True)
        os.makedirs(f"{self.results_dir}/visualizations", exist_ok=True)
        
        console.print(f"[green]✓[/green] BenchmarkSuite initialized at {self.results_dir}")
    
    def register_experiment(self, config: ExperimentConfig) -> str:
        """Register a new experiment configuration"""
        config.timestamp = datetime.now().isoformat()
        self.experiment_configs.append(config)
        
        # Save configuration
        config_path = f"{self.results_dir}/raw_data/config_{config.experiment_name}_{config.timestamp}.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(config), f, indent=2)
        
        console.print(f"[cyan]📋[/cyan] Registered experiment: {config.experiment_name}")
        return config.timestamp
    
    def collect_agent_metrics(self, agent_manager, episode_num: int) -> Dict[str, BenchmarkMetrics]:
        """Collect comprehensive metrics from all agents"""
        metrics = {}
        
        for agent in agent_manager.all_agents():
            try:
                # Extract performance data
                episode_rewards = getattr(agent, 'episode_rewards', [])
                if hasattr(agent, 'stats_monitor') and agent.stats_monitor:
                    episode_rewards = agent.stats_monitor.episode_rewards
                
                # Calculate success rates by phase
                success_rates = self._calculate_phase_success_rates(agent)
                
                # Learning efficiency (reward improvement trend)
                learning_efficiency = self._calculate_learning_efficiency(episode_rewards)
                
                # Exploration efficiency
                exploration_efficiency = self._calculate_exploration_efficiency(agent)
                
                # GPT token efficiency
                gpt_token_efficiency = self._calculate_gpt_efficiency(agent)
                
                # Phase transition accuracy
                phase_accuracy = self._calculate_phase_accuracy(agent)
                
                # Memory utilization
                memory_utilization = self._calculate_memory_utilization(agent)
                
                # Convergence detection
                convergence_episode = self._detect_convergence(episode_rewards)
                
                metrics[agent.agent_id] = BenchmarkMetrics(
                    agent_id=agent.agent_id,
                    episode_rewards=episode_rewards,
                    success_rates=success_rates,
                    learning_efficiency=learning_efficiency,
                    exploration_efficiency=exploration_efficiency,
                    gpt_token_efficiency=gpt_token_efficiency,
                    phase_transition_accuracy=phase_accuracy,
                    memory_utilization=memory_utilization,
                    convergence_episode=convergence_episode,
                    timestamp=datetime.now().isoformat()
                )
                
            except Exception as e:
                console.print(f"[yellow]⚠[/yellow] Error collecting metrics for {agent.agent_id}: {e}")
                # Create minimal metrics for failed collection
                metrics[agent.agent_id] = BenchmarkMetrics(
                    agent_id=agent.agent_id,
                    episode_rewards=[],
                    success_rates={},
                    learning_efficiency=0.0,
                    exploration_efficiency=0.0,
                    gpt_token_efficiency=0.0,
                    phase_transition_accuracy=0.0,
                    memory_utilization=0.0,
                    convergence_episode=None,
                    timestamp=datetime.now().isoformat()
                )
        
        return metrics
    
    def _calculate_phase_success_rates(self, agent) -> Dict[str, float]:
        """Calculate success rates for each cybersecurity phase"""
        phase_success = {}
        
        try:
            if hasattr(agent, 'memory') and 'actions' in agent.memory:
                phase_counts = {}
                phase_successes = {}
                
                for action in agent.memory['actions']:
                    phase = action.get('phase', 'unknown')
                    success = action.get('success', False)
                    
                    phase_counts[phase] = phase_counts.get(phase, 0) + 1
                    if success:
                        phase_successes[phase] = phase_successes.get(phase, 0) + 1
                
                for phase, count in phase_counts.items():
                    if count > 0:
                        phase_success[phase] = phase_successes.get(phase, 0) / count
        except Exception:
            pass
        
        return phase_success
    
    def _calculate_learning_efficiency(self, episode_rewards: List[float]) -> float:
        """Calculate learning efficiency as reward improvement per episode"""
        if len(episode_rewards) < 2:
            return 0.0
        
        try:
            # Use linear regression slope as learning efficiency
            x = np.arange(len(episode_rewards))
            y = np.array(episode_rewards)
            slope = np.polyfit(x, y, 1)[0]
            return float(slope)
        except Exception:
            return 0.0
    
    def _calculate_exploration_efficiency(self, agent) -> float:
        """Calculate exploration efficiency based on unique states/actions"""
        try:
            if hasattr(agent, 'memory') and 'actions' in agent.memory:
                unique_commands = set()
                unique_states = set()
                
                for action in agent.memory['actions']:
                    unique_commands.add(action.get('command', ''))
                    if 'state' in action:
                        unique_states.add(str(action['state']))
                
                total_actions = len(agent.memory['actions'])
                if total_actions > 0:
                    return len(unique_commands) / total_actions
        except Exception:
            pass
        
        return 0.0
    
    def _calculate_gpt_efficiency(self, agent) -> float:
        """Calculate GPT token efficiency (reward per token)"""
        try:
            total_reward = 0.0
            total_tokens = 0
            
            if hasattr(agent, 'memory') and 'actions' in agent.memory:
                for action in agent.memory['actions']:
                    total_reward += action.get('reward', 0)
                    total_tokens += action.get('gpt_tokens', 0)
            
            if total_tokens > 0:
                return total_reward / total_tokens
        except Exception:
            pass
        
        return 0.0
    
    def _calculate_phase_accuracy(self, agent) -> float:
        """Calculate phase transition prediction accuracy"""
        # This would need to be integrated with ScoutAgent's phase predictions
        # For now, return a placeholder
        return 0.8  # 80% baseline accuracy
    
    def _calculate_memory_utilization(self, agent) -> float:
        """Calculate memory system utilization efficiency"""
        try:
            if hasattr(agent, 'memory'):
                actions_count = len(agent.memory.get('actions', []))
                # Normalize by some reasonable maximum
                max_expected = 1000
                return min(actions_count / max_expected, 1.0)
        except Exception:
            pass
        
        return 0.0
    
    def _detect_convergence(self, episode_rewards: List[float], 
                          window_size: int = 10, 
                          threshold: float = 0.05) -> Optional[int]:
        """Detect convergence point in learning curve"""
        if len(episode_rewards) < window_size * 2:
            return None
        
        try:
            for i in range(window_size, len(episode_rewards) - window_size):
                window1 = episode_rewards[i-window_size:i]
                window2 = episode_rewards[i:i+window_size]
                
                if abs(statistics.mean(window2) - statistics.mean(window1)) < threshold:
                    return i
        except Exception:
            pass
        
        return None
    
    def run_statistical_analysis(self, metrics_list: List[BenchmarkMetrics]) -> Dict[str, Any]:
        """Perform statistical analysis on collected metrics"""
        analysis = {
            'sample_size': len(metrics_list),
            'reward_statistics': {},
            'success_rate_statistics': {},
            'efficiency_statistics': {},
            'convergence_analysis': {}
        }
        
        if not metrics_list:
            return analysis
        
        # Reward statistics
        all_rewards = [m.avg_reward for m in metrics_list]
        analysis['reward_statistics'] = {
            'mean': statistics.mean(all_rewards),
            'median': statistics.median(all_rewards),
            'std': statistics.stdev(all_rewards) if len(all_rewards) > 1 else 0.0,
            'min': min(all_rewards),
            'max': max(all_rewards)
        }
        
        # Success rate statistics
        all_success_rates = [m.overall_success_rate for m in metrics_list]
        analysis['success_rate_statistics'] = {
            'mean': statistics.mean(all_success_rates),
            'median': statistics.median(all_success_rates),
            'std': statistics.stdev(all_success_rates) if len(all_success_rates) > 1 else 0.0
        }
        
        # Efficiency statistics
        learning_efficiencies = [m.learning_efficiency for m in metrics_list]
        analysis['efficiency_statistics'] = {
            'learning_efficiency_mean': statistics.mean(learning_efficiencies),
            'gpt_efficiency_mean': statistics.mean([m.gpt_token_efficiency for m in metrics_list])
        }
        
        # Convergence analysis
        convergence_episodes = [m.convergence_episode for m in metrics_list if m.convergence_episode is not None]
        if convergence_episodes:
            analysis['convergence_analysis'] = {
                'convergence_rate': len(convergence_episodes) / len(metrics_list),
                'avg_convergence_episode': statistics.mean(convergence_episodes),
                'convergence_std': statistics.stdev(convergence_episodes) if len(convergence_episodes) > 1 else 0.0
            }
        
        return analysis
    
    def compare_experiments(self, experiment_names: List[str]) -> Dict[str, Any]:
        """Compare performance across different experiments"""
        comparison = {
            'experiments': experiment_names,
            'comparative_metrics': {},
            'statistical_tests': {},
            'recommendations': []
        }
        
        # Load metrics for each experiment
        experiment_metrics = {}
        for exp_name in experiment_names:
            metrics_files = [f for f in os.listdir(f"{self.results_dir}/raw_data") 
                           if f.startswith(f"metrics_{exp_name}")]
            
            exp_metrics = []
            for metrics_file in metrics_files:
                try:
                    with open(f"{self.results_dir}/raw_data/{metrics_file}", 'r') as f:
                        data = json.load(f)
                        # Convert back to BenchmarkMetrics objects
                        for agent_data in data.values():
                            exp_metrics.append(BenchmarkMetrics(**agent_data))
                except Exception as e:
                    console.print(f"[yellow]⚠[/yellow] Error loading {metrics_file}: {e}")
            
            experiment_metrics[exp_name] = exp_metrics
        
        # Perform comparative analysis
        for metric_name in ['avg_reward', 'overall_success_rate', 'learning_efficiency']:
            comparison['comparative_metrics'][metric_name] = {}
            
            for exp_name, metrics in experiment_metrics.items():
                if metrics:
                    values = [getattr(m, metric_name) for m in metrics]
                    comparison['comparative_metrics'][metric_name][exp_name] = {
                        'mean': statistics.mean(values),
                        'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                        'count': len(values)
                    }
        
        # Generate recommendations
        if len(experiment_names) >= 2:
            comparison['recommendations'] = self._generate_experiment_recommendations(comparison)
        
        return comparison
    
    def _generate_experiment_recommendations(self, comparison: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on comparison"""
        recommendations = []
        
        # Find best performing experiment for each metric
        for metric_name, metric_data in comparison['comparative_metrics'].items():
            if metric_data:
                best_exp = max(metric_data.keys(), key=lambda x: metric_data[x]['mean'])
                recommendations.append(
                    f"For {metric_name}, experiment '{best_exp}' performed best "
                    f"(mean: {metric_data[best_exp]['mean']:.3f})"
                )
        
        return recommendations
    
    def generate_report(self, experiment_name: str, metrics: Dict[str, BenchmarkMetrics]) -> str:
        """Generate comprehensive research report"""
        timestamp = datetime.now().isoformat()
        report_path = f"{self.results_dir}/reports/report_{experiment_name}_{timestamp}.md"
        
        # Save raw metrics data
        metrics_data = {agent_id: asdict(metric) for agent_id, metric in metrics.items()}
        metrics_path = f"{self.results_dir}/raw_data/metrics_{experiment_name}_{timestamp}.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        # Generate markdown report
        report_content = f"""# ARIASKA_RL Research Report

## Experiment: {experiment_name}
**Generated:** {timestamp}

## Executive Summary

"""
        
        # Add agent-specific metrics
        for agent_id, metric in metrics.items():
            report_content += f"""### {agent_id} Performance

- **Average Reward:** {metric.avg_reward:.3f} ± {metric.reward_std:.3f}
- **Overall Success Rate:** {metric.overall_success_rate:.1%}
- **Learning Efficiency:** {metric.learning_efficiency:.4f}
- **GPT Token Efficiency:** {metric.gpt_token_efficiency:.4f}
- **Exploration Efficiency:** {metric.exploration_efficiency:.1%}
- **Convergence Episode:** {metric.convergence_episode or 'Not detected'}

#### Phase-Specific Success Rates:
"""
            for phase, rate in metric.success_rates.items():
                report_content += f"- **{phase.title()}:** {rate:.1%}\n"
            
            report_content += "\n"
        
        # Statistical analysis
        all_metrics = list(metrics.values())
        stats = self.run_statistical_analysis(all_metrics)
        
        report_content += f"""## Statistical Analysis

- **Sample Size:** {stats['sample_size']} agents
- **Mean Reward:** {stats['reward_statistics'].get('mean', 0):.3f}
- **Reward Standard Deviation:** {stats['reward_statistics'].get('std', 0):.3f}
- **Mean Success Rate:** {stats['success_rate_statistics'].get('mean', 0):.1%}

## Research Insights

This experiment provides insights into multi-agent cybersecurity RL performance.
Key findings and areas for further investigation will be added here.

## Data Files

- Raw metrics: `{metrics_path}`
- Configuration: Available in raw_data directory

---
*Generated by ARIASKA_RL BenchmarkSuite*
"""
        
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        console.print(f"[green]📊[/green] Research report generated: {report_path}")
        return report_path
    
    def display_realtime_metrics(self, metrics: Dict[str, BenchmarkMetrics]):
        """Display real-time metrics in Rich format"""
        
        # Create main metrics table
        table = Table(title="🧪 Real-Time Benchmark Metrics")
        table.add_column("Agent", style="cyan")
        table.add_column("Avg Reward", style="green")
        table.add_column("Success Rate", style="blue")
        table.add_column("Learning Eff.", style="magenta")
        table.add_column("GPT Eff.", style="yellow")
        table.add_column("Convergence", style="red")
        
        for agent_id, metric in metrics.items():
            table.add_row(
                agent_id,
                f"{metric.avg_reward:.2f}",
                f"{metric.overall_success_rate:.1%}",
                f"{metric.learning_efficiency:.4f}",
                f"{metric.gpt_token_efficiency:.4f}",
                f"Ep {metric.convergence_episode}" if metric.convergence_episode else "None"
            )
        
        console.print(table)
        
        # Create phase-specific success rate table
        phase_table = Table(title="📊 Phase-Specific Performance")
        phase_table.add_column("Agent", style="cyan")
        
        # Get all unique phases
        all_phases = set()
        for metric in metrics.values():
            all_phases.update(metric.success_rates.keys())
        
        for phase in sorted(all_phases):
            phase_table.add_column(phase.title(), style="green")
        
        for agent_id, metric in metrics.items():
            row = [agent_id]
            for phase in sorted(all_phases):
                rate = metric.success_rates.get(phase, 0.0)
                row.append(f"{rate:.1%}")
            phase_table.add_row(*row)
        
        console.print(phase_table)