"""
Advanced Metrics Analyzer for ARIASKA_RL Research

Provides sophisticated statistical analysis, performance visualization,
and research insights for multi-agent cybersecurity RL experiments.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import statistics
from collections import defaultdict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

@dataclass 
class PerformanceAnalysis:
    """Comprehensive performance analysis results"""
    agent_id: str
    learning_curve: List[float]
    convergence_analysis: Dict[str, Any]
    stability_metrics: Dict[str, float]
    efficiency_analysis: Dict[str, float]
    comparative_ranking: Optional[int] = None

class MetricsAnalyzer:
    """
    Advanced statistical analysis and visualization toolkit for ARIASKA_RL research.
    
    Features:
    - Learning curve analysis with trend detection
    - Statistical significance testing
    - Performance stability assessment  
    - Comparative agent ranking
    - Research-grade visualization
    - Automated insight generation
    """
    
    def __init__(self, results_dir: str = "research_results"):
        self.results_dir = results_dir
        self.visualization_dir = os.path.join(results_dir, "visualizations")
        os.makedirs(self.visualization_dir, exist_ok=True)
        
        # Configure matplotlib for research-quality plots
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.titlesize': 18
        })
        
        console.print(f"[green]✓[/green] MetricsAnalyzer initialized")
    
    def analyze_learning_curves(self, experiment_data: Dict[str, Any]) -> Dict[str, PerformanceAnalysis]:
        """Comprehensive learning curve analysis for all agents"""
        
        analyses = {}
        
        for agent_id, agent_data in experiment_data.items():
            if not isinstance(agent_data, dict) or 'episode_rewards' not in agent_data:
                continue
                
            rewards = agent_data['episode_rewards']
            if not rewards:
                continue
            
            # Learning curve analysis
            learning_curve = self._smooth_curve(rewards)
            
            # Convergence analysis
            convergence = self._analyze_convergence(rewards)
            
            # Stability metrics
            stability = self._calculate_stability_metrics(rewards)
            
            # Efficiency analysis
            efficiency = self._analyze_efficiency(agent_data)
            
            analyses[agent_id] = PerformanceAnalysis(
                agent_id=agent_id,
                learning_curve=learning_curve,
                convergence_analysis=convergence,
                stability_metrics=stability,
                efficiency_analysis=efficiency
            )
        
        # Add comparative rankings
        self._add_comparative_rankings(analyses)
        
        return analyses
    
    def _smooth_curve(self, values: List[float], window: int = 10) -> List[float]:
        """Apply moving average smoothing to learning curve"""
        if len(values) < window:
            return values
        
        smoothed = []
        for i in range(len(values)):
            start_idx = max(0, i - window // 2)
            end_idx = min(len(values), i + window // 2 + 1)
            smoothed.append(statistics.mean(values[start_idx:end_idx]))
        
        return smoothed
    
    def _analyze_convergence(self, rewards: List[float]) -> Dict[str, Any]:
        """Detailed convergence analysis of learning curve"""
        
        analysis = {
            'converged': False,
            'convergence_episode': None,
            'convergence_value': None,
            'convergence_stability': 0.0,
            'plateau_detection': {},
            'trend_analysis': {}
        }
        
        if len(rewards) < 20:
            return analysis
        
        # Detect convergence using multiple criteria
        window_size = min(20, len(rewards) // 4)
        stability_threshold = np.std(rewards) * 0.1  # 10% of overall std
        
        for i in range(window_size, len(rewards) - window_size):
            window = rewards[i:i + window_size]
            window_std = np.std(window)
            
            if window_std < stability_threshold:
                analysis['converged'] = True
                analysis['convergence_episode'] = i
                analysis['convergence_value'] = np.mean(window)
                analysis['convergence_stability'] = 1.0 - (window_std / np.std(rewards))
                break
        
        # Plateau detection
        analysis['plateau_detection'] = self._detect_plateaus(rewards)
        
        # Trend analysis
        analysis['trend_analysis'] = self._analyze_trends(rewards)
        
        return analysis
    
    def _detect_plateaus(self, values: List[float], min_length: int = 15) -> Dict[str, Any]:
        """Detect performance plateaus in learning curve"""
        
        plateaus = []
        current_plateau = None
        threshold = np.std(values) * 0.05  # 5% of overall std
        
        for i in range(1, len(values)):
            diff = abs(values[i] - values[i-1])
            
            if diff < threshold:
                if current_plateau is None:
                    current_plateau = {'start': i-1, 'values': [values[i-1], values[i]]}
                else:
                    current_plateau['values'].append(values[i])
            else:
                if current_plateau and len(current_plateau['values']) >= min_length:
                    current_plateau['end'] = i-1
                    current_plateau['length'] = len(current_plateau['values'])
                    current_plateau['mean'] = np.mean(current_plateau['values'])
                    plateaus.append(current_plateau)
                current_plateau = None
        
        return {
            'num_plateaus': len(plateaus),
            'plateaus': plateaus,
            'longest_plateau': max(plateaus, key=lambda x: x['length']) if plateaus else None
        }
    
    def _analyze_trends(self, values: List[float]) -> Dict[str, Any]:
        """Analyze learning trends and patterns"""
        
        if len(values) < 10:
            return {}
        
        # Linear trend
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)
        
        # Polynomial trend (degree 2)
        poly_coeffs = np.polyfit(x, values, 2)
        
        # Segment analysis (early, middle, late learning)
        segment_size = len(values) // 3
        early = values[:segment_size]
        middle = values[segment_size:2*segment_size]
        late = values[2*segment_size:]
        
        return {
            'linear_slope': slope,
            'linear_intercept': intercept,
            'overall_trend': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable',
            'polynomial_coeffs': poly_coeffs.tolist(),
            'segment_analysis': {
                'early_mean': np.mean(early),
                'middle_mean': np.mean(middle),
                'late_mean': np.mean(late),
                'early_to_middle_improvement': np.mean(middle) - np.mean(early),
                'middle_to_late_improvement': np.mean(late) - np.mean(middle)
            }
        }
    
    def _calculate_stability_metrics(self, rewards: List[float]) -> Dict[str, float]:
        """Calculate various stability and consistency metrics"""
        
        if len(rewards) < 2:
            return {}
        
        # Basic statistics
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        
        # Coefficient of variation
        cv = std_reward / mean_reward if mean_reward != 0 else float('inf')
        
        # Volatility (average absolute change)
        changes = [abs(rewards[i] - rewards[i-1]) for i in range(1, len(rewards))]
        volatility = np.mean(changes) if changes else 0.0
        
        # Consistency score (inverse of CV, bounded [0,1])
        consistency = 1.0 / (1.0 + cv) if cv != float('inf') else 0.0
        
        # Stability trend (are oscillations decreasing over time?)
        if len(rewards) >= 20:
            first_half_vol = np.std(rewards[:len(rewards)//2])
            second_half_vol = np.std(rewards[len(rewards)//2:])
            stability_trend = (first_half_vol - second_half_vol) / first_half_vol if first_half_vol != 0 else 0.0
        else:
            stability_trend = 0.0
        
        return {
            'coefficient_of_variation': cv,
            'volatility': volatility,
            'consistency_score': consistency,
            'stability_trend': stability_trend,
            'reward_range': max(rewards) - min(rewards),
            'relative_stability': 1.0 - (std_reward / (max(rewards) - min(rewards)) if max(rewards) != min(rewards) else 0.0)
        }
    
    def _analyze_efficiency(self, agent_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze learning and resource efficiency"""
        
        efficiency = {}
        
        # Learning efficiency (reward per episode)
        rewards = agent_data.get('episode_rewards', [])
        if rewards:
            efficiency['reward_per_episode'] = np.mean(rewards)
            
            # Sample efficiency (episodes to reach 90% of max reward)
            max_reward = max(rewards)
            target_reward = 0.9 * max_reward
            
            episodes_to_target = None
            for i, reward in enumerate(rewards):
                if reward >= target_reward:
                    episodes_to_target = i + 1
                    break
            
            efficiency['sample_efficiency'] = episodes_to_target / len(rewards) if episodes_to_target else 1.0
        
        # GPT token efficiency
        efficiency['gpt_token_efficiency'] = agent_data.get('gpt_token_efficiency', 0.0)
        
        # Exploration efficiency
        efficiency['exploration_efficiency'] = agent_data.get('exploration_efficiency', 0.0)
        
        # Memory utilization
        efficiency['memory_utilization'] = agent_data.get('memory_utilization', 0.0)
        
        return efficiency
    
    def _add_comparative_rankings(self, analyses: Dict[str, PerformanceAnalysis]):
        """Add comparative rankings across agents"""
        
        agents = list(analyses.keys())
        metrics = ['avg_reward', 'consistency', 'sample_efficiency']
        
        rankings = defaultdict(list)
        
        for metric in metrics:
            if metric == 'avg_reward':
                values = [(agent, np.mean(analyses[agent].learning_curve)) for agent in agents]
            elif metric == 'consistency':
                values = [(agent, analyses[agent].stability_metrics.get('consistency_score', 0)) for agent in agents]
            elif metric == 'sample_efficiency':
                values = [(agent, analyses[agent].efficiency_analysis.get('sample_efficiency', 0)) for agent in agents]
            
            # Sort by value (descending for better metrics)
            values.sort(key=lambda x: x[1], reverse=True)
            
            for rank, (agent, value) in enumerate(values):
                rankings[agent].append(rank + 1)
        
        # Calculate overall ranking (average of metric rankings)
        for agent in agents:
            if rankings[agent]:
                analyses[agent].comparative_ranking = int(np.mean(rankings[agent]))
    
    def generate_performance_report(self, analyses: Dict[str, PerformanceAnalysis], 
                                  experiment_name: str) -> str:
        """Generate comprehensive performance analysis report"""
        
        timestamp = datetime.now().isoformat()
        report_path = os.path.join(self.results_dir, f"performance_analysis_{experiment_name}_{timestamp}.md")
        
        report_content = f"""# Performance Analysis Report

## Experiment: {experiment_name}
**Generated:** {timestamp}

## Executive Summary

This report provides comprehensive performance analysis for {len(analyses)} agents.

"""
        
        # Overall rankings
        ranked_agents = sorted(analyses.items(), key=lambda x: x[1].comparative_ranking or 999)
        
        report_content += "### Agent Performance Rankings\n\n"
        for rank, (agent_id, analysis) in enumerate(ranked_agents, 1):
            avg_reward = np.mean(analysis.learning_curve) if analysis.learning_curve else 0
            consistency = analysis.stability_metrics.get('consistency_score', 0)
            report_content += f"{rank}. **{agent_id}** - Avg Reward: {avg_reward:.3f}, Consistency: {consistency:.3f}\n"
        
        report_content += "\n## Detailed Agent Analysis\n\n"
        
        # Individual agent analyses
        for agent_id, analysis in analyses.items():
            avg_reward = np.mean(analysis.learning_curve) if analysis.learning_curve else 0
            
            report_content += f"### {agent_id}\n\n"
            
            # Performance metrics
            report_content += f"- **Average Reward:** {avg_reward:.3f}\n"
            report_content += f"- **Ranking:** #{analysis.comparative_ranking}\n"
            
            # Convergence analysis
            conv = analysis.convergence_analysis
            if conv.get('converged'):
                report_content += f"- **Convergence:** Episode {conv['convergence_episode']} to value {conv['convergence_value']:.3f}\n"
            else:
                report_content += f"- **Convergence:** Not detected\n"
            
            # Stability metrics
            stability = analysis.stability_metrics
            report_content += f"- **Consistency Score:** {stability.get('consistency_score', 0):.3f}\n"
            report_content += f"- **Volatility:** {stability.get('volatility', 0):.3f}\n"
            
            # Efficiency metrics
            efficiency = analysis.efficiency_analysis
            report_content += f"- **Sample Efficiency:** {efficiency.get('sample_efficiency', 0):.3f}\n"
            report_content += f"- **GPT Token Efficiency:** {efficiency.get('gpt_token_efficiency', 0):.4f}\n"
            
            # Trend analysis
            trend = analysis.convergence_analysis.get('trend_analysis', {})
            if trend:
                overall_trend = trend.get('overall_trend', 'unknown')
                report_content += f"- **Learning Trend:** {overall_trend}\n"
            
            report_content += "\n"
        
        # Research insights
        report_content += "## Research Insights\n\n"
        insights = self._generate_research_insights(analyses)
        for insight in insights:
            report_content += f"- {insight}\n"
        
        report_content += f"\n---\n*Generated by ARIASKA_RL MetricsAnalyzer*\n"
        
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        console.print(f"[green]📊[/green] Performance analysis report generated: {report_path}")
        return report_path
    
    def _generate_research_insights(self, analyses: Dict[str, PerformanceAnalysis]) -> List[str]:
        """Generate automated research insights from analysis"""
        
        insights = []
        
        if not analyses:
            return insights
        
        # Find best performing agent
        best_agent = max(analyses.items(), key=lambda x: np.mean(x[1].learning_curve) if x[1].learning_curve else 0)
        insights.append(f"**{best_agent[0]}** achieved the highest average performance")
        
        # Convergence insights
        converged_agents = [agent for agent, analysis in analyses.items() if analysis.convergence_analysis.get('converged')]
        if converged_agents:
            insights.append(f"{len(converged_agents)}/{len(analyses)} agents showed convergence")
        else:
            insights.append("No agents showed clear convergence - consider longer training")
        
        # Stability insights
        consistent_agents = [agent for agent, analysis in analyses.items() 
                           if analysis.stability_metrics.get('consistency_score', 0) > 0.7]
        if consistent_agents:
            insights.append(f"High consistency observed in: {', '.join(consistent_agents)}")
        
        # Efficiency insights
        efficient_agents = [agent for agent, analysis in analyses.items()
                          if analysis.efficiency_analysis.get('sample_efficiency', 0) > 0.8]
        if efficient_agents:
            insights.append(f"Fast learning observed in: {', '.join(efficient_agents)}")
        
        return insights
    
    def create_learning_curve_visualization(self, analyses: Dict[str, PerformanceAnalysis], 
                                          experiment_name: str, save: bool = True) -> str:
        """Create research-quality learning curve visualization"""
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Main learning curves
        for agent_id, analysis in analyses.items():
            if analysis.learning_curve:
                episodes = range(1, len(analysis.learning_curve) + 1)
                ax1.plot(episodes, analysis.learning_curve, label=agent_id, linewidth=2, alpha=0.8)
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title(f'Learning Curves - {experiment_name}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Performance comparison (final 20% of episodes)
        final_performance = {}
        for agent_id, analysis in analyses.items():
            if analysis.learning_curve:
                final_20_percent = analysis.learning_curve[int(len(analysis.learning_curve) * 0.8):]
                final_performance[agent_id] = np.mean(final_20_percent) if final_20_percent else 0
        
        if final_performance:
            agents = list(final_performance.keys())
            performance = list(final_performance.values())
            
            bars = ax2.bar(agents, performance, alpha=0.7)
            ax2.set_ylabel('Average Final Performance')
            ax2.set_title('Final Performance Comparison (Last 20% Episodes)')
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save:
            filename = f"learning_curves_{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = os.path.join(self.visualization_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            console.print(f"[green]📈[/green] Learning curve visualization saved: {filepath}")
            plt.close()
            return filepath
        else:
            plt.show()
            return ""
    
    def create_stability_analysis_visualization(self, analyses: Dict[str, PerformanceAnalysis],
                                              experiment_name: str, save: bool = True) -> str:
        """Create stability and consistency analysis visualization"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Consistency scores
        agents = list(analyses.keys())
        consistency_scores = [analyses[agent].stability_metrics.get('consistency_score', 0) for agent in agents]
        
        bars1 = ax1.bar(agents, consistency_scores, alpha=0.7, color='skyblue')
        ax1.set_ylabel('Consistency Score')
        ax1.set_title('Agent Consistency Scores')
        ax1.tick_params(axis='x', rotation=45)
        ax1.set_ylim(0, 1)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # 2. Volatility comparison
        volatility_scores = [analyses[agent].stability_metrics.get('volatility', 0) for agent in agents]
        
        bars2 = ax2.bar(agents, volatility_scores, alpha=0.7, color='lightcoral')
        ax2.set_ylabel('Volatility')
        ax2.set_title('Agent Volatility (Lower is Better)')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Sample efficiency
        sample_efficiency = [analyses[agent].efficiency_analysis.get('sample_efficiency', 0) for agent in agents]
        
        bars3 = ax3.bar(agents, sample_efficiency, alpha=0.7, color='lightgreen')
        ax3.set_ylabel('Sample Efficiency')
        ax3.set_title('Learning Sample Efficiency')
        ax3.tick_params(axis='x', rotation=45)
        ax3.set_ylim(0, 1)
        
        # 4. GPT token efficiency
        gpt_efficiency = [analyses[agent].efficiency_analysis.get('gpt_token_efficiency', 0) for agent in agents]
        
        bars4 = ax4.bar(agents, gpt_efficiency, alpha=0.7, color='gold')
        ax4.set_ylabel('Reward per Token')
        ax4.set_title('GPT Token Efficiency')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save:
            filename = f"stability_analysis_{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = os.path.join(self.visualization_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            console.print(f"[green]📊[/green] Stability analysis visualization saved: {filepath}")
            plt.close()
            return filepath
        else:
            plt.show()
            return ""
    
    def display_realtime_analysis(self, analyses: Dict[str, PerformanceAnalysis]):
        """Display real-time analysis in Rich format"""
        
        # Performance summary table
        table = Table(title="🔬 Performance Analysis Summary")
        table.add_column("Agent", style="cyan")
        table.add_column("Rank", style="bold")
        table.add_column("Avg Reward", style="green")
        table.add_column("Consistency", style="blue")
        table.add_column("Convergence", style="magenta")
        table.add_column("Trend", style="yellow")
        
        for agent_id, analysis in analyses.items():
            avg_reward = np.mean(analysis.learning_curve) if analysis.learning_curve else 0
            consistency = analysis.stability_metrics.get('consistency_score', 0)
            converged = "✓" if analysis.convergence_analysis.get('converged') else "✗"
            trend = analysis.convergence_analysis.get('trend_analysis', {}).get('overall_trend', 'unknown')
            
            table.add_row(
                agent_id,
                f"#{analysis.comparative_ranking}" if analysis.comparative_ranking else "N/A",
                f"{avg_reward:.3f}",
                f"{consistency:.3f}",
                converged,
                trend.title()
            )
        
        console.print(table)
        
        # Insights panel
        insights = self._generate_research_insights(analyses)
        if insights:
            insights_text = "\n".join([f"• {insight}" for insight in insights])
            panel = Panel(insights_text, title="🔍 Research Insights", style="blue")
            console.print(panel)