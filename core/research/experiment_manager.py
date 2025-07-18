"""
Experiment Manager for Reproducible Research

Provides comprehensive experiment management capabilities including:
- Experiment configuration versioning
- Reproducible random seed management  
- Automated hyperparameter sweeps
- A/B testing framework
- Git integration for code versioning
"""

import os
import json
import hashlib
import random
import numpy as np
import subprocess
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from itertools import product
from rich.console import Console
from rich.progress import Progress, track

console = Console()

@dataclass
class ExperimentConfig:
    """Comprehensive experiment configuration"""
    name: str
    description: str
    agents: List[str]
    episodes: int
    max_steps: int
    environment_type: str
    random_seed: int
    hyperparameters: Dict[str, Any]
    gpt_config: Dict[str, str]
    memory_config: Dict[str, Any]
    training_config: Dict[str, Any]
    version: str
    git_commit: Optional[str] = None
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.git_commit is None:
            self.git_commit = self._get_git_commit()
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash for reproducibility"""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'], 
                capture_output=True, 
                text=True, 
                check=True
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
    
    @property
    def config_hash(self) -> str:
        """Generate unique hash for this configuration"""
        config_str = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

class ExperimentManager:
    """
    Advanced experiment management for ARIASKA_RL research.
    
    Features:
    - Configuration versioning and reproducibility
    - Hyperparameter optimization
    - A/B testing framework
    - Automated experiment scheduling
    - Statistical validation
    """
    
    def __init__(self, base_dir: str = "experiments"):
        self.base_dir = base_dir
        self.experiments_dir = os.path.join(base_dir, "configs")
        self.results_dir = os.path.join(base_dir, "results")
        
        # Create directory structure
        os.makedirs(self.experiments_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(os.path.join(base_dir, "sweeps"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "ab_tests"), exist_ok=True)
        
        self.experiment_registry: Dict[str, ExperimentConfig] = {}
        self._load_existing_experiments()
        
        console.print(f"[green]✓[/green] ExperimentManager initialized at {base_dir}")
    
    def _load_existing_experiments(self):
        """Load previously saved experiments"""
        if not os.path.exists(self.experiments_dir):
            return
        
        for filename in os.listdir(self.experiments_dir):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(self.experiments_dir, filename), 'r') as f:
                        config_data = json.load(f)
                        config = ExperimentConfig(**config_data)
                        self.experiment_registry[config.name] = config
                except Exception as e:
                    console.print(f"[yellow]⚠[/yellow] Failed to load {filename}: {e}")
    
    def create_experiment(self, config: ExperimentConfig) -> str:
        """Create and register a new experiment"""
        # Ensure reproducibility
        self._set_random_seeds(config.random_seed)
        
        # Save configuration
        config_path = os.path.join(self.experiments_dir, f"{config.name}_{config.config_hash}.json")
        with open(config_path, 'w') as f:
            json.dump(asdict(config), f, indent=2)
        
        # Register in memory
        self.experiment_registry[config.name] = config
        
        console.print(f"[cyan]🧪[/cyan] Created experiment: {config.name}")
        console.print(f"[dim]Config hash: {config.config_hash}[/dim]")
        console.print(f"[dim]Git commit: {config.git_commit}[/dim]")
        
        return config.config_hash
    
    def _set_random_seeds(self, seed: int):
        """Set all random seeds for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        # Set PyTorch seeds if available
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass
    
    def create_baseline_config(self, name: str, description: str = "") -> ExperimentConfig:
        """Create a baseline experiment configuration"""
        return ExperimentConfig(
            name=name,
            description=description or f"Baseline configuration for {name}",
            agents=["RedAgent", "BlueAgent", "ScoutAgent", "ShadowAgent", "OrionAgent"],
            episodes=100,
            max_steps=50,
            environment_type="simulated",
            random_seed=42,
            hyperparameters={
                "learning_rate": 0.001,
                "batch_size": 64,
                "gamma": 0.99,
                "epsilon_start": 1.0,
                "epsilon_end": 0.01,
                "epsilon_decay": 0.995,
                "replay_buffer_size": 10000,
                "target_update_frequency": 10
            },
            gpt_config={
                "primary_model": "gpt-4o-mini",
                "fallback_model": "gpt-4o-mini", 
                "embedding_model": "gpt-4.1-nano",
                "token_limit": 5000
            },
            memory_config={
                "use_prioritized_replay": True,
                "alpha": 0.6,
                "beta": 0.4,
                "memory_sharing": True,
                "deduplication": True
            },
            training_config={
                "curriculum_learning": False,
                "multi_objective": False,
                "transfer_learning": False
            },
            version="1.0"
        )
    
    def create_hyperparameter_sweep(self, 
                                  base_config: ExperimentConfig,
                                  sweep_params: Dict[str, List[Any]],
                                  sweep_name: str) -> List[ExperimentConfig]:
        """Create experiments for hyperparameter sweep"""
        
        # Generate all combinations
        param_names = list(sweep_params.keys())
        param_values = list(sweep_params.values())
        combinations = list(product(*param_values))
        
        sweep_configs = []
        for i, combination in enumerate(combinations):
            # Create new config based on base
            config_dict = asdict(base_config)
            config_dict['name'] = f"{sweep_name}_variant_{i:03d}"
            config_dict['description'] = f"Hyperparameter sweep variant {i}"
            config_dict['random_seed'] = base_config.random_seed + i  # Vary seed slightly
            
            # Update hyperparameters
            for param_name, param_value in zip(param_names, combination):
                if '.' in param_name:  # Nested parameter like 'hyperparameters.learning_rate'
                    keys = param_name.split('.')
                    target = config_dict
                    for key in keys[:-1]:
                        target = target[key]
                    target[keys[-1]] = param_value
                else:
                    config_dict['hyperparameters'][param_name] = param_value
            
            sweep_config = ExperimentConfig(**config_dict)
            sweep_configs.append(sweep_config)
        
        # Save sweep information
        sweep_info = {
            'name': sweep_name,
            'base_config': base_config.name,
            'sweep_params': sweep_params,
            'num_variants': len(combinations),
            'configs': [config.config_hash for config in sweep_configs],
            'created_at': datetime.now().isoformat()
        }
        
        sweep_path = os.path.join(self.base_dir, "sweeps", f"{sweep_name}.json")
        with open(sweep_path, 'w') as f:
            json.dump(sweep_info, f, indent=2)
        
        console.print(f"[magenta]🔄[/magenta] Created hyperparameter sweep: {sweep_name}")
        console.print(f"[dim]Generated {len(sweep_configs)} experiment variants[/dim]")
        
        return sweep_configs
    
    def create_ab_test(self, 
                      config_a: ExperimentConfig,
                      config_b: ExperimentConfig,
                      test_name: str,
                      num_replications: int = 5) -> Dict[str, List[ExperimentConfig]]:
        """Create A/B test with multiple replications"""
        
        ab_configs = {'A': [], 'B': []}
        
        # Create replications for each condition
        for i in range(num_replications):
            # Condition A
            config_a_dict = asdict(config_a)
            config_a_dict['name'] = f"{test_name}_A_rep_{i:02d}"
            config_a_dict['random_seed'] = config_a.random_seed + i * 100
            ab_configs['A'].append(ExperimentConfig(**config_a_dict))
            
            # Condition B  
            config_b_dict = asdict(config_b)
            config_b_dict['name'] = f"{test_name}_B_rep_{i:02d}"
            config_b_dict['random_seed'] = config_b.random_seed + i * 100
            ab_configs['B'].append(ExperimentConfig(**config_b_dict))
        
        # Save A/B test information
        ab_info = {
            'name': test_name,
            'condition_a': config_a.name,
            'condition_b': config_b.name,
            'num_replications': num_replications,
            'configs': {
                'A': [config.config_hash for config in ab_configs['A']],
                'B': [config.config_hash for config in ab_configs['B']]
            },
            'created_at': datetime.now().isoformat()
        }
        
        ab_path = os.path.join(self.base_dir, "ab_tests", f"{test_name}.json")
        with open(ab_path, 'w') as f:
            json.dump(ab_info, f, indent=2)
        
        console.print(f"[blue]⚖️[/blue] Created A/B test: {test_name}")
        console.print(f"[dim]Condition A: {len(ab_configs['A'])} replications[/dim]")
        console.print(f"[dim]Condition B: {len(ab_configs['B'])} replications[/dim]")
        
        return ab_configs
    
    def run_experiment(self, 
                      config: ExperimentConfig,
                      agent_manager_factory: Callable,
                      benchmark_suite=None) -> Dict[str, Any]:
        """Run a single experiment with the given configuration"""
        
        console.print(f"[cyan]🚀[/cyan] Starting experiment: {config.name}")
        
        # Set random seeds for reproducibility
        self._set_random_seeds(config.random_seed)
        
        # Create agent manager with config
        try:
            agent_manager = agent_manager_factory(config)
        except Exception as e:
            console.print(f"[red]❌[/red] Failed to create agent manager: {e}")
            return {'status': 'failed', 'error': str(e)}
        
        # Track experiment start
        start_time = time.time()
        results = {
            'config': asdict(config),
            'status': 'running',
            'start_time': start_time,
            'episodes_completed': 0,
            'metrics': {}
        }
        
        try:
            # Run training episodes
            with Progress() as progress:
                task = progress.add_task(f"Training {config.name}", total=config.episodes)
                
                for episode in range(config.episodes):
                    # Run single episode
                    agent_manager.simulate_all_agents(episodes=1, max_steps=config.max_steps)
                    
                    # Collect metrics periodically
                    if benchmark_suite and (episode + 1) % 10 == 0:
                        episode_metrics = benchmark_suite.collect_agent_metrics(agent_manager, episode + 1)
                        results['metrics'][f'episode_{episode + 1}'] = {
                            agent_id: asdict(metrics) for agent_id, metrics in episode_metrics.items()
                        }
                    
                    results['episodes_completed'] = episode + 1
                    progress.update(task, advance=1)
            
            # Final metrics collection
            if benchmark_suite:
                final_metrics = benchmark_suite.collect_agent_metrics(agent_manager, config.episodes)
                results['final_metrics'] = {
                    agent_id: asdict(metrics) for agent_id, metrics in final_metrics.items()
                }
            
            results['status'] = 'completed'
            results['duration'] = time.time() - start_time
            
            console.print(f"[green]✅[/green] Experiment completed: {config.name}")
            
        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            results['duration'] = time.time() - start_time
            console.print(f"[red]❌[/red] Experiment failed: {config.name} - {e}")
        
        finally:
            # Save results
            results_path = os.path.join(self.results_dir, f"{config.name}_{config.config_hash}_results.json")
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
        
        return results
    
    def run_hyperparameter_sweep(self,
                               sweep_configs: List[ExperimentConfig],
                               agent_manager_factory: Callable,
                               benchmark_suite=None) -> Dict[str, Any]:
        """Run complete hyperparameter sweep"""
        
        console.print(f"[magenta]🔄[/magenta] Starting hyperparameter sweep with {len(sweep_configs)} variants")
        
        sweep_results = {
            'configs': len(sweep_configs),
            'completed': 0,
            'failed': 0,
            'results': {},
            'best_config': None,
            'best_metric': 0.0
        }
        
        for config in track(sweep_configs, description="Running sweep..."):
            result = self.run_experiment(config, agent_manager_factory, benchmark_suite)
            sweep_results['results'][config.name] = result
            
            if result['status'] == 'completed':
                sweep_results['completed'] += 1
                
                # Track best performing configuration
                if 'final_metrics' in result:
                    avg_reward = 0.0
                    for agent_metrics in result['final_metrics'].values():
                        avg_reward += agent_metrics.get('avg_reward', 0.0)
                    avg_reward /= len(result['final_metrics'])
                    
                    if avg_reward > sweep_results['best_metric']:
                        sweep_results['best_metric'] = avg_reward
                        sweep_results['best_config'] = config.name
            else:
                sweep_results['failed'] += 1
        
        console.print(f"[green]✅[/green] Hyperparameter sweep completed")
        console.print(f"[green]Completed: {sweep_results['completed']}, Failed: {sweep_results['failed']}[/green]")
        if sweep_results['best_config']:
            console.print(f"[yellow]🏆[/yellow] Best configuration: {sweep_results['best_config']} (reward: {sweep_results['best_metric']:.3f})")
        
        return sweep_results
    
    def analyze_ab_test(self, test_name: str) -> Dict[str, Any]:
        """Analyze results from A/B test with statistical significance"""
        
        ab_path = os.path.join(self.base_dir, "ab_tests", f"{test_name}.json")
        if not os.path.exists(ab_path):
            console.print(f"[red]❌[/red] A/B test not found: {test_name}")
            return {}
        
        with open(ab_path, 'r') as f:
            ab_info = json.load(f)
        
        # Collect results for both conditions
        condition_results = {'A': [], 'B': []}
        
        for condition in ['A', 'B']:
            for config_hash in ab_info['configs'][condition]:
                # Find results file
                results_files = [f for f in os.listdir(self.results_dir) 
                               if config_hash in f and f.endswith('_results.json')]
                
                for results_file in results_files:
                    try:
                        with open(os.path.join(self.results_dir, results_file), 'r') as f:
                            result = json.load(f)
                            if result['status'] == 'completed' and 'final_metrics' in result:
                                # Extract average reward across all agents
                                avg_reward = 0.0
                                for agent_metrics in result['final_metrics'].values():
                                    avg_reward += agent_metrics.get('avg_reward', 0.0)
                                avg_reward /= len(result['final_metrics'])
                                condition_results[condition].append(avg_reward)
                    except Exception as e:
                        console.print(f"[yellow]⚠[/yellow] Error loading result: {e}")
        
        # Perform statistical analysis
        analysis = {
            'test_name': test_name,
            'condition_a': {
                'name': ab_info['condition_a'],
                'sample_size': len(condition_results['A']),
                'mean': np.mean(condition_results['A']) if condition_results['A'] else 0,
                'std': np.std(condition_results['A']) if condition_results['A'] else 0,
                'values': condition_results['A']
            },
            'condition_b': {
                'name': ab_info['condition_b'],
                'sample_size': len(condition_results['B']),
                'mean': np.mean(condition_results['B']) if condition_results['B'] else 0,
                'std': np.std(condition_results['B']) if condition_results['B'] else 0,
                'values': condition_results['B']
            }
        }
        
        # Simple statistical test (t-test approximation)
        if condition_results['A'] and condition_results['B']:
            from scipy.stats import ttest_ind
            statistic, p_value = ttest_ind(condition_results['A'], condition_results['B'])
            
            analysis['statistical_test'] = {
                'test_type': 'independent_t_test',
                'statistic': statistic,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'effect_size': abs(analysis['condition_a']['mean'] - analysis['condition_b']['mean'])
            }
            
            # Determine winner
            if analysis['statistical_test']['significant']:
                if analysis['condition_a']['mean'] > analysis['condition_b']['mean']:
                    analysis['winner'] = 'A'
                else:
                    analysis['winner'] = 'B'
            else:
                analysis['winner'] = 'No significant difference'
        
        console.print(f"[blue]📊[/blue] A/B Test Analysis: {test_name}")
        console.print(f"Condition A: {analysis['condition_a']['mean']:.3f} ± {analysis['condition_a']['std']:.3f}")
        console.print(f"Condition B: {analysis['condition_b']['mean']:.3f} ± {analysis['condition_b']['std']:.3f}")
        if 'statistical_test' in analysis:
            console.print(f"Winner: {analysis['winner']} (p={analysis['statistical_test']['p_value']:.4f})")
        
        return analysis
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of all experiments"""
        summary = {
            'total_experiments': len(self.experiment_registry),
            'experiments_by_type': {},
            'recent_experiments': [],
            'git_commits': set()
        }
        
        for config in self.experiment_registry.values():
            # Count by description keywords
            if 'baseline' in config.description.lower():
                summary['experiments_by_type']['baseline'] = summary['experiments_by_type'].get('baseline', 0) + 1
            elif 'sweep' in config.description.lower():
                summary['experiments_by_type']['hyperparameter_sweep'] = summary['experiments_by_type'].get('hyperparameter_sweep', 0) + 1
            elif 'ab' in config.description.lower() or 'test' in config.description.lower():
                summary['experiments_by_type']['ab_test'] = summary['experiments_by_type'].get('ab_test', 0) + 1
            else:
                summary['experiments_by_type']['other'] = summary['experiments_by_type'].get('other', 0) + 1
            
            if config.git_commit:
                summary['git_commits'].add(config.git_commit)
        
        # Get recent experiments
        sorted_configs = sorted(
            self.experiment_registry.values(),
            key=lambda x: x.timestamp,
            reverse=True
        )
        summary['recent_experiments'] = [
            {'name': config.name, 'timestamp': config.timestamp} 
            for config in sorted_configs[:5]
        ]
        
        summary['git_commits'] = list(summary['git_commits'])
        
        return summary