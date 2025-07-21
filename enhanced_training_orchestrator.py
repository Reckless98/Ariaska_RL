#!/usr/bin/env python3
"""
Enhanced Training Orchestrator for ARIASKA_RL
🚀 Advanced Multi-Agent Training | 🧠 GPT-4o-mini Integration | 📊 Real-time Analytics | 🎯 Curriculum Learning
"""

import time
import json
import asyncio
import logging
import traceback
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import torch
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

# Import our enhanced components
from core.ui_helpers_enhanced import EnhancedAgentDashboard, display_enhanced_agent_status, display_learning_progress
from core.neural_networks_advanced import AdvancedDQNTrainer, AdaptiveCurriculum
from core.coordination_system import (
    AdvancedTaskAllocator, StrategicCoordinator, CoordinationTask, 
    AgentCapability, AgentRole, TaskPriority
)

# Import existing components
from core.environment.cyber_environment import CyberEnvironment
from core.multiagent.memory_router import MemoryRouter
from core.agents.red_agent import RedAgent
from core.agents.blue_agent import BlueAgent
from core.agents.scout_agent import ScoutAgent
from core.agents.shadow_agent import ShadowAgent
from core.agents.orion_agent import OrionAgent

console = Console()

@dataclass
class EnhancedTrainingConfig:
    """Enhanced training configuration with advanced parameters."""
    
    # Basic training parameters
    max_episodes: int = 1000
    max_steps_per_episode: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    gamma: float = 0.99
    
    # Advanced RL parameters
    use_double_dqn: bool = True
    use_dueling_networks: bool = True
    use_prioritized_replay: bool = True
    use_noisy_networks: bool = True
    use_rainbow: bool = True
    
    # Curriculum learning
    enable_curriculum: bool = True
    initial_difficulty: float = 0.2
    max_difficulty: float = 1.0
    curriculum_adaptation_rate: float = 0.1
    
    # GPT integration
    gpt_model: str = "gpt-4o-mini"
    initial_gpt_dependency: float = 0.8
    target_gpt_dependency: float = 0.2
    gpt_reduction_rate: float = 0.01
    
    # Coordination parameters
    enable_advanced_coordination: bool = True
    coordination_update_frequency: int = 10
    strategic_planning_frequency: int = 50
    
    # UI and monitoring
    enable_enhanced_ui: bool = True
    ui_update_frequency: float = 1.0
    save_frequency: int = 100
    
    # Performance optimization
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    enable_mixed_precision: bool = True

class EnhancedTrainingOrchestrator:
    """Advanced training orchestrator with comprehensive agent coordination."""
    
    def __init__(self, config: EnhancedTrainingConfig):
        self.config = config
        self.session_id = f"enhanced_session_{int(time.time())}"
        
        # Setup logging
        self.setup_logging()
        
        # Initialize environment
        self.environment = CyberEnvironment()
        self.memory_router = MemoryRouter()
        
        # Initialize agents
        self.agents = self.initialize_agents()
        
        # Initialize coordination system
        self.setup_coordination_system()
        
        # Initialize curriculum learning
        self.curriculum = AdaptiveCurriculum(
            initial_difficulty=config.initial_difficulty,
            max_difficulty=config.max_difficulty,
            adaptation_rate=config.curriculum_adaptation_rate
        ) if config.enable_curriculum else None
        
        # Initialize UI
        self.dashboard = EnhancedAgentDashboard(
            self.agents, 
            update_interval=config.ui_update_frequency
        ) if config.enable_enhanced_ui else None
        
        # Training state
        self.training_state = {
            'current_episode': 0,
            'total_steps': 0,
            'session_start_time': time.time(),
            'best_performance': 0.0,
            'episode_rewards': [],
            'success_rates': [],
            'coordination_scores': [],
            'gpt_usage_rates': []
        }
        
        self.logger.info(f"Enhanced Training Orchestrator initialized with session ID: {self.session_id}")
    
    def setup_logging(self):
        """Setup enhanced logging system."""
        log_filename = f"logs/enhanced_training_{int(time.time())}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(f"EnhancedTrainer_{self.session_id}")
    
    def initialize_agents(self) -> Dict[str, Any]:
        """Initialize all agents with enhanced capabilities."""
        agents = {}
        
        try:
            # RedAgent with advanced neural networks
            if self.config.use_rainbow:
                from core.neural_networks_advanced import AdvancedDQNTrainer
                
                # Get state and action dimensions from environment
                state_dim = self.environment.get_state_dimension()
                action_dim = self.environment.get_action_dimension()
                
                neural_trainer = AdvancedDQNTrainer(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    learning_rate=self.config.learning_rate,
                    gamma=self.config.gamma,
                    device=self.config.device
                )
                
                agents['RedAgent'] = RedAgent(
                    agent_id='RedAgent',
                    memory_router=self.memory_router,
                    neural_trainer=neural_trainer,
                    gpt_dependency=self.config.initial_gpt_dependency
                )
            else:
                agents['RedAgent'] = RedAgent(
                    agent_id='RedAgent',
                    memory_router=self.memory_router
                )
            
            # Initialize other agents
            agents['BlueAgent'] = BlueAgent(agent_id='BlueAgent', memory_router=self.memory_router)
            agents['ScoutAgent'] = ScoutAgent(agent_id='ScoutAgent', memory_router=self.memory_router)
            agents['ShadowAgent'] = ShadowAgent(agent_id='ShadowAgent', memory_router=self.memory_router)
            agents['OrionAgent'] = OrionAgent(agent_id='OrionAgent', memory_router=self.memory_router)
            
            self.logger.info(f"Successfully initialized {len(agents)} agents")
            
        except Exception as e:
            self.logger.error(f"Error initializing agents: {str(e)}")
            traceback.print_exc()
            raise
        
        return agents
    
    def setup_coordination_system(self):
        """Setup advanced coordination system."""
        
        # Create agent capabilities
        agent_capabilities = {}
        
        capability_configs = {
            'RedAgent': {
                'role': AgentRole.ATTACKER,
                'skills': {
                    'exploitation': 0.8,
                    'vulnerability_assessment': 0.7,
                    'social_engineering': 0.6,
                    'post_exploitation': 0.75
                },
                'specializations': ['web_attacks', 'network_exploitation', 'privilege_escalation']
            },
            'BlueAgent': {
                'role': AgentRole.DEFENDER,
                'skills': {
                    'threat_detection': 0.9,
                    'incident_response': 0.8,
                    'forensics': 0.7,
                    'monitoring': 0.85
                },
                'specializations': ['siem_analysis', 'malware_detection', 'network_monitoring']
            },
            'ScoutAgent': {
                'role': AgentRole.SCOUT,
                'skills': {
                    'reconnaissance': 0.9,
                    'osint': 0.8,
                    'network_mapping': 0.85,
                    'service_enumeration': 0.8
                },
                'specializations': ['passive_recon', 'active_scanning', 'osint_gathering']
            },
            'ShadowAgent': {
                'role': AgentRole.STEALTH,
                'skills': {
                    'stealth': 0.9,
                    'persistence': 0.8,
                    'evasion': 0.85,
                    'covert_operations': 0.8
                },
                'specializations': ['av_evasion', 'log_manipulation', 'covert_channels']
            },
            'OrionAgent': {
                'role': AgentRole.COMMANDER,
                'skills': {
                    'strategic_planning': 0.95,
                    'coordination': 0.9,
                    'decision_making': 0.85,
                    'resource_management': 0.8
                },
                'specializations': ['tactical_planning', 'resource_optimization', 'risk_assessment']
            }
        }
        
        for agent_id, config in capability_configs.items():
            agent_capabilities[agent_id] = AgentCapability(
                agent_id=agent_id,
                role=config['role'],
                skills=config['skills'],
                specializations=config['specializations']
            )
        
        # Initialize coordination components
        self.task_allocator = AdvancedTaskAllocator(agent_capabilities)
        self.strategic_coordinator = StrategicCoordinator(self.agents, self.task_allocator)
        
        self.logger.info("Advanced coordination system initialized")
    
    async def run_enhanced_training(self):
        """Run enhanced training loop with full coordination."""
        
        self.logger.info("Starting enhanced training session")
        
        if self.dashboard:
            console.print(Panel(
                "[bold green]🚀 ARIASKA_RL Enhanced Training Started[/bold green]\n"
                f"Session ID: {self.session_id}\n"
                f"Device: {self.config.device}\n"
                f"Advanced Features: Rainbow DQN, Coordination, Curriculum Learning",
                title="Enhanced Training System",
                border_style="green"
            ))
        
        try:
            # Setup live dashboard if enabled
            if self.dashboard:
                with Live(self.dashboard.create_live_dashboard(self.training_state), 
                         refresh_per_second=2, console=console) as live:
                    await self._training_loop(live)
            else:
                await self._training_loop()
                
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
            console.print("\n[yellow]Training interrupted by user[/yellow]")
        except Exception as e:
            self.logger.error(f"Training error: {str(e)}")
            traceback.print_exc()
            raise
        finally:
            await self._cleanup_training()
    
    async def _training_loop(self, live_display=None):
        """Main enhanced training loop."""
        
        for episode in range(self.config.max_episodes):
            self.training_state['current_episode'] = episode
            episode_start_time = time.time()
            
            # Update curriculum difficulty
            if self.curriculum and episode > 0:
                recent_success_rate = np.mean(self.training_state['success_rates'][-10:]) if self.training_state['success_rates'] else 0.0
                self.curriculum.update_difficulty(recent_success_rate)
                
                # Update environment with new difficulty
                curriculum_config = self.curriculum.get_scenario_config()
                self.environment.update_scenario_config(curriculum_config)
            
            # Reset environment
            initial_state = self.environment.reset()
            episode_reward = 0.0
            episode_success = False
            step_count = 0
            
            # Episode coordination tasks
            episode_tasks = self._generate_episode_tasks()
            for task in episode_tasks:
                self.task_allocator.add_task(task)
            
            # Process initial task allocation
            allocated_tasks = self.task_allocator.process_task_queue()
            
            # Episode step loop
            for step in range(self.config.max_steps_per_episode):
                step_count = step + 1
                self.training_state['total_steps'] += 1
                
                # Strategic coordination update
                if step % self.config.coordination_update_frequency == 0:
                    await self._update_coordination(initial_state)
                
                # Execute agent actions
                agent_actions = await self._execute_coordinated_actions(initial_state)
                
                # Environment step
                next_state, rewards, done, info = self.environment.step(agent_actions)
                
                # Update agent learning
                await self._update_agent_learning(
                    initial_state, agent_actions, rewards, next_state, done
                )
                
                # Update rewards and success tracking
                total_reward = sum(rewards.values()) if isinstance(rewards, dict) else rewards
                episode_reward += total_reward
                
                if info.get('mission_success', False):
                    episode_success = True
                
                # Update display
                if self.dashboard and live_display and step % 5 == 0:
                    self._update_training_state(episode, step, agent_actions, rewards)
                    live_display.update(self.dashboard.create_live_dashboard(self.training_state))
                
                initial_state = next_state
                
                if done:
                    break
            
            # Episode completion
            episode_duration = time.time() - episode_start_time
            success_rate = 1.0 if episode_success else 0.0
            
            # Update training metrics
            self.training_state['episode_rewards'].append(episode_reward)
            self.training_state['success_rates'].append(success_rate)
            
            # Complete episode tasks
            for task in allocated_tasks:
                self.task_allocator.complete_task(task.task_id, episode_success)
            
            # Reduce GPT dependency for RedAgent
            await self._update_gpt_dependency()
            
            # Strategic planning update
            if episode % self.config.strategic_planning_frequency == 0:
                await self._strategic_planning_update()
            
            # Performance logging
            self.logger.info(
                f"Episode {episode}: Reward={episode_reward:.2f}, "
                f"Success={episode_success}, Duration={episode_duration:.2f}s, "
                f"Steps={step_count}"
            )
            
            # Save checkpoints
            if episode % self.config.save_frequency == 0:
                await self._save_checkpoint(episode)
            
            # Update live display
            if self.dashboard and live_display:
                self._update_training_state(episode, step_count, {}, {})
                live_display.update(self.dashboard.create_live_dashboard(self.training_state))
    
    def _generate_episode_tasks(self) -> List[CoordinationTask]:
        """Generate coordination tasks for the episode."""
        tasks = []
        
        # Reconnaissance task
        recon_task = CoordinationTask(
            task_id=f"recon_{int(time.time())}",
            task_type="reconnaissance",
            priority=TaskPriority.HIGH,
            assigned_agents=[],
            parameters={
                'required_skills': {'reconnaissance': 0.7, 'osint': 0.6},
                'min_team_size': 1,
                'max_team_size': 2,
                'complexity': 0.6
            }
        )
        tasks.append(recon_task)
        
        # Attack coordination task
        attack_task = CoordinationTask(
            task_id=f"attack_{int(time.time())}",
            task_type="exploitation",
            priority=TaskPriority.CRITICAL,
            assigned_agents=[],
            dependencies=[recon_task.task_id],
            parameters={
                'required_skills': {'exploitation': 0.8, 'vulnerability_assessment': 0.7},
                'min_team_size': 1,
                'max_team_size': 3,
                'complexity': 0.8
            }
        )
        tasks.append(attack_task)
        
        # Defense monitoring task
        defense_task = CoordinationTask(
            task_id=f"defense_{int(time.time())}",
            task_type="monitoring",
            priority=TaskPriority.MEDIUM,
            assigned_agents=[],
            parameters={
                'required_skills': {'threat_detection': 0.8, 'monitoring': 0.7},
                'min_team_size': 1,
                'max_team_size': 2,
                'complexity': 0.5
            }
        )
        tasks.append(defense_task)
        
        return tasks
    
    async def _update_coordination(self, environment_state):
        """Update strategic coordination based on environment state."""
        
        # Analyze tactical situation
        tactical_analysis = self.strategic_coordinator.analyze_tactical_situation(
            self.environment.get_state_dict()
        )
        
        # Update strategic state
        self.strategic_coordinator.strategic_state.update({
            'threat_level': tactical_analysis['threat_level'],
            'opportunity_score': tactical_analysis['opportunity_score'],
            'recommended_phase': tactical_analysis['recommended_phase']
        })
        
        # Process new task allocations
        allocated_tasks = self.task_allocator.process_task_queue()
        
        # Update coordination scores
        if len(self.training_state['episode_rewards']) > 0:
            recent_performance = self.training_state['episode_rewards'][-1] > 0
            self.training_state['coordination_scores'].append(
                tactical_analysis['coordination_effectiveness']
            )
    
    async def _execute_coordinated_actions(self, state) -> Dict[str, Any]:
        """Execute coordinated agent actions."""
        
        agent_actions = {}
        
        # Get active task assignments
        active_assignments = {}
        for task in self.task_allocator.active_tasks.values():
            for agent_id in task.assigned_agents:
                active_assignments[agent_id] = task
        
        # Execute actions for each agent
        for agent_id, agent in self.agents.items():
            try:
                # Get assigned task context
                task_context = active_assignments.get(agent_id)
                
                if hasattr(agent, 'select_action'):
                    # Neural network agents (RedAgent)
                    if agent_id == 'RedAgent' and hasattr(agent, 'neural_trainer'):
                        # Use advanced neural network
                        epsilon = max(0.05, 1.0 - (self.training_state['current_episode'] / 500))
                        action = agent.neural_trainer.select_action(state, epsilon)
                    else:
                        action = agent.select_action(state)
                else:
                    # Rule-based agents
                    action = agent.get_action(state, task_context)
                
                agent_actions[agent_id] = action
                
            except Exception as e:
                self.logger.error(f"Error executing action for {agent_id}: {str(e)}")
                agent_actions[agent_id] = self.environment.get_random_action()
        
        return agent_actions
    
    async def _update_agent_learning(self, state, actions, rewards, next_state, done):
        """Update agent learning systems."""
        
        # Update RedAgent neural network
        red_agent = self.agents.get('RedAgent')
        if red_agent and hasattr(red_agent, 'neural_trainer'):
            red_action = actions.get('RedAgent')
            red_reward = rewards.get('RedAgent', 0.0) if isinstance(rewards, dict) else rewards
            
            # Store experience
            red_agent.neural_trainer.store_experience(
                state, red_action, red_reward, next_state, done
            )
            
            # Update network
            if len(red_agent.neural_trainer.replay_buffer) > self.config.batch_size:
                update_metrics = red_agent.neural_trainer.update(self.config.batch_size)
                
                # Log training metrics
                if update_metrics and 'loss' in update_metrics:
                    self.logger.debug(f"RedAgent training - Loss: {update_metrics['loss']:.4f}")
        
        # Update other agents' learning systems (if any)
        for agent_id, agent in self.agents.items():
            if agent_id != 'RedAgent' and hasattr(agent, 'update_learning'):
                agent_reward = rewards.get(agent_id, 0.0) if isinstance(rewards, dict) else rewards
                agent.update_learning(state, actions[agent_id], agent_reward, next_state, done)
    
    async def _update_gpt_dependency(self):
        """Gradually reduce GPT dependency for neural agents."""
        
        red_agent = self.agents.get('RedAgent')
        if red_agent and hasattr(red_agent, 'gpt_dependency'):
            current_dependency = red_agent.gpt_dependency
            target_dependency = self.config.target_gpt_dependency
            
            if current_dependency > target_dependency:
                new_dependency = max(
                    target_dependency,
                    current_dependency - self.config.gpt_reduction_rate
                )
                red_agent.gpt_dependency = new_dependency
                
                self.training_state['gpt_usage_rates'].append(new_dependency)
    
    async def _strategic_planning_update(self):
        """Perform strategic planning update."""
        
        # Generate coordination report
        coord_report = self.strategic_coordinator.generate_coordination_report()
        
        # Log strategic insights
        self.logger.info(f"Strategic Planning Update:")
        self.logger.info(f"  Coordination Effectiveness: {coord_report['coordination_effectiveness']:.3f}")
        self.logger.info(f"  Task Success Rate: {coord_report['task_statistics']['success_rate']:.3f}")
        
        # Apply strategic recommendations
        for recommendation in coord_report['recommendations']:
            self.logger.info(f"  Recommendation: {recommendation}")
    
    def _update_training_state(self, episode: int, step: int, actions: Dict, rewards: Dict):
        """Update training state for UI display."""
        
        self.training_state.update({
            'current_episode': episode,
            'current_step': step,
            'session_id': self.session_id,
            'device': self.config.device,
            'uptime': f"{time.time() - self.training_state['session_start_time']:.1f}s",
            'recent_rewards': self.training_state['episode_rewards'][-10:] if self.training_state['episode_rewards'] else [0],
            'avg_reward': np.mean(self.training_state['episode_rewards'][-10:]) if self.training_state['episode_rewards'] else 0.0,
            'success_rate': np.mean(self.training_state['success_rates'][-10:]) if self.training_state['success_rates'] else 0.0,
            'best_reward': max(self.training_state['episode_rewards']) if self.training_state['episode_rewards'] else 0.0,
            'best_success_rate': max(self.training_state['success_rates']) if self.training_state['success_rates'] else 0.0,
            'gpt_usage': self.training_state['gpt_usage_rates'][-1] if self.training_state['gpt_usage_rates'] else self.config.initial_gpt_dependency,
            'coordination_score': np.mean(self.training_state['coordination_scores'][-10:]) if self.training_state['coordination_scores'] else 0.5,
            'total_episodes': self.config.max_episodes,
            'coordination_matrix': self.task_allocator.coordination_matrix.matrix.tolist() if hasattr(self.task_allocator, 'coordination_matrix') else [[]]
        })
    
    async def _save_checkpoint(self, episode: int):
        """Save training checkpoint."""
        
        checkpoint_path = f"models/enhanced/checkpoint_episode_{episode}_{self.session_id}.pth"
        
        # Save RedAgent neural network
        red_agent = self.agents.get('RedAgent')
        if red_agent and hasattr(red_agent, 'neural_trainer'):
            red_agent.neural_trainer.save(checkpoint_path)
        
        # Save training state
        state_path = f"logs/training_state_{episode}_{self.session_id}.json"
        with open(state_path, 'w') as f:
            json.dump(self.training_state, f, indent=2, default=str)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    async def _cleanup_training(self):
        """Cleanup training resources."""
        
        self.logger.info("Cleaning up training session")
        
        # Final save
        await self._save_checkpoint(self.training_state['current_episode'])
        
        # Generate final report
        final_report = self.strategic_coordinator.generate_coordination_report()
        
        # Save final report
        report_path = f"logs/final_report_{self.session_id}.json"
        with open(report_path, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        self.logger.info(f"Training session {self.session_id} completed successfully")
        
        # Display final summary
        if self.dashboard:
            console.print(Panel(
                f"[bold green]✅ Training Session Completed[/bold green]\n"
                f"Episodes: {self.training_state['current_episode']}\n"
                f"Total Steps: {self.training_state['total_steps']}\n"
                f"Final Success Rate: {self.training_state['success_rates'][-1] if self.training_state['success_rates'] else 0:.1%}\n"
                f"Best Performance: {max(self.training_state['episode_rewards']) if self.training_state['episode_rewards'] else 0:.2f}\n"
                f"Session Duration: {time.time() - self.training_state['session_start_time']:.1f}s",
                title="Training Complete",
                border_style="green"
            ))

# Enhanced training runner
async def run_enhanced_training():
    """Run enhanced training with all advanced features."""
    
    config = EnhancedTrainingConfig(
        max_episodes=1000,
        max_steps_per_episode=100,
        use_rainbow=True,
        enable_curriculum=True,
        enable_advanced_coordination=True,
        enable_enhanced_ui=True
    )
    
    trainer = EnhancedTrainingOrchestrator(config)
    await trainer.run_enhanced_training()

if __name__ == "__main__":
    asyncio.run(run_enhanced_training())
