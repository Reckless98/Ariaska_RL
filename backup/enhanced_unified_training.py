#!/usr/bin/env python3
"""
ARIASKA_RL Enhanced Unified Training System - FINAL VERSION
🧠 Complete Multi-Agent Training | 📊 Real-Time Analytics | 🎯 Maximum Learning Efficiency

COMPLETE FIXES:
✅ Proper GPU detection and utilization (GTX 1060 3GB)
✅ All 5 agents take actions every step
✅ Detailed agent activity tracking and display
✅ Fixed slice error in action processing
✅ Clean, informative UI without table cutoffs
✅ Memory router with enhanced persistence
✅ Comprehensive error handling and logging
"""

import sys
import os
import time
import json
import random
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from collections import deque, defaultdict, OrderedDict
from dataclasses import dataclass, field
from enum import Enum

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Rich imports for UI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich import box
from rich.text import Text

# Project imports
from core.ui_helpers import display_detailed_agent_status, display_training_metrics, display_gpu_status
from core.ui.enhanced_agent_dashboard import EnhancedAgentDashboard
from core.agents.red_agent import RedAgent
from core.agents.blue_agent import BlueAgent  
from core.agents.scout_agent import ScoutAgent
from core.agents.shadow_agent import ShadowAgent
from core.agents.orion_agent import OrionAgent
from core.environment.cyber_environment import CyberEnvironment
from core.utils.stats_monitor import StatsMonitor
from core.memory.enhanced_memory_router import EnhancedMemoryRouter

@dataclass
class AgentAction:
    """Enhanced agent action record with comprehensive tracking."""
    agent_id: str
    command: str
    target: str
    output: str
    reward: float
    success: bool
    phase: str
    timestamp: float
    gpt_tokens_used: int = 0
    learning_loss: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class TrainingPhase(Enum):
    """Training phases for curriculum learning."""
    RECONNAISSANCE = "reconnaissance"
    ENUMERATION = "enumeration"
    EXPLOITATION = "exploitation"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    EXFILTRATION = "exfiltration"

class EnhancedUnifiedTrainingSystem:
    """
    Final enhanced unified training system with complete functionality.
    """
    
    def __init__(
        self,
        episodes: int = 100,
        max_steps_per_episode: int = 50,
        save_interval: int = 10,
        log_dir: str = "logs/enhanced_training",
        model_dir: str = "models/enhanced",
        session_id: Optional[str] = None,
        target_ip: str = "10.10.10.10",
        enable_gpu: bool = True,
        curriculum_learning: bool = True,
        verbosity: str = "detailed"
    ):
        # Core configuration
        self.episodes = episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.save_interval = save_interval
        self.log_dir = Path(log_dir)
        self.model_dir = Path(model_dir)
        self.session_id = session_id or f"final_{int(time.time())}"
        self.target_ip = target_ip
        self.curriculum_learning = curriculum_learning
        self.verbosity = verbosity
        
        # Enhanced GPU detection and setup
        if enable_gpu and torch.cuda.is_available():
            self.enable_gpu = True
            self.device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"🚀 GPU Acceleration: ENABLED - {gpu_name}")
            print(f"💾 CUDA Memory: {gpu_memory:.1f} GB")
        else:
            self.enable_gpu = False
            self.device = torch.device("cpu")
            if enable_gpu:
                print("⚠️ GPU requested but not available, using CPU")
            else:
                print("💻 Using CPU (GPU disabled)")
        
        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self._setup_logging()
        
        # Initialize systems
        self.agents: Dict[str, Any] = {}
        self.action_history: List[AgentAction] = []
        self.episode_metrics: List[Dict] = []
        self.current_episode = 0
        
        # Enhanced UI Dashboard
        try:
            self.dashboard = EnhancedAgentDashboard(update_interval=0.5)
            self.dashboard_available = True
        except Exception as e:
            print(f"⚠️ Dashboard initialization failed: {e}")
            self.dashboard = None
            self.dashboard_available = False
        
        # Training state
        self.training_active = False
        self.start_time = None
        
        # Console for rich output
        self.console = Console()
        
        print(f"✅ Enhanced Training System initialized - Session: {self.session_id}")
        print(f"📊 Target: {self.target_ip} | Episodes: {episodes} | GPU: {'Enabled' if self.enable_gpu else 'Disabled'}")

    def _setup_logging(self) -> None:
        """Set up comprehensive logging."""
        log_file = self.log_dir / f"enhanced_training_{self.session_id}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Enhanced training system initialized - Session: {self.session_id}")

    def setup_agents(self) -> bool:
        """Initialize all 5 agents with proper GPU support."""
        try:
            self.console.print("🤖 Initializing Enhanced Multi-Agent System...")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=self.console
            ) as progress:
                
                task = progress.add_task("Initializing agents...", total=5)
                
                # Initialize RedAgent with GPU support
                progress.update(task, description="Creating RedAgent...")
                try:
                    from core.agents.red_agent import RedAgent
                    self.agents['RedAgent'] = RedAgent(
                        agent_id=f"RedAgent_{self.session_id}",
                        device=str(self.device)
                    )
                    progress.advance(task)
                    self.logger.info("RedAgent initialized successfully")
                except Exception as e:
                    self.logger.error(f"RedAgent initialization failed: {e}")
                    return False
                
                # Initialize BlueAgent with GPU support
                progress.update(task, description="Creating BlueAgent...")
                try:
                    from core.agents.blue_agent import BlueAgent
                    self.agents['BlueAgent'] = BlueAgent(
                        agent_id=f"BlueAgent_{self.session_id}",
                        device=str(self.device)
                    )
                    progress.advance(task)
                    self.logger.info("BlueAgent initialized successfully")
                except Exception as e:
                    self.logger.error(f"BlueAgent initialization failed: {e}")
                    return False
                
                # Initialize ScoutAgent
                progress.update(task, description="Creating ScoutAgent...")
                try:
                    from core.agents.scout_agent import ScoutAgent
                    # ScoutAgent doesn't support device parameter
                    self.agents['ScoutAgent'] = ScoutAgent(
                        agent_id=f"ScoutAgent_{self.session_id}"
                    )
                    progress.advance(task)
                    self.logger.info("ScoutAgent initialized successfully")
                except Exception as e:
                    self.logger.error(f"ScoutAgent initialization failed: {e}")
                    return False
                
                # Initialize ShadowAgent
                progress.update(task, description="Creating ShadowAgent...")
                try:
                    from core.agents.shadow_agent import ShadowAgent
                    # ShadowAgent doesn't support device parameter
                    self.agents['ShadowAgent'] = ShadowAgent(
                        agent_id=f"ShadowAgent_{self.session_id}"
                    )
                    progress.advance(task)
                    self.logger.info("ShadowAgent initialized successfully")
                except Exception as e:
                    self.logger.error(f"ShadowAgent initialization failed: {e}")
                    return False
                
                # Initialize OrionAgent
                progress.update(task, description="Creating OrionAgent...")
                try:
                    from core.agents.orion_agent import OrionAgent
                    # OrionAgent doesn't support device parameter
                    self.agents['OrionAgent'] = OrionAgent(
                        agent_id=f"OrionAgent_{self.session_id}"
                    )
                    progress.advance(task)
                    self.logger.info("OrionAgent initialized successfully")
                except Exception as e:
                    self.logger.error(f"OrionAgent initialization failed: {e}")
                    return False
            
            # Initialize support systems
            self._initialize_support_systems()
            
            # Display agent summary
            self._display_agent_summary()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Agent setup failed: {e}")
            return False

    def _initialize_support_systems(self) -> None:
        """Initialize memory router, environment, and stats monitor."""
        try:
            # Initialize memory router
            try:
                from core.memory.enhanced_memory_router import EnhancedMemoryRouter
                self.memory_router = EnhancedMemoryRouter(
                    persistence_path=self.log_dir / f"memory_{self.session_id}.db"
                )
            except ImportError:
                self.logger.warning("Enhanced memory router not available, using basic fallback")
                self.memory_router = None
            
            # Initialize environment
            try:
                from core.environment.cyber_environment import CyberEnvironment
                self.environment = CyberEnvironment(
                    scenario=self.target_ip
                )
            except ImportError:
                self.logger.warning("CyberEnvironment not available, using mock environment")
                self.environment = None
            
            # Initialize stats monitor
            try:
                from core.utils.stats_monitor import StatsMonitor
                self.stats_monitor = StatsMonitor()
            except ImportError:
                self.logger.warning("StatsMonitor not available, using basic tracking")
                self.stats_monitor = None
            
            self.logger.info("Support systems initialized")
            
        except Exception as e:
            self.logger.error(f"Support system initialization failed: {e}")

    def _display_agent_summary(self) -> None:
        """Display agent initialization summary."""
        summary_table = Table(title="Agent Initialization Summary", 
                            show_header=True, header_style="bold", box=box.SIMPLE)
        summary_table.add_column("Agent", width=15)
        summary_table.add_column("Status", width=10)
        summary_table.add_column("Type", width=20)
        summary_table.add_column("Device", width=10)
        
        for agent_name, agent in self.agents.items():
            status = "✅ Ready"
            agent_type = "Neural" if hasattr(agent, 'neural_trainer') else "Rule-based"
            device = str(self.device) if hasattr(agent, 'device') else "N/A"
            summary_table.add_row(agent_name, status, agent_type, device)
        
        self.console.print(Panel(summary_table, title=f"🤖 All Agents Ready ({len(self.agents)}/5)"))
        
        # Display GPU status
        if 'display_gpu_status' in globals():
            display_gpu_status(self.device)

    def run_training(self) -> Dict[str, Any]:
        """Execute the complete training process with enhanced dashboard."""
        self.training_active = True
        self.start_time = time.time()
        
        # Start live dashboard if available
        if self.dashboard_available and self.dashboard:
            self.dashboard.start_live_dashboard()
            print("🎮 Live Agent Dashboard Started")
        
        self.console.print(Panel(
            f"🚀 Starting Enhanced Training\n"
            f"Episodes: {self.episodes} | Steps: {self.max_steps_per_episode}\n"
            f"Target: {self.target_ip} | Device: {self.device}\n"
            f"Session: {self.session_id}",
            title="Training Start",
            border_style="green"
        ))
        
        try:
            # Main training loop
            for episode in range(1, self.episodes + 1):
                self.current_episode = episode
                episode_start = time.time()
                
                episode_results = self._run_episode(episode)
                
                episode_duration = time.time() - episode_start
                self.episode_metrics.append(episode_results)
                
                # Update coordination status
                if self.dashboard_available and self.dashboard:
                    self._update_coordination_dashboard(episode)
                
                # Log episode completion
                self.logger.info(f"Episode {episode}/{self.episodes} completed in {episode_duration:.2f}s")
                
                # Display progress
                if episode % 5 == 0 or episode == self.episodes:
                    self._display_training_progress(episode)
                
                # Save checkpoints
                if episode % self.save_interval == 0:
                    self._save_checkpoint(episode)
            
            # Training completed
            training_duration = time.time() - self.start_time
            results = self._generate_final_results(training_duration)
            
            self.console.print(Panel(
                f"✅ Training Completed Successfully!\n"
                f"Duration: {training_duration:.1f}s | Episodes: {self.episodes}\n"
                f"Total Actions: {results.get('total_actions', 0)}\n"
                f"Success Rate: {results.get('success_rate', 0):.1f}%",
                title="Training Complete",
                border_style="green"
            ))
            
            return results
            
        except KeyboardInterrupt:
            self.console.print("\n⚠️ Training interrupted by user")
            return self._generate_final_results(time.time() - self.start_time)
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            self.console.print(f"❌ Training failed: {e}")
            return {}
        finally:
            self.training_active = False
            # Dashboard will be closed automatically when training stops

    def _run_episode(self, episode: int) -> Dict[str, Any]:
        """Execute a single training episode."""
        episode_results = {
            'episode': episode,
            'actions': [],
            'rewards': defaultdict(float),
            'agent_actions': defaultdict(list),
            'success_count': 0,
            'total_actions': 0
        }
        
        # Reset environment if available
        if self.environment:
            try:
                state = self.environment.reset()
            except Exception as e:
                self.logger.warning(f"Environment reset failed: {e}")
                state = {"phase": "reconnaissance", "target": self.target_ip}
        else:
            state = {"phase": "reconnaissance", "target": self.target_ip}
        
        # Execute steps
        for step in range(1, self.max_steps_per_episode + 1):
            step_results = self._execute_step(episode, step, state, episode_results)
            
            # Update state based on step results
            if step_results and 'new_state' in step_results:
                state = step_results['new_state']
        
        return episode_results

    def _execute_step(self, episode: int, step: int, state: Dict, episode_results: Dict) -> Dict[str, Any]:
        """Execute a single training step with all agents."""
        step_results = {
            'step': step,
            'agent_data': {},
            'total_reward': 0.0,
            'actions_taken': 0
        }
        
        agent_data = {}
        
        # Execute actions for each agent
        for agent_name, agent in self.agents.items():
            try:
                # Get agent action with enhanced details
                action_result = self._get_agent_action(agent, state, agent_name, episode, step)
                
                # Process action (fix slice error)
                if isinstance(action_result, (list, tuple, np.ndarray)):
                    # Convert array/list actions to proper format
                    if hasattr(action_result, '__len__') and len(action_result) > 0:
                        action_command = str(action_result[0])
                    else:
                        action_command = "no_action"
                elif isinstance(action_result, dict):
                    action_command = action_result.get('command', 'unknown')
                else:
                    action_command = str(action_result)
                
                # Execute in environment
                env_result = self._execute_in_environment(action_command, agent_name)
                
                # Calculate reward
                reward = self._calculate_reward(action_command, env_result, agent_name)
                
                # Get agent decision info for dashboard
                agent_decision_info = self._get_agent_decision_info(agent, action_command, reward)
                
                # Get stored decision context if available
                decision_context = getattr(agent, '_last_decision_context', {})
                
                # Update dashboard with real-time agent status including thought process
                if self.dashboard_available and self.dashboard:
                    self.dashboard.update_agent_status(agent_name, {
                        "action": action_command,
                        "target": env_result.get('target', 'N/A'),
                        "output": env_result.get('output', 'No output'),
                        "thought_process": decision_context.get('thought_process', f"Executing {action_command}"),
                        "next_step": decision_context.get('next_step', "Analyzing results..."),
                        "reasoning": agent_decision_info.get("reasoning", f"Executing {action_command}"),
                        "reward": reward,
                        "epsilon": agent_decision_info.get("epsilon", 0.0),
                        "confidence": agent_decision_info.get("confidence", 0.5),
                        "status": "ACTIVE",
                        "phase": state.get('phase', 'unknown'),
                        "step_count": step,
                        "success_rate": agent_decision_info.get("success_rate", 0.0),
                        "neural_state": agent_decision_info.get("neural_state", {}),
                        "gpt_calls": agent_decision_info.get("gpt_calls", 0),
                        "success": env_result.get('success', False)
                    })
                
                # Create action record
                action_record = AgentAction(
                    agent_id=agent_name,
                    command=action_command,
                    target=env_result.get('target', self.target_ip),
                    output=env_result.get('output', ''),
                    reward=reward,
                    success=env_result.get('success', False),
                    phase=state.get('phase', 'unknown'),
                    timestamp=time.time(),
                    metadata={'episode': episode, 'step': step}
                )
                
                # Store results
                self.action_history.append(action_record)
                episode_results['actions'].append(action_record)
                episode_results['rewards'][agent_name] += reward
                episode_results['agent_actions'][agent_name].append(action_command)
                episode_results['total_actions'] += 1
                
                if env_result.get('success', False):
                    episode_results['success_count'] += 1
                
                # Store agent data for display
                agent_data[agent_name] = {
                    'action': action_command,
                    'target': env_result.get('target', self.target_ip),
                    'reward': reward,
                    'success': env_result.get('success', False),
                    'output': env_result.get('output', '')[:50] + "..." if len(env_result.get('output', '')) > 50 else env_result.get('output', '')
                }
                
                step_results['total_reward'] += reward
                step_results['actions_taken'] += 1
                
            except Exception as e:
                self.logger.error(f"Action execution failed for {agent_name}: {e}")
                # Store error data
                agent_data[agent_name] = {
                    'action': 'ERROR',
                    'target': self.target_ip,
                    'reward': 0.0,
                    'success': False,
                    'output': f"Error: {str(e)[:30]}"
                }
        
        # Display agent activity
        if self.verbosity == "detailed":
            if self.dashboard_available and self.dashboard:
                # Dashboard handles the display automatically
                pass
            else:
                self._display_basic_activity(agent_data, episode, step)
        
        step_results['agent_data'] = agent_data
        return step_results

    def _get_agent_action(self, agent: Any, state: Dict, agent_name: str, episode: int, step: int) -> str:
        """Get agent action with detailed thought process and next step analysis."""
        try:
            # Prepare agent state with enhanced context
            agent_state = {
                'target': self.target_ip,
                'phase': state.get('phase', 'reconnaissance'),
                'episode': episode,
                'step': step,
                'previous_actions': len(self.action_history),
                'recent_outputs': [action.output for action in self.action_history[-3:]] if self.action_history else [],
                'last_success': self.action_history[-1].success if self.action_history else False
            }
            
            # Get action based on agent type - handle different return formats
            action_result = None
            decision_context = {}
            
            if hasattr(agent, 'act'):
                action_result = agent.act(agent_state)
                # Try to extract thought process from agent
                if hasattr(agent, 'last_reasoning'):
                    decision_context['thought_process'] = getattr(agent, 'last_reasoning', '')
                if hasattr(agent, 'next_planned_action'):
                    decision_context['next_step'] = getattr(agent, 'next_planned_action', '')
            elif hasattr(agent, 'select_action'):
                action_result = agent.select_action(agent_state)
                # For neural agents, try to get planning info
                if hasattr(agent, 'get_action_reasoning'):
                    try:
                        reasoning = agent.get_action_reasoning(agent_state)
                        decision_context['thought_process'] = reasoning
                    except:
                        pass
            elif hasattr(agent, 'get_action'):
                action_result = agent.get_action(agent_state)
                # Extract decision info if available
                if isinstance(action_result, tuple) and len(action_result) > 1:
                    decision_info = action_result[1]
                    if isinstance(decision_info, dict):
                        decision_context.update(decision_info)
            else:
                # Fallback action
                return f"scan {self.target_ip}"
            
            # Handle different return formats
            final_action = ""
            if isinstance(action_result, tuple) and len(action_result) > 0:
                # get_action returns (action, decision_info)
                action = action_result[0]
                if len(action_result) > 1 and isinstance(action_result[1], dict):
                    decision_context.update(action_result[1])
                    
                if isinstance(action, dict) and 'command' in action:
                    final_action = action['command']
                elif isinstance(action, str):
                    final_action = action
                else:
                    final_action = str(action)
            elif isinstance(action_result, dict):
                # act methods return dict with 'command' or 'action'
                if 'command' in action_result:
                    final_action = action_result['command']
                elif 'action' in action_result:
                    final_action = action_result['action']
                else:
                    final_action = str(action_result)
                # Store additional context
                decision_context.update({k: v for k, v in action_result.items() 
                                       if k not in ['command', 'action']})
            elif isinstance(action_result, str):
                # Direct string action
                final_action = action_result
            else:
                # Convert to string
                final_action = str(action_result)
            
            # Generate intelligent thought process if not provided
            if 'thought_process' not in decision_context:
                decision_context['thought_process'] = self._generate_thought_process(agent_name, final_action, agent_state)
            
            # Generate next step if not provided
            if 'next_step' not in decision_context:
                decision_context['next_step'] = self._generate_next_step(agent_name, final_action, agent_state)
            
            # Store decision context for dashboard update
            if hasattr(agent, '_last_decision_context'):
                agent._last_decision_context = decision_context
            else:
                setattr(agent, '_last_decision_context', decision_context)
            
            return final_action
            
        except Exception as e:
            self.logger.error(f"Action generation failed for {agent_name}: {e}")
            return f"error_action_{agent_name}"

    def _generate_thought_process(self, agent_name: str, action: str, state: Dict) -> str:
        """Generate intelligent thought process for agents."""
        phase = state.get('phase', 'unknown')
        target = state.get('target', 'unknown')
        
        if agent_name == "RedAgent":
            if "nmap" in action.lower():
                return f"Initiating network reconnaissance on {target}. Need to identify open ports and services."
            elif "exploit" in action.lower():
                return f"Attempting exploitation based on discovered vulnerabilities. High-value target identified."
            elif "scan" in action.lower():
                return f"Conducting systematic scan of {target} to map attack surface."
            else:
                return f"Analyzing target {target} for optimal attack vector in {phase} phase."
        elif agent_name == "BlueAgent":
            return f"Monitoring defensive posture. Analyzing threats in {phase} phase and strengthening security."
        elif agent_name == "OrionAgent":
            return f"Coordinating team strategy for {phase}. Optimizing agent collaboration and resource allocation."
        elif agent_name == "ScoutAgent":
            return f"Gathering intelligence on {target}. Mapping environment for tactical advantage."
        elif agent_name == "ShadowAgent":
            return f"Operating in stealth mode during {phase}. Maintaining covert presence and avoiding detection."
        else:
            return f"Processing tactical decision for {action} in {phase} phase."

    def _generate_next_step(self, agent_name: str, current_action: str, state: Dict) -> str:
        """Generate next planned step for agents."""
        phase = state.get('phase', 'unknown')
        
        if agent_name == "RedAgent":
            if "nmap" in current_action.lower():
                return "Analyze scan results and identify vulnerable services for targeted exploitation"
            elif "scan" in current_action.lower():
                return "Process scan data and select high-priority targets for deeper enumeration"
            elif "exploit" in current_action.lower():
                return "Establish persistence and escalate privileges if exploitation successful"
            else:
                return "Evaluate results and plan next attack vector based on discovered information"
        elif agent_name == "BlueAgent":
            return "Analyze security logs and implement countermeasures against detected threats"
        elif agent_name == "OrionAgent":
            return "Review team performance metrics and issue strategic directives for optimization"
        elif agent_name == "ScoutAgent":
            return "Compile intelligence report and identify new reconnaissance targets"
        elif agent_name == "ShadowAgent":
            return "Maintain operational security and prepare for covert operations"
        else:
            return f"Continue tactical operations in {phase} phase"

    def _execute_in_environment(self, action: str, agent_name: str) -> Dict[str, Any]:
        """Execute action in environment with fallback."""
        try:
            if self.environment and hasattr(self.environment, 'step'):
                result = self.environment.step(action)
                
                # Handle different return formats from environment
                if isinstance(result, tuple) and len(result) >= 4:
                    # Standard gym format: (state, reward, done, info)
                    state, reward, done, info = result
                    return {
                        'output': info.get('output', f"Executed: {action}") if isinstance(info, dict) else f"Executed: {action}",
                        'success': reward > 0,  # Positive reward indicates success
                        'target': self.target_ip,
                        'execution_time': info.get('execution_time', 1.0) if isinstance(info, dict) else 1.0,
                        'reward': reward,
                        'done': done,
                        'state': state
                    }
                elif isinstance(result, dict):
                    # Already in correct format
                    return result
                else:
                    # Fallback for unexpected formats
                    return {
                        'output': f"Executed: {action}",
                        'success': True,
                        'target': self.target_ip,
                        'execution_time': 1.0
                    }
            else:
                # Mock environment execution
                success = random.random() > 0.3  # 70% success rate
                return {
                    'output': f"Executed: {action}",
                    'success': success,
                    'target': self.target_ip,
                    'execution_time': random.uniform(0.1, 2.0)
                }
        except Exception as e:
            self.logger.error(f"Environment execution failed: {e}")
            return {
                'output': f"Error: {str(e)}",
                'success': False,
                'target': self.target_ip,
                'execution_time': 0.0
            }

    def _get_agent_decision_info(self, agent: Any, action: str, reward: float) -> Dict[str, Any]:
        """Extract decision information from agent for dashboard display."""
        try:
            info = {
                "reasoning": f"Executing action: {action}",
                "epsilon": 0.0,
                "confidence": 0.5,
                "success_rate": 0.0,
                "neural_state": {},
                "gpt_calls": 0
            }
            
            # Extract epsilon for neural agents
            if hasattr(agent, 'epsilon'):
                info["epsilon"] = float(getattr(agent, 'epsilon', 0.0))
            
            # Extract neural network state
            if hasattr(agent, 'neural_trainer') and agent.neural_trainer:
                neural_trainer = agent.neural_trainer
                if hasattr(neural_trainer, 'current_loss'):
                    info["neural_state"]["loss"] = float(getattr(neural_trainer, 'current_loss', 0.0))
                if hasattr(neural_trainer, 'learning_rate'):
                    info["neural_state"]["learning_rate"] = float(getattr(neural_trainer, 'learning_rate', 0.001))
            
            # Extract GPT call count
            if hasattr(agent, 'gpt_manager') and hasattr(agent.gpt_manager, 'total_calls'):
                info["gpt_calls"] = int(getattr(agent.gpt_manager, 'total_calls', 0))
            
            # Calculate confidence based on recent rewards
            info["confidence"] = min(1.0, max(0.0, (reward + 1.0) / 2.0))
            
            # Extract reasoning from GPT decisions
            if hasattr(agent, 'last_gpt_reasoning'):
                info["reasoning"] = str(getattr(agent, 'last_gpt_reasoning', info["reasoning"]))
            
            return info
            
        except Exception as e:
            self.logger.warning(f"Failed to extract agent decision info: {e}")
            return {
                "reasoning": f"Executing action: {action}",
                "epsilon": 0.0,
                "confidence": 0.5,
                "success_rate": 0.0,
                "neural_state": {},
                "gpt_calls": 0
            }

    def _update_coordination_dashboard(self, episode: int):
        """Update coordination status for OrionAgent oversight."""
        try:
            if not (self.dashboard_available and self.dashboard):
                return
                
            # Calculate team metrics
            total_agents = len(self.agents)
            active_agents = total_agents  # All are active during training
            
            # Calculate team coherence based on action diversity
            recent_actions = [action.command for action in self.action_history[-50:]]
            unique_actions = len(set(recent_actions)) if recent_actions else 1
            total_recent = len(recent_actions) if recent_actions else 1
            coherence_score = min(1.0, unique_actions / total_recent)
            
            # Calculate team efficiency from recent rewards
            recent_rewards = [action.reward for action in self.action_history[-20:]]
            avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0
            team_efficiency = min(1.0, max(0.0, (avg_reward + 1.0) / 2.0))
            
            # Determine strategy based on phase
            current_phase = "balanced"
            if self.action_history:
                last_phase = self.action_history[-1].phase
                if last_phase in ["reconnaissance", "enumeration"]:
                    current_phase = "stealth"
                elif last_phase in ["exploitation", "privilege_escalation"]:
                    current_phase = "aggressive"
            
            # Generate OrionAgent insights
            insights = f"Episode {episode}: Team coordination optimal. "
            if team_efficiency > 0.7:
                insights += "High performance maintained."
            elif team_efficiency > 0.4:
                insights += "Performance stable, monitoring for improvements."
            else:
                insights += "Performance below target, adjusting strategy."
            
            self.dashboard.update_coordination_status({
                "directives_active": min(5, len(self.agents)),
                "global_strategy": current_phase,
                "coherence_score": coherence_score,
                "crisis_interventions": max(0, int((1.0 - team_efficiency) * 10)),
                "orion_insights": insights,
                "team_efficiency": team_efficiency
            })
            
            # Update dashboard display
            self.dashboard.update_display()
            
        except Exception as e:
            self.logger.warning(f"Failed to update coordination dashboard: {e}")

    def _calculate_reward(self, action: str, env_result: Dict, agent_name: str) -> float:
        """Calculate reward for agent action."""
        base_reward = 1.0 if env_result.get('success', False) else -0.1
        
        # Bonus for specific actions
        if 'scan' in action.lower():
            base_reward += 0.2
        elif 'exploit' in action.lower():
            base_reward += 0.5
        elif 'privilege' in action.lower():
            base_reward += 0.8
        
        # Agent-specific bonuses
        if agent_name == 'RedAgent' and env_result.get('success', False):
            base_reward += 0.3
        elif agent_name == 'BlueAgent' and 'defend' in action.lower():
            base_reward += 0.3
        
        return round(base_reward, 3)

    def _display_basic_activity(self, agent_data: Dict, episode: int, step: int) -> None:
        """Display basic agent activity if rich UI not available."""
        print(f"\n--- Episode {episode}, Step {step} ---")
        for agent_name, data in agent_data.items():
            status = "✓" if data['success'] else "✗"
            print(f"{agent_name}: {data['action']} -> {status} (Reward: {data['reward']:.2f})")

    def _display_training_progress(self, episode: int) -> None:
        """Display training progress."""
        if len(self.episode_metrics) > 0:
            recent_metrics = self.episode_metrics[-5:]  # Last 5 episodes
            avg_actions = sum(m.get('total_actions', 0) for m in recent_metrics) / len(recent_metrics)
            avg_success = sum(m.get('success_count', 0) for m in recent_metrics) / len(recent_metrics)
            
            progress_data = {
                'episodes_completed': episode,
                'total_episodes': self.episodes,
                'avg_actions': avg_actions,
                'avg_success_rate': (avg_success / max(avg_actions, 1)) * 100,
                'runtime': time.time() - (self.start_time or time.time())
            }
            
            if 'display_training_metrics' in globals() and not self.dashboard_available:
                display_training_metrics(progress_data, self.session_id, progress_data['runtime'])
            else:
                print(f"Progress: {episode}/{self.episodes} episodes, {avg_actions:.1f} avg actions, {progress_data['avg_success_rate']:.1f}% success")

    def _save_checkpoint(self, episode: int) -> None:
        """Save training checkpoint."""
        try:
            checkpoint_data = {
                'session_id': self.session_id,
                'episode': episode,
                'episode_metrics': self.episode_metrics,
                'total_actions': len(self.action_history),
                'timestamp': datetime.now().isoformat()
            }
            
            checkpoint_file = self.log_dir / f"checkpoint_{self.session_id}_{episode}.json"
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            self.logger.info(f"Checkpoint saved: {checkpoint_file}")
            
        except Exception as e:
            self.logger.error(f"Checkpoint save failed: {e}")

    def _generate_final_results(self, training_duration: float) -> Dict[str, Any]:
        """Generate final training results."""
        total_actions = len(self.action_history)
        successful_actions = sum(1 for action in self.action_history if action.success)
        
        results = {
            'session_id': self.session_id,
            'episodes_completed': len(self.episode_metrics),
            'total_episodes': self.episodes,
            'total_actions': total_actions,
            'successful_actions': successful_actions,
            'success_rate': (successful_actions / max(total_actions, 1)) * 100,
            'training_duration': training_duration,
            'device_used': str(self.device),
            'gpu_enabled': self.enable_gpu,
            'agent_count': len(self.agents),
            'final_metrics': {
                'avg_reward': sum(action.reward for action in self.action_history) / max(total_actions, 1),
                'actions_per_episode': total_actions / max(len(self.episode_metrics), 1),
                'coordination_score': successful_actions / max(len(self.agents), 1)
            }
        }
        
        # Save final results
        results_file = self.log_dir / f"final_results_{self.session_id}.json"
        try:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"Final results saved: {results_file}")
        except Exception as e:
            self.logger.error(f"Results save failed: {e}")
        
        return results

def main():
    """Main execution function for direct script usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced ARIASKA_RL Training System")
    parser.add_argument("--episodes", type=int, default=10, help="Number of training episodes")
    parser.add_argument("--steps", type=int, default=20, help="Max steps per episode")
    parser.add_argument("--target", type=str, default="10.10.10.10", help="Target IP address")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU acceleration")
    parser.add_argument("--verbose", action="store_true", help="Enable detailed output")
    
    args = parser.parse_args()
    
    print("🧠 ARIASKA_RL Enhanced Training System - FINAL VERSION")
    print("=" * 60)
    
    # Initialize training system
    trainer = EnhancedUnifiedTrainingSystem(
        episodes=args.episodes,
        max_steps_per_episode=args.steps,
        target_ip=args.target,
        enable_gpu=args.gpu,
        verbosity="detailed" if args.verbose else "standard"
    )
    
    # Setup agents
    if not trainer.setup_agents():
        print("❌ Agent setup failed, exiting...")
        return 1
    
    # Run training
    results = trainer.run_training()
    
    if results:
        print("\n🎉 Training completed successfully!")
        print(f"📊 Total Actions: {results.get('total_actions', 0)}")
        print(f"✅ Success Rate: {results.get('success_rate', 0):.1f}%")
        print(f"⏱️ Duration: {results.get('training_duration', 0):.1f}s")
        print(f"🔧 Device: {results.get('device_used', 'unknown')}")
        return 0
    else:
        print("❌ Training failed!")
        return 1

def test_dashboard():
    """Test the dashboard functionality."""
    print("🎮 Testing Enhanced Agent Dashboard...")
    
    try:
        from core.ui.enhanced_agent_dashboard import EnhancedAgentDashboard
        
        # Initialize dashboard
        dashboard = EnhancedAgentDashboard(update_interval=1.0)
        dashboard.start_live_dashboard()
        
        # Simulate agent activity for testing
        import time
        import random
        
        agents = ['RedAgent', 'BlueAgent', 'ScoutAgent', 'ShadowAgent', 'OrionAgent']
        actions = ['scan', 'exploit', 'enumerate', 'defend', 'infiltrate']
        
        print("📊 Running dashboard simulation for 30 seconds...")
        
        for i in range(30):
            for agent in agents:
                action = random.choice(actions)
                reward = random.uniform(-0.5, 1.5)
                success = reward > 0
                
                dashboard.update_agent_status(agent, {
                    "action": f"{action} target_system",
                    "reasoning": f"Strategic {action} based on current intelligence",
                    "reward": reward,
                    "epsilon": random.uniform(0.1, 0.9),
                    "confidence": random.uniform(0.3, 0.95),
                    "status": "ACTIVE",
                    "phase": random.choice(["reconnaissance", "exploitation", "enumeration"]),
                    "step_count": i + 1,
                    "success_rate": random.uniform(0.4, 0.9),
                    "neural_state": {
                        "loss": random.uniform(0.01, 0.5),
                        "learning_rate": 0.001
                    },
                    "gpt_calls": random.randint(0, 50),
                    "success": success
                })
            
            # Update coordination status
            dashboard.update_coordination_status({
                "directives_active": len(agents),
                "global_strategy": random.choice(["stealth", "aggressive", "balanced"]),
                "coherence_score": random.uniform(0.6, 0.95),
                "crisis_interventions": random.randint(0, 3),
                "orion_insights": f"Simulation step {i+1}: Team performance optimal",
                "team_efficiency": random.uniform(0.5, 0.95)
            })
            
            dashboard.update_display()
            time.sleep(1)
        
        # Dashboard will be closed automatically
        print("✅ Dashboard test completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Dashboard test failed: {e}")
        return 1

if __name__ == "__main__":
    import sys
    
    # Add dashboard test option
    if len(sys.argv) > 1 and sys.argv[1] == "--test-dashboard":
        sys.exit(test_dashboard())
    else:
        sys.exit(main())
