#!/usr/bin/env python3
"""
ARIASKA_RL - Enhanced Unified Training System
Complete rewrite with superior visibility and cleaner architecture.
"""

import sys
import os
import json
import time
import random
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.text import Text
from rich.rule import Rule
from rich.columns import Columns
from rich.align import Align
from rich import box

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.gpt_manager import GPTManager


@dataclass
class AgentAction:
    """Detailed action information for an agent"""
    agent_name: str
    episode: int
    step: int
    target: str
    command: str
    full_output: str
    output_summary: str
    reward: float
    success: bool
    timestamp: float
    learning_improvement: bool


@dataclass
class EpisodeReport:
    """Complete episode report"""
    episode: int
    duration: float
    total_commands: int
    successful_commands: int
    total_reward: float
    accuracy: float
    agent_performances: Dict[str, Dict[str, Any]]
    discoveries: List[str]
    major_actions: List[AgentAction]


class SimpleDeepQNetwork(nn.Module):
    """Lightweight Deep Q-Network for RL"""
    
    def __init__(self, state_size: int = 64, action_size: int = 10, hidden_size: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
    
    def forward(self, x):
        return self.network(x)


class IntelligentAgent:
    """Advanced intelligent agent with deep learning and memory"""
    
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        
        # Neural network for action selection
        self.q_network = SimpleDeepQNetwork()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        
        # Learning parameters
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        self.skill_level = 0.0
        self.gpt_dependency = 1.0
        
        # Memory and experience
        self.replay_buffer = deque(maxlen=2000)
        self.learned_patterns = {}
        self.command_history = []
        self.success_history = []
        self.memory = {}
        
        # Current state
        self.current_target = "10.10.10.10"
        self.current_command = ""
        self.last_output = ""
        self.last_reward = 0.0
        
        # Role-specific commands
        self.command_sets = {
            "RedAgent": ["nmap", "enum4linux", "hydra", "sqlmap", "metasploit"],
            "BlueAgent": ["netstat", "ps", "iptables", "tcpdump", "auditd"],
            "ScoutAgent": ["ping", "traceroute", "whois", "dig", "nslookup"],
            "ShadowAgent": ["steganography", "cryptography", "social_engineering", "osint", "footprinting"],
            "OrionAgent": ["strategy", "coordination", "analysis", "reporting", "planning"]
        }
    
    def select_action(self, state: np.ndarray, available_commands: List[str]) -> str:
        """Select action using epsilon-greedy with Q-learning"""
        # Check if we have learned patterns for this situation
        state_key = self._encode_state(state)
        if state_key in self.learned_patterns and random.random() > self.gpt_dependency:
            # Use learned pattern
            command = self.learned_patterns[state_key]
            self.current_command = command
            return command
        
        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            # Explore: random action
            commands = self.command_sets.get(self.name, ["ping", "nmap"])
            command = random.choice(commands)
        else:
            # Exploit: use Q-network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                action_idx = q_values.argmax().item()
                commands = self.command_sets.get(self.name, ["ping", "nmap"])
                command = commands[action_idx % len(commands)]
        
        self.current_command = command
        return command
    
    def update(self, state: np.ndarray, action: str, reward: float, next_state: np.ndarray, done: bool):
        """Update Q-network and learning parameters"""
        # Store experience
        self.replay_buffer.append((state, action, reward, next_state, done))
        
        # Update success history
        self.success_history.append(reward > 0)
        if len(self.success_history) > 100:
            self.success_history.pop(0)
        
        # Update skill level based on recent performance
        if self.success_history:
            recent_success_rate = sum(self.success_history[-20:]) / min(20, len(self.success_history))
            self.skill_level = min(1.0, recent_success_rate)
        
        # Reduce GPT dependency as agent learns
        if reward > 0:
            self.gpt_dependency = max(0.1, self.gpt_dependency * 0.99)
        
        # Learn patterns
        state_key = self._encode_state(state)
        if reward > 5:  # Successful action
            self.learned_patterns[state_key] = action
        
        # Replay training
        if len(self.replay_buffer) > 32:
            self._replay_training()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        self.last_reward = reward
    
    def _encode_state(self, state: np.ndarray) -> str:
        """Encode state for pattern matching"""
        return str(hash(tuple(state.round(2))))
    
    def _replay_training(self):
        """Train Q-network on replay buffer"""
        batch_size = 32
        if len(self.replay_buffer) < batch_size:
            return
        
        # Sample random batch
        batch = random.sample(list(self.replay_buffer), batch_size)
        
        states = torch.FloatTensor([e[0] for e in batch])
        actions = [e[1] for e in batch]
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        # Current Q values
        current_q_values = self.q_network(states)
        
        # Target Q values
        with torch.no_grad():
            next_q_values = self.q_network(next_states)
            target_q_values = rewards + (0.99 * next_q_values.max(1)[0] * ~dones)
        
        # Convert actions to indices (simplified)
        commands = self.command_sets.get(self.name, ["ping"])
        action_indices = torch.LongTensor([commands.index(action) % len(commands) for action in actions])
        current_q_values = current_q_values.gather(1, action_indices.unsqueeze(1)).squeeze()
        
        # Compute loss and update
        loss = F.mse_loss(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class CybersecurityEnvironment:
    """Advanced cybersecurity simulation environment"""
    
    def __init__(self):
        self.targets = ["10.10.10.10", "192.168.1.100", "172.16.0.50"]
        self.current_target = random.choice(self.targets)
        self.discovered_services = set()
        self.discovered_vulnerabilities = set()
        self.discovered_credentials = set()
        self.detection_level = 0.0
        
        # Realistic command outputs database
        self.command_outputs = {
            "nmap": self._generate_nmap_output,
            "enum4linux": self._generate_enum4linux_output,
            "hydra": self._generate_hydra_output,
            "sqlmap": self._generate_sqlmap_output,
            "metasploit": self._generate_metasploit_output,
            "netstat": self._generate_netstat_output,
            "ps": self._generate_ps_output,
            "tcpdump": self._generate_tcpdump_output,
            "ping": self._generate_ping_output,
            "traceroute": self._generate_traceroute_output,
            "whois": self._generate_whois_output
        }
    
    def execute_command(self, agent_name: str, command: str) -> Tuple[str, float, bool]:
        """Execute command and return output, reward, and success status"""
        # Get command base (e.g., "nmap" from "nmap -sT target")
        cmd_base = command.split()[0] if command else "unknown"
        
        # Generate realistic output
        if cmd_base in self.command_outputs:
            full_output = self.command_outputs[cmd_base](self.current_target)
        else:
            full_output = f"Command '{command}' executed successfully on {self.current_target}"
        
        # Calculate reward based on command type and discoveries
        reward = self._calculate_reward(agent_name, command, full_output)
        
        # Update detection level
        self._update_detection_level(command)
        
        # Check for discoveries
        self._process_discoveries(full_output)
        
        # Create output summary
        output_summary = full_output[:50] + "..." if len(full_output) > 50 else full_output
        
        success = reward > 0
        return full_output, reward, success
    
    def _generate_nmap_output(self, target: str) -> str:
        """Generate realistic nmap output"""
        ports = [22, 80, 443, 3389, 1433, 3306]
        open_ports = random.sample(ports, random.randint(2, 4))
        
        output = f"Starting Nmap scan against {target}\\n"
        output += f"Host is up (0.00{random.randint(10, 99)}s latency).\\n"
        
        for port in sorted(open_ports):
            service = {22: "ssh", 80: "http", 443: "https", 3389: "rdp", 1433: "mssql", 3306: "mysql"}[port]
            output += f"{port}/tcp open  {service}\\n"
        
        self.discovered_services.update([f"{target}:{port}" for port in open_ports])
        return output
    
    def _generate_enum4linux_output(self, target: str) -> str:
        """Generate realistic enum4linux output"""
        usernames = ["administrator", "guest", "john.doe", "admin", "service_account"]
        shares = ["C$", "ADMIN$", "IPC$", "shared_docs", "backup"]
        
        output = f"Enum4linux scan results for {target}:\\n"
        output += f"Domain: WORKGROUP\\n"
        
        discovered_users = random.sample(usernames, random.randint(2, 4))
        for user in discovered_users:
            output += f"User: {user}\\n"
        
        discovered_shares = random.sample(shares, random.randint(2, 3))
        for share in discovered_shares:
            output += f"Share: {share}\\n"
        
        self.discovered_credentials.update(discovered_users)
        return output
    
    def _generate_hydra_output(self, target: str) -> str:
        """Generate realistic hydra bruteforce output"""
        if random.random() < 0.3:  # 30% chance of finding credentials
            username = random.choice(["admin", "root", "administrator"])
            password = random.choice(["password", "123456", "admin", "root"])
            output = f"Hydra bruteforce against {target}:\\n"
            output += f"[22][ssh] host: {target} login: {username} password: {password}\\n"
            output += f"1 of 1 target successfully completed, 1 valid password found\\n"
            self.discovered_credentials.add(f"{username}:{password}")
            return output
        else:
            return f"Hydra bruteforce against {target}: No valid credentials found"
    
    def _generate_sqlmap_output(self, target: str) -> str:
        """Generate realistic sqlmap output"""
        if random.random() < 0.4:  # 40% chance of finding SQL injection
            output = f"SQLMap scan against {target}:\\n"
            output += f"Parameter 'id' is vulnerable to SQL injection\\n"
            output += f"Payload: 1' UNION SELECT NULL,version(),NULL--\\n"
            output += f"Database: MySQL 5.7.21\\n"
            self.discovered_vulnerabilities.add(f"SQL_Injection_{target}")
            return output
        else:
            return f"SQLMap scan against {target}: No SQL injection vulnerabilities found"
    
    def _generate_metasploit_output(self, target: str) -> str:
        """Generate realistic metasploit output"""
        exploits = ["ms17_010_eternalblue", "cve_2019_0708_bluekeep", "apache_struts_rce"]
        exploit = random.choice(exploits)
        
        if random.random() < 0.25:  # 25% chance of successful exploit
            output = f"Metasploit exploit {exploit} against {target}:\\n"
            output += f"[*] Sending stage (175174 bytes) to {target}\\n"
            output += f"[*] Meterpreter session 1 opened\\n"
            output += f"meterpreter > shell\\n"
            self.discovered_vulnerabilities.add(f"Exploited_{exploit}_{target}")
            return output
        else:
            return f"Metasploit exploit {exploit} failed against {target}"
    
    def _generate_netstat_output(self, target: str) -> str:
        """Generate realistic netstat output"""
        connections = [
            f"tcp        0      0 0.0.0.0:22              0.0.0.0:*               LISTEN",
            f"tcp        0      0 0.0.0.0:80              0.0.0.0:*               LISTEN",
            f"tcp        0      0 {target}:80          192.168.1.{random.randint(100,200)}:4{random.randint(1000,9999)}     ESTABLISHED",
            f"udp        0      0 0.0.0.0:53              0.0.0.0:*"
        ]
        return "Active Internet connections:\\n" + "\\n".join(connections)
    
    def _generate_ps_output(self, target: str) -> str:
        """Generate realistic ps output"""
        processes = [
            "root         1  0.0  0.1  225868  9088 ?        Ss   Jan01   0:02 /sbin/init",
            "root         2  0.0  0.0       0     0 ?        S    Jan01   0:00 [kthreadd]",
            "www-data  1234  0.0  0.2  123456  8192 ?        S    10:30   0:00 /usr/sbin/apache2",
            "mysql     5678  0.1  1.5  987654 45678 ?        Sl   10:25   0:03 /usr/sbin/mysqld"
        ]
        return "PID  USER     TIME  COMMAND\\n" + "\\n".join(processes)
    
    def _generate_tcpdump_output(self, target: str) -> str:
        """Generate realistic tcpdump output"""
        packets = [
            f"10:30:45.123456 IP {target}.22 > 192.168.1.100.4567: Flags [P.], seq 1:49, ack 1, win 64, length 48",
            f"10:30:45.124567 IP 192.168.1.100.4567 > {target}.22: Flags [.], ack 49, win 502, length 0",
            f"10:30:45.125678 IP {target}.80 > 192.168.1.101.8080: Flags [S], seq 0, win 65535, length 0"
        ]
        return "tcpdump: listening on eth0\\n" + "\\n".join(packets)
    
    def _generate_ping_output(self, target: str) -> str:
        """Generate realistic ping output"""
        return f"PING {target} 56 data bytes\\n64 bytes from {target}: icmp_seq=1 ttl=64 time=0.{random.randint(100,999)}ms\\n64 bytes from {target}: icmp_seq=2 ttl=64 time=0.{random.randint(100,999)}ms"
    
    def _generate_traceroute_output(self, target: str) -> str:
        """Generate realistic traceroute output"""
        hops = [
            f"1  192.168.1.1 ({random.randint(1,5)}.{random.randint(100,999)}ms)",
            f"2  10.0.0.1 ({random.randint(10,50)}.{random.randint(100,999)}ms)",
            f"3  {target} ({random.randint(50,100)}.{random.randint(100,999)}ms)"
        ]
        return f"traceroute to {target}:\\n" + "\\n".join(hops)
    
    def _generate_whois_output(self, target: str) -> str:
        """Generate realistic whois output"""
        return f"Domain Name: example.com\\nRegistrar: Example Registrar\\nCreation Date: 2020-01-01\\nExpiry Date: 2025-01-01\\nName Server: ns1.example.com"
    
    def _calculate_reward(self, agent_name: str, command: str, output: str) -> float:
        """Calculate reward based on command effectiveness"""
        base_reward = 1.0
        
        # Role-specific bonuses
        role_bonuses = {
            "RedAgent": {"nmap": 2.0, "enum4linux": 1.5, "hydra": 3.0, "sqlmap": 2.5, "metasploit": 4.0},
            "BlueAgent": {"netstat": 1.5, "ps": 1.5, "iptables": 2.0, "tcpdump": 2.5},
            "ScoutAgent": {"ping": 1.0, "traceroute": 1.5, "whois": 1.5, "dig": 1.5},
            "ShadowAgent": {"steganography": 2.0, "cryptography": 2.5, "osint": 2.0},
            "OrionAgent": {"strategy": 3.0, "analysis": 2.5, "coordination": 2.0}
        }
        
        cmd_base = command.split()[0] if command else "unknown"
        role_reward = role_bonuses.get(agent_name, {}).get(cmd_base, base_reward)
        
        # Discovery bonuses
        discovery_bonus = 0.0
        if "open" in output.lower() or "vulnerable" in output.lower():
            discovery_bonus += 2.0
        if "password" in output.lower() or "credentials" in output.lower():
            discovery_bonus += 3.0
        if "meterpreter" in output.lower() or "shell" in output.lower():
            discovery_bonus += 5.0
        
        return role_reward + discovery_bonus
    
    def _update_detection_level(self, command: str):
        """Update detection level based on command aggressiveness"""
        aggressive_commands = ["hydra", "sqlmap", "metasploit", "nmap"]
        if any(cmd in command for cmd in aggressive_commands):
            self.detection_level = min(1.0, self.detection_level + 0.1)
    
    def _process_discoveries(self, output: str):
        """Process discoveries from command output"""
        # This would typically parse output for specific patterns
        pass
    
    def get_state(self) -> np.ndarray:
        """Get current environment state as vector"""
        # Create state vector representing current environment
        state = np.zeros(64)
        
        # Target information (first 8 elements)
        target_hash = hash(self.current_target) % 256
        state[0] = target_hash / 256.0
        
        # Discovery information
        state[1] = len(self.discovered_services) / 10.0
        state[2] = len(self.discovered_vulnerabilities) / 5.0
        state[3] = len(self.discovered_credentials) / 5.0
        state[4] = self.detection_level
        
        # Random environmental factors
        state[5:] = np.random.random(59) * 0.1  # Small random noise
        
        return state


class EnhancedUnifiedTrainer:
    """Enhanced unified trainer with superior visibility"""
    
    def __init__(self, episodes: int = 10):
        self.episodes = episodes
        self.current_episode = 0
        self.session_id = int(time.time())
        
        # Initialize console and logging
        self.console = Console()
        self.logger = self._setup_logging()
        
        # Initialize environment and agents
        self.environment = CybersecurityEnvironment()
        self.gpt_manager = GPTManager.get_instance()
        
        # Create intelligent agents
        self.agents = {
            "RedAgent": IntelligentAgent("RedAgent", "Penetration Testing"),
            "BlueAgent": IntelligentAgent("BlueAgent", "Defense & Monitoring"),
            "ScoutAgent": IntelligentAgent("ScoutAgent", "Reconnaissance"),
            "ShadowAgent": IntelligentAgent("ShadowAgent", "Stealth Operations"),
            "OrionAgent": IntelligentAgent("OrionAgent", "Strategic Coordination")
        }
        
        # Training state
        self.episode_reports = []
        self.current_actions = {}
        self.total_rewards = {name: 0.0 for name in self.agents.keys()}
        self.current_target = self.environment.current_target
        
        # Load agent memories
        self._load_agent_memories()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup enhanced logging"""
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        logger = logging.getLogger("EnhancedTrainer")
        logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(logs_dir / f"enhanced_training_{self.session_id}.log")
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def train(self) -> Dict[str, Any]:
        """Main training loop with enhanced visibility"""
        self.console.print(Panel(
            f"🚀 Enhanced ARIASKA_RL Training Session {self.session_id}\\n"
            f"📊 Episodes: {self.episodes} | Agents: {len(self.agents)}\\n"
            f"🎯 Target: {self.current_target}\\n"
            f"🧠 Mode: Advanced Deep RL + local-llm + Memory Persistence",
            title="🧠 Training Initialization",
            border_style="bright_blue"
        ))
        
        start_time = time.time()
        
        # Create enhanced UI layout
        layout = self._create_enhanced_layout()
        
        with Live(layout, refresh_per_second=3, console=self.console) as live:
            for episode in range(1, self.episodes + 1):
                self.current_episode = episode
                self.logger.info(f"Starting Episode {episode}/{self.episodes}")
                
                # Run episode
                episode_report = self._run_episode(episode)
                self.episode_reports.append(episode_report)
                
                # Update UI
                self._update_enhanced_layout(layout)
                live.update(layout)
                
                # Small delay for visibility
                time.sleep(0.5)
        
        # Training completed
        duration = time.time() - start_time
        results = self._generate_final_results(duration)
        
        self.console.print(Panel(
            f"✅ Training Complete!\\n"
            f"⏱️  Duration: {duration:.2f}s | Episodes: {self.episodes}\\n"
            f"📈 Final Results: {results['summary']}",
            title="🎯 Training Results",
            border_style="bright_green"
        ))
        
        # Save agent memories
        self._save_agent_memories()
        
        return results
    
    def _create_enhanced_layout(self) -> Layout:
        """Create responsive UI layout with comprehensive visibility"""
        layout = Layout()
        
        # Split into header, main content, and footer
        layout.split_column(
            Layout(name="header", size=4),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=4)
        )
        
        # Split main into three panels for better organization
        layout["main"].split_row(
            Layout(name="agents_panel", ratio=2),
            Layout(name="actions_panel", ratio=3),
            Layout(name="stats_panel", ratio=2)
        )
        
        # Split agents panel into status and performance
        layout["agents_panel"].split_column(
            Layout(name="agent_status"),
            Layout(name="agent_performance")
        )
        
        # Split actions panel into current actions and command details
        layout["actions_panel"].split_column(
            Layout(name="current_commands", ratio=2),
            Layout(name="command_details", ratio=1)
        )
        
        # Split stats panel into environment and episode stats
        layout["stats_panel"].split_column(
            Layout(name="environment_state"),
            Layout(name="episode_statistics")
        )
        
        return layout
    
    def _update_enhanced_layout(self, layout: Layout):
        """Update layout with comprehensive agent and training information"""
        # Header with session info
        layout["header"].update(Panel(
            Align.center(
                f"🧠 ARIASKA_RL Enhanced Training Dashboard\n"
                f"Episode {self.current_episode}/{self.episodes} | Session {self.session_id}\n"
                f"🎯 Target: {self.current_target} | 🤖 Agents: {len(self.agents)}"
            ),
            title="Training Control Center",
            border_style="bright_blue",
            style="bold"
        ))
        
        # Agent Status Table - Compact but informative
        agent_table = Table(show_header=True, header_style="bold cyan", box=box.ROUNDED)
        agent_table.add_column("Agent", style="white", width=10)
        agent_table.add_column("Skill", style="green", width=6)
        agent_table.add_column("GPT", style="yellow", width=5)
        agent_table.add_column("Reward", style="blue", width=8)
        agent_table.add_column("Cmds", style="magenta", width=5)
        
        for name, agent in self.agents.items():
            short_name = name.replace("Agent", "")
            agent_table.add_row(
                short_name,
                f"{agent.skill_level:.2f}",
                f"{agent.gpt_dependency:.2f}",
                f"{self.total_rewards[name]:.0f}",
                str(len(agent.command_history))
            )
        
        layout["agent_status"].update(Panel(
            agent_table, 
            title="🤖 Agent Status", 
            border_style="green"
        ))
        
        # Agent Performance Averages
        if self.episode_reports:
            perf_table = Table(show_header=True, header_style="bold yellow", box=box.SIMPLE)
            perf_table.add_column("Agent", style="white", width=10)
            perf_table.add_column("Avg Reward", style="green", width=10)
            perf_table.add_column("Success %", style="blue", width=9)
            perf_table.add_column("Commands/Ep", style="cyan", width=11)
            
            for name, agent in self.agents.items():
                agent_actions = []
                for report in self.episode_reports:
                    agent_actions.extend([a for a in report.major_actions if a.agent_name == name])
                
                if agent_actions:
                    avg_reward = sum(a.reward for a in agent_actions) / len(agent_actions)
                    success_rate = sum(1 for a in agent_actions if a.success) / len(agent_actions) * 100
                    cmds_per_ep = len(agent_actions) / len(self.episode_reports)
                else:
                    avg_reward, success_rate, cmds_per_ep = 0, 0, 0
                
                short_name = name.replace("Agent", "")
                perf_table.add_row(
                    short_name,
                    f"{avg_reward:.1f}",
                    f"{success_rate:.0f}%",
                    f"{cmds_per_ep:.1f}"
                )
            
            layout["agent_performance"].update(Panel(
                perf_table,
                title="📊 Performance Averages",
                border_style="yellow"
            ))
        else:
            layout["agent_performance"].update(Panel(
                "No performance data yet...",
                title="📊 Performance Averages",
                border_style="yellow"
            ))
        
        # Current Commands - Show what each agent is doing RIGHT NOW
        if self.current_actions:
            cmd_table = Table(show_header=True, header_style="bold white", box=box.DOUBLE_EDGE)
            cmd_table.add_column("Agent", style="cyan", width=8)
            cmd_table.add_column("Command", style="white", width=15)
            cmd_table.add_column("Target", style="green", width=12)
            cmd_table.add_column("Reward", style="blue", width=6)
            cmd_table.add_column("Success", style="yellow", width=7)
            
            for action in list(self.current_actions.values())[-5:]:  # Last 5 actions
                short_name = action.agent_name.replace("Agent", "")
                success_icon = "✅" if action.success else "❌"
                cmd_table.add_row(
                    short_name,
                    action.command[:13] + "..." if len(action.command) > 15 else action.command,
                    action.target,
                    f"{action.reward:.1f}",
                    success_icon
                )
            
            layout["current_commands"].update(Panel(
                cmd_table,
                title="⚡ Live Command Execution",
                border_style="bright_magenta"
            ))
        else:
            layout["current_commands"].update(Panel(
                "Waiting for agent actions...",
                title="⚡ Live Command Execution",
                border_style="bright_magenta"
            ))
        
        # Command Details - Show actual outputs
        if self.current_actions:
            latest_action = list(self.current_actions.values())[-1]
            detail_text = (
                f"🔧 Last Command: {latest_action.command}\n"
                f"🎯 Target: {latest_action.target}\n"
                f"🏆 Reward: {latest_action.reward:.1f}\n"
                f"📤 Output: {latest_action.output_summary[:60]}..."
            )
            layout["command_details"].update(Panel(
                detail_text,
                title=f"🔍 {latest_action.agent_name} Details",
                border_style="bright_green"
            ))
        else:
            layout["command_details"].update(Panel(
                "No command details available yet",
                title="🔍 Command Details",
                border_style="bright_green"
            ))
        
        # Environment State - What's been discovered
        discoveries_text = ""
        if self.environment.discovered_services:
            services = list(self.environment.discovered_services)[:3]
            discoveries_text += f"🔍 Services: {', '.join(services)}\n"
        
        if self.environment.discovered_credentials:
            creds = list(self.environment.discovered_credentials)[:3]  
            discoveries_text += f"🔑 Credentials: {', '.join(creds)}\n"
        
        if self.environment.discovered_vulnerabilities:
            vulns = list(self.environment.discovered_vulnerabilities)[:2]
            discoveries_text += f"⚠️ Vulnerabilities: {', '.join(vulns)}\n"
        
        if not discoveries_text:
            discoveries_text = "🔍 No discoveries yet...\n"
        
        discoveries_text += f"\n🚨 Detection Level: {self.environment.detection_level:.0%}"
        discoveries_text += f"\n🎯 Active Target: {self.current_target}"
        
        layout["environment_state"].update(Panel(
            discoveries_text,
            title="🌐 Environment Intel",
            border_style="bright_blue"
        ))
        
        # Episode Statistics
        if self.episode_reports:
            stats_table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE)
            stats_table.add_column("Ep", style="white", width=3)
            stats_table.add_column("Dur", style="green", width=5)
            stats_table.add_column("Cmds", style="cyan", width=4)
            stats_table.add_column("Acc%", style="yellow", width=4)
            stats_table.add_column("Rew", style="blue", width=5)
            
            for report in self.episode_reports[-6:]:  # Last 6 episodes
                accuracy = report.accuracy * 100
                stats_table.add_row(
                    str(report.episode),
                    f"{report.duration:.1f}s",
                    str(report.total_commands),
                    f"{accuracy:.0f}%",
                    f"{report.total_reward:.0f}"
                )
            
            layout["episode_statistics"].update(Panel(
                stats_table,
                title="📈 Episode History",
                border_style="magenta"
            ))
        else:
            layout["episode_statistics"].update(Panel(
                "No episodes completed yet",
                title="📈 Episode History", 
                border_style="magenta"
            ))
        
        # Footer with comprehensive summary
        if self.episode_reports:
            total_commands = sum(r.total_commands for r in self.episode_reports)
            avg_accuracy = sum(r.accuracy for r in self.episode_reports) / len(self.episode_reports) * 100
            total_discoveries = (len(self.environment.discovered_services) + 
                               len(self.environment.discovered_credentials) + 
                               len(self.environment.discovered_vulnerabilities))
            total_patterns = sum(len(agent.learned_patterns) for agent in self.agents.values())
            
            footer_text = (
                f"📊 Session Summary: {len(self.episode_reports)} episodes | "
                f"Avg Accuracy: {avg_accuracy:.1f}% | Total Commands: {total_commands} | "
                f"🔍 Discoveries: {total_discoveries} | 🧠 Patterns Learned: {total_patterns}"
            )
        else:
            footer_text = "🚀 Training session initializing... Agents preparing for intelligent cybersecurity operations"
        
        layout["footer"].update(Panel(
            Align.center(footer_text),
            title="Session Dashboard",
            border_style="dim"
        ))
    
    def _run_episode(self, episode: int) -> EpisodeReport:
        """Run a single training episode with detailed tracking"""
        episode_start = time.time()
        episode_actions = []
        episode_discoveries = []
        
        # Set new target for this episode
        self.current_target = random.choice(self.environment.targets)
        self.environment.current_target = self.current_target
        
        # Get initial state
        state = self.environment.get_state()
        
        # Run steps for this episode
        steps = random.randint(8, 15)  # Variable episode length
        
        for step in range(steps):
            # Each agent takes an action
            for agent_name, agent in self.agents.items():
                # Select action
                available_commands = agent.command_sets.get(agent_name, ["ping"])
                action = agent.select_action(state, available_commands)
                
                # Execute command in environment
                full_output, reward, success = self.environment.execute_command(agent_name, action)
                
                # Create action record
                agent_action = AgentAction(
                    agent_name=agent_name,
                    episode=episode,
                    step=step,
                    target=self.current_target,
                    command=action,
                    full_output=full_output,
                    output_summary=full_output[:50] + "..." if len(full_output) > 50 else full_output,
                    reward=reward,
                    success=success,
                    timestamp=time.time(),
                    learning_improvement=reward > agent.last_reward
                )
                
                episode_actions.append(agent_action)
                self.current_actions[agent_name] = agent_action
                
                # Update agent
                next_state = self.environment.get_state()
                agent.update(state, action, reward, next_state, step == steps - 1)
                
                # Update total rewards
                self.total_rewards[agent_name] += reward
                
                # Log action
                self.logger.info(f"Episode {episode}, Step {step}: {agent_name} executed '{action}' on {self.current_target}, reward: {reward:.1f}")
        
        # Calculate episode metrics
        episode_duration = time.time() - episode_start
        total_commands = len(episode_actions)
        successful_commands = sum(1 for action in episode_actions if action.success)
        total_reward = sum(action.reward for action in episode_actions)
        accuracy = successful_commands / total_commands if total_commands > 0 else 0
        
        # Agent performances
        agent_performances = {}
        for agent_name, agent in self.agents.items():
            agent_actions = [a for a in episode_actions if a.agent_name == agent_name]
            agent_performances[agent_name] = {
                "commands": len(agent_actions),
                "successful": sum(1 for a in agent_actions if a.success),
                "total_reward": sum(a.reward for a in agent_actions),
                "skill_level": agent.skill_level,
                "gpt_dependency": agent.gpt_dependency,
                "patterns_learned": len(agent.learned_patterns)
            }
        
        # Create episode report
        report = EpisodeReport(
            episode=episode,
            duration=episode_duration,
            total_commands=total_commands,
            successful_commands=successful_commands,
            total_reward=total_reward,
            accuracy=accuracy,
            agent_performances=agent_performances,
            discoveries=list(self.environment.discovered_services | self.environment.discovered_credentials | self.environment.discovered_vulnerabilities),
            major_actions=episode_actions[-5:]  # Keep last 5 actions as major actions
        )
        
        return report
    
    def _load_agent_memories(self):
        """Load agent memories from disk"""
        memory_dir = Path("core/memories")
        memory_dir.mkdir(exist_ok=True)
        
        for name, agent in self.agents.items():
            memory_file = memory_dir / f"{name}_memory.json"
            if memory_file.exists():
                try:
                    with open(memory_file, 'r') as f:
                        memory_data = json.load(f)
                        agent.learned_patterns = memory_data.get("learned_patterns", {})
                        agent.skill_level = memory_data.get("skill_level", 0.0)
                        agent.gpt_dependency = memory_data.get("gpt_dependency", 1.0)
                        self.logger.info(f"Loaded memory for {name}: {len(agent.learned_patterns)} patterns")
                except Exception as e:
                    self.logger.warning(f"Failed to load memory for {name}: {e}")
    
    def _save_agent_memories(self):
        """Save agent memories to disk"""
        memory_dir = Path("core/memories")
        memory_dir.mkdir(exist_ok=True)
        
        for name, agent in self.agents.items():
            memory_file = memory_dir / f"{name}_memory.json"
            memory_data = {
                "learned_patterns": agent.learned_patterns,
                "skill_level": agent.skill_level,
                "gpt_dependency": agent.gpt_dependency,
                "command_history": agent.command_history[-100:],  # Keep last 100 commands
                "last_updated": int(time.time())
            }
            
            try:
                with open(memory_file, 'w') as f:
                    json.dump(memory_data, f, indent=2)
                self.logger.info(f"Saved memory for {name}: {len(agent.learned_patterns)} patterns")
            except Exception as e:
                self.logger.error(f"Failed to save memory for {name}: {e}")
    
    def _generate_final_results(self, duration: float) -> Dict[str, Any]:
        """Generate comprehensive final results"""
        if not self.episode_reports:
            return {"summary": "No episodes completed", "detailed": {}}
        
        # Calculate overall statistics
        total_commands = sum(report.total_commands for report in self.episode_reports)
        total_successful = sum(report.successful_commands for report in self.episode_reports)
        avg_accuracy = sum(report.accuracy for report in self.episode_reports) / len(self.episode_reports)
        total_discoveries = len(self.environment.discovered_services) + len(self.environment.discovered_credentials) + len(self.environment.discovered_vulnerabilities)
        
        # Agent-specific statistics
        agent_stats = {}
        for name, agent in self.agents.items():
            agent_stats[name] = {
                "final_skill_level": agent.skill_level,
                "gpt_dependency_reduction": 1.0 - agent.gpt_dependency,
                "patterns_learned": len(agent.learned_patterns),
                "total_reward": self.total_rewards[name],
                "command_count": len(agent.command_history),
                "success_rate": sum(agent.success_history) / len(agent.success_history) if agent.success_history else 0
            }
        
        results = {
            "summary": f"Completed {len(self.episode_reports)} episodes with {avg_accuracy:.1%} accuracy",
            "detailed": {
                "session_info": {
                    "session_id": self.session_id,
                    "duration": duration,
                    "episodes": len(self.episode_reports),
                    "total_commands": total_commands,
                    "target": self.current_target
                },
                "performance": {
                    "success_rate": total_successful / total_commands if total_commands > 0 else 0,
                    "average_accuracy": avg_accuracy,
                    "total_discoveries": total_discoveries,
                    "total_rewards": sum(self.total_rewards.values())
                },
                "learning_progress": agent_stats,
                "episode_progression": [
                    {
                        "episode": report.episode,
                        "reward": report.total_reward,
                        "accuracy": report.accuracy,
                        "discoveries": len(report.discoveries)
                    }
                    for report in self.episode_reports
                ],
                "discoveries": {
                    "services": list(self.environment.discovered_services),
                    "credentials": list(self.environment.discovered_credentials),
                    "vulnerabilities": list(self.environment.discovered_vulnerabilities)
                }
            }
        }
        
        # Save final results
        results_path = Path("logs") / f"enhanced_results_{self.session_id}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results


def create_enhanced_trainer(episodes: int) -> EnhancedUnifiedTrainer:
    """Factory function to create enhanced trainer"""
    return EnhancedUnifiedTrainer(episodes=episodes)


if __name__ == "__main__":
    # Direct execution for testing
    if len(sys.argv) > 1:
        episodes = int(sys.argv[1])
    else:
        episodes = 3
    
    trainer = EnhancedUnifiedTrainer(episodes=episodes)
    results = trainer.train()
    print(f"\\nTraining completed: {results['summary']}")
