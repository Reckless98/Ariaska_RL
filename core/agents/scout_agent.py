# core/agents/scout_agent.py — ARIASKA ScoutAgent v1.0
# 🔍 Recon Specialist | GPT-Enhanced Scanning Planner | Port Prioritization Engine

import os
import torch
import random
from typing import List, Dict, Any, Optional, Union
from rich.console import Console
from rich.table import Table

from core.interfaces.agent_interface import AgentInterface
from core.interfaces.memory_sync_interface import MemorySyncInterface
from core.utils.llm_orchestrator import LLMOrchestrator
from core.utils.replay_buffer import ReplayBuffer
from core.gpt_manager import GPTManager
from core.multiagent.memory_router import MemoryRouter

console = Console()

class ScoutAgent(AgentInterface, MemorySyncInterface):
    """
    ScoutAgent: Reconnaissance specialist that plans port-scanning and fingerprinting
    using LLM-assisted decision making.
    
    - Uses LLM to formulate efficient scanning plans based on current environment state
    - Prioritizes targets based on potential vulnerabilities
    - Applies stealth techniques (scan timing randomization, less noisy methods)
    - Interfaces with MemoryRouter for sharing discovered information
    """
    
    @property
    def agent_id(self):
        return self._agent_id
    
    @agent_id.setter
    def agent_id(self, value):
        self._agent_id = value
    
    @property
    def role(self):
        return self._role
    
    @role.setter
    def role(self, value):
        self._role = value
    
    def __init__(
        self,
        agent_id: str = "ScoutAgent",
        role: str = "ReconSpecialist",
        agent_manager=None,
        memory_router=None,
        verbosity: str = "standard"
    ):
        self._agent_id = agent_id
        self._role = role
        self.agent_manager = agent_manager
        self.memory_router = memory_router or MemoryRouter()
        self.verbosity = verbosity
        
        # Initialize LLM orchestrator for intelligent planning
        self.llm_router = LLMOrchestrator(cache_dir="cache/scout_agent_responses")
        
        # Initialize GPT manager for fallback and advanced reasoning
        self.gpt_manager = GPTManager()
        
        # Store recent scan plans and discoveries
        self.scan_history = []
        self.discovered_hosts = []
        self.discovered_services = {}
        self.current_phase = "recon"
        
        # Stealth parameters
        self.stealth_mode = False  # Set to True for quieter scanning
        self.randomize_timing = True  # Randomize timing between scans
        self.noise_level = 1  # 1-10 scale (1=quiet, 10=noisy)
        
        # Command history for redundancy detection
        self.command_history = []
        
        # Memory store for scan results
        self.replay_buffer = ReplayBuffer(
            capacity=500,
            use_sqlite=True,
            db_path="core/memories/scout_memory/replay_buffer.sqlite3"
        )
        
        # Default target for simulation
        self.default_target = "10.10.10.10"
        
        # Link to other agents (will be set in _init_multiagent_links)
        self.red_agent = None
        self.shadow_agent = None
        self.orion_agent = None
        self.blue_agent = None
        
        # Configuration presets for different scan strategies
        self.scan_strategies = {
            "stealth": {
                "timing": "slow",
                "techniques": ["SYN", "NULL", "FIN"],
                "packet_rate": "low",
                "randomize_ports": True,
                "avoid_ids_patterns": True
            },
            "balanced": {
                "timing": "medium",
                "techniques": ["SYN", "TCP"],
                "packet_rate": "medium",
                "randomize_ports": True,
                "avoid_ids_patterns": False
            },
            "aggressive": {
                "timing": "fast",
                "techniques": ["SYN", "TCP", "UDP", "VERSION"],
                "packet_rate": "high",
                "randomize_ports": False,
                "avoid_ids_patterns": False
            }
        }
        
        console.print(f"[green]✓ {self.agent_id} initialized[/green]")
        
    def _init_multiagent_links(self):
        """Initialize links to other agents in the system"""
        if self.agent_manager:
            self.red_agent = self.agent_manager.get_agent("RedAgent")
            self.shadow_agent = self.agent_manager.get_agent("ShadowAgent")
            self.orion_agent = self.agent_manager.get_agent("OrionAgent")
            self.blue_agent = self.agent_manager.get_agent("BlueAgent")
    
    def request_strategy(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Request a scanning strategy from the LLM orchestrator.
        
        Args:
            context: Current environment context
            
        Returns:
            Dict with scan strategy recommendations
        """
        # Get strategy from LLM orchestrator
        result = self.llm_router.request_strategy(
            context=context,
            task_type="scan",
            agent_id=self.agent_id
        )
        
        return result
    
    def advise_phase(self, state: Dict[str, Any], all_agents: Optional[List[Any]] = None) -> str:
        """
        Determine the most appropriate phase for the current state.
        Used by RedAgent to decide which phase to operate in.
        
        Args:
            state: Current environment state
            all_agents: List of all agents in the system
            
        Returns:
            String with recommended phase
        """
        # Extract relevant state information
        open_ports = state.get("open_ports", [])
        services = state.get("services", [])
        privilege_level = state.get("privilege_level", "none") 
        credentials_found = state.get("credentials_found", False)
        data_exfiltrated = state.get("data_exfiltrated", False)
        blue_team_alert = state.get("blue_team_alert", 0)
        detection_risk = state.get("detection_risk", 0)
        
        # Create context for LLM decision
        context = {
            "open_ports": open_ports,
            "services": services,
            "privilege_level": privilege_level,
            "credentials_found": credentials_found,
            "data_exfiltrated": data_exfiltrated,
            "blue_team_alert": blue_team_alert,
            "detection_risk": detection_risk,
            "current_phase": self.current_phase
        }
        
        # Add recent commands from RedAgent if available
        if self.red_agent and hasattr(self.red_agent, "command_history"):
            context["previous_commands"] = self.red_agent.command_history[-5:] if len(self.red_agent.command_history) > 5 else self.red_agent.command_history
        
        # Get phase recommendation from LLM
        phase_prompt = (
            f"As a cybersecurity reconnaissance specialist, analyze the current state:\n"
            f"Open ports: {open_ports}\n"
            f"Services: {services}\n"
            f"Privilege level: {privilege_level}\n"
            f"Credentials found: {credentials_found}\n"
            f"Data exfiltrated: {data_exfiltrated}\n"
            f"Blue team alert level: {blue_team_alert}\n"
            f"Detection risk: {detection_risk}\n\n"
            f"Based on this information, what phase should the red team be operating in?\n"
            f"Choose exactly one of: 'recon', 'enumeration', 'exploit', 'privesc', 'exfiltrate'.\n"
            f"Respond with ONLY the phase name, no explanation."
        )
        
        try:
            response = self.llm_router.route_task(
                task_type="tactical",
                prompt=phase_prompt,
                agent_id=self.agent_id
            )
            
            # Clean and validate response
            response = response.strip().lower()
            valid_phases = ["recon", "enumeration", "exploit", "privesc", "exfiltrate"]
            
            # Check if response contains a valid phase
            for phase in valid_phases:
                if phase in response:
                    self.current_phase = phase
                    if self.verbosity in ("standard", "verbose", "debug"):
                        console.print(f"[blue]🔍 ScoutAgent advises phase: {phase}[/blue]")
                    return phase
            
            # Fallback if response doesn't contain a valid phase
            console.print(f"[yellow]⚠ Invalid phase recommendation: {response}. Using current phase.[/yellow]")
            return self.current_phase
            
        except Exception as e:
            console.print(f"[yellow]⚠ Error in phase recommendation: {e}. Using current phase.[/yellow]")
            return self.current_phase
    
    def plan_scan(self, target_ip: str, discovered_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Plan a scanning strategy for a target based on current information.
        
        Args:
            target_ip: Target IP address
            discovered_info: Information already discovered about the target
            
        Returns:
            Dict with scan plan details
        """
        context = {
            "target_ip": target_ip,
            "discovered_hosts": self.discovered_hosts,
            "discovered_services": self.discovered_services,
            "previous_actions": self.scan_history[-5:] if len(self.scan_history) > 5 else self.scan_history,
            "stealth_mode": self.stealth_mode,
            "noise_level": self.noise_level,
            "phase": self.current_phase
        }
        
        # Add any additional discovered info
        if discovered_info:
            context.update(discovered_info)
        
        # Get scan strategy from LLM
        scan_strategy = self.request_strategy(context)
        
        # If in stealth mode, modify strategy to be quieter
        if self.stealth_mode and self.shadow_agent:
            shadow_feedback = self.shadow_agent.evaluate_scan_plan(scan_strategy)
            if shadow_feedback.get("too_noisy", False):
                # Try again with explicit stealth requirement
                context["stealth_required"] = True
                context["stealth_reason"] = shadow_feedback.get("reason", "High detection risk")
                scan_strategy = self.request_strategy(context)
        
        # Save to history
        if "command" in scan_strategy:
            self.scan_history.append(scan_strategy["command"])
            self.command_history.append(scan_strategy["command"])
        
        return scan_strategy
    
    def simulate_step(self, episode: int = 1, step: int = 1, shared_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Simulate a step for this agent in the training loop.
        
        Args:
            episode: Current episode number
            step: Current step number
            shared_context: Shared context from agent manager
            
        Returns:
            Dict with step results
        """
        # Get current environment state from RedAgent if available
        state = {}
        if self.red_agent and hasattr(self.red_agent, "env"):
            state = self.red_agent.env.get_global_state()
        elif shared_context:
            state = shared_context
        
        # Update discovered information from state
        if "open_ports" in state:
            target_ip = state.get("target_ip", self.default_target)
            if target_ip not in self.discovered_hosts:
                self.discovered_hosts.append(target_ip)
            
            if target_ip not in self.discovered_services:
                self.discovered_services[target_ip] = []
            
            # Add services if available
            if "services" in state:
                for service in state["services"]:
                    if service not in self.discovered_services[target_ip]:
                        self.discovered_services[target_ip].append(service)
        
        # Get current phase recommendation
        phase = self.advise_phase(state, all_agents=None)
        
        # Plan the next scan
        target_ip = state.get("target_ip", self.default_target)
        scan_plan = self.plan_scan(target_ip, discovered_info=state)
        
        # Extract command from scan plan
        command = scan_plan.get("command", f"nmap -sS {target_ip}")
        
        # Adjust stealth parameters based on ShadowAgent if available
        if self.shadow_agent and hasattr(self.shadow_agent, "get_stealth_recommendation"):
            stealth_rec = self.shadow_agent.get_stealth_recommendation()
            self.stealth_mode = stealth_rec.get("stealth_mode", self.stealth_mode)
            self.noise_level = stealth_rec.get("recommended_noise_level", self.noise_level)
        
        # Apply scan timing randomization if enabled
        if self.randomize_timing and random.random() < 0.3:
            command = self._add_timing_parameter(command)
        
        # Log to memory router
        if self.memory_router:
            scan_target = scan_plan.get("targets", [target_ip])[0] if scan_plan.get("targets") else target_ip
            scan_ports = scan_plan.get("ports", [])
            scan_type = scan_plan.get("scan_type", "tcp")
            
            self.memory_router.log_scout_scan(
                target=scan_target,
                ports=scan_ports,
                scan_type=scan_type,
                command=command,
                reasoning=scan_plan.get("reasoning", ""),
                step=step,
                episode=episode
            )
        
        # Update replay buffer
        experience = {
            "command": command,
            "phase": phase,
            "stealth_mode": self.stealth_mode,
            "noise_level": self.noise_level,
            "target": target_ip,
            "discovered": {
                "hosts": self.discovered_hosts,
                "services": self.discovered_services
            }
        }
        
        self.replay_buffer.add(experience)
        
        # Display information based on verbosity
        if self.verbosity in ("standard", "verbose", "debug"):
            console.print(f"[blue]🔍 ScoutAgent step {step}: {command}[/blue]")
            
        if self.verbosity in ("verbose", "debug"):
            table = Table(title=f"🔍 ScoutAgent Scan Plan (Step {step})")
            table.add_column("Field", style="cyan")
            table.add_column("Value", style="green")
            table.add_row("Phase", phase)
            table.add_row("Command", command)
            table.add_row("Stealth Mode", str(self.stealth_mode))
            table.add_row("Noise Level", str(self.noise_level))
            console.print(table)
        
        # Return step information
        return {
            "agent_id": self.agent_id,
            "phase": phase,
            "command": command,
            "stealth_mode": self.stealth_mode,
            "noise_level": self.noise_level,
            "discovered_hosts": self.discovered_hosts,
            "discovered_services": self.discovered_services,
            "step": step,
            "episode": episode
        }
    
    def _add_timing_parameter(self, command: str) -> str:
        """
        Add timing parameter to nmap command for stealth.
        
        Args:
            command: Original command
            
        Returns:
            Modified command with timing parameters
        """
        if "nmap" in command and "-T" not in command:
            timing_level = 1 if self.stealth_mode else random.randint(2, 4)
            command = command.replace("nmap", f"nmap -T{timing_level}")
        return command
    
    def sync_memory(self):
        """
        Synchronize memory with the global memory router.
        Implementation of MemorySyncInterface.
        """
        if self.memory_router:
            # Share discovered hosts and services with global memory
            self.memory_router.update_scout_discoveries(
                self.agent_id,
                self.discovered_hosts,
                self.discovered_services
            )
    
    def get_port_priority(self, ports: List[int]) -> List[int]:
        """
        Prioritize ports based on their likely vulnerability.
        
        Args:
            ports: List of port numbers
            
        Returns:
            Sorted list of ports by priority
        """
        # Common vulnerable services and their ports
        high_priority = [22, 21, 80, 443, 8080, 8443, 3389, 5985, 5986]  # SSH, FTP, HTTP, RDP, WinRM
        medium_priority = [25, 110, 143, 445, 1433, 3306, 5432]  # Mail, SMB, SQL
        
        # Sort ports by priority
        result = []
        result.extend([p for p in ports if p in high_priority])
        result.extend([p for p in ports if p in medium_priority])
        result.extend([p for p in ports if p not in high_priority and p not in medium_priority])
        
        return result
    
    def get_base_commands(self):
        """Return base reconnaissance commands for CLI completion."""
        return [
            "nmap -sS",
            "nmap -sT", 
            "nmap -sU",
            "nmap -sV",
            "nmap -O",
            "nmap -A",
            "masscan",
            "gobuster",
            "nikto",
            "enum4linux",
            "smbmap",
            "smbclient",
            "wpscan",
            "ffuf",
            "dirb"
        ]
    
    def reset(self):
        """Reset agent state for a new episode."""
        self.scan_history = []
        self.command_history = []

# Main execution for testing
if __name__ == "__main__":
    agent = ScoutAgent(verbosity="verbose")
    
    # Test scan planning
    target = "10.10.10.10"
    discovered_info = {
        "open_ports": [22, 80, 443],
        "services": ["ssh", "http", "https"],
        "phase": "recon"
    }
    
    scan_plan = agent.plan_scan(target, discovered_info)
    console.print(f"[bold cyan]Scan plan:[/bold cyan]")
    console.print(scan_plan)
    
    # Test phase recommendation
    state = {
        "open_ports": [22, 80, 443],
        "services": ["ssh", "http", "https"],
        "privilege_level": "user",
        "credentials_found": True,
        "data_exfiltrated": False,
        "blue_team_alert": 15.0,
        "detection_risk": 25.0
    }
    
    phase = agent.advise_phase(state)
    console.print(f"[bold green]Recommended phase:[/bold green] {phase}")
    
    # Test simulation step
    result = agent.simulate_step(episode=1, step=1, shared_context=state)
    console.print(f"[bold magenta]Simulation result:[/bold magenta]")
    console.print(result)
