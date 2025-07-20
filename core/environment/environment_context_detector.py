# core/environment/environment_context_detector.py — ARIASKA Environment Context Detector v1.0
# 🌐 Environment Mode Detection | 🔐 Safety Controls | 🏫 Curriculum Management

import os
import json
import random
import time
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from rich.console import Console

console = Console()

class EnvironmentContextDetector:
    """
    Detects and adapts to environment context (simulated vs live mode).
    
    Features:
    - Auto-adjusts agent strategies for safety based on environment type
    - Enforces curriculum scheduling and domain randomization
    - Manages safety boundaries between simulated and live environments
    - Provides environmental awareness to agents for context-aware decision making
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the environment context detector.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config_path = config_path or os.path.join("config", "environment.json")
        self.config = self._load_config()
        self.current_mode = self.config.get("default_mode", "simulated")
        self.safety_level = self.config.get("safety_level", "strict")
        self.curriculum_stage = self.config.get("curriculum_stage", 1)
        self.domain_randomization = self.config.get("domain_randomization", True)
        
        # Enhanced curriculum parameters
        self.curriculum_history = self.config.get("curriculum_history", [])
        self.min_success_rate = self.config.get("min_success_rate", 0.7)  # 70% success rate to advance
        self.consecutive_successes = self.config.get("consecutive_successes", 0)
        self.success_threshold = self.config.get("success_threshold", 3)  # Need 3 consecutive successes
        self.last_curriculum_update = time.time()
        
        # Load target configurations
        self.targets = self.config.get("targets", {})
        
        # Track environment parameters for dynamic adjustment
        self.environment_parameters = {
            "ports": self.config.get("ports", [22, 80, 443]),
            "services": self.config.get("services", ["ssh", "http", "https"]),
            "os_types": self.config.get("os_types", ["linux"]),
            "difficulty": self.curriculum_stage,
            "vulnerability_density": 0.2 + (0.1 * self.curriculum_stage),  # Increases with curriculum
            "defense_sophistication": min(0.8, 0.2 + (0.1 * self.curriculum_stage))  # Increases with curriculum
        }
        
        console.print(f"[green]✓ Environment Context Detector initialized: Mode={self.current_mode}, Curriculum Stage={self.curriculum_stage}[/green]")
        
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file or use defaults.
        
        Returns:
            Dict of configuration parameters
        """
        default_config = {
            "default_mode": "simulated",
            "safety_level": "strict",
            "curriculum_stage": 1,
            "domain_randomization": True,
            "ports": [22, 80, 443],
            "services": ["ssh", "http", "https"],
            "os_types": ["linux"],
            "min_success_rate": 0.7,
            "consecutive_successes": 0,
            "success_threshold": 3,
            "curriculum_history": [],
            "targets": {
                "simulated": {
                    "ip": "10.10.10.10",
                    "allowed_actions": ["all"]
                },
                "live": {
                    "ip": "CTF_TARGET",
                    "allowed_actions": ["scan", "enum"]
                }
            }
        }
        
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r") as f:
                    return json.load(f)
            else:
                # Create default config if not exists
                os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
                with open(self.config_path, "w") as f:
                    json.dump(default_config, f, indent=2)
                console.print(f"[yellow]⚠ Created default environment config at {self.config_path}[/yellow]")
                return default_config
        except Exception as e:
            console.print(f"[red]❌ Error loading environment config: {e}. Using defaults.[/red]")
            return default_config
            
    def detect_environment_type(self, target_ip: str = None) -> str:
        """
        Detect environment type based on target IP or configuration.
        
        Args:
            target_ip: Target IP address (optional)
            
        Returns:
            Environment type ('simulated' or 'live')
        """
        # If target IP is provided, check if it matches live target
        if target_ip:
            live_ip = self.targets.get("live", {}).get("ip")
            if live_ip and target_ip == live_ip:
                return "live"
            elif "10." in target_ip or "192.168." in target_ip or "127." in target_ip:
                return "simulated"
                
        # Default to configured mode
        return self.current_mode
        
    def is_action_allowed(self, action: str, environment_type: str = None) -> bool:
        """
        Check if an action is allowed in the current environment.
        
        Args:
            action: Action to check
            environment_type: Environment type to check against (optional)
            
        Returns:
            True if action is allowed, False otherwise
        """
        if environment_type is None:
            environment_type = self.current_mode
            
        target_config = self.targets.get(environment_type, {})
        allowed_actions = target_config.get("allowed_actions", [])
        
        # If all actions are allowed
        if "all" in allowed_actions:
            return True
            
        # Check if action type is allowed
        action_type = self._get_action_type(action)
        return action_type in allowed_actions
        
    def _get_action_type(self, action: str) -> str:
        """
        Determine the type of an action.
        
        Args:
            action: Action to categorize
            
        Returns:
            Action type ('scan', 'enum', 'exploit', 'privesc', or 'exfil')
        """
        action = action.lower()
        if any(cmd in action for cmd in ["nmap", "masscan", "ping"]):
            return "scan"
        elif any(cmd in action for cmd in ["gobuster", "enum4linux", "wpscan", "dirb"]):
            return "enum"
        elif any(cmd in action for cmd in ["exploit", "msfconsole", "sqlmap", "hydra", "bruteforce"]):
            return "exploit"
        elif any(cmd in action for cmd in ["linpeas", "winpeas", "sudo", "nc -e", "powershell"]):
            return "privesc"
        elif any(cmd in action for cmd in ["zip", "tar", "scp", "exfil", "download"]):
            return "exfil"
        return "unknown"
        
    def adjust_agent_parameters(self, agent, environment_type: str = None) -> None:
        """
        Adjust agent parameters based on environment context.
        
        Args:
            agent: Agent to adjust
            environment_type: Environment type to adjust for (optional)
        """
        if environment_type is None:
            environment_type = self.current_mode
            
        # Adjust exploration/exploitation parameters
        if hasattr(agent, "epsilon"):
            if environment_type == "live":
                # Less exploration in live mode for safety
                agent.epsilon = max(0.05, agent.epsilon * 0.5)
            else:
                # More exploration in simulated mode
                agent.epsilon = min(0.8, agent.epsilon * 1.2)
                
        # Adjust risk tolerance
        if hasattr(agent, "risk_tolerance"):
            if environment_type == "live":
                agent.risk_tolerance = 0.3  # Low risk tolerance in live mode
            else:
                agent.risk_tolerance = 0.7  # Higher risk tolerance in simulated mode
                
        # Adjust command patterns
        if hasattr(agent, "command_pattern"):
            if environment_type == "live":
                agent.command_pattern = "stealth"
            else:
                agent.command_pattern = "aggressive"

        # Adjust learning rate based on curriculum stage
        if hasattr(agent, "learning_rate"):
            # Lower learning rate as curriculum advances
            agent.learning_rate = max(0.001, 0.1 / (1 + 0.2 * self.curriculum_stage))
                
        console.print(f"[blue]🔧 Adjusted {agent.agent_id} parameters for {environment_type} environment (curriculum stage {self.curriculum_stage})[/blue]")
        
    def get_environment_context(self, target_ip: str = None) -> Dict[str, Any]:
        """
        Get complete environment context data.
        
        Args:
            target_ip: Target IP address (optional)
            
        Returns:
            Dict of environment context data
        """
        environment_type = self.detect_environment_type(target_ip)
        
        context = {
            "mode": environment_type,
            "safety_level": self.safety_level,
            "curriculum_stage": self.curriculum_stage,
            "parameters": self.environment_parameters,
            "target": self.targets.get(environment_type, {}),
            "is_simulated": environment_type == "simulated",
            "vulnerability_density": self.environment_parameters["vulnerability_density"],
            "defense_sophistication": self.environment_parameters["defense_sophistication"]
        }
        
        return context
        
    def advance_curriculum(self) -> None:
        """
        Advance to the next curriculum stage.
        Increases difficulty and complexity of the environment.
        """
        self.curriculum_stage += 1
        self.environment_parameters["difficulty"] = self.curriculum_stage
        
        # Track when this curriculum stage was reached
        self.curriculum_history.append({
            "stage": self.curriculum_stage,
            "timestamp": time.time(),
            "reason": "manual_advance"
        })
        
        # Add more complex parameters as curriculum advances
        if self.curriculum_stage >= 2:
            self.environment_parameters["ports"].extend([8080, 3306])
            self.environment_parameters["services"].extend(["mysql", "proxy"])
            
        if self.curriculum_stage >= 3:
            self.environment_parameters["os_types"].extend(["windows"])
            self.environment_parameters["ports"].extend([445, 3389])
            self.environment_parameters["services"].extend(["smb", "rdp"])
            
        if self.curriculum_stage >= 4:
            self.environment_parameters["ports"].extend([21, 25, 53, 139])
            self.environment_parameters["services"].extend(["ftp", "smtp", "dns", "netbios"])
            
        # Update vulnerability and defense parameters
        self.environment_parameters["vulnerability_density"] = min(0.8, 0.2 + (0.1 * self.curriculum_stage))
        self.environment_parameters["defense_sophistication"] = min(0.8, 0.2 + (0.1 * self.curriculum_stage))
        
        console.print(f"[green]📚 Advanced to curriculum stage {self.curriculum_stage}[/green]")
        
        # Reset consecutive successes counter
        self.consecutive_successes = 0
        
        # Save updated configuration
        self.config["curriculum_stage"] = self.curriculum_stage
        self.config["curriculum_history"] = self.curriculum_history
        self.config["consecutive_successes"] = self.consecutive_successes
        self._save_config()
        
    def evaluate_curriculum_advancement(self, mission_success: bool, metrics: Dict[str, float]) -> bool:
        """
        Evaluate if curriculum should be advanced based on agent performance.
        
        Args:
            mission_success: Whether the mission was successful
            metrics: Performance metrics from the episode
            
        Returns:
            True if curriculum was advanced, False otherwise
        """
        # Reset consecutive successes if mission failed
        if not mission_success:
            self.consecutive_successes = 0
            self.config["consecutive_successes"] = 0
            self._save_config()
            return False
            
        # Check if minimum cooldown period has passed (12 hours)
        if time.time() - self.last_curriculum_update < (12 * 3600):
            return False
            
        # Increment consecutive successes
        self.consecutive_successes += 1
        self.config["consecutive_successes"] = self.consecutive_successes
        
        # Check for advancement criteria
        should_advance = False
        
        # Check success rate and stealth metrics
        success_rate = metrics.get("success_rate", 0.0)
        stealth_score = metrics.get("stealth_score", 0.0)
        
        # Advanced criteria - must have consecutive successes and good metrics
        if (self.consecutive_successes >= self.success_threshold and 
            success_rate >= self.min_success_rate and
            stealth_score >= (0.5 + 0.1 * self.curriculum_stage)):
            should_advance = True
            
        if should_advance:
            # Record reason for advancement
            advancement_reason = {
                "consecutive_successes": self.consecutive_successes,
                "success_rate": success_rate,
                "stealth_score": stealth_score
            }
            
            # Update tracking vars
            self.last_curriculum_update = time.time()
            
            # Perform the advancement
            self.curriculum_stage += 1
            self.environment_parameters["difficulty"] = self.curriculum_stage
            
            # Log curriculum history
            self.curriculum_history.append({
                "stage": self.curriculum_stage,
                "timestamp": time.time(),
                "reason": "auto_advance",
                "metrics": advancement_reason
            })
            
            # Update parameters for new curriculum stage
            self._update_curriculum_parameters()
            
            # Reset consecutive successes
            self.consecutive_successes = 0
            
            # Save updated configuration
            self.config["curriculum_stage"] = self.curriculum_stage
            self.config["curriculum_history"] = self.curriculum_history
            self.config["consecutive_successes"] = self.consecutive_successes
            self._save_config()
            
            console.print(f"[green bold]📚 CURRICULUM AUTOMATICALLY ADVANCED TO STAGE {self.curriculum_stage}[/green bold]")
            console.print(f"[green]Based on: {advancement_reason}[/green]")
            
            return True
            
        self._save_config()
        return False
        
    def _update_curriculum_parameters(self) -> None:
        """Update environment parameters based on current curriculum stage."""
        # Base ports and services for curriculum level 1
        base_ports = [22, 80, 443]
        base_services = ["ssh", "http", "https"]
        
        # Add ports and services based on curriculum stage
        if self.curriculum_stage >= 2:
            base_ports.extend([8080, 3306])
            base_services.extend(["mysql", "proxy"])
            
        if self.curriculum_stage >= 3:
            base_ports.extend([445, 3389])
            base_services.extend(["smb", "rdp"])
            
        if self.curriculum_stage >= 4:
            base_ports.extend([21, 25, 53, 139])
            base_services.extend(["ftp", "smtp", "dns", "netbios"])
            
        if self.curriculum_stage >= 5:
            base_ports.extend([23, 161, 1433, 5432])
            base_services.extend(["telnet", "snmp", "mssql", "postgres"])
            
        # Update environment parameters
        self.environment_parameters["ports"] = base_ports
        self.environment_parameters["services"] = base_services
        self.environment_parameters["os_types"] = ["linux"] if self.curriculum_stage < 3 else ["linux", "windows"]
        self.environment_parameters["vulnerability_density"] = min(0.8, 0.2 + (0.1 * self.curriculum_stage))
        self.environment_parameters["defense_sophistication"] = min(0.8, 0.2 + (0.1 * self.curriculum_stage))
        
    def randomize_domain(self) -> Dict[str, Any]:
        """
        Randomize domain parameters for training variety.
        
        Returns:
            Dict of randomized environment parameters
        """
        if not self.domain_randomization:
            return self.environment_parameters
            
        # Create a copy to avoid modifying the original
        params = self.environment_parameters.copy()
        
        # Build port pools based on curriculum stage
        common_ports = [21, 22, 23, 25, 80, 443, 445, 3306, 8080, 8443]
        advanced_ports = [53, 88, 110, 135, 139, 389, 1433, 3389, 5432, 5900]
        rare_ports = [161, 623, 1521, 2049, 3260, 5060, 5601, 6379, 9000, 27017]
        
        # Get potential ports based on curriculum
        potential_ports = common_ports
        if self.curriculum_stage >= 3:
            potential_ports.extend(advanced_ports)
        if self.curriculum_stage >= 5:
            potential_ports.extend(rare_ports)
            
        # Determine number of ports based on curriculum stage
        min_ports = 3 + self.curriculum_stage
        max_ports = 4 + (2 * self.curriculum_stage)
        num_ports = random.randint(min_ports, max_ports)
        
        # Select ports
        params["ports"] = sorted(random.sample(potential_ports, min(num_ports, len(potential_ports))))
        
        # Service pools based on curriculum
        common_services = ["ftp", "ssh", "telnet", "http", "https", "smb", "mysql"]
        advanced_services = ["smtp", "rdp", "ldap", "proxy", "dns", "postgres"]
        rare_services = ["snmp", "mssql", "mongodb", "redis", "elasticsearch", "sip"]
        
        # Get potential services based on curriculum
        potential_services = common_services
        if self.curriculum_stage >= 3:
            potential_services.extend(advanced_services)
        if self.curriculum_stage >= 5:
            potential_services.extend(rare_services)
            
        # Determine number of services based on curriculum stage
        num_services = min(len(params["ports"]), len(potential_services))
        
        # Select services
        params["services"] = random.sample(potential_services, num_services)
        
        # Randomize OS type with bias based on curriculum
        if self.curriculum_stage < 3:
            os_weights = [0.9, 0.1, 0.0]  # Linux, Windows, Mixed
        elif self.curriculum_stage < 5:
            os_weights = [0.6, 0.3, 0.1]
        else:
            os_weights = [0.4, 0.3, 0.3]
            
        os_choice = np.random.choice([0, 1, 2], p=os_weights)
        if os_choice == 0:
            params["os_types"] = ["linux"]
        elif os_choice == 1:
            params["os_types"] = ["windows"]
        else:
            params["os_types"] = ["linux", "windows"]
            
        # Randomize vulnerability density within range for curriculum
        base_density = 0.2 + (0.1 * self.curriculum_stage)
        params["vulnerability_density"] = max(0.1, min(0.8, base_density + random.uniform(-0.1, 0.1)))
        
        # Randomize defense sophistication
        base_defense = 0.2 + (0.1 * self.curriculum_stage)
        params["defense_sophistication"] = max(0.1, min(0.8, base_defense + random.uniform(-0.1, 0.1)))
        
        console.print(f"[blue]🎲 Randomized domain parameters: {len(params['ports'])} ports, {len(params['services'])} services[/blue]")
        console.print(f"[blue]   OS Types: {params['os_types']}, Vuln Density: {params['vulnerability_density']:.2f}, Defense: {params['defense_sophistication']:.2f}[/blue]")
        return params
        
    def _save_config(self) -> None:
        """Save the current configuration to file."""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, "w") as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            console.print(f"[red]❌ Failed to save environment config: {e}[/red]")
            
    def set_mode(self, mode: str) -> None:
        """
        Set the current environment mode.
        
        Args:
            mode: Environment mode ('simulated' or 'live')
        """
        if mode not in ["simulated", "live"]:
            console.print(f"[red]❌ Invalid environment mode: {mode}. Using {self.current_mode}.[/red]")
            return
            
        self.current_mode = mode
        self.config["default_mode"] = mode
        self._save_config()
        console.print(f"[green]✓ Environment mode set to {mode}[/green]")
        
        # If switching to live mode, enforce stricter safety
        if mode == "live":
            self.safety_level = "strict"
            self.config["safety_level"] = "strict"
            self._save_config()
            console.print(f"[red]⚠ Live mode enabled: Safety level set to strict[/red]")
            
    def calculate_stealth_score(self, actions_history: List[Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate a stealth score based on agent actions.
        Higher score means stealthier operation.
        
        Args:
            actions_history: List of agent actions with metadata
            
        Returns:
            Tuple of (stealth_score, stealth_metrics)
        """
        if not actions_history:
            return 10.0, {"details": "No actions to evaluate"}
            
        # Initialize stealth metrics
        base_score = 10.0
        penalties = []
        bonuses = []
        
        # Action type counts
        action_counts = {
            "scan": 0,
            "enum": 0,
            "exploit": 0,
            "privesc": 0,
            "exfil": 0,
            "unknown": 0
        }
        
        # Track noisy actions
        noisy_actions = 0
        stealthy_actions = 0
        redundant_actions = 0
        effective_actions = 0
        
        # Analyze actions
        for action in actions_history:
            cmd = action.get("command", "")
            if not cmd:
                continue
                
            # Categorize action
            action_type = self._get_action_type(cmd)
            action_counts[action_type] = action_counts.get(action_type, 0) + 1
            
            # Check for noisy vs stealthy techniques
            if any(noisy in cmd.lower() for noisy in ["-A", "aggressive", "bruteforce", "hydra", "masscan"]):
                noisy_actions += 1
                penalties.append(("Noisy technique used", 0.5))
            
            if any(stealthy in cmd.lower() for stealthy in ["-sS", "stealth", "passive"]):
                stealthy_actions += 1
                bonuses.append(("Stealth technique used", 0.3))
                
            # Check for redundancy
            if action.get("redundant", False):
                redundant_actions += 1
                penalties.append(("Redundant action", 0.2))
                
            # Check for effectiveness
            if action.get("success", False):
                effective_actions += 1
                
        # Calculate base penalty from action distribution
        total_actions = sum(action_counts.values())
        
        # Penalty for too many scans relative to exploitation
        if action_counts["scan"] > (action_counts["exploit"] * 3) and action_counts["scan"] > 5:
            penalties.append(("Excessive scanning", 1.0))
            
        # Calculate stealth score
        stealth_deduction = sum(penalty[1] for penalty in penalties)
        stealth_bonus = sum(bonus[1] for bonus in bonuses)
        
        # Apply noisy vs stealthy ratio influence
        if noisy_actions + stealthy_actions > 0:
            stealth_ratio = stealthy_actions / (noisy_actions + stealthy_actions)
            stealth_influence = (stealth_ratio - 0.5) * 2.0  # Range from -1.0 to 1.0
            stealth_modifier = stealth_influence * 1.5  # Scale influence
            if stealth_modifier > 0:
                bonuses.append(("Stealth technique ratio", stealth_modifier))
            else:
                penalties.append(("Noisy technique ratio", abs(stealth_modifier)))
                stealth_deduction += abs(stealth_modifier)
                
        # Calculate final score
        final_score = max(1.0, min(10.0, base_score - stealth_deduction + stealth_bonus))
        
        # Create detailed metrics
        metrics = {
            "base_score": base_score,
            "final_score": round(final_score, 2),
            "penalties": penalties,
            "bonuses": bonuses,
            "action_counts": action_counts,
            "noisy_ratio": noisy_actions / total_actions if total_actions > 0 else 0,
            "redundancy_ratio": redundant_actions / total_actions if total_actions > 0 else 0,
            "effectiveness": effective_actions / total_actions if total_actions > 0 else 0
        }
        
        return final_score, metrics

# === CLI Testing ===
if __name__ == "__main__":
    console.print("[bold cyan]🚀 Testing EnvironmentContextDetector[/bold cyan]")
    
    detector = EnvironmentContextDetector()
    
    # Test environment detection
    test_ips = ["10.10.10.10", "192.168.1.1", "example.com"]
    for ip in test_ips:
        env_type = detector.detect_environment_type(ip)
        console.print(f"IP {ip}: Detected as {env_type}")
        
    # Test action permissions
    test_actions = [
        "nmap -sS -sV 10.10.10.10",
        "gobuster dir -u http://10.10.10.10",
        "msfconsole",
        "zip -r /tmp/data /etc/passwd"
    ]
    
    for action in test_actions:
        allowed_sim = detector.is_action_allowed(action, "simulated")
        allowed_live = detector.is_action_allowed(action, "live")
        console.print(f"Action {action}: Simulated={allowed_sim}, Live={allowed_live}")
        
    # Test curriculum advancement
    detector.advance_curriculum()
    console.print(f"Curriculum stage: {detector.curriculum_stage}")
    
    # Test domain randomization
    random_params = detector.randomize_domain()
    console.print(f"Randomized parameters: {random_params}")
    
    # Test stealth scoring
    test_history = [
        {"command": "nmap -sS 10.10.10.10", "success": True},
        {"command": "nmap -A 10.10.10.10", "success": True},
        {"command": "gobuster dir -u http://10.10.10.10", "success": True},
        {"command": "hydra -l admin -P wordlist.txt 10.10.10.10 ssh", "success": False, "redundant": True}
    ]
    
    stealth_score, metrics = detector.calculate_stealth_score(test_history)
    console.print(f"Stealth score: {stealth_score}/10")
    console.print(f"Stealth metrics: {metrics}")
