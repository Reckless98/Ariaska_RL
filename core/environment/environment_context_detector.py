# core/environment/environment_context_detector.py — ARIASKA Environment Context Detector v1.0
# 🌐 Environment Mode Detection | 🔐 Safety Controls | 🏫 Curriculum Management

import os
import json
from typing import Dict, Any, Optional
from rich.console import Console

console = Console()

class EnvironmentContextDetector:
    """
    Detects and adapts to environment context (simulated vs live mode).
    - Auto-adjusts agent strategies for safety based on environment
    - Enforces curriculum scheduling and domain randomization
    - Manages safety boundaries between simulated and live environments
    - Provides environmental awareness to agents for context-aware decision making
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the environment context detector.
        
        Args:
            config_path (str, optional): Path to configuration file
        """
        self.config_path = config_path or os.path.join("config", "environment.json")
        self.config = self._load_config()
        self.current_mode = self.config.get("default_mode", "simulated")
        self.safety_level = self.config.get("safety_level", "strict")
        self.curriculum_stage = self.config.get("curriculum_stage", 1)
        self.domain_randomization = self.config.get("domain_randomization", True)
        
        # Load target configurations
        self.targets = self.config.get("targets", {})
        
        # Track environment parameters for dynamic adjustment
        self.environment_parameters = {
            "ports": self.config.get("ports", [22, 80, 443]),
            "services": self.config.get("services", ["ssh", "http", "https"]),
            "os_types": self.config.get("os_types", ["linux"]),
            "difficulty": self.curriculum_stage
        }
        
        console.print(f"[green]✓ Environment Context Detector initialized: Mode={self.current_mode}, Curriculum Stage={self.curriculum_stage}[/green]")
        
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Returns:
            dict: Configuration parameters
        """
        default_config = {
            "default_mode": "simulated",
            "safety_level": "strict",
            "curriculum_stage": 1,
            "domain_randomization": True,
            "ports": [22, 80, 443],
            "services": ["ssh", "http", "https"],
            "os_types": ["linux"],
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
            target_ip (str, optional): Target IP address
            
        Returns:
            str: Environment type ('simulated' or 'live')
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
            action (str): Action to check
            environment_type (str, optional): Environment type to check against
            
        Returns:
            bool: True if action is allowed, False otherwise
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
            action (str): Action to categorize
            
        Returns:
            str: Action type
        """
        action = action.lower()
        if any(cmd in action for cmd in ["nmap", "masscan", "ping"]):
            return "scan"
        elif any(cmd in action for cmd in ["gobuster", "enum4linux", "wpscan"]):
            return "enum"
        elif any(cmd in action for cmd in ["exploit", "msfconsole", "sqlmap"]):
            return "exploit"
        elif any(cmd in action for cmd in ["linpeas", "winpeas", "sudo"]):
            return "privesc"
        elif any(cmd in action for cmd in ["zip", "tar", "scp", "exfil"]):
            return "exfil"
        return "unknown"
        
    def adjust_agent_parameters(self, agent, environment_type: str = None) -> None:
        """
        Adjust agent parameters based on environment context.
        
        Args:
            agent: Agent to adjust
            environment_type (str, optional): Environment type to adjust for
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
                
        console.print(f"[blue]🔧 Adjusted {agent.agent_id} parameters for {environment_type} environment[/blue]")
        
    def get_environment_context(self, target_ip: str = None) -> Dict[str, Any]:
        """
        Get complete environment context data.
        
        Args:
            target_ip (str, optional): Target IP address
            
        Returns:
            dict: Environment context data
        """
        environment_type = self.detect_environment_type(target_ip)
        
        context = {
            "mode": environment_type,
            "safety_level": self.safety_level,
            "curriculum_stage": self.curriculum_stage,
            "parameters": self.environment_parameters,
            "target": self.targets.get(environment_type, {}),
            "is_simulated": environment_type == "simulated"
        }
        
        return context
        
    def advance_curriculum(self) -> None:
        """
        Advance to the next curriculum stage.
        Increases difficulty and complexity of the environment.
        """
        self.curriculum_stage += 1
        self.environment_parameters["difficulty"] = self.curriculum_stage
        
        # Add more complex parameters as curriculum advances
        if self.curriculum_stage >= 2:
            self.environment_parameters["ports"].extend([8080, 3306])
            self.environment_parameters["services"].extend(["mysql", "proxy"])
            
        if self.curriculum_stage >= 3:
            self.environment_parameters["os_types"].extend(["windows"])
            
        console.print(f"[green]📚 Advanced to curriculum stage {self.curriculum_stage}[/green]")
        
        # Save updated configuration
        self.config["curriculum_stage"] = self.curriculum_stage
        self._save_config()
        
    def randomize_domain(self) -> Dict[str, Any]:
        """
        Randomize domain parameters for training variety.
        
        Returns:
            dict: Randomized environment parameters
        """
        import random
        
        if not self.domain_randomization:
            return self.environment_parameters
            
        # Create a copy to avoid modifying the original
        params = self.environment_parameters.copy()
        
        # Randomize available ports
        all_ports = [21, 22, 23, 25, 80, 443, 445, 3306, 8080, 8443]
        num_ports = random.randint(2, 5)
        params["ports"] = sorted(random.sample(all_ports, num_ports))
        
        # Randomize available services
        all_services = ["ftp", "ssh", "telnet", "smtp", "http", "https", "smb", "mysql", "proxy"]
        num_services = random.randint(2, 5)
        params["services"] = random.sample(all_services, num_services)
        
        # Randomize OS type
        params["os_types"] = random.choice([["linux"], ["windows"], ["linux", "windows"]])
        
        console.print(f"[blue]🎲 Randomized domain parameters: {len(params['ports'])} ports, {len(params['services'])} services[/blue]")
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
            mode (str): Environment mode ('simulated' or 'live')
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
