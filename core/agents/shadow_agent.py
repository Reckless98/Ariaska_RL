# core/agents/shadow_agent.py — ARIASKA ShadowAgent v1.0
# 🕵️ Stealth Monitor | Action Override System | Alert Prevention Specialist

import os
import logging
import time
import random
from typing import Dict, Any, List, Optional, Tuple, Union
from rich.console import Console
from rich.table import Table

from core.interfaces.agent_interface import AgentInterface
from core.interfaces.memory_sync_interface import MemorySyncInterface
from core.utils.llm_orchestrator import LLMOrchestrator
from core.utils.replay_buffer import ReplayBuffer
from core.gpt_manager import GPTManager
from core.multiagent.memory_router import MemoryRouter

console = Console()
logger = logging.getLogger("ariaska.shadow_agent")

class ShadowAgent(AgentInterface, MemorySyncInterface):
    """
    ShadowAgent: Stealth monitoring agent that intervenes when actions would trigger alerts.
    
    - Monitors stealth metrics and alert thresholds
    - Suggests quieter alternatives to noisy actions
    - Can override RedAgent actions if they're too risky
    - Uses LLM to generate stealthy alternatives
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
        agent_id: str = "ShadowAgent",
        role: str = "StealthMonitor",
        agent_manager=None,
        memory_router=None,
        verbosity: str = "standard"
    ):
        self._agent_id = agent_id
        self._role = role
        self.agent_manager = agent_manager
        self.memory_router = memory_router or MemoryRouter()
        self.verbosity = verbosity
        
        # Initialize LLM orchestrator for stealth recommendations
        self.llm_router = LLMOrchestrator(cache_dir="cache/shadow_agent_responses")
        
        # Initialize GPT manager for fallback
        self.gpt_manager = GPTManager.get_instance()
        
        # Stealth monitoring parameters
        self.alert_threshold = 50.0  # Alert level threshold for intervention
        self.stealth_mode_threshold = 30.0  # Threshold to activate stealth mode
        self.intervention_history = []
        self.alert_scores = []
        self.current_alert_score = 0.0
        self.stealth_mode = False
        
        # Command history for analysis
        self.command_history = []
        self.override_history = []
        
        # Track scan rates for timing analysis
        self.scan_timestamps = []
        self.scan_threshold = 5  # Max scans in short time period
        self.scan_window = 60  # Time window in seconds
        
        # Replay buffer for stealth decisions
        self.replay_buffer = ReplayBuffer(
            capacity=500,
            use_sqlite=True,
            db_path="core/memories/shadow_memory/replay_buffer.sqlite3"
        )
        
        # Add environment and stats monitor for main.py compatibility
        from core.environment.cyber_environment import CyberEnvironment
        from core.utils.stats_monitor import StatsMonitor
        self.env = CyberEnvironment(agent_manager=agent_manager, defer_reset=True) if agent_manager else None
        self.stats_monitor = StatsMonitor()
        
        # Link to other agents (will be set in _init_multiagent_links)
        self.red_agent = None
        self.scout_agent = None
        self.orion_agent = None
        self.blue_agent = None
        
        # Stealth technique recommendations
        self.stealth_techniques = {
            "scan": [
                {"technique": "slower_scan", "description": "Use -T1 or -T2 timing for slower, stealthier scans"},
                {"technique": "reduced_ports", "description": "Scan fewer ports at once"},
                {"technique": "syn_scan", "description": "Use SYN scan (-sS) instead of full TCP connect"},
                {"technique": "decoy", "description": "Use decoy addresses to confuse detection (-D)"},
                {"technique": "fragmentation", "description": "Fragment packets (-f) to bypass simple packet filters"}
            ],
            "enumeration": [
                {"technique": "passive_enum", "description": "Use passive enumeration techniques when possible"},
                {"technique": "single_service", "description": "Enumerate one service at a time rather than all"},
                {"technique": "throttle", "description": "Add delays between requests to appear more human-like"}
            ],
            "exploit": [
                {"technique": "manual_exploit", "description": "Use manual exploitation instead of automated tools when possible"},
                {"technique": "fewer_attempts", "description": "Reduce number of exploit attempts"},
                {"technique": "targeted_payload", "description": "Use specifically crafted payloads instead of generic ones"}
            ]
        }
        
        logger.debug(f"{self.agent_id} initialized")
        
    def _init_multiagent_links(self):
        """Initialize links to other agents in the system"""
        if self.agent_manager:
            self.red_agent = self.agent_manager.get_agent("RedAgent")
            self.scout_agent = self.agent_manager.get_agent("ScoutAgent")
            self.orion_agent = self.agent_manager.get_agent("OrionAgent")
            self.blue_agent = self.agent_manager.get_agent("BlueAgent")
    
    def calculate_alert_score(self, state: Dict[str, Any], command: str) -> float:
        """
        Calculate an alert score based on current state and proposed command.
        
        Args:
            state: Current environment state
            command: Proposed command
            
        Returns:
            Alert score (0-100)
        """
        # Base score from environment if available
        base_score = state.get("blue_team_alert", 0) + state.get("detection_risk", 0)
        
        # Add score based on command characteristics
        command_score = 0
        
        # Check for noisy scan patterns
        if "nmap" in command:
            if "-T4" in command or "-T5" in command:
                command_score += 15  # Very fast scans are noisy
            elif "-A" in command:
                command_score += 10  # Aggressive scan
            elif "-sT" in command:
                command_score += 5   # Full TCP connect scan
            elif "-sS" in command:
                command_score += 2   # SYN scan (quieter)
                
            # Check for full port scans
            if "-p-" in command or "-p 1-65535" in command:
                command_score += 10  # Full port scan is noisy
        
        # Check for brute force attempts
        if any(term in command for term in ["hydra", "medusa", "brutespray", "gobuster", "ffuf", "wfuzz"]):
            command_score += 15
        
        # Check for Metasploit usage (very noisy)
        if "msfconsole" in command or "msf" in command:
            command_score += 20
        
        # Check scan rate and timing
        current_time = time.time()
        self.scan_timestamps = [t for t in self.scan_timestamps if current_time - t < self.scan_window]
        self.scan_timestamps.append(current_time)
        if len(self.scan_timestamps) > self.scan_threshold:
            command_score += 10  # Too many scans in short time period
        
        # Calculate total score
        total_score = base_score + command_score
        
        # Cap at 100
        return min(total_score, 100.0)
    
    def suggest_quieter_alternative(self, command: str) -> Optional[str]:
        """
        Suggest a quieter alternative to the given command if needed.
        
        Args:
            command: Original command
            
        Returns:
            Quieter alternative command or None if no change needed
        """
        # Only suggest alternatives if alert is above threshold or in stealth mode
        if self.current_alert_score <= self.stealth_mode_threshold and not self.stealth_mode:
            return None
        
        # Calculate noise score
        noise_score = self._calculate_command_noise(command)
        
        # If noise score is acceptable, don't change
        if noise_score < self.alert_threshold:
            return None
        
        # For nmap commands, suggest quieter alternatives
        if command.startswith("nmap"):
            quieter_command = command
            
            # Replace timing if too aggressive
            if "-T4" in command or "-T5" in command:
                quieter_command = quieter_command.replace("-T4", "-T2").replace("-T5", "-T2")
            elif "-T3" in command and "-T" in command:
                quieter_command = quieter_command.replace("-T3", "-T2")
            elif not any(f"-T{i}" in command for i in range(1, 6)):
                quieter_command += " -T2"  # Add slow timing if none specified
            
            # Add SYN scan if not specified
            if "-sS" not in quieter_command and "-s" not in quieter_command:
                quieter_command += " -sS"
            
            # Add fragmentation for additional stealth if high alert
            if "-f" not in quieter_command and self.current_alert_score > 70:
                quieter_command += " -f"
            
            # Add decoys for very high alert levels
            if "-D" not in quieter_command and self.current_alert_score > 85:
                quieter_command += " -D RND:3"  # Add 3 random decoys
                
            # If command was modified, return it
            if quieter_command != command:
                if self.verbosity not in ["quiet", "silent"]:
                    console.print(f"[yellow]🔍 ShadowAgent suggested quieter alternative: {quieter_command}[/yellow]")
                return quieter_command
        
        # For other command types, use LLM to suggest alternatives
        if hasattr(self, "llm_router") and self.current_alert_score > self.alert_threshold + 10:
            try:
                context = {
                    "original_command": command,
                    "alert_score": self.current_alert_score,
                    "stealth_required": True
                }
                
                prompt = f"""
                The command '{command}' might generate too much noise.
                Current alert score: {self.current_alert_score}/100
                
                Suggest a quieter alternative command that achieves the same goal but with lower detection risk.
                If you can't suggest a better alternative, respond with "ORIGINAL".
                """
                
                alternative = self.llm_router.route_task(
                    task_type="tactical",
                    prompt=prompt,
                    agent_id=self.agent_id
                )
                
                if alternative and isinstance(alternative, str) and alternative.strip().upper() != "ORIGINAL":
                    if self.verbosity not in ["quiet", "silent"]:
                        console.print(f"[yellow]🔍 ShadowAgent LLM suggested quieter alternative: {alternative}[/yellow]")
                    return alternative
            except Exception as e:
                if self.verbosity not in ["quiet", "silent"]:
                    console.print(f"[yellow]⚠️ ShadowAgent LLM error: {e}[/yellow]")
        
        return None
    
    def _determine_command_type(self, command: str) -> str:
        """
        Determine the type of command for stealth analysis.
        
        Args:
            command: Command to analyze
            
        Returns:
            Command type string
        """
        if any(scan_tool in command for scan_tool in ["nmap", "masscan", "ping", "traceroute"]):
            return "scan"
        elif any(enum_tool in command for enum_tool in ["gobuster", "enum4linux", "wpscan", "nikto"]):
            return "enumeration"
        elif any(exploit_tool in command for exploit_tool in ["sqlmap", "msfconsole", "exploit", "hydra", "medusa"]):
            return "exploit"
        elif any(priv_tool in command for priv_tool in ["sudo", "su", "linpeas", "winpeas"]):
            return "privesc"
        else:
            return "other"
    
    def evaluate_scan_plan(self, scan_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a scan plan from ScoutAgent for stealth issues.
        
        Args:
            scan_plan: Scan plan from ScoutAgent
            
        Returns:
            Evaluation results with stealth recommendations
        """
        # Extract scan parameters
        command = scan_plan.get("command", "")
        scan_type = scan_plan.get("scan_type", "tcp")
        ports = scan_plan.get("ports", [])
        
        if not command:
            return {"too_noisy": False, "reason": ""}
            
        # Calculate noise level based on command
        noise_score = self._calculate_command_noise(command)
        
        # Check if noise score is too high given current alert levels
        threshold = self.alert_threshold - 20.0  # Lower threshold for proactive protection
        if noise_score > threshold:
            result = {
                "too_noisy": True,
                "reason": f"Command noise score {noise_score:.1f} exceeds threshold {threshold:.1f}"
            }
            
            if self.verbosity not in ["quiet", "silent"]:
                console.print(f"[yellow]⚠️ ShadowAgent: {command} is too noisy (score: {noise_score:.1f})[/yellow]")
                
            return result
        
        return {"too_noisy": False, "reason": ""}
        
    def _calculate_command_noise(self, command: str) -> float:
        """
        Calculate a noise score for a command.
        
        Args:
            command: Command to evaluate
        
        Returns:
            Noise score (0-100)
        """
        base_score = 50.0  # Default moderate noise
        
        # Look for indicators of noisy scanning
        if "-T4" in command or "-T5" in command:
            base_score += 30.0  # Very fast scanning is noisy
        elif "-T3" in command:
            base_score += 15.0  # Normal speed scanning is moderately noisy
        elif "-T1" in command or "-T2" in command:
            base_score -= 20.0  # Slow scanning is quieter
        
        if "-A" in command:  # Aggressive scan
            base_score += 25.0
        
        if "-p-" in command:  # All ports scan
            base_score += 15.0
        
        if "-sS" in command:  # SYN scan (stealthier)
            base_score -= 10.0
        
        if "-sT" in command:  # Full TCP connect (noisier)
            base_score += 10.0
        
        if "-f" in command or "-ff" in command:  # Fragmented packets
            base_score -= 15.0
        
        if "-D" in command:  # Decoy scan
            base_score -= 20.0
        
        # Clamp between 0-100
        return max(0.0, min(100.0, base_score))
    
    def get_stealth_recommendation(self) -> Dict[str, Any]:
        """
        Get stealth mode and noise level recommendations based on current alerts.
        
        Returns:
            Dict containing stealth mode and recommended noise level
        """
        # Default recommendations
        recommendation = {
            "stealth_mode": self.stealth_mode,
            "recommended_noise_level": 5.0  # Default balanced noise level
        }
        
        # If alert score is above threshold, recommend stealth mode
        if self.current_alert_score > self.alert_threshold:
            recommendation["stealth_mode"] = True
            
            # Calculate recommended noise level based on alert score
            # Higher alert = lower noise level (more stealth)
            alert_ratio = self.current_alert_score / 100.0
            noise_level = max(1.0, 10.0 * (1.0 - alert_ratio))
            recommendation["recommended_noise_level"] = noise_level
            
            if self.verbosity not in ["quiet", "silent"]:
                console.print(f"[yellow]🔍 ShadowAgent recommends stealth mode with noise level {noise_level:.1f}[/yellow]")
        
        return recommendation

    def _get_technique_recommendations(self) -> List[Dict[str, str]]:
        """
        Get stealth technique recommendations.
        
        Returns:
            List of technique recommendations
        """
        # Determine current phase if available from ScoutAgent or RedAgent
        phase = "scan"  # Default
        if self.scout_agent and hasattr(self.scout_agent, "current_phase"):
            phase = self.scout_agent.current_phase
        elif self.red_agent and hasattr(self.red_agent, "_last_state") and hasattr(self.red_agent._last_state, "get"):
            phase = self.red_agent._last_state.get("phase", "scan")
            
        # Map phase to technique type
        technique_type = "scan"
        if phase in ["enumeration", "recon"]:
            technique_type = "enumeration"
        elif phase in ["exploit", "privesc", "exfiltrate"]:
            technique_type = "exploit"
            
        # Get techniques for this type
        techniques = self.stealth_techniques.get(technique_type, [])
        
        # Return 2-3 random techniques
        return random.sample(techniques, min(3, len(techniques)))
    
    def get_smart_command(self, state: Dict[str, Any], phase: str) -> tuple[str, str]:
        """Generate stealth command for main.py compatibility."""
        try:
            # Check detection risk from state
            detection_risk = state.get("detection_risk", 0)
            blue_team_alert = state.get("blue_team_alert", 0)
            
            if detection_risk > 50 or blue_team_alert > 30:
                # High risk - use very quiet commands
                stealth_commands = [
                    "nmap -sT -T1 10.10.10.10",  # Slow TCP SYN scan
                    "nc -nz 10.10.10.10 22",      # Simple port check
                    "dig 10.10.10.10",            # DNS lookup
                ]
                command = random.choice(stealth_commands)
                reason = f"High detection risk ({detection_risk}%) - using stealth approach"
            else:
                # Lower risk - normal commands
                normal_commands = [
                    "nmap -sT 10.10.10.10 -p 22,80,443",
                    "curl -I http://10.10.10.10",
                    "ping -c 1 10.10.10.10"
                ]
                command = random.choice(normal_commands)
                reason = f"Moderate stealth (detection: {detection_risk}%)"
            
            return command, reason
        except Exception as e:
            # Fallback stealth command
            return "ping -c 1 10.10.10.10", f"Stealth fallback due to error: {e}"
    
    def act(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Take stealth action based on current state.
        
        Args:
            state: Current environment state
            
        Returns:
            Dict with action, success, reward, and info
        """
        # Shadow agent focuses on stealth and evasion
        stealth_level = state.get("stealth_level", 50)
        detection_risk = state.get("detection_risk", 30)
        active_scans = state.get("active_scans", 0)
        
        if detection_risk > 60:
            # High detection risk - go dark
            action = "echo 'Going dark - waiting for better opportunity'"
            success = True
            reward = 10  # Reward for avoiding detection
            info = {"action_type": "evasion", "risk_assessment": "high"}
        elif stealth_level > 70:
            # Good stealth - perform reconnaissance
            action = "netstat -tulpn | grep LISTEN"
            success = True  
            reward = 20  # Reward for stealth recon
            info = {"action_type": "stealth_recon", "stealth_level": stealth_level}
        elif active_scans > 3:
            # Too much activity - blend in
            action = "ps aux | head -20"
            success = True
            reward = 15  # Reward for blending
            info = {"action_type": "blending", "active_scans": active_scans}
        else:
            # Normal stealth operation
            action = "lsof -i | grep ESTABLISHED"
            success = True
            reward = 25  # Standard stealth reward
            info = {"action_type": "stealth_operation", "stealth_level": stealth_level}
        
        return {
            "action": action,
            "success": success,
            "reward": reward,
            "info": info
        }

    def simulate_step(self, episode: int = 1, step: int = 1, shared_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Simulate a step for this agent in the training loop.
        
        Args:
            episode: Current episode number
            step: Current step number
            shared_context: Shared context from agent manager
            
        Returns:
            Dict with step results
        """
        # Get current state
        state = {}
        if self.red_agent and hasattr(self.red_agent, "env"):
            state = self.red_agent.env.get_global_state()
        elif shared_context:
            state = shared_context
            
        # Calculate current alert score based on state and recent RedAgent command
        last_command = ""
        alert_score = 0.0
        if self.red_agent and hasattr(self.red_agent, "command_history") and self.red_agent.command_history:
            last_command = self.red_agent.command_history[-1]
            alert_score = self.calculate_alert_score(state, last_command)
            self.current_alert_score = alert_score
            self.alert_scores.append(alert_score)
            
            # Check if we need to intervene
            intervention_needed = alert_score > self.alert_threshold
            alternative = None
            
            if intervention_needed:
                alternative = self.suggest_quieter_alternative(last_command)
                if alternative:
                    self.override_history.append({
                        "original": last_command,
                        "alternative": alternative,
                        "alert_score": alert_score,
                        "step": step,
                        "episode": episode
                    })
        
        # Get stealth recommendations
        recommendations = self.get_stealth_recommendation()
        
        # Log to memory router    
        if self.memory_router and hasattr(self.memory_router, 'log_shadow_alert'):
            try:
                self.memory_router.log_shadow_alert(  # type: ignore
                    alert_score=self.current_alert_score,
                    stealth_mode=self.stealth_mode,
                    recommendations=recommendations,
                    step=step,
                    episode=episode
                )
            except AttributeError:
                # Method doesn't exist, silently continue
                pass
            
        # Display information based on verbosity
        if self.verbosity in ("standard", "verbose", "debug"):
            mode_str = "[red]STEALTH[/red]" if self.stealth_mode else "[green]NORMAL[/green]"
            console.print(f"[blue]🕵️ ShadowAgent: Alert score {self.current_alert_score:.1f} | Mode {mode_str}[/blue]")
        
        if self.verbosity in ("verbose", "debug"):
            table = Table(title=f"🕵️ ShadowAgent Status (Step {step})")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            table.add_row("Alert Score", f"{self.current_alert_score:.1f}")
            table.add_row("Stealth Mode", str(self.stealth_mode))
            table.add_row("Blue Alert", f"{state.get('blue_team_alert', 0):.1f}")
            table.add_row("Detection Risk", f"{state.get('detection_risk', 0):.1f}")
            if last_command:
                table.add_row("Last Command", last_command[:40])
            alternative = self.override_history[-1]["alternative"] if self.override_history else None
            if alternative:
                table.add_row("Alternative", alternative[:40])
            console.print(table)
            
        # Return step information
        return {
            "agent_id": self.agent_id,
            "alert_score": self.current_alert_score,
            "stealth_mode": self.stealth_mode,
            "blue_team_alert": state.get("blue_team_alert", 0),
            "detection_risk": state.get("detection_risk", 0),
            "last_command": last_command,
            "alternative": self.override_history[-1]["alternative"] if self.override_history else None,
            "step": step,
            "episode": episode
        }
    
    def sync_memory(self) -> bool:
        """
        Synchronize memory with the global memory router.
        Implementation of MemorySyncInterface.
        """
        try:
            if self.memory_router and hasattr(self.memory_router, 'update_shadow_status'):
                # Share alert score and stealth recommendations with global memory
                self.memory_router.update_shadow_status(  # type: ignore
                    self.agent_id,
                    self.current_alert_score,
                    self.stealth_mode,
                    self.get_stealth_recommendation()
                )
                return True
            return False
        except Exception as e:
            console.print(f"[red]❌ ShadowAgent memory sync error: {e}[/red]")
            return False
        
    def optimize_all_agents_memory(self, agents: List[Any]) -> None:
        """
        Optimize memory usage across all agents by analyzing redundancy patterns.
        
        Args:
            agents: List of all agents in the system
        """
        # Import here to avoid circular imports
        from core.logic.redundancy_detector import detect_redundancy_batch
        
        for agent in agents:
            if hasattr(agent, "replay_buffer") and hasattr(agent.replay_buffer, "prune_redundancy"):
                try:
                    agent.replay_buffer.prune_redundancy(lambda cmds: detect_redundancy_batch(cmds))
                    console.print(f"[blue]♻️ ShadowAgent optimized {agent.agent_id}'s memory[/blue]")
                except Exception as e:
                    console.print(f"[yellow]⚠ Error optimizing {agent.agent_id}'s memory: {e}[/yellow]")
    
    def get_base_commands(self):
        """Return base stealth commands for CLI completion."""
        return [
            "nmap -T1",
            "nmap -T2",
            "nmap -sT",
            "nmap -f",
            "nmap -D",
            "nmap --spoof-mac",
            "proxychains",
            "macchanger",
            "tor"
        ]
    
    def reset(self):
        """Reset agent state for a new episode."""
        self.intervention_history = []
        self.command_history = []
        self.override_history = []
        self.alert_scores = []
        self.current_alert_score = 0.0
        self.stealth_mode = False
        self.scan_timestamps = []
    
    def provide_reasoning(self, context_type: str, context_data: dict) -> str:
        """Provide reasoning for a given context - only available in OrionAgent."""
        return f"ShadowAgent does not provide strategic reasoning. Use OrionAgent for strategic insights."

# Main execution for testing
if __name__ == "__main__":
    agent = ShadowAgent(verbosity="verbose")
    
    # Test alert score calculation
    state = {
        "blue_team_alert": 15.0,
        "detection_risk": 10.0
    }
    commands = [
        "nmap -sT -T2 10.10.10.10",
        "nmap -sT -T4 -A -p- 10.10.10.10",
        "gobuster dir -u http://10.10.10.10 -w /usr/share/dirb/wordlists/common.txt",
        "hydra -l admin -P /usr/share/nmap/nselib/data/passwords.lst ssh://10.10.10.10",
        "msfconsole -q -x 'use exploit/multi/handler'"
    ]
    console.print("[bold cyan]Alert score testing:[/bold cyan]")
    for command in commands:
        score = agent.calculate_alert_score(state, command)
        console.print(f"[bold]{command}[/bold]: {score:.1f}/100")
        
        # Test stealth alternatives
        if score > 30:
            alt = agent.suggest_quieter_alternative(command)
            if alt:
                console.print(f"  [green]Alternative:[/green] {alt}")
    
    # Test scan plan evaluation
    scan_plan = {
        "command": "nmap -sT -T4 -p- 10.10.10.10",
        "scan_type": "tcp",
        "ports": list(range(1, 1001))
    }
    eval_result = agent.evaluate_scan_plan(scan_plan)
    console.print("\n[bold cyan]Scan plan evaluation:[/bold cyan]")
    console.print(eval_result)
    
    # Test simulation step
    agent.red_agent = type('obj', (object,), {
        'command_history': ["nmap -sT -T4 -A -p- 10.10.10.10"],
        'env': type('obj', (object,), {
            'get_global_state': lambda: state
        })
    })
    result = agent.simulate_step(episode=1, step=1)
    console.print("\n[bold magenta]Simulation result:[/bold magenta]")
    console.print(result)
    
    # Test stealth recommendations
    recommendations = agent.get_stealth_recommendation()
    console.print("\n[bold cyan]Stealth recommendations:[/bold cyan]")
    console.print(recommendations)