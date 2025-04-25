# core/agents/scout_agent.py — ARIASKA ScoutAgent v11.1 APEX NAVIGATOR
# 🧭 Phase Navigator | 🔄 Environment Observer | 🧠 GPT-4o-mini Powered | 📚 Adaptive Phase Library

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import os
import json
import time
import random
import hashlib
from typing import Dict, List, Any, Optional
from core.interfaces.agent_interface import AgentInterface
from core.interfaces.memory_sync_interface import MemorySyncInterface
from core.gpt_manager import GPTManager
from core.logic.rule_engine import phase_distribution_for_visualization
from core.utils.local_llm_manager import LocalLLMManager

console = Console()

class ScoutAgent(AgentInterface, MemorySyncInterface):
    """
    ScoutAgent: Phase navigator and environment observer.
    - Advises phase for RedAgent and others.
    - Uses GPTManager for phase detection and reasoning.
    - Unified memory schema: actions, rewards, scenarios.
    """
    
    def __init__(
        self,
        agent_id="ScoutAgent",
        role="PhaseNavigator",
        agent_manager=None,
        memory_router=None,
        memory_manager=None,
        verbosity="standard",
    ):
        self.agent_id = agent_id
        self.role = role
        self.agent_manager = agent_manager
        self.memory_router = memory_router
        self.memory_manager = memory_manager
        self.verbosity = verbosity
        
        # Unified memory schema
        self.memory = {
            "actions": [],
            "rewards": {},
            "scenarios": []
        }
        self.current_phase = "recon"
        self.phase_history = []
        self.phase_cache = {}  # Cache for phase advice - {state_hash: (phase, timestamp)}
        self.phase_cache_ttl = 300  # TTL in seconds for phase cache entries
        self.phase_switch_cooldown = 0  # Steps to wait before switching phases again
        
        # Environment metrics
        self.env_complexity = 1.0
        self.last_state = {}
        
        # Phase duration tracking
        self.phase_durations = {p: 0 for p in ["recon", "enumeration", "exploit", "privesc", "exfiltrate"]}
        self.total_steps = 0
        
        # Phase advice confidence
        self.confidence_scores = {}
        self.last_advice_time = 0
        self.gpt_calls_this_episode = 0
        self.max_gpt_calls_per_episode = 10  # Limit GPT calls per episode
        
        # Use GPTManager for all LLM operations
        self.gpt_manager = GPTManager()
        self.local_llm = LocalLLMManager(model_name="wahidmounir/SenecaLLM_x_Qwen2.5-7B-CyberSecurity-Q8_0-GGUF")
        
        # Logging
        self.phase_log_path = os.path.join("logs", f"{self.agent_id}_phase_log.jsonl")
        os.makedirs(os.path.dirname(self.phase_log_path), exist_ok=True)
        
        console.print(f"[cyan]🧭 {self.agent_id} initialized — Phase Navigator Active[/cyan]")

    def _init_multiagent_links(self):
        """Initialize links to other agents for phase coordination."""
        pass  # Scout can operate independently for now

    def advise_phase(self, state: dict, all_agents=None) -> str:
        """
        Use dual-LLM critique for phase navigation if available.
        """
        # Track call in history
        self.total_steps += 1
        
        # First, check if we're on phase cooldown (prevent rapid switching)
        if self.phase_switch_cooldown > 0:
            self.phase_switch_cooldown -= 1
            return self.current_phase
        
        # Check if we have a cached result for this state
        state_hash = self._hash_state(state)
        if state_hash in self.phase_cache:
            cached_phase, timestamp = self.phase_cache[state_hash]
            # Check if cache is still valid
            if time.time() - timestamp < self.phase_cache_ttl:
                return cached_phase
        
        # Check if we've exceeded our GPT call budget for this episode
        if self.gpt_calls_this_episode >= self.max_gpt_calls_per_episode:
            # If over budget, use heuristic phase selection
            phase = self._heuristic_phase_selection(state)
            if self.verbosity in ("standard", "verbose"):
                console.print(f"[yellow]⚠️ {self.agent_id}: Using heuristic phase selection due to GPT call limit ({phase})[/yellow]")
            return phase
        
        # Alternative rule-based phase detection first
        rule_based_phase = self._rule_based_phase_detection(state)
        if rule_based_phase:
            self.phase_cache[state_hash] = (rule_based_phase, time.time())
            return rule_based_phase
        
        # Last resort: Use GPT for phase detection (with rate limiting)
        current_time = time.time()
        if current_time - self.last_advice_time < 10:  # At most one call per 10 seconds
            # Use the most recent phase if too frequent
            if self.phase_history:
                return self.phase_history[-1]
            return self.current_phase
            
        # Increment GPT call counter
        self.gpt_calls_this_episode += 1
        self.last_advice_time = current_time
        
        # Use GPTManager to detect phase (with token tracking)
        phase = self._gpt_phase_detection(state)
        
        # Store in cache for future use
        self.phase_cache[state_hash] = (phase, time.time())
        
        # Update phase history
        self.phase_history.append(phase)
        if len(self.phase_history) > 20:  # Keep history manageable
            self.phase_history = self.phase_history[-20:]
        
        # Set cooldown to prevent rapid phase changes
        self.phase_switch_cooldown = 2
        
        # Update current phase
        self.current_phase = phase
        
        # Log phase selection
        self._log_phase_selection(state, phase)
        
        # Optionally: integrate dual-LLM feedback for phase advice
        if hasattr(self, "gpt_manager"):
            task_desc = (
                f"State: {state}. Determine optimal next phase (recon, enumeration, exploit, privesc, exfiltrate)."
            )
            dual_feedback = self.gpt_manager.dual_llm_feedback(task_desc, agent_id=self.agent_id)
            # Optionally log or use dual_feedback for phase selection

        return phase
        
    def _hash_state(self, state: Dict[str, Any]) -> str:
        """
        Generate a hash for the relevant parts of the environment state.
        This allows efficient caching of phase advice.
        """
        # Create a simplified state representation for consistent hashing
        simple_state = {
            "open_ports": sorted(state.get("open_ports", [])),
            "privilege_level": state.get("privilege_level", "none"),
            "services": sorted(state.get("services", [])),
            "blue_team_alert": round(state.get("blue_team_alert", 0)),
            "detection_risk": round(state.get("detection_risk", 0) * 10) / 10,
            "credentials_found": state.get("credentials_found", False),
            "data_exfiltrated": state.get("data_exfiltrated", False)
        }
        
        # Convert to string and hash
        state_str = json.dumps(simple_state, sort_keys=True)
        return hashlib.md5(state_str.encode()).hexdigest()

    def _rule_based_phase_detection(self, state: Dict[str, Any]) -> Optional[str]:
        """
        Use rule-based heuristics to determine the phase without GPT.
        Returns a phase string or None if uncertain.
        """
        # Check for phase indicators based on environment state
        privilege = state.get("privilege_level", "none")
        
        # Complete exfiltration conditions
        if state.get("data_exfiltrated", False):
            return "exfiltrate"
            
        # Privilege escalation conditions
        if privilege in ["root", "administrator"] and not state.get("data_exfiltrated", False):
            return "privesc"
            
        # Exploit conditions: low privilege obtained but not root yet
        if privilege in ["user", "low"] and not state.get("credentials_found", False):
            return "exploit"
            
        # Enumeration conditions: ports discovered but no access yet
        if len(state.get("open_ports", [])) > 0 and privilege == "none":
            return "enumeration"
            
        # Recon conditions: early stage, few ports known
        if len(state.get("open_ports", [])) <= 2:
            return "recon"
            
        # If no clear indicator, return None to use GPT
        return None

    def _heuristic_phase_selection(self, state: Dict[str, Any]) -> str:
        """
        Select the next phase based on heuristics when GPT is unavailable.
        This is a fallback method that doesn't use GPT.
        """
        # Simple progression logic based on discovered assets
        privilege = state.get("privilege_level", "none")
        blue_alert = state.get("blue_team_alert", 0)
        
        if state.get("data_exfiltrated", False):
            # Data already exfiltrated, restart cycle
            return "recon"
            
        if privilege in ["root", "administrator"]:
            # We have highest privileges, move to exfiltration
            return "exfiltrate"
            
        if privilege in ["user", "low"]:
            # We have basic access, move to privilege escalation
            return "privesc"
            
        if len(state.get("open_ports", [])) > 3 or len(state.get("services", [])) > 2:
            # We've found several ports/services, time to exploit
            return "exploit"
            
        if len(state.get("open_ports", [])) > 0:
            # We've found some ports, move to enumeration
            return "enumeration"
            
        # Default to recon if nothing else matches
        return "recon"

    def _gpt_phase_detection(self, state: dict) -> str:
        """
        Use both SenecaLLM and GPTManager for phase navigation.
        """
        simplified_state = {
            "ports": state.get("open_ports", []),
            "privilege": state.get("privilege_level", "none"),
            "services": state.get("services", []),
            "alert": state.get("blue_team_alert", 0),
            "risk": state.get("detection_risk", 0),
            "found_creds": state.get("credentials_found", False),
            "data_taken": state.get("data_exfiltrated", False)
        }
        task_desc = (
            f"State: {simplified_state}. "
            "Determine next optimal phase transition (recon, enumeration, exploit, privesc, exfiltrate)."
        )
        try:
            seneca_phase = self.local_llm.query(task_desc)
            review_prompt = (
                f"As a cyber navigation strategist, review the AI's suggested phase:\n\n"
                f"Task: {task_desc}\n"
                f"Suggestion: {seneca_phase}\n\n"
                f"Do you approve this phase? If not, refine it. Respond ONLY with the final phase."
            )
            phase = self.gpt_manager.gpt_request(
                review_prompt, task_type="navigation", model="gpt-4o-mini"
            )
            phase = str(phase).strip().lower()
            if phase not in ["recon", "enumeration", "exploit", "privesc", "exfiltrate"]:
                phase = "recon"
            return phase
        except Exception as e:
            console.print(f"[red]❌ _gpt_phase_detection error: {e}[/red]")
            return "recon"

    def log_phase_advice(self, state, phase):
        """
        Log phase advice to memory and file.
        """
        entry = {
            "state": state,
            "phase": phase,
            "timestamp": time.time()
        }
        self.memory["actions"].append(entry)
        with open(self.phase_log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def sync_memory(self):
        """
        Sync memory with MemoryRouter for global insights.
        """
        if self.memory_router:
            self.memory_router.save_memory(self.agent_id, self.memory)

    def simulate_step(self, episode=1, step=1, shared_context=None):
        """
        Simulate a step for the ScoutAgent.
        This agent focuses on environment analysis and phase advice rather than direct actions.
        """
        state = None
        
        try:
            # Get global state from environment or shared context
            if (shared_context and "environment_state" in shared_context):
                state = shared_context["environment_state"]
            elif self.agent_manager:
                # Try to get state from RedAgent's environment
                red_agent = self.agent_manager.get_agent("RedAgent")
                if (red_agent and hasattr(red_agent, "env")):
                    state = red_agent.env.get_global_state()
            
            if not state:
                # Fallback state if nothing available
                state = {
                    "phase": "recon",
                    "open_ports": [],
                    "privilege_level": "none",
                    "blue_team_alert": 0,
                    "detection_risk": 0
                }
                
            # Track environment changes for trend analysis
            if self.last_state:
                self._analyze_state_change(self.last_state, state)
                
            self.last_state = state.copy()
                
            # Determine the optimal phase
            phase = self.advise_phase(state, self.agent_manager.all_agents() if self.agent_manager else None)
            
            # Track duration in each phase
            if phase in self.phase_durations:
                self.phase_durations[phase] += 1
                
            # Broadcast phase to shared context
            if (shared_context is not None and self.agent_manager and hasattr(self.agent_manager, "broadcast")):
                self.agent_manager.broadcast(f"{self.agent_id}_phase", phase, sender=self.agent_id)
                
            # Calculate environment complexity score (useful for curriculum)
            self._update_complexity_score(state)
            
            # Provide advice based on phase
            advice = self._generate_phase_advice(phase, state)
            
            # Log step information
            if self.verbosity in ("standard", "verbose"):
                console.print(f"[cyan]🧭 {self.agent_id}:[/cyan] Advised phase: [bold]{phase}[/bold]")
                if self.verbosity == "verbose":
                    console.print(f"[cyan]🧭 Advice:[/cyan] {advice}")
                    
            # Return step results
            return {
                "command": "phase_advice",
                "phase": phase,
                "reward": 0,  # Scout doesn't receive direct rewards
                "output": advice,
                "gpt_calls": self.gpt_calls_this_episode,
                "reasoning": f"Environment analysis indicates {phase} phase is optimal",
                "step": step,
                "episode": episode,
                "agent_id": self.agent_id,
            }
            
        except Exception as e:
            console.print(f"[red]❌ Error in ScoutAgent simulate_step: {e}[/red]")
            import traceback
            console.print(traceback.format_exc())
            
            return {
                "command": "error",
                "phase": "unknown",
                "reward": 0,
                "output": f"Error: {e}",
                "gpt_calls": self.gpt_calls_this_episode,
                "reasoning": "Error occurred",
                "step": step,
                "episode": episode,
                "agent_id": self.agent_id
            }

    def _analyze_state_change(self, old_state: Dict[str, Any], new_state: Dict[str, Any]):
        """Analyze changes between states to detect trends."""
        # Track changes in key metrics
        if "blue_team_alert" in old_state and "blue_team_alert" in new_state:
            alert_change = new_state["blue_team_alert"] - old_state["blue_team_alert"]
            if alert_change > 3 and self.verbosity != "quiet":
                console.print(f"[yellow]⚠️ {self.agent_id}: Blue team alert increasing rapidly (+{alert_change:.1f})[/yellow]")
                
        # Track newly discovered ports
        old_ports = set(old_state.get("open_ports", []))
        new_ports = set(new_state.get("open_ports", []))
        new_discoveries = new_ports - old_ports
        if new_discoveries and self.verbosity != "quiet":
            console.print(f"[cyan]🧭 {self.agent_id}: New ports discovered: {new_discoveries}[/cyan]")
            
        # Track privilege level changes
        old_priv = old_state.get("privilege_level", "none")
        new_priv = new_state.get("privilege_level", "none")
        if new_priv != old_priv and self.verbosity != "quiet":
            console.print(f"[green]🧭 {self.agent_id}: Privilege level changed from {old_priv} to {new_priv}[/green]")

    def _update_complexity_score(self, state: Dict[str, Any]):
        """
        Update the environment complexity score based on current state.
        This can be used for curriculum scheduling.
        """
        # Basic complexity factors
        factors = [
            len(state.get("open_ports", [])) * 0.2,  # More ports = more complex
            {"none": 0, "user": 0.5, "low": 0.5, "high": 0.7, "root": 1.0}.get(state.get("privilege_level", "none"), 0),
            len(state.get("services", [])) * 0.15,  # More services = more complex
            state.get("blue_team_alert", 0) * 0.1,  # Higher alert = more complex
            state.get("detection_risk", 0) * 0.1   # Higher risk = more complex
        ]
        
        # Calculate complexity (normalize to 0.0-1.0 range)
        self.env_complexity = min(1.0, sum(factors) / 5.0)

    def _generate_phase_advice(self, phase: str, state: Dict[str, Any]) -> str:
        """Generate advice based on the current phase."""
        phase_advice = {
            "recon": "Conduct target discovery and initial port scanning.",
            "enumeration": "Identify services and gather detailed information.",
            "exploit": "Attempt to gain initial access by exploiting vulnerabilities.",
            "privesc": "Escalate privileges for deeper system access.",
            "exfiltrate": "Extract valuable data while minimizing traces."
        }
        
        # Basic advice based on phase
        advice = phase_advice.get(phase, "Analyze the environment and proceed cautiously.")
        
        # Add context-specific advice
        alert_level = state.get("blue_team_alert", 0)
        if alert_level > 7:
            advice += " Blue team alert is high, prioritize stealth."
        elif alert_level > 4:
            advice += " Exercise caution, blue team is moderately alert."
            
        return advice

    def act(self, state):
        """
        Implementation of AgentInterface.act()
        Returns the suggested phase for the current state.
        """
        return self.advise_phase(state)

    def learn(self):
        """
        Implementation of AgentInterface.learn()
        Scout agent doesn't do traditional learning.
        """
        pass

    def sync_memory(self):
        """
        Sync memory with MemoryRouter for global insights.
        """
        if self.memory_router:
            self.memory_router.save_memory(self.agent_id, self.memory)

    def _prune_phase_cache(self):
        """Remove expired entries from the phase cache."""
        current_time = time.time()
        expired_keys = []
        
        for key, (_, timestamp) in self.phase_cache.items():
            if current_time - timestamp > self.phase_cache_ttl:
                expired_keys.append(key)
                
        for key in expired_keys:
            del self.phase_cache[key]
            
        if expired_keys and self.verbosity == "verbose":
            console.print(f"[dim]♻️ {self.agent_id}: Pruned {len(expired_keys)} expired cache entries[/dim]")

    def display_advanced_status(self):
        """Display detailed agent status for monitoring."""
        # Create a table for the basic status
        table = Table(title=f"{self.agent_id} Status Overview")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        # Basic metrics
        table.add_row("Role", self.role)
        table.add_row("Current Phase", self.current_phase)
        table.add_row("Total Steps", str(self.total_steps))
        table.add_row("GPT Calls This Episode", str(self.gpt_calls_this_episode))
        table.add_row("Environment Complexity", f"{self.env_complexity:.2f}")
        
        # Phase durations
        phases_table = Table(title="Phase Distribution")
        phases_table.add_column("Phase", style="cyan")
        phases_table.add_column("Duration", style="green")
        phases_table.add_column("Percentage", style="yellow")
        
        total_steps = sum(self.phase_durations.values())
        for phase, duration in self.phase_durations.items():
            percentage = (duration / total_steps * 100) if total_steps > 0 else 0
            phases_table.add_row(phase, str(duration), f"{percentage:.1f}%")
            
        # Display both tables
        console.print(Panel(table, title="📊 Scout Status", border_style="cyan"))
        console.print(Panel(phases_table, title="📈 Phase Distribution", border_style="cyan"))

    def reset(self):
        """Reset agent for new episode."""
        self.gpt_calls_this_episode = 0
        self.phase_history.clear()
        self.current_phase = "recon"
        
        # Don't clear the phase cache between episodes for efficiency
        # But we can limit its size to prevent unbounded growth
        if len(self.phase_cache) > 1000:
            self._prune_phase_cache()

    def query_tactical_gpt(self, prompt, complexity="standard"):
        """
        Use GPTManager for all LLM calls, with caching, fallback, and output sanitization.
        """
        return self.gpt_manager.gpt_request(prompt, task_type="reasoning", model="gpt-4o-mini")

    # NOTE: If ScoutAgent is upgraded to DQN, follow the RedAgent/BlueAgent pattern:
    # - Add policy_net, target_net, replay_buffer, epsilon, etc.
    # - Use select_action() with argmax Q-value.
    # - Use train_on_batch() with DQN update.

# For standalone testing
if __name__ == "__main__":
    console.print("[cyan]Testing ScoutAgent in standalone mode[/cyan]")
    
    # Create a dummy state for testing
    test_state = {
        "open_ports": [22, 80, 443],
        "privilege_level": "none",
        "services": ["ssh", "http", "https"],
        "blue_team_alert": 3.5,
        "detection_risk": 0.4
    }
    
    # Create agent in test mode
    agent = ScoutAgent(verbosity="verbose")
    
    # Test phase advice
    phase = agent.advise_phase(test_state)
    console.print(f"[bold]Advised phase:[/bold] {phase}")
    
    # Test phase advice with modified state
    test_state["privilege_level"] = "user"
    test_state["blue_team_alert"] = 6.0
    phase = agent.advise_phase(test_state)
    console.print(f"[bold]Advised phase after privilege gain:[/bold] {phase}")
    
    # Test simulated step
    result = agent.simulate_step()
    console.print(Panel(f"Step result: {result}", title="Test Step"))
