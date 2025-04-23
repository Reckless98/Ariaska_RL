# core/agents/scout_agent.py — ARIASKA ScoutAgent v3.0
# 🧭 Phase Navigator | GPT-4.1-Nano Optimized | Zero-Redundancy | Multi-Agent Coordination | Global Phase Awareness

import random
import subprocess
import os
import json
from rich.console import Console
from core.monitor.stats_monitor import StatsMonitor
from core.interfaces.agent_interface import AgentInterface

console = Console()

PHASES = ["recon", "enumeration", "exploit", "privesc", "exfiltrate"]

class ScoutAgent(AgentInterface):
    def __init__(self, agent_id="ScoutAgent", memory_manager=None, agent_manager=None, memory_router=None, verbosity="standard"):
        self.agent_id = agent_id
        # Import MemoryManager here to avoid circular imports
        if memory_manager is None:
            from core.utils.memory_manager import MemoryManager
            self.memory_manager = MemoryManager(agent_name="scout_agent")
        else:
            self.memory_manager = memory_manager
        self.cache = self.memory_manager.load_gpt_cache() if hasattr(self.memory_manager, "load_gpt_cache") else {}
        self.agent_manager = agent_manager  # Set by AgentManager or passed in
        self.memory_router = memory_router  # Optional, if needed
        self.training_log_path = os.path.join("logs", f"{self.agent_id}_training.log")
        os.makedirs("logs", exist_ok=True)
        self.stats_monitor = StatsMonitor(agents_list=[self.agent_id])
        self.phase_history = []
        self.verbosity = verbosity
        self.last_phase = None
        self.phase_repeat_count = 0
        self.gpt_phase_cooldown = 0
        console.print(f"[cyan]🧭 {self.agent_id} initialized — Phase navigation online.[/cyan]")

    def set_agent_manager(self, agent_manager):
        self.agent_manager = agent_manager

    def advise_phase(self, state, all_agents=None, shared_context=None):
        """
        Suggest the most logical next phase based on the current state of all agents.
        Uses cache, then GPT-4o-mini (token-optimized), and global strategy from Orion if needed.
        Tracks token usage per step for visualization.
        
        Parameters:
            state: Current environment state
            all_agents: Optional list of all agents for coordination (defaults to None for backward compatibility)
            shared_context: Optional shared context for consensus building (defaults to None)
        """
        context_key = self._generate_context_key(state)

        # Cap GPT calls per episode
        self.gpt_calls_this_episode = getattr(self, "gpt_calls_this_episode", 0)
        if self.gpt_calls_this_episode >= 10:
            if self.verbosity != "quiet":
                console.print("[yellow]⚠ ScoutAgent GPT call cap reached for this episode. Using cached phase.[/yellow]")
            phase = self.phase_history[-1] if self.phase_history else "recon"
            return phase

        # Use cache if available
        if (context_key in self.cache):
            phase = self.cache[context_key]
            # Fix: If phase is a dict (from cache_gpt_response), extract 'response'
            if isinstance(phase, dict) and "response" in phase:
                phase = phase["response"]
            console.print(f"[blue]🔹 ScoutAgent (cache): Recommended Phase → {phase}[/blue]")
            self._log_training_event(f"Advise phase: {phase} for state: {state}")
            self.stats_monitor.log_gpt_call(self.agent_id)
            self.phase_history.append(phase)
            if self.phase_history[-4:] == ['recon'] * 4:
                phase = self._suggest_alternative_phase()
            return phase

        # Use shared_context to build consensus if available
        if shared_context:
            phase_votes = [v for k, v in shared_context.items() if k.endswith("_phase")]
            if phase_votes:
                # Consensus: pick most common phase among agents
                from collections import Counter
                most_common = Counter(phase_votes).most_common(1)
                if most_common:
                    consensus_phase = most_common[0][0]
                    console.print(f"[green]🧭 Consensus phase: {consensus_phase}[/green]")
                    self.stats_monitor.log_gpt_call(self.agent_id)
                    self.phase_history.append(consensus_phase)
                    if self.phase_history[-4:] == ['recon'] * 4:
                        consensus_phase = self._suggest_alternative_phase()
                    return consensus_phase

        # Query Orion for global phase strategy before GPT suggestion
        global_phase = None
        if all_agents:
            global_phase = self._get_orion_global_phase(all_agents)
        if global_phase:
            console.print(f"[green]🧭 Orion suggests phase: {global_phase}[/green]")
            self._log_training_event(f"Advise phase: {global_phase} for state: {state}")
            self.stats_monitor.log_gpt_call(self.agent_id)
            self.phase_history.append(global_phase)
            if self.phase_history[-4:] == ['recon'] * 4:
                global_phase = self._suggest_alternative_phase()
            return global_phase

        # Use GPT-4o-mini for phase suggestion (token-optimized)
        current_phase = state.get("phase", "recon")
        if self.last_phase == current_phase:
            self.phase_repeat_count += 1
        else:
            self.phase_repeat_count = 1
            self.last_phase = current_phase
        # Avoid GPT spam: if phase repeats ≥ 3, suppress GPT call and use cached
        suppress_gpt_call = self.phase_repeat_count >= 3 or self.gpt_phase_cooldown > 0
        if suppress_gpt_call:
            if self.verbosity in ("standard", "detailed"):
                console.print(f"[yellow][ScoutAgent] Phase '{current_phase}' repeated x{self.phase_repeat_count}, using cached phase.[/yellow]")
            self.gpt_phase_cooldown = max(self.gpt_phase_cooldown - 1, 0)
            return current_phase

        phase = self._gpt_advise_phase(state)
        if phase:
            self.cache[context_key] = phase
            if hasattr(self.memory_manager, "cache_gpt_response"):
                self.memory_manager.cache_gpt_response(context_key, phase)
        self._log_training_event(f"Advise phase: {phase} for state: {state}")
        self.stats_monitor.log_gpt_call(self.agent_id)
        self.phase_history.append(phase)
        if self.phase_history[-4:] == ['recon'] * 4:
            phase = self._suggest_alternative_phase()
        # Track phase for visualization
        if hasattr(self.agent_manager, "stats_monitor"):
            self.agent_manager.stats_monitor.visualize_phase_distribution(self.agent_id)

        # Streamlined phase advice logging
        if len(self.phase_history) >= 5 and all(ph == self.phase_history[-1] for ph in self.phase_history[-5:]):
            if self.verbosity != "quiet":
                console.print(f"[ScoutAgent] Phase advice stable: {self.phase_history[-1]} (x5)")

        # After GPT call, set cooldown
        self.gpt_phase_cooldown = 2

        return phase

    def _generate_context_key(self, state):
        return f"{state.get('phase')}-{state.get('privilege_level')}-{state.get('blue_team_alert')}"

    def _get_orion_global_phase(self, all_agents):
        """
        Get global phase suggestion based on feedback from all agents and Orion.
        This ensures that the phase decision aligns with the global strategy.
        """
        # Gather feedback from all agents
        global_phase = None
        for agent in all_agents:
            if hasattr(agent, 'current_mode') and agent.current_mode in PHASES:
                if global_phase is None:
                    global_phase = agent.current_mode
                elif agent.current_mode != global_phase:
                    global_phase = "recon"  # Default to recon if there’s conflict in phases
                    break
        return global_phase

    def _gpt_advise_phase(self, state):
        """
        Query GPT-4o-mini to recommend the next tactical phase based on current state.
        Token-optimized prompt.
        """
        prompt = (
            f"Phase: {state.get('phase')}, Privilege: {state.get('privilege_level')}, "
            f"Ports: {state.get('open_ports')}, Alert: {state.get('blue_team_alert')}, "
            f"Risk: {state.get('detection_risk')}. "
            "Suggest next phase: recon, enumeration, exploit, privesc, or exfiltrate. Respond with phase only."
        )
        try:
            result = subprocess.run(
                ["sgpt", "--model", "gpt-4o-mini", "--temperature", "0.15", "--role", "aria", prompt],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10, text=True
            )
            phase = result.stdout.strip().lower()
            if phase in PHASES:
                console.print(f"[cyan]🧭 ScoutAgent (GPT-4o-mini): Next Phase → {phase}[/cyan]")
                self.gpt_calls_this_episode += 1
                return phase
            else:
                console.print(f"[yellow]⚠ GPT returned invalid phase: {phase}. Defaulting to current.[/yellow]")
                return state.get("phase", "recon")
        except Exception as e:
            console.print(f"[red]⚠ ScoutAgent GPT error: {e}. Defaulting to current phase.[/red]")
            return state.get("phase", "recon")

    def _suggest_alternative_phase(self):
        # Suggest a phase not recently used
        for phase in PHASES:
            if self.phase_history.count(phase) < 2:
                return phase
        return random.choice(PHASES)

    def detect_redundancy(self, command_history):
        """
        Detect redundant commands across the history of actions.
        Uses GPT for complex situations when needed.
        """
        # Import detect_redundancy here to avoid circular imports
        from core.logic.redundancy_detector import detect_redundancy, detect_redundancy_batch
        
        # First check for simple redundancy
        if any(detect_redundancy(command_history[:i], cmd) for i, cmd in enumerate(command_history)):
            console.print(f"[yellow]⚠ Redundant commands detected in history![/yellow]")
            self._log_training_event(f"Redundancy check: {command_history}")
            return True
            
        # Then do batch analysis for pattern detection
        redundant_indices = detect_redundancy_batch(command_history)
        if redundant_indices:
            console.print(f"[yellow]⚠ Pattern-based redundancy detected in history! ({len(redundant_indices)} instances)[/yellow]")
            self._log_training_event(f"Redundancy check: {command_history}")
            return True
        
        return False

    def optimize_memory(self, all_agents):
        """
        Optimize the memory of all agents by using feedback from the global system.
        Helps to eliminate inefficiencies and improve agent training.
        """
        for agent in all_agents:
            # Pass memory_router as an attribute of agent, do not import agent classes here
            if hasattr(agent, 'memory_router'):
                agent.memory_router.optimize_memory(agent.agent_id)
                console.print(f"[green]✔ {self.agent_id}: Optimized memory for {agent.agent_id}[/green]")
                console.print(f"[bold green][ScoutAgent] Optimized memory for {agent.agent_id}[/bold green]")

    def log_memory_to_file(self):
        """
        Log memory data to a file for analysis and redundancy detection.
        """
        memory_data = self.memory_manager.get_all_memory()
        file_path = f"{self.agent_id}_memory.json"
        with open(file_path, "w") as f:
            json.dump(memory_data, f, indent=2)
        console.print(f"[green]✔ Memory logged to {file_path}[/green]")

    def safe_shutdown(self):
        """
        Clean shutdown procedure for ScoutAgent.
        """
        self.log_memory_to_file()
        console.print(f"[cyan]🧭 {self.agent_id}: Safe shutdown complete.[/cyan]")

    def _log_training_event(self, msg):
        with open(self.training_log_path, "a") as f:
            f.write(f"{msg}\n")

    def generate_hint(self):
        # New: Provide a phase hint
        return f"Recommended phase: {random.choice(PHASES)}"

    def execute_command(self, command):
        # New: Simulate a command execution for CLI
        output = f"ScoutAgent cannot execute commands directly. Input: {command}"
        return {
            "output": output,
            "recommendations": [],
            "phase": "unknown",
            "reward": 0,
            "alert": 0.0,
            "entropy": None,
        }

    def get_base_commands(self):
        # For CLI completion/autosuggest
        return [
            "nmap", "masscan", "gobuster", "ffuf", "amass", "enum4linux"
        ]

    def reset(self):
        """Reset stats for compatibility with multi-agent manager."""
        if hasattr(self, "stats_monitor"):
            self.stats_monitor.reset()

    def get_average_reward(self):
        """For compatibility with OrionAgent and stats_monitor."""
        if hasattr(self, "stats_monitor"):
            return self.stats_monitor.get_average_reward(self.agent_id)
        return 0.0

if __name__ == "__main__":
    # Import MemoryManager here to avoid top-level import
    from core.utils.memory_manager import MemoryManager
    memory_manager = MemoryManager(agent_name="scout_agent")
    agent = ScoutAgent(memory_manager=memory_manager)
    agent.optimize_memory([agent])
    agent.detect_redundancy(["command1", "command1", "command1", "command2"])
    agent.safe_shutdown()  # Ensures memory logging and shutdown
