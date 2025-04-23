# core/agents/shadow_agent.py — ARIASKA ShadowAgent v3.0
# ♻️ Redundancy Detection | Hybrid Logic | Smart GPT Cleanup | Memory Optimizer | Cross-Agent Coordination

import subprocess
import json
import os
from rich.console import Console

console = Console()

# Only import non-agent modules at the top level to avoid circular imports
from core.utils.memory_manager import MemoryManager

class ShadowAgent:
    def __init__(self, agent_manager=None, memory_router=None, verbosity="standard"):
        self.agent_id = "ShadowAgent"
        self.memory_manager = MemoryManager(agent_name="shadow_agent")
        self.cache = self.memory_manager.load_gpt_cache() if hasattr(self.memory_manager, "load_gpt_cache") else {}
        self.agent_manager = agent_manager
        self.memory_router = memory_router  # Optional, if needed
        self.training_log_path = os.path.join("logs", f"{self.agent_id}_training.log")
        os.makedirs("logs", exist_ok=True)
        self.verbosity = verbosity
        console.print(
            f"[purple]♻️ {self.agent_id} initialized — Monitoring for redundancy.[/purple]"
        )

    def set_agent_manager(self, agent_manager):
        self.agent_manager = agent_manager

    def optimize_memory(self, target_agent_id, all_agents=None):
        """
        Scan the target agent's memory for redundant command patterns.
        Use rule-based detection first, GPT for deeper insights.
        Also optimizes memory across all agents based on global performance feedback from Orion.
        
        Args:
            target_agent_id: ID of the agent whose memory to optimize
            all_agents: Optional list of all agents (defaults to None)
        """
        console.print(f"[bold magenta][ShadowAgent] Optimizing memory for {target_agent_id}[/bold magenta]")
        memory = self.memory_manager.load_shared_knowledge(
            filename=f"{target_agent_id}_insights.json"
        )
        commands = [
            a.get("template") for a in memory.get("actions", []) if a.get("template")
        ]

        redundant_patterns = self._detect_redundancy_patterns(commands)
        # Also detect ineffective patterns
        from core.logic.redundancy_detector import suggest_memory_pruning
        rewards = [a.get("reward", 0) for a in memory.get("actions", [])]
        prune_indices = suggest_memory_pruning(commands, rewards, threshold=0.0)
        if redundant_patterns or prune_indices:
            if redundant_patterns and self.verbosity != "quiet":
                console.print(
                    f"[yellow]♻ Detected {len(redundant_patterns)} redundant patterns.[/yellow]"
                )
            if prune_indices and self.verbosity != "quiet":
                console.print(f"[yellow]🧹 Pruning {len(prune_indices)} ineffective memory entries.[/yellow]")
                memory["actions"] = [a for i, a in enumerate(memory.get("actions", [])) if i not in prune_indices]
                self.memory_manager.save_memory()
            if hasattr(self.agent_manager, "stats_monitor"):
                self.agent_manager.stats_monitor.visualize_phase_distribution(target_agent_id)
        else:
            if self.verbosity == "detailed":
                console.print(
                    f"[green]✔ No significant redundancy found in {target_agent_id} memory.[/green]"
                )
        self._log_training_event(f"Optimized memory for {target_agent_id}")

    def _detect_redundancy_patterns(self, templates):
        """
        Simple frequency-based detection of redundant command templates across agents.
        """
        freq = {}
        for tmpl in templates:
            freq[tmpl] = freq.get(tmpl, 0) + 1
        return [cmd for cmd, count in freq.items() if count >= 3]  # Threshold: 3+

    def _resolve_with_gpt(self, patterns, target_agent_id, all_agents):
        """
        Ask GPT-4.1-nano to suggest optimizations for redundant command patterns across all agents.
        Uses GPT-4o-mini for main tasks and GPT-4.1 for complex queries.
        """
        # Only access agent.memory_manager, not agent classes directly
        for pattern in patterns:
            if (pattern in self.cache):
                console.print(f"[blue]⚡ Cached optimization for:[/blue] {pattern}")
                continue

            agents_feedback = []
            for agent in all_agents:
                # Defensive: agent must have memory_manager and agent_id
                if hasattr(agent, "memory_manager") and hasattr(agent, "agent_id"):
                    feedback = agent.memory_manager.load_shared_knowledge(
                        f"{agent.agent_id}_insights.json"
                    )
                    agents_feedback.append(feedback)

            prompt = f"""
You are a cybersecurity memory optimizer.
Detected redundant command pattern:
{pattern}

Suggest how to consolidate or improve this to reduce inefficiency.
Respond concisely.

Additionally, consider the following agents' feedback for optimization:
{json.dumps(agents_feedback, indent=2)}
"""
            try:
                result = subprocess.run(
                    [
                        "sgpt",
                        "--model",
                        "gpt-4.1",
                        "--temperature",
                        "0.3",
                        "--role",
                        "aria",
                        prompt,
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=20,
                    text=True,
                )
                suggestion = result.stdout.strip()
                self.cache[pattern] = suggestion
                self.memory_manager.cache_gpt_response(pattern, suggestion)
                console.print(f"[purple]♻ GPT Optimization:[/purple] {suggestion}")
            except Exception as e:
                console.print(f"[red]⚠ GPT failed to optimize {pattern}: {e}[/red]")

    def detect_redundant_patterns(self, command_history):
        """
        Check for redundant commands across the history of actions.
        """
        # Import locally to avoid circular import
        from core.logic.redundancy_detector import detect_redundancy_batch
        redundant_indices = detect_redundancy_batch(command_history)
        if redundant_indices:
            console.print(f"[yellow]♻ Detected {len(redundant_indices)} redundant commands in history.[/yellow]")
            self._log_training_event(f"Redundant patterns checked: {command_history}")
            return redundant_indices
        return []

    def analyze_output_and_feedback(self, command, output, state):
        """
        Analyze output and determine if it's a redundant or ineffective command.
        May trigger GPT feedback if needed.
        """
        # Import locally to avoid circular import
        from core.logic.output_interpreter import analyze_output
        parsed_output = analyze_output(command, output)
        if (parsed_output.get("ineffective", False)):
            suggestion_prompt = f"Given the output: {output}, suggest an optimized command for the {state['phase']} phase."
            new_command = self.query_tactical_gpt(suggestion_prompt, complexity="high")
            return new_command
        return command

    def query_tactical_gpt(self, prompt, complexity="standard"):
        """
        Query GPT for tactical analysis and suggestions. GPT-4o-mini used for efficiency, GPT-4.1 for rare, complex tasks.
        """
        model = "gpt-4o-mini" if complexity == "standard" else "gpt-4.1"
        try:
            result = subprocess.run(
                ["sgpt", "--model", model, "--role", "shadow_agent", prompt],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=15,
                text=True,
            )
            response = result.stdout.strip()
            console.print(f"[magenta]🧠 GPT Tactical Insight:[/magenta] {response}")
            return response
        except Exception as e:
            console.print(f"[red]⚠ GPT query failed: {e}[/red]")
            return "Optimize command manually."

    def optimize_all_agents_memory(self, agents):
        """
        Use feedback from Orion and the other agents to optimize each agent's memory.
        """
        for agent in agents:
            self.optimize_memory(agent.agent_id, agents)

    def log_memory_to_file(self):
        """
        Export the current memory for redundancy detection and optimization.
        """
        memory_data = self.memory_manager.get_all_memory()
        file_path = f"{self.agent_id}_memory.json"
        with open(file_path, "w") as f:
            json.dump(memory_data, f, indent=2)
        console.print(f"[green]✔ Memory logged to {file_path}[/green]")

    def safe_shutdown(self):
        """
        Clean shutdown procedure for ShadowAgent.
        """
        self.log_memory_to_file()
        console.print(f"[purple]♻️ {self.agent_id}: Safe shutdown complete.[/purple]")

    def _log_training_event(self, msg):
        with open(self.training_log_path, "a") as f:
            f.write(f"{msg}\n")

    def generate_hint(self):
        # New: Provide a memory optimization hint
        return "Optimize memory for redundancy reduction."

    def execute_command(self, command):
        # New: Simulate a command execution for CLI
        output = f"ShadowAgent cannot execute commands directly. Input: {command}"
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
            "nmap", "hydra", "msfconsole", "sqlmap", "ffuf", "gobuster",
            "linpeas", "winpeas", "evil-winrm", "masscan", "amass", "crackmapexec", "enum4linux", "pspy"
        ]

# ─────────────────────────────────────────────
# 🎬 Execution Test Hook
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Only instantiate ShadowAgent for standalone test, not other agents
    agent = ShadowAgent()
    agent.optimize_memory("RedAgent", [agent])
    agent.optimize_memory("BlueAgent", [agent])
    agent.detect_redundant_patterns(["command1", "command1", "command1", "command2"])
    agent.safe_shutdown()  # Ensures memory logging and shutdown
