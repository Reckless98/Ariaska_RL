# core/agents/shadow_agent.py — ARIASKA ShadowAgent v3.0
# ♻️ Redundancy Detection | Hybrid Logic | Smart GPT Cleanup | Memory Optimizer | Cross-Agent Coordination

import subprocess
import json
import os
from rich.console import Console

console = Console()

# Only import non-agent modules at the top level to avoid circular imports
from core.utils.memory_manager import MemoryManager
from core.memory_router import MemoryRouter
from core.gpt_manager import GPTManager
from core.utils.local_llm_manager import LocalLLMManager


class ShadowAgent:
    """
    ShadowAgent: Memory optimizer and redundancy detector.
    - Scans and optimizes agent memory for redundancy.
    - Uses GPTManager for optimization suggestions.
    - Unified memory schema: actions, rewards, scenarios.
    """

    def __init__(self, agent_manager=None, memory_router=None, verbosity="standard"):
        self.agent_id = "ShadowAgent"
        self.memory_manager = MemoryManager(agent_name="shadow_agent")
        self.memory = {
            "actions": [],
            "rewards": {},
            "scenarios": []
        }
        self.cache = (
            self.memory_manager.load_gpt_cache()
            if hasattr(self.memory_manager, "load_gpt_cache")
            else {}
        )
        self.agent_manager = agent_manager
        self.memory_router = memory_router or MemoryRouter()
        self.training_log_path = os.path.join("logs", f"{self.agent_id}_training.log")
        os.makedirs("logs", exist_ok=True)
        self.verbosity = verbosity
        self.gpt_manager = GPTManager()
        self.local_llm = LocalLLMManager(model_name="wahidmounir/SenecaLLM_x_Qwen2.5-7B-CyberSecurity-Q8_0-GGUF")
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
        """
        console.print(
            f"[bold magenta][ShadowAgent] Optimizing memory for {target_agent_id}[/bold magenta]"
        )
        memory = self.memory_manager.load_shared_knowledge(
            filename=f"{target_agent_id}_insights.json"
        )
        commands = [
            a.get("template") for a in memory.get("actions", []) if a.get("template")
        ]
        # --- Improved Redundancy Detection ---
        redundant_patterns = self._detect_redundancy_patterns(commands)
        from core.logic.redundancy_detector import suggest_memory_pruning
        rewards = [a.get("reward", 0) for a in memory.get("actions", [])]
        prune_indices = suggest_memory_pruning(commands, rewards, threshold=0.0)
        # --- OrionAgent Synchronization ---
        orion_agent = None
        if self.agent_manager and hasattr(self.agent_manager, "orion_agent"):
            orion_agent = self.agent_manager.orion_agent
        if orion_agent:
            orion_feedback = orion_agent._analyze_memory_patterns(memory, target_agent_id)
            console.print(f"[cyan]👁 OrionAgent feedback: {orion_feedback}[/cyan]")
        # --- Intelligent Caching & Cleanup ---
        if redundant_patterns or prune_indices:
            if redundant_patterns and self.verbosity != "quiet":
                console.print(
                    f"[yellow]♻ Detected {len(redundant_patterns)} redundant patterns.[/yellow]"
                )
            if prune_indices and self.verbosity != "quiet":
                console.print(
                    f"[yellow]🧹 Pruning {len(prune_indices)} ineffective memory entries.[/yellow]"
                )
                memory["actions"] = [
                    a
                    for i, a in enumerate(memory.get("actions", []))
                    if i not in prune_indices
                ]
                self.memory_manager.save_memory()
            # --- Log cleanup event ---
            console.print(
                f"[green]✔ Memory cleanup complete for {target_agent_id}: {len(prune_indices)} pruned, {len(redundant_patterns)} redundant patterns.[/green]"
            )
            self._log_training_event(
                f"Memory cleanup: {len(prune_indices)} pruned, {len(redundant_patterns)} redundant patterns for {target_agent_id}"
            )
            if hasattr(self.agent_manager, "stats_monitor"):
                self.agent_manager.stats_monitor.visualize_phase_distribution(
                    target_agent_id
                )
        else:
            if self.verbosity == "detailed":
                console.print(
                    f"[green]✔ No significant redundancy found in {target_agent_id} memory.[/green]"
                )
        # --- Reduce memory bloat by limiting cache size ---
        max_cache = 200
        if hasattr(self, "cache") and isinstance(self.cache, dict):
            if len(self.cache) > max_cache:
                old_keys = list(self.cache.keys())[: len(self.cache) - max_cache]
                for k in old_keys:
                    del self.cache[k]
                console.print(f"[dim]♻ ShadowAgent cache trimmed to {max_cache} entries.[/dim]")

    def _detect_redundancy_patterns(self, templates):
        """
        Improved: Frequency-based and pattern-based detection of redundant command templates.
        """
        from collections import Counter
        freq = Counter(templates)
        redundant = [cmd for cmd, count in freq.items() if count >= 3]
        # Pattern-based: detect A-B-A-B or A-B-C-A-B-C
        patterns = set()
        for i in range(len(templates) - 3):
            if templates[i] == templates[i + 2] and templates[i + 1] == templates[i + 3]:
                patterns.add(templates[i])
                patterns.add(templates[i + 1])
        return list(set(redundant) | patterns)

    def _resolve_with_gpt(self, patterns, target_agent_id, all_agents):
        """
        Ask GPT-4.1-nano to suggest optimizations for redundant command patterns across all agents.
        Uses GPTManager for all LLM calls (with caching, fallback, and token tracking).
        """
        for pattern in patterns:
            if pattern in self.cache:
                console.print(f"[blue]⚡ Cached optimization for:[/blue] {pattern}")
                continue

            agents_feedback = []
            for agent in all_agents:
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
                response = self.gpt_manager.gpt_request(prompt, task_type="reasoning", model="gpt-4o-mini")
                self.cache[pattern] = response
                self.memory_manager.cache_gpt_response(pattern, response)
                console.print(f"[purple]♻ GPT Optimization:[/purple] {response}")
            except Exception as e:
                console.print(f"[red]⚠ GPTManager failed to optimize {pattern}: {e}[/red]")

    def detect_redundant_patterns(self, command_history):
        """
        Check for redundant commands across the history of actions.
        """
        # Import locally to avoid circular import
        from core.logic.redundancy_detector import detect_redundancy_batch

        redundant_indices = detect_redundancy_batch(command_history)
        if redundant_indices:
            console.print(
                f"[yellow]♻ Detected {len(redundant_indices)} redundant commands in history.[/yellow]"
            )
            self._log_training_event(f"Redundant patterns checked: {command_history}")
            return redundant_indices
        return []

    def analyze_output_and_feedback(self, command, output, state):
        """
        Analyze output and determine if it's a redundant or ineffective command.
        Use both LLMs for optimization suggestions.
        """
        from core.logic.output_interpreter import analyze_output

        parsed_output = analyze_output(command, output)
        if parsed_output.get("ineffective", False):
            suggestion_prompt = f"Given the output: {output}, suggest an optimized command for the {state['phase']} phase."
            try:
                seneca_suggestion = self.local_llm.query(suggestion_prompt)
                review_prompt = (
                    f"As a memory optimization strategist, review the AI's suggestion:\n\n"
                    f"Task: {suggestion_prompt}\n"
                    f"Suggestion: {seneca_suggestion}\n\n"
                    f"Do you approve this command? If not, refine it. Respond ONLY with the final Linux command."
                )
                new_command = self.gpt_manager.gpt_request(
                    review_prompt, task_type="optimize", model="gpt-4o-mini"
                )
                return new_command
            except Exception as e:
                console.print(f"[red]❌ analyze_output_and_feedback error: {e}[/red]")
                return command
        return command

    def query_tactical_gpt(self, prompt, complexity="standard"):
        """
        Use both LLMs for memory optimization suggestions.
        """
        try:
            seneca_suggestion = self.local_llm.query(prompt)
            review_prompt = (
                f"As a memory optimization strategist, review the AI's suggestion:\n\n"
                f"Task: {prompt}\n"
                f"Suggestion: {seneca_suggestion}\n\n"
                f"Do you approve this command? If not, refine it. Respond ONLY with the final Linux command."
            )
            final_command = self.gpt_manager.gpt_request(
                review_prompt, task_type="optimize", model="gpt-4o-mini"
            )
            return final_command
        except Exception as e:
            console.print(f"[red]❌ query_tactical_gpt error: {e}[/red]")
            return self.gpt_manager.smart_decision(task_type="optimize", task_description=prompt)

    def optimize_all_agents_memory(self, agents):
        """
        Optimize memory for all agents using redundancy detection and GPTManager.
        """
        for agent in agents:
            if hasattr(agent, "memory_manager"):
                self.optimize_memory(agent.agent_id, all_agents=agents)
        # --- Log global memory optimization event ---
        console.print(f"[bold green]♻ ShadowAgent: Global memory optimization complete for all agents.[/bold green]")
        self._log_training_event("Global memory optimization complete for all agents.")

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
        # Add timestamp for better traceability
        import time
        with open(self.training_log_path, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}\n")

    def generate_hint(self):
        # New: Provide a memory optimization hint
        return "Optimize memory for redundancy reduction."

    def execute_command(self, command):
        try:
            output = f"ShadowAgent cannot execute commands directly. Input: {command}"
            return {
                "output": output,
                "recommendations": [],
                "phase": "unknown",
                "reward": 0,
                "alert": 0.0,
                "entropy": None,
            }
        except Exception as e:
            console.print(f"[red]❌ Error executing command: {e}[/red]")
            return {
                "output": f"Error executing command: {e}",
                "recommendations": [],
                "phase": "unknown",
                "reward": 0,
                "alert": 0.0,
                "entropy": None,
            }

    def get_base_commands(self):
        # For CLI completion/autosuggest
        return [
            "nmap",
            "hydra",
            "msfconsole",
            "sqlmap",
            "ffuf",
            "gobuster",
            "linpeas",
            "winpeas",
            "evil-winrm",
            "masscan",
            "amass",
            "crackmapexec",
            "enum4linux",
            "pspy",
        ]

    def log_memory_event(self, state, action, reward, next_state, gpt_tokens=0):
        """
        Log a memory event to MemoryRouter.
        """
        self.memory_router.log_transition(
            self.agent_id,
            state,
            action,
            reward,
            next_state,
            priority=abs(reward)+0.01,
            gpt_tokens=gpt_tokens
        )

    def train_on_batch(self, batch_size=32):
        batch = self.memory_router.sample_batch(self.agent_id, batch_size=batch_size)
        if not batch:
            console.print("[yellow]⚠ Not enough experiences for batch training.[/yellow]")
            return
        # ...add your batch training logic here if needed...

    # NOTE: If ShadowAgent is upgraded to DQN, follow the RedAgent/BlueAgent pattern:
    # - Add policy_net, target_net, replay_buffer, epsilon, etc.
    # - Use select_action() with argmax Q-value.
    # - Use train_on_batch() with DQN update.


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
