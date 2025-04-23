# core/agents/orion_agent.py — ARIASKA OrionAgent v3.0
# 👁 Strategic Overseer | GPT-4.1 Intelligence | Post-Training Optimization | Curriculum Architect

import os
import json
import subprocess
from rich.console import Console
from rich.panel import Panel
from core.interfaces.agent_interface import AgentInterface
from core.utils.gpt_cache_handler import GPTCacheHandler
from datetime import datetime
import random

console = Console()


class OrionAgent(AgentInterface):
    def __init__(self, agent_manager, memory_router=None, verbosity="standard"):
        self.agent_id = "OrionAgent"
        self.agent_manager = agent_manager
        self.memory_router = memory_router  # Optional, if needed
        self.verbosity = verbosity
        self.last_curriculum_suggestion = None
        # Delay imports to avoid circular dependencies
        from core.utils.memory_manager import MemoryManager
        from core.teach.teach import TeachModule
        from core.monitor.stats_monitor import StatsMonitor

        self.memory_manager = MemoryManager(agent_name="orion_agent")
        self.cache = self.memory_manager.load_gpt_cache() if hasattr(self.memory_manager, "load_gpt_cache") else {}
        self.teach = TeachModule()
        self.stats_monitor = StatsMonitor()  # Access to multi-agent statistics
        self.training_log_path = os.path.join("logs", f"{self.agent_id}_training.log")
        os.makedirs("logs", exist_ok=True)
        self.gpt_handler = GPTCacheHandler()
        console.print(
            f"[bold blue]👁 {self.agent_id} initialized — Overseer protocols active.[/bold blue]"
        )

    def generate_strategic_chain(self, memory, force_update=False, verbosity="standard"):
        """
        Generate a strategic action chain using GPT-4.1 Full.
        Cache and update as needed.
        """
        cache_key = "orion_strategic_chain"
        if not force_update and hasattr(self, "chain_cache") and cache_key in self.chain_cache:
            chain = self.chain_cache[cache_key]
        else:
            prompt = (
                "You are ARIASKA's strategic commander (role: aria). "
                "Generate a 5-step, phase-diverse offensive chain (Recon, Enumeration, Exploit, PrivEsc, Exfil) "
                "for a red team operation. Use advanced tactics, avoid repetition and trivial commands. "
                "Respond ONLY with 5 unique, phase-ordered commands, one per line."
            )
            try:
                result = subprocess.run(
                    ["sgpt", "--model", "gpt-4.1", "--role", "aria", prompt],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=60,
                    text=True,
                )
                chain = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
                if not hasattr(self, "chain_cache"):
                    self.chain_cache = {}
                self.chain_cache[cache_key] = chain
            except Exception as e:
                console.print(f"[red]⚠ OrionAgent chain generation failed: {e}[/red]")
                chain = []
        # Add metadata to each chain step
        now = datetime.now().isoformat()
        chain_with_meta = []
        for i, cmd in enumerate(chain):
            chain_with_meta.append({
                "phase": ["recon", "enumeration", "exploit", "privesc", "exfiltrate"][i] if i < 5 else "unknown",
                "command": cmd,
                "risk_level": random.choice(["low", "medium", "high"]),
                "expected_reward": random.randint(5, 30),
                "confidence_score": round(random.uniform(0.7, 0.99), 2),
                "trigger_reason": "Orion preemptive strategy",
                "timestamp": now,
                "agent": self.agent_id
            })
        self.current_chain = chain_with_meta
        # Display concise summary
        if verbosity != "silent" and chain:
            summary = ", ".join(chain)
            console.print(f"[bold blue]Orion Strategic Chain:[/bold blue] {summary}")
        return chain_with_meta

    # ─────────────────────────────────────────────
    # 🎯 Strategic Analysis (Post-Training Optimization)
    # ─────────────────────────────────────────────
    def analyze_training(self, agents: list):
        """
        Perform deep post-training analysis across all agents.
        Identifies inefficiencies, phase weaknesses, and curriculum improvement suggestions.
        """
        console.rule(f"[bold cyan]👁 Orion: Commencing Strategic Analysis[/bold cyan]")
        summary = {}

        for agent in agents:
            memory = agent.memory_manager.load_memory()
            stats = self._analyze_memory_patterns(memory, agent.agent_id)
            performance = self._evaluate_agent_performance(agent)

            # Integrating stats from StatsMonitor
            stats["performance_metrics"] = performance
            summary[agent.agent_id] = stats

        self._generate_strategic_report(summary)
        self._log_training_event("Strategic analysis complete.")

    def _analyze_memory_patterns(self, memory, agent_id):
        """
        Analyze agent's memory using GPT-4o-mini to detect inefficiencies and suggest optimizations.
        """
        actions = memory.get("actions", [])
        if not actions:
            console.print(f"[yellow]⚠ No actions to analyze for {agent_id}[/yellow]")
            return {"status": "No Data"}

        sample = actions[:15] if len(actions) > 15 else actions
        prompt_data = json.dumps(sample, indent=2)

        # GPT-4o-mini prompt to analyze actions and detect inefficiencies
        prompt = f"""
You are ARIASKA's strategic overseer (role: aria).
Analyze the following command patterns for agent {agent_id}:

{prompt_data}

Identify:
- Redundancy issues
- Weak phases
- Opportunities for curriculum enhancement
- Any risky or inefficient behaviors

Respond in structured JSON:
{{"redundancy": "...", "weak_phases": "...", "curriculum_suggestion": "...", "risks": "..."}}
"""
        try:
            result = subprocess.run(
                [
                    "sgpt",
                    "--model",
                    "gpt-4o-mini",
                    "--temperature",
                    "0.25",
                    "--role",
                    "aria",
                    prompt,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
                text=True,
            )
            analysis = json.loads(result.stdout.strip())
            console.print(f"[green]✔ Orion analyzed {agent_id} successfully.[/green]")
            return analysis
        except Exception as e:
            console.print(
                f"[yellow]⚠ GPT-4o-mini failed: {e}. Trying GPT-4.1...[/yellow]"
            )
            return self._fallback_analysis(prompt)

    def _fallback_analysis(self, prompt):
        """
        If GPT-4o-mini fails, fall back to GPT-4.1 for complex analysis.
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
                timeout=60,
                text=True,
            )
            return json.loads(result.stdout.strip())
        except Exception as e:
            console.print(f"[red]❌ Orion fallback failed: {e}[/red]")
            return {"status": "Analysis Failed"}

    def _generate_strategic_report(self, summary):
        """
        Generate and save the strategic optimization report after analyzing all agents.
        """
        path = os.path.join("logs", "orion_strategic_report.json")
        os.makedirs("logs", exist_ok=True)

        with open(path, "w") as f:
            json.dump(summary, f, indent=2)

        panel_text = "\n".join(
            f"[bold]{agent}[/bold]: {details.get('curriculum_suggestion', 'No Data')}"
            for agent, details in summary.items()
        )

        console.print(Panel(panel_text, title="👁 Orion Strategic Recommendations"))
        console.print(f"[green]📄 Full report saved to {path}[/green]")

    def apply_orion_strategic_adjustments(self, agents):
        """
        Apply Orion's strategic adjustments based on global performance insights.
        This modulates epsilon, entropy, and reward thresholds across all agents.
        """
        console.rule(f"[bold cyan]👁 Orion: Applying Strategic Adjustments[/bold cyan]")

        for agent in agents:
            # Skip agents without stats_monitor (e.g., ScoutAgent)
            if not hasattr(agent, "stats_monitor"):
                continue
            agent_performance = self._evaluate_agent_performance(agent)
            agent_feedback = self._generate_performance_feedback(agent_performance)
            self._adjust_epsilon(agent, agent_feedback)
            self._adjust_entropy(agent, agent_feedback)
            self._suggest_curriculum_changes(agent_feedback)
            # Track token usage for visualization
            if hasattr(self.stats_monitor, "log_gpt_call"):
                self.stats_monitor.log_gpt_call(self.agent_id)
            console.print(f"[bold blue][OrionAgent] Adjusted agent: {agent.agent_id} | Epsilon: {getattr(agent, 'epsilon', 'N/A')} | Entropy: {getattr(agent, 'entropy_beta', 'N/A')}[/bold blue]")

            if hasattr(agent, "epsilon_min"):
                agent.epsilon = max(agent.epsilon * 0.98, agent.epsilon_min)
            if hasattr(agent, "entropy_beta"):
                agent.entropy_beta = max(agent.entropy_beta * 0.98, 0.005)

            if getattr(agent, "avg_reward", 0) > 50:
                self.promote_agent_to_next_curriculum(agent)
            if getattr(agent, "redundancy_rate", 0) > 0.15:
                self.force_exploration(agent)
            if getattr(agent, "gpt_tokens_used", 0) > 1000:
                self.enforce_gpt_minimization(agent)

        self._log_training_event("Strategic adjustments applied.")

    def promote_agent_to_next_curriculum(self, agent):
        """
        Promote the agent to the next curriculum level.
        """
        pass

    def force_exploration(self, agent):
        """
        Force the agent to explore new strategies.
        """
        pass

    def enforce_gpt_minimization(self, agent):
        """
        Enforce minimization of GPT token usage.
        """
        pass

    def _evaluate_agent_performance(self, agent):
        """
        Evaluate the agent's performance based on its reward, epsilon, and entropy.
        Skip agents without stats_monitor.
        """
        if not hasattr(agent, "stats_monitor"):
            return {}
        # Defensive: Use get_average_reward if available
        reward = agent.stats_monitor.get_average_reward(agent.agent_id) if hasattr(agent.stats_monitor, "get_average_reward") else 0.0
        performance = {
            "reward": reward,
            "epsilon": getattr(agent, "epsilon", 0.1),
            "entropy": getattr(agent, "entropy_beta", 0.01),
        }
        return performance

    def _generate_performance_feedback(self, performance):
        """
        Generate feedback based on the agent's performance using GPT-4.1 or fallback.
        This feedback will drive real-time adjustments.
        """
        prompt = f"""
        Evaluate the following agent performance metrics:
        Reward: {performance['reward']}, Epsilon: {performance['epsilon']}, Entropy: {performance['entropy']}
        Suggest strategic adjustments, including changes in epsilon, entropy, and reward thresholds.
        """
        feedback = self.query_tactical_gpt(prompt, complexity="high")
        return json.loads(feedback)

    def _adjust_epsilon(self, agent, feedback):
        """
        Adjust the agent's epsilon based on the strategic feedback from Orion.
        """
        if "epsilon" in feedback and hasattr(agent, "epsilon") and hasattr(agent, "epsilon_min"):
            new_epsilon = feedback["epsilon"]
            agent.epsilon = max(new_epsilon, agent.epsilon_min)
            console.print(
                f"[yellow]🎯 Adjusted epsilon for {agent.agent_id} to {new_epsilon:.4f}[/yellow]"
            )
        else:
            console.print(f"[dim]Skipping epsilon adjustment for {getattr(agent, 'agent_id', agent)} (no epsilon_min)[/dim]")

    def _adjust_entropy(self, agent, feedback):
        """
        Adjust the agent's entropy based on strategic feedback.
        """
        if "entropy" in feedback and hasattr(agent, "entropy_beta"):
            new_entropy = feedback["entropy"]
            agent.entropy_beta = max(new_entropy, 0.005)
            console.print(
                f"[yellow]🎯 Adjusted entropy for {agent.agent_id} to {new_entropy:.4f}[/yellow]"
            )
        else:
            console.print(f"[dim]Skipping entropy adjustment for {getattr(agent, 'agent_id', agent)} (no entropy_beta)[/dim]")

    def _suggest_curriculum_changes(self, feedback):
        if "curriculum_suggestion" in feedback:
            suggestion = feedback["curriculum_suggestion"]
            if not hasattr(self, "teach") or self.teach is None:
                from core.teach.teach import TeachModule
                self.teach = TeachModule()
            if suggestion and suggestion != self.last_curriculum_suggestion:
                console.print(f"[green]👁 Orion's suggestion: {suggestion}[/green]")
                self.teach.add_action(command="Curriculum Update", description=suggestion)
                self.last_curriculum_suggestion = suggestion

    def optimize_agent_memory(self, agents):
        """
        Optimize the memory of each agent using feedback from Orion.
        This ensures agents retain only the most relevant experiences and avoid redundancy.
        """
        console.rule(f"[bold cyan]👁 Orion: Optimizing Agent Memory[/bold cyan]")

        for agent in agents:
            memory = agent.memory_router.get_memory(agent.agent_id)
            optimized_memory = self._process_memory(memory, agent)
            self._update_agent_memory(agent, optimized_memory)

    def _process_memory(self, memory, agent):
        """
        Process and filter the agent's memory to retain the most valuable experiences.
        Uses GPT-4.1-nano to suggest which experiences to retain or discard.
        """
        prompt = f"""
        Analyze the following memory for agent {agent.agent_id} and suggest improvements:
        {json.dumps(memory, indent=2)}

        - Identify redundant experiences
        - Suggest improvements or deletions
        - Suggest new strategies for memory optimization
        """
        memory_feedback = self.query_tactical_gpt(prompt, complexity="high")
        return json.loads(memory_feedback)

    def _update_agent_memory(self, agent, optimized_memory):
        """
        Update the agent's memory with the optimized memory suggested by Orion.
        """
        agent.memory_router.update_memory(agent.agent_id, optimized_memory)
        console.print(
            f"[green]✔ {self.agent_id}: Memory for {agent.agent_id} optimized successfully.[/green]"
        )

    def update_global_strategy(self, agents, environment):
        """
        Update the global strategy for all agents based on the current environment and agent feedback.
        This ensures all agents work towards the same goal efficiently.
        """
        console.rule(f"[bold cyan]👁 Orion: Updating Global Strategy[/bold cyan]")
        strategy_feedback = self._generate_global_strategy(agents, environment)
        self._apply_global_strategy(agents, strategy_feedback)

    def _generate_global_strategy(self, agents, environment):
        """
        Generate a global strategy based on the current state of the environment and agent performance.
        Uses GPT-4.1-nano for strategic feedback.
        """
        prompt = f"""
        Given the current environment state: {environment}, and the following performance data for agents: {json.dumps([agent.stats_monitor.get_metrics() for agent in agents], indent=2)},
        suggest a global strategy to optimize agent collaboration and performance.
        """
        strategy_feedback = self.query_tactical_gpt(prompt, complexity="high")
        return json.loads(strategy_feedback)

    def _apply_global_strategy(self, agents, strategy_feedback):
        """
        Apply the global strategy to each agent, adjusting parameters and suggesting action plans.
        """
        for agent in agents:
            if "adjust_entropy" in strategy_feedback:
                agent.entropy_beta = strategy_feedback["adjust_entropy"]
            if "adjust_epsilon" in strategy_feedback:
                agent.epsilon = strategy_feedback["adjust_epsilon"]
            if "suggest_mode" in strategy_feedback:
                agent.current_mode = strategy_feedback["suggest_mode"]

            console.print(
                f"[magenta]🚨 Global Strategy applied to {agent.agent_id}:[/magenta] {strategy_feedback}"
            )

    def generate_strategic_report(self, agents, environment):
        """
        Generate and save the strategic optimization report based on agent performance and environment state.
        This includes feedback on agent performance, memory, and suggested curriculum changes.
        """
        console.rule(f"[bold cyan]👁 Orion: Generating Strategic Report[/bold cyan]")
        report = {}

        for agent in agents:
            performance_metrics = self._evaluate_agent_performance(agent)
            memory_feedback = self._analyze_memory_patterns(
                agent.memory_router.get_memory(agent.agent_id), agent.agent_id
            )
            report[agent.agent_id] = {
                "performance": performance_metrics,
                "memory_feedback": memory_feedback,
            }

        # Generate and save report
        report_path = os.path.join("logs", "orion_final_report.json")
        os.makedirs("logs", exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        console.print(f"[green]📄 Strategic report saved to {report_path}[/green]")

        # Optional: Show a condensed version of the report
        panel_text = "\n".join(
            f"[bold]{agent}[/bold]: {details.get('performance', 'No Data')}"
            for agent, details in report.items()
        )
        console.print(Panel(panel_text, title="👁 Orion Final Strategic Insights"))

    def execute(self, agents, environment, episodes=10):
        """
        Execute the full training and strategic adjustment cycle for all agents.
        """
        console.rule(f"[bold cyan]👁 Orion: Starting Full Execution[/bold cyan]")

        # Execute training and adjustments for each agent
        for episode in range(episodes):
            console.print(
                f"\n[cyan]🔄 Episode {episode + 1}/{episodes} started...[/cyan]"
            )

            # Evaluate and adjust agent strategies
            self.apply_orion_strategic_adjustments(agents)
            self.optimize_agent_memory(agents)

            for agent in agents:
                agent.simulate_train(episodes=1)

            # End of episode feedback
            self.generate_strategic_report(agents, environment)

        console.print(
            "[green]📚 Training complete. Strategic optimization report generated.[/green]"
        )

    def query_tactical_gpt(self, prompt, complexity="high"):
        """
        Query a tactical GPT model for feedback or suggestions.
        This is a stub for integration with your GPT querying system.
        """
        # Cap GPT calls per episode (early training)
        if hasattr(self, "gpt_calls_this_episode") and self.gpt_calls_this_episode >= 10:
            if self.verbosity != "quiet":
                console.print("[yellow]⚠ OrionAgent GPT call cap reached for this episode. Using cached logic.[/yellow]")
            return json.dumps({
                "epsilon": 0.1,
                "entropy": 0.01,
                "curriculum_suggestion": "Use cached logic."
            })
        self.gpt_calls_this_episode = getattr(self, "gpt_calls_this_episode", 0) + 1
        return json.dumps({
            "epsilon": 0.1,
            "entropy": 0.01,
            "curriculum_suggestion": "Increase exploration in early episodes."
        })

    def generate_dynamic_scenario(self, scenario, default_services):
        """
        Generate a dynamic scenario profile for the environment.
        Returns a dict with keys: difficulty, traceback_threshold, training_mode, blue_aggressiveness, services.
        """
        import random

        if scenario == "dynamic":
            difficulty = random.choice([10, 15, 20, 25])
            traceback_threshold = random.choice([60, 75, 90])
            training_mode = random.choice(["adaptive", "standard", "aggressive"])
            blue_aggressiveness = random.choice([2, 3, 4, 5])
            services = random.sample(default_services, k=min(5, len(default_services)))
        else:
            difficulty = 20
            traceback_threshold = 75
            training_mode = "adaptive"
            blue_aggressiveness = 3
            services = default_services[:5]

        return {
            "difficulty": difficulty,
            "traceback_threshold": traceback_threshold,
            "training_mode": training_mode,
            "blue_aggressiveness": blue_aggressiveness,
            "services": services,
        }

    def override_decision(self, command, state, agent_id=None):
        """
        Allow Orion to override an agent's decision based on strategic considerations.
        
        Args:
            command: The original command
            state: Current environment state
            agent_id: The ID of the agent making the decision
            
        Returns:
            str: The original command or an overridden command
        """
        return command
    
    def provide_reasoning(self, command, state):
        """
        Provide strategic reasoning about why a command is good or bad
        
        Args:
            command: The command to explain
            state: Current environment state
            
        Returns:
            str: Tactical reasoning about the command
        """
        current_phase = state.get("phase", "unknown")
        return f"Command aligns with {current_phase} phase strategic objectives."
        
    def evaluate_environment(self, state):
        """
        Evaluate the current environment state and provide strategic insights.
        
        Args:
            state: Current environment state
            
        Returns:
            str: Strategic insight or None
        """
        detection_risk = state.get("detection_risk", 0)
        phase = state.get("phase", "unknown")
        
        if detection_risk > 7.0:
            return "Increase stealth operations; blue team alert is elevated."
        elif "exfiltrate" in phase:
            return "Prepare counter-measures against exfiltration detection."
            
        return None

    def _log_training_event(self, msg):
        with open(self.training_log_path, "a") as f:
            f.write(f"{msg}\n")

    def generate_hint(self):
        """
        Provide a global strategy hint.
        """
        return "Orion recommends: synchronize agent strategies."

    def execute_command(self, command):
        try:
            output = f"OrionAgent cannot execute commands directly. Input: {command}"
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
            "nmap", "hydra", "msfconsole", "sqlmap", "ffuf", "gobuster",
            "linpeas", "winpeas", "evil-winrm", "masscan", "amass", "crackmapexec", "enum4linux", "pspy"
        ]
