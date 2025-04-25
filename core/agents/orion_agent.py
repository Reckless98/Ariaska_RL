# core/agents/orion_agent.py — ARIASKA OrionAgent v12.0 APEX OVERSEER
# 👁️ Strategic Overseer | 🔄 Adaptive Chain Builder | ♻️ Dynamic Memory Integration | 🧠 High-Level Performance Analysis

import os
import json
import time
import random
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from core.interfaces.agent_interface import AgentInterface
from core.interfaces.memory_sync_interface import MemorySyncInterface
from core.gpt_manager import GPTManager
from core.utils.llm_orchestrator import LLMRouter

console = Console()

class OrionAgent(AgentInterface, MemorySyncInterface):
    """
    OrionAgent: Strategic overseer for the multi-agent system.
    - Analyzes agent performance and provides high-level strategic guidance
    - Creates/modifies attack chains based on performance data
    - Provides dynamic adjustments to agent parameters (epsilon, learning rate)
    - Uses GPTManager for all LLM calls with efficient caching and token tracking
    - Unified memory schema: actions, rewards, scenarios
    - Provides dynamic scenario generation for CyberEnvironment (difficulty, thresholds, etc.)
    """

    def __init__(
        self,
        agent_id="OrionAgent",
        role="StrategicOverseer",
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
        
        # Strategic chain tracking
        self.current_chain = []
        self.chain_history = []
        self.chain_performance = {}
        
        # GPT usage and memory paths
        self.gpt_manager = GPTManager()
        self.insights_path = os.path.join("logs", f"{self.agent_id}_insights.jsonl")
        self.training_log_path = os.path.join("logs", f"{self.agent_id}_training.log")
        os.makedirs(os.path.dirname(self.insights_path), exist_ok=True)
        
        # Performance tracking
        self.agent_performance = {}
        self.global_strategy = "balanced"
        self.strategy_history = []
        self.last_analysis_time = 0
        self.analysis_cooldown = 300  # 5 minutes between full analyses
        
        self.llm_router = LLMRouter()
        
        console.print(f"[magenta]👁️ {self.agent_id} initialized — Strategic Oversight Active[/magenta]")

    def analyze_training(self, agents):
        """
        Analyze training performance across all agents and generate strategic insights.
        Uses GPTManager for efficient token usage and caching.
        
        Args:
            agents (list): List of agent objects to analyze
        """
        console.print(f"[bold magenta]👁️ {self.agent_id}: Analyzing multi-agent performance...[/bold magenta]")
        
        # Skip if called too frequently (except in verbose mode)
        current_time = time.time()
        if current_time - self.last_analysis_time < self.analysis_cooldown and self.verbosity != "verbose":
            console.print("[dim]👁️ Analysis skipped: cooldown period active.[/dim]")
            return
        
        self.last_analysis_time = current_time
        
        # Gather performance data from all agents
        training_data = {}
        for agent in agents:
            if agent.agent_id == self.agent_id:
                continue
                
            agent_data = {
                "rewards": [],
                "actions": [],
                "phases": []
            }
            
            # Extract metrics if available
            if hasattr(agent, "stats_monitor") and agent.stats_monitor:
                stats = agent.stats_monitor.agent_stats.get(agent.agent_id, {})
                agent_data["rewards"] = stats.get("rewards", [])[-50:]
                agent_data["gpt_calls"] = stats.get("gpt_calls", 0)
                
            # Extract command history if available
            if hasattr(agent, "command_history"):
                agent_data["actions"] = agent.command_history[-50:] if agent.command_history else []
            
            # Extract phases from redagent_brain if available
            if hasattr(agent, "redagent_brain") and hasattr(agent.redagent_brain, "episodic_memory"):
                episodes = agent.redagent_brain.episodic_memory
                if episodes:
                    last_episode = episodes[-1]
                    phases = [step.get("state", {}).get("phase") for step in last_episode.get("steps", [])]
                    agent_data["phases"] = [p for p in phases if p]
            
            training_data[agent.agent_id] = agent_data
        
        # Generate performance feedback using GPT
        feedback = self._generate_performance_feedback(training_data)
        console.print(f"[magenta]👁️ Strategic Analysis:[/magenta] {feedback}")
        
        # Log insights to file
        self._log_insight({
            "timestamp": time.time(),
            "analysis": feedback,
            "training_data": training_data
        })
        
        # Update memory with strategic insights
        self.memory["actions"].append({
            "command": "analyze_training",
            "phase": "strategic",
            "reward": 10.0,
            "output": feedback,
            "timestamp": time.time()
        })
        
        # Return feedback for agent coordination
        return feedback

    def _generate_performance_feedback(self, training_data):
        """
        Generate performance feedback using GPTManager instead of subprocess call.
        
        Args:
            training_data (dict): Training data for all agents
        
        Returns:
            str: Strategic feedback
        """
        try:
            # Prepare summary data for the GPT prompt
            prompt = f"""
As Orion, the strategic overseer for a multi-agent cybersecurity system, analyze this training data 
and provide strategic recommendations:

{json.dumps(training_data, indent=2)}

Focus on:
1. Success/failure patterns in RedAgent's actions
2. Key areas for improvement
3. Agent coordination opportunities
4. Suggested tactical adjustments

Respond with 3-4 concrete, actionable recommendations.
"""
            # Use GPTManager instead of direct subprocess call
            response = self.gpt_manager.gpt_request(
                prompt=prompt, 
                task_type="reflection",
                agent_id=self.agent_id
            )
            
            return response
            
        except Exception as e:
            console.print(f"[red]❌ GPT analysis error: {e}. Using fallback analysis.[/red]")
            return "Automatic analysis unavailable. Focus on diversifying agent strategies and improving attack phases."

    def apply_orion_strategic_adjustments(self, agents):
        """
        Apply strategic adjustments to agents based on performance analysis.
        
        Args:
            agents (list): List of agent objects to adjust
        """
        if not agents:
            return
            
        console.print(f"[magenta]👁️ {self.agent_id}: Applying strategic adjustments...[/magenta]")
        
        # Perform analysis to get fresh insights
        insights = self.analyze_training(agents)
        
        # Extract key signals from insights for agent adjustment
        focus_more_exploration = "explore" in insights.lower() or "diverse" in insights.lower()
        focus_more_exploitation = "exploit" in insights.lower() or "leverage" in insights.lower()
        improve_phase = None
        
        for phase in ["recon", "enumeration", "exploit", "privesc", "exfiltrate"]:
            if phase in insights.lower():
                improve_phase = phase
                break
        
        # Apply adjustments based on insights
        for agent in agents:
            if agent.agent_id == self.agent_id:
                continue
                
            # Adjust epsilon (exploration rate)
            if hasattr(agent, "epsilon"):
                if focus_more_exploration:
                    agent.epsilon = min(agent.epsilon * 1.2, 0.95)
                    console.print(f"[cyan]👁️ Increased exploration for {agent.agent_id}: {agent.epsilon:.3f}[/cyan]")
                elif focus_more_exploitation:
                    agent.epsilon = max(agent.epsilon * 0.8, agent.epsilon_min)
                    console.print(f"[cyan]👁️ Decreased exploration for {agent.agent_id}: {agent.epsilon:.3f}[/cyan]")
            
            # Adjust entropy beta (action diversity)
            if hasattr(agent, "entropy_beta"):
                if focus_more_exploration:
                    agent.entropy_beta = min(agent.entropy_beta * 1.3, 0.05)
                    console.print(f"[cyan]👁️ Increased entropy for {agent.agent_id}: {agent.entropy_beta:.3f}[/cyan]")
                elif focus_more_exploitation:
                    agent.entropy_beta = max(agent.entropy_beta * 0.7, 0.001)
                    console.print(f"[cyan]👁️ Decreased entropy for {agent.agent_id}: {agent.entropy_beta:.3f}[/cyan]")

            # --- Stuck/Redundancy Detection ---
            if self.agent_stuck(agent):
                suggestion = self.llm_router.route_task(
                    "strategic",
                    f"Agent {agent.agent_id} is stuck with reward {getattr(agent, 'last_reward', 0)}. Suggest new action."
                )
                console.print(f"[yellow]🔄 GPT Suggestion for {agent.agent_id}:[/yellow] {suggestion}")
                # Optionally: store suggestion or inject as next action
                if hasattr(agent, "priority_queue"):
                    agent.priority_queue = [suggestion]
            # ...existing curriculum adjustment logic...
            # Example: If agent is performing well, increase difficulty or reduce epsilon
            avg_reward = getattr(agent.stats_monitor, "get_average_reward", lambda: 0.0)()
            if avg_reward > 15 and hasattr(agent, "epsilon"):
                agent.epsilon = max(agent.epsilon * 0.95, agent.epsilon_min)
                console.print(f"[cyan]👁️ {agent.agent_id}: High reward, reducing epsilon to {agent.epsilon:.3f}[/cyan]")
            if avg_reward > 20 and hasattr(agent, "env") and hasattr(agent.env, "difficulty_level"):
                agent.env.difficulty_level = min(getattr(agent.env, "difficulty_level", 1) + 1, getattr(agent.env, "max_difficulty", 20))
                console.print(f"[magenta]👁️ {agent.agent_id}: Raising environment difficulty to {agent.env.difficulty_level}[/magenta]")
        
        # Generate new attack chain if needed
        if random.random() < 0.3:  # 30% chance of new chain
            self.generate_attack_chain(agents)
            
        # Return insights
        return insights

    def agent_stuck(self, agent, stagnation_window=5, min_reward_delta=0.1):
        """
        Detect if agent is stuck: reward hasn't improved for N steps or repeated actions.
        """
        # Check reward stagnation
        rewards = getattr(agent.stats_monitor, "agent_stats", {}).get(agent.agent_id, {}).get("rewards", [])
        if len(rewards) >= stagnation_window:
            recent = rewards[-stagnation_window:]
            if max(recent) - min(recent) < min_reward_delta:
                return True
        # Check repeated actions
        cmd_hist = getattr(agent, "command_history", [])
        if len(cmd_hist) >= stagnation_window and len(set(cmd_hist[-stagnation_window:])) == 1:
            return True
        return False

    def _log_insight(self, insight):
        """
        Log an insight to the insights file.
        
        Args:
            insight (dict): Insight to log
        """
        try:
            with open(self.insights_path, "a") as f:
                f.write(json.dumps(insight) + "\n")
        except Exception as e:
            console.print(f"[red]❌ Failed to log insight: {e}[/red]")

    def _log_training_event(self, message):
        """
        Log a training event.
        
        Args:
            message (str): Message to log
        """
        try:
            with open(self.training_log_path, "a") as f:
                f.write(f"[{time.time()}] {message}\n")
        except Exception as e:
            console.print(f"[red]❌ Failed to log training event: {e}[/red]")

    def generate_attack_chain(self, agents):
        """
        Generate a strategic attack chain using LLMRouter (SenecaLLM→LilyLLM→GPT-4o fallback).
        """
        console.print(f"[magenta]🔗 {self.agent_id}: Generating strategic attack chain...[/magenta]")
        
        # Get RedAgent from agents list
        red_agent = None
        for agent in agents:
            if (agent.agent_id == "RedAgent"):
                red_agent = agent
                break
                
        if not red_agent:
            console.print("[yellow]⚠ RedAgent not found, cannot generate attack chain.[/yellow]")
            return
            
        # Gather successful commands from RedAgent memory
        successful_commands = []
        if hasattr(red_agent, "memory_manager") and red_agent.memory_manager:
            memory = red_agent.memory_manager.memory
            for action in memory.get("actions", []):
                if action.get("reward", 0) > 5:
                    successful_commands.append(action.get("command"))
        
        # If not enough successful commands, add default commands
        if len(successful_commands) < 3:
            successful_commands.extend([
                "nmap -sS -sV 10.10.10.10",
                "gobuster dir -u http://10.10.10.10 -w /usr/share/wordlists/common.txt",
                "hydra -l admin -P /usr/share/wordlists/rockyou.txt ssh://10.10.10.10"
            ])
        
        # Generate chain using LLMRouter
        task_desc = "Generate optimized 5-step attack chain based on recent successes: " + ", ".join(successful_commands)
        chain = self.llm_router.route_task("planner", task_desc)
        
        # Store and broadcast chain
        self.current_chain = chain
        self.chain_history.append({
            "timestamp": time.time(),
            "chain": chain,
            "performance": None  # Will be updated after execution
        })
        
        console.print(f"[magenta]🔗 New attack chain generated: {', '.join(chain)}[/magenta]")
        
        # Return the generated chain
        return chain

    def _gpt_generate_chain(self, successful_commands):
        """
        Use GPTManager to generate an optimized attack chain.
        
        Args:
            successful_commands (list): List of successful commands
        
        Returns:
            list: Generated attack chain
        """
        # Limit context to avoid token bloat
        cmd_context = successful_commands[-15:]
        
        prompt = f"""
As OrionAgent, the strategic overseer, create an optimized 5-step attack chain using the following successful commands as reference:

{json.dumps(cmd_context)}

The attack chain should:
1. Follow a logical progression through phases (recon → enumeration → exploit → privesc → exfiltrate)
2. Be coherent and build on previous steps
3. Include specific commands (not generic descriptions)

Respond with a JSON array of 5 commands representing the optimal attack chain:
"""

        try:
            # Use GPTManager for the request
            response = self.gpt_manager.gpt_request(
                prompt=prompt,
                task_type="strategic",
                agent_id=self.agent_id,
                model="gpt-4.1"  # Use primary model for strategic work
            )
            
            # Parse JSON response
            try:
                chain = json.loads(response)
                if isinstance(chain, list) and len(chain) > 0:
                    return chain[:5]  # Ensure we have at most 5 commands
            except json.JSONDecodeError:
                # Fallback: extract commands using regex
                import re
                commands = re.findall(r'"(.*?)"', response)
                if commands:
                    return commands[:5]
            
        except Exception as e:
            console.print(f"[red]❌ Chain generation failed: {e}[/red]")
        
        # Fallback chain
        return [
            "nmap -sS -sV 10.10.10.10",
            "gobuster dir -u http://10.10.10.10 -w /usr/share/wordlists/common.txt",
            "searchsploit apache 2.4.49",
            "python3 exploit.py",
            "zip -r /tmp/data.zip /etc/passwd"
        ]

    def _analyze_memory_patterns(self, memory, agent_id):
        """
        Analyze memory patterns for an agent and provide optimization suggestions.
        Used by ShadowAgent for memory optimization.
        
        Args:
            memory (dict): Memory to analyze
            agent_id (str): Agent ID
            
        Returns:
            str: Optimization suggestions
        """
        actions = memory.get("actions", [])
        if not actions:
            return "No actions to analyze."
            
        # Extract key metrics for analysis
        command_counts = {}
        phase_transition = []
        rewards = []
        
        for action in actions:
            cmd = action.get("command", "")
            if cmd:
                command_counts[cmd] = command_counts.get(cmd, 0) + 1
            phase = action.get("phase")
            if phase:
                phase_transition.append(phase)
            reward = action.get("reward")
            if reward is not None:
                rewards.append(reward)
                
        # Find patterns
        repeated_commands = [cmd for cmd, count in command_counts.items() if count > 3]
        avg_reward = sum(rewards) / len(rewards if rewards else 0)
        
        # Generate insights using GPT
        prompt = f"""
Analyze these memory patterns for {agent_id}:
- Most repeated commands: {repeated_commands[:5]}
- Phase transitions: {phase_transition[-10:]}
- Average reward: {avg_reward:.2f}

Suggest memory optimization strategies in a single sentence.
"""
        
        try:
            response = self.gpt_manager.gpt_request(
                prompt=prompt,
                task_type="analysis",
                agent_id=self.agent_id,
                model="gpt-4o-mini"  # Use lightweight model for analysis
            )
            return response
        except Exception as e:
            return f"Memory analysis error: {e}"

    def sync_memory(self):
        """
        Sync memory with MemoryRouter for global insights.
        Implementation for MemorySyncInterface.
        """
        if self.memory_router:
            self.memory_router.save_memory(self.agent_id, self.memory)

    def display_status(self):
        """Display agent status in the terminal."""
        table = Table(title=f"👁️ {self.agent_id} Status")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Role", self.role)
        table.add_row("Strategy", self.global_strategy)
        table.add_row("Insights Count", str(len(self.memory.get("actions", []))))
        table.add_row("Current Chain", str(len(self.current_chain)))
        
        # Show current attack chain if available
        if self.current_chain:
            chain_table = Table(title="🔗 Current Attack Chain")
            chain_table.add_column("Step", style="cyan")
            chain_table.add_column("Command", style="yellow")
            
            for i, cmd in enumerate(self.current_chain):
                chain_table.add_row(str(i+1), cmd)
                
            console.print(Panel(chain_table, title="Attack Chain", border_style="cyan"))
        
        console.print(Panel(table, title=f"👁️ {self.agent_id} Overview", border_style="magenta"))
        
    def generate_hint(self):
        """
        Generate a strategic hint using LLMRouter.
        
        Returns:
            str: A helpful hint for the user
        """
        return self.llm_router.route_task("strategic", "Provide a strategic hint for advancing in current cyber phase")

    def query_tactical_gpt(self, prompt, complexity="standard"):
        """
        Use GPTManager for all LLM calls, with caching, fallback, and output sanitization.
        
        Args:
            prompt (str): Prompt to send to GPT
            complexity (str): Complexity level
            
        Returns:
            str: GPT response
        """
        return self.gpt_manager.gpt_request(prompt, task_type="reasoning", model="gpt-4o-mini")

    def save_models(self, prefix="models/orion_agent"):
        """
        Save models to disk.
        
        Args:
            prefix (str): Path prefix for saved models
        """
        os.makedirs(os.path.dirname(prefix), exist_ok=True)
        try:
            # Save current chain and insights
            with open(f"{prefix}_chain.json", "w") as f:
                json.dump(self.current_chain, f, indent=2)
            console.print(f"[green]💾 {self.agent_id}: Chain saved to {prefix}_chain.json[/green]")
        except Exception as e:
            console.print(f"[red]❌ {self.agent_id}: Chain save failed: {e}[/red]")

    def load_models(self, prefix="models/orion_agent"):
        """
        Load models from disk.
        
        Args:
            prefix (str): Path prefix for saved models
        """
        try:
            # Load saved chain
            if os.path.exists(f"{prefix}_chain.json"):
                with open(f"{prefix}_chain.json", "r") as f:
                    self.current_chain = json.load(f)
                console.print(f"[green]✓ {self.agent_id}: Chain loaded from {prefix}_chain.json[/green]")
        except Exception as e:
            console.print(f"[red]⚠ {self.agent_id}: Chain load failed: {e}[/red]")

    def reset(self):
        """Reset agent for new episode."""
        # Keep chain history but reset current episode tracking
        self.memory["actions"] = []
        self._log_training_event("Reset for new episode")

    def strategic_chain_planning(self, prompt):
        # Use LLMRouter for planning
        chain = self.llm_router.route_task("planner", prompt)
        return chain

    def analyze_agent_stuck(self, agent_id, reward_history):
        """
        Detect if agent is stuck (no reward improvement for N episodes).
        If so, call LLMRouter for advice and log it.
        """
        N = 5
        if len(reward_history) >= N and all(r <= reward_history[0] for r in reward_history[-N:]):
            advice = self.llm_router.route_task("strategic", f"Agent {agent_id} is stuck, suggest next steps.")
            # Log or print advice
            print(f"[Orion] Advice for {agent_id}: {advice}")
            return advice
        return None

    def adjust_agent_parameters(self, agent, dqn_insights):
        """
        Adjust agent parameters (e.g., epsilon) based on DQN insights.
        """
        if dqn_insights.get("increase_exploration"):
            agent.epsilon = min(1.0, agent.epsilon * 1.1)
        if dqn_insights.get("decrease_exploration"):
            agent.epsilon = max(agent.epsilon * 0.9, agent.epsilon_min)

    def generate_dynamic_scenario(self, scenario, services):
        """
        Generate a dynamic scenario profile for CyberEnvironment.
        Args:
            scenario (str): Scenario name or type (e.g., 'dynamic', 'ctf', etc.)
            services (list): List of available services (e.g., ['ssh', 'http', ...])
        Returns:
            dict: Scenario profile with keys: difficulty, traceback_threshold, training_mode, blue_aggressiveness, services
        """
        try:
            # Validate input
            if not isinstance(services, list) or not services:
                services = ["ssh", "http", "ftp", "smb"]
            # Example: Use scenario to influence difficulty
            if scenario == "ctf":
                difficulty = random.randint(10, 18)
                training_mode = "live"
            elif scenario == "simulated":
                difficulty = random.randint(5, 12)
                training_mode = "simulated"
            else:
                difficulty = random.randint(7, 16)
                training_mode = "adaptive"
            profile = {
                "difficulty": difficulty,
                "traceback_threshold": random.randint(60, 90),
                "training_mode": training_mode,
                "blue_aggressiveness": random.randint(2, 5),
                "services": services,
            }
            # Optionally, allow for future config-driven overrides here
            return profile
        except Exception as e:
            # Fallback defaults
            console.print(f"[yellow]⚠ OrionAgent.generate_dynamic_scenario failed: {e}[/yellow]")
            return {
                "difficulty": 10,
                "traceback_threshold": 75,
                "training_mode": "adaptive",
                "blue_aggressiveness": 3,
                "services": services if services else ["ssh", "http"],
            }

    def provide_reasoning(self, context, data):
        """
        Generate a strategic reasoning statement based on context and data.
        """
        # For now, return a stub or use LLM if available
        try:
            prompt = f"As OrionAgent, provide reasoning for context: {context} with data: {data}"
            if hasattr(self, 'gpt_manager'):
                return self.gpt_manager.gpt_request(prompt, task_type="strategic", model="gpt-4o-mini")
            return f"OrionAgent strategic reasoning for {context}: {data}"
        except Exception as e:
            return f"[OrionAgent] Reasoning unavailable: {e}"

# For CLI testing
if __name__ == "__main__":
    console.print("[bold blue]Testing OrionAgent in standalone mode[/bold blue]")
    
    orion = OrionAgent()
    orion.display_status()
    
    # Test GPT integration - should use GPTManager
    test_data = {
        "RedAgent": {
            "rewards": [5, 8, 10, -2, 15],
            "actions": ["nmap -sV 10.10.10.10", "gobuster dir -u http://10.10.10.10"],
            "phases": ["recon", "enumeration"]
        }
    }
    
    feedback = orion._generate_performance_feedback(test_data)