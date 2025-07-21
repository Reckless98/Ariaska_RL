# core/agents/orion_agent.py — ARIASKA OrionAgent v2.0 APEX STRATEGIST
# 👁️ Strategic Overseer | 🧠 Agent Optimization | ♻️ Training Adaptation | 🎮 Meta-Controller

import os
import random
import time
import torch
import numpy as np
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from core.interfaces.agent_interface import AgentInterface
from core.interfaces.memory_sync_interface import MemorySyncInterface
from core.utils.llm_orchestrator import LLMOrchestrator
from core.utils.replay_buffer import ReplayBuffer
from core.gpt_manager import GPTManager
from core.multiagent.memory_router import MemoryRouter
from core.multiagent.strategic_directive import (
    StrategicDirective,
    DirectiveType,
    directive_manager,
)

console = Console()
logger = logging.getLogger(__name__)


class OrionAgent(AgentInterface, MemorySyncInterface):
    """
    OrionAgent: Strategic overseer for the ARIASKA multi-agent system.

    Serves as hierarchical coordinator and meta-controller:
    - Provides strategic direction to other agents
    - Analyzes global performance and adapts agent parameters
    - Performs periodic reviews using GPT-4o models
    - Optimizes resource allocation and prevents redundant actions
    - Manages training curriculum progression
    """

    @property
    def agent_id(self):
        return self._agent_id

    def process_directive(self, directive_type: str, parameters: Dict[str, Any], 
                         priority: int = 1, source_agent: str = "system") -> Dict[str, Any]:
        """
        Process a directive received from another agent or the system.
        
        Args:
            directive_type: Type of directive to process
            parameters: Parameters for the directive
            priority: Priority level (1-5)
            source_agent: Agent ID that issued the directive
            
        Returns:
            Dict with processing results
        """
        try:
            # Log the received directive
            if self.verbosity not in ["quiet", "silent"]:
                console.print(f"[blue]👁️ OrionAgent received directive: {directive_type} from {source_agent}[/blue]")
            
            processing_result = {
                "status": "acknowledged",
                "directive_type": directive_type,
                "source_agent": source_agent,
                "action_taken": "processed"
            }
            
            # Process based on directive type
            directive_type_lower = directive_type.lower()
            
            if "strategic_review" in directive_type_lower:
                # Trigger immediate strategic review
                env_state = {}
                if self.red_agent and hasattr(self.red_agent, "env"):
                    env_state = self.red_agent.env.get_global_state()
                context = self._gather_context(env_state)
                review = self.perform_strategic_review(context)
                processing_result["action_taken"] = f"Performed strategic review: {review[:100]}..."
                
            elif "tactical_review" in directive_type_lower:
                # Trigger immediate tactical review
                env_state = {}
                if self.red_agent and hasattr(self.red_agent, "env"):
                    env_state = self.red_agent.env.get_global_state()
                context = self._gather_context(env_state)
                review = self.perform_tactical_review(context)
                processing_result["action_taken"] = f"Performed tactical review: {review[:100]}..."
                
            elif "adjust_parameters" in directive_type_lower:
                # Adjust agent parameters
                env_state = {}
                if self.red_agent and hasattr(self.red_agent, "env"):
                    env_state = self.red_agent.env.get_global_state()
                context = self._gather_context(env_state)
                adjustments = self.adjust_agent_parameters(context)
                processing_result["action_taken"] = f"Adjusted parameters: {adjustments}"
                
            elif "change_strategy" in directive_type_lower:
                # Change global strategy
                new_strategy = parameters.get("strategy", "balanced")
                if new_strategy in ["balanced", "aggressive", "stealth"]:
                    self.global_strategy = new_strategy
                    processing_result["action_taken"] = f"Changed global strategy to: {new_strategy}"
                else:
                    processing_result["action_taken"] = f"Invalid strategy requested: {new_strategy}"
                    
            elif "generate_chain" in directive_type_lower:
                # Generate new action chain
                env_state = {}
                if self.red_agent and hasattr(self.red_agent, "env"):
                    env_state = self.red_agent.env.get_global_state()
                context = self._gather_context(env_state)
                chain = self.generate_action_chain(context)
                self.current_chain = chain
                processing_result["action_taken"] = f"Generated action chain with {len(chain)} commands"
                
            else:
                processing_result["action_taken"] = f"Acknowledged unknown directive type: {directive_type}"
            
            # Log to memory router if available
            if self.memory_router and hasattr(self.memory_router, 'log_directive_processed'):
                directive_id = f"{source_agent}_{directive_type}_{priority}_{self.step_counter}"
                self.memory_router.log_directive_processed(
                    directive_id=directive_id,
                    result=processing_result["action_taken"]
                )
            
            return processing_result
            
        except Exception as e:
            console.print(f"[red]❌ Error processing directive: {e}[/red]")
            return {
                "status": "error",
                "directive_type": directive_type,
                "source_agent": source_agent,
                "error": str(e)
            }

    def provide_reasoning(self, context_type, context_data):
        """
        Generate strategic reasoning based on context.
        For now, return a placeholder string for simulation continuity.
        """
        return f"[Orion Insight] Context: {context_type} | Data: {context_data}"

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
        agent_id: str = "OrionAgent",
        role: str = "StrategicOverseer",
        agent_manager=None,
        memory_router=None,
        verbosity: str = "standard",
    ):
        self._agent_id = agent_id
        self._role = role
        self.agent_manager = agent_manager
        self.memory_router = memory_router or MemoryRouter()
        self.verbosity = verbosity

        # Initialize LLM systems
        self.llm_orchestrator = LLMOrchestrator(cache_dir="core/memory/llm_cache/orion_responses")
        self.gpt_manager = GPTManager()
        # All LLM functionality now handled by self.gpt_manager

        # Strategic parameters
        self.strategic_review_frequency = 10  # Steps between strategic reviews
        self.tactical_review_frequency = 5  # Steps between tactical reviews
        self.curriculum_advancement_threshold = 3.0  # Avg reward needed to advance
        self.global_strategy = (
            "balanced"  # Initial strategy (balanced, aggressive, stealth)
        )
        self.current_chain = []  # Current action chain/plan
        self.step_counter = 0
        self.episode_counter = 0

        # Agent subordinates registry
        self.subordinate_agents = {}

        # Agent parameter targets
        self.agent_parameter_targets = {
            "RedAgent": {
                "epsilon": 0.2,
                "entropy_beta": 0.01,
                "current_mode": "balanced",
            },
            "BlueAgent": {
                "epsilon": 0.2,
                "entropy_beta": 0.01,
                "current_mode": "balanced",
            },
            "ScoutAgent": {"stealth_mode": False, "noise_level": 5.0},
            "ShadowAgent": {"stealth_mode": False, "alert_threshold": 50.0},
        }

        # Memory for strategic analysis
        self.replay_buffer = ReplayBuffer(
            capacity=200,
            use_sqlite=True,
            db_path="core/memory/orionagent/replay_buffer.sqlite3",
        )

        # Performance tracking over episodes
        self.agent_performance = {
            agent_id: []
            for agent_id in ["RedAgent", "BlueAgent", "ScoutAgent", "ShadowAgent"]
        }
        self.episode_stats = []

        # Strategic directives
        self.strategic_directives = {
            "recon": [
                "Focus on thorough port scanning with stealth",
                "Identify all potential services and versions",
                "Document found ports and services methodically",
            ],
            "enumeration": [
                "Perform detailed service enumeration systematically",
                "Identify potential vulnerabilities in each service",
                "Use both automated and manual enumeration techniques",
            ],
            "exploit": [
                "Target high-probability vulnerabilities first",
                "Avoid triggering alerts through noisy exploitation",
                "Establish persistence after successful exploit",
            ],
            "privesc": [
                "Enumerate all permission settings and sudo rights",
                "Look for kernel exploits and SUID binaries",
                "Create reliable privilege escalation chain",
            ],
            "exfiltrate": [
                "Prioritize valuable data for exfiltration",
                "Use encrypted channels for data transfer",
                "Remove evidence of compromise during exfiltration",
            ],
        }

        console.print(f"[green]✓ {self.agent_id} initialized[/green]")
        
        # Add environment and stats monitor for main.py compatibility
        from core.environment.cyber_environment import CyberEnvironment
        from core.utils.stats_monitor import StatsMonitor
        self.env = CyberEnvironment(agent_manager=agent_manager, defer_reset=True) if agent_manager else None
        self.stats_monitor = StatsMonitor()

    def _init_multiagent_links(self):
        """Initialize links to other agents in the system."""
        if self.agent_manager:
            self.red_agent = self.agent_manager.get_agent("RedAgent")
            self.blue_agent = self.agent_manager.get_agent("BlueAgent")
            self.scout_agent = self.agent_manager.get_agent("ScoutAgent")
            self.shadow_agent = self.agent_manager.get_agent("ShadowAgent")

            # Register agents as subordinates
            self.register_subordinate(self.red_agent)
            self.register_subordinate(self.blue_agent)
            self.register_subordinate(self.scout_agent)
            self.register_subordinate(self.shadow_agent)

    def register_subordinate(self, agent):
        """Register an agent as a subordinate to Orion."""
        if agent and hasattr(agent, "agent_id"):
            self.subordinate_agents[agent.agent_id] = agent
            if self.verbosity not in ["quiet", "silent"]:
                console.print(
                    f"[blue]👁️ Orion registered subordinate: {agent.agent_id}[/blue]"
                )

    def issue_directive(
        self,
        directive_type: Union[DirectiveType, str],
        target_agent: str,
        parameters: Optional[Dict[str, Any]] = None,
        priority: int = 1,
    ) -> StrategicDirective:
        """
        Issue a strategic directive to a specific agent.

        Args:
            directive_type: Type of directive
            target_agent: Agent ID to receive the directive
            parameters: Additional parameters for the directive
            priority: Priority level (1-5)

        Returns:
            The created strategic directive
        """
        parameters = parameters or {}

        # Create directive
        directive = StrategicDirective(
            directive_type=directive_type,
            target_agent=target_agent,
            parameters=parameters,
            priority=priority,
            source_agent=self.agent_id,
        )

        # Add to directive manager
        if hasattr(directive_manager, 'issue_directive'):
            # DirectiveManager already handles the directive creation, so we store it directly
            directive_manager.directives[directive.id] = directive
            directive_manager._add_to_index(directive_manager.directives_by_target, directive.target_agent, directive)
            directive_manager._add_to_index(directive_manager.directives_by_source, directive.source_agent, directive)
            directive_manager._add_to_index(directive_manager.directives_by_type, directive.get_type_name(), directive)
            directive_manager._add_to_index(directive_manager.directives_by_status, directive.status.name, directive)
        else:
            # Fallback: store directive locally if manager doesn't support adding
            if not hasattr(self, '_local_directives'):
                self._local_directives = []
            self._local_directives.append(directive)

        if self.verbosity not in ["quiet", "silent"]:
            console.print(f"[blue]👁️ OrionAgent issued directive: {directive}[/blue]")

        # Log to memory router
        if self.memory_router and hasattr(self.memory_router, "log_directive"):
            self.memory_router.log_directive(
                source_agent=self.agent_id,
                target_agent=target_agent,
                directive_type=str(directive_type),
                parameters=parameters,
                priority=priority,
                step=self.step_counter,
                episode=self.episode_counter,
            )

        return directive

    def issue_global_directive(
        self,
        directive_type: Union[DirectiveType, str],
        parameters: Optional[Dict[str, Any]] = None,
        priority: int = 1,
    ) -> List[StrategicDirective]:
        """
        Issue a strategic directive to all active agents.

        Args:
            directive_type: Type of directive
            parameters: Additional parameters for the directive
            priority: Priority level (1-5)

        Returns:
            List of created strategic directives
        """
        directives = []

        if self.agent_manager:
            for agent in self.agent_manager.all_agents():
                if agent.agent_id != self.agent_id:  # Don't issue directive to self
                    directive = self.issue_directive(
                        directive_type=directive_type,
                        target_agent=agent.agent_id,
                        parameters=parameters,
                        priority=priority,
                    )
                    directives.append(directive)

        if self.verbosity not in ["quiet", "silent"]:
            console.print(
                f"[blue]👁️ OrionAgent issued global directive to {len(directives)} agents[/blue]"
            )

        return directives
    
    def get_smart_command(self, state: Dict[str, Any], phase: str) -> tuple[str, str]:
        """Generate strategic command for main.py compatibility."""
        try:
            # OrionAgent provides strategic oversight rather than direct commands
            # Analyze current state and provide high-level guidance
            open_ports = state.get("open_ports", [])
            privilege_level = state.get("privilege_level", "user")
            detection_risk = state.get("detection_risk", 0)
            
            if phase == "recon":
                if detection_risk > 40:
                    command = "nmap -sS -T2 10.10.10.10 -p 22,80,443"  # Slower scan
                    reason = "Strategic directive: Prioritize stealth over speed in recon"
                else:
                    command = "nmap -sS -sV 10.10.10.10"  # Version detection
                    reason = "Strategic directive: Gather comprehensive service information"
            elif phase == "exploit":
                if open_ports and 22 in open_ports:
                    command = "hydra -l admin -P rockyou.txt ssh://10.10.10.10"
                    reason = "Strategic directive: Target SSH service for credential attack"
                else:
                    command = "searchsploit $(nmap -sS 10.10.10.10 --script=version)"
                    reason = "Strategic directive: Search for known vulnerabilities"
            else:
                command = "whoami && id"  # Safe command for any phase
                reason = f"Strategic directive: Assess current capabilities in {phase} phase"
            
            return command, reason
        except Exception as e:
            # Strategic fallback
            return "echo 'Strategic analysis in progress'", f"Orion strategic fallback: {e}"

    def act(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Take strategic action based on current state.
        
        Args:
            state: Current environment state
            
        Returns:
            Dict with action, success, reward, and info
        """
        # Orion agent focuses on strategic oversight and coordination
        mission_progress = state.get("mission_progress", 0)
        agent_coordination = state.get("agent_coordination", 50)
        strategic_value = state.get("strategic_value", 30)
        
        if mission_progress < 25:
            # Early mission phase - strategic planning
            action = "echo 'Analyzing target infrastructure and planning approach'"
            success = True
            reward = 40  # High reward for strategic planning
            info = {"action_type": "strategic_planning", "progress": mission_progress}
        elif agent_coordination < 60:
            # Need better coordination
            action = "echo 'Coordinating multi-agent operations for optimal efficiency'"
            success = True
            reward = 35  # Reward for coordination
            info = {"action_type": "coordination", "coordination_level": agent_coordination}
        elif strategic_value > 70:
            # High value target identified
            action = "echo 'High-value target identified - directing focused operations'"
            success = True
            reward = 50  # High reward for strategic insight
            info = {"action_type": "target_prioritization", "value": strategic_value}
        else:
            # Strategic oversight
            action = "echo 'Monitoring overall mission status and adjusting strategy'"
            success = True
            reward = 30  # Standard oversight reward
            info = {"action_type": "strategic_oversight", "progress": mission_progress}
        
        return {
            "action": action,
            "success": success,
            "reward": reward,
            "info": info
        }

    def simulate_step(
        self, episode: int = 1, step: int = 1, shared_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Simulate a step for this agent in the training loop.

        Args:
            episode: Current episode number
            step: Current step number
            shared_context: Shared context from agent manager

        Returns:
            Dict with step results
        """
        # Update counters
        self.step_counter = step
        self.episode_counter = episode

        # Get current environment state
        env_state = {}
        if self.red_agent and hasattr(self.red_agent, "env"):
            env_state = self.red_agent.env.get_global_state()
        elif shared_context:
            env_state = shared_context

        # Determine if strategic review is needed
        perform_strategic_review = step % self.strategic_review_frequency == 0
        perform_tactical_review = step % self.tactical_review_frequency == 0

        # Initialize results
        results = {
            "agent_id": self.agent_id,
            "step": step,
            "episode": episode,
            "strategic_review": False,
            "tactical_review": False,
            "directives": [],
            "parameter_adjustments": {},
        }

        # Gather context for decision making
        context = self._gather_context(env_state)

        # Strategic Review (less frequent, more comprehensive)
        if perform_strategic_review:
            strategic_review = self.perform_strategic_review(context)
            results["strategic_review"] = True
            results["strategic_insights"] = strategic_review

            # Check for curriculum advancement
            self._check_curriculum_advancement(context)

            # Generate action chain/plan for next steps
            chain = self.generate_action_chain(context)
            if chain:
                self.current_chain = chain
                results["action_chain"] = chain

            # Apply strategic parameter adjustments
            param_adjustments = self.adjust_agent_parameters(context)
            results["parameter_adjustments"] = param_adjustments

            # Log to replay buffer
            experience = {
                "state": env_state,
                "action": "strategic_review",
                "outcome": strategic_review,
                "parameters": param_adjustments,
            }
            self.replay_buffer.add(experience)

        # Tactical Review (more frequent, focused on current phase/situation)
        elif perform_tactical_review:
            tactical_review = self.perform_tactical_review(context)
            results["tactical_review"] = True
            results["tactical_insights"] = tactical_review

            # Provide phase-specific directives
            current_phase = env_state.get("phase", "recon")
            directives = self.get_phase_directives(current_phase)
            results["directives"] = directives

            # Log to replay buffer
            experience = {
                "state": env_state,
                "action": "tactical_review",
                "outcome": tactical_review,
                "directives": directives,
            }
            self.replay_buffer.add(experience)

        # Normal step - provide oversight
        else:
            # Check for crisis intervention
            crisis = self._check_for_crisis(context)
            if crisis:
                intervention = self._crisis_intervention(context, crisis)
                results["crisis_intervention"] = intervention

            # Issue routine directives based on current phase
            # This ensures agents always have some guidance even between reviews
            self._issue_routine_directives(env_state)

        # Always provide the current strategic directives
        current_phase = env_state.get("phase", "recon")
        results["phase_directives"] = self.get_phase_directives(current_phase)

        # Log step to memory router
        if self.memory_router:
            oversight_data = {
                "strategic_review": results.get("strategic_review", False),
                "tactical_review": results.get("tactical_review", False),
                "directives": results.get("directives", []),
                "parameter_adjustments": results.get("parameter_adjustments", {}),
                "step": step,
                "episode": episode,
                "agent_id": self.agent_id,
                "global_strategy": self.global_strategy
            }
            self.memory_router.log_orion_oversight(oversight_data)

        # Display information based on verbosity
        if self.verbosity in ("standard", "verbose", "debug"):
            if perform_strategic_review:
                console.print(
                    f"[blue]👁️ OrionAgent performed strategic review at step {step}[/blue]"
                )
            elif perform_tactical_review:
                console.print(
                    f"[cyan]🔍 OrionAgent performed tactical review at step {step}[/cyan]"
                )

        # Adaptive visualization for different verbosity levels
        if self.verbosity == "verbose" or self.verbosity == "debug":
            # Create a table with detailed info
            table = Table(title=f"👁️ OrionAgent Step {step}")
            table.add_column("Parameter", style="cyan")
            table.add_column("Value", style="green")

            # Add step info
            table.add_row("Episode", str(episode))
            table.add_row("Step", str(step))
            table.add_row("Current Phase", env_state.get("phase", "unknown"))

            # Add review info if performed
            if perform_strategic_review:
                table.add_row("Strategic Review", "YES")
                insights = results.get("strategic_insights", "")
                table.add_row(
                    "Insights",
                    str(insights)[:100] + ("..." if len(str(insights)) > 100 else ""),
                )
            elif perform_tactical_review:
                table.add_row("Tactical Review", "YES")
                insights = results.get("tactical_insights", "")
                table.add_row(
                    "Insights",
                    str(insights)[:100] + ("..." if len(str(insights)) > 100 else ""),
                )

            console.print(table)

        # Add active directives to results
        directive_stats = directive_manager.get_directive_stats()
        results["active_directives"] = directive_stats["active_count"]
        results["directives_by_agent"] = directive_stats["by_agent"]

        return results

    def _gather_context(self, env_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gather context from all available sources for decision making.

        Args:
            env_state: Current environment state

        Returns:
            Dict with comprehensive context
        """
        context = {"env_state": env_state}

        # Add agent statuses
        agent_statuses = {}
        for agent_id, agent in self.subordinate_agents.items():
            status = {}

            # Get common attributes
            if hasattr(agent, "current_mode"):
                status["mode"] = agent.current_mode
            if hasattr(agent, "epsilon"):
                status["epsilon"] = agent.epsilon
            if hasattr(agent, "entropy_beta"):
                status["entropy_beta"] = agent.entropy_beta
            if hasattr(agent, "command_history") and agent.command_history:
                status["last_command"] = (
                    agent.command_history[-1] if agent.command_history else None
                )
            if hasattr(agent, "stats_monitor"):
                status["avg_reward"] = (
                    agent.stats_monitor.get_average_reward()
                    if hasattr(agent.stats_monitor, "get_average_reward")
                    else 0.0
                )

            # Get specific attributes
            if agent_id == "ScoutAgent":
                if hasattr(agent, "stealth_mode"):
                    status["stealth_mode"] = agent.stealth_mode
                if hasattr(agent, "noise_level"):
                    status["noise_level"] = agent.noise_level
            elif agent_id == "ShadowAgent":
                if hasattr(agent, "current_alert_score"):
                    status["alert_score"] = agent.current_alert_score
                if hasattr(agent, "stealth_mode"):
                    status["stealth_mode"] = agent.stealth_mode

            agent_statuses[agent_id] = status

        context["agent_statuses"] = agent_statuses

        # Add memory router insights if available
        if self.memory_router and hasattr(self.memory_router, "get_global_insights"):
            try:
                context["global_insights"] = self.memory_router.get_global_insights()
            except (AttributeError, Exception):
                context["global_insights"] = {}
        else:
            context["global_insights"] = {}

        # Add current action chain
        context["current_chain"] = {
            "commands": self.current_chain,
            "length": len(self.current_chain),
            "phase": env_state.get("phase", "recon")
        }

        return context

    def perform_strategic_review(self, context: Dict[str, Any]) -> str:
        """
        Perform a comprehensive strategic review using LLM.

        Args:
            context: Comprehensive context about environment and agents

        Returns:
            String with strategic insights
        """
        env_state = context.get("env_state", {})
        agent_statuses = context.get("agent_statuses", {})
        current_phase = env_state.get("phase", "recon")

        # Format context for LLM
        formatted_context = f"Environment: Phase {current_phase}, "
        formatted_context += f"Blue team alert: {env_state.get('blue_team_alert', 0)}, "
        formatted_context += f"Detection risk: {env_state.get('detection_risk', 0)}, "
        formatted_context += (
            f"Privilege level: {env_state.get('privilege_level', 'none')}\n"
        )

        # Add agent information
        formatted_context += "Agent statuses:\n"
        for agent_id, status in agent_statuses.items():
            formatted_context += f"- {agent_id}: Mode {status.get('mode', 'N/A')}, "
            if "avg_reward" in status:
                formatted_context += f"Avg reward: {status['avg_reward']:.2f}, "
            if "last_command" in status:
                formatted_context += f"Last action: {status['last_command']}\n"

        prompt = f"""
        You are OrionAgent, the strategic overseer of the ARIASKA multi-agent cybersecurity system.
        
        Current context:
        {formatted_context}
        
        Perform a comprehensive strategic review addressing:
        1. Overall mission progress and phase appropriateness
        2. Coordination between RedAgent, BlueAgent, ScoutAgent, and ShadowAgent
        3. Strategic suggestions for improving performance
        
        Provide no more than 3 key insights and recommendations in bullet point form.
        """

        # Use LLM orchestrator for strategic insight
        try:
            response = self.llm_orchestrator.route_task(
                task_type="strategic", prompt=prompt, agent_id=self.agent_id
            )
            # Extract response text if it's a dict
            if isinstance(response, dict):
                response = response.get("response", str(response))
        except Exception as e:
            console.print(
                f"[yellow]⚠ Error in strategic review: {e}. Using GPT fallback.[/yellow]"
            )
            response = self.gpt_manager.gpt_request(
                prompt, "gpt-4o-mini", agent_id=self.agent_id
            )
            # Ensure response is a string
            if not isinstance(response, str):
                response = str(response)

        # Store strategic review in memory for future reference
        if self.memory_router and hasattr(self.memory_router, "log_strategic_review"):
            self.memory_router.log_strategic_review({
                "agent_id": self.agent_id,
                "phase": current_phase,
                "context": formatted_context,
                "insights": response,
                "episode": self.episode_counter,
                "step": self.step_counter,
            })

        # Issue strategic directives based on review
        self._issue_directives_from_review(response, env_state)

        return response

    def _issue_directives_from_review(
        self, review: str, env_state: Dict[str, Any]
    ) -> None:
        """
        Issue strategic directives based on the strategic review.

        Args:
            review: Strategic review text
            env_state: Current environment state
        """
        # Extract key signals from review text
        review_lower = review.lower()

        # Check for stealth signals
        if any(
            term in review_lower
            for term in ["stealth", "detection", "quiet", "careful", "avoid alerting"]
        ):
            self.issue_global_directive(
                directive_type=DirectiveType.INCREASE_STEALTH,
                parameters={"reason": "Strategic review indicates need for stealth"},
                priority=(
                    4
                    if "high alert" in review_lower or "detection risk" in review_lower
                    else 3
                ),
            )

        # Check for aggressive signals
        elif any(
            term in review_lower
            for term in ["aggressive", "faster", "speed", "accelerate", "quickly"]
        ):
            self.issue_global_directive(
                directive_type=DirectiveType.DECREASE_STEALTH,
                parameters={"reason": "Strategic review indicates need for aggression"},
                priority=3,
            )

        # Check for target focus
        if "focus" in review_lower and any(
            service in review_lower
            for service in ["http", "ssh", "ftp", "smb", "database"]
        ):
            # Extract service to focus on
            services = ["http", "ssh", "ftp", "smb", "database", "web", "mysql", "rdp"]
            focus_services = [
                service for service in services if service in review_lower
            ]

            if focus_services:
                self.issue_directive(
                    directive_type=DirectiveType.FOCUS_TARGET,
                    target_agent="RedAgent",
                    parameters={"service_focus": focus_services[0]},
                    priority=3,
                )

                self.issue_directive(
                    directive_type=DirectiveType.FOCUS_TARGET,
                    target_agent="ScoutAgent",
                    parameters={"service_focus": focus_services[0]},
                    priority=3,
                )

        # Check for phase change suggestions
        phases = ["recon", "enumeration", "exploit", "privesc", "exfiltrate"]
        current_phase = env_state.get("phase", "recon")

        for phase in phases:
            if (
                phase != current_phase
                and f"move to {phase}" in review_lower
                or f"switch to {phase}" in review_lower
            ):
                self.issue_directive(
                    directive_type=DirectiveType.CHANGE_STRATEGY,
                    target_agent="RedAgent",
                    parameters={"new_phase": phase},
                    priority=4,
                )
                break

        # Issue coordination directives when coordination is mentioned
        if any(
            term in review_lower
            for term in ["coordinate", "coordination", "together", "synchronize"]
        ):
            self.issue_global_directive(
                directive_type=DirectiveType.CHANGE_STRATEGY,
                parameters={"coordinate_on": current_phase},
                priority=2,
            )

    def perform_tactical_review(self, context: Dict[str, Any]) -> str:
        """
        Perform a focused tactical review for current phase/situation.

        Args:
            context: Comprehensive context about environment and agents

        Returns:
            String with tactical insights
        """
        env_state = context.get("env_state", {})
        current_phase = env_state.get("phase", "recon")

        # Format context for LLM
        formatted_context = f"Current phase: {current_phase}\n"
        formatted_context += f"Open ports: {env_state.get('open_ports', [])}\n"
        formatted_context += f"Services: {env_state.get('services', [])}\n"
        formatted_context += f"Blue team alert: {env_state.get('blue_team_alert', 0)}\n"
        formatted_context += f"Detection risk: {env_state.get('detection_risk', 0)}\n"

        # Get recent actions
        recent_actions = []
        for agent_id, agent in self.subordinate_agents.items():
            if hasattr(agent, "command_history") and agent.command_history:
                recent_actions.append(f"{agent_id}: {agent.command_history[-1]}")

        if recent_actions:
            formatted_context += "Recent actions:\n" + "\n".join(recent_actions) + "\n"

        # Use GPT-4o-mini for concise tactical advice first for efficiency
        prompt = f"""
        As a tactical advisor for phase '{current_phase}', provide concise guidance.
        
        Context:
        {formatted_context}
        
        Provide 1-2 specific tactical recommendations for immediate execution.
        Keep it brief and actionable.
        """

        try:
            # Use GPTManager for tactical recommendations
            gpt_response = self.gpt_manager.smart_decision(
                task_type="tactical", 
                task_description=prompt
            )

            # Ensure gpt_response is a string
            if isinstance(gpt_response, dict):
                gpt_response = str(gpt_response.get("response", gpt_response.get("content", str(gpt_response))))

            # Review and enhance with GPT if needed
            review_prompt = f"""
            Review this tactical advice for phase '{current_phase}':
            
            "{gpt_response}"
            
            Context:
            {formatted_context}
            
            Is this advice good? If yes, repeat it. If not, provide 1-2 better tactical recommendations.
            Keep it concise and actionable.
            """

            response = self.gpt_manager.gpt_request(
                review_prompt,
                model="gpt-4o-mini",
                task_type="tactical",
                agent_id=self.agent_id,
            )
        except Exception as e:
            console.print(
                f"[yellow]⚠ Error in tactical review: {e}. Using GPT fallback.[/yellow]"
            )
            response = self.gpt_manager.gpt_request(
                prompt,
                model="gpt-4o-mini",
                task_type="tactical",
                agent_id=self.agent_id,
            )

        # Store tactical review in memory for future reference
        if self.memory_router and hasattr(self.memory_router, "log_tactical_review"):
            self.memory_router.log_tactical_review({
                "agent_id": self.agent_id,
                "phase": current_phase,
                "context": formatted_context,
                "insights": response,
                "episode": self.episode_counter,
                "step": self.step_counter,
            })
        elif self.memory_router and hasattr(self.memory_router, "log_action"):
            # Fallback to generic log method if tactical review method doesn't exist
            self.memory_router.log_action({
                "agent_id": self.agent_id,
                "action_type": "tactical_review",
                "action_data": {
                    "phase": current_phase,
                    "context": formatted_context,
                    "insights": response,
                    "episode": self.episode_counter,
                    "step": self.step_counter,
                }
            })
        elif self.memory_router and hasattr(self.memory_router, "store_memory"):
            # Another fallback option
            self.memory_router.store_memory({
                "agent_id": self.agent_id,
                "memory_type": "tactical_review",
                "content": {
                    "phase": current_phase,
                    "context": formatted_context,
                    "insights": response,
                    "episode": self.episode_counter,
                    "step": self.step_counter,
                }
            })

        return response

    def get_phase_directives(self, phase: str) -> List[str]:
        """
        Get strategic directives for the current phase.

        Args:
            phase: Current mission phase

        Returns:
            List of directive strings
        """
        if (phase in self.strategic_directives):
            return self.strategic_directives[phase]
        else:
            return [
                "Analyze current state and determine appropriate action",
                "Document all findings systematically",
                "Proceed with caution to avoid detection",
            ]

    def generate_action_chain(self, context: Dict[str, Any]) -> List[str]:
        """
        Generate a sequence of actions (chain) for achieving goals in current phase.

        Args:
            context: Comprehensive context about environment and agents

        Returns:
            List of ordered actions forming a chain/plan
        """
        env_state = context.get("env_state", {})
        current_phase = env_state.get("phase", "recon")

        # Create prompt for action chain generation
        formatted_context = f"Current phase: {current_phase}\n"
        formatted_context += f"Open ports: {env_state.get('open_ports', [])}\n"
        formatted_context += f"Services: {env_state.get('services', [])}\n"
        formatted_context += (
            f"Privilege level: {env_state.get('privilege_level', 'none')}\n"
        )
        formatted_context += f"Blue team alert: {env_state.get('blue_team_alert', 0)}\n"
        formatted_context += f"Detection risk: {env_state.get('detection_risk', 0)}\n"

        prompt = f"""
        Generate an optimal sequence of 3-5 cybersecurity commands for phase '{current_phase}'.
        
        Context:
        {formatted_context}
        
        The commands should:
        1. Progressively build upon each other
        2. Be realistic Linux/cybersecurity commands
        3. Maximize information gain while minimizing detection
        
        Output the commands as a JSON array:
        ["command1", "command2", "command3", ...]
        """

        try:
            response = self.gpt_manager.gpt_request(
                prompt, model="gpt-4o-mini", task_type="chains", agent_id=self.agent_id
            )

            # Parse JSON response
            try:
                chain = json.loads(response)
                if isinstance(chain, list) and all(
                    isinstance(cmd, str) for cmd in chain
                ):
                    # Log the chain
                    if self.memory_router and hasattr(
                        self.memory_router, "log_action_chain"
                    ):
                        self.memory_router.log_action_chain(
                            agent_id=self.agent_id,
                            phase=current_phase,
                            chain=chain,
                            episode=self.episode_counter,
                            step=self.step_counter,
                        )

                    # Issue action chain directive to RedAgent
                    self.issue_directive(
                        directive_type=DirectiveType.CHANGE_STRATEGY,
                        target_agent="RedAgent",
                        parameters={"action_chain": chain, "phase": current_phase},
                        priority=3,
                    )

                    if self.verbosity not in ["quiet", "silent"]:
                        console.print(
                            f"[blue]👁️ OrionAgent generated action chain for phase '{current_phase}':[/blue]"
                        )
                        for i, cmd in enumerate(chain, 1):
                            console.print(f"[cyan]  {i}. {cmd}[/cyan]")

                    return chain
                else:
                    console.print(
                        "[yellow]⚠ Invalid action chain format from GPT. Using default.[/yellow]"
                    )
            except json.JSONDecodeError:
                console.print(
                    "[yellow]⚠ Failed to parse action chain JSON. Using default.[/yellow]"
                )
        except Exception as e:
            console.print(
                f"[yellow]⚠ Error generating action chain: {e}. Using default.[/yellow]"
            )

        # Fallback: Return phase-appropriate default chain
        default_chains = {
            "recon": [
                "nmap -sS -T2 -p- TARGET_IP",
                "nmap -sV -p OPEN_PORTS TARGET_IP",
                "nmap -A -p OPEN_PORTS TARGET_IP",
            ],
            "enumeration": [
                "gobuster dir -u http://TARGET_IP -w /usr/share/wordlists/dirbuster/directory-list-2.3-medium.txt",
                "enum4linux -a TARGET_IP",
                "nikto -h TARGET_IP",
            ],
            "exploit": [
                "searchsploit SERVICE_NAME VERSION",
                "msfconsole -x 'use exploit/SERVICE_PATH; set RHOSTS TARGET_IP; exploit'",
                "hydra -L /usr/share/wordlists/user.txt -P /usr/share/wordlists/rockyou.txt ssh://TARGET_IP",
            ],
            "privesc": ["find / -perm -u=s -type f 2>/dev/null", "sudo -l", "uname -a"],
            "exfiltrate": [
                "zip -r /tmp/data.zip /path/to/data",
                "python3 -m http.server",
                "nc -lvp 4444 > data.zip",
            ],
        }

        return default_chains.get(
            current_phase,
            ["nmap -sS TARGET", "gobuster dir -u http://TARGET", "enum4linux TARGET"],
        )

    def adjust_agent_parameters(
        self, context: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Adjust parameters for all agents based on performance and context.

        Args:
            context: Comprehensive context about environment and agents

        Returns:
            Dict with parameter adjustments per agent
        """
        env_state = context.get("env_state", {})
        agent_statuses = context.get("agent_statuses", {})

        # Initialize parameter adjustments
        param_adjustments = {}

        # Environmental factors that influence adjustments
        blue_alert = env_state.get("blue_team_alert", 0)
        detection_risk = env_state.get("detection_risk", 0)
        current_phase = env_state.get("phase", "recon")

        # Determine global strategy based on environmental factors
        if blue_alert > 70 or detection_risk > 7:
            self.global_strategy = "stealth"
            # Issue high-priority stealth directive
            self.issue_global_directive(
                directive_type=DirectiveType.INCREASE_STEALTH,
                parameters={
                    "reason": "High alert or detection risk",
                    "level": "maximum",
                },
                priority=5,
            )
        elif current_phase in ["exploit", "privesc", "exfiltrate"]:
            self.global_strategy = "aggressive"
            # Issue standard priority aggressive directive
            self.issue_global_directive(
                directive_type=DirectiveType.DECREASE_STEALTH,
                parameters={
                    "reason": f"Critical phase: {current_phase}",
                    "level": "standard",
                },
                priority=3,
            )
        else:
            self.global_strategy = "balanced"

        # Adjust RedAgent parameters
        if "RedAgent" in self.subordinate_agents:
            red_agent = self.subordinate_agents["RedAgent"]
            red_adjustments = {}

            # Adjust epsilon based on progress and strategy
            if hasattr(red_agent, "epsilon"):
                current_epsilon = red_agent.epsilon

                # Target epsilon based on global strategy
                if self.global_strategy == "stealth":
                    target_epsilon = 0.1  # Less exploration in stealth mode
                elif self.global_strategy == "aggressive":
                    target_epsilon = 0.3  # More exploration in aggressive mode
                else:
                    target_epsilon = 0.2  # Balanced approach

                # Calculate adjustment (gradual changes)
                epsilon_delta = (target_epsilon - current_epsilon) * 0.2
                if abs(epsilon_delta) > 0.001:  # Only adjust if change is significant
                    new_epsilon = current_epsilon + epsilon_delta
                    if hasattr(red_agent, "epsilon_min"):
                        new_epsilon = max(new_epsilon, red_agent.epsilon_min)
                    red_adjustments["epsilon"] = new_epsilon

            # Adjust entropy_beta
            if hasattr(red_agent, "entropy_beta"):
                current_beta = red_agent.entropy_beta

                # Target beta based on global strategy
                if self.global_strategy == "stealth":
                    target_beta = 0.005  # Less entropy in stealth mode
                elif self.global_strategy == "aggressive":
                    target_beta = 0.02  # More entropy in aggressive mode
                else:
                    target_beta = 0.01  # Balanced approach

                # Calculate adjustment (gradual changes)
                beta_delta = (target_beta - current_beta) * 0.2
                if abs(beta_delta) > 0.0001:  # Only adjust if change is significant
                    new_beta = current_beta + beta_delta
                    red_adjustments["entropy_beta"] = new_beta

            # Set current mode
            red_adjustments["current_mode"] = self.global_strategy

            # Apply adjustments if there are any
            if red_adjustments:
                param_adjustments["RedAgent"] = red_adjustments

                # Actually apply the changes to the agent
                for param, value in red_adjustments.items():
                    if hasattr(red_agent, param):
                        setattr(red_agent, param, value)
                        if self.verbosity not in ["quiet", "silent"]:
                            console.print(
                                f"[blue]👁️ OrionAgent adjusted RedAgent.{param} = {value}[/blue]"
                            )

        # Adjust BlueAgent parameters
        if "BlueAgent" in self.subordinate_agents:
            blue_agent = self.subordinate_agents["BlueAgent"]
            blue_adjustments = {}

            # Adjust epsilon based on red team progress and alert level
            if hasattr(blue_agent, "epsilon"):
                current_epsilon = blue_agent.epsilon

                # Target epsilon based on environmental factors
                if blue_alert > 70:
                    target_epsilon = 0.1  # More focused defense when alerts are high
                elif detection_risk > 7:
                    target_epsilon = 0.15  # Slightly more focused when risk is high
                else:
                    target_epsilon = 0.2  # Normal exploration

                # Calculate adjustment
                epsilon_delta = (target_epsilon - current_epsilon) * 0.2
                if abs(epsilon_delta) > 0.001:
                    new_epsilon = current_epsilon + epsilon_delta
                    if hasattr(blue_agent, "epsilon_min"):
                        new_epsilon = max(new_epsilon, blue_agent.epsilon_min)
                    blue_adjustments["epsilon"] = new_epsilon

            # Set defensive mode based on alert level
            if blue_alert > 60:
                blue_adjustments["current_mode"] = "Defensive"
            elif blue_alert > 30:
                blue_adjustments["current_mode"] = "Alert"
            else:
                blue_adjustments["current_mode"] = "Standard"

            # Apply adjustments if there are any
            if blue_adjustments:
                param_adjustments["BlueAgent"] = blue_adjustments

                # Actually apply the changes to the agent
                for param, value in blue_adjustments.items():
                    if hasattr(blue_agent, param):
                        setattr(blue_agent, param, value)
                        if self.verbosity not in ["quiet", "silent"]:
                            console.print(
                                f"[blue]👁️ OrionAgent adjusted BlueAgent.{param} = {value}[/blue]"
                            )

        # Adjust ScoutAgent parameters
        if "ScoutAgent" in self.subordinate_agents:
            scout_agent = self.subordinate_agents["ScoutAgent"]
            scout_adjustments = {}

            # Adjust stealth mode based on global strategy
            if hasattr(scout_agent, "stealth_mode"):
                stealth_mode = self.global_strategy == "stealth"
                scout_adjustments["stealth_mode"] = stealth_mode

            # Adjust noise level based on strategy and alert level
            if hasattr(scout_agent, "noise_level"):
                if self.global_strategy == "stealth":
                    noise_level = 2.0  # Quiet scanning
                elif self.global_strategy == "aggressive":
                    noise_level = 7.0  # Faster scanning
                else:
                    noise_level = 5.0  # Balanced

                # Further reduce noise if alert level is high
                if blue_alert > 60:
                    noise_level = max(1.0, noise_level - 2.0)

                scout_adjustments["noise_level"] = noise_level

            # Apply adjustments if there are any
            if scout_adjustments:
                param_adjustments["ScoutAgent"] = scout_adjustments

                # Actually apply the changes to the agent
                for param, value in scout_adjustments.items():
                    if hasattr(scout_agent, param):
                        setattr(scout_agent, param, value)
                        if self.verbosity not in ["quiet", "silent"]:
                            console.print(
                                f"[blue]👁️ OrionAgent adjusted ScoutAgent.{param} = {value}[/blue]"
                            )

        # Adjust ShadowAgent parameters
        if "ShadowAgent" in self.subordinate_agents:
            shadow_agent = self.subordinate_agents["ShadowAgent"]
            shadow_adjustments = {}

            # Adjust stealth mode based on global strategy
            if hasattr(shadow_agent, "stealth_mode"):
                stealth_mode = self.global_strategy == "stealth"
                shadow_adjustments["stealth_mode"] = stealth_mode

            # Adjust alert threshold based on strategy
            if hasattr(shadow_agent, "alert_threshold"):
                if self.global_strategy == "stealth":
                    alert_threshold = 30.0  # Lower threshold for stealth mode
                elif self.global_strategy == "aggressive":
                    alert_threshold = 70.0  # Higher threshold for aggressive mode
                else:
                    alert_threshold = 50.0  # Balanced

                shadow_adjustments["alert_threshold"] = alert_threshold

            # Apply adjustments if there are any
            if shadow_adjustments:
                param_adjustments["ShadowAgent"] = shadow_adjustments

                # Actually apply the changes to the agent
                for param, value in shadow_adjustments.items():
                    if hasattr(shadow_agent, param):
                        setattr(shadow_agent, param, value)
                        if self.verbosity not in ["quiet", "silent"]:
                            console.print(
                                f"[blue]👁️ OrionAgent adjusted ShadowAgent.{param} = {value}[/blue]"
                            )

        return param_adjustments

    def _check_curriculum_advancement(self, context: Dict[str, Any]) -> bool:
        """
        Check if curriculum should be advanced based on agent performance.

        Args:
            context: Comprehensive context

        Returns:
            True if curriculum was advanced, False otherwise
        """
        # Get average rewards
        agent_statuses = context.get("agent_statuses", {})
        red_reward = agent_statuses.get("RedAgent", {}).get("avg_reward", 0.0)

        # Check if curriculum advancement threshold is met
        if red_reward > self.curriculum_advancement_threshold:
            # Get environment from RedAgent
            if self.red_agent and hasattr(self.red_agent, "env"):
                env = self.red_agent.env

                # Advance difficulty
                if hasattr(env, "difficulty_level"):
                    current_difficulty = env.difficulty_level
                    new_difficulty = current_difficulty + 1

                    # Apply new difficulty
                    env.difficulty_level = new_difficulty

                    if self.verbosity not in ["quiet", "silent"]:
                        console.print(
                            f"[green bold]🚀 OrionAgent advanced curriculum difficulty: {current_difficulty} → {new_difficulty}[/green bold]"
                        )

                    # Apply domain randomization if available
                    if hasattr(env, "context_detector") and hasattr(
                        env.context_detector, "randomize_domain"
                    ):
                        try:
                            randomized_params = env.context_detector.randomize_domain()
                            console.print(
                                f"[green]🎲 Domain randomized: {len(randomized_params.get('ports', []))} ports, {len(randomized_params.get('services', []))} services[/green]"
                            )
                        except Exception as e:
                            console.print(
                                f"[yellow]⚠ Domain randomization failed: {e}[/yellow]"
                            )

                    return True

        return False

    def _check_for_crisis(self, context: Dict[str, Any]) -> Optional[str]:
        """
        Check if there's a crisis situation requiring intervention.

        Args:
            context: Comprehensive context

        Returns:
            Crisis type if detected, None otherwise
        """
        env_state = context.get("env_state", {})
        agent_statuses = context.get("agent_statuses", {})

        # Check for high alert level
        if env_state.get("blue_team_alert", 0) > 85:
            return "high_alert"

        # Check for agent stuck in repetitive actions
        red_status = agent_statuses.get("RedAgent", {})
        if (
            hasattr(self.red_agent, "repeat_steps")
            and getattr(self.red_agent, "repeat_steps", 0) > 3
        ):
            return "agent_stuck"

        # Check for lack of progress
        if hasattr(self.red_agent, "stats_monitor"):
            rewards = self.red_agent.stats_monitor.agent_stats.get("RedAgent", {}).get(
                "rewards", []
            )
            if len(rewards) >= 5 and max(rewards[-5:]) - min(rewards[-5:]) < 0.001:
                return "no_progress"

        return None

    def _crisis_intervention(
        self, context: Dict[str, Any], crisis_type: str
    ) -> Dict[str, Any]:
        """
        Perform crisis intervention when critical issues are detected.

        Args:
            context: Comprehensive context
            crisis_type: Type of crisis detected

        Returns:
            Dict with intervention details
        """
        intervention = {"type": crisis_type, "actions": []}

        if crisis_type == "high_alert":
            # Intervention for high alert: Switch to stealth mode
            self.global_strategy = "stealth"

            # Apply emergency stealth parameters to all agents
            if "RedAgent" in self.subordinate_agents:
                red_agent = self.subordinate_agents["RedAgent"]
                if hasattr(red_agent, "epsilon"):
                    red_agent.epsilon = 0.05  # Minimal exploration
                if hasattr(red_agent, "current_mode"):
                    red_agent.current_mode = "stealth"
                intervention["actions"].append("Set RedAgent to stealth mode")

            if "ScoutAgent" in self.subordinate_agents:
                scout_agent = self.subordinate_agents["ScoutAgent"]
                if hasattr(scout_agent, "stealth_mode"):
                    scout_agent.stealth_mode = True
                if hasattr(scout_agent, "noise_level"):
                    scout_agent.noise_level = 1.0  # Minimal noise
                intervention["actions"].append("Set ScoutAgent to maximum stealth")

            if "ShadowAgent" in self.subordinate_agents:
                shadow_agent = self.subordinate_agents["ShadowAgent"]
                if hasattr(shadow_agent, "stealth_mode"):
                    shadow_agent.stealth_mode = True
                if hasattr(shadow_agent, "alert_threshold"):
                    shadow_agent.alert_threshold = 20.0  # Very sensitive alerts
                intervention["actions"].append("Set ShadowAgent to maximum sensitivity")

            # Issue emergency directive
            self.issue_global_directive(
                directive_type=DirectiveType.ADAPTIVE_DEFENSE,
                parameters={"reason": "Crisis: High alert level", "emergency": True},
                priority=5,  # Maximum priority
            )

            console.print(
                "[red bold]🚨 CRISIS: High alert level detected! Switching to emergency stealth mode.[/red bold]"
            )

        elif crisis_type == "agent_stuck":
            # Initialize novel_command at method level to ensure it's defined
            novel_command = "nmap -sS -sV target"  # Default fallback command
            
            # Intervention for stuck agent: Force exploration
            if "RedAgent" in self.subordinate_agents:
                red_agent = self.subordinate_agents["RedAgent"]
                if hasattr(red_agent, "epsilon"):
                    red_agent.epsilon = 0.9  # Force exploration
                # Clear redundancy counter
                if hasattr(red_agent, "redundancy_counter"):
                    red_agent.redundancy_counter = 0
                # Reset repeat steps
                if hasattr(red_agent, "repeat_steps"):
                    red_agent.repeat_steps = 0

                intervention["actions"].append("Forced RedAgent exploration")

                # Generate a novel command with GPT (try to improve default)
                env_state = context.get("env_state", {})
                current_phase = env_state.get("phase", "recon")

                prompt = f"""
                The RedAgent is stuck repeating actions in phase '{current_phase}'.
                
                Context:
                - Open ports: {env_state.get('open_ports', [])}
                - Services: {env_state.get('services', [])}
                - Alert level: {env_state.get('blue_team_alert', 0)}
                
                Generate a novel, non-repetitive offensive security command appropriate for this phase.
                Respond with JUST the command, no explanations.
                """

                try:
                    gpt_response = self.gpt_manager.gpt_request(
                        prompt,
                        model="gpt-4o-mini",
                        task_type="intervention",
                        agent_id=self.agent_id,
                    )

                    if (
                        gpt_response
                        and isinstance(gpt_response, str)
                        and len(gpt_response.split()) >= 2
                    ):
                        novel_command = gpt_response  # Override with GPT response
                        # Add to RedAgent command history
                        if hasattr(red_agent, "command_history"):
                            red_agent.command_history.append(novel_command)

                        intervention["actions"].append(
                            f"Suggested novel command: {novel_command}"
                        )
                        console.print(
                            f"[yellow]🔄 OrionAgent intervention: Suggested novel command for stuck agent: [bold]{novel_command}[/bold][/yellow]"
                        )
                except Exception as e:
                    console.print(
                        f"[yellow]⚠ Error generating novel command: {e}[/yellow]"
                    )
                    # novel_command already has fallback value

            # Issue directive to unstuck agent
            self.issue_directive(
                directive_type=DirectiveType.COORDINATE_ACTION,
                target_agent="RedAgent",
                parameters={"forced_exploration": True, "novel_command": novel_command},
                priority=4,
            )

            console.print(
                "[yellow bold]🚨 CRISIS: Agent stuck detected! Forcing exploration.[/yellow bold]"
            )

        elif crisis_type == "no_progress":
            # Intervention for no progress: Change phase
            env_state = context.get("env_state", {})
            current_phase = env_state.get("phase", "recon")

            # Determine next phase
            phase_order = ["recon", "enumeration", "exploit", "privesc", "exfiltrate"]
            try:
                current_idx = phase_order.index(current_phase)
                next_phase = (
                    phase_order[current_idx + 1]
                    if current_idx < len(phase_order) - 1
                    else phase_order[0]
                )
            except (ValueError, IndexError):
                next_phase = "enumeration"  # Default fallback

            # Suggest phase change
            intervention["actions"].append(
                f"Suggested phase change: {current_phase} → {next_phase}"
            )

            # Apply phase change if possible
            if (
                self.red_agent
                and hasattr(self.red_agent, "env")
                and hasattr(self.red_agent.env, "current_phase")
            ):
                self.red_agent.env.current_phase = next_phase

                console.print(
                    f"[magenta bold]🚨 CRISIS: No progress detected! Changed phase: {current_phase} → {next_phase}[/magenta bold]"
                )

            # Generate appropriate commands for new phase
            new_chain = self.generate_action_chain({"env_state": {"phase": next_phase}})
            if new_chain:
                self.current_chain = new_chain
                intervention["actions"].append(
                    f"Generated new action chain for phase {next_phase}"
                )

            # Issue phase change directive
            self.issue_directive(
                directive_type=DirectiveType.CHANGE_STRATEGY,
                target_agent="RedAgent",
                parameters={
                    "new_phase": next_phase,
                    "reason": "No progress in current phase",
                },
                priority=4,
            )

        # Log intervention to memory
        experience = {
            "state": context.get("env_state", {}),
            "action": "crisis_intervention",
            "crisis_type": crisis_type,
            "intervention": intervention["actions"],
        }
        self.replay_buffer.add(experience)

        return intervention

    def _issue_routine_directives(self, env_state: Dict[str, Any]) -> None:
        """
        Issue routine directives based on current environment state.
        Provides baseline guidance between strategic/tactical reviews.

        Args:
            env_state: Current environment state
        """
        current_phase = env_state.get("phase", "recon")

        # Recon phase directives
        if current_phase == "recon":
            # If few ports discovered, encourage broad scanning
            if len(env_state.get("open_ports", [])) < 5:
                self.issue_directive(
                    directive_type=DirectiveType.FOCUS_TARGET,
                    target_agent="ScoutAgent",
                    parameters={
                        "action": "broad_scan",
                        "reason": "Insufficient port discovery",
                    },
                    priority=2,
                )

        # Enumeration phase directives
        elif current_phase == "enumeration":
            # If few services identified, prioritize service enumeration
            if len(env_state.get("services", [])) < 3:
                self.issue_directive(
                    directive_type=DirectiveType.FOCUS_TARGET,
                    target_agent="RedAgent",
                    parameters={
                        "action": "service_enumeration",
                        "reason": "Insufficient service identification",
                    },
                    priority=2,
                )

        # Exploit phase directives
        elif current_phase == "exploit":
            # If no credentials found, prioritize credential harvesting
            if not env_state.get("credentials_found", False):
                self.issue_directive(
                    directive_type=DirectiveType.FOCUS_TARGET,
                    target_agent="RedAgent",
                    parameters={
                        "action": "credential_harvest",
                        "reason": "No credentials discovered",
                    },
                    priority=2,
                )

        # Privesc phase directives
        elif current_phase == "privesc":
            # If user-level access, prioritize escalation methods
            if env_state.get("privilege_level", "none") == "user":
                self.issue_directive(
                    directive_type=DirectiveType.FOCUS_TARGET,
                    target_agent="RedAgent",
                    parameters={
                        "action": "privesc_enumeration",
                        "reason": "Need higher privileges",
                    },
                    priority=2,
                )

        # Exfiltrate phase directives
        elif current_phase == "exfiltrate":
            # If not exfiltrated, prioritize data identification
            if not env_state.get("data_exfiltrated", False):
                self.issue_directive(
                    directive_type=DirectiveType.FOCUS_TARGET,
                    target_agent="RedAgent",
                    parameters={
                        "action": "identify_valuable_data",
                        "reason": "Data exfiltration required",
                    },
                    priority=2,
                )

        # Adjust BlueAgent defensive posture based on red team progress
        blue_alert = env_state.get("blue_team_alert", 0)
        if blue_alert > 50:
            self.issue_directive(
                directive_type=DirectiveType.ADAPTIVE_DEFENSE,
                target_agent="BlueAgent",
                parameters={
                    "defensive_posture": "active",
                    "reason": "Elevated alert level",
                },
                priority=3,
            )
        elif blue_alert > 20:
            self.issue_directive(
                directive_type=DirectiveType.ADAPTIVE_DEFENSE,
                target_agent="BlueAgent",
                parameters={
                    "defensive_posture": "vigilant",
                    "reason": "Moderate alert level",
                },
                priority=2,
            )

    def analyze_training(self, agents: List[Any]) -> Dict[str, Any]:
        """
        Analyze training performance across all agents and episodes.

        Args:
            agents: List of all agents to analyze

        Returns:
            Dict with analysis results
        """
        # Collect performance stats
        agent_stats = {}

        for agent in agents:
            if hasattr(agent, "stats_monitor") and hasattr(
                agent.stats_monitor, "get_average_reward"
            ):
                avg_reward = agent.stats_monitor.get_average_reward()
                agent_stats[agent.agent_id] = {
                    "avg_reward": avg_reward,
                    "total_steps": getattr(agent, "total_steps", 0),
                }

                # Track performance over time for this agent
                self.agent_performance[agent.agent_id].append(avg_reward)

        # Prepare analysis data
        red_performance = self.agent_performance.get("RedAgent", [])
        blue_performance = self.agent_performance.get("BlueAgent", [])

        # Calculate performance trend
        red_trend = (
            "improving"
            if len(red_performance) >= 2 and red_performance[-1] > red_performance[-2]
            else "declining"
        )

        # Save episode stats
        episode_stat = {
            "episode": self.episode_counter,
            "agent_stats": agent_stats,
            "red_trend": red_trend,
        }
        self.episode_stats.append(episode_stat)

        # Generate insights using GPT
        if red_performance and len(red_performance) >= 2:
            performance_trend = ", ".join(
                [f"{i+1}:{p:.2f}" for i, p in enumerate(red_performance[-5:]) if i < 5]
            )

            prompt = f"""
            Analyze this training performance trend for RedAgent:
            Episodes: {performance_trend}
            
            Current trend: {red_trend}
            
            In 1-2 sentences, suggest an optimization strategy based on this performance trend.
            """

            try:
                insight = self.gpt_manager.gpt_request(
                    prompt,
                    model="gpt-4o-mini",
                    task_type="analysis",
                    agent_id=self.agent_id,
                )

                if self.verbosity not in ["quiet", "silent"]:
                    console.print(f"[blue]👁️ OrionAgent Performance Analysis:[/blue]")
                    console.print(f"[magenta]{insight}[/magenta]")

                # Update directives based on insight
                self._update_directives_from_insight(insight)

                episode_stat["gpt_insight"] = insight

            except Exception as e:
                console.print(
                    f"[yellow]⚠ Error generating training analysis: {e}[/yellow]"
                )

        return episode_stat

    def _update_directives_from_insight(self, insight: str) -> None:
        """
        Update strategic directives based on GPT insight.

        Args:
            insight: GPT-generated insight string
        """
        # Extract keywords from insight
        keywords = set(insight.lower().split())

        # Check for stealth-related insights
        if any(
            word in keywords for word in ["stealth", "quiet", "alerts", "detection"]
        ):
            # Update recon directives to emphasize stealth
            self.strategic_directives["recon"] = [
                "Use slower scan timing (-T1 or -T2) to avoid detection",
                "Utilize SYN scans (-sS) instead of full connect scans",
                "Scan small batches of ports with delays between scans",
            ]

            # Update exploit directives for stealth
            self.strategic_directives["exploit"] = [
                "Use manual exploitation instead of noisy automated tools",
                "Test exploits on a single target before wider deployment",
                "Avoid brute force attacks that generate excessive traffic",
            ]

        # Check for thoroughness-related insights
        elif any(
            word in keywords
            for word in ["thorough", "complete", "comprehensive", "discovery"]
        ):
            # Update recon directives to emphasize thoroughness
            self.strategic_directives["recon"] = [
                "Perform comprehensive port scans across all port ranges",
                "Use service version detection (-sV) to identify exact versions",
                "Utilize multiple scan types to ensure complete coverage",
            ]

            # Update enumeration directives for thoroughness
            self.strategic_directives["enumeration"] = [
                "Systematically enumerate each discovered service",
                "Use multiple enumeration tools to cross-validate findings",
                "Create detailed inventory of all discovered attack surfaces",
            ]

        # Check for exploitation-focused insights
        elif any(
            word in keywords
            for word in ["exploit", "attack", "vulnerability", "access"]
        ):
            # Update exploit directives
            self.strategic_directives["exploit"] = [
                "Prioritize high-probability vulnerabilities in exposed services",
                "Follow up service fingerprinting immediately with exploit attempts",
                "Use both automated and manual exploitation techniques",
            ]

            # Update privesc directives
            self.strategic_directives["privesc"] = [
                "Focus on obtaining and leveraging credentials first",
                "Target misconfigured permissions and SUID binaries",
                "Establish persistence after initial privilege escalation",
            ]

    def apply_orion_strategic_adjustments(self, agents: List[Any]) -> Dict[str, Any]:
        """
        Apply strategic adjustments to all agents and calculate coherence metrics.

        Args:
            agents: List of all agents to adjust

        Returns:
            Dict with adjustment results and coherence metrics
        """
        # Get context
        env_state = {}
        if self.red_agent and hasattr(self.red_agent, "env"):
            env_state = self.red_agent.env.get_global_state()

        context = self._gather_context(env_state)

        # Apply adjustments
        adjustments = self.adjust_agent_parameters(context)

        # Check for curriculum advancement
        curriculum_advanced = self._check_curriculum_advancement(context)

        # Calculate strategic coherence between agents
        coherence = self._calculate_strategic_coherence(agents)

        # Log strategic adjustments to memory router
        if self.memory_router:
            self.memory_router.log_orion_strategic_adjustment({
                "global_strategy": self.global_strategy,
                "adjustments": adjustments,
                "coherence": coherence,
                "episode": self.episode_counter,
                "step": self.step_counter,
            })

        return {
            "parameter_adjustments": adjustments,
            "curriculum_advanced": curriculum_advanced,
            "global_strategy": self.global_strategy,
            "coherence": coherence,
        }

    def _calculate_strategic_coherence(self, agents: List[Any]) -> float:
        """
        Calculate a strategic coherence score between all agents.

        Args:
            agents: List of all agents

        Returns:
            Coherence score (0.0-1.0)
        """
        # Base coherence
        coherence_score = 0.5

        # Count how many agents are aligned with the global strategy
        aligned_count = 0
        total_count = 0

        for agent in agents:
            # Skip non-relevant agents
            if not agent or not hasattr(agent, "agent_id"):
                continue

            total_count += 1

            # Check mode alignment
            if hasattr(agent, "current_mode") and isinstance(agent.current_mode, str):
                if (
                    (
                        self.global_strategy == "stealth"
                        and "stealth" in agent.current_mode.lower()
                    )
                    or (
                        self.global_strategy == "aggressive"
                        and "aggressive" in agent.current_mode.lower()
                    )
                    or (
                        self.global_strategy == "balanced"
                        and "balanced" in agent.current_mode.lower()
                    )
                ):
                    aligned_count += 1

            # Check stealth mode alignment for scout and shadow agents
            if hasattr(agent, "stealth_mode") and isinstance(agent.stealth_mode, bool):
                if (self.global_strategy == "stealth" and agent.stealth_mode) or (
                    self.global_strategy == "aggressive" and not agent.stealth_mode
                ):
                    aligned_count += 1
                else:
                    aligned_count -= 0.5  # Penalty for misalignment

        # Calculate coherence as a ratio of aligned agents
        if total_count > 0:
            alignment_ratio = aligned_count / total_count
            # Scale between 0.2-1.0 to avoid too low scores
            coherence_score = 0.2 + (0.8 * max(0.0, min(1.0, alignment_ratio)))

        return coherence_score

    def update_global_strategy(
        self, agents: List[Any], environment: str = "dynamic"
    ) -> str:
        """
        Update global strategy based on agent performance and environment.

        Args:
            agents: List of all agents
            environment: Environment type (static, dynamic)

        Returns:
            Strategic insight
        """
        # Get performance metrics from agents
        red_performance = []
        blue_performance = []
        detection_rates = []

        for agent in agents:
            if not agent or not hasattr(agent, "agent_id"):
                continue

            if agent.agent_id == "RedAgent" and hasattr(agent, "stats_monitor"):
                red_performance.append(agent.stats_monitor.get_average_reward())
                if hasattr(agent.stats_monitor, "get_detection_rate"):
                    detection_rates.append(agent.stats_monitor.get_detection_rate())
            elif agent.agent_id == "BlueAgent" and hasattr(agent, "stats_monitor"):
                blue_performance.append(agent.stats_monitor.get_average_reward())

        # Calculate metrics
        avg_red_reward = sum(red_performance) / max(1, len(red_performance))
        avg_blue_reward = sum(blue_performance) / max(1, len(blue_performance))
        avg_detection = sum(detection_rates) / max(1, len(detection_rates))

        # Determine strategy based on metrics and environment
        if avg_detection > 0.5:  # High detection rate
            self.global_strategy = "stealth"
            strategy_insight = "High detection rate observed. Switching to stealth-focused strategy to minimize footprint."

            # Issue high priority stealth directive
            self.issue_global_directive(
                directive_type=DirectiveType.INCREASE_STEALTH,
                parameters={"reason": "High detection rate", "level": "maximum"},
                priority=4,
            )

        elif avg_red_reward < 0 and environment == "dynamic":
            self.global_strategy = "balanced"
            strategy_insight = "Red team struggling. Adopting balanced approach to establish baseline capabilities."

            # No specific directive - balanced is default

        elif avg_red_reward > 10 and avg_blue_reward > 5:
            self.global_strategy = "aggressive"
            strategy_insight = "Both teams performing well. Shifting to aggressive strategy to push performance boundaries."

            # Issue aggressive directive
            self.issue_global_directive(
                directive_type=DirectiveType.DECREASE_STEALTH,
                parameters={"reason": "High performance teams", "level": "optimized"},
                priority=3,
            )

        else:
            # Keep current strategy or set balanced as default
            if not hasattr(self, "global_strategy") or not self.global_strategy:
                self.global_strategy = "balanced"
            strategy_insight = f"Maintaining {self.global_strategy} strategy based on current performance metrics."

        # Log the strategy update
        if self.verbosity not in ["quiet", "silent"]:
            console.print(
                f"[blue]👁️ OrionAgent updated global strategy: {self.global_strategy} - {strategy_insight}[/blue]"
            )

        # Apply the new strategy to agents
        self.apply_orion_strategic_adjustments(agents)

        return strategy_insight

    def generate_dynamic_scenario(
        self, scenario_type: str, default_services: List[str]
    ) -> Dict[str, Any]:
        """
        Generate a dynamic scenario configuration.

        Args:
            scenario_type: Type of scenario to generate
            default_services: List of default services to include

        Returns:
            Dict with scenario configuration
        """
        # Generate dynamic difficulty based on agent performance
        red_performance = self.agent_performance.get("RedAgent", [])
        avg_reward = (
            sum(red_performance[-3:]) / max(len(red_performance[-3:]), 1)
            if red_performance
            else 0
        )

        # Scale difficulty: higher reward → higher difficulty
        difficulty = 1
        if avg_reward > 20:
            difficulty = min(10, int(avg_reward / 10) + 1)
        elif avg_reward > 10:
            difficulty = min(5, int(avg_reward / 5) + 1)

        # Select service subset, weighted by difficulty
        num_services = min(5 + int(difficulty / 2), len(default_services))
        services = random.sample(default_services, num_services)

        # Generate scenario configuration
        scenario = {
            "difficulty": difficulty,
            "services": services,
            "training_mode": "adaptive",
            "blue_aggressiveness": min(5, int(difficulty / 2) + 1),
            "traceback_threshold": max(50, 100 - difficulty * 5),
        }

        if self.verbosity not in ["quiet", "silent"]:
            console.print(f"[blue]👁️ OrionAgent generated dynamic scenario:[/blue]")
            console.print(
                f"[cyan]  Difficulty: {difficulty}, Services: {len(services)}, Blue Agg: {scenario['blue_aggressiveness']}[/cyan]"
            )

        return scenario

    def evaluate_environment(self, state: Dict[str, Any]) -> str:
        """
        Evaluate the current environment state and provide insights.

        Args:
            state: Current environment state

        Returns:
            String with environmental insights
        """
        # Extract key metrics
        phase = state.get("phase", "recon")
        blue_team_alert = state.get("blue_team_alert", 0)
        detection_risk = state.get("detection_risk", 0)

        # Check critical thresholds
        if blue_team_alert > 80:
            return "Critical alert level detected! Recommend immediate increase in stealth operations."
        if detection_risk > 8:
            return (
                "High detection risk! Consider pausing operations or rotating tactics."
            )

        # Phase-specific insights
        if phase == "recon" and len(state.get("open_ports", [])) < 3:
            return "Recommend broader port scanning with moderate stealth."
        if phase == "enumeration" and len(state.get("services", [])) < 3:
            return "Service discovery is incomplete. Focus on service detection."
        if phase == "exploit" and not state.get("credentials_found", False):
            return "Prioritize credential harvesting before proceeding to advanced exploits."
        if phase == "privesc" and state.get("privilege_level", "none") == "user":
            return (
                "Enumerate sudo permissions and SUID binaries for privilege escalation."
            )
        if phase == "exfiltrate" and state.get("privilege_level", "none") != "root":
            return "Insufficient privileges for complete exfiltration. Focus on escalation first."

        # Default insights based on blue team alert
        if blue_team_alert > 60:
            return "Prepare counter-measures for blue team response. Consider defensive maneuvers."
        elif blue_team_alert > 30:
            return (
                "Moderate blue team activity detected. Increase stealth in operations."
            )
        else:
            return "Environment is stable. Proceed with current phase objectives."

    def sync_memory(self) -> bool:
        """
        Synchronize memory with the global memory router.
        Implementation of MemorySyncInterface.
        """
        try:
            if self.memory_router:
                # Generate strategic summary
                env_state = {}
                if self.red_agent and hasattr(self.red_agent, "env"):
                    env_state = self.red_agent.env.get_global_state()

                context = self._gather_context(env_state)

                # Periodically generate insights
                if self.step_counter % self.strategic_review_frequency == 0:
                    insight = self.perform_strategic_review(context)

                    # Report insight to memory router
                    self.memory_router.log_orion_insight({
                        "insight": insight,
                        "global_strategy": self.global_strategy,
                        "episode": self.episode_counter,
                        "step": self.step_counter,
                    })
            return True
        except Exception as e:
            logger.warning(f"Failed to sync memory: {e}")
            return False

    def notify_stagnation(self, agent_id: str, stagnation_data: Dict[str, Any]):
        """Handle notification of agent stagnation and take corrective action."""
        try:
            if self.verbosity not in ["quiet", "silent"]:
                console.print(f"[yellow]⚠️ OrionAgent: Stagnation detected in {agent_id}[/yellow]")
            
            # Log stagnation event
            if self.memory_router:
                self.memory_router.log_orion_insight({
                    "type": "stagnation_notification",
                    "agent_id": agent_id,
                    "stagnation_data": stagnation_data,
                    "timestamp": time.time(),
                    "episode": self.episode_counter,
                    "step": self.step_counter
                })
            
            # Analyze stagnation cause
            stagnation_analysis = self._analyze_stagnation(agent_id, stagnation_data)
            
            # Take corrective action
            corrective_actions = self._apply_stagnation_corrections(agent_id, stagnation_analysis)
            
            if self.verbosity not in ["quiet", "silent"]:
                console.print(f"[green]✓ OrionAgent: Applied {len(corrective_actions)} corrective actions for {agent_id}[/green]")
                
            return {
                "status": "handled",
                "analysis": stagnation_analysis,
                "actions_taken": corrective_actions
            }
            
        except Exception as e:
            logger.warning(f"Failed to handle stagnation notification: {e}")
            return {"status": "error", "error": str(e)}
    
    def _analyze_stagnation(self, agent_id: str, stagnation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the cause of agent stagnation."""
        try:
            # Basic analysis based on stagnation data
            analysis = {
                "agent_id": agent_id,
                "likely_causes": [],
                "severity": "medium",
                "recommendations": []
            }
            
            # Check for common stagnation patterns
            if stagnation_data.get("reward_stagnation", False):
                analysis["likely_causes"].append("reward_plateau")
                analysis["recommendations"].append("increase_exploration")
                
            if stagnation_data.get("action_repetition", False):
                analysis["likely_causes"].append("action_loops")
                analysis["recommendations"].append("vary_action_selection")
                
            if stagnation_data.get("low_learning_rate", False):
                analysis["likely_causes"].append("insufficient_learning")
                analysis["recommendations"].append("adjust_learning_parameters")
                
            # Determine severity
            if len(analysis["likely_causes"]) >= 2:
                analysis["severity"] = "high"
            elif len(analysis["likely_causes"]) == 0:
                analysis["severity"] = "low"
                
            return analysis
            
        except Exception as e:
            logger.warning(f"Failed to analyze stagnation: {e}")
            return {"agent_id": agent_id, "error": str(e)}
    
    def _apply_stagnation_corrections(self, agent_id: str, analysis: Dict[str, Any]) -> List[str]:
        """Apply corrective actions based on stagnation analysis."""
        actions_taken = []
        
        try:
            # Find the stagnating agent
            target_agent = None
            if hasattr(self, 'agent_manager') and self.agent_manager:
                for agent in getattr(self.agent_manager, 'agents', []):
                    if hasattr(agent, 'agent_id') and agent.agent_id == agent_id:
                        target_agent = agent
                        break
            
            if not target_agent:
                return ["agent_not_found"]
            
            # Apply recommendations
            for recommendation in analysis.get("recommendations", []):
                if recommendation == "increase_exploration" and hasattr(target_agent, 'epsilon'):
                    old_epsilon = getattr(target_agent, 'epsilon', 0.1)
                    new_epsilon = min(old_epsilon * 1.5, 0.8)
                    target_agent.epsilon = new_epsilon
                    actions_taken.append(f"increased_exploration_{old_epsilon:.3f}_to_{new_epsilon:.3f}")
                    
                elif recommendation == "vary_action_selection" and hasattr(target_agent, 'action_selection_mode'):
                    target_agent.action_selection_mode = "varied"
                    actions_taken.append("enabled_action_variation")
                    
                elif recommendation == "adjust_learning_parameters" and hasattr(target_agent, 'learning_rate'):
                    old_lr = getattr(target_agent, 'learning_rate', 0.001)
                    new_lr = old_lr * 1.2
                    target_agent.learning_rate = new_lr
                    actions_taken.append(f"adjusted_learning_rate_{old_lr:.6f}_to_{new_lr:.6f}")
                    
            # If no specific actions, apply general reset
            if not actions_taken:
                if hasattr(target_agent, 'reset_internal_state'):
                    target_agent.reset_internal_state()
                    actions_taken.append("reset_internal_state")
                else:
                    actions_taken.append("no_action_available")
                    
        except Exception as e:
            logger.warning(f"Failed to apply stagnation corrections: {e}")
            actions_taken.append(f"error_{str(e)}")
            
        return actions_taken

    def get_base_commands(self):
        """Return empty list as OrionAgent doesn't execute commands directly."""
        return []

    def reset(self):
        """Reset agent state for a new episode."""
        self.step_counter = 0
        self.episode_counter += 1

        # Reset current chain
        self.current_chain = []

        # Log episode stats
        if self.red_agent and hasattr(self.red_agent, "stats_monitor"):
            red_reward = self.red_agent.stats_monitor.get_average_reward()
            self.agent_performance["RedAgent"].append(red_reward)


# Test execution
if __name__ == "__main__":
    agent = OrionAgent(verbosity="verbose")

    # Test dynamic scenario generation
    scenario = agent.generate_dynamic_scenario(
        "standard", ["ssh", "http", "ftp", "smb", "rdp", "mysql"]
    )
    console.print(f"[bold cyan]Generated scenario:[/bold cyan]")
    console.print(scenario)

    # Test strategic review
    context = {
        "env_state": {
            "phase": "recon",
            "open_ports": [22, 80, 443],
            "services": ["ssh", "http", "https"],
            "blue_team_alert": 20.0,
            "detection_risk": 3.0,
        },
        "agent_statuses": {
            "RedAgent": {
                "mode": "balanced",
                "epsilon": 0.3,
                "avg_reward": 5.0,
                "last_command": "nmap -sS 10.10.10.10",
            },
            "BlueAgent": {"mode": "Standard", "epsilon": 0.2, "avg_reward": 2.0},
        },
    }

    review = agent.perform_strategic_review(context)
    console.print(f"[bold green]Strategic review:[/bold green]")
    console.print(review)

    # Test action chain generation
    chain = agent.generate_action_chain(context)
    console.print(f"[bold magenta]Action chain:[/bold magenta]")
    for command in chain:
        console.print(f"- {command}")

    # Test parameter adjustments
    adjustments = agent.adjust_agent_parameters(context)
    console.print(f"[bold blue]Parameter adjustments:[/bold blue]")
    console.print(adjustments)
