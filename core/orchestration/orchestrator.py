#!/usr/bin/env python3
"""
core/orchestration/orchestrator.py — ARIASKA Multi-Agent Orchestrator v1.0

Coordinates all 5 agents (Scout, Red, Blue, Orion, Shadow) in a unified training loop.
Each agent has its own ApprenticeCoach for GPT mentorship.
"""

import os
import time
import logging
import hashlib
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from core.gpt_manager import GPTManager
    from core.environment.cyber_environment import CyberEnvironment
    from core.tracing import TraceWriter, StepTrace
    from core.postmortem import SkillLibrary

logger = logging.getLogger("ariaska.orchestrator")


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""
    
    # Agent activation
    enable_scout: bool = True
    enable_red: bool = True
    enable_blue: bool = True
    enable_orion: bool = True
    enable_shadow: bool = True
    
    # Mentor settings (shared across agents)
    mentor_mode: str = "anneal"  # "anneal", "threshold", "always", "never"
    mentor_warmup_episodes: int = 1
    mentor_min_rate: float = 0.15
    mentor_max_rate: float = 1.0
    
    # Stuck detection
    stuck_threshold: int = 3         # Same action N times = stuck
    stuck_negative_streak: int = 5   # N consecutive negative rewards = stuck
    stuck_force_mentor: bool = True  # Force mentor call when stuck
    
    # Execution
    max_steps_per_episode: int = 50
    
    # Logging
    mentor_log_dir: str = "traces"  # mentor.jsonl goes here


@dataclass
class AgentStepResult:
    """Result from a single agent's step."""
    agent_name: str
    proposed_action: str
    chosen_action: str
    mentor_call: bool
    model_used: Optional[str]
    mentor_guidance: Optional[str]
    mentor_delta: str
    confidence: float
    skill_ids: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent_name,
            "proposed_action": self.proposed_action,
            "chosen_action": self.chosen_action,
            "mentor_call": self.mentor_call,
            "model_used": self.model_used,
            "mentor_guidance": self.mentor_guidance,
            "mentor_delta": self.mentor_delta,
            "confidence": self.confidence,
            "skill_ids": self.skill_ids,
        }


class Orchestrator:
    """
    Multi-agent orchestrator for ARIASKA training.
    
    Coordinates:
    - ScoutAgent: Reconnaissance suggestions
    - RedAgent: Attack execution
    - BlueAgent: Defense actions
    - OrionAgent: Strategic critique
    - ShadowAgent: Memory routing
    
    Each agent has its own ApprenticeCoach for GPT mentorship.
    """
    
    AGENT_ORDER = ["ScoutAgent", "RedAgent", "BlueAgent", "OrionAgent", "ShadowAgent"]
    
    def __init__(
        self,
        env: "CyberEnvironment",
        gpt_manager: "GPTManager",
        trace_writer: Optional["TraceWriter"] = None,
        skill_library: Optional["SkillLibrary"] = None,
        config: Optional[OrchestratorConfig] = None,
        verbosity: str = "standard",
    ):
        self.env = env
        self.gpt_manager = gpt_manager
        self.trace_writer = trace_writer
        self.skill_library = skill_library
        self.config = config or OrchestratorConfig()
        self.verbosity = verbosity
        
        # Current run directory for mentor logs
        self.run_dir: Optional[str] = None
        
        # Initialize agents and coaches
        self.agents: Dict[str, Any] = {}
        self.coaches: Dict[str, Any] = {}
        
        self._init_agents()
        self._init_coaches()
        
        # Step tracking
        self.current_episode = 0
        self.current_step = 0
        self.episode_results: List[List[AgentStepResult]] = []
        
        # Stuck detection tracking (per-agent)
        self.action_history: Dict[str, List[str]] = {}  # agent -> recent actions
        self.reward_history: List[float] = []  # recent rewards
        self.stuck_agents: set = set()  # agents currently in stuck state
        
        logger.info(f"Orchestrator initialized with {len(self.agents)} agents")
    
    def _init_agents(self):
        """Initialize all agents."""
        from core.multiagent.memory_router import MemoryRouter
        
        memory_router = MemoryRouter()
        
        # ScoutAgent
        if self.config.enable_scout:
            try:
                from core.agents.scout_agent import ScoutAgent
                self.agents["ScoutAgent"] = ScoutAgent(
                    agent_id="ScoutAgent",
                    memory_router=memory_router,
                    verbosity=self.verbosity,
                )
            except Exception as e:
                logger.warning(f"Failed to init ScoutAgent: {e}")
        
        # RedAgent
        if self.config.enable_red:
            try:
                from core.agents.red_agent import RedAgent
                self.agents["RedAgent"] = RedAgent(
                    agent_id="RedAgent",
                    role="CyberOffense",
                    memory_router=memory_router,
                    verbosity=self.verbosity,
                )
            except Exception as e:
                logger.warning(f"Failed to init RedAgent: {e}")
        
        # BlueAgent
        if self.config.enable_blue:
            try:
                from core.agents.blue_agent import BlueAgent
                self.agents["BlueAgent"] = BlueAgent(
                    agent_id="BlueAgent",
                    memory_router=memory_router,
                    verbosity=self.verbosity,
                )
            except Exception as e:
                logger.warning(f"Failed to init BlueAgent: {e}")
        
        # OrionAgent
        if self.config.enable_orion:
            try:
                from core.agents.orion_agent import OrionAgent
                self.agents["OrionAgent"] = OrionAgent(
                    agent_id="OrionAgent",
                    memory_router=memory_router,
                    verbosity=self.verbosity,
                )
            except Exception as e:
                logger.warning(f"Failed to init OrionAgent: {e}")
        
        # ShadowAgent
        if self.config.enable_shadow:
            try:
                from core.agents.shadow_agent import ShadowAgent
                self.agents["ShadowAgent"] = ShadowAgent(
                    agent_id="ShadowAgent",
                    memory_router=memory_router,
                    verbosity=self.verbosity,
                )
            except Exception as e:
                logger.warning(f"Failed to init ShadowAgent: {e}")
        
        logger.info(f"Initialized agents: {list(self.agents.keys())}")
    
    def _init_coaches(self):
        """Initialize ApprenticeCoach for each agent."""
        from core.training.apprentice_coach import ApprenticeCoach
        from core.training.mentor_policy import MentorPolicy, MentorPolicyConfig
        
        policy_config = MentorPolicyConfig(
            mode=self.config.mentor_mode,
            warmup_episodes=self.config.mentor_warmup_episodes,
            min_mentor_rate=self.config.mentor_min_rate,
            max_mentor_rate=self.config.mentor_max_rate,
        )
        
        for agent_name in self.agents.keys():
            # Each agent gets its own policy instance (to track separately)
            policy = MentorPolicy(policy_config)
            
            self.coaches[agent_name] = ApprenticeCoach(
                agent_name=agent_name,
                gpt_manager=self.gpt_manager,
                mentor_policy=policy,
                skill_library=self.skill_library,
                trace_writer=self.trace_writer,
                mentor_log_path=None,  # Set per-run
            )
        
        logger.info(f"Initialized coaches: {list(self.coaches.keys())}")
    
    def set_run_dir(self, run_dir: str):
        """Set the run directory for mentor logs."""
        self.run_dir = run_dir
        mentor_log_path = os.path.join(run_dir, "mentor.jsonl")
        
        # Update all coaches
        for coach in self.coaches.values():
            coach.mentor_log_path = mentor_log_path
    
    def run_episode(
        self,
        episode_id: str,
        episode_number: int,
        max_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run a complete episode with all agents.
        
        Per global step t:
        A) Scout proposes intel/recon suggestion
        B) Red proposes attack command
        C) Blue proposes defense action
        D) Execute env.step once with Red+Blue actions
        E) Orion produces critique
        F) Shadow produces memory routing
        G) Log StepTrace for each agent
        
        Returns:
            Episode summary with metrics
        """
        max_steps = max_steps or self.config.max_steps_per_episode
        self.current_episode = episode_number
        
        # Reset token budgets for this episode
        if self.gpt_manager:
            self.gpt_manager.reset_episode(episode_id=episode_number)
        
        # Reset coaches for new episode
        for coach in self.coaches.values():
            coach.reset_episode(episode_number)
        
        # Reset stuck detection
        self.action_history.clear()
        self.reward_history.clear()
        self.stuck_agents.clear()
        
        # Reset environment
        state = self.env.reset()
        if not state:
            state = self._default_state()
        
        # Episode tracking
        episode_reward = 0.0
        step_results: List[List[AgentStepResult]] = []
        done = False
        
        for step in range(max_steps):
            self.current_step = step
            
            # Get phase from state
            phase = state.get("phase", "recon") if isinstance(state, dict) else "recon"
            
            # Run all agents for this step
            step_agent_results, env_reward, done = self._run_step(
                episode_id=episode_id,
                step=step,
                state=state,
                phase=phase,
            )
            
            step_results.append(step_agent_results)
            episode_reward += env_reward
            
            # Update state from environment
            if done:
                break
            
            state = self.env.get_global_state() if hasattr(self.env, 'get_global_state') else state
        
        # Compute episode metrics
        metrics = self._compute_episode_metrics(step_results, episode_reward, done)
        
        return metrics
    
    def _run_step(
        self,
        episode_id: str,
        step: int,
        state: Dict[str, Any],
        phase: str,
    ) -> Tuple[List[AgentStepResult], float, bool]:
        """
        Run a single step with all agents.
        
        Returns:
            (agent_results, reward, done)
        """
        from core.training.apprentice_coach import StepContext
        from core.tracing import StepTrace
        
        agent_results: List[AgentStepResult] = []
        
        # Build context
        target_ip = state.get("target_ip", "10.10.10.10") if isinstance(state, dict) else "10.10.10.10"
        open_ports = state.get("open_ports", []) if isinstance(state, dict) else []
        detection_risk = state.get("detection_risk", 0.0) if isinstance(state, dict) else 0.0
        blue_alert = state.get("blue_team_alert", 0.0) if isinstance(state, dict) else 0.0
        history = state.get("history", []) if isinstance(state, dict) else []
        
        # Compute state digest (small hash)
        state_str = str(sorted(state.items())) if isinstance(state, dict) else str(state)
        state_digest = hashlib.md5(state_str.encode()).hexdigest()[:8]
        
        # Actions to execute (Red + Blue)
        red_action = None
        blue_action = None
        
        # Process each agent in order
        for agent_name in self.AGENT_ORDER:
            if agent_name not in self.agents or agent_name not in self.coaches:
                continue
            
            agent = self.agents[agent_name]
            coach = self.coaches[agent_name]
            
            # Check if agent is stuck (repeated actions)
            is_stuck = self._check_if_stuck(agent_name)
            force_mentor = is_stuck and self.config.stuck_force_mentor
            
            # Get agent's command history for anti-loop logic
            agent_cmd_history = self.action_history.get(agent_name, [])
            
            # Inject command history into state so coach can see what was tried
            enriched_state = dict(state) if isinstance(state, dict) else {}
            enriched_state["command_history"] = agent_cmd_history[-15:]
            if agent_cmd_history:
                enriched_state["last_command"] = agent_cmd_history[-1]
            
            # Build step context
            ctx = StepContext(
                episode=self.current_episode,
                step=step,
                phase=phase,
                state=enriched_state,
                agent_name=agent_name,
                history=agent_cmd_history[-5:],  # Use AGENT's history, not global
                target_ip=target_ip,
                open_ports=open_ports,
                detection_risk=detection_risk,
                blue_alert_level=blue_alert,
            )
            
            # Get agent's proposed action
            proposed_action, confidence = self._get_agent_proposal(agent, state, phase)
            
            # Force low confidence if stuck to trigger mentor
            if force_mentor:
                confidence = 0.1  # Force mentor call
            
            # Consult coach
            decision = coach.decide(ctx, proposed_action, confidence)
            
            # Track action for stuck detection
            self._record_action(agent_name, decision.chosen_action)
            
            # Record result
            result = AgentStepResult(
                agent_name=agent_name,
                proposed_action=proposed_action,
                chosen_action=decision.chosen_action,
                mentor_call=decision.mentor_call,
                model_used=decision.model_used,
                mentor_guidance=decision.mentor_guidance,
                mentor_delta=decision.mentor_delta,
                confidence=decision.confidence,
                skill_ids=[s.get("id", "") for s in decision.skill_cards],
            )
            agent_results.append(result)
            
            # Collect executable actions
            if agent_name == "RedAgent":
                red_action = decision.chosen_action
            elif agent_name == "BlueAgent":
                blue_action = decision.chosen_action
        
        # Execute environment step with Red + Blue actions
        env_reward, done = self._execute_env_step(red_action, blue_action)
        
        # Record reward for stuck detection
        self.record_step_reward(env_reward)
        
        # Log traces for all agents
        for result in agent_results:
            self._log_step_trace(
                episode_id=episode_id,
                step=step,
                phase=phase,
                result=result,
                global_reward=env_reward,
                done=done,
                state_digest=state_digest,
            )
        
        return agent_results, env_reward, done
    
    def _get_agent_proposal(
        self,
        agent: Any,
        state: Dict[str, Any],
        phase: str,
    ) -> Tuple[str, float]:
        """Get proposed action and confidence from agent."""
        try:
            # Try different method signatures
            if hasattr(agent, 'propose_action'):
                result = agent.propose_action(state, phase)
                if isinstance(result, tuple):
                    return result[0], result[1] if len(result) > 1 else 0.5
                return str(result), 0.5
            
            if hasattr(agent, 'select_action'):
                action = agent.select_action(state)
                return str(action), 0.5
            
            if hasattr(agent, 'get_action'):
                action = agent.get_action(state)
                return str(action), 0.5
            
            # Fallback: generate placeholder based on agent type
            agent_name = getattr(agent, 'agent_id', 'Unknown')
            return self._get_placeholder_action(agent_name, phase, state), 0.4
            
        except Exception as e:
            logger.debug(f"Agent proposal failed: {e}")
            agent_name = getattr(agent, 'agent_id', 'Unknown')
            return self._get_placeholder_action(agent_name, phase, state), 0.3
    
    def _get_placeholder_action(
        self,
        agent_name: str,
        phase: str,
        state: Dict[str, Any],
    ) -> str:
        """Generate placeholder action for agent."""
        target = state.get("target_ip", "10.10.10.10") if isinstance(state, dict) else "10.10.10.10"
        
        placeholders = {
            "ScoutAgent": f"nmap -sV -T2 {target}",
            "RedAgent": f"gobuster dir -u http://{target}" if phase == "recon" else f"exploit {target}",
            "BlueAgent": "monitor_network" if phase == "recon" else "block_suspicious_ip",
            "OrionAgent": f"Suggest: continue {phase} with caution",
            "ShadowAgent": f"Remember: {phase} step completed",
        }
        
        return placeholders.get(agent_name, "noop")
    
    def _execute_env_step(
        self,
        red_action: Optional[str],
        blue_action: Optional[str],
    ) -> Tuple[float, bool]:
        """Execute environment step with agent actions."""
        try:
            # Build action dict
            agent_actions = []
            if red_action:
                agent_actions.append({
                    "agent_id": "RedAgent",
                    "command": red_action,
                })
            if blue_action:
                agent_actions.append({
                    "agent_id": "BlueAgent",
                    "command": blue_action,
                })
            
            # Execute step
            if agent_actions:
                result = self.env.step(red_action or "noop")
                
                # Handle tuple or dict return
                if isinstance(result, tuple):
                    reward = result[1] if len(result) > 1 else 0.0
                    done = result[2] if len(result) > 2 else False
                elif isinstance(result, dict):
                    reward = result.get("reward", 0.0)
                    done = result.get("done", False)
                else:
                    reward = 0.0
                    done = False
                
                return reward, done
            
            return 0.0, False
            
        except Exception as e:
            logger.debug(f"Env step failed: {e}")
            return 0.0, False
    
    def _log_step_trace(
        self,
        episode_id: str,
        step: int,
        phase: str,
        result: AgentStepResult,
        global_reward: float,
        done: bool,
        state_digest: str,
    ):
        """Log StepTrace for an agent."""
        if not self.trace_writer:
            return
        
        from core.tracing import StepTrace
        
        trace = StepTrace(
            episode_id=episode_id,
            step=step,
            agent=result.agent_name,
            phase=phase,
            proposed_action=result.proposed_action,
            chosen_action=result.chosen_action,
            mentor_call=result.mentor_call,
            model_used=result.model_used,
            reward=global_reward,
            done=done,
            mentor_response=result.mentor_guidance,
            confidence=result.confidence,
        )
        
        # Add optional fields
        trace_dict = trace.to_dict()
        trace_dict["mentor_guidance"] = result.mentor_guidance
        trace_dict["mentor_delta"] = result.mentor_delta
        trace_dict["skill_ids_emitted"] = result.skill_ids
        trace_dict["global_reward"] = global_reward
        trace_dict["env_phase"] = phase
        trace_dict["state_digest"] = state_digest
        
        self.trace_writer.log_step(trace)
    
    def _compute_episode_metrics(
        self,
        step_results: List[List[AgentStepResult]],
        total_reward: float,
        done: bool,
    ) -> Dict[str, Any]:
        """Compute metrics for the episode."""
        metrics = {
            "total_steps": len(step_results),
            "total_reward": total_reward,
            "done": done,
            "agents": {},
        }
        
        # Per-agent metrics
        for agent_name in self.AGENT_ORDER:
            agent_steps = [
                r for results in step_results
                for r in results
                if r.agent_name == agent_name
            ]
            
            if agent_steps:
                metrics["agents"][agent_name] = {
                    "steps": len(agent_steps),
                    "mentor_calls": sum(1 for r in agent_steps if r.mentor_call),
                    "actions_changed": sum(1 for r in agent_steps if r.mentor_delta == "changed"),
                    "avg_confidence": sum(r.confidence for r in agent_steps) / len(agent_steps),
                }
        
        # Total mentor calls
        total_mentor_calls = sum(
            m["mentor_calls"] for m in metrics["agents"].values()
        )
        metrics["total_mentor_calls"] = total_mentor_calls
        
        return metrics
    
    def _default_state(self) -> Dict[str, Any]:
        """Get default state when environment doesn't provide one."""
        return {
            "phase": "recon",
            "target_ip": "10.10.10.10",
            "open_ports": [],
            "detection_risk": 0.0,
            "blue_team_alert": 0.0,
            "history": [],
        }
    
    def get_all_agent_stats(self) -> Dict[str, Any]:
        """Get statistics from all coaches."""
        return {
            name: coach.get_stats()
            for name, coach in self.coaches.items()
        }
    
    def _record_action(self, agent_name: str, action: str):
        """Record an action for stuck detection."""
        if agent_name not in self.action_history:
            self.action_history[agent_name] = []
        
        # Keep last N actions (threshold + 1)
        max_history = self.config.stuck_threshold + 1
        self.action_history[agent_name].append(action)
        if len(self.action_history[agent_name]) > max_history:
            self.action_history[agent_name] = self.action_history[agent_name][-max_history:]
    
    def _check_if_stuck(self, agent_name: str) -> bool:
        """
        Check if agent is stuck (repeating same action).
        
        Stuck conditions:
        - Same action repeated N times (config.stuck_threshold)
        - N consecutive negative rewards (config.stuck_negative_streak)
        """
        # Check repeated actions
        if agent_name in self.action_history:
            recent = self.action_history[agent_name]
            if len(recent) >= self.config.stuck_threshold:
                # Check if last N actions are identical
                last_n = recent[-self.config.stuck_threshold:]
                if len(set(last_n)) == 1:
                    if agent_name not in self.stuck_agents:
                        self.stuck_agents.add(agent_name)
                        logger.warning(f"Agent {agent_name} is STUCK: repeated '{last_n[0][:30]}...' x{len(last_n)}")
                    return True
        
        # Check negative reward streak (global, affects all agents)
        if len(self.reward_history) >= self.config.stuck_negative_streak:
            recent_rewards = self.reward_history[-self.config.stuck_negative_streak:]
            if all(r < 0 for r in recent_rewards):
                if agent_name not in self.stuck_agents:
                    logger.warning(f"Agent {agent_name} in negative reward streak ({len(recent_rewards)} steps)")
                return True
        
        # Not stuck - remove from stuck set if was there
        self.stuck_agents.discard(agent_name)
        return False
    
    def record_step_reward(self, reward: float):
        """Record step reward for stuck detection."""
        self.reward_history.append(reward)
        # Keep last N rewards
        max_history = self.config.stuck_negative_streak + 5
        if len(self.reward_history) > max_history:
            self.reward_history = self.reward_history[-max_history:]
