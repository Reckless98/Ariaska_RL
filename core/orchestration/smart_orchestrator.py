"""
Smart Orchestrator - Enhanced orchestrator with intelligent command generation.

This orchestrator integrates:
- SmartCoach for command registry validation
- Attack context for rich state representation  
- Smart reward calculation for better learning
- Phase progression tracking
- LiveDashboard for real-time visibility
"""

import os
import time
import logging
import hashlib
import torch
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field

from enum import Enum
from core.commands.command_registry import (
    AttackPhase,
    get_phase_from_state,
    COMMAND_REGISTRY,
)
from core.llm.smart_mentor import AttackContext


class TerminationReason(Enum):
    """Reasons for episode termination - Phase 0.1."""
    MAX_STEPS = "max_steps"
    GOAL_REACHED = "goal_reached"
    STUCK_ABORT = "stuck_abort"  # Too many forced-novel failures
    ENV_DONE = "env_done"
    ERROR = "error"
from core.llm.reward_calculator import SmartRewardCalculator, RewardBreakdown
from core.training.smart_coach import SmartCoach, SmartDecisionResult, SmartStepContext
from core.observability import LiveDashboard, DashboardConfig

if TYPE_CHECKING:
    from core.gpt_manager import GPTManager
    from core.environment.cyber_environment import CyberEnvironment
    from core.tracing import TraceWriter
    from core.postmortem import SkillLibrary

logger = logging.getLogger("ariaska.smart_orchestrator")


@dataclass
class SmartOrchestratorConfig:
    """Configuration for the smart orchestrator."""
    
    # Agent activation
    enable_scout: bool = True
    enable_red: bool = True
    enable_blue: bool = True
    enable_orion: bool = True
    enable_shadow: bool = True
    
    # Smart mentor settings
    model: str = "gpt-4o-mini"
    mentor_mode: str = "anneal"
    mentor_warmup_episodes: int = 1
    mentor_min_rate: float = 0.15
    mentor_max_rate: float = 1.0
    
    # Stuck detection (legacy)
    stuck_threshold: int = 3
    stuck_negative_streak: int = 5
    stuck_force_mentor: bool = True
    stuck_force_exploration: bool = True
    
    # Phase 0.1: Enhanced stuck-escape config knobs
    stuck_repeat_threshold: int = 5  # Consecutive repeats before forcing novel action
    stuck_history_k: int = 15  # Look back K actions for tag overlap calculation
    stuck_tag_overlap_threshold: float = 0.8  # Mask actions with >= this tag overlap
    stuck_forced_abort_threshold: int = 10  # Terminate episode after N forced-novel failures
    
    # Execution - reduced for faster feedback loops
    max_steps_per_episode: int = 50
    
    # Logging
    mentor_log_dir: str = "traces"
    
    # Attack context
    default_target: str = "10.10.10.10"
    default_difficulty: str = "medium"
    default_platform: str = "unknown"
    
    # Dashboard settings
    dashboard_enabled: bool = True
    dashboard_mode: str = "live"  # "off", "summary", "live"
    dashboard_watch_rate: float = 1.0


@dataclass
class SmartStepResult:
    """Result from a smart step with full context."""
    agent_name: str
    decision: SmartDecisionResult
    reward_breakdown: Optional[RewardBreakdown] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent_name,
            "command": self.decision.command,
            "template": self.decision.template_name,
            "params": self.decision.params,
            "mentor_call": self.decision.mentor_call,
            "model_used": self.decision.model_used,
            "reasoning": self.decision.mentor_reasoning,
            "phase": self.decision.phase.name,
            "confidence": self.decision.confidence,
            "reward": self.reward_breakdown.total if self.reward_breakdown else 0.0,
        }


class SmartOrchestrator:
    """
    Enhanced orchestrator with intelligent command generation.
    
    Key improvements over base Orchestrator:
    1. Uses SmartCoach for validated, structured commands
    2. Maintains shared AttackContext for all agents
    3. Uses SmartRewardCalculator for phase-aware rewards
    4. Tracks command effectiveness and learns over time
    5. INTELLIGENT AGENT SEQUENCING based on attack phase
    
    Can be used as drop-in replacement for Orchestrator.
    """
    
    # Default order (can be overridden by phase-based logic)
    AGENT_ORDER = ["ScoutAgent", "RedAgent", "BlueAgent", "OrionAgent", "ShadowAgent"]
    
    # Phase-optimized agent ordering for maximum synergy
    PHASE_AGENT_ORDER = {
        # RECON: Scout leads, Shadow for stealth, then Red to probe
        "RECON": ["ScoutAgent", "ShadowAgent", "RedAgent", "OrionAgent", "BlueAgent"],
        
        # ENUMERATION: Scout continues, Red probes, Shadow stealthily checks
        "ENUMERATION": ["ScoutAgent", "RedAgent", "ShadowAgent", "OrionAgent", "BlueAgent"],
        
        # EXPLOITATION: Red leads for attacks, Blue for defense, Orion coordinates
        "EXPLOITATION": ["RedAgent", "OrionAgent", "ShadowAgent", "ScoutAgent", "BlueAgent"],
        
        # PRIVESC: Red exploits, Shadow for persistence, Orion for strategy
        "PRIVILEGE_ESCALATION": ["RedAgent", "ShadowAgent", "OrionAgent", "BlueAgent", "ScoutAgent"],
        
        # LATERAL: Shadow leads stealth movement, Red assists, Orion coordinates
        "LATERAL_MOVEMENT": ["ShadowAgent", "RedAgent", "OrionAgent", "BlueAgent", "ScoutAgent"],
        
        # POST_EX: Shadow for persistence/exfil, Red for cleanup
        "POST_EXPLOITATION": ["ShadowAgent", "RedAgent", "OrionAgent", "BlueAgent", "ScoutAgent"],
        
        # EXFIL: Shadow leads stealth extraction, Blue monitors defense
        "EXFILTRATION": ["ShadowAgent", "RedAgent", "OrionAgent", "BlueAgent", "ScoutAgent"],
    }
    
    def get_optimal_agent_order(self, phase: str = "RECON") -> List[str]:
        """
        Get the optimal agent execution order for the current attack phase.
        
        Different phases need different agent leadership:
        - RECON: Scout leads to gather intel, Shadow for stealth recon
        - EXPLOITATION: Red leads the attack, Orion coordinates
        - POST_EX: Shadow leads for stealth persistence/exfil
        
        Returns:
            List of agent names in optimal execution order
        """
        phase_upper = phase.upper().replace(" ", "_")
        
        # Get phase-specific order or default
        optimal_order = self.PHASE_AGENT_ORDER.get(phase_upper, self.AGENT_ORDER)
        
        # Filter to only include enabled agents
        return [agent for agent in optimal_order if agent in self.agents]
    
    def __init__(
        self,
        env: "CyberEnvironment",
        gpt_manager: "GPTManager",
        trace_writer: Optional["TraceWriter"] = None,
        skill_library: Optional["SkillLibrary"] = None,
        config: Optional[SmartOrchestratorConfig] = None,
        verbosity: str = "standard",
    ):
        self.env = env
        self.gpt_manager = gpt_manager
        self.trace_writer = trace_writer
        self.skill_library = skill_library
        self.config = config or SmartOrchestratorConfig()
        self.verbosity = verbosity
        
        self.run_dir: Optional[str] = None
        
        # Initialize agents
        self.agents: Dict[str, Any] = {}
        self._init_agents()
        
        # Initialize smart coaches
        self.coaches: Dict[str, SmartCoach] = {}
        self._init_smart_coaches()
        
        # Shared attack context (all agents see same state)
        self.attack_context: Optional[AttackContext] = None
        
        # Global reward calculator (for episode-level tracking)
        self.global_reward_calc = SmartRewardCalculator()
        
        # Episode tracking
        self.current_episode = 0
        self.current_step = 0
        self.total_episodes = 0
        self.run_id: Optional[str] = None
        self.start_time: Optional[float] = None
        
        # Enhanced stuck detection
        self.action_history: Dict[str, List[str]] = {}
        self.stuck_agents: set = set()
        
        # Phase 0.1: Per-agent stuck tracking
        self.repeat_stuck_count: Dict[str, int] = {}  # Consecutive repeats per agent
        self.deep_stuck_count: Dict[str, int] = {}  # Forced-novel failures per agent
        self.forced_novel_count: Dict[str, int] = {}  # Successful forced-novel actions per agent
        self.phase_progressed_this_episode: bool = False  # True if phase advanced
        self._phase_start_step: Dict[str, int] = {}  # Track when each phase started
        self.episode_termination_reason: TerminationReason = TerminationReason.MAX_STEPS
        self.previous_discoveries: Dict[str, Any] = {}  # For discoveries_delta calculation
        
        # ─── PHASE 4: Cross-Agent Discovery Board ────────────────────
        # Shared state that all agents can read. Populated after each
        # agent step so that later agents benefit from earlier agents'
        # discoveries within the same step.
        self.discovery_board: Dict[str, Any] = {
            "ports": set(),
            "services": set(),
            "credentials": set(),
            "vulns": set(),
            "shells": set(),
            "users": set(),
            "web_paths": set(),
            "phase": "RECON",
            "flags_set": set(),
        }
        # Stagnation tracking per agent
        self._steps_without_discoveries: Dict[str, int] = {}
        
        # Initialize LiveDashboard for real-time visibility
        self.dashboard = self._init_dashboard()
        
        # ─── PHASE 3: PPO Agent Integration ──────────────────────────
        # Creates a PPO actor-critic that runs alongside the existing
        # SmartCoach pipeline. Collects trajectories during episodes
        # and updates after each episode.
        self.ppo_agent = None
        self._ppo_trajectory: List[Dict] = []  # Per-episode trajectory
        try:
            from core.algorithms.ppo_agent import PPOAgent, PPOConfig
            ppo_config = PPOConfig(
                state_dim=512,
                action_dim=5,  # recon, enumeration, exploit, privesc, exfiltrate
                hidden_dims=[512, 512, 256],
                clip_epsilon=0.2,
                gamma=0.99,
                gae_lambda=0.95,
                learning_rate=3e-4,
                epochs_per_update=3,
                minibatch_size=8,
                entropy_coef=0.01,
                rollout_size=32,
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.ppo_agent = PPOAgent(config=ppo_config, device=device)
            logger.info("PHASE 3: PPO Actor-Critic initialized for Red agent")
        except Exception as e:
            logger.warning(f"PHASE 3: PPO init failed (falling back to DQN): {e}")
        
        logger.info(f"SmartOrchestrator initialized with {len(self.agents)} agents")
    
    # =========================================================================
    # PHASE 2A: Smart Agent Activation Schedule
    # =========================================================================
    # Not all agents need to run every step. Phase-based activation saves
    # API calls by only activating agents when they're useful.
    # Value = run every Nth step. 1 = every step, 2 = every other step.
    # Red always runs (core attacker). Blue/Shadow now run more often
    # to ensure proper token usage and defensive coverage.
    AGENT_ACTIVATION_SCHEDULE = {
        # RECON: Scout leads, Red every step, Blue/Shadow observe every 2 steps
        "RECON": {
            "ScoutAgent": 1, "RedAgent": 1, "ShadowAgent": 2,
            "OrionAgent": 3, "BlueAgent": 2,
        },
        "ENUMERATION": {
            "ScoutAgent": 2, "RedAgent": 1, "ShadowAgent": 2,
            "OrionAgent": 3, "BlueAgent": 2,
        },
        "EXPLOITATION": {
            "ScoutAgent": 3, "RedAgent": 1, "ShadowAgent": 2,
            "OrionAgent": 2, "BlueAgent": 1,
        },
        "PRIVILEGE_ESCALATION": {
            "ScoutAgent": 3, "RedAgent": 1, "ShadowAgent": 2,
            "OrionAgent": 3, "BlueAgent": 1,
        },
        "LATERAL_MOVEMENT": {
            "ScoutAgent": 3, "RedAgent": 1, "ShadowAgent": 1,
            "OrionAgent": 2, "BlueAgent": 1,
        },
        "POST_EXPLOITATION": {
            "ScoutAgent": 3, "RedAgent": 1, "ShadowAgent": 1,
            "OrionAgent": 2, "BlueAgent": 1,
        },
        "EXFILTRATION": {
            "ScoutAgent": 3, "RedAgent": 1, "ShadowAgent": 1,
            "OrionAgent": 3, "BlueAgent": 1,
        },
    }
    
    def _should_activate(self, agent_name: str, step: int, phase: str) -> bool:
        """
        Determine if an agent should be activated this step based on phase.
        
        Args:
            agent_name: Name of the agent
            step: Current step number (0-indexed from orchestrator)
            phase: Current attack phase (e.g. "RECON", "EXPLOITATION")
            
        Returns:
            True if agent should run this step
        """
        phase_upper = phase.upper().replace(" ", "_")
        schedule = self.AGENT_ACTIVATION_SCHEDULE.get(phase_upper, {})
        frequency = schedule.get(agent_name, 1)  # Default: every step
        # Use (step + 1) so step 0 behaves like step 1 (not always-activate)
        return (step + 1) % frequency == 0
    
    def _init_dashboard(self) -> LiveDashboard:
        """Initialize the live dashboard for training visibility."""
        dash_config = DashboardConfig(
            enabled=self.config.dashboard_enabled,
            mode=self.config.dashboard_mode,
            watch_rate=self.config.dashboard_watch_rate,
            show_reward_breakdown=True,
            max_action_width=40,
        )
        dashboard = LiveDashboard(config=dash_config)
        logger.debug("LiveDashboard initialized")
        return dashboard
    
    def _init_agents(self):
        """Initialize all agents."""
        from core.multiagent.memory_router import MemoryRouter
        
        memory_router = MemoryRouter()
        
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
    
    def _init_smart_coaches(self):
        """Initialize SmartCoach for each agent."""
        from core.training.mentor_policy import MentorPolicy, MentorPolicyConfig
        
        policy_config = MentorPolicyConfig(
            mode=self.config.mentor_mode,
            warmup_episodes=self.config.mentor_warmup_episodes,
            min_mentor_rate=self.config.mentor_min_rate,
            max_mentor_rate=self.config.mentor_max_rate,
        )
        
        for agent_name in self.agents.keys():
            policy = MentorPolicy(policy_config)
            
            self.coaches[agent_name] = SmartCoach(
                agent_name=agent_name,
                gpt_manager=self.gpt_manager,
                mentor_policy=policy,
                skill_library=self.skill_library,
                trace_writer=self.trace_writer,
                mentor_log_path=None,
                model=self.config.model,
            )
        
        logger.info(f"Initialized smart coaches: {list(self.coaches.keys())}")
    
    def set_run_dir(self, run_dir: str):
        """Set the run directory for logs."""
        self.run_dir = run_dir
        mentor_log_path = os.path.join(run_dir, "smart_mentor.jsonl")
        
        for coach in self.coaches.values():
            coach.mentor_log_path = mentor_log_path
    
    def init_attack(
        self,
        target: str,
        difficulty: str = "medium",
        platform: str = "unknown",
    ) -> AttackContext:
        """
        Initialize attack context for a new target.
        
        Args:
            target: Target IP or hostname
            difficulty: Target difficulty (easy, medium, hard, insane)
            platform: Target platform (linux, windows, unknown)
            
        Returns:
            Shared AttackContext
        """
        self.attack_context = AttackContext(
            target=target,
            difficulty=difficulty,
            platform=platform,
            current_phase=AttackPhase.RECON,
        )
        
        # Share context with all coaches
        for coach in self.coaches.values():
            coach.attack_context = self.attack_context
        
        return self.attack_context
    
    def run_episode(
        self,
        episode_id: str,
        episode_number: int,
        max_steps: Optional[int] = None,
        target: Optional[str] = None,
        difficulty: Optional[str] = None,
        platform: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run a complete episode with smart command generation.
        
        Args:
            episode_id: Unique episode identifier
            episode_number: Episode number
            max_steps: Maximum steps (default from config)
            target: Target IP (optional, uses env or default)
            difficulty: Target difficulty
            platform: Target platform
            
        Returns:
            Episode metrics with detailed reward breakdown
        """
        max_steps = max_steps or self.config.max_steps_per_episode
        self.current_episode = episode_number
        
        # Reset token budgets
        if self.gpt_manager:
            self.gpt_manager.reset_episode(episode_id=episode_number)
        
        # PHASE 2A: Reset per-agent GPT call counters
        for agent in self.agents.values():
            if hasattr(agent, 'gpt_calls_this_episode'):
                agent.gpt_calls_this_episode = 0
        
        # Reset coaches and reward calculator
        for coach in self.coaches.values():
            coach.reset_episode(episode_number)
        self.global_reward_calc.reset()
        
        # PHASE 3: Clear PPO trajectory for new episode
        self._ppo_trajectory = []
        
        # Phase 4: Reset discovery board
        self.discovery_board = {
            "ports": set(), "services": set(), "credentials": set(),
            "vulns": set(), "shells": set(), "users": set(),
            "web_paths": set(), "phase": "RECON", "flags_set": set(),
        }
        
        # Reset stuck detection
        self.action_history.clear()
        self.stuck_agents.clear()
        
        # Phase 0.1: Reset per-agent stuck tracking
        self.repeat_stuck_count = {agent: 0 for agent in self.agents}
        self.deep_stuck_count = {agent: 0 for agent in self.agents}
        self.forced_novel_count = {agent: 0 for agent in self.agents}
        self._steps_without_discoveries = {agent: 0 for agent in self.agents}  # Stagnation counter
        self.phase_progressed_this_episode = False
        self._phase_start_step = {"RECON": 0}  # Reset phase timing
        self.episode_termination_reason = TerminationReason.MAX_STEPS
        self.previous_discoveries = {}
        
        # Reset dashboard for new episode
        self.dashboard.reset_episode()
        self.dashboard.current_episode = episode_number
        
        # Reset environment
        state = self.env.reset()
        if not state:
            state = self._default_state()
        
        # Initialize attack context
        target = target or state.get("target_ip", self.config.default_target)
        difficulty = difficulty or self.config.default_difficulty
        platform = platform or state.get("os", self.config.default_platform)
        self.init_attack(target, difficulty, platform)
        
        # Update context from initial state
        self._update_context_from_state(state)
        
        # Update dashboard with initial env state
        self.dashboard.update_env_snapshot({
            "target_ip": target,
            "phase": self.attack_context.current_phase.name.lower(),
            "discovered_ports": list(self.attack_context.discoveries.get("open_port", [])),
            "discovered_services": {s: None for s in self.attack_context.services_found},
        })
        
        # Episode tracking
        episode_reward = 0.0
        step_results: List[List[SmartStepResult]] = []
        phase_progression: List[str] = [self.attack_context.current_phase.name]
        done = False
        total_mentor_calls = 0
        
        for step in range(max_steps):
            self.current_step = step
            
            # Run all agents
            step_agent_results, env_reward, new_state, done = self._run_step(
                episode_id=episode_id,
                step=step,
                state=state,
            )
            
            step_results.append(step_agent_results)
            episode_reward += env_reward
            
            # Record step to dashboard
            dashboard_results = []
            for result in step_agent_results:
                total_mentor_calls += 1 if result.decision.mentor_call else 0
                
                # Get tokens from decision (now properly tracked)
                tokens_for_step = getattr(result.decision, 'tokens_used', 0)
                
                dashboard_results.append({
                    "agent": result.agent_name,
                    "agent_name": result.agent_name,
                    "chosen_action": result.decision.command,
                    "proposed_action": result.decision.command,
                    "mentor_call": result.decision.mentor_call,
                    "mentor_success": result.decision.mentor_call,
                    "model_used": result.decision.model_used,
                    "confidence": result.decision.confidence,
                    "mentor_delta": result.decision.mentor_delta,
                    "mentor_reasoning": result.decision.mentor_reasoning or "",  # Pass reasoning
                    "tokens_used": tokens_for_step,
                    "command_output": result.decision.command_output or "",  # Simulated output
                })
            
            # Get reward breakdown for display - include reasoning from each agent
            reward_breakdown_dict = None
            if step_agent_results:
                # Collect reasons from all agents
                agent_reasons = []
                for result in step_agent_results:
                    if result.decision.mentor_reasoning:
                        agent_reasons.append(f"{result.agent_name[:6]}: {result.decision.mentor_reasoning[:30]}")
                
                if step_agent_results[0].reward_breakdown:
                    rb = step_agent_results[0].reward_breakdown
                    reward_breakdown_dict = {
                        "base": rb.base_reward,
                        "novelty_bonus": rb.novelty_bonus,
                        "redundancy_penalty": rb.redundancy_penalty,
                        "phase_bonus": rb.phase_advance_bonus,
                        "total": rb.total,
                        "reason": " | ".join(agent_reasons[:3]) if agent_reasons else f"Phase: {self.attack_context.current_phase.name}",
                    }
                else:
                    reward_breakdown_dict = {
                        "reason": " | ".join(agent_reasons[:3]) if agent_reasons else "",
                    }
            
            # Record to dashboard
            self.dashboard.record_step(
                step=step,
                phase=self.attack_context.current_phase.name.lower(),
                agent_results=dashboard_results,
                global_reward=env_reward,
                done=done,
                reward_breakdown=reward_breakdown_dict,
            )
            
            # Print step table for live visibility
            self.dashboard.print_step_table(step)
            
            # Update attack context from new state
            if new_state:
                self._update_context_from_state(new_state)
                state = new_state
                
                # Update dashboard env snapshot
                self.dashboard.update_env_snapshot({
                    "target_ip": self.attack_context.target,
                    "phase": self.attack_context.current_phase.name.lower(),
                    "discovered_ports": list(self.attack_context.discoveries.get("open_port", [])),
                    "discovered_services": {s: None for s in self.attack_context.services_found},
                    "root_achieved": self.attack_context.state_flags.get("root_shell_obtained", False),
                    "credentials": ["found"] if self.attack_context.state_flags.get("credentials_known") else [],
                })
            
            # Track phase progression
            current_phase = self.attack_context.current_phase.name
            if current_phase != phase_progression[-1]:
                phase_progression.append(current_phase)
                self.phase_progressed_this_episode = True  # Phase 0.1
                logger.info(f"Phase advanced: {phase_progression[-2]} → {current_phase}")
                self.dashboard.add_event(
                    "phase_change",
                    f"Advanced to {current_phase}",
                    agent="system"
                )
            
            # =================================================================
            # PHASE 4: TIME-BASED AUTO-ADVANCEMENT REMOVED
            # Previously auto-set credentials_known/shell_obtained/admin_access
            # after N steps in each phase. This defeated learning — the agent
            # never needed to EARN phase transitions through actual commands.
            # Now the agent must discover or exploit its way forward.
            # =================================================================
            # (Phase 2A code removed — agent must earn advancement through discoveries)
            
            # Track phase start steps
            new_phase = self.attack_context.current_phase.name
            if new_phase != current_phase:
                self._phase_start_step[new_phase] = step
            
            # =========================================================================
            # PHASE 0.1: CHECK STUCK_ABORT TERMINATION
            # =========================================================================
            total_deep_stuck = sum(self.deep_stuck_count.values())
            if total_deep_stuck >= self.config.stuck_forced_abort_threshold:
                self.episode_termination_reason = TerminationReason.STUCK_ABORT
                logger.warning(
                    f"[STUCK_ABORT] Episode terminated: "
                    f"deep_stuck_count={self.deep_stuck_count} "
                    f"threshold={self.config.stuck_forced_abort_threshold}"
                )
                self.dashboard.add_event(
                    "stuck_abort",
                    f"Episode aborted: too many stuck failures",
                    agent="system"
                )
                done = True
            
            if done:
                if self.episode_termination_reason == TerminationReason.MAX_STEPS:
                    self.episode_termination_reason = TerminationReason.ENV_DONE
                break
        
        # Set termination reason if loop exhausted
        if not done:
            self.episode_termination_reason = TerminationReason.MAX_STEPS
        
        # Print episode summary
        self.dashboard.print_episode_summary(
            episode=episode_number,
            total_reward=episode_reward,
            total_steps=len(step_results),
            mentor_calls=total_mentor_calls,
        )
        
        # Compute metrics
        metrics = self._compute_episode_metrics(
            step_results, episode_reward, done, phase_progression
        )
        
        # ─── PHASE 4: Per-Coach PPO Updates ─────────────────────────
        # Each SmartCoach has its own PPOAgent; trigger update at end of episode
        ppo_updates_fired = 0
        ppo_total_policy_loss = 0.0
        ppo_total_value_loss = 0.0
        ppo_total_entropy = 0.0
        for coach_name, coach in self.coaches.items():
            if hasattr(coach, 'end_episode_ppo'):
                try:
                    ppo_metrics = coach.end_episode_ppo(done=done)
                    if ppo_metrics:
                        ppo_updates_fired += 1
                        ppo_total_policy_loss += ppo_metrics.get("policy_loss", 0.0)
                        ppo_total_value_loss += ppo_metrics.get("value_loss", 0.0)
                        ppo_total_entropy += ppo_metrics.get("entropy", 0.0)
                        metrics[f"ppo_{coach_name}_policy_loss"] = ppo_metrics.get("policy_loss", 0.0)
                        metrics[f"ppo_{coach_name}_value_loss"] = ppo_metrics.get("value_loss", 0.0)
                        metrics[f"ppo_{coach_name}_entropy"] = ppo_metrics.get("entropy", 0.0)
                except Exception as e:
                    logger.warning(f"PPO update error for {coach_name}: {e}")

        # Aggregate PPO metrics
        metrics["ppo_updates_fired"] = ppo_updates_fired
        if ppo_updates_fired > 0:
            metrics["ppo_avg_policy_loss"] = ppo_total_policy_loss / ppo_updates_fired
            metrics["ppo_avg_value_loss"] = ppo_total_value_loss / ppo_updates_fired
            metrics["ppo_avg_entropy"] = ppo_total_entropy / ppo_updates_fired
        
        # Count decision sources across all step results
        source_counts = {"ppo": 0, "playbook": 0, "registry": 0, "anti_repeat": 0, "other": 0}
        for sr in step_results:
            for ar in sr:
                src = getattr(ar.decision, "source", "unknown") if ar.decision else "unknown"
                # Map "unknown" → "registry" (default path)
                if src == "unknown":
                    src = "registry"
                if src in source_counts:
                    source_counts[src] += 1
                else:
                    source_counts["other"] += 1
        metrics["decisions_ppo"] = source_counts["ppo"]
        metrics["decisions_playbook"] = source_counts["playbook"]
        metrics["decisions_registry"] = source_counts["registry"]
        metrics["decisions_anti_repeat"] = source_counts["anti_repeat"]

        # Legacy global PPO (kept for backward compat, no-op if trajectory empty)
        if self.ppo_agent and self._ppo_trajectory:
            try:
                for t in self._ppo_trajectory:
                    self.ppo_agent.store_transition(
                        state=t["state"],
                        action=t["action"],
                        log_prob=t["log_prob"],
                        reward=t["reward"],
                        value=t["value"],
                        done=t["done"],
                    )
                last_value = self._ppo_trajectory[-1]["value"] if self._ppo_trajectory[-1]["done"] else 0.0
                ppo_metrics = self.ppo_agent.update(last_value=last_value)
                if ppo_metrics:
                    metrics["ppo_policy_loss"] = ppo_metrics.get("policy_loss", 0.0)
                    metrics["ppo_value_loss"] = ppo_metrics.get("value_loss", 0.0)
            except Exception as e:
                logger.warning(f"Legacy PPO update error: {e}")
            finally:
                self._ppo_trajectory.clear()
        
        return metrics
    
    # =========================================================================
    # PPO Checkpoint Persistence
    # =========================================================================
    
    def save_ppo_checkpoints(self, directory: str = "models/ppo_checkpoints"):
        """Save all per-coach PPO checkpoints for persistence across runs.
        
        Args:
            directory: Directory to save checkpoints into.
        """
        import os
        os.makedirs(directory, exist_ok=True)
        saved = 0
        for coach_name, coach in self.coaches.items():
            if hasattr(coach, 'ppo_agent') and coach.ppo_agent is not None:
                path = os.path.join(directory, f"ppo_{coach_name}.pt")
                try:
                    coach.ppo_agent.save(path)
                    saved += 1
                    logger.info(f"Saved PPO checkpoint: {path}")
                except Exception as e:
                    logger.warning(f"Failed to save PPO for {coach_name}: {e}")
        logger.info(f"Saved {saved} PPO checkpoints to {directory}")
    
    def load_ppo_checkpoints(self, directory: str = "models/ppo_checkpoints"):
        """Load per-coach PPO checkpoints from a previous run.
        
        Args:
            directory: Directory to load checkpoints from.
        """
        import os
        if not os.path.isdir(directory):
            logger.info(f"No PPO checkpoint directory found: {directory}")
            return
        loaded = 0
        for coach_name, coach in self.coaches.items():
            if hasattr(coach, 'ppo_agent') and coach.ppo_agent is not None:
                path = os.path.join(directory, f"ppo_{coach_name}.pt")
                if os.path.isfile(path):
                    try:
                        coach.ppo_agent.load(path)
                        loaded += 1
                        logger.info(f"Loaded PPO checkpoint: {path} (updates={coach.ppo_agent.updates_done})")
                    except Exception as e:
                        logger.warning(f"Failed to load PPO for {coach_name}: {e}")
        logger.info(f"Loaded {loaded} PPO checkpoints from {directory}")
    
    def _run_step(
        self,
        episode_id: str,
        step: int,
        state: Dict[str, Any],
    ) -> Tuple[List[SmartStepResult], float, Dict[str, Any], bool]:
        """
        Run a single step with all agents using smart coaching.
        
        Each agent picks DIFFERENT commands based on their role.
        
        Returns:
            (agent_results, reward, new_state, done)
        """
        agent_results: List[SmartStepResult] = []
        
        # Build step context
        ctx = self.attack_context
        
        # Actions to execute
        red_action = None
        blue_action = None
        
        # Clear used commands for this step (deduplication)
        step_used_commands: set = set()
        for coach in self.coaches.values():
            if hasattr(coach, 'clear_step_commands'):
                coach.clear_step_commands()
        
        # Process each agent IN ORDER - each sees what previous agents picked
        # Use PHASE-OPTIMIZED order for maximum synergy
        current_phase = self.attack_context.current_phase.name if self.attack_context else "RECON"
        agent_order = self.get_optimal_agent_order(current_phase)
        
        for agent_name in agent_order:
            if agent_name not in self.agents or agent_name not in self.coaches:
                continue
            
            # PHASE 2A: Smart activation — skip agents that don't need to run this step
            if not self._should_activate(agent_name, step, current_phase):
                continue
            
            agent = self.agents[agent_name]
            coach = self.coaches[agent_name]
            
            # Share used commands with this coach
            if hasattr(coach, 'step_used_commands'):
                coach.step_used_commands = step_used_commands
            
            # =========================================================================
            # PHASE 0.1: ENHANCED STUCK DETECTION
            # =========================================================================
            
            # Check legacy stuck (for backward compat)
            is_legacy_stuck = self._check_if_stuck(agent_name)
            
            # Check repeat-stuck (Phase 0.1: consecutive same actions OR stagnation)
            is_repeat_stuck, repeat_count = self._check_repeat_stuck(agent_name)
            
            # Check deep-stuck (too many forced-novel failures)
            is_deep_stuck = self._check_deep_stuck(agent_name)
            
            # Debug log for stuck detection (every 10 steps, DEBUG level)
            if step % 10 == 0:
                stagnation = self._steps_without_discoveries.get(agent_name, 0)
                logger.debug(
                    f"[STUCK-CHECK][{agent_name}] step={step} "
                    f"repeat_stuck={is_repeat_stuck} repeat_count={repeat_count} "
                    f"stagnation={stagnation}/{self.config.stuck_repeat_threshold}"
                )
            
            # Build smart step context
            # Phase 4: Inject discovery board into state for cross-agent awareness
            enriched_state = dict(state) if isinstance(state, dict) else {}
            enriched_state["discovery_board"] = {
                k: list(v) if isinstance(v, set) else v
                for k, v in self.discovery_board.items()
            }
            step_ctx = SmartStepContext(
                episode=self.current_episode,
                step=step,
                agent_name=agent_name,
                attack_context=ctx,
                state=enriched_state,
            )
            
            # =========================================================================
            # PHASE 0.1: STUCK-ESCAPE LOGIC
            # =========================================================================
            decision = None
            
            if is_repeat_stuck:
                # Update repeat stuck counter
                self.repeat_stuck_count[agent_name] = self.repeat_stuck_count.get(agent_name, 0) + 1
                
                # Force novel action with tag-based masking
                decision = coach._force_novel_action(
                    step_ctx,
                    thresholds=[
                        self.config.stuck_tag_overlap_threshold,
                        0.6,
                        0.4,
                        0.0,
                    ],
                )
                
                # Check if forced action is same as last (deep stuck)
                last_action = self.action_history.get(agent_name, [""])[-1] if self.action_history.get(agent_name) else ""
                if decision.command == last_action:
                    self.deep_stuck_count[agent_name] = self.deep_stuck_count.get(agent_name, 0) + 1
                    logger.warning(
                        f"[DEEP-STUCK][{agent_name}] forced-novel returned same action "
                        f"count={self.deep_stuck_count[agent_name]}/{self.config.stuck_forced_abort_threshold}"
                    )
                else:
                    # Successful novel action
                    self.forced_novel_count[agent_name] = self.forced_novel_count.get(agent_name, 0) + 1
                    self.repeat_stuck_count[agent_name] = 0  # Reset repeat counter
                    
                    logger.info(
                        f"[FORCED-NOVEL][{agent_name}] "
                        f"prev={last_action[:30]}... → new={decision.command[:30]}... "
                        f"tags_recent={{...}} excluded={decision.excluded_count}"
                    )
                    
                    self.dashboard.add_event(
                        "forced_novel",
                        f"Forced: {decision.template_name}",
                        agent_name
                    )
            else:
                # Normal decision flow
                # Get agent's proposed action (for comparison)
                proposed_action, confidence = self._get_agent_proposal(agent, state)
                
                # Force low confidence if legacy stuck
                force_mentor = is_legacy_stuck and self.config.stuck_force_mentor
                if force_mentor:
                    confidence = 0.1
                
                # Get smart decision (role-aware)
                decision = coach.decide(step_ctx, proposed_action, confidence)
                # Only set source if coach didn't already set a specific one
                # Coach may have set: "ppo", "playbook", "anti_repeat", etc.
                if decision.source == "unknown":
                    decision.source = "mentor" if decision.mentor_call else "registry"
            
            # Track this command as used for deduplication
            step_used_commands.add(decision.template_name)
            step_used_commands.add(decision.command[:50])  # Also track command prefix
            
            # Track action for stuck detection
            self._record_action(agent_name, decision.command)
            
            # CRITICAL: Add command to attack_context.command_history IMMEDIATELY
            # This enables the anti-repeat guard in SmartCoach to work properly
            if ctx and decision.command:
                ctx.command_history.append(decision.command)
                # Keep history bounded
                if len(ctx.command_history) > 100:
                    ctx.command_history = ctx.command_history[-100:]
            
            # Create result
            result = SmartStepResult(
                agent_name=agent_name,
                decision=decision,
            )
            agent_results.append(result)
            
            # Collect executable actions
            if agent_name == "RedAgent":
                red_action = decision.command
            elif agent_name == "BlueAgent":
                blue_action = decision.command
        
        # Execute environment step
        env_result, new_state, done = self._execute_env_step(red_action, blue_action)
        env_reward = env_result.get("reward", 0.0) if isinstance(env_result, dict) else env_result
        
        # Get output from environment (may be empty in simulation mode)
        env_output = env_result.get("output", "") if isinstance(env_result, dict) else ""
        
        # Generate simulated outputs for ALL agent commands
        # These will be used for discovery parsing AND display
        for result in agent_results:
            sim_output = self._generate_simulated_output(result.decision.command)
            result.decision.command_output = sim_output  # Store output on decision
        
        # CRITICAL FIX: Parse SIMULATED outputs for discoveries in simulation mode
        # This allows agents to get positive rewards even without a real target
        smart_reward_total = 0.0
        
        # Detect real terminal output vs stringified state dict
        # Real output has newlines and doesn't look like a Python dict repr
        env_has_real_output = (
            env_output
            and not env_output.startswith("[SIM]")
            and not env_output.startswith("{")
            and not env_output.startswith("(")
            and "\n" in env_output
            and len(env_output) > 40
        )
        
        for result in agent_results:
            # Use per-agent simulated output for discovery parsing unless
            # env returned real terminal output (live mode with real target)
            output_to_parse = env_output if env_has_real_output else result.decision.command_output
            
            # Parse discoveries from this agent's output
            agent_discoveries = self._parse_output_for_discoveries(output_to_parse)
            
            # =====================================================================
            # PHASE 2A: DISCOVERY → STATE FLAG BRIDGE
            # Map parsed discoveries to AttackContext state_flags so phase can
            # advance through RECON → ENUMERATION → EXPLOITATION → PRIVESC → POST
            # =====================================================================
            if agent_discoveries and self.attack_context:
                ctx = self.attack_context
                
                # Service discoveries → set_state_flag (triggers phase auto-advance)
                for svc in agent_discoveries.get("service", []):
                    ctx.add_service(svc)  # add_service already calls set_state_flag
                
                # Port discoveries → add to context
                for port in agent_discoveries.get("open_port", []):
                    ctx.add_discovery("open_port", port)
                
                # Credential discovery → advance to EXPLOITATION
                if "credential" in agent_discoveries:
                    ctx.set_state_flag("credentials_known")
                    logger.info(f"[PHASE-ADVANCE] credentials_known set by {result.agent_name}")
                
                # Vulnerability/SQLi discovery → advance to EXPLOITATION
                if agent_discoveries.get("vulnerability"):
                    ctx.set_state_flag("vulnerability_found")
                    cves = agent_discoveries.get("cve", [])
                    if cves:
                        ctx.add_discovery("cve", cves)
                    # Check for specific vuln types that advance phase further
                    output_lower = (output_to_parse or "").lower()
                    if "sql injection" in output_lower or "sqli" in output_lower:
                        ctx.set_state_flag("sqli_confirmed")
                        logger.info(f"[PHASE-ADVANCE] sqli_confirmed set by {result.agent_name}")
                
                # Shell discovery → advance to PRIVILEGE_ESCALATION
                if agent_discoveries.get("shell"):
                    ctx.set_state_flag("shell_obtained")
                    logger.info(f"[PHASE-ADVANCE] shell_obtained set by {result.agent_name}")
                    if agent_discoveries.get("root_shell"):
                        ctx.set_state_flag("root_shell_obtained")
                        ctx.set_state_flag("admin_access_obtained")
                        logger.info(f"[PHASE-ADVANCE] root_shell_obtained set by {result.agent_name}")
                
                # User discoveries
                for user in agent_discoveries.get("user", []):
                    ctx.add_discovery("user", user)
                
                # SMB shares
                for share in agent_discoveries.get("smb_share", []):
                    ctx.add_discovery("smb_share", share)
                    ctx.set_state_flag("smb_service_found")
                
                # Web paths → mark services as enumerated
                if agent_discoveries.get("web_path"):
                    ctx.set_state_flag("services_enumerated")
                    for path in agent_discoveries["web_path"]:
                        ctx.add_discovery("web_path", path)
                
                # Database discovery
                if agent_discoveries.get("database"):
                    ctx.set_state_flag("database_found")
                    for db in agent_discoveries.get("db_name", []):
                        ctx.add_discovery("database", db)
                
                # Sensitive file discovery
                if agent_discoveries.get("sensitive_file"):
                    ctx.add_discovery("sensitive_file", True)
                    ctx.set_state_flag("services_enumerated")
                
                # Hash discovery → lateral movement
                if agent_discoveries.get("hash_dump"):
                    ctx.set_state_flag("hash_known")
                    logger.info(f"[PHASE-ADVANCE] hash_known set by {result.agent_name}")
                
                # Lateral target → lateral movement
                if agent_discoveries.get("lateral_target"):
                    ctx.set_state_flag("lateral_target_found")
                    logger.info(f"[PHASE-ADVANCE] lateral_target_found set by {result.agent_name}")
                
                # Domain admin → post-exploitation
                if agent_discoveries.get("domain_admin"):
                    ctx.set_state_flag("domain_admin_obtained")
                    ctx.set_state_flag("admin_access_obtained")
                    logger.info(f"[PHASE-ADVANCE] domain_admin_obtained set by {result.agent_name}")

                # Persistence → exfiltration
                if agent_discoveries.get("persistence"):
                    ctx.set_state_flag("persistence_established")
                    logger.info(f"[PHASE-ADVANCE] persistence_established set by {result.agent_name}")
                
                # Data exfiltration → exfiltration phase
                if agent_discoveries.get("data_exfiltrated"):
                    ctx.set_state_flag("data_exfiltrated")
                    logger.info(f"[PHASE-ADVANCE] data_exfiltrated set by {result.agent_name}")

                # ─── PHASE 4: Update discovery board for cross-agent sharing ─
                for port in agent_discoveries.get("open_port", []):
                    self.discovery_board["ports"].add(port)
                for svc in agent_discoveries.get("service", []):
                    self.discovery_board["services"].add(svc)
                for user in agent_discoveries.get("user", []):
                    self.discovery_board["users"].add(user)
                if agent_discoveries.get("credential"):
                    self.discovery_board["credentials"].add("found")
                if agent_discoveries.get("shell"):
                    self.discovery_board["shells"].add(result.agent_name)
                if agent_discoveries.get("vulnerability"):
                    self.discovery_board["vulns"].add("found")
                for path in agent_discoveries.get("web_path", []):
                    self.discovery_board["web_paths"].add(str(path))
                self.discovery_board["phase"] = ctx.current_phase.name
                self.discovery_board["flags_set"] = set(
                    k for k, v in ctx.state_flags.items() if v
                )
            
            # Determine success based on simulated output quality
            # Commands that produce meaningful output are considered successful
            sim_success = bool(output_to_parse and not output_to_parse.startswith("[SIM]") and len(output_to_parse) > 20)
            
            # Record result with discoveries (for ALL agents, not just RedAgent)
            if result.agent_name in self.coaches:
                breakdown = self.coaches[result.agent_name].record_result(
                    decision=result.decision,
                    success=sim_success or env_reward >= 0,
                    raw_output=output_to_parse,
                    new_discoveries=agent_discoveries,
                    done=done,  # Phase 4: pass done for PPO trajectory
                )
                result.reward_breakdown = breakdown
                # Accumulate smart rewards from all agents
                if breakdown:
                    smart_reward_total += breakdown.total
                
                # Phase 0.1: Update stagnation counter
                if agent_discoveries:
                    self._steps_without_discoveries[result.agent_name] = 0  # Reset on discovery
                else:
                    self._steps_without_discoveries[result.agent_name] = (
                        self._steps_without_discoveries.get(result.agent_name, 0) + 1
                    )
        
        # Use smart reward if available, otherwise fall back to env reward
        final_reward = smart_reward_total if smart_reward_total != 0 else env_reward
        
        # ─── PHASE 4: PPO trajectory now collected per-coach in SmartCoach ──
        # The old global PPO trajectory collection was disconnected:
        # PPO.select_action() returned a different action than SmartCoach chose,
        # creating incoherent training signal. Now each SmartCoach has its own
        # PPOAgent and records trajectory in record_result().
        # (Global PPO kept for backward compat but no longer collects trajectory)
        
        # Log traces
        for result in agent_results:
            self._log_step_trace(
                episode_id=episode_id,
                step=step,
                result=result,
                global_reward=final_reward,
                done=done,
            )
        
        return agent_results, final_reward, new_state, done
    
    def _update_context_from_state(self, state: Dict[str, Any]):
        """Update attack context from environment state."""
        if not self.attack_context:
            return
        
        ctx = self.attack_context
        
        # Update from state dict
        if isinstance(state, dict):
            # Open ports
            if "open_ports" in state:
                for port in state["open_ports"]:
                    if f"open_port:{port}" not in ctx.discoveries:
                        ctx.add_discovery("open_port", port)
            
            # Services
            if "services" in state:
                for svc in state["services"]:
                    if isinstance(svc, dict):
                        ctx.add_service(svc.get("name", ""), svc.get("port"))
                    else:
                        ctx.add_service(str(svc))
            
            # Platform detection
            if "os" in state and ctx.platform == "unknown":
                os_str = state["os"].lower()
                if "windows" in os_str:
                    ctx.platform = "windows"
                elif "linux" in os_str or "unix" in os_str:
                    ctx.platform = "linux"
            
            # Update last command
            if "last_command" in state:
                ctx.command_history.append(state["last_command"])
    
    def _parse_output_for_discoveries(self, output: str) -> Dict[str, Any]:
        """Parse command output for new discoveries - rewards good simulated actions."""
        discoveries = {}
        
        if not output or output.startswith("[SIM]") and len(output) < 30:
            return discoveries
        
        output_lower = output.lower()
        
        import re
        
        # Port discovery patterns (multiple formats)
        port_patterns = [
            r"(\d+)/(?:tcp|udp)\s+open",  # nmap format
            r"open port (\d+)/",           # masscan format
            r"Open \S+:(\d+)",             # rustscan format
            r"\[(\d+)\]\[",                # hydra format
            r":(\d+)\s+\(",                # netstat/ss format
        ]
        ports = set()
        for pattern in port_patterns:
            ports.update(re.findall(pattern, output_lower))
        if ports:
            discoveries["open_port"] = [int(p) for p in ports if p.isdigit()]
        
        # Service discovery (enhanced)
        service_patterns = {
            "ssh": r"ssh|openssh|sshd",
            "http": r"http|apache|nginx|iis|web server",
            "https": r"https|ssl|tls|443",
            "smb": r"smb|samba|microsoft-ds|445/tcp",
            "ftp": r"ftp|vsftpd|proftpd|21/tcp",
            "mysql": r"mysql|mariadb|3306",
            "mssql": r"ms-sql|mssql|1433",
            "postgresql": r"postgresql|postgres|5432",
            "rdp": r"rdp|3389|remote desktop",
            "smtp": r"smtp|postfix|sendmail|25/tcp",
        }
        
        for svc, pattern in service_patterns.items():
            if re.search(pattern, output_lower):
                if "service" not in discoveries:
                    discoveries["service"] = []
                if svc not in discoveries.get("service", []):
                    discoveries["service"].append(svc)
        
        # Credential patterns (enhanced)
        cred_patterns = [
            r"password[:\s]+\S+",
            r"login:\s*\w+\s+password",
            r"\(Pwn3d!\)",
            r"NTLMv[12] Hash:",
            r"valid credentials",
            r"authentication successful",
        ]
        for pattern in cred_patterns:
            if re.search(pattern, output, re.IGNORECASE):
                discoveries["credential"] = "password_found"
                break
        
        # User discovery
        user_patterns = [
            r"user:\[(\w+)\]",             # rpcclient
            r"user found:\s*(\w+)",        # wpscan
            r"Admin Email:\s*(\S+)",       # whois
            r"login:\s*(\w+)",             # hydra
        ]
        users = []
        for pattern in user_patterns:
            users.extend(re.findall(pattern, output, re.IGNORECASE))
        if users:
            discoveries["user"] = list(set(users))
        
        # Vulnerability patterns
        vuln_patterns = [
            r"CVE-\d{4}-\d+",              # CVE IDs
            r"vulnerable|vulnerability",
            r"exploit|exploitable",
            r"OSVDB-\d+",
            r"Remote Code Execution",
            r"Buffer Overflow",
            r"SQL Injection",
            r"Path Traversal",
        ]
        for pattern in vuln_patterns:
            if re.search(pattern, output, re.IGNORECASE):
                discoveries["vulnerability"] = True
                # Extract CVE IDs
                cves = re.findall(r"CVE-\d{4}-\d+", output, re.IGNORECASE)
                if cves:
                    discoveries["cve"] = list(set(cves))
                break
        
        # Directory/path discovery (web)
        if re.search(r"(?:Status:|CODE:)\s*200", output):
            path_matches = re.findall(r"/([\w\-\.]+)(?:\s*\(Status:\s*200|\s*\[Status:\s*200|CODE:200)", output)
            if path_matches:
                discoveries["web_path"] = list(set(path_matches))
                discoveries["directory"] = True
        
        # Share discovery (SMB)
        share_matches = re.findall(r"(?:Disk|IPC):\s*(\w+)|\\\\[^\\]+\\(\w+)", output)
        if share_matches:
            shares = [s[0] or s[1] for s in share_matches if s[0] or s[1]]
            if shares:
                discoveries["smb_share"] = list(set(shares))
        
        # File discovery (sensitive files)
        sensitive_patterns = [
            r"\.ssh/id_rsa",
            r"\.htaccess",
            r"\.backup",
            r"password",
            r"\.env",
            r"config\.",
            r"wp-config",
        ]
        for pattern in sensitive_patterns:
            if re.search(pattern, output_lower):
                discoveries["sensitive_file"] = True
                break
        
        # Shell indicators
        shell_patterns = [
            r"shell\s*session\s*\d+\s*opened",
            r"www-data@",
            r"root@",
            r"meterpreter\s*>",
            r"\$\s*$",
            r"#\s*$",
            r"C:\\>",
            r"PS\s+C:\\",
            r"nt authority\\system",
            r"uid=0\(root\)",
        ]
        
        for pattern in shell_patterns:
            if re.search(pattern, output, re.MULTILINE):
                discoveries["shell"] = True
                if re.search(r"root@|uid=0|nt authority\\system|domain admin", output, re.IGNORECASE):
                    discoveries["root_shell"] = True
                break
        
        # Hash/credential dump patterns → triggers LATERAL_MOVEMENT
        hash_patterns = [
            r"NTLMv[12]\s*Hash",
            r"[a-f0-9]{32}:{3}",               # NT hash format
            r"\$krb5tgs\$",                      # Kerberoast
            r"\$krb5asrep\$",                    # AS-REP roast
            r"Hash\s*dumped",
            r"secretsdump|hashdump",
            r"mimikatz.*NTLM",
        ]
        for pattern in hash_patterns:
            if re.search(pattern, output, re.IGNORECASE):
                discoveries["hash_dump"] = True
                break
        
        # Lateral movement indicators → triggers LATERAL_MOVEMENT
        lateral_patterns = [
            r"Lateral target:\s*\S+",
            r"Domain Admin found",
            r"PsExec|WmiExec|SmbExec|AtExec|DcomExec",
            r"Evil-WinRM shell",
            r"proxychains.*OK",
            r"Tunnel established",
            r"session#\d+:\s*tun pair",
        ]
        for pattern in lateral_patterns:
            if re.search(pattern, output, re.IGNORECASE):
                discoveries["lateral_target"] = True
                break
        
        # Domain admin indicators → triggers POST_EXPLOITATION
        domain_admin_patterns = [
            r"Domain\s*Admin",
            r"nt authority\\system",
            r"Enterprise\s*Admin",
            r"memberOf.*Domain Admins",
        ]
        for pattern in domain_admin_patterns:
            if re.search(pattern, output, re.IGNORECASE):
                discoveries["domain_admin"] = True
                break
        
        # Persistence indicators → triggers EXFILTRATION
        persistence_patterns = [
            r"Persistence\s*(cron|added|established|installed)",
            r"crontab.*backup|systemd.*service\s*enabled",
            r"Registry\s*key\s*(added|set)",
            r"backdoor.*installed|implant.*deployed",
            r"ssh.*authorized_keys|\.ssh/authorized_keys",
            r"scheduled\s*task\s*created",
        ]
        for pattern in persistence_patterns:
            if re.search(pattern, output, re.IGNORECASE):
                discoveries["persistence"] = True
                break
        
        # Data exfiltration indicators → triggers EXFILTRATION
        exfil_patterns = [
            r"exfiltrat(ed|ion|ing)",
            r"data\s*(extracted|downloaded|stolen|copied|transferred)",
            r"(file|archive|dump)\s*(uploaded|sent|exfil)",
            r"curl.*-F|wget.*--post-file|nc.*<\s*\S+",
            r"scp\s+\S+\s+\S+@",
            r"base64.*encoded.*sent",
        ]
        for pattern in exfil_patterns:
            if re.search(pattern, output, re.IGNORECASE):
                discoveries["data_exfiltrated"] = True
                break
        
        # Database discovery
        if re.search(r"database|DBMS|mysql|postgresql|mssql|mongodb", output_lower):
            discoveries["database"] = True
            db_names = re.findall(r"(?:database|schema):\s*(\w+)", output_lower)
            if db_names:
                discoveries["db_name"] = list(set(db_names))
        
        return discoveries
    
    def _get_agent_proposal(
        self,
        agent: Any,
        state: Dict[str, Any],
    ) -> Tuple[str, float]:
        """Get proposed action from agent (for comparison)."""
        try:
            if hasattr(agent, 'propose_action'):
                result = agent.propose_action(state)
                if isinstance(result, tuple):
                    return result[0], result[1] if len(result) > 1 else 0.5
                return str(result), 0.5
            
            if hasattr(agent, 'select_action'):
                return str(agent.select_action(state)), 0.5
            
            if hasattr(agent, 'get_action'):
                return str(agent.get_action(state)), 0.5
            
            return "noop", 0.3
            
        except Exception as e:
            logger.debug(f"Agent proposal failed: {e}")
            return "noop", 0.3
    
    def _execute_env_step(
        self,
        red_action: Optional[str],
        blue_action: Optional[str],
    ) -> Tuple[Any, Dict[str, Any], bool]:
        """Execute environment step."""
        try:
            action = red_action or "noop"
            result = self.env.step(action)
            
            if isinstance(result, tuple):
                new_state = result[0] if len(result) > 0 else {}
                reward = result[1] if len(result) > 1 else 0.0
                done = result[2] if len(result) > 2 else False
                
                return {"reward": reward, "output": str(new_state)}, new_state, done
            
            elif isinstance(result, dict):
                return result, result, result.get("done", False)
            
            return {"reward": 0.0}, {}, False
            
        except Exception as e:
            logger.debug(f"Env step failed: {e}")
            return {"reward": 0.0}, {}, False
    
    def _generate_simulated_output(self, command: str) -> str:
        """Generate realistic simulated output for a command with discoverable patterns."""
        if not command:
            return ""
        
        import random
        import hashlib
        
        target = self.attack_context.target if self.attack_context else "10.10.10.10"
        cmd_lower = command.lower().split()[0] if command.split() else ""
        
        # Use command hash for deterministic but varied results
        cmd_hash = int(hashlib.md5(command.encode()).hexdigest()[:8], 16)
        random.seed(cmd_hash)
        
        # Variable ports and services for discovery variety
        ports = random.sample([21, 22, 25, 80, 110, 139, 443, 445, 1433, 3306, 3389, 5432, 8080, 8443], k=random.randint(3, 6))
        services = {21: "ftp", 22: "ssh", 25: "smtp", 80: "http", 110: "pop3", 139: "netbios", 
                   443: "https", 445: "smb", 1433: "mssql", 3306: "mysql", 3389: "rdp", 
                   5432: "postgresql", 8080: "http-alt", 8443: "https-alt"}
        
        # Enhanced simulated outputs with discoverable patterns
        SIMULATED_OUTPUTS = {
            "nmap": "\n".join([f"{p}/tcp open  {services.get(p, 'unknown')}" for p in sorted(ports)]) + f"\nNmap done: 1 IP ({target})",
            "masscan": "\n".join([f"Discovered open port {p}/tcp on {target}" for p in ports[:4]]),
            "rustscan": "\n".join([f"Open {target}:{p}" for p in ports]) + f"\n[~] Running nmap on {target}",
            "enum4linux": f"[+] Target: {target}\n[+] RID cycling: administrator, guest, backup\n[+] Shares: IPC$, ADMIN$, C$\n[+] Password policy: MinLen=7",
            "smbclient": f"\\\\{target}\\IPC$\nSharename  Type  Comment\nIPC$       IPC   Remote IPC\nADMIN$     Disk  Admin share\nbackup     Disk  Backup files",
            "smbmap": f"[+] IP: {target}:445  Name: TARGET\n[+] Disk: backup (READ)\n[+] Disk: ADMIN$ (NO ACCESS)",
            "rpcclient": "$> enumdomusers\nuser:[Administrator] rid:[0x1f4]\nuser:[Guest] rid:[0x1f5]\nuser:[backup_svc] rid:[0x3e8]",
            "gobuster": f"/admin (Status: 200, Size: 3456)\n/login (Status: 200, Size: 1234)\n/backup (Status: 403)\n/api (Status: 200, Size: 567)",
            "dirb": f"+ http://{target}/admin (CODE:200|SIZE:3456)\n+ http://{target}/robots.txt (CODE:200|SIZE:123)",
            "feroxbuster": f"200  GET  /admin/\n200  GET  /login.php\n301  GET  /images/\n403  GET  /backup/",
            "ffuf": "admin [Status: 200, Size: 3456]\nlogin [Status: 200, Size: 1234]\napi [Status: 200, Size: 567]",
            "dirsearch": f"[200] http://{target}/admin/\n[200] http://{target}/login.php\n[403] http://{target}/.htaccess",
            "nikto": f"+ Server: Apache/2.4.41\n+ /admin/: Admin page found\n+ X-Frame-Options header not set\n+ OSVDB-3092: /backup/: Backup dir found",
            "nuclei": f"[CVE-2021-41773] Apache Path Traversal: {target}:80\n[info] Web server detected: Apache/2.4.41",
            "curl": f"HTTP/1.1 200 OK\nServer: Apache/2.4.41 (Ubuntu)\nX-Powered-By: PHP/7.4.3\nSet-Cookie: PHPSESSID=abc123",
            "whatweb": f"http://{target} [200 OK] Apache[2.4.41], PHP[7.4.3], Bootstrap, jQuery[3.5.1], PasswordField",
            "wpscan": f"[+] WordPress version 5.7.2 identified\n[+] User found: admin\n[!] Vulnerable plugin: contact-form-7 (5.4.1)",
            "hydra": f"[22][ssh] host: {target} login: admin password: admin123\n[22][ssh] host: {target} login: backup password: backup2024",
            "crackmapexec": f"SMB  {target}  445  TARGET  [+] admin:Password123! (Pwn3d!)",
            "searchsploit": "Apache 2.4.41 - Remote Code Execution | linux/remote/12345.py\nPHP 7.4 - Buffer Overflow | php/remote/54321.py",
            "sqlmap": f"[INFO] the back-end DBMS is MySQL\n[INFO] fetching database names\navailable databases [2]: information_schema, webapp_db",
            "netstat": "Proto Recv-Q Local Address  Foreign Address State  PID/Program\ntcp   0      0.0.0.0:22    0.0.0.0:*      LISTEN 789/sshd\ntcp   0      0.0.0.0:80    0.0.0.0:*      LISTEN 1234/apache2",
            "ss": f"tcp  LISTEN 0 128 0.0.0.0:22  0.0.0.0:*  users:((\"sshd\",pid=789))\ntcp  LISTEN 0 128 0.0.0.0:80  0.0.0.0:*  users:((\"apache2\",pid=1234))",
            "ps": "PID   USER  %CPU %MEM CMD\n1     root  0.0  0.1  /sbin/init\n789   root  0.1  0.2  /usr/sbin/sshd\n1234  www   1.2  0.5  /usr/sbin/apache2",
            "last": f"admin  pts/0  192.168.1.10  Sat Jan  4 10:30   still logged in\nroot   tty1                  Sat Jan  4 08:00 - 09:30",
            "who": f"admin    pts/0        2026-01-04 10:30 (192.168.1.10)\nroot     tty1         2026-01-04 08:00",
            "lsof": f"sshd    789  root   3u  IPv4 12345  TCP *:22 (LISTEN)\napache2 1234 www    4u  IPv4 23456  TCP *:80 (LISTEN)",
            "find": "/tmp/suspicious.sh\n/var/www/.backup.zip\n/home/admin/.ssh/id_rsa\n/opt/scripts/db_backup.sh",
            "cat": "root:x:0:0:root:/root:/bin/bash\nadmin:x:1000:1000:Admin:/home/admin:/bin/bash\nbackup:x:1001:1001::/home/backup:/bin/sh",
            "w": f"USER   TTY    FROM           LOGIN@  IDLE  WHAT\nadmin  pts/0  192.168.1.10  10:30   0.00s bash\nroot   tty1   -             08:00   2:30m -bash",
            "crontab": "*/5 * * * * /usr/local/bin/backup.sh\n0 2 * * * /opt/scripts/db_backup.sh",
            "systemctl": "apache2.service loaded active running Apache HTTP Server\nmysql.service  loaded active running MySQL Community Server",
            "dig": f";; ANSWER SECTION:\n{target}. 300 IN A 10.10.10.10\n{target}. 300 IN MX 10 mail.{target}",
            "whois": f"Domain Name: TARGET.COM\nRegistrar: Example Registrar\nAdmin Email: admin@target.com",
            "ssh-audit": f"(gen) banner: SSH-2.0-OpenSSH_8.2p1 Ubuntu-4ubuntu0.3\n(gen) compatibility: OpenSSH 7.4+\n(rec) Use of weak key exchange: diffie-hellman-group14-sha1",
            "linpeas": "[+] Possible sudo/suid/caps binaries:\n/usr/bin/pkexec (CVE-2021-4034)\n/usr/bin/sudo\n[+] Writable /etc/passwd",
            "pspy": "CMD: UID=0 PID=1234 /bin/bash /root/backup_cron.sh\nCMD: UID=0 PID=5678 /opt/scripts/check_services.sh",
            "responder": f"[+] Listening for events...\n[HTTP] NTLMv2 Hash: admin::CORP:abc123def456",
            "impacket": f"[*] SMBv3.0 dialect used\n[+] admin:Password123!@{target}:445",
            "msfconsole": f"msf6 exploit(multi/handler) > exploit\n[*] Started reverse handler\n[*] Meterpreter session 1 opened ({target}:4444 -> 10.10.14.2:4444)\nmeterpreter > getuid\nServer username: www-data",
            "exploit": f"[*] Exploiting target {target}\n[+] shell session 1 opened ({target}:4444 -> 10.10.14.2:8080)\nwww-data@target:/var/www$",
            "python": f"Listening on 0.0.0.0:4444\nConnection from {target} on port 4444\n$ id\nuid=33(www-data) gid=33(www-data) groups=33(www-data)",
            "nc": f"listening on [any] 4444 ...\nconnect to [{target}] from (UNKNOWN) [{target}]\n$ id\nuid=33(www-data) gid=33(www-data)",
            "sudo": "User admin may run the following commands on target:\n    (ALL) NOPASSWD: /usr/bin/vim\n    (ALL) NOPASSWD: /usr/bin/python3",
            "scp": f"admin@{target}'s password:\n100% 4096KB 3.5MB/s 00:01",
            "john": "admin:admin123\nbackup:backup2024\nroot:toor\n3 password hashes cracked, 0 left",
            "hashcat": "Hash.Mode: 1000 (NTLM)\nadmin:Password123!\nbackup:Backup2024!\nStatus: Cracked",
            "metasploit": f"[*] Meterpreter session 1 opened ({target}:4444 -> 10.10.14.2:4444)\nmeterpreter > sysinfo\nComputer: TARGET\nOS: Linux 5.4.0 (Ubuntu 20.04)",
            "pkexec": f"[+] CVE-2021-4034 exploit successful\nroot@target:/# id\nuid=0(root) gid=0(root) groups=0(root)",
            "wget": "Saving to: 'linpeas.sh'\n100%[============>] 776,423 1.83MB/s in 0.4s\n2026-01-04 10:30:01 (1.83 MB/s) - saved [776423/776423]",
            "chmod": "",
            "cp": "",
            "mv": "",
            "arping": f"ARPING {target}\n60 bytes from {target}: index=0 time=1.234 msec\n60 bytes from {target}: index=1 time=0.876 msec",
            "waybackurls": f"http://{target}/admin\nhttp://{target}/login.php\nhttp://{target}/backup/",
            # Privilege escalation outputs
            "secretsdump": f"Impacket - Dumping LSA secrets\n[*] Target: {target}\nAdministrator:500:aad3b435b51404eeaad3b435b51404ee:31d6cfe0d16ae931b73c59d7e0c089c0:::\nNTLMv2 Hash: admin::CORP:abc123def456\n[+] Hash dumped: 3 accounts",
            "mimikatz": f"  .#####.   mimikatz 2.2.0\n * Username : admin\n * Domain   : CORP\n * NTLM     : 31d6cfe0d16ae931b73c59d7e0c089c0\n * SHA1     : da39a3ee5e6b4b0d3255bfef95601890afd80709\nAuthentication Id: admin (S-1-5-21-CORP)",
            "bloodhound": "[+] Collecting domain data\n[+] Users: 45 | Groups: 12 | Computers: 8\n[+] Domain Admin found: admin@CORP.LOCAL\n[+] Lateral target: DC01.CORP.LOCAL (10.10.10.100)\n[+] Kerberoastable users: svc_backup",
            "kerbrute": f"2026/01/04 10:30:01 >  [+] VALID USERNAME:	admin@{target}\n2026/01/04 10:30:02 >  [+] VALID USERNAME:	svc_backup@{target}",
            "chisel": f"server: session#1: tun pair: 127.0.0.1:8080 → {target}:80\n[+] Tunnel established",
            "socat": f"listening on 0.0.0.0:4444\nconnection from {target}\n$ id\nuid=0(root) gid=0(root)",
            "ncrack": f"Discovered credentials on {target} 22/tcp:\n22/tcp ssh: 'admin' 'Password123!'",
            "medusa": f"ACCOUNT FOUND: [ssh] Host: {target} User: admin Password: admin123 [SUCCESS]",
            "wfuzz": f"000000001:  200  95 L  251 W  3456 Ch  \"admin\"\n000000015:  200  30 L   89 W  1234 Ch  \"login\"",
            "tplmap": f"[+] Tplmap 0.5\n[+] Testing if GET parameter 'name' is injectable\n[+] Smarty plugin has confirmed injection\n[+] OS Shell command execution available",
            "commix": f"[+] The GET parameter 'cmd' is vulnerable to OS command injection\n[+] Target OS: Linux\n$ id\nuid=33(www-data) gid=33(www-data)",
            "dalfox": f"[POC][R][GET] http://{target}/page?q=<script>alert(1)</script>\n[*] Found 1 XSS vulnerability",
            "xsstrike": f"[~] Checking for DOM vulnerabilities\n[+] Vulnerable parameter: q\n[+] Payload: <img src=x onerror=alert(1)>",
            "gospider": f"[url] http://{target}/admin\n[url] http://{target}/api/v1\n[form] http://{target}/login",
            "katana": f"http://{target}/admin/\nhttp://{target}/api/v1/users\nhttp://{target}/login.php",
            "ldapsearch": f"# CORP.LOCAL\ndn: DC=corp,DC=local\n# admin, Users, corp.local\ndn: CN=admin,CN=Users,DC=corp,DC=local\nmemberOf: CN=Domain Admins",
            "evil-winrm": f"Evil-WinRM shell v3.4\nInfo: Establishing connection to remote endpoint\n*Evil-WinRM* PS C:\\Users\\admin> whoami\ncorp\\admin",
            "psexec": f"Impacket v0.10.0 - PsExec\n[*] Requesting shares on {target}\n[*] Found writable share ADMIN$\n[*] Uploading shell\nMicrosoft Windows [Version 10.0.19041]\nC:\\WINDOWS\\system32> whoami\nnt authority\\system",
            "wmiexec": f"Impacket v0.10.0 - WmiExec\n[*] SMBv3.0 dialect used\nC:\\> whoami\ncorp\\admin",
            "smbexec": f"Impacket v0.10.0 - SmbExec\n[*] admin@{target}\nC:\\WINDOWS\\system32> whoami\nnt authority\\system",
            "atexec": f"Impacket v0.10.0 - AtExec\n[*] Creating task\n[*] Running task\nnt authority\\system",
            "dcomexec": f"Impacket v0.10.0 - DcomExec\n[*] {target} - admin authenticated\nC:\\> whoami\ncorp\\admin",
            "getTGT": f"Impacket - getTGT\n[*] Saving ticket in admin.ccache\n[+] Kerberos TGT obtained for admin@CORP.LOCAL",
            "getST": f"Impacket - getST\n[*] Getting service ticket for admin@CORP.LOCAL\n[+] Service ticket saved to admin_svc.ccache",
            "GetUserSPNs": f"ServicePrincipalName  Name     MemberOf\nHTTP/web.corp.local   svc_web  Domain Users\n$krb5tgs$23$*svc_web$CORP.LOCAL*$hash",
            "GetNPUsers": f"[-] User admin does not require preauth\n$krb5asrep$23$admin@CORP.LOCAL:hash_value",
            "mssqlclient": f"Impacket v0.10.0 - MSSQLClient\n[*] Logged in to {target}:1433\nSQL> SELECT * FROM users\nadmin  | Password123!\nbackup | Backup2024!",
            "xfreerdp": f"[INFO] Connected to {target}:3389\n[INFO] Domain: CORP\n[INFO] Authentication successful",
            "proxychains": f"[proxychains] Strict chain ... 127.0.0.1:1080 ... {target}:445 ... OK",
            "ssh": f"admin@{target}'s password:\nWelcome to Ubuntu 20.04.3 LTS\nadmin@target:~$ id\nuid=1000(admin) gid=1000(admin) groups=1000(admin),27(sudo)",
            # Post-exploitation / persistence
            "crontab": "*/5 * * * * /usr/local/bin/backup.sh\n0 2 * * * /opt/scripts/db_backup.sh\n[+] Persistence cron added",
            # Exfiltration / persistence outputs
            "tar": f"tar: creating archive /tmp/loot.tar.gz\n[+] Data extracted: /etc/shadow, /etc/passwd, /home/admin/.ssh/\n[+] Archive uploaded to 10.10.14.2 via nc\n[+] Data exfiltrated successfully",
            "base64": "[+] File encoded to base64 and sent\n[+] Exfiltration complete: data transferred via DNS",
            "exfiltrate": "[+] Exfiltrating sensitive data\n[+] /etc/shadow: 42 entries\n[+] SSH keys: 3 found\n[+] Data exfiltrated to C2",
            "systemctl": "[+] Created systemd service: /etc/systemd/system/backup.service\n[+] Service enabled and started\n[+] Persistence established via systemd",
            "reg": "[+] Registry key added: HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run\\Updater\n[+] Persistence established via registry",
            "schtasks": "[+] Scheduled task created: \\Microsoft\\Windows\\Maintenance\\Backup\n[+] Persistence established via scheduled task",
            "useradd": "[+] User 'svc_backup' added to system\n[+] Added to sudoers with NOPASSWD\n[+] Persistence established via backdoor user",
        }
        
        for prefix, output in SIMULATED_OUTPUTS.items():
            if cmd_lower.startswith(prefix.lower()):
                return output
        
        # Fallback: try matching command keywords for exploit-like commands
        if "exploit" in cmd_lower or "meterpreter" in cmd_lower:
            return f"[*] Exploiting target {target}\n[+] shell session 1 opened ({target}:4444 -> 10.10.14.2:8080)\nwww-data@target:/var/www$"
        if "shell" in cmd_lower or "reverse" in cmd_lower:
            return f"[+] Reverse shell received\nwww-data@target:/tmp$ id\nuid=33(www-data) gid=33(www-data)"
        
        return f"[SIM] {command[:50]}... executed"
    
    def _log_step_trace(
        self,
        episode_id: str,
        step: int,
        result: SmartStepResult,
        global_reward: float,
        done: bool,
    ):
        """Log step trace with tokens and reward breakdown."""
        if not self.trace_writer:
            return
        
        from core.tracing import StepTrace
        
        # Build reward breakdown dict
        rb_dict = None
        if result.reward_breakdown:
            rb_dict = {
                "base_reward": result.reward_breakdown.base_reward,
                "novelty_bonus": result.reward_breakdown.novelty_bonus,
                "progress_bonus": result.reward_breakdown.progress_bonus,
                "phase_advance_bonus": result.reward_breakdown.phase_advance_bonus,
                "discovery_bonus": result.reward_breakdown.discovery_bonus,
                "redundancy_penalty": result.reward_breakdown.redundancy_penalty,
                "total": result.reward_breakdown.total,
            }
        
        # Get token stats (cumulative)
        tokens_step = result.decision.tokens_used
        tokens_episode = self.gpt_manager.tokens_used if self.gpt_manager else 0
        
        trace = StepTrace(
            episode_id=episode_id,
            step=step,
            agent=result.agent_name,
            phase=result.decision.phase.name.lower(),
            proposed_action=result.decision.command,
            chosen_action=result.decision.command,
            mentor_call=result.decision.mentor_call,
            model_used=result.decision.model_used,
            reward=result.reward_breakdown.total if result.reward_breakdown else global_reward,
            done=done,
            mentor_response=result.decision.mentor_reasoning,
            confidence=result.decision.confidence,
            tokens_used_step=tokens_step,
            tokens_used_episode=tokens_episode,
            reward_breakdown=rb_dict,
        )
        
        self.trace_writer.log_step(trace)
    
    def _compute_episode_metrics(
        self,
        step_results: List[List[SmartStepResult]],
        total_reward: float,
        done: bool,
        phase_progression: List[str],
    ) -> Dict[str, Any]:
        """Compute detailed episode metrics."""
        metrics = {
            "total_steps": len(step_results),
            "total_reward": total_reward,
            "done": done,
            "phase_progression": phase_progression,
            "highest_phase": phase_progression[-1] if phase_progression else "RECON",
            "phases_reached": len(set(phase_progression)),
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
                total_reward_agent = sum(
                    r.reward_breakdown.total if r.reward_breakdown else 0.0
                    for r in agent_steps
                )
                
                metrics["agents"][agent_name] = {
                    "steps": len(agent_steps),
                    "mentor_calls": sum(1 for r in agent_steps if r.decision.mentor_call),
                    "avg_confidence": sum(r.decision.confidence for r in agent_steps) / len(agent_steps),
                    "total_reward": total_reward_agent,
                    "unique_commands": len(set(r.decision.template_name for r in agent_steps)),
                }
        
        # Reward breakdown summary
        if step_results:
            all_breakdowns = [
                r.reward_breakdown for results in step_results
                for r in results if r.reward_breakdown
            ]
            
            if all_breakdowns:
                metrics["reward_summary"] = {
                    "avg_novelty_bonus": sum(b.novelty_bonus for b in all_breakdowns) / len(all_breakdowns),
                    "avg_discovery_bonus": sum(b.discovery_bonus for b in all_breakdowns) / len(all_breakdowns),
                    "avg_redundancy_penalty": sum(b.redundancy_penalty for b in all_breakdowns) / len(all_breakdowns),
                    "total_phase_advance_bonus": sum(b.phase_advance_bonus for b in all_breakdowns),
                }
        
        return metrics
    
    def _default_state(self) -> Dict[str, Any]:
        """Default state when environment doesn't provide one."""
        return {
            "phase": "recon",
            "target_ip": self.config.default_target,
            "open_ports": [],
            "detection_risk": 0.0,
            "services": [],
        }
    
    def _record_action(self, agent_name: str, action: str):
        """Record action for stuck detection."""
        if agent_name not in self.action_history:
            self.action_history[agent_name] = []
        
        self.action_history[agent_name].append(action)
        max_history = self.config.stuck_threshold + 2
        if len(self.action_history[agent_name]) > max_history:
            self.action_history[agent_name] = self.action_history[agent_name][-max_history:]
    
    def _check_if_stuck(self, agent_name: str) -> bool:
        """Check if agent is stuck."""
        if agent_name not in self.action_history:
            return False
        
        recent = self.action_history[agent_name]
        if len(recent) < self.config.stuck_threshold:
            return False
        
        # Check for repeated actions
        last_n = recent[-self.config.stuck_threshold:]
        if len(set(last_n)) == 1:
            if agent_name not in self.stuck_agents:
                self.stuck_agents.add(agent_name)
                logger.warning(f"Agent {agent_name} STUCK: repeated '{last_n[0][:40]}...'")
                self.dashboard.add_event("stuck", f"Repeated: {last_n[0][:30]}...", agent_name)
            return True
        
        self.stuck_agents.discard(agent_name)
        return False
    
    # =========================================================================
    # PHASE 0.1: STUCK-ESCAPE METHODS
    # =========================================================================
    
    def _compute_discoveries_delta(self) -> Dict[str, Any]:
        """
        Compute the difference in discoveries since last call.
        
        Phase 0.1: Used for per-step reward decomposition.
        
        Returns:
            Dictionary of new discoveries this step
        """
        if not self.attack_context:
            return {}
        
        current = dict(self.attack_context.discoveries)
        delta = {}
        
        for key, value in current.items():
            prev_value = self.previous_discoveries.get(key)
            
            if prev_value is None:
                # New discovery type
                delta[key] = value
            elif isinstance(value, list) and isinstance(prev_value, list):
                # List discovery - find new items
                new_items = [v for v in value if v not in prev_value]
                if new_items:
                    delta[key] = new_items
            elif value != prev_value:
                # Changed value
                delta[key] = value
        
        # Update previous for next call
        self.previous_discoveries = current.copy()
        
        return delta
    
    def _check_repeat_stuck(self, agent_name: str) -> Tuple[bool, int]:
        """
        Phase 0.1: Check if agent is repeat-stuck.
        
        Triggers on:
        1. Consecutive identical actions (>= threshold)
        2. OR: No discovery progress for K steps (stagnation)
        
        Returns:
            (is_stuck, repeat_count)
        """
        if agent_name not in self.action_history:
            return False, 0
        
        recent = self.action_history[agent_name]
        if len(recent) < 2:
            return False, 0
        
        # Check 1: Count consecutive repeats from the end
        last_action = recent[-1]
        repeat_count = 1
        for i in range(len(recent) - 2, -1, -1):
            if recent[i] == last_action:
                repeat_count += 1
            else:
                break
        
        is_exact_stuck = repeat_count >= self.config.stuck_repeat_threshold
        
        # Check 2: Stagnation check - no discoveries in last K steps
        # (triggers after stuck_repeat_threshold steps of zero progress)
        stagnation_window = self.config.stuck_repeat_threshold
        steps_without_progress = getattr(self, '_steps_without_discoveries', {}).get(agent_name, 0)
        is_stagnant = steps_without_progress >= stagnation_window
        
        is_stuck = is_exact_stuck or is_stagnant
        return is_stuck, max(repeat_count, steps_without_progress)
    
    def _check_deep_stuck(self, agent_name: str) -> bool:
        """
        Phase 0.1: Check if agent is deep-stuck (too many forced-novel failures).
        
        Returns:
            True if should abort episode for this agent
        """
        return self.deep_stuck_count.get(agent_name, 0) >= self.config.stuck_forced_abort_threshold
    
    def get_attack_summary(self) -> Dict[str, Any]:
        """Get summary of current attack state."""
        if not self.attack_context:
            return {"status": "no_attack_context"}
        
        ctx = self.attack_context
        return {
            "target": ctx.target,
            "platform": ctx.platform,
            "difficulty": ctx.difficulty,
            "current_phase": ctx.current_phase.name,
            "services_found": ctx.services_found,
            "discoveries": dict(ctx.discoveries),
            "commands_executed": len(ctx.command_history),
            "failed_attempts": len(ctx.failed_attempts),
            "state_flags": {k: v for k, v in ctx.state_flags.items() if v},
        }
    
    def get_all_agent_stats(self) -> Dict[str, Any]:
        """Get statistics from all coaches."""
        return {
            name: coach.get_stats()
            for name, coach in self.coaches.items()
        }
    
    # =========================================================================
    # MAIN TRAINING ENTRY POINT
    # =========================================================================
    
    def run_training(
        self,
        episodes: int = 10,
        target_ip: Optional[str] = None,
        difficulty: str = "medium",
        platform: str = "unknown",
    ) -> Dict[str, Any]:
        """
        Run complete training loop with smart command generation.
        
        This is the main entry point for training with SmartOrchestrator.
        
        Args:
            episodes: Number of episodes to run
            target_ip: Target IP address
            difficulty: Target difficulty
            platform: Target platform (linux, windows, unknown)
            
        Returns:
            Training results with metrics
        """
        import uuid
        
        self.run_id = f"smart_{uuid.uuid4().hex[:8]}"
        self.total_episodes = episodes
        self.start_time = time.time()
        
        # Set up dashboard
        self.dashboard.set_run_info(self.run_id, episodes)
        
        # Results tracking
        all_metrics: List[Dict[str, Any]] = []
        episode_rewards: List[float] = []
        phase_progressions: List[List[str]] = []
        
        target = target_ip or self.config.default_target
        
        logger.info(f"Starting smart training: {episodes} episodes, target={target}")
        
        for ep in range(episodes):
            episode_id = f"{self.run_id}_ep{ep:04d}"
            
            # Run episode
            metrics = self.run_episode(
                episode_id=episode_id,
                episode_number=ep,
                target=target,
                difficulty=difficulty,
                platform=platform,
            )
            
            all_metrics.append(metrics)
            episode_rewards.append(metrics["total_reward"])
            phase_progressions.append(metrics.get("phase_progression", ["RECON"]))
            
            # Update skill library size in dashboard
            if self.skill_library:
                self.dashboard.set_skill_library_size(len(self.skill_library))
        
        # Compute final metrics
        total_time = time.time() - self.start_time
        
        final_metrics = {
            "avg_reward_recent": sum(episode_rewards[-10:]) / min(len(episode_rewards), 10),
            "avg_confidence_recent": 0.5,  # TODO: track from coaches
            "avg_mentor_rate_recent": 0.2,  # TODO: track from coaches
            "skill_library_size": len(self.skill_library) if self.skill_library else 0,
            "reward_trend": self._calculate_trend(episode_rewards),
        }
        
        # Print final run summary
        self.dashboard.print_run_summary(
            run_id=self.run_id,
            total_episodes=episodes,
            total_time=total_time,
            final_metrics=final_metrics,
        )
        
        # Return results compatible with existing training system
        return {
            "session_id": self.run_id,
            "episodes_completed": episodes,
            "total_training_time": total_time,
            "final_score": final_metrics["avg_reward_recent"],
            "final_coordination": 0.8,  # TODO: compute from agent interactions
            "final_metrics": {
                "avg_reward": sum(episode_rewards) / max(len(episode_rewards), 1),
                "coordination_score": 0.8,
                "highest_phase": max(
                    (p[-1] for p in phase_progressions),
                    key=lambda x: ["RECON", "ENUMERATION", "EXPLOITATION", 
                                   "PRIVILEGE_ESCALATION", "LATERAL_MOVEMENT",
                                   "POST_EXPLOITATION", "EXFILTRATION"].index(x)
                    if x in ["RECON", "ENUMERATION", "EXPLOITATION", 
                             "PRIVILEGE_ESCALATION", "LATERAL_MOVEMENT",
                             "POST_EXPLOITATION", "EXFILTRATION"] else 0,
                    default="RECON"
                ),
            },
            "all_episode_metrics": all_metrics,
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend from values."""
        if len(values) < 3:
            return "stable"
        
        mid = len(values) // 2
        first_half = sum(values[:mid]) / max(mid, 1)
        second_half = sum(values[mid:]) / max(len(values) - mid, 1)
        
        diff = second_half - first_half
        threshold = 0.1 * max(abs(first_half), 1.0)
        
        if diff > threshold:
            return "increasing"
        elif diff < -threshold:
            return "decreasing"
        return "stable"
