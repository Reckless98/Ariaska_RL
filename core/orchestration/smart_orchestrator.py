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
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field

from core.commands.command_registry import (
    AttackPhase,
    get_phase_from_state,
    COMMAND_REGISTRY,
)
from core.llm.smart_mentor import AttackContext
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
    
    # Stuck detection (enhanced)
    stuck_threshold: int = 3
    stuck_negative_streak: int = 5
    stuck_force_mentor: bool = True
    stuck_force_exploration: bool = True
    
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
        
        # Initialize LiveDashboard for real-time visibility
        self.dashboard = self._init_dashboard()
        
        logger.info(f"SmartOrchestrator initialized with {len(self.agents)} agents")
    
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
        
        # Reset coaches and reward calculator
        for coach in self.coaches.values():
            coach.reset_episode(episode_number)
        self.global_reward_calc.reset()
        
        # Reset stuck detection
        self.action_history.clear()
        self.stuck_agents.clear()
        
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
                logger.info(f"Phase advanced: {phase_progression[-2]} → {current_phase}")
                self.dashboard.add_event(
                    "phase_change",
                    f"Advanced to {current_phase}",
                    agent="system"
                )
            
            if done:
                break
        
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
        
        return metrics
    
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
            
            agent = self.agents[agent_name]
            coach = self.coaches[agent_name]
            
            # Share used commands with this coach
            if hasattr(coach, 'step_used_commands'):
                coach.step_used_commands = step_used_commands
            
            # Check if stuck and force exploration
            is_stuck = self._check_if_stuck(agent_name)
            force_mentor = is_stuck and self.config.stuck_force_mentor
            
            # Build smart step context
            step_ctx = SmartStepContext(
                episode=self.current_episode,
                step=step,
                agent_name=agent_name,
                attack_context=ctx,
                state=state if isinstance(state, dict) else {},
            )
            
            # Get agent's proposed action (for comparison)
            proposed_action, confidence = self._get_agent_proposal(agent, state)
            
            # Force low confidence if stuck
            if force_mentor:
                confidence = 0.1
            
            # Get smart decision (role-aware)
            decision = coach.decide(step_ctx, proposed_action, confidence)
            
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
        for result in agent_results:
            # Use simulated output if env output is empty (simulation mode)
            output_to_parse = env_output if env_output and not env_output.startswith("[SIM]") else result.decision.command_output
            
            # Parse discoveries from this agent's output
            agent_discoveries = self._parse_output_for_discoveries(output_to_parse)
            
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
                )
                result.reward_breakdown = breakdown
                # Accumulate smart rewards from all agents
                if breakdown:
                    smart_reward_total += breakdown.total
        
        # Use smart reward if available, otherwise fall back to env reward
        final_reward = smart_reward_total if smart_reward_total != 0 else env_reward
        
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
        ]
        
        for pattern in shell_patterns:
            if re.search(pattern, output, re.MULTILINE):
                discoveries["shell"] = True
                if "root@" in output or "# " in output or "UID=0" in output:
                    discoveries["root_shell"] = True
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
        }
        
        for prefix, output in SIMULATED_OUTPUTS.items():
            if cmd_lower.startswith(prefix.lower()):
                return output
        
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
