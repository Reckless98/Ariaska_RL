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
from typing import Optional, Dict, Any, List, TYPE_CHECKING
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
)
from core.llm.reward_calculator import (
    SmartRewardCalculator,
    RewardBreakdown,
)
from core.training.mentor_policy import MentorPolicy, MentorPolicyConfig

if TYPE_CHECKING:
    from core.gpt_manager import GPTManager
    from core.postmortem import SkillLibrary
    from core.tracing import TraceWriter

logger = logging.getLogger("ariaska.smart_coach")


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
    
    # Phase info
    phase: AttackPhase = AttackPhase.RECON
    phase_advanced: bool = False
    
    # Reward info
    reward_breakdown: Optional[RewardBreakdown] = None
    
    # Confidence and metadata
    confidence: float = 0.5
    skill_cards: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    
    @property
    def chosen_action(self) -> str:
        """Alias for compatibility with existing code."""
        return self.command


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
    # AGENT ROLE DEFINITIONS - Each agent has unique focus and commands
    # ==========================================================================
    AGENT_ROLES = {
        "ScoutAgent": {
            "role": "recon",
            "description": "🔍 Reconnaissance & Discovery - port scanning, service enumeration, OSINT",
            "primary_phases": [AttackPhase.RECON, AttackPhase.ENUMERATION],
            "preferred_commands": [
                # Port scanning variety
                "nmap_quick_scan", "nmap_top_ports", "nmap_full_tcp", "nmap_service_version",
                "masscan_fast", "nmap_udp_scan", "nmap_os_detection",
                # Service enumeration
                "whatweb", "curl_headers", "dig_any", "whois_lookup", "dns_zone_transfer",
                # SMB/Windows recon
                "enum4linux", "smbclient_list", "rpcclient_enum",
                # SNMP
                "snmpwalk", "onesixtyone",
                # Web discovery
                "gobuster_dir", "ffuf_fuzz", "nuclei_scan",
            ],
            "command_tags": {"network", "discovery", "scanning", "dns", "recon", "enum", "web"},
            "avoid_tags": {"exploit", "privesc", "persistence", "defense", "attack", "bruteforce"},
        },
        "RedAgent": {
            "role": "offensive",
            "description": "⚔️ Offensive Operations - aggressive scanning, exploitation, brute force",
            "primary_phases": [AttackPhase.RECON, AttackPhase.ENUMERATION, AttackPhase.EXPLOITATION, AttackPhase.PRIVILEGE_ESCALATION],
            "preferred_commands": [
                # Aggressive recon
                "nmap_vuln_scan", "nmap_all_ports", "masscan_full", "nmap_scripts_all",
                # Exploitation prep
                "searchsploit", "nikto", "gobuster_dir", "wfuzz_dir",
                # Brute force
                "hydra_ssh", "hydra_ftp", "hydra_smb", "crackmapexec_smb",
                # Exploitation
                "sqlmap_get", "sqlmap_post", "impacket_psexec", "evil_winrm",
            ],
            "command_tags": {"exploit", "attack", "offensive", "bruteforce", "vuln", "aggressive"},
            "avoid_tags": {"defense", "monitoring", "passive"},
            # Red can do recon but prefers aggressive versions
            "aggressive_recon": True,
        },
        "BlueAgent": {
            "role": "defensive",
            "description": "🛡️ Defensive Analysis - log review, threat detection, security monitoring",
            "primary_phases": [AttackPhase.ENUMERATION, AttackPhase.POST_EXPLOITATION],
            "preferred_commands": [],  # Uses custom defensive commands
            "command_tags": {"defense", "monitoring", "analysis", "logs", "forensics"},
            "avoid_tags": {"exploit", "attack", "bruteforce"},
            "custom_commands": [
                ("netstat -tlnp", "List listening TCP ports and processes"),
                ("ss -tlnp", "Socket statistics - listening ports"),
                ("ps aux --sort=-%mem | head -20", "Top 20 memory-consuming processes"),
                ("last -n 15", "Recent login history"),
                ("cat /var/log/auth.log 2>/dev/null | tail -30 || journalctl -u ssh --no-pager | tail -30", "Recent auth logs"),
                ("who", "Currently logged in users"),
                ("w", "Logged in users and their activity"),
                ("lsof -i -P -n | head -30", "Open network connections"),
                ("find /tmp -type f -mmin -30 2>/dev/null | head -20", "Recently modified files in /tmp"),
                ("cat /etc/passwd | grep -v nologin | grep -v false", "Users with shell access"),
                ("crontab -l 2>/dev/null || echo 'No crontab'", "Scheduled tasks"),
                ("systemctl list-units --type=service --state=running | head -20", "Running services"),
            ],
        },
        "OrionAgent": {
            "role": "strategic",
            "description": "🎯 Strategic Coordination - comprehensive analysis, attack planning",
            "primary_phases": [AttackPhase.RECON, AttackPhase.ENUMERATION, AttackPhase.EXPLOITATION],
            "preferred_commands": [
                "nmap_vuln_scan", "nikto", "enum4linux", "bloodhound_collection",
                "ldapsearch_base", "snmpwalk", "rpcclient_enum", "smbclient_list",
            ],
            "command_tags": {"comprehensive", "enum", "analysis", "vuln"},
            "avoid_tags": {"defense"},
            "is_coordinator": True,  # Orion coordinates other agents
        },
        "ShadowAgent": {
            "role": "stealth",
            "description": "👤 Stealth Operations - passive recon, persistence, evasion",
            "primary_phases": [AttackPhase.ENUMERATION, AttackPhase.POST_EXPLOITATION, AttackPhase.EXFILTRATION],
            "preferred_commands": [
                "curl_headers", "wget_download", "nc_listener", "socat_tunnel",
                "ssh_key_login", "chisel_client", "linpeas", "pspy",
            ],
            "command_tags": {"stealth", "evasion", "quiet", "passive", "persistence", "exfil"},
            "avoid_tags": {"loud", "aggressive", "bruteforce", "scanning"},
            "stealth_mode": True,
        },
    }
    
    def __init__(
        self,
        agent_name: str,
        gpt_manager: "GPTManager",
        mentor_policy: Optional[MentorPolicy] = None,
        skill_library: Optional["SkillLibrary"] = None,
        trace_writer: Optional["TraceWriter"] = None,
        learned_store: Optional[LearnedCommandStore] = None,
        reward_calculator: Optional[SmartRewardCalculator] = None,
        mentor_log_path: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ):
        self.agent_name = agent_name
        self.gpt_manager = gpt_manager
        self.mentor_policy = mentor_policy or MentorPolicy()
        self.skill_library = skill_library
        self.trace_writer = trace_writer
        self.mentor_log_path = mentor_log_path
        self.model = model
        
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
        
        logger.info(f"SmartCoach initialized for {agent_name} | Role: {self.agent_role['role']} | {self.agent_role['description']}")
    
    def _init_smart_mentor(self):
        """Initialize the smart mentor with LLM client."""
        try:
            if hasattr(self.gpt_manager, 'client') and self.gpt_manager.client:
                self.smart_mentor = SmartMentor(
                    llm_client=self.gpt_manager.client,
                    learned_store=self.learned_store,
                    model=self.model,
                )
                logger.debug(f"SmartMentor initialized for {self.agent_name}")
            else:
                logger.debug(f"No LLM client available, SmartMentor disabled")
        except Exception as e:
            logger.warning(f"Failed to init SmartMentor: {e}")
    
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
        
        # Track command history
        if "last_command" in state:
            ctx.command_history.append(state["last_command"])
        
        # Update phase based on state
        ctx.current_phase = get_phase_from_state(ctx.state_flags)
        
        return ctx
    
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
        
        # Track previous phase for transition detection
        prev_phase = getattr(self, '_last_phase', None)
        current_phase = ctx.current_phase
        self._last_phase = current_phase
        
        # HYBRID MODE: Determine if we need GPT
        should_call_gpt = False
        gpt_reason = None
        
        # Condition A: Force mentor (e.g., stuck agent)
        if force_mentor or confidence < 0.15:
            should_call_gpt = True
            gpt_reason = "forced_mentor" if force_mentor else "low_confidence"
        
        # Condition B: Phase transition - need strategic planning
        elif prev_phase is not None and current_phase != prev_phase:
            should_call_gpt = True
            gpt_reason = f"phase_transition:{prev_phase.name}->{current_phase.name}"
        
        # Condition C: Check if we're stuck (recent rewards negative)
        elif self.reward_calculator.is_stuck():
            should_call_gpt = True
            gpt_reason = "stuck_low_rewards"
        
        # Condition D: No valid registry commands for current state
        else:
            valid_commands = get_valid_commands_for_state(ctx.state_flags, ctx.current_phase)
            if not valid_commands:
                should_call_gpt = True
                gpt_reason = "no_registry_match"
        
        # Check if GPT is available
        gpt_available = (
            self.smart_mentor is not None
            and self.gpt_manager is not None
            and not self.gpt_manager.is_offline()
        )
        
        # Make decision based on hybrid logic
        if should_call_gpt and gpt_available:
            logger.debug(f"[{self.agent_name}] GPT call triggered: {gpt_reason}")
            result = self._decide_with_mentor(step_ctx, proposed_action, confidence)
            result.mentor_reasoning = f"[{gpt_reason}] {result.mentor_reasoning or ''}"
        else:
            # Registry-first: efficient, no token usage
            result = self._decide_from_registry(step_ctx, proposed_action, confidence)
            if should_call_gpt and not gpt_available:
                logger.debug(f"[{self.agent_name}] GPT needed but unavailable, using registry")
        
        # Record decision
        self.decisions.append(result)
        
        # Log mentor call if applicable
        if result.mentor_call and self.mentor_log_path:
            self._log_mentor_call(step_ctx, proposed_action, result)
        
        return result
    
    def _filter_commands_for_role(self, commands: List[CommandTemplate]) -> List[CommandTemplate]:
        """
        Filter commands based on agent's role.
        
        Each agent has:
        - preferred_commands: Commands they should prioritize
        - command_tags: Tags they should look for
        - avoid_tags: Tags they should NOT use
        """
        role = self.agent_role
        preferred_names = set(role.get("preferred_commands", []))
        wanted_tags = role.get("command_tags", set())
        avoid_tags = role.get("avoid_tags", set())
        primary_phases = role.get("primary_phases", [])
        
        filtered = []
        for cmd in commands:
            # Skip commands with avoided tags
            if cmd.tags & avoid_tags:
                continue
            
            # Skip commands already used this step (deduplication)
            if cmd.name in self.step_used_commands:
                continue
            
            # Prioritize preferred commands
            if cmd.name in preferred_names:
                filtered.insert(0, cmd)  # Add to front
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
                (f"nikto -h http://{ctx.target}", "🎯 Strategy: Web vulnerability scan"),
                (f"enum4linux -a {ctx.target}", "🎯 Strategy: SMB enumeration - Windows likely"),
                (f"nmap --script=vuln {ctx.target}", "🎯 Strategy: Vulnerability scanning"),
                (f"gobuster dir -u http://{ctx.target} -w /usr/share/wordlists/dirb/common.txt -q", "🎯 Strategy: Web directory enumeration"),
                (f"whatweb -v {ctx.target}", "🎯 Strategy: Web technology fingerprinting"),
            ]
        elif phase == AttackPhase.EXPLOITATION:
            strategic_pool = [
                (f"searchsploit --update 2>/dev/null; searchsploit linux kernel", "🎯 Strategy: Search for kernel exploits"),
                (f"msfconsole -q -x 'search type:exploit; exit'", "🎯 Strategy: Metasploit exploit enumeration"),
                (f"crackmapexec smb {ctx.target} --shares", "🎯 Strategy: SMB share access check"),
                (f"hydra -L /usr/share/wordlists/rockyou.txt -P /usr/share/wordlists/rockyou.txt ssh://{ctx.target}", "🎯 Strategy: Credential attack"),
            ]
        else:
            strategic_pool = [
                (f"echo '[Orion] Phase: {phase.name} | Ready to coordinate'", f"🎯 Strategy: {phase.name} phase coordination"),
                (f"nmap -sV {ctx.target}", "🎯 Strategy: Re-scan to update intel"),
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
        
        if not filtered_commands:
            # Fallback based on role with variety
            step = step_ctx.step
            
            if role_name == "recon":
                recon_fallbacks = [
                    f"nmap -sT --top-ports 100 {ctx.target}",
                    f"nmap -sV {ctx.target}",
                    f"masscan -p1-1000 --rate=500 {ctx.target}",
                    f"nmap -sU --top-ports 20 {ctx.target}",
                ]
                return SmartDecisionResult(
                    command=recon_fallbacks[step % len(recon_fallbacks)],
                    template_name="nmap_top_ports",
                    params={"target": ctx.target},
                    mentor_call=False,
                    mentor_reasoning="🔍 Scout fallback: port scanning",
                    confidence=0.4,
                    phase=AttackPhase.RECON,
                )
            elif role_name == "offensive":
                # Red agent aggressive fallbacks - changes each step
                red_fallbacks = [
                    (f"nmap --script=vuln {ctx.target}", "⚔️ Vuln scan for exploits"),
                    (f"nikto -h http://{ctx.target} 2>/dev/null || echo 'No HTTP'", "⚔️ Web vuln scan"),
                    (f"searchsploit linux kernel 5", "⚔️ Kernel exploit search"),
                    (f"gobuster dir -u http://{ctx.target} -w /usr/share/wordlists/dirb/common.txt -q 2>/dev/null || echo 'No HTTP'", "⚔️ Directory brute force"),
                    (f"hydra -l admin -P /usr/share/wordlists/rockyou.txt -t 4 ssh://{ctx.target} 2>/dev/null || echo 'SSH attack'", "⚔️ SSH credential attack"),
                    (f"crackmapexec smb {ctx.target} --shares 2>/dev/null || echo 'No SMB'", "⚔️ SMB share enum"),
                ]
                cmd, reason = red_fallbacks[step % len(red_fallbacks)]
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
                stealth_fallbacks = [
                    (f"curl -s -I http://{ctx.target} 2>/dev/null | head -20", "👤 Passive HTTP headers"),
                    (f"dig {ctx.target} ANY +noall +answer", "👤 DNS record query"),
                    (f"wget -q --spider http://{ctx.target}", "👤 Quiet web check"),
                    (f"nc -zv {ctx.target} 22 2>&1 | head -1", "👤 Quiet SSH probe"),
                ]
                cmd, reason = stealth_fallbacks[step % len(stealth_fallbacks)]
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
                return SmartDecisionResult(
                    command=f"nmap -sT {ctx.target}",
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
        # RANDOMIZED SELECTION FROM TOP CANDIDATES (avoid determinism)
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
            # Missing required param, use fallback
            command = f"nmap -sT {ctx.target}"
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
    
    def _decide_with_mentor(
        self,
        step_ctx: SmartStepContext,
        proposed_action: Optional[str],
        confidence: float,
    ) -> SmartDecisionResult:
        """
        Make decision using the smart mentor (LLM).
        """
        ctx = step_ctx.attack_context
        
        try:
            # Get command from smart mentor
            mentor_response = self.smart_mentor.get_command(ctx)
            
            # Track command in history
            ctx.command_history.append(mentor_response.command)
            
            # Determine if mentor changed the action
            delta = "changed" if proposed_action and mentor_response.command != proposed_action else "kept"
            
            return SmartDecisionResult(
                command=mentor_response.command,
                template_name=mentor_response.template_name,
                params=mentor_response.params,
                mentor_call=True,
                model_used=self.model,
                mentor_reasoning=mentor_response.reasoning,
                mentor_delta=delta,
                confidence=mentor_response.confidence,
                phase=mentor_response.phase,
            )
            
        except Exception as e:
            logger.warning(f"Smart mentor failed: {e}, falling back to registry")
            return self._decide_from_registry(step_ctx, proposed_action, confidence)
    
    def record_result(
        self,
        decision: SmartDecisionResult,
        success: bool,
        raw_output: str,
        new_discoveries: Optional[Dict[str, Any]] = None,
    ) -> RewardBreakdown:
        """
        Record the result of a command execution and calculate reward.
        
        Args:
            decision: The decision that was executed
            success: Whether command succeeded
            raw_output: Raw output from command
            new_discoveries: New things discovered
            
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
        
        # Add failed command to context if failed
        if not success:
            self.attack_context.failed_attempts.append(decision.command)
        
        # Record outcome for adaptive mentor policy learning
        self.mentor_policy.record_outcome(
            agent_name=self.agent_name,
            reward=breakdown.total,
            used_mentor=decision.mentor_call,
        )
        
        return breakdown
    
    def reset_episode(self, episode: int):
        """Reset for new episode."""
        self.current_episode = episode
        self.mentor_policy.reset_episode(episode)
        self.reward_calculator.reset()
        
        # Reset episode-level command tracking for variety
        self.episode_used_commands.clear()
        self.command_repeat_count.clear()
        self.decisions.clear()  # Clear decision history for fresh episode
        
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
    model: str = "gpt-4o-mini",
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
