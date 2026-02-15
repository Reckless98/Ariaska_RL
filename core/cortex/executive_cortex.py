"""
Executive Cortex — Episode-level strategic planner for Ariaska_RL.

Phase 9.3: Generates and maintains an AttackPlan at the episode level.
Operates at a higher abstraction than TacticalCortex — plans across
phases rather than evaluating individual commands.

Architecture:
  - ExecutiveCortex: Strategic planning engine
  - AttackPlan: Multi-phase plan with prioritized objectives
  - PhaseObjective: Per-phase goals and success criteria
  - PlanRevision: Tracks plan changes triggered by phase transitions

Integration:
  SmartOrchestrator.run_episode() calls:
    1. ExecutiveCortex.create_plan() at episode start
    2. ExecutiveCortex.revise_plan() on phase transitions
    3. ExecutiveCortex.get_phase_guidance() before each step block
    4. ExecutiveCortex.end_episode() for metrics

Rule-based planning with optional LLM enhancement:
  - LLM called only at episode start and on major plan revisions
  - Maximum 3 LLM calls per episode for planning
  - All plans work fully offline with rule-based defaults

Usage:
    from core.cortex.executive_cortex import ExecutiveCortex
    cortex = ExecutiveCortex(gpt_manager=gpt, target_profile=profile)
    plan = cortex.create_plan(initial_state, target_ip)
    guidance = cortex.get_phase_guidance("ENUMERATION")
    cortex.revise_plan("EXPLOITATION", discovery_board)
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from core.gpt_manager import GPTManager
    from core.knowledge.target_profiler import TargetProfile

logger = logging.getLogger("ariaska.executive_cortex")


# ─── Data Structures ────────────────────────────────────────────────────────

class PlanPriority(Enum):
    """Priority levels for objectives."""
    CRITICAL = auto()   # Must achieve for success
    HIGH = auto()       # Strongly recommended
    MEDIUM = auto()     # Beneficial if time allows
    LOW = auto()        # Nice to have


@dataclass
class PhaseObjective:
    """A specific objective within an attack phase."""
    objective_id: str
    phase: str
    description: str
    priority: PlanPriority
    target_commands: List[str] = field(default_factory=list)  # Recommended command templates
    success_criteria: List[str] = field(default_factory=list)  # State flags that indicate completion
    max_steps: int = 15          # Budget for this objective
    completed: bool = False
    steps_spent: int = 0


@dataclass
class PlanRevision:
    """Record of a plan revision event."""
    trigger: str                 # "phase_transition", "stagnation", "discovery", "llm"
    old_phase: str
    new_phase: str
    changes: List[str]           # Human-readable list of changes
    timestamp: float = field(default_factory=time.time)


@dataclass
class AttackPlan:
    """
    Complete attack plan for an episode.
    
    Contains ordered objectives across attack phases,
    with priorities and step budgets.
    """
    target_ip: str
    target_type: str = "unknown"   # "ms2", "ms3", "ad", "htb", "generic"
    objectives: List[PhaseObjective] = field(default_factory=list)
    revisions: List[PlanRevision] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    llm_enhanced: bool = False

    @property
    def current_objectives(self) -> List[PhaseObjective]:
        """Get incomplete objectives ordered by priority."""
        return [
            o for o in self.objectives
            if not o.completed
        ]

    def get_phase_objectives(self, phase: str) -> List[PhaseObjective]:
        """Get objectives for a specific phase."""
        return [o for o in self.objectives if o.phase == phase]

    def mark_completed(self, objective_id: str) -> None:
        """Mark an objective as completed."""
        for obj in self.objectives:
            if obj.objective_id == objective_id:
                obj.completed = True
                break

    def add_revision(self, revision: PlanRevision) -> None:
        """Record a plan revision."""
        self.revisions.append(revision)

    @property
    def completion_ratio(self) -> float:
        """Fraction of objectives completed."""
        if not self.objectives:
            return 0.0
        return sum(1 for o in self.objectives if o.completed) / len(self.objectives)


# ─── Executive Cortex ────────────────────────────────────────────────────────

class ExecutiveCortex:
    """
    Episode-level strategic planning engine.
    
    Generates multi-phase attack plans and revises them
    as the engagement progresses.
    
    Args:
        gpt_manager: Optional GPT manager for LLM-enhanced planning.
        target_profile: Optional target profile from TargetProfiler.
        max_llm_calls: Maximum LLM calls for planning per episode.
        enable_llm: Whether LLM enhancement is enabled.
    """

    # ── Standard attack phase templates ──
    _PHASE_TEMPLATES = {
        "RECON": {
            "objectives": [
                ("recon_host_discovery", "Discover alive hosts and open ports",
                 PlanPriority.CRITICAL,
                 ["nmap_quick_scan", "nmap_top_ports", "masscan_fast"],
                 ["ports_discovered"], 8),
                ("recon_service_enum", "Identify service versions on open ports",
                 PlanPriority.HIGH,
                 ["nmap_service_version", "nmap_os_detection"],
                 ["services_identified"], 6),
            ],
        },
        "ENUMERATION": {
            "objectives": [
                ("enum_web", "Enumerate web application directories and parameters",
                 PlanPriority.HIGH,
                 ["gobuster_dir", "nikto_scan", "ffuf_fuzz", "whatweb"],
                 ["web_dirs_found"], 10),
                ("enum_smb", "Enumerate SMB shares and users",
                 PlanPriority.HIGH,
                 ["enum4linux_full", "smbclient_list", "smbmap_shares"],
                 ["smb_enumerated"], 6),
                ("enum_users", "Discover valid usernames",
                 PlanPriority.MEDIUM,
                 ["rpcclient_enumdomusers", "ldapsearch_users", "hydra_ssh"],
                 ["users_enumerated"], 6),
                ("enum_vulns", "Identify known vulnerabilities",
                 PlanPriority.HIGH,
                 ["searchsploit", "nmap_vuln_scan"],
                 ["vulns_identified"], 5),
            ],
        },
        "EXPLOITATION": {
            "objectives": [
                ("exploit_easy_wins", "Exploit low-hanging fruit (backdoors, default creds)",
                 PlanPriority.CRITICAL,
                 ["vsftpd_exploit", "unrealircd_exploit", "telnet_1524",
                  "rsh_root", "rlogin_root", "psql_rce"],
                 ["shell_obtained"], 10),
                ("exploit_web", "Exploit web application vulnerabilities",
                 PlanPriority.HIGH,
                 ["sqlmap_get", "lfi_etc_passwd", "cmd_inject_semicolon",
                  "ssti_detect_jinja2", "shellshock_cgi"],
                 ["web_exploited"], 12),
                ("exploit_auth", "Brute-force or spray credentials",
                 PlanPriority.MEDIUM,
                 ["hydra_ssh", "hydra_ftp", "hydra_smb", "cme_smb_bruteforce"],
                 ["credentials_found"], 8),
                ("exploit_services", "Exploit specific service vulnerabilities",
                 PlanPriority.HIGH,
                 ["msfconsole_exploit", "war_deploy", "nfs_mount"],
                 ["service_exploited"], 10),
            ],
        },
        "PRIVILEGE_ESCALATION": {
            "objectives": [
                ("privesc_enum", "Enumerate privilege escalation vectors",
                 PlanPriority.CRITICAL,
                 ["linpeas", "sudo_list", "find_suid", "find_capabilities"],
                 ["privesc_vectors_found"], 8),
                ("privesc_execute", "Execute privilege escalation",
                 PlanPriority.CRITICAL,
                 ["sudo_check", "kernel_exploit_check", "docker_privesc",
                  "writable_etc_passwd"],
                 ["root_obtained"], 10),
            ],
        },
        "LATERAL_MOVEMENT": {
            "objectives": [
                ("lateral_pivot", "Establish pivot to internal network",
                 PlanPriority.HIGH,
                 ["chisel_server", "ssh_tunnel_local", "ssh_tunnel_dynamic"],
                 ["pivot_established"], 6),
                ("lateral_move", "Move to additional targets",
                 PlanPriority.MEDIUM,
                 ["ssh_lateral", "impacket_pth_psexec", "crackmapexec_pth"],
                 ["lateral_access"], 8),
            ],
        },
        "EXFILTRATION": {
            "objectives": [
                ("exfil_creds", "Exfiltrate credentials and sensitive data",
                 PlanPriority.CRITICAL,
                 ["dump_shadow", "dump_passwd", "exfil_shadow",
                  "exfil_ssh_keys", "credential_dump"],
                 ["data_exfiltrated"], 8),
                ("exfil_data", "Exfiltrate application and database data",
                 PlanPriority.HIGH,
                 ["exfil_mysql_dump", "nc_exfil", "curl_exfil"],
                 ["app_data_exfiltrated"], 6),
            ],
        },
        "POST_EXPLOITATION": {
            "objectives": [
                ("post_persist", "Establish persistence mechanisms",
                 PlanPriority.HIGH,
                 ["cron_backdoor", "ssh_key_persistence", "plant_ssh_key"],
                 ["persistence_established"], 6),
                ("post_harvest", "Harvest additional credentials and data",
                 PlanPriority.MEDIUM,
                 ["history_dump", "network_config_dump", "ssh_key_harvest"],
                 ["post_harvest_done"], 6),
            ],
        },
        "CLOSEOUT": {
            "objectives": [
                ("closeout_cleanup", "Remove tools and artifacts",
                 PlanPriority.HIGH,
                 ["remove_uploaded_tools", "cleanup_tmp_artifacts",
                  "remove_ssh_keys_planted", "remove_cron_backdoors"],
                 ["cleanup_done"], 6),
                ("closeout_verify", "Verify target stability and generate report",
                 PlanPriority.CRITICAL,
                 ["verify_target_stable", "generate_report"],
                 ["report_generated"], 4),
            ],
        },
    }

    # ── Metasploitable 2 specific plan overrides ──
    _MS2_PRIORITY_OVERRIDES = {
        "exploit_easy_wins": PlanPriority.CRITICAL,  # MS2 has many easy backdoors
        "exploit_web": PlanPriority.MEDIUM,           # Less productive on MS2
        "enum_smb": PlanPriority.CRITICAL,            # Samba 3.0.20 is exploitable
    }

    _MS2_EXTRA_OBJECTIVES = [
        PhaseObjective(
            objective_id="ms2_backdoors",
            phase="EXPLOITATION",
            description="Exploit MS2 backdoors: vsftpd 2.3.4, UnrealIRCd, ingreslock, rservices",
            priority=PlanPriority.CRITICAL,
            target_commands=[
                "vsftpd_exploit", "unrealircd_exploit", "telnet_1524",
                "rsh_root", "rlogin_root",
            ],
            success_criteria=["shell_obtained", "root_shell"],
            max_steps=8,
        ),
        PhaseObjective(
            objective_id="ms2_services",
            phase="EXPLOITATION",
            description="Exploit MS2 services: PostgreSQL RCE, Tomcat manager, NFS mount",
            priority=PlanPriority.HIGH,
            target_commands=[
                "psql_rce", "war_deploy", "nfs_mount", "ssh_key_plant",
                "psql_default_creds", "tomcat_cred_test",
            ],
            success_criteria=["service_exploited"],
            max_steps=10,
        ),
    ]

    def __init__(
        self,
        gpt_manager: Optional[GPTManager] = None,
        target_profile: Optional[TargetProfile] = None,
        max_llm_calls: int = 3,
        enable_llm: bool = True,
        knowledge_retriever: Optional[Any] = None,
    ):
        self._gpt_manager = gpt_manager
        self._target_profile = target_profile
        self._max_llm_calls = max_llm_calls
        self._enable_llm = enable_llm
        self._knowledge_retriever = knowledge_retriever
        self._llm_calls_this_episode = 0
        self._plan: Optional[AttackPlan] = None
        self._phase_step_counts: Dict[str, int] = defaultdict(int)

    # ─── Public API ──────────────────────────────────────────────────────────

    def create_plan(
        self,
        initial_state: Dict[str, Any],
        target_ip: str = "10.0.0.1",
        target_type: str = "unknown",
        max_steps: int = 120,
    ) -> AttackPlan:
        """
        Create an attack plan for a new episode.
        
        Args:
            initial_state: Initial environment state.
            target_ip: Target IP address.
            target_type: Target classification ("ms2", "ms3", "ad", etc.)
            max_steps: Maximum steps in the episode.
            
        Returns:
            AttackPlan with ordered objectives.
        """
        self._llm_calls_this_episode = 0
        self._phase_step_counts.clear()

        # Detect target type from profile if available
        if target_type == "unknown" and self._target_profile is not None:
            target_type = getattr(self._target_profile, "target_type", "unknown")

        plan = AttackPlan(
            target_ip=target_ip,
            target_type=target_type,
        )

        # Build objectives from phase templates
        step_budget_remaining = max_steps
        for phase_name, template in self._PHASE_TEMPLATES.items():
            for obj_tuple in template["objectives"]:
                oid, desc, priority, cmds, criteria, steps = obj_tuple
                
                # Apply target-specific priority overrides
                if target_type == "ms2" and oid in self._MS2_PRIORITY_OVERRIDES:
                    priority = self._MS2_PRIORITY_OVERRIDES[oid]

                # Scale step budget based on remaining steps
                scaled_steps = min(steps, max(3, step_budget_remaining // 4))

                plan.objectives.append(PhaseObjective(
                    objective_id=oid,
                    phase=phase_name,
                    description=desc,
                    priority=priority,
                    target_commands=cmds,
                    success_criteria=criteria,
                    max_steps=scaled_steps,
                ))
                step_budget_remaining -= scaled_steps

        # Add target-specific extra objectives
        if target_type == "ms2":
            for extra in self._MS2_EXTRA_OBJECTIVES:
                # Deep copy to avoid shared state
                plan.objectives.append(PhaseObjective(
                    objective_id=extra.objective_id,
                    phase=extra.phase,
                    description=extra.description,
                    priority=extra.priority,
                    target_commands=list(extra.target_commands),
                    success_criteria=list(extra.success_criteria),
                    max_steps=extra.max_steps,
                ))

        # Optional LLM enhancement at episode start
        if (
            self._enable_llm
            and self._gpt_manager is not None
            and not self._gpt_manager.is_offline()
            and self._llm_calls_this_episode < self._max_llm_calls
        ):
            self._llm_enhance_plan(plan, initial_state)
            plan.llm_enhanced = True
            self._llm_calls_this_episode += 1

        self._plan = plan
        logger.info(
            f"[EXECUTIVE] Created plan: {len(plan.objectives)} objectives, "
            f"target={target_type}, llm={plan.llm_enhanced}"
        )
        return plan

    def revise_plan(
        self,
        new_phase: str,
        discovery_board: Dict[str, Any],
        old_phase: str = "",
        step: int = 0,
    ) -> Optional[PlanRevision]:
        """
        Revise the plan on a phase transition.
        
        Args:
            new_phase: The phase we're transitioning to.
            discovery_board: Current shared discovery state.
            old_phase: Phase we're leaving.
            step: Current step number.
            
        Returns:
            PlanRevision record, or None if no revision needed.
        """
        if self._plan is None:
            return None

        changes: List[str] = []

        # Auto-complete objectives from the old phase
        for obj in self._plan.get_phase_objectives(old_phase):
            if not obj.completed:
                # Check if success criteria are met
                if self._criteria_met(obj.success_criteria, discovery_board):
                    obj.completed = True
                    changes.append(f"Auto-completed '{obj.objective_id}' (criteria met)")
                elif obj.steps_spent >= obj.max_steps:
                    obj.completed = True  # Budget exhausted
                    changes.append(f"Budget-closed '{obj.objective_id}' (max steps reached)")

        # Boost priority of upcoming objectives based on discoveries
        shells = discovery_board.get("shells", set())
        creds = discovery_board.get("credentials", set())
        ports = discovery_board.get("ports", set())

        if shells:
            # We have a shell — privesc objectives become critical
            for obj in self._plan.get_phase_objectives("PRIVILEGE_ESCALATION"):
                if obj.priority != PlanPriority.CRITICAL:
                    obj.priority = PlanPriority.CRITICAL
                    changes.append(f"Promoted '{obj.objective_id}' to CRITICAL (shell obtained)")

        if creds and new_phase in ("EXPLOITATION", "LATERAL_MOVEMENT"):
            # We have creds — exploitation with known creds is high priority
            for obj in self._plan.get_phase_objectives("EXPLOITATION"):
                if "auth" in obj.objective_id and obj.priority != PlanPriority.CRITICAL:
                    obj.priority = PlanPriority.CRITICAL
                    changes.append(f"Promoted '{obj.objective_id}' to CRITICAL (creds found)")

        if not changes:
            return None

        revision = PlanRevision(
            trigger="phase_transition",
            old_phase=old_phase,
            new_phase=new_phase,
            changes=changes,
        )
        self._plan.add_revision(revision)

        logger.debug(
            f"[EXECUTIVE] Plan revised ({old_phase}→{new_phase}): "
            f"{len(changes)} changes"
        )
        return revision

    def get_phase_guidance(
        self,
        current_phase: str,
        step: int = 0,
    ) -> Dict[str, Any]:
        """
        Get guidance for the current phase.
        
        Returns a dict with:
          - objectives: List of active objectives for this phase
          - recommended_commands: Priority-ordered command templates
          - step_budget: Remaining steps for this phase's objectives
          - focus: One-line focus description
        """
        if self._plan is None:
            return {
                "objectives": [],
                "recommended_commands": [],
                "step_budget": 15,
                "focus": f"Default {current_phase} operations",
            }

        phase_objs = [
            o for o in self._plan.get_phase_objectives(current_phase)
            if not o.completed
        ]

        # Collect recommended commands in priority order
        recommended: List[str] = []
        total_budget = 0
        for obj in sorted(phase_objs, key=lambda o: o.priority.value):
            for cmd in obj.target_commands:
                if cmd not in recommended:
                    recommended.append(cmd)
            total_budget += max(0, obj.max_steps - obj.steps_spent)

        # Build focus string
        if phase_objs:
            top = phase_objs[0]
            focus = f"{top.description} ({top.priority.name})"
        else:
            focus = f"{current_phase}: all objectives completed or budgeted"

        return {
            "objectives": [
                {
                    "id": o.objective_id,
                    "description": o.description,
                    "priority": o.priority.name,
                    "commands": o.target_commands,
                    "budget": max(0, o.max_steps - o.steps_spent),
                    "completed": o.completed,
                }
                for o in phase_objs
            ],
            "recommended_commands": recommended[:10],
            "step_budget": total_budget,
            "focus": focus,
            "kr_enriched": self._kr_enrich_recommendations(
                current_phase, recommended,
            ),
        }

    def _kr_enrich_recommendations(
        self,
        phase: str,
        existing: List[str],
    ) -> List[str]:
        """Enrich command recommendations using KnowledgeRetriever."""
        if self._knowledge_retriever is None:
            return []
        try:
            phase_entries = self._knowledge_retriever.by_phase(phase, max_results=5)
            kr_commands = []
            for entry in phase_entries:
                cmds = entry.get("commands", []) or entry.get("exploitation_commands", [])
                for cmd in cmds[:2]:
                    template_name = cmd.get("template_name", "")
                    if template_name and template_name not in existing and template_name not in kr_commands:
                        kr_commands.append(template_name)
            return kr_commands[:5]
        except Exception:
            return []

    def record_step(
        self,
        phase: str,
        template_name: str,
        had_discovery: bool = False,
        discovery_board: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a step for budget tracking and auto-completion."""
        self._phase_step_counts[phase] += 1

        if self._plan is None:
            return

        # Increment step count on matching objectives
        for obj in self._plan.get_phase_objectives(phase):
            if obj.completed:
                continue
            if template_name in obj.target_commands:
                obj.steps_spent += 1
            # Check for auto-completion
            if discovery_board and self._criteria_met(obj.success_criteria, discovery_board):
                obj.completed = True
                logger.debug(f"[EXECUTIVE] Objective '{obj.objective_id}' auto-completed")

    def end_episode(self) -> Dict[str, Any]:
        """
        Generate end-of-episode metrics.
        
        Returns dict with plan execution metrics.
        """
        if self._plan is None:
            return {"plan_available": False}

        total = len(self._plan.objectives)
        completed = sum(1 for o in self._plan.objectives if o.completed)
        critical_total = sum(
            1 for o in self._plan.objectives if o.priority == PlanPriority.CRITICAL
        )
        critical_done = sum(
            1 for o in self._plan.objectives
            if o.priority == PlanPriority.CRITICAL and o.completed
        )

        return {
            "plan_available": True,
            "target_type": self._plan.target_type,
            "total_objectives": total,
            "completed_objectives": completed,
            "completion_ratio": self._plan.completion_ratio,
            "critical_total": critical_total,
            "critical_completed": critical_done,
            "critical_ratio": critical_done / max(critical_total, 1),
            "revisions": len(self._plan.revisions),
            "llm_enhanced": self._plan.llm_enhanced,
            "llm_calls": self._llm_calls_this_episode,
            "phase_step_counts": dict(self._phase_step_counts),
        }

    def reset_episode(self) -> None:
        """Reset for a new episode."""
        self._plan = None
        self._llm_calls_this_episode = 0
        self._phase_step_counts.clear()

    @property
    def current_plan(self) -> Optional[AttackPlan]:
        """Get the current attack plan."""
        return self._plan

    # ─── Internal Helpers ────────────────────────────────────────────────────

    def _criteria_met(
        self,
        criteria: List[str],
        discovery_board: Dict[str, Any],
    ) -> bool:
        """Check if success criteria are met based on discovery board."""
        if not criteria:
            return False

        flags = discovery_board.get("flags_set", set())
        if isinstance(flags, list):
            flags = set(flags)

        # Also check discovery board directly
        ports = discovery_board.get("ports", set())
        services = discovery_board.get("services", set())
        creds = discovery_board.get("credentials", set())
        shells = discovery_board.get("shells", set())
        vulns = discovery_board.get("vulns", set())

        for criterion in criteria:
            c = criterion.lower()
            if c in flags:
                return True
            if c == "ports_discovered" and ports:
                return True
            if c == "services_identified" and services:
                return True
            if c == "credentials_found" and creds:
                return True
            if c == "shell_obtained" and shells:
                return True
            if c in ("root_obtained", "root_shell") and any(
                "root" in str(s).lower() for s in shells
            ):
                return True
            if c == "vulns_identified" and vulns:
                return True
            if c == "data_exfiltrated" and "data_exfiltrated" in flags:
                return True

        return False

    def _llm_enhance_plan(
        self,
        plan: AttackPlan,
        state: Dict[str, Any],
    ) -> None:
        """Optionally enhance the plan with LLM insight at episode start."""
        if self._gpt_manager is None:
            return

        objectives_summary = "\n".join(
            f"  [{o.priority.name}] {o.objective_id}: {o.description}"
            for o in plan.objectives[:10]
        )

        prompt = (
            f"ATTACK PLAN REVIEW — Target: {plan.target_ip} ({plan.target_type})\n"
            f"Objectives (first 10):\n{objectives_summary}\n\n"
            f"Given target type '{plan.target_type}', which 3 objectives should be "
            f"HIGHEST priority? Reply with objective IDs, comma-separated.\n"
            f"Then give ONE strategic tip (1 sentence)."
        )

        try:
            response = self._gpt_manager.gpt_request(
                prompt,
                task_type="strategic",
                agent_id="executive_cortex",
                max_tokens=80,
            )
            if response:
                # Parse response — look for objective IDs to boost
                for obj in plan.objectives:
                    if obj.objective_id in response:
                        obj.priority = PlanPriority.CRITICAL
                logger.debug(f"[EXECUTIVE] LLM plan enhancement: {response[:100]}")
        except Exception as e:
            logger.debug(f"[EXECUTIVE] LLM enhancement failed: {e}")

    # ─── Query Support ───────────────────────────────────────────────────────

    def get_recommended_commands_for_state(
        self,
        current_phase: str,
        discovery_board: Dict[str, Any],
        step: int = 0,
    ) -> List[str]:
        """
        Get priority-ordered command recommendations based on plan + state.
        
        Useful for SmartCoach to bias registry selection toward plan objectives.
        """
        guidance = self.get_phase_guidance(current_phase, step)
        recommended = guidance.get("recommended_commands", [])

        # Filter out commands that might not be productive
        # (e.g., if we already have shells, don't recommend scanning)
        shells = discovery_board.get("shells", set())
        if shells and current_phase in ("RECON", "ENUMERATION"):
            # Deprioritize scanning commands if we have shells
            exploit_cmds = [c for c in recommended if "exploit" in c or "rce" in c]
            scan_cmds = [c for c in recommended if c not in exploit_cmds]
            recommended = exploit_cmds + scan_cmds

        return recommended
