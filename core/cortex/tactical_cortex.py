"""
Tactical Cortex — Per-step tactical assessment for Ariaska_RL.

Phase 9.3: Evaluates each proposed command BEFORE execution using
rule-based heuristics and optional LLM escalation. Acts as a
quality gate between SmartCoach.decide() and command execution.

Architecture:
  - TacticalCortex: Main assessment engine
  - TacticalAssessment: Structured result of each evaluation
  - TacticalRule: Individual rule with condition and recommendation
  
Decision Pipeline Integration:
  SmartCoach.decide() → TacticalCortex.assess() → execute if approved
  
Rule Categories:
  1. PRECONDITION CHECK — Are all preconditions truly met?
  2. SEQUENCING CHECK — Does this command follow the right predecessors?
  3. CONTRADICTION CHECK — Does not_when match current state?
  4. AGENT MISMATCH — Is this the right agent for this command?
  5. RISK ASSESSMENT — Is detection risk acceptable for this command?
  6. STAGNATION CHECK — Is agent repeating patterns without progress?
  7. OPPORTUNITY CHECK — Are higher-value alternatives available?
  
LLM Escalation (optional, max 5 calls/episode):
  Triggered ONLY when rules produce ambiguous results (confidence 0.3-0.7)
  and discovery_board shows no recent progress. Uses budget-gated
  GPT call with tight context window.

Usage:
    from core.cortex.tactical_cortex import TacticalCortex
    cortex = TacticalCortex(gpt_manager=gpt)
    assessment = cortex.assess(command, state, agent_role, history)
    if assessment.approved:
        execute(command)
    else:
        use(assessment.alternative)
"""

from __future__ import annotations

import logging
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from core.gpt_manager import GPTManager
    from core.commands.command_registry import CommandTemplate

logger = logging.getLogger("ariaska.tactical_cortex")


# ─── Enums ───────────────────────────────────────────────────────────────────

class TacticalVerdict(Enum):
    """Outcome of a tactical assessment."""
    APPROVE = auto()      # Command is appropriate, proceed
    REDIRECT = auto()     # Command is suboptimal, suggest alternative
    BLOCK = auto()        # Command is inappropriate, reject
    ESCALATE = auto()     # Ambiguous, needs LLM evaluation


class RuleCategory(Enum):
    """Categories of tactical rules."""
    PRECONDITION = "precondition"
    SEQUENCING = "sequencing"
    CONTRADICTION = "contradiction"
    AGENT_MISMATCH = "agent_mismatch"
    RISK = "risk"
    STAGNATION = "stagnation"
    OPPORTUNITY = "opportunity"


# ─── Data Structures ────────────────────────────────────────────────────────

@dataclass
class TacticalRule:
    """A single tactical rule evaluation result."""
    category: RuleCategory
    passed: bool
    severity: float = 0.0     # 0.0 = info, 0.5 = warning, 1.0 = critical
    reason: str = ""
    suggestion: str = ""      # Alternative command name if applicable


@dataclass
class TacticalAssessment:
    """Complete tactical assessment of a proposed command."""
    command: str
    template_name: str
    agent_role: str
    verdict: TacticalVerdict
    confidence: float              # 0.0-1.0, how confident cortex is in verdict
    rules_evaluated: List[TacticalRule] = field(default_factory=list)
    alternative: Optional[str] = None    # Suggested alternative command
    alternative_template: Optional[str] = None
    reasoning: str = ""
    llm_consulted: bool = False
    elapsed_ms: float = 0.0
    
    @property
    def approved(self) -> bool:
        """Whether the command should proceed."""
        return self.verdict == TacticalVerdict.APPROVE

    @property
    def critical_failures(self) -> List[TacticalRule]:
        """Rules that critically failed."""
        return [r for r in self.rules_evaluated if not r.passed and r.severity >= 0.8]

    @property
    def warnings(self) -> List[TacticalRule]:
        """Rules that generated warnings."""
        return [r for r in self.rules_evaluated if not r.passed and 0.3 <= r.severity < 0.8]


# ─── Tactical Cortex ────────────────────────────────────────────────────────

class TacticalCortex:
    """
    Per-step tactical assessment engine.
    
    Evaluates proposed commands against 7 rule categories before execution.
    Optionally escalates to LLM when rules produce ambiguous results.
    
    Args:
        gpt_manager: Optional GPT manager for LLM escalation.
        max_llm_calls: Maximum LLM calls per episode (default 5).
        enable_llm: Whether LLM escalation is enabled.
    """

    # ── Phase ordering for sequencing checks ──
    _PHASE_ORDER = {
        "RECON": 0,
        "ENUMERATION": 1,
        "EXPLOITATION": 2,
        "PRIVILEGE_ESCALATION": 3,
        "LATERAL_MOVEMENT": 4,
        "EXFILTRATION": 5,
        "POST_EXPLOITATION": 6,
        "CLOSEOUT": 7,
    }

    # ── High-value commands that should be prioritized when preconditions met ──
    _HIGH_VALUE_EXPLOITS = {
        "vsftpd_exploit", "unrealircd_exploit", "telnet_1524",
        "rsh_root", "rlogin_root", "psql_rce", "war_deploy",
        "nfs_mount", "ssh_key_plant",
    }

    # ── Commands that are noisy (IDS-detectable) ──
    _NOISY_COMMANDS = {
        "nmap_vuln_scan", "nikto_scan", "nuclei_scan", "masscan_fast",
        "nmap_comprehensive", "nmap_full_tcp", "wpscan", "dirsearch",
        "feroxbuster", "gobuster_dir",
    }

    def __init__(
        self,
        gpt_manager: Optional[GPTManager] = None,
        max_llm_calls: int = 5,
        enable_llm: bool = True,
        knowledge_retriever: Optional[Any] = None,
    ):
        self._gpt_manager = gpt_manager
        self._max_llm_calls = max_llm_calls
        self._enable_llm = enable_llm
        self._knowledge_retriever = knowledge_retriever
        self._llm_calls_this_episode = 0
        self._episode_history: List[str] = []       # Command names this episode
        self._episode_templates: List[str] = []     # Template names this episode
        self._discovery_count = 0                    # Discoveries this episode
        self._last_discovery_step = 0                # Last step with a discovery
        self._current_step = 0

    # ─── Public API ──────────────────────────────────────────────────────────

    def reset_episode(self) -> None:
        """Reset per-episode state."""
        self._llm_calls_this_episode = 0
        self._episode_history.clear()
        self._episode_templates.clear()
        self._discovery_count = 0
        self._last_discovery_step = 0
        self._current_step = 0

    def record_step(
        self,
        command: str,
        template_name: str,
        had_discovery: bool = False,
        step: int = 0,
    ) -> None:
        """Record a step's outcome for future assessments."""
        self._episode_history.append(command)
        self._episode_templates.append(template_name)
        self._current_step = step
        if had_discovery:
            self._discovery_count += 1
            self._last_discovery_step = step

    def assess(
        self,
        command: str,
        template: Optional[CommandTemplate] = None,
        state: Optional[Dict[str, Any]] = None,
        agent_role: str = "red",
        discovery_board: Optional[Dict[str, Any]] = None,
        current_phase: str = "RECON",
        detection_risk: float = 0.0,
        step: int = 0,
    ) -> TacticalAssessment:
        """
        Assess a proposed command before execution.
        
        Args:
            command: The full command string to evaluate.
            template: The CommandTemplate (if known).
            state: Current environment state dict.
            agent_role: Role of the agent proposing this command.
            discovery_board: Shared discovery state.
            current_phase: Current attack phase name.
            detection_risk: Current detection risk (0.0-1.0).
            step: Current step number.
            
        Returns:
            TacticalAssessment with verdict and recommendations.
        """
        t0 = time.time()
        self._current_step = step
        state = state or {}
        discovery_board = discovery_board or {}
        template_name = template.name if template else self._extract_template_name(command)

        rules: List[TacticalRule] = []

        # ── Rule 1: Precondition Check ──
        if template and template.preconditions:
            state_flags = state.get("flags_set", set())
            if isinstance(state_flags, list):
                state_flags = set(state_flags)
            missing = template.preconditions - state_flags
            if missing:
                rules.append(TacticalRule(
                    category=RuleCategory.PRECONDITION,
                    passed=False,
                    severity=0.9,
                    reason=f"Missing preconditions: {', '.join(sorted(missing))}",
                    suggestion=self._suggest_precondition_command(missing, current_phase),
                ))
            else:
                rules.append(TacticalRule(
                    category=RuleCategory.PRECONDITION,
                    passed=True,
                    reason="All preconditions satisfied",
                ))

        # ── Rule 2: Sequencing Check ──
        if template and template.follows_after:
            executed_templates = set(self._episode_templates)
            predecessors_met = any(
                pred in executed_templates for pred in template.follows_after
            )
            if not predecessors_met and self._current_step > 3:
                # Allow early steps to skip sequencing (bootstrap)
                rules.append(TacticalRule(
                    category=RuleCategory.SEQUENCING,
                    passed=False,
                    severity=0.5,
                    reason=(
                        f"Expected predecessors not executed: "
                        f"{', '.join(template.follows_after[:3])}"
                    ),
                    suggestion=template.follows_after[0] if template.follows_after else "",
                ))
            else:
                rules.append(TacticalRule(
                    category=RuleCategory.SEQUENCING,
                    passed=True,
                    reason="Sequencing order respected",
                ))

        # ── Rule 3: Contradiction Check (not_when) ──
        if template and template.not_when:
            contradiction = self._check_contradiction(
                template.not_when, state, discovery_board, current_phase,
            )
            if contradiction:
                rules.append(TacticalRule(
                    category=RuleCategory.CONTRADICTION,
                    passed=False,
                    severity=0.7,
                    reason=f"Contraindication active: {template.not_when[:100]}",
                    suggestion="",
                ))
            else:
                rules.append(TacticalRule(
                    category=RuleCategory.CONTRADICTION,
                    passed=True,
                    reason="No contraindications",
                ))

        # ── Rule 4: Agent Mismatch ──
        if template and template.assigned_agents:
            if agent_role.lower() not in template.assigned_agents:
                rules.append(TacticalRule(
                    category=RuleCategory.AGENT_MISMATCH,
                    passed=False,
                    severity=0.6,
                    reason=(
                        f"Command '{template_name}' assigned to "
                        f"{template.assigned_agents}, not '{agent_role}'"
                    ),
                ))
            else:
                rules.append(TacticalRule(
                    category=RuleCategory.AGENT_MISMATCH,
                    passed=True,
                    reason=f"Agent '{agent_role}' is authorized",
                ))

        # ── Rule 5: Risk Assessment ──
        if template_name in self._NOISY_COMMANDS and detection_risk > 0.6:
            rules.append(TacticalRule(
                category=RuleCategory.RISK,
                passed=False,
                severity=0.6,
                reason=(
                    f"Command '{template_name}' is noisy (IDS-detectable) "
                    f"and detection_risk={detection_risk:.1f} is high"
                ),
                suggestion="nmap_stealth_scan" if "nmap" in template_name else "",
            ))
        else:
            rules.append(TacticalRule(
                category=RuleCategory.RISK,
                passed=True,
                reason="Risk level acceptable",
            ))

        # ── Rule 6: Stagnation Check ──
        if len(self._episode_templates) >= 5:
            recent = self._episode_templates[-5:]
            unique_recent = len(set(recent))
            if unique_recent <= 2:
                rules.append(TacticalRule(
                    category=RuleCategory.STAGNATION,
                    passed=False,
                    severity=0.7,
                    reason=(
                        f"Stagnation detected: only {unique_recent} unique commands "
                        f"in last 5 steps"
                    ),
                ))
            else:
                rules.append(TacticalRule(
                    category=RuleCategory.STAGNATION,
                    passed=True,
                    reason="Command diversity acceptable",
                ))

        # ── Rule 7: Opportunity Check ──
        # If high-value exploits are available and agent is still scanning, flag it
        if template and current_phase in ("RECON", "ENUMERATION") and step > 15:
            hvt = self._find_available_high_value(state, discovery_board)
            if hvt and template_name not in self._HIGH_VALUE_EXPLOITS:
                rules.append(TacticalRule(
                    category=RuleCategory.OPPORTUNITY,
                    passed=False,
                    severity=0.4,
                    reason=(
                        f"High-value exploit '{hvt}' is available but agent "
                        f"is still in {current_phase}"
                    ),
                    suggestion=hvt,
                ))

        # ── Compute Verdict ──
        verdict, confidence = self._compute_verdict(rules)

        # ── LLM Escalation (ambiguous cases only) ──
        llm_consulted = False
        if (
            verdict == TacticalVerdict.ESCALATE
            and self._enable_llm
            and self._gpt_manager is not None
            and not self._gpt_manager.is_offline()
            and self._llm_calls_this_episode < self._max_llm_calls
        ):
            llm_result = self._llm_evaluate(
                command, template, state, agent_role,
                discovery_board, current_phase, rules,
            )
            if llm_result is not None:
                verdict, confidence, llm_consulted = llm_result
                self._llm_calls_this_episode += 1

        # ── Build alternative if blocked/redirected ──
        alternative = None
        alt_template = None
        if verdict in (TacticalVerdict.BLOCK, TacticalVerdict.REDIRECT):
            # Use suggestion from highest-severity failing rule
            failing = sorted(
                [r for r in rules if not r.passed and r.suggestion],
                key=lambda r: r.severity,
                reverse=True,
            )
            if failing:
                alt_template = failing[0].suggestion
                alternative = self._build_alternative_command(
                    alt_template, state, discovery_board,
                )

        elapsed_ms = (time.time() - t0) * 1000

        assessment = TacticalAssessment(
            command=command,
            template_name=template_name,
            agent_role=agent_role,
            verdict=verdict,
            confidence=confidence,
            rules_evaluated=rules,
            alternative=alternative,
            alternative_template=alt_template,
            reasoning=self._build_reasoning(rules, verdict),
            llm_consulted=llm_consulted,
            elapsed_ms=elapsed_ms,
        )

        logger.debug(
            f"[TACTICAL] {template_name}: {verdict.name} "
            f"(conf={confidence:.2f}, rules={len(rules)}, "
            f"fail={len([r for r in rules if not r.passed])}, "
            f"{elapsed_ms:.1f}ms)"
        )

        return assessment

    # ─── Rule Helpers ────────────────────────────────────────────────────────

    def _check_contradiction(
        self,
        not_when: str,
        state: Dict[str, Any],
        discovery_board: Dict[str, Any],
        current_phase: str,
    ) -> bool:
        """Check if not_when conditions are currently active."""
        nw = not_when.lower()
        
        # Check for "already have root" / "already have shell"
        shells = discovery_board.get("shells", set())
        if isinstance(shells, list):
            shells = set(shells)
        if ("already have root" in nw or "already have system" in nw):
            if any("root" in str(s).lower() for s in shells):
                return True
        if "already have" in nw and "shell" in nw:
            if shells:
                return True

        # Check for "not windows" / "not linux"
        state_flags = state.get("flags_set", set())
        if isinstance(state_flags, list):
            state_flags = set(state_flags)
        if "not windows" in nw or "target is not windows" in nw:
            # If we know it's Linux, this contradiction fires
            if "os_linux" in state_flags or not any("windows" in str(f) for f in state_flags):
                pass  # Can't confirm without positive OS info
        if "not linux" in nw or "target is not linux" in nw:
            if "os_windows" in state_flags:
                return True

        # Check for port-specific contradictions
        ports = discovery_board.get("ports", set())
        if isinstance(ports, list):
            ports = set(ports)
        port_patterns = re.findall(r'port\s+(\d+)\s+(?:not open|closed)', nw)
        for port in port_patterns:
            if port not in {str(p) for p in ports}:
                return True

        # Check "already ran X"
        if "already ran" in nw:
            for tmpl in self._episode_templates:
                if tmpl.lower() in nw:
                    return True

        # Check "credentials already obtained/known"
        creds = discovery_board.get("credentials", set())
        if isinstance(creds, list):
            creds = set(creds)
        if ("credentials already" in nw or "already have valid credentials" in nw):
            if creds:
                return True

        return False

    def _suggest_precondition_command(
        self,
        missing: Set[str],
        current_phase: str,
    ) -> str:
        """Suggest a command that satisfies missing preconditions."""
        # Common precondition → command mapping
        suggestions = {
            "ports_discovered": "nmap_top_ports",
            "services_identified": "nmap_service_version",
            "web_server_found": "nmap_service_version",
            "smb_available": "nmap_service_version",
            "credentials_found": "hydra_ssh",
            "shell_obtained": "ssh_login",
            "root_obtained": "linpeas",
            "nfs_available": "showmount",
            "ftp_available": "ftp_anonymous",
            "ssh_available": "nmap_service_version",
            "ldap_available": "ldapsearch_base",
            "http_service": "nmap_service_version",
            "vuln_found": "nmap_vuln_scan",
        }
        for flag in missing:
            for key, cmd in suggestions.items():
                if key in flag:
                    return cmd
        return "nmap_top_ports"  # Safe default

    def _find_available_high_value(
        self,
        state: Dict[str, Any],
        discovery_board: Dict[str, Any],
    ) -> Optional[str]:
        """Find high-value exploits whose preconditions are met.
        
        Phase 9.4: Also queries KnowledgeRetriever by discovered ports
        to find exploitation paths from the knowledge base.
        """
        try:
            from core.commands.command_registry import COMMAND_REGISTRY
        except ImportError:
            return None

        state_flags = state.get("flags_set", set())
        if isinstance(state_flags, list):
            state_flags = set(state_flags)
        ports = discovery_board.get("ports", set())
        if isinstance(ports, list):
            ports = set(ports)

        # Check each high-value exploit
        for hvt_name in self._HIGH_VALUE_EXPLOITS:
            tmpl = COMMAND_REGISTRY.get(hvt_name)
            if tmpl is None:
                continue
            if tmpl.name in self._episode_templates:
                continue  # Already tried
            # Check preconditions
            if tmpl.preconditions and not tmpl.preconditions.issubset(state_flags):
                continue
            return hvt_name

        # Phase 9.4: KR-enriched port-based exploit suggestion
        if self._knowledge_retriever is not None and ports:
            try:
                for port in list(ports)[:5]:
                    port_num = int(port) if isinstance(port, str) else port
                    suggestions = self._knowledge_retriever.suggest_next(
                        port=port_num, phase="EXPLOITATION",
                    )
                    if suggestions:
                        for sugg in suggestions[:2]:
                            tmpl_name = sugg.get("template_name", "")
                            if tmpl_name and tmpl_name not in self._episode_templates:
                                tmpl = COMMAND_REGISTRY.get(tmpl_name)
                                if tmpl and (not tmpl.preconditions or tmpl.preconditions.issubset(state_flags)):
                                    return tmpl_name
            except Exception:
                pass

        return None

    def _compute_verdict(
        self,
        rules: List[TacticalRule],
    ) -> Tuple[TacticalVerdict, float]:
        """Compute verdict from rule results."""
        if not rules:
            return TacticalVerdict.APPROVE, 0.8

        critical = [r for r in rules if not r.passed and r.severity >= 0.8]
        warnings = [r for r in rules if not r.passed and 0.3 <= r.severity < 0.8]
        info = [r for r in rules if not r.passed and r.severity < 0.3]

        if critical:
            # Critical failure → block
            return TacticalVerdict.BLOCK, 0.9

        if len(warnings) >= 2:
            # Multiple warnings → redirect (or escalate if ambiguous)
            avg_severity = sum(w.severity for w in warnings) / len(warnings)
            if avg_severity > 0.6:
                return TacticalVerdict.REDIRECT, 0.7
            return TacticalVerdict.ESCALATE, 0.5

        if len(warnings) == 1:
            w = warnings[0]
            if w.severity >= 0.6:
                return TacticalVerdict.REDIRECT, 0.6
            # Ambiguous → escalate if LLM available
            return TacticalVerdict.ESCALATE, 0.5

        if info:
            # Only info-level issues → approve with lower confidence
            return TacticalVerdict.APPROVE, 0.7

        # All rules passed → full approval
        return TacticalVerdict.APPROVE, 0.9

    def _build_reasoning(
        self,
        rules: List[TacticalRule],
        verdict: TacticalVerdict,
    ) -> str:
        """Build human-readable reasoning string."""
        parts = [f"Verdict: {verdict.name}"]
        failing = [r for r in rules if not r.passed]
        if failing:
            for r in failing:
                parts.append(f"  [{r.category.value}] {r.reason}")
                if r.suggestion:
                    parts.append(f"    → Suggest: {r.suggestion}")
        else:
            parts.append("  All tactical checks passed.")
        return " | ".join(parts)

    def _build_alternative_command(
        self,
        template_name: str,
        state: Dict[str, Any],
        discovery_board: Dict[str, Any],
    ) -> Optional[str]:
        """Build an executable command from a template name."""
        try:
            from core.commands.command_registry import COMMAND_REGISTRY
        except ImportError:
            return None

        tmpl = COMMAND_REGISTRY.get(template_name)
        if tmpl is None:
            return None

        # Fill required params with defaults
        target = state.get("target", "10.0.0.1")
        cmd = tmpl.template
        cmd = cmd.replace("{target}", target)
        cmd = cmd.replace("{ip}", target)

        # Fill optional params with defaults
        for pname, pval in tmpl.optional_params.items():
            cmd = cmd.replace(f"{{{pname}}}", pval)

        return cmd

    def _extract_template_name(self, command: str) -> str:
        """Extract likely template name from a command string."""
        # First word or known tool name
        parts = command.strip().split()
        if not parts:
            return "unknown"
        tool = parts[0].split("/")[-1]  # Handle full paths
        return tool

    # ─── LLM Escalation ─────────────────────────────────────────────────────

    def _llm_evaluate(
        self,
        command: str,
        template: Optional[CommandTemplate],
        state: Dict[str, Any],
        agent_role: str,
        discovery_board: Dict[str, Any],
        current_phase: str,
        rules: List[TacticalRule],
    ) -> Optional[Tuple[TacticalVerdict, float, bool]]:
        """
        Escalate ambiguous assessment to LLM.
        
        Returns (verdict, confidence, llm_consulted) or None if call fails.
        """
        if self._gpt_manager is None:
            return None

        # Build compact context
        failing_rules = [
            f"[{r.category.value}] {r.reason}" for r in rules if not r.passed
        ]
        passing_rules = [
            r.category.value for r in rules if r.passed
        ]

        ports = list(discovery_board.get("ports", set()))[:10]
        services = list(discovery_board.get("services", set()))[:5]
        creds = bool(discovery_board.get("credentials", set()))
        shells = bool(discovery_board.get("shells", set()))

        prompt = (
            f"TACTICAL ASSESSMENT — evaluate this pentesting decision:\n"
            f"Command: {command}\n"
            f"Phase: {current_phase}\n"
            f"Agent: {agent_role}\n"
            f"Ports found: {ports}\n"
            f"Services: {services}\n"
            f"Has credentials: {creds}\n"
            f"Has shell: {shells}\n"
            f"Rules PASSING: {', '.join(passing_rules)}\n"
            f"Rules FAILING: {'; '.join(failing_rules)}\n\n"
            f"Should this command proceed? Reply with ONE word: APPROVE, REDIRECT, or BLOCK.\n"
            f"Then briefly explain why (1 sentence)."
        )

        try:
            response = self._gpt_manager.gpt_request(
                prompt,
                task_type="classification",
                agent_id=f"tactical_cortex_{agent_role}",
                max_tokens=60,
            )
            if not response:
                return None

            resp = response.strip().upper()
            if resp.startswith("APPROVE"):
                return TacticalVerdict.APPROVE, 0.75, True
            elif resp.startswith("REDIRECT"):
                return TacticalVerdict.REDIRECT, 0.70, True
            elif resp.startswith("BLOCK"):
                return TacticalVerdict.BLOCK, 0.80, True
            else:
                # Parse didn't match → treat as approve with low confidence
                return TacticalVerdict.APPROVE, 0.55, True

        except Exception as e:
            logger.debug(f"[TACTICAL] LLM escalation failed: {e}")
            return None

    # ─── Stats ───────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Get tactical cortex statistics for this episode."""
        return {
            "llm_calls": self._llm_calls_this_episode,
            "llm_budget_remaining": self._max_llm_calls - self._llm_calls_this_episode,
            "steps_assessed": len(self._episode_history),
            "discoveries": self._discovery_count,
            "unique_commands": len(set(self._episode_templates)),
            "total_commands": len(self._episode_templates),
        }
