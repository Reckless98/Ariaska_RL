"""
Smart Mentor - Structured GPT prompting for intelligent command generation.

This module provides rich context to the LLM for generating high-quality
pentesting commands based on current attack state, phase, and history.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum

from ..commands.command_registry import (
    AttackPhase,
    CommandTemplate,
    CommandChoice,
    COMMAND_REGISTRY,
    get_valid_commands_for_state,
    get_phase_from_state,
    render_command,
)
from ..commands.learned_commands import (
    LearnedCommandStore,
    get_learned_store
)


@dataclass
class AttackContext:
    """
    Rich context about the current attack state for LLM prompting.
    
    Attributes:
        target: Target IP or hostname
        current_phase: Current phase in attack chain
        state_flags: Dictionary of discovered facts
        command_history: Recent commands executed
        discoveries: Important findings (open ports, users, etc.)
        failed_attempts: Commands that didn't work
        time_budget: Remaining time or tokens
        difficulty: Target difficulty (easy, medium, hard, insane)
        platform: Target platform (linux, windows, unknown)
        services_found: Discovered services
    """
    target: str
    current_phase: AttackPhase = AttackPhase.RECON
    state_flags: Dict[str, Any] = field(default_factory=dict)
    command_history: List[str] = field(default_factory=list)
    discoveries: Dict[str, Any] = field(default_factory=dict)
    failed_attempts: List[str] = field(default_factory=list)
    time_budget: Optional[int] = None
    difficulty: str = "unknown"
    platform: str = "unknown"
    services_found: List[str] = field(default_factory=list)
    _r66_scan_hints: List[str] = field(default_factory=list)
    
    def to_narrative(self) -> str:
        """Convert context to human-readable narrative for LLM."""
        lines = []
        
        # Target info
        lines.append(f"TARGET: {self.target}")
        lines.append(f"PHASE: {self.current_phase.name}")
        lines.append(f"PLATFORM: {self.platform}")
        if self.difficulty != "unknown":
            lines.append(f"DIFFICULTY: {self.difficulty}")
        
        # Services
        if self.services_found:
            lines.append(f"\nDISCOVERED SERVICES:")
            for svc in self.services_found:
                lines.append(f"  • {svc}")
        
        # Key discoveries
        if self.discoveries:
            lines.append(f"\nKEY DISCOVERIES:")
            for key, value in self.discoveries.items():
                if isinstance(value, list):
                    lines.append(f"  • {key}: {', '.join(str(v) for v in value[:5])}")
                else:
                    lines.append(f"  • {key}: {value}")
        
        # Recent commands (last 5)
        if self.command_history:
            lines.append(f"\nRECENT COMMANDS (last 5):")
            for cmd in self.command_history[-5:]:
                lines.append(f"  > {cmd[:80]}...")
        
        # Failed attempts (to avoid)
        if self.failed_attempts:
            lines.append(f"\nFAILED ATTEMPTS (avoid these):")
            for cmd in self.failed_attempts[-3:]:
                lines.append(f"  ✗ {cmd[:60]}...")
        
        # Current state flags summary
        active_flags = [k for k, v in self.state_flags.items() if v]
        if active_flags:
            lines.append(f"\nCURRENT STATE:")
            for flag in active_flags[:10]:
                lines.append(f"  ✓ {flag}")
        
        return "\n".join(lines)
    
    def add_discovery(self, key: str, value: Any) -> None:
        """Add a discovery to the context."""
        if key in self.discoveries:
            existing = self.discoveries[key]
            if isinstance(existing, (list, set)):
                if isinstance(value, list):
                    if isinstance(existing, set):
                        existing.update(value)
                    else:
                        existing.extend(value)
                else:
                    if isinstance(existing, set):
                        existing.add(value)
                    else:
                        existing.append(value)
            else:
                self.discoveries[key] = [existing, value]
        else:
            self.discoveries[key] = [value]
    
    def set_state_flag(self, flag: str, value: bool = True) -> None:
        """Set a state flag."""
        self.state_flags[flag] = value
        
        # Phase 21: Track highest reached phase for sequential enforcement.
        # Pass it via state_flags so get_phase_from_state can clamp advances.
        self.state_flags["_highest_reached_phase"] = self.current_phase
        
        # Auto-detect phase advancement (sequential — max +1 phase per call)
        self.current_phase = get_phase_from_state(self.state_flags)
        
        # Update tracker after advancement
        self.state_flags["_highest_reached_phase"] = self.current_phase
    
    def add_service(self, service: str, port: Optional[int] = None) -> None:
        """Add a discovered service."""
        svc_str = f"{service}:{port}" if port else service
        if svc_str not in self.services_found:
            self.services_found.append(svc_str)
        
        # Set appropriate state flag
        service_lower = service.lower()
        if "http" in service_lower or "web" in service_lower:
            self.set_state_flag("http_service_found")
        elif "ssh" in service_lower:
            self.set_state_flag("ssh_service_found")
        elif "smb" in service_lower or "445" in str(port):
            self.set_state_flag("smb_service_found")
        elif "ftp" in service_lower:
            self.set_state_flag("ftp_service_found")
        elif "ldap" in service_lower:
            self.set_state_flag("ldap_service_found")
        elif "dns" in service_lower:
            self.set_state_flag("dns_service_found")
        elif "snmp" in service_lower:
            self.set_state_flag("snmp_service_found")
        elif "winrm" in service_lower or "5985" in str(port):
            self.set_state_flag("winrm_service_found")
        elif "kerberos" in service_lower or "88" in str(port):
            self.set_state_flag("kerberos_service_found")


@dataclass
class MentorResponse:
    """
    Structured response from the smart mentor.
    
    Phase 6.3: Enhanced with intent, candidate_actions, expected_observation,
    and risk fields for deeper reasoning trace.
    
    Attributes:
        command: The actual command to execute
        template_name: Name of the CommandTemplate used
        params: Parameters used
        reasoning: Why this command was chosen
        confidence: Confidence in this choice (0-1)
        next_phase_hint: Suggested next phase
        alternative_commands: Backup commands if this fails
        phase: Current attack phase
        tokens_used: Tokens consumed by this mentor call
        intent: Strategic intent (e.g. "discover services", "exploit vsftpd backdoor")
        expected_observation: What we expect to see if command succeeds
        risk: Risk assessment ("low", "medium", "high")
        candidate_actions: Ranked list of candidate actions considered
    """
    command: str
    template_name: str
    params: Dict[str, str]
    reasoning: str
    confidence: float
    next_phase_hint: Optional[str] = None
    alternative_commands: List[str] = field(default_factory=list)
    phase: AttackPhase = AttackPhase.RECON
    tokens_used: int = 0
    
    # Phase 6.3: Structured reasoning fields
    intent: str = ""  # Strategic intent of this action
    expected_observation: str = ""  # What we expect to see on success
    risk: str = "low"  # "low", "medium", "high"
    candidate_actions: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def is_valid(self) -> bool:
        """Check if this is a valid response."""
        return bool(self.command and self.template_name)


class SmartMentor:
    """
    Intelligent mentor that generates pentesting commands using LLM.
    
    Provides structured prompts with rich context and validates
    output against the command registry.
    """
    
    def __init__(
        self,
        llm_client: Any,
        learned_store: Optional[LearnedCommandStore] = None,
        model: str = "local-llm",  # Phase 12.1: full reasoning model for all mentor calls
        temperature: float = 0.7,
        max_retries: int = 2,
        target_profile: str = "generic",
    ):
        """
        Initialize the smart mentor.
        
        Args:
            llm_client: LLM client with chat completion capability
            learned_store: Store for learned commands
            model: Model to use
            temperature: Sampling temperature
            max_retries: Max retries for failed parses
            target_profile: Target environment profile (generic, metasploitable2, metasploitable3, htb)
        """
        self.llm_client = llm_client
        self.learned_store = learned_store or get_learned_store()
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.target_profile = target_profile
        
        # Phase 9.2: ReflectiveCortex insight injection point
        self._reflective_insights: List[str] = []
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt with dynamic knowledge injection from knowledge_packs."""
        # Dynamic knowledge injection from knowledge_packs module
        # Phase 42: Gate MS2/MS3 knowledge behind target_profile to prevent
        # credential hallucination on non-Metasploitable targets (e.g., HTB)
        _is_ms2 = self.target_profile in ("metasploitable2", "ms2")
        _is_ms3 = self.target_profile in ("metasploitable3", "ms3")
        try:
            from core.knowledge.knowledge_packs import (
                get_mentor_knowledge_text, PHASE_REASONING,
                HTB_DECISION_RULES, HTB_CVES,
            )
            ms2_knowledge = get_mentor_knowledge_text("metasploitable2") if _is_ms2 else ""
            ms3_knowledge = get_mentor_knowledge_text("metasploitable3") if _is_ms3 else ""
            htb_knowledge = get_mentor_knowledge_text("htb")
            # Build phase reasoning text
            phase_reasoning_text = "\n=== PHASE REASONING (WHY and WHEN) ===\n"
            for phase, reasoning in PHASE_REASONING.items():
                phase_reasoning_text += f"\n{phase.upper()}:\n"
                for key, val in reasoning.items():
                    if isinstance(val, str):
                        phase_reasoning_text += f"  {key}: {val[:150]}\n"
            # Build decision rules text for rapid expert-like reasoning
            decision_rules_text = "\n=== EXPERT DECISION RULES (IF → THEN) ===\n"
            decision_rules_text += "Apply these rules like a human pentester — check conditions, execute matching actions:\n\n"
            for rule in HTB_DECISION_RULES:
                decision_rules_text += f"[P{rule['priority']}] IF: {rule['if_condition']}\n"
                decision_rules_text += f"     THEN: {rule['then_action']}\n"
                decision_rules_text += f"     WHY: {rule['reasoning']}\n\n"
            # Build CVE quick-reference
            cve_text = "\n=== CVE QUICK REFERENCE ===\n"
            for cve in HTB_CVES[:10]:  # Top 10 most relevant
                cve_text += f"  {cve.cve_id}: {cve.service} — {cve.description[:80]}\n"
                if cve.module:
                    cve_text += f"    Exploit: {cve.module}\n"
        except ImportError:
            ms2_knowledge = ""
            ms3_knowledge = ""
            htb_knowledge = ""
            phase_reasoning_text = ""
            decision_rules_text = ""
            cve_text = ""

        # Phase 9: Web exploitation reference from massive knowledge seed
        try:
            from core.knowledge.massive_knowledge_seed import get_web_attack_reference
            _raw = get_web_attack_reference()
            web_exploit_reference = _raw[:2000] if len(_raw) > 2000 else _raw
        except Exception:
            web_exploit_reference = ""
        
        # Phase 9: Venice aha context — inject cross-episode insights
        venice_aha_context = ""
        try:
            from core.memory.unified_cognitive_bus import CognitiveBus
            bus = CognitiveBus()  # singleton via __new__
            mentor_ctx = bus.get_mentor_context(max_items=5)
            if mentor_ctx:
                venice_aha_context = "\n=== COGNITIVE BUS — CROSS-EPISODE INSIGHTS ===\n"
                venice_aha_context += mentor_ctx
        except Exception:
            pass

        # Phase 9.2: ReflectiveCortex insights — inject meta-learning insights
        reflective_insights_text = ""
        try:
            from core.llm.reflective_cortex import ReflectiveCortex
            # Check if cortex has active insights (injected by SmartOrchestrator)
            if hasattr(self, '_reflective_insights') and self._reflective_insights:
                reflective_insights_text = "\n=== REFLECTIVE CORTEX — META-LEARNING INSIGHTS ===\n"
                for insight in self._reflective_insights[:3]:
                    reflective_insights_text += f"• {insight}\n"
        except Exception:
            pass

        # Phase 9.2: Knowledge Graph context — exploit paths for current target
        kg_exploit_context = ""
        try:
            from core.knowledge.kg_manager import KnowledgeGraph
            kg = KnowledgeGraph.get_instance()
            if kg:
                attack_surface = kg.get_attack_surface(host_id="target")  # type: ignore[arg-type]
                if attack_surface:
                    kg_exploit_context = "\n=== KNOWLEDGE GRAPH — ATTACK SURFACE ===\n"
                    items = list(attack_surface.items())[:8]
                    for node_id, data in items:
                        label = data.get("label", node_id) if isinstance(data, dict) else str(data)
                        ntype = data.get("node_type", "") if isinstance(data, dict) else ""
                        kg_exploit_context += f"  • [{ntype}] {label}\n"
        except Exception:
            pass

        return f"""You are an elite penetration tester, red team operator, and AI MENTOR teaching autonomous agents to think like expert hackers.

=== YOUR ROLE: TEACH AUTONOMOUS REASONING ===
You are not just selecting commands — you are TEACHING the AI agent to:
1. REASON about attack surfaces like an expert ("I see port X running service Y, which means...")
2. CHAIN discoveries into attack paths ("Credential Z from MySQL + SSH service = lateral movement")
3. ADAPT strategy when stuck ("Three nmap scans failed → try different approach: UDP, specific scripts")
4. BUILD MENTAL MODELS of the target ("This is a Linux box, likely Ubuntu, with Java stack → check deserialization")
5. DEVELOP INTUITION about what to try next without needing a mentor

Your reasoning in the "reasoning" field is CRITICAL — the agent memorizes it. Be specific, tactical, educational.
Explain WHY this command, not just what command. The agent should eventually make this choice independently.

{ms2_knowledge}

=== REASONING QUALITY REQUIREMENTS ===
Your "reasoning" field TEACHES the agent. Write it like an expert explaining to an apprentice:
  BAD:  "Use nmap to scan"
  GOOD: "We haven't enumerated services yet. nmap -sV reveals version info which maps to known CVEs.
         Version strings → searchsploit/exploit-db → targeted exploitation. Service versions → exploit selection."
  
  BAD:  "Try the exploit"
  GOOD: "Port 80 shows Apache with a login form. Version 2.4.49 is vulnerable to path traversal
         (CVE-2021-41773). We can try /cgi-bin/.%2e/.%2e/etc/passwd to confirm and then
         escalate to RCE via mod_cgi. This is faster than bruteforcing credentials."

=== ADAPTIVE STRATEGY GUIDANCE ===
When the agent is STUCK (repeating commands, low discovery rate):
  • Suggest a COMPLETELY different attack vector, not variations of the same approach
  • Explain WHY the current approach is failing and what that tells us about the target
  • Propose a hypothesis about the target and a command to test it

When the agent is SUCCEEDING (making discoveries):
  • Guide toward DEEPER exploitation, not more recon
  • Chain the current discovery into the next logical step
  • Explain the attack graph: "Discovery X unlocks paths A, B, C"

CRITICAL RULES:
1. ALWAYS select from the provided command list - never invent commands
2. NEVER repeat a command from the "COMMANDS ALREADY TRIED" section!
3. If you see a command was tried multiple times, it's STUCK - pick something completely different
4. Consider the current phase and what discoveries have been made
5. Progress through phases: Recon → Enumeration → Exploitation → PrivEsc → Lateral → Post-Ex
6. Be creative but realistic - use the right tool for the situation
7. Variety is key - try different approaches if current ones aren't working
8. NEVER assume default credentials unless discovered via actual enumeration — do NOT hallucinate creds

OUTPUT FORMAT (JSON only):
{{
    "intent": "strategic goal of this action",
    "selected_command": "template_name from the list",
    "parameters": {{"param1": "value1", "param2": "value2"}},
    "reasoning": "Detailed WHY explanation — teach the agent to think like you",
    "expected_observation": "what we expect if this succeeds and how to interpret it",
    "risk": "low|medium|high",
    "confidence": 0.8,
    "next_phase_hint": "what to try if this works AND what to try if it fails",
    "candidate_actions": [
        {{"command": "alternative_template_name", "why": "detailed reason with attack chain logic"}}
    ]
}}

=== HACK THE BOX COMMON PATTERNS ===
80% of HTB boxes start with web. Master these patterns:

WEB → SHELL:
  • SQL Injection: sqlmap --os-shell, UNION SELECT INTO OUTFILE
  • SSTI: {{7*7}} in Jinja2/Twig → {{config.__class__.__init__.__globals__['os'].popen('id').read()}}
  • File Upload: .php/.phtml/.php5 webshell, bypass extension filters
  • LFI → RCE: php://filter, /proc/self/environ, log poisoning
  • Command Injection: ; id, | id, $(id), `id`

LINUX PRIVESC (after user shell):
  • SUID: find / -perm -4000 -type f 2>/dev/null → check GTFOBins
  • sudo -l: Look for specific commands with shell escape
  • Kernel: uname -r → linux-exploit-suggester
  • Cron: cat /etc/crontab, pspy64 for hidden crons
  • Capabilities: getcap -r /usr /bin /sbin 2>/dev/null → cap_setuid = root
  • Docker group: docker run -v /:/host -it alpine chroot /host sh

WINDOWS PRIVESC:
  • Token: whoami /priv → SeImpersonatePrivilege → PrintSpoofer/JuicyPotato
  • Kerberoasting: GetUserSPNs.py → hashcat -m 13100
  • BloodHound: bloodhound-python -c all → find shortest path to DA

=== FLAGS ===
USER: cat /home/*/user.txt | ROOT: cat /root/root.txt | Formats: FLAG{{...}}, HTB{{...}}, 32-hex.
After ANY shell, read the flag file FIRST.

{ms3_knowledge}

{htb_knowledge}

{decision_rules_text}

{cve_text}

{phase_reasoning_text}

{web_exploit_reference}

{venice_aha_context}

{reflective_insights_text}

{kg_exploit_context}
"""
    
    def _build_user_prompt(
        self,
        context: AttackContext,
        valid_commands: List[CommandTemplate]
    ) -> str:
        """Build the user prompt with context and available commands."""
        lines = []
        
        # Attack context narrative
        lines.append("=== CURRENT ATTACK STATE ===")
        lines.append(context.to_narrative())
        lines.append("")
        
        # === CRITICAL: Show what was ALREADY TRIED to prevent loops ===
        if context.command_history:
            lines.append("=== COMMANDS ALREADY TRIED (DO NOT REPEAT) ===")
            # Show last 15 commands with count of how many times each was used
            recent_commands = context.command_history[-15:]
            from collections import Counter
            cmd_counts = Counter(recent_commands)
            for cmd, count in cmd_counts.most_common(10):
                # Truncate long commands
                cmd_display = cmd[:60] + "..." if len(cmd) > 60 else cmd
                if count > 1:
                    lines.append(f"  ❌ {cmd_display} (tried {count}x - AVOID!)")
                else:
                    lines.append(f"  ⚠️ {cmd_display} (already tried)")
            lines.append("")
            lines.append("⚠️ IMPORTANT: Choose a DIFFERENT command from the list below!")
            lines.append("")
        
        # Available commands
        lines.append("=== AVAILABLE COMMANDS ===")
        lines.append(f"(Phase: {context.current_phase.name})")
        lines.append("")
        
        for cmd in valid_commands[:25]:  # Limit to avoid token overflow
            params = ", ".join(cmd.required_params) if cmd.required_params else "none"
            optional = ", ".join(f"{k}={v}" for k, v in list(cmd.optional_params.items())[:3])
            
            lines.append(f"• {cmd.name}")
            lines.append(f"  Required params: {params}")
            if optional:
                lines.append(f"  Optional: {optional}")
            lines.append(f"  Description: {cmd.description}")
            
            # Include WHY/WHEN context for smarter decisions
            if hasattr(cmd, 'why') and cmd.why:
                lines.append(f"  WHY: {cmd.why}")
            if hasattr(cmd, 'when') and cmd.when:
                lines.append(f"  WHEN: {cmd.when}")
            if hasattr(cmd, 'not_when') and cmd.not_when:
                lines.append(f"  AVOID: {cmd.not_when}")
            if hasattr(cmd, 'follows_after') and cmd.follows_after:
                lines.append(f"  FOLLOWS: {', '.join(cmd.follows_after[:3])}")
            if hasattr(cmd, 'enables') and cmd.enables:
                lines.append(f"  ENABLES: {', '.join(cmd.enables[:3])}")
            
            lines.append("")
        
        # Learned command suggestions (filtered by valid_commands if provided)
        learned_suggestions = self.learned_store.get_commands_for_prompt(
            phase=context.current_phase,
            preconditions=set(k for k, v in context.state_flags.items() if v),
            limit=5  # Get more, then filter
        )
        
        # Filter learned suggestions to only include commands in valid_commands
        if learned_suggestions and valid_commands:
            valid_names = {cmd.name.lower() for cmd in valid_commands}
            filtered_suggestions = []
            for suggestion in learned_suggestions:
                # Extract command name from suggestion string (format: "• command_name: ...")
                if "•" in suggestion:
                    cmd_name = suggestion.split("•")[1].split(":")[0].strip().lower()
                    if cmd_name in valid_names:
                        filtered_suggestions.append(suggestion)
            learned_suggestions = filtered_suggestions[:3]
        
        if learned_suggestions:
            lines.append("=== PROVEN SUCCESSFUL COMMANDS ===")
            for suggestion in learned_suggestions:
                lines.append(suggestion)
            lines.append("")
        
        # RAG: Query knowledge base for relevant context
        try:
            from core.knowledge.knowledge_query import build_rag_context
            rag_context = build_rag_context(
                phase=context.current_phase.name if context.current_phase else "RECON",
                discoveries=context.discoveries,
                target=context.target,
                max_snippets=3,
            )
            if rag_context:
                lines.append(rag_context)
        except Exception:
            pass  # RAG is optional — graceful degradation

        # Phase 9.1: MEGA KNOWLEDGE BASE injection — 14.6 MB across 6 repos
        try:
            from data.knowledge_retriever import get_knowledge_retriever
            kr = get_knowledge_retriever()
            # Build state dict for knowledge retriever
            kb_state = {
                "phase": context.current_phase.name if context.current_phase else "RECON",
                "discovered_ports": list(context.discoveries.get("ports", set())),
                "discovered_services": list(context.discoveries.get("services", set())),
                "target_ip": context.target or "",
            }
            kb_context = kr.for_mentor_prompt(kb_state, max_tokens=1500)
            if kb_context and len(kb_context) > 50:
                lines.append("")
                lines.append(kb_context)
        except Exception:
            pass  # Knowledge base is optional — graceful degradation
        
        # Instruction
        lines.append("=== YOUR TASK ===")
        lines.append("Select the BEST next command from the AVAILABLE COMMANDS list above.")
        lines.append("IMPORTANT: Only use commands from the list - do not invent or use other commands.")
        lines.append("Fill in all required parameters based on the target and context.")
        lines.append("Respond with ONLY valid JSON, no markdown or explanation outside JSON.")
        
        return "\n".join(lines)
    
    def _parse_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse LLM response into structured format."""
        # Try to extract JSON from response
        text = response_text.strip()
        
        # Remove markdown code blocks if present
        if text.startswith("```"):
            # Find the end
            lines = text.split("\n")
            json_lines = []
            in_json = False
            for line in lines:
                if line.startswith("```") and not in_json:
                    in_json = True
                    continue
                elif line.startswith("```") and in_json:
                    break
                elif in_json:
                    json_lines.append(line)
            text = "\n".join(json_lines)
        
        # Try to find JSON object
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if json_match:
            text = json_match.group()
        
        try:
            data = json.loads(text)
            return data
        except json.JSONDecodeError:
            return None
    
    def _validate_and_build_response(
        self,
        parsed: Dict[str, Any],
        context: AttackContext,
        valid_commands: List[CommandTemplate]
    ) -> Optional[MentorResponse]:
        """Validate parsed response and build MentorResponse."""
        template_name = parsed.get("selected_command", "")
        params = parsed.get("parameters", {})
        reasoning = parsed.get("reasoning", "")
        confidence = float(parsed.get("confidence", 0.5))
        next_hint = parsed.get("next_phase_hint", "")
        
        # Find the template
        template = COMMAND_REGISTRY.get(template_name)
        if not template:
            # Try fuzzy match
            for name, cmd in COMMAND_REGISTRY.items():
                if template_name.lower() in name.lower() or name.lower() in template_name.lower():
                    template = cmd
                    template_name = name
                    break
        
        if not template:
            return None
        
        # Ensure required params are filled
        for req_param in template.required_params:
            if req_param not in params or not params[req_param]:
                # Try to fill from context
                if req_param == "target":
                    params[req_param] = context.target
                elif req_param == "url" and context.target:
                    # Construct URL from target
                    port = 80
                    for svc in context.services_found:
                        if "http" in svc.lower():
                            parts = svc.split(":")
                            if len(parts) > 1:
                                try:
                                    port = int(parts[1])
                                except ValueError:
                                    pass
                    proto = "https" if port == 443 else "http"
                    params[req_param] = f"{proto}://{context.target}:{port}"
                elif req_param == "ports" and "open_ports" in context.discoveries:
                    params[req_param] = ",".join(str(p) for p in context.discoveries["open_ports"][:10])
                else:
                    # Still missing required param
                    return None
        
        # Render the command
        try:
            command = render_command(template, params)
        except ValueError as e:
            return None
        
        return MentorResponse(
            command=command,
            template_name=template_name,
            params=params,
            reasoning=reasoning,
            confidence=confidence,
            next_phase_hint=next_hint,
            phase=template.phase,
            # Phase 6.3: Structured reasoning fields
            intent=parsed.get("intent", ""),
            expected_observation=parsed.get("expected_observation", ""),
            risk=parsed.get("risk", "low"),
            candidate_actions=parsed.get("candidate_actions", []),
        )
    
    async def get_command_async(
        self,
        context: AttackContext,
        filtered_commands: Optional[List[CommandTemplate]] = None
    ) -> MentorResponse:
        """
        Get the next command asynchronously.
        
        Args:
            context: Current attack context
            filtered_commands: Pre-filtered commands (by agent role). If None, uses global registry.
            
        Returns:
            MentorResponse with command and reasoning
        """
        # Use pre-filtered commands if provided (role-aware), else fallback to global
        if filtered_commands:
            valid_commands = filtered_commands
        else:
            # Get valid commands for current state
            valid_commands = get_valid_commands_for_state(
                context.state_flags,
                context.current_phase
            )
            
            # If no valid commands for current phase, try all phases
            if not valid_commands:
                valid_commands = get_valid_commands_for_state(context.state_flags)
            
            # If still none, use recon commands (always valid)
            if not valid_commands:
                valid_commands = [
                    cmd for cmd in COMMAND_REGISTRY.values()
                    if cmd.phase == AttackPhase.RECON and not cmd.preconditions
                ]
        
        # Build prompts
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(context, valid_commands)
        
        # Track tokens for this call (accumulate across retries)
        tokens_used = 0

        # Call local LLM via standard chat completions
        for attempt in range(self.max_retries + 1):
            try:
                request_params = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "max_tokens": 500,
                    "temperature": self.temperature,
                }

                response = await self.llm_client.chat.completions.create(**request_params)

                # Capture token usage from response (accumulate!)
                if hasattr(response, 'usage') and response.usage:
                    tokens_used += response.usage.total_tokens

                # Extract text from chat completion
                response_text = response.choices[0].message.content  # type: ignore[union-attr]
                
                # Parse response
                parsed = self._parse_response(response_text)
                if not parsed:
                    continue
                
                # Validate and build response
                mentor_response = self._validate_and_build_response(
                    parsed, context, valid_commands
                )
                
                if mentor_response:
                    mentor_response.tokens_used = tokens_used
                    return mentor_response
                
            except Exception as e:
                if attempt == self.max_retries:
                    # Return a safe default (with token count)
                    fallback = self._get_fallback_command(context, valid_commands)
                    fallback.tokens_used = tokens_used
                    return fallback
        
        # If all retries exhausted, return fallback with accumulated tokens
        fallback = self._get_fallback_command(context, valid_commands)
        fallback.tokens_used = tokens_used
        return fallback
    
    def get_command(
        self,
        context: AttackContext,
        filtered_commands: Optional[List[CommandTemplate]] = None
    ) -> MentorResponse:
        """
        Get the next command synchronously.
        
        Args:
            context: Current attack context
            filtered_commands: Pre-filtered commands (by agent role). If None, uses global registry.
            
        Returns:
            MentorResponse with command and reasoning
        """
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, create a new task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        self.get_command_async(context, filtered_commands)
                    )
                    return future.result()
            else:
                return loop.run_until_complete(self.get_command_async(context, filtered_commands))
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(self.get_command_async(context, filtered_commands))
    
    def _get_fallback_command(
        self,
        context: AttackContext,
        valid_commands: List[CommandTemplate]
    ) -> MentorResponse:
        """Get a safe fallback command when LLM fails - WITH ANTI-LOOP LOGIC."""
        import random
        
        # Get commands that were ALREADY TRIED (from context history)
        tried_commands = set()
        if context.command_history:
            # Extract base command names (first word) for matching
            for cmd in context.command_history[-20:]:  # Last 20 commands
                # Get the command prefix (e.g., "nmap", "gobuster", "nikto")
                parts = cmd.strip().split()
                if parts:
                    tried_commands.add(parts[0].lower())
        
        if not valid_commands:
            # Ultimate fallback pool - VARIETY of commands to avoid loops
            fallback_pool = [
                (f"nmap -sT --top-ports 100 {context.target}", "nmap_top_ports", "Fast TCP scan"),
                (f"nmap -sV {context.target}", "nmap_version", "Service version detection"),
                (f"masscan -p1-1000 --rate=500 {context.target}", "masscan_fast", "Fast port scan"),
                (f"nmap -sU --top-ports 20 {context.target}", "nmap_udp", "UDP scan"),
                (f"curl -s -I http://{context.target}", "curl_headers", "HTTP headers"),
                (f"dig {context.target} ANY +short", "dig_any", "DNS lookup"),
                (f"whois {context.target} | head -30", "whois", "Domain info"),
            ]
            
            # Filter out commands whose prefix was already tried
            untried = [(c, t, r) for c, t, r in fallback_pool 
                       if c.split()[0].lower() not in tried_commands]
            
            if untried:
                cmd, template_name, reason = random.choice(untried)
            else:
                # All tried - pick a random one anyway (variety within same tools)
                cmd, template_name, reason = random.choice(fallback_pool)
            
            return MentorResponse(
                command=cmd,
                template_name=template_name,
                params={"target": context.target},
                reasoning=f"🔄 Fallback: {reason}",
                confidence=0.3,
                phase=AttackPhase.RECON
            )
        
        # Filter out commands that were already tried (by prefix matching)
        untried_templates = [cmd for cmd in valid_commands 
                            if cmd.name.lower().replace('_', '-').split('_')[0] not in tried_commands
                            and cmd.name.lower().split('_')[0] not in tried_commands]
        
        if untried_templates:
            # Shuffle and pick from untried commands (variety)
            random.shuffle(untried_templates)
            # Weight by reward but not too heavily
            weights = [1.0 + cmd.typical_reward * 0.3 for cmd in untried_templates[:5]]
            template = random.choices(untried_templates[:5], weights=weights[:5], k=1)[0]
        else:
            # All templates tried - just pick randomly for variety
            random.shuffle(valid_commands)
            template = random.choice(valid_commands[:5]) if len(valid_commands) >= 5 else valid_commands[0]
        
        # Try to fill params
        params = {}
        for p in template.required_params:
            if p == "target":
                params[p] = context.target
            elif p == "url":
                params[p] = f"http://{context.target}"
            elif p == "ports" and "open_ports" in context.discoveries:
                params[p] = ",".join(str(p) for p in context.discoveries["open_ports"][:10])
            else:
                params[p] = "{" + p + "}"  # Placeholder
        
        # Add optional defaults
        params.update(template.optional_params)
        
        try:
            command = render_command(template, params)
        except ValueError:
            command = template.template.format_map(params)
        
        return MentorResponse(
            command=command,
            template_name=template.name,
            params=params,
            reasoning=f"Fallback: {template.description}",
            confidence=0.3,
            phase=template.phase
        )
    
    def record_result(
        self,
        response: MentorResponse,
        success: bool,
        reward: float,
        context: AttackContext
    ) -> None:
        """
        Record the result of a command for learning.
        
        Args:
            response: The MentorResponse that was executed
            success: Whether it succeeded
            reward: Reward received
            context: Attack context when executed
        """
        context_tags = {context.platform, context.difficulty}
        preconditions = set(k for k, v in context.state_flags.items() if v)
        
        if success:
            self.learned_store.record_success(
                template_name=response.template_name,
                params=response.params,
                reward=reward,
                context_tags=context_tags,
                preconditions_met=preconditions,
                phase=response.phase
            )
        else:
            self.learned_store.record_failure(
                template_name=response.template_name,
                params=response.params
            )


@dataclass
class DualMentorResponse:
    """
    Response from dual-mentor system with responses from both providers.
    
    Attributes:
        primary: Response from primary mentor (GPT)
        secondary: Response from secondary mentor (Venice)
        chosen: The chosen response to use
        strategy: Strategy used for selection
        provider_used: Which provider's response was used
    """
    primary: Optional[MentorResponse] = None
    secondary: Optional[MentorResponse] = None
    chosen: Optional[MentorResponse] = None
    strategy: str = "gpt_first"
    provider_used: str = "gpt"
    tokens_total: int = 0
    
    @property
    def is_valid(self) -> bool:
        """Check if we have a valid chosen response."""
        return self.chosen is not None and self.chosen.is_valid


class DualMentor:
    """
    Dual-mentor system that leverages both GPT and a secondary LLM for command generation.
    
    Phase 43: Secondary provider is local GPU LLM (replaces Venice AI).
    Falls back to Venice if local LLM is not available.
    
    Strategies:
    - gpt_first: Use GPT, fallback to secondary on error
    - local_first: Use local LLM, fallback to GPT on error (Phase 43)
    - venice_first: Alias for local_first (backward compat)
    - round_robin: Alternate between GPT and secondary
    - parallel: Query both, choose best by confidence
    - consensus: Query both, require agreement or escalate
    
    Having two mentors enables:
    - Higher mentor call frequency (double the budget)
    - Fallback resilience
    - Different perspectives on attack strategies
    - Local LLM: zero-cost secondary calls on GPU
    """
    
    def __init__(
        self,
        gpt_client: Any,
        venice_client: Optional[Any] = None,
        gpt_model: str = "local-llm",  # Phase 12.1: full reasoning model
        venice_model: str = "qwen3-coder-480b-a35b-instruct",
        learned_store: Optional[LearnedCommandStore] = None,
        strategy: str = "gpt_first",
        temperature: float = 0.7,
        max_retries: int = 2,
        *,
        secondary_client: Optional[Any] = None,
        secondary_model: Optional[str] = None,
    ):
        """
        Initialize the dual-mentor system.
        
        Args:
            gpt_client: OpenAI client for GPT
            venice_client: OpenAI-compatible client for secondary provider (legacy name)
            gpt_model: GPT model to use (Phase 12.1: default local-llm)
            venice_model: Secondary model to use (legacy name, default: qwen3-coder)
            learned_store: Store for learned commands
            strategy: Mentor selection strategy
            temperature: Sampling temperature
            max_retries: Max retries for failed parses
            secondary_client: Phase 43: Explicit secondary client (overrides venice_client)
            secondary_model: Phase 43: Explicit secondary model name (overrides venice_model)
        """
        self.learned_store = learned_store or get_learned_store()
        # Phase 43: Normalize strategy — venice_first → local_first
        if strategy == "venice_first":
            strategy = "local_first"
        self.strategy = strategy
        self._call_count = 0
        
        # Phase 43: secondary_client takes priority over venice_client
        _sec_client = secondary_client if secondary_client is not None else venice_client
        _sec_model = secondary_model if secondary_model is not None else venice_model
        
        # Initialize GPT mentor
        self.gpt_mentor = SmartMentor(
            llm_client=gpt_client,
            learned_store=self.learned_store,
            model=gpt_model,
            temperature=temperature,
            max_retries=max_retries
        ) if gpt_client else None
        
        # Initialize secondary mentor (local LLM or Venice)
        self.secondary_mentor = SmartMentor(
            llm_client=_sec_client,
            learned_store=self.learned_store,
            model=_sec_model,
            temperature=temperature + 0.1,  # Secondary slightly more creative
            max_retries=max_retries
        ) if _sec_client else None
        
        # Backward compat: expose as venice_mentor
        self.venice_mentor = self.secondary_mentor
        
        self._has_both = self.gpt_mentor is not None and self.secondary_mentor is not None
        
        import logging
        self.logger = logging.getLogger("ariaska.dual_mentor")
        self.logger.info(
            f"DualMentor initialized | Strategy: {strategy} | "
            f"GPT: {'✓' if self.gpt_mentor else '✗'} | "
            f"Secondary: {'✓' if self.secondary_mentor else '✗'}"
        )
    
    def has_dual_capability(self) -> bool:
        """Check if both mentors are available."""
        return self._has_both
    
    def _get_provider_for_call(self) -> str:
        """Determine which provider to use for this call."""
        if not self._has_both:
            return "gpt" if self.gpt_mentor else "secondary"
        
        if self.strategy in ("local_first", "venice_first"):
            return "secondary"
        elif self.strategy == "round_robin":
            self._call_count += 1
            return "secondary" if self._call_count % 2 == 0 else "gpt"
        elif self.strategy in ("parallel", "consensus"):
            return "both"
        else:  # gpt_first (default)
            return "gpt"
    
    async def get_command_async(
        self,
        context: AttackContext,
        filtered_commands: Optional[List[CommandTemplate]] = None,
        force_provider: Optional[str] = None
    ) -> DualMentorResponse:
        """
        Get next command using dual-mentor system asynchronously.
        
        Args:
            context: Current attack context
            filtered_commands: Pre-filtered commands
            force_provider: Force specific provider ("gpt", "venice", "both")
            
        Returns:
            DualMentorResponse with chosen command
        """
        import asyncio
        
        result = DualMentorResponse(strategy=self.strategy)
        provider = force_provider or self._get_provider_for_call()
        
        # === PARALLEL / CONSENSUS: Query both ===
        if provider == "both" and self._has_both:
            try:
                # Run both in parallel
                gpt_task = asyncio.create_task(
                    self.gpt_mentor.get_command_async(context, filtered_commands)  # type: ignore
                )
                secondary_task = asyncio.create_task(
                    self.secondary_mentor.get_command_async(context, filtered_commands)  # type: ignore
                )
                
                results = await asyncio.gather(
                    gpt_task, secondary_task, return_exceptions=True
                )
                
                # Handle exceptions - safely cast to MentorResponse or None
                gpt_resp: Optional[MentorResponse] = None
                secondary_resp: Optional[MentorResponse] = None
                
                if isinstance(results[0], MentorResponse):
                    gpt_resp = results[0]
                elif isinstance(results[0], Exception):
                    self.logger.warning(f"GPT mentor error: {results[0]}")
                
                if isinstance(results[1], MentorResponse):
                    secondary_resp = results[1]
                elif isinstance(results[1], Exception):
                    self.logger.warning(f"Secondary mentor error: {results[1]}")
                
                result.primary = gpt_resp
                result.secondary = secondary_resp
                result.tokens_total = (
                    (gpt_resp.tokens_used if gpt_resp else 0) +
                    (secondary_resp.tokens_used if secondary_resp else 0)
                )
                
                # Choose best response
                if self.strategy == "consensus":
                    result.chosen, result.provider_used = self._consensus_choice(
                        gpt_resp, secondary_resp, context
                    )
                else:  # parallel - choose by confidence
                    result.chosen, result.provider_used = self._confidence_choice(
                        gpt_resp, secondary_resp
                    )
                
                return result
                
            except Exception as e:
                self.logger.error(f"Parallel query failed: {e}")
                # Fall through to sequential
                provider = "gpt"
        
        # === SEQUENTIAL: Primary then fallback ===
        primary_mentor = self.gpt_mentor if provider == "gpt" else self.secondary_mentor
        fallback_mentor = self.secondary_mentor if provider == "gpt" else self.gpt_mentor
        
        # Try primary
        if primary_mentor:
            try:
                primary_resp = await primary_mentor.get_command_async(context, filtered_commands)
                result.primary = primary_resp
                result.chosen = primary_resp
                result.provider_used = provider
                result.tokens_total = primary_resp.tokens_used
                
                if primary_resp.is_valid:
                    return result
                    
            except Exception as e:
                self.logger.warning(f"Primary mentor ({provider}) failed: {e}")
        
        # Try fallback
        if fallback_mentor:
            fallback_provider = "secondary" if provider == "gpt" else "gpt"
            try:
                fallback_resp = await fallback_mentor.get_command_async(context, filtered_commands)
                result.secondary = fallback_resp
                result.chosen = fallback_resp
                result.provider_used = fallback_provider
                result.tokens_total += fallback_resp.tokens_used
                
                self.logger.info(f"Used fallback mentor: {fallback_provider}")
                return result
                
            except Exception as e:
                self.logger.error(f"Fallback mentor ({fallback_provider}) also failed: {e}")
        
        # Ultimate fallback - return empty response
        return result
    
    def get_command(
        self,
        context: AttackContext,
        filtered_commands: Optional[List[CommandTemplate]] = None,
        force_provider: Optional[str] = None
    ) -> DualMentorResponse:
        """
        Get next command using dual-mentor system synchronously.
        
        Args:
            context: Current attack context
            filtered_commands: Pre-filtered commands
            force_provider: Force specific provider
            
        Returns:
            DualMentorResponse with chosen command
        """
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        self.get_command_async(context, filtered_commands, force_provider)
                    )
                    return future.result()
            else:
                return loop.run_until_complete(
                    self.get_command_async(context, filtered_commands, force_provider)
                )
        except RuntimeError:
            return asyncio.run(
                self.get_command_async(context, filtered_commands, force_provider)
            )
    
    def _confidence_choice(
        self,
        gpt_resp: Optional[MentorResponse],
        secondary_resp: Optional[MentorResponse]
    ) -> Tuple[Optional[MentorResponse], str]:
        """Choose response with higher confidence."""
        if not gpt_resp and not secondary_resp:
            return None, "none"
        if not gpt_resp:
            return secondary_resp, "secondary"
        if not secondary_resp:
            return gpt_resp, "gpt"
        
        # Both available - choose by confidence
        if secondary_resp.confidence > gpt_resp.confidence + 0.1:  # Secondary needs 0.1 edge
            return secondary_resp, "secondary"
        else:
            return gpt_resp, "gpt"
    
    def _consensus_choice(
        self,
        gpt_resp: Optional[MentorResponse],
        secondary_resp: Optional[MentorResponse],
        context: AttackContext
    ) -> Tuple[Optional[MentorResponse], str]:
        """Choose based on consensus between mentors."""
        if not gpt_resp and not secondary_resp:
            return None, "none"
        if not gpt_resp:
            return secondary_resp, "secondary"
        if not secondary_resp:
            return gpt_resp, "gpt"
        
        # Check if both chose same template
        if gpt_resp.template_name == secondary_resp.template_name:
            # Consensus! Use higher confidence
            if secondary_resp.confidence > gpt_resp.confidence:
                return secondary_resp, "secondary_consensus"
            return gpt_resp, "gpt_consensus"
        
        # No consensus - check if commands target same phase
        if gpt_resp.phase == secondary_resp.phase:
            # Same phase, use GPT (more conservative)
            return gpt_resp, "gpt_phase_match"
        
        # Different phases - use the one matching context
        if gpt_resp.phase == context.current_phase:
            return gpt_resp, "gpt_phase_aligned"
        if secondary_resp.phase == context.current_phase:
            return secondary_resp, "secondary_phase_aligned"
        
        # Default to GPT (primary)
        return gpt_resp, "gpt_default"
    
    def record_result(
        self,
        response: MentorResponse,
        success: bool,
        reward: float,
        context: AttackContext
    ) -> None:
        """Record result to learned store (shared between mentors)."""
        if self.gpt_mentor:
            self.gpt_mentor.record_result(response, success, reward, context)


def create_smart_mentor(
    llm_client: Any,
    model: str = "local-llm",  # Phase 12.1: full reasoning model
    temperature: float = 0.7
) -> SmartMentor:
    """
    Factory function to create a SmartMentor.
    
    Args:
        llm_client: LLM client with chat completion capability
        model: Model to use
        temperature: Sampling temperature
        
    Returns:
        Configured SmartMentor instance
    """
    return SmartMentor(
        llm_client=llm_client,
        model=model,
        temperature=temperature
    )


def create_dual_mentor(
    gpt_client: Any,
    venice_client: Optional[Any] = None,
    gpt_model: str = "local-llm",  # Phase 12.1: full reasoning model
    venice_model: str = "qwen3-coder-480b-a35b-instruct",
    strategy: str = "gpt_first",
    temperature: float = 0.7,
    *,
    secondary_client: Optional[Any] = None,
    secondary_model: Optional[str] = None,
) -> DualMentor:
    """
    Factory function to create a DualMentor.
    
    Phase 43: Supports secondary_client (local LLM) as explicit override.
    venice_client/venice_model still accepted for backward compat.
    
    Args:
        gpt_client: OpenAI client for GPT
        venice_client: OpenAI-compatible client for secondary provider (legacy name)
        gpt_model: GPT model to use
        venice_model: Secondary model (legacy name)
        strategy: Mentor selection strategy (gpt_first, local_first, round_robin, parallel, consensus)
        temperature: Sampling temperature
        secondary_client: Phase 43: Explicit secondary client (overrides venice_client)
        secondary_model: Phase 43: Explicit secondary model name (overrides venice_model)
        
    Returns:
        Configured DualMentor instance
    """
    return DualMentor(
        gpt_client=gpt_client,
        venice_client=venice_client,
        gpt_model=gpt_model,
        venice_model=venice_model,
        strategy=strategy,
        temperature=temperature,
        secondary_client=secondary_client,
        secondary_model=secondary_model,
    )
