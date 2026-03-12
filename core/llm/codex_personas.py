#!/usr/bin/env python3
"""
core/llm/codex_personas.py — ARIASKA Codex Persona Router v1.0

Four distinct Codex/Claude personas for multi-layered reasoning:

  1. TACTICAL ADVISOR     — Immediate action queries (short horizon, 1-3 steps).
                            Answers "what command should I run RIGHT NOW?"
  2. STRATEGIC PLANNER    — Long-term macro/phase sequencing (5-20 steps).
                            Answers "what is the overall attack plan?"
  3. FIELD RESEARCHER     — Background info on targets/CVEs/services.
                            Answers "what do I know about this service?"
  4. VENTRILOQUIST COORD  — Coherence & final decision fusion.
                            Answers "do these agent actions make sense together?"

ALL persona outputs are validated against the command registry.
Any out-of-registry term is rejected and logged.

Integration:
  - Called from SmartCoach when stagnation or phase transition triggers.
  - Each persona has its own system prompt, model tier, and token budget.
  - Results are logged via JSONL watcher with persona tag.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from core.commands.command_registry import (
    COMMAND_REGISTRY,
    CommandTemplate,
    AttackPhase,
    get_valid_commands_for_state,
)

if TYPE_CHECKING:
    from core.gpt_manager import GPTManager

logger = logging.getLogger("ariaska.llm.codex_personas")


# ─────────────────────────────────────────────────────────────────────────────
# Persona Enum & Config
# ─────────────────────────────────────────────────────────────────────────────

class CodexPersona(str, Enum):
    """Four Codex reasoning personas."""
    TACTICAL = "tactical_advisor"
    STRATEGIC = "strategic_planner"
    RESEARCHER = "field_researcher"
    VENTRILOQUIST = "ventriloquist_coordinator"


@dataclass
class PersonaConfig:
    """Configuration for a single persona."""
    persona: CodexPersona
    model: str                  # LLM model to use
    task_type: str              # GPTManager task_type routing key
    max_tokens: int             # Max response tokens
    temperature: float = 0.3   # Lower = more deterministic
    system_prompt: str = ""    # Role-specific system prompt
    cooldown_steps: int = 2    # Min steps between calls
    max_per_episode: int = 5   # Max calls per episode


# ── Persona Defaults ──

PERSONA_CONFIGS: Dict[CodexPersona, PersonaConfig] = {
    CodexPersona.TACTICAL: PersonaConfig(
        persona=CodexPersona.TACTICAL,
        model="local-llm",
        task_type="tactical",
        max_tokens=200,
        temperature=0.2,
        cooldown_steps=1,
        max_per_episode=16,  # Phase 9.1: doubled for knowledge-augmented tactical reasoning
        system_prompt=(
            "You are a TACTICAL ADVISOR for an autonomous red-team agent. "
            "Given the agent's current state, phase, and recent actions, "
            "recommend the SINGLE BEST NEXT COMMAND to execute. "
            "You must choose from the approved command registry templates. "
            "Think in 1-3 step horizons. Be concrete and actionable. "
            "IMPORTANT: Respond ONLY with valid JSON — no markdown, no backticks.\n\n"
            "Response format:\n"
            '{"recommended_template": "template_name", "reason": "brief", "confidence": 0.85}'
        ),
    ),
    CodexPersona.STRATEGIC: PersonaConfig(
        persona=CodexPersona.STRATEGIC,
        model="local-llm",
        task_type="strategic",
        max_tokens=400,
        temperature=0.3,
        cooldown_steps=3,
        max_per_episode=8,  # Phase 9.1: doubled for deeper strategic planning
        system_prompt=(
            "You are a STRATEGIC PLANNER for an autonomous red-team agent attacking "
            "Metasploitable 2/3 (Linux). Plan a sequence of 5-10 macro-level actions "
            "that form an optimal attack chain. Consider the cyber kill chain: "
            "RECON → ENUMERATION → EXPLOITATION → PRIV_ESC → LATERAL → POST_EXPLOIT → EXFIL.\n\n"
            "Key MS2 attack paths:\n"
            "- vsftpd 2.3.4 backdoor (port 21) → root shell\n"
            "- Samba 3.0.20 usermap_script (port 139/445) → root\n"
            "- ingreslock backdoor (port 1524) → instant root\n"
            "- Tomcat default creds (port 8180) → WAR deploy → shell\n"
            "- PostgreSQL default creds (port 5432) → RCE\n\n"
            "IMPORTANT: All templates must come from the approved registry. "
            "Respond ONLY with valid JSON — no markdown.\n\n"
            "Response format:\n"
            '{"attack_chain": [{"template": "name", "reason": "why"}], '
            '"overall_strategy": "brief plan", "confidence": 0.8}'
        ),
    ),
    CodexPersona.RESEARCHER: PersonaConfig(
        persona=CodexPersona.RESEARCHER,
        model="local-llm",
        task_type="tactical",
        max_tokens=300,
        temperature=0.4,
        cooldown_steps=2,
        max_per_episode=12,  # Phase 9.1: doubled for knowledge-base-powered research
        system_prompt=(
            "You are a FIELD RESEARCHER for an autonomous red-team agent. "
            "Given a discovered service or vulnerability, provide concise "
            "background intelligence: known CVEs, default credentials, "
            "exploitation techniques, and which command templates to use.\n\n"
            "Focus on actionable intelligence that maps to approved commands. "
            "IMPORTANT: Respond ONLY with valid JSON — no markdown.\n\n"
            "Response format:\n"
            '{"service": "name", "cves": ["CVE-XXXX-YYYY"], '
            '"default_creds": ["user:pass"], "recommended_templates": ["t1", "t2"], '
            '"exploitation_notes": "brief", "confidence": 0.9}'
        ),
    ),
    CodexPersona.VENTRILOQUIST: PersonaConfig(
        persona=CodexPersona.VENTRILOQUIST,
        model="local-llm",
        task_type="strategic",
        max_tokens=350,
        temperature=0.2,
        cooldown_steps=4,
        max_per_episode=6,  # Phase 9.1: doubled for multi-agent coherence
        system_prompt=(
            "You are a VENTRILOQUIST COORDINATOR overseeing 5 autonomous agents: "
            "Red (offensive), Blue (defensive), Scout (recon), Shadow (stealth), "
            "Orion (strategic). Your role is to ensure COHERENCE across agents.\n\n"
            "Given each agent's planned action, evaluate:\n"
            "1. Are actions complementary or conflicting?\n"
            "2. Is any agent wasting a step on redundant work?\n"
            "3. Should any agent's action be swapped for better synergy?\n\n"
            "IMPORTANT: All suggested templates must be from the approved registry. "
            "Respond ONLY with valid JSON — no markdown.\n\n"
            "Response format:\n"
            '{"coherence_score": 0.85, "conflicts": [], '
            '"overrides": [{"agent": "name", "new_template": "t", "reason": "why"}], '
            '"assessment": "brief"}'
        ),
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Persona Query Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PersonaResult:
    """Result from a persona query."""
    persona: CodexPersona
    success: bool
    template_name: Optional[str] = None       # Validated registry template
    templates: List[str] = field(default_factory=list)  # For strategic chains
    reasoning: str = ""
    confidence: float = 0.0
    raw_response: str = ""
    model_used: str = ""
    tokens_used: int = 0
    latency_ms: float = 0.0
    error: Optional[str] = None

    def to_telemetry(self) -> Dict[str, Any]:
        """Serialise for JSONL logging."""
        return {
            "persona": self.persona.value,
            "success": self.success,
            "template": self.template_name,
            "templates": self.templates[:5],
            "reasoning": self.reasoning[:120],
            "confidence": round(self.confidence, 3),
            "model": self.model_used,
            "tokens": self.tokens_used,
            "latency_ms": round(self.latency_ms, 1),
        }


# ─────────────────────────────────────────────────────────────────────────────
# CodexPersonaRouter
# ─────────────────────────────────────────────────────────────────────────────

class CodexPersonaRouter:
    """
    Routes queries to the appropriate Codex persona and validates outputs.

    All persona outputs are cross-referenced against CommandRegistry.
    Out-of-registry suggestions are rejected with a logged warning.

    Usage:
        router = CodexPersonaRouter(gpt_manager)
        result = router.query_tactical(state_context)
        if result.success and result.template_name:
            # Use result.template_name with CommandActionMapper
    """

    def __init__(
        self,
        gpt_manager: "GPTManager",
        configs: Optional[Dict[CodexPersona, PersonaConfig]] = None,
    ):
        self.gpt = gpt_manager
        self.configs = configs or dict(PERSONA_CONFIGS)

        # Per-episode tracking
        self._calls_per_persona: Dict[CodexPersona, int] = {p: 0 for p in CodexPersona}
        self._cooldowns: Dict[CodexPersona, int] = {p: 0 for p in CodexPersona}
        self._total_calls = 0
        self._total_rejections = 0

        # Build valid template name set for fast lookup
        self._valid_templates: Set[str] = set(COMMAND_REGISTRY.keys())

        logger.info(
            f"CodexPersonaRouter initialized: {len(self._valid_templates)} templates, "
            f"4 personas"
        )

    # ─────────────────────────────────────────────────────────────────────
    # Episode lifecycle
    # ─────────────────────────────────────────────────────────────────────

    def reset_episode(self) -> None:
        """Reset per-episode state."""
        self._calls_per_persona = {p: 0 for p in CodexPersona}
        self._cooldowns = {p: 0 for p in CodexPersona}

    def tick_cooldowns(self) -> None:
        """Call once per step to decrement cooldowns."""
        for p in CodexPersona:
            if self._cooldowns[p] > 0:
                self._cooldowns[p] -= 1

    # ─────────────────────────────────────────────────────────────────────
    # Public query methods
    # ─────────────────────────────────────────────────────────────────────

    def query_tactical(
        self,
        phase: str,
        target: str,
        state_flags: Dict[str, Any],
        recent_commands: List[str],
        discoveries: Dict[str, Any],
        agent_name: str = "RedAgent",
    ) -> PersonaResult:
        """Ask Tactical Advisor for immediate next command."""
        config = self.configs[CodexPersona.TACTICAL]
        if not self._can_call(CodexPersona.TACTICAL):
            return PersonaResult(persona=CodexPersona.TACTICAL, success=False, error="budget/cooldown")

        # Build context prompt
        _ports = sorted(list(discoveries.get("ports", [])))[:15]
        _services = sorted(list(discoveries.get("services", [])))[:10]
        _creds = list(discoveries.get("credentials", []))[:5]
        _shells = list(discoveries.get("shells", []))[:3]

        prompt = (
            f"Phase: {phase} | Target: {target} | Agent: {agent_name}\n"
            f"Ports: {_ports}\n"
            f"Services: {_services}\n"
            f"Credentials: {_creds if _creds else 'msfadmin:msfadmin (default)'}\n"
            f"Shells: {_shells}\n"
            f"Flags: shell={'Y' if state_flags.get('shell_obtained') else 'N'}, "
            f"root={'Y' if state_flags.get('root_shell_obtained') else 'N'}, "
            f"creds={'Y' if state_flags.get('credentials_known') else 'N'}\n"
            f"Recent commands: {', '.join(c[:40] for c in recent_commands[-5:])}\n\n"
            f"What is the SINGLE BEST command template to run next?"
        )

        result = self._query_persona(CodexPersona.TACTICAL, prompt, agent_name)

        # Parse tactical response → single template
        if result.success and result.raw_response:
            parsed = self._parse_tactical_response(result.raw_response)
            if parsed:
                result.template_name = parsed["template"]
                result.reasoning = parsed.get("reason", "")
                result.confidence = parsed.get("confidence", 0.8)
            else:
                result.success = False
                result.error = "no_valid_template"

        return result

    def query_strategic(
        self,
        phase: str,
        target: str,
        state_flags: Dict[str, Any],
        discoveries: Dict[str, Any],
        episode_history: List[str],
        agent_name: str = "RedAgent",
    ) -> PersonaResult:
        """Ask Strategic Planner for an attack chain."""
        config = self.configs[CodexPersona.STRATEGIC]
        if not self._can_call(CodexPersona.STRATEGIC):
            return PersonaResult(persona=CodexPersona.STRATEGIC, success=False, error="budget/cooldown")

        _ports = sorted(list(discoveries.get("ports", [])))[:15]
        _services = sorted(list(discoveries.get("services", [])))[:10]

        prompt = (
            f"Phase: {phase} | Target: {target}\n"
            f"Ports: {_ports}\n"
            f"Services: {_services}\n"
            f"Flags: shell={'Y' if state_flags.get('shell_obtained') else 'N'}, "
            f"root={'Y' if state_flags.get('root_shell_obtained') else 'N'}\n"
            f"Episode so far ({len(episode_history)} commands): "
            f"{', '.join(c[:30] for c in episode_history[-8:])}\n\n"
            f"Plan the next 5-10 step attack chain using approved templates."
        )

        result = self._query_persona(CodexPersona.STRATEGIC, prompt, agent_name)

        if result.success and result.raw_response:
            parsed = self._parse_strategic_response(result.raw_response)
            if parsed:
                result.templates = parsed.get("chain", [])
                result.reasoning = parsed.get("strategy", "")
                result.confidence = parsed.get("confidence", 0.8)
                if result.templates:
                    result.template_name = result.templates[0]
            else:
                result.success = False
                result.error = "no_valid_chain"

        return result

    def query_researcher(
        self,
        service: str,
        port: int,
        version: Optional[str] = None,
        target: str = "172.28.0.10",
        agent_name: str = "ScoutAgent",
    ) -> PersonaResult:
        """Ask Field Researcher about a specific service/vulnerability."""
        if not self._can_call(CodexPersona.RESEARCHER):
            return PersonaResult(persona=CodexPersona.RESEARCHER, success=False, error="budget/cooldown")

        prompt = (
            f"Service: {service} on port {port}"
            + (f" version {version}" if version else "")
            + f"\nTarget: {target} (Metasploitable Linux)\n\n"
            f"Provide: known CVEs, default credentials, exploitation templates "
            f"from the approved command registry."
        )

        # Phase 9.1: Inject knowledge base context for richer research
        try:
            from data.knowledge_retriever import get_knowledge_retriever
            kr = get_knowledge_retriever()
            kb_entries = kr.by_port(port, max_results=3)
            if not kb_entries:
                kb_entries = kr.by_service(service, max_results=3)
            if kb_entries:
                kb_context = "\n\n=== KNOWLEDGE BASE REFERENCE ===\n"
                for entry in kb_entries[:2]:
                    if isinstance(entry, dict):
                        kb_context += f"Service: {entry.get('service_name', service)}\n"
                        creds = entry.get("default_credentials", [])
                        if creds:
                            kb_context += f"Credentials: {creds[:3]}\n"
                        vulns = entry.get("common_vulnerabilities", [])
                        if vulns:
                            kb_context += f"CVEs: {vulns[:5]}\n"
                        reasoning = entry.get("reasoning", "")
                        if reasoning:
                            kb_context += f"Notes: {reasoning[:300]}\n"
                prompt += kb_context
        except Exception:
            pass  # Knowledge base is optional

        result = self._query_persona(CodexPersona.RESEARCHER, prompt, agent_name)

        if result.success and result.raw_response:
            parsed = self._parse_researcher_response(result.raw_response)
            if parsed:
                result.templates = parsed.get("templates", [])
                result.reasoning = parsed.get("notes", "")
                result.confidence = parsed.get("confidence", 0.9)
                if result.templates:
                    result.template_name = result.templates[0]

        return result

    def query_ventriloquist(
        self,
        agent_plans: Dict[str, str],
        phase: str,
        discoveries: Dict[str, Any],
        agent_name: str = "OrionAgent",
    ) -> PersonaResult:
        """Ask Ventriloquist to evaluate cross-agent coherence."""
        if not self._can_call(CodexPersona.VENTRILOQUIST):
            return PersonaResult(persona=CodexPersona.VENTRILOQUIST, success=False, error="budget/cooldown")

        _plan_lines = "\n".join(
            f"  {name}: {template[:60]}" for name, template in agent_plans.items()
        )

        prompt = (
            f"Phase: {phase}\n"
            f"Agent plans:\n{_plan_lines}\n"
            f"Ports: {len(discoveries.get('ports', []))}, "
            f"Shells: {len(discoveries.get('shells', []))}\n\n"
            f"Evaluate coherence. Suggest any overrides using approved templates only."
        )

        result = self._query_persona(CodexPersona.VENTRILOQUIST, prompt, agent_name)

        if result.success and result.raw_response:
            parsed = self._parse_ventriloquist_response(result.raw_response)
            if parsed:
                result.reasoning = parsed.get("assessment", "")
                result.confidence = parsed.get("coherence", 0.85)
                result.templates = parsed.get("overrides", [])

        return result

    # ─────────────────────────────────────────────────────────────────────
    # Private: query + validation
    # ─────────────────────────────────────────────────────────────────────

    def _can_call(self, persona: CodexPersona) -> bool:
        """Check budget and cooldown."""
        config = self.configs[persona]
        if self._calls_per_persona[persona] >= config.max_per_episode:
            return False
        if self._cooldowns[persona] > 0:
            return False
        if self.gpt is None or self.gpt.is_offline():
            return False
        return True

    def _query_persona(
        self,
        persona: CodexPersona,
        user_prompt: str,
        agent_id: str,
    ) -> PersonaResult:
        """Execute a persona query via GPTManager."""
        config = self.configs[persona]
        start = time.time()

        try:
            response = self.gpt.gpt_request(
                prompt=f"{config.system_prompt}\n\n---\n\n{user_prompt}",
                task_type=config.task_type,
                agent_id=agent_id,
                max_tokens=config.max_tokens,
            )

            latency = (time.time() - start) * 1000

            self._calls_per_persona[persona] += 1
            self._cooldowns[persona] = config.cooldown_steps
            self._total_calls += 1

            if response and isinstance(response, str) and len(response.strip()) > 5:
                return PersonaResult(
                    persona=persona,
                    success=True,
                    raw_response=response.strip(),
                    model_used=config.model,
                    latency_ms=latency,
                )
            else:
                return PersonaResult(
                    persona=persona,
                    success=False,
                    error="empty_response",
                    model_used=config.model,
                    latency_ms=latency,
                )

        except Exception as e:
            latency = (time.time() - start) * 1000
            logger.warning(f"[PERSONA-{persona.value}] Query failed: {e}")
            return PersonaResult(
                persona=persona,
                success=False,
                error=str(e)[:100],
                latency_ms=latency,
            )

    def _validate_template(self, name: str) -> Optional[str]:
        """Validate template name against registry. Returns normalised name or None."""
        if not name:
            return None

        # Exact match
        if name in self._valid_templates:
            return name

        # Case-insensitive match
        name_lower = name.lower().strip()
        for valid in self._valid_templates:
            if valid.lower() == name_lower:
                return valid

        # Fuzzy: check if name is a substring of any template
        for valid in self._valid_templates:
            if name_lower in valid.lower() or valid.lower() in name_lower:
                return valid

        self._total_rejections += 1
        logger.debug(f"[PERSONA] Rejected out-of-registry template: '{name}'")
        return None

    # ─────────────────────────────────────────────────────────────────────
    # Parsers (per persona response format)
    # ─────────────────────────────────────────────────────────────────────

    def _parse_json_safe(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from LLM response, stripping markdown fences."""
        clean = text.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            return json.loads(clean)
        except (json.JSONDecodeError, ValueError):
            # Try to find JSON object in response
            start = clean.find("{")
            end = clean.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(clean[start:end])
                except (json.JSONDecodeError, ValueError):
                    pass
        return None

    def _parse_tactical_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse tactical advisor response → {template, reason, confidence}."""
        parsed = self._parse_json_safe(response)
        if not parsed:
            return None

        template = self._validate_template(
            parsed.get("recommended_template", "")
        )
        if not template:
            return None

        return {
            "template": template,
            "reason": str(parsed.get("reason", ""))[:120],
            "confidence": min(0.95, max(0.3, float(parsed.get("confidence", 0.8)))),
        }

    def _parse_strategic_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse strategic planner response → {chain, strategy, confidence}."""
        parsed = self._parse_json_safe(response)
        if not parsed:
            return None

        raw_chain = parsed.get("attack_chain", [])
        validated_chain = []
        for item in raw_chain[:10]:
            if isinstance(item, dict):
                t = self._validate_template(item.get("template", ""))
            elif isinstance(item, str):
                t = self._validate_template(item)
            else:
                continue
            if t:
                validated_chain.append(t)

        if not validated_chain:
            return None

        return {
            "chain": validated_chain,
            "strategy": str(parsed.get("overall_strategy", ""))[:200],
            "confidence": min(0.95, max(0.3, float(parsed.get("confidence", 0.8)))),
        }

    def _parse_researcher_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse field researcher response → {templates, notes, confidence}."""
        parsed = self._parse_json_safe(response)
        if not parsed:
            return None

        raw_templates = parsed.get("recommended_templates", [])
        validated = [
            t for t in (self._validate_template(t) for t in raw_templates) if t
        ]

        return {
            "templates": validated,
            "cves": parsed.get("cves", [])[:5],
            "default_creds": parsed.get("default_creds", [])[:5],
            "notes": str(parsed.get("exploitation_notes", ""))[:200],
            "confidence": min(0.95, max(0.3, float(parsed.get("confidence", 0.9)))),
        }

    def _parse_ventriloquist_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse ventriloquist coordinator response → {coherence, overrides, assessment}."""
        parsed = self._parse_json_safe(response)
        if not parsed:
            return None

        overrides = []
        for ovr in parsed.get("overrides", []):
            if isinstance(ovr, dict):
                t = self._validate_template(ovr.get("new_template", ""))
                if t:
                    overrides.append(t)

        return {
            "coherence": min(1.0, max(0.0, float(parsed.get("coherence_score", 0.85)))),
            "overrides": overrides,
            "conflicts": parsed.get("conflicts", []),
            "assessment": str(parsed.get("assessment", ""))[:200],
        }

    # ─────────────────────────────────────────────────────────────────────
    # Metrics
    # ─────────────────────────────────────────────────────────────────────

    def get_metrics(self) -> Dict[str, Any]:
        """Return persona usage metrics."""
        return {
            "total_calls": self._total_calls,
            "total_rejections": self._total_rejections,
            "per_persona": {
                p.value: self._calls_per_persona[p] for p in CodexPersona
            },
        }
