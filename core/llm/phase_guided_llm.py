#!/usr/bin/env python3
"""
core/llm/phase_guided_llm.py — Phase 34: PhaseGuidedLLM Self-Guidance + Distillation

Produces structured JSON guidance for Ariaska agents:
  - Phase decision with evidence-driven stay/move conditions
  - 3-6 candidate next actions as template picks
  - Anomaly probes for stagnation/flag-hunting
  - Distillation packet (MentorTrace target) for apprentice policy training

All LLM calls go through GPTManager (never ``import openai`` directly).
Model routing: local-llm for all schema-bound JSON calls.
Escalation logic retained for confidence < 0.45, contradictions, or stall >= 8 steps.

Author: Phase 34
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.gpt_manager import GPTManager

logger = logging.getLogger("ariaska.llm.phase_guided")

# ── Constants ───────────────────────────────────────────────────────────────

_PHASE_TAG = "P34"

# Confidence below which we escalate to codex
_CODEX_ESCALATION_CONFIDENCE = 0.45

# Steps of zero discovery that count as semantic stall
_STALL_THRESHOLD = 8

# Max candidates in output
_MIN_CANDIDATES = 3
_MAX_CANDIDATES = 6

# Phase ordering for advancement logic
_PHASE_ORDER: Dict[str, int] = {
    "RECON": 0,
    "ENUM": 1,
    "ENUMERATION": 1,
    "EXPLOIT": 2,
    "EXPLOITATION": 2,
    "PRIVESC": 3,
    "PRIVILEGE_ESCALATION": 3,
    "POST": 4,
    "POST_EXPLOITATION": 4,
    "LATERAL_MOVEMENT": 4,
    "EXFIL": 5,
    "EXFILTRATION": 5,
    "CLOSEOUT": 6,
}

# Families allowed per phase (loose guide)
_PHASE_FAMILIES: Dict[str, List[str]] = {
    "RECON": ["nmap", "misc", "dns"],
    "ENUM": ["nmap", "web", "ssh", "smb", "dns", "misc"],
    "EXPLOIT": ["ssh", "web", "smb", "privesc", "file", "misc"],
    "PRIVESC": ["privesc", "file", "misc"],
    "POST": ["file", "privesc", "misc"],
    "EXFIL": ["file", "misc"],
    "CLOSEOUT": ["misc"],
}


# ── Output Dataclasses ──────────────────────────────────────────────────────

@dataclass
class PhaseDecision:
    """Phase decision block with P34 tag."""
    chosen_phase: str = ""
    phase_confidence: float = 0.0
    phase_goal: str = ""
    stay_conditions: List[str] = field(default_factory=list)
    move_on_conditions: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    phase_tag: str = _PHASE_TAG

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["phase_tag"] = _PHASE_TAG  # enforce
        return d


@dataclass
class Anomaly:
    """An anomaly worth probing."""
    finding: str = ""
    why_interesting: str = ""
    probe_template: str = ""
    risk: str = "low"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Candidate:
    """A candidate next action."""
    template_name: str = ""
    family: str = ""
    why: str = ""
    expected_outcome: str = ""
    stop_condition: str = ""
    confidence: float = 0.5
    risk: str = "low"
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Selection:
    """Selection of best candidate."""
    best_template_name: str = ""
    runner_up_template_name: str = ""
    selection_reason: str = ""
    should_escalate_to_codex: bool = False
    escalation_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GatingNotes:
    """Evidence gate assessment."""
    expected_gate_result: str = "PASS"
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ActionTarget:
    """Distillation action target."""
    template_name: str = ""
    why: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DistillationPacket:
    """Distillation packet for MentorTrace integration."""
    observation: str = ""
    reasoning: str = ""
    action_target: ActionTarget = field(default_factory=ActionTarget)
    expected_outcome: str = ""
    phase_target: str = ""
    confidence_target: float = 0.5
    gating_notes: GatingNotes = field(default_factory=GatingNotes)
    phase_tag: str = _PHASE_TAG

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "observation": self.observation,
            "reasoning": self.reasoning,
            "action_target": self.action_target.to_dict(),
            "expected_outcome": self.expected_outcome,
            "phase_target": self.phase_target,
            "confidence_target": round(self.confidence_target, 3),
            "gating_notes": self.gating_notes.to_dict(),
            "phase_tag": _PHASE_TAG,
        }
        return d


@dataclass
class PhaseGuidanceResult:
    """Full Phase 34 guidance output."""
    phase_decision: PhaseDecision = field(default_factory=PhaseDecision)
    anomalies: List[Anomaly] = field(default_factory=list)
    candidates: List[Candidate] = field(default_factory=list)
    selection: Selection = field(default_factory=Selection)
    distillation_packet: DistillationPacket = field(default_factory=DistillationPacket)
    model_used: str = ""
    escalated_to_codex: bool = False
    guidance_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase_decision": self.phase_decision.to_dict(),
            "anomalies": [a.to_dict() for a in self.anomalies],
            "candidates": [c.to_dict() for c in self.candidates],
            "selection": self.selection.to_dict(),
            "distillation_packet": self.distillation_packet.to_dict(),
            "model_used": self.model_used,
            "escalated_to_codex": self.escalated_to_codex,
            "guidance_id": self.guidance_id,
        }

    def validate(self) -> bool:
        """Check hard requirements: phase_tag present in both blocks."""
        return (
            self.phase_decision.phase_tag == _PHASE_TAG
            and self.distillation_packet.phase_tag == _PHASE_TAG
        )


# ── Helper: safe JSON extraction ────────────────────────────────────────────

def _fix_json_quirks(text: str) -> str:
    """Fix common LLM JSON quirks: trailing commas, single quotes, etc."""
    # Remove trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    # Replace single-quoted keys/values with double-quoted (simple heuristic)
    # Only when the text has no double-quotes (all single-quoted)
    if "'" in text and '"' not in text:
        text = text.replace("'", '"')
    return text


def _try_parse(text: str) -> Optional[Dict[str, Any]]:
    """Try json.loads, then fix quirks and retry."""
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass
    # Retry with quirks fixed
    try:
        fixed = _fix_json_quirks(text)
        obj = json.loads(fixed)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _brace_match(text: str) -> Optional[str]:
    """Extract the outermost balanced {...} block using brace counting."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    # Unbalanced — try closing it
    if depth > 0:
        return text[start:] + "}" * depth
    return None


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON object from LLM response using multiple strategies.

    Strategies (tried in order):
      1. Direct parse of full text
      2. Extract from ```json ... ``` fences
      3. Balanced brace-matching extraction
      4. First { to last } with quirk fixing
    """
    if not text or not isinstance(text, str):
        return None
    text = text.strip()

    # Reject obvious non-JSON (offline/echo placeholders)
    if text.startswith(("[OFFLINE]", "echo ", "nmap ", "ping ", "netstat ")):
        logger.debug("[PHASE-GUIDE] _extract_json: rejected placeholder: %s", text[:60])
        return None

    # Strategy 1: Direct parse
    result = _try_parse(text)
    if result:
        return result

    # Strategy 2: Fenced code block
    fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
    if fenced:
        result = _try_parse(fenced.group(1).strip())
        if result:
            return result

    # Strategy 3: Balanced brace matching
    matched = _brace_match(text)
    if matched:
        result = _try_parse(matched)
        if result:
            return result

    # Strategy 4: First { to last } (last resort)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        result = _try_parse(text[start:end + 1])
        if result:
            return result

    logger.debug("[PHASE-GUIDE] _extract_json: all strategies failed on: %s", text[:120])
    return None


def _safe_get(d: Any, key: str, default: Any = "") -> Any:
    """Safe dict get that handles non-dict inputs."""
    if isinstance(d, dict):
        return d.get(key, default)
    return default


# ── Phase inference from evidence ────────────────────────────────────────────

def _infer_phase(discovery_board: Dict[str, Any], current_phase: str) -> str:
    """Infer the correct phase from evidence (don't blindly trust input)."""
    ports = discovery_board.get("ports", [])
    services = discovery_board.get("services", [])
    creds = discovery_board.get("creds", [])
    shells = discovery_board.get("shells", [])
    flags = discovery_board.get("flags", [])
    vulns = discovery_board.get("vulns", [])

    # If we have root shell or flags → POST or EXFIL
    if any(s.get("type") == "root" for s in shells if isinstance(s, dict)):
        return "POST"
    if flags:
        return "EXFIL"
    # If we have user shell → PRIVESC
    if any(s.get("type") == "user" for s in shells if isinstance(s, dict)):
        return "PRIVESC"
    # If we have creds → at least EXPLOIT
    if creds:
        return "EXPLOIT"
    # If we have vulns → EXPLOIT candidate
    if vulns:
        return "EXPLOIT"
    # If we have services with details → ENUM done, could EXPLOIT
    if len(services) >= 3 and len(ports) >= 3:
        return "ENUM"
    # Sparse → RECON
    if len(ports) < 3:
        return "RECON"
    return current_phase.upper().split("_")[0] if current_phase else "RECON"


# ── Prompt builder ───────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are PHASE GUIDE for Ariaska_RL pentesting lab. "
    "Output ONLY a raw JSON object. No prose, no markdown fences. "
    'Include "phase_tag":"P34" in phase_decision and distillation_packet. '
    "Keys: phase_decision, anomalies, candidates, selection, distillation_packet. "
    "phase_decision needs: chosen_phase, phase_confidence (0-1), phase_goal, stay_conditions[], move_on_conditions[], contradictions[], phase_tag. "
    "candidates[]: template_name, family, why, expected_outcome, stop_condition, confidence, risk, tags[]. "
    "selection: best_template_name, runner_up_template_name, selection_reason, should_escalate_to_codex, escalation_reason. "
    "distillation_packet: observation, reasoning, action_target{template_name,why}, expected_outcome, phase_target, confidence_target, gating_notes{expected_gate_result,reasons[]}, phase_tag."
)

_JSON_RETRY_PROMPT = "Your previous response was not valid JSON. Output ONLY a raw JSON object. No text before or after. Start with { and end with }."

_MINIMAL_RETRY_PROMPT = '''Fill in this JSON template. Replace ALL "__" with real values. Output ONLY the completed JSON.
{"phase_decision":{"chosen_phase":"__","phase_confidence":0.5,"phase_goal":"__","stay_conditions":[],"move_on_conditions":[],"contradictions":[],"phase_tag":"P34"},"anomalies":[],"candidates":[{"template_name":"__","family":"__","why":"__","expected_outcome":"__","stop_condition":"__","confidence":0.5,"risk":"low","tags":[]}],"selection":{"best_template_name":"__","runner_up_template_name":"","selection_reason":"__","should_escalate_to_codex":false,"escalation_reason":""},"distillation_packet":{"observation":"__","reasoning":"__","action_target":{"template_name":"__","why":"__"},"expected_outcome":"__","phase_target":"__","confidence_target":0.5,"gating_notes":{"expected_gate_result":"PASS","reasons":[]},"phase_tag":"P34"}}'''


def _build_user_prompt(input_data: Dict[str, Any]) -> str:
    """Build the user prompt from input data."""
    return (
        "Analyze the following tactical state and return STRICT JSON guidance.\n\n"
        f"```json\n{json.dumps(input_data, indent=2, default=str)}\n```\n\n"
        "Return your response as a single JSON object with keys: "
        "phase_decision, anomalies, candidates, selection, distillation_packet. "
        "Include phase_tag:'P34' in both phase_decision and distillation_packet."
    )


# ── Main class ───────────────────────────────────────────────────────────────

class PhaseGuidedLLM:
    """
    Phase 34: Structured phase-aware guidance with distillation packets.

    Consumes discovery_board + phase_state + available_templates and produces
    ``PhaseGuidanceResult`` with evidence-driven candidates and a distillation
    packet compatible with MentorTrace.

    Model routing:
      - Default: local-llm (schema-bound JSON generation)
      - Additional escalation if:
        (a) confidence < 0.45 after scoring, OR
        (b) contradictions detected, OR
        (c) semantic stall >= 8 steps with no new discoveries

    All LLM calls go through the injected GPTManager instance.
    """

    def __init__(self, gpt_manager: "GPTManager") -> None:
        self._gpt = gpt_manager
        self._call_count = 0
        self._escalation_count = 0
        self._cache: Dict[str, PhaseGuidanceResult] = {}

    def reset_episode(self) -> None:
        """Reset per-episode state."""
        self._call_count = 0
        self._escalation_count = 0
        self._cache.clear()

    # ── Public API ───────────────────────────────────────────────────────

    def guide(
        self,
        episode_id: str,
        step_id: int,
        agent_role: str,
        current_phase: str,
        phase_state: Dict[str, Any],
        discovery_board: Dict[str, Any],
        available_templates: List[Dict[str, Any]],
        last_output_excerpt: str = "",
        agent_comms: str = "",
    ) -> Optional[PhaseGuidanceResult]:
        """Produce structured guidance for the given tactical state.

        Args:
            agent_comms: Phase 56 inter-agent bus messages formatted by
                ``AgentMessageBus.inject_into_prompt()``. Injected into the
                LLM prompt so phase decisions account for multi-agent context.

        Returns PhaseGuidanceResult on success, None if budget exhausted
        or LLM call fails.
        """
        if not self._gpt.can_make_request():
            logger.debug("[PHASE-GUIDE] Budget exhausted")
            return None

        # Phase 56: Local-only → fast single-call path (replaces P55 blanket skip)
        if getattr(self._gpt, 'is_local_only', lambda: False)():
            return self._fast_local_guide(
                episode_id, step_id, agent_role, current_phase,
                phase_state, discovery_board, available_templates,
                agent_comms=agent_comms,
            )

        # Determine stagnation
        stagnation_steps = phase_state.get("stagnation_steps", 0)
        recent_deltas = phase_state.get("recent_discovery_deltas", [])

        # Infer phase from evidence (don't blindly trust input)
        inferred_phase = _infer_phase(discovery_board, current_phase)

        # Detect contradictions
        contradictions = self._detect_contradictions(
            current_phase, inferred_phase, discovery_board
        )

        # Decide model: default mini, escalate to codex if needed
        should_escalate = (
            len(contradictions) > 0
            or stagnation_steps >= _STALL_THRESHOLD
        )
        # MODEL POLICY: Always use codex for schema-bound JSON generation.
        # Escalation flag kept for logging / downstream awareness.
        model = "local-llm"

        # Build input payload
        input_data = {
            "episode_id": episode_id,
            "step_id": step_id,
            "agent_role": agent_role,
            "current_phase": current_phase,
            "inferred_phase": inferred_phase,
            "phase_state": phase_state,
            "discovery_board": self._sanitize_board(discovery_board),
            "available_templates": available_templates[:20],  # cap
            "last_output_excerpt": last_output_excerpt[:500],
        }
        # Phase 56: Inject inter-agent comms for multi-agent awareness
        if agent_comms:
            input_data["agent_communications"] = agent_comms

        # LLM call — use strategic task_type for 9b brain routing + 20s timeout
        prompt = _build_user_prompt(input_data)
        raw = self._gpt.gpt_request(
            prompt=prompt,
            task_type="strategic",
            agent_id=f"phase_guide_{agent_role}",
            max_tokens=1024,  # Phase 56: generous ceiling — model thinks then outputs compact JSON
            model=model,
            system_prompt=_SYSTEM_PROMPT,
            timeout=600,
        )
        self._call_count += 1
        logger.debug("[PHASE-GUIDE] Raw response (attempt 1, %d chars): %.200s", len(raw) if raw else 0, raw)

        # Parse response
        parsed = _extract_json(raw)
        if parsed is None:
            # Retry once with explicit JSON-only instruction
            logger.debug("[PHASE-GUIDE] First parse failed, retrying with JSON-only prompt")
            raw_retry = self._gpt.gpt_request(
                prompt=_JSON_RETRY_PROMPT + "\n\n" + prompt,
                task_type="analysis",
                agent_id=f"phase_guide_{agent_role}",
                max_tokens=1024,  # Phase 56: generous ceiling
                model=model,
                system_prompt="You are a JSON-only API. Output ONLY valid JSON. No prose.",
                timeout=600,
            )
            self._call_count += 1
            logger.debug("[PHASE-GUIDE] Raw response (attempt 2, %d chars): %.200s", len(raw_retry) if raw_retry else 0, raw_retry)
            parsed = _extract_json(raw_retry)

        # Phase 52: Removed 3rd retry — 2 attempts max for cost optimization
        if parsed is None:
            logger.warning("[PHASE-GUIDE] Both JSON attempts failed — using deterministic heuristics")
            return self._heuristic_fallback(
                inferred_phase, discovery_board, available_templates,
                stagnation_steps, agent_role, model,
            )

        # Build result from parsed JSON
        result = self._build_result(
            parsed, inferred_phase, contradictions,
            model, should_escalate,
        )

        # Phase 38 D1: Post-parse codex escalation removed — model is always
        # local-llm since PR-4, so the re-run was dead code burning tokens.

        # Validate phase_tag presence
        if not result.validate():
            logger.warning("[PHASE-GUIDE] Missing phase_tag, forcing P34")
            result.phase_decision.phase_tag = _PHASE_TAG
            result.distillation_packet.phase_tag = _PHASE_TAG

        return result

    # ── Phase 56: Fast local single-call path ──────────────────────────

    def _fast_local_guide(
        self,
        episode_id: str,
        step_id: int,
        agent_role: str,
        current_phase: str,
        phase_state: Dict[str, Any],
        discovery_board: Dict[str, Any],
        available_templates: List[Dict[str, Any]],
        agent_comms: str = "",
    ) -> Optional[PhaseGuidanceResult]:
        """Single-call fast path for local 4B LLM.

        Instead of the full schema-bound prompt + retry loop, ask a single
        compact question: stay-or-advance + suggest 3 candidates.
        Falls through to heuristic on failure.
        """
        if not self._gpt.can_make_request():
            return None

        inferred_phase = _infer_phase(discovery_board, current_phase)
        stagnation_steps = phase_state.get("stagnation_steps", 0)

        # Sanitize board for prompt
        ports = sorted(str(p) for p in list(discovery_board.get("ports", []))[:8])
        services = sorted(str(s) for s in list(discovery_board.get("services", []))[:8])
        creds = list(discovery_board.get("creds", []))[:3]
        shells = list(discovery_board.get("shells", []))[:3]
        tmpl_names = [
            t.get("name", str(t)) if isinstance(t, dict) else str(t)
            for t in available_templates[:10]
        ]

        # Phase 56: Include inter-agent comms if available
        comms_line = f"\nagent_comms={agent_comms[:200]}" if agent_comms else ""

        prompt = (
            "/no_think\n"
            f"phase={current_phase} inferred={inferred_phase} stagnation={stagnation_steps}\n"
            f"ports={ports} services={services} creds={creds} shells={shells}\n"
            f"templates={tmpl_names}{comms_line}\n"
            "Should we STAY in current phase or ADVANCE? Suggest 3 candidate templates.\n"
            "Reply ONLY JSON:\n"
            '{"stay_or_advance":"stay|advance","reason":"why",'
            '"candidates":["tmpl1","tmpl2","tmpl3"],"confidence":0.0-1.0}'
        )

        import time as _time
        _t0 = _time.monotonic()
        try:
            raw = self._gpt.gpt_request(
                prompt=prompt,
                task_type="classification",
                agent_id=f"phase_guide_local_{agent_role}",
                max_tokens=120,
                model="local-llm",
                timeout=30,
            )
            _latency_ms = int((_time.monotonic() - _t0) * 1000)
            self._call_count += 1

            parsed = _extract_json(raw)
            if parsed and parsed.get("candidates"):
                stay_or_advance = str(parsed.get("stay_or_advance", "stay"))
                reason = str(parsed.get("reason", ""))
                confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
                raw_candidates = parsed.get("candidates", [])
                if isinstance(raw_candidates, list):
                    candidate_names = [str(c) for c in raw_candidates[:_MAX_CANDIDATES]]
                else:
                    candidate_names = []

                chosen_phase = inferred_phase if stay_or_advance == "advance" else current_phase
                phase_decision = PhaseDecision(
                    chosen_phase=chosen_phase,
                    phase_confidence=confidence,
                    phase_goal=reason,
                    stay_conditions=["LLM advises stay"] if stay_or_advance == "stay" else [],
                    move_on_conditions=["LLM advises advance"] if stay_or_advance == "advance" else [],
                    contradictions=[],
                    phase_tag=_PHASE_TAG,
                )

                candidates = []
                for name in candidate_names:
                    candidates.append(Candidate(
                        template_name=name,
                        family="",
                        why=reason[:120],
                        expected_outcome="discovery or confirmation",
                        stop_condition="timeout or error",
                        confidence=confidence,
                        risk="low",
                        tags=["LOCAL_LLM"],
                    ))

                best_name = candidate_names[0] if candidate_names else ""
                selection = Selection(
                    best_template_name=best_name,
                    runner_up_template_name=candidate_names[1] if len(candidate_names) > 1 else "",
                    selection_reason=f"local_llm: {reason[:80]}",
                    should_escalate_to_codex=False,
                    escalation_reason="",
                )

                distillation_packet = DistillationPacket(
                    observation=f"{len(ports)} ports, {len(services)} services",
                    reasoning=reason[:200],
                    action_target=ActionTarget(template_name=best_name, why=reason[:80]),
                    expected_outcome="New discovery",
                    phase_target=chosen_phase,
                    confidence_target=confidence,
                    gating_notes=GatingNotes(expected_gate_result="PASS", reasons=[]),
                    phase_tag=_PHASE_TAG,
                )

                result = PhaseGuidanceResult(
                    phase_decision=phase_decision,
                    anomalies=[],
                    candidates=candidates,
                    selection=selection,
                    distillation_packet=distillation_packet,
                    model_used="local-llm-4b",
                    escalated_to_codex=False,
                )

                # Publish reasoning for dashboard
                self._gpt._last_reasoning = {
                    "source": "phase_guided",
                    "stay_or_advance": stay_or_advance,
                    "candidates": candidate_names,
                    "reasoning": reason[:200],
                    "confidence": confidence,
                    "latency_ms": _latency_ms,
                    "model": "qwen3.5:4b",
                }
                logger.info(
                    f"PG:local {stay_or_advance} conf={confidence:.2f} "
                    f"best={best_name} latency={_latency_ms}ms"
                )
                return result

        except Exception as e:
            logger.debug(f"[PHASE-GUIDE:local] Fast path failed: {e}")

        # Fallback: heuristic
        return self._heuristic_fallback(
            inferred_phase, discovery_board, available_templates,
            stagnation_steps, agent_role, "local-llm-heuristic",
        )

    # ── Internal helpers ─────────────────────────────────────────────────

    def _detect_contradictions(
        self,
        current_phase: str,
        inferred_phase: str,
        discovery_board: Dict[str, Any],
    ) -> List[str]:
        """Detect phase contradictions between input and evidence."""
        contradictions: List[str] = []
        curr_order = _PHASE_ORDER.get(current_phase.upper(), 0)
        inf_order = _PHASE_ORDER.get(inferred_phase.upper(), 0)

        if abs(curr_order - inf_order) >= 2:
            contradictions.append(
                f"Phase mismatch: input={current_phase} vs evidence={inferred_phase}"
            )

        # Exploit phase but no services
        if curr_order >= 2 and not discovery_board.get("services"):
            contradictions.append("In EXPLOIT+ phase but no services discovered")

        # PRIVESC but no shells
        if curr_order >= 3 and not discovery_board.get("shells"):
            contradictions.append("In PRIVESC+ phase but no shells obtained")

        return contradictions

    def _sanitize_board(self, board: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize discovery board for JSON serialization."""
        sanitized: Dict[str, Any] = {}
        for key, val in board.items():
            if isinstance(val, set):
                sanitized[key] = sorted(str(v) for v in val)
            elif isinstance(val, (list, dict, str, int, float, bool)):
                sanitized[key] = val
            else:
                sanitized[key] = str(val)
        return sanitized

    def _build_result(
        self,
        parsed: Dict[str, Any],
        inferred_phase: str,
        contradictions: List[str],
        model: str,
        escalated: bool,
    ) -> PhaseGuidanceResult:
        """Build PhaseGuidanceResult from parsed JSON."""
        # Phase decision
        pd_raw = _safe_get(parsed, "phase_decision", {})
        phase_decision = PhaseDecision(
            chosen_phase=_safe_get(pd_raw, "chosen_phase", inferred_phase),
            phase_confidence=float(_safe_get(pd_raw, "phase_confidence", 0.5)),
            phase_goal=_safe_get(pd_raw, "phase_goal", ""),
            stay_conditions=_safe_get(pd_raw, "stay_conditions", []),
            move_on_conditions=_safe_get(pd_raw, "move_on_conditions", []),
            contradictions=contradictions or _safe_get(pd_raw, "contradictions", []),
            phase_tag=_PHASE_TAG,
        )

        # Anomalies
        anomalies_raw = _safe_get(parsed, "anomalies", [])
        anomalies = []
        if isinstance(anomalies_raw, list):
            for a in anomalies_raw[:5]:
                if isinstance(a, dict):
                    anomalies.append(Anomaly(
                        finding=_safe_get(a, "finding", ""),
                        why_interesting=_safe_get(a, "why_interesting", ""),
                        probe_template=_safe_get(a, "probe_template", ""),
                        risk=_safe_get(a, "risk", "low"),
                    ))

        # Candidates
        candidates_raw = _safe_get(parsed, "candidates", [])
        candidates = []
        if isinstance(candidates_raw, list):
            for c in candidates_raw[:_MAX_CANDIDATES]:
                if isinstance(c, dict):
                    tags_raw = _safe_get(c, "tags", [])
                    tags = tags_raw if isinstance(tags_raw, list) else []
                    candidates.append(Candidate(
                        template_name=_safe_get(c, "template_name", ""),
                        family=_safe_get(c, "family", ""),
                        why=_safe_get(c, "why", ""),
                        expected_outcome=_safe_get(c, "expected_outcome", ""),
                        stop_condition=_safe_get(c, "stop_condition", ""),
                        confidence=float(_safe_get(c, "confidence", 0.5)),
                        risk=_safe_get(c, "risk", "low"),
                        tags=tags,
                    ))

        # Selection
        sel_raw = _safe_get(parsed, "selection", {})
        selection = Selection(
            best_template_name=_safe_get(sel_raw, "best_template_name", ""),
            runner_up_template_name=_safe_get(sel_raw, "runner_up_template_name", ""),
            selection_reason=_safe_get(sel_raw, "selection_reason", ""),
            should_escalate_to_codex=bool(_safe_get(sel_raw, "should_escalate_to_codex", False)),
            escalation_reason=_safe_get(sel_raw, "escalation_reason", ""),
        )

        # Distillation packet
        dp_raw = _safe_get(parsed, "distillation_packet", {})
        at_raw = _safe_get(dp_raw, "action_target", {})
        gn_raw = _safe_get(dp_raw, "gating_notes", {})
        gn_reasons = _safe_get(gn_raw, "reasons", [])
        distillation_packet = DistillationPacket(
            observation=_safe_get(dp_raw, "observation", ""),
            reasoning=_safe_get(dp_raw, "reasoning", ""),
            action_target=ActionTarget(
                template_name=_safe_get(at_raw, "template_name", ""),
                why=_safe_get(at_raw, "why", ""),
            ),
            expected_outcome=_safe_get(dp_raw, "expected_outcome", ""),
            phase_target=_safe_get(dp_raw, "phase_target", inferred_phase),
            confidence_target=float(_safe_get(dp_raw, "confidence_target", 0.5)),
            gating_notes=GatingNotes(
                expected_gate_result=_safe_get(gn_raw, "expected_gate_result", "PASS"),
                reasons=gn_reasons if isinstance(gn_reasons, list) else [],
            ),
            phase_tag=_PHASE_TAG,
        )

        if escalated:
            self._escalation_count += 1

        return PhaseGuidanceResult(
            phase_decision=phase_decision,
            anomalies=anomalies,
            candidates=candidates,
            selection=selection,
            distillation_packet=distillation_packet,
            model_used=model,
            escalated_to_codex=escalated,
        )

    def _heuristic_fallback(
        self,
        inferred_phase: str,
        discovery_board: Dict[str, Any],
        available_templates: List[Dict[str, Any]],
        stagnation_steps: int,
        agent_role: str,
        model: str,
    ) -> PhaseGuidanceResult:
        """Build a heuristic fallback when LLM parsing fails."""
        # Pick templates matching phase
        phase_key = inferred_phase.upper().split("_")[0]
        allowed_families = _PHASE_FAMILIES.get(phase_key, ["misc"])

        candidates: List[Candidate] = []
        for tmpl in available_templates[:_MAX_CANDIDATES]:
            name = tmpl.get("name", "") if isinstance(tmpl, dict) else str(tmpl)
            family = tmpl.get("family", "misc") if isinstance(tmpl, dict) else "misc"
            if family in allowed_families or len(candidates) < _MIN_CANDIDATES:
                candidates.append(Candidate(
                    template_name=name,
                    family=family,
                    why=f"Phase-matched for {inferred_phase}",
                    expected_outcome="discovery or confirmation",
                    stop_condition="no output or error",
                    confidence=0.4,
                    risk="low",
                    tags=["EVIDENCE_DRIVEN"],
                ))

        best_name = candidates[0].template_name if candidates else ""

        ports = discovery_board.get("ports", [])
        services = discovery_board.get("services", [])
        obs = f"{len(ports)} ports, {len(services)} services discovered"

        phase_decision = PhaseDecision(
            chosen_phase=inferred_phase,
            phase_confidence=0.4,
            phase_goal=f"Continue {inferred_phase} with available evidence",
            stay_conditions=[f"Only {len(ports)} ports known"],
            move_on_conditions=["3+ services confirmed with versions"],
            contradictions=[],
            phase_tag=_PHASE_TAG,
        )

        distillation_packet = DistillationPacket(
            observation=obs,
            reasoning=f"Heuristic fallback for {agent_role} in {inferred_phase}",
            action_target=ActionTarget(
                template_name=best_name,
                why="Best available template for current phase",
            ),
            expected_outcome="New discovery or confirmation",
            phase_target=inferred_phase,
            confidence_target=0.4,
            gating_notes=GatingNotes(
                expected_gate_result="PASS",
                reasons=["Heuristic fallback — low confidence"],
            ),
            phase_tag=_PHASE_TAG,
        )

        return PhaseGuidanceResult(
            phase_decision=phase_decision,
            anomalies=[],
            candidates=candidates,
            selection=Selection(
                best_template_name=best_name,
                runner_up_template_name=candidates[1].template_name if len(candidates) > 1 else "",
                selection_reason="Heuristic fallback (LLM parse failed)",
                should_escalate_to_codex=True,
                escalation_reason="LLM parse failed",
            ),
            distillation_packet=distillation_packet,
            model_used=model,
            escalated_to_codex=False,
        )

    def to_mentor_trace_kwargs(
        self, result: PhaseGuidanceResult, step: int, episode: int,
        agent_id: str, state_vector: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Convert guidance result into kwargs for MentorTrace constructor.

        This bridges Phase 34 guidance into the existing MentorTrace/TeacherTrace
        distillation pipeline.
        """
        dp = result.distillation_packet
        sel = result.selection
        best_candidates = [
            c.template_name for c in result.candidates[:3] if c.template_name
        ]
        return {
            "command": "",  # Filled by caller after template rendering
            "template_name": sel.best_template_name,
            "reasoning": dp.reasoning[:512],
            "confidence": dp.confidence_target,
            "alternatives": best_candidates,
            "expected_outcome": dp.expected_outcome[:256],
            "phase": result.phase_decision.chosen_phase,
            "step": step,
            "episode": episode,
            "agent_id": agent_id,
            "state_vector": state_vector,
        }

    def get_stats(self) -> Dict[str, int]:
        """Return call statistics."""
        return {
            "guide_calls": self._call_count,
            "escalations": self._escalation_count,
            "cache_size": len(self._cache),
        }
