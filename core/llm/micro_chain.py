#!/usr/bin/env python3
"""
core/llm/micro_chain.py — Phase 27: Iterative Nano→Mini→Nano Scoring Chain

3-stage LLM reasoning chain for command selection:
  Stage 1 (nano):  Classify tactical situation
  Stage 2 (mini):  Generate candidate commands (JSON)
  Stage 3 (nano):  Score candidates for phase fit / evidence support / novelty

Optional codex escalation when top score < threshold.

All LLM calls via GPTManager — no direct openai imports.

Author: Phase 27
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.gpt_manager import GPTManager

logger = logging.getLogger("ariaska.llm.micro_chain")

# ── Constants ───────────────────────────────────────────────────────────────

# Phase 32: Tunable via MC_ESCALATE_THRESHOLD env var (default 0.40).
ESCALATION_THRESHOLD = float(os.environ.get("MC_ESCALATE_THRESHOLD", "0.40"))
STAGNATION_ESCALATION_STEPS = 9  # Also escalate if stagnation >= this
STAGNATION_ESCALATION_SCORE = 0.55  # ... and scores are mediocre
_MAX_CANDIDATES = 3  # Phase 33.3: hard cap on candidates

# P34-EXT: Ablation toggle — bypass nano stages 1+3 for A/B testing.
# Set MC_NANO_ABLATION=1 to disable nano classify + nano score stages.
# The chain becomes: heuristic_classify → mini_generate → heuristic_score.
NANO_ABLATION = os.environ.get("MC_NANO_ABLATION", "").strip() in ("1", "true", "yes")

# Phases at or above EXPLOIT that relax escalation gating
_EXPLOIT_PHASES = frozenset({
    "EXPLOITATION", "PRIVILEGE_ESCALATION",
    "LATERAL_MOVEMENT", "POST_EXPLOITATION", "EXFILTRATION",
})


# ── Dataclasses ─────────────────────────────────────────────────────────────

@dataclass
class MicroChainCandidate:
    """A single command candidate produced by the micro-chain."""
    command: str = ""
    template_name: str = ""
    reasoning: str = ""
    score: float = 0.0
    phase_fit: float = 0.0
    evidence_support: float = 0.0
    novelty: float = 0.0

    # P36.1 Fast-Learn: Structured quality fields (HARD REQUIREMENT)
    evidence_used: List[str] = field(default_factory=list)
    hypothesis: str = ""
    test: str = ""          # template_name alias for structured output
    expected_observable: str = ""
    stop_condition: str = ""
    confidence: float = 0.0

    @property
    def quality_complete(self) -> bool:
        """Check if all P36.1 structured fields are populated."""
        return bool(
            self.evidence_used
            and self.hypothesis
            and (self.test or self.template_name)
            and self.expected_observable
            and self.stop_condition
            and self.confidence > 0
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "command": self.command,
            "template_name": self.template_name,
            "reasoning": self.reasoning,
            "score": round(self.score, 3),
            "phase_fit": round(self.phase_fit, 3),
            "evidence_support": round(self.evidence_support, 3),
            "novelty": round(self.novelty, 3),
            "evidence_used": self.evidence_used,
            "hypothesis": self.hypothesis,
            "test": self.test or self.template_name,
            "expected_observable": self.expected_observable,
            "stop_condition": self.stop_condition,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class MicroChainResult:
    """Result of the micro-chain decision process."""
    selected: MicroChainCandidate = field(default_factory=MicroChainCandidate)
    candidates: List[MicroChainCandidate] = field(default_factory=list)
    escalated: bool = False
    chain_cost_usd: float = 0.0
    chain_tokens: int = 0
    stages_used: List[str] = field(default_factory=list)
    model_trace: str = ""  # e.g. "nano→mini→nano" or "nano→mini→nano→codex"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected": self.selected.to_dict(),
            "candidates": [c.to_dict() for c in self.candidates],
            "escalated": self.escalated,
            "chain_cost_usd": round(self.chain_cost_usd, 6),
            "chain_tokens": self.chain_tokens,
            "stages_used": self.stages_used,
            "model_trace": self.model_trace,
        }


# ── Helpers ─────────────────────────────────────────────────────────────────

def _safe_json_load(text: str) -> Optional[dict]:
    """Extract and parse JSON from LLM response, handling fenced blocks."""
    if not text or not text.strip():
        return None
    cleaned = text.strip()
    # Strip markdown fences
    if cleaned.startswith("```"):
        # Remove opening fence (```json or ```)
        first_nl = cleaned.find("\n")
        if first_nl != -1:
            cleaned = cleaned[first_nl + 1:]
        # Remove closing fence
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3].rstrip()
    # Try direct parse
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
        return None
    except (json.JSONDecodeError, ValueError):
        pass
    # Try to find JSON object in text
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _safe_json_load_list(text: Optional[str]) -> Optional[list]:
    """Extract and parse a JSON array from LLM response."""
    if not text or not text.strip():
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        first_nl = cleaned.find("\n")
        if first_nl != -1:
            cleaned = cleaned[first_nl + 1:]
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3].rstrip()
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, list):
            return obj
        return None
    except (json.JSONDecodeError, ValueError):
        pass
    match = re.search(r'\[.*\]', cleaned, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            if isinstance(obj, list):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _build_cache_key(
    phase: str,
    discovery_board: Dict[str, Any],
    agent_role: str,
    stagnation_steps: int,
    available_templates: List[str],
) -> str:
    """Build a stable, low-entropy cache key."""
    ports = sorted(str(p) for p in discovery_board.get("ports", []))
    services = sorted(str(s) for s in discovery_board.get("services", []))
    # Phase 38 B2: Semantic stagnation tiers instead of integer division
    if stagnation_steps == 0:
        stag_tier = "fresh"
    elif stagnation_steps <= 3:
        stag_tier = "warming"
    elif stagnation_steps <= 8:
        stag_tier = "stale"
    else:
        stag_tier = "stuck"
    # Hash templates list for stability
    tmpl_hash = hashlib.md5(
        ",".join(sorted(available_templates[:20])).encode()
    ).hexdigest()[:8]
    return f"mc:{phase}:{','.join(ports[:10])}:{','.join(services[:10])}:{agent_role}:{stag_tier}:{tmpl_hash}"


# ── MicroChain Class ───────────────────────────────────────────────────────

class MicroChain:
    """
    Iterative nano→mini→nano scoring chain for command selection.

    Stage 1 (nano):  Classify tactical situation → situation label
    Stage 2 (mini):  Generate candidate commands (structured JSON)
    Stage 3 (nano):  Score each candidate for phase_fit, evidence_support, novelty
    Optional (codex): Escalate if best score < ESCALATION_THRESHOLD
    """

    def __init__(
        self,
        gpt_manager: "GPTManager",
        cache: Optional[Dict[str, MicroChainResult]] = None,
    ) -> None:
        self._gpt = gpt_manager
        self._cache: Dict[str, MicroChainResult] = cache if cache is not None else {}
        self._total_tokens = 0
        self._total_cost = 0.0
        # P36.1: Model call counter for 70/30 nano/mini routing
        self._nano_calls = 0
        self._mini_calls = 0

    def decide(
        self,
        phase: str,
        discovery_board: Dict[str, Any],
        recent_commands: List[str],
        available_templates: List[str],
        agent_role: str,
        stagnation_steps: int = 0,
    ) -> Optional[MicroChainResult]:
        """
        Run the micro-chain to select a command.

        Returns MicroChainResult on success, None on failure or budget exhaustion.
        """
        # Budget check
        if not self._gpt.can_make_request():
            logger.debug("[MICRO-CHAIN] Budget exhausted, returning None")
            return None

        # Cache check
        cache_key = _build_cache_key(
            phase, discovery_board, agent_role, stagnation_steps, available_templates
        )
        if cache_key in self._cache:
            logger.debug(f"[MICRO-CHAIN] Cache hit: {cache_key[:40]}")
            return self._cache[cache_key]

        stages_used: List[str] = []
        total_tokens = 0

        # ── Stage 1: Classify situation (nano or heuristic) ─────────
        if NANO_ABLATION:
            # Ablation: skip nano classify, use heuristic mapping
            situation = self._heuristic_classify(phase, discovery_board)
            stages_used.append("heuristic_classify")
        else:
            situation = self._classify(
                phase, discovery_board, recent_commands, agent_role
            )
            stages_used.append("nano_classify")
            total_tokens += 80  # estimate

        # ── Stage 2: Generate candidates (codex) ─────────────────────
        candidates = self._generate_candidates(
            phase, discovery_board, situation, available_templates, agent_role
        )
        if candidates is None:
            logger.debug("[MICRO-CHAIN] Stage 2 failed (malformed JSON), returning None")
            return None
        stages_used.append("codex_generate")
        total_tokens += 200  # estimate

        if not candidates:
            logger.debug("[MICRO-CHAIN] Stage 2 returned empty candidates")
            return None

        # Phase 33.3: Enforce max 3 candidates + template validation
        candidates = candidates[:_MAX_CANDIDATES]
        if available_templates:
            tmpl_set = set(available_templates)
            candidates = [
                c for c in candidates
                if not c.template_name or c.template_name in tmpl_set
            ]
            if not candidates:
                logger.debug("[MICRO-CHAIN] All candidates filtered (no template match)")
                return None

        # ── Stage 3: Score candidates (codex or heuristic) ─────────────
        if NANO_ABLATION:
            scored = self._heuristic_select(candidates, recent_commands)
            stages_used.append("heuristic_score")
        else:
            scored = self._score_candidates(
                candidates, phase, discovery_board, recent_commands
            )
            stages_used.append("codex_score")
            total_tokens += 100  # estimate

            if scored is None:
                # Stage 3 failed — select by heuristic
                scored = self._heuristic_select(candidates, recent_commands)
                stages_used[-1] = "heuristic_score"

        # Sort by score descending
        scored.sort(key=lambda c: c.score, reverse=True)
        best = scored[0]

        # ── Optional: Escalate to codex ─────────────────────────────
        # Phase 33.3: Escalation only if:
        #   top_score < 0.40  AND  (stagnation_bucket >= 2  OR  phase >= EXPLOIT)
        escalated = False
        stag_bucket = stagnation_steps // 3
        phase_upper = phase.upper() if isinstance(phase, str) else str(phase).upper()
        escalation_gate = (
            best.score < ESCALATION_THRESHOLD
            and (stag_bucket >= 2 or phase_upper in _EXPLOIT_PHASES)
        )
        if not escalation_gate and stagnation_steps >= STAGNATION_ESCALATION_STEPS:
            escalation_gate = best.score < STAGNATION_ESCALATION_SCORE

        if escalation_gate and self._gpt.can_make_request():
            codex_candidate = self._escalate_to_codex(
                phase, discovery_board, scored, agent_role
            )
            if codex_candidate is not None:
                scored.insert(0, codex_candidate)
                best = codex_candidate
                escalated = True
                stages_used.append("codex_escalate")
                total_tokens += 300  # estimate

        # Phase 33.3: Compact trace line
        scores_str = ",".join(f"{c.score:.2f}" for c in scored[:_MAX_CANDIDATES])
        logger.info(
            f"MC: {situation} | scores=[{scores_str}] | escalated={'yes' if escalated else 'no'}"
        )

        model_trace = "→".join(stages_used)
        result = MicroChainResult(
            selected=best,
            candidates=scored,
            escalated=escalated,
            chain_tokens=total_tokens,
            chain_cost_usd=total_tokens * 0.000001,  # rough estimate
            stages_used=stages_used,
            model_trace=model_trace,
        )

        # Cache
        self._cache[cache_key] = result
        self._total_tokens += total_tokens
        return result

    def _heuristic_classify(
        self,
        phase: str,
        discovery_board: Dict[str, Any],
    ) -> str:
        """P34-EXT: Ablation fallback — classify situation from phase + board state."""
        ports = discovery_board.get("ports", set())
        services = discovery_board.get("services", set())
        creds = discovery_board.get("credentials", set())
        shells = discovery_board.get("shells", set())

        phase_upper = phase.upper() if isinstance(phase, str) else str(phase).upper()

        if shells:
            if phase_upper in ("PRIVILEGE_ESCALATION", "LATERAL_MOVEMENT"):
                return "privesc_needed"
            return "post_exploit"
        if creds:
            return "exploit_ready"
        if len(services) >= 3:
            return "exploit_ready" if len(ports) >= 5 else "enum_needed"
        if len(ports) >= 2:
            return "enum_needed"
        return "recon_gap"

    def _classify(
        self,
        phase: str,
        discovery_board: Dict[str, Any],
        recent_commands: List[str],
        agent_role: str,
    ) -> str:
        """Stage 1: Classify tactical situation using nano model."""
        ports = list(discovery_board.get("ports", []))[:10]
        services = list(discovery_board.get("services", []))[:10]
        recent = recent_commands[-5:] if recent_commands else []

        prompt = (
            f"Classify the tactical situation in one word.\n"
            f"Phase: {phase}\nRole: {agent_role}\n"
            f"Ports: {ports}\nServices: {services}\n"
            f"Recent commands: {recent}\n"
            f"Options: recon_gap, enum_needed, exploit_ready, privesc_needed, "
            f"post_exploit, lateral_move, stalled\n"
            f"Reply with ONLY the situation label."
        )
        try:
            response = self._gpt.gpt_request(
                prompt=prompt,
                task_type="classification",
                agent_id="micro_chain",
                max_tokens=20,
                model="gpt-5-nano",
            )
            # Extract first word
            label = response.strip().split()[0].lower().rstrip(".,;:")
            if label in ("recon_gap", "enum_needed", "exploit_ready",
                         "privesc_needed", "post_exploit", "lateral_move", "stalled"):
                return label
            return "recon_gap"  # default fallback
        except Exception as e:
            logger.debug(f"[MICRO-CHAIN] Stage 1 classify failed: {e}")
            return "recon_gap"

    def _generate_candidates(
        self,
        phase: str,
        discovery_board: Dict[str, Any],
        situation: str,
        available_templates: List[str],
        agent_role: str,
    ) -> Optional[List[MicroChainCandidate]]:
        """Stage 2: Generate candidate commands using mini model."""
        ports = list(discovery_board.get("ports", []))[:10]
        services = list(discovery_board.get("services", []))[:10]
        creds = list(discovery_board.get("credentials", []))[:3]
        templates_sample = available_templates[:15]

        # P36.1: Structured quality prompt — every candidate MUST include
        # evidence_used, hypothesis, test, expected_observable, stop_condition, confidence
        prompt = (
            f"Generate 3 candidate commands for phase={phase}, situation={situation}, "
            f"role={agent_role}.\n"
            f"Known ports: {ports}\nKnown services: {services}\n"
            f"Known credentials: {creds}\n"
            f"Available templates: {templates_sample}\n\n"
            f"Reply with ONLY a JSON array. EVERY object MUST include ALL fields:\n"
            f'[{{"command":"...", "template_name":"...", "reasoning":"...", '
            f'"evidence_used":["port_22","service_ssh"], '
            f'"hypothesis":"SSH may allow password auth", '
            f'"test":"template_name", '
            f'"expected_observable":"SSH banner or auth prompt", '
            f'"stop_condition":"auth denied or timeout", '
            f'"confidence":0.7}}]\n'
            f"No markdown, no explanation outside JSON."
        )

        # MODEL POLICY: codex for all schema-bound JSON generation.
        _gen_model = "gpt-5.2-codex"
        self._codex_calls = getattr(self, '_codex_calls', 0) + 1

        try:
            response = self._gpt.gpt_request(
                prompt=prompt,
                task_type="analysis",
                agent_id="micro_chain",
                max_tokens=400,
                model=_gen_model,
                timeout=20,
            )
            parsed = _safe_json_load_list(response)
            if parsed is None:
                # Try as dict with "candidates" key
                obj = _safe_json_load(response)
                if obj and "candidates" in obj:
                    parsed = obj["candidates"]
                else:
                    return None

            candidates = []
            for item in parsed[:5]:  # limit to 5
                if not isinstance(item, dict):
                    continue
                c = MicroChainCandidate(
                    command=str(item.get("command", "")),
                    template_name=str(item.get("template_name", "")),
                    reasoning=str(item.get("reasoning", "")),
                    evidence_used=item.get("evidence_used", []) if isinstance(item.get("evidence_used"), list) else [],
                    hypothesis=str(item.get("hypothesis", "")),
                    test=str(item.get("test", item.get("template_name", ""))),
                    expected_observable=str(item.get("expected_observable", "")),
                    stop_condition=str(item.get("stop_condition", "")),
                    confidence=float(item.get("confidence", 0.0)) if isinstance(item.get("confidence"), (int, float)) else 0.0,
                )
                if c.command:
                    candidates.append(c)

            # P36.1: Quality enforcement — if any candidate missing structured fields,
            # try to fill with a nano call, then escalate to mini if still incomplete
            if candidates:
                incomplete = [c for c in candidates if not c.quality_complete]
                if incomplete:
                    self._fill_quality_fields(incomplete, phase, discovery_board)

            return candidates if candidates else None
        except Exception as e:
            logger.debug(f"[MICRO-CHAIN] Stage 2 generate failed: {e}")
            return None

    def _fill_quality_fields(
        self,
        candidates: List[MicroChainCandidate],
        phase: str,
        discovery_board: Dict[str, Any],
    ) -> None:
        """P36.1: Fill missing structured quality fields on candidates.

        First tries nano, then escalates to mini if still incomplete.
        """
        cmds = [{"cmd": c.command[:80], "tmpl": c.template_name} for c in candidates[:3]]
        ports = list(discovery_board.get("ports", []))[:10]
        services = list(discovery_board.get("services", []))[:10]

        fill_prompt = (
            f"For each command, provide the missing context fields.\n"
            f"Phase: {phase}\nPorts: {ports}\nServices: {services}\n"
            f"Commands: {json.dumps(cmds)}\n\n"
            f"Reply ONLY with JSON array:\n"
            f'[{{"idx":0, "evidence_used":["port_22"], '
            f'"hypothesis":"...", "test":"template_name", '
            f'"expected_observable":"...", "stop_condition":"...", '
            f'"confidence":0.7}}]\n'
            f"No markdown."
        )

        # MODEL POLICY: codex for schema-bound JSON fill/enrich.
        for _attempt in range(2):
            _fill_model = "gpt-5.2-codex"
            try:
                self._codex_calls = getattr(self, '_codex_calls', 0) + 1

                resp = self._gpt.gpt_request(
                    prompt=fill_prompt,
                    task_type="analysis",
                    agent_id="micro_chain",
                    max_tokens=250,
                    model=_fill_model,
                    timeout=20,
                )
                parsed = _safe_json_load_list(resp)
                if not parsed:
                    continue

                for item in parsed:
                    if not isinstance(item, dict):
                        continue
                    idx = int(item.get("idx", -1))
                    if 0 <= idx < len(candidates):
                        c = candidates[idx]
                        if not c.evidence_used and isinstance(item.get("evidence_used"), list):
                            c.evidence_used = item["evidence_used"]
                        if not c.hypothesis:
                            c.hypothesis = str(item.get("hypothesis", ""))
                        if not c.test:
                            c.test = str(item.get("test", c.template_name))
                        if not c.expected_observable:
                            c.expected_observable = str(item.get("expected_observable", ""))
                        if not c.stop_condition:
                            c.stop_condition = str(item.get("stop_condition", ""))
                        if c.confidence <= 0:
                            c.confidence = float(item.get("confidence", 0.5)) if isinstance(item.get("confidence"), (int, float)) else 0.5

                # Check if all are now complete
                still_incomplete = [c for c in candidates if not c.quality_complete]
                if not still_incomplete:
                    return
                if _attempt == 0:
                    logger.debug("[MICRO-CHAIN] P36.1: nano fill incomplete, escalating to mini")
                    continue
            except Exception as e:
                logger.debug(f"[MICRO-CHAIN] P36.1: fill_quality_fields failed ({_fill_model}): {e}")
                continue

        # Final fallback: set defaults for any remaining incomplete fields
        for c in candidates:
            if not c.evidence_used:
                c.evidence_used = ["implicit_phase_context"]
            if not c.hypothesis:
                c.hypothesis = f"Command may yield discoveries in {phase}"
            if not c.test:
                c.test = c.template_name or "unknown"
            if not c.expected_observable:
                c.expected_observable = "new data or error output"
            if not c.stop_condition:
                c.stop_condition = "timeout or completion"
            if c.confidence <= 0:
                c.confidence = 0.3  # Low confidence for unfilled

    def _score_candidates(
        self,
        candidates: List[MicroChainCandidate],
        phase: str,
        discovery_board: Dict[str, Any],
        recent_commands: List[str],
    ) -> Optional[List[MicroChainCandidate]]:
        """Stage 3: Score candidates using nano model with schema validation + retry."""
        cmd_list = [{"idx": i, "cmd": c.command[:80]} for i, c in enumerate(candidates)]
        recent = recent_commands[-5:] if recent_commands else []
        ports = list(discovery_board.get("ports", []))[:10]

        prompt = (
            f"Score these commands for phase={phase}.\n"
            f"Commands: {json.dumps(cmd_list)}\n"
            f"Known ports: {ports}\nRecent: {recent}\n\n"
            f"Reply ONLY with JSON array of objects:\n"
            f'[{{"idx":0,"phase_fit":0.8,"evidence_support":0.7,"novelty":0.6}}]\n'
            f"Scores 0.0-1.0. No markdown."
        )

        # P34-EXT: Retry loop (max 2 attempts) with schema validation
        for _attempt in range(2):
            try:
                # MODEL POLICY: codex for schema-bound JSON scoring.
                response = self._gpt.gpt_request(
                    prompt=prompt,
                    task_type="analysis",
                    agent_id="micro_chain",
                    max_tokens=200,
                    model="gpt-5.2-codex",
                    timeout=20,
                )
                parsed = _safe_json_load_list(response)
                if parsed is None:
                    if _attempt == 0:
                        logger.debug("[MICRO-CHAIN] Stage 3 JSON parse failed, retrying")
                        continue
                    return None

                # P34-EXT: Validate schema — each item must have idx + 3 float scores
                valid = True
                for item in parsed:
                    if not isinstance(item, dict):
                        valid = False
                        break
                    idx = item.get("idx")
                    if not isinstance(idx, (int, float)):
                        valid = False
                        break
                    for key in ("phase_fit", "evidence_support", "novelty"):
                        val = item.get(key)
                        if val is None or not isinstance(val, (int, float)):
                            valid = False
                            break
                        # Clamp to [0, 1]
                        item[key] = max(0.0, min(1.0, float(val)))
                    if not valid:
                        break

                if not valid:
                    if _attempt == 0:
                        logger.debug("[MICRO-CHAIN] Stage 3 schema validation failed, retrying")
                        continue
                    return None

                for item in parsed:
                    idx = int(item.get("idx", -1))
                    if 0 <= idx < len(candidates):
                        c = candidates[idx]
                        c.phase_fit = float(item.get("phase_fit", 0.5))
                        c.evidence_support = float(item.get("evidence_support", 0.5))
                        c.novelty = float(item.get("novelty", 0.5))
                        # Weighted score
                        c.score = (
                            0.40 * c.phase_fit
                            + 0.35 * c.evidence_support
                            + 0.25 * c.novelty
                        )
                return candidates
            except Exception as e:
                logger.debug(f"[MICRO-CHAIN] Stage 3 scoring failed (attempt {_attempt}): {e}")
                if _attempt == 0:
                    continue
                return None
        return None

    def _heuristic_select(
        self,
        candidates: List[MicroChainCandidate],
        recent_commands: List[str],
    ) -> List[MicroChainCandidate]:
        """Fallback: score candidates heuristically when nano scoring fails."""
        recent_set = set(recent_commands[-10:]) if recent_commands else set()
        for c in candidates:
            c.phase_fit = 0.5
            c.evidence_support = 0.5
            c.novelty = 0.8 if c.command not in recent_set else 0.2
            c.score = 0.40 * c.phase_fit + 0.35 * c.evidence_support + 0.25 * c.novelty
        return candidates

    def _escalate_to_codex(
        self,
        phase: str,
        discovery_board: Dict[str, Any],
        candidates: List[MicroChainCandidate],
        agent_role: str,
    ) -> Optional[MicroChainCandidate]:
        """Escalate to codex when scores are too low."""
        ports = list(discovery_board.get("ports", []))[:10]
        services = list(discovery_board.get("services", []))[:10]
        existing = [c.command[:60] for c in candidates[:3]]

        prompt = (
            f"The following candidate commands scored poorly for phase={phase}, role={agent_role}.\n"
            f"Candidates: {existing}\n"
            f"Known ports: {ports}\nKnown services: {services}\n\n"
            f"Suggest ONE better command as JSON:\n"
            f'{{"command":"...", "template_name":"...", "reasoning":"..."}}\n'
            f"No markdown."
        )
        try:
            response = self._gpt.gpt_request(
                prompt=prompt,
                task_type="analysis",
                agent_id="micro_chain",
                max_tokens=200,
                model="gpt-5.2-codex",
                timeout=30,
            )
            obj = _safe_json_load(response)
            if obj and obj.get("command"):
                return MicroChainCandidate(
                    command=str(obj["command"]),
                    template_name=str(obj.get("template_name", "")),
                    reasoning=str(obj.get("reasoning", "codex_escalation")),
                    score=0.70,  # Codex gets a baseline confidence boost
                    phase_fit=0.70,
                    evidence_support=0.65,
                    novelty=0.80,
                )
            return None
        except Exception as e:
            logger.debug(f"[MICRO-CHAIN] Codex escalation failed: {e}")
            return None

    def reset_cache(self) -> None:
        """Clear cache (call at episode start)."""
        self._cache.clear()

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def model_call_mix(self) -> Dict[str, int]:
        """P36.1: Return nano/mini call counts for routing audit."""
        return {
            "nano": self._nano_calls,
            "mini": self._mini_calls,
            "total": self._nano_calls + self._mini_calls,
        }

    @property
    def nano_ratio(self) -> float:
        """P36.1: Current nano% of learning-support calls."""
        total = self._nano_calls + self._mini_calls
        return self._nano_calls / max(total, 1)
