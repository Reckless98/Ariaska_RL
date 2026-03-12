#!/usr/bin/env python3
"""
core/state/coherence_chain.py — Phase 35: 4-Step Nano Coherence Micro-Chain

Runs BEFORE PhaseGuide/Teaching output each step:
  Step A — CLASSIFY (nano):  Phase guess + evidence summary
  Step B — CONTRADICTION_CHECK (nano):  Detect state-vs-guidance desync
  Step C — SUMMARIZE (nano):  Compact "state postcard" for UI/logs
  Step D — SCORE (nano):  Coherence/quality metrics

Model Routing:
  - nano (80%): Steps A, B, C, D — fast classification/scoring
  - mini (20%): Only if contradiction severity=high → re-summarize
  - codex: Never called here (reserved for PhaseGuide escalation)

All LLM calls via GPTManager — no direct openai imports.

Author: Phase 35
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.gpt_manager import GPTManager
    from core.state.canonical_state import CanonicalState

logger = logging.getLogger("ariaska.state.coherence_chain")

__all__ = [
    "CoherenceChainResult",
    "ClassifyResult",
    "ContradictionResult",
    "SummarizeResult",
    "ScoreResult",
    "CoherenceChain",
]


# ── Result Dataclasses ─────────────────────────────────────────────────────

@dataclass
class ClassifyResult:
    """Step A output: phase classification from evidence."""
    phase_guess: str = "RECON"
    phase_confidence: float = 0.5
    key_evidence: List[str] = field(default_factory=list)
    missing_evidence: List[str] = field(default_factory=list)
    next_best_families: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase_guess": self.phase_guess,
            "phase_confidence": round(self.phase_confidence, 3),
            "key_evidence": self.key_evidence,
            "missing_evidence": self.missing_evidence,
            "next_best_families": self.next_best_families,
        }


@dataclass
class ContradictionResult:
    """Step B output: contradiction detection."""
    contradiction_detected: bool = False
    contradictions: List[str] = field(default_factory=list)
    severity: str = "low"  # low | med | high
    fix_hint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contradiction_detected": self.contradiction_detected,
            "contradictions": self.contradictions,
            "severity": self.severity,
            "fix_hint": self.fix_hint,
        }


@dataclass
class SummarizeResult:
    """Step C output: compact state postcard."""
    postcard: str = ""
    evidence_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "postcard": self.postcard,
            "evidence_counts": self.evidence_counts,
        }


@dataclass
class ScoreResult:
    """Step D output: coherence quality metrics."""
    coherence_score: float = 0.5
    novelty_score: float = 0.5
    repeat_risk: float = 0.5
    confidence_calibration: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coherence_score": round(self.coherence_score, 3),
            "novelty_score": round(self.novelty_score, 3),
            "repeat_risk": round(self.repeat_risk, 3),
            "confidence_calibration": round(self.confidence_calibration, 3),
        }


@dataclass
class CoherenceChainResult:
    """Combined output of the 4-step coherence chain."""
    classify: ClassifyResult = field(default_factory=ClassifyResult)
    contradiction: ContradictionResult = field(default_factory=ContradictionResult)
    summary: SummarizeResult = field(default_factory=SummarizeResult)
    score: ScoreResult = field(default_factory=ScoreResult)
    nano_tokens: int = 0
    mini_tokens: int = 0
    elapsed_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "classify": self.classify.to_dict(),
            "contradiction": self.contradiction.to_dict(),
            "summary": self.summary.to_dict(),
            "score": self.score.to_dict(),
            "nano_tokens": self.nano_tokens,
            "mini_tokens": self.mini_tokens,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


# ── Phase / Evidence Mapping ──────────────────────────────────────────────

_PHASE_ORDER = [
    "RECON", "ENUMERATION", "EXPLOITATION", "PRIVILEGE_ESCALATION",
    "LATERAL_MOVEMENT", "POST_EXPLOITATION", "EXFILTRATION",
]

_PHASE_EVIDENCE_REQUIREMENTS = {
    "RECON": {"ports": 0, "services": 0},
    "ENUMERATION": {"ports": 2, "services": 1},
    "EXPLOITATION": {"ports": 2, "services": 2},
    "PRIVILEGE_ESCALATION": {"shells": 1},
    "LATERAL_MOVEMENT": {"shells": 1},
    "POST_EXPLOITATION": {"shells": 1},
    "EXFILTRATION": {"shells": 1},
}

_NEXT_FAMILIES = {
    "RECON": ["nmap", "web", "ssh"],
    "ENUMERATION": ["web", "nmap", "ssh", "file"],
    "EXPLOITATION": ["ssh", "web", "privesc", "misc"],
    "PRIVILEGE_ESCALATION": ["privesc", "file", "misc"],
    "LATERAL_MOVEMENT": ["ssh", "misc"],
    "POST_EXPLOITATION": ["file", "misc", "privesc"],
    "EXFILTRATION": ["file", "misc"],
}


# ── Heuristic Engine (Zero-LLM fallback) ──────────────────────────────────

def heuristic_classify(state: "CanonicalState") -> ClassifyResult:
    """
    Evidence-driven phase classification — ZERO LLM calls.
    Used as fallback when nano is unavailable or for offline testing.
    """
    counts = state.evidence_counts()
    key_evidence: List[str] = []
    missing_evidence: List[str] = []

    if counts["ports"] > 0:
        key_evidence.append(f"{counts['ports']} ports")
    else:
        missing_evidence.append("no ports discovered")
    if counts["services"] > 0:
        key_evidence.append(f"{counts['services']} services")
    else:
        missing_evidence.append("no services identified")
    if counts["creds"] > 0:
        key_evidence.append(f"{counts['creds']} credentials")
    if counts["shells"] > 0:
        key_evidence.append(f"{counts['shells']} shells")
    if counts["vulns"] > 0:
        key_evidence.append(f"{counts['vulns']} vulns")
    if counts["paths"] > 0:
        key_evidence.append(f"{counts['paths']} web paths")

    # Determine best phase from evidence
    if counts["shells"] > 0:
        if counts["flags"] > 0:
            phase_guess = "EXFILTRATION"
            confidence = 0.90
        else:
            phase_guess = "PRIVILEGE_ESCALATION"
            confidence = 0.80
    elif counts["creds"] > 0 or counts["vulns"] > 0:
        phase_guess = "EXPLOITATION"
        confidence = 0.75
    elif counts["services"] >= 2 or counts["paths"] > 0:
        phase_guess = "ENUMERATION"
        confidence = 0.70
    elif counts["ports"] >= 2:
        phase_guess = "ENUMERATION"
        confidence = 0.60
    else:
        phase_guess = "RECON"
        confidence = 0.55

    return ClassifyResult(
        phase_guess=phase_guess,
        phase_confidence=confidence,
        key_evidence=key_evidence,
        missing_evidence=missing_evidence,
        next_best_families=_NEXT_FAMILIES.get(phase_guess, ["nmap"]),
    )


def heuristic_contradiction_check(
    state: "CanonicalState",
    proposed_phase: Optional[str] = None,
    guidance_claims: Optional[Dict[str, Any]] = None,
) -> ContradictionResult:
    """
    Deterministic contradiction detection — ZERO LLM calls.
    Catches the most common hallucinations.
    """
    contradictions: List[str] = []
    counts = state.evidence_counts()
    guidance = guidance_claims or {}

    # Type 1: Ports exist but guidance says none
    if counts["ports"] > 0 and guidance.get("claims_no_ports", False):
        contradictions.append(
            f"DESYNC: {counts['ports']} ports in state but guidance claims none"
        )

    # Type 2: Shell exists but guidance says no foothold
    if counts["shells"] > 0 and guidance.get("claims_no_foothold", False):
        contradictions.append(
            f"DESYNC: {counts['shells']} shells in state but guidance claims no foothold"
        )

    # Type 3: Web paths exist but guidance says no web enumeration
    if counts["paths"] > 0 and guidance.get("claims_no_web", False):
        contradictions.append(
            f"DESYNC: {counts['paths']} web paths but guidance claims no web enum done"
        )

    # Type 4: Creds exist but guidance says no creds
    if counts["creds"] > 0 and guidance.get("claims_no_creds", False):
        contradictions.append(
            f"DESYNC: {counts['creds']} credentials but guidance claims none found"
        )

    # Type 5: Phase impossible given evidence
    check_phase = proposed_phase or state.current_phase
    if check_phase:
        phase_upper = check_phase.upper()
        if phase_upper == "PRIVILEGE_ESCALATION" and counts["shells"] == 0:
            contradictions.append(
                "DESYNC: PRIVESC phase without any shell"
            )
        if phase_upper == "EXFILTRATION" and counts["shells"] == 0:
            contradictions.append(
                "DESYNC: EXFILTRATION phase without any shell"
            )
        if phase_upper == "RECON" and counts["ports"] >= 5 and counts["services"] >= 3:
            contradictions.append(
                f"DESYNC: Still in RECON but have {counts['ports']} ports + "
                f"{counts['services']} services — should advance"
            )

    # Type 6: Phase behind evidence (stagnation indicator)
    if state.stagnation_steps >= 8 and counts["ports"] >= 2 and counts["services"] >= 1:
        if state.current_phase == "RECON":
            contradictions.append(
                f"DESYNC: {state.stagnation_steps} stagnation steps in RECON "
                f"with {counts['ports']} ports — should advance"
            )

    if not contradictions:
        return ContradictionResult()

    severity = "low"
    if len(contradictions) >= 3:
        severity = "high"
    elif len(contradictions) >= 2:
        severity = "med"
    elif any("shell" in c.lower() or "PRIVESC" in c for c in contradictions):
        severity = "high"

    fix_hint = contradictions[0].replace("DESYNC: ", "Fix: ")

    return ContradictionResult(
        contradiction_detected=True,
        contradictions=contradictions,
        severity=severity,
        fix_hint=fix_hint,
    )


def heuristic_summarize(state: "CanonicalState") -> SummarizeResult:
    """Build compact state postcard — ZERO LLM calls."""
    counts = state.evidence_counts()
    parts = [f"Phase: {state.current_phase}"]
    if counts["ports"]:
        parts.append(f"{counts['ports']} ports open")
    if counts["services"]:
        parts.append(f"{counts['services']} services ID'd")
    if counts["creds"]:
        parts.append(f"{counts['creds']} creds found")
    if counts["shells"]:
        parts.append(f"{counts['shells']} shells active")
    if counts["vulns"]:
        parts.append(f"{counts['vulns']} vulns known")
    if counts["paths"]:
        parts.append(f"{counts['paths']} web paths")
    if state.stagnation_steps > 0:
        parts.append(f"stagnation={state.stagnation_steps}")

    return SummarizeResult(
        postcard=". ".join(parts) + ".",
        evidence_counts=counts,
    )


def heuristic_score(
    state: "CanonicalState",
    recent_commands: Optional[List[str]] = None,
) -> ScoreResult:
    """Compute coherence quality metrics — ZERO LLM calls."""
    counts = state.evidence_counts()
    recent = recent_commands or state.recent_commands

    # Coherence: does the phase match the evidence?
    classify = heuristic_classify(state)
    phase_match = 1.0 if classify.phase_guess == state.current_phase else 0.4
    evidence_density = min(1.0, (counts["ports"] + counts["services"] * 2 + counts["creds"] * 3) / 15)
    coherence = 0.5 * phase_match + 0.5 * evidence_density

    # Novelty: likelihood of new discoveries
    novelty = max(0.1, 1.0 - state.stagnation_steps * 0.1)

    # Repeat risk
    if len(recent) >= 2:
        unique_ratio = len(set(recent[-10:])) / max(1, len(recent[-10:]))
        repeat_risk = 1.0 - unique_ratio
    else:
        repeat_risk = 0.2

    # Confidence calibration: confidence matches evidence density
    if state.phase_confidence > 0:
        cal_delta = abs(state.phase_confidence - evidence_density)
        confidence_cal = max(0.0, 1.0 - cal_delta)
    else:
        confidence_cal = 0.5

    return ScoreResult(
        coherence_score=round(min(1.0, coherence), 3),
        novelty_score=round(min(1.0, novelty), 3),
        repeat_risk=round(min(1.0, repeat_risk), 3),
        confidence_calibration=round(min(1.0, confidence_cal), 3),
    )


# ── CoherenceChain Class ──────────────────────────────────────────────────

class CoherenceChain:
    """
    4-step nano coherence chain. Runs before PhaseGuide each step.

    Model split: nano 80% / mini 20% (mini only on high contradictions).
    All calls via GPTManager.
    """

    def __init__(self, gpt_manager: Optional["GPTManager"] = None) -> None:
        self._gpt = gpt_manager
        self._total_nano_tokens = 0
        self._total_mini_tokens = 0
        self._call_count = 0

    def run(
        self,
        state: "CanonicalState",
        proposed_phase: Optional[str] = None,
        guidance_claims: Optional[Dict[str, Any]] = None,
        use_llm: bool = True,
    ) -> CoherenceChainResult:
        """
        Execute the 4-step coherence chain.

        Args:
            state: Current canonical state snapshot
            proposed_phase: Phase proposed by PhaseGuide (for contradiction check)
            guidance_claims: Claims from last guidance (for contradiction check)
            use_llm: If False, use only heuristic (zero API calls)

        Returns:
            CoherenceChainResult with all 4 step outputs
        """
        t0 = time.time()
        self._call_count += 1

        # Step A: Classify
        classify = self._step_a_classify(state, use_llm)

        # Step B: Contradiction check
        contradiction = self._step_b_contradiction(
            state, proposed_phase, guidance_claims
        )

        # Step C: Summarize
        # Use mini if contradiction severity is high (20% budget)
        use_mini_summary = (
            contradiction.contradiction_detected
            and contradiction.severity == "high"
            and use_llm
        )
        summary = self._step_c_summarize(state, use_mini=use_mini_summary)

        # Step D: Score
        score = self._step_d_score(state)

        elapsed_ms = (time.time() - t0) * 1000

        return CoherenceChainResult(
            classify=classify,
            contradiction=contradiction,
            summary=summary,
            score=score,
            nano_tokens=self._total_nano_tokens,
            mini_tokens=self._total_mini_tokens,
            elapsed_ms=elapsed_ms,
        )

    def _step_a_classify(
        self,
        state: "CanonicalState",
        use_llm: bool,
    ) -> ClassifyResult:
        """Step A: Classify phase from evidence. Always heuristic-first,
        optionally verified by nano."""
        result = heuristic_classify(state)

        if not use_llm or self._gpt is None:
            return result

        # Nano verification — only if budget allows and > 5 steps
        if state.step_id < 3 or not self._gpt.can_make_request():
            return result

        # Use nano to verify heuristic — 80% of the time
        try:
            prompt = (
                f"Verify phase classification. State: {state.compact_summary()}\n"
                f"Heuristic says: {result.phase_guess} (conf={result.phase_confidence:.2f})\n"
                f"Reply JSON: {{\"phase_guess\":\"...\",\"confidence\":0.0-1.0,"
                f"\"agree\":true/false}}\n"
                f"No markdown."
            )
            resp = self._gpt.gpt_request(
                prompt=prompt,
                task_type="classification",
                agent_id="coherence_chain",
                max_tokens=60,
                model="local-llm",
            )
            self._total_nano_tokens += 60

            # Parse only if valid
            from core.llm.micro_chain import _safe_json_load
            obj = _safe_json_load(resp)
            if obj and isinstance(obj.get("agree"), bool):
                if not obj["agree"] and obj.get("phase_guess"):
                    # Nano disagrees — take its suggestion with lower confidence
                    result.phase_guess = str(obj["phase_guess"]).upper()
                    result.phase_confidence = float(obj.get("confidence", 0.5))
        except Exception as e:
            logger.debug(f"[COHERENCE-A] Nano verify failed: {e}")

        return result

    def _step_b_contradiction(
        self,
        state: "CanonicalState",
        proposed_phase: Optional[str],
        guidance_claims: Optional[Dict[str, Any]],
    ) -> ContradictionResult:
        """Step B: Contradiction check. Always deterministic — no LLM needed."""
        return heuristic_contradiction_check(state, proposed_phase, guidance_claims)

    def _step_c_summarize(
        self,
        state: "CanonicalState",
        use_mini: bool = False,
    ) -> SummarizeResult:
        """Step C: Build postcard. Heuristic by default, mini for high contradictions."""
        result = heuristic_summarize(state)

        if use_mini and self._gpt is not None and self._gpt.can_make_request():
            try:
                prompt = (
                    f"Write a 1-sentence penetration test status summary.\n"
                    f"State: {state.compact_summary()}\n"
                    f"Key ports: {state.ports[:10]}\n"
                    f"Services: {state.services[:10]}\n"
                    f"Reply with ONLY the sentence, no JSON."
                )
                resp = self._gpt.gpt_request(
                    prompt=prompt,
                    task_type="playbook",
                    agent_id="coherence_chain",
                    max_tokens=80,
                    model="local-llm",
                )
                self._total_mini_tokens += 80
                if resp and len(resp.strip()) > 10:
                    result.postcard = resp.strip()[:200]
            except Exception as e:
                logger.debug(f"[COHERENCE-C] Mini summarize failed: {e}")

        return result

    def _step_d_score(self, state: "CanonicalState") -> ScoreResult:
        """Step D: Coherence scoring. Always deterministic — no LLM needed."""
        return heuristic_score(state)

    def reset(self) -> None:
        """Reset counters at episode start."""
        self._total_nano_tokens = 0
        self._total_mini_tokens = 0
        self._call_count = 0

    @property
    def call_count(self) -> int:
        return self._call_count
