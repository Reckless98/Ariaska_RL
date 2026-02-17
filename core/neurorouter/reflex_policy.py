#!/usr/bin/env python3
"""
core/neurorouter/reflex_policy.py — Phase 15.0: Reflex Arc (Deterministic)

Fast deterministic overrides evaluated BEFORE action selection.
Reflex can only output one of four safe actions:
  - verify: re-run a verification command to confirm a finding
  - reduce_aggression: back off to quieter approach
  - collect_evidence: gather missing evidence before proceeding
  - stop_noisy: halt a noisy/detected scan

Reflex NEVER produces exploitation commands or template actions.
Reflex is stateless — pure function of current context.

Feature-flag gated: FF_REFLEX_POLICY.

Author: Phase 15.0 Contract
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ariaska.neurorouter.reflex")


# ── Enums ───────────────────────────────────────────────────────────────────

class ReflexAction(str, Enum):
    """The four allowed reflex actions."""
    VERIFY = "verify"
    REDUCE_AGGRESSION = "reduce_aggression"
    COLLECT_EVIDENCE = "collect_evidence"
    STOP_NOISY = "stop_noisy"


# ── Schemas ─────────────────────────────────────────────────────────────────

@dataclass
class ReflexOverride:
    """A reflex override decision."""
    triggered: bool = False
    action: Optional[ReflexAction] = None
    reason: str = ""
    confidence: float = 0.0
    source_rule: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.action is not None:
            d["action"] = self.action.value
        return d


@dataclass
class ReflexContext:
    """Input context for reflex evaluation."""
    detection_risk: float = 0.0
    blue_team_alert: float = 0.0
    last_command_failed: bool = False
    last_command_noisy: bool = False
    confidence_min: float = 1.0
    unverified_findings: int = 0
    evidence_gaps: int = 0
    steps_since_discovery: int = 0
    aggression_level: float = 0.3
    ne_level: float = 0.3  # norepinephrine (uncertainty)
    repeated_failures: int = 0
    phase: str = "RECON"
    brute_force_attempts: int = 0  # password mutation attempts this episode
    brute_force_cap: int = 10      # strict cap on brute force attempts


# ── Reflex Policy ───────────────────────────────────────────────────────────

class ReflexPolicy:
    """
    Deterministic reflex arc — evaluated before action selection.

    Rules are evaluated in priority order; first trigger wins.
    All thresholds are constants (no learning).
    """

    # Thresholds (constants)
    DETECTION_RISK_THRESHOLD = 0.7
    BLUE_ALERT_THRESHOLD = 0.8
    CONFIDENCE_LOW_THRESHOLD = 0.3
    UNVERIFIED_THRESHOLD = 3
    EVIDENCE_GAP_THRESHOLD = 2
    AGGRESSION_HIGH_THRESHOLD = 0.8
    REPEATED_FAILURE_THRESHOLD = 3
    BRUTE_FORCE_AGGRESSION_GATE = 0.7  # block brute force when aggression exceeds this

    def evaluate(self, ctx: ReflexContext) -> ReflexOverride:
        """
        Evaluate reflex rules in priority order.

        Returns ReflexOverride with triggered=True if a reflex fires,
        or triggered=False if no reflex needed.
        """
        # Rule 1: STOP_NOISY — detection risk too high or blue team alert
        if ctx.detection_risk > self.DETECTION_RISK_THRESHOLD or ctx.blue_team_alert > self.BLUE_ALERT_THRESHOLD:
            return ReflexOverride(
                triggered=True,
                action=ReflexAction.STOP_NOISY,
                reason=f"detection_risk={ctx.detection_risk:.2f}, blue_alert={ctx.blue_team_alert:.2f}",
                confidence=0.95,
                source_rule="detection_threshold",
            )

        # Rule 2: STOP_NOISY — last command was noisy
        if ctx.last_command_noisy and ctx.detection_risk > 0.5:
            return ReflexOverride(
                triggered=True,
                action=ReflexAction.STOP_NOISY,
                reason="noisy_command + elevated_risk",
                confidence=0.85,
                source_rule="noisy_command",
            )

        # Rule 3: REDUCE_AGGRESSION — aggression too high with failures
        if ctx.aggression_level > self.AGGRESSION_HIGH_THRESHOLD and ctx.repeated_failures >= 2:
            return ReflexOverride(
                triggered=True,
                action=ReflexAction.REDUCE_AGGRESSION,
                reason=f"aggression={ctx.aggression_level:.2f}, failures={ctx.repeated_failures}",
                confidence=0.80,
                source_rule="high_aggression_failure",
            )

        # Rule 4: VERIFY — unverified findings piling up
        if ctx.unverified_findings >= self.UNVERIFIED_THRESHOLD:
            return ReflexOverride(
                triggered=True,
                action=ReflexAction.VERIFY,
                reason=f"unverified_findings={ctx.unverified_findings}",
                confidence=0.75,
                source_rule="unverified_backlog",
            )

        # Rule 5: VERIFY — low confidence on recent outputs
        if ctx.confidence_min < self.CONFIDENCE_LOW_THRESHOLD and ctx.ne_level > 0.6:
            return ReflexOverride(
                triggered=True,
                action=ReflexAction.VERIFY,
                reason=f"low_confidence={ctx.confidence_min:.2f}, ne={ctx.ne_level:.2f}",
                confidence=0.70,
                source_rule="low_confidence_uncertainty",
            )

        # Rule 6: COLLECT_EVIDENCE — evidence gaps blocking progress
        if ctx.evidence_gaps >= self.EVIDENCE_GAP_THRESHOLD and ctx.steps_since_discovery > 3:
            return ReflexOverride(
                triggered=True,
                action=ReflexAction.COLLECT_EVIDENCE,
                reason=f"evidence_gaps={ctx.evidence_gaps}, stale={ctx.steps_since_discovery}",
                confidence=0.70,
                source_rule="evidence_gap",
            )

        # Rule 7: STOP_NOISY — brute force cap exceeded or gated by aggression/detection
        if ctx.brute_force_attempts >= ctx.brute_force_cap:
            return ReflexOverride(
                triggered=True,
                action=ReflexAction.STOP_NOISY,
                reason=f"brute_force_cap_hit={ctx.brute_force_attempts}/{ctx.brute_force_cap}",
                confidence=0.95,
                source_rule="brute_force_cap",
            )
        if ctx.brute_force_attempts > 0 and (
            ctx.aggression_level > self.BRUTE_FORCE_AGGRESSION_GATE
            or ctx.detection_risk > 0.6
        ):
            return ReflexOverride(
                triggered=True,
                action=ReflexAction.REDUCE_AGGRESSION,
                reason=f"brute_force_gated: aggression={ctx.aggression_level:.2f}, det={ctx.detection_risk:.2f}",
                confidence=0.85,
                source_rule="brute_force_aggression_gate",
            )

        # Rule 8: REDUCE_AGGRESSION — repeated failures
        if ctx.repeated_failures >= self.REPEATED_FAILURE_THRESHOLD:
            return ReflexOverride(
                triggered=True,
                action=ReflexAction.REDUCE_AGGRESSION,
                reason=f"repeated_failures={ctx.repeated_failures}",
                confidence=0.65,
                source_rule="repeated_failure",
            )

        # No reflex triggered
        return ReflexOverride(triggered=False)

    def get_reflex_command(self, override: ReflexOverride, phase: str = "RECON") -> Optional[str]:
        """
        Map a reflex override to a safe command suggestion.

        Returns a safe command string or None if no override.
        These are suggestions — the caller decides final execution.
        """
        if not override.triggered or override.action is None:
            return None

        if override.action == ReflexAction.STOP_NOISY:
            return "wait --seconds 5 --reason stealth_cooldown"
        elif override.action == ReflexAction.REDUCE_AGGRESSION:
            return "nmap -sS -T2 --top-ports 20 {target}"
        elif override.action == ReflexAction.VERIFY:
            return "nmap -sV -p- --reason {target}"
        elif override.action == ReflexAction.COLLECT_EVIDENCE:
            return "nmap -sC -sV --top-ports 100 {target}"

        return None
