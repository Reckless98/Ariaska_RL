"""
core/ops/phase_invariants.py — Phase 38.2: Phase Hardening Invariant Checker

Enforces strict phase ordering and transition requirements.
Prevents phase skipping, validates preconditions for each phase,
and provides rollback recommendations.

Rules:
  - Phase transitions must follow the kill chain order.
  - Each phase has mandatory preconditions (discovery gates).
  - Violations are logged and optionally enforced (demotion).
  - Shell validation gates EXPLOITATION→PRIVILEGE_ESCALATION.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("ariaska.ops.phase_invariants")

# ── Phase order (canonical kill chain) ────────────────────────────────────────

PHASE_ORDER: List[str] = [
    "RECON",
    "ENUMERATION",
    "EXPLOITATION",
    "PRIVILEGE_ESCALATION",
    "LATERAL_MOVEMENT",
    "POST_EXPLOITATION",
    "EXFILTRATION",
    "CLOSEOUT",
]

_PHASE_INDEX: Dict[str, int] = {p: i for i, p in enumerate(PHASE_ORDER)}


# ── Phase preconditions ──────────────────────────────────────────────────────

# Each phase requires these state_flags to be True before entry.
# The invariant checker validates these before allowing transitions.
PHASE_PRECONDITIONS: Dict[str, Dict[str, Any]] = {
    "RECON": {},  # No preconditions
    "ENUMERATION": {
        "required_flags": {"ports_discovered": True},
        "min_discoveries": {"ports": 1},
    },
    "EXPLOITATION": {
        "required_flags": {"services_enumerated": True},
        "min_discoveries": {"services": 1},
    },
    "PRIVILEGE_ESCALATION": {
        "required_flags": {"shell_obtained": True},
        "min_discoveries": {"shells": 1},
    },
    "LATERAL_MOVEMENT": {
        "required_flags": {"shell_obtained": True},
    },
    "POST_EXPLOITATION": {
        "required_flags": {"shell_obtained": True},
    },
    "EXFILTRATION": {
        "required_flags": {"shell_obtained": True},
    },
    "CLOSEOUT": {},
}


@dataclass
class PhaseValidationResult:
    """Result of phase transition validation."""
    valid: bool
    current_phase: str
    requested_phase: str
    violations: List[str] = field(default_factory=list)
    recommended_phase: str = ""
    details: str = ""

    def __repr__(self) -> str:
        tag = "OK" if self.valid else f"REJECT({len(self.violations)} violations)"
        return f"PhaseValidation({tag}, {self.current_phase}→{self.requested_phase})"


class PhaseInvariantChecker:
    """
    Validates phase transitions against kill chain invariants.

    Usage:
        checker = PhaseInvariantChecker()
        result = checker.validate_transition(
            current_phase="RECON",
            requested_phase="EXPLOITATION",
            state_flags=state_flags,
            discovery_board=discovery_board,
        )
        if not result.valid:
            # Demote to result.recommended_phase
    """

    def __init__(self, strict: bool = True) -> None:
        """
        Args:
            strict: If True, enforce sequential order (no skipping).
                    If False, only check preconditions.
        """
        self._strict = strict
        self._transition_log: List[Dict[str, Any]] = []
        logger.debug("PhaseInvariantChecker initialised (strict=%s)", strict)

    def validate_transition(
        self,
        current_phase: str,
        requested_phase: str,
        state_flags: Optional[Dict[str, bool]] = None,
        discovery_board: Optional[Dict[str, Any]] = None,
    ) -> PhaseValidationResult:
        """
        Validate a phase transition.

        Args:
            current_phase: Current phase name (e.g. "RECON").
            requested_phase: Requested next phase.
            state_flags: Current state flags dict.
            discovery_board: Current discovery board dict.

        Returns:
            PhaseValidationResult with validity and details.
        """
        if state_flags is None:
            state_flags = {}
        if discovery_board is None:
            discovery_board = {}

        current = current_phase.upper()
        requested = requested_phase.upper()

        violations: List[str] = []

        # 1. Check phase ordering
        cur_idx = _PHASE_INDEX.get(current, -1)
        req_idx = _PHASE_INDEX.get(requested, -1)

        if req_idx < 0:
            violations.append(f"Unknown phase: {requested}")
        elif cur_idx < 0:
            violations.append(f"Unknown current phase: {current}")
        else:
            # Allow same phase (no transition)
            if req_idx == cur_idx:
                return PhaseValidationResult(
                    valid=True,
                    current_phase=current,
                    requested_phase=requested,
                    recommended_phase=current,
                )

            # Reject backward transitions
            if req_idx < cur_idx:
                violations.append(
                    f"Backward transition {current}→{requested} "
                    f"(index {cur_idx}→{req_idx})"
                )

            # Strict mode: reject skipping more than 1 phase
            if self._strict and req_idx > cur_idx + 1:
                violations.append(
                    f"Phase skip {current}→{requested} "
                    f"(skips {req_idx - cur_idx - 1} phases)"
                )

        # 2. Check preconditions for requested phase
        precond = PHASE_PRECONDITIONS.get(requested, {})

        # Required flags
        req_flags = precond.get("required_flags", {})
        for flag_name, required_val in req_flags.items():
            actual = state_flags.get(flag_name, False)
            if actual != required_val:
                violations.append(
                    f"Precondition failed: {flag_name}={actual} "
                    f"(need {required_val}) for {requested}"
                )

        # Minimum discoveries
        min_disc = precond.get("min_discoveries", {})
        for disc_type, min_count in min_disc.items():
            actual_set = discovery_board.get(disc_type, set())
            actual_count = len(actual_set) if isinstance(actual_set, (set, list)) else 0
            if actual_count < min_count:
                violations.append(
                    f"Discovery gate failed: {disc_type}={actual_count} "
                    f"(need {min_count}) for {requested}"
                )

        # 3. Determine recommended phase
        recommended = self._find_highest_valid_phase(
            state_flags, discovery_board,
        )

        result = PhaseValidationResult(
            valid=len(violations) == 0,
            current_phase=current,
            requested_phase=requested,
            violations=violations,
            recommended_phase=recommended,
            details="; ".join(violations) if violations else "valid",
        )

        self._transition_log.append({
            "from": current,
            "to": requested,
            "valid": result.valid,
            "recommended": recommended,
            "violations_count": len(violations),
        })

        if violations:
            logger.warning(
                "Phase transition %s→%s rejected: %s",
                current, requested, result.details[:120],
            )

        return result

    def _find_highest_valid_phase(
        self,
        state_flags: Dict[str, bool],
        discovery_board: Dict[str, Any],
    ) -> str:
        """Find the highest phase whose preconditions are met."""
        highest = PHASE_ORDER[0]
        for phase in PHASE_ORDER:
            precond = PHASE_PRECONDITIONS.get(phase, {})

            # Check required flags
            flags_ok = True
            for flag_name, required_val in precond.get("required_flags", {}).items():
                if state_flags.get(flag_name, False) != required_val:
                    flags_ok = False
                    break

            # Check minimum discoveries
            disc_ok = True
            for disc_type, min_count in precond.get("min_discoveries", {}).items():
                actual_set = discovery_board.get(disc_type, set())
                actual_count = len(actual_set) if isinstance(actual_set, (set, list)) else 0
                if actual_count < min_count:
                    disc_ok = False
                    break

            if flags_ok and disc_ok:
                highest = phase
            else:
                break

        return highest

    def validate_state_consistency(
        self,
        current_phase: str,
        state_flags: Dict[str, bool],
        discovery_board: Dict[str, Any],
    ) -> List[str]:
        """
        Check for state inconsistencies without a transition.

        Returns list of warning strings (empty = consistent).
        """
        warnings: List[str] = []
        phase = current_phase.upper()

        # Shell flag without actual shells
        if state_flags.get("shell_obtained") and not discovery_board.get("shells"):
            warnings.append("shell_obtained=True but no shells in discovery_board")

        # In EXPLOITATION+ without services
        exploit_phases = {"EXPLOITATION", "PRIVILEGE_ESCALATION", "LATERAL_MOVEMENT",
                         "POST_EXPLOITATION", "EXFILTRATION"}
        if phase in exploit_phases and not discovery_board.get("services"):
            warnings.append(f"In {phase} but no services discovered")

        # In PRIVESC+ without shell
        privesc_plus = {"PRIVILEGE_ESCALATION", "LATERAL_MOVEMENT",
                       "POST_EXPLOITATION", "EXFILTRATION"}
        if phase in privesc_plus and not state_flags.get("shell_obtained"):
            warnings.append(f"In {phase} without shell_obtained")

        # Credentials found but flag not set
        if discovery_board.get("credentials") and not state_flags.get("credentials_known", False):
            warnings.append("credentials in board but credentials_known=False")

        return warnings

    def get_transition_log(self) -> List[Dict[str, Any]]:
        """Return phase transition audit log."""
        return list(self._transition_log)

    def reset(self) -> None:
        """Reset checker state."""
        self._transition_log.clear()
