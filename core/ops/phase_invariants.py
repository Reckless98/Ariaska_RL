"""
core/ops/phase_invariants.py — Phase 38.3: Phase DAG Invariant Checker

Enforces phase transitions via a precondition DAG rather than strict linear
ordering. LATERAL_MOVEMENT and POST_EXPLOITATION are optional branches that
only activate when multi-host or persistence scenarios are detected.

Rules:
  - Each phase has mandatory preconditions (discovery gates).
  - Transitions must follow the DAG (no backward to *dependencies*).
  - Optional phases (LATERAL_MOVEMENT, POST_EXPLOITATION) are skippable
    when their triggers are absent (single-host, no persistence needed).
  - Shell validation gates EXPLOITATION→PRIVILEGE_ESCALATION.
  - Violations are logged and optionally enforced (demotion).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

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

# ── Phase DAG (precondition-based, replaces strict linear chain) ──────────────
# Keys are phases; values are the set of phases that MUST have been visited
# (or whose preconditions are met) before entry is allowed.
# LATERAL_MOVEMENT and POST_EXPLOITATION are optional branches — the main
# highway is RECON → ENUM → EXPLOIT → PRIVESC → EXFIL → CLOSEOUT.
PHASE_DAG: Dict[str, FrozenSet[str]] = {
    "RECON":                  frozenset(),
    "ENUMERATION":            frozenset({"RECON"}),
    "EXPLOITATION":           frozenset({"ENUMERATION"}),
    "PRIVILEGE_ESCALATION":   frozenset({"EXPLOITATION"}),
    "LATERAL_MOVEMENT":       frozenset({"EXPLOITATION"}),       # optional branch
    "POST_EXPLOITATION":      frozenset({"PRIVILEGE_ESCALATION"}),  # optional branch
    "EXFILTRATION":           frozenset({"PRIVILEGE_ESCALATION"}),
    "CLOSEOUT":               frozenset({"EXFILTRATION"}),
}

# Phases that can be skipped if their activation conditions are not met.
# Single-host targets skip LATERAL_MOVEMENT; non-persistence scenarios skip
# POST_EXPLOITATION. The agent can jump directly past these.
OPTIONAL_PHASES: FrozenSet[str] = frozenset({
    "LATERAL_MOVEMENT",
    "POST_EXPLOITATION",
})


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

    def __init__(self, strict: bool = True, use_dag: bool = True) -> None:
        """
        Args:
            strict: If True and use_dag=False, enforce sequential order.
                    Ignored when use_dag=True (DAG mode).
            use_dag: If True (default), use the precondition DAG which allows
                     skipping optional phases (LATERAL_MOVEMENT, POST_EXPLOITATION)
                     when their triggers are absent.
        """
        self._strict = strict
        self._use_dag = use_dag
        self._transition_log: List[Dict[str, Any]] = []
        self._visited_phases: Set[str] = {"RECON"}  # Track visited phases for DAG
        logger.debug(
            "PhaseInvariantChecker initialised (strict=%s, use_dag=%s)",
            strict, use_dag,
        )

    def validate_transition(
        self,
        current_phase: str,
        requested_phase: str,
        state_flags: Optional[Dict[str, bool]] = None,
        discovery_board: Optional[Dict[str, Any]] = None,
    ) -> PhaseValidationResult:
        """
        Validate a phase transition using the DAG or strict linear mode.

        DAG mode (default): Allows skipping optional phases when preconditions
        for the target phase are met. The main highway is:
            RECON → ENUM → EXPLOIT → PRIVESC → EXFIL → CLOSEOUT

        LATERAL_MOVEMENT is only required when multiple hosts are discovered.
        POST_EXPLOITATION is only required when persistence is needed.

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

        # 1. Check phase ordering (DAG-aware)
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

            # Reject backward transitions (still enforced in DAG mode)
            if req_idx < cur_idx:
                violations.append(
                    f"Backward transition {current}→{requested} "
                    f"(index {cur_idx}→{req_idx})"
                )

            if self._use_dag:
                # DAG mode: check that all *required* dependencies are satisfied.
                # Optional phases (LATERAL_MOVEMENT, POST_EXPLOITATION) are
                # implicitly satisfied when skipped.
                dag_deps = PHASE_DAG.get(requested, frozenset())
                for dep in dag_deps:
                    if dep in OPTIONAL_PHASES:
                        continue  # Optional deps don't block
                    dep_precond = PHASE_PRECONDITIONS.get(dep, {})
                    dep_flags = dep_precond.get("required_flags", {})
                    dep_satisfied = all(
                        state_flags.get(f, False) == v
                        for f, v in dep_flags.items()
                    )
                    if not dep_satisfied and dep not in self._visited_phases:
                        violations.append(
                            f"DAG dependency not met: {requested} requires "
                            f"{dep} (preconditions not satisfied)"
                        )
            else:
                # Legacy strict mode: reject skipping more than 1 phase
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

        # 3. Determine recommended phase (DAG-aware)
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
        else:
            # Valid transition — record visited phase for DAG tracking
            self._visited_phases.add(requested)

        return result

    def _find_highest_valid_phase(
        self,
        state_flags: Dict[str, bool],
        discovery_board: Dict[str, Any],
    ) -> str:
        """Find the highest phase whose preconditions are met.

        In DAG mode, optional phases (LATERAL_MOVEMENT, POST_EXPLOITATION)
        are skipped if their activation signals are absent, allowing the
        agent to advance directly along the main highway.
        """
        highest = PHASE_ORDER[0]
        for phase in PHASE_ORDER:
            # DAG mode: skip optional phases whose triggers are absent
            if self._use_dag and phase in OPTIONAL_PHASES:
                if phase == "LATERAL_MOVEMENT":
                    # Only relevant when multiple hosts discovered
                    net_state = discovery_board.get("network_topology", {})
                    hosts = 0
                    if isinstance(net_state, dict):
                        hosts = net_state.get("hosts_found", 0)
                    if hosts < 2:
                        continue  # Skip — single host, no lateral needed
                elif phase == "POST_EXPLOITATION":
                    # Only relevant when persistence is explicitly required
                    # (e.g., long-running engagements, not CTF)
                    if phase not in self._visited_phases:
                        continue  # Skip unless explicitly entered

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
                # In DAG mode, optional phases don't block progression
                if self._use_dag and phase in OPTIONAL_PHASES:
                    continue
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
        """Reset checker state for new episode."""
        self._transition_log.clear()
        self._visited_phases = {"RECON"}
