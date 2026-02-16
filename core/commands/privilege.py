#!/usr/bin/env python3
"""
core/commands/privilege.py — Phase 10.1: Privilege-Aware Command Gating

Provides PrivilegeLevel enum, privilege filtering logic, and telemetry
counters for sudo/privilege gating in the decision pipeline.

Usage:
    from core.commands.privilege import PrivilegeLevel, filter_by_privilege

Design:
    - CommandTemplate gains: requires_privilege, privilege_reason, safety_tags,
      verify_template
    - Decision pipeline filters candidates that need privilege the agent
      hasn't earned yet
    - Sudo is NEVER automatic — requires explicit discovery ("sudo -l" evidence)
      or has_root state
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("ariaska.privilege")


class PrivilegeLevel(Enum):
    """Required privilege level for a command template."""
    NONE = "none"          # No special privilege needed
    SUDO = "sudo"          # Needs sudo access (discovered via sudo -l or similar)
    ROOT = "root"          # Needs root shell


# Safety tag constants
SAFETY_REQUIRES_ROOT = "requires_root"
SAFETY_NOISY = "noisy"
SAFETY_DESTRUCTIVE = "destructive"
SAFETY_NETWORK_DISRUPTIVE = "network_disruptive"
SAFETY_MODIFIES_TARGET = "modifies_target"


@dataclass
class PrivilegeFilterResult:
    """Result of privilege filtering on a candidate list."""
    allowed: List[Any] = field(default_factory=list)
    filtered: List[Any] = field(default_factory=list)
    filter_reasons: Dict[str, str] = field(default_factory=dict)  # template_name -> reason


@dataclass
class PrivilegeTelemetry:
    """Telemetry counters for privilege gating within an episode."""
    candidates_filtered_no_privilege: int = 0
    candidates_filtered_no_sudo: int = 0
    candidates_filtered_phase_gate: int = 0
    sudo_attempted: int = 0
    sudo_allowed: int = 0
    sudo_denied: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "filtered_no_privilege": self.candidates_filtered_no_privilege,
            "filtered_no_sudo": self.candidates_filtered_no_sudo,
            "filtered_phase_gate": self.candidates_filtered_phase_gate,
            "sudo_attempted": self.sudo_attempted,
            "sudo_allowed": self.sudo_allowed,
            "sudo_denied": self.sudo_denied,
        }

    def merge(self, other: "PrivilegeTelemetry") -> None:
        """Merge another telemetry record into this one."""
        self.candidates_filtered_no_privilege += other.candidates_filtered_no_privilege
        self.candidates_filtered_no_sudo += other.candidates_filtered_no_sudo
        self.candidates_filtered_phase_gate += other.candidates_filtered_phase_gate
        self.sudo_attempted += other.sudo_attempted
        self.sudo_allowed += other.sudo_allowed
        self.sudo_denied += other.sudo_denied


def _has_privilege(state: Dict[str, Any], required: PrivilegeLevel) -> bool:
    """Check if current state satisfies the required privilege level."""
    if required == PrivilegeLevel.NONE:
        return True

    state_flags = state.get("state_flags", {})
    if isinstance(state_flags, set):
        # Some code passes state_flags as a set of names
        has_root = "root_shell_obtained" in state_flags
        has_shell = "shell_obtained" in state_flags
        has_sudo = "sudo_rights_discovered" in state_flags
    else:
        has_root = state_flags.get("root_shell_obtained", False)
        has_shell = state_flags.get("shell_obtained", False)
        has_sudo = state_flags.get("sudo_rights_discovered", False)

    # Also check top-level state keys (CyberEnvironment)
    priv_level = state.get("privilege_level", "none")
    if priv_level == "root":
        has_root = True
    if priv_level in ("user", "root"):
        has_shell = True

    if required == PrivilegeLevel.ROOT:
        return has_root
    if required == PrivilegeLevel.SUDO:
        return has_root or has_sudo

    return False


def _phase_allows_privilege(state: Dict[str, Any], required: PrivilegeLevel) -> bool:
    """Check if current phase allows privilege-requiring commands.

    Privilege commands (SUDO/ROOT) are allowed only when:
    - Phase is PRIVILEGE_ESCALATION or later, OR
    - Shell has been obtained (any phase), OR
    - Sudo rights discovered (any phase)
    """
    if required == PrivilegeLevel.NONE:
        return True

    phase = state.get("phase", "recon").upper()
    allowed_phases = {
        "PRIVILEGE_ESCALATION", "PRIVESC",
        "LATERAL_MOVEMENT",
        "POST_EXPLOITATION",
        "EXFILTRATION",
        "CLOSEOUT",
    }
    if phase in allowed_phases:
        return True

    # Also allow if shell obtained (even in earlier phases)
    state_flags = state.get("state_flags", {})
    if isinstance(state_flags, set):
        return "shell_obtained" in state_flags or "sudo_rights_discovered" in state_flags
    return (
        state_flags.get("shell_obtained", False)
        or state_flags.get("sudo_rights_discovered", False)
    )


def filter_by_privilege(
    candidates: List[Any],
    state: Dict[str, Any],
    telemetry: Optional[PrivilegeTelemetry] = None,
) -> PrivilegeFilterResult:
    """Filter command templates by privilege requirements.

    Args:
        candidates: List of CommandTemplate objects
        state: Current environment state dict
        telemetry: Optional telemetry counter to update

    Returns:
        PrivilegeFilterResult with allowed and filtered lists
    """
    result = PrivilegeFilterResult()
    tel = telemetry or PrivilegeTelemetry()

    for cmd in candidates:
        required = getattr(cmd, "requires_privilege", PrivilegeLevel.NONE)
        if isinstance(required, str):
            try:
                required = PrivilegeLevel(required)
            except ValueError:
                required = PrivilegeLevel.NONE

        if required == PrivilegeLevel.NONE:
            result.allowed.append(cmd)
            continue

        # Phase gate check
        if not _phase_allows_privilege(state, required):
            result.filtered.append(cmd)
            reason = f"phase_gate:{state.get('phase', 'unknown')}"
            result.filter_reasons[cmd.name] = reason
            tel.candidates_filtered_phase_gate += 1
            logger.debug(
                "Privilege filter: %s blocked (phase gate, need %s, phase=%s)",
                cmd.name, required.value, state.get("phase"),
            )
            continue

        # Privilege check
        if not _has_privilege(state, required):
            result.filtered.append(cmd)
            reason = f"no_privilege:{required.value}"
            result.filter_reasons[cmd.name] = reason
            if required == PrivilegeLevel.SUDO:
                tel.candidates_filtered_no_sudo += 1
            else:
                tel.candidates_filtered_no_privilege += 1
            logger.debug(
                "Privilege filter: %s blocked (need %s, not available)",
                cmd.name, required.value,
            )
            continue

        result.allowed.append(cmd)

    return result


def check_sudo_allowed(state: Dict[str, Any], ff_allow_sudo: bool = False) -> bool:
    """Check if sudo execution is currently allowed.

    Sudo is allowed when:
    1. Feature flag FF_ALLOW_SUDO is True, AND
    2. State shows sudo rights discovered OR root shell obtained

    Args:
        state: Current environment state
        ff_allow_sudo: Value of FF_ALLOW_SUDO feature flag

    Returns:
        True if sudo can be used
    """
    if not ff_allow_sudo:
        return False

    state_flags = state.get("state_flags", {})
    if isinstance(state_flags, set):
        return (
            "sudo_rights_discovered" in state_flags
            or "root_shell_obtained" in state_flags
        )
    return (
        state_flags.get("sudo_rights_discovered", False)
        or state_flags.get("root_shell_obtained", False)
    )
