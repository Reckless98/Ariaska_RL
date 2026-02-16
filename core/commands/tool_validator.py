#!/usr/bin/env python3
"""
core/commands/tool_validator.py — Phase 11.0: Tool Registry + Privilege Validation

Validates that commands meet privilege requirements before execution.
Works with CommandTemplate.requires_privilege and the current agent's
privilege state to determine if a command can run.

Also provides tool availability checks (is the binary installed?)
and phase-appropriate tool filtering.

Author: Filip Volf — Phase 11.0
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from core.commands.command_registry import CommandTemplate

logger = logging.getLogger("ariaska.tool_validator")


# Common tool binary names mapped from template names
TOOL_BINARY_MAP = {
    "nmap": "nmap",
    "nmap_full": "nmap",
    "nmap_vuln": "nmap",
    "nmap_version": "nmap",
    "nmap_scripts": "nmap",
    "nikto": "nikto",
    "gobuster": "gobuster",
    "dirb": "dirb",
    "hydra": "hydra",
    "sqlmap": "sqlmap",
    "enum4linux": "enum4linux",
    "smbclient": "smbclient",
    "wfuzz": "wfuzz",
    "ffuf": "ffuf",
    "whatweb": "whatweb",
    "curl": "curl",
    "wget": "wget",
    "netcat": "nc",
    "telnet": "telnet",
    "ftp_anon": "ftp",
    "ssh_connect": "ssh",
    "mysql_nopass": "mysql",
    "psql_default": "psql",
    "msfconsole": "msfconsole",
    "searchsploit": "searchsploit",
    "john": "john",
    "hashcat": "hashcat",
    "crackmapexec": "crackmapexec",
    "responder": "responder",
    "impacket_psexec": "psexec.py",
    "impacket_smbexec": "smbexec.py",
    "impacket_wmiexec": "wmiexec.py",
}


@dataclass
class ValidationResult:
    """Result of tool validation for a single command."""
    valid: bool = True
    tool_name: str = ""
    template_name: str = ""
    # Privilege
    requires_privilege: str = "none"  # "none" | "sudo" | "root"
    current_privilege: str = "user"
    privilege_ok: bool = True
    privilege_reason: str = ""
    # Tool availability
    tool_available: bool = True
    tool_binary: str = ""
    # Phase
    phase_appropriate: bool = True
    phase_reason: str = ""
    # Teaching
    teaching_point: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "tool_name": self.tool_name,
            "template_name": self.template_name,
            "requires_privilege": self.requires_privilege,
            "current_privilege": self.current_privilege,
            "privilege_ok": self.privilege_ok,
            "privilege_reason": self.privilege_reason,
            "tool_available": self.tool_available,
            "tool_binary": self.tool_binary,
            "phase_appropriate": self.phase_appropriate,
            "phase_reason": self.phase_reason,
            "teaching_point": self.teaching_point,
        }


class ToolValidator:
    """
    Validates commands against privilege state and tool availability.

    Usage:
        validator = ToolValidator()
        result = validator.validate(template, privilege_state="user")
        if not result.valid:
            print(f"Blocked: {result.teaching_point}")
    """

    def __init__(self, check_availability: bool = False):
        """
        Args:
            check_availability: If True, also checks if binary is installed.
                                Disabled by default for performance.
        """
        self._check_availability = check_availability
        self._availability_cache: Dict[str, bool] = {}
        self._stats = {
            "total_checks": 0,
            "privilege_blocked": 0,
            "tool_missing": 0,
            "phase_blocked": 0,
            "passed": 0,
        }

    def validate(
        self,
        template: "CommandTemplate",
        privilege_state: str = "user",
        current_phase: Optional[str] = None,
    ) -> ValidationResult:
        """
        Validate a CommandTemplate against current state.

        Args:
            template: The command template to validate
            privilege_state: Current privilege level ("user", "sudo", "root")
            current_phase: Current attack phase name (for phase validation)

        Returns:
            ValidationResult with detailed breakdown
        """
        self._stats["total_checks"] += 1

        result = ValidationResult(
            tool_name=template.name,
            template_name=template.name,
            requires_privilege=getattr(template, 'requires_privilege', 'none'),
            current_privilege=privilege_state,
        )

        # Check privilege requirement
        req_priv = getattr(template, 'requires_privilege', 'none')
        if req_priv == "root" and privilege_state not in ("root",):
            result.valid = False
            result.privilege_ok = False
            result.privilege_reason = f"Requires root, current state: {privilege_state}"
            result.teaching_point = (
                f"Command '{template.name}' requires root access. "
                f"Current privilege: {privilege_state}. "
                f"Need to escalate privileges first."
            )
            self._stats["privilege_blocked"] += 1
        elif req_priv == "sudo" and privilege_state not in ("sudo", "root"):
            result.valid = False
            result.privilege_ok = False
            result.privilege_reason = f"Requires sudo, current state: {privilege_state}"
            result.teaching_point = (
                f"Command '{template.name}' requires sudo access. "
                f"Try sudo -l to check available privilege escalations."
            )
            self._stats["privilege_blocked"] += 1

        # Check tool availability (optional, cached)
        if self._check_availability:
            binary = TOOL_BINARY_MAP.get(template.name, "")
            if binary:
                result.tool_binary = binary
                available = self._is_available(binary)
                result.tool_available = available
                if not available:
                    result.valid = False
                    result.teaching_point = (
                        f"Tool '{binary}' is not installed. "
                        f"Install it or use an alternative approach."
                    )
                    self._stats["tool_missing"] += 1

        if result.valid:
            self._stats["passed"] += 1

        return result

    def _is_available(self, binary: str) -> bool:
        """Check if a binary is available on the system (cached)."""
        if binary in self._availability_cache:
            return self._availability_cache[binary]

        available = shutil.which(binary) is not None
        self._availability_cache[binary] = available
        return available

    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return dict(self._stats)

    def reset_cache(self) -> None:
        """Clear the binary availability cache."""
        self._availability_cache.clear()
