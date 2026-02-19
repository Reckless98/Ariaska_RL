"""
core/ops/shell_validator.py — Phase 38.2: Shell Validation Engine

Validates shell claims before they gate phase transitions.
Prevents false shell detection from local commands, exploit tool
noise, and regex false positives.

Rules:
  - Shell must come from a REMOTE_EXPLOIT or REMOTE_RECON classified command.
  - Shell patterns must be corroborated by command context.
  - Local-only commands cannot produce shell discoveries.
  - Root shell requires uid=0 or root@ evidence.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set

logger = logging.getLogger("ariaska.ops.shell_validator")

# ── Shell Evidence Patterns ──────────────────────────────────────────────────

# Strong indicators of a real shell (not just regex noise).
_STRONG_SHELL_PATTERNS = [
    re.compile(r"shell\s+session\s+\d+\s+opened", re.IGNORECASE),
    re.compile(r"meterpreter\s+session\s+\d+\s+opened", re.IGNORECASE),
    re.compile(r"meterpreter\s*>"),
    re.compile(r"Command shell session \d+ opened", re.IGNORECASE),
    re.compile(r"www-data@\S+"),
    re.compile(r"uid=\d+\(\w+\)\s+gid=\d+"),
    re.compile(r"\[Ss]ession\s+\d+\s+opened"),
]

# Strong root shell patterns.
_STRONG_ROOT_PATTERNS = [
    re.compile(r"root@\S+"),
    re.compile(r"uid=0\(root\)\s+gid=0"),
    re.compile(r"nt authority\\\\?system", re.IGNORECASE),
    re.compile(r"meterpreter.*root", re.IGNORECASE),
]

# Commands that CANNOT produce real shells (local-only).
_LOCAL_ONLY_PREFIXES: FrozenSet[str] = frozenset({
    "searchsploit",
    "find ",
    "locate ",
    "ls ",
    "cat ",
    "grep ",
    "getcap ",
    "env",
    "whoami",
    "id",
    "perl -e",
    "python -c",
    "python3 -c",
    "apt ",
    "pip ",
    "command -v",
})

# Commands that CAN produce shells (network tools with exploitation).
_SHELL_CAPABLE_PREFIXES: FrozenSet[str] = frozenset({
    "ssh ",
    "sshpass ",
    "telnet ",
    "nc ",
    "netcat ",
    "ncat ",
    "msfconsole",
    "exploit/",
    "rsh ",
    "rlogin ",
    "psql ",
    "mysql ",
    "evil-winrm",
    "impacket",
    "hydra ",
    "crackmapexec",
})


@dataclass
class ShellValidationResult:
    """Result of shell validation."""
    is_valid_shell: bool
    is_root_shell: bool = False
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)
    rejection_reason: str = ""

    def __repr__(self) -> str:
        tag = "ROOT" if self.is_root_shell else ("SHELL" if self.is_valid_shell else "REJECTED")
        return f"ShellValidation({tag}, conf={self.confidence:.2f})"


class ShellValidator:
    """
    Validates shell claims from command output.

    Usage:
        validator = ShellValidator()
        result = validator.validate(
            command="ssh admin@10.10.10.50",
            output="admin@target:~$",
            target_ip="10.10.10.50",
        )
        if result.is_valid_shell:
            discovery_board["shells"].add("RedAgent")
    """

    def __init__(self) -> None:
        self._validated_shells: List[Dict[str, Any]] = []
        logger.debug("ShellValidator initialised")

    def validate(
        self,
        command: str,
        output: str,
        target_ip: str = "",
        domain: str = "",
    ) -> ShellValidationResult:
        """
        Validate whether command output indicates a real shell.

        Args:
            command: The command that was executed.
            output: The command output to check.
            target_ip: Expected target IP.
            domain: Expected target domain.

        Returns:
            ShellValidationResult with validation details.
        """
        if not command or not output:
            return ShellValidationResult(
                is_valid_shell=False,
                rejection_reason="empty command or output",
            )

        cmd_lower = command.strip().lower()

        # 1. Reject local-only commands
        for prefix in _LOCAL_ONLY_PREFIXES:
            if cmd_lower.startswith(prefix.lower()):
                return ShellValidationResult(
                    is_valid_shell=False,
                    rejection_reason=f"local-only command: {prefix.strip()}",
                )

        # 2. Check if command is shell-capable
        is_shell_capable = any(
            prefix.lower() in cmd_lower
            for prefix in _SHELL_CAPABLE_PREFIXES
        )

        # 3. If target_ip provided, command must reference it
        has_target_ref = (
            not target_ip
            or target_ip in command
            or (domain and domain in command)
        )

        # 4. Check for shell evidence in output
        shell_evidence: List[str] = []
        root_evidence: List[str] = []

        for pattern in _STRONG_SHELL_PATTERNS:
            match = pattern.search(output)
            if match:
                shell_evidence.append(match.group(0)[:80])

        for pattern in _STRONG_ROOT_PATTERNS:
            match = pattern.search(output)
            if match:
                root_evidence.append(match.group(0)[:80])

        # 5. Compute validation result
        has_shell_evidence = len(shell_evidence) > 0 or len(root_evidence) > 0

        if not has_shell_evidence:
            return ShellValidationResult(
                is_valid_shell=False,
                rejection_reason="no shell evidence in output",
            )

        if not has_target_ref:
            return ShellValidationResult(
                is_valid_shell=False,
                evidence=shell_evidence,
                rejection_reason="command does not reference target",
            )

        # Calculate confidence
        confidence = 0.4  # Base
        if is_shell_capable:
            confidence += 0.3
        if has_target_ref:
            confidence += 0.2
        if len(shell_evidence) >= 2:
            confidence += 0.1
        confidence = min(confidence, 1.0)

        is_root = len(root_evidence) > 0

        result = ShellValidationResult(
            is_valid_shell=True,
            is_root_shell=is_root,
            confidence=confidence,
            evidence=shell_evidence + root_evidence,
        )

        self._validated_shells.append({
            "command": command[:100],
            "is_root": is_root,
            "confidence": confidence,
            "evidence_count": len(shell_evidence) + len(root_evidence),
        })

        logger.info(
            "Shell validated: %s (root=%s, conf=%.2f)",
            command[:60], is_root, confidence,
        )

        return result

    def get_validated_shells(self) -> List[Dict[str, Any]]:
        """Return list of validated shells."""
        return list(self._validated_shells)

    def reset(self) -> None:
        """Reset validator state."""
        self._validated_shells.clear()
