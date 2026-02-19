"""
core/ops/execution_classifier.py — Local vs Remote command classification

Classifies commands into execution contexts to prevent:
  - Remote exploit commands running locally.
  - Local-only commands generating remote discoveries.
  - Phase-inappropriate local filesystem access.

Used by ShadowAgent as a guardrail layer.
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Optional, Tuple

logger = logging.getLogger("ariaska.ops.classifier")


class ExecutionClass(Enum):
    """Classification of command execution context."""

    LOCAL_OPS = "local_ops"           # Local machine setup/maintenance
    REMOTE_EXPLOIT = "remote_exploit" # Offensive against target
    REMOTE_RECON = "remote_recon"     # Reconnaissance against target
    AMBIGUOUS = "ambiguous"           # Cannot determine


# ── Patterns ──────────────────────────────────────────────────────────────────

# Commands that are inherently local-only
_LOCAL_ONLY_PATTERNS = re.compile(
    r"^(apt(-get)?\s|dpkg\s|pip3?\s+install|pipx\s|go\s+install|"
    r"gem\s+install|snap\s+install|which\s|command\s+-v|"
    r"ls\s+/usr/share|cat\s+/usr/share|find\s+/usr/share|"
    r"ls\s+/opt|cat\s+/etc/(?!hosts)|"
    r"mkdir\s|export\s|echo\s.*>|tee\s)",
    re.IGNORECASE,
)

# Commands that reference local filesystem for reference (not attack)
_LOCAL_REFERENCE_PATTERNS = re.compile(
    r"(searchsploit|msfconsole\s+.*search|locate\s|find\s+/usr/share|"
    r"cat\s+/usr/share/|head\s+/usr/share/|grep\s+.*\s+/usr/share)",
    re.IGNORECASE,
)

# Commands that are inherently remote
_REMOTE_PATTERNS = re.compile(
    r"(nmap\s|masscan\s|nikto\s|gobuster\s|hydra\s|medusa\s|"
    r"sqlmap\s|curl\s+-|wget\s|netcat\s|nc\s+-|"
    r"smbclient\s|enum4linux\s|crackmapexec\s|netexec\s|"
    r"ssh\s|sshpass\s|telnet\s|ftp\s|"
    r"msfconsole\s+.*-x\s.*exploit|rpcclient\s|"
    r"impacket-|proxychains\s|socat\s|wfuzz\s|ffuf\s|"
    r"feroxbuster\s|whatweb\s|wpscan\s|dnsrecon\s|"
    r"dig\s|nslookup\s|host\s)",
    re.IGNORECASE,
)

# Exploit phase — commands that MUST reference target
_EXPLOIT_REMOTE_REQUIRED = re.compile(
    r"(hydra|sqlmap|msfconsole|metasploit|exploit|"
    r"crackmapexec|netexec|impacket)",
    re.IGNORECASE,
)


class ExecutionClassifier:
    """
    Classifies commands into LOCAL_OPS vs REMOTE execution contexts.

    Security invariants:
      - LOCAL_OPS commands NEVER generate remote discoveries.
      - REMOTE_EXPLOIT commands MUST reference target IP or domain.
      - Phase-context validation prevents misclassified execution.
    """

    @staticmethod
    def classify(
        command: str,
        target_ip: str = "",
        domain: Optional[str] = None,
    ) -> ExecutionClass:
        """
        Classify a command's execution context.

        Args:
            command: The shell command string.
            target_ip: Expected target IP address.
            domain: Expected target domain (optional).

        Returns:
            ExecutionClass enum value.
        """
        if not command or not command.strip():
            return ExecutionClass.AMBIGUOUS

        cmd = command.strip()

        # Local-only commands
        if _LOCAL_ONLY_PATTERNS.match(cmd):
            return ExecutionClass.LOCAL_OPS

        # Local reference tools (searchsploit, local msf search)
        if _LOCAL_REFERENCE_PATTERNS.search(cmd):
            # But check if it also targets remote
            has_target = _has_target_reference(cmd, target_ip, domain)
            if not has_target:
                return ExecutionClass.LOCAL_OPS

        # Remote patterns with target reference
        if _REMOTE_PATTERNS.search(cmd):
            has_target = _has_target_reference(cmd, target_ip, domain)
            if has_target:
                # Determine recon vs exploit
                if _EXPLOIT_REMOTE_REQUIRED.search(cmd):
                    return ExecutionClass.REMOTE_EXPLOIT
                return ExecutionClass.REMOTE_RECON
            else:
                # Remote tool but no target — ambiguous/local
                return ExecutionClass.AMBIGUOUS

        # Fallback with target check
        if target_ip and target_ip in cmd:
            return ExecutionClass.REMOTE_RECON
        if domain and domain in cmd:
            return ExecutionClass.REMOTE_RECON

        return ExecutionClass.AMBIGUOUS

    @staticmethod
    def validate_execution_context(
        command: str,
        classification: ExecutionClass,
        current_phase: str,
    ) -> Tuple[bool, str]:
        """
        Validate that the command classification is appropriate for the phase.

        Args:
            command: The command being validated.
            classification: Its ExecutionClass.
            current_phase: Current attack phase string.

        Returns:
            Tuple of (is_valid, reason).
        """
        phase_lower = current_phase.lower()

        # REJECT: Remote exploit command without target
        if classification == ExecutionClass.AMBIGUOUS:
            if _EXPLOIT_REMOTE_REQUIRED.search(command):
                return False, "Exploit command has no target reference"

        # REJECT: Local OPS in exploit/privesc/exfil phases generating discoveries
        if classification == ExecutionClass.LOCAL_OPS:
            if phase_lower in ("exploit", "exploitation", "privesc",
                               "privilege_escalation", "exfiltration", "exfiltrate"):
                return True, "LOCAL_OPS allowed but no discoveries will be extracted"

        # REJECT: Remote exploit in recon phase (too early)
        if classification == ExecutionClass.REMOTE_EXPLOIT:
            if phase_lower in ("recon", "reconnaissance"):
                return False, "Exploit commands not appropriate in RECON phase"

        return True, "ok"

    @staticmethod
    def should_extract_discoveries(classification: ExecutionClass) -> bool:
        """
        Whether discoveries should be extracted from command output.

        LOCAL_OPS commands NEVER generate remote target discoveries.
        """
        return classification in (
            ExecutionClass.REMOTE_EXPLOIT,
            ExecutionClass.REMOTE_RECON,
        )


def _has_target_reference(cmd: str, target_ip: str, domain: Optional[str]) -> bool:
    """Check if command references the target IP or domain."""
    if target_ip and target_ip in cmd:
        return True
    if domain and domain.lower() in cmd.lower():
        return True
    # Check for common target placeholders
    if "{target}" in cmd or "{ip}" in cmd or "{domain}" in cmd:
        return True
    return False
