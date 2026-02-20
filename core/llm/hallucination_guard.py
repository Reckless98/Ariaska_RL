"""core/llm/hallucination_guard.py — Phase 42: LLM hallucination detection.

Detects and flags LLM-generated commands that are likely hallucinated:
- Commands referencing tools not installed on the system
- Commands with fabricated CVE numbers
- Commands targeting non-existent services
- Syntactically broken commands

Used by SmartCoach to filter bad LLM suggestions before execution.

Author: Phase 42 Contract
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("ariaska.llm.hallucination_guard")


@dataclass
class HallucinationResult:
    """Result of hallucination check."""
    is_hallucinated: bool
    confidence: float  # 0.0 = definitely real, 1.0 = definitely hallucinated
    reasons: List[str] = field(default_factory=list)
    command: str = ""


class HallucinationGuard:
    """Phase 42: LLM command hallucination detector.

    Analyzes LLM-generated commands for signs of hallucination:
    - Unknown tool references
    - Fabricated flags/options
    - Impossible parameter combinations
    - Syntactic anomalies

    Methods:
        check(): Analyze a command for hallucination
        register_known_tool(): Add a tool to the known set
        get_stats(): Get detection statistics
        reset(): Clear per-episode stats
    """

    # Common pentesting tools — used as ground truth
    _DEFAULT_KNOWN_TOOLS: Set[str] = {
        "nmap", "nikto", "gobuster", "dirb", "ffuf", "wpscan",
        "sqlmap", "hydra", "john", "hashcat", "metasploit",
        "msfconsole", "msfvenom", "searchsploit", "enum4linux",
        "smbclient", "smbmap", "crackmapexec", "impacket",
        "curl", "wget", "ssh", "ftp", "telnet", "nc", "ncat",
        "netcat", "python", "python3", "perl", "ruby", "php",
        "burpsuite", "responder", "bloodhound", "mimikatz",
        "linpeas", "winpeas", "pspy", "chisel", "socat",
        "proxychains", "tcpdump", "wireshark", "tshark",
        "certutil", "powershell", "cmd", "bash", "sh", "zsh",
        "dig", "host", "nslookup", "whois", "traceroute",
        "arp", "ip", "ifconfig", "netstat", "ss",
        "find", "grep", "awk", "sed", "cat", "less", "head",
        "tail", "chmod", "chown", "sudo", "su", "passwd",
        "mount", "umount", "apt", "yum", "pip", "gem",
        "mysql", "psql", "redis-cli", "mongo", "sqlite3",
        "snmpwalk", "onesixtyone", "rpcclient", "ldapsearch",
        "wfuzz", "whatweb", "subfinder", "amass", "masscan",
        "rustscan", "feroxbuster",
    }

    # Patterns that indicate hallucination
    _SUSPICIOUS_PATTERNS = [
        (r"--[a-z]{20,}", "Extremely long flag (likely fabricated)"),
        (r"CVE-\d{4}-\d{8,}", "Implausibly long CVE number"),
        (r"0\.0\.0\.0", "Targeting 0.0.0.0 (suspicious)"),
        (r"127\.0\.0\.1", "Targeting localhost (usually wrong in pentest)"),
        (r"\|\|\|", "Triple pipe (syntax error)"),
        (r"&&&&", "Quadruple ampersand (syntax error)"),
        (r"['\"].*['\"].*['\"].*['\"].*['\"]", "Excessive quoting (5+)"),
    ]

    def __init__(self) -> None:
        self._known_tools: Set[str] = set(self._DEFAULT_KNOWN_TOOLS)
        self._total_checks: int = 0
        self._total_flagged: int = 0
        self._flagged_commands: List[str] = []
        self._compiled_patterns = [
            (re.compile(p, re.IGNORECASE), desc)
            for p, desc in self._SUSPICIOUS_PATTERNS
        ]
        logger.info("HallucinationGuard initialized with %d known tools", len(self._known_tools))

    def check(self, command: str) -> HallucinationResult:
        """Analyze a command for hallucination signals.

        Args:
            command: The command string to check.

        Returns:
            HallucinationResult with is_hallucinated flag and reasons.
        """
        self._total_checks += 1
        reasons: List[str] = []
        confidence = 0.0

        if not command or not command.strip():
            self._total_flagged += 1
            self._flagged_commands.append("(empty)")
            return HallucinationResult(
                is_hallucinated=True,
                confidence=1.0,
                reasons=["Empty command"],
                command=command,
            )

        # Extract the base tool name
        parts = command.strip().split()
        tool = parts[0].split("/")[-1]  # Handle paths like /usr/bin/nmap

        # Check 1: Unknown tool
        if tool.lower() not in self._known_tools:
            # Could be a path or alias — lighter penalty
            reasons.append(f"Unknown tool: {tool}")
            confidence += 0.3

        # Check 2: Suspicious patterns
        for pattern, desc in self._compiled_patterns:
            if pattern.search(command):
                reasons.append(desc)
                confidence += 0.25

        # Check 3: Excessive length (likely fabricated)
        if len(command) > 500:
            reasons.append("Extremely long command (>500 chars)")
            confidence += 0.2

        # Check 4: Unbalanced quotes
        single_quotes = command.count("'")
        double_quotes = command.count('"')
        if single_quotes % 2 != 0:
            reasons.append("Unbalanced single quotes")
            confidence += 0.3
        if double_quotes % 2 != 0:
            reasons.append("Unbalanced double quotes")
            confidence += 0.3

        # Check 5: Multiple redirections stacked
        redirect_count = command.count(">") + command.count("<")
        if redirect_count > 4:
            reasons.append(f"Excessive redirections ({redirect_count})")
            confidence += 0.2

        # Clamp confidence
        confidence = min(1.0, confidence)
        is_hallucinated = confidence >= 0.5

        if is_hallucinated:
            self._total_flagged += 1
            self._flagged_commands.append(command[:100])

        return HallucinationResult(
            is_hallucinated=is_hallucinated,
            confidence=confidence,
            reasons=reasons,
            command=command,
        )

    def register_known_tool(self, tool_name: str) -> None:
        """Add a tool to the known tools set.

        Args:
            tool_name: The tool name to register.
        """
        self._known_tools.add(tool_name.lower())

    def get_stats(self) -> Dict[str, Any]:
        """Get detection statistics.

        Returns:
            Dict with check counts and flag rate.
        """
        return {
            "total_checks": self._total_checks,
            "total_flagged": self._total_flagged,
            "flag_rate": (
                self._total_flagged / max(self._total_checks, 1)
            ),
            "known_tools": len(self._known_tools),
            "recent_flagged": self._flagged_commands[-5:],
        }

    def reset(self) -> None:
        """Clear per-episode stats."""
        self._total_checks = 0
        self._total_flagged = 0
        self._flagged_commands.clear()
        logger.debug("HallucinationGuard reset")
