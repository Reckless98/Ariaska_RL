"""core/orchestration/output_parser_utils.py — Phase 41: Discovery parsing utilities.

Extracted from SmartOrchestrator._parse_output_for_discoveries().
Provides standalone flag-detection and discovery-extraction helpers
that SmartOrchestrator delegates to.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

logger = logging.getLogger("ariaska.orchestration.output_parser_utils")

# ── CTF flag patterns ──────────────────────────────────────────
FLAG_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"HTB\{[A-Za-z0-9_!@#$%^&*()\-+=]+\}"),
    re.compile(r"THM\{[A-Za-z0-9_!@#$%^&*()\-+=]+\}"),
    re.compile(r"flag\{[A-Za-z0-9_!@#$%^&*()\-+=]+\}", re.IGNORECASE),
    re.compile(r"FLAG\{[A-Za-z0-9_!@#$%^&*()\-+=]+\}"),
    re.compile(r"ctf\{[A-Za-z0-9_!@#$%^&*()\-+=]+\}", re.IGNORECASE),
    re.compile(r"(?:^|\s)([0-9a-f]{32})(?:\s|$)", re.MULTILINE),
]

# Well-known false-positive hex strings to ignore
_HEX_FALSE_POSITIVES: FrozenSet[str] = frozenset({
    "00000000000000000000000000000000",
    "d41d8cd98f00b204e9800998ecf8427e",  # MD5 of empty string
})


@dataclass
class FlagMatch:
    """A detected CTF flag."""
    flag_type: str
    value: str
    source: str = "command_output"


def detect_flags(output: str) -> List[FlagMatch]:
    """Scan command output for CTF flag patterns.

    Args:
        output: Raw command output text.

    Returns:
        List of detected flag matches, deduplicated.
    """
    results: List[FlagMatch] = []
    seen: Set[str] = set()

    type_map = ["HTB", "THM", "flag", "FLAG", "CTF", "MD5"]
    for idx, pattern in enumerate(FLAG_PATTERNS):
        for m in pattern.finditer(output):
            val = m.group(1) if m.lastindex else m.group(0)
            val = val.strip()
            if val in seen or val in _HEX_FALSE_POSITIVES:
                continue
            seen.add(val)
            results.append(FlagMatch(
                flag_type=type_map[min(idx, len(type_map) - 1)],
                value=val,
            ))
    return results


# ── Port / service extraction helpers ──────────────────────────
_PORT_LINE_RE = re.compile(
    r"(\d{1,5})/(tcp|udp)\s+(open|filtered)\s+(\S+)"
)

_CREDENTIAL_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"login:\s*(\S+)\s+password:\s*(\S+)", re.IGNORECASE),
    re.compile(r"\[\d+\]\[\w+\]\s+host:.+login:\s*(\S+)\s+password:\s*(\S+)", re.IGNORECASE),
    re.compile(r"(\w+):(\S+)@", re.IGNORECASE),
]


def extract_ports(output: str) -> List[Tuple[int, str, str]]:
    """Extract (port, protocol, service) tuples from nmap-style output."""
    results: List[Tuple[int, str, str]] = []
    seen: Set[int] = set()
    for m in _PORT_LINE_RE.finditer(output):
        port = int(m.group(1))
        if port not in seen and 1 <= port <= 65535:
            seen.add(port)
            results.append((port, m.group(2), m.group(4)))
    return results


def extract_credentials(output: str) -> List[Tuple[str, str]]:
    """Extract (username, password) pairs from command output."""
    results: List[Tuple[str, str]] = []
    seen: Set[str] = set()
    for pat in _CREDENTIAL_PATTERNS:
        for m in pat.finditer(output):
            key = f"{m.group(1)}:{m.group(2)}"
            if key not in seen:
                seen.add(key)
                results.append((m.group(1), m.group(2)))
    return results
