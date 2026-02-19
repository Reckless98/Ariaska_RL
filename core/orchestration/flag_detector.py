"""core/orchestration/flag_detector.py — Phase 41: CTF flag detection.

Standalone module for detecting CTF flags in command output.
Supports HTB, THM, OSCP, and generic flag formats.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import FrozenSet, List, Set

logger = logging.getLogger("ariaska.orchestration.flag_detector")

# ── Flag patterns by platform ─────────────────────────────────
FLAG_PATTERNS = [
    ("HTB", re.compile(r"HTB\{[A-Za-z0-9_!@#$%^&*()\-+=]+\}")),
    ("THM", re.compile(r"THM\{[A-Za-z0-9_!@#$%^&*()\-+=]+\}")),
    ("flag", re.compile(r"flag\{[A-Za-z0-9_!@#$%^&*()\-+=]+\}", re.IGNORECASE)),
    ("FLAG", re.compile(r"FLAG\{[A-Za-z0-9_!@#$%^&*()\-+=]+\}")),
    ("CTF", re.compile(r"ctf\{[A-Za-z0-9_!@#$%^&*()\-+=]+\}", re.IGNORECASE)),
    ("MD5", re.compile(r"(?:^|\s)([0-9a-f]{32})(?:\s|$)", re.MULTILINE)),
]

# False-positive hex strings to ignore
_HEX_FALSE_POSITIVES: FrozenSet[str] = frozenset({
    "00000000000000000000000000000000",
    "d41d8cd98f00b204e9800998ecf8427e",  # MD5 of empty
    "ffffffffffffffffffffffffffffffff",
})

# Contexts that indicate flag reading
_FLAG_CONTEXTS = re.compile(
    r"cat\s+(user\.txt|root\.txt|proof\.txt|local\.txt|flag\.txt)",
    re.IGNORECASE,
)


@dataclass
class FlagMatch:
    """A detected CTF flag."""
    flag_type: str
    value: str
    source: str = "command_output"
    is_user_flag: bool = False
    is_root_flag: bool = False


def detect_flags(output: str, command: str = "") -> List[FlagMatch]:
    """Scan command output for CTF flag patterns.

    Args:
        output: Raw command output text.
        command: The command that produced this output (for context).

    Returns:
        List of detected flag matches, deduplicated.
    """
    results: List[FlagMatch] = []
    seen: Set[str] = set()

    for flag_type, pattern in FLAG_PATTERNS:
        for m in pattern.finditer(output):
            val = m.group(1) if m.lastindex else m.group(0)
            val = val.strip()
            if val in seen or val in _HEX_FALSE_POSITIVES:
                continue
            # Skip very short hex (likely not a flag)
            if flag_type == "MD5" and len(val) < 32:
                continue
            seen.add(val)
            fm = FlagMatch(flag_type=flag_type, value=val)
            # Determine user vs root context
            if command:
                cmd_lower = command.lower()
                if "user.txt" in cmd_lower or "local.txt" in cmd_lower:
                    fm.is_user_flag = True
                elif "root.txt" in cmd_lower or "proof.txt" in cmd_lower:
                    fm.is_root_flag = True
            results.append(fm)

    # Check for flag file reading context
    if _FLAG_CONTEXTS.search(command):
        for fm in results:
            if fm.flag_type == "MD5":
                ctx_match = _FLAG_CONTEXTS.search(command)
                if ctx_match:
                    fname = ctx_match.group(1).lower()
                    if "user" in fname or "local" in fname:
                        fm.is_user_flag = True
                    elif "root" in fname or "proof" in fname:
                        fm.is_root_flag = True

    return results


def format_flag_celebration(flags: List[FlagMatch]) -> str:
    """Format a rich-compatible celebration string for captured flags."""
    if not flags:
        return ""
    lines = []
    for f in flags:
        role = ""
        if f.is_user_flag:
            role = " [USER FLAG]"
        elif f.is_root_flag:
            role = " [ROOT FLAG]"
        lines.append(f"[bold green]FLAG CAPTURED{role}: {f.value}[/bold green]")
    return "\n".join(lines)
