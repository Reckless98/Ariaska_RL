"""core/execution/ctf_mode.py — Phase 41 C5: CTF-mode execution helpers.

Adds CTF-specific behaviour on top of LiveCommandExecutor:
  - Automatic flag pattern detection in command output
  - Priority queue of flag-harvesting commands
  - Submission helper (HTB API stub)
  - Elapsed budget tracking toward CTF time limits
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("ariaska.ctf_mode")

# Common CTF flag patterns (HTB, TryHackMe, custom)
_FLAG_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"[a-f0-9]{32}", re.IGNORECASE),  # MD5-style (HTB standard)
    re.compile(r"flag\{[^\}]+\}", re.IGNORECASE),  # flag{...}
    re.compile(r"THM\{[^\}]+\}", re.IGNORECASE),  # TryHackMe
    re.compile(r"CTF\{[^\}]+\}", re.IGNORECASE),  # generic CTF
    re.compile(r"HTB\{[^\}]+\}", re.IGNORECASE),  # HackTheBox wrapped
]

# Commands commonly used to read flags
FLAG_HARVEST_COMMANDS: List[str] = [
    "cat /home/*/user.txt",
    "cat /root/root.txt",
    "type C:\\Users\\*\\Desktop\\user.txt",
    "type C:\\Users\\Administrator\\Desktop\\root.txt",
    "find / -name user.txt -exec cat {} \\; 2>/dev/null",
    "find / -name root.txt -exec cat {} \\; 2>/dev/null",
]


@dataclass
class FlagCapture:
    """Represents a captured flag."""
    value: str
    flag_type: str  # "user" | "root" | "unknown"
    source_command: str = ""
    captured_at: float = 0.0
    agent: str = ""

    def __post_init__(self) -> None:
        if self.captured_at == 0.0:
            self.captured_at = time.time()


@dataclass
class CTFConfig:
    """Configuration for CTF mode."""
    enabled: bool = True
    time_limit_minutes: int = 120  # 2-hour default
    auto_harvest: bool = True  # Inject flag-reading commands
    submit_flags: bool = False  # Auto-submit to platform (stub)
    flag_patterns: List[re.Pattern[str]] = field(default_factory=lambda: list(_FLAG_PATTERNS))


class CTFModeTracker:
    """Tracks CTF-mode state: flag captures, time budget, harvest queue."""

    def __init__(self, config: Optional[CTFConfig] = None) -> None:
        self._config = config or CTFConfig()
        self._flags: Dict[str, FlagCapture] = {}  # value -> FlagCapture
        self._start_time = time.time()
        self._harvest_queue: List[str] = list(FLAG_HARVEST_COMMANDS)
        self._submitted: Set[str] = set()
        logger.info(
            "[CTF-MODE] Tracker initialized "
            f"(limit={self._config.time_limit_minutes}m, "
            f"auto_harvest={self._config.auto_harvest})"
        )

    # ── Flag detection ─────────────────────────────────────────────

    def scan_output(
        self,
        output: str,
        command: str = "",
        agent: str = "",
    ) -> List[FlagCapture]:
        """Scan command output for flag patterns. Returns new captures."""
        if not output:
            return []

        new_captures: List[FlagCapture] = []
        for pat in self._config.flag_patterns:
            for match in pat.finditer(output):
                value = match.group(0)
                if value in self._flags:
                    continue  # Already captured
                flag_type = self._infer_flag_type(command, value)
                capture = FlagCapture(
                    value=value,
                    flag_type=flag_type,
                    source_command=command,
                    agent=agent,
                )
                self._flags[value] = capture
                new_captures.append(capture)
                logger.info(
                    f"[CTF-MODE] Flag captured: {flag_type} "
                    f"({value[:8]}...{value[-4:] if len(value) > 12 else value})"
                )
        return new_captures

    @staticmethod
    def _infer_flag_type(command: str, value: str) -> str:
        """Heuristic to classify flag as user or root."""
        cmd_lower = command.lower()
        val_lower = value.lower()
        if "root" in cmd_lower or "/root/" in cmd_lower:
            return "root"
        if "user" in cmd_lower or "/home/" in cmd_lower:
            return "user"
        if "root" in val_lower:
            return "root"
        return "unknown"

    # ── Harvest queue ──────────────────────────────────────────────

    def next_harvest_command(self) -> Optional[str]:
        """Pop next flag-reading command from the queue.

        Returns None when queue is exhausted or auto_harvest is off.
        """
        if not self._config.auto_harvest:
            return None
        if self._harvest_queue:
            return self._harvest_queue.pop(0)
        return None

    def add_harvest_command(self, cmd: str) -> None:
        """Add a custom flag-reading command to the queue."""
        if cmd not in self._harvest_queue:
            self._harvest_queue.append(cmd)

    # ── Time budget ────────────────────────────────────────────────

    @property
    def elapsed_minutes(self) -> float:
        return (time.time() - self._start_time) / 60.0

    @property
    def remaining_minutes(self) -> float:
        return max(0.0, self._config.time_limit_minutes - self.elapsed_minutes)

    @property
    def time_expired(self) -> bool:
        return self.elapsed_minutes >= self._config.time_limit_minutes

    # ── Flag state ─────────────────────────────────────────────────

    @property
    def has_user_flag(self) -> bool:
        return any(f.flag_type == "user" for f in self._flags.values())

    @property
    def has_root_flag(self) -> bool:
        return any(f.flag_type == "root" for f in self._flags.values())

    @property
    def both_flags(self) -> bool:
        return self.has_user_flag and self.has_root_flag

    @property
    def captured_flags(self) -> List[FlagCapture]:
        return list(self._flags.values())

    @property
    def flag_count(self) -> int:
        return len(self._flags)

    # ── Submission stub ────────────────────────────────────────────

    def submit_flag(self, value: str) -> bool:
        """Submit a flag to the CTF platform (stub).

        Returns True if submission accepted (always True in stub).
        """
        if value in self._submitted:
            return False  # Already submitted
        self._submitted.add(value)
        logger.info(f"[CTF-MODE] Flag submitted (stub): {value[:8]}...")
        return True

    # ── Stats ──────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        return {
            "enabled": self._config.enabled,
            "elapsed_minutes": round(self.elapsed_minutes, 1),
            "remaining_minutes": round(self.remaining_minutes, 1),
            "time_expired": self.time_expired,
            "flags_captured": self.flag_count,
            "has_user_flag": self.has_user_flag,
            "has_root_flag": self.has_root_flag,
            "both_flags": self.both_flags,
            "harvest_remaining": len(self._harvest_queue),
            "flags_submitted": len(self._submitted),
        }

    def reset(self) -> None:
        """Reset tracker state for new engagement."""
        self._flags.clear()
        self._submitted.clear()
        self._harvest_queue = list(FLAG_HARVEST_COMMANDS)
        self._start_time = time.time()
