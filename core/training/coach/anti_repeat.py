"""Anti-Repeat Guard — prevents command repetition and provides alternatives.

Phase 41: Extracted from SmartCoach for modularity.
"""
from __future__ import annotations

import logging
from typing import Dict, Any, List, Set, Optional
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger("ariaska.coach.anti_repeat")


@dataclass
class AntiRepeatConfig:
    """Configuration for anti-repeat behavior."""
    max_recent: int = 30
    prefix_window: int = 10
    exact_block: bool = True
    prefix_block: bool = True


class AntiRepeatGuard:
    """Tracks recent commands and prevents repetition."""

    def __init__(self, config: Optional[AntiRepeatConfig] = None) -> None:
        self.config = config or AntiRepeatConfig()
        self._recent_commands: deque = deque(maxlen=self.config.max_recent)
        self._step_commands: Set[str] = set()
        self._recent_prefixes: deque = deque(maxlen=self.config.prefix_window)
        self._repeat_count: int = 0
        self._block_count: int = 0

    def is_repeat(self, command: str) -> bool:
        """Check if a command is a repeat."""
        if not command:
            return False
        cmd = command.strip()
        if self.config.exact_block and cmd in self._step_commands:
            self._repeat_count += 1
            return True
        if self.config.exact_block and cmd in self._recent_commands:
            self._repeat_count += 1
            return True
        if self.config.prefix_block:
            prefix = self._extract_prefix(cmd)
            if list(self._recent_prefixes).count(prefix) >= 2:
                self._repeat_count += 1
                return True
        return False

    def record_command(self, command: str) -> None:
        """Record a command as used."""
        cmd = command.strip()
        self._recent_commands.append(cmd)
        self._step_commands.add(cmd)
        self._recent_prefixes.append(self._extract_prefix(cmd))

    def clear_step(self) -> None:
        """Clear per-step tracking."""
        self._step_commands.clear()

    def get_recent_tags(self, k: int = 15) -> Set[str]:
        """Get tags from recent command prefixes."""
        recent = list(self._recent_commands)[-k:]
        return {self._extract_prefix(c) for c in recent if c}

    def get_stats(self) -> Dict[str, int]:
        """Return anti-repeat statistics."""
        return {"repeat_count": self._repeat_count, "block_count": self._block_count, "recent_commands": len(self._recent_commands)}

    def reset(self) -> None:
        """Full reset."""
        self._recent_commands.clear()
        self._step_commands.clear()
        self._recent_prefixes.clear()
        self._repeat_count = 0
        self._block_count = 0

    @staticmethod
    def _extract_prefix(cmd: str) -> str:
        """Extract the tool prefix from a command string."""
        if not cmd:
            return ""
        parts = cmd.strip().split()
        if not parts:
            return ""
        return parts[0].split("/")[-1].lower()
