"""core/execution/executor_config.py — Phase 41: Execution mode configuration.

Provides a config dataclass that controls whether SmartOrchestrator
runs in simulated or live mode, plus dangerous command blocking.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import FrozenSet, List, Optional

logger = logging.getLogger("ariaska.execution.executor_config")


@dataclass
class ExecutorConfig:
    """Configuration for command execution mode."""
    mode: str = "simulated"  # "simulated" or "live"
    ssh_target: Optional[str] = None
    ssh_user: Optional[str] = None
    ssh_key_path: Optional[str] = None
    command_timeout: int = 30
    dangerous_command_confirmation: bool = True

    @classmethod
    def from_env(cls) -> "ExecutorConfig":
        """Create config from environment variables."""
        return cls(
            mode=os.environ.get("ARIASKA_EXECUTION_MODE", "simulated"),
            ssh_target=os.environ.get("ARIASKA_SSH_TARGET"),
            ssh_user=os.environ.get("ARIASKA_SSH_USER"),
            ssh_key_path=os.environ.get("ARIASKA_SSH_KEY"),
            command_timeout=int(os.environ.get("ARIASKA_CMD_TIMEOUT", "30")),
        )

    @property
    def is_live(self) -> bool:
        """Check if running in live mode."""
        return self.mode == "live"


# ── Dangerous command patterns ─────────────────────────────────
_DANGEROUS_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"rm\s+-rf\s+/(?!tmp)"),
    re.compile(r"shutdown"),
    re.compile(r"reboot"),
    re.compile(r"dd\s+if="),
    re.compile(r"mkfs"),
    re.compile(r":\(\)\{"),  # fork bomb
    re.compile(r"mv\s+/\s"),
    re.compile(r"chmod\s+777\s+/"),
]


def is_dangerous_command(command: str) -> bool:
    """Check if a command matches dangerous patterns.

    Args:
        command: Command string to check.

    Returns:
        True if the command is considered dangerous.
    """
    for pattern in _DANGEROUS_PATTERNS:
        if pattern.search(command):
            logger.warning("Dangerous command blocked: %s", command[:80])
            return True
    return False


@dataclass
class ExecutionResult:
    """Result of a command execution."""
    output: str = ""
    exit_code: int = 0
    execution_time: float = 0.0
    is_simulated: bool = True
    command: str = ""
    error: str = ""
