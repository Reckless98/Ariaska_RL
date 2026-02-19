"""
core/ops/sudo_handler.py — Privileged local OPS executor

Loads SUDO_PASSWORD from .env at startup.
Rules:
  - Password NEVER printed, logged, exposed via repr/str, or sent to LLM.
  - Used only for allowlisted local OPS commands (apt, hosts, mkdir).
  - Non-interactive: echo "$pw" | sudo -S <cmd>
  - Respects ARIASKA_DRY_RUN — log intent without execution.
  - On failure: log command (not password), do not retry blindly.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
from dataclasses import dataclass, field
from typing import FrozenSet, Optional

logger = logging.getLogger("ariaska.ops.sudo")

# ── Allow / Deny lists ────────────────────────────────────────────────────────

ALLOWED_SUDO_PREFIXES: FrozenSet[str] = frozenset({
    "apt install",
    "apt-get install",
    "apt update",
    "apt-get update",
    "tee -a ",
    "tee ",
    "sed -i",
    "mkdir -p",
    "apt-get install -y",
    "apt install -y",
})

DENIED_SUDO_PATTERNS: FrozenSet[str] = frozenset({
    "rm ",
    "rm -",
    "chmod ",
    "chown ",
    "dd ",
    "mkfs",
    "rmdir",
    "shutdown",
    "reboot",
    "modprobe",
    "insmod",
    "systemctl disable",
    "init ",
    "kill -9 1",
    ":(){ :|:& };:",
})


@dataclass
class SudoResult:
    """Result of a privileged OPS command."""

    command: str
    success: bool
    stdout: str = ""
    stderr: str = ""
    return_code: int = -1
    blocked: bool = False
    blocked_reason: str = ""
    dry_run: bool = False

    def __repr__(self) -> str:
        # Never include password in repr
        tag = "DRY" if self.dry_run else ("OK" if self.success else "FAIL")
        return f"SudoResult({tag}, cmd={self.command!r:.60}, rc={self.return_code})"


class SudoHandler:
    """
    Manages privileged local OPS execution using SUDO_PASSWORD from .env.

    Security invariants:
      - _password is NEVER exposed via property, log, repr, str, or LLM context.
      - Only ALLOWED_SUDO_PREFIXES commands are executed.
      - DENIED_SUDO_PATTERNS are hard-rejected.
    """

    def __init__(self) -> None:
        self._password: Optional[str] = self._load_password()
        self._lock = threading.Lock()
        self._is_dry_run = os.environ.get("ARIASKA_DRY_RUN", "") == "1"
        self._execution_log: list[dict] = []
        if self._password:
            logger.info("SudoHandler initialised (password loaded)")
        else:
            logger.warning("SudoHandler initialised WITHOUT password — sudo ops will fail")

    # ── Public API ────────────────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        """Whether a sudo password is loaded."""
        return self._password is not None

    def execute_privileged(self, command: str, timeout: int = 120) -> SudoResult:
        """
        Execute an allowlisted command with sudo.

        Args:
            command: The raw command (without sudo prefix).
            timeout: Execution timeout in seconds.

        Returns:
            SudoResult with success/failure details.
        """
        with self._lock:
            return self._execute_locked(command, timeout)

    def get_audit_log(self) -> list[dict]:
        """Return audit log of sudo operations (no passwords)."""
        return list(self._execution_log)

    # ── Private ───────────────────────────────────────────────────────────

    @staticmethod
    def _load_password() -> Optional[str]:
        """Load SUDO_PASSWORD from env / .env file. Never log the value."""
        # Try env first (may already be loaded by python-dotenv)
        pw = os.environ.get("SUDO_PASSWORD")
        if pw:
            return pw

        # Manual .env fallback
        env_paths = [".env", os.path.expanduser("~/.env")]
        for path in env_paths:
            if os.path.isfile(path):
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        for line in fh:
                            line = line.strip()
                            if line.startswith("SUDO_PASSWORD="):
                                val = line.split("=", 1)[1].strip().strip("'\"")
                                if val:
                                    return val
                except OSError:
                    pass
        return None

    def _execute_locked(self, command: str, timeout: int) -> SudoResult:
        """Execute with lock held."""
        # ── Validate against deny list ────────────────────────────────
        cmd_lower = command.strip().lower()
        for pattern in DENIED_SUDO_PATTERNS:
            if pattern.lower() in cmd_lower:
                result = SudoResult(
                    command=command, success=False,
                    blocked=True,
                    blocked_reason=f"DENIED: matches pattern '{pattern}'",
                )
                self._log_op(command, result)
                logger.warning("sudo DENIED: %s (pattern: %s)", command[:80], pattern)
                return result

        # ── Validate against allow list ───────────────────────────────
        allowed = False
        for prefix in ALLOWED_SUDO_PREFIXES:
            if cmd_lower.startswith(prefix.lower()):
                allowed = True
                break
        if not allowed:
            result = SudoResult(
                command=command, success=False,
                blocked=True,
                blocked_reason="NOT in ALLOWED_SUDO_PREFIXES",
            )
            self._log_op(command, result)
            logger.warning("sudo BLOCKED: %s — not in allow list", command[:80])
            return result

        # ── Dry-run gate ──────────────────────────────────────────────
        if self._is_dry_run:
            result = SudoResult(
                command=command, success=True,
                stdout=f"[DRY_RUN] Would execute: sudo {command}",
                dry_run=True,
            )
            self._log_op(command, result)
            logger.info("sudo DRY_RUN: %s", command[:80])
            return result

        # ── Password check ────────────────────────────────────────────
        if not self._password:
            result = SudoResult(
                command=command, success=False,
                stderr="SUDO_PASSWORD not configured",
                blocked=True,
                blocked_reason="No SUDO_PASSWORD in environment",
            )
            self._log_op(command, result)
            logger.error("sudo FAILED: no password configured")
            return result

        # ── Execute ───────────────────────────────────────────────────
        full_cmd = f'echo "{self._password}" | sudo -S {command}'
        try:
            proc = subprocess.run(
                full_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                stdin=subprocess.DEVNULL,
            )
            success = proc.returncode == 0
            result = SudoResult(
                command=command,
                success=success,
                stdout=proc.stdout,
                stderr=proc.stderr,
                return_code=proc.returncode,
            )
            if success:
                logger.info("sudo OK: %s", command[:80])
            else:
                logger.warning("sudo FAILED (rc=%d): %s", proc.returncode, command[:80])

        except subprocess.TimeoutExpired:
            result = SudoResult(
                command=command, success=False,
                stderr=f"Timeout ({timeout}s) exceeded",
                return_code=-1,
            )
            logger.warning("sudo TIMEOUT: %s", command[:80])

        except OSError as exc:
            result = SudoResult(
                command=command, success=False,
                stderr=str(exc),
                return_code=-1,
            )
            logger.error("sudo OS error: %s — %s", command[:80], exc)

        self._log_op(command, result)
        return result

    def _log_op(self, command: str, result: SudoResult) -> None:
        """Append to audit log — NEVER include password."""
        self._execution_log.append({
            "command": command,
            "success": result.success,
            "blocked": result.blocked,
            "blocked_reason": result.blocked_reason,
            "return_code": result.return_code,
            "dry_run": result.dry_run,
        })

    def __repr__(self) -> str:
        return f"SudoHandler(available={self.is_available}, ops={len(self._execution_log)})"

    def __str__(self) -> str:
        return self.__repr__()
