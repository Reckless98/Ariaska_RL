"""
core/ops/command_lockout.py — Phase 38.3: Command Lockout Engine

Tracks command execution results and locks out commands after
repeated failures.  Prevents wasting steps on proven-dead-end
commands while allowing cooldown-based retry.

Rules:
  - Commands locked out after LOCKOUT_THRESHOLD consecutive failures.
  - Lockout decays after LOCKOUT_DECAY_STEPS (can retry).
  - Distinct from anti-repeat: lockout is failure-based, anti-repeat
    is repetition-based.
  - Template-level lockout (not exact command string).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set

logger = logging.getLogger("ariaska.ops.command_lockout")

# ── Constants ────────────────────────────────────────────────────────────────

LOCKOUT_THRESHOLD: int = 3         # Consecutive failures before lockout
LOCKOUT_DECAY_STEPS: int = 15      # Steps until lockout expires
MAX_LOCKOUT_ENTRIES: int = 200     # Cap on tracked templates


@dataclass
class LockoutEntry:
    """Tracks failure history for a single command template."""
    template_name: str
    consecutive_failures: int = 0
    total_attempts: int = 0
    total_failures: int = 0
    locked_at_step: int = -1
    last_attempt_step: int = -1
    last_failure_reason: str = ""

    @property
    def is_locked(self) -> bool:
        return self.locked_at_step >= 0

    @property
    def success_rate(self) -> float:
        if self.total_attempts == 0:
            return 0.0
        return (self.total_attempts - self.total_failures) / self.total_attempts


class CommandLockout:
    """
    Failure-based command lockout engine.

    Usage:
        lockout = CommandLockout()
        if lockout.is_locked("vsftpd_backdoor", current_step=50):
            # Skip this command, it keeps failing
            pass
        else:
            # Execute command...
            lockout.record_result("vsftpd_backdoor", success=False, step=50,
                                  reason="Connection refused")
    """

    def __init__(
        self,
        threshold: int = LOCKOUT_THRESHOLD,
        decay_steps: int = LOCKOUT_DECAY_STEPS,
    ) -> None:
        self._threshold = threshold
        self._decay_steps = decay_steps
        self._entries: Dict[str, LockoutEntry] = {}
        logger.debug(
            "CommandLockout initialised (threshold=%d, decay=%d)",
            threshold, decay_steps,
        )

    def record_result(
        self,
        template_name: str,
        success: bool,
        step: int = 0,
        reason: str = "",
    ) -> bool:
        """
        Record a command execution result.

        Args:
            template_name: The command template name.
            success: Whether the command succeeded.
            step: Current step number.
            reason: Failure reason (if any).

        Returns:
            True if the command is now locked out.
        """
        if not template_name:
            return False

        entry = self._entries.get(template_name)
        if entry is None:
            entry = LockoutEntry(template_name=template_name)
            self._entries[template_name] = entry

        entry.total_attempts += 1
        entry.last_attempt_step = step

        if success:
            # Success resets consecutive failures and unlocks
            entry.consecutive_failures = 0
            if entry.is_locked:
                logger.info("Unlocked template '%s' after success", template_name)
                entry.locked_at_step = -1
            return False

        # Failure
        entry.consecutive_failures += 1
        entry.total_failures += 1
        entry.last_failure_reason = reason

        if entry.consecutive_failures >= self._threshold and not entry.is_locked:
            entry.locked_at_step = step
            logger.warning(
                "Locked template '%s' after %d consecutive failures: %s",
                template_name, entry.consecutive_failures, reason[:80],
            )
            return True

        return entry.is_locked

    def is_locked(self, template_name: str, current_step: int = 0) -> bool:
        """
        Check if a command template is locked out.

        Lockout expires after decay_steps.

        Args:
            template_name: The command template to check.
            current_step: Current step number for decay check.

        Returns:
            True if locked out.
        """
        entry = self._entries.get(template_name)
        if entry is None or not entry.is_locked:
            return False

        # Check decay
        steps_since_lockout = current_step - entry.locked_at_step
        if steps_since_lockout >= self._decay_steps:
            logger.info(
                "Lockout expired for '%s' after %d steps",
                template_name, steps_since_lockout,
            )
            entry.locked_at_step = -1
            entry.consecutive_failures = 0
            return False

        return True

    def get_locked_templates(self, current_step: int = 0) -> List[str]:
        """Return list of currently locked template names."""
        return [
            name for name, entry in self._entries.items()
            if entry.is_locked and not self._is_decayed(entry, current_step)
        ]

    def get_entry(self, template_name: str) -> Optional[LockoutEntry]:
        """Get lockout entry for a template (or None)."""
        return self._entries.get(template_name)

    def get_stats(self) -> Dict[str, Any]:
        """Return lockout statistics."""
        total = len(self._entries)
        locked = sum(1 for e in self._entries.values() if e.is_locked)
        return {
            "total_tracked": total,
            "currently_locked": locked,
            "avg_success_rate": (
                sum(e.success_rate for e in self._entries.values()) / total
                if total > 0 else 0.0
            ),
        }

    def _is_decayed(self, entry: LockoutEntry, current_step: int) -> bool:
        """Check if lockout has decayed."""
        if not entry.is_locked:
            return True
        return (current_step - entry.locked_at_step) >= self._decay_steps

    def reset(self) -> None:
        """Reset all lockout state."""
        self._entries.clear()
        logger.debug("CommandLockout reset")
