"""
core/execution/self_debug.py — Phase 28: GPT self-debug loop.

When a command fails with a recognisable error pattern, ask GPT-nano
for a corrected command and return it for a single retry.

Usage (inside SmartOrchestrator._run_step or SmartCoach post-exec):
    from core.execution.self_debug import SelfDebugger
    debugger = SelfDebugger(gpt_manager)
    fix = debugger.suggest_fix(command, output, phase)
    if fix.should_retry:
        ... execute fix.corrected_command ...
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from core.gpt_manager import GPTManager

logger = logging.getLogger("ariaska.self_debug")

# ── Recognisable error patterns that are fixable ─────────────────────────
_FIXABLE_PATTERNS = {
    "command not found": "tool_missing",
    "No such file or directory": "bad_path",
    "unknown option": "bad_flag",
    "unrecognized option": "bad_flag",
    "invalid option": "bad_flag",
    "Connection refused": "port_closed",
    "syntax error": "syntax",
    "unable to resolve host": "dns_fail",
    "Name or service not known": "dns_fail",
}

# Max output chars sent to GPT to keep token cost low
_MAX_OUTPUT_CHARS = 400


@dataclass
class DebugFix:
    """Result of a self-debug attempt."""
    should_retry: bool = False
    corrected_command: str = ""
    error_class: str = ""       # e.g. "bad_flag", "tool_missing"
    reasoning: str = ""
    original_command: str = ""
    original_error: str = ""


class SelfDebugger:
    """
    Lightweight GPT-powered command debugger.

    Asks nano to fix a failed command based on error output.
    Conservative: only fires on recognisable error classes,
    returns a single corrected command (no recursive loops).
    """

    def __init__(self, gpt_manager: Optional["GPTManager"] = None) -> None:
        self._gpt = gpt_manager
        self._fixes_attempted = 0
        self._fixes_succeeded = 0

    def suggest_fix(
        self,
        command: str,
        output: str,
        phase: str = "",
        target_ip: str = "",
    ) -> DebugFix:
        """
        Analyse a failed command and suggest a corrected version.

        Args:
            command: The command that failed.
            output: Combined stdout+stderr from the execution.
            phase: Current attack phase (for context).
            target_ip: Target IP for substitution.

        Returns:
            DebugFix — check .should_retry to see if a fix was found.
        """
        if not command or not output:
            return DebugFix()

        # 1. Classify the error
        error_class = self._classify_error(output)
        if not error_class:
            return DebugFix()

        # 2. If no GPT available, return a heuristic fix
        if self._gpt is None or not hasattr(self._gpt, "gpt_request"):
            return self._heuristic_fix(command, output, error_class, target_ip)

        # 3. Ask GPT-nano for a corrected command
        self._fixes_attempted += 1
        truncated_output = output[:_MAX_OUTPUT_CHARS]
        prompt = (
            f"A pentesting command failed. Fix it.\n"
            f"Command: {command}\n"
            f"Error ({error_class}): {truncated_output}\n"
            f"Phase: {phase}\n"
            f"Target: {target_ip}\n\n"
            f"Reply with ONLY the corrected command on a single line. "
            f"No explanation. If unfixable, reply UNFIXABLE."
        )

        try:
            response = self._gpt.gpt_request(
                prompt,
                task_type="classification",
                agent_id="SelfDebugger",
            )
        except Exception as exc:
            logger.debug(f"[SELF-DEBUG] GPT call failed: {exc}")
            return self._heuristic_fix(command, output, error_class, target_ip)

        if not response or not isinstance(response, str):
            return DebugFix()

        corrected = response.strip().split("\n")[0].strip()
        if not corrected or "UNFIXABLE" in corrected.upper():
            return DebugFix(
                error_class=error_class,
                original_command=command,
                original_error=truncated_output,
            )

        # Sanitize: strip markdown fences if GPT wraps the command
        corrected = re.sub(r"^```\w*\s*", "", corrected)
        corrected = re.sub(r"\s*```$", "", corrected)
        corrected = corrected.strip()

        if corrected and corrected != command:
            self._fixes_succeeded += 1
            logger.debug(
                f"[SELF-DEBUG] Fix ({error_class}): "
                f"{command[:50]} → {corrected[:50]}"
            )
            return DebugFix(
                should_retry=True,
                corrected_command=corrected,
                error_class=error_class,
                reasoning=f"GPT fix for {error_class}",
                original_command=command,
                original_error=truncated_output,
            )

        return DebugFix(
            error_class=error_class,
            original_command=command,
            original_error=truncated_output,
        )

    # ── Internals ─────────────────────────────────────────────────

    def _classify_error(self, output: str) -> str:
        """Match output against fixable patterns."""
        output_lower = output.lower()
        for pattern, cls in _FIXABLE_PATTERNS.items():
            if pattern.lower() in output_lower:
                return cls
        return ""

    def _heuristic_fix(
        self,
        command: str,
        output: str,
        error_class: str,
        target_ip: str,
    ) -> DebugFix:
        """Attempt a rule-based fix without GPT."""
        corrected = ""

        if error_class == "tool_missing":
            # Try common tool renames: nmapx → nmap, etc.
            parts = command.split()
            if parts:
                tool = parts[0]
                # Strip trailing 'x' or version suffix
                clean = re.sub(r"[xX]$", "", tool)
                if clean != tool:
                    parts[0] = clean
                    corrected = " ".join(parts)

        if error_class == "port_closed" and target_ip:
            # If target is wrong placeholder, substitute
            if "10.10.10.10" in command:
                corrected = command.replace("10.10.10.10", target_ip)

        if corrected and corrected != command:
            return DebugFix(
                should_retry=True,
                corrected_command=corrected,
                error_class=error_class,
                reasoning=f"Heuristic fix for {error_class}",
                original_command=command,
                original_error=output[:_MAX_OUTPUT_CHARS],
            )

        return DebugFix(
            error_class=error_class,
            original_command=command,
            original_error=output[:_MAX_OUTPUT_CHARS],
        )

    @property
    def stats(self) -> dict:
        """Return fix attempt statistics."""
        return {
            "fixes_attempted": self._fixes_attempted,
            "fixes_succeeded": self._fixes_succeeded,
            "fix_rate": (
                self._fixes_succeeded / max(1, self._fixes_attempted)
            ),
        }
