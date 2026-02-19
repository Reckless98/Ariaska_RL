"""
core/ops/tool_installer.py — Automated tool installation for OPS pre-flight

Wraps the existing ToolRegistry with:
  - Pre-flight batch verification.
  - LLM-assisted install fallback via GPTManager (codex contract prompt).
  - SudoHandler integration for privileged installs.
  - Respects per-episode / per-run install caps.
  - Feature-flag gated: FF_ALLOW_LIVE_INSTALL.
"""

from __future__ import annotations

import logging
import os
import subprocess
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger("ariaska.ops.tool_installer")

# Contract prompt for codex to generate install commands
_INSTALL_CONTRACT = """CONTRACT:
- Output ONLY valid shell install commands.
- No commentary.
- No markdown.
- Debian/Ubuntu/Kali system.
- Non-interactive (use -y flags for apt).
- Include verification command at the end (command -v <tool>).
- One command per line.
- Tool to install: {tool_name}
"""

# Commands allowed in LLM-generated install scripts
_ALLOWED_INSTALL_PATTERNS = re.compile(
    r"^(apt(-get)?\s+(install|update)|pip3?\s+install|pipx\s+install|"
    r"go\s+install|gem\s+install|command\s+-v|echo\s|which\s)",
    re.IGNORECASE,
)

# Dangerous patterns to reject from LLM output
_DANGEROUS_INSTALL_PATTERNS = re.compile(
    r"(rm\s|curl\s.*\|\s*(ba)?sh|wget\s.*\|\s*(ba)?sh|chmod\s+[0-7]*\s+/|"
    r"dd\s|mkfs|shutdown|reboot|>>\s*/etc/|python.*-c\s|eval\s)",
    re.IGNORECASE,
)


@dataclass
class InstallAttempt:
    """Result of a tool install attempt."""

    tool_name: str
    success: bool
    method: str = ""  # "registry", "llm_codex", "skipped"
    message: str = ""
    commands_executed: List[str] = field(default_factory=list)


class ToolInstaller:
    """
    Manages tool verification and installation for OPS pre-flight.

    Layers:
      1. Check existing ToolRegistry.
      2. If registered — delegate to ToolRegistry.install_tool().
      3. If not registered — use GPTManager codex to generate install.
      4. All sudo operations go through SudoHandler.

    Respects:
      - FF_ALLOW_LIVE_INSTALL flag.
      - MAX_INSTALLS_PER_EPISODE / MAX_INSTALLS_PER_RUN caps.
      - ARIASKA_DRY_RUN mode.
    """

    def __init__(self, sudo_handler: object) -> None:
        """
        Args:
            sudo_handler: SudoHandler instance for privileged installs.
        """
        self._sudo = sudo_handler
        self._is_dry_run = os.environ.get("ARIASKA_DRY_RUN", "") == "1"
        self._install_log: List[InstallAttempt] = []
        self._llm_install_count = 0
        self._MAX_LLM_INSTALLS = 3  # Max LLM-guided installs per session
        logger.debug("ToolInstaller initialised")

    # ── Public API ────────────────────────────────────────────────────────

    def verify_tool(self, tool_name: str) -> bool:
        """
        Check if a tool is available on the system.

        Args:
            tool_name: Name of the tool binary.

        Returns:
            True if tool is found in PATH.
        """
        if self._is_dry_run:
            return True  # Assume available in dry-run
        try:
            result = subprocess.run(
                f"command -v {tool_name}",
                shell=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False

    def pre_flight_check(self, required_tools: List[str]) -> Dict[str, bool]:
        """
        Batch-verify a list of tools.

        Args:
            required_tools: List of tool binary names.

        Returns:
            Dict mapping tool_name → is_available.
        """
        return {tool: self.verify_tool(tool) for tool in required_tools}

    def install_missing(
        self,
        tool_name: str,
        gpt_manager: Optional[object] = None,
    ) -> InstallAttempt:
        """
        Install a missing tool, trying registry first, then LLM-guided.

        Args:
            tool_name: Name of the tool to install.
            gpt_manager: Optional GPTManager for LLM-guided install fallback.

        Returns:
            InstallAttempt with result details.
        """
        # Already available?
        if self.verify_tool(tool_name):
            attempt = InstallAttempt(
                tool_name=tool_name, success=True,
                method="already_installed",
                message="Tool already available",
            )
            self._install_log.append(attempt)
            return attempt

        # Check feature flags
        from core.feature_flags import get_feature_flags
        ff = get_feature_flags()
        if not ff.allow_live_install:
            attempt = InstallAttempt(
                tool_name=tool_name, success=False,
                method="blocked",
                message="FF_ALLOW_LIVE_INSTALL is disabled",
            )
            self._install_log.append(attempt)
            return attempt

        # Try ToolRegistry first
        attempt = self._try_registry_install(tool_name, ff)
        if attempt.success:
            self._install_log.append(attempt)
            return attempt

        # Try LLM-guided install
        if gpt_manager and self._llm_install_count < self._MAX_LLM_INSTALLS:
            attempt = self._try_llm_install(tool_name, gpt_manager)
            self._install_log.append(attempt)
            return attempt

        # All methods exhausted
        attempt = InstallAttempt(
            tool_name=tool_name, success=False,
            method="exhausted",
            message="Neither registry nor LLM install succeeded",
        )
        self._install_log.append(attempt)
        return attempt

    def get_install_log(self) -> List[Dict]:
        """Return audit log of install attempts."""
        return [
            {
                "tool": a.tool_name,
                "success": a.success,
                "method": a.method,
                "message": a.message,
            }
            for a in self._install_log
        ]

    # ── Private ───────────────────────────────────────────────────────────

    def _try_registry_install(self, tool_name: str, ff: object) -> InstallAttempt:
        """Attempt install via ToolRegistry."""
        from core.tools.tool_registry import get_tool_registry
        registry = get_tool_registry()

        if not registry.can_live_install():
            return InstallAttempt(
                tool_name=tool_name, success=False,
                method="registry",
                message="Install cap reached",
            )

        result = registry.install_tool(
            name=tool_name,
            dry_run=self._is_dry_run,
            allow_sudo=ff.allow_sudo,
        )
        return InstallAttempt(
            tool_name=tool_name,
            success=result.success,
            method="registry",
            message=result.message,
        )

    def _try_llm_install(self, tool_name: str, gpt_manager: object) -> InstallAttempt:
        """Attempt LLM-guided install via GPTManager codex contract."""
        prompt = _INSTALL_CONTRACT.format(tool_name=tool_name)

        try:
            response = gpt_manager.gpt_request(
                prompt=prompt,
                task_type="tactical",
                agent_id="OPS_ToolInstaller",
            )
        except Exception as exc:
            logger.warning("LLM install request failed for %s: %s", tool_name, exc)
            return InstallAttempt(
                tool_name=tool_name, success=False,
                method="llm_codex",
                message=f"LLM request failed: {exc}",
            )

        if not response or not isinstance(response, str):
            return InstallAttempt(
                tool_name=tool_name, success=False,
                method="llm_codex",
                message="Empty LLM response",
            )

        # Parse and validate lines
        commands = self._parse_install_commands(response)
        if not commands:
            return InstallAttempt(
                tool_name=tool_name, success=False,
                method="llm_codex",
                message="No valid install commands in LLM response",
            )

        # Execute validated commands
        executed: List[str] = []
        for cmd in commands:
            if self._is_dry_run:
                logger.info("DRY_RUN: Would execute install cmd: %s", cmd)
                executed.append(cmd)
                continue

            # Determine if sudo required
            needs_sudo = cmd.strip().startswith(("apt", "apt-get"))
            if needs_sudo:
                result = self._sudo.execute_privileged(cmd)
                success = result.success
            else:
                try:
                    proc = subprocess.run(
                        cmd, shell=True, capture_output=True,
                        text=True, timeout=120,
                    )
                    success = proc.returncode == 0
                except (subprocess.TimeoutExpired, OSError):
                    success = False

            if success:
                executed.append(cmd)
            else:
                logger.warning("Install command failed: %s", cmd[:80])

        # Verify installation
        self._llm_install_count += 1
        verified = self.verify_tool(tool_name) if not self._is_dry_run else True
        return InstallAttempt(
            tool_name=tool_name,
            success=verified,
            method="llm_codex",
            message="Installed via LLM" if verified else "LLM install did not verify",
            commands_executed=executed,
        )

    @staticmethod
    def _parse_install_commands(response: str) -> List[str]:
        """Parse and validate install commands from LLM response."""
        commands = []
        for line in response.strip().splitlines():
            line = line.strip()
            # Skip empty, comments, markdown
            if not line or line.startswith("#") or line.startswith("```"):
                continue
            # Strip leading $ or > prompt markers
            if line.startswith("$ ") or line.startswith("> "):
                line = line[2:]
            # Validate against allow list
            if not _ALLOWED_INSTALL_PATTERNS.match(line):
                logger.debug("Rejected install line (not allowed): %s", line[:80])
                continue
            # Validate against danger list
            if _DANGEROUS_INSTALL_PATTERNS.search(line):
                logger.warning("Rejected dangerous install line: %s", line[:80])
                continue
            commands.append(line)
        return commands
