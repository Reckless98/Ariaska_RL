"""
LiveCommandExecutor — Per-agent real command execution for LIVE mode.

This module provides the execution bridge between SmartOrchestrator and
the real target (MS2/MS3/HTB). In LIVE mode, every agent command is
executed via subprocess against the real target. In SIM mode, this
module is NEVER imported or called.

Architecture:
    SmartOrchestrator._run_step()
        └─ if live_mode: LiveCommandExecutor.execute(command, agent_name)
               └─ RealToolRunner.execute(command, timeout)
                      └─ subprocess.run(...)
                             └─ LiveCommandResult(stdout, stderr, ...)

Author: Filip Volf
Phase: 6.1 — Environment Separation
"""

import time
import logging
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set

logger = logging.getLogger("ariaska.live_executor")


# ─── Output Channel Tags ────────────────────────────────────────────────
class OutputChannel:
    """Output channel identifiers for structured step output."""
    COMMAND = "COMMAND"          # The raw command string
    RAW_OUTPUT = "RAW_OUTPUT"    # Unmodified stdout + stderr from subprocess
    PARSED_VIEW = "PARSED_VIEW"  # Structured discoveries extracted from output
    META = "META"                # Execution metadata (duration, return_code, blocked, etc.)


@dataclass
class LiveCommandResult:
    """
    Result from executing a real command against a live target.
    
    This is the ONLY output representation in LIVE mode.
    No simulated fallback. If the command fails, stderr captures why.
    """
    command: str
    agent_name: str
    
    # Execution outcome
    executed: bool = False
    stdout: str = ""
    stderr: str = ""
    return_code: int = -1
    
    # Safety
    blocked: bool = False
    blocked_reason: str = ""
    
    # Timing
    duration_ms: float = 0.0
    
    # Target info
    target_ip: str = ""
    
    @property
    def output(self) -> str:
        """Combined output (stdout preferred, stderr as supplement)."""
        parts = []
        if self.stdout and self.stdout.strip():
            parts.append(self.stdout.strip())
        if self.stderr and self.stderr.strip():
            # Only include stderr if it has real content (not just warnings)
            parts.append(self.stderr.strip())
        return "\n".join(parts) if parts else ""
    
    @property
    def success(self) -> bool:
        """Whether the command executed successfully."""
        return self.executed and not self.blocked and self.return_code == 0
    
    @property
    def has_output(self) -> bool:
        """Whether there's any meaningful output."""
        return bool(self.stdout.strip() or self.stderr.strip())
    
    def to_channels(self) -> Dict[str, Any]:
        """Return structured output channels."""
        return {
            OutputChannel.COMMAND: self.command,
            OutputChannel.RAW_OUTPUT: self.output,
            OutputChannel.META: {
                "agent": self.agent_name,
                "executed": self.executed,
                "blocked": self.blocked,
                "blocked_reason": self.blocked_reason,
                "return_code": self.return_code,
                "duration_ms": self.duration_ms,
                "target_ip": self.target_ip,
                "success": self.success,
                "has_output": self.has_output,
            },
        }


@dataclass
class StepOutput:
    """
    Structured output for a single agent step.
    
    Contains all four output channels:
    - COMMAND: The raw command string
    - RAW_OUTPUT: Unmodified output from execution (live) or simulation
    - PARSED_VIEW: Structured discoveries extracted from output
    - META: Execution metadata
    
    In LIVE mode, RAW_OUTPUT comes from subprocess.
    In SIM mode, RAW_OUTPUT comes from _generate_simulated_output.
    These two paths NEVER mix.
    """
    command: str = ""
    raw_output: str = ""
    parsed_view: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    
    # Source tracking
    is_live: bool = False
    agent_name: str = ""
    
    @property
    def output_snippet(self) -> str:
        """First 200 chars of output for display."""
        if not self.raw_output:
            return "(no output)"
        lines = self.raw_output.strip().split("\n")
        if len(lines) <= 3:
            return self.raw_output.strip()[:200]
        return "\n".join(lines[:3]) + f"\n... ({len(lines)} lines total)"


class LiveCommandExecutor:
    """
    Executes real commands against a live target.
    
    This is the LIVE mode execution engine. It wraps RealToolRunner
    with per-agent tracking, timeout management, and structured output.
    
    In LIVE mode, this is the ONLY way commands are executed.
    _generate_simulated_output() is NEVER called.
    
    Usage:
        executor = LiveCommandExecutor(target_ip="172.28.0.10")
        result = executor.execute("nmap -sV -p 21,22,80 172.28.0.10", "ScoutAgent")
        print(result.stdout)  # Real nmap output
    """
    
    # Default timeouts per command category (seconds)
    COMMAND_TIMEOUTS = {
        "{": 20,            # Phase 6.5: piped ingreslock commands have built-in timeout 10 + sleep 2
        "nmap": 45,         # Phase 6.5: reduced from 120s — targeted scans should finish quickly
        "masscan": 60,
        "nikto": 90,
        "gobuster": 60,
        "dirb": 60,
        "hydra": 45,        # Batch 15: reduced from 120s — 45s enough for small wordlists, no burn
        "medusa": 45,       # Batch 15: match hydra reduction
        "sqlmap": 120,
        "searchsploit": 15,
        "msfconsole": 45,   # Phase 6.5: reduced from 180s — enough for module load + exploit fire
        "curl": 15,
        "wget": 15,
        "nc": 15,           # Phase 6.4: nc can take time when used as shell
        "ncat": 15,
        "telnet": 20,       # Phase 6.5: piped ingreslock commands include sleep 2
        "ssh": 20,
        "sshpass": 20,     # Phase 6.5: sshpass wraps ssh with auto-password
        "smbclient": 15,
        "rpcclient": 15,
        "showmount": 10,
        "rpcinfo": 10,
        "finger": 10,
        "mysql": 15,
        "psql": 15,
        "ftp": 15,
        "snmpwalk": 20,
        "enum4linux": 60,
        "wfuzz": 60,
        "ffuf": 60,
        "nuclei": 90,
        "whatweb": 15,
        "dig": 10,
        "host": 10,
        "whois": 15,
        "traceroute": 20,
        "ping": 10,
        "mount": 15,        # Phase 6.4: NFS mount
        "rsh": 10,          # Phase 6.4: rservices
        "rlogin": 10,
        "vncviewer": 15,
        "find": 15,         # Batch 15: find commands can hang on remote FS — fail fast
        "getcap": 15,       # Batch 15: getcap -r can scan entire FS — fail fast
        "script": 5,        # Batch 15: script -q opens interactive session — kill fast
        "(script": 5,       # Batch 15: subshell-wrapped script — kill fast
    }
    DEFAULT_TIMEOUT = 30
    
    def __init__(
        self,
        target_ip: str,
        allowed_targets: Optional[List[str]] = None,
        dry_run: bool = False,
    ):
        """
        Initialize LiveCommandExecutor.
        
        Args:
            target_ip: Primary target IP (must be RFC1918)
            allowed_targets: Additional allowed IPs/CIDRs
            dry_run: If True, log commands but don't execute
        """
        from core.testing.tool_runner import RealToolRunner
        
        self.target_ip = target_ip
        self.dry_run = dry_run
        
        # Build allowed targets list
        targets = [target_ip]
        if allowed_targets:
            targets.extend(allowed_targets)
        
        self._runner = RealToolRunner(
            allowed_targets=targets,
            allow_rfc1918=True,
            allow_lab_ranges=True,
        )
        
        # Per-agent execution tracking
        self._agent_history: Dict[str, List[LiveCommandResult]] = {}
        self._total_executions = 0
        self._total_blocked = 0
        self._total_duration_ms = 0.0
        
        logger.info(
            f"LiveCommandExecutor initialized: target={target_ip}, "
            f"dry_run={dry_run}"
        )
    
    def execute(
        self,
        command: str,
        agent_name: str,
        timeout_override: Optional[int] = None,
    ) -> LiveCommandResult:
        """
        Execute a real command against the live target.
        
        This is the ONLY execution path in LIVE mode.
        No simulated output is ever generated.
        
        Args:
            command: Shell command to execute
            agent_name: Name of the agent issuing the command
            timeout_override: Override the default timeout
            
        Returns:
            LiveCommandResult with real stdout/stderr
        """
        if not command or not command.strip():
            return LiveCommandResult(
                command=command or "",
                agent_name=agent_name,
                executed=False,
                stderr="Empty command",
                target_ip=self.target_ip,
            )
        
        # Ensure target IP is in command (substitute placeholder)
        command = self._inject_target(command)
        
        # Determine timeout
        timeout = timeout_override or self._get_timeout(command)
        
        # Dry run mode — log but don't execute
        if self.dry_run:
            result = LiveCommandResult(
                command=command,
                agent_name=agent_name,
                executed=False,
                stdout=f"[DRY_RUN] Would execute: {command}",
                target_ip=self.target_ip,
            )
            self._record(agent_name, result)
            return result
        
        # Execute via RealToolRunner
        start = time.monotonic()
        tool_result = self._runner.execute(command, timeout=timeout)
        elapsed_ms = (time.monotonic() - start) * 1000
        
        result = LiveCommandResult(
            command=command,
            agent_name=agent_name,
            executed=tool_result.executed,
            stdout=tool_result.stdout or "",
            stderr=tool_result.stderr or "",
            return_code=tool_result.return_code,
            blocked=tool_result.blocked,
            blocked_reason=tool_result.blocked_reason or "",
            duration_ms=elapsed_ms,
            target_ip=self.target_ip,
        )
        
        self._record(agent_name, result)
        
        # Log execution
        status = "BLOCKED" if result.blocked else ("OK" if result.success else "FAIL")
        output_preview = result.output[:80].replace("\n", " ") if result.output else "(empty)"
        logger.debug(
            f"[LIVE][{agent_name}] {status} ({elapsed_ms:.0f}ms) "
            f"cmd={command[:60]}... → {output_preview}"
        )
        
        return result
    
    def _inject_target(self, command: str) -> str:
        """
        Ensure the target IP is present in the command.
        
        Replaces common placeholders:
        - {target} → actual IP
        - 10.10.10.10 → actual IP (default placeholder)
        """
        command = command.replace("{target}", self.target_ip)
        command = command.replace("10.10.10.10", self.target_ip)
        
        # ── Batch 15: Auto-wrap local-only post-exploitation commands ──
        # Commands like sudo -l, find -perm, getcap, cat /etc/shadow etc.
        # are meant to run ON THE TARGET, not locally on the Kali host.
        # If the command doesn't reference the target IP and looks like a
        # local post-exploitation command, wrap it with sshpass+ssh to
        # execute on the target.
        if self.target_ip not in command and not command.strip().startswith(("{", "sshpass", "ssh ")):
            _LOCAL_PRIVESC_PREFIXES = (
                "sudo ", "find ", "getcap ", "cat /etc/", "ls -la /etc/",
                "id", "whoami", "uname ", "env ", "printenv",
                "crontab ", "docker ", "lxd ", "./linpeas", "linpeas",
                "pspy", "history ", "grep ", "awk ", "sed ",
                "chmod ", "chown ", "mount ", "df ", "ps ",
                "netstat ", "ss ", "ip addr", "ifconfig",
                "base64 /", "head /", "tail /", "xxd /",
                "strings /", "file /", "dpkg ", "apt ",
                "service ", "systemctl ", "journalctl ",
            )
            cmd_stripped = command.strip()
            if any(cmd_stripped.startswith(p) for p in _LOCAL_PRIVESC_PREFIXES):
                # Escape single quotes in the command for safe ssh wrapping
                escaped_cmd = cmd_stripped.replace("'", "'\\''")
                command = (
                    f"sshpass -p msfadmin ssh -o StrictHostKeyChecking=no "
                    f"-o ConnectTimeout=5 msfadmin@{self.target_ip} "
                    f"'{escaped_cmd}'"
                )
                logger.debug(
                    f"[LIVE-AUTOWRAP] Local command wrapped with sshpass: "
                    f"{cmd_stripped[:50]}..."
                )
        
        return command
    
    def _get_timeout(self, command: str) -> int:
        """Get appropriate timeout for a command based on its tool."""
        cmd_prefix = command.strip().split()[0].lower() if command.strip() else ""
        # Strip path prefix (e.g., /usr/bin/nmap → nmap)
        cmd_prefix = cmd_prefix.rsplit("/", 1)[-1]
        return self.COMMAND_TIMEOUTS.get(cmd_prefix, self.DEFAULT_TIMEOUT)
    
    def _record(self, agent_name: str, result: LiveCommandResult) -> None:
        """Record execution in per-agent history."""
        if agent_name not in self._agent_history:
            self._agent_history[agent_name] = []
        self._agent_history[agent_name].append(result)
        self._total_executions += 1
        if result.blocked:
            self._total_blocked += 1
        self._total_duration_ms += result.duration_ms
    
    def get_agent_history(self, agent_name: str) -> List[LiveCommandResult]:
        """Get execution history for a specific agent."""
        return self._agent_history.get(agent_name, [])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            "total_executions": self._total_executions,
            "total_blocked": self._total_blocked,
            "total_duration_ms": self._total_duration_ms,
            "avg_duration_ms": (
                self._total_duration_ms / self._total_executions
                if self._total_executions > 0 else 0.0
            ),
            "per_agent": {
                name: len(history)
                for name, history in self._agent_history.items()
            },
        }
    
    def reset_episode(self) -> None:
        """Reset per-episode tracking (keep cumulative stats)."""
        self._agent_history.clear()
