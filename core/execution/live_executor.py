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
import os
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
    
    Credential Management:
        SSH auto-wrap uses credentials from the discovery_board (set via
        set_credentials()). Defaults to NO auto-wrap if no credentials
        are known — commands that need remote execution must include
        their own SSH wrapper or use discovered creds.
    
    Usage:
        executor = LiveCommandExecutor(target_ip="10.129.4.210")
        executor.set_credentials("nathan", "Buck3tH4TF0RM3!")
        result = executor.execute("id", "RedAgent")
        # → sshpass -p 'Buck3tH4TF0RM3!' ssh nathan@10.129.4.210 'id'
    """
    
    # Default timeouts per command category (seconds)
    COMMAND_TIMEOUTS = {
        "{": 20,            # Phase 6.5: piped ingreslock commands have built-in timeout 10 + sleep 2
        "nmap": 60,         # HTB: nmap -sC -sV needs 60s; -T2 stealth will timeout
        "masscan": 60,
        "nikto": 90,
        "gobuster": 60,
        "dirb": 60,
        "hydra": 25,        # R46: reduced from 45s — small wordlists only, 25s sufficient
        "medusa": 25,       # R46: match hydra reduction to 25s
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

    # Phase 28: Retry config for transient network errors
    MAX_RETRIES = 2              # up to 2 retries (3 total attempts)
    RETRY_BASE_DELAY = 1.5       # exponential backoff base (seconds)
    _TRANSIENT_PATTERNS = (
        "Connection refused",
        "Connection timed out",
        "No route to host",
        "Connection reset by peer",
        "Network is unreachable",
        "Resource temporarily unavailable",
    )
    
    def __init__(
        self,
        target_ip: str,
        allowed_targets: Optional[List[str]] = None,
        dry_run: bool = False,
        allowed_hostnames: Optional[List[str]] = None,
    ):
        """
        Initialize LiveCommandExecutor.
        
        Args:
            target_ip: Primary target IP (must be RFC1918 or HTB VPN)
            allowed_targets: Additional allowed IPs/CIDRs
            dry_run: If True, log commands but don't execute
            allowed_hostnames: Hostname patterns for HTB (e.g. ["*.htb"])
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
            allowed_hostnames=allowed_hostnames or [],
        )
        
        # ── Phase 17: Verify critical pentesting tools are installed ─
        self._available_tools: Dict[str, bool] = {}
        self._check_tool_availability()
        
        # ── Credential store for SSH auto-wrap ──────────────────────
        # Populated via set_credentials() when creds are discovered.
        # Multiple credential pairs supported (tried in order).
        self._ssh_credentials: List[Dict[str, str]] = []
        # Default credential (empty = no auto-wrap)
        self._default_ssh_user: Optional[str] = None
        self._default_ssh_pass: Optional[str] = None
        
        # Per-agent execution tracking
        self._agent_history: Dict[str, List[LiveCommandResult]] = {}
        self._total_executions = 0
        self._total_blocked = 0
        self._total_duration_ms = 0.0
        
        logger.debug(
            f"LiveCommandExecutor initialized: target={target_ip}, "
            f"dry_run={dry_run}"
        )
    
    # ── Phase 17: Tool availability verification ─────────────────
    _CRITICAL_TOOLS = [
        "nmap", "gobuster", "hydra", "sshpass", "nikto", "searchsploit",
        "msfconsole", "curl", "wget", "netcat", "mysql", "psql",
        "telnet", "ssh", "smbclient", "enum4linux",
    ]
    
    def _check_tool_availability(self) -> None:
        """Check which pentesting tools are installed on the system."""
        import shutil
        missing = []
        for tool in self._CRITICAL_TOOLS:
            available = shutil.which(tool) is not None
            self._available_tools[tool] = available
            if not available:
                missing.append(tool)
        if missing:
            logger.warning(
                f"[P17-TOOLS] Missing tools ({len(missing)}): {', '.join(missing)}. "
                f"Some commands may fail."
            )
        else:
            logger.debug(f"[P17-TOOLS] All {len(self._CRITICAL_TOOLS)} critical tools available.")
    
    def is_tool_available(self, tool_name: str) -> bool:
        """Check if a specific tool is available on the system."""
        return self._available_tools.get(tool_name, False)
    
    def get_missing_tools(self) -> List[str]:
        """Return list of missing critical tools."""
        return [t for t, avail in self._available_tools.items() if not avail]
    
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
        
        # Safety: Block commands that would run interactively on the local machine
        # Commands like 'sudo -l' without SSH wrapper will prompt for local password
        # Phase 42: If SUDO_PASSWORD is in env, allow sudo by piping password non-interactively
        _cmd_stripped = command.strip()
        _LOCAL_DANGER_PREFIXES = (
            "sudo ",       # Would prompt for local password
            "su ",         # Would prompt for local password
            "passwd",      # Would prompt interactively
            "visudo",      # Interactive editor
        )
        if _cmd_stripped.startswith(_LOCAL_DANGER_PREFIXES):
            # Only block if not wrapped in SSH/sshpass (which targets remote)
            _is_remote = any(w in command for w in ("ssh ", "sshpass ", "ssh -"))
            if not _is_remote:
                # Phase 42: Allow sudo if SUDO_PASSWORD is set — pipe it non-interactively
                _sudo_pw = os.environ.get("SUDO_PASSWORD")
                if _cmd_stripped.startswith("sudo ") and _sudo_pw:
                    # Rewrite: sudo <cmd> → echo "$pw" | sudo -S <cmd>
                    _inner_cmd = _cmd_stripped[5:]  # Strip "sudo " prefix
                    command = f'echo "{_sudo_pw}" | sudo -S {_inner_cmd}'
                    logger.debug(
                        f"[LIVE-SUDO] Injected SUDO_PASSWORD for: sudo {_inner_cmd[:50]}..."
                    )
                else:
                    return LiveCommandResult(
                        command=command,
                        agent_name=agent_name,
                        executed=False,
                        stderr=f"[BLOCKED] Cannot run '{_cmd_stripped[:30]}...' locally — needs remote shell or SUDO_PASSWORD env var",
                        target_ip=self.target_ip,
                    )
        
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
        
        # Execute via RealToolRunner (with Phase 28 retry for transient errors)
        result = self._execute_with_retry(command, agent_name, timeout)
        
        self._record(agent_name, result)
        
        # Log execution
        status = "BLOCKED" if result.blocked else ("OK" if result.success else "FAIL")
        output_preview = result.output[:80].replace("\n", " ") if result.output else "(empty)"
        logger.debug(
            f"[LIVE][{agent_name}] {status} ({result.duration_ms:.0f}ms) "
            f"cmd={command[:60]}... → {output_preview}"
        )
        
        return result
    
    def set_credentials(self, username: str, password: str, service: str = "ssh") -> None:
        """
        Register discovered credentials for SSH auto-wrap.
        
        Called by the orchestrator when credentials are discovered
        (e.g., from PCAP analysis, FTP sessions, brute-force).
        The most recently added credentials become the default for auto-wrap.
        
        Args:
            username: Discovered username
            password: Discovered password
            service: Service these creds were found for (ssh, ftp, etc.)
        """
        cred = {"username": username, "password": password, "service": service}
        # Avoid duplicates
        if cred not in self._ssh_credentials:
            self._ssh_credentials.append(cred)
            logger.debug(f"[CRED-STORE] Registered credential: {username}@{service}")
        # Always update default to latest
        self._default_ssh_user = username
        self._default_ssh_pass = password
    
    def get_credentials(self) -> List[Dict[str, str]]:
        """Return all discovered credentials."""
        return list(self._ssh_credentials)
    
    def _inject_target(self, command: str) -> str:
        """
        Ensure the target IP is present in the command.
        
        Replaces common placeholders:
        - {target} → actual IP
        - 10.10.10.10 → actual IP (default placeholder)
        
        SSH auto-wrap: If the command is a local post-exploitation command
        (sudo, find, getcap, etc.) and we have discovered credentials,
        wraps it with sshpass+ssh using those credentials. If NO credentials
        are known, the command is NOT wrapped (it would fail anyway).
        """
        command = command.replace("{target}", self.target_ip)
        command = command.replace("10.10.10.10", self.target_ip)
        
        # ── Auto-wrap local-only post-exploitation commands ──
        # Commands like sudo -l, find -perm, getcap, cat /etc/shadow etc.
        # are meant to run ON THE TARGET, not locally on the Kali host.
        # Only wrap if we have discovered SSH credentials.
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
                "python3 -c", "python -c",  # For cap_setuid exploitation
                "perl -e", "ruby -e",       # Alternative cap exploitation
                "/usr/bin/python", "/usr/bin/perl",  # Absolute paths for cap_setuid
                "cat /home/", "cat /root/",  # Reading flags
            )
            cmd_stripped = command.strip()
            if any(cmd_stripped.startswith(p) for p in _LOCAL_PRIVESC_PREFIXES):
                if self._default_ssh_user and self._default_ssh_pass:
                    # Escape single quotes in the command for safe ssh wrapping
                    escaped_cmd = cmd_stripped.replace("'", "'\\''")
                    # Escape single quotes in password too
                    escaped_pass = self._default_ssh_pass.replace("'", "'\\''")
                    command = (
                        f"sshpass -p '{escaped_pass}' ssh -o StrictHostKeyChecking=no "
                        f"-o HostKeyAlgorithms=+ssh-rsa "
                        f"-o KexAlgorithms=+diffie-hellman-group1-sha1 "
                        f"-o ConnectTimeout=5 {self._default_ssh_user}@{self.target_ip} "
                        f"'{escaped_cmd}'"
                    )
                    logger.debug(
                        f"[LIVE-AUTOWRAP] Wrapped with discovered creds "
                        f"({self._default_ssh_user}): {cmd_stripped[:50]}..."
                    )
                else:
                    logger.debug(
                        f"[LIVE-NOWRAP] No SSH credentials known — "
                        f"cannot wrap: {cmd_stripped[:50]}..."
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

    # ── Phase 28: Retry logic for transient network errors ───────

    def _is_transient_failure(self, result: LiveCommandResult) -> bool:
        """Check if a failure looks transient and worth retrying."""
        if result.blocked or result.success:
            return False
        combined = (result.stdout + " " + result.stderr).strip()
        return any(p.lower() in combined.lower() for p in self._TRANSIENT_PATTERNS)

    def _execute_with_retry(
        self,
        command: str,
        agent_name: str,
        timeout: int,
    ) -> LiveCommandResult:
        """Execute with exponential-backoff retry for transient errors."""
        last_result: Optional[LiveCommandResult] = None
        for attempt in range(self.MAX_RETRIES + 1):
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
            last_result = result

            if result.success or result.blocked or not self._is_transient_failure(result):
                break  # non-transient — stop immediately

            if attempt < self.MAX_RETRIES:
                delay = self.RETRY_BASE_DELAY * (2 ** attempt)
                logger.debug(
                    f"[LIVE-RETRY] Transient failure on attempt {attempt + 1}, "
                    f"retrying in {delay:.1f}s: {command[:60]}"
                )
                time.sleep(delay)

        assert last_result is not None
        return last_result
