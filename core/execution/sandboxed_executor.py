# core/execution/sandboxed_executor.py — Ariaska RL Sandboxed Command Executor
# Safe command execution layer for real target testing (Metasploitable).
# Enforces scope, timeouts, and blocklists to prevent accidental damage.

"""
Sandboxed Executor for Ariaska RL.

Provides safe command execution against real targets (e.g., Metasploitable)
with scope enforcement, command validation, timeout control, and audit logging.

Usage:
    executor = SandboxedExecutor(
        allowed_targets=["192.168.56.101"],
        mode="live",
        timeout=30,
    )
    result = executor.execute("nmap -sV 192.168.56.101")
"""

import logging
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Execution mode for the sandbox."""
    SIMULATED = "simulated"  # No real execution, return simulated output
    LIVE = "live"            # Execute against real targets
    DRY_RUN = "dry_run"     # Validate but don't execute


@dataclass
class ExecutionResult:
    """Result of a sandboxed command execution."""
    command: str
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    timed_out: bool = False
    blocked: bool = False
    block_reason: str = ""
    execution_time: float = 0.0
    mode: str = "simulated"
    target: str = ""
    timestamp: float = field(default_factory=time.time)

    @property
    def success(self) -> bool:
        """Whether the command executed successfully."""
        return self.exit_code == 0 and not self.timed_out and not self.blocked

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for logging."""
        return {
            "command": self.command,
            "stdout_length": len(self.stdout),
            "stderr_length": len(self.stderr),
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "blocked": self.blocked,
            "block_reason": self.block_reason,
            "execution_time": round(self.execution_time, 3),
            "mode": self.mode,
            "target": self.target,
            "success": self.success,
        }


class SandboxedExecutor:
    """
    Sandboxed command executor for Ariaska RL.

    Enforces:
    - Target scope: Only allowed IPs/subnets
    - Command blocklist: Prevents dangerous commands (rm -rf, dd, mkfs, etc.)
    - Timeout: Kills long-running commands
    - Audit log: Every execution is logged
    - Rate limiting: Prevents command flooding

    Args:
        allowed_targets: List of allowed target IPs/subnets
        mode: Execution mode (simulated, live, dry_run)
        timeout: Max seconds per command
        max_output_bytes: Truncate output beyond this size
        rate_limit_per_minute: Max commands per minute
        audit_log_path: Path to audit log file
    """

    # Commands that are NEVER allowed regardless of context
    BLOCKED_COMMANDS = {
        "rm", "rmdir", "mkfs", "dd", "format", "fdisk",
        "shutdown", "reboot", "halt", "poweroff", "init",
        "iptables", "ip6tables", "ufw",  # Don't mess with host firewall
        "systemctl", "service",  # Don't manage host services
        "kill", "killall", "pkill",  # Don't kill host processes
        "chmod 777", "chown",  # Don't change host permissions
    }

    # Phase 6.5: sshpass handles SSH password auth automatically.
    # sudo/su commands execute on target via ingreslock, not on host.

    # Patterns that indicate out-of-scope targeting
    SCOPE_VIOLATION_PATTERNS = [
        r"127\.0\.0\.1",  # localhost
        r"0\.0\.0\.0",    # all interfaces
        r"::1",           # IPv6 localhost
    ]

    # Allowed pentesting tools
    ALLOWED_TOOLS = {
        "nmap", "masscan", "rustscan",  # Scanning
        "gobuster", "dirb", "dirsearch", "feroxbuster", "ffuf", "wfuzz",  # Web enum
        "nikto", "whatweb", "wpscan", "nuclei",  # Web vuln scanning
        "enum4linux", "smbclient", "smbmap", "rpcclient",  # SMB
        "hydra", "medusa", "crackmapexec", "ncrack",  # Brute force
        "msfconsole", "msfvenom",  # Metasploit
        "searchsploit",  # Exploit search
        "curl", "wget",  # HTTP
        "ssh", "sshpass", "scp", "ftp", "telnet", "nc", "ncat",  # Connectivity
        "python", "python3", "perl", "ruby",  # Scripting
        "cat", "ls", "id", "whoami", "uname", "hostname",  # Recon on target
        "grep", "find", "locate", "which",  # Search
        "netstat", "ss", "ifconfig", "ip",  # Network info
        "echo", "printf",  # Output
        "base64", "xxd", "hexdump",  # Encoding
        "john", "hashcat",  # Password cracking
        "tcpdump", "wireshark", "tshark",  # Packet capture
        "dig", "nslookup", "host", "whois",  # DNS
        "snmpwalk", "snmpget",  # SNMP
        "ldapsearch",  # LDAP
        "impacket-smbexec", "impacket-wmiexec", "impacket-psexec",  # Impacket
    }

    def __init__(
        self,
        allowed_targets: Optional[List[str]] = None,
        mode: str = "simulated",
        timeout: int = 30,
        max_output_bytes: int = 50_000,
        rate_limit_per_minute: int = 30,
        audit_log_path: str = "logs/execution_audit.jsonl",
    ):
        self.allowed_targets: Set[str] = set(allowed_targets or [])
        self.mode = ExecutionMode(mode)
        self.timeout = timeout
        self.max_output_bytes = max_output_bytes
        self.rate_limit_per_minute = rate_limit_per_minute
        self.audit_log_path = audit_log_path

        # Rate limiting
        self._command_timestamps: List[float] = []

        # Stats
        self.total_executions = 0
        self.total_blocked = 0
        self.total_timeouts = 0

        # Ensure audit log directory exists
        os.makedirs(os.path.dirname(self.audit_log_path), exist_ok=True)

        logger.info(
            f"SandboxedExecutor initialized: mode={self.mode.value}, "
            f"targets={self.allowed_targets}, timeout={self.timeout}s"
        )

    def execute(self, command: str, target: Optional[str] = None) -> ExecutionResult:
        """
        Execute a command within the sandbox constraints.

        Args:
            command: The command to execute
            target: Target IP (for scope validation)

        Returns:
            ExecutionResult with output and metadata
        """
        self.total_executions += 1

        # Step 1: Validate command
        validation = self._validate_command(command, target)
        if not validation["valid"]:
            self.total_blocked += 1
            result = ExecutionResult(
                command=command,
                blocked=True,
                block_reason=validation["reason"],
                mode=self.mode.value,
                target=target or "",
            )
            self._audit_log(result)
            return result

        # Step 2: Check rate limit
        if not self._check_rate_limit():
            result = ExecutionResult(
                command=command,
                blocked=True,
                block_reason="Rate limit exceeded",
                mode=self.mode.value,
                target=target or "",
            )
            self._audit_log(result)
            return result

        # Step 3: Execute based on mode
        if self.mode == ExecutionMode.SIMULATED:
            result = self._execute_simulated(command, target)
        elif self.mode == ExecutionMode.DRY_RUN:
            result = ExecutionResult(
                command=command,
                stdout=f"[DRY_RUN] Would execute: {command}",
                mode="dry_run",
                target=target or "",
            )
        elif self.mode == ExecutionMode.LIVE:
            result = self._execute_live(command, target)
        else:
            result = ExecutionResult(
                command=command,
                blocked=True,
                block_reason=f"Unknown mode: {self.mode}",
            )

        self._audit_log(result)
        return result

    def _validate_command(self, command: str, target: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate a command before execution.

        Checks:
        1. Command is not empty
        2. Base command is in allowed tools list
        3. No blocked commands/patterns
        4. Target is in scope (if live mode)
        """
        if not command or not command.strip():
            return {"valid": False, "reason": "Empty command"}

        # Parse base command
        try:
            parts = shlex.split(command)
        except ValueError:
            parts = command.split()

        if not parts:
            return {"valid": False, "reason": "Could not parse command"}

        base_cmd = os.path.basename(parts[0]).lower()

        # Check blocked commands
        for blocked in self.BLOCKED_COMMANDS:
            if blocked in command.lower():
                return {"valid": False, "reason": f"Blocked command pattern: {blocked}"}

        # Phase 6.5: ssh/scp commands use sshpass for auto-password entry.
        # sudo/su commands are piped to remote target via ingreslock, never run on host.
        # stdin=DEVNULL in _execute_live() prevents any residual interactive prompts.

        # Check scope violations (localhost, etc.)
        for pattern in self.SCOPE_VIOLATION_PATTERNS:
            if re.search(pattern, command):
                return {"valid": False, "reason": f"Scope violation: targets localhost/self"}

        # In live mode, enforce tool allowlist and target scope
        if self.mode == ExecutionMode.LIVE:
            if base_cmd not in self.ALLOWED_TOOLS:
                return {"valid": False, "reason": f"Tool not in allowlist: {base_cmd}"}

            # Validate target scope
            if target and self.allowed_targets and target not in self.allowed_targets:
                return {"valid": False, "reason": f"Target {target} not in scope: {self.allowed_targets}"}

            # Check for IPs in command that aren't in scope
            ip_pattern = r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b'
            found_ips = set(re.findall(ip_pattern, command))
            if found_ips and self.allowed_targets:
                out_of_scope = found_ips - self.allowed_targets
                if out_of_scope:
                    return {
                        "valid": False,
                        "reason": f"IP(s) out of scope: {out_of_scope}",
                    }

        return {"valid": True, "reason": ""}

    def _execute_live(self, command: str, target: Optional[str] = None) -> ExecutionResult:
        """
        Execute a command in a real subprocess with timeout.

        Args:
            command: Validated command to execute
            target: Target IP

        Returns:
            ExecutionResult with real output
        """
        start_time = time.time()

        try:
            process = subprocess.Popen(
                command,
                shell=True,
                stdin=subprocess.DEVNULL,  # Phase 6.5: prevent interactive prompts (ssh, sudo, scp)
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env={
                    **os.environ,
                    "TERM": "dumb",  # Prevent color codes
                },
            )

            try:
                stdout, stderr = process.communicate(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                self.total_timeouts += 1
                return ExecutionResult(
                    command=command,
                    stdout="",
                    stderr="Command timed out",
                    exit_code=-1,
                    timed_out=True,
                    execution_time=self.timeout,
                    mode="live",
                    target=target or "",
                )

            # Truncate output if too large
            if len(stdout) > self.max_output_bytes:
                stdout = stdout[: self.max_output_bytes] + "\n... [OUTPUT TRUNCATED]"
            if len(stderr) > self.max_output_bytes:
                stderr = stderr[: self.max_output_bytes] + "\n... [OUTPUT TRUNCATED]"

            return ExecutionResult(
                command=command,
                stdout=stdout,
                stderr=stderr,
                exit_code=process.returncode,
                execution_time=time.time() - start_time,
                mode="live",
                target=target or "",
            )

        except Exception as e:
            return ExecutionResult(
                command=command,
                stderr=f"Execution error: {str(e)}",
                exit_code=-1,
                execution_time=time.time() - start_time,
                mode="live",
                target=target or "",
            )

    def _execute_simulated(self, command: str, target: Optional[str] = None) -> ExecutionResult:
        """
        Return simulated output (delegates to env's simulation).

        In simulated mode, no real commands are executed.
        The caller should feed the result to the output parser.
        """
        return ExecutionResult(
            command=command,
            stdout=f"[SIMULATED] {command}",
            mode="simulated",
            target=target or "",
        )

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        now = time.time()
        cutoff = now - 60  # 1-minute window
        self._command_timestamps = [t for t in self._command_timestamps if t > cutoff]
        if len(self._command_timestamps) >= self.rate_limit_per_minute:
            return False
        self._command_timestamps.append(now)
        return True

    def _audit_log(self, result: ExecutionResult):
        """Append execution result to audit log."""
        import json

        try:
            with open(self.audit_log_path, "a") as f:
                f.write(json.dumps(result.to_dict()) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write audit log: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            "total_executions": self.total_executions,
            "total_blocked": self.total_blocked,
            "total_timeouts": self.total_timeouts,
            "mode": self.mode.value,
            "allowed_targets": list(self.allowed_targets),
        }

    def add_target(self, target: str):
        """Add a target IP to the allowed scope."""
        self.allowed_targets.add(target)
        logger.info(f"Added target to scope: {target}")

    def remove_target(self, target: str):
        """Remove a target IP from the allowed scope."""
        self.allowed_targets.discard(target)
        logger.info(f"Removed target from scope: {target}")
