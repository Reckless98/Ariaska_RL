#!/usr/bin/env python3
"""
core/testing/tool_runner.py — ToolRunner abstraction with safety validation

Provides:
- Abstract ToolRunner protocol
- RealToolRunner with RFC1918 / allowlist target validation
- StubToolRunner for tests (no actual execution)

Usage:
    from core.testing import RealToolRunner, StubToolRunner
    
    # Production with safety checks:
    runner = RealToolRunner(allowed_targets=["10.10.10.0/24", "192.168.1.0/24"])
    result = runner.execute("nmap -sV 10.10.10.10")
    
    # Testing without execution:
    stub = StubToolRunner()
    result = stub.execute("nmap -sV 10.10.10.10")
    assert result.executed == False  # Stub never runs
"""

import ipaddress
import logging
import re
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set

logger = logging.getLogger(__name__)


# RFC1918 private address ranges
RFC1918_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
]

# Common HTB/CTF ranges (always safe in lab context)
LAB_NETWORKS = [
    ipaddress.ip_network("10.10.10.0/24"),   # HTB common
    ipaddress.ip_network("10.129.0.0/16"),   # HTB VPN
    ipaddress.ip_network("127.0.0.0/8"),     # localhost
]


@dataclass
class ToolResult:
    """Result from tool execution."""
    command: str
    executed: bool
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    blocked: bool = False
    blocked_reason: str = ""
    target_validated: bool = False
    targets_found: List[str] = field(default_factory=list)


class ToolRunner(ABC):
    """Abstract base class for tool execution."""
    
    @abstractmethod
    def execute(self, command: str, timeout: int = 30) -> ToolResult:
        """Execute a command and return the result."""
        pass
    
    @abstractmethod
    def validate_target(self, target: str) -> bool:
        """Check if a target is allowed."""
        pass
    
    @abstractmethod
    def is_safe_command(self, command: str) -> tuple[bool, str]:
        """Check if a command is safe to execute."""
        pass


class RealToolRunner(ToolRunner):
    """
    Production tool runner with safety validation.
    
    Features:
    - RFC1918 private range validation
    - Custom allowlist support
    - Hostname whitelist support (for HTB *.htb domains)
    - Command injection prevention
    - Timeout enforcement
    - Dangerous command blocking
    """
    
    # Commands that should never be executed
    BLOCKED_COMMANDS = {
        "rm -rf /",
        "rm -rf /*",
        "dd if=/dev/zero",
        "mkfs",
        ":(){:|:&};:",  # Fork bomb
        "chmod -R 777 /",
        "shutdown",
        "reboot",
        "init 0",
        "init 6",
    }
    
    # Patterns for dangerous commands
    DANGEROUS_PATTERNS = [
        r"rm\s+-r[f]?\s+/[^\.a-zA-Z]",  # rm -rf /something (but allow ./relative)
        r">\s*/dev/sd[a-z]",             # Overwrite disk
        r"dd\s+.*of=/dev/sd",            # DD to disk
        r"wget.*\|\s*bash",              # Pipe to bash
        r"curl.*\|\s*bash",              # Pipe to bash
    ]
    
    def __init__(
        self,
        allowed_targets: Optional[List[str]] = None,
        allow_rfc1918: bool = True,
        allow_lab_ranges: bool = True,
        allow_localhost: bool = True,
        blocked_commands: Optional[Set[str]] = None,
        allowed_hostnames: Optional[List[str]] = None,
    ):
        """
        Initialize RealToolRunner.
        
        Args:
            allowed_targets: List of allowed IP/CIDR targets
            allow_rfc1918: Allow all RFC1918 private addresses
            allow_lab_ranges: Allow common HTB/CTF ranges
            allow_localhost: Allow localhost/127.0.0.0/8
            blocked_commands: Additional commands to block
            allowed_hostnames: List of allowed hostname patterns (e.g. ["*.htb", "active.htb"])
        """
        self.allow_rfc1918 = allow_rfc1918
        self.allow_lab_ranges = allow_lab_ranges
        self.allow_localhost = allow_localhost
        
        # Build allowed networks
        self._allowed_networks: List[ipaddress.IPv4Network] = []
        
        if allowed_targets:
            for target in allowed_targets:
                try:
                    net = ipaddress.ip_network(target, strict=False)
                    self._allowed_networks.append(net)
                except ValueError:
                    logger.warning(f"Invalid target network: {target}")
        
        # Build allowed hostnames set (for HTB boxes with *.htb domains)
        self._allowed_hostnames: Set[str] = set()
        if allowed_hostnames:
            for h in allowed_hostnames:
                self._allowed_hostnames.add(h.lower())
        
        # Build blocked commands set
        self._blocked_commands = self.BLOCKED_COMMANDS.copy()
        if blocked_commands:
            self._blocked_commands.update(blocked_commands)
        
        # Compile dangerous patterns
        self._dangerous_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.DANGEROUS_PATTERNS
        ]
        
        # Track executions
        self._execution_history: List[ToolResult] = []
        
        logger.info(
            f"RealToolRunner initialized: rfc1918={allow_rfc1918}, "
            f"lab_ranges={allow_lab_ranges}, custom_targets={len(self._allowed_networks)}, "
            f"allowed_hostnames={self._allowed_hostnames}"
        )
    
    def validate_target(self, target: str) -> bool:
        """
        Check if a target IP/hostname is allowed.
        
        Args:
            target: IP address or hostname to validate
            
        Returns:
            True if target is allowed, False otherwise
        """
        # Handle hostnames
        if not self._is_ip_address(target):
            if target.lower() in ("localhost", "127.0.0.1", "::1"):
                return self.allow_localhost
            # Check allowed hostnames list (supports *.htb wildcards)
            target_lower = target.lower()
            for allowed_h in self._allowed_hostnames:
                if allowed_h.startswith("*."):
                    # Wildcard match: *.htb matches foo.htb
                    suffix = allowed_h[1:]  # .htb
                    if target_lower.endswith(suffix) or target_lower == allowed_h[2:]:
                        return True
                elif target_lower == allowed_h:
                    return True
            # Block non-IP hostnames by default for safety
            logger.warning(f"Hostname target not allowed: {target}")
            return False
        
        try:
            ip = ipaddress.ip_address(target)
        except ValueError:
            logger.warning(f"Invalid IP address: {target}")
            return False
        
        # Check localhost first - if disabled, block loopback early
        if ip.is_loopback:
            return self.allow_localhost
        
        # Check RFC1918
        if self.allow_rfc1918:
            for net in RFC1918_NETWORKS:
                if ip in net:
                    return True
        
        # Check lab ranges
        if self.allow_lab_ranges:
            for net in LAB_NETWORKS:
                if ip in net:
                    return True
        
        # Check custom allowlist
        for net in self._allowed_networks:
            if ip in net:
                return True
        
        logger.warning(f"Target not in allowlist: {target}")
        return False
    
    def is_safe_command(self, command: str) -> tuple[bool, str]:
        """
        Check if a command is safe to execute.
        
        Args:
            command: Command string to check
            
        Returns:
            Tuple of (is_safe, reason_if_blocked)
        """
        # Check blocked commands
        cmd_lower = command.lower().strip()
        for blocked in self._blocked_commands:
            if blocked.lower() in cmd_lower:
                return False, f"Blocked command pattern: {blocked}"
        
        # Check dangerous patterns
        for pattern in self._dangerous_patterns:
            if pattern.search(command):
                return False, f"Dangerous command pattern detected"
        
        # Extract and validate targets
        targets = self._extract_targets(command)
        for target in targets:
            if not self.validate_target(target):
                return False, f"Target not allowed: {target}"
        
        return True, ""
    
    def execute(self, command: str, timeout: int = 30) -> ToolResult:
        """
        Execute a command with safety validation.
        
        Args:
            command: Command to execute
            timeout: Timeout in seconds
            
        Returns:
            ToolResult with execution details
        """
        targets = self._extract_targets(command)
        
        # Safety check
        is_safe, reason = self.is_safe_command(command)
        if not is_safe:
            result = ToolResult(
                command=command,
                executed=False,
                blocked=True,
                blocked_reason=reason,
                targets_found=targets
            )
            self._execution_history.append(result)
            logger.warning(f"Blocked command: {command} - {reason}")
            return result
        
        # Execute the command
        try:
            proc = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                stdin=subprocess.DEVNULL,  # Phase 6.5: prevent interactive prompts
                start_new_session=True,  # Detach from controlling tty to prevent SIGTTIN
            )
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Kill entire process group to clean up child processes (hydra threads, etc.)
                import os
                import signal
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    proc.kill()
                proc.wait()
                result = ToolResult(
                    command=command,
                    executed=True,
                    stderr=f"Command timed out after {timeout}s",
                    return_code=-1,
                    targets_found=targets
                )
                self._execution_history.append(result)
                return result
            result = ToolResult(
                command=command,
                executed=True,
                stdout=stdout,
                stderr=stderr,
                return_code=proc.returncode,
                target_validated=True,
                targets_found=targets
            )
        except Exception as e:
            result = ToolResult(
                command=command,
                executed=False,
                stderr=str(e),
                return_code=-1,
                targets_found=targets
            )
        
        self._execution_history.append(result)
        return result
    
    def get_history(self) -> List[ToolResult]:
        """Get execution history."""
        return self._execution_history.copy()
    
    def clear_history(self):
        """Clear execution history."""
        self._execution_history.clear()
    
    def _is_ip_address(self, target: str) -> bool:
        """Check if target looks like an IP address."""
        try:
            ipaddress.ip_address(target)
            return True
        except ValueError:
            return False
    
    def _extract_targets(self, command: str) -> List[str]:
        """
        Extract IP addresses from a command.
        
        Args:
            command: Command string
            
        Returns:
            List of IP addresses found
        """
        # IP address pattern (simplified)
        ip_pattern = r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b'
        matches = re.findall(ip_pattern, command)
        
        # Validate they're actual IPs
        valid_ips = []
        for match in matches:
            try:
                ipaddress.ip_address(match)
                valid_ips.append(match)
            except ValueError:
                pass
        
        return valid_ips


class StubToolRunner(ToolRunner):
    """
    Stub tool runner for testing - never executes commands.
    
    Features:
    - Records all commands for assertions
    - Provides configurable fake outputs
    - Never makes real system calls
    """
    
    # Default fake outputs by command type
    DEFAULT_OUTPUTS = {
        "nmap": "PORT   STATE SERVICE\n22/tcp open  ssh\n80/tcp open  http",
        "ping": "PING 10.10.10.10: 64 bytes, icmp_seq=1 ttl=64 time=1.23ms",
        "curl": "HTTP/1.1 200 OK\n<html>...</html>",
        "gobuster": "/index.html (Status: 200)\n/robots.txt (Status: 200)",
        "ssh": "SSH-2.0-OpenSSH_8.4",
        "nc": "Connection to 10.10.10.10 22 port [tcp/ssh] succeeded!",
        "whoami": "root",
        "id": "uid=0(root) gid=0(root) groups=0(root)",
    }
    
    def __init__(self, custom_outputs: Optional[Dict[str, str]] = None):
        """
        Initialize StubToolRunner.
        
        Args:
            custom_outputs: Custom outputs for specific command prefixes
        """
        self._outputs = {**self.DEFAULT_OUTPUTS}
        if custom_outputs:
            self._outputs.update(custom_outputs)
        
        self._commands: List[str] = []
        self._results: List[ToolResult] = []
        
        logger.info("StubToolRunner initialized (no real execution)")
    
    def validate_target(self, target: str) -> bool:
        """Always returns True for stub."""
        return True
    
    def is_safe_command(self, command: str) -> tuple[bool, str]:
        """Always returns safe for stub."""
        return True, ""
    
    def execute(self, command: str, timeout: int = 30) -> ToolResult:
        """
        Record command and return fake output.
        
        Args:
            command: Command to "execute"
            timeout: Ignored in stub
            
        Returns:
            ToolResult with fake output
        """
        self._commands.append(command)
        
        # Get fake output based on command prefix
        output = self._get_fake_output(command)
        
        result = ToolResult(
            command=command,
            executed=False,  # Stub never really executes
            stdout=output,
            stderr="",
            return_code=0,
            target_validated=True,
            targets_found=self._extract_targets(command)
        )
        
        self._results.append(result)
        return result
    
    def get_commands(self) -> List[str]:
        """Get all recorded commands for test assertions."""
        return self._commands.copy()
    
    def get_results(self) -> List[ToolResult]:
        """Get all results."""
        return self._results.copy()
    
    def clear(self):
        """Clear recorded commands and results."""
        self._commands.clear()
        self._results.clear()
    
    def set_output(self, prefix: str, output: str):
        """Set custom output for command prefix."""
        self._outputs[prefix] = output
    
    def _get_fake_output(self, command: str) -> str:
        """Get fake output for a command."""
        cmd_lower = command.lower().split()[0] if command.split() else ""
        
        for prefix, output in self._outputs.items():
            if cmd_lower.startswith(prefix.lower()):
                return output
        
        return f"[STUB] Executed: {command}"
    
    def _extract_targets(self, command: str) -> List[str]:
        """Extract IP addresses from command."""
        ip_pattern = r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b'
        return re.findall(ip_pattern, command)


def get_tool_runner(testing: bool = False, **kwargs) -> ToolRunner:
    """
    Factory function to get appropriate ToolRunner.
    
    Args:
        testing: If True, returns StubToolRunner
        **kwargs: Passed to RealToolRunner if not testing
        
    Returns:
        ToolRunner instance
    """
    if testing:
        return StubToolRunner()
    return RealToolRunner(**kwargs)
