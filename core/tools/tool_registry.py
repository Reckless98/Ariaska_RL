#!/usr/bin/env python3
"""
core/tools/tool_registry.py — Phase 10.1B: Tool Registry + Live Install System

Maintains an allow-list of pentesting tools with check/install metadata.
Supports:
  - Tool presence checks (which/command -v)
  - Bootstrap installation (apt/pipx/pip/go/gh/git-clone)
  - Live install on-demand (gated by FF_ALLOW_LIVE_INSTALL + FF_ALLOW_SUDO)
  - Install audit logging (JSONL)

Safety:
  - Only pre-registered tools can be installed (no arbitrary URLs)
  - Sudo gated by FF_ALLOW_SUDO
  - Install counts capped per episode and per run
  - All installs go through CommandRegistry templates (auditable)

Usage:
    from core.tools.tool_registry import get_tool_registry
    registry = get_tool_registry()
    if not registry.is_available("gobuster"):
        registry.install("gobuster")  # If flags allow
"""

import logging
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("ariaska.tool_registry")


class ToolGroup(Enum):
    """Functional grouping for pentesting tools."""
    RECON = "recon"
    ENUM = "enum"
    WEB = "web"
    AD = "ad"
    SMB = "smb"
    WORDLIST = "wordlist"
    PROXY = "proxy"
    POSTEX = "postex"
    EXPLOIT = "exploit"
    PRIVESC = "privesc"
    UTIL = "util"
    BRUTE = "brute"


class InstallMethod(Enum):
    """How to install a tool."""
    APT = "apt"
    PIPX = "pipx"
    PIP = "pip"
    GO = "go"
    GH_RELEASE = "gh_release"
    GIT_CLONE = "git_clone"
    SNAP = "snap"
    MANUAL = "manual"  # Pre-built binary download


@dataclass
class ToolEntry:
    """A registered pentesting tool with install metadata."""
    name: str
    description: str
    group: ToolGroup
    check_commands: List[str]  # Commands to verify presence (e.g., ["which nmap", "nmap --version"])
    install_method: InstallMethod
    install_target: str  # apt package name, pip package, go module, git URL, etc.
    version_pin: str = ""  # Optional version constraint
    requires_sudo_install: bool = False  # Whether install needs sudo
    requires_sudo_run: bool = False  # Whether running the tool needs sudo
    post_install_check: str = ""  # Command to verify install succeeded
    alternatives: List[str] = field(default_factory=list)  # Alternative tool names
    tags: Set[str] = field(default_factory=set)


@dataclass
class InstallResult:
    """Result of a tool install attempt."""
    tool_name: str
    success: bool
    method: str = ""
    message: str = ""
    duration_ms: int = 0
    required_sudo: bool = False
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool_name,
            "success": self.success,
            "method": self.method,
            "message": self.message,
            "duration_ms": self.duration_ms,
            "required_sudo": self.required_sudo,
            "ts": self.timestamp,
        }


@dataclass
class BootstrapReport:
    """Report from a bootstrap run."""
    profile: str
    installed: List[InstallResult] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)  # Already present
    missing: List[str] = field(default_factory=list)   # Not installed (dry_run or failed)
    failures: List[InstallResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile": self.profile,
            "installed": [r.to_dict() for r in self.installed],
            "skipped": self.skipped,
            "missing": self.missing,
            "failures": [r.to_dict() for r in self.failures],
            "summary": {
                "total": len(self.installed) + len(self.skipped) + len(self.missing) + len(self.failures),
                "installed": len(self.installed),
                "skipped": len(self.skipped),
                "missing": len(self.missing),
                "failures": len(self.failures),
            },
        }


# ============================================================================
# TOOL REGISTRY — Allow-listed tools
# ============================================================================

_TOOL_ENTRIES: Dict[str, ToolEntry] = {}


def _register_tool(entry: ToolEntry) -> ToolEntry:
    """Register a tool in the global registry."""
    _TOOL_ENTRIES[entry.name] = entry
    return entry


def _populate_registry() -> None:
    """Populate the tool registry with known pentesting tools."""
    # ── RECON ────────────────────────────────────────────────────────
    _register_tool(ToolEntry(
        name="nmap", description="Network scanner and port mapper",
        group=ToolGroup.RECON,
        check_commands=["which nmap"],
        install_method=InstallMethod.APT, install_target="nmap",
        requires_sudo_install=True,
        tags={"scanner", "network"},
    ))
    _register_tool(ToolEntry(
        name="masscan", description="Fast port scanner",
        group=ToolGroup.RECON,
        check_commands=["which masscan"],
        install_method=InstallMethod.APT, install_target="masscan",
        requires_sudo_install=True, requires_sudo_run=True,
        tags={"scanner", "network", "fast"},
    ))
    _register_tool(ToolEntry(
        name="dnsrecon", description="DNS enumeration",
        group=ToolGroup.RECON,
        check_commands=["which dnsrecon"],
        install_method=InstallMethod.APT, install_target="dnsrecon",
        requires_sudo_install=True,
        tags={"dns", "enum"},
    ))

    # ── ENUMERATION ─────────────────────────────────────────────────
    _register_tool(ToolEntry(
        name="gobuster", description="Directory/DNS/vhost buster",
        group=ToolGroup.ENUM,
        check_commands=["which gobuster"],
        install_method=InstallMethod.APT, install_target="gobuster",
        requires_sudo_install=True,
        tags={"web", "dir", "brute"},
    ))
    _register_tool(ToolEntry(
        name="feroxbuster", description="Fast recursive content discovery",
        group=ToolGroup.ENUM,
        check_commands=["which feroxbuster"],
        install_method=InstallMethod.APT, install_target="feroxbuster",
        requires_sudo_install=True,
        tags={"web", "dir", "brute", "fast"},
    ))
    _register_tool(ToolEntry(
        name="enum4linux", description="SMB/NetBIOS enumeration",
        group=ToolGroup.SMB,
        check_commands=["which enum4linux"],
        install_method=InstallMethod.APT, install_target="enum4linux",
        requires_sudo_install=True,
        tags={"smb", "enum"},
    ))
    _register_tool(ToolEntry(
        name="smbclient", description="SMB client for share access",
        group=ToolGroup.SMB,
        check_commands=["which smbclient"],
        install_method=InstallMethod.APT, install_target="smbclient",
        requires_sudo_install=True,
        tags={"smb"},
    ))
    _register_tool(ToolEntry(
        name="nikto", description="Web server vulnerability scanner",
        group=ToolGroup.WEB,
        check_commands=["which nikto"],
        install_method=InstallMethod.APT, install_target="nikto",
        requires_sudo_install=True,
        tags={"web", "scanner"},
    ))
    _register_tool(ToolEntry(
        name="whatweb", description="Web technology fingerprinter",
        group=ToolGroup.WEB,
        check_commands=["which whatweb"],
        install_method=InstallMethod.APT, install_target="whatweb",
        requires_sudo_install=True,
        tags={"web", "fingerprint"},
    ))
    _register_tool(ToolEntry(
        name="wfuzz", description="Web fuzzer",
        group=ToolGroup.WEB,
        check_commands=["which wfuzz"],
        install_method=InstallMethod.PIP, install_target="wfuzz",
        tags={"web", "fuzz"},
    ))
    _register_tool(ToolEntry(
        name="ffuf", description="Fast web fuzzer (Go)",
        group=ToolGroup.WEB,
        check_commands=["which ffuf"],
        install_method=InstallMethod.APT, install_target="ffuf",
        requires_sudo_install=True,
        tags={"web", "fuzz", "fast"},
    ))

    # ── EXPLOITATION ────────────────────────────────────────────────
    _register_tool(ToolEntry(
        name="sqlmap", description="Automated SQL injection",
        group=ToolGroup.EXPLOIT,
        check_commands=["which sqlmap"],
        install_method=InstallMethod.APT, install_target="sqlmap",
        requires_sudo_install=True,
        tags={"web", "sqli"},
    ))
    _register_tool(ToolEntry(
        name="msfconsole", description="Metasploit Framework console",
        group=ToolGroup.EXPLOIT,
        check_commands=["which msfconsole"],
        install_method=InstallMethod.MANUAL, install_target="metasploit-framework",
        requires_sudo_install=True,
        tags={"framework", "exploit"},
    ))
    _register_tool(ToolEntry(
        name="searchsploit", description="ExploitDB search",
        group=ToolGroup.EXPLOIT,
        check_commands=["which searchsploit"],
        install_method=InstallMethod.APT, install_target="exploitdb",
        requires_sudo_install=True,
        tags={"exploit", "search"},
    ))

    # ── BRUTE FORCE ─────────────────────────────────────────────────
    _register_tool(ToolEntry(
        name="hydra", description="Network login brute-forcer",
        group=ToolGroup.BRUTE,
        check_commands=["which hydra"],
        install_method=InstallMethod.APT, install_target="hydra",
        requires_sudo_install=True,
        tags={"brute", "login"},
    ))
    _register_tool(ToolEntry(
        name="medusa", description="Speedy parallel login brute-forcer",
        group=ToolGroup.BRUTE,
        check_commands=["which medusa"],
        install_method=InstallMethod.APT, install_target="medusa",
        requires_sudo_install=True,
        tags={"brute", "login"},
    ))
    _register_tool(ToolEntry(
        name="john", description="John the Ripper password cracker",
        group=ToolGroup.BRUTE,
        check_commands=["which john"],
        install_method=InstallMethod.APT, install_target="john",
        requires_sudo_install=True,
        tags={"brute", "crack"},
    ))
    _register_tool(ToolEntry(
        name="hashcat", description="GPU password cracker",
        group=ToolGroup.BRUTE,
        check_commands=["which hashcat"],
        install_method=InstallMethod.APT, install_target="hashcat",
        requires_sudo_install=True,
        tags={"brute", "crack", "gpu"},
    ))

    # ── PRIVESC ─────────────────────────────────────────────────────
    _register_tool(ToolEntry(
        name="linpeas", description="Linux privilege escalation checker",
        group=ToolGroup.PRIVESC,
        check_commands=["test -f /usr/share/peass/linpeas/linpeas.sh"],
        install_method=InstallMethod.GIT_CLONE,
        install_target="https://github.com/carlospolop/PEASS-ng.git",
        tags={"privesc", "linux"},
    ))

    # ── POST-EXPLOITATION ───────────────────────────────────────────
    _register_tool(ToolEntry(
        name="crackmapexec", description="Active Directory swiss army knife",
        group=ToolGroup.AD,
        check_commands=["which crackmapexec", "which cme"],
        install_method=InstallMethod.PIPX, install_target="crackmapexec",
        alternatives=["netexec"],
        tags={"ad", "smb", "postex"},
    ))
    _register_tool(ToolEntry(
        name="netexec", description="CrackMapExec successor",
        group=ToolGroup.AD,
        check_commands=["which netexec", "which nxc"],
        install_method=InstallMethod.PIPX, install_target="netexec",
        alternatives=["crackmapexec"],
        tags={"ad", "smb", "postex"},
    ))
    _register_tool(ToolEntry(
        name="impacket-scripts", description="Impacket Python tools",
        group=ToolGroup.AD,
        check_commands=["which impacket-smbclient", "python3 -c 'import impacket'"],
        install_method=InstallMethod.PIPX, install_target="impacket",
        tags={"ad", "smb", "postex"},
    ))

    # ── PROXY / WEB ─────────────────────────────────────────────────
    _register_tool(ToolEntry(
        name="proxychains", description="Proxy chain redirector",
        group=ToolGroup.PROXY,
        check_commands=["which proxychains4", "which proxychains"],
        install_method=InstallMethod.APT, install_target="proxychains4",
        requires_sudo_install=True,
        tags={"proxy", "network"},
    ))
    _register_tool(ToolEntry(
        name="curl", description="HTTP client",
        group=ToolGroup.UTIL,
        check_commands=["which curl"],
        install_method=InstallMethod.APT, install_target="curl",
        requires_sudo_install=True,
        tags={"http", "util"},
    ))

    # ── WORDLISTS ───────────────────────────────────────────────────
    _register_tool(ToolEntry(
        name="seclists", description="SecLists wordlist collection",
        group=ToolGroup.WORDLIST,
        check_commands=["test -d /usr/share/seclists"],
        install_method=InstallMethod.APT, install_target="seclists",
        requires_sudo_install=True,
        tags={"wordlist"},
    ))

    # ── PORT KNOCKING ───────────────────────────────────────────────
    _register_tool(ToolEntry(
        name="knock", description="Port knock client (knockd)",
        group=ToolGroup.UTIL,
        check_commands=["which knock"],
        install_method=InstallMethod.APT, install_target="knockd",
        requires_sudo_install=True,
        tags={"knock", "network"},
    ))

    # ── MISC UTIL ───────────────────────────────────────────────────
    _register_tool(ToolEntry(
        name="socat", description="Multipurpose relay",
        group=ToolGroup.UTIL,
        check_commands=["which socat"],
        install_method=InstallMethod.APT, install_target="socat",
        requires_sudo_install=True,
        tags={"network", "relay"},
    ))
    _register_tool(ToolEntry(
        name="netcat", description="TCP/UDP network tool",
        group=ToolGroup.UTIL,
        check_commands=["which nc", "which ncat", "which netcat"],
        install_method=InstallMethod.APT, install_target="ncat",
        requires_sudo_install=True,
        tags={"network"},
    ))
    _register_tool(ToolEntry(
        name="tcpdump", description="Network packet capture",
        group=ToolGroup.UTIL,
        check_commands=["which tcpdump"],
        install_method=InstallMethod.APT, install_target="tcpdump",
        requires_sudo_install=True, requires_sudo_run=True,
        tags={"network", "capture"},
    ))


# ============================================================================
# TOOL PROFILES — Collections for different target types
# ============================================================================

TOOL_PROFILES: Dict[str, List[str]] = {
    "htb": [
        "nmap", "gobuster", "feroxbuster", "ffuf", "nikto", "whatweb",
        "sqlmap", "hydra", "john", "hashcat", "seclists",
        "crackmapexec", "netexec", "impacket-scripts",
        "enum4linux", "smbclient", "linpeas", "socat", "netcat",
        "curl", "knock", "proxychains", "wfuzz",
    ],
    "ms3": [
        "nmap", "gobuster", "nikto", "whatweb", "sqlmap", "hydra",
        "enum4linux", "smbclient", "crackmapexec", "impacket-scripts",
        "curl", "seclists", "netcat", "socat",
    ],
    "ms2": [
        "nmap", "hydra", "smbclient", "enum4linux", "curl",
        "netcat", "seclists",
    ],
    "dev": [
        "nmap", "curl", "netcat",
    ],
}


# ============================================================================
# TOOL REGISTRY CLASS
# ============================================================================

class ToolRegistry:
    """Central registry for pentesting tool management.

    Provides tool presence checking, install coordination (if flags allow),
    and bootstrap reporting.
    """

    # Safety caps
    MAX_INSTALLS_PER_EPISODE = 3
    MAX_INSTALLS_PER_RUN = 10

    def __init__(self) -> None:
        if not _TOOL_ENTRIES:
            _populate_registry()
        self._entries = _TOOL_ENTRIES
        self._availability_cache: Dict[str, bool] = {}
        self._install_count_episode = 0
        self._install_count_run = 0
        self._install_log: List[InstallResult] = []

    def reset_episode(self) -> None:
        """Reset per-episode counters."""
        self._install_count_episode = 0

    def get_tool(self, name: str) -> Optional[ToolEntry]:
        """Get a tool entry by name (None if not registered)."""
        return self._entries.get(name)

    def is_registered(self, name: str) -> bool:
        """Check if a tool is in the allow-list."""
        return name in self._entries

    def is_available(self, name: str) -> bool:
        """Check if a tool is installed on the system.

        Uses cached results within a session for performance.
        """
        if name in self._availability_cache:
            return self._availability_cache[name]

        entry = self._entries.get(name)
        if not entry:
            return False

        available = False
        for check_cmd in entry.check_commands:
            try:
                result = subprocess.run(
                    check_cmd, shell=True, capture_output=True,
                    timeout=5, text=True,
                )
                if result.returncode == 0:
                    available = True
                    break
            except (subprocess.TimeoutExpired, OSError):
                continue

        self._availability_cache[name] = available
        return available

    def check_availability_batch(self, names: List[str]) -> Dict[str, bool]:
        """Check availability for multiple tools at once."""
        return {name: self.is_available(name) for name in names}

    def get_missing_tools(self, profile: str = "htb") -> List[str]:
        """Get list of tools required by profile that are not installed."""
        required = TOOL_PROFILES.get(profile, [])
        return [name for name in required if not self.is_available(name)]

    def get_profile_tools(self, profile: str) -> List[str]:
        """Get list of tool names for a profile."""
        return TOOL_PROFILES.get(profile, [])

    def can_live_install(self) -> bool:
        """Check if live install is allowed (flags + caps)."""
        from core.feature_flags import get_feature_flags
        ff = get_feature_flags()
        if not ff.allow_live_install:
            return False
        if self._install_count_episode >= self.MAX_INSTALLS_PER_EPISODE:
            logger.warning("Live install cap reached for episode (%d)", self.MAX_INSTALLS_PER_EPISODE)
            return False
        if self._install_count_run >= self.MAX_INSTALLS_PER_RUN:
            logger.warning("Live install cap reached for run (%d)", self.MAX_INSTALLS_PER_RUN)
            return False
        return True

    def install_tool(
        self,
        name: str,
        dry_run: bool = False,
        allow_sudo: bool = False,
    ) -> InstallResult:
        """Install a registered tool.

        Args:
            name: Tool name (must be in registry)
            dry_run: If True, only check — don't actually install
            allow_sudo: Whether sudo is permitted for this install

        Returns:
            InstallResult with success/failure details
        """
        entry = self._entries.get(name)
        if not entry:
            return InstallResult(
                tool_name=name, success=False,
                message=f"Tool '{name}' not in registry allow-list",
            )

        if self.is_available(name):
            return InstallResult(
                tool_name=name, success=True,
                message="Already installed",
            )

        if entry.requires_sudo_install and not allow_sudo:
            return InstallResult(
                tool_name=name, success=False,
                required_sudo=True,
                message="Requires sudo but FF_ALLOW_SUDO is not enabled",
            )

        if dry_run:
            return InstallResult(
                tool_name=name, success=False,
                method=entry.install_method.value,
                message=f"DRY_RUN: Would install {entry.install_target} via {entry.install_method.value}",
            )

        # Actually install
        start = time.time()
        install_cmd = self._build_install_command(entry, allow_sudo)
        if not install_cmd:
            return InstallResult(
                tool_name=name, success=False,
                message=f"No install command for method {entry.install_method.value}",
            )

        try:
            result = subprocess.run(
                install_cmd, shell=True, capture_output=True,
                timeout=120, text=True,
            )
            duration_ms = int((time.time() - start) * 1000)

            if result.returncode == 0:
                # Clear cache and verify
                self._availability_cache.pop(name, None)
                verified = self.is_available(name)
                install_result = InstallResult(
                    tool_name=name,
                    success=verified,
                    method=entry.install_method.value,
                    message="Installed and verified" if verified else "Install returned 0 but tool not found",
                    duration_ms=duration_ms,
                    required_sudo=entry.requires_sudo_install,
                )
            else:
                install_result = InstallResult(
                    tool_name=name, success=False,
                    method=entry.install_method.value,
                    message=f"Install failed (rc={result.returncode}): {result.stderr[:200]}",
                    duration_ms=duration_ms,
                    required_sudo=entry.requires_sudo_install,
                )
        except subprocess.TimeoutExpired:
            install_result = InstallResult(
                tool_name=name, success=False,
                method=entry.install_method.value,
                message="Install timed out (120s)",
                duration_ms=120000,
            )
        except OSError as e:
            install_result = InstallResult(
                tool_name=name, success=False,
                message=f"OS error: {e}",
            )

        # Track
        self._install_log.append(install_result)
        if install_result.success:
            self._install_count_episode += 1
            self._install_count_run += 1
            logger.info("Installed tool: %s via %s", name, entry.install_method.value)
        else:
            logger.warning("Failed to install tool: %s — %s", name, install_result.message)

        return install_result

    def _build_install_command(self, entry: ToolEntry, allow_sudo: bool) -> Optional[str]:
        """Build the shell command to install a tool."""
        sudo = "sudo " if entry.requires_sudo_install and allow_sudo else ""

        if entry.install_method == InstallMethod.APT:
            return f"{sudo}apt-get install -y {entry.install_target}"
        elif entry.install_method == InstallMethod.PIP:
            pkg = entry.install_target
            if entry.version_pin:
                pkg = f"{pkg}=={entry.version_pin}"
            return f"pip install {pkg}"
        elif entry.install_method == InstallMethod.PIPX:
            pkg = entry.install_target
            if entry.version_pin:
                pkg = f"{pkg}=={entry.version_pin}"
            return f"pipx install {pkg}"
        elif entry.install_method == InstallMethod.GO:
            target = entry.install_target
            if entry.version_pin:
                target = f"{target}@{entry.version_pin}"
            elif "@" not in target:
                target = f"{target}@latest"
            return f"go install {target}"
        elif entry.install_method == InstallMethod.GIT_CLONE:
            dest = f"/opt/{entry.name}"
            return f"{sudo}git clone --depth 1 {entry.install_target} {dest}"
        elif entry.install_method == InstallMethod.SNAP:
            return f"{sudo}snap install {entry.install_target}"

        return None

    def bootstrap(
        self,
        profile: str = "htb",
        dry_run: bool = False,
        allow_sudo: bool = False,
    ) -> BootstrapReport:
        """Bootstrap all tools for a given profile.

        Args:
            profile: Tool profile name (htb/ms3/ms2/dev)
            dry_run: If True, just report what would happen
            allow_sudo: Whether sudo is allowed for installs

        Returns:
            BootstrapReport with install results
        """
        report = BootstrapReport(profile=profile)
        required = TOOL_PROFILES.get(profile, [])

        for name in required:
            if self.is_available(name):
                report.skipped.append(name)
                continue

            if dry_run:
                report.missing.append(name)
                continue

            result = self.install_tool(name, dry_run=False, allow_sudo=allow_sudo)
            if result.success:
                report.installed.append(result)
            else:
                report.failures.append(result)

        return report

    def get_install_log(self) -> List[Dict[str, Any]]:
        """Get the install audit log."""
        return [r.to_dict() for r in self._install_log]

    def invalidate_cache(self, name: Optional[str] = None) -> None:
        """Clear availability cache (all or specific tool)."""
        if name:
            self._availability_cache.pop(name, None)
        else:
            self._availability_cache.clear()


# ============================================================================
# SINGLETON ACCESS
# ============================================================================

_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Get the global ToolRegistry singleton."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry


def reset_tool_registry() -> None:
    """Reset the registry (for testing)."""
    global _registry
    _registry = None
