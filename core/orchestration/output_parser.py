"""Output Parser — standalone discovery extraction from command output.

Phase 41: Extracted parsing logic from SmartOrchestrator._parse_output_for_discoveries.
These are stateless utility functions that can be used independently.
"""
from __future__ import annotations

import re
import logging
from typing import Dict, Any, List, Set, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("ariaska.orchestration.output_parser")

# ── Compiled regex patterns ──────────────────────────────────────────────
PORT_OPEN_RE = re.compile(r"(\d+)/tcp\s+open\s+(\S+)", re.IGNORECASE)
PORT_UDP_RE = re.compile(r"(\d+)/udp\s+open\s+(\S+)", re.IGNORECASE)
IP_RE = re.compile(r"\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b")
MAC_RE = re.compile(r"(?:[0-9a-f]{2}:){3,}[0-9a-f]{2}", re.IGNORECASE)
CVE_RE = re.compile(r"CVE-\d{4}-\d+", re.IGNORECASE)
VERSION_RE = re.compile(
    r"(?:version|ver\.?|v)\s*[:\s]?\s*(\d+(?:\.\d+){1,3}(?:[\-_]\w+)?)",
    re.IGNORECASE,
)
HASH_RE = re.compile(r"\b[a-f0-9]{32,128}\b", re.IGNORECASE)
DB_NAME_RE = re.compile(r"(?:database|schema)[\s:]+?(\w+)", re.IGNORECASE)
SMB_SHARE_RE = re.compile(r"(?:Disk|IPC):\s*(\w+)|\\\\[^\\]+\\(\w+)", re.IGNORECASE)
WEB_PATH_200_RE = re.compile(
    r"(?:200|Status:\s*200).*?(/[\w\-\./]+)", re.IGNORECASE
)
FEROX_PATH_RE = re.compile(
    r"(?:200|301|302)\s+\w+\s+((?:/[\w\-\.]+)+)", re.IGNORECASE
)
CAPABILITY_RE = re.compile(r"(/\S+)\s*=\s*cap_\w+", re.IGNORECASE)
FLAG_RE = re.compile(
    r"(?:flag|ctf|HTB|THM)\{[^}]+\}|[a-f0-9]{32}", re.IGNORECASE
)
SHELL_INDICATORS = [
    "root@", "uid=0(root)", "nt authority\\system",
    "meterpreter >", "shell >", "# $",
]
ROOT_SHELL_INDICATORS = ["root@", "uid=0(root)", "nt authority\\system"]

CREDENTIAL_PATTERNS: List[re.Pattern] = [  # type: ignore[type-arg]
    re.compile(r"login:\s*(\S+)\s+password:\s*(\S+)", re.IGNORECASE),
    re.compile(r"(\w+):(\S+)@", re.IGNORECASE),
    re.compile(r"credentials?[:\s]+?(\w+)[:/](\S+)", re.IGNORECASE),
]

USER_PATTERNS: List[re.Pattern] = [  # type: ignore[type-arg]
    re.compile(r"uid=\d+\((\w+)\)", re.IGNORECASE),
    re.compile(r"^(\w+):\w?:\d+:", re.MULTILINE),
]

# Port → service defaults
PORT_SERVICE_DEFAULTS: Dict[int, str] = {
    21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp", 53: "dns",
    80: "http", 110: "pop3", 111: "rpcbind", 139: "netbios-ssn",
    143: "imap", 443: "https", 445: "microsoft-ds", 512: "exec",
    513: "login", 514: "shell", 1099: "java-rmi", 1433: "mssql",
    1524: "ingreslock", 2049: "nfs", 2121: "ftp", 3306: "mysql",
    3389: "rdp", 3632: "distccd", 5432: "postgresql", 5900: "vnc",
    6000: "x11", 6667: "irc", 6697: "ircs", 8009: "ajp13",
    8080: "http-proxy", 8180: "http-alt", 8443: "https-alt",
    8888: "http", 9200: "elasticsearch",
}


@dataclass
class DiscoveryResult:
    """Result of parsing command output for discoveries."""
    ports: Set[int] = field(default_factory=set)
    services: Set[str] = field(default_factory=set)
    credentials: List[Tuple[str, str]] = field(default_factory=list)
    users: Set[str] = field(default_factory=set)
    vulns: Set[str] = field(default_factory=set)
    web_paths: Set[str] = field(default_factory=set)
    flags: Set[str] = field(default_factory=set)
    hashes: Set[str] = field(default_factory=set)
    versions: Dict[str, str] = field(default_factory=dict)
    shells: Set[str] = field(default_factory=set)
    capabilities: Set[str] = field(default_factory=set)
    smb_shares: Set[str] = field(default_factory=set)
    databases: Set[str] = field(default_factory=set)
    cves: Set[str] = field(default_factory=set)

    @property
    def total_count(self) -> int:
        """Total number of discoveries."""
        return (
            len(self.ports) + len(self.services) + len(self.credentials)
            + len(self.users) + len(self.vulns) + len(self.web_paths)
            + len(self.flags) + len(self.hashes) + len(self.shells)
            + len(self.capabilities) + len(self.smb_shares)
            + len(self.databases) + len(self.cves)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ports": sorted(self.ports),
            "services": sorted(self.services),
            "credentials": self.credentials,
            "users": sorted(self.users),
            "vulns": sorted(self.vulns),
            "web_paths": sorted(self.web_paths),
            "flags": sorted(self.flags),
            "hashes": sorted(self.hashes),
            "shells": sorted(self.shells),
            "capabilities": sorted(self.capabilities),
            "smb_shares": sorted(self.smb_shares),
            "databases": sorted(self.databases),
            "cves": sorted(self.cves),
        }

    def merge_into_board(self, board: Dict[str, Any]) -> int:
        """Merge discoveries into an existing discovery_board, returns new count."""
        new_count = 0
        for port in self.ports:
            s = board.setdefault("ports", set())
            if port not in s:
                s.add(port)
                new_count += 1
        for svc in self.services:
            s = board.setdefault("services", set())
            if svc not in s:
                s.add(svc)
                new_count += 1
        for user in self.users:
            s = board.setdefault("users", set())
            if user not in s:
                s.add(user)
                new_count += 1
        for vuln in self.vulns:
            s = board.setdefault("vulns", set())
            if vuln not in s:
                s.add(vuln)
                new_count += 1
        for wp in self.web_paths:
            s = board.setdefault("web_paths", set())
            if wp not in s:
                s.add(wp)
                new_count += 1
        for flag in self.flags:
            s = board.setdefault("flags_set", set())
            if flag not in s:
                s.add(flag)
                new_count += 1
        for h in self.hashes:
            s = board.setdefault("hashes", set())
            if h not in s:
                s.add(h)
                new_count += 1
        for sh in self.shells:
            s = board.setdefault("shells", set())
            if sh not in s:
                s.add(sh)
                new_count += 1
        for cap in self.capabilities:
            s = board.setdefault("capabilities", set())
            if cap not in s:
                s.add(cap)
                new_count += 1
        for share in self.smb_shares:
            s = board.setdefault("smb_shares", set())
            if share not in s:
                s.add(share)
                new_count += 1
        for db in self.databases:
            s = board.setdefault("databases", set())
            if db not in s:
                s.add(db)
                new_count += 1
        for cve in self.cves:
            s = board.setdefault("vulns", set())
            if cve not in s:
                s.add(cve)
                new_count += 1
        for cred in self.credentials:
            lst = board.setdefault("credentials", set())
            key = f"{cred[0]}:{cred[1]}"
            if key not in lst:
                lst.add(key)
                new_count += 1
        return new_count


def parse_ports(output: str) -> Set[int]:
    """Extract open ports from scan output."""
    ports: Set[int] = set()
    for m in PORT_OPEN_RE.finditer(output):
        try:
            p = int(m.group(1))
            if 1 <= p <= 65535:
                ports.add(p)
        except ValueError:
            pass
    for m in PORT_UDP_RE.finditer(output):
        try:
            p = int(m.group(1))
            if 1 <= p <= 65535:
                ports.add(p)
        except ValueError:
            pass
    return ports


def parse_services(output: str, ports: Optional[Set[int]] = None) -> Set[str]:
    """Extract service names from output."""
    services: Set[str] = set()
    for m in PORT_OPEN_RE.finditer(output):
        svc = m.group(2).strip().lower()
        if svc and svc not in ("unknown", "tcpwrapped"):
            services.add(svc)
    if ports:
        for p in ports:
            default = PORT_SERVICE_DEFAULTS.get(p)
            if default:
                services.add(default)
    return services


def parse_credentials(output: str) -> List[Tuple[str, str]]:
    """Extract credential pairs from output."""
    creds: List[Tuple[str, str]] = []
    seen: Set[str] = set()
    for pat in CREDENTIAL_PATTERNS:
        for m in pat.finditer(output):
            user, pw = m.group(1), m.group(2)
            key = f"{user}:{pw}"
            if key not in seen and len(user) < 64 and len(pw) < 128:
                seen.add(key)
                creds.append((user, pw))
    return creds


def parse_users(output: str) -> Set[str]:
    """Extract usernames from output."""
    users: Set[str] = set()
    skip = {"root", "daemon", "bin", "sys", "sync", "games", "man", "lp",
            "mail", "news", "nobody", "systemd", "syslog", "_apt"}
    for pat in USER_PATTERNS:
        for m in pat.finditer(output):
            u = m.group(1).strip()
            if u and u.lower() not in skip and len(u) < 64:
                users.add(u)
    return users


def parse_web_paths(output: str) -> Set[str]:
    """Extract discovered web paths."""
    paths: Set[str] = set()
    for m in WEB_PATH_200_RE.finditer(output):
        p = m.group(1).strip()
        if p and len(p) > 1:
            paths.add(p)
    for m in FEROX_PATH_RE.finditer(output):
        p = m.group(1).strip()
        if p and len(p) > 1:
            paths.add(p)
    return paths


def parse_cves(output: str) -> Set[str]:
    """Extract CVE identifiers."""
    return {m.group().upper() for m in CVE_RE.finditer(output)}


def parse_flags(output: str) -> Set[str]:
    """Extract CTF flags."""
    return {m.group() for m in FLAG_RE.finditer(output)}


def parse_hashes(output: str) -> Set[str]:
    """Extract password hashes."""
    hashes: Set[str] = set()
    for m in HASH_RE.finditer(output):
        h = m.group()
        if len(h) >= 32 and not MAC_RE.match(h):
            hashes.add(h)
    return hashes


def detect_shell(output: str) -> Optional[str]:
    """Detect if output indicates a shell was obtained."""
    out_lower = output.lower()
    for indicator in ROOT_SHELL_INDICATORS:
        if indicator.lower() in out_lower:
            return "root_shell"
    for indicator in SHELL_INDICATORS:
        if indicator.lower() in out_lower:
            return "shell"
    return None


def parse_smb_shares(output: str) -> Set[str]:
    """Extract SMB share names."""
    shares: Set[str] = set()
    for m in SMB_SHARE_RE.finditer(output):
        name = m.group(1) or m.group(2)
        if name:
            shares.add(name)
    return shares


def parse_capabilities(output: str) -> Set[str]:
    """Extract Linux capabilities (SUID/capabilities for privesc)."""
    return {m.group(1) for m in CAPABILITY_RE.finditer(output)}


def parse_databases(output: str) -> Set[str]:
    """Extract database names."""
    return {m.group(1) for m in DB_NAME_RE.finditer(output)}


def parse_all(output: str) -> DiscoveryResult:
    """Parse command output for all discovery types."""
    if not output or not output.strip():
        return DiscoveryResult()
    ports = parse_ports(output)
    return DiscoveryResult(
        ports=ports,
        services=parse_services(output, ports),
        credentials=parse_credentials(output),
        users=parse_users(output),
        web_paths=parse_web_paths(output),
        cves=parse_cves(output),
        flags=parse_flags(output),
        hashes=parse_hashes(output),
        shells={detect_shell(output)} - {None} if detect_shell(output) else set(),
        capabilities=parse_capabilities(output),
        smb_shares=parse_smb_shares(output),
        databases=parse_databases(output),
    )
