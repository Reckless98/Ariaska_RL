"""Evidence Gate — validates exploit-phase commands have supporting evidence.

Phase 41: Extracted from SmartCoach for modularity.
"""
from __future__ import annotations

import re
import logging
from typing import Dict, Any, List, Set, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger("ariaska.coach.evidence_gate")

_LOCAL_CMDS = (
    "sudo", "find", "getcap", "id", "uname", "cat ", "ls ",
    "whoami", "hostname", "ifconfig", "ip ", "env", "echo ",
    "grep ", "awk ", "sed ", "head ", "tail ", "which ",
    "ps ", "netstat", "ss ", "mount", "df ", "lsof",
    "history", "crontab", "systemctl",
)

_EXPLOIT_TOOLS: Set[str] = {
    "hydra", "medusa", "sqlmap", "msfconsole", "metasploit",
    "searchsploit", "smbclient", "enum4linux", "rpcclient",
    "wpscan", "john", "hashcat", "crackmapexec",
    "impacket", "psexec", "winrm", "evil-winrm",
}

_HALLUCINATION_CHECKS: Dict[str, Set[str]] = {
    "ircd": {"irc", "ircd", "unreal"},
    "ftp": {"ftp", "vsftpd", "proftpd"},
    "samba": {"smb", "samba", "microsoft-ds"},
    "telnet": {"telnet"},
    "mysql": {"mysql", "mariadb"},
    "postgresql": {"postgresql", "postgres"},
    "vnc": {"vnc"},
    "rmi": {"rmi", "java-rmi"},
}

_PORT_SERVICE_MAP: Dict[int, str] = {
    21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp", 80: "http",
    110: "pop3", 139: "smb", 143: "imap", 443: "https", 445: "smb",
    1433: "mssql", 1524: "ingreslock", 3306: "mysql", 3389: "rdp",
    5432: "postgresql", 5900: "vnc", 6667: "irc", 8080: "http",
    8180: "http", 8443: "https",
}

_EXPLOIT_PHASES: Set[str] = {
    "EXPLOITATION", "PRIVILEGE_ESCALATION", "LATERAL_MOVEMENT",
    "POST_EXPLOITATION", "EXFILTRATION", "CLOSEOUT",
}

_SERVICE_TOOL_MAP: Dict[str, Dict[str, Set[str]]] = {
    "hydra": {"ssh": {"ssh"}, "ftp": {"ftp"}, "smb": {"smb", "samba"},
              "http": {"http", "https", "web"}},
    "sqlmap": {"_any": {"http", "https", "web"}},
    "nikto": {"_any": {"http", "https", "web"}},
    "ffuf": {"_any": {"http", "https", "web"}},
    "gobuster": {"_any": {"http", "https", "web"}},
    "curl": {"_any": {"http", "https", "web"}},
    "wpscan": {"_any": {"http", "https", "web"}},
    "smbclient": {"_any": {"smb", "samba", "microsoft-ds"}},
    "enum4linux": {"_any": {"smb", "samba", "microsoft-ds"}},
    "rpcclient": {"_any": {"smb", "samba", "microsoft-ds", "rpc"}},
}

_PORT_RE = [
    re.compile(r"-p\s*(\d+)"),
    re.compile(r":(\d+)(?:/|\s|$)"),
    re.compile(r"--port[= ](\d+)"),
]


@dataclass
class EvidenceGateConfig:
    """Configuration for evidence gate behavior."""
    mode: str = "enforce"  # "off", "log", "enforce"


class EvidenceGate:
    """Standalone evidence gate for exploit-phase command validation."""

    def __init__(self, config: Optional[EvidenceGateConfig] = None) -> None:
        self.config = config or EvidenceGateConfig()
        self._reject_count: int = 0
        self._pass_count: int = 0

    def validate(
        self, command: str, phase_name: str, discovery_board: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        """Validate a command has supporting evidence.

        Returns:
            (valid, reasons) tuple.
        """
        if self.config.mode == "off":
            return True, []

        reasons: List[str] = []
        command = (command or "").strip()
        if not command:
            return True, []

        known_ports = {str(p) for p in discovery_board.get("ports", set())}
        known_services = {str(s).lower() for s in discovery_board.get("services", set())}
        cmd_lower = command.lower()
        cmd_tool = cmd_lower.split()[0].split("/")[-1] if cmd_lower else ""

        if "ssh " in cmd_lower or "sshpass" in cmd_lower:
            reasons.append("ssh_wrapped_command_allowed")
            self._pass_count += 1
            return True, reasons

        if any(cmd_lower.startswith(lc) for lc in _LOCAL_CMDS):
            self._pass_count += 1
            return True, []

        if not known_ports and cmd_tool in _EXPLOIT_TOOLS:
            reasons.append(f"no_ports_discovered_yet_cannot_use_{cmd_tool}")
            self._reject_count += 1
            return False, reasons

        known_port_set = {int(p) for p in known_ports if str(p).isdigit()}
        if known_ports:
            for svc_key, svc_names in _HALLUCINATION_CHECKS.items():
                if any(sn in cmd_lower for sn in svc_names):
                    svc_found = bool(svc_names.intersection(known_services))
                    port_found = any(
                        p in known_port_set
                        for p, s in _PORT_SERVICE_MAP.items()
                        if s in svc_names or s == svc_key
                    )
                    if not svc_found and not port_found:
                        reasons.append(f"hallucination_{svc_key}_not_on_target")

        if phase_name not in _EXPLOIT_PHASES:
            valid = len(reasons) == 0 or all("allowed" in r for r in reasons)
            if valid:
                self._pass_count += 1
            else:
                self._reject_count += 1
            return valid, reasons

        for pat in _PORT_RE:
            m = pat.search(command)
            if m:
                port = m.group(1)
                if port not in known_ports:
                    reasons.append(f"port_{port}_not_in_discovery_board")

        for tool, svc_map in _SERVICE_TOOL_MAP.items():
            if tool in cmd_tool:
                for key, required_svcs in svc_map.items():
                    if key == "_any" or key in cmd_lower:
                        if not required_svcs.intersection(known_services):
                            reasons.append(f"service_evidence_missing_for_{tool}")
                            break

        cve_match = re.search(r"CVE-\d{4}-\d+", command, re.IGNORECASE)
        if cve_match:
            cve_id = cve_match.group().upper()
            known_vulns = {str(v).upper() for v in discovery_board.get("vulns", set())}
            if cve_id not in known_vulns:
                reasons.append(f"cve_{cve_id}_not_in_discovery_board")

        valid = len(reasons) == 0 or all("allowed" in r for r in reasons)
        if valid:
            self._pass_count += 1
        else:
            self._reject_count += 1
        return valid, reasons

    def get_stats(self) -> Dict[str, Any]:
        """Return gate statistics."""
        return {"pass_count": self._pass_count, "reject_count": self._reject_count, "mode": self.config.mode}

    def reset_stats(self) -> None:
        """Reset counters."""
        self._pass_count = 0
        self._reject_count = 0
