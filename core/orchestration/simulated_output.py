"""Simulated Output Generator — canned outputs for dry-run / training mode.

Phase 41: Extracted from SmartOrchestrator._generate_simulated_output.
Maps command prefixes to realistic simulated outputs for offline training.
"""
from __future__ import annotations

import random
import logging
from typing import Dict, Any, Optional, Callable

logger = logging.getLogger("ariaska.orchestration.simulated_output")

# ── Success rate by command category ────────────────────────────────────
CATEGORY_SUCCESS_RATES: Dict[str, float] = {
    "recon": 0.80,
    "enum": 0.70,
    "brute": 0.35,
    "exploit": 0.40,
    "privesc": 0.30,
    "shell": 0.30,
    "web": 0.65,
    "lateral": 0.25,
    "exfil": 0.50,
    "cleanup": 0.90,
}

# ── Command category detection ──────────────────────────────────────────
_CATEGORY_PREFIXES: Dict[str, list] = {
    "recon": ["nmap", "masscan", "ping", "traceroute", "arp-scan", "netdiscover"],
    "enum": ["enum4linux", "smbclient", "snmpwalk", "rpcclient", "nbtscan",
             "showmount", "rpcinfo", "finger", "smtp-user-enum", "ldapsearch"],
    "web": ["gobuster", "dirb", "nikto", "feroxbuster", "ffuf", "wpscan",
            "curl", "wget", "dirsearch", "wfuzz", "nuclei", "whatweb"],
    "brute": ["hydra", "medusa", "john", "hashcat", "crackmapexec"],
    "exploit": ["msfconsole", "searchsploit", "sqlmap", "exploit/"],
    "privesc": ["sudo", "find / -perm", "getcap", "linpeas", "linenum"],
    "shell": ["bash -i", "python -c", "nc -e", "reverse"],
    "cleanup": ["rm -rf", "shred", "timestomp", "clear_"],
    "exfil": ["scp ", "base64 ", "exfil"],
}


def get_command_category(command: str) -> str:
    """Classify a command into a category."""
    cmd_lower = command.lower().strip()
    for category, prefixes in _CATEGORY_PREFIXES.items():
        for prefix in prefixes:
            if cmd_lower.startswith(prefix) or prefix in cmd_lower:
                return category
    return "recon"


def should_succeed(command: str, phase: str = "RECON",
                   rng: Optional[random.Random] = None) -> bool:
    """Determine if a simulated command should succeed."""
    r = rng or random.Random()
    cat = get_command_category(command)
    rate = CATEGORY_SUCCESS_RATES.get(cat, 0.50)
    # Phase modifiers
    if phase in ("EXPLOITATION", "PRIVILEGE_ESCALATION") and cat in ("exploit", "privesc"):
        rate *= 1.2  # Slightly higher when in correct phase
    rate = min(rate, 1.0)
    return r.random() < rate


# ── Nmap simulation ─────────────────────────────────────────────────────
def simulate_nmap(target: str, ports: Optional[str] = None) -> str:
    """Generate simulated nmap output."""
    lines = [
        f"Starting Nmap 7.94 ( https://nmap.org )",
        f"Nmap scan report for {target}",
        f"Host is up (0.0021s latency).",
        "",
    ]
    default_ports = [
        (21, "ftp", "vsftpd 2.3.4"),
        (22, "ssh", "OpenSSH 4.7p1"),
        (23, "telnet", "Linux telnetd"),
        (25, "smtp", "Postfix smtpd"),
        (80, "http", "Apache httpd 2.2.8"),
        (139, "netbios-ssn", "Samba smbd 3.X"),
        (445, "microsoft-ds", "Samba smbd 3.X"),
        (3306, "mysql", "MySQL 5.0.51a"),
        (5432, "postgresql", "PostgreSQL DB 8.3"),
        (8080, "http-proxy", "Apache Tomcat/Coyote JSP 1.1"),
    ]
    lines.append("PORT      STATE SERVICE        VERSION")
    for port, svc, ver in default_ports:
        lines.append(f"{port}/tcp   open  {svc:<14s} {ver}")
    lines.append("")
    lines.append(f"Service detection performed. {len(default_ports)} services scanned.")
    lines.append("Nmap done: 1 IP address (1 host up)")
    return "\n".join(lines)


def simulate_gobuster(target: str) -> str:
    """Generate simulated gobuster output."""
    paths = [
        "/index.html (Status: 200) [Size: 891]",
        "/phpinfo.php (Status: 200) [Size: 47585]",
        "/robots.txt (Status: 200) [Size: 26]",
        "/icons/ (Status: 403) [Size: 292]",
        "/cgi-bin/ (Status: 403) [Size: 294]",
        "/doc/ (Status: 403) [Size: 290]",
        "/wp-admin/ (Status: 301) [Size: 324]",
        "/phpmyadmin/ (Status: 200) [Size: 8834]",
        "/test/ (Status: 200) [Size: 47]",
    ]
    return "\n".join(paths)


def simulate_hydra(target: str, service: str = "ssh") -> str:
    """Generate simulated hydra output."""
    return (
        f"Hydra v9.5 (c) 2023 by van Hauser\n"
        f"[DATA] attacking {service}://{target}:22\n"
        f"[22][{service}] host: {target}   login: msfadmin   password: msfadmin\n"
        f"1 of 1 target successfully completed, 1 valid password found"
    )


def simulate_searchsploit(query: str) -> str:
    """Generate simulated searchsploit output."""
    return (
        f"Exploit Title                           |  Path\n"
        f"--------------------------------------- | ------\n"
        f"vsftpd 2.3.4 - Backdoor Command Exec   | exploits/unix/remote/17491.rb\n"
        f"UnrealIRCd 3.2.8.1 - Backdoor           | exploits/linux/remote/16922.rb\n"
        f"Samba 3.0.20 - usermap_script RCE        | exploits/unix/remote/16320.rb\n"
    )


def simulate_enum4linux(target: str) -> str:
    """Generate simulated enum4linux output."""
    return (
        f"Starting enum4linux v0.9.1\n"
        f"Target: {target}\n"
        f"[+] Server {target} allows sessions using username '', password ''\n"
        f"[+] Got domain/workgroup name: WORKGROUP\n"
        f"[+] Users found: msfadmin, user, service\n"
        f"[+] Share: //METASPLOITABLE/tmp  Type: Disk\n"
        f"[+] Share: //METASPLOITABLE/opt  Type: Disk\n"
    )


_SIMULATOR_MAP: Dict[str, Callable] = {}  # Populated at module load


def generate_simulated_output(
    command: str,
    target: str = "10.10.10.1",
    phase: str = "RECON",
    rng: Optional[random.Random] = None,
) -> str:
    """Generate simulated command output.

    Args:
        command: The command to simulate
        target: Target IP/hostname
        phase: Current attack phase
        rng: Optional RNG for reproducibility

    Returns:
        Simulated output string
    """
    cmd_lower = command.lower().strip()
    cmd_tool = cmd_lower.split()[0].split("/")[-1] if cmd_lower else ""

    # Direct matches
    if cmd_tool == "nmap" or cmd_lower.startswith("nmap"):
        return simulate_nmap(target)
    elif cmd_tool in ("gobuster", "dirb", "dirsearch", "feroxbuster", "ffuf"):
        return simulate_gobuster(target)
    elif cmd_tool == "hydra":
        svc = "ssh"
        if "ftp" in cmd_lower:
            svc = "ftp"
        elif "http" in cmd_lower:
            svc = "http"
        return simulate_hydra(target, svc)
    elif cmd_tool == "searchsploit":
        return simulate_searchsploit(cmd_lower)
    elif cmd_tool == "enum4linux":
        return simulate_enum4linux(target)
    elif cmd_tool in ("cat", "ls", "id", "whoami", "uname", "hostname"):
        return _simulate_local_cmd(cmd_tool, command)
    elif not should_succeed(command, phase, rng):
        return f"Error: {cmd_tool}: command failed or timed out"

    return f"[simulated] {cmd_tool} completed successfully against {target}"


def _simulate_local_cmd(tool: str, command: str) -> str:
    """Simulate local info-gathering commands."""
    if tool == "id":
        return "uid=0(root) gid=0(root) groups=0(root)"
    elif tool == "whoami":
        return "root"
    elif tool == "uname":
        return "Linux metasploitable 2.6.24-16-server #1 SMP x86_64 GNU/Linux"
    elif tool == "hostname":
        return "metasploitable"
    elif tool == "cat":
        if "/etc/passwd" in command:
            return "root:x:0:0:root:/root:/bin/bash\nmsfadmin:x:1000:1000::/home/msfadmin:/bin/bash\nuser:x:1001:1001::/home/user:/bin/bash"
        elif "/etc/shadow" in command:
            return "root:$6$salt$hash:18000:0:99999:7:::\nmsfadmin:$6$salt$hash2:18000:0:99999:7:::"
        return ""
    elif tool == "ls":
        return "total 0\ndrwxr-xr-x 2 root root 40 Jan  1 00:00 ."
    return f"[simulated] {tool} output"
