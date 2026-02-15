#!/usr/bin/env python3
"""
data/knowledge_ingestion.py — ARIASKA Knowledge Mega-Ingestion Pipeline v1.0

Downloads, parses, and structures cybersecurity knowledge from top GitHub repositories
into a unified JSON knowledge base that Ariaska's agents can query at runtime.

Sources (21 repos, prioritized):
    1. HackTricks-wiki/hacktricks        — Per-port/service pentesting guides (LARGEST)
    2. swisskyrepo/PayloadsAllTheThings   — Vuln-specific payloads & methodology
    3. 0xsyr0/OSCP                        — Command-first cheat sheet (6000+ lines)
    4. The-Art-of-Hacking/h4cker          — Methodology + metasploit cheat sheets
    5. vulhub/vulhub                      — CVE reproduction environments
    6. nixawk/pentest-wiki                — Kill-chain organized pentest wiki
    7. GTFOBins/GTFOBins.github.io        — Unix binary privesc/shell escape
    8. LOLBAS-Project/LOLBAS-Project.github.io — Windows LOLBins
    9. WADComs/WADComs.github.io          — Windows/AD command cheatsheet
   10. redcanaryco/atomic-red-team        — MITRE ATT&CK atomic tests
   11. mitre/cti                          — MITRE ATT&CK STIX data
   12. offensive-security/exploitdb       — 45K+ exploit database
   13. swisskyrepo/InternalAllTheThings   — AD/pivoting/internal pentest
   14. HackTricks-wiki/hacktricks-cloud   — Cloud attack methodology
   15. peass-ng/PEASS-ng                  — LinPEAS/WinPEAS privesc detection
   16. fortra/impacket                    — AD exploitation tools docs
   17. samratashok/nishang                — PowerShell offensive tools
   18. Crypto-Cat/CTF                     — Structured CTF writeups
   19. carlospolop/PEASS-ng               — (alias for peass-ng)
   20. danielmiessler/SecLists            — Wordlists + fuzzing payloads
   21. rapid7/metasploit-framework        — Metasploit module docs

Output:
    data/knowledge_base/
        ├── services.json          — Per-service exploitation knowledge
        ├── ports.json             — Per-port pentesting commands
        ├── cves.json              — CVE details with exploit methodology
        ├── payloads.json          — Vulnerability-specific payloads
        ├── methodology.json       — Kill-chain phase methodology
        ├── commands.json          — Tool-specific command references
        ├── kill_chains.json       — Full exploitation chains with reasoning
        ├── privesc_linux.json     — Linux privilege escalation techniques
        ├── privesc_windows.json   — Windows privilege escalation techniques
        ├── cheatsheets.json       — Consolidated cheat sheets
        ├── techniques.json        — MITRE ATT&CK techniques with commands
        ├── binaries.json          — GTFOBins/LOLBAS binary abuse methods
        ├── exploitdb.json         — ExploitDB exploit entries
        ├── ad_attacks.json        — Active Directory attack methodology
        ├── cloud_attacks.json     — Cloud (AWS/Azure/GCP) attack patterns
        ├── ctf_writeups.json      — CTF challenge solutions & reasoning
        ├── wordlists_meta.json    — Wordlist metadata (names, sizes, contexts)
        └── manifest.json          — Ingestion metadata & stats

Usage:
    python -m data.knowledge_ingestion            # Full ingestion
    python -m data.knowledge_ingestion --quick     # Parse only (skip clone if exists)
    python -m data.knowledge_ingestion --repo hacktricks  # Single repo

Author: Filip Volf / Ariaska System
"""

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict

logger = logging.getLogger("ariaska.knowledge_ingestion")

# ─── Configuration ───────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
CLONE_DIR = BASE_DIR / "knowledge_repos"
OUTPUT_DIR = BASE_DIR / "knowledge_base"

REPOS = {
    "hacktricks": {
        "url": "https://github.com/HackTricks-wiki/hacktricks.git",
        "desc": "Comprehensive per-port/service pentesting methodology",
        "priority": 1,
        "shallow": True,  # --depth 1 to save space
    },
    "payloads": {
        "url": "https://github.com/swisskyrepo/PayloadsAllTheThings.git",
        "desc": "Vulnerability-specific payloads and methodology",
        "priority": 2,
        "shallow": True,
    },
    "oscp": {
        "url": "https://github.com/0xsyr0/OSCP.git",
        "desc": "OSCP command-first cheat sheet (6000+ lines)",
        "priority": 3,
        "shallow": True,
    },
    "h4cker": {
        "url": "https://github.com/The-Art-of-Hacking/h4cker.git",
        "desc": "Pentesting methodology + metasploit cheat sheets",
        "priority": 4,
        "shallow": True,
    },
    "vulhub": {
        "url": "https://github.com/vulhub/vulhub.git",
        "desc": "CVE reproduction environments with exploitation tutorials",
        "priority": 5,
        "shallow": True,
    },
    "pentest_wiki": {
        "url": "https://github.com/nixawk/pentest-wiki.git",
        "desc": "Kill-chain organized pentest methodology",
        "priority": 6,
        "shallow": True,
    },
    # ─── Phase 9.2: Expanded repos (15 new) ─────────────────────────
    "gtfobins": {
        "url": "https://github.com/GTFOBins/GTFOBins.github.io.git",
        "desc": "Unix binary privesc/shell escape methods (~200 binaries)",
        "priority": 7,
        "shallow": True,
    },
    "lolbas": {
        "url": "https://github.com/LOLBAS-Project/LOLBAS-Project.github.io.git",
        "desc": "Windows living-off-the-land binaries",
        "priority": 8,
        "shallow": True,
    },
    "wadcoms": {
        "url": "https://github.com/WADComs/WADComs.github.io.git",
        "desc": "Windows/Active Directory command cheatsheet (YAML)",
        "priority": 9,
        "shallow": True,
    },
    "atomic_red_team": {
        "url": "https://github.com/redcanaryco/atomic-red-team.git",
        "desc": "MITRE ATT&CK atomic tests (~1500 techniques with commands)",
        "priority": 10,
        "shallow": True,
    },
    "mitre_cti": {
        "url": "https://github.com/mitre/cti.git",
        "desc": "MITRE ATT&CK STIX data (techniques, tactics, procedures)",
        "priority": 11,
        "shallow": True,
    },
    "exploitdb": {
        "url": "https://gitlab.com/exploit-database/exploitdb.git",
        "desc": "45K+ exploit database with metadata CSV",
        "priority": 12,
        "shallow": False,
    },
    "internal_all_the_things": {
        "url": "https://github.com/swisskyrepo/InternalAllTheThings.git",
        "desc": "Active Directory, pivoting, internal pentest methodology",
        "priority": 13,
        "shallow": True,
    },
    "hacktricks_cloud": {
        "url": "https://github.com/HackTricks-wiki/hacktricks-cloud.git",
        "desc": "AWS/Azure/GCP cloud attack methodology",
        "priority": 14,
        "shallow": True,
    },
    "peass_ng": {
        "url": "https://github.com/peass-ng/PEASS-ng.git",
        "desc": "LinPEAS/WinPEAS privilege escalation detection scripts",
        "priority": 15,
        "shallow": True,
    },
    "impacket": {
        "url": "https://github.com/fortra/impacket.git",
        "desc": "Python AD exploitation tools with documentation",
        "priority": 16,
        "shallow": True,
    },
    "nishang": {
        "url": "https://github.com/samratashok/nishang.git",
        "desc": "PowerShell offensive security scripts",
        "priority": 17,
        "shallow": True,
    },
    "ctf_writeups": {
        "url": "https://github.com/Crypto-Cat/CTF.git",
        "desc": "Structured CTF writeups with solutions",
        "priority": 18,
        "shallow": True,
    },
    "seclists": {
        "url": "https://github.com/danielmiessler/SecLists.git",
        "desc": "Wordlists, fuzzing payloads, usernames, passwords",
        "priority": 19,
        "shallow": True,
    },
    "metasploit": {
        "url": "https://github.com/rapid7/metasploit-framework.git",
        "desc": "Metasploit modules with documentation",
        "priority": 20,
        "shallow": True,
    },
}

# ─── Data Structures ────────────────────────────────────────────────────────

@dataclass
class ServiceEntry:
    """Knowledge about a specific service/protocol."""
    service_name: str
    port: int = 0
    protocol: str = "tcp"
    description: str = ""
    version_info: List[str] = field(default_factory=list)
    enumeration_commands: List[Dict[str, str]] = field(default_factory=list)
    exploitation_commands: List[Dict[str, str]] = field(default_factory=list)
    default_credentials: List[Dict[str, str]] = field(default_factory=list)
    common_vulnerabilities: List[str] = field(default_factory=list)
    privilege_escalation: List[str] = field(default_factory=list)
    methodology: str = ""
    reasoning: str = ""
    kill_chain_phase: str = ""
    source: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class CVEEntry:
    """Knowledge about a specific CVE."""
    cve_id: str
    description: str = ""
    affected_software: str = ""
    affected_versions: str = ""
    severity: str = ""
    exploit_methodology: str = ""
    exploit_commands: List[Dict[str, str]] = field(default_factory=list)
    metasploit_module: str = ""
    reasoning: str = ""
    references: List[str] = field(default_factory=list)
    source: str = ""


@dataclass
class PayloadEntry:
    """A specific vulnerability payload with context."""
    vuln_type: str  # sqli, xss, ssti, lfi, rce, etc.
    payload: str
    context: str = ""  # When to use this payload
    target_tech: str = ""  # PHP, Java, Python, etc.
    bypass_technique: str = ""
    reasoning: str = ""
    source: str = ""


@dataclass
class MethodologyEntry:
    """Kill-chain phase methodology."""
    phase: str  # RECON, ENUMERATION, EXPLOITATION, etc.
    title: str = ""
    steps: List[str] = field(default_factory=list)
    commands: List[Dict[str, str]] = field(default_factory=list)
    reasoning: str = ""
    when_to_use: str = ""
    common_mistakes: List[str] = field(default_factory=list)
    source: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class KillChainEntry:
    """Full exploitation chain from recon to shell/root."""
    target_service: str
    target_port: int = 0
    chain_name: str = ""
    steps: List[Dict[str, str]] = field(default_factory=list)
    total_commands: int = 0
    end_result: str = ""  # "shell", "root_shell", "data_exfil", etc.
    reasoning: str = ""
    difficulty: str = ""  # easy, medium, hard
    source: str = ""


@dataclass
class CommandEntry:
    """A specific tool/command with usage context."""
    tool_name: str
    command: str
    description: str = ""
    phase: str = ""  # RECON, ENUMERATION, EXPLOITATION, etc.
    use_case: str = ""
    expected_output: str = ""
    reasoning: str = ""
    related_tools: List[str] = field(default_factory=list)
    source: str = ""


@dataclass
class CheatsheetEntry:
    """Consolidated cheat sheet section."""
    tool_or_topic: str
    section: str = ""
    content: str = ""
    commands: List[Dict[str, str]] = field(default_factory=list)
    source: str = ""


@dataclass
class TechniqueEntry:
    """MITRE ATT&CK technique with atomic tests and commands."""
    technique_id: str          # e.g. T1059.001
    technique_name: str = ""
    tactic: str = ""           # e.g. execution, persistence, privilege-escalation
    description: str = ""
    platforms: List[str] = field(default_factory=list)  # windows, linux, macos
    atomic_tests: List[Dict[str, str]] = field(default_factory=list)  # name, command, cleanup
    detection: str = ""
    prerequisites: List[str] = field(default_factory=list)
    reasoning: str = ""        # WHY this technique works
    source: str = ""


@dataclass
class BinaryEntry:
    """GTFOBins/LOLBAS binary abuse method."""
    binary_name: str
    platform: str = "linux"    # linux or windows
    functions: List[Dict[str, Any]] = field(default_factory=list)  # shell, file-read, suid, sudo, etc.
    description: str = ""
    reasoning: str = ""
    source: str = ""


@dataclass
class ExploitDBEntry:
    """Exploit from ExploitDB."""
    edb_id: str
    description: str = ""
    date_published: str = ""
    author: str = ""
    platform: str = ""
    exploit_type: str = ""     # local, remote, webapps, dos
    port: int = 0
    cve: str = ""
    codes: List[str] = field(default_factory=list)
    source_file: str = ""
    source: str = "exploitdb"


@dataclass
class ADAttackEntry:
    """Active Directory attack methodology."""
    attack_name: str
    category: str = ""         # enumeration, credential-access, lateral-movement, persistence
    description: str = ""
    commands: List[Dict[str, str]] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    reasoning: str = ""
    source: str = ""


@dataclass
class CloudAttackEntry:
    """Cloud attack methodology (AWS/Azure/GCP)."""
    attack_name: str
    cloud_provider: str = ""   # aws, azure, gcp
    category: str = ""
    description: str = ""
    commands: List[Dict[str, str]] = field(default_factory=list)
    reasoning: str = ""
    source: str = ""


@dataclass
class CTFWriteupEntry:
    """CTF challenge writeup with solution."""
    challenge_name: str
    category: str = ""         # web, pwn, crypto, forensics, misc, rev
    difficulty: str = ""
    description: str = ""
    solution_steps: List[str] = field(default_factory=list)
    commands: List[Dict[str, str]] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    flag: str = ""
    reasoning: str = ""
    source: str = ""


# ─── Parsing Utilities ──────────────────────────────────────────────────────

# Port to common service mapping
PORT_SERVICE_MAP = {
    21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp", 53: "dns",
    69: "tftp", 80: "http", 88: "kerberos", 110: "pop3", 111: "rpcbind",
    135: "msrpc", 139: "netbios", 143: "imap", 161: "snmp", 389: "ldap",
    443: "https", 445: "smb", 464: "kpasswd", 512: "rexec", 513: "rlogin",
    514: "rsh", 515: "lpd", 548: "afp", 554: "rtsp", 587: "submission",
    623: "ipmi", 631: "ipp", 636: "ldaps", 873: "rsync", 993: "imaps",
    995: "pop3s", 1099: "java-rmi", 1433: "mssql", 1521: "oracle",
    1524: "ingreslock", 2049: "nfs", 2181: "zookeeper", 3000: "grafana",
    3306: "mysql", 3389: "rdp", 3632: "distcc", 4369: "erlang",
    5432: "postgresql", 5555: "adb", 5601: "kibana", 5672: "amqp",
    5900: "vnc", 5985: "winrm", 6379: "redis", 6667: "irc",
    8000: "http-alt", 8080: "http-proxy", 8180: "tomcat", 8443: "https-alt",
    8888: "sun-answerbook", 9090: "prometheus", 9200: "elasticsearch",
    9418: "git", 11211: "memcached", 27017: "mongodb", 50000: "ibm-db2",
}

# Known common CVE patterns
CVE_PATTERN = re.compile(r'CVE-\d{4}-\d{4,}', re.IGNORECASE)

# Command extraction patterns
COMMAND_BLOCK_PATTERN = re.compile(
    r'```(?:bash|sh|shell|console|powershell|cmd|python|ruby|perl)?\s*\n(.*?)\n```',
    re.DOTALL,
)

# Credential patterns
CRED_PATTERN = re.compile(
    r'(?:username|user|login|cred)[\s:=]+["\']?(\w+)["\']?.*?'
    r'(?:password|pass|pwd)[\s:=]+["\']?(\S+)["\']?',
    re.IGNORECASE,
)

# Port extraction from headings
PORT_HEADING_PATTERN = re.compile(
    r'(?:^#+\s*)?(\d+)\s*[-–/]\s*(.+)',
    re.MULTILINE,
)


def extract_commands_from_markdown(text: str) -> List[Dict[str, str]]:
    """Extract command blocks from markdown with surrounding context."""
    commands = []
    # Find fenced code blocks
    for match in COMMAND_BLOCK_PATTERN.finditer(text):
        cmd_text = match.group(1).strip()
        if not cmd_text or len(cmd_text) < 3:
            continue
        # Get context (preceding heading or paragraph)
        pre_start = max(0, match.start() - 500)
        pre_text = text[pre_start:match.start()]
        # Find last heading
        heading_match = re.findall(r'^#+\s*(.+)$', pre_text, re.MULTILINE)
        heading = heading_match[-1].strip() if heading_match else ""
        # Find preceding paragraph
        paras = [p.strip() for p in pre_text.split('\n\n') if p.strip()]
        context = paras[-1][:200] if paras else ""

        # Split multi-line commands
        for line in cmd_text.split('\n'):
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('//'):
                continue
            # Skip pure output lines (common in examples)
            if line.startswith('[') and ']' in line[:20]:
                continue
            if line.startswith('>>>') or line.startswith('...'):
                line = line.lstrip('>. ')

            commands.append({
                "command": line[:500],
                "context": context[:300],
                "heading": heading[:200],
            })

    # Also extract inline commands (backtick-wrapped)
    inline_cmds = re.findall(r'(?:run|execute|use|type|enter):\s*`([^`]+)`', text, re.IGNORECASE)
    for cmd in inline_cmds:
        if len(cmd) > 5:
            commands.append({"command": cmd[:500], "context": "inline reference", "heading": ""})

    return commands


def extract_cves(text: str) -> List[str]:
    """Extract CVE identifiers from text."""
    return list(set(CVE_PATTERN.findall(text)))


def extract_credentials(text: str) -> List[Dict[str, str]]:
    """Extract username:password pairs from text."""
    creds = []
    for match in CRED_PATTERN.finditer(text):
        creds.append({"username": match.group(1), "password": match.group(2)})

    # Also catch user:pass patterns in code blocks
    for match in re.finditer(r'(\w+):(\S+)@', text):
        u, p = match.group(1), match.group(2)
        if len(u) > 2 and len(p) > 2 and u not in ('http', 'https', 'ftp', 'ssh'):
            creds.append({"username": u, "password": p})

    return creds


def extract_metasploit_modules(text: str) -> List[str]:
    """Extract Metasploit module paths from text."""
    return list(set(re.findall(
        r'(?:use\s+)?(exploit/\S+|auxiliary/\S+|post/\S+|payload/\S+)',
        text,
    )))


def classify_phase(text: str, heading: str = "") -> str:
    """Classify text content into kill-chain phase."""
    combined = (text + " " + heading).lower()
    if any(w in combined for w in ['recon', 'footprint', 'osint', 'dns enum']):
        return "RECON"
    if any(w in combined for w in ['enumerat', 'scan', 'fingerprint', 'nmap', 'port scan']):
        return "ENUMERATION"
    if any(w in combined for w in ['exploit', 'vulnerability', 'rce', 'injection', 'payload']):
        return "EXPLOITATION"
    if any(w in combined for w in ['privesc', 'privilege', 'escalat', 'root', 'sudo', 'suid']):
        return "PRIVILEGE_ESCALATION"
    if any(w in combined for w in ['lateral', 'pivot', 'tunnel', 'port forward']):
        return "LATERAL_MOVEMENT"
    if any(w in combined for w in ['post-exploit', 'post exploit', 'persistence', 'backdoor']):
        return "POST_EXPLOITATION"
    if any(w in combined for w in ['exfiltrat', 'data steal', 'extract']):
        return "EXFILTRATION"
    return "GENERAL"


def split_markdown_sections(text: str) -> List[Tuple[str, str, int]]:
    """Split markdown into (heading, content, level) tuples."""
    sections = []
    # Match markdown headings
    pattern = re.compile(r'^(#{1,4})\s+(.+)$', re.MULTILINE)
    matches = list(pattern.finditer(text))

    for i, match in enumerate(matches):
        level = len(match.group(1))
        heading = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if content:
            sections.append((heading, content, level))

    # If no headings found, treat entire text as one section
    if not sections and text.strip():
        sections.append(("", text.strip(), 0))

    return sections


# ─── Per-Repo Parsers ───────────────────────────────────────────────────────

class HackTricksParser:
    """Parse HackTricks repository — per-port/service pentesting guides."""

    def __init__(self, repo_dir: Path):
        self.repo_dir = repo_dir
        self.services: List[ServiceEntry] = []
        self.kill_chains: List[KillChainEntry] = []
        self.commands: List[CommandEntry] = []
        self.methodology: List[MethodologyEntry] = []
        self.privesc_linux: List[Dict[str, Any]] = []
        self.privesc_windows: List[Dict[str, Any]] = []

    def parse(self) -> Dict[str, Any]:
        """Parse all HackTricks content."""
        logger.info("Parsing HackTricks repository...")
        stats = {"files": 0, "services": 0, "commands": 0, "methodology": 0}

        # Parse network services (pentesting-by-port)
        # HackTricks uses src/ subdirectory in newer versions
        for base in [self.repo_dir, self.repo_dir / "src"]:
            services_dir = base / "network-services-pentesting"
            if services_dir.exists():
                self._parse_services_dir(services_dir)
                break

            alt_dir = base / "pentesting-network"
            if alt_dir.exists():
                self._parse_services_dir(alt_dir)
                break
        stats["services"] = len(self.services)

        # Parse Linux privesc — check both root and src/ paths
        linux_privesc = None
        for base in [self.repo_dir, self.repo_dir / "src"]:
            for path in [
                base / "linux-hardening" / "privilege-escalation",
                base / "linux-unix" / "privilege-escalation",
            ]:
                if path.exists():
                    linux_privesc = path
                    break
            if linux_privesc:
                break
        if linux_privesc:
            self._parse_privesc_dir(linux_privesc, "linux")

        # Parse Windows privesc — check both root and src/ paths
        win_privesc = None
        for base in [self.repo_dir, self.repo_dir / "src"]:
            for path in [
                base / "windows-hardening" / "windows-local-privilege-escalation",
                base / "windows" / "windows-local-privilege-escalation",
            ]:
                if path.exists():
                    win_privesc = path
                    break
            if win_privesc:
                break
        if win_privesc:
            self._parse_privesc_dir(win_privesc, "windows")

        # Parse generic methodology
        for md_file in self.repo_dir.rglob("*.md"):
            if any(skip in str(md_file) for skip in ['.git', 'node_modules', 'SUMMARY']):
                continue
            try:
                text = md_file.read_text(encoding='utf-8', errors='ignore')
                if len(text) < 100:
                    continue
                self._extract_commands(text, str(md_file.relative_to(self.repo_dir)))
                stats["files"] += 1
            except Exception:
                continue

        stats["commands"] = len(self.commands)
        stats["methodology"] = len(self.methodology)
        logger.info(f"HackTricks: {stats}")
        return stats

    def _parse_services_dir(self, services_dir: Path):
        """Parse per-port/service markdown files."""
        for md_file in sorted(services_dir.rglob("*.md")):
            if '.git' in str(md_file):
                continue
            try:
                text = md_file.read_text(encoding='utf-8', errors='ignore')
                if len(text) < 50:
                    continue

                fname = md_file.stem.lower()
                # Extract port number from filename
                port_match = re.match(r'(\d+)', fname)
                port = int(port_match.group(1)) if port_match else 0

                # Determine service name
                service_name = PORT_SERVICE_MAP.get(port, fname.replace('-', ' ').replace('_', ' '))

                # Extract sections
                sections = split_markdown_sections(text)
                enum_cmds = []
                exploit_cmds = []
                methodology_text = []
                reasoning_parts = []
                cves = extract_cves(text)
                creds = extract_credentials(text)
                msf_modules = extract_metasploit_modules(text)

                for heading, content, level in sections:
                    h_lower = heading.lower()
                    cmds = extract_commands_from_markdown(content)

                    if any(w in h_lower for w in ['enumerat', 'scan', 'discover', 'banner', 'basic']):
                        for c in cmds:
                            c["phase"] = "ENUMERATION"
                        enum_cmds.extend(cmds)
                    elif any(w in h_lower for w in ['exploit', 'attack', 'rce', 'inject', 'brute']):
                        for c in cmds:
                            c["phase"] = "EXPLOITATION"
                        exploit_cmds.extend(cmds)
                    elif any(w in h_lower for w in ['priv', 'escalat', 'root', 'post']):
                        for c in cmds:
                            c["phase"] = "PRIVILEGE_ESCALATION"
                        exploit_cmds.extend(cmds)
                    else:
                        enum_cmds.extend(cmds)

                    # Collect methodology text
                    if len(content) > 50:
                        methodology_text.append(f"## {heading}\n{content[:1000]}")

                    # Collect reasoning (anything that explains WHY)
                    if any(w in content.lower() for w in [
                        'because', 'reason', 'since', 'therefore', 'this works',
                        'the idea', 'exploit works', 'vulnerability', 'allows',
                        'this means', 'we can', 'trick is',
                    ]):
                        reasoning_parts.append(content[:500])

                entry = ServiceEntry(
                    service_name=service_name,
                    port=port,
                    description=sections[0][1][:500] if sections else "",
                    enumeration_commands=enum_cmds[:50],
                    exploitation_commands=exploit_cmds[:50],
                    default_credentials=creds[:10],
                    common_vulnerabilities=cves[:20],
                    methodology="\n\n".join(methodology_text[:10])[:5000],
                    reasoning="\n".join(reasoning_parts[:5])[:3000],
                    kill_chain_phase=classify_phase(text, fname),
                    source="hacktricks",
                    tags=list(set([service_name, str(port)] + msf_modules[:5])),
                )
                self.services.append(entry)

                # Build kill chains from exploit sections
                if exploit_cmds and port > 0:
                    chain = KillChainEntry(
                        target_service=service_name,
                        target_port=port,
                        chain_name=f"hacktricks_{service_name}_{port}",
                        steps=[{"step": i + 1, **c} for i, c in enumerate(exploit_cmds[:15])],
                        total_commands=len(exploit_cmds),
                        end_result="shell" if any('shell' in c.get("context", "").lower()
                                                   for c in exploit_cmds) else "access",
                        reasoning=entry.reasoning[:1000],
                        source="hacktricks",
                    )
                    self.kill_chains.append(chain)

            except Exception as e:
                logger.debug(f"Error parsing {md_file}: {e}")

    def _parse_privesc_dir(self, privesc_dir: Path, os_type: str):
        """Parse privilege escalation guides."""
        target = self.privesc_linux if os_type == "linux" else self.privesc_windows
        for md_file in sorted(privesc_dir.rglob("*.md")):
            if '.git' in str(md_file):
                continue
            try:
                text = md_file.read_text(encoding='utf-8', errors='ignore')
                sections = split_markdown_sections(text)
                for heading, content, level in sections:
                    cmds = extract_commands_from_markdown(content)
                    if cmds:
                        target.append({
                            "technique": heading,
                            "description": content[:1000],
                            "commands": cmds[:20],
                            "os": os_type,
                            "reasoning": content[:500] if any(
                                w in content.lower() for w in ['because', 'allows', 'this works']
                            ) else "",
                            "source": f"hacktricks/{md_file.stem}",
                        })
            except Exception:
                continue

    def _extract_commands(self, text: str, source: str):
        """Extract standalone commands with context."""
        cmds = extract_commands_from_markdown(text)
        for c in cmds[:20]:  # Cap per file
            phase = classify_phase(c.get("context", "") + c.get("heading", ""))
            tool = c["command"].split()[0] if c["command"].split() else ""
            self.commands.append(CommandEntry(
                tool_name=tool,
                command=c["command"],
                description=c.get("heading", ""),
                phase=phase,
                use_case=c.get("context", ""),
                source=f"hacktricks/{source}",
            ))


class PayloadsParser:
    """Parse PayloadsAllTheThings — vulnerability-specific payloads."""

    def __init__(self, repo_dir: Path):
        self.repo_dir = repo_dir
        self.payloads: List[PayloadEntry] = []
        self.methodology: List[MethodologyEntry] = []
        self.commands: List[CommandEntry] = []

    def parse(self) -> Dict[str, Any]:
        logger.info("Parsing PayloadsAllTheThings...")
        stats = {"files": 0, "payloads": 0, "methodology": 0}

        # Map directory names to vuln types
        vuln_type_map = {
            'sql injection': 'sqli', 'xss': 'xss', 'ssti': 'ssti',
            'command injection': 'cmdi', 'ssrf': 'ssrf', 'lfi': 'lfi',
            'file inclusion': 'lfi', 'rfi': 'rfi', 'xxe': 'xxe',
            'deserialization': 'deserialization', 'upload': 'file_upload',
            'directory traversal': 'traversal', 'ldap': 'ldap_injection',
            'nosql': 'nosql_injection', 'graphql': 'graphql',
            'cors': 'cors', 'csrf': 'csrf', 'idor': 'idor',
            'open redirect': 'open_redirect', 'race condition': 'race',
            'request smuggling': 'smuggling', 'prototype pollution': 'prototype_pollution',
            'reverse shell': 'reverse_shell', 'methodology': 'methodology',
            'linux privilege': 'privesc_linux', 'windows privilege': 'privesc_windows',
        }

        for md_file in sorted(self.repo_dir.rglob("*.md")):
            if any(skip in str(md_file) for skip in ['.git', 'node_modules']):
                continue
            try:
                text = md_file.read_text(encoding='utf-8', errors='ignore')
                if len(text) < 100:
                    continue

                # Determine vuln type from path
                rel_path = str(md_file.relative_to(self.repo_dir)).lower()
                vuln_type = "general"
                for key, vtype in vuln_type_map.items():
                    if key in rel_path:
                        vuln_type = vtype
                        break

                sections = split_markdown_sections(text)
                for heading, content, level in sections:
                    cmds = extract_commands_from_markdown(content)

                    # Extract payloads (often in code blocks or inline)
                    for cmd_info in cmds:
                        cmd = cmd_info["command"]
                        # Detect if it's a payload vs a regular command
                        is_payload = any(w in cmd.lower() for w in [
                            "' or", "' and", "<script", "{{", "${", "../",
                            "union select", "sleep(", "ping -c", "curl ",
                            "%00", "..\\", "<img", "onerror", "onload",
                        ])
                        if is_payload:
                            self.payloads.append(PayloadEntry(
                                vuln_type=vuln_type,
                                payload=cmd[:1000],
                                context=cmd_info.get("heading", "") or heading,
                                reasoning=cmd_info.get("context", ""),
                                source=f"payloads/{rel_path}",
                            ))
                        else:
                            self.commands.append(CommandEntry(
                                tool_name=cmd.split()[0] if cmd.split() else "",
                                command=cmd,
                                description=heading,
                                phase=classify_phase(content, heading),
                                use_case=cmd_info.get("context", ""),
                                source=f"payloads/{rel_path}",
                            ))

                    # Build methodology entries from substantial sections
                    if len(content) > 200:
                        self.methodology.append(MethodologyEntry(
                            phase=classify_phase(content, heading),
                            title=f"{vuln_type}: {heading}",
                            steps=[s.strip() for s in content.split('\n') if s.strip().startswith(('1.', '2.', '3.', '-', '*'))],
                            commands=cmds[:10],
                            reasoning=content[:1000],
                            source=f"payloads/{rel_path}",
                            tags=[vuln_type],
                        ))

                stats["files"] += 1

            except Exception as e:
                logger.debug(f"Error parsing {md_file}: {e}")

        stats["payloads"] = len(self.payloads)
        stats["methodology"] = len(self.methodology)
        logger.info(f"PayloadsAllTheThings: {stats}")
        return stats


class OSCPParser:
    """Parse 0xsyr0/OSCP — massive cheat sheet."""

    def __init__(self, repo_dir: Path):
        self.repo_dir = repo_dir
        self.commands: List[CommandEntry] = []
        self.cheatsheets: List[CheatsheetEntry] = []
        self.methodology: List[MethodologyEntry] = []

    def parse(self) -> Dict[str, Any]:
        logger.info("Parsing OSCP cheat sheet...")
        stats = {"files": 0, "commands": 0, "sections": 0}

        # Main README is the massive cheat sheet
        for md_file in sorted(self.repo_dir.rglob("*.md")):
            if '.git' in str(md_file):
                continue
            try:
                text = md_file.read_text(encoding='utf-8', errors='ignore')
                if len(text) < 100:
                    continue

                sections = split_markdown_sections(text)
                for heading, content, level in sections:
                    cmds = extract_commands_from_markdown(content)

                    if cmds:
                        # Determine the tool/topic
                        tool = heading.split('(')[0].strip().split('/')[0].strip()

                        self.cheatsheets.append(CheatsheetEntry(
                            tool_or_topic=tool,
                            section=heading,
                            content=content[:3000],
                            commands=cmds[:30],
                            source="oscp",
                        ))

                        for c in cmds[:15]:
                            self.commands.append(CommandEntry(
                                tool_name=c["command"].split()[0] if c["command"].split() else tool,
                                command=c["command"],
                                description=heading,
                                phase=classify_phase(content, heading),
                                use_case=c.get("context", ""),
                                source="oscp",
                            ))

                    # Methodology for substantial sections
                    if len(content) > 300:
                        self.methodology.append(MethodologyEntry(
                            phase=classify_phase(content, heading),
                            title=heading,
                            commands=cmds[:10],
                            reasoning=content[:2000],
                            source="oscp",
                            tags=[tool.lower()] if tool else [],
                        ))

                stats["files"] += 1
                stats["sections"] += len(sections)

            except Exception:
                continue

        stats["commands"] = len(self.commands)
        logger.info(f"OSCP: {stats}")
        return stats


class H4ckerParser:
    """Parse The-Art-of-Hacking/h4cker — methodology + cheat sheets."""

    def __init__(self, repo_dir: Path):
        self.repo_dir = repo_dir
        self.commands: List[CommandEntry] = []
        self.cheatsheets: List[CheatsheetEntry] = []
        self.methodology: List[MethodologyEntry] = []

    def parse(self) -> Dict[str, Any]:
        logger.info("Parsing h4cker repository...")
        stats = {"files": 0, "commands": 0, "methodology": 0}

        priority_dirs = [
            "cheat_sheets", "cheat-sheets",
            "methodology",
            "post-exploitation", "post_exploitation",
            "buffer_overflow_examples", "buffer-overflow-examples",
            "temp/Ethical_Hacking",
        ]

        for md_file in sorted(self.repo_dir.rglob("*.md")):
            if any(skip in str(md_file) for skip in ['.git', 'node_modules']):
                continue
            try:
                text = md_file.read_text(encoding='utf-8', errors='ignore')
                if len(text) < 100:
                    continue

                rel_path = str(md_file.relative_to(self.repo_dir))
                is_priority = any(d in rel_path.lower() for d in priority_dirs)

                sections = split_markdown_sections(text)
                for heading, content, level in sections:
                    cmds = extract_commands_from_markdown(content)

                    if cmds and is_priority:
                        tool = md_file.stem.replace('_', ' ').replace('-', ' ')
                        self.cheatsheets.append(CheatsheetEntry(
                            tool_or_topic=tool,
                            section=heading,
                            content=content[:3000],
                            commands=cmds[:25],
                            source=f"h4cker/{rel_path}",
                        ))

                    for c in cmds[:10]:
                        self.commands.append(CommandEntry(
                            tool_name=c["command"].split()[0] if c["command"].split() else "",
                            command=c["command"],
                            description=heading,
                            phase=classify_phase(content, heading),
                            use_case=c.get("context", ""),
                            source=f"h4cker/{rel_path}",
                        ))

                    if len(content) > 200 and is_priority:
                        self.methodology.append(MethodologyEntry(
                            phase=classify_phase(content, heading),
                            title=heading,
                            commands=cmds[:10],
                            reasoning=content[:1500],
                            source=f"h4cker/{rel_path}",
                        ))

                stats["files"] += 1

            except Exception:
                continue

        stats["commands"] = len(self.commands)
        stats["methodology"] = len(self.methodology)
        logger.info(f"h4cker: {stats}")
        return stats


class VulhubParser:
    """Parse vulhub — CVE reproduction environments with exploitation tutorials."""

    def __init__(self, repo_dir: Path):
        self.repo_dir = repo_dir
        self.cves: List[CVEEntry] = []
        self.kill_chains: List[KillChainEntry] = []

    def parse(self) -> Dict[str, Any]:
        logger.info("Parsing Vulhub...")
        stats = {"environments": 0, "cves": 0}

        for readme in sorted(self.repo_dir.rglob("README.md")):
            if '.git' in str(readme):
                continue
            try:
                text = readme.read_text(encoding='utf-8', errors='ignore')
                if len(text) < 100:
                    continue

                rel_path = str(readme.parent.relative_to(self.repo_dir))
                parts = rel_path.split('/')

                # Skip root README
                if len(parts) < 2:
                    continue

                software = parts[0] if parts else "unknown"
                cve_ids = extract_cves(text)
                cmds = extract_commands_from_markdown(text)
                msf = extract_metasploit_modules(text)

                for cve_id in cve_ids:
                    self.cves.append(CVEEntry(
                        cve_id=cve_id,
                        affected_software=software,
                        description=text[:500],
                        exploit_methodology=text[:3000],
                        exploit_commands=cmds[:20],
                        metasploit_module=msf[0] if msf else "",
                        reasoning=text[:2000],
                        source=f"vulhub/{rel_path}",
                    ))

                # Build kill chain from README exploitation steps
                if cmds:
                    self.kill_chains.append(KillChainEntry(
                        target_service=software,
                        chain_name=f"vulhub_{rel_path.replace('/', '_')}",
                        steps=[{"step": i + 1, **c} for i, c in enumerate(cmds[:15])],
                        total_commands=len(cmds),
                        end_result="rce" if any(
                            w in text.lower() for w in ['rce', 'shell', 'reverse']
                        ) else "access",
                        reasoning=text[:1000],
                        source=f"vulhub/{rel_path}",
                    ))

                stats["environments"] += 1

            except Exception:
                continue

        stats["cves"] = len(self.cves)
        logger.info(f"Vulhub: {stats}")
        return stats


class PentestWikiParser:
    """Parse nixawk/pentest-wiki — kill-chain organized pentesting."""

    def __init__(self, repo_dir: Path):
        self.repo_dir = repo_dir
        self.commands: List[CommandEntry] = []
        self.methodology: List[MethodologyEntry] = []
        self.kill_chains: List[KillChainEntry] = []
        self.services: List[ServiceEntry] = []

    def parse(self) -> Dict[str, Any]:
        logger.info("Parsing pentest-wiki...")
        stats = {"files": 0, "commands": 0, "methodology": 0}

        # Map directory structure to phases
        phase_map = {
            '1.information-gathering': 'RECON',
            '1.information_gathering': 'RECON',
            '2.vulnerability-assessment': 'ENUMERATION',
            '2.vulnerability_assessment': 'ENUMERATION',
            '3.exploitation': 'EXPLOITATION',
            '3.exploitation-tools': 'EXPLOITATION',
            '4.post-exploitation': 'POST_EXPLOITATION',
            '4.post_exploitation': 'POST_EXPLOITATION',
            'ctfs': 'EXPLOITATION',
        }

        for md_file in sorted(self.repo_dir.rglob("*.md")):
            if '.git' in str(md_file):
                continue
            try:
                text = md_file.read_text(encoding='utf-8', errors='ignore')
                if len(text) < 100:
                    continue

                rel_path = str(md_file.relative_to(self.repo_dir)).lower()

                # Determine phase from directory
                phase = "GENERAL"
                for dir_name, p in phase_map.items():
                    if dir_name in rel_path:
                        phase = p
                        break

                sections = split_markdown_sections(text)
                cmds = extract_commands_from_markdown(text)

                for c in cmds[:20]:
                    self.commands.append(CommandEntry(
                        tool_name=c["command"].split()[0] if c["command"].split() else "",
                        command=c["command"],
                        description=c.get("heading", md_file.stem),
                        phase=phase,
                        use_case=c.get("context", ""),
                        source=f"pentest_wiki/{rel_path}",
                    ))

                for heading, content, level in sections:
                    if len(content) > 200:
                        section_cmds = extract_commands_from_markdown(content)
                        self.methodology.append(MethodologyEntry(
                            phase=phase,
                            title=heading,
                            commands=section_cmds[:10],
                            reasoning=content[:2000],
                            source=f"pentest_wiki/{rel_path}",
                            tags=[phase.lower()],
                        ))

                # Extract port-based service entries from exploitation tools tables
                port_matches = re.findall(r'\|\s*(\d+)\s*\|\s*([^|]+)\|', text)
                for port_str, svc_desc in port_matches:
                    try:
                        port = int(port_str)
                        if 1 <= port <= 65535:
                            svc_name = svc_desc.strip().split('(')[0].strip().lower()
                            self.services.append(ServiceEntry(
                                service_name=svc_name[:50],
                                port=port,
                                description=svc_desc.strip()[:200],
                                source=f"pentest_wiki/{rel_path}",
                            ))
                    except ValueError:
                        continue

                stats["files"] += 1

            except Exception:
                continue

        stats["commands"] = len(self.commands)
        stats["methodology"] = len(self.methodology)
        logger.info(f"pentest-wiki: {stats}")
        return stats


# ─── Phase 9.2: NEW PARSERS (15 repos) ─────────────────────────────────────

class GTFOBinsParser:
    """Parse GTFOBins — Unix binary privesc/shell escape methods."""

    def __init__(self, repo_dir: Path):
        self.repo_dir = repo_dir
        self.binaries: List[BinaryEntry] = []
        self.commands: List[CommandEntry] = []

    def parse(self) -> Dict[str, Any]:
        logger.info("Parsing GTFOBins...")
        stats = {"files": 0, "binaries": 0, "commands": 0}
        # GTFOBins stores YAML + markdown in _gtfobins/ directory
        bins_dir = self.repo_dir / "_gtfobins"
        if not bins_dir.exists():
            # Try alternate structure
            for alt in ["gtfobins", "GTFOBins"]:
                alt_dir = self.repo_dir / alt
                if alt_dir.exists():
                    bins_dir = alt_dir
                    break

        if not bins_dir.exists():
            # Parse any .md files at root
            bins_dir = self.repo_dir

        # GTFOBins files are extensionless YAML files (not .md)
        candidates = list(sorted(bins_dir.iterdir()))
        for entry_file in candidates:
            if not entry_file.is_file():
                continue
            if entry_file.name.startswith(("_", ".")) or entry_file.suffix:
                continue  # Skip dotfiles, underscored, and files WITH extensions
            try:
                text = entry_file.read_text(encoding="utf-8", errors="ignore")
                binary_name = entry_file.name  # extensionless filename IS the binary name
                functions = []

                # GTFOBins files are pure YAML with functions: dict
                try:
                    import yaml
                    data = yaml.safe_load(text)
                    if isinstance(data, dict) and "functions" in data:
                        func_data = data["functions"]
                        if isinstance(func_data, dict):
                            for func_name, func_entries in func_data.items():
                                if isinstance(func_entries, list):
                                    for entry in func_entries:
                                        code = entry.get("code", "") if isinstance(entry, dict) else str(entry)
                                        desc = entry.get("description", "") if isinstance(entry, dict) else ""
                                        functions.append({
                                            "type": func_name,
                                            "code": code,
                                            "description": desc,
                                        })
                                        if code:
                                            self.commands.append(CommandEntry(
                                                tool_name=binary_name,
                                                command=code,
                                                description=f"GTFOBins {func_name}: {desc}" if desc else f"GTFOBins {func_name}",
                                                phase="PRIVILEGE_ESCALATION" if func_name in ("suid", "sudo", "capabilities") else "EXPLOITATION",
                                                use_case=f"Abuse {binary_name} for {func_name}",
                                                reasoning=f"Binary {binary_name} can be abused via {func_name} function to escalate privileges or escape restricted shells",
                                                source=f"gtfobins/{binary_name}",
                                            ))
                except ImportError:
                    pass  # No yaml module
                except Exception:
                    pass

                if functions:
                    self.binaries.append(BinaryEntry(
                        binary_name=binary_name,
                        platform="linux",
                        functions=functions,
                        description=f"GTFOBins: {binary_name} — {len(functions)} abuse methods",
                        source="gtfobins",
                    ))
                    stats["binaries"] += 1

                stats["files"] += 1
            except Exception:
                continue

        stats["commands"] = len(self.commands)
        logger.info(f"GTFOBins: {stats}")
        return stats


class LOLBASParser:
    """Parse LOLBAS — Windows Living Off The Land Binaries."""

    def __init__(self, repo_dir: Path):
        self.repo_dir = repo_dir
        self.binaries: List[BinaryEntry] = []
        self.commands: List[CommandEntry] = []

    def parse(self) -> Dict[str, Any]:
        logger.info("Parsing LOLBAS...")
        stats = {"files": 0, "binaries": 0, "commands": 0}

        # LOLBAS stores YAML-frontmatter .md files in _lolbas/ directory
        lolbas_dir = self.repo_dir / "_lolbas"
        if not lolbas_dir.exists():
            # Try alternate dirs
            for alt in ["yml", "_data"]:
                alt_dir = self.repo_dir / alt
                if alt_dir.exists():
                    lolbas_dir = alt_dir
                    break

        if not lolbas_dir.exists():
            logger.warning("LOLBAS _lolbas/ directory not found")
            return stats

        for md_file in sorted(lolbas_dir.rglob("*.md")):
            if md_file.name in ("README.md", "NOTICE.md") or ".git" in str(md_file):
                continue
            try:
                text = md_file.read_text(encoding="utf-8", errors="ignore")
                if not text.startswith("---"):
                    continue

                parts = text.split("---", 2)
                if len(parts) < 3:
                    continue

                import yaml
                data = yaml.safe_load(parts[1])
                if not isinstance(data, dict):
                    continue

                binary_name = data.get("Name", md_file.stem)
                desc = data.get("Description", "")
                functions = []

                commands_data = data.get("Commands", [])
                if isinstance(commands_data, list):
                    for cmd_entry in commands_data:
                        if not isinstance(cmd_entry, dict):
                            continue
                        cmd_str = cmd_entry.get("Command", "")
                        cmd_desc = cmd_entry.get("Description", "")
                        usecase = cmd_entry.get("Usecase", "")
                        category = cmd_entry.get("Category", "")
                        mitre_id = cmd_entry.get("MitreID", "")

                        if cmd_str:
                            functions.append({
                                "type": category or "generic",
                                "code": cmd_str,
                                "description": cmd_desc,
                                "usecase": usecase,
                                "mitre_id": mitre_id,
                            })
                            self.commands.append(CommandEntry(
                                tool_name=binary_name,
                                command=cmd_str,
                                description=f"LOLBAS {category}: {cmd_desc}" if cmd_desc else f"LOLBAS {binary_name}",
                                phase=self._category_to_phase(category),
                                use_case=usecase,
                                reasoning=f"Windows LOLBin {binary_name} — legitimate binary abusable for {category}",
                                source=f"lolbas/{binary_name}",
                            ))

                if functions:
                    self.binaries.append(BinaryEntry(
                        binary_name=binary_name,
                        platform="windows",
                        functions=functions,
                        description=desc[:500],
                        source="lolbas",
                    ))
                    stats["binaries"] += 1

                stats["files"] += 1
            except Exception:
                continue

        stats["commands"] = len(self.commands)
        logger.info(f"LOLBAS: {stats}")
        return stats

    @staticmethod
    def _category_to_phase(cat: str) -> str:
        cat_l = cat.lower()
        if any(w in cat_l for w in ["execute", "awl bypass"]):
            return "EXPLOITATION"
        if any(w in cat_l for w in ["download", "upload", "copy"]):
            return "EXFILTRATION"
        if any(w in cat_l for w in ["recon", "discover"]):
            return "ENUMERATION"
        if "compile" in cat_l:
            return "POST_EXPLOITATION"
        return "EXPLOITATION"


class WADComsParser:
    """Parse WADComs — Windows/AD command cheatsheet."""

    def __init__(self, repo_dir: Path):
        self.repo_dir = repo_dir
        self.commands: List[CommandEntry] = []
        self.ad_attacks: List[ADAttackEntry] = []

    def parse(self) -> Dict[str, Any]:
        logger.info("Parsing WADComs...")
        stats = {"files": 0, "commands": 0, "ad_attacks": 0}

        # WADComs uses _wadcoms/ with YAML-frontmatter markdown files
        wadcoms_dir = self.repo_dir / "_wadcoms"
        if not wadcoms_dir.exists():
            wadcoms_dir = self.repo_dir

        for md_file in sorted(wadcoms_dir.rglob("*.md")):
            if md_file.name.startswith(("_", ".")):
                continue
            try:
                text = md_file.read_text(encoding="utf-8", errors="ignore")
                # Parse YAML front matter
                if text.startswith("---"):
                    parts = text.split("---", 2)
                    if len(parts) >= 3:
                        try:
                            import yaml
                            data = yaml.safe_load(parts[1])
                            if isinstance(data, dict):
                                title = data.get("title", md_file.stem)
                                desc = data.get("description", "")
                                command = data.get("command", "")
                                items = data.get("items", [])

                                if command:
                                    self.commands.append(CommandEntry(
                                        tool_name=title.split()[0] if title else md_file.stem,
                                        command=command,
                                        description=desc,
                                        phase="EXPLOITATION",
                                        use_case=f"AD attack: {title}",
                                        reasoning=f"Active Directory attack using {title}",
                                        source=f"wadcoms/{md_file.stem}",
                                    ))

                                # Build AD attack entry
                                tools = [it for it in items if isinstance(it, str)] if isinstance(items, list) else []
                                body = parts[2] if len(parts) > 2 else ""
                                body_cmds = extract_commands_from_markdown(body)
                                all_cmds = [{"command": command}] if command else []
                                all_cmds.extend(body_cmds)

                                for bc in body_cmds:
                                    self.commands.append(CommandEntry(
                                        tool_name=bc["command"].split()[0] if bc["command"].split() else title,
                                        command=bc["command"],
                                        description=bc.get("heading", title),
                                        phase="EXPLOITATION",
                                        source=f"wadcoms/{md_file.stem}",
                                    ))

                                self.ad_attacks.append(ADAttackEntry(
                                    attack_name=title,
                                    category="active-directory",
                                    description=desc,
                                    commands=all_cmds[:10],
                                    tools_used=tools[:10],
                                    source=f"wadcoms/{md_file.stem}",
                                ))
                                stats["ad_attacks"] += 1
                        except ImportError:
                            pass
                        except Exception:
                            pass

                stats["files"] += 1
            except Exception:
                continue

        stats["commands"] = len(self.commands)
        logger.info(f"WADComs: {stats}")
        return stats


class AtomicRedTeamParser:
    """Parse Atomic Red Team — MITRE ATT&CK technique tests."""

    def __init__(self, repo_dir: Path):
        self.repo_dir = repo_dir
        self.techniques: List[TechniqueEntry] = []
        self.commands: List[CommandEntry] = []

    def parse(self) -> Dict[str, Any]:
        logger.info("Parsing Atomic Red Team...")
        stats = {"files": 0, "techniques": 0, "commands": 0}

        atomics_dir = self.repo_dir / "atomics"
        if not atomics_dir.exists():
            logger.warning("atomics/ directory not found")
            return stats

        for tech_dir in sorted(atomics_dir.iterdir()):
            if not tech_dir.is_dir() or not tech_dir.name.startswith("T"):
                continue

            yaml_file = tech_dir / f"{tech_dir.name}.yaml"
            if not yaml_file.exists():
                yaml_file = tech_dir / f"{tech_dir.name}.yml"
            if not yaml_file.exists():
                continue

            try:
                import yaml
                data = yaml.safe_load(yaml_file.read_text(encoding="utf-8", errors="ignore"))
                if not isinstance(data, dict):
                    continue

                tech_id = data.get("attack_technique", tech_dir.name)
                tech_name = data.get("display_name", "")
                atomic_tests = data.get("atomic_tests", [])

                technique_tests = []
                for test in atomic_tests:
                    if not isinstance(test, dict):
                        continue
                    test_name = test.get("name", "")
                    test_desc = test.get("description", "")
                    platforms = test.get("supported_platforms", [])
                    executor = test.get("executor", {})
                    command = executor.get("command", "") if isinstance(executor, dict) else ""
                    cleanup = executor.get("cleanup_command", "") if isinstance(executor, dict) else ""

                    if command:
                        # Clean the command — remove #{...} input placeholders with defaults
                        input_args = test.get("input_arguments", {})
                        if isinstance(input_args, dict):
                            for arg_name, arg_data in input_args.items():
                                if isinstance(arg_data, dict):
                                    default = str(arg_data.get("default", arg_name))
                                    command = command.replace(f"#{{{arg_name}}}", default)

                        technique_tests.append({
                            "name": test_name,
                            "command": command[:1000],
                            "cleanup": cleanup[:500] if cleanup else "",
                            "platforms": platforms,
                            "description": test_desc[:500],
                        })

                        # Extract individual command lines
                        for line in command.split("\n"):
                            line = line.strip()
                            if line and not line.startswith("#") and len(line) > 5:
                                phase = classify_phase(test_desc + " " + test_name)
                                self.commands.append(CommandEntry(
                                    tool_name=line.split()[0] if line.split() else tech_id,
                                    command=line[:500],
                                    description=f"ATT&CK {tech_id}: {test_name}",
                                    phase=phase if phase != "GENERAL" else "EXPLOITATION",
                                    use_case=test_desc[:300],
                                    reasoning=f"MITRE ATT&CK technique {tech_id} ({tech_name}) — {test_name}",
                                    source=f"atomic_red_team/{tech_id}",
                                ))

                if technique_tests:
                    self.techniques.append(TechniqueEntry(
                        technique_id=tech_id,
                        technique_name=tech_name,
                        tactic=classify_phase(tech_name).lower(),
                        description=f"{tech_name}: {len(technique_tests)} atomic tests",
                        platforms=list(set(p for t in technique_tests for p in t.get("platforms", []))),
                        atomic_tests=technique_tests,
                        source="atomic_red_team",
                    ))
                    stats["techniques"] += 1

                stats["files"] += 1
            except ImportError:
                break  # No yaml
            except Exception:
                continue

        stats["commands"] = len(self.commands)
        logger.info(f"Atomic Red Team: {stats}")
        return stats


class MITRECTIParser:
    """Parse MITRE CTI — ATT&CK STIX data (techniques, tactics, procedures)."""

    def __init__(self, repo_dir: Path):
        self.repo_dir = repo_dir
        self.techniques: List[TechniqueEntry] = []

    def parse(self) -> Dict[str, Any]:
        logger.info("Parsing MITRE CTI (STIX)...")
        stats = {"files": 0, "techniques": 0}

        # MITRE CTI stores STIX JSON bundles
        for domain_dir in ["enterprise-attack", "mobile-attack", "ics-attack"]:
            attack_dir = self.repo_dir / domain_dir
            if not attack_dir.exists():
                continue

            # The main bundle file
            bundle_file = attack_dir / f"{domain_dir}.json"
            if not bundle_file.exists():
                # Try alternate path
                for jf in attack_dir.rglob("*.json"):
                    if jf.stat().st_size > 100_000:
                        bundle_file = jf
                        break

            if not bundle_file.exists():
                continue

            try:
                data = json.loads(bundle_file.read_text(encoding="utf-8", errors="ignore"))
                objects = data.get("objects", [])

                for obj in objects:
                    if obj.get("type") != "attack-pattern":
                        continue

                    ext_refs = obj.get("external_references", [])
                    tech_id = ""
                    for ref in ext_refs:
                        if ref.get("source_name") == "mitre-attack":
                            tech_id = ref.get("external_id", "")
                            break

                    if not tech_id:
                        continue

                    name = obj.get("name", "")
                    desc = obj.get("description", "")
                    platforms = obj.get("x_mitre_platforms", [])
                    kill_chain = obj.get("kill_chain_phases", [])
                    tactics = [kc.get("phase_name", "") for kc in kill_chain] if kill_chain else []

                    # Clean markdown from description
                    desc_clean = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', desc)  # Remove links
                    desc_clean = re.sub(r'<[^>]+>', '', desc_clean)  # Remove HTML

                    self.techniques.append(TechniqueEntry(
                        technique_id=tech_id,
                        technique_name=name,
                        tactic=", ".join(tactics) if tactics else "",
                        description=desc_clean[:2000],
                        platforms=platforms,
                        detection=obj.get("x_mitre_detection", "")[:1000],
                        reasoning=f"MITRE ATT&CK {tech_id}: {name} — used in {', '.join(tactics)}",
                        source=f"mitre_cti/{domain_dir}",
                    ))
                    stats["techniques"] += 1

                stats["files"] += 1
            except Exception as e:
                logger.debug(f"MITRE CTI parse error: {e}")

        logger.info(f"MITRE CTI: {stats}")
        return stats


class ExploitDBParser:
    """Parse ExploitDB — 45K+ exploits with CSV metadata."""

    def __init__(self, repo_dir: Path):
        self.repo_dir = repo_dir
        self.exploits: List[ExploitDBEntry] = []
        self.cves: List[CVEEntry] = []
        self.commands: List[CommandEntry] = []

    def parse(self) -> Dict[str, Any]:
        logger.info("Parsing ExploitDB...")
        stats = {"files": 0, "exploits": 0, "cves": 0, "commands": 0}

        # ExploitDB has files_exploits.csv with metadata
        csv_file = self.repo_dir / "files_exploits.csv"
        if not csv_file.exists():
            logger.warning("files_exploits.csv not found in ExploitDB repo")
            # Try parsing exploit files directly
            return self._parse_exploits_dir(stats)

        try:
            import csv
            with open(csv_file, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    edb_id = row.get("id", "")
                    desc = row.get("description", "")
                    date = row.get("date_published", "")
                    author = row.get("author", "")
                    platform = row.get("platform", "")
                    etype = row.get("type", "")
                    port_str = row.get("port", "0")
                    codes_str = row.get("codes", "")

                    try:
                        port = int(port_str) if port_str and port_str.isdigit() else 0
                    except ValueError:
                        port = 0

                    # Extract CVEs from codes field
                    cves_found = []
                    if codes_str:
                        cves_found = [c.strip() for c in codes_str.split(";") if c.strip().startswith("CVE-")]

                    self.exploits.append(ExploitDBEntry(
                        edb_id=edb_id,
                        description=desc[:500],
                        date_published=date,
                        author=author,
                        platform=platform,
                        exploit_type=etype,
                        port=port,
                        cve=cves_found[0] if cves_found else "",
                        codes=cves_found,
                    ))

                    # Create CVE entries for found CVEs
                    for cve_id in cves_found:
                        self.cves.append(CVEEntry(
                            cve_id=cve_id,
                            description=desc[:500],
                            affected_software=platform,
                            severity="unknown",
                            exploit_methodology=f"ExploitDB EDB-{edb_id}: {desc[:200]}",
                            source=f"exploitdb/EDB-{edb_id}",
                        ))

                    stats["exploits"] += 1
                    if stats["exploits"] % 10000 == 0:
                        logger.info(f"  ExploitDB progress: {stats['exploits']:,} exploits...")

        except Exception as e:
            logger.error(f"ExploitDB CSV parse error: {e}")

        stats["cves"] = len(self.cves)
        stats["commands"] = len(self.commands)
        logger.info(f"ExploitDB: {stats}")
        return stats

    def _parse_exploits_dir(self, stats):
        """Fallback: parse exploit files directly from exploits/ directory."""
        exploits_dir = self.repo_dir / "exploits"
        if not exploits_dir.exists():
            return stats
        count = 0
        for efile in exploits_dir.rglob("*"):
            if not efile.is_file() or efile.suffix not in (".py", ".rb", ".c", ".txt", ".pl", ".sh"):
                continue
            if count >= 50000:
                break
            try:
                text = efile.read_text(encoding="utf-8", errors="ignore")[:5000]
                cves = extract_cves(text)
                desc = efile.stem.replace("-", " ").replace("_", " ")
                self.exploits.append(ExploitDBEntry(
                    edb_id=efile.stem,
                    description=desc[:500],
                    platform=efile.parent.name,
                    cve=cves[0] if cves else "",
                    codes=cves,
                    source_file=str(efile.relative_to(self.repo_dir)),
                ))
                count += 1
                stats["exploits"] += 1
            except Exception:
                continue
        return stats


class InternalAllTheThingsParser:
    """Parse InternalAllTheThings — AD, pivoting, internal pentest methodology."""

    def __init__(self, repo_dir: Path):
        self.repo_dir = repo_dir
        self.commands: List[CommandEntry] = []
        self.ad_attacks: List[ADAttackEntry] = []
        self.methodology: List[MethodologyEntry] = []

    def parse(self) -> Dict[str, Any]:
        logger.info("Parsing InternalAllTheThings...")
        stats = {"files": 0, "commands": 0, "ad_attacks": 0}

        ad_categories = {
            "active-directory": "active-directory",
            "kerberos": "credential-access",
            "ldap": "enumeration",
            "ntlm": "credential-access",
            "privilege-escalation": "privilege-escalation",
            "persistence": "persistence",
            "lateral-movement": "lateral-movement",
            "pivoting": "lateral-movement",
            "port-forwarding": "lateral-movement",
            "tunneling": "lateral-movement",
        }

        for md_file in sorted(self.repo_dir.rglob("*.md")):
            if ".git" in str(md_file) or md_file.name.startswith("_"):
                continue
            try:
                text = md_file.read_text(encoding="utf-8", errors="ignore")
                if len(text) < 100:
                    continue

                rel_path = str(md_file.relative_to(self.repo_dir)).lower()
                sections = split_markdown_sections(text)
                cmds = extract_commands_from_markdown(text)

                # Determine AD category from path
                category = "general"
                for key, cat in ad_categories.items():
                    if key in rel_path:
                        category = cat
                        break

                for c in cmds[:30]:
                    phase = classify_phase(c.get("context", "") + " " + c.get("heading", ""))
                    if phase == "GENERAL":
                        phase = "EXPLOITATION"
                    self.commands.append(CommandEntry(
                        tool_name=c["command"].split()[0] if c["command"].split() else "",
                        command=c["command"],
                        description=c.get("heading", md_file.stem),
                        phase=phase,
                        use_case=c.get("context", ""),
                        reasoning=f"Internal pentest technique: {category}",
                        source=f"internal_all_the_things/{rel_path}",
                    ))

                # Build AD attack entries from substantial sections
                for heading, content, level in sections:
                    if len(content) > 200:
                        section_cmds = extract_commands_from_markdown(content)
                        if section_cmds:
                            tools = list(set(c["command"].split()[0] for c in section_cmds if c["command"].split()))[:10]
                            self.ad_attacks.append(ADAttackEntry(
                                attack_name=heading,
                                category=category,
                                description=content[:1000],
                                commands=section_cmds[:10],
                                tools_used=tools,
                                reasoning=content[:500],
                                source=f"internal_all_the_things/{rel_path}",
                            ))
                            stats["ad_attacks"] += 1

                stats["files"] += 1
            except Exception:
                continue

        stats["commands"] = len(self.commands)
        logger.info(f"InternalAllTheThings: {stats}")
        return stats


class HackTricksCloudParser:
    """Parse HackTricks Cloud — AWS/Azure/GCP attack methodology."""

    def __init__(self, repo_dir: Path):
        self.repo_dir = repo_dir
        self.commands: List[CommandEntry] = []
        self.cloud_attacks: List[CloudAttackEntry] = []

    def parse(self) -> Dict[str, Any]:
        logger.info("Parsing HackTricks Cloud...")
        stats = {"files": 0, "commands": 0, "cloud_attacks": 0}

        cloud_providers = {"aws": "aws", "azure": "azure", "gcp": "gcp", "google": "gcp"}

        for base in [self.repo_dir, self.repo_dir / "src"]:
            for md_file in sorted(base.rglob("*.md")):
                if ".git" in str(md_file) or md_file.name in ("SUMMARY.md", "README.md"):
                    continue
                try:
                    text = md_file.read_text(encoding="utf-8", errors="ignore")
                    if len(text) < 100:
                        continue

                    rel_path = str(md_file.relative_to(self.repo_dir)).lower()

                    # Determine cloud provider
                    provider = ""
                    for key, prov in cloud_providers.items():
                        if key in rel_path:
                            provider = prov
                            break

                    cmds = extract_commands_from_markdown(text)
                    sections = split_markdown_sections(text)

                    for c in cmds[:20]:
                        self.commands.append(CommandEntry(
                            tool_name=c["command"].split()[0] if c["command"].split() else "",
                            command=c["command"],
                            description=c.get("heading", md_file.stem),
                            phase=classify_phase(c.get("context", "")),
                            use_case=f"Cloud attack ({provider}): {c.get('heading', '')}",
                            source=f"hacktricks_cloud/{rel_path}",
                        ))

                    for heading, content, level in sections:
                        if len(content) > 300:
                            section_cmds = extract_commands_from_markdown(content)
                            if section_cmds or any(w in content.lower() for w in ["exploit", "attack", "abuse", "escalat"]):
                                self.cloud_attacks.append(CloudAttackEntry(
                                    attack_name=heading,
                                    cloud_provider=provider or "multi",
                                    category=classify_phase(heading).lower(),
                                    description=content[:1500],
                                    commands=section_cmds[:10],
                                    reasoning=content[:500],
                                    source=f"hacktricks_cloud/{rel_path}",
                                ))
                                stats["cloud_attacks"] += 1

                    stats["files"] += 1
                except Exception:
                    continue

        stats["commands"] = len(self.commands)
        logger.info(f"HackTricks Cloud: {stats}")
        return stats


class PEASSParser:
    """Parse PEASS-ng — LinPEAS/WinPEAS privilege escalation checks."""

    def __init__(self, repo_dir: Path):
        self.repo_dir = repo_dir
        self.commands: List[CommandEntry] = []
        self.privesc_checks: List[Dict[str, Any]] = []

    def parse(self) -> Dict[str, Any]:
        logger.info("Parsing PEASS-ng...")
        stats = {"files": 0, "commands": 0, "checks": 0}

        # Parse linPEAS — extract check descriptions and commands from shell scripts
        for script_dir in [self.repo_dir / "linPEAS", self.repo_dir / "winPEAS"]:
            if not script_dir.exists():
                continue
            platform = "linux" if "lin" in script_dir.name.lower() else "windows"

            for efile in sorted(script_dir.rglob("*")):
                if not efile.is_file():
                    continue
                if efile.suffix not in (".sh", ".bat", ".ps1", ".py", ".rb", ".md"):
                    continue
                try:
                    text = efile.read_text(encoding="utf-8", errors="ignore")
                    if len(text) < 50:
                        continue

                    # Extract check descriptions from comments
                    check_pattern = re.compile(r'#\s*[-=]+\s*(.+?)[-=]*\s*$', re.MULTILINE)
                    for match in check_pattern.finditer(text):
                        check_name = match.group(1).strip()
                        if len(check_name) > 10 and not check_name.startswith("#"):
                            # Get the command block following this check
                            block_start = match.end()
                            block_end = min(block_start + 2000, len(text))
                            block = text[block_start:block_end]

                            # Extract commands from block
                            cmd_lines = []
                            for line in block.split("\n")[:20]:
                                line = line.strip()
                                if line and not line.startswith("#") and not line.startswith("echo") and len(line) > 5:
                                    if any(c in line for c in ["$(", "|", "/", "grep", "find", "cat", "ls"]):
                                        cmd_lines.append(line[:300])

                            if cmd_lines:
                                self.privesc_checks.append({
                                    "check_name": check_name,
                                    "platform": platform,
                                    "commands": cmd_lines[:5],
                                    "source": f"peass_ng/{efile.name}",
                                })
                                stats["checks"] += 1

                                for cmd in cmd_lines[:3]:
                                    self.commands.append(CommandEntry(
                                        tool_name=cmd.split()[0] if cmd.split() else "check",
                                        command=cmd,
                                        description=f"PEASS {platform} check: {check_name}",
                                        phase="PRIVILEGE_ESCALATION",
                                        use_case=f"Privesc detection: {check_name}",
                                        reasoning=f"Automated privilege escalation check from {'LinPEAS' if platform == 'linux' else 'WinPEAS'}",
                                        source=f"peass_ng/{efile.name}",
                                    ))

                    stats["files"] += 1
                except Exception:
                    continue

        stats["commands"] = len(self.commands)
        logger.info(f"PEASS-ng: {stats}")
        return stats


class ImpacketParser:
    """Parse Impacket — Python AD exploitation tools."""

    def __init__(self, repo_dir: Path):
        self.repo_dir = repo_dir
        self.commands: List[CommandEntry] = []
        self.ad_attacks: List[ADAttackEntry] = []

    def parse(self) -> Dict[str, Any]:
        logger.info("Parsing Impacket...")
        stats = {"files": 0, "commands": 0, "tools": 0}

        # Parse impacket tool scripts (examples/ directory)
        examples_dir = self.repo_dir / "examples"
        if not examples_dir.exists():
            examples_dir = self.repo_dir / "impacket" / "examples"

        if examples_dir.exists():
            for py_file in sorted(examples_dir.glob("*.py")):
                try:
                    text = py_file.read_text(encoding="utf-8", errors="ignore")
                    # Extract docstring/description from header
                    desc = ""
                    desc_match = re.search(r'"""(.*?)"""', text[:3000], re.DOTALL)
                    if not desc_match:
                        desc_match = re.search(r"'''(.*?)'''", text[:3000], re.DOTALL)
                    if desc_match:
                        desc = desc_match.group(1).strip()[:500]

                    tool_name = py_file.stem
                    # Generate typical usage command
                    usage_cmd = f"python {tool_name}.py -target-ip TARGET DOMAIN/USER:PASS"
                    usage_match = re.search(r'usage.*?:\s*(.+)', text[:5000], re.IGNORECASE)
                    if usage_match:
                        usage_cmd = usage_match.group(1).strip()[:300]

                    self.commands.append(CommandEntry(
                        tool_name=tool_name,
                        command=usage_cmd,
                        description=f"Impacket {tool_name}: {desc[:200]}",
                        phase="EXPLOITATION",
                        use_case=f"AD exploitation tool: {tool_name}",
                        reasoning=f"Impacket tool {tool_name} — {desc[:200]}",
                        source=f"impacket/{tool_name}",
                    ))

                    self.ad_attacks.append(ADAttackEntry(
                        attack_name=f"impacket-{tool_name}",
                        category="exploitation",
                        description=desc[:500],
                        commands=[{"command": usage_cmd, "description": desc[:200]}],
                        tools_used=[f"impacket/{tool_name}"],
                        source=f"impacket/{tool_name}",
                    ))

                    stats["tools"] += 1
                    stats["files"] += 1
                except Exception:
                    continue

        stats["commands"] = len(self.commands)
        logger.info(f"Impacket: {stats}")
        return stats


class NishangParser:
    """Parse Nishang — PowerShell offensive security scripts."""

    def __init__(self, repo_dir: Path):
        self.repo_dir = repo_dir
        self.commands: List[CommandEntry] = []

    def parse(self) -> Dict[str, Any]:
        logger.info("Parsing Nishang...")
        stats = {"files": 0, "commands": 0, "scripts": 0}

        for ps1_file in sorted(self.repo_dir.rglob("*.ps1")):
            if ".git" in str(ps1_file):
                continue
            try:
                text = ps1_file.read_text(encoding="utf-8", errors="ignore")
                if len(text) < 50:
                    continue

                script_name = ps1_file.stem
                rel_path = str(ps1_file.relative_to(self.repo_dir))

                # Extract synopsis/description from PowerShell help comments
                desc = ""
                synopsis_match = re.search(r'\.SYNOPSIS\s*\n(.+?)(?:\n\.|$)', text[:3000], re.DOTALL)
                if synopsis_match:
                    desc = synopsis_match.group(1).strip()[:500]

                # Extract example usage
                examples = []
                for ex_match in re.finditer(r'\.EXAMPLE\s*\n(.+?)(?:\n\.|$)', text[:5000], re.DOTALL):
                    ex_text = ex_match.group(1).strip()
                    if ex_text:
                        examples.append(ex_text[:300])

                # Determine category from directory
                parent = ps1_file.parent.name.lower()
                phase = "EXPLOITATION"
                if "gather" in parent or "recon" in parent:
                    phase = "ENUMERATION"
                elif "escalat" in parent or "privesc" in parent:
                    phase = "PRIVILEGE_ESCALATION"
                elif "backdoor" in parent or "persist" in parent:
                    phase = "POST_EXPLOITATION"
                elif "shell" in parent or "reverse" in parent:
                    phase = "EXPLOITATION"

                usage = f"Import-Module .\\{script_name}.ps1; {script_name}"
                if examples:
                    usage = examples[0]

                self.commands.append(CommandEntry(
                    tool_name=script_name,
                    command=usage,
                    description=f"Nishang {script_name}: {desc[:200]}",
                    phase=phase,
                    use_case=desc[:300],
                    reasoning=f"PowerShell offensive tool from Nishang — {parent}/{script_name}",
                    source=f"nishang/{rel_path}",
                ))
                stats["scripts"] += 1

                # Add example commands
                for ex in examples[1:3]:
                    self.commands.append(CommandEntry(
                        tool_name=script_name,
                        command=ex,
                        description=f"Nishang {script_name} example",
                        phase=phase,
                        source=f"nishang/{rel_path}",
                    ))

                stats["files"] += 1
            except Exception:
                continue

        stats["commands"] = len(self.commands)
        logger.info(f"Nishang: {stats}")
        return stats


class CTFWriteupsParser:
    """Parse CTF writeups — challenge solutions with reasoning."""

    def __init__(self, repo_dir: Path):
        self.repo_dir = repo_dir
        self.writeups: List[CTFWriteupEntry] = []
        self.commands: List[CommandEntry] = []

    def parse(self) -> Dict[str, Any]:
        logger.info("Parsing CTF writeups...")
        stats = {"files": 0, "writeups": 0, "commands": 0}

        ctf_categories = {"web", "pwn", "crypto", "forensics", "misc", "rev", "reversing",
                          "binary", "network", "stego", "steganography", "osint"}

        for md_file in sorted(self.repo_dir.rglob("*.md")):
            if ".git" in str(md_file):
                continue
            try:
                text = md_file.read_text(encoding="utf-8", errors="ignore")
                if len(text) < 200:
                    continue

                rel_path = str(md_file.relative_to(self.repo_dir)).lower()

                # Determine category from path
                category = "misc"
                for cat in ctf_categories:
                    if cat in rel_path:
                        category = cat
                        break

                challenge_name = md_file.stem.replace("-", " ").replace("_", " ")
                sections = split_markdown_sections(text)
                cmds = extract_commands_from_markdown(text)
                tools = list(set(c["command"].split()[0] for c in cmds if c["command"].split()))[:10]

                # Extract solution steps from sections
                solution_steps = []
                for heading, content, level in sections:
                    if any(w in heading.lower() for w in ["solution", "solve", "step", "approach", "exploit", "payload"]):
                        solution_steps.append(f"{heading}: {content[:500]}")

                if cmds or solution_steps:
                    self.writeups.append(CTFWriteupEntry(
                        challenge_name=challenge_name,
                        category=category,
                        description=text[:500],
                        solution_steps=solution_steps[:10],
                        commands=[c for c in cmds[:15]],
                        tools_used=tools,
                        reasoning="\n".join(solution_steps[:3])[:1000] if solution_steps else "",
                        source=f"ctf_writeups/{rel_path}",
                    ))
                    stats["writeups"] += 1

                for c in cmds[:10]:
                    self.commands.append(CommandEntry(
                        tool_name=c["command"].split()[0] if c["command"].split() else "",
                        command=c["command"],
                        description=f"CTF {category}: {c.get('heading', challenge_name)}",
                        phase=classify_phase(c.get("context", "")),
                        source=f"ctf_writeups/{rel_path}",
                    ))

                stats["files"] += 1
            except Exception:
                continue

        stats["commands"] = len(self.commands)
        logger.info(f"CTF Writeups: {stats}")
        return stats


class SecListsParser:
    """Parse SecLists — Extract wordlist metadata (not the full wordlists)."""

    def __init__(self, repo_dir: Path):
        self.repo_dir = repo_dir
        self.wordlists_meta: List[Dict[str, Any]] = []
        self.commands: List[CommandEntry] = []

    def parse(self) -> Dict[str, Any]:
        logger.info("Parsing SecLists (metadata extraction)...")
        stats = {"files": 0, "wordlists": 0, "commands": 0}

        # Index all wordlist files with metadata (don't load content — too large)
        categories = {
            "Passwords": "credentials",
            "Usernames": "enumeration",
            "Discovery": "enumeration",
            "Fuzzing": "exploitation",
            "Web-Shells": "exploitation",
            "Pattern-Matching": "analysis",
        }

        for category_dir, cat_type in categories.items():
            cat_path = self.repo_dir / category_dir
            if not cat_path.exists():
                continue

            for wl_file in sorted(cat_path.rglob("*")):
                if not wl_file.is_file():
                    continue
                if ".git" in str(wl_file) or wl_file.name.startswith("."):
                    continue

                try:
                    size = wl_file.stat().st_size
                    # Count lines for small files, estimate for large
                    line_count = 0
                    if size < 10_000_000:  # < 10MB
                        try:
                            with open(wl_file, "r", encoding="utf-8", errors="ignore") as f:
                                line_count = sum(1 for _ in f)
                        except Exception:
                            line_count = size // 20  # Estimate
                    else:
                        line_count = size // 20

                    rel_path = str(wl_file.relative_to(self.repo_dir))

                    self.wordlists_meta.append({
                        "name": wl_file.name,
                        "path": rel_path,
                        "category": category_dir,
                        "type": cat_type,
                        "size_bytes": size,
                        "line_count": line_count,
                        "description": f"{category_dir}/{wl_file.name} — {line_count:,} entries",
                    })
                    stats["wordlists"] += 1

                    # Generate usage commands for notable wordlists
                    if line_count > 100:
                        if "password" in rel_path.lower() or category_dir == "Passwords":
                            self.commands.append(CommandEntry(
                                tool_name="hydra",
                                command=f"hydra -L users.txt -P {rel_path} TARGET ssh",
                                description=f"Brute-force with {wl_file.name} ({line_count:,} passwords)",
                                phase="EXPLOITATION",
                                use_case=f"Password brute-force using {wl_file.name}",
                                source="seclists",
                            ))
                        elif "directory" in rel_path.lower() or "discovery" in rel_path.lower():
                            self.commands.append(CommandEntry(
                                tool_name="gobuster",
                                command=f"gobuster dir -u http://TARGET -w {rel_path}",
                                description=f"Directory brute-force with {wl_file.name} ({line_count:,} entries)",
                                phase="ENUMERATION",
                                source="seclists",
                            ))
                        elif "fuzz" in rel_path.lower():
                            self.commands.append(CommandEntry(
                                tool_name="ffuf",
                                command=f"ffuf -u http://TARGET/FUZZ -w {rel_path}",
                                description=f"Fuzzing with {wl_file.name} ({line_count:,} patterns)",
                                phase="EXPLOITATION",
                                source="seclists",
                            ))

                    stats["files"] += 1
                except Exception:
                    continue

        stats["commands"] = len(self.commands)
        logger.info(f"SecLists: {stats}")
        return stats


class MetasploitParser:
    """Parse Metasploit Framework — extract module metadata from Ruby files."""

    def __init__(self, repo_dir: Path):
        self.repo_dir = repo_dir
        self.commands: List[CommandEntry] = []
        self.cves: List[CVEEntry] = []

    def parse(self) -> Dict[str, Any]:
        logger.info("Parsing Metasploit Framework modules...")
        stats = {"files": 0, "modules": 0, "cves": 0, "commands": 0}

        modules_dir = self.repo_dir / "modules"
        if not modules_dir.exists():
            logger.warning("modules/ directory not found in Metasploit")
            return stats

        # Parse module Ruby files for metadata
        for rb_file in sorted(modules_dir.rglob("*.rb")):
            if ".git" in str(rb_file):
                continue
            try:
                text = rb_file.read_text(encoding="utf-8", errors="ignore")
                if len(text) < 100:
                    continue

                rel_path = str(rb_file.relative_to(modules_dir))
                module_path = rel_path.replace(".rb", "").replace("/", "/")

                # Extract metadata from module info hash
                name = ""
                desc = ""
                author = ""
                cves = []
                platform = ""
                port = 0

                name_match = re.search(r"['\"]Name['\"]\s*=>\s*['\"](.+?)['\"]", text)
                if name_match:
                    name = name_match.group(1)

                desc_match = re.search(r"['\"]Description['\"]\s*=>\s*['\"](.+?)['\"]", text, re.DOTALL)
                if not desc_match:
                    desc_match = re.search(r"['\"]Description['\"]\s*=>\s*%q\{(.+?)\}", text, re.DOTALL)
                if desc_match:
                    desc = desc_match.group(1).strip()[:500]

                cve_matches = CVE_PATTERN.findall(text)
                cves = list(set(cve_matches))

                port_match = re.search(r"['\"]RPORT['\"]\s*=>\s*(\d+)", text)
                if port_match:
                    port = int(port_match.group(1))

                platform_match = re.search(r"['\"]Platform['\"]\s*=>\s*['\"](.+?)['\"]", text)
                if platform_match:
                    platform = platform_match.group(1)

                if name or desc:
                    # Generate MSF usage command
                    msf_cmd = f"use {module_path}"
                    phase = "EXPLOITATION"
                    if "auxiliary" in module_path:
                        phase = "ENUMERATION"
                    elif "post" in module_path:
                        phase = "POST_EXPLOITATION"

                    self.commands.append(CommandEntry(
                        tool_name="msfconsole",
                        command=msf_cmd,
                        description=f"MSF: {name}" if name else f"MSF module: {module_path}",
                        phase=phase,
                        use_case=desc[:300],
                        reasoning=f"Metasploit module targeting {'port ' + str(port) if port else platform or 'multi'}",
                        related_tools=["metasploit"],
                        source=f"metasploit/{module_path}",
                    ))
                    stats["modules"] += 1

                    for cve_id in cves:
                        self.cves.append(CVEEntry(
                            cve_id=cve_id,
                            description=f"{name}: {desc[:200]}",
                            affected_software=platform,
                            metasploit_module=module_path,
                            source=f"metasploit/{module_path}",
                        ))

                    stats["files"] += 1
                    if stats["modules"] % 1000 == 0:
                        logger.info(f"  Metasploit progress: {stats['modules']:,} modules...")

            except Exception:
                continue

        stats["cves"] = len(self.cves)
        stats["commands"] = len(self.commands)
        logger.info(f"Metasploit: {stats}")
        return stats


# ─── Deduplication & Quality ────────────────────────────────────────────────

def deduplicate_commands(commands: List[Dict[str, Any]], key: str = "command") -> List[Dict[str, Any]]:
    """Remove duplicate commands, keeping the one with most context."""
    seen: Dict[str, Dict[str, Any]] = {}
    for cmd in commands:
        cmd_text = cmd.get(key, "").strip()
        if not cmd_text:
            continue
        # Normalize for dedup
        normalized = re.sub(r'\s+', ' ', cmd_text.lower())
        fingerprint = hashlib.md5(normalized.encode()).hexdigest()[:12]
        if fingerprint not in seen:
            seen[fingerprint] = cmd
        else:
            # Keep the one with more context
            existing = seen[fingerprint]
            if len(cmd.get("context", "")) > len(existing.get("context", "")):
                seen[fingerprint] = cmd
    return list(seen.values())


def quality_filter(entries: List[Dict[str, Any]], min_fields: int = 2) -> List[Dict[str, Any]]:
    """Filter out low-quality entries with too few populated fields."""
    filtered = []
    for entry in entries:
        populated = sum(
            1 for v in entry.values()
            if v and v != [] and v != {} and v != "" and v != 0
        )
        if populated >= min_fields:
            filtered.append(entry)
    return filtered


# ─── Main Pipeline ──────────────────────────────────────────────────────────

def clone_repo(name: str, config: Dict, force: bool = False) -> Optional[Path]:
    """Clone a GitHub repository. Returns path or None on failure."""
    target = CLONE_DIR / name
    if target.exists() and not force:
        logger.info(f"[{name}] Already cloned at {target}")
        return target

    logger.info(f"[{name}] Cloning {config['url']}...")
    os.makedirs(CLONE_DIR, exist_ok=True)

    cmd = ["git", "clone"]
    if config.get("shallow", True):
        cmd.extend(["--depth", "1"])
    cmd.extend([config["url"], str(target)])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            logger.info(f"[{name}] Clone complete: {target}")
            return target
        else:
            logger.error(f"[{name}] Clone failed: {result.stderr[:200]}")
            return None
    except subprocess.TimeoutExpired:
        logger.error(f"[{name}] Clone timed out (600s)")
        return None
    except Exception as e:
        logger.error(f"[{name}] Clone error: {e}")
        return None


def run_ingestion(repos: Optional[List[str]] = None, quick: bool = False):
    """Run the full ingestion pipeline."""
    start = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Select repos
    repo_list = repos or list(REPOS.keys())
    logger.info(f"Starting knowledge ingestion for: {repo_list}")

    # Aggregated knowledge — original
    all_services: List[Dict] = []
    all_cves: List[Dict] = []
    all_payloads: List[Dict] = []
    all_methodology: List[Dict] = []
    all_commands: List[Dict] = []
    all_kill_chains: List[Dict] = []
    all_cheatsheets: List[Dict] = []
    all_privesc_linux: List[Dict] = []
    all_privesc_windows: List[Dict] = []
    # Phase 9.2 — expanded knowledge types
    all_techniques: List[Dict] = []       # MITRE ATT&CK + Atomic Red Team
    all_binaries: List[Dict] = []         # GTFOBins + LOLBAS
    all_exploitdb: List[Dict] = []        # ExploitDB entries
    all_ad_attacks: List[Dict] = []       # AD attack chains (WADComs, Internal, Impacket)
    all_cloud_attacks: List[Dict] = []    # Cloud attack chains (HackTricks Cloud)
    all_ctf_writeups: List[Dict] = []     # CTF writeups with reasoning
    all_wordlists_meta: List[Dict] = []   # SecLists wordlist metadata
    all_privesc_checks: List[Dict] = []   # PEASS privesc check enumeration

    manifest = {
        "version": "1.0",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "repos_processed": [],
        "stats": {},
    }

    for repo_name in repo_list:
        config = REPOS.get(repo_name)
        if not config:
            logger.warning(f"Unknown repo: {repo_name}")
            continue

        # Clone
        if not quick:
            repo_dir = clone_repo(repo_name, config)
        else:
            repo_dir = CLONE_DIR / repo_name
            if not repo_dir.exists():
                logger.warning(f"[{repo_name}] Not found (use full mode to clone)")
                continue

        if not repo_dir or not repo_dir.exists():
            continue

        # Parse based on repo type
        try:
            if repo_name == "hacktricks":
                parser = HackTricksParser(repo_dir)
                stats = parser.parse()
                all_services.extend(asdict(s) for s in parser.services)
                all_kill_chains.extend(asdict(k) for k in parser.kill_chains)
                all_commands.extend(asdict(c) for c in parser.commands)
                all_methodology.extend(asdict(m) for m in parser.methodology)
                all_privesc_linux.extend(parser.privesc_linux)
                all_privesc_windows.extend(parser.privesc_windows)

            elif repo_name == "payloads":
                parser = PayloadsParser(repo_dir)
                stats = parser.parse()
                all_payloads.extend(asdict(p) for p in parser.payloads)
                all_commands.extend(asdict(c) for c in parser.commands)
                all_methodology.extend(asdict(m) for m in parser.methodology)

            elif repo_name == "oscp":
                parser = OSCPParser(repo_dir)
                stats = parser.parse()
                all_commands.extend(asdict(c) for c in parser.commands)
                all_cheatsheets.extend(asdict(s) for s in parser.cheatsheets)
                all_methodology.extend(asdict(m) for m in parser.methodology)

            elif repo_name == "h4cker":
                parser = H4ckerParser(repo_dir)
                stats = parser.parse()
                all_commands.extend(asdict(c) for c in parser.commands)
                all_cheatsheets.extend(asdict(s) for s in parser.cheatsheets)
                all_methodology.extend(asdict(m) for m in parser.methodology)

            elif repo_name == "vulhub":
                parser = VulhubParser(repo_dir)
                stats = parser.parse()
                all_cves.extend(asdict(c) for c in parser.cves)
                all_kill_chains.extend(asdict(k) for k in parser.kill_chains)

            elif repo_name == "pentest_wiki":
                parser = PentestWikiParser(repo_dir)
                stats = parser.parse()
                all_commands.extend(asdict(c) for c in parser.commands)
                all_methodology.extend(asdict(m) for m in parser.methodology)
                all_kill_chains.extend(asdict(k) for k in parser.kill_chains)
                all_services.extend(asdict(s) for s in parser.services)

            # ─── Phase 9.2: New Repo Parsers ───

            elif repo_name == "gtfobins":
                parser = GTFOBinsParser(repo_dir)
                stats = parser.parse()
                all_binaries.extend(asdict(b) for b in parser.binaries)
                all_commands.extend(asdict(c) for c in parser.commands)

            elif repo_name == "lolbas":
                parser = LOLBASParser(repo_dir)
                stats = parser.parse()
                all_binaries.extend(asdict(b) for b in parser.binaries)
                all_commands.extend(asdict(c) for c in parser.commands)

            elif repo_name == "wadcoms":
                parser = WADComsParser(repo_dir)
                stats = parser.parse()
                all_commands.extend(asdict(c) for c in parser.commands)
                all_ad_attacks.extend(asdict(a) for a in parser.ad_attacks)

            elif repo_name == "atomic_red_team":
                parser = AtomicRedTeamParser(repo_dir)
                stats = parser.parse()
                all_techniques.extend(asdict(t) for t in parser.techniques)
                all_commands.extend(asdict(c) for c in parser.commands)

            elif repo_name == "mitre_cti":
                parser = MITRECTIParser(repo_dir)
                stats = parser.parse()
                all_techniques.extend(asdict(t) for t in parser.techniques)

            elif repo_name == "exploitdb":
                parser = ExploitDBParser(repo_dir)
                stats = parser.parse()
                all_exploitdb.extend(asdict(e) for e in parser.exploits)
                all_cves.extend(asdict(c) for c in parser.cves)
                all_commands.extend(asdict(c) for c in parser.commands)

            elif repo_name == "internal_all_the_things":
                parser = InternalAllTheThingsParser(repo_dir)
                stats = parser.parse()
                all_commands.extend(asdict(c) for c in parser.commands)
                all_ad_attacks.extend(asdict(a) for a in parser.ad_attacks)
                all_methodology.extend(asdict(m) for m in parser.methodology)

            elif repo_name == "hacktricks_cloud":
                parser = HackTricksCloudParser(repo_dir)
                stats = parser.parse()
                all_commands.extend(asdict(c) for c in parser.commands)
                all_cloud_attacks.extend(asdict(a) for a in parser.cloud_attacks)

            elif repo_name == "peass_ng":
                parser = PEASSParser(repo_dir)
                stats = parser.parse()
                all_commands.extend(asdict(c) for c in parser.commands)
                all_privesc_checks.extend(parser.privesc_checks)

            elif repo_name == "impacket":
                parser = ImpacketParser(repo_dir)
                stats = parser.parse()
                all_commands.extend(asdict(c) for c in parser.commands)
                all_ad_attacks.extend(asdict(a) for a in parser.ad_attacks)

            elif repo_name == "nishang":
                parser = NishangParser(repo_dir)
                stats = parser.parse()
                all_commands.extend(asdict(c) for c in parser.commands)

            elif repo_name == "ctf_writeups":
                parser = CTFWriteupsParser(repo_dir)
                stats = parser.parse()
                all_commands.extend(asdict(c) for c in parser.commands)
                all_ctf_writeups.extend(asdict(w) for w in parser.writeups)

            elif repo_name == "seclists":
                parser = SecListsParser(repo_dir)
                stats = parser.parse()
                all_commands.extend(asdict(c) for c in parser.commands)
                all_wordlists_meta.extend(parser.wordlists_meta)

            elif repo_name == "metasploit":
                parser = MetasploitParser(repo_dir)
                stats = parser.parse()
                all_commands.extend(asdict(c) for c in parser.commands)
                all_cves.extend(asdict(c) for c in parser.cves)

            else:
                logger.warning(f"[{repo_name}] No parser implemented, skipping")
                stats = {"skipped": True}

            manifest["repos_processed"].append(repo_name)
            manifest["stats"][repo_name] = stats

        except Exception as e:
            logger.error(f"[{repo_name}] Parse failed: {e}")
            manifest["stats"][repo_name] = {"error": str(e)}

    # ─── Deduplication & Quality Filtering ───
    logger.info("Running deduplication and quality filtering...")

    all_commands = deduplicate_commands(all_commands, key="command")
    all_commands = quality_filter(all_commands, min_fields=3)

    # Build ports.json from services
    ports_data = {}
    for svc in all_services:
        port = svc.get("port", 0)
        if port > 0:
            if port not in ports_data:
                ports_data[port] = svc
            else:
                # Merge: keep the entry with more data
                existing = ports_data[port]
                for key in ["enumeration_commands", "exploitation_commands", "default_credentials"]:
                    existing_list = existing.get(key, [])
                    new_list = svc.get(key, [])
                    merged = existing_list + [c for c in new_list if c not in existing_list]
                    existing[key] = merged[:50]
                if len(svc.get("methodology", "")) > len(existing.get("methodology", "")):
                    existing["methodology"] = svc["methodology"]
                if len(svc.get("reasoning", "")) > len(existing.get("reasoning", "")):
                    existing["reasoning"] = svc["reasoning"]

    # ─── Write Output Files ───
    logger.info("Writing knowledge base files...")

    output_files = {
        "services.json": all_services,
        "ports.json": ports_data,
        "cves.json": all_cves,
        "payloads.json": all_payloads,
        "methodology.json": all_methodology,
        "commands.json": all_commands,
        "kill_chains.json": all_kill_chains,
        "cheatsheets.json": all_cheatsheets,
        "privesc_linux.json": all_privesc_linux,
        "privesc_windows.json": all_privesc_windows,
        # Phase 9.2 — expanded
        "techniques.json": all_techniques,
        "binaries.json": all_binaries,
        "exploitdb.json": all_exploitdb,
        "ad_attacks.json": all_ad_attacks,
        "cloud_attacks.json": all_cloud_attacks,
        "ctf_writeups.json": all_ctf_writeups,
        "wordlists_meta.json": all_wordlists_meta,
        "privesc_checks.json": all_privesc_checks,
    }

    total_size = 0
    for fname, data in output_files.items():
        fpath = OUTPUT_DIR / fname
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1, ensure_ascii=False, default=str)
        size = fpath.stat().st_size
        total_size += size
        count = len(data) if isinstance(data, list) else len(data.keys())
        logger.info(f"  {fname}: {count:,} entries, {size / 1024 / 1024:.1f} MB")
        manifest["stats"][fname] = {"entries": count, "size_bytes": size}

    # Write manifest
    manifest["total_size_bytes"] = total_size
    manifest["total_size_mb"] = round(total_size / 1024 / 1024, 1)
    manifest["duration_seconds"] = round(time.time() - start, 1)

    with open(OUTPUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        f"\n{'=' * 60}\n"
        f"Knowledge ingestion complete!\n"
        f"  Repos processed: {len(manifest['repos_processed'])}\n"
        f"  Services:        {len(all_services):,}\n"
        f"  CVEs:            {len(all_cves):,}\n"
        f"  Payloads:        {len(all_payloads):,}\n"
        f"  Commands:        {len(all_commands):,}\n"
        f"  Kill Chains:     {len(all_kill_chains):,}\n"
        f"  Methodology:     {len(all_methodology):,}\n"
        f"  Cheatsheets:     {len(all_cheatsheets):,}\n"
        f"  Privesc Linux:   {len(all_privesc_linux):,}\n"
        f"  Privesc Windows: {len(all_privesc_windows):,}\n"
        f"  Techniques:      {len(all_techniques):,}\n"
        f"  Binaries:        {len(all_binaries):,}\n"
        f"  ExploitDB:       {len(all_exploitdb):,}\n"
        f"  AD Attacks:      {len(all_ad_attacks):,}\n"
        f"  Cloud Attacks:   {len(all_cloud_attacks):,}\n"
        f"  CTF Writeups:    {len(all_ctf_writeups):,}\n"
        f"  Wordlists Meta:  {len(all_wordlists_meta):,}\n"
        f"  Privesc Checks:  {len(all_privesc_checks):,}\n"
        f"  Total size:      {manifest['total_size_mb']:.1f} MB\n"
        f"  Duration:        {manifest['duration_seconds']:.0f}s\n"
        f"{'=' * 60}"
    )
    return manifest


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    quick = "--quick" in sys.argv
    repo_filter = None
    for arg in sys.argv[1:]:
        if arg.startswith("--repo="):
            repo_filter = [arg.split("=")[1]]
        elif arg == "--repo" and sys.argv.index(arg) + 1 < len(sys.argv):
            repo_filter = [sys.argv[sys.argv.index(arg) + 1]]

    manifest = run_ingestion(repos=repo_filter, quick=quick)
    print(f"\nManifest: {OUTPUT_DIR / 'manifest.json'}")
