#!/usr/bin/env python3
"""
data/knowledge_ingestion.py — ARIASKA Knowledge Mega-Ingestion Pipeline v1.0

Downloads, parses, and structures cybersecurity knowledge from top GitHub repositories
into a unified JSON knowledge base that Ariaska's agents can query at runtime.

Sources (prioritized):
    1. HackTricks-wiki/hacktricks        — Per-port/service pentesting guides (LARGEST)
    2. swisskyrepo/PayloadsAllTheThings   — Vuln-specific payloads & methodology
    3. 0xsyr0/OSCP                        — Command-first cheat sheet (6000+ lines)
    4. The-Art-of-Hacking/h4cker          — Methodology + metasploit cheat sheets
    5. vulhub/vulhub                      — CVE reproduction environments
    6. nixawk/pentest-wiki                — Kill-chain organized pentest wiki

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

    # Aggregated knowledge
    all_services: List[Dict] = []
    all_cves: List[Dict] = []
    all_payloads: List[Dict] = []
    all_methodology: List[Dict] = []
    all_commands: List[Dict] = []
    all_kill_chains: List[Dict] = []
    all_cheatsheets: List[Dict] = []
    all_privesc_linux: List[Dict] = []
    all_privesc_windows: List[Dict] = []

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
