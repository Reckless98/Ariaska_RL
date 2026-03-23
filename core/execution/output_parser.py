# core/execution/output_parser.py — Ariaska RL Real Output Parser
# Parses real tool output (nmap, msfconsole, hydra, etc.) into structured
# data that the RL environment can use for reward calculation and state updates.

"""
Output Parser for Ariaska RL.

Converts raw tool output into structured dictionaries that the
CyberEnvironment can process for state transitions and rewards.

Supports:
- Nmap (port/service/OS discovery)
- Gobuster/Dirb/Feroxbuster (web directory discovery)
- Hydra/CrackMapExec (credential discovery)
- Metasploit (exploit/session results)
- Enum4linux/SMBClient (SMB enumeration)
- Nikto/WPScan (web vulnerability scanning)
- Generic command output fallback

Usage:
    parser = OutputParser()
    result = parser.parse("nmap -sV 192.168.56.101", output_text)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from core.gpt_manager import GPTManager

logger = logging.getLogger(__name__)

# ── LLM extraction constants ──────────────────────────────────────────
LLM_PARSER_MODEL = "nano"
_LLM_MAX_OUTPUT_CHARS = 1500      # Max chars for local 4B (fast, structured)
_LLM_LONG_OUTPUT_THRESHOLD = 1500 # Outputs longer than this route to cloud 70B
_LLM_CLOUD_MAX_CHARS = 12000      # Max chars for cloud 70B (handles web pages, linpeas)


@dataclass
class ParsedOutput:
    """Structured parsed output from a command."""
    command: str
    tool: str = ""
    success: bool = False
    phase: str = "recon"
    
    # Discoveries
    open_ports: List[int] = field(default_factory=list)
    services: Dict[int, str] = field(default_factory=dict)
    os_info: str = ""
    web_paths: List[str] = field(default_factory=list)
    credentials: List[Dict[str, str]] = field(default_factory=list)
    vulnerabilities: List[str] = field(default_factory=list)
    users: List[str] = field(default_factory=list)
    shares: List[str] = field(default_factory=list)
    sessions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Artifacts (generic key-value discoveries)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    
    # Meta
    error: str = ""
    raw_output_length: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for env state updates."""
        return {
            "command": self.command,
            "tool": self.tool,
            "success": self.success,
            "phase": self.phase,
            "open_ports": self.open_ports,
            "services": self.services,
            "os_info": self.os_info,
            "web_paths": self.web_paths,
            "credentials": self.credentials,
            "vulnerabilities": self.vulnerabilities,
            "users": self.users,
            "shares": self.shares,
            "sessions": self.sessions,
            "artifacts": self.artifacts,
        }
    
    @property
    def discovery_count(self) -> int:
        """Total number of new discoveries."""
        return (
            len(self.open_ports) + len(self.services) + len(self.web_paths)
            + len(self.credentials) + len(self.vulnerabilities) + len(self.users)
            + len(self.shares) + len(self.sessions) + (1 if self.os_info else 0)
        )


class OutputParser:
    """
    Parses real command output into structured data for RL state updates.
    
    Maintains a registry of known tool parsers and falls back to
    heuristic parsing for unknown output formats.
    
    Args:
        known_ports_path: Optional path to known-ports file for enrichment
    """
    
    # Map tool names to their parser methods
    TOOL_PARSERS = {
        "nmap": "_parse_nmap",
        "masscan": "_parse_masscan",
        "rustscan": "_parse_rustscan",
        "gobuster": "_parse_web_enum",
        "dirb": "_parse_web_enum",
        "dirsearch": "_parse_web_enum",
        "feroxbuster": "_parse_web_enum",
        "ffuf": "_parse_web_enum",
        "nikto": "_parse_nikto",
        "whatweb": "_parse_whatweb",
        "wpscan": "_parse_wpscan",
        "nuclei": "_parse_nuclei",
        "hydra": "_parse_hydra",
        "crackmapexec": "_parse_crackmapexec",
        "enum4linux": "_parse_enum4linux",
        "smbclient": "_parse_smbclient",
        "smbmap": "_parse_smbmap",
        "msfconsole": "_parse_metasploit",
        "searchsploit": "_parse_searchsploit",
        "curl": "_parse_curl",
        "wget": "_parse_curl",
        # HTB Capability Upgrade
        "strings": "_parse_pcap_strings",
        "tshark": "_parse_tshark",
        "getcap": "_parse_getcap",
        "hashcat": "_parse_hashcat",
        "john": "_parse_john",
        "sshpass": "_parse_ssh_session",
        "gpp-decrypt": "_parse_gpp_decrypt",
        "impacket-GetNPUsers": "_parse_impacket_hash",
        "impacket-GetUserSPNs": "_parse_impacket_hash",
        "bloodhound-python": "_parse_bloodhound",
        # Phase 19: CrushFTP / Erlang / vhost
        "erl": "_parse_erlang",
    }
    
    def __init__(
        self,
        known_ports_path: Optional[str] = None,
        gpt_manager: Optional["GPTManager"] = None,
    ):
        self._gpt = gpt_manager
        self._all_discovered_ports: Set[int] = set()
        self._all_discovered_services: Dict[int, str] = {}
        self._all_credentials: List[Dict[str, str]] = []
        self._all_vulns: List[str] = []
    
    def parse(self, command: str, output: str) -> ParsedOutput:
        """Parse command output into structured data.

        Pipeline:
          1. LLM extraction (if GPTManager available) — robust, handles
             non-standard formats and mixed output.
          2. Regex fallback — zero-cost, deterministic, tool-specific.
        """
        if not command or not output:
            return ParsedOutput(command=command or "", error="Empty command or output")

        # Strip ANSI escape codes that break both regex and LLM parsing.
        output = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', output)
        output = output.replace('\r', '')

        tool = self._identify_tool(command)

        # ── Stage 1: LLM extraction (primary) ────────────────────────
        if self._gpt and len(output.strip()) >= 30:
            try:
                llm_result = self._llm_extract(command, output, tool)
                if llm_result and llm_result.discovery_count > 0:
                    llm_result.command = command
                    llm_result.tool = tool
                    llm_result.raw_output_length = len(output)
                    self._normalize_services(llm_result)
                    self._guard_credentials(llm_result)
                    self._sanitize_web_paths(llm_result)
                    self._update_global_discoveries(llm_result)
                    logger.debug(
                        f"[OUTPUT-PARSER-LLM] {tool}: "
                        f"{llm_result.discovery_count} discoveries"
                    )
                    return llm_result
            except Exception as e:
                logger.debug(f"[OUTPUT-PARSER-LLM] {tool} failed, regex fallback: {e}")

        # ── Stage 2: Regex fallback ───────────────────────────────────
        parser_name = self.TOOL_PARSERS.get(tool, "_parse_generic")
        parser_method = getattr(self, parser_name, self._parse_generic)

        try:
            result = parser_method(command, output)
            result.command = command
            result.tool = tool
            result.raw_output_length = len(output)
            self._normalize_services(result)
            self._guard_credentials(result)
            self._sanitize_web_paths(result)
            self._update_global_discoveries(result)
            return result
        except Exception as e:
            logger.warning(f"Parse error for {tool}: {e}")
            return ParsedOutput(
                command=command,
                tool=tool,
                error=str(e),
                raw_output_length=len(output),
            )

    # ------------------------------------------------------------------
    # LLM-based structured extraction
    # ------------------------------------------------------------------

    _LLM_SYSTEM_PROMPT = (
        "You are a penetration-testing output parser. "
        "Given a command and its STDOUT, extract structured discoveries. "
        "Return ONLY valid JSON — no markdown, no explanation."
    )

    _LLM_EXTRACTION_PROMPT = """\
Parse this penetration testing tool output and extract all discoveries as JSON.

Tool: {tool}
Command: {command}
STDOUT:
```
{output}
```

Return a JSON object with ONLY the keys that have findings (omit empty keys):
- "open_ports": [list of integer port numbers]
- "services": {{"port_int": "service_name version_string"}}  (port as string key)
- "os_info": "OS identification string"
- "web_paths": [list of discovered URL paths]
- "credentials": [{{"username": "...", "password": "...", "service": "..."}}]
- "vulnerabilities": [list of CVE IDs or vulnerability descriptions]
- "users": [list of discovered usernames]
- "shares": [list of SMB/NFS share names]
- "sessions": [{{"type": "meterpreter|shell|root", "id": N}}]
- "artifacts": {{"key": "value"}} for any other notable findings
- "success": true/false whether the command achieved its goal

JSON:"""

    def _llm_extract(
        self, command: str, output: str, tool: str,
    ) -> Optional[ParsedOutput]:
        """Use LLM to parse STDOUT into a ParsedOutput.

        Smart routing (Phase 57):
          - Short outputs (<=1500 chars): local 4B, fast classification
          - Long outputs (>1500 chars): cloud 70B with more context
            (web pages, linpeas, big scans lose critical info when truncated)
        """
        if not self._gpt:
            return None

        is_long = len(output) > _LLM_LONG_OUTPUT_THRESHOLD
        if is_long:
            # Route to cloud 70B — can handle full web pages, linpeas reports
            truncated = output[:_LLM_CLOUD_MAX_CHARS]
            task_type = "strategic"   # model_router sends to cloud 70B
            max_tokens = 600          # cloud can produce richer extraction
        else:
            # Route to local 4B — fast structured extraction
            truncated = output[:_LLM_MAX_OUTPUT_CHARS]
            task_type = "classification"  # model_router sends to local 4B
            max_tokens = 300

        prompt = self._LLM_EXTRACTION_PROMPT.format(
            tool=tool, command=command, output=truncated,
        )

        response = self._gpt.gpt_request(
            prompt=prompt,
            task_type=task_type,
            agent_id=f"output_parser_{tool}",
            max_tokens=max_tokens,
            model=LLM_PARSER_MODEL,
            temperature=0.1,
            system_prompt=self._LLM_SYSTEM_PROMPT,
            response_format="json_object",
        )

        if not response:
            return None

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            # Fallback: extract first JSON object from response
            m = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if not m:
                return None
            data = json.loads(m.group())

        return self._json_to_parsed_output(data, command, tool)

    @staticmethod
    def _json_to_parsed_output(
        data: Dict[str, Any], command: str, tool: str,
    ) -> ParsedOutput:
        """Convert a raw JSON dict from LLM into a ParsedOutput dataclass."""
        result = ParsedOutput(command=command, tool=tool)

        # open_ports
        for p in data.get("open_ports", []):
            try:
                port = int(p)
                if 0 < port < 65536:
                    result.open_ports.append(port)
            except (ValueError, TypeError):
                pass

        # services  {"22": "ssh OpenSSH 7.2p2", ...}
        raw_svcs = data.get("services", {})
        if isinstance(raw_svcs, dict):
            for k, v in raw_svcs.items():
                try:
                    result.services[int(k)] = str(v)
                except (ValueError, TypeError):
                    pass

        # os_info
        if data.get("os_info"):
            result.os_info = str(data["os_info"])

        # web_paths
        for wp in data.get("web_paths", []):
            if isinstance(wp, str) and wp.startswith("/"):
                result.web_paths.append(wp)

        # credentials
        for cred in data.get("credentials", []):
            if isinstance(cred, dict):
                result.credentials.append({
                    k: str(v) for k, v in cred.items()
                    if k in ("username", "password", "service", "host", "port", "source", "hash")
                })

        # vulnerabilities
        for v in data.get("vulnerabilities", []):
            if isinstance(v, str):
                result.vulnerabilities.append(v)

        # users
        for u in data.get("users", []):
            if isinstance(u, str) and u not in result.users:
                result.users.append(u)

        # shares
        for s in data.get("shares", []):
            if isinstance(s, str):
                result.shares.append(s)

        # sessions
        for sess in data.get("sessions", []):
            if isinstance(sess, dict):
                result.sessions.append(sess)

        # artifacts
        raw_artifacts = data.get("artifacts", {})
        if isinstance(raw_artifacts, dict):
            result.artifacts.update(raw_artifacts)

        # success
        result.success = bool(data.get("success", result.discovery_count > 0))

        # Infer phase from discoveries
        if result.sessions:
            result.phase = "privesc"
        elif result.credentials:
            result.phase = "exploit"
        elif result.vulnerabilities:
            result.phase = "exploit"
        elif result.open_ports:
            result.phase = "recon"
        elif result.web_paths or result.users or result.shares:
            result.phase = "enumeration"

        return result
    
    def _identify_tool(self, command: str) -> str:
        """Identify the base tool from a command string."""
        import os
        import shlex
        
        try:
            parts = shlex.split(command)
        except ValueError:
            parts = command.split()
        
        if not parts:
            return "unknown"
        
        base = os.path.basename(parts[0]).lower()
        
        # Handle sudo prefix
        if base == "sudo" and len(parts) > 1:
            base = os.path.basename(parts[1]).lower()
        
        # Handle python/perl scripts that wrap tools
        if base in ("python", "python3", "perl", "ruby") and len(parts) > 1:
            script_name = os.path.basename(parts[1]).lower()
            for tool_name in self.TOOL_PARSERS:
                if tool_name in script_name:
                    return tool_name
        
        return base
    
    # =========================================================================
    # Tool-specific parsers
    # =========================================================================
    
    def _parse_nmap(self, command: str, output: str) -> ParsedOutput:
        """Parse nmap output for ports, services, OS info."""
        result = ParsedOutput(command=command, phase="recon")
        
        # Parse open ports: "22/tcp open ssh OpenSSH 7.2p2"
        port_pattern = r'(\d+)/(?:tcp|udp)\s+open\s+(\S+)(?:\s+(.+))?'
        for match in re.finditer(port_pattern, output):
            port = int(match.group(1))
            service = match.group(2)
            version = match.group(3) or ""
            result.open_ports.append(port)
            result.services[port] = f"{service} {version}".strip()
        
        # Parse OS detection
        os_pattern = r'OS details?:\s*(.+?)(?:\n|$)'
        os_match = re.search(os_pattern, output)
        if os_match:
            result.os_info = os_match.group(1).strip()
        
        # Aggressive OS guess
        os_guess_pattern = r'Aggressive OS guesses?:\s*(.+?)(?:\n|$)'
        os_guess = re.search(os_guess_pattern, output)
        if os_guess and not result.os_info:
            result.os_info = os_guess.group(1).strip()
        
        # Check for script results (NSE)
        vuln_pattern = r'(CVE-\d{4}-\d+|VULNERABLE|EXPLOITABLE)'
        for match in re.finditer(vuln_pattern, output, re.IGNORECASE):
            result.vulnerabilities.append(match.group(1))
        
        result.success = len(result.open_ports) > 0
        if result.success:
            result.phase = "recon" if len(result.open_ports) < 5 else "enumeration"
        
        return result
    
    def _parse_masscan(self, command: str, output: str) -> ParsedOutput:
        """Parse masscan output."""
        result = ParsedOutput(command=command, phase="recon")
        
        # "Discovered open port 22/tcp on 192.168.56.101"
        pattern = r'Discovered open port (\d+)/(?:tcp|udp) on (\S+)'
        for match in re.finditer(pattern, output):
            port = int(match.group(1))
            result.open_ports.append(port)
        
        result.success = len(result.open_ports) > 0
        return result
    
    def _parse_rustscan(self, command: str, output: str) -> ParsedOutput:
        """Parse rustscan output."""
        result = ParsedOutput(command=command, phase="recon")
        
        # "Open 192.168.56.101:22"
        pattern = r'Open\s+\S+:(\d+)'
        for match in re.finditer(pattern, output):
            result.open_ports.append(int(match.group(1)))
        
        result.success = len(result.open_ports) > 0
        return result
    
    def _parse_web_enum(self, command: str, output: str) -> ParsedOutput:
        """Parse gobuster/dirb/feroxbuster/ffuf/dirsearch output."""
        result = ParsedOutput(command=command, phase="enumeration")
        
        # Various formats for discovered paths
        patterns = [
            r'(/\S+)\s+\(Status:\s*(\d+)',      # gobuster
            r'\+\s+https?://\S+(/.+?)\s+\(CODE:(\d+)',  # dirb
            r'(\d{3})\s+\w+\s+(/\S+)',            # feroxbuster
            r'(\S+)\s+\[Status:\s*(\d+)',          # ffuf
            r'\[(\d{3})\]\s+(https?://\S+)',       # dirsearch
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, output):
                groups = match.groups()
                # Extract the path part
                path = None
                status = None
                for g in groups:
                    if g and g.startswith('/'):
                        path = g
                    elif g and g.isdigit() and int(g) < 600:
                        status = int(g)
                    elif g and 'http' in g:
                        # Extract path from URL
                        url_path = re.search(r'https?://[^/]+(/.+)', g)
                        if url_path:
                            path = url_path.group(1)
                
                if path and path not in result.web_paths:
                    result.web_paths.append(path)
        
        # Fallback: look for any path-like strings with status codes
        if not result.web_paths:
            simple_pattern = r'(/[a-zA-Z0-9_./\-]+)'
            for match in re.finditer(simple_pattern, output):
                path = match.group(1)
                if len(path) > 1 and path not in result.web_paths:
                    result.web_paths.append(path)
        
        result.success = len(result.web_paths) > 0
        return result
    
    def _parse_nikto(self, command: str, output: str) -> ParsedOutput:
        """Parse nikto vulnerability scanner output."""
        result = ParsedOutput(command=command, phase="enumeration")
        
        # "+ Server: Apache/2.4.41"
        server_match = re.search(r'\+\s+Server:\s*(.+)', output)
        if server_match:
            result.artifacts["server"] = server_match.group(1).strip()
        
        # "+ OSVDB-XXXX: /path: description"
        vuln_pattern = r'\+\s+(OSVDB-\d+|CVE-\d{4}-\d+):\s*(.+)'
        for match in re.finditer(vuln_pattern, output):
            result.vulnerabilities.append(f"{match.group(1)}: {match.group(2).strip()}")
        
        # Discovered paths
        path_pattern = r'\+\s+(/\S+):\s+(.+)'
        for match in re.finditer(path_pattern, output):
            path = match.group(1)
            if path not in result.web_paths:
                result.web_paths.append(path)
        
        result.success = len(result.vulnerabilities) > 0 or len(result.web_paths) > 0
        return result
    
    def _parse_whatweb(self, command: str, output: str) -> ParsedOutput:
        """Parse whatweb output for technology detection."""
        result = ParsedOutput(command=command, phase="enumeration")
        
        # Extract technology markers: "Apache[2.4.41], PHP[7.4.3]"
        tech_pattern = r'(\w+)\[([^\]]+)\]'
        for match in re.finditer(tech_pattern, output):
            tech = match.group(1)
            version = match.group(2)
            result.artifacts[tech] = version
        
        result.success = len(result.artifacts) > 0
        return result
    
    def _parse_wpscan(self, command: str, output: str) -> ParsedOutput:
        """Parse WPScan output."""
        result = ParsedOutput(command=command, phase="enumeration")
        
        # WordPress version
        wp_ver = re.search(r'WordPress version (\S+)', output)
        if wp_ver:
            result.artifacts["wordpress_version"] = wp_ver.group(1)
        
        # Users
        user_pattern = r'User found:\s*(\S+)'
        for match in re.finditer(user_pattern, output):
            result.users.append(match.group(1))
        
        # Vulnerable plugins
        vuln_pattern = r'Vulnerable (?:plugin|theme):\s*(.+?)(?:\n|$)'
        for match in re.finditer(vuln_pattern, output, re.IGNORECASE):
            result.vulnerabilities.append(match.group(1).strip())
        
        result.success = len(result.users) > 0 or len(result.vulnerabilities) > 0
        return result
    
    def _parse_nuclei(self, command: str, output: str) -> ParsedOutput:
        """Parse nuclei vulnerability scanner output."""
        result = ParsedOutput(command=command, phase="enumeration")
        
        # "[CVE-2021-41773] Apache Path Traversal: target:80"
        vuln_pattern = r'\[(CVE-\d{4}-\d+|[a-z\-]+)\]\s+(.+?)(?:\n|$)'
        for match in re.finditer(vuln_pattern, output):
            vuln_id = match.group(1)
            description = match.group(2).strip()
            result.vulnerabilities.append(f"{vuln_id}: {description}")
        
        result.success = len(result.vulnerabilities) > 0
        if result.vulnerabilities:
            result.phase = "exploit"
        return result
    
    def _parse_hydra(self, command: str, output: str) -> ParsedOutput:
        """Parse hydra brute-force output."""
        result = ParsedOutput(command=command, phase="exploit")
        
        # "[22][ssh] host: 192.168.56.101 login: admin password: admin123"
        cred_pattern = r'\[(\d+)\]\[(\w+)\]\s+host:\s+(\S+)\s+login:\s+(\S+)\s+password:\s+(\S+)'
        for match in re.finditer(cred_pattern, output):
            result.credentials.append({
                "port": match.group(1),
                "service": match.group(2),
                "host": match.group(3),
                "username": match.group(4),
                "password": match.group(5),
            })
        
        result.success = len(result.credentials) > 0
        return result
    
    def _parse_crackmapexec(self, command: str, output: str) -> ParsedOutput:
        """Parse CrackMapExec output."""
        result = ParsedOutput(command=command, phase="exploit")
        
        # "SMB 192.168.56.101 445 TARGET [+] admin:Password123! (Pwn3d!)"
        cred_pattern = r'\[\+\]\s+(\S+):(\S+)'
        pwn_pattern = r'\(Pwn3d!\)'
        
        for match in re.finditer(cred_pattern, output):
            cred = {
                "username": match.group(1),
                "password": match.group(2),
                "pwned": bool(re.search(pwn_pattern, output)),
            }
            result.credentials.append(cred)
        
        result.success = len(result.credentials) > 0
        return result
    
    def _parse_enum4linux(self, command: str, output: str) -> ParsedOutput:
        """Parse enum4linux output."""
        result = ParsedOutput(command=command, phase="enumeration")
        
        # Users
        user_pattern = r'user:\[(\S+)\]'
        for match in re.finditer(user_pattern, output):
            result.users.append(match.group(1))
        
        # Shares
        share_pattern = r'Shares?:\s*(\S+)'
        for match in re.finditer(share_pattern, output):
            shares = match.group(1).split(',')
            result.shares.extend([s.strip() for s in shares])
        
        # Password policy
        pwd_policy = re.search(r'Password policy:?\s*(.+)', output, re.IGNORECASE)
        if pwd_policy:
            result.artifacts["password_policy"] = pwd_policy.group(1).strip()
        
        result.success = len(result.users) > 0 or len(result.shares) > 0
        return result
    
    def _parse_smbclient(self, command: str, output: str) -> ParsedOutput:
        """Parse smbclient output."""
        result = ParsedOutput(command=command, phase="enumeration")
        
        # ShareName  Type  Comment
        share_pattern = r'(\S+)\s+(Disk|IPC|Printer)\s'
        for match in re.finditer(share_pattern, output):
            result.shares.append(match.group(1))
        
        result.success = len(result.shares) > 0
        return result
    
    def _parse_smbmap(self, command: str, output: str) -> ParsedOutput:
        """Parse smbmap output."""
        result = ParsedOutput(command=command, phase="enumeration")
        
        # "[+] Disk: backup (READ)"
        share_pattern = r'Disk:\s+(\S+)\s+\((\w+)\)'
        for match in re.finditer(share_pattern, output):
            share_name = match.group(1)
            access = match.group(2)
            result.shares.append(share_name)
            result.artifacts[f"share_{share_name}_access"] = access
        
        result.success = len(result.shares) > 0
        return result
    
    def _parse_metasploit(self, command: str, output: str) -> ParsedOutput:
        """Parse Metasploit console output."""
        result = ParsedOutput(command=command, phase="exploit")
        
        # Session opened
        session_pattern = r'(Meterpreter|shell)\s+session\s+(\d+)\s+opened\s+\((\S+)\s*->\s*(\S+)\)'
        for match in re.finditer(session_pattern, output, re.IGNORECASE):
            result.sessions.append({
                "type": match.group(1),
                "id": int(match.group(2)),
                "source": match.group(3),
                "target": match.group(4),
            })
        
        # Exploit successful
        if re.search(r'exploit completed|session \d+ opened|command shell session', output, re.IGNORECASE):
            result.success = True
            result.phase = "privesc"
        
        # Root/SYSTEM
        if re.search(r'uid=0\(root\)|NT AUTHORITY\\SYSTEM|Administrator', output):
            result.artifacts["privilege_level"] = "root"
            result.phase = "exfiltrate"
        
        return result
    
    def _parse_searchsploit(self, command: str, output: str) -> ParsedOutput:
        """Parse searchsploit output."""
        result = ParsedOutput(command=command, phase="enumeration")
        
        # "Apache 2.4.41 | exploits/linux/remote/12345.py"
        exploit_pattern = r'(.+?)\s+\|\s+(exploits/\S+)'
        for match in re.finditer(exploit_pattern, output):
            result.vulnerabilities.append(f"{match.group(1).strip()}: {match.group(2)}")
        
        result.success = len(result.vulnerabilities) > 0
        return result
    
    def _parse_curl(self, command: str, output: str) -> ParsedOutput:
        """Parse curl/wget output for headers and content."""
        result = ParsedOutput(command=command, phase="enumeration")
        
        # HTTP headers
        server_match = re.search(r'Server:\s*(.+?)(?:\r?\n|$)', output)
        if server_match:
            result.artifacts["server"] = server_match.group(1).strip()
        
        powered_by = re.search(r'X-Powered-By:\s*(.+?)(?:\r?\n|$)', output)
        if powered_by:
            result.artifacts["x_powered_by"] = powered_by.group(1).strip()
        
        cookies = re.findall(r'Set-Cookie:\s*(\S+)', output)
        if cookies:
            result.artifacts["cookies"] = cookies
        
        # Check for file download indicators 
        if "saved" in output.lower() or re.search(r'\d+ bytes', output):
            result.artifacts["file_downloaded"] = True
        
        result.success = len(result.artifacts) > 0
        return result
    
    # =========================================================================
    # HTB Capability Upgrade — New tool parsers
    # =========================================================================
    
    def _parse_pcap_strings(self, command: str, output: str) -> ParsedOutput:
        """Parse strings/tshark output from PCAP files for credentials."""
        result = ParsedOutput(command=command, phase="exploit")
        
        # FTP USER/PASS pairs
        ftp_users = re.findall(r'USER\s+(\S+)', output)
        ftp_passes = re.findall(r'PASS\s+(\S+)', output)
        if ftp_users and ftp_passes:
            for user, passwd in zip(ftp_users, ftp_passes):
                if user.lower() not in ("anonymous", "ftp"):
                    result.credentials.append({
                        "username": user,
                        "password": passwd,
                        "service": "ftp",
                        "source": "pcap",
                    })
        
        # HTTP Basic Auth (base64 in headers)
        auth_match = re.findall(r'Authorization:\s*Basic\s+(\S+)', output)
        if auth_match:
            import base64
            for b64 in auth_match:
                try:
                    decoded = base64.b64decode(b64).decode('utf-8', errors='replace')
                    if ':' in decoded:
                        user, passwd = decoded.split(':', 1)
                        result.credentials.append({
                            "username": user,
                            "password": passwd,
                            "service": "http",
                            "source": "pcap",
                        })
                except Exception:
                    pass
        
        # Telnet login/password
        telnet_user = re.findall(r'login:\s*(\S+)', output, re.IGNORECASE)
        telnet_pass = re.findall(r'Password:\s*(\S+)', output, re.IGNORECASE)
        if telnet_user and telnet_pass:
            for user, passwd in zip(telnet_user, telnet_pass):
                result.credentials.append({
                    "username": user,
                    "password": passwd,
                    "service": "telnet",
                    "source": "pcap",
                })
        
        # Generic username/password discovery
        for match in re.finditer(r'(?:user(?:name)?|login)[:\s=]+(\S+)', output, re.IGNORECASE):
            user = match.group(1)
            if user not in result.users and len(user) > 1:
                result.users.append(user)
        
        result.success = len(result.credentials) > 0
        if result.credentials:
            result.artifacts["pcap_credentials_found"] = True
        return result
    
    def _parse_getcap(self, command: str, output: str) -> ParsedOutput:
        """Parse getcap output for Linux capabilities."""
        result = ParsedOutput(command=command, phase="privesc")
        
        # Pattern: /usr/bin/python3.8 cap_setuid,cap_net_bind_service=eip
        cap_pattern = r'(\S+)\s*=\s*(cap_\S+)'
        for match in re.finditer(cap_pattern, output):
            binary = match.group(1)
            caps = match.group(2)
            result.artifacts[f"capability_{binary}"] = caps
            result.vulnerabilities.append(f"capability:{binary}={caps}")
            
            # Flag dangerous capabilities
            if "cap_setuid" in caps:
                result.artifacts["cap_setuid_binary"] = binary
                result.artifacts["privesc_vector"] = "cap_setuid"
        
        # Also match format: /usr/bin/python3.8 = cap_setuid+ep
        cap_pattern2 = r'(\S+)\s+(cap_\S+\+\S+)'
        for match in re.finditer(cap_pattern2, output):
            binary = match.group(1)
            caps = match.group(2)
            result.artifacts[f"capability_{binary}"] = caps
            
            if "cap_setuid" in caps:
                result.artifacts["cap_setuid_binary"] = binary
                result.artifacts["privesc_vector"] = "cap_setuid"
        
        result.success = len(result.vulnerabilities) > 0
        return result
    
    def _parse_hashcat(self, command: str, output: str) -> ParsedOutput:
        """Parse hashcat output for cracked passwords."""
        result = ParsedOutput(command=command, phase="exploit")
        
        # Cracked format: hash:password
        if "Status...........: Cracked" in output or "Cracked" in output:
            # Look for the cracked result
            crack_pattern = r'(\S+):(\S+)$'
            for line in output.strip().split('\n'):
                match = re.match(crack_pattern, line.strip())
                if match and not line.startswith('#'):
                    result.credentials.append({
                        "hash": match.group(1),
                        "password": match.group(2),
                        "source": "hashcat",
                    })
        
        result.success = len(result.credentials) > 0
        return result
    
    def _parse_john(self, command: str, output: str) -> ParsedOutput:
        """Parse John the Ripper output for cracked passwords."""
        result = ParsedOutput(command=command, phase="exploit")
        
        # John --show format: username:password:uid:gid:...
        for line in output.strip().split('\n'):
            if ':' in line and not line.startswith('#') and not line.startswith('Using'):
                parts = line.strip().split(':')
                if len(parts) >= 2 and parts[1]:
                    result.credentials.append({
                        "username": parts[0],
                        "password": parts[1],
                        "source": "john",
                    })
        
        # "X password hashes cracked"
        cracked_match = re.search(r'(\d+) password hash(?:es)? cracked', output)
        if cracked_match:
            result.artifacts["hashes_cracked"] = int(cracked_match.group(1))
        
        result.success = len(result.credentials) > 0
        return result
    
    def _parse_ssh_session(self, command: str, output: str) -> ParsedOutput:
        """Parse sshpass/SSH session output for successful login + discoveries."""
        result = ParsedOutput(command=command, phase="exploit")
        
        # Successful login indicators
        if re.search(r'uid=\d+', output):
            result.success = True
            # Extract uid info
            uid_match = re.search(r'uid=(\d+)\((\w+)\)', output)
            if uid_match:
                uid = int(uid_match.group(1))
                username = uid_match.group(2)
                result.users.append(username)
                if uid == 0:
                    result.artifacts["privilege_level"] = "root"
                    result.sessions.append({"type": "root_shell", "user": username})
                else:
                    result.artifacts["privilege_level"] = "user"
                    result.sessions.append({"type": "user_shell", "user": username})
        
        # Check for /etc/passwd content (useful for user enumeration)
        passwd_pattern = r'(\w+):x:(\d+):\d+:.*:/home/\1'
        for match in re.finditer(passwd_pattern, output):
            user = match.group(1)
            if user not in result.users:
                result.users.append(user)
        
        # Permission denied = failed
        if "Permission denied" in output or "Connection refused" in output:
            result.success = False
            result.error = "Authentication failed"
        
        return result
    
    def _parse_gpp_decrypt(self, command: str, output: str) -> ParsedOutput:
        """Parse gpp-decrypt output for decrypted password."""
        result = ParsedOutput(command=command, phase="exploit")
        
        # gpp-decrypt just outputs the plaintext password
        decrypted = output.strip()
        if decrypted and len(decrypted) < 100 and not decrypted.startswith("Usage"):
            result.credentials.append({
                "password": decrypted,
                "source": "gpp-decrypt",
                "service": "active_directory",
            })
            result.success = True
        
        return result
    
    def _parse_impacket_hash(self, command: str, output: str) -> ParsedOutput:
        """Parse impacket GetNPUsers/GetUserSPNs output for Kerberos hashes."""
        result = ParsedOutput(command=command, phase="exploit")
        
        # AS-REP hash: $krb5asrep$23$username@DOMAIN:...
        asrep_pattern = r'(\$krb5asrep\$\d+\$\S+)'
        for match in re.finditer(asrep_pattern, output):
            result.artifacts.setdefault("hashes", []).append(match.group(1))
            result.vulnerabilities.append("AS-REP Roastable account found")
        
        # TGS hash: $krb5tgs$23$*username$DOMAIN$...
        tgs_pattern = r'(\$krb5tgs\$\d+\$\S+)'
        for match in re.finditer(tgs_pattern, output):
            result.artifacts.setdefault("hashes", []).append(match.group(1))
            result.vulnerabilities.append("Kerberoastable service account found")
        
        result.success = len(result.artifacts.get("hashes", [])) > 0
        return result
    
    def _parse_bloodhound(self, command: str, output: str) -> ParsedOutput:
        """Parse bloodhound-python output for collected data."""
        result = ParsedOutput(command=command, phase="enumeration")
        
        # "Done in 00mXXs... X users, X computers, X groups"
        done_match = re.search(r'Done.*?(\d+)\s+users.*?(\d+)\s+computers.*?(\d+)\s+groups', output)
        if done_match:
            result.artifacts["ad_users"] = int(done_match.group(1))
            result.artifacts["ad_computers"] = int(done_match.group(2))
            result.artifacts["ad_groups"] = int(done_match.group(3))
            result.success = True
        
        return result
    
    def _parse_generic(self, command: str, output: str) -> ParsedOutput:
        """
        Generic fallback parser using heuristics.
        
        Looks for common patterns: IPs, ports, usernames, file paths, etc.
        """
        result = ParsedOutput(command=command, phase="recon")
        
        # Look for open ports
        port_pattern = r'(\d{1,5})/(?:tcp|udp)\s+open'
        for match in re.finditer(port_pattern, output):
            port = int(match.group(1))
            if 0 < port < 65536:
                result.open_ports.append(port)
        
        # Look for usernames
        user_patterns = [
            r'user:\s*(\S+)',
            r'username:\s*(\S+)',
            r'login:\s*(\S+)',
            r'uid=\d+\((\S+)\)',
        ]
        for pattern in user_patterns:
            for match in re.finditer(pattern, output, re.IGNORECASE):
                user = match.group(1)
                if user not in result.users:
                    result.users.append(user)
        
        # Look for CVEs
        cve_pattern = r'(CVE-\d{4}-\d+)'
        for match in re.finditer(cve_pattern, output):
            if match.group(1) not in result.vulnerabilities:
                result.vulnerabilities.append(match.group(1))
        
        # Look for password/credential patterns
        # P50: Tighter regex — require at least 2 alphanumeric chars in value
        # to avoid matching garbled output like "password: x\r\n');"
        cred_pattern = r'(?:password|passwd|pass)[\s:=]+([a-zA-Z0-9][a-zA-Z0-9!@#$%^&*._+-]{1,127})'
        for match in re.finditer(cred_pattern, output, re.IGNORECASE):
            result.credentials.append({"password": match.group(1)})
        
        # Check for success indicators
        success_indicators = [
            "success", "found", "discovered", "opened", "connected",
            "authenticated", "granted", "Pwn3d"
        ]
        output_lower = output.lower()
        result.success = any(ind.lower() in output_lower for ind in success_indicators)
        
        return result
    
    def _update_global_discoveries(self, result: ParsedOutput):
        """Update global discovery tracking."""
        self._all_discovered_ports.update(result.open_ports)
        self._all_discovered_services.update(result.services)
        self._all_credentials.extend(result.credentials)
        self._all_vulns.extend(result.vulnerabilities)
    
    def get_all_discoveries(self) -> Dict[str, Any]:
        """Get cumulative discoveries across all parsed outputs."""
        return {
            "total_ports": len(self._all_discovered_ports),
            "ports": sorted(self._all_discovered_ports),
            "services": dict(self._all_discovered_services),
            "credentials": self._all_credentials,
            "vulnerabilities": list(set(self._all_vulns)),
        }
    
    def reset(self):
        """Reset all discovery tracking."""
        self._all_discovered_ports.clear()
        self._all_discovered_services.clear()
        self._all_credentials.clear()
        self._all_vulns.clear()

    # =========================================================================
    # Phase 19: Ultra-Smart Parsers — CrushFTP, Erlang, tshark, vhost
    # =========================================================================

    def _parse_tshark(self, command: str, output: str) -> ParsedOutput:
        """Parse tshark PCAP extraction output for credentials."""
        result = ParsedOutput(command=command, phase="enumeration")

        # FTP USER/PASS in tshark -T fields format (tab-separated)
        users: list = []
        current_user = None
        for line in output.split("\n"):
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                cmd_part, arg = parts[0].strip(), parts[1].strip()
                if cmd_part == "USER":
                    current_user = arg
                elif cmd_part == "PASS" and current_user:
                    if current_user.lower() not in ("anonymous", "ftp", "guest"):
                        result.credentials.append({
                            "username": current_user,
                            "password": arg,
                            "source": "pcap_tshark",
                        })
                        if current_user not in result.users:
                            result.users.append(current_user)
                    current_user = None

        # Also catch inline credential patterns
        for match in re.finditer(r'credential:\s*(\S+):(\S+)', output):
            result.credentials.append({
                "username": match.group(1),
                "password": match.group(2),
                "source": "pcap",
            })

        result.success = len(result.credentials) > 0
        return result

    def _parse_erlang(self, command: str, output: str) -> ParsedOutput:
        """Parse Erlang OTP RCE output — uid, flags, cookie."""
        result = ParsedOutput(command=command, phase="privilege_escalation")

        # Erlang cookie in output
        cookie_match = re.search(r'cookie[:\s]+(\S+)', output, re.IGNORECASE)
        if cookie_match:
            result.credentials.append({
                "username": "erlang_cookie",
                "password": cookie_match.group(1),
                "source": "erlang",
            })

        # uid=0(root) — root shell confirmation
        uid_match = re.search(r'uid=0\(root\)', output)
        if uid_match:
            result.success = True
            result.sessions.append({
                "type": "root_shell",
                "via": "erlang_otp_rce",
            })

        # Flag extraction
        flag_match = re.search(r'FLAG\{[^}]+\}', output)
        if flag_match:
            result.extra_info = result.extra_info or ""
            result.extra_info += f" flag={flag_match.group(0)}"
            result.success = True

        return result

    def parse_crushftp(self, command: str, output: str) -> ParsedOutput:
        """Parse CrushFTP API responses — user lists, password resets."""
        result = ParsedOutput(command=command, phase="exploitation")

        # XML user list from getUserList
        user_pattern = r'<username>(\w+)</username>'
        for match in re.finditer(user_pattern, output):
            username = match.group(1)
            if username not in result.users:
                result.users.append(username)

        # CVE detection
        if "CVE-2025-31161" in output or "auth bypass" in output.lower():
            result.vulnerabilities.append("CVE-2025-31161")
            result.success = True

        # Credential extraction from credential: markers
        for match in re.finditer(r'credential:\s*(\S+):(\S+)', output):
            result.credentials.append({
                "username": match.group(1),
                "password": match.group(2),
                "source": "crushftp",
            })

        # Password reset confirmation
        if "password changed" in output.lower() or "success" in output.lower():
            result.success = True

        return result

    # ── Phase 38.1: Post-parse normalization ──────────────────────────────

    # Canonical service name mapping — collapse aliases into canonical forms
    # so "http" and "HTTP" and "www" and "httpd" all become "http".
    _SERVICE_ALIASES: Dict[str, str] = {
        "www": "http", "httpd": "http", "apache": "http", "nginx": "http",
        "www-http": "http", "http-proxy": "http",
        "ssl/http": "https", "ssl/https": "https", "http-ssl": "https",
        "sshd": "ssh", "openssh": "ssh",
        "smbd": "smb", "microsoft-ds": "smb", "netbios-ssn": "smb",
        "ftpd": "ftp", "vsftpd": "ftp", "proftpd": "ftp",
        "mysqld": "mysql", "mariadb": "mysql",
        "postgres": "postgresql",
        "domain": "dns", "named": "dns",
        "imapd": "imap", "imap4": "imap",
        "pop3d": "pop3", "pop-3": "pop3",
        "smtp": "smtp", "smtpd": "smtp", "postfix": "smtp",
        "ms-sql-s": "mssql", "ms-sql": "mssql",
        "rpcbind": "rpc",
    }

    # Well-known ports → expected service (for validation)
    _PORT_SERVICE_MAP: Dict[int, str] = {
        21: "ftp", 22: "ssh", 25: "smtp", 53: "dns",
        80: "http", 110: "pop3", 111: "rpc", 139: "smb",
        143: "imap", 443: "https", 445: "smb", 993: "imaps",
        995: "pop3s", 1433: "mssql", 1521: "oracle",
        3306: "mysql", 3389: "rdp", 5432: "postgresql",
        5900: "vnc", 6379: "redis", 8080: "http", 8443: "https",
        27017: "mongodb",
    }

    # Junk web paths to filter out
    # Phase 42: Added IP-only paths, version paths, gobuster/tool progress garbage
    _JUNK_WEB_PATHS: FrozenSet[str] = frozenset({
        "/", "/index.html", "/index.php", "/favicon.ico",
        "/.htaccess", "/.htpasswd", "/server-status",
        "/cgi-bin/", "/icons/", "/manual/",
        # Phase 42: Tool output garbage
        "/3.6", "/3.5", "/3.4", "/2.0", "/1.0", "/1.1",  # version-like paths from tool output
        "/v1", "/v2", "/v3",  # version prefixes
        # IIS-specific nikto false positives (invalid on nginx/Apache)
        "/_vti_adm/admin.dll", "/_vti_aut/author.dll",
        "/_vti_bin/shtml.dll", "/_vti_bin/_vti_adm/admin.dll",
        "/_vti_bin/_vti_aut/author.dll", "/_vti_bin/shtml.dll",
        "/_vti_cnf/", "/_vti_pvt/", "/_vti_log/",
        "/shtml.dll", "/admin.dll", "/author.dll",
        "/_vti_inf.html", "/_vti_rpc",
    })

    # Phase 42: Regex patterns that indicate garbage web paths from tool output
    _WEB_PATH_GARBAGE_RE = re.compile(
        r'^(?:'
        r'//\d+\.\d+\.\d+\.\d+|'          # //10.129.3.142 (gobuster URL prefix)
        r'/\d+\.\d+(?:\.\d+)?$|'            # /3.6, /1.18.0 (version numbers)
        r'/\d+$|'                              # /50, /302, /404 (status codes/numbers)
        r'http[s]?://\d+\.\d+\.\d+\.\d+$'  # bare IP URLs without path
        r')'
    )

    # Credential false positive patterns — match whole credential values only
    _CRED_FALSE_POSITIVES = re.compile(
        r"^(?:"
        r"example\.com|test@test|admin:admin|root:root|"
        r"user:pass|username:password|changeme|"
        r"\*\*\*|xxx|placeholder"
        r")$",
        re.IGNORECASE,
    )

    # P50: Regex for garbled/non-printable credential values
    # Catches control chars, unbalanced quotes/parens, HTML fragments, etc.
    _CRED_GARBAGE_RE = re.compile(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f]"  # Control chars (except \t \n \r)
        r"|\\[rnt]"                          # Literal backslash-escaped control chars
        r"|['\");}{<>]{2,}"                  # Multiple special chars (garbled output)
        r"|^[^a-zA-Z0-9]"                    # Starts with non-alphanumeric
        r"|^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$"  # Bare IP address
    )

    def _normalize_services(self, result: ParsedOutput) -> None:
        """
        Normalize discovered services to canonical names.

        Collapses aliases (httpd → http, vsftpd → ftp, etc.) and
        validates port↔service consistency.
        """
        if not result.services:
            return

        normalized: Dict[int, str] = {}
        for port, svc in result.services.items():
            svc_lower = svc.strip().lower()
            canonical = self._SERVICE_ALIASES.get(svc_lower, svc_lower)
            normalized[port] = canonical

        result.services = normalized

    def _guard_credentials(self, result: ParsedOutput) -> None:
        """
        Filter out credential false positives.

        Rejects placeholder/example credentials, empty values,
        garbled strings with control characters, and obviously
        non-credential data (IPs, HTML fragments, etc.).
        """
        if not result.credentials:
            return

        clean: List[Dict[str, str]] = []
        for cred in result.credentials:
            username = cred.get("username", "").strip()
            password = cred.get("password", "").strip()

            # Skip empty
            if not username and not password:
                continue

            # Skip false positives
            combined = f"{username}:{password}"
            if self._CRED_FALSE_POSITIVES.search(combined):
                logger.debug("Credential false positive filtered: %s", username[:20])
                continue

            # Skip very short passwords (< 2 chars → likely parsing noise)
            if password and len(password) < 2:
                continue

            # P50: Skip garbled/non-printable values
            if password and self._CRED_GARBAGE_RE.search(password):
                logger.debug("Credential garbage filtered (password): %s", password[:30])
                continue
            if username and self._CRED_GARBAGE_RE.search(username):
                logger.debug("Credential garbage filtered (username): %s", username[:30])
                continue

            # P50: Skip if password is entirely non-alphanumeric (punctuation-only)
            if password and not any(c.isalnum() for c in password):
                logger.debug("Credential non-alnum filtered: %s", password[:30])
                continue

            # P50: Skip unreasonably long credentials (>128 chars = garbled output)
            if (password and len(password) > 128) or (username and len(username) > 64):
                logger.debug("Credential too long filtered: u=%d p=%d", len(username), len(password))
                continue

            clean.append(cred)

        result.credentials = clean

    def _sanitize_web_paths(self, result: ParsedOutput) -> None:
        """
        Sanitize discovered web paths.

        Removes junk paths (/, /index.html, etc.), deduplicates,
        and normalizes trailing slashes.
        """
        if not result.web_paths:
            return

        clean: List[str] = []
        seen: Set[str] = set()
        for path in result.web_paths:
            path = path.strip()
            if not path:
                continue

            # Normalize: remove trailing slash for comparison (except root /)
            norm = path.rstrip("/") if len(path) > 1 else path

            # Skip junk
            if norm in self._JUNK_WEB_PATHS or path in self._JUNK_WEB_PATHS:
                continue

            # Skip local filesystem paths
            if norm.startswith("/usr/") or norm.startswith("/etc/") or norm.startswith("/var/"):
                continue

            # Phase 42: Skip garbage paths from tool output (IPs, versions, status codes)
            if self._WEB_PATH_GARBAGE_RE.match(norm):
                continue

            # Phase 42: Skip paths that are too short (single char like / already caught)
            if len(norm) <= 1:
                continue

            # Dedup
            if norm in seen:
                continue
            seen.add(norm)
            clean.append(path)

        result.web_paths = clean
