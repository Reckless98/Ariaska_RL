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

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


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
    }
    
    def __init__(self, known_ports_path: Optional[str] = None):
        self._all_discovered_ports: Set[int] = set()
        self._all_discovered_services: Dict[int, str] = {}
        self._all_credentials: List[Dict[str, str]] = []
        self._all_vulns: List[str] = []
    
    def parse(self, command: str, output: str) -> ParsedOutput:
        """
        Parse command output into structured data.
        
        Args:
            command: The command that was executed
            output: Raw stdout from the command
            
        Returns:
            ParsedOutput with structured discoveries
        """
        if not command or not output:
            return ParsedOutput(command=command or "", error="Empty command or output")
        
        # Identify tool from command
        tool = self._identify_tool(command)
        
        # Get appropriate parser
        parser_name = self.TOOL_PARSERS.get(tool, "_parse_generic")
        parser_method = getattr(self, parser_name, self._parse_generic)
        
        try:
            result = parser_method(command, output)
            result.command = command
            result.tool = tool
            result.raw_output_length = len(output)
            
            # Update global discovery tracking
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
        
        result.success = len(result.artifacts) > 0
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
        cred_pattern = r'(?:password|passwd|pass)[\s:=]+(\S+)'
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
