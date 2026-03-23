#!/usr/bin/env python3
"""
generate_htb_data.py — Comprehensive HTB Training Data Generator

Generates massive, high-quality SFT + DPO training data for ariaska-cybersec3.
Covers every major attack vector seen in HTB easy/medium boxes.

Sources:
  1. 50+ attack chain templates covering all HTB vectors
  2. HTB walkthrough mining (12 .md files → structured examples)
  3. DPO preference pairs (correct vs common-mistake actions)

Output:
  - htb_sft_v5.jsonl: SFT data (microchain_fast_local + smart_mentor + phase_classifier)
  - htb_strategic_v5.jsonl: Strategic planning data (for cloud 70B alignment)
  - htb_dpo_v5.jsonl: DPO preference pairs (chosen vs rejected)
  - htb_output_parse_v5.jsonl: Output parsing training (teaches 4B to parse tool output)

Usage:
  python generate_htb_data.py                # Generate all data
  python generate_htb_data.py --sft-count 15000 --dpo-count 2500
  python generate_htb_data.py --stats        # Show stats only
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("ariaska.generate_htb_data")

# ═══════════════════════════════════════════════════════════════════════════════
# ATTACK CHAIN LIBRARY — 50+ chains covering every HTB vector
# ═══════════════════════════════════════════════════════════════════════════════
#
# Each chain: list of (phase, command_template_name, reasoning, expected_output_hint)
# command_template_name MUST exist in COMMAND_REGISTRY (279 commands verified)

# ── Web Attack Chains ────────────────────────────────────────────────────────

WEB_CHAINS = [
    {
        "name": "lfi_log_poison_rce",
        "category": "web",
        "difficulty": "easy",
        "description": "LFI via log poisoning to RCE",
        "steps": [
            ("RECON", "whatweb", "Fingerprint web stack to identify PHP/Apache"),
            ("ENUMERATION", "gobuster_dir", "Discover hidden directories and endpoints"),
            ("ENUMERATION", "ffuf_fuzz", "Fuzz parameters for file inclusion vulnerabilities"),
            ("EXPLOITATION", "lfi_etc_passwd", "Confirm LFI via /etc/passwd read"),
            ("EXPLOITATION", "lfi_log_poison", "Poison Apache access log with PHP payload"),
            ("EXPLOITATION", "revshell_bash", "Trigger reverse shell via poisoned log"),
        ],
    },
    {
        "name": "lfi_php_filter_creds",
        "category": "web",
        "difficulty": "easy",
        "description": "LFI via php://filter to read source code and extract creds",
        "steps": [
            ("RECON", "whatweb", "Identify PHP application"),
            ("ENUMERATION", "gobuster_dir", "Find PHP files and directories"),
            ("EXPLOITATION", "lfi_php_filter", "Use php://filter/convert.base64-encode to read source"),
            ("EXPLOITATION", "lfi_etc_passwd", "Read /etc/passwd for usernames"),
            ("EXPLOITATION", "ssh_login", "Login with extracted database credentials"),
        ],
    },
    {
        "name": "lfi_ssh_key_steal",
        "category": "web",
        "difficulty": "easy",
        "description": "LFI to steal SSH private key",
        "steps": [
            ("ENUMERATION", "gobuster_dir", "Find vulnerable endpoints"),
            ("EXPLOITATION", "lfi_etc_passwd", "Read /etc/passwd for home directories"),
            ("EXPLOITATION", "lfi_ssh_key", "Read /home/user/.ssh/id_rsa via LFI"),
            ("EXPLOITATION", "ssh_key_login", "Login with stolen SSH key"),
        ],
    },
    {
        "name": "sqli_union_to_shell",
        "category": "web",
        "difficulty": "easy",
        "description": "SQL injection → database dump → OS shell",
        "steps": [
            ("ENUMERATION", "gobuster_dir", "Find web application login/search pages"),
            ("ENUMERATION", "sqlmap_test", "Test for SQL injection vulnerabilities"),
            ("EXPLOITATION", "sqlmap_get", "Extract database contents via union-based SQLi"),
            ("EXPLOITATION", "sqlmap_shell", "Obtain OS shell via --os-shell"),
            ("POST_EXPLOITATION", "credential_dump", "Extract credentials from database"),
        ],
    },
    {
        "name": "sqli_post_login_bypass",
        "category": "web",
        "difficulty": "easy",
        "description": "POST-based SQLi to bypass authentication",
        "steps": [
            ("ENUMERATION", "gobuster_dir", "Discover login page"),
            ("ENUMERATION", "sqlmap_test", "Test login form for SQL injection"),
            ("EXPLOITATION", "sqlmap_post", "Extract admin credentials via POST-based SQLi"),
            ("EXPLOITATION", "curl_web_path", "Login as admin with extracted creds"),
        ],
    },
    {
        "name": "ssrf_cloud_metadata",
        "category": "web",
        "difficulty": "medium",
        "description": "SSRF to access cloud metadata and internal services",
        "steps": [
            ("ENUMERATION", "gobuster_dir", "Find web application endpoints"),
            ("ENUMERATION", "ffuf_fuzz", "Fuzz for SSRF-vulnerable parameters"),
            ("EXPLOITATION", "ssrf_localhost_scan", "Scan localhost services via SSRF"),
            ("EXPLOITATION", "ssrf_cloud_metadata", "Access cloud metadata for credentials"),
            ("EXPLOITATION", "ssrf_internal_admin", "Access internal admin panel via SSRF"),
        ],
    },
    {
        "name": "file_upload_webshell",
        "category": "web",
        "difficulty": "easy",
        "description": "File upload bypass → webshell → reverse shell",
        "steps": [
            ("ENUMERATION", "gobuster_dir", "Find upload functionality"),
            ("EXPLOITATION", "upload_php_double_ext", "Bypass filter with .php.jpg double extension"),
            ("EXPLOITATION", "webshell_cmd", "Execute commands via uploaded webshell"),
            ("EXPLOITATION", "revshell_bash", "Upgrade to interactive reverse shell"),
        ],
    },
    {
        "name": "file_upload_htaccess",
        "category": "web",
        "difficulty": "medium",
        "description": "Upload .htaccess to enable PHP execution in upload dir",
        "steps": [
            ("ENUMERATION", "gobuster_dir", "Find upload directory"),
            ("EXPLOITATION", "upload_htaccess", "Upload .htaccess to enable PHP execution"),
            ("EXPLOITATION", "upload_php_magic_bytes", "Upload PHP shell with magic bytes"),
            ("EXPLOITATION", "webshell_cmd", "Execute commands via webshell"),
            ("EXPLOITATION", "revshell_python", "Spawn reverse shell"),
        ],
    },
    {
        "name": "jwt_none_bypass",
        "category": "web",
        "difficulty": "medium",
        "description": "JWT none algorithm bypass for admin access",
        "steps": [
            ("RECON", "whatweb", "Identify JWT-based authentication"),
            ("EXPLOITATION", "jwt_none_attack", "Bypass JWT with none algorithm"),
            ("EXPLOITATION", "curl_web_path", "Access admin endpoints with forged JWT"),
        ],
    },
    {
        "name": "jwt_weak_secret",
        "category": "web",
        "difficulty": "medium",
        "description": "Crack JWT secret key and forge admin token",
        "steps": [
            ("RECON", "whatweb", "Identify JWT authentication"),
            ("EXPLOITATION", "jwt_crack_secret", "Crack JWT secret with wordlist"),
            ("EXPLOITATION", "curl_web_path", "Access admin panel with forged token"),
            ("EXPLOITATION", "revshell_bash", "Exploit admin functionality for shell"),
        ],
    },
    {
        "name": "ssti_jinja2_rce",
        "category": "web",
        "difficulty": "medium",
        "description": "Server-Side Template Injection (Jinja2) to RCE",
        "steps": [
            ("RECON", "whatweb", "Identify Python/Flask application"),
            ("ENUMERATION", "gobuster_dir", "Find input fields and endpoints"),
            ("EXPLOITATION", "ssti_detect_jinja2", "Confirm SSTI with {{7*7}} payload"),
            ("EXPLOITATION", "ssti_exploit_jinja2", "Exploit Jinja2 SSTI for RCE"),
            ("EXPLOITATION", "revshell_python", "Obtain reverse shell via SSTI"),
        ],
    },
    {
        "name": "ssti_twig_rce",
        "category": "web",
        "difficulty": "medium",
        "description": "SSTI in Twig template engine",
        "steps": [
            ("RECON", "whatweb", "Identify PHP application with Twig"),
            ("EXPLOITATION", "ssti_detect_twig", "Detect Twig SSTI vulnerability"),
            ("EXPLOITATION", "ssti_exploit_twig", "Exploit Twig SSTI for command execution"),
            ("EXPLOITATION", "revshell_bash", "Obtain reverse shell"),
        ],
    },
    {
        "name": "xxe_file_read",
        "category": "web",
        "difficulty": "medium",
        "description": "XML External Entity injection for file read",
        "steps": [
            ("ENUMERATION", "gobuster_dir", "Find XML upload/processing endpoints"),
            ("EXPLOITATION", "xxe_file_read", "Read /etc/passwd via XXE"),
            ("EXPLOITATION", "lfi_ssh_key", "Read SSH key via XXE file:// protocol"),
            ("EXPLOITATION", "ssh_key_login", "Login with extracted key"),
        ],
    },
    {
        "name": "cmd_injection_rce",
        "category": "web",
        "difficulty": "easy",
        "description": "OS command injection to reverse shell",
        "steps": [
            ("ENUMERATION", "gobuster_dir", "Find input functionality"),
            ("EXPLOITATION", "cmd_inject_semicolon", "Test command injection with semicolon"),
            ("EXPLOITATION", "cmd_inject_pipe", "Confirm injection with pipe operator"),
            ("EXPLOITATION", "revshell_bash", "Inject reverse shell payload"),
        ],
    },
    {
        "name": "nosqli_auth_bypass",
        "category": "web",
        "difficulty": "medium",
        "description": "NoSQL injection to bypass authentication",
        "steps": [
            ("RECON", "whatweb", "Identify Node.js/MongoDB application"),
            ("ENUMERATION", "gobuster_dir", "Find login endpoint"),
            ("EXPLOITATION", "nosqli_login_bypass", "Bypass login with NoSQL injection"),
            ("POST_EXPLOITATION", "credential_dump", "Extract user data from MongoDB"),
        ],
    },
    {
        "name": "deserialization_java",
        "category": "web",
        "difficulty": "medium",
        "description": "Java deserialization exploit for RCE",
        "steps": [
            ("RECON", "whatweb", "Identify Java application (Tomcat/Spring)"),
            ("ENUMERATION", "searchsploit", "Search for deserialization exploits"),
            ("EXPLOITATION", "ysoserial_java", "Generate deserialization payload"),
            ("EXPLOITATION", "revshell_bash", "Obtain reverse shell via deserialization"),
        ],
    },
    {
        "name": "idor_to_creds",
        "category": "web",
        "difficulty": "easy",
        "description": "IDOR to access other users' data (Cap-style)",
        "steps": [
            ("ENUMERATION", "gobuster_dir", "Discover web application structure"),
            ("ENUMERATION", "idor_curl_range", "Enumerate IDOR parameter range"),
            ("EXPLOITATION", "idor_download_file", "Download sensitive file via IDOR"),
            ("EXPLOITATION", "pcap_strings_extract", "Extract credentials from pcap/data"),
            ("EXPLOITATION", "ssh_login", "Login with extracted credentials"),
        ],
    },
    {
        "name": "wordpress_exploit",
        "category": "web",
        "difficulty": "easy",
        "description": "WordPress plugin exploit to shell",
        "steps": [
            ("RECON", "whatweb", "Identify WordPress installation"),
            ("ENUMERATION", "wpscan", "Enumerate WordPress plugins and users"),
            ("ENUMERATION", "searchsploit", "Find exploits for vulnerable plugins"),
            ("EXPLOITATION", "msfconsole_exploit", "Exploit vulnerable plugin for shell"),
        ],
    },
    {
        "name": "tomcat_war_deploy",
        "category": "web",
        "difficulty": "easy",
        "description": "Tomcat manager WAR deploy for shell",
        "steps": [
            ("RECON", "nmap_service_version", "Identify Tomcat version"),
            ("ENUMERATION", "tomcat_cred_test", "Test default/weak credentials"),
            ("EXPLOITATION", "msfvenom_payload", "Generate WAR reverse shell payload"),
            ("EXPLOITATION", "tomcat_war_deploy", "Deploy WAR via manager"),
            ("EXPLOITATION", "revshell_bash", "Trigger reverse shell from deployed WAR"),
        ],
    },
    {
        "name": "shellshock_cgi",
        "category": "web",
        "difficulty": "easy",
        "description": "Shellshock CGI exploit",
        "steps": [
            ("ENUMERATION", "gobuster_dir", "Find CGI scripts"),
            ("ENUMERATION", "nikto_scan", "Scan for Shellshock vulnerability"),
            ("EXPLOITATION", "shellshock_cgi", "Exploit Shellshock for command execution"),
            ("EXPLOITATION", "revshell_bash", "Obtain reverse shell via Shellshock"),
        ],
    },
    {
        "name": "drupal_rce",
        "category": "web",
        "difficulty": "easy",
        "description": "Drupalgeddon2 RCE",
        "steps": [
            ("RECON", "whatweb", "Identify Drupal CMS"),
            ("ENUMERATION", "droopescan", "Enumerate Drupal version and plugins"),
            ("EXPLOITATION", "drupalgeddon2", "Exploit Drupalgeddon2 for RCE"),
            ("EXPLOITATION", "revshell_bash", "Upgrade to interactive shell"),
        ],
    },
    {
        "name": "log4shell_exploit",
        "category": "web",
        "difficulty": "medium",
        "description": "Log4Shell (CVE-2021-44228) RCE",
        "steps": [
            ("RECON", "nmap_service_version", "Identify Java services"),
            ("EXPLOITATION", "log4shell_detect", "Test for Log4Shell vulnerability"),
            ("EXPLOITATION", "revshell_bash", "Obtain shell via JNDI callback"),
        ],
    },
    {
        "name": "rfi_to_shell",
        "category": "web",
        "difficulty": "medium",
        "description": "Remote File Inclusion to PHP shell",
        "steps": [
            ("ENUMERATION", "gobuster_dir", "Find PHP include endpoints"),
            ("EXPLOITATION", "rfi_php_shell", "Include remote PHP shell"),
            ("EXPLOITATION", "webshell_cmd", "Execute commands via RFI shell"),
            ("EXPLOITATION", "revshell_python", "Upgrade to reverse shell"),
        ],
    },
    {
        "name": "phpggc_deserialize",
        "category": "web",
        "difficulty": "medium",
        "description": "PHP deserialization via PHPGGC (Laravel)",
        "steps": [
            ("RECON", "whatweb", "Identify Laravel application"),
            ("EXPLOITATION", "phpggc_laravel", "Generate PHP deserialization payload"),
            ("EXPLOITATION", "revshell_bash", "Trigger reverse shell via deserialization"),
        ],
    },
]

# ── Network/Service Chains ────────────────────────────────────────────────────

SERVICE_CHAINS = [
    {
        "name": "ftp_anonymous_creds",
        "category": "service",
        "difficulty": "easy",
        "description": "FTP anonymous login → find credentials",
        "steps": [
            ("RECON", "nmap_service_version", "Discover FTP service and version"),
            ("ENUMERATION", "ftp_anonymous", "Login via anonymous FTP"),
            ("POST_EXPLOITATION", "credential_dump", "Extract credentials from FTP files"),
            ("EXPLOITATION", "ssh_login", "Login via SSH with found credentials"),
        ],
    },
    {
        "name": "smb_null_session_enum",
        "category": "service",
        "difficulty": "easy",
        "description": "SMB null session → share enum → credential extraction",
        "steps": [
            ("RECON", "nmap_service_version", "Identify SMB service"),
            ("ENUMERATION", "smbclient_null_list", "List shares via null session"),
            ("ENUMERATION", "smbmap_shares", "Map share permissions"),
            ("ENUMERATION", "enum4linux_full", "Full enumeration via null session"),
            ("EXPLOITATION", "smbclient_get_file", "Download sensitive files from shares"),
        ],
    },
    {
        "name": "smb_cred_reuse",
        "category": "service",
        "difficulty": "easy",
        "description": "SMB credential enumeration and reuse",
        "steps": [
            ("ENUMERATION", "enum4linux_full", "Enumerate SMB users and shares"),
            ("ENUMERATION", "smbclient_auth", "Access shares with found credentials"),
            ("EXPLOITATION", "smbclient_get_file", "Download config files/backups"),
            ("EXPLOITATION", "cred_reuse_ssh", "Reuse credentials on SSH"),
        ],
    },
    {
        "name": "snmp_enum_to_creds",
        "category": "service",
        "difficulty": "easy",
        "description": "SNMP community string → system enumeration",
        "steps": [
            ("RECON", "nmap_udp_scan", "Discover SNMP on UDP 161"),
            ("ENUMERATION", "onesixtyone", "Brute-force SNMP community strings"),
            ("ENUMERATION", "snmpwalk", "Walk SNMP tree for usernames/passwords"),
            ("EXPLOITATION", "ssh_login", "Login with extracted credentials"),
        ],
    },
    {
        "name": "redis_unauth_rce",
        "category": "service",
        "difficulty": "medium",
        "description": "Redis unauthenticated → SSH key plant",
        "steps": [
            ("RECON", "nmap_service_version", "Discover Redis on port 6379"),
            ("ENUMERATION", "redis_cli", "Connect to unauthenticated Redis"),
            ("EXPLOITATION", "ssh_key_plant", "Plant SSH key via Redis CONFIG SET"),
            ("EXPLOITATION", "ssh_key_login", "Login with planted SSH key"),
        ],
    },
    {
        "name": "nfs_mount_root",
        "category": "service",
        "difficulty": "easy",
        "description": "NFS no_root_squash → mount and read sensitive files",
        "steps": [
            ("RECON", "rpcinfo_check", "Check for NFS/RPC services"),
            ("ENUMERATION", "showmount", "List NFS exports"),
            ("EXPLOITATION", "nfs_mount", "Mount NFS share"),
            ("EXPLOITATION", "nfs_mount_root", "Exploit no_root_squash to read root files"),
        ],
    },
    {
        "name": "mysql_weak_creds",
        "category": "service",
        "difficulty": "easy",
        "description": "MySQL weak credentials to data extraction",
        "steps": [
            ("RECON", "nmap_service_version", "Identify MySQL service"),
            ("EXPLOITATION", "mysql_login", "Login with default/weak credentials"),
            ("POST_EXPLOITATION", "credential_dump", "Extract user credentials from database"),
            ("EXPLOITATION", "cred_reuse_ssh", "Reuse database credentials on SSH"),
        ],
    },
    {
        "name": "mssql_exec",
        "category": "service",
        "difficulty": "medium",
        "description": "MSSQL xp_cmdshell to RCE",
        "steps": [
            ("RECON", "nmap_service_version", "Identify MSSQL service"),
            ("EXPLOITATION", "mssql_login", "Login to MSSQL with found credentials"),
            ("EXPLOITATION", "revshell_powershell", "Execute via xp_cmdshell for reverse shell"),
        ],
    },
    {
        "name": "postgres_rce",
        "category": "service",
        "difficulty": "medium",
        "description": "PostgreSQL default creds to RCE",
        "steps": [
            ("RECON", "nmap_service_version", "Identify PostgreSQL on port 5432"),
            ("ENUMERATION", "psql_default_creds", "Test default PostgreSQL credentials"),
            ("EXPLOITATION", "psql_rce", "Execute system commands via PostgreSQL"),
            ("EXPLOITATION", "revshell_bash", "Obtain reverse shell"),
        ],
    },
    {
        "name": "vsftpd_backdoor",
        "category": "service",
        "difficulty": "easy",
        "description": "vsFTPd 2.3.4 backdoor exploit",
        "steps": [
            ("RECON", "nmap_service_version", "Identify vsFTPd 2.3.4"),
            ("EXPLOITATION", "vsftpd_exploit", "Exploit vsFTPd backdoor"),
            ("EXPLOITATION", "root_shell_confirm", "Confirm root shell access"),
        ],
    },
    {
        "name": "heartbleed_exploit",
        "category": "service",
        "difficulty": "easy",
        "description": "Heartbleed memory leak for credentials",
        "steps": [
            ("RECON", "nmap_vuln_scan", "Scan for Heartbleed vulnerability"),
            ("EXPLOITATION", "heartbleed_exploit", "Exploit Heartbleed to leak memory"),
            ("POST_EXPLOITATION", "credential_dump", "Extract credentials from leaked memory"),
            ("EXPLOITATION", "ssh_login", "Login with leaked credentials"),
        ],
    },
    {
        "name": "samba_usermap",
        "category": "service",
        "difficulty": "easy",
        "description": "Samba username map script RCE",
        "steps": [
            ("RECON", "nmap_service_version", "Identify Samba version"),
            ("ENUMERATION", "searchsploit", "Find Samba exploit"),
            ("EXPLOITATION", "samba_usermap_exploit", "Exploit Samba username map script"),
        ],
    },
    {
        "name": "unreal_ircd_backdoor",
        "category": "service",
        "difficulty": "easy",
        "description": "UnrealIRCd 3.2.8.1 backdoor",
        "steps": [
            ("RECON", "nmap_service_version", "Identify UnrealIRCd on port 6667"),
            ("EXPLOITATION", "unrealircd_exploit", "Exploit UnrealIRCd backdoor"),
        ],
    },
    {
        "name": "winrm_cred_spray",
        "category": "service",
        "difficulty": "medium",
        "description": "WinRM credential spray and execution",
        "steps": [
            ("RECON", "nmap_service_version", "Identify WinRM on port 5985"),
            ("EXPLOITATION", "crackmapexec_winrm", "Spray credentials against WinRM"),
            ("EXPLOITATION", "evil_winrm", "Login via Evil-WinRM with valid credentials"),
        ],
    },
]

# ── Active Directory Chains ──────────────────────────────────────────────────

AD_CHAINS = [
    {
        "name": "kerberoast_to_dc",
        "category": "ad",
        "difficulty": "medium",
        "description": "Kerberoasting → crack → DCSync → domain admin",
        "steps": [
            ("RECON", "nmap_service_version", "Identify AD services (88, 389, 636)"),
            ("ENUMERATION", "ldapsearch_base", "Enumerate LDAP base DN and domain info"),
            ("ENUMERATION", "ldapsearch_users", "Extract domain user accounts"),
            ("EXPLOITATION", "impacket_GetUserSPNs", "Kerberoast — extract TGS hashes"),
            ("EXPLOITATION", "hashcat_krb5", "Crack Kerberos TGS hashes offline"),
            ("EXPLOITATION", "evil_winrm", "Login with cracked credentials"),
            ("POST_EXPLOITATION", "mimikatz_dcsync", "DCSync to extract all domain hashes"),
            ("LATERAL_MOVEMENT", "impacket_psexec", "PsExec to domain controller"),
        ],
    },
    {
        "name": "asreproast_lateral",
        "category": "ad",
        "difficulty": "medium",
        "description": "AS-REP Roasting → lateral movement",
        "steps": [
            ("RECON", "nmap_service_version", "Scan for Kerberos and LDAP ports"),
            ("EXPLOITATION", "impacket_GetNPUsers", "AS-REP Roast — no preauth accounts"),
            ("EXPLOITATION", "hashcat_krb5", "Crack AS-REP hashes"),
            ("EXPLOITATION", "cred_reuse_ssh", "Test credential reuse across services"),
            ("LATERAL_MOVEMENT", "crackmapexec_pth", "Pass-the-hash lateral movement"),
            ("POST_EXPLOITATION", "impacket_secretsdump", "Dump secrets from compromised host"),
        ],
    },
    {
        "name": "bloodhound_shortest_path",
        "category": "ad",
        "difficulty": "medium",
        "description": "BloodHound enumeration → find shortest path to DA",
        "steps": [
            ("ENUMERATION", "bloodhound_python", "Collect AD data with BloodHound"),
            ("ENUMERATION", "ldapsearch_users", "Enumerate users and groups"),
            ("EXPLOITATION", "impacket_GetUserSPNs", "Kerberoast SPNs on attack path"),
            ("EXPLOITATION", "hashcat_krb5", "Crack service account hashes"),
            ("LATERAL_MOVEMENT", "impacket_pth_psexec", "PSExec with cracked creds"),
            ("POST_EXPLOITATION", "mimikatz_dcsync", "DCSync for domain dominance"),
        ],
    },
    {
        "name": "adcs_esc1_abuse",
        "category": "ad",
        "difficulty": "medium",
        "description": "ADCS ESC1 — request cert as admin",
        "steps": [
            ("ENUMERATION", "certipy_find", "Enumerate ADCS certificate templates"),
            ("EXPLOITATION", "certipy_req", "Request certificate for admin user"),
            ("EXPLOITATION", "evil_winrm", "Authenticate with forged certificate"),
            ("POST_EXPLOITATION", "mimikatz_dcsync", "DCSync with admin privileges"),
        ],
    },
    {
        "name": "gpp_password_decrypt",
        "category": "ad",
        "difficulty": "easy",
        "description": "Group Policy Preferences password decryption",
        "steps": [
            ("ENUMERATION", "smbclient_null_list", "Enumerate SYSVOL share"),
            ("ENUMERATION", "smbclient_auth", "Browse Group Policy Objects"),
            ("EXPLOITATION", "smbclient_get_file", "Download Groups.xml from GPO"),
            ("EXPLOITATION", "gpp_decrypt", "Decrypt GPP cPassword"),
            ("EXPLOITATION", "evil_winrm", "Login with decrypted credentials"),
        ],
    },
    {
        "name": "ntlm_relay_attack",
        "category": "ad",
        "difficulty": "medium",
        "description": "NTLM relay to gain access",
        "steps": [
            ("ENUMERATION", "cme_smb_shares", "Enumerate SMB signing status"),
            ("LATERAL_MOVEMENT", "responder", "Capture NTLM hashes with Responder"),
            ("LATERAL_MOVEMENT", "ntlmrelayx", "Relay NTLM authentication"),
            ("POST_EXPLOITATION", "impacket_secretsdump", "Dump secrets from relayed target"),
        ],
    },
]

# ── Linux Privilege Escalation Chains ────────────────────────────────────────

LINUX_PRIVESC_CHAINS = [
    {
        "name": "suid_gtfobins",
        "category": "privesc_linux",
        "difficulty": "easy",
        "description": "Find SUID binary → GTFOBins escalation",
        "steps": [
            ("PRIVILEGE_ESCALATION", "find_suid", "Find all SUID binaries"),
            ("PRIVILEGE_ESCALATION", "sudo_list", "Check sudo permissions"),
            ("PRIVILEGE_ESCALATION", "linpeas", "Run LinPEAS for comprehensive privesc enum"),
        ],
    },
    {
        "name": "sudo_misconfigure",
        "category": "privesc_linux",
        "difficulty": "easy",
        "description": "Sudo misconfiguration exploit (NOPASSWD)",
        "steps": [
            ("PRIVILEGE_ESCALATION", "sudo_list", "List sudo permissions with sudo -l"),
            ("PRIVILEGE_ESCALATION", "privesc_sudo_l", "Exploit sudo NOPASSWD entry"),
        ],
    },
    {
        "name": "capability_privesc",
        "category": "privesc_linux",
        "difficulty": "easy",
        "description": "Linux capabilities privesc (Cap-style: cap_setuid on Python)",
        "steps": [
            ("PRIVILEGE_ESCALATION", "find_capabilities", "Find binaries with capabilities"),
            ("PRIVILEGE_ESCALATION", "privesc_getcap", "Identify cap_setuid capability"),
            ("PRIVILEGE_ESCALATION", "privesc_cap_setuid_python", "Exploit Python cap_setuid for root"),
        ],
    },
    {
        "name": "docker_group_escape",
        "category": "privesc_linux",
        "difficulty": "easy",
        "description": "Docker group membership → root shell",
        "steps": [
            ("PRIVILEGE_ESCALATION", "linpeas", "Identify docker group membership"),
            ("PRIVILEGE_ESCALATION", "docker_privesc", "Mount host filesystem via Docker"),
        ],
    },
    {
        "name": "docker_socket_escape",
        "category": "privesc_linux",
        "difficulty": "medium",
        "description": "Docker socket access → container escape",
        "steps": [
            ("PRIVILEGE_ESCALATION", "linpeas", "Find exposed Docker socket"),
            ("PRIVILEGE_ESCALATION", "docker_sock_escape", "Escape container via Docker socket"),
        ],
    },
    {
        "name": "lxd_escape",
        "category": "privesc_linux",
        "difficulty": "medium",
        "description": "LXD group → mount host filesystem",
        "steps": [
            ("PRIVILEGE_ESCALATION", "linpeas", "Identify LXD group membership"),
            ("PRIVILEGE_ESCALATION", "lxd_privesc", "Exploit LXD for root"),
        ],
    },
    {
        "name": "cron_injection",
        "category": "privesc_linux",
        "difficulty": "easy",
        "description": "Writable cron job → inject payload for root",
        "steps": [
            ("PRIVILEGE_ESCALATION", "cron_check", "Enumerate cron jobs"),
            ("PRIVILEGE_ESCALATION", "pspy", "Monitor processes for hidden crons"),
            ("POST_EXPLOITATION", "cron_backdoor", "Inject payload into writable cron script"),
        ],
    },
    {
        "name": "kernel_exploit",
        "category": "privesc_linux",
        "difficulty": "medium",
        "description": "Kernel version → known exploit (DirtyPipe, DirtyCow)",
        "steps": [
            ("PRIVILEGE_ESCALATION", "kernel_exploit_check", "Check kernel version"),
            ("ENUMERATION", "searchsploit", "Search for kernel exploits"),
            ("PRIVILEGE_ESCALATION", "linpeas", "Confirm vulnerable kernel"),
            ("EXPLOITATION", "msfconsole_exploit", "Run kernel exploit for root"),
        ],
    },
    {
        "name": "writable_passwd",
        "category": "privesc_linux",
        "difficulty": "easy",
        "description": "Writable /etc/passwd → add root user",
        "steps": [
            ("PRIVILEGE_ESCALATION", "find_writable_etc", "Find writable files in /etc"),
            ("PRIVILEGE_ESCALATION", "writable_etc_passwd", "Add root user to /etc/passwd"),
        ],
    },
    {
        "name": "nfs_no_root_squash",
        "category": "privesc_linux",
        "difficulty": "easy",
        "description": "NFS no_root_squash → mount and create SUID binary",
        "steps": [
            ("ENUMERATION", "showmount_enum", "Check NFS exports for no_root_squash"),
            ("EXPLOITATION", "nfs_mount_root", "Mount share and create SUID binary"),
        ],
    },
    {
        "name": "erlang_cookie_rce",
        "category": "privesc_linux",
        "difficulty": "medium",
        "description": "Erlang cookie extraction → RCE via distributed Erlang",
        "steps": [
            ("PRIVILEGE_ESCALATION", "erlang_cookie_extract", "Extract .erlang.cookie"),
            ("PRIVILEGE_ESCALATION", "erlang_otp_rce", "RCE via Erlang distributed protocol"),
        ],
    },
    {
        "name": "sgid_find",
        "category": "privesc_linux",
        "difficulty": "easy",
        "description": "SGID binary exploitation",
        "steps": [
            ("PRIVILEGE_ESCALATION", "find_sgid", "Find SGID binaries"),
            ("PRIVILEGE_ESCALATION", "linpeas", "Cross-reference with GTFOBins"),
        ],
    },
]

# ── Windows Privilege Escalation Chains ──────────────────────────────────────

WINDOWS_PRIVESC_CHAINS = [
    {
        "name": "seimpersonate_potato",
        "category": "privesc_windows",
        "difficulty": "medium",
        "description": "SeImpersonate → Potato attack for SYSTEM",
        "steps": [
            ("PRIVILEGE_ESCALATION", "whoami_all", "Check privileges — look for SeImpersonate"),
            ("PRIVILEGE_ESCALATION", "systeminfo", "Get OS version for Potato variant"),
            ("EXPLOITATION", "msfconsole_exploit", "Run Potato attack for SYSTEM shell"),
        ],
    },
    {
        "name": "winpeas_enum",
        "category": "privesc_windows",
        "difficulty": "easy",
        "description": "WinPEAS comprehensive enumeration → escalation",
        "steps": [
            ("PRIVILEGE_ESCALATION", "winpeas", "Run WinPEAS for comprehensive enumeration"),
            ("PRIVILEGE_ESCALATION", "powerup", "Run PowerUp for additional checks"),
            ("PRIVILEGE_ESCALATION", "accesschk_services", "Check service permissions"),
        ],
    },
    {
        "name": "windows_exploit_suggest",
        "category": "privesc_windows",
        "difficulty": "medium",
        "description": "Windows Exploit Suggester → known kernel exploit",
        "steps": [
            ("PRIVILEGE_ESCALATION", "systeminfo", "Collect systeminfo for exploit suggester"),
            ("PRIVILEGE_ESCALATION", "windows_exploit_suggester", "Find applicable kernel exploits"),
            ("EXPLOITATION", "msfconsole_exploit", "Run suggested exploit"),
        ],
    },
]

# ── Pivoting/Lateral Movement Chains ────────────────────────────────────────

LATERAL_CHAINS = [
    {
        "name": "chisel_pivot",
        "category": "lateral",
        "difficulty": "medium",
        "description": "Chisel tunnel → pivot to internal network",
        "steps": [
            ("LATERAL_MOVEMENT", "chisel_server", "Start Chisel server on attacker"),
            ("LATERAL_MOVEMENT", "chisel_client", "Connect Chisel client from target"),
            ("LATERAL_MOVEMENT", "nmap_pivot", "Scan internal network via tunnel"),
            ("LATERAL_MOVEMENT", "proxychains_scan", "Enumerate internal services"),
        ],
    },
    {
        "name": "ssh_tunnel_pivot",
        "category": "lateral",
        "difficulty": "medium",
        "description": "SSH tunnel to reach internal services",
        "steps": [
            ("LATERAL_MOVEMENT", "ssh_tunnel_local", "Create local SSH port forward"),
            ("LATERAL_MOVEMENT", "ssh_tunnel_dynamic", "Create SOCKS proxy via SSH"),
            ("LATERAL_MOVEMENT", "pivot_scan", "Scan internal hosts via tunnel"),
        ],
    },
    {
        "name": "pth_lateral",
        "category": "lateral",
        "difficulty": "medium",
        "description": "Pass-the-hash lateral movement across domain",
        "steps": [
            ("POST_EXPLOITATION", "mimikatz_logonpasswords", "Dump NTLM hashes from memory"),
            ("LATERAL_MOVEMENT", "crackmapexec_pth", "Spray hash across subnet"),
            ("LATERAL_MOVEMENT", "impacket_pth_psexec", "PsExec with NTLM hash"),
            ("POST_EXPLOITATION", "impacket_secretsdump", "Dump secrets from new target"),
        ],
    },
]

# ── Full Box Chains (realistic end-to-end HTB box patterns) ─────────────────

FULL_BOX_CHAINS = [
    {
        "name": "htb_easy_linux_web",
        "category": "full_box",
        "difficulty": "easy",
        "description": "Typical easy Linux HTB: web vuln → user shell → privesc",
        "steps": [
            ("RECON", "nmap_quick_scan", "Quick port scan to identify services"),
            ("RECON", "nmap_service_version", "Detailed version scan on open ports"),
            ("ENUMERATION", "gobuster_dir", "Enumerate web directories"),
            ("ENUMERATION", "whatweb", "Fingerprint web technology"),
            ("EXPLOITATION", "cmd_inject_semicolon", "Exploit command injection for user shell"),
            ("EXPLOITATION", "tty_stabilize", "Stabilize shell with Python PTY"),
            ("PRIVILEGE_ESCALATION", "sudo_list", "Check sudo permissions"),
            ("PRIVILEGE_ESCALATION", "find_suid", "Find SUID binaries"),
            ("POST_EXPLOITATION", "read_user_flag", "Read user.txt flag"),
            ("POST_EXPLOITATION", "read_root_flag", "Read root.txt flag"),
        ],
    },
    {
        "name": "htb_easy_linux_service",
        "category": "full_box",
        "difficulty": "easy",
        "description": "Easy Linux HTB: service exploit → privesc",
        "steps": [
            ("RECON", "nmap_quick_scan", "Initial port scan"),
            ("RECON", "nmap_service_version", "Version detection on all ports"),
            ("ENUMERATION", "searchsploit", "Search for service exploits"),
            ("EXPLOITATION", "msfconsole_exploit", "Exploit vulnerable service"),
            ("EXPLOITATION", "tty_stabilize", "Stabilize reverse shell"),
            ("PRIVILEGE_ESCALATION", "linpeas", "Run LinPEAS"),
            ("PRIVILEGE_ESCALATION", "find_capabilities", "Check file capabilities"),
            ("POST_EXPLOITATION", "read_user_flag", "Read user.txt"),
            ("POST_EXPLOITATION", "read_root_flag", "Read root.txt"),
        ],
    },
    {
        "name": "htb_medium_linux_web_pivot",
        "category": "full_box",
        "difficulty": "medium",
        "description": "Medium Linux HTB: web → user → internal pivot → root",
        "steps": [
            ("RECON", "nmap_full_tcp", "Full TCP port scan"),
            ("RECON", "nmap_service_version", "Service version detection"),
            ("ENUMERATION", "gobuster_dir", "Web directory enumeration"),
            ("ENUMERATION", "gobuster_vhost", "Virtual host discovery"),
            ("ENUMERATION", "ffuf_fuzz", "Parameter fuzzing for hidden inputs"),
            ("EXPLOITATION", "ssti_detect_jinja2", "Test for SSTI vulnerability"),
            ("EXPLOITATION", "ssti_exploit_jinja2", "Exploit SSTI for command execution"),
            ("EXPLOITATION", "revshell_python", "Reverse shell as web user"),
            ("PRIVILEGE_ESCALATION", "sudo_list", "Check sudo permissions"),
            ("PRIVILEGE_ESCALATION", "pspy", "Monitor processes for privesc vectors"),
            ("PRIVILEGE_ESCALATION", "cron_check", "Check cron jobs for writable scripts"),
            ("POST_EXPLOITATION", "read_root_flag", "Read root.txt after privilege escalation"),
        ],
    },
    {
        "name": "htb_easy_windows_ad",
        "category": "full_box",
        "difficulty": "easy",
        "description": "Easy Windows HTB: SMB → AD enum → admin",
        "steps": [
            ("RECON", "nmap_quick_scan", "Port scan — look for 88, 389, 445"),
            ("RECON", "nmap_service_version", "Version scan including SMB"),
            ("ENUMERATION", "smbclient_null_list", "Check for null session access"),
            ("ENUMERATION", "enum4linux_full", "Full SMB/AD enumeration"),
            ("ENUMERATION", "ldapsearch_users", "LDAP user enumeration"),
            ("EXPLOITATION", "impacket_GetNPUsers", "AS-REP roast no-preauth users"),
            ("EXPLOITATION", "hashcat_krb5", "Crack AS-REP hashes"),
            ("EXPLOITATION", "evil_winrm", "Login with cracked credentials"),
            ("PRIVILEGE_ESCALATION", "winpeas", "Run WinPEAS for privesc"),
            ("POST_EXPLOITATION", "read_user_flag", "Read user.txt"),
            ("POST_EXPLOITATION", "read_root_flag", "Read root.txt"),
        ],
    },
]

# ── Real HTB Box Chains (based on actual retired boxes) ───────────────────────
# These encode REAL attack paths from known HTB/THM boxes.
# The reasoning is realistic and teaches the model actual pentester logic.

REAL_HTB_BOXES = [
    # ── Easy Linux ─────────────────────────────────────────────────────────
    {
        "name": "htb_lame",
        "category": "full_box",
        "difficulty": "easy",
        "description": "Lame: Samba 3.0.20 username map script to root (CVE-2007-2447)",
        "box_info": {"os": "Linux", "services": "ftp/21 ssh/22 smb/139,445 distccd/3632"},
        "steps": [
            ("RECON", "nmap_quick_scan", "Quick scan shows 21,22,139,445,3632 — classic Linux with SMB"),
            ("RECON", "nmap_service_version", "Version scan: vsftpd 2.3.4, OpenSSH 4.7p1, Samba 3.0.20 — all old, Samba version is exploitable"),
            ("ENUMERATION", "smbclient_null_list", "Null session lists shares: tmp (read/write), opt, IPC$ — tmp share is writable"),
            ("ENUMERATION", "searchsploit", "searchsploit Samba 3.0.20 finds username map script RCE (CVE-2007-2447)"),
            ("EXPLOITATION", "samba_usermap_exploit", "Exploit Samba username map script — inject backtick command in username field for root shell"),
            ("POST_EXPLOITATION", "read_root_flag", "Already root — read both user.txt and root.txt"),
        ],
    },
    {
        "name": "htb_blue",
        "category": "full_box",
        "difficulty": "easy",
        "description": "Blue: EternalBlue MS17-010 to SYSTEM (Windows SMB)",
        "box_info": {"os": "Windows", "services": "msrpc/135 netbios/139 smb/445 rdp/49152-49157"},
        "steps": [
            ("RECON", "nmap_quick_scan", "Quick scan: 135,139,445 — Windows box, SMB open"),
            ("RECON", "nmap_vuln_scan", "Vuln scan confirms ms17-010 (EternalBlue) — critical RCE in SMBv1"),
            ("EXPLOITATION", "msfconsole_exploit", "use exploit/windows/smb/ms17_010_eternalblue — target Windows 7 SP1 gets SYSTEM shell"),
            ("POST_EXPLOITATION", "read_root_flag", "Already SYSTEM — read both flags from user desktop and administrator desktop"),
        ],
    },
    {
        "name": "htb_jerry",
        "category": "full_box",
        "difficulty": "easy",
        "description": "Jerry: Tomcat default creds then WAR deploy for SYSTEM",
        "box_info": {"os": "Windows", "services": "http/8080(Tomcat)"},
        "steps": [
            ("RECON", "nmap_service_version", "Only 8080 open — Apache Tomcat 7.0.88, manager interface exposed"),
            ("ENUMERATION", "tomcat_cred_test", "Test default creds: tomcat/s3cret works on /manager/html — weak credentials"),
            ("EXPLOITATION", "msfvenom_payload", "msfvenom -p java/jsp_shell_reverse_tcp generates malicious WAR file"),
            ("EXPLOITATION", "tomcat_war_deploy", "Deploy WAR via Tomcat manager — auto-deploys as new context path"),
            ("POST_EXPLOITATION", "read_root_flag", "Running as SYSTEM (Tomcat runs as SYSTEM) — read both flags from same file on desktop"),
        ],
    },
    {
        "name": "htb_bashed",
        "category": "full_box",
        "difficulty": "easy",
        "description": "Bashed: phpbash webshell already on server then sudo scriptmanager then cron root",
        "box_info": {"os": "Linux", "services": "http/80(Apache)"},
        "steps": [
            ("RECON", "nmap_quick_scan", "Only port 80 — Apache 2.4.18 on Ubuntu"),
            ("ENUMERATION", "gobuster_dir", "Discover /dev/ directory containing phpbash.php — an existing webshell"),
            ("EXPLOITATION", "webshell_cmd", "Use phpbash.php webshell to execute commands as www-data"),
            ("EXPLOITATION", "revshell_python", "Upgrade to proper reverse shell via Python"),
            ("PRIVILEGE_ESCALATION", "sudo_list", "sudo -l shows www-data can run anything as scriptmanager (NOPASSWD)"),
            ("POST_EXPLOITATION", "cron_backdoor", "Cron runs scripts/*.py as root every minute — write Python reverse shell there"),
        ],
    },
    {
        "name": "htb_shocker",
        "category": "full_box",
        "difficulty": "easy",
        "description": "Shocker: Shellshock CGI then user shell then sudo perl root",
        "box_info": {"os": "Linux", "services": "http/80(Apache) ssh/2222"},
        "steps": [
            ("RECON", "nmap_service_version", "Port 80 (Apache 2.4.18) and 2222 (OpenSSH 7.2p2) — nonstandard SSH port"),
            ("ENUMERATION", "gobuster_dir", "Enumerate /cgi-bin/ directory — find user.sh script"),
            ("EXPLOITATION", "shellshock_cgi", "Shellshock on /cgi-bin/user.sh — inject via User-Agent header for RCE"),
            ("EXPLOITATION", "revshell_bash", "Reverse shell as shelly user"),
            ("PRIVILEGE_ESCALATION", "sudo_list", "sudo -l shows (root) NOPASSWD: /usr/bin/perl"),
            ("PRIVILEGE_ESCALATION", "privesc_sudo_l", "perl -e exec /bin/bash with sudo gives instant root"),
        ],
    },
    {
        "name": "htb_nibbles",
        "category": "full_box",
        "difficulty": "easy",
        "description": "Nibbles: Nibbleblog file upload then sudo personal script",
        "box_info": {"os": "Linux", "services": "http/80 ssh/22"},
        "steps": [
            ("RECON", "nmap_service_version", "SSH 22 and HTTP 80 — Apache 2.4.18"),
            ("ENUMERATION", "gobuster_dir", "Source code HTML comment reveals /nibbleblog/ path"),
            ("ENUMERATION", "wpscan", "Nibbleblog admin panel at /nibbleblog/admin.php — guess admin/nibbles"),
            ("EXPLOITATION", "upload_php_double_ext", "Upload PHP shell via My Image plugin (CVE-2015-6967) — arbitrary file upload"),
            ("EXPLOITATION", "revshell_bash", "Trigger uploaded shell for reverse connection"),
            ("PRIVILEGE_ESCALATION", "sudo_list", "sudo -l: (root) NOPASSWD: /home/nibbles/personal/stuff/monitor.sh — file doesnt exist"),
            ("POST_EXPLOITATION", "cron_backdoor", "Create monitor.sh with reverse shell payload then sudo it for root"),
        ],
    },
    {
        "name": "htb_valentine",
        "category": "full_box",
        "difficulty": "easy",
        "description": "Valentine: Heartbleed then hex-encoded SSH key then tmux root session",
        "box_info": {"os": "Linux", "services": "ssh/22 http/80 https/443"},
        "steps": [
            ("RECON", "nmap_vuln_scan", "Vuln scan finds Heartbleed (CVE-2014-0160) on port 443"),
            ("EXPLOITATION", "heartbleed_exploit", "Heartbleed memory leak reveals base64 string — looks like a passphrase"),
            ("ENUMERATION", "gobuster_dir", "Find /dev/ directory with notes.txt mentioning encode and hype_key file"),
            ("EXPLOITATION", "curl_web_path", "Download /dev/hype_key — hex-encoded RSA private key, decode it"),
            ("EXPLOITATION", "ssh_key_login", "SSH as hype with decoded key plus heartbleed passphrase"),
            ("PRIVILEGE_ESCALATION", "linpeas", "Find root tmux session at /.devs/dev_sess — tmux attach for root"),
        ],
    },
    {
        "name": "htb_cap",
        "category": "full_box",
        "difficulty": "easy",
        "description": "Cap: IDOR to pcap with plaintext creds then Python cap_setuid root",
        "box_info": {"os": "Linux", "services": "ftp/21 ssh/22 http/80(Gunicorn)"},
        "steps": [
            ("RECON", "nmap_service_version", "Ports 21(FTP), 22(SSH), 80(HTTP gunicorn) — Flask web app"),
            ("ENUMERATION", "gobuster_dir", "Web app at /data/N generates pcap captures — IDOR in the N parameter"),
            ("EXPLOITATION", "idor_curl_range", "Enumerate /data/0, /data/1, ... — find pcap with actual traffic"),
            ("EXPLOITATION", "pcap_strings_extract", "Download and analyze pcap — plaintext FTP creds nathan:Buck3tH4TF0RM3!"),
            ("EXPLOITATION", "ssh_login", "SSH as nathan with extracted FTP credentials — password reuse"),
            ("PRIVILEGE_ESCALATION", "find_capabilities", "getcap shows /usr/bin/python3.8 has cap_setuid+ep"),
            ("PRIVILEGE_ESCALATION", "privesc_cap_setuid_python", "Python cap_setuid: setuid(0) then spawn bash for root"),
        ],
    },
    {
        "name": "htb_mirai",
        "category": "full_box",
        "difficulty": "easy",
        "description": "Mirai: Pi-hole default creds then sudo root then USB backup for root flag",
        "box_info": {"os": "Linux", "services": "ssh/22 dns/53 http/80(lighttpd) upnp/1099 http/32400(Plex)"},
        "steps": [
            ("RECON", "nmap_service_version", "Multiple services — 80 shows lighttpd, Pi-hole dashboard detected"),
            ("ENUMERATION", "gobuster_dir", "Find /admin (Pi-hole admin) — default Raspberry Pi installation"),
            ("EXPLOITATION", "ssh_login", "Default Raspberry Pi creds pi:raspberry work on SSH"),
            ("PRIVILEGE_ESCALATION", "sudo_list", "sudo -l: pi can run ALL commands — sudo su for instant root"),
            ("POST_EXPLOITATION", "read_root_flag", "root.txt says flag deleted check USB — find /media/usbstick, strings the device for flag"),
        ],
    },
    {
        "name": "htb_blocky",
        "category": "full_box",
        "difficulty": "easy",
        "description": "Blocky: WordPress plus exposed Java plugin then decompile for DB creds then sudo root",
        "box_info": {"os": "Linux", "services": "ssh/22 http/80(WordPress) ftp/21 minecraft/25565"},
        "steps": [
            ("RECON", "nmap_service_version", "SSH/FTP/HTTP/25565 — Minecraft server with WordPress site"),
            ("ENUMERATION", "wpscan", "WPScan finds user notch — WordPress user enumeration"),
            ("ENUMERATION", "gobuster_dir", "Discover /plugins/ directory with BlockyCore.jar and griefprevention.jar"),
            ("EXPLOITATION", "curl_web_path", "Download BlockyCore.jar — decompile with jd-gui reveals SQL root password"),
            ("EXPLOITATION", "ssh_login", "SSH as notch with the SQL password — password reuse across services"),
            ("PRIVILEGE_ESCALATION", "sudo_list", "sudo -l: notch can run ALL commands as root then sudo su"),
        ],
    },
    # ── Easy/Medium Windows ────────────────────────────────────────────────
    {
        "name": "htb_devel",
        "category": "full_box",
        "difficulty": "easy",
        "description": "Devel: FTP anonymous write to IIS webroot then aspx shell then kernel exploit",
        "box_info": {"os": "Windows", "services": "ftp/21 http/80(IIS7.5)"},
        "steps": [
            ("RECON", "nmap_service_version", "FTP 21 (anonymous allowed, IIS root) and HTTP 80 (IIS 7.5) — FTP maps to web root"),
            ("ENUMERATION", "ftp_anonymous", "Anonymous FTP login — can see iisstart.htm, confirming its the web root"),
            ("EXPLOITATION", "msfvenom_payload", "Generate aspx reverse shell: msfvenom -p windows/meterpreter/reverse_tcp -f aspx"),
            ("EXPLOITATION", "upload_php_double_ext", "Upload aspx shell via FTP to web root — browse to it for execution"),
            ("PRIVILEGE_ESCALATION", "systeminfo", "systeminfo shows Windows 7 Enterprise 6.1.7600 — unpatched"),
            ("EXPLOITATION", "msfconsole_exploit", "Use local exploit suggester then MS10-015 kitrap0d for SYSTEM"),
        ],
    },
    {
        "name": "htb_grandpa",
        "category": "full_box",
        "difficulty": "easy",
        "description": "Grandpa: IIS 6.0 WebDAV buffer overflow then churrasco token impersonation",
        "box_info": {"os": "Windows", "services": "http/80(IIS6.0)"},
        "steps": [
            ("RECON", "nmap_service_version", "Only port 80 — IIS 6.0, extremely old (Windows Server 2003)"),
            ("EXPLOITATION", "msfconsole_exploit", "exploit/windows/iis/iis_webdav_scstoragepathfromurl for initial shell as NETWORK SERVICE"),
            ("PRIVILEGE_ESCALATION", "whoami_all", "Running as NETWORK SERVICE with SeImpersonatePrivilege"),
            ("EXPLOITATION", "msfconsole_exploit", "Migrate to stable process then churrasco.exe token kidnapping for SYSTEM"),
        ],
    },
    {
        "name": "htb_optimum",
        "category": "full_box",
        "difficulty": "easy",
        "description": "Optimum: HFS 2.3 RCE then MS16-032 kernel exploit",
        "box_info": {"os": "Windows", "services": "http/80(HFS)"},
        "steps": [
            ("RECON", "nmap_service_version", "Only port 80 — HttpFileServer 2.3 (Rejetto HFS)"),
            ("ENUMERATION", "searchsploit", "searchsploit HFS 2.3 finds CVE-2014-6287 RCE via null byte"),
            ("EXPLOITATION", "msfconsole_exploit", "exploit/windows/http/rejetto_hfs_exec gives user shell as kostas"),
            ("PRIVILEGE_ESCALATION", "systeminfo", "Windows Server 2012 R2 — check for MS16-032 secondary logon handle"),
            ("EXPLOITATION", "msfconsole_exploit", "MS16-032 gives SYSTEM shell"),
        ],
    },
    # ── Medium Linux ───────────────────────────────────────────────────────
    {
        "name": "htb_friendzone",
        "category": "full_box",
        "difficulty": "easy",
        "description": "FriendZone: DNS zone transfer then SMB creds then LFI then Python library hijack cron",
        "box_info": {"os": "Linux", "services": "ftp/21 ssh/22 dns/53 http/80 smb/139,445 https/443"},
        "steps": [
            ("RECON", "nmap_service_version", "Many services — DNS, HTTP, HTTPS, SMB all open"),
            ("ENUMERATION", "smbclient_null_list", "SMB shares: general (read), Development (read/write) — general has creds.txt"),
            ("EXPLOITATION", "smbclient_get_file", "Download creds.txt from general share reveals admin password"),
            ("ENUMERATION", "gobuster_vhost", "DNS zone transfer reveals subdomains: administrator1.friendzone.red"),
            ("EXPLOITATION", "ssh_login", "Login to admin panel with SMB creds — dashboard has LFI parameter"),
            ("EXPLOITATION", "lfi_log_poison", "Upload PHP shell to SMB Development share then include via LFI for RCE"),
            ("PRIVILEGE_ESCALATION", "pspy", "pspy shows root cron runs reporter.py which imports os module"),
            ("POST_EXPLOITATION", "cron_backdoor", "Writable python os.py — inject reverse shell then root on next cron run"),
        ],
    },
    {
        "name": "htb_traverxec",
        "category": "full_box",
        "difficulty": "easy",
        "description": "Traverxec: Nostromo RCE then htpasswd crack then SSH key then journalctl pager root",
        "box_info": {"os": "Linux", "services": "ssh/22 http/80(nostromo)"},
        "steps": [
            ("RECON", "nmap_service_version", "SSH 22 and HTTP 80 — nostromo 1.9.6 web server (rare)"),
            ("ENUMERATION", "searchsploit", "searchsploit nostromo finds CVE-2019-16278 directory traversal RCE"),
            ("EXPLOITATION", "msfconsole_exploit", "Exploit nostromo RCE gets shell as www-data"),
            ("ENUMERATION", "gobuster_dir", "Find .htpasswd in nostromo config — crack hash with john"),
            ("EXPLOITATION", "hashcat_krb5", "John cracks the hash to get password Nowonly4me"),
            ("EXPLOITATION", "lfi_ssh_key", "Nostromo serves ~/public_www — find backup SSH key in protected-file-area"),
            ("EXPLOITATION", "ssh_key_login", "SSH as david with cracked passphrase plus extracted key"),
            ("PRIVILEGE_ESCALATION", "sudo_list", "Can run journalctl as root — uses less pager then !/bin/bash for root"),
        ],
    },
    {
        "name": "htb_openadmin",
        "category": "full_box",
        "difficulty": "easy",
        "description": "OpenAdmin: OpenNetAdmin RCE then internal Apache then SSH key then nano sudo root",
        "box_info": {"os": "Linux", "services": "ssh/22 http/80(Apache)"},
        "steps": [
            ("RECON", "nmap_service_version", "SSH and HTTP — Apache 2.4.29 on Ubuntu"),
            ("ENUMERATION", "gobuster_dir", "Find /music/ and /ona/ — OpenNetAdmin v18.1.1 interface"),
            ("EXPLOITATION", "msfconsole_exploit", "OpenNetAdmin 18.1.1 RCE (CVE-2019-12725) gives shell as www-data"),
            ("ENUMERATION", "curl_web_path", "Find database config with password n1nj4W4rri0R! — jimmy uses same password"),
            ("EXPLOITATION", "ssh_login", "su to jimmy with reused password"),
            ("EXPLOITATION", "curl_web_path", "Jimmy can access internal Apache on port 52846 which shows joannas SSH key"),
            ("EXPLOITATION", "ssh_key_login", "Crack joannas SSH key passphrase then SSH as joanna"),
            ("PRIVILEGE_ESCALATION", "sudo_list", "sudo -l: (root) NOPASSWD: /bin/nano /opt/priv — nano escape gives root"),
        ],
    },
    {
        "name": "htb_postman",
        "category": "full_box",
        "difficulty": "easy",
        "description": "Postman: Redis unauthenticated then SSH key plant then Webmin CVE then root",
        "box_info": {"os": "Linux", "services": "ssh/22 http/80 redis/6379 https/10000(Webmin)"},
        "steps": [
            ("RECON", "nmap_service_version", "SSH, HTTP, Redis 6379, Webmin 10000 — Redis with no auth is a red flag"),
            ("EXPLOITATION", "redis_cli", "Connect to Redis — no authentication required"),
            ("EXPLOITATION", "ssh_key_plant", "Generate SSH keypair then use Redis CONFIG SET dir/dbfilename to write authorized_keys"),
            ("EXPLOITATION", "ssh_key_login", "SSH as redis user with planted key"),
            ("ENUMERATION", "linpeas", "Find Matts encrypted SSH key backup in /opt then crack with john: computer2008"),
            ("EXPLOITATION", "ssh_login", "Webmin login as Matt:computer2008 on port 10000"),
            ("EXPLOITATION", "msfconsole_exploit", "Webmin 1.910 CVE-2019-12840 package updates RCE gives root"),
        ],
    },
    {
        "name": "htb_traceback",
        "category": "full_box",
        "difficulty": "easy",
        "description": "Traceback: Existing webshell then Lua sudo then motd write then root",
        "box_info": {"os": "Linux", "services": "ssh/22 http/80"},
        "steps": [
            ("RECON", "nmap_service_version", "SSH and HTTP — HTML source has comment about best web shells"),
            ("ENUMERATION", "gobuster_dir", "Search for common webshell names from the hint — find smevk.php"),
            ("EXPLOITATION", "webshell_cmd", "Login to smevk.php webshell (admin/admin) — execute commands as webadmin"),
            ("EXPLOITATION", "revshell_bash", "Get proper reverse shell as webadmin"),
            ("PRIVILEGE_ESCALATION", "sudo_list", "sudo -l: can run /home/sysadmin/luvit as sysadmin (Lua interpreter)"),
            ("PRIVILEGE_ESCALATION", "privesc_sudo_l", "Execute Lua: os.execute /bin/bash to become sysadmin"),
            ("POST_EXPLOITATION", "cron_backdoor", "sysadmin is in motd group — write to /etc/update-motd.d/00-header runs as root on SSH login"),
        ],
    },
    # ── Easy/Medium AD ─────────────────────────────────────────────────────
    {
        "name": "htb_active",
        "category": "full_box",
        "difficulty": "easy",
        "description": "Active: SMB GPP password then Kerberoast Administrator then psexec",
        "box_info": {"os": "Windows", "services": "dns/53 kerberos/88 msrpc/135 netbios/139 ldap/389 smb/445 gc/3268"},
        "steps": [
            ("RECON", "nmap_service_version", "Classic AD box — DNS/Kerberos/LDAP/SMB all present, domain: active.htb"),
            ("ENUMERATION", "smbclient_null_list", "Null session allows listing shares — Replication share readable"),
            ("EXPLOITATION", "smbclient_get_file", "Browse SYSVOL/Replication then find Groups.xml with cPassword"),
            ("EXPLOITATION", "gpp_decrypt", "gpp-decrypt cPassword reveals GPP password for SVC_TGS account"),
            ("EXPLOITATION", "impacket_GetUserSPNs", "Kerberoast with SVC_TGS creds gets TGS hash for Administrator"),
            ("EXPLOITATION", "hashcat_krb5", "Crack TGS hash: Ticketmaster1968"),
            ("EXPLOITATION", "impacket_psexec", "psexec.py with Administrator creds gets SYSTEM shell"),
        ],
    },
    {
        "name": "htb_forest",
        "category": "full_box",
        "difficulty": "easy",
        "description": "Forest: AS-REP Roast then WinRM then Exchange Windows Permissions then DCSync",
        "box_info": {"os": "Windows", "services": "dns/53 kerberos/88 ldap/389 smb/445 winrm/5985 gc/3269"},
        "steps": [
            ("RECON", "nmap_service_version", "Full AD stack — note WinRM 5985 open, domain: htb.local"),
            ("ENUMERATION", "ldapsearch_users", "LDAP anonymous bind enumerates users — find svc-alfresco service account"),
            ("EXPLOITATION", "impacket_GetNPUsers", "AS-REP Roast svc-alfresco — no preauth required, get AS-REP hash"),
            ("EXPLOITATION", "hashcat_krb5", "Crack AS-REP hash: s3rvice"),
            ("EXPLOITATION", "evil_winrm", "Evil-WinRM as svc-alfresco:s3rvice gives user shell"),
            ("ENUMERATION", "bloodhound_python", "BloodHound shows svc-alfresco can WriteDacl on domain via Exchange group"),
            ("POST_EXPLOITATION", "mimikatz_dcsync", "Add to Exchange Windows Permissions then grant DCSync rights then secretsdump"),
        ],
    },
    {
        "name": "htb_sauna",
        "category": "full_box",
        "difficulty": "easy",
        "description": "Sauna: Username enumeration then AS-REP Roast then WinRM then AutoLogon creds then DCSync",
        "box_info": {"os": "Windows", "services": "dns/53 kerberos/88 ldap/389 smb/445 winrm/5985"},
        "steps": [
            ("RECON", "nmap_service_version", "AD box with WinRM — domain: EGOTISTICAL-BANK.LOCAL"),
            ("ENUMERATION", "curl_web_path", "Website About Us page shows employee names — construct usernames"),
            ("EXPLOITATION", "impacket_GetNPUsers", "AS-REP Roast with username list shows fsmith has no preauth"),
            ("EXPLOITATION", "hashcat_krb5", "Crack fsmith AS-REP hash: Thestrokes23"),
            ("EXPLOITATION", "evil_winrm", "Evil-WinRM as fsmith gets user flag"),
            ("PRIVILEGE_ESCALATION", "winpeas", "WinPEAS finds AutoLogon credentials for svc_loanmgr"),
            ("EXPLOITATION", "evil_winrm", "Evil-WinRM as svc_loanmgr"),
            ("POST_EXPLOITATION", "mimikatz_dcsync", "svc_loanmgr has Replication rights so DCSync for Administrator hash"),
        ],
    },
    # ── TryHackMe-style patterns ───────────────────────────────────────────
    {
        "name": "thm_basic_pentesting",
        "category": "full_box",
        "difficulty": "easy",
        "description": "Basic Pentesting: SMB user enum then SSH brute then sudo bash",
        "box_info": {"os": "Linux", "services": "ssh/22 http/80 smb/139,445 http/8080(Tomcat)"},
        "steps": [
            ("RECON", "nmap_service_version", "SSH, HTTP, SMB, Tomcat — multiple attack surfaces"),
            ("ENUMERATION", "enum4linux_full", "SMB enumeration reveals users: kay, jan"),
            ("EXPLOITATION", "hydra_ssh", "Hydra brute force jans SSH password: armando"),
            ("EXPLOITATION", "ssh_login", "SSH as jan with cracked password"),
            ("ENUMERATION", "lfi_ssh_key", "Find kays SSH key in /home/kay/.ssh/ — but need passphrase"),
            ("EXPLOITATION", "hashcat_krb5", "john id_rsa crack SSH key passphrase: beeswax"),
            ("EXPLOITATION", "ssh_key_login", "SSH as kay with cracked key"),
            ("PRIVILEGE_ESCALATION", "sudo_list", "sudo -l: kay can run all commands so sudo /bin/bash gives root"),
        ],
    },
    {
        "name": "thm_rootme",
        "category": "full_box",
        "difficulty": "easy",
        "description": "RootMe: File upload filter bypass then PHP reverse shell then SUID Python",
        "box_info": {"os": "Linux", "services": "ssh/22 http/80(Apache)"},
        "steps": [
            ("RECON", "nmap_quick_scan", "SSH and HTTP open"),
            ("ENUMERATION", "gobuster_dir", "Find /panel/ (upload page) and /uploads/ directory"),
            ("EXPLOITATION", "upload_php_double_ext", ".php blocked — bypass with .php5 or .phtml extension"),
            ("EXPLOITATION", "revshell_bash", "Trigger uploaded PHP shell for reverse connection"),
            ("PRIVILEGE_ESCALATION", "find_suid", "Find /usr/bin/python with SUID bit set"),
            ("PRIVILEGE_ESCALATION", "privesc_cap_setuid_python", "Python SUID: setuid(0) then spawn bash"),
        ],
    },
    {
        "name": "thm_kenobi",
        "category": "full_box",
        "difficulty": "easy",
        "description": "Kenobi: SMB plus NFS then ProFTPd mod_copy then SSH key steal then SUID path injection",
        "box_info": {"os": "Linux", "services": "ftp/21 ssh/22 http/80 smb/139,445 rpc/111 nfs/2049"},
        "steps": [
            ("RECON", "nmap_service_version", "Many services — FTP (ProFTPd 1.3.5), SMB, NFS all exploitable"),
            ("ENUMERATION", "smbclient_null_list", "Anonymous SMB access — find log.txt mentioning /home/kenobi/.ssh/"),
            ("ENUMERATION", "showmount", "NFS showmount: /var is shared"),
            ("EXPLOITATION", "msfconsole_exploit", "ProFTPd mod_copy CVE-2015-3306: CPFR id_rsa then CPTO /var/tmp/id_rsa"),
            ("EXPLOITATION", "nfs_mount", "Mount /var via NFS then retrieve copied id_rsa"),
            ("EXPLOITATION", "ssh_key_login", "SSH as kenobi with stolen key"),
            ("PRIVILEGE_ESCALATION", "find_suid", "SUID binary: /usr/bin/menu — strings shows it calls curl without full path"),
            ("PRIVILEGE_ESCALATION", "privesc_sudo_l", "PATH injection: create fake curl binary then run SUID menu for root"),
        ],
    },
    {
        "name": "thm_skynet",
        "category": "full_box",
        "difficulty": "easy",
        "description": "Skynet: SMB then Cuppa CMS RFI then tar wildcard cron root",
        "box_info": {"os": "Linux", "services": "ssh/22 http/80 smb/139,445 imap/143 pop3/110"},
        "steps": [
            ("RECON", "nmap_service_version", "Web, Mail, SMB — SMB has interesting shares"),
            ("ENUMERATION", "smbclient_null_list", "SMB anonymous: milesdyson share readable with found password"),
            ("EXPLOITATION", "smbclient_get_file", "Find important.txt with CMS path: /45kra24zxs28v3yd0/"),
            ("ENUMERATION", "gobuster_dir", "Discover Cuppa CMS at hidden path"),
            ("EXPLOITATION", "rfi_php_shell", "Cuppa CMS Remote File Inclusion — include reverse shell from attacker HTTP"),
            ("EXPLOITATION", "revshell_bash", "Get shell as www-data"),
            ("PRIVILEGE_ESCALATION", "cron_check", "Cron runs backup.sh with tar wildcard in /var/www/html/"),
            ("POST_EXPLOITATION", "cron_backdoor", "Tar wildcard injection: checkpoint and checkpoint-action=exec gives root"),
        ],
    },
    {
        "name": "htb_delivery",
        "category": "full_box",
        "difficulty": "easy",
        "description": "Delivery: MatterMost plus OSTicket email verification bypass then hashcat rules",
        "box_info": {"os": "Linux", "services": "ssh/22 http/80 http/8065(MatterMost)"},
        "steps": [
            ("RECON", "nmap_service_version", "SSH, HTTP (nginx), MatterMost on 8065"),
            ("ENUMERATION", "gobuster_dir", "Find HelpDesk (OSTicket) on helpdesk.delivery.htb"),
            ("EXPLOITATION", "curl_web_path", "Create OSTicket ticket to get @delivery.htb email address for verification"),
            ("EXPLOITATION", "ssh_login", "Register MatterMost with OSTicket email then verify via ticket reply"),
            ("POST_EXPLOITATION", "credential_dump", "MatterMost channel has root SSH creds for maildeliverer"),
            ("EXPLOITATION", "ssh_login", "SSH as maildeliverer for user shell"),
            ("PRIVILEGE_ESCALATION", "linpeas", "Find MatterMost config.json with MySQL creds then dump user hashes"),
            ("EXPLOITATION", "hashcat_krb5", "Hashcat with rules based on password hint PleaseSubscribe gives root password"),
        ],
    },
    {
        "name": "htb_sau",
        "category": "full_box",
        "difficulty": "easy",
        "description": "Sau: Request Baskets SSRF then Maltrail RCE then systemctl pager root",
        "box_info": {"os": "Linux", "services": "ssh/22 filtered/80 http/55555(Request-Baskets)"},
        "steps": [
            ("RECON", "nmap_service_version", "SSH, filtered 80, Request Baskets on 55555 — port 80 filtered externally"),
            ("ENUMERATION", "curl_web_path", "Request Baskets v1.2.1 — SSRF vulnerability (CVE-2023-27163)"),
            ("EXPLOITATION", "ssrf_localhost_scan", "Create basket with forward_url to 127.0.0.1:80 for SSRF to see filtered Maltrail"),
            ("EXPLOITATION", "cmd_inject_semicolon", "Maltrail v0.53 has unauthenticated OS command injection in login page"),
            ("EXPLOITATION", "revshell_bash", "Inject reverse shell via SSRF plus Maltrail RCE — shell as puma"),
            ("PRIVILEGE_ESCALATION", "sudo_list", "sudo -l: NOPASSWD: /usr/bin/systemctl status trail.service"),
            ("PRIVILEGE_ESCALATION", "privesc_sudo_l", "systemctl uses less pager — !/bin/bash for root"),
        ],
    },
]

ALL_CHAINS = WEB_CHAINS + SERVICE_CHAINS + AD_CHAINS + LINUX_PRIVESC_CHAINS + WINDOWS_PRIVESC_CHAINS + LATERAL_CHAINS + FULL_BOX_CHAINS + REAL_HTB_BOXES


# ═══════════════════════════════════════════════════════════════════════════════
# COMMON DATA HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

_COMMON_PORTS = {
    21: "ftp", 22: "ssh", 25: "smtp", 53: "dns", 80: "http", 110: "pop3",
    111: "rpcbind", 135: "msrpc", 139: "netbios", 143: "imap", 443: "https",
    445: "smb", 993: "imaps", 995: "pop3s", 1433: "mssql", 1521: "oracle",
    3306: "mysql", 3389: "rdp", 5432: "postgres", 5985: "winrm",
    6379: "redis", 8080: "http-alt", 8443: "https-alt", 27017: "mongodb",
    88: "kerberos", 389: "ldap", 636: "ldaps", 1524: "ingreslock",
    2049: "nfs", 6667: "irc", 8180: "http-alt", 9200: "elasticsearch",
    161: "snmp", 500: "ike", 4444: "meterpreter",
}

_WEB_PATHS = [
    "/admin", "/login", "/api", "/upload", "/config", "/backup", "/wp-admin",
    "/wp-login.php", "/robots.txt", "/.git", "/phpmyadmin", "/console",
    "/manager/html", "/cgi-bin", "/server-status", "/.env", "/wp-content",
    "/xmlrpc.php", "/api/v1", "/graphql", "/swagger", "/.htpasswd",
    "/actuator", "/debug", "/info", "/status", "/health", "/shell",
    "/dashboard", "/portal", "/webmail", "/owa", "/exchange", "/panel",
]

_SERVICE_VERSIONS = {
    "ftp": ["vsftpd 2.3.4", "ProFTPD 1.3.5", "vsftpd 3.0.3", "Pure-FTPd"],
    "ssh": ["OpenSSH 7.2p2", "OpenSSH 8.2p1", "OpenSSH 7.9p1", "OpenSSH 8.9p1"],
    "http": ["Apache/2.4.49", "nginx/1.18.0", "Apache/2.4.41", "IIS/10.0", "Werkzeug/2.0.1"],
    "smb": ["Samba 3.0.20", "Samba 4.6.2", "Windows Server 2019"],
    "mysql": ["MySQL 5.7.29", "MariaDB 10.3.22", "MySQL 8.0.26"],
    "mssql": ["SQL Server 2019", "SQL Server 2017"],
    "postgres": ["PostgreSQL 12.3", "PostgreSQL 14.1"],
}

_VULN_EXAMPLES = [
    "CVE-2021-44228 (Log4Shell)", "CVE-2023-22515 (Confluence)", "ms17-010 (EternalBlue)",
    "CVE-2021-3156 (Baron Samedit)", "CVE-2022-0847 (DirtyPipe)", "CVE-2016-5195 (DirtyCow)",
    "CVE-2021-41773 (Apache Path Traversal)", "CVE-2014-6271 (Shellshock)",
    "CVE-2017-0143 (EternalBlue)", "CVE-2019-15107 (Webmin RCE)",
    "CVE-2021-4034 (PwnKit)", "CVE-2022-22963 (Spring4Shell)",
    "CVE-2018-15473 (SSH User Enum)", "CVE-2014-0160 (Heartbleed)",
    "CVE-2019-14287 (Sudo bypass)", "CVE-2023-32315 (Openfire)",
]

_PHASE_MAP = {
    "RECON": "recon", "ENUMERATION": "enumeration", "EXPLOITATION": "exploitation",
    "POST_EXPLOITATION": "post_exploitation", "PRIVILEGE_ESCALATION": "privesc",
    "LATERAL_MOVEMENT": "lateral", "EXFILTRATION": "exfil", "CLOSEOUT": "closeout",
}

SYSTEM_PROMPTS = {
    "command_select": (
        "You are a fast command selector for Ariaska pentesting system. "
        "Output ONLY a JSON object with: command, template_name, reasoning, "
        "score (0.0-1.0). No markdown."
    ),
    "mentor": (
        "You are an elite pentester AI MENTOR for Ariaska system. Select the "
        "BEST next action. Output ONLY valid JSON with: intent, selected_command "
        "(MUST be a template_name like nmap_fast_scan), parameters (dict), "
        "reasoning, expected_observation, risk (low/medium/high), confidence "
        "(0.0-1.0), next_phase_hint, candidate_actions (list of dicts). No markdown."
    ),
    "strategic": (
        "You are the STRATEGIC ADVISOR for Ariaska pentesting system. Analyze the "
        "target state and produce a multi-step attack plan. Output ONLY valid JSON "
        "with: situation_assessment, recommended_approach, step_plan (list of "
        "{step_number, command, reasoning, expected_outcome}), risk_assessment, "
        "confidence (0.0-1.0), alternative_approaches (list). No markdown."
    ),
    "classifier": (
        "You are a phase classifier for Ariaska pentesting system. "
        "Classify the current engagement phase. "
        "Output ONLY one word: recon, enumeration, exploitation, "
        "privesc, lateral, post_exploitation, exfil, or closeout."
    ),
    "output_parse": (
        "You are a penetration-testing output parser. "
        "Given a command and its STDOUT, extract structured discoveries. "
        "Return ONLY valid JSON — no markdown, no explanation."
    ),
}


def _random_target() -> str:
    return f"10.10.10.{random.randint(2, 254)}"


def _random_discovery_board(
    phase: str, chain_type: str = "web", difficulty: str = "easy",
) -> Dict[str, Any]:
    """Generate a realistic discovery board matching the engagement state."""
    # More ports for harder boxes
    n_ports = random.randint(2, 5) if difficulty == "easy" else random.randint(4, 10)
    ports = random.sample(list(_COMMON_PORTS.keys()), min(n_ports, len(_COMMON_PORTS)))

    # Add type-appropriate ports
    if chain_type in ("ad", "lateral"):
        for p in [88, 389, 445, 636, 5985]:
            if p not in ports:
                ports.append(p)
    elif chain_type in ("web", "full_box"):
        for p in [80, 443]:
            if p not in ports:
                ports.append(p)
    elif chain_type == "service":
        pass  # random ports are fine

    services = {}
    for p in ports:
        svc = _COMMON_PORTS.get(p, "unknown")
        versions = _SERVICE_VERSIONS.get(svc, [f"{svc} unknown"])
        services[str(p)] = random.choice(versions) if versions else svc

    board: Dict[str, Any] = {
        "ports": sorted([str(p) for p in ports]),
        "services": services,
        "credentials": [],
        "vulns": [],
        "shells": [],
        "users": [],
        "web_paths": [],
    }

    # Progressive discovery based on phase
    if phase in ("EXPLOITATION", "POST_EXPLOITATION", "PRIVILEGE_ESCALATION", "LATERAL_MOVEMENT"):
        n_creds = random.randint(1, 3)
        board["credentials"] = [
            {"username": f"user{i}", "password": f"{''.join(random.choices('abcdef0123456789', k=8))}", "service": random.choice(["ssh", "ftp", "mysql", "web"])}
            for i in range(n_creds)
        ]
        board["users"] = [f"user{i}" for i in range(random.randint(2, 6))]
    if phase in ("POST_EXPLOITATION", "PRIVILEGE_ESCALATION", "LATERAL_MOVEMENT"):
        board["shells"] = [{"type": "user", "user": "www-data", "host": "target"}]
        board["vulns"] = random.sample(_VULN_EXAMPLES, k=random.randint(1, 3))
    if phase in ("LATERAL_MOVEMENT",):
        board["shells"] = [{"type": "root", "user": "root", "host": "target"}]

    if chain_type in ("web", "full_box"):
        board["web_paths"] = random.sample(_WEB_PATHS, k=random.randint(2, 6))

    return board


def _get_available_commands(chain: Dict, step_idx: int, n_distractors: int = 6) -> List[str]:
    """Get the correct command + realistic distractors."""
    correct = chain["steps"][step_idx][1]
    # Get all unique commands from all chains
    all_cmds = list({s[1] for c in ALL_CHAINS for s in c["steps"] if s[1] != correct})
    distractors = random.sample(all_cmds, min(n_distractors, len(all_cmds)))
    available = [correct] + distractors
    random.shuffle(available)
    return available


# ═══════════════════════════════════════════════════════════════════════════════
# SFT DATA GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════

def gen_command_select(
    target: str, phase: str, command: str, reasoning: str,
    board: Dict, available: List[str],
) -> Dict[str, Any]:
    """microchain_fast_local: fast command selection."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS["command_select"]},
            {
                "role": "user",
                "content": (
                    f"Pick the single best next command.\n\n"
                    f"Target: {target}\nPhase: {phase}\nRole: offensive\n"
                    f"Stagnation: {random.randint(0, 5)} steps\n"
                    f"Discovery board:\n{json.dumps(board, indent=2)}\n"
                    f"Recent commands: {', '.join(random.sample(available, min(3, len(available))))}\n"
                    f"Available templates: {', '.join(available)}"
                ),
            },
            {
                "role": "assistant",
                "content": json.dumps({
                    "command": command,
                    "template_name": command,
                    "reasoning": reasoning,
                    "score": round(random.uniform(0.75, 0.95), 2),
                }),
            },
        ],
        "schema_type": "microchain_fast_local",
    }


def gen_mentor(
    target: str, phase: str, command: str, reasoning: str,
    board: Dict, available: List[str], next_hint: str = "",
) -> Dict[str, Any]:
    """smart_mentor: tactical advice with reasoning."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS["mentor"]},
            {
                "role": "user",
                "content": (
                    f"Select the best next command. selected_command MUST be a template_name from available.\n\n"
                    f"Target: {target}\nPhase: {phase}\nRole: offensive\n"
                    f"Stagnation: {random.randint(0, 8)} steps\n"
                    f"Discovery board:\n{json.dumps(board, indent=2)}\n"
                    f"Recent commands: {', '.join(random.sample(available, min(3, len(available))))}\n"
                    f"Available templates: {', '.join(available)}"
                ),
            },
            {
                "role": "assistant",
                "content": json.dumps({
                    "intent": reasoning.split(".")[0] + ".",
                    "selected_command": command,
                    "parameters": {"target": target},
                    "reasoning": reasoning,
                    "expected_observation": f"Output from {command} revealing actionable information.",
                    "risk": random.choice(["low", "medium", "high"]),
                    "confidence": round(random.uniform(0.7, 0.95), 2),
                    "next_phase_hint": next_hint or f"Continue with {phase.lower().replace('_', ' ')} objectives.",
                    "candidate_actions": [
                        {"command": c, "description": f"Alternative: {c}"}
                        for c in random.sample(available, min(2, len(available)))
                    ],
                }),
            },
        ],
        "schema_type": "smart_mentor",
    }


def gen_phase_classify(
    target: str, phase: str, board: Dict,
) -> Dict[str, Any]:
    """phase_classifier: classify current engagement phase."""
    mapped = _PHASE_MAP.get(phase, "recon")
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS["classifier"]},
            {
                "role": "user",
                "content": (
                    f"Classify engagement phase.\n\nTarget: {target}\n"
                    f"Discovery board:\n{json.dumps(board, indent=2)}"
                ),
            },
            {"role": "assistant", "content": mapped},
        ],
        "schema_type": "phase_classifier",
    }


def gen_strategic(
    target: str, chain: Dict, board: Dict,
) -> Dict[str, Any]:
    """strategic_plan: multi-step attack plan (for cloud 70B alignment)."""
    steps = chain["steps"]
    step_plan = [
        {
            "step_number": i + 1,
            "command": cmd,
            "reasoning": reason,
            "expected_outcome": f"Progress toward {phase.lower().replace('_', ' ')}.",
        }
        for i, (phase, cmd, reason) in enumerate(steps)
    ]

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS["strategic"]},
            {
                "role": "user",
                "content": (
                    f"Analyze target state and create an attack plan.\n\n"
                    f"Target: {target}\nPhase: {steps[0][0]}\n"
                    f"Discovery board:\n{json.dumps(board, indent=2)}\n"
                    f"Attack chain type: {chain['name']}\n"
                    f"Difficulty: {chain['difficulty']}\n"
                    f"Objective: Achieve maximum privilege on target."
                ),
            },
            {
                "role": "assistant",
                "content": json.dumps({
                    "situation_assessment": (
                        f"Target at {target} shows {len(board['ports'])} open ports. "
                        f"{chain['description']}. "
                        f"Difficulty: {chain['difficulty']}."
                    ),
                    "recommended_approach": chain["description"],
                    "step_plan": step_plan,
                    "risk_assessment": f"{'Low' if chain['difficulty'] == 'easy' else 'Medium'} — {chain['category']} attack chain.",
                    "confidence": round(random.uniform(0.75, 0.92), 2),
                    "alternative_approaches": [
                        "Try alternative service if primary attack stalls",
                        "Look for credential reuse across services",
                        "Enumerate more thoroughly before committing to exploitation",
                    ],
                }),
            },
        ],
        "schema_type": "strategic_plan",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT PARSING TRAINING DATA
# ═══════════════════════════════════════════════════════════════════════════════

_NMAP_OUTPUTS = [
    # Typical easy Linux box
    """Starting Nmap 7.94 ( https://nmap.org ) at 2024-01-15 10:30 EST
Nmap scan report for 10.10.10.245
PORT   STATE SERVICE VERSION
22/tcp open  ssh     OpenSSH 8.2p1 Ubuntu 4ubuntu0.2
80/tcp open  http    gunicorn
Service Info: OS: Linux; CPE: cpe:/o:linux:linux_kernel""",
    # AD box
    """Nmap scan report for 10.10.10.100
PORT     STATE SERVICE       VERSION
53/tcp   open  domain        Simple DNS Plus
88/tcp   open  kerberos-sec  Microsoft Windows Kerberos
135/tcp  open  msrpc         Microsoft Windows RPC
389/tcp  open  ldap          Microsoft Windows Active Directory LDAP
445/tcp  open  microsoft-ds  Windows Server 2019 Standard 17763
5985/tcp open  http          Microsoft HTTPAPI httpd 2.0 (SSDP/UPnP)
Service Info: Host: DC; OS: Windows; CPE: cpe:/o:microsoft:windows""",
    # Web-heavy box
    """Nmap scan report for 10.10.10.88
PORT     STATE SERVICE VERSION
22/tcp   open  ssh     OpenSSH 7.2p2 Ubuntu 4ubuntu2.8
80/tcp   open  http    Apache httpd 2.4.18 ((Ubuntu))
3306/tcp open  mysql   MySQL 5.7.29-0ubuntu0.16.04.1
8080/tcp open  http    Apache Tomcat 9.0.30""",
]

_GOBUSTER_OUTPUTS = [
    """===============================================================
Gobuster v3.1.0
===============================================================
/admin                (Status: 302) [Size: 0] [--> /login]
/api                  (Status: 200) [Size: 48]
/backup               (Status: 403) [Size: 277]
/config               (Status: 200) [Size: 1523]
/dashboard            (Status: 302) [Size: 0] [--> /login]
/login                (Status: 200) [Size: 2345]
/uploads              (Status: 301) [Size: 314] [--> http://10.10.10.88/uploads/]
/wp-admin             (Status: 301) [Size: 317]
/.git                 (Status: 403) [Size: 277]
/.env                 (Status: 200) [Size: 892]""",
    """===============================================================
/cgi-bin              (Status: 403) [Size: 294]
/icons                (Status: 403) [Size: 294]
/index.php            (Status: 200) [Size: 10918]
/server-status        (Status: 403) [Size: 294]
/phpmyadmin           (Status: 200) [Size: 8245]
/robots.txt           (Status: 200) [Size: 45]""",
]

_LINPEAS_OUTPUTS = [
    """╔══════════╣ SUID - Check easy privesc,01-exploits andடறே GTFOBins
-rwsr-xr-x 1 root root 16712 Sep 28 2021 /usr/bin/python3.8
-rwsr-xr-x 1 root root 40152 Nov 22 2021 /usr/bin/find
-rwsr-xr-x 1 root root 44680 Feb  6 2021 /usr/bin/newgrp
-rwsr-xr-x 1 root root 68208 Jul 14 2021 /usr/bin/passwd
-rwsr-xr-x 1 root root 85064 Jul 14 2021 /usr/bin/chfn

╔══════════╣ Capabilities
/usr/bin/python3.8 = cap_setuid+ep
/usr/bin/perl = cap_setuid+ep

╔══════════╣ Cron Jobs
*/5 * * * * root /opt/scripts/cleanup.sh
* * * * * root cd /var/www && php artisan schedule:run""",
    """╔══════════╣ Users with console
root:x:0:0:root:/root:/bin/bash
nathan:x:1000:1000::/home/nathan:/bin/bash

╔══════════╣ Docker
User is member of 'docker' group

╔══════════╣ Interesting Files
-rw-r--r-- 1 root root 1234 /opt/app/.env
-rw-r--r-- 1 nathan nathan 3456 /home/nathan/.mysql_history""",
]

_CURL_OUTPUTS = [
    """<!DOCTYPE html>
<html>
<head><title>Internal Dashboard</title></head>
<body>
<!-- DEBUG: admin_token=eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0.eyJ1c2VyIjoiYWRtaW4ifQ. -->
<h1>Welcome to the dashboard</h1>
<form action="/api/exec" method="POST">
  <input name="cmd" placeholder="Enter command">
</form>
<!-- TODO: Remove debug endpoint /api/debug?file= -->
</body>
</html>""",
    """HTTP/1.1 200 OK
Server: Werkzeug/2.0.1 Python/3.8.10
Content-Type: application/json

{"users": [{"id": 1, "username": "admin", "role": "administrator"},
{"id": 2, "username": "nathan", "role": "user"}],
"database": "mysql://root:r00tme@localhost/app",
"debug": true, "secret_key": "supersecretkey123"}""",
]


def gen_output_parse_examples(count: int = 2000, seed: int = 42) -> List[Dict]:
    """Generate output parsing training examples."""
    random.seed(seed)
    examples = []

    output_map = {
        "nmap": (_NMAP_OUTPUTS, lambda out: _parse_nmap_expected(out)),
        "gobuster": (_GOBUSTER_OUTPUTS, lambda out: _parse_gobuster_expected(out)),
        "linpeas": (_LINPEAS_OUTPUTS, lambda out: _parse_linpeas_expected(out)),
        "curl": (_CURL_OUTPUTS, lambda out: _parse_curl_expected(out)),
    }

    for _ in range(count):
        tool = random.choice(list(output_map.keys()))
        outputs, parser = output_map[tool]
        output = random.choice(outputs)
        expected = parser(output)

        target = _random_target()
        if tool == "nmap":
            cmd = f"nmap -sV {target}"
        elif tool == "gobuster":
            cmd = f"gobuster dir -u http://{target} -w /usr/share/wordlists/dirb/common.txt"
        elif tool == "linpeas":
            cmd = "bash linpeas.sh"
        else:
            cmd = f"curl -s http://{target}/"

        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPTS["output_parse"]},
                {
                    "role": "user",
                    "content": (
                        f"Parse this penetration testing tool output.\n\n"
                        f"Tool: {tool}\nCommand: {cmd}\nSTDOUT:\n```\n{output}\n```\n\nJSON:"
                    ),
                },
                {"role": "assistant", "content": json.dumps(expected)},
            ],
            "schema_type": "output_parse",
        })

    return examples


def _parse_nmap_expected(output: str) -> Dict:
    ports = []
    services = {}
    os_info = ""
    for m in re.finditer(r'(\d+)/tcp\s+open\s+(\S+)\s+(.*)', output):
        port = int(m.group(1))
        ports.append(port)
        services[str(port)] = f"{m.group(2)} {m.group(3).strip()}"
    if "Linux" in output:
        os_info = "Linux"
    elif "Windows" in output:
        os_info = "Windows"
    return {"open_ports": ports, "services": services, "os_info": os_info, "success": True}


def _parse_gobuster_expected(output: str) -> Dict:
    paths = []
    for m in re.finditer(r'(/\S+)\s+\(Status: (\d+)\)', output):
        paths.append(m.group(1))
    return {"web_paths": paths, "success": True}


def _parse_linpeas_expected(output: str) -> Dict:
    result: Dict[str, Any] = {"success": True, "artifacts": {}}
    suids = re.findall(r'-rwsr.+?(/\S+)', output)
    if suids:
        result["artifacts"]["suid_binaries"] = suids
    caps = re.findall(r'(/\S+)\s*=\s*(cap_\S+)', output)
    if caps:
        result["artifacts"]["capabilities"] = {p: c for p, c in caps}
    if "docker" in output.lower():
        result["artifacts"]["docker_group"] = True
    crons = re.findall(r'[\*/\d]+\s+[\*/\d]+\s+[\*/\d]+\s+[\*/\d]+\s+[\*/\d]+\s+\S+\s+(.+)', output)
    if crons:
        result["artifacts"]["cron_jobs"] = crons
    users = re.findall(r'^(\w+):x:\d+:\d+:.*:/bin/(?:ba)?sh$', output, re.MULTILINE)
    if users:
        result["users"] = users
    return result


def _parse_curl_expected(output: str) -> Dict:
    result: Dict[str, Any] = {"success": True, "artifacts": {}, "credentials": [], "web_paths": []}
    # Find hidden comments
    for m in re.finditer(r'<!--.*?-->', output, re.DOTALL):
        comment = m.group()
        if any(kw in comment.lower() for kw in ["debug", "token", "password", "secret", "todo", "admin"]):
            result["artifacts"]["html_comments"] = result["artifacts"].get("html_comments", [])
            result["artifacts"]["html_comments"].append(comment.strip())
    # Find URLs/paths in forms/links
    for m in re.finditer(r'(?:action|href|src)="(/[^"]+)"', output):
        result["web_paths"].append(m.group(1))
    # Find credentials in JSON
    for m in re.finditer(r'"(?:password|secret[_-]?key|token|api[_-]?key)"\s*:\s*"([^"]+)"', output, re.IGNORECASE):
        result["credentials"].append({"type": "api_key_or_password", "value": m.group(1)})
    # Find database URLs
    for m in re.finditer(r'(mysql|postgres|mongodb)://(\S+)', output):
        result["artifacts"]["database_url"] = m.group(0)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# DPO PREFERENCE PAIR GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

# Common mistakes HTB players make (rejected choices)
_COMMON_MISTAKES = {
    "RECON": [
        ("nmap_quick_scan", "Repeating recon scan that was already done — wastes time"),
        ("whois_lookup", "Whois lookup on internal HTB IP — irrelevant information"),
    ],
    "ENUMERATION": [
        ("nmap_quick_scan", "Going back to basic recon when enumeration is needed"),
        ("nmap_udp_scan", "Starting slow UDP scan when faster web enum would find more"),
    ],
    "EXPLOITATION": [
        ("gobuster_dir", "Still enumerating when an exploit path is clear"),
        ("nmap_service_version", "Retreating to recon when exploitation is the right phase"),
        ("nikto_scan", "Running noisy scanner when a specific vuln is already identified"),
    ],
    "PRIVILEGE_ESCALATION": [
        ("nmap_quick_scan", "Going back to port scanning after getting a shell"),
        ("gobuster_dir", "Web enumeration from a shell — should be doing privesc"),
        ("revshell_bash", "Trying another reverse shell when already on the box"),
    ],
    "POST_EXPLOITATION": [
        ("nmap_quick_scan", "Port scanning from compromised host without purpose"),
        ("linpeas", "Running LinPEAS again after already having root"),
    ],
}


def gen_dpo_pair(
    target: str, phase: str, chosen_cmd: str, chosen_reason: str,
    board: Dict, available: List[str],
) -> Dict[str, Any]:
    """Generate a DPO preference pair (chosen vs rejected)."""
    # Pick a realistic mistake for the phase
    mistakes = _COMMON_MISTAKES.get(phase, _COMMON_MISTAKES["RECON"])
    rejected_cmd, rejected_reason = random.choice(mistakes)

    prompt = (
        f"Pick the single best next command.\n\n"
        f"Target: {target}\nPhase: {phase}\nRole: offensive\n"
        f"Stagnation: {random.randint(0, 5)} steps\n"
        f"Discovery board:\n{json.dumps(board, indent=2)}\n"
        f"Available templates: {', '.join(available + [rejected_cmd])}"
    )

    chosen_response = json.dumps({
        "command": chosen_cmd,
        "template_name": chosen_cmd,
        "reasoning": chosen_reason,
        "score": round(random.uniform(0.8, 0.95), 2),
    })

    rejected_response = json.dumps({
        "command": rejected_cmd,
        "template_name": rejected_cmd,
        "reasoning": rejected_reason,
        "score": round(random.uniform(0.3, 0.6), 2),
    })

    return {
        "prompt": prompt,
        "chosen": chosen_response,
        "rejected": rejected_response,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# HTB WALKTHROUGH MINER
# ═══════════════════════════════════════════════════════════════════════════════

_KNOWN_TOOLS = {
    "nmap", "gobuster", "ffuf", "curl", "wget", "python", "python3", "perl",
    "ssh", "scp", "nc", "ncat", "socat", "nikto", "sqlmap", "hydra",
    "john", "hashcat", "crackmapexec", "evil-winrm", "smbclient",
    "enum4linux", "rpcclient", "ldapsearch", "bloodhound", "certipy",
    "rubeus", "mimikatz", "searchsploit", "msfconsole",
    "msfvenom", "chisel", "ligolo", "feroxbuster", "dirsearch",
    "whatweb", "wpscan", "linpeas", "winpeas", "sudo", "find",
    "cat", "grep", "awk", "sed", "chmod", "chown", "ls", "cd",
    "mkdir", "cp", "mv", "rm", "tar", "unzip", "base64", "xxd",
    "openssl", "strings", "file", "id", "whoami", "uname",
    "ifconfig", "ip", "netstat", "ss", "ps", "mount", "umount",
    "showmount", "ftp", "telnet", "redis-cli", "mysql", "psql",
    "dig", "host", "dnsrecon", "amass", "rustscan", "masscan",
    "responder", "ntlmrelayx", "secretsdump", "psexec", "wmiexec",
    "smbmap", "snmpwalk", "nbtscan", "kerbrute", "getnpusers",
    "getuserspns", "tshark", "tcpdump",
    "export", "bash", "sh", "php", "ruby", "java", "javac",
    "powershell", "certutil", "net", "reg", "wmic", "icacls",
    "docker", "kubectl", "git", "pip", "pip3", "npm", "go",
    "gcc", "g++", "make", "cmake", "cargo", "rustc",
    "./", "../",  # Relative path execution
}


def _is_valid_command(line: str) -> bool:
    """Strict check: line must start with a known tool or look like a command invocation."""
    if len(line) < 4 or len(line) > 300:
        return False

    # Reject obvious non-commands
    if line.startswith(("!", "-", "#", "//", "/*", "<!--", "|", ">", "*", "=")):
        return False
    if line.startswith(("SF:", "PORT", "Host", "MAC", "Nmap", "Service")):
        return False  # nmap output
    if re.match(r'^\d+/(tcp|udp)', line):
        return False  # Port listing output
    if "\\x" in line or "\\r\\n" in line:
        return False  # Hex-encoded output
    if line.startswith("[") and "]" in line[:30]:
        return False  # Bracketed output like [*] or [+]
    if re.match(r'^[A-Z][a-z].*\s.*\s', line) and "/" not in line[:20]:
        return False  # Prose starting with capitalized word
    if "(Status:" in line or "[Size:" in line:
        return False  # gobuster/feroxbuster output
    if re.match(r'^(GCC|GNU|ELF|LSB|MSB)\b', line):
        return False  # file/strings output
    if re.match(r'^(sudo|ssh|nmap|python)\s+version\b', line, re.IGNORECASE):
        return False  # version output, not command
    if re.match(r'^[a-f0-9]{32,}$', line):
        return False  # Hash output
    if re.match(r'^(total|drwx|lrwx|-rwx|-rw-)', line):
        return False  # ls output

    # Must start with a known tool or common command pattern
    first_word = line.split()[0].lower().rstrip(":") if line.split() else ""
    # Strip path prefix (e.g., /usr/bin/nmap → nmap)
    first_word_base = first_word.rsplit("/", 1)[-1] if "/" in first_word else first_word

    if first_word_base in _KNOWN_TOOLS or first_word in _KNOWN_TOOLS:
        return True
    # Allow relative/absolute path execution
    if line.startswith(("./", "../", "/")):
        return True
    # Allow env var setting (KEY=val command)
    if re.match(r'^[A-Z_]+=\S+\s+\w', line):
        return True

    return False


def mine_walkthroughs(walkthrough_dir: Path) -> List[Dict]:
    """Parse HTB/THM walkthrough .md files into training examples.

    Handles:
      - Flat directory of .md files
      - Nested repo structures (box-name/README.md, box-name/writeup.md)
      - Various code block formats (```bash, ```shell, ```console, ```)
      - Section-aware phase detection from headers
    """
    examples = []
    seen_commands: set = set()

    # Find all text-based writeup files recursively (.md, .txt, .rst)
    md_files = sorted(
        list(walkthrough_dir.rglob("*.md"))
        + list(walkthrough_dir.rglob("*.txt"))
        + list(walkthrough_dir.rglob("*.rst"))
    )
    print(f"  Found {len(md_files)} writeup files to mine")

    for md_file in md_files:
        if md_file.name.lower() in ("readme.md", "index.md", "contributing.md", "license.md"):
            # For nested repos, README.md inside a box directory IS the writeup
            if md_file.parent == walkthrough_dir:
                continue  # Skip top-level README

        content = md_file.read_text(errors="ignore")
        if len(content) < 200:
            continue  # Skip tiny files

        # Infer box name from parent dir or filename
        box_name = md_file.parent.name if md_file.name.lower() in ("readme.md", "writeup.md") else md_file.stem
        box_name = box_name.replace("-", " ").replace("_", " ").title().strip()
        if not box_name or box_name.lower() in ("data", "htb", "thm", "writeups", "hackthebox"):
            box_name = md_file.stem.title()

        # Detect platform from path
        path_str = str(md_file).lower()
        platform = "HTB" if "htb" in path_str or "hackthebox" in path_str else (
            "THM" if "thm" in path_str or "tryhackme" in path_str else "HTB"
        )

        # Detect OS from content
        content_lower = content.lower()
        os_type = "windows" if any(k in content_lower for k in [
            "evil-winrm", "winpeas", "powershell", "mimikatz", "kerberos",
            "active directory", ".exe", "certutil", "msfvenom.*windows",
            "\\\\", "c:\\", "whoami /priv",
        ]) else "linux"

        # Detect difficulty from content
        difficulty = "medium"
        for d in ["easy", "medium", "hard", "insane"]:
            if d in content_lower[:500]:
                difficulty = d
                break

        # Infer category from content
        category = _infer_category(content_lower)

        # ── Extract commands from code blocks (multiple formats) ────────
        code_blocks = re.findall(
            r'```(?:bash|sh|shell|console|powershell|cmd|zsh|kali)?\n(.*?)```',
            content, re.DOTALL | re.IGNORECASE,
        )

        # Also capture $ prompt lines outside code blocks
        prompt_lines = re.findall(
            r'(?:^|\n)\s*[\$#>❯]\s+(.+?)(?:\n|$)', content,
        )

        # Extract inline commands (expanded patterns)
        inline_cmds = re.findall(
            r'`([^`\n]{5,200})`',
            content, re.IGNORECASE,
        )
        # Filter inline to actual commands
        CMD_KEYWORDS = {
            "nmap", "gobuster", "ffuf", "curl", "python", "ssh", "nc ",
            "linpeas", "winpeas", "john", "hashcat", "hydra", "sqlmap",
            "wfuzz", "wget", "chmod", "find ", "cat ", "grep", "sudo",
            "nikto", "wpscan", "searchsploit", "enum4linux", "smbclient",
            "crackmapexec", "evil-winrm", "impacket", "bloodhound",
            "tshark", "strings", "getcap", "chisel", "ligolo", "feroxbuster",
            "responder", "certipy", "rubeus", "sharphound", "msfconsole",
            "msfvenom", "netcat", "socat", "dirsearch", "whatweb",
            "wkhtmltopdf", "burp", "xxd", "base64", "openssl",
            "rpcclient", "ldapsearch", "kerbrute", "getnpusers",
            "secretsdump", "psexec", "smbmap", "snmpwalk", "dig",
            "host ", "dnsrecon", "sublist3r", "amass", "rustscan",
            "masscan", "nbtscan", "showmount", "mount ", "ftp ",
            "telnet", "redis-cli", "mysql", "mssqlclient", "psql",
        }
        inline_cmds = [c for c in inline_cmds if any(k in c.lower() for k in CMD_KEYWORDS)]

        all_commands = []
        for block in code_blocks:
            for line in block.strip().split("\n"):
                line = line.strip()
                # Strip common prompt prefixes
                line = re.sub(r'^[\$#>❯]\s*', '', line)
                line = re.sub(r'^(root|kali|user|htb|www-data)@[\w.-]+[:#~]\s*', '', line)
                if line and _is_valid_command(line):
                    all_commands.append(line)
        # Filter prompt lines too
        all_commands.extend(l for l in prompt_lines if _is_valid_command(l))
        all_commands.extend(inline_cmds)  # Already filtered by CMD_KEYWORDS

        # Deduplicate while preserving order
        unique_cmds = []
        for cmd in all_commands:
            cmd_key = cmd.strip()[:80].lower()
            if cmd_key not in seen_commands and len(cmd_key) > 5:
                seen_commands.add(cmd_key)
                unique_cmds.append(cmd.strip())

        if not unique_cmds:
            continue

        # ── Extract section context for richer reasoning ────────────────
        sections = _extract_sections(content)

        # Generate examples from extracted commands
        target = _random_target()
        skipped = 0
        for i, cmd in enumerate(unique_cmds[:30]):  # Cap at 30 per writeup
            phase = _infer_phase(cmd)

            # ── CRITICAL: map raw command → valid template name ──────────
            template_name = _map_command_to_template(cmd, phase, os_type)
            if template_name is None:
                skipped += 1
                continue  # Skip commands that don't map to a template

            # Ensure template is in the available list for this phase
            available = _get_available_commands_for_phase(phase, os_type)
            if template_name not in available:
                available.append(template_name)
            random.shuffle(available)

            board = _random_discovery_board(phase, category, difficulty)
            board["web_paths"].append(f"/{box_name.lower().replace(' ', '-')}")
            if os_type == "windows":
                board["services"]["5985"] = "Microsoft HTTPAPI httpd 2.0 (WinRM)"
                board["services"]["445"] = "Microsoft Windows SMB"

            # Find surrounding context for reasoning
            section_ctx = _find_section_for_command(sections, cmd)
            reasoning = _build_walkthrough_reasoning(
                box_name, platform, phase, cmd, template_name,
                i, unique_cmds, section_ctx, os_type, board
            )

            prev_step = "Initial scan" if i == 0 else unique_cmds[i-1][:80]

            # Create mentor example with validated template command
            examples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPTS["mentor"]},
                    {
                        "role": "user",
                        "content": (
                            f"Select the best next command for {platform} box {box_name}.\n\n"
                            f"Target: {target}\nPhase: {phase}\nRole: offensive\n"
                            f"OS: {os_type}\nDifficulty: {difficulty}\n"
                            f"Stagnation: {min(i, 8)} steps\n"
                            f"Discovery board:\n{json.dumps(board, indent=2)}\n"
                            f"{platform} Box: {box_name}\n"
                            f"Previous step: {prev_step}\n"
                            f"Available templates: {', '.join(available)}"
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps({
                            "intent": f"Progress {phase.lower()} on {platform} {box_name} ({os_type}).",
                            "selected_command": template_name,
                            "parameters": {"target": target},
                            "reasoning": reasoning,
                            "expected_observation": _expected_observation(phase, cmd),
                            "risk": _assess_risk(phase, cmd),
                            "confidence": round(0.7 + random.random() * 0.25, 2),
                            "next_phase_hint": _next_phase_hint(phase, i, len(unique_cmds)),
                            "candidate_actions": [],
                        }),
                    },
                ],
                "schema_type": "smart_mentor_walkthrough",
            })

            # Also generate a command_select example from each walkthrough command
            if random.random() < 0.4:
                if available:
                    examples.append({
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPTS["command_select"]},
                            {
                                "role": "user",
                                "content": (
                                    f"Phase: {phase}\nTarget: {target}\n"
                                    f"OS: {os_type}\nPrevious: {prev_step}\n"
                                    f"Discovery board:\n{json.dumps(board, indent=2)}\n"
                                    f"Available: {', '.join(available)}"
                                ),
                            },
                            {
                                "role": "assistant",
                                "content": json.dumps({
                                    "selected_command": template_name,
                                    "confidence": round(0.7 + random.random() * 0.25, 2),
                                    "reason": f"{phase} stage on {os_type} — {template_name} targets known attack surface.",
                                }),
                            },
                        ],
                        "schema_type": "microchain_fast_local",
                    })
        if skipped:
            logger.debug(f"[WALKTHROUGH] {box_name}: skipped {skipped} unmappable commands")

    return examples


def _infer_category(content_lower: str) -> str:
    """Infer attack category from writeup content."""
    if any(k in content_lower for k in ["active directory", "kerberos", "bloodhound", "ldap", "domain controller"]):
        return "ad"
    if any(k in content_lower for k in ["lateral", "pivot", "proxychains", "chisel", "ligolo"]):
        return "lateral"
    if any(k in content_lower for k in ["winpeas", "powershell", "mimikatz", "windows privilege"]):
        return "privesc_windows"
    if any(k in content_lower for k in ["linpeas", "sudo -l", "suid", "linux privilege", "getcap"]):
        return "privesc_linux"
    if any(k in content_lower for k in ["sqli", "xss", "lfi", "rfi", "ssrf", "idor", "web shell", "deserialization"]):
        return "web"
    if any(k in content_lower for k in ["smb", "ftp", "snmp", "redis", "mysql", "mssql"]):
        return "service"
    return "full_box"


def _extract_sections(content: str) -> List[tuple]:
    """Extract (header, body) sections from markdown."""
    parts = re.split(r'^(#{1,3}\s+.+)$', content, flags=re.MULTILINE)
    sections = []
    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        sections.append((header, body))
    return sections


def _find_section_for_command(sections: List[tuple], cmd: str) -> str:
    """Find which section a command appears in."""
    cmd_short = cmd[:50].lower()
    for header, body in sections:
        if cmd_short in body.lower():
            return header.lstrip("#").strip()
    return ""


def _get_available_commands_for_phase(phase: str, os_type: str) -> List[str]:
    """Get likely commands for a given phase and OS."""
    phase_commands = {
        "RECON": ["nmap_full_tcp", "nmap_udp_top100", "rustscan_fast", "masscan_top1000", "ping_sweep"],
        "ENUMERATION": (
            ["gobuster_dir", "ffuf_vhost", "nikto_scan", "dirsearch", "feroxbuster", "whatweb"]
            if os_type == "linux" else
            ["enum4linux", "smbmap", "ldapsearch", "kerbrute_userenum", "rpcclient", "crackmapexec_smb"]
        ),
        "EXPLOITATION": (
            ["sqlmap_test", "hydra_ssh", "searchsploit_check", "msfconsole", "reverse_shell_bash"]
            if os_type == "linux" else
            ["evil-winrm", "psexec", "msfvenom_windows", "responder", "ntlm_relay"]
        ),
        "PRIVILEGE_ESCALATION": (
            ["linpeas", "sudo_check", "suid_find", "getcap_check", "crontab_check", "kernel_exploit"]
            if os_type == "linux" else
            ["winpeas", "whoami_priv", "juicypotato", "printspoofer", "certutil_download", "mimikatz"]
        ),
        "POST_EXPLOITATION": ["secretsdump", "hashdump", "credential_harvest", "lateral_move", "data_exfil"],
    }
    return phase_commands.get(phase, ["nmap_full_tcp", "gobuster_dir"])


def _build_reasoning(box_name: str, platform: str, phase: str, cmd: str,
                     step_idx: int, all_cmds: list, section_ctx: str, os_type: str) -> str:
    """Build detailed reasoning for a template-based command."""
    cmd_tool = cmd.split()[0] if cmd.split() else cmd[:20]
    ctx_part = f" In the '{section_ctx}' phase." if section_ctx else ""

    reasoning_templates = [
        f"{platform} {box_name} ({os_type}): at step {step_idx+1}, using {cmd_tool} for {phase.lower()}.{ctx_part} "
        f"This tool is effective for {'initial reconnaissance' if phase == 'RECON' else 'progressing the attack'} on this box type.",

        f"Analyzing {platform} box {box_name}: {cmd_tool} is the right choice at {phase.lower()} stage.{ctx_part} "
        f"Based on {'previous scan results' if step_idx > 0 else 'standard methodology'}, "
        f"this will {'reveal attack surface' if phase in ('RECON', 'ENUMERATION') else 'advance exploitation'}.",

        f"For {os_type} target {box_name}: {phase.lower()} requires {cmd_tool} here.{ctx_part} "
        f"{'Windows-specific tooling selected.' if os_type == 'windows' else 'Linux enumeration approach.'} "
        f"Step {step_idx+1} of the attack chain.",
    ]
    return random.choice(reasoning_templates)


def _build_walkthrough_reasoning(
    box_name: str, platform: str, phase: str, raw_cmd: str,
    template_name: str, step_idx: int, all_cmds: list,
    section_ctx: str, os_type: str, board: dict,
) -> str:
    """Build deep tactical reasoning for walkthrough-mined commands.

    Unlike template reasoning, this uses the raw command context + discovery
    board state to produce 2-3 sentence strategic justifications.
    """
    raw_tool = raw_cmd.split()[0] if raw_cmd.split() else raw_cmd[:20]
    num_ports = len(board.get("ports", []))
    has_creds = len(board.get("credentials", [])) > 0
    has_shells = len(board.get("shells", [])) > 0
    services = board.get("services", {})
    ctx_part = f" (writeup section: {section_ctx})" if section_ctx else ""

    # Phase-specific reasoning builders
    if phase == "RECON":
        return random.choice([
            f"Target has {num_ports} port{'s' if num_ports != 1 else ''} discovered so far. "
            f"Running {template_name} to fingerprint services and identify software versions — "
            f"version-specific CVEs are the fastest path to foothold on {os_type} boxes like {box_name}.{ctx_part}",

            f"Initial scan shows limited service information. {template_name} will map the full attack surface "
            f"before committing to an attack vector. On {platform} {box_name}, thorough recon prevents wasted "
            f"time on rabbit holes.{ctx_part}",

            f"Standard {os_type} recon methodology: {template_name} identifies service banners and versions. "
            f"With {num_ports} ports open, we need version data to prioritize which services to enumerate deeper. "
            f"{box_name} is rated {board.get('difficulty', 'medium')} — methodical approach required.{ctx_part}",
        ])

    if phase == "ENUMERATION":
        svc_list = ", ".join(f"{p}:{s}" for p, s in list(services.items())[:3])
        return random.choice([
            f"Services identified: {svc_list}. {template_name} targets {'web application directories' if 'gobuster' in template_name or 'dir' in template_name else 'service-specific misconfigurations'}. "
            f"On {box_name}, enumeration depth determines whether we find the intended attack path or get stuck.{ctx_part}",

            f"Deepening enumeration on {box_name} ({os_type}). With {num_ports} ports and services like {svc_list}, "
            f"{template_name} checks for {'exposed admin panels, backup files, and hidden endpoints' if phase == 'ENUMERATION' else 'lateral movement paths'}. "
            f"This is step {step_idx+1} — {'building the attack map' if step_idx < 4 else 'filling enumeration gaps'}.{ctx_part}",

            f"Discovery board shows {'credentials available' if has_creds else 'no credentials yet'} and {num_ports} open ports. "
            f"{template_name} specifically targets {'authentication endpoints' if has_creds else 'information disclosure vectors'} "
            f"that commonly lead to foothold on {os_type} hosts like {box_name}.{ctx_part}",
        ])

    if phase in ("EXPLOITATION", "PRIVILEGE_ESCALATION"):
        escalation = phase == "PRIVILEGE_ESCALATION"
        return random.choice([
            f"{'Escalation phase' if escalation else 'Exploitation phase'} on {box_name}: "
            f"{template_name} {'leverages discovered misconfig for root/SYSTEM' if escalation else 'converts enumeration findings into shell access'}. "
            f"{'Current shell is unprivileged — ' if has_shells else ''}"
            f"Raw command was `{raw_cmd[:60]}` — mapped to template for consistent execution.{ctx_part}",

            f"Attack vector identified on {box_name} ({os_type}). {template_name} "
            f"{'exploits the privilege escalation path' if escalation else 'delivers the initial payload'}. "
            f"With {'credentials in hand' if has_creds else 'no credentials'}, "
            f"this {'authenticated attack' if has_creds else 'unauthenticated vector'} is the optimal next step.{ctx_part}",

            f"Step {step_idx+1} in the {box_name} kill chain: {template_name} "
            f"{'moves from user to root/SYSTEM' if escalation else 'establishes initial foothold'}. "
            f"{'Board shows existing shells — this is the escalation attempt.' if has_shells else 'No shells yet — this is the foothold attempt.'} "
            f"Selected based on {os_type}-specific attack patterns.{ctx_part}",
        ])

    # POST_EXPLOITATION / LATERAL / default
    return (
        f"Post-exploitation on {box_name}: {template_name} "
        f"{'extracts credentials for lateral movement' if 'secret' in template_name or 'dump' in template_name else 'secures persistent access'}. "
        f"With {len(board.get('users', []))} known users and {'active shells' if has_shells else 'foothold established'}, "
        f"this maximizes the engagement value before closeout.{ctx_part}"
    )


def _expected_observation(phase: str, cmd: str) -> str:
    """Generate expected observation for a command."""
    observations = {
        "RECON": "Discover open ports, services, and potential attack vectors.",
        "ENUMERATION": "Identify specific vulnerabilities, directories, or misconfigurations.",
        "EXPLOITATION": "Gain initial foothold — shell access or code execution.",
        "PRIVILEGE_ESCALATION": "Escalate to root/SYSTEM privileges.",
        "POST_EXPLOITATION": "Extract credentials, secrets, or move laterally.",
    }
    return observations.get(phase, "New information to progress the engagement.")


def _assess_risk(phase: str, cmd: str) -> str:
    """Assess risk level of a command."""
    cmd_lower = cmd.lower()
    if any(k in cmd_lower for k in ["exploit", "reverse", "shell", "msfconsole", "payload"]):
        return "high"
    if any(k in cmd_lower for k in ["hydra", "sqlmap", "brute", "inject"]):
        return "high"
    if any(k in cmd_lower for k in ["nmap", "ping", "dig", "whois"]):
        return "low"
    return "medium"


def _next_phase_hint(phase: str, step_idx: int, total_steps: int) -> str:
    """Suggest next phase based on progress."""
    phase_order = ["RECON", "ENUMERATION", "EXPLOITATION", "PRIVILEGE_ESCALATION", "POST_EXPLOITATION"]
    try:
        idx = phase_order.index(phase)
        progress = step_idx / max(total_steps, 1)
        if progress > 0.7 and idx < len(phase_order) - 1:
            return f"Consider moving to {phase_order[idx + 1].lower()}."
        return f"Continue {phase.lower()} — more enumeration may reveal additional vectors."
    except ValueError:
        return f"Continue {phase.lower()} phase."


_TEMPLATE_NAMES: set[str] = {
    "accesschk_services", "bloodhound_python", "certipy_find", "certipy_req",
    "chisel_client", "chisel_server", "cmd_inject_pipe", "cmd_inject_semicolon",
    "cme_smb_shares", "crackmapexec_pth", "crackmapexec_smb", "crackmapexec_winrm",
    "cred_reuse_ssh", "credential_dump", "cron_backdoor", "cron_check", "crontab_check",
    "curl_web_path", "dirsearch", "docker_privesc", "docker_sock_escape", "droopescan",
    "drupalgeddon2", "enum4linux", "enum4linux_full", "erlang_cookie_extract",
    "erlang_otp_rce", "evil_winrm", "feroxbuster", "ffuf_fuzz", "ffuf_vhost",
    "find_capabilities", "find_sgid", "find_suid", "find_writable_etc", "ftp_anonymous",
    "getcap_check", "gobuster_dir", "gobuster_vhost", "gpp_decrypt", "hashcat_krb5",
    "heartbleed_exploit", "hydra_ssh", "idor_curl_range", "idor_download_file",
    "impacket_psexec", "impacket_pth_psexec", "impacket_secretsdump",
    "jwt_crack_secret", "jwt_none_attack", "kerbrute_userenum", "kernel_exploit",
    "kernel_exploit_check", "ldapsearch", "ldapsearch_base", "ldapsearch_users",
    "lfi_etc_passwd", "lfi_log_poison", "lfi_php_filter", "lfi_ssh_key", "linpeas",
    "log4shell_detect", "lxd_privesc", "masscan_top1000", "mimikatz_dcsync",
    "mimikatz_logonpasswords", "msfconsole", "msfconsole_exploit", "msfvenom_payload",
    "msfvenom_windows", "mssql_login", "mysql_login", "nfs_mount", "nfs_mount_root",
    "nikto_scan", "nmap_full_tcp", "nmap_pivot", "nmap_quick_scan",
    "nmap_service_version", "nmap_udp_scan", "nmap_udp_top100", "nmap_vuln_scan",
    "nosqli_login_bypass", "ntlm_relay", "ntlmrelayx", "onesixtyone",
    "pcap_strings_extract", "phpggc_laravel", "ping_sweep", "pivot_scan", "powerup",
    "privesc_cap_setuid_python", "privesc_getcap", "privesc_sudo_l", "proxychains_scan",
    "psexec", "pspy", "psql_default_creds", "psql_rce", "read_root_flag",
    "read_user_flag", "redis_cli", "responder", "reverse_shell_bash", "revshell_bash",
    "revshell_powershell", "revshell_python", "rfi_php_shell", "root_shell_confirm",
    "rpcclient", "rpcinfo_check", "rustscan_fast", "samba_usermap_exploit",
    "searchsploit", "searchsploit_check", "shellshock_cgi", "showmount",
    "showmount_enum", "smbclient_auth", "smbclient_get_file", "smbclient_null_list",
    "smbmap", "smbmap_shares", "snmpwalk", "sqlmap_get", "sqlmap_post", "sqlmap_shell",
    "sqlmap_test", "ssh_key_login", "ssh_key_plant", "ssh_login", "ssh_tunnel_dynamic",
    "ssh_tunnel_local", "ssrf_cloud_metadata", "ssrf_internal_admin",
    "ssrf_localhost_scan", "ssti_detect_jinja2", "ssti_detect_twig",
    "ssti_exploit_jinja2", "ssti_exploit_twig", "sudo_check", "sudo_list", "suid_find",
    "systeminfo", "tomcat_cred_test", "tomcat_war_deploy", "tty_stabilize",
    "unrealircd_exploit", "upload_htaccess", "upload_php_double_ext",
    "upload_php_magic_bytes", "vsftpd_exploit", "webshell_cmd", "whatweb", "whoami_all",
    "windows_exploit_suggester", "winpeas", "wpscan", "writable_etc_passwd",
    "xxe_file_read", "ysoserial_java",
}

# Raw command tool → template name mapping for walkthrough mining
_TOOL_TO_TEMPLATE: dict[str, str] = {
    "nmap": "nmap_service_version", "rustscan": "rustscan_fast", "masscan": "masscan_top1000",
    "whatweb": "whatweb", "nikto": "nikto_scan", "gobuster": "gobuster_dir",
    "dirsearch": "dirsearch", "feroxbuster": "feroxbuster", "ffuf": "ffuf_fuzz",
    "wpscan": "wpscan", "droopescan": "droopescan", "enum4linux": "enum4linux",
    "smbclient": "smbclient_null_list", "smbmap": "smbmap_shares", "rpcclient": "rpcclient",
    "crackmapexec": "crackmapexec_smb", "evil-winrm": "evil_winrm",
    "impacket-psexec": "impacket_psexec", "psexec.py": "impacket_psexec",
    "secretsdump": "impacket_secretsdump", "secretsdump.py": "impacket_secretsdump",
    "GetNPUsers.py": "kerbrute_userenum", "kerbrute": "kerbrute_userenum",
    "bloodhound-python": "bloodhound_python", "bloodhound": "bloodhound_python",
    "ldapsearch": "ldapsearch", "sqlmap": "sqlmap_test", "hydra": "hydra_ssh",
    "searchsploit": "searchsploit_check", "msfconsole": "msfconsole_exploit",
    "msfvenom": "msfvenom_payload", "curl": "curl_web_path", "wget": "curl_web_path",
    "ftp": "ftp_anonymous", "ssh": "ssh_login", "mysql": "mysql_login",
    "psql": "psql_default_creds", "mssqlclient": "mssql_login",
    "mssqlclient.py": "mssql_login", "redis-cli": "redis_cli",
    "snmpwalk": "snmpwalk", "onesixtyone": "onesixtyone", "showmount": "showmount_enum",
    "mount": "nfs_mount", "linpeas": "linpeas", "winpeas": "winpeas",
    "pspy": "pspy", "chisel": "chisel_client", "proxychains": "proxychains_scan",
    "certutil": "certutil_download", "mimikatz": "mimikatz_logonpasswords",
    "hashcat": "hashcat_krb5", "john": "hashcat_krb5",
    "python3": "revshell_python", "python": "revshell_python",
    "nc": "reverse_shell_bash", "bash": "revshell_bash",
    "cat": "read_user_flag", "sudo": "sudo_check", "find": "find_suid",
    "getcap": "getcap_check", "responder": "responder",
    "certipy": "certipy_find", "dig": "nmap_service_version",
}


def _map_command_to_template(raw_cmd: str, phase: str, os_type: str) -> str | None:
    """Map a raw shell command from a writeup to the nearest valid template name.

    Returns the template name or None if no mapping found.
    """
    if not raw_cmd or len(raw_cmd) < 3:
        return None

    # Extract the base tool name
    first_word = raw_cmd.split()[0].strip()
    tool_base = first_word.rsplit("/", 1)[-1]  # strip path prefix like /usr/bin/

    # Direct match
    if tool_base in _TOOL_TO_TEMPLATE:
        return _TOOL_TO_TEMPLATE[tool_base]

    # Try lowercase
    if tool_base.lower() in _TOOL_TO_TEMPLATE:
        return _TOOL_TO_TEMPLATE[tool_base.lower()]

    # Check if the tool itself is a valid template name
    tool_snake = re.sub(r'[^a-z0-9]', '_', tool_base.lower()).strip('_')
    if tool_snake in _TEMPLATE_NAMES:
        return tool_snake

    # Content-based heuristics for ambiguous commands
    cmd_lower = raw_cmd.lower()
    if "reverse" in cmd_lower or "shell" in cmd_lower or "nc -" in cmd_lower:
        return "revshell_bash" if os_type == "linux" else "revshell_powershell"
    if "sudo -l" in cmd_lower or "sudo -" in cmd_lower:
        return "sudo_check"
    if "find / -perm" in cmd_lower or "find / -type" in cmd_lower:
        return "find_suid"
    if "getcap" in cmd_lower:
        return "getcap_check"
    if "suid" in cmd_lower:
        return "suid_find"
    if "docker" in cmd_lower:
        return "docker_privesc"
    if "lxd" in cmd_lower or "lxc" in cmd_lower:
        return "lxd_privesc"
    if "cron" in cmd_lower:
        return "crontab_check"
    if "whoami" in cmd_lower or "id" == tool_base:
        return "whoami_all"
    if "systeminfo" in cmd_lower:
        return "systeminfo"

    # No valid mapping — skip this command
    return None


def _infer_phase(cmd: str) -> str:
    """Infer attack phase from command string."""
    cmd_lower = cmd.lower()
    if any(k in cmd_lower for k in ["nmap", "masscan", "whatweb", "ping"]):
        return "RECON"
    if any(k in cmd_lower for k in ["gobuster", "ffuf", "nikto", "enum4linux", "smbclient", "ldapsearch", "wpscan", "searchsploit", "dirsearch"]):
        return "ENUMERATION"
    if any(k in cmd_lower for k in ["sqlmap", "hydra", "exploit", "msfconsole", "reverse", "shell", "nc -", "revshell", "curl.*-d", "upload"]):
        return "EXPLOITATION"
    if any(k in cmd_lower for k in ["linpeas", "sudo -l", "find.*suid", "getcap", "winpeas", "privesc", "kernel"]):
        return "PRIVILEGE_ESCALATION"
    if any(k in cmd_lower for k in ["mimikatz", "hashdump", "secretsdump", "credential"]):
        return "POST_EXPLOITATION"
    if any(k in cmd_lower for k in ["pivot", "tunnel", "chisel", "proxychains", "lateral"]):
        return "LATERAL_MOVEMENT"
    return "ENUMERATION"  # default


# ═══════════════════════════════════════════════════════════════════════════════
# CLOUD LLM ENHANCEMENT — Use Groq 70B to generate expert-quality reasoning
# ═══════════════════════════════════════════════════════════════════════════════

class CloudEnhancer:
    """Uses cloud LLM (Groq/OpenRouter/Together free tier) to generate
    high-quality reasoning for training examples instead of templates.

    This is essentially knowledge distillation: the 70B teacher generates
    what the 4B student should learn to produce.
    """

    _PROVIDERS = [
        {
            "name": "groq",
            "base_url": "https://api.groq.com/openai/v1",
            "model": "llama-3.3-70b-versatile",
            "env_key": "GROQ_API_KEY",
            "rpm": 30,
        },
        {
            "name": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "model": "meta-llama/llama-3.3-70b-instruct",
            "env_key": "OPENROUTER_API_KEY",
            "rpm": 20,
        },
        {
            "name": "together",
            "base_url": "https://api.together.xyz/v1",
            "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "env_key": "TOGETHER_API_KEY",
            "rpm": 20,
        },
    ]

    def __init__(self) -> None:
        self._clients: List[Dict[str, Any]] = []
        self._current_idx = 0
        self._call_count = 0
        self._error_count = 0
        self._init_providers()

    def _init_providers(self) -> None:
        """Initialize available cloud providers."""
        try:
            from openai import OpenAI
        except ImportError:
            print("[CLOUD] openai package not installed — cloud enhance disabled")
            return

        for p in self._PROVIDERS:
            api_key = os.getenv(p["env_key"], "")
            if api_key:
                client = OpenAI(api_key=api_key, base_url=p["base_url"])
                self._clients.append({
                    "client": client,
                    "model": p["model"],
                    "name": p["name"],
                    "rpm": p["rpm"],
                    "errors": 0,
                    "last_call_time": 0.0,
                })
                print(f"  [CLOUD] {p['name']} available ({p['model']})")

        if self._clients:
            print(f"  [CLOUD] {len(self._clients)} providers ready — "
                  f"combined capacity ~{sum(c['rpm'] for c in self._clients)} req/min")
        else:
            print("  [CLOUD] No API keys found — set GROQ_API_KEY, OPENROUTER_API_KEY, or TOGETHER_API_KEY")

    @property
    def available(self) -> bool:
        return len(self._clients) > 0

    def _pick_ready_provider(self) -> int:
        """Pick the provider whose rate-limit cooldown has expired first.

        Round-robins across all providers, sleeping only for the shortest wait.
        """
        if len(self._clients) == 1:
            return 0
        now = time.time()
        best_idx = self._current_idx
        best_wait = float("inf")
        for i in range(len(self._clients)):
            idx = (self._current_idx + i) % len(self._clients)
            c = self._clients[idx]
            min_interval = 60.0 / c["rpm"]
            wait = max(0.0, min_interval - (now - c["last_call_time"]))
            if wait < best_wait:
                best_wait = wait
                best_idx = idx
            if wait == 0.0:
                break  # no need to check further
        if best_wait > 0:
            time.sleep(best_wait)
        return best_idx

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 800) -> Optional[str]:
        """Generate a response from cloud LLM with round-robin failover.

        Picks whichever provider is ready soonest, rotates on every call.
        Returns None if all providers fail.
        """
        if not self._clients:
            return None

        tried: set = set()
        while len(tried) < len(self._clients):
            idx = self._pick_ready_provider()
            if idx in tried:
                # All remaining have been tried this round
                idx = next(
                    (i for i in range(len(self._clients)) if i not in tried),
                    idx,
                )
            current = self._clients[idx]
            tried.add(idx)

            try:
                resp = current["client"].chat.completions.create(
                    model=current["model"],
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.7,
                    max_tokens=max_tokens,
                )
                current["last_call_time"] = time.time()
                self._call_count += 1
                # Advance round-robin for next call
                self._current_idx = (idx + 1) % len(self._clients)
                content = resp.choices[0].message.content
                if content:
                    return content.strip()

            except Exception as e:
                self._error_count += 1
                current["errors"] += 1
                logger.warning(f"[CLOUD] {current['name']} error: {e}")

        return None

    def enhance_mentor_reasoning(
        self, target: str, phase: str, command: str, board: Dict,
        chain_name: str = "", box_info: Dict = None,
    ) -> Optional[Dict]:
        """Use cloud LLM to generate expert mentor reasoning for a step."""
        box_context = ""
        if box_info:
            box_context = f"\nBox: {chain_name} ({box_info.get('os', 'Unknown')} — {box_info.get('services', '')})"

        prompt = (
            f"You are an expert pentester advising on HTB/THM box exploitation.\n"
            f"Generate a detailed tactical recommendation for the next step.\n\n"
            f"Target: {target}\nPhase: {phase}\nCommand to recommend: {command}"
            f"{box_context}\n"
            f"Current discovery state:\n{json.dumps(board, indent=2)}\n\n"
            f"Respond with ONLY valid JSON (no markdown):\n"
            f'{{"intent": "1-sentence goal", "selected_command": "{command}", '
            f'"parameters": {{"target": "{target}"}}, '
            f'"reasoning": "2-3 sentences explaining WHY this command, what you expect to find, '
            f'and how it connects to the overall attack path", '
            f'"expected_observation": "what specific output to look for", '
            f'"risk": "low/medium/high", "confidence": 0.0-1.0, '
            f'"next_phase_hint": "what to do after this step"}}'
        )

        result = self.generate(SYSTEM_PROMPTS["mentor"], prompt, max_tokens=600)
        if not result:
            return None

        # Validate JSON
        try:
            parsed = json.loads(result)
            if "selected_command" in parsed and "reasoning" in parsed:
                # Ensure command matches
                parsed["selected_command"] = command
                return parsed
        except json.JSONDecodeError:
            # Try to extract JSON from response
            m = re.search(r'\{[^{}]*"reasoning"[^{}]*\}', result, re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group())
                    parsed["selected_command"] = command
                    return parsed
                except json.JSONDecodeError:
                    pass

        return None

    def enhance_strategic_plan(
        self, target: str, chain: Dict, board: Dict,
    ) -> Optional[Dict]:
        """Use cloud LLM to generate expert strategic plan."""
        steps_desc = "\n".join(
            f"  {i+1}. [{phase}] {cmd}: {reason}"
            for i, (phase, cmd, reason) in enumerate(chain["steps"])
        )
        box_info = chain.get("box_info", {})
        box_context = ""
        if box_info:
            box_context = f"\nOS: {box_info.get('os', 'Unknown')}, Services: {box_info.get('services', '')}"

        prompt = (
            f"You are a senior penetration tester creating a strategic attack plan.\n"
            f"Analyze the target and create a detailed, realistic plan.\n\n"
            f"Target: {target}{box_context}\n"
            f"Attack chain: {chain['name']} — {chain['description']}\n"
            f"Difficulty: {chain['difficulty']}\n"
            f"Known steps:\n{steps_desc}\n"
            f"Discovery state:\n{json.dumps(board, indent=2)}\n\n"
            f"Respond with ONLY valid JSON (no markdown):\n"
            f'{{"situation_assessment": "detailed analysis of current state and attack surface", '
            f'"recommended_approach": "primary strategy with rationale", '
            f'"step_plan": [{{...}}], "risk_assessment": "...", '
            f'"confidence": 0.0-1.0, "alternative_approaches": ["..."]}}'
        )

        result = self.generate(SYSTEM_PROMPTS["strategic"], prompt, max_tokens=1000)
        if not result:
            return None

        try:
            parsed = json.loads(result)
            if "situation_assessment" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass

        return None

    def enhance_dpo_pair(
        self, target: str, phase: str, chosen_cmd: str, rejected_cmd: str,
        board: Dict, chain_name: str = "",
    ) -> Optional[Tuple[str, str]]:
        """Use cloud LLM to generate expert chosen + rejected reasoning."""
        prompt = (
            f"You are an expert pentester evaluating two possible next actions.\n"
            f"Target: {target}\nPhase: {phase}\n"
            f"Discovery state:\n{json.dumps(board, indent=2)}\n\n"
            f"GOOD choice: {chosen_cmd}\n"
            f"BAD choice: {rejected_cmd}\n\n"
            f"Generate TWO JSON responses. First the GOOD reasoning (expert), "
            f"then the BAD reasoning (novice mistake).\n\n"
            f"GOOD (valid JSON only):\n"
            f'{{"command": "{chosen_cmd}", "template_name": "{chosen_cmd}", '
            f'"reasoning": "expert explanation of why this is correct", "score": 0.85}}\n\n'
            f"BAD (valid JSON only):\n"
            f'{{"command": "{rejected_cmd}", "template_name": "{rejected_cmd}", '
            f'"reasoning": "novice-sounding explanation showing poor judgment", "score": 0.3}}'
        )

        result = self.generate(SYSTEM_PROMPTS["command_select"], prompt, max_tokens=600)
        if not result:
            return None

        # Extract the two JSON objects
        jsons = re.findall(r'\{[^{}]*"reasoning"[^{}]*\}', result)
        if len(jsons) >= 2:
            try:
                good = json.loads(jsons[0])
                bad = json.loads(jsons[1])
                good["command"] = good["template_name"] = chosen_cmd
                bad["command"] = bad["template_name"] = rejected_cmd
                return (json.dumps(good), json.dumps(bad))
            except json.JSONDecodeError:
                pass

        return None

    def get_stats(self) -> Dict[str, int]:
        return {
            "calls": self._call_count,
            "errors": self._error_count,
            "providers": len(self._clients),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# ADDITIONAL OUTPUT PARSING EXAMPLES — SMB, searchsploit, WinPEAS, etc.
# ═══════════════════════════════════════════════════════════════════════════════

_SMBCLIENT_OUTPUTS = [
    """    Sharename       Type      Comment
    ---------       ----      -------
    print$          Disk      Printer Drivers
    users           Disk
    IPC$            IPC       IPC Service (Samba 4.7.6-Ubuntu)
    Development     Disk      Developer share
    Replication     Disk
    ADMIN$          IPC       Remote Admin
    SYSVOL          Disk      Logon server share
SMB1 disabled -- no workgroup available""",
    """    Sharename       Type      Comment
    ---------       ----      -------
    tmp             Disk      oh nance!
    opt             Disk
    IPC$            IPC       IPC Service (lame server (Samba 3.0.20-Debian))
    ADMIN$          IPC       IPC Service""",
]

_SEARCHSPLOIT_OUTPUTS = [
    """-------------------------------------------------------------------------- ---------------------------------
 Exploit Title                                                            |  Path
-------------------------------------------------------------------------- ---------------------------------
Samba 3.0.20 < 3.0.25rc3 - 'Username' map script' Command Execution (Met | unix/remote/16320.rb
Samba 3.0.20 < 3.0.25rc3 - 'Username' map script' Command Execution      | multiple/remote/16859.rb
-------------------------------------------------------------------------- ---------------------------------""",
    """-------------------------------------------------------------------------- ---------------------------------
 Exploit Title                                                            |  Path
-------------------------------------------------------------------------- ---------------------------------
HttpFileServer 2.3.x - Remote Command Execution (1)                       | windows/remote/34668.txt
HttpFileServer 2.3.x - Remote Command Execution (2)                       | windows/remote/39161.py
Rejetto HttpFileServer 2.3.x - Remote Command Execution (3)               | windows/webapps/49125.py
-------------------------------------------------------------------------- ---------------------------------""",
    """-------------------------------------------------------------------------- ---------------------------------
 Exploit Title                                                            |  Path
-------------------------------------------------------------------------- ---------------------------------
nostromo 1.9.6 - Remote Code Execution                                    | multiple/remote/47837.py
nostromo nhttpd 1.9.3 - Directory Traversal Remote Command Execution       | linux/remote/35466.sh
-------------------------------------------------------------------------- ---------------------------------""",
]

_WINPEAS_OUTPUTS = [
    """====================================( Users Information )=====================================
  [+] Users
   [?] Check if you have some admin equivalent privileges https://book.hacktricks.xyz/windows/windows-local-privilege-escalation
    Current user: svc-alfresco
    Current groups: Domain Users, Exchange Windows Permissions, Account Operators

====================================( Services Information )===================================
  [+] Interesting Services -nonass. processes-
    [?] Check if you can overwrite some service binary or perform a DLL hijack
    Apache2.4(Apache Software Foundation - Apache/2.4) - C:\\xampp\\apache\\bin\\httpd.exe - Auto - Running

====================================( Interesting Files and Registry )=========================
  [+] Putty Sessions
    SessionName: DC01

  [+] AutoLogon credentials
    DefaultDomainName: EGOTISTICALBANK
    DefaultUserName: svc_loanmanager
    DefaultPassword: Moneymakestheworldgoround!""",
]

_HYDRA_OUTPUTS = [
    """Hydra v9.4 (c) 2022 by van Hauser/THC & David Maciejak

Hydra (https://github.com/vanhauser-thc/hydra) starting at 2024-01-15 10:30:00
[WARNING] Many SSH configurations limit the number of parallel tasks
[DATA] max 16 tasks per 1 server, overall 16 tasks
[DATA] attacking ssh://10.10.10.100:22/
[22][ssh] host: 10.10.10.100   login: admin   password: monkey123
1 of 1 target successfully completed, 1 valid password found""",
]

_HASHCAT_OUTPUTS = [
    """hashcat (v6.2.6) starting
$krb5tgs$23$*Administrator$ACTIVE.HTB$active.htb/Administrator*$...:Ticketmaster1968

Session..........: hashcat
Status...........: Cracked
Hash.Mode........: 13100 (Kerberos 5, etype 23, TGS-REP)
Hash.Target......: $krb5tgs$23$*Administrator$ACTIVE.HTB*$...
Time.Started.....: Mon Jan 15 10:30:00 2024
Speed.#1.........:  1234.5 kH/s
Recovered........: 1/1 (100.00%)""",
    """hashcat (v6.2.6) starting
$krb5asrep$23$svc-alfresco@HTB.LOCAL:...:s3rvice

Session..........: hashcat
Status...........: Cracked
Hash.Mode........: 18200 (Kerberos 5, etype 23, AS-REP)
Recovered........: 1/1 (100.00%)""",
]


def _parse_smbclient_expected(output: str) -> Dict:
    shares = []
    for m in re.finditer(r'^\s+(\S+)\s+Disk\s*(.*?)$', output, re.MULTILINE):
        shares.append({"name": m.group(1), "comment": m.group(2).strip()})
    samba_ver = ""
    m = re.search(r'Samba (\S+)', output)
    if m:
        samba_ver = m.group(1)
    return {"shares": shares, "samba_version": samba_ver, "success": True}


def _parse_searchsploit_expected(output: str) -> Dict:
    exploits = []
    for m in re.finditer(r'^\s*(.+?)\s*\|\s*(\S+)\s*$', output, re.MULTILINE):
        title = m.group(1).strip()
        path = m.group(2).strip()
        if title and not title.startswith("---") and not title.startswith("Exploit"):
            exploits.append({"title": title, "path": path})
    return {"exploits": exploits, "count": len(exploits), "success": True}


def _parse_winpeas_expected(output: str) -> Dict:
    result: Dict[str, Any] = {"success": True, "artifacts": {}}
    # AutoLogon
    m = re.search(r'DefaultPassword:\s*(\S+)', output)
    if m:
        result["artifacts"]["autologon_password"] = m.group(1)
    m = re.search(r'DefaultUserName:\s*(\S+)', output)
    if m:
        result["artifacts"]["autologon_user"] = m.group(1)
    # Groups
    groups = re.findall(r'Current groups:\s*(.+)', output)
    if groups:
        result["artifacts"]["user_groups"] = [g.strip() for g in groups[0].split(",")]
    # Services
    services = re.findall(r'(\w+)\(.*?\)\s*-\s*(C:\\[^\s]+)', output)
    if services:
        result["artifacts"]["services"] = [{"name": s[0], "path": s[1]} for s in services]
    return result


def _parse_hydra_expected(output: str) -> Dict:
    creds = []
    for m in re.finditer(r'\[\d+\]\[\w+\]\s*host:\s*(\S+)\s+login:\s*(\S+)\s+password:\s*(\S+)', output):
        creds.append({"host": m.group(1), "username": m.group(2), "password": m.group(3)})
    return {"credentials": creds, "success": len(creds) > 0}


def _parse_hashcat_expected(output: str) -> Dict:
    result: Dict[str, Any] = {"success": False}
    if "Cracked" in output:
        result["success"] = True
        # Extract cracked password (after last colon on the hash line)
        for line in output.split("\n"):
            if "$krb5" in line and ":" in line:
                parts = line.rsplit(":", 1)
                if len(parts) == 2:
                    result["cracked_password"] = parts[1].strip()
                    break
        m = re.search(r'Hash\.Mode.*?:\s*\d+\s*\((.+?)\)', output)
        if m:
            result["hash_type"] = m.group(1)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def generate_all(
    output_dir: Path,
    sft_count: int = 15000,
    strategic_count: int = 3000,
    dpo_count: int = 2500,
    output_parse_count: int = 2000,
    seed: int = 42,
    cloud_enhance: bool = False,
    cloud_enhance_count: int = 3000,
) -> Dict[str, int]:
    """Generate all training data.

    Args:
        cloud_enhance: Use cloud LLM (Groq 70B) to generate expert reasoning
                       for a subset of examples. Much higher quality but slower.
        cloud_enhance_count: How many examples to enhance with cloud LLM.
                             Remaining use template reasoning.
    """
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats: Dict[str, int] = {}
    schema_dist: Counter = Counter()

    # ── 0. Initialize cloud enhancer if requested ─────────────────────────
    enhancer: Optional[CloudEnhancer] = None
    cloud_enhanced = 0
    if cloud_enhance:
        print("[CLOUD] Initializing cloud LLM enhancer (Groq/OpenRouter/Together 70B)...")
        enhancer = CloudEnhancer()
        if not enhancer.available:
            print("[CLOUD] No cloud providers available — falling back to template generation")
            enhancer = None

    # ── 1. SFT Data (command_select + mentor + phase_classify) ────────────
    print(f"Generating {sft_count} SFT examples from {len(ALL_CHAINS)} attack chains...")
    # Prioritize real box chains for cloud enhancement (they have richer context)
    real_box_chains = [c for c in ALL_CHAINS if c.get("box_info")]
    sft_examples: List[Dict] = []

    for i in range(sft_count):
        # For cloud enhancement, preferentially use real box chains
        if enhancer and cloud_enhanced < cloud_enhance_count and real_box_chains:
            chain = random.choice(real_box_chains)
        else:
            chain = random.choice(ALL_CHAINS)

        target = _random_target()
        step_idx = random.randint(0, len(chain["steps"]) - 1)
        phase, command, reasoning = chain["steps"][step_idx]
        board = _random_discovery_board(phase, chain["category"], chain["difficulty"])
        available = _get_available_commands(chain, step_idx)

        # Try cloud enhancement for mentor examples (highest value)
        if enhancer and cloud_enhanced < cloud_enhance_count and random.random() < 0.6:
            cloud_result = enhancer.enhance_mentor_reasoning(
                target, phase, command, board,
                chain_name=chain["name"],
                box_info=chain.get("box_info"),
            )
            if cloud_result:
                # Build cloud-enhanced mentor example
                ex = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPTS["mentor"]},
                        {
                            "role": "user",
                            "content": (
                                f"Select the best next command. selected_command MUST be a template_name from available.\n\n"
                                f"Target: {target}\nPhase: {phase}\nRole: offensive\n"
                                f"Stagnation: {random.randint(0, 8)} steps\n"
                                f"Discovery board:\n{json.dumps(board, indent=2)}\n"
                                f"Recent commands: {', '.join(random.sample(available, min(3, len(available))))}\n"
                                f"Available templates: {', '.join(available)}"
                            ),
                        },
                        {"role": "assistant", "content": json.dumps(cloud_result)},
                    ],
                    "schema_type": "smart_mentor_cloud",
                }
                sft_examples.append(ex)
                schema_dist["smart_mentor_cloud"] += 1
                cloud_enhanced += 1
                if cloud_enhanced % 100 == 0:
                    print(f"  [CLOUD] Enhanced {cloud_enhanced}/{cloud_enhance_count} examples "
                          f"({enhancer.get_stats()['errors']} errors)")
                continue

        # Template-based generation (fallback or non-cloud mode)
        r = random.random()
        if r < 0.35:
            ex = gen_command_select(target, phase, command, reasoning, board, available)
        elif r < 0.70:
            next_hint = ""
            if step_idx < len(chain["steps"]) - 1:
                next_phase = chain["steps"][step_idx + 1][0]
                next_hint = f"Transition to {next_phase.lower().replace('_', ' ')} after this step."
            ex = gen_mentor(target, phase, command, reasoning, board, available, next_hint)
        else:
            ex = gen_phase_classify(target, phase, board)

        sft_examples.append(ex)
        schema_dist[ex["schema_type"]] += 1

    # ── 2. Mine HTB/THM walkthroughs (all directories) ─────────────────
    walkthrough_dirs = [
        Path("data/htb_walkthroughs"),        # Our curated writeups
        Path("/tmp/hackthebox-writeups"),      # Scraped repos
        Path("/tmp/Writeups"),
        Path("/tmp/HackTheBox-Writeups"),
        Path("/tmp/htb-write-up"),
        Path("/tmp/OSCP"),                    # OSCP methodology and examples
        Path("/tmp/PayloadsAllTheThings"),    # Attack techniques reference
    ]
    for walkthrough_dir in walkthrough_dirs:
        if walkthrough_dir.exists():
            print(f"Mining walkthroughs from {walkthrough_dir}...")
            walkthrough_examples = mine_walkthroughs(walkthrough_dir)
            sft_examples.extend(walkthrough_examples)
            schema_dist["smart_mentor_walkthrough"] += len(walkthrough_examples)
            print(f"  Mined {len(walkthrough_examples)} examples from {walkthrough_dir.name}")

    # Write SFT
    sft_path = output_dir / "htb_sft_v5.jsonl"
    random.shuffle(sft_examples)
    with open(sft_path, "w") as f:
        for ex in sft_examples:
            f.write(json.dumps(ex) + "\n")
    stats["sft"] = len(sft_examples)
    print(f"  SFT: {len(sft_examples)} examples → {sft_path}")

    # ── 3. Strategic planning data (cloud-enhanced when available) ─────────
    print(f"Generating {strategic_count} strategic planning examples...")
    strategic_examples: List[Dict] = []
    cloud_strategic = 0
    for i in range(strategic_count):
        chain = random.choice(ALL_CHAINS)
        target = _random_target()
        board = _random_discovery_board(chain["steps"][0][0], chain["category"], chain["difficulty"])

        # Try cloud enhancement for strategic plans (very high value)
        if enhancer and cloud_strategic < min(500, cloud_enhance_count // 3):
            cloud_plan = enhancer.enhance_strategic_plan(target, chain, board)
            if cloud_plan:
                ex = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPTS["strategic"]},
                        {
                            "role": "user",
                            "content": (
                                f"Analyze target state and create an attack plan.\n\n"
                                f"Target: {target}\nPhase: {chain['steps'][0][0]}\n"
                                f"Discovery board:\n{json.dumps(board, indent=2)}\n"
                                f"Attack chain type: {chain['name']}\n"
                                f"Difficulty: {chain['difficulty']}\n"
                                f"Objective: Achieve maximum privilege on target."
                            ),
                        },
                        {"role": "assistant", "content": json.dumps(cloud_plan)},
                    ],
                    "schema_type": "strategic_plan_cloud",
                }
                strategic_examples.append(ex)
                cloud_strategic += 1
                continue

        ex = gen_strategic(target, chain, board)
        strategic_examples.append(ex)

    strat_path = output_dir / "htb_strategic_v5.jsonl"
    with open(strat_path, "w") as f:
        for ex in strategic_examples:
            f.write(json.dumps(ex) + "\n")
    stats["strategic"] = len(strategic_examples)
    if cloud_strategic:
        print(f"  Strategic: {len(strategic_examples)} examples ({cloud_strategic} cloud-enhanced) → {strat_path}")
    else:
        print(f"  Strategic: {len(strategic_examples)} examples → {strat_path}")

    # ── 4. Output parsing data (expanded with SMB, searchsploit, etc.) ────
    print(f"Generating {output_parse_count} output parsing examples...")
    parse_examples = gen_output_parse_examples_v2(output_parse_count, seed)
    parse_path = output_dir / "htb_output_parse_v5.jsonl"
    with open(parse_path, "w") as f:
        for ex in parse_examples:
            f.write(json.dumps(ex) + "\n")
    stats["output_parse"] = len(parse_examples)
    print(f"  Output Parse: {len(parse_examples)} examples → {parse_path}")

    # ── 5. DPO preference pairs (cloud-enhanced when available) ───────────
    print(f"Generating {dpo_count} DPO preference pairs...")
    dpo_examples: List[Dict] = []
    cloud_dpo = 0
    for i in range(dpo_count):
        chain = random.choice(ALL_CHAINS)
        target = _random_target()
        step_idx = random.randint(0, len(chain["steps"]) - 1)
        phase, command, reasoning = chain["steps"][step_idx]
        board = _random_discovery_board(phase, chain["category"], chain["difficulty"])
        available = _get_available_commands(chain, step_idx)

        # Try cloud enhancement for DPO (quality of chosen/rejected reasoning matters)
        mistakes = _COMMON_MISTAKES.get(phase, _COMMON_MISTAKES["RECON"])
        rejected_cmd, _ = random.choice(mistakes)

        if enhancer and cloud_dpo < min(500, cloud_enhance_count // 4):
            cloud_pair = enhancer.enhance_dpo_pair(
                target, phase, command, rejected_cmd, board, chain["name"],
            )
            if cloud_pair:
                chosen_resp, rejected_resp = cloud_pair
                prompt = (
                    f"Pick the single best next command.\n\n"
                    f"Target: {target}\nPhase: {phase}\nRole: offensive\n"
                    f"Stagnation: {random.randint(0, 5)} steps\n"
                    f"Discovery board:\n{json.dumps(board, indent=2)}\n"
                    f"Available templates: {', '.join(available + [rejected_cmd])}"
                )
                dpo_examples.append({
                    "prompt": prompt,
                    "chosen": chosen_resp,
                    "rejected": rejected_resp,
                })
                cloud_dpo += 1
                continue

        pair = gen_dpo_pair(target, phase, command, reasoning, board, available)
        dpo_examples.append(pair)

    dpo_path = output_dir / "htb_dpo_v5.jsonl"
    with open(dpo_path, "w") as f:
        for ex in dpo_examples:
            f.write(json.dumps(ex) + "\n")
    stats["dpo"] = len(dpo_examples)
    if cloud_dpo:
        print(f"  DPO: {len(dpo_examples)} pairs ({cloud_dpo} cloud-enhanced) → {dpo_path}")
    else:
        print(f"  DPO: {len(dpo_examples)} pairs → {dpo_path}")

    # ── Summary ──────────────────────────────────────────────────────────
    total = sum(stats.values())
    print(f"\n{'='*60}")
    print(f"TOTAL: {total} training examples generated")
    print(f"  Attack chains: {len(ALL_CHAINS)} ({len(real_box_chains)} real HTB/THM boxes)")
    print(f"  Schema distribution: {dict(schema_dist)}")
    print(f"  Categories: {Counter(c['category'] for c in ALL_CHAINS)}")
    if enhancer:
        cs = enhancer.get_stats()
        print(f"  Cloud LLM: {cs['calls']} calls, {cs['errors']} errors, "
              f"{cloud_enhanced} SFT + {cloud_strategic} strategic + {cloud_dpo} DPO enhanced")
    print(f"{'='*60}")

    return stats


def gen_output_parse_examples_v2(count: int = 2000, seed: int = 42) -> List[Dict]:
    """Generate output parsing training examples (expanded tool coverage)."""
    random.seed(seed)
    examples = []

    output_map = {
        "nmap": (_NMAP_OUTPUTS, _parse_nmap_expected, "nmap -sV {target}"),
        "gobuster": (_GOBUSTER_OUTPUTS, _parse_gobuster_expected,
                     "gobuster dir -u http://{target} -w /usr/share/wordlists/dirb/common.txt"),
        "linpeas": (_LINPEAS_OUTPUTS, _parse_linpeas_expected, "bash linpeas.sh"),
        "curl": (_CURL_OUTPUTS, _parse_curl_expected, "curl -s http://{target}/"),
        "smbclient": (_SMBCLIENT_OUTPUTS, _parse_smbclient_expected,
                      "smbclient -L //{target} -N"),
        "searchsploit": (_SEARCHSPLOIT_OUTPUTS, _parse_searchsploit_expected,
                         "searchsploit {service}"),
        "winpeas": (_WINPEAS_OUTPUTS, _parse_winpeas_expected, "winPEASx64.exe"),
        "hydra": (_HYDRA_OUTPUTS, _parse_hydra_expected,
                  "hydra -l admin -P /usr/share/wordlists/rockyou.txt ssh://{target}"),
        "hashcat": (_HASHCAT_OUTPUTS, _parse_hashcat_expected,
                    "hashcat -m 13100 hash.txt /usr/share/wordlists/rockyou.txt"),
    }

    for _ in range(count):
        tool = random.choice(list(output_map.keys()))
        outputs, parser, cmd_template = output_map[tool]
        output = random.choice(outputs)
        expected = parser(output)
        target = _random_target()
        cmd = cmd_template.format(target=target, service=random.choice(["samba", "hfs", "nostromo"]))

        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPTS["output_parse"]},
                {
                    "role": "user",
                    "content": (
                        f"Parse this penetration testing tool output.\n\n"
                        f"Tool: {tool}\nCommand: {cmd}\nSTDOUT:\n```\n{output}\n```\n\nJSON:"
                    ),
                },
                {"role": "assistant", "content": json.dumps(expected)},
            ],
            "schema_type": "output_parse",
        })

    return examples


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HTB Training Data Generator V5")
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--sft-count", type=int, default=15000)
    parser.add_argument("--strategic-count", type=int, default=3000)
    parser.add_argument("--dpo-count", type=int, default=2500)
    parser.add_argument("--output-parse-count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cloud-enhance", action="store_true",
                        help="Use cloud LLM (Groq 70B) to generate expert reasoning")
    parser.add_argument("--cloud-enhance-count", type=int, default=3000,
                        help="Number of examples to enhance with cloud LLM")
    parser.add_argument("--stats", action="store_true", help="Show chain stats only")
    args = parser.parse_args()

    if args.stats:
        print(f"Attack chains: {len(ALL_CHAINS)}")
        real_boxes = [c for c in ALL_CHAINS if c.get("box_info")]
        print(f"  Real HTB/THM boxes: {len(real_boxes)}")
        cats = Counter(c["category"] for c in ALL_CHAINS)
        for cat, n in cats.most_common():
            print(f"  {cat}: {n}")
            for c in ALL_CHAINS:
                if c["category"] == cat:
                    print(f"    {c['name']} ({c['difficulty']}): {len(c['steps'])} steps")
        sys.exit(0)

    generate_all(
        args.output_dir,
        sft_count=args.sft_count,
        strategic_count=args.strategic_count,
        dpo_count=args.dpo_count,
        output_parse_count=args.output_parse_count,
        seed=args.seed,
        cloud_enhance=args.cloud_enhance,
        cloud_enhance_count=args.cloud_enhance_count,
    )
