#!/usr/bin/env python3
"""
Tool Dependency Audit for Ariaska_RL Knowledge Corpus v2.

Scans all 18 JSONL files in data/knowledge_candidates_v2/,
extracts every tool reference, checks installation status,
and produces a structured report.
"""

import json
import glob
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "knowledge_candidates_v2"

# ─── Tool classification knowledge base ─────────────────────────────────────
BUILTINS = {
    "echo", "cat", "cd", "ls", "cp", "mv", "rm", "mkdir", "rmdir",
    "chmod", "chown", "chgrp", "ln", "pwd", "export", "source", "alias",
    "unalias", "set", "unset", "env", "printenv", "read", "eval",
    "exec", "exit", "test", "true", "false", "type", "which", "where",
    "printf", "kill", "wait", "trap", "shift", "return", "break",
    "continue", "for", "while", "until", "if", "then", "else", "fi",
    "case", "esac", "do", "done", "in", "function", "select", "time",
    "ulimit", "umask", "getopts", "hash", "history", "jobs", "fg", "bg",
    "disown", "suspend", "logout", "login", "su", "sudo", "whoami",
    "id", "groups", "users", "who", "w", "last", "lastlog", "finger",
    "basename", "dirname", "head", "tail", "wc", "sort", "uniq", "cut",
    "paste", "tr", "tee", "xargs", "find", "grep", "egrep", "fgrep",
    "sed", "awk", "gawk", "diff", "patch", "tar", "gzip", "gunzip",
    "bzip2", "bunzip2", "zip", "unzip", "file", "stat", "touch",
    "date", "cal", "bc", "expr", "seq", "yes", "sleep", "watch",
    "tput", "stty", "column", "fold", "fmt", "nl", "od", "xxd",
    "hexdump", "strings", "rev", "shuf", "comm", "join", "split",
    "csplit", "expand", "unexpand", "pr", "less", "more", "man",
    "info", "help", "whatis", "apropos", "locate", "updatedb",
    "mount", "umount", "df", "du", "free", "top", "ps", "kill",
    "nice", "renice", "nohup", "screen", "tmux", "crontab", "at",
    "batch", "systemctl", "service", "journalctl", "dmesg",
    "lsof", "fuser", "strace", "ltrace", "ldd", "nm", "readelf",
    "objdump", "ar", "ranlib", "strip", "install", "make", "cmake",
    "gcc", "g++", "cc", "ld", "as", "cpp", "m4",
    "ping", "traceroute", "tracepath", "ifconfig", "ip", "route",
    "netstat", "ss", "hostname", "host", "dig", "nslookup",
    "wget", "curl", "scp", "sftp", "rsync", "ftp", "telnet",
    "ssh", "ssh-keygen", "ssh-copy-id", "ssh-agent", "ssh-add",
    "nc", "ncat", "socat", "openssl", "base64", "md5sum", "sha256sum",
    "sha1sum", "sha512sum", "gpg", "gpg2",
    "useradd", "userdel", "usermod", "groupadd", "groupdel", "groupmod",
    "passwd", "chpasswd", "newgrp", "chage", "visudo",
    "iptables", "ip6tables", "nft", "ufw", "firewall-cmd",
    "adduser", "deluser", "dpkg", "rpm",
    "dd", "mkfs", "fdisk", "parted", "lsblk", "blkid",
    "cmp", "md5", "sha1", "sha256", "mktemp",
    "bash", "sh", "zsh", "dash", "csh", "tcsh", "ksh", "fish",
    "python", "python3", "python2", "perl", "ruby", "php", "node",
    "java", "javac", "jar",
}

MSF_TOOLS = {
    "msfconsole", "msfvenom", "msf", "msfdb", "msfpayload", "msfencode",
    "msfcli", "msfrpc", "msfrpcd", "meterpreter", "metasploit",
    "auxiliary", "exploit", "post", "use", "set", "run",
}

# Tools known to be installable via apt (Kali/Debian packages)
APT_TOOLS = {
    "nmap": "nmap",
    "nikto": "nikto",
    "gobuster": "gobuster",
    "dirb": "dirb",
    "dirbuster": "dirbuster",
    "wfuzz": "wfuzz",
    "sqlmap": "sqlmap",
    "hydra": "hydra",
    "medusa": "medusa",
    "john": "john",
    "hashcat": "hashcat",
    "aircrack-ng": "aircrack-ng",
    "wireshark": "wireshark",
    "tshark": "tshark",
    "tcpdump": "tcpdump",
    "ettercap": "ettercap-common",
    "arpspoof": "dsniff",
    "dsniff": "dsniff",
    "responder": "responder",
    "smbclient": "smbclient",
    "smbmap": "smbmap",
    "enum4linux": "enum4linux",
    "nbtscan": "nbtscan",
    "onesixtyone": "onesixtyone",
    "snmpwalk": "snmp",
    "snmpcheck": "snmpcheck",
    "snmp-check": "snmpcheck",
    "whatweb": "whatweb",
    "wpscan": "wpscan",
    "masscan": "masscan",
    "netcat": "netcat-traditional",
    "netcat-traditional": "netcat-traditional",
    "hping3": "hping3",
    "arping": "arping",
    "fping": "fping",
    "ncrack": "ncrack",
    "recon-ng": "recon-ng",
    "maltego": "maltego",
    "burpsuite": "burpsuite",
    "zaproxy": "zaproxy",
    "fierce": "fierce",
    "dnsenum": "dnsenum",
    "dnsrecon": "dnsrecon",
    "sublist3r": "sublist3r",
    "amass": "amass",
    "theharvester": "theharvester",
    "dmitry": "dmitry",
    "whois": "whois",
    "traceroute": "traceroute",
    "proxychains": "proxychains4",
    "proxychains4": "proxychains4",
    "tor": "tor",
    "torsocks": "torsocks",
    "macchanger": "macchanger",
    "steghide": "steghide",
    "stegseek": "stegseek",
    "binwalk": "binwalk",
    "exiftool": "libimage-exiftool-perl",
    "foremost": "foremost",
    "volatility": "volatility",
    "autopsy": "autopsy",
    "sleuthkit": "sleuthkit",
    "radare2": "radare2",
    "r2": "radare2",
    "gdb": "gdb",
    "ltrace": "ltrace",
    "strace": "strace",
    "valgrind": "valgrind",
    "ghidra": "ghidra",
    "cewl": "cewl",
    "crunch": "crunch",
    "wordlists": "wordlists",
    "seclists": "seclists",
    "rlwrap": "rlwrap",
    "xsltproc": "xsltproc",
    "rdesktop": "rdesktop",
    "xfreerdp": "freerdp2-x11",
    "freerdp": "freerdp2-x11",
    "cadaver": "cadaver",
    "davtest": "davtest",
    "smtp-user-enum": "smtp-user-enum",
    "swaks": "swaks",
    "sendemail": "sendemail",
    "commix": "commix",
    "beef-xss": "beef-xss",
    "set": "set",
    "yersinia": "yersinia",
    "bettercap": "bettercap",
    "mitmproxy": "mitmproxy",
    "proxmark3": "proxmark3",
    "mimikatz": "mimikatz",  # kali package
    "pth-toolkit": "passing-the-hash",
    "crackmapexec": "crackmapexec",
    "evil-winrm": "evil-winrm",
    "evil-winrm": "evil-winrm",
    "impacket-scripts": "impacket-scripts",
    "patator": "patator",
    "wordlists": "wordlists",
    "arp-scan": "arp-scan",
    "lsof": "lsof",
    "jq": "jq",
    "xmlstarlet": "xmlstarlet",
    "xmllint": "libxml2-utils",
    "tree": "tree",
    "git": "git",
    "vim": "vim",
    "nano": "nano",
    "nfs-common": "nfs-common",
    "showmount": "nfs-common",
    "rpcinfo": "rpcbind",
    "rpcclient": "samba-common-bin",
    "net": "samba-common-bin",
    "ldapsearch": "ldap-utils",
    "ldapdomaindump": "ldapdomaindump",
    "bloodhound": "bloodhound",
    "neo4j": "neo4j",
    "chisel": "chisel",
    "socat": "socat",
    "pwncat": "pwncat",
    "ffuf": "ffuf",
    "feroxbuster": "feroxbuster",
    "gpp-decrypt": "gpp-decrypt",
    "hashid": "hashid",
    "hash-identifier": "hash-identifier",
    "testssl.sh": "testssl.sh",
    "testssl": "testssl.sh",
    "sslyze": "sslyze",
    "sslscan": "sslscan",
    "amap": "amap",
    "p0f": "p0f",
    "xsser": "xsser",
    "skipfish": "skipfish",
    "wafw00f": "wafw00f",
    "wkhtmltopdf": "wkhtmltopdf",
    "pdftotext": "poppler-utils",
    "convert": "imagemagick",
    "identify": "imagemagick",
    "7z": "p7zip-full",
    "unrar": "unrar",
    "p7zip": "p7zip-full",
    "remmina": "remmina",
    "vncviewer": "tigervnc-viewer",
    "enum4linux-ng": "enum4linux-ng",
    "manspider": "manspider",
    "adb": "adb",
    "mysql": "mysql-client",
    "psql": "postgresql-client",
    "redis-cli": "redis-tools",
}

# Tools installable via pip
PIP_TOOLS = {
    "impacket": "impacket",
    "impacket-smbserver": "impacket",
    "impacket-psexec": "impacket",
    "impacket-wmiexec": "impacket",
    "impacket-atexec": "impacket",
    "impacket-dcomexec": "impacket",
    "impacket-smbexec": "impacket",
    "impacket-secretsdump": "impacket",
    "impacket-getTGT": "impacket",
    "impacket-getST": "impacket",
    "impacket-GetNPUsers": "impacket",
    "impacket-GetUserSPNs": "impacket",
    "impacket-ntlmrelayx": "impacket",
    "impacket-lookupsid": "impacket",
    "impacket-reg": "impacket",
    "impacket-services": "impacket",
    "ntlmrelayx": "impacket",
    "ntlmrelayx.py": "impacket",
    "smbserver.py": "impacket",
    "psexec.py": "impacket",
    "wmiexec.py": "impacket",
    "atexec.py": "impacket",
    "dcomexec.py": "impacket",
    "smbexec.py": "impacket",
    "secretsdump.py": "impacket",
    "getTGT.py": "impacket",
    "getST.py": "impacket",
    "GetNPUsers.py": "impacket",
    "GetUserSPNs.py": "impacket",
    "ntlmrelayx.py": "impacket",
    "lookupsid.py": "impacket",
    "reg.py": "impacket",
    "services.py": "impacket",
    "ticketConverter.py": "impacket",
    "ticketer.py": "impacket",
    "addcomputer.py": "impacket",
    "rbcd.py": "impacket",
    "dacledit.py": "impacket",
    "findDelegation.py": "impacket",
    "samrdump.py": "impacket",
    "rpcdump.py": "impacket",
    "esentutl.py": "impacket",
    "mssqlclient.py": "impacket",
    "pywerview": "pywerview",
    "bloodhound-python": "bloodhound",
    "certipy": "certipy-ad",
    "certipy-ad": "certipy-ad",
    "pypykatz": "pypykatz",
    "mitm6": "mitm6",
    "pwntools": "pwntools",
    "pwn": "pwntools",
    "ropper": "ropper",
    "ropgadget": "ROPgadget",
    "ROPgadget": "ROPgadget",
    "one_gadget": "one_gadget",  # actually ruby gem
    "angr": "angr",
    "z3": "z3-solver",
    "pycryptodome": "pycryptodome",
    "scapy": "scapy",
    "requests": "requests",
    "beautifulsoup": "beautifulsoup4",
    "paramiko": "paramiko",
    "flask": "flask",
    "django": "django",
    "twisted": "twisted",
    "droopescan": "droopescan",
    "dirsearch": "dirsearch",
    "autorecon": "autorecon",
    "reconftw": "reconftw",
    "subfinder": "subfinder",
    "httpx": "httpx",
    "nuclei": "nuclei",
    "volatility3": "volatility3",
    "oletools": "oletools",
    "pylnk3": "pylnk3",
    "updog": "updog",
    "uploadserver": "uploadserver",
    "name-that-hash": "name-that-hash",
    "nth": "name-that-hash",
    "search-that-hash": "search-that-hash",
    "ssh-audit": "ssh-audit",
    "wesng": "wesng",
    "windows-exploit-suggester": "wesng",
    "ldeep": "ldeep",
    "adidnsdump": "adidnsdump",
    "coercer": "coercer",
    "petitpotam": "petitpotam",
    "krbrelayx": "krbrelayx",
    "pkinittools": "pkinittools",
    "donpapi": "donpapi",
    "netexec": "netexec",
    "nxc": "netexec",
    "cme": "crackmapexec",
}

# Tools installed via Go
GO_TOOLS = {
    "gobuster": "github.com/OJ/gobuster/v3@latest",
    "ffuf": "github.com/ffuf/ffuf/v2@latest",
    "kerbrute": "github.com/ropnop/kerbrute@latest",
    "chisel": "github.com/jpillora/chisel@latest",
    "ligolo-ng": "github.com/nicocha30/ligolo-ng@latest",
    "ligolo": "github.com/nicocha30/ligolo-ng@latest",
    "httpx": "github.com/projectdiscovery/httpx/cmd/httpx@latest",
    "nuclei": "github.com/projectdiscovery/nuclei/v3/cmd/nuclei@latest",
    "subfinder": "github.com/projectdiscovery/subfinder/v2/cmd/subfinder@latest",
    "amass": "github.com/owasp-amass/amass/v4/...@master",
    "aquatone": "github.com/michenriksen/aquatone@latest",
    "hakrawler": "github.com/hakluke/hakrawler@latest",
    "gau": "github.com/lc/gau/v2/cmd/gau@latest",
    "waybackurls": "github.com/tomnomnom/waybackurls@latest",
    "anew": "github.com/tomnomnom/anew@latest",
    "assetfinder": "github.com/tomnomnom/assetfinder@latest",
    "httprobe": "github.com/tomnomnom/httprobe@latest",
    "unfurl": "github.com/tomnomnom/unfurl@latest",
    "meg": "github.com/tomnomnom/meg@latest",
    "gf": "github.com/tomnomnom/gf@latest",
    "qsreplace": "github.com/tomnomnom/qsreplace@latest",
    "dalfox": "github.com/hahwul/dalfox/v2@latest",
    "kxss": "github.com/Emoe/kxss@latest",
    "interactsh-client": "github.com/projectdiscovery/interactsh/cmd/interactsh-client@latest",
    "notify": "github.com/projectdiscovery/notify/cmd/notify@latest",
    "naabu": "github.com/projectdiscovery/naabu/v2/cmd/naabu@latest",
    "mapcidr": "github.com/projectdiscovery/mapcidr/cmd/mapcidr@latest",
    "dnsx": "github.com/projectdiscovery/dnsx/cmd/dnsx@latest",
    "katana": "github.com/projectdiscovery/katana/cmd/katana@latest",
    "chaos-client": "github.com/projectdiscovery/chaos-client/cmd/chaos@latest",
    "trufflehog": "github.com/trufflesecurity/trufflehog@latest",
    "feroxbuster": "n/a",  # actually Rust/cargo
    "rustscan": "n/a",     # Rust/cargo
    "puredns": "github.com/d3mondev/puredns/v2@latest",
    "gospider": "github.com/jaeles-project/gospider@latest",
}

# Tools requiring git clone / manual build
GIT_TOOLS = {
    "linpeas": "https://github.com/peass-ng/PEASS-ng",
    "linpeas.sh": "https://github.com/peass-ng/PEASS-ng",
    "winpeas": "https://github.com/peass-ng/PEASS-ng",
    "winpeas.exe": "https://github.com/peass-ng/PEASS-ng",
    "winPEAS": "https://github.com/peass-ng/PEASS-ng",
    "linPEAS": "https://github.com/peass-ng/PEASS-ng",
    "pspy": "https://github.com/DominicBreuker/pspy",
    "pspy64": "https://github.com/DominicBreuker/pspy",
    "linenum": "https://github.com/rebootuser/LinEnum",
    "linenum.sh": "https://github.com/rebootuser/LinEnum",
    "LinEnum": "https://github.com/rebootuser/LinEnum",
    "unix-privesc-check": "https://github.com/pentestmonkey/unix-privesc-check",
    "linux-exploit-suggester": "https://github.com/mzet-/linux-exploit-suggester",
    "les.sh": "https://github.com/mzet-/linux-exploit-suggester",
    "linux-smart-enumeration": "https://github.com/diego-treitos/linux-smart-enumeration",
    "lse.sh": "https://github.com/diego-treitos/linux-smart-enumeration",
    "seatbelt": "https://github.com/GhostPack/Seatbelt",
    "sharphound": "https://github.com/BloodHoundAD/SharpHound",
    "SharpHound": "https://github.com/BloodHoundAD/SharpHound",
    "rubeus": "https://github.com/GhostPack/Rubeus",
    "Rubeus": "https://github.com/GhostPack/Rubeus",
    "certify": "https://github.com/GhostPack/Certify",
    "whisker": "https://github.com/eladshamir/Whisker",
    "powerview": "https://github.com/PowerShellMafia/PowerSploit",
    "PowerView": "https://github.com/PowerShellMafia/PowerSploit",
    "powersploit": "https://github.com/PowerShellMafia/PowerSploit",
    "PowerSploit": "https://github.com/PowerShellMafia/PowerSploit",
    "powerup": "https://github.com/PowerShellMafia/PowerSploit",
    "PowerUp": "https://github.com/PowerShellMafia/PowerSploit",
    "privesccheck": "https://github.com/itm4n/PrivescCheck",
    "PrivescCheck": "https://github.com/itm4n/PrivescCheck",
    "nishang": "https://github.com/samratashok/nishang",
    "powershell-empire": "https://github.com/BC-SECURITY/Empire",
    "empire": "https://github.com/BC-SECURITY/Empire",
    "covenant": "https://github.com/cobbr/Covenant",
    "sliver": "https://github.com/BishopFox/sliver",
    "havoc": "https://github.com/HavocFramework/Havoc",
    "cobalt-strike": "commercial",
    "searchsploit": "https://gitlab.com/exploit-database/exploitdb",
    "joomscan": "https://github.com/OWASP/joomscan",
    "droopescan": "https://github.com/SamJoan/droopescan",
    "gitdumper": "https://github.com/arthaud/git-dumper",
    "git-dumper": "https://github.com/arthaud/git-dumper",
    "gittools": "https://github.com/internetwache/GitTools",
    "ysoserial": "https://github.com/frohoff/ysoserial",
    "ysoserial.net": "https://github.com/pwntester/ysoserial.net",
    "juicypotato": "https://github.com/ohpe/juicy-potato",
    "printspoofer": "https://github.com/itm4n/PrintSpoofer",
    "godpotato": "https://github.com/BeichenDream/GodPotato",
    "sweetpotato": "https://github.com/CCob/SweetPotato",
    "rottenpotato": "https://github.com/foxglovesec/RottenPotato",
    "potato": "various potato exploits",
    "chisel": "https://github.com/jpillora/chisel",
    "ligolo-ng": "https://github.com/nicocha30/ligolo-ng",
    "villain": "https://github.com/t3l3machus/Villain",
    "phpmyadmin": "https://github.com/phpmyadmin/phpmyadmin",
    "webshell": "various",
    "phpbash": "https://github.com/Arrexel/phpbash",
    "p0wny-shell": "https://github.com/flozz/p0wny-shell",
    "revshells": "https://www.revshells.com/",
    "penelope": "https://github.com/brightio/penelope",
    "conptyshell": "https://github.com/antonioCoco/ConPtyShell",
    "villain": "https://github.com/t3l3machus/Villain",
    "responder": "https://github.com/lgandx/Responder",
    "pretender": "https://github.com/RedTeamPentesting/pretender",
    "krbrelayx": "https://github.com/dirkjanm/krbrelayx",
    "PKINITtools": "https://github.com/dirkjanm/PKINITtools",
    "petitpotam.py": "https://github.com/topotam/PetitPotam",
    "dfscoerce": "https://github.com/Wh04m1001/DFSCoerce",
    "shadowcoerce": "https://github.com/ShutdownRepo/ShadowCoerce",
    "printerbug.py": "https://github.com/dirkjanm/krbrelayx",
    "targetedKerberoast": "https://github.com/ShutdownRepo/targetedKerberoast",
    "windapsearch": "https://github.com/ropnop/windapsearch",
    "adidnsdump": "https://github.com/dirkjanm/adidnsdump",
}

# ─── Noise / non-tool tokens to skip ────────────────────────────────────────
NOISE_TOKENS = {
    "", "-", "--", "---", "#", "##", "###", "//", "/*", "*/",
    "root@kali:~#", "kali@kali:~$", "$", ">", ">>", "|", "&&",
    "the", "a", "an", "to", "of", "in", "on", "at", "by", "for",
    "with", "from", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "must",
    "this", "that", "these", "those", "it", "its", "not", "no",
    "or", "and", "but", "if", "then", "else", "when", "where",
    "how", "what", "which", "who", "whom", "whose", "why",
    "all", "any", "both", "each", "every", "few", "more", "most",
    "other", "some", "such", "only", "own", "same", "so", "than",
    "too", "very", "just", "also", "now", "here", "there",
    "http", "https", "www", "com", "org", "net", "io", "git",
    "c", "c++", "h", "cpp", "py", "js", "ts", "rb", "go", "rs",
    "txt", "md", "json", "xml", "html", "css", "yml", "yaml",
    "exe", "dll", "so", "o", "bin", "elf", "msi", "bat", "ps1",
    "png", "jpg", "jpeg", "gif", "svg", "pdf", "doc", "docx",
    "1", "2", "3", "4", "5", "0", "10", "100", "192.168.1.1",
    "127.0.0.1", "target", "target_ip", "attacker_ip", "ip",
    "port", "user", "password", "username", "admin", "root",
    "none", "null", "true", "false", "yes", "no",
    "use", "set", "run", "show", "options", "exploit", "sessions",
    "payload", "lhost", "lport", "rhost", "rport", "rhosts",
    "output", "input", "file", "path", "name", "value", "key",
    "string", "int", "float", "bool", "list", "dict", "tuple",
    "class", "def", "import", "from", "return", "print", "self",
    "try", "except", "finally", "raise", "pass", "lambda",
    "global", "nonlocal", "assert", "yield", "del", "with", "as",
    "\\n", "\\t", "\\r", "\\\\", "\\'", '\\"',
    "sudo", "su",  # these are prefixes not tools themselves for our purposes
}

# Patterns that indicate noise (not real tools)
NOISE_PATTERNS = [
    re.compile(r'^-'),             # flags like --name, -h
    re.compile(r'^```'),           # markdown code fences
    re.compile(r'^\d+$'),          # pure numbers
    re.compile(r'^\d+\.\d+'),     # version numbers
    re.compile(r'^[<>{}\[\]()]+$'), # brackets
    re.compile(r'^\\'),            # escape sequences
    re.compile(r'^edb-\d+$', re.I), # ExploitDB IDs like EDB-16929
    re.compile(r'^cve-', re.I),    # CVE IDs
    re.compile(r"^it'"),           # English contractions
    re.compile(r'^https?://'),     # URLs
    re.compile(r'^\*+$'),         # markdown emphasis
    re.compile(r'^note$', re.I),
    re.compile(r'^example$', re.I),
    re.compile(r'^verify$', re.I),
    re.compile(r'^you$', re.I),
    re.compile(r'^copy$', re.I),
    re.compile(r'^remove$', re.I),
    re.compile(r'^dump$', re.I),
    re.compile(r'^upload$', re.I),
    re.compile(r'^network$', re.I),
    re.compile(r'^credential$', re.I),
    re.compile(r'^print_info$', re.I),
    re.compile(r'^clear$', re.I),
    re.compile(r'^apt-get$', re.I),
    re.compile(r'^dir$', re.I),
    re.compile(r'^shred$', re.I),
    re.compile(r'^cron$', re.I),
    re.compile(r'^get$', re.I),
    re.compile(r'^using$', re.I),
    re.compile(r'^public$', re.I),
    re.compile(r'^foreach$', re.I),
    re.compile(r'^moreover$', re.I),
    re.compile(r'^however$', re.I),
    re.compile(r'^therefore$', re.I),
    re.compile(r'^although$', re.I),
    re.compile(r'^furthermore$', re.I),
    re.compile(r'^we$', re.I),
    re.compile(r'^they$', re.I),
    re.compile(r'^he$', re.I),
    re.compile(r'^she$', re.I),
    re.compile(r'^generate$', re.I),
    re.compile(r'^create$', re.I),
    re.compile(r'^select$', re.I),
    re.compile(r'^enable$', re.I),
    re.compile(r'^disable$', re.I),
    re.compile(r'^configure$', re.I),
    re.compile(r'^evil$', re.I),
    re.compile(r'^register$', re.I),
    re.compile(r'^sc$', re.I),
    re.compile(r'^reg$', re.I),
    re.compile(r'^iex$', re.I),
    re.compile(r'^pip$', re.I),
    re.compile(r'^apt$', re.I),
    re.compile(r'^print_2title$', re.I),
    re.compile(r'^\d+\)$'),           # "2)", "3)", etc
    re.compile(r'^\.\\'),             # Windows paths like .\rubeus.exe
    re.compile(r'^msf>$', re.I),      # MSF prompt artifact
    re.compile(r'^winrm$', re.I),     # protocol not tool
    re.compile(r'^ollama$', re.I),    # LLM runner, not pentest tool
]

# PowerShell cmdlets — not real Linux tools but worth tracking
POWERSHELL_CMDLETS = {
    "write-host", "import-module", "set-itemproperty", "new-item",
    "new-itemproperty", "get-childitem", "start-process", "get-process",
    "invoke-webrequest", "invoke-expression", "invoke-command",
    "invoke-mimikatz", "invoke-kerberoast", "invoke-bloodhound",
    "invoke-obfuscation", "get-content", "set-content", "out-file",
    "get-aduser", "get-adgroup", "get-adcomputer", "get-addomain",
    "get-addomaincontroller", "get-acl", "set-acl", "get-service",
    "add-type", "new-object", "convertto-securestring",
    "convertfrom-securestring", "get-wmiobject", "get-ciminstance",
    "test-connection", "test-path", "get-item", "set-item",
    "remove-item", "move-item", "copy-item", "rename-item",
    "get-eventlog", "clear-eventlog", "new-eventlog",
    "get-netfirewallrule", "set-netfirewallrule",
    "get-mppreference", "set-mppreference", "add-mppreference",
    "get-localuser", "new-localuser", "set-localuser",
    "get-localgroupmember", "add-localgroupmember",
    "reg.exe", "sc.exe", "net.exe", "icacls.exe", "takeown.exe",
    "certutil.exe", "powershell.exe", "cmd.exe", "wscript.exe",
    "cscript.exe", "mshta.exe", "regsvr32.exe", "rundll32.exe",
    "msiexec.exe", "schtasks.exe", "bitsadmin.exe", "wmic.exe",
}

# Windows-only binaries (not tools to install on Linux)
WINDOWS_BINS = {
    "cmd", "powershell", "powershell.exe", "cmd.exe",
    "wmic", "wmic.exe", "rundll32.exe", "rundll32",
    "schtasks", "schtasks.exe", "bitsadmin", "bitsadmin.exe",
    "certutil", "certutil.exe", "mshta", "mshta.exe",
    "regsvr32", "regsvr32.exe", "msiexec", "msiexec.exe",
    "cscript", "cscript.exe", "wscript", "wscript.exe",
    "sc.exe", "icacls.exe", "takeown.exe", "net.exe",
    "reg.exe", "attrib.exe", "tasklist.exe", "taskkill.exe",
    "netsh", "netsh.exe", "systeminfo", "systeminfo.exe",
    "whoami.exe", "ipconfig", "ipconfig.exe", "arp.exe",
    "start-process", "write-host", "import-module",
    "set-itemproperty", "new-item", "new-itemproperty",
    "get-childitem", "get-process", "invoke-webrequest",
    "invoke-expression", "invoke-command",
}

# Cloud CLIs
CLOUD_TOOLS = {
    "aws": ("CLOUD_CLI", "pip install awscli  /  curl https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip"),
    "az": ("CLOUD_CLI", "curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash"),
    "gcloud": ("CLOUD_CLI", "snap install google-cloud-cli --classic"),
    "gsutil": ("CLOUD_CLI", "included with gcloud SDK"),
    "kubectl": ("CLOUD_CLI", "snap install kubectl --classic"),
    "doctl": ("CLOUD_CLI", "snap install doctl"),
    "terraform": ("CLOUD_CLI", "apt install terraform"),
    "ansible": ("CLOUD_CLI", "pip install ansible"),
    "vagrant": ("CLOUD_CLI", "apt install vagrant"),
    "packer": ("CLOUD_CLI", "apt install packer"),
}

# Attack technique names (not tools)
ATTACK_TECHNIQUES = {
    "xxe", "ssrf", "ssti", "xss", "csrf", "sqli", "lfi", "rfi",
    "rce", "idor", "jwt", "nosqli", "shellshock", "heartbleed",
    "eternalblue", "mimikatz-golden", "kerberoasting",
    "pass-the-hash", "pass-the-ticket", "dcsync", "zerologon",
    "petitpotam", "printspooler", "printnightmare",
    "nfs", "tcp", "udp", "dns", "http", "smb", "rdp", "vnc",
    "mssql", "tomcat", "redis", "memcached", "elasticsearch",
    "ldap", "kerberos", "ftp", "smtp", "snmp", "imap", "pop3",
    "exfil", "revshell", "webshell", "payload", "backdoor",
    "kernel", "windows", "linux", "macos",
}

def is_noise(token: str) -> bool:
    """Check if a token is noise rather than a real tool."""
    if token in NOISE_TOKENS:
        return True
    if token in ATTACK_TECHNIQUES:
        return True
    if token in WINDOWS_BINS:
        return False  # Track these separately
    if token in POWERSHELL_CMDLETS:
        return False  # Track these separately
    for pat in NOISE_PATTERNS:
        if pat.match(token):
            return True
    # Skip if contains spaces, quotes, or looks like natural language
    if ' ' in token or '"' in token or "'" in token:
        return True
    return False


def clean_tool_name(raw: str) -> str:
    """Normalize a tool name."""
    if not raw:
        return ""
    # Strip shell prompt prefixes
    raw = re.sub(r'^(root@[\w:~#$]+\s*|kali@[\w:~#$]+\s*|\$\s*|#\s*|>\s*)', '', raw)
    # Strip leading ./ or /usr/bin/ etc
    raw = re.sub(r'^\./', '', raw)
    raw = re.sub(r'^/[\w/]+/', '', raw)
    # Strip trailing punctuation
    raw = raw.strip().rstrip(':;,.')
    return raw


def extract_first_word(cmd: str) -> str:
    """Extract the binary name (first word) from a command string."""
    if not cmd:
        return ""
    cmd = cmd.strip()
    # Remove sudo/su prefix
    cmd = re.sub(r'^(sudo\s+(-\w+\s+)*|su\s+(-\w+\s+)*)', '', cmd)
    # Remove env var assignments (FOO=bar cmd)
    cmd = re.sub(r'^(\w+=\S+\s+)+', '', cmd)
    # Remove shell prompt
    cmd = re.sub(r'^(root@[\w:~#$\-]+\s*|kali@[\w:~#$\-]+\s*|\$\s*|#\s*|>\s*)', '', cmd)
    # Get first word
    parts = cmd.split()
    if not parts:
        return ""
    first = parts[0]
    # Clean it
    first = clean_tool_name(first)
    # Strip path
    if '/' in first:
        first = first.split('/')[-1]
    return first.lower()


def categorize_tool(name: str) -> tuple:
    """Return (category, install_command) for a tool name."""
    name_lower = name.lower()

    # Attack techniques / protocol names — not tools
    if name_lower in ATTACK_TECHNIQUES:
        return ("TECHNIQUE", "not a tool — attack technique / protocol name")

    # Cloud CLIs
    if name_lower in CLOUD_TOOLS:
        return CLOUD_TOOLS[name_lower]

    # Windows binaries / PowerShell cmdlets
    if name_lower in WINDOWS_BINS or name_lower in POWERSHELL_CMDLETS:
        return ("WINDOWS", "Windows-only binary / PowerShell cmdlet")

    # Check builtins first
    if name_lower in BUILTINS or name in BUILTINS:
        return ("BUILTIN", "built-in / coreutils")

    # MSF tools
    if name_lower in MSF_TOOLS or name_lower.startswith("msf") or name_lower.startswith("meterpreter"):
        return ("MSF", "apt install metasploit-framework")

    # Check if it's a known apt tool
    if name_lower in APT_TOOLS:
        return ("APT", f"apt install {APT_TOOLS[name_lower]}")
    if name in APT_TOOLS:
        return ("APT", f"apt install {APT_TOOLS[name]}")

    # Check pip
    if name_lower in PIP_TOOLS:
        return ("PIP", f"pip install {PIP_TOOLS[name_lower]}")
    if name in PIP_TOOLS:
        return ("PIP", f"pip install {PIP_TOOLS[name]}")
    # .py suffix is often pip/impacket
    if name.endswith('.py'):
        base = name[:-3]
        if base in PIP_TOOLS:
            return ("PIP", f"pip install {PIP_TOOLS[base]}")

    # Check Go
    if name_lower in GO_TOOLS:
        return ("GO", f"go install {GO_TOOLS[name_lower]}")

    # Check git/manual
    if name_lower in GIT_TOOLS:
        return ("GIT", f"git clone {GIT_TOOLS[name_lower]}")
    if name in GIT_TOOLS:
        return ("GIT", f"git clone {GIT_TOOLS[name]}")

    return ("UNKNOWN", "—")


def main():
    tool_counter = Counter()
    total_entries = 0
    entries_with_tools = 0
    file_stats = {}

    jsonl_files = sorted(glob.glob(str(CORPUS_DIR / "*.jsonl")))
    print(f"Scanning {len(jsonl_files)} JSONL files in {CORPUS_DIR}\n")

    for filepath in jsonl_files:
        fname = os.path.basename(filepath)
        file_entry_count = 0
        file_tool_refs = 0

        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                total_entries += 1
                file_entry_count += 1
                tools_found = set()

                # === Source 1: references.tools ===
                ref_tools = entry.get("references", {}).get("tools", [])
                for t in ref_tools:
                    t_clean = clean_tool_name(str(t)).lower()
                    if t_clean and len(t_clean) > 1 and not is_noise(t_clean):
                        tools_found.add(t_clean)

                # === Source 2: execution.command_templates → extract tool prefix ===
                templates = entry.get("execution", {}).get("command_templates", [])
                for tmpl in templates:
                    if not tmpl:
                        continue
                    # Template names like "nmap_full_tcp" → tool = "nmap"
                    parts = tmpl.split("_")
                    tool_prefix = parts[0].lower()
                    if tool_prefix and len(tool_prefix) > 1 and not is_noise(tool_prefix):
                        tools_found.add(tool_prefix)

                # === Source 3: raw_preservation.original_commands → first word ===
                orig_cmds = entry.get("raw_preservation", {}).get("original_commands", [])
                for cmd in orig_cmds:
                    if not isinstance(cmd, str):
                        continue
                    first = extract_first_word(cmd)
                    if first and len(first) > 1 and not is_noise(first):
                        # Skip if it looks like a sentence fragment, not a command
                        if not first.startswith('(') and not first.startswith('{'):
                            tools_found.add(first)

                # === Source 4: source.raw_ref for commands origin (tool:command format) ===
                origin = entry.get("source", {}).get("origin", "")
                if origin == "commands":
                    raw_ref = entry.get("source", {}).get("raw_ref", "")
                    if ":" in raw_ref:
                        tool_part = raw_ref.split(":")[0].strip().lower()
                        tool_part = clean_tool_name(tool_part)
                        if tool_part and len(tool_part) > 1 and not is_noise(tool_part):
                            tools_found.add(tool_part)

                # === Source 5: title for binaries origin ===
                if origin == "binaries":
                    title = entry.get("title", "").strip().lower()
                    title = clean_tool_name(title)
                    if title and len(title) > 1 and not is_noise(title):
                        tools_found.add(title)

                # === Source 6: references.modules (MSF modules) ===
                modules = entry.get("references", {}).get("modules", [])
                for mod in modules:
                    if mod and isinstance(mod, str):
                        # e.g., "exploit/unix/ftp/vsftpd_234_backdoor" → msfconsole
                        if mod.startswith(("exploit/", "auxiliary/", "post/", "payload/")):
                            tools_found.add("msfconsole")

                if tools_found:
                    entries_with_tools += 1
                    file_tool_refs += 1
                    for t in tools_found:
                        tool_counter[t] += 1

        file_stats[fname] = {"entries": file_entry_count, "with_tools": file_tool_refs}

    # ─── Build report ────────────────────────────────────────────────────────
    print("=" * 100)
    print("  ARIASKA_RL KNOWLEDGE CORPUS v2 — TOOL DEPENDENCY AUDIT")
    print("=" * 100)
    print()

    # File breakdown
    print("─" * 80)
    print("  FILE BREAKDOWN")
    print("─" * 80)
    print(f"  {'File':<40} {'Entries':>10} {'With Tools':>12} {'Coverage':>10}")
    print(f"  {'─'*38}  {'─'*8}  {'─'*10}  {'─'*8}")
    for fname, stats in sorted(file_stats.items()):
        pct = (stats['with_tools'] / stats['entries'] * 100) if stats['entries'] > 0 else 0
        print(f"  {fname:<40} {stats['entries']:>10,} {stats['with_tools']:>12,} {pct:>9.1f}%")
    print(f"  {'─'*38}  {'─'*8}  {'─'*10}  {'─'*8}")
    print(f"  {'TOTAL':<40} {total_entries:>10,} {entries_with_tools:>12,} {(entries_with_tools/total_entries*100) if total_entries else 0:>9.1f}%")
    print()

    # Summary
    unique_tools = len(tool_counter)
    total_refs = sum(tool_counter.values())
    print("─" * 80)
    print("  SUMMARY")
    print("─" * 80)
    print(f"  Total entries scanned:         {total_entries:>10,}")
    print(f"  Entries with tool references:   {entries_with_tools:>10,}")
    print(f"  Total tool references:          {total_refs:>10,}")
    print(f"  Unique tools identified:        {unique_tools:>10,}")
    print()

    # Top 100 tools with installation check
    top_100 = tool_counter.most_common(100)

    print("─" * 120)
    print("  TOP 100 TOOLS — FREQUENCY, INSTALLATION STATUS & CATEGORY")
    print("─" * 120)
    print(f"  {'#':>3}  {'Tool':<28} {'Count':>8}  {'Inst?':>5}  {'Category':<10}  {'Install Command'}")
    print(f"  {'─'*2}  {'─'*26}  {'─'*6}   {'─'*4}  {'─'*8}   {'─'*50}")

    installed_count = 0
    missing_count = 0
    category_counts = Counter()
    category_installed = Counter()
    category_missing = Counter()

    results = []
    for rank, (tool, count) in enumerate(top_100, 1):
        is_installed = shutil.which(tool) is not None
        category, install_cmd = categorize_tool(tool)

        if is_installed:
            installed_count += 1
            inst_str = "  Y  "
            category_installed[category] += 1
        else:
            missing_count += 1
            inst_str = "  N  "
            category_missing[category] += 1

        category_counts[category] += 1
        results.append((rank, tool, count, is_installed, category, install_cmd))

        print(f"  {rank:>3}  {tool:<28} {count:>8,}  {inst_str}  {category:<10}  {install_cmd}")

    print()

    # Category breakdown
    print("─" * 80)
    print("  CATEGORY BREAKDOWN (Top 100)")
    print("─" * 80)
    print(f"  {'Category':<12} {'Total':>7} {'Installed':>10} {'Missing':>10}")
    print(f"  {'─'*10}  {'─'*6}  {'─'*8}   {'─'*7}")
    for cat in sorted(category_counts.keys()):
        print(f"  {cat:<12} {category_counts[cat]:>7} {category_installed.get(cat,0):>10} {category_missing.get(cat,0):>10}")
    print(f"  {'─'*10}  {'─'*6}  {'─'*8}   {'─'*7}")
    print(f"  {'TOTAL':<12} {sum(category_counts.values()):>7} {installed_count:>10} {missing_count:>10}")
    print()

    # Installation summary
    print("─" * 80)
    print("  INSTALLATION SUMMARY (Top 100)")
    print("─" * 80)
    print(f"  ✅ Installed:  {installed_count:>4} / 100")
    print(f"  ❌ Missing:    {missing_count:>4} / 100")
    print()

    # Missing tools by category (actionable)
    print("─" * 80)
    print("  MISSING TOOLS — ACTIONABLE INSTALL COMMANDS")
    print("─" * 80)
    for cat in ["APT", "PIP", "GO", "GIT", "MSF"]:
        missing_in_cat = [(r[1], r[2], r[5]) for r in results if r[4] == cat and not r[3]]
        if missing_in_cat:
            print(f"\n  [{cat}] — {len(missing_in_cat)} tools:")
            for tool, count, cmd in sorted(missing_in_cat, key=lambda x: -x[1]):
                print(f"    {tool:<28} (refs: {count:>6,})  →  {cmd}")

    unknown_missing = [(r[1], r[2]) for r in results if r[4] == "UNKNOWN" and not r[3]]
    if unknown_missing:
        print(f"\n  [UNKNOWN] — {len(unknown_missing)} tools (manual research needed):")
        for tool, count in sorted(unknown_missing, key=lambda x: -x[1]):
            print(f"    {tool:<28} (refs: {count:>6,})")

    print()
    print("=" * 100)
    print("  AUDIT COMPLETE")
    print("=" * 100)

    # Also output all unique tools (beyond top 100) count
    print(f"\n  Remaining {unique_tools - 100} tools (outside top 100) account for {sum(c for _, c in tool_counter.most_common()[100:]):,} references.")
    print(f"  Full unique tool list has {unique_tools} entries.\n")


if __name__ == "__main__":
    main()
