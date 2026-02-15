#!/usr/bin/env python3
"""
Refactor + Enrich v1 Knowledge Corpus → KnowledgeCandidate v2 JSONL.

Phase 10.0 — Knowledge Governance.

Outputs:
  data/knowledge_candidates_v2/   (partitioned JSONL by source type)
  data/knowledge_quarantine_v2.jsonl  (entries that cannot be mapped safely)

Enrichment:
  - Normalize typos (preserve originals in raw_preservation)
  - Infer taxonomy (vuln_family, service_archetype, exploit_archetype, killchain_step)
  - Generate evidence_gate from tool/service/port/command semantics
  - Map to COMMAND_REGISTRY templates where possible
  - Attach CVE references, MITRE ATT&CK mappings
  - Score quality (template_mappable, evidence_coverage, prereq_coverage)
"""

import json
import os
import re
import sys
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.knowledge.knowledge_candidate_v2 import (
    KnowledgeCandidate, SourceInfo, Taxonomy, MitreMapping, PlatformInfo,
    EvidenceGate, Execution, ExecutionStep, Verification, References,
    Governance, QualityMetrics, RuntimeStats, RawPreservation,
    save_candidates_jsonl,
)

KB_DIR = PROJECT_ROOT / "data" / "knowledge_base"
V2_DIR = PROJECT_ROOT / "data" / "knowledge_candidates_v2"
QUARANTINE_PATH = PROJECT_ROOT / "data" / "knowledge_quarantine_v2.jsonl"


# ─── Registry Template Lookup ────────────────────────────────────────────────

def load_registry_templates() -> Set[str]:
    """Load all known command registry template names."""
    registry_file = PROJECT_ROOT / "core" / "commands" / "command_registry.py"
    names = set()
    try:
        content = registry_file.read_text()
        for m in re.finditer(r'name="([^"]+)"', content):
            names.add(m.group(1))
    except Exception:
        pass
    return names

REGISTRY_TEMPLATES = load_registry_templates()


# ─── Typo Normalization ──────────────────────────────────────────────────────

TYPO_MAP = {
    "enumaration": "enumeration",
    "enumarate": "enumerate",
    "credintials": "credentials",
    "certifcate": "certificate",
    "enviroment": "environment",
    "equalivent": "equivalent",
    "preforming": "performing",
    "potenial": "potential",
    "neaar": "near",
    "Mimilkatz": "Mimikatz",
    "privilage": "privilege",
    "escallation": "escalation",
    "exfiltation": "exfiltration",
    "reconnaisance": "reconnaissance",
    "vunerability": "vulnerability",
    "vulnerabilty": "vulnerability",
    "authentification": "authentication",
    "persistance": "persistence",
    "laterl": "lateral",
    "priv esc": "privilege escalation",
}


def normalize_typos(text: str) -> str:
    """Fix known typos while preserving structure."""
    if not text:
        return text
    result = text
    for typo, fix in TYPO_MAP.items():
        result = re.sub(re.escape(typo), fix, result, flags=re.IGNORECASE)
    return result


# ─── Tool → Registry Template Mapping ────────────────────────────────────────

TOOL_TO_TEMPLATES = {
    "nmap": ["nmap_quick_scan", "nmap_service_version", "nmap_vuln_scan", "nmap_stealth_scan", "nmap_full_tcp", "nmap_os_detection", "nmap_udp_scan", "nmap_top_ports"],
    "nikto": ["nikto_scan"],
    "gobuster": ["gobuster_dir", "gobuster_vhost"],
    "dirb": ["gobuster_dir", "dirsearch"],
    "dirsearch": ["dirsearch"],
    "feroxbuster": ["feroxbuster"],
    "ffuf": ["ffuf_fuzz"],
    "wfuzz": ["wfuzz_params"],
    "sqlmap": ["sqlmap_test", "sqlmap_get", "sqlmap_post", "sqlmap_shell"],
    "hydra": ["hydra_ssh", "hydra_ftp", "hydra_smb", "hydra_http_form"],
    "enum4linux": ["enum4linux_scan", "enum4linux_full", "enum4linux_ng"],
    "smbclient": ["smbclient_list", "smbclient_connect", "smbclient_auth"],
    "smbmap": ["smbmap_shares"],
    "crackmapexec": ["crackmapexec_smb_bruteforce", "crackmapexec_password_spray", "crackmapexec_pth", "crackmapexec_winrm"],
    "cme": ["cme_smb_shares", "cme_smb_bruteforce", "cme_ldap_users", "cme_exec_command"],
    "rpcclient": ["rpcclient_null", "rpcclient_enumdomusers"],
    "evil-winrm": ["evil_winrm", "evil_winrm_hash"],
    "winrm": ["winrm_exec", "evil_winrm"],
    "impacket": ["impacket_psexec", "impacket_smbexec", "impacket_wmiexec", "impacket_secretsdump", "impacket_GetNPUsers", "impacket_GetUserSPNs", "impacket_dcomexec", "impacket_atexec", "impacket_pth_psexec", "impacket_reg"],
    "psexec": ["impacket_psexec", "impacket_pth_psexec"],
    "smbexec": ["impacket_smbexec"],
    "wmiexec": ["impacket_wmiexec"],
    "secretsdump": ["impacket_secretsdump"],
    "mimikatz": ["mimikatz_logonpasswords", "mimikatz_sam", "mimikatz_dcsync"],
    "bloodhound": ["bloodhound_python", "sharphound"],
    "sharphound": ["sharphound"],
    "rubeus": ["rubeus_kerberoast", "rubeus_asreproast"],
    "certipy": ["certipy_find", "certipy_req"],
    "responder": ["responder"],
    "ntlmrelayx": ["ntlmrelayx"],
    "linpeas": ["linpeas", "linpeas_local"],
    "winpeas": ["winpeas"],
    "pspy": ["pspy", "pspy_monitor"],
    "searchsploit": ["searchsploit"],
    "metasploit": ["msfconsole_exploit", "msfconsole_auto"],
    "msfconsole": ["msfconsole_exploit", "msfconsole_auto"],
    "msfvenom": ["msfvenom_payload"],
    "whatweb": ["whatweb"],
    "wpscan": ["wpscan"],
    "joomscan": ["joomscan"],
    "droopescan": ["droopescan"],
    "ssh": ["ssh_login", "ssh_key_login", "ssh_lateral", "ssh_audit"],
    "ftp": ["ftp_anonymous", "hydra_ftp"],
    "telnet": ["telnet_1524"],
    "mysql": ["mysql_login"],
    "mssql": ["mssql_login"],
    "psql": ["psql_default_creds", "psql_rce"],
    "postgres": ["psql_default_creds", "psql_rce"],
    "redis": ["redis_cli"],
    "snmpwalk": ["snmpwalk"],
    "onesixtyone": ["onesixtyone"],
    "dig": ["dig_any", "dig_axfr"],
    "dnsrecon": ["dnsrecon"],
    "ldapsearch": ["ldapsearch_base", "ldapsearch_users"],
    "windapsearch": ["windapsearch"],
    "showmount": ["showmount", "showmount_enum"],
    "nfs": ["nfs_mount", "showmount_enum"],
    "curl": ["curl_headers", "curl_exfil"],
    "nuclei": ["nuclei_scan"],
    "chisel": ["chisel_server", "chisel_client"],
    "ligolo": ["ligolo_agent"],
    "masscan": ["masscan_fast"],
    "whois": ["whois_lookup"],
    "nc": ["nc_exfil", "revshell_bash"],
    "netcat": ["nc_exfil", "revshell_bash"],
    "docker": ["docker_privesc", "docker_sock_escape"],
    "lxd": ["lxd_privesc", "lxd_escape"],
    "sudo": ["sudo_check", "sudo_list"],
    "find": ["find_suid", "find_sgid", "find_capabilities", "find_writable_etc"],
    "cron": ["cron_check", "cron_backdoor"],
    "tplmap": ["tplmap_scan"],
    "log4shell": ["log4shell_detect"],
    "heartbleed": ["heartbleed_exploit"],
    "shellshock": ["shellshock_cgi"],
    "drupalgeddon": ["drupalgeddon2"],
}


# ─── Phase / Killchain Inference ─────────────────────────────────────────────

PHASE_KEYWORDS = {
    "RECON": ["recon", "discovery", "scan", "enumerate", "enum", "footprint", "whois", "dns", "osint", "ping", "traceroute"],
    "ENUMERATION": ["enumerate", "enumeration", "version", "service", "banner", "fingerprint", "directory", "brute", "fuzz", "spider", "crawl", "user enum"],
    "EXPLOITATION": ["exploit", "attack", "injection", "rce", "shell", "reverse", "payload", "buffer overflow", "sqli", "xss", "lfi", "rfi", "ssrf", "ssti", "deserial", "upload", "backdoor"],
    "PRIVILEGE_ESCALATION": ["privesc", "privilege", "escalat", "suid", "sgid", "sudo", "kernel", "capability", "writable", "cron", "path hijack", "dll", "token", "impersonate", "potato"],
    "LATERAL_MOVEMENT": ["lateral", "pivot", "tunnel", "proxy", "relay", "pass the hash", "pth", "pass the ticket", "wmi", "psexec", "dcom", "winrm", "ssh tunnel"],
    "POST_EXPLOITATION": ["post-exploit", "persist", "dump", "credential", "hashdump", "mimikatz", "secretsdump", "keylog", "screenshot", "data collect", "loot"],
    "EXFILTRATION": ["exfil", "transfer", "extract", "download", "upload", "dns exfil", "icmp exfil", "encrypt", "compress", "archive"],
}

KILLCHAIN_MAP = {
    "RECON": "recon",
    "ENUMERATION": "recon",
    "EXPLOITATION": "foothold",
    "PRIVILEGE_ESCALATION": "privesc",
    "LATERAL_MOVEMENT": "lateral",
    "POST_EXPLOITATION": "persist",
    "EXFILTRATION": "exfil",
}


def infer_phases(text: str, category: str = "") -> List[str]:
    """Infer attack phases from text content."""
    text_lower = (text + " " + category).lower()
    phases = []
    for phase, keywords in PHASE_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            phases.append(phase)
    return phases if phases else ["ENUMERATION"]


def infer_killchain(phases: List[str]) -> str:
    """Map phases to killchain step."""
    for p in phases:
        if p in KILLCHAIN_MAP:
            return KILLCHAIN_MAP[p]
    return "recon"


# ─── Service / Vuln Family Inference ─────────────────────────────────────────

SERVICE_PATTERNS = {
    "smb": r"\b(smb|samba|cifs|445|139)\b",
    "ssh": r"\b(ssh|22/tcp|openssh)\b",
    "ftp": r"\b(ftp|21/tcp|vsftpd|proftpd)\b",
    "http": r"\b(http|web|apache|nginx|iis|tomcat|80/tcp|443/tcp|8080|8443)\b",
    "database": r"\b(mysql|postgres|mssql|oracle|mongo|redis|3306|5432|1433|27017|6379)\b",
    "ldap": r"\b(ldap|389|636|active.directory)\b",
    "dns": r"\b(dns|53/tcp|53/udp|bind|named)\b",
    "mail": r"\b(smtp|pop3|imap|25/tcp|110|143|postfix|dovecot)\b",
    "rdp": r"\b(rdp|3389|remote.desktop)\b",
    "vnc": r"\b(vnc|5900|5901)\b",
    "kerberos": r"\b(kerberos|88/tcp|krb5|ticket|tgt|tgs)\b",
    "snmp": r"\b(snmp|161/udp|community.string)\b",
    "nfs": r"\b(nfs|2049|showmount|rpcbind)\b",
    "rpc": r"\b(rpc|rpcbind|111|msrpc|135)\b",
    "winrm": r"\b(winrm|5985|5986|wsman)\b",
    "irc": r"\b(irc|ircd|6667|6697)\b",
    "java-rmi": r"\b(rmi|1099|java.rmi)\b",
}

VULN_FAMILY_PATTERNS = {
    "authentication-bypass": r"\b(bypass|default.cred|no.auth|anonymous|null.session|weak.pass)\b",
    "rce": r"\b(rce|remote.code|command.execution|code.exec|os.command)\b",
    "command-injection": r"\b(command.inject|cmd.inject|os.inject)\b",
    "sql-injection": r"\b(sqli|sql.inject|blind.sql|union.select)\b",
    "lfi": r"\b(lfi|local.file.incl|path.travers|directory.travers)\b",
    "rfi": r"\b(rfi|remote.file.incl)\b",
    "xss": r"\b(xss|cross.site.script|reflected|stored.xss)\b",
    "ssrf": r"\b(ssrf|server.side.request)\b",
    "ssti": r"\b(ssti|server.side.template|template.inject)\b",
    "deserialization": r"\b(deserializ|unserializ|pickle|ysoserial|java.serial)\b",
    "buffer-overflow": r"\b(buffer.overflow|bof|stack.overflow|heap.overflow)\b",
    "info-disclosure": r"\b(info.disclos|information.leak|data.expos|sensitive.data)\b",
    "privilege-escalation": r"\b(privesc|privilege.escalat|suid|sudo|setuid|capability)\b",
    "credential-theft": r"\b(credential|password.dump|hash.dump|mimikatz|secretsdump|sam.dump)\b",
    "backdoor": r"\b(backdoor|back.door|trojan|implant|rootkit)\b",
    "misconfiguration": r"\b(misconfig|default.config|open.relay|world.readable|weak.perm)\b",
    "file-upload": r"\b(file.upload|upload.shell|webshell|unrestricted.upload)\b",
    "xxe": r"\b(xxe|xml.external|xml.inject)\b",
}

EXPLOIT_ARCHETYPE_PATTERNS = {
    "default-credentials": r"\b(default.cred|default.pass|default.user|msfadmin|admin.admin|root.root|tomcat.tomcat)\b",
    "command-injection": r"\b(command.inject|cmd.inject|backtick|semicolon|pipe.inject)\b",
    "known-exploit": r"\b(cve-|exploit/|metasploit|searchsploit|edb-id)\b",
    "brute-force": r"\b(brute.force|bruteforce|hydra|password.spray|dictionary.attack)\b",
    "credential-reuse": r"\b(credential.reuse|pass.the.hash|pth|pass.the.ticket|relay)\b",
    "service-exploit": r"\b(service.exploit|backdoor|vsftpd|unrealircd|samba.usermap)\b",
    "web-exploit": r"\b(web.exploit|sqli|xss|lfi|rfi|ssrf|ssti|upload)\b",
    "kernel-exploit": r"\b(kernel.exploit|dirty.cow|dirty.pipe|overlayfs|netfilter)\b",
    "misconfig-exploit": r"\b(misconfig|suid|writable|cron.job|path.hijack|docker.sock)\b",
}


def infer_service_archetype(text: str) -> str:
    text_lower = text.lower()
    for svc, pattern in SERVICE_PATTERNS.items():
        if re.search(pattern, text_lower):
            return svc
    return "general"


def infer_vuln_family(text: str) -> str:
    text_lower = text.lower()
    for family, pattern in VULN_FAMILY_PATTERNS.items():
        if re.search(pattern, text_lower):
            return family
    return "unknown"


def infer_exploit_archetype(text: str) -> str:
    text_lower = text.lower()
    for archetype, pattern in EXPLOIT_ARCHETYPE_PATTERNS.items():
        if re.search(pattern, text_lower):
            return archetype
    return "manual"


# ─── Evidence Gate Inference ─────────────────────────────────────────────────

PORT_EVIDENCE = {
    "21": "port_21_open",
    "22": "port_22_open",
    "23": "port_23_open",
    "25": "port_25_open",
    "53": "port_53_open",
    "80": "port_80_open",
    "110": "port_110_open",
    "111": "port_111_open",
    "135": "port_135_open",
    "139": "port_139_open",
    "143": "port_143_open",
    "389": "port_389_open",
    "443": "port_443_open",
    "445": "port_445_open",
    "512": "port_512_open",
    "513": "port_513_open",
    "514": "port_514_open",
    "636": "port_636_open",
    "1099": "port_1099_open",
    "1433": "port_1433_open",
    "1524": "port_1524_open",
    "2049": "port_2049_open",
    "3306": "port_3306_open",
    "3389": "port_3389_open",
    "5432": "port_5432_open",
    "5900": "port_5900_open",
    "5985": "port_5985_open",
    "6379": "port_6379_open",
    "6667": "port_6667_open",
    "8080": "port_8080_open",
    "8180": "port_8180_open",
    "8443": "port_8443_open",
    "27017": "port_27017_open",
}

SERVICE_PREREQS = {
    "smb": ["ports_discovered", "port_445_open"],
    "ssh": ["ports_discovered", "port_22_open"],
    "ftp": ["ports_discovered", "port_21_open"],
    "http": ["ports_discovered"],
    "database": ["ports_discovered"],
    "ldap": ["ports_discovered", "port_389_open"],
    "dns": ["ports_discovered", "port_53_open"],
    "kerberos": ["ports_discovered", "port_88_open"],
    "snmp": ["ports_discovered"],
    "nfs": ["ports_discovered", "port_2049_open"],
    "winrm": ["ports_discovered", "port_5985_open"],
    "rdp": ["ports_discovered", "port_3389_open"],
}


def infer_evidence_gate(text: str, service: str, phases: List[str], tools: List[str] = None) -> EvidenceGate:
    """Infer evidence gate requirements from content."""
    text_lower = text.lower()
    requirements = []
    prerequisites = ["ports_discovered"]
    anti_reqs = []

    # Port-based evidence
    for port, evidence in PORT_EVIDENCE.items():
        if port in text_lower or f":{port}" in text_lower:
            requirements.append(evidence)

    # Service-based prerequisites
    if service in SERVICE_PREREQS:
        prerequisites.extend(SERVICE_PREREQS[service])

    # Service detection requirement
    if service != "general":
        requirements.append(f"{service}_detected")

    # Phase-based requirements
    if any(p in phases for p in ["EXPLOITATION", "PRIVILEGE_ESCALATION", "LATERAL_MOVEMENT"]):
        prerequisites.append("services_identified")
    if any(p in phases for p in ["PRIVILEGE_ESCALATION", "POST_EXPLOITATION"]):
        prerequisites.append("shell_obtained")
    if any(p in phases for p in ["LATERAL_MOVEMENT"]):
        prerequisites.append("credentials_found")

    # Anti-requirements
    if "privesc" in text_lower or "privilege" in text_lower:
        anti_reqs.append("already_root")
    if any(p in phases for p in ["RECON", "ENUMERATION"]):
        anti_reqs.append("service_not_present")

    # Confidence based on content quality
    confidence = 0.5
    if any(t in text_lower for t in ["cve-", "exploit/", "msf"]):
        confidence = 0.8
    elif tools and len(tools) > 0:
        confidence = 0.6

    # Risk assessment
    risk = 0.3
    if any(p in phases for p in ["EXPLOITATION", "PRIVILEGE_ESCALATION"]):
        risk = 0.7
    elif any(p in phases for p in ["LATERAL_MOVEMENT", "POST_EXPLOITATION"]):
        risk = 0.8

    # Detection risk
    detection = 0.3
    if any(t in text_lower for t in ["stealth", "evasion", "quiet", "low and slow"]):
        detection = 0.1
    elif any(t in text_lower for t in ["noisy", "brute", "spray", "scan"]):
        detection = 0.6

    return EvidenceGate(
        evidence_requirements=list(set(requirements)),
        prerequisites=list(set(prerequisites)),
        anti_requirements=list(set(anti_reqs)),
        confidence=round(confidence, 2),
        risk=round(risk, 2),
        detection_risk=round(detection, 2),
    )


# ─── Template Mapping ────────────────────────────────────────────────────────

def map_to_templates(text: str, tools: List[str], category: str = "") -> Tuple[List[str], bool]:
    """Map entry to COMMAND_REGISTRY templates. Returns (templates, is_mappable)."""
    mapped = set()
    combined = (text + " " + category + " " + " ".join(tools)).lower()

    for tool_key, templates in TOOL_TO_TEMPLATES.items():
        if tool_key.lower() in combined:
            for t in templates:
                if t in REGISTRY_TEMPLATES:
                    mapped.add(t)

    # Direct template name matching
    for tmpl in REGISTRY_TEMPLATES:
        tmpl_words = tmpl.replace("_", " ").lower()
        if any(w in combined for w in tmpl_words.split() if len(w) > 3):
            # Only add if multiple words match for less common tools
            matching_words = [w for w in tmpl_words.split() if len(w) > 3 and w in combined]
            if len(matching_words) >= 2 or tmpl.replace("_", " ") in combined:
                mapped.add(tmpl)

    return sorted(mapped), len(mapped) > 0


# ─── MITRE ATT&CK Inference ─────────────────────────────────────────────────

MITRE_TACTIC_MAP = {
    "recon": ("TA0043", "Reconnaissance"),
    "enumeration": ("TA0007", "Discovery"),
    "exploitation": ("TA0001", "Initial Access"),
    "privilege_escalation": ("TA0004", "Privilege Escalation"),
    "lateral_movement": ("TA0008", "Lateral Movement"),
    "persistence": ("TA0003", "Persistence"),
    "credential_access": ("TA0006", "Credential Access"),
    "exfiltration": ("TA0010", "Exfiltration"),
    "defense_evasion": ("TA0005", "Defense Evasion"),
    "execution": ("TA0002", "Execution"),
    "collection": ("TA0009", "Collection"),
}


def infer_mitre(text: str, technique_id: str = "") -> MitreMapping:
    """Infer MITRE ATT&CK mapping."""
    if technique_id and technique_id.startswith("T"):
        # Have a real technique ID
        tactic = ""
        text_lower = text.lower()
        for key, (ta_id, _) in MITRE_TACTIC_MAP.items():
            if key in text_lower:
                tactic = ta_id
                break
        return MitreMapping(technique=technique_id, tactic=tactic)

    # Infer from text
    text_lower = text.lower()
    tactic_id = ""
    for key, (ta_id, _) in MITRE_TACTIC_MAP.items():
        if key.replace("_", " ") in text_lower or key in text_lower:
            tactic_id = ta_id
            break

    return MitreMapping(technique="", tactic=tactic_id)


# ─── Platform Inference ──────────────────────────────────────────────────────

def infer_platform(text: str, tools: List[str] = None) -> PlatformInfo:
    """Infer target platform from content."""
    text_lower = text.lower()
    os_list = []
    if any(w in text_lower for w in ["linux", "unix", "bash", "sh ", "/etc/", "/bin/", "chmod", "chown", "sudo"]):
        os_list.append("linux")
    if any(w in text_lower for w in ["windows", "cmd", "powershell", "mimikatz", "sam", "ntlm", ".exe", "winrm", "rdp", "active directory"]):
        os_list.append("windows")
    if not os_list:
        os_list = ["linux", "windows"]

    requires_gui = any(w in text_lower for w in ["vnc", "rdp", "gui", "desktop", "screenshot"])

    return PlatformInfo(os=os_list, arch=["x86", "x64"], requires_gui=requires_gui)


# ─── Success/Failure Indicator Inference ─────────────────────────────────────

SUCCESS_PATTERNS = {
    "shell": ["uid=", "whoami", "root@", "meterpreter", "session opened", "$ ", "# "],
    "credential": ["password:", "hash:", "credential", "SAM", "NTLM", "username:password"],
    "service": ["open", "detected", "version", "running", "banner"],
    "vuln": ["vulnerable", "exploit completed", "payload delivered", "SUCCESS"],
    "enum": ["found", "discovered", "enumerated", "shares:", "users:", "groups:"],
    "privesc": ["root", "SYSTEM", "NT AUTHORITY", "uid=0", "euid=0"],
}

FAILURE_PATTERNS = [
    "connection refused", "timeout", "access denied", "permission denied",
    "authentication failed", "no route", "not found", "error", "failed",
    "connection reset", "filtered", "closed",
]


def infer_indicators(text: str, vuln_family: str) -> Tuple[List[str], List[str]]:
    """Infer success and failure indicators."""
    success = []
    for category, patterns in SUCCESS_PATTERNS.items():
        if category in vuln_family or any(p.lower() in text.lower() for p in patterns[:2]):
            success.extend(patterns)

    if not success:
        success = ["SUCCESS", "completed", "found"]

    return list(set(success))[:8], FAILURE_PATTERNS[:6]


# ─── Extract Tags ────────────────────────────────────────────────────────────

TAG_PATTERNS = {
    "web": r"\b(web|http|https|html|php|asp|jsp|cgi|api|rest)\b",
    "ad": r"\b(active.directory|domain|dc|kerberos|ldap|gpo|ntlm|bloodhound)\b",
    "smb": r"\b(smb|samba|cifs|shares)\b",
    "ssh": r"\b(ssh|sshd|openssh)\b",
    "ftp": r"\b(ftp|vsftpd|proftpd)\b",
    "database": r"\b(mysql|postgres|mssql|oracle|sqlite|mongo|redis)\b",
    "cloud": r"\b(aws|azure|gcp|cloud|s3|ec2|lambda|iam)\b",
    "container": r"\b(docker|kubernetes|k8s|container|pod)\b",
    "wireless": r"\b(wifi|wireless|wpa|wep|802\.11)\b",
    "crypto": r"\b(crypto|encrypt|decrypt|cipher|hash|certificate|ssl|tls)\b",
    "social": r"\b(phishing|social.eng|pretexting|vishing|smishing)\b",
    "binary": r"\b(binary|reverse.eng|assembly|elf|pe|disassembl|debug)\b",
    "privesc": r"\b(privesc|privilege|suid|sudo|setuid|capability)\b",
    "persist": r"\b(persist|backdoor|implant|cron|startup|service)\b",
}


def extract_tags(text: str, category: str = "") -> List[str]:
    combined = (text + " " + category).lower()
    tags = []
    for tag, pattern in TAG_PATTERNS.items():
        if re.search(pattern, combined):
            tags.append(tag)
    return tags


# ─── Per-Source Transformers ─────────────────────────────────────────────────

def transform_ad_attack(entry: dict) -> KnowledgeCandidate:
    """Transform an ad_attacks.json entry."""
    name = entry.get("attack_name", "")
    desc = entry.get("description", "")
    cat = entry.get("category", "")
    cmds = [c.get("command", "") for c in entry.get("commands", []) if isinstance(c, dict)]
    tools = entry.get("tools_used", [])
    source = entry.get("source", "")

    full_text = f"{name} {desc} {cat} {' '.join(cmds)}"
    norm_desc = normalize_typos(desc)
    phases = infer_phases(full_text, cat)
    service = infer_service_archetype(full_text)
    vuln_fam = infer_vuln_family(full_text)
    exploit_arch = infer_exploit_archetype(full_text)
    templates, mappable = map_to_templates(full_text, tools, cat)
    success_ind, failure_ind = infer_indicators(full_text, vuln_fam)

    kc = KnowledgeCandidate(
        title=normalize_typos(name.replace("-", " ").replace("_", " ")),
        summary=norm_desc[:200].strip() if norm_desc else name,
        source=SourceInfo(
            origin="wadcoms",
            path=f"data/knowledge_base/ad_attacks.json",
            raw_ref=name,
            ingested_at=datetime.now(timezone.utc).isoformat(),
            source_hash=hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()[:16],
        ),
        taxonomy=Taxonomy(
            vuln_family=vuln_fam,
            service_archetype=service,
            exploit_archetype=exploit_arch,
            killchain_step=infer_killchain(phases),
            phase_fit=phases,
            mitre=infer_mitre(full_text),
            platform=infer_platform(full_text, tools),
            tags=extract_tags(full_text, cat),
        ),
        evidence_gate=infer_evidence_gate(full_text, service, phases, tools),
        execution=Execution(
            command_templates=templates,
            parameters={"target": "{target_ip}"},
            sequencing=[ExecutionStep(template=t, why=f"Part of {name} attack chain", expects="Success indicator") for t in templates[:5]],
            not_when=["already_root"] if "privesc" in full_text.lower() else [],
            success_indicators=success_ind,
            failure_indicators=failure_ind,
        ),
        references=References(
            tools=tools,
        ),
        governance=Governance(
            quality=QualityMetrics(
                template_mappable=mappable,
                evidence_coverage=min(1.0, len(templates) * 0.3) if mappable else 0.0,
                prereq_coverage=0.6 if mappable else 0.2,
            ),
        ),
        raw_preservation=RawPreservation(
            original_text=desc,
            original_commands=cmds,
            notes=f"Source: {source}. Category: {cat}. Tools: {', '.join(tools)}",
        ),
    )
    kc.generate_id()
    return kc


def transform_service(entry: dict) -> KnowledgeCandidate:
    """Transform a services.json entry."""
    name = entry.get("service_name", "")
    port = entry.get("port", "")
    proto = entry.get("protocol", "tcp")
    desc = entry.get("description", "")
    enum_cmds = entry.get("enumeration_commands", [])
    exploit_cmds = entry.get("exploitation_commands", [])
    default_creds = entry.get("default_credentials", [])
    vulns = entry.get("common_vulnerabilities", [])
    methodology = entry.get("methodology", "")
    tags = entry.get("tags", [])
    kcp = entry.get("kill_chain_phase", "")

    all_cmds = []
    if isinstance(enum_cmds, list):
        all_cmds.extend([c if isinstance(c, str) else str(c) for c in enum_cmds])
    if isinstance(exploit_cmds, list):
        all_cmds.extend([c if isinstance(c, str) else str(c) for c in exploit_cmds])

    full_text = f"{name} {port} {desc} {methodology} {' '.join(all_cmds)} {' '.join(str(v) for v in vulns)}"
    phases = infer_phases(full_text)
    if kcp:
        phases = list(set(phases + [kcp.upper().replace(" ", "_")]))
    service = name.lower().replace(" ", "-") if name else infer_service_archetype(full_text)
    templates, mappable = map_to_templates(full_text, tags if isinstance(tags, list) else [])
    success_ind, failure_ind = infer_indicators(full_text, infer_vuln_family(full_text))

    kc = KnowledgeCandidate(
        title=f"Service: {name} ({port}/{proto})" if port else f"Service: {name}",
        summary=normalize_typos(desc[:200]) if desc else f"Service enumeration and exploitation for {name}",
        source=SourceInfo(origin="services", path="data/knowledge_base/services.json", raw_ref=name,
                          ingested_at=datetime.now(timezone.utc).isoformat(),
                          source_hash=hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()[:16]),
        taxonomy=Taxonomy(
            vuln_family=infer_vuln_family(full_text),
            service_archetype=service,
            exploit_archetype=infer_exploit_archetype(full_text),
            killchain_step=infer_killchain(phases),
            phase_fit=phases,
            mitre=infer_mitre(full_text),
            platform=infer_platform(full_text),
            tags=extract_tags(full_text) + (tags if isinstance(tags, list) else []),
        ),
        evidence_gate=infer_evidence_gate(full_text, service, phases),
        execution=Execution(
            command_templates=templates,
            parameters={"target": "{target_ip}", "port": str(port) if port else ""},
            success_indicators=success_ind,
            failure_indicators=failure_ind,
        ),
        references=References(
            cves=[str(v) for v in vulns if isinstance(v, str) and v.startswith("CVE")],
        ),
        governance=Governance(
            quality=QualityMetrics(template_mappable=mappable,
                                   evidence_coverage=min(1.0, len(templates) * 0.25),
                                   prereq_coverage=0.7 if port else 0.3),
        ),
        raw_preservation=RawPreservation(
            original_text=desc,
            original_commands=all_cmds[:10],
            notes=f"Port: {port}/{proto}. Default creds: {len(default_creds) if isinstance(default_creds, list) else 0}. Vulns: {len(vulns) if isinstance(vulns, list) else 0}",
        ),
    )
    kc.generate_id()
    return kc


def transform_technique(entry: dict) -> KnowledgeCandidate:
    """Transform a techniques.json (MITRE ATT&CK) entry."""
    tid = entry.get("technique_id", "")
    name = entry.get("technique_name", "")
    tactic = entry.get("tactic", "")
    desc = entry.get("description", "")
    platforms = entry.get("platforms", [])
    tests = entry.get("atomic_tests", [])
    prereqs = entry.get("prerequisites", [])

    test_cmds = []
    if isinstance(tests, list):
        for t in tests:
            if isinstance(t, dict):
                for c in t.get("commands", t.get("executor", {}).get("command", "")):
                    if isinstance(c, str):
                        test_cmds.append(c)

    full_text = f"{tid} {name} {tactic} {desc} {' '.join(test_cmds[:5])}"
    phases = infer_phases(full_text, tactic)
    templates, mappable = map_to_templates(full_text, [])

    kc = KnowledgeCandidate(
        title=f"MITRE {tid}: {name}",
        summary=normalize_typos(desc[:200]) if desc else name,
        source=SourceInfo(origin="atomic-red-team", path="data/knowledge_base/techniques.json", raw_ref=tid,
                          ingested_at=datetime.now(timezone.utc).isoformat(),
                          source_hash=hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()[:16]),
        taxonomy=Taxonomy(
            vuln_family=infer_vuln_family(full_text),
            service_archetype=infer_service_archetype(full_text),
            exploit_archetype=infer_exploit_archetype(full_text),
            killchain_step=infer_killchain(phases),
            phase_fit=phases,
            mitre=MitreMapping(technique=tid, tactic=tactic if tactic else ""),
            platform=PlatformInfo(
                os=[p.lower() for p in platforms] if isinstance(platforms, list) and platforms else ["linux", "windows"],
                arch=["x86", "x64"],
                requires_gui=False,
            ),
            tags=extract_tags(full_text, tactic),
        ),
        evidence_gate=infer_evidence_gate(full_text, infer_service_archetype(full_text), phases),
        execution=Execution(
            command_templates=templates,
            parameters={"target": "{target_ip}"},
            success_indicators=["completed", "success"],
            failure_indicators=["failed", "error", "denied"],
        ),
        references=References(tools=[]),
        governance=Governance(
            quality=QualityMetrics(template_mappable=mappable,
                                   evidence_coverage=min(1.0, len(templates) * 0.2),
                                   prereq_coverage=0.5),
        ),
        raw_preservation=RawPreservation(
            original_text=desc,
            original_commands=test_cmds[:5],
            notes=f"MITRE technique {tid}. Tactic: {tactic}. Platforms: {platforms}",
        ),
    )
    kc.generate_id()
    return kc


def transform_cve(entry: dict) -> KnowledgeCandidate:
    """Transform a cves.json entry."""
    cve_id = entry.get("cve_id", "")
    desc = entry.get("description", "")
    affected = entry.get("affected_software", "")
    versions = entry.get("affected_versions", "")
    severity = entry.get("severity", "")
    methodology = entry.get("exploit_methodology", "")
    exploit_cmds = entry.get("exploit_commands", [])
    msf_module = entry.get("metasploit_module", "")
    refs = entry.get("references", [])

    all_cmds = [c if isinstance(c, str) else str(c) for c in (exploit_cmds if isinstance(exploit_cmds, list) else [])]
    full_text = f"{cve_id} {desc} {affected} {severity} {methodology} {msf_module} {' '.join(all_cmds[:3])}"
    phases = infer_phases(full_text)
    templates, mappable = map_to_templates(full_text, [])
    if msf_module:
        templates = list(set(templates + ["msfconsole_exploit"]))
        mappable = True

    kc = KnowledgeCandidate(
        title=f"{cve_id}: {str(affected)[:60]}",
        summary=normalize_typos(desc[:200]) if desc else cve_id,
        source=SourceInfo(origin="cve", path="data/knowledge_base/cves.json", raw_ref=cve_id,
                          ingested_at=datetime.now(timezone.utc).isoformat(),
                          source_hash=hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()[:16]),
        taxonomy=Taxonomy(
            vuln_family=infer_vuln_family(full_text),
            service_archetype=infer_service_archetype(full_text),
            exploit_archetype="known-exploit",
            killchain_step=infer_killchain(phases),
            phase_fit=phases,
            mitre=infer_mitre(full_text),
            platform=infer_platform(full_text),
            tags=extract_tags(full_text) + [cve_id.lower()] if cve_id else extract_tags(full_text),
        ),
        evidence_gate=infer_evidence_gate(full_text, infer_service_archetype(full_text), phases),
        execution=Execution(
            command_templates=templates,
            parameters={"target": "{target_ip}"},
            success_indicators=["exploit completed", "session", "shell"],
            failure_indicators=["not vulnerable", "failed", "timeout"],
        ),
        references=References(
            cves=[cve_id] if cve_id else [],
            modules=[msf_module] if msf_module else [],
            advisories=refs if isinstance(refs, list) else [],
        ),
        governance=Governance(
            quality=QualityMetrics(template_mappable=mappable,
                                   evidence_coverage=0.7 if mappable else 0.3,
                                   prereq_coverage=0.6 if mappable else 0.2),
        ),
        raw_preservation=RawPreservation(
            original_text=desc,
            original_commands=all_cmds[:5],
            notes=f"Severity: {severity}. Affected: {affected} {versions}. MSF: {msf_module}",
        ),
    )
    kc.generate_id()
    return kc


def transform_command(entry: dict) -> KnowledgeCandidate:
    """Transform a commands.json entry."""
    tool = entry.get("tool_name", "")
    cmd = entry.get("command", "")
    desc = entry.get("description", "")
    phase = entry.get("phase", "")
    use_case = entry.get("use_case", "")
    expected = entry.get("expected_output", "")
    related = entry.get("related_tools", [])

    full_text = f"{tool} {cmd} {desc} {phase} {use_case}"
    phases = infer_phases(full_text, phase)
    templates, mappable = map_to_templates(full_text, [tool] + (related if isinstance(related, list) else []))

    kc = KnowledgeCandidate(
        title=f"Command: {tool} - {use_case[:50]}" if use_case else f"Command: {tool}",
        summary=normalize_typos(desc[:200]) if desc else f"{tool} command usage",
        source=SourceInfo(origin="commands", path="data/knowledge_base/commands.json", raw_ref=f"{tool}:{cmd[:30]}",
                          ingested_at=datetime.now(timezone.utc).isoformat(),
                          source_hash=hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()[:16]),
        taxonomy=Taxonomy(
            vuln_family=infer_vuln_family(full_text),
            service_archetype=infer_service_archetype(full_text),
            exploit_archetype=infer_exploit_archetype(full_text),
            killchain_step=infer_killchain(phases),
            phase_fit=phases,
            mitre=infer_mitre(full_text),
            platform=infer_platform(full_text),
            tags=extract_tags(full_text),
        ),
        evidence_gate=infer_evidence_gate(full_text, infer_service_archetype(full_text), phases),
        execution=Execution(
            command_templates=templates,
            parameters={"target": "{target_ip}"},
            success_indicators=[expected[:100]] if expected else ["completed"],
            failure_indicators=["error", "failed", "not found"],
        ),
        references=References(tools=[tool] + (related if isinstance(related, list) else [])),
        governance=Governance(
            quality=QualityMetrics(template_mappable=mappable,
                                   evidence_coverage=min(1.0, len(templates) * 0.3),
                                   prereq_coverage=0.5),
        ),
        raw_preservation=RawPreservation(
            original_text=desc,
            original_commands=[cmd] if cmd else [],
            notes=f"Tool: {tool}. Phase: {phase}. Use case: {use_case}",
        ),
    )
    kc.generate_id()
    return kc


def transform_exploitdb(entry: dict) -> KnowledgeCandidate:
    """Transform an exploitdb.json entry."""
    edb_id = entry.get("edb_id", "")
    desc = entry.get("description", "")
    platform = entry.get("platform", "")
    exploit_type = entry.get("exploit_type", "")
    port = entry.get("port", "")
    cve = entry.get("cve", "")
    author = entry.get("author", "")

    full_text = f"{edb_id} {desc} {platform} {exploit_type} {port}"
    phases = ["EXPLOITATION"]
    templates, mappable = map_to_templates(full_text, [])
    if not templates:
        templates = ["searchsploit"]
        mappable = True

    cve_list = []
    if cve:
        if isinstance(cve, list):
            cve_list = [str(c) for c in cve if str(c).startswith("CVE")]
        elif isinstance(cve, str) and cve.startswith("CVE"):
            cve_list = [cve]

    kc = KnowledgeCandidate(
        title=f"EDB-{edb_id}: {desc[:60]}" if desc else f"ExploitDB {edb_id}",
        summary=normalize_typos(desc[:200]) if desc else f"ExploitDB entry {edb_id}",
        source=SourceInfo(origin="exploitdb", path="data/knowledge_base/exploitdb.json", raw_ref=str(edb_id),
                          ingested_at=datetime.now(timezone.utc).isoformat(),
                          source_hash=hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()[:16]),
        taxonomy=Taxonomy(
            vuln_family=infer_vuln_family(full_text),
            service_archetype=infer_service_archetype(full_text),
            exploit_archetype="known-exploit",
            killchain_step="foothold",
            phase_fit=phases,
            mitre=infer_mitre(full_text),
            platform=PlatformInfo(
                os=[platform.lower()] if platform else ["linux", "windows"],
                arch=["x86", "x64"],
                requires_gui=False,
            ),
            tags=extract_tags(full_text) + ([f"edb-{edb_id}"] if edb_id else []),
        ),
        evidence_gate=infer_evidence_gate(full_text, infer_service_archetype(full_text), phases),
        execution=Execution(
            command_templates=templates,
            parameters={"target": "{target_ip}", "port": str(port) if port else ""},
            success_indicators=["exploit", "shell", "session"],
            failure_indicators=["failed", "not vulnerable"],
        ),
        references=References(
            cves=cve_list,
            tools=[f"EDB-{edb_id}"],
        ),
        governance=Governance(
            quality=QualityMetrics(template_mappable=mappable,
                                   evidence_coverage=0.4,
                                   prereq_coverage=0.3),
        ),
        raw_preservation=RawPreservation(
            original_text=desc,
            original_commands=[],
            notes=f"EDB-{edb_id}. Platform: {platform}. Type: {exploit_type}. Author: {author}",
        ),
    )
    kc.generate_id()
    return kc


def transform_privesc(entry: dict, os_type: str = "linux") -> KnowledgeCandidate:
    """Transform privesc_linux/windows.json entry."""
    technique = entry.get("technique", "")
    desc = entry.get("description", "")
    cmds = entry.get("commands", [])
    entry_os = entry.get("os", os_type)

    cmd_list = [c if isinstance(c, str) else str(c) for c in (cmds if isinstance(cmds, list) else [])]
    full_text = f"{technique} {desc} {' '.join(cmd_list[:3])}"
    phases = ["PRIVILEGE_ESCALATION"]
    templates, mappable = map_to_templates(full_text, [])

    kc = KnowledgeCandidate(
        title=f"PrivEsc ({os_type}): {technique}",
        summary=normalize_typos(desc[:200]) if desc else technique,
        source=SourceInfo(origin=f"privesc_{os_type}", path=f"data/knowledge_base/privesc_{os_type}.json", raw_ref=technique,
                          ingested_at=datetime.now(timezone.utc).isoformat(),
                          source_hash=hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()[:16]),
        taxonomy=Taxonomy(
            vuln_family="privilege-escalation",
            service_archetype="general",
            exploit_archetype=infer_exploit_archetype(full_text),
            killchain_step="privesc",
            phase_fit=phases,
            mitre=MitreMapping(technique="", tactic="TA0004"),
            platform=PlatformInfo(os=[entry_os.lower() if entry_os else os_type], arch=["x86", "x64"]),
            tags=extract_tags(full_text) + ["privesc", os_type],
        ),
        evidence_gate=EvidenceGate(
            evidence_requirements=["shell_obtained"],
            prerequisites=["shell_obtained", "services_identified"],
            anti_requirements=["already_root"],
            confidence=0.6,
            risk=0.7,
            detection_risk=0.5,
        ),
        execution=Execution(
            command_templates=templates,
            parameters={"target": "{target_ip}"},
            success_indicators=["root", "SYSTEM", "uid=0", "NT AUTHORITY"],
            failure_indicators=["denied", "not permitted", "failed"],
        ),
        governance=Governance(
            quality=QualityMetrics(template_mappable=mappable,
                                   evidence_coverage=0.6 if mappable else 0.3,
                                   prereq_coverage=0.7),
        ),
        raw_preservation=RawPreservation(
            original_text=desc,
            original_commands=cmd_list[:10],
            notes=f"PrivEsc technique for {os_type}",
        ),
    )
    kc.generate_id()
    return kc


def transform_generic(entry: dict, source_file: str, origin: str) -> KnowledgeCandidate:
    """Generic transformer for remaining file types."""
    # Try to extract meaningful fields
    name = (entry.get("attack_name", "") or entry.get("technique", "") or
            entry.get("name", "") or entry.get("tool_name", "") or
            entry.get("title", "") or entry.get("service_name", "") or
            str(list(entry.values())[0])[:60] if entry else "Unknown")

    desc = (entry.get("description", "") or entry.get("methodology", "") or
            entry.get("reasoning", "") or "")

    cmds = []
    for key in ["commands", "command", "exploit_commands", "enumeration_commands"]:
        val = entry.get(key, [])
        if isinstance(val, list):
            for c in val:
                if isinstance(c, str):
                    cmds.append(c)
                elif isinstance(c, dict):
                    cmds.append(c.get("command", str(c)))
        elif isinstance(val, str) and val:
            cmds.append(val)

    cat = entry.get("category", entry.get("phase", entry.get("tactic", "")))
    full_text = f"{name} {desc} {cat} {' '.join(cmds[:3])}"

    phases = infer_phases(full_text, str(cat))
    service = infer_service_archetype(full_text)
    templates, mappable = map_to_templates(full_text, [])

    kc = KnowledgeCandidate(
        title=normalize_typos(str(name)[:80]),
        summary=normalize_typos(str(desc)[:200]) if desc else str(name),
        source=SourceInfo(origin=origin, path=f"data/knowledge_base/{source_file}", raw_ref=str(name)[:80],
                          ingested_at=datetime.now(timezone.utc).isoformat(),
                          source_hash=hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()[:16]),
        taxonomy=Taxonomy(
            vuln_family=infer_vuln_family(full_text),
            service_archetype=service,
            exploit_archetype=infer_exploit_archetype(full_text),
            killchain_step=infer_killchain(phases),
            phase_fit=phases,
            mitre=infer_mitre(full_text),
            platform=infer_platform(full_text),
            tags=extract_tags(full_text, str(cat)),
        ),
        evidence_gate=infer_evidence_gate(full_text, service, phases),
        execution=Execution(
            command_templates=templates,
            parameters={"target": "{target_ip}"},
            success_indicators=["completed", "found", "success"],
            failure_indicators=["error", "failed", "denied"],
        ),
        governance=Governance(
            quality=QualityMetrics(template_mappable=mappable,
                                   evidence_coverage=min(1.0, len(templates) * 0.2),
                                   prereq_coverage=0.4),
        ),
        raw_preservation=RawPreservation(
            original_text=str(desc)[:2000],
            original_commands=cmds[:10],
            notes=f"Source: {source_file}. Category: {cat}",
        ),
    )
    kc.generate_id()
    return kc


# ─── File → Transformer Routing ──────────────────────────────────────────────

FILE_TRANSFORMERS = {
    "ad_attacks.json": ("wadcoms", transform_ad_attack),
    "services.json": ("services", transform_service),
    "techniques.json": ("techniques", transform_technique),
    "cves.json": ("cves", transform_cve),
    "commands.json": ("commands", transform_command),
    "exploitdb.json": ("exploitdb", transform_exploitdb),
    "privesc_linux.json": ("privesc_linux", lambda e: transform_privesc(e, "linux")),
    "privesc_windows.json": ("privesc_windows", lambda e: transform_privesc(e, "windows")),
}

GENERIC_FILES = {
    "binaries.json": "binaries",
    "cheatsheets.json": "cheatsheets",
    "cloud_attacks.json": "cloud",
    "ctf_writeups.json": "ctf",
    "kill_chains.json": "kill_chains",
    "methodology.json": "methodology",
    "payloads.json": "payloads",
    "privesc_checks.json": "privesc_checks",
    "wordlists_meta.json": "wordlists",
}


# ─── Main Pipeline ───────────────────────────────────────────────────────────

def main():
    print(f"[REFACTOR V2] Starting corpus enrichment pipeline...")
    print(f"[REFACTOR V2] Registry templates available: {len(REGISTRY_TEMPLATES)}")

    os.makedirs(V2_DIR, exist_ok=True)

    total_candidates = 0
    total_quarantined = 0
    total_mappable = 0
    quarantine_entries = []
    stats = {}

    # Process each file
    all_files = dict(FILE_TRANSFORMERS)
    for fname, origin in GENERIC_FILES.items():
        all_files[fname] = (origin, None)  # None means use generic

    for filename, config in all_files.items():
        filepath = KB_DIR / filename
        if not filepath.exists():
            print(f"  [SKIP] {filename} — file not found")
            continue

        origin = config[0] if isinstance(config, tuple) else config
        transformer = config[1] if isinstance(config, tuple) and len(config) > 1 else None

        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, Exception) as e:
            print(f"  [ERROR] {filename}: {e}")
            continue

        if not isinstance(data, list):
            print(f"  [SKIP] {filename} — not a JSON array")
            continue

        if not data:
            print(f"  [SKIP] {filename} — empty array")
            continue

        candidates = []
        file_quarantine = 0
        file_mappable = 0

        for i, entry in enumerate(data):
            if not isinstance(entry, dict):
                continue
            try:
                if transformer:
                    kc = transformer(entry)
                else:
                    kc = transform_generic(entry, filename, origin)

                if kc.governance.quality.template_mappable:
                    candidates.append(kc)
                    file_mappable += 1
                else:
                    # Still include but mark as not template-mappable
                    candidates.append(kc)
            except Exception as e:
                # Quarantine bad entries
                quarantine_entries.append({
                    "source_file": filename,
                    "entry_index": i,
                    "error": str(e),
                    "entry_preview": str(entry)[:200],
                })
                file_quarantine += 1

        # Write partition JSONL
        if candidates:
            partition_name = origin.replace("/", "_")
            output_path = V2_DIR / f"{partition_name}.jsonl"
            save_candidates_jsonl(candidates, str(output_path))

        total_candidates += len(candidates)
        total_quarantined += file_quarantine
        total_mappable += file_mappable
        stats[filename] = {
            "source_entries": len(data),
            "v2_candidates": len(candidates),
            "template_mappable": file_mappable,
            "quarantined": file_quarantine,
        }

        print(f"  {filename:30s}  {len(data):>8,} → {len(candidates):>8,} candidates  ({file_mappable:,} mappable, {file_quarantine:,} quarantined)")

    # Write quarantine file
    if quarantine_entries:
        with open(QUARANTINE_PATH, "w", encoding="utf-8") as f:
            for q in quarantine_entries:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")
        print(f"\n[QUARANTINE] {len(quarantine_entries)} entries written to {QUARANTINE_PATH}")

    # Summary
    print(f"\n{'='*60}")
    print(f"[REFACTOR V2] COMPLETE")
    print(f"  Total v2 candidates:  {total_candidates:>10,}")
    print(f"  Template mappable:    {total_mappable:>10,}")
    print(f"  Quarantined:          {total_quarantined:>10,}")
    print(f"  Mappable %:           {total_mappable/max(total_candidates,1)*100:>9.1f}%")
    print(f"  Output directory:     {V2_DIR}")
    print(f"{'='*60}")

    return {
        "total_candidates": total_candidates,
        "total_mappable": total_mappable,
        "total_quarantined": total_quarantined,
        "stats": stats,
    }


if __name__ == "__main__":
    result = main()
