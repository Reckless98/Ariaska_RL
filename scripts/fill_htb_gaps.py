#!/usr/bin/env python3
"""
scripts/fill_htb_gaps.py — Phase 10.1 Section 1: Fill HTB readiness gaps.

Creates structured KnowledgeCandidateV2 entries for:
  1. CVE-2023-32315 (Openfire auth bypass — Sau)
  2. Port 79  (finger — Sunday)
  3. Port 515 (printer/LPD — Sunday)
  4. Port 464 (kpasswd — Active)
  5. Port 3269 (LDAPS — Active)

Appends to the correct v2 partition JSONL files.
Does NOT duplicate existing entries (checks candidate_id collision).
"""

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

V2_DIR = PROJECT_ROOT / "data" / "knowledge_candidates_v2"
NOW = datetime.now(timezone.utc).isoformat()


def make_id(origin: str, raw_ref: str, title: str) -> str:
    key = f"{origin}:{raw_ref}:{title}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def source_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ─── Gap 1: CVE-2023-32315 — Openfire auth bypass (Sau) ──────────────────────

def cve_2023_32315() -> dict:
    title = "CVE-2023-32315: Openfire Admin Console Authentication Bypass"
    cid = make_id("cve-manual", "CVE-2023-32315", title)
    return {
        "candidate_id": cid,
        "schema_version": "2.0.0",
        "title": title,
        "summary": (
            "Openfire Admin Console before 4.7.5 and 4.6.8 is vulnerable to "
            "a path traversal attack via the setup environment that allows an "
            "unauthenticated user to access restricted pages. Combined with "
            "CVE-2023-32315, an attacker can upload a malicious plugin to "
            "achieve remote code execution."
        ),
        "source": {
            "origin": "cve-manual",
            "path": "manual/CVE-2023-32315",
            "raw_ref": "CVE-2023-32315",
            "ingested_at": NOW,
            "source_hash": source_hash("CVE-2023-32315-openfire-auth-bypass"),
        },
        "taxonomy": {
            "vuln_family": "authentication-bypass",
            "service_archetype": "web-admin-console",
            "exploit_archetype": "path-traversal",
            "killchain_step": "foothold",
            "phase_fit": ["EXPLOITATION", "ENUMERATION"],
            "mitre": {"technique": "T1190", "tactic": "TA0001"},
            "platform": {"os": ["linux", "windows"], "arch": ["x86", "x64"], "requires_gui": False},
            "tags": [
                "openfire", "CVE-2023-32315", "auth-bypass", "path-traversal",
                "web", "admin-console", "plugin-upload", "rce", "9090", "9091",
            ],
        },
        "evidence_gate": {
            "evidence_requirements": ["openfire_detected", "port_9090_open"],
            "prerequisites": ["ports_discovered"],
            "anti_requirements": ["openfire_patched"],
            "confidence": 0.85,
            "risk": 0.6,
            "detection_risk": 0.4,
        },
        "execution": {
            "command_templates": [
                "nmap_service_version", "nmap_vuln_scan", "searchsploit",
                "hydra_http_form", "nmap_full_tcp",
            ],
            "parameters": {"target": "{target_ip}", "port": "9090"},
            "sequencing": [
                {"template": "nmap_service_version", "why": "Confirm Openfire version", "expects": "version < 4.7.5"},
                {"template": "searchsploit", "why": "Find CVE-2023-32315 exploit", "expects": "exploit found"},
            ],
            "not_when": ["openfire_version_ge_4.7.5"],
            "success_indicators": ["Openfire", "4.7", "admin", "setup", "plugin", "upload", "authenticated"],
            "failure_indicators": ["connection refused", "timeout", "403 Forbidden", "patched"],
            "verification": {"template": "nmap_service_version", "expects": "openfire running"},
        },
        "references": {
            "cves": ["CVE-2023-32315"],
            "advisories": ["GHSA-gw42-f939-fhvm"],
            "papers": [],
            "tools": ["searchsploit", "curl", "msfconsole"],
            "modules": ["exploit/multi/http/openfire_auth_bypass_upload_rce_cve_2023_32315"],
        },
        "governance": {
            "allowed_scopes": ["htb", "tryhackme", "lab", "ctf"],
            "safety_constraints": ["registry_only", "evidence_gated", "no_freeform"],
            "quality": {"template_mappable": True, "evidence_coverage": 1.0, "prereq_coverage": 0.9},
        },
        "runtime_stats": {"times_offered": 0, "times_selected": 0, "times_succeeded": 0},
        "raw_preservation": {
            "original_text": (
                "Openfire Admin Console authentication bypass via path traversal. "
                "GET /setup/setup-s/%u002e%u002e/%u002e%u002e/log.jsp bypasses auth. "
                "Upload malicious .jar plugin for RCE. Affects Openfire < 4.7.5."
            ),
            "original_commands": [
                "curl -k 'http://{target}:9090/setup/setup-s/%u002e%u002e/%u002e%u002e/log.jsp'",
                "msfconsole -q -x 'use exploit/multi/http/openfire_auth_bypass_upload_rce_cve_2023_32315; set RHOSTS {target}; run'",
            ],
            "notes": "HTB Sau exploitation chain: request-baskets SSRF → Openfire auth bypass → RCE",
        },
    }


# ─── Gap 2: Port 79 — finger (Sunday / Solaris) ──────────────────────────────

def port_79_finger() -> dict:
    title = "Service: finger (79/tcp) — User Enumeration"
    cid = make_id("service-manual", "finger-79", title)
    return {
        "candidate_id": cid,
        "schema_version": "2.0.0",
        "title": title,
        "summary": (
            "The finger protocol (port 79) reveals logged-in users, home "
            "directories, shells, and idle times. On Solaris systems, finger "
            "enumeration often reveals usernames for brute-force SSH attacks. "
            "finger @target lists all users; finger username@target shows details."
        ),
        "source": {
            "origin": "service-manual",
            "path": "manual/finger-79",
            "raw_ref": "finger-79",
            "ingested_at": NOW,
            "source_hash": source_hash("finger-79-user-enum"),
        },
        "taxonomy": {
            "vuln_family": "info-disclosure",
            "service_archetype": "finger",
            "exploit_archetype": "user-enumeration",
            "killchain_step": "recon",
            "phase_fit": ["RECON", "ENUMERATION"],
            "mitre": {"technique": "T1589.001", "tactic": "TA0043"},
            "platform": {"os": ["solaris", "linux", "unix"], "arch": ["sparc", "x86", "x64"], "requires_gui": False},
            "tags": ["finger", "user-enumeration", "79", "solaris", "unix", "info-disclosure", "recon"],
        },
        "evidence_gate": {
            "evidence_requirements": ["finger_service_detected"],
            "prerequisites": ["ports_discovered"],
            "anti_requirements": ["finger_disabled"],
            "confidence": 0.9,
            "risk": 0.2,
            "detection_risk": 0.2,
        },
        "execution": {
            "command_templates": [
                "nmap_service_version", "nmap_full_tcp", "nmap_quick_scan",
            ],
            "parameters": {"target": "{target_ip}", "port": "79"},
            "sequencing": [
                {"template": "nmap_service_version", "why": "Confirm finger service on port 79", "expects": "finger open"},
            ],
            "not_when": [],
            "success_indicators": ["Login", "Name", "Directory", "Shell", "finger", "user"],
            "failure_indicators": ["connection refused", "timeout", "filtered"],
            "verification": None,
        },
        "references": {
            "cves": [],
            "advisories": [],
            "papers": [],
            "tools": ["finger", "finger-user-enum", "nmap"],
            "modules": ["auxiliary/scanner/finger/finger_users"],
        },
        "governance": {
            "allowed_scopes": ["htb", "tryhackme", "lab", "ctf"],
            "safety_constraints": ["registry_only", "evidence_gated", "no_freeform"],
            "quality": {"template_mappable": True, "evidence_coverage": 1.0, "prereq_coverage": 0.9},
        },
        "runtime_stats": {"times_offered": 0, "times_selected": 0, "times_succeeded": 0},
        "raw_preservation": {
            "original_text": "finger @{target} — list logged-in users. finger root@{target} — query specific user.",
            "original_commands": [
                "finger @{target}",
                "finger root@{target}",
                "finger-user-enum.pl -U /usr/share/seclists/Usernames/Names/names.txt -t {target}",
                "nmap -sV -p 79 --script finger {target}",
            ],
            "notes": "HTB Sunday: finger enum reveals sammy and sunny users for SSH brute-force on port 22022.",
        },
    }


# ─── Gap 3: Port 515 — printer / LPD (Sunday) ───────────────────────────────

def port_515_printer() -> dict:
    title = "Service: printer/LPD (515/tcp) — Line Printer Daemon"
    cid = make_id("service-manual", "printer-515", title)
    return {
        "candidate_id": cid,
        "schema_version": "2.0.0",
        "title": title,
        "summary": (
            "Line Printer Daemon (LPD) on port 515. Can expose print queue info, "
            "reveal hostnames, and on misconfigured systems allow file reads or "
            "command execution. On Solaris, LPD may be combined with other legacy "
            "services for lateral movement."
        ),
        "source": {
            "origin": "service-manual",
            "path": "manual/printer-515",
            "raw_ref": "printer-515",
            "ingested_at": NOW,
            "source_hash": source_hash("printer-515-lpd"),
        },
        "taxonomy": {
            "vuln_family": "info-disclosure",
            "service_archetype": "printer-lpd",
            "exploit_archetype": "service-enumeration",
            "killchain_step": "recon",
            "phase_fit": ["RECON", "ENUMERATION"],
            "mitre": {"technique": "T1005", "tactic": "TA0009"},
            "platform": {"os": ["solaris", "linux", "unix"], "arch": ["sparc", "x86", "x64"], "requires_gui": False},
            "tags": ["printer", "lpd", "515", "solaris", "unix", "line-printer-daemon", "print-queue"],
        },
        "evidence_gate": {
            "evidence_requirements": ["printer_lpd_detected"],
            "prerequisites": ["ports_discovered"],
            "anti_requirements": ["lpd_disabled"],
            "confidence": 0.7,
            "risk": 0.2,
            "detection_risk": 0.2,
        },
        "execution": {
            "command_templates": [
                "nmap_service_version", "nmap_full_tcp", "nmap_quick_scan",
            ],
            "parameters": {"target": "{target_ip}", "port": "515"},
            "sequencing": [
                {"template": "nmap_service_version", "why": "Confirm LPD service on port 515", "expects": "lpd or printer detected"},
            ],
            "not_when": [],
            "success_indicators": ["printer", "lpd", "queue", "ready", "Solaris"],
            "failure_indicators": ["connection refused", "timeout", "filtered"],
            "verification": None,
        },
        "references": {
            "cves": [],
            "advisories": [],
            "papers": [],
            "tools": ["lpc", "lpq", "nmap"],
            "modules": ["auxiliary/scanner/printer/printer_list_queues"],
        },
        "governance": {
            "allowed_scopes": ["htb", "tryhackme", "lab", "ctf"],
            "safety_constraints": ["registry_only", "evidence_gated", "no_freeform"],
            "quality": {"template_mappable": True, "evidence_coverage": 1.0, "prereq_coverage": 0.8},
        },
        "runtime_stats": {"times_offered": 0, "times_selected": 0, "times_succeeded": 0},
        "raw_preservation": {
            "original_text": "LPD on port 515. Check print queues, attempt file read via lpq/lpc.",
            "original_commands": [
                "nmap -sV -p 515 --script lpd-info {target}",
                "lpq -h {target}",
            ],
            "notes": "HTB Sunday: LPD present but not primary attack vector. Useful for host enumeration.",
        },
    }


# ─── Gap 4: Port 464 — kpasswd (Active / AD) ────────────────────────────────

def port_464_kpasswd() -> dict:
    title = "Service: kpasswd (464/tcp) — Kerberos Password Change"
    cid = make_id("service-manual", "kpasswd-464", title)
    return {
        "candidate_id": cid,
        "schema_version": "2.0.0",
        "title": title,
        "summary": (
            "Kerberos password change service (kpasswd) on port 464. Indicates "
            "Active Directory domain controller. Can be used to confirm Kerberos "
            "presence, enumerate domain info, and attempt password changes for "
            "compromised accounts. Integral to Kerberoasting attack chains."
        ),
        "source": {
            "origin": "service-manual",
            "path": "manual/kpasswd-464",
            "raw_ref": "kpasswd-464",
            "ingested_at": NOW,
            "source_hash": source_hash("kpasswd-464-kerberos"),
        },
        "taxonomy": {
            "vuln_family": "active-directory",
            "service_archetype": "kerberos",
            "exploit_archetype": "credential-attack",
            "killchain_step": "foothold",
            "phase_fit": ["ENUMERATION", "EXPLOITATION", "PRIVILEGE_ESCALATION"],
            "mitre": {"technique": "T1558.003", "tactic": "TA0006"},
            "platform": {"os": ["windows"], "arch": ["x64"], "requires_gui": False},
            "tags": [
                "kpasswd", "kerberos", "464", "active-directory", "domain-controller",
                "kerberoast", "password-change", "credential-attack",
            ],
        },
        "evidence_gate": {
            "evidence_requirements": ["kerberos_detected", "domain_controller_identified"],
            "prerequisites": ["ports_discovered"],
            "anti_requirements": [],
            "confidence": 0.8,
            "risk": 0.5,
            "detection_risk": 0.4,
        },
        "execution": {
            "command_templates": [
                "nmap_service_version", "nmap_full_tcp", "nmap_vuln_scan",
                "ldapsearch_base", "ldapsearch_users",
                "cme_smb_shares", "cme_smb_bruteforce", "cme_ldap_users",
                "enum4linux_full",
            ],
            "parameters": {"target": "{target_ip}", "port": "464", "domain": "{domain}"},
            "sequencing": [
                {"template": "nmap_service_version", "why": "Confirm kpasswd on 464", "expects": "kpasswd open"},
                {"template": "ldapsearch_base", "why": "Enumerate domain base DN", "expects": "domain info"},
                {"template": "cme_ldap_users", "why": "Enumerate domain users for Kerberoast", "expects": "user list"},
            ],
            "not_when": [],
            "success_indicators": ["kpasswd", "kerberos", "Active Directory", "domain", "DC"],
            "failure_indicators": ["connection refused", "timeout", "filtered"],
            "verification": None,
        },
        "references": {
            "cves": [],
            "advisories": [],
            "papers": [],
            "tools": ["impacket", "kerbrute", "GetUserSPNs.py", "crackmapexec", "ldapsearch"],
            "modules": ["auxiliary/gather/kerberos_enumusers"],
        },
        "governance": {
            "allowed_scopes": ["htb", "tryhackme", "lab", "ctf"],
            "safety_constraints": ["registry_only", "evidence_gated", "no_freeform"],
            "quality": {"template_mappable": True, "evidence_coverage": 1.0, "prereq_coverage": 0.9},
        },
        "runtime_stats": {"times_offered": 0, "times_selected": 0, "times_succeeded": 0},
        "raw_preservation": {
            "original_text": "kpasswd on 464 — Kerberos password change. Confirms AD DC.",
            "original_commands": [
                "nmap -sV -p 464 {target}",
                "impacket-GetUserSPNs {domain}/{user}:{pass} -dc-ip {target} -request",
                "kpasswd {user}@{domain}",
            ],
            "notes": "HTB Active: kpasswd confirms DC. Chain: SMB anon → GPP creds → Kerberoast → Admin.",
        },
    }


# ─── Gap 5: Port 3269 — LDAPS (Active / AD) ─────────────────────────────────

def port_3269_ldaps() -> dict:
    title = "Service: LDAPS Global Catalog (3269/tcp) — Encrypted LDAP"
    cid = make_id("service-manual", "ldaps-3269", title)
    return {
        "candidate_id": cid,
        "schema_version": "2.0.0",
        "title": title,
        "summary": (
            "LDAPS Global Catalog on port 3269 provides SSL-encrypted LDAP access "
            "to Active Directory. Allows domain enumeration (users, groups, GPOs, "
            "SPNs) when credentials are available. Key indicator of domain controller "
            "presence. Can be queried with ldapsearch using TLS."
        ),
        "source": {
            "origin": "service-manual",
            "path": "manual/ldaps-3269",
            "raw_ref": "ldaps-3269",
            "ingested_at": NOW,
            "source_hash": source_hash("ldaps-3269-global-catalog"),
        },
        "taxonomy": {
            "vuln_family": "active-directory",
            "service_archetype": "ldap",
            "exploit_archetype": "credential-attack",
            "killchain_step": "recon",
            "phase_fit": ["RECON", "ENUMERATION", "EXPLOITATION"],
            "mitre": {"technique": "T1018", "tactic": "TA0007"},
            "platform": {"os": ["windows"], "arch": ["x64"], "requires_gui": False},
            "tags": [
                "ldaps", "ldap", "3269", "active-directory", "global-catalog",
                "domain-controller", "ssl", "tls", "domain-enumeration",
            ],
        },
        "evidence_gate": {
            "evidence_requirements": ["ldap_service_detected", "domain_controller_identified"],
            "prerequisites": ["ports_discovered"],
            "anti_requirements": [],
            "confidence": 0.8,
            "risk": 0.3,
            "detection_risk": 0.3,
        },
        "execution": {
            "command_templates": [
                "nmap_service_version", "nmap_full_tcp", "nmap_vuln_scan",
                "ldapsearch_base", "ldapsearch_users",
                "cme_ldap_users", "enum4linux_full",
            ],
            "parameters": {"target": "{target_ip}", "port": "3269", "domain": "{domain}"},
            "sequencing": [
                {"template": "nmap_service_version", "why": "Confirm LDAPS on 3269", "expects": "ldaps/ssl open"},
                {"template": "ldapsearch_base", "why": "Query domain base DN via TLS", "expects": "domain info"},
                {"template": "ldapsearch_users", "why": "Enumerate domain users", "expects": "user objects"},
            ],
            "not_when": [],
            "success_indicators": ["LDAP", "SSL", "Global Catalog", "domain", "DC", "basedn"],
            "failure_indicators": ["connection refused", "timeout", "certificate error"],
            "verification": None,
        },
        "references": {
            "cves": [],
            "advisories": [],
            "papers": [],
            "tools": ["ldapsearch", "crackmapexec", "bloodhound", "enum4linux-ng"],
            "modules": [],
        },
        "governance": {
            "allowed_scopes": ["htb", "tryhackme", "lab", "ctf"],
            "safety_constraints": ["registry_only", "evidence_gated", "no_freeform"],
            "quality": {"template_mappable": True, "evidence_coverage": 1.0, "prereq_coverage": 0.8},
        },
        "runtime_stats": {"times_offered": 0, "times_selected": 0, "times_succeeded": 0},
        "raw_preservation": {
            "original_text": "LDAPS Global Catalog on 3269 — encrypted LDAP for AD enumeration.",
            "original_commands": [
                "nmap -sV -p 3269 --script ssl-cert {target}",
                "ldapsearch -H ldaps://{target}:3269 -x -b '' -s base namingContexts",
                "crackmapexec ldap {target} -u {user} -p {pass} --users",
            ],
            "notes": "HTB Active: LDAPS GC confirms DC. Supports anonymous base DN queries.",
        },
    }


def existing_ids(partition_path: Path) -> set:
    """Collect existing candidate_ids from a JSONL file."""
    ids = set()
    if partition_path.exists():
        with open(partition_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        ids.add(json.loads(line).get("candidate_id", ""))
                    except json.JSONDecodeError:
                        pass
    return ids


def append_candidate(partition: str, candidate: dict) -> bool:
    """Append a single candidate to a v2 partition JSONL. Returns True if written."""
    path = V2_DIR / f"{partition}.jsonl"
    cid = candidate["candidate_id"]

    ids = existing_ids(path)
    if cid in ids:
        print(f"  SKIP {cid} — already exists in {partition}.jsonl")
        return False

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(candidate, ensure_ascii=False) + "\n")
    print(f"  ADDED {cid} → {partition}.jsonl  [{candidate['title'][:60]}]")
    return True


def main():
    print("=" * 70)
    print("Phase 10.1 Section 1 — Fill HTB Readiness Gaps")
    print("=" * 70)

    gaps = [
        ("cves",     cve_2023_32315()),
        ("services", port_79_finger()),
        ("services", port_515_printer()),
        ("services", port_464_kpasswd()),
        ("services", port_3269_ldaps()),
    ]

    added = 0
    for partition, candidate in gaps:
        if append_candidate(partition, candidate):
            added += 1

    print(f"\nTotal added: {added}/{len(gaps)}")
    print(f"\nNow re-run: python scripts/build_knowledge_indices.py")
    print(f"Then re-run: python scripts/htb_readiness.py")


if __name__ == "__main__":
    main()
