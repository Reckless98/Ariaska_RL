#!/usr/bin/env python3
"""
scripts/htb_readiness.py — Phase 10.0: HTB Readiness Harness

Generates machine-specific readiness reports showing:
  - Knowledge coverage (v2 candidates matching machine services/ports)
  - Command registry coverage (templates available for exploitation paths)
  - Parser broker readiness (known output patterns for tools)
  - Missing knowledge gaps

Usage:
    python scripts/htb_readiness.py                    # All machines
    python scripts/htb_readiness.py --machine cap      # Specific machine
    python scripts/htb_readiness.py --format json       # JSON output

Author: Filip Volf / Ariaska System
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Machine profiles (HTB retired + Metasploitable 2)
MACHINE_PROFILES = {
    "metasploitable2": {
        "ip": "10.0.2.4",
        "os": "linux",
        "difficulty": "easy",
        "services": {
            21: "vsftpd 2.3.4",
            22: "OpenSSH 4.7p1",
            23: "telnet",
            25: "Postfix smtpd",
            80: "Apache httpd 2.2.8",
            139: "Samba smbd 3.0.20",
            445: "Samba smbd 3.0.20",
            512: "rexecd",
            513: "rlogin",
            514: "rsh",
            1099: "Java RMI",
            1524: "ingreslock",
            2049: "NFS",
            3306: "MySQL 5.0.51a",
            5432: "PostgreSQL 8.3.0",
            5900: "VNC",
            6667: "UnrealIRCd 3.2.8.1",
            8180: "Apache Tomcat 5.5",
        },
        "key_cves": ["CVE-2007-2447", "CVE-2004-2687", "CVE-2010-2075"],
        "expected_shells": ["root via vsftpd", "root via samba", "root via ingreslock", "root via ircd"],
    },
    "cap": {
        "ip": "10.10.10.245",
        "os": "linux",
        "difficulty": "easy",
        "services": {21: "FTP", 22: "SSH", 80: "HTTP (Gunicorn)"},
        "key_cves": [],
        "expected_shells": ["user via pcap creds", "root via cap_setuid"],
    },
    "bizness": {
        "ip": "10.10.11.252",
        "os": "linux",
        "difficulty": "easy",
        "services": {22: "SSH", 80: "HTTP", 443: "HTTPS (Apache OFBiz)"},
        "key_cves": ["CVE-2023-49070"],
        "expected_shells": ["user via OFBiz RCE", "root via password hash"],
    },
    "sau": {
        "ip": "10.10.11.224",
        "os": "linux",
        "difficulty": "easy",
        "services": {22: "SSH", 80: "HTTP (filtered)", 55555: "HTTP (request-baskets)"},
        "key_cves": ["CVE-2023-27163", "CVE-2023-32315"],
        "expected_shells": ["user via Maltrail RCE", "root via systemctl"],
    },
    "sunday": {
        "ip": "10.10.10.76",
        "os": "solaris",
        "difficulty": "easy",
        "services": {79: "finger", 111: "rpcbind", 22022: "SSH", 515: "printer"},
        "key_cves": [],
        "expected_shells": ["user via finger enum + shadow", "root via sudo wget"],
    },
    "chemistry": {
        "ip": "10.10.11.38",
        "os": "linux",
        "difficulty": "easy",
        "services": {22: "SSH", 5000: "HTTP (Python)"},
        "key_cves": ["CVE-2024-23346"],
        "expected_shells": ["user via pymatgen RCE", "root via path hijack"],
    },
    "active": {
        "ip": "10.10.10.100",
        "os": "windows",
        "difficulty": "easy",
        "services": {53: "DNS", 88: "Kerberos", 135: "MSRPC", 139: "NetBIOS", 389: "LDAP", 445: "SMB", 464: "kpasswd", 593: "HTTP RPC", 636: "LDAPS", 3268: "LDAP", 3269: "LDAPS", 49152: "MSRPC"},
        "key_cves": [],
        "expected_shells": ["user via GPP creds", "root via Kerberoast"],
    },
}


def load_v2_index(index_name: str) -> Dict[str, List]:
    """Load a v2 index file."""
    idx_path = PROJECT_ROOT / "data" / "knowledge_indices" / f"{index_name}.json"
    if idx_path.exists():
        return json.loads(idx_path.read_text())
    return {}


def load_command_registry_templates() -> set:
    """Load all template names from command registry."""
    templates = set()
    try:
        # Check the v2 index
        tmpl_index = load_v2_index("by_template")
        templates.update(tmpl_index.keys())
    except Exception:
        pass

    # Also check the command registry directly
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from core.commands.command_registry import CommandRegistry
        registry = CommandRegistry()
        for tmpl in registry.get_all_templates():
            templates.add(tmpl.name)
    except Exception:
        pass

    return templates


def assess_machine(name: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    """Assess readiness for a specific machine."""
    port_index = load_v2_index("by_port")
    service_index = load_v2_index("by_service")
    cve_index = load_v2_index("by_cve")
    phase_index = load_v2_index("by_phase")
    templates = load_command_registry_templates()

    result = {
        "machine": name,
        "os": profile["os"],
        "difficulty": profile["difficulty"],
        "total_services": len(profile["services"]),
        "port_coverage": {},
        "cve_coverage": {},
        "template_availability": 0,
        "gaps": [],
        "readiness_score": 0.0,
    }

    # Port coverage
    covered_ports = 0
    for port, svc_name in profile["services"].items():
        port_str = str(port)
        candidates = port_index.get(port_str, [])
        has_coverage = len(candidates) > 0

        # Also check service name
        svc_lower = svc_name.lower().split()[0] if svc_name else ""
        svc_candidates = service_index.get(svc_lower, [])

        total = len(candidates) + len(svc_candidates)
        result["port_coverage"][port_str] = {
            "service": svc_name,
            "v2_candidates": total,
            "covered": total > 0,
        }
        if total > 0:
            covered_ports += 1
        else:
            result["gaps"].append(f"No v2 knowledge for port {port} ({svc_name})")

    # CVE coverage
    for cve_id in profile.get("key_cves", []):
        cve_candidates = cve_index.get(cve_id.upper(), [])
        result["cve_coverage"][cve_id] = {
            "v2_candidates": len(cve_candidates),
            "covered": len(cve_candidates) > 0,
        }
        if not cve_candidates:
            result["gaps"].append(f"No v2 knowledge for {cve_id}")

    # Template availability (recon + exploit phases)
    recon_templates = len(phase_index.get("RECON", []))
    exploit_templates = len(phase_index.get("EXPLOITATION", []))
    result["template_availability"] = len(templates)

    # Readiness score
    total_checks = max(len(profile["services"]) + len(profile.get("key_cves", [])), 1)
    covered = covered_ports + sum(
        1 for c in result["cve_coverage"].values() if c["covered"]
    )
    result["readiness_score"] = round(covered / total_checks * 100, 1)

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="HTB Readiness Harness")
    parser.add_argument("--machine", type=str, default="", help="Specific machine name")
    parser.add_argument("--format", type=str, choices=["text", "json"], default="text")
    args = parser.parse_args()

    machines = MACHINE_PROFILES
    if args.machine:
        if args.machine not in machines:
            print(f"Unknown machine: {args.machine}")
            print(f"Available: {', '.join(machines.keys())}")
            return
        machines = {args.machine: machines[args.machine]}

    results = {}
    for name, profile in machines.items():
        results[name] = assess_machine(name, profile)

    if args.format == "json":
        print(json.dumps(results, indent=2))
        return

    # Text output
    print("=" * 70)
    print("ARIASKA HTB READINESS REPORT — Phase 10.0")
    print("=" * 70)

    for name, r in results.items():
        score = r["readiness_score"]
        icon = "✅" if score >= 80 else "⚠️" if score >= 50 else "❌"
        print(f"\n{icon} {name.upper()} ({r['os']}, {r['difficulty']}) — {score:.0f}% ready")
        print(f"  Services: {r['total_services']}, Templates: {r['template_availability']}")

        for port, info in r["port_coverage"].items():
            status = "✓" if info["covered"] else "✗"
            print(f"    [{status}] Port {port:>5s}: {info['service']} "
                  f"({info['v2_candidates']} candidates)")

        for cve, info in r.get("cve_coverage", {}).items():
            status = "✓" if info["covered"] else "✗"
            print(f"    [{status}] {cve}: {info['v2_candidates']} candidates")

        if r["gaps"]:
            print(f"  Gaps:")
            for gap in r["gaps"][:5]:
                print(f"    ⚠ {gap}")

    # Summary
    print(f"\n{'=' * 70}")
    scores = [r["readiness_score"] for r in results.values()]
    avg = sum(scores) / max(len(scores), 1)
    print(f"Average readiness: {avg:.0f}%")
    print(f"Machines assessed: {len(results)}")
    print(f"Ready (≥80%): {sum(1 for s in scores if s >= 80)}/{len(scores)}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
