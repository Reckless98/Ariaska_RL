#!/usr/bin/env python3
"""
Extract HTB Walkthroughs → Structured Knowledge + V2 Candidates.

Phase 10.0 — Knowledge Governance.

Parses markdown walkthrough files from data/htb_walkthroughs/
Extracts: machine info, commands, vulnerabilities, techniques, tools
Outputs:
  data/htb_extractions/<machine>.json  — structured extraction per machine
  data/knowledge_candidates_v2/htb.jsonl — enriched v2 candidates merged
"""

import json
import os
import re
import sys
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.knowledge.knowledge_candidate_v2 import (
    KnowledgeCandidate, SourceInfo, Taxonomy, MitreMapping, PlatformInfo,
    EvidenceGate, Execution, ExecutionStep, References, Governance,
    QualityMetrics, RawPreservation, save_candidates_jsonl,
)

HTB_DIR = PROJECT_ROOT / "data" / "htb_walkthroughs"
EXTRACT_DIR = PROJECT_ROOT / "data" / "htb_extractions"
V2_DIR = PROJECT_ROOT / "data" / "knowledge_candidates_v2"

# Registry templates for mapping
REGISTRY_TEMPLATES = set()
try:
    content = (PROJECT_ROOT / "core" / "commands" / "command_registry.py").read_text()
    for m in re.finditer(r'name="([^"]+)"', content):
        REGISTRY_TEMPLATES.add(m.group(1))
except Exception:
    pass

# Tool → template mapping (subset for HTB context)
TOOL_TEMPLATE_MAP = {
    "nmap": ["nmap_quick_scan", "nmap_service_version", "nmap_vuln_scan", "nmap_full_tcp"],
    "nikto": ["nikto_scan"],
    "gobuster": ["gobuster_dir"],
    "dirsearch": ["dirsearch"],
    "ffuf": ["ffuf_fuzz"],
    "sqlmap": ["sqlmap_test", "sqlmap_get"],
    "hydra": ["hydra_ssh", "hydra_ftp", "hydra_http_form"],
    "linpeas": ["linpeas"],
    "winpeas": ["winpeas"],
    "enum4linux": ["enum4linux_scan"],
    "smbclient": ["smbclient_list"],
    "smbmap": ["smbmap_shares"],
    "searchsploit": ["searchsploit"],
    "metasploit": ["msfconsole_exploit"],
    "msfconsole": ["msfconsole_exploit"],
    "curl": ["curl_headers"],
    "whatweb": ["whatweb"],
    "wpscan": ["wpscan"],
    "ssh": ["ssh_login"],
    "ftp": ["ftp_anonymous"],
    "mysql": ["mysql_login"],
    "psql": ["psql_default_creds"],
    "crackmapexec": ["crackmapexec_smb_bruteforce"],
    "bloodhound": ["bloodhound_python"],
    "impacket": ["impacket_psexec", "impacket_secretsdump"],
    "john": [],
    "hashcat": [],
    "chisel": ["chisel_server", "chisel_client"],
    "nc": ["revshell_bash"],
    "python": ["revshell_python"],
    "sudo": ["sudo_check", "sudo_list"],
    "find": ["find_suid", "find_sgid"],
}


def extract_commands(text: str) -> List[str]:
    """Extract shell commands from markdown code blocks."""
    commands = []
    # Match ```bash, ```sh, ```shell, ``` code blocks
    for m in re.finditer(r'```(?:bash|sh|shell|zsh|kali|console)?\s*\n(.*?)```', text, re.DOTALL):
        block = m.group(1).strip()
        for line in block.split('\n'):
            line = line.strip()
            # Skip comments, empty lines, output lines
            if not line or line.startswith('#') or line.startswith('//'):
                continue
            # Skip typical output indicators
            if any(line.startswith(p) for p in ['[', '|', '+', '-', '=', 'PORT', 'STATE', 'SERVICE']):
                continue
            # Clean prompt prefixes
            for prefix in ['$ ', '# ', '> ', '└─$ ', '└─# ', '❯ ']:
                if line.startswith(prefix):
                    line = line[len(prefix):]
                    break
            if line.startswith('<strong>'):
                line = re.sub(r'</?strong>', '', line)
            if line and len(line) > 3:
                commands.append(line)

    # Also match inline code that looks like commands
    for m in re.finditer(r'<pre[^>]*><code[^>]*>(.*?)</code></pre>', text, re.DOTALL):
        block = re.sub(r'<[^>]+>', '', m.group(1)).strip()
        for line in block.split('\n'):
            line = line.strip()
            if line and len(line) > 3 and not line.startswith('['):
                commands.append(line)

    return commands


def extract_machine_info(text: str, filename: str) -> Dict:
    """Extract machine metadata from walkthrough."""
    machine_name = filename.replace('.md', '').title()

    # Extract description
    desc_match = re.search(r'### Machine Description\s*\n(.*?)(?=\n###|\n##|\n<)', text, re.DOTALL)
    description = desc_match.group(1).strip() if desc_match else ""

    # Extract target IP
    ip_match = re.search(r'Target IP[:\s]*`?(\d+\.\d+\.\d+\.\d+)', text)
    target_ip = ip_match.group(1) if ip_match else "10.10.10.x"

    # Extract vulnerabilities
    vuln_match = re.search(r'### Vulnerabilities\s*\n(.*?)(?=\n###|\n##)', text, re.DOTALL)
    vulns = [v.strip() for v in vuln_match.group(1).split('|')] if vuln_match else []

    # Extract tools
    tools_match = re.search(r'### Security Tools\s*\n(.*?)(?=\n###|\n##)', text, re.DOTALL)
    tools = [t.strip() for t in tools_match.group(1).split('|')] if tools_match else []

    # Extract techniques
    tech_match = re.search(r'### Techniques\s*\n(.*?)(?=\n###|\n##)', text, re.DOTALL)
    techniques = [t.strip() for t in tech_match.group(1).split('|')] if tech_match else []

    # Extract OS
    os_type = "linux"
    if any(w in text.lower() for w in ["windows", "powershell", "cmd.exe", "aspx", "iis"]):
        os_type = "windows"

    # Extract CVEs
    cves = list(set(re.findall(r'CVE-\d{4}-\d{4,}', text)))

    return {
        "machine_name": machine_name,
        "target_ip": target_ip,
        "description": description,
        "vulnerabilities": vulns,
        "tools": tools,
        "techniques": techniques,
        "os": os_type,
        "cves": cves,
    }


def extract_sections(text: str) -> List[Dict]:
    """Extract task/section-level steps from walkthrough."""
    sections = []
    # Match ## Task N or ## sections
    parts = re.split(r'\n## (.*?)\n', text)
    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        content = parts[i + 1] if i + 1 < len(parts) else ""
        commands = extract_commands(content)
        sections.append({
            "title": title,
            "content": content[:1000],
            "commands": commands,
        })
    return sections


def map_commands_to_templates(commands: List[str], tools: List[str]) -> List[str]:
    """Map extracted commands to registry templates."""
    templates = set()
    combined = " ".join(commands + tools).lower()

    for tool, tmpls in TOOL_TEMPLATE_MAP.items():
        if tool.lower() in combined:
            for t in tmpls:
                if t in REGISTRY_TEMPLATES:
                    templates.add(t)

    return sorted(templates)


def create_candidates_from_machine(machine_info: Dict, sections: List[Dict], filepath: str) -> List[KnowledgeCandidate]:
    """Create KnowledgeCandidate v2 objects from extracted machine data."""
    candidates = []
    machine = machine_info["machine_name"]
    all_commands = []
    for s in sections:
        all_commands.extend(s.get("commands", []))

    templates = map_commands_to_templates(all_commands, machine_info.get("tools", []))

    # Main machine candidate
    kc = KnowledgeCandidate(
        title=f"HTB: {machine} — Full Attack Chain",
        summary=machine_info.get("description", f"HackTheBox {machine} machine walkthrough")[:200],
        source=SourceInfo(
            origin="htb",
            path=filepath,
            raw_ref=machine,
            ingested_at=datetime.now(timezone.utc).isoformat(),
            source_hash=hashlib.sha256(json.dumps(machine_info, sort_keys=True).encode()).hexdigest()[:16],
        ),
        taxonomy=Taxonomy(
            vuln_family="multi-stage",
            service_archetype="multi-service",
            exploit_archetype="known-exploit",
            killchain_step="foothold",
            phase_fit=["RECON", "ENUMERATION", "EXPLOITATION", "PRIVILEGE_ESCALATION"],
            mitre=MitreMapping(technique="", tactic="TA0001"),
            platform=PlatformInfo(
                os=[machine_info.get("os", "linux")],
                arch=["x86", "x64"],
                requires_gui=False,
            ),
            tags=["htb", machine.lower()] + [t.lower() for t in machine_info.get("tools", [])],
        ),
        evidence_gate=EvidenceGate(
            evidence_requirements=["ports_discovered"],
            prerequisites=["ports_discovered"],
            confidence=0.8,
            risk=0.5,
            detection_risk=0.3,
        ),
        execution=Execution(
            command_templates=templates,
            parameters={"target": "{target_ip}"},
            sequencing=[
                ExecutionStep(template=t, why=f"Part of {machine} attack chain", expects="progress")
                for t in templates[:8]
            ],
            success_indicators=["root", "shell", "flag", "user.txt", "root.txt"],
            failure_indicators=["denied", "failed", "timeout"],
        ),
        references=References(
            cves=machine_info.get("cves", []),
            tools=machine_info.get("tools", []),
        ),
        governance=Governance(
            allowed_scopes=["htb", "lab", "ctf"],
            quality=QualityMetrics(
                template_mappable=len(templates) > 0,
                evidence_coverage=min(1.0, len(templates) * 0.15),
                prereq_coverage=0.7,
            ),
        ),
        raw_preservation=RawPreservation(
            original_text=machine_info.get("description", ""),
            original_commands=all_commands[:30],
            notes=f"HTB machine: {machine}. Vulns: {', '.join(machine_info.get('vulnerabilities', []))}. Techniques: {', '.join(machine_info.get('techniques', []))}",
        ),
    )
    kc.generate_id()
    candidates.append(kc)

    # Per-section candidates for multi-step chains
    for section in sections:
        if not section.get("commands"):
            continue
        title = section["title"]
        sec_templates = map_commands_to_templates(section["commands"], machine_info.get("tools", []))

        # Determine phase from section title
        phases = ["ENUMERATION"]
        title_lower = title.lower()
        if "recon" in title_lower or "scan" in title_lower or "task 1" in title_lower:
            phases = ["RECON", "ENUMERATION"]
        elif "exploit" in title_lower or "foothold" in title_lower or "initial" in title_lower:
            phases = ["EXPLOITATION"]
        elif "priv" in title_lower or "escal" in title_lower or "root" in title_lower:
            phases = ["PRIVILEGE_ESCALATION"]
        elif "lateral" in title_lower or "pivot" in title_lower:
            phases = ["LATERAL_MOVEMENT"]
        elif "exfil" in title_lower or "flag" in title_lower or "loot" in title_lower:
            phases = ["POST_EXPLOITATION", "EXFILTRATION"]

        skc = KnowledgeCandidate(
            title=f"HTB {machine}: {title[:50]}",
            summary=f"{machine} — {title}: {len(section['commands'])} commands extracted",
            source=SourceInfo(origin="htb", path=filepath, raw_ref=f"{machine}/{title}",
                              ingested_at=datetime.now(timezone.utc).isoformat(),
                              source_hash=hashlib.sha256(f"{machine}:{title}".encode()).hexdigest()[:16]),
            taxonomy=Taxonomy(
                vuln_family="multi-stage",
                service_archetype="multi-service",
                exploit_archetype="known-exploit",
                killchain_step="foothold" if "exploit" in title_lower else "recon",
                phase_fit=phases,
                platform=PlatformInfo(os=[machine_info.get("os", "linux")]),
                tags=["htb", machine.lower(), title_lower.split()[0] if title_lower else ""],
            ),
            evidence_gate=EvidenceGate(
                evidence_requirements=["ports_discovered"],
                prerequisites=["ports_discovered"],
                confidence=0.7,
                risk=0.5,
            ),
            execution=Execution(
                command_templates=sec_templates,
                parameters={"target": "{target_ip}"},
                success_indicators=["found", "open", "success"],
                failure_indicators=["error", "denied"],
            ),
            governance=Governance(
                allowed_scopes=["htb", "lab", "ctf"],
                quality=QualityMetrics(
                    template_mappable=len(sec_templates) > 0,
                    evidence_coverage=min(1.0, len(sec_templates) * 0.2),
                    prereq_coverage=0.5,
                ),
            ),
            raw_preservation=RawPreservation(
                original_text=section["content"][:500],
                original_commands=section["commands"][:15],
                notes=f"HTB {machine} section: {title}",
            ),
        )
        skc.generate_id()
        candidates.append(skc)

    return candidates


def main():
    print(f"[HTB EXTRACT] Scanning {HTB_DIR} ...")
    os.makedirs(EXTRACT_DIR, exist_ok=True)

    md_files = sorted(HTB_DIR.glob("*.md"))
    md_files = [f for f in md_files if f.name != "README.md"]
    print(f"[HTB EXTRACT] Found {len(md_files)} walkthrough files")

    all_candidates = []
    extraction_stats = {}

    for md_file in md_files:
        machine_name = md_file.stem
        try:
            text = md_file.read_text(encoding="utf-8")
        except Exception as e:
            print(f"  [ERROR] {md_file.name}: {e}")
            continue

        machine_info = extract_machine_info(text, md_file.name)
        sections = extract_sections(text)
        commands_total = sum(len(s.get("commands", [])) for s in sections)

        # Save structured extraction
        extraction = {
            "machine": machine_info,
            "sections": sections,
            "total_commands": commands_total,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
        }
        extract_path = EXTRACT_DIR / f"{machine_name}.json"
        extract_path.write_text(json.dumps(extraction, indent=2, ensure_ascii=False))

        # Create v2 candidates
        candidates = create_candidates_from_machine(machine_info, sections, str(md_file))
        all_candidates.extend(candidates)

        extraction_stats[machine_name] = {
            "sections": len(sections),
            "commands": commands_total,
            "candidates": len(candidates),
            "tools": machine_info.get("tools", []),
            "vulns": machine_info.get("vulnerabilities", []),
            "cves": machine_info.get("cves", []),
        }

        print(f"  {machine_name:20s}  {len(sections):>3} sections  {commands_total:>4} commands  → {len(candidates):>3} candidates")

    # Write v2 JSONL
    if all_candidates:
        htb_output = V2_DIR / "htb.jsonl"
        save_candidates_jsonl(all_candidates, str(htb_output))

    print(f"\n{'='*60}")
    print(f"[HTB EXTRACT] COMPLETE")
    print(f"  Machines processed:   {len(extraction_stats):>5}")
    print(f"  Total candidates:     {len(all_candidates):>5}")
    print(f"  Extraction dir:       {EXTRACT_DIR}")
    print(f"  V2 output:            {V2_DIR / 'htb.jsonl'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
