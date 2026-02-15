#!/usr/bin/env python3
"""
Build Knowledge Indices from v2 Corpus.

Phase 10.0 — Knowledge Governance.

Builds multi-dimensional indices for fast lookup:
  - ports, services, cves, techniques, phases, tags, killchain, templates, vuln_families

Outputs: data/knowledge_indices/ (JSON index files)
"""

import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
V2_DIR = PROJECT_ROOT / "data" / "knowledge_candidates_v2"
INDEX_DIR = PROJECT_ROOT / "data" / "knowledge_indices"
KB_DIR = PROJECT_ROOT / "data" / "knowledge_base"


def load_all_v2_candidates() -> list:
    """Load all v2 candidates from JSONL partitions."""
    candidates = []
    if not V2_DIR.exists():
        print(f"[WARN] V2 directory not found: {V2_DIR}")
        return candidates

    for jsonl_file in sorted(V2_DIR.glob("*.jsonl")):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        candidates.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return candidates


def build_indices(candidates: list) -> dict:
    """Build multi-dimensional indices."""
    indices = {
        "by_port": defaultdict(list),
        "by_service": defaultdict(list),
        "by_cve": defaultdict(list),
        "by_phase": defaultdict(list),
        "by_killchain": defaultdict(list),
        "by_tag": defaultdict(list),
        "by_vuln_family": defaultdict(list),
        "by_exploit_archetype": defaultdict(list),
        "by_template": defaultdict(list),
        "by_origin": defaultdict(list),
        "by_platform": defaultdict(list),
    }

    for i, c in enumerate(candidates):
        cid = c.get("candidate_id", str(i))
        ref = {"candidate_id": cid, "title": c.get("title", ""), "idx": i}

        tax = c.get("taxonomy", {})
        eg = c.get("evidence_gate", {})
        exe = c.get("execution", {})
        refs = c.get("references", {})
        src = c.get("source", {})

        # Port index (from evidence requirements)
        for req in eg.get("evidence_requirements", []):
            if req.startswith("port_") and req.endswith("_open"):
                port_str = req.replace("port_", "").replace("_open", "")
                indices["by_port"][port_str].append(ref)

        # Service index
        svc = tax.get("service_archetype", "")
        if svc and svc != "general":
            indices["by_service"][svc].append(ref)

        # CVE index
        for cve in refs.get("cves", []):
            if cve:
                indices["by_cve"][cve.upper()].append(ref)

        # Phase index
        for phase in tax.get("phase_fit", []):
            indices["by_phase"][phase].append(ref)

        # Killchain index
        kc = tax.get("killchain_step", "")
        if kc:
            indices["by_killchain"][kc].append(ref)

        # Tag index
        for tag in tax.get("tags", []):
            if tag:
                indices["by_tag"][tag.lower()].append(ref)

        # Vuln family index
        vf = tax.get("vuln_family", "")
        if vf and vf != "unknown":
            indices["by_vuln_family"][vf].append(ref)

        # Exploit archetype index
        ea = tax.get("exploit_archetype", "")
        if ea and ea != "manual":
            indices["by_exploit_archetype"][ea].append(ref)

        # Template index
        for tmpl in exe.get("command_templates", []):
            if tmpl:
                indices["by_template"][tmpl].append(ref)

        # Origin index
        origin = src.get("origin", "")
        if origin:
            indices["by_origin"][origin].append(ref)

        # Platform index
        for os_name in tax.get("platform", {}).get("os", []):
            indices["by_platform"][os_name].append(ref)

    return indices


def main():
    print(f"[INDEX] Building knowledge indices from v2 corpus...")

    candidates = load_all_v2_candidates()
    print(f"[INDEX] Loaded {len(candidates):,} candidates from {V2_DIR}")

    if not candidates:
        # Fall back to v1 corpus
        print(f"[INDEX] No v2 candidates found, falling back to v1 corpus...")
        for jf in sorted(KB_DIR.glob("*.json")):
            if jf.name == "manifest.json":
                continue
            try:
                data = json.loads(jf.read_text())
                if isinstance(data, list):
                    candidates.extend(data)
            except Exception:
                pass
        print(f"[INDEX] Loaded {len(candidates):,} v1 entries")

    indices = build_indices(candidates)

    # Save indices
    os.makedirs(INDEX_DIR, exist_ok=True)
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_candidates": len(candidates),
        "index_dimensions": {},
    }

    for index_name, index_data in indices.items():
        # Convert defaultdict to regular dict for JSON
        output = {k: v for k, v in index_data.items()}
        output_path = INDEX_DIR / f"{index_name}.json"
        output_path.write_text(json.dumps(output, ensure_ascii=False))
        meta["index_dimensions"][index_name] = len(output)
        print(f"  {index_name:25s}  {len(output):>6,} keys")

    # Save metadata
    (INDEX_DIR / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    # Verification: sample queries
    print(f"\n[INDEX] Verification queries:")
    for port in ["22", "80", "445", "3306"]:
        count = len(indices["by_port"].get(port, []))
        print(f"  port {port}: {count} candidates")
    for svc in ["smb", "ssh", "http", "database"]:
        count = len(indices["by_service"].get(svc, []))
        print(f"  service {svc}: {count} candidates")
    for phase in ["RECON", "EXPLOITATION", "PRIVILEGE_ESCALATION"]:
        count = len(indices["by_phase"].get(phase, []))
        print(f"  phase {phase}: {count} candidates")

    print(f"\n[INDEX] Indices written to {INDEX_DIR}")


if __name__ == "__main__":
    main()
