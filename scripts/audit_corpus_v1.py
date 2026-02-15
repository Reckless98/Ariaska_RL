#!/usr/bin/env python3
"""
Audit v1 Knowledge Corpus — Scan actual JSON files and produce reality-based report.

Phase 10.0 — Knowledge Governance.
Outputs: data/corpus_audit_report_v1.json

This script DOES NOT trust the manifest. It scans actual files on disk.
"""

import json
import os
import sys
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KB_DIR = PROJECT_ROOT / "data" / "knowledge_base"
OUTPUT = PROJECT_ROOT / "data" / "corpus_audit_report_v1.json"


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def analyze_file(path: Path) -> dict:
    """Analyze a single knowledge base JSON file."""
    result = {
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256_prefix": hash_file(path),
        "entry_count": 0,
        "is_array": False,
        "sample_keys": [],
        "categories": [],
        "sources": [],
        "has_commands": False,
        "has_cves": False,
        "has_descriptions": True,
        "empty_descriptions": 0,
        "errors": [],
    }

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        result["errors"].append(f"JSON parse error: {e}")
        return result

    if isinstance(data, list):
        result["is_array"] = True
        result["entry_count"] = len(data)
        if data:
            result["sample_keys"] = list(data[0].keys()) if isinstance(data[0], dict) else []

            categories = Counter()
            sources = Counter()
            cmds = 0
            cves = 0
            empty_desc = 0

            for entry in data:
                if not isinstance(entry, dict):
                    continue
                cat = entry.get("category", entry.get("tactic", entry.get("phase", "")))
                if cat:
                    categories[str(cat)] += 1
                src = entry.get("source", "")
                if src:
                    src_prefix = str(src).split("/")[0] if "/" in str(src) else str(src)
                    sources[src_prefix] += 1

                # Check for commands
                if entry.get("commands") or entry.get("command") or entry.get("exploit_commands") or entry.get("enumeration_commands"):
                    cmds += 1
                # Check for CVEs
                cve_val = entry.get("cve_id", entry.get("cve", ""))
                if cve_val and str(cve_val).startswith("CVE"):
                    cves += 1
                elif isinstance(cve_val, list):
                    cves += len([c for c in cve_val if str(c).startswith("CVE")])
                # Check descriptions
                desc = entry.get("description", "")
                if not desc or len(str(desc).strip()) < 5:
                    empty_desc += 1

            result["categories"] = dict(categories.most_common(20))
            result["sources"] = dict(sources.most_common(20))
            result["has_commands"] = cmds > 0
            result["command_entries"] = cmds
            result["has_cves"] = cves > 0
            result["cve_count"] = cves
            result["empty_descriptions"] = empty_desc
    elif isinstance(data, dict):
        result["is_array"] = False
        result["entry_count"] = len(data)
        result["sample_keys"] = list(data.keys())[:20]
    else:
        result["errors"].append(f"Unexpected root type: {type(data).__name__}")

    return result


def main():
    print(f"[AUDIT V1] Scanning {KB_DIR} ...")

    if not KB_DIR.exists():
        print(f"[ERROR] Knowledge base directory not found: {KB_DIR}")
        sys.exit(1)

    json_files = sorted(KB_DIR.glob("*.json"))
    print(f"[AUDIT V1] Found {len(json_files)} JSON files")

    file_reports = []
    total_entries = 0
    total_size = 0
    total_with_commands = 0
    total_cves = 0

    for f in json_files:
        if f.name == "manifest.json":
            continue  # Skip stale manifest
        report = analyze_file(f)
        file_reports.append(report)
        total_entries += report["entry_count"]
        total_size += report["size_bytes"]
        if report["has_commands"]:
            total_with_commands += report.get("command_entries", 0)
        total_cves += report.get("cve_count", 0)
        print(f"  {report['filename']:30s}  {report['entry_count']:>8,} entries  {report['size_bytes']:>12,} bytes")

    # Build summary
    audit_report = {
        "audit_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus_directory": str(KB_DIR),
        "summary": {
            "total_files": len(file_reports),
            "total_entries": total_entries,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "entries_with_commands": total_with_commands,
            "entries_with_cves": total_cves,
            "manifest_stale": True,
            "manifest_stale_reason": "manifest.json claims 0 entries but actual count is " + str(total_entries),
        },
        "files": file_reports,
        "schema_inventory": {
            f["filename"]: f["sample_keys"] for f in file_reports
        },
        "recommendations": [
            "Rebuild manifest from actual file scan (do not trust manifest.json)",
            "Normalize entry schemas to KnowledgeCandidate v2 format",
            "Map raw commands to COMMAND_REGISTRY templates",
            "Add evidence gates based on service/port/tool requirements",
            "Extract CVE references into structured references field",
        ],
    }

    os.makedirs(OUTPUT.parent, exist_ok=True)
    OUTPUT.write_text(json.dumps(audit_report, indent=2, ensure_ascii=False))
    print(f"\n[AUDIT V1] Report written to {OUTPUT}")
    print(f"[AUDIT V1] Total: {total_entries:,} entries across {len(file_reports)} files ({total_size / (1024*1024):.1f} MB)")

    return audit_report


if __name__ == "__main__":
    main()
