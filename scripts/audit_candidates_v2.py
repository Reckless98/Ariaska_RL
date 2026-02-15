#!/usr/bin/env python3
"""
Audit v2 Knowledge Candidates + Generate corpus_manifest_v2.json.

Phase 10.0 — Knowledge Governance.

Outputs:
  data/corpus_audit_report_v2.json   — quality analysis of v2 candidates
  data/corpus_manifest_v2.json       — rebuilt manifest from actual v2 corpus scan
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
V2_DIR = PROJECT_ROOT / "data" / "knowledge_candidates_v2"
MANIFEST_OUT = PROJECT_ROOT / "data" / "corpus_manifest_v2.json"
AUDIT_OUT = PROJECT_ROOT / "data" / "corpus_audit_report_v2.json"


def main():
    print(f"[AUDIT V2] Scanning {V2_DIR} ...")

    if not V2_DIR.exists():
        print(f"[ERROR] V2 directory not found: {V2_DIR}")
        sys.exit(1)

    jsonl_files = sorted(V2_DIR.glob("*.jsonl"))
    print(f"[AUDIT V2] Found {len(jsonl_files)} JSONL partitions")

    # Counters
    total_candidates = 0
    total_mappable = 0
    total_size = 0
    origin_counts = Counter()
    phase_counts = Counter()
    killchain_counts = Counter()
    service_counts = Counter()
    vuln_family_counts = Counter()
    exploit_arch_counts = Counter()
    tag_counts = Counter()
    template_counts = Counter()
    platform_os_counts = Counter()
    confidence_sum = 0.0
    risk_sum = 0.0
    evidence_coverage_sum = 0.0
    prereq_coverage_sum = 0.0

    partition_stats = {}
    sample_candidates = []
    quality_issues = []
    duplicate_ids = Counter()

    for jsonl_file in jsonl_files:
        partition_name = jsonl_file.stem
        file_size = jsonl_file.stat().st_size
        total_size += file_size
        count = 0
        mappable = 0

        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    quality_issues.append({"file": partition_name, "line": line_num, "issue": "invalid JSON"})
                    continue

                count += 1
                cid = entry.get("candidate_id", "")
                duplicate_ids[cid] += 1

                # Source
                origin = entry.get("source", {}).get("origin", "unknown")
                origin_counts[origin] += 1

                # Taxonomy
                tax = entry.get("taxonomy", {})
                for phase in tax.get("phase_fit", []):
                    phase_counts[phase] += 1
                killchain_counts[tax.get("killchain_step", "unknown")] += 1
                service_counts[tax.get("service_archetype", "unknown")] += 1
                vuln_family_counts[tax.get("vuln_family", "unknown")] += 1
                exploit_arch_counts[tax.get("exploit_archetype", "unknown")] += 1
                for tag in tax.get("tags", []):
                    tag_counts[tag] += 1
                for os_name in tax.get("platform", {}).get("os", []):
                    platform_os_counts[os_name] += 1

                # Execution
                exec_info = entry.get("execution", {})
                for tmpl in exec_info.get("command_templates", []):
                    template_counts[tmpl] += 1

                # Quality
                quality = entry.get("governance", {}).get("quality", {})
                if quality.get("template_mappable"):
                    mappable += 1
                evidence_coverage_sum += quality.get("evidence_coverage", 0.0)
                prereq_coverage_sum += quality.get("prereq_coverage", 0.0)

                # Evidence gate
                eg = entry.get("evidence_gate", {})
                confidence_sum += eg.get("confidence", 0.0)
                risk_sum += eg.get("risk", 0.0)

                # Sample first 3 from each partition
                if count <= 3:
                    sample_candidates.append({
                        "partition": partition_name,
                        "candidate_id": cid,
                        "title": entry.get("title", ""),
                        "templates": exec_info.get("command_templates", [])[:3],
                        "template_mappable": quality.get("template_mappable", False),
                    })

        total_candidates += count
        total_mappable += mappable
        partition_stats[partition_name] = {
            "file": jsonl_file.name,
            "size_bytes": file_size,
            "size_mb": round(file_size / (1024 * 1024), 2),
            "candidates": count,
            "template_mappable": mappable,
            "mappable_pct": round(mappable / max(count, 1) * 100, 1),
        }
        print(f"  {partition_name:25s}  {count:>8,} candidates  {mappable:>8,} mappable  ({mappable/max(count,1)*100:.0f}%)  {file_size/(1024*1024):.1f} MB")

    # Duplicate check
    duplicates = {cid: cnt for cid, cnt in duplicate_ids.items() if cnt > 1}
    if duplicates:
        quality_issues.append({"issue": f"{len(duplicates)} duplicate candidate_ids found", "count": len(duplicates)})

    # Build manifest
    manifest = {
        "manifest_version": "2.0",
        "schema_version": "2.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus_directory": str(V2_DIR),
        "total_candidates": total_candidates,
        "total_template_mappable": total_mappable,
        "total_size_bytes": total_size,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "partitions": partition_stats,
        "origins": dict(origin_counts.most_common()),
        "registry_templates_referenced": len(template_counts),
        "top_templates": dict(template_counts.most_common(30)),
    }

    os.makedirs(MANIFEST_OUT.parent, exist_ok=True)
    MANIFEST_OUT.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\n[MANIFEST V2] Written to {MANIFEST_OUT}")

    # Build audit report
    audit_report = {
        "audit_version": "2.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus_directory": str(V2_DIR),
        "summary": {
            "total_candidates": total_candidates,
            "total_template_mappable": total_mappable,
            "mappable_pct": round(total_mappable / max(total_candidates, 1) * 100, 1),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "partitions": len(partition_stats),
            "avg_confidence": round(confidence_sum / max(total_candidates, 1), 3),
            "avg_risk": round(risk_sum / max(total_candidates, 1), 3),
            "avg_evidence_coverage": round(evidence_coverage_sum / max(total_candidates, 1), 3),
            "avg_prereq_coverage": round(prereq_coverage_sum / max(total_candidates, 1), 3),
            "duplicate_ids": len(duplicates),
            "quality_issues": len(quality_issues),
        },
        "distributions": {
            "origins": dict(origin_counts.most_common()),
            "phases": dict(phase_counts.most_common()),
            "killchain_steps": dict(killchain_counts.most_common()),
            "service_archetypes": dict(service_counts.most_common(20)),
            "vuln_families": dict(vuln_family_counts.most_common(20)),
            "exploit_archetypes": dict(exploit_arch_counts.most_common(20)),
            "tags": dict(tag_counts.most_common(30)),
            "platform_os": dict(platform_os_counts.most_common()),
        },
        "template_coverage": {
            "unique_templates_referenced": len(template_counts),
            "top_30_templates": dict(template_counts.most_common(30)),
        },
        "partitions": partition_stats,
        "sample_candidates": sample_candidates[:20],
        "quality_issues": quality_issues[:50],
    }

    AUDIT_OUT.write_text(json.dumps(audit_report, indent=2, ensure_ascii=False))
    print(f"[AUDIT V2] Report written to {AUDIT_OUT}")

    print(f"\n{'='*60}")
    print(f"[AUDIT V2] SUMMARY")
    print(f"  Total candidates:     {total_candidates:>10,}")
    print(f"  Template mappable:    {total_mappable:>10,}  ({total_mappable/max(total_candidates,1)*100:.1f}%)")
    print(f"  Avg confidence:       {confidence_sum/max(total_candidates,1):>10.3f}")
    print(f"  Avg evidence cov:     {evidence_coverage_sum/max(total_candidates,1):>10.3f}")
    print(f"  Unique templates:     {len(template_counts):>10}")
    print(f"  Quality issues:       {len(quality_issues):>10}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
