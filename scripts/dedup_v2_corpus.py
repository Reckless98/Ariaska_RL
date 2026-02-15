#!/usr/bin/env python3
"""
Phase 10.1 S3: Deduplicate v2 knowledge corpus.

Strategy (conservative, zero information loss):
  1. Remove exact candidate_id duplicates — keep FIRST occurrence per file
  2. Report stats before/after

Author: Filip Volf / Ariaska System
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict

CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "knowledge_candidates_v2"


def dedup_file(filepath: Path, dry_run: bool = True) -> dict:
    """Deduplicate a single JSONL file by candidate_id. Keep first occurrence."""
    seen_ids = set()
    kept = []
    removed = 0
    total = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                # Keep malformed lines as-is
                kept.append(line)
                continue

            cid = entry.get("candidate_id", "")
            if cid and cid in seen_ids:
                removed += 1
            else:
                if cid:
                    seen_ids.add(cid)
                kept.append(line)

    if not dry_run and removed > 0:
        with open(filepath, "w", encoding="utf-8") as f:
            for line in kept:
                f.write(line + "\n")

    return {
        "file": filepath.name,
        "total_before": total,
        "removed": removed,
        "total_after": total - removed,
        "unique_ids": len(seen_ids),
    }


def dedup_cross_file(corpus_dir: Path, dry_run: bool = True) -> dict:
    """
    After per-file dedup, check for cross-file ID duplicates.
    Keep entry in the file where it appears first (alphabetical file order).
    """
    global_ids = {}  # id -> first_file
    cross_dupes = defaultdict(list)  # id -> list of (file, line_idx)

    files = sorted(corpus_dir.glob("*.jsonl"))
    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cid = entry.get("candidate_id", "")
                if not cid:
                    continue
                if cid in global_ids:
                    cross_dupes[cid].append((filepath.name, line_no))
                else:
                    global_ids[cid] = filepath.name

    # Remove cross-file duplicates (keep in first file)
    files_to_rewrite = defaultdict(set)  # file -> set of line_nos to remove
    for cid, locations in cross_dupes.items():
        for fname, line_no in locations:
            files_to_rewrite[fname].add(line_no)

    cross_removed = 0
    if not dry_run and files_to_rewrite:
        for fname, remove_lines in files_to_rewrite.items():
            filepath = corpus_dir / fname
            kept = []
            with open(filepath, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    if line_no not in remove_lines:
                        kept.append(line.rstrip("\n"))
                    else:
                        cross_removed += 1
            with open(filepath, "w", encoding="utf-8") as f:
                for line in kept:
                    f.write(line + "\n")
    else:
        cross_removed = sum(len(v) for v in files_to_rewrite.values())

    return {
        "cross_file_dupe_ids": len(cross_dupes),
        "cross_file_entries_removed": cross_removed,
        "files_affected": list(files_to_rewrite.keys()),
    }


def main():
    dry_run = "--apply" not in sys.argv
    if dry_run:
        print("═" * 60)
        print("  DRY RUN — pass --apply to actually write changes")
        print("═" * 60)
    else:
        print("═" * 60)
        print("  APPLYING DEDUPLICATION")
        print("═" * 60)

    print()

    # Phase 1: Per-file dedup
    print("Phase 1: Per-file candidate_id deduplication")
    print("-" * 50)
    files = sorted(CORPUS_DIR.glob("*.jsonl"))
    total_before = 0
    total_removed = 0
    results = []

    for filepath in files:
        result = dedup_file(filepath, dry_run=dry_run)
        results.append(result)
        total_before += result["total_before"]
        total_removed += result["removed"]
        if result["removed"] > 0:
            print(f"  {result['file']:40s}  {result['total_before']:>6d} → {result['total_after']:>6d}  "
                  f"(-{result['removed']})")
        else:
            print(f"  {result['file']:40s}  {result['total_before']:>6d}  (clean)")

    print()
    print(f"  Per-file total: {total_before} → {total_before - total_removed}  "
          f"(-{total_removed} removed)")

    # Phase 2: Cross-file dedup
    print()
    print("Phase 2: Cross-file candidate_id deduplication")
    print("-" * 50)
    cross_result = dedup_cross_file(CORPUS_DIR, dry_run=dry_run)
    print(f"  Cross-file duplicate IDs: {cross_result['cross_file_dupe_ids']}")
    print(f"  Cross-file entries removed: {cross_result['cross_file_entries_removed']}")
    if cross_result['files_affected']:
        print(f"  Files affected: {', '.join(cross_result['files_affected'])}")

    # Summary
    grand_removed = total_removed + cross_result['cross_file_entries_removed']
    grand_after = total_before - grand_removed
    print()
    print("═" * 60)
    print(f"  SUMMARY: {total_before} → {grand_after}  "
          f"(-{grand_removed} entries, {grand_removed/total_before*100:.1f}% reduction)")
    print("═" * 60)

    if dry_run:
        print()
        print("  Run with --apply to execute deduplication.")


if __name__ == "__main__":
    main()
