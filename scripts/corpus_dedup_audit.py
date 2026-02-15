#!/usr/bin/env python3
"""
Corpus Deduplication Audit — Ariaska_RL Knowledge Corpus v2
Reads all 18 JSONL files line-by-line (memory-efficient) and detects duplicates.
"""

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "knowledge_candidates_v2"

# Placeholder patterns to strip for near-duplicate detection
PLACEHOLDER_RE = re.compile(
    r"\{(?:target(?:_ip)?|ip|port|host|lhost|lport|rhost|rport|rhosts|"
    r"username|user|password|pass|domain|url|path|file|output|interface|"
    r"wordlist|payload|session|module|exploit|target_port)\}",
    re.IGNORECASE,
)


def normalize_template(s: str) -> str:
    """Lowercase, strip whitespace, remove common variable placeholders."""
    s = s.lower().strip()
    s = PLACEHOLDER_RE.sub("", s)
    # Collapse multiple spaces / tabs
    s = re.sub(r"\s+", " ", s).strip()
    return s


def main():
    jsonl_files = sorted(CORPUS_DIR.glob("*.jsonl"))
    if not jsonl_files:
        print(f"ERROR: No JSONL files found in {CORPUS_DIR}")
        sys.exit(1)

    print(f"Corpus directory : {CORPUS_DIR}")
    print(f"JSONL files found: {len(jsonl_files)}")
    print()

    # --- Tracking structures ---
    file_counts: dict[str, int] = {}
    total_entries = 0

    # Maps: value -> list of (candidate_id, filename, title)
    id_index: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    title_index: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    # For templates: we use the sorted tuple of command_templates list
    template_index: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    # For raw original_commands
    raw_cmd_index: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    # Near-duplicate: normalized original_commands
    near_cmd_index: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
    # Near-duplicate: normalized command_templates (joined)
    near_tpl_index: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)

    # --- Pass 1: Read all files line-by-line ---
    for fpath in jsonl_files:
        fname = fpath.name
        count = 0
        with open(fpath, "r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"  WARN: {fname}:{line_no} — JSON decode error: {e}")
                    continue

                count += 1
                cid = entry.get("candidate_id", f"MISSING_{fname}_{line_no}")
                title = entry.get("title", "")
                execution = entry.get("execution", {})
                cmd_templates = execution.get("command_templates", [])
                raw_pres = entry.get("raw_preservation", {})
                original_cmds = raw_pres.get("original_commands", [])

                info = (cid, fname, title)

                # 1. candidate_id index
                id_index[cid].append(info)

                # 2. title index
                if title:
                    title_index[title].append(info)

                # 3. command_templates index (sorted, joined as key)
                if cmd_templates:
                    tpl_key = "|".join(sorted(cmd_templates))
                    template_index[tpl_key].append(info)

                    # Near-duplicate on template names
                    near_tpl_key = normalize_template(tpl_key)
                    if near_tpl_key:
                        near_tpl_index[near_tpl_key].append(
                            (cid, fname, title, tpl_key)
                        )

                # 4. original_commands index (joined as key)
                if original_cmds:
                    # Filter out empty / whitespace-only commands
                    cmds = [c.strip() for c in original_cmds if isinstance(c, str) and c.strip()]
                    if cmds:
                        raw_key = "|||".join(cmds)
                        raw_cmd_index[raw_key].append(info)

                        # Near-duplicate on commands
                        near_key = "|||".join(normalize_template(c) for c in cmds)
                        if near_key:
                            near_cmd_index[near_key].append(
                                (cid, fname, title, raw_key[:120])
                            )

        file_counts[fname] = count
        total_entries += count

    # --- Analysis ---
    id_dupes = {k: v for k, v in id_index.items() if len(v) > 1}
    title_dupes = {k: v for k, v in title_index.items() if len(v) > 1}
    template_dupes = {k: v for k, v in template_index.items() if len(v) > 1}
    raw_cmd_dupes = {k: v for k, v in raw_cmd_index.items() if len(v) > 1}

    # Near-dupes: only groups where the normalized key differs from the original
    # (i.e., entries that wouldn't be caught by exact match but are near-matches)
    near_tpl_dupes = {k: v for k, v in near_tpl_index.items() if len(v) > 1}
    near_cmd_dupes = {k: v for k, v in near_cmd_index.items() if len(v) > 1}

    # Filter near-dupes to exclude groups already fully caught by exact match
    # A near-dupe group is "novel" if it merges entries from different exact-match groups
    def novel_near_dupes(near_index, exact_index):
        """Return near-dupe groups that combine entries NOT already in the same exact group."""
        novel = {}
        for nkey, entries in near_index.items():
            if len(entries) < 2:
                continue
            # Check if all entries share the same original key
            originals = set(e[3] for e in entries)  # 4th element is original key
            if len(originals) > 1:
                # This near-dupe group merges entries from ≥2 different exact keys
                novel[nkey] = entries
        return novel

    novel_near_tpl = novel_near_dupes(near_tpl_index, template_index)
    novel_near_cmd = novel_near_dupes(near_cmd_index, raw_cmd_index)

    # --- Calculate removable entries ---
    # For each dupe category, count entries that could be removed (keep 1 per group)
    def removable(dupe_dict):
        return sum(len(v) - 1 for v in dupe_dict.values())

    # Collect all candidate_ids that appear as duplicates (union across categories)
    all_dupe_cids = set()
    for groups in [id_dupes, title_dupes, template_dupes, raw_cmd_dupes]:
        for entries in groups.values():
            cids = [e[0] for e in entries]
            # Keep first, mark rest as removable
            for cid in cids[1:]:
                all_dupe_cids.add(cid)
    for groups in [novel_near_tpl, novel_near_cmd]:
        for entries in groups.values():
            cids = [e[0] for e in entries]
            for cid in cids[1:]:
                all_dupe_cids.add(cid)

    # ===================== REPORT =====================
    SEP = "=" * 80

    print(SEP)
    print("  ARIASKA_RL CORPUS DEDUPLICATION AUDIT REPORT")
    print(f"  Corpus: knowledge_candidates_v2 (v2 format)")
    print(SEP)
    print()

    # --- File-level counts ---
    print("─" * 60)
    print("  FILE-LEVEL ENTRY COUNTS")
    print("─" * 60)
    for fname in sorted(file_counts):
        print(f"  {fname:<35s} {file_counts[fname]:>8,d}")
    print(f"  {'TOTAL':<35s} {total_entries:>8,d}")
    print()

    # --- Summary ---
    print("─" * 60)
    print("  DUPLICATE SUMMARY")
    print("─" * 60)
    print(f"  Total entries across all files:            {total_entries:>8,d}")
    print(f"  Exact candidate_id duplicate groups:       {len(id_dupes):>8,d}")
    print(f"    → entries involved:                      {sum(len(v) for v in id_dupes.values()):>8,d}")
    print(f"    → removable (keep 1 per group):          {removable(id_dupes):>8,d}")
    print(f"  Exact title duplicate groups:              {len(title_dupes):>8,d}")
    print(f"    → entries involved:                      {sum(len(v) for v in title_dupes.values()):>8,d}")
    print(f"    → removable (keep 1 per group):          {removable(title_dupes):>8,d}")
    print(f"  Exact command_templates dupe groups:       {len(template_dupes):>8,d}")
    print(f"    → entries involved:                      {sum(len(v) for v in template_dupes.values()):>8,d}")
    print(f"    → removable (keep 1 per group):          {removable(template_dupes):>8,d}")
    print(f"  Exact original_commands dupe groups:       {len(raw_cmd_dupes):>8,d}")
    print(f"    → entries involved:                      {sum(len(v) for v in raw_cmd_dupes.values()):>8,d}")
    print(f"    → removable (keep 1 per group):          {removable(raw_cmd_dupes):>8,d}")
    print(f"  Near-dupe template groups (novel only):    {len(novel_near_tpl):>8,d}")
    print(f"    → entries involved:                      {sum(len(v) for v in novel_near_tpl.values()):>8,d}")
    print(f"    → removable (keep 1 per group):          {removable(novel_near_tpl):>8,d}")
    print(f"  Near-dupe raw command groups (novel only): {len(novel_near_cmd):>8,d}")
    print(f"    → entries involved:                      {sum(len(v) for v in novel_near_cmd.values()):>8,d}")
    print(f"    → removable (keep 1 per group):          {removable(novel_near_cmd):>8,d}")
    print()
    print(f"  TOTAL unique candidate_ids marked removable")
    print(f"  (union across all categories):             {len(all_dupe_cids):>8,d}")
    print()

    # --- Detailed: Exact ID duplicates ---
    if id_dupes:
        print(SEP)
        print(f"  EXACT candidate_id DUPLICATES  ({len(id_dupes)} groups)")
        print(SEP)
        for i, (cid, entries) in enumerate(sorted(id_dupes.items()), 1):
            print(f"\n  Group {i}: candidate_id = {cid}  ({len(entries)} entries)")
            for e_cid, e_file, e_title in entries:
                print(f"    • [{e_file}] {e_title[:90]}")

    # --- Detailed: Exact title duplicates (limit output for large groups) ---
    if title_dupes:
        print()
        print(SEP)
        print(f"  EXACT TITLE DUPLICATES  ({len(title_dupes)} groups)")
        print(SEP)
        # Sort by group size descending
        sorted_title = sorted(title_dupes.items(), key=lambda x: -len(x[1]))
        shown = 0
        MAX_GROUPS = 50
        for title, entries in sorted_title:
            if shown >= MAX_GROUPS:
                remaining = len(sorted_title) - MAX_GROUPS
                print(f"\n  ... and {remaining} more title duplicate groups (truncated)")
                break
            shown += 1
            print(f"\n  Group {shown}: \"{title[:100]}\"  ({len(entries)} entries)")
            for e_cid, e_file, e_title in entries[:10]:
                print(f"    • [{e_file}] id={e_cid}")
            if len(entries) > 10:
                print(f"    ... +{len(entries) - 10} more")

    # --- Detailed: Exact template duplicates ---
    if template_dupes:
        print()
        print(SEP)
        print(f"  EXACT command_templates DUPLICATES  ({len(template_dupes)} groups)")
        print(SEP)
        sorted_tpl = sorted(template_dupes.items(), key=lambda x: -len(x[1]))
        shown = 0
        MAX_GROUPS = 30
        for tpl_key, entries in sorted_tpl:
            if shown >= MAX_GROUPS:
                remaining = len(sorted_tpl) - MAX_GROUPS
                print(f"\n  ... and {remaining} more template duplicate groups (truncated)")
                break
            shown += 1
            print(f"\n  Group {shown}: templates=[{tpl_key}]  ({len(entries)} entries)")
            for e_cid, e_file, e_title in entries[:8]:
                print(f"    • [{e_file}] id={e_cid}  \"{e_title[:80]}\"")
            if len(entries) > 8:
                print(f"    ... +{len(entries) - 8} more")

    # --- Detailed: Exact raw command duplicates ---
    if raw_cmd_dupes:
        print()
        print(SEP)
        print(f"  EXACT original_commands DUPLICATES  ({len(raw_cmd_dupes)} groups)")
        print(SEP)
        sorted_raw = sorted(raw_cmd_dupes.items(), key=lambda x: -len(x[1]))
        shown = 0
        MAX_GROUPS = 30
        for raw_key, entries in sorted_raw:
            if shown >= MAX_GROUPS:
                remaining = len(sorted_raw) - MAX_GROUPS
                print(f"\n  ... and {remaining} more raw command duplicate groups (truncated)")
                break
            shown += 1
            display_key = raw_key[:120] + ("..." if len(raw_key) > 120 else "")
            print(f"\n  Group {shown}: cmd=\"{display_key}\"  ({len(entries)} entries)")
            for e_cid, e_file, e_title in entries[:8]:
                print(f"    • [{e_file}] id={e_cid}  \"{e_title[:80]}\"")
            if len(entries) > 8:
                print(f"    ... +{len(entries) - 8} more")

    # --- Detailed: Novel near-dupe templates ---
    if novel_near_tpl:
        print()
        print(SEP)
        print(f"  NEAR-DUPLICATE command_templates (novel)  ({len(novel_near_tpl)} groups)")
        print(SEP)
        sorted_near_tpl = sorted(novel_near_tpl.items(), key=lambda x: -len(x[1]))
        shown = 0
        MAX_GROUPS = 20
        for nkey, entries in sorted_near_tpl:
            if shown >= MAX_GROUPS:
                remaining = len(sorted_near_tpl) - MAX_GROUPS
                print(f"\n  ... and {remaining} more near-dupe template groups (truncated)")
                break
            shown += 1
            print(f"\n  Group {shown}: normalized=\"{nkey[:100]}\"  ({len(entries)} entries)")
            for e_cid, e_file, e_title, e_orig in entries[:8]:
                print(f"    • [{e_file}] id={e_cid}  \"{e_title[:60]}\" orig=[{e_orig[:60]}]")
            if len(entries) > 8:
                print(f"    ... +{len(entries) - 8} more")

    # --- Detailed: Novel near-dupe raw commands ---
    if novel_near_cmd:
        print()
        print(SEP)
        print(f"  NEAR-DUPLICATE original_commands (novel)  ({len(novel_near_cmd)} groups)")
        print(SEP)
        sorted_near_cmd = sorted(novel_near_cmd.items(), key=lambda x: -len(x[1]))
        shown = 0
        MAX_GROUPS = 20
        for nkey, entries in sorted_near_cmd:
            if shown >= MAX_GROUPS:
                remaining = len(sorted_near_cmd) - MAX_GROUPS
                print(f"\n  ... and {remaining} more near-dupe command groups (truncated)")
                break
            shown += 1
            display_key = nkey[:100] + ("..." if len(nkey) > 100 else "")
            print(f"\n  Group {shown}: normalized=\"{display_key}\"  ({len(entries)} entries)")
            for e_cid, e_file, e_title, e_orig in entries[:8]:
                print(f"    • [{e_file}] id={e_cid}  \"{e_title[:60]}\"")
            if len(entries) > 8:
                print(f"    ... +{len(entries) - 8} more")

    # --- Cross-file duplication heatmap ---
    print()
    print(SEP)
    print("  CROSS-FILE DUPLICATE HEATMAP (title dupes spanning files)")
    print(SEP)
    cross_file_pairs: dict[tuple[str, str], int] = defaultdict(int)
    for title, entries in title_dupes.items():
        files_in_group = set(e[1] for e in entries)
        if len(files_in_group) > 1:
            flist = sorted(files_in_group)
            for i in range(len(flist)):
                for j in range(i + 1, len(flist)):
                    cross_file_pairs[(flist[i], flist[j])] += 1

    if cross_file_pairs:
        sorted_pairs = sorted(cross_file_pairs.items(), key=lambda x: -x[1])
        for (f1, f2), count in sorted_pairs[:30]:
            print(f"  {f1:<30s} ↔ {f2:<30s}  {count:>5d} shared titles")
    else:
        print("  No cross-file title duplicates found.")

    print()
    print(SEP)
    print("  END OF AUDIT REPORT")
    print(SEP)


if __name__ == "__main__":
    main()
