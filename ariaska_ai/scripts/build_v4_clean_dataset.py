#!/usr/bin/env python3
"""
Build v4 Clean Dataset — Dedup, decontaminate, merge V2+V3 into the
definitive training corpus for Qwen3-8B fine-tuning.

Steps:
  1. Load V3 per-task files (primary, 34K unique clean samples)
  2. Load V2 train.jsonl, dedup against V3, filter contamination
  3. Strip bad IPs (non-RFC1918), placeholder leaks, refusals
  4. Normalize all samples to consistent ChatML format with task_family
  5. Merge generated gap-fill samples (from codex_generate_v4.py)
  6. Stratified split: 85% train / 7.5% val / 7.5% holdout
  7. Write v4/ output with stats

Usage:
    python build_v4_clean_dataset.py [--include-generated ariaska_ai/dataset/v4_generated.jsonl]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"
V2_DIR = DATASET_DIR / "v2"
V3_DIR = DATASET_DIR / "v3"
V4_DIR = DATASET_DIR / "v4"

# IPs that are known hallucinations
BAD_IP_PATTERNS = re.compile(
    r"\b(?:"
    r"192\.168\.0\.1"        # router/hallucinated
    r"|123\.123\.123\.123"   # placeholder
    r")\b"
)

# RFC1918 check
def is_rfc1918_or_safe(ip: str) -> bool:
    """Return True if IP is RFC1918, loopback, or link-local."""
    parts = ip.split(".")
    if len(parts) != 4:
        return False
    try:
        o = [int(p) for p in parts]
    except ValueError:
        return False
    if o[0] == 10:
        return True
    if o[0] == 172 and 16 <= o[1] <= 31:
        return True
    if o[0] == 192 and o[1] == 168 and not (o[2] == 0 and o[3] == 1):
        return True
    if o[0] == 127:
        return True
    if o[0] == 169 and o[1] == 254:
        return True
    return False


def has_public_ip(text: str) -> bool:
    """Check if text contains non-RFC1918 IPs (excluding version strings like 3.2.8.1)."""
    ips = re.findall(r"\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b", text)
    for ip in ips:
        parts = ip.split(".")
        o1 = int(parts[0])
        # Skip things that look like version numbers (single-digit first octet)
        if o1 < 10 and o1 != 0:
            continue
        if o1 == 0 or o1 >= 224:
            continue
        if not is_rfc1918_or_safe(ip):
            if BAD_IP_PATTERNS.search(ip):
                return True
            # Check if it's clearly not a version string
            if o1 >= 10:
                return True
    return False


# Placeholder patterns — case-insensitive angle-bracket and curly-brace forms.
# CHANGEME only flagged when standalone (not part of a credential like tomcat:changeme).
_ANGLE_PH_RE = re.compile(r"<target>|<ip>|\{target\}|\{ip\}", re.IGNORECASE)
_CHANGEME_STANDALONE_RE = re.compile(r"(?<![:\w])changeme(?!\w)", re.IGNORECASE)


def is_contaminated(sample: dict) -> tuple[bool, str]:
    """Return (contaminated, reason) for a sample."""
    msgs = sample.get("messages", [])
    if not msgs:
        return True, "no_messages"

    asst = [m for m in msgs if m.get("role") == "assistant"]
    if not asst:
        return True, "no_assistant"

    content = asst[0].get("content", "")
    if not content or len(content) < 10:
        return True, "empty_or_short"

    # Check JSON validity for JSON-expected responses
    stripped = content.strip()
    if stripped.startswith("{") and not stripped.endswith("}"):
        return True, "truncated_json"

    # Prose responses (V1 style) — reject
    if not stripped.startswith("{"):
        return True, "prose_not_json"

    # Validate JSON parses
    try:
        json.loads(stripped)
    except json.JSONDecodeError:
        return True, "invalid_json"

    # Placeholder leaks — check all non-system roles (system prompts legitimately
    # use schema markers like {target}; user/assistant content must not contain them)
    # Use case-insensitive matching for angle-bracket placeholders.
    # CHANGEME is excluded when it forms part of a credential (e.g. tomcat:changeme).
    for m in msgs:
        if m.get("role") == "system":
            continue
        mtext = m.get("content", "")
        if _ANGLE_PH_RE.search(mtext):
            return True, "placeholder_angle"
        if _CHANGEME_STANDALONE_RE.search(mtext):
            return True, "placeholder_CHANGEME"

    # Refusal/identity leaks
    lower = content.lower()
    for term in ["as an ai", "i cannot", "i can't help", "i apologize", "i do not have",
                 "chatgpt", "openai api"]:
        if term in lower:
            return True, f"refusal_{term}"

    # <think> block leak
    if "<think>" in content or "</think>" in content:
        return True, "think_block_leak"

    # Public IP contamination
    full_text = " ".join(m.get("content", "") for m in msgs)
    if has_public_ip(full_text):
        return True, "public_ip"

    return False, ""


def content_hash(sample: dict) -> str:
    """Hash based on user + assistant content for dedup."""
    msgs = sample.get("messages", [])
    user = "".join(m["content"] for m in msgs if m.get("role") == "user")
    asst = "".join(m["content"] for m in msgs if m.get("role") == "assistant")
    return hashlib.md5((user + asst).encode()).hexdigest()


def asst_hash(sample: dict) -> str:
    """Hash based on assistant content only — for detecting oversampled responses."""
    msgs = sample.get("messages", [])
    asst = "".join(m["content"] for m in msgs if m.get("role") == "assistant")
    return hashlib.md5(asst.encode()).hexdigest()


# Patterns indicating template/generic reasoning in retry_or_pivot responses.
_TEMPLATE_REASONING_RE = re.compile(
    r"^(Escalating|Pivoting|Retrying) (from|to) \w+ (to|from) \w+",
    re.IGNORECASE,
)
_GENERIC_PHRASE_RE = re.compile(
    r"trying (a |different |another |new )?approach|^generic$|^default$",
    re.IGNORECASE,
)


def is_low_quality(sample: dict) -> tuple[bool, str]:
    """Return (low_quality, reason) for task-family-specific quality checks."""
    tf = sample.get("task_family", "")
    if tf != "retry_or_pivot":
        return False, ""

    msgs = sample.get("messages", [])
    asst_content = next((m.get("content", "") for m in msgs if m.get("role") == "assistant"), "")
    try:
        parsed = json.loads(asst_content.strip())
    except json.JSONDecodeError:
        return False, ""  # already caught by is_contaminated

    reasoning = parsed.get("reasoning", "")
    if _TEMPLATE_REASONING_RE.match(reasoning):
        return True, "template_reasoning"
    if _GENERIC_PHRASE_RE.search(reasoning):
        return True, "generic_phrase"
    if len(reasoning) < 30:
        return True, "reasoning_too_short"
    return False, ""


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file, skipping malformed lines."""
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return samples


def normalize_sample(sample: dict) -> dict:
    """Normalize to consistent format with task_family at top level."""
    tf = sample.get("task_family")
    if not tf:
        md = sample.get("metadata", {})
        tf = md.get("task_family", "unknown")

    msgs = sample.get("messages", [])
    normalized = {"messages": msgs, "task_family": tf}
    if "metadata" in sample:
        normalized["metadata"] = sample["metadata"]
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v4 clean dataset")
    parser.add_argument("--include-generated", type=str, default=None,
                        help="Path to generated gap-fill JSONL from Codex")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    V4_DIR.mkdir(parents=True, exist_ok=True)

    seen_hashes: set[str] = set()
    clean_samples: list[dict] = []
    reject_reasons: Counter = Counter()
    source_counts: Counter = Counter()

    # --- Phase 1: Load V3 per-task files (primary source) ---
    print("Phase 1: Loading V3 per-task files...")
    v3_tasks = [
        "command_validate", "evidence_check", "next_step",
        "phase_classification", "postmortem", "retrieval_reasoning",
        "retry_or_pivot", "state_summary", "tool_output_parse",
    ]
    for task in v3_tasks:
        path = V3_DIR / f"{task}.jsonl"
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping")
            continue
        samples = load_jsonl(path)
        added = 0
        for s in samples:
            contaminated, reason = is_contaminated(s)
            if contaminated:
                reject_reasons[f"v3_{task}:{reason}"] += 1
                continue
            low_q, lq_reason = is_low_quality(s)
            if low_q:
                reject_reasons[f"v3_{task}:low_quality_{lq_reason}"] += 1
                continue
            h = content_hash(s)
            if h in seen_hashes:
                reject_reasons[f"v3_{task}:duplicate"] += 1
                continue
            seen_hashes.add(h)
            norm = normalize_sample(s)
            norm["source"] = "v3"
            clean_samples.append(norm)
            added += 1
        source_counts[f"v3/{task}"] = added
        print(f"  {task}: {len(samples)} -> {added} clean")

    # --- Phase 2: Load V2 train.jsonl (salvage non-overlapping) ---
    print("\nPhase 2: Loading V2 train.jsonl (salvage)...")
    v2_path = V2_DIR / "train.jsonl"
    if v2_path.exists():
        v2_samples = load_jsonl(v2_path)
        added = 0
        for s in v2_samples:
            contaminated, reason = is_contaminated(s)
            if contaminated:
                reject_reasons[f"v2:{reason}"] += 1
                continue
            low_q, lq_reason = is_low_quality(s)
            if low_q:
                reject_reasons[f"v2:low_quality_{lq_reason}"] += 1
                continue
            h = content_hash(s)
            if h in seen_hashes:
                reject_reasons["v2:duplicate_with_v3"] += 1
                continue
            seen_hashes.add(h)
            norm = normalize_sample(s)
            norm["source"] = "v2"
            clean_samples.append(norm)
            added += 1
        source_counts["v2/train"] = added
        print(f"  V2: {len(v2_samples)} -> {added} clean (non-overlapping)")

    # --- Phase 3: Load generated gap-fill samples ---
    if args.include_generated:
        gen_path = Path(args.include_generated)
        if gen_path.exists():
            print(f"\nPhase 3: Loading generated samples from {gen_path}...")
            gen_samples = load_jsonl(gen_path)
            added = 0
            for s in gen_samples:
                contaminated, reason = is_contaminated(s)
                if contaminated:
                    reject_reasons[f"generated:{reason}"] += 1
                    continue
                h = content_hash(s)
                if h in seen_hashes:
                    reject_reasons["generated:duplicate"] += 1
                    continue
                seen_hashes.add(h)
                norm = normalize_sample(s)
                norm["source"] = "generated"
                clean_samples.append(norm)
                added += 1
            source_counts["generated"] = added
            print(f"  Generated: {len(gen_samples)} -> {added} clean")
        else:
            print(f"\nPhase 3: Generated file {gen_path} not found, skipping")

    # --- Phase 4: DPO pairs from V3 ---
    dpo_path = V3_DIR / "dpo_pairs.jsonl"
    dpo_pairs: list[dict] = []
    if dpo_path.exists():
        print("\nPhase 4: Loading V3 DPO pairs...")
        dpo_raw = load_jsonl(dpo_path)
        for d in dpo_raw:
            if "prompt" in d and "chosen" in d and "rejected" in d:
                dpo_pairs.append(d)
        print(f"  DPO pairs: {len(dpo_pairs)}")

    # --- Phase 4b: Cap oversampled assistant responses (max 5 per unique response) ---
    # The v3/v2 RL trace data has 566 unique responses each appearing >5x, covering
    # 35% of the dataset. Cap at 5 to prevent the model from memorising them.
    print(f"\nPhase 4b: Deduplicating oversampled assistant responses (max 5)...")
    asst_resp_count: dict[str, int] = {}
    deduped_samples: list[dict] = []
    for s in clean_samples:
        ah = asst_hash(s)
        cnt = asst_resp_count.get(ah, 0)
        if cnt < 5:
            asst_resp_count[ah] = cnt + 1
            deduped_samples.append(s)
        else:
            reject_reasons["asst_oversampled"] += 1
    removed_oversampled = len(clean_samples) - len(deduped_samples)
    print(f"  Removed {removed_oversampled:,} oversampled responses → {len(deduped_samples):,} clean")
    clean_samples = deduped_samples

    # --- Phase 5: Stratified split ---
    print(f"\nPhase 5: Splitting {len(clean_samples)} clean samples...")
    random.shuffle(clean_samples)

    # Stratify by task_family
    by_task: dict[str, list[dict]] = {}
    for s in clean_samples:
        tf = s.get("task_family", "unknown")
        by_task.setdefault(tf, []).append(s)

    train, val, holdout = [], [], []
    for tf, samples in by_task.items():
        random.shuffle(samples)
        n = len(samples)
        n_val = max(1, int(n * 0.075))
        n_holdout = max(1, int(n * 0.075))
        holdout.extend(samples[:n_holdout])
        val.extend(samples[n_holdout : n_holdout + n_val])
        train.extend(samples[n_holdout + n_val :])

    # --- Phase 6: Write outputs ---
    print(f"\nPhase 6: Writing v4 dataset...")

    def write_jsonl(path: Path, data: list[dict]) -> None:
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    write_jsonl(V4_DIR / "train.jsonl", train)
    write_jsonl(V4_DIR / "val.jsonl", val)
    write_jsonl(V4_DIR / "holdout.jsonl", holdout)
    if dpo_pairs:
        write_jsonl(V4_DIR / "dpo_pairs.jsonl", dpo_pairs)

    # Also write per-task files for reference
    for tf, samples in by_task.items():
        write_jsonl(V4_DIR / f"{tf}.jsonl", samples)

    # --- Stats ---
    tf_counts = Counter(s["task_family"] for s in clean_samples)
    stats = {
        "version": "4.0",
        "total_samples": len(clean_samples),
        "train": len(train),
        "val": len(val),
        "holdout": len(holdout),
        "dpo_pairs": len(dpo_pairs),
        "sources": dict(source_counts),
        "per_task": {tf: c for tf, c in tf_counts.most_common()},
        "rejection_summary": {k: v for k, v in reject_reasons.most_common(30)},
    }
    with open(V4_DIR / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*60}")
    print(f"V4 DATASET BUILT SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"  Total clean:  {len(clean_samples):,}")
    print(f"  Train:        {len(train):,}")
    print(f"  Val:          {len(val):,}")
    print(f"  Holdout:      {len(holdout):,}")
    print(f"  DPO pairs:    {len(dpo_pairs):,}")
    print(f"\nPer task:")
    for tf, c in tf_counts.most_common():
        print(f"  {tf:30s}: {c:,}")
    print(f"\nSources:")
    for src, c in source_counts.most_common():
        print(f"  {src:30s}: {c:,}")
    print(f"\nTop rejections:")
    for reason, c in reject_reasons.most_common(15):
        print(f"  {reason:45s}: {c:,}")
    print(f"\nOutput: {V4_DIR}")


if __name__ == "__main__":
    main()
