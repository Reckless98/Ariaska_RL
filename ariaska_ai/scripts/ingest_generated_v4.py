#!/usr/bin/env python3
"""
Ingest and validate raw Codex-generated JSONL, then run the full v4 build.

Usage:
    # Paste Codex output into generated_raw.jsonl, then:
    python ariaska_ai/scripts/ingest_generated_v4.py generated_raw.jsonl

    # Or pipe directly:
    cat <codex_output.jsonl> | python ariaska_ai/scripts/ingest_generated_v4.py -

    # Then build v4:
    python ariaska_ai/scripts/build_v4_clean_dataset.py \\
        --include-generated ariaska_ai/dataset/v4_generated.jsonl
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"
OUT_PATH = DATASET_DIR / "v4_generated.jsonl"

VALID_TASKS = {
    "tool_output_parse", "retry_or_pivot", "retrieval_reasoning", "postmortem",
    "evidence_check", "next_step", "state_summary", "phase_classification", "command_validate",
}

BAD_IP_RE = re.compile(r"\b192\.168\.0\.1\b|\b123\.123\.123\.123\b")
PUBLIC_IP_RE = re.compile(r"\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b")
THINK_RE = re.compile(r"</?think>", re.I)
PLACEHOLDER_RE = re.compile(r"\{target\}|\{ip\}|<TARGET>|<IP>|CHANGEME|PLACEHOLDER", re.I)
REFUSAL_RE = re.compile(r"\bas an ai\b|i cannot\b|i can't help|i apologize|chatgpt|openai api", re.I)


def is_rfc1918(o1: int, o2: int) -> bool:
    if o1 == 10:
        return True
    if o1 == 172 and 16 <= o2 <= 31:
        return True
    if o1 == 192 and o2 == 168:
        return True
    if o1 == 127:
        return True
    if o1 == 169 and o2 == 254:
        return True
    return False


def has_public_ip(text: str) -> bool:
    for m in PUBLIC_IP_RE.finditer(text):
        o1, o2, o3, o4 = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        if o1 < 10 and o1 != 0:
            continue  # version string
        if o1 == 0 or o1 >= 224:
            continue
        if not is_rfc1918(o1, o2):
            return True
    return False


def validate(raw: str) -> tuple[dict | None, str]:
    """Parse and validate one line. Returns (sample, error_reason)."""
    raw = raw.strip()
    if not raw:
        return None, "empty"

    # Strip any markdown fences Codex may have added
    if raw.startswith("```"):
        raw = raw.lstrip("`").strip()
    if raw.endswith("```"):
        raw = raw.rstrip("`").strip()

    try:
        sample = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, f"json_error:{e}"

    msgs = sample.get("messages", [])
    if not isinstance(msgs, list) or len(msgs) < 2:
        return None, "no_messages"

    roles = [m.get("role") for m in msgs]
    if "assistant" not in roles:
        return None, "no_assistant"
    if "user" not in roles:
        return None, "no_user"

    asst = next(m["content"] for m in msgs if m.get("role") == "assistant")
    if not asst or len(asst.strip()) < 10:
        return None, "empty_assistant"

    stripped = asst.strip()

    # Must be JSON response
    if not stripped.startswith("{"):
        return None, "prose_not_json"

    if not stripped.endswith("}"):
        return None, "truncated_json"

    try:
        json.loads(stripped)
    except json.JSONDecodeError:
        return None, "invalid_assistant_json"

    # Content checks
    full_text = " ".join(m.get("content", "") for m in msgs)
    if THINK_RE.search(asst):
        return None, "think_block_leak"
    if PLACEHOLDER_RE.search(full_text):
        return None, "placeholder_leak"
    if REFUSAL_RE.search(asst.lower()):
        return None, "refusal_leak"
    if BAD_IP_RE.search(full_text):
        return None, "bad_ip"
    if has_public_ip(full_text):
        return None, "public_ip"

    # Normalize task_family
    tf = sample.get("task_family")
    if not tf or tf not in VALID_TASKS:
        return None, f"unknown_task:{tf}"

    return sample, ""


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else "-"

    if src == "-":
        lines = sys.stdin.readlines()
    else:
        with open(src) as f:
            lines = f.readlines()

    print(f"Read {len(lines):,} lines from {src}")

    good: list[dict] = []
    bad: Counter = Counter()

    for line in lines:
        sample, reason = validate(line)
        if sample:
            good.append(sample)
        else:
            bad[reason] += 1

    tf_counts = Counter(s["task_family"] for s in good)

    print(f"\nValidation results:")
    print(f"  PASS: {len(good):,}")
    print(f"  FAIL: {sum(bad.values()):,}")

    print(f"\nPer task:")
    for tf, c in tf_counts.most_common():
        print(f"  {tf:30s}: {c:,}")

    if bad:
        print(f"\nRejection reasons:")
        for reason, c in bad.most_common():
            print(f"  {reason:40s}: {c:,}")

    if not good:
        print("\nNo valid samples — check input format.")
        sys.exit(1)

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for s in good:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(good):,} clean samples to {OUT_PATH}")
    print(f"\nNext step:")
    print(f"  python ariaska_ai/scripts/build_v4_clean_dataset.py \\")
    print(f"    --include-generated {OUT_PATH}")


if __name__ == "__main__":
    main()
