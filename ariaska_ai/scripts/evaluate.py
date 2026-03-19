#!/usr/bin/env python3
"""Evaluate fine-tuned Ariaska model against baseline on V3 holdout set.

Covers all 9 task families:
  command_validate, evidence_check, next_step, phase_classification,
  postmortem, retrieval_reasoning, retry_or_pivot, state_summary,
  tool_output_parse

Metrics per family: JSON parse rate, phase accuracy, command quality,
reasoning quality, latency.  Aggregate + per-family comparison.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ariaska.eval")

# ── Paths ────────────────────────────────────────────────────────────────────
EVAL_DIR = Path(__file__).resolve().parents[1] / "dataset" / "v3"
HOLDOUT_FILE = EVAL_DIR / "holdout.jsonl"
# Fallback to old eval.jsonl if v3 holdout doesn't exist
LEGACY_EVAL_FILE = Path(__file__).resolve().parents[1] / "dataset" / "eval.jsonl"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "outputs"

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
BASELINE_MODEL = os.environ.get("ARIASKA_BASELINE_MODEL", "jaahas/qwen3.5-uncensored:4b")
FINETUNED_MODEL = os.environ.get("ARIASKA_FINETUNED_MODEL", "ariaska-cybersec")

MAX_EVAL_SAMPLES = int(os.environ.get("ARIASKA_EVAL_MAX", "500"))
SAMPLES_PER_FAMILY = int(os.environ.get("ARIASKA_EVAL_PER_FAMILY", "60"))

# ── Task family definitions ──────────────────────────────────────────────────
TASK_FAMILIES = [
    "command_validate", "evidence_check", "next_step", "phase_classification",
    "postmortem", "retrieval_reasoning", "retry_or_pivot", "state_summary",
    "tool_output_parse",
]

JSON_FAMILIES = {"command_validate", "evidence_check", "next_step",
                 "state_summary", "tool_output_parse"}
PHASE_FAMILIES = {"phase_classification"}
COMMAND_FAMILIES = {"command_validate", "next_step", "retry_or_pivot"}
REASONING_FAMILIES = {"evidence_check", "retrieval_reasoning",
                      "postmortem", "retry_or_pivot"}

ATTACK_PHASES = [
    "RECON", "ENUMERATION", "EXPLOITATION", "PRIVILEGE_ESCALATION",
    "LATERAL_MOVEMENT", "POST_EXPLOITATION", "EXFILTRATION", "CLOSEOUT",
]


# ── Ollama client ────────────────────────────────────────────────────────────

def ollama_generate(model: str, messages: list[dict], timeout: int = 60) -> tuple[str, float]:
    """Call Ollama chat API, return (response_text, latency_seconds)."""
    t0 = time.time()
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 768},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        latency = time.time() - t0
        return data.get("message", {}).get("content", ""), latency
    except Exception as e:
        return f"ERROR: {e}", time.time() - t0


# ── Scorers ──────────────────────────────────────────────────────────────────

def try_parse_json(text: str) -> dict | list | None:
    """Try to parse JSON from text, including embedded JSON."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Markdown code block
    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # First { or [
    for ch in ("{", "["):
        idx = text.find(ch)
        if idx >= 0:
            try:
                return json.loads(text[idx:])
            except json.JSONDecodeError:
                pass
    return None


def score_json_parse(response: str) -> float:
    """1.0 if valid JSON, 0.0 otherwise."""
    return 1.0 if try_parse_json(response) is not None else 0.0


def score_phase_classification(response: str, expected: str) -> float:
    """Score phase classification; 1.0 exact, 0.3 adjacent, 0.0 wrong."""
    if not expected:
        return 0.0
    expected_phase = expected.upper().strip()
    for phase in ATTACK_PHASES:
        if phase in expected_phase:
            expected_phase = phase
            break

    response_upper = response.upper()
    if expected_phase in response_upper:
        return 1.0
    # Partial credit for adjacent phases
    for i, phase in enumerate(ATTACK_PHASES):
        if phase == expected_phase:
            neighbors = set()
            if i > 0:
                neighbors.add(ATTACK_PHASES[i - 1])
            if i < len(ATTACK_PHASES) - 1:
                neighbors.add(ATTACK_PHASES[i + 1])
            for n in neighbors:
                if n in response_upper:
                    return 0.3
            break
    return 0.0


def _extract_tool(text: str) -> str:
    """Extract first tool/command name from text or JSON."""
    parsed = try_parse_json(text)
    if parsed and isinstance(parsed, dict):
        cmd = parsed.get("command", parsed.get("cmd", parsed.get("tool", "")))
        if cmd:
            return cmd.split()[0].strip()
    for line in text.split("\n"):
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("//"):
            return line.split()[0].strip()
    return ""


def score_command_quality(response: str, expected_response: str) -> float:
    """Score command suggestion quality."""
    if not expected_response:
        return 0.0
    expected_tool = _extract_tool(expected_response)
    response_tool = _extract_tool(response)
    if not expected_tool:
        return 0.5
    if expected_tool.lower() == response_tool.lower():
        return 1.0
    if expected_tool.lower() in response.lower():
        return 0.7
    # Tool family match
    _fam = {
        "nmap": "scan", "masscan": "scan", "rustscan": "scan",
        "nikto": "web", "gobuster": "web", "dirb": "web", "wfuzz": "web",
        "feroxbuster": "web",
        "hydra": "brute", "medusa": "brute", "john": "brute", "hashcat": "brute",
        "ssh": "access", "telnet": "access", "ftp": "access",
        "msfconsole": "exploit", "searchsploit": "exploit",
        "cat": "read", "find": "search", "grep": "search", "ls": "enum",
        "wget": "download", "curl": "download",
        "python3": "script", "python": "script",
    }
    ef = _fam.get(expected_tool.lower(), "")
    rf = _fam.get(response_tool.lower(), "")
    if ef and ef == rf:
        return 0.5
    return 0.0


def score_reasoning_quality(response: str, expected: str) -> float:
    """Keyword-overlap F1 between response and expected text."""
    if not expected or not response:
        return 0.0
    _stop = {"the", "and", "for", "with", "this", "that", "from", "have",
             "has", "are", "was", "been", "will", "should", "could", "would",
             "not", "but", "can", "does", "did"}

    def kw(t: str) -> set[str]:
        return set(re.findall(r"\b[a-zA-Z_]{3,}\b", t.lower())) - _stop

    ek, rk = kw(expected), kw(response)
    if not ek:
        return 0.5
    overlap = len(ek & rk)
    prec = overlap / len(rk) if rk else 0
    rec = overlap / len(ek) if ek else 0
    if prec + rec == 0:
        return 0.0
    return min(2 * prec * rec / (prec + rec) * 1.5, 1.0)


# ── Per-family evaluation ────────────────────────────────────────────────────

def evaluate_model(model_name: str, samples_by_family: dict[str, list[dict]]) -> dict:
    """Evaluate a model on stratified holdout samples."""
    total = sum(len(v) for v in samples_by_family.values())
    log.info(f"Evaluating: {model_name} ({total} samples, "
             f"{len(samples_by_family)} families)")

    families_result: dict[str, dict] = {}
    all_latencies: list[float] = []
    all_scores: list[float] = []
    total_errors = 0

    for family, samples in sorted(samples_by_family.items()):
        fmet = {"json": [], "phase": [], "cmd": [], "reas": [],
                "lat": [], "err": 0, "count": len(samples)}

        for i, sample in enumerate(samples):
            messages = sample.get("messages", [])
            if not messages:
                continue

            # Split input vs expected
            expected_response = ""
            input_messages = []
            for msg in messages:
                if msg["role"] == "assistant":
                    expected_response = msg["content"]
                else:
                    input_messages.append(msg)
            if not input_messages:
                continue

            response, latency = ollama_generate(model_name, input_messages)
            if response.startswith("ERROR:"):
                fmet["err"] += 1
                total_errors += 1
                continue

            fmet["lat"].append(latency)
            all_latencies.append(latency)

            # Apply relevant scorers
            sample_scores: list[float] = []
            if family in JSON_FAMILIES:
                s = score_json_parse(response)
                fmet["json"].append(s)
                sample_scores.append(s)
            if family in PHASE_FAMILIES:
                phase = sample.get("metadata", {}).get("phase", expected_response)
                s = score_phase_classification(response, phase)
                fmet["phase"].append(s)
                sample_scores.append(s)
            if family in COMMAND_FAMILIES:
                s = score_command_quality(response, expected_response)
                fmet["cmd"].append(s)
                sample_scores.append(s)
            if family in REASONING_FAMILIES:
                s = score_reasoning_quality(response, expected_response)
                fmet["reas"].append(s)
                sample_scores.append(s)

            if sample_scores:
                all_scores.append(sum(sample_scores) / len(sample_scores))

            if (i + 1) % 20 == 0:
                log.info(f"  [{family}] {i + 1}/{len(samples)}")

        # Aggregate per family
        def _avg(lst: list[float]) -> float:
            return sum(lst) / len(lst) if lst else 0.0

        families_result[family] = {
            "count": fmet["count"],
            "errors": fmet["err"],
            "json_parse_rate": _avg(fmet["json"]),
            "phase_accuracy": _avg(fmet["phase"]),
            "command_quality": _avg(fmet["cmd"]),
            "reasoning_quality": _avg(fmet["reas"]),
            "avg_latency": _avg(fmet["lat"]),
        }

    # Global aggregates
    def _avg(lst: list[float]) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    json_rates = [families_result[f]["json_parse_rate"]
                  for f in JSON_FAMILIES
                  if f in families_result and families_result[f]["count"] > 0]
    cmd_rates = [families_result[f]["command_quality"]
                 for f in COMMAND_FAMILIES
                 if f in families_result and families_result[f]["count"] > 0]
    reas_rates = [families_result[f]["reasoning_quality"]
                  for f in REASONING_FAMILIES
                  if f in families_result and families_result[f]["reasoning_quality"] > 0]

    aggregate = {
        "overall_score": _avg(all_scores),
        "json_parse_rate": _avg(json_rates),
        "command_quality": _avg(cmd_rates),
        "reasoning_quality": _avg(reas_rates),
        "phase_accuracy": families_result.get("phase_classification", {}).get(
            "phase_accuracy", 0),
        "avg_latency": _avg(all_latencies),
        "p95_latency": (sorted(all_latencies)[int(len(all_latencies) * 0.95)]
                        if all_latencies else 0),
        "total_errors": total_errors,
    }

    return {"model": model_name, "total_samples": total,
            "families": families_result, "aggregate": aggregate}


# ── Data loading ─────────────────────────────────────────────────────────────

def load_holdout(max_total: int, per_family: int) -> dict[str, list[dict]]:
    """Load holdout samples, stratified by task family."""
    eval_file = HOLDOUT_FILE if HOLDOUT_FILE.exists() else LEGACY_EVAL_FILE
    if not eval_file.exists():
        log.error(f"No eval file found at {HOLDOUT_FILE} or {LEGACY_EVAL_FILE}")
        log.error("Run extract_dataset_v3.py first.")
        sys.exit(1)

    by_family: dict[str, list[dict]] = defaultdict(list)
    with open(eval_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            family = rec.get("task_family", "unknown")
            by_family[family].append(rec)

    log.info(f"Holdout loaded from {eval_file.name}: "
             f"{sum(len(v) for v in by_family.values())} total, "
             f"{len(by_family)} families")
    for fam in sorted(by_family):
        log.info(f"  {fam}: {len(by_family[fam])}")

    # Stratified sample
    random.seed(42)
    sampled: dict[str, list[dict]] = {}
    total = 0
    for fam in sorted(by_family):
        pool = by_family[fam]
        n = min(len(pool), per_family)
        if total + n > max_total:
            n = max(0, max_total - total)
        if n > 0:
            sampled[fam] = random.sample(pool, n) if n < len(pool) else pool
            total += n
    log.info(f"Sampled {total} for evaluation")
    return sampled


# ── Comparison printer ───────────────────────────────────────────────────────

def print_comparison(baseline: dict, finetuned: dict) -> None:
    """Print side-by-side comparison of baseline vs fine-tuned."""
    log.info("")
    log.info("=" * 78)
    log.info("  EVALUATION COMPARISON — Baseline vs Fine-Tuned")
    log.info("=" * 78)

    ba, fa = baseline["aggregate"], finetuned["aggregate"]
    metrics = [
        ("Overall Score", "overall_score", True),
        ("JSON Parse Rate", "json_parse_rate", True),
        ("Command Quality", "command_quality", True),
        ("Reasoning Quality", "reasoning_quality", True),
        ("Phase Accuracy", "phase_accuracy", True),
        ("Avg Latency (s)", "avg_latency", False),
        ("P95 Latency (s)", "p95_latency", False),
        ("Errors", "total_errors", False),
    ]

    log.info("")
    log.info("── Aggregate ──")
    for name, key, higher in metrics:
        bv, fv = ba.get(key, 0), fa.get(key, 0)
        delta = fv - bv
        if isinstance(bv, int) and isinstance(fv, int):
            mk = "+" if (delta > 0) == higher else ("-" if delta else "=")
            log.info(f"  {name:25s}: base={bv:5d}  ft={fv:5d}  [{mk}]")
        else:
            mk = "+" if (delta > 0) == higher else ("-" if delta else "=")
            log.info(f"  {name:25s}: base={bv:.3f}  ft={fv:.3f}  "
                     f"({delta:+.3f}) [{mk}]")

    log.info("")
    log.info("── Per-Family ──")
    all_fam = sorted(set(list(baseline["families"]) + list(finetuned["families"])))
    for fam in all_fam:
        bf = baseline["families"].get(fam, {})
        ff = finetuned["families"].get(fam, {})
        log.info(f"  {fam}:")
        for m in ("json_parse_rate", "phase_accuracy", "command_quality",
                  "reasoning_quality", "avg_latency"):
            bv, fv = bf.get(m, 0), ff.get(m, 0)
            if bv or fv:
                log.info(f"    {m:25s}: base={bv:.3f}  ft={fv:.3f}  "
                         f"({fv - bv:+.3f})")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Ariaska V3 models")
    parser.add_argument("--model", help="Single model (skip baseline comparison)")
    parser.add_argument("--baseline", default=BASELINE_MODEL)
    parser.add_argument("--finetuned", default=FINETUNED_MODEL)
    parser.add_argument("--max-samples", type=int, default=MAX_EVAL_SAMPLES)
    parser.add_argument("--per-family", type=int, default=SAMPLES_PER_FAMILY)
    parser.add_argument("--output", type=str, help="Output JSON path")
    args = parser.parse_args()

    samples = load_holdout(args.max_samples, args.per_family)
    results: dict = {}

    if args.model:
        log.info(f"\n{'=' * 60}")
        log.info(f"Evaluating: {args.model}")
        log.info(f"{'=' * 60}")
        results["model"] = evaluate_model(args.model, samples)
    else:
        log.info(f"\n{'=' * 60}")
        log.info(f"BASELINE: {args.baseline}")
        log.info(f"{'=' * 60}")
        results["baseline"] = evaluate_model(args.baseline, samples)

        log.info(f"\n{'=' * 60}")
        log.info(f"FINE-TUNED: {args.finetuned}")
        log.info(f"{'=' * 60}")
        results["finetuned"] = evaluate_model(args.finetuned, samples)

        print_comparison(results["baseline"], results["finetuned"])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else RESULTS_DIR / "eval_v3_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
