#!/usr/bin/env python3
"""
fp8_patch_schema.py — Use FP8 teacher from /dev/shm to regenerate missing schema examples.

Loads Qwen3-32B-FP8 from /dev/shm (already downloaded, ~32GB) for inference.
Generates only the missing phase_guided and microchain_generate examples.
Validates each with the exact same schema validation as v4 pipeline.
Appends valid examples to the schema file.

Usage:
    python3 /workspace/scripts/fp8_patch_schema.py
"""
import json
import random
import re
import sys
import time
import logging
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
SCHEMA_DATA = Path("/workspace/data/ariaska_schema_sft.jsonl")
FP8_TEACHER_DIR = Path("/dev/shm/qwen3-32b-fp8")

# ── Targets ──────────────────────────────────────────────────────────────────
TARGETS = {
    "phase_guided": 1200,
    "microchain_generate": 2000,
}

# ── Validation helpers (exact copy from v4 pipeline) ─────────────────────────
def _clean_json_text(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(l for l in lines if not l.strip().startswith("```")).strip()
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text

def _validate_json_array(text: str, required_fields: list[str]) -> bool:
    text = _clean_json_text(text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return False
    if not isinstance(parsed, list) or len(parsed) == 0:
        return False
    return all(isinstance(item, dict) and all(f in item for f in required_fields) for item in parsed)

def _validate_json_obj(text: str, required_fields: list[str]) -> bool:
    text = _clean_json_text(text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, dict) and all(f in parsed for f in required_fields)

# ── Schema configs (same as v4 pipeline) ─────────────────────────────────────
SCHEMA_CONFIGS = {
    "microchain_generate": {
        "system": "You are a command generator for Ariaska pentesting system. Output ONLY a JSON array of 1-3 objects, each with these exact keys: command, template_name, reasoning, evidence_used, hypothesis, test, expected_observable, stop_condition, confidence. confidence is a float 0.0-1.0. evidence_used is a list of strings. No markdown fences.",
        "max_tokens": 500,
        "validate": lambda t: _validate_json_array(t, ["command", "template_name", "reasoning", "evidence_used", "hypothesis", "test", "expected_observable", "stop_condition", "confidence"]),
    },
    "phase_guided": {
        "system": 'You are PHASE GUIDE for Ariaska_RL pentesting system. Output ONLY a raw JSON object with keys: phase_decision, anomalies, candidates, selection, distillation_packet. phase_decision must contain: chosen_phase, phase_confidence, phase_goal, stay_conditions, move_on_conditions, contradictions, phase_tag. phase_tag MUST be "P34". candidates is a list of objects with: template_name, family, why, expected_outcome, stop_condition, confidence, risk, tags. selection has: best_template_name, runner_up_template_name, selection_reason, should_escalate_to_codex, escalation_reason. distillation_packet has: observation, reasoning, action_target, expected_outcome, phase_target, confidence_target, gating_notes, phase_tag. No markdown fences.',
        "max_tokens": 800,
        "validate": lambda t: _validate_json_obj(t, ["phase_decision", "candidates", "selection", "distillation_packet"]),
    },
}

# ── Prompt generation helpers ────────────────────────────────────────────────
TEMPLATES = [
    "nmap_fast_scan", "nmap_version_detection", "nmap_vuln_scan", "nmap_all_ports",
    "gobuster_dir", "nikto_scan", "ffuf_dir", "hydra_ssh", "hydra_ftp",
    "ssh_login", "ftp_anonymous", "smbclient_list", "enum4linux",
    "sqlmap_test", "searchsploit", "msfconsole_exploit", "curl_request",
    "find_suid", "sudo_check", "linpeas", "getcap_check", "cat_file",
    "id_check", "whoami_check", "wget_download", "nc_reverse_shell",
]
PHASES = ["RECON", "ENUMERATION", "EXPLOITATION", "PRIVILEGE_ESCALATION", "POST_EXPLOITATION", "EXFILTRATION"]
ROLES = ["offensive", "recon", "defensive", "stealth", "strategic"]
IP_TARGETS = [
    "10.10.10.5", "10.10.10.40", "10.10.10.79", "192.168.1.10", "10.129.5.115",
    "10.10.10.15", "10.10.10.25", "10.10.10.50", "10.10.10.65", "10.10.10.95",
    "10.10.10.100", "10.10.10.120", "10.10.10.150", "10.10.10.175", "10.10.10.200",
    "10.129.1.10", "10.129.2.30", "10.129.3.50", "10.129.4.70",
    "192.168.1.15", "192.168.1.25", "192.168.1.45", "172.16.0.5", "172.16.0.15",
]


def _random_board():
    ports = random.sample(range(1, 65536), random.randint(0, 8))
    svcs = random.sample(["ssh", "http", "ftp", "smb", "mysql", "smtp", "dns", "rdp"], min(len(ports), random.randint(0, 5)))
    return {
        "ports": [str(p) for p in ports],
        "services": svcs,
        "credentials": [f"user{i}:pass{i}" for i in range(random.randint(0, 2))],
        "vulns": random.sample(["CVE-2021-44228", "CVE-2017-0144", "ms17-010", "CVE-2023-22515"], random.randint(0, 2)),
        "shells": ["root@target"] if random.random() < 0.15 else [],
        "users": [f"user{i}" for i in range(random.randint(0, 3))],
        "web_paths": random.sample(["/admin", "/login", "/api", "/upload", "/.git"], random.randint(0, 3)),
    }


def _make_user_prompt(schema_type, board, phase, role, target, stagnation, recent, templates):
    board_str = json.dumps(board, indent=1)
    recent_str = ", ".join(recent[-5:]) if recent else "none"
    tmpl_str = ", ".join(templates[:8])
    base = (
        f"Target: {target}\nPhase: {phase}\nRole: {role}\n"
        f"Stagnation: {stagnation} steps\n"
        f"Discovery board:\n{board_str}\n"
        f"Recent commands: {recent_str}\n"
        f"Available templates: {tmpl_str}"
    )
    prompts = {
        "microchain_generate": f"Classification: {random.choice(['recon_gap', 'enum_needed', 'exploit_ready', 'privesc_needed'])}\nGenerate 1-3 command candidates.\n\n{base}",
        "phase_guided": f"Provide full phase guidance with decision, candidates, selection, and distillation packet.\n\n{base}",
    }
    return prompts[schema_type]


def gpu_mem():
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"{used:.1f}/{total:.0f}GB"
    return "no GPU"


def generate_examples(model, tokenizer, schema_type: str, needed: int) -> list[dict]:
    """Generate examples for a schema type using FP8 model."""
    cfg = SCHEMA_CONFIGS[schema_type]
    system_prompt = cfg["system"]
    validate_fn = cfg["validate"]
    max_tokens = cfg["max_tokens"]

    # Over-generate by 3x to account for validation failures (FP8 should be better than AWQ)
    to_generate = needed * 3 + 10
    log.info(f"  Generating {to_generate} candidates for {needed} needed ({schema_type})")

    valid_examples = []
    attempts = 0

    for i in range(to_generate):
        if len(valid_examples) >= needed:
            break

        board = _random_board()
        phase = random.choice(PHASES)
        role = random.choice(ROLES)
        target = random.choice(IP_TARGETS)
        stagnation = random.randint(0, 25)
        recent = random.sample(TEMPLATES, min(len(TEMPLATES), random.randint(2, 6)))
        templates = random.sample(TEMPLATES, min(len(TEMPLATES), random.randint(4, 10)))

        user_prompt = _make_user_prompt(schema_type, board, phase, role, target, stagnation, recent, templates)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

        inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=2048)
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                temperature=0.4,
                top_p=0.9,
                repetition_penalty=1.05,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        # Decode only the generated tokens
        generated_ids = outputs[0][input_ids.shape[1]:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Clean think tags
        text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(l for l in lines if not l.strip().startswith("```")).strip()

        attempts += 1

        if validate_fn(text):
            valid_examples.append({
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": text},
                ],
                "schema_type": schema_type,
            })
            if len(valid_examples) % 5 == 0:
                log.info(f"    {schema_type}: {len(valid_examples)}/{needed} valid ({attempts} attempts, rate={len(valid_examples)*100//attempts}%)")

        # If struggling, try with stricter prompt on retry
        elif attempts > needed * 2 and len(valid_examples) < needed // 2:
            # Retry with reinforced prompt
            retry_messages = [
                {"role": "system", "content": system_prompt + "\n\nCRITICAL: Output ONLY the raw JSON. No explanation, no markdown. Every required key MUST be present."},
                {"role": "user", "content": user_prompt},
            ]
            retry_formatted = tokenizer.apply_chat_template(
                retry_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            retry_inputs = tokenizer(retry_formatted, return_tensors="pt", truncation=True, max_length=2048)
            retry_ids = retry_inputs["input_ids"].to(model.device)
            retry_mask = retry_inputs["attention_mask"].to(model.device)

            with torch.no_grad():
                retry_out = model.generate(
                    input_ids=retry_ids,
                    attention_mask=retry_mask,
                    max_new_tokens=max_tokens,
                    temperature=0.6,
                    top_p=0.95,
                    repetition_penalty=1.1,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            retry_gen = retry_out[0][retry_ids.shape[1]:]
            retry_text = tokenizer.decode(retry_gen, skip_special_tokens=True).strip()
            retry_text = re.sub(r'<think>.*?</think>\s*', '', retry_text, flags=re.DOTALL).strip()
            if retry_text.startswith("```"):
                lines = retry_text.split("\n")
                retry_text = "\n".join(l for l in lines if not l.strip().startswith("```")).strip()

            if validate_fn(retry_text):
                valid_examples.append({
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": retry_text},
                    ],
                    "schema_type": schema_type,
                })

    rate = len(valid_examples) * 100 // max(attempts, 1)
    log.info(f"  {schema_type}: {len(valid_examples)} valid from {attempts} attempts (rate={rate}%)")
    return valid_examples[:needed]


def main():
    random.seed(42)
    start = time.time()

    # Check FP8 teacher
    if not (FP8_TEACHER_DIR / "config.json").exists():
        log.error(f"FP8 teacher not found at {FP8_TEACHER_DIR}")
        sys.exit(1)

    # Count existing examples
    existing_counts = {}
    with open(SCHEMA_DATA) as f:
        for line in f:
            obj = json.loads(line)
            st = obj.get("schema_type", "unknown")
            existing_counts[st] = existing_counts.get(st, 0) + 1

    log.info("Current schema counts:")
    total_needed = 0
    for st, target in sorted(TARGETS.items()):
        count = existing_counts.get(st, 0)
        needed = max(0, target - count)
        total_needed += needed
        log.info(f"  {st}: {count}/{target} (need {needed})")

    if total_needed == 0:
        log.info("All types at target! Nothing to do.")
        return

    # Load FP8 teacher for inference
    log.info(f"Loading FP8 teacher from {FP8_TEACHER_DIR}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(FP8_TEACHER_DIR), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(FP8_TEACHER_DIR),
        dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    log.info(f"FP8 teacher loaded. GPU: {gpu_mem()}")

    # Generate missing examples
    new_examples = []
    for schema_type, target in TARGETS.items():
        count = existing_counts.get(schema_type, 0)
        needed = target - count
        if needed <= 0:
            log.info(f"\n{schema_type}: already at target ({count}/{target})")
            continue

        log.info(f"\n=== Generating {needed} {schema_type} examples ===")
        examples = generate_examples(model, tokenizer, schema_type, needed)
        new_examples.extend(examples)
        log.info(f"  Got {len(examples)} valid examples")

    # Unload model
    del model
    torch.cuda.empty_cache()
    import gc; gc.collect()
    log.info(f"Model unloaded. GPU: {gpu_mem()}")

    if not new_examples:
        log.info("No new examples generated!")
        return

    # Append to schema file
    with open(SCHEMA_DATA, "a") as f:
        for ex in new_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    log.info(f"\nAppended {len(new_examples)} examples to {SCHEMA_DATA}")

    # Final validation
    final_counts = {}
    total = 0
    invalid = 0
    with open(SCHEMA_DATA) as f:
        for line in f:
            total += 1
            obj = json.loads(line)
            st = obj.get("schema_type", "unknown")
            final_counts[st] = final_counts.get(st, 0) + 1
            # Validate assistant JSON
            try:
                json.loads(_clean_json_text(obj["messages"][2]["content"]))
            except Exception:
                invalid += 1

    log.info(f"\n{'='*60}")
    log.info(f"FINAL VALIDATION")
    log.info(f"{'='*60}")
    log.info(f"Total examples: {total}")
    log.info(f"Invalid JSON: {invalid}")
    all_ok = True
    for st in sorted(set(list(TARGETS.keys()) + list(final_counts.keys()))):
        count = final_counts.get(st, 0)
        target = TARGETS.get(st, count)
        pct = count / max(target, 1) * 100
        status = "OK" if count >= target else "LOW"
        if count < target:
            all_ok = False
        log.info(f"  {st:30s} {count:5d}/{target:5d} = {pct:5.1f}% [{status}]")

    elapsed = time.time() - start
    log.info(f"\nPatch complete in {elapsed/60:.1f} min")
    if all_ok:
        log.info("ALL TYPES AT TARGET OR ABOVE!")
    else:
        log.info("WARNING: Some types still below target — may need more generation attempts")


if __name__ == "__main__":
    main()
