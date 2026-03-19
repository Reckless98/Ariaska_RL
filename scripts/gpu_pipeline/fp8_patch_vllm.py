#!/usr/bin/env python3
"""
fp8_patch_vllm.py — Use vLLM + FP8 teacher to regenerate missing schema examples.

vLLM handles FP8 natively and is 10-20x faster than transformers for generation.
Uses offline batch inference for maximum throughput.

Usage:
    python3 /workspace/scripts/fp8_patch_vllm.py
"""
import json
import random
import re
import sys
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
SCHEMA_DATA = Path("/workspace/data/ariaska_schema_sft.jsonl")
FP8_TEACHER_DIR = "/dev/shm/qwen3-32b-fp8"

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

# ── Schema configs ───────────────────────────────────────────────────────────
SCHEMA_CONFIGS = {
    "microchain_generate": {
        "system": "You are a command generator for Ariaska pentesting system. Output ONLY a JSON array of 1-3 objects, each with these exact keys: command, template_name, reasoning, evidence_used, hypothesis, test, expected_observable, stop_condition, confidence. confidence is a float 0.0-1.0. evidence_used is a list of strings. No markdown fences. No thinking. No explanation. Raw JSON only.",
        "max_tokens": 700,
        "validate": lambda t: _validate_json_array(t, ["command", "template_name", "reasoning", "evidence_used", "hypothesis", "test", "expected_observable", "stop_condition", "confidence"]),
    },
    "phase_guided": {
        "system": 'You are PHASE GUIDE for Ariaska_RL pentesting system. Output ONLY a raw JSON object with keys: phase_decision, anomalies, candidates, selection, distillation_packet. phase_decision must contain: chosen_phase, phase_confidence, phase_goal, stay_conditions, move_on_conditions, contradictions, phase_tag. phase_tag MUST be "P34". candidates is a list of objects with: template_name, family, why, expected_outcome, stop_condition, confidence, risk, tags. selection has: best_template_name, runner_up_template_name, selection_reason, should_escalate_to_codex, escalation_reason. distillation_packet has: observation, reasoning, action_target, expected_outcome, phase_target, confidence_target, gating_notes, phase_tag. No markdown fences. No thinking. No explanation. Raw JSON only.',
        "max_tokens": 1200,
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


def build_prompts(schema_type: str, count: int) -> list[list[dict]]:
    """Build a list of chat message sets for batch generation."""
    prompts = []
    for _ in range(count):
        board = _random_board()
        phase = random.choice(PHASES)
        role = random.choice(ROLES)
        target = random.choice(IP_TARGETS)
        stagnation = random.randint(0, 25)
        recent = random.sample(TEMPLATES, min(len(TEMPLATES), random.randint(2, 6)))
        templates = random.sample(TEMPLATES, min(len(TEMPLATES), random.randint(4, 10)))

        user_prompt = _make_user_prompt(schema_type, board, phase, role, target, stagnation, recent, templates)
        cfg = SCHEMA_CONFIGS[schema_type]

        prompts.append([
            {"role": "system", "content": cfg["system"]},
            {"role": "user", "content": user_prompt},
        ])
    return prompts


def strip_thinking(text: str) -> str:
    """Remove thinking tags and any content between them."""
    text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()
    # Also handle unclosed thinking tags
    text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL).strip()
    return text


def main():
    random.seed(42)
    start = time.time()

    # Check FP8 teacher
    if not Path(FP8_TEACHER_DIR, "config.json").exists():
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
    types_to_generate = {}
    for st, target in sorted(TARGETS.items()):
        count = existing_counts.get(st, 0)
        needed = max(0, target - count)
        if needed > 0:
            # Over-generate by 4x to account for validation failures
            types_to_generate[st] = (needed, needed * 4 + 20)
        log.info(f"  {st}: {count}/{target} (need {needed})")

    if not types_to_generate:
        log.info("All types at target! Nothing to do.")
        return

    # Load vLLM engine with FP8 model
    log.info(f"Loading vLLM engine with FP8 model from {FP8_TEACHER_DIR}...")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=FP8_TEACHER_DIR,
        dtype="auto",  # Will use FP8 as configured in model
        tensor_parallel_size=1,
        gpu_memory_utilization=0.92,
        max_model_len=2048,
        trust_remote_code=True,
    )
    log.info("vLLM engine loaded!")

    # Process each type
    all_new_examples = []

    for schema_type, (needed, generate_count) in types_to_generate.items():
        cfg = SCHEMA_CONFIGS[schema_type]
        validate_fn = cfg["validate"]
        max_tokens = cfg["max_tokens"]

        log.info(f"\n=== {schema_type}: generating {generate_count} candidates for {needed} needed ===")

        # Build all prompts
        prompt_sets = build_prompts(schema_type, generate_count)

        # Use tokenizer to format chat templates
        tokenizer = llm.get_tokenizer()
        formatted_prompts = []
        for msgs in prompt_sets:
            try:
                text = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False
                )
            except TypeError:
                # Fallback if enable_thinking not supported
                text = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
            formatted_prompts.append(text)

        # Batch generate
        sampling = SamplingParams(
            temperature=0.4,
            top_p=0.9,
            max_tokens=max_tokens,
            repetition_penalty=1.05,
        )

        log.info(f"  Batch generating {len(formatted_prompts)} prompts...")
        t0 = time.time()
        outputs = llm.generate(formatted_prompts, sampling)
        gen_time = time.time() - t0
        log.info(f"  Generation took {gen_time:.1f}s ({gen_time/len(formatted_prompts):.2f}s/example)")

        # Validate outputs
        valid_examples = []
        fail_reasons = {"json_parse": 0, "missing_fields": 0, "thinking": 0, "empty": 0}

        for i, output in enumerate(outputs):
            text = output.outputs[0].text.strip()

            # Strip thinking tags
            original_len = len(text)
            text = strip_thinking(text)
            if len(text) < original_len * 0.5 and original_len > 50:
                fail_reasons["thinking"] += 1

            # Clean and validate
            if not text:
                fail_reasons["empty"] += 1
                continue

            if validate_fn(text):
                valid_examples.append({
                    "messages": [
                        {"role": "system", "content": cfg["system"]},
                        {"role": "user", "content": prompt_sets[i][1]["content"]},
                        {"role": "assistant", "content": text},
                    ],
                    "schema_type": schema_type,
                })
            else:
                # Diagnose failure
                cleaned = _clean_json_text(text)
                try:
                    json.loads(cleaned)
                    fail_reasons["missing_fields"] += 1
                except json.JSONDecodeError:
                    fail_reasons["json_parse"] += 1

            if len(valid_examples) >= needed:
                break

        rate = len(valid_examples) * 100 // max(len(outputs), 1)
        log.info(f"  Results: {len(valid_examples)} valid / {len(outputs)} total = {rate}% pass rate")
        log.info(f"  Failures: {fail_reasons}")

        # If not enough, try a retry batch with reinforced prompting
        if len(valid_examples) < needed:
            shortfall = needed - len(valid_examples)
            retry_count = shortfall * 5 + 10
            log.info(f"  Need {shortfall} more — retrying with {retry_count} reinforced prompts...")

            retry_prompts = build_prompts(schema_type, retry_count)
            # Add reinforcement to system prompt
            for msgs in retry_prompts:
                msgs[0]["content"] += "\n\nCRITICAL: Output ONLY the raw JSON. Every required key MUST be present. Start directly with { or [."

            retry_formatted = []
            for msgs in retry_prompts:
                try:
                    text = tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True,
                        enable_thinking=False
                    )
                except TypeError:
                    text = tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True
                    )
                retry_formatted.append(text)

            retry_sampling = SamplingParams(
                temperature=0.6,
                top_p=0.95,
                max_tokens=max_tokens + 300,  # Extra headroom
                repetition_penalty=1.1,
            )

            retry_outputs = llm.generate(retry_formatted, retry_sampling)
            for i, output in enumerate(retry_outputs):
                if len(valid_examples) >= needed:
                    break
                text = strip_thinking(output.outputs[0].text.strip())
                if text and validate_fn(text):
                    valid_examples.append({
                        "messages": [
                            {"role": "system", "content": cfg["system"]},
                            {"role": "user", "content": retry_prompts[i][1]["content"]},
                            {"role": "assistant", "content": text},
                        ],
                        "schema_type": schema_type,
                    })

            log.info(f"  After retry: {len(valid_examples)} valid total")

        all_new_examples.extend(valid_examples[:needed])
        log.info(f"  Using {min(len(valid_examples), needed)} examples")

    # Cleanup vLLM engine
    del llm
    import gc
    gc.collect()

    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    if not all_new_examples:
        log.error("No new examples generated!")
        sys.exit(1)

    # Append to schema file
    with open(SCHEMA_DATA, "a") as f:
        for ex in all_new_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    log.info(f"\nAppended {len(all_new_examples)} examples to {SCHEMA_DATA}")

    # Final validation
    final_counts = {}
    total = 0
    with open(SCHEMA_DATA) as f:
        for line in f:
            total += 1
            obj = json.loads(line)
            st = obj.get("schema_type", "unknown")
            final_counts[st] = final_counts.get(st, 0) + 1

    log.info(f"\n{'='*60}")
    log.info(f"FINAL COUNTS (total: {total})")
    log.info(f"{'='*60}")
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
        log.info("ALL TYPES AT TARGET! Ready for v4 pipeline.")
    else:
        log.info("WARNING: Some types still below target")


if __name__ == "__main__":
    main()
