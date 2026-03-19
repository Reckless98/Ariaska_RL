#!/usr/bin/env python3
"""
ariaska_cybersec4_nuclear.py — Ariaska Cybersec 2.0 NUCLEAR Pipeline

THE NUCLEAR OPTION: KD-fused SFT — teacher guides the student on the EXACT
schema-compliant examples. No separate KD stage. Teacher's soft logits and
hard labels are combined in a SINGLE training pass on the format-critical data.

Pipeline (6 stages):
  Stage 1: Distill schema data from AWQ teacher (vLLM, data generation)
  Stage 2: Delete AWQ teacher, download Qwen3-4B base student
  Stage 3: CPT — domain knowledge injection (LoRA r=128 on base student)
  Stage 4: KD-SFT FUSION — FP8 teacher loaded alongside CPT-enhanced student
           Combined loss: α_kd * KL(softmax(S/T), softmax(T_teacher/T)) * T²
                        + α_ce * CE(student_logits, hard_labels)
           Schema 8x oversampled. Teacher guides on EXACT right outputs.
  Stage 5: DPO — preference alignment (sticks last)
  Stage 6: Merge all adapters → GGUF Q8_0 + Q6_K

Why this is the strongest approach:
  - Teacher's soft probability distribution guides student on format-critical
    examples — KD signal is PERFECTLY aligned with schema compliance
  - Zero risk of SFT overwriting KD knowledge (they happen simultaneously)
  - Student learns FORMAT from hard labels AND REASONING DEPTH from teacher
    soft logits in one pass
  - Eliminates an entire stage (separate KD + SFT → single KD-SFT)
  - CPT first gives domain knowledge, KD-SFT then teaches format + depth

Teachers:
  - AWQ (Stage 1 only): Qwen3-32B-AWQ on disk for vLLM data generation
  - FP8 (Stage 4 KD-SFT): Qwen3-32B-FP8 via /dev/shm (RAM-backed, 47GB tmpfs)

Memory budget (RTX 4090, 49GB VRAM):
  Stage 1: AWQ via vLLM ~19GB
  Stage 4: FP8 teacher ~34GB + student bf16 LoRA ~10GB = ~44GB (tight, fits)
  Other:   Student bf16 LoRA ~15GB (comfortable)

Disk plan (40GB total):
  AWQ teacher: 19GB on disk (deleted after Stage 1)
  FP8 teacher: in /dev/shm RAM (0 disk usage!)
  Student base: ~8GB on disk
  Adapters + merged: ~12GB
  GGUFs: ~7GB
"""

from __future__ import annotations
import gc, json, logging, math, os, random, re, shutil, subprocess, sys, time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler("/workspace/training.log", mode="a")],
)
log = logging.getLogger("ariaska.pipeline")

# ── Paths ────────────────────────────────────────────────────────────────────
AWQ_TEACHER_MODEL = "/workspace/models/qwen3-32b-awq"
FP8_TEACHER_ID = "Qwen/Qwen3-32B-FP8"
FP8_TEACHER_DIR = Path("/dev/shm/qwen3-32b-fp8")  # RAM-backed! No disk usage

STUDENT_MODEL_ID = "Qwen/Qwen3-4B"
STUDENT_MODEL_DIR = Path("/workspace/models/qwen3-4b")

DATA_DIR = Path("/workspace/data/v3")
SCHEMA_DATA = Path("/workspace/data/ariaska_schema_sft.jsonl")
OUTPUT = Path("/workspace/output")
CPT_OUT = OUTPUT / "cpt"
KDSFT_OUT = OUTPUT / "kdsft"    # Nuclear KD-SFT fusion output
DPO_OUT = OUTPUT / "dpo"
MERGED_DIR = OUTPUT / "merged"
GGUF_DIR = OUTPUT / "gguf"

ALL_LINEAR = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# ── KD-SFT Fusion Hyperparameters ───────────────────────────────────────────
KDSFT_CONFIG = {
    "temperature_start": 4.0,       # High T = softer distributions = more dark knowledge
    "temperature_end": 2.0,         # Anneal to sharper distributions
    "alpha_kd": 0.6,                # Weight for KD loss (teacher soft targets)
    "alpha_ce": 0.4,                # Weight for hard label CE loss
    "learning_rate": 1e-4,          # Between KD(5e-5) and SFT(2e-4)
    "num_epochs": 3,                # Between KD(2) and SFT(4)
    "batch_size": 1,                # batch=1 to fit FP8 teacher (34GB) + student (10GB)
    "gradient_accumulation": 16,    # Effective batch = 16
    "max_seq_len": 1024,            # VRAM compromise with teacher loaded
    "warmup_ratio": 0.08,
    "lora_r": 128,                  # Full-rank since this IS the format-teaching stage
    "lora_alpha": 256,
    "lora_dropout": 0.05,
    "max_grad_norm": 1.0,
    "weight_decay": 0.01,
    "schema_oversample": 8,         # 8x schema oversample — FORMAT IS THE TARGET
    "neftune_noise_alpha": 5.0,     # NEFTune noise injection for generalization
}


def gpu_mem():
    if torch.cuda.is_available():
        return f"{torch.cuda.memory_allocated()/1e9:.1f}/{torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB"
    return "no-gpu"

def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def _load_and_merge(base_dir: Path, adapters: list[Path],
                    device_map: str = "auto",
                    dtype: torch.dtype = torch.bfloat16) -> "PreTrainedModel":
    """Load base model and sequentially merge LoRA adapters."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    kwargs: dict = dict(
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if device_map != "cpu":
        kwargs["attn_implementation"] = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(str(base_dir), **kwargs)
    for adapter_path in adapters:
        adapter_path = Path(adapter_path)
        if adapter_path.exists() and (adapter_path / "adapter_config.json").exists():
            log.info(f"  Merging adapter: {adapter_path}")
            model = PeftModel.from_pretrained(model, str(adapter_path), torch_dtype=dtype)
            model = model.merge_and_unload()
    return model


# ── Validation helpers ───────────────────────────────────────────────────────

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


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1: Distill Schema Data from AWQ Teacher (DATA generation, not KD)
# ══════════════════════════════════════════════════════════════════════════════

def stage1_distill_schema():
    """Generate schema-perfect training data using Qwen3-32B-AWQ teacher via vLLM."""
    if SCHEMA_DATA.exists() and SCHEMA_DATA.stat().st_size > 1000:
        lines = sum(1 for _ in open(SCHEMA_DATA))
        if lines >= 8000:
            log.info(f"Schema data exists: {lines} lines, skipping generation")
            return
        log.info(f"Schema data only {lines} lines, regenerating...")

    log.info("=" * 60)
    log.info("STAGE 1: Distill Schema Data from AWQ Teacher (data generation)")
    log.info("=" * 60)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(AWQ_TEACHER_MODEL, trust_remote_code=True)

    log.info(f"Loading AWQ teacher for data gen: {AWQ_TEACHER_MODEL}")
    llm = LLM(
        model=AWQ_TEACHER_MODEL,
        quantization="awq",
        max_model_len=4096,
        gpu_memory_utilization=0.88,
        trust_remote_code=True,
        dtype="float16",
    )
    log.info(f"AWQ teacher loaded. GPU: {gpu_mem()}")

    SCHEMA_CONFIGS = {
        "microchain_classify": {
            "system": "You are a tactical situation classifier for Ariaska pentesting system. Respond with ONLY one word from: recon_gap, enum_needed, exploit_ready, privesc_needed, post_exploit, lateral_move, stalled",
            "max_tokens": 20, "count": 1500,
            "validate": lambda t: t.strip().lower().rstrip(".") in {"recon_gap", "enum_needed", "exploit_ready", "privesc_needed", "post_exploit", "lateral_move", "stalled"},
        },
        "microchain_generate": {
            "system": "You are a command generator for Ariaska pentesting system. Output ONLY a JSON array of 1-3 objects, each with these exact keys: command, template_name, reasoning, evidence_used, hypothesis, test, expected_observable, stop_condition, confidence. confidence is a float 0.0-1.0. evidence_used is a list of strings. No markdown fences.",
            "max_tokens": 500, "count": 2000,
            "validate": lambda t: _validate_json_array(t, ["command", "template_name", "reasoning", "evidence_used", "hypothesis", "test", "expected_observable", "stop_condition", "confidence"]),
        },
        "microchain_score": {
            "system": "You are a command scorer for Ariaska pentesting system. Output ONLY a JSON array of score objects, each with: idx (integer), phase_fit (0.0-1.0), evidence_support (0.0-1.0), novelty (0.0-1.0). No markdown.",
            "max_tokens": 200, "count": 1800,
            "validate": lambda t: _validate_json_array(t, ["idx", "phase_fit", "evidence_support", "novelty"]),
        },
        "microchain_fast_local": {
            "system": "You are a fast command selector for Ariaska pentesting system. Output ONLY a JSON object with: command, template_name, reasoning, score (0.0-1.0). No markdown.",
            "max_tokens": 200, "count": 1800,
            "validate": lambda t: _validate_json_obj(t, ["command", "template_name", "reasoning", "score"]),
        },
        "phase_guided": {
            "system": 'You are PHASE GUIDE for Ariaska_RL pentesting system. Output ONLY a raw JSON object with keys: phase_decision, anomalies, candidates, selection, distillation_packet. phase_decision must contain: chosen_phase, phase_confidence, phase_goal, stay_conditions, move_on_conditions, contradictions, phase_tag. phase_tag MUST be "P34". candidates is a list of objects with: template_name, family, why, expected_outcome, stop_condition, confidence, risk, tags. selection has: best_template_name, runner_up_template_name, selection_reason, should_escalate_to_codex, escalation_reason. distillation_packet has: observation, reasoning, action_target, expected_outcome, phase_target, confidence_target, gating_notes, phase_tag. No markdown fences.',
            "max_tokens": 800, "count": 1200,
            "validate": lambda t: _validate_json_obj(t, ["phase_decision", "candidates", "selection", "distillation_packet"]),
        },
        "phase_guided_fast_local": {
            "system": 'You are a fast phase advisor for Ariaska pentesting system. Output ONLY a JSON object with: stay_or_advance ("stay" or "advance"), reason, candidates (list of template name strings), confidence (0.0-1.0). No markdown.',
            "max_tokens": 200, "count": 1800,
            "validate": lambda t: _validate_json_obj(t, ["stay_or_advance", "reason", "candidates", "confidence"]),
        },
        "smart_mentor": {
            "system": "You are an elite pentester AI MENTOR for Ariaska system. Select the BEST next action. Output ONLY valid JSON with: intent, selected_command (MUST be a template_name like nmap_fast_scan), parameters (dict), reasoning, expected_observation, risk (low/medium/high), confidence (0.0-1.0), next_phase_hint, candidate_actions (list of dicts). No markdown.",
            "max_tokens": 400, "count": 1800,
            "validate": lambda t: _validate_json_obj(t, ["intent", "selected_command", "parameters", "reasoning", "expected_observation", "risk", "confidence"]),
        },
        "coherence_classify": {
            "system": "You are a state coherence classifier for Ariaska pentesting system. Output ONLY a JSON object with: phase_guess (RECON/ENUMERATION/EXPLOITATION/PRIVILEGE_ESCALATION/LATERAL_MOVEMENT/POST_EXPLOITATION/EXFILTRATION), phase_confidence (0.0-1.0), key_evidence (list), missing_evidence (list), next_best_families (list). No markdown.",
            "max_tokens": 250, "count": 1000,
            "validate": lambda t: _validate_json_obj(t, ["phase_guess", "phase_confidence", "key_evidence"]),
        },
        "coherence_summarize": {
            "system": "You are a state summarizer for Ariaska pentesting system. Output ONLY a JSON object with: postcard (one-line state summary string), evidence_counts (dict of category:count). No markdown.",
            "max_tokens": 200, "count": 1000,
            "validate": lambda t: _validate_json_obj(t, ["postcard", "evidence_counts"]),
        },
        "coherence_score": {
            "system": "You are a coherence scorer for Ariaska pentesting system. Output ONLY a JSON object with: coherence_score (0.0-1.0), novelty_score (0.0-1.0), repeat_risk (0.0-1.0), confidence_calibration (0.0-1.0). No markdown.",
            "max_tokens": 100, "count": 1000,
            "validate": lambda t: _validate_json_obj(t, ["coherence_score", "novelty_score", "repeat_risk", "confidence_calibration"]),
        },
    }

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
    TARGETS = ["10.10.10.5", "10.10.10.40", "10.10.10.79", "192.168.1.10", "10.129.5.115"]

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
            "microchain_classify": f"Classify the current tactical situation.\n\n{base}",
            "microchain_generate": f"Classification: {random.choice(['recon_gap', 'enum_needed', 'exploit_ready', 'privesc_needed'])}\nGenerate 1-3 command candidates.\n\n{base}",
            "microchain_score": f"Score these candidates for phase fit:\n{json.dumps([{'command': f'nmap -sV {target}', 'template_name': templates[0]}, {'command': f'gobuster dir -u http://{target}', 'template_name': templates[1] if len(templates)>1 else 'gobuster_dir'}])}\n\n{base}",
            "microchain_fast_local": f"Pick the single best next command.\n\n{base}",
            "phase_guided": f"Provide full phase guidance with decision, candidates, selection, and distillation packet.\n\n{base}",
            "phase_guided_fast_local": f"Quick phase decision: stay or advance? Give 3 candidate template names.\n\n{base}",
            "smart_mentor": f"Select the best next command. selected_command MUST be a template_name from available.\n\n{base}",
            "coherence_classify": f"Classify the current phase from this evidence.\n\n{base}",
            "coherence_summarize": f"Summarize the current state as a compact postcard.\n\n{base}",
            "coherence_score": f"Score the coherence, novelty, repeat risk, and confidence calibration.\n\n{base}",
        }
        return prompts.get(schema_type, base)

    all_examples = []
    total_target = sum(c["count"] for c in SCHEMA_CONFIGS.values())
    log.info(f"Target: {total_target} schema examples across {len(SCHEMA_CONFIGS)} types")

    for schema_type, cfg in SCHEMA_CONFIGS.items():
        target_count = cfg["count"]
        gen_count = int(target_count * 1.5)
        max_tok = cfg["max_tokens"]
        validate_fn = cfg["validate"]
        system_prompt = cfg["system"]
        batch_size = 128 if max_tok <= 100 else 96 if max_tok <= 250 else 64 if max_tok <= 500 else 48

        log.info(f"\n--- {schema_type}: target={target_count}, generating={gen_count} ---")

        all_prompts_formatted = []
        all_user_prompts = []
        for _ in range(gen_count):
            board = _random_board()
            phase = random.choice(PHASES)
            role = random.choice(ROLES)
            target = random.choice(TARGETS)
            stagnation = random.randint(0, 25)
            recent = random.sample(TEMPLATES, min(len(TEMPLATES), random.randint(2, 6)))
            templates = random.sample(TEMPLATES, min(len(TEMPLATES), random.randint(4, 10)))
            user_prompt = _make_user_prompt(schema_type, board, phase, role, target, stagnation, recent, templates)
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            all_prompts_formatted.append(formatted)
            all_user_prompts.append(user_prompt)

        valid_examples = []
        failed_prompts_formatted = []
        failed_user_prompts = []
        params = SamplingParams(temperature=0.4, top_p=0.9, max_tokens=max_tok, repetition_penalty=1.05)

        for batch_start in range(0, len(all_prompts_formatted), batch_size):
            batch_end = min(batch_start + batch_size, len(all_prompts_formatted))
            batch_formatted = all_prompts_formatted[batch_start:batch_end]
            batch_users = all_user_prompts[batch_start:batch_end]
            try:
                outputs = llm.generate(batch_formatted, params)
            except Exception as e:
                log.warning(f"Batch failed: {e}")
                continue

            for output, user_prompt in zip(outputs, batch_users):
                text = output.outputs[0].text.strip()
                text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()
                if text.startswith("```"):
                    lines = text.split("\n")
                    text = "\n".join(l for l in lines if not l.strip().startswith("```")).strip()
                if validate_fn(text):
                    valid_examples.append({
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                            {"role": "assistant", "content": text},
                        ],
                        "schema_type": schema_type,
                    })
                else:
                    failed_prompts_formatted.append(
                        tokenizer.apply_chat_template(
                            [{"role": "system", "content": system_prompt + "\n\nCRITICAL: Output ONLY the raw JSON. No explanation, no markdown."},
                             {"role": "user", "content": user_prompt}],
                            tokenize=False, add_generation_prompt=True, enable_thinking=False))
                    failed_user_prompts.append(user_prompt)
            print(f"  {schema_type}: {len(valid_examples)}/{target_count} valid  ({batch_end}/{gen_count} generated)", end="\r")

        # Retry failures
        if len(valid_examples) < target_count and failed_prompts_formatted:
            retry_count = min(len(failed_prompts_formatted), target_count - len(valid_examples) + 100)
            retry_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=max_tok, repetition_penalty=1.1)
            log.info(f"  Retrying {retry_count} failures...")
            for batch_start in range(0, retry_count, batch_size):
                batch_end = min(batch_start + batch_size, retry_count)
                batch = failed_prompts_formatted[batch_start:batch_end]
                batch_users = failed_user_prompts[batch_start:batch_end]
                try:
                    outputs = llm.generate(batch, retry_params)
                    for output, user_prompt in zip(outputs, batch_users):
                        text = output.outputs[0].text.strip()
                        text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()
                        if text.startswith("```"):
                            lines = text.split("\n")
                            text = "\n".join(l for l in lines if not l.strip().startswith("```")).strip()
                        if validate_fn(text):
                            valid_examples.append({
                                "messages": [{"role": "system", "content": system_prompt},
                                             {"role": "user", "content": user_prompt},
                                             {"role": "assistant", "content": text}],
                                "schema_type": schema_type,
                            })
                except Exception as e:
                    log.warning(f"Retry batch failed: {e}")

        final = valid_examples[:target_count]
        rate = len(valid_examples) * 100 // max(gen_count, 1)
        log.info(f"  {schema_type}: {len(final)} examples (validation rate: {rate}%)")
        all_examples.extend(final)

    random.shuffle(all_examples)
    SCHEMA_DATA.parent.mkdir(parents=True, exist_ok=True)
    with open(SCHEMA_DATA, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    log.info(f"\nSchema data written: {len(all_examples)} examples -> {SCHEMA_DATA}")

    del llm
    cleanup()
    log.info("AWQ teacher unloaded from GPU")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2: Delete AWQ Teacher, Download Student Base
# ══════════════════════════════════════════════════════════════════════════════

def stage2_prepare_student():
    """Delete AWQ teacher (free 19GB disk), download Qwen3-4B base."""
    teacher_path = Path(AWQ_TEACHER_MODEL)
    if teacher_path.exists():
        size = sum(f.stat().st_size for f in teacher_path.rglob("*") if f.is_file()) / 1e9
        log.info(f"Deleting AWQ teacher ({size:.1f}GB) to free disk...")
        shutil.rmtree(teacher_path)
        log.info("AWQ teacher deleted.")

    # Clean HF caches
    for hf_cache_dir in [Path("/workspace/.hf_home"), Path.home() / ".cache" / "huggingface"]:
        if hf_cache_dir.exists():
            cache_size = sum(f.stat().st_size for f in hf_cache_dir.rglob("*") if f.is_file()) / 1e9
            if cache_size > 0.1:
                log.info(f"Cleaning HF cache: {hf_cache_dir} ({cache_size:.1f}GB)")
                shutil.rmtree(hf_cache_dir, ignore_errors=True)

    if (STUDENT_MODEL_DIR / "config.json").exists():
        log.info(f"Student model already at {STUDENT_MODEL_DIR}")
        return STUDENT_MODEL_DIR

    log.info("=" * 60)
    log.info(f"STAGE 2: Downloading {STUDENT_MODEL_ID}")
    log.info("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    STUDENT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Downloading {STUDENT_MODEL_ID}...")

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID, trust_remote_code=True)
    tokenizer.save_pretrained(str(STUDENT_MODEL_DIR))

    model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL_ID, torch_dtype=torch.bfloat16, device_map="cpu",
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    model.save_pretrained(str(STUDENT_MODEL_DIR), safe_serialization=True)
    del model
    cleanup()

    hf_cache = Path.home() / ".cache" / "huggingface"
    if hf_cache.exists():
        shutil.rmtree(hf_cache, ignore_errors=True)
        log.info("Cleaned HF cache")

    log.info(f"Student saved: {STUDENT_MODEL_DIR}")
    return STUDENT_MODEL_DIR


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3: CPT — Continual Pre-Training on Base Student
# ══════════════════════════════════════════════════════════════════════════════

def stage3_cpt(base_dir: Path) -> Path:
    """CPT: domain knowledge injection on base student. Runs BEFORE KD-SFT."""
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer, SFTConfig

    final = CPT_OUT / "final"
    if (final / "adapter_config.json").exists():
        log.info(f"CPT adapter exists: {final}")
        return final

    CPT_OUT.mkdir(parents=True, exist_ok=True)
    log.info("=" * 60)
    log.info("STAGE 3: CPT — Full LoRA bf16, r=128 (domain knowledge)")
    log.info("=" * 60)

    cpt_docs = load_jsonl(DATA_DIR / "cpt_corpus.jsonl")
    log.info(f"CPT corpus: {len(cpt_docs)} documents")

    split_idx = int(len(cpt_docs) * 0.97)
    random.seed(42)
    random.shuffle(cpt_docs)
    train_ds = Dataset.from_list(cpt_docs[:split_idx])
    eval_ds = Dataset.from_list(cpt_docs[split_idx:])

    tokenizer = AutoTokenizer.from_pretrained(str(base_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info("Loading base student in bf16...")
    model = AutoModelForCausalLM.from_pretrained(
        str(base_dir), torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="sdpa",
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    lora_config = LoraConfig(
        r=128, lora_alpha=256, lora_dropout=0.05,
        target_modules=ALL_LINEAR, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    log.info(f"GPU after LoRA: {gpu_mem()}")

    sft_config = SFTConfig(
        output_dir=str(CPT_OUT), num_train_epochs=2,
        per_device_train_batch_size=4, per_device_eval_batch_size=4,
        gradient_accumulation_steps=4, learning_rate=5e-5,
        lr_scheduler_type="cosine", warmup_ratio=0.05, weight_decay=0.01,
        max_length=768, packing=True, logging_steps=50,
        save_strategy="steps", save_steps=500,
        eval_strategy="steps", eval_steps=500, save_total_limit=2,
        fp16=False, bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=4, dataloader_prefetch_factor=2,
        seed=42, report_to="none", dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model, args=sft_config,
        train_dataset=train_ds, eval_dataset=eval_ds, processing_class=tokenizer,
    )

    log.info(f"CPT: {len(train_ds)} train, {len(eval_ds)} eval, 2 epochs, LoRA r=128")
    trainer.train()

    final.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))

    metrics = trainer.evaluate()
    log.info(f"CPT metrics: {json.dumps(metrics, indent=2)}")

    del model, trainer
    cleanup()
    return final


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4: KD-SFT FUSION — THE NUCLEAR OPTION
# ══════════════════════════════════════════════════════════════════════════════
#
# FP8 teacher loaded alongside CPT-enhanced student. Combined loss:
#
#   loss = α_kd * KL(softmax(student/T), softmax(teacher/T)) * T²
#        + α_ce * CE(student_logits, hard_labels)
#
# Teacher guides student on the EXACT schema-compliant outputs.
# Format compliance + reasoning depth in a SINGLE training pass.
#
# This is superior to separate KD→SFT because:
#   - Teacher signal is applied WHERE it matters (format-critical examples)
#   - No risk of SFT overwriting KD knowledge
#   - Student learns from BOTH soft distribution AND hard labels simultaneously
#   - One pass instead of two = faster + more coherent
# ══════════════════════════════════════════════════════════════════════════════

def _stage4_download_fp8_teacher():
    """Download FP8 teacher to /dev/shm (RAM-backed, no disk usage)."""
    log.info("=" * 60)
    log.info("STAGE 4a: Downloading FP8 teacher to /dev/shm (RAM)")
    log.info("=" * 60)

    if (FP8_TEACHER_DIR / "config.json").exists():
        log.info(f"FP8 teacher already at {FP8_TEACHER_DIR}")
        return

    shm_stat = shutil.disk_usage("/dev/shm")
    shm_free_gb = shm_stat.free / 1e9
    log.info(f"/dev/shm: {shm_free_gb:.1f}GB free / {shm_stat.total/1e9:.0f}GB total")

    if shm_free_gb < 34:
        log.error(f"FATAL: /dev/shm only has {shm_free_gb:.1f}GB free. Need ~34GB for FP8 teacher.")
        sys.exit(1)

    from huggingface_hub import snapshot_download

    os.environ["HF_HOME"] = "/dev/shm/.hf_cache"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    FP8_TEACHER_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Downloading {FP8_TEACHER_ID} to /dev/shm...")
    snapshot_download(
        FP8_TEACHER_ID,
        local_dir=str(FP8_TEACHER_DIR),
        local_dir_use_symlinks=False,
    )

    shm_cache = Path("/dev/shm/.hf_cache")
    if shm_cache.exists():
        shutil.rmtree(shm_cache, ignore_errors=True)
        log.info("Cleaned HF cache from /dev/shm")

    os.environ.pop("HF_HOME", None)
    log.info(f"FP8 teacher downloaded to RAM: {FP8_TEACHER_DIR}")


def _stage4_prepare_kdsft_data(tokenizer) -> list[dict]:
    """
    Prepare data for KD-SFT fusion.
    Schema 8x oversampled + v3 task data + specialized files.
    This is the SAME data that would go into SFT, but now the teacher
    will guide the student on every example.
    """
    log.info("Preparing KD-SFT fusion data...")

    all_messages = []

    # Schema data — 8x oversample (FORMAT IS THE TARGET)
    if SCHEMA_DATA.exists():
        schema = load_jsonl(SCHEMA_DATA)
        schema_msgs = [r["messages"] for r in schema if "messages" in r]
        oversample = KDSFT_CONFIG["schema_oversample"]
        all_messages.extend(schema_msgs * oversample)
        log.info(f"Schema data: {len(schema_msgs)} base x{oversample} = {len(schema_msgs) * oversample}")

    # V3 training data (task-diverse)
    train_path = DATA_DIR / "train.jsonl"
    if train_path.exists():
        train = load_jsonl(train_path)
        train_msgs = [r["messages"] for r in train if "messages" in r]
        all_messages.extend(train_msgs)
        log.info(f"V3 train: {len(train_msgs)}")

    # Specialized task files
    for fname in ["command_validate.jsonl", "evidence_check.jsonl", "next_step.jsonl",
                   "phase_classification.jsonl", "tool_output_parse.jsonl",
                   "retry_or_pivot.jsonl", "state_summary.jsonl", "postmortem.jsonl"]:
        fpath = DATA_DIR / fname
        if fpath.exists():
            records = load_jsonl(fpath)
            msgs = [r["messages"] for r in records if "messages" in r]
            all_messages.extend(msgs)
            log.info(f"  {fname}: {len(msgs)}")

    random.seed(42)
    random.shuffle(all_messages)

    kdsft_data = []
    max_len = KDSFT_CONFIG["max_seq_len"]
    for msgs in all_messages:
        text = tokenizer.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len]
        if len(token_ids) < 10:
            continue

        # Identify response tokens for masked loss (only train on assistant output)
        # Find the last assistant turn start marker
        full_text = tokenizer.decode(token_ids)
        # Use simple heuristic: train on everything (teacher guides all tokens)
        kdsft_data.append({"input_ids": token_ids, "length": len(token_ids)})

    log.info(f"KD-SFT data prepared: {len(kdsft_data)} examples, "
             f"lengths {kdsft_data[0]['length']}-{kdsft_data[-1]['length']}")
    return kdsft_data


class KDSFTTrainer:
    """
    NUCLEAR KD-SFT Fusion Trainer.

    Combines knowledge distillation and supervised fine-tuning in a single pass.
    The FP8 teacher guides the student on the exact schema-compliant outputs.

    Loss = α_kd * KL(softmax(student_logits/T), softmax(teacher_logits/T)) * T²
         + α_ce * CrossEntropy(student_logits, hard_labels)

    The T² scaling ensures gradients from soft targets are comparable
    in magnitude to hard targets regardless of temperature.

    NEFTune noise injection is applied to embeddings for better generalization.
    """

    def __init__(self, teacher, student, tokenizer, config: dict):
        self.teacher = teacher
        self.student = student
        self.tokenizer = tokenizer
        self.config = config
        self.device = next(student.parameters()).device

        self.temp_start = config["temperature_start"]
        self.temp_end = config["temperature_end"]
        self.alpha_kd = config["alpha_kd"]
        self.alpha_ce = config["alpha_ce"]
        self.neftune_alpha = config.get("neftune_noise_alpha", 0.0)

        trainable = [p for p in student.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable, lr=config["learning_rate"],
            weight_decay=config["weight_decay"], betas=(0.9, 0.999),
        )
        self.grad_accum = config["gradient_accumulation"]
        self.max_grad_norm = config["max_grad_norm"]
        self.global_step = 0
        self.total_steps = 0

        # Learning rate scheduler (cosine with warmup)
        self.warmup_ratio = config.get("warmup_ratio", 0.08)
        self._base_lr = config["learning_rate"]

    def get_lr(self) -> float:
        """Cosine schedule with linear warmup."""
        if self.total_steps == 0:
            return self._base_lr
        warmup_steps = int(self.total_steps * self.warmup_ratio)
        if self.global_step < warmup_steps:
            return self._base_lr * (self.global_step / max(warmup_steps, 1))
        progress = (self.global_step - warmup_steps) / max(self.total_steps - warmup_steps, 1)
        return self._base_lr * 0.5 * (1 + math.cos(math.pi * progress))

    def _update_lr(self):
        lr = self.get_lr()
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    def get_temperature(self) -> float:
        if self.total_steps == 0:
            return self.temp_start
        progress = min(self.global_step / self.total_steps, 1.0)
        cos_val = 0.5 * (1 + math.cos(math.pi * progress))
        return self.temp_end + (self.temp_start - self.temp_end) * cos_val

    def kd_loss(self, student_logits, teacher_logits, temperature):
        """Forward KL divergence with temperature scaling."""
        student_soft = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
        kl = F.kl_div(student_soft, teacher_soft, reduction="batchmean")
        return kl * (temperature ** 2)

    def _add_neftune_noise(self, embeddings: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        """NEFTune: add noise to embeddings proportional to sqrt(seq_len * hidden_dim)."""
        if self.neftune_alpha <= 0:
            return embeddings
        dims = embeddings.shape[-1]
        mag = self.neftune_alpha / (input_lengths.unsqueeze(1).unsqueeze(2) * dims) ** 0.5
        noise = torch.zeros_like(embeddings).uniform_(-1, 1) * mag
        return embeddings + noise

    @torch.no_grad()
    def get_teacher_logits(self, input_ids, attention_mask):
        """Teacher forward pass — frozen, no gradient tracking."""
        outputs = self.teacher(
            input_ids=input_ids, attention_mask=attention_mask,
            output_attentions=False, use_cache=False,
        )
        return outputs.logits

    def train_step(self, batch):
        """Single training step with fused KD + CE loss."""
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        temperature = self.get_temperature()

        # Teacher forward (frozen, no grad)
        teacher_logits = self.get_teacher_logits(input_ids, attention_mask)

        # Student forward with optional NEFTune noise
        if self.neftune_alpha > 0 and self.student.training:
            # Get embeddings, add noise, then forward through rest of model
            embed_layer = self.student.get_input_embeddings()
            if hasattr(self.student, 'base_model'):
                embed_layer = self.student.base_model.get_input_embeddings()
            embeds = embed_layer(input_ids)
            input_lengths = attention_mask.sum(dim=1).float()
            embeds = self._add_neftune_noise(embeds, input_lengths)
            student_outputs = self.student(
                inputs_embeds=embeds, attention_mask=attention_mask,
                output_attentions=False, use_cache=False,
            )
        else:
            student_outputs = self.student(
                input_ids=input_ids, attention_mask=attention_mask,
                output_attentions=False, use_cache=False,
            )
        student_logits = student_outputs.logits

        # Shift for next-token prediction
        shift_student = student_logits[:, :-1, :].contiguous()
        shift_teacher = teacher_logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        valid_mask = shift_labels != -100
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device), {}

        # KD loss on valid (non-padding) positions — teacher's soft distribution
        flat_student = shift_student[valid_mask]
        flat_teacher = shift_teacher[valid_mask]
        loss_kd = self.kd_loss(flat_student, flat_teacher, temperature)

        # Hard label CE loss — exact format compliance
        loss_ce = F.cross_entropy(
            shift_student.view(-1, shift_student.size(-1)),
            shift_labels.view(-1), ignore_index=-100,
        )

        # NUCLEAR: Combined loss — format AND depth in one shot
        total_loss = self.alpha_kd * loss_kd + self.alpha_ce * loss_ce
        scaled_loss = total_loss / self.grad_accum
        scaled_loss.backward()

        # Free teacher logits immediately to save VRAM
        del teacher_logits, shift_teacher, flat_teacher
        cleanup()

        metrics = {
            "loss_total": total_loss.item(),
            "loss_kd": loss_kd.item(),
            "loss_ce": loss_ce.item(),
            "temperature": temperature,
        }
        return total_loss, metrics

    def optimizer_step(self):
        """Gradient clipping + optimizer step + LR update."""
        trainable = [p for p in self.student.parameters() if p.requires_grad]
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, self.max_grad_norm)
        lr = self._update_lr()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.global_step += 1
        return grad_norm.item(), lr


def stage4_kdsft_fusion(base_dir: Path, cpt_adapter: Path) -> Path:
    """
    NUCLEAR Stage 4: KD-SFT Fusion.

    1. Download FP8 teacher to /dev/shm
    2. Load CPT-enhanced student (base + CPT merged)
    3. Train with combined KD + CE loss on schema+task data
    4. Save adapter, clean up teacher

    Returns path to KD-SFT adapter.
    """
    kdsft_final = KDSFT_OUT / "final"
    if (kdsft_final / "adapter_config.json").exists():
        log.info(f"KD-SFT adapter exists: {kdsft_final}")
        return kdsft_final

    log.info("=" * 60)
    log.info("STAGE 4: KD-SFT FUSION — THE NUCLEAR OPTION")
    log.info(f"Teacher: {FP8_TEACHER_ID} (FP8, via /dev/shm RAM)")
    log.info(f"Student: {base_dir} + CPT adapter")
    log.info("Method: KL divergence on soft logits + hard label CE (FUSED)")
    log.info(f"Schema oversample: {KDSFT_CONFIG['schema_oversample']}x")
    log.info(f"α_kd={KDSFT_CONFIG['alpha_kd']}, α_ce={KDSFT_CONFIG['alpha_ce']}")
    log.info(f"Temperature: {KDSFT_CONFIG['temperature_start']} → {KDSFT_CONFIG['temperature_end']} (cosine)")
    log.info(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
    log.info(f"VRAM: {gpu_mem()}")
    log.info("=" * 60)

    # 4a: Download FP8 teacher to /dev/shm
    _stage4_download_fp8_teacher()

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(base_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4b: Prepare KD-SFT data (schema 8x + task data)
    kdsft_data = _stage4_prepare_kdsft_data(tokenizer)

    # 4c: Load both models
    from peft import LoraConfig, get_peft_model, PeftModel
    from transformers import AutoModelForCausalLM

    # Load FP8 teacher (frozen, inference only)
    log.info("Loading FP8 teacher model (frozen)...")
    teacher = AutoModelForCausalLM.from_pretrained(
        str(FP8_TEACHER_DIR),
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    log.info(f"FP8 teacher loaded. GPU: {gpu_mem()}")

    # Load CPT-enhanced student (base + CPT adapter merged)
    log.info("Loading student (base + CPT merged, trainable LoRA)...")
    student = AutoModelForCausalLM.from_pretrained(
        str(base_dir),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    # Merge CPT adapter before adding new LoRA
    if cpt_adapter and (Path(cpt_adapter) / "adapter_config.json").exists():
        log.info(f"Merging CPT adapter: {cpt_adapter}")
        student = PeftModel.from_pretrained(student, str(cpt_adapter), torch_dtype=torch.bfloat16)
        student = student.merge_and_unload()
        log.info("CPT adapter merged into student")

    student.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # New LoRA for KD-SFT fusion (r=128 — this is the main format-learning stage)
    lora_config = LoraConfig(
        r=KDSFT_CONFIG["lora_r"],
        lora_alpha=KDSFT_CONFIG["lora_alpha"],
        lora_dropout=KDSFT_CONFIG["lora_dropout"],
        target_modules=ALL_LINEAR,
        bias="none",
        task_type="CAUSAL_LM",
    )
    student = get_peft_model(student, lora_config)
    student.print_trainable_parameters()
    log.info(f"Student loaded with LoRA r={KDSFT_CONFIG['lora_r']}. GPU: {gpu_mem()}")

    # 4d: Run KD-SFT fusion training
    trainer = KDSFTTrainer(teacher, student, tokenizer, KDSFT_CONFIG)

    batch_size = KDSFT_CONFIG["batch_size"]
    num_epochs = KDSFT_CONFIG["num_epochs"]
    steps_per_epoch = len(kdsft_data) // batch_size
    trainer.total_steps = steps_per_epoch * num_epochs
    log.info(f"KD-SFT training: {trainer.total_steps} steps "
             f"({steps_per_epoch}/epoch x {num_epochs}), "
             f"effective_batch={batch_size * KDSFT_CONFIG['gradient_accumulation']}")

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    best_loss = float("inf")
    accum_count = 0

    for epoch in range(num_epochs):
        log.info(f"\n{'='*40}")
        log.info(f"KD-SFT Epoch {epoch + 1}/{num_epochs}")
        log.info(f"{'='*40}")
        epoch_losses = []
        epoch_kd_losses = []
        epoch_ce_losses = []
        epoch_start = time.time()

        # Shuffle data each epoch
        random.shuffle(kdsft_data)

        for step in range(0, len(kdsft_data) - batch_size + 1, batch_size):
            batch_items = kdsft_data[step:step + batch_size]
            max_len = min(
                max(item["length"] for item in batch_items),
                KDSFT_CONFIG["max_seq_len"]
            )

            input_ids_batch, attention_mask_batch, labels_batch = [], [], []
            for item in batch_items:
                ids = item["input_ids"][:max_len]
                padding = max_len - len(ids)
                input_ids_batch.append(ids + [pad_id] * padding)
                attention_mask_batch.append([1] * len(ids) + [0] * padding)
                labels_batch.append(ids + [-100] * padding)

            batch = {
                "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long),
                "labels": torch.tensor(labels_batch, dtype=torch.long),
            }

            try:
                loss, metrics = trainer.train_step(batch)
                accum_count += 1

                if accum_count >= trainer.grad_accum:
                    grad_norm, lr = trainer.optimizer_step()
                    accum_count = 0
                    epoch_losses.append(metrics["loss_total"])
                    epoch_kd_losses.append(metrics["loss_kd"])
                    epoch_ce_losses.append(metrics["loss_ce"])

                    if trainer.global_step % 50 == 0:
                        avg_loss = sum(epoch_losses[-50:]) / min(len(epoch_losses), 50)
                        avg_kd = sum(epoch_kd_losses[-50:]) / min(len(epoch_kd_losses), 50)
                        avg_ce = sum(epoch_ce_losses[-50:]) / min(len(epoch_ce_losses), 50)
                        log.info(
                            f"  Step {trainer.global_step}/{trainer.total_steps} | "
                            f"Loss: {avg_loss:.4f} (KD: {avg_kd:.4f}, CE: {avg_ce:.4f}) | "
                            f"T: {metrics['temperature']:.2f} | LR: {lr:.2e} | "
                            f"GradNorm: {grad_norm:.3f} | GPU: {gpu_mem()}"
                        )

            except torch.cuda.OutOfMemoryError:
                log.warning(f"OOM at step {trainer.global_step}, seq_len={max_len}. Skipping.")
                cleanup()
                trainer.optimizer.zero_grad()
                accum_count = 0
                continue

        epoch_time = time.time() - epoch_start
        avg_epoch_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        avg_epoch_kd = sum(epoch_kd_losses) / max(len(epoch_kd_losses), 1)
        avg_epoch_ce = sum(epoch_ce_losses) / max(len(epoch_ce_losses), 1)
        log.info(
            f"KD-SFT Epoch {epoch + 1} complete: "
            f"avg_loss={avg_epoch_loss:.4f} (KD={avg_epoch_kd:.4f}, CE={avg_epoch_ce:.4f}), "
            f"time={epoch_time/60:.1f}min"
        )

        # Save best checkpoint
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            ckpt = KDSFT_OUT / f"checkpoint-epoch{epoch + 1}"
            ckpt.mkdir(parents=True, exist_ok=True)
            student.save_pretrained(str(ckpt))
            tokenizer.save_pretrained(str(ckpt))
            log.info(f"Best KD-SFT checkpoint: {ckpt} (loss={best_loss:.4f})")

    # Save final KD-SFT adapter
    kdsft_final.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(str(kdsft_final))
    tokenizer.save_pretrained(str(kdsft_final))
    log.info(f"KD-SFT adapter saved: {kdsft_final}")

    # Unload teacher
    del teacher, student, trainer
    cleanup()

    # Clean /dev/shm (free the ~33GB RAM)
    if FP8_TEACHER_DIR.exists():
        shutil.rmtree(FP8_TEACHER_DIR)
        log.info("FP8 teacher cleaned from /dev/shm")

    log.info(f"KD-SFT FUSION complete. Adapter: {kdsft_final}")
    return kdsft_final


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5: DPO — Preference Alignment (sticks LAST)
# ══════════════════════════════════════════════════════════════════════════════

def stage5_dpo(base_dir: Path, cpt_adapter: Path, kdsft_adapter: Path) -> Path:
    """DPO: preference alignment on fully-trained student. Last training stage."""
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer
    from trl import DPOTrainer, DPOConfig

    final = DPO_OUT / "final"
    if (final / "adapter_config.json").exists():
        log.info(f"DPO adapter exists: {final}")
        return final

    DPO_OUT.mkdir(parents=True, exist_ok=True)
    log.info("=" * 60)
    log.info("STAGE 5: DPO — Preference alignment (LAST training stage)")
    log.info("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(str(base_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dpo_pairs = load_jsonl(DATA_DIR / "dpo_pairs.jsonl")
    log.info(f"DPO pairs: {len(dpo_pairs)}")

    dpo_data = []
    for pair in dpo_pairs:
        prompt_messages = json.loads(pair["prompt"]) if isinstance(pair["prompt"], str) else pair["prompt"]
        prompt_str = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        dpo_data.append({
            "prompt": prompt_str,
            "chosen": pair["chosen"],
            "rejected": pair["rejected"],
        })

    split_idx = int(len(dpo_data) * 0.9)
    train_ds = Dataset.from_list(dpo_data[:split_idx])
    eval_ds = Dataset.from_list(dpo_data[split_idx:])

    # Load student: base → CPT → KD-SFT (all merged)
    log.info("Loading policy model: base → CPT → KD-SFT (merged)...")
    model = _load_and_merge(base_dir, [cpt_adapter, kdsft_adapter])
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    lora_config = LoraConfig(
        r=64, lora_alpha=128, lora_dropout=0.05,
        target_modules=ALL_LINEAR, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    log.info("Loading reference model: base → CPT → KD-SFT (merged)...")
    ref_model = _load_and_merge(base_dir, [cpt_adapter, kdsft_adapter])

    dpo_config = DPOConfig(
        output_dir=str(DPO_OUT), num_train_epochs=2,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=8, learning_rate=5e-6,
        lr_scheduler_type="cosine", warmup_ratio=0.1, weight_decay=0.01,
        max_length=2048, max_prompt_length=1024,
        logging_steps=10, save_strategy="epoch", save_total_limit=1,
        fp16=False, bf16=True, gradient_checkpointing=True,
        seed=42, report_to="none", beta=0.1,
    )

    trainer = DPOTrainer(
        model=model, ref_model=ref_model, args=dpo_config,
        train_dataset=train_ds, eval_dataset=eval_ds, processing_class=tokenizer,
    )

    log.info(f"DPO: {len(train_ds)} train, {len(eval_ds)} eval, 2 epochs")
    trainer.train()

    final.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))

    metrics = trainer.evaluate()
    log.info(f"DPO metrics: {json.dumps(metrics, indent=2)}")

    del model, ref_model, trainer
    cleanup()
    return final


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 6: Merge All & Export GGUF
# ══════════════════════════════════════════════════════════════════════════════

def stage6_merge_export(base_dir: Path, cpt_adapter: Path,
                        kdsft_adapter: Path, dpo_adapter: Path):
    """Merge all adapters (CPT → KD-SFT → DPO) on base student → GGUF."""
    log.info("=" * 60)
    log.info("STAGE 6: Merge & Export GGUF")
    log.info("=" * 60)

    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    GGUF_DIR.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    if not (MERGED_DIR / "config.json").exists():
        log.info("Loading base student + merging all adapters (CPT → KD-SFT → DPO) on CPU fp16...")
        model = _load_and_merge(
            base_dir, [cpt_adapter, kdsft_adapter, dpo_adapter],
            device_map="cpu", dtype=torch.float16,
        )
        model.save_pretrained(str(MERGED_DIR), safe_serialization=True)
        tokenizer = AutoTokenizer.from_pretrained(str(base_dir), trust_remote_code=True)
        tokenizer.save_pretrained(str(MERGED_DIR))
        del model
        cleanup()
        log.info(f"Final merged model saved: {MERGED_DIR}")
    else:
        log.info(f"Merged model exists: {MERGED_DIR}")

    # Free disk: delete intermediate outputs
    for cleanup_dir in [STUDENT_MODEL_DIR, CPT_OUT, KDSFT_OUT, DPO_OUT]:
        if cleanup_dir.exists() and cleanup_dir != MERGED_DIR:
            size = sum(f.stat().st_size for f in cleanup_dir.rglob("*") if f.is_file()) / 1e9
            log.info(f"Cleaning {cleanup_dir.name} ({size:.1f}GB) for GGUF disk space...")
            shutil.rmtree(cleanup_dir)

    # Build llama.cpp if needed
    llama_cpp = Path("/workspace/llama.cpp")
    convert_script = llama_cpp / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        log.info("Building llama.cpp for GGUF export...")
        subprocess.run(["git", "clone", "--depth=1",
                       "https://github.com/ggerganov/llama.cpp.git", str(llama_cpp)], check=True)
        subprocess.run(["pip", "install", "-r", str(llama_cpp / "requirements.txt")], check=True)

    for quant, name in [("q8_0", "ariaska-cybersec2-q8_0.gguf"), ("q6_k", "ariaska-cybersec2-q6_k.gguf")]:
        gguf_path = GGUF_DIR / name
        if gguf_path.exists():
            log.info(f"GGUF {quant} exists: {gguf_path}")
            continue

        log.info(f"Exporting GGUF {quant.upper()}...")
        cmd = [sys.executable, str(convert_script), str(MERGED_DIR),
               "--outtype", quant, "--outfile", str(gguf_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode == 0:
            size_mb = gguf_path.stat().st_size / 1e6
            log.info(f"GGUF {quant}: {gguf_path} ({size_mb:.0f} MB)")
        else:
            log.error(f"GGUF {quant} failed: {result.stderr[-500:]}")

    log.info("=" * 60)
    log.info("PIPELINE COMPLETE — ariaska-cybersec 2.0 (NUCLEAR KD-SFT)")
    log.info("Training: Data Distill → CPT → KD-SFT FUSION(FP8) → DPO → GGUF")
    log.info(f"GGUFs: {GGUF_DIR}")
    log.info("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    start = time.time()
    log.info("=" * 70)
    log.info("  ARIASKA CYBERSEC 2.0 — NUCLEAR PIPELINE (KD-SFT FUSION)")
    log.info("")
    log.info("  THE NUCLEAR OPTION: Teacher guides student on EXACT format")
    log.info("  Schema compliance target: 42% → 100%")
    log.info("")
    log.info("  Stage 1: Schema data from AWQ teacher (vLLM data generation)")
    log.info("  Stage 2: Delete AWQ, download Qwen3-4B base student")
    log.info("  Stage 3: CPT — domain knowledge (LoRA r=128)")
    log.info("  Stage 4: KD-SFT FUSION — FP8 teacher + student (NUCLEAR)")
    log.info("           Combined: KL(soft_targets) + CE(hard_labels)")
    log.info(f"           Schema {KDSFT_CONFIG['schema_oversample']}x oversampled")
    log.info(f"           α_kd={KDSFT_CONFIG['alpha_kd']}, α_ce={KDSFT_CONFIG['alpha_ce']}")
    log.info("  Stage 5: DPO — preference alignment (sticks last)")
    log.info("  Stage 6: Merge all → GGUF Q8_0 + Q6_K")
    log.info("")
    log.info(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
    log.info(f"  VRAM: {gpu_mem()}")
    log.info("=" * 70)

    # Stage 1: Data distillation from AWQ teacher
    stage1_distill_schema()

    # Stage 2: Delete AWQ teacher, get base student
    base_dir = stage2_prepare_student()

    # Stage 3: CPT on base student (domain knowledge first)
    cpt_adapter = stage3_cpt(base_dir)

    # Stage 4: NUCLEAR KD-SFT FUSION
    # FP8 teacher loaded alongside CPT-enhanced student
    # Combined KD + CE loss on schema+task data
    kdsft_adapter = stage4_kdsft_fusion(base_dir, cpt_adapter)

    # Stage 5: DPO — preference alignment (sticks last)
    dpo_adapter = stage5_dpo(base_dir, cpt_adapter, kdsft_adapter)

    # Stage 6: Merge all + GGUF export
    stage6_merge_export(base_dir, cpt_adapter, kdsft_adapter, dpo_adapter)

    elapsed = time.time() - start
    log.info(f"\nTotal pipeline time: {elapsed/3600:.1f} hours")
    log.info("Download: scp -P 19312 root@<server>:/workspace/output/gguf/ariaska-cybersec2-*.gguf .")


if __name__ == "__main__":
    main()
