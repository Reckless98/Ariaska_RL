#!/usr/bin/env python3
"""
ariaska_cybersec2_pipeline.py — Ariaska Cybersec 2.0 Full Pipeline

Teacher: Qwen3-32B-AWQ (32B, 4-bit, already on disk)
Student: Qwen3-4B (pure text, downloaded after teacher deleted)

Pipeline:
  Stage 1: Distill schema data from 32B teacher (chat template + enable_thinking=False)
  Stage 2: Delete teacher, download Qwen3-4B base
  Stage 3: CPT — domain knowledge injection (full LoRA bf16, r=128)
  Stage 4: SFT — task training + schema data 5x oversampled (full LoRA bf16, r=128)
  Stage 5: DPO — preference alignment (full LoRA bf16, r=64)
  Stage 6: Merge all adapters → GGUF Q8_0 + Q6_K

Target: 1x RTX 4090 48GB — full LoRA (no quantization), bf16
"""

from __future__ import annotations
import gc, json, logging, os, random, re, shutil, subprocess, sys, time
from pathlib import Path
from typing import Optional

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler("/workspace/training.log", mode="a")],
)
log = logging.getLogger("ariaska.pipeline")

# ── Paths ────────────────────────────────────────────────────────────────────
TEACHER_MODEL = "/workspace/models/qwen3-32b-awq"
STUDENT_MODEL_ID = "Qwen/Qwen3-4B"
STUDENT_MODEL_DIR = Path("/workspace/models/qwen3-4b")

DATA_DIR = Path("/workspace/data/v3")
SCHEMA_DATA = Path("/workspace/data/ariaska_schema_sft.jsonl")
OUTPUT = Path("/workspace/output")
CPT_OUT = OUTPUT / "cpt"
SFT_OUT = OUTPUT / "sft"
DPO_OUT = OUTPUT / "dpo"
MERGED_DIR = OUTPUT / "merged"
GGUF_DIR = OUTPUT / "gguf"

ALL_LINEAR = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


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


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1: Distill Schema Data from Teacher
# ══════════════════════════════════════════════════════════════════════════════

def stage1_distill_schema():
    """Generate schema-perfect training data using Qwen3-32B-AWQ teacher."""
    if SCHEMA_DATA.exists() and SCHEMA_DATA.stat().st_size > 1000:
        lines = sum(1 for _ in open(SCHEMA_DATA))
        if lines >= 8000:
            log.info(f"Schema data exists: {lines} lines, skipping generation")
            return
        log.info(f"Schema data only {lines} lines, regenerating...")

    log.info("=" * 60)
    log.info("STAGE 1: Distill Schema Data from 32B Teacher")
    log.info("=" * 60)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL, trust_remote_code=True)

    log.info(f"Loading teacher: {TEACHER_MODEL}")
    llm = LLM(
        model=TEACHER_MODEL,
        quantization="awq",
        max_model_len=4096,
        gpu_memory_utilization=0.88,
        trust_remote_code=True,
        dtype="float16",
    )
    log.info(f"Teacher loaded. GPU: {gpu_mem()}")

    # Schema definitions — what each prompt type should output
    SCHEMA_CONFIGS = {
        "microchain_classify": {
            "system": "You are a tactical situation classifier for Ariaska pentesting system. Respond with ONLY one word from: recon_gap, enum_needed, exploit_ready, privesc_needed, post_exploit, lateral_move, stalled",
            "max_tokens": 20,
            "count": 1500,
            "validate": lambda t: t.strip().lower().rstrip(".") in {"recon_gap", "enum_needed", "exploit_ready", "privesc_needed", "post_exploit", "lateral_move", "stalled"},
        },
        "microchain_generate": {
            "system": "You are a command generator for Ariaska pentesting system. Output ONLY a JSON array of 1-3 objects, each with these exact keys: command, template_name, reasoning, evidence_used, hypothesis, test, expected_observable, stop_condition, confidence. confidence is a float 0.0-1.0. evidence_used is a list of strings. No markdown fences.",
            "max_tokens": 500,
            "count": 2000,
            "validate": lambda t: _validate_json_array(t, ["command", "template_name", "reasoning", "evidence_used", "hypothesis", "test", "expected_observable", "stop_condition", "confidence"]),
        },
        "microchain_score": {
            "system": "You are a command scorer for Ariaska pentesting system. Output ONLY a JSON array of score objects, each with: idx (integer), phase_fit (0.0-1.0), evidence_support (0.0-1.0), novelty (0.0-1.0). No markdown.",
            "max_tokens": 200,
            "count": 1800,
            "validate": lambda t: _validate_json_array(t, ["idx", "phase_fit", "evidence_support", "novelty"]),
        },
        "microchain_fast_local": {
            "system": "You are a fast command selector for Ariaska pentesting system. Output ONLY a JSON object with: command, template_name, reasoning, score (0.0-1.0). No markdown.",
            "max_tokens": 200,
            "count": 1800,
            "validate": lambda t: _validate_json_obj(t, ["command", "template_name", "reasoning", "score"]),
        },
        "phase_guided": {
            "system": 'You are PHASE GUIDE for Ariaska_RL pentesting system. Output ONLY a raw JSON object with keys: phase_decision, anomalies, candidates, selection, distillation_packet. phase_decision must contain: chosen_phase, phase_confidence, phase_goal, stay_conditions, move_on_conditions, contradictions, phase_tag. phase_tag MUST be "P34". candidates is a list of objects with: template_name, family, why, expected_outcome, stop_condition, confidence, risk, tags. selection has: best_template_name, runner_up_template_name, selection_reason, should_escalate_to_codex, escalation_reason. distillation_packet has: observation, reasoning, action_target, expected_outcome, phase_target, confidence_target, gating_notes, phase_tag. No markdown fences.',
            "max_tokens": 800,
            "count": 1200,
            "validate": lambda t: _validate_json_obj(t, ["phase_decision", "candidates", "selection", "distillation_packet"]),
        },
        "phase_guided_fast_local": {
            "system": 'You are a fast phase advisor for Ariaska pentesting system. Output ONLY a JSON object with: stay_or_advance ("stay" or "advance"), reason, candidates (list of template name strings), confidence (0.0-1.0). No markdown.',
            "max_tokens": 200,
            "count": 1800,
            "validate": lambda t: _validate_json_obj(t, ["stay_or_advance", "reason", "candidates", "confidence"]),
        },
        "smart_mentor": {
            "system": "You are an elite pentester AI MENTOR for Ariaska system. Select the BEST next action. Output ONLY valid JSON with: intent, selected_command (MUST be a template_name like nmap_fast_scan), parameters (dict), reasoning, expected_observation, risk (low/medium/high), confidence (0.0-1.0), next_phase_hint, candidate_actions (list of dicts). No markdown.",
            "max_tokens": 400,
            "count": 1800,
            "validate": lambda t: _validate_json_obj(t, ["intent", "selected_command", "parameters", "reasoning", "expected_observation", "risk", "confidence"]),
        },
        "coherence_classify": {
            "system": "You are a state coherence classifier for Ariaska pentesting system. Output ONLY a JSON object with: phase_guess (RECON/ENUMERATION/EXPLOITATION/PRIVILEGE_ESCALATION/LATERAL_MOVEMENT/POST_EXPLOITATION/EXFILTRATION), phase_confidence (0.0-1.0), key_evidence (list), missing_evidence (list), next_best_families (list). No markdown.",
            "max_tokens": 250,
            "count": 1000,
            "validate": lambda t: _validate_json_obj(t, ["phase_guess", "phase_confidence", "key_evidence"]),
        },
        "coherence_summarize": {
            "system": "You are a state summarizer for Ariaska pentesting system. Output ONLY a JSON object with: postcard (one-line state summary string), evidence_counts (dict of category:count). No markdown.",
            "max_tokens": 200,
            "count": 1000,
            "validate": lambda t: _validate_json_obj(t, ["postcard", "evidence_counts"]),
        },
        "coherence_score": {
            "system": "You are a coherence scorer for Ariaska pentesting system. Output ONLY a JSON object with: coherence_score (0.0-1.0), novelty_score (0.0-1.0), repeat_risk (0.0-1.0), confidence_calibration (0.0-1.0). No markdown.",
            "max_tokens": 100,
            "count": 1000,
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

        if schema_type == "microchain_classify":
            return f"Classify the current tactical situation.\n\n{base}"
        elif schema_type == "microchain_generate":
            situation = random.choice(["recon_gap", "enum_needed", "exploit_ready", "privesc_needed"])
            return f"Classification: {situation}\nGenerate 1-3 command candidates.\n\n{base}"
        elif schema_type == "microchain_score":
            cands = [{"command": f"nmap -sV {target}", "template_name": templates[0]},
                     {"command": f"gobuster dir -u http://{target}", "template_name": templates[1] if len(templates)>1 else "gobuster_dir"}]
            return f"Score these candidates for phase fit:\n{json.dumps(cands)}\n\n{base}"
        elif schema_type == "microchain_fast_local":
            return f"Pick the single best next command.\n\n{base}"
        elif schema_type == "phase_guided":
            return f"Provide full phase guidance with decision, candidates, selection, and distillation packet.\n\n{base}"
        elif schema_type == "phase_guided_fast_local":
            return f"Quick phase decision: stay or advance? Give 3 candidate template names.\n\n{base}"
        elif schema_type == "smart_mentor":
            return f"Select the best next command. selected_command MUST be a template_name from available.\n\n{base}"
        elif schema_type == "coherence_classify":
            return f"Classify the current phase from this evidence.\n\n{base}"
        elif schema_type == "coherence_summarize":
            return f"Summarize the current state as a compact postcard.\n\n{base}"
        elif schema_type == "coherence_score":
            return f"Score the coherence, novelty, repeat risk, and confidence calibration.\n\n{base}"
        return base

    all_examples = []
    total_target = sum(c["count"] for c in SCHEMA_CONFIGS.values())
    log.info(f"Target: {total_target} schema examples across {len(SCHEMA_CONFIGS)} types")

    for schema_type, cfg in SCHEMA_CONFIGS.items():
        target_count = cfg["count"]
        gen_count = int(target_count * 1.5)  # Generate 1.5x for stricter quality filtering
        max_tok = cfg["max_tokens"]
        validate_fn = cfg["validate"]
        system_prompt = cfg["system"]
        # Adaptive batch size: larger batches for shorter outputs
        if max_tok <= 100:
            batch_size = 128
        elif max_tok <= 250:
            batch_size = 96
        elif max_tok <= 500:
            batch_size = 64
        else:
            batch_size = 48

        log.info(f"\n--- {schema_type}: target={target_count}, generating={gen_count} ---")

        # Build all prompts
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
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            # KEY FIX: Use chat template with thinking disabled
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            all_prompts_formatted.append(formatted)
            all_user_prompts.append(user_prompt)

        # Generate in batches
        valid_examples = []
        failed_prompts_formatted = []
        failed_user_prompts = []

        params = SamplingParams(temperature=0.4, top_p=0.9, max_tokens=max_tok,
                                repetition_penalty=1.05)

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
                # Strip any residual think tags
                text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()
                # Strip markdown fences
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

        # Batch retry failures with higher temperature
        if len(valid_examples) < target_count and failed_prompts_formatted:
            retry_count = min(len(failed_prompts_formatted), target_count - len(valid_examples) + 100)
            retry_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=max_tok,
                                         repetition_penalty=1.1)
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
                                "messages": [
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_prompt},
                                    {"role": "assistant", "content": text},
                                ],
                                "schema_type": schema_type,
                            })
                except Exception as e:
                    log.warning(f"Retry batch failed: {e}")

        # Take up to target
        final = valid_examples[:target_count]
        rate = len(valid_examples) * 100 // max(gen_count, 1)
        log.info(f"  {schema_type}: {len(final)} examples (validation rate: {rate}%)")
        all_examples.extend(final)

    # Shuffle and write
    random.shuffle(all_examples)
    SCHEMA_DATA.parent.mkdir(parents=True, exist_ok=True)
    with open(SCHEMA_DATA, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    log.info(f"\nSchema data written: {len(all_examples)} examples -> {SCHEMA_DATA}")

    # Cleanup teacher from GPU
    del llm
    cleanup()
    log.info("Teacher unloaded from GPU")


# ── Validation helpers ───────────────────────────────────────────────────────

def _clean_json_text(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(l for l in lines if not l.strip().startswith("```")).strip()
    text = re.sub(r",\s*([}\]])", r"\1", text)  # trailing commas
    return text

def _validate_json_array(text: str, required_fields: list[str]) -> bool:
    text = _clean_json_text(text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return False
    if not isinstance(parsed, list) or len(parsed) == 0:
        return False
    for item in parsed:
        if not isinstance(item, dict):
            return False
        for field in required_fields:
            if field not in item:
                return False
    return True

def _validate_json_obj(text: str, required_fields: list[str]) -> bool:
    text = _clean_json_text(text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return False
    if not isinstance(parsed, dict):
        return False
    for field in required_fields:
        if field not in parsed:
            return False
    return True


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


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2: Delete Teacher, Download Student
# ══════════════════════════════════════════════════════════════════════════════

def stage2_prepare_student():
    """Delete teacher model (free 19GB), download Qwen3-4B base."""
    # Delete teacher to free disk
    teacher_path = Path(TEACHER_MODEL)
    if teacher_path.exists():
        size = sum(f.stat().st_size for f in teacher_path.rglob("*") if f.is_file()) / 1e9
        log.info(f"Deleting teacher model ({size:.1f}GB) to free disk...")
        shutil.rmtree(teacher_path)
        log.info("Teacher deleted.")

    # Download student
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
        STUDENT_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.save_pretrained(str(STUDENT_MODEL_DIR), safe_serialization=True)
    del model
    cleanup()

    # Clean HF cache to save disk (model already saved locally)
    hf_cache = Path.home() / ".cache" / "huggingface"
    if hf_cache.exists():
        shutil.rmtree(hf_cache, ignore_errors=True)
        log.info("Cleaned HF cache to save disk")

    log.info(f"Student saved: {STUDENT_MODEL_DIR}")
    return STUDENT_MODEL_DIR


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3: CPT — Continual Pre-Training (Full LoRA bf16)
# ══════════════════════════════════════════════════════════════════════════════

def stage3_cpt(base_dir: Path) -> Path:
    """CPT: domain knowledge injection, full LoRA on bf16 model."""
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
    log.info("STAGE 3: CPT — Full LoRA bf16, r=128")
    log.info("=" * 60)

    # Load full CPT corpus
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

    # FULL LoRA: No quantization, bf16 — uses more VRAM but much better quality
    log.info("Loading model in bf16 (full precision, no quantization)...")
    model = AutoModelForCausalLM.from_pretrained(
        str(base_dir),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    lora_config = LoraConfig(
        r=128,
        lora_alpha=256,
        lora_dropout=0.05,
        target_modules=ALL_LINEAR,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    log.info(f"GPU after LoRA: {gpu_mem()}")

    sft_config = SFTConfig(
        output_dir=str(CPT_OUT),
        num_train_epochs=2,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_length=768,
        packing=True,
        logging_steps=50,
        save_strategy="steps",
        save_steps=500,
        eval_strategy="steps",
        eval_steps=500,
        save_total_limit=2,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=4,
        dataloader_prefetch_factor=2,
        seed=42,
        report_to="none",
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    log.info(f"CPT: {len(train_ds)} train, {len(eval_ds)} eval, 2 epochs, full LoRA r=128")
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
# STAGE 4: SFT — Supervised Fine-Tuning (Full LoRA bf16)
# ══════════════════════════════════════════════════════════════════════════════

def stage4_sft(base_dir: Path, cpt_adapter: Path) -> Path:
    """SFT: task training + schema data oversampled, full LoRA bf16."""
    from datasets import Dataset, concatenate_datasets
    from peft import LoraConfig, get_peft_model, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer, SFTConfig

    final = SFT_OUT / "final"
    if (final / "adapter_config.json").exists():
        log.info(f"SFT adapter exists: {final}")
        return final

    SFT_OUT.mkdir(parents=True, exist_ok=True)
    log.info("=" * 60)
    log.info("STAGE 4: SFT — Full LoRA bf16, r=128, schema 5x oversampled")
    log.info("=" * 60)

    # Load v3 SFT data
    train_records = load_jsonl(DATA_DIR / "train.jsonl")
    eval_records = load_jsonl(DATA_DIR / "val.jsonl")
    train_data = [{"messages": r["messages"]} for r in train_records if "messages" in r]
    eval_data = [{"messages": r["messages"]} for r in eval_records if "messages" in r]

    # Load schema data and oversample 5x
    schema_records = load_jsonl(SCHEMA_DATA) if SCHEMA_DATA.exists() else []
    schema_data = [{"messages": r["messages"]} for r in schema_records if "messages" in r]

    log.info(f"V3 train: {len(train_data)}, V3 eval: {len(eval_data)}, Schema: {len(schema_data)}")

    # 8x oversample schema for maximum schema compliance (was 42% at 5x)
    schema_oversampled = schema_data * 8
    random.seed(42)
    random.shuffle(schema_oversampled)

    combined_train = train_data + schema_oversampled
    random.shuffle(combined_train)
    log.info(f"Combined SFT train: {len(combined_train)} ({len(train_data)} v3 + {len(schema_oversampled)} schema 5x)")

    tokenizer = AutoTokenizer.from_pretrained(str(base_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Format with chat template (enable_thinking=False avoids <think> tokens in training)
    def _fmt(msgs):
        return tokenizer.apply_chat_template(msgs, tokenize=False, enable_thinking=False)

    train_formatted = [{"text": _fmt(item["messages"])} for item in combined_train]
    eval_formatted = [{"text": _fmt(item["messages"])} for item in eval_data]

    train_ds = Dataset.from_list(train_formatted)
    eval_ds = Dataset.from_list(eval_formatted)

    # Load base model in bf16, merge CPT adapter
    log.info("Loading model in bf16...")
    model = AutoModelForCausalLM.from_pretrained(
        str(base_dir),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    if cpt_adapter and (cpt_adapter / "adapter_config.json").exists():
        log.info(f"Merging CPT adapter: {cpt_adapter}")
        model = PeftModel.from_pretrained(model, str(cpt_adapter), torch_dtype=torch.bfloat16)
        model = model.merge_and_unload()
        log.info("CPT adapter merged")

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    lora_config = LoraConfig(
        r=128,
        lora_alpha=256,
        lora_dropout=0.05,
        target_modules=ALL_LINEAR,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    log.info(f"GPU after LoRA: {gpu_mem()}")

    sft_config = SFTConfig(
        output_dir=str(SFT_OUT),
        num_train_epochs=4,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_length=2048,
        packing=True,
        neftune_noise_alpha=5.0,
        logging_steps=25,
        save_strategy="steps",
        save_steps=500,
        eval_strategy="steps",
        eval_steps=500,
        save_total_limit=2,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=4,
        dataloader_prefetch_factor=2,
        seed=42,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    # Resume from checkpoint if available
    from transformers.trainer_utils import get_last_checkpoint
    resume = get_last_checkpoint(str(SFT_OUT)) if SFT_OUT.exists() else None
    if resume:
        log.info(f"Resuming SFT from: {resume}")

    log.info(f"SFT: {len(combined_train)} train, {len(eval_data)} eval, 4 epochs")
    trainer.train(resume_from_checkpoint=resume)

    final.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))

    metrics = trainer.evaluate()
    log.info(f"SFT metrics: {json.dumps(metrics, indent=2)}")

    del model, trainer
    cleanup()
    return final


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5: DPO — Direct Preference Optimization (Full LoRA bf16)
# ══════════════════════════════════════════════════════════════════════════════

def stage5_dpo(base_dir: Path, cpt_adapter: Path, sft_adapter: Path) -> Path:
    """DPO: preference alignment, full LoRA bf16."""
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
    log.info("STAGE 5: DPO — Full LoRA bf16, r=64")
    log.info("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(str(base_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load DPO pairs — use tokenizer for safe prompt formatting
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

    # Load base + merge CPT + SFT adapters (full chain required)
    log.info("Loading policy model: base → CPT → SFT...")
    model = _load_and_merge(base_dir, [cpt_adapter, sft_adapter])
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=ALL_LINEAR,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Reference model (base + CPT + SFT merged, frozen)
    log.info("Loading reference model: base → CPT → SFT...")
    ref_model = _load_and_merge(base_dir, [cpt_adapter, sft_adapter])

    dpo_config = DPOConfig(
        output_dir=str(DPO_OUT),
        num_train_epochs=2,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_length=2048,
        max_prompt_length=1024,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        seed=42,
        report_to="none",
        beta=0.1,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
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
# STAGE 6: Merge & Export GGUF
# ══════════════════════════════════════════════════════════════════════════════

def stage6_merge_export(base_dir: Path, cpt_adapter: Path, sft_adapter: Path, dpo_adapter: Path):
    """Merge all adapters (CPT → SFT → DPO) and export GGUF Q8_0 + Q6_K."""
    log.info("=" * 60)
    log.info("STAGE 6: Merge & Export GGUF")
    log.info("=" * 60)

    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    GGUF_DIR.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    if not (MERGED_DIR / "config.json").exists():
        log.info("Loading base + merging all adapters (CPT → SFT → DPO) on CPU fp16...")
        model = _load_and_merge(
            base_dir, [cpt_adapter, sft_adapter, dpo_adapter],
            device_map="cpu", dtype=torch.float16,
        )

        model.save_pretrained(str(MERGED_DIR), safe_serialization=True)
        tokenizer = AutoTokenizer.from_pretrained(str(base_dir), trust_remote_code=True)
        tokenizer.save_pretrained(str(MERGED_DIR))

        del model
        cleanup()
        log.info(f"Merged model saved: {MERGED_DIR}")
    else:
        log.info(f"Merged model exists: {MERGED_DIR}")

    # Build llama.cpp if needed
    llama_cpp = Path("/workspace/llama.cpp")
    convert_script = llama_cpp / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        log.info("Building llama.cpp for GGUF export...")
        subprocess.run(["git", "clone", "--depth=1", "https://github.com/ggerganov/llama.cpp.git",
                       str(llama_cpp)], check=True)
        subprocess.run(["pip", "install", "-r", str(llama_cpp / "requirements.txt")], check=True)

    # Export Q8_0 and Q6_K
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
    log.info("PIPELINE COMPLETE — ariaska-cybersec 2.0")
    log.info(f"GGUFs: {GGUF_DIR}")
    log.info("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    start = time.time()
    log.info("=" * 60)
    log.info("ARIASKA CYBERSEC 2.0 — Full Pipeline")
    log.info(f"Teacher: Qwen3-32B-AWQ (distillation)")
    log.info(f"Student: Qwen3-4B (full LoRA bf16)")
    log.info(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
    log.info(f"VRAM: {gpu_mem()}")
    log.info("=" * 60)

    # Stage 1: Distill schema data from teacher
    stage1_distill_schema()

    # Stage 2: Delete teacher, get student
    base_dir = stage2_prepare_student()

    # Stage 3: CPT
    cpt_adapter = stage3_cpt(base_dir)

    # Stage 4: SFT
    sft_adapter = stage4_sft(base_dir, cpt_adapter)

    # Stage 5: DPO
    dpo_adapter = stage5_dpo(base_dir, cpt_adapter, sft_adapter)

    # Stage 6: Merge + GGUF
    stage6_merge_export(base_dir, cpt_adapter, sft_adapter, dpo_adapter)

    elapsed = time.time() - start
    log.info(f"\nTotal time: {elapsed/3600:.1f} hours")


if __name__ == "__main__":
    main()
