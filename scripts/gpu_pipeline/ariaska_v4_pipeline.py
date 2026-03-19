#!/usr/bin/env python3
"""
ariaska_v4_pipeline.py — Complete Ariaska V4 Training Pipeline

Single-file training pipeline that:
1. Downloads Qwen3.5-4B VLM and extracts text backbone
2. Uses existing v3 training data for domain knowledge preservation
3. Generates schema-perfect data via Qwen3-32B-AWQ teacher
4. CPT → SFT (v3 + schema) → DPO → Merge → GGUF export

Preserves ALL ariaska-cybersec knowledge through the original training data
PLUS adds schema compliance through 32B teacher distillation.
"""

import gc
import json
import logging
import os
import re
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import numpy as np

log = logging.getLogger("ariaska_v4")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/workspace/pipeline.log"),
    ],
)

# ─── Paths ─────────────────────────────────────────────────────────────────
WORKSPACE = Path("/workspace")
MODELS_DIR = WORKSPACE / "models"
DATA_DIR = WORKSPACE / "data"
OUTPUT_DIR = WORKSPACE / "outputs"
GGUF_DIR = OUTPUT_DIR / "gguf"

VLM_NAME = "Qwen/Qwen3.5-4B"  # Newest VLM — we extract the text backbone
TEACHER_NAME = "Qwen/Qwen3-32B-AWQ"

TEXT_BASE_DIR = MODELS_DIR / "qwen3.5-4b-text"
TEACHER_DIR = MODELS_DIR / "qwen3-32b-awq"
V3_DATA = DATA_DIR / "v3"
SCHEMA_DATA = DATA_DIR / "ariaska_schema_sft.jsonl"

CPT_OUTPUT = OUTPUT_DIR / "cpt"
SFT_OUTPUT = OUTPUT_DIR / "sft"
DPO_OUTPUT = OUTPUT_DIR / "dpo"
MERGED_DIR = OUTPUT_DIR / "ariaska-v4-merged"


def load_jsonl(path: Path | str) -> List[dict]:
    """Load JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: List[dict], path: Path | str):
    """Save records as JSONL."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 0: Model Setup
# ═══════════════════════════════════════════════════════════════════════════

def stage0_extract_text_model() -> Path:
    """Download Qwen3.5-4B VLM and extract text-only backbone."""
    out_dir = TEXT_BASE_DIR
    if (out_dir / "config.json").exists():
        log.info(f"Text base model already at {out_dir}")
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("=" * 70)
    log.info("STAGE 0: Extract text backbone from Qwen3.5-4B VLM")
    log.info("=" * 70)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info(f"Downloading {VLM_NAME}...")
    # Qwen3.5-4B is a VLM — we need the internal language_model
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
        vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            VLM_NAME,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        text_model = vlm.language_model
        log.info(f"Extracted language_model: {type(text_model).__name__}")
    except Exception as e:
        log.warning(f"VLM extraction failed ({e}), trying direct CausalLM load...")
        text_model = AutoModelForCausalLM.from_pretrained(
            VLM_NAME,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

    params = sum(p.numel() for p in text_model.parameters())
    log.info(f"Text model: {params:,} parameters")

    text_model.save_pretrained(out_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(VLM_NAME, trust_remote_code=True)
    tokenizer.save_pretrained(out_dir)

    # Clean up VLM from memory AND disk (delete HF cache of VLM to reclaim space)
    del text_model
    if 'vlm' in dir():
        del vlm
    gc.collect()
    torch.cuda.empty_cache()

    # Delete cached VLM download to reclaim ~8GB disk
    import shutil
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    if hf_cache.exists():
        for d in hf_cache.iterdir():
            if d.is_dir() and "qwen3.5" in d.name.lower():
                sz = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e9
                shutil.rmtree(d)
                log.info(f"Cleaned VLM cache: {d.name} ({sz:.1f} GB freed)")

    log.info(f"Text base model saved: {out_dir}")
    return out_dir


def stage0_ensure_teacher() -> Path:
    """Ensure Qwen3-32B-AWQ teacher is downloaded."""
    if (TEACHER_DIR / "config.json").exists():
        log.info(f"Teacher model already at {TEACHER_DIR}")
        return TEACHER_DIR

    log.info("Downloading teacher model...")
    from huggingface_hub import snapshot_download
    snapshot_download(TEACHER_NAME, local_dir=str(TEACHER_DIR))
    log.info(f"Teacher downloaded: {TEACHER_DIR}")
    return TEACHER_DIR


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 1: Generate Schema-Perfect Data (32B Teacher)
# ═══════════════════════════════════════════════════════════════════════════

def stage1_generate_schema_data() -> Path:
    """Generate schema-perfect training data using 32B teacher via vLLM."""
    if SCHEMA_DATA.exists() and sum(1 for _ in open(SCHEMA_DATA)) >= 1000:
        count = sum(1 for _ in open(SCHEMA_DATA))
        log.info(f"Schema data already exists: {count} examples")
        return SCHEMA_DATA

    log.info("=" * 70)
    log.info("STAGE 1: Generate Schema-Perfect Training Data")
    log.info("=" * 70)

    # Use the existing generate_schema_data.py script
    cmd = [
        sys.executable, "/workspace/scripts/generate_schema_data.py",
        "--teacher-model", str(TEACHER_DIR),
        "--output", str(SCHEMA_DATA),
        "--num-examples", "12000",
    ]
    log.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)

    count = sum(1 for _ in open(SCHEMA_DATA))
    log.info(f"Generated {count} schema-perfect examples")
    return SCHEMA_DATA


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 2: Continued Pre-Training (CPT)
# ═══════════════════════════════════════════════════════════════════════════

def stage2_cpt(base_dir: Path) -> Optional[Path]:
    """CPT: Domain knowledge injection from v3 corpus."""
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTTrainer, SFTConfig

    final = CPT_OUTPUT / "final"
    if (final / "adapter_config.json").exists():
        log.info(f"CPT adapter already at {final}")
        return final

    corpus_path = V3_DATA / "cpt_corpus.jsonl"
    if not corpus_path.exists():
        log.warning("No CPT corpus found — skipping CPT stage")
        return None

    CPT_OUTPUT.mkdir(parents=True, exist_ok=True)
    log.info("=" * 70)
    log.info("STAGE 2: Continued Pre-Training (Domain Knowledge)")
    log.info("=" * 70)

    # Load FULL corpus — maximize domain knowledge retention
    docs = load_jsonl(corpus_path)
    log.info(f"CPT corpus: {len(docs)} documents (using ALL for max knowledge)")

    split_idx = int(len(docs) * 0.95)
    train_docs = docs[:split_idx]
    eval_docs = docs[split_idx:]

    train_dataset = Dataset.from_list(train_docs)
    eval_dataset = Dataset.from_list(eval_docs)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(base_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(base_dir),
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # LoRA for CPT — higher rank for deep knowledge absorption
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    sft_config = SFTConfig(
        output_dir=str(CPT_OUTPUT),
        num_train_epochs=2,          # 2 passes over full corpus for deep knowledge
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_length=768,              # Longer context for CPT knowledge
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
        seed=42,
        report_to="none",
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    log.info(f"CPT: {len(train_docs)} train, {len(eval_docs)} eval, 2 epochs (FULL corpus)")

    # Resume support
    from transformers.trainer_utils import get_last_checkpoint
    ckpt = get_last_checkpoint(str(CPT_OUTPUT)) if CPT_OUTPUT.is_dir() else None
    trainer.train(resume_from_checkpoint=ckpt)

    final.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))

    metrics = trainer.evaluate()
    log.info(f"CPT metrics: {json.dumps(metrics, indent=2)}")

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    log.info(f"CPT adapter saved: {final}")
    return final


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 3: SFT (v3 tasks + schema data combined)
# ═══════════════════════════════════════════════════════════════════════════

def stage3_sft(base_dir: Path, cpt_adapter: Optional[Path]) -> Path:
    """SFT: Combined v3 task training + schema compliance."""
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTTrainer, SFTConfig

    final = SFT_OUTPUT / "final"
    if (final / "adapter_config.json").exists():
        log.info(f"SFT adapter already at {final}")
        return final

    SFT_OUTPUT.mkdir(parents=True, exist_ok=True)
    log.info("=" * 70)
    log.info("STAGE 3: Combined SFT (v3 Tasks + Schema Compliance)")
    log.info("=" * 70)

    # ── Load and combine training data ──
    all_train = []
    all_eval = []

    # 1. v3 task training data (existing domain knowledge)
    v3_train_path = V3_DATA / "train.jsonl"
    if v3_train_path.exists():
        v3_records = load_jsonl(v3_train_path)
        v3_data = [{"messages": r["messages"]} for r in v3_records if "messages" in r]
        log.info(f"v3 train data: {len(v3_data)} examples")
        all_train.extend(v3_data)

    v3_val_path = V3_DATA / "val.jsonl"
    if v3_val_path.exists():
        v3_eval = load_jsonl(v3_val_path)
        v3_eval_data = [{"messages": r["messages"]} for r in v3_eval if "messages" in r]
        log.info(f"v3 val data: {len(v3_eval_data)} examples")
        all_eval.extend(v3_eval_data)

    # 2. v3 per-task data files
    task_files = [
        "command_validate.jsonl", "evidence_check.jsonl", "next_step.jsonl",
        "phase_classification.jsonl", "state_summary.jsonl",
        "tool_output_parse.jsonl", "postmortem.jsonl",
        "retry_or_pivot.jsonl", "retrieval_reasoning.jsonl",
    ]
    for fname in task_files:
        fpath = V3_DATA / fname
        if fpath.exists():
            records = load_jsonl(fpath)
            task_data = [{"messages": r["messages"]} for r in records if "messages" in r]
            log.info(f"  {fname}: {len(task_data)} examples")
            all_train.extend(task_data)

    # 3. Schema-perfect synthetic data (NEW — from 32B teacher)
    # CRITICAL: Oversample schema data 3x to stamp in perfect JSON compliance
    if SCHEMA_DATA.exists():
        schema_records = load_jsonl(SCHEMA_DATA)
        schema_data = [{"messages": r["messages"]} for r in schema_records if "messages" in r]
        log.info(f"Schema data: {len(schema_data)} examples (3x oversampled for schema mastery)")

        # Split schema data 95/5
        random.seed(42)
        random.shuffle(schema_data)
        split_idx = int(len(schema_data) * 0.95)
        schema_train = schema_data[:split_idx]
        schema_eval = schema_data[split_idx:]
        
        # 3x oversample schema data — model MUST master these formats
        all_train.extend(schema_train * 3)
        all_eval.extend(schema_eval)

    # Shuffle combined training data
    random.seed(42)
    random.shuffle(all_train)
    random.shuffle(all_eval)

    log.info(f"Combined SFT: {len(all_train)} train, {len(all_eval)} eval")

    train_dataset = Dataset.from_list(all_train)
    eval_dataset = Dataset.from_list(all_eval) if all_eval else None

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(base_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(base_dir),
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    # If we have CPT adapter, merge it first
    if cpt_adapter and (cpt_adapter / "adapter_config.json").exists():
        log.info(f"Merging CPT adapter from {cpt_adapter}...")
        model = PeftModel.from_pretrained(model, str(cpt_adapter))
        model = model.merge_and_unload()
        log.info("CPT adapter merged")

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # LoRA for SFT — maximum rank for perfect task + schema learning
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    sft_config = SFTConfig(
        output_dir=str(SFT_OUTPUT),
        num_train_epochs=4,              # 4 epochs for deep schema + knowledge mastery
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_length=2048,                 # Longer context for complex schemas
        packing=True,
        neftune_noise_alpha=5.0,         # NEFTune regularization
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        eval_strategy="steps",
        eval_steps=500,
        save_total_limit=3,
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
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # numpy safe globals for torch.load compat
    try:
        import numpy as _np
        torch.serialization.add_safe_globals([
            _np._core.multiarray._reconstruct,
            _np.ndarray, _np.dtype,
            _np.core.multiarray._reconstruct,
            _np.core.multiarray.scalar,
        ])
    except Exception:
        pass

    # Resume support
    from transformers.trainer_utils import get_last_checkpoint
    ckpt = get_last_checkpoint(str(SFT_OUTPUT)) if SFT_OUTPUT.is_dir() else None
    if ckpt:
        log.info(f"Resuming SFT from {ckpt}")

    log.info(f"SFT: {len(all_train)} samples, 4 epochs, r=64, 2048tok, schema 3x oversampled")
    trainer.train(resume_from_checkpoint=ckpt)

    final.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))

    metrics = trainer.evaluate()
    log.info(f"SFT metrics: {json.dumps(metrics, indent=2)}")
    with open(SFT_OUTPUT / "sft_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    log.info(f"SFT adapter saved: {final}")
    return final


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 4: DPO Alignment
# ═══════════════════════════════════════════════════════════════════════════

def stage4_dpo(base_dir: Path, sft_adapter: Path) -> Path:
    """DPO: Preference alignment with v3 + schema rejection pairs."""
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOTrainer, DPOConfig

    final = DPO_OUTPUT / "final"
    if (final / "adapter_config.json").exists():
        log.info(f"DPO adapter already at {final}")
        return final

    DPO_OUTPUT.mkdir(parents=True, exist_ok=True)
    log.info("=" * 70)
    log.info("STAGE 4: Direct Preference Optimization (DPO)")
    log.info("=" * 70)

    # ── Load DPO pairs ──
    all_dpo = []

    # v3 DPO pairs
    v3_dpo_path = V3_DATA / "dpo_pairs.jsonl"
    if v3_dpo_path.exists():
        v3_pairs = load_jsonl(v3_dpo_path)
        log.info(f"v3 DPO pairs: {len(v3_pairs)}")
        for pair in v3_pairs:
            prompt_messages = json.loads(pair["prompt"]) if isinstance(pair["prompt"], str) else pair["prompt"]
            prompt_parts = []
            for msg in prompt_messages:
                prompt_parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
            prompt_str = "\n".join(prompt_parts) + "\n<|im_start|>assistant\n"
            all_dpo.append({
                "prompt": prompt_str,
                "chosen": pair["chosen"],
                "rejected": pair["rejected"],
            })

    # Schema DPO pairs (generated during this pipeline if available)
    schema_dpo_path = DATA_DIR / "ariaska_schema_dpo.jsonl"
    if schema_dpo_path.exists():
        schema_pairs = load_jsonl(schema_dpo_path)
        log.info(f"Schema DPO pairs: {len(schema_pairs)}")
        for pair in schema_pairs:
            if isinstance(pair.get("prompt"), list):
                prompt_str = "\n".join(
                    f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>"
                    for m in pair["prompt"]
                ) + "\n<|im_start|>assistant\n"
            else:
                prompt_str = pair["prompt"]
            all_dpo.append({
                "prompt": prompt_str,
                "chosen": pair["chosen"],
                "rejected": pair["rejected"],
            })

    if not all_dpo:
        log.warning("No DPO pairs found — skipping DPO stage")
        return sft_adapter

    random.seed(42)
    random.shuffle(all_dpo)

    split_idx = int(len(all_dpo) * 0.9)
    train_dpo = all_dpo[:split_idx]
    eval_dpo = all_dpo[split_idx:]

    train_dataset = Dataset.from_list(train_dpo)
    eval_dataset = Dataset.from_list(eval_dpo)

    log.info(f"DPO: {len(train_dpo)} train, {len(eval_dpo)} eval")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(base_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model + SFT adapter
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(base_dir),
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    # Merge SFT adapter
    log.info(f"Merging SFT adapter from {sft_adapter}...")
    model = PeftModel.from_pretrained(model, str(sft_adapter))
    model = model.merge_and_unload()
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # Fresh LoRA for DPO — higher rank for stronger preference learning
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Reference model
    ref_model = AutoModelForCausalLM.from_pretrained(
        str(base_dir),
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    if (sft_adapter / "adapter_config.json").exists():
        ref_model = PeftModel.from_pretrained(ref_model, str(sft_adapter))
        ref_model = ref_model.merge_and_unload()

    dpo_config = DPOConfig(
        output_dir=str(DPO_OUTPUT),
        num_train_epochs=2,              # 2 epochs for stronger preference alignment
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_length=2048,                 # Match SFT context length
        max_prompt_length=768,           # Longer prompts for complex schemas
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
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
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    log.info(f"DPO: {len(train_dpo)} train, {len(eval_dpo)} eval, 2 epochs, r=16, 2048tok")
    trainer.train()

    final.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))

    metrics = trainer.evaluate()
    log.info(f"DPO metrics: {json.dumps(metrics, indent=2)}")

    del model, ref_model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    log.info(f"DPO adapter saved: {final}")
    return final


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 5: Merge & Export GGUF
# ═══════════════════════════════════════════════════════════════════════════

def stage5_merge_and_export(base_dir: Path, final_adapter: Path) -> Path:
    """Merge final adapter and export to GGUF Q8_0 + Q6_K."""
    if (MERGED_DIR / "config.json").exists():
        log.info(f"Merged model exists at {MERGED_DIR}")
    else:
        MERGED_DIR.mkdir(parents=True, exist_ok=True)
        log.info("=" * 70)
        log.info("STAGE 5: Merge & Export GGUF")
        log.info("=" * 70)

        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        log.info("Loading base model in FP16...")
        model = AutoModelForCausalLM.from_pretrained(
            str(base_dir),
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        log.info(f"Loading adapter from {final_adapter}...")
        model = PeftModel.from_pretrained(model, str(final_adapter))
        model = model.merge_and_unload()

        log.info(f"Saving merged model to {MERGED_DIR}...")
        model.save_pretrained(MERGED_DIR, safe_serialization=True)

        tokenizer = AutoTokenizer.from_pretrained(str(base_dir), trust_remote_code=True)
        tokenizer.save_pretrained(MERGED_DIR)

        del model
        gc.collect()

    # ── Build llama.cpp if needed ──
    GGUF_DIR.mkdir(parents=True, exist_ok=True)
    llama_dir = WORKSPACE / "llama.cpp"

    if not llama_dir.exists():
        log.info("Cloning llama.cpp...")
        subprocess.run([
            "git", "clone", "--depth=1",
            "https://github.com/ggml-org/llama.cpp.git",
            str(llama_dir),
        ], check=True)

        build_dir = llama_dir / "build"
        build_dir.mkdir(exist_ok=True)
        subprocess.run([
            "cmake", "..",
            "-DGGML_CUDA=ON",
            "-DCMAKE_BUILD_TYPE=Release",
        ], cwd=build_dir, check=True)
        subprocess.run(["cmake", "--build", ".", "-j", str(os.cpu_count())],
                      cwd=build_dir, check=True)

    # ── Convert to F16 GGUF ──
    convert_script = llama_dir / "convert_hf_to_gguf.py"
    f16_path = GGUF_DIR / "ariaska-v4-f16.gguf"

    if not f16_path.exists():
        log.info("Converting to F16 GGUF...")
        subprocess.run([
            sys.executable, str(convert_script),
            str(MERGED_DIR),
            "--outfile", str(f16_path),
            "--outtype", "f16",
        ], check=True)
        log.info(f"F16: {f16_path} ({f16_path.stat().st_size / 1e9:.2f} GB)")

    # ── Quantize ──
    quantize_bin = llama_dir / "build" / "bin" / "llama-quantize"
    if not quantize_bin.exists():
        quantize_bin = llama_dir / "build" / "llama-quantize"

    for quant_type in ["Q8_0", "Q6_K"]:
        out_path = GGUF_DIR / f"ariaska-v4-{quant_type.lower().replace('_', '-')}.gguf"
        if out_path.exists():
            log.info(f"{quant_type} already exists: {out_path}")
            continue

        log.info(f"Quantizing to {quant_type}...")
        subprocess.run([
            str(quantize_bin), str(f16_path), str(out_path), quant_type,
        ], check=True)
        log.info(f"{quant_type}: {out_path} ({out_path.stat().st_size / 1e9:.2f} GB)")

    # ── Summary ──
    log.info("\n=== GGUF FILES ===")
    for f in sorted(GGUF_DIR.glob("*.gguf")):
        log.info(f"  {f.name}: {f.stat().st_size / 1e9:.2f} GB")

    return MERGED_DIR


# ═══════════════════════════════════════════════════════════════════════════
# MAIN: Full Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def main():
    start = time.time()
    log.info("╔══════════════════════════════════════════════════════════════╗")
    log.info("║           ARIASKA V4 FULL TRAINING PIPELINE                ║")
    log.info("║  Domain Knowledge (v3) + Schema Compliance (32B Teacher)   ║")
    log.info("╚══════════════════════════════════════════════════════════════╝")

    # Verify GPU
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"GPU: {gpu} ({vram:.1f} GB)")
    else:
        log.error("No GPU available!")
        sys.exit(1)

    # ---- Disk-aware execution order ----
    # We have very limited disk (~5GB free). Teacher = 19GB is already downloaded.
    # Strategy: Run data gen with teacher FIRST, then delete teacher to free 19GB,
    # then download base model (~8GB) and train.

    # Stage 0a: Ensure teacher is present (should already be downloaded)
    teacher_dir = stage0_ensure_teacher()

    # Stage 1: Generate schema data FIRST (needs teacher in VRAM)
    schema_path = stage1_generate_schema_data()

    # Free disk: delete teacher model after data generation
    log.info("=" * 70)
    log.info("DISK CLEANUP: Removing teacher model to free ~19GB")
    log.info("=" * 70)
    import shutil
    teacher_path = MODELS_DIR / "qwen3-32b-awq"
    if teacher_path.exists():
        size_before = sum(f.stat().st_size for f in teacher_path.rglob("*") if f.is_file()) / 1e9
        shutil.rmtree(teacher_path)
        log.info(f"Deleted {teacher_path} ({size_before:.1f} GB freed)")
    # Also clear HF cache for teacher
    hf_cache = Path.home() / ".cache" / "huggingface"
    if hf_cache.exists():
        cache_size = sum(f.stat().st_size for f in hf_cache.rglob("*") if f.is_file()) / 1e9
        if cache_size > 1.0:
            shutil.rmtree(hf_cache)
            log.info(f"Cleared HF cache ({cache_size:.1f} GB freed)")
    gc.collect()

    # Stage 0b: Now download base model (plenty of disk space)
    base_dir = stage0_extract_text_model()

    # Stage 2: CPT (domain knowledge)
    cpt_adapter = stage2_cpt(base_dir)

    # Stage 3: SFT (combined v3 + schema)
    sft_adapter = stage3_sft(base_dir, cpt_adapter)

    # Stage 4: DPO
    dpo_adapter = stage4_dpo(base_dir, sft_adapter)

    # Stage 5: Merge & export
    final_adapter = dpo_adapter if (Path(str(dpo_adapter)) / "adapter_config.json").exists() else sft_adapter
    stage5_merge_and_export(base_dir, final_adapter)

    elapsed = time.time() - start
    log.info(f"\n{'='*70}")
    log.info(f"PIPELINE COMPLETE in {elapsed/3600:.1f} hours")
    log.info(f"GGUF files: {GGUF_DIR}/")
    log.info(f"Transfer: scp -P 19312 root@<server>:{GGUF_DIR}/*.gguf .")
    log.info(f"{'='*70}")


if __name__ == "__main__":
    main()
