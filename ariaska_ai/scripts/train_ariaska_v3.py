#!/usr/bin/env python3
"""Ariaska V3 Training Pipeline — CPT → SFT → DPO on 2x RTX 5070 Ti.

Three-stage training for Qwen3.5-4B:
  Stage 1: CPT (Continued Pretraining) — domain knowledge injection from 96K docs
  Stage 2: SFT (Supervised Fine-Tuning) — task-specific training on 27K samples
  Stage 3: DPO (Direct Preference Optimization) — preference alignment on 1.8K pairs

Target: 2x RTX 5070 Ti 16GB each (32GB total via FSDP/DDP)
Output: LoRA adapter → merge → GGUF Q6_K for CPU inference
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/root/ariaska_finetune/training.log", mode="a"),
    ],
)
log = logging.getLogger("ariaska.train")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path("/root/ariaska_finetune")
DATASET_DIR = BASE_DIR / "dataset" / "v3"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

VLM_NAME = "Qwen/Qwen3.5-4B"
TEXT_MODEL_DIR = MODEL_DIR / "qwen3.5-4b-text-only"

CPT_OUTPUT = OUTPUT_DIR / "cpt"
SFT_OUTPUT = OUTPUT_DIR / "sft"
DPO_OUTPUT = OUTPUT_DIR / "dpo"
FINAL_MERGED = OUTPUT_DIR / "ariaska-cybersec-merged"
GGUF_OUTPUT = OUTPUT_DIR / "gguf"


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file into list of dicts."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except json.JSONDecodeError:
                continue
    return records


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 0: Extract text-only backbone from VLM
# ══════════════════════════════════════════════════════════════════════════════

def extract_text_model() -> Path:
    """Extract text-only CausalLM from Qwen3.5 VLM."""
    out_dir = TEXT_MODEL_DIR

    if (out_dir / "config.json").exists():
        log.info(f"Text model already at {out_dir}")
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModel, AutoTokenizer

    log.info(f"Downloading VLM: {VLM_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(VLM_NAME, trust_remote_code=True)

    log.info("Loading full VLM in FP16 (CPU)...")
    vlm = AutoModel.from_pretrained(
        VLM_NAME,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    log.info("Extracting language_model backbone...")
    if hasattr(vlm, "language_model"):
        text_model = vlm.language_model
    elif hasattr(vlm, "model") and hasattr(vlm.model, "language_model"):
        text_model = vlm.model.language_model
    elif hasattr(vlm, "model"):
        text_model = vlm.model
    else:
        raise RuntimeError(f"Cannot find language_model. Attrs: {dir(vlm)}")

    param_count = sum(p.numel() for p in text_model.parameters()) / 1e9
    log.info(f"Extracted: {type(text_model).__name__} — {param_count:.2f}B params")

    text_model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)

    # Fix config for standalone CausalLM
    config_path = out_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    config["architectures"] = ["Qwen3_5ForCausalLM"]
    config["model_type"] = "qwen3_5_text"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    del vlm, text_model
    gc.collect()
    torch.cuda.empty_cache()

    log.info(f"Text model saved: {out_dir}")
    return out_dir


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1: Continued Pretraining (CPT)
# ══════════════════════════════════════════════════════════════════════════════

def run_cpt(base_model_dir: Path) -> Path:
    """CPT: domain knowledge injection from cybersecurity corpus."""
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTTrainer, SFTConfig

    output = CPT_OUTPUT
    final = output / "final"
    if (final / "adapter_config.json").exists():
        log.info(f"CPT adapter already at {final}")
        return final

    output.mkdir(parents=True, exist_ok=True)
    log.info("=" * 60)
    log.info("STAGE 1: Continued Pretraining (CPT)")
    log.info("=" * 60)

    # Load CPT corpus
    cpt_path = DATASET_DIR / "cpt_corpus.jsonl"
    cpt_docs = load_jsonl(cpt_path)
    log.info(f"CPT corpus: {len(cpt_docs)} documents")

    # Subsample if too large (keep training reasonable)
    if len(cpt_docs) > 20000:
        import random
        random.seed(42)
        random.shuffle(cpt_docs)
        cpt_docs = cpt_docs[:20000]
        log.info(f"Subsampled to {len(cpt_docs)} docs for CPT")

    # Split 95/5 for CPT
    split_idx = int(len(cpt_docs) * 0.95)
    train_docs = cpt_docs[:split_idx]
    eval_docs = cpt_docs[split_idx:]

    train_dataset = Dataset.from_list(train_docs)
    eval_dataset = Dataset.from_list(eval_docs)

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(str(base_model_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # Note: kernels-community/flash-attn2 crashes on Qwen3.5's hybrid attention
    # (chunk_gated_delta_rule + CUDA illegal memory access on Blackwell).
    # SDPA with packing is safe for CPT (independent text chunks).
    model = AutoModelForCausalLM.from_pretrained(
        str(base_model_dir),
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # LoRA for CPT — lower rank, wider target (include embeddings)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    sft_config = SFTConfig(
        output_dir=str(output),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_length=512,
        packing=True,  # safe for CPT (independent text chunks, no cross-contamination risk)
        logging_steps=50,
        save_strategy="steps",
        save_steps=25,
        eval_strategy="steps",
        eval_steps=50,
        save_total_limit=3,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=2,
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

    log.info(f"CPT: {len(train_docs)} train, {len(eval_docs)} eval, 1 epoch")
    trainer.train()

    final.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))

    metrics = trainer.evaluate()
    log.info(f"CPT final metrics: {json.dumps(metrics, indent=2)}")

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    log.info(f"CPT adapter saved: {final}")
    return final


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2: Supervised Fine-Tuning (SFT)
# ══════════════════════════════════════════════════════════════════════════════

def run_sft(base_model_dir: Path, cpt_adapter: Path | None = None) -> Path:
    """SFT: task-specific training on Ariaska dataset."""
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTTrainer, SFTConfig

    output = SFT_OUTPUT
    final = output / "final"
    if (final / "adapter_config.json").exists():
        log.info(f"SFT adapter already at {final}")
        return final

    output.mkdir(parents=True, exist_ok=True)
    log.info("=" * 60)
    log.info("STAGE 2: Supervised Fine-Tuning (SFT)")
    log.info("=" * 60)

    # Load dataset
    train_records = load_jsonl(DATASET_DIR / "train.jsonl")
    eval_records = load_jsonl(DATASET_DIR / "val.jsonl")

    # Extract just messages for SFT
    train_data = [{"messages": r["messages"]} for r in train_records if "messages" in r]
    eval_data = [{"messages": r["messages"]} for r in eval_records if "messages" in r]

    log.info(f"SFT: {len(train_data)} train, {len(eval_data)} eval")

    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(base_model_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # Note: kernels-community/flash-attn2 crashes on Qwen3.5's hybrid attention
    # (chunk_gated_delta_rule CUDA error on Blackwell GPUs).
    # SDPA + packing: the warning about cross-contamination is acceptable
    # because SFT learns task structure, not memorizing specific samples.
    # Phase 57: DDP-aware device placement.
    # With accelerate launch (LOCAL_RANK set), each process places the full model
    # on its own GPU for true data parallelism (2x throughput).
    # Fallback: single-GPU "auto" placement.
    import torch as _torch
    _local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if _local_rank >= 0:
        # DDP mode: each process loads full model on its own GPU
        _dev_map = {"": _local_rank}
        log.info(f"DDP mode: process {_local_rank}, device_map={{'':{_local_rank}}}")
    else:
        _n_gpus = _torch.cuda.device_count()
        _dev_map = "auto"
        log.info(f"Single-process mode: device_map='auto' across {_n_gpus} GPU(s)")

    model = AutoModelForCausalLM.from_pretrained(
        str(base_model_dir),
        quantization_config=bnb_config,
        device_map=_dev_map,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    # If we have CPT adapter, merge it first
    if cpt_adapter and (cpt_adapter / "adapter_config.json").exists():
        log.info(f"Merging CPT adapter from {cpt_adapter}...")
        model = PeftModel.from_pretrained(model, str(cpt_adapter))
        model = model.merge_and_unload()
        log.info("CPT adapter merged into base model")

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # LoRA for SFT — higher rank for task learning
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    sft_config = SFTConfig(
        output_dir=str(output),
        num_train_epochs=2,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_length=1024,
        packing=True,  # SDPA packing — minor cross-contamination acceptable for SFT
        neftune_noise_alpha=5.0,  # NEFTune: +5-10% quality, zero cost
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        eval_strategy="steps",
        eval_steps=200,
        save_total_limit=3,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=4,       # Phase 56: 2→4, reduces GPU stall from data loading
        dataloader_prefetch_factor=2,   # Phase 56: prefetch 2 batches per worker
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

    # Phase 56: Allowlist numpy globals for torch.load in PyTorch >= 2.10 nightly.
    # Checkpoints saved by older torch versions embed numpy arrays in optimizer.pt;
    # newer torch.load(weights_only=True) rejects them without this allowlist.
    try:
        import numpy as _np
        _safe = [
            _np._core.multiarray._reconstruct,  # type: ignore[attr-defined]
            _np.ndarray,
            _np.dtype,
            _np.core.multiarray._reconstruct,   # type: ignore[attr-defined]
            _np.core.multiarray.scalar,         # type: ignore[attr-defined]
        ]
        torch.serialization.add_safe_globals(_safe)
        log.info("Registered numpy globals for safe torch.load (PyTorch >= 2.10 nightly)")
    except Exception as _e:
        log.debug(f"numpy safe_globals registration skipped: {_e}")

    # Phase 56: Resume from last checkpoint if present (supports kill-and-restart)
    import os as _os
    from transformers.trainer_utils import get_last_checkpoint
    _last_ckpt = trainer.args.output_dir
    _resume = get_last_checkpoint(_last_ckpt) if _os.path.isdir(_last_ckpt) else None
    if _resume:
        log.info(f"Resuming SFT from checkpoint: {_resume}")
    log.info(f"SFT: {len(train_data)} samples, 2 epochs, effective batch=16")
    trainer.train(resume_from_checkpoint=_resume)

    final.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))

    metrics = trainer.evaluate()
    log.info(f"SFT final metrics: {json.dumps(metrics, indent=2)}")
    with open(output / "sft_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    log.info(f"SFT adapter saved: {final}")
    return final


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3: Direct Preference Optimization (DPO)
# ══════════════════════════════════════════════════════════════════════════════

def run_dpo(base_model_dir: Path, sft_adapter: Path) -> Path:
    """DPO: preference alignment from high/low reward pairs."""
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOTrainer, DPOConfig

    output = DPO_OUTPUT
    final = output / "final"
    if (final / "adapter_config.json").exists():
        log.info(f"DPO adapter already at {final}")
        return final

    output.mkdir(parents=True, exist_ok=True)
    log.info("=" * 60)
    log.info("STAGE 3: Direct Preference Optimization (DPO)")
    log.info("=" * 60)

    # Load DPO pairs
    dpo_pairs = load_jsonl(DATASET_DIR / "dpo_pairs.jsonl")
    log.info(f"DPO pairs: {len(dpo_pairs)}")

    # Format for DPO trainer: needs prompt, chosen, rejected
    dpo_data = []
    for pair in dpo_pairs:
        prompt_messages = json.loads(pair["prompt"])
        # Build prompt string from messages
        prompt_parts = []
        for msg in prompt_messages:
            role = msg["role"]
            content = msg["content"]
            prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        prompt_str = "\n".join(prompt_parts) + "\n<|im_start|>assistant\n"

        dpo_data.append({
            "prompt": prompt_str,
            "chosen": pair["chosen"],
            "rejected": pair["rejected"],
        })

    # Split 90/10
    split_idx = int(len(dpo_data) * 0.9)
    train_dpo = dpo_data[:split_idx]
    eval_dpo = dpo_data[split_idx:]

    train_dataset = Dataset.from_list(train_dpo)
    eval_dataset = Dataset.from_list(eval_dpo)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(base_model_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model + SFT adapter
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(base_model_dir),
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

    # Fresh LoRA for DPO — small rank
    from peft import get_peft_model
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # DPO needs a reference model — use the same base (frozen)
    ref_model = AutoModelForCausalLM.from_pretrained(
        str(base_model_dir),
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    if (sft_adapter / "adapter_config.json").exists():
        ref_model = PeftModel.from_pretrained(ref_model, str(sft_adapter))
        ref_model = ref_model.merge_and_unload()

    dpo_config = DPOConfig(
        output_dir=str(output),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_length=1024,
        max_prompt_length=512,
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
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    log.info(f"DPO: {len(train_dpo)} train, {len(eval_dpo)} eval, 1 epoch")
    trainer.train()

    final.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))

    metrics = trainer.evaluate()
    log.info(f"DPO final metrics: {json.dumps(metrics, indent=2)}")

    del model, ref_model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    log.info(f"DPO adapter saved: {final}")
    return final


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4: Merge & Export GGUF
# ══════════════════════════════════════════════════════════════════════════════

def merge_and_export(base_model_dir: Path, final_adapter: Path) -> Path:
    """Merge final adapter into base and export GGUF Q6_K."""
    merged_dir = FINAL_MERGED

    if (merged_dir / "config.json").exists():
        log.info(f"Merged model already at {merged_dir}")
        return merged_dir

    merged_dir.mkdir(parents=True, exist_ok=True)
    log.info("=" * 60)
    log.info("STAGE 4: Merge Adapter & Export")
    log.info("=" * 60)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("Loading base model in FP16...")
    model = AutoModelForCausalLM.from_pretrained(
        str(base_model_dir),
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    log.info(f"Loading adapter from {final_adapter}...")
    model = PeftModel.from_pretrained(model, str(final_adapter))
    log.info("Merging adapter into base model...")
    model = model.merge_and_unload()

    log.info(f"Saving merged model to {merged_dir}...")
    model.save_pretrained(merged_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(str(base_model_dir), trust_remote_code=True)
    tokenizer.save_pretrained(merged_dir)

    del model
    gc.collect()

    log.info(f"Merged model saved: {merged_dir}")

    # ── Auto-export GGUF Q8_0 and Q6_K ──────────────────────────────────────
    GGUF_OUTPUT.mkdir(parents=True, exist_ok=True)
    gguf_targets = [
        ("q8_0", GGUF_OUTPUT / "ariaska-cybersec-q8_0.gguf"),
        ("q6_k", GGUF_OUTPUT / "ariaska-cybersec-q6_k.gguf"),
    ]

    # Try llama-cpp-python convert first, fall back to llama.cpp CLI
    for quant_type, gguf_path in gguf_targets:
        if gguf_path.exists():
            log.info(f"GGUF {quant_type} already exists: {gguf_path}")
            continue

        log.info(f"Exporting GGUF {quant_type.upper()}...")
        try:
            import subprocess
            # Try using llama.cpp convert_hf_to_gguf.py if available
            convert_scripts = [
                "/root/llama.cpp/convert_hf_to_gguf.py",
                "/usr/local/bin/convert_hf_to_gguf.py",
            ]
            convert_script = None
            for s in convert_scripts:
                if Path(s).exists():
                    convert_script = s
                    break

            if convert_script:
                cmd = [
                    sys.executable, convert_script,
                    str(merged_dir),
                    "--outtype", quant_type,
                    "--outfile", str(gguf_path),
                ]
                log.info(f"Running: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
                if result.returncode == 0:
                    size_mb = gguf_path.stat().st_size / 1e6
                    log.info(f"GGUF {quant_type.upper()} exported: {gguf_path} ({size_mb:.0f} MB)")
                else:
                    log.warning(f"GGUF {quant_type} export failed: {result.stderr[:500]}")
            else:
                log.info(f"No convert script found. Manual export command:")
                log.info(f"  python convert_hf_to_gguf.py {merged_dir} --outtype {quant_type} --outfile {gguf_path}")
        except Exception as e:
            log.warning(f"GGUF {quant_type} export error: {e}")
            log.info(f"  Manual: python convert_hf_to_gguf.py {merged_dir} --outtype {quant_type} --outfile {gguf_path}")

    return merged_dir


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5: Fine-tune 9B Teacher + Knowledge Distillation (9B → 4B)
# ══════════════════════════════════════════════════════════════════════════════

TEACHER_VLM_NAME = "Qwen/Qwen3.5-9B"
TEACHER_BASE_DIR = MODEL_DIR / "qwen3.5-9b-text-only"
TEACHER_CPT_OUTPUT = OUTPUT_DIR / "teacher_cpt"
TEACHER_SFT_OUTPUT = OUTPUT_DIR / "teacher_sft"
TEACHER_MERGED_DIR = OUTPUT_DIR / "teacher-9b-merged"
DISTILL_OUTPUT = OUTPUT_DIR / "distill"


def extract_teacher_base() -> Path:
    """Download and extract text-only model from Qwen3.5-9B."""
    out_dir = TEACHER_BASE_DIR
    if (out_dir / "config.json").exists():
        log.info(f"9B teacher base already at {out_dir}")
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("=" * 60)
    log.info("STAGE 5a: Download 9B Teacher Base Model")
    log.info("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info(f"Downloading {TEACHER_VLM_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_VLM_NAME,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    log.info(f"Saving 9B text model to {out_dir}...")
    model.save_pretrained(out_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(TEACHER_VLM_NAME, trust_remote_code=True)
    tokenizer.save_pretrained(out_dir)

    del model
    gc.collect()
    log.info(f"9B teacher base saved: {out_dir}")
    return out_dir


def finetune_teacher(base_dir: Path) -> Path:
    """Fine-tune 9B teacher with same CPT → SFT pipeline as the 4B student.

    Uses the same dataset, same hyperparameters (adjusted for 9B size).
    The teacher must be a domain expert before distillation has any value.
    """
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTTrainer, SFTConfig

    merged = TEACHER_MERGED_DIR
    if (merged / "config.json").exists():
        log.info(f"Fine-tuned 9B teacher already at {merged}")
        return merged

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(str(base_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Stage 5b: Teacher CPT ──────────────────────────────────────────────
    cpt_final = TEACHER_CPT_OUTPUT / "final"
    if not (cpt_final / "adapter_config.json").exists():
        TEACHER_CPT_OUTPUT.mkdir(parents=True, exist_ok=True)
        log.info("=" * 60)
        log.info("STAGE 5b: Teacher CPT (9B domain injection)")
        log.info("=" * 60)

        cpt_docs = load_jsonl(DATASET_DIR / "cpt_corpus.jsonl")
        if len(cpt_docs) > 20000:
            import random
            random.seed(42)
            random.shuffle(cpt_docs)
            cpt_docs = cpt_docs[:20000]

        split_idx = int(len(cpt_docs) * 0.95)
        train_ds = Dataset.from_list(cpt_docs[:split_idx])
        eval_ds = Dataset.from_list(cpt_docs[split_idx:])

        model = AutoModelForCausalLM.from_pretrained(
            str(base_dir), quantization_config=bnb_config,
            device_map="auto", trust_remote_code=True, attn_implementation="sdpa",
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

        lora = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()

        cfg = SFTConfig(
            output_dir=str(TEACHER_CPT_OUTPUT), num_train_epochs=1,
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
            gradient_accumulation_steps=16, learning_rate=5e-5,
            lr_scheduler_type="cosine", warmup_ratio=0.05, weight_decay=0.01,
            max_length=512, packing=True, logging_steps=50,
            save_strategy="steps", save_steps=50, eval_strategy="steps",
            eval_steps=100, save_total_limit=2,
            fp16=False, bf16=True, gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataloader_num_workers=2, seed=42, report_to="none",
            dataset_text_field="text",
        )

        trainer = SFTTrainer(
            model=model, args=cfg,
            train_dataset=train_ds, eval_dataset=eval_ds,
            processing_class=tokenizer,
        )
        log.info(f"Teacher CPT: {len(train_ds)} train, {len(eval_ds)} eval")
        trainer.train()

        cpt_final.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(cpt_final))
        tokenizer.save_pretrained(str(cpt_final))
        del model, trainer
        gc.collect()
        torch.cuda.empty_cache()
    else:
        log.info(f"Teacher CPT adapter exists: {cpt_final}")

    # ── Stage 5c: Teacher SFT ──────────────────────────────────────────────
    sft_final = TEACHER_SFT_OUTPUT / "final"
    if not (sft_final / "adapter_config.json").exists():
        TEACHER_SFT_OUTPUT.mkdir(parents=True, exist_ok=True)
        log.info("=" * 60)
        log.info("STAGE 5c: Teacher SFT (9B task learning)")
        log.info("=" * 60)

        train_records = load_jsonl(DATASET_DIR / "train.jsonl")
        eval_records = load_jsonl(DATASET_DIR / "val.jsonl")
        train_data = [{"messages": r["messages"]} for r in train_records if "messages" in r]
        eval_data = [{"messages": r["messages"]} for r in eval_records if "messages" in r]

        train_ds = Dataset.from_list(train_data)
        eval_ds = Dataset.from_list(eval_data)

        model = AutoModelForCausalLM.from_pretrained(
            str(base_dir), quantization_config=bnb_config,
            device_map="auto", trust_remote_code=True, attn_implementation="sdpa",
        )

        # Merge CPT adapter first
        if (cpt_final / "adapter_config.json").exists():
            log.info("Merging teacher CPT adapter...")
            model = PeftModel.from_pretrained(model, str(cpt_final))
            model = model.merge_and_unload()

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

        lora = LoraConfig(
            r=32, lora_alpha=64, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()

        cfg = SFTConfig(
            output_dir=str(TEACHER_SFT_OUTPUT), num_train_epochs=2,
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
            gradient_accumulation_steps=16, learning_rate=2e-4,
            lr_scheduler_type="cosine", warmup_ratio=0.05, weight_decay=0.01,
            max_length=1024, packing=True,
            neftune_noise_alpha=5.0,
            logging_steps=10, save_strategy="steps", save_steps=200,
            eval_strategy="steps", eval_steps=200, save_total_limit=3,
            fp16=False, bf16=True, gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataloader_num_workers=2, seed=42, report_to="none",
            load_best_model_at_end=True, metric_for_best_model="eval_loss",
        )

        trainer = SFTTrainer(
            model=model, args=cfg,
            train_dataset=train_ds, eval_dataset=eval_ds,
            processing_class=tokenizer,
        )
        log.info(f"Teacher SFT: {len(train_data)} train, {len(eval_data)} eval, 2 epochs")
        trainer.train()

        sft_final.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(sft_final))
        tokenizer.save_pretrained(str(sft_final))
        del model, trainer
        gc.collect()
        torch.cuda.empty_cache()
    else:
        log.info(f"Teacher SFT adapter exists: {sft_final}")

    # ── Merge teacher into single model ────────────────────────────────────
    log.info("=" * 60)
    log.info("STAGE 5d: Merge Teacher (CPT+SFT → single 9B)")
    log.info("=" * 60)

    from transformers import AutoModelForCausalLM as AMLM

    model = AMLM.from_pretrained(
        str(base_dir), torch_dtype=torch.float16,
        device_map="cpu", trust_remote_code=True, low_cpu_mem_usage=True,
    )

    # Merge CPT
    if (cpt_final / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, str(cpt_final))
        model = model.merge_and_unload()
        log.info("Teacher CPT adapter merged")

    # Merge SFT
    if (sft_final / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, str(sft_final))
        model = model.merge_and_unload()
        log.info("Teacher SFT adapter merged")

    merged.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(merged, safe_serialization=True)
    tokenizer.save_pretrained(merged)
    del model
    gc.collect()
    log.info(f"Fine-tuned 9B teacher merged: {merged}")
    return merged


def run_distillation(student_dir: Path) -> Path:
    """Distill fine-tuned 9B teacher knowledge into 4B student.

    Pipeline:
      1. Fine-tune 9B with same CPT → SFT as student (if not already done)
      2. Generate teacher responses on training prompts using fine-tuned 9B
      3. Train student on teacher's expert responses
    """
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTTrainer, SFTConfig

    final = DISTILL_OUTPUT / "final"
    if (final / "adapter_config.json").exists():
        log.info(f"Distill adapter already at {final}")
        return final

    DISTILL_OUTPUT.mkdir(parents=True, exist_ok=True)
    log.info("=" * 60)
    log.info("STAGE 5: Knowledge Distillation (fine-tuned 9B → 4B)")
    log.info("=" * 60)

    # Step 0: Ensure 9B teacher is downloaded and fine-tuned
    teacher_base = extract_teacher_base()
    teacher_dir = finetune_teacher(teacher_base)
    log.info(f"Using fine-tuned 9B teacher from: {teacher_dir}")

    # Step 1: Generate teacher responses on training prompts
    teacher_responses_path = DISTILL_OUTPUT / "teacher_responses.jsonl"
    if not teacher_responses_path.exists():
        log.info("Generating fine-tuned teacher responses on training prompts...")
        tokenizer_t = AutoTokenizer.from_pretrained(str(teacher_dir), trust_remote_code=True)
        if tokenizer_t.pad_token is None:
            tokenizer_t.pad_token = tokenizer_t.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        teacher = AutoModelForCausalLM.from_pretrained(
            str(teacher_dir),
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        teacher.eval()

        # Load subset of training data for distillation
        train_records = load_jsonl(DATASET_DIR / "train.jsonl")
        import random
        random.seed(42)
        random.shuffle(train_records)
        distill_records = train_records[:10000]  # 10K samples for distillation
        log.info(f"Generating teacher responses for {len(distill_records)} samples")

        distill_data = []
        for i, rec in enumerate(distill_records):
            msgs = rec.get("messages", [])
            if not msgs:
                continue

            # Extract prompt (system + user) and let teacher generate
            input_msgs = [m for m in msgs if m["role"] != "assistant"]
            if not input_msgs:
                continue

            text = tokenizer_t.apply_chat_template(input_msgs, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer_t(text, return_tensors="pt", truncation=True, max_length=768).to(teacher.device)

            with torch.no_grad():
                outputs = teacher.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                )

            response = tokenizer_t.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            # Build distillation sample: original prompt + teacher's response
            distill_msgs = input_msgs + [{"role": "assistant", "content": response}]
            distill_data.append({"messages": distill_msgs, "task_family": rec.get("task_family", "unknown")})

            if (i + 1) % 500 == 0:
                log.info(f"  Teacher generation: {i + 1}/{len(distill_records)}")

        # Save teacher responses
        with open(teacher_responses_path, "w") as f:
            for d in distill_data:
                f.write(json.dumps(d) + "\n")
        log.info(f"Teacher responses saved: {len(distill_data)} samples")

        del teacher
        gc.collect()
        torch.cuda.empty_cache()
    else:
        log.info(f"Loading cached teacher responses from {teacher_responses_path}")
        distill_data = load_jsonl(teacher_responses_path)

    # Step 2: Fine-tune student on teacher's expert responses
    log.info("Training student on fine-tuned teacher responses...")
    train_dataset = Dataset.from_list([{"messages": d["messages"]} for d in distill_data])

    split = train_dataset.train_test_split(test_size=0.05, seed=42)
    train_ds, eval_ds = split["train"], split["test"]

    tokenizer_s = AutoTokenizer.from_pretrained(str(student_dir), trust_remote_code=True)
    if tokenizer_s.pad_token is None:
        tokenizer_s.pad_token = tokenizer_s.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    student = AutoModelForCausalLM.from_pretrained(
        str(student_dir),
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    student = prepare_model_for_kbit_training(student, use_gradient_checkpointing=True)

    # Use moderate LoRA for distillation
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    student = get_peft_model(student, lora_config)
    student.print_trainable_parameters()

    distill_config = SFTConfig(
        output_dir=str(DISTILL_OUTPUT),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_length=1024,
        packing=True,  # safe with SDPA for distillation
        neftune_noise_alpha=5.0,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        save_total_limit=2,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=2,
        seed=42,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=student,
        args=distill_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer_s,
    )

    log.info(f"Distill: {len(train_ds)} train, {len(eval_ds)} eval, 1 epoch")
    trainer.train()

    final.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final))
    tokenizer_s.save_pretrained(str(final))

    metrics = trainer.evaluate()
    log.info(f"Distillation final metrics: {json.dumps(metrics, indent=2)}")

    del student, trainer
    gc.collect()
    torch.cuda.empty_cache()

    log.info(f"Distillation adapter saved: {final}")
    return final


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Ariaska V3 Training Pipeline")
    parser.add_argument("--stage", choices=["all", "extract", "cpt", "sft", "dpo", "merge", "distill"], default="all")
    parser.add_argument("--skip-cpt", action="store_true", help="Skip CPT stage")
    parser.add_argument("--skip-dpo", action="store_true", help="Skip DPO stage")
    parser.add_argument("--distill", action="store_true", help="Run 9B→4B knowledge distillation after main pipeline")
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("ARIASKA V3 TRAINING PIPELINE")
    log.info(f"  Stage: {args.stage}")
    log.info(f"  Dataset: {DATASET_DIR}")
    log.info("=" * 70)

    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            log.info(f"  GPU {i}: {props.name} — {props.total_memory / 1e9:.1f} GB")
    else:
        log.error("No CUDA GPU found!")
        sys.exit(1)

    # Stage 0: Extract text model
    if args.stage in ("all", "extract"):
        text_model_dir = extract_text_model()
    else:
        text_model_dir = TEXT_MODEL_DIR
        if not (text_model_dir / "config.json").exists():
            log.info("Text model not found, extracting...")
            text_model_dir = extract_text_model()

    # Stage 1: CPT
    cpt_adapter = None
    if args.stage in ("all", "cpt") and not args.skip_cpt:
        cpt_adapter = run_cpt(text_model_dir)

    # Stage 2: SFT
    if args.stage in ("all", "sft"):
        sft_adapter = run_sft(text_model_dir, cpt_adapter)
    else:
        sft_adapter = SFT_OUTPUT / "final"

    # Stage 3: DPO
    if args.stage in ("all", "dpo") and not args.skip_dpo:
        final_adapter = run_dpo(text_model_dir, sft_adapter)
    else:
        final_adapter = sft_adapter

    # Stage 4: Merge
    if args.stage in ("all", "merge"):
        merged = merge_and_export(text_model_dir, final_adapter)
        log.info("=" * 70)
        log.info("TRAINING COMPLETE!")
        log.info(f"  Merged model: {merged}")
        log.info(f"  Next: Export GGUF for Ollama deployment")
        log.info("=" * 70)

    # Stage 5: Knowledge Distillation (optional, run with --distill)
    if args.stage == "distill" or args.distill:
        # Use the merged model as student base (already has CPT+SFT+DPO baked in)
        student_base = FINAL_MERGED if (FINAL_MERGED / "config.json").exists() else text_model_dir
        distill_adapter = run_distillation(student_base)

        # Re-merge with distillation adapter
        distill_merged = OUTPUT_DIR / "ariaska-cybersec-distilled"
        if not (distill_merged / "config.json").exists():
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            log.info("Merging distillation adapter into student...")
            student = AutoModelForCausalLM.from_pretrained(
                str(student_base),
                torch_dtype=torch.float16,
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            student = PeftModel.from_pretrained(student, str(distill_adapter))
            student = student.merge_and_unload()
            distill_merged.mkdir(parents=True, exist_ok=True)
            student.save_pretrained(distill_merged, safe_serialization=True)
            tokenizer = AutoTokenizer.from_pretrained(str(student_base), trust_remote_code=True)
            tokenizer.save_pretrained(distill_merged)
            del student
            gc.collect()

        log.info("=" * 70)
        log.info("DISTILLATION COMPLETE!")
        log.info(f"  Distilled model: {distill_merged}")
        log.info("=" * 70)


if __name__ == "__main__":
    main()
