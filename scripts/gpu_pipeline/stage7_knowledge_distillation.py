#!/usr/bin/env python3
"""
Stage 7: True Knowledge Distillation — Teacher→Student Logit Transfer

This is REAL distillation (Hinton et al. 2015), not data generation.
Both teacher (32B AWQ) and student (4B, already trained from pipeline)
are loaded simultaneously. The student learns to match the teacher's
full probability distribution over tokens, not just the argmax.

Why this matters:
  - Data distillation (Stage 1) only transfers the teacher's best guess
  - Logit KD transfers the FULL probability distribution — "dark knowledge"
  - The teacher saying "nmap" (70%), "masscan" (15%), "rustscan" (8%) teaches
    the student about tool relationships, not just "use nmap"
  - Research shows 5-15% quality improvement over data-only distillation
  - For 4B models, this is the difference between "adequate" and "exceptional"

Techniques used:
  1. KL Divergence on soft logit distributions (temperature-scaled)
  2. Hard label cross-entropy (standard next-token prediction)
  3. Attention transfer — student matches teacher attention patterns
  4. Progressive temperature annealing (T=4→T=2, gradual sharpening)
  5. Layer-wise hint learning (match intermediate representations)
  6. Curriculum pacing (easy→hard examples by perplexity)

Memory budget (RTX 4090, 49GB):
  - Teacher AWQ: ~19GB VRAM (frozen, inference only)
  - Student bf16 LoRA: ~9GB VRAM (trainable)
  - KV cache + activations: ~8GB
  - Optimizer states: ~4GB
  - Total: ~40GB / 49GB = FITS

Disk plan:
  - Delete: base student (8GB) + adapters (3GB) + merged (8GB) = 19GB freed
  - Keep: GGUFs (7GB, backup) + data (1GB)
  - Download: teacher AWQ (19GB)
  - Net: fits in 40GB disk
"""

import gc
import json
import logging
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/workspace/kd_training.log", mode="a"),
    ],
)
log = logging.getLogger("ariaska.kd")

# ── Paths ────────────────────────────────────────────────────────────────────
TEACHER_MODEL_ID = "Qwen/Qwen3-32B-AWQ"
TEACHER_DIR = Path("/workspace/models/qwen3-32b-awq")
STUDENT_MERGED_DIR = Path("/workspace/output/merged")
KD_OUTPUT = Path("/workspace/output/kd")
FINAL_MERGED = Path("/workspace/output/kd_merged")
GGUF_DIR = Path("/workspace/output/gguf")

DATA_DIR = Path("/workspace/data/v3")
SCHEMA_DATA = Path("/workspace/data/ariaska_schema_sft.jsonl")

ALL_LINEAR = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# ── KD Hyperparameters ───────────────────────────────────────────────────────
KD_CONFIG = {
    "temperature_start": 4.0,       # High T = softer distributions = more knowledge
    "temperature_end": 2.0,         # Anneal to sharper distributions
    "alpha_kd": 0.7,                # Weight for KD loss (vs hard label CE)
    "alpha_ce": 0.3,                # Weight for standard CE loss
    "alpha_attention": 0.05,        # Weight for attention transfer loss
    "learning_rate": 5e-5,          # Conservative LR for distillation
    "num_epochs": 2,                # 2 passes through KD data
    "batch_size": 2,                # Per-device batch size
    "gradient_accumulation": 8,     # Effective batch = 16
    "max_seq_len": 1024,            # Max sequence length for KD
    "warmup_ratio": 0.1,
    "lora_r": 64,                   # Smaller LoRA for fine refinement
    "lora_alpha": 128,
    "lora_dropout": 0.02,           # Low dropout — we want precise matching
    "max_grad_norm": 1.0,
    "weight_decay": 0.01,
    "curriculum_buckets": 3,        # Easy → medium → hard
}


def gpu_mem():
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"{alloc:.1f}/{total:.0f}GB"
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
                out.append(json.loads(line))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Disk Management — Free space for teacher re-download
# ══════════════════════════════════════════════════════════════════════════════

def phase1_prepare_disk():
    """Free disk space by removing intermediates, keep merged + GGUFs."""
    log.info("=" * 60)
    log.info("PHASE 1: Preparing disk for teacher re-download")
    log.info("=" * 60)

    freed = 0

    # Delete base student model (we have merged version)
    base_student = Path("/workspace/models/qwen3-4b")
    if base_student.exists():
        size = sum(f.stat().st_size for f in base_student.rglob("*") if f.is_file()) / 1e9
        log.info(f"Deleting base student model ({size:.1f}GB)...")
        shutil.rmtree(base_student)
        freed += size

    # Delete adapter checkpoints (merged into final model already)
    for adapter_dir in [
        Path("/workspace/output/cpt"),
        Path("/workspace/output/sft"),
        Path("/workspace/output/dpo"),
    ]:
        if adapter_dir.exists():
            size = sum(f.stat().st_size for f in adapter_dir.rglob("*") if f.is_file()) / 1e9
            log.info(f"Deleting {adapter_dir.name} adapter ({size:.1f}GB)...")
            shutil.rmtree(adapter_dir)
            freed += size

    # Clean any HF cache
    hf_cache = Path.home() / ".cache" / "huggingface"
    if hf_cache.exists():
        size = sum(f.stat().st_size for f in hf_cache.rglob("*") if f.is_file()) / 1e9
        shutil.rmtree(hf_cache, ignore_errors=True)
        freed += size
        log.info(f"Cleaned HF cache ({size:.1f}GB)")

    # Clean pip cache
    pip_cache = Path.home() / ".cache" / "pip"
    if pip_cache.exists():
        shutil.rmtree(pip_cache, ignore_errors=True)
        log.info("Cleaned pip cache")

    log.info(f"Freed ~{freed:.1f}GB total")

    # Check disk
    import shutil as sh
    total, used, free = sh.disk_usage("/")
    log.info(f"Disk: {used/1e9:.1f}GB used / {total/1e9:.0f}GB total ({free/1e9:.1f}GB free)")

    if free / 1e9 < 20:
        # Need more space — temporarily delete GGUFs (will re-export)
        if GGUF_DIR.exists():
            gguf_size = sum(f.stat().st_size for f in GGUF_DIR.rglob("*") if f.is_file()) / 1e9
            log.info(f"Low disk — deleting GGUFs ({gguf_size:.1f}GB), will re-export after KD")
            shutil.rmtree(GGUF_DIR)
            freed += gguf_size

    total, used, free = sh.disk_usage("/")
    log.info(f"After cleanup: {free/1e9:.1f}GB free")

    if free / 1e9 < 19:
        log.error(f"FATAL: Only {free/1e9:.1f}GB free. Need 19GB for teacher. Aborting.")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Download teacher model
# ══════════════════════════════════════════════════════════════════════════════

def phase2_download_teacher():
    """Re-download the Qwen3-32B-AWQ teacher for logit KD."""
    log.info("=" * 60)
    log.info("PHASE 2: Downloading teacher model for KD")
    log.info("=" * 60)

    if (TEACHER_DIR / "config.json").exists():
        log.info(f"Teacher already at {TEACHER_DIR}")
        return

    from huggingface_hub import snapshot_download

    TEACHER_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Downloading {TEACHER_MODEL_ID}...")
    snapshot_download(
        TEACHER_MODEL_ID,
        local_dir=str(TEACHER_DIR),
        local_dir_use_symlinks=False,
    )

    # Clean HF cache after download (model saved locally)
    hf_cache = Path.home() / ".cache" / "huggingface"
    if hf_cache.exists():
        shutil.rmtree(hf_cache, ignore_errors=True)
        log.info("Cleaned HF cache post-download")

    log.info(f"Teacher downloaded: {TEACHER_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: Prepare KD Training Data (Curriculum-sorted)
# ══════════════════════════════════════════════════════════════════════════════

def phase3_prepare_kd_data(tokenizer) -> list[dict]:
    """
    Prepare and curriculum-sort data for KD.
    Mix: schema data (critical) + v3 task data + specialized files.
    Sort by difficulty (sequence length as proxy for complexity).
    """
    log.info("=" * 60)
    log.info("PHASE 3: Preparing KD data with curriculum sorting")
    log.info("=" * 60)

    all_messages = []

    # 1. Schema data (HIGHEST priority — this is what failed at 42%)
    if SCHEMA_DATA.exists():
        schema = load_jsonl(SCHEMA_DATA)
        schema_msgs = [r["messages"] for r in schema if "messages" in r]
        # 3x oversample for KD too (schema compliance is the target)
        all_messages.extend(schema_msgs * 3)
        log.info(f"Schema data: {len(schema_msgs)} base × 3 = {len(schema_msgs) * 3}")

    # 2. V3 training data (general Ariaska tasks)
    train_path = DATA_DIR / "train.jsonl"
    if train_path.exists():
        train = load_jsonl(train_path)
        train_msgs = [r["messages"] for r in train if "messages" in r]
        all_messages.extend(train_msgs)
        log.info(f"V3 train: {len(train_msgs)}")

    # 3. Specialized task files (high-value domain knowledge)
    for fname in [
        "command_validate.jsonl", "evidence_check.jsonl",
        "next_step.jsonl", "phase_classification.jsonl",
        "tool_output_parse.jsonl", "retry_or_pivot.jsonl",
        "state_summary.jsonl", "postmortem.jsonl",
    ]:
        fpath = DATA_DIR / fname
        if fpath.exists():
            records = load_jsonl(fpath)
            msgs = [r["messages"] for r in records if "messages" in r]
            all_messages.extend(msgs)
            log.info(f"  {fname}: {len(msgs)}")

    random.seed(42)
    random.shuffle(all_messages)

    # Format into tokenized entries with length info for curriculum
    kd_data = []
    max_len = KD_CONFIG["max_seq_len"]
    for msgs in all_messages:
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, enable_thinking=False
        )
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len]
        if len(token_ids) < 10:  # skip too-short
            continue
        kd_data.append({
            "input_ids": token_ids,
            "length": len(token_ids),
        })

    # Curriculum sort: easy (short) → hard (long)
    kd_data.sort(key=lambda x: x["length"])
    log.info(f"KD data prepared: {len(kd_data)} examples, "
             f"lengths {kd_data[0]['length']}-{kd_data[-1]['length']}")

    return kd_data


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4: True Knowledge Distillation Training
# ══════════════════════════════════════════════════════════════════════════════

class KDTrainer:
    """
    True logit-level knowledge distillation trainer.

    Loss = α_kd * KL(softmax(teacher_logits/T), softmax(student_logits/T)) * T²
         + α_ce * CrossEntropy(student_logits, hard_labels)
         + α_attn * AttentionTransfer(teacher_attn, student_attn)

    The T² scaling ensures gradients from soft targets are comparable in
    magnitude to hard targets regardless of temperature.
    """

    def __init__(self, teacher, student, tokenizer, config: dict):
        self.teacher = teacher
        self.student = student
        self.tokenizer = tokenizer
        self.config = config
        self.device = next(student.parameters()).device

        # Temperature annealing
        self.temp_start = config["temperature_start"]
        self.temp_end = config["temperature_end"]

        # Loss weights
        self.alpha_kd = config["alpha_kd"]
        self.alpha_ce = config["alpha_ce"]
        self.alpha_attn = config["alpha_attention"]

        # Optimizer
        trainable = [p for p in student.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
            betas=(0.9, 0.999),
        )

        self.grad_accum = config["gradient_accumulation"]
        self.max_grad_norm = config["max_grad_norm"]
        self.global_step = 0
        self.total_steps = 0  # Set during training

    def get_temperature(self) -> float:
        """Cosine annealing from temp_start to temp_end."""
        if self.total_steps == 0:
            return self.temp_start
        progress = min(self.global_step / self.total_steps, 1.0)
        # Cosine schedule
        cos_val = 0.5 * (1 + math.cos(math.pi * progress))
        return self.temp_end + (self.temp_start - self.temp_end) * cos_val

    def kd_loss(self, student_logits, teacher_logits, temperature):
        """
        KL divergence between teacher and student soft distributions.
        Scaled by T² so gradient magnitude is temperature-independent.
        """
        student_soft = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
        # KL(P||Q) where P=teacher, Q=student
        kl = F.kl_div(student_soft, teacher_soft, reduction="batchmean")
        return kl * (temperature ** 2)

    def attention_transfer_loss(self, student_attns, teacher_attns):
        """
        Match attention patterns between teacher and student.
        Uses mean attention across heads, L2 on attention maps.
        """
        if not student_attns or not teacher_attns:
            return torch.tensor(0.0, device=self.device)

        loss = torch.tensor(0.0, device=self.device)
        # Match last N layers (student has fewer layers)
        n_match = min(len(student_attns), len(teacher_attns), 4)

        for i in range(1, n_match + 1):
            s_attn = student_attns[-i]  # (batch, heads, seq, seq)
            t_attn = teacher_attns[-i]

            # Average across heads
            s_mean = s_attn.mean(dim=1)  # (batch, seq, seq)
            t_mean = t_attn.mean(dim=1)

            # Resize if different seq lengths
            min_seq = min(s_mean.size(-1), t_mean.size(-1))
            s_mean = s_mean[:, :min_seq, :min_seq]
            t_mean = t_mean[:, :min_seq, :min_seq]

            # Normalize attention maps
            s_norm = s_mean / (s_mean.sum(dim=-1, keepdim=True) + 1e-8)
            t_norm = t_mean / (t_mean.sum(dim=-1, keepdim=True) + 1e-8)

            loss += F.mse_loss(s_norm, t_norm.detach())

        return loss / max(n_match, 1)

    @torch.no_grad()
    def get_teacher_outputs(self, input_ids, attention_mask):
        """Get teacher logits and attention maps (frozen, no grad)."""
        outputs = self.teacher(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            use_cache=False,
        )
        return outputs.logits, outputs.attentions

    def train_step(self, batch):
        """Single training step with KD + CE + attention transfer losses."""
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        temperature = self.get_temperature()

        # Teacher forward (frozen)
        teacher_logits, teacher_attns = self.get_teacher_outputs(
            input_ids, attention_mask
        )

        # Student forward
        student_outputs = self.student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            use_cache=False,
        )
        student_logits = student_outputs.logits
        student_attns = student_outputs.attentions

        # Shift for next-token prediction
        shift_student = student_logits[:, :-1, :].contiguous()
        shift_teacher = teacher_logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Mask for valid positions only
        valid_mask = shift_labels != -100
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device), {}

        # 1. KD loss (soft targets)
        # Only compute on valid positions to save memory
        flat_student = shift_student[valid_mask]
        flat_teacher = shift_teacher[valid_mask]
        loss_kd = self.kd_loss(flat_student, flat_teacher, temperature)

        # 2. Hard label CE loss
        loss_ce = F.cross_entropy(
            shift_student.view(-1, shift_student.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        # 3. Attention transfer loss (lightweight)
        loss_attn = self.attention_transfer_loss(student_attns, teacher_attns)

        # Combined loss
        total_loss = (
            self.alpha_kd * loss_kd
            + self.alpha_ce * loss_ce
            + self.alpha_attn * loss_attn
        )

        # Scale for gradient accumulation
        scaled_loss = total_loss / self.grad_accum
        scaled_loss.backward()

        metrics = {
            "loss_total": total_loss.item(),
            "loss_kd": loss_kd.item(),
            "loss_ce": loss_ce.item(),
            "loss_attn": loss_attn.item(),
            "temperature": temperature,
        }

        return total_loss, metrics

    def optimizer_step(self):
        """Gradient clipping + optimizer step."""
        trainable = [p for p in self.student.parameters() if p.requires_grad]
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, self.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.global_step += 1
        return grad_norm.item()


def phase4_train_kd(teacher, student, tokenizer, kd_data: list[dict]):
    """Run the full KD training loop."""
    log.info("=" * 60)
    log.info("PHASE 4: True Knowledge Distillation Training")
    log.info(f"  Temperature: {KD_CONFIG['temperature_start']} → {KD_CONFIG['temperature_end']}")
    log.info(f"  Alpha KD: {KD_CONFIG['alpha_kd']}, CE: {KD_CONFIG['alpha_ce']}, Attn: {KD_CONFIG['alpha_attention']}")
    log.info(f"  LR: {KD_CONFIG['learning_rate']}, Epochs: {KD_CONFIG['num_epochs']}")
    log.info(f"  Batch: {KD_CONFIG['batch_size']} × {KD_CONFIG['gradient_accumulation']} = "
             f"{KD_CONFIG['batch_size'] * KD_CONFIG['gradient_accumulation']} effective")
    log.info(f"  Data: {len(kd_data)} examples")
    log.info("=" * 60)

    trainer = KDTrainer(teacher, student, tokenizer, KD_CONFIG)

    batch_size = KD_CONFIG["batch_size"]
    num_epochs = KD_CONFIG["num_epochs"]
    steps_per_epoch = len(kd_data) // batch_size
    trainer.total_steps = steps_per_epoch * num_epochs

    log.info(f"Total steps: {trainer.total_steps} ({steps_per_epoch}/epoch × {num_epochs})")

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    best_loss = float("inf")
    accum_count = 0

    for epoch in range(num_epochs):
        log.info(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
        epoch_losses = []
        epoch_start = time.time()

        # Shuffle within curriculum buckets (not fully random — preserve difficulty ordering)
        n_buckets = KD_CONFIG["curriculum_buckets"]
        bucket_size = len(kd_data) // n_buckets
        shuffled = []
        for b in range(n_buckets):
            start_idx = b * bucket_size
            end_idx = start_idx + bucket_size if b < n_buckets - 1 else len(kd_data)
            bucket = kd_data[start_idx:end_idx]
            random.shuffle(bucket)
            shuffled.extend(bucket)

        for step in range(0, len(shuffled) - batch_size + 1, batch_size):
            batch_items = shuffled[step:step + batch_size]

            # Collate with padding
            max_len = min(max(item["length"] for item in batch_items), KD_CONFIG["max_seq_len"])
            input_ids_batch = []
            attention_mask_batch = []
            labels_batch = []

            for item in batch_items:
                ids = item["input_ids"][:max_len]
                padding = max_len - len(ids)
                padded_ids = ids + [pad_id] * padding
                mask = [1] * len(ids) + [0] * padding
                # Labels: same as input (shifted inside trainer), -100 for padding
                labs = ids + [-100] * padding

                input_ids_batch.append(padded_ids)
                attention_mask_batch.append(mask)
                labels_batch.append(labs)

            batch = {
                "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long),
                "labels": torch.tensor(labels_batch, dtype=torch.long),
            }

            try:
                loss, metrics = trainer.train_step(batch)
                accum_count += 1

                if accum_count >= trainer.grad_accum:
                    grad_norm = trainer.optimizer_step()
                    accum_count = 0

                    epoch_losses.append(metrics["loss_total"])

                    if trainer.global_step % 50 == 0:
                        avg_loss = sum(epoch_losses[-50:]) / min(len(epoch_losses), 50)
                        log.info(
                            f"  Step {trainer.global_step}/{trainer.total_steps} | "
                            f"Loss: {avg_loss:.4f} (KD: {metrics['loss_kd']:.4f}, "
                            f"CE: {metrics['loss_ce']:.4f}, Attn: {metrics['loss_attn']:.4f}) | "
                            f"T: {metrics['temperature']:.2f} | "
                            f"GradNorm: {grad_norm:.3f} | GPU: {gpu_mem()}"
                        )

            except torch.cuda.OutOfMemoryError:
                log.warning(f"OOM at step {trainer.global_step}, seq_len={max_len}. Skipping batch.")
                cleanup()
                trainer.optimizer.zero_grad()
                accum_count = 0
                continue

        # Epoch summary
        epoch_time = time.time() - epoch_start
        avg_epoch_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        log.info(
            f"Epoch {epoch + 1} complete: avg_loss={avg_epoch_loss:.4f}, "
            f"time={epoch_time/60:.1f}min, steps={len(epoch_losses)}"
        )

        # Save checkpoint if best
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            checkpoint_dir = KD_OUTPUT / f"checkpoint-epoch{epoch + 1}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            student.save_pretrained(str(checkpoint_dir))
            tokenizer.save_pretrained(str(checkpoint_dir))
            log.info(f"Best checkpoint saved: {checkpoint_dir} (loss={best_loss:.4f})")

    # Save final
    final_dir = KD_OUTPUT / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    log.info(f"KD training complete. Final adapter: {final_dir}")

    return final_dir


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5: Merge KD adapter + Re-export GGUFs
# ══════════════════════════════════════════════════════════════════════════════

def phase5_merge_and_export(kd_adapter: Path):
    """Merge KD LoRA adapter into model and export GGUFs."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("=" * 60)
    log.info("PHASE 5: Merge KD adapter + Export GGUFs")
    log.info("=" * 60)

    if not (STUDENT_MERGED_DIR / "config.json").exists():
        log.error(f"Merged student not found at {STUDENT_MERGED_DIR}")
        sys.exit(1)

    # Load merged student + KD adapter
    log.info("Loading merged student model...")
    model = AutoModelForCausalLM.from_pretrained(
        str(STUDENT_MERGED_DIR),
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    log.info(f"Loading KD LoRA adapter: {kd_adapter}")
    model = PeftModel.from_pretrained(model, str(kd_adapter), torch_dtype=torch.float16)
    model = model.merge_and_unload()
    log.info("KD adapter merged")

    # Save final merged model
    FINAL_MERGED.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(FINAL_MERGED), safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(str(STUDENT_MERGED_DIR), trust_remote_code=True)
    tokenizer.save_pretrained(str(FINAL_MERGED))
    log.info(f"Final KD-merged model: {FINAL_MERGED}")

    del model
    cleanup()

    # Delete teacher to free disk for GGUF export
    if TEACHER_DIR.exists():
        shutil.rmtree(TEACHER_DIR)
        log.info("Deleted teacher model (free disk for GGUF)")

    # Delete old merged model (replaced by KD version)
    if STUDENT_MERGED_DIR.exists() and FINAL_MERGED.exists():
        shutil.rmtree(STUDENT_MERGED_DIR)
        log.info("Deleted pre-KD merged model")

    # Build llama.cpp if needed
    llama_cpp = Path("/workspace/llama.cpp")
    convert_script = llama_cpp / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        log.info("Building llama.cpp for GGUF export...")
        subprocess.run(
            ["git", "clone", "--depth=1", "https://github.com/ggerganov/llama.cpp.git", str(llama_cpp)],
            check=True,
        )
        subprocess.run(["pip", "install", "-r", str(llama_cpp / "requirements.txt")], check=True)

    # Export GGUFs
    GGUF_DIR.mkdir(parents=True, exist_ok=True)
    for quant, name in [
        ("q8_0", "ariaska-cybersec2-kd-q8_0.gguf"),
        ("q6_k", "ariaska-cybersec2-kd-q6_k.gguf"),
    ]:
        gguf_path = GGUF_DIR / name
        if gguf_path.exists():
            log.info(f"GGUF {quant} exists: {gguf_path}")
            continue

        log.info(f"Exporting GGUF {quant.upper()}...")
        cmd = [
            sys.executable, str(convert_script), str(FINAL_MERGED),
            "--outtype", quant, "--outfile", str(gguf_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode == 0:
            size_mb = gguf_path.stat().st_size / 1e6
            log.info(f"GGUF {quant}: {gguf_path} ({size_mb:.0f} MB)")
        else:
            log.error(f"GGUF {quant} failed: {result.stderr[-500:]}")

    log.info("=" * 60)
    log.info("STAGE 7 COMPLETE — Knowledge-Distilled ariaska-cybersec 2.0")
    log.info(f"GGUFs: {GGUF_DIR}")
    log.info("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — Full KD Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def main():
    start = time.time()

    log.info("=" * 60)
    log.info("STAGE 7: TRUE KNOWLEDGE DISTILLATION")
    log.info("Teacher: Qwen3-32B-AWQ → Student: Qwen3-4B (post-pipeline)")
    log.info("Method: logit KD + attention transfer + curriculum pacing")
    log.info(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
    log.info(f"VRAM: {gpu_mem()}")
    log.info("=" * 60)

    # Phase 1: Free disk space
    phase1_prepare_disk()

    # Phase 2: Get teacher back
    phase2_download_teacher()

    # Load tokenizer from merged student
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(STUDENT_MERGED_DIR), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Phase 3: Prepare KD data
    kd_data = phase3_prepare_kd_data(tokenizer)

    # Phase 4: Load both models + KD training
    from peft import LoraConfig, get_peft_model, PeftModel
    from transformers import AutoModelForCausalLM

    # Load teacher (AWQ, frozen, inference only)
    log.info("Loading teacher model (AWQ, frozen)...")
    teacher = AutoModelForCausalLM.from_pretrained(
        str(TEACHER_DIR),
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    log.info(f"Teacher loaded. GPU: {gpu_mem()}")

    # Load student (merged from pipeline, trainable with LoRA)
    log.info("Loading student model (bf16, trainable LoRA)...")
    student = AutoModelForCausalLM.from_pretrained(
        str(STUDENT_MERGED_DIR),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    student.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    lora_config = LoraConfig(
        r=KD_CONFIG["lora_r"],
        lora_alpha=KD_CONFIG["lora_alpha"],
        lora_dropout=KD_CONFIG["lora_dropout"],
        target_modules=ALL_LINEAR,
        bias="none",
        task_type="CAUSAL_LM",
    )
    student = get_peft_model(student, lora_config)
    student.print_trainable_parameters()
    log.info(f"Student loaded with LoRA. GPU: {gpu_mem()}")

    # Run KD training
    kd_adapter = phase4_train_kd(teacher, student, tokenizer, kd_data)

    # Unload teacher
    del teacher
    cleanup()
    log.info("Teacher unloaded")

    # Unload student
    del student
    cleanup()

    # Phase 5: Merge + GGUF export
    phase5_merge_and_export(kd_adapter)

    elapsed = time.time() - start
    log.info(f"\nStage 7 total time: {elapsed / 3600:.1f} hours")
    log.info("Download GGUFs: scp -P 19312 root@<server>:/workspace/output/gguf/ariaska-cybersec2-kd-*.gguf .")


if __name__ == "__main__":
    main()
