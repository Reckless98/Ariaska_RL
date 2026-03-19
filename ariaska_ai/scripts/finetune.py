#!/usr/bin/env python3
"""QLoRA fine-tuning script for Qwen3.5-4B as Ariaska cybersecurity coprocessor.

Usage:
    python ariaska_ai/scripts/finetune.py --config ariaska_ai/configs/qlora_qwen35_4b.yaml

Designed for single 3090 (24GB VRAM). Uses:
  - QLoRA (4-bit NF4) to fit in VRAM
  - Gradient checkpointing for memory efficiency
  - SFTTrainer from trl for chat-format training
  - Exports merged model + GGUF after training
"""

import argparse
import json
import logging
import os
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ariaska.finetune")


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_dataset_from_jsonl(path: str) -> Dataset:
    """Load JSONL chat dataset into HuggingFace Dataset."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)


def setup_quantization(cfg: dict) -> BitsAndBytesConfig:
    """Configure 4-bit quantization."""
    q = cfg["qlora"]["quantization"]
    compute_dtype = getattr(torch, q.get("bnb_4bit_compute_dtype", "bfloat16"))
    return BitsAndBytesConfig(
        load_in_4bit=q.get("load_in_4bit", True),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=q.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=q.get("bnb_4bit_use_double_quant", True),
    )


def setup_lora(cfg: dict) -> LoraConfig:
    """Configure LoRA adapter."""
    lora = cfg["qlora"]
    return LoraConfig(
        r=lora.get("r", 64),
        lora_alpha=lora.get("lora_alpha", 128),
        lora_dropout=lora.get("lora_dropout", 0.05),
        target_modules=lora.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        bias=lora.get("bias", "none"),
        task_type=lora.get("task_type", "CAUSAL_LM"),
    )


def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tune Qwen3.5-4B for Ariaska")
    parser.add_argument("--config", type=str, default="ariaska_ai/configs/qlora_qwen35_4b.yaml")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_name = cfg["model"]["name"]
    train_cfg = cfg["training"]
    ds_cfg = cfg["dataset"]

    log.info(f"Loading model: {model_name}")

    # ── Tokenizer ──────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=cfg["model"].get("trust_remote_code", True),
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Quantization ───────────────────────────────────────────────────────
    bnb_config = setup_quantization(cfg)

    # ── Model ──────────────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=getattr(torch, cfg["model"].get("torch_dtype", "bfloat16")),
        device_map="auto",
        trust_remote_code=cfg["model"].get("trust_remote_code", True),
        attn_implementation=cfg["model"].get("attn_implementation", "flash_attention_2"),
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # ── LoRA ───────────────────────────────────────────────────────────────
    lora_config = setup_lora(cfg)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Dataset ────────────────────────────────────────────────────────────
    log.info(f"Loading train data: {ds_cfg['train_file']}")
    train_dataset = load_dataset_from_jsonl(ds_cfg["train_file"])
    log.info(f"Loading eval data: {ds_cfg['eval_file']}")
    eval_dataset = load_dataset_from_jsonl(ds_cfg["eval_file"])
    log.info(f"Train: {len(train_dataset)} samples, Eval: {len(eval_dataset)} samples")

    # ── Training Args ──────────────────────────────────────────────────────
    output_dir = train_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=train_cfg.get("learning_rate", 2e-4),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.05),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        max_seq_length=train_cfg.get("max_seq_length", 2048),
        packing=train_cfg.get("packing", True),
        logging_steps=train_cfg.get("logging_steps", 10),
        save_strategy=train_cfg.get("save_strategy", "steps"),
        save_steps=train_cfg.get("save_steps", 200),
        eval_strategy=train_cfg.get("eval_strategy", "steps"),
        eval_steps=train_cfg.get("eval_steps", 200),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        fp16=train_cfg.get("fp16", False),
        bf16=train_cfg.get("bf16", True),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs=train_cfg.get("gradient_checkpointing_kwargs", {"use_reentrant": False}),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 4),
        seed=train_cfg.get("seed", 42),
        report_to=train_cfg.get("report_to", "none"),
    )

    # ── Trainer ────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # ── Train ──────────────────────────────────────────────────────────────
    log.info("Starting training...")
    if args.resume_from:
        log.info(f"Resuming from checkpoint: {args.resume_from}")
        trainer.train(resume_from_checkpoint=args.resume_from)
    else:
        trainer.train()

    # ── Save ───────────────────────────────────────────────────────────────
    final_dir = os.path.join(output_dir, "final")
    log.info(f"Saving final adapter to {final_dir}")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    # ── Eval ───────────────────────────────────────────────────────────────
    log.info("Running final evaluation...")
    metrics = trainer.evaluate()
    log.info(f"Final eval metrics: {json.dumps(metrics, indent=2)}")
    with open(os.path.join(output_dir, "final_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    log.info("Training complete!")
    log.info(f"Next step: python ariaska_ai/scripts/export_gguf.py --adapter-path {final_dir}")


if __name__ == "__main__":
    main()
