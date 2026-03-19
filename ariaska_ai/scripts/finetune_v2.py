#!/usr/bin/env python3
"""QLoRA fine-tuning: Qwen3-4B-abliterated -> Ariaska cybersecurity coprocessor.

Usage:
    python finetune_v2.py --config /root/ariaska_finetune/configs/qlora_qwen35_4b.yaml

Designed for RTX 5060 Ti 16GB. Uses QLoRA 4-bit NF4, gradient checkpointing,
sequence packing, cosine LR with warmup.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/root/ariaska_finetune/training.log", mode="w"),
    ],
)
log = logging.getLogger("ariaska.finetune")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                # Ensure 'messages' key exists with proper format
                if "messages" in rec:
                    records.append({"messages": rec["messages"]})
            except json.JSONDecodeError:
                continue
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_name = cfg["model"]["name"]
    tcfg = cfg["training"]
    dcfg = cfg["dataset"]

    log.info(f"=== Ariaska Fine-Tune V2 ===")
    log.info(f"Model: {model_name}")
    log.info(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    log.info(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB" if torch.cuda.is_available() else "N/A")

    # ── Imports (after logging setup) ──
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTTrainer, SFTConfig

    # ── Tokenizer ──
    log.info(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log.info(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")

    # ── QLoRA quantization config ──
    qcfg = cfg["qlora"]["quantization"]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=qcfg.get("load_in_4bit", True),
        bnb_4bit_compute_dtype=getattr(torch, qcfg.get("bnb_4bit_compute_dtype", "bfloat16")),
        bnb_4bit_quant_type=qcfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=qcfg.get("bnb_4bit_use_double_quant", True),
    )

    # ── Load model in 4-bit ──
    log.info(f"Loading model {model_name} in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        dtype=getattr(torch, cfg["model"].get("torch_dtype", "bfloat16")),
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=cfg["model"].get("attn_implementation", "sdpa"),
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    log.info("Model loaded and prepared for QLoRA training")

    # ── LoRA adapter ──
    lcfg = cfg["qlora"]
    lora_config = LoraConfig(
        r=lcfg.get("r", 64),
        lora_alpha=lcfg.get("lora_alpha", 128),
        lora_dropout=lcfg.get("lora_dropout", 0.05),
        target_modules=lcfg.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]),
        bias=lcfg.get("bias", "none"),
        task_type=lcfg.get("task_type", "CAUSAL_LM"),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Dataset ──
    log.info(f"Loading training data: {dcfg['train_file']}")
    train_records = load_jsonl(dcfg["train_file"])
    log.info(f"Loading eval data: {dcfg['eval_file']}")
    eval_records = load_jsonl(dcfg["eval_file"])
    log.info(f"Train: {len(train_records)} samples, Eval: {len(eval_records)} samples")

    train_dataset = Dataset.from_list(train_records)
    eval_dataset = Dataset.from_list(eval_records)

    # ── Training config ──
    output_dir = tcfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=tcfg.get("num_train_epochs", 3),
        per_device_train_batch_size=tcfg.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=tcfg.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=tcfg.get("gradient_accumulation_steps", 8),
        learning_rate=tcfg.get("learning_rate", 2e-4),
        lr_scheduler_type=tcfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=tcfg.get("warmup_ratio", 0.05),
        weight_decay=tcfg.get("weight_decay", 0.01),
        max_grad_norm=tcfg.get("max_grad_norm", 1.0),
        max_seq_length=tcfg.get("max_seq_length", 2048),
        packing=tcfg.get("packing", True),
        logging_steps=tcfg.get("logging_steps", 10),
        save_strategy=tcfg.get("save_strategy", "steps"),
        save_steps=tcfg.get("save_steps", 500),
        eval_strategy=tcfg.get("eval_strategy", "steps"),
        eval_steps=tcfg.get("eval_steps", 500),
        save_total_limit=tcfg.get("save_total_limit", 2),
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=tcfg.get("dataloader_num_workers", 2),
        seed=tcfg.get("seed", 42),
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    # ── Trainer ──
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # ── Train ──
    log.info("=" * 60)
    log.info("STARTING TRAINING")
    log.info("=" * 60)

    if args.resume:
        log.info(f"Resuming from: {args.resume}")
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()

    # ── Save final adapter ──
    final_dir = os.path.join(output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    log.info(f"Saving final adapter to {final_dir}")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    # ── Final eval ──
    log.info("Running final evaluation...")
    metrics = trainer.evaluate()
    log.info(f"Final metrics: {json.dumps(metrics, indent=2)}")
    with open(os.path.join(output_dir, "final_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    log.info("=" * 60)
    log.info("TRAINING COMPLETE")
    log.info(f"Adapter saved to: {final_dir}")
    log.info(f"Next: python export_gguf_v2.py --adapter {final_dir}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
