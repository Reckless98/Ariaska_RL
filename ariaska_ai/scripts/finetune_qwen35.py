#!/usr/bin/env python3
"""Extract Qwen3.5-4B text-only LM backbone from VLM, then QLoRA fine-tune.

Qwen3.5-4B is a VLM (image-text-to-text). The Ollama model jaahas/qwen3.5-uncensored:4b
uses just the text backbone. This script:
1. Downloads Qwen/Qwen3.5-4B VLM
2. Extracts the language_model (Qwen3_5ForCausalLM) 
3. Saves it as standalone text-only model
4. Runs QLgfoRA fine-tuning on Ariaska cybersecurity data
5. Saves final adapter

Target: RTX 5060 Ti 16GB VRAM
"""

import argparse
import gc
import json
import logging
import os
import sys

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

EXTRACTED_TEXT_MODEL_DIR = "/root/ariaska_finetune/models/qwen3.5-4b-text-only"


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
                if "messages" in rec:
                    records.append({"messages": rec["messages"]})
            except json.JSONDecodeError:
                continue
    return records


def extract_text_model(vlm_name: str = "Qwen/Qwen3.5-4B") -> str:
    """Extract text-only CausalLM from Qwen3.5 VLM and save standalone."""
    out_dir = EXTRACTED_TEXT_MODEL_DIR

    # Check if already extracted
    if os.path.exists(os.path.join(out_dir, "config.json")):
        log.info(f"Text model already extracted at {out_dir}")
        return out_dir

    os.makedirs(out_dir, exist_ok=True)

    log.info(f"Downloading VLM: {vlm_name} ...")
    from transformers import AutoModel, AutoTokenizer, AutoConfig

    # Load tokenizer (shared between VLM and text model)
    log.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(vlm_name, trust_remote_code=True)

    # Load full VLM in FP16 on CPU
    log.info("Loading full VLM in FP16 (CPU)... This downloads ~8GB")
    vlm = AutoModel.from_pretrained(
        vlm_name,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # Extract the language model submodule
    log.info("Extracting language_model backbone...")
    if hasattr(vlm, "language_model"):
        text_model = vlm.language_model
    elif hasattr(vlm, "model"):
        # Some VLMs nest under .model.language_model
        if hasattr(vlm.model, "language_model"):
            text_model = vlm.model.language_model
        else:
            text_model = vlm.model
    else:
        raise RuntimeError(f"Cannot find language_model in VLM. Attrs: {dir(vlm)}")

    log.info(f"Extracted text model: {type(text_model).__name__}")
    param_count = sum(p.numel() for p in text_model.parameters()) / 1e9
    log.info(f"Text model parameters: {param_count:.2f}B")

    # Save the extracted text model
    log.info(f"Saving text-only model to {out_dir}...")
    text_model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)

    # Fix the config to be standalone CausalLM
    config_path = os.path.join(out_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    config["architectures"] = ["Qwen3_5ForCausalLM"]
    config["model_type"] = "qwen3_5_text"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Clean up VLM from memory
    del vlm, text_model
    gc.collect()
    torch.cuda.empty_cache()

    log.info(f"Text-only model saved to {out_dir}")
    return out_dir


def run_finetune(text_model_dir: str, cfg: dict):
    """QLoRA fine-tune the extracted text model."""
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTTrainer, SFTConfig

    tcfg = cfg["training"]
    dcfg = cfg["dataset"]

    # ── Tokenizer ──
    log.info(f"Loading tokenizer from {text_model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(text_model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log.info(f"Tokenizer loaded. Vocab: {len(tokenizer)}")

    # ── QLoRA quantization ──
    qcfg = cfg["qlora"]["quantization"]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=qcfg.get("load_in_4bit", True),
        bnb_4bit_compute_dtype=getattr(torch, qcfg.get("bnb_4bit_compute_dtype", "bfloat16")),
        bnb_4bit_quant_type=qcfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=qcfg.get("bnb_4bit_use_double_quant", True),
    )

    # ── Load text model in 4-bit ──
    log.info(f"Loading text model from {text_model_dir} in 4-bit QLoRA...")
    model = AutoModelForCausalLM.from_pretrained(
        text_model_dir,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    log.info("Model loaded and prepared for QLoRA")

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
        max_length=tcfg.get("max_seq_length", 2048),
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
    log.info("STARTING QLORA TRAINING")
    log.info(f"  Model: Qwen3.5-4B text backbone (extracted)")
    log.info(f"  Samples: {len(train_records)} train, {len(eval_records)} eval")
    log.info(f"  Epochs: {tcfg.get('num_train_epochs', 3)}")
    log.info(f"  Effective batch: {tcfg.get('per_device_train_batch_size', 2) * tcfg.get('gradient_accumulation_steps', 8)}")
    log.info("=" * 60)

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
    log.info("TRAINING COMPLETE!")
    log.info(f"Adapter: {final_dir}")
    log.info(f"Next: python export_gguf_v2.py --adapter {final_dir} --base-model {text_model_dir}")
    log.info("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--skip-extract", action="store_true", help="Skip VLM extraction if already done")
    args = parser.parse_args()

    cfg = load_config(args.config)

    log.info("=" * 60)
    log.info("ARIASKA QWEN3.5-4B FINE-TUNE PIPELINE")
    log.info("=" * 60)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        log.info(f"GPU: {props.name}")
        log.info(f"VRAM: {props.total_memory / 1e9:.1f} GB")
    else:
        log.error("No CUDA GPU found! Cannot train.")
        sys.exit(1)

    # Phase 1: Extract text model from VLM
    log.info("\n=== PHASE 1: Extract text-only backbone ===")
    text_model_dir = extract_text_model(cfg["model"]["name"])

    # Phase 2: Fine-tune
    log.info("\n=== PHASE 2: QLoRA Fine-tune ===")
    # Overwrite model name to use extracted text model
    cfg["model"]["name"] = text_model_dir
    run_finetune(text_model_dir, cfg)


if __name__ == "__main__":
    main()
