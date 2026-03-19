#!/usr/bin/env python3
"""
train_dpo.py - DPO (Direct Preference Optimization) for Ariaska

Generates rejection examples from base 4B model, then trains DPO to prefer
schema-perfect outputs over malformed ones.
"""

import json
import os
import re
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import DPOTrainer, DPOConfig


def strip_think_tags(text: str) -> str:
    return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()


def generate_rejection_pairs(
    model_path: str,
    sft_data_path: str,
    output_path: str,
    max_pairs: int = 5000,
):
    """Generate DPO pairs: chosen=teacher output, rejected=4B SFT model output."""
    from vllm import LLM, SamplingParams
    
    print(f"Loading SFT model from {model_path} for rejection sampling...")
    llm = LLM(
        model=model_path,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        dtype="float16",
    )
    
    # Load SFT data (has perfect teacher outputs)
    examples = []
    with open(sft_data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    
    # Sample subset for DPO
    import random
    random.seed(42)
    if len(examples) > max_pairs:
        examples = random.sample(examples, max_pairs)
    
    print(f"Generating rejections for {len(examples)} pairs...")
    
    # Extract prompts
    prompts = []
    for ex in examples:
        msgs = ex["messages"]
        # Build prompt without assistant response
        prompt_msgs = [m for m in msgs if m["role"] != "assistant"]
        prompts.append(prompt_msgs)
    
    # Format as chat prompts for vLLM
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    formatted_prompts = []
    for msgs in prompts:
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        formatted_prompts.append(text)
    
    # Generate rejections in batches
    params = SamplingParams(
        temperature=0.7,  # Higher temp for diverse (potentially wrong) outputs
        top_p=0.9,
        max_tokens=600,
    )
    
    batch_size = 64
    all_rejections = []
    for i in range(0, len(formatted_prompts), batch_size):
        batch = formatted_prompts[i:i + batch_size]
        outputs = llm.generate(batch, params)
        for output in outputs:
            text = strip_think_tags(output.outputs[0].text)
            all_rejections.append(text)
        print(f"  Generated {len(all_rejections)}/{len(formatted_prompts)} rejections")
    
    # Build DPO pairs
    dpo_pairs = []
    for ex, rejection in zip(examples, all_rejections):
        msgs = ex["messages"]
        chosen = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        
        # Only include pair if rejection is different from chosen
        if rejection.strip() != chosen.strip():
            pair = {
                "prompt": [m for m in msgs if m["role"] != "assistant"],
                "chosen": chosen,
                "rejected": rejection,
                "schema_type": ex.get("schema_type", "unknown"),
            }
            dpo_pairs.append(pair)
    
    print(f"Generated {len(dpo_pairs)} DPO pairs (filtered from {len(examples)})")
    
    with open(output_path, "w") as f:
        for pair in dpo_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    
    print(f"DPO pairs saved to {output_path}")
    return dpo_pairs


def load_dpo_dataset(path: str) -> Dataset:
    """Load DPO dataset from JSONL."""
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    
    # Format for DPO trainer
    formatted = []
    for ex in examples:
        prompt_msgs = ex["prompt"]
        prompt_text = "\n".join(f"{m['role']}: {m['content']}" for m in prompt_msgs)
        formatted.append({
            "prompt": prompt_text,
            "chosen": ex["chosen"],
            "rejected": ex["rejected"],
        })
    
    return Dataset.from_list(formatted)


def main():
    parser = argparse.ArgumentParser(description="DPO Training for Ariaska")
    parser.add_argument("--base-model", required=True, help="Path to ariaska-cybersec HF base model")
    parser.add_argument("--sft-adapter", required=True, help="Path to SFT adapter (output/final/)")
    parser.add_argument("--teacher-model", default=None, help="Path to teacher model for rejection gen")
    parser.add_argument("--sft-data", default="/workspace/data/ariaska_schema_sft.jsonl")
    parser.add_argument("--dpo-data", default="/workspace/data/ariaska_dpo_pairs.jsonl")
    parser.add_argument("--output", default="/workspace/outputs/schema_dpo")
    parser.add_argument("--skip-rejection-gen", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    args = parser.parse_args()
    
    print(f"=== Ariaska DPO Training ===")
    
    # Step 1: Build merged SFT model for rejection generation
    # Merge base + SFT adapter for vLLM-based rejection sampling
    merged_sft_dir = str(Path(args.output) / "_merged_sft_for_rejection")
    
    if not args.skip_rejection_gen:
        print("\n[1/3] Merging base+SFT adapter for rejection sampling...")
        from peft import PeftModel as _PeftModel
        _base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        _base = _PeftModel.from_pretrained(_base, args.sft_adapter)
        _base = _base.merge_and_unload()
        Path(merged_sft_dir).mkdir(parents=True, exist_ok=True)
        _base.save_pretrained(merged_sft_dir, safe_serialization=True)
        _tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        _tok.save_pretrained(merged_sft_dir)
        del _base, _tok
        import gc; gc.collect(); torch.cuda.empty_cache()
        
        print("[1/3] Generating rejection pairs...")
        generate_rejection_pairs(
            model_path=merged_sft_dir,
            sft_data_path=args.sft_data,
            output_path=args.dpo_data,
            max_pairs=args.max_pairs,
        )
    else:
        print("[1/3] Skipping rejection generation (--skip-rejection-gen)")
    
    # Step 2: Load base model + merge SFT adapter for fresh DPO LoRA
    print("\n[2/3] Loading model and dataset...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True, padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",  # flash_attention_2 crashes on Qwen3.5
    )
    
    # Merge SFT adapter into base
    from peft import PeftModel
    print(f"  Merging SFT adapter from {args.sft_adapter}...")
    model = PeftModel.from_pretrained(model, args.sft_adapter)
    model = model.merge_and_unload()
    model = prepare_model_for_kbit_training(model)
    
    # LoRA for DPO (smaller rank than SFT)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load DPO data
    dataset = load_dpo_dataset(args.dpo_data)
    split = dataset.train_test_split(test_size=0.05, seed=42)
    
    print(f"Train: {len(split['train'])}, Eval: {len(split['test'])}")
    
    # Step 3: DPO Training
    print("\n[3/3] Starting DPO training...")
    
    training_args = DPOConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        report_to="none",
        beta=args.beta,
        max_length=args.max_seq_len,
        max_prompt_length=args.max_seq_len // 2,
    )
    
    # Need ref model for DPO (base + SFT, frozen)
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    if Path(args.sft_adapter).exists():
        ref_model = PeftModel.from_pretrained(ref_model, args.sft_adapter)
        ref_model = ref_model.merge_and_unload()
    
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tokenizer,
    )
    
    trainer.train()
    
    # Save final adapter
    final_dir = str(Path(args.output) / "final")
    Path(final_dir).mkdir(parents=True, exist_ok=True)
    print(f"\nSaving DPO adapter to {final_dir}...")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    # Evaluate
    metrics = trainer.evaluate()
    print(f"DPO eval metrics: {json.dumps(metrics, indent=2)}")
    with open(str(Path(args.output) / "dpo_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n=== DPO TRAINING COMPLETE ===")
    print(f"LoRA adapter: {final_dir}")


if __name__ == "__main__":
    main()
