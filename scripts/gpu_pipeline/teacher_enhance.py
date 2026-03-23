#!/usr/bin/env python3
"""
teacher_enhance.py — Use a local 27B teacher to enhance SFT data quality.

Loads Qwen3.5-27B (AWQ 4-bit) and rewrites shallow reasoning
in SFT examples to produce deeper tactical analysis.

Usage:
  python teacher_enhance.py --input data/htb_sft_v5.jsonl --output data/htb_sft_v6.jsonl
  python teacher_enhance.py --input data/htb_sft_v5.jsonl --output data/htb_sft_v6.jsonl --model QuantTrio/Qwen3.5-27B-AWQ
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("teacher_enhance")

# ── Enhancement prompt ───────────────────────────────────────────────────────

ENHANCE_SYSTEM = """You are an elite penetration testing instructor. Your job is to rewrite
shallow command reasoning into detailed 2-3 sentence tactical analysis.

Rules:
- Output ONLY the improved reasoning text, nothing else
- Reference specific services, ports, or findings from the discovery board
- Explain WHY this command matters at this stage of the engagement
- Mention what the attacker expects to find and how it connects to the next step
- Be specific to the box/scenario, not generic
- Keep it under 120 words"""

ENHANCE_TEMPLATE = """Rewrite this shallow reasoning into deep tactical analysis.

Context:
- Target: {target}
- Phase: {phase}
- Command: {command}
- OS: {os_type}
- Discovery board ports: {ports}
- Discovery board services: {services}
- Box: {box_name}

Original shallow reasoning:
"{original_reasoning}"

Rewrite as 2-3 sentences of detailed tactical analysis:"""


def load_teacher(model_name: str) -> tuple:
    """Load teacher model with 4-bit quantization."""
    log.info(f"Loading teacher model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Try AWQ first (pre-quantized), fall back to bitsandbytes 4-bit
    if "AWQ" in model_name or "awq" in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

    log.info(f"Teacher loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    return model, tokenizer


def enhance_reasoning(
    model, tokenizer, original: str, target: str, phase: str,
    command: str, board: dict, box_name: str = "", os_type: str = "linux",
) -> str | None:
    """Use teacher to rewrite shallow reasoning into deep tactical analysis."""
    ports = ", ".join(board.get("ports", [])[:8])
    services = ", ".join(f"{p}:{s}" for p, s in list(board.get("services", {}).items())[:5])

    prompt = ENHANCE_TEMPLATE.format(
        target=target,
        phase=phase,
        command=command,
        os_type=os_type,
        ports=ports or "unknown",
        services=services or "unknown",
        box_name=box_name or "unknown",
        original_reasoning=original,
    )

    messages = [
        {"role": "system", "content": ENHANCE_SYSTEM},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    # Validate: must be > 50 chars, < 500 chars, no markdown, no JSON
    if len(response) < 50 or len(response) > 500:
        return None
    if response.startswith("{") or response.startswith("[") or response.startswith("#"):
        return None

    return response


def process_file(input_path: Path, output_path: Path, model_name: str, max_enhance: int = 0):
    """Process SFT file, enhancing shallow reasoning with teacher."""
    model, tokenizer = load_teacher(model_name)

    with open(input_path) as f:
        examples = [json.loads(line) for line in f]

    log.info(f"Loaded {len(examples)} examples from {input_path}")

    enhanced = 0
    skipped = 0
    errors = 0

    for i, ex in enumerate(examples):
        schema = ex.get("schema_type", "")

        # Only enhance mentor and microchain examples with shallow reasoning
        if schema not in ("smart_mentor", "smart_mentor_walkthrough", "microchain_fast_local"):
            continue

        msgs = ex["messages"]
        assistant_msg = msgs[-1]["content"]
        user_msg = msgs[-2]["content"] if len(msgs) > 1 else ""

        try:
            resp = json.loads(assistant_msg)
        except json.JSONDecodeError:
            continue

        reasoning = resp.get("reasoning", resp.get("reason", ""))
        if len(reasoning) >= 80:
            continue  # Already has decent reasoning

        if max_enhance and enhanced >= max_enhance:
            break

        # Extract context from user message
        target = "10.10.10.1"
        phase = "ENUMERATION"
        command = resp.get("selected_command", resp.get("command", ""))
        box_name = ""
        os_type = "linux"
        board = {}

        for line in user_msg.split("\n"):
            if line.startswith("Target:"):
                target = line.split(":", 1)[1].strip()
            elif line.startswith("Phase:"):
                phase = line.split(":", 1)[1].strip()
            elif line.startswith("OS:"):
                os_type = line.split(":", 1)[1].strip()
            elif "Box:" in line:
                box_name = line.split("Box:", 1)[1].strip()

        # Try to parse discovery board from user message
        if "Discovery board:" in user_msg:
            board_json = user_msg.split("Discovery board:")[-1].split("Recent commands:")[0]
            board_json = board_json.split("Available templates:")[0]
            board_json = board_json.split("HTB Box:")[0]
            try:
                board = json.loads(board_json.strip())
            except json.JSONDecodeError:
                board = {}

        # Enhance with teacher
        try:
            new_reasoning = enhance_reasoning(
                model, tokenizer, reasoning, target, phase,
                command, board, box_name, os_type,
            )
        except Exception as e:
            errors += 1
            if errors <= 5:
                log.warning(f"Enhancement error at example {i}: {e}")
            continue

        if new_reasoning:
            # Update reasoning in the response
            if "reasoning" in resp:
                resp["reasoning"] = new_reasoning
            elif "reason" in resp:
                resp["reason"] = new_reasoning
            msgs[-1]["content"] = json.dumps(resp)
            enhanced += 1

            if enhanced % 100 == 0:
                log.info(f"Enhanced {enhanced} examples ({errors} errors, {i+1}/{len(examples)} processed)")
        else:
            skipped += 1

    # Write output
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    log.info(f"Done: {enhanced} enhanced, {skipped} skipped (bad output), {errors} errors")
    log.info(f"Output: {output_path}")
    return enhanced, skipped, errors


def main():
    parser = argparse.ArgumentParser(description="Enhance SFT data with teacher model")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=str, default="QuantTrio/Qwen3.5-27B-AWQ")
    parser.add_argument("--max-enhance", type=int, default=0, help="Max examples to enhance (0=all)")
    args = parser.parse_args()

    process_file(args.input, args.output, args.model, args.max_enhance)


if __name__ == "__main__":
    main()
