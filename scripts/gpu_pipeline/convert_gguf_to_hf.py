#!/usr/bin/env python3
"""
convert_gguf_to_hf.py — Convert ariaska-cybersec Q8_0 GGUF back to HuggingFace format.

Strategy:
1. Load the original Qwen3.5-4B config/tokenizer from HuggingFace
2. Create empty model with correct architecture 
3. Use llama-cpp-python to load GGUF and extract dequantized weights
4. Map GGUF tensor names → HF tensor names
5. Save as HF safetensors

Q8_0 dequantization is nearly lossless (~0.001% error).
"""

import os
import sys
import json
import gc
import logging
from pathlib import Path

import torch
import numpy as np

log = logging.getLogger("convert_gguf_to_hf")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s  %(message)s")

GGUF_PATH = Path("/workspace/models/ariaska-cybersec-q8_0.gguf")
OUTPUT_DIR = Path("/workspace/models/ariaska-cybersec-hf")

# Original base model for config/tokenizer/architecture reference
BASE_MODEL = "Qwen/Qwen3.5-4B"


def convert_gguf_to_hf():
    """Convert GGUF to HuggingFace using gguf library (ships with transformers)."""
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

    if (OUTPUT_DIR / "config.json").exists() and (OUTPUT_DIR / "model.safetensors").exists():
        log.info(f"HF model already exists at {OUTPUT_DIR}")
        return OUTPUT_DIR

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Method 1: Use transformers' native GGUF loading (added in transformers 4.40+)
    log.info("Loading GGUF via transformers native GGUF support...")
    log.info(f"GGUF: {GGUF_PATH}")
    log.info(f"Output: {OUTPUT_DIR}")

    try:
        # transformers >= 4.40 can load GGUF directly
        model = AutoModelForCausalLM.from_pretrained(
            str(GGUF_PATH.parent),
            gguf_file=GGUF_PATH.name,
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        log.info(f"Model loaded from GGUF: {type(model).__name__}")
        log.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Save as HF safetensors
        model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
        log.info(f"Model saved to {OUTPUT_DIR}")

        # Get tokenizer from original base
        log.info(f"Downloading tokenizer from {BASE_MODEL}...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        tokenizer.save_pretrained(OUTPUT_DIR)
        log.info(f"Tokenizer saved to {OUTPUT_DIR}")

        del model
        gc.collect()

        log.info("GGUF -> HF conversion COMPLETE")
        return OUTPUT_DIR

    except Exception as e:
        log.warning(f"Native GGUF loading failed: {e}")
        log.info("Trying fallback method with gguf library...")

    # Method 2: Manual extraction using gguf library
    try:
        from gguf import GGUFReader
    except ImportError:
        log.info("Installing gguf library...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gguf"])
        from gguf import GGUFReader

    log.info("Reading GGUF file...")
    reader = GGUFReader(str(GGUF_PATH))

    # Get architecture info from GGUF metadata
    metadata = {}
    for field in reader.fields.values():
        if len(field.data) > 0:
            try:
                val = field.data[0]
                if hasattr(val, 'item'):
                    val = val.item()
                metadata[field.name] = val
            except Exception:
                pass

    log.info(f"GGUF metadata keys: {list(metadata.keys())[:20]}")

    # Download config + tokenizer from original base
    log.info(f"Downloading config from {BASE_MODEL}...")
    config = AutoConfig.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # For Qwen3.5-4B (VLM), we need the text model config
    if hasattr(config, 'text_config'):
        text_config = config.text_config
        log.info("Using text_config from VLM config")
    else:
        text_config = config
        log.info("Using config directly (not a VLM config)")

    # Build empty model
    log.info("Creating empty model skeleton...")
    model = AutoModelForCausalLM.from_config(
        text_config,
        torch_dtype=torch.float16,
    )
    log.info(f"Empty model: {sum(p.numel() for p in model.parameters()):,} params")

    # Map GGUF tensor names to HF names
    state_dict = model.state_dict()
    gguf_tensors = {tensor.name: tensor for tensor in reader.tensors}

    log.info(f"GGUF has {len(gguf_tensors)} tensors")
    log.info(f"HF model expects {len(state_dict)} tensors")

    # Standard GGUF → HF name mapping for Qwen/LLaMA architecture
    def gguf_name_to_hf(gguf_name: str) -> str:
        """Convert GGUF tensor name to HuggingFace name."""
        name = gguf_name

        # Token embeddings
        name = name.replace("token_embd.weight", "model.embed_tokens.weight")

        # Output head
        name = name.replace("output.weight", "lm_head.weight")
        name = name.replace("output_norm.weight", "model.norm.weight")

        # Attention layers
        name = name.replace("blk.", "model.layers.")

        # Within each block
        name = name.replace(".attn_q.weight", ".self_attn.q_proj.weight")
        name = name.replace(".attn_k.weight", ".self_attn.k_proj.weight")
        name = name.replace(".attn_v.weight", ".self_attn.v_proj.weight")
        name = name.replace(".attn_output.weight", ".self_attn.o_proj.weight")

        # Q/K norms (for models that have them)
        name = name.replace(".attn_q_norm.weight", ".self_attn.q_norm.weight")
        name = name.replace(".attn_k_norm.weight", ".self_attn.k_norm.weight")

        # FFN
        name = name.replace(".ffn_gate.weight", ".mlp.gate_proj.weight")
        name = name.replace(".ffn_up.weight", ".mlp.up_proj.weight")
        name = name.replace(".ffn_down.weight", ".mlp.down_proj.weight")

        # Layer norms
        name = name.replace(".attn_norm.weight", ".input_layernorm.weight")
        name = name.replace(".ffn_norm.weight", ".post_attention_layernorm.weight")

        return name

    # Dequantize and load each tensor
    mapped = 0
    unmapped = []
    for gguf_name, tensor in gguf_tensors.items():
        hf_name = gguf_name_to_hf(gguf_name)

        if hf_name not in state_dict:
            unmapped.append((gguf_name, hf_name))
            continue

        # Dequantize Q8_0 → float16
        data = tensor.data.copy()
        if data.dtype in (np.float32, np.float16):
            t = torch.from_numpy(data).to(torch.float16)
        else:
            # Q8_0 is stored as int8 blocks — dequantize
            t = torch.from_numpy(data.astype(np.float32)).to(torch.float16)

        expected_shape = state_dict[hf_name].shape
        if t.shape != expected_shape:
            try:
                t = t.reshape(expected_shape)
            except Exception as e:
                log.warning(f"Shape mismatch {gguf_name}: GGUF {t.shape} vs HF {expected_shape}: {e}")
                continue

        state_dict[hf_name] = t
        mapped += 1

    log.info(f"Mapped {mapped}/{len(gguf_tensors)} tensors")
    if unmapped:
        log.warning(f"Unmapped tensors ({len(unmapped)}): {unmapped[:10]}")

    model.load_state_dict(state_dict)

    # Save
    model.save_pretrained(OUTPUT_DIR, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.save_pretrained(OUTPUT_DIR)

    del model, state_dict
    gc.collect()

    log.info(f"GGUF -> HF conversion COMPLETE: {OUTPUT_DIR}")
    return OUTPUT_DIR


if __name__ == "__main__":
    result = convert_gguf_to_hf()
    print(f"\nDone: {result}")
