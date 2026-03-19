#!/usr/bin/env python3
"""GGUF export V3: Disk-efficient merge + export using /dev/shm staging.

Re-downloads VLM, extracts text backbone, merges LoRA, converts to GGUF
using /dev/shm (RAM-backed tmpfs) as staging to avoid overlay disk pressure.

Usage:
    python export_gguf_v3.py \
        --adapter /root/ariaska_finetune/outputs/qwen3-4b-ariaska/final \
        --vlm-name Qwen/Qwen3.5-4B
"""

import argparse
import gc
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("ariaska.export_v3")

SHM_DIR = Path("/dev/shm/ariaska_export")
OVERLAY_GGUF_DIR = Path("/root/ariaska_finetune/outputs/gguf")
LLAMA_DIR = Path("/root/llama.cpp")


def disk_free_mb(path: str = "/") -> int:
    st = os.statvfs(path)
    return (st.f_bavail * st.f_frsize) // (1024 * 1024)


def cleanup_shm():
    """Remove all staging data from /dev/shm."""
    if SHM_DIR.exists():
        shutil.rmtree(SHM_DIR, ignore_errors=True)
        log.info("Cleaned /dev/shm staging")


def extract_and_merge(vlm_name: str, adapter_path: str) -> str:
    """Download VLM → extract text backbone → wrap as CausalLM → merge LoRA → save.

    Builds Qwen3_5ForCausalLM in-memory from the VLM's language_model (TextModel)
    without any intermediate disk saves (avoids mmap/shm issues).
    Merged model saved to /dev/shm for GGUF conversion.

    Returns path to merged model directory (in /dev/shm).
    """
    import torch
    from peft import PeftModel
    from transformers import AutoConfig, AutoModel, AutoTokenizer, Qwen3_5ForCausalLM

    SHM_DIR.mkdir(parents=True, exist_ok=True)
    merged_dir = str(SHM_DIR / "merged")

    if os.path.exists(os.path.join(merged_dir, "config.json")):
        log.info(f"Merged model already exists at {merged_dir}")
        return merged_dir

    # --- Step 1: Download VLM and extract text backbone ---
    log.info(f"Downloading VLM '{vlm_name}' (overlay HF cache)...")
    log.info(f"  Overlay free: {disk_free_mb('/')} MB, /dev/shm free: {disk_free_mb('/dev/shm')} MB")

    vlm = AutoModel.from_pretrained(
        vlm_name,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # AutoModel → Qwen3_5Model. Its .language_model is Qwen3_5TextModel.
    text_model = vlm.language_model
    log.info(f"  Extracted: {type(text_model).__name__} "
             f"(hidden={text_model.config.hidden_size}, layers={text_model.config.num_hidden_layers})")

    # Get the text config (needed for CausalLM construction)
    text_config = AutoConfig.from_pretrained(vlm_name, trust_remote_code=True).text_config

    # Free VLM visual encoder (text_model kept via our reference)
    del vlm
    gc.collect()

    # Free HF cache from overlay
    hf_cache_model = Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen3.5-4B"
    if hf_cache_model.exists():
        cache_mb = sum(f.stat().st_size for f in hf_cache_model.rglob("*") if f.is_file()) // (1024 * 1024)
        log.info(f"  Deleting VLM HF cache ({cache_mb} MB)...")
        shutil.rmtree(hf_cache_model, ignore_errors=True)

    log.info(f"  After cleanup — overlay: {disk_free_mb('/')} MB, /dev/shm: {disk_free_mb('/dev/shm')} MB")

    # --- Step 2: Wrap TextModel as CausalLM (in-memory, no disk IO) ---
    # Qwen3_5ForCausalLM expects: .model = TextModel, .lm_head (tied to embed_tokens).
    # We create an empty shell and assign the real text_model into it.
    log.info("Building Qwen3_5ForCausalLM wrapper in memory...")

    from accelerate import init_empty_weights
    with init_empty_weights():
        causal_lm = Qwen3_5ForCausalLM(text_config)

    # Replace the meta-tensor model with our real text model
    causal_lm.model = text_model
    # tie_word_embeddings=True → lm_head.weight = model.embed_tokens.weight
    causal_lm.tie_weights()

    log.info(f"  Built: {type(causal_lm).__name__}")
    assert hasattr(causal_lm, "prepare_inputs_for_generation"), \
        f"{type(causal_lm).__name__} missing prepare_inputs_for_generation"

    del text_model
    gc.collect()

    # --- Step 3: Load LoRA adapter and merge ---
    log.info(f"Loading LoRA adapter from {adapter_path}...")
    model_with_lora = PeftModel.from_pretrained(causal_lm, adapter_path)

    log.info("Merging adapter weights...")
    merged = model_with_lora.merge_and_unload()

    del causal_lm, model_with_lora
    gc.collect()

    # --- Step 4: Save merged model to /dev/shm ---
    log.info(f"Saving merged model to {merged_dir}...")
    log.info(f"  /dev/shm free: {disk_free_mb('/dev/shm')} MB")
    os.makedirs(merged_dir, exist_ok=True)
    merged.save_pretrained(merged_dir, safe_serialization=True)

    # Save tokenizer from adapter dir
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    tokenizer.save_pretrained(merged_dir)

    del merged, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    merged_size = sum(f.stat().st_size for f in Path(merged_dir).rglob("*") if f.is_file()) // (1024 * 1024)
    log.info(f"Merged model saved: {merged_dir} ({merged_size} MB)")
    log.info(f"  /dev/shm free: {disk_free_mb('/dev/shm')} MB")
    return merged_dir


def convert_and_quantize(merged_dir: str, quantizations: list[str]) -> list[str]:
    """Convert merged model to GGUF and quantize. Output on overlay disk."""
    OVERLAY_GGUF_DIR.mkdir(parents=True, exist_ok=True)

    convert_script = LLAMA_DIR / "convert_hf_to_gguf.py"
    quantize_bin = LLAMA_DIR / "build" / "bin" / "llama-quantize"

    if not convert_script.exists():
        raise FileNotFoundError(f"llama.cpp convert script not found: {convert_script}")
    if not quantize_bin.exists():
        raise FileNotFoundError(f"llama-quantize not found: {quantize_bin}")

    # Step 1: Convert to F16 GGUF (write to overlay)
    f16_gguf = str(OVERLAY_GGUF_DIR / "ariaska-qwen3-4b-f16.gguf")
    if not os.path.exists(f16_gguf):
        log.info(f"Converting to F16 GGUF...")
        log.info(f"  Overlay free: {disk_free_mb('/')} MB")
        cmd = [
            sys.executable, str(convert_script),
            merged_dir,
            "--outfile", f16_gguf,
            "--outtype", "f16",
        ]
        log.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log.error(f"Convert failed:\n{result.stderr[-2000:]}")
            raise RuntimeError("F16 GGUF conversion failed")
        f16_size = os.path.getsize(f16_gguf) // (1024 * 1024)
        log.info(f"  F16 GGUF: {f16_size} MB")
    else:
        log.info(f"F16 GGUF already exists: {f16_gguf}")

    # Free shm: merged model no longer needed
    cleanup_shm()
    log.info(f"  Overlay free after shm cleanup: {disk_free_mb('/')} MB")

    # Step 2: Quantize one at a time, checking disk
    output_files = []
    for quant in quantizations:
        quant_file = str(OVERLAY_GGUF_DIR / f"ariaska-qwen3-4b-{quant.lower().replace('_', '-')}.gguf")
        if os.path.exists(quant_file):
            log.info(f"Already exists: {quant_file}")
            output_files.append(quant_file)
            continue

        free_mb = disk_free_mb("/")
        log.info(f"Quantizing to {quant}... (overlay free: {free_mb} MB)")
        if free_mb < 4000:
            log.warning(f"Low disk space ({free_mb} MB), skipping {quant}")
            continue

        cmd = [str(quantize_bin), f16_gguf, quant_file, quant]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log.error(f"Quantize {quant} failed:\n{result.stderr[-2000:]}")
            continue
        size_mb = os.path.getsize(quant_file) // (1024 * 1024)
        log.info(f"  {quant}: {size_mb} MB")
        output_files.append(quant_file)

    # Clean up F16 intermediate
    if output_files and os.path.exists(f16_gguf):
        log.info("Removing intermediate F16 GGUF...")
        os.remove(f16_gguf)
        log.info(f"  Overlay free: {disk_free_mb('/')} MB")

    return output_files


def create_modelfile(gguf_path: str) -> str:
    """Create Ollama Modelfile."""
    modelfile = str(OVERLAY_GGUF_DIR / "Modelfile_ariaska_qwen3")
    gguf_name = os.path.basename(gguf_path)

    content = f"""FROM ./{gguf_name}

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 2048
PARAMETER repeat_penalty 1.1
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"

SYSTEM \"\"\"You are Ariaska, a cybersecurity AI coprocessor for authorized penetration testing in lab environments. You analyze engagement state, classify attack phases, recommend tactical next actions, parse tool outputs, and provide structured JSON reasoning. Be concise, precise, and always output valid JSON when requested. Focus on evidence-driven analysis.\"\"\"

TEMPLATE \"\"\"{{{{- if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{- end }}}}
<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
{{{{ .Response }}}}<|im_end|>\"\"\"
"""
    with open(modelfile, "w") as f:
        f.write(content)
    log.info(f"Modelfile: {modelfile}")
    return modelfile


def main():
    parser = argparse.ArgumentParser(description="Disk-efficient GGUF export using /dev/shm staging")
    parser.add_argument("--adapter", required=True, help="Path to final LoRA adapter")
    parser.add_argument("--vlm-name", default="Qwen/Qwen3.5-4B", help="HuggingFace VLM to extract text backbone from")
    parser.add_argument("--quantizations", nargs="+", default=["Q6_K", "Q8_0"])
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("Ariaska GGUF Export V3 (shm-staged)")
    log.info(f"  Adapter: {args.adapter}")
    log.info(f"  VLM: {args.vlm_name}")
    log.info(f"  Quantizations: {args.quantizations}")
    log.info(f"  Overlay free: {disk_free_mb('/')} MB")
    log.info(f"  /dev/shm free: {disk_free_mb('/dev/shm')} MB")
    log.info("=" * 60)

    try:
        # Phase 1: Extract + merge (all in /dev/shm)
        merged_dir = extract_and_merge(args.vlm_name, args.adapter)

        # Phase 2: Convert + quantize (output to overlay)
        gguf_files = convert_and_quantize(merged_dir, args.quantizations)

        # Phase 3: Create Modelfile
        if gguf_files:
            create_modelfile(gguf_files[0])

        log.info("=" * 60)
        log.info("EXPORT COMPLETE")
        for f in gguf_files:
            size_mb = os.path.getsize(f) // (1024 * 1024)
            log.info(f"  {os.path.basename(f)}: {size_mb} MB")
        log.info(f"  Overlay free: {disk_free_mb('/')} MB")
        log.info("=" * 60)
        log.info("Download: scp -P 35188 root@<gpu-ip>:/root/ariaska_finetune/outputs/gguf/*.gguf .")
        log.info("Import:   ollama create ariaska-qwen35 -f Modelfile_ariaska_qwen3")

    except Exception:
        log.exception("Export failed")
        cleanup_shm()
        raise
    finally:
        cleanup_shm()


if __name__ == "__main__":
    main()
