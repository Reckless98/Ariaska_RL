#!/usr/bin/env python3
"""Merge LoRA adapter + export to GGUF Q6_K for CPU inference.

Usage:
    python export_gguf_v2.py --adapter /root/ariaska_finetune/outputs/qwen3-4b-ariaska/final

Requires llama.cpp (cloned to /root/llama.cpp)
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ariaska.export")


def ensure_llama_cpp():
    """Clone and build llama.cpp if not present."""
    llama_dir = Path("/root/llama.cpp")
    if not llama_dir.exists():
        log.info("Cloning llama.cpp...")
        subprocess.run(
            ["git", "clone", "--depth=1", "https://github.com/ggerganov/llama.cpp", str(llama_dir)],
            check=True,
        )
    # Build if needed
    quantize_bin = llama_dir / "build" / "bin" / "llama-quantize"
    if not quantize_bin.exists():
        log.info("Building llama.cpp...")
        build_dir = llama_dir / "build"
        build_dir.mkdir(exist_ok=True)
        subprocess.run(["cmake", "..", "-DGGML_CUDA=ON"], cwd=build_dir, check=True)
        subprocess.run(["cmake", "--build", ".", "-j", str(os.cpu_count() or 4)], cwd=build_dir, check=True)
    return llama_dir


def merge_adapter(adapter_path: str, output_dir: str, base_model: str | None = None) -> str:
    """Merge LoRA adapter into base model, save as FP16."""
    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    merged_dir = os.path.join(output_dir, "merged_fp16")
    if os.path.exists(os.path.join(merged_dir, "config.json")):
        log.info(f"Merged model already exists at {merged_dir}, skipping merge")
        return merged_dir

    log.info(f"Loading adapter config from {adapter_path}")
    peft_cfg = PeftConfig.from_pretrained(adapter_path)
    base_name = base_model or peft_cfg.base_model_name_or_path
    log.info(f"Base model: {base_name}")

    log.info("Loading base model in FP16 on CPU...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_name,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )

    log.info("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    log.info("Merging adapter weights...")
    merged = model.merge_and_unload()

    # Free disk: delete text model now that weights are in memory
    if base_name and os.path.isdir(base_name) and "text-only" in base_name:
        log.info(f"Deleting text model dir to free disk: {base_name}")
        shutil.rmtree(base_name, ignore_errors=True)

    os.makedirs(merged_dir, exist_ok=True)
    log.info(f"Saving merged FP16 model to {merged_dir}")
    merged.save_pretrained(merged_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    tokenizer.save_pretrained(merged_dir)

    # Free memory
    del merged, model, base_model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    log.info(f"Merged model saved: {merged_dir}")
    return merged_dir


def convert_and_quantize(merged_dir: str, output_dir: str, quantizations: list[str], llama_dir: Path):
    """Convert HF model to GGUF and quantize."""
    os.makedirs(output_dir, exist_ok=True)

    convert_script = llama_dir / "convert_hf_to_gguf.py"
    quantize_bin = llama_dir / "build" / "bin" / "llama-quantize"

    # Step 1: Convert to F16 GGUF
    f16_gguf = os.path.join(output_dir, "ariaska-qwen3-4b-f16.gguf")
    if not os.path.exists(f16_gguf):
        log.info(f"Converting to F16 GGUF...")
        cmd = [
            sys.executable, str(convert_script),
            merged_dir,
            "--outfile", f16_gguf,
            "--outtype", "f16",
        ]
        log.info(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        # Free disk: delete merged FP16 safetensors after GGUF conversion
        log.info(f"Deleting merged_fp16 dir to free disk...")
        shutil.rmtree(merged_dir, ignore_errors=True)
    else:
        log.info(f"F16 GGUF already exists: {f16_gguf}")

    # Step 2: Quantize
    output_files = []
    for quant in quantizations:
        quant_file = os.path.join(output_dir, f"ariaska-qwen3-4b-{quant.lower().replace('_', '-')}.gguf")
        if os.path.exists(quant_file):
            log.info(f"Already exists: {quant_file}")
            output_files.append(quant_file)
            continue

        log.info(f"Quantizing to {quant}...")
        cmd = [str(quantize_bin), f16_gguf, quant_file, quant]
        subprocess.run(cmd, check=True)
        output_files.append(quant_file)
        size_mb = os.path.getsize(quant_file) / (1024 * 1024)
        log.info(f"  {quant}: {size_mb:.0f} MB")

    # Clean up F16 intermediate
    if output_files and os.path.exists(f16_gguf):
        log.info(f"Removing intermediate F16 GGUF...")
        os.remove(f16_gguf)

    return output_files


def create_modelfile(gguf_path: str, output_dir: str) -> str:
    """Create Ollama Modelfile for the fine-tuned model."""
    modelfile = os.path.join(output_dir, "Modelfile_ariaska_qwen3")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="Path to final adapter dir")
    parser.add_argument("--base-model", default=None, help="Override base model path (e.g. extracted text-only model)")
    parser.add_argument("--output-dir", default="/root/ariaska_finetune/outputs/gguf")
    parser.add_argument("--quantizations", nargs="+", default=["Q6_K", "Q8_0"])
    args = parser.parse_args()

    log.info("=== Ariaska GGUF Export V2 ===")
    log.info(f"Adapter: {args.adapter}")
    log.info(f"Quantizations: {args.quantizations}")

    # Ensure llama.cpp is built
    llama_dir = ensure_llama_cpp()

    # Merge adapter into base
    merged_dir = merge_adapter(args.adapter, args.output_dir, base_model=args.base_model)

    # Convert + quantize
    gguf_files = convert_and_quantize(merged_dir, args.output_dir, args.quantizations, llama_dir)

    # Create Modelfile for primary quantization
    if gguf_files:
        create_modelfile(gguf_files[0], args.output_dir)

    log.info("=" * 60)
    log.info("EXPORT COMPLETE")
    for f in gguf_files:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        log.info(f"  {os.path.basename(f)}: {size_mb:.0f} MB")
    log.info("=" * 60)
    log.info("To download: scp -P 35188 root@<gpu-ip>:/root/ariaska_finetune/outputs/gguf/*.gguf .")
    log.info("To import:   ollama create ariaska-qwen3 -f Modelfile_ariaska_qwen3")


if __name__ == "__main__":
    main()
