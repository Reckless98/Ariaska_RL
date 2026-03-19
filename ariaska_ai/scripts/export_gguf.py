#!/usr/bin/env python3
"""Merge LoRA adapter into base model and export to GGUF for CPU inference.

Usage:
    python ariaska_ai/scripts/export_gguf.py \
        --adapter-path ariaska_ai/outputs/qwen35-4b-ariaska/final \
        --output-dir ariaska_ai/outputs/gguf

Requires: llama-cpp-python (for convert/quantize) or llama.cpp repo
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ariaska.export_gguf")


def merge_adapter(adapter_path: str, output_dir: str) -> str:
    """Merge LoRA adapter weights into base model and save as full model."""
    log.info(f"Loading adapter from {adapter_path}")

    # Load adapter config to get base model name
    from peft import PeftConfig
    peft_config = PeftConfig.from_pretrained(adapter_path)
    base_model_name = peft_config.base_model_name_or_path
    log.info(f"Base model: {base_model_name}")

    # Load base model in fp16
    log.info("Loading base model in fp16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )

    # Load and merge adapter
    log.info("Merging LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    merged = model.merge_and_unload()

    # Save merged model
    merged_dir = os.path.join(output_dir, "merged_fp16")
    os.makedirs(merged_dir, exist_ok=True)
    log.info(f"Saving merged model to {merged_dir}")
    merged.save_pretrained(merged_dir, safe_serialization=True)

    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    tokenizer.save_pretrained(merged_dir)

    log.info(f"Merged model saved to {merged_dir}")
    return merged_dir


def convert_to_gguf(merged_dir: str, output_dir: str, quantizations: list[str]) -> list[str]:
    """Convert merged HuggingFace model to GGUF format using llama.cpp.

    Returns list of output GGUF file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_files = []

    # Try to find llama.cpp convert script
    convert_script = None
    for candidate in [
        "llama.cpp/convert_hf_to_gguf.py",
        os.path.expanduser("~/llama.cpp/convert_hf_to_gguf.py"),
        "/opt/llama.cpp/convert_hf_to_gguf.py",
    ]:
        if os.path.exists(candidate):
            convert_script = candidate
            break

    if not convert_script:
        log.warning("llama.cpp convert script not found locally. Trying pip install...")
        subprocess.run([sys.executable, "-m", "pip", "install", "llama-cpp-python"], check=True)
        # Fall back to using the Python package
        convert_script = "llama_cpp_convert"

    # Step 1: Convert to f16 GGUF
    f16_gguf = os.path.join(output_dir, "ariaska-qwen35-4b-f16.gguf")
    log.info(f"Converting to f16 GGUF: {f16_gguf}")

    if convert_script == "llama_cpp_convert":
        # Use the installed Python package approach
        log.info("Using transformers outtype conversion...")
        from transformers import AutoModelForCausalLM as AMCLM
        # For GGUF export we need llama.cpp tools
        log.error("llama.cpp tools required for GGUF conversion. Please install:")
        log.error("  git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make")
        log.error("Then re-run with the llama.cpp directory available")
        return []
    else:
        cmd = [
            sys.executable, convert_script,
            merged_dir,
            "--outfile", f16_gguf,
            "--outtype", "f16",
        ]
        log.info(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    # Step 2: Quantize to target formats
    # Find llama-quantize binary
    quantize_bin = None
    for candidate in [
        "llama.cpp/build/bin/llama-quantize",
        "llama.cpp/llama-quantize",
        os.path.expanduser("~/llama.cpp/build/bin/llama-quantize"),
        os.path.expanduser("~/llama.cpp/llama-quantize"),
        "/opt/llama.cpp/build/bin/llama-quantize",
        shutil.which("llama-quantize") or "",
    ]:
        if candidate and os.path.exists(candidate):
            quantize_bin = candidate
            break

    if not quantize_bin:
        log.warning("llama-quantize not found. Saving f16 GGUF only.")
        output_files.append(f16_gguf)
        return output_files

    for quant in quantizations:
        quant_file = os.path.join(output_dir, f"ariaska-qwen35-4b-{quant.lower()}.gguf")
        log.info(f"Quantizing to {quant}: {quant_file}")
        cmd = [quantize_bin, f16_gguf, quant_file, quant]
        log.info(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        output_files.append(quant_file)

    # Clean up f16 if quantized versions exist
    if output_files and os.path.exists(f16_gguf):
        log.info(f"Removing intermediate f16 GGUF ({f16_gguf})")
        os.remove(f16_gguf)

    return output_files


def create_modelfile(gguf_path: str, output_dir: str) -> str:
    """Create an Ollama Modelfile for the fine-tuned model."""
    modelfile_path = os.path.join(output_dir, "Modelfile_ariaska")
    gguf_basename = os.path.basename(gguf_path)

    content = f"""FROM ./{gguf_basename}

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 2048
PARAMETER repeat_penalty 1.1
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"

SYSTEM \"\"\"You are Ariaska, a cybersecurity AI coprocessor specialized in authorized penetration testing. You analyze engagement state, classify attack phases, recommend tactical actions, parse tool outputs, and provide structured reasoning for offensive security operations. You operate within authorized lab environments only. Be concise, precise, and always output in the requested format.\"\"\"

TEMPLATE \"\"\"{{{{- if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{- end }}}}
<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
{{{{ .Response }}}}<|im_end|>\"\"\"
"""
    with open(modelfile_path, "w") as f:
        f.write(content)

    log.info(f"Ollama Modelfile written to {modelfile_path}")
    return modelfile_path


def main():
    parser = argparse.ArgumentParser(description="Export fine-tuned Qwen3.5-4B to GGUF")
    parser.add_argument("--adapter-path", required=True, help="Path to trained LoRA adapter")
    parser.add_argument("--output-dir", default="ariaska_ai/outputs/gguf")
    parser.add_argument("--quantizations", nargs="+", default=["Q4_K_M", "Q5_K_M"])
    parser.add_argument("--skip-merge", action="store_true", help="Skip merge, use pre-merged model")
    parser.add_argument("--merged-dir", type=str, help="Pre-merged model directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Merge adapter
    if args.skip_merge and args.merged_dir:
        merged_dir = args.merged_dir
    else:
        merged_dir = merge_adapter(args.adapter_path, args.output_dir)

    # Convert to GGUF
    gguf_files = convert_to_gguf(merged_dir, args.output_dir, args.quantizations)

    if gguf_files:
        log.info(f"GGUF files created:")
        for gf in gguf_files:
            size_mb = os.path.getsize(gf) / (1024 * 1024)
            log.info(f"  {gf} ({size_mb:.0f} MB)")

        # Create Ollama Modelfile for Q4_K_M (primary)
        primary = next((f for f in gguf_files if "q4_k_m" in f.lower()), gguf_files[0])
        create_modelfile(primary, args.output_dir)

        log.info("\nTo load in Ollama:")
        log.info(f"  cd {args.output_dir}")
        log.info(f"  ollama create ariaska-cybersec -f Modelfile_ariaska")
        log.info(f"  ollama run ariaska-cybersec")
    else:
        log.warning("No GGUF files produced. Check llama.cpp installation.")
        log.info(f"Merged model available at: {merged_dir}")
        log.info("You can manually convert with:")
        log.info(f"  python llama.cpp/convert_hf_to_gguf.py {merged_dir} --outfile model.gguf --outtype f16")
        log.info(f"  llama-quantize model.gguf model-q4_k_m.gguf Q4_K_M")


if __name__ == "__main__":
    main()
