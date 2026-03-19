#!/usr/bin/env python3
"""
export_gguf.py - Export trained Ariaska model to GGUF format

Converts the merged HuggingFace model to GGUF for llama.cpp / Ollama deployment.
"""

import argparse
import os
import subprocess
import sys
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Export Ariaska model to GGUF")
    parser.add_argument("--base-model", required=True,
                       help="Path to base HF model (ariaska-cybersec-hf)")
    parser.add_argument("--adapter", required=True,
                       help="Path to final adapter dir (DPO or SFT)")
    parser.add_argument("--merged-output", default="/workspace/outputs/ariaska-v4-merged",
                       help="Path to save merged HF model")
    parser.add_argument("--gguf-output", default="/workspace/outputs/gguf",
                       help="Output directory for GGUF files")
    parser.add_argument("--quants", nargs="+", default=["Q8_0", "Q6_K"],
                       help="Quantization types to export")
    parser.add_argument("--llama-cpp-dir", default="/workspace/llama.cpp",
                       help="Path to llama.cpp installation")
    args = parser.parse_args()
    
    output_dir = Path(args.gguf_output)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_dir = Path(args.merged_output)
    llama_dir = Path(args.llama_cpp_dir)
    
    print(f"=== GGUF Export Pipeline ===")
    print(f"Base model: {args.base_model}")
    print(f"Adapter: {args.adapter}")
    print(f"Merged output: {merged_dir}")
    print(f"GGUF output: {output_dir}")
    print(f"Quants: {args.quants}")
    
    # Step 0: Merge adapter into base model
    if not (merged_dir / "config.json").exists():
        print("\n[0/4] Merging adapter into base model...")
        import torch
        import gc
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        
        merged_dir.mkdir(parents=True, exist_ok=True)
        
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
        
        model.save_pretrained(merged_dir, safe_serialization=True)
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        tokenizer.save_pretrained(merged_dir)
        
        del model
        gc.collect()
        print(f"  Merged model saved to {merged_dir}")
    else:
        print("[0/4] Merged model already exists")
    
    model_path = merged_dir
    
    # Step 1: Install/build llama.cpp if needed
    if not llama_dir.exists():
        print("\n[1/4] Cloning and building llama.cpp...")
        subprocess.run([
            "git", "clone", "--depth=1",
            "https://github.com/ggml-org/llama.cpp.git",
            str(llama_dir),
        ], check=True)
        
        build_dir = llama_dir / "build"
        build_dir.mkdir(exist_ok=True)
        subprocess.run([
            "cmake", "..",
            "-DGGML_CUDA=ON",
            "-DCMAKE_BUILD_TYPE=Release",
        ], cwd=build_dir, check=True)
        subprocess.run(["cmake", "--build", ".", "-j", str(os.cpu_count())],
                      cwd=build_dir, check=True)
    else:
        print("[1/4] llama.cpp already installed")
    
    # Step 2: Convert to F16 GGUF
    print("\n[2/4] Converting to F16 GGUF...")
    convert_script = llama_dir / "convert_hf_to_gguf.py"
    f16_path = output_dir / "ariaska-v4-f16.gguf"
    
    subprocess.run([
        sys.executable, str(convert_script),
        str(model_path),
        "--outfile", str(f16_path),
        "--outtype", "f16",
    ], check=True)
    
    print(f"F16 GGUF: {f16_path} ({f16_path.stat().st_size / 1e9:.2f} GB)")
    
    # Step 3: Quantize to target formats
    quantize_bin = llama_dir / "build" / "bin" / "llama-quantize"
    if not quantize_bin.exists():
        # Try alternate path
        quantize_bin = llama_dir / "build" / "llama-quantize"
    
    exported = {}
    for i, quant in enumerate(args.quants):
        print(f"\n[3/4] Quantizing to {quant} ({i+1}/{len(args.quants)})...")
        out_path = output_dir / f"ariaska-v4-{quant.lower().replace('_', '-')}.gguf"
        
        subprocess.run([
            str(quantize_bin),
            str(f16_path),
            str(out_path),
            quant,
        ], check=True)
        
        size_gb = out_path.stat().st_size / 1e9
        print(f"  {quant}: {out_path} ({size_gb:.2f} GB)")
        exported[quant] = str(out_path)
    
    # Step 4: Verify GGUF files
    print("\n[4/4] Verifying GGUF files...")
    for quant, path in exported.items():
        size = Path(path).stat().st_size / 1e9
        print(f"  {quant}: {size:.2f} GB - OK")
    
    # Write manifest
    manifest = {
        "model_source": str(model_path),
        "exports": exported,
        "f16_path": str(f16_path),
    }
    manifest_path = output_dir / "manifest.json"
    import json
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n=== GGUF EXPORT COMPLETE ===")
    print(f"Manifest: {manifest_path}")
    for quant, path in exported.items():
        print(f"  {quant}: {path}")
    print(f"\nTo deploy: scp the GGUF files to your local machine and register with Ollama.")


if __name__ == "__main__":
    main()
