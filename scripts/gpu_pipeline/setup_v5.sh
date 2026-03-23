#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# V5 HTB Training — GPU Server Setup
# Target: vast.ai RTX 4090 (24GB VRAM) — ~$0.30-0.50/hr
#
# Two-track LoRA training:
#   Track A: ariaska-cybersec2 FP16 → ariaska-cybersec3 (4B, LoRA r=64)
#   Track B: Qwen2.5-7B-Instruct → ariaska-strategic-9b (7B, LoRA r=128)
#
# Usage:
#   scp setup_v5.sh root@<ip>:/workspace/
#   ssh root@<ip> 'bash /workspace/setup_v5.sh'
#   scp -r data/ scripts/gpu_pipeline/ root@<ip>:/workspace/ariaska/
#   ssh root@<ip> 'cd /workspace/ariaska && python scripts/gpu_pipeline/ariaska_v5_htb_training.py --all'
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

echo "═══════════════════════════════════════════════════════"
echo " ARIASKA V5 HTB Training — GPU Setup"
echo " $(date)"
echo "═══════════════════════════════════════════════════════"

# ── Directory structure ──────────────────────────────────────────────────────
mkdir -p /workspace/{models,data,output,ariaska}
mkdir -p /workspace/output/{cybersec3-lora,cybersec3-merged,cybersec3-gguf}
mkdir -p /workspace/output/{strategic-sft,strategic-merged,strategic-gguf}

# ── Core ML stack ────────────────────────────────────────────────────────────
echo ">>> [1/5] Installing PyTorch + training stack..."
pip3 install --upgrade pip setuptools wheel 2>&1 | tail -1

# PyTorch with CUDA 12.4
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -3

# Training: transformers + peft + trl (LoRA SFT)
pip3 install \
    "transformers>=4.48.0" \
    "datasets>=3.2.0" \
    "accelerate>=1.2.0" \
    "peft>=0.14.0" \
    "trl>=0.13.0" \
    "bitsandbytes>=0.45.0" \
    sentencepiece protobuf huggingface_hub \
    2>&1 | tail -3

# Flash Attention 2 (2x faster training on 4090)
echo ">>> [2/5] Installing Flash Attention 2..."
pip3 install flash-attn --no-build-isolation 2>&1 | tail -3 || {
    echo "WARN: flash-attn failed (non-fatal — training will use sdpa fallback)"
}

# ── CUDA verification ────────────────────────────────────────────────────────
echo ">>> [3/5] Verifying CUDA..."
python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
dev = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_mem / 1e9
print(f'  GPU: {dev} | VRAM: {vram:.1f}GB | CUDA: {torch.version.cuda}')
assert vram >= 20, f'Need 24GB+ VRAM, got {vram:.1f}GB'
print('  CUDA OK')
"

# ── llama.cpp for GGUF conversion ────────────────────────────────────────────
echo ">>> [4/5] Building llama.cpp for GGUF conversion..."
if [ ! -f /workspace/llama.cpp/build/bin/llama-quantize ]; then
    cd /workspace
    git clone --depth 1 https://github.com/ggerganov/llama.cpp.git 2>/dev/null || true
    cd llama.cpp
    pip3 install -r requirements.txt 2>&1 | tail -1
    cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 2>&1 | tail -3
    cmake --build build --config Release -j$(nproc) 2>&1 | tail -3
    echo "  llama.cpp built: $(ls build/bin/llama-quantize 2>/dev/null && echo 'OK' || echo 'FAIL')"
    cd /workspace
else
    echo "  llama.cpp already built, skipping"
fi

# ── Download models ──────────────────────────────────────────────────────────
echo ">>> [5/5] Downloading base models..."

# Track A: We need the FP16 of ariaska-cybersec2
# Option 1: Upload from local (scp the merged FP16 dir)
# Option 2: If on HuggingFace, download it
if [ ! -d /workspace/models/ariaska-cybersec2-fp16 ] || [ -z "$(ls -A /workspace/models/ariaska-cybersec2-fp16 2>/dev/null)" ]; then
    echo "  Track A: ariaska-cybersec2 FP16 not found."
    echo "  Upload it: scp -r models/ariaska-cybersec2-merged/ root@<ip>:/workspace/models/ariaska-cybersec2-fp16/"
    echo "  Or convert from GGUF: python scripts/gpu_pipeline/convert_gguf_to_hf.py"
    echo "  (Continuing — Track A will fail without this model)"
fi

# Track B: Download Qwen2.5-7B-Instruct (strategic reasoning base)
python3 -c "
from pathlib import Path
target = Path('/workspace/models/strategic-base')
if target.exists() and any(target.iterdir()):
    print('  Track B: strategic base already downloaded, skipping')
else:
    print('  Track B: Downloading Qwen2.5-7B-Instruct...')
    from huggingface_hub import snapshot_download
    snapshot_download(
        'Qwen/Qwen2.5-7B-Instruct',
        local_dir=str(target),
        max_workers=8,
    )
    print('  Track B: Base model ready')
"

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════"
echo " SETUP COMPLETE — $(date)"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  1. Upload project files:"
echo "     scp -r data/ariaska_schema_sft_v4.jsonl root@<ip>:/workspace/data/"
echo "     scp -r scripts/gpu_pipeline/ root@<ip>:/workspace/ariaska/scripts/gpu_pipeline/"
echo ""
echo "  2. Upload cybersec2 FP16 (for Track A):"
echo "     scp -r models/ariaska-cybersec2-merged/ root@<ip>:/workspace/models/ariaska-cybersec2-fp16/"
echo ""
echo "  3. Run training:"
echo "     cd /workspace/ariaska"
echo "     python scripts/gpu_pipeline/ariaska_v5_htb_training.py --all"
echo ""
echo "  4. Download results:"
echo "     scp root@<ip>:/workspace/output/cybersec3-gguf/*.gguf ./"
echo "     scp root@<ip>:/workspace/output/strategic-gguf/*.gguf ./"
echo ""
echo "Estimated time: ~6-8h total | Cost: ~\$3-4 on 4090"
