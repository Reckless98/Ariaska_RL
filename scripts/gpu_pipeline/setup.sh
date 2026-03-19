#!/bin/bash
# GPU Server Setup Script for Ariaska Schema-Perfect Training Pipeline
# Target: vast.ai 48GB "4090" — ssh -p 19312 root@175.155.64.148
set -e

echo "=== ARIASKA GPU PIPELINE SETUP ==="
echo "$(date): Starting environment setup"

# Create workspace
mkdir -p /workspace/ariaska
mkdir -p /workspace/models
mkdir -p /workspace/data
mkdir -p /workspace/outputs
mkdir -p /workspace/checkpoints

# Install core ML packages
echo ">>> Installing PyTorch + core ML stack..."
pip3 install --upgrade pip
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip3 install transformers==4.48.0 datasets==3.2.0 accelerate==1.2.0
pip3 install peft==0.14.0 trl==0.13.0 bitsandbytes==0.45.0
pip3 install sentencepiece protobuf huggingface_hub
pip3 install vllm==0.6.6.post1
pip3 install jsonschema

# Verify CUDA
echo ">>> Verifying CUDA..."
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f}GB')"

# Download Qwen3-32B for teacher inference via vLLM
echo ">>> Downloading Qwen3-32B-AWQ teacher model (~17GB)..."
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'Qwen/Qwen3-32B-AWQ',
    local_dir='/workspace/models/qwen3-32b-awq',
    max_workers=8,
)
print('Teacher model downloaded successfully')
"

# Also download existing ariaska-cybersec 4B for training base
echo ">>> Downloading Qwen3.5-4B base for SFT..."
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'Qwen/Qwen3.5-4B',
    local_dir='/workspace/models/qwen3.5-4b',
    max_workers=8,
)
print('Base model downloaded successfully')
"

echo "=== SETUP COMPLETE ==="
echo "$(date): Ready for pipeline execution"
echo "Models in /workspace/models/"
echo "Data will go to /workspace/data/"
echo "Checkpoints to /workspace/checkpoints/"
