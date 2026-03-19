#!/bin/bash
# Setup script for GPU server fine-tuning environment
# Run this on the remote GPU server after uploading files
set -e

echo "=== Ariaska Fine-Tuning Environment Setup ==="

# Create a working directory
WORK_DIR=/root/ariaska_finetune
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

echo "[1/6] System packages..."
apt-get update -qq
apt-get install -y -qq git build-essential cmake python3-pip python3-venv 2>/dev/null || true

echo "[2/6] Python venv..."
python3 -m venv venv
source venv/bin/activate

echo "[3/6] PyTorch with CUDA..."
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "[4/6] Training dependencies..."
pip install transformers accelerate peft trl datasets bitsandbytes
pip install sentencepiece protobuf pyyaml flash-attn --no-build-isolation
pip install scipy einops

echo "[5/6] GGUF export tools (llama.cpp)..."
if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp
    cmake -B build
    cmake --build build --config Release -j$(nproc)
    cd ..
fi

echo "[6/6] Verifying GPU..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
else:
    print('WARNING: No GPU detected!')
"

echo ""
echo "=== Setup complete ==="
echo "Working directory: $WORK_DIR"
echo ""
echo "Next steps:"
echo "  1. Upload training data:  scp ariaska_ai/dataset/*.jsonl root@server:$WORK_DIR/dataset/"
echo "  2. Upload scripts:        scp -r ariaska_ai/scripts ariaska_ai/configs root@server:$WORK_DIR/"
echo "  3. Run training:          cd $WORK_DIR && source venv/bin/activate && python scripts/finetune.py --config configs/qlora_qwen35_4b.yaml"
echo "  4. Export GGUF:           python scripts/export_gguf.py --adapter-path outputs/qwen35-4b-ariaska/final"
