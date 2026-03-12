#!/bin/bash
# scripts/vastai_setup.sh
# Phase 44: Automated deployment script for Vast.ai (Ubuntu-based images)
# Sets up environment to run Ariaska local LLM stack on a rented GPU (e.g. RTX 3090, 4090)

set -e

echo "==============================================="
echo "   Ariaska RL - Vast.ai GPU Setup Script       "
echo "==============================================="

# 1. Update and install system dependencies
echo "[1] Updating system dependencies..."
apt-get update -y
apt-get install -y python3-pip python3-venv git curl wget htop tmux build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev

# 2. Setup Python Virtual Environment
echo "[2] Setting up Python virtual environment..."
python3 -m venv /workspace/ariaska_venv
source /workspace/ariaska_venv/bin/activate
pip install --upgrade pip

# 3. Clone Repository (if not already synced)
if [ ! -d "/workspace/Ariaska_RL" ]; then
    echo "[3] Cloning Ariaska_RL repository..."
    # Note: replace with actual repo URL if tracking via git, or rsync from local
    echo "Warning: /workspace/Ariaska_RL not found. Please sync your code."
else
    echo "[3] Found Ariaska_RL in /workspace."
    cd /workspace/Ariaska_RL
    
    # 4. Install ML/GPU Dependencies
    echo "[4] Installing PyTorch and LLM Dependencies..."
    # Install PyTorch with CUDA 12.1 support (standard for current GPU hosts)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    # 5. Install Ariaska core requirements
    echo "[5] Installing Ariaska core requirements..."
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    fi
    
    # Install vLLM and llama-cpp-python with CUDA support
    echo "[6] Installing vLLM and llama-cpp-python (CUDA)..."
    pip install vllm
    CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir
    
    # 6. Download Model
    echo "[7] Downloading Local Model (Default: Qwen2.5-7B-Instruct)..."
    export HF_HOME="/workspace/huggingface_cache"
    python core/llm/model_manager.py download --model Qwen/Qwen2.5-7B-Instruct --format hf
fi

echo "==============================================="
echo "Setup Complete!"
echo "To run Ariaska:"
echo "1. source /workspace/ariaska_venv/bin/activate"
echo "2. cd /workspace/Ariaska_RL"
echo "3. FF_LOCAL_LLM=1 FF_LOCAL_LLM_OFFLOAD_ALL=1 python run_agent.py (or your entrypoint)"
echo "==============================================="
