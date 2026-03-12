#!/bin/bash
# scripts/ollama_setup.sh
# Phase 44: Automated deployment script for Ollama on CPU/iGPU laptops

set -e

echo "==============================================="
echo "   Ariaska RL - Ollama CPU/iGPU Setup Script   "
echo "==============================================="

if ! command -v ollama &> /dev/null; then
    echo "[1] Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "[1] Ollama is already installed."
fi

echo "[2] Ensuring Ollama service is running..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo systemctl enable ollama || true
    sudo systemctl start ollama || true
elif [[ "$OSTYPE" == "darwin"* ]]; then
    brew services start ollama || true
fi

echo "[3] Configuring Ollama for Dual-Model Pipeline (System 1 + System 2)..."
export OLLAMA_MAX_LOADED_MODELS=2
# Make it persistent for systemd (if applicable)
if [[ "$OSTYPE" == "linux-gnu"* ]] && command -v systemctl &> /dev/null; then
    sudo sh -c 'echo "Environment=\"OLLAMA_MAX_LOADED_MODELS=2\"" >> /etc/systemd/system/ollama.service' || true
    sudo systemctl daemon-reload || true
    sudo systemctl restart ollama || true
fi

echo "[4] Pulling optimal high-speed models for 24GB RAM CPU (qwen2.5-coder:3b, pki/nova-24b-cybersec)..."
ollama pull qwen2.5-coder:3b || true
ollama pull pki/nova-24b-cybersec || true

echo "==============================================="
echo "Setup Complete!"
echo "To run Ariaska:"
echo "1. Activate your virtual environment (if any)"
echo "2. cd /home/zer0/Projects/Ariaska_RL"
echo "3. FF_LOCAL_LLM=1 FF_LOCAL_LLM_OFFLOAD_ALL=1 python run_agent.py (or your entrypoint)"
echo "Ariaska will auto-detect the lack of an NVIDIA GPU and natively route traffic through Ollama."
echo "==============================================="
