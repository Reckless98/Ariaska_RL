#!/usr/bin/env bash
# deploy_vast.sh — Deploy ariaska-cybersec3 training to vast.ai RTX 5090
#
# Base model pulled from HF (zer0aria/ariaska-cybersec2) — no slow SCP needed.
# Output uploaded to HF (zer0aria/ariaska-cybersec3) — download locally from there.
#
# Usage:
#   ./scripts/gpu_pipeline/deploy_vast.sh           # Full: upload scripts + install + train
#   ./scripts/gpu_pipeline/deploy_vast.sh --upload   # Upload scripts only
#   ./scripts/gpu_pipeline/deploy_vast.sh --train    # Start training (scripts already there)
#
# Server: ssh -p 34652 root@1.193.136.196 -L 8080:localhost:8080

set -euo pipefail

HOST="root@1.193.136.196"
PORT="34652"
SSH="ssh -o StrictHostKeyChecking=no -p $PORT $HOST"
SCP="scp -o StrictHostKeyChecking=no -P $PORT"

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
PIPELINE_DIR="$PROJECT_DIR/scripts/gpu_pipeline"

# HuggingFace repos
HF_BASE_REPO="zer0aria/ariaska-cybersec2"
HF_OUTPUT_REPO="zer0aria/ariaska-cybersec3"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

step() { echo -e "${GREEN}[$(date +%H:%M:%S)] $1${NC}"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] $1${NC}"; }
info() { echo -e "${CYAN}[$(date +%H:%M:%S)] $1${NC}"; }

# ── Parse args ────────────────────────────────────────────────────────────
MODE="all"
[[ "${1:-}" == "--upload" ]] && MODE="upload"
[[ "${1:-}" == "--train" ]] && MODE="train"

# ── Load API keys from .env (grep only the keys we need — .env has non-bash lines) ──
if [[ -f "$PROJECT_DIR/.env" ]]; then
    eval "$(grep -E '^(GROQ_API_KEY|OPENROUTER_API_KEY|TOGETHER_API_KEY|HF_TOKEN)=' "$PROJECT_DIR/.env")"
    export GROQ_API_KEY OPENROUTER_API_KEY TOGETHER_API_KEY HF_TOKEN 2>/dev/null || true
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN not set in .env — needed for model download/upload"
    exit 1
fi

# ── Upload scripts + data ─────────────────────────────────────────────────
if [[ "$MODE" == "all" || "$MODE" == "upload" ]]; then
    step "Creating workspace directories..."
    $SSH "mkdir -p /workspace/{models/ariaska-cybersec2-fp16,data/htb_walkthroughs,scripts,output}"

    step "Uploading training scripts (2 files)..."
    $SCP "$PIPELINE_DIR/ariaska_v5_htb_training.py" \
         "$PIPELINE_DIR/generate_htb_data.py" \
         "$HOST:/workspace/scripts/"

    step "Uploading schema data (28MB)..."
    $SCP "$PROJECT_DIR/data/ariaska_schema_sft_v4.jsonl" \
         "$HOST:/workspace/data/"

    step "Uploading HTB walkthroughs..."
    $SCP "$PROJECT_DIR/data/htb_walkthroughs/"*.md \
         "$HOST:/workspace/data/htb_walkthroughs/" 2>/dev/null || true

    step "Scripts uploaded!"
fi

if [[ "$MODE" == "upload" ]]; then
    echo ""
    step "Upload complete. Run with --train to start."
    exit 0
fi

# ── Install dependencies + pull base model from HF ───────────────────────
if [[ "$MODE" == "all" || "$MODE" == "train" ]]; then
    step "Installing Python dependencies..."
    $SSH "pip install -q torch transformers peft trl datasets accelerate \
          bitsandbytes flash-attn huggingface_hub openai 2>&1 | tail -5"

    step "Setting up llama.cpp for GGUF conversion..."
    $SSH "if [ ! -d /workspace/llama.cpp ]; then
        cd /workspace && git clone --depth 1 https://github.com/ggerganov/llama.cpp.git
        cd llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build -j\$(nproc) --target llama-quantize
    fi" 2>&1 | tail -5

    # ── HuggingFace login (needed for private repos) ──────────────────────
    step "Logging into HuggingFace..."
    $SSH "huggingface-cli login --token $HF_TOKEN --add-to-git-credential"

    # ── Pull base model from HF (datacenter speed >> SCP) ─────────────────
    step "Pulling FP16 base model from HF: $HF_BASE_REPO ..."
    $SSH "if [ ! -f /workspace/models/ariaska-cybersec2-fp16/config.json ]; then
        huggingface-cli download $HF_BASE_REPO \
            --include 'fp16-merged/*' \
            --local-dir /workspace/models/_hf_download \
            --local-dir-use-symlinks False
        mv /workspace/models/_hf_download/fp16-merged/* /workspace/models/ariaska-cybersec2-fp16/
        rm -rf /workspace/models/_hf_download
        echo 'Base model downloaded!'
    else
        echo 'Base model already present — skipping download'
    fi"

    # ── Set API keys on server ────────────────────────────────────────────
    step "Configuring API keys..."
    $SSH "cat > /workspace/.env << ENVEOF
export GROQ_API_KEY=${GROQ_API_KEY:-}
export OPENROUTER_API_KEY=${OPENROUTER_API_KEY:-}
export TOGETHER_API_KEY=${TOGETHER_API_KEY:-}
export HF_TOKEN=${HF_TOKEN}
ENVEOF
"

    # ── Start training in tmux ────────────────────────────────────────────
    step "Starting training pipeline in tmux..."
    $SSH "tmux kill-session -t train 2>/dev/null || true"
    $SSH "tmux new-session -d -s train 'cd /workspace && source .env && \
        python scripts/ariaska_v5_htb_training.py \
            --all \
            --gpu 5090 \
            --cloud-enhance \
            --cloud-enhance-count 10000 \
            --sft-count 15000 \
            --dpo-count 2500 \
            --hf-upload \
            --hf-repo $HF_OUTPUT_REPO \
        2>&1 | tee /workspace/v5_training.log; exec bash'"

    echo ""
    step "Training started in tmux session 'train'!"
    echo ""
    info "  Monitor:  ssh -p $PORT $HOST 'tmux attach -t train'"
    info "  Logs:     ssh -p $PORT $HOST 'tail -f /workspace/v5_training.log'"
    echo ""
    echo "  Pipeline (RTX 5090 32GB):"
    echo "    Stage 0: Data gen + cloud KD (10K × Groq+OpenRouter 70B)  ~2-3h"
    echo "    Stage 1: LoRA SFT (r=64, bs=4, 3 epochs, 15K examples)   ~2h"
    echo "    Stage 2: DPO alignment (r=32, 2 epochs, 2.5K pairs)      ~1h"
    echo "    Stage 3: Merge → FP16 + GGUF Q8_0                        ~15min"
    echo "    Stage 4: Upload to HuggingFace ($HF_OUTPUT_REPO)          ~15min"
    echo "    Total:                                                     ~5-6h"
    echo ""
    echo "  When done, download locally:"
    echo "    huggingface-cli download $HF_OUTPUT_REPO --local-dir ~/models/ariaska-cybersec3"
    echo ""
    echo "  Then register in Ollama:"
    echo "    ollama create ariaska-cybersec3 -f Modelfile_ariaska_cybersec3"
fi
