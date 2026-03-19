#!/bin/bash
# run_pipeline.sh — Master Ariaska Model Pipeline
# Orchestrates: setup → data gen → SFT → DPO → GGUF export
set -euo pipefail

WORKSPACE="/workspace"
MODELS_DIR="$WORKSPACE/models"
DATA_DIR="$WORKSPACE/data"
SCRIPTS_DIR="$WORKSPACE/scripts"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[$(date +%H:%M:%S)] $1${NC}"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] WARNING: $1${NC}"; }
err() { echo -e "${RED}[$(date +%H:%M:%S)] ERROR: $1${NC}"; exit 1; }

# ── Pre-flight checks ──────────────────────────────────────────────────────
log "=== ARIASKA MODEL PIPELINE ==="
log "Starting pre-flight checks..."

nvidia-smi > /dev/null 2>&1 || err "No GPU detected!"
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null || {
    log "PyTorch not installed or CUDA not available. Running setup first..."
    bash "$SCRIPTS_DIR/setup.sh"
}

mkdir -p "$MODELS_DIR" "$DATA_DIR" "$WORKSPACE/logs"

# ── Step 1: Download Models (if needed) ────────────────────────────────────
TEACHER_MODEL="$MODELS_DIR/qwen3-32b-awq"
BASE_MODEL="$MODELS_DIR/qwen3.5-4b"

if [ ! -d "$TEACHER_MODEL" ]; then
    log "Step 1a: Downloading teacher model (Qwen3-32B-AWQ)..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-32B-AWQ', local_dir='$TEACHER_MODEL', 
                  local_dir_use_symlinks=False)
print('Teacher model downloaded!')
"
else
    log "Step 1a: Teacher model already present."
fi

if [ ! -d "$BASE_MODEL" ]; then
    log "Step 1b: Downloading base model (Qwen3.5-4B)..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.5-4B', local_dir='$BASE_MODEL',
                  local_dir_use_symlinks=False)
print('Base model downloaded!')
"
else
    log "Step 1b: Base model already present."
fi

# ── Step 2: Generate Schema-Perfect Training Data ──────────────────────────
SFT_DATA="$DATA_DIR/ariaska_schema_sft.jsonl"

if [ ! -f "$SFT_DATA" ] || [ "$(wc -l < "$SFT_DATA")" -lt 1000 ]; then
    log "Step 2: Generating schema-perfect training data..."
    python3 "$SCRIPTS_DIR/generate_schema_data.py" \
        --model "$TEACHER_MODEL" \
        --output "$SFT_DATA" \
        --scale 1.0 \
        2>&1 | tee "$WORKSPACE/logs/datagen_$(date +%Y%m%d_%H%M%S).log"
    
    LINES=$(wc -l < "$SFT_DATA")
    log "Generated $LINES training examples."
    [ "$LINES" -lt 500 ] && err "Too few examples generated ($LINES). Check logs."
else
    LINES=$(wc -l < "$SFT_DATA")
    log "Step 2: Training data already present ($LINES examples). Skipping."
fi

# ── Step 3: Schema SFT Training ───────────────────────────────────────────
SFT_MODEL="$MODELS_DIR/ariaska-schema-sft"
SFT_MERGED="$MODELS_DIR/ariaska-schema-sft-merged"

if [ ! -d "$SFT_MERGED" ]; then
    log "Step 3: Running Schema SFT training..."
    python3 "$SCRIPTS_DIR/train_schema_sft.py" \
        --data "$SFT_DATA" \
        --base-model "$BASE_MODEL" \
        --output "$SFT_MODEL" \
        --epochs 3 \
        --lr 2e-5 \
        --batch-size 4 \
        --grad-accum 4 \
        --max-seq-len 2048 \
        --lora-r 32 \
        --lora-alpha 64 \
        2>&1 | tee "$WORKSPACE/logs/sft_$(date +%Y%m%d_%H%M%S).log"
    
    [ -d "$SFT_MERGED" ] || err "SFT training failed - merged model not found!"
    log "SFT training complete!"
else
    log "Step 3: SFT model already present. Skipping."
fi

# ── Step 4: DPO Training ──────────────────────────────────────────────────
DPO_DATA="$DATA_DIR/ariaska_dpo_pairs.jsonl"
DPO_MODEL="$MODELS_DIR/ariaska-schema-dpo"
DPO_MERGED="$MODELS_DIR/ariaska-schema-dpo-merged"

if [ ! -d "$DPO_MERGED" ]; then
    log "Step 4: Running DPO training..."
    python3 "$SCRIPTS_DIR/train_dpo.py" \
        --sft-model "$SFT_MERGED" \
        --sft-data "$SFT_DATA" \
        --dpo-data "$DPO_DATA" \
        --output "$DPO_MODEL" \
        --max-pairs 5000 \
        --epochs 1 \
        --lr 5e-7 \
        --beta 0.1 \
        --batch-size 2 \
        --grad-accum 8 \
        2>&1 | tee "$WORKSPACE/logs/dpo_$(date +%Y%m%d_%H%M%S).log"
    
    [ -d "$DPO_MERGED" ] || err "DPO training failed - merged model not found!"
    log "DPO training complete!"
else
    log "Step 4: DPO model already present. Skipping."
fi

# ── Step 5: GGUF Export ────────────────────────────────────────────────────
GGUF_DIR="$MODELS_DIR/gguf"

if [ ! -f "$GGUF_DIR/ariaska-schema-q8-0.gguf" ]; then
    log "Step 5: Exporting to GGUF..."
    python3 "$SCRIPTS_DIR/export_gguf.py" \
        --model "$DPO_MERGED" \
        --output-dir "$GGUF_DIR" \
        --quants Q8_0 Q6_K \
        2>&1 | tee "$WORKSPACE/logs/gguf_$(date +%Y%m%d_%H%M%S).log"
    
    [ -f "$GGUF_DIR/ariaska-schema-q8-0.gguf" ] || err "GGUF export failed!"
    log "GGUF export complete!"
else
    log "Step 5: GGUF files already present. Skipping."
fi

# ── Step 6: Summary ───────────────────────────────────────────────────────
log ""
log "=== PIPELINE COMPLETE ==="
log ""
log "Generated artifacts:"
ls -lh "$GGUF_DIR"/*.gguf 2>/dev/null || warn "No GGUF files found"
log ""
log "To download to local machine:"
log "  scp -P 19312 root@175.155.64.148:$GGUF_DIR/ariaska-schema-q8-0.gguf /tmp/"
log "  scp -P 19312 root@175.155.64.148:$GGUF_DIR/ariaska-schema-q6-k.gguf /tmp/"
log ""
log "To deploy to Ollama:"
log "  ollama create ariaska-cybersec -f Modelfile_ariaska_cybersec_new"
