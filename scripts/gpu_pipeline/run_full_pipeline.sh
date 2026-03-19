#!/usr/bin/env bash
# run_full_pipeline.sh — Complete Ariaska Schema Training Pipeline
# Preserves ariaska-cybersec knowledge + adds 32B teacher schema compliance
#
# Usage: bash /workspace/scripts/run_full_pipeline.sh
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/workspace:${PYTHONPATH:-}"
export HF_HOME="/workspace/hf_cache"
export TRANSFORMERS_CACHE="/workspace/hf_cache"

SCRIPTS="/workspace/scripts"
MODELS="/workspace/models"
DATA="/workspace/data"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ═══════════════════════════════════════════════════════════════════════════
# STAGE 0: Environment + Model Setup
# ═══════════════════════════════════════════════════════════════════════════
log "=== STAGE 0: Environment Setup ==="

# Install any missing packages
pip install -q gguf sentencepiece protobuf 2>/dev/null || true

# Create directories
mkdir -p "$MODELS" "$DATA" /workspace/outputs /workspace/hf_cache

# Download Qwen3-32B-AWQ teacher if not present
TEACHER_DIR="$MODELS/qwen3-32b-awq"
if [ ! -d "$TEACHER_DIR" ] || [ ! -f "$TEACHER_DIR/config.json" ]; then
    log "Downloading Qwen3-32B-AWQ teacher (~17GB)..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-32B-AWQ', local_dir='/workspace/models/qwen3-32b-awq')
print('Teacher download complete')
"
else
    log "Teacher model already present at $TEACHER_DIR"
fi

# ═══════════════════════════════════════════════════════════════════════════
# STAGE 1: Convert GGUF → HuggingFace 
# ═══════════════════════════════════════════════════════════════════════════
HF_BASE="$MODELS/ariaska-cybersec-hf"
if [ ! -f "$HF_BASE/config.json" ]; then
    log "=== STAGE 1: Converting GGUF → HuggingFace ==="
    python3 "$SCRIPTS/convert_gguf_to_hf.py"
else
    log "=== STAGE 1: SKIP (HF model already exists) ==="
fi

# Verify HF model
if [ ! -f "$HF_BASE/config.json" ]; then
    log "ERROR: HF conversion failed. No config.json at $HF_BASE"
    exit 1
fi
log "Base model: $HF_BASE"

# ═══════════════════════════════════════════════════════════════════════════
# STAGE 2: Generate Schema-Perfect Training Data (32B Teacher)
# ═══════════════════════════════════════════════════════════════════════════
SCHEMA_DATA="$DATA/ariaska_schema_sft.jsonl"
if [ ! -f "$SCHEMA_DATA" ] || [ "$(wc -l < "$SCHEMA_DATA")" -lt 1000 ]; then
    log "=== STAGE 2: Generating Schema Training Data ==="
    python3 "$SCRIPTS/generate_schema_data.py" \
        --teacher-model "$TEACHER_DIR" \
        --output "$SCHEMA_DATA" \
        --num-examples 12000
else
    COUNT=$(wc -l < "$SCHEMA_DATA")
    log "=== STAGE 2: SKIP ($COUNT examples already generated) ==="
fi

# Verify data
if [ ! -f "$SCHEMA_DATA" ]; then
    log "ERROR: Data generation failed"
    exit 1
fi
DATA_COUNT=$(wc -l < "$SCHEMA_DATA")
log "Schema training data: $DATA_COUNT examples"

# ═══════════════════════════════════════════════════════════════════════════
# STAGE 3: Schema SFT on ariaska-cybersec base
# ═══════════════════════════════════════════════════════════════════════════
SFT_OUTPUT="/workspace/outputs/schema_sft"
SFT_FINAL="$SFT_OUTPUT/final"
if [ ! -f "$SFT_FINAL/adapter_config.json" ]; then
    log "=== STAGE 3: Schema SFT Training ==="
    python3 "$SCRIPTS/train_schema_sft.py" \
        --base-model "$HF_BASE" \
        --data "$SCHEMA_DATA" \
        --output "$SFT_OUTPUT"
else
    log "=== STAGE 3: SKIP (SFT adapter exists) ==="
fi

# ═══════════════════════════════════════════════════════════════════════════
# STAGE 4: DPO Alignment  
# ═══════════════════════════════════════════════════════════════════════════
DPO_OUTPUT="/workspace/outputs/schema_dpo"
DPO_FINAL="$DPO_OUTPUT/final"
if [ ! -f "$DPO_FINAL/adapter_config.json" ]; then
    log "=== STAGE 4: DPO Alignment ==="
    python3 "$SCRIPTS/train_dpo.py" \
        --base-model "$HF_BASE" \
        --sft-adapter "$SFT_FINAL" \
        --teacher-model "$TEACHER_DIR" \
        --output "$DPO_OUTPUT"
else
    log "=== STAGE 4: SKIP (DPO adapter exists) ==="
fi

# ═══════════════════════════════════════════════════════════════════════════
# STAGE 5: Merge + Export GGUF
# ═══════════════════════════════════════════════════════════════════════════
MERGED="/workspace/outputs/ariaska-v4-merged"
GGUF_DIR="/workspace/outputs/gguf"
log "=== STAGE 5: Merge + GGUF Export ==="
python3 "$SCRIPTS/export_gguf.py" \
    --base-model "$HF_BASE" \
    --adapter "$DPO_FINAL" \
    --merged-output "$MERGED" \
    --gguf-output "$GGUF_DIR"

# ═══════════════════════════════════════════════════════════════════════════
# STAGE 6: Validate
# ═══════════════════════════════════════════════════════════════════════════
log "=== STAGE 6: Validation ==="
log "GGUF files:"
ls -lh "$GGUF_DIR/"*.gguf 2>/dev/null || log "No GGUF files found!"

log ""
log "=== PIPELINE COMPLETE ==="
log "Models at: $GGUF_DIR/"
log "Transfer with: scp -P 19312 root@175.155.64.148:$GGUF_DIR/ariaska-v4-q8_0.gguf ."
