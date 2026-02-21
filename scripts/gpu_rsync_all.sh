#!/usr/bin/env bash
# gpu_rsync_all.sh — Comprehensive rsync from GPU to local every 10 minutes
# Downloads ALL training data: models, traces, results, logs, postmortems
# Usage: nohup bash scripts/gpu_rsync_all.sh &

set -euo pipefail

GPU_HOST="root@212.247.220.172"
GPU_PORT="25107"
GPU_ROOT="/root/Ariaska_RL"
LOCAL_ROOT="/home/zer0/Projects/Ariaska_RL"
SSH_OPTS="-o ConnectTimeout=20 -o ServerAliveInterval=10 -o StrictHostKeyChecking=no"
RSYNC_SSH="ssh ${SSH_OPTS} -p ${GPU_PORT}"
INTERVAL=600  # 10 minutes

LOG_FILE="${LOCAL_ROOT}/logs/gpu_rsync.log"
mkdir -p "${LOCAL_ROOT}/logs"
mkdir -p "${LOCAL_ROOT}/models/distilled"
mkdir -p "${LOCAL_ROOT}/models/enhanced"
mkdir -p "${LOCAL_ROOT}/models/unified"
mkdir -p "${LOCAL_ROOT}/models/progress"
mkdir -p "${LOCAL_ROOT}/traces/h200_distill"
mkdir -p "${LOCAL_ROOT}/results/h200_distill"
mkdir -p "${LOCAL_ROOT}/postmortems"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

sync_one() {
    local label="$1" src="$2" dst="$3"
    log "  Syncing ${label}..."
    if rsync -avz --progress --partial \
        -e "${RSYNC_SSH}" \
        "${GPU_HOST}:${src}" "${dst}" >> "$LOG_FILE" 2>&1; then
        log "  ✓ ${label} done"
    else
        log "  ✗ ${label} FAILED (rc=$?)"
    fi
}

log "=========================================="
log "GPU Rsync All — Starting loop (every ${INTERVAL}s)"
log "GPU: ${GPU_HOST}:${GPU_PORT}"
log "=========================================="

CYCLE=0
while true; do
    CYCLE=$((CYCLE + 1))
    log ""
    log "--- Cycle ${CYCLE} at $(date '+%H:%M:%S') ---"

    # 1) MODEL CHECKPOINTS — the most important
    sync_one "models/distilled (*.pt)" \
        "${GPU_ROOT}/models/distilled/" \
        "${LOCAL_ROOT}/models/distilled/"

    # 1b) UNIFIED CHECKPOINTS (priority format)
    sync_one "models/unified" \
        "${GPU_ROOT}/models/unified/" \
        "${LOCAL_ROOT}/models/unified/"

    # 2) ENHANCED MODELS (agent snapshots + metrics)
    sync_one "models/enhanced" \
        "${GPU_ROOT}/models/enhanced/" \
        "${LOCAL_ROOT}/models/enhanced/"

    # 3) PROGRESS LABELS
    sync_one "models/progress" \
        "${GPU_ROOT}/models/progress/" \
        "${LOCAL_ROOT}/models/progress/"

    # 4) TRACES (JSONL episode traces)
    sync_one "traces/h200_distill" \
        "${GPU_ROOT}/traces/h200_distill/" \
        "${LOCAL_ROOT}/traces/h200_distill/"

    # 5) TRACE RUN DIRS (per-run trace subdirs)
    sync_one "traces (run dirs)" \
        "${GPU_ROOT}/traces/" \
        "${LOCAL_ROOT}/traces/"

    # 6) RESULTS (reports)
    sync_one "results/h200_distill" \
        "${GPU_ROOT}/results/h200_distill/" \
        "${LOCAL_ROOT}/results/h200_distill/"

    # 7) POSTMORTEMS
    sync_one "postmortems" \
        "${GPU_ROOT}/postmortems/" \
        "${LOCAL_ROOT}/postmortems/"

    # 8) TRAINING LOGS
    log "  Syncing training logs..."
    rsync -avz -e "${RSYNC_SSH}" \
        "${GPU_HOST}:/root/distill_*.log" \
        "${LOCAL_ROOT}/logs/" >> "$LOG_FILE" 2>&1 || true
    log "  ✓ logs done"

    # 9) RESULTS (top-level eval, grpo, cap)
    sync_one "results (all)" \
        "${GPU_ROOT}/results/" \
        "${LOCAL_ROOT}/results/"

    # Summary
    LOCAL_MODELS=$(find "${LOCAL_ROOT}/models/" -name "*.pt" -type f 2>/dev/null | wc -l)
    LOCAL_TRACES=$(find "${LOCAL_ROOT}/traces/h200_distill/" -name "*.jsonl" -type f 2>/dev/null | wc -l)
    LOCAL_RESULTS=$(find "${LOCAL_ROOT}/results/h200_distill/" -name "*.json" -type f 2>/dev/null | wc -l)
    LOCAL_POSTMORTEMS=$(find "${LOCAL_ROOT}/postmortems/" -type f 2>/dev/null | wc -l)
    log "  Summary: ${LOCAL_MODELS} models, ${LOCAL_TRACES} traces, ${LOCAL_RESULTS} results, ${LOCAL_POSTMORTEMS} postmortems"

    log "--- Cycle ${CYCLE} complete. Sleeping ${INTERVAL}s ---"
    sleep "$INTERVAL"
done
