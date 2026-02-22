#!/usr/bin/env bash
# Phase 50: Bidirectional sync every 30 minutes
# Pulls GPU artifacts (traces, models, logs) and pushes local code updates.
# Usage: bash scripts/gpu_bidirectional_sync.sh
set -uo pipefail

# ── Config ──────────────────────────────────────────────────────────────
GPU_HOST="root@212.247.220.172"
GPU_PORT=25107
SSH_OPTS="-o ConnectTimeout=20 -o ServerAliveInterval=10 -o StrictHostKeyChecking=no"
RSYNC_SSH="ssh $SSH_OPTS -p $GPU_PORT"

LOCAL_DIR="/home/zer0/Projects/Ariaska_RL"
REMOTE_DIR="/root/Ariaska_RL"

SYNC_INTERVAL=1800     # 30 minutes
MAX_CYCLES=20          # 10 hours of syncing
LOCAL_LOG="$LOCAL_DIR/logs/bidirectional_sync.log"
mkdir -p "$(dirname "$LOCAL_LOG")"

# ── Helpers ─────────────────────────────────────────────────────────────
log() {
    local ts
    ts=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$ts] $*" | tee -a "$LOCAL_LOG"
}

# ── Pull: GPU → Local (artifacts, traces, models, logs) ────────────────
pull_from_gpu() {
    log "PULL: GPU → Local (artifacts, traces, models, logs)"

    # Pull training artifacts
    rsync -avz --ignore-existing \
        -e "$RSYNC_SSH" \
        "$GPU_HOST:$REMOTE_DIR/artifacts/" \
        "$LOCAL_DIR/artifacts/" 2>&1 | tail -3

    # Pull traces
    rsync -avz --ignore-existing \
        -e "$RSYNC_SSH" \
        "$GPU_HOST:$REMOTE_DIR/traces/" \
        "$LOCAL_DIR/traces/" 2>&1 | tail -3

    # Pull model checkpoints
    rsync -avz \
        --include='*.pt' --include='*.pth' --include='*/' --exclude='*' \
        -e "$RSYNC_SSH" \
        "$GPU_HOST:$REMOTE_DIR/models/" \
        "$LOCAL_DIR/models/" 2>&1 | tail -3

    # Pull logs
    rsync -avz --ignore-existing \
        -e "$RSYNC_SSH" \
        --include='*.log' --include='*.jsonl' --include='*/' --exclude='*' \
        "$GPU_HOST:$REMOTE_DIR/logs/" \
        "$LOCAL_DIR/logs/" 2>&1 | tail -3

    # Pull results
    rsync -avz --ignore-existing \
        -e "$RSYNC_SSH" \
        "$GPU_HOST:$REMOTE_DIR/results/" \
        "$LOCAL_DIR/results/" 2>&1 | tail -3

    # Pull postmortems
    rsync -avz --ignore-existing \
        -e "$RSYNC_SSH" \
        "$GPU_HOST:$REMOTE_DIR/postmortems/" \
        "$LOCAL_DIR/postmortems/" 2>&1 | tail -3

    log "PULL complete"
}

# ── Push: Local → GPU (code only) ──────────────────────────────────────
push_to_gpu() {
    log "PUSH: Local → GPU (code updates)"

    rsync -avz \
        --exclude='.venv' --exclude='__pycache__' --exclude='.git' \
        --exclude='models/' --exclude='data/knowledge_candidates_v2' \
        --exclude='data/knowledge_indices' --exclude='data/knowledge_repos' \
        --exclude='artifacts/' --exclude='cache/' --exclude='logs/' \
        --exclude='traces/' --exclude='postmortems/' --exclude='results/' \
        --exclude='runs/' --exclude='*.pyc' --exclude='.mypy_cache' \
        -e "$RSYNC_SSH" \
        "$LOCAL_DIR/" "$GPU_HOST:$REMOTE_DIR/" 2>&1 | tail -5

    log "PUSH complete"
}

# ── Main Loop ───────────────────────────────────────────────────────────
log "=========================================="
log "Bidirectional Sync — Phase 50"
log "Every ${SYNC_INTERVAL}s, max ${MAX_CYCLES} cycles ($(( SYNC_INTERVAL * MAX_CYCLES / 3600 ))h)"
log "Local: $LOCAL_DIR"
log "Remote: $GPU_HOST:$REMOTE_DIR"
log "=========================================="

for ((i=1; i<=MAX_CYCLES; i++)); do
    log "--- Sync cycle $i/$MAX_CYCLES ---"

    # Check connectivity first
    if ! ssh $SSH_OPTS -p $GPU_PORT $GPU_HOST "echo ok" >/dev/null 2>&1; then
        log "WARNING: GPU unreachable, skipping cycle"
        sleep "$SYNC_INTERVAL"
        continue
    fi

    # Pull first (get latest artifacts)
    pull_from_gpu

    # Auto-commit pulled artifacts locally
    cd "$LOCAL_DIR" || exit 1
    if [[ -n $(git status --porcelain artifacts/ traces/ models/ logs/ results/ postmortems/ 2>/dev/null) ]]; then
        git add artifacts/ traces/ models/ logs/ results/ postmortems/ 2>/dev/null
        git commit -m "sync: pull GPU artifacts $(date '+%H:%M')" --quiet 2>/dev/null || true
        log "Committed pulled artifacts"
    fi

    # Push code updates (if any new commits)
    push_to_gpu

    if [[ $i -lt $MAX_CYCLES ]]; then
        log "Next sync in ${SYNC_INTERVAL}s ($(( SYNC_INTERVAL / 60 ))m)"
        sleep "$SYNC_INTERVAL"
    fi
done

log "=========================================="
log "Sync complete ($MAX_CYCLES cycles done)"
log "=========================================="
