#!/usr/bin/env bash
# =============================================================================
# Auto-pull GPU results to local laptop
#
# Downloads checkpoints, results, traces, and distill_prep artifacts from
# the GPU box every 10 minutes.
#
# Usage:
#   chmod +x pull_from_gpu.sh
#   ./pull_from_gpu.sh            # One-shot pull
#   ./pull_from_gpu.sh --loop     # Continuous pull every 10 min
# =============================================================================
set -euo pipefail

GPU_HOST="root@212.247.220.172"
GPU_PORT=25107
GPU_REPO="/root/Ariaska_RL"
LOCAL_DIR="./gpu_pull/Ariaska_RL"
INTERVAL=600  # 10 minutes

do_pull() {
    echo "[$(date '+%H:%M:%S')] Pulling from GPU..."
    mkdir -p "$LOCAL_DIR"

    rsync -avz --progress \
        -e "ssh -p $GPU_PORT" \
        --include "models/***" \
        --include "results/***" \
        --include "traces/***" \
        --include "data/distill_prep/***" \
        --include "data/expert_trajectories/***" \
        --exclude "*" \
        "$GPU_HOST:$GPU_REPO/" "$LOCAL_DIR/"

    echo "[PULL OK] $(date)"
    echo ""

    # Show latest checkpoints
    if [ -d "$LOCAL_DIR/models/distilled" ]; then
        echo "Latest checkpoints:"
        ls -lhrt "$LOCAL_DIR/models/distilled/" 2>/dev/null | tail -5
    fi

    # Show latest results
    if [ -d "$LOCAL_DIR/results/h200_distill" ]; then
        echo ""
        echo "Latest results:"
        ls -lhrt "$LOCAL_DIR/results/h200_distill/" 2>/dev/null | tail -3
    fi
}

if [ "${1:-}" = "--loop" ]; then
    echo "Starting continuous pull loop (every ${INTERVAL}s)..."
    echo "Press Ctrl+C to stop."
    while true; do
        do_pull || echo "[PULL FAILED] $(date) — will retry"
        sleep "$INTERVAL"
    done
else
    do_pull
fi
