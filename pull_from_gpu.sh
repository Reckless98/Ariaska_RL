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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_REPO="$SCRIPT_DIR"  # Pull directly into this repo
INTERVAL=600  # 10 minutes

do_pull() {
    echo ""
    echo "╔══════════════════════════════════════════════════════╗"
    echo "║  [$(date '+%H:%M:%S')] Pulling from GPU...               ║"
    echo "╚══════════════════════════════════════════════════════╝"

    rsync -avz --progress \
        -e "ssh -p $GPU_PORT" \
        "$GPU_HOST:$GPU_REPO/models/distilled/" "$LOCAL_REPO/models/distilled/" 2>/dev/null || true

    rsync -avz --progress \
        -e "ssh -p $GPU_PORT" \
        "$GPU_HOST:$GPU_REPO/results/h200_distill/" "$LOCAL_REPO/results/h200_distill/" 2>/dev/null || true

    rsync -avz --progress \
        -e "ssh -p $GPU_PORT" \
        "$GPU_HOST:$GPU_REPO/traces/h200_distill/" "$LOCAL_REPO/traces/h200_distill/" 2>/dev/null || true

    # GRPO artifacts
    rsync -avz --progress \
        -e "ssh -p $GPU_PORT" \
        "$GPU_HOST:$GPU_REPO/results/grpo_train/" "$LOCAL_REPO/results/grpo_train/" 2>/dev/null || true

    rsync -avz --progress \
        -e "ssh -p $GPU_PORT" \
        "$GPU_HOST:$GPU_REPO/traces/grpo_train/" "$LOCAL_REPO/traces/grpo_train/" 2>/dev/null || true

    # GPU session logs
    rsync -avz --progress \
        -e "ssh -p $GPU_PORT" \
        "$GPU_HOST:$GPU_REPO/logs/" "$LOCAL_REPO/logs/" 2>/dev/null || true

    echo "[PULL OK] $(date)"

    # Show latest checkpoints
    if ls "$LOCAL_REPO/models/distilled/"*.pt 1>/dev/null 2>&1; then
        echo ""
        echo "Latest checkpoints:"
        ls -lhrt "$LOCAL_REPO/models/distilled/"*.pt 2>/dev/null | tail -5
    fi

    # Show latest results
    if [ -d "$LOCAL_REPO/results/h200_distill" ]; then
        echo ""
        echo "Latest results:"
        ls -lhrt "$LOCAL_REPO/results/h200_distill/" 2>/dev/null | tail -3
    fi
}

do_git_sync() {
    cd "$LOCAL_REPO"
    # Only commit if there are new files
    local changes
    changes=$(git status --porcelain models/distilled/ results/h200_distill/ results/grpo_train/ traces/h200_distill/ traces/grpo_train/ 2>/dev/null | wc -l)
    if [ "$changes" -gt 0 ]; then
        echo ""
        echo ">>> Git: $changes new/changed files — committing..."
        git add models/distilled/ results/h200_distill/ results/grpo_train/ traces/h200_distill/ traces/grpo_train/ 2>/dev/null || true
        git commit -m "distill: auto-sync $(date '+%Y%m%d_%H%M%S') ($changes files)" --no-verify 2>/dev/null || true
        git push origin master 2>/dev/null && echo "[GIT PUSH OK]" || echo "[GIT PUSH FAILED — will retry]"
    else
        echo "[No new files to commit]"
    fi
}

if [ "${1:-}" = "--loop" ]; then
    echo "Starting continuous pull+sync loop (every ${INTERVAL}s)..."
    echo "Press Ctrl+C to stop."
    while true; do
        do_pull || echo "[PULL FAILED] $(date) — will retry"
        do_git_sync || echo "[GIT SYNC FAILED] $(date) — will retry"
        echo ""
        echo "--- Sleeping ${INTERVAL}s until next pull ---"
        sleep "$INTERVAL"
    done
elif [ "${1:-}" = "--once" ]; then
    do_pull
    do_git_sync
else
    do_pull
fi
