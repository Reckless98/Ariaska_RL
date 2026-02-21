#!/usr/bin/env bash
# GPU Monitor + Rsync Loop — runs locally, syncs every 10 min
# Usage: bash scripts/gpu_monitor_loop.sh

set -euo pipefail

GPU_HOST="root@212.247.220.172"
GPU_PORT="25107"
GPU_DIR="/root/Ariaska_RL"
LOCAL_DIR="/home/zer0/Projects/Ariaska_RL"
LOG_FILE="/root/distill_6h.log"
SSH_OPTS="-o ConnectTimeout=15 -o ServerAliveInterval=10 -o StrictHostKeyChecking=no -p $GPU_PORT"

RSYNC_DIRS="models/distilled/ traces/ results/ logs/ postmortems/"

echo "[$(date)] GPU Monitor + Rsync loop started"

while true; do
    echo ""
    echo "============================================"
    echo "[$(date)] === RSYNC CYCLE ==="
    echo "============================================"

    # Rsync each directory
    for dir in $RSYNC_DIRS; do
        echo "[$(date)] Syncing $dir ..."
        rsync -avz --progress \
            -e "ssh $SSH_OPTS" \
            "$GPU_HOST:$GPU_DIR/$dir" \
            "$LOCAL_DIR/$dir" 2>/dev/null || echo "  WARN: rsync $dir failed, will retry next cycle"
    done

    # Also grab the log file
    echo "[$(date)] Syncing training log..."
    rsync -avz -e "ssh $SSH_OPTS" \
        "$GPU_HOST:$LOG_FILE" \
        "$LOCAL_DIR/logs/distill_6h.log" 2>/dev/null || echo "  WARN: log sync failed"

    echo "[$(date)] Rsync complete. Sleeping 600s..."
    sleep 600
done
