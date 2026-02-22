#!/usr/bin/env bash
# Phase 50: GPU training monitor + auto-relaunch script
# Checks training health hourly for 10 hours, relaunches if crashed.
# Usage: bash scripts/gpu_monitor_relaunch.sh
set -uo pipefail

# ── Config ──────────────────────────────────────────────────────────────
GPU_HOST="root@212.247.220.172"
GPU_PORT=25107
SSH_OPTS="-o ConnectTimeout=20 -o ServerAliveInterval=10 -o StrictHostKeyChecking=no"
SSH_CMD="ssh $SSH_OPTS -p $GPU_PORT $GPU_HOST"

TRAIN_TMUX="train_p50"
TRAIN_CMD="cd /root/Ariaska_RL && PYTHONPATH=/root/Ariaska_RL .venv/bin/python scripts/h200_run_distill_3h.py --max-hours 12 2>&1 | tee /root/Ariaska_RL/logs/train_p50.log"

CHECK_INTERVAL=3600  # 1 hour
MAX_CHECKS=10        # 10 hours total
STALL_THRESHOLD=30   # minutes without new episode = stalled

LOCAL_LOG="/home/zer0/Projects/Ariaska_RL/logs/gpu_monitor.log"
mkdir -p "$(dirname "$LOCAL_LOG")"

# ── Helpers ─────────────────────────────────────────────────────────────
log() {
    local ts
    ts=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$ts] $*" | tee -a "$LOCAL_LOG"
}

ssh_exec() {
    $SSH_CMD "$@" 2>/dev/null
}

is_training_running() {
    ssh_exec "tmux has-session -t $TRAIN_TMUX 2>/dev/null && echo 'yes' || echo 'no'"
}

get_gpu_stats() {
    ssh_exec "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null"
}

get_last_episode_age() {
    # Check how many minutes since last episode was logged
    ssh_exec "
        LOGFILE=/root/Ariaska_RL/logs/train_p50.log
        if [[ -f \$LOGFILE ]]; then
            LAST_EP=\$(grep -oP 'Episode \d+' \$LOGFILE | tail -1)
            if [[ -n \$LAST_EP ]]; then
                LAST_MOD=\$(stat -c %Y \$LOGFILE 2>/dev/null || echo 0)
                NOW=\$(date +%s)
                echo \$(( (NOW - LAST_MOD) / 60 ))
            else
                echo 999
            fi
        else
            echo 999
        fi
    "
}

get_training_progress() {
    ssh_exec "
        LOGFILE=/root/Ariaska_RL/logs/train_p50.log
        if [[ -f \$LOGFILE ]]; then
            echo '--- Last 5 lines ---'
            tail -5 \$LOGFILE
            echo '--- Episode count ---'
            grep -c 'Episode' \$LOGFILE 2>/dev/null || echo 0
            echo '--- Errors ---'
            grep -ic 'error\|exception\|traceback' \$LOGFILE 2>/dev/null || echo 0
        else
            echo 'No log file found'
        fi
    "
}

launch_training() {
    log "LAUNCHING training in tmux session: $TRAIN_TMUX"
    ssh_exec "
        # Kill any existing session
        tmux kill-session -t $TRAIN_TMUX 2>/dev/null || true
        sleep 1
        # Start new session
        tmux new-session -d -s $TRAIN_TMUX '$TRAIN_CMD'
    "
    sleep 5
    local status
    status=$(is_training_running)
    if [[ "$status" == "yes" ]]; then
        log "Training launched successfully"
        return 0
    else
        log "ERROR: Training failed to launch!"
        return 1
    fi
}

kill_training() {
    log "KILLING training session: $TRAIN_TMUX"
    ssh_exec "tmux kill-session -t $TRAIN_TMUX 2>/dev/null || true"
}

# ── Main Loop ───────────────────────────────────────────────────────────
log "=========================================="
log "GPU Monitor + Relaunch — Phase 50"
log "Checks every ${CHECK_INTERVAL}s, max ${MAX_CHECKS} checks"
log "GPU: $GPU_HOST:$GPU_PORT"
log "=========================================="

# Initial launch
log "Checking if training is already running..."
STATUS=$(is_training_running)
if [[ "$STATUS" != "yes" ]]; then
    log "No training running, launching now..."
    launch_training || { log "FATAL: Cannot launch training"; exit 1; }
else
    log "Training already running in $TRAIN_TMUX"
fi

# Monitor loop
for ((i=1; i<=MAX_CHECKS; i++)); do
    log "--- Check $i/$MAX_CHECKS (sleeping ${CHECK_INTERVAL}s) ---"
    sleep "$CHECK_INTERVAL"

    # Check connectivity
    if ! ssh_exec "echo ok" >/dev/null 2>&1; then
        log "WARNING: GPU unreachable! Will retry next check."
        continue
    fi

    # GPU stats
    GPU_STATS=$(get_gpu_stats)
    log "GPU: $GPU_STATS"

    # Check training status
    STATUS=$(is_training_running)
    log "Training tmux: $STATUS"

    if [[ "$STATUS" != "yes" ]]; then
        log "ALERT: Training crashed! Collecting diagnostics..."
        get_training_progress
        log "Relaunching..."
        launch_training || log "ERROR: Relaunch failed!"
        continue
    fi

    # Check for stalls
    AGE=$(get_last_episode_age)
    log "Last log activity: ${AGE}m ago"

    if [[ "$AGE" -gt "$STALL_THRESHOLD" ]]; then
        log "ALERT: Training stalled (${AGE}m > ${STALL_THRESHOLD}m threshold)"
        log "Collecting diagnostics before relaunch..."
        get_training_progress
        kill_training
        sleep 3
        log "Relaunching after stall..."
        launch_training || log "ERROR: Relaunch after stall failed!"
        continue
    fi

    # Progress report
    log "Training healthy. Progress:"
    get_training_progress 2>&1 | while read -r line; do log "  $line"; done
done

log "=========================================="
log "Monitor complete ($MAX_CHECKS checks done)"
log "=========================================="
