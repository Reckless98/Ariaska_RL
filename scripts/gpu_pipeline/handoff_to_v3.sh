#!/usr/bin/env bash
# handoff_to_v3.sh — Watches for v2 pipeline Stage 1 completion, then kills v2 and launches v3
#
# The v2 pipeline writes /workspace/data/ariaska_schema_sft.jsonl when Stage 1 finishes.
# v3 pipeline auto-detects this file (≥8000 lines → skip Stage 1) and starts at Stage 2.
#
# Usage: nohup bash /workspace/scripts/handoff_to_v3.sh > /workspace/handoff.log 2>&1 &

set -euo pipefail

SCHEMA_FILE="/workspace/data/ariaska_schema_sft.jsonl"
MIN_LINES=8000
OLD_PIPELINE_PID=7623
V3_SCRIPT="/workspace/scripts/ariaska_cybersec3_pipeline.py"
CHECK_INTERVAL=60  # seconds

log() {
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*"
}

log "=== HANDOFF WATCHDOG STARTED ==="
log "Watching for: ${SCHEMA_FILE} with ≥${MIN_LINES} lines"
log "Will kill old pipeline PID ${OLD_PIPELINE_PID} and launch v3"

# Wait for schema data file to appear with sufficient data
while true; do
    if [[ -f "$SCHEMA_FILE" ]]; then
        lines=$(wc -l < "$SCHEMA_FILE" 2>/dev/null || echo 0)
        if (( lines >= MIN_LINES )); then
            log "Schema data ready: ${lines} lines (≥${MIN_LINES})"
            break
        else
            log "Schema file exists but only ${lines} lines, waiting..."
        fi
    else
        log "Schema file not yet written, waiting... (checking every ${CHECK_INTERVAL}s)"
    fi
    sleep "$CHECK_INTERVAL"
done

# Give the old pipeline 10 seconds to finish writing (flush)
log "Waiting 10s for file flush..."
sleep 10

# Double-check the file is complete
final_lines=$(wc -l < "$SCHEMA_FILE")
log "Final schema line count: ${final_lines}"

# Kill old pipeline processes
log "Killing old v2 pipeline..."
kill -TERM "$OLD_PIPELINE_PID" 2>/dev/null || log "PID $OLD_PIPELINE_PID already dead"

# Kill Stage 7 launcher if running
for pid in $(pgrep -f "launch_stage7_after_pipeline" 2>/dev/null); do
    log "Killing Stage 7 launcher PID ${pid}"
    kill -TERM "$pid" 2>/dev/null || true
done

# Wait for old pipeline to die
sleep 5
if kill -0 "$OLD_PIPELINE_PID" 2>/dev/null; then
    log "Force killing PID $OLD_PIPELINE_PID"
    kill -9 "$OLD_PIPELINE_PID" 2>/dev/null || true
fi

log "Old pipeline killed."

# Clean GPU memory
log "Cleaning GPU memory..."
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
sleep 3

# Report state
log "Disk: $(df -h / | tail -1)"
log "GPU: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null)"

# Launch v3 pipeline
log "=== LAUNCHING V3 PIPELINE ==="
log "Script: ${V3_SCRIPT}"

# Truncate old pipeline log (v3 writes to same training.log, keep it)
: > /workspace/pipeline_stdout.log

cd /workspace
exec python3 -u "$V3_SCRIPT" >> /workspace/pipeline_stdout.log 2>&1
