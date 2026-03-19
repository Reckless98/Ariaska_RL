#!/bin/bash
# Hourly pipeline monitor — checks GPU server every 30 minutes
# Logs to /home/zer0/Projects/Ariaska_RL/logs/pipeline_monitor.log
# Alerts on: process death, errors, disk full, OOM

LOG="/home/zer0/Projects/Ariaska_RL/logs/pipeline_monitor.log"
INTERVAL=1800  # 30 minutes
SSH="ssh -o ConnectTimeout=15 -o StrictHostKeyChecking=no -o BatchMode=yes -p 19312 root@175.155.64.148"

mkdir -p "$(dirname "$LOG")"

echo "========================================" >> "$LOG"
echo "Pipeline monitor started: $(date)" >> "$LOG"
echo "Checking every ${INTERVAL}s (30 min)" >> "$LOG"
echo "========================================" >> "$LOG"

check_count=0

while true; do
    check_count=$((check_count + 1))
    echo "" >> "$LOG"
    echo "--- Check #${check_count} at $(date) ---" >> "$LOG"

    # Run remote health check
    RESULT=$($SSH 'bash -s' 2>&1 <<'REMOTE_CHECK'
echo "PROCESS_STATUS:"
if pgrep -f "ariaska_cybersec2_pipeline" > /dev/null 2>&1; then
    echo "ALIVE"
    ps aux | grep "ariaska_cybersec2_pipeline" | grep -v grep | awk '{print "PID="$2, "CPU="$3"%", "MEM="$4"%", "TIME="$10}'
else
    echo "DEAD"
fi

echo "GPU_STATUS:"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader 2>/dev/null || echo "GPU_QUERY_FAILED"

echo "DISK_STATUS:"
df -h / | tail -1 | awk '{print "USED="$3, "AVAIL="$4, "PCT="$5}'

echo "ERROR_COUNT:"
grep -ci "error\|traceback\|exception\|killed\|oom\|no space\|cuda out of memory" /workspace/pipeline_stdout.log 2>/dev/null || echo "0"

echo "LAST_STAGE:"
grep -E "^=== STAGE|STAGE [0-9]|stage_[0-9]|Starting Stage|Completed Stage|microchain|phase_guided|smart_mentor|coherence|Training|Merging|GGUF|CPT|SFT|DPO" /workspace/pipeline_stdout.log 2>/dev/null | tail -5

echo "LAST_LINES:"
tail -3 /workspace/pipeline_stdout.log 2>/dev/null

echo "RECENT_ERRORS:"
grep -i "error\|traceback\|exception\|killed\|oom\|no space\|cuda out of memory" /workspace/pipeline_stdout.log 2>/dev/null | tail -5
REMOTE_CHECK
)

    echo "$RESULT" >> "$LOG"

    # Parse for critical issues
    if echo "$RESULT" | grep -q "PROCESS_STATUS:" && echo "$RESULT" | grep -q "DEAD"; then
        echo "!!! CRITICAL: Pipeline process DEAD at $(date) !!!" >> "$LOG"
        echo "!!! CRITICAL: Pipeline process DEAD at $(date) !!!" | tee /dev/stderr
        # Try to get exit info
        $SSH 'dmesg | tail -5; cat /workspace/pipeline_stdout.log | tail -20' >> "$LOG" 2>&1
    fi

    if echo "$RESULT" | grep -qi "cuda out of memory\|oom\|killed"; then
        echo "!!! CRITICAL: OOM/KILLED detected at $(date) !!!" >> "$LOG"
        echo "!!! CRITICAL: OOM/KILLED detected at $(date) !!!" | tee /dev/stderr
    fi

    if echo "$RESULT" | grep -q "PCT=100%\|PCT=99%\|PCT=98%"; then
        echo "!!! CRITICAL: Disk nearly full at $(date) !!!" >> "$LOG"
        echo "!!! CRITICAL: Disk nearly full at $(date) !!!" | tee /dev/stderr
    fi

    # Check if SSH failed entirely
    if echo "$RESULT" | grep -qi "connection refused\|connection timed out\|no route\|permission denied"; then
        echo "!!! WARNING: SSH connection failed at $(date) !!!" >> "$LOG"
        echo "!!! WARNING: SSH connection failed at $(date) !!!" | tee /dev/stderr
    fi

    # Check for completion
    if echo "$RESULT" | grep -qi "GGUF.*complete\|All stages complete\|Pipeline complete\|stage 6.*done\|stage_6.*complete"; then
        echo "*** PIPELINE COMPLETE at $(date) ***" >> "$LOG"
        echo "*** PIPELINE COMPLETE at $(date) ***" | tee /dev/stderr
        
        # Get final summary
        $SSH 'echo "=== FINAL SUMMARY ===" && ls -lh /workspace/output/*.gguf 2>/dev/null && echo "---" && tail -30 /workspace/pipeline_stdout.log' >> "$LOG" 2>&1
        echo "Monitor stopping — pipeline finished." >> "$LOG"
        break
    fi

    sleep $INTERVAL
done

echo "Monitor ended: $(date)" >> "$LOG"
