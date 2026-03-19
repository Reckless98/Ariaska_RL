#!/bin/bash
# Auto-launch Stage 7 KD after main pipeline finishes
# Monitors pipeline_stdout.log for completion signal, then fires KD
#
# Usage: Run this on the GPU server AFTER the pipeline starts
#   nohup bash /workspace/scripts/launch_stage7_after_pipeline.sh > /workspace/stage7_launcher.log 2>&1 &

LOG="/workspace/stage7_launcher.log"

echo "========================================" | tee -a "$LOG"
echo "Stage 7 Auto-Launcher started: $(date)" | tee -a "$LOG"
echo "Waiting for main pipeline to complete..." | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

# Wait for main pipeline to finish
while true; do
    # Check if pipeline process is still running
    if pgrep -f "ariaska_cybersec2_pipeline" > /dev/null 2>&1; then
        # Still running — check GGUF completion as a double-check
        if grep -qi "PIPELINE COMPLETE" /workspace/pipeline_stdout.log 2>/dev/null; then
            echo "[$(date)] Pipeline complete signal found!" | tee -a "$LOG"
            break
        fi
        echo "[$(date)] Pipeline still running..." >> "$LOG"
        sleep 300  # Check every 5 minutes
    else
        # Process not running — check if it completed or crashed
        if grep -qi "PIPELINE COMPLETE\|Total time:" /workspace/pipeline_stdout.log 2>/dev/null; then
            echo "[$(date)] Pipeline completed (process exited cleanly)" | tee -a "$LOG"
            break
        else
            echo "[$(date)] WARNING: Pipeline process died without completion signal!" | tee -a "$LOG"
            echo "[$(date)] Checking if merged model exists..." | tee -a "$LOG"
            if [ -f "/workspace/output/merged/config.json" ]; then
                echo "[$(date)] Merged model exists — proceeding with Stage 7 anyway" | tee -a "$LOG"
                break
            else
                echo "[$(date)] FATAL: No merged model found. Cannot run Stage 7." | tee -a "$LOG"
                exit 1
            fi
        fi
    fi
done

# Brief cooldown for GPU memory to fully release
echo "[$(date)] Waiting 30s for GPU memory release..." | tee -a "$LOG"
sleep 30

# Verify merged model exists
if [ ! -f "/workspace/output/merged/config.json" ]; then
    echo "[$(date)] FATAL: Merged model not at /workspace/output/merged/" | tee -a "$LOG"
    exit 1
fi

echo "" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
echo "[$(date)] LAUNCHING STAGE 7: Knowledge Distillation" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

# Launch Stage 7 with output captured
cd /workspace || exit 1
python3 /workspace/scripts/stage7_knowledge_distillation.py \
    > /workspace/kd_stdout.log 2>&1

KD_EXIT=$?

echo "" | tee -a "$LOG"
if [ $KD_EXIT -eq 0 ]; then
    echo "[$(date)] Stage 7 COMPLETED SUCCESSFULLY (exit=$KD_EXIT)" | tee -a "$LOG"
    echo "GGUFs:" | tee -a "$LOG"
    ls -lh /workspace/output/gguf/ariaska-cybersec2-kd-*.gguf 2>/dev/null | tee -a "$LOG"
else
    echo "[$(date)] Stage 7 FAILED (exit=$KD_EXIT)" | tee -a "$LOG"
    echo "Last 20 lines of KD log:" | tee -a "$LOG"
    tail -20 /workspace/kd_stdout.log | tee -a "$LOG"
fi

echo "Stage 7 auto-launcher done: $(date)" | tee -a "$LOG"
