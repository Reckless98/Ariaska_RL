#!/usr/bin/env bash
# =============================================================================
# H200 GPU Training Session — 3-Stage Pipeline
#
# Stage 1: Test run (30 steps) — validates everything works
# Stage 2: Full distillation (2h 10m) — weighted reward, consistency, auto-load
# Stage 3: Evaluate → either 2h more distill OR 2h GRPO
#
# Usage (on GPU box, inside tmux 'train' session):
#   bash scripts/gpu_run_session.sh
#
# Prerequisites:
#   - h200_gpu_bootstrap.sh already ran (venv, mentor serving, .env)
#   - gpt-oss-120b mentor on localhost:8192
# =============================================================================
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/Ariaska_RL}"
cd "$REPO_DIR"
source .venv/bin/activate
export PYTHONPATH="$REPO_DIR"

# Timestamp for this session
SESSION_ID=$(date -u +%Y%m%dT%H%M%SZ)
LOG_DIR="$REPO_DIR/logs/gpu_session_${SESSION_ID}"
mkdir -p "$LOG_DIR"

echo "====================================================================="
echo "  ARIASKA GPU TRAINING SESSION"
echo "  Started: $(date -u)"
echo "  Session: $SESSION_ID"
echo "  Log dir: $LOG_DIR"
echo "====================================================================="

# ── Helper: check if reward improved ────────────────────────────
check_improvement() {
    local result_file="$1"
    if [ ! -f "$result_file" ]; then
        echo "no_result"
        return
    fi
    # Extract avg_reward from last episode summary in the result
    local avg_reward
    avg_reward=$(python3 -c "
import json, sys
try:
    with open('$result_file') as f:
        data = json.load(f)
    r = data.get('avg_reward_per_episode', data.get('avg_reward', 0))
    print(f'{r:.2f}')
except Exception:
    print('0.00')
" 2>/dev/null)
    echo "$avg_reward"
}

# ═══════════════════════════════════════════════════════════════
# STAGE 1: Test Run (30 steps, ~2 min)
# ═══════════════════════════════════════════════════════════════
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STAGE 1: Test Run (30 steps)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Quick mentor health check
echo "  Checking mentor endpoint..."
if curl -s --max-time 5 "http://127.0.0.1:8192/v1/models" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'  ✓ Mentor online: {d[\"data\"][0][\"id\"]}')"; then
    MENTOR_OK=true
else
    echo "  ✗ Mentor unreachable — will run without mentor (--no-mentor)"
    MENTOR_OK=false
fi

MENTOR_FLAG=""
if [ "$MENTOR_OK" = false ]; then
    MENTOR_FLAG="--no-mentor"
fi

# Run 30-step test (1 episode, 30 steps)
echo "  Running test: 1 episode, 30 steps..."
python -m scripts.h200_run_distill_3h \
    --seed 42 \
    --max-hours 0.1 \
    --max-episodes 1 \
    --max-steps 30 \
    --checkpoint-every 5m \
    --reward-weights "format=2.0,code=1.5,math=1.5,reasoning=0.5" \
    $MENTOR_FLAG \
    2>&1 | tee "$LOG_DIR/stage1_test.log"

TEST_EXIT=$?
if [ $TEST_EXIT -ne 0 ]; then
    echo ""
    echo "  ✗ TEST FAILED (exit $TEST_EXIT) — aborting session"
    echo "  Check: $LOG_DIR/stage1_test.log"
    exit 1
fi

echo ""
echo "  ✓ Stage 1 PASSED — proceeding to full training"
echo ""

# ═══════════════════════════════════════════════════════════════
# STAGE 2: Full Distillation (2h)
# ═══════════════════════════════════════════════════════════════
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STAGE 2: Full Distillation (2h 10m)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Reward weights: format=2.0, code=1.5, math=1.5, reasoning=0.5"
echo "  Auto-loading best checkpoint as base"
echo ""

STAGE2_START=$(date +%s)

python -m scripts.h200_run_distill_3h \
    --seed 42 \
    --max-hours 2.17 \
    --max-episodes 500 \
    --checkpoint-every 10m \
    --reward-weights "format=2.0,code=1.5,math=1.5,reasoning=0.5" \
    $MENTOR_FLAG \
    2>&1 | tee "$LOG_DIR/stage2_distill_2h10m.log"

STAGE2_EXIT=$?
STAGE2_END=$(date +%s)
STAGE2_DURATION=$(( (STAGE2_END - STAGE2_START) / 60 ))

echo ""
echo "  Stage 2 finished in ${STAGE2_DURATION}m (exit: $STAGE2_EXIT)"

# Auto-sync after stage 2
echo "  Syncing to GitHub..."
bash /root/ariaska_autopush.sh 2>/dev/null || echo "  (sync skipped)"

# ═══════════════════════════════════════════════════════════════
# EVALUATION: Run eval on Stage 2 checkpoint
# ═══════════════════════════════════════════════════════════════
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  EVALUATION: Running inference eval (20 episodes)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python -m scripts.inference \
    --episodes 20 \
    --max-steps 100 \
    --seed 42 \
    --output "results/eval/stage2_${SESSION_ID}.json" \
    2>&1 | tee "$LOG_DIR/stage2_eval.log"

echo "  ✓ Evaluation complete"
echo ""

# ═══════════════════════════════════════════════════════════════
# DECISION GATE: Analyze Stage 2 results
# ═══════════════════════════════════════════════════════════════
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  DECISION GATE: Analyzing Stage 2 results..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check if reward trend is improving or plateauing
DECISION=$(python3 << 'PYEOF'
import json, glob, os, sys

# Find latest h200 distill result
results = sorted(glob.glob("results/h200_distill/*.json"), key=os.path.getmtime)
if not results:
    print("distill")  # no data → try more distill
    sys.exit(0)

try:
    with open(results[-1]) as f:
        data = json.load(f)

    # Check reward trend from the training run
    reward_hist = data.get("reward_history", data.get("episode_rewards", []))
    if len(reward_hist) < 10:
        print("distill")  # too few episodes → more distill
        sys.exit(0)

    # Compare first half vs second half
    mid = len(reward_hist) // 2
    first_half = sum(reward_hist[:mid]) / max(mid, 1)
    second_half = sum(reward_hist[mid:]) / max(len(reward_hist) - mid, 1)

    improvement = second_half - first_half
    print(f"# First half avg: {first_half:.2f}, Second half avg: {second_half:.2f}", file=sys.stderr)
    print(f"# Improvement: {improvement:+.2f}", file=sys.stderr)

    if improvement < 1.0:
        # Plateaued → switch to GRPO for fresh gradient signal
        print("grpo")
    else:
        # Still improving → more distill
        print("distill")
except Exception as e:
    print(f"# Error analyzing: {e}", file=sys.stderr)
    print("distill")  # default: more distill
PYEOF
)

echo "  Decision: $DECISION"
echo ""

# ═══════════════════════════════════════════════════════════════
# STAGE 3: 2h more (distill or GRPO based on decision)
# ═══════════════════════════════════════════════════════════════
if [ "$DECISION" = "grpo" ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  STAGE 3: GRPO Training (2h) — reward plateaued"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Group size: 4, KL coef: 0.04, LR: 1e-4"
    echo "  Auto-loading best checkpoint from Stage 2"
    echo ""

    python -m scripts.train_grpo \
        --seed 42 \
        --max-hours 2 \
        --group-size 4 \
        --learning-rate 1e-4 \
        --reward-weights "format=2.0,code=1.5,math=1.5,reasoning=0.5" \
        --checkpoint-every 600 \
        2>&1 | tee "$LOG_DIR/stage3_grpo_2h.log"
else
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  STAGE 3: More Distillation (2h) — still improving"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Continuing with same config, auto-loads Stage 2 checkpoint"
    echo ""

    python -m scripts.h200_run_distill_3h \
        --seed 1337 \
        --max-hours 2 \
        --max-episodes 500 \
        --checkpoint-every 10m \
        --reward-weights "format=2.0,code=1.5,math=1.5,reasoning=0.5" \
        $MENTOR_FLAG \
        2>&1 | tee "$LOG_DIR/stage3_distill_2h.log"
fi

STAGE3_EXIT=$?

# Final sync
echo ""
echo "  Final sync to GitHub..."
bash /root/ariaska_autopush.sh 2>/dev/null || echo "  (sync skipped)"

# ═══════════════════════════════════════════════════════════════
# SESSION SUMMARY
# ═══════════════════════════════════════════════════════════════
echo ""
echo "====================================================================="
echo "  SESSION COMPLETE — $SESSION_ID"
echo "====================================================================="
echo "  Stage 1: Test run (30 steps)  ✓"
echo "  Stage 2: Distill 2h10m        exit=$STAGE2_EXIT"
echo "  Stage 3: ${DECISION^^} 2h         exit=$STAGE3_EXIT"
echo "  Checkpoints: $(ls models/distilled/*.pt 2>/dev/null | wc -l) .pt files in models/distilled/"
echo "  Logs: $LOG_DIR/"
echo ""
echo "  NEXT: Pull results to laptop:"
echo "    ./pull_from_gpu.sh"
echo "    make eval-compare"
echo "====================================================================="
