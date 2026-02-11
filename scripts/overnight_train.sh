#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# ARIASKA_RL — Overnight Training Script
# Phase 7.2: Progressive difficulty, auto-checkpointing, full logging
#
# Usage:
#   ./scripts/overnight_train.sh              # Full overnight (300 episodes)
#   ./scripts/overnight_train.sh --quick      # Quick test (30 episodes)
#   ./scripts/overnight_train.sh --medium     # Medium only (100 episodes)
#   ./scripts/overnight_train.sh --hard       # Hard only (100 episodes)
#
# Author: Filip Volf
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

PYTHON="${PROJECT_DIR}/.venv/bin/python"
if [[ ! -f "$PYTHON" ]]; then
    PYTHON="python"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/overnight_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

# Seed for reproducibility (varied per phase for diversity)
BASE_SEED=42

# ── Parse Arguments ────────────────────────────────────────────────────────
MODE="full"
if [[ "${1:-}" == "--quick" ]]; then
    MODE="quick"
elif [[ "${1:-}" == "--medium" ]]; then
    MODE="medium"
elif [[ "${1:-}" == "--hard" ]]; then
    MODE="hard"
fi

# ── Phase Configuration ───────────────────────────────────────────────────
# Each phase progressively increases difficulty.
# Checkpoints are saved every 10 episodes and between phases.
case "$MODE" in
    quick)
        MEDIUM_EPISODES=15
        HARD_EPISODES=15
        STEPS=40
        ;;
    medium)
        MEDIUM_EPISODES=100
        HARD_EPISODES=0
        STEPS=40
        ;;
    hard)
        MEDIUM_EPISODES=0
        HARD_EPISODES=100
        STEPS=40
        ;;
    full)
        MEDIUM_EPISODES=100
        HARD_EPISODES=200
        STEPS=40
        ;;
esac

TOTAL_EPISODES=$((MEDIUM_EPISODES + HARD_EPISODES))

# ── Banner ─────────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════"
echo " 🌙 ARIASKA Overnight Training — Phase 7.2"
echo "═══════════════════════════════════════════════════════════════"
echo " Mode:       $MODE"
echo " Total:      $TOTAL_EPISODES episodes ($MEDIUM_EPISODES medium + $HARD_EPISODES hard)"
echo " Steps/ep:   $STEPS"
echo " Base seed:  $BASE_SEED"
echo " Logs:       $LOG_DIR/"
echo " Started:    $(date)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ── Pre-flight checks ─────────────────────────────────────────────────────
echo "🔍 Pre-flight checks..."

# Test suite validation
echo -n "   Running tests... "
TEST_OUTPUT=$($PYTHON -m pytest tests/ -x -q 2>&1 | tail -1)
if echo "$TEST_OUTPUT" | grep -q "passed"; then
    echo "✅ $TEST_OUTPUT"
else
    echo "❌ Tests failed! Aborting overnight run."
    echo "$TEST_OUTPUT"
    exit 1
fi

# Check disk space (need at least 1GB free)
FREE_KB=$(df "$PROJECT_DIR" --output=avail | tail -1 | tr -d ' ')
if [[ "$FREE_KB" -lt 1048576 ]]; then
    echo "⚠️  Low disk space: $(( FREE_KB / 1024 ))MB free. Need at least 1GB."
    exit 1
fi
echo "   Disk space: ✅ $(( FREE_KB / 1024 ))MB free"

# Check OpenAI key (warn but don't abort — works offline too)
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "   ⚠️  OPENAI_API_KEY not set — running in offline/simulation mode"
else
    echo "   OpenAI API: ✅ Key present"
fi

echo ""
echo "🚀 Starting training pipeline..."
echo ""

# ── Helper function ────────────────────────────────────────────────────────
run_phase() {
    local PHASE_NAME="$1"
    local EPISODES="$2"
    local DIFFICULTY="$3"
    local SEED="$4"
    local LOG_FILE="${LOG_DIR}/${PHASE_NAME}.log"
    local RESULTS_FILE="${LOG_DIR}/${PHASE_NAME}_results.json"
    
    if [[ "$EPISODES" -eq 0 ]]; then
        echo "   ⏭️  Skipping $PHASE_NAME (0 episodes)"
        return 0
    fi
    
    echo "┌─────────────────────────────────────────────────────────────"
    echo "│ 📊 Phase: $PHASE_NAME"
    echo "│ Episodes: $EPISODES | Difficulty: $DIFFICULTY | Seed: $SEED"
    echo "│ Log: $LOG_FILE"
    echo "│ Started: $(date +%H:%M:%S)"
    echo "└─────────────────────────────────────────────────────────────"
    
    # Run training
    $PYTHON ariaska_cli.py smart-train \
        --episodes "$EPISODES" \
        --steps "$STEPS" \
        --seed "$SEED" \
        --env ms3 \
        --difficulty "$DIFFICULTY" \
        --verbosity standard \
        --checkpoint-every 10 \
        --seed-skills \
        2>&1 | tee "$LOG_FILE" | grep -E "^(━━|│|💾|⚠️|❌|📊|Phase)" | tail -20
    
    EXIT_CODE=${PIPESTATUS[0]}
    
    # Copy results artifact
    if [[ -f "artifacts/phase5_${EPISODES}ep_results.json" ]]; then
        cp "artifacts/phase5_${EPISODES}ep_results.json" "$RESULTS_FILE" 2>/dev/null || true
    fi
    
    # Extract summary stats from log
    local CLOSEOUT_PCT=$(grep -c "CLOSEOUT" "$LOG_FILE" 2>/dev/null | head -1 || echo "?")
    local AVG_REWARD=$(grep "Avg Reward" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oP '\+[\d.]+' || echo "?")
    local MENTOR_CALLS=$(grep -c "MENTOR-REASONING" "$LOG_FILE" 2>/dev/null || echo "0")
    
    echo ""
    echo "   ✅ $PHASE_NAME complete | Avg Reward: $AVG_REWARD | Mentor calls: $MENTOR_CALLS"
    echo "   Finished: $(date +%H:%M:%S)"
    echo ""
    
    return $EXIT_CODE
}

# ── Phase 1: Medium Difficulty ─────────────────────────────────────────────
START_TIME=$(date +%s)

run_phase "01_ms3_medium" "$MEDIUM_EPISODES" "ms3_medium" "$BASE_SEED" || {
    echo "⚠️ Medium phase had errors (exit $?), continuing to hard..."
}

# ── Phase 2: Hard Difficulty ───────────────────────────────────────────────
run_phase "02_ms3_hard" "$HARD_EPISODES" "ms3_hard" "$((BASE_SEED + 100))" || {
    echo "⚠️ Hard phase had errors (exit $?)"
}

# ── Summary ────────────────────────────────────────────────────────────────
END_TIME=$(date +%s)
DURATION=$(( END_TIME - START_TIME ))
HOURS=$(( DURATION / 3600 ))
MINUTES=$(( (DURATION % 3600) / 60 ))
SECONDS=$(( DURATION % 60 ))

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " 🏁 ARIASKA Overnight Training COMPLETE"
echo "═══════════════════════════════════════════════════════════════"
echo " Mode:       $MODE"
echo " Duration:   ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo " Episodes:   $TOTAL_EPISODES"
echo " Logs:       $LOG_DIR/"
echo " Finished:   $(date)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Log summary to file
cat > "${LOG_DIR}/summary.txt" << EOF
ARIASKA Overnight Training Summary
===================================
Mode:       $MODE
Duration:   ${HOURS}h ${MINUTES}m ${SECONDS}s
Episodes:   $TOTAL_EPISODES ($MEDIUM_EPISODES medium + $HARD_EPISODES hard)
Steps/ep:   $STEPS
Base seed:  $BASE_SEED
Started:    $(date -d @$START_TIME)
Finished:   $(date -d @$END_TIME)

Phase 7.2 Features Active:
- Forward-only phase gating
- Credential-aware command filtering
- Smart codex-mini reasoning checks
- Negative PPO reward for blocked commands
- Exploited-service tracking
- BlueAgent defensive-only mode
- Postmortem every 10 episodes (gpt-5.2-codex)
EOF

echo "📋 Summary saved to ${LOG_DIR}/summary.txt"

# List checkpoint files
echo ""
echo "💾 Checkpoints saved:"
ls -la models/enhanced/ppo_*.pt 2>/dev/null | tail -10 || echo "   No checkpoints found"

echo ""
echo "🌙 Good night! Training is done. Check ${LOG_DIR}/ for details."
