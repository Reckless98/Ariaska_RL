#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# gpu_autopush_full.sh — Comprehensive GPU→GitHub auto-push
# ═══════════════════════════════════════════════════════════════
#
# Pushes ALL 15+ intelligence paths to master every cycle.
# Replaces the minimal 9-path autopush from h200_gpu_bootstrap.sh.
#
# Run in a tmux `sync` session on the GPU:
#   while true; do bash scripts/gpu_autopush_full.sh; sleep 600; done
#
set -euo pipefail

REPO_DIR="${ARIASKA_DIR:-/root/Ariaska_RL}"
BRANCH="${GIT_BRANCH:-master}"
PUSH_INTERVAL_DESC="manual / loop"

cd "$REPO_DIR"

# ── Step 1: Ensure git identity ──────────────────────────────
git config user.email "ariaska-gpu@runner" 2>/dev/null || true
git config user.name "Ariaska GPU Runner" 2>/dev/null || true

# ── Step 2: Stage all intelligence paths ─────────────────────
# Models + checkpoints
git add -A models/distilled/      2>/dev/null || true
git add -A models/unified/        2>/dev/null || true

# Training results
git add -A results/h200_distill/  2>/dev/null || true
git add -A results/grpo_train/    2>/dev/null || true

# Trace files
git add -A traces/h200_distill/   2>/dev/null || true
git add -A traces/grpo_train/     2>/dev/null || true

# Unified data (converted)
git add -A data/unified/          2>/dev/null || true

# Knowledge + distill prep
git add -A data/distill_prep/     2>/dev/null || true
git add -A data/expert_trajectories/ 2>/dev/null || true

# Learned commands (runtime discoveries)
git add -A data/learned_commands.json 2>/dev/null || true

# Logs (those that are tracked)
git add -A logs/gpu_session_*     2>/dev/null || true

# Postmortems and reports
git add -A postmortems/           2>/dev/null || true
git add -A reports/               2>/dev/null || true

# Config changes
git add -A config/                2>/dev/null || true

# Training scripts (if modified on GPU)
git add scripts/h200_run_distill_3h.py 2>/dev/null || true
git add scripts/unified_data_schema.py 2>/dev/null || true
git add scripts/unify_training_data.py 2>/dev/null || true

# ── Step 3: Check if anything changed ────────────────────────
if git diff --cached --quiet 2>/dev/null; then
    # Nothing staged — skip
    exit 0
fi

# ── Step 4: Commit with timestamp ────────────────────────────
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
STATS=$(git diff --cached --stat --no-color 2>/dev/null | tail -1)
git commit -m "gpu-sync: ${TIMESTAMP} — full intelligence push (${STATS})" \
    --allow-empty >/dev/null 2>&1 || true

# ── Step 5: Push ─────────────────────────────────────────────
if git push origin "$BRANCH" >/dev/null 2>&1; then
    echo "[SYNC] ${TIMESTAMP} pushed to ${BRANCH} (${STATS})"
else
    echo "[SYNC] ${TIMESTAMP} push FAILED — will retry next cycle"
fi
