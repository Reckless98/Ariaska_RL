#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# local_sync_loop.sh — Bidirectional local↔GPU sync loop
# ═══════════════════════════════════════════════════════════════
#
# Runs on LOCAL machine. Every cycle:
#   1) git pull (get GPU autopush commits)
#   2) rsync intelligence from GPU (via gpu_rsync_all.sh)
#   3) Optionally push local changes to GPU
#
# Usage:
#   nohup bash scripts/local_sync_loop.sh &
#   # or: bash scripts/local_sync_loop.sh        (foreground)
#
set -euo pipefail

REPO_DIR="${ARIASKA_DIR:-/home/zer0/Projects/Ariaska_RL}"
GPU_HOST="${GPU_HOST:-root@212.247.220.172}"
GPU_PORT="${GPU_PORT:-25107}"
GPU_ROOT="${GPU_ROOT:-/root/Ariaska_RL}"
INTERVAL="${SYNC_INTERVAL:-600}"  # 10 minutes default
SSH_OPTS="-o ConnectTimeout=20 -o ServerAliveInterval=10 -o StrictHostKeyChecking=no"

LOG_FILE="${REPO_DIR}/logs/local_sync.log"
mkdir -p "$(dirname "$LOG_FILE")"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# ── Pre-flight checks ───────────────────────────────────────
if [[ ! -d "${REPO_DIR}/.git" ]]; then
    echo "ERROR: ${REPO_DIR} is not a git repository"
    exit 1
fi

log "═══════════════════════════════════════════"
log "Local Sync Loop — Starting"
log "  Repo: ${REPO_DIR}"
log "  GPU:  ${GPU_HOST}:${GPU_PORT}"
log "  Interval: ${INTERVAL}s"
log "═══════════════════════════════════════════"

CYCLE=0

while true; do
    CYCLE=$((CYCLE + 1))
    log "────── Cycle ${CYCLE} ──────"

    cd "$REPO_DIR"

    # ── Step 1: Git pull (get GPU autopush commits) ──────────
    log "[1/3] Git pull from origin..."

    # NEVER use --rebase (rewrites history, conflicts with autopush)
    # NEVER use git stash (leaves orphan stashes on failed merges)
    # Strategy: fetch + merge with --no-rebase, auto-resolve trace conflicts
    git fetch origin master >/dev/null 2>&1 || true

    LOCAL_HEAD=$(git rev-parse HEAD 2>/dev/null)
    REMOTE_HEAD=$(git rev-parse origin/master 2>/dev/null)

    if [[ "$LOCAL_HEAD" == "$REMOTE_HEAD" ]]; then
        log "  · Already up to date"
    else
        # Auto-commit any uncommitted rsync'd files first (prevents merge conflicts)
        if ! git diff --quiet 2>/dev/null || ! git diff --cached --quiet 2>/dev/null; then
            TIMESTAMP_PRE=$(date -u +%Y%m%d_%H%M%S)
            git add -A traces/ models/unified/ data/unified/ postmortems/ reports/ results/ 2>/dev/null || true
            git commit -m "local-sync: pre-pull auto-commit ${TIMESTAMP_PRE}" \
                --allow-empty >/dev/null 2>&1 || true
        fi

        if git merge origin/master --no-edit -m "local-sync: merge GPU autopush" >/dev/null 2>&1; then
            AHEAD=$(git rev-list --count origin/master..HEAD 2>/dev/null || echo "?")
            log "  ✓ Merged (local ahead by: ${AHEAD})"
        else
            # Conflict — auto-resolve: keep GPU version for data files, keep local for code
            log "  ✗ Merge conflict — auto-resolving..."
            git checkout --theirs traces/ models/ data/ postmortems/ reports/ results/ 2>/dev/null || true
            git checkout --ours scripts/ core/ tests/ Makefile 2>/dev/null || true
            git add -A 2>/dev/null || true
            git commit --no-edit -m "local-sync: auto-resolved merge conflict" >/dev/null 2>&1 || {
                # Last resort: abort and retry next cycle
                git merge --abort 2>/dev/null || true
                log "  ✗ Could not resolve — will retry next cycle"
            }
        fi
    fi

    # ── Step 2: Rsync from GPU ───────────────────────────────
    log "[2/3] Rsync from GPU..."
    RSYNC_SCRIPT="${REPO_DIR}/scripts/gpu_rsync_all.sh"
    if [[ -x "$RSYNC_SCRIPT" ]]; then
        # Run ONE cycle of rsync (no loop — we handle the loop)
        # gpu_rsync_all.sh has its own loop, so we invoke the inner sync
        # directly by sourcing with a single-iteration override.
        # Instead, run a quick targeted rsync of critical paths:
        RSYNC_SSH="ssh ${SSH_OPTS} -p ${GPU_PORT}"

        # Critical paths to sync from GPU
        declare -a SYNC_PATHS=(
            "models/distilled/:models/distilled/"
            "models/unified/:models/unified/"
            "traces/h200_distill/:traces/h200_distill/"
            "results/h200_distill/:results/h200_distill/"
            "data/unified/:data/unified/"
            "data/learned_commands.json:data/learned_commands.json"
            "data/distill_prep/:data/distill_prep/"
            "postmortems/:postmortems/"
            "reports/:reports/"
        )

        sync_ok=0
        sync_fail=0
        for entry in "${SYNC_PATHS[@]}"; do
            src="${entry%%:*}"
            dst="${entry##*:}"
            mkdir -p "${REPO_DIR}/$(dirname "$dst")"
            if rsync -avz --partial \
                -e "$RSYNC_SSH" \
                "${GPU_HOST}:${GPU_ROOT}/${src}" \
                "${REPO_DIR}/${dst}" >> "$LOG_FILE" 2>&1; then
                sync_ok=$((sync_ok + 1))
            else
                sync_fail=$((sync_fail + 1))
            fi
        done
        log "  ✓ Rsync: ${sync_ok} ok, ${sync_fail} failed"
    else
        log "  · Rsync script not found, skipping"
    fi

    # ── Step 3: Push local changes (if any) ──────────────────
    log "[3/3] Push local changes..."
    cd "$REPO_DIR"

    # Stage only specific local-generated files (don't clobber GPU data)
    git add -A scripts/ core/ tests/ Makefile .gitignore 2>/dev/null || true
    git add -A data/unified/ 2>/dev/null || true

    if ! git diff --cached --quiet 2>/dev/null; then
        TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
        git commit -m "local-sync: ${TIMESTAMP} — code + unified data" \
            --allow-empty >/dev/null 2>&1 || true
        if git push origin master >/dev/null 2>&1; then
            log "  ✓ Local changes pushed"
        else
            log "  ✗ Push failed (will retry)"
        fi
    else
        log "  · Nothing to push"
    fi

    # ── Summary ──────────────────────────────────────────────
    MODEL_COUNT=$(find "${REPO_DIR}/models/" -name "*.pt" -type f 2>/dev/null | wc -l)
    UNIFIED_COUNT=$(find "${REPO_DIR}/data/unified/" -name "*.jsonl" -type f 2>/dev/null | wc -l)
    log "  Summary: ${MODEL_COUNT} models, ${UNIFIED_COUNT} unified JSONL files"
    log "  Sleeping ${INTERVAL}s..."

    sleep "$INTERVAL"
done
