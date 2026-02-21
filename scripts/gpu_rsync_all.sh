#!/usr/bin/env bash
# gpu_rsync_all.sh — COMPREHENSIVE rsync from GPU to local every 10 minutes
# Downloads ALL training data, models, intelligence DBs, memory stores, traces
# so local CPU Ariaska has full GPU-trained intelligence.
#
# Sections:
#   1. Model Checkpoints (distilled, unified, enhanced, progress)
#   2. Traces & Results (episode traces, reports, postmortems)
#   3. Runtime Intelligence (learned_commands, skill_library, campaign, KG, chroma)
#   4. Memory & Replay DBs (memory_router, memory.sqlite3, replay_buffer)
#   5. Training Artifacts (chains, distill_prep, expert_trajectories)
#   6. Training Logs
#
# Usage: nohup bash scripts/gpu_rsync_all.sh &

set -euo pipefail

GPU_HOST="root@212.247.220.172"
GPU_PORT="25107"
GPU_ROOT="/root/Ariaska_RL"
LOCAL_ROOT="/home/zer0/Projects/Ariaska_RL"
SSH_OPTS="-o ConnectTimeout=20 -o ServerAliveInterval=10 -o StrictHostKeyChecking=no"
RSYNC_SSH="ssh ${SSH_OPTS} -p ${GPU_PORT}"
INTERVAL=600  # 10 minutes

LOG_FILE="${LOCAL_ROOT}/logs/gpu_rsync.log"

# Pre-create all target directories
mkdir -p "${LOCAL_ROOT}/logs"
mkdir -p "${LOCAL_ROOT}/models/distilled"
mkdir -p "${LOCAL_ROOT}/models/enhanced"
mkdir -p "${LOCAL_ROOT}/models/unified"
mkdir -p "${LOCAL_ROOT}/models/progress"
mkdir -p "${LOCAL_ROOT}/traces/h200_distill"
mkdir -p "${LOCAL_ROOT}/results/h200_distill"
mkdir -p "${LOCAL_ROOT}/postmortems"
mkdir -p "${LOCAL_ROOT}/data/kg_store"
mkdir -p "${LOCAL_ROOT}/data/chromadb"
mkdir -p "${LOCAL_ROOT}/data/vector_memory"
mkdir -p "${LOCAL_ROOT}/data/chains"
mkdir -p "${LOCAL_ROOT}/data/distill_prep"
mkdir -p "${LOCAL_ROOT}/data/expert_trajectories"
mkdir -p "${LOCAL_ROOT}/chroma_storage"
mkdir -p "${LOCAL_ROOT}/core/memory"
mkdir -p "${LOCAL_ROOT}/core/memories"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# Sync a directory (creates dst if needed, skips gracefully if src missing on GPU)
sync_dir() {
    local label="$1" src="$2" dst="$3"
    log "  Syncing ${label}..."
    if rsync -avz --progress --partial \
        -e "${RSYNC_SSH}" \
        "${GPU_HOST}:${src}" "${dst}" >> "$LOG_FILE" 2>&1; then
        log "  ✓ ${label} done"
    else
        log "  ✗ ${label} skipped or failed (rc=$?)"
    fi
}

# Sync a single file (graceful skip if file doesn't exist on GPU)
sync_file() {
    local label="$1" src="$2" dst="$3"
    log "  Syncing ${label}..."
    if rsync -avz --partial \
        -e "${RSYNC_SSH}" \
        "${GPU_HOST}:${src}" "${dst}" >> "$LOG_FILE" 2>&1; then
        log "  ✓ ${label} done"
    else
        log "  · ${label} not found on GPU (skipped)"
    fi
}

log "=========================================="
log "GPU Rsync All v2 — Starting loop (every ${INTERVAL}s)"
log "GPU: ${GPU_HOST}:${GPU_PORT}"
log "Sections: Models, Traces, Intelligence, Memory DBs, Artifacts, Logs"
log "=========================================="

CYCLE=0
while true; do
    CYCLE=$((CYCLE + 1))
    log ""
    log "=== Cycle ${CYCLE} at $(date '+%Y-%m-%d %H:%M:%S') ==="

    # ──────────────────────────────────────────────────────────
    # SECTION 1: MODEL CHECKPOINTS
    # ──────────────────────────────────────────────────────────
    log "[1/6] Model Checkpoints"

    sync_dir "models/distilled (*.pt)" \
        "${GPU_ROOT}/models/distilled/" \
        "${LOCAL_ROOT}/models/distilled/"

    sync_dir "models/unified (priority format)" \
        "${GPU_ROOT}/models/unified/" \
        "${LOCAL_ROOT}/models/unified/"

    sync_dir "models/enhanced (agent snapshots)" \
        "${GPU_ROOT}/models/enhanced/" \
        "${LOCAL_ROOT}/models/enhanced/"

    sync_dir "models/progress (labels)" \
        "${GPU_ROOT}/models/progress/" \
        "${LOCAL_ROOT}/models/progress/"

    # ──────────────────────────────────────────────────────────
    # SECTION 2: TRACES & RESULTS
    # ──────────────────────────────────────────────────────────
    log "[2/6] Traces & Results"

    sync_dir "traces (all)" \
        "${GPU_ROOT}/traces/" \
        "${LOCAL_ROOT}/traces/"

    sync_dir "results (all)" \
        "${GPU_ROOT}/results/" \
        "${LOCAL_ROOT}/results/"

    sync_dir "postmortems" \
        "${GPU_ROOT}/postmortems/" \
        "${LOCAL_ROOT}/postmortems/"

    # ──────────────────────────────────────────────────────────
    # SECTION 3: RUNTIME INTELLIGENCE
    # These are the databases/files that accumulate learned
    # knowledge during training — critical for CPU inference.
    # ──────────────────────────────────────────────────────────
    log "[3/6] Runtime Intelligence"

    # 3a) Learned commands — runtime-discovered command patterns (872KB+)
    sync_file "learned_commands.json" \
        "${GPU_ROOT}/data/learned_commands.json" \
        "${LOCAL_ROOT}/data/learned_commands.json"

    # 3b) Skill library — SkillCards from postmortem analysis
    sync_file "skill_library.json" \
        "${GPU_ROOT}/core/memory/skill_library.json" \
        "${LOCAL_ROOT}/core/memory/skill_library.json"

    # 3c) Skill audit trail — append-only JSONL of skill changes
    sync_file "skill_audit.jsonl" \
        "${GPU_ROOT}/core/memory/skill_audit.jsonl" \
        "${LOCAL_ROOT}/core/memory/skill_audit.jsonl"

    # 3d) Campaign state — cross-episode intel (creds, shells, vulns)
    sync_file "campaign_state.json" \
        "${GPU_ROOT}/data/campaign_state.json" \
        "${LOCAL_ROOT}/data/campaign_state.json"

    # 3e) Knowledge Graph LMDB store (35MB+)
    sync_dir "kg_store (LMDB)" \
        "${GPU_ROOT}/data/kg_store/" \
        "${LOCAL_ROOT}/data/kg_store/"

    # 3f) Knowledge Graph JSON snapshot (17MB)
    sync_file "kg_snapshot.json" \
        "${GPU_ROOT}/data/kg_snapshot.json" \
        "${LOCAL_ROOT}/data/kg_snapshot.json"

    # 3g) ChromaDB semantic memory vectors
    sync_dir "chromadb vectors" \
        "${GPU_ROOT}/data/chromadb/" \
        "${LOCAL_ROOT}/data/chromadb/"

    # 3h) Agent experience vectors
    sync_dir "vector_memory" \
        "${GPU_ROOT}/data/vector_memory/" \
        "${LOCAL_ROOT}/data/vector_memory/"

    # 3i) RAG knowledge query vectors (top-level chroma_storage/)
    sync_dir "chroma_storage (RAG)" \
        "${GPU_ROOT}/chroma_storage/" \
        "${LOCAL_ROOT}/chroma_storage/"

    # ──────────────────────────────────────────────────────────
    # SECTION 4: MEMORY & REPLAY DATABASES
    # SQLite DBs accumulating agent memories and replay data.
    # ──────────────────────────────────────────────────────────
    log "[4/6] Memory & Replay DBs"

    # 4a) Memory router — GPT cache + agent memories
    sync_file "memory_router.db" \
        "${GPU_ROOT}/core/memory/memory_router.db" \
        "${LOCAL_ROOT}/core/memory/memory_router.db"

    # 4b) Experience memory store
    sync_file "memory.sqlite3" \
        "${GPU_ROOT}/core/memories/memory.sqlite3" \
        "${LOCAL_ROOT}/core/memories/memory.sqlite3"

    # 4c) PER replay buffer
    sync_file "replay_buffer.sqlite3" \
        "${GPU_ROOT}/replay_buffer.sqlite3" \
        "${LOCAL_ROOT}/replay_buffer.sqlite3"

    # ──────────────────────────────────────────────────────────
    # SECTION 5: TRAINING ARTIFACTS
    # Pre-computed chains, distillation data, expert trajectories.
    # ──────────────────────────────────────────────────────────
    log "[5/6] Training Artifacts"

    # 5a) Cached attack chain plans
    sync_dir "chains" \
        "${GPU_ROOT}/data/chains/" \
        "${LOCAL_ROOT}/data/chains/"

    # 5b) Distillation prep — synthetic traces + teacher trajectories
    sync_dir "distill_prep" \
        "${GPU_ROOT}/data/distill_prep/" \
        "${LOCAL_ROOT}/data/distill_prep/"

    # 5c) Expert trajectories for behavioral cloning
    sync_dir "expert_trajectories" \
        "${GPU_ROOT}/data/expert_trajectories/" \
        "${LOCAL_ROOT}/data/expert_trajectories/"

    # ──────────────────────────────────────────────────────────
    # SECTION 6: TRAINING LOGS
    # ──────────────────────────────────────────────────────────
    log "[6/6] Training Logs"

    log "  Syncing distill_*.log..."
    rsync -avz -e "${RSYNC_SSH}" \
        "${GPU_HOST}:/root/distill_*.log" \
        "${LOCAL_ROOT}/logs/" >> "$LOG_FILE" 2>&1 || true
    log "  ✓ training logs done"

    # ──────────────────────────────────────────────────────────
    # SUMMARY
    # ──────────────────────────────────────────────────────────
    LOCAL_MODELS=$(find "${LOCAL_ROOT}/models/" -name "*.pt" -type f 2>/dev/null | wc -l)
    LOCAL_UNIFIED=$(find "${LOCAL_ROOT}/models/unified/" -name "*.pt" -type f 2>/dev/null | wc -l)
    LOCAL_TRACES=$(find "${LOCAL_ROOT}/traces/" -name "*.jsonl" -type f 2>/dev/null | wc -l)
    LOCAL_RESULTS=$(find "${LOCAL_ROOT}/results/" -name "*.json" -type f 2>/dev/null | wc -l)
    LOCAL_POSTMORTEMS=$(find "${LOCAL_ROOT}/postmortems/" -type f 2>/dev/null | wc -l)
    LOCAL_INTEL=0
    for f in "${LOCAL_ROOT}/data/learned_commands.json" \
             "${LOCAL_ROOT}/core/memory/skill_library.json" \
             "${LOCAL_ROOT}/core/memory/skill_audit.jsonl" \
             "${LOCAL_ROOT}/data/campaign_state.json" \
             "${LOCAL_ROOT}/data/kg_snapshot.json" \
             "${LOCAL_ROOT}/core/memory/memory_router.db" \
             "${LOCAL_ROOT}/core/memories/memory.sqlite3" \
             "${LOCAL_ROOT}/replay_buffer.sqlite3"; do
        [[ -f "$f" ]] && LOCAL_INTEL=$((LOCAL_INTEL + 1))
    done
    LOCAL_INTEL_DIRS=0
    for d in "${LOCAL_ROOT}/data/kg_store" \
             "${LOCAL_ROOT}/data/chromadb" \
             "${LOCAL_ROOT}/data/vector_memory" \
             "${LOCAL_ROOT}/chroma_storage" \
             "${LOCAL_ROOT}/data/chains" \
             "${LOCAL_ROOT}/data/distill_prep" \
             "${LOCAL_ROOT}/data/expert_trajectories"; do
        [[ -d "$d" ]] && [[ "$(ls -A "$d" 2>/dev/null)" ]] && LOCAL_INTEL_DIRS=$((LOCAL_INTEL_DIRS + 1))
    done

    log "  ╔══════════════════════════════════════════╗"
    log "  ║ Sync Summary                             ║"
    log "  ╠══════════════════════════════════════════╣"
    log "  ║ Models:      ${LOCAL_MODELS} total (${LOCAL_UNIFIED} unified)"
    log "  ║ Traces:      ${LOCAL_TRACES} JSONL files"
    log "  ║ Results:     ${LOCAL_RESULTS} JSON reports"
    log "  ║ Postmortems: ${LOCAL_POSTMORTEMS} files"
    log "  ║ Intel files: ${LOCAL_INTEL}/8 synced"
    log "  ║ Intel dirs:  ${LOCAL_INTEL_DIRS}/7 populated"
    log "  ╚══════════════════════════════════════════╝"

    log "=== Cycle ${CYCLE} complete. Sleeping ${INTERVAL}s ==="
    sleep "$INTERVAL"
done
