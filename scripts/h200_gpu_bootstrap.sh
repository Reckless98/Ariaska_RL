#!/usr/bin/env bash
# =============================================================================
# H200 GPU Bootstrap — Ariaska Online Distillation
#
# Automates Phases 1-7: SSH setup, env, deps, mentor, autosync.
# Run this ON THE GPU BOX after rsync + scp of repo and .env.
#
# Usage:
#   # From laptop, first transfer:
#   rsync -avz --delete -e "ssh -p 25107" \
#     --exclude ".venv" --exclude "__pycache__" --exclude ".git" \
#     --exclude "node_modules" --exclude ".mypy_cache" --exclude ".ruff_cache" \
#     ./Ariaska_RL/ root@212.247.220.172:/root/Ariaska_RL/
#
#   scp -P 25107 ./Ariaska_RL/.env root@212.247.220.172:/root/Ariaska_RL/.env
#
#   # Then SSH in and run:
#   ssh -p 25107 root@212.247.220.172 -L 8080:localhost:8080
#   bash /root/Ariaska_RL/scripts/h200_gpu_bootstrap.sh
# =============================================================================
set -euo pipefail

REPO_DIR="/root/Ariaska_RL"
MENTOR_PORT=8192
MENTOR_MODEL="openai/gpt-oss-120b"
FALLBACK_MODEL="openai/gpt-oss-120b"
GPU_UTIL=0.55
MAX_MODEL_LEN=8192
MENTOR_LOG="/var/log/ariaska_mentor.log"

echo "====================================================================="
echo "  ARIASKA H200 GPU BOOTSTRAP"
echo "  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "====================================================================="

# ── Phase 1: System packages ────────────────────────────────────
echo ""
echo "[Phase 1] Installing system packages..."
apt-get update -qq
apt-get install -y -qq git rsync tmux python3-venv python3-pip curl jq htop >/dev/null 2>&1
echo "  ✓ System packages installed"

# ── Phase 3: Python venv + deps ─────────────────────────────────
echo ""
echo "[Phase 3] Setting up Python environment..."
cd "$REPO_DIR"

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "  ✓ Created venv"
else
    echo "  ✓ venv already exists"
fi

source .venv/bin/activate
pip install -U pip wheel -q
echo "  ✓ pip/wheel upgraded"

echo "  Installing requirements.txt..."
pip install -r requirements.txt -q 2>&1 | tail -3
echo "  ✓ Base deps installed"

echo "  Installing GPU extras..."
pip install -U vllm sentence-transformers faiss-gpu tensorboard rich requests -q 2>&1 | tail -3
echo "  ✓ GPU extras installed"

echo ""
echo "[Phase 3b] GPU verification..."
python3 -c "
import torch
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  Device: {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    print(f'  Memory: {props.total_mem / 1e9:.1f} GB')
    print(f'  Compute: {props.major}.{props.minor}')
else:
    print('  WARNING: No GPU detected!')
"

# ── Phase 4-5: Mentor model server ──────────────────────────────
echo ""
echo "[Phase 4-5] Starting vLLM mentor server..."

# Kill any existing mentor
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
sleep 2

# Create mentor tmux session
tmux kill-session -t mentor 2>/dev/null || true
tmux new-session -d -s mentor

# Try primary model first
echo "  Attempting primary model: $MENTOR_MODEL (MXFP4 quantization)"
tmux send-keys -t mentor "cd $REPO_DIR && source .venv/bin/activate" C-m
sleep 1
tmux send-keys -t mentor "python -m vllm.entrypoints.openai.api_server \
  --model $MENTOR_MODEL \
  --host 0.0.0.0 --port $MENTOR_PORT \
  --max-model-len $MAX_MODEL_LEN \
  --gpu-memory-utilization $GPU_UTIL \
  --dtype auto \
  --trust-remote-code \
  --disable-log-requests \
  > $MENTOR_LOG 2>&1 &" C-m

echo "  Waiting for mentor to load (up to 15 minutes)..."
MENTOR_READY=false
for i in $(seq 1 180); do
    if curl -s "http://127.0.0.1:$MENTOR_PORT/v1/models" | jq -e '.data[0].id' >/dev/null 2>&1; then
        MENTOR_READY=true
        ACTIVE_MODEL=$(curl -s "http://127.0.0.1:$MENTOR_PORT/v1/models" | jq -r '.data[0].id')
        echo "  ✓ MENTOR READY after $((i*5))s — model: $ACTIVE_MODEL"
        break
    fi
    # Check for OOM or load failure
    if [ $i -gt 60 ] && ! pgrep -f "vllm" >/dev/null 2>&1; then
        echo "  ✗ Primary model failed — trying fallback: $FALLBACK_MODEL"

        tmux send-keys -t mentor C-c
        sleep 2
        tmux send-keys -t mentor "python -m vllm.entrypoints.openai.api_server \
          --model $FALLBACK_MODEL \
          --host 0.0.0.0 --port $MENTOR_PORT \
          --max-model-len $MAX_MODEL_LEN \
          --gpu-memory-utilization $GPU_UTIL \
          --dtype auto \
          --trust-remote-code \
          --disable-log-requests \
          > $MENTOR_LOG 2>&1 &" C-m

        # Wait for fallback
        for _j in $(seq 1 180); do
            if curl -s "http://127.0.0.1:$MENTOR_PORT/v1/models" | jq -e '.data[0].id' >/dev/null 2>&1; then
                MENTOR_READY=true
                ACTIVE_MODEL=$(curl -s "http://127.0.0.1:$MENTOR_PORT/v1/models" | jq -r '.data[0].id')
                echo "  ✓ FALLBACK MENTOR READY — model: $ACTIVE_MODEL"
                break 2
            fi
            sleep 5
        done
        break
    fi
    sleep 5
done

if [ "$MENTOR_READY" = false ]; then
    echo "  ✗ WARNING: Mentor server did not start — training will proceed without mentor"
fi

# ── Phase 6: Configure .env for local mentor ────────────────────
echo ""
echo "[Phase 6] Configuring .env for local mentor..."

# Only append if not already configured
if ! grep -q "FF_LOCAL_LLM" "$REPO_DIR/.env" 2>/dev/null; then
    cat >> "$REPO_DIR/.env" << EOF

# === H200 distillation run ($(date -u +%Y-%m-%dT%H:%M:%SZ)) ===
FF_LOCAL_LLM=1
ARIASKA_LOCAL_BACKEND=vllm
ARIASKA_LOCAL_LLM_BASE_URL=http://127.0.0.1:$MENTOR_PORT/v1
ARIASKA_LOCAL_LLM_MODEL=${ACTIVE_MODEL:-$MENTOR_MODEL}
MENTOR_STRATEGY=local_first
ENABLE_DUAL_MENTOR=1
OPENAI_BUDGET_MODE=token_saver
MENTOR_HARD_CASE_ONLY=1
ARIASKA_DRY_RUN=0
EOF
    echo "  ✓ .env updated"
else
    echo "  ✓ .env already configured"
fi

# ── Phase 7: Auto-sync to GitHub ────────────────────────────────
echo ""
echo "[Phase 7] Setting up auto-sync..."

cd "$REPO_DIR"
git config user.email "ariaska-gpu@runner"
git config user.name "Ariaska GPU Runner"

cat > /root/ariaska_autopush.sh << 'SYNCEOF'
#!/usr/bin/env bash
set -euo pipefail
cd /root/Ariaska_RL
git add -A models/distilled/ results/h200_distill/ traces/h200_distill/ data/distill_prep/ data/expert_trajectories/ scripts/h200_run_distill_3h.py 2>/dev/null || true
git diff --cached --quiet && exit 0  # Nothing to commit
git commit -m "gpu-sync: $(date -u +%Y%m%dT%H%M%SZ) — distill checkpoints" --allow-empty >/dev/null 2>&1 || true
git push origin master >/dev/null 2>&1 || echo "[SYNC] push failed (will retry)"
SYNCEOF
chmod +x /root/ariaska_autopush.sh

tmux kill-session -t sync 2>/dev/null || true
tmux new-session -d -s sync
tmux send-keys -t sync "while true; do /root/ariaska_autopush.sh; sleep 600; done" C-m
echo "  ✓ Auto-sync loop started (tmux: sync, every 10 min)"

# ── Summary ─────────────────────────────────────────────────────
echo ""
echo "====================================================================="
echo "  BOOTSTRAP COMPLETE"
echo "====================================================================="
echo "  Repo:     $REPO_DIR"
echo "  venv:     $REPO_DIR/.venv"
echo "  Mentor:   http://127.0.0.1:$MENTOR_PORT/v1"
echo "  Model:    ${ACTIVE_MODEL:-UNAVAILABLE}"
echo "  Sync:     every 10 min → GitHub master"
echo ""
echo "  NEXT: Start training in tmux 'train' session:"
echo "    tmux new -s train"
echo "    cd $REPO_DIR && source .venv/bin/activate"
echo "    python -m scripts.h200_run_distill_3h --seed 42 --max-hours 3 --checkpoint-every 10m"
echo "====================================================================="
