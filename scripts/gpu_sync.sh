#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Ariaska GPU Sync — Phase 43
# Bidirectional artifact sync between GPU instance and local machine.
#
# Usage:
#   ./scripts/gpu_sync.sh push    # Push code to GPU
#   ./scripts/gpu_sync.sh pull    # Pull artifacts from GPU
#   ./scripts/gpu_sync.sh status  # Check GPU instance status
#   ./scripts/gpu_sync.sh logs    # Stream LLM server logs
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; NC='\033[0m'

info()  { echo -e "${CYAN}[SYNC]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }

# ── Config ───────────────────────────────────────────────────────────
GPU_HOST="${ARIASKA_GPU_HOST:-194.228.55.129}"
GPU_PORT="${ARIASKA_GPU_PORT:-37127}"
GPU_DIR="${ARIASKA_GPU_DIR:-/root/Ariaska_RL}"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10"

ssh_cmd() { ssh ${SSH_OPTS} -p "${GPU_PORT}" "root@${GPU_HOST}" "$@"; }
scp_to()  { scp ${SSH_OPTS} -P "${GPU_PORT}" "$1" "root@${GPU_HOST}:$2"; }
scp_from(){ scp ${SSH_OPTS} -P "${GPU_PORT}" "root@${GPU_HOST}:$1" "$2"; }

# ── Commands ─────────────────────────────────────────────────────────
do_push() {
    info "Pushing code to GPU (${GPU_HOST}:${GPU_PORT})..."
    
    # Git-based sync: commit and push locally, pull on GPU
    cd "${LOCAL_DIR}"
    
    # Check for uncommitted changes
    if ! git diff --quiet HEAD 2>/dev/null; then
        warn "Local changes detected — committing..."
        git add -A
        git commit -m "sync: push to GPU $(date +%Y%m%d_%H%M%S)" --allow-empty
    fi
    
    git push origin master 2>/dev/null || warn "Git push failed (check remote)"
    
    # Pull on GPU
    ssh_cmd "cd ${GPU_DIR} && git fetch origin && git reset --hard origin/master"
    ok "Code pushed to GPU"
}

do_pull() {
    info "Pulling artifacts from GPU (${GPU_HOST}:${GPU_PORT})..."
    
    # Force-add and commit artifacts on GPU, then pull locally
    ssh_cmd "cd ${GPU_DIR} && \
        git add -f traces/ postmortems/ artifacts/ runs/ results/ 2>/dev/null; \
        git add -f models/*.pt models/*.pth 2>/dev/null; \
        git commit -m 'sync: GPU artifacts $(date +%Y%m%d_%H%M%S)' --allow-empty 2>/dev/null; \
        git push origin master 2>/dev/null" || warn "GPU commit/push failed"
    
    cd "${LOCAL_DIR}"
    git pull origin master 2>/dev/null || warn "Local pull failed"
    ok "Artifacts pulled from GPU"
}

do_status() {
    info "GPU instance status (${GPU_HOST}:${GPU_PORT})..."
    echo ""
    
    # GPU info
    ssh_cmd "nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu,utilization.gpu \
        --format=csv,noheader 2>/dev/null || echo 'No GPU detected'" || echo "SSH connection failed"
    echo ""
    
    # LLM server status
    LLM_PORT="${ARIASKA_LOCAL_LLM_PORT:-8192}"
    ssh_cmd "curl -s http://127.0.0.1:${LLM_PORT}/v1/models 2>/dev/null | python3 -m json.tool 2>/dev/null || echo 'LLM server not running'" || true
    echo ""
    
    # Disk usage
    ssh_cmd "df -h / | tail -1" || true
    
    # Running processes
    ssh_cmd "ps aux | grep -E 'llama_cpp|ariaska' | grep -v grep || echo 'No Ariaska processes running'" || true
}

do_logs() {
    info "Streaming LLM server logs..."
    ssh_cmd "tail -f /var/log/ariaska_llm.log 2>/dev/null || echo 'No log file found'"
}

do_shell() {
    info "Opening SSH shell to GPU..."
    ssh ${SSH_OPTS} -p "${GPU_PORT}" "root@${GPU_HOST}"
}

do_restart_llm() {
    info "Restarting LLM server on GPU..."
    ssh_cmd "cd ${GPU_DIR} && \
        pkill -f 'llama_cpp.server' 2>/dev/null; sleep 2; \
        source .venv/bin/activate && \
        nohup python -m llama_cpp.server \
            --model /models/Qwen3-32B-Instruct-Q4_K_M.gguf \
            --host 0.0.0.0 --port 8192 \
            --n_gpu_layers 99 --n_ctx 8192 --n_batch 512 \
            --chat_format chatml \
            > /var/log/ariaska_llm.log 2>&1 &"
    sleep 3
    ssh_cmd "curl -s http://127.0.0.1:8192/v1/models 2>/dev/null | python3 -m json.tool 2>/dev/null || echo 'Still starting...'"
    ok "LLM server restart initiated"
}

# ── Main ─────────────────────────────────────────────────────────────
case "${1:-help}" in
    push)        do_push ;;
    pull)        do_pull ;;
    status)      do_status ;;
    logs)        do_logs ;;
    shell)       do_shell ;;
    restart-llm) do_restart_llm ;;
    help|*)
        echo "Ariaska GPU Sync — Phase 43"
        echo ""
        echo "Usage: $0 <command>"
        echo ""
        echo "Commands:"
        echo "  push         Push code to GPU instance"
        echo "  pull         Pull artifacts from GPU"
        echo "  status       Check GPU instance status"
        echo "  logs         Stream LLM server logs"
        echo "  shell        SSH into GPU instance"
        echo "  restart-llm  Restart local LLM server"
        echo ""
        echo "Environment:"
        echo "  ARIASKA_GPU_HOST  GPU hostname (default: ${GPU_HOST})"
        echo "  ARIASKA_GPU_PORT  SSH port (default: ${GPU_PORT})"
        ;;
esac
