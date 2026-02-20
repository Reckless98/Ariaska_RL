#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Ariaska GPU Setup Script — Phase 43
# Sets up a fresh GPU instance (vast.ai / runpod / bare metal)
# Usage: ssh -p PORT root@HOST < scripts/gpu_setup.sh
#        OR: scp scripts/gpu_setup.sh root@HOST: && ssh root@HOST bash gpu_setup.sh
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Colors ───────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail()  { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }

# ── Config ───────────────────────────────────────────────────────────
REPO_URL="${ARIASKA_REPO_URL:-https://github.com/Reckless98/Ariaska_RL.git}"
BRANCH="${ARIASKA_BRANCH:-master}"
INSTALL_DIR="${ARIASKA_INSTALL_DIR:-/root/Ariaska_RL}"
MODEL_DIR="${ARIASKA_MODEL_DIR:-/models}"
MODEL_NAME="Qwen3-32B-Instruct-Q4_K_M.gguf"
MODEL_REPO="Qwen/Qwen3-32B-Instruct-GGUF"
VENV_DIR="${INSTALL_DIR}/.venv"
LOCAL_LLM_PORT="${ARIASKA_LOCAL_LLM_PORT:-8192}"

echo "═══════════════════════════════════════════════════════════════════"
echo "  Ariaska GPU Setup — Phase 43"
echo "═══════════════════════════════════════════════════════════════════"

# ── Step 1: System info ──────────────────────────────────────────────
info "System diagnostics..."
echo "  Hostname: $(hostname)"
echo "  OS: $(cat /etc/os-release 2>/dev/null | grep PRETTY_NAME | cut -d= -f2 || uname -s)"
echo "  CPU: $(nproc) cores"
echo "  RAM: $(free -h | awk '/Mem:/{print $2}')"
echo "  Disk: $(df -h / | awk 'NR==2{print $4}') free"

# ── Step 2: Check GPU ────────────────────────────────────────────────
info "Checking GPU..."
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    ok "GPU found: ${GPU_NAME} (${GPU_VRAM} MiB VRAM)"
else
    warn "nvidia-smi not found — CPU-only mode"
fi

# ── Step 3: Install system deps ──────────────────────────────────────
info "Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq git python3 python3-pip python3-venv curl wget jq htop > /dev/null 2>&1
ok "System dependencies installed"

# ── Step 4: Clone / update repo ──────────────────────────────────────
if [ -d "${INSTALL_DIR}/.git" ]; then
    info "Repo already exists, pulling latest..."
    cd "${INSTALL_DIR}"
    git fetch origin
    git reset --hard "origin/${BRANCH}"
    ok "Repo updated"
else
    info "Cloning Ariaska_RL..."
    git clone --branch "${BRANCH}" --single-branch "${REPO_URL}" "${INSTALL_DIR}"
    cd "${INSTALL_DIR}"
    ok "Repo cloned to ${INSTALL_DIR}"
fi

# ── Step 5: Python venv + deps ──────────────────────────────────────
info "Setting up Python virtual environment..."
if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"

info "Installing Python dependencies..."
pip install --upgrade pip setuptools wheel -q
pip install -r requirements.txt --no-cache-dir -q 2>&1 | tail -3

# Ensure llama-cpp-python is installed with CUDA support
if python -c "import llama_cpp" 2>/dev/null; then
    ok "llama-cpp-python already installed"
else
    info "Installing llama-cpp-python with CUDA support..."
    CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-cache-dir -q 2>&1 | tail -3
fi

# Ensure huggingface-hub is available for model downloads
pip install huggingface-hub -q 2>&1 | tail -1
ok "Python environment ready"

# ── Step 6: Download model ──────────────────────────────────────────
mkdir -p "${MODEL_DIR}"
MODEL_PATH="${MODEL_DIR}/${MODEL_NAME}"

if [ -f "${MODEL_PATH}" ]; then
    MODEL_SIZE=$(du -h "${MODEL_PATH}" | cut -f1)
    ok "Model already downloaded: ${MODEL_PATH} (${MODEL_SIZE})"
else
    info "Downloading ${MODEL_NAME} from HuggingFace..."
    info "This may take 10-20 minutes depending on connection speed."
    huggingface-cli download "${MODEL_REPO}" "${MODEL_NAME}" \
        --local-dir "${MODEL_DIR}" \
        --local-dir-use-symlinks False
    
    if [ -f "${MODEL_PATH}" ]; then
        MODEL_SIZE=$(du -h "${MODEL_PATH}" | cut -f1)
        ok "Model downloaded: ${MODEL_PATH} (${MODEL_SIZE})"
    else
        fail "Model download failed — file not found at ${MODEL_PATH}"
    fi
fi

# ── Step 7: Create .env ─────────────────────────────────────────────
ENV_FILE="${INSTALL_DIR}/.env"
if [ ! -f "${ENV_FILE}" ]; then
    info "Creating .env from template..."
    cp "${INSTALL_DIR}/.env.example" "${ENV_FILE}"
fi

# Ensure GPU-specific env vars are set
if ! grep -q "FF_LOCAL_LLM" "${ENV_FILE}"; then
    cat >> "${ENV_FILE}" <<EOF

# ── Phase 43: GPU Acceleration (auto-configured) ────────────────────
FF_LOCAL_LLM=1
FF_LOCAL_LLM_OFFLOAD_NANO=1
FF_LOCAL_LLM_OFFLOAD_MINI=1
ARIASKA_LOCAL_MODEL_PATH=${MODEL_PATH}
ARIASKA_LOCAL_LLM_PORT=${LOCAL_LLM_PORT}
ARIASKA_LOCAL_BACKEND=llama-cpp
MENTOR_STRATEGY=local_first
ENABLE_DUAL_MENTOR=1
ARIASKA_DRY_RUN=0
EOF
    ok ".env configured for GPU acceleration"
else
    ok ".env already has GPU config"
fi

# ── Step 8: Start local LLM server ──────────────────────────────────
info "Starting local LLM server on port ${LOCAL_LLM_PORT}..."

# Kill any existing server
pkill -f "llama_cpp.server" 2>/dev/null || true
sleep 1

# Detect GPU layers
N_GPU_LAYERS=99  # Offload all layers to GPU
if [ -n "${GPU_VRAM:-}" ] && [ "${GPU_VRAM:-0}" -lt 20000 ]; then
    N_GPU_LAYERS=40  # Partial offload for smaller GPUs
    warn "Limited VRAM (${GPU_VRAM} MiB) — using partial GPU offload (${N_GPU_LAYERS} layers)"
fi

# Start server in background
nohup python -m llama_cpp.server \
    --model "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port "${LOCAL_LLM_PORT}" \
    --n_gpu_layers "${N_GPU_LAYERS}" \
    --n_ctx 8192 \
    --n_batch 512 \
    --chat_format chatml \
    > /var/log/ariaska_llm.log 2>&1 &

LLM_PID=$!
info "LLM server starting (PID: ${LLM_PID})..."

# Wait for server to be ready
MAX_WAIT=120
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s "http://127.0.0.1:${LOCAL_LLM_PORT}/v1/models" >/dev/null 2>&1; then
        ok "Local LLM server ready on port ${LOCAL_LLM_PORT}"
        break
    fi
    sleep 2
    WAITED=$((WAITED + 2))
    if [ $((WAITED % 10)) -eq 0 ]; then
        info "Waiting for LLM server... (${WAITED}s / ${MAX_WAIT}s)"
    fi
done

if [ $WAITED -ge $MAX_WAIT ]; then
    warn "LLM server did not start within ${MAX_WAIT}s — check /var/log/ariaska_llm.log"
else
    # Quick health check
    info "Running LLM health check..."
    RESPONSE=$(curl -s "http://127.0.0.1:${LOCAL_LLM_PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "local",
            "messages": [{"role": "user", "content": "Say OK"}],
            "max_tokens": 5
        }' 2>/dev/null || echo '{}')
    
    if echo "${RESPONSE}" | jq -e '.choices[0].message.content' >/dev/null 2>&1; then
        ok "LLM health check passed"
    else
        warn "LLM health check inconclusive — response: ${RESPONSE:0:200}"
    fi
fi

# ── Step 9: Run smoke test ──────────────────────────────────────────
info "Running Ariaska smoke test..."
cd "${INSTALL_DIR}"
source "${VENV_DIR}/bin/activate"
ARIASKA_DRY_RUN=1 PYTHONPATH="${INSTALL_DIR}" python -m pytest tests/ -x --tb=short -q --timeout=120 \
    --ignore=tests/test_online_makes_mentor_calls.py 2>&1 | tail -10

# ── Done ────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  ✓ Ariaska GPU Setup Complete"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "  Install dir:  ${INSTALL_DIR}"
echo "  Model:        ${MODEL_PATH}"
echo "  LLM server:   http://127.0.0.1:${LOCAL_LLM_PORT}"
echo "  LLM log:      /var/log/ariaska_llm.log"
echo ""
echo "  Quick start:"
echo "    cd ${INSTALL_DIR} && source .venv/bin/activate"
echo "    python ariaska_cli.py smart-train --target 10.10.10.10 --steps 200"
echo ""
echo "  GPU status:"
echo "    nvidia-smi"
echo "    curl http://127.0.0.1:${LOCAL_LLM_PORT}/v1/models"
echo ""
