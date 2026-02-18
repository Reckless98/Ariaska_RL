#!/usr/bin/env bash
# ============================================================================
# ARIASKA_RL — Phase 17: Unified HTB Launch Script
# ============================================================================
# Usage:
#   ./scripts/launch_htb.sh [TARGET_IP] [EPISODES] [STEPS]
#
# Examples:
#   ./scripts/launch_htb.sh 172.28.0.10 50 40   # MS2, 50 episodes
#   ./scripts/launch_htb.sh 172.28.0.11 100 60  # MS3, 100 episodes
#   ./scripts/launch_htb.sh 10.10.10.40 30 60   # HTB box
#   ./scripts/launch_htb.sh                      # Defaults: MS3, 20ep, 40steps
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ── Defaults ────────────────────────────────────────────────────────────────
TARGET="${1:-172.28.0.11}"
EPISODES="${2:-20}"
STEPS="${3:-40}"
SEED="${ARIASKA_SEED:-42}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          ARIASKA_RL — HTB Launch Sequence                   ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Target:   $TARGET"
echo "║  Episodes: $EPISODES"
echo "║  Steps:    $STEPS"
echo "║  Seed:     $SEED"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Step 1: Check Python environment ────────────────────────────────────────
echo "[1/6] Checking Python environment..."
if [[ -f ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
else
    echo "ERROR: No Python found. Run: make venv"
    exit 1
fi
echo "  ✓ Python: $($PYTHON --version)"

# ── Step 2: Verify API key ──────────────────────────────────────────────────
echo "[2/6] Checking API key..."
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    if [[ -f ".env" ]]; then
        # Try to source .env
        set +u
        source .env 2>/dev/null || true
        set -u
    fi
fi
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "  ⚠ OPENAI_API_KEY not set — running in OFFLINE mode (no LLM)"
    echo "    Set via: export OPENAI_API_KEY=sk-..."
else
    echo "  ✓ API key present (${#OPENAI_API_KEY} chars)"
fi

# ── Step 3: Activate all HTB feature flags ──────────────────────────────────
echo "[3/6] Activating HTB feature flags..."
source "$SCRIPT_DIR/activate_htb_flags.sh"
echo "  ✓ All flags activated"

# ── Step 4: Verify flags ────────────────────────────────────────────────────
echo "[4/6] Verifying flag activation..."
if [[ -f "$SCRIPT_DIR/verify_flags.py" ]]; then
    $PYTHON "$SCRIPT_DIR/verify_flags.py" 2>/dev/null || echo "  ⚠ Flag verification had warnings (non-fatal)"
else
    echo "  ⚠ verify_flags.py not found, skipping"
fi

# ── Step 5: Check critical tools ────────────────────────────────────────────
echo "[5/6] Checking critical pentesting tools..."
MISSING_TOOLS=()
for tool in nmap gobuster hydra sshpass nikto searchsploit curl wget ssh telnet mysql smbclient; do
    if ! command -v "$tool" &>/dev/null; then
        MISSING_TOOLS+=("$tool")
    fi
done

if [[ ${#MISSING_TOOLS[@]} -gt 0 ]]; then
    echo "  ⚠ Missing tools: ${MISSING_TOOLS[*]}"
    echo "    Install via: apt install ${MISSING_TOOLS[*]}"
    echo "    Continuing anyway — commands using these tools will fail gracefully."
else
    echo "  ✓ All critical tools available"
fi

# ── Step 6: Check target reachability ───────────────────────────────────────
echo "[6/6] Checking target reachability..."
if ping -c 1 -W 2 "$TARGET" &>/dev/null; then
    echo "  ✓ Target $TARGET is reachable"
else
    echo "  ⚠ Target $TARGET unreachable — check Docker/VPN"
    echo "    For MS2/MS3: docker compose -f docker-compose.metasploitable.yml up -d"
    echo "    For HTB: ensure VPN is connected"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Launching Ariaska training..."
echo "════════════════════════════════════════════════════════════════"
echo ""

# ── Launch ──────────────────────────────────────────────────────────────────
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
exec $PYTHON ariaska_cli.py smart-train \
    --episodes "$EPISODES" \
    --steps "$STEPS" \
    --seed "$SEED" \
    --env msf \
    --target "$TARGET"
