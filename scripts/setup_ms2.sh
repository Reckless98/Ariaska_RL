#!/usr/bin/env bash
# scripts/setup_ms2.sh — Metasploitable 2 Lab Setup for Ariaska_RL
# ═══════════════════════════════════════════════════════════════════
# This script sets up a Docker-based Metasploitable 2 pentesting lab.
#
# What it does:
#   1. Verifies Docker + Docker Compose are installed
#   2. Pulls the Metasploitable 2 image
#   3. Starts the lab network (MS2 + ariaska-attacker)
#   4. Waits for MS2 services to be ready
#   5. Runs a health check against key vulnerable services
#   6. Reports lab status
#
# Usage:
#   ./scripts/setup_ms2.sh          # Full setup + start
#   ./scripts/setup_ms2.sh start    # Start existing lab
#   ./scripts/setup_ms2.sh stop     # Stop lab
#   ./scripts/setup_ms2.sh status   # Check lab status
#   ./scripts/setup_ms2.sh health   # Run health checks
#   ./scripts/setup_ms2.sh reset    # Stop + remove + restart
#
# Requirements:
#   - Docker Engine 20.10+
#   - Docker Compose v2+
#   - ~800MB disk space for MS2 image
#   - OPENAI_API_KEY in .env (for live training)

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────
MS2_IP="172.28.0.10"
MS2_IMAGE="tleemcjr/metasploitable2"
MS2_CONTAINER="metasploitable2"
COMPOSE_FILES="-f docker-compose.yml -f docker-compose.metasploitable.yml"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HEALTH_TIMEOUT=120  # seconds to wait for MS2 services

# ── Colors ───────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log_info()  { echo -e "${CYAN}[INFO]${NC} $1"; }
log_ok()    { echo -e "${GREEN}[✓]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[⚠]${NC} $1"; }
log_error() { echo -e "${RED}[✗]${NC} $1"; }

# ── Check Prerequisites ─────────────────────────────────────────────
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Docker
    if ! command -v docker &>/dev/null; then
        log_error "Docker is not installed. Install from: https://docs.docker.com/get-docker/"
        exit 1
    fi
    docker_version=$(docker version --format '{{.Server.Version}}' 2>/dev/null || echo "unknown")
    log_ok "Docker installed (${docker_version})"

    # Docker Compose (v2)
    if docker compose version &>/dev/null; then
        compose_version=$(docker compose version --short 2>/dev/null || echo "unknown")
        log_ok "Docker Compose v2 installed (${compose_version})"
    elif command -v docker-compose &>/dev/null; then
        log_warn "Using legacy docker-compose. Consider upgrading to Compose v2."
    else
        log_error "Docker Compose is not installed."
        exit 1
    fi

    # Check Docker daemon
    if ! docker info &>/dev/null; then
        log_error "Docker daemon is not running. Start it with: sudo systemctl start docker"
        exit 1
    fi
    log_ok "Docker daemon is running"

    # .env file
    if [ -f "${PROJECT_DIR}/.env" ]; then
        if grep -q "OPENAI_API_KEY" "${PROJECT_DIR}/.env"; then
            log_ok "OPENAI_API_KEY found in .env"
        else
            log_warn "OPENAI_API_KEY not found in .env — live training will use offline mode"
        fi
    else
        log_warn "No .env file found — create one with OPENAI_API_KEY for live training"
    fi
}

# ── Pull MS2 Image ───────────────────────────────────────────────────
pull_image() {
    log_info "Pulling Metasploitable 2 Docker image..."
    if docker image inspect "$MS2_IMAGE" &>/dev/null; then
        log_ok "Image ${MS2_IMAGE} already exists"
    else
        docker pull "$MS2_IMAGE"
        log_ok "Image ${MS2_IMAGE} pulled successfully"
    fi
}

# ── Start Lab ────────────────────────────────────────────────────────
start_lab() {
    log_info "Starting Metasploitable 2 lab..."
    cd "$PROJECT_DIR"
    docker compose $COMPOSE_FILES up -d metasploitable2
    log_ok "Metasploitable 2 container started"
}

# ── Stop Lab ─────────────────────────────────────────────────────────
stop_lab() {
    log_info "Stopping Metasploitable 2 lab..."
    cd "$PROJECT_DIR"
    docker compose $COMPOSE_FILES down
    log_ok "Lab stopped"
}

# ── Reset Lab ────────────────────────────────────────────────────────
reset_lab() {
    log_info "Resetting Metasploitable 2 lab..."
    cd "$PROJECT_DIR"
    docker compose $COMPOSE_FILES down -v --remove-orphans
    docker compose $COMPOSE_FILES up -d metasploitable2
    log_ok "Lab reset complete"
}

# ── Wait for Services ────────────────────────────────────────────────
wait_for_services() {
    log_info "Waiting for MS2 services to come up (timeout: ${HEALTH_TIMEOUT}s)..."

    local start_time=$(date +%s)
    local key_ports=(21 22 23 80 445 1524 3306)
    local ready_count=0
    local required=4  # Need at least 4 of 7 key ports

    while true; do
        local elapsed=$(( $(date +%s) - start_time ))
        if [ $elapsed -ge $HEALTH_TIMEOUT ]; then
            log_warn "Timeout after ${HEALTH_TIMEOUT}s — ${ready_count}/${#key_ports[@]} ports ready"
            return 1
        fi

        ready_count=0
        for port in "${key_ports[@]}"; do
            if timeout 2 bash -c "echo >/dev/tcp/${MS2_IP}/${port}" 2>/dev/null; then
                ready_count=$((ready_count + 1))
            fi
        done

        if [ $ready_count -ge $required ]; then
            log_ok "MS2 services ready (${ready_count}/${#key_ports[@]} ports responding after ${elapsed}s)"
            return 0
        fi

        echo -ne "\r  Waiting... ${ready_count}/${#key_ports[@]} ports up (${elapsed}s elapsed)"
        sleep 3
    done
}

# ── Health Check ─────────────────────────────────────────────────────
health_check() {
    log_info "Running MS2 health checks..."
    echo ""

    local checks=0
    local passed=0

    # Port checks with service identification
    declare -A service_map=(
        [21]="vsftpd 2.3.4 (backdoor)"
        [22]="OpenSSH 4.7p1"
        [23]="Telnet (default creds)"
        [25]="Postfix SMTP"
        [80]="Apache 2.2.8 (DVWA)"
        [139]="Samba 3.0.20 (CVE-2007-2447)"
        [445]="SMB"
        [512]="rexec (no auth)"
        [1099]="Java RMI (RCE)"
        [1524]="ingreslock backdoor (instant root)"
        [2049]="NFS (world-readable)"
        [3306]="MySQL 5.0.51a (no password)"
        [5432]="PostgreSQL 8.3.0 (RCE)"
        [5900]="VNC (password: password)"
        [6667]="UnrealIRCd 3.2.8.1 (backdoor)"
        [8180]="Tomcat (tomcat:tomcat)"
    )

    for port in 21 22 23 25 80 139 445 512 1099 1524 2049 3306 5432 5900 6667 8180; do
        checks=$((checks + 1))
        local svc="${service_map[$port]:-unknown}"
        if timeout 2 bash -c "echo >/dev/tcp/${MS2_IP}/${port}" 2>/dev/null; then
            log_ok "Port ${port}: ${svc}"
            passed=$((passed + 1))
        else
            log_error "Port ${port}: ${svc} — NOT RESPONDING"
        fi
    done

    echo ""
    if [ $passed -ge 12 ]; then
        log_ok "Health check: ${passed}/${checks} services up — MS2 is FULLY OPERATIONAL"
    elif [ $passed -ge 8 ]; then
        log_warn "Health check: ${passed}/${checks} services up — MS2 is PARTIALLY operational"
    else
        log_error "Health check: ${passed}/${checks} services up — MS2 may have issues"
    fi
    echo ""
    return 0
}

# ── Status ───────────────────────────────────────────────────────────
show_status() {
    log_info "Lab status:"
    echo ""

    # Container status
    if docker ps --format '{{.Names}}' | grep -q "$MS2_CONTAINER"; then
        local ms2_status=$(docker inspect --format '{{.State.Status}}' "$MS2_CONTAINER" 2>/dev/null || echo "unknown")
        local ms2_uptime=$(docker inspect --format '{{.State.StartedAt}}' "$MS2_CONTAINER" 2>/dev/null || echo "unknown")
        log_ok "Metasploitable 2: ${ms2_status} (started: ${ms2_uptime})"
        log_info "  IP: ${MS2_IP}"
        log_info "  Image: ${MS2_IMAGE}"
    else
        log_error "Metasploitable 2: NOT RUNNING"
        log_info "  Start with: ./scripts/setup_ms2.sh start"
    fi

    # Attacker container
    if docker ps --format '{{.Names}}' | grep -q "ariaska-attacker"; then
        log_ok "Ariaska attacker: running"
    else
        log_info "Ariaska attacker: not running (start with full compose up)"
    fi

    # Network
    if docker network ls | grep -q "pentest-net"; then
        log_ok "pentest-net: exists (172.28.0.0/24)"
    else
        log_warn "pentest-net: not created yet"
    fi
    echo ""
}

# ── Training Command Helper ─────────────────────────────────────────
show_training_help() {
    echo ""
    log_info "=== LIVE TRAINING COMMANDS ==="
    echo ""
    echo "  # From host (targets MS2 in Docker):"
    echo "  python ariaska_cli.py smart-train --env msf --target ${MS2_IP} --episodes 10"
    echo ""
    echo "  # From Docker attacker container:"
    echo "  docker compose ${COMPOSE_FILES} run ariaska-attacker"
    echo ""
    echo "  # Quick validation (3 episodes):"
    echo "  make train-msf"
    echo ""
}

# ── Main ─────────────────────────────────────────────────────────────
main() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║     Ariaska_RL — Metasploitable 2 Lab Setup                ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""

    local action="${1:-setup}"

    case "$action" in
        setup|install)
            check_prerequisites
            pull_image
            start_lab
            wait_for_services
            health_check
            show_training_help
            ;;
        start)
            start_lab
            wait_for_services
            health_check
            ;;
        stop)
            stop_lab
            ;;
        reset)
            reset_lab
            wait_for_services
            health_check
            ;;
        status)
            show_status
            ;;
        health)
            health_check
            ;;
        help|--help|-h)
            echo "Usage: $0 [setup|start|stop|status|health|reset|help]"
            echo ""
            echo "  setup   - Full setup: check prerequisites, pull image, start, health check"
            echo "  start   - Start existing lab"
            echo "  stop    - Stop lab"
            echo "  reset   - Stop, remove, and restart lab"
            echo "  status  - Show container status"
            echo "  health  - Run health checks on MS2 services"
            echo "  help    - Show this help"
            ;;
        *)
            log_error "Unknown action: $action"
            echo "Usage: $0 [setup|start|stop|status|health|reset|help]"
            exit 1
            ;;
    esac
}

main "$@"
