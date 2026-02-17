# ARIASKA_RL Makefile — Phase 12.1
#
# Unified entry point: ariaska_cli.py smart-train
#
# Usage:
#   make venv            - Create/upgrade .venv, install dependencies
#   make test            - Run pytest (full suite)
#   make test-cap        - Run Cap regression harness only
#   make test-fast       - Run fast subset (no integration)
#   make train           - Default training (MS3 medium, 100 eps)
#   make train-quick     - Quick training (MS3 medium, 10 eps)
#   make train-ms3       - MS3 medium 10 episodes
#   make train-ms3-medium- MS3 medium 100 episodes
#   make train-ms3-hard  - MS3 hard 10 episodes
#   make train-ms2       - MS2 10 episodes (live)
#   make train-ms2-hard  - MS2 hard 100 episodes (live)
#   make train-htb       - HTB target training (requires --target)
#   make smoke           - Quick smoke test (3 eps)
#   make overnight       - Overnight progressive training (300 eps)
#   make last            - View last training run
#   make status          - System diagnostics
#   make clean           - Clean temporary files

.PHONY: venv test test-cap test-fast train train-quick train-ms3 train-ms3-medium \
        train-ms3-hard train-ms2 train-ms2-hard train-htb train-msf smoke \
        ms2-setup ms2-status ms2-health ms2-stop overnight overnight-quick \
        last status clean help traces

PYTHON := .venv/bin/python
PIP := .venv/bin/pip
PYTEST := .venv/bin/pytest

# Default target
help:
	@echo "ARIASKA_RL v12.1 — Multi-Agent Cybersec RL"
	@echo ""
	@echo "  Training (MS3 LIVE — default):"
	@echo "    make train           - MS3 LIVE medium, 100 eps (default)"
	@echo "    make train-quick     - MS3 LIVE medium, 10 eps (quick test)"
	@echo "    make train-ms3       - MS3 LIVE medium, 10 eps"
	@echo "    make train-ms3-medium- MS3 LIVE medium, 100 eps"
	@echo "    make train-ms3-hard  - MS3 LIVE hard, 10 eps"
	@echo ""
	@echo "  Training (MS2):"
	@echo "    make train-ms2       - MS2 live, 10 eps"
	@echo "    make train-ms2-hard  - MS2 live, 100 eps"
	@echo "    make train-msf       - MS2 live, 20 eps"
	@echo ""
	@echo "  Training (HTB):"
	@echo "    make train-htb TARGET=10.129.x.x  - HTB target, 50 eps"
	@echo ""
	@echo "  Testing:"
	@echo "    make test            - Full pytest suite"
	@echo "    make test-cap        - Cap regression harness only"
	@echo "    make test-fast       - Fast subset (skip integration)"
	@echo ""
	@echo "  Utilities:"
	@echo "    make smoke           - 3 eps, fast validation"
	@echo "    make overnight       - Progressive 300 eps"
	@echo "    make overnight-quick - Quick overnight 30 eps"
	@echo "    make venv            - Setup virtual environment"
	@echo "    make status          - System diagnostics"
	@echo "    make last            - View last training run"
	@echo "    make clean           - Clean temp files"

# Create/upgrade virtual environment and install dependencies
venv:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install torch --index-url https://download.pytorch.org/whl/cpu 2>/dev/null || $(PIP) install torch
	$(PIP) install rich pytest
	@echo "✓ Virtual environment ready. Activate with: source .venv/bin/activate"

# Run tests
test:
	$(PYTEST) -q

# Cap regression harness only
test-cap:
	$(PYTEST) tests/test_cap_regression.py -v --tb=short

# Fast tests (skip slow integration tests)
test-fast:
	$(PYTEST) -q -x --ignore=tests/test_smart_integration.py --ignore=tests/test_training_smoke.py

# ── MS3 LIVE Training (default) ─────────────────────────────────────────────
# Default: MS3 LIVE, 100 episodes (ms3_live difficulty)
train:
	$(PYTHON) ariaska_cli.py smart-train --episodes 100 --steps 40 --seed 42 --env ms3 --difficulty ms3_live --verbosity verbose

# Quick: MS3 LIVE, 10 episodes
train-quick:
	$(PYTHON) ariaska_cli.py smart-train --episodes 10 --steps 40 --seed 42 --env ms3 --difficulty ms3_live --verbosity verbose

# MS3 LIVE, 10 episodes (alias)
train-ms3:
	$(PYTHON) ariaska_cli.py smart-train --episodes 10 --steps 40 --seed 42 --env ms3 --difficulty ms3_live --verbosity verbose

# MS3 LIVE medium (easier), 100 episodes
train-ms3-medium:
	$(PYTHON) ariaska_cli.py smart-train --episodes 100 --steps 40 --seed 42 --env ms3 --difficulty ms3_medium --verbosity verbose

# MS3 LIVE hard (blocks creds too), 10 episodes
train-ms3-hard:
	$(PYTHON) ariaska_cli.py smart-train --episodes 10 --steps 40 --seed 42 --env ms3 --difficulty ms3_hard --verbosity verbose

# MS3 LIVE, 10 episodes with custom seed (for iterative runs)
train-ms3-iter:
	$(PYTHON) ariaska_cli.py smart-train --episodes 10 --steps 40 --seed $(SEED) --env ms3 --difficulty ms3_live --verbosity verbose

# ── MS2 Training ─────────────────────────────────────────────────────────────
# MS2 live, 10 episodes
train-ms2:
	$(PYTHON) ariaska_cli.py smart-train --episodes 10 --steps 40 --seed 42 --env ms2 --verbosity verbose

# MS2 live, 100 episodes (longer runs)
train-ms2-hard:
	$(PYTHON) ariaska_cli.py smart-train --episodes 100 --steps 40 --seed 42 --env ms2 --difficulty hard --verbosity verbose

# Live Metasploitable 2 training (legacy)
train-msf:
	$(PYTHON) ariaska_cli.py smart-train --episodes 20 --steps 40 --env ms2 --verbosity verbose

# HTB target training (pass TARGET=<ip>)
TARGET ?= 10.129.5.41
train-htb:
	$(PYTHON) ariaska_cli.py smart-train --episodes 50 --steps 60 --seed 42 --env htb --target $(TARGET) --verbosity verbose

# MS2 Docker lab setup
ms2-setup:
	@bash scripts/setup_ms2.sh setup

ms2-status:
	@bash scripts/setup_ms2.sh status

ms2-health:
	@bash scripts/setup_ms2.sh health

ms2-stop:
	@bash scripts/setup_ms2.sh stop

# Quick smoke test (MS3 LIVE, 3 eps)
smoke:
	$(PYTHON) ariaska_cli.py smart-train --episodes 3 --steps 20 --seed 1337 --env ms3 --difficulty ms3_live --verbosity verbose

# Overnight training: progressive difficulty (300 episodes)
overnight:
	bash scripts/overnight_train.sh

# Overnight quick test (30 episodes)
overnight-quick:
	bash scripts/overnight_train.sh --quick

# View last training run
last:
	@LAST_RUN=$$(ls -t traces/ 2>/dev/null | head -1); \
	if [ -z "$$LAST_RUN" ]; then echo "No training runs found"; exit 1; fi; \
	echo "Run ID: $$LAST_RUN"; \
	echo ""; \
	echo "=== Steps (last 20) ==="; \
	tail -20 traces/$$LAST_RUN/steps.jsonl 2>/dev/null || echo "(no steps.jsonl)"; \
	echo ""; \
	echo "=== Mentor Calls (last 10) ==="; \
	tail -10 traces/$$LAST_RUN/mentor.jsonl 2>/dev/null || echo "(no mentor.jsonl)"

# System diagnostics
status:
	$(PYTHON) ariaska_cli.py status

# Clean temporary files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	rm -rf .pytest_cache 2>/dev/null || true
	@echo "✓ Cleaned temporary files"

# Show trace files
traces:
	@echo "Recent training runs:"
	@ls -lt traces/ | head -10
