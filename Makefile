# ARIASKA_RL Makefile — Phase 5
#
# Unified entry point: ariaska_cli.py smart-train
#
# Usage:
#   make venv          - Create/upgrade .venv, install dependencies
#   make test          - Run pytest
#   make train         - Training (100 eps, 120 steps, simulated)
#   make train-quick   - Quick training (10 eps, 120 steps)
#   make train-msf     - Live Metasploitable 2 training
#   make smoke         - Quick smoke test (3 eps, 10 steps)
#   make last          - View last training run
#   make status        - System diagnostics
#   make clean         - Clean temporary files

.PHONY: venv test train train-quick train-msf smoke last status clean help

PYTHON := .venv/bin/python
PIP := .venv/bin/pip
PYTEST := .venv/bin/pytest

# Default target
help:
	@echo "ARIASKA_RL v5.0 — Metasploitable 2 Ready"
	@echo ""
	@echo "  make venv          - Create/upgrade .venv, install deps"
	@echo "  make test          - Run pytest"
	@echo "  make train         - Full training (100 eps, 120 steps)"
	@echo "  make train-quick   - Quick training (10 eps, 120 steps)"
	@echo "  make train-msf     - Live Metasploitable 2 training"
	@echo "  make smoke         - Smoke test (3 eps, 10 steps)"
	@echo "  make last          - View last training run traces"
	@echo "  make status        - System diagnostics"
	@echo "  make clean         - Clean temporary files"

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

# Full MS2 live training (100 episodes, 40 steps)
train:
	$(PYTHON) ariaska_cli.py smart-train --episodes 100 --steps 40 --seed 42 --env ms2 --verbosity verbose

# Quick MS2 live training (10 episodes)
train-quick:
	$(PYTHON) ariaska_cli.py smart-train --episodes 10 --steps 40 --seed 42 --env ms2 --verbosity verbose

# Live Metasploitable 2 training (longer)
train-msf:
	$(PYTHON) ariaska_cli.py smart-train --episodes 20 --steps 40 --env ms2 --verbosity verbose

# MS2 Docker lab setup
ms2-setup:
	@bash scripts/setup_ms2.sh setup

ms2-status:
	@bash scripts/setup_ms2.sh status

ms2-health:
	@bash scripts/setup_ms2.sh health

ms2-stop:
	@bash scripts/setup_ms2.sh stop

# Quick smoke test (MS2 live)
smoke:
	$(PYTHON) ariaska_cli.py smart-train --episodes 3 --steps 20 --seed 1337 --env ms2 --verbosity verbose

# MS3 training: medium difficulty (10 episodes)
train-ms3:
	$(PYTHON) ariaska_cli.py smart-train --episodes 10 --steps 40 --seed 42 --env sim --difficulty ms3_medium --verbosity verbose

# MS3 training: hard difficulty (10 episodes)
train-ms3-hard:
	$(PYTHON) ariaska_cli.py smart-train --episodes 10 --steps 40 --seed 42 --env sim --difficulty ms3_hard --verbosity verbose

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
