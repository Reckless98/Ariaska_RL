# ARIASKA_RL Makefile — Quick Commands for Training and Development
# 
# Usage:
#   make venv          - Create/upgrade .venv, install dependencies
#   make test          - Run pytest
#   make train         - Online training (5 eps, 50 steps)
#   make train-offline - Offline training (no API key)
#   make watch         - Online run with live dashboard
#   make watch-offline - Offline run with live dashboard
#   make smoke         - Quick smoke test
#   make last          - View last training run
#   make clean         - Clean temporary files

.PHONY: venv test train train-offline run run-offline watch watch-offline smoke last clean help

PYTHON := .venv/bin/python
PIP := .venv/bin/pip
PYTEST := .venv/bin/pytest

# Default target
help:
	@echo "ARIASKA_RL — Available Commands:"
	@echo "  make venv          - Create/upgrade .venv, install dependencies"
	@echo "  make test          - Run pytest"
	@echo "  make train         - Online training (5 eps, 50 steps)"
	@echo "  make train-offline - Offline training (5 eps, no API key)"
	@echo "  make watch         - Online run with live dashboard (10 eps)"
	@echo "  make watch-offline - Offline run with live dashboard"
	@echo "  make smoke         - Quick offline smoke test (2 eps, 3 steps)"
	@echo "  make last          - View last training run traces"
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

# Online training (default)
train:
	$(PYTHON) -m core.training.ariaska_trainer --episodes 5 --max-steps 50 --verbosity standard --ui live

# Offline training (no API key)
train-offline:
	unset OPENAI_API_KEY && $(PYTHON) -m core.training.ariaska_trainer --offline --episodes 5 --max-steps 20 --verbosity standard --ui live

# Online run (legacy alias)
run:
	$(PYTHON) -m core.training.ariaska_trainer --episodes 1 --max-steps 50 --verbosity standard

# Offline run (legacy alias)
run-offline:
	$(PYTHON) -m core.training.ariaska_trainer --offline --episodes 2 --max-steps 10 --verbosity standard

# Online run with live dashboard
watch:
	$(PYTHON) -m core.training.ariaska_trainer --episodes 10 --max-steps 50 --verbosity standard --ui live

# Offline run with live dashboard
watch-offline:
	$(PYTHON) -m core.training.ariaska_trainer --offline --episodes 5 --max-steps 10 --verbosity standard --ui live

# Quick smoke test (acceptance criteria A)
smoke:
	unset OPENAI_API_KEY && $(PYTHON) -m core.training.ariaska_trainer --offline --episodes 2 --max-steps 3 --seed 1337 --verbosity standard --ui summary

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
