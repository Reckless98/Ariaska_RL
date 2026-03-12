# ═══════════════════════════════════════════════════════════════════════
#  ARIASKA_RL — Phase 43 Makefile
#  Multi-Agent Autonomous RL Pentesting System
# ═══════════════════════════════════════════════════════════════════════
#
# Quick reference:
#   make ariaska          - Interactive command center (Rich TUI)
#   make ctf target=IP    - Live CTF engagement
#   make test             - Full test suite
#   make help             - Show all targets

.PHONY: venv test test-cap test-fast train train-quick train-ms3 train-ms3-medium \
        train-ms3-hard train-ms2 train-ms2-hard train-htb train-msf smoke \
        ms2-setup ms2-status ms2-health ms2-stop overnight overnight-quick \
        last status clean help traces ctf \
        ariaska gpu gpu-setup gpu-quick gpu-ctf gpu-htb gpu-sync gpu-status \
        gpu-shell gpu-model gpu-restart-llm \
        gpu-distill gpu-distill-finetune gpu-grpo gpu-grpo-large gpu-session \
        eval-baseline eval-after eval-compare \
        unify-all unify-validate unify-stats sync-local \
        watch-gpu gpu-watch gpu-dashboard gpu-logs gpu-train-status

# ── GPU SSH Configuration ────────────────────────────────────────────
GPU_HOST ?= root@212.247.220.172
GPU_PORT ?= 25107
GPU_SSH  = ssh -o ConnectTimeout=20 -o ServerAliveInterval=10 -p $(GPU_PORT) $(GPU_HOST)

PYTHON := .venv/bin/python
PIP := .venv/bin/pip
PYTEST := .venv/bin/pytest

# ── Default target ───────────────────────────────────────────────────
help:
	@echo ""
	@echo "  ┌──────────────────────────────────────────────────────────┐"
	@echo "  │          ARIASKA_RL — Phase 43 Command Reference        │"
	@echo "  │    Multi-Agent Autonomous RL Pentesting System          │"
	@echo "  └──────────────────────────────────────────────────────────┘"
	@echo ""
	@echo "  ENGAGE"
	@echo "    make ariaska                        Interactive command center (Rich TUI)"
	@echo "    make ctf target=IP                  Live CTF engagement (500 steps)"
	@echo "    make ctf target=IP steps=N seed=N   Custom CTF engagement"
	@echo ""
	@echo "  TRAIN"
	@echo "    make train                          MS3 live, 100 episodes (default)"
	@echo "    make train-quick                    MS3 live, 10 episodes"
	@echo "    make train-ms3                      MS3 live, 10 episodes"
	@echo "    make train-ms3-medium               MS3 medium difficulty, 100 episodes"
	@echo "    make train-ms3-hard                 MS3 hard difficulty, 10 episodes"
	@echo "    make train-ms2                      MS2 live, 10 episodes"
	@echo "    make train-ms2-hard                 MS2 live, 100 episodes"
	@echo "    make train-htb TARGET=IP            HTB target, 50 episodes"
	@echo "    make smoke                          Quick validation (3 episodes)"
	@echo "    make overnight                      Progressive 300 episodes"
	@echo ""
	@echo "  GPU / LLM"
	@echo "    make gpu                            GPU-accelerated training"
	@echo "    make gpu-setup                      Setup GPU instance (vast.ai/runpod)"
	@echo "    make gpu-quick                      Quick GPU training (50 steps)"
	@echo "    make gpu-ctf T=IP                   GPU CTF engagement"
	@echo "    make gpu-htb T=IP                   GPU HTB engagement"
	@echo "    make gpu-sync                       Sync artifacts from GPU"
	@echo "    make gpu-status                     GPU instance status"
	@echo "    make gpu-shell                      SSH into GPU instance"
	@echo "    make gpu-model                      Download / manage models"
	@echo ""
	@echo "  GPU TRAINING (H200)"
	@echo "    make gpu-distill                    H200 distillation (3h, gpt-oss-120)"
	@echo "    make gpu-distill-finetune           H200 fine-tune (lower LR, 6h)"
	@echo "    make gpu-grpo                       GRPO group training (3h)"
	@echo "    make gpu-grpo-large                 GRPO large groups (group_size=8, 6h)"
	@echo "    make gpu-session                    Full session: test→2h distill→2h more/GRPO"
	@echo ""
	@echo "  EVALUATION"
	@echo "    make eval-baseline                  Eval baseline (before training)"
	@echo "    make eval-after                     Eval latest checkpoint"
	@echo "    make eval-compare                   Compare baseline vs latest"
	@echo ""
	@echo "  TEST"
	@echo "    make test                           Full pytest suite"
	@echo "    make test-cap                       Cap regression harness only"
	@echo "    make test-fast                      Fast subset (skip integration)"
	@echo ""
	@echo "  UTIL"
	@echo "    make venv                           Create / upgrade virtual environment"
	@echo "    make status                         System diagnostics"
	@echo "    make last                           View last training run"
	@echo "    make traces                         List recent trace files"
	@echo "    make clean                          Remove temp files + caches"
	@echo ""
	@echo "  DATA PIPELINE"
	@echo "    make unify-all                      Convert all data to unified format"
	@echo "    make unify-validate                 Validate unified data"
	@echo "    make unify-stats                    Show unified data statistics"
	@echo "    make sync-local                     Start local↔GPU sync loop"
	@echo ""

# ── Virtual Environment ──────────────────────────────────────────────
venv:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install torch --index-url https://download.pytorch.org/whl/cpu 2>/dev/null || $(PIP) install torch
	$(PIP) install rich pytest
	@echo "✓ Virtual environment ready. Activate with: source .venv/bin/activate"

# ── Test Suite ───────────────────────────────────────────────────────
test:
	$(PYTEST) -q

# Cap regression harness only
test-cap:
	$(PYTEST) tests/test_cap_regression.py -v --tb=short

# Fast tests (skip slow integration tests)
test-fast:
	$(PYTEST) -q -x --ignore=tests/test_smart_integration.py --ignore=tests/test_training_smoke.py

# ── Training: MS3 (default) ──────────────────────────────────────────
# Default: MS3 LIVE, 100 episodes
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

# ── Training: MS2 ────────────────────────────────────────────────────
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

# ── CTF Engagement (live) ────────────────────────────────────────────
# Usage:
#   make ctf target=10.129.1.54                   # default: 500 steps, no seed
#   make ctf target=10.129.1.54 steps=600         # custom step limit
#   make ctf target=10.129.1.54 seed=42           # reproducible
#   make ctf target=10.129.1.54 steps=800 seed=7  # both
#
# Output is logged to runs/ctf_<timestamp>/live_output.log
target ?=
steps  ?= 500
seed   ?=
ctf:
ifndef target
	$(error target is required. Usage: make ctf target=10.129.x.x)
endif
	@RUN_DIR="runs/ctf_$$(date +%Y%m%d_%H%M%S)_$(target)" && \
	mkdir -p "$$RUN_DIR" && \
	echo "╔══════════════════════════════════════════════════════════════╗" && \
	echo "║  ARIASKA CTF — Live Engagement                              ║" && \
	echo "║  Target : $(target)                                         ║" && \
	echo "║  Steps  : $(steps)                                          ║" && \
	echo "║  Seed   : $(or $(seed),random)                              ║" && \
	echo "║  Log    : $$RUN_DIR/live_output.log                         ║" && \
	echo "╚══════════════════════════════════════════════════════════════╝" && \
	$(PYTHON) ariaska_cli.py smart-train \
		--target $(target) \
		--steps $(steps) \
		--ctf \
		$(if $(seed),--seed $(seed)) \
		2>&1 | tee "$$RUN_DIR/live_output.log"

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

# ── GPU / LLM Acceleration ───────────────────────────────────────────

# Interactive command center (Rich TUI menu)
ariaska:
	$(PYTHON) scripts/ariaska_menu.py

# GPU-accelerated training (local LLM for nano/mini, OpenAI for codex/full)
gpu:
	FF_LOCAL_LLM=1 $(PYTHON) ariaska_cli.py smart-train --target $(or $(T),10.10.10.10) --steps $(or $(S),200) --verbosity verbose

# Setup GPU instance from scratch
gpu-setup:
	bash scripts/gpu_setup.sh

# Quick GPU training (50 steps, fast validation)
gpu-quick:
	FF_LOCAL_LLM=1 $(PYTHON) ariaska_cli.py smart-train --target $(or $(T),10.10.10.10) --steps 50 --seed 42 --verbosity verbose

# GPU CTF engagement
gpu-ctf:
	FF_LOCAL_LLM=1 $(PYTHON) ariaska_cli.py smart-train --target $(T) --steps $(or $(S),150) --ctf --verbosity verbose

# GPU HTB engagement
gpu-htb:
	FF_LOCAL_LLM=1 $(PYTHON) ariaska_cli.py smart-train --target $(T) --steps $(or $(S),300) --ctf --verbosity verbose

# Sync artifacts from GPU instance
gpu-sync:
	bash scripts/gpu_sync.sh pull

# GPU instance status
gpu-status:
	bash scripts/gpu_sync.sh status

# SSH into GPU instance
gpu-shell:
	bash scripts/gpu_sync.sh shell

# Download/manage models
gpu-model:
	$(PYTHON) -c "from scripts.ariaska_menu import download_model; download_model()"

# Restart local LLM server
gpu-restart-llm:
	bash scripts/gpu_sync.sh restart-llm

# ── GPU H200 Distillation Training ───────────────────────────────────

# Standard distillation: 3h, gpt-oss-120b mentor, default reward weights
gpu-distill:
	PYTHONPATH="$$(pwd)" $(PYTHON) -m scripts.h200_run_distill_3h \
		--seed 42 --max-hours 3 --checkpoint-every 10m \
		--reward-weights "format=2.0,code=1.5,math=1.5,reasoning=0.5"

# Fine-tuning: lower LR (5e-6), 6h, resumes best checkpoint
gpu-distill-finetune:
	PYTHONPATH="$$(pwd)" $(PYTHON) -m scripts.h200_run_distill_3h \
		--seed 42 --max-hours 6 --learning-rate 5e-6 --checkpoint-every 10m \
		--reward-weights "format=2.0,code=1.5,math=1.5,reasoning=0.5"

# GRPO group training: 3h, group_size=4
gpu-grpo:
	PYTHONPATH="$$(pwd)" $(PYTHON) -m scripts.train_grpo \
		--seed 42 --max-hours 3 --group-size 4 --learning-rate 1e-4

# GRPO large groups: 6h, group_size=8 (more stable advantages)
gpu-grpo-large:
	PYTHONPATH="$$(pwd)" $(PYTHON) -m scripts.train_grpo \
		--seed 42 --max-hours 6 --group-size 8 --learning-rate 5e-5 \
		--reward-weights "format=2.5,code=1.5,math=1.0,reasoning=0.3"

# Full session: test → 2h distill → auto-decide → 2h more (runs on GPU box)
gpu-session:
	bash scripts/gpu_run_session.sh

# ── Evaluation / Inference ────────────────────────────────────────────

# Evaluate baseline (random-init or current base model)
eval-baseline:
	PYTHONPATH="$$(pwd)" ARIASKA_DRY_RUN=1 $(PYTHON) -m scripts.inference \
		--model-path base --episodes 50 --output results/eval/baseline.json

# Evaluate latest GPU checkpoint
eval-after:
	PYTHONPATH="$$(pwd)" ARIASKA_DRY_RUN=1 $(PYTHON) -m scripts.inference \
		--episodes 50 --output results/eval/after_training.json

# Compare baseline vs latest
eval-compare:
	PYTHONPATH="$$(pwd)" $(PYTHON) -m scripts.inference \
		--compare results/eval/baseline.json results/eval/after_training.json

# ── Utilities ────────────────────────────────────────────────────────

# System diagnostics
status:
	$(PYTHON) ariaska_cli.py status

# ── Cleanup ──────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	rm -rf .pytest_cache 2>/dev/null || true
	@echo "✓ Cleaned temporary files"

# ── Lint (deprecated imports check) ──────────────────────────────────
lint-deprecated:
	@echo "Checking for un-gated MS2/MS3 exploit graph imports..."
	@! grep -rn "from core.knowledge.ms2_exploit_graph import" core/ --include="*.py" \
		| grep -v "feature_flags" | grep -v "__pycache__" \
		| grep -v "# deprecated-ok" | grep -v "test" && echo "✓ No un-gated MS2 imports" || true
	@! grep -rn "from core.knowledge.ms3_exploit_graph import" core/ --include="*.py" \
		| grep -v "feature_flags" | grep -v "__pycache__" \
		| grep -v "# deprecated-ok" | grep -v "test" && echo "✓ No un-gated MS3 imports" || true
	@echo "✓ Deprecation lint passed"

# ── Distillation Prep ─────────────────────────────────────────────────
distill-prep-generate:
	@echo "▶ Generating synthetic traces + teacher trajectories..."
	PYTHONPATH="$$(pwd)" $(PYTHON) -m scripts.distill_prep.generate_synthetic_traces --runs 500 --seed 42
	PYTHONPATH="$$(pwd)" $(PYTHON) -m scripts.distill_prep.generate_teacher_trajectories --trajectories 300 --seed 42
	PYTHONPATH="$$(pwd)" $(PYTHON) -m scripts.distill_prep.summarize_artifacts --manifest --seed 42
	@echo "✓ Distill prep generation complete"

distill-prep-validate:
	@echo "▶ Validating distill prep artifacts..."
	PYTHONPATH="$$(pwd)" $(PYTHON) -m scripts.distill_prep.validate_artifacts
	@echo "✓ Validation complete"

distill-prep-summary:
	@echo "▶ Summarizing distill prep artifacts..."
	PYTHONPATH="$$(pwd)" $(PYTHON) -m scripts.distill_prep.summarize_artifacts
	@echo "✓ Summary complete"

# ── Unified Data Pipeline ────────────────────────────────────────────
unify-all:
	@echo "▶ Converting ALL training data to unified format..."
	PYTHONPATH="$$(pwd)" $(PYTHON) -m scripts.unify_training_data convert-all
	@echo "✓ Unified data conversion complete"

unify-validate:
	@echo "▶ Validating unified data..."
	PYTHONPATH="$$(pwd)" $(PYTHON) -m scripts.unify_training_data validate
	@echo "✓ Validation complete"

unify-stats:
	@echo "▶ Unified data statistics..."
	PYTHONPATH="$$(pwd)" $(PYTHON) -m scripts.unify_training_data stats

sync-local:
	@echo "▶ Starting local sync loop (foreground, Ctrl-C to stop)..."
	bash scripts/local_sync_loop.sh

# ── GPU Live Monitoring ──────────────────────────────────────────────

# Watch live GPU training output (streams tmux pane, refreshes every 2s)
watch-gpu:
	@echo "\033[1;36m▶ Watching GPU training (Ctrl-C to stop)\033[0m"
	@while true; do \
		clear; \
		echo "\033[1;36m═══ ARIASKA GPU TRAINING MONITOR ═══  $$(date '+%H:%M:%S')\033[0m"; \
		echo ""; \
		$(GPU_SSH) '\
			echo "\033[1;33m── Process ──\033[0m"; \
			ps aux | grep h200_run | grep -v grep | awk "{printf \"PID: %s | CPU: %s%% | MEM: %s%% | Elapsed: %s\\n\", \$$2, \$$3, \$$4, \$$10}" || echo "NOT RUNNING"; \
			echo ""; \
			echo "\033[1;33m── GPU ──\033[0m"; \
			nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null | \
				awk -F", " "{printf \"GPU: %s | VRAM: %s / %s | Temp: %s\\n\", \$$1, \$$2, \$$3, \$$4}"; \
			echo ""; \
			echo "\033[1;33m── Training Output (last 20 lines) ──\033[0m"; \
			tmux capture-pane -t train -p 2>/dev/null | grep -v "^$$" | tail -20; \
			echo ""; \
			echo "\033[1;33m── Sync Status ──\033[0m"; \
			tmux capture-pane -t sync -p 2>/dev/null | grep -E "SYNC|sleep|push" | tail -3 \
		' 2>/dev/null || echo "\033[31mSSH connection failed\033[0m"; \
		sleep 2; \
	done

# Full Rich dashboard (one-shot or looping)
gpu-dashboard:
	@PYTHONPATH="$$(pwd)" $(PYTHON) scripts/live_dashboard.py --analysis

# Loop dashboard every 5 min
gpu-dashboard-loop:
	@PYTHONPATH="$$(pwd)" $(PYTHON) scripts/live_dashboard.py --loop 300 --analysis

# Stream raw GPU training logs
gpu-logs:
	@echo "\033[1;36m▶ Streaming GPU training logs (Ctrl-C to stop)\033[0m"
	@$(GPU_SSH) 'tmux capture-pane -t train -p 2>/dev/null | tail -50'

# Quick training status check (no loop)
gpu-train-status:
	@$(GPU_SSH) '\
		echo "=== Process ==="; \
		ps aux | grep h200_run | grep -v grep | awk "{print \"PID:\", \$$2, \"CPU:\", \$$3\"%\", \"MEM:\", \$$4\"%\", \"Time:\", \$$10}" || echo "NOT RUNNING"; \
		echo "=== Latest Trace ==="; \
		ls -lt /root/Ariaska_RL/traces/h200_distill/*.jsonl 2>/dev/null | head -1; \
		LATEST=$$(ls -t /root/Ariaska_RL/traces/h200_distill/*.jsonl 2>/dev/null | head -1); \
		if [ -n "$$LATEST" ]; then \
			echo "Lines: $$(wc -l < $$LATEST)"; \
			echo "Episodes: $$(grep -c episode_end $$LATEST 2>/dev/null)"; \
		fi; \
		echo "=== Budget ==="; \
		tmux capture-pane -t train -p 2>/dev/null | grep -oP "\$$[0-9.]+" | tail -1; \
		echo "=== Sync ==="; \
		tmux capture-pane -t sync -p 2>/dev/null | grep -E "SYNC|push" | tail -2 \
	' 2>/dev/null

# ── Traces ───────────────────────────────────────────────────────────

# Alias: gpu-watch → watch-gpu
gpu-watch: watch-gpu

traces:
	@echo "Recent training runs:"
	@ls -lt traces/ | head -10

# ── Claude Code Integration ──────────────────────────────────────────

cc-start:  ## Start Claude Code session (usage: make cc-start task="describe task")
	@./scripts/cc_sync.sh start "$(task)"

cc-audit:  ## Ask Claude a question (usage: make cc-audit q="your question")
	@./scripts/cc_sync.sh audit "$(q)"

cc-plan:   ## Submit plan for review (usage: make cc-plan p="your plan")
	@./scripts/cc_sync.sh plan "$(p)"

cc-done:   ## Mark done + post-audit (usage: make cc-done s="summary")
	@./scripts/cc_sync.sh done "$(s)"

cc-status: ## Show session status
	@./scripts/cc_sync.sh status

cc-review: ## Full session review by Claude
	@./scripts/cc_sync.sh review

cc-archive: ## Archive current session
	@./scripts/cc_sync.sh archive
