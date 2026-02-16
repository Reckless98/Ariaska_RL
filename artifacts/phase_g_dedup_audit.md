# Phase G: Dedup + Dead Code Audit Report

**Date:** 2026-02-16  
**Scope:** Full codebase scan of `/home/zer0/Projects/Ariaska_RL`  
**Method:** Static analysis — import tracing, file headers, line counts, grep across all modules  

---

## 1. Duplicate / Near-Duplicate Modules

### 1A. Orchestrators (2 files)

| File | Lines | Imported By | Recommendation |
|------|-------|-------------|----------------|
| `core/orchestration/smart_orchestrator.py` | 5220 | `ariaska_cli.py` (main entry) | **KEEP** — primary brain |
| `core/orchestration/orchestrator.py` | 678 | `core/training/ariaska_trainer.py` only | **DEPRECATE** — legacy v1 orchestrator, superseded by SmartOrchestrator |

**Overlap:** Both coordinate multi-agent training loops. `orchestrator.py` is the older ApprenticeCoach-based loop. `smart_orchestrator.py` is the active production orchestrator. Only `ariaska_trainer.py` still references the old one.

---

### 1B. Trainers (9 files — severe overlap)

| File | Lines | Imported By | Recommendation |
|------|-------|-------------|----------------|
| `core/training/smart_coach.py` | 6587 | SmartOrchestrator (active pipeline) | **KEEP** — primary decision pipeline |
| `core/training/ariaska_trainer.py` | 712 | Tests, CLI, `__init__.py` | **KEEP** — high-level training config, uses old Orchestrator |
| `core/training/apprentice_coach.py` | 552 | `orchestrator.py`, `__init__.py` | **DEPRECATE** — superseded by SmartCoach |
| `core/training/apprentice_trainer.py` | 536 | `ariaska_trainer.py`, tests | **DEPRECATE** — superseded by SmartOrchestrator loop |
| `core/training/enhanced_unified_trainer.py` | 1017 | **NOTHING** | **REMOVE** — zero imports, never used |
| `core/training/enhanced_trainer.py` | 0 | **NOTHING** | **REMOVE** — empty file |
| `core/training/unified_trainer.py` | 0 | **NOTHING** | **REMOVE** — empty file |
| `core/training/sleek_trainer.py` | 0 | **NOTHING** | **REMOVE** — empty file |
| `core/multi_agent_trainer.py` | 725 | **NOTHING** | **REMOVE** — zero imports from anywhere, top-level imports will break (uses old modules) |
| `core/trainer.py` | 920 | **NOTHING** | **REMOVE** — zero imports from anywhere, legacy training loop |

**Summary:** 5 of 9 trainer files are completely unused. Only `smart_coach.py` + `ariaska_trainer.py` are in the active path. The apprentice pair is legacy but still referenced.

---

### 1C. LLM Utility Files (3 files)

| File | Lines | Imported By | Recommendation |
|------|-------|-------------|----------------|
| `core/gpt_manager.py` | 1175 | Everywhere (canonical gateway) | **KEEP** — THE LLM gateway |
| `core/llm_utils.py` | 204 | **NOTHING** (no imports found) | **DEPRECATE** — provides `get_gpt_manager()` wrapper, but nothing imports it |
| `core/llm_utils_simple.py` | 0 | **NOTHING** | **REMOVE** — empty file |
| `core/gpt_distiller.py` | 358 | **NOTHING** | **REMOVE** — zero imports, standalone distillation tool never invoked |

---

### 1D. Neural Network Files (2 overlapping)

| File | Lines | Imported By | Recommendation |
|------|-------|-------------|----------------|
| `core/models/advanced_networks.py` | 524 | `enhanced_agent_base.py`, `red_agent.py`, `ppo_agent.py` | **KEEP** — active, provides ResidualBlock, attention, create_advanced_* |
| `core/neural_networks_advanced.py` | 504 | **NOTHING** | **REMOVE** — zero imports, contains PrioritizedReplayBuffer + Rainbow DQN code duplicated elsewhere |

**Overlap:** Both provide advanced NN architectures. The `core/models/` version is the canonical one used by agents and PPO. The top-level file is dead.

---

### 1E. Replay Buffers (3 files)

| File | Lines | Imported By | Recommendation |
|------|-------|-------------|----------------|
| `core/algorithms/replay_buffer.py` | 559 | `core/trainer.py` (dead) | **REVIEW** — only imported by dead trainer.py |
| `core/utils/replay_buffer.py` | 278 | `red_agent`, `blue_agent`, `scout_agent`, `shadow_agent`, `orion_agent` | **KEEP** — used by all agents |
| `core/neural_networks_advanced.py` | (inline PrioritizedReplayBuffer) | **NOTHING** | Part of dead file above |

**Note:** `core/algorithms/replay_buffer.py` is a more advanced PER with SumTree, but its only importer (`core/trainer.py`) is itself dead.

---

### 1F. Context Encoders (2 files)

| File | Lines | Imported By | Recommendation |
|------|-------|-------------|----------------|
| `core/models/gpt_context_encoder.py` | 88 | `policy_net.py`, `value_net.py` | **KEEP** — used by legacy networks |
| `core/utils/context_encoder.py` | 483 | **NOTHING** | **REMOVE** — zero imports, provides context summarization utilities never used |

---

### 1G. ChromaDB / Vector Memory (3+ files)

| File | Lines | Imported By | Recommendation |
|------|-------|-------------|----------------|
| `core/memory/chroma_memory_store.py` | 587 | **NOTHING** | **REMOVE** — zero imports from any module |
| `core/semantic_memory/chromadb_store.py` | 399 | **NOTHING** | **REMOVE** — zero imports from any module |
| `core/memory/hybrid_memory.py` | 999 | `smart_coach.py`, `cognition_node.py` | **KEEP** — actively used |
| `core/vector_search.py` | 178 | `cli_interface.py`, `chainbuilder.py` | **DEPRECATE** — only used by legacy modules (see below) |

**Overlap:** Three separate ChromaDB/vector memory implementations, but only `hybrid_memory.py` is actually in the active execution path.

---

### 1H. Environment Context Detectors (2 files)

| File | Lines | Imported By | Recommendation |
|------|-------|-------------|----------------|
| `core/environment/environment_context_detector.py` | 654 | `cyber_environment.py` | **KEEP** — used by main environment |
| `core/environment/context_detector.py` | 301 | **NOTHING** | **REMOVE** — zero imports, older version with identical header comment |

---

### 1I. JSONL Loggers (2 files)

| File | Lines | Imported By | Recommendation |
|------|-------|-------------|----------------|
| `core/logging/jsonl_logger.py` | 195 | `smart_orchestrator.py` (StepRecord, EpisodeSummary, RunLogger) | **KEEP** — R66 per-step logging |
| `core/telemetry/jsonl_logger.py` | 174 | `smart_orchestrator.py` (JSONLLogger), tests | **KEEP** — Phase 9.7 event writer |

**Not duplicates** — serve different purposes (step-level logging vs event telemetry). Both are actively used. **KEEP both.**

---

### 1J. Stats Monitors (5 files!)

| File | Lines | Imported By | Recommendation |
|------|-------|-------------|----------------|
| `core/utils/stats_monitor.py` | 631 | 8 importers (all agents, agent_manager, etc.) | **KEEP** — canonical stats monitor |
| `core/utils/stats_monitor_clean.py` | 0 | **NOTHING** | **REMOVE** — empty |
| `core/utils/stats_monitor_fallback.py` | 29 | **NOTHING** | **REMOVE** — unused fallback stub |
| `core/utils/stats_monitor_fixed.py` | 0 | **NOTHING** | **REMOVE** — empty |
| `core/utils/stats_monitor_new.py` | 0 | **NOTHING** | **REMOVE** — empty |

---

### 1K. UI / Visualization Files (5+ overlapping)

| File | Lines | Imported By | Recommendation |
|------|-------|-------------|----------------|
| `core/ui/textual_dashboard.py` | ? | `smart_orchestrator.py`, tests | **KEEP** — active TUI |
| `core/ui/enhanced_agent_dashboard.py` | ? | **NOTHING** | **REMOVE** — zero imports |
| `core/ui_helpers.py` | 483 | `red_agent.py`, `blue_agent.py` | **KEEP** — provides `display_status_bar`, `get_action_description` |
| `core/ui_helpers_enhanced.py` | 418 | **NOTHING** | **REMOVE** — zero imports |
| `core/visualization.py` | 212 | **NOTHING** | **REMOVE** — banner display, GPU status, never imported |
| `core/visualization/training_visualizer.py` | ? | `multi_agent_trainer.py` (dead), `trainer.py` (dead) | **REMOVE** — only imported by dead modules |
| `core/observability/live_dashboard.py` | 995 | `smart_orchestrator.py` | **KEEP** — active live dashboard |
| `core/observability/live_dashboard.py.bak` | 657 | **NOTHING** | **REMOVE** — backup file |
| `core/observability/live_dashboard_v2_backup.py` | 657 | **NOTHING** | **REMOVE** — backup file |

---

### 1L. LLM Router / Orchestrator Variants (4 files in utils)

| File | Lines | Imported By | Recommendation |
|------|-------|-------------|----------------|
| `core/utils/llm_orchestrator.py` | 340 | `shadow_agent`, `scout_agent`, `orion_agent` | **KEEP** — used by 3 agents |
| `core/utils/llm_orchestrator_simple.py` | 0 | **NOTHING** | **REMOVE** — empty |
| `core/utils/llm_router.py` | 374 | `teach.py`, `chainbuilder.py`, `llm_utils.py` | **KEEP** — used (but assess if callers are alive) |
| `core/utils/llm_router_simple.py` | 0 | **NOTHING** | **REMOVE** — empty |
| `core/utils/test_llm_orchestration.py` | 0 | **NOTHING** | **REMOVE** — empty test file |

---

### 1M. Other Dead Top-Level core/ Files

| File | Lines | Imported By | Recommendation |
|------|-------|-------------|----------------|
| `core/coordination_system.py` | 560 | **NOTHING** | **REMOVE** — zero imports, parallel coordination system never used |
| `core/strategy_optimizer.py` | 267 | **NOTHING** | **REMOVE** — zero imports, performance analyzer / strategy optimizer never called |
| `core/knowledge_embedder.py` | 86 | **NOTHING** | **REMOVE** — zero imports, FAISS+SentenceTransformer embedder never used |
| `core/cli_interface.py` | 126 | **NOTHING** | **REMOVE** — zero imports, prompt_toolkit CLI, superseded by ariaska_cli.py |

---

### 1N. Other Dead Files

| File | Lines | Imported By | Recommendation |
|------|-------|-------------|----------------|
| `core/memory/enhanced_memory_router.py` | 248 | `ariaska_trainer.py` only | **DEPRECATE** — only used by partially-legacy trainer |
| `core/memory/enhanced_memory_sync.py` | 510 | `enhanced_agent_base.py` only | **KEEP** — used by agent base class |
| `core/memory/campaign_memory.py` | 490 | `smart_orchestrator.py`, tests | **KEEP** — active |
| `core/memory/unified_cognitive_bus.py` | 794 | `smart_mentor`, `smart_coach`, `smart_orchestrator` | **KEEP** — active |
| `core/analytics/replay_analyzer.py` | ? | **NOTHING** | **REMOVE** — zero imports from outside itself |

---

## 2. Dead Scripts in `scripts/`

| Script | Lines | Purpose | Referenced By Main Codebase? | Recommendation |
|--------|-------|---------|------------------------------|----------------|
| `scripts/build_knowledge_indices.py` | 192 | Rebuild search indices | No (standalone CLI) | **KEEP** — operational tool |
| `scripts/fill_htb_gaps.py` | 508 | Fill HTB readiness gaps | No | **KEEP** — operational tool |
| `scripts/dedup_v2_corpus.py` | 181 | Deduplicate v2 corpus | No | **KEEP** — maintenance tool |
| `scripts/htb_readiness.py` | 266 | Assess HTB readiness | No | **KEEP** — assessment tool |
| `scripts/tool_dependency_audit.py` | 919 | Audit tool deps | No | **KEEP** — audit tool |
| `scripts/watch.py` | 256 | Live telemetry viewer | No | **KEEP** — useful during training |
| `scripts/aria` | 123 | Bash command runner | No | **KEEP** — convenience wrapper |
| `scripts/tools/bootstrap_tools.py` | 103 | Tool installer | No | **KEEP** — setup utility |
| `scripts/audit_candidates_v2.py` | 214 | v2 corpus audit | No | **KEEP** — one-time analysis |
| `scripts/audit_corpus_v1.py` | 171 | v1 corpus audit | No | **DEPRECATE** — v1 corpus is legacy |
| `scripts/corpus_dedup_audit.py` | 362 | Dedup audit | No | **KEEP** — overlaps with dedup_v2 but different focus |
| `scripts/refactor_corpus_to_v2.py` | 1216 | v1→v2 migration | No | **DEPRECATE** — one-time migration already done |
| `scripts/extract_htb_walkthroughs.py` | 396 | HTB extraction | No | **KEEP** — useful for future HTB data |
| `scripts/run_ingestion.py` | 24 | Run knowledge ingestion | No | **KEEP** — operational |
| `scripts/overnight_train.sh` | ? | Overnight training | No | **KEEP** — operational |
| `scripts/setup_ms2.sh` | ? | Setup Metasploitable 2 | No | **KEEP** — operational |

**None of the scripts are imported by the main codebase** (all standalone CLI tools). Most are useful operational/maintenance utilities. Only `audit_corpus_v1.py` and `refactor_corpus_to_v2.py` are truly one-shot legacy.

---

## 3. Unused/Dead Files Summary

### Immediately Removable (0 imports, 0 lines or fully orphaned):

| # | File | Lines | Reason |
|---|------|-------|--------|
| 1 | `core/training/enhanced_trainer.py` | 0 | Empty |
| 2 | `core/training/unified_trainer.py` | 0 | Empty |
| 3 | `core/training/sleek_trainer.py` | 0 | Empty |
| 4 | `core/llm_utils_simple.py` | 0 | Empty |
| 5 | `core/utils/stats_monitor_clean.py` | 0 | Empty |
| 6 | `core/utils/stats_monitor_fixed.py` | 0 | Empty |
| 7 | `core/utils/stats_monitor_new.py` | 0 | Empty |
| 8 | `core/utils/llm_orchestrator_simple.py` | 0 | Empty |
| 9 | `core/utils/llm_router_simple.py` | 0 | Empty |
| 10 | `core/utils/test_llm_orchestration.py` | 0 | Empty |
| 11 | `core/training/enhanced_unified_trainer.py` | 1017 | Zero imports |
| 12 | `core/neural_networks_advanced.py` | 504 | Zero imports |
| 13 | `core/coordination_system.py` | 560 | Zero imports |
| 14 | `core/strategy_optimizer.py` | 267 | Zero imports |
| 15 | `core/knowledge_embedder.py` | 86 | Zero imports |
| 16 | `core/gpt_distiller.py` | 358 | Zero imports |
| 17 | `core/cli_interface.py` | 126 | Zero imports |
| 18 | `core/visualization.py` | 212 | Zero imports |
| 19 | `core/ui_helpers_enhanced.py` | 418 | Zero imports |
| 20 | `core/ui/enhanced_agent_dashboard.py` | ? | Zero imports |
| 21 | `core/memory/chroma_memory_store.py` | 587 | Zero imports |
| 22 | `core/semantic_memory/chromadb_store.py` | 399 | Zero imports |
| 23 | `core/utils/context_encoder.py` | 483 | Zero imports |
| 24 | `core/environment/context_detector.py` | 301 | Zero imports (older duplicate) |
| 25 | `core/multi_agent_trainer.py` | 725 | Zero imports |
| 26 | `core/trainer.py` | 920 | Zero imports |
| 27 | `core/llm_utils.py` | 204 | Zero imports |
| 28 | `core/observability/live_dashboard.py.bak` | 657 | Backup file |
| 29 | `core/observability/live_dashboard_v2_backup.py` | 657 | Backup file |
| 30 | `core/utils/stats_monitor_fallback.py` | 29 | Zero imports |
| 31 | `core/analytics/replay_analyzer.py` | ? | Zero imports |

**Total dead lines removable: ~8,500+ lines**

---

## 4. Duplicate Command Registry Names

10 command names are registered **twice** in `core/commands/command_registry.py`:

| Command Name | First Registration (line) | Second Registration (line) |
|---|---|---|
| `hydra_ssh` | 634 | 1657 |
| `hydra_ftp` | 651 | 1670 |
| `hydra_http_form` | 680 | 1683 |
| `linpeas` | 897 | 1758 |
| `find_suid` | 934 | 1863 |
| `kernel_exploit_check` | 982 | 1887 |
| `nfs_mount` | 579 | 2463 |
| `chisel_server` | 1149 | 3217 |
| `chisel_client` | 1161 | 3231 |
| `ssh_tunnel_local` | 1185 | 3246 |

**Recommendation:** Audit each pair. If templates are identical, remove the second. If they differ (e.g., different phases/preconditions), rename one to disambiguate (e.g., `hydra_ssh_brute` vs `hydra_ssh_spray`). The `register()` function likely silently overwrites the first, meaning the first registration is effectively dead code.

---

## 5. Partially-Dead Chains

These modules are alive but exist only in legacy call chains that are themselves partially dead:

| Module | Used By | Status |
|--------|---------|--------|
| `core/visualization/training_visualizer.py` | `multi_agent_trainer.py` (dead), `trainer.py` (dead) | **Dead chain** — remove with parents |
| `core/algorithms/replay_buffer.py` (559L) | `core/trainer.py` (dead) | **Dead chain** — only importer is dead |
| `core/vector_search.py` (178L) | `cli_interface.py` (dead), `chainbuilder.py` (assess) | **Mostly dead** — cli_interface is dead |
| `core/utils/gpt_cache_handler.py` (25L) | `red_agent.py`, `blue_agent.py` | **KEEP** — still imported by live agents |
| `core/memory/enhanced_memory_router.py` (248L) | `ariaska_trainer.py` | **Assess** — only importer is a semi-legacy trainer |

---

## 6. Summary Statistics

| Category | Count | Dead Lines |
|----------|-------|------------|
| Empty files (0 lines) | 10 | 0 |
| Orphan modules (>0 lines, 0 imports) | 21 | ~7,500 |
| Backup/bak files | 2 | ~1,300 |
| Duplicate command registrations | 10 names × 2 | N/A |
| **Total removable** | **33 files** | **~8,500+ lines** |

---

## 7. Recommended Cleanup Order

1. **Phase G.1** — Delete 10 empty files (risk: zero)
2. **Phase G.2** — Delete 2 backup files in observability/ (risk: zero)  
3. **Phase G.3** — Delete 21 orphaned modules with zero importers (risk: low — run `make test` after each batch)
4. **Phase G.4** — Deduplicate 10 command registry entries (risk: low — check template differences first)
5. **Phase G.5** — Assess legacy chains: `orchestrator.py` + `apprentice_coach.py` + `apprentice_trainer.py` + `ariaska_trainer.py` — decide if the old training path should be preserved or fully removed
6. **Phase G.6** — Clean up `core/utils/` — remove unused variants, verify `llm_router.py` callers are still alive

**Critical safety rule:** Run `make test` (423 tests) after each cleanup batch to confirm no regressions.
