# Phase 10.1 Hardening — Maintenance & Dead Code Audit

**Date:** 2025-02-16  
**Scope:** Full codebase dedup/dead-code audit for Pre-HTB Capability Hardening  
**Baseline:** 423 tests passing, 130+ modules, 105K+ lines of Python  

---

## Executive Summary

Static analysis identified **33 files flagged for removal/deprecation** totaling
**~8,500+ dead lines**, plus **10 duplicate command registry entries**.  The active
execution path (SmartOrchestrator → SmartCoach → PPOAgent → 5 Agents) is clean.
Dead code is concentrated in legacy training modules, empty placeholder files,
and abandoned ChromaDB/vector memory experiments.

---

## Dead Code Registry

### Tier 1: Empty Files (0 bytes, 0 risk to remove)

| # | File |
|---|------|
| 1 | `core/training/enhanced_trainer.py` |
| 2 | `core/training/unified_trainer.py` |
| 3 | `core/training/sleek_trainer.py` |
| 4 | `core/llm_utils_simple.py` |
| 5 | `core/utils/stats_monitor_clean.py` |
| 6 | `core/utils/stats_monitor_fixed.py` |
| 7 | `core/utils/stats_monitor_new.py` |
| 8 | `core/utils/llm_orchestrator_simple.py` |
| 9 | `core/utils/llm_router_simple.py` |
| 10 | `core/utils/test_llm_orchestration.py` |

### Tier 2: Backup Files

| # | File | Lines |
|---|------|-------|
| 1 | `core/observability/live_dashboard.py.bak` | 657 |
| 2 | `core/observability/live_dashboard_v2_backup.py` | 657 |

### Tier 3: Orphaned Modules (>0 lines, zero importers)

| # | File | Lines | Purpose |
|---|------|-------|---------|
| 1 | `core/training/enhanced_unified_trainer.py` | 1017 | Abandoned unified trainer |
| 2 | `core/neural_networks_advanced.py` | 504 | Duplicate of `core/models/advanced_networks.py` |
| 3 | `core/coordination_system.py` | 560 | Parallel coordination, never integrated |
| 4 | `core/strategy_optimizer.py` | 267 | Performance analyzer, never used |
| 5 | `core/knowledge_embedder.py` | 86 | FAISS embedder, never used |
| 6 | `core/gpt_distiller.py` | 358 | Standalone distillation tool |
| 7 | `core/cli_interface.py` | 126 | prompt_toolkit CLI, replaced by ariaska_cli.py |
| 8 | `core/visualization.py` | 212 | Banner/GPU display, never imported |
| 9 | `core/ui_helpers_enhanced.py` | 418 | Enhanced UI helpers, nothing imports |
| 10 | `core/memory/chroma_memory_store.py` | 587 | ChromaDB store v1, unused |
| 11 | `core/semantic_memory/chromadb_store.py` | 399 | ChromaDB store v2, unused |
| 12 | `core/utils/context_encoder.py` | 483 | Context summarizer, unused |
| 13 | `core/environment/context_detector.py` | 301 | Older env context detector duplicate |
| 14 | `core/multi_agent_trainer.py` | 725 | Legacy multi-agent trainer |
| 15 | `core/trainer.py` | 920 | Legacy training loop |
| 16 | `core/llm_utils.py` | 204 | Wrapper for gpt_manager, unused |
| 17 | `core/utils/stats_monitor_fallback.py` | 29 | Unused stub |

### Tier 4: Duplicate Command Registrations

10 commands registered twice in `command_registry.py` (second overwrites first):

- `hydra_ssh`, `hydra_ftp`, `hydra_http_form`
- `linpeas`, `find_suid`, `kernel_exploit_check`
- `nfs_mount`, `chisel_server`, `chisel_client`, `ssh_tunnel_local`

**Action:** Audit each pair for template differences. If identical, remove the
duplicate. If different phases/preconditions, rename to disambiguate.

---

## Recommended Cleanup Order

1. Delete Tier 1 (empty files) → `make test`
2. Delete Tier 2 (backups) → `make test`
3. Delete Tier 3 (orphaned modules) in 3 batches → `make test` after each
4. Deduplicate Tier 4 (registry entries) → `make test`
5. Assess legacy chains: `orchestrator.py` → `apprentice_coach.py` → `ariaska_trainer.py`

**Safety rule:** `make test` must remain at 423+ passing after each batch.

---

## Phase 10.1 New Modules Added

| Module | Purpose | Lines | Tests |
|--------|---------|-------|-------|
| `core/commands/privilege.py` | Privilege-aware command gating | ~190 | 23 |
| `core/tools/tool_registry.py` | Tool availability + install coordination | ~450 | 26 |
| `core/tools/wordlist_engine.py` | Password mutation engine | ~400 | 17 |
| `core/tools/knock_sequence.py` | Port knocking inference | ~340 | 19 |
| `core/tools/web_proxy_layer.py` | HAR proxy capture integration | ~500 | 29 |
| `core/tools/payload_encoder.py` | Payload encoding engine | ~330 | 31 |
| `scripts/tools/bootstrap_tools.py` | CLI tool bootstrap installer | ~100 | — |

**Total new code:** ~2,310 lines, 145 tests

---

## Feature Flags Added (Phase 10.1)

| Flag | Default | Purpose |
|------|---------|---------|
| `privilege_gating` | `True` | Filter commands by privilege level |
| `allow_sudo` | `False` | Allow sudo-requiring commands |
| `allow_live_install` | `False` | Allow runtime tool installation |
| `sudo_mode` | `"prompt"` | How sudo is handled |
| `wordlist_mutation` | `False` | Enable wordlist mutation engine |
| `port_knocking` | `False` | Enable port knock inference |
| `proxy_capture` | `False` | Enable HAR proxy ingestion |
| `payload_encoding` | `False` | Enable payload encoding |
