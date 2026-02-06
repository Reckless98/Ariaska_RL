# GPTManager Instantiation Audit

**Phase 1 Tracking Document**  
Generated: 2026-01-13

## Summary

This document tracks all locations where `GPTManager()` is directly instantiated in the codebase.
Phase 0 added injection support to Red and Blue agents. Full refactoring is deferred to Phase 2.

## Current State

- **Total Instantiation Sites**: ~45+
- **Phase 0 Fixed (injection-capable)**: RedAgent, BlueAgent (with fallback)
- **Remaining to refactor**: ~40+ sites

## Instantiation Sites by Category

### 1. Core Agents (CRITICAL PATH)

| File | Line | Status |
|------|------|--------|
| `core/agents/red_agent.py` | 218 | ✅ Uses injected or fallback |
| `core/agents/blue_agent.py` | 128 | ✅ Uses injected or fallback |
| `core/agents/scout_agent.py` | 65 | ⚠️ Creates own instance |
| `core/agents/shadow_agent.py` | 64 | ⚠️ Creates own instance |
| `core/agents/orion_agent.py` | 180 | ⚠️ Creates own instance |
| `core/agents/enhanced_agent_base.py` | 70 | ⚠️ Creates own instance |

### 2. Core Utilities

| File | Line | Status |
|------|------|--------|
| `core/gpt_manager.py` | 1089 | Singleton fallback |
| `core/llm_utils.py` | 26 | Module-level lazy init |
| `core/vector_search.py` | 38 | Creates own instance |
| `core/strategy_optimizer.py` | 120 | Supports injection |
| `core/gpt_distiller.py` | 48 | Creates own instance |

### 3. LLM/Model Infrastructure

| File | Line | Status |
|------|------|--------|
| `core/utils/llm_router.py` | 71 | Creates own instance |
| `core/utils/llm_orchestrator.py` | 61 | Creates own instance |
| `core/utils/context_encoder.py` | 256 | Creates own instance |
| `core/utils/gpt_cache_handler.py` | 11 | Creates own instance |
| `core/models/gpt_context_encoder.py` | 24 | Creates own instance |
| `core/models/value_net.py` | 71 | Creates own instance |
| `core/logic/output_interpreter.py` | 197 | Creates own instance |
| `core/logic/rule_engine.py` | 37 | Creates own instance |

### 4. Training Infrastructure

| File | Line | Status |
|------|------|--------|
| `core/training/enhanced_unified_trainer.py` | 475 | Creates own instance |
| `core/training/apprentice_trainer.py` | 199 | Supports injection |

### 5. Memory Systems

| File | Line | Status |
|------|------|--------|
| `core/memory/enhanced_memory_sync.py` | 53 | Creates own instance |
| `core/memory/chroma_memory_store.py` | 556 | Creates own instance |

### 6. Postmortem/Analysis

| File | Line | Status |
|------|------|--------|
| `core/postmortem/orion_postmortem.py` | 221 | Creates own instance |

### 7. CLI Entry Points

| File | Line | Status |
|------|------|--------|
| `ariaska_cli.py` | 110, 244, 488 | Creates instances in commands |
| `main.py` | 46 | Creates module-level instance |

### 8. Tests (Not included in refactor scope)

| File | Lines | Notes |
|------|-------|-------|
| `tests/test_smart_integration.py` | 33, 75, 149, 207, 275 | Test fixtures |
| `tests/test_ariaska_systems.py` | 333, 346, 731 | Test fixtures |
| `tests/test_testing_utilities.py` | various | Uses FakeGPTManager |
| `test_role_separation.py` | 21 | Test file |

## Phase 2 Refactoring Plan

### Priority 1: Core Agents (High Impact)
- Add `gpt_manager=None` parameter to ScoutAgent, ShadowAgent, OrionAgent
- Pass AgentManager's shared instance during initialization

### Priority 2: Training Pipeline
- Inject shared GPTManager into trainers from orchestrator

### Priority 3: Utilities
- Convert to lazy singleton pattern or dependency injection
- Consider a central `get_gpt_manager()` factory

### Not Recommended for Refactor
- Test files (acceptable to create test instances)
- CLI entry points (these are top-level, OK to instantiate)

## Notes

The GPTManager class already has caching (`_gpt_manager_instance` in `get_gpt_manager()`),
so multiple instantiations share cached responses. The main concern is:
1. Multiple API client initializations
2. Memory overhead from duplicate instances
3. Potential for inconsistent configuration

Current impact is manageable for Phase 1. Full consolidation deferred to Phase 2.
