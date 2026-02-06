# Phase 1 Hardening Summary

**Date**: 2026-01-13  
**Git Commit**: 5f948de  
**Status**: ✅ Complete

## Overview

Phase 1 hardening for Ariaska_RL following Phase 0 stabilization. This phase focused on:
1. Fixing the AgentManager verbosity initialization bug
2. Adding pytest tests to lock Phase 0 invariants
3. Running deterministic training verification
4. Creating GPTManager duplication audit

---

## Task A: AgentManager Verbosity Fix

### Problem
Agent initialization was inside a verbosity conditional - with `verbosity="quiet"` or `"silent"`, agents were never initialized.

### Solution
Restructured `__init__` in `core/multiagent/agent_manager.py`:
- All agent/environment initialization ALWAYS runs regardless of verbosity
- Only print/log output (banners, progress bars, dashboard) is gated by verbosity
- Added `_verbose` flag for cleaner conditional checks

### Verification
```python
from core.multiagent.agent_manager import AgentManager
am = AgentManager(verbosity="quiet")
assert am.red_agent is not None  # ✅ PASS
assert am.blue_agent is not None  # ✅ PASS
assert am.red_agent.env is am.blue_agent.env  # ✅ PASS (env sharing works)
```

---

## Task B: Pytest Test Harness

### Created: `tests/test_phase0_invariants.py`

**9 Tests - All Passing**

| Test | Description | Status |
|------|-------------|--------|
| `test_env_is_shared_between_red_and_blue` | Verifies Red and Blue share same env | ✅ |
| `test_blueagent_has_react_to_action_method` | BlueAgent has react_to_action | ✅ |
| `test_react_to_action_returns_correct_schema` | Returns dict with required keys | ✅ |
| `test_blueagent_simulate_step_does_not_call_env_step` | simulate_step doesn't double-step | ✅ |
| `test_alert_scale_is_0_to_100` | Alertness uses 0-100 scale | ✅ |
| `test_stealth_report_uses_100_scale` | Stealth report consistent with 0-100 | ✅ |
| `test_deterministic_mode_reproducibility` | Seeds produce identical sequences | ✅ |
| `test_quiet_mode_initializes_agents` | Quiet mode still creates agents | ✅ |
| `test_silent_mode_initializes_agents` | Silent mode still creates agents | ✅ |

### Run Command
```bash
.venv/bin/python -m pytest tests/test_phase0_invariants.py -v
```

### Output
```
============================== 9 passed in 11.65s ==============================
```

---

## Task C: Deterministic Training Run

### Configuration
- **Mode**: Deterministic (`ARIASKA_DETERMINISTIC=true`, `ARIASKA_SEED=42`)
- **Episodes**: 5 (target)
- **Environment**: Simulated (`--sim`)

### Results
| Metric | Value |
|--------|-------|
| Steps Completed | 81 |
| Episodes | 0/5 (partial) |
| Cumulative Reward | +1273.8 |
| Runtime | ~5 minutes |
| Log Lines | 3664 |
| Phase | enumeration |

### Key Observations
1. **Environment Sharing**: Confirmed working - all agents operate on same env
2. **Single Step Loop**: No double-stepping detected
3. **Net-Positive Reward**: Cumulative reward stayed positive throughout
4. **Anti-Repeat System**: Working correctly, blocking redundant commands
5. **Role Enforcement**: Violations detected and blocked (e.g., OrionAgent attempting RedAgent commands)

### Log Location
`artifacts/phase1_smart_train.log` (3664 lines)

---

## Task D: GPTManager Audit

### Document Created
`artifacts/gptmanager_instantiations.md`

### Summary
| Category | Count | Status |
|----------|-------|--------|
| Core Agents | 6 | 2 fixed (Red/Blue), 4 remaining |
| Core Utilities | 5 | Not yet refactored |
| LLM Infrastructure | 8 | Not yet refactored |
| Training | 2 | 1 supports injection |
| Memory Systems | 2 | Not yet refactored |
| Tests | ~15 | Acceptable (test fixtures) |
| **Total** | ~45 | Phase 2 scope |

### Phase 0 Status
- ✅ RedAgent: Uses injected GPTManager with fallback
- ✅ BlueAgent: Uses injected GPTManager with fallback

---

## Bug Fixes (Discovered During Testing)

### 1. `_trigger_blue_team_response()` - AttributeError
- **File**: `core/environment/cyber_environment.py`
- **Issue**: Referenced `self.detection_rate` (doesn't exist) and `self.score` (doesn't exist)
- **Fix**: Changed to `self.detection_risk` (correct attribute)

---

## Files Modified

1. `core/multiagent/agent_manager.py` - Verbosity fix
2. `core/environment/cyber_environment.py` - Bug fixes in `_trigger_blue_team_response()`
3. `tests/test_phase0_invariants.py` - NEW: Phase 0 test suite
4. `artifacts/phase1_smart_train.log` - NEW: Training log
5. `artifacts/phase1_summary.md` - NEW: This document
6. `artifacts/gptmanager_instantiations.md` - NEW: GPTManager audit

---

## Reproduction Steps

```bash
# 1. Run Phase 0 invariant tests
.venv/bin/python -m pytest tests/test_phase0_invariants.py -v

# 2. Verify quiet mode initialization
.venv/bin/python -c "
from core.multiagent.agent_manager import AgentManager
am = AgentManager(verbosity='quiet')
print(f'Red: {am.red_agent is not None}')
print(f'Blue: {am.blue_agent is not None}')
print(f'Shared env: {am.red_agent.env is am.blue_agent.env}')
"

# 3. Run deterministic training
ARIASKA_DETERMINISTIC=true ARIASKA_SEED=42 \
  .venv/bin/python ariaska_cli.py smart-train 5 --sim
```

---

## Next Steps (Phase 2)

1. **GPTManager Consolidation**: Refactor remaining agents (Scout, Shadow, Orion) to accept injected GPTManager
2. **Trainer Injection**: Pass shared GPTManager through training pipeline
3. **Utility Singleton**: Convert utilities to use lazy singleton pattern
4. **Performance Profiling**: Measure impact of GPTManager consolidation
