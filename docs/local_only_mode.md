# Local-Only Mode — Offline LLM Architecture

> Phase 55: Full local-only operation without OpenAI API key.

## Overview

Ariaska can now run **live engagements against real targets** (e.g., Metasploitable 2)
without any cloud API key. The system detects when only a local LLM backend (Ollama)
is available and automatically skips expensive multi-stage LLM pipelines, falling
through to pure RL (PPO + Registry + Playbook) decisions.

## How It Works

### Detection

`GPTManager.is_local_only()` returns `True` when:
- No `OPENAI_API_KEY` is set (or empty)
- A local LLM provider (Ollama) is detected and available

This is distinct from `is_offline()`, which returns `True` only when **no** LLM
provider is available at all.

### Decision Pipeline in Local-Only Mode

The SmartCoach 4-stage decision pipeline normally calls multiple LLM-dependent
sub-systems. In local-only mode, these are all **skipped** to avoid blocking on
slow CPU-only inference:

| Component | Normal Mode | Local-Only Mode |
|-----------|-------------|-----------------|
| MicroChain (3-stage nano→codex→nano) | 3 LLM calls | **Skipped** → returns None |
| PhaseGuidedLLM | 1 LLM call | **Skipped** → returns None |
| Codex Meta Check | 1 LLM call | **Skipped** → returns None |
| Codex Strategic Check | 1 LLM call | **Skipped** → returns None |
| Mentor/Reasoning calls | Gated by `gpt_available` | **Disabled** (`gpt_available=False`) |
| **PPO RL policy** | Active | **Active** (primary decision source) |
| **Command Registry** | Active | **Active** (fallback) |
| **Playbook curriculum** | Active | **Active** (annealing) |

### Timeout Caps

Even for non-skipped Ollama calls (e.g., postmortem analysis), timeouts are capped:
- `make_gpt_request()` inner timeout: capped to **60s** when `_use_local`
- `ThreadPoolExecutor` outer timeout: capped to **60s** when `_use_local`
- Ollama provider default: **60s** (unchanged)

### Token Budget

In local-only mode, token usage is **$0.00** — all local model costs are zeroed
in `GPTManager.COST_PER_1K_TOKENS`.

## Files Modified

| File | Change |
|------|--------|
| `core/gpt_manager.py` | Added `is_local_only()` method; capped local timeouts to 60s |
| `core/llm/micro_chain.py` | Early return `None` when `is_local_only()` |
| `core/llm/phase_guided_llm.py` | Early return `None` when `is_local_only()` |
| `core/training/smart_coach.py` | `gpt_available` check includes `is_local_only()`; codex meta/strategic skips |

## Running

```bash
# No API key needed — just have Ollama running on localhost:11434
OPENAI_API_KEY= python ariaska_cli.py smart-train --target 172.28.0.10 --steps 50
```

The system will log:
```
[PROFILE] Resolved profile: OFFLINE (api_key=no, pytest=no)
Auto-detected CPU: activated Ollama local LLM fallback.
Local LLM enabled | all tiers → Ollama
```

## MS2 Validation Run (2026-03-14)

- **Target**: Metasploitable 2 at 172.28.0.10 (Docker)
- **Steps**: 5
- **Result**: RECON → ENUMERATION phase progression
- **Discoveries**: ports 21, 23, 80, 514, 5432; PostgreSQL service identified
- **Rewards**: ScoutAgent +437.9, RedAgent +437.9, OrionAgent +253.8
- **Decision source**: PPO (pure RL)
- **Token cost**: $0.00
- **LLM calls**: 0 (all skipped in local-only mode)

## Root Cause: Pre-Fix Hang

Before these changes, step 1 would hang indefinitely because:
1. `is_offline()` returned `False` when Ollama was available
2. All LLM calls routed to CPU-only Ollama with 600s default timeout
3. MicroChain alone made 3 sequential LLM calls per agent per step
4. With 3 agents, that's 9+ Ollama calls × 600s = potential 90+ minute hang per step

The fix: skip multi-stage LLM pipelines entirely when `is_local_only()`, let pure
RL make decisions, and cap any remaining Ollama calls to 60s.
