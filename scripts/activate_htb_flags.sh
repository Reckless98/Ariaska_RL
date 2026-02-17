#!/usr/bin/env bash
# ============================================================================
# ARIASKA_RL — Phase 15 NEUROVORTEX: Full Flag Activation for HTB Lab Mode
# ============================================================================
# Usage:
#   source scripts/activate_htb_flags.sh
#   python scripts/verify_flags.py
#
# Enables ALL feature flags for authorized lab (HTB) operation.
# MS2/MS3 knowledge packs kept OFF (not training specific targets).
# ============================================================================

# ── Phase 9.5: Correctness ──────────────────────────────────────────────────
export FF_DDQN_SINGLE_SELECT_PER_STEP=1
export FF_SINGLE_PARSE_CACHE=1
export FF_PPO_REWARD_ATTRIBUTION_FIX=1

# ── Phase 9.6: Architecture Wiring ─────────────────────────────────────────
export FF_TACTICAL_CORTEX_GATE=1
export FF_TACTICAL_PPO_BLOCK=1
export FF_EXECUTIVE_CORTEX=1
export FF_TARGET_PROFILER_PIPELINE=1
export FF_KR_ENRICHMENT=1

# ── Phase 9.7: Telemetry & UI ──────────────────────────────────────────────
export FF_JSONL_TELEMETRY=1
export FF_RICH_HUD=1
export FF_TRANSPARENT_MODE=1

# ── Phase 9.7+: KG & LLM ───────────────────────────────────────────────────
export FF_KG_WRITE=1
export FF_LLM_STRATEGIC_PLANNER=1
export FF_LLM_TACTICAL_ADVISOR=1
export FF_LLM_JUDGE_RANKER=1
export FF_LLM_POSTMORTEM_SKILLS=1
export FF_DAGGER_CORRECTIONS=1
export FF_INGEST_PIPELINE=1

# ── Phase 10.1: Pre-HTB Capability Hardening ───────────────────────────────
export FF_PRIVILEGE_GATING=1
export FF_ALLOW_SUDO=1
export FF_ALLOW_LIVE_INSTALL=1
export FF_WORDLIST_MUTATION=1
export FF_PORT_KNOCKING=1
export FF_PROXY_CAPTURE=1
export FF_PAYLOAD_ENCODING=1

# ── Phase 11.0: Full Visibility & Step Discipline ──────────────────────────
export FF_PARSER_MODE=teacher
export FF_STRICT_PHASE_LADDER=1
export FF_ADAPTIVE_BUDGET=1
export FF_LEARNING_SIGNAL_EXPORT=1

# ── Phase 14.0: Autonomous Reasoning Architecture ──────────────────────────
export FF_EVIDENCE_GRAPH=1
export FF_HYPOTHESIS_ENGINE=1
export FF_STRATEGY_PLAN=1
export FF_AUTONOMY_SCHEDULER=1
export FF_BC_LOSS=1
export FF_TEACHER_TRACE=1
export FF_AUX_HEADS=1
export FF_PARSER_TEACHER=1
# MS2/MS3 packs: OFF unless actively training these specific targets
export FF_MS2_KNOWLEDGE_PACK=0
export FF_MS3_KNOWLEDGE_PACK=0
export FF_MS2_SIMULATED_OUTPUT=0

# ── Phase 15.0: NEUROVORTEX ────────────────────────────────────────────────
export FF_NEUROMODULATORS=1
export FF_REFLEX_POLICY=1
export FF_ACTION_ARBITRATOR=1
export FF_WORKING_MEMORY=1
export FF_CONSOLIDATION=1
export FF_AGGRESSION_CONTROLLER=1
export FF_SEMANTIC_INDEX=1
export FF_BUDGET_MANAGER_V2=1
export FF_SENSORY_BUFFER=1

# ── Runtime Safety ─────────────────────────────────────────────────────────
# DO NOT set ARIASKA_DRY_RUN=1 here — this is for live lab mode

echo "[ARIASKA] All HTB flags activated. Run: python scripts/verify_flags.py"
