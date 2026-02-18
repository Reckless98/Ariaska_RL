#!/usr/bin/env python3
"""
scripts/verify_flags.py — Phase 15 Flag Verification

Reads every field from FeatureFlags, prints canonical table,
and exits non-zero if any required flag is not in its expected state.
"""

import os
import sys
from dataclasses import fields

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.feature_flags import FeatureFlags, reset_feature_flags, get_feature_flags

# Force a fresh read of environment
reset_feature_flags()
ff = get_feature_flags()

# ── Flag → env_var mapping ──────────────────────────────────────────────────

FIELD_ENV_MAP = {
    "ddqn_single_select": "FF_DDQN_SINGLE_SELECT_PER_STEP",
    "single_parse_cache": "FF_SINGLE_PARSE_CACHE",
    "ppo_reward_attribution_fix": "FF_PPO_REWARD_ATTRIBUTION_FIX",
    "tactical_cortex_gate": "FF_TACTICAL_CORTEX_GATE",
    "tactical_ppo_block": "FF_TACTICAL_PPO_BLOCK",
    "executive_cortex": "FF_EXECUTIVE_CORTEX",
    "target_profiler_pipeline": "FF_TARGET_PROFILER_PIPELINE",
    "kr_enrichment": "FF_KR_ENRICHMENT",
    "kr_confidence_threshold": None,  # no env var
    "jsonl_telemetry": "FF_JSONL_TELEMETRY",
    "rich_hud": "FF_RICH_HUD",
    "transparent_mode": "FF_TRANSPARENT_MODE",
    "kg_write": "FF_KG_WRITE",
    "llm_strategic_planner": "FF_LLM_STRATEGIC_PLANNER",
    "llm_tactical_advisor": "FF_LLM_TACTICAL_ADVISOR",
    "llm_judge_ranker": "FF_LLM_JUDGE_RANKER",
    "llm_postmortem_skills": "FF_LLM_POSTMORTEM_SKILLS",
    "dagger_corrections": "FF_DAGGER_CORRECTIONS",
    "ingest_pipeline": "FF_INGEST_PIPELINE",
    "privilege_gating": "FF_PRIVILEGE_GATING",
    "allow_sudo": "FF_ALLOW_SUDO",
    "allow_live_install": "FF_ALLOW_LIVE_INSTALL",
    "sudo_mode": None,  # string, no env var
    "wordlist_mutation": "FF_WORDLIST_MUTATION",
    "port_knocking": "FF_PORT_KNOCKING",
    "proxy_capture": "FF_PROXY_CAPTURE",
    "payload_encoding": "FF_PAYLOAD_ENCODING",
    "parser_mode": "FF_PARSER_MODE",
    "strict_phase_ladder": "FF_STRICT_PHASE_LADDER",
    "adaptive_budget": "FF_ADAPTIVE_BUDGET",
    "learning_signal_export": "FF_LEARNING_SIGNAL_EXPORT",
    "evidence_graph": "FF_EVIDENCE_GRAPH",
    "hypothesis_engine": "FF_HYPOTHESIS_ENGINE",
    "strategy_plan": "FF_STRATEGY_PLAN",
    "autonomy_scheduler": "FF_AUTONOMY_SCHEDULER",
    "bc_loss": "FF_BC_LOSS",
    "teacher_trace": "FF_TEACHER_TRACE",
    "aux_heads": "FF_AUX_HEADS",
    "parser_teacher": "FF_PARSER_TEACHER",
    "ms2_knowledge_pack": "FF_MS2_KNOWLEDGE_PACK",
    "ms3_knowledge_pack": "FF_MS3_KNOWLEDGE_PACK",
    "ms2_simulated_output": "FF_MS2_SIMULATED_OUTPUT",
    "neuromodulators": "FF_NEUROMODULATORS",
    "reflex_policy": "FF_REFLEX_POLICY",
    "action_arbitrator": "FF_ACTION_ARBITRATOR",
    "working_memory": "FF_WORKING_MEMORY",
    "consolidation": "FF_CONSOLIDATION",
    "aggression_controller": "FF_AGGRESSION_CONTROLLER",
    "semantic_index": "FF_SEMANTIC_INDEX",
    "budget_manager_v2": "FF_BUDGET_MANAGER_V2",
    "sensory_buffer": "FF_SENSORY_BUFFER",
    # Phase 16.0
    "progress_estimator": "FF_PROGRESS_ESTIMATOR",
}

# Expected values for HTB mode
EXPECTED_HTB = {
    # All bool flags ON except MS2/MS3 packs
    "ms2_knowledge_pack": False,
    "ms3_knowledge_pack": False,
    "ms2_simulated_output": False,
    # String flags
    "parser_mode": "teacher",
    "sudo_mode": "prompt",
    "kr_confidence_threshold": 0.3,
}

# ── Verification ────────────────────────────────────────────────────────────

print("=" * 80)
print("ARIASKA_RL — Phase 15 NEUROVORTEX Flag Ledger")
print("=" * 80)
print(f"{'Flag':<30} {'Type':<8} {'Value':<25} {'Env Var':<35} {'OK?'}")
print("-" * 80)

errors = []

for f in fields(ff):
    name = f.name
    val = getattr(ff, name)
    env_var = FIELD_ENV_MAP.get(name, "???")
    env_display = env_var if env_var else "(hardcoded)"

    # Determine expected value
    if name in EXPECTED_HTB:
        expected = EXPECTED_HTB[name]
    elif isinstance(val, bool):
        expected = True
    else:
        expected = val  # string/float: just verify it resolved

    ok = (val == expected)
    status = "✓" if ok else "✗"

    val_display = str(val)
    if isinstance(val, bool):
        val_display = "ON" if val else "OFF"

    print(f"{name:<30} {type(val).__name__:<8} {val_display:<25} {env_display:<35} {status}")

    if not ok:
        errors.append(f"{name}: got {val!r}, expected {expected!r}")

print("-" * 80)

if errors:
    print(f"\n❌ {len(errors)} FLAG(S) NOT IN EXPECTED STATE:")
    for e in errors:
        print(f"   • {e}")
    sys.exit(1)
else:
    print(f"\n✓ ALL {len(fields(ff))} FLAGS IN EXPECTED HTB STATE")
    sys.exit(0)
