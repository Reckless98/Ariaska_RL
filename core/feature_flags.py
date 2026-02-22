#!/usr/bin/env python3
"""
core/feature_flags.py — Phase 9.5+ Feature Flag Registry

Centralized feature flags for incremental rollout and rollback.
All flags default to safe values. Override via environment variables
(prefix FF_) or programmatic set_feature_flag().

Usage:
    from core.feature_flags import ff
    if ff.ddqn_single_select:
        # new behavior
    else:
        # old behavior

Rollback:
    export FF_DDQN_SINGLE_SELECT_PER_STEP=0   # revert to double-call
    export FF_SINGLE_PARSE_CACHE=0             # revert to triple-parse
"""

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Dict, Any

_lock = threading.Lock()


def _env_bool(name: str, default: bool = True) -> bool:
    """Read a boolean from env. '0'/'false'/'no' → False, else default."""
    val = os.environ.get(name, "").strip().lower()
    if val in ("0", "false", "no", "off"):
        return False
    if val in ("1", "true", "yes", "on"):
        return True
    return default


_VALID_GATE_MODES = {"off", "log", "enforce"}


def _validate_exploit_gate(value: str) -> str:
    """Validate strict_exploit_gate value; fallback to 'log' on invalid."""
    cleaned = value.strip().lower()
    if cleaned in _VALID_GATE_MODES:
        return cleaned
    logging.getLogger("ariaska.feature_flags").warning(
        f"Invalid FF_STRICT_EXPLOIT_GATE='{value}', falling back to 'log'")
    return "log"


@dataclass
class FeatureFlags:
    """Phase 9.5–9.7 feature flags with env-var overrides."""

    # ── Phase 9.5: Correctness ──────────────────────────────────────
    ddqn_single_select: bool = field(
        default_factory=lambda: _env_bool("FF_DDQN_SINGLE_SELECT_PER_STEP", True))
    single_parse_cache: bool = field(
        default_factory=lambda: _env_bool("FF_SINGLE_PARSE_CACHE", True))
    ppo_reward_attribution_fix: bool = field(
        default_factory=lambda: _env_bool("FF_PPO_REWARD_ATTRIBUTION_FIX", True))

    # ── Phase 9.6: Architecture Wiring ──────────────────────────────
    tactical_cortex_gate: bool = field(
        default_factory=lambda: _env_bool("FF_TACTICAL_CORTEX_GATE", True))
    tactical_ppo_block: bool = field(
        default_factory=lambda: _env_bool("FF_TACTICAL_PPO_BLOCK", True))
    executive_cortex: bool = field(
        default_factory=lambda: _env_bool("FF_EXECUTIVE_CORTEX", True))
    target_profiler_pipeline: bool = field(
        default_factory=lambda: _env_bool("FF_TARGET_PROFILER_PIPELINE", True))
    kr_enrichment: bool = field(
        default_factory=lambda: _env_bool("FF_KR_ENRICHMENT", True))
    kr_confidence_threshold: float = 0.3

    # ── Phase 9.7: Telemetry & UI ───────────────────────────────────
    jsonl_telemetry: bool = field(
        default_factory=lambda: _env_bool("FF_JSONL_TELEMETRY", True))
    rich_hud: bool = field(
        default_factory=lambda: _env_bool("FF_RICH_HUD", True))
    transparent_mode: bool = field(
        default_factory=lambda: _env_bool("FF_TRANSPARENT_MODE", True))

    # ── Phase 9.7+: KG & LLM ───────────────────────────────────────
    kg_write: bool = field(
        default_factory=lambda: _env_bool("FF_KG_WRITE", True))
    llm_strategic_planner: bool = field(
        default_factory=lambda: _env_bool("FF_LLM_STRATEGIC_PLANNER", True))
    llm_tactical_advisor: bool = field(
        default_factory=lambda: _env_bool("FF_LLM_TACTICAL_ADVISOR", True))
    llm_judge_ranker: bool = field(
        default_factory=lambda: _env_bool("FF_LLM_JUDGE_RANKER", True))
    llm_postmortem_skills: bool = field(
        default_factory=lambda: _env_bool("FF_LLM_POSTMORTEM_SKILLS", True))
    dagger_corrections: bool = field(
        default_factory=lambda: _env_bool("FF_DAGGER_CORRECTIONS", True))
    ingest_pipeline: bool = field(
        default_factory=lambda: _env_bool("FF_INGEST_PIPELINE", True))

    # ── Phase 10.1: Pre-HTB Capability Hardening ────────────────────
    privilege_gating: bool = field(
        default_factory=lambda: _env_bool("FF_PRIVILEGE_GATING", True))
    allow_sudo: bool = field(
        default_factory=lambda: _env_bool("FF_ALLOW_SUDO", True))
    allow_live_install: bool = field(
        default_factory=lambda: _env_bool("FF_ALLOW_LIVE_INSTALL", True))
    sudo_mode: str = "noninteractive"  # Full autonomy
    wordlist_mutation: bool = field(
        default_factory=lambda: _env_bool("FF_WORDLIST_MUTATION", True))
    port_knocking: bool = field(
        default_factory=lambda: _env_bool("FF_PORT_KNOCKING", True))
    proxy_capture: bool = field(
        default_factory=lambda: _env_bool("FF_PROXY_CAPTURE", True))
    payload_encoding: bool = field(
        default_factory=lambda: _env_bool("FF_PAYLOAD_ENCODING", True))

    # ── Phase 11.0: Full Visibility & Step Discipline ───────────────
    parser_mode: str = field(
        default_factory=lambda: os.environ.get("FF_PARSER_MODE", "intelligent_fullparse"))
    strict_phase_ladder: bool = field(
        default_factory=lambda: _env_bool("FF_STRICT_PHASE_LADDER", True))  # Post-Phase 20: ON by default
    adaptive_budget: bool = field(
        default_factory=lambda: _env_bool("FF_ADAPTIVE_BUDGET", True))
    learning_signal_export: bool = field(
        default_factory=lambda: _env_bool("FF_LEARNING_SIGNAL_EXPORT", True))

    # ── Phase 14.0: Autonomous Reasoning Architecture ───────────────
    evidence_graph: bool = field(
        default_factory=lambda: _env_bool("FF_EVIDENCE_GRAPH", True))
    hypothesis_engine: bool = field(
        default_factory=lambda: _env_bool("FF_HYPOTHESIS_ENGINE", True))
    strategy_plan: bool = field(
        default_factory=lambda: _env_bool("FF_STRATEGY_PLAN", True))
    autonomy_scheduler: bool = field(
        default_factory=lambda: _env_bool("FF_AUTONOMY_SCHEDULER", True))
    bc_loss: bool = field(
        default_factory=lambda: _env_bool("FF_BC_LOSS", True))
    teacher_trace: bool = field(
        default_factory=lambda: _env_bool("FF_TEACHER_TRACE", True))
    aux_heads: bool = field(
        default_factory=lambda: _env_bool("FF_AUX_HEADS", True))
    parser_teacher: bool = field(
        default_factory=lambda: _env_bool("FF_PARSER_TEACHER", True))
    ms2_knowledge_pack: bool = field(
        default_factory=lambda: _env_bool("FF_MS2_KNOWLEDGE_PACK", True))
    ms3_knowledge_pack: bool = field(
        default_factory=lambda: _env_bool("FF_MS3_KNOWLEDGE_PACK", True))
    ms2_simulated_output: bool = field(
        default_factory=lambda: _env_bool("FF_MS2_SIMULATED_OUTPUT", True))

    # ── Phase 15.0: NEUROVORTEX — Biologically-Inspired Control ─────
    neuromodulators: bool = field(
        default_factory=lambda: _env_bool("FF_NEUROMODULATORS", True))
    reflex_policy: bool = field(
        default_factory=lambda: _env_bool("FF_REFLEX_POLICY", True))
    action_arbitrator: bool = field(
        default_factory=lambda: _env_bool("FF_ACTION_ARBITRATOR", True))
    working_memory: bool = field(
        default_factory=lambda: _env_bool("FF_WORKING_MEMORY", True))
    consolidation: bool = field(
        default_factory=lambda: _env_bool("FF_CONSOLIDATION", True))
    aggression_controller: bool = field(
        default_factory=lambda: _env_bool("FF_AGGRESSION_CONTROLLER", True))
    semantic_index: bool = field(
        default_factory=lambda: _env_bool("FF_SEMANTIC_INDEX", True))
    budget_manager_v2: bool = field(
        default_factory=lambda: _env_bool("FF_BUDGET_MANAGER_V2", True))
    sensory_buffer: bool = field(
        default_factory=lambda: _env_bool("FF_SENSORY_BUFFER", True))

    # ── Phase 16.0: Progress Estimator ──────────────────────────────
    progress_estimator: bool = field(
        default_factory=lambda: _env_bool("FF_PROGRESS_ESTIMATOR", True))

    # ── Phase 19.0: HTB Hardening & Intelligence ────────────────────
    htb_auto_profile: bool = field(
        default_factory=lambda: _env_bool("FF_HTB_AUTO_PROFILE", True))
    phase19_regex: bool = field(
        default_factory=lambda: _env_bool("FF_PHASE19_REGEX", True))
    pcap_extraction: bool = field(
        default_factory=lambda: _env_bool("FF_PCAP_EXTRACTION", True))
    crushftp_templates: bool = field(
        default_factory=lambda: _env_bool("FF_CRUSHFTP_TEMPLATES", True))
    erlang_templates: bool = field(
        default_factory=lambda: _env_bool("FF_ERLANG_TEMPLATES", True))

    # ── Phase 27.0: Intelligence Pipeline ───────────────────────────
    use_micro_chain: bool = field(
        default_factory=lambda: _env_bool("FF_USE_MICRO_CHAIN", True))
    strict_exploit_gate: str = field(
        default_factory=lambda: _validate_exploit_gate(
            os.environ.get("FF_STRICT_EXPLOIT_GATE", "enforce")))
    parallel_agents: bool = field(
        default_factory=lambda: _env_bool("FF_PARALLEL_AGENTS", True))
    episode_summary_embedding: bool = field(
        default_factory=lambda: _env_bool("FF_EPISODE_SUMMARY_EMBEDDING", True))

    # ── Phase 37.0: Level 5 GPT↔RL Neural Integration ──────────────
    llm_policy_bridge: bool = field(
        default_factory=lambda: _env_bool("FF_LLM_POLICY_BRIDGE", True))

    # ── Phase 38.0: OPS Authority & Full Autonomy ───────────────────
    ops_preflight: bool = field(
        default_factory=lambda: _env_bool("FF_OPS_PREFLIGHT", True))
    sudo_handler: bool = field(
        default_factory=lambda: _env_bool("FF_SUDO_HANDLER", True))
    execution_classifier: bool = field(
        default_factory=lambda: _env_bool("FF_EXECUTION_CLASSIFIER", True))

    # ── Phase 38.1: Discovery Trust & Clean Parser ──────────────────
    discovery_trust: bool = field(
        default_factory=lambda: _env_bool("FF_DISCOVERY_TRUST", True))
    service_normalization: bool = field(
        default_factory=lambda: _env_bool("FF_SERVICE_NORMALIZATION", True))
    spike_guard: bool = field(
        default_factory=lambda: _env_bool("FF_SPIKE_GUARD", True))

    # ── Phase 38.2: Phase Hardening + Shell Validation ──────────────
    phase_invariants: bool = field(
        default_factory=lambda: _env_bool("FF_PHASE_INVARIANTS", True))
    shell_validator: bool = field(
        default_factory=lambda: _env_bool("FF_SHELL_VALIDATOR", True))
    domain_manager: bool = field(
        default_factory=lambda: _env_bool("FF_DOMAIN_MANAGER", True))

    # ── Phase 38.3: Hypothesis + Anti-Repeat + Cooldown ────────────
    command_lockout: bool = field(
        default_factory=lambda: _env_bool("FF_COMMAND_LOCKOUT", True))
    exploit_confidence: bool = field(
        default_factory=lambda: _env_bool("FF_EXPLOIT_CONFIDENCE", True))
    exploit_cooldown: bool = field(
        default_factory=lambda: _env_bool("FF_EXPLOIT_COOLDOWN", True))

    # ── Phase 38.4: Token Flex + Engagement Metrics ────────────────
    engagement_metrics: bool = field(
        default_factory=lambda: _env_bool("FF_ENGAGEMENT_METRICS", True))
    token_flex: bool = field(
        default_factory=lambda: _env_bool("FF_TOKEN_FLEX", True))

    # ── Phase 38.5: StateEncoder + Dashboard ───────────────────────
    ops_state_encoder: bool = field(
        default_factory=lambda: _env_bool("FF_OPS_STATE_ENCODER", True))
    ops_dashboard: bool = field(
        default_factory=lambda: _env_bool("FF_OPS_DASHBOARD", True))

    # ── Phase 38.6: OpsHub Integration ─────────────────────────────
    ops_hub: bool = field(
        default_factory=lambda: _env_bool("FF_OPS_HUB", True))

    # ── Phase 39.1: Orion Deep-Rethink Escalation ──────────────────
    orion_rethink: bool = field(
        default_factory=lambda: _env_bool("FF_ORION_RETHINK", True))

    # ── Phase 39.2: Trust Weights + Annealing ──────────────────────
    trust_weights: bool = field(
        default_factory=lambda: _env_bool("FF_TRUST_WEIGHTS", True))

    # ── Phase 39.3: CAP Regression Gate ────────────────────────────
    cap_gate: bool = field(
        default_factory=lambda: _env_bool("FF_CAP_GATE", True))

    # ── Phase 39.4: Debug Trace Instrumentation ────────────────────
    debug_trace: bool = field(
        default_factory=lambda: _env_bool("FF_DEBUG_TRACE", True))

    # ── Phase 40.0: UI/UX Overhaul ─────────────────────────────────
    dashboard_v6: bool = field(
        default_factory=lambda: _env_bool("FF_DASHBOARD_V6", True))

    # ── Phase 40.1: Persistent SSH Session Pool ────────────────────
    ssh_pool: bool = field(
        default_factory=lambda: _env_bool("FF_SSH_POOL", True))

    # ── Phase 40.2: Parallel Agent Execution ───────────────────────
    parallel_exec: bool = field(
        default_factory=lambda: _env_bool("FF_PARALLEL_EXEC", True))

    # ── Phase 40.3: OS-Aware Command Filtering ────────────────────
    os_aware_filter: bool = field(
        default_factory=lambda: _env_bool("FF_OS_AWARE_FILTER", True))

    # ── Phase 40.4: Adaptive Command Pool Narrowing ────────────────
    pool_narrower: bool = field(
        default_factory=lambda: _env_bool("FF_POOL_NARROWER", True))

    # ── Phase 40.5: Web Path Auto-Discovery Pipeline ──────────────
    auto_web_probe: bool = field(
        default_factory=lambda: _env_bool("FF_AUTO_WEB_PROBE", True))

    # ── Phase 43: GPU Acceleration + Local LLM ─────────────────
    local_llm: bool = field(
        default_factory=lambda: _env_bool("FF_LOCAL_LLM", False))  # Auto-detected at startup
    local_llm_offload_nano: bool = field(
        default_factory=lambda: _env_bool("FF_LOCAL_LLM_OFFLOAD_NANO", True))
    local_llm_offload_mini: bool = field(
        default_factory=lambda: _env_bool("FF_LOCAL_LLM_OFFLOAD_MINI", True))

    # ── Phase 42: Deep Wiring + HTB Evolution ──────────────────────
    her_wiring: bool = field(
        default_factory=lambda: _env_bool("FF_HER_WIRING", True))
    dagger_wiring: bool = field(
        default_factory=lambda: _env_bool("FF_DAGGER_WIRING", True))
    reflective_meta_learner: bool = field(
        default_factory=lambda: _env_bool("FF_REFLECTIVE_META_LEARNER", True))
    phase_timeout: bool = field(
        default_factory=lambda: _env_bool("FF_PHASE_TIMEOUT", True))
    ctf_tracker: bool = field(
        default_factory=lambda: _env_bool("FF_CTF_TRACKER", True))
    credential_sprayer: bool = field(
        default_factory=lambda: _env_bool("FF_CREDENTIAL_SPRAYER", True))
    episodic_memory_vote: bool = field(
        default_factory=lambda: _env_bool("FF_EPISODIC_MEMORY_VOTE", True))
    contrastive_ppo: bool = field(
        default_factory=lambda: _env_bool("FF_CONTRASTIVE_PPO", True))
    evidence_graph_v2: bool = field(
        default_factory=lambda: _env_bool("FF_EVIDENCE_GRAPH_V2", True))
    action_grammar: bool = field(
        default_factory=lambda: _env_bool("FF_ACTION_GRAMMAR", True))
    service_fingerprints: bool = field(
        default_factory=lambda: _env_bool("FF_SERVICE_FINGERPRINTS", True))
    ttf_tracker: bool = field(
        default_factory=lambda: _env_bool("FF_TTF_TRACKER", True))
    chain_scorer: bool = field(
        default_factory=lambda: _env_bool("FF_CHAIN_SCORER", True))
    hallucination_guard: bool = field(
        default_factory=lambda: _env_bool("FF_HALLUCINATION_GUARD", True))

    # ── C07: Algorithm Module Flags ─────────────────────────────────
    cognition_node: bool = field(
        default_factory=lambda: _env_bool("FF_COGNITION_NODE", True))
    sac_shadow: bool = field(
        default_factory=lambda: _env_bool("FF_SAC_SHADOW", True))
    reptile_meta: bool = field(
        default_factory=lambda: _env_bool("FF_REPTILE_META", False))
    optuna_sweep: bool = field(
        default_factory=lambda: _env_bool("FF_OPTUNA_SWEEP", False))
    per_loss_grad_log: bool = field(
        default_factory=lambda: _env_bool("FF_PER_LOSS_GRAD_LOG", False))
    source_win_rate_flag: bool = field(
        default_factory=lambda: _env_bool("FF_SOURCE_WIN_RATE", True))
    heldout_eval: bool = field(
        default_factory=lambda: _env_bool("FF_HELDOUT_EVAL", False))


# Global singleton
_ff: FeatureFlags = FeatureFlags()


def get_feature_flags() -> FeatureFlags:
    """Get current feature flags (thread-safe read)."""
    with _lock:
        return _ff


def set_feature_flag(name: str, value: Any) -> None:
    """Set a single feature flag by name (thread-safe)."""
    with _lock:
        if hasattr(_ff, name):
            setattr(_ff, name, value)
        else:
            raise ValueError(f"Unknown feature flag: {name}")


def reset_feature_flags() -> None:
    """Reset all flags to defaults (for testing)."""
    global _ff
    with _lock:
        _ff = FeatureFlags()


# ── Phase 10.0: Profile Resolver ────────────────────────────────────────────

def resolve_profile() -> str:
    """
    Auto-detect runtime profile and flip feature flags accordingly.

    Profiles:
      CLOUD          — OPENAI_API_KEY present → turn ON all 5 LLM role flags
      DETERMINISTIC  — Running under pytest → all LLM flags OFF (safe)
      OFFLINE        — No API key → all LLM flags OFF

    DOES NOT modify token budgets, call limits, or LLM model routing.
    Only flips the 5 feature flags: llm_strategic_planner, llm_tactical_advisor,
    llm_judge_ranker, llm_postmortem_skills, dagger_corrections.

    Returns the detected profile name.
    """
    global _ff
    import sys

    # Detect if running under pytest
    is_pytest = (
        "pytest" in sys.modules
        or "_pytest" in sys.modules
        or os.environ.get("ARIASKA_DRY_RUN", "") == "1"
        or os.environ.get("PYTEST_CURRENT_TEST", "") != ""
    )

    has_api_key = bool(os.environ.get("OPENAI_API_KEY", "").strip())

    if is_pytest:
        profile = "DETERMINISTIC"
    elif has_api_key:
        profile = "CLOUD"
    else:
        profile = "OFFLINE"

    llm_flags = [
        "llm_strategic_planner",
        "llm_tactical_advisor",
        "llm_judge_ranker",
        "llm_postmortem_skills",
        "dagger_corrections",
    ]

    with _lock:
        if profile == "CLOUD":
            for flag in llm_flags:
                if not _env_bool(f"FF_{flag.upper()}", True):
                    # Explicit env override says OFF — respect it
                    continue
                setattr(_ff, flag, True)
            # Phase 11.0: CLOUD profile gets full-parse mode by default
            if not os.environ.get("FF_PARSER_MODE"):
                _ff.parser_mode = "intelligent_fullparse"
        else:
            for flag in llm_flags:
                setattr(_ff, flag, False)
            # Post-Phase 20: Always intelligent_fullparse, all profiles
            if not os.environ.get("FF_PARSER_MODE"):
                _ff.parser_mode = "intelligent_fullparse"
            if profile == "DETERMINISTIC":
            # Post-Phase 20: Keep strict_phase_ladder ON even in tests
            # _ff.strict_phase_ladder = False  # Legacy: was disabled in tests
                _ff.learning_signal_export = False

        # Phase 19: Auto-switch to fullparse for HTB/live targets
        # Even non-CLOUD profiles benefit from fullparse when parsing
        # real tool output from live targets.
        if not os.environ.get("FF_PARSER_MODE"):
            _target = os.environ.get("ARIASKA_TARGET", "")
            _env_type = os.environ.get("ARIASKA_ENV", "")
            if (_target.startswith("10.10.10.") or
                _target.startswith("10.10.11.") or
                _env_type in ("htb", "live", "msf")):
                _ff.parser_mode = "intelligent_fullparse"
                logger.info("[P19] Auto-switched parser to intelligent_fullparse for live/HTB target")

    logger.info(f"[PROFILE] Resolved profile: {profile} "
                f"(api_key={'yes' if has_api_key else 'no'}, "
                f"pytest={'yes' if is_pytest else 'no'})")
    return profile


# Module-level logger
logger = logging.getLogger("ariaska.feature_flags")


# Convenience alias
ff = _ff
