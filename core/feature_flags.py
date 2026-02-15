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
        default_factory=lambda: _env_bool("FF_TACTICAL_PPO_BLOCK", False))
    executive_cortex: bool = field(
        default_factory=lambda: _env_bool("FF_EXECUTIVE_CORTEX", True))
    target_profiler_pipeline: bool = field(
        default_factory=lambda: _env_bool("FF_TARGET_PROFILER_PIPELINE", False))
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
        default_factory=lambda: _env_bool("FF_KG_WRITE", False))
    llm_strategic_planner: bool = field(
        default_factory=lambda: _env_bool("FF_LLM_STRATEGIC_PLANNER", False))
    llm_tactical_advisor: bool = field(
        default_factory=lambda: _env_bool("FF_LLM_TACTICAL_ADVISOR", False))
    llm_judge_ranker: bool = field(
        default_factory=lambda: _env_bool("FF_LLM_JUDGE_RANKER", False))
    llm_postmortem_skills: bool = field(
        default_factory=lambda: _env_bool("FF_LLM_POSTMORTEM_SKILLS", False))
    dagger_corrections: bool = field(
        default_factory=lambda: _env_bool("FF_DAGGER_CORRECTIONS", False))
    ingest_pipeline: bool = field(
        default_factory=lambda: _env_bool("FF_INGEST_PIPELINE", False))


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


# Convenience alias
ff = _ff
