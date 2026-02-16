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

    # ── Phase 10.1: Pre-HTB Capability Hardening ────────────────────
    privilege_gating: bool = field(
        default_factory=lambda: _env_bool("FF_PRIVILEGE_GATING", True))
    allow_sudo: bool = field(
        default_factory=lambda: _env_bool("FF_ALLOW_SUDO", False))
    allow_live_install: bool = field(
        default_factory=lambda: _env_bool("FF_ALLOW_LIVE_INSTALL", False))
    sudo_mode: str = "prompt"  # "prompt" | "noninteractive"
    wordlist_mutation: bool = field(
        default_factory=lambda: _env_bool("FF_WORDLIST_MUTATION", False))
    port_knocking: bool = field(
        default_factory=lambda: _env_bool("FF_PORT_KNOCKING", False))
    proxy_capture: bool = field(
        default_factory=lambda: _env_bool("FF_PROXY_CAPTURE", False))
    payload_encoding: bool = field(
        default_factory=lambda: _env_bool("FF_PAYLOAD_ENCODING", False))


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
        else:
            for flag in llm_flags:
                setattr(_ff, flag, False)

    logger.info(f"[PROFILE] Resolved profile: {profile} "
                f"(api_key={'yes' if has_api_key else 'no'}, "
                f"pytest={'yes' if is_pytest else 'no'})")
    return profile


# Module-level logger
logger = logging.getLogger("ariaska.feature_flags")


# Convenience alias
ff = _ff
