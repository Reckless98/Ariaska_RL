"""
core/ops/ops_state_encoder.py — Phase 38.5: OPS State Encoder Extension

Fills reserved zero-padded dims [237-269] of the 512-dim state vector
with OPS-specific signals:
  - Command lockout pressure
  - Exploit confidence distribution
  - Exploit cooldown pressure
  - Engagement stagnation and velocity
  - Token flex scale
  - Shell validation confidence
  - Phase invariant pressure
  - Domain discovery progress

This module provides a function ``inject_ops_features`` that takes an
existing 512-dim numpy array and writes into the reserved dims.
It does NOT create or replace the state vector — it augments it.

Section 17: OPS Intelligence Encoding (33 dims) [237-269]
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from core.ops.command_lockout import CommandLockout
    from core.ops.engagement_metrics import EngagementMetrics
    from core.ops.exploit_confidence import ExploitConfidenceTracker
    from core.ops.exploit_cooldown import ExploitCooldownManager
    from core.ops.token_flex import TokenFlexEngine

logger = logging.getLogger("ariaska.ops.state_encoder")

# ── Constants ────────────────────────────────────────────────────────────────
OPS_SECTION_START = 237
OPS_SECTION_DIMS = 33
OPS_SECTION_END = OPS_SECTION_START + OPS_SECTION_DIMS  # 270


def inject_ops_features(
    vec: np.ndarray,
    *,
    lockout_pressure: float = 0.0,
    lockout_count: int = 0,
    max_lockout_templates: int = 20,
    confidence_mean: float = 0.5,
    confidence_min: float = 0.5,
    confidence_max: float = 0.5,
    low_confidence_ratio: float = 0.0,
    cooldown_pressure: float = 0.0,
    cooldown_active_count: int = 0,
    max_cooldown_templates: int = 20,
    stagnation_level: float = 0.0,
    phase_velocity: float = 0.0,
    token_efficiency: float = 0.0,
    token_flex_scale: float = 1.0,
    engagement_progress: float = 0.0,
    discovery_rate: float = 0.0,
    exploit_success_rate: float = 0.0,
    shells_obtained: int = 0,
    flags_captured: int = 0,
    domain_count: int = 0,
    vhost_count: int = 0,
    phase_transitions_total: int = 0,
    tokens_used: int = 0,
    max_tokens: int = 1_148_850,
    codex_tier_pressure: float = 0.0,
    mini_tier_pressure: float = 0.0,
) -> np.ndarray:
    """
    Inject OPS-specific features into the state vector.

    Writes into dims [237-269] of the 512-dim vector.
    Returns the modified vector (mutation in-place + return).

    Args:
        vec: The 512-dim state vector (numpy array, modified in-place).
        lockout_pressure: Fraction of templates currently locked out.
        lockout_count: Number of locked-out templates.
        max_lockout_templates: Max expected lockout templates.
        confidence_mean: Mean exploit confidence across tracked exploits.
        confidence_min: Minimum exploit confidence.
        confidence_max: Maximum exploit confidence.
        low_confidence_ratio: Fraction of exploits below confidence threshold.
        cooldown_pressure: Fraction of templates currently on cooldown.
        cooldown_active_count: Number of templates on active cooldown.
        max_cooldown_templates: Max expected cooldown templates.
        stagnation_level: Engagement stagnation [0.0, 1.0].
        phase_velocity: Phase transitions per 100 steps.
        token_efficiency: Discoveries per 1000 tokens.
        token_flex_scale: Current token flex scale [0.5, 1.5].
        engagement_progress: Kill chain progress [0.0, 1.0].
        discovery_rate: Discoveries per step.
        exploit_success_rate: Exploit success rate [0.0, 1.0].
        shells_obtained: Number of shells obtained.
        flags_captured: Number of flags captured.
        domain_count: Number of tracked domains.
        vhost_count: Number of tracked vhosts.
        phase_transitions_total: Total phase transitions.
        tokens_used: Total tokens consumed.
        max_tokens: Maximum token budget.
        codex_tier_pressure: Codex tier budget utilisation [0.0, 1.0].
        mini_tier_pressure: Mini tier budget utilisation [0.0, 1.0].

    Returns:
        The modified vec array.
    """
    if vec.shape[0] < OPS_SECTION_END:
        logger.warning(
            "State vector too short (%d < %d), skipping OPS injection",
            vec.shape[0], OPS_SECTION_END,
        )
        return vec

    idx = OPS_SECTION_START

    # ── Command Lockout (3 dims) [237-239] ────────────────────────────
    vec[idx] = min(lockout_pressure, 1.0)
    idx += 1
    vec[idx] = min(float(lockout_count) / max(max_lockout_templates, 1), 1.0)
    idx += 1
    # Binary: any lockout active
    vec[idx] = 1.0 if lockout_count > 0 else 0.0
    idx += 1  # 240

    # ── Exploit Confidence (4 dims) [240-243] ─────────────────────────
    vec[idx] = min(confidence_mean, 1.0)
    idx += 1
    vec[idx] = min(confidence_min, 1.0)
    idx += 1
    vec[idx] = min(confidence_max, 1.0)
    idx += 1
    vec[idx] = min(low_confidence_ratio, 1.0)
    idx += 1  # 244

    # ── Exploit Cooldown (3 dims) [244-246] ───────────────────────────
    vec[idx] = min(cooldown_pressure, 1.0)
    idx += 1
    vec[idx] = min(
        float(cooldown_active_count) / max(max_cooldown_templates, 1), 1.0,
    )
    idx += 1
    # Binary: any cooldown active
    vec[idx] = 1.0 if cooldown_active_count > 0 else 0.0
    idx += 1  # 247

    # ── Engagement Progress (6 dims) [247-252] ────────────────────────
    vec[idx] = min(stagnation_level, 1.0)
    idx += 1
    vec[idx] = min(phase_velocity / 10.0, 1.0)  # normalise to ~10 max
    idx += 1
    vec[idx] = min(token_efficiency / 5.0, 1.0)  # normalise to ~5 max
    idx += 1
    vec[idx] = min(engagement_progress, 1.0)
    idx += 1
    vec[idx] = min(discovery_rate, 1.0)
    idx += 1
    vec[idx] = min(exploit_success_rate, 1.0)
    idx += 1  # 253

    # ── Token Flex (4 dims) [253-256] ─────────────────────────────────
    # Normalise scale from [0.5, 1.5] to [0.0, 1.0]
    vec[idx] = max(0.0, min((token_flex_scale - 0.5) / 1.0, 1.0))
    idx += 1
    # Token budget utilisation
    vec[idx] = min(float(tokens_used) / max(max_tokens, 1), 1.0)
    idx += 1
    vec[idx] = min(codex_tier_pressure, 1.0)
    idx += 1
    vec[idx] = min(mini_tier_pressure, 1.0)
    idx += 1  # 257

    # ── Shell & Flag Progress (4 dims) [257-260] ──────────────────────
    vec[idx] = min(float(shells_obtained) / 3.0, 1.0)
    idx += 1
    vec[idx] = 1.0 if shells_obtained > 0 else 0.0
    idx += 1
    vec[idx] = min(float(flags_captured) / 2.0, 1.0)
    idx += 1
    vec[idx] = 1.0 if flags_captured >= 2 else 0.0
    idx += 1  # 261

    # ── Domain Discovery (3 dims) [261-263] ───────────────────────────
    vec[idx] = min(float(domain_count) / 10.0, 1.0)
    idx += 1
    vec[idx] = min(float(vhost_count) / 10.0, 1.0)
    idx += 1
    vec[idx] = 1.0 if domain_count > 0 else 0.0
    idx += 1  # 264

    # ── Phase Dynamics (3 dims) [264-266] ─────────────────────────────
    vec[idx] = min(float(phase_transitions_total) / 10.0, 1.0)
    idx += 1
    # Normalised engagement progress squared (urgency signal)
    vec[idx] = min(engagement_progress ** 2, 1.0)
    idx += 1
    # Stagnation + cooldown compound pressure
    compound_pressure = (stagnation_level * 0.5 + cooldown_pressure * 0.3
                         + lockout_pressure * 0.2)
    vec[idx] = min(compound_pressure, 1.0)
    idx += 1  # 267

    # ── Reserve (3 dims) [267-269] ────────────────────────────────────
    # Unused, zero-padded for future OPS extensions
    idx += 3  # 270

    assert idx == OPS_SECTION_END, f"OPS encoder wrote to idx {idx}, expected {OPS_SECTION_END}"

    return vec


def collect_ops_signals(
    lockout: Optional[Any] = None,
    confidence: Optional[Any] = None,
    cooldown: Optional[Any] = None,
    metrics: Optional[Any] = None,
    flex_result: Optional[Any] = None,
    domain_manager: Optional[Any] = None,
    budget_stats: Optional[Dict[str, Any]] = None,
    current_step: int = 0,
) -> Dict[str, float]:
    """
    Collect OPS signals from live module instances into a flat dict
    suitable for passing to ``inject_ops_features(**signals)``.

    Each module is optional — missing modules contribute default values.

    Args:
        lockout: CommandLockout instance.
        confidence: ExploitConfidenceTracker instance.
        cooldown: ExploitCooldownManager instance.
        metrics: EngagementMetrics instance.
        flex_result: TokenFlexResult from TokenFlexEngine.compute().
        domain_manager: DomainManager instance.
        budget_stats: Dict from BudgetManagerV2.get_stats().
        current_step: Current step for cooldown availability checks.

    Returns:
        Dict of keyword args for ``inject_ops_features()``.
    """
    signals: Dict[str, float] = {}

    # Command Lockout
    if lockout is not None:
        stats = lockout.get_stats()
        locked = stats.get("currently_locked", 0)
        total = stats.get("total_tracked", 1)
        signals["lockout_pressure"] = locked / max(total, 1)
        signals["lockout_count"] = locked
    else:
        signals["lockout_pressure"] = 0.0
        signals["lockout_count"] = 0

    # Exploit Confidence
    if confidence is not None:
        ranked = confidence.get_ranked_exploits()
        if ranked:
            confs = [e.current_confidence for e in ranked]
            signals["confidence_mean"] = sum(confs) / len(confs)
            signals["confidence_min"] = min(confs)
            signals["confidence_max"] = max(confs)
            low = sum(1 for c in confs if c < 0.35)
            signals["low_confidence_ratio"] = low / len(confs)
        else:
            signals["confidence_mean"] = 0.5
            signals["confidence_min"] = 0.5
            signals["confidence_max"] = 0.5
            signals["low_confidence_ratio"] = 0.0
    else:
        signals["confidence_mean"] = 0.5
        signals["confidence_min"] = 0.5
        signals["confidence_max"] = 0.5
        signals["low_confidence_ratio"] = 0.0

    # Exploit Cooldown
    if cooldown is not None:
        entries = cooldown._entries if hasattr(cooldown, "_entries") else {}
        total_tracked = len(entries)
        if total_tracked > 0:
            all_names = list(entries.keys())
            avail = cooldown.get_available_exploits(all_names, current_step)
            on_cooldown = max(0, total_tracked - len(avail))
        else:
            on_cooldown = 0
        signals["cooldown_pressure"] = on_cooldown / max(total_tracked, 1)
        signals["cooldown_active_count"] = on_cooldown
    else:
        signals["cooldown_pressure"] = 0.0
        signals["cooldown_active_count"] = 0

    # Engagement Metrics
    if metrics is not None:
        progress = metrics.get_progress()
        signals["stagnation_level"] = metrics.get_stagnation_level()
        signals["phase_velocity"] = metrics.get_phase_velocity()
        signals["token_efficiency"] = metrics.get_token_efficiency()
        signals["engagement_progress"] = progress.get("phase_progress", 0.0)
        signals["discovery_rate"] = progress.get("discovery_rate", 0.0)
        signals["exploit_success_rate"] = progress.get("exploit_success_rate", 0.0)
        signals["shells_obtained"] = progress.get("shells_obtained", 0)
        signals["flags_captured"] = progress.get("flags_count", 0)
        signals["phase_transitions_total"] = progress.get("phase_transitions", 0)
        signals["tokens_used"] = progress.get("tokens_used", 0)
    else:
        signals["stagnation_level"] = 0.0
        signals["phase_velocity"] = 0.0
        signals["token_efficiency"] = 0.0
        signals["engagement_progress"] = 0.0
        signals["discovery_rate"] = 0.0
        signals["exploit_success_rate"] = 0.0
        signals["shells_obtained"] = 0
        signals["flags_captured"] = 0
        signals["phase_transitions_total"] = 0
        signals["tokens_used"] = 0

    # Token Flex
    if flex_result is not None:
        signals["token_flex_scale"] = flex_result.scale
    else:
        signals["token_flex_scale"] = 1.0

    # Domain Manager
    if domain_manager is not None:
        confirmed = domain_manager.get_confirmed_domains()
        vhosts = domain_manager.get_vhosts()
        signals["domain_count"] = len(confirmed)
        signals["vhost_count"] = len(vhosts)
    else:
        signals["domain_count"] = 0
        signals["vhost_count"] = 0

    # Budget stats
    if budget_stats is not None:
        tiers = budget_stats.get("tiers", {})
        codex_tier = tiers.get("codex", {})
        mini_tier = tiers.get("mini", {})
        signals["codex_tier_pressure"] = codex_tier.get("used", 0) / max(
            codex_tier.get("budget", 1), 1)
        signals["mini_tier_pressure"] = mini_tier.get("used", 0) / max(
            mini_tier.get("budget", 1), 1)
    else:
        signals["codex_tier_pressure"] = 0.0
        signals["mini_tier_pressure"] = 0.0

    return signals
