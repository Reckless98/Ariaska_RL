"""
core/ops/discovery_trust.py — Phase 38.1: Discovery Trust Engine

Trust scoring for discoveries based on verification level, source
credibility, and cross-corroboration.

Rules:
  - Every discovery gets a VerificationLevel: UNVERIFIED, SINGLE_SOURCE,
    CORROBORATED, CONFIRMED.
  - Trust score [0.0, 1.0] gates reward multiplier.
  - Regex-only discoveries start at SINGLE_SOURCE (0.6).
  - LLM-confirmed or multi-stage discoveries get CORROBORATED (0.8).
  - Cross-agent or multi-tool corroboration → CONFIRMED (1.0).
  - Spike guard: caps per-step discovery bonus at MAX_STEP_BONUS.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

logger = logging.getLogger("ariaska.ops.discovery_trust")

# ── Verification Levels ──────────────────────────────────────────────────────

class VerificationLevel(Enum):
    """How thoroughly a discovery has been verified."""
    UNVERIFIED = "unverified"           # Raw extraction, not validated
    SINGLE_SOURCE = "single_source"     # One parser stage confirmed
    CORROBORATED = "corroborated"       # Multiple stages or LLM confirmed
    CONFIRMED = "confirmed"             # Cross-agent or multi-tool

    @property
    def trust_score(self) -> float:
        """Default trust score for this level."""
        return _LEVEL_SCORES[self]


_LEVEL_SCORES: Dict[VerificationLevel, float] = {
    VerificationLevel.UNVERIFIED: 0.3,
    VerificationLevel.SINGLE_SOURCE: 0.6,
    VerificationLevel.CORROBORATED: 0.8,
    VerificationLevel.CONFIRMED: 1.0,
}


# ── Constants ────────────────────────────────────────────────────────────────

# Maximum discovery bonus per step (spike guard).
# Prevents a single step from giving +200 reward due to noisy parsing.
MAX_STEP_DISCOVERY_BONUS: float = 45.0

# Source stage → base verification level mapping.
_STAGE_TO_LEVEL: Dict[str, VerificationLevel] = {
    "regex": VerificationLevel.SINGLE_SOURCE,
    "llm": VerificationLevel.CORROBORATED,
    "venice": VerificationLevel.CORROBORATED,
    "broker": VerificationLevel.SINGLE_SOURCE,
    "smart_parser": VerificationLevel.SINGLE_SOURCE,
}

# High-value discovery types that require CORROBORATED+ to get full reward.
_HIGH_VALUE_TYPES: FrozenSet[str] = frozenset({
    "credential", "password", "shell", "root_shell",
    "flag", "user_flag", "root_flag", "hash",
    "pcap_credential", "gpp_password", "token",
})

# Discovery types that are inherently reliable (port scans are deterministic).
_INHERENTLY_RELIABLE: FrozenSet[str] = frozenset({
    "open_port", "service",
})


@dataclass
class TrustResult:
    """Result of trust evaluation for a set of discoveries."""
    original_bonus: float
    trusted_bonus: float
    discoveries_downgraded: int = 0
    spike_capped: bool = False
    details: List[str] = field(default_factory=list)


class DiscoveryTrustEngine:
    """
    Evaluates discovery trust and applies reward scaling.

    Usage:
        engine = DiscoveryTrustEngine()
        trust = engine.evaluate(discoveries, source_stage, existing_discoveries)
        adjusted_bonus = trust.trusted_bonus
    """

    def __init__(self, max_step_bonus: float = MAX_STEP_DISCOVERY_BONUS) -> None:
        self._max_step_bonus = max_step_bonus
        # Track seen discoveries for cross-step corroboration
        self._seen: Dict[str, Set[str]] = {}  # type -> set of values
        self._corroboration_sources: Dict[str, Set[str]] = {}  # key -> set of source commands
        logger.debug("DiscoveryTrustEngine initialised (max_step_bonus=%.1f)", max_step_bonus)

    def reset(self) -> None:
        """Reset trust state for new episode."""
        self._seen.clear()
        self._corroboration_sources.clear()

    def assign_verification_level(
        self,
        discovery_type: str,
        value: Any,
        source_stage: str = "regex",
        command: str = "",
    ) -> VerificationLevel:
        """
        Assign a verification level to a single discovery.

        Args:
            discovery_type: Discovery type string (e.g. "open_port", "credential").
            value: The discovery value.
            source_stage: Which parser stage produced this.
            command: The command that produced the output.

        Returns:
            VerificationLevel for this discovery.
        """
        # Base level from source stage
        level = _STAGE_TO_LEVEL.get(source_stage, VerificationLevel.UNVERIFIED)

        # Inherently reliable types get at least SINGLE_SOURCE
        if discovery_type in _INHERENTLY_RELIABLE and level == VerificationLevel.UNVERIFIED:
            level = VerificationLevel.SINGLE_SOURCE

        # Check for cross-step corroboration
        key = f"{discovery_type}:{value}"
        if key in self._corroboration_sources:
            prev_sources = self._corroboration_sources[key]
            if command and command not in prev_sources:
                # Seen from a different command → confirmed
                level = VerificationLevel.CONFIRMED
                prev_sources.add(command)
            elif len(prev_sources) >= 2:
                level = VerificationLevel.CONFIRMED
        else:
            self._corroboration_sources[key] = {command} if command else set()

        # Track for future corroboration
        self._seen.setdefault(discovery_type, set()).add(str(value))

        return level

    def evaluate(
        self,
        discoveries: Dict[str, Any],
        source_stage: str = "regex",
        command: str = "",
        bonus_table: Optional[Dict[str, float]] = None,
    ) -> TrustResult:
        """
        Evaluate trust for a set of discoveries and compute adjusted bonus.

        Args:
            discoveries: Flat discovery dict from parser.
            source_stage: Parser stage that produced these.
            command: The command that produced the output.
            bonus_table: Reward bonus table (DISCOVERY_BONUSES).

        Returns:
            TrustResult with original and adjusted bonuses.
        """
        if not bonus_table:
            bonus_table = {}

        original_bonus = 0.0
        trusted_bonus = 0.0
        downgraded = 0
        details: List[str] = []

        for disc_type, values in discoveries.items():
            base_bonus = bonus_table.get(disc_type, 0.0)
            if base_bonus <= 0:
                continue

            items = values if isinstance(values, list) else [values]
            for val in items:
                if isinstance(val, bool) and not val:
                    continue

                level = self.assign_verification_level(
                    disc_type, val, source_stage, command,
                )
                trust = level.trust_score
                original_bonus += base_bonus
                weighted = base_bonus * trust

                # High-value types below CORROBORATED get extra penalty
                if disc_type in _HIGH_VALUE_TYPES and level.value in (
                    "unverified", "single_source",
                ):
                    weighted *= 0.5
                    downgraded += 1
                    details.append(
                        f"{disc_type}:{val} downgraded ({level.value}, trust={trust:.1f})"
                    )

                trusted_bonus += weighted

        # Spike guard: cap per-step discovery bonus
        spike_capped = trusted_bonus > self._max_step_bonus
        if spike_capped:
            details.append(
                f"spike_guard: capped {trusted_bonus:.1f} → {self._max_step_bonus:.1f}"
            )
            trusted_bonus = self._max_step_bonus

        return TrustResult(
            original_bonus=original_bonus,
            trusted_bonus=trusted_bonus,
            discoveries_downgraded=downgraded,
            spike_capped=spike_capped,
            details=details,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Return trust engine statistics."""
        return {
            "total_tracked": sum(len(v) for v in self._seen.values()),
            "types_seen": list(self._seen.keys()),
            "corroboration_keys": len(self._corroboration_sources),
        }
