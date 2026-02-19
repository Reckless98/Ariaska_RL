"""core/config/phase_weights.py — Phase 41: Configurable phase readiness weights.

Per-target-profile weight overrides for phase transition scoring.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger("ariaska.config.phase_weights")


@dataclass
class PhaseWeights:
    """Per-phase readiness weight multipliers."""
    recon: float = 1.0
    enumeration: float = 1.2
    exploitation: float = 1.5
    privilege_escalation: float = 2.0
    lateral_movement: float = 1.3
    post_exploitation: float = 1.8
    exfiltration: float = 2.5
    closeout: float = 1.0

    def get(self, phase_name: str) -> float:
        """Look up weight by phase name (case-insensitive)."""
        key = phase_name.lower().replace(" ", "_").replace("-", "_")
        return getattr(self, key, 1.0)


# ── Per-profile presets ────────────────────────────────────────
PROFILE_WEIGHTS: Dict[str, PhaseWeights] = {
    "default": PhaseWeights(),
    "metasploitable2": PhaseWeights(
        recon=0.8, enumeration=1.0, exploitation=2.0,
        privilege_escalation=1.5,
    ),
    "metasploitable3": PhaseWeights(
        enumeration=1.5, exploitation=1.8,
        privilege_escalation=2.5,
    ),
    "htb_easy": PhaseWeights(
        recon=0.8, enumeration=1.0, exploitation=1.8,
        privilege_escalation=1.5, exfiltration=2.0,
    ),
    "htb_medium": PhaseWeights(
        recon=1.0, enumeration=1.3, exploitation=2.0,
        privilege_escalation=2.5, lateral_movement=1.8,
        exfiltration=3.0,
    ),
    "htb_hard": PhaseWeights(
        recon=1.2, enumeration=1.5, exploitation=2.5,
        privilege_escalation=3.0, lateral_movement=2.0,
        post_exploitation=2.5, exfiltration=3.5,
    ),
}


def get_weights(profile: Optional[str] = None) -> PhaseWeights:
    """Return phase weights for the given profile.

    Checks ``ARIASKA_TARGET_PROFILE`` env var if *profile* is None.
    Falls back to ``"default"`` for unknown profiles.
    """
    if profile is None:
        profile = os.environ.get("ARIASKA_TARGET_PROFILE", "default")
    result = PROFILE_WEIGHTS.get(profile, PROFILE_WEIGHTS["default"])
    if profile not in PROFILE_WEIGHTS:
        logger.warning("Unknown profile %r, using default weights", profile)
    return result
