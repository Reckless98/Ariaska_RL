"""Anti-Repeat Guard — prevents command repetition and provides alternatives.

Phase 41: Extracted from SmartCoach for modularity.
Phase 56: Centralized all anti-repeat thresholds into AntiRepeatConfig.
"""
from __future__ import annotations

import logging
from typing import Dict, Any, List, Set, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger("ariaska.coach.anti_repeat")


@dataclass
class StagnationTier:
    """Thresholds at a specific stagnation level."""
    min_stagnation: int
    exact: int
    prefix: int
    family: int


@dataclass
class AntiRepeatConfig:
    """Centralized configuration for ALL anti-repeat thresholds.

    Phase 56: Consolidates 7 threshold locations from SmartCoach into one
    config object. Each field documents which pipeline stage uses it.
    """

    # ── Main post-selection guard (L3853-3955) ──
    # Base thresholds (stagnation < first tier)
    exact_threshold: int = 3
    prefix_threshold: int = 5
    family_threshold: int = 8

    # Stagnation-aware tiers: raised thresholds when system is stuck
    stagnation_tiers: list = field(default_factory=lambda: [
        StagnationTier(min_stagnation=15, exact=6, prefix=8, family=12),
        StagnationTier(min_stagnation=25, exact=8, prefix=12, family=16),
    ])

    # ── PPO pre-selection masking (L7300, L7343) ──
    ppo_mask_max_repeats: int = 1       # Hard-mask commands used >= N times
    ppo_mask_prefix_max: int = 3        # Hard-mask prefixes used >= N times

    # ── PPO bypass revocation (L3907) ──
    ppo_bypass_template_max: int = 3    # Revoke PPO bypass after N template repeats

    # ── Mentor pre-check (L8211, L8218) ──
    mentor_prefix_max: int = 2          # Reject mentor if prefix used >= N in window
    mentor_prefix_window: int = 10      # Window size for mentor prefix check
    mentor_exact_window: int = 5        # Window size for mentor exact check

    # ── R67 logit bias (L7353-7359) ──
    logit_bias_per_use: float = 1.0     # Logit penalty per repeat
    logit_bias_max: float = 4.0         # Maximum logit penalty

    # ── Codex meta spike trigger (L2062) ──
    codex_antirepeat_hits: int = 4      # Trigger codex after N anti-repeat blocks

    # ── Graded penalties (L3933, L3946) ──
    exact_penalty_per_repeat: float = -2.0   # Per-occurrence penalty for exact repeats
    prefix_penalty_per_repeat: float = -1.0  # Per-occurrence penalty for prefix repeats
    prefix_penalty_min_count: int = 2        # Start penalizing prefix at this count

    # Legacy fields (kept for backward compatibility with AntiRepeatGuard)
    max_recent: int = 30
    prefix_window: int = 10
    exact_block: bool = True
    prefix_block: bool = True

    def get_thresholds(self, stagnation_steps: int) -> Tuple[int, int, int]:
        """Return (exact, prefix, family) thresholds for the given stagnation level.

        Iterates stagnation tiers in descending order of min_stagnation,
        returning the first match. Falls back to base thresholds.
        """
        for tier in sorted(self.stagnation_tiers,
                           key=lambda t: t.min_stagnation, reverse=True):
            if stagnation_steps >= tier.min_stagnation:
                return tier.exact, tier.prefix, tier.family
        return self.exact_threshold, self.prefix_threshold, self.family_threshold


class AntiRepeatGuard:
    """Tracks recent commands and prevents repetition."""

    def __init__(self, config: Optional[AntiRepeatConfig] = None) -> None:
        self.config = config or AntiRepeatConfig()
        self._recent_commands: deque = deque(maxlen=self.config.max_recent)
        self._step_commands: Set[str] = set()
        self._recent_prefixes: deque = deque(maxlen=self.config.prefix_window)
        self._repeat_count: int = 0
        self._block_count: int = 0

    def is_repeat(self, command: str) -> bool:
        """Check if a command is a repeat."""
        if not command:
            return False
        cmd = command.strip()
        if self.config.exact_block and cmd in self._step_commands:
            self._repeat_count += 1
            return True
        if self.config.exact_block and cmd in self._recent_commands:
            self._repeat_count += 1
            return True
        if self.config.prefix_block:
            prefix = self._extract_prefix(cmd)
            if list(self._recent_prefixes).count(prefix) >= 2:
                self._repeat_count += 1
                return True
        return False

    def record_command(self, command: str) -> None:
        """Record a command as used."""
        cmd = command.strip()
        self._recent_commands.append(cmd)
        self._step_commands.add(cmd)
        self._recent_prefixes.append(self._extract_prefix(cmd))

    def clear_step(self) -> None:
        """Clear per-step tracking."""
        self._step_commands.clear()

    def get_recent_tags(self, k: int = 15) -> Set[str]:
        """Get tags from recent command prefixes."""
        recent = list(self._recent_commands)[-k:]
        return {self._extract_prefix(c) for c in recent if c}

    def get_stats(self) -> Dict[str, int]:
        """Return anti-repeat statistics."""
        return {"repeat_count": self._repeat_count, "block_count": self._block_count, "recent_commands": len(self._recent_commands)}

    def reset(self) -> None:
        """Full reset."""
        self._recent_commands.clear()
        self._step_commands.clear()
        self._recent_prefixes.clear()
        self._repeat_count = 0
        self._block_count = 0

    @staticmethod
    def _extract_prefix(cmd: str) -> str:
        """Extract the tool prefix from a command string."""
        if not cmd:
            return ""
        parts = cmd.strip().split()
        if not parts:
            return ""
        return parts[0].split("/")[-1].lower()
