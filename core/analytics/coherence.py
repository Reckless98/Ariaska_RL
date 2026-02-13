"""
Coherence metric for Ariaska decision quality assessment.

R66: Computes a rolling coherence score [0,1] from multiple signals:
  - Anti-repeat rate (lower is better)
  - Discovery rate (higher is better)
  - Phase progression velocity (higher is better)
  - Command success ratio (higher is better)
  - DDQN macro confidence (higher is better)

Used for:
  - PPO entropy gating (low coherence → more exploration)
  - Codex trigger decisions (low coherence → strategic intervention)
  - Live HUD display

Usage:
    from core.analytics.coherence import CoherenceTracker
    ct = CoherenceTracker()
    ct.record_step(source="ppo", had_discovery=True, phase_ord=3, success=True, macro_conf=0.7)
    score = ct.coherence  # 0.0 .. 1.0
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List

logger = logging.getLogger("ariaska.analytics.coherence")


@dataclass
class CoherenceTracker:
    """Rolling coherence metric from recent step signals.
    
    Window of last N steps. Each component scored [0,1], then weighted average.
    """
    window_size: int = 10

    # Internal buffers
    _sources: Deque[str] = field(default_factory=lambda: deque(maxlen=10))
    _discoveries: Deque[bool] = field(default_factory=lambda: deque(maxlen=10))
    _phase_ords: Deque[int] = field(default_factory=lambda: deque(maxlen=10))
    _successes: Deque[bool] = field(default_factory=lambda: deque(maxlen=10))
    _macro_confs: Deque[float] = field(default_factory=lambda: deque(maxlen=10))

    def __post_init__(self):
        # Re-create deques with proper maxlen
        self._sources = deque(maxlen=self.window_size)
        self._discoveries = deque(maxlen=self.window_size)
        self._phase_ords = deque(maxlen=self.window_size)
        self._successes = deque(maxlen=self.window_size)
        self._macro_confs = deque(maxlen=self.window_size)

    def record_step(
        self,
        source: str = "ppo",
        had_discovery: bool = False,
        phase_ord: int = 0,
        success: bool = True,
        macro_conf: float = 0.5,
    ) -> None:
        """Record a step's signals into the rolling window."""
        self._sources.append(source)
        self._discoveries.append(had_discovery)
        self._phase_ords.append(phase_ord)
        self._successes.append(success)
        self._macro_confs.append(macro_conf)

    @property
    def coherence(self) -> float:
        """Compute coherence score [0, 1]. Higher = more coherent execution.
        
        Components (weighted):
          - anti_repeat_score (0.25): 1 - (anti_repeat_count / window)
          - discovery_score (0.20): discovery_count / window
          - phase_velocity (0.15): max_phase_ord change over window
          - success_score (0.20): success_count / window
          - macro_conf_score (0.20): avg macro confidence
        """
        n = len(self._sources)
        if n == 0:
            return 0.5  # Neutral before any data

        # Anti-repeat score: fewer anti_repeat → higher score
        ar_count = sum(1 for s in self._sources if s == "anti_repeat")
        ar_score = 1.0 - (ar_count / n)

        # Discovery score: more discoveries → higher score
        disc_score = sum(1 for d in self._discoveries if d) / n

        # Phase velocity: ordinal range across window
        if len(self._phase_ords) >= 2:
            phase_range = max(self._phase_ords) - min(self._phase_ords)
            phase_score = min(1.0, phase_range / 3.0)  # 3 phase jumps = perfect
        else:
            phase_score = 0.0

        # Success score
        succ_score = sum(1 for s in self._successes if s) / n

        # Macro confidence score
        macro_score = sum(self._macro_confs) / n if self._macro_confs else 0.5

        # Weighted average
        coherence = (
            0.25 * ar_score
            + 0.20 * disc_score
            + 0.15 * phase_score
            + 0.20 * succ_score
            + 0.20 * macro_score
        )
        return max(0.0, min(1.0, coherence))

    @property
    def anti_repeat_rate(self) -> float:
        """Fraction of recent steps that were anti_repeat."""
        n = len(self._sources)
        if n == 0:
            return 0.0
        return sum(1 for s in self._sources if s == "anti_repeat") / n

    @property
    def is_collapsing(self) -> bool:
        """True if coherence has dropped critically (codex trigger)."""
        return self.coherence < 0.30

    @property
    def is_stagnating(self) -> bool:
        """True if no discoveries and low phase velocity."""
        n = len(self._discoveries)
        if n < 3:
            return False
        recent_disc = sum(1 for d in list(self._discoveries)[-5:] if d)
        return recent_disc == 0 and self.coherence < 0.45

    def reset_episode(self) -> None:
        """Clear all buffers for new episode."""
        self._sources.clear()
        self._discoveries.clear()
        self._phase_ords.clear()
        self._successes.clear()
        self._macro_confs.clear()
