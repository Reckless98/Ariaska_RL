"""
core/training/dagger.py — Phase 41: DAgger (Dataset Aggregation) Wiring

Bridges mentor demonstrations into the RL training loop via the DAgger
algorithm: at each step where the mentor is consulted, the mentor's
preferred action is stored alongside the policy's own choice.  The
resulting dataset is periodically mixed into the BC loss to keep the
policy grounded in expert knowledge while still learning from its own
experience.

Integration points:
  - ApprenticeCoach.decide() produces MentorTrace
  - DAgger stores (state, mentor_action, policy_action) triples
  - PPO BC loss samples from DAgger buffer (via BCBuffer compat)
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger("ariaska.dagger")


def _env_bool(key: str, default: bool = False) -> bool:
    return os.environ.get(key, str(default)).lower() in ("1", "true", "yes")


import os


@dataclass
class DAggerConfig:
    """Configuration for DAgger buffer and mixing."""
    capacity: int = 4000
    min_samples_for_train: int = 32
    mix_ratio: float = 0.5  # fraction of BC batch from DAgger vs BCBuffer
    decay_factor: float = 0.995  # exponential weight decay per episode
    dedup: bool = True
    enabled: bool = field(default_factory=lambda: _env_bool("FF_DAGGER", True))


@dataclass
class DAggerSample:
    """Single DAgger demonstration sample."""
    state_hash: str
    state_vector: List[float]
    mentor_action_idx: int
    mentor_command: str
    policy_action_idx: int
    policy_command: str
    mentor_confidence: float
    phase: str
    episode: int
    step: int
    weight: float = 1.0
    timestamp: float = field(default_factory=time.time)


class DAggerBuffer:
    """
    Thread-safe ring buffer for DAgger demonstration pairs.

    Stores (state, mentor_action, policy_action) triples collected
    during training.  Provides sampling compatible with the BCBuffer
    interface so PPO can blend DAgger samples into its BC loss.
    """

    def __init__(self, config: Optional[DAggerConfig] = None) -> None:
        self._config = config or DAggerConfig()
        self._buffer: Deque[DAggerSample] = deque(maxlen=self._config.capacity)
        self._lock = threading.Lock()
        self._seen_hashes: set = set()
        self._stats = {
            "stored": 0,
            "duplicates_skipped": 0,
            "sampled": 0,
        }
        logger.info(
            f"[DAGGER] Buffer initialized (capacity={self._config.capacity}, "
            f"mix_ratio={self._config.mix_ratio})"
        )

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def store(
        self,
        state_hash: str,
        state_vector: List[float],
        mentor_action_idx: int,
        mentor_command: str,
        policy_action_idx: int,
        policy_command: str,
        mentor_confidence: float,
        phase: str,
        episode: int,
        step: int,
    ) -> bool:
        """Store a DAgger sample.  Returns True if stored, False if dedup."""
        if not self._config.enabled:
            return False
        dedup_key = f"{state_hash}:{mentor_action_idx}"
        with self._lock:
            if self._config.dedup and dedup_key in self._seen_hashes:
                self._stats["duplicates_skipped"] += 1
                return False
            sample = DAggerSample(
                state_hash=state_hash,
                state_vector=state_vector,
                mentor_action_idx=mentor_action_idx,
                mentor_command=mentor_command,
                policy_action_idx=policy_action_idx,
                policy_command=policy_command,
                mentor_confidence=mentor_confidence,
                phase=phase,
                episode=episode,
                step=step,
            )
            self._buffer.append(sample)
            if self._config.dedup:
                self._seen_hashes.add(dedup_key)
                # Prune hash set if it grows too large
                if len(self._seen_hashes) > self._config.capacity * 2:
                    self._seen_hashes = {
                        f"{s.state_hash}:{s.mentor_action_idx}"
                        for s in self._buffer
                    }
            self._stats["stored"] += 1
            return True

    def sample(self, batch_size: int) -> List[DAggerSample]:
        """Sample a batch for BC training.  Higher-weight samples preferred."""
        import random
        with self._lock:
            if len(self._buffer) < self._config.min_samples_for_train:
                return []
            population = list(self._buffer)
        weights = [s.weight * s.mentor_confidence for s in population]
        total = sum(weights)
        if total <= 0:
            return []
        k = min(batch_size, len(population))
        chosen = random.choices(population, weights=weights, k=k)
        self._stats["sampled"] += k
        return chosen

    def can_train(self) -> bool:
        """Return True if enough samples for a training batch."""
        with self._lock:
            return len(self._buffer) >= self._config.min_samples_for_train

    def decay_weights(self) -> None:
        """Apply exponential decay to all sample weights (call per episode)."""
        with self._lock:
            for sample in self._buffer:
                sample.weight *= self._config.decay_factor

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                **self._stats,
                "buffer_size": len(self._buffer),
                "capacity": self._config.capacity,
                "enabled": self._config.enabled,
            }

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()
            self._seen_hashes.clear()
            for k in self._stats:
                self._stats[k] = 0
        logger.info("[DAGGER] Buffer cleared")


class DAggerMixin:
    """
    Mixin for SmartCoach / ApprenticeCoach to record DAgger samples.

    Usage:
        class MyCoach(DAggerMixin, ...):
            def __init__(self, ...):
                self.init_dagger()
            
            def on_mentor_decision(self, ...):
                self.record_dagger_sample(...)
    """

    def init_dagger(
        self, config: Optional[DAggerConfig] = None
    ) -> None:
        """Initialize DAgger buffer."""
        self._dagger_buffer = DAggerBuffer(config)

    def record_dagger_sample(
        self,
        state_hash: str,
        state_vector: List[float],
        mentor_action_idx: int,
        mentor_command: str,
        policy_action_idx: int,
        policy_command: str,
        mentor_confidence: float,
        phase: str,
        episode: int,
        step: int,
    ) -> bool:
        """Record a mentor-vs-policy sample into DAgger buffer."""
        if not hasattr(self, "_dagger_buffer"):
            return False
        return self._dagger_buffer.store(
            state_hash=state_hash,
            state_vector=state_vector,
            mentor_action_idx=mentor_action_idx,
            mentor_command=mentor_command,
            policy_action_idx=policy_action_idx,
            policy_command=policy_command,
            mentor_confidence=mentor_confidence,
            phase=phase,
            episode=episode,
            step=step,
        )

    def get_dagger_buffer(self) -> Optional[DAggerBuffer]:
        """Return DAgger buffer if initialized."""
        return getattr(self, "_dagger_buffer", None)
