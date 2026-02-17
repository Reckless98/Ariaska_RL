#!/usr/bin/env python3
"""
core/reasoning/teacher_trace.py — Phase 14.0: TeacherTrace + BCBuffer

First-class training artifacts for mentor→student distillation.

Golden Rule: Every mentor call MUST produce a TeacherTrace containing:
  - The teacher's decision (action, rationale, confidence)
  - The student's parallel prediction (what PPO would have chosen)
  - Divergence between teacher and student
  - Post-hoc marking of whether student would have succeeded

TeacherTraces are stored in:
  - BCBuffer (full, with state_vector) for BC loss training
  - EpisodeLog (full) for postmortem analysis
  - CampaignMemory (compact, no state_vector) for cross-episode learning

Author: Phase 14.0 Contract C1.1 + C1.2
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("ariaska.teacher_trace")

try:
    import torch
except ImportError:
    torch = None  # type: ignore


@dataclass
class TeacherTrace:
    """
    First-class training artifact produced by every mentor call.

    Contract C1.1: Every mentor call MUST produce this. Stored in
    BCBuffer + EpisodeLog + CampaignMemory (compact).
    """
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    state_id: str = ""                              # hash(state_vector) for dedup

    # State (stored in BCBuffer only, omitted in compact)
    state_vector: Optional[List[float]] = None      # 512-dim

    # Teacher decision
    teacher_action_idx: int = 0                     # CommandActionMapper index
    teacher_command: str = ""                        # Full command string
    teacher_template: str = ""                       # CommandTemplate name
    rationale: str = ""                              # WHY (≤512 chars)
    expected_obs: str = ""                           # Expected outcome (≤256 chars)
    alt_actions: List[str] = field(default_factory=list)  # Alternatives (max 3)
    confidence: float = 0.5                          # 0-1 mentor confidence
    uncertainty: float = 0.5                         # 0-1 teacher uncertainty

    # Student parallel prediction (Golden Rule)
    student_action_idx: int = -1                     # PPO's choice
    student_command: str = ""                        # Student's command string
    student_template: str = ""                       # Student's template name
    student_log_prob: float = 0.0                    # PPO log-probability
    student_confidence: float = 0.0                  # Entropy-based confidence
    divergence: float = 0.0                          # |teacher - student| or KL
    student_would_have_succeeded: Optional[bool] = None  # Set post-hoc

    # Context
    episode: int = 0
    step: int = 0
    agent_id: str = ""
    phase: str = ""
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        # Truncate string fields to contracted bounds
        if len(self.rationale) > 512:
            self.rationale = self.rationale[:512]
        if len(self.expected_obs) > 256:
            self.expected_obs = self.expected_obs[:256]
        if len(self.alt_actions) > 3:
            self.alt_actions = self.alt_actions[:3]

    def to_compact(self) -> Dict[str, Any]:
        """Compact form for CampaignMemory — omits state_vector."""
        d = asdict(self)
        d.pop("state_vector", None)
        return d

    def to_bc_sample(self, decay_factor: float = 1.0) -> Optional["BCSample"]:
        """Convert to BCSample for BCBuffer. Returns None if no state_vector."""
        if torch is None or self.state_vector is None:
            return None
        state_tensor = torch.tensor(self.state_vector, dtype=torch.float32)
        weight = self.confidence * decay_factor
        rationale_hash = hash(self.rationale) & 0xFFFFFFFF
        return BCSample(
            state=state_tensor,
            teacher_action=self.teacher_action_idx,
            weight=weight,
            rationale_hash=rationale_hash,
            episode=self.episode,
            step=self.step,
        )

    def compute_divergence(self) -> float:
        """Compute divergence between teacher and student predictions."""
        if self.teacher_action_idx < 0 or self.student_action_idx < 0:
            return 0.0
        # Simple action divergence: 1.0 if different, 0.0 if same
        self.divergence = 0.0 if self.teacher_action_idx == self.student_action_idx else 1.0
        return self.divergence

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TeacherTrace":
        """Reconstruct from dict. Backward-compatible."""
        valid_fields = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class BCSample:
    """Single BC training sample derived from TeacherTrace. Contract C1.2."""
    state: Any  # torch.Tensor (512,)
    teacher_action: int
    weight: float
    rationale_hash: int
    episode: int = 0
    step: int = 0


class BCBuffer:
    """
    Circular buffer for teacher demonstrations. Thread-safe.

    Contract C1.2:
      - Capacity: 2000
      - Dedup by (state_hash, action) pairs
      - Provides sample(batch_size) for BC loss
    """

    def __init__(self, capacity: int = 2000):
        self.capacity = capacity
        self._samples: Deque[BCSample] = deque(maxlen=capacity)
        self._dedup_hashes: Set[int] = set()
        self._lock = threading.Lock()
        self._stats = {
            "total_stored": 0,
            "dedup_skipped": 0,
            "overflows": 0,
        }

    def __getstate__(self) -> Dict[str, Any]:
        """Support deepcopy/pickle by excluding threading.Lock."""
        state = self.__dict__.copy()
        state.pop("_lock", None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore from pickle/deepcopy with fresh lock."""
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def store(self, trace: TeacherTrace, decay_factor: float = 1.0) -> bool:
        """
        Store a TeacherTrace as a BCSample. Returns True if stored.

        Dedup: identical (state_hash, action) pairs are not stored twice.
        """
        sample = trace.to_bc_sample(decay_factor=decay_factor)
        if sample is None:
            return False

        # Dedup key: hash of (state_id, teacher_action)
        dedup_key = hash((trace.state_id, trace.teacher_action_idx))

        with self._lock:
            if dedup_key in self._dedup_hashes:
                self._stats["dedup_skipped"] += 1
                return False

            was_full = len(self._samples) >= self.capacity
            self._samples.append(sample)
            self._dedup_hashes.add(dedup_key)

            # Clean dedup set if it grows too large (buffer overflow evictions)
            if was_full:
                self._stats["overflows"] += 1
                # Rebuild dedup set from current samples (expensive but rare)
                if len(self._dedup_hashes) > self.capacity * 2:
                    self._dedup_hashes.clear()

            self._stats["total_stored"] += 1
            return True

    def sample(self, batch_size: int) -> List[BCSample]:
        """Sample a batch of BCSamples for BC loss computation."""
        with self._lock:
            if len(self._samples) == 0:
                return []
            import random
            n = min(batch_size, len(self._samples))
            indices = random.sample(range(len(self._samples)), n)
            return [self._samples[i] for i in indices]

    def __len__(self) -> int:
        with self._lock:
            return len(self._samples)

    def clear(self) -> None:
        """Clear all samples and dedup hashes."""
        with self._lock:
            self._samples.clear()
            self._dedup_hashes.clear()
            self._stats = {"total_stored": 0, "dedup_skipped": 0, "overflows": 0}

    def get_stats(self) -> Dict[str, Any]:
        """Return buffer statistics."""
        with self._lock:
            unique_states = len({s.rationale_hash for s in self._samples})
            avg_weight = (
                sum(s.weight for s in self._samples) / len(self._samples)
                if self._samples else 0.0
            )
            return {
                "size": len(self._samples),
                "capacity": self.capacity,
                "unique_rationale_hashes": unique_states,
                "avg_weight": avg_weight,
                **self._stats,
            }
