#!/usr/bin/env python3
"""
core/memory/working_memory.py — Phase 15.0: Working Memory (Bounded Scratchpad)

Short-term bounded memory for the current episode with TTL-based eviction.
Max 8 slots, each with TTL of 8 steps. Provides a 32-dim vector for state
injection and a bounded prompt fragment for cortex-level reasoning.

Feature-flag gated: FF_WORKING_MEMORY.

Author: Phase 15.0 Contract
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ariaska.memory.working_memory")

_MAX_SLOTS = 8
_DEFAULT_TTL = 8  # steps
_VECTOR_DIM = 32
_PROMPT_MAX_CHARS = 256


# ── Schemas ─────────────────────────────────────────────────────────────────

@dataclass
class WorkingMemorySlot:
    """A single slot in working memory."""
    key: str = ""                    # slot identifier
    content: str = ""                # bounded textual content (max 256 chars)
    slot_type: str = ""              # "hypothesis", "subgoal", "failures", "evidence"
    created_step: int = 0
    ttl_steps: int = _DEFAULT_TTL
    priority: float = 0.5           # [0,1] for eviction ranking
    numeric_features: List[float] = field(default_factory=list)  # up to 8 floats

    def is_expired(self, current_step: int) -> bool:
        return current_step - self.created_step >= self.ttl_steps

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def content_hash(self) -> str:
        return hashlib.sha256(
            f"{self.key}:{self.content}".encode("utf-8", errors="replace")
        ).hexdigest()[:8]


# ── Working Memory ──────────────────────────────────────────────────────────

class WorkingMemory:
    """
    Bounded working memory scratchpad.

    - Max 8 slots
    - TTL-based eviction (default 8 steps)
    - Provides 32-dim vector for state injection
    - Provides bounded prompt fragment
    """

    def __init__(
        self,
        max_slots: int = _MAX_SLOTS,
        default_ttl: int = _DEFAULT_TTL,
    ) -> None:
        self._max_slots = max(1, min(max_slots, 16))  # hard cap at 16
        self._default_ttl = default_ttl
        self._slots: Dict[str, WorkingMemorySlot] = {}
        self._current_step = 0

    def push(
        self,
        key: str,
        content: str,
        slot_type: str = "",
        priority: float = 0.5,
        ttl_steps: Optional[int] = None,
        numeric_features: Optional[List[float]] = None,
    ) -> bool:
        """
        Push or update a slot. Returns True if added/updated.

        Content is bounded to 256 chars.
        """
        content = content[:_PROMPT_MAX_CHARS]
        ttl = ttl_steps if ttl_steps is not None else self._default_ttl

        slot = WorkingMemorySlot(
            key=key,
            content=content,
            slot_type=slot_type,
            created_step=self._current_step,
            ttl_steps=ttl,
            priority=max(0.0, min(1.0, priority)),
            numeric_features=(numeric_features or [])[:8],
        )

        # If key exists, update in-place
        if key in self._slots:
            self._slots[key] = slot
            return True

        # If at capacity, evict lowest-priority expired slot first
        if len(self._slots) >= self._max_slots:
            self._evict_one()

        if len(self._slots) >= self._max_slots:
            # Still at capacity — evict lowest priority
            self._evict_lowest_priority()

        if len(self._slots) < self._max_slots:
            self._slots[key] = slot
            return True

        return False

    def step(self, current_step: int) -> int:
        """
        Advance step counter and evict expired slots.
        Returns number of evicted slots.
        """
        self._current_step = current_step
        expired_keys = [
            k for k, s in self._slots.items()
            if s.is_expired(current_step)
        ]
        for k in expired_keys:
            del self._slots[k]
        return len(expired_keys)

    def get(self, key: str) -> Optional[WorkingMemorySlot]:
        """Get a slot by key, or None if not present or expired."""
        slot = self._slots.get(key)
        if slot and slot.is_expired(self._current_step):
            del self._slots[key]
            return None
        return slot

    def to_vector(self, dim: int = _VECTOR_DIM) -> List[float]:
        """
        Produce a fixed-size vector representation.

        Layout (32 dims):
          [0]: num_active_slots / max_slots
          [1]: mean priority
          [2-5]: slot type counts (hypothesis, subgoal, failures, evidence)
          [6-7]: min/max TTL remaining
          [8-31]: concatenated numeric_features from top-priority slots
        """
        vec = [0.0] * dim
        active = [s for s in self._slots.values() if not s.is_expired(self._current_step)]

        if not active:
            return vec

        # [0] occupancy
        vec[0] = len(active) / self._max_slots

        # [1] mean priority
        vec[1] = sum(s.priority for s in active) / len(active)

        # [2-5] type counts (normalized by max_slots)
        type_map = {"hypothesis": 2, "subgoal": 3, "failures": 4, "evidence": 5}
        for s in active:
            idx = type_map.get(s.slot_type, -1)
            if 0 <= idx < dim:
                vec[idx] += 1.0 / self._max_slots

        # [6-7] TTL range
        ttls = [max(0, s.ttl_steps - (self._current_step - s.created_step)) for s in active]
        vec[6] = min(ttls) / self._default_ttl if ttls else 0.0
        vec[7] = max(ttls) / self._default_ttl if ttls else 0.0

        # [8-31] numeric features from top-priority slots
        sorted_slots = sorted(active, key=lambda s: s.priority, reverse=True)
        feat_idx = 8
        for s in sorted_slots:
            for f in s.numeric_features:
                if feat_idx < dim:
                    vec[feat_idx] = f
                    feat_idx += 1

        return vec

    def to_prompt_fragment(self, max_chars: int = _PROMPT_MAX_CHARS) -> str:
        """
        Produce a bounded text summary for prompt injection.
        Sorted by priority descending.
        """
        active = [s for s in self._slots.values() if not s.is_expired(self._current_step)]
        active.sort(key=lambda s: s.priority, reverse=True)

        lines: List[str] = []
        total = 0
        for s in active:
            line = f"[{s.slot_type}] {s.content[:60]}"
            if total + len(line) + 1 > max_chars:
                break
            lines.append(line)
            total += len(line) + 1
        return "\n".join(lines)

    def _evict_one(self) -> None:
        """Evict one expired slot if any."""
        for k, s in list(self._slots.items()):
            if s.is_expired(self._current_step):
                del self._slots[k]
                return

    def _evict_lowest_priority(self) -> None:
        """Evict the slot with lowest priority."""
        if not self._slots:
            return
        worst_key = min(self._slots, key=lambda k: self._slots[k].priority)
        del self._slots[worst_key]

    def clear(self) -> None:
        self._slots.clear()
        self._current_step = 0

    def __len__(self) -> int:
        return len(self._slots)

    def get_stats(self) -> Dict[str, Any]:
        active = [s for s in self._slots.values() if not s.is_expired(self._current_step)]
        return {
            "active_slots": len(active),
            "max_slots": self._max_slots,
            "current_step": self._current_step,
        }
