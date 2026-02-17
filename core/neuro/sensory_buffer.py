#!/usr/bin/env python3
"""
core/neuro/sensory_buffer.py — Phase 15.0: Sensory Buffer (PNS layer)

Ring buffer of recent sensory observations from parsed command output.
Bounded to K=16 entries, dedup by signature hash with TTL=60s.
Provides a bounded summary string for cortex-level prompting.

Feature-flag gated: FF_SENSORY_BUFFER.

Author: Phase 15.0 Contract
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger("ariaska.neuro.sensory_buffer")

# ── Constants ───────────────────────────────────────────────────────────────

_DEFAULT_CAPACITY = 16
_DEFAULT_TTL = 60.0  # seconds
_SUMMARY_MAX_CHARS = 512


# ── Schemas ─────────────────────────────────────────────────────────────────

@dataclass
class SensoryBufferEntry:
    """A single sensory observation from command output parsing."""
    timestamp: float = field(default_factory=time.time)
    command: str = ""
    output_signature: str = ""      # hash of salient output features
    discoveries: List[str] = field(default_factory=list)
    confidence: float = 0.0
    phase: str = ""
    learning_features_dim: int = 0  # number of nonzero LearningFeatures dims
    evidence_delta: int = 0         # new evidence graph nodes

    def signature_hash(self) -> str:
        """Hash for dedup: command + output_signature."""
        data = f"{self.command}:{self.output_signature}".encode("utf-8", errors="replace")
        return hashlib.sha256(data).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary_line(self) -> str:
        """Single-line summary, bounded to 128 chars."""
        disc = ",".join(self.discoveries[:4])
        line = f"[{self.phase}] {self.command[:40]} → {disc} (conf={self.confidence:.2f})"
        return line[:128]


# ── Buffer ──────────────────────────────────────────────────────────────────

class SensoryBuffer:
    """
    Ring buffer for recent sensory observations.

    - Capacity: K entries (default 16)
    - Dedup: entries with same signature_hash within TTL seconds are merged
    - Bounded summary: max 512 chars for prompt injection
    """

    def __init__(
        self,
        capacity: int = _DEFAULT_CAPACITY,
        ttl: float = _DEFAULT_TTL,
    ) -> None:
        self._capacity = max(1, capacity)
        self._ttl = ttl
        self._buffer: Deque[SensoryBufferEntry] = deque(maxlen=self._capacity)
        self._seen_hashes: Dict[str, float] = {}  # hash → timestamp
        self._total_pushed = 0
        self._total_deduped = 0

    def push(self, entry: SensoryBufferEntry) -> bool:
        """
        Push a new entry. Returns True if added, False if deduped.
        """
        sig = entry.signature_hash()
        now = entry.timestamp or time.time()

        # Dedup check
        if sig in self._seen_hashes:
            last_seen = self._seen_hashes[sig]
            if now - last_seen < self._ttl:
                self._total_deduped += 1
                return False

        # Evict expired hashes periodically
        if len(self._seen_hashes) > self._capacity * 4:
            self._evict_expired(now)

        self._seen_hashes[sig] = now
        self._buffer.append(entry)
        self._total_pushed += 1
        return True

    def _evict_expired(self, now: float) -> None:
        """Remove expired signature hashes."""
        expired = [k for k, ts in self._seen_hashes.items() if now - ts > self._ttl]
        for k in expired:
            del self._seen_hashes[k]

    def get_summary(self, max_chars: int = _SUMMARY_MAX_CHARS) -> str:
        """
        Return a bounded summary string of recent observations.
        Suitable for prompt injection.
        """
        lines: List[str] = []
        total_len = 0
        # Iterate newest first
        for entry in reversed(self._buffer):
            line = entry.summary_line()
            if total_len + len(line) + 1 > max_chars:
                break
            lines.append(line)
            total_len += len(line) + 1
        lines.reverse()
        return "\n".join(lines)

    def get_recent(self, n: int = 4) -> List[SensoryBufferEntry]:
        """Return the N most recent entries (copy)."""
        entries = list(self._buffer)
        return entries[-n:]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "size": len(self._buffer),
            "capacity": self._capacity,
            "total_pushed": self._total_pushed,
            "total_deduped": self._total_deduped,
        }

    def clear(self) -> None:
        self._buffer.clear()
        self._seen_hashes.clear()
        self._total_pushed = 0
        self._total_deduped = 0

    def __len__(self) -> int:
        return len(self._buffer)
