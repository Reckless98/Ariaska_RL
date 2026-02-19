"""
core/llm/cot_cache.py — Phase 41: Chain-of-Thought Reasoning Cache

Caches LLM reasoning chains indexed by (phase, state_fingerprint) so
repeated queries in similar states can reuse prior reasoning without
burning token budget.

Features:
  - State fingerprint hashing for cache keys
  - TTL-based expiration (configurable, default 300s)
  - Max entries cap to bound memory
  - Phase-aware: different phases get separate cache partitions
  - Thread-safe with read-write lock pattern
  - Hit/miss statistics for budget monitoring
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ariaska.cot_cache")


def _env_bool(key: str, default: bool = False) -> bool:
    import os
    return os.environ.get(key, str(default)).lower() in ("1", "true", "yes")


@dataclass
class CoTCacheConfig:
    """Configuration for the Chain-of-Thought cache."""
    max_entries: int = 512
    ttl_seconds: float = 300.0  # 5 minutes default
    enabled: bool = field(default_factory=lambda: _env_bool("FF_COT_CACHE", True))
    phase_partitioned: bool = True  # Separate caches per phase


@dataclass
class CoTEntry:
    """A cached reasoning chain entry."""
    key: str
    phase: str
    reasoning_chain: str
    command_suggestion: Optional[str] = None
    confidence: float = 0.0
    model_used: Optional[str] = None
    token_cost: int = 0
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)

    @property
    def is_expired(self) -> bool:
        """Check if this entry has exceeded its TTL."""
        # TTL is checked externally by the cache; this is a convenience
        return False  # Managed by CoTCache

    def touch(self) -> None:
        """Update access timestamp and count."""
        self.access_count += 1
        self.last_accessed = time.time()


class CoTCache:
    """
    Thread-safe LRU cache for Chain-of-Thought reasoning chains.

    Keys are computed from (phase, state_fingerprint) pairs.  When a
    cache hit occurs, the original reasoning is returned without making
    an LLM call, saving token budget.
    """

    def __init__(self, config: Optional[CoTCacheConfig] = None) -> None:
        self._config = config or CoTCacheConfig()
        self._cache: OrderedDict[str, CoTEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expired": 0,
            "tokens_saved": 0,
        }
        logger.info(
            f"[COT-CACHE] Initialized (max={self._config.max_entries}, "
            f"ttl={self._config.ttl_seconds}s)"
        )

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    @staticmethod
    def compute_fingerprint(
        phase: str,
        state_dict: Dict[str, Any],
        extra_keys: Optional[List[str]] = None,
    ) -> str:
        """
        Compute a deterministic fingerprint from phase + state.

        Only selected state keys are used to avoid over-specificity.
        """
        relevant_keys = [
            "ports_discovered", "services_discovered", "shell_obtained",
            "credentials_found", "detection_risk", "phase",
        ]
        if extra_keys:
            relevant_keys.extend(extra_keys)

        parts = [f"phase={phase}"]
        for k in sorted(relevant_keys):
            if k in state_dict:
                val = state_dict[k]
                if isinstance(val, (set, frozenset)):
                    val = sorted(val)
                parts.append(f"{k}={val}")

        raw = "|".join(parts)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _make_key(self, phase: str, fingerprint: str) -> str:
        if self._config.phase_partitioned:
            return f"{phase}:{fingerprint}"
        return fingerprint

    def get(
        self, phase: str, fingerprint: str
    ) -> Optional[CoTEntry]:
        """Look up cached reasoning.  Returns None on miss or expiry."""
        if not self._config.enabled:
            return None
        key = self._make_key(phase, fingerprint)
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._stats["misses"] += 1
                return None
            # Check TTL
            age = time.time() - entry.created_at
            if age > self._config.ttl_seconds:
                del self._cache[key]
                self._stats["expired"] += 1
                self._stats["misses"] += 1
                return None
            # LRU: move to end
            self._cache.move_to_end(key)
            entry.touch()
            self._stats["hits"] += 1
            self._stats["tokens_saved"] += entry.token_cost
            return entry

    def put(
        self,
        phase: str,
        fingerprint: str,
        reasoning_chain: str,
        command_suggestion: Optional[str] = None,
        confidence: float = 0.0,
        model_used: Optional[str] = None,
        token_cost: int = 0,
    ) -> None:
        """Store a reasoning chain in the cache."""
        if not self._config.enabled:
            return
        key = self._make_key(phase, fingerprint)
        entry = CoTEntry(
            key=key,
            phase=phase,
            reasoning_chain=reasoning_chain,
            command_suggestion=command_suggestion,
            confidence=confidence,
            model_used=model_used,
            token_cost=token_cost,
        )
        with self._lock:
            if key in self._cache:
                # Update existing
                self._cache[key] = entry
                self._cache.move_to_end(key)
            else:
                # Evict if at capacity
                while len(self._cache) >= self._config.max_entries:
                    self._cache.popitem(last=False)  # Remove oldest
                    self._stats["evictions"] += 1
                self._cache[key] = entry

    def invalidate_phase(self, phase: str) -> int:
        """Remove all entries for a given phase.  Returns count removed."""
        if not self._config.phase_partitioned:
            return 0
        with self._lock:
            keys_to_remove = [
                k for k in self._cache if k.startswith(f"{phase}:")
            ]
            for k in keys_to_remove:
                del self._cache[k]
            return len(keys_to_remove)

    def clear(self) -> None:
        """Clear entire cache."""
        with self._lock:
            self._cache.clear()
        logger.info("[COT-CACHE] Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total if total > 0 else 0.0
            return {
                **self._stats,
                "size": len(self._cache),
                "capacity": self._config.max_entries,
                "hit_rate": round(hit_rate, 4),
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)
