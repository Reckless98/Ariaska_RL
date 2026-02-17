#!/usr/bin/env python3
"""
core/llm/call_cache.py — Phase 14.0: LLM Call Cache

Simple hash-based cache for LLM responses to avoid duplicate calls.
Bounded capacity with LRU eviction. Thread-safe.

Author: Phase 14.0 Contract T1
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger("ariaska.llm_call_cache")


@dataclass
class CacheEntry:
    """A cached LLM response."""
    prompt_hash: str
    response: str
    tokens_used: int = 0
    model: str = ""
    hits: int = 0
    roi_tag: str = ""            # Phase 15: ROI tag for budget tracking
    prompt_version: str = ""     # Phase 15: prompt version for cache invalidation
    cross_episode: bool = False  # Phase 15: allow cross-episode caching


class LLMCallCache:
    """
    Bounded LRU cache for LLM responses.

    Keyed by prompt hash (SHA-256 of prompt text + model).
    Thread-safe via lock.
    """

    def __init__(self, capacity: int = 500) -> None:
        self._capacity = capacity
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._total_hits = 0
        self._total_misses = 0

    def _hash_key(self, prompt: str, model: str = "", prompt_version: str = "") -> str:
        """Create hash key from prompt, model, and prompt_version."""
        data = f"{model}:{prompt_version}:{prompt}".encode("utf-8", errors="replace")
        return hashlib.sha256(data).hexdigest()[:16]

    def get(self, prompt: str, model: str = "", prompt_version: str = "") -> Optional[str]:
        """Look up cached response. Returns None on miss."""
        key = self._hash_key(prompt, model, prompt_version)
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                entry.hits += 1
                self._total_hits += 1
                # Move to end (LRU refresh)
                self._cache.move_to_end(key)
                return entry.response
            self._total_misses += 1
            return None

    def put(
        self, prompt: str, response: str,
        model: str = "", tokens_used: int = 0,
        roi_tag: str = "", prompt_version: str = "",
        cross_episode: bool = False,
    ) -> None:
        """Store a response in cache."""
        key = self._hash_key(prompt, model, prompt_version)
        with self._lock:
            if key in self._cache:
                # Update existing
                self._cache[key].response = response
                self._cache[key].tokens_used = tokens_used
                self._cache.move_to_end(key)
            else:
                # Add new
                self._cache[key] = CacheEntry(
                    prompt_hash=key,
                    response=response,
                    tokens_used=tokens_used,
                    model=model,
                    roi_tag=roi_tag,
                    prompt_version=prompt_version,
                    cross_episode=cross_episode,
                )
                # Evict LRU if over capacity
                while len(self._cache) > self._capacity:
                    self._cache.popitem(last=False)

    def get_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        with self._lock:
            return {
                "size": len(self._cache),
                "capacity": self._capacity,
                "total_hits": self._total_hits,
                "total_misses": self._total_misses,
                "hit_rate": (
                    self._total_hits / max(1, self._total_hits + self._total_misses)
                ),
            }

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._total_hits = 0
            self._total_misses = 0

    def clear_episode(self) -> None:
        """Clear only per-episode entries, keep cross-episode ones."""
        with self._lock:
            cross_episode_keys = [
                k for k, v in self._cache.items() if v.cross_episode
            ]
            new_cache: OrderedDict[str, CacheEntry] = OrderedDict()
            for k in cross_episode_keys:
                new_cache[k] = self._cache[k]
            self._cache = new_cache

    def get_roi_stats(self) -> Dict[str, int]:
        """Return count of cached entries per ROI tag."""
        with self._lock:
            roi: Dict[str, int] = {}
            for entry in self._cache.values():
                tag = entry.roi_tag or "untagged"
                roi[tag] = roi.get(tag, 0) + 1
            return roi
