"""Tests for B5: Chain-of-Thought reasoning cache."""
from __future__ import annotations

import os
import time
import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


class TestCoTCacheConfig:
    def test_defaults(self):
        from core.llm.cot_cache import CoTCacheConfig
        cfg = CoTCacheConfig()
        assert cfg.max_entries == 512
        assert cfg.ttl_seconds == 300.0
        assert cfg.phase_partitioned is True


class TestFingerprint:
    def test_deterministic(self):
        from core.llm.cot_cache import CoTCache
        state = {"ports_discovered": True, "phase": "RECON"}
        fp1 = CoTCache.compute_fingerprint("RECON", state)
        fp2 = CoTCache.compute_fingerprint("RECON", state)
        assert fp1 == fp2
        assert len(fp1) == 16

    def test_different_phase(self):
        from core.llm.cot_cache import CoTCache
        state = {"ports_discovered": True}
        fp1 = CoTCache.compute_fingerprint("RECON", state)
        fp2 = CoTCache.compute_fingerprint("EXPLOIT", state)
        assert fp1 != fp2

    def test_different_state(self):
        from core.llm.cot_cache import CoTCache
        fp1 = CoTCache.compute_fingerprint("RECON", {"ports_discovered": True})
        fp2 = CoTCache.compute_fingerprint("RECON", {"ports_discovered": False})
        assert fp1 != fp2


class TestCoTCache:
    def _make_cache(self, **kwargs):
        from core.llm.cot_cache import CoTCache, CoTCacheConfig
        cfg = CoTCacheConfig(**kwargs)
        return CoTCache(cfg)

    def test_put_and_get(self):
        cache = self._make_cache()
        cache.put("RECON", "fp123", "Think step by step...", confidence=0.8)
        entry = cache.get("RECON", "fp123")
        assert entry is not None
        assert entry.reasoning_chain == "Think step by step..."
        assert entry.confidence == 0.8

    def test_miss(self):
        cache = self._make_cache()
        entry = cache.get("RECON", "nonexistent")
        assert entry is None

    def test_ttl_expiry(self):
        cache = self._make_cache(ttl_seconds=0.01)
        cache.put("RECON", "fp1", "reasoning")
        time.sleep(0.02)
        entry = cache.get("RECON", "fp1")
        assert entry is None
        stats = cache.get_stats()
        assert stats["expired"] >= 1

    def test_eviction(self):
        cache = self._make_cache(max_entries=2)
        cache.put("RECON", "fp1", "r1")
        cache.put("RECON", "fp2", "r2")
        cache.put("RECON", "fp3", "r3")  # Should evict fp1
        assert cache.get("RECON", "fp1") is None
        assert cache.get("RECON", "fp3") is not None
        assert len(cache) == 2

    def test_lru_ordering(self):
        cache = self._make_cache(max_entries=2)
        cache.put("RECON", "fp1", "r1")
        cache.put("RECON", "fp2", "r2")
        # Access fp1 to make it recently used
        cache.get("RECON", "fp1")
        # Add fp3, should evict fp2 (least recently used)
        cache.put("RECON", "fp3", "r3")
        assert cache.get("RECON", "fp1") is not None
        assert cache.get("RECON", "fp2") is None

    def test_invalidate_phase(self):
        cache = self._make_cache()
        cache.put("RECON", "fp1", "r1")
        cache.put("RECON", "fp2", "r2")
        cache.put("EXPLOIT", "fp3", "r3")
        removed = cache.invalidate_phase("RECON")
        assert removed == 2
        assert cache.get("EXPLOIT", "fp3") is not None

    def test_stats_tracking(self):
        cache = self._make_cache()
        cache.put("RECON", "fp1", "reasoning", token_cost=100)
        cache.get("RECON", "fp1")  # hit
        cache.get("RECON", "missing")  # miss
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["tokens_saved"] == 100
        assert stats["hit_rate"] == 0.5

    def test_disabled_cache(self):
        cache = self._make_cache(enabled=False)
        cache.put("RECON", "fp1", "reasoning")
        entry = cache.get("RECON", "fp1")
        assert entry is None
        assert len(cache) == 0

    def test_clear(self):
        cache = self._make_cache()
        cache.put("RECON", "fp1", "r1")
        cache.put("RECON", "fp2", "r2")
        cache.clear()
        assert len(cache) == 0

    def test_phase_partitioned_false(self):
        from core.llm.cot_cache import CoTCache, CoTCacheConfig
        cfg = CoTCacheConfig(phase_partitioned=False)
        cache = CoTCache(cfg)
        cache.put("RECON", "fp1", "reasoning")
        # Same fingerprint, different phase should still hit
        entry = cache.get("EXPLOIT", "fp1")
        assert entry is not None

    def test_access_count(self):
        cache = self._make_cache()
        cache.put("RECON", "fp1", "reasoning")
        cache.get("RECON", "fp1")
        cache.get("RECON", "fp1")
        entry = cache.get("RECON", "fp1")
        assert entry is not None
        assert entry.access_count == 3
