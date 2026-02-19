"""Tests for B6: Episodic Memory."""
from __future__ import annotations

import os
import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


@pytest.mark.skipif(not _HAS_TORCH, reason="torch required")
class TestEpisodicMemory:
    def test_import(self):
        from core.algorithms.episodic_memory import EpisodicMemory
        em = EpisodicMemory()
        assert em is not None
        assert em.size == 0

    def test_store_and_retrieve(self):
        from core.algorithms.episodic_memory import EpisodicMemory, EpisodicMemoryConfig
        em = EpisodicMemory(config=EpisodicMemoryConfig(capacity=100, k_neighbors=3))
        state = torch.randn(512)
        em.store(state, action_idx=1, reward=5.0, phase="RECON")
        assert em.size == 1
        results = em.retrieve(state, k=1)
        assert len(results) == 1
        assert results[0].phase == "RECON"

    def test_capacity_eviction(self):
        from core.algorithms.episodic_memory import EpisodicMemory, EpisodicMemoryConfig
        em = EpisodicMemory(config=EpisodicMemoryConfig(capacity=5))
        for i in range(10):
            em.store(torch.randn(512), action_idx=i, reward=float(i))
        assert em.size == 5

    def test_empty_retrieve(self):
        from core.algorithms.episodic_memory import EpisodicMemory
        em = EpisodicMemory()
        results = em.retrieve(torch.randn(512))
        assert results == []

    def test_format_for_injection(self):
        from core.algorithms.episodic_memory import EpisodicMemory
        em = EpisodicMemory()
        em.store(torch.randn(512), 0, 1.0)
        em.store(torch.randn(512), 1, 2.0)
        entries = em.retrieve(torch.randn(512), k=2)
        tensor = em.format_for_injection(entries)
        assert tensor is not None
        assert tensor.shape[0] == 2

    def test_clear(self):
        from core.algorithms.episodic_memory import EpisodicMemory
        em = EpisodicMemory()
        em.store(torch.randn(512), 0, 1.0)
        em.clear()
        assert em.size == 0
