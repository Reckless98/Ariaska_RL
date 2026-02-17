#!/usr/bin/env python3
"""
tests/test_phase15_consolidation.py — Phase 15.0: Consolidation Engine tests

Tests ConsolidationSample priority, ConsolidationBatch bounding,
ConsolidationEngine build and run, dedup.
"""

import os
import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


class TestConsolidationSample:
    """ConsolidationSample schema tests."""

    def test_priority_score(self):
        from core.training.consolidation import ConsolidationSample
        # High DA + confirmed hypothesis → high priority
        s = ConsolidationSample(da_level=1.0, ach_level=0.8, hypothesis_confirmed=True)
        assert s.priority_score() > 0.5

    def test_low_priority(self):
        from core.training.consolidation import ConsolidationSample
        s = ConsolidationSample(da_level=0.5, ach_level=0.0, hypothesis_confirmed=False)
        assert s.priority_score() < 0.2

    def test_to_dict(self):
        from core.training.consolidation import ConsolidationSample
        s = ConsolidationSample(step=3, command="nmap", reward=5.0)
        d = s.to_dict()
        assert d["step"] == 3
        assert d["command"] == "nmap"


class TestConsolidationBatch:
    """ConsolidationBatch bounded tests."""

    def test_max_size(self):
        from core.training.consolidation import ConsolidationBatch, ConsolidationSample
        batch = ConsolidationBatch()
        for i in range(100):
            batch.add(ConsolidationSample(step=i, state_hash=f"h{i}"))
        assert batch.size <= 64

    def test_dedup(self):
        from core.training.consolidation import ConsolidationBatch, ConsolidationSample
        batch = ConsolidationBatch()
        batch.add(ConsolidationSample(step=1, state_hash="abc"))
        batch.add(ConsolidationSample(step=2, state_hash="abc"))  # dup
        assert batch.size == 1

    def test_budget_check(self):
        from core.training.consolidation import ConsolidationBatch
        batch = ConsolidationBatch(token_budget=100, tokens_used=99)
        assert not batch.is_budget_exceeded()
        batch.tokens_used = 100
        assert batch.is_budget_exceeded()


class TestConsolidationEngine:
    """ConsolidationEngine build and run tests."""

    def test_build_batch_sorted_by_priority(self):
        from core.training.consolidation import ConsolidationEngine, ConsolidationSample
        engine = ConsolidationEngine()
        samples = [
            ConsolidationSample(step=0, da_level=0.5, ach_level=0.0, state_hash="a"),
            ConsolidationSample(step=1, da_level=1.0, ach_level=0.9, hypothesis_confirmed=True, state_hash="b"),
            ConsolidationSample(step=2, da_level=0.3, ach_level=0.1, state_hash="c"),
        ]
        batch = engine.build_batch(samples)
        assert batch.size == 3
        # Highest priority should be first
        assert batch.samples[0].step == 1

    def test_build_batch_bounded(self):
        from core.training.consolidation import ConsolidationEngine, ConsolidationSample
        engine = ConsolidationEngine()
        samples = [
            ConsolidationSample(step=i, da_level=0.8, state_hash=f"h{i}")
            for i in range(100)
        ]
        batch = engine.build_batch(samples)
        assert batch.size <= 64

    def test_run_without_targets(self):
        from core.training.consolidation import ConsolidationEngine, ConsolidationSample, ConsolidationBatch
        engine = ConsolidationEngine()
        batch = ConsolidationBatch()
        batch.add(ConsolidationSample(step=0, command="test", source="ppo"))
        metrics = engine.run(batch)
        assert metrics.samples_selected == 1

    def test_compute_state_hash(self):
        from core.training.consolidation import ConsolidationEngine
        h1 = ConsolidationEngine.compute_state_hash("nmap", "RECON", 5)
        h2 = ConsolidationEngine.compute_state_hash("nmap", "RECON", 5)
        h3 = ConsolidationEngine.compute_state_hash("nmap", "RECON", 6)
        assert h1 == h2
        assert h1 != h3
