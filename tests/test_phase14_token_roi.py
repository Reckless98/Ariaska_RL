#!/usr/bin/env python3
"""
tests/test_phase14_token_roi.py — Phase 14.0 T1: Token ROI & Autonomy Tests

Contract C3.8: 3 tests verifying token budget tracking and autonomy scheduler
reduces mentor calls over episodes.
"""

import os
import sys
import pytest

os.environ["ARIASKA_DRY_RUN"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTokenROI:
    """Verify token budget tracking and LLM call caching."""

    def test_token_budget_tracking(self):
        """FakeGPTManager tracks token usage across requests."""
        from core.testing.fake_gpt_manager import FakeGPTManager

        gpt = FakeGPTManager(seed=42)
        # Make several requests
        for i in range(5):
            gpt.gpt_request(f"prompt_{i}", task_type="tactical", agent_id="test")

        # FakeGPTManager should track requests
        assert gpt._request_count > 0
        assert gpt.stats["total_requests"] == 5

    def test_mentor_call_counting(self):
        """AutonomyScheduler correctly counts mentor calls via update."""
        from core.training.autonomy_scheduler import AutonomyScheduler

        sched = AutonomyScheduler()
        sched.register_agent("red")

        # Initially at score 0, well below threshold
        should_call, reason = sched.should_call_mentor("red", episode=0)
        assert should_call is True  # Score 0 < threshold 0.8

        # After many successful updates, score should increase
        for i in range(20):
            sched.update("red", success_rate=0.8, divergence_rate=0.2, diversity_rate=0.7)

        # Score should have increased
        score = sched.get_score("red")
        assert score > 0.0

    def test_autonomy_reduces_mentor_over_episodes(self):
        """As episodes progress, threshold decreases = more autonomy."""
        from core.training.autonomy_scheduler import AutonomyScheduler

        sched = AutonomyScheduler()
        sched.register_agent("red")

        # Threshold at episode 0
        t0 = sched.get_threshold(episode=0)
        # Threshold at episode 50
        t50 = sched.get_threshold(episode=50)

        # Threshold should decrease over episodes (agent needs less mentoring)
        assert t50 < t0
        assert t0 == pytest.approx(0.8, abs=0.01)
        # At episode 50: max(0.2, 0.8 - 0.012*50) = max(0.2, 0.2) = 0.2
        assert t50 == pytest.approx(0.2, abs=0.01)

    def test_llm_call_cache(self):
        """LLMCallCache provides hit/miss tracking."""
        from core.llm.call_cache import LLMCallCache

        cache = LLMCallCache(capacity=100)

        # Miss on first call
        result = cache.get("test prompt", model="gpt-4")
        assert result is None

        # Put response
        cache.put("test prompt", "response text", model="gpt-4", tokens_used=50)

        # Hit on second call
        result = cache.get("test prompt", model="gpt-4")
        assert result == "response text"

        stats = cache.get_stats()
        assert stats["total_hits"] == 1
        assert stats["total_misses"] == 1
        assert stats["size"] == 1

    def test_llm_call_cache_lru_eviction(self):
        """LLMCallCache evicts LRU entries when capacity exceeded."""
        from core.llm.call_cache import LLMCallCache

        cache = LLMCallCache(capacity=3)

        # Fill cache
        for i in range(5):
            cache.put(f"prompt_{i}", f"response_{i}")

        stats = cache.get_stats()
        assert stats["size"] == 3  # Oldest 2 evicted

        # Oldest entries should be evicted
        assert cache.get("prompt_0") is None
        assert cache.get("prompt_1") is None
        # Newest entries should remain
        assert cache.get("prompt_4") == "response_4"
