#!/usr/bin/env python3
"""Phase 33.2: Dynamic Burst Pool tests."""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ.setdefault("ARIASKA_DRY_RUN", "1")


class TestBurstPoolConstants:
    """Burst pool sizing relative to max budget."""

    def test_burst_pool_ratio(self):
        from core.llm.budget_manager import _BURST_POOL_RATIO
        assert _BURST_POOL_RATIO == pytest.approx(0.12, abs=0.001)

    def test_burst_step_cap_ratio(self):
        from core.llm.budget_manager import _BURST_STEP_CAP_RATIO
        assert _BURST_STEP_CAP_RATIO == pytest.approx(0.03, abs=0.001)

    def test_burst_cooldown(self):
        from core.llm.budget_manager import _BURST_COOLDOWN_STEPS
        assert _BURST_COOLDOWN_STEPS == 5

    def test_burst_only_mini_codex(self):
        from core.llm.budget_manager import _BURST_TIERS
        assert _BURST_TIERS == frozenset({"mini", "codex"})


class TestBurstPoolBehavior:
    """Burst pool request/available/cooldown logic."""

    def test_burst_available_initially(self):
        from core.llm.budget_manager import BudgetManagerV2
        bm = BudgetManagerV2()
        assert bm.burst_available(step=0, tier="mini") is True
        assert bm.burst_available(step=0, tier="codex") is True

    def test_burst_denied_for_nano(self):
        from core.llm.budget_manager import BudgetManagerV2
        bm = BudgetManagerV2()
        assert bm.burst_available(step=0, tier="nano") is False
        assert bm.burst_request(step=0, tier="nano", tokens_needed=1000,
                                trigger="test") == 0

    def test_burst_denied_for_full(self):
        from core.llm.budget_manager import BudgetManagerV2
        bm = BudgetManagerV2()
        assert bm.burst_available(step=0, tier="full") is False

    def test_burst_request_grants_tokens(self):
        from core.llm.budget_manager import BudgetManagerV2
        bm = BudgetManagerV2()
        granted = bm.burst_request(
            step=0, tier="mini", tokens_needed=5000,
            trigger="semantic_stall",
        )
        assert granted > 0
        assert granted <= 5000

    def test_burst_capped_per_step(self):
        from core.llm.budget_manager import BudgetManagerV2, _TOTAL_BUDGET, _BURST_STEP_CAP_RATIO
        bm = BudgetManagerV2()
        step_cap = int(_TOTAL_BUDGET * _BURST_STEP_CAP_RATIO)
        # Request way more than step cap
        granted = bm.burst_request(
            step=0, tier="codex", tokens_needed=999_999,
            trigger="strategy_pivot",
        )
        assert granted == step_cap

    def test_burst_cooldown_enforced(self):
        from core.llm.budget_manager import BudgetManagerV2, _BURST_COOLDOWN_STEPS
        bm = BudgetManagerV2()
        # First burst at step 0
        g1 = bm.burst_request(step=0, tier="mini", tokens_needed=1000,
                              trigger="semantic_stall")
        assert g1 > 0
        # Try again at step 1 — should be denied (cooldown = 5)
        assert bm.burst_available(step=1, tier="mini") is False
        g2 = bm.burst_request(step=1, tier="mini", tokens_needed=1000,
                              trigger="semantic_stall")
        assert g2 == 0
        # Try at step 5 — should work (cooldown elapsed)
        assert bm.burst_available(step=_BURST_COOLDOWN_STEPS, tier="mini") is True
        g3 = bm.burst_request(step=_BURST_COOLDOWN_STEPS, tier="mini",
                              tokens_needed=1000, trigger="mc_escalation_high")
        assert g3 > 0

    def test_burst_pool_exhaustion(self):
        from core.llm.budget_manager import BudgetManagerV2, _TOTAL_BUDGET, _BURST_POOL_RATIO
        bm = BudgetManagerV2()
        pool_total = int(_TOTAL_BUDGET * _BURST_POOL_RATIO)
        # Drain pool across multiple steps (respecting cooldown)
        step = 0
        total_granted = 0
        while total_granted < pool_total:
            g = bm.burst_request(step=step, tier="codex",
                                 tokens_needed=pool_total,
                                 trigger="strategy_pivot")
            if g == 0:
                break
            total_granted += g
            step += 5  # respect cooldown
        # Pool should be exhausted
        assert bm.burst_available(step=step + 5, tier="codex") is False

    def test_burst_reset_on_episode(self):
        from core.llm.budget_manager import BudgetManagerV2
        bm = BudgetManagerV2()
        bm.burst_request(step=0, tier="mini", tokens_needed=5000,
                         trigger="semantic_stall")
        stats_before = bm.get_burst_stats()
        assert stats_before["burst_used"] > 0
        bm.reset_episode("ep2")
        stats_after = bm.get_burst_stats()
        assert stats_after["burst_used"] == 0
        assert stats_after["burst_count"] == 0

    def test_burst_stats_complete(self):
        from core.llm.budget_manager import BudgetManagerV2
        bm = BudgetManagerV2()
        bm.burst_request(step=0, tier="codex", tokens_needed=3000,
                         trigger="low_confidence_streak")
        s = bm.get_burst_stats()
        assert "burst_pool_total" in s
        assert "burst_used" in s
        assert "burst_remaining" in s
        assert "burst_count" in s
        assert s["burst_count"] == 1
        assert s["burst_remaining"] == s["burst_pool_total"] - s["burst_used"]
