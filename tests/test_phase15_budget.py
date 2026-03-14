#!/usr/bin/env python3
"""
tests/test_phase15_budget.py — Phase 15.0: BudgetManagerV2 tests

Tests per-tier budgets, ROI tags, budget denial, and spend recording.
"""

import os
import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


class TestBudgetManagerV2:
    """BudgetManagerV2 tests."""

    def test_initial_budget(self):
        from core.llm.budget_manager import BudgetManagerV2, _TOTAL_BUDGET
        bm = BudgetManagerV2()
        stats = bm.get_stats()
        assert stats["total_budget"] == _TOTAL_BUDGET
        assert stats["total_used"] == 0

    def test_check_budget_allowed(self):
        from core.llm.budget_manager import BudgetManagerV2
        bm = BudgetManagerV2()
        decision = bm.check_budget("local-llm", 100, "classification")
        assert decision.allowed is True
        assert decision.tier == "local"

    def test_check_budget_always_allowed(self):
        """All local models are always allowed (zero cost)."""
        from core.llm.budget_manager import BudgetManagerV2
        bm = BudgetManagerV2(tier_budgets={"codex": 100, "full": 100, "mini": 100, "nano": 100})
        bm.record_spend("local-llm", 999_999, "strategy_plan")
        decision = bm.check_budget("local-llm", 1, "strategy_plan")
        assert decision.allowed is True
        assert decision.tier == "local"

    def test_record_spend(self):
        from core.llm.budget_manager import BudgetManagerV2
        bm = BudgetManagerV2()
        bm.record_spend("local-llm", 500, "classification")
        stats = bm.get_stats()
        assert stats["total_used"] == 500
        assert "classification" in stats["roi"]

    def test_cache_hit_no_spend(self):
        from core.llm.budget_manager import BudgetManagerV2
        bm = BudgetManagerV2()
        bm.record_spend("local-llm", 500, "classification", cache_hit=True)
        stats = bm.get_stats()
        assert stats["total_used"] == 0  # cache hit → no spend

    def test_reset_episode(self):
        from core.llm.budget_manager import BudgetManagerV2
        bm = BudgetManagerV2()
        bm.record_spend("local-llm", 1000, "verification")
        bm.reset_episode("ep_2")
        stats = bm.get_stats()
        assert stats["total_used"] == 0
        assert stats["episode_id"] == "ep_2"

    def test_tier_mapping(self):
        from core.llm.budget_manager import BudgetManagerV2
        bm = BudgetManagerV2()
        assert bm.get_tier_for_model("local-llm") == "local"
        assert bm.get_tier_for_model("jaahas/qwen3.5-uncensored:4b") == "local"
        assert bm.get_tier_for_model("jaahas/qwen3.5-uncensored:9b") == "local"
        assert bm.get_tier_for_model("unknown-model") == "mini"  # fallback

    def test_stable_roi_tags(self):
        from core.llm.budget_manager import BudgetManagerV2
        bm = BudgetManagerV2()
        assert bm.is_stable_roi("classification") is True
        assert bm.is_stable_roi("verification") is True
        assert bm.is_stable_roi("strategy_plan") is False
