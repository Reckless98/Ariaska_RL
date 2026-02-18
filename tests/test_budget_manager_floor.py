"""Phase 27.6 + Phase 32: Budget Manager floor clamp tests."""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ.setdefault("ARIASKA_DRY_RUN", "1")

from core.llm.budget_manager import (
    BudgetManagerV2,
    _TOTAL_BUDGET,
    _MIN_BUDGET,
    _MIN_SCALE,
)


class TestBudgetManagerFloor:
    """Validate 50% floor on dynamic budget scaling."""

    def test_min_budget_is_half_of_max(self):
        """_MIN_BUDGET >= 50% of _TOTAL_BUDGET."""
        assert _MIN_BUDGET >= _TOTAL_BUDGET * 0.5
        assert _MIN_BUDGET == _TOTAL_BUDGET // 2

    def test_min_scale_is_half(self):
        assert _MIN_SCALE == pytest.approx(0.5, abs=0.01)

    def test_full_maturity_never_below_floor(self):
        """Even with maturity=1.0, scale should not drop below 0.5."""
        bm = BudgetManagerV2()
        scale = bm.compute_dynamic_budget(
            avg_success_rate=1.0,
            skill_count=100,
            max_skills=50,
            discovery_efficiency=1.0,
            stagnation_rate=0.0,
            episode=100,
        )
        assert scale >= 0.5
        assert scale == pytest.approx(_MIN_SCALE, abs=0.01)

    def test_warmup_still_full(self):
        """First 5 episodes should still get scale=1.0."""
        bm = BudgetManagerV2()
        for ep in range(5):
            scale = bm.compute_dynamic_budget(
                avg_success_rate=1.0, skill_count=50,
                episode=ep,
            )
            assert scale == 1.0

    def test_apply_dynamic_scale_respects_floor(self):
        """apply_dynamic_scale with floor value should give >= 50% budgets."""
        bm = BudgetManagerV2()
        bm.compute_dynamic_budget(
            avg_success_rate=1.0, skill_count=100, max_skills=50,
            discovery_efficiency=1.0, stagnation_rate=0.0, episode=50,
        )
        bm.apply_dynamic_scale()
        stats = bm.get_stats()
        assert stats["total_budget"] >= _TOTAL_BUDGET * 0.5

    def test_zero_maturity_full_budget(self):
        """Zero maturity should give full budget."""
        bm = BudgetManagerV2()
        scale = bm.compute_dynamic_budget(
            avg_success_rate=0.0, skill_count=0, max_skills=50,
            discovery_efficiency=0.0, stagnation_rate=1.0, episode=10,
        )
        assert scale == pytest.approx(1.0, abs=0.05)
