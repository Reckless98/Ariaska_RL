#!/usr/bin/env python3
"""
tests/test_phase14_autonomy.py — Phase 14.0: AutonomyScheduler Tests

Contract C3.3: 8 required tests for autonomy scoring, thresholds, re-escalation.
"""

import os
import pytest

os.environ["ARIASKA_DRY_RUN"] = "1"


class TestAutonomyScheduler:
    """C3.3: AutonomyScheduler monotonic invariants and scoring."""

    def test_initial_score_zero(self):
        """New agent starts with score 0 (no updates yet)."""
        from core.training.autonomy_scheduler import AutonomyScheduler
        sched = AutonomyScheduler()
        sched.register_agent("RedAgent")
        assert sched.get_score("RedAgent") == 0.0

    def test_threshold_decreases_with_episode(self):
        """Threshold decreases as episodes progress."""
        from core.training.autonomy_scheduler import AutonomyScheduler
        sched = AutonomyScheduler()
        t0 = sched.get_threshold(0)
        t10 = sched.get_threshold(10)
        t50 = sched.get_threshold(50)
        assert t0 > t10 > t50
        assert t0 == pytest.approx(0.8)
        assert t50 == pytest.approx(0.2)  # floor at 0.2

    def test_threshold_floor(self):
        """Threshold never goes below 0.2."""
        from core.training.autonomy_scheduler import AutonomyScheduler
        sched = AutonomyScheduler()
        assert sched.get_threshold(1000) == pytest.approx(0.2)
        assert sched.get_threshold(200) == pytest.approx(0.2)

    def test_score_computation_formula(self):
        """Score = ALPHA*success + BETA*(1-div) + GAMMA*diversity."""
        from core.training.autonomy_scheduler import AutonomyScheduler
        sched = AutonomyScheduler()
        sched.register_agent("Red")
        sched.update("Red", success_rate=1.0, divergence_rate=0.0, diversity_rate=1.0)
        # 0.4*1.0 + 0.3*1.0 + 0.3*1.0 = 1.0
        assert sched.get_score("Red") == pytest.approx(1.0)

    def test_monotonic_enforcement(self):
        """Score cannot decrease (monotonic non-decreasing)."""
        from core.training.autonomy_scheduler import AutonomyScheduler
        sched = AutonomyScheduler()
        sched.register_agent("Red")

        # First update: high metrics
        sched.update("Red", success_rate=0.8, divergence_rate=0.1, diversity_rate=0.7)
        score_high = sched.get_score("Red")

        # Second update: lower metrics — score should NOT decrease
        sched.update("Red", success_rate=0.2, divergence_rate=0.9, diversity_rate=0.1)
        score_after = sched.get_score("Red")

        assert score_after >= score_high

    def test_should_call_mentor_early_episodes(self):
        """Early episodes should call mentor (score < threshold)."""
        from core.training.autonomy_scheduler import AutonomyScheduler
        sched = AutonomyScheduler()
        sched.register_agent("Red")
        should_call, reason = sched.should_call_mentor("Red", episode=0)
        assert should_call is True

    def test_valid_reescalation_reasons(self):
        """Only 3 valid re-escalation reasons accepted."""
        from core.training.autonomy_scheduler import AutonomyScheduler
        sched = AutonomyScheduler()
        assert sched.can_reescalate("uncertainty_spike") is True
        assert sched.can_reescalate("hypothesis_failure_streak") is True
        assert sched.can_reescalate("guardrail_triggered") is True
        assert sched.can_reescalate("bad_reason") is False
        assert sched.can_reescalate("") is False

    def test_reescalation_reduces_score(self):
        """Valid re-escalation reduces autonomy score."""
        from core.training.autonomy_scheduler import AutonomyScheduler
        sched = AutonomyScheduler()
        sched.register_agent("Red")
        sched.update("Red", success_rate=0.8, divergence_rate=0.0, diversity_rate=0.8)
        score_before = sched.get_score("Red")

        result = sched.reescalate("Red", "uncertainty_spike", reduction=0.1)
        assert result is True
        assert sched.get_score("Red") < score_before

        # Invalid reason rejected
        result2 = sched.reescalate("Red", "invalid_reason")
        assert result2 is False
