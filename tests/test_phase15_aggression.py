#!/usr/bin/env python3
"""
tests/test_phase15_aggression.py — Phase 15.0: Aggression Controller tests

Tests AggressionController computation, bounding, reason codes,
and AggressionHistory.
"""

import os
import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


class TestAggressionController:
    """AggressionController computation tests."""

    def test_default_level(self):
        from core.neuro.aggression_controller import AggressionController, AggressionInputs
        ctrl = AggressionController()
        result = ctrl.compute(AggressionInputs())
        assert 0.0 <= result.level <= 1.0

    def test_recon_phase_low(self):
        from core.neuro.aggression_controller import AggressionController, AggressionInputs
        ctrl = AggressionController()
        result = ctrl.compute(AggressionInputs(phase="RECON"))
        assert result.level < 0.5  # RECON baseline is 0.2

    def test_exploitation_phase_high(self):
        from core.neuro.aggression_controller import AggressionController, AggressionInputs
        ctrl = AggressionController()
        result = ctrl.compute(AggressionInputs(phase="EXPLOITATION"))
        assert result.level > 0.4  # EXPLOITATION baseline is 0.6

    def test_bounded_output(self):
        from core.neuro.aggression_controller import AggressionController, AggressionInputs
        ctrl = AggressionController()
        # Extreme inputs
        result = ctrl.compute(AggressionInputs(
            phase="PRIVILEGE_ESCALATION",
            da_level=1.0,
            sht_level=0.0,
            steps_since_progress=20,
            recent_successes=10,
            shell_obtained=True,
        ))
        assert result.level <= 1.0
        assert result.level >= 0.0

    def test_reason_codes_present(self):
        from core.neuro.aggression_controller import AggressionController, AggressionInputs
        ctrl = AggressionController()
        result = ctrl.compute(AggressionInputs(
            phase="EXPLOITATION",
            steps_since_progress=10,
        ))
        assert len(result.reason_codes) > 0
        assert any("phase_base" in r for r in result.reason_codes)

    def test_detection_risk_dampens(self):
        from core.neuro.aggression_controller import AggressionController, AggressionInputs
        ctrl = AggressionController()
        low_risk = ctrl.compute(AggressionInputs(phase="EXPLOITATION", detection_risk=0.0))
        high_risk = ctrl.compute(AggressionInputs(phase="EXPLOITATION", detection_risk=0.9))
        assert high_risk.level < low_risk.level


class TestAggressionHistory:
    """AggressionHistory bounded tests."""

    def test_bounded(self):
        from core.neuro.aggression_controller import AggressionHistory, AggressionState
        hist = AggressionHistory(max_steps=5)
        for i in range(10):
            hist.record(AggressionState(level=float(i) / 10))
        assert len(hist) <= 5

    def test_mean(self):
        from core.neuro.aggression_controller import AggressionHistory, AggressionState
        hist = AggressionHistory()
        hist.record(AggressionState(level=0.2))
        hist.record(AggressionState(level=0.8))
        assert abs(hist.get_mean() - 0.5) < 0.01
