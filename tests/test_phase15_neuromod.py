#!/usr/bin/env python3
"""
tests/test_phase15_neuromod.py — Phase 15.0: Neuromodulator tests

Tests NeuromodulatorState, NeuromodulatorEngine, NeuromodulatorHistory.
All values bounded [0,1], history bounded to 256.
"""

import os
import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


class TestNeuromodulatorState:
    """NeuromodulatorState schema tests."""

    def test_default_values(self):
        from core.neuro.neuromodulators import NeuromodulatorState
        s = NeuromodulatorState()
        assert s.da == 0.5
        assert s.ne == 0.3
        assert s.ach == 0.4
        assert s.sht == 0.5

    def test_to_dict(self):
        from core.neuro.neuromodulators import NeuromodulatorState
        s = NeuromodulatorState(da=0.8, ne=0.2, ach=0.6, sht=0.4)
        d = s.to_dict()
        assert d["da"] == 0.8
        assert d["ne"] == 0.2

    def test_to_vector(self):
        from core.neuro.neuromodulators import NeuromodulatorState
        s = NeuromodulatorState(da=0.1, ne=0.2, ach=0.3, sht=0.4)
        v = s.to_vector()
        assert len(v) == 4
        assert v == [0.1, 0.2, 0.3, 0.4]

    def test_from_dict_clamps(self):
        from core.neuro.neuromodulators import NeuromodulatorState
        s = NeuromodulatorState.from_dict({"da": 1.5, "ne": -0.3})
        assert s.da == 1.0
        assert s.ne == 0.0


class TestNeuromodulatorEngine:
    """NeuromodulatorEngine tests."""

    def test_compute_default(self):
        from core.neuro.neuromodulators import NeuromodulatorEngine, NeuromodulatorInputs
        engine = NeuromodulatorEngine()
        result = engine.compute(NeuromodulatorInputs())
        assert 0.0 <= result.da <= 1.0
        assert 0.0 <= result.ne <= 1.0
        assert 0.0 <= result.ach <= 1.0
        assert 0.0 <= result.sht <= 1.0

    def test_compute_high_rpe(self):
        from core.neuro.neuromodulators import (
            NeuromodulatorEngine, NeuromodulatorInputs, NeuromodulatorState,
        )
        engine = NeuromodulatorEngine(alpha=1.0)  # no smoothing
        inputs = NeuromodulatorInputs(
            predicted_value=0.0,
            realized_reward=5.0,  # big positive RPE
        )
        result = engine.compute(inputs)
        assert result.da > 0.7  # high DA from positive RPE

    def test_compute_high_uncertainty(self):
        from core.neuro.neuromodulators import NeuromodulatorEngine, NeuromodulatorInputs
        engine = NeuromodulatorEngine(alpha=1.0)
        inputs = NeuromodulatorInputs(
            policy_entropy=1.0,
            confidence_min=0.0,
            confidence_disagreements=3,
            hypothesis_refuted_rate=1.0,
        )
        result = engine.compute(inputs)
        assert result.ne > 0.5  # high NE from uncertainty

    def test_apply_modulation_keys(self):
        from core.neuro.neuromodulators import NeuromodulatorEngine, NeuromodulatorState
        engine = NeuromodulatorEngine()
        state = NeuromodulatorState()
        mods = engine.apply_modulation(state)
        expected_keys = {
            "lr_mult", "entropy_coef_mult", "bc_weight_mult",
            "exploration_bonus", "mentor_gate_threshold_offset",
            "consolidation_priority_mult", "aggression_offset",
        }
        assert set(mods.keys()) == expected_keys

    def test_apply_modulation_bounded(self):
        from core.neuro.neuromodulators import NeuromodulatorEngine, NeuromodulatorState
        engine = NeuromodulatorEngine()
        for da in [0.0, 0.5, 1.0]:
            for ne in [0.0, 0.5, 1.0]:
                s = NeuromodulatorState(da=da, ne=ne)
                mods = engine.apply_modulation(s)
                assert mods["lr_mult"] >= 0.0
                assert mods["lr_mult"] <= 1.0
                assert mods["entropy_coef_mult"] >= 0.0


class TestNeuromodulatorHistory:
    """NeuromodulatorHistory bounded tests."""

    def test_bounded(self):
        from core.neuro.neuromodulators import NeuromodulatorHistory, NeuromodulatorState
        hist = NeuromodulatorHistory(max_steps=10)
        for i in range(20):
            hist.record(NeuromodulatorState(da=float(i) / 20))
        assert len(hist) <= 10

    def test_get_means(self):
        from core.neuro.neuromodulators import NeuromodulatorHistory, NeuromodulatorState
        hist = NeuromodulatorHistory()
        hist.record(NeuromodulatorState(da=0.2, ne=0.4, ach=0.6, sht=0.8))
        hist.record(NeuromodulatorState(da=0.8, ne=0.6, ach=0.4, sht=0.2))
        means = hist.get_means()
        assert abs(means["da"] - 0.5) < 0.01
        assert abs(means["ne"] - 0.5) < 0.01

    def test_clear(self):
        from core.neuro.neuromodulators import NeuromodulatorHistory, NeuromodulatorState
        hist = NeuromodulatorHistory()
        hist.record(NeuromodulatorState())
        hist.clear()
        assert len(hist) == 0
