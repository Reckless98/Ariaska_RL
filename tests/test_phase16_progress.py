#!/usr/bin/env python3
"""
tests/test_phase16_progress.py — Phase 16.0: Progress Estimator Tests

Tests for:
1. ProgressLabel / ProgressEstimate dataclasses
2. ProgressDataset circular buffer + JSONL persistence
3. ProgressMLP training + prediction
4. ProgressEstimator heuristic estimation
5. ProgressEstimator autonomy schedule
6. Feature flag gating (FF_PROGRESS_ESTIMATOR)
7. NeuromodulatorInputs.progress_estimate integration
8. ConsolidationSample.progress_delta integration
9. RewardCalculator.progress_delta integration
10. State encoder dims 168-172
"""

import json
import os
import sys
import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["ARIASKA_DRY_RUN"] = "1"


# ── Section 1: ProgressLabel ────────────────────────────────────────────────

class TestProgressLabel:
    """Test ProgressLabel dataclass."""

    def test_defaults(self):
        from core.neuro.progress_estimator import ProgressLabel
        label = ProgressLabel()
        assert label.foothold_progress == 0.0
        assert label.root_progress == 0.0
        assert label.source == "heuristic"
        assert label.step == 0

    def test_to_dict(self):
        from core.neuro.progress_estimator import ProgressLabel
        label = ProgressLabel(foothold_progress=0.5, root_progress=0.3, source="gpt")
        d = label.to_dict()
        assert d["foothold_progress"] == 0.5
        assert d["root_progress"] == 0.3
        assert d["source"] == "gpt"

    def test_from_dict(self):
        from core.neuro.progress_estimator import ProgressLabel
        d = {"foothold_progress": 0.7, "root_progress": 0.4, "source": "gpt", "step": 5}
        label = ProgressLabel.from_dict(d)
        assert label.foothold_progress == 0.7
        assert label.root_progress == 0.4
        assert label.step == 5

    def test_from_dict_defaults(self):
        from core.neuro.progress_estimator import ProgressLabel
        label = ProgressLabel.from_dict({})
        assert label.foothold_progress == 0.0
        assert label.source == "heuristic"


# ── Section 2: ProgressEstimate ─────────────────────────────────────────────

class TestProgressEstimate:
    """Test ProgressEstimate dataclass."""

    def test_combined_property(self):
        from core.neuro.progress_estimator import ProgressEstimate
        est = ProgressEstimate(foothold_progress=0.8, root_progress=0.5)
        # combined = 0.6 * 0.8 + 0.4 * 0.5 = 0.48 + 0.20 = 0.68
        assert abs(est.combined - 0.68) < 1e-6

    def test_to_dict(self):
        from core.neuro.progress_estimator import ProgressEstimate
        est = ProgressEstimate(foothold_progress=0.5, root_progress=0.3, confidence=0.8)
        d = est.to_dict()
        assert "foothold_progress" in d
        assert "combined" in d
        assert "confidence" in d
        assert d["source"] == "heuristic"

    def test_combined_zero(self):
        from core.neuro.progress_estimator import ProgressEstimate
        est = ProgressEstimate()
        assert est.combined == 0.0


# ── Section 3: ProgressDataset ──────────────────────────────────────────────

class TestProgressDataset:
    """Test ProgressDataset circular buffer."""

    def test_empty_dataset(self):
        from core.neuro.progress_estimator import ProgressDataset
        ds = ProgressDataset(capacity=100)
        assert ds.size == 0
        X, Y = ds.get_training_data()
        assert len(X) == 0
        assert len(Y) == 0

    def test_add_entries(self):
        from core.neuro.progress_estimator import ProgressDataset, ProgressLabel
        ds = ProgressDataset(capacity=100)
        vec = [0.0] * 512
        label = ProgressLabel(foothold_progress=0.5, root_progress=0.3)
        ds.add(vec, label)
        assert ds.size == 1
        X, Y = ds.get_training_data()
        assert len(X) == 1
        assert len(Y) == 1
        assert Y[0] == (0.5, 0.3)

    def test_circular_buffer_eviction(self):
        from core.neuro.progress_estimator import ProgressDataset, ProgressLabel
        ds = ProgressDataset(capacity=5)
        for i in range(10):
            ds.add([float(i)] * 512, ProgressLabel(foothold_progress=i / 10.0))
        assert ds.size == 5
        X, Y = ds.get_training_data()
        # Should have entries 5-9 (oldest evicted)
        assert X[0][0] == 5.0

    def test_save_load_roundtrip(self):
        from core.neuro.progress_estimator import ProgressDataset, ProgressLabel
        ds = ProgressDataset(capacity=100)
        for i in range(3):
            ds.add([float(i)] * 512, ProgressLabel(foothold_progress=i * 0.1))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_labels.jsonl"
            ds.save(path)
            assert path.exists()
            
            ds2 = ProgressDataset(capacity=100)
            ds2.load(path)
            assert ds2.size == 3

    def test_load_nonexistent_file(self):
        from core.neuro.progress_estimator import ProgressDataset
        ds = ProgressDataset()
        ds.load(Path("/nonexistent/path.jsonl"))
        assert ds.size == 0

    def test_clear(self):
        from core.neuro.progress_estimator import ProgressDataset, ProgressLabel
        ds = ProgressDataset()
        ds.add([0.0] * 512, ProgressLabel())
        assert ds.size == 1
        ds.clear()
        assert ds.size == 0


# ── Section 4: ProgressMLP ──────────────────────────────────────────────────

class TestProgressMLP:
    """Test ProgressMLP training and prediction."""

    def test_untrained_predict_returns_zeros(self):
        from core.neuro.progress_estimator import ProgressMLP
        mlp = ProgressMLP()
        fp, rp = mlp.predict([0.0] * 512)
        assert fp == 0.0
        assert rp == 0.0
        assert not mlp.is_trained

    def test_insufficient_data_skips_training(self):
        from core.neuro.progress_estimator import ProgressMLP
        mlp = ProgressMLP()
        # Less than MIN_TRAIN_SAMPLES (20)
        X = [[0.0] * 512 for _ in range(5)]
        Y = [(0.5, 0.3)] * 5
        metrics = mlp.train_on_dataset(X, Y)
        assert metrics["loss"] == -1.0
        assert not mlp.is_trained

    def test_training_with_sufficient_data(self):
        from core.neuro.progress_estimator import ProgressMLP
        mlp = ProgressMLP()
        # Create 30 samples (above threshold of 20)
        X = [[float(i % 10) / 10.0] * 512 for i in range(30)]
        Y = [(i / 30.0, i / 60.0) for i in range(30)]
        metrics = mlp.train_on_dataset(X, Y, epochs=2)
        assert metrics["loss"] >= 0
        assert metrics["samples"] == 30
        assert mlp.is_trained

    def test_predict_after_training(self):
        from core.neuro.progress_estimator import ProgressMLP
        mlp = ProgressMLP()
        X = [[1.0] * 512 for _ in range(25)]
        Y = [(0.8, 0.6)] * 25
        mlp.train_on_dataset(X, Y, epochs=10)
        assert mlp.is_trained
        fp, rp = mlp.predict([1.0] * 512)
        # Should be in [0, 1] range (sigmoid output)
        assert 0.0 <= fp <= 1.0
        assert 0.0 <= rp <= 1.0


# ── Section 5: Heuristic Estimation ─────────────────────────────────────────

class TestHeuristicEstimation:
    """Test ProgressEstimator._heuristic_estimate."""

    def _make_estimator(self):
        from core.neuro.progress_estimator import ProgressEstimator
        with tempfile.TemporaryDirectory() as tmpdir:
            return ProgressEstimator(persist_dir=tmpdir)

    def test_empty_board(self):
        pe = self._make_estimator()
        fp, rp = pe._heuristic_estimate({})
        assert fp == 0.0
        assert rp == 0.0

    def test_ports_only(self):
        pe = self._make_estimator()
        fp, rp = pe._heuristic_estimate({
            "ports": {21, 22, 80, 443, 3306},
            "services": set(),
            "credentials": set(),
        })
        # 5+ ports = 0.10 full credit
        assert fp == pytest.approx(0.10, abs=0.02)
        assert rp < 0.02  # very low without shell

    def test_shell_obtained(self):
        pe = self._make_estimator()
        fp, rp = pe._heuristic_estimate({
            "ports": {21, 22, 80},
            "services": {"ssh", "http"},
            "credentials": {"admin:password"},
            "vulns": {"CVE-2021-1234"},
            "shells": {"user_shell@target"},
        })
        # Shell = 0.20 + ports + services + creds + vulns
        assert fp > 0.5
        # Shell → rp starts at 0.4
        assert rp >= 0.4

    def test_root_shell(self):
        pe = self._make_estimator()
        fp, rp = pe._heuristic_estimate({
            "ports": {21, 22},
            "shells": {"root_shell"},
            "root_shell": True,
        })
        assert rp == 1.0

    def test_flags_trigger_root(self):
        pe = self._make_estimator()
        fp, rp = pe._heuristic_estimate({
            "ports": {22},
            "shells": {"shell"},
            "flags_set": {"flag1"},
        })
        assert rp == 1.0

    def test_no_shell_low_root(self):
        pe = self._make_estimator()
        fp, rp = pe._heuristic_estimate({
            "ports": {21, 22, 80, 443, 3306, 5432, 8080},
            "services": {"ssh", "http", "mysql"},
            "credentials": set(),
            "vulns": set(),
        })
        # Without shell, root should be very low
        assert rp < 0.15


# ── Section 6: ProgressEstimator Full API ────────────────────────────────────

class TestProgressEstimator:
    """Test ProgressEstimator main API."""

    def _make(self, tmpdir=None):
        from core.neuro.progress_estimator import ProgressEstimator
        d = tmpdir or tempfile.mkdtemp()
        return ProgressEstimator(persist_dir=d)

    def test_init(self):
        pe = self._make()
        assert pe.dataset_size == 0
        assert pe.confidence == 0.0
        assert pe.get_autonomy_level() == "bootstrap"

    def test_estimate_heuristic_fallback(self):
        pe = self._make()
        est = pe.estimate([0.0] * 512, {
            "ports": {21, 22},
            "services": {"ssh"},
        })
        assert est.source == "heuristic"
        assert est.confidence == 0.0
        assert 0.0 <= est.foothold_progress <= 1.0

    def test_estimate_delta_tracking(self):
        pe = self._make()
        est1 = pe.estimate([0.0] * 512, {"ports": set()})
        est2 = pe.estimate([0.0] * 512, {
            "ports": {21, 22, 80},
            "services": {"ssh"},
            "credentials": {"admin:pass"},
        })
        # Delta should be positive (progress made)
        assert est2.delta > 0

    def test_momentum_smoothing(self):
        pe = self._make()
        pe.estimate([0.0] * 512, {"ports": set()})
        pe.estimate([0.0] * 512, {"ports": {22}})
        pe.estimate([0.0] * 512, {"ports": {22, 80}})
        est = pe.estimate([0.0] * 512, {"ports": {22, 80, 443}})
        # Momentum should be positive (consistently gaining)
        assert est.momentum >= 0

    def test_confidence_ramp(self):
        pe = self._make()
        assert pe.confidence == 0.0
        # Add 250 samples → confidence = 250/500 = 0.5
        from core.neuro.progress_estimator import ProgressLabel
        for i in range(250):
            pe._dataset.add([0.0] * 512, ProgressLabel())
        assert abs(pe.confidence - 0.5) < 0.01
        # Add 250 more → confidence = 500/500 = 1.0
        for i in range(250):
            pe._dataset.add([0.0] * 512, ProgressLabel())
        assert pe.confidence == 1.0

    def test_reset_episode(self):
        pe = self._make()
        pe.estimate([0.0] * 512, {"ports": {22}})
        assert pe._prev_estimate is not None
        pe.reset_episode()
        assert pe._prev_estimate is None
        assert pe._momentum == 0.0

    def test_label_episode_heuristic(self):
        pe = self._make()
        states = [[0.0] * 512, [0.0] * 512, [0.0] * 512]
        boards = [
            {"ports": set()},
            {"ports": {22}},
            {"ports": {22, 80}, "services": {"ssh"}},
        ]
        labels = pe.label_episode_heuristic(states, boards, episode_id="test_ep")
        assert len(labels) == 3
        # Progress should increase across steps
        assert labels[2].foothold_progress >= labels[0].foothold_progress
        # All added to dataset
        assert pe.dataset_size == 3

    def test_label_episode_retroactively_no_gpt(self):
        pe = self._make()
        run_trace = {
            "run_id": "ep1",
            "total_reward": 50.0,
            "total_steps": 5,
            "success_rate": 0.0,
            "phase_progression": ["RECON"],
            "discoveries": {
                "open_port": [22, 80],
                "service": ["ssh"],
            },
        }
        labels = pe.label_episode_retroactively(run_trace, gpt_manager=None)
        assert len(labels) == 5
        # Linear ramp — last step should have highest values
        assert labels[-1].foothold_progress >= labels[0].foothold_progress

    def test_get_metrics(self):
        pe = self._make()
        m = pe.get_metrics()
        assert "dataset_size" in m
        assert "confidence" in m
        assert "mlp_trained" in m
        assert "autonomy_level" in m

    def test_repr(self):
        pe = self._make()
        r = repr(pe)
        assert "ProgressEstimator" in r
        assert "dataset=" in r


# ── Section 7: Autonomy Schedule ────────────────────────────────────────────

class TestAutonomySchedule:
    """Test GPT labeling frequency schedule."""

    def test_always_label_episode_1(self):
        pe = self._make_pe()
        assert pe.should_gpt_label(1) is True

    def test_bootstrap_labels_every_episode(self):
        pe = self._make_pe()
        # confidence = 0 (bootstrap)
        for ep in range(1, 10):
            assert pe.should_gpt_label(ep) is True

    def test_learning_labels_every_3rd(self):
        pe = self._make_pe()
        # Add enough samples for confidence > 0.7 (need > 350 samples)
        from core.neuro.progress_estimator import ProgressLabel
        for i in range(400):
            pe._dataset.add([0.0] * 512, ProgressLabel())
        assert pe.confidence > 0.7
        # Should label every 3rd
        assert pe.should_gpt_label(3) is True
        assert pe.should_gpt_label(4) is False
        assert pe.should_gpt_label(6) is True

    def test_autonomous_labels_every_10th(self):
        pe = self._make_pe()
        # Add enough samples for confidence > 0.9 (need > 450 samples)
        from core.neuro.progress_estimator import ProgressLabel
        for i in range(500):
            pe._dataset.add([0.0] * 512, ProgressLabel())
        assert pe.confidence >= 0.9
        # Should label every 10th
        assert pe.should_gpt_label(10) is True
        assert pe.should_gpt_label(11) is False
        assert pe.should_gpt_label(20) is True

    def test_autonomy_level_strings(self):
        pe = self._make_pe()
        assert pe.get_autonomy_level() == "bootstrap"
        from core.neuro.progress_estimator import ProgressLabel
        for i in range(400):
            pe._dataset.add([0.0] * 512, ProgressLabel())
        assert pe.get_autonomy_level() == "learning"
        for i in range(150):
            pe._dataset.add([0.0] * 512, ProgressLabel())
        assert pe.get_autonomy_level() == "autonomous"

    def _make_pe(self):
        from core.neuro.progress_estimator import ProgressEstimator
        return ProgressEstimator(persist_dir=tempfile.mkdtemp())


# ── Section 8: Feature Flag ─────────────────────────────────────────────────

class TestFeatureFlag:
    """Test FF_PROGRESS_ESTIMATOR flag."""

    def test_default_on(self):
        from core.feature_flags import FeatureFlags
        ff = FeatureFlags()
        assert ff.progress_estimator is True  # Post-Phase 20: default ON

    def test_env_enable(self):
        os.environ["FF_PROGRESS_ESTIMATOR"] = "1"
        try:
            from core.feature_flags import FeatureFlags
            ff = FeatureFlags()
            assert ff.progress_estimator is True
        finally:
            del os.environ["FF_PROGRESS_ESTIMATOR"]


# ── Section 9: NeuromodulatorInputs Integration ─────────────────────────────

class TestNeuromodulatorIntegration:
    """Test progress_estimate field in NeuromodulatorInputs."""

    def test_default_progress_estimate(self):
        from core.neuro.neuromodulators import NeuromodulatorInputs
        inputs = NeuromodulatorInputs()
        assert inputs.progress_estimate == 0.5

    def test_custom_progress_estimate(self):
        from core.neuro.neuromodulators import NeuromodulatorInputs
        inputs = NeuromodulatorInputs(progress_estimate=0.8)
        assert inputs.progress_estimate == 0.8

    def test_da_blends_with_progress(self):
        """DA should blend RPE (70%) with progress_estimate (30%)."""
        from core.neuro.neuromodulators import NeuromodulatorEngine, NeuromodulatorInputs
        engine = NeuromodulatorEngine(alpha=1.0)  # no smoothing

        # High progress + zero RPE → DA above 0.5
        inputs_high = NeuromodulatorInputs(
            predicted_value=0.0, realized_reward=0.0,
            progress_estimate=1.0,
        )
        state_high = engine.compute(inputs_high)
        # RPE=0 → raw_da_rpe=0.5, progress=1.0 → 0.7*0.5 + 0.3*1.0 = 0.65
        assert state_high.da > 0.6

        # Low progress + zero RPE → DA below 0.5
        inputs_low = NeuromodulatorInputs(
            predicted_value=0.0, realized_reward=0.0,
            progress_estimate=0.0,
        )
        state_low = engine.compute(inputs_low)
        assert state_low.da < 0.4

    def test_sht_uses_progress_signal(self):
        """5-HT should incorporate progress_estimate."""
        from core.neuro.neuromodulators import NeuromodulatorEngine, NeuromodulatorInputs
        engine = NeuromodulatorEngine(alpha=1.0)

        # High progress → high 5-HT (stability)
        inputs_high = NeuromodulatorInputs(progress_estimate=1.0)
        state_high = engine.compute(inputs_high)

        # Low progress → lower 5-HT
        inputs_low = NeuromodulatorInputs(progress_estimate=0.0)
        state_low = engine.compute(inputs_low)

        assert state_high.sht > state_low.sht


# ── Section 10: ConsolidationSample Integration ─────────────────────────────

class TestConsolidationIntegration:
    """Test progress_delta field in ConsolidationSample."""

    def test_default_progress_delta(self):
        from core.training.consolidation import ConsolidationSample
        cs = ConsolidationSample()
        assert cs.progress_delta == 0.0

    def test_progress_delta_affects_priority(self):
        from core.training.consolidation import ConsolidationSample
        # Sample with progress → higher priority
        cs_progress = ConsolidationSample(progress_delta=0.5)
        cs_no_progress = ConsolidationSample(progress_delta=0.0)
        assert cs_progress.priority_score() > cs_no_progress.priority_score()

    def test_negative_delta_also_informative(self):
        from core.training.consolidation import ConsolidationSample
        # Negative progress is also informative (|delta| used)
        cs_neg = ConsolidationSample(progress_delta=-0.5)
        cs_zero = ConsolidationSample(progress_delta=0.0)
        assert cs_neg.priority_score() > cs_zero.priority_score()

    def test_priority_score_normalized(self):
        from core.training.consolidation import ConsolidationSample
        # All maxed out inputs
        cs = ConsolidationSample(
            da_level=1.0, ach_level=1.0,
            hypothesis_confirmed=True, progress_delta=1.0,
        )
        score = cs.priority_score()
        # Should be close to 1.0 (all components maxed)
        assert 0.8 <= score <= 1.1


# ── Section 11: RewardCalculator Integration ────────────────────────────────

class TestRewardCalculatorIntegration:
    """Test progress_delta in calculate_reward."""

    def _make_calc(self):
        from core.llm.reward_calculator import SmartRewardCalculator
        return SmartRewardCalculator()

    def test_flat_bonus_without_estimator(self):
        calc = self._make_calc()
        from core.commands.command_registry import AttackPhase
        breakdown = calc.calculate_reward(
            template_name="nmap_basic",
            command="nmap -sV 192.168.1.1",
            success=True,
            raw_output="22/tcp open ssh",
            current_phase=AttackPhase.RECON,
            state_flags={},
        )
        # Default flat 1.0 progress bonus
        assert breakdown.progress_bonus == 1.0

    def test_low_confidence_falls_back_to_flat(self):
        calc = self._make_calc()
        from core.commands.command_registry import AttackPhase
        breakdown = calc.calculate_reward(
            template_name="nmap_basic",
            command="nmap -sV 192.168.1.1",
            success=True,
            raw_output="22/tcp open ssh",
            current_phase=AttackPhase.RECON,
            state_flags={},
            progress_delta=0.5,
            estimator_confidence=0.2,  # below 0.3 threshold
        )
        # Should fall back to flat
        assert breakdown.progress_bonus == 1.0

    def test_positive_delta_boosts_reward(self):
        calc = self._make_calc()
        from core.commands.command_registry import AttackPhase
        breakdown = calc.calculate_reward(
            template_name="nmap_basic",
            command="nmap -sV 192.168.1.1",
            success=True,
            raw_output="22/tcp open ssh",
            current_phase=AttackPhase.RECON,
            state_flags={},
            progress_delta=0.8,
            estimator_confidence=1.0,
        )
        # 0.8 * 3.0 = 2.4 (positive, clamped to [−1, 3])
        # Blended with flat at confidence=1.0 → pure shaped
        assert breakdown.progress_bonus > 1.0

    def test_negative_delta_penalizes(self):
        calc = self._make_calc()
        from core.commands.command_registry import AttackPhase
        breakdown = calc.calculate_reward(
            template_name="nmap_basic",
            command="nmap -sV 192.168.1.1",
            success=True,
            raw_output="22/tcp open ssh",
            current_phase=AttackPhase.RECON,
            state_flags={},
            progress_delta=-0.5,
            estimator_confidence=1.0,
        )
        # -0.5 * 3.0 = -1.5 → clamped to -1.0
        # Blended with flat at confidence=1.0 → negative
        assert breakdown.progress_bonus < 0


# ── Section 12: State Encoder Dims ──────────────────────────────────────────

class TestStateEncoderDims:
    """Test progress signals in state encoder dims 168-172."""

    def test_progress_dims_zero_by_default(self):
        from core.models.state_encoder import encode_state
        import torch
        vec = encode_state({}, torch.device("cpu"))
        # Dims 168, 169, 171 should be 0 by default
        for i in [168, 169, 171]:
            assert vec[i].item() == pytest.approx(0.0, abs=0.01), f"dim {i} not zero"
        # Dims 170, 172 (delta/momentum) remap [-1,1]→[0,1]: zero maps to 0.5
        assert vec[170].item() == pytest.approx(0.5, abs=0.01), "dim 170 (delta remap)"
        assert vec[172].item() == pytest.approx(0.5, abs=0.01), "dim 172 (momentum remap)"

    def test_progress_dims_populated(self):
        from core.models.state_encoder import encode_state
        import torch
        vec = encode_state(
            {},
            torch.device("cpu"),
            progress_foothold=0.7,
            progress_root=0.4,
            progress_delta=0.3,
            estimator_confidence=0.9,
            progress_momentum=0.1,
        )
        # Dim 168: foothold_progress = 0.7
        assert vec[168].item() == pytest.approx(0.7, abs=0.01)
        # Dim 169: root_progress = 0.4
        assert vec[169].item() == pytest.approx(0.4, abs=0.01)
        # Dim 170: progress_delta = 0.3 → remapped to 0.65
        assert vec[170].item() == pytest.approx(0.65, abs=0.01)
        # Dim 171: estimator_confidence = 0.9
        assert vec[171].item() == pytest.approx(0.9, abs=0.01)
        # Dim 172: progress_momentum = 0.1 → remapped to 0.55
        assert vec[172].item() == pytest.approx(0.55, abs=0.01)

    def test_state_dim_unchanged(self):
        from core.models.state_encoder import encode_state, STATE_DIM
        import torch
        vec = encode_state({}, torch.device("cpu"))
        assert vec.shape[0] == STATE_DIM
        assert STATE_DIM == 512


# ── Section 13: Budget Allocation ───────────────────────────────────────────

class TestBudgetAllocation:
    """Test Phase 16.0 budget reallocation."""

    def test_tier_budgets_sum(self):
        from core.llm.budget_manager import _TIER_BUDGETS, _TOTAL_BUDGET
        total = sum(_TIER_BUDGETS.values())
        assert total == _TOTAL_BUDGET, f"Tier budgets sum {total} != {_TOTAL_BUDGET}"

    def test_codex_increased(self):
        from core.llm.budget_manager import _TIER_BUDGETS, _TOTAL_BUDGET
        # Codex tier should be ~20% of total budget
        assert _TIER_BUDGETS["codex"] > 0
        assert _TIER_BUDGETS["codex"] <= _TOTAL_BUDGET * 0.25


# ── Section 14: End-to-End Integration ──────────────────────────────────────

class TestEndToEnd:
    """Test full progress estimation flow."""

    def test_heuristic_to_mlp_transition(self):
        """Bootstrap with heuristics, train MLP, verify MLP takes over."""
        from core.neuro.progress_estimator import ProgressEstimator, ProgressLabel
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            pe = ProgressEstimator(persist_dir=tmpdir)

            # Phase 1: Bootstrap with heuristic labels
            for ep in range(25):
                n_steps = 5
                states = [[float(ep * 10 + step)] * 512 for step in range(n_steps)]
                boards = []
                for step in range(n_steps):
                    db: dict = {"ports": set(range(min(step + 1, 5)))}
                    if step > 1:
                        db["services"] = {"ssh"}
                    if ep > 10 and step >= 2:
                        db["credentials"] = {"admin:pass"}
                    if ep > 15 and step >= 3:
                        db["shells"] = {"shell"}
                    boards.append(db)
                pe.label_episode_heuristic(states, boards, episode_id=f"ep{ep}")

            # Should have enough data for training
            assert pe.dataset_size >= 25 * 5

            # Train MLP
            metrics = pe.train_mlp()
            assert metrics["loss"] >= 0
            assert pe._mlp.is_trained

            # Confidence should be above 0 now
            assert pe.confidence > 0

    def test_persistence_across_sessions(self):
        """Dataset persists across ProgressEstimator instances."""
        from core.neuro.progress_estimator import ProgressEstimator, ProgressLabel
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # Session 1: Create and populate
            pe1 = ProgressEstimator(persist_dir=tmpdir)
            for i in range(5):
                pe1._dataset.add([float(i)] * 512, ProgressLabel(foothold_progress=i * 0.1))
            pe1.save()
            assert pe1.dataset_size == 5

            # Session 2: Reload
            pe2 = ProgressEstimator(persist_dir=tmpdir)
            assert pe2.dataset_size == 5
