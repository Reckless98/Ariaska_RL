#!/usr/bin/env python3
"""Tests for Phase 36.1 Fast-Learn Mode.

Covers:
  - FastLearnSnapshot dataclass serialization
  - ModelCallTracker rolling mix percentages
  - BudgetGovernorState improvement detection
  - FastLearnMetrics record_step + JSONL persistence
  - MicroChainCandidate quality_complete property
  - BudgetManagerV2 learning boost methods
"""

import json
import os
import tempfile

import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


# ── FastLearnSnapshot tests ────────────────────────────────────────────────


class TestFastLearnSnapshot:
    """Snapshot dataclass serialization."""

    def test_to_dict_defaults(self):
        from core.analytics.fast_learn_metrics import FastLearnSnapshot
        s = FastLearnSnapshot()
        d = s.to_dict()
        assert d["step_id"] == 0
        assert d["phase"] == ""
        assert d["value_pred"] == 0.0
        assert d["governor_learn_boost"] == 1.0
        assert "micro_chain_success_rate" in d

    def test_to_dict_round_trip(self):
        from core.analytics.fast_learn_metrics import FastLearnSnapshot
        s = FastLearnSnapshot(
            step_id=42, episode=3, phase="RECON",
            value_pred=0.567, td_error=0.123,
            model_nano_pct=0.70, model_mini_pct=0.25,
        )
        d = s.to_dict()
        assert d["step_id"] == 42
        assert d["episode"] == 3
        assert d["model_nano_pct"] == 0.70

    def test_to_dict_serializable(self):
        from core.analytics.fast_learn_metrics import FastLearnSnapshot
        s = FastLearnSnapshot(step_id=1, phase="EXPLOIT")
        text = json.dumps(s.to_dict())
        parsed = json.loads(text)
        assert parsed["step_id"] == 1


# ── ModelCallTracker tests ──────────────────────────────────────────────────


class TestModelCallTracker:
    """Rolling model-call mix percentages."""

    def test_empty_returns_zeros(self):
        from core.analytics.fast_learn_metrics import ModelCallTracker
        t = ModelCallTracker(window=10)
        mix = t.get_mix()
        assert mix["nano"] == 0.0
        assert mix["mini"] == 0.0
        assert mix["codex"] == 0.0

    def test_single_tier_100_pct(self):
        from core.analytics.fast_learn_metrics import ModelCallTracker
        t = ModelCallTracker(window=10)
        for _ in range(5):
            t.record("nano")
        mix = t.get_mix()
        assert mix["nano"] == pytest.approx(1.0)
        assert mix["mini"] == pytest.approx(0.0)

    def test_70_30_mix(self):
        from core.analytics.fast_learn_metrics import ModelCallTracker
        t = ModelCallTracker(window=100)
        for _ in range(70):
            t.record("nano")
        for _ in range(30):
            t.record("mini")
        mix = t.get_mix()
        assert mix["nano"] == pytest.approx(0.70)
        assert mix["mini"] == pytest.approx(0.30)
        assert mix["codex"] == pytest.approx(0.0)

    def test_rolling_window_eviction(self):
        from core.analytics.fast_learn_metrics import ModelCallTracker
        t = ModelCallTracker(window=5)
        # Fill with nano
        for _ in range(5):
            t.record("nano")
        assert t.get_mix()["nano"] == pytest.approx(1.0)
        # Now add 5 mini → pushes out all nano
        for _ in range(5):
            t.record("mini")
        assert t.get_mix()["mini"] == pytest.approx(1.0)
        assert t.get_mix()["nano"] == pytest.approx(0.0)

    def test_total_count(self):
        from core.analytics.fast_learn_metrics import ModelCallTracker
        t = ModelCallTracker(window=10)
        t.record("nano")
        t.record("mini")
        t.record("codex")
        assert t.total == 3

    def test_unknown_tier_maps_to_mini(self):
        from core.analytics.fast_learn_metrics import ModelCallTracker
        t = ModelCallTracker(window=5)
        t.record("unknown_model_tier")
        mix = t.get_mix()
        assert mix["mini"] == pytest.approx(1.0)

    def test_get_total_counts(self):
        from core.analytics.fast_learn_metrics import ModelCallTracker
        t = ModelCallTracker(window=10)
        t.record("nano")
        t.record("nano")
        t.record("mini")
        counts = t.get_total_counts()
        assert counts["nano"] == 2
        assert counts["mini"] == 1


# ── BudgetGovernorState tests ───────────────────────────────────────────────


class TestBudgetGovernorState:
    """Governor state dataclass."""

    def test_defaults(self):
        from core.analytics.fast_learn_metrics import BudgetGovernorState
        g = BudgetGovernorState()
        assert g.learn_boost_factor == 1.0
        assert g.window_cost_usd == 0.0

    def test_to_dict(self):
        from core.analytics.fast_learn_metrics import BudgetGovernorState
        g = BudgetGovernorState(learn_boost_factor=1.3, window_cost_usd=2.50)
        d = g.to_dict()
        assert d["learn_boost_factor"] == 1.3
        assert d["window_cost_usd"] == 2.50


# ── FastLearnMetrics tests ──────────────────────────────────────────────────


class TestFastLearnMetrics:
    """Full metrics collector: recording, persistence, governor."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        from core.analytics.fast_learn_metrics import FastLearnMetrics
        self.log_dir = str(tmp_path)
        self.flm = FastLearnMetrics(log_dir=self.log_dir, log_every=5)
        self.flm.reset_episode(episode=0)

    def teardown_method(self):
        self.flm.close()

    def test_record_step_returns_none_before_log_every(self):
        result = self.flm.record_step(step=1, phase="RECON", template="nmap_basic")
        assert result is None

    def test_record_step_persists_at_log_every(self):
        for i in range(1, 6):
            result = self.flm.record_step(
                step=i, phase="RECON", template="nmap_basic",
                value_pred=0.3, step_cost=0.05,
            )
        assert result is not None
        assert result.step_id == 5
        assert result.phase == "RECON"
        assert result.value_pred == pytest.approx(0.3, abs=0.01)

    def test_jsonl_written(self):
        for i in range(1, 6):
            self.flm.record_step(
                step=i, phase="RECON", template="nmap",
                step_cost=0.10,
            )
        self.flm.close()
        path = os.path.join(self.log_dir, "learn_metrics.jsonl")
        assert os.path.exists(path)
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) >= 1
        record = json.loads(lines[-1])
        assert record["type"] == "fast_learn_step"
        assert record["step_id"] == 5

    def test_cumulative_cost_tracks(self):
        for i in range(1, 4):
            self.flm.record_step(step=i, phase="RECON", step_cost=1.0)
        assert self.flm.cumulative_cost == pytest.approx(3.0)

    def test_hallucination_counting(self):
        self.flm.record_step(step=1, phase="RECON", hallucination_flags=2)
        self.flm.record_step(step=2, phase="RECON", hallucination_flags=1)
        snap = self.flm.get_latest_snapshot()
        assert snap is not None
        assert snap.hallucination_flags_count == 3

    def test_contradiction_counting(self):
        self.flm.record_step(step=1, phase="RECON", contradictions=3)
        snap = self.flm.get_latest_snapshot()
        assert snap is not None
        assert snap.contradictions_count == 3

    def test_mc_success_rate_rolling(self):
        self.flm.record_step(step=1, phase="RECON", mc_success=True)
        self.flm.record_step(step=2, phase="RECON", mc_success=True)
        self.flm.record_step(step=3, phase="RECON", mc_success=False)
        snap = self.flm.get_latest_snapshot()
        # 2/3 ≈ 0.6667
        assert snap.micro_chain_success_rate == pytest.approx(0.6667, abs=0.01)

    def test_model_call_recording(self):
        self.flm.record_model_call("nano")
        self.flm.record_model_call("nano")
        self.flm.record_model_call("mini")
        mix = self.flm.get_model_mix()
        assert mix["nano"] == pytest.approx(2 / 3, abs=0.01)
        assert mix["mini"] == pytest.approx(1 / 3, abs=0.01)

    def test_ppo_stats_cache(self):
        self.flm.record_ppo_stats(
            value_pred=0.5, value_target=0.8,
            advantage_mean=0.12, advantage_std=0.05,
        )
        self.flm.record_step(step=1, phase="RECON")
        snap = self.flm.get_latest_snapshot()
        assert snap.value_pred == pytest.approx(0.5, abs=0.01)
        assert snap.td_error == pytest.approx(0.3, abs=0.01)  # |0.8-0.5|
        assert snap.advantage_mean == pytest.approx(0.12, abs=0.01)

    def test_dashboard_line_waiting(self):
        line = self.flm.get_dashboard_line()
        assert "waiting" in line

    def test_dashboard_line_after_steps(self):
        self.flm.record_model_call("nano")
        self.flm.record_step(step=1, phase="RECON", value_pred=0.32)
        line = self.flm.get_dashboard_line()
        assert "FL:" in line
        assert "RECON" in line
        assert "V=" in line

    def test_reset_episode_clears_state(self):
        self.flm.record_step(step=1, phase="RECON", step_cost=1.0, hallucination_flags=5)
        self.flm.reset_episode(episode=1)
        assert self.flm.cumulative_cost == pytest.approx(0.0)
        snap = self.flm.get_latest_snapshot()
        assert snap is None

    def test_distillation_score_clamped(self):
        self.flm.record_distillation_score(1.5)  # Should clamp to 1.0
        self.flm.record_distillation_score(-0.5)  # Should clamp to 0.0
        self.flm.record_step(step=1, phase="RECON")
        snap = self.flm.get_latest_snapshot()
        # Average of 1.0 and 0.0 = 0.5
        assert snap.distillation_packet_score == pytest.approx(0.5, abs=0.01)


# ── Governor Logic tests ───────────────────────────────────────────────────


class TestBudgetGovernor:
    """Budget governor adaptive evaluation."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        from core.analytics.fast_learn_metrics import FastLearnMetrics
        self.flm = FastLearnMetrics(log_dir=str(tmp_path), log_every=5)
        self.flm.reset_episode(episode=0)

    def teardown_method(self):
        self.flm.close()

    def test_governor_no_eval_before_window(self):
        """Governor should not change before 40 steps."""
        self.flm.record_step(step=10, phase="RECON")
        gov = self.flm.evaluate_governor(step=10)
        assert gov.learn_boost_factor == 1.0

    def test_governor_boost_on_improvement(self):
        """When >= 2 of 5 criteria improve, boost goes up."""
        # Seed enough steps for evaluation
        for i in range(1, 41):
            self.flm.record_step(
                step=i, phase="RECON", mc_success=True,
                distillation_score=0.8, step_cost=0.05,
            )
        # First eval at step 40 — prev baselines are all 0, so improvement detected
        gov = self.flm.evaluate_governor(step=40)
        assert gov.learn_boost_factor > 1.0

    def test_governor_tighten_on_overbudget_no_improvement(self):
        """Over budget without improvement tightens boost."""
        for i in range(1, 41):
            self.flm.record_step(
                step=i, phase="RECON", mc_success=False,
                step_cost=0.25,  # $10 over 40 steps >> $3 target
            )
        # First eval — mc_success is all False, but prev was also 0 → no improvement
        # Need to establish baseline first
        gov = self.flm.evaluate_governor(step=40)
        # Baseline established with mc_rate=0.0 and high cost
        # Now run another window with same bad metrics
        for i in range(41, 81):
            self.flm.record_step(
                step=i, phase="RECON", mc_success=False,
                step_cost=0.25,
            )
        gov = self.flm.evaluate_governor(step=80)
        # Over budget ($10 >> $3) without improvement → should tighten
        assert gov.learn_boost_factor < 1.0

    def test_governor_boost_capped_at_2(self):
        """Boost factor cannot exceed 2.0."""
        from core.analytics.fast_learn_metrics import BudgetGovernorState
        # Manually set high boost, check it stays ≤ 2.0
        self.flm._governor.learn_boost_factor = 1.9
        for i in range(1, 41):
            self.flm.record_step(
                step=i, phase="RECON", mc_success=True,
                distillation_score=0.9,
            )
        gov = self.flm.evaluate_governor(step=40)
        assert gov.learn_boost_factor <= 2.0

    def test_governor_boost_floor_at_half(self):
        """Boost factor cannot go below 0.5."""
        self.flm._governor.learn_boost_factor = 0.6
        for i in range(1, 41):
            self.flm.record_step(
                step=i, phase="RECON", mc_success=False,
                step_cost=0.25,  # Over budget
            )
        # Baseline eval
        self.flm.evaluate_governor(step=40)
        for i in range(41, 81):
            self.flm.record_step(
                step=i, phase="RECON", mc_success=False,
                step_cost=0.25,
            )
        gov = self.flm.evaluate_governor(step=80)
        assert gov.learn_boost_factor >= 0.5

    def test_governor_state_serializable(self):
        gov = self.flm.get_governor_state()
        text = json.dumps(gov.to_dict())
        parsed = json.loads(text)
        assert "learn_boost_factor" in parsed


# ── MicroChainCandidate quality_complete tests ──────────────────────────────


class TestMicroChainCandidateQuality:
    """P36.1: Structured quality fields on MicroChainCandidate."""

    def test_quality_complete_all_filled(self):
        from core.llm.micro_chain import MicroChainCandidate
        c = MicroChainCandidate(
            command="nmap -sV 10.10.10.1",
            template_name="nmap_full",
            evidence_used=["port_22_open"],
            hypothesis="SSH service vulnerable to brute force",
            test="nmap_full",
            expected_observable="version banner",
            stop_condition="version detected or timeout",
            confidence=0.8,
        )
        assert c.quality_complete is True

    def test_quality_incomplete_missing_hypothesis(self):
        from core.llm.micro_chain import MicroChainCandidate
        c = MicroChainCandidate(
            command="nmap -sV 10.10.10.1",
            evidence_used=["port_22"],
            hypothesis="",  # Missing
            test="nmap_full",
            expected_observable="banner",
            stop_condition="done",
            confidence=0.5,
        )
        assert c.quality_complete is False

    def test_quality_incomplete_no_evidence(self):
        from core.llm.micro_chain import MicroChainCandidate
        c = MicroChainCandidate(
            command="nmap -sV 10.10.10.1",
            evidence_used=[],  # Empty
            hypothesis="test",
            test="nmap_full",
            expected_observable="banner",
            stop_condition="done",
            confidence=0.5,
        )
        assert c.quality_complete is False

    def test_quality_incomplete_zero_confidence(self):
        from core.llm.micro_chain import MicroChainCandidate
        c = MicroChainCandidate(
            command="nmap -sV 10.10.10.1",
            evidence_used=["port_80"],
            hypothesis="test",
            test="nmap_full",
            expected_observable="banner",
            stop_condition="done",
            confidence=0.0,  # Zero → incomplete
        )
        assert c.quality_complete is False

    def test_quality_uses_template_name_fallback(self):
        from core.llm.micro_chain import MicroChainCandidate
        c = MicroChainCandidate(
            command="nmap -sV 10.10.10.1",
            template_name="nmap_full",
            evidence_used=["port_22"],
            hypothesis="scan for versions",
            test="",  # Empty, but template_name is set
            expected_observable="version banner",
            stop_condition="scan completes",
            confidence=0.7,
        )
        assert c.quality_complete is True  # test="" but template_name="nmap_full"

    def test_to_dict_includes_quality_fields(self):
        from core.llm.micro_chain import MicroChainCandidate
        c = MicroChainCandidate(
            command="nmap -sV 10.10.10.1",
            template_name="nmap_full",
            evidence_used=["port_22", "service_ssh"],
            hypothesis="SSH version detection",
            expected_observable="OpenSSH banner",
            stop_condition="version found",
            confidence=0.85,
        )
        d = c.to_dict()
        assert d["evidence_used"] == ["port_22", "service_ssh"]
        assert d["hypothesis"] == "SSH version detection"
        assert d["confidence"] == 0.85
        assert d["test"] == "nmap_full"  # Falls back to template_name


# ── BudgetManagerV2 learning boost tests ────────────────────────────────────


class TestBudgetManagerLearnBoost:
    """P36.1: Learning-acceleration ROI tag budget methods."""

    def test_learning_accel_tags_exist(self):
        from core.llm.budget_manager import LEARNING_ACCEL_ROI_TAGS
        assert "micro_chain_scoring" in LEARNING_ACCEL_ROI_TAGS
        assert "distillation_quality" in LEARNING_ACCEL_ROI_TAGS
        assert "reward_shaping" in LEARNING_ACCEL_ROI_TAGS
        assert len(LEARNING_ACCEL_ROI_TAGS) >= 9

    def test_is_learning_accel_tag_true(self):
        from core.llm.budget_manager import BudgetManagerV2
        bm = BudgetManagerV2()
        assert bm.is_learning_accel_tag("micro_chain_scoring") is True
        assert bm.is_learning_accel_tag("value_target_shaping") is True

    def test_is_learning_accel_tag_false(self):
        from core.llm.budget_manager import BudgetManagerV2
        bm = BudgetManagerV2()
        assert bm.is_learning_accel_tag("postmortem") is False
        assert bm.is_learning_accel_tag("classification") is False
        assert bm.is_learning_accel_tag("") is False

    def test_get_effective_tokens_no_boost(self):
        from core.llm.budget_manager import BudgetManagerV2
        bm = BudgetManagerV2()
        # Default boost = 1.0, no reduction
        effective = bm.get_effective_tokens(1000, "micro_chain_scoring")
        assert effective == 1000

    def test_get_effective_tokens_with_boost(self):
        from core.llm.budget_manager import BudgetManagerV2
        bm = BudgetManagerV2()
        bm.set_learn_boost_factor(2.0)
        effective = bm.get_effective_tokens(1000, "micro_chain_scoring")
        assert effective == 500  # 1000 / 2.0

    def test_get_effective_tokens_non_accel_tag_unchanged(self):
        from core.llm.budget_manager import BudgetManagerV2
        bm = BudgetManagerV2()
        bm.set_learn_boost_factor(2.0)
        effective = bm.get_effective_tokens(1000, "postmortem")
        assert effective == 1000  # Not a learning-accel tag

    def test_set_learn_boost_clamped(self):
        from core.llm.budget_manager import BudgetManagerV2
        bm = BudgetManagerV2()
        bm.set_learn_boost_factor(5.0)
        assert bm._learn_boost_factor == 2.0  # max
        bm.set_learn_boost_factor(0.1)
        assert bm._learn_boost_factor == 0.5  # min

    def test_effective_tokens_at_least_1(self):
        from core.llm.budget_manager import BudgetManagerV2
        bm = BudgetManagerV2()
        bm.set_learn_boost_factor(2.0)
        effective = bm.get_effective_tokens(1, "micro_chain_scoring")
        assert effective >= 1


# ── MicroChain model tracking tests ─────────────────────────────────────────


class TestMicroChainModelTracking:
    """P36.1: Model call mix tracking on MicroChain."""

    def test_model_call_mix_property(self):
        from core.llm.micro_chain import MicroChain

        class _DummyGPT:
            def can_make_request(self, **kw): return False

        mc = MicroChain(_DummyGPT())  # type: ignore[arg-type]
        mix = mc.model_call_mix
        assert "nano" in mix
        assert "mini" in mix

    def test_nano_ratio_default_zero(self):
        from core.llm.micro_chain import MicroChain

        class _DummyGPT:
            def can_make_request(self, **kw): return False

        mc = MicroChain(_DummyGPT())  # type: ignore[arg-type]
        assert mc.nano_ratio == 0.0

    def test_nano_ratio_after_calls(self):
        from core.llm.micro_chain import MicroChain

        class _DummyGPT:
            def can_make_request(self, **kw): return False

        mc = MicroChain(_DummyGPT())  # type: ignore[arg-type]
        mc._nano_calls = 7
        mc._mini_calls = 3
        assert mc.nano_ratio == pytest.approx(0.7)
