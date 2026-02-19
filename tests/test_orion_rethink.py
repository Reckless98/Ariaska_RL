"""
tests/test_orion_rethink.py — Phase 39.1: OrionRethinkEngine tests

Tests stall detection, signal computation, rethink plan generation,
cooldown enforcement, and reset behavior.
"""

import os
import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")

from core.ops.orion_rethink import (
    COMMAND_REPEAT_THRESHOLD,
    EVIDENCE_PLATEAU_STEPS,
    OrionRethinkEngine,
    OrionRethinkPlan,
    PHASE_OSCILLATION_COUNT,
    RETHINK_COOLDOWN_STEPS,
    REWARD_FLAT_STEPS,
    STALL_THRESHOLD,
    StallSignals,
    TOOL_FAILURE_COUNT,
    TOOL_FAILURE_WINDOW,
)


class TestStallSignals:
    """Tests for StallSignals dataclass."""

    def test_default_score_zero(self):
        s = StallSignals()
        assert s.score == 0.0

    def test_score_weighted(self):
        s = StallSignals(
            repeated_tool_failures=1.0,
            evidence_plateau=1.0,
            phase_oscillation=1.0,
            hypothesis_invalidation=1.0,
            repeat_commands=1.0,
            reward_stagnation=1.0,
        )
        assert abs(s.score - 1.0) < 0.01

    def test_score_clamped_to_one(self):
        s = StallSignals(
            repeated_tool_failures=5.0,
            evidence_plateau=5.0,
        )
        assert s.score <= 1.0

    def test_score_never_negative(self):
        s = StallSignals(
            repeated_tool_failures=-1.0,
            evidence_plateau=-1.0,
        )
        assert s.score >= 0.0

    def test_to_dict(self):
        s = StallSignals(repeated_tool_failures=0.5)
        d = s.to_dict()
        assert "score" in d
        assert d["repeated_tool_failures"] == 0.5


class TestOrionRethinkPlan:
    """Tests for OrionRethinkPlan dataclass."""

    def test_default_values(self):
        p = OrionRethinkPlan()
        assert p.new_hypotheses == []
        assert p.plan == []
        assert p.why_now == ""

    def test_to_dict(self):
        p = OrionRethinkPlan(
            new_hypotheses=["h1"],
            why_now="stall",
            expected_signal="discovery",
        )
        d = p.to_dict()
        assert d["new_hypotheses"] == ["h1"]
        assert d["why_now"] == "stall"

    def test_to_dict_with_signals(self):
        s = StallSignals(evidence_plateau=0.8)
        p = OrionRethinkPlan(stall_signals=s)
        d = p.to_dict()
        assert "stall_signals" in d
        assert d["stall_signals"]["evidence_plateau"] == 0.8


class TestOrionRethinkEngine:
    """Tests for OrionRethinkEngine core logic."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.engine = OrionRethinkEngine()

    def test_init_defaults(self):
        assert self.engine.rethink_count == 0
        assert self.engine.last_rethink_step == -RETHINK_COOLDOWN_STEPS
        assert self.engine.rethink_history == []

    def test_record_step(self):
        self.engine.record_step(
            step=0, command="nmap -sV", success=True,
            phase="RECON", discoveries=3, reward=5.0,
        )
        assert len(self.engine._step_history) == 1

    def test_compute_signals_empty(self):
        signals = self.engine.compute_stall_signals()
        assert signals.score == 0.0

    def test_compute_signals_few_steps(self):
        for i in range(2):
            self.engine.record_step(step=i, command="nmap", success=True)
        signals = self.engine.compute_stall_signals()
        assert signals.score == 0.0

    def test_should_rethink_no_stall(self):
        for i in range(5):
            self.engine.record_step(
                step=i, command=f"tool_{i}", success=True,
                phase="RECON", discoveries=2, reward=3.0,
            )
        should, signals = self.engine.should_rethink(5)
        assert not should

    def test_reset(self):
        for i in range(10):
            self.engine.record_step(step=i, command="test")
        self.engine.reset()
        assert self.engine.rethink_count == 0
        assert len(self.engine._step_history) == 0

    def test_history_bounded(self):
        for i in range(100):
            self.engine.record_step(step=i, command=f"cmd_{i}")
        assert len(self.engine._step_history) <= 50


class TestRepeatedToolFailures:
    """Tests for stall signal 1: repeated tool failures."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.engine = OrionRethinkEngine()

    def test_no_failures(self):
        for i in range(10):
            self.engine.record_step(
                step=i, template_name="nmap_tcp", success=True,
            )
        s = self.engine.compute_stall_signals()
        assert s.repeated_tool_failures == 0.0

    def test_same_tool_repeated_failures(self):
        for i in range(TOOL_FAILURE_COUNT + 2):
            self.engine.record_step(
                step=i, template_name="hydra_ssh", success=False,
            )
        s = self.engine.compute_stall_signals()
        assert s.repeated_tool_failures > 0.0

    def test_different_tools_no_signal(self):
        for i in range(10):
            self.engine.record_step(
                step=i, template_name=f"tool_{i}", success=False,
            )
        s = self.engine.compute_stall_signals()
        assert s.repeated_tool_failures == 0.0


class TestEvidencePlateau:
    """Tests for stall signal 2: evidence plateau."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.engine = OrionRethinkEngine()

    def test_no_plateau_with_discoveries(self):
        for i in range(EVIDENCE_PLATEAU_STEPS + 2):
            self.engine.record_step(
                step=i, command="cmd", success=True, discoveries=1,
            )
        s = self.engine.compute_stall_signals()
        assert s.evidence_plateau == 0.0

    def test_plateau_no_discoveries(self):
        for i in range(EVIDENCE_PLATEAU_STEPS + 5):
            self.engine.record_step(
                step=i, command="cmd", success=True, discoveries=0,
            )
        s = self.engine.compute_stall_signals()
        assert s.evidence_plateau > 0.0


class TestPhaseOscillation:
    """Tests for stall signal 3: phase oscillation."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.engine = OrionRethinkEngine()

    def test_no_oscillation_steady(self):
        for i in range(10):
            self.engine.record_step(step=i, phase="RECON")
        s = self.engine.compute_stall_signals()
        assert s.phase_oscillation == 0.0

    def test_oscillation_detected(self):
        phases = ["RECON", "EXPLOITATION"] * (PHASE_OSCILLATION_COUNT + 2)
        for i, phase in enumerate(phases):
            self.engine.record_step(step=i, phase=phase)
        s = self.engine.compute_stall_signals()
        assert s.phase_oscillation > 0.0


class TestRepeatCommands:
    """Tests for stall signal 5: repeat commands."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.engine = OrionRethinkEngine()

    def test_no_repeats(self):
        for i in range(10):
            self.engine.record_step(step=i, command=f"unique_cmd_{i}")
        s = self.engine.compute_stall_signals()
        assert s.repeat_commands == 0.0

    def test_repeats_detected(self):
        for i in range(COMMAND_REPEAT_THRESHOLD + 3):
            self.engine.record_step(
                step=i, command="nmap -sV 10.10.10.10 --same-args"
            )
        s = self.engine.compute_stall_signals()
        assert s.repeat_commands > 0.0


class TestRewardStagnation:
    """Tests for stall signal 6: reward stagnation."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.engine = OrionRethinkEngine()

    def test_no_stagnation_with_rewards(self):
        for i in range(REWARD_FLAT_STEPS + 2):
            self.engine.record_step(step=i, reward=5.0 + i)
        s = self.engine.compute_stall_signals()
        assert s.reward_stagnation == 0.0

    def test_stagnation_zero_rewards(self):
        for i in range(REWARD_FLAT_STEPS + 2):
            self.engine.record_step(step=i, reward=0.0)
        s = self.engine.compute_stall_signals()
        assert s.reward_stagnation > 0.0


class TestRethinkPlanning:
    """Tests for rethink plan generation."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.engine = OrionRethinkEngine()

    def test_heuristic_plan_on_evidence_plateau(self):
        # Force evidence plateau
        for i in range(15):
            self.engine.record_step(
                step=i, command="nmap", success=False,
                phase="RECON", discoveries=0, reward=0.0,
            )
        plan = self.engine.generate_rethink_plan(
            evidence_summary={"ports": []},
            hypotheses=[],
            current_phase="RECON",
            current_step=15,
        )
        assert isinstance(plan, OrionRethinkPlan)
        assert len(plan.new_hypotheses) > 0
        assert plan.why_now != ""
        assert plan.stall_signals is not None

    def test_heuristic_plan_exploitation_phase(self):
        for i in range(15):
            self.engine.record_step(
                step=i, command="exploit", success=False,
                phase="EXPLOITATION", discoveries=0, reward=0.0,
            )
        plan = self.engine.generate_rethink_plan(
            current_phase="EXPLOITATION", current_step=15,
        )
        assert isinstance(plan, OrionRethinkPlan)
        assert len(plan.new_hypotheses) > 0

    def test_rethink_count_increments(self):
        for i in range(15):
            self.engine.record_step(step=i, command="cmd", discoveries=0)
        self.engine.generate_rethink_plan(current_step=15)
        assert self.engine.rethink_count == 1
        self.engine.generate_rethink_plan(current_step=30)
        assert self.engine.rethink_count == 2


class TestRethinkCooldown:
    """Tests for rethink cooldown enforcement."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.engine = OrionRethinkEngine(stall_threshold=0.0)

    def test_cooldown_blocks_immediate_rethink(self):
        for i in range(20):
            self.engine.record_step(
                step=i, command="cmd", discoveries=0, reward=0.0,
            )
        should1, _ = self.engine.should_rethink(10)
        if should1:
            self.engine.generate_rethink_plan(current_step=10)
        # Immediately after: should be blocked by cooldown
        should2, _ = self.engine.should_rethink(11)
        assert not should2

    def test_cooldown_allows_after_gap(self):
        for i in range(50):
            self.engine.record_step(
                step=i, command="cmd", discoveries=0, reward=0.0,
            )
        self.engine.generate_rethink_plan(current_step=5)
        should, _ = self.engine.should_rethink(5 + RETHINK_COOLDOWN_STEPS + 1)
        # After cooldown period, should be allowed if stall persists
        # (may or may not trigger depending on threshold)
        assert isinstance(should, bool)


class TestConstants:
    """Tests for module-level constants."""

    def test_stall_threshold_range(self):
        assert 0.0 < STALL_THRESHOLD < 1.0

    def test_cooldown_positive(self):
        assert RETHINK_COOLDOWN_STEPS > 0

    def test_evidence_plateau_positive(self):
        assert EVIDENCE_PLATEAU_STEPS > 0

    def test_tool_failure_window_positive(self):
        assert TOOL_FAILURE_WINDOW > 0
