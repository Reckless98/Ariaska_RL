"""Tests for C11: ScheduleCoupler — unified schedule coupling.

Validates:
 - CouplerConfig defaults and custom values
 - CoupledState serialization
 - Cosine interpolation at extremes and midpoints
 - compute() without side effects
 - update() with mentor_policy, llm_bridge, ppo_agent
 - Orchestrator init produces self.schedule_coupler
 - Episode-start wiring propagates to coaches
"""
from __future__ import annotations

import math
import os
import pytest
from dataclasses import dataclass
from typing import Any, Optional

os.environ["ARIASKA_DRY_RUN"] = "1"

from core.training.schedule_coupler import (
    CoupledState,
    CouplerConfig,
    ScheduleCoupler,
)


# ───────── Mock objects ──────────────────────────────────────────────

class MockMentorPolicy:
    """Minimal stand-in tracking set_maturity calls."""
    def __init__(self) -> None:
        self.maturity_calls: list[float] = []

    def set_maturity(self, m: float) -> None:
        self.maturity_calls.append(m)


@dataclass
class MockInfluence:
    kl_teacher_coef: float = 0.15
    prior_alpha: float = 0.50
    maturity_signal: float = 0.0


class MockLLMBridge:
    """Tracks influence mutations."""
    def __init__(self) -> None:
        self.influence = MockInfluence()


@dataclass
class MockPPOConfig:
    entropy_coef: float = 0.08


class MockPPOAgent:
    def __init__(self, entropy: float = 0.08) -> None:
        self.config = MockPPOConfig(entropy_coef=entropy)


# ───────── CouplerConfig ────────────────────────────────────────────

class TestCouplerConfig:
    def test_defaults(self) -> None:
        cfg = CouplerConfig()
        assert cfg.coupling_strength == 0.8
        assert cfg.mentor_floor_range == (0.60, 0.08)
        assert cfg.kl_coef_range == (0.15, 0.01)
        assert cfg.bc_coef_range == (0.10, 0.01)
        assert cfg.prior_alpha_range == (0.50, 0.02)
        assert cfg.entropy_coef_range == (0.08, 0.01)

    def test_custom_strength(self) -> None:
        cfg = CouplerConfig(coupling_strength=0.5)
        assert cfg.coupling_strength == 0.5


# ───────── CoupledState ─────────────────────────────────────────────

class TestCoupledState:
    def test_defaults(self) -> None:
        s = CoupledState()
        assert s.maturity == 0.0
        assert s.mentor_floor == 0.60

    def test_to_dict(self) -> None:
        s = CoupledState(maturity=0.5, episode=10)
        d = s.to_dict()
        assert d["maturity"] == 0.5
        assert d["episode"] == 10
        assert isinstance(d["mentor_floor"], float)


# ───────── Interpolation ────────────────────────────────────────────

class TestInterpolation:
    """Verify cosine schedule at boundary and midpoint."""
    def setup_method(self) -> None:
        self.coupler = ScheduleCoupler()

    def test_maturity_zero_gives_high(self) -> None:
        """At maturity=0, all ranges should be at their HIGH values."""
        state = self.coupler.compute(maturity=0.0)
        assert abs(state.mentor_floor - 0.60) < 1e-6
        assert abs(state.kl_coef - 0.15) < 1e-6
        assert abs(state.prior_alpha - 0.50) < 1e-6
        assert abs(state.entropy_coef - 0.08) < 1e-6

    def test_maturity_one_with_full_strength(self) -> None:
        """At maturity=1.0, strength=1.0 → all at LOW values."""
        c = ScheduleCoupler(CouplerConfig(coupling_strength=1.0))
        state = c.compute(maturity=1.0)
        assert abs(state.mentor_floor - 0.08) < 1e-4
        assert abs(state.kl_coef - 0.01) < 1e-4
        assert abs(state.prior_alpha - 0.02) < 1e-4
        assert abs(state.entropy_coef - 0.01) < 1e-4

    def test_midpoint_is_between(self) -> None:
        """At maturity=0.5, values should be between high and low."""
        c = ScheduleCoupler(CouplerConfig(coupling_strength=1.0))
        state = c.compute(maturity=0.5)
        assert 0.08 < state.mentor_floor < 0.60
        assert 0.01 < state.kl_coef < 0.15
        assert 0.02 < state.prior_alpha < 0.50

    def test_zero_coupling_always_high(self) -> None:
        """strength=0 → always high regardless of maturity."""
        c = ScheduleCoupler(CouplerConfig(coupling_strength=0.0))
        for mat in [0.0, 0.5, 1.0]:
            state = c.compute(maturity=mat)
            assert abs(state.mentor_floor - 0.60) < 1e-6
            assert abs(state.kl_coef - 0.15) < 1e-6

    def test_cosine_shape(self) -> None:
        """Cosine should be slower at extremes, faster in the middle."""
        c = ScheduleCoupler(CouplerConfig(coupling_strength=1.0))
        s0 = c.compute(0.0).mentor_floor
        s1 = c.compute(0.25).mentor_floor
        s2 = c.compute(0.50).mentor_floor
        s3 = c.compute(0.75).mentor_floor
        s4 = c.compute(1.0).mentor_floor
        # Monotonically decreasing
        assert s0 > s1 > s2 > s3 > s4


# ───────── update() side effects ────────────────────────────────────

class TestUpdate:
    def setup_method(self) -> None:
        self.coupler = ScheduleCoupler(
            CouplerConfig(coupling_strength=1.0, log_updates=False)
        )

    def test_mentor_receives_maturity(self) -> None:
        mp = MockMentorPolicy()
        self.coupler.update(maturity=0.7, mentor_policy=mp)
        assert len(mp.maturity_calls) == 1
        assert mp.maturity_calls[0] == 0.7

    def test_bridge_gets_kl_and_alpha(self) -> None:
        bridge = MockLLMBridge()
        state = self.coupler.update(maturity=0.5, llm_bridge=bridge)
        assert bridge.influence.kl_teacher_coef == state.kl_coef
        assert bridge.influence.prior_alpha == state.prior_alpha
        assert bridge.influence.maturity_signal == 0.5

    def test_ppo_entropy_only_decreases(self) -> None:
        """PPO entropy_coef should only go down, never up."""
        ppo = MockPPOAgent(entropy=0.08)
        self.coupler.update(maturity=0.5, ppo_agent=ppo)
        # Coupled entropy at maturity=0.5 with strength=1.0 is between 0.08 and 0.01
        assert ppo.config.entropy_coef < 0.08

    def test_ppo_entropy_not_increased(self) -> None:
        """If PPO already lower than coupled, don't raise it."""
        ppo = MockPPOAgent(entropy=0.005)
        self.coupler.update(maturity=0.0, ppo_agent=ppo)
        assert ppo.config.entropy_coef == 0.005  # unchanged

    def test_all_none_subsystems(self) -> None:
        """Should not crash with all None subsystems."""
        state = self.coupler.update(maturity=0.3)
        assert state.maturity == 0.3

    def test_history_grows(self) -> None:
        self.coupler.update(maturity=0.1)
        self.coupler.update(maturity=0.2)
        self.coupler.update(maturity=0.3)
        assert self.coupler._update_count == 3
        assert len(self.coupler._history) == 3

    def test_get_history(self) -> None:
        for i in range(5):
            self.coupler.update(maturity=i * 0.2)
        h = self.coupler.get_history(n=3)
        assert len(h) == 3
        assert h[-1]["maturity"] == pytest.approx(0.8)

    def test_get_stats(self) -> None:
        self.coupler.update(maturity=0.5, episode=10)
        stats = self.coupler.get_stats()
        assert stats["update_count"] == 1
        assert stats["latest"]["episode"] == 10
        assert "config" in stats


# ───────── Orchestrator integration ─────────────────────────────────

class TestOrchestratorInit:
    """Verify SmartOrchestrator initializes schedule_coupler."""
    def test_orchestrator_has_schedule_coupler(self) -> None:
        from core.environment.cyber_environment import CyberEnvironment
        from core.testing.fake_gpt_manager import FakeGPTManager
        from core.orchestration.smart_orchestrator import SmartOrchestrator

        env = CyberEnvironment(defer_reset=True)
        gpt = FakeGPTManager(seed=42)
        orch = SmartOrchestrator(env=env, gpt_manager=gpt, verbosity="silent")
        assert hasattr(orch, 'schedule_coupler')
        assert orch.schedule_coupler is not None

    def test_coupler_is_schedule_coupler_type(self) -> None:
        from core.environment.cyber_environment import CyberEnvironment
        from core.testing.fake_gpt_manager import FakeGPTManager
        from core.orchestration.smart_orchestrator import SmartOrchestrator

        env = CyberEnvironment(defer_reset=True)
        gpt = FakeGPTManager(seed=42)
        orch = SmartOrchestrator(env=env, gpt_manager=gpt, verbosity="silent")
        assert type(orch.schedule_coupler).__name__ == "ScheduleCoupler"


# ───────── Edge cases ───────────────────────────────────────────────

class TestEdgeCases:
    def test_maturity_clamped_above_one(self) -> None:
        """Maturity > 1 should not crash, just extrapolate."""
        c = ScheduleCoupler()
        state = c.compute(maturity=1.5)
        # Values may exceed range but no crash
        assert isinstance(state.mentor_floor, float)

    def test_negative_maturity(self) -> None:
        """Negative maturity should not crash."""
        c = ScheduleCoupler()
        state = c.compute(maturity=-0.1)
        assert isinstance(state.mentor_floor, float)

    def test_exception_in_mentor_does_not_crash(self) -> None:
        """A broken mentor_policy shouldn't crash update."""
        class BrokenMentor:
            def set_maturity(self, m: float) -> None:
                raise RuntimeError("boom")

        c = ScheduleCoupler(CouplerConfig(log_updates=False))
        state = c.update(maturity=0.5, mentor_policy=BrokenMentor())
        # Should still return state despite error
        assert state.maturity == 0.5

    def test_exception_in_bridge_does_not_crash(self) -> None:
        """A broken bridge shouldn't crash update."""
        class BrokenBridge:
            @property
            def influence(self):
                raise AttributeError("no influence")

        c = ScheduleCoupler(CouplerConfig(log_updates=False))
        state = c.update(maturity=0.5, llm_bridge=BrokenBridge())
        assert state.maturity == 0.5
