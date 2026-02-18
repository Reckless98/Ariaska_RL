#!/usr/bin/env python3
"""
tests/test_phase15_wiring.py — T15.2 Wiring Integration Tests

Validates:
1. SmartCoach P15 init: components present when flags ON, absent when OFF
2. PPO apply_neuromodulation hook: bounded, no crash
3. Arbitrator flag gating in decide() path
4. Sensory buffer push integration
5. CAP contract: flags OFF = zero behavior change

Phase 15.0 — Neurovortex
"""

import os
import pytest
import math

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


# ── Helpers ─────────────────────────────────────────────────────────────────

def _make_fake_gpt():
    from core.testing.fake_gpt_manager import FakeGPTManager
    return FakeGPTManager(seed=42)


def _make_ppo_agent():
    """Create a minimal PPO agent for testing."""
    from core.algorithms.ppo_agent import PPOAgent, PPOConfig
    config = PPOConfig(
        state_dim=512,
        action_dim=5,
        hidden_dims=[64, 64],
        rollout_size=8,
        minibatch_size=4,
        epochs_per_update=1,
    )
    return PPOAgent(config=config)


def _make_coach(agent_name="TestAgent"):
    """Create a SmartCoach with correct constructor signature."""
    gpt = _make_fake_gpt()
    from core.training.smart_coach import SmartCoach
    return SmartCoach(
        agent_name=agent_name,
        gpt_manager=gpt,  # type: ignore[arg-type]
        model="test-model",
    )


def _make_step_ctx(coach=None, agent_name="TestAgent"):
    """Build a minimal SmartStepContext."""
    from core.training.smart_coach import SmartStepContext
    from core.llm.smart_mentor import AttackContext
    from core.commands.command_registry import AttackPhase
    ctx = AttackContext(
        target="10.0.0.1",
        current_phase=AttackPhase.RECON,
        state_flags={},
    )
    return SmartStepContext(
        episode=0,
        step=0,
        agent_name=agent_name,
        attack_context=ctx,
        state={},
    )


# ── T15.2 Tests: PPO Neuromodulation Hook ──────────────────────────────────

class TestPPONeuromodHook:
    """Test PPO.apply_neuromodulation()."""

    def test_apply_neuromod_exists(self):
        """PPO agent has apply_neuromodulation method."""
        ppo = _make_ppo_agent()
        assert hasattr(ppo, "apply_neuromodulation")
        assert callable(ppo.apply_neuromodulation)

    def test_apply_neuromod_default_no_crash(self):
        """Calling with defaults does not crash."""
        ppo = _make_ppo_agent()
        ppo.apply_neuromodulation()  # all defaults = 1.0

    def test_apply_neuromod_entropy_bounded(self):
        """Entropy coef is bounded by multiplier."""
        ppo = _make_ppo_agent()
        original_entropy = ppo.config.entropy_coef

        # High multiplier: capped at 1.5x
        ppo.apply_neuromodulation(entropy_coef_mult=5.0)
        assert ppo.entropy_coef <= original_entropy * 1.5 + 1e-8

        # Low multiplier: capped at 0.5x (but not below entropy_coef_min)
        ppo.apply_neuromodulation(entropy_coef_mult=0.1)
        assert ppo.entropy_coef >= ppo.config.entropy_coef_min

    def test_apply_neuromod_lr_bounded(self):
        """Learning rate never exceeds base config LR."""
        ppo = _make_ppo_agent()
        base_lr = ppo.config.learning_rate

        ppo.apply_neuromodulation(lr_mult=10.0)
        for pg in ppo.optimizer.param_groups:
            assert pg["lr"] <= base_lr + 1e-8

        # Low multiplier: at least lr_min
        ppo.apply_neuromodulation(lr_mult=0.01)
        for pg in ppo.optimizer.param_groups:
            assert pg["lr"] >= ppo.config.lr_min

    def test_apply_neuromod_bc_weight_bounded(self):
        """BC weight multiplier bounded [0.5, 1.5]."""
        ppo = _make_ppo_agent()
        ppo.config.use_bc_loss = True
        ppo.config.bc_loss_coef = 0.1

        ppo.apply_neuromodulation(bc_weight_mult=3.0)
        assert ppo.config.bc_loss_coef <= 0.15 + 1e-8  # 0.1 * 1.5

        ppo.apply_neuromodulation(bc_weight_mult=0.1)
        assert ppo.config.bc_loss_coef >= 0.05 - 1e-8  # 0.1 * 0.5


# ── T15.2 Tests: SmartCoach P15 Init ────────────────────────────────────────

class TestSmartCoachP15Init:
    """Test SmartCoach Phase 15 component initialization."""

    def test_p15_attrs_exist_flags_off(self, monkeypatch):
        """With flags explicitly OFF, P15 components are None."""
        for flag in [
            "FF_NEUROMODULATORS", "FF_REFLEX_POLICY", "FF_ACTION_ARBITRATOR",
            "FF_WORKING_MEMORY", "FF_CONSOLIDATION", "FF_AGGRESSION_CONTROLLER",
            "FF_SEMANTIC_INDEX", "FF_SENSORY_BUFFER",
        ]:
            monkeypatch.setenv(flag, "0")
        from core.feature_flags import reset_feature_flags
        reset_feature_flags()
        coach = _make_coach("TestAgent")
        # All P15 attrs should exist and be None
        assert coach._p15_neuromod_engine is None
        assert coach._p15_neuromod_state is None
        assert coach._p15_reflex_policy is None
        assert coach._p15_action_arbitrator is None
        assert coach._p15_aggression_controller is None
        assert coach._p15_sensory_buffer is None
        assert coach._p15_working_memory is None
        assert coach._p15_consolidation_engine is None
        reset_feature_flags()

    def test_p15_attrs_neuromod_on(self, monkeypatch):
        """With FF_NEUROMODULATORS=1, engine is initialized."""
        monkeypatch.setenv("FF_NEUROMODULATORS", "1")
        from core.feature_flags import reset_feature_flags
        reset_feature_flags()

        coach = _make_coach("TestAgent")
        assert coach._p15_neuromod_engine is not None
        assert coach._p15_neuromod_state is not None
        assert coach._p15_neuromod_history is not None

        # Cleanup
        monkeypatch.delenv("FF_NEUROMODULATORS", raising=False)
        reset_feature_flags()

    def test_p15_attrs_reflex_on(self, monkeypatch):
        """With FF_REFLEX_POLICY=1, reflex is initialized."""
        monkeypatch.setenv("FF_REFLEX_POLICY", "1")
        from core.feature_flags import reset_feature_flags
        reset_feature_flags()

        coach = _make_coach("TestAgent")
        assert coach._p15_reflex_policy is not None

        monkeypatch.delenv("FF_REFLEX_POLICY", raising=False)
        reset_feature_flags()


# ── T15.2 Tests: Neuromod Engine Applied in decide() ───────────────────────

class TestNeuromodInDecide:
    """Verify neuromod compute does not crash in decide() context."""

    def test_neuromod_state_after_decide_flags_on(self):
        """Post-Phase 20: flags ON by default, neuromod state initialized after decide()."""
        coach = _make_coach("TestAgent")
        step_ctx = _make_step_ctx(agent_name="TestAgent")
        result = coach.decide(step_ctx)
        assert result is not None
        # Post-Phase 20: neuromod defaults ON, state should be populated
        assert coach._p15_neuromod_state is not None


# ── T15.2 Tests: CAP Contract ──────────────────────────────────────────────

class TestCAPContractT152:
    """
    CAP = Capability Assurance Protocol.
    Flags OFF → zero behavior change compared to Phase 14.
    """

    def test_decide_returns_valid_result_flags_on(self):
        """Post-Phase 20: decide() returns a valid SmartDecisionResult with all flags ON."""
        coach = _make_coach("RedAgent")
        step_ctx = _make_step_ctx(agent_name="RedAgent")
        result = coach.decide(step_ctx)
        # Must always get a result back
        assert result is not None
        assert result.command is not None

    def test_decide_no_crash_all_flags_on(self):
        """Post-Phase 20: decide() with all P15 components active doesn't crash."""
        coach = _make_coach("ScoutAgent")
        step_ctx = _make_step_ctx(agent_name="ScoutAgent")
        result = coach.decide(step_ctx)
        assert result is not None


# ── T15.2 Tests: Aggression Level Tracking ─────────────────────────────────

class TestAggressionTracking:
    """Verify aggression level is properly tracked in SmartCoach."""

    def test_default_aggression_level(self):
        """Default aggression level is 0.3."""
        coach = _make_coach("TestAgent")
        assert coach._p15_aggression_level == 0.3
