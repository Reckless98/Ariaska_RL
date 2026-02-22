#!/usr/bin/env python3
"""
tests/test_grad_norms.py — C05: Per-loss gradient norm logging tests

Verifies:
  1. PPO update returns total_grad_norm in metrics (always)
  2. PPO update returns per-loss norms when log_grad_norms=True
  3. SIL grad norm captured when SIL is active
  4. GradNorms dataclass serialization
  5. SmartCoach populates DecisionPacket.grad_norms after end_episode_ppo()
  6. training_metrics tracks total_grad_norm history
"""

import os
import sys
import pytest
import torch
import numpy as np

os.environ["ARIASKA_DRY_RUN"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── Helpers ───────────────────────────────────────────────────────


def _make_ppo(action_dim: int = 5, log_grad_norms: bool = False):
    """Create a minimal PPO agent for testing."""
    from core.algorithms.ppo_agent import PPOAgent, PPOConfig

    config = PPOConfig(
        state_dim=512,
        action_dim=action_dim,
        hidden_dims=[64, 64],
        rollout_size=8,
        minibatch_size=4,
        epochs_per_update=1,
        grad_accum_steps=1,
        log_grad_norms=log_grad_norms,
        # Disable advanced features to keep test fast
        use_phase_gates=False,
        use_ema_anchor=False,
        use_dual_horizon_gae=False,
        use_phase_advantage_whitening=False,
        use_spectral_norm_critic=False,
        use_adaptive_clip=False,
        use_entropy_rebound=False,
        use_phase_prediction_aux=False,
        use_cosine_entropy=False,
        return_variance_entropy=False,
        use_kl_adaptive_lr=False,
        use_contrastive_loss=False,
        sil_coef=0.0,  # Disable SIL by default
    )
    return PPOAgent(config=config, device="cpu")


def _fill_buffer(ppo, n: int = 8):
    """Store n transitions in PPO buffer."""
    for i in range(n):
        state = torch.randn(512)
        action, log_prob, value = ppo.select_action(state, training=True)
        ppo.store_transition(
            state=state,
            action=action,
            log_prob=log_prob,
            reward=float(i) * 0.5,
            value=value,
            done=(i == n - 1),
        )


# ═══════════════════════════════════════════════════════════════════
# 1. PPO update returns total_grad_norm (always)
# ═══════════════════════════════════════════════════════════════════


class TestTotalGradNorm:
    """total_grad_norm must always be present in PPO metrics."""

    def test_total_grad_norm_in_metrics(self):
        ppo = _make_ppo()
        _fill_buffer(ppo, 8)
        metrics = ppo.update(last_value=0.0)
        assert "total_grad_norm" in metrics
        assert metrics["total_grad_norm"] >= 0.0

    def test_total_grad_norm_positive_after_update(self):
        ppo = _make_ppo()
        _fill_buffer(ppo, 8)
        metrics = ppo.update(last_value=0.0)
        # Non-trivial network should produce non-zero gradients
        assert metrics["total_grad_norm"] > 0.0

    def test_total_grad_norm_finite(self):
        ppo = _make_ppo()
        _fill_buffer(ppo, 8)
        metrics = ppo.update(last_value=0.0)
        assert np.isfinite(metrics["total_grad_norm"])

    def test_total_grad_norm_in_training_metrics_history(self):
        ppo = _make_ppo()
        _fill_buffer(ppo, 8)
        ppo.update(last_value=0.0)
        assert "total_grad_norm" in ppo.training_metrics
        assert len(ppo.training_metrics["total_grad_norm"]) == 1

    def test_total_grad_norm_accumulates_across_updates(self):
        ppo = _make_ppo()
        for _ in range(3):
            _fill_buffer(ppo, 8)
            ppo.update(last_value=0.0)
        assert len(ppo.training_metrics["total_grad_norm"]) == 3


# ═══════════════════════════════════════════════════════════════════
# 2. Per-loss gradient norms (log_grad_norms=True)
# ═══════════════════════════════════════════════════════════════════


class TestPerLossGradNorms:
    """When log_grad_norms=True, individual loss norms appear in metrics."""

    def test_per_loss_norms_present(self):
        ppo = _make_ppo(log_grad_norms=True)
        _fill_buffer(ppo, 8)
        metrics = ppo.update(last_value=0.0)
        # Core loss components always active
        assert "policy_grad_norm" in metrics
        assert "value_grad_norm" in metrics
        assert "entropy_grad_norm" in metrics

    def test_per_loss_norms_positive(self):
        ppo = _make_ppo(log_grad_norms=True)
        _fill_buffer(ppo, 8)
        metrics = ppo.update(last_value=0.0)
        assert metrics["policy_grad_norm"] > 0.0
        assert metrics["value_grad_norm"] > 0.0
        assert metrics["entropy_grad_norm"] > 0.0

    def test_per_loss_norms_absent_when_disabled(self):
        ppo = _make_ppo(log_grad_norms=False)
        _fill_buffer(ppo, 8)
        metrics = ppo.update(last_value=0.0)
        # Total always present, per-loss should not be
        assert "total_grad_norm" in metrics
        assert "policy_grad_norm" not in metrics

    def test_per_loss_norms_finite(self):
        ppo = _make_ppo(log_grad_norms=True)
        _fill_buffer(ppo, 8)
        metrics = ppo.update(last_value=0.0)
        for k in ["policy_grad_norm", "value_grad_norm", "entropy_grad_norm"]:
            assert np.isfinite(metrics[k]), f"{k} is not finite"

    def test_total_still_present_with_per_loss(self):
        ppo = _make_ppo(log_grad_norms=True)
        _fill_buffer(ppo, 8)
        metrics = ppo.update(last_value=0.0)
        assert "total_grad_norm" in metrics
        assert metrics["total_grad_norm"] > 0.0


# ═══════════════════════════════════════════════════════════════════
# 3. SIL grad norm
# ═══════════════════════════════════════════════════════════════════


class TestSILGradNorm:
    """SIL backward has its own clip_grad_norm_ → sil_grad_norm."""

    def test_sil_grad_norm_when_sil_active(self):
        from core.algorithms.ppo_agent import PPOAgent, PPOConfig

        config = PPOConfig(
            state_dim=512,
            action_dim=5,
            hidden_dims=[64, 64],
            rollout_size=8,
            minibatch_size=4,
            epochs_per_update=1,
            grad_accum_steps=1,
            sil_coef=0.25,
            sil_buffer_size=100,
            sil_epochs=1,
            # Disable non-essential features
            use_phase_gates=False,
            use_ema_anchor=False,
            use_dual_horizon_gae=False,
            use_phase_advantage_whitening=False,
            use_spectral_norm_critic=False,
            use_adaptive_clip=False,
            use_entropy_rebound=False,
            use_phase_prediction_aux=False,
            use_cosine_entropy=False,
            return_variance_entropy=False,
            use_kl_adaptive_lr=False,
            use_contrastive_loss=False,
        )
        ppo = PPOAgent(config=config, device="cpu")

        # Fill SIL buffer first with a high-reward episode
        states = [torch.randn(512) for _ in range(8)]
        actions = [0, 1, 2, 0, 1, 2, 0, 1]
        rewards = [10.0] * 8
        ppo.store_sil_episode(states, actions, rewards)

        # Now do a normal update
        _fill_buffer(ppo, 8)
        metrics = ppo.update(last_value=0.0)

        # SIL should have run and produced a grad norm
        if metrics.get("sil_loss", 0) > 0:
            assert "sil_grad_norm" in metrics
            assert metrics["sil_grad_norm"] > 0.0

    def test_sil_grad_norm_absent_when_sil_disabled(self):
        ppo = _make_ppo()  # sil_coef=0.0
        _fill_buffer(ppo, 8)
        metrics = ppo.update(last_value=0.0)
        assert metrics.get("sil_grad_norm", 0.0) == 0.0


# ═══════════════════════════════════════════════════════════════════
# 4. GradNorms dataclass
# ═══════════════════════════════════════════════════════════════════


class TestGradNormsDataclass:
    """GradNorms serialization and fields."""

    def test_default_values(self):
        from core.training.decision_packet import GradNorms

        gn = GradNorms()
        assert gn.total_grad_norm == 0.0
        assert gn.policy_grad_norm == 0.0
        assert gn.sil_grad_norm == 0.0

    def test_to_dict_keys(self):
        from core.training.decision_packet import GradNorms

        gn = GradNorms(total_grad_norm=1.5, policy_grad_norm=0.3)
        d = gn.to_dict()
        assert d["total"] == 1.5
        assert d["policy"] == 0.3
        assert "value_reg" in d
        assert "contrastive" in d
        assert len(d) == 10  # 8 original + value_reg + contrastive

    def test_to_dict_all_fields(self):
        from core.training.decision_packet import GradNorms

        gn = GradNorms(
            policy_grad_norm=1.0,
            value_grad_norm=2.0,
            entropy_grad_norm=3.0,
            bc_grad_norm=4.0,
            sil_grad_norm=5.0,
            kl_teacher_grad_norm=6.0,
            ranking_grad_norm=7.0,
            value_reg_grad_norm=8.0,
            contrastive_grad_norm=9.0,
            total_grad_norm=10.0,
        )
        d = gn.to_dict()
        assert d == {
            "policy": 1.0,
            "value": 2.0,
            "entropy": 3.0,
            "bc": 4.0,
            "sil": 5.0,
            "kl_teacher": 6.0,
            "ranking": 7.0,
            "value_reg": 8.0,
            "contrastive": 9.0,
            "total": 10.0,
        }


# ═══════════════════════════════════════════════════════════════════
# 5. SmartCoach populates grad_norms on DecisionPacket
# ═══════════════════════════════════════════════════════════════════


class TestSmartCoachGradNorms:
    """SmartCoach.end_episode_ppo() populates DecisionPacket.grad_norms."""

    @pytest.fixture(autouse=True)
    def setup(self):
        os.environ["ARIASKA_DRY_RUN"] = "1"
        from core.testing.fake_gpt_manager import FakeGPTManager

        self.gpt = FakeGPTManager(seed=42)

    def test_grad_norms_populated_after_end_episode(self):
        from core.training.smart_coach import SmartCoach
        from core.training.decision_packet import DecisionPacket
        from core.llm.smart_mentor import AttackContext

        coach = SmartCoach(agent_name="RedAgent", gpt_manager=self.gpt)
        coach.attack_context = AttackContext(target="10.10.10.1")

        # Create DecisionPacket via mock step context
        class _MockCtx:
            episode = 1
            step = 5
            agent_name = "RedAgent"
            attack_context = None
            state = {}
        dp = DecisionPacket.from_step_context(_MockCtx())
        coach._current_decision_packet = dp

        # Fill PPO trajectory manually
        if coach.ppo_agent:
            for i in range(8):
                state = torch.randn(512)
                action, log_prob, value = coach.ppo_agent.select_action(
                    state, training=True
                )
                coach._ppo_trajectory.append(
                    {
                        "state": state,
                        "action": action,
                        "log_prob": log_prob,
                        "reward": 1.0,
                        "value": value,
                        "done": (i == 7),
                    }
                )

            metrics = coach.end_episode_ppo(done=True, highest_phase="RECON")
            if metrics:
                assert dp.grad_norms.total_grad_norm >= 0.0

    def test_last_grad_norms_stored_on_coach(self):
        from core.training.smart_coach import SmartCoach
        from core.training.decision_packet import DecisionPacket
        from core.llm.smart_mentor import AttackContext

        coach = SmartCoach(agent_name="RedAgent", gpt_manager=self.gpt)
        coach.attack_context = AttackContext(target="10.10.10.1")

        class _MockCtx:
            episode = 1
            step = 5
            agent_name = "RedAgent"
            attack_context = None
            state = {}
        dp = DecisionPacket.from_step_context(_MockCtx())
        coach._current_decision_packet = dp

        if coach.ppo_agent:
            for i in range(8):
                state = torch.randn(512)
                action, log_prob, value = coach.ppo_agent.select_action(
                    state, training=True
                )
                coach._ppo_trajectory.append(
                    {
                        "state": state,
                        "action": action,
                        "log_prob": log_prob,
                        "reward": 2.0,
                        "value": value,
                        "done": (i == 7),
                    }
                )

            metrics = coach.end_episode_ppo(done=True, highest_phase="RECON")
            if metrics:
                assert "total" in coach._last_grad_norms
                assert coach._last_grad_norms["total"] >= 0.0

    def test_last_grad_norms_initially_empty(self):
        from core.training.smart_coach import SmartCoach

        coach = SmartCoach(agent_name="ScoutAgent", gpt_manager=self.gpt)
        assert coach._last_grad_norms == {}


# ═══════════════════════════════════════════════════════════════════
# 6. PPOConfig log_grad_norms flag
# ═══════════════════════════════════════════════════════════════════


class TestPPOConfigFlag:
    """PPOConfig.log_grad_norms default and override."""

    def test_default_off(self):
        from core.algorithms.ppo_agent import PPOConfig

        config = PPOConfig()
        assert config.log_grad_norms is False

    def test_can_enable(self):
        from core.algorithms.ppo_agent import PPOConfig

        config = PPOConfig(log_grad_norms=True)
        assert config.log_grad_norms is True


# ═══════════════════════════════════════════════════════════════════
# 7. Gradient accumulation interaction
# ═══════════════════════════════════════════════════════════════════


class TestGradAccumInteraction:
    """Grad norms accumulate correctly across grad_accum steps."""

    def test_grad_norm_with_accum_2(self):
        from core.algorithms.ppo_agent import PPOAgent, PPOConfig

        config = PPOConfig(
            state_dim=512,
            action_dim=5,
            hidden_dims=[64, 64],
            rollout_size=16,
            minibatch_size=4,
            epochs_per_update=1,
            grad_accum_steps=2,
            use_phase_gates=False,
            use_ema_anchor=False,
            use_dual_horizon_gae=False,
            use_phase_advantage_whitening=False,
            use_spectral_norm_critic=False,
            use_adaptive_clip=False,
            use_entropy_rebound=False,
            use_phase_prediction_aux=False,
            use_cosine_entropy=False,
            return_variance_entropy=False,
            use_kl_adaptive_lr=False,
            use_contrastive_loss=False,
            sil_coef=0.0,
        )
        ppo = PPOAgent(config=config, device="cpu")
        for i in range(16):
            state = torch.randn(512)
            action, log_prob, value = ppo.select_action(state, training=True)
            ppo.store_transition(
                state=state,
                action=action,
                log_prob=log_prob,
                reward=float(i) * 0.3,
                value=value,
                done=(i == 15),
            )
        metrics = ppo.update(last_value=0.0)
        assert "total_grad_norm" in metrics
        assert metrics["total_grad_norm"] > 0.0
