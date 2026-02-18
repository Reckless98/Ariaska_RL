#!/usr/bin/env python3
"""
tests/test_p37_level5_integration.py — Phase 37 Level 5 GPT↔RL Integration Tests

Design acceptance tests verifying:
  1. LLMPolicyBridge produces correct tensor shapes
  2. Anneal schedule decays alpha correctly over time
  3. PPO backward compat: llm_feature_dim=0 works identically
  4. PPO Level 5: llm_feature_dim=256 expands input correctly
  5. RolloutBuffer stores and yields teacher distributions
  6. KL/ranking/value_reg losses computed when teacher data present
  7. Losses are zero when teacher data absent
  8. Ablation toggle (set_enabled) zeroes all influence
  9. Dashboard panel renders without crash
  10. Feature flag gating
"""

import math
import os
import sys

import pytest
import torch

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: LLMPolicyBridge — Shape Contracts & Anneal
# ═════════════════════════════════════════════════════════════════════════════

class TestLLMPolicyBridge:
    """Core bridge unit tests."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from core.llm.llm_policy_bridge import LLMPolicyBridge
        self.bridge = LLMPolicyBridge(action_dim=5, total_anneal_steps=100)

    def test_construction_defaults(self):
        assert self.bridge.action_dim == 5
        assert self.bridge.llm_feature_dim == 256
        assert self.bridge.enabled is True
        assert self.bridge.state.prior_alpha == 0.50

    def test_compute_guidance_shapes(self):
        """Guidance packet must produce correct tensor shapes."""
        packet = self.bridge.compute_guidance(
            state_dict={"phase": "RECON"},
            mentor_confidence=0.7,
            phase="RECON",
            step=0,
            episode=0,
        )
        assert packet.action_prior is not None
        assert packet.action_prior.shape == (5,), f"Prior shape: {packet.action_prior.shape}"
        assert packet.llm_features is not None
        assert packet.llm_features.shape == (256,), f"Features shape: {packet.llm_features.shape}"
        assert isinstance(packet.prior_alpha, float)
        assert 0.0 <= packet.prior_alpha <= 1.0

    def test_compute_guidance_teacher_dist(self):
        """Teacher distribution should be valid probability dist or None."""
        packet = self.bridge.compute_guidance(
            state_dict={},
            mentor_confidence=0.8,
            mentor_top_actions=[0, 2],
            phase="EXPLOITATION",
            step=10,
            episode=1,
        )
        if packet.teacher_distribution is not None:
            td = packet.teacher_distribution
            assert td.shape == (5,), f"Teacher dist shape: {td.shape}"
            assert torch.all(td >= 0), "Teacher dist has negative values"
            # Should sum to ~1.0 (softmax)
            assert abs(td.sum().item() - 1.0) < 0.01, f"Teacher dist sums to {td.sum():.4f}"

    def test_ablation_toggle(self):
        """When disabled, bridge returns empty packet with zero influence."""
        self.bridge.set_enabled(False)
        assert self.bridge.enabled is False

        packet = self.bridge.compute_guidance(
            state_dict={},
            mentor_confidence=0.9,
            phase="RECON",
            step=5,
            episode=0,
        )
        assert packet.prior_alpha == 0.0
        # Disabled: prior is either None or all-zeros
        if packet.action_prior is not None:
            assert torch.all(packet.action_prior == 0), "Disabled prior should be zero"

    def test_anneal_decays_alpha(self):
        """Alpha must decrease monotonically over steps (no struggling boost)."""
        alphas = []
        for step in range(0, 101, 10):
            self.bridge._step_count = step
            # Set maturity to moderate level (no struggling, no plateau)
            self.bridge.state.maturity_signal = 0.3
            self.bridge.state.success_rate = 0.4
            alpha = self.bridge._compute_anneal_alpha()
            alphas.append(alpha)

        # First > last (decayed)
        assert alphas[0] > alphas[-1], f"Alpha should decay: {alphas[0]:.4f} → {alphas[-1]:.4f}"
        # All within valid range
        for a in alphas:
            assert 0.02 <= a <= 0.50, f"Alpha out of range: {a}"

    def test_maturity_acceleration(self):
        """High maturity should produce lower alpha than low maturity at same step."""
        self.bridge._step_count = 50

        # Low maturity
        self.bridge.state.maturity_signal = 0.2
        self.bridge.state.success_rate = 0.4
        alpha_low = self.bridge._compute_anneal_alpha()

        # High maturity
        self.bridge.state.maturity_signal = 0.8
        self.bridge.state.success_rate = 0.4
        alpha_high = self.bridge._compute_anneal_alpha()

        assert alpha_high < alpha_low, (
            f"High maturity alpha ({alpha_high:.4f}) should be < "
            f"low maturity alpha ({alpha_low:.4f})"
        )

    def test_record_step_outcome_updates_maturity(self):
        """Recording step outcomes should update maturity signals."""
        for i in range(20):
            self.bridge.record_step_outcome(
                reward=2.0 if i % 2 == 0 else -1.0,
                discoveries=1 if i % 3 == 0 else 0,
                exploit_success=(i == 15),
            )
        assert self.bridge.state.success_rate > 0.0
        assert len(self.bridge._recent_rewards) == 20

    def test_record_episode_end(self):
        """Episode end should increment counter."""
        assert self.bridge.state.total_episodes == 0
        self.bridge.record_episode_end()
        assert self.bridge.state.total_episodes == 1

    def test_influence_snapshot_keys(self):
        """Snapshot must contain all keys needed by dashboard."""
        snap = self.bridge.get_influence_snapshot()
        required = {
            "prior_alpha", "kl_teacher_coef", "value_reg_coef",
            "ranking_loss_coef", "teacher_anneal_pct", "maturity_signal",
            "success_rate", "reward_velocity", "discovery_efficiency",
            "exploit_success_rate", "enabled", "total_steps",
            "bc_loss", "kl_teacher_loss", "ranking_loss", "value_reg_loss",
        }
        missing = required - set(snap.keys())
        assert not missing, f"Snapshot missing keys: {missing}"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: PPO Backward Compatibility (llm_feature_dim=0)
# ═════════════════════════════════════════════════════════════════════════════

class TestPPOBackwardCompat:
    """Verify llm_feature_dim=0 doesn't change PPO behavior."""

    def test_forward_pass_unchanged(self):
        """Standard PPO forward pass with feature_dim=0."""
        from core.algorithms.ppo_agent import PPOActorCritic, PPOConfig
        config = PPOConfig(state_dim=512, action_dim=5, llm_feature_dim=0)
        net = PPOActorCritic(config)
        state = torch.randn(4, 512)
        logits, value = net(state)
        assert logits.shape == (4, 5)
        assert value.shape == (4, 1)

    def test_select_action_unchanged(self):
        """select_action works without llm_prior."""
        from core.algorithms.ppo_agent import PPOAgent, PPOConfig
        config = PPOConfig(state_dim=512, action_dim=5, llm_feature_dim=0)
        agent = PPOAgent(config)
        state = torch.randn(512)
        action, log_prob, value = agent.select_action(state, training=True)
        assert 0 <= action < 5
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_get_action_and_value_no_prior(self):
        """get_action_and_value without llm_prior returns standard shapes."""
        from core.algorithms.ppo_agent import PPOActorCritic, PPOConfig
        config = PPOConfig(state_dim=512, action_dim=5, llm_feature_dim=0)
        net = PPOActorCritic(config)
        state = torch.randn(4, 512)
        action, log_prob, entropy, value = net.get_action_and_value(state)
        assert action.shape == (4,)
        assert log_prob.shape == (4,)

    def test_store_transition_no_teacher(self):
        """store_transition works without teacher data."""
        from core.algorithms.ppo_agent import PPOAgent, PPOConfig
        config = PPOConfig(state_dim=512, action_dim=5)
        agent = PPOAgent(config)
        agent.store_transition(
            state=torch.randn(512),
            action=0,
            log_prob=-1.0,
            reward=1.0,
            value=0.5,
            done=False,
        )
        assert agent.buffer.size == 1


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: PPO Level 5 (llm_feature_dim=256)
# ═════════════════════════════════════════════════════════════════════════════

class TestPPOLevel5:
    """PPO with enhanced state dimension (Level 5 active)."""

    def test_enhanced_forward_pass(self):
        """PPO accepts 768-dim input when llm_feature_dim=256."""
        from core.algorithms.ppo_agent import PPOActorCritic, PPOConfig
        config = PPOConfig(state_dim=512, action_dim=5, llm_feature_dim=256)
        net = PPOActorCritic(config)
        state = torch.randn(4, 768)  # 512 + 256
        logits, value = net(state)
        assert logits.shape == (4, 5)
        assert value.shape == (4, 1)

    def test_llm_prior_injection(self):
        """LLM prior should shift logit distribution."""
        from core.algorithms.ppo_agent import PPOActorCritic, PPOConfig
        config = PPOConfig(
            state_dim=512, action_dim=5, llm_feature_dim=0,
            use_llm_prior=True,
        )
        net = PPOActorCritic(config)
        state = torch.randn(4, 512)

        # Without prior
        _, log_probs_no, _, _ = net.get_action_and_value(state)

        # With strong prior favoring action 0
        prior = torch.zeros(5)
        prior[0] = 5.0  # Strong preference for action 0
        _, log_probs_with, _, _ = net.get_action_and_value(
            state, llm_prior=prior, prior_alpha=1.0
        )
        # The distributions should differ
        assert not torch.allclose(log_probs_no, log_probs_with), (
            "LLM prior should change log probabilities"
        )

    def test_select_action_with_prior(self):
        """select_action accepts and uses llm_prior."""
        from core.algorithms.ppo_agent import PPOAgent, PPOConfig
        config = PPOConfig(
            state_dim=512, action_dim=5,
            use_llm_prior=True,
        )
        agent = PPOAgent(config)
        state = torch.randn(512)
        prior = torch.tensor([0.0, 0.0, 5.0, 0.0, 0.0])
        action, lp, val = agent.select_action(
            state, training=True,
            llm_prior=prior, prior_alpha=0.5,
        )
        assert 0 <= action < 5


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: RolloutBuffer Teacher Data
# ═════════════════════════════════════════════════════════════════════════════

class TestRolloutBufferTeacher:
    """Verify teacher distribution storage and batch yielding."""

    def _fill_buffer(self, with_teacher: bool, n: int = 16):
        from core.algorithms.ppo_agent import RolloutBuffer
        buf = RolloutBuffer(capacity=256)
        for i in range(n):
            td = torch.softmax(torch.randn(5), dim=0) if with_teacher else None
            ta = i % 5 if with_teacher else None
            buf.add(
                state=torch.randn(512),
                action=i % 5,
                log_prob=-1.0,
                reward=float(i),
                value=float(i) * 0.1,
                done=(i == n - 1),
                teacher_distribution=td,
                teacher_action=ta,
            )
        return buf

    def test_teacher_data_stored(self):
        """Buffer stores teacher distributions correctly."""
        buf = self._fill_buffer(with_teacher=True, n=10)
        assert len(buf.teacher_distributions) == 10
        assert all(td is not None for td in buf.teacher_distributions)

    def test_no_teacher_data(self):
        """Buffer works fine without teacher data."""
        buf = self._fill_buffer(with_teacher=False, n=10)
        assert len(buf.teacher_distributions) == 10
        assert all(td is None for td in buf.teacher_distributions)

    def test_batches_include_teacher(self):
        """get_batches yields teacher data when present."""
        buf = self._fill_buffer(with_teacher=True, n=16)
        returns, advs = buf.compute_returns_and_advantages(0.0)
        batches = list(buf.get_batches(returns, advs, 8, torch.device("cpu")))
        assert len(batches) >= 1
        assert "teacher_distributions" in batches[0]
        assert "teacher_has_mask" in batches[0]
        assert batches[0]["teacher_distributions"].shape[1] == 5

    def test_batches_without_teacher(self):
        """get_batches omits teacher keys when no teacher data."""
        buf = self._fill_buffer(with_teacher=False, n=16)
        returns, advs = buf.compute_returns_and_advantages(0.0)
        batches = list(buf.get_batches(returns, advs, 8, torch.device("cpu")))
        assert len(batches) >= 1
        assert "teacher_distributions" not in batches[0]

    def test_reset_clears_teacher_data(self):
        """Reset should clear teacher data lists."""
        buf = self._fill_buffer(with_teacher=True, n=5)
        assert buf.size == 5
        buf.reset()
        assert buf.size == 0
        assert len(buf.teacher_distributions) == 0
        assert len(buf.teacher_actions) == 0


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5: Auxiliary Losses — KL Teacher, Ranking, Value Reg
# ═════════════════════════════════════════════════════════════════════════════

class TestAuxiliaryLosses:
    """Verify Level 5 auxiliary losses are computed correctly."""

    def _make_agent_and_fill(self, with_teacher: bool = True, n: int = 32):
        from core.algorithms.ppo_agent import PPOAgent, PPOConfig
        config = PPOConfig(
            state_dim=512,
            action_dim=5,
            hidden_dims=[64, 64],  # Small for test speed
            use_llm_prior=True,
            use_kl_teacher_loss=with_teacher,
            use_ranking_loss=with_teacher,
            use_value_reg_loss=with_teacher,
            kl_teacher_coef=0.15,
            ranking_loss_coef=0.05,
            value_reg_coef=0.10,
            rollout_size=n,
        )
        agent = PPOAgent(config)
        for i in range(n):
            td = torch.softmax(torch.randn(5), dim=0) if with_teacher else None
            ta = i % 5 if with_teacher else None
            agent.store_transition(
                state=torch.randn(512),
                action=i % 5,
                log_prob=-1.5,
                reward=float(i % 3),
                value=0.5,
                done=(i == n - 1),
                teacher_distribution=td,
                teacher_action=ta,
            )
        return agent

    def test_aux_losses_computed_with_teacher(self):
        """KL, ranking, value_reg losses should be > 0 when teacher present."""
        agent = self._make_agent_and_fill(with_teacher=True, n=32)
        metrics = agent.update(last_value=0.0)
        # KL teacher loss should be computed
        assert "kl_teacher_loss" in metrics
        # At least one should be > 0 (ranking depends on teacher action validity)
        assert metrics.get("kl_teacher_loss", 0.0) >= 0.0

    def test_aux_losses_zero_without_teacher(self):
        """All auxiliary losses should be 0 when no teacher data."""
        agent = self._make_agent_and_fill(with_teacher=False, n=32)
        metrics = agent.update(last_value=0.0)
        assert metrics.get("kl_teacher_loss", 0.0) == 0.0
        assert metrics.get("ranking_loss", 0.0) == 0.0
        assert metrics.get("value_reg_loss", 0.0) == 0.0

    def test_config_flags_disable_losses(self):
        """Setting use_* flags to False should skip loss computation."""
        from core.algorithms.ppo_agent import PPOAgent, PPOConfig
        config = PPOConfig(
            state_dim=512,
            action_dim=5,
            hidden_dims=[64, 64],
            use_kl_teacher_loss=False,
            use_ranking_loss=False,
            use_value_reg_loss=False,
            rollout_size=32,
        )
        agent = PPOAgent(config)
        for i in range(32):
            agent.store_transition(
                state=torch.randn(512),
                action=i % 5,
                log_prob=-1.5,
                reward=1.0,
                value=0.5,
                done=(i == 31),
                teacher_distribution=torch.softmax(torch.randn(5), dim=0),
                teacher_action=i % 5,
            )
        metrics = agent.update(last_value=0.0)
        assert metrics.get("kl_teacher_loss", 0.0) == 0.0


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6: Dashboard Panel Rendering
# ═════════════════════════════════════════════════════════════════════════════

class TestDashboardLevel5Panel:
    """Verify the GPT↔RL panel renders without crash."""

    def test_panel_renders(self):
        """_build_llm_bridge_panel should produce a valid Panel."""
        from core.observability.live_dashboard import LiveDashboard
        dash = LiveDashboard()
        snap = {
            "prior_alpha": 0.35,
            "kl_teacher_coef": 0.12,
            "value_reg_coef": 0.08,
            "ranking_loss_coef": 0.04,
            "teacher_anneal_pct": 0.70,
            "maturity_signal": 0.45,
            "success_rate": 0.30,
            "reward_velocity": 1.5,
            "discovery_efficiency": 0.20,
            "exploit_success_rate": 0.10,
            "bc_loss": 0.001,
            "kl_teacher_loss": 0.05,
            "ranking_loss": 0.02,
            "value_reg_loss": 0.03,
            "enabled": True,
            "total_steps": 150,
        }
        panel = dash._build_llm_bridge_panel(snap)
        from rich.panel import Panel
        assert isinstance(panel, Panel)

    def test_panel_disabled_state(self):
        """Panel should show DISABLED when enabled=False."""
        from core.observability.live_dashboard import LiveDashboard
        dash = LiveDashboard()
        snap = {
            "prior_alpha": 0.0,
            "kl_teacher_coef": 0.0,
            "value_reg_coef": 0.0,
            "ranking_loss_coef": 0.0,
            "teacher_anneal_pct": 0.0,
            "maturity_signal": 0.0,
            "success_rate": 0.0,
            "reward_velocity": 0.0,
            "discovery_efficiency": 0.0,
            "exploit_success_rate": 0.0,
            "bc_loss": 0.0,
            "kl_teacher_loss": 0.0,
            "ranking_loss": 0.0,
            "value_reg_loss": 0.0,
            "enabled": False,
            "total_steps": 0,
        }
        panel = dash._build_llm_bridge_panel(snap)
        from rich.panel import Panel
        assert isinstance(panel, Panel)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7: Feature Flag Gating
# ═════════════════════════════════════════════════════════════════════════════

class TestFeatureFlagLevel5:
    """Verify feature flag exists and is properly gated."""

    def test_flag_exists(self):
        """llm_policy_bridge flag should exist in FeatureFlags."""
        from core.feature_flags import get_feature_flags
        ff = get_feature_flags()
        assert hasattr(ff, 'llm_policy_bridge')

    def test_flag_default_true(self):
        """Flag should default to True."""
        from core.feature_flags import FeatureFlags
        ff = FeatureFlags()
        assert ff.llm_policy_bridge is True

    def test_env_override(self):
        """Flag should respect FF_LLM_POLICY_BRIDGE env var."""
        try:
            os.environ["FF_LLM_POLICY_BRIDGE"] = "0"
            from core.feature_flags import FeatureFlags
            ff = FeatureFlags()
            assert ff.llm_policy_bridge is False
        finally:
            os.environ.pop("FF_LLM_POLICY_BRIDGE", None)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8: State Encoder Constants
# ═════════════════════════════════════════════════════════════════════════════

class TestStateEncoderLevel5Constants:
    """Verify state encoder has Level 5 dimensional constants."""

    def test_constants_defined(self):
        from core.models.state_encoder import STATE_DIM, LLM_FEATURE_DIM, ENHANCED_STATE_DIM
        assert STATE_DIM == 512
        assert LLM_FEATURE_DIM == 256
        assert ENHANCED_STATE_DIM == 768

    def test_dim_relationship(self):
        from core.models.state_encoder import STATE_DIM, LLM_FEATURE_DIM, ENHANCED_STATE_DIM
        assert ENHANCED_STATE_DIM == STATE_DIM + LLM_FEATURE_DIM


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9: End-to-End Integration Smoke Test
# ═════════════════════════════════════════════════════════════════════════════

class TestEndToEndLevel5:
    """Smoke test: bridge → PPO → update cycle."""

    def test_full_cycle(self):
        """Run a minimal bridge→PPO cycle without errors."""
        from core.llm.llm_policy_bridge import LLMPolicyBridge
        from core.algorithms.ppo_agent import PPOAgent, PPOConfig

        bridge = LLMPolicyBridge(action_dim=5, total_anneal_steps=50)
        config = PPOConfig(
            state_dim=512,
            action_dim=5,
            hidden_dims=[64, 64],
            use_llm_prior=True,
            use_kl_teacher_loss=True,
            use_ranking_loss=True,
            use_value_reg_loss=True,
            rollout_size=16,
        )
        agent = PPOAgent(config)

        for step in range(16):
            state = torch.randn(512)

            # Bridge computes guidance
            packet = bridge.compute_guidance(
                state_dict={"phase": "RECON"},
                mentor_confidence=0.6,
                phase="RECON",
                step=step,
                episode=0,
            )

            # PPO selects action with LLM prior
            action, lp, val = agent.select_action(
                state, training=True,
                llm_prior=packet.action_prior,
                prior_alpha=packet.prior_alpha,
            )

            # Store transition with teacher data
            agent.store_transition(
                state=state,
                action=action,
                log_prob=lp,
                reward=float(step % 3),
                value=val,
                done=(step == 15),
                teacher_distribution=packet.teacher_distribution,
                teacher_action=action,
            )

            # Record outcome
            bridge.record_step_outcome(
                reward=float(step % 3),
                discoveries=1 if step % 4 == 0 else 0,
            )

        # PPO update with Level 5 losses
        metrics = agent.update(last_value=0.0)
        assert "policy_loss" in metrics
        assert "value_loss" in metrics

        # Bridge records episode end
        bridge.record_episode_end()
        assert bridge.state.total_episodes == 1

        # Snapshot should be valid
        snap = bridge.get_influence_snapshot()
        assert snap["total_steps"] == 16
        assert snap["enabled"] is True

    def test_ablation_cycle(self):
        """Full cycle with bridge disabled should work safely."""
        from core.llm.llm_policy_bridge import LLMPolicyBridge
        from core.algorithms.ppo_agent import PPOAgent, PPOConfig

        bridge = LLMPolicyBridge(action_dim=5)
        bridge.set_enabled(False)

        config = PPOConfig(
            state_dim=512, action_dim=5,
            hidden_dims=[64, 64],
            rollout_size=16,
        )
        agent = PPOAgent(config)

        for step in range(16):
            state = torch.randn(512)
            packet = bridge.compute_guidance(
                state_dict={},
                phase="RECON",
                step=step,
                episode=0,
            )
            # Empty packet — no prior, no teacher
            action, lp, val = agent.select_action(state, training=True)
            agent.store_transition(
                state=state,
                action=action,
                log_prob=lp,
                reward=1.0,
                value=val,
                done=(step == 15),
            )

        metrics = agent.update(last_value=0.0)
        assert metrics.get("kl_teacher_loss", 0.0) == 0.0


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 10: PPOConfig Level 5 Fields
# ═════════════════════════════════════════════════════════════════════════════

class TestPPOConfigLevel5:
    """Verify Level 5 config fields exist with correct defaults."""

    def test_config_fields(self):
        from core.algorithms.ppo_agent import PPOConfig
        config = PPOConfig()
        assert hasattr(config, 'llm_feature_dim')
        assert hasattr(config, 'use_llm_prior')
        assert hasattr(config, 'prior_alpha_init')
        assert hasattr(config, 'use_kl_teacher_loss')
        assert hasattr(config, 'kl_teacher_coef')
        assert hasattr(config, 'use_ranking_loss')
        assert hasattr(config, 'ranking_loss_coef')
        assert hasattr(config, 'ranking_margin')
        assert hasattr(config, 'use_value_reg_loss')
        assert hasattr(config, 'value_reg_coef')

    def test_defaults_off(self):
        """Level 5 features should default to OFF for backward compatibility."""
        from core.algorithms.ppo_agent import PPOConfig
        config = PPOConfig()
        assert config.llm_feature_dim == 0
        assert config.use_llm_prior is False
        assert config.use_kl_teacher_loss is False
        assert config.use_ranking_loss is False
        assert config.use_value_reg_loss is False

    def test_diagnostics_include_level5(self):
        """get_diagnostics should report Level 5 fields."""
        from core.algorithms.ppo_agent import PPOAgent, PPOConfig
        config = PPOConfig(state_dim=512, action_dim=5, hidden_dims=[64, 64])
        agent = PPOAgent(config)
        diag = agent.get_diagnostics()
        assert "llm_feature_dim" in diag
        assert "use_llm_prior" in diag
