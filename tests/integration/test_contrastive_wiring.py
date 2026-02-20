"""Phase 42 Stage 1H: ContrastiveLoss → PPO wiring tests."""

import os
import pytest
import torch

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


class TestContrastiveWiring:
    """Verify ContrastiveLoss integration in PPOAgent."""

    @pytest.fixture(autouse=True)
    def setup(self):
        os.environ["ARIASKA_DRY_RUN"] = "1"
        from core.feature_flags import reset_feature_flags
        reset_feature_flags()

    def test_ppo_config_has_contrastive_fields(self):
        """PPOConfig exposes use_contrastive_loss and contrastive_coef."""
        from core.algorithms.ppo_agent import PPOConfig
        cfg = PPOConfig()
        assert hasattr(cfg, "use_contrastive_loss")
        assert cfg.use_contrastive_loss is False
        assert hasattr(cfg, "contrastive_coef")
        assert cfg.contrastive_coef == 0.05

    def test_ppo_agent_has_contrastive_attr(self):
        """PPOAgent initializes _contrastive_loss attribute."""
        from core.algorithms.ppo_agent import PPOAgent, PPOConfig
        cfg = PPOConfig(use_contrastive_loss=False)
        agent = PPOAgent(config=cfg, device="cpu")
        assert hasattr(agent, "_contrastive_loss")
        assert agent._contrastive_loss is None

    def test_ppo_agent_inits_contrastive_when_enabled(self):
        """PPOAgent lazy-inits ContrastiveLoss when flag is on."""
        from core.algorithms.ppo_agent import PPOAgent, PPOConfig
        from core.feature_flags import set_feature_flag
        set_feature_flag("contrastive_ppo", True)
        cfg = PPOConfig(use_contrastive_loss=True)
        agent = PPOAgent(config=cfg, device="cpu")
        assert agent._contrastive_loss is not None

    def test_ppo_agent_skips_contrastive_when_flag_off(self):
        """PPOAgent doesn't init ContrastiveLoss when FF is off."""
        from core.algorithms.ppo_agent import PPOAgent, PPOConfig
        from core.feature_flags import set_feature_flag
        set_feature_flag("contrastive_ppo", False)
        cfg = PPOConfig(use_contrastive_loss=True)
        agent = PPOAgent(config=cfg, device="cpu")
        assert agent._contrastive_loss is None

    def test_backbone_features_method(self):
        """PPOActorCritic.get_backbone_features returns features."""
        from core.algorithms.ppo_agent import PPOActorCritic, PPOConfig
        cfg = PPOConfig()
        net = PPOActorCritic(cfg)
        state = torch.randn(2, 512)
        feats = net.get_backbone_features(state)
        assert feats.shape[0] == 2
        assert feats.shape[1] == cfg.hidden_dims[-1]

    def test_update_with_contrastive_no_crash(self):
        """PPO update with contrastive loss doesn't crash."""
        from core.algorithms.ppo_agent import PPOAgent, PPOConfig
        from core.feature_flags import set_feature_flag
        set_feature_flag("contrastive_ppo", True)
        cfg = PPOConfig(
            use_contrastive_loss=True,
            contrastive_coef=0.05,
            rollout_size=8,
            minibatch_size=4,
        )
        agent = PPOAgent(config=cfg, device="cpu")

        # Store a few transitions
        for i in range(8):
            state = torch.randn(512)
            # Vary phase encoding to create different phase groups
            if i < 4:
                state[0] = 1.0  # recon phase
            else:
                state[3] = 1.0  # exploit phase
            action_idx, log_prob, value = agent.select_action(state)
            reward = float(i) * 0.5
            agent.store_transition(state, action_idx, log_prob, reward, value, False)

        # Run update
        metrics = agent.update(last_value=0.0)
        # Should not crash
        assert isinstance(metrics, dict)
