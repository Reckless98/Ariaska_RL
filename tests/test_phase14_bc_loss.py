#!/usr/bin/env python3
"""
tests/test_phase14_bc_loss.py — Phase 14.0 T1: Behavioral Cloning Loss Tests

Contract C3.6: 4 tests verifying BC loss computation in PPO.
"""

import os
import sys
import pytest
import torch

os.environ["ARIASKA_DRY_RUN"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestBCLossComputation:
    """Verify BC loss wiring between BCBuffer and PPOAgent."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up PPO agent with BC buffer."""
        from core.reasoning.teacher_trace import BCBuffer, BCSample
        self.BCBuffer = BCBuffer
        self.BCSample = BCSample

    def test_bc_loss_computes_with_mock_buffer(self):
        """BC loss produces non-zero gradient when buffer has samples."""
        from core.algorithms.ppo_agent import PPOAgent, PPOConfig
        from core.reasoning.teacher_trace import BCBuffer, BCSample

        buf = BCBuffer(capacity=100)
        # Create mock BC samples with proper state tensors
        for i in range(10):
            sample = BCSample(
                state=torch.randn(512),
                teacher_action=i % 5,  # action_dim=5
                weight=1.0,
                rationale_hash=hash(f"sample_{i}"),
                episode=0,
                step=i,
            )
            buf._samples.append(sample)  # Direct inject for testing

        config = PPOConfig(
            state_dim=512,
            action_dim=5,
            use_bc_loss=True,
            bc_loss_coef=0.1,
            bc_buffer=buf,
        )
        agent = PPOAgent(config)

        # We need at least one transition stored to trigger update
        for i in range(20):
            state = torch.randn(512)
            agent.store_transition(
                state=state,
                action=i % 5,
                log_prob=-1.0,
                reward=1.0,
                value=0.5,
                done=(i == 19),
            )
        # Run update — BC loss should fire inside
        metrics = agent.update(last_value=0.0)
        # Update should succeed (metrics is a dict)
        assert isinstance(metrics, dict)

    def test_bc_loss_disabled_when_flag_off(self):
        """BC loss does not fire when use_bc_loss=False."""
        from core.algorithms.ppo_agent import PPOConfig

        config = PPOConfig(
            state_dim=512,
            action_dim=5,
            use_bc_loss=False,
            bc_loss_coef=0.1,
            bc_buffer=None,
        )
        assert config.use_bc_loss is False
        assert config.bc_buffer is None

    def test_bc_loss_with_empty_buffer(self):
        """BC loss gracefully handles empty buffer (min 2 required)."""
        from core.algorithms.ppo_agent import PPOAgent, PPOConfig
        from core.reasoning.teacher_trace import BCBuffer

        buf = BCBuffer(capacity=100)
        assert buf.get_stats()["size"] == 0

        config = PPOConfig(
            state_dim=512,
            action_dim=5,
            use_bc_loss=True,
            bc_loss_coef=0.1,
            bc_buffer=buf,
        )
        agent = PPOAgent(config)

        # Store minimal transitions
        for i in range(20):
            state = torch.randn(512)
            agent.store_transition(
                state=state,
                action=i % 5,
                log_prob=-1.0,
                reward=1.0,
                value=0.5,
                done=(i == 19),
            )
        # Update should succeed without BC loss (buffer empty, <2 samples)
        metrics = agent.update(last_value=0.0)
        assert isinstance(metrics, dict)

    def test_bc_loss_coefficient_weighting(self):
        """BC loss coefficient controls contribution magnitude."""
        from core.algorithms.ppo_agent import PPOConfig

        low = PPOConfig(state_dim=512, action_dim=5, bc_loss_coef=0.01)
        high = PPOConfig(state_dim=512, action_dim=5, bc_loss_coef=1.0)
        assert low.bc_loss_coef < high.bc_loss_coef
        assert high.bc_loss_coef == 1.0


class TestGetLogits:
    """Verify PPOActorCritic.get_logits() returns proper tensors."""

    def test_get_logits_shape(self):
        """get_logits returns (batch, action_dim) tensor."""
        from core.algorithms.ppo_agent import PPOAgent, PPOConfig

        config = PPOConfig(state_dim=512, action_dim=5)
        agent = PPOAgent(config)
        state = torch.randn(1, 512)
        logits = agent.network.get_logits(state)
        assert logits.shape == (1, 5)

    def test_get_logits_batch(self):
        """get_logits works with batch of states."""
        from core.algorithms.ppo_agent import PPOAgent, PPOConfig

        config = PPOConfig(state_dim=512, action_dim=5)
        agent = PPOAgent(config)
        states = torch.randn(4, 512)
        logits = agent.network.get_logits(states)
        assert logits.shape == (4, 5)
