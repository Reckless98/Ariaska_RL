"""Tests for B4: N-step return computation."""
from __future__ import annotations

import os
import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


class TestNStepConfig:
    def test_defaults(self):
        from core.algorithms.nstep_returns import NStepConfig
        cfg = NStepConfig()
        assert cfg.n == 5
        assert cfg.gamma == 0.99
        assert cfg.blend_alpha == 0.3

    def test_custom(self):
        from core.algorithms.nstep_returns import NStepConfig
        cfg = NStepConfig(n=3, gamma=0.95)
        assert cfg.n == 3


class TestComputeNStepReturns:
    def test_empty(self):
        from core.algorithms.nstep_returns import compute_nstep_returns, NStepConfig
        result = compute_nstep_returns([], [], [], NStepConfig(n=3))
        assert result == []

    def test_single_step_terminal(self):
        from core.algorithms.nstep_returns import compute_nstep_returns, NStepConfig
        rewards = [10.0]
        values = [5.0]
        dones = [True]
        result = compute_nstep_returns(rewards, values, dones, NStepConfig(n=3, gamma=0.99))
        assert abs(result[0] - 10.0) < 1e-6  # Terminal, no bootstrap

    def test_single_step_non_terminal(self):
        from core.algorithms.nstep_returns import compute_nstep_returns, NStepConfig
        rewards = [1.0]
        values = [5.0]
        dones = [False]
        # n=3 but only 1 step, should bootstrap with gamma^1 * V(s1) if available
        # No V(s1) available since values has same length as rewards
        result = compute_nstep_returns(rewards, values, dones, NStepConfig(n=3, gamma=1.0))
        # r[0] + no bootstrap (no values beyond index 0)
        assert result[0] == 1.0

    def test_three_steps_no_terminal(self):
        from core.algorithms.nstep_returns import compute_nstep_returns, NStepConfig
        rewards = [1.0, 2.0, 3.0]
        values = [10.0, 10.0, 10.0, 10.0]  # Extra bootstrap value
        dones = [False, False, False]
        cfg = NStepConfig(n=3, gamma=1.0)
        result = compute_nstep_returns(rewards, values, dones, cfg)
        # G[0] = 1 + 2 + 3 + gamma^3 * V[3] = 6 + 10 = 16
        assert abs(result[0] - 16.0) < 1e-6

    def test_terminal_mid_trajectory(self):
        from core.algorithms.nstep_returns import compute_nstep_returns, NStepConfig
        rewards = [1.0, 2.0, 100.0, 3.0]
        values = [5.0, 5.0, 5.0, 5.0]
        dones = [False, False, True, False]
        cfg = NStepConfig(n=5, gamma=1.0)
        result = compute_nstep_returns(rewards, values, dones, cfg)
        # G[0] = 1 + 2 + 100 = 103 (terminal at step 2, no bootstrap)
        assert abs(result[0] - 103.0) < 1e-6

    def test_discounting(self):
        from core.algorithms.nstep_returns import compute_nstep_returns, NStepConfig
        rewards = [1.0, 1.0, 1.0]
        values = [0.0, 0.0, 0.0]
        dones = [False, False, True]
        cfg = NStepConfig(n=3, gamma=0.5)
        result = compute_nstep_returns(rewards, values, dones, cfg)
        # G[0] = 1 + 0.5*1 + 0.25*1 = 1.75 (terminal at end)
        assert abs(result[0] - 1.75) < 1e-6


class TestBlendReturns:
    def test_pure_gae(self):
        from core.algorithms.nstep_returns import blend_returns
        gae = [1.0, 2.0, 3.0]
        nstep = [10.0, 20.0, 30.0]
        result = blend_returns(gae, nstep, alpha=0.0)
        assert result == [1.0, 2.0, 3.0]

    def test_pure_nstep(self):
        from core.algorithms.nstep_returns import blend_returns
        gae = [1.0, 2.0, 3.0]
        nstep = [10.0, 20.0, 30.0]
        result = blend_returns(gae, nstep, alpha=1.0)
        assert result == [10.0, 20.0, 30.0]

    def test_half_blend(self):
        from core.algorithms.nstep_returns import blend_returns
        gae = [0.0, 0.0]
        nstep = [10.0, 20.0]
        result = blend_returns(gae, nstep, alpha=0.5)
        assert abs(result[0] - 5.0) < 1e-6
        assert abs(result[1] - 10.0) < 1e-6

    def test_length_mismatch(self):
        from core.algorithms.nstep_returns import blend_returns
        with pytest.raises(ValueError):
            blend_returns([1.0], [1.0, 2.0])


class TestTensorVersion:
    @pytest.fixture(autouse=True)
    def skip_no_torch(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")

    def test_tensor_nstep(self):
        import torch
        from core.algorithms.nstep_returns import compute_nstep_returns_tensor, NStepConfig
        rewards = torch.tensor([1.0, 2.0, 3.0])
        values = torch.tensor([0.0, 0.0, 0.0, 10.0])
        dones = torch.tensor([0.0, 0.0, 0.0])
        cfg = NStepConfig(n=3, gamma=1.0)
        result = compute_nstep_returns_tensor(rewards, values, dones, cfg)
        assert result.shape == (3,)
        assert abs(result[0].item() - 16.0) < 1e-4
