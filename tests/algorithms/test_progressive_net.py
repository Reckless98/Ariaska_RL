"""Tests for B9: Progressive Network Expansion."""
from __future__ import annotations

import os
import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


@pytest.mark.skipif(not _HAS_TORCH, reason="torch required")
class TestProgressiveExpander:
    def test_import(self):
        from core.algorithms.progressive_net import ProgressiveExpander
        pe = ProgressiveExpander()
        assert pe is not None

    def test_should_not_expand_disabled(self):
        from core.algorithms.progressive_net import ProgressiveExpander, ExpansionConfig
        pe = ProgressiveExpander(config=ExpansionConfig(enabled=False))
        assert pe.should_expand(1000, 0.5) is False

    def test_should_not_expand_before_milestone(self):
        from core.algorithms.progressive_net import ProgressiveExpander, ExpansionConfig
        pe = ProgressiveExpander(config=ExpansionConfig(enabled=True, milestones=[200]))
        for i in range(100):
            assert pe.should_expand(i, 0.5) is False

    def test_widen_layer_preserves_weights(self):
        from core.algorithms.progressive_net import ProgressiveExpander
        old = nn.Linear(64, 32)
        new = ProgressiveExpander.widen_layer(old, 128, 64, noise_scale=0.0)
        assert new.in_features == 128
        assert new.out_features == 64
        # Old weights preserved in top-left
        assert torch.allclose(
            new.weight[:32, :64], old.weight, atol=1e-6
        )

    def test_get_target_dims(self):
        from core.algorithms.progressive_net import ProgressiveExpander, ExpansionConfig
        pe = ProgressiveExpander(config=ExpansionConfig(
            target_dims=[[256, 256], [512, 512]]
        ))
        assert pe.get_target_dims() == [256, 256]
        pe.record_expansion()
        assert pe.get_target_dims() == [512, 512]
        pe.record_expansion()
        assert pe.get_target_dims() is None
