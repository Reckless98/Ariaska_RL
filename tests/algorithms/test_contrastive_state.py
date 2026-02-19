"""Tests for B7: Contrastive State Learning."""
from __future__ import annotations

import os
import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


@pytest.mark.skipif(not _HAS_TORCH, reason="torch required")
class TestContrastiveLoss:
    def test_import(self):
        from core.algorithms.contrastive_state import ContrastiveLoss
        cl = ContrastiveLoss()
        assert cl is not None

    def test_single_sample_zero_loss(self):
        from core.algorithms.contrastive_state import ContrastiveLoss
        cl = ContrastiveLoss()
        features = torch.randn(1, 256)
        labels = torch.tensor([0])
        loss = cl.compute_loss(features, labels)
        assert loss.item() == 0.0

    def test_all_same_phase_zero_loss(self):
        from core.algorithms.contrastive_state import ContrastiveLoss
        cl = ContrastiveLoss()
        features = torch.randn(4, 256)
        labels = torch.tensor([0, 0, 0, 0])
        loss = cl.compute_loss(features, labels)
        assert loss.item() == 0.0

    def test_different_phases_positive_loss(self):
        from core.algorithms.contrastive_state import ContrastiveLoss
        cl = ContrastiveLoss()
        features = torch.randn(6, 256)
        labels = torch.tensor([0, 0, 0, 1, 1, 1])
        loss = cl.compute_loss(features, labels)
        assert loss.item() > 0.0

    def test_gradient_flow(self):
        from core.algorithms.contrastive_state import ContrastiveLoss
        cl = ContrastiveLoss()
        features = torch.randn(4, 256, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1])
        loss = cl.compute_loss(features, labels)
        loss.backward()
        assert features.grad is not None
