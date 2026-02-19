"""Tests for B8: Transformer State Encoder."""
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
class TestStateTransformerEncoder:
    def test_import(self):
        from core.models.transformer_encoder import StateTransformerEncoder
        enc = StateTransformerEncoder()
        assert enc is not None

    def test_forward_shape(self):
        from core.models.transformer_encoder import StateTransformerEncoder, TransformerEncoderConfig
        cfg = TransformerEncoderConfig(window_size=4, state_dim=512)
        enc = StateTransformerEncoder(config=cfg)
        x = torch.randn(2, 4, 512)
        out = enc(x)
        assert out.shape == (2, 512)

    def test_forward_with_mask(self):
        from core.models.transformer_encoder import StateTransformerEncoder, TransformerEncoderConfig
        cfg = TransformerEncoderConfig(window_size=4, state_dim=512)
        enc = StateTransformerEncoder(config=cfg)
        x = torch.randn(2, 4, 512)
        mask = torch.tensor([[True, True, False, False], [True, False, False, False]])
        out = enc(x, src_key_padding_mask=mask)
        assert out.shape == (2, 512)

    def test_backward(self):
        from core.models.transformer_encoder import StateTransformerEncoder
        enc = StateTransformerEncoder()
        x = torch.randn(1, 4, 512, requires_grad=True)
        out = enc(x)
        out.sum().backward()
        assert x.grad is not None


@pytest.mark.skipif(not _HAS_TORCH, reason="torch required")
class TestStateWindowBuffer:
    def test_empty_window(self):
        from core.models.transformer_encoder import StateWindowBuffer
        buf = StateWindowBuffer(window_size=4, state_dim=512)
        w = buf.get_window()
        assert w.shape == (4, 512)
        assert w.sum().item() == 0.0

    def test_add_and_get(self):
        from core.models.transformer_encoder import StateWindowBuffer
        buf = StateWindowBuffer(window_size=3, state_dim=512)
        buf.add(torch.ones(512))
        buf.add(torch.ones(512) * 2)
        w = buf.get_window()
        assert w.shape == (3, 512)
        assert w[-1].sum().item() == 512 * 2

    def test_rolling(self):
        from core.models.transformer_encoder import StateWindowBuffer
        buf = StateWindowBuffer(window_size=2, state_dim=512)
        buf.add(torch.ones(512) * 1)
        buf.add(torch.ones(512) * 2)
        buf.add(torch.ones(512) * 3)
        assert buf.size == 2
        w = buf.get_window()
        assert w[0].mean().item() == pytest.approx(2.0)
        assert w[1].mean().item() == pytest.approx(3.0)

    def test_reset(self):
        from core.models.transformer_encoder import StateWindowBuffer
        buf = StateWindowBuffer()
        buf.add(torch.randn(512))
        buf.reset()
        assert buf.size == 0
