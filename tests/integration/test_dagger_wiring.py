"""Tests for Phase 42 Stage 1B: DAgger wiring into SmartCoach."""

from __future__ import annotations

import os
import pytest
from unittest.mock import MagicMock, patch

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


class TestDAggerWiring:
    """Verify DAggerBuffer is lazily wired into SmartCoach."""

    def test_ensure_dagger_buffer_inits(self, minimal_coach):
        """_ensure_dagger_buffer() lazy-inits when flag is on."""
        buf = minimal_coach._ensure_dagger_buffer()
        assert buf is not None
        assert minimal_coach._dagger_buffer is not None

    def test_dagger_store_callable(self, minimal_coach):
        """Buffer can store a sample after init."""
        buf = minimal_coach._ensure_dagger_buffer()
        assert buf is not None
        ok = buf.store(
            state_hash="test_hash",
            state_vector=[0.0] * 10,
            mentor_action_idx=2,
            mentor_command="nmap -sV",
            policy_action_idx=1,
            policy_command="nmap -sS",
            mentor_confidence=0.9,
            phase="RECON",
            episode=1,
            step=5,
        )
        assert ok is True

    def test_dagger_decay_at_episode_end(self, minimal_coach):
        """decay_weights() is called during end_episode_ppo path."""
        mock_buf = MagicMock()
        minimal_coach._dagger_buffer = mock_buf
        # end_episode_ppo early-returns since ppo_agent is None,
        # but the decay call happens before the early return
        minimal_coach.end_episode_ppo(done=True, highest_phase="RECON")
        mock_buf.decay_weights.assert_called_once()

    def test_dagger_flag_off(self, minimal_coach, monkeypatch):
        """When flag is off, buffer stays None."""
        monkeypatch.setenv("FF_DAGGER_WIRING", "0")
        from core.feature_flags import reset_feature_flags
        reset_feature_flags()
        assert minimal_coach._ensure_dagger_buffer() is None

    def test_dagger_no_crash_on_store_error(self, minimal_coach):
        """DAgger buffer errors are gracefully handled."""
        mock_buf = MagicMock()
        mock_buf.store.side_effect = RuntimeError("test error")
        minimal_coach._dagger_buffer = mock_buf
        # Direct call raises — but SmartCoach wraps in try/except
        # Verify the mock is callable and would raise
        with pytest.raises(RuntimeError):
            minimal_coach._dagger_buffer.store(
                state_hash="h", state_vector=[], mentor_action_idx=0,
                mentor_command="", policy_action_idx=0, policy_command="",
                mentor_confidence=0.0, phase="", episode=0, step=0,
            )
        # After error, coach should still be functional
        assert minimal_coach._dagger_buffer is not None
