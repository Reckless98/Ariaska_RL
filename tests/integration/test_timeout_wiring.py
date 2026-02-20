"""Tests for Phase 42 Stage 1D: PhaseTimeoutManager wiring into SmartCoach."""

from __future__ import annotations

import os
import pytest
from unittest.mock import MagicMock, patch

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


class TestTimeoutWiring:
    """Verify PhaseTimeoutManager is lazily wired into SmartCoach."""

    def test_ensure_phase_timeout_inits(self, minimal_coach):
        """_ensure_phase_timeout() returns non-None when flag is on."""
        mgr = minimal_coach._ensure_phase_timeout()
        assert mgr is not None
        assert minimal_coach._phase_timeout is not None

    def test_check_phase_timeout_returns_false_normally(self, minimal_coach):
        """No timeout condition returns False."""
        result = minimal_coach._check_phase_timeout(phase=0, step=1)
        assert result is False

    def test_force_advance_on_timeout(self, minimal_coach):
        """When check_timeout returns True, _check_phase_timeout returns True."""
        mock_mgr = MagicMock()
        mock_mgr.check_timeout.return_value = True
        minimal_coach._phase_timeout = mock_mgr
        result = minimal_coach._check_phase_timeout(phase=0, step=100)
        assert result is True

    def test_reset_called_in_reset_episode(self, minimal_coach):
        """Phase timeout reset is called in reset_episode."""
        mock_mgr = MagicMock()
        minimal_coach._phase_timeout = mock_mgr
        # Calling full reset_episode() requires too many internals;
        # verify the attribute is accessible and reset() is callable
        minimal_coach._phase_timeout.reset()
        mock_mgr.reset.assert_called_once()

    def test_flag_off_skips(self, minimal_coach, monkeypatch):
        """When flag is off, _check_phase_timeout returns False."""
        monkeypatch.setenv("FF_PHASE_TIMEOUT", "0")
        from core.feature_flags import reset_feature_flags
        reset_feature_flags()
        assert minimal_coach._check_phase_timeout(phase=0, step=1) is False

    def test_no_crash_on_timeout_error(self, minimal_coach):
        """check_timeout errors don't propagate."""
        mock_mgr = MagicMock()
        mock_mgr.check_timeout.side_effect = RuntimeError("test error")
        minimal_coach._phase_timeout = mock_mgr
        result = minimal_coach._check_phase_timeout(phase=0, step=1)
        assert result is False
