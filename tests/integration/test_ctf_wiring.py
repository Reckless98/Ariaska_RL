"""Tests for Phase 42 Stage 1E: CTFModeTracker wiring into SmartCoach."""

from __future__ import annotations

import os
import pytest
from unittest.mock import MagicMock

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


class TestCTFWiring:
    """Verify CTFModeTracker is lazily wired into SmartCoach."""

    def test_ensure_ctf_tracker_inits(self, minimal_coach):
        """_ensure_ctf_tracker() returns non-None when flag is on."""
        tracker = minimal_coach._ensure_ctf_tracker()
        assert tracker is not None
        assert minimal_coach._ctf_tracker is not None

    def test_scan_output_callable(self, minimal_coach):
        """scan_output can be called without error."""
        tracker = minimal_coach._ensure_ctf_tracker()
        assert tracker is not None
        result = tracker.scan_output(
            "Some output with FLAG{test123}", "cat /flag.txt", "RedAgent"
        )
        # Result type depends on CTFModeTracker implementation
        assert isinstance(result, list)

    def test_no_crash_on_empty_output(self, minimal_coach):
        """Empty output doesn't crash."""
        tracker = minimal_coach._ensure_ctf_tracker()
        if tracker is not None:
            result = tracker.scan_output("", "ls", "RedAgent")
            assert isinstance(result, list)

    def test_flag_off_skips(self, minimal_coach, monkeypatch):
        """When flag is off, _ensure_ctf_tracker returns None."""
        monkeypatch.setenv("FF_CTF_TRACKER", "0")
        from core.feature_flags import reset_feature_flags
        reset_feature_flags()
        assert minimal_coach._ensure_ctf_tracker() is None
