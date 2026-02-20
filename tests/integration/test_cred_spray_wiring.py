"""Tests for Phase 42 Stage 1F: CredentialSprayer wiring into SmartCoach."""

from __future__ import annotations

import os
import pytest
from unittest.mock import MagicMock

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


class TestCredSprayWiring:
    """Verify CredentialSprayer is lazily wired into SmartCoach."""

    def test_ensure_cred_sprayer_inits(self, minimal_coach):
        """_ensure_cred_sprayer() returns non-None when flag is on."""
        sprayer = minimal_coach._ensure_cred_sprayer()
        assert sprayer is not None
        assert minimal_coach._cred_sprayer is not None

    def test_register_credential_callable(self, minimal_coach):
        """register_credential can be called without error."""
        sprayer = minimal_coach._ensure_cred_sprayer()
        assert sprayer is not None
        sprayer.register_credential(
            username="admin", password="password123", source="hydra",
        )

    def test_register_service_callable(self, minimal_coach):
        """register_service can be called without error."""
        sprayer = minimal_coach._ensure_cred_sprayer()
        assert sprayer is not None
        sprayer.register_service(
            host="10.10.10.1", port=22, service="ssh",
        )

    def test_get_spray_commands_returns_list(self, minimal_coach):
        """get_spray_commands returns a list."""
        result = minimal_coach.get_spray_commands()
        assert isinstance(result, list)

    def test_get_spray_commands_returns_empty_when_none(self, minimal_coach):
        """When _cred_sprayer is None, returns empty list."""
        minimal_coach._cred_sprayer = None
        result = minimal_coach.get_spray_commands()
        assert result == []

    def test_flag_off_skips(self, minimal_coach, monkeypatch):
        """When flag is off, _ensure_cred_sprayer returns None."""
        monkeypatch.setenv("FF_CREDENTIAL_SPRAYER", "0")
        from core.feature_flags import reset_feature_flags
        reset_feature_flags()
        assert minimal_coach._ensure_cred_sprayer() is None
