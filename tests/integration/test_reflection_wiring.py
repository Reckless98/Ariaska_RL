"""Tests for Phase 42 Stage 1C: ReflectiveMetaLearner wiring into SmartOrchestrator."""

from __future__ import annotations

import os
import pytest
from unittest.mock import MagicMock, patch

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


class TestReflectionWiring:
    """Verify ReflectiveMetaLearner is lazily wired into SmartOrchestrator."""

    def test_ensure_meta_learner_inits(self, minimal_orchestrator):
        """_ensure_meta_learner() returns non-None when flag is on."""
        meta = minimal_orchestrator._ensure_meta_learner()
        assert meta is not None
        assert minimal_orchestrator._meta_learner is not None

    def test_reflect_on_episode_callable(self, minimal_orchestrator):
        """reflect_on_episode can be called without error."""
        meta = minimal_orchestrator._ensure_meta_learner()
        assert meta is not None
        # Call reflect_on_episode with test data
        episode_data = {
            "episode": 1,
            "total_reward": 10.0,
            "max_phase": "RECON",
            "steps": 50,
            "discoveries": 5,
        }
        meta.reflect_on_episode(episode_data, gpt_manager=minimal_orchestrator.gpt_manager)

    def test_context_injection_returns_string(self, minimal_orchestrator):
        """get_context_injection() returns a string."""
        meta = minimal_orchestrator._ensure_meta_learner()
        assert meta is not None
        ctx = meta.get_context_injection()
        assert isinstance(ctx, str)

    def test_reflection_context_empty_when_none(self, minimal_orchestrator):
        """When _meta_learner is None, _reflection_context stays empty."""
        minimal_orchestrator._meta_learner = None
        assert minimal_orchestrator._reflection_context == ""

    def test_flag_off_skips_init(self, minimal_orchestrator, monkeypatch):
        """When flag is off, _ensure_meta_learner() returns None."""
        monkeypatch.setenv("FF_REFLECTIVE_META_LEARNER", "0")
        from core.feature_flags import reset_feature_flags
        reset_feature_flags()
        assert minimal_orchestrator._ensure_meta_learner() is None
