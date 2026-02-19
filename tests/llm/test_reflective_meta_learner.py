"""Tests for B3: Reflective Meta-Learning."""
from __future__ import annotations

import os
import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


class TestReflectiveMetaLearner:
    def test_import(self):
        from core.llm.reflective_meta_learner import ReflectiveMetaLearner
        rml = ReflectiveMetaLearner()
        assert rml is not None

    def test_reflect_disabled(self):
        from core.llm.reflective_meta_learner import ReflectiveMetaLearner, ReflectionConfig
        rml = ReflectiveMetaLearner(config=ReflectionConfig(enabled=False))
        result = rml.reflect_on_episode({"total_reward": 10.0})
        assert result.insights == []

    def test_reflect_negative_reward(self):
        from core.llm.reflective_meta_learner import ReflectiveMetaLearner
        rml = ReflectiveMetaLearner()
        result = rml.reflect_on_episode({"total_reward": -5.0, "steps": 10, "highest_phase": "RECON"})
        assert len(result.insights) > 0

    def test_reflect_stuck(self):
        from core.llm.reflective_meta_learner import ReflectiveMetaLearner
        rml = ReflectiveMetaLearner()
        result = rml.reflect_on_episode({"steps": 50, "highest_phase": "RECON", "total_reward": 0})
        assert any("stuck" in i.lower() or "Stuck" in i for i in result.insights)

    def test_context_injection_empty(self):
        from core.llm.reflective_meta_learner import ReflectiveMetaLearner
        rml = ReflectiveMetaLearner()
        assert rml.get_context_injection() == ""

    def test_context_injection_with_data(self):
        from core.llm.reflective_meta_learner import ReflectiveMetaLearner
        rml = ReflectiveMetaLearner()
        rml.reflect_on_episode({"total_reward": -5.0, "steps": 10, "highest_phase": "RECON"})
        ctx = rml.get_context_injection()
        assert "REFLECTION" in ctx

    def test_history_window(self):
        from core.llm.reflective_meta_learner import ReflectiveMetaLearner, ReflectionConfig
        rml = ReflectiveMetaLearner(config=ReflectionConfig(history_window=3))
        for i in range(5):
            rml.reflect_on_episode({"total_reward": float(i)})
        assert len(rml.history) == 3

    def test_empty_episode_data(self):
        from core.llm.reflective_meta_learner import ReflectiveMetaLearner
        rml = ReflectiveMetaLearner()
        result = rml.reflect_on_episode({})
        assert result is not None
        assert result.episode == 1
