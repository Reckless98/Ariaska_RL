#!/usr/bin/env python3
"""
tests/test_cloud_roles.py — Phase 9.7: Cloud LLM role tests

Tests for LLM acceleration roles — all OFF by default.
Verifies feature-flag gating, budget limits, and factory behavior.
"""

import pytest
from unittest.mock import MagicMock

from core.llm.cloud_roles import (
    LLMRole, BaseLLMRole, RoleConfig,
    StrategicPlanner, TacticalAdvisor, JudgeRanker,
    PostmortemSkillExtractor, DAggerCorrector,
    get_role, reset_role_cache,
)
from core.feature_flags import get_feature_flags, set_feature_flag, reset_feature_flags


class TestFeatureFlagGating:
    """All cloud LLM roles should be OFF by default."""

    def setup_method(self):
        reset_feature_flags()
        reset_role_cache()

    def teardown_method(self):
        reset_feature_flags()
        reset_role_cache()

    def test_all_roles_on_by_default(self):
        """Post-Phase 20: All flags default ON for max intelligence."""
        ff = get_feature_flags()
        assert ff.llm_strategic_planner is True
        assert ff.llm_tactical_advisor is True
        assert ff.llm_judge_ranker is True
        assert ff.llm_postmortem_skills is True
        assert ff.dagger_corrections is True

    def test_get_role_returns_none_when_off(self):
        gpt = MagicMock()
        # Explicitly turn off to test gating still works
        for flag in ["llm_strategic_planner", "llm_tactical_advisor",
                      "llm_judge_ranker", "llm_postmortem_skills", "dagger_corrections"]:
            set_feature_flag(flag, False)
        for role in LLMRole:
            assert get_role(role, gpt) is None

    def test_get_role_returns_instance_when_on(self):
        gpt = MagicMock()
        # Already ON by default, but be explicit
        set_feature_flag("llm_strategic_planner", True)
        planner = get_role(LLMRole.STRATEGIC_PLANNER, gpt)
        assert planner is not None
        assert planner.enabled is True
        assert isinstance(planner, StrategicPlanner)

    def test_role_enabled_respects_flag_toggle(self):
        gpt = MagicMock()
        role = StrategicPlanner(gpt_manager=gpt)
        assert role.enabled is True  # ON by default now
        set_feature_flag("llm_strategic_planner", False)
        assert role.enabled is False
        set_feature_flag("llm_strategic_planner", True)
        assert role.enabled is True


class TestBudgetLimits:
    """LLM roles respect per-episode call budgets."""

    def setup_method(self):
        reset_feature_flags()
        set_feature_flag("llm_tactical_advisor", True)

    def teardown_method(self):
        reset_feature_flags()

    def test_can_call_respects_budget(self):
        gpt = MagicMock()
        advisor = TacticalAdvisor(gpt_manager=gpt, config=RoleConfig(max_calls_per_episode=2))
        assert advisor.can_call() is True
        advisor._calls_this_episode = 2
        assert advisor.can_call() is False

    def test_reset_episode_clears_calls(self):
        gpt = MagicMock()
        advisor = TacticalAdvisor(gpt_manager=gpt, config=RoleConfig(max_calls_per_episode=5))
        advisor._calls_this_episode = 5
        assert advisor.can_call() is False
        advisor.reset_episode()
        assert advisor.can_call() is True
        assert advisor._calls_this_episode == 0

    def test_can_call_requires_gpt_manager(self):
        set_feature_flag("llm_tactical_advisor", True)
        advisor = TacticalAdvisor(gpt_manager=None)
        assert advisor.can_call() is False


class TestStrategicPlanner:
    def setup_method(self):
        reset_feature_flags()
        set_feature_flag("llm_strategic_planner", True)

    def teardown_method(self):
        reset_feature_flags()

    def test_generate_plan_disabled(self):
        reset_feature_flags()
        set_feature_flag("llm_strategic_planner", False)  # Explicitly disable
        gpt = MagicMock()
        planner = StrategicPlanner(gpt_manager=gpt)
        result = planner.generate_plan("10.0.0.1", "ms2", ["ssh", "ftp"])
        assert "disabled" in result["reasoning"]
        gpt.gpt_request.assert_not_called()

    def test_generate_plan_calls_gpt(self):
        gpt = MagicMock()
        gpt.gpt_request.return_value = '{"phases": ["recon", "exploit"]}'
        planner = StrategicPlanner(gpt_manager=gpt)
        result = planner.generate_plan("10.0.0.1", "ms2", ["ssh", "ftp"])
        gpt.gpt_request.assert_called_once()
        assert result["source"] == "llm_planner"


class TestTacticalAdvisor:
    def setup_method(self):
        reset_feature_flags()
        set_feature_flag("llm_tactical_advisor", True)

    def teardown_method(self):
        reset_feature_flags()

    def test_assess_command_approved(self):
        gpt = MagicMock()
        gpt.gpt_request.return_value = "APPROVE: Command is appropriate for this phase."
        advisor = TacticalAdvisor(gpt_manager=gpt)
        result = advisor.assess_command("nmap -sV 10.0.0.1", "RECON", {"ports": []})
        assert result["approved"] is True

    def test_assess_command_redirect(self):
        gpt = MagicMock()
        gpt.gpt_request.return_value = "REDIRECT: Too noisy for recon phase."
        advisor = TacticalAdvisor(gpt_manager=gpt)
        result = advisor.assess_command("exploit foo", "RECON", {"ports": []})
        assert result["approved"] is False


class TestJudgeRanker:
    def setup_method(self):
        reset_feature_flags()
        set_feature_flag("llm_judge_ranker", True)

    def teardown_method(self):
        reset_feature_flags()

    def test_rank_empty_candidates(self):
        gpt = MagicMock()
        ranker = JudgeRanker(gpt_manager=gpt)
        result = ranker.rank_candidates([], "RECON")
        assert result == []
        gpt.gpt_request.assert_not_called()

    def test_rank_candidates_adds_llm_rank(self):
        gpt = MagicMock()
        gpt.gpt_request.return_value = "2, 1, 3"
        candidates = [
            {"command": "nmap -sV", "template_name": "nmap_version", "source": "registry", "confidence": 0.5},
            {"command": "nmap -sS", "template_name": "nmap_syn", "source": "ppo", "confidence": 0.7},
        ]
        ranker = JudgeRanker(gpt_manager=gpt)
        result = ranker.rank_candidates(candidates, "RECON")
        assert len(result) == 2
        assert all("llm_rank" in c for c in result)


class TestDAggerCorrector:
    def setup_method(self):
        reset_feature_flags()
        set_feature_flag("dagger_corrections", True)

    def teardown_method(self):
        reset_feature_flags()

    def test_no_correction_for_positive_reward(self):
        gpt = MagicMock()
        corrector = DAggerCorrector(gpt_manager=gpt)
        result = corrector.get_correction("state", "nmap", 5.0, "RECON", ["nmap", "nikto"])
        assert result is None
        gpt.gpt_request.assert_not_called()

    def test_correction_for_negative_reward(self):
        gpt = MagicMock()
        gpt.gpt_request.return_value = "nikto -h 10.0.0.1"
        corrector = DAggerCorrector(gpt_manager=gpt)
        result = corrector.get_correction("state", "nmap -sV", -3.0, "RECON", ["nmap", "nikto"])
        assert result is not None
        assert result["source"] == "dagger"
        assert "nikto" in result["expert_command"]


class TestGetStats:
    def setup_method(self):
        reset_feature_flags()

    def teardown_method(self):
        reset_feature_flags()

    def test_stats_structure(self):
        gpt = MagicMock()
        set_feature_flag("llm_tactical_advisor", False)  # Explicitly disable for test
        advisor = TacticalAdvisor(gpt_manager=gpt)
        stats = advisor.get_stats()
        assert stats["role"] == "tactical_advisor"
        assert stats["enabled"] is False
        assert stats["calls_this_episode"] == 0
        assert stats["total_calls"] == 0
        assert "budget" in stats


class TestRoleCache:
    def setup_method(self):
        reset_feature_flags()
        reset_role_cache()

    def teardown_method(self):
        reset_feature_flags()
        reset_role_cache()

    def test_cache_returns_same_instance(self):
        set_feature_flag("llm_tactical_advisor", True)
        gpt = MagicMock()
        a1 = get_role(LLMRole.TACTICAL_ADVISOR, gpt)
        a2 = get_role(LLMRole.TACTICAL_ADVISOR, gpt)
        assert a1 is a2

    def test_reset_clears_cache(self):
        set_feature_flag("llm_tactical_advisor", True)
        gpt = MagicMock()
        a1 = get_role(LLMRole.TACTICAL_ADVISOR, gpt)
        reset_role_cache()
        a2 = get_role(LLMRole.TACTICAL_ADVISOR, gpt)
        assert a1 is not a2
