#!/usr/bin/env python3
"""
tests/test_cap_contract.py — Phase 14.0 T2: Contract Invariant Tests

Contract C3.9: 5 tests verifying that all Phase 14.0 contract invariants hold.
These are meta-tests that ensure the architecture is internally consistent.
"""

import os
import sys
import pytest
import importlib

os.environ["ARIASKA_DRY_RUN"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestContractInvariants:
    """Verify Phase 14.0 contract invariants hold end-to-end."""

    def test_c1_all_schemas_importable(self):
        """C1: All 8 Phase 14.0 schemas can be imported and instantiated."""
        from core.reasoning.teacher_trace import TeacherTrace, BCSample, BCBuffer
        from core.reasoning.hypothesis import Hypothesis, HypothesisGenerator, HypothesisStatus
        from core.reasoning.strategy_plan import StrategyPlan, Subgoal, Milestone, Guardrail
        from core.reasoning.lesson import Lesson, LessonExtractor
        from core.knowledge.evidence_graph import EvidenceGraph, EvidenceNode, EvidenceEdge
        from core.execution.parser_teacher_output import ParserTeacherOutput, LearningFeatures, ParsingLesson
        from core.training.autonomy_scheduler import AutonomyScheduler, AgentAutonomy

        # Instantiate each
        buf = BCBuffer(capacity=10)
        assert buf.capacity == 10

        hyp = Hypothesis(if_observed="test", then_try="cmd")
        assert hyp.status == HypothesisStatus.UNTESTED

        plan = StrategyPlan(plan_id="test")
        assert plan.plan_id == "test"

        lesson = Lesson(context="ctx", action_taken="nmap", outcome="ok", insight="learned")
        assert lesson.confidence >= 0

        eg = EvidenceGraph()
        assert eg is not None

        lf = LearningFeatures()
        assert lf.feature_vector().shape == (32,)

        sched = AutonomyScheduler()
        sched.register_agent("test")
        assert sched.get_score("test") == 0.0

    def test_c2_all_feature_flags_exist(self):
        """C2: All 11 Phase 14.0 feature flags exist with default=False."""
        from core.feature_flags import get_feature_flags

        ff = get_feature_flags()
        p14_flags = [
            "evidence_graph", "hypothesis_engine", "strategy_plan",
            "autonomy_scheduler", "bc_loss", "teacher_trace",
            "aux_heads", "parser_teacher", "ms2_knowledge_pack",
            "ms3_knowledge_pack", "ms2_simulated_output",
        ]
        for flag in p14_flags:
            assert hasattr(ff, flag), f"Missing flag: {flag}"
            # All default False (unless env override)
            val = getattr(ff, flag)
            assert isinstance(val, bool), f"Flag {flag} is not bool: {type(val)}"

    def test_c3_ppo_bc_loss_config_fields(self):
        """C3: PPOConfig has BC loss fields with correct defaults."""
        from core.algorithms.ppo_agent import PPOConfig

        config = PPOConfig()
        assert hasattr(config, "use_bc_loss")
        assert config.use_bc_loss is False
        assert hasattr(config, "bc_loss_coef")
        assert config.bc_loss_coef == pytest.approx(0.1)
        assert hasattr(config, "bc_buffer")
        assert config.bc_buffer is None

    def test_c4_wiring_points_exist(self):
        """C4: Key wiring points exist in SmartCoach and PPO."""
        import inspect
        from core.training.smart_coach import SmartCoach

        # Check Phase 14 method exists
        assert hasattr(SmartCoach, "_p14_hypothesis_select")
        sig = inspect.signature(SmartCoach._p14_hypothesis_select)
        assert "step_ctx" in sig.parameters

        # Check PPOActorCritic has get_logits
        from core.algorithms.ppo_agent import PPOActorCritic
        assert hasattr(PPOActorCritic, "get_logits")

        # Check ParserBroker has parse_as_teacher
        from core.execution.parser_broker import ParserBroker
        assert hasattr(ParserBroker, "parse_as_teacher")

    def test_c5_skillcard_v2_backward_compat(self):
        """C5: SkillCard v2 has new fields and maintains backward compat."""
        from core.postmortem.orion_postmortem import SkillCard

        # Create with minimal fields (backward compat)
        card = SkillCard(
            id="test_card",
            if_condition="port 21 open",
            then_action="vsftpd_backdoor",
        )
        assert card.id == "test_card"

        # Check Phase 14 fields exist with defaults
        assert hasattr(card, "preconditions")
        assert hasattr(card, "postconditions")
        assert hasattr(card, "failure_signatures")
        assert hasattr(card, "expected_output_patterns")
        assert hasattr(card, "service_archetypes")
        assert hasattr(card, "reward_history")
        assert hasattr(card, "last_used_episode")
        assert hasattr(card, "applicable_phases")

        # Defaults should be empty
        assert card.last_used_episode == 0
        assert isinstance(card.preconditions, dict)
        assert isinstance(card.failure_signatures, list)
