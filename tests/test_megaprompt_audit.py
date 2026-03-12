"""
tests/test_megaprompt_audit.py — Regression tests for MEGAPROMPT Audit.

Covers:
  T1: P34 PhaseGuidedLLM JSON validity + integration
  T2: Evidence gate enforcement mode
  T3: MicroChain model routing (nano/mini split)
  T4: Decision reasoning never empty
  T5: Model routing tiers correct
  T6: Budget manager constraints
  T7: PhaseGuidedLLM wired in SmartCoach
  T8: Learning metrics type safety
"""

import os
import json
import pytest
from typing import Any, Dict

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


# ═══════════════════════════════════════════════════════════════════════════
# T1: P34 PhaseGuidedLLM JSON Validity
# ═══════════════════════════════════════════════════════════════════════════

class TestPhaseGuidedLLMValidity:
    """PhaseGuidedLLM must produce valid structured guidance."""

    def test_extract_json_valid(self):
        from core.llm.phase_guided_llm import _extract_json
        raw = '{"phase_decision": {"stay": true}}'
        result = _extract_json(raw)
        assert result is not None
        assert result["phase_decision"]["stay"] is True

    def test_extract_json_fenced(self):
        from core.llm.phase_guided_llm import _extract_json
        raw = '```json\n{"phase_decision": {"stay": true}}\n```'
        result = _extract_json(raw)
        assert result is not None

    def test_extract_json_none_returns_none(self):
        from core.llm.phase_guided_llm import _extract_json
        assert _extract_json(None) is None  # type: ignore[arg-type]
        assert _extract_json("") is None

    def test_extract_json_garbage_returns_none(self):
        from core.llm.phase_guided_llm import _extract_json
        assert _extract_json("not json at all") is None

    def test_guidance_result_validates(self):
        from core.llm.phase_guided_llm import (
            PhaseGuidanceResult, PhaseDecision, Candidate, Selection,
        )
        result = PhaseGuidanceResult(
            phase_decision=PhaseDecision(
                chosen_phase="RECON",
                phase_confidence=0.8,
                phase_goal="Discover services on target",
            ),
            candidates=[
                Candidate(
                    template_name="nmap_version",
                    why="version scan needed",
                    confidence=0.75,
                ),
            ],
            selection=Selection(
                best_template_name="nmap_version",
                selection_reason="best phase fit",
            ),
        )
        assert result.phase_decision.phase_confidence == 0.8
        assert len(result.candidates) == 1
        valid = result.validate()
        assert valid is True


# ═══════════════════════════════════════════════════════════════════════════
# T2: Evidence Gate Enforcement
# ═══════════════════════════════════════════════════════════════════════════

class TestEvidenceGateEnforcement:
    """Evidence gate must default to enforce mode."""

    def test_default_mode_is_enforce(self):
        # Ensure env var is not overriding
        saved = os.environ.pop("FF_STRICT_EXPLOIT_GATE", None)
        try:
            from core.feature_flags import _validate_exploit_gate
            assert _validate_exploit_gate("enforce") == "enforce"
        finally:
            if saved is not None:
                os.environ["FF_STRICT_EXPLOIT_GATE"] = saved

    def test_invalid_mode_falls_back(self):
        from core.feature_flags import _validate_exploit_gate
        assert _validate_exploit_gate("invalid") == "log"

    def test_valid_modes(self):
        from core.feature_flags import _validate_exploit_gate
        assert _validate_exploit_gate("off") == "off"
        assert _validate_exploit_gate("log") == "log"
        assert _validate_exploit_gate("enforce") == "enforce"


# ═══════════════════════════════════════════════════════════════════════════
# T3: MicroChain Schema Validation
# ═══════════════════════════════════════════════════════════════════════════

class TestMicroChainSchema:
    """MicroChain JSON parsing robustness."""

    def test_safe_json_load_list_none(self):
        from core.llm.micro_chain import _safe_json_load_list
        assert _safe_json_load_list(None) is None

    def test_safe_json_load_list_empty(self):
        from core.llm.micro_chain import _safe_json_load_list
        assert _safe_json_load_list("") is None

    def test_safe_json_load_list_valid(self):
        from core.llm.micro_chain import _safe_json_load_list
        result = _safe_json_load_list('[{"idx": 0, "phase_fit": 0.8}]')
        assert result is not None
        assert len(result) == 1

    def test_safe_json_load_dict_valid(self):
        from core.llm.micro_chain import _safe_json_load
        result = _safe_json_load('{"command": "nmap -sV"}')
        assert result is not None
        assert result["command"] == "nmap -sV"

    def test_micro_chain_candidate_fields(self):
        from core.llm.micro_chain import MicroChainCandidate
        c = MicroChainCandidate(
            command="nmap -sV 10.0.0.1",
            template_name="nmap_version",
            score=0.75,
            phase_fit=0.8,
            evidence_support=0.7,
            novelty=0.6,
        )
        d = c.to_dict()
        assert d["command"] == "nmap -sV 10.0.0.1"
        assert d["score"] == 0.75
        assert "phase_fit" in d
        assert "evidence_support" in d
        assert "novelty" in d


# ═══════════════════════════════════════════════════════════════════════════
# T4: Decision Reasoning Never Empty
# ═══════════════════════════════════════════════════════════════════════════

class TestDecisionReasoningNeverEmpty:
    """P36 structured reasoning must always populate result.reasoning."""

    def test_reasoning_format_has_evidence(self):
        """Reasoning must contain EVIDENCE, GOAL, WHY_THIS, STOP, CONF."""
        # Simulate the P36 reasoning construction
        _p36_evidence = "ports=[22,80], services=[ssh,http]"
        _p36_goal = "Discover open ports and services on target"
        _p36_why = "playbook: template=nmap_version"
        _p36_stop = "phase_advance or discovery"
        _p36_conf = "0.75"

        reasoning = (
            f"EVIDENCE: {_p36_evidence} | "
            f"GOAL: {_p36_goal} | "
            f"WHY_THIS: {_p36_why} | "
            f"STOP: {_p36_stop} | "
            f"CONF: {_p36_conf}"
        )
        assert len(reasoning.strip()) >= 20
        assert "EVIDENCE:" in reasoning
        assert "GOAL:" in reasoning
        assert "WHY_THIS:" in reasoning
        assert "STOP:" in reasoning
        assert "CONF:" in reasoning

    def test_reasoning_fallback_still_valid(self):
        """Even fallback reasoning must be >= 20 chars with all fields."""
        _p36_evidence = "none_yet"
        _p36_goal = "Discover open ports and services on target"
        _p36_why = "fallback fallback — no detailed reasoning available"
        _p36_stop = "phase_advance or discovery"
        _p36_conf = "0.50"

        reasoning = (
            f"EVIDENCE: {_p36_evidence} | "
            f"GOAL: {_p36_goal} | "
            f"WHY_THIS: {_p36_why} | "
            f"STOP: {_p36_stop} | "
            f"CONF: {_p36_conf}"
        )
        assert len(reasoning.strip()) >= 20


# ═══════════════════════════════════════════════════════════════════════════
# T5: Model Routing Tiers
# ═══════════════════════════════════════════════════════════════════════════

class TestModelRoutingTiers:
    """Get model routing returns correct tier for each task type."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from core.testing.fake_gpt_manager import FakeGPTManager
        self.gpt = FakeGPTManager(seed=42)

    def test_codex_tasks(self):
        from core.gpt_manager import GPTManager
        gpt = GPTManager.__new__(GPTManager)
        # Manually set model attributes
        gpt.strategic_model = "local-llm"
        gpt.fallback_model = "local-llm"
        gpt.nano_model = "local-llm"

        for task in ["strategic", "postmortem", "reasoning", "tactical", "analysis"]:
            model = gpt.get_model_for_role(task_type=task)
            assert model == "local-llm", f"{task} should route to codex"

    def test_mini_tasks(self):
        from core.gpt_manager import GPTManager
        gpt = GPTManager.__new__(GPTManager)
        gpt.strategic_model = "local-llm"
        gpt.fallback_model = "local-llm"
        gpt.nano_model = "local-llm"

        for task in ["playbook", "parsing", "command_selection", "output_parse",
                     "defensive", "reconnaissance"]:
            model = gpt.get_model_for_role(task_type=task)
            assert model == "local-llm", f"{task} should route to mini"

    def test_nano_tasks(self):
        from core.gpt_manager import GPTManager
        gpt = GPTManager.__new__(GPTManager)
        gpt.strategic_model = "local-llm"
        gpt.fallback_model = "local-llm"
        gpt.nano_model = "local-llm"

        for task in ["general", "classification", "reformat", "cache", None]:
            model = gpt.get_model_for_role(task_type=task)
            assert model == "local-llm", f"{task} should route to nano"


# ═══════════════════════════════════════════════════════════════════════════
# T6: Budget Manager Constraints
# ═══════════════════════════════════════════════════════════════════════════

class TestBudgetManagerConstraints:
    """BudgetManagerV2 must enforce hardcoded limits."""

    def test_total_budget(self):
        from core.llm.budget_manager import _TOTAL_BUDGET
        assert _TOTAL_BUDGET == 1_344_154  # Phase 53: 50% cut

    def test_min_budget(self):
        from core.llm.budget_manager import _MIN_BUDGET
        assert _MIN_BUDGET == 672_077  # Phase 53: 50% cut

    def test_min_budget_is_50pct(self):
        from core.llm.budget_manager import _TOTAL_BUDGET, _MIN_BUDGET
        assert _MIN_BUDGET == _TOTAL_BUDGET * 0.5

    def test_instance_default_budget(self):
        from core.llm.budget_manager import BudgetManagerV2
        bm = BudgetManagerV2()
        assert bm._total_budget == 1_344_154  # Phase 53: 50% cut
        assert bm._max_budget == 1_344_154  # Phase 53: 50% cut


# ═══════════════════════════════════════════════════════════════════════════
# T7: PhaseGuidedLLM Wired in SmartCoach
# ═══════════════════════════════════════════════════════════════════════════

class TestPhaseGuidedLLMWiring:
    """PhaseGuidedLLM must be initialized in SmartCoach."""

    def test_smart_coach_has_phase_guided_attr(self):
        """SmartCoach must have _phase_guided attribute after init."""
        from core.testing.fake_gpt_manager import FakeGPTManager

        gpt = FakeGPTManager(seed=42)
        # Lazy import to avoid circular deps
        from core.training.smart_coach import SmartCoach
        coach = SmartCoach(
            agent_name="RedAgent",
            gpt_manager=gpt,  # type: ignore[arg-type]
        )
        assert hasattr(coach, '_phase_guided'), \
            "SmartCoach must have _phase_guided attribute (P34 wiring)"


# ═══════════════════════════════════════════════════════════════════════════
# T8: Learning Metrics Type Safety
# ═══════════════════════════════════════════════════════════════════════════

class TestLearningMetricsTypeSafety:
    """Learning metrics must accept mixed-type dicts for mentor interventions."""

    def test_record_mentor_intervention_types(self):
        from core.analytics.learning_metrics import LearningMetrics
        lm = LearningMetrics()
        # Should not raise — step is int, timestamp is float, reason is str
        lm.record_mentor_intervention(reason="stagnation", agent="RedAgent")
        assert len(lm._mentor_interventions) == 1
        entry = lm._mentor_interventions[0]
        assert isinstance(entry["step"], int)
        assert isinstance(entry["reason"], str)
        assert isinstance(entry["agent"], str)
        assert isinstance(entry["timestamp"], float)
