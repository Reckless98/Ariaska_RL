#!/usr/bin/env python3
"""
tests/test_phase110_visibility.py — Phase 11.0 Full Visibility tests

Tests for:
- Feature flags (Phase 11.0 additions)
- UnifiedStepTrace schema and serialization
- AdaptiveBudgetController budget pacing
- ParserBroker v2.0 dual-mode dispatch
- ToolValidator privilege validation
- LearningSignalExporter JSONL output
- StepEvent Phase 11.0 fields
- Phase ladder gate in SmartCoach
"""

import json
import os
import tempfile
import pytest

os.environ["ARIASKA_DRY_RUN"] = "1"


# ─── Feature Flags ──────────────────────────────────────────────────────────

class TestPhase110FeatureFlags:
    """Phase 11.0 feature flag additions."""

    def test_new_flags_exist(self):
        from core.feature_flags import FeatureFlags
        ff = FeatureFlags()
        assert hasattr(ff, "parser_mode")
        assert hasattr(ff, "strict_phase_ladder")
        assert hasattr(ff, "adaptive_budget")
        assert hasattr(ff, "learning_signal_export")

    def test_default_values(self):
        from core.feature_flags import FeatureFlags
        ff = FeatureFlags()
        assert ff.parser_mode == "fast"
        assert ff.strict_phase_ladder is False
        assert ff.adaptive_budget is True
        assert ff.learning_signal_export is True

    def test_set_feature_flag_parser_mode(self):
        from core.feature_flags import reset_feature_flags, set_feature_flag, get_feature_flags
        reset_feature_flags()
        set_feature_flag("parser_mode", "intelligent_fullparse")
        assert get_feature_flags().parser_mode == "intelligent_fullparse"
        reset_feature_flags()

    def test_resolve_profile_deterministic(self):
        from core.feature_flags import resolve_profile, get_feature_flags, reset_feature_flags
        reset_feature_flags()
        profile = resolve_profile()
        ff = get_feature_flags()
        assert profile == "DETERMINISTIC"
        # In deterministic mode, Phase 11.0 flags should be disabled
        assert ff.strict_phase_ladder is False
        assert ff.adaptive_budget is False
        assert ff.learning_signal_export is False
        reset_feature_flags()


# ─── UnifiedStepTrace ────────────────────────────────────────────────────────

class TestUnifiedStepTrace:
    """UnifiedStepTrace schema and serialization."""

    def test_defaults(self):
        from core.telemetry.unified_trace import UnifiedStepTrace
        trace = UnifiedStepTrace()
        d = trace.to_dict()
        assert d["agent_name"] == ""
        assert d["step"] == 0
        assert d["parse_mode"] == "fast"
        assert d["discovery_count"] == 0
        assert isinstance(d["parse_explanations"], list)
        assert isinstance(d["phase_state"], dict)
        assert isinstance(d["budget_snapshot"], dict)
        assert isinstance(d["teaching_points"], list)
        assert "ts" in d

    def test_populated(self):
        from core.telemetry.unified_trace import (
            UnifiedStepTrace, ParseExplanation, BudgetSnapshot, PhaseState,
        )
        trace = UnifiedStepTrace(
            episode_id=1,
            step=5,
            agent_name="RedAgent",
            command="nmap -sV 10.0.0.1",
            source="ppo",
            confidence=0.9,
            parse_mode="intelligent_fullparse",
            discovery_count=2,
            reward_total=12.5,
            teaching_points=["Phase RECON: step 1/3 minimum"],
        )
        d = trace.to_dict()
        assert d["agent_name"] == "RedAgent"
        assert d["step"] == 5
        assert d["parse_mode"] == "intelligent_fullparse"
        assert d["discovery_count"] == 2
        assert d["reward_total"] == 12.5
        assert len(d["teaching_points"]) == 1

    def test_parse_explanation_serialization(self):
        from core.telemetry.unified_trace import ParseExplanation
        pe = ParseExplanation(
            stage="regex",
            stage_number=1,
            discovery_type="open_port",
            discovery_value="22",
            confidence=1.0,
            reasoning="Regex matched 'PORT 22/tcp open'",
        )
        d = pe.to_dict()
        assert d["stage"] == "regex"
        assert d["stage_number"] == 1
        assert d["discovery_type"] == "open_port"
        assert d["discovery_value"] == "22"

    def test_budget_snapshot_serialization(self):
        from core.telemetry.unified_trace import BudgetSnapshot
        bs = BudgetSnapshot(
            mentor_budget_remaining=20,
            mentor_budget_total=30,
            budget_pressure=0.33,
        )
        d = bs.to_dict()
        assert d["mentor_remaining"] == 20
        assert d["mentor_total"] == 30
        assert d["budget_pressure"] == 0.33

    def test_phase_state_serialization(self):
        from core.telemetry.unified_trace import PhaseState
        ps = PhaseState(
            current_phase="EXPLOITATION",
            steps_in_phase=5,
            min_steps_required=2,
            can_advance=True,
        )
        d = ps.to_dict()
        assert d["current_phase"] == "EXPLOITATION"
        assert d["can_advance"] is True

    def test_from_decision_result(self):
        """Test building trace from a mock decision result."""
        from core.telemetry.unified_trace import UnifiedStepTrace
        from unittest.mock import MagicMock
        decision = MagicMock()
        decision.command = "nmap -sV target"
        decision.template_name = "nmap_version"
        decision.source = "registry"
        decision.confidence = 0.8
        decision.reasoning = "Registry selected nmap"
        decision.mentor_call = False
        decision.mentor_reasoning = ""
        decision.tokens_used = 0
        decision.agent_name = "ScoutAgent"

        trace = UnifiedStepTrace.from_decision_result(decision, episode_id=1, step=3)
        assert trace.command == "nmap -sV target"
        assert trace.source == "registry"
        assert trace.confidence == 0.8


# ─── AdaptiveBudgetController ────────────────────────────────────────────────

class TestAdaptiveBudgetController:
    """Adaptive budget pacing tests."""

    def test_init_defaults(self):
        from core.training.adaptive_budget import AdaptiveBudgetController
        budget = AdaptiveBudgetController()
        assert budget.config.mentor_budget_total == 30
        assert budget.config.token_budget_total == 50_000

    def test_reset_episode(self):
        from core.training.adaptive_budget import AdaptiveBudgetController
        budget = AdaptiveBudgetController()
        budget.record_mentor_call(tokens_used=500)
        budget.reset_episode(max_steps=40)
        snapshot = budget.get_snapshot()
        assert snapshot["mentor_budget_remaining"] == 30
        assert snapshot["token_budget_remaining"] == 50_000

    def test_pressure_increases(self):
        from core.training.adaptive_budget import AdaptiveBudgetController, BudgetConfig
        budget = AdaptiveBudgetController(config=BudgetConfig(mentor_budget_total=10))
        budget.reset_episode(max_steps=10)
        budget.step_tick(1)
        p1 = budget.get_pressure()
        # Use 5 of 10 calls
        for _ in range(5):
            budget.record_mentor_call(tokens_used=100)
        budget.step_tick(2)
        p2 = budget.get_pressure()
        assert p2 > p1  # Pressure should increase after heavy spending

    def test_can_call_mentor_basic(self):
        from core.training.adaptive_budget import AdaptiveBudgetController, BudgetConfig
        budget = AdaptiveBudgetController(config=BudgetConfig(mentor_budget_total=5))
        budget.reset_episode(max_steps=40)
        budget.step_tick(1)
        assert budget.can_call_mentor("RECON") is True
        # Exhaust budget
        for _ in range(5):
            budget.record_mentor_call()
        assert budget.can_call_mentor("RECON") is False

    def test_spend_rate_tracking(self):
        from core.training.adaptive_budget import AdaptiveBudgetController
        budget = AdaptiveBudgetController()
        budget.reset_episode(max_steps=40)
        budget.record_mentor_call()
        budget.record_no_call()
        budget.record_no_call()
        rate = budget.get_spend_rate()
        assert 0 < rate < 1  # 1 call out of 3 steps

    def test_get_summary(self):
        from core.training.adaptive_budget import AdaptiveBudgetController
        budget = AdaptiveBudgetController()
        budget.reset_episode(max_steps=40)
        budget.record_mentor_call(tokens_used=100)
        budget.record_venice_call(tokens_used=50)
        summary = budget.get_summary()
        assert summary["mentor_calls"] == 1
        assert summary["venice_calls"] == 1
        assert summary["tokens_used"] == 150


# ─── ParserBroker v2.0 ──────────────────────────────────────────────────────

class TestParserBrokerV2:
    """ParserBroker dual-mode dispatch tests."""

    def test_init_with_mode(self):
        from core.execution.parser_broker import ParserBroker
        broker = ParserBroker(default_mode="fast")
        assert broker._default_mode == "fast"

    def test_init_fullparse_mode(self):
        from core.execution.parser_broker import ParserBroker
        broker = ParserBroker(default_mode="intelligent_fullparse")
        assert broker._default_mode == "intelligent_fullparse"

    def test_stats_track_mode(self):
        from core.execution.parser_broker import ParserBroker
        broker = ParserBroker(default_mode="fast")
        broker.parse("nmap 10.0.0.1", "PORT 22/tcp open ssh", mode="fast")
        stats = broker.get_stats()
        assert stats["fast_calls"] >= 1
        assert stats["fullparse_calls"] == 0

    def test_empty_output_returns_empty(self):
        from core.execution.parser_broker import ParserBroker
        broker = ParserBroker(default_mode="fast")
        events = broker.parse("nmap 10.0.0.1", "")
        assert events == []

    def test_parse_with_explanations_returns_tuple(self):
        from core.execution.parser_broker import ParserBroker
        broker = ParserBroker(default_mode="fast")
        result = broker.parse_with_explanations(
            "nmap -sV 10.0.0.1",
            "22/tcp open ssh OpenSSH 7.9",
        )
        assert isinstance(result, tuple)
        assert len(result) == 4  # events, explanations, latency, stage

    def test_reset_episode(self):
        from core.execution.parser_broker import ParserBroker
        broker = ParserBroker(default_mode="fast")
        broker._llm_calls = 5
        broker._venice_calls = 3
        broker.reset_episode()
        assert broker._llm_calls == 0
        assert broker._venice_calls == 0

    def test_get_stage_distribution(self):
        from core.execution.parser_broker import ParserBroker
        broker = ParserBroker(default_mode="fast")
        dist = broker.get_stage_distribution()
        assert "regex_pct" in dist
        assert "venice_pct" in dist
        assert "gpt_pct" in dist


# ─── ToolValidator ───────────────────────────────────────────────────────────

class TestToolValidator:
    """Tool privilege validation tests."""

    def test_init(self):
        from core.commands.tool_validator import ToolValidator
        validator = ToolValidator()
        assert validator._stats["total_checks"] == 0

    def test_validate_no_privilege_required(self):
        from core.commands.tool_validator import ToolValidator, ValidationResult
        from unittest.mock import MagicMock
        validator = ToolValidator()
        template = MagicMock()
        template.name = "nmap"
        template.requires_privilege = "none"
        result = validator.validate(template, privilege_state="user")
        assert result.valid is True
        assert result.privilege_ok is True

    def test_validate_root_required_blocked(self):
        from core.commands.tool_validator import ToolValidator
        from unittest.mock import MagicMock
        validator = ToolValidator()
        template = MagicMock()
        template.name = "tcpdump"
        template.requires_privilege = "root"
        result = validator.validate(template, privilege_state="user")
        assert result.valid is False
        assert result.privilege_ok is False
        assert "root" in result.privilege_reason
        assert len(result.teaching_point) > 0

    def test_validate_root_required_allowed(self):
        from core.commands.tool_validator import ToolValidator
        from unittest.mock import MagicMock
        validator = ToolValidator()
        template = MagicMock()
        template.name = "tcpdump"
        template.requires_privilege = "root"
        result = validator.validate(template, privilege_state="root")
        assert result.valid is True
        assert result.privilege_ok is True

    def test_validate_sudo_required(self):
        from core.commands.tool_validator import ToolValidator
        from unittest.mock import MagicMock
        validator = ToolValidator()
        template = MagicMock()
        template.name = "apt-get"
        template.requires_privilege = "sudo"
        # Should fail for regular user
        result = validator.validate(template, privilege_state="user")
        assert result.valid is False
        # Should pass for sudo
        result2 = validator.validate(template, privilege_state="sudo")
        assert result2.valid is True
        # Should pass for root
        result3 = validator.validate(template, privilege_state="root")
        assert result3.valid is True

    def test_stats_tracking(self):
        from core.commands.tool_validator import ToolValidator
        from unittest.mock import MagicMock
        validator = ToolValidator()
        template = MagicMock()
        template.name = "nmap"
        template.requires_privilege = "none"
        validator.validate(template, privilege_state="user")
        stats = validator.get_stats()
        assert stats["total_checks"] == 1
        assert stats["passed"] == 1

    def test_validation_result_to_dict(self):
        from core.commands.tool_validator import ValidationResult
        result = ValidationResult(
            valid=False,
            tool_name="tcpdump",
            requires_privilege="root",
            current_privilege="user",
            privilege_ok=False,
            teaching_point="Need root access",
        )
        d = result.to_dict()
        assert d["valid"] is False
        assert d["tool_name"] == "tcpdump"
        assert d["teaching_point"] == "Need root access"


# ─── LearningSignalExporter ─────────────────────────────────────────────────

class TestLearningSignalExporter:
    """Learning signal JSONL export tests."""

    def test_init_disabled(self):
        from core.telemetry.learning_signal_exporter import LearningSignalExporter
        exporter = LearningSignalExporter(enabled=False)
        assert exporter._enabled is False
        stats = exporter.get_stats()
        assert stats["enabled"] is False

    def test_write_and_read(self):
        from core.telemetry.learning_signal_exporter import LearningSignalExporter
        from core.telemetry.unified_trace import UnifiedStepTrace, PhaseState
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = LearningSignalExporter(
                run_id="test_run",
                output_dir=tmpdir,
                enabled=True,
            )
            exporter.start_episode(episode_id=1)

            trace = UnifiedStepTrace(
                episode_id=1,
                step=0,
                agent_name="RedAgent",
                command="nmap -sV 10.0.0.1",
                source="ppo",
                reward_total=5.0,
                phase_state=PhaseState(current_phase="RECON"),
            )
            exporter.record_step(trace)
            exporter.end_episode({"total_reward": 5.0})
            exporter.close()

            stats = exporter.get_stats()
            assert stats["total_steps"] == 1
            assert stats["total_records"] >= 3  # start + step + end

            # Verify JSONL file
            filepath = os.path.join(tmpdir, "learning_signals_test_run.jsonl")
            assert os.path.exists(filepath)
            with open(filepath) as f:
                lines = f.readlines()
            assert len(lines) >= 3
            # Parse each line as JSON
            for line in lines:
                record = json.loads(line.strip())
                assert "type" in record
                assert "ts" in record


# ─── StepEvent Phase 11.0 Fields ─────────────────────────────────────────────

class TestStepEventPhase110:
    """StepEvent Phase 11.0 field additions."""

    def test_new_fields_exist(self):
        from core.telemetry.events import StepEvent
        ev = StepEvent()
        assert hasattr(ev, "parse_mode")
        assert hasattr(ev, "parse_latency_ms")
        assert hasattr(ev, "parse_stage_reached")
        assert hasattr(ev, "teaching_points")
        assert hasattr(ev, "budget_pressure")
        assert hasattr(ev, "mentor_budget_remaining")
        assert hasattr(ev, "phase_steps_in_current")
        assert hasattr(ev, "phase_ladder_blocked")
        assert hasattr(ev, "tool_privilege_state")
        assert hasattr(ev, "tool_validation_passed")

    def test_to_dict_includes_new_fields(self):
        from core.telemetry.events import StepEvent
        ev = StepEvent(
            parse_mode="intelligent_fullparse",
            parse_latency_ms=5.2,
            parse_stage_reached=3,
            teaching_points=["RECON: step 1/3 minimum"],
            budget_pressure=0.45,
            mentor_budget_remaining=20,
            phase_steps_in_current=2,
            phase_ladder_blocked=True,
            tool_privilege_state="user",
            tool_validation_passed=False,
        )
        d = ev.to_dict()
        assert d["parse_mode"] == "intelligent_fullparse"
        assert d["parse_latency_ms"] == 5.2
        assert d["parse_stage_reached"] == 3
        assert d["teaching_points"] == ["RECON: step 1/3 minimum"]
        assert d["budget_pressure"] == 0.45
        assert d["mentor_budget_remaining"] == 20
        assert d["phase_steps_in_current"] == 2
        assert d["phase_ladder_blocked"] is True
        assert d["tool_privilege_state"] == "user"
        assert d["tool_validation_passed"] is False

    def test_episode_event_new_fields(self):
        from core.telemetry.events import EpisodeEvent
        ev = EpisodeEvent(
            avg_budget_pressure=0.3,
            total_teaching_points=5,
            total_phase_ladder_blocks=2,
            parse_mode_used="fast",
            avg_parse_latency_ms=1.5,
        )
        d = ev.to_dict()
        assert d["avg_budget_pressure"] == 0.3
        assert d["total_teaching_points"] == 5
        assert d["total_phase_ladder_blocks"] == 2
        assert d["parse_mode_used"] == "fast"
        assert d["avg_parse_latency_ms"] == 1.5


# ─── Phase Ladder ────────────────────────────────────────────────────────────

class TestPhaseLadder:
    """Phase ladder gate in SmartCoach."""

    def test_ladder_min_steps_defined(self):
        from core.training.smart_coach import SmartCoach
        assert hasattr(SmartCoach, "PHASE_LADDER_MIN_STEPS")
        ladder = SmartCoach.PHASE_LADDER_MIN_STEPS
        assert "RECON" in ladder
        assert "EXPLOITATION" in ladder
        assert ladder["RECON"] >= 1

    def test_ladder_all_phases_covered(self):
        from core.training.smart_coach import SmartCoach
        expected_phases = [
            "RECON", "ENUMERATION", "EXPLOITATION",
            "PRIVILEGE_ESCALATION", "LATERAL_MOVEMENT",
            "POST_EXPLOITATION", "EXFILTRATION", "CLOSEOUT",
        ]
        for phase in expected_phases:
            assert phase in SmartCoach.PHASE_LADDER_MIN_STEPS, f"Missing phase: {phase}"
