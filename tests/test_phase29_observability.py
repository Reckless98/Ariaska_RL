"""Phase 29: AgentStepInfo extensions + observability completion tests."""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ.setdefault("ARIASKA_DRY_RUN", "1")

from core.observability.live_dashboard import AgentStepInfo
from core.telemetry.unified_trace import UnifiedStepTrace


class TestAgentStepInfoExtensions:
    """Validate Phase 29 fields on AgentStepInfo."""

    def test_new_fields_have_defaults(self):
        info = AgentStepInfo(
            agent_name="RedAgent",
            command="nmap -sV 10.0.0.1",
            command_output="22/tcp open ssh",
            mentor_reasoning="",
            source="ppo",
            confidence=0.8,
            reward=5.0,
            mentor_call=False,
            tokens_used=0,
        )
        assert info.evidence_gate_result == ""
        assert info.micro_chain_escalated is False
        assert info.self_debug_fix == ""
        assert info.retry_count == 0

    def test_new_fields_settable(self):
        info = AgentStepInfo(
            agent_name="RedAgent",
            command="nmap -sV 10.0.0.1",
            command_output="",
            mentor_reasoning="",
            source="micro_chain",
            confidence=0.6,
            reward=3.0,
            mentor_call=False,
            tokens_used=50,
            evidence_gate_result="pass",
            micro_chain_escalated=True,
            self_debug_fix="nmap -sC 10.0.0.1",
            retry_count=1,
        )
        assert info.evidence_gate_result == "pass"
        assert info.micro_chain_escalated is True
        assert info.self_debug_fix == "nmap -sC 10.0.0.1"
        assert info.retry_count == 1


class TestUnifiedStepTraceExtensions:
    """Validate Phase 29 fields on UnifiedStepTrace."""

    def test_new_fields_have_defaults(self):
        trace = UnifiedStepTrace()
        assert trace.micro_chain_escalated is False
        assert trace.micro_chain_cost_usd == 0.0
        assert trace.self_debug_fix == ""
        assert trace.retry_count == 0
        assert trace.watchdog_trigger == ""

    def test_to_dict_includes_new_fields(self):
        trace = UnifiedStepTrace(
            micro_chain_escalated=True,
            micro_chain_cost_usd=0.0012,
            self_debug_fix="fixed_cmd",
            retry_count=2,
            watchdog_trigger="semantic_stall",
        )
        d = trace.to_dict()
        assert d["micro_chain_escalated"] is True
        assert d["micro_chain_cost_usd"] == pytest.approx(0.0012, abs=1e-5)
        assert d["self_debug_fix"] == "fixed_cmd"
        assert d["retry_count"] == 2
        assert d["watchdog_trigger"] == "semantic_stall"

    def test_to_dict_still_has_evidence_gate(self):
        trace = UnifiedStepTrace(
            evidence_gate_result="log_reject",
            evidence_gate_reasons=["no port evidence"],
        )
        d = trace.to_dict()
        assert d["evidence_gate_result"] == "log_reject"
        assert "no port evidence" in d["evidence_gate_reasons"]
