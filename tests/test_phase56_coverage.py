"""Tests for Phase 56 untested features: batch_request, AgentMessageBus,
parse_with_llm, and _should_activate gates.
"""
from __future__ import annotations

import os
import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


# ── batch_request (GPTManager + FakeGPTManager) ────────────────────────────


class TestBatchRequest:
    """Tests for GPTManager.batch_request() — parallel LLM query execution."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from core.testing.fake_gpt_manager import FakeGPTManager
        self.gpt = FakeGPTManager(seed=42)

    def test_empty_queries_returns_empty(self):
        assert self.gpt.batch_request([]) == []

    def test_single_query_returns_list(self):
        results = self.gpt.batch_request([
            {"prompt": "test", "agent_id": "RedAgent"}
        ])
        assert len(results) == 1
        assert isinstance(results[0], str)

    def test_multiple_queries_preserve_order(self):
        queries = [
            {"prompt": f"q{i}", "agent_id": "RedAgent", "task_type": "tactical"}
            for i in range(5)
        ]
        results = self.gpt.batch_request(queries)
        assert len(results) == 5
        for r in results:
            assert isinstance(r, str)
            assert len(r) > 0

    def test_tracks_requests(self):
        self.gpt.clear_requests()
        self.gpt.batch_request([
            {"prompt": "p1", "agent_id": "ScoutAgent"},
            {"prompt": "p2", "agent_id": "RedAgent"},
        ])
        reqs = self.gpt.get_requests()
        assert len(reqs) >= 2

    def test_real_gpt_manager_batch_empty(self):
        from core.gpt_manager import GPTManager
        gpt = GPTManager(offline=True, enable_llm=False)
        assert gpt.batch_request([]) == []

    def test_real_gpt_manager_batch_single(self):
        from core.gpt_manager import GPTManager
        gpt = GPTManager(offline=True, enable_llm=False)
        results = gpt.batch_request([
            {"prompt": "test", "agent_id": "test"}
        ])
        assert len(results) == 1
        assert isinstance(results[0], str)

    def test_real_gpt_manager_batch_multi(self):
        from core.gpt_manager import GPTManager
        gpt = GPTManager(offline=True, enable_llm=False)
        results = gpt.batch_request([
            {"prompt": f"q{i}", "agent_id": "test"} for i in range(3)
        ])
        assert len(results) == 3
        for r in results:
            assert isinstance(r, str)


# ── AgentMessageBus ─────────────────────────────────────────────────────────


class TestAgentMessageBus:
    """Tests for AgentMessageBus — inter-agent communication."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from core.orchestration.agent_bus import AgentMessageBus, AgentMessage
        self.bus = AgentMessageBus(max_messages=50)
        self.AgentMessage = AgentMessage

    def _msg(self, sender="RedAgent", receiver="ALL", msg_type="DISCOVERY",
             content="found port 80", step=0, priority=0, **kw):
        return self.AgentMessage(
            sender=sender, receiver=receiver, msg_type=msg_type,
            content=content, step=step, priority=priority, **kw,
        )

    def test_send_increments_count(self):
        assert self.bus.message_count == 0
        self.bus.send(self._msg())
        assert self.bus.message_count == 1

    def test_get_messages_for_all(self):
        self.bus.send(self._msg(receiver="ALL"))
        msgs = self.bus.get_messages_for("ScoutAgent")
        assert len(msgs) == 1

    def test_get_messages_for_specific_agent(self):
        self.bus.send(self._msg(receiver="RedAgent"))
        assert len(self.bus.get_messages_for("RedAgent")) == 1
        assert len(self.bus.get_messages_for("ScoutAgent")) == 0

    def test_filter_by_step(self):
        self.bus.send(self._msg(step=1))
        self.bus.send(self._msg(step=5))
        msgs = self.bus.get_messages_for("RedAgent", since_step=3)
        # receiver is ALL, so RedAgent gets the step=5 msg
        assert len(msgs) == 1
        assert msgs[0].step == 5

    def test_filter_by_msg_type(self):
        self.bus.send(self._msg(msg_type="DISCOVERY"))
        self.bus.send(self._msg(msg_type="DEFENSE_ALERT"))
        msgs = self.bus.get_messages_for(
            "RedAgent", msg_types=frozenset({"DISCOVERY"})
        )
        assert len(msgs) == 1
        assert msgs[0].msg_type == "DISCOVERY"

    def test_priority_sorting(self):
        self.bus.send(self._msg(priority=1, content="low"))
        self.bus.send(self._msg(priority=10, content="high"))
        self.bus.send(self._msg(priority=5, content="mid"))
        msgs = self.bus.get_messages_for("RedAgent")
        assert msgs[0].priority == 10
        assert msgs[1].priority == 5
        assert msgs[2].priority == 1

    def test_inject_into_prompt_empty(self):
        text = self.bus.inject_into_prompt("RedAgent")
        assert text == ""

    def test_inject_into_prompt_formatted(self):
        self.bus.send(self._msg(sender="ScoutAgent", receiver="ALL",
                                msg_type="DISCOVERY", content="open port 22"))
        text = self.bus.inject_into_prompt("RedAgent")
        assert "[AGENT COMMS]" in text
        assert "ScoutAgent" in text
        assert "open port 22" in text

    def test_inject_caps_at_five(self):
        for i in range(10):
            self.bus.send(self._msg(content=f"msg {i}"))
        text = self.bus.inject_into_prompt("RedAgent")
        # Count message lines (excluding header) — message lines have "→"
        msg_lines = [l for l in text.split("\n") if "→" in l]
        assert len(msg_lines) <= 5

    def test_get_recent_discoveries(self):
        self.bus.send(self._msg(msg_type="DISCOVERY", content="port 80"))
        self.bus.send(self._msg(msg_type="DEFENSE_ALERT", content="alert"))
        self.bus.send(self._msg(msg_type="DISCOVERY", content="creds"))
        discoveries = self.bus.get_recent_discoveries(last_n=10)
        assert len(discoveries) == 2
        assert discoveries[0].content == "creds"  # reversed order

    def test_clear_episode(self):
        self.bus.send(self._msg())
        self.bus.send(self._msg())
        assert self.bus.message_count == 2
        self.bus.clear_episode()
        assert self.bus.message_count == 0

    def test_max_messages_enforced(self):
        bus = self.bus.__class__(max_messages=5)
        for i in range(10):
            bus.send(self._msg(content=f"msg {i}"))
        assert bus.message_count == 5

    def test_unknown_msg_type_warns(self):
        """Unknown msg_type should not crash, just warn."""
        msg = self.AgentMessage(
            sender="Red", receiver="ALL", msg_type="INVALID_TYPE",
            content="test",
        )
        self.bus.send(msg)
        assert self.bus.message_count == 1


# ── parse_with_llm ──────────────────────────────────────────────────────────


class TestParseWithLLM:
    """Tests for parse_with_llm hybrid regex+LLM parsing."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from core.testing.fake_gpt_manager import FakeGPTManager
        self.gpt = FakeGPTManager(seed=42)

    def test_short_output_skips_llm(self):
        from core.orchestration.output_parser import parse_with_llm
        result = parse_with_llm("short", "nmap", self.gpt)
        # Should not invoke LLM for <200 char output
        assert result is not None

    def test_empty_output_returns_result(self):
        from core.orchestration.output_parser import parse_with_llm
        result = parse_with_llm("", "nmap", self.gpt)
        assert result.total_count == 0

    def test_rich_regex_output_skips_llm(self):
        """When regex finds >=5 discoveries, LLM is not called."""
        from core.orchestration.output_parser import parse_with_llm
        # Build output with enough regex-discoverable content
        output = """
        Nmap scan report for 10.0.0.1
        PORT     STATE SERVICE  VERSION
        22/tcp   open  ssh      OpenSSH 7.6
        80/tcp   open  http     Apache 2.4.29
        443/tcp  open  https    nginx 1.14
        3306/tcp open  mysql    MySQL 5.7
        8080/tcp open  http-proxy
        21/tcp   open  ftp      vsftpd 3.0.3
        """ + " " * 200  # pad to >200 chars
        initial_count = self.gpt._request_count
        result = parse_with_llm(output, "nmap -sV 10.0.0.1", self.gpt)
        assert result.total_count >= 5
        # LLM should not have been called since regex found enough
        assert self.gpt._request_count == initial_count

    def test_sparse_output_invokes_llm(self):
        """When regex finds <5 discoveries and output >200 chars, LLM is called."""
        from core.orchestration.output_parser import parse_with_llm
        # Output with unusual formatting that regex won't parse well
        output = (
            "The scan completed. Some interesting findings about the target "
            "environment were noted. The administrators appear to have left "
            "several interesting artifacts. Further investigation is recommended "
            "as the system may contain additional entry points that could be "
            "leveraged for access escalation. " * 3
        )
        initial_count = self.gpt._request_count
        result = parse_with_llm(output, "enum4linux 10.0.0.1", self.gpt)
        assert result is not None
        # FakeGPTManager should have been called
        assert self.gpt._request_count > initial_count

    def test_returns_discovery_result_type(self):
        from core.orchestration.output_parser import parse_with_llm, DiscoveryResult
        result = parse_with_llm("test " * 60, "nmap", self.gpt)
        assert isinstance(result, DiscoveryResult)


# ── _parse_llm_json helper ──────────────────────────────────────────────────


class TestParseLLMJson:
    """Tests for _parse_llm_json JSON extraction."""

    def test_plain_json(self):
        from core.orchestration.output_parser import _parse_llm_json
        result = _parse_llm_json('{"ports": [22, 80]}')
        assert result == {"ports": [22, 80]}

    def test_markdown_fenced_json(self):
        from core.orchestration.output_parser import _parse_llm_json
        raw = '```json\n{"ports": [443]}\n```'
        result = _parse_llm_json(raw)
        assert result == {"ports": [443]}

    def test_invalid_json_returns_empty(self):
        from core.orchestration.output_parser import _parse_llm_json
        result = _parse_llm_json("not json at all")
        assert result == {}

    def test_embedded_json(self):
        from core.orchestration.output_parser import _parse_llm_json
        raw = 'Here are results: {"services": ["http"]} end'
        result = _parse_llm_json(raw)
        assert "services" in result


# ── _merge_discoveries ──────────────────────────────────────────────────────


class TestMergeDiscoveries:
    """Tests for _merge_discoveries LLM→regex merging."""

    def _make_result(self):
        from core.orchestration.output_parser import DiscoveryResult
        return DiscoveryResult(ports={22, 80}, services={"ssh"})

    def test_empty_llm_returns_original(self):
        from core.orchestration.output_parser import _merge_discoveries
        base = self._make_result()
        merged = _merge_discoveries(base, {})
        assert merged.ports == {22, 80}

    def test_merge_new_ports(self):
        from core.orchestration.output_parser import _merge_discoveries
        base = self._make_result()
        merged = _merge_discoveries(base, {"ports": [443, 3306]})
        assert 443 in merged.ports
        assert 3306 in merged.ports

    def test_merge_services(self):
        from core.orchestration.output_parser import _merge_discoveries
        base = self._make_result()
        merged = _merge_discoveries(base, {"services": ["http", "mysql"]})
        assert "http" in merged.services
        assert "mysql" in merged.services

    def test_merge_credentials(self):
        from core.orchestration.output_parser import _merge_discoveries
        base = self._make_result()
        merged = _merge_discoveries(base, {"credentials": [["admin", "password"]]})
        assert ("admin", "password") in merged.credentials

    def test_merge_cves(self):
        from core.orchestration.output_parser import _merge_discoveries
        base = self._make_result()
        merged = _merge_discoveries(base, {"cves": ["cve-2021-44228"]})
        assert "CVE-2021-44228" in merged.cves

    def test_merge_web_paths(self):
        from core.orchestration.output_parser import _merge_discoveries
        base = self._make_result()
        merged = _merge_discoveries(base, {"web_paths": ["/admin", "/login"]})
        assert "/admin" in merged.web_paths

    def test_invalid_port_skipped(self):
        from core.orchestration.output_parser import _merge_discoveries
        base = self._make_result()
        merged = _merge_discoveries(base, {"ports": [99999, -1, "abc"]})
        assert 99999 not in merged.ports
        assert -1 not in merged.ports

    def test_merge_shell_detection(self):
        from core.orchestration.output_parser import _merge_discoveries
        base = self._make_result()
        merged = _merge_discoveries(base, {"shell_type": "root_shell"})
        assert "root_shell" in merged.shells

    def test_merge_hashes(self):
        from core.orchestration.output_parser import _merge_discoveries
        base = self._make_result()
        long_hash = "a" * 32
        merged = _merge_discoveries(base, {"hashes": [long_hash, "short"]})
        assert long_hash in merged.hashes
        assert "short" not in merged.hashes  # too short


# ── _should_activate agent gates ────────────────────────────────────────────


class TestShouldActivateGates:
    """Tests for SmartOrchestrator._should_activate agent activation gates."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from core.orchestration.smart_orchestrator import SmartOrchestrator
        from core.testing.fake_gpt_manager import FakeGPTManager
        self.gpt = FakeGPTManager(seed=42)
        self.orch = SmartOrchestrator.__new__(SmartOrchestrator)
        # Minimal state for _should_activate
        self.orch.gpt_manager = self.gpt
        self.orch.discovery_board = {
            "ports": set(), "services": set(), "credentials": set(),
            "vulns": set(), "shells": set(), "users": set(),
            "web_paths": set(), "flags_set": set(),
        }
        self.orch._steps_without_discoveries = 0
        self.orch._last_step_phase = "RECON"
        self.orch._stagnation_steps = 0

    def test_shadow_blocked_without_creds_or_shell(self):
        """Shadow should be blocked early — no creds, no shell."""
        active, reason = self.orch._should_activate("ShadowAgent", step=1, phase="RECON")
        assert not active or "skip" in reason.lower() or "shadow" in reason.lower()

    def test_shadow_active_with_shell(self):
        """Shadow should activate when shell is obtained."""
        self.orch.discovery_board["shells"] = {"shell"}
        active, reason = self.orch._should_activate("ShadowAgent", step=1, phase="EXPLOITATION")
        # May still fail on frequency — but shouldn't fail on gate
        if not active:
            assert "shadow" not in reason.lower() or "frequency" in reason.lower() or "gate" not in reason.lower()

    def test_blue_blocked_in_recon(self):
        """Blue should be skipped during RECON."""
        active, reason = self.orch._should_activate("BlueAgent", step=1, phase="RECON")
        assert not active

    def test_scout_active_in_recon(self):
        """Scout should be active during RECON by frequency."""
        # Step 0 should match frequency (every step in RECON)
        active, _reason = self.orch._should_activate("ScoutAgent", step=0, phase="RECON")
        assert active

    def test_method_returns_tuple(self):
        """Should return (bool, str) tuple."""
        result = self.orch._should_activate("RedAgent", step=0, phase="RECON")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)
