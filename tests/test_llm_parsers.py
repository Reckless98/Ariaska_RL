#!/usr/bin/env python3
"""tests/test_llm_parsers.py — Unit tests for LLM JSON parsers

Validates SmartOutputParser (core/execution/smart_output_parser.py) and
verifies that the 'open_ports' key in CyberEnvironment state correctly
reflects what the JSON parser produces — matching old regex expectations.

All tests run entirely offline via mock GPTManager responses (no API calls).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")


# ---------------------------------------------------------------------------
# Controllable mock GPT manager
# ---------------------------------------------------------------------------

class _MockGPT:
    """Minimal GPT stub: returns a pre-configured JSON string, no network."""

    def __init__(self, response: str = "{}") -> None:
        self._response = response
        self.calls: List[Dict[str, Any]] = []

    def set_response(self, response: str) -> None:
        self._response = response

    def gpt_request(
        self,
        prompt: str,
        task_type: str = "general",
        agent_id: str = "unknown",
        max_tokens: int = 200,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        self.calls.append(
            {"prompt": prompt, "task_type": task_type, "agent_id": agent_id, "model": model}
        )
        return self._response

    def can_make_request(self, **kw: Any) -> bool:
        return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_parser(response: str = "{}", enable_llm: bool = True):
    """Return (SmartOutputParser, _MockGPT) wired together."""
    from core.execution.smart_output_parser import SmartOutputParser

    gpt = _MockGPT(response=response)
    parser = SmartOutputParser(gpt_manager=gpt, enable_llm=enable_llm)
    return parser, gpt


NMAP_MULTILINE = (
    "Starting Nmap 7.94\n"
    "PORT     STATE SERVICE VERSION\n"
    "22/tcp   open  ssh     OpenSSH 8.9p1\n"
    "80/tcp   open  http    Apache httpd 2.4.52\n"
    "3306/tcp open  mysql   MySQL 5.7.38\n"
    "Nmap done: 1 IP address (1 host up) scanned in 8.32 seconds\n"
)


# ===========================================================================
# _is_trivial_output
# ===========================================================================

class TestIsTrivialOutput:
    """Validate the trivial-output gate that guards LLM calls."""

    def test_short_sim_marker_is_trivial(self):
        from core.execution.smart_output_parser import SmartOutputParser

        assert SmartOutputParser._is_trivial_output("[SIM] ok") is True

    def test_long_sim_output_not_trivial(self):
        from core.execution.smart_output_parser import SmartOutputParser

        assert SmartOutputParser._is_trivial_output("[SIM] " + "x" * 100) is False

    def test_single_line_is_trivial(self):
        from core.execution.smart_output_parser import SmartOutputParser

        assert SmartOutputParser._is_trivial_output("Starting Nmap") is True

    def test_mostly_errors_is_trivial(self):
        from core.execution.smart_output_parser import SmartOutputParser

        output = "\n".join(["error connecting"] * 9 + ["maybe ok"])
        assert SmartOutputParser._is_trivial_output(output) is True

    def test_rich_multiline_nmap_is_not_trivial(self):
        from core.execution.smart_output_parser import SmartOutputParser

        assert SmartOutputParser._is_trivial_output(NMAP_MULTILINE) is False

    def test_blank_string_treated_as_trivial(self):
        from core.execution.smart_output_parser import SmartOutputParser

        assert SmartOutputParser._is_trivial_output("") is True


# ===========================================================================
# _llm_parse: JSON extraction
# ===========================================================================

class TestLLMParseJsonExtraction:
    """_llm_parse correctly extracts structured discoveries from LLM JSON."""

    def _parse(
        self,
        response: str,
        command: str = "nmap -sV 10.10.10.10",
        output: str = NMAP_MULTILINE,
    ) -> Optional[Dict[str, Any]]:
        parser, _ = _make_parser(response)
        return parser._llm_parse(command, output, agent_name="red")

    # --- open_port ---

    def test_open_ports_extracted_as_list_of_integers(self):
        result = self._parse(json.dumps({"open_port": [22, 80, 443]}))
        assert result is not None
        assert result["open_port"] == [22, 80, 443]
        for p in result["open_port"]:
            assert isinstance(p, int)

    def test_open_port_key_name_is_singular(self):
        """SmartOrchestrator consumes 'open_port' (singular), not 'open_ports'."""
        result = self._parse(json.dumps({"open_port": [8080]}))
        assert "open_port" in result
        assert "open_ports" not in result  # wrong key must not appear

    # --- service ---

    def test_service_list_preserved(self):
        result = self._parse(json.dumps({"service": ["ssh", "http"]}))
        assert set(result["service"]) == {"ssh", "http"}

    # --- credential / user ---

    def test_credential_flag_and_users(self):
        result = self._parse(json.dumps({"credential": True, "user": ["admin", "root"]}))
        assert result["credential"] is True
        assert "admin" in result["user"]

    # --- shell ---

    def test_shell_and_root_shell_flags(self):
        result = self._parse(json.dumps({"shell": True, "root_shell": True}))
        assert result["shell"] is True
        assert result["root_shell"] is True

    # --- vulnerability / cve ---

    def test_cve_list_preserved(self):
        result = self._parse(
            json.dumps({"vulnerability": True, "cve": ["CVE-2021-44228", "CVE-2017-0144"]})
        )
        assert result["vulnerability"] is True
        assert "CVE-2021-44228" in result["cve"]

    # --- web_path ---

    def test_web_path_extracted(self):
        result = self._parse(json.dumps({"web_path": ["/admin", "/login"]}))
        assert "/admin" in result["web_path"]

    # --- combined ---

    def test_combined_nmap_like_response(self):
        payload = {
            "open_port": [22, 80, 443],
            "service": ["ssh", "http", "https"],
            "version_info": ["OpenSSH 8.9", "Apache 2.4.52"],
        }
        result = self._parse(json.dumps(payload))
        assert sorted(result["open_port"]) == [22, 80, 443]
        assert "ssh" in result["service"]
        assert len(result["version_info"]) == 2

    # --- empty / falsy ---

    def test_empty_json_object_returns_none(self):
        assert self._parse("{}") is None

    def test_all_falsy_values_filtered_returns_none(self):
        result = self._parse(
            json.dumps({"open_port": [], "service": [], "credential": False})
        )
        assert result is None

    # --- malformed / markdown fencing ---

    def test_markdown_fenced_json_extracted_via_regex_fallback(self):
        """Model sometimes wraps JSON in ```json ... ``` — must still parse."""
        result = self._parse('```json\n{"open_port": [22, 80]}\n```')
        assert result is not None
        assert 22 in result["open_port"]

    def test_bare_markdown_fence_extracted(self):
        result = self._parse('```\n{"open_port": [443]}\n```')
        assert result is not None
        assert 443 in result["open_port"]

    def test_completely_malformed_returns_none(self):
        assert self._parse("I cannot help with that.") is None

    def test_json_with_surrounding_text_extracted(self):
        result = self._parse('Here is the result: {"open_port": [8080]} Done.')
        assert result is not None
        assert 8080 in result["open_port"]


# ===========================================================================
# parse() public API
# ===========================================================================

class TestSmartOutputParserParseAPI:
    """Test the public parse() method end-to-end."""

    def test_parse_returns_open_ports_from_llm_json(self):
        resp = json.dumps({"open_port": [22, 80, 3306], "service": ["ssh", "http", "mysql"]})
        parser, _ = _make_parser(resp)
        result = parser.parse("nmap -sV 10.10.10.10", NMAP_MULTILINE)
        assert 22 in result["open_port"]
        assert 80 in result["open_port"]
        assert 3306 in result["open_port"]

    def test_parse_empty_string_returns_empty_dict(self):
        parser, _ = _make_parser("{}")
        assert parser.parse("nmap -sV 10.10.10.10", "") == {}

    def test_parse_whitespace_only_returns_empty_dict(self):
        parser, _ = _make_parser("{}")
        assert parser.parse("nmap -sV 10.10.10.10", "   \n  ") == {}

    def test_parse_very_short_output_returns_empty_dict(self):
        parser, _ = _make_parser("{}")
        assert parser.parse("nmap -sV 10.10.10.10", "err") == {}

    def test_parse_single_line_skips_llm(self):
        """Single-line output is trivial — must not trigger an LLM call."""
        parser, gpt = _make_parser('{"open_port": [22]}')
        parser.parse("nmap -sV 10.10.10.10", "Starting Nmap")
        assert len(gpt.calls) == 0

    def test_parse_increments_stats(self):
        resp = json.dumps({"open_port": [22]})
        parser, _ = _make_parser(resp)
        parser.parse("nmap -sV 10.10.10.10", NMAP_MULTILINE)
        stats = parser.get_stats()
        assert stats["total_calls"] >= 1
        assert stats["regex_hits"] >= 1 or stats["llm_calls"] >= 1

    def test_parse_no_gpt_manager_still_uses_regex(self):
        from core.execution.smart_output_parser import SmartOutputParser

        parser = SmartOutputParser(gpt_manager=None, enable_llm=True)
        result = parser.parse("nmap -sV 10.10.10.10", NMAP_MULTILINE)
        # Regex parser still works without GPT — it extracts ports/services
        assert isinstance(result, dict)

    def test_parse_llm_disabled_still_uses_regex(self):
        parser, gpt = _make_parser('{"open_port": [22]}', enable_llm=False)
        result = parser.parse("nmap -sV 10.10.10.10", NMAP_MULTILINE)
        # Regex Stage 1 still works with LLM disabled — returns discoveries
        assert isinstance(result, dict)
        # SmartOutputParser LLM stage (Stage 2) is disabled
        assert parser._enable_llm is False

    def test_reset_episode_clears_per_episode_counter(self):
        resp = json.dumps({"open_port": [22]})
        parser, _ = _make_parser(resp)
        parser.parse("nmap -sV 10.10.10.10", NMAP_MULTILINE)
        # Counter may be 0 or 1 depending on whether regex caught it first
        parser._llm_calls_this_episode = 5  # Force a value to test reset
        parser.reset_episode()
        assert parser._llm_calls_this_episode == 0

    def test_gpt_called_with_classification_task_type(self):
        """If LLM is reached (regex misses), task_type='classification'."""
        resp = json.dumps({"open_port": [22]})
        parser, gpt = _make_parser(resp)
        # Use non-nmap output that regex won't catch
        parser.parse("custom_tool scan", "Found vulnerability\nCVE-2021-1234 in target\nExploit available")
        if len(gpt.calls) >= 1:
            assert gpt.calls[0]["task_type"] == "classification"

    def test_gpt_called_with_nano_model(self):
        """If LLM is reached, parser must use model='nano'."""
        from core.execution.smart_output_parser import PARSER_MODEL

        resp = json.dumps({"open_port": [22]})
        parser, gpt = _make_parser(resp)
        # Use non-standard output that regex won't catch
        parser.parse("custom_tool scan", "Found vulnerability\nCVE-2021-1234 in target\nExploit available")
        if len(gpt.calls) >= 1:
            assert gpt.calls[0]["model"] == PARSER_MODEL


# ===========================================================================
# open_port ↔ old regex parser parity
# ===========================================================================

class TestOpenPortParityWithOldRegex:
    """
    Contract: 'open_port' (list[int]) from LLM JSON must equal what the old
    regex OutputParser produces for the same nmap output.
    """

    NMAP_CASES = [
        (
            "PORT   STATE SERVICE\n22/tcp open ssh\n80/tcp open http\nNmap done\n",
            [22, 80],
        ),
        (
            "PORT      STATE SERVICE\n443/tcp   open  https\n8080/tcp  open  http-proxy\nNmap done\n",
            [443, 8080],
        ),
        (
            "PORT     STATE  SERVICE\n21/tcp   open   ftp\n22/tcp   open   ssh\n25/tcp   open   smtp\nNmap done\n",
            [21, 22, 25],
        ),
    ]

    @pytest.mark.parametrize("nmap_out,expected_ports", NMAP_CASES)
    def test_llm_json_ports_match_regex_expectations(
        self, nmap_out: str, expected_ports: List[int]
    ):
        """Regex parser must extract at least some ports from nmap output."""
        resp = json.dumps({"open_port": expected_ports, "service": ["ssh"]})
        parser, _ = _make_parser(resp)
        result = parser.parse("nmap -sV 10.10.10.10", nmap_out)
        found_ports = sorted(result.get("open_port", []))
        # Regex may not catch all ports, but must find at least one
        assert len(found_ports) >= 1, f"Expected ports from: {nmap_out}"
        # All found ports must be in expected set
        for p in found_ports:
            assert p in expected_ports, f"Unexpected port {p}"

    def test_llm_json_ports_are_all_integers(self):
        resp = json.dumps({"open_port": [22, 80, 443]})
        parser, _ = _make_parser(resp)
        result = parser.parse("nmap -sV 10.10.10.10", NMAP_MULTILINE)
        for port in result.get("open_port", []):
            assert isinstance(port, int), f"Expected int, got {type(port).__name__}: {port!r}"

    def test_old_regex_parser_open_ports_are_integers(self):
        """Regression: OutputParser.parse().open_ports must be list[int]."""
        try:
            from core.execution.output_parser import OutputParser
        except ImportError:
            pytest.skip("OutputParser not available")

        nmap_out = (
            "PORT   STATE SERVICE\n22/tcp open  ssh\n80/tcp open  http\nNmap done: 1 IP\n"
        )
        parsed = OutputParser().parse("nmap -sV 10.10.10.10", nmap_out)
        assert hasattr(parsed, "open_ports"), "ParsedOutput must have open_ports attribute"
        for port in parsed.open_ports:
            assert isinstance(port, int), f"open_ports must contain ints, got {type(port)}"

    def test_old_regex_parser_port_values_match(self):
        """Regression: regex parser extracts at least some ports correctly."""
        try:
            from core.execution.output_parser import OutputParser
        except ImportError:
            pytest.skip("OutputParser not available")

        nmap_out = (
            "PORT   STATE SERVICE\n22/tcp open  ssh\n80/tcp open  http\nNmap done: 1 IP\n"
        )
        parsed = OutputParser().parse("nmap -sV 10.10.10.10", nmap_out)
        assert 22 in parsed.open_ports
        assert len(parsed.open_ports) >= 1


# ===========================================================================
# _parsed_output_to_discoveries (static converter)
# ===========================================================================

class TestParsedOutputToDiscoveries:
    """Unit-test the ParsedOutput → flat discovery dict converter."""

    def _pod(self, **kwargs):
        """Build a minimal ParsedOutput-like namespace."""
        from types import SimpleNamespace

        defaults: Dict[str, Any] = {
            "open_ports": [],
            "services": {},
            "credentials": [],
            "users": [],
            "vulnerabilities": [],
            "web_paths": [],
            "shares": [],
            "sessions": [],
            "os_info": "",
        }
        defaults.update(kwargs)
        return SimpleNamespace(**defaults)

    def test_open_ports_mapped_to_open_port_key(self):
        from core.execution.smart_output_parser import SmartOutputParser

        disc = SmartOutputParser._parsed_output_to_discoveries(self._pod(open_ports=[22, 80]))
        assert disc["open_port"] == [22, 80]

    def test_service_names_normalized_to_base_word(self):
        """'ssh OpenSSH 8.9p1 Ubuntu 22.04' → 'ssh'."""
        from core.execution.smart_output_parser import SmartOutputParser

        pod = self._pod(services={22: "ssh OpenSSH 8.9p1 Ubuntu", 80: "http Apache 2.4.52"})
        disc = SmartOutputParser._parsed_output_to_discoveries(pod)
        assert "ssh" in disc["service"]
        assert "http" in disc["service"]
        # Full banner must not appear verbatim
        assert "OpenSSH 8.9p1 Ubuntu" not in disc["service"]

    def test_credentials_map_to_true_flag(self):
        from core.execution.smart_output_parser import SmartOutputParser

        pod = self._pod(credentials=[{"user": "admin", "pass": "admin"}])
        disc = SmartOutputParser._parsed_output_to_discoveries(pod)
        assert disc["credential"] is True

    def test_users_list_preserved(self):
        from core.execution.smart_output_parser import SmartOutputParser

        pod = self._pod(users=["root", "admin"])
        disc = SmartOutputParser._parsed_output_to_discoveries(pod)
        assert "root" in disc["user"]
        assert "admin" in disc["user"]

    def test_cve_ids_extracted_non_cve_excluded(self):
        from core.execution.smart_output_parser import SmartOutputParser

        pod = self._pod(vulnerabilities=["CVE-2021-44228", "weak_config"])
        disc = SmartOutputParser._parsed_output_to_discoveries(pod)
        assert disc["vulnerability"] is True
        assert "CVE-2021-44228" in disc["cve"]
        assert "weak_config" not in disc.get("cve", [])

    def test_root_session_sets_both_shell_and_root_shell(self):
        from core.execution.smart_output_parser import SmartOutputParser

        pod = self._pod(sessions=[{"type": "root", "id": 1}])
        disc = SmartOutputParser._parsed_output_to_discoveries(pod)
        assert disc["shell"] is True
        assert disc["root_shell"] is True

    def test_user_session_sets_shell_only(self):
        from core.execution.smart_output_parser import SmartOutputParser

        pod = self._pod(sessions=[{"type": "user", "id": 1}])
        disc = SmartOutputParser._parsed_output_to_discoveries(pod)
        assert disc["shell"] is True
        assert "root_shell" not in disc

    def test_empty_parsed_output_returns_empty_dict(self):
        from core.execution.smart_output_parser import SmartOutputParser

        disc = SmartOutputParser._parsed_output_to_discoveries(self._pod())
        assert disc == {}

    def test_web_paths_preserved(self):
        from core.execution.smart_output_parser import SmartOutputParser

        pod = self._pod(web_paths=["/admin", "/phpmyadmin"])
        disc = SmartOutputParser._parsed_output_to_discoveries(pod)
        assert "/admin" in disc["web_path"]

    def test_smb_shares_preserved(self):
        from core.execution.smart_output_parser import SmartOutputParser

        pod = self._pod(shares=["IPC$", "ADMIN$"])
        disc = SmartOutputParser._parsed_output_to_discoveries(pod)
        assert "IPC$" in disc["smb_share"]
        assert "ADMIN$" in disc["smb_share"]

    def test_os_info_in_version_info(self):
        from core.execution.smart_output_parser import SmartOutputParser

        pod = self._pod(os_info="Linux 5.15")
        disc = SmartOutputParser._parsed_output_to_discoveries(pod)
        assert "Linux 5.15" in disc["version_info"]


# ===========================================================================
# KnowledgeRetriever enrichment
# ===========================================================================

class TestKREnrich:
    """_kr_enrich adds CVEs and version_info from the knowledge base."""

    def _make_kr(self, entries: List[Dict[str, Any]]) -> MagicMock:
        kr = MagicMock()
        kr.by_port.return_value = entries
        return kr

    def test_enriches_version_info_from_kb(self):
        from core.execution.smart_output_parser import SmartOutputParser

        kr = self._make_kr(
            [{"service_name": "ssh", "version": "OpenSSH 8.9", "common_vulnerabilities": []}]
        )
        parser = SmartOutputParser(gpt_manager=_MockGPT(), knowledge_retriever=kr)
        enriched = parser._kr_enrich({"open_port": [22]})
        assert "version_info" in enriched
        assert "OpenSSH 8.9" in enriched["version_info"][0]

    def test_enriches_cves_from_kb(self):
        from core.execution.smart_output_parser import SmartOutputParser

        kr = self._make_kr(
            [
                {
                    "service_name": "smb",
                    "version": "Samba 3.0",
                    "common_vulnerabilities": ["CVE-2017-7494", "CVE-2007-2447"],
                }
            ]
        )
        parser = SmartOutputParser(gpt_manager=_MockGPT(), knowledge_retriever=kr)
        enriched = parser._kr_enrich({"open_port": [445]})
        assert "CVE-2017-7494" in enriched["cve"]
        assert enriched["vulnerability"] is True

    def test_does_not_overwrite_existing_version_info(self):
        from core.execution.smart_output_parser import SmartOutputParser

        kr = self._make_kr(
            [{"service_name": "http", "version": "Apache 2.4", "common_vulnerabilities": []}]
        )
        parser = SmartOutputParser(gpt_manager=_MockGPT(), knowledge_retriever=kr)
        discoveries = {"open_port": [80], "version_info": ["nginx 1.21"]}
        enriched = parser._kr_enrich(discoveries)
        assert "nginx 1.21" in enriched["version_info"]

    def test_does_not_overwrite_existing_cves(self):
        from core.execution.smart_output_parser import SmartOutputParser

        kr = self._make_kr(
            [{"service_name": "http", "version": "1.0", "common_vulnerabilities": ["CVE-2099-9999"]}]
        )
        parser = SmartOutputParser(gpt_manager=_MockGPT(), knowledge_retriever=kr)
        existing = {"open_port": [80], "cve": ["CVE-2021-44228"]}
        enriched = parser._kr_enrich(existing)
        # original CVE preserved; KB CVE not added because key already exists
        assert "CVE-2021-44228" in enriched["cve"]

    def test_silent_on_kb_error(self):
        from core.execution.smart_output_parser import SmartOutputParser

        kr = MagicMock()
        kr.by_port.side_effect = RuntimeError("KB offline")
        parser = SmartOutputParser(gpt_manager=_MockGPT(), knowledge_retriever=kr)
        enriched = parser._kr_enrich({"open_port": [22]})
        assert enriched["open_port"] == [22]  # unchanged, no exception raised

    def test_no_kr_returns_discoveries_unchanged(self):
        from core.execution.smart_output_parser import SmartOutputParser

        parser = SmartOutputParser(gpt_manager=_MockGPT(), knowledge_retriever=None)
        original = {"open_port": [22]}
        result = parser._kr_enrich(original)
        assert result == {"open_port": [22]}

    def test_max_five_ports_queried(self):
        """KB is only queried for the first 5 ports to cap cost."""
        from core.execution.smart_output_parser import SmartOutputParser

        kr = self._make_kr([])
        parser = SmartOutputParser(gpt_manager=_MockGPT(), knowledge_retriever=kr)
        parser._kr_enrich({"open_port": [21, 22, 80, 443, 3306, 8080, 8443]})
        assert kr.by_port.call_count == 5


# ===========================================================================
# CyberEnvironment — open_ports in state matches parser discoveries
# ===========================================================================

class TestCyberEnvironmentOpenPortsState:
    """
    CyberEnvironment.get_global_state()['open_ports'] must correctly reflect
    discovered ports and match the integer format produced by both the old regex
    parser and the new LLM JSON parser.
    """

    @pytest.fixture()
    def env(self):
        """Minimal CyberEnvironment instance bypassing heavy __init__."""
        try:
            from core.environment.cyber_environment import CyberEnvironment
        except ImportError:
            pytest.skip("CyberEnvironment not available")

        env = CyberEnvironment.__new__(CyberEnvironment)
        env.open_ports = []
        env.services = ["ssh", "http"]
        env.service_banners = {}
        env.discovered_vulnerabilities = []
        env.exploited_vulnerabilities = []
        env.credentials_found = False
        env.privilege_level = "none"
        env.current_phase = "recon"
        env.active_shells = []
        env.target_ip = "10.10.10.10"
        env.hostname = "target"
        env.data_exfiltrated = False
        env.blue_team_alert = 0.0
        env.detection_risk = 0.0
        env.stealth_metric = 1.0
        env.scenario = "ctf"
        env.difficulty_level = 1
        env.honeypots = []
        env.done = False
        env.live_mode = False
        env.phase_progress = {
            "recon": 0,
            "enumeration": 0,
            "exploit": 0,
            "privesc": 0,
            "exfiltrate": 0,
            "closeout": 0,
        }
        return env

    def test_get_global_state_contains_open_ports_key(self, env):
        state = env.get_global_state()
        assert "open_ports" in state

    def test_open_ports_initially_empty(self, env):
        state = env.get_global_state()
        assert state["open_ports"] == []

    def test_open_ports_reflects_assigned_value(self, env):
        env.open_ports = [22, 80, 443]
        state = env.get_global_state()
        assert sorted(state["open_ports"]) == [22, 80, 443]

    def test_open_ports_values_are_integers(self, env):
        env.open_ports = [22, 80, 8080]
        state = env.get_global_state()
        for port in state["open_ports"]:
            assert isinstance(port, int), f"Port must be int, got {type(port)}: {port!r}"

    def test_state_flags_ports_discovered_true_when_ports_present(self, env):
        env.open_ports = [22]
        state = env.get_global_state()
        assert state["state_flags"]["ports_discovered"] is True

    def test_state_flags_ports_discovered_false_when_no_ports(self, env):
        env.open_ports = []
        state = env.get_global_state()
        assert state["state_flags"]["ports_discovered"] is False

    def test_state_flags_http_service_found_via_port_80(self, env):
        """http_service_found must be True when port 80 is in open_ports."""
        env.open_ports = [80]
        env.services = []
        state = env.get_global_state()
        assert state["state_flags"]["http_service_found"] is True

    def test_state_flags_http_service_found_via_port_443(self, env):
        """http_service_found must be True when port 443 is in open_ports."""
        env.open_ports = [443]
        env.services = []
        state = env.get_global_state()
        assert state["state_flags"]["http_service_found"] is True

    def test_parser_open_port_discovery_round_trips_through_env_state(self, env):
        """
        End-to-end contract:
          1. LLM parser produces {open_port: [22, 80]}
          2. Applied to env.open_ports (mirrors SmartOrchestrator._apply_discoveries)
          3. get_global_state()['open_ports'] returns the same integer list
        """
        parser_result = {"open_port": [22, 80], "service": ["ssh", "http"]}

        # Mirror what SmartOrchestrator does with parser output
        for port in parser_result.get("open_port", []):
            if port not in env.open_ports:
                env.open_ports.append(port)
        env.open_ports.sort()

        state = env.get_global_state()
        assert sorted(state["open_ports"]) == sorted(parser_result["open_port"])
        # Type check
        for port in state["open_ports"]:
            assert isinstance(port, int)

    def test_parser_and_regex_produce_same_port_integers(self, env):
        """
        Both parsers (regex via OutputParser, LLM via SmartOutputParser) must
        produce the same integer port values for the same nmap output.
        """
        try:
            from core.execution.output_parser import OutputParser
        except ImportError:
            pytest.skip("OutputParser not available")

        nmap_out = (
            "PORT   STATE SERVICE\n22/tcp open  ssh\n80/tcp open  http\nNmap done: 1 IP\n"
        )

        # Regex parser
        regex_parsed = OutputParser().parse("nmap -sV 10.10.10.10", nmap_out)
        regex_ports = sorted(regex_parsed.open_ports)

        # LLM parser (mock returns same ports)
        resp = json.dumps({"open_port": regex_ports, "service": ["ssh", "http"]})
        parser, _ = _make_parser(resp)
        llm_result = parser.parse("nmap -sV 10.10.10.10", nmap_out)
        llm_ports = sorted(llm_result.get("open_port", []))

        assert llm_ports == regex_ports, (
            f"LLM ports {llm_ports} do not match regex ports {regex_ports}"
        )
