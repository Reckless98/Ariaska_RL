"""
tests/llm/test_mentor_backend.py — Mentor backend resolution + GPTManager wiring.

Pins the Mentor2Apprentice seam:
  * resolve_mentor_backend() is a pure, exhaustive policy
  * GPTManager derives is_offline()/is_local_only()/is_configured() from it
  * cloud-only mentorship works WITHOUT a local LLM server (previously impossible)
  * ARIASKA_MENTOR_BACKEND=none keeps the mentor-free RL path even with providers present
  * hybrid routes System 3 to cloud and System 1/2 to local
  * cloud cost is accounted; local stays $0
No network: providers are in-memory stubs.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")

from core.llm.llm_provider import LLMProvider, LLMResponse  # noqa: E402
from core.llm.mentor_backend import (  # noqa: E402
    MentorBackend,
    MentorBackendState,
    requested_backend_from_env,
    resolve_mentor_backend,
)


# ── Stubs ────────────────────────────────────────────────────────────────────

class _StubProvider(LLMProvider):
    """In-memory LLMProvider that records calls and returns canned text."""

    def __init__(self, name: str, model: str, text: str = "nmap -sV {target}", cost: float = 0.0,
                 default_timeout: float = 5.0) -> None:
        self.name = name
        self.model = model
        self.text = text
        self.cost = cost
        self.default_timeout = default_timeout
        self.calls: List[Dict[str, Any]] = []

    def chat_completion(self, messages, model, max_tokens=150, temperature=0.3,
                        response_format=None, timeout=None) -> LLMResponse:
        self.calls.append({"messages": messages, "model": model, "max_tokens": max_tokens,
                           "timeout": timeout, "response_format": response_format})
        return LLMResponse(text=self.text, model=self.model, provider=self.name,
                           input_tokens=100, output_tokens=20, total_tokens=120,
                           latency_ms=1.0, cost_usd=self.cost)

    def is_available(self) -> bool:
        return True

    def get_model_name(self) -> str:
        return self.model

    def get_provider_name(self) -> str:
        return self.name


class _StubLocal(_StubProvider):
    def get_provider_name(self) -> str:  # GPTManager uses the provider directly when "ollama"
        return "ollama"


def _make_gpt(monkeypatch, *, requested: str, local: Optional[_StubProvider],
              cloud: Optional[_StubProvider], detect_counter: Optional[Dict[str, int]] = None):
    """Build a GPTManager with detection replaced by in-memory stubs."""
    from core.gpt_manager import GPTManager
    import core.llm.model_router as mr

    monkeypatch.setenv("ARIASKA_MENTOR_BACKEND", requested)
    monkeypatch.setattr(mr, "_cloud_available", False)
    monkeypatch.setattr(mr, "_local_strategic_available", False)
    counter = detect_counter if detect_counter is not None else {}

    def _detect_local(self):
        counter["local"] = counter.get("local", 0) + 1
        if local is None:
            return False
        self._local_llm_provider = local
        self._local_llm_enabled = True
        return True

    def _detect_cloud(self):
        counter["cloud"] = counter.get("cloud", 0) + 1
        if cloud is None:
            return False
        self._install_cloud_provider(cloud)
        return True

    monkeypatch.setattr(GPTManager, "_detect_local_llm", _detect_local)
    monkeypatch.setattr(GPTManager, "_detect_local_strategic", lambda self: False)
    monkeypatch.setattr(GPTManager, "_detect_cloud_provider", _detect_cloud)
    return GPTManager(offline=False, enable_llm=True, require_llm=False)


# ── Pure policy ──────────────────────────────────────────────────────────────

class TestResolveMentorBackend:
    @pytest.mark.parametrize(
        "requested, local, cloud, expected",
        [
            # auto: behaviour-preserving
            ("auto", False, False, MentorBackend.NONE),
            ("auto", True, False, MentorBackend.LOCAL),
            ("auto", False, True, MentorBackend.CLOUD),   # NEW capability
            ("auto", True, True, MentorBackend.LOCAL),    # Phase 58: local wins
            # explicit none always wins
            ("none", True, True, MentorBackend.NONE),
            ("none", False, False, MentorBackend.NONE),
            # explicit local / cloud are strict (no silent substitution)
            ("local", True, True, MentorBackend.LOCAL),
            ("local", False, True, MentorBackend.NONE),
            ("cloud", True, True, MentorBackend.CLOUD),
            ("cloud", True, False, MentorBackend.NONE),
            # hybrid degrades gracefully
            ("hybrid", True, True, MentorBackend.HYBRID),
            ("hybrid", False, True, MentorBackend.CLOUD),
            ("hybrid", True, False, MentorBackend.LOCAL),
            ("hybrid", False, False, MentorBackend.NONE),
            # garbage → auto
            ("banana", True, False, MentorBackend.LOCAL),
            ("", False, True, MentorBackend.CLOUD),
        ],
    )
    def test_matrix(self, requested, local, cloud, expected):
        backend, reason = resolve_mentor_backend(requested, local, cloud)
        assert backend is expected
        assert isinstance(reason, str) and reason

    def test_env_reader_validates(self, monkeypatch):
        monkeypatch.setenv("ARIASKA_MENTOR_BACKEND", "CLOUD ")
        assert requested_backend_from_env() == "cloud"
        monkeypatch.setenv("ARIASKA_MENTOR_BACKEND", "nonsense")
        assert requested_backend_from_env() == "auto"
        monkeypatch.delenv("ARIASKA_MENTOR_BACKEND", raising=False)
        assert requested_backend_from_env() == "auto"

    def test_state_helpers(self):
        s = MentorBackendState(MentorBackend.HYBRID, "hybrid", True, True)
        assert s.uses_cloud and s.uses_local and not s.is_mentor_free
        assert s.to_dict()["backend"] == "hybrid"
        n = MentorBackendState(MentorBackend.NONE, "auto", False, False)
        assert n.is_mentor_free and not n.uses_cloud


# ── GPTManager wiring ────────────────────────────────────────────────────────

class TestGPTManagerBackendWiring:
    def test_mentor_free_when_nothing_available(self, monkeypatch):
        gpt = _make_gpt(monkeypatch, requested="auto", local=None, cloud=None)
        assert gpt.mentor_backend == "none"
        assert gpt.is_offline() is True
        assert gpt.is_configured() is False
        assert gpt.is_local_only() is True  # legacy contract preserved
        out = gpt.gpt_request("anything", task_type="tactical", agent_id="RedAgent")
        assert out == gpt._get_offline_placeholder("tactical")

    def test_cloud_only_is_online_and_routes_to_cloud(self, monkeypatch):
        cloud = _StubProvider("anthropic", "claude-opus-5-5", text="gobuster dir -u http://{target}", cost=0.0042)
        gpt = _make_gpt(monkeypatch, requested="auto", local=None, cloud=cloud)
        assert gpt.mentor_backend == "cloud"
        assert gpt.is_offline() is False
        assert gpt.is_configured() is True
        assert gpt.is_local_only() is False  # full LLM pipelines allowed

        out = gpt.gpt_request("pick next step", task_type="strategic", agent_id="OrionAgent")
        assert "gobuster" in out
        assert len(cloud.calls) == 1
        # outer timeout honours the provider's declared cap
        assert cloud.calls[0]["timeout"] <= cloud.default_timeout
        # cost accounted (local stays 0.0, cloud is real)
        assert gpt._episode_cost_usd == pytest.approx(0.0042)
        assert gpt._cumulative_cost_usd == pytest.approx(0.0042)
        summary = gpt.get_cost_summary()
        assert summary["mentor_backend"] == "cloud"
        assert summary["models"]["claude-opus-5-5"]["cost_usd"] == pytest.approx(0.0042)
        assert gpt.get_stats()["mentor_backend"] == "cloud"
        assert gpt.get_recent_calls(1)[0]["cost_usd"] == pytest.approx(0.0042)

    def test_cloud_only_tactical_call_also_routes_to_cloud(self, monkeypatch):
        cloud = _StubProvider("anthropic", "claude-opus-5-5", text="nikto -h {target}")
        gpt = _make_gpt(monkeypatch, requested="cloud", local=None, cloud=cloud)
        out = gpt.gpt_request("tactical please", task_type="tactical", agent_id="RedAgent")
        assert "nikto" in out and len(cloud.calls) == 1

    def test_requested_none_keeps_mentor_free_even_with_cloud(self, monkeypatch):
        cloud = _StubProvider("anthropic", "claude-opus-5-5")
        counter: Dict[str, int] = {}
        gpt = _make_gpt(monkeypatch, requested="none", local=None, cloud=cloud, detect_counter=counter)
        assert gpt.mentor_backend == "none"
        assert gpt.is_offline() is True
        # no probing at all when the operator asked for mentor-free
        assert counter.get("local", 0) == 0 and counter.get("cloud", 0) == 0
        gpt.gpt_request("x", task_type="tactical", agent_id="RedAgent")
        assert cloud.calls == []

    def test_requested_local_skips_cloud_detection(self, monkeypatch):
        local = _StubLocal("ollama", "ariaska-cybersec4")
        counter: Dict[str, int] = {}
        gpt = _make_gpt(monkeypatch, requested="local", local=local, cloud=None, detect_counter=counter)
        assert gpt.mentor_backend == "local"
        assert counter.get("cloud", 0) == 0
        assert gpt.is_local_only() is True and gpt.is_offline() is False

    def test_requested_cloud_without_cloud_is_mentor_free(self, monkeypatch):
        local = _StubLocal("ollama", "ariaska-cybersec4")
        gpt = _make_gpt(monkeypatch, requested="cloud", local=local, cloud=None)
        assert gpt.mentor_backend == "none"
        assert gpt.is_offline() is True

    def test_auto_with_both_prefers_local_and_never_calls_cloud(self, monkeypatch):
        local = _StubLocal("ollama", "ariaska-cybersec4", text="nmap -sC {target}")
        cloud = _StubProvider("anthropic", "claude-opus-5-5", text="SHOULD-NOT-BE-USED")
        gpt = _make_gpt(monkeypatch, requested="auto", local=local, cloud=cloud)
        assert gpt.mentor_backend == "local"
        assert gpt.is_local_only() is True
        out = gpt.gpt_request("plan", task_type="strategic", agent_id="OrionAgent")
        assert "nmap" in out
        assert cloud.calls == [] and len(local.calls) == 1

    def test_hybrid_routes_system3_to_cloud_and_system1_to_local(self, monkeypatch):
        local = _StubLocal("ollama", "ariaska-cybersec4", text="local-answer")
        cloud = _StubProvider("anthropic", "claude-opus-5-5", text="cloud-answer")
        gpt = _make_gpt(monkeypatch, requested="hybrid", local=local, cloud=cloud)
        assert gpt.mentor_backend == "hybrid"
        assert gpt.is_local_only() is True  # fast paths kept by default
        gpt.gpt_request("deep plan", task_type="strategic", agent_id="OrionAgent")
        gpt.gpt_request("classify", task_type="classification", agent_id="ScoutAgent")
        assert len(cloud.calls) == 1 and len(local.calls) == 1
        assert gpt._get_model_router().get_stats()["cloud_system3"] is True

    def test_hybrid_fast_path_can_be_disabled(self, monkeypatch):
        monkeypatch.setenv("ARIASKA_HYBRID_FAST_PATH", "0")
        local = _StubLocal("ollama", "ariaska-cybersec4")
        cloud = _StubProvider("anthropic", "claude-opus-5-5")
        gpt = _make_gpt(monkeypatch, requested="hybrid", local=local, cloud=cloud)
        assert gpt.is_local_only() is False

    def test_backend_state_snapshot(self, monkeypatch):
        cloud = _StubProvider("anthropic", "claude-opus-5-5")
        gpt = _make_gpt(monkeypatch, requested="auto", local=None, cloud=cloud)
        state = gpt.get_mentor_backend_state()
        assert state["backend"] == "cloud" and state["requested"] == "auto"
        assert state["cloud_provider"] == "anthropic" and state["cloud_model"] == "claude-opus-5-5"
        assert state["offline"] is False and state["local_only"] is False

    def test_require_llm_raises_only_when_no_backend(self, monkeypatch):
        from core.gpt_manager import GPTManager
        import core.llm.model_router as mr
        monkeypatch.setattr(mr, "_cloud_available", False)
        monkeypatch.setenv("ARIASKA_MENTOR_BACKEND", "auto")
        monkeypatch.setattr(GPTManager, "_detect_local_llm", lambda self: False)
        monkeypatch.setattr(GPTManager, "_detect_local_strategic", lambda self: False)
        monkeypatch.setattr(GPTManager, "_detect_cloud_provider", lambda self: False)
        with pytest.raises(RuntimeError):
            GPTManager(offline=False, enable_llm=True, require_llm=True)

        cloud = _StubProvider("anthropic", "claude-opus-5-5")
        monkeypatch.setattr(GPTManager, "_detect_cloud_provider",
                            lambda self: (self._install_cloud_provider(cloud) or True))
        gpt = GPTManager(offline=False, enable_llm=True, require_llm=True)
        assert gpt.mentor_backend == "cloud"

    def test_forced_offline_ignores_providers(self, monkeypatch):
        from core.gpt_manager import GPTManager
        monkeypatch.setenv("ARIASKA_MENTOR_BACKEND", "auto")
        gpt = GPTManager(offline=True, enable_llm=False)
        assert gpt.is_offline() is True and gpt.mentor_backend == "none"


class TestFakeGPTManagerContract:
    def test_fake_exposes_backend_surface(self):
        from core.testing.fake_gpt_manager import FakeGPTManager
        fake = FakeGPTManager()
        assert fake.is_local_only() is False
        assert fake.is_offline() is False
        assert fake.mentor_backend == "fake"
        assert fake.get_mentor_backend_state()["backend"] == "fake"
