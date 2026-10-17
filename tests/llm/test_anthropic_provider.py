"""
tests/llm/test_anthropic_provider.py — Optional Anthropic mentor provider.

Everything runs without network and without requiring the `anthropic` SDK:
the provider accepts an injected client, and error classification is by
exception class name / status_code so plain fake exceptions suffice.
"""

from __future__ import annotations

import os
import sys
import types
from typing import Any, Dict, List

import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")

from core.llm.anthropic_provider import (  # noqa: E402
    AnthropicProvider,
    DEFAULT_MODEL,
    estimate_cost_usd,
)


# ── Fakes ────────────────────────────────────────────────────────────────────

class _Block:
    def __init__(self, type_: str, text: str = "") -> None:
        self.type = type_
        self.text = text


class _Usage:
    def __init__(self, i: int, o: int) -> None:
        self.input_tokens = i
        self.output_tokens = o


class _StopDetails:
    def __init__(self, category: str) -> None:
        self.category = category


class _Resp:
    def __init__(self, text="", stop_reason="end_turn", model="claude-opus-5-5", i=100, o=50,
                 category=None, blocks=None):
        self.content = blocks if blocks is not None else [_Block("thinking", ""), _Block("text", text)]
        self.stop_reason = stop_reason
        self.model = model
        self.usage = _Usage(i, o)
        self.stop_details = _StopDetails(category) if category else None


class _Messages:
    def __init__(self, owner: "_FakeClient", beta: bool) -> None:
        self._owner = owner
        self._beta = beta

    def create(self, **kwargs) -> Any:
        self._owner.calls.append({"beta": self._beta, **kwargs})
        action = self._owner.script.pop(0) if self._owner.script else _Resp("nmap -sV 10.0.0.1")
        if isinstance(action, BaseException):
            raise action
        return action


class _FakeClient:
    def __init__(self, script: List[Any] | None = None) -> None:
        self.script = list(script or [])
        self.calls: List[Dict[str, Any]] = []
        self.with_options_kwargs: List[Dict[str, Any]] = []
        self.messages = _Messages(self, beta=False)
        self.beta = types.SimpleNamespace(messages=_Messages(self, beta=True))

    def with_options(self, **kwargs) -> "_FakeClient":
        self.with_options_kwargs.append(kwargs)
        return self


class RateLimitError(Exception):
    status_code = 429


class AuthenticationError(Exception):
    status_code = 401


class BadRequestError(Exception):
    status_code = 400


class APIConnectionError(Exception):
    pass


def _provider(script=None, **kw) -> tuple[AnthropicProvider, _FakeClient]:
    client = _FakeClient(script)
    kw.setdefault("effort", "medium")
    kw.setdefault("use_fallbacks", True)
    kw.setdefault("timeout", 60.0)
    kw.setdefault("min_output_tokens", 2048)
    kw.setdefault("workspace_id", "")
    p = AnthropicProvider(api_key="sk-test", model="claude-opus-5-5", client=client, **kw)
    return p, client


# ── Detection ────────────────────────────────────────────────────────────────

class TestAutoDetect:
    def test_none_without_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert AnthropicProvider.auto_detect() is None

    def test_none_without_sdk(self, monkeypatch, caplog):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setattr(AnthropicProvider, "sdk_available", classmethod(lambda cls: False))
        assert AnthropicProvider.auto_detect() is None
        assert "not installed" in caplog.text

    def test_detects_with_key_and_sdk(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setenv("ARIASKA_ANTHROPIC_MODEL", "claude-sonnet-5-5")
        monkeypatch.setenv("ARIASKA_ANTHROPIC_EFFORT", "low")
        monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_123")
        monkeypatch.setattr(AnthropicProvider, "sdk_available", classmethod(lambda cls: True))
        p = AnthropicProvider.auto_detect()
        assert p is not None
        assert p.get_provider_name() == "anthropic"
        assert p.get_model_name() == "claude-sonnet-5-5"
        assert p.effort == "low" and p.workspace_id == "wrkspc_123"
        assert p.is_available() is True

    def test_default_model_and_invalid_effort(self, monkeypatch):
        monkeypatch.delenv("ARIASKA_ANTHROPIC_MODEL", raising=False)
        monkeypatch.setenv("ARIASKA_ANTHROPIC_EFFORT", "ultra")
        p = AnthropicProvider(api_key="k", workspace_id="")
        assert p.model == DEFAULT_MODEL == "claude-opus-5-5"
        assert p.effort == "medium"

    def test_registry_lists_anthropic_first_in_auto(self):
        from core.llm.cloud_provider_registry import detection_order, normalize_preference
        assert detection_order("auto")[0] == "anthropic"
        assert detection_order("groq") == ["openai-compatible"]
        assert normalize_preference("claude") == "anthropic"
        assert normalize_preference("nope") == "auto"

    def test_registry_detect_prefers_available(self, monkeypatch):
        from core.llm import cloud_provider_registry as reg
        from core.llm.llm_provider import LLMProvider, LLMResponse

        class _P(LLMProvider):
            def __init__(self, name): self.name = name
            def chat_completion(self, *a, **k): return LLMResponse(text="", model="m", provider=self.name)
            def is_available(self): return True
            def get_model_name(self): return "m"
            def get_provider_name(self): return self.name

        monkeypatch.setitem(reg._FACTORIES, "anthropic", lambda: None)
        monkeypatch.setitem(reg._FACTORIES, "openai-compatible", lambda: _P("groq"))
        assert reg.detect_cloud_provider("auto").get_provider_name() == "groq"
        monkeypatch.setitem(reg._FACTORIES, "anthropic", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        assert reg.detect_cloud_provider("auto").get_provider_name() == "groq"  # factory error skipped
        assert reg.detect_cloud_provider("anthropic") is None


# ── Request translation ──────────────────────────────────────────────────────

class TestBuildRequest:
    def test_system_split_json_and_floor(self):
        msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hi"}]
        params = AnthropicProvider.build_request(msgs, "claude-opus-5-5", 150, "low",
                                                 response_format={"type": "json_object"}, min_output_tokens=2048)
        assert params["model"] == "claude-opus-5-5"
        assert params["max_tokens"] == 2048  # floor applied
        assert params["system"].startswith("SYS") and "JSON" in params["system"]
        assert params["messages"] == [{"role": "user", "content": "hi"}]
        assert params["output_config"] == {"effort": "low"}
        assert "temperature" not in params and "top_p" not in params

    def test_first_message_must_be_user(self):
        params = AnthropicProvider.build_request([{"role": "system", "content": "only system"}],
                                                 "claude-opus-5-5", 4000, "medium")
        assert params["messages"][0]["role"] == "user"
        assert params["max_tokens"] == 4000  # above floor → unchanged
        assert "system" in params

    def test_cost_table(self):
        assert estimate_cost_usd("claude-opus-5-5", 1_000_000, 0) == pytest.approx(4.0)
        assert estimate_cost_usd("claude-opus-5-5", 0, 1_000_000) == pytest.approx(20.0)
        assert estimate_cost_usd("claude-haiku-5-5", 1_000_000, 1_000_000) == pytest.approx(0.60)
        assert estimate_cost_usd("unknown-model", 1000, 1000) == 0.0


# ── chat_completion behaviour ────────────────────────────────────────────────

class TestChatCompletion:
    def test_success_uses_beta_fallbacks_and_reports_cost(self):
        p, client = _provider([_Resp("gobuster dir -u http://10.0.0.1", i=1000, o=100)])
        r = p.chat_completion([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}],
                              model="local-llm", max_tokens=300, timeout=12.0)
        assert r.success and "gobuster" in r.text
        assert r.provider == "anthropic" and r.model == "claude-opus-5-5"
        assert r.input_tokens == 1000 and r.output_tokens == 100 and r.total_tokens == 1100
        assert r.cost_usd == pytest.approx(1000 * 4.0 / 1e6 + 100 * 20.0 / 1e6)
        call = client.calls[0]
        assert call["beta"] is True and call["fallbacks"] == "default"
        assert call["betas"] == ["server-side-fallback-2026-07-01"]
        assert call["model"] == "claude-opus-5-5"  # 'local-llm' alias ignored
        assert client.with_options_kwargs == [{"timeout": 12.0}]
        assert p.get_stats()["total_cost_usd"] == pytest.approx(r.cost_usd)

    def test_explicit_claude_model_override(self):
        p, client = _provider([_Resp("x", model="claude-haiku-5-5")])
        p.chat_completion([{"role": "user", "content": "U"}], model="claude-haiku-5-5")
        assert client.calls[0]["model"] == "claude-haiku-5-5"

    def test_without_fallbacks_uses_standard_endpoint(self):
        p, client = _provider([_Resp("ok")], use_fallbacks=False)
        r = p.chat_completion([{"role": "user", "content": "U"}], model="x")
        assert r.success and client.calls[0]["beta"] is False
        assert "fallbacks" not in client.calls[0]

    def test_refusal_is_error_not_exception(self):
        p, _ = _provider([_Resp("", stop_reason="refusal", category="cyber", blocks=[])])
        r = p.chat_completion([{"role": "user", "content": "U"}], model="x")
        assert r.success is False and r.text == "" and r.error == "refusal:cyber"
        assert p.get_stats()["refusals"] == 1

    def test_empty_text_is_error(self):
        p, _ = _provider([_Resp("", stop_reason="max_tokens", blocks=[_Block("thinking", "")])])
        r = p.chat_completion([{"role": "user", "content": "U"}], model="x")
        assert not r.success and "empty response" in (r.error or "")

    def test_rate_limit_cools_down_then_short_circuits(self):
        p, client = _provider([RateLimitError("429 slow down"), _Resp("never")])
        r1 = p.chat_completion([{"role": "user", "content": "U"}], model="x")
        assert not r1.success and "RateLimitError" in (r1.error or "")
        r2 = p.chat_completion([{"role": "user", "content": "U"}], model="x")
        assert not r2.success and "cooling down" in (r2.error or "")
        assert len(client.calls) == 1  # second call never hit the API
        assert p.is_available() is True  # temporary, not disabled

    def test_auth_error_disables_provider(self):
        p, client = _provider([AuthenticationError("bad key")])
        r = p.chat_completion([{"role": "user", "content": "U"}], model="x")
        assert not r.success
        assert p.is_available() is False
        r2 = p.chat_completion([{"role": "user", "content": "U"}], model="x")
        assert "disabled" in (r2.error or "") and len(client.calls) == 1

    def test_connection_error_is_soft(self):
        p, _ = _provider([APIConnectionError("dns")])
        r = p.chat_completion([{"role": "user", "content": "U"}], model="x")
        assert not r.success and "APIConnectionError" in (r.error or "")
        assert p.is_available() is True

    def test_fallback_rejection_retries_without_fallbacks(self):
        p, client = _provider([BadRequestError("Unknown beta: server-side-fallback / fallbacks not supported"),
                               _Resp("retry-ok")])
        r = p.chat_completion([{"role": "user", "content": "U"}], model="x")
        assert r.success and r.text == "retry-ok"
        assert client.calls[0]["beta"] is True and client.calls[1]["beta"] is False
        assert p.use_fallbacks is False and p.get_stats()["fallback_disabled"] is True

    def test_other_bad_request_is_reported_once(self):
        p, client = _provider([BadRequestError("workspace header required")])
        r = p.chat_completion([{"role": "user", "content": "U"}], model="x")
        assert not r.success and "BadRequestError" in (r.error or "")
        assert len(client.calls) == 1 and p.use_fallbacks is True

    def test_client_construction_passes_workspace_header(self, monkeypatch):
        captured: Dict[str, Any] = {}

        class _Anthropic:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.messages = types.SimpleNamespace(create=lambda **k: _Resp("ok"))
                self.beta = types.SimpleNamespace(messages=types.SimpleNamespace(create=lambda **k: _Resp("ok")))
            def with_options(self, **kwargs):
                return self

        fake_sdk = types.ModuleType("anthropic")
        fake_sdk.Anthropic = _Anthropic
        monkeypatch.setitem(sys.modules, "anthropic", fake_sdk)
        p = AnthropicProvider(api_key="sk-test", workspace_id="wrkspc_abc", timeout=42.0, use_fallbacks=False)
        r = p.chat_completion([{"role": "user", "content": "U"}], model="x")
        assert r.success
        assert captured["api_key"] == "sk-test"
        assert captured["default_headers"] == {"anthropic-workspace-id": "wrkspc_abc"}
        assert captured["timeout"] == 42.0 and captured["max_retries"] == 1

    def test_no_workspace_header_when_unset(self, monkeypatch):
        captured: Dict[str, Any] = {}

        class _Anthropic:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.messages = types.SimpleNamespace(create=lambda **k: _Resp("ok"))
                self.beta = types.SimpleNamespace(messages=types.SimpleNamespace(create=lambda **k: _Resp("ok")))
            def with_options(self, **kwargs):
                return self

        fake_sdk = types.ModuleType("anthropic")
        fake_sdk.Anthropic = _Anthropic
        monkeypatch.setitem(sys.modules, "anthropic", fake_sdk)
        p = AnthropicProvider(api_key="sk-test", workspace_id="", use_fallbacks=False)
        p.chat_completion([{"role": "user", "content": "U"}], model="x")
        assert captured["default_headers"] is None


class TestGPTManagerWithAnthropicProvider:
    def test_end_to_end_routing_without_local_llm(self, monkeypatch):
        """GPTManager + AnthropicProvider (fake client): cloud-only mentor works, RL stays in charge."""
        from core.gpt_manager import GPTManager
        import core.llm.model_router as mr

        monkeypatch.setenv("ARIASKA_MENTOR_BACKEND", "auto")
        monkeypatch.setattr(mr, "_cloud_available", False)
        monkeypatch.setattr(mr, "_local_strategic_available", False)
        p, client = _provider([_Resp("searchsploit vsftpd 2.3.4", i=500, o=40)])
        monkeypatch.setattr(GPTManager, "_detect_local_llm", lambda self: False)
        monkeypatch.setattr(GPTManager, "_detect_local_strategic", lambda self: False)
        monkeypatch.setattr(GPTManager, "_detect_cloud_provider",
                            lambda self: (self._install_cloud_provider(p) or True))
        gpt = GPTManager(offline=False, enable_llm=True, require_llm=False)
        assert gpt.mentor_backend == "cloud" and not gpt.is_offline() and not gpt.is_local_only()
        out = gpt.gpt_request("what next?", task_type="tactical", agent_id="RedAgent", max_tokens=200)
        assert "searchsploit" in out
        assert client.calls[0]["max_tokens"] >= 2048  # floor protects against thinking starvation
        assert gpt._cumulative_cost_usd == pytest.approx(500 * 4.0 / 1e6 + 40 * 20.0 / 1e6)
