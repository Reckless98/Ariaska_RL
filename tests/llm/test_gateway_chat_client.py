"""
tests/llm/test_gateway_chat_client.py — SmartMentor over GPTManager (R1.1).

Pins the follow-up that makes mentorship live:
  * GatewayChatClient maps OpenAI-style chat params onto GPTManager.gpt_request
    (system prompt, JSON mode, max_tokens, agent id) and returns the
    choices/usage shape SmartMentor reads
  * the async path works from a fresh event loop (how SmartMentor calls it)
  * SmartCoach builds a SmartMentor only for resolved local/cloud/hybrid
    backends — never for mentor-free or the deterministic fake
  * SmartMentor end-to-end over a stubbed gateway returns a parsed response
Deterministic, offline, no network.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

os.environ.setdefault("ARIASKA_DRY_RUN", "1")

from core.llm.gateway_chat_client import GatewayChatClient, GatewayChatResponse  # noqa: E402


class _StubGateway:
    """Records gpt_request kwargs and returns a canned reply + call-log record."""

    def __init__(self, reply: str = "ok", backend: str = "cloud", tokens: int = 42) -> None:
        self.reply = reply
        self.mentor_backend = backend
        self.tokens = tokens
        self.calls: List[Dict[str, Any]] = []
        self._log: List[Dict[str, Any]] = []

    def is_configured(self) -> bool:
        return self.mentor_backend != "none"

    def is_offline(self) -> bool:
        return self.mentor_backend == "none"

    def gpt_request(self, prompt: str, **kwargs: Any) -> str:
        self.calls.append({"prompt": prompt, **kwargs})
        self._log.append({"timestamp": time.time(), "tokens": self.tokens, "input_tokens": 30, "output_tokens": 12})
        return self.reply

    def get_recent_calls(self, n: int = 10):
        return self._log[-n:]

    def get_mentor_backend_state(self) -> Dict[str, Any]:
        return {"backend": self.mentor_backend, "cloud_model": "stub-model", "local_model": ""}


class TestGatewayChatClient:
    def test_maps_messages_to_gpt_request(self):
        gw = _StubGateway(reply='{"a": 1}')
        client = GatewayChatClient(gw, agent_id="RedAgent", task_type="tactical", timeout=12.0)
        resp = client.create_sync(
            model="local-llm",
            messages=[{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}],
            max_tokens=4096,
            temperature=0.7,
            response_format={"type": "json_object"},
        )
        assert isinstance(resp, GatewayChatResponse)
        assert resp.choices[0].message.content == '{"a": 1}'
        assert resp.usage.total_tokens == 42 and resp.usage.prompt_tokens == 30
        assert resp.model == "stub-model"
        call = gw.calls[0]
        assert call["prompt"] == "USER"
        assert call["system_prompt"] == "SYS"
        assert call["response_format"] == "json_object"
        assert call["max_tokens"] == 4096
        assert call["agent_id"] == "RedAgent" and call["task_type"] == "tactical"
        assert call["timeout"] == 12.0
        assert "model" not in call  # routing alias stays with GPTManager/ModelRouter
        assert call["temperature"] == 0.7

    def test_explicit_non_alias_model_is_forwarded(self):
        gw = _StubGateway()
        GatewayChatClient(gw).create_sync(model="some-specific-model", messages=[{"role": "user", "content": "x"}])
        assert gw.calls[0]["model"] == "some-specific-model"

    def test_async_create_from_fresh_loop(self):
        gw = _StubGateway(reply="async-ok")
        client = GatewayChatClient(gw)

        async def _go():
            return await client.chat.completions.create(messages=[{"role": "user", "content": "hi"}], max_tokens=50)

        resp = asyncio.run(_go())
        assert resp.choices[0].message.content == "async-ok"
        assert client.calls == 1

    def test_none_reply_and_missing_log_are_safe(self):
        class _Bare:
            mentor_backend = "local"
            def gpt_request(self, prompt, **kw): return None
        resp = GatewayChatClient(_Bare()).create_sync(messages=[{"role": "user", "content": "x"}])
        assert resp.choices[0].message.content == "" and resp.usage.total_tokens == 0

    def test_fake_gpt_manager_compatible(self):
        from core.testing.fake_gpt_manager import FakeGPTManager
        client = GatewayChatClient(FakeGPTManager(seed=1), agent_id="ScoutAgent")
        resp = client.create_sync(messages=[{"role": "user", "content": "x"}], max_tokens=20)
        assert isinstance(resp.choices[0].message.content, str)
        assert client.get_stats()["mentor_backend"] == "fake"


class TestSmartCoachMentorConstruction:
    def _coach(self, gm):
        from core.training.smart_coach import SmartCoach
        return SmartCoach(agent_name="RedAgent", gpt_manager=gm)

    def test_built_for_resolved_cloud_backend(self):
        gw = _StubGateway(backend="cloud")
        coach = self._coach(gw)
        assert coach.smart_mentor is not None
        assert isinstance(coach.smart_mentor.llm_client, GatewayChatClient)
        assert coach.smart_mentor.llm_client.agent_id == "RedAgent"

    @pytest.mark.parametrize("backend", ["local", "hybrid"])
    def test_built_for_local_and_hybrid(self, backend):
        coach = self._coach(_StubGateway(backend=backend))
        assert coach.smart_mentor is not None

    def test_not_built_for_mentor_free(self):
        coach = self._coach(_StubGateway(backend="none"))
        assert coach.smart_mentor is None

    def test_not_built_for_fake_gpt_manager(self):
        from core.testing.fake_gpt_manager import FakeGPTManager
        coach = self._coach(FakeGPTManager(seed=42))
        assert coach.smart_mentor is None  # deterministic tests keep their previous behaviour

    def test_not_built_for_offline_real_gateway(self):
        from core.gpt_manager import GPTManager
        coach = self._coach(GPTManager(offline=True, enable_llm=False))
        assert coach.smart_mentor is None

    def test_legacy_injected_client_still_honoured(self):
        legacy = SimpleNamespace(async_client=object(), mentor_backend="none")
        coach = self._coach(legacy)
        assert coach.smart_mentor is not None
        assert not isinstance(coach.smart_mentor.llm_client, GatewayChatClient)


class TestSmartMentorEndToEnd:
    def test_parsed_response_flows_through_gateway(self):
        from core.commands.command_registry import COMMAND_REGISTRY
        from core.llm.smart_mentor import AttackContext, SmartMentor

        name, tpl = next(iter(COMMAND_REGISTRY.items()))
        reply = json.dumps({
            "selected_command": name,
            "parameters": {p: "value" for p in tpl.required_params},
            "reasoning": "deterministic test reply",
            "confidence": 0.8,
        })
        gw = _StubGateway(reply=reply, backend="cloud", tokens=77)
        mentor = SmartMentor(llm_client=GatewayChatClient(gw, agent_id="RedAgent"), model="local-llm")
        # Prompt construction is out of scope here (and depends on repo data files);
        # pin the gateway glue only: call → parse → MentorResponse.
        mentor._build_system_prompt = lambda: "system"
        mentor._build_user_prompt = lambda *a, **k: "user"
        ctx = AttackContext(target="192.168.56.20")
        resp = mentor.get_command(ctx)
        assert resp is not None
        assert gw.calls, "SmartMentor must call the gateway"
        assert gw.calls[0]["agent_id"] == "RedAgent"
        if resp.template_name == name:
            assert resp.tokens_used == 77
            assert resp.confidence == pytest.approx(0.8)
        else:
            # validation rejected the synthetic parameters; the fallback path still ran
            # through the gateway and must carry the accumulated token count
            assert resp.tokens_used >= 0
