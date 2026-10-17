#!/usr/bin/env python3
"""
core/llm/gateway_chat_client.py — OpenAI-chat-shaped adapter over GPTManager.

``SmartMentor`` (and any legacy caller) expects an object exposing
``await client.chat.completions.create(model=..., messages=..., max_tokens=..., ...)``
that returns ``response.choices[0].message.content`` and ``response.usage.total_tokens``.

This adapter provides exactly that surface while routing EVERY call through
``GPTManager.gpt_request`` — so invariant #1 (all LLM calls through GPTManager)
holds and the resolved mentor backend (none / local / cloud / hybrid), budgets,
caching, cost accounting and offline placeholders apply uniformly. It owns no
SDK and opens no network connection of its own.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ariaska.llm.gateway_chat_client")


@dataclass
class GatewayMessage:
    content: str
    role: str = "assistant"


@dataclass
class GatewayChoice:
    message: GatewayMessage
    index: int = 0
    finish_reason: str = "stop"


@dataclass
class GatewayUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class GatewayChatResponse:
    """Minimal chat-completion-shaped response built from a gpt_request() result."""

    choices: List[GatewayChoice]
    usage: GatewayUsage = field(default_factory=GatewayUsage)
    model: str = "gateway"
    id: str = "gateway-chat"
    provider: str = "gpt_manager"


def _wants_json(response_format: Any) -> bool:
    if isinstance(response_format, dict):
        return response_format.get("type") == "json_object"
    return isinstance(response_format, str) and response_format == "json_object"


def _split_messages(messages: Optional[List[Dict[str, Any]]]) -> tuple[Optional[str], str]:
    """Return (system_prompt, user_prompt) from OpenAI-style messages."""
    system_parts: List[str] = []
    user_parts: List[str] = []
    for m in messages or []:
        role = m.get("role")
        content = m.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        if role == "system":
            system_parts.append(content)
        elif role == "assistant":
            user_parts.append(f"[assistant]\n{content}")
        else:  # user / tool / anything else carries the request
            user_parts.append(content)
    system = "\n\n".join(p for p in system_parts if p) or None
    return system, "\n\n".join(p for p in user_parts if p)


class _Completions:
    def __init__(self, client: "GatewayChatClient") -> None:
        self._client = client

    async def create(self, **params: Any) -> GatewayChatResponse:
        # gpt_request is synchronous (it manages its own timeouts); keep the
        # caller's event loop responsive by running it in a worker thread.
        return await asyncio.to_thread(self._client.create_sync, **params)


class _Chat:
    def __init__(self, client: "GatewayChatClient") -> None:
        self.completions = _Completions(client)


class GatewayChatClient:
    """Chat-completions-shaped façade whose only backend is ``GPTManager.gpt_request``."""

    def __init__(
        self,
        gpt_manager: Any,
        agent_id: str = "mentor",
        task_type: str = "tactical",
        timeout: Optional[float] = None,
    ) -> None:
        self._gm = gpt_manager
        self.agent_id = agent_id
        self.task_type = task_type
        self.timeout = timeout
        self.chat = _Chat(self)
        self.calls = 0

    # ── core ─────────────────────────────────────────────────────────────

    def create_sync(
        self,
        model: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 150,
        temperature: Optional[float] = None,  # routing/provider decides; accepted for compatibility
        response_format: Any = None,
        use_cache: Optional[bool] = None,
        **_ignored: Any,
    ) -> GatewayChatResponse:
        system_prompt, prompt = _split_messages(messages)
        fmt = "json_object" if _wants_json(response_format) else None
        started = time.time()
        self.calls += 1

        kwargs: Dict[str, Any] = {
            "task_type": self.task_type,
            "agent_id": self.agent_id,
            "max_tokens": int(max_tokens or 150),
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if use_cache is not None:
            kwargs["use_cache"] = use_cache
        if system_prompt:
            kwargs["system_prompt"] = system_prompt
        if fmt:
            kwargs["response_format"] = fmt
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        # Model selection stays with GPTManager/ModelRouter (mentor backend authority);
        # an explicit model is forwarded only when it is not a routing alias.
        if model and model not in ("local-llm", "local", "gateway", "cloud-reasoning"):
            kwargs["model"] = model

        text = self._gm.gpt_request(prompt, **kwargs)
        if text is None:
            text = ""
        text = str(text)

        record_reader = getattr(self._gm, "get_request_record", None)
        if callable(record_reader):
            rec = record_reader()
            usage = GatewayUsage(
                prompt_tokens=int(rec.get("input_tokens", 0)),
                completion_tokens=int(rec.get("output_tokens", 0)),
                total_tokens=int(rec.get("tokens", 0)),
            )
            served_model = rec.get("model") or model or "gateway"
        else:  # Compatibility with older injected gateways.
            usage = self._usage_since(started)
            served_model = self._served_model(model)
        return GatewayChatResponse(
            choices=[GatewayChoice(message=GatewayMessage(content=text))],
            usage=usage,
            model=served_model,
        )

    # ── helpers ──────────────────────────────────────────────────────────

    def _usage_since(self, started: float) -> GatewayUsage:
        """Read token usage of the call just made from GPTManager's call log, if any."""
        try:
            recent = getattr(self._gm, "get_recent_calls", None)
            if callable(recent):
                records = recent(1) or []
                if records:
                    rec = records[-1]
                    if float(rec.get("timestamp", 0.0)) >= started - 1e-3:
                        return GatewayUsage(
                            prompt_tokens=int(rec.get("input_tokens", 0) or 0),
                            completion_tokens=int(rec.get("output_tokens", 0) or 0),
                            total_tokens=int(rec.get("tokens", 0) or 0),
                        )
        except Exception:  # noqa: BLE001 — accounting must never break a mentor call
            pass
        return GatewayUsage()

    def _served_model(self, requested: Optional[str]) -> str:
        try:
            state = getattr(self._gm, "get_mentor_backend_state", None)
            if callable(state):
                st = state() or {}
                return str(st.get("cloud_model") or st.get("local_model") or requested or "gateway")
        except Exception:  # noqa: BLE001
            pass
        return str(requested or "gateway")

    def get_stats(self) -> Dict[str, Any]:
        return {
            "calls": self.calls,
            "agent_id": self.agent_id,
            "task_type": self.task_type,
            "mentor_backend": str(getattr(self._gm, "mentor_backend", "unknown")),
        }


__all__ = [
    "GatewayChatClient",
    "GatewayChatResponse",
    "GatewayChoice",
    "GatewayMessage",
    "GatewayUsage",
]
