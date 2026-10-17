#!/usr/bin/env python3
"""
core/llm/mentor_backend.py — Mentor backend resolution (Mentor2Apprentice seam).

Ariaska's intelligence is its five RL agents (Red, Blue, Scout, Shadow, Orion)
driven by PPO Actor-Critic plus the registry/playbook decision systems. An LLM
is only ever a MENTOR: optional, selectively consulted, budget-controlled and
interchangeable. This module decides WHICH mentor backend a process uses:

    NONE    mentor-free. Pure RL. GPTManager.is_offline() -> True.
            (Reported owner baseline: PPO solved Metasploitable 1 & 2 here.)
    LOCAL   local model server (Ollama / llama-cpp / vLLM). CPU-friendly fast
            paths and rate limits apply (GPTManager.is_local_only() -> True).
    CLOUD   hosted API (Anthropic, Groq, OpenRouter, Together, custom
            OpenAI-compatible). Full multi-stage LLM pipelines are allowed.
    HYBRID  both available: System 1/2 tasks -> local, System 3 -> cloud.

Resolution is a PURE function of (operator request, local availability, cloud
availability). It never performs I/O, so it is trivially testable and the
same policy is applied everywhere.

Operator control:
    ARIASKA_MENTOR_BACKEND = auto | none | local | cloud | hybrid   (default auto)

``auto`` preserves the pre-existing behaviour exactly where it already worked
(local present -> LOCAL, nothing present -> NONE) and adds the previously
impossible case (cloud present, no local -> CLOUD). With both present ``auto``
still prefers LOCAL (Phase 58 decision: local wins when available); HYBRID and
CLOUD-with-local must be requested explicitly.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Tuple

logger = logging.getLogger("ariaska.llm.mentor_backend")

ENV_VAR = "ARIASKA_MENTOR_BACKEND"
VALID_REQUESTS: Tuple[str, ...] = ("auto", "none", "local", "cloud", "hybrid")


class MentorBackend(str, Enum):
    """Which LLM mentor (if any) serves this process."""

    NONE = "none"
    LOCAL = "local"
    CLOUD = "cloud"
    HYBRID = "hybrid"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


@dataclass(frozen=True)
class MentorBackendState:
    """Resolved backend plus the inputs that produced it (for logs/telemetry)."""

    backend: MentorBackend
    requested: str
    local_available: bool
    cloud_available: bool
    local_model: str = ""
    cloud_provider: str = ""
    cloud_model: str = ""
    reason: str = ""

    @property
    def is_mentor_free(self) -> bool:
        return self.backend == MentorBackend.NONE

    @property
    def uses_cloud(self) -> bool:
        return self.backend in (MentorBackend.CLOUD, MentorBackend.HYBRID)

    @property
    def uses_local(self) -> bool:
        return self.backend in (MentorBackend.LOCAL, MentorBackend.HYBRID)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["backend"] = self.backend.value
        return data


def requested_backend_from_env(env: Optional[Mapping[str, str]] = None) -> str:
    """Read and validate ``ARIASKA_MENTOR_BACKEND`` (defaults to ``auto``)."""
    source = os.environ if env is None else env
    raw = (source.get(ENV_VAR, "auto") or "auto").strip().lower()
    if raw not in VALID_REQUESTS:
        logger.warning(
            f"[MENTOR] Unknown {ENV_VAR}={raw!r}; expected one of {VALID_REQUESTS}. Using 'auto'."
        )
        return "auto"
    return raw


def resolve_mentor_backend(
    requested: str,
    local_available: bool,
    cloud_available: bool,
) -> Tuple[MentorBackend, str]:
    """Pure resolution of the mentor backend.

    Returns ``(backend, reason)`` where ``reason`` is a short human-readable
    explanation suitable for a startup log line.
    """
    req = (requested or "auto").strip().lower()
    if req not in VALID_REQUESTS:
        req = "auto"

    if req == "none":
        return MentorBackend.NONE, "operator requested mentor-free RL (ARIASKA_MENTOR_BACKEND=none)"

    if req == "local":
        if local_available:
            return MentorBackend.LOCAL, "operator requested local mentor; local server available"
        return MentorBackend.NONE, "operator requested local mentor but no local LLM server is reachable"

    if req == "cloud":
        if cloud_available:
            return MentorBackend.CLOUD, "operator requested cloud mentor; cloud provider configured"
        return MentorBackend.NONE, "operator requested cloud mentor but no cloud provider is configured"

    if req == "hybrid":
        if local_available and cloud_available:
            return MentorBackend.HYBRID, "operator requested hybrid; local + cloud available"
        if cloud_available:
            return MentorBackend.CLOUD, "operator requested hybrid; only cloud available"
        if local_available:
            return MentorBackend.LOCAL, "operator requested hybrid; only local available"
        return MentorBackend.NONE, "operator requested hybrid but neither local nor cloud is available"

    # auto — behaviour-preserving default
    if local_available:
        if cloud_available:
            return MentorBackend.LOCAL, "auto: local + cloud available; local preferred (set hybrid/cloud to use cloud)"
        return MentorBackend.LOCAL, "auto: local LLM server available"
    if cloud_available:
        return MentorBackend.CLOUD, "auto: no local server; cloud provider configured"
    return MentorBackend.NONE, "auto: no LLM backend available — mentor-free RL"


__all__ = [
    "ENV_VAR",
    "VALID_REQUESTS",
    "MentorBackend",
    "MentorBackendState",
    "requested_backend_from_env",
    "resolve_mentor_backend",
]
