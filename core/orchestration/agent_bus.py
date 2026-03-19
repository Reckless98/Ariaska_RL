"""Inter-Agent Message Bus — structured communication between Ariaska agents.

Phase 56: Enables agents to send requests, share findings, and coordinate
without going through the orchestrator. Messages are injected into agent
prompts by the orchestrator before each step.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("ariaska.orchestration.agent_bus")

# Valid message types for inter-agent communication
MSG_TYPES = frozenset({
    "REQUEST_RECON",       # Red→Scout: "enumerate port X deeper"
    "EXPLOIT_RESULT",      # Red→ALL: "shell obtained on target"
    "DEFENSE_ALERT",       # Blue→ALL: "detection risk high, go stealth"
    "STRATEGY_UPDATE",     # Orion→ALL: "pivot to web attack vector"
    "MEMORY_INSIGHT",      # Shadow→ALL: "similar target: use this approach"
    "PHASE_SUGGESTION",    # Any→Orion: "suggest phase transition"
    "DISCOVERY",           # Any→ALL: "found credentials/vuln/path"
    "COORDINATION",        # Any→specific: "wait for my result before acting"
})


@dataclass(slots=True)
class AgentMessage:
    """A single inter-agent message."""
    sender: str
    receiver: str  # agent name or "ALL"
    msg_type: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    step: int = 0
    priority: int = 0  # higher = more important

    def __post_init__(self) -> None:
        if self.msg_type not in MSG_TYPES:
            logger.warning(f"Unknown message type: {self.msg_type}")


class AgentMessageBus:
    """Centralized message bus for inter-agent communication.

    Messages persist for the current episode and are cleared between episodes.
    Each agent can query messages addressed to it (or ALL) and inject them
    into its LLM prompt context.
    """

    def __init__(self, max_messages: int = 100) -> None:
        self._messages: deque[AgentMessage] = deque(maxlen=max_messages)
        self._step_messages: dict[int, list[AgentMessage]] = {}

    def send(self, msg: AgentMessage) -> None:
        """Post a message to the bus."""
        self._messages.append(msg)
        self._step_messages.setdefault(msg.step, []).append(msg)
        logger.debug(f"[AgentBus] {msg.sender}→{msg.receiver}: {msg.msg_type}")

    def get_messages_for(
        self,
        agent: str,
        since_step: int = 0,
        msg_types: frozenset[str] | None = None,
    ) -> list[AgentMessage]:
        """Get messages addressed to a specific agent (or ALL), optionally filtered."""
        result = []
        for msg in self._messages:
            if msg.step < since_step:
                continue
            if msg.receiver not in (agent, "ALL"):
                continue
            if msg_types and msg.msg_type not in msg_types:
                continue
            result.append(msg)
        return sorted(result, key=lambda m: -m.priority)

    def inject_into_prompt(self, agent: str, since_step: int = 0) -> str:
        """Format pending messages as context string for prompt injection."""
        msgs = self.get_messages_for(agent, since_step=since_step)
        if not msgs:
            return ""
        lines = ["\n[AGENT COMMS]"]
        for msg in msgs[:5]:  # cap at 5 most important
            lines.append(f"  [{msg.sender}→{agent}] {msg.msg_type}: {msg.content}")
        return "\n".join(lines)

    def get_recent_discoveries(self, last_n: int = 5) -> list[AgentMessage]:
        """Get recent DISCOVERY messages from any agent."""
        return [
            m for m in reversed(self._messages)
            if m.msg_type == "DISCOVERY"
        ][:last_n]

    def clear_episode(self) -> None:
        """Clear all messages for a new episode."""
        self._messages.clear()
        self._step_messages.clear()

    @property
    def message_count(self) -> int:
        return len(self._messages)
