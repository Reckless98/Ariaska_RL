"""core/multiagent/agent_comm.py — Phase 49: Multi-Agent Communication.

Structured message passing between agents for coordinated decision-making.
Implements a communication protocol where agents can share observations,
requests, and strategic directives.

Design:
  - Typed messages: OBSERVATION, REQUEST, DIRECTIVE, ACK
  - Priority-based message queue per agent.
  - Attention-based message aggregation for neural integration.
  - Message history with TTL for recency weighting.

Integration:
  - Each agent has a CommChannel attached in AgentManager.
  - Messages fused into state representation via attention.
  - Compatible with existing strategic_directive.py protocol.

Architecture:
    Agent_i → CommChannel → MessageBus → CommChannel → Agent_j
              ↓                                        ↓
           encode_msg                              decode_msgs
              ↓                                        ↓
           state_fusion (attention over messages)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("ariaska.multiagent.agent_comm")


class MessageType(Enum):
    """Types of inter-agent messages."""
    OBSERVATION = "observation"   # Sharing what was seen
    REQUEST = "request"          # Asking another agent to do something
    DIRECTIVE = "directive"      # Strategic command (from Orion)
    ACK = "ack"                  # Acknowledgment
    ALERT = "alert"              # Urgent notification (from Shadow)
    DISCOVERY = "discovery"      # Sharing a new discovery


@dataclass
class AgentMessage:
    """A message between agents.

    Attributes:
        sender: Agent ID of the sender.
        receiver: Agent ID of the receiver (or "*" for broadcast).
        msg_type: Type of message.
        content: Message payload (dict).
        priority: Priority level (0=low, 1=normal, 2=high, 3=critical).
        timestamp: Creation time.
        ttl: Time-to-live in seconds.
        embedding: Optional dense embedding for neural processing.
    """
    sender: str
    receiver: str
    msg_type: MessageType
    content: Dict[str, Any]
    priority: int = 1
    timestamp: float = field(default_factory=time.time)
    ttl: float = 60.0  # 60 second default TTL
    embedding: Optional[torch.Tensor] = None

    @property
    def is_expired(self) -> bool:
        """Check if message has expired."""
        return (time.time() - self.timestamp) > self.ttl

    @property
    def age(self) -> float:
        """Age of message in seconds."""
        return time.time() - self.timestamp


@dataclass
class CommConfig:
    """Configuration for multi-agent communication.

    Args:
        enabled: Master switch.
        msg_dim: Dimension of message embeddings.
        max_queue_size: Max messages per agent queue.
        attention_heads: Number of attention heads for message fusion.
        state_dim: Agent state dimension for fusion.
        ttl_default: Default message TTL in seconds.
        broadcast_enabled: Allow broadcast messages.
    """
    enabled: bool = True
    msg_dim: int = 64
    max_queue_size: int = 50
    attention_heads: int = 4
    state_dim: int = 512
    ttl_default: float = 60.0
    broadcast_enabled: bool = True


class MessageEncoder(nn.Module):
    """Encodes message content into dense embeddings for neural processing."""

    def __init__(self, msg_dim: int = 64) -> None:
        super().__init__()
        # Encode: [msg_type_onehot(6) + priority(1) + age_norm(1) + content_hash(8)] → msg_dim
        self.net = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, msg_dim),
        )

    def forward(self, messages: List[AgentMessage]) -> torch.Tensor:
        """Encode a list of messages into embeddings.

        Args:
            messages: List of AgentMessage objects.

        Returns:
            (N, msg_dim) tensor of message embeddings.
        """
        if not messages:
            return torch.zeros(1, self.net[-1].out_features)

        features = []
        for msg in messages:
            feat = torch.zeros(16)
            # Message type one-hot (6 types)
            type_idx = list(MessageType).index(msg.msg_type)
            feat[type_idx] = 1.0
            # Priority normalised
            feat[6] = msg.priority / 3.0
            # Age normalised (0=new, 1=old)
            feat[7] = min(1.0, msg.age / max(msg.ttl, 1.0))
            # Content hash features (simple hash → 8 dims)
            content_str = str(sorted(msg.content.items()))
            h = hash(content_str)
            for j in range(8):
                feat[8 + j] = ((h >> (j * 4)) & 0xF) / 15.0
            features.append(feat)

        batch = torch.stack(features)
        return self.net(batch)


class MessageFusion(nn.Module):
    """Attention-based fusion of messages into agent state.

    Takes agent's current state and incoming messages, produces
    an augmented state that incorporates communication context.
    """

    def __init__(self, state_dim: int, msg_dim: int, n_heads: int = 4) -> None:
        super().__init__()
        self.state_proj = nn.Linear(state_dim, msg_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=msg_dim,
            num_heads=n_heads,
            batch_first=True,
        )
        self.gate = nn.Sequential(
            nn.Linear(msg_dim, state_dim),
            nn.Sigmoid(),
        )
        self.value_proj = nn.Linear(msg_dim, state_dim)

    def forward(
        self,
        state: torch.Tensor,
        msg_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse messages into state via gated attention.

        Args:
            state: (state_dim,) or (1, state_dim) agent state.
            msg_embeddings: (N, msg_dim) message embeddings.

        Returns:
            Augmented state: same shape as input state.
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Project state to message space
        state_proj = self.state_proj(state).unsqueeze(1)  # (1, 1, msg_dim)
        msgs = msg_embeddings.unsqueeze(0)  # (1, N, msg_dim)

        # Cross-attention: state attends to messages
        attended, _ = self.attention(
            query=state_proj,
            key=msgs,
            value=msgs,
        )  # (1, 1, msg_dim)
        attended = attended.squeeze(1)  # (1, msg_dim)

        # Gated residual fusion
        gate_val = self.gate(attended)  # (1, state_dim)
        msg_value = self.value_proj(attended)  # (1, state_dim)
        augmented = state + gate_val * msg_value

        return augmented.squeeze(0)


class CommChannel:
    """Communication channel for a single agent.

    Each agent gets one CommChannel that manages its inbox/outbox.
    """

    def __init__(
        self,
        agent_id: str,
        config: Optional[CommConfig] = None,
    ) -> None:
        self.agent_id = agent_id
        self.config = config or CommConfig()
        self._inbox: List[AgentMessage] = []
        self._outbox: List[AgentMessage] = []
        self._encoder = MessageEncoder(self.config.msg_dim)
        self._fusion = MessageFusion(
            self.config.state_dim,
            self.config.msg_dim,
            self.config.attention_heads,
        )
        self._sent_count: int = 0
        self._received_count: int = 0

    def send(
        self,
        receiver: str,
        msg_type: MessageType,
        content: Dict[str, Any],
        priority: int = 1,
        ttl: Optional[float] = None,
    ) -> AgentMessage:
        """Create and queue a message for sending.

        Args:
            receiver: Target agent ID or "*" for broadcast.
            msg_type: Type of message.
            content: Message payload.
            priority: Priority (0-3).
            ttl: Time-to-live in seconds.

        Returns:
            The created AgentMessage.
        """
        msg = AgentMessage(
            sender=self.agent_id,
            receiver=receiver,
            msg_type=msg_type,
            content=content,
            priority=min(3, max(0, priority)),
            ttl=ttl or self.config.ttl_default,
        )
        self._outbox.append(msg)
        self._sent_count += 1
        return msg

    def receive(self, message: AgentMessage) -> None:
        """Receive a message into the inbox."""
        if message.is_expired:
            return
        # Priority insertion (higher priority first)
        inserted = False
        for i, existing in enumerate(self._inbox):
            if message.priority > existing.priority:
                self._inbox.insert(i, message)
                inserted = True
                break
        if not inserted:
            self._inbox.append(message)
        # Enforce queue size
        while len(self._inbox) > self.config.max_queue_size:
            self._inbox.pop()  # Drop lowest priority
        self._received_count += 1

    def get_messages(
        self,
        msg_type: Optional[MessageType] = None,
        min_priority: int = 0,
    ) -> List[AgentMessage]:
        """Get non-expired messages from inbox, optionally filtered.

        Args:
            msg_type: Filter by message type, or None for all.
            min_priority: Minimum priority threshold.

        Returns:
            List of matching messages.
        """
        # Purge expired messages
        self._inbox = [m for m in self._inbox if not m.is_expired]

        msgs = self._inbox
        if msg_type is not None:
            msgs = [m for m in msgs if m.msg_type == msg_type]
        if min_priority > 0:
            msgs = [m for m in msgs if m.priority >= min_priority]
        return msgs

    def flush_outbox(self) -> List[AgentMessage]:
        """Flush and return all outgoing messages."""
        msgs = self._outbox[:]
        self._outbox.clear()
        return msgs

    def fuse_messages_into_state(
        self,
        state_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Augment agent state with incoming message context.

        Args:
            state_tensor: (state_dim,) agent state.

        Returns:
            Augmented state tensor of same shape.
        """
        messages = self.get_messages()
        if not messages:
            return state_tensor

        # Encode messages
        with torch.no_grad():
            msg_embeddings = self._encoder(messages)

        # Fuse into state
        return self._fusion(state_tensor, msg_embeddings)

    def clear(self) -> None:
        """Clear inbox and outbox."""
        self._inbox.clear()
        self._outbox.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Return channel statistics."""
        return {
            "agent_id": self.agent_id,
            "inbox_size": len(self._inbox),
            "outbox_size": len(self._outbox),
            "sent_total": self._sent_count,
            "received_total": self._received_count,
        }


class MessageBus:
    """Central message bus that routes messages between agents.

    Usage::

        bus = MessageBus()
        bus.register_agent("RedAgent")
        bus.register_agent("ScoutAgent")

        # Agent sends
        bus.channels["RedAgent"].send("ScoutAgent", MessageType.REQUEST, {"scan": "full"})

        # Bus routes
        bus.route_messages()

        # Agent receives
        msgs = bus.channels["ScoutAgent"].get_messages()
    """

    def __init__(self, config: Optional[CommConfig] = None) -> None:
        self.config = config or CommConfig()
        self.channels: Dict[str, CommChannel] = {}
        self._total_routed: int = 0

    def register_agent(self, agent_id: str) -> CommChannel:
        """Register an agent and create its communication channel."""
        if agent_id not in self.channels:
            channel = CommChannel(agent_id, self.config)
            self.channels[agent_id] = channel
            logger.debug(f"Registered comm channel for '{agent_id}'")
        return self.channels[agent_id]

    def route_messages(self) -> int:
        """Route all pending messages from outboxes to inboxes.

        Returns:
            Number of messages routed.
        """
        count = 0
        for sender_id, channel in self.channels.items():
            outgoing = channel.flush_outbox()
            for msg in outgoing:
                if msg.receiver == "*" and self.config.broadcast_enabled:
                    # Broadcast to all other agents
                    for recv_id, recv_channel in self.channels.items():
                        if recv_id != sender_id:
                            recv_channel.receive(msg)
                            count += 1
                elif msg.receiver in self.channels:
                    self.channels[msg.receiver].receive(msg)
                    count += 1
                else:
                    logger.debug(
                        f"Message from {sender_id} to unknown agent '{msg.receiver}'"
                    )
        self._total_routed += count
        return count

    def broadcast(
        self,
        sender_id: str,
        msg_type: MessageType,
        content: Dict[str, Any],
        priority: int = 1,
    ) -> None:
        """Send a broadcast message from one agent to all others."""
        if sender_id in self.channels:
            self.channels[sender_id].send("*", msg_type, content, priority)

    def get_stats(self) -> Dict[str, Any]:
        """Return bus statistics."""
        return {
            "n_agents": len(self.channels),
            "agent_ids": list(self.channels.keys()),
            "total_routed": self._total_routed,
            "channel_stats": {
                aid: ch.get_stats() for aid, ch in self.channels.items()
            },
        }

    def reset(self) -> None:
        """Clear all channels."""
        for channel in self.channels.values():
            channel.clear()
