#!/usr/bin/env python3
"""
core/tracing/event_bus.py — ARIASKA Event Bus v1.0

Lightweight event-driven architecture for decoupling training loop
from UI/logging/checkpoint subscribers.

StepEvent is the canonical aggregate payload emitted after each training step
(across all agents). Subscribers register callbacks to consume events.

Usage:
    bus = EventBus()
    bus.subscribe(my_dashboard.on_step)
    bus.subscribe(my_jsonl_logger.on_step)

    # In orchestrator after _run_step:
    bus.publish(step_event)
"""

import json
import time
import logging
from pathlib import Path
from typing import (
    Any, Callable, Dict, List, Optional, Set, Protocol, runtime_checkable,
)
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque

logger = logging.getLogger("ariaska.event_bus")


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

class EventKind(str, Enum):
    """Kinds of events the bus can carry."""
    STEP = "step"
    EPISODE_START = "episode_start"
    EPISODE_END = "episode_end"
    RUN_START = "run_start"
    RUN_END = "run_end"
    PHASE_TRANSITION = "phase_transition"
    MENTOR_CALL = "mentor_call"
    CHECKPOINT = "checkpoint"
    WARNING = "warning"
    ERROR = "error"
    TARGET_HEALTH = "target_health"


# ---------------------------------------------------------------------------
# Per-agent step record (carried inside StepEvent)
# ---------------------------------------------------------------------------

@dataclass
class AgentStepRecord:
    """One agent's contribution to a training step."""
    agent_name: str
    role: str
    decision_source: str          # "ppo", "playbook", "registry", "mentor", "anti_repeat", "skill", "fallback"
    phase: str                    # Phase name when agent acted
    command: str                  # Final command chosen
    command_family: str = ""      # Action family (e.g. "nmap", "hydra")
    reward: float = 0.0
    reward_breakdown: Optional[Dict[str, float]] = None
    mentor_call: bool = False
    mentor_model: Optional[str] = None       # "local-llm", "local-llm", "local-llm"
    mentor_tier: Optional[str] = None        # "reactive", "deliberative", "postmortem"
    exec_ms: float = 0.0          # Execution latency
    stdout_snippet: str = ""      # First 200 chars of output
    discoveries: List[str] = field(default_factory=list)   # New discoveries this step
    flags_set: List[str] = field(default_factory=list)     # Flags set this step
    tokens_used: int = 0
    confidence: float = 0.5
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict."""
        d = asdict(self)
        # Remove None values for compactness
        return {k: v for k, v in d.items() if v is not None}


# ---------------------------------------------------------------------------
# StepEvent — canonical per-step aggregate
# ---------------------------------------------------------------------------

@dataclass
class StepEvent:
    """
    Aggregate event emitted after ALL agents have processed one training step.

    This is the single source of truth for what happened in a step.
    All UI renderers, loggers, and dashboards consume this.
    """
    kind: EventKind = EventKind.STEP

    # Identifiers
    episode_id: str = ""
    episode_num: int = 0
    step_num: int = 0
    timestamp: float = field(default_factory=time.time)

    # Per-agent records
    agent_records: List[AgentStepRecord] = field(default_factory=list)

    # Phase state
    phase_before: str = ""
    phase_after: str = ""
    phase_progress: float = 0.0   # 0-1 within current phase

    # Discoveries delta (new this step, across all agents)
    new_discoveries: Dict[str, List[str]] = field(default_factory=dict)
    # e.g. {"ports": ["80", "443"], "credentials": ["msfadmin:msfadmin"]}

    # Aggregates
    step_reward_total: float = 0.0
    step_tokens_total: int = 0
    mentor_calls_total: int = 0

    # Episode running totals
    episode_reward_so_far: float = 0.0
    episode_steps_so_far: int = 0
    episode_mentor_calls_so_far: int = 0

    # Target info
    target_ip: str = ""
    mode: str = "sim"             # "sim" or "live"

    # Metadata
    wall_clock_ms: float = 0.0    # Wall-clock time for entire step

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict for JSONL logging."""
        d = {
            "kind": self.kind.value,
            "episode_id": self.episode_id,
            "episode_num": self.episode_num,
            "step_num": self.step_num,
            "timestamp": self.timestamp,
            "phase_before": self.phase_before,
            "phase_after": self.phase_after,
            "phase_progress": self.phase_progress,
            "new_discoveries": self.new_discoveries,
            "step_reward_total": self.step_reward_total,
            "step_tokens_total": self.step_tokens_total,
            "mentor_calls_total": self.mentor_calls_total,
            "episode_reward_so_far": self.episode_reward_so_far,
            "episode_steps_so_far": self.episode_steps_so_far,
            "episode_mentor_calls_so_far": self.episode_mentor_calls_so_far,
            "target_ip": self.target_ip,
            "mode": self.mode,
            "wall_clock_ms": self.wall_clock_ms,
            "agent_records": [r.to_dict() for r in self.agent_records],
        }
        return d


# ---------------------------------------------------------------------------
# Lightweight generic event (for non-step events)
# ---------------------------------------------------------------------------

@dataclass
class GenericEvent:
    """A non-step event (phase transition, checkpoint, warning, etc.)."""
    kind: EventKind
    timestamp: float = field(default_factory=time.time)
    episode_id: str = ""
    episode_num: int = 0
    step_num: int = 0
    data: Dict[str, Any] = field(default_factory=dict)
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "timestamp": self.timestamp,
            "episode_id": self.episode_id,
            "episode_num": self.episode_num,
            "step_num": self.step_num,
            "data": self.data,
            "message": self.message,
        }


# ---------------------------------------------------------------------------
# Subscriber protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class EventSubscriber(Protocol):
    """Protocol for event subscribers (duck-typing ok too)."""
    def on_event(self, event: Any) -> None: ...


# Type alias for callback-style subscribers
EventCallback = Callable[[Any], None]


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------

class EventBus:
    """
    Central event bus for Ariaska training loop.

    - Accepts callback or EventSubscriber subscribers
    - Maintains in-memory ring buffer (last N events)
    - Optional JSONL file sink for persistence with rotation
    - Synchronous dispatch (subscribers run inline)
    
    Phase 6.3: Added rotating JSONL sink (max_bytes, keep_rotated).
    """

    # Default rotation: 50 MB per file, keep 3 rotated files
    DEFAULT_MAX_BYTES: int = 50 * 1024 * 1024  # 50 MB
    DEFAULT_KEEP_ROTATED: int = 3

    def __init__(
        self,
        buffer_size: int = 500,
        jsonl_path: Optional[str] = None,
        max_jsonl_bytes: int = DEFAULT_MAX_BYTES,
        keep_rotated: int = DEFAULT_KEEP_ROTATED,
    ):
        self._subscribers: List[EventCallback] = []
        self._buffer: deque = deque(maxlen=buffer_size)
        self._jsonl_path = jsonl_path
        self._jsonl_fh = None
        self._event_count = 0
        self._max_jsonl_bytes = max_jsonl_bytes
        self._keep_rotated = keep_rotated
        self._jsonl_bytes_written = 0

        if jsonl_path:
            path = Path(jsonl_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._jsonl_fh = open(path, "a", encoding="utf-8")
            # Track current file size
            try:
                self._jsonl_bytes_written = path.stat().st_size
            except OSError:
                self._jsonl_bytes_written = 0
            logger.info(f"EventBus JSONL sink → {path} (rotation: {max_jsonl_bytes // (1024*1024)}MB)")

    # -- Subscription --

    def subscribe(self, callback: EventCallback) -> None:
        """Register a callback to receive events."""
        if callback not in self._subscribers:
            self._subscribers.append(callback)
            logger.debug(f"EventBus: +subscriber ({len(self._subscribers)} total)")

    def subscribe_object(self, subscriber: EventSubscriber) -> None:
        """Register an object implementing on_event()."""
        self.subscribe(subscriber.on_event)

    def unsubscribe(self, callback: EventCallback) -> None:
        """Remove a subscriber."""
        try:
            self._subscribers.remove(callback)
        except ValueError:
            pass

    # -- Publishing --

    def publish(self, event: Any) -> None:
        """
        Publish an event to all subscribers.

        Events are also buffered and optionally written to JSONL.
        """
        self._event_count += 1
        self._buffer.append(event)

        # JSONL sink with rotation
        if self._jsonl_fh is not None:
            try:
                if hasattr(event, "to_dict"):
                    line = json.dumps(event.to_dict(), default=str)
                else:
                    line = json.dumps({"raw": str(event)})
                encoded = (line + "\n").encode("utf-8")
                self._jsonl_bytes_written += len(encoded)
                self._jsonl_fh.write(line + "\n")
                self._jsonl_fh.flush()
                # Rotate if size exceeded
                if self._jsonl_bytes_written >= self._max_jsonl_bytes:
                    self._rotate_jsonl()
            except Exception as exc:
                logger.warning(f"EventBus JSONL write failed: {exc}")

        # Dispatch to subscribers
        for cb in self._subscribers:
            try:
                cb(event)
            except Exception as exc:
                logger.warning(f"EventBus subscriber error: {exc}")

    def publish_generic(
        self,
        kind: EventKind,
        message: str = "",
        data: Optional[Dict[str, Any]] = None,
        episode_id: str = "",
        episode_num: int = 0,
        step_num: int = 0,
    ) -> None:
        """Convenience: publish a GenericEvent."""
        evt = GenericEvent(
            kind=kind,
            message=message,
            data=data or {},
            episode_id=episode_id,
            episode_num=episode_num,
            step_num=step_num,
        )
        self.publish(evt)

    # -- Query --

    @property
    def event_count(self) -> int:
        return self._event_count

    def recent(self, n: int = 20, kind: Optional[EventKind] = None) -> List[Any]:
        """Get recent events from ring buffer, optionally filtered by kind."""
        events = list(self._buffer)
        if kind is not None:
            events = [e for e in events if getattr(e, "kind", None) == kind]
        return events[-n:]

    def recent_steps(self, n: int = 20) -> List[StepEvent]:
        """Get recent StepEvents."""
        return [e for e in list(self._buffer) if isinstance(e, StepEvent)][-n:]

    # -- Rotation --

    def _rotate_jsonl(self) -> None:
        """Rotate current JSONL file: close, rename with numbered suffix, open fresh."""
        if self._jsonl_fh is None or self._jsonl_path is None:
            return
        try:
            self._jsonl_fh.flush()
            self._jsonl_fh.close()
        except Exception:
            pass

        base = Path(self._jsonl_path)

        # Shift existing rotated files:  .2 → .3 (delete .3 if beyond keep)
        for i in range(self._keep_rotated, 0, -1):
            src = base.with_suffix(f"{base.suffix}.{i}")
            if i >= self._keep_rotated:
                src.unlink(missing_ok=True)
            elif src.exists():
                dst = base.with_suffix(f"{base.suffix}.{i + 1}")
                try:
                    src.rename(dst)
                except OSError:
                    pass

        # Current file → .1
        if base.exists():
            try:
                base.rename(base.with_suffix(f"{base.suffix}.1"))
            except OSError:
                pass

        # Open fresh file
        self._jsonl_fh = open(base, "a", encoding="utf-8")
        self._jsonl_bytes_written = 0
        logger.info(f"EventBus JSONL rotated → {base}")

    # -- Lifecycle --

    def close(self) -> None:
        """Flush and close JSONL sink."""
        if self._jsonl_fh is not None:
            try:
                self._jsonl_fh.flush()
                self._jsonl_fh.close()
            except Exception:
                pass
            self._jsonl_fh = None

    def __del__(self):
        self.close()

    def get_stats(self) -> Dict[str, Any]:
        """Get bus statistics."""
        return {
            "event_count": self._event_count,
            "subscriber_count": len(self._subscribers),
            "buffer_size": len(self._buffer),
            "jsonl_active": self._jsonl_fh is not None,
            "jsonl_bytes_written": self._jsonl_bytes_written,
            "jsonl_max_bytes": self._max_jsonl_bytes,
        }
