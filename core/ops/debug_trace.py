"""
core/ops/debug_trace.py — Phase 39.4: Debug Instrumentation

Structured JSONL debug telemetry for OPS subsystems. Captures:
  - stall_score time series
  - phase transitions
  - evidence store size increments
  - trust weights and mentor prior deltas
  - top-3 alternative actions and why rejected

All entries are written as one JSON object per line with a run_id field.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ariaska.ops.debug_trace")


@dataclass
class DebugTraceEntry:
    """Single structured debug event."""

    run_id: str = ""
    event_type: str = ""
    step: int = 0
    episode: int = 0
    timestamp: float = field(default_factory=time.time)
    phase: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    def to_jsonl(self) -> str:
        """Serialize to compact JSON (one line)."""
        d = {
            "run_id": self.run_id,
            "event": self.event_type,
            "step": self.step,
            "episode": self.episode,
            "ts": round(self.timestamp, 3),
            "phase": self.phase,
        }
        d.update(self.data)
        return json.dumps(d, separators=(",", ":"), default=str)


class DebugTracer:
    """
    Structured JSONL debug logger for OPS subsystems.

    Writes one JSON object per line to a rotating log file.
    All entries include run_id for correlation.

    Usage:
        tracer = DebugTracer(log_dir="logs/debug", run_id="r80a")
        tracer.log_stall(step=10, stall_score=0.65, signals={...})
        tracer.log_phase_transition(step=12, from_phase="RECON", to_phase="ENUM")
        tracer.log_trust_update(step=15, source="gpt", trust=0.72, delta=-0.08)
        tracer.close()
    """

    MAX_FILE_SIZE_MB: int = 25

    def __init__(
        self,
        log_dir: str = "logs/debug",
        run_id: str = "default",
        enabled: bool = True,
    ) -> None:
        self._log_dir = Path(log_dir)
        self._run_id = run_id
        self._enabled = enabled
        self._file = None
        self._file_path: Optional[Path] = None
        self._entries_written: int = 0

        if self._enabled:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            self._open_file()

    def _open_file(self) -> None:
        """Open a new JSONL file for this run."""
        from datetime import datetime

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._file_path = self._log_dir / f"debug_{self._run_id}_{ts}.jsonl"
        try:
            self._file = open(self._file_path, "a", buffering=1)
        except Exception as e:
            logger.warning("Failed to open debug trace file: %s", e)
            self._file = None

    def _write(self, entry: DebugTraceEntry) -> None:
        """Write a single entry to the log file."""
        if not self._enabled or self._file is None:
            return

        try:
            self._file.write(entry.to_jsonl() + "\n")
            self._entries_written += 1

            # Auto-rotate
            if self._entries_written % 1000 == 0:
                try:
                    size = self._file_path.stat().st_size if self._file_path else 0
                    if size > self.MAX_FILE_SIZE_MB * 1024 * 1024:
                        self._file.close()
                        self._open_file()
                except Exception:
                    pass
        except Exception as e:
            logger.debug("Debug trace write failed: %s", e)

    def log_stall(
        self,
        step: int,
        stall_score: float,
        signals: Optional[Dict[str, float]] = None,
        episode: int = 0,
        phase: str = "",
    ) -> None:
        """Log stall score and component signals."""
        self._write(DebugTraceEntry(
            run_id=self._run_id,
            event_type="stall",
            step=step,
            episode=episode,
            phase=phase,
            data={
                "stall_score": round(stall_score, 4),
                "signals": signals or {},
            },
        ))

    def log_phase_transition(
        self,
        step: int,
        from_phase: str,
        to_phase: str,
        episode: int = 0,
        reason: str = "",
    ) -> None:
        """Log a phase transition."""
        self._write(DebugTraceEntry(
            run_id=self._run_id,
            event_type="phase_transition",
            step=step,
            episode=episode,
            phase=to_phase,
            data={
                "from": from_phase,
                "to": to_phase,
                "reason": reason,
            },
        ))

    def log_evidence_update(
        self,
        step: int,
        evidence_count: int,
        delta: int = 0,
        episode: int = 0,
        phase: str = "",
    ) -> None:
        """Log evidence store size change."""
        self._write(DebugTraceEntry(
            run_id=self._run_id,
            event_type="evidence_update",
            step=step,
            episode=episode,
            phase=phase,
            data={
                "evidence_count": evidence_count,
                "delta": delta,
            },
        ))

    def log_trust_update(
        self,
        step: int,
        source: str,
        trust: float,
        delta: float = 0.0,
        event: str = "",
        episode: int = 0,
    ) -> None:
        """Log trust weight change for a mentor source."""
        self._write(DebugTraceEntry(
            run_id=self._run_id,
            event_type="trust_update",
            step=step,
            episode=episode,
            data={
                "source": source,
                "trust": round(trust, 4),
                "delta": round(delta, 4),
                "trust_event": event,
            },
        ))

    def log_prior_injection(
        self,
        step: int,
        source: str,
        magnitude: float,
        trust: float,
        changed_action: bool = False,
        original_action: int = -1,
        influenced_action: int = -1,
        episode: int = 0,
    ) -> None:
        """Log mentor prior injection into policy."""
        self._write(DebugTraceEntry(
            run_id=self._run_id,
            event_type="prior_injection",
            step=step,
            episode=episode,
            data={
                "source": source,
                "magnitude": round(magnitude, 4),
                "trust": round(trust, 4),
                "changed_action": changed_action,
                "original": original_action,
                "influenced": influenced_action,
            },
        ))

    def log_alternatives(
        self,
        step: int,
        chosen: str,
        alternatives: List[Dict[str, Any]],
        episode: int = 0,
        phase: str = "",
    ) -> None:
        """Log top-3 alternative actions and why rejected."""
        self._write(DebugTraceEntry(
            run_id=self._run_id,
            event_type="alternatives",
            step=step,
            episode=episode,
            phase=phase,
            data={
                "chosen": chosen[:80],
                "alternatives": alternatives[:3],
            },
        ))

    def log_rethink(
        self,
        step: int,
        stall_score: float,
        plan_summary: Dict[str, Any],
        episode: int = 0,
        phase: str = "",
    ) -> None:
        """Log Orion rethink invocation and plan."""
        self._write(DebugTraceEntry(
            run_id=self._run_id,
            event_type="orion_rethink",
            step=step,
            episode=episode,
            phase=phase,
            data={
                "stall_score": round(stall_score, 4),
                "plan": plan_summary,
            },
        ))

    def log_decision(
        self,
        step: int,
        agent: str,
        command: str,
        source: str,
        confidence: float,
        episode: int = 0,
        phase: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a decision with full context."""
        data = {
            "agent": agent,
            "command": command[:100],
            "source": source,
            "confidence": round(confidence, 3),
        }
        if extra:
            data.update(extra)
        self._write(DebugTraceEntry(
            run_id=self._run_id,
            event_type="decision",
            step=step,
            episode=episode,
            phase=phase,
            data=data,
        ))

    def log_cap_eval(
        self,
        scenario: str,
        success_rate: float,
        runs: int,
        avg_steps: float,
        failures: List[Dict[str, str]],
        git_sha: str = "",
    ) -> None:
        """Log CAP evaluation result."""
        self._write(DebugTraceEntry(
            run_id=self._run_id,
            event_type="cap_eval",
            data={
                "scenario": scenario,
                "success_rate": round(success_rate, 3),
                "runs": runs,
                "avg_steps": round(avg_steps, 1),
                "failures": failures[:10],
                "git_sha": git_sha,
            },
        ))

    @property
    def entries_written(self) -> int:
        return self._entries_written

    @property
    def run_id(self) -> str:
        return self._run_id

    def close(self) -> None:
        """Close the log file."""
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None

    def __del__(self) -> None:
        self.close()
