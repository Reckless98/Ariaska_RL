#!/usr/bin/env python3
"""
core/telemetry/learning_signal_exporter.py — Phase 11.0: Learning Signal Exporter

Exports RL-ready JSONL records from UnifiedStepTrace data for offline analysis,
replay, and training signal debugging.

Each record contains:
  - State description (phase, discoveries, step)
  - Action taken (command, source, template)
  - Outcome (reward, discoveries found, phase advance)
  - Context (parse explanations, budget state, teaching points)

Output: logs/learning_signals_<run_id>.jsonl

Author: Filip Volf — Phase 11.0
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.telemetry.unified_trace import UnifiedStepTrace

logger = logging.getLogger("ariaska.learning_signal_exporter")


class LearningSignalExporter:
    """
    Exports structured learning signals from UnifiedStepTrace to JSONL.

    Each line is a self-contained record with everything needed to understand
    why a step produced its reward, what was discovered, and what the agent
    could have done differently.

    Usage:
        exporter = LearningSignalExporter(run_id="run_001")
        exporter.start_episode(episode_id=1)

        # After each step:
        exporter.record_step(trace)

        # End of episode:
        exporter.end_episode(summary_dict)

        # End of run:
        exporter.close()
    """

    def __init__(
        self,
        run_id: str = "",
        output_dir: str = "logs",
        enabled: bool = True,
    ):
        self._run_id = run_id or f"run_{int(time.time())}"
        self._output_dir = Path(output_dir)
        self._enabled = enabled
        self._file = None
        self._episode_id: int = 0
        self._step_count: int = 0
        self._episode_records: int = 0
        self._total_records: int = 0

        if self._enabled:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            filepath = self._output_dir / f"learning_signals_{self._run_id}.jsonl"
            try:
                self._file = open(filepath, "a", encoding="utf-8")
                logger.info(f"[LEARNING-EXPORT] Writing to {filepath}")
            except OSError as e:
                logger.warning(f"[LEARNING-EXPORT] Cannot open {filepath}: {e}")
                self._enabled = False

    def start_episode(self, episode_id: int) -> None:
        """Mark the start of a new episode."""
        self._episode_id = episode_id
        self._episode_records = 0
        self._write_record({
            "type": "episode_start",
            "ts": _now_iso(),
            "run_id": self._run_id,
            "episode_id": episode_id,
        })

    def record_step(self, trace: "UnifiedStepTrace") -> None:
        """Export a learning signal record from a UnifiedStepTrace."""
        if not self._enabled:
            return

        record = {
            "type": "step",
            "ts": trace.ts or _now_iso(),
            "run_id": self._run_id,
            "episode_id": trace.episode_id,
            "step": trace.step,
            "agent": trace.agent_name,
            # State
            "phase": trace.phase_state.current_phase,
            "steps_in_phase": trace.phase_state.steps_in_phase,
            # Action
            "command": trace.command,
            "template": trace.template_name,
            "source": trace.source,
            "confidence": round(trace.confidence, 3),
            # Outcome
            "reward": round(trace.reward_total, 3),
            "reward_breakdown": {k: round(v, 3) for k, v in trace.reward_breakdown.items()},
            "discovery_count": trace.discovery_count,
            "discoveries": trace.discoveries,
            # Parse
            "parse_mode": trace.parse_mode,
            "parse_stage": trace.parse_stage_reached,
            "parse_latency_ms": round(trace.parse_latency_ms, 2),
            "parse_explanations": [e.to_dict() for e in trace.parse_explanations],
            # Budget
            "budget_pressure": trace.budget_snapshot.budget_pressure,
            "mentor_remaining": trace.budget_snapshot.mentor_budget_remaining,
            # Teaching
            "teaching_points": trace.teaching_points,
            # Tool
            "tool_privilege": trace.tool_privilege.to_dict(),
        }

        self._write_record(record)
        self._episode_records += 1
        self._step_count += 1

    def end_episode(self, summary: Optional[Dict[str, Any]] = None) -> None:
        """Mark end of episode with optional summary."""
        record = {
            "type": "episode_end",
            "ts": _now_iso(),
            "run_id": self._run_id,
            "episode_id": self._episode_id,
            "steps_recorded": self._episode_records,
        }
        if summary:
            record["summary"] = summary
        self._write_record(record)

    def close(self) -> None:
        """Close the output file."""
        if self._file:
            try:
                self._file.close()
            except OSError:
                pass
            self._file = None
        logger.info(
            f"[LEARNING-EXPORT] Closed. Total records: {self._total_records}, "
            f"steps: {self._step_count}"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get exporter statistics."""
        return {
            "enabled": self._enabled,
            "total_records": self._total_records,
            "total_steps": self._step_count,
            "current_episode": self._episode_id,
            "episode_records": self._episode_records,
        }

    def _write_record(self, record: Dict[str, Any]) -> None:
        """Write a single JSON record to the output file."""
        if not self._enabled or not self._file:
            return
        try:
            self._file.write(json.dumps(record, default=str) + "\n")
            self._file.flush()
            self._total_records += 1
        except (OSError, ValueError) as e:
            logger.debug(f"[LEARNING-EXPORT] Write error: {e}")


def _now_iso() -> str:
    """ISO 8601 timestamp."""
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
