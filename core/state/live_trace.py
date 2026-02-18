#!/usr/bin/env python3
"""
core/state/live_trace.py — Phase 35: Append-Only JSONL Live Trace

Each step writes a single JSON line to:
    runs/<episode_id>/live_trace.jsonl

Contains: canonical_hash, coherence chain outputs, agent actions,
contradiction status, model/token/cost breakdown.

Resume capability: read last line's step_id to continue from.

Author: Phase 35
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.state.canonical_state import CanonicalState
    from core.state.coherence_chain import CoherenceChainResult

logger = logging.getLogger("ariaska.state.live_trace")

__all__ = ["LiveTraceWriter", "LiveTraceReader"]


class LiveTraceWriter:
    """Append-only JSONL writer for per-step engagement trace."""

    def __init__(self, base_dir: str = "runs") -> None:
        self._base_dir = base_dir
        self._fh: Optional[Any] = None
        self._episode_id: Optional[str] = None
        self._path: Optional[str] = None
        self._lines_written = 0

    def start_episode(self, episode_id: str) -> str:
        """Open trace file for a new episode. Returns the file path."""
        self.close()
        self._episode_id = str(episode_id)
        ep_dir = os.path.join(self._base_dir, self._episode_id)
        os.makedirs(ep_dir, exist_ok=True)
        self._path = os.path.join(ep_dir, "live_trace.jsonl")
        self._fh = open(self._path, "a", encoding="utf-8")
        self._lines_written = 0
        logger.debug(f"[TRACE] Opened {self._path}")
        return self._path

    def write_step(
        self,
        step: int,
        canonical_state: Optional["CanonicalState"] = None,
        coherence_result: Optional["CoherenceChainResult"] = None,
        agent_actions: Optional[List[Dict[str, Any]]] = None,
        reward: float = 0.0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write one JSONL line for a step."""
        if self._fh is None:
            return

        line: Dict[str, Any] = {
            "ts": time.time(),
            "episode_id": self._episode_id,
            "step": step,
        }

        if canonical_state is not None:
            line["canonical_hash"] = canonical_state.canonical_hash
            line["canonical_version"] = canonical_state.version
            line["phase"] = canonical_state.current_phase
            line["evidence"] = canonical_state.evidence_counts()

        if coherence_result is not None:
            line["coherence"] = {
                "classify_phase": coherence_result.classify.phase_guess,
                "classify_confidence": coherence_result.classify.phase_confidence,
                "contradiction_detected": coherence_result.contradiction.contradiction_detected,
                "contradiction_severity": coherence_result.contradiction.severity,
                "contradictions": coherence_result.contradiction.contradictions[:3],
                "coherence_score": coherence_result.score.coherence_score,
                "novelty_score": coherence_result.score.novelty_score,
                "repeat_risk": coherence_result.score.repeat_risk,
                "nano_tokens": coherence_result.nano_tokens,
                "mini_tokens": coherence_result.mini_tokens,
            }

        if agent_actions:
            line["actions"] = [
                {
                    "agent": a.get("agent", ""),
                    "command": a.get("command", "")[:120],
                    "source": a.get("source", ""),
                }
                for a in agent_actions[:5]
            ]

        line["reward"] = round(reward, 3)

        if extra:
            line["extra"] = extra

        try:
            self._fh.write(json.dumps(line, default=str) + "\n")
            self._fh.flush()
            self._lines_written += 1
        except Exception as e:
            logger.debug(f"[TRACE] Write error: {e}")

    def close(self) -> None:
        """Flush and close the trace file."""
        if self._fh is not None:
            try:
                self._fh.flush()
                self._fh.close()
            except Exception:
                pass
            self._fh = None
            logger.debug(f"[TRACE] Closed {self._path} ({self._lines_written} lines)")

    @property
    def path(self) -> Optional[str]:
        return self._path

    @property
    def lines_written(self) -> int:
        return self._lines_written

    def __del__(self) -> None:
        self.close()


class LiveTraceReader:
    """Read JSONL trace for resume capability."""

    @staticmethod
    def last_step(trace_path: str) -> int:
        """
        Read the last step_id from a trace file.
        Returns -1 if file doesn't exist or is empty.
        """
        if not os.path.exists(trace_path):
            return -1
        last_step = -1
        try:
            with open(trace_path, "r", encoding="utf-8") as f:
                for raw_line in f:
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    try:
                        obj = json.loads(raw_line)
                        step = obj.get("step", -1)
                        if isinstance(step, int) and step > last_step:
                            last_step = step
                    except json.JSONDecodeError:
                        pass
        except Exception:
            pass
        return last_step

    @staticmethod
    def load_all(trace_path: str) -> List[Dict[str, Any]]:
        """Load all trace lines as list of dicts."""
        if not os.path.exists(trace_path):
            return []
        entries: List[Dict[str, Any]] = []
        try:
            with open(trace_path, "r", encoding="utf-8") as f:
                for raw_line in f:
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    try:
                        entries.append(json.loads(raw_line))
                    except json.JSONDecodeError:
                        pass
        except Exception:
            pass
        return entries

    @staticmethod
    def contradiction_summary(trace_path: str) -> Dict[str, Any]:
        """Summarize contradiction events from a trace."""
        entries = LiveTraceReader.load_all(trace_path)
        total_steps = len(entries)
        desync_steps = 0
        all_contradictions: List[str] = []
        for e in entries:
            coh = e.get("coherence", {})
            if coh.get("contradiction_detected", False):
                desync_steps += 1
                all_contradictions.extend(coh.get("contradictions", []))
        return {
            "total_steps": total_steps,
            "desync_steps": desync_steps,
            "desync_rate": round(desync_steps / max(1, total_steps), 3),
            "unique_contradictions": list(set(all_contradictions)),
        }
