"""Pipeline Stages — decision pipeline stage helpers.

Phase 41: Shared constants and stage metadata for the decision pipeline.
"""
from __future__ import annotations

import time as _time
import logging
from typing import Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("ariaska.coach.pipeline_stages")


class PipelineStage(Enum):
    """Stages in the SmartCoach decision pipeline."""
    PLAYBOOK = "playbook"
    PPO = "ppo"
    REGISTRY = "registry"
    MENTOR = "mentor"
    DUAL_MENTOR = "dual_mentor"
    MICRO_CHAIN = "micro_chain"
    PHASE_GUIDED = "phase_guided"
    ANTI_REPEAT = "anti_repeat"
    EVIDENCE_GATE = "evidence_gate"
    TACTICAL_CORTEX = "tactical_cortex"
    FALLBACK = "fallback"


@dataclass
class StageTrace:
    """Record of a single pipeline stage execution."""
    stage: str
    detail: str = ""
    score: float = 0.0
    passed: bool = True
    timestamp_ms: float = 0.0


@dataclass
class PipelineResult:
    """Aggregated result of a full pipeline run."""
    traces: List[StageTrace] = field(default_factory=list)
    source: str = "unknown"
    final_command: str = ""
    confidence: float = 0.5

    def add_trace(
        self, stage: str, detail: str = "", score: float = 0.0, passed: bool = True
    ) -> None:
        """Add a stage trace entry."""
        self.traces.append(StageTrace(
            stage=stage, detail=detail, score=score,
            passed=passed, timestamp_ms=_time.time() * 1000,
        ))

    @property
    def rejected_stages(self) -> List[str]:
        """Return list of stages that rejected the command."""
        return [t.stage for t in self.traces if not t.passed]

    def to_dict(self) -> List[Dict[str, Any]]:
        """Convert traces to list of dicts."""
        return [
            {"stage": t.stage, "detail": t.detail, "score": t.score, "passed": t.passed}
            for t in self.traces
        ]
