#!/usr/bin/env python3
"""
core/execution/parser_teacher_output.py — Phase 14.0: Parser-as-Teacher Output

Contract C1.7: Structured output from ParserBroker when FF_PARSER_TEACHER=True.
Wraps DiscoveryEvents + LearningFeatures + ParsingLessons so the parser
can teach the RL agent about output interpretation patterns.

Author: Phase 14.0 Contract C1.7
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ariaska.parser_teacher_output")

try:
    import torch
except ImportError:
    torch = None  # type: ignore


@dataclass
class ParsingLesson:
    """
    A single parsing lesson learned from output interpretation.

    Compact: pattern + discovery_type + confidence + bounded example.
    """
    pattern: str
    discovery_type: str
    confidence: float
    example_output: str = ""

    def __post_init__(self) -> None:
        # Bound example_output to ≤128 chars
        if len(self.example_output) > 128:
            self.example_output = self.example_output[:128]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern": self.pattern,
            "discovery_type": self.discovery_type,
            "confidence": self.confidence,
            "example_output": self.example_output,
        }


@dataclass
class LearningFeatures:
    """
    Aggregated learning features from a single parser invocation.

    Contract: feature_vector() always returns exactly (32,) tensor.
    """
    discovery_counts: Dict[str, int] = field(default_factory=dict)
    confidence_mean: float = 0.0
    confidence_min: float = 0.0
    stage_reached: int = 0  # 1=regex, 2=SOP, 3=Venice, 4=GPT
    novel_discoveries: int = 0
    repeated_discoveries: int = 0
    total_events: int = 0
    parser_latency_ms: float = 0.0

    def feature_vector(self, max_dims: int = 32) -> Any:
        """
        Encode learning features as fixed-size tensor.

        Contract: ALWAYS returns exactly (max_dims,) shape.
        """
        if torch is None:
            return [0.0] * max_dims

        vec = torch.zeros(max_dims, dtype=torch.float32)

        # Discovery type counts (dims 0-11) — up to 12 discovery types
        dt_order = [
            "open_port", "service", "version_info", "credential",
            "vulnerability", "cve", "shell", "root_shell",
            "web_path", "flag", "hash", "user"
        ]
        for i, dt in enumerate(dt_order):
            if i < 12:
                vec[i] = min(10.0, self.discovery_counts.get(dt, 0)) / 10.0

        # Confidence stats (dims 12-13)
        vec[12] = self.confidence_mean
        vec[13] = self.confidence_min

        # Stage reached (dims 14-17) — one-hot
        if 1 <= self.stage_reached <= 4:
            vec[13 + self.stage_reached] = 1.0

        # Novel vs repeated (dims 18-19)
        vec[18] = min(1.0, self.novel_discoveries / max(1, self.total_events))
        vec[19] = min(1.0, self.repeated_discoveries / max(1, self.total_events))

        # Total events normalized (dim 20)
        vec[20] = min(1.0, self.total_events / 20.0)

        # Parser latency normalized (dim 21) — cap at 5000ms
        vec[21] = min(1.0, self.parser_latency_ms / 5000.0)

        # Dims 22-31 reserved (zero-padded)

        return vec

    def to_dict(self) -> Dict[str, Any]:
        return {
            "discovery_counts": self.discovery_counts,
            "confidence_mean": self.confidence_mean,
            "confidence_min": self.confidence_min,
            "stage_reached": self.stage_reached,
            "novel_discoveries": self.novel_discoveries,
            "repeated_discoveries": self.repeated_discoveries,
            "total_events": self.total_events,
            "parser_latency_ms": self.parser_latency_ms,
        }


@dataclass
class ParserTeacherOutput:
    """
    Full parser output when FF_PARSER_TEACHER=True.

    Wraps standard DiscoveryEvents with additional LearningFeatures
    and ParsingLessons for RL training signal injection.

    Contract C1.7: max 3 lessons, max 3 confidence_disagreements.
    """
    events: List[Any] = field(default_factory=list)  # List[DiscoveryEvent]
    learning_features: LearningFeatures = field(default_factory=LearningFeatures)
    lessons: List[ParsingLesson] = field(default_factory=list)
    confidence_disagreements: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Enforce max 3 lessons
        if len(self.lessons) > 3:
            self.lessons = self.lessons[:3]
        # Enforce max 3 confidence disagreements
        if len(self.confidence_disagreements) > 3:
            self.confidence_disagreements = self.confidence_disagreements[:3]

    @staticmethod
    def empty() -> ParserTeacherOutput:
        """Create an empty ParserTeacherOutput (no discoveries)."""
        return ParserTeacherOutput(
            events=[],
            learning_features=LearningFeatures(),
            lessons=[],
            confidence_disagreements=[],
        )

    def has_discoveries(self) -> bool:
        """Check if any events were found."""
        return len(self.events) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "events_count": len(self.events),
            "learning_features": self.learning_features.to_dict(),
            "lessons": [l.to_dict() for l in self.lessons],
            "confidence_disagreements": self.confidence_disagreements[:3],
        }
