#!/usr/bin/env python3
"""
core/reasoning/lesson.py — Phase 14.0: Compact Teaching Points

Contract C1.6: Lessons are compact, bounded teaching artifacts extracted
after meaningful steps (discovery, exploit, hypothesis test).

Fed into SkillLibrary, CampaignMemory, and BC training buffer.

Author: Phase 14.0 Contract C1.6
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("ariaska.lesson")

try:
    import torch
except ImportError:
    torch = None  # type: ignore


@dataclass
class Lesson:
    """
    Compact teaching point extracted after a meaningful step.

    Contract C1.6: All string fields are bounded. The learning_vector
    is exactly (32,) — no unbounded prompt growth.
    """
    lesson_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    context: str = ""              # ≤256 chars situation description
    action_taken: str = ""         # CommandTemplate name
    outcome: str = ""              # ≤128 chars what happened
    insight: str = ""              # ≤256 chars WHY (from mentor/parser)
    service_archetype: str = ""    # ServiceArchetype enum value
    phase: str = ""                # Kill chain phase
    confidence: float = 0.5
    source: str = "parser"         # mentor | parser | hypothesis_test | campaign
    applicable_when: Dict[str, Any] = field(default_factory=dict)
    reward_signal: float = 0.0

    def __post_init__(self) -> None:
        if len(self.context) > 256:
            self.context = self.context[:256]
        if len(self.outcome) > 128:
            self.outcome = self.outcome[:128]
        if len(self.insight) > 256:
            self.insight = self.insight[:256]

    def to_skill_card(self) -> Dict[str, Any]:
        """Convert to SkillCard v2 compatible dict."""
        return {
            "id": f"lesson_{self.lesson_id}",
            "if_condition": self.context,
            "then_action": self.action_taken,
            "confidence": self.confidence,
            "preconditions": self.applicable_when,
            "postconditions": {"outcome": self.outcome},
            "service_archetypes": [self.service_archetype] if self.service_archetype else [],
            "applicable_phases": [self.phase] if self.phase else [],
            "reward_history": [self.reward_signal] if self.reward_signal else [],
        }

    def to_training_pair(self) -> Tuple[str, str]:
        """Return (situation, action) pair for BC-style training."""
        return (self.context, self.action_taken)

    def to_learning_vector(self, max_dims: int = 32) -> Any:
        """
        Produce a bounded (32,) tensor embedding of this lesson.

        Contract Rule 5: No unbounded prompt growth. Fixed-size vector.
        Uses hash-based embedding (no sentence-transformer dependency).
        """
        if torch is None:
            return None

        vec = torch.zeros(max_dims, dtype=torch.float32)

        # Hash-based features from lesson content
        text = f"{self.context}|{self.action_taken}|{self.outcome}"
        h = hash(text) & 0xFFFFFFFF

        # Spread hash across first 8 dims
        for i in range(8):
            vec[i] = ((h >> (i * 4)) & 0xF) / 15.0 - 0.5

        # Confidence and reward signal
        vec[8] = self.confidence
        vec[9] = min(1.0, max(-1.0, self.reward_signal / 50.0))  # Normalized

        # Phase encoding (one-hot in dims 10-16)
        phases = ["RECON", "ENUMERATION", "EXPLOITATION", "PRIVILEGE_ESCALATION",
                  "LATERAL_MOVEMENT", "POST_EXPLOITATION", "EXFILTRATION"]
        if self.phase in phases:
            vec[10 + phases.index(self.phase)] = 1.0

        # Source encoding (one-hot in dims 17-20)
        sources = ["mentor", "parser", "hypothesis_test", "campaign"]
        if self.source in sources:
            vec[17 + sources.index(self.source)] = 1.0

        # Service archetype hash in dims 21-24
        if self.service_archetype:
            sa_hash = hash(self.service_archetype) & 0xFFFF
            for i in range(4):
                vec[21 + i] = ((sa_hash >> (i * 4)) & 0xF) / 15.0 - 0.5

        # Precondition features in dims 25-31
        n_preconds = min(7, len(self.applicable_when))
        vec[25] = n_preconds / 7.0
        for i, (k, v) in enumerate(list(self.applicable_when.items())[:6]):
            vec[26 + i] = 1.0 if v else 0.0

        return vec

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Lesson":
        valid = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in valid})


class LessonExtractor:
    """
    Extracts Lesson objects from step execution results.

    Sources:
    - Parser discoveries + InterpretationLessons
    - Hypothesis test results (CONFIRMED/REFUTED)
    - Mentor suggestions (rationale)
    """

    def __init__(self) -> None:
        self._episode_lessons: List[Lesson] = []

    def extract_from_step(
        self,
        command: str,
        output: str,
        discoveries: List[Any],
        hypotheses: Optional[List[Any]] = None,
        mentor_rationale: Optional[str] = None,
        phase: str = "",
        reward: float = 0.0,
        service_archetype: str = "",
    ) -> List[Lesson]:
        """
        Extract lessons from a single step's results.

        Returns list of Lesson objects (typically 0-3 per step).
        """
        lessons: List[Lesson] = []

        # Lesson from discoveries
        if discoveries:
            discovery_types = set()
            for d in discoveries:
                if hasattr(d, "discovery_type"):
                    dt = d.discovery_type
                    if hasattr(dt, "value"):
                        dt = dt.value
                    discovery_types.add(str(dt))

            if discovery_types:
                lessons.append(Lesson(
                    context=f"Command {command[:80]} in phase {phase}",
                    action_taken=command[:128],
                    outcome=f"Discovered: {', '.join(list(discovery_types)[:5])}",
                    insight=f"Found {len(discoveries)} items",
                    service_archetype=service_archetype,
                    phase=phase,
                    confidence=0.8,
                    source="parser",
                    reward_signal=reward,
                ))

        # Lesson from hypothesis results
        if hypotheses:
            for h in hypotheses:
                if hasattr(h, "status") and hasattr(h, "result_summary"):
                    status_val = h.status
                    if hasattr(status_val, "value"):
                        status_val = status_val.value
                    if status_val in ("confirmed", "refuted"):
                        lessons.append(Lesson(
                            context=f"Hypothesis test: {getattr(h, 'if_observed', '')[:80]}",
                            action_taken=getattr(h, "then_try", ""),
                            outcome=f"Hypothesis {status_val}",
                            insight=getattr(h, "result_summary", "") or "",
                            phase=phase,
                            confidence=0.9 if status_val == "confirmed" else 0.7,
                            source="hypothesis_test",
                            reward_signal=reward,
                        ))

        # Lesson from mentor rationale
        if mentor_rationale:
            lessons.append(Lesson(
                context=f"Mentor guidance in phase {phase}",
                action_taken=command[:128],
                outcome="Mentor-directed action",
                insight=mentor_rationale[:256],
                phase=phase,
                confidence=0.6,
                source="mentor",
                reward_signal=reward,
            ))

        self._episode_lessons.extend(lessons)
        return lessons

    def get_episode_lessons(self) -> List[Lesson]:
        """Return all lessons extracted this episode."""
        return list(self._episode_lessons)

    def get_high_confidence_lessons(self, min_confidence: float = 0.7) -> List[Lesson]:
        """Return lessons with confidence above threshold."""
        return [l for l in self._episode_lessons if l.confidence >= min_confidence]

    def reset(self) -> None:
        """Reset for new episode."""
        self._episode_lessons.clear()
