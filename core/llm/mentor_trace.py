"""
core/llm/mentor_trace.py — Phase 30: MentorTrace for Artificial Fine-Tuning.

Extends TeacherTrace with structured mentor→apprentice transfer metadata.
Each mentor call produces a MentorTrace containing:
  - The mentor's decision rationale + confidence
  - Structured feature extraction for episode-level learning
  - Summary embedding seed for state encoder Section 16

This is the "GPT as Artificial Fine-Tuning" bridge: mentor knowledge
gets distilled into PPO policy via BCBuffer + episodic summaries.

Usage:
    trace = MentorTrace.from_mentor_response(
        command="nmap -sV 10.0.0.1",
        reasoning="Port scan to discover services",
        confidence=0.85,
        phase="RECON",
        state_vector=state_tensor.tolist(),
    )
    bc_sample = trace.to_teacher_trace().to_bc_sample()
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ariaska.mentor_trace")


@dataclass
class MentorTrace:
    """
    Structured mentor→apprentice knowledge trace.

    One MentorTrace per mentor call. Contains:
      - Decision: what the mentor chose + why
      - Context: phase, step, discoveries at time of call
      - Quality signals: confidence, alternatives, expected outcome
      - Embedding seed: key features for episode summary

    Consumed by:
      - BCBuffer (via to_teacher_trace) for behavioral cloning
      - EpisodeSummary for embedding in state encoder Section 16
    """
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    # ── Mentor Decision ──────────────────────────────────────────
    command: str = ""
    template_name: str = ""
    reasoning: str = ""         # Full GPT rationale (≤512 chars)
    confidence: float = 0.5
    alternatives: List[str] = field(default_factory=list)  # max 3
    expected_outcome: str = ""  # What mentor expects (≤256 chars)

    # ── Context at call time ─────────────────────────────────────
    phase: str = ""
    step: int = 0
    episode: int = 0
    agent_id: str = ""
    discoveries_at_call: int = 0    # How many discoveries existed
    stagnation_steps: int = 0       # Steps without new discovery

    # ── State vector (for BC) ────────────────────────────────────
    state_vector: Optional[List[float]] = None  # 512-dim

    # ── Quality tracking (set post-execution) ────────────────────
    actual_reward: float = 0.0
    produced_discovery: bool = False
    mentor_was_correct: Optional[bool] = None

    # ── Timing ───────────────────────────────────────────────────
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if len(self.reasoning) > 512:
            self.reasoning = self.reasoning[:512]
        if len(self.expected_outcome) > 256:
            self.expected_outcome = self.expected_outcome[:256]
        if len(self.alternatives) > 3:
            self.alternatives = self.alternatives[:3]

    def to_teacher_trace(self) -> Any:
        """Convert to TeacherTrace for BCBuffer storage."""
        from core.reasoning.teacher_trace import TeacherTrace
        return TeacherTrace(
            trace_id=self.trace_id,
            state_vector=self.state_vector,
            teacher_command=self.command,
            teacher_template=self.template_name,
            rationale=self.reasoning,
            expected_obs=self.expected_outcome,
            alt_actions=self.alternatives,
            confidence=self.confidence,
            episode=self.episode,
            step=self.step,
            agent_id=self.agent_id,
            phase=self.phase,
        )

    def to_summary_features(self) -> Dict[str, float]:
        """Extract 16 scalar features for episode summary embedding.

        These feed into state encoder Section 16 (dims 221-236).
        """
        return {
            "mentor_confidence": self.confidence,
            "mentor_stagnation": min(1.0, self.stagnation_steps / 20.0),
            "mentor_discovery_density": min(1.0, self.discoveries_at_call / 30.0),
            "mentor_was_correct": 1.0 if self.mentor_was_correct else 0.0,
            "mentor_produced_discovery": 1.0 if self.produced_discovery else 0.0,
            "mentor_reward_signal": max(-1.0, min(1.0, self.actual_reward / 50.0)),
            "mentor_phase_recon": 1.0 if self.phase in ("RECON", "ENUMERATION") else 0.0,
            "mentor_phase_exploit": 1.0 if self.phase in ("EXPLOITATION", "PRIVILEGE_ESCALATION") else 0.0,
            "mentor_phase_post": 1.0 if self.phase in ("POST_EXPLOITATION", "EXFILTRATION", "CLOSEOUT") else 0.0,
            "mentor_step_early": 1.0 if self.step < 10 else 0.0,
            "mentor_step_mid": 1.0 if 10 <= self.step < 25 else 0.0,
            "mentor_step_late": 1.0 if self.step >= 25 else 0.0,
            "mentor_has_alternatives": 1.0 if len(self.alternatives) > 0 else 0.0,
            "mentor_high_confidence": 1.0 if self.confidence >= 0.8 else 0.0,
            "mentor_low_confidence": 1.0 if self.confidence < 0.3 else 0.0,
            "mentor_call_count_norm": 0.0,  # Set by EpisodeSummary
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSONL logging."""
        return {
            "trace_id": self.trace_id,
            "command": self.command,
            "template_name": self.template_name,
            "reasoning": self.reasoning,
            "confidence": round(self.confidence, 3),
            "alternatives": self.alternatives,
            "expected_outcome": self.expected_outcome,
            "phase": self.phase,
            "step": self.step,
            "episode": self.episode,
            "agent_id": self.agent_id,
            "discoveries_at_call": self.discoveries_at_call,
            "stagnation_steps": self.stagnation_steps,
            "actual_reward": round(self.actual_reward, 3),
            "produced_discovery": self.produced_discovery,
            "mentor_was_correct": self.mentor_was_correct,
            "timestamp": self.timestamp,
        }


@dataclass
class EpisodeSummary:
    """
    Aggregated mentor traces for one episode.

    Computes a 16-dim summary vector used by state encoder Section 16
    to help PPO understand "what kind of episode is happening" —
    are we in heavy-mentor mode? Are mentors accurate? Is the agent
    stagnating and relying on GPT?
    """
    episode_id: int = 0
    traces: List[MentorTrace] = field(default_factory=list)

    def add_trace(self, trace: MentorTrace) -> None:
        """Add a mentor trace to the summary."""
        self.traces.append(trace)

    def compute_embedding(self) -> List[float]:
        """Compute 16-dim episode summary for state encoder.

        Returns:
            List of 16 floats, each in [0, 1].
        """
        if not self.traces:
            return [0.0] * 16

        n = len(self.traces)
        # Aggregate features from all traces
        features = [t.to_summary_features() for t in self.traces]

        # Average each feature
        keys = list(features[0].keys())
        embedding = []
        for key in keys:
            vals = [f[key] for f in features]
            avg = sum(vals) / max(1, len(vals))
            embedding.append(avg)

        # Override "mentor_call_count_norm" with actual normalized count
        call_count_idx = keys.index("mentor_call_count_norm")
        embedding[call_count_idx] = min(1.0, n / 20.0)  # 20 calls = saturated

        return embedding[:16]  # Safety: always 16 dims

    @property
    def mentor_accuracy(self) -> float:
        """Fraction of mentor calls that produced discoveries."""
        if not self.traces:
            return 0.0
        correct = sum(1 for t in self.traces if t.produced_discovery)
        return correct / len(self.traces)

    @property
    def avg_confidence(self) -> float:
        """Average mentor confidence."""
        if not self.traces:
            return 0.0
        return sum(t.confidence for t in self.traces) / len(self.traces)
