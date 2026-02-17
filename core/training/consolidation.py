#!/usr/bin/env python3
"""
core/training/consolidation.py — Phase 15.0: Consolidation Engine ("Sleep Replay")

End-of-episode consolidation that prioritizes high-signal experiences:
  - Samples with high |DA| (strong reward prediction error)
  - Samples with high ACh (active hypothesis testing, confirmed evidence)
  - Confirmed hypotheses

Consolidation replays selected samples into:
  - BCBuffer (for future BC loss)
  - SkillLibrary (promote confirmed chains)
  - Hypothesis template updates (bounded)

Bounded: max 64 samples per consolidation batch.
Feature-flag gated: FF_CONSOLIDATION.

Author: Phase 15.0 Contract
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ariaska.training.consolidation")

_MAX_BATCH_SIZE = 64
_MAX_TRACE_CHARS = 256


# ── Schemas ─────────────────────────────────────────────────────────────────

@dataclass
class ConsolidationSample:
    """A single sample selected for consolidation replay."""
    step: int = 0
    command: str = ""
    reward: float = 0.0
    da_level: float = 0.5
    ach_level: float = 0.4
    source: str = ""              # "ppo", "mentor", "hypothesis", etc.
    hypothesis_confirmed: bool = False
    trace_summary: str = ""       # bounded to 256 chars
    state_hash: str = ""          # for dedup

    def priority_score(self) -> float:
        """Compute consolidation priority score."""
        da_score = abs(self.da_level - 0.5) * 2.0  # distance from baseline
        ach_score = self.ach_level
        confirm_bonus = 1.0 if self.hypothesis_confirmed else 0.0
        return 0.4 * da_score + 0.3 * ach_score + 0.3 * confirm_bonus

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConsolidationBatch:
    """A bounded batch of samples for consolidation."""
    samples: List[ConsolidationSample] = field(default_factory=list)
    episode_id: str = ""
    token_budget: int = 50000     # max tokens for this consolidation
    tokens_used: int = 0

    @property
    def size(self) -> int:
        return len(self.samples)

    def is_full(self) -> bool:
        return len(self.samples) >= _MAX_BATCH_SIZE

    def is_budget_exceeded(self) -> bool:
        return self.tokens_used >= self.token_budget

    def add(self, sample: ConsolidationSample) -> bool:
        """Add a sample if not full and not duplicate. Returns True if added."""
        if self.is_full():
            return False

        # Dedup by state_hash
        existing = {s.state_hash for s in self.samples if s.state_hash}
        if sample.state_hash and sample.state_hash in existing:
            return False

        self.samples.append(sample)
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "size": self.size,
            "token_budget": self.token_budget,
            "tokens_used": self.tokens_used,
            "samples": [s.to_dict() for s in self.samples],
        }


@dataclass
class ConsolidationMetrics:
    """Metrics from a consolidation run."""
    samples_considered: int = 0
    samples_selected: int = 0
    samples_deduped: int = 0
    bc_buffer_pushes: int = 0
    skill_promotions: int = 0
    hypothesis_updates: int = 0
    tokens_used: int = 0
    budget_exceeded: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Engine ──────────────────────────────────────────────────────────────────

class ConsolidationEngine:
    """
    End-of-episode consolidation ("sleep replay").

    1. Build ConsolidationBatch from episode data
    2. Run consolidation: replay into BCBuffer, SkillLibrary, Hypothesis templates
    3. Export metrics
    """

    def __init__(self, token_budget: int = 50000) -> None:
        self._token_budget = token_budget

    def build_batch(
        self,
        episode_samples: List[ConsolidationSample],
        episode_id: str = "",
    ) -> ConsolidationBatch:
        """
        Build a ConsolidationBatch from raw episode samples.

        Selects top samples by priority_score, bounded to MAX_BATCH_SIZE.
        Deduplicates by state_hash.
        """
        batch = ConsolidationBatch(
            episode_id=episode_id,
            token_budget=self._token_budget,
        )

        # Score and sort by priority (descending)
        scored = sorted(
            episode_samples,
            key=lambda s: s.priority_score(),
            reverse=True,
        )

        seen_hashes: set = set()
        for sample in scored:
            if batch.is_full():
                break
            # Dedup
            if sample.state_hash:
                if sample.state_hash in seen_hashes:
                    continue
                seen_hashes.add(sample.state_hash)
            # Bound trace summary
            sample.trace_summary = sample.trace_summary[:_MAX_TRACE_CHARS]
            batch.add(sample)

        return batch

    def run(
        self,
        batch: ConsolidationBatch,
        bc_buffer: Any = None,
        skill_library: Any = None,
    ) -> ConsolidationMetrics:
        """
        Execute consolidation replay.

        Pushes high-priority samples into:
          - bc_buffer (if provided) via store()
          - skill_library (if provided) via promote()
          - hypothesis template updates (logged only, actual update by caller)

        Budget-gated: stops if tokens_used exceeds budget.
        """
        metrics = ConsolidationMetrics(
            samples_considered=batch.size,
        )

        for sample in batch.samples:
            if batch.is_budget_exceeded():
                metrics.budget_exceeded = True
                break

            # Replay into BC buffer
            if bc_buffer is not None and sample.source in ("mentor", "hypothesis"):
                try:
                    bc_buffer.store(
                        state=None,  # caller should provide actual state tensor
                        teacher_action=0,
                        weight=sample.priority_score(),
                        rationale_hash=hash(sample.command),
                    )
                    metrics.bc_buffer_pushes += 1
                except Exception as e:
                    logger.debug(f"BC buffer push failed: {e}")

            # Promote to skill library
            if skill_library is not None and sample.hypothesis_confirmed:
                try:
                    # Use skill_library interface if available
                    if hasattr(skill_library, "promote"):
                        skill_library.promote(sample.command, sample.reward)
                        metrics.skill_promotions += 1
                except Exception as e:
                    logger.debug(f"Skill promotion failed: {e}")

            # Track hypothesis template updates
            if sample.hypothesis_confirmed:
                metrics.hypothesis_updates += 1

            metrics.samples_selected += 1

            # Estimate token cost (rough: 50 tokens per sample processing)
            estimated_tokens = 50
            batch.tokens_used += estimated_tokens
            metrics.tokens_used += estimated_tokens

        return metrics

    @staticmethod
    def compute_state_hash(command: str, phase: str, step: int) -> str:
        """Create a state hash for dedup."""
        data = f"{command}:{phase}:{step}".encode("utf-8", errors="replace")
        return hashlib.sha256(data).hexdigest()[:12]
