"""core/algorithms/hindsight_replay.py — Phase 41: Hindsight Experience Replay.

HER wrapper around existing ReplayBuffer.  Relabels failed episodes
so the agent learns from near-misses (e.g. reached EXPLOITATION but
targeted EXFILTRATION → relabel goal to EXPLOITATION → positive reward).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ariaska.algorithms.hindsight_replay")

# Canonical phase ordering (index = rank)
PHASE_ORDER: List[str] = [
    "RECON", "ENUMERATION", "EXPLOITATION", "PRIVILEGE_ESCALATION",
    "LATERAL_MOVEMENT", "POST_EXPLOITATION", "EXFILTRATION", "CLOSEOUT",
]
_PHASE_RANK: Dict[str, int] = {p: i for i, p in enumerate(PHASE_ORDER)}


@dataclass
class HERConfig:
    """Hindsight Experience Replay configuration."""
    enabled: bool = True
    k_future: int = 4
    strategy: str = "future"
    relabel_reward_success: float = 10.0
    relabel_reward_fail: float = -1.0


class HindsightReplay:
    """Generates hindsight-relabeled transitions for the replay buffer.

    After each episode, ``process_episode`` creates synthetic transitions
    where the goal is relabeled to a phase the agent *actually* achieved,
    turning failures into learning signal.
    """

    def __init__(self, config: Optional[HERConfig] = None) -> None:
        self.config = config or HERConfig()
        self.total_relabeled: int = 0

    # ── public API ─────────────────────────────────────────────
    def relabel_episode(
        self,
        transitions: List[Dict[str, Any]],
        achieved_phase: str,
        target_phase: str,
    ) -> List[Dict[str, Any]]:
        """Relabel *transitions* with *achieved_phase* as new goal.

        Returns new transitions with recomputed rewards.  Original list
        is not mutated.
        """
        if not transitions:
            return []
        achieved_rank = _PHASE_RANK.get(achieved_phase, -1)
        target_rank = _PHASE_RANK.get(target_phase, -1)
        if achieved_rank < 0 or target_rank < 0:
            return []

        relabeled: List[Dict[str, Any]] = []
        for t in transitions:
            new_t = dict(t)
            new_t["original_goal"] = target_phase
            new_t["relabeled_goal"] = achieved_phase
            t_phase_rank = _PHASE_RANK.get(
                str(t.get("phase", "RECON")), 0
            )
            if t_phase_rank >= achieved_rank:
                new_t["reward"] = self.config.relabel_reward_success
            else:
                new_t["reward"] = self.config.relabel_reward_fail
            new_t["is_her"] = True
            relabeled.append(new_t)
        return relabeled

    def process_episode(
        self,
        episode_transitions: List[Dict[str, Any]],
        target_phase: str,
        achieved_phase: str,
    ) -> int:
        """Generate HER transitions and return count added.

        Uses the ``future`` strategy: for each transition, relabel
        with up to ``k_future`` future-achieved phases.
        """
        if not self.config.enabled or not episode_transitions:
            return 0

        achieved_rank = _PHASE_RANK.get(achieved_phase, 0)
        target_rank = _PHASE_RANK.get(target_phase, 0)

        if achieved_rank >= target_rank:
            return 0  # Already succeeded, no relabeling needed

        count = 0
        for k in range(min(self.config.k_future, achieved_rank + 1)):
            alt_phase = PHASE_ORDER[max(0, achieved_rank - k)]
            relabeled = self.relabel_episode(
                episode_transitions, alt_phase, target_phase
            )
            count += len(relabeled)

        self.total_relabeled += count
        logger.debug(
            "HER: relabeled %d transitions (achieved=%s, target=%s)",
            count, achieved_phase, target_phase,
        )
        return count
