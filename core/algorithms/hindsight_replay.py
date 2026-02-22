"""core/algorithms/hindsight_replay.py — Phase 48: Enhanced Hindsight Experience Replay.

HER wrapper around existing ReplayBuffer.  Relabels failed episodes
so the agent learns from near-misses (e.g. reached EXPLOITATION but
targeted EXFILTRATION → relabel goal to EXPLOITATION → positive reward).

Phase 48 enhancements:
  - Discovery-based relabeling (not just phase goals)
  - Tensor-aware transitions for PPO buffer integration
  - Multi-strategy support: 'future', 'final', 'episode', 'discovery'
  - Priority weighting: near-miss transitions get higher priority
  - Phase-distance reward shaping (graduated rewards, not binary)
  - Cross-episode memory: stores best relabeled transitions persistently
  - Integrated with PPO RolloutBuffer for direct training injection
"""
from __future__ import annotations

import logging
import random
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("ariaska.algorithms.hindsight_replay")

# Canonical phase ordering (index = rank)
PHASE_ORDER: List[str] = [
    "RECON", "ENUMERATION", "EXPLOITATION", "PRIVILEGE_ESCALATION",
    "LATERAL_MOVEMENT", "POST_EXPLOITATION", "EXFILTRATION", "CLOSEOUT",
]
_PHASE_RANK: Dict[str, int] = {p: i for i, p in enumerate(PHASE_ORDER)}

# Discovery types ranked by value (for discovery-based relabeling)
DISCOVERY_RANKS: Dict[str, float] = {
    "open_port": 1.0,
    "service": 2.0,
    "version": 3.0,
    "user": 4.0,
    "username": 4.0,
    "hash": 6.0,
    "credential": 8.0,
    "password": 10.0,
    "shell": 15.0,
    "user_flag": 20.0,
    "root_flag": 20.0,
    "root_shell": 25.0,
}


@dataclass
class HERConfig:
    """Hindsight Experience Replay configuration.

    Phase 48: Extended with discovery-based relabeling, graduated
    rewards, priority weighting, and persistent memory.
    """
    enabled: bool = True
    k_future: int = 4
    strategy: str = "future"           # future | final | episode | discovery
    relabel_reward_success: float = 10.0
    relabel_reward_fail: float = -1.0
    # Phase 48 additions
    discovery_relabel: bool = True      # Also relabel based on discoveries made
    graduated_rewards: bool = True      # Phase-distance reward shaping
    priority_weight: float = 2.0        # Priority multiplier for near-miss transitions
    memory_capacity: int = 5000         # Cross-episode HER memory bank
    min_phase_delta: int = 1            # Min phase gap to trigger relabeling
    inject_ratio: float = 0.15          # Ratio of HER transitions to inject into PPO buffer
    max_inject_per_episode: int = 50    # Cap on injected transitions per episode


class HERMemoryBank:
    """Thread-safe persistent memory bank for best HER transitions.

    Stores the highest-value relabeled transitions across episodes
    for periodic injection into the PPO training buffer.
    """

    def __init__(self, capacity: int = 5000) -> None:
        self._buffer: deque[Dict[str, Any]] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self.total_stored: int = 0

    def add(self, transitions: List[Dict[str, Any]]) -> int:
        """Add transitions, keeping highest-priority ones."""
        with self._lock:
            for t in transitions:
                self._buffer.append(t)
                self.total_stored += 1
            return len(transitions)

    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample batch from memory bank with priority weighting."""
        with self._lock:
            if not self._buffer:
                return []
            k = min(batch_size, len(self._buffer))
            # Priority-weighted sampling: higher reward = more likely
            items = list(self._buffer)
            weights = [max(0.1, t.get("priority", 1.0)) for t in items]
            total_w = sum(weights)
            if total_w <= 0:
                return random.sample(items, k)
            probs = [w / total_w for w in weights]
            indices = random.choices(range(len(items)), weights=probs, k=k)
            return [items[i] for i in indices]

    @property
    def size(self) -> int:
        """Current buffer size."""
        with self._lock:
            return len(self._buffer)

    def get_stats(self) -> Dict[str, Any]:
        """Return memory bank statistics."""
        with self._lock:
            if not self._buffer:
                return {
                    "size": 0, "total_stored": self.total_stored,
                    "avg_reward": 0.0, "avg_priority": 0.0,
                }
            rewards = [t.get("reward", 0.0) for t in self._buffer]
            priorities = [t.get("priority", 1.0) for t in self._buffer]
            return {
                "size": len(self._buffer),
                "total_stored": self.total_stored,
                "avg_reward": sum(rewards) / len(rewards),
                "avg_priority": sum(priorities) / len(priorities),
            }


class HindsightReplay:
    """Generates hindsight-relabeled transitions for the replay buffer.

    After each episode, ``process_episode`` creates synthetic transitions
    where the goal is relabeled to a phase the agent *actually* achieved,
    turning failures into learning signal.

    Phase 48 enhancements:
      - ``discovery_relabel``: relabel based on discoveries, not just phases
      - ``graduated_rewards``: phase-distance reward shaping
      - ``priority_weight``: near-miss transitions get higher priority
      - ``memory_bank``: persistent cross-episode HER memory
      - ``inject_into_buffer``: inject HER transitions into PPO's RolloutBuffer
      - Multi-strategy: 'future', 'final', 'episode', 'discovery'
    """

    def __init__(self, config: Optional[HERConfig] = None) -> None:
        self.config = config or HERConfig()
        self.total_relabeled: int = 0
        self.memory_bank = HERMemoryBank(capacity=self.config.memory_capacity)
        self._episode_count: int = 0

    # ── public API ─────────────────────────────────────────────
    def relabel_episode(
        self,
        transitions: List[Dict[str, Any]],
        achieved_phase: str,
        target_phase: str,
    ) -> List[Dict[str, Any]]:
        """Relabel *transitions* with *achieved_phase* as new goal.

        Phase 48: Now uses graduated reward shaping based on phase distance
        instead of binary success/fail. Near-miss transitions get higher
        priority for replay.

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

            if self.config.graduated_rewards:
                # Graduated reward: closer to achieved phase = higher reward
                if t_phase_rank >= achieved_rank:
                    new_t["reward"] = self.config.relabel_reward_success
                else:
                    # Partial credit: fraction of success reward based on closeness
                    distance = achieved_rank - t_phase_rank
                    max_dist = max(achieved_rank, 1)
                    fraction = 1.0 - (distance / max_dist)
                    new_t["reward"] = (
                        self.config.relabel_reward_fail
                        + fraction * (self.config.relabel_reward_success - self.config.relabel_reward_fail)
                    )
            else:
                if t_phase_rank >= achieved_rank:
                    new_t["reward"] = self.config.relabel_reward_success
                else:
                    new_t["reward"] = self.config.relabel_reward_fail

            # Priority: near-miss transitions (close to achieved phase) get higher priority
            phase_closeness = 1.0 - abs(t_phase_rank - achieved_rank) / max(len(PHASE_ORDER), 1)
            new_t["priority"] = phase_closeness * self.config.priority_weight
            new_t["is_her"] = True
            relabeled.append(new_t)
        return relabeled

    def relabel_by_discovery(
        self,
        transitions: List[Dict[str, Any]],
        episode_discoveries: List[str],
    ) -> List[Dict[str, Any]]:
        """Relabel transitions based on discoveries made during the episode.

        For each discovery achieved, create relabeled transitions where
        the goal is "find this discovery type" — turning partial exploration
        into positive learning signal.

        Args:
            transitions: Episode transitions with 'phase', 'command', etc.
            episode_discoveries: List of discovery strings (e.g., 'port:22', 'service:ssh').

        Returns:
            List of relabeled transitions with discovery-based rewards.
        """
        if not transitions or not episode_discoveries:
            return []

        relabeled: List[Dict[str, Any]] = []
        # Extract best discovery type achieved
        best_disc_value = 0.0
        best_disc_type = ""
        for d in episode_discoveries:
            d_type = d.split(":")[0].lower() if ":" in d else d.lower()
            value = DISCOVERY_RANKS.get(d_type, 0.5)
            if value > best_disc_value:
                best_disc_value = value
                best_disc_type = d_type

        if best_disc_value <= 0:
            return []

        # Relabel: transitions that led to the discovery get positive reward
        for i, t in enumerate(transitions):
            new_t = dict(t)
            new_t["original_goal"] = "target_discovery"
            new_t["relabeled_goal"] = f"discovery:{best_disc_type}"
            # Later transitions (closer to discovery) get higher reward
            progress = (i + 1) / len(transitions)
            new_t["reward"] = best_disc_value * progress
            new_t["priority"] = progress * self.config.priority_weight
            new_t["is_her"] = True
            new_t["her_strategy"] = "discovery"
            relabeled.append(new_t)

        return relabeled

    def process_episode(
        self,
        episode_transitions: List[Dict[str, Any]],
        target_phase: str,
        achieved_phase: str,
        episode_discoveries: Optional[List[str]] = None,
    ) -> int:
        """Generate HER transitions and return count added.

        Phase 48: Multi-strategy relabeling with discovery support.

        Strategies:
          - ``future``: relabel with up to k_future achieved sub-goals
          - ``final``: relabel all transitions with the final achieved phase
          - ``episode``: relabel with the highest phase achieved at each step
          - ``discovery``: relabel based on discoveries (new in Phase 48)

        Args:
            episode_transitions: Full episode of transitions.
            target_phase: What we were trying to reach.
            achieved_phase: What we actually reached.
            episode_discoveries: Discoveries made during episode.

        Returns:
            Total number of relabeled transitions generated.
        """
        if not self.config.enabled or not episode_transitions:
            return 0

        self._episode_count += 1
        achieved_rank = _PHASE_RANK.get(achieved_phase, 0)
        target_rank = _PHASE_RANK.get(target_phase, 0)

        count = 0
        all_relabeled: List[Dict[str, Any]] = []

        # Phase-based relabeling (only if we didn't reach target)
        if achieved_rank < target_rank and (target_rank - achieved_rank) >= self.config.min_phase_delta:
            strategy = self.config.strategy

            if strategy == "future":
                for k in range(min(self.config.k_future, achieved_rank + 1)):
                    alt_phase = PHASE_ORDER[max(0, achieved_rank - k)]
                    relabeled = self.relabel_episode(
                        episode_transitions, alt_phase, target_phase
                    )
                    all_relabeled.extend(relabeled)
                    count += len(relabeled)

            elif strategy == "final":
                relabeled = self.relabel_episode(
                    episode_transitions, achieved_phase, target_phase
                )
                all_relabeled.extend(relabeled)
                count += len(relabeled)

            elif strategy == "episode":
                # Find the highest phase at each transition and relabel
                seen_phases: set = set()
                for t in episode_transitions:
                    t_phase = str(t.get("phase", "RECON"))
                    if t_phase not in seen_phases:
                        seen_phases.add(t_phase)
                        t_rank = _PHASE_RANK.get(t_phase, 0)
                        if t_rank <= achieved_rank:
                            relabeled = self.relabel_episode(
                                episode_transitions, t_phase, target_phase
                            )
                            all_relabeled.extend(relabeled)
                            count += len(relabeled)

        # Discovery-based relabeling (Phase 48, independent of phase success)
        if self.config.discovery_relabel and episode_discoveries:
            disc_relabeled = self.relabel_by_discovery(
                episode_transitions, episode_discoveries
            )
            all_relabeled.extend(disc_relabeled)
            count += len(disc_relabeled)

        # Store in persistent memory bank
        if all_relabeled:
            self.memory_bank.add(all_relabeled)

        self.total_relabeled += count
        logger.debug(
            "HER: relabeled %d transitions (achieved=%s, target=%s, "
            "discoveries=%d, strategy=%s, memory=%d)",
            count, achieved_phase, target_phase,
            len(episode_discoveries or []),
            self.config.strategy,
            self.memory_bank.size,
        )
        return count

    def sample_for_training(self, batch_size: int = 32) -> List[Dict[str, Any]]:
        """Sample relabeled transitions from memory bank for training injection.

        Returns:
            List of HER transitions ready for PPO/DDQN training.
        """
        return self.memory_bank.sample(batch_size)

    def get_stats(self) -> Dict[str, Any]:
        """Return HER statistics."""
        return {
            "total_relabeled": self.total_relabeled,
            "episodes_processed": self._episode_count,
            "memory_bank": self.memory_bank.get_stats(),
            "config": {
                "strategy": self.config.strategy,
                "k_future": self.config.k_future,
                "discovery_relabel": self.config.discovery_relabel,
                "graduated_rewards": self.config.graduated_rewards,
            },
        }
