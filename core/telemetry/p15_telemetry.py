#!/usr/bin/env python3
"""
core/telemetry/p15_telemetry.py — Phase 15.0: Neurovortex Telemetry

Collects and exports Phase 15 metrics from SmartCoach instances:
- Neuromodulator levels (DA, NE, ACh, 5-HT)
- Aggression levels per agent
- Working memory occupancy
- Sensory buffer sizes
- Semantic index stats
- Consolidation metrics
- BudgetManagerV2 tier utilization
- Reflex/arbitrator activation counts

All bounded. Feature-flag gated: only collects when P15 components are active.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ariaska.telemetry.p15")


@dataclass
class P15AgentSnapshot:
    """Per-agent P15 telemetry snapshot."""
    agent_name: str = ""
    neuromod_da: float = 0.5
    neuromod_ne: float = 0.3
    neuromod_ach: float = 0.4
    neuromod_sht: float = 0.5
    aggression_level: float = 0.3
    working_memory_slots: int = 0
    working_memory_max: int = 8
    sensory_buffer_size: int = 0
    semantic_index_entries: int = 0
    consolidation_samples: int = 0
    reflex_triggers: int = 0
    arbitrator_uses: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class P15EpisodeMetrics:
    """Aggregated P15 metrics for an episode."""
    episode_id: str = ""
    agent_snapshots: List[P15AgentSnapshot] = field(default_factory=list)
    budget_stats: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "agents": [s.to_dict() for s in self.agent_snapshots],
            "budget": self.budget_stats,
        }


def collect_agent_snapshot(coach: Any, agent_name: str = "") -> P15AgentSnapshot:
    """
    Collect a P15 telemetry snapshot from a SmartCoach instance.

    Safe to call even if P15 components are not initialized (returns defaults).
    """
    snap = P15AgentSnapshot(agent_name=agent_name or getattr(coach, 'agent_name', ''))

    # Neuromodulator state
    nm_state = getattr(coach, '_p15_neuromod_state', None)
    if nm_state is not None:
        snap.neuromod_da = getattr(nm_state, 'da', 0.5)
        snap.neuromod_ne = getattr(nm_state, 'ne', 0.3)
        snap.neuromod_ach = getattr(nm_state, 'ach', 0.4)
        snap.neuromod_sht = getattr(nm_state, 'sht', 0.5)

    # Aggression
    snap.aggression_level = getattr(coach, '_p15_aggression_level', 0.3)

    # Working memory
    wm = getattr(coach, '_p15_working_memory', None)
    if wm is not None:
        snap.working_memory_slots = len(wm)
        snap.working_memory_max = getattr(wm, '_max_slots', 8)

    # Sensory buffer
    sb = getattr(coach, '_p15_sensory_buffer', None)
    if sb is not None:
        snap.sensory_buffer_size = len(sb)

    # Semantic index
    si = getattr(coach, '_p15_semantic_index', None)
    if si is not None:
        snap.semantic_index_entries = len(si)

    # Consolidation samples
    cs = getattr(coach, '_p15_consolidation_samples', [])
    snap.consolidation_samples = len(cs)

    return snap


def collect_episode_metrics(
    coaches: Dict[str, Any],
    gpt_manager: Any = None,
    episode_id: str = "",
) -> P15EpisodeMetrics:
    """
    Collect P15 metrics across all coaches for one episode.

    Args:
        coaches: Dict of coach_name → SmartCoach instances
        gpt_manager: GPTManager instance (for BudgetManagerV2 stats)
        episode_id: Episode identifier

    Returns:
        P15EpisodeMetrics with per-agent snapshots and budget stats.
    """
    metrics = P15EpisodeMetrics(episode_id=episode_id)

    for name, coach in coaches.items():
        try:
            snap = collect_agent_snapshot(coach, agent_name=name)
            metrics.agent_snapshots.append(snap)
        except Exception as e:
            logger.debug(f"[P15] Telemetry collection failed for {name}: {e}")

    # Budget stats
    if gpt_manager is not None:
        bm2 = getattr(gpt_manager, '_budget_manager_v2', None)
        if bm2 is not None:
            try:
                metrics.budget_stats = bm2.get_stats()
            except Exception:
                pass

    return metrics
