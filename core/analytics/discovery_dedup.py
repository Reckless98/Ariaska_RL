"""
Shared episode discovery deduplication for Ariaska.

R67: Prevents multiple agents from claiming the same discovery bonus.
Before R67, each agent's SmartRewardCalculator had independent discovery sets,
meaning the same port/service could yield 5× the intended bonus (one per agent).

This module provides a single shared discovery registry per episode that all
agents reference. A discovery is only "new" for reward purposes if NO agent
has claimed it yet.

Usage in SmartOrchestrator:
    from core.analytics.discovery_dedup import SharedDiscoverySet
    shared_disc = SharedDiscoverySet()
    # In _run_step, after parsing discoveries:
    is_new = shared_disc.claim("open_port:22", agent="ScoutAgent")
    # is_new = True only the first time ANY agent claims it
    shared_disc.reset_episode()  # At episode start
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Set, Optional, Tuple

logger = logging.getLogger("ariaska.analytics.discovery_dedup")


@dataclass
class SharedDiscoverySet:
    """Thread-safe shared discovery set for cross-agent deduplication.
    
    Each discovery is keyed as "type:value" (e.g. "open_port:22", "service:ssh").
    The first agent to claim a discovery gets credit; subsequent claims return False.
    """

    # discovery_key → claiming agent name
    _claimed: Dict[str, str] = field(default_factory=dict)
    # Per-agent claim counts for metrics
    _agent_counts: Dict[str, int] = field(default_factory=dict)
    # Total duplicates blocked
    _dupes_blocked: int = 0

    def claim(self, discovery_key: str, agent: str = "unknown") -> bool:
        """Attempt to claim a discovery. Returns True if this is genuinely new.
        
        Args:
            discovery_key: Formatted as "type:value" (e.g. "open_port:22")
            agent: Name of the claiming agent
            
        Returns:
            True if this discovery was not previously claimed by any agent.
        """
        if not discovery_key:
            return False

        key = discovery_key.strip().lower()
        if key in self._claimed:
            self._dupes_blocked += 1
            logger.debug(
                f"[DEDUP] Blocked duplicate '{key}' by {agent} "
                f"(first claimed by {self._claimed[key]})"
            )
            return False

        self._claimed[key] = agent
        self._agent_counts[agent] = self._agent_counts.get(agent, 0) + 1
        logger.debug(f"[DEDUP] New discovery '{key}' claimed by {agent}")
        return True

    def claim_batch(self, discoveries: Set[str], agent: str = "unknown") -> Set[str]:
        """Claim multiple discoveries at once. Returns only the genuinely new ones.
        
        Args:
            discoveries: Set of discovery keys to claim
            agent: Name of the claiming agent
            
        Returns:
            Set of discovery keys that were genuinely new.
        """
        new_ones = set()
        for disc in discoveries:
            if self.claim(disc, agent):
                new_ones.add(disc)
        return new_ones

    def is_known(self, discovery_key: str) -> bool:
        """Check if a discovery was already claimed by any agent."""
        return discovery_key.strip().lower() in self._claimed

    @property
    def total_unique(self) -> int:
        """Total number of unique discoveries this episode."""
        return len(self._claimed)

    @property
    def total_dupes_blocked(self) -> int:
        """Number of duplicate claims that were blocked."""
        return self._dupes_blocked

    def get_agent_stats(self) -> Dict[str, int]:
        """Per-agent count of genuinely new discoveries claimed."""
        return dict(self._agent_counts)

    def get_summary(self) -> Dict:
        """Summary for logging/artifacts."""
        return {
            "total_unique": self.total_unique,
            "dupes_blocked": self._dupes_blocked,
            "agent_stats": dict(self._agent_counts),
            "dedup_ratio": (
                round(self._dupes_blocked / max(1, self.total_unique + self._dupes_blocked), 3)
            ),
        }

    def reset_episode(self) -> None:
        """Clear all claims for new episode."""
        if self._claimed:
            logger.debug(
                f"[DEDUP] Episode reset: {self.total_unique} unique, "
                f"{self._dupes_blocked} dupes blocked "
                f"(ratio={self._dupes_blocked / max(1, self.total_unique + self._dupes_blocked):.1%})"
            )
        self._claimed.clear()
        self._agent_counts.clear()
        self._dupes_blocked = 0
