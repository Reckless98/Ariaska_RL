#!/usr/bin/env python3
"""
core/execution/step_parse_cache.py — Phase 9.5: Per-step parse result cache

Ensures SmartOutputParser.parse() is called exactly once per (episode, step,
agent, output_hash) combination. Dashboard, watchdog, and other consumers
read cached ParseResult instead of re-parsing (which could trigger nano-LLM).

Controlled by FF_SINGLE_PARSE_CACHE feature flag.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("ariaska.step_parse_cache")


@dataclass
class ParseCacheStats:
    """Metrics for parse cache effectiveness."""
    hits: int = 0
    misses: int = 0
    total_calls: int = 0

    @property
    def hit_rate(self) -> float:
        return self.hits / max(1, self.total_calls)


class StepParseCache:
    """Per-step cache for parsed discovery results.
    
    Key: (episode_id, step_idx, agent_id, output_hash)
    Value: Dict[str, Any] — parsed discoveries
    
    Lifecycle:
    - Created once per SmartOrchestrator
    - Cleared at start of each episode
    - Queried before any parse call; populated on miss
    """

    def __init__(self):
        self._cache: Dict[Tuple[int, int, str, str], Dict[str, Any]] = {}
        self.stats = ParseCacheStats()

    def _make_key(
        self,
        episode_id: int,
        step_idx: int,
        agent_id: str,
        output: str,
    ) -> Tuple[int, int, str, str]:
        """Create cache key from step context + output hash."""
        output_hash = hashlib.md5(output.encode("utf-8", errors="replace")).hexdigest()[:12]
        return (episode_id, step_idx, agent_id, output_hash)

    def get(
        self,
        episode_id: int,
        step_idx: int,
        agent_id: str,
        output: str,
    ) -> Optional[Dict[str, Any]]:
        """Get cached parse result, or None on miss."""
        self.stats.total_calls += 1
        key = self._make_key(episode_id, step_idx, agent_id, output)
        result = self._cache.get(key)
        if result is not None:
            self.stats.hits += 1
            logger.debug(
                f"[PARSE-CACHE] HIT ep={episode_id} step={step_idx} "
                f"agent={agent_id} (avoided re-parse)"
            )
        else:
            self.stats.misses += 1
        return result

    def put(
        self,
        episode_id: int,
        step_idx: int,
        agent_id: str,
        output: str,
        result: Dict[str, Any],
    ) -> None:
        """Store parse result in cache."""
        key = self._make_key(episode_id, step_idx, agent_id, output)
        self._cache[key] = result

    def reset_episode(self) -> None:
        """Clear cache at episode start."""
        self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Return cache stats for telemetry."""
        return {
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "total_calls": self.stats.total_calls,
            "hit_rate": round(self.stats.hit_rate, 3),
            "cache_size": len(self._cache),
        }
