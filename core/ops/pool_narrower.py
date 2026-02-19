"""
core/ops/pool_narrower.py - Phase 40.2: Adaptive Command Pool Narrower

Dynamically narrows the candidate command pool based on:
- Target OS detection (Linux vs Windows vs any)
- Service relevance (only commands relevant to discovered services)
- Historical success rate (deprioritize commands that keep failing)
- Phase appropriateness (score by phase fit)
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("ariaska.pool_narrower")

# Default minimum weight floor to avoid completely excluding commands
_MIN_WEIGHT = 0.05
# After this many failures with 0 successes, deprioritize becomes aggressive
_FAIL_THRESHOLD = 5
# Decay factor for success rate smoothing (exponential moving average)
_EMA_ALPHA = 0.3


@dataclass
class TemplateStats:
    """Tracks per-template success/failure stats."""
    template_name: str
    successes: int = 0
    attempts: int = 0
    total_reward: float = 0.0
    last_used_step: int = 0
    consecutive_failures: int = 0

    @property
    def success_rate(self) -> float:
        if self.attempts == 0:
            return 0.5  # Prior: unknown templates get benefit of doubt
        return self.successes / self.attempts

    @property
    def avg_reward(self) -> float:
        if self.attempts == 0:
            return 0.0
        return self.total_reward / self.attempts


@dataclass
class NarrowerConfig:
    """Configuration for CommandPoolNarrower."""
    min_weight: float = _MIN_WEIGHT
    fail_threshold: int = _FAIL_THRESHOLD
    ema_alpha: float = _EMA_ALPHA
    os_filter_enabled: bool = True
    service_filter_enabled: bool = True
    success_weighting_enabled: bool = True
    phase_bonus: float = 1.5  # Weight multiplier for phase-matched commands
    recency_bonus_steps: int = 50  # Commands not used in N steps get slight boost
    thread_safe: bool = True  # Phase 41: When False, skip locking (single-agent perf)


class _NoOpLock:
    """No-op context manager for single-threaded mode."""
    def __enter__(self) -> "_NoOpLock":
        return self
    def __exit__(self, *args: Any) -> None:
        pass


class CommandPoolNarrower:
    """Narrows command candidate pool based on OS, services, and success history."""

    def __init__(self, config: Optional[NarrowerConfig] = None):
        self._config = config or NarrowerConfig()
        self._template_stats: Dict[str, TemplateStats] = {}
        self._detected_os: Optional[str] = None  # "linux", "windows", None
        self._lock: Any = threading.Lock() if self._config.thread_safe else _NoOpLock()
        self._total_narrowed = 0
        self._total_input = 0
        logger.info("[POOL-NARROWER] Initialized")

    def _acquire(self) -> threading.Lock:
        """Return lock context manager (noop when thread_safe=False)."""
        return self._lock

    def set_target_os(self, os_type: str) -> None:
        """Set detected target OS ("linux", "windows", "unknown")."""
        normalized = os_type.lower().strip()
        with self._lock:
            if normalized in ("linux", "windows", "freebsd", "macos"):
                self._detected_os = normalized
            else:
                self._detected_os = None
        logger.info(f"[POOL-NARROWER] Target OS set: {self._detected_os}")

    def detect_os_from_output(self, output: str) -> Optional[str]:
        """Infer target OS from command output heuristics."""
        output_lower = output.lower()
        linux_signals = [
            "linux", "ubuntu", "debian", "centos", "fedora",
            "/etc/passwd", "/bin/bash", "/bin/sh", "gnu/linux",
            "openssh", "apache/2", "www-data", "root:",
        ]
        windows_signals = [
            "windows", "microsoft", "iis/", ".exe",
            "c:\\windows", "powershell", "cmd.exe", ".dll",
            "nt authority", "net user", "meterpreter",
        ]
        linux_score = sum(1 for s in linux_signals if s in output_lower)
        windows_score = sum(1 for s in windows_signals if s in output_lower)

        if linux_score > windows_score and linux_score >= 2:
            self.set_target_os("linux")  # set_target_os acquires lock
            return "linux"
        elif windows_score > linux_score and windows_score >= 2:
            self.set_target_os("windows")  # set_target_os acquires lock
            return "windows"
        return None

    def record_result(
        self,
        template_name: str,
        success: bool,
        reward: float = 0.0,
        step: int = 0,
    ) -> None:
        """Record result of a template execution."""
        with self._lock:
            if template_name not in self._template_stats:
                self._template_stats[template_name] = TemplateStats(
                    template_name=template_name
                )
            stats = self._template_stats[template_name]
            stats.attempts += 1
            stats.total_reward += reward
            stats.last_used_step = step
            if success:
                stats.successes += 1
                stats.consecutive_failures = 0
            else:
                stats.consecutive_failures += 1

    def _compute_weight(self, template_name: str, step: int) -> float:
        with self._lock:
            stats = self._template_stats.get(template_name)

        if stats is None:
            return 1.0  # Unknown template: full weight

        weight = 1.0

        # Success rate factor
        if self._config.success_weighting_enabled and stats.attempts > 0:
            sr = stats.success_rate
            if stats.attempts >= self._config.fail_threshold and sr == 0.0:
                weight *= self._config.min_weight  # Heavy deprioritize
            elif stats.attempts >= 3:
                weight *= max(self._config.min_weight, 0.3 + 0.7 * sr)

        # Consecutive failure penalty
        if stats.consecutive_failures >= 3:
            penalty = max(0.1, 1.0 - 0.15 * stats.consecutive_failures)
            weight *= penalty

        # Recency bonus: boost commands not used recently
        if step > 0 and stats.last_used_step > 0:
            steps_since = step - stats.last_used_step
            if steps_since > self._config.recency_bonus_steps:
                weight *= 1.2  # Slight exploration boost

        return max(self._config.min_weight, min(weight, 3.0))

    def narrow_for_os(
        self,
        candidates: List[Any],
        target_os: Optional[str] = None,
    ) -> List[Any]:
        """Filter candidates by OS affinity."""
        if not self._config.os_filter_enabled:
            return candidates

        with self._lock:
            _detected = self._detected_os
        os_type = target_os or _detected
        if not os_type:
            return candidates

        narrowed = []
        for c in candidates:
            affinity = getattr(c, "os_affinity", "any")
            if affinity == "any" or affinity == os_type:
                narrowed.append(c)

        if len(narrowed) < 3:
            return candidates  # Don't narrow too aggressively

        return narrowed

    def narrow_for_services(
        self,
        candidates: List[Any],
        known_services: Set[str],
    ) -> List[Any]:
        """Filter candidates by service relevance."""
        if not self._config.service_filter_enabled or not known_services:
            return candidates

        service_lower = {s.lower() for s in known_services}

        scored: List[Tuple[float, Any]] = []
        for c in candidates:
            tags = getattr(c, "tags", set()) or set()
            tags_lower = {t.lower() for t in tags}

            relevance = 1.0
            # Check tag overlap with discovered services
            overlap = tags_lower & service_lower
            if overlap:
                relevance += 0.5 * len(overlap)

            # Check if any service keyword appears in description
            desc = getattr(c, "description", "") or ""
            desc_lower = desc.lower()
            for svc in service_lower:
                if svc in desc_lower:
                    relevance += 0.3

            scored.append((relevance, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored]

    def get_weighted_candidates(
        self,
        candidates: List[Any],
        phase: Optional[str] = None,
        known_services: Optional[Set[str]] = None,
        step: int = 0,
    ) -> List[Tuple[float, Any]]:
        """Return candidates with computed weights, sorted by weight descending."""
        with self._lock:
            self._total_input += len(candidates)

        # Stage 1: OS filter
        filtered = self.narrow_for_os(candidates)

        # Stage 2: Service relevance
        if known_services:
            filtered = self.narrow_for_services(filtered, known_services)

        # Stage 3: Compute weights
        weighted: List[Tuple[float, Any]] = []
        for c in filtered:
            name = getattr(c, "name", "") or str(c)
            weight = self._compute_weight(name, step)

            # Phase bonus
            if phase:
                cmd_phase = getattr(c, "phase", None)
                if cmd_phase is not None:
                    phase_name = cmd_phase.name if hasattr(cmd_phase, "name") else str(cmd_phase)
                    if phase_name.upper() == phase.upper():
                        weight *= self._config.phase_bonus

            weighted.append((weight, c))

        weighted.sort(key=lambda x: x[0], reverse=True)
        with self._lock:
            self._total_narrowed += len(weighted)
        return weighted

    def get_stats(self) -> Dict[str, Any]:
        """Return narrower statistics."""
        with self._lock:
            total_templates = len(self._template_stats)
            avg_sr = 0.0
            if total_templates:
                avg_sr = sum(
                    s.success_rate for s in self._template_stats.values()
                ) / total_templates
            detected_os = self._detected_os
            total_input = self._total_input
            total_narrowed = self._total_narrowed
        return {
            "detected_os": detected_os,
            "tracked_templates": total_templates,
            "avg_success_rate": round(avg_sr, 3),
            "total_input": total_input,
            "total_output": total_narrowed,
            "narrowing_ratio": round(
                total_narrowed / max(1, total_input), 3
            ),
        }

    def reset(self) -> None:
        """Reset all stats."""
        with self._lock:
            self._template_stats.clear()
            self._detected_os = None
            self._total_narrowed = 0
            self._total_input = 0
