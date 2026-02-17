"""Overnight training guardrails for safe unattended operation.

Phase 6.6: Safety mechanisms for long-running autonomous training:
    - BudgetGovernor:  Token budget enforcement (per-episode + per-run)
    - HealthChecker:   Target reachability verification before each episode
    - StagnationWatchdog: Detect K consecutive zero-progress steps → early end
    - CheckpointScheduler: Auto-save PPO weights every N episodes
    - LogRotator: Keep log files under size limits
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("ariaska.guardrails")


# ═══════════════════════════════════════════════════════════════════════
# 1. BUDGET GOVERNOR — Token budget enforcement
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BudgetGovernor:
    """Enforce token budgets per-episode and per-run.

    Tracks cumulative token usage and provides should_stop() checks.
    """

    max_tokens_per_run: Optional[int] = None      # Total token cap (None = unlimited)
    max_tokens_per_episode: int = 75_000           # Phase 12.0: +50% (was 50K) per-episode soft cap
    warning_threshold_pct: float = 0.80            # Warn at 80% of run budget
    
    # Runtime state
    _run_tokens: int = 0
    _episode_tokens: int = 0
    _warned: bool = False

    def record_tokens(self, count: int) -> None:
        """Record token usage."""
        self._run_tokens += count
        self._episode_tokens += count

    def reset_episode(self) -> None:
        """Reset per-episode counter."""
        self._episode_tokens = 0

    @property
    def run_tokens_used(self) -> int:
        return self._run_tokens

    @property
    def episode_tokens_used(self) -> int:
        return self._episode_tokens

    def should_stop_run(self) -> bool:
        """Check if run budget is exhausted."""
        if self.max_tokens_per_run is None:
            return False
        if self._run_tokens >= self.max_tokens_per_run:
            logger.warning(
                f"[BUDGET] Run token limit reached: {self._run_tokens:,} / {self.max_tokens_per_run:,}"
            )
            return True
        # Warning at threshold
        if not self._warned and self._run_tokens >= self.max_tokens_per_run * self.warning_threshold_pct:
            logger.warning(
                f"[BUDGET] ⚠️ {self.warning_threshold_pct*100:.0f}% of run budget used: "
                f"{self._run_tokens:,} / {self.max_tokens_per_run:,}"
            )
            self._warned = True
        return False

    def should_end_episode(self) -> bool:
        """Check if episode budget is exhausted."""
        return self._episode_tokens >= self.max_tokens_per_episode

    def summary(self) -> Dict[str, int]:
        return {
            "run_tokens_used": self._run_tokens,
            "max_tokens_per_run": self.max_tokens_per_run or -1,
            "episode_tokens_used": self._episode_tokens,
            "max_tokens_per_episode": self.max_tokens_per_episode,
        }


# ═══════════════════════════════════════════════════════════════════════
# 2. HEALTH CHECKER — Target reachability
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class HealthChecker:
    """Verify target is reachable before each episode.

    In live mode, pings the target. If unreachable for max_retries,
    pauses training with exponential backoff.
    """

    target_ip: str = ""
    enabled: bool = True
    max_retries: int = 3
    retry_delay_base: float = 10.0  # seconds, doubles each retry
    timeout: float = 5.0

    def check_target(self) -> bool:
        """Ping target and return True if reachable."""
        if not self.enabled or not self.target_ip:
            return True
        
        for attempt in range(self.max_retries):
            try:
                result = subprocess.run(
                    ["ping", "-c", "1", "-W", str(int(self.timeout)), self.target_ip],
                    capture_output=True,
                    timeout=self.timeout + 2,
                )
                if result.returncode == 0:
                    if attempt > 0:
                        logger.info(f"[HEALTH] Target {self.target_ip} recovered after {attempt} retries")
                    return True
            except (subprocess.TimeoutExpired, OSError) as e:
                logger.warning(f"[HEALTH] Ping attempt {attempt+1}/{self.max_retries} failed: {e}")
            
            if attempt < self.max_retries - 1:
                delay = self.retry_delay_base * (2 ** attempt)
                logger.warning(f"[HEALTH] Target unreachable, waiting {delay:.0f}s before retry...")
                time.sleep(delay)
        
        logger.error(f"[HEALTH] ❌ Target {self.target_ip} unreachable after {self.max_retries} retries")
        return False


# ═══════════════════════════════════════════════════════════════════════
# 3. STAGNATION WATCHDOG — Zero-progress detection
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class StagnationWatchdog:
    """Detect consecutive zero-progress steps and take action.

    If K consecutive steps produce zero new discoveries AND zero phase
    progress, triggers early episode end or mentor call.
    """

    max_zero_progress_steps: int = 8    # Early-end episode after this many
    mentor_trigger_steps: int = 4       # Force mentor call after this many

    # Runtime state
    _zero_progress_count: int = 0
    _total_triggers: int = 0

    def reset(self) -> None:
        """Reset for new episode."""
        self._zero_progress_count = 0

    def record_step(self, had_progress: bool) -> None:
        """Record whether a step made progress."""
        if had_progress:
            self._zero_progress_count = 0
        else:
            self._zero_progress_count += 1

    @property
    def zero_streak(self) -> int:
        return self._zero_progress_count

    def should_force_mentor(self) -> bool:
        """Check if we should force a mentor call."""
        return self._zero_progress_count >= self.mentor_trigger_steps

    def should_end_episode(self) -> bool:
        """Check if we should end the episode early."""
        if self._zero_progress_count >= self.max_zero_progress_steps:
            self._total_triggers += 1
            logger.warning(
                f"[WATCHDOG] ⚠️ {self.max_zero_progress_steps} consecutive zero-progress steps — "
                f"ending episode early (trigger #{self._total_triggers})"
            )
            return True
        return False


# ═══════════════════════════════════════════════════════════════════════
# 4. CHECKPOINT SCHEDULER — Periodic PPO saves
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CheckpointScheduler:
    """Auto-save PPO checkpoints every N episodes.

    Maintains a rolling window of checkpoints to avoid filling disk.
    """

    interval: int = 10             # Save every N episodes
    max_checkpoints: int = 5       # Keep only last N checkpoints
    checkpoint_dir: str = "models/enhanced"
    
    # Runtime
    _saved_paths: List[str] = field(default_factory=list)

    def should_save(self, episode_num: int) -> bool:
        """Check if we should save a checkpoint now."""
        return (episode_num + 1) % self.interval == 0

    def get_save_path(self, episode_num: int, difficulty: str = "normal") -> str:
        """Get the path for a checkpoint save."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(
            self.checkpoint_dir,
            f"ppo_ep{episode_num:04d}_{difficulty}.pt",
        )
        return path

    def record_save(self, path: str) -> None:
        """Record a saved checkpoint and prune old ones."""
        self._saved_paths.append(path)
        # Prune old checkpoints beyond max
        while len(self._saved_paths) > self.max_checkpoints:
            old_path = self._saved_paths.pop(0)
            try:
                if os.path.exists(old_path):
                    os.remove(old_path)
                    logger.info(f"[CHECKPOINT] Pruned old checkpoint: {old_path}")
            except OSError as e:
                logger.warning(f"[CHECKPOINT] Failed to prune {old_path}: {e}")


# ═══════════════════════════════════════════════════════════════════════
# 5. LOG ROTATOR — Size-based log rotation
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class LogRotator:
    """Rotate log files when they exceed size limits."""

    max_size_mb: float = 50.0
    log_paths: List[str] = field(default_factory=lambda: [
        "logs/execution_audit.jsonl",
    ])

    def check_and_rotate(self) -> None:
        """Check all log files and rotate if needed."""
        max_bytes = self.max_size_mb * 1024 * 1024
        for path in self.log_paths:
            if os.path.exists(path):
                size = os.path.getsize(path)
                if size > max_bytes:
                    rotated = f"{path}.{int(time.time())}.bak"
                    try:
                        os.rename(path, rotated)
                        logger.info(
                            f"[LOG-ROTATE] Rotated {path} ({size/1024/1024:.1f}MB) → {rotated}"
                        )
                    except OSError as e:
                        logger.warning(f"[LOG-ROTATE] Failed to rotate {path}: {e}")


# ═══════════════════════════════════════════════════════════════════════
# COMPOSITE GUARDRAIL MANAGER
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class GuardrailManager:
    """Composite manager for all overnight guardrails.

    Usage:
        gm = GuardrailManager(...)
        gm.pre_episode_check(ep)      # Before each episode
        gm.post_step_check(progress)  # After each step
        gm.post_episode_check(ep)     # After each episode
    """

    budget: BudgetGovernor = field(default_factory=BudgetGovernor)
    health: HealthChecker = field(default_factory=HealthChecker)
    watchdog: StagnationWatchdog = field(default_factory=StagnationWatchdog)
    checkpoint: CheckpointScheduler = field(default_factory=CheckpointScheduler)
    log_rotator: LogRotator = field(default_factory=LogRotator)

    def pre_episode_check(self, episode_num: int) -> bool:
        """Run pre-episode checks. Returns False if training should stop."""
        # Check run budget
        if self.budget.should_stop_run():
            logger.warning(f"[GUARDRAIL] Run budget exhausted at episode {episode_num}")
            return False
        
        # Check target health (live mode)
        if not self.health.check_target():
            logger.error(f"[GUARDRAIL] Target unreachable, stopping at episode {episode_num}")
            return False
        
        # Reset per-episode state
        self.budget.reset_episode()
        self.watchdog.reset()
        
        # Rotate logs periodically
        if episode_num % 10 == 0:
            self.log_rotator.check_and_rotate()
        
        return True

    def post_step_check(self, had_progress: bool) -> Dict[str, bool]:
        """Run post-step checks. Returns action flags."""
        self.watchdog.record_step(had_progress)
        return {
            "force_mentor": self.watchdog.should_force_mentor(),
            "end_episode": (
                self.watchdog.should_end_episode()
                or self.budget.should_end_episode()
            ),
        }

    def post_episode_check(
        self, episode_num: int, difficulty: str = "normal"
    ) -> Optional[str]:
        """Run post-episode checks. Returns checkpoint path if saved, else None."""
        if self.checkpoint.should_save(episode_num):
            path = self.checkpoint.get_save_path(episode_num, difficulty)
            return path
        return None

    def summary(self) -> Dict[str, any]:
        return {
            "budget": self.budget.summary(),
            "watchdog_streak": self.watchdog.zero_streak,
            "watchdog_triggers": self.watchdog._total_triggers,
        }
