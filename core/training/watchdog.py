#!/usr/bin/env python3
"""
core/training/watchdog.py — ARIASKA Training Watchdog v1.0

Production watchdog for overnight training safety.

5 Trigger Classes:
  1. STALL        — No new discoveries for N consecutive steps
  2. FAMILY_FLOOD — Same command family (nmap, hydra, etc.) repeated M+ times
  3. TARGET_DOWN  — Target unreachable (LIVE mode: ping fails T seconds)
  4. PHASE_STUCK  — Phase hasn't advanced in K steps
  5. WALL_CLOCK   — Single step exceeds W seconds (hung process)

Each trigger has an auto-heal action:
  1. force_mentor    — Force a mentor call next step to break out
  2. mask_family     — Block the offending command family for N steps
  3. pause_retry     — Sleep + retry target health check
  4. soft_abort      — End episode early, move to next
  5. kill_skip       — Kill the current step, log and skip

Usage:
    watchdog = TrainingWatchdog(config)
    # After each step in _run_step():
    verdict = watchdog.check(step_snapshot)
    if verdict.should_intervene:
        apply_intervention(verdict)
"""

import time
import logging
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import Counter, deque

logger = logging.getLogger("ariaska.watchdog")


class WatchdogTrigger(str, Enum):
    """What triggered the watchdog."""
    STALL = "stall"
    FAMILY_FLOOD = "family_flood"
    TARGET_DOWN = "target_down"
    PHASE_STUCK = "phase_stuck"
    WALL_CLOCK = "wall_clock"


class HealAction(str, Enum):
    """Auto-heal action to apply."""
    FORCE_MENTOR = "force_mentor"
    MASK_FAMILY = "mask_family"
    PAUSE_RETRY = "pause_retry"
    SOFT_ABORT = "soft_abort"
    KILL_SKIP = "kill_skip"
    NONE = "none"


@dataclass
class WatchdogConfig:
    """Configuration knobs for the training watchdog."""
    # Stall: no discoveries for N steps across all agents
    stall_threshold: int = 15
    stall_heal: HealAction = HealAction.FORCE_MENTOR

    # Family flood: same command family M+ times in last K steps
    family_flood_count: int = 6
    family_flood_window: int = 12
    family_flood_heal: HealAction = HealAction.MASK_FAMILY
    family_mask_duration: int = 5  # How many steps to mask the family

    # Target down: ping fails for LIVE mode
    target_down_retries: int = 3
    target_down_sleep: float = 10.0
    target_down_heal: HealAction = HealAction.PAUSE_RETRY

    # Phase stuck: same phase for K steps without progress
    phase_stuck_threshold: int = 30
    phase_stuck_heal: HealAction = HealAction.FORCE_MENTOR

    # Wall clock: single step exceeds W seconds
    wall_clock_limit: float = 60.0
    wall_clock_heal: HealAction = HealAction.KILL_SKIP

    # Episode-level wall clock: entire episode exceeds E seconds
    episode_wall_clock_limit: float = 600.0  # 10 minutes per episode


@dataclass
class StepSnapshot:
    """Snapshot of the current step for watchdog analysis."""
    step_num: int
    phase: str
    agent_name: str
    command: str
    command_family: str  # e.g. "nmap", "hydra", "sqlmap"
    discoveries: Dict[str, Any] = field(default_factory=dict)
    step_start_time: float = 0.0
    step_end_time: float = 0.0
    is_live_mode: bool = False
    target_ip: str = ""


@dataclass
class WatchdogVerdict:
    """Result of a watchdog check — what to do next."""
    should_intervene: bool = False
    trigger: Optional[WatchdogTrigger] = None
    heal_action: HealAction = HealAction.NONE
    severity: str = "info"  # "info", "warning", "critical"
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    # For MASK_FAMILY action
    masked_family: Optional[str] = None
    mask_until_step: int = 0

    # For SOFT_ABORT
    abort_episode: bool = False

    def __bool__(self) -> bool:
        return self.should_intervene


class TrainingWatchdog:
    """
    Production watchdog monitoring training health.
    
    Call check() after every agent step. Returns a verdict
    with intervention instructions if something is wrong.
    
    Thread-safe: state is per-episode and reset via reset_episode().
    """

    def __init__(self, config: Optional[WatchdogConfig] = None):
        self.config = config or WatchdogConfig()

        # Per-episode state
        self._steps_without_discovery: int = 0
        self._command_history: deque = deque(maxlen=200)
        self._family_history: deque = deque(maxlen=200)
        self._phase_history: List[str] = []
        self._phase_entered_step: Dict[str, int] = {}
        self._current_phase: str = "RECON"
        self._current_step: int = 0
        self._episode_start_time: float = 0.0

        # Family masking state
        self._masked_families: Dict[str, int] = {}  # family → unmasked_at_step

        # Cumulative stats
        self._trigger_counts: Counter = Counter()
        self._total_interventions: int = 0
        self._heal_actions_taken: Counter = Counter()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset_episode(self) -> None:
        """Reset per-episode tracking for a new episode."""
        self._steps_without_discovery = 0
        self._command_history.clear()
        self._family_history.clear()
        self._phase_history = []
        self._phase_entered_step = {}
        self._current_phase = "RECON"
        self._current_step = 0
        self._episode_start_time = time.time()
        self._masked_families.clear()

    # ------------------------------------------------------------------
    # Main check — call after each agent step
    # ------------------------------------------------------------------

    def check(self, snapshot: StepSnapshot) -> WatchdogVerdict:
        """
        Analyze a step snapshot for anomalies.
        
        Returns WatchdogVerdict with intervention details.
        Multiple triggers are checked in priority order;
        the FIRST one that fires wins.
        
        Priority: WALL_CLOCK > TARGET_DOWN > FAMILY_FLOOD > PHASE_STUCK > STALL
        """
        self._current_step = snapshot.step_num
        self._command_history.append(snapshot.command)
        self._family_history.append(snapshot.command_family)

        # Track phase
        if snapshot.phase != self._current_phase:
            self._phase_entered_step[snapshot.phase] = snapshot.step_num
            self._phase_history.append(snapshot.phase)
            self._current_phase = snapshot.phase

        # Track discoveries
        if snapshot.discoveries:
            self._steps_without_discovery = 0
        else:
            self._steps_without_discovery += 1

        # ── 1. WALL_CLOCK: step took too long ───────────────────────
        if snapshot.step_end_time > 0 and snapshot.step_start_time > 0:
            step_duration = snapshot.step_end_time - snapshot.step_start_time
            if step_duration > self.config.wall_clock_limit:
                return self._fire(
                    WatchdogTrigger.WALL_CLOCK,
                    self.config.wall_clock_heal,
                    "critical",
                    f"Step {snapshot.step_num} took {step_duration:.1f}s "
                    f"(limit: {self.config.wall_clock_limit:.0f}s)",
                    {"duration_s": step_duration, "agent": snapshot.agent_name},
                )

        # ── Episode wall clock ──────────────────────────────────────
        if self._episode_start_time > 0:
            ep_elapsed = time.time() - self._episode_start_time
            if ep_elapsed > self.config.episode_wall_clock_limit:
                v = self._fire(
                    WatchdogTrigger.WALL_CLOCK,
                    HealAction.SOFT_ABORT,
                    "critical",
                    f"Episode exceeded {self.config.episode_wall_clock_limit:.0f}s wall clock",
                    {"elapsed_s": ep_elapsed},
                )
                v.abort_episode = True
                return v

        # ── 2. FAMILY_FLOOD: same command family too many times ─────
        if snapshot.command_family:
            recent = list(self._family_history)[-self.config.family_flood_window:]
            family_count = recent.count(snapshot.command_family)
            if family_count >= self.config.family_flood_count:
                v = self._fire(
                    WatchdogTrigger.FAMILY_FLOOD,
                    self.config.family_flood_heal,
                    "warning",
                    f"Family '{snapshot.command_family}' used {family_count}x "
                    f"in last {self.config.family_flood_window} steps",
                    {"family": snapshot.command_family, "count": family_count},
                )
                v.masked_family = snapshot.command_family
                v.mask_until_step = snapshot.step_num + self.config.family_mask_duration
                self._masked_families[snapshot.command_family] = v.mask_until_step
                return v

        # ── 3. PHASE_STUCK: same phase too long ────────────────────
        entered = self._phase_entered_step.get(self._current_phase, 0)
        steps_in_phase = snapshot.step_num - entered
        if steps_in_phase >= self.config.phase_stuck_threshold:
            return self._fire(
                WatchdogTrigger.PHASE_STUCK,
                self.config.phase_stuck_heal,
                "warning",
                f"Stuck in {self._current_phase} for {steps_in_phase} steps "
                f"(threshold: {self.config.phase_stuck_threshold})",
                {"phase": self._current_phase, "steps_in_phase": steps_in_phase},
            )

        # ── 4. STALL: no discoveries ────────────────────────────────
        if self._steps_without_discovery >= self.config.stall_threshold:
            return self._fire(
                WatchdogTrigger.STALL,
                self.config.stall_heal,
                "warning",
                f"No discoveries for {self._steps_without_discovery} steps "
                f"(threshold: {self.config.stall_threshold})",
                {"steps_dry": self._steps_without_discovery},
            )

        # All clear
        return WatchdogVerdict()

    def check_target_health(self, target_ip: str) -> WatchdogVerdict:
        """
        Standalone target health check (for LIVE mode).
        
        Called before each episode or periodically.
        """
        import subprocess
        for attempt in range(self.config.target_down_retries):
            try:
                result = subprocess.run(
                    ["ping", "-c", "1", "-W", "3", target_ip],
                    capture_output=True,
                    timeout=5.0,
                )
                if result.returncode == 0:
                    return WatchdogVerdict()  # All clear
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                pass

            if attempt < self.config.target_down_retries - 1:
                logger.warning(
                    f"[WATCHDOG] Target {target_ip} not responding, "
                    f"retry {attempt + 1}/{self.config.target_down_retries} "
                    f"in {self.config.target_down_sleep}s"
                )
                time.sleep(self.config.target_down_sleep)

        # All retries exhausted
        v = self._fire(
            WatchdogTrigger.TARGET_DOWN,
            self.config.target_down_heal,
            "critical",
            f"Target {target_ip} unreachable after "
            f"{self.config.target_down_retries} retries",
            {"target_ip": target_ip},
        )
        v.abort_episode = True
        return v

    # ------------------------------------------------------------------
    # Family masking queries
    # ------------------------------------------------------------------

    def is_family_masked(self, family: str) -> bool:
        """Check if a command family is currently masked."""
        if family not in self._masked_families:
            return False
        if self._current_step >= self._masked_families[family]:
            del self._masked_families[family]
            return False
        return True

    def get_masked_families(self) -> Set[str]:
        """Return set of currently masked command families."""
        # Clean expired masks
        expired = [
            f for f, until in self._masked_families.items()
            if self._current_step >= until
        ]
        for f in expired:
            del self._masked_families[f]
        return set(self._masked_families.keys())

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get watchdog statistics for reporting."""
        return {
            "total_interventions": self._total_interventions,
            "trigger_counts": dict(self._trigger_counts),
            "heal_actions": dict(self._heal_actions_taken),
            "steps_without_discovery": self._steps_without_discovery,
            "current_phase": self._current_phase,
            "masked_families": dict(self._masked_families),
            "episode_elapsed_s": (
                time.time() - self._episode_start_time
                if self._episode_start_time > 0 else 0.0
            ),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fire(
        self,
        trigger: WatchdogTrigger,
        heal: HealAction,
        severity: str,
        message: str,
        details: Dict[str, Any],
    ) -> WatchdogVerdict:
        """Create a verdict and update stats."""
        self._trigger_counts[trigger.value] += 1
        self._total_interventions += 1
        self._heal_actions_taken[heal.value] += 1

        log_fn = logger.warning if severity == "warning" else logger.error
        log_fn(f"[WATCHDOG:{trigger.value.upper()}] {message}")

        return WatchdogVerdict(
            should_intervene=True,
            trigger=trigger,
            heal_action=heal,
            severity=severity,
            message=message,
            details=details,
        )


def extract_command_family(command: str) -> str:
    """
    Extract the tool/command family from a full command string.
    
    Examples:
        "nmap -sV 10.10.10.10"  →  "nmap"
        "hydra -l admin ..."     →  "hydra"
        "msfconsole -x ..."      →  "msfconsole"
        "echo test"              →  "echo"
    """
    if not command:
        return "unknown"
    # Strip leading env vars, sudo, etc.
    parts = command.strip().split()
    for part in parts:
        if "=" in part:
            continue  # Skip env vars like FOO=bar
        if part in ("sudo", "bash", "-c", "sh", "timeout", "nice"):
            continue
        # Return first real command token
        return part.split("/")[-1]  # Handle /usr/bin/nmap → nmap
    return "unknown"
