"""Episode Lifecycle — episode start/end hooks and PPO episode wrappers.

Phase 41: Extracted from SmartCoach for modularity.
"""
from __future__ import annotations

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("ariaska.coach.episode_lifecycle")


@dataclass
class EpisodeConfig:
    """Configuration for episode lifecycle."""
    max_steps: int = 40
    incomplete_closeout_penalty: float = -5.0
    closeout_speed_bonus_max: float = 30.0


@dataclass
class EpisodeState:
    """Mutable state for the current episode."""
    episode: int = 0
    step: int = 0
    total_reward: float = 0.0
    discoveries: int = 0
    unique_commands: int = 0
    highest_phase: str = "RECON"
    is_done: bool = False
    command_set: set = field(default_factory=set)

    def record_step(self, command: str, reward: float, discoveries: int = 0) -> None:
        self.step += 1
        self.total_reward += reward
        self.discoveries += discoveries
        if command and command not in self.command_set:
            self.command_set.add(command)
            self.unique_commands += 1

    @property
    def diversity_ratio(self) -> float:
        return self.unique_commands / max(1, self.step)


class EpisodeLifecycle:
    """Manages episode-level bookkeeping and terminal reward injection."""

    PHASE_ORDER = [
        "RECON", "ENUMERATION", "EXPLOITATION", "PRIVILEGE_ESCALATION",
        "LATERAL_MOVEMENT", "POST_EXPLOITATION", "EXFILTRATION", "CLOSEOUT",
    ]
    TERMINAL_REWARDS: Dict[str, float] = {
        "RECON": 0.0, "ENUMERATION": 5.0, "EXPLOITATION": 15.0,
        "PRIVILEGE_ESCALATION": 30.0, "LATERAL_MOVEMENT": 45.0,
        "POST_EXPLOITATION": 60.0, "EXFILTRATION": 75.0, "CLOSEOUT": 90.0,
    }

    def __init__(self, config: Optional[EpisodeConfig] = None) -> None:
        self.config = config or EpisodeConfig()
        self.current = EpisodeState()
        self._episode_count: int = 0

    def start_episode(self, episode: int) -> None:
        self._episode_count += 1
        self.current = EpisodeState(episode=episode)

    def record_step(self, command: str, reward: float, discoveries: int = 0) -> None:
        self.current.record_step(command, reward, discoveries)

    def update_highest_phase(self, phase: str) -> None:
        current_idx = self.PHASE_ORDER.index(self.current.highest_phase) if self.current.highest_phase in self.PHASE_ORDER else 0
        new_idx = self.PHASE_ORDER.index(phase) if phase in self.PHASE_ORDER else 0
        if new_idx > current_idx:
            self.current.highest_phase = phase

    def compute_terminal_bonus(self, highest_phase: str, traj_len: int) -> float:
        bonus = self.TERMINAL_REWARDS.get(highest_phase, 0.0)
        if highest_phase == "EXFILTRATION":
            bonus += self.config.incomplete_closeout_penalty
        if highest_phase == "CLOSEOUT" and traj_len > 0:
            bonus += min(self.config.closeout_speed_bonus_max, max(0.0, (self.config.max_steps - traj_len) * 1.0))
        return bonus

    def get_summary(self) -> Dict[str, Any]:
        return {
            "episode": self.current.episode, "steps": self.current.step,
            "total_reward": self.current.total_reward, "discoveries": self.current.discoveries,
            "unique_commands": self.current.unique_commands, "diversity_ratio": self.current.diversity_ratio,
            "highest_phase": self.current.highest_phase,
        }
