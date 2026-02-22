"""core/algorithms/llm_reward_shaper.py — Phase 49: LLM-Based Reward Shaping.

Uses GPT to retroactively score trajectory quality for denser reward
signal, filling the sparse-reward gap between discoveries.

Design:
  - After each episode, send a summary of the trajectory to GPT.
  - GPT returns per-step quality annotations (0-1 scale).
  - Annotations become auxiliary reward signal for PPO training.
  - Budget-aware: only shapes when BudgetManagerV2 allows.
  - Falls back to heuristic shaping when offline/budget-exhausted.

Integration:
  - Called from SmartOrchestrator after episode completion.
  - Shaped rewards stored in replay buffer alongside original rewards.
  - PPO uses: final_reward = (1-α)*env_reward + α*shaped_reward
  - α anneals from 0.3 → 0.05 as learning matures.

All LLM calls go through GPTManager (project invariant).
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.gpt_manager import GPTManager

logger = logging.getLogger("ariaska.algorithms.llm_reward_shaper")

# Phase labels for context
_PHASE_NAMES = [
    "RECON", "ENUMERATION", "EXPLOITATION", "PRIVILEGE_ESCALATION",
    "LATERAL_MOVEMENT", "POST_EXPLOITATION", "EXFILTRATION", "CLOSEOUT",
]


@dataclass
class RewardShaperConfig:
    """Configuration for LLM-based reward shaping.

    Args:
        enabled: Master switch.
        alpha_init: Initial blending weight for shaped rewards.
        alpha_min: Minimum alpha after annealing.
        anneal_episodes: Episodes over which alpha anneals.
        max_steps_per_call: Max trajectory steps to include per GPT call.
        heuristic_fallback: Use heuristic shaping when LLM unavailable.
        phase_progression_bonus: Reward bonus per phase transition.
        novelty_decay: Decay factor for command novelty scoring.
        batch_size: Chunk trajectory into batches of this size for scoring.
    """
    enabled: bool = True
    alpha_init: float = 0.30
    alpha_min: float = 0.05
    anneal_episodes: int = 200
    max_steps_per_call: int = 50
    heuristic_fallback: bool = True
    phase_progression_bonus: float = 5.0
    novelty_decay: float = 0.95
    batch_size: int = 25


@dataclass
class StepAnnotation:
    """Per-step quality annotation from LLM or heuristic."""
    step: int
    quality: float        # [0, 1] — overall step quality
    reasoning: str        # Brief justification
    phase_fit: float      # How well the command fits the current phase
    novelty: float        # How novel/non-redundant the command is
    progress: float       # How much the step advances toward goal
    shaped_reward: float  # Final shaped reward = quality * scale_factor


class LLMRewardShaper:
    """Shapes rewards using GPT trajectory analysis.

    Usage::

        shaper = LLMRewardShaper(config, gpt_manager=gpt)
        # After episode:
        annotations = shaper.shape_episode(trajectory, episode_num=42)
        # Blend:
        final_rewards = shaper.blend_rewards(original_rewards, annotations)
    """

    def __init__(
        self,
        config: Optional[RewardShaperConfig] = None,
        gpt_manager: Optional["GPTManager"] = None,
    ) -> None:
        self.config = config or RewardShaperConfig()
        self._gpt = gpt_manager
        self._episode_count: int = 0
        self._command_history: Dict[str, int] = {}  # cmd → use count

    @property
    def alpha(self) -> float:
        """Current blending weight, annealed over episodes."""
        if self.config.anneal_episodes <= 0:
            return self.config.alpha_init
        progress = min(1.0, self._episode_count / self.config.anneal_episodes)
        return (
            self.config.alpha_init
            + (self.config.alpha_min - self.config.alpha_init) * progress
        )

    def shape_episode(
        self,
        trajectory: List[Dict[str, Any]],
        episode_num: int = 0,
    ) -> List[StepAnnotation]:
        """Score each step of a trajectory for quality.

        Tries LLM-based scoring first, falls back to heuristic.

        Args:
            trajectory: List of step dicts, each with 'command', 'phase',
                       'reward', 'discoveries', etc.
            episode_num: Current episode number for annealing.

        Returns:
            List of StepAnnotation, one per trajectory step.
        """
        self._episode_count = episode_num

        # Try LLM scoring
        if self._gpt is not None and self.config.enabled:
            try:
                return self._llm_score(trajectory)
            except Exception as e:
                logger.warning(f"LLM scoring failed, falling back to heuristic: {e}")

        # Heuristic fallback
        if self.config.heuristic_fallback:
            return self._heuristic_score(trajectory)

        # No shaping available
        return [
            StepAnnotation(
                step=i, quality=0.5, reasoning="no_shaping",
                phase_fit=0.5, novelty=0.5, progress=0.0, shaped_reward=0.0,
            )
            for i in range(len(trajectory))
        ]

    def _llm_score(
        self, trajectory: List[Dict[str, Any]],
    ) -> List[StepAnnotation]:
        """Score trajectory using GPT via GPTManager."""
        annotations: List[StepAnnotation] = []

        # Chunk trajectory into batches
        for batch_start in range(0, len(trajectory), self.config.batch_size):
            batch = trajectory[batch_start:batch_start + self.config.batch_size]
            batch_annotations = self._score_batch(batch, batch_start)
            annotations.extend(batch_annotations)

        return annotations

    def _score_batch(
        self,
        batch: List[Dict[str, Any]],
        offset: int,
    ) -> List[StepAnnotation]:
        """Score a batch of steps via GPT."""
        # Build compact trajectory summary
        steps_text = []
        for i, step in enumerate(batch):
            cmd = step.get("command", "unknown")[:80]
            phase = step.get("phase", "?")
            reward = step.get("reward", 0.0)
            disc = step.get("discoveries", [])
            steps_text.append(
                f"  Step {offset + i}: [{phase}] {cmd} → reward={reward:.1f}"
                + (f" discoveries={disc}" if disc else "")
            )

        prompt = (
            "Rate each pentest step for quality (0.0-1.0) considering:\n"
            "- phase_fit: Does the command match the current attack phase?\n"
            "- novelty: Is it non-redundant vs previous steps?\n"
            "- progress: Does it advance toward the goal?\n\n"
            "Trajectory:\n" + "\n".join(steps_text) + "\n\n"
            "Return JSON array: [{\"step\": N, \"quality\": 0.X, "
            "\"phase_fit\": 0.X, \"novelty\": 0.X, \"progress\": 0.X, "
            "\"reasoning\": \"brief\"}]"
        )

        try:
            assert self._gpt is not None
            response = self._gpt.gpt_request(
                prompt,
                task_type="reward_shaping",
                agent_id="RewardShaper",
            )

            if response:
                return self._parse_llm_response(response, batch, offset)
        except Exception as e:
            logger.debug(f"GPT scoring failed for batch at offset {offset}: {e}")

        # Fallback to heuristic for this batch
        return self._heuristic_score(batch, offset)

    def _parse_llm_response(
        self,
        response: str,
        batch: List[Dict[str, Any]],
        offset: int,
    ) -> List[StepAnnotation]:
        """Parse GPT JSON response into StepAnnotations."""
        annotations: List[StepAnnotation] = []
        try:
            # Extract JSON from response
            text = response.strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            data = json.loads(text)

            if isinstance(data, list):
                for i, item in enumerate(data):
                    step_idx = item.get("step", offset + i)
                    quality = max(0.0, min(1.0, float(item.get("quality", 0.5))))
                    phase_fit = max(0.0, min(1.0, float(item.get("phase_fit", 0.5))))
                    novelty = max(0.0, min(1.0, float(item.get("novelty", 0.5))))
                    progress = max(0.0, min(1.0, float(item.get("progress", 0.0))))
                    reasoning = str(item.get("reasoning", ""))[:200]

                    shaped = quality * 10.0  # Scale to reward-range
                    annotations.append(StepAnnotation(
                        step=step_idx,
                        quality=quality,
                        reasoning=reasoning,
                        phase_fit=phase_fit,
                        novelty=novelty,
                        progress=progress,
                        shaped_reward=shaped,
                    ))
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.debug(f"Failed to parse LLM response: {e}")
            return self._heuristic_score(batch, offset)

        # Pad if LLM returned fewer annotations than batch size
        while len(annotations) < len(batch):
            idx = offset + len(annotations)
            annotations.append(StepAnnotation(
                step=idx, quality=0.5, reasoning="padding",
                phase_fit=0.5, novelty=0.5, progress=0.0, shaped_reward=0.0,
            ))

        return annotations[:len(batch)]

    def _heuristic_score(
        self,
        trajectory: List[Dict[str, Any]],
        offset: int = 0,
    ) -> List[StepAnnotation]:
        """Score trajectory using rule-based heuristics (no LLM needed).

        Scoring rules:
          - Phase fit: higher if command family matches expected phase
          - Novelty: decays with repeated commands
          - Progress: discoveries, phase transitions get bonuses
        """
        annotations: List[StepAnnotation] = []
        prev_phase: Optional[str] = None

        for i, step in enumerate(trajectory):
            cmd = step.get("command", "")
            phase = step.get("phase", "RECON")
            reward = step.get("reward", 0.0)
            discoveries = step.get("discoveries", [])

            # Phase fit heuristic
            phase_fit = self._heuristic_phase_fit(cmd, phase)

            # Novelty via command dedup
            cmd_key = cmd.split()[0] if cmd else ""
            use_count = self._command_history.get(cmd_key, 0)
            novelty = max(0.1, self.config.novelty_decay ** use_count)
            self._command_history[cmd_key] = use_count + 1

            # Progress heuristic
            progress = 0.0
            if discoveries:
                progress += min(1.0, len(discoveries) * 0.3)
            if prev_phase and phase != prev_phase:
                # Phase transition = progress
                try:
                    old_idx = _PHASE_NAMES.index(prev_phase)
                    new_idx = _PHASE_NAMES.index(phase)
                    if new_idx > old_idx:
                        progress += 0.5
                except ValueError:
                    pass

            quality = 0.3 * phase_fit + 0.3 * novelty + 0.4 * progress
            shaped_reward = quality * 10.0

            annotations.append(StepAnnotation(
                step=offset + i,
                quality=round(quality, 4),
                reasoning="heuristic",
                phase_fit=round(phase_fit, 4),
                novelty=round(novelty, 4),
                progress=round(progress, 4),
                shaped_reward=round(shaped_reward, 4),
            ))
            prev_phase = phase

        return annotations

    @staticmethod
    def _heuristic_phase_fit(command: str, phase: str) -> float:
        """Heuristic: does the command family match the phase?"""
        cmd_lower = command.lower()
        phase_upper = phase.upper()

        recon_cmds = {"nmap", "ping", "traceroute", "dig", "whois", "masscan"}
        enum_cmds = {"gobuster", "nikto", "enum4linux", "smbclient", "dirb", "ffuf"}
        exploit_cmds = {"msfconsole", "exploit", "sqlmap", "hydra", "john", "hashcat"}
        privesc_cmds = {"linpeas", "sudo", "getcap", "find_suid", "pspy"}
        exfil_cmds = {"scp", "rsync", "curl", "wget", "nc", "base64"}

        cmd_family = cmd_lower.split()[0] if cmd_lower else ""

        if phase_upper in ("RECON",) and cmd_family in recon_cmds:
            return 0.9
        if phase_upper in ("ENUMERATION",) and cmd_family in enum_cmds:
            return 0.9
        if phase_upper in ("EXPLOITATION",) and cmd_family in exploit_cmds:
            return 0.9
        if phase_upper in ("PRIVILEGE_ESCALATION",) and cmd_family in privesc_cmds:
            return 0.9
        if phase_upper in ("EXFILTRATION",) and cmd_family in exfil_cmds:
            return 0.9

        return 0.4  # Neutral fit

    def blend_rewards(
        self,
        original_rewards: List[float],
        annotations: List[StepAnnotation],
    ) -> List[float]:
        """Blend original rewards with shaped rewards.

        Formula: final = (1 - alpha) * original + alpha * shaped
        """
        alpha = self.alpha
        blended = []
        for orig, ann in zip(original_rewards, annotations):
            final = (1.0 - alpha) * orig + alpha * ann.shaped_reward
            blended.append(round(final, 4))
        return blended

    def get_stats(self) -> Dict[str, Any]:
        """Return shaper statistics."""
        return {
            "enabled": self.config.enabled,
            "alpha": round(self.alpha, 4),
            "episode_count": self._episode_count,
            "unique_commands_seen": len(self._command_history),
            "has_gpt": self._gpt is not None,
        }

    def reset(self) -> None:
        """Reset episode counter and command history."""
        self._episode_count = 0
        self._command_history.clear()
