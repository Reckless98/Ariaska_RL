"""GRPO-Style Group Relative Policy Optimization — Ariaska RL.

Implements Group Relative Policy Optimization (GRPO) adapted for the
Ariaska pentesting RL domain. Instead of single-rollout PPO, this samples
GROUP_SIZE rollouts per state, computes group-relative advantages, and
updates with clipped trust-region policy gradient — no value network needed.

Key differences from standard PPO distillation (h200_run_distill_3h.py):
  - Group sampling: N rollouts per state → relative ranking within group
  - No critic: advantage = (reward - group_mean) / group_std  (DeepSeekMath-style)
  - Configurable multi-channel reward: format + code + math + reasoning
  - Auto-loads best .pt checkpoint as base (persistent memory)
  - Compatible with vLLM gpt-oss-120b mentor for teacher signals

Usage:
    # Standard GRPO training (3h)
    python -m scripts.train_grpo --seed 42 --max-hours 3

    # Fine-tuning with lower LR and larger groups
    python -m scripts.train_grpo --learning-rate 5e-6 --group-size 8

    # Resume from checkpoint
    python -m scripts.train_grpo --resume models/distilled/h200_xxx_ep0100.pt

    # Custom reward weights (format strongest, reasoning weakest)
    python -m scripts.train_grpo --reward-weights "format=2.5,code=1.5,math=1.0,reasoning=0.3"
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Load .env ────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path, override=False)
except ImportError:
    pass

logger = logging.getLogger("ariaska.train_grpo")

# ── Rich output (project invariant: never bare print) ────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
except ImportError:
    raise SystemExit("ERROR: 'rich' is required. Install with: pip install rich")

console = Console(force_terminal=True)

# ── TensorBoard ──────────────────────────────────────────────────────
try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TENSORBOARD = True
except ImportError:
    _HAS_TENSORBOARD = False
    SummaryWriter = None  # type: ignore[misc,assignment]

TENSORBOARD_DIR = Path("runs/grpo_train")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATE_DIM = 512
ACTION_DIM = 5
REWARD_MIN = -15.0
REWARD_MAX = 100.0
CHECKPOINT_DIR = Path("models/distilled")
RESULTS_DIR = Path("results/grpo_train")
TRACES_DIR = Path("traces/grpo_train")

# GRPO hyperparameters
DEFAULT_GROUP_SIZE = 4          # rollouts per state for relative advantage
DEFAULT_CLIP_EPSILON = 0.2      # PPO-style clip
DEFAULT_KL_COEF = 0.04          # KL penalty against reference policy
DEFAULT_ENTROPY_COEF = 0.01     # entropy bonus for exploration
DEFAULT_REFERENCE_UPDATE = 50   # update reference policy every N episodes

# Reward channels
DEFAULT_REWARD_WEIGHTS: Dict[str, float] = {
    "format": 2.0,       # Phase progression (strongest signal)
    "code": 1.5,         # Discovery / tool execution
    "math": 1.5,         # Exploitation logic verification
    "reasoning": 0.5,    # Consistency (weakest — gameable)
}

# Phase ordering
PHASES = [
    "RECON", "ENUMERATION", "EXPLOITATION", "PRIVILEGE_ESCALATION",
    "LATERAL_MOVEMENT", "POST_EXPLOITATION", "EXFILTRATION", "CLOSEOUT",
]

PHASE_REWARDS: Dict[str, float] = {
    "RECON": 0.0, "ENUMERATION": 5.0, "EXPLOITATION": 15.0,
    "PRIVILEGE_ESCALATION": 30.0, "LATERAL_MOVEMENT": 45.0,
    "POST_EXPLOITATION": 60.0, "EXFILTRATION": 75.0, "CLOSEOUT": 90.0,
}

# Phase-command alignment for consistency reward
PHASE_ALIGNMENT: Dict[str, set] = {
    "RECON": {"nmap", "masscan", "ping", "traceroute", "whatweb", "dig", "host"},
    "ENUMERATION": {"gobuster", "nikto", "dirb", "wfuzz", "enum4linux",
                     "smbclient", "showmount", "snmpwalk", "searchsploit"},
    "EXPLOITATION": {"msfconsole", "hydra", "sqlmap", "curl", "wget",
                      "python", "python3", "exploit", "searchsploit"},
    "PRIVILEGE_ESCALATION": {"sudo", "su", "find", "linpeas", "linenum",
                              "getcap", "cat", "ls", "id", "whoami"},
    "POST_EXPLOITATION": {"cat", "ls", "whoami", "hostname", "id"},
    "EXFILTRATION": {"tar", "scp", "nc", "curl", "wget"},
}

# Training defaults
DEFAULT_MAX_STEPS_PER_EPISODE = 150
DEFAULT_EPISODES = 500
DEFAULT_MAX_HOURS = 3.0
DEFAULT_CHECKPOINT_SEC = 600
DEFAULT_LEARNING_RATE = 1e-4  # lower than PPO — group advantage is less noisy


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GRPOConfig:
    """GRPO training configuration."""
    state_dim: int = STATE_DIM
    action_dim: int = ACTION_DIM
    group_size: int = DEFAULT_GROUP_SIZE
    clip_epsilon: float = DEFAULT_CLIP_EPSILON
    kl_coef: float = DEFAULT_KL_COEF
    entropy_coef: float = DEFAULT_ENTROPY_COEF
    learning_rate: float = DEFAULT_LEARNING_RATE
    reference_update_interval: int = DEFAULT_REFERENCE_UPDATE
    reward_weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_REWARD_WEIGHTS))
    max_steps_per_episode: int = DEFAULT_MAX_STEPS_PER_EPISODE
    max_episodes: int = DEFAULT_EPISODES
    max_hours: float = DEFAULT_MAX_HOURS
    checkpoint_interval_sec: float = DEFAULT_CHECKPOINT_SEC


@dataclass
class GroupRollout:
    """A single rollout within a group sample."""
    states: List[Any] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    commands: List[str] = field(default_factory=list)
    total_reward: float = 0.0
    discoveries: int = 0
    final_phase: str = "RECON"


@dataclass
class GRPOMetrics:
    """Aggregated training metrics."""
    episodes_completed: int = 0
    total_steps: int = 0
    total_updates: int = 0
    wall_hours: float = 0.0
    avg_group_reward: float = 0.0
    avg_advantage_std: float = 0.0
    best_episode_reward: float = float("-inf")
    avg_kl_divergence: float = 0.0
    checkpoints_saved: int = 0
    reward_history: List[float] = field(default_factory=list)
    phase_distribution: Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# GRPO Trainer
# ---------------------------------------------------------------------------

class GRPOTrainer:
    """Group Relative Policy Optimization trainer for pentesting RL.

    Core GRPO idea (DeepSeekMath / DeepSeek-R1):
      For each state, sample GROUP_SIZE rollouts. Compute group-relative
      advantage: A_i = (R_i - mean(R_group)) / std(R_group).
      Update policy with clipped ratio * advantage (no critic needed).
      Add KL penalty against frozen reference policy to prevent drift.
    """

    def __init__(
        self,
        config: GRPOConfig,
        seed: int = 42,
        resume_from: Optional[str] = None,
        device: str = "auto",
    ) -> None:
        self.config = config
        self.seed = seed
        self.resume_from = resume_from
        self.metrics = GRPOMetrics()
        self._consistency_recent_cmds: List[str] = []
        self._run_id = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        import torch
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._seed_all(seed)

        # Lazy-init
        self._ppo: Any = None         # active policy (PPOAgent)
        self._ref_ppo: Any = None     # frozen reference policy
        self._env: Any = None
        self._tb: Optional[Any] = None
        self._last_checkpoint_time: float = 0.0
        self._trace_path: Optional[Path] = None

    # ── Initialization ───────────────────────────────────────────

    def _seed_all(self, seed: int) -> None:
        import torch
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _init_ppo(self) -> None:
        """Initialize active + reference PPO policies."""
        import torch
        import copy
        from core.algorithms.ppo_agent import PPOAgent, PPOConfig

        ppo_config = PPOConfig(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
        )
        if hasattr(ppo_config, "learning_rate"):
            ppo_config.learning_rate = self.config.learning_rate

        self._ppo = PPOAgent(config=ppo_config, device=str(self.device))

        # Load checkpoint
        if self.resume_from:
            logger.info("Resuming from: %s", self.resume_from)
            self._ppo.load(self.resume_from)
        else:
            loaded = self._auto_load_best_checkpoint()
            if loaded:
                logger.info("Auto-loaded checkpoint: %s", loaded)
            else:
                logger.info("No prior checkpoints — training from scratch")

        # Clone reference policy (frozen — no gradients)
        self._ref_ppo = PPOAgent(config=ppo_config, device=str(self.device))
        # Copy weights
        if hasattr(self._ppo, "actor_critic") and hasattr(self._ref_ppo, "actor_critic"):
            self._ref_ppo.actor_critic.load_state_dict(
                self._ppo.actor_critic.state_dict()
            )
        elif hasattr(self._ppo, "policy_net") and hasattr(self._ref_ppo, "policy_net"):
            self._ref_ppo.policy_net.load_state_dict(
                self._ppo.policy_net.state_dict()
            )

        logger.info(
            "GRPO initialized on %s (group_size=%d, clip=%.2f, kl_coef=%.3f, lr=%.1e)",
            self.device, self.config.group_size, self.config.clip_epsilon,
            self.config.kl_coef, self.config.learning_rate,
        )

    def _init_env(self) -> None:
        from core.environment.cyber_environment import CyberEnvironment
        self._env = CyberEnvironment()

    def _init_dirs(self) -> None:
        for d in [CHECKPOINT_DIR, RESULTS_DIR, TRACES_DIR]:
            d.mkdir(parents=True, exist_ok=True)
        self._trace_path = TRACES_DIR / f"grpo_{self._run_id}.jsonl"

        if _HAS_TENSORBOARD:
            tb_dir = TENSORBOARD_DIR / self._run_id
            tb_dir.mkdir(parents=True, exist_ok=True)
            self._tb = SummaryWriter(log_dir=str(tb_dir), flush_secs=30)  # type: ignore[misc]
            console.print(f"[dim]TensorBoard: tensorboard --logdir {TENSORBOARD_DIR}[/dim]")

    def _auto_load_best_checkpoint(self) -> Optional[str]:
        """Auto-load best available .pt checkpoint (persistent memory).

        Priority: latest distilled > enhanced live > enhanced sim.
        """
        import re

        best_path: Optional[str] = None
        best_priority = -1

        # 1. Scan distilled (highest priority — GPU training output)
        distilled_dir = Path("models/distilled")
        if distilled_dir.is_dir():
            pattern = re.compile(r"h200_(.+?)_ep(\d+)\.pt$")
            candidates: list[tuple[Path, str, int]] = []
            for f in distilled_dir.iterdir():
                m = pattern.match(f.name)
                if m:
                    candidates.append((f, m.group(1), int(m.group(2))))
            if candidates:
                latest_run = max(set(c[1] for c in candidates))
                run_cands = [c for c in candidates if c[1] == latest_run]
                best = max(run_cands, key=lambda c: c[2])
                best_path = str(best[0])
                best_priority = 1000 + best[2]

        # 2. Scan enhanced (per-agent PPO checkpoints)
        enhanced_dir = Path("models/enhanced")
        if enhanced_dir.is_dir():
            for entry in sorted(enhanced_dir.iterdir(), reverse=True):
                if not entry.is_dir():
                    continue
                for pt in entry.iterdir():
                    if pt.suffix == ".pt" and pt.name.startswith("ppo_"):
                        pri = 10
                        if "live" in entry.name:
                            pri = 80
                        elif "hard" in entry.name:
                            pri = 30
                        elif "sim" in entry.name:
                            pri = 50
                        if pri > best_priority and best_path is None:
                            best_path = str(pt)
                            best_priority = pri

        if best_path:
            try:
                self._ppo.load(best_path)
                console.print(f"[green]✔ Auto-loaded:[/green] {best_path}")
                return best_path
            except Exception as e:
                logger.warning("Failed to load %s: %s", best_path, e)

        return None

    # ── State encoding ───────────────────────────────────────────

    def _encode_state(self, state: Dict[str, Any], step: int = 0) -> Any:
        from core.models.state_encoder import encode_state
        return encode_state(state, self.device, current_step=step,
                            max_steps=self.config.max_steps_per_episode)

    # ── Action mapping ───────────────────────────────────────────

    def _action_to_command(self, action_idx: int, state: Dict[str, Any]) -> str:
        """Map PPO action index to command string."""
        phase = state.get("phase", "RECON")
        target = state.get("target_ip", "172.28.128.3")
        try:
            from core.algorithms.command_action_mapper import CommandActionMapper
            role = state.get("role", "attacker")
            mapper = CommandActionMapper(role=role)
            tpl = mapper.action_to_command(action_idx)
            if tpl is not None:
                return tpl.template.replace("{target}", target)
        except Exception:
            pass

        # Phase-aware fallbacks
        fallbacks: Dict[int, Dict[str, str]] = {
            0: {"RECON": f"nmap -sV -sC -p- -T4 {target}", "ENUMERATION": f"nmap -sV --script=vuln {target}"},
            1: {"RECON": f"whatweb http://{target}", "ENUMERATION": f"gobuster dir -u http://{target} -w /usr/share/wordlists/dirb/common.txt"},
            2: {"EXPLOITATION": f"msfconsole -q -x 'search type:exploit {target}; exit'", "RECON": "searchsploit --nmap /tmp/nmap_scan.xml"},
            3: {"PRIVILEGE_ESCALATION": "sudo -l", "EXPLOITATION": "find / -perm -4000 -type f 2>/dev/null | head -20"},
            4: {"EXFILTRATION": "tar czf /tmp/loot.tar.gz /home 2>/dev/null", "RECON": f"curl -s http://{target}/robots.txt"},
        }
        phase_map = fallbacks.get(action_idx, {})
        return phase_map.get(phase, next(iter(phase_map.values()), f"nmap -sV -p 80 {target}"))

    # ── Reward decomposition ─────────────────────────────────────

    def _compute_reward(
        self,
        base_reward: float,
        state: Dict[str, Any],
        next_state: Dict[str, Any],
        command: str,
        info: Dict[str, Any],
        step: int,
    ) -> float:
        """Multi-channel weighted reward with consistency signal."""
        w = self.config.reward_weights

        # Format: phase progression
        cur_phase = state.get("phase", "RECON")
        nxt_phase = next_state.get("phase", cur_phase)
        phase_delta = PHASE_REWARDS.get(nxt_phase, 0) - PHASE_REWARDS.get(cur_phase, 0)
        format_r = max(0.0, phase_delta) * w["format"]

        # Code: discovery bonuses
        discoveries = info.get("discoveries", [])
        code_r = 0.0
        for d in (discoveries if isinstance(discoveries, list) else []):
            d_str = str(d).lower() if not isinstance(d, str) else d.lower()
            if "shell" in d_str or "root" in d_str:
                code_r += 10.0
            elif "credential" in d_str or "password" in d_str:
                code_r += 5.0
            elif "service" in d_str or "version" in d_str:
                code_r += 2.0
            elif "port" in d_str:
                code_r += 1.0
            else:
                code_r += 0.5
        code_r *= w["code"]

        # Math: exploitation verification
        math_r = 0.0
        if base_reward > 10.0:
            math_r = min(base_reward * 0.3, 15.0) * w["math"]
        elif base_reward > 2.0:
            math_r = min(base_reward * 0.2, 5.0) * w["math"]

        # Reasoning: consistency
        cmd_family = command.strip().split()[0].rsplit("/", 1)[-1] if command.strip() else "noop"
        aligned = PHASE_ALIGNMENT.get(cur_phase, set())
        align_bonus = 1.5 if cmd_family in aligned else 0.0
        repeat_penalty = 0.0
        if command in self._consistency_recent_cmds:
            repeat_penalty = -2.0
        elif cmd_family in [c.strip().split()[0].rsplit("/", 1)[-1]
                            for c in self._consistency_recent_cmds if c.strip()]:
            repeat_penalty = -0.5
        self._consistency_recent_cmds.append(command)
        if len(self._consistency_recent_cmds) > 5:
            self._consistency_recent_cmds.pop(0)
        reasoning_r = (align_bonus + repeat_penalty) * w["reasoning"]

        total = format_r + code_r + math_r + reasoning_r
        blended = 0.5 * base_reward + 0.5 * total
        return float(np.clip(blended, REWARD_MIN, REWARD_MAX))

    # ── Group Rollout ────────────────────────────────────────────

    def _run_group_rollouts(
        self,
        initial_state: Dict[str, Any],
        episode: int,
    ) -> List[GroupRollout]:
        """Run GROUP_SIZE rollouts from the same initial state.

        Each rollout uses stochastic sampling from the current policy,
        producing different trajectories from the same starting point.
        """
        import torch

        group: List[GroupRollout] = []
        max_steps = self.config.max_steps_per_episode

        for g in range(self.config.group_size):
            rollout = GroupRollout()
            self._env.reset()
            state = dict(initial_state)
            self._consistency_recent_cmds = []

            for step in range(max_steps):
                state_tensor = self._encode_state(state, step)
                with torch.no_grad():
                    action_idx, log_prob, _ = self._ppo.select_action(state_tensor)

                command = self._action_to_command(action_idx, state)
                # Phase 42 fix: pass string directly — dict has no handler in step()
                next_state, base_reward, done, info = self._env.step(command)

                reward = self._compute_reward(
                    base_reward, state, next_state, command, info, step,
                )

                rollout.states.append(state_tensor)
                rollout.actions.append(action_idx)
                rollout.log_probs.append(
                    log_prob.item() if hasattr(log_prob, "item") else float(log_prob)
                )
                rollout.rewards.append(reward)
                rollout.commands.append(command)

                if isinstance(next_state, dict):
                    rollout.discoveries += len(info.get("discoveries", []))
                    rollout.final_phase = next_state.get("phase", "RECON")

                state = next_state if isinstance(next_state, dict) else state
                self.metrics.total_steps += 1

                if done:
                    break

            rollout.total_reward = sum(rollout.rewards)
            group.append(rollout)

        return group

    # ── GRPO Update ──────────────────────────────────────────────

    def _grpo_update(
        self,
        group: List[GroupRollout],
        episode: int,
    ) -> Dict[str, float]:
        """Compute group-relative advantages and update policy.

        GRPO advantage: A_i = (R_i - mean(R)) / max(std(R), 1e-8)
        Loss = -sum(ratio * A * mask) + kl_coef * KL(pi || pi_ref) - entropy_coef * H(pi)

        Returns dict of loss components for logging.
        """
        import torch
        import torch.nn.functional as F

        if not group or not any(r.rewards for r in group):
            return {"loss": 0.0, "kl": 0.0, "advantage_std": 0.0}

        # 1. Compute group-relative advantages
        total_rewards = np.array([r.total_reward for r in group])
        mean_r = float(np.mean(total_rewards))
        std_r = float(np.std(total_rewards))
        if std_r < 1e-8:
            std_r = 1.0  # avoid division by zero

        advantages = [(r - mean_r) / std_r for r in total_rewards]

        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        total_kl = 0.0
        n_updates = 0

        for rollout, advantage in zip(group, advantages):
            if not rollout.states:
                continue

            for t, (state_t, action_t, old_log_prob) in enumerate(
                zip(rollout.states, rollout.actions, rollout.log_probs)
            ):
                # Current policy log prob
                if hasattr(state_t, "unsqueeze"):
                    s = state_t
                else:
                    s = torch.tensor(state_t, device=self.device, dtype=torch.float32)
                if s.dim() == 1:
                    s = s.unsqueeze(0)

                with torch.no_grad():
                    _, ref_log_prob, _ = self._ref_ppo.select_action(s)
                    ref_lp = ref_log_prob.item() if hasattr(ref_log_prob, "item") else float(ref_log_prob)

                _, new_log_prob, _ = self._ppo.select_action(s)
                new_lp = new_log_prob if isinstance(new_log_prob, torch.Tensor) else torch.tensor(
                    new_log_prob, device=self.device, dtype=torch.float32
                )
                old_lp = torch.tensor(old_log_prob, device=self.device, dtype=torch.float32)

                # PPO-style clipped ratio
                ratio = torch.exp(new_lp - old_lp)
                adv_t = torch.tensor(advantage, device=self.device, dtype=torch.float32)
                surr1 = ratio * adv_t
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_epsilon,
                                     1.0 + self.config.clip_epsilon) * adv_t
                policy_loss = -torch.min(surr1, surr2)

                # KL penalty against reference
                kl = old_lp - ref_lp  # approximate KL
                total_kl += abs(float(kl))

                kl_loss = self.config.kl_coef * torch.tensor(
                    abs(kl), device=self.device, dtype=torch.float32,
                )

                step_loss = policy_loss + kl_loss
                total_loss = total_loss + step_loss
                n_updates += 1

        if n_updates > 0:
            avg_loss = total_loss / n_updates
            # Backward + step
            if hasattr(self._ppo, "optimizer") and self._ppo.optimizer is not None:
                self._ppo.optimizer.zero_grad()
                avg_loss.backward()
                # Gradient clipping
                if hasattr(self._ppo, "actor_critic"):
                    torch.nn.utils.clip_grad_norm_(
                        self._ppo.actor_critic.parameters(), 0.5
                    )
                elif hasattr(self._ppo, "policy_net"):
                    torch.nn.utils.clip_grad_norm_(
                        self._ppo.policy_net.parameters(), 0.5
                    )
                self._ppo.optimizer.step()
                self.metrics.total_updates += 1

        avg_kl = total_kl / max(n_updates, 1)
        return {
            "loss": float(total_loss.item()) / max(n_updates, 1),
            "kl": avg_kl,
            "advantage_std": std_r,
            "mean_group_reward": mean_r,
        }

    # ── Reference policy update ──────────────────────────────────

    def _update_reference(self) -> None:
        """Copy active policy weights to reference (hard update)."""
        if hasattr(self._ppo, "actor_critic") and hasattr(self._ref_ppo, "actor_critic"):
            self._ref_ppo.actor_critic.load_state_dict(
                self._ppo.actor_critic.state_dict()
            )
        elif hasattr(self._ppo, "policy_net") and hasattr(self._ref_ppo, "policy_net"):
            self._ref_ppo.policy_net.load_state_dict(
                self._ppo.policy_net.state_dict()
            )
        logger.info("Reference policy updated (episode %d)", self.metrics.episodes_completed)

    # ── Checkpointing ────────────────────────────────────────────

    def _maybe_checkpoint(self, episode: int) -> None:
        now = time.time()
        if now - self._last_checkpoint_time < self.config.checkpoint_interval_sec:
            return

        path = CHECKPOINT_DIR / f"grpo_{self._run_id}_ep{episode:04d}.pt"
        try:
            self._ppo.save(str(path))
            self.metrics.checkpoints_saved += 1
            self._last_checkpoint_time = now
            console.print(f"[green]✔ Checkpoint saved:[/green] {path}")
        except Exception as e:
            logger.warning("Checkpoint failed: %s", e)

    # ── Trace logging ────────────────────────────────────────────

    def _log_trace(self, episode: int, group_stats: Dict[str, float]) -> None:
        if self._trace_path is None:
            return
        entry = {
            "ts": datetime.now(tz=timezone.utc).isoformat(),
            "episode": episode,
            **group_stats,
            "total_steps": self.metrics.total_steps,
        }
        try:
            with open(self._trace_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass

    # ── Display ──────────────────────────────────────────────────

    def _print_episode_summary(
        self, episode: int, group: List[GroupRollout], stats: Dict[str, float],
    ) -> None:
        rewards = [r.total_reward for r in group]
        phases = [r.final_phase for r in group]
        discoveries = [r.discoveries for r in group]

        # Compact per-episode line
        best_r = max(rewards)
        mean_r = float(np.mean(rewards))
        best_phase = max(phases, key=lambda p: PHASES.index(p) if p in PHASES else 0)
        total_disc = sum(discoveries)

        phase_idx = PHASES.index(best_phase) if best_phase in PHASES else 0
        phase_icon = "🔍📋💥👑🔀🏴📤🧹"[min(phase_idx, 7)]

        console.print(
            f"  ep {episode:>4d} │ "
            f"grp_mean={mean_r:+6.1f} best={best_r:+6.1f} │ "
            f"adv_std={stats.get('advantage_std', 0):.3f} "
            f"kl={stats.get('kl', 0):.4f} │ "
            f"{phase_icon} {best_phase[:5]:5s} │ "
            f"disc={total_disc:>3d} │ "
            f"loss={stats.get('loss', 0):.4f}"
        )

    def _print_final_report(self) -> None:
        """Print rich final training report."""
        table = Table(
            title=f"GRPO Training Report — {self._run_id}",
            show_lines=True,
            box=box.ROUNDED,
        )
        table.add_column("Metric", style="cyan", min_width=30)
        table.add_column("Value", style="green")

        m = self.metrics
        table.add_row("Episodes", str(m.episodes_completed))
        table.add_row("Total Steps", f"{m.total_steps:,}")
        table.add_row("Policy Updates", str(m.total_updates))
        table.add_row("Wall Time", f"{m.wall_hours:.2f}h")
        table.add_row("Avg Group Reward", f"{m.avg_group_reward:.2f}")
        table.add_row("Best Episode Reward", f"{m.best_episode_reward:.2f}")
        table.add_row("Avg Advantage Std", f"{m.avg_advantage_std:.3f}")
        table.add_row("Avg KL Divergence", f"{m.avg_kl_divergence:.4f}")
        table.add_row("Checkpoints Saved", str(m.checkpoints_saved))
        table.add_row("Group Size", str(self.config.group_size))
        table.add_row("Learning Rate", f"{self.config.learning_rate:.1e}")
        table.add_row("Device", str(self.device))

        # Phase distribution
        if m.phase_distribution:
            phase_str = ", ".join(
                f"{p}: {c}" for p, c in sorted(
                    m.phase_distribution.items(),
                    key=lambda x: PHASES.index(x[0]) if x[0] in PHASES else 99,
                )
            )
            table.add_row("Phase Distribution", phase_str)

        # Reward weights
        rw_str = ", ".join(f"{k}={v}" for k, v in self.config.reward_weights.items())
        table.add_row("Reward Weights", rw_str)

        console.print(table)

    # ── Main training loop ───────────────────────────────────────

    def run(self) -> GRPOMetrics:
        """Execute GRPO training loop."""
        start_time = time.time()

        # Initialize
        self._init_dirs()
        self._init_ppo()
        self._init_env()
        self._last_checkpoint_time = time.time()

        console.print(Panel(
            f"[bold cyan]GRPO Training[/bold cyan]\n"
            f"Group size: {self.config.group_size} | "
            f"Clip: {self.config.clip_epsilon} | "
            f"KL coef: {self.config.kl_coef}\n"
            f"LR: {self.config.learning_rate:.1e} | "
            f"Max episodes: {self.config.max_episodes} | "
            f"Max hours: {self.config.max_hours}\n"
            f"Device: {self.device} | Seed: {self.seed}",
            title="🚀 GRPO Training Start",
            border_style="cyan",
        ))

        adv_stds: List[float] = []
        kl_values: List[float] = []
        group_rewards: List[float] = []

        for ep in range(self.config.max_episodes):
            # Time check
            elapsed_h = (time.time() - start_time) / 3600
            if elapsed_h >= self.config.max_hours:
                console.print(f"\n[yellow]Time limit reached ({self.config.max_hours}h)[/yellow]")
                break

            # Reset env and get initial state
            initial_state = self._env.reset()
            if not isinstance(initial_state, dict):
                initial_state = {"phase": "RECON", "target_ip": "172.28.128.3"}

            # Run group rollouts
            group = self._run_group_rollouts(initial_state, ep)

            # GRPO update
            stats = self._grpo_update(group, ep)

            # Track metrics
            group_mean = float(np.mean([r.total_reward for r in group]))
            group_rewards.append(group_mean)
            adv_stds.append(stats.get("advantage_std", 0))
            kl_values.append(stats.get("kl", 0))
            self.metrics.reward_history.append(group_mean)

            best_in_group = max(r.total_reward for r in group)
            if best_in_group > self.metrics.best_episode_reward:
                self.metrics.best_episode_reward = best_in_group

            # Phase distribution
            for r in group:
                self.metrics.phase_distribution[r.final_phase] = (
                    self.metrics.phase_distribution.get(r.final_phase, 0) + 1
                )

            self.metrics.episodes_completed += 1

            # Display
            self._print_episode_summary(ep, group, stats)

            # Update reference policy periodically
            if ep > 0 and ep % self.config.reference_update_interval == 0:
                self._update_reference()

            # Checkpoint
            self._maybe_checkpoint(ep)

            # Log trace
            self._log_trace(ep, stats)

            # TensorBoard
            if self._tb is not None:
                self._tb.add_scalar("grpo/group_mean_reward", group_mean, ep)
                self._tb.add_scalar("grpo/advantage_std", stats.get("advantage_std", 0), ep)
                self._tb.add_scalar("grpo/kl_divergence", stats.get("kl", 0), ep)
                self._tb.add_scalar("grpo/loss", stats.get("loss", 0), ep)

        # Final metrics
        self.metrics.wall_hours = (time.time() - start_time) / 3600
        self.metrics.avg_group_reward = float(np.mean(group_rewards)) if group_rewards else 0.0
        self.metrics.avg_advantage_std = float(np.mean(adv_stds)) if adv_stds else 0.0
        self.metrics.avg_kl_divergence = float(np.mean(kl_values)) if kl_values else 0.0

        # Final checkpoint
        final_path = CHECKPOINT_DIR / f"grpo_{self._run_id}_final.pt"
        try:
            self._ppo.save(str(final_path))
            console.print(f"[green]✔ Final checkpoint:[/green] {final_path}")
        except Exception as e:
            logger.warning("Final checkpoint failed: %s", e)

        # Save report
        report_path = RESULTS_DIR / f"grpo_{self._run_id}_report.json"
        try:
            with open(report_path, "w") as f:
                json.dump(asdict(self.metrics), f, indent=2, default=str)
            console.print(f"[green]✔ Report saved:[/green] {report_path}")
        except Exception as e:
            logger.warning("Report save failed: %s", e)

        # Close TensorBoard
        if self._tb is not None:
            self._tb.close()

        self._print_final_report()
        return self.metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GRPO-Style Group Relative Policy Optimization for Ariaska RL",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-hours", type=float, default=DEFAULT_MAX_HOURS)
    parser.add_argument("--max-episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS_PER_EPISODE)
    parser.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE,
                        help=f"Rollouts per state (default: {DEFAULT_GROUP_SIZE})")
    parser.add_argument("--clip-epsilon", type=float, default=DEFAULT_CLIP_EPSILON)
    parser.add_argument("--kl-coef", type=float, default=DEFAULT_KL_COEF)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE,
                        help="GRPO learning rate (default: 1e-4)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_SEC,
                        help="Checkpoint interval in seconds")
    parser.add_argument("--reward-weights", type=str, default=None,
                        help="Reward weights: 'format=2.0,code=1.5,math=1.5,reasoning=0.5'")
    parser.add_argument("--reference-update", type=int, default=DEFAULT_REFERENCE_UPDATE,
                        help=f"Update reference policy every N episodes (default: {DEFAULT_REFERENCE_UPDATE})")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Parse reward weights
    rw: Optional[Dict[str, float]] = None
    if args.reward_weights:
        rw = {}
        for pair in args.reward_weights.split(","):
            k, v = pair.strip().split("=")
            rw[k.strip()] = float(v.strip())

    config = GRPOConfig(
        group_size=args.group_size,
        clip_epsilon=args.clip_epsilon,
        kl_coef=args.kl_coef,
        learning_rate=args.learning_rate,
        max_steps_per_episode=args.max_steps,
        max_episodes=args.max_episodes,
        max_hours=args.max_hours,
        checkpoint_interval_sec=float(args.checkpoint_every),
        reference_update_interval=args.reference_update,
    )
    if rw:
        config.reward_weights.update(rw)

    trainer = GRPOTrainer(
        config=config,
        seed=args.seed,
        resume_from=args.resume,
        device=args.device,
    )

    metrics = trainer.run()

    if metrics.episodes_completed > 0:
        console.print(
            f"\n[green]✓ GRPO complete: {metrics.episodes_completed} episodes, "
            f"{metrics.total_updates} updates, {metrics.wall_hours:.2f}h[/green]"
        )
        sys.exit(0)
    else:
        console.print("\n[red]✗ No episodes completed[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
