#!/usr/bin/env python3
"""
scripts/offline_distill.py — Phase 47: Offline Distillation Trainer

Pure offline training that distills LLM mentor knowledge into the PPO/DDQN
agents WITHOUT any live LLM calls. Uses traces + checkpoints from prior
GPU training runs.

What this does to YOUR AGENT (not the LLM):
  1. BC (Behavioral Cloning):   PPO learns to mimic teacher actions
  2. KL Teacher Distillation:   PPO policy aligns with teacher distribution
  3. SIL (Self-Imitation):      PPO replays its own high-reward episodes
  4. CQL Regularization:        DDQN avoids out-of-distribution actions
  5. Advantage-Weighted BC:     Only clone teacher actions with positive advantage
  6. Cross-episode replay:      Learn from ALL historical traces

Zero API calls. Zero LLM. Pure neural network training.

Usage:
    # Basic: load latest checkpoint + all traces, run 100 offline epochs
    python -m scripts.offline_distill --checkpoint latest --epochs 100

    # Full: specify checkpoint + trace dir + device
    python -m scripts.offline_distill \\
        --checkpoint models/unified/ariaska_20260222T005136Z_ep0003.pt \\
        --traces traces/h200_distill/ \\
        --epochs 200 \\
        --batch-size 32 \\
        --device cuda \\
        --seed 42

    # With CQL for DDQN:
    python -m scripts.offline_distill --checkpoint latest --epochs 100 --cql

Author: Phase 47 — Offline Distillation
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("ERROR: PyTorch required. Install with: pip install torch")
    sys.exit(1)

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

console = Console()
logger = logging.getLogger("ariaska.offline_distill")


# ── Constants ────────────────────────────────────────────────────
STATE_DIM = 512
ACTION_DIM = 5
NUM_MACROS = 9              # DDQN macro intents
PHASES = ["RECON", "ENUMERATION", "EXPLOITATION", "PRIVILEGE_ESCALATION",
          "LATERAL_MOVEMENT", "POST_EXPLOITATION", "EXFILTRATION", "CLOSEOUT"]
PHASE_TO_GROUP = {
    "RECON": 0, "ENUMERATION": 0,
    "EXPLOITATION": 1, "PRIVILEGE_ESCALATION": 1,
    "LATERAL_MOVEMENT": 2, "POST_EXPLOITATION": 2,
    "EXFILTRATION": 2, "CLOSEOUT": 2,
}

# Default paths
CHECKPOINT_DIR = Path("models/unified")
TRACES_DIR = Path("traces/h200_distill")
RESULTS_DIR = Path("results/offline_distill")
OFFLINE_CHECKPOINT_DIR = Path("models/offline_distill")


@dataclass
class OfflineConfig:
    """Configuration for offline distillation training."""
    # Data
    checkpoint_path: Optional[str] = None
    traces_dir: str = str(TRACES_DIR)
    # Training
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4         # Lower than online (stability)
    lr_min: float = 1e-6
    weight_decay: float = 1e-5
    max_grad_norm: float = 0.5
    # Loss weights
    bc_coef: float = 0.30               # Behavioral cloning weight
    kl_coef: float = 0.20               # KL teacher distillation
    sil_coef: float = 0.25              # Self-imitation learning
    ppo_coef: float = 0.15              # PPO surrogate (offline variant)
    ranking_coef: float = 0.05          # Ranking margin loss
    value_reg_coef: float = 0.05        # Value function regularization
    # Advanced
    advantage_weighted_bc: bool = True  # Weight BC by advantage
    advantage_clip: float = 4.0
    min_reward_for_sil: float = 5.0     # SIL threshold
    gamma: float = 0.99
    gae_lambda: float = 0.97
    # CQL (for DDQN)
    use_cql: bool = False
    cql_alpha: float = 1.0              # CQL regularization weight
    cql_temp: float = 1.0               # CQL logsumexp temperature
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    # Checkpointing
    save_every: int = 10                # Save every N epochs
    eval_every: int = 5                 # Evaluate every N epochs


class TraceDataset(Dataset):
    """PyTorch Dataset from JSONL trace files.

    Loads all step traces with state_vector data into memory.
    For traces WITHOUT state_vector (older runs), reconstructs a
    minimal state from available metadata.
    """

    def __init__(self, traces_dir: str, min_reward_for_sil: float = 5.0):
        self.steps: List[Dict[str, Any]] = []
        self.episodes: Dict[int, List[Dict]] = defaultdict(list)
        self.sil_episodes: List[int] = []  # High-reward episodes for SIL
        self._episode_rewards: Dict[int, float] = defaultdict(float)
        self._load_traces(traces_dir)
        self._identify_sil_episodes(min_reward_for_sil)

    def _load_traces(self, traces_dir: str) -> None:
        """Load all JSONL trace files from directory."""
        traces_path = Path(traces_dir)
        if not traces_path.exists():
            logger.warning("Traces dir %s not found", traces_dir)
            return

        jsonl_files = sorted(traces_path.glob("*.jsonl"))
        console.print(f"  Loading {len(jsonl_files)} trace files from {traces_dir}")

        total_steps = 0
        total_with_state = 0
        total_with_teacher = 0

        for fpath in jsonl_files:
            try:
                with open(fpath) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        data = record.get("data", record)
                        if data.get("kind") != "step":
                            continue

                        step = {
                            "episode": data.get("episode", 0),
                            "step": data.get("step", 0),
                            "action_idx": data.get("action_idx", 0),
                            "reward": float(data.get("reward", 0.0)),
                            "phase": data.get("phase", "RECON"),
                            "done": data.get("done", False),
                            "teacher_action": data.get("teacher_action", -1),
                            "teacher_overrode": data.get("teacher_overrode", False),
                            "cmd_family": data.get("cmd_family", ""),
                            "command": data.get("command", ""),
                            "mentor_queried": data.get("mentor_queried", False),
                            # Phase 47 fields (may be None for older traces)
                            "state_vector": data.get("state_vector"),
                            "log_prob": float(data.get("log_prob", 0.0)),
                            "value": float(data.get("value", 0.0)),
                            "teacher_dist": data.get("teacher_dist"),
                        }

                        total_steps += 1
                        if step["state_vector"] is not None:
                            total_with_state += 1
                        if step["teacher_dist"] is not None:
                            total_with_teacher += 1

                        ep_id = step["episode"]
                        self.episodes[ep_id].append(step)
                        self._episode_rewards[ep_id] += step["reward"]

            except Exception as e:
                logger.warning("Error loading %s: %s", fpath, e)

        # Flatten episodes into step list (sorted by episode, step)
        for ep_id in sorted(self.episodes.keys()):
            ep_steps = sorted(self.episodes[ep_id], key=lambda s: s["step"])
            self.steps.extend(ep_steps)

        console.print(f"  Loaded {total_steps} steps, {len(self.episodes)} episodes")
        console.print(f"  With state_vector: {total_with_state}/{total_steps}")
        console.print(f"  With teacher_dist: {total_with_teacher}/{total_steps}")

        if total_with_state == 0:
            console.print("[yellow]  WARNING: No state vectors in traces. "
                          "Will use synthetic states (reduced quality).[/yellow]")

    def _identify_sil_episodes(self, min_reward: float) -> None:
        """Identify high-reward episodes for SIL."""
        for ep_id, total_reward in self._episode_rewards.items():
            avg_reward = total_reward / max(len(self.episodes[ep_id]), 1)
            if avg_reward >= min_reward:
                self.sil_episodes.append(ep_id)

        console.print(f"  SIL episodes (avg_reward >= {min_reward}): "
                      f"{len(self.sil_episodes)}/{len(self.episodes)}")

    def _build_state_tensor(self, step: Dict) -> torch.Tensor:
        """Build state tensor from step data."""
        if step["state_vector"] is not None:
            return torch.tensor(step["state_vector"], dtype=torch.float32)

        # Reconstruct minimal state from metadata
        state = torch.zeros(STATE_DIM, dtype=torch.float32)
        phase = step.get("phase", "RECON")
        if phase in PHASES:
            phase_idx = PHASES.index(phase)
            state[phase_idx] = 1.0          # Phase one-hot
            state[10] = phase_idx / 7.0     # Phase progress
        state[59] = step.get("step", 0) / 150.0   # Step progress
        state[60] = step.get("reward", 0.0) / 50.0  # Normalized reward
        return state

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        step = self.steps[idx]
        state = self._build_state_tensor(step)
        phase_group = PHASE_TO_GROUP.get(step["phase"], 0)

        item = {
            "state": state,
            "action": torch.tensor(step["action_idx"], dtype=torch.long),
            "reward": torch.tensor(step["reward"], dtype=torch.float32),
            "phase_group": torch.tensor(phase_group, dtype=torch.long),
            "log_prob": torch.tensor(step["log_prob"], dtype=torch.float32),
            "value": torch.tensor(step["value"], dtype=torch.float32),
            "done": torch.tensor(step["done"], dtype=torch.float32),
            "has_state_vector": torch.tensor(
                1.0 if step["state_vector"] is not None else 0.0,
                dtype=torch.float32,
            ),
            "teacher_action": torch.tensor(
                step["teacher_action"] if step["teacher_action"] >= 0 else step["action_idx"],
                dtype=torch.long,
            ),
            "teacher_overrode": torch.tensor(
                1.0 if step["teacher_overrode"] else 0.0,
                dtype=torch.float32,
            ),
        }

        # Teacher distribution (action_dim probability vector)
        if step["teacher_dist"] is not None:
            item["teacher_dist"] = torch.tensor(
                step["teacher_dist"], dtype=torch.float32,
            )
            item["has_teacher_dist"] = torch.tensor(1.0)
        else:
            item["teacher_dist"] = torch.zeros(ACTION_DIM, dtype=torch.float32)
            item["has_teacher_dist"] = torch.tensor(0.0)

        return item

    def get_sil_batch(self, batch_size: int) -> Optional[Dict[str, torch.Tensor]]:
        """Sample a batch from high-reward episodes for SIL."""
        if not self.sil_episodes:
            return None

        samples = []
        for _ in range(batch_size):
            ep_id = random.choice(self.sil_episodes)
            ep_steps = self.episodes[ep_id]
            if ep_steps:
                step = random.choice(ep_steps)
                samples.append(step)

        if not samples:
            return None

        states = torch.stack([self._build_state_tensor(s) for s in samples])
        actions = torch.tensor([s["action_idx"] for s in samples], dtype=torch.long)
        rewards = torch.tensor([s["reward"] for s in samples], dtype=torch.float32)

        return {"states": states, "actions": actions, "rewards": rewards}


class OfflineDistillTrainer:
    """Pure offline trainer that distills LLM knowledge into PPO/DDQN agents.

    Loads a checkpoint (PPO network weights) and trace data, then runs
    multiple offline training epochs with:
      - BC loss (clone teacher actions)
      - KL teacher distillation (match teacher distribution)
      - SIL (replay high-reward episodes)
      - Ranking margin loss (teacher action > others)
      - Value regularization
      - Optional CQL for DDQN
    """

    def __init__(self, config: OfflineConfig):
        self.config = config
        self.device = torch.device(config.device)
        self._seed_all(config.seed)
        self._run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

        # Create output dirs
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        OFFLINE_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

        # Load PPO network from checkpoint
        self.ppo = self._load_ppo(config.checkpoint_path)
        self.ddqn = self._load_ddqn(config.checkpoint_path) if config.use_cql else None

        # Load trace data
        self.dataset = TraceDataset(
            config.traces_dir,
            min_reward_for_sil=config.min_reward_for_sil,
        )
        if len(self.dataset) > 0:
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=config.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=0,
            )
        else:
            self.dataloader = None

        # Optimizer: separate from PPO's internal optimizer for clean offline training
        self.optimizer = torch.optim.AdamW(
            self.ppo.network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=config.lr_min,
        )

        # DDQN optimizer (if CQL)
        if self.ddqn is not None:
            self.ddqn_optimizer = torch.optim.AdamW(
                self.ddqn.online_net.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )

        # Metrics
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self._best_total_loss = float("inf")
        self._epochs_without_improvement = 0

    def _seed_all(self, seed: int) -> None:
        """Reproducible training."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _load_ppo(self, checkpoint_path: Optional[str]) -> Any:
        """Load PPO agent from checkpoint."""
        # Lazy import to avoid circular deps
        from core.algorithms.ppo_agent import PPOAgent, PPOConfig

        if checkpoint_path and checkpoint_path != "latest":
            cp_path = Path(checkpoint_path)
        else:
            # Find latest checkpoint
            cp_path = self._find_latest_checkpoint()

        if cp_path is None or not cp_path.exists():
            console.print("[yellow]No checkpoint found — initializing fresh PPO[/yellow]")
            config = PPOConfig(
                state_dim=STATE_DIM,
                action_dim=ACTION_DIM,
                use_bc_loss=True,
                bc_loss_coef=self.config.bc_coef,
                use_kl_teacher_loss=True,
                kl_teacher_coef=self.config.kl_coef,
                use_ranking_loss=True,
                ranking_loss_coef=self.config.ranking_coef,
                sil_coef=self.config.sil_coef,
            )
            return PPOAgent(config=config, device=str(self.device))

        console.print(f"  Loading PPO from: {cp_path}")
        ckpt = torch.load(cp_path, map_location=self.device, weights_only=False)

        # Handle unified checkpoint format
        ppo_state = ckpt.get("ppo_state", ckpt)
        ppo_config_data = ppo_state.get("config", {})

        if isinstance(ppo_config_data, PPOConfig):
            config = ppo_config_data
        elif isinstance(ppo_config_data, dict):
            config = PPOConfig(**{
                k: v for k, v in ppo_config_data.items()
                if k in PPOConfig.__dataclass_fields__
            })
        else:
            config = PPOConfig(state_dim=STATE_DIM, action_dim=ACTION_DIM)

        # Enable distillation losses
        config.use_bc_loss = True
        config.bc_loss_coef = self.config.bc_coef
        config.use_kl_teacher_loss = True
        config.kl_teacher_coef = self.config.kl_coef
        config.use_ranking_loss = True
        config.ranking_loss_coef = self.config.ranking_coef
        config.sil_coef = self.config.sil_coef

        ppo = PPOAgent(config=config, device=str(self.device))
        ppo.load_from_state_dict(ppo_state)

        total_params = sum(p.numel() for p in ppo.network.parameters())
        trainable = sum(p.numel() for p in ppo.network.parameters() if p.requires_grad)
        console.print(f"  PPO: {total_params:,} params ({trainable:,} trainable)")

        return ppo

    def _load_ddqn(self, checkpoint_path: Optional[str]) -> Optional[Any]:
        """Load DDQN macro agent if available in checkpoint."""
        try:
            from core.algorithms.ddqn_macro import DDQNMacro, DDQNConfig
        except ImportError:
            logger.warning("DDQN not available")
            return None

        config = DDQNConfig(state_dim=STATE_DIM, num_macros=NUM_MACROS)
        ddqn = DDQNMacro(config=config)
        ddqn.online_net.to(self.device)
        ddqn.target_net.to(self.device)

        # Try loading from checkpoint
        if checkpoint_path and checkpoint_path != "latest":
            cp_path = Path(checkpoint_path)
        else:
            cp_path = self._find_latest_checkpoint()

        if cp_path and cp_path.exists():
            ckpt = torch.load(cp_path, map_location=self.device, weights_only=False)
            ddqn_state = ckpt.get("ddqn_state")
            if ddqn_state:
                try:
                    ddqn.load_state_dict(ddqn_state)
                    console.print("  DDQN: loaded from checkpoint")
                except Exception as e:
                    logger.warning("DDQN load failed: %s", e)

        return ddqn

    def _find_latest_checkpoint(self) -> Optional[Path]:
        """Find the most recent unified checkpoint."""
        if not CHECKPOINT_DIR.exists():
            return None
        pts = sorted(CHECKPOINT_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime)
        return pts[-1] if pts else None

    # ── Loss Functions ───────────────────────────────────────────

    def _bc_loss(
        self,
        logits: torch.Tensor,
        teacher_actions: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Behavioral cloning loss: NLL of teacher actions under policy.

        If advantage_weighted_bc is enabled, weight BC loss by positive
        advantages — only clone teacher actions that were actually good.
        """
        log_probs = F.log_softmax(logits, dim=-1)
        nll = F.nll_loss(log_probs, teacher_actions, reduction="none")

        if weights is not None:
            nll = nll * weights

        return nll.mean()

    def _kl_teacher_loss(
        self,
        logits: torch.Tensor,
        teacher_dist: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """KL divergence: policy → teacher distribution.

        Only computed for steps where teacher distribution is available.
        """
        if mask.sum() < 1:
            return torch.tensor(0.0, device=self.device)

        policy_logprobs = F.log_softmax(logits[mask.bool()], dim=-1)
        teacher_probs = teacher_dist[mask.bool()]

        # Clamp teacher probs to avoid log(0)
        teacher_probs = teacher_probs.clamp(min=1e-8)
        teacher_logprobs = teacher_probs.log()

        # KL(teacher || policy) = sum(teacher * (log_teacher - log_policy))
        kl = F.kl_div(policy_logprobs, teacher_probs, reduction="batchmean",
                       log_target=False)
        return kl

    def _ranking_loss(
        self,
        logits: torch.Tensor,
        teacher_actions: torch.Tensor,
        margin: float = 1.0,
    ) -> torch.Tensor:
        """Margin-based ranking: teacher action should have higher logit.

        loss = max(0, margin - (logit[teacher] - max(logit[non-teacher])))
        """
        batch_size = logits.size(0)
        teacher_logits = logits.gather(1, teacher_actions.unsqueeze(1)).squeeze(1)

        # Max logit among non-teacher actions
        mask_teacher = torch.zeros_like(logits).scatter_(
            1, teacher_actions.unsqueeze(1), 1.0,
        )
        non_teacher_logits = logits.masked_fill(mask_teacher.bool(), -1e9)
        max_non_teacher = non_teacher_logits.max(dim=1).values

        loss = F.relu(margin - (teacher_logits - max_non_teacher))
        return loss.mean()

    def _sil_loss(
        self,
        sil_batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Self-Imitation Learning: imitate own high-reward episodes.

        Only takes gradient from positive advantages.
        """
        states = sil_batch["states"].to(self.device)
        actions = sil_batch["actions"].to(self.device)
        rewards = sil_batch["rewards"].to(self.device)

        logits, values = self.ppo.network(states)
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Advantage = return - value (only positive)
        advantages = (rewards - values.detach().squeeze()).clamp(min=0)

        policy_loss = -(action_log_probs * advantages).mean()
        value_loss = F.smooth_l1_loss(values.squeeze(), rewards)

        return policy_loss + 0.5 * value_loss

    def _cql_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Conservative Q-Learning regularization for DDQN.

        Pushes down Q-values for out-of-distribution actions while
        keeping Q-values for in-distribution (data) actions stable.

        CQL = α * [log(sum_a exp(Q(s,a))) - Q(s, a_data)]
        """
        if self.ddqn is None:
            return torch.tensor(0.0, device=self.device)

        q_values = self.ddqn.online_net(states)

        # Q-value for data actions
        q_data = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # logsumexp over all actions (soft max Q)
        logsumexp_q = torch.logsumexp(
            q_values / self.config.cql_temp, dim=1,
        ) * self.config.cql_temp

        cql_reg = (logsumexp_q - q_data).mean()
        return self.config.cql_alpha * cql_reg

    # ── Training Loop ────────────────────────────────────────────

    def train(self) -> Dict[str, Any]:
        """Run offline distillation training."""
        console.print(Panel.fit(
            f"[bold cyan]Offline Distillation Trainer[/bold cyan]\n"
            f"Epochs: {self.config.epochs} | Batch: {self.config.batch_size} | "
            f"LR: {self.config.learning_rate}\n"
            f"BC: {self.config.bc_coef} | KL: {self.config.kl_coef} | "
            f"SIL: {self.config.sil_coef} | Ranking: {self.config.ranking_coef}\n"
            f"Dataset: {len(self.dataset)} steps, "
            f"{len(self.dataset.episodes)} episodes\n"
            f"SIL episodes: {len(self.dataset.sil_episodes)} | "
            f"CQL: {'ON' if self.config.use_cql else 'OFF'}\n"
            f"Device: {self.device}",
            title="Phase 47",
        ))

        if len(self.dataset) == 0 or self.dataloader is None:
            console.print("[red]No trace data found! Cannot train.[/red]")
            return {"error": "no_data"}

        self.ppo.network.train()
        start_time = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("Training...", total=self.config.epochs)

            for epoch in range(self.config.epochs):
                epoch_metrics = self._train_epoch(epoch)

                # Log
                for k, v in epoch_metrics.items():
                    self.metrics[k].append(v)

                # Progress display
                total_loss = epoch_metrics.get("total_loss", 0.0)
                bc_loss = epoch_metrics.get("bc_loss", 0.0)
                kl_loss = epoch_metrics.get("kl_loss", 0.0)
                sil_loss = epoch_metrics.get("sil_loss", 0.0)

                progress.update(
                    task, advance=1,
                    description=(
                        f"E{epoch:03d} | loss={total_loss:.4f} "
                        f"BC={bc_loss:.4f} KL={kl_loss:.4f} SIL={sil_loss:.4f} "
                        f"LR={self.scheduler.get_last_lr()[0]:.2e}"
                    ),
                )

                # Eval
                if (epoch + 1) % self.config.eval_every == 0:
                    self._eval_epoch(epoch)

                # Checkpoint
                if (epoch + 1) % self.config.save_every == 0:
                    self._save_checkpoint(epoch)

                # Early stopping check
                if total_loss < self._best_total_loss:
                    self._best_total_loss = total_loss
                    self._epochs_without_improvement = 0
                else:
                    self._epochs_without_improvement += 1

                if self._epochs_without_improvement >= 20:
                    console.print(
                        f"[yellow]Early stopping at epoch {epoch} "
                        f"(20 epochs without improvement)[/yellow]"
                    )
                    break

                self.scheduler.step()

        elapsed = time.time() - start_time

        # Final save
        self._save_checkpoint(self.config.epochs - 1, is_final=True)

        # Report
        report = self._build_report(elapsed)
        self._save_report(report)
        self._print_summary(report)

        return report

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch over all trace data."""
        epoch_losses = defaultdict(float)
        num_batches = 0

        for batch in self.dataloader:  # pyright: ignore[reportOptionalIterable]
            # Move to device
            states = batch["state"].to(self.device)
            actions = batch["action"].to(self.device)
            rewards = batch["reward"].to(self.device)
            teacher_actions = batch["teacher_action"].to(self.device)
            teacher_dist = batch["teacher_dist"].to(self.device)
            has_teacher = batch["has_teacher_dist"].to(self.device)
            has_state = batch["has_state_vector"].to(self.device)
            phase_groups = batch["phase_group"].to(self.device)
            teacher_overrode = batch["teacher_overrode"].to(self.device)
            values_old = batch["value"].to(self.device)

            # Forward pass
            logits, values = self.ppo.network(states, phase_group=phase_groups)

            # ── 1. BC Loss ───────────────────────────────────────
            # Only learn from steps where teacher actually intervened
            bc_mask = teacher_overrode > 0.5
            if bc_mask.sum() > 0:
                bc_logits = logits[bc_mask]
                bc_targets = teacher_actions[bc_mask]

                # Advantage-weighted BC: weight by positive advantage
                if self.config.advantage_weighted_bc:
                    bc_rewards = rewards[bc_mask]
                    bc_values = values[bc_mask].detach().squeeze(-1)
                    advantages = (bc_rewards - bc_values).clamp(min=0.0)
                    # Normalize advantages
                    if advantages.numel() > 1 and advantages.std() > 1e-8:
                        advantages = advantages / (advantages.std() + 1e-8)
                    bc_weights = advantages.clamp(min=0.1)  # Floor weight
                else:
                    bc_weights = None

                bc_loss = self._bc_loss(bc_logits, bc_targets, bc_weights)
            else:
                bc_loss = torch.tensor(0.0, device=self.device)

            # ── 2. KL Teacher Distillation ───────────────────────
            kl_loss = self._kl_teacher_loss(logits, teacher_dist, has_teacher)

            # ── 3. Ranking Margin Loss ───────────────────────────
            ranking_loss = self._ranking_loss(logits, teacher_actions)

            # ── 4. Value Regularization ──────────────────────────
            # Encourage critic to match the observed rewards
            value_loss = F.smooth_l1_loss(values.squeeze(-1), rewards)

            # ── 5. SIL Loss ──────────────────────────────────────
            sil_batch = self.dataset.get_sil_batch(min(16, self.config.batch_size))
            if sil_batch is not None:
                sil_loss = self._sil_loss(sil_batch)
            else:
                sil_loss = torch.tensor(0.0, device=self.device)

            # ── 6. CQL Loss (DDQN) ──────────────────────────────
            cql_loss = torch.tensor(0.0, device=self.device)
            if self.config.use_cql and self.ddqn is not None:
                # Map PPO actions to DDQN macro space (simplified)
                macro_actions = actions.clamp(max=NUM_MACROS - 1)
                q_values = self.ddqn.online_net(states.detach())
                q_data = q_values.gather(1, macro_actions.unsqueeze(1)).squeeze(1)
                logsumexp_q = torch.logsumexp(
                    q_values / self.config.cql_temp, dim=1,
                ) * self.config.cql_temp
                cql_loss = self.config.cql_alpha * (logsumexp_q - q_data).mean()

            # ── Phase prediction auxiliary loss ──────────────────
            # predict_phase() uses features cached from the forward() above
            phase_pred_loss = torch.tensor(0.0, device=self.device)
            if hasattr(self.ppo.network, 'predict_phase'):
                phase_logits = self.ppo.network.predict_phase(states)
                if (phase_logits is not None
                        and phase_groups.max() < phase_logits.size(-1)):
                    phase_pred_loss = F.cross_entropy(phase_logits, phase_groups)

            # ── Total Loss ───────────────────────────────────────
            total_loss = (
                self.config.bc_coef * bc_loss
                + self.config.kl_coef * kl_loss
                + self.config.ranking_coef * ranking_loss
                + self.config.value_reg_coef * value_loss
                + self.config.sil_coef * sil_loss
                + cql_loss
                + 0.05 * phase_pred_loss
            )

            # Backward + clip + step
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(
                self.ppo.network.parameters(),
                self.config.max_grad_norm,
            )
            self.optimizer.step()

            # CQL backward (separate optimizer, separate forward pass)
            if self.config.use_cql and self.ddqn is not None and cql_loss.item() > 0:
                self.ddqn_optimizer.zero_grad()
                # Re-compute CQL loss with fresh graph for DDQN backward
                q_vals = self.ddqn.online_net(states.detach())
                q_d = q_vals.gather(1, actions.clamp(max=NUM_MACROS - 1).unsqueeze(1)).squeeze(1)
                lse_q = torch.logsumexp(q_vals / self.config.cql_temp, dim=1) * self.config.cql_temp
                cql_loss_ddqn = self.config.cql_alpha * (lse_q - q_d).mean()
                cql_loss_ddqn.backward()
                nn.utils.clip_grad_norm_(
                    self.ddqn.online_net.parameters(),
                    self.config.max_grad_norm,
                )
                self.ddqn_optimizer.step()

                # Soft target update
                tau = 0.005
                for tp, op in zip(
                    self.ddqn.target_net.parameters(),
                    self.ddqn.online_net.parameters(),
                ):
                    tp.data.copy_(tau * op.data + (1 - tau) * tp.data)

            # Track losses
            epoch_losses["total_loss"] += total_loss.item()
            epoch_losses["bc_loss"] += bc_loss.item()
            epoch_losses["kl_loss"] += kl_loss.item()
            epoch_losses["ranking_loss"] += ranking_loss.item()
            epoch_losses["value_loss"] += value_loss.item()
            epoch_losses["sil_loss"] += sil_loss.item()
            epoch_losses["cql_loss"] += cql_loss.item()
            epoch_losses["phase_pred_loss"] += phase_pred_loss.item()
            num_batches += 1

        # Average
        if num_batches > 0:
            for k in epoch_losses:
                epoch_losses[k] /= num_batches

        epoch_losses["lr"] = self.scheduler.get_last_lr()[0]
        epoch_losses["epoch"] = epoch
        return dict(epoch_losses)

    @torch.no_grad()
    def _eval_epoch(self, epoch: int) -> None:
        """Evaluate policy quality on trace data."""
        self.ppo.network.eval()

        correct = 0
        total = 0
        total_value_error = 0.0

        for batch in self.dataloader:  # pyright: ignore[reportOptionalIterable]
            states = batch["state"].to(self.device)
            teacher_actions = batch["teacher_action"].to(self.device)
            rewards = batch["reward"].to(self.device)
            teacher_overrode = batch["teacher_overrode"].to(self.device)
            phase_groups = batch["phase_group"].to(self.device)

            logits, values = self.ppo.network(states, phase_group=phase_groups)
            preds = logits.argmax(dim=-1)

            # Only count accuracy on teacher-overridden steps
            mask = teacher_overrode > 0.5
            if mask.sum() > 0:
                correct += (preds[mask] == teacher_actions[mask]).sum().item()
                total += mask.sum().item()

            total_value_error += F.mse_loss(
                values.squeeze(-1), rewards,
            ).item() * states.size(0)

        acc = correct / max(total, 1)
        avg_value_err = total_value_error / max(len(self.dataset), 1)

        console.print(
            f"  [cyan]Eval E{epoch:03d}[/cyan]: "
            f"teacher_match={acc:.1%} ({correct}/{total}), "
            f"value_mse={avg_value_err:.4f}"
        )

        self.metrics["eval_teacher_accuracy"].append(acc)
        self.metrics["eval_value_mse"].append(avg_value_err)

        self.ppo.network.train()

    def _save_checkpoint(self, epoch: int, is_final: bool = False) -> None:
        """Save offline distillation checkpoint."""
        tag = "final" if is_final else f"ep{epoch:04d}"
        path = OFFLINE_CHECKPOINT_DIR / f"offline_{self._run_id}_{tag}.pt"

        save_dict = {
            "__offline_distill__": True,
            "format_version": 1,
            "run_id": self._run_id,
            "epoch": epoch,
            "config": self.config,
            "ppo_state": {
                "network_state_dict": self.ppo.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "config": self.ppo.config,
                "total_steps": self.ppo.total_steps,
                "updates_done": self.ppo.updates_done,
                "entropy_coef": self.ppo.entropy_coef,
                "training_metrics": self.ppo.training_metrics,
                "reward_norm": {
                    "mean": self.ppo._reward_mean,
                    "var": self.ppo._reward_var,
                    "count": self.ppo._reward_count,
                    "return_mean": self.ppo._return_mean,
                    "return_var": self.ppo._return_var,
                    "return_count": self.ppo._return_count,
                },
                "has_phase_gates": getattr(self.ppo.network, 'has_phase_gates', False),
                "sil_baseline": self.ppo.sil_buffer._return_baseline,
                "sil_count": self.ppo.sil_buffer._return_count,
                "use_symlog": self.ppo.config.use_symlog,
                "use_cosine_entropy": self.ppo.config.use_cosine_entropy,
                "ema_network_state": (
                    self.ppo.ema_network.state_dict()
                    if self.ppo.ema_network else None
                ),
                "clip_epsilon_current": self.ppo.config.clip_epsilon,
                "clip_fraction_history": self.ppo._clip_fraction_history,
                "entropy_below_count": self.ppo._entropy_below_count,
            },
            "metrics": dict(self.metrics),
        }

        if self.ddqn is not None:
            save_dict["ddqn_state"] = {
                "online_net": self.ddqn.online_net.state_dict(),
                "target_net": self.ddqn.target_net.state_dict(),
            }

        torch.save(save_dict, path)
        console.print(f"  Saved: {path}")

    def _build_report(self, elapsed: float) -> Dict[str, Any]:
        """Build final training report."""
        return {
            "run_id": self._run_id,
            "type": "offline_distillation",
            "elapsed_hours": elapsed / 3600,
            "config": {
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "bc_coef": self.config.bc_coef,
                "kl_coef": self.config.kl_coef,
                "sil_coef": self.config.sil_coef,
                "ranking_coef": self.config.ranking_coef,
                "use_cql": self.config.use_cql,
                "advantage_weighted_bc": self.config.advantage_weighted_bc,
                "device": self.config.device,
                "seed": self.config.seed,
            },
            "data": {
                "total_steps": len(self.dataset),
                "total_episodes": len(self.dataset.episodes),
                "sil_episodes": len(self.dataset.sil_episodes),
                "steps_with_state_vector": sum(
                    1 for s in self.dataset.steps
                    if s.get("state_vector") is not None
                ),
                "steps_with_teacher_dist": sum(
                    1 for s in self.dataset.steps
                    if s.get("teacher_dist") is not None
                ),
            },
            "final_metrics": {
                k: v[-1] if v else 0.0
                for k, v in self.metrics.items()
            },
            "best_total_loss": self._best_total_loss,
            "eval_teacher_accuracy": (
                self.metrics["eval_teacher_accuracy"][-1]
                if self.metrics["eval_teacher_accuracy"] else 0.0
            ),
        }

    def _save_report(self, report: Dict) -> None:
        """Save report as JSON."""
        path = RESULTS_DIR / f"offline_report_{self._run_id}.json"
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        console.print(f"  Report: {path}")

    def _print_summary(self, report: Dict) -> None:
        """Print summary table."""
        table = Table(title="Offline Distillation Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Run ID", report["run_id"])
        table.add_row("Duration", f"{report['elapsed_hours']:.2f}h")
        table.add_row("Data Steps", f"{report['data']['total_steps']:,}")
        table.add_row("Episodes", str(report["data"]["total_episodes"]))
        table.add_row("SIL Episodes", str(report["data"]["sil_episodes"]))
        table.add_row("With State Vec", f"{report['data']['steps_with_state_vector']:,}")
        table.add_row("With Teacher Dist", f"{report['data']['steps_with_teacher_dist']:,}")
        table.add_row("Best Loss", f"{report['best_total_loss']:.6f}")
        table.add_row("Teacher Match", f"{report['eval_teacher_accuracy']:.1%}")

        fm = report.get("final_metrics", {})
        table.add_row("Final BC Loss", f"{fm.get('bc_loss', 0):.6f}")
        table.add_row("Final KL Loss", f"{fm.get('kl_loss', 0):.6f}")
        table.add_row("Final SIL Loss", f"{fm.get('sil_loss', 0):.6f}")
        table.add_row("Final Value Loss", f"{fm.get('value_loss', 0):.6f}")

        console.print(table)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Offline Distillation Trainer — Phase 47",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint", type=str, default="latest",
        help="Path to PPO checkpoint (.pt) or 'latest' (default: latest)",
    )
    parser.add_argument(
        "--traces", type=str, default=str(TRACES_DIR),
        help=f"Traces directory (default: {TRACES_DIR})",
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--bc-coef", type=float, default=0.30,
        help="BC loss coefficient (default: 0.30)",
    )
    parser.add_argument(
        "--kl-coef", type=float, default=0.20,
        help="KL teacher distillation coefficient (default: 0.20)",
    )
    parser.add_argument(
        "--sil-coef", type=float, default=0.25,
        help="SIL loss coefficient (default: 0.25)",
    )
    parser.add_argument(
        "--cql", action="store_true",
        help="Enable CQL regularization for DDQN",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (default: cuda if available)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--save-every", type=int, default=10,
        help="Save checkpoint every N epochs (default: 10)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    config = OfflineConfig(
        checkpoint_path=args.checkpoint,
        traces_dir=args.traces,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        bc_coef=args.bc_coef,
        kl_coef=args.kl_coef,
        sil_coef=args.sil_coef,
        use_cql=args.cql,
        device=args.device,
        seed=args.seed,
        save_every=args.save_every,
    )

    trainer = OfflineDistillTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
