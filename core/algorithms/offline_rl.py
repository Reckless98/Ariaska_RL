"""core/algorithms/offline_rl.py — Phase 49: Offline RL (CQL / IQL).

Learn from logged pentest traces without live execution.

Algorithms:
  - CQL (Conservative Q-Learning): adds conservative penalty to Q-values
    to avoid overestimating OOD actions.
  - IQL (Implicit Q-Learning): uses expectile regression to learn
    value function without max-Q, avoiding overestimation.

Usage::

    from core.algorithms.offline_rl import OfflineRLTrainer, OfflineDataset

    transitions = load_jsonl_traces("traces/episode_001.jsonl")
    dataset = OfflineDataset.from_transitions(transitions)
    trainer = OfflineRLTrainer(config)
    losses = trainer.train(dataset, epochs=100)
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger("ariaska.algorithms.offline_rl")


# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────


@dataclass
class OfflineRLConfig:
    """Configuration for offline RL training.

    Args:
        enabled: Master switch.
        state_dim: State dimension.
        action_dim: Action dimension.
        hidden_dim: Hidden layer width.
        lr: Learning rate.
        gamma: Discount factor.
        tau: Soft target update coefficient.
        algorithm: 'cql' or 'iql'.
        cql_alpha: Conservative penalty weight for CQL.
        iql_expectile: Expectile for IQL value regression.
        batch_size: Minibatch size.
    """
    enabled: bool = True
    state_dim: int = 512
    action_dim: int = 5
    hidden_dim: int = 256
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    algorithm: str = "cql"
    cql_alpha: float = 1.0
    iql_expectile: float = 0.7
    batch_size: int = 64


# ──────────────────────────────────────────────────────────────
# Networks
# ──────────────────────────────────────────────────────────────


class QNetwork(nn.Module):
    """Q-network: state → Q(s, a) for all actions."""

    def __init__(
        self,
        state_dim: int = 512,
        action_dim: int = 5,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns Q-values for all actions: (B, action_dim)."""
        return self.net(state)


class ValueNetwork(nn.Module):
    """Value network: state → V(s) scalar."""

    def __init__(
        self,
        state_dim: int = 512,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns value: (B, 1)."""
        return self.net(state)


# ──────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────


class OfflineDataset(Dataset):
    """Offline RL dataset from logged transitions.

    Each item returns (state, action, reward, next_state, done).
    """

    def __init__(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states
        self.dones = dones

    def __len__(self) -> int:
        return self.states.size(0) if self.states.numel() > 0 else 0

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )

    @classmethod
    def from_transitions(
        cls,
        transitions: List[Dict[str, Any]],
        state_dim: int = 512,
    ) -> "OfflineDataset":
        """Build dataset from list of transition dicts.

        Each dict must have keys: state, action, reward, next_state, done.
        """
        if not transitions:
            return cls(
                states=torch.empty(0, state_dim),
                actions=torch.empty(0, dtype=torch.long),
                rewards=torch.empty(0),
                next_states=torch.empty(0, state_dim),
                dones=torch.empty(0),
            )

        states = torch.tensor(
            [t["state"] for t in transitions], dtype=torch.float32
        )
        actions = torch.tensor(
            [t["action"] for t in transitions], dtype=torch.long
        )
        rewards = torch.tensor(
            [t["reward"] for t in transitions], dtype=torch.float32
        )
        next_states = torch.tensor(
            [t["next_state"] for t in transitions], dtype=torch.float32
        )
        dones = torch.tensor(
            [float(t["done"]) for t in transitions], dtype=torch.float32
        )

        logger.info(
            f"[OFFLINE] Built dataset with {len(transitions)} transitions, "
            f"state_dim={states.shape[1]}"
        )
        return cls(states, actions, rewards, next_states, dones)

    @classmethod
    def from_jsonl(cls, path: str, state_dim: int = 512) -> "OfflineDataset":
        """Load transitions from a JSONL trace file."""
        import json
        from pathlib import Path

        transitions: List[Dict[str, Any]] = []
        p = Path(path)
        if not p.exists():
            logger.warning(f"[OFFLINE] Trace file not found: {path}")
            return cls.from_transitions([], state_dim=state_dim)

        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if all(
                        k in entry
                        for k in ("state", "action", "reward", "next_state", "done")
                    ):
                        transitions.append(entry)
                except json.JSONDecodeError:
                    continue

        logger.info(f"[OFFLINE] Loaded {len(transitions)} transitions from {path}")
        return cls.from_transitions(transitions, state_dim=state_dim)


# ──────────────────────────────────────────────────────────────
# CQL Trainer
# ──────────────────────────────────────────────────────────────


class CQLTrainer:
    """Conservative Q-Learning trainer.

    Adds conservative penalty:  alpha * (E[log sum_a exp Q(s,a)] - E[Q(s, a_data)])
    to prevent Q-value overestimation on out-of-distribution actions.
    """

    def __init__(self, config: Optional[OfflineRLConfig] = None) -> None:
        self.config = config or OfflineRLConfig()
        self.q_net = QNetwork(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dim,
        )
        self.target_q_net = copy.deepcopy(self.q_net)
        self.optimizer = torch.optim.Adam(
            self.q_net.parameters(), lr=self.config.lr
        )
        self._steps = 0

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single CQL training step.

        Args:
            batch: Dict with keys states, actions, rewards, next_states, dones.

        Returns:
            Dict with bellman_loss, cql_loss, total_loss.
        """
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]

        # Current Q-values for selected actions
        q_values = self.q_net(states)
        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values (no grad)
        with torch.no_grad():
            target_q = self.target_q_net(next_states)
            max_next_q = target_q.max(dim=1).values
            td_target = rewards + self.config.gamma * max_next_q * (1.0 - dones)

        # Bellman loss
        bellman_loss = F.mse_loss(q_selected, td_target)

        # CQL conservative penalty
        logsumexp_q = torch.logsumexp(q_values, dim=1).mean()
        data_q = q_selected.mean()
        cql_loss = self.config.cql_alpha * (logsumexp_q - data_q)

        total_loss = bellman_loss + cql_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        # Soft target update
        self._soft_update()
        self._steps += 1

        return {
            "bellman_loss": bellman_loss.item(),
            "cql_loss": cql_loss.item(),
            "total_loss": total_loss.item(),
        }

    def _soft_update(self) -> None:
        """Polyak soft update of target network."""
        for tp, p in zip(
            self.target_q_net.parameters(), self.q_net.parameters()
        ):
            tp.data.copy_(
                self.config.tau * p.data + (1.0 - self.config.tau) * tp.data
            )


# ──────────────────────────────────────────────────────────────
# IQL Trainer
# ──────────────────────────────────────────────────────────────


class IQLTrainer:
    """Implicit Q-Learning trainer.

    Uses expectile regression on V(s) relative to Q(s,a) to learn
    the optimal value function without explicitly maximising over actions.
    """

    def __init__(self, config: Optional[OfflineRLConfig] = None) -> None:
        self.config = config or OfflineRLConfig(algorithm="iql")
        self.q_net = QNetwork(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dim,
        )
        self.target_q_net = copy.deepcopy(self.q_net)
        self.value_net = ValueNetwork(
            self.config.state_dim, self.config.hidden_dim
        )
        self.q_optimizer = torch.optim.Adam(
            self.q_net.parameters(), lr=self.config.lr
        )
        self.v_optimizer = torch.optim.Adam(
            self.value_net.parameters(), lr=self.config.lr
        )
        self._steps = 0

    def _expectile_loss(
        self, diff: torch.Tensor, expectile: float
    ) -> torch.Tensor:
        """Asymmetric L2 loss for expectile regression."""
        weight = torch.where(diff > 0, expectile, 1.0 - expectile)
        return (weight * diff.pow(2)).mean()

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single IQL training step.

        Returns:
            Dict with value_loss, q_loss.
        """
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]

        # Value loss (expectile regression)
        with torch.no_grad():
            target_q = self.target_q_net(states)
            q_for_actions = target_q.gather(
                1, actions.unsqueeze(1)
            ).squeeze(1)

        v = self.value_net(states).squeeze(1)
        value_loss = self._expectile_loss(
            q_for_actions - v, self.config.iql_expectile
        )

        self.v_optimizer.zero_grad()
        value_loss.backward()
        self.v_optimizer.step()

        # Q loss (standard Bellman with V as next-state value)
        with torch.no_grad():
            next_v = self.value_net(next_states).squeeze(1)
            td_target = rewards + self.config.gamma * next_v * (1.0 - dones)

        q_values = self.q_net(states)
        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        q_loss = F.mse_loss(q_selected, td_target)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Soft target update
        for tp, p in zip(
            self.target_q_net.parameters(), self.q_net.parameters()
        ):
            tp.data.copy_(
                self.config.tau * p.data + (1.0 - self.config.tau) * tp.data
            )
        self._steps += 1

        return {
            "value_loss": value_loss.item(),
            "q_loss": q_loss.item(),
        }


# ──────────────────────────────────────────────────────────────
# High-level trainer
# ──────────────────────────────────────────────────────────────


class OfflineRLTrainer:
    """High-level offline RL trainer wrapping CQL/IQL.

    Usage::

        trainer = OfflineRLTrainer(config)
        dataset = OfflineDataset.from_transitions(transitions)
        epoch_losses = trainer.train(dataset, epochs=100)
    """

    def __init__(self, config: Optional[OfflineRLConfig] = None) -> None:
        self.config = config or OfflineRLConfig()
        if self.config.algorithm == "iql":
            self._trainer = IQLTrainer(self.config)
        else:
            self._trainer = CQLTrainer(self.config)
        self._epoch_losses: List[Dict[str, float]] = []

    def train(
        self,
        dataset: OfflineDataset,
        epochs: int = 50,
    ) -> List[Dict[str, float]]:
        """Train on an offline dataset for N epochs.

        Args:
            dataset: OfflineDataset instance.
            epochs: Number of passes over the dataset.

        Returns:
            List of per-epoch average losses.
        """
        if len(dataset) == 0:
            logger.warning("[OFFLINE] Empty dataset — skipping training")
            return []

        loader = DataLoader(
            dataset,
            batch_size=max(1, self.config.batch_size),
            shuffle=True,
            drop_last=False,
        )

        self._epoch_losses = []
        for epoch in range(epochs):
            epoch_loss_accum: Dict[str, float] = {}
            steps = 0
            for batch_tuple in loader:
                states, actions, rewards, next_states, dones = batch_tuple
                batch = {
                    "states": states,
                    "actions": actions,
                    "rewards": rewards,
                    "next_states": next_states,
                    "dones": dones,
                }
                step_losses = self._trainer.train_step(batch)
                for k, v in step_losses.items():
                    epoch_loss_accum[k] = epoch_loss_accum.get(k, 0.0) + v
                steps += 1

            if steps > 0:
                avg_losses = {k: v / steps for k, v in epoch_loss_accum.items()}
            else:
                avg_losses = {}
            self._epoch_losses.append(avg_losses)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"[OFFLINE] Epoch {epoch + 1}/{epochs} — "
                    f"losses: {avg_losses}"
                )

        return self._epoch_losses

    def get_stats(self) -> Dict[str, Any]:
        """Return training statistics."""
        return {
            "algorithm": self.config.algorithm,
            "epochs_trained": len(self._epoch_losses),
            "state_dim": self.config.state_dim,
            "action_dim": self.config.action_dim,
            "last_losses": (
                self._epoch_losses[-1] if self._epoch_losses else {}
            ),
        }
