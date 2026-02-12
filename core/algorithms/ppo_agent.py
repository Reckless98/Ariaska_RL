#!/usr/bin/env python3
"""
core/algorithms/ppo_agent.py — ARIASKA PPO Actor-Critic v3.0
🧠 Proximal Policy Optimization | 🎯 GAE-λ | 📊 Clipped Surrogate | 🔬 Entropy Bonus

State-of-the-art on-policy RL algorithm for the Red agent. Uses:
  - AdvancedPolicyNetwork (attention + residual + NoisyLinear) as backbone
  - Generalised Advantage Estimation (GAE-λ) for low-variance advantages
  - Clipped surrogate objective to prevent destructive policy updates
  - Entropy bonus for exploration
  - Value function clipping for stable critic learning
  - Gradient clipping for training stability
  - Learning rate annealing with cosine schedule

Usage in SmartOrchestrator:
    ppo = PPOAgent(state_dim=512, action_dim=5, device="cuda")
    # During episode:
    action, log_prob, value = ppo.select_action(state_tensor)
    # After episode:
    ppo.store_transition(state, action, log_prob, reward, value, done)
    ppo.update()  # Multi-epoch PPO update on collected trajectory
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import math


@dataclass
class PPOConfig:
    """Hyperparameters for PPO training."""
    state_dim: int = 512
    action_dim: int = 5
    hidden_dims: List[int] = field(default_factory=lambda: [512, 512, 256])
    # PPO core
    clip_epsilon: float = 0.2           # Surrogate clipping range
    clip_value: float = 0.2             # Value function clipping
    gamma: float = 0.99                 # Discount factor
    gae_lambda: float = 0.97            # Phase 8.1: 0.95→0.97 for longer-horizon attack planning
    # Training
    learning_rate: float = 3e-4         # Adam LR (PPO standard)
    lr_min: float = 1e-5                # Minimum LR after annealing
    epochs_per_update: int = 4          # PPO epochs per rollout
    minibatch_size: int = 16            # R43: 64→16 — short episodes (6-9 steps) had 0 updates with 64
    max_grad_norm: float = 0.5          # Gradient clipping norm
    # Loss coefficients
    value_loss_coef: float = 0.5        # Critic loss weight
    entropy_coef: float = 0.01          # Entropy bonus weight
    entropy_coef_min: float = 0.003     # Phase 8.1: 0.001→0.003 — higher floor prevents exploration collapse
    # Rollout
    rollout_size: int = 256             # Steps per rollout before update
    # Annealing
    total_timesteps: int = 10000        # Phase 8.0: Faster annealing (was 50k)
    # Architecture
    use_attention: bool = True
    use_residual: bool = True
    dropout_rate: float = 0.05          # Lighter dropout for PPO


class PPOActorCritic(nn.Module):
    """Combined actor-critic network for PPO with advanced architecture.

    Uses shared backbone with optional attention-enhanced residual blocks
    from advanced_networks.py. Separate actor (policy) and critic (value)
    heads share feature extraction to reduce parameters.

    Architecture:
        Input (512) → LayerNorm → GELU → [Shared Backbone] →
        [AttentionResidual] → ├─ Actor Head → logits (action_dim)
                              └─ Critic Head → value (1)
    """

    def __init__(self, config: PPOConfig):
        super().__init__()
        self.config = config
        dims = config.hidden_dims

        # ── Shared Feature Extractor ─────────────────────────────────
        self.input_proj = nn.Linear(config.state_dim, dims[0])
        self.input_norm = nn.LayerNorm(dims[0])

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(config.dropout_rate))
        self.shared_backbone = nn.Sequential(*layers)

        final_dim = dims[-1]

        # ── Advanced Residual Block (Phase 5.2+) ────────────────────
        # Try to use ResidualBlock from advanced_networks.py for
        # attention-enhanced feature extraction. Falls back to simple
        # residual if import fails.
        self.has_residual = False
        self.has_attention_residual = False
        if config.use_residual and len(dims) >= 2 and dims[-2] == dims[-1]:
            try:
                from core.models.advanced_networks import ResidualBlock
                self.adv_residual = ResidualBlock(
                    dim=final_dim,
                    dropout=config.dropout_rate,
                    use_attention=config.use_attention,
                )
                self.has_attention_residual = True
                self.has_residual = True
            except (ImportError, Exception):
                # Fallback: simple residual without attention
                self.residual = nn.Sequential(
                    nn.Linear(final_dim, final_dim),
                    nn.LayerNorm(final_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout_rate),
                    nn.Linear(final_dim, final_dim),
                )
                self.has_residual = True

        # ── Actor Head (policy) ──────────────────────────────────────
        self.actor = nn.Sequential(
            nn.Linear(final_dim, final_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(final_dim // 2, config.action_dim),
        )

        # ── Critic Head (value) ──────────────────────────────────────
        self.critic = nn.Sequential(
            nn.Linear(final_dim, final_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(final_dim // 2, 1),
        )

        # Initialize with orthogonal initialization (PPO standard)
        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization — standard for PPO networks."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Actor output layer: smaller gain for initial exploration
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        # Critic output layer: unit gain
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action logits and state value.

        Args:
            state: (B, state_dim) tensor.

        Returns:
            (action_logits, value) — shapes (B, action_dim), (B, 1).
        """
        x = self.input_proj(state)
        x = self.input_norm(x)
        x = F.gelu(x)

        x = self.shared_backbone(x)

        if self.has_attention_residual:
            x = self.adv_residual(x)  # ResidualBlock with attention + skip
        elif self.has_residual:
            x = x + self.residual(x)  # simple skip connection

        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

    def get_action_and_value(
        self, state: torch.Tensor, action: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log_prob, entropy, and value.

        If *action* is None, sample from the policy. Otherwise compute
        log_prob and entropy for the given action (used during PPO update).

        Args:
            state: (B, state_dim) tensor.
            action: If provided, compute log_prob for this action.
            action_mask: Optional (B, action_dim) or (action_dim,) bool tensor.
                True = valid action, False = masked out (logit → -inf).
        """
        logits, value = self.forward(state)

        # Apply action mask: invalid actions get logit = -inf
        if action_mask is not None:
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0).expand_as(logits)
            action_mask = action_mask.to(logits.device)
            logits = logits.masked_fill(~action_mask, float("-inf"))

        dist = torch.distributions.Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value.squeeze(-1)


class RolloutBuffer:
    """Stores trajectory data for on-policy PPO updates.

    Efficiently manages episode trajectories with GAE computation.
    """

    def __init__(self, capacity: int = 256):
        self.capacity = capacity
        self.reset()

    def reset(self):
        """Clear all stored transitions."""
        self.states: List[torch.Tensor] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []
        self.size = 0

    def add(
        self,
        state: torch.Tensor,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ):
        """Add a single transition."""
        self.states.append(state.detach().cpu())
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.size += 1

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and discounted returns.

        Args:
            last_value: V(s_T) for bootstrapping the final step.
            gamma: Discount factor.
            gae_lambda: GAE lambda for bias-variance tradeoff.

        Returns:
            (returns, advantages) each of shape (N,).
        """
        n = self.size
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_non_terminal = 1.0 - float(self.dones[t])
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - float(self.dones[t])

            delta = (
                self.rewards[t]
                + gamma * next_value * next_non_terminal
                - self.values[t]
            )
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        values_arr = np.array(self.values, dtype=np.float32)
        returns = advantages + values_arr

        return torch.tensor(returns), torch.tensor(advantages)

    def get_batches(
        self,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        minibatch_size: int,
        device: torch.device,
    ):
        """Yield random minibatches for PPO epochs.

        Yields:
            dict with keys: states, actions, old_log_probs, returns, advantages, values
        """
        n = self.size
        indices = np.arange(n)
        np.random.shuffle(indices)

        states_t = torch.stack(self.states, dim=0)
        actions_t = torch.tensor(self.actions, dtype=torch.long)
        old_log_probs_t = torch.tensor(self.log_probs, dtype=torch.float32)
        values_t = torch.tensor(self.values, dtype=torch.float32)

        for start in range(0, n, minibatch_size):
            end = min(start + minibatch_size, n)
            mb_idx = indices[start:end]

            yield {
                "states": states_t[mb_idx].to(device),
                "actions": actions_t[mb_idx].to(device),
                "old_log_probs": old_log_probs_t[mb_idx].to(device),
                "returns": returns[mb_idx].to(device),
                "advantages": advantages[mb_idx].to(device),
                "values": values_t[mb_idx].to(device),
            }

    def __len__(self):
        return self.size


class PPOAgent:
    """Proximal Policy Optimization agent for Red team.

    Manages the actor-critic network, rollout buffer, and PPO update loop.
    Integrates with SmartOrchestrator via ``select_action()`` / ``propose_action()``.
    """

    def __init__(
        self,
        config: Optional[PPOConfig] = None,
        device: str = "cuda",
    ):
        self.config = config or PPOConfig()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # ── Network ──────────────────────────────────────────────────
        self.network = PPOActorCritic(self.config).to(self.device)

        # ── Optimizer ────────────────────────────────────────────────
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate,
            eps=1e-5,  # PPO standard
        )

        # ── LR Scheduler (cosine annealing) ──────────────────────────
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(self.config.total_timesteps // self.config.rollout_size, 1),
            eta_min=self.config.lr_min,
        )

        # ── Rollout Buffer ───────────────────────────────────────────
        self.buffer = RolloutBuffer(capacity=self.config.rollout_size)

        # ── Tracking ─────────────────────────────────────────────────
        self.total_steps = 0
        self.updates_done = 0
        self.entropy_coef = self.config.entropy_coef
        self.training_metrics: Dict[str, List[float]] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_fraction": [],
            "explained_variance": [],
        }

        # ── Phase 6: Running reward normalization ────────────────────
        # Welford online algorithm for reward mean/variance tracking
        # This stabilizes PPO value function by normalizing returns
        self._reward_mean = 0.0
        self._reward_var = 1.0
        self._reward_count = 0
        self._return_mean = 0.0
        self._return_var = 1.0
        self._return_count = 0

    # ── Action Selection ─────────────────────────────────────────────

    @torch.no_grad()
    def select_action(
        self, state_tensor: torch.Tensor, training: bool = True,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[int, float, float]:
        """Select an action from the current policy with optional masking.

        Args:
            state_tensor: (state_dim,) tensor from the state encoder.
            training: If True, sample from the distribution.
                      If False, take the greedy (argmax) action.
            action_mask: Optional (action_dim,) bool tensor.
                True = valid, False = masked (logit → -inf).

        Returns:
            (action_index, log_probability, state_value)
        """
        self.network.eval()
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)

        state_tensor = state_tensor.to(self.device)

        if not training:
            # Greedy for evaluation
            logits, v = self.network(state_tensor)
            if action_mask is not None:
                mask = action_mask.to(logits.device)
                if mask.dim() == 1:
                    mask = mask.unsqueeze(0)
                logits = logits.masked_fill(~mask, float("-inf"))
            action = logits.argmax(dim=-1)
            dist = torch.distributions.Categorical(logits=logits)
            log_prob = dist.log_prob(action)
            value = v.squeeze(-1)
        else:
            action, log_prob, _entropy, value = self.network.get_action_and_value(
                state_tensor, action_mask=action_mask,
            )

        self.network.train()
        return (
            action.item(),
            log_prob.item(),
            value.item(),
        )

    def propose_action(
        self, state: Dict[str, Any], **kwargs
    ) -> Tuple[str, float]:
        """SmartOrchestrator-compatible interface.

        Returns (action_name, confidence) instead of (int, logprob, value).
        Uses CommandActionMapper if action_dim > 5 (per-role mode),
        otherwise falls back to phase name mapping.
        """
        from core.models.state_encoder import encode_state

        state_tensor = encode_state(state, self.device, **kwargs)
        action_idx, log_prob, value = self.select_action(state_tensor)

        # Confidence from softmax probability
        with torch.no_grad():
            logits, _ = self.network(state_tensor.unsqueeze(0).to(self.device))
            probs = F.softmax(logits, dim=-1)
            confidence = probs[0, action_idx].item()

        # Map action index → name
        # If using per-role CommandActionMapper (action_dim > 5), try to get template name
        if self.config.action_dim > 5:
            try:
                from core.algorithms.command_action_mapper import get_mapper
                # We don't know the role here, so return the raw index
                action_name = f"action_{action_idx}"
            except ImportError:
                action_name = f"action_{action_idx}"
        else:
            phase_actions = ["recon", "enumeration", "exploit", "privesc", "exfiltrate"]
            action_name = phase_actions[action_idx] if action_idx < len(phase_actions) else "recon"

        return action_name, confidence

    # ── Transition Storage ───────────────────────────────────────────

    def _update_reward_stats(self, reward: float):
        """Online Welford update for running reward mean/variance."""
        self._reward_count += 1
        delta = reward - self._reward_mean
        self._reward_mean += delta / self._reward_count
        delta2 = reward - self._reward_mean
        self._reward_var += (delta * delta2 - self._reward_var) / max(self._reward_count, 2)

    def _update_return_stats(self, ret: float):
        """Online Welford update for running return mean/variance."""
        self._return_count += 1
        delta = ret - self._return_mean
        self._return_mean += delta / self._return_count
        delta2 = ret - self._return_mean
        self._return_var += (delta * delta2 - self._return_var) / max(self._return_count, 2)

    def normalize_reward(self, reward: float) -> float:
        """Normalize reward using running statistics.

        Maps rewards to roughly zero-mean unit-variance, which is critical
        for PPO value function convergence. Without this, value loss
        scales with reward magnitude (we saw ~20,000 value loss).
        """
        self._update_reward_stats(reward)
        std = max(math.sqrt(abs(self._reward_var)), 1e-4)
        return (reward - self._reward_mean) / std

    def store_transition(
        self,
        state: torch.Tensor,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ):
        """Store a single transition in the rollout buffer.

        Phase 6: Rewards are normalized using running statistics before
        storage, which stabilizes the value function and prevents the
        ~20,000 value loss we saw with raw rewards.
        """
        normalized_reward = self.normalize_reward(reward)
        self.buffer.add(state, action, log_prob, normalized_reward, value, done)
        self.total_steps += 1

    # ── PPO Update ───────────────────────────────────────────────────

    def update(self, last_value: float = 0.0) -> Dict[str, float]:
        """Run PPO update on collected trajectory.

        Should be called after each episode or when the rollout buffer
        reaches capacity.

        Args:
            last_value: Bootstrap value V(s_T) for the last state.

        Returns:
            Dict of training metrics for this update.
        """
        if self.buffer.size < self.config.minibatch_size:
            return {}

        # ── Compute GAE Advantages ───────────────────────────────────
        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_value=last_value,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )

        # Track return statistics for monitoring
        for r in returns.numpy():
            self._update_return_stats(float(r))

        # Normalise advantages (critical for PPO stability)
        adv_std = advantages.std()
        if adv_std > 1e-8:
            advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

        # ── Multi-Epoch PPO ──────────────────────────────────────────
        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }
        num_batches = 0

        for _epoch in range(self.config.epochs_per_update):
            for batch in self.buffer.get_batches(
                returns, advantages, self.config.minibatch_size, self.device
            ):
                # Forward pass
                _, new_log_probs, entropy, new_values = (
                    self.network.get_action_and_value(
                        batch["states"], batch["actions"]
                    )
                )

                # ── Policy Loss (clipped surrogate) ──────────────────
                log_ratio = new_log_probs - batch["old_log_probs"]
                ratio = torch.exp(log_ratio)

                # Approximate KL divergence for early stopping
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()

                clip_adv = (
                    torch.clamp(
                        ratio,
                        1.0 - self.config.clip_epsilon,
                        1.0 + self.config.clip_epsilon,
                    )
                    * batch["advantages"]
                )
                policy_loss = -torch.min(
                    ratio * batch["advantages"], clip_adv
                ).mean()

                # ── Value Loss (Huber for outlier robustness) ─────────
                # Phase 6: Huber loss is less sensitive to reward outliers
                # than MSE, preventing value function divergence
                # R46: PPO-style value clipping — clamp new values to
                # old values ± clip_epsilon to prevent catastrophic updates
                # (EP8/R45 had value_loss=22.56 without this)
                value_pred_clipped = batch["values"] + torch.clamp(
                    new_values - batch["values"],
                    -self.config.clip_value,
                    self.config.clip_value,
                )
                value_loss_unclipped = F.huber_loss(
                    new_values, batch["returns"], delta=10.0, reduction="none"
                )
                value_loss_clipped = F.huber_loss(
                    value_pred_clipped, batch["returns"], delta=10.0, reduction="none"
                )
                value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()

                # ── Entropy Bonus ────────────────────────────────────
                entropy_loss = entropy.mean()

                # ── Combined Loss ────────────────────────────────────
                loss = (
                    policy_loss
                    + self.config.value_loss_coef * value_loss
                    - self.entropy_coef * entropy_loss
                )

                # ── Backward + Clip ──────────────────────────────────
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()

                # ── Track Metrics ────────────────────────────────────
                with torch.no_grad():
                    clip_frac = (
                        (torch.abs(ratio - 1.0) > self.config.clip_epsilon)
                        .float()
                        .mean()
                        .item()
                    )

                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += entropy_loss.item()
                metrics["approx_kl"] += approx_kl
                metrics["clip_fraction"] += clip_frac
                num_batches += 1

            # Early stopping if KL divergence is too large
            if approx_kl > 0.015:
                break

        # Average metrics
        if num_batches > 0:
            for k in metrics:
                metrics[k] /= num_batches

        # Explained variance
        with torch.no_grad():
            values_arr = np.array(self.buffer.values, dtype=np.float32)
            returns_arr = returns.numpy()
            var_y = np.var(returns_arr)
            if var_y > 1e-8:
                metrics["explained_variance"] = float(
                    1.0 - np.var(returns_arr - values_arr) / var_y
                )
            else:
                metrics["explained_variance"] = 0.0

        # Step scheduler
        self.lr_scheduler.step()
        self.updates_done += 1

        # Anneal entropy coefficient
        progress = min(self.total_steps / self.config.total_timesteps, 1.0)
        self.entropy_coef = self.config.entropy_coef + progress * (
            self.config.entropy_coef_min - self.config.entropy_coef
        )

        # Store metrics
        for k, v in metrics.items():
            if k in self.training_metrics:
                self.training_metrics[k].append(v)
                if len(self.training_metrics[k]) > 500:
                    self.training_metrics[k].pop(0)

        # Clear buffer for next rollout
        self.buffer.reset()

        return metrics

    # ── Persistence ──────────────────────────────────────────────────

    def save(self, path: str):
        """Save PPO checkpoint."""
        torch.save(
            {
                "network_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.lr_scheduler.state_dict(),
                "total_steps": self.total_steps,
                "updates_done": self.updates_done,
                "entropy_coef": self.entropy_coef,
                "config": self.config,
                "training_metrics": self.training_metrics,
                # Phase 6: reward normalization stats for continuity
                "reward_norm": {
                    "mean": self._reward_mean,
                    "var": self._reward_var,
                    "count": self._reward_count,
                    "return_mean": self._return_mean,
                    "return_var": self._return_var,
                    "return_count": self._return_count,
                },
            },
            path,
        )

    def load(self, path: str):
        """Load PPO checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(ckpt["network_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            self.lr_scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.total_steps = ckpt.get("total_steps", 0)
        self.updates_done = ckpt.get("updates_done", 0)
        self.entropy_coef = ckpt.get("entropy_coef", self.config.entropy_coef)
        # Phase 6: Restore reward normalization stats
        if "reward_norm" in ckpt:
            rn = ckpt["reward_norm"]
            self._reward_mean = rn.get("mean", 0.0)
            self._reward_var = rn.get("var", 1.0)
            self._reward_count = rn.get("count", 0)
            self._return_mean = rn.get("return_mean", 0.0)
            self._return_var = rn.get("return_var", 1.0)
            self._return_count = rn.get("return_count", 0)

    # ── Diagnostics ──────────────────────────────────────────────────

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return current training diagnostics."""
        lr = self.optimizer.param_groups[0]["lr"]
        return {
            "total_steps": self.total_steps,
            "updates_done": self.updates_done,
            "learning_rate": lr,
            "entropy_coef": self.entropy_coef,
            "buffer_size": len(self.buffer),
            "reward_norm": {
                "mean": round(self._reward_mean, 4),
                "std": round(math.sqrt(abs(self._reward_var)), 4),
                "count": self._reward_count,
            },
            "latest_metrics": {
                k: v[-1] if v else None for k, v in self.training_metrics.items()
            },
        }
