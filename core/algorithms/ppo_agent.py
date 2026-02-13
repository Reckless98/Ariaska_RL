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
    # R68: Phase-gated actor heads (HRL-lite)
    use_phase_gates: bool = True        # 3 phase-specific gate networks modulate actor logits
    num_phase_groups: int = 3           # recon=0, exploit=1, post-exploit=2
    # R70: Self-Imitation Learning (SIL) + Advantage Clipping
    advantage_clip: float = 4.0         # Clip normalized advantages to [-clip, +clip]
    sil_buffer_size: int = 500          # Max entries in SIL replay buffer
    sil_coef: float = 0.1              # SIL loss coefficient (small, doesn't overpower PPO)
    sil_epochs: int = 1                 # SIL epochs per PPO update


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

        # ── R68: Phase-Gated Actor Heads (HRL-lite) ──────────────────
        # 3 lightweight gate networks that produce additive logit offsets
        # per phase group. Shared base actor handles general policy;
        # phase gates specialize: recon=0, exploit=1, post-exploit=2.
        # Phase group is extracted from state dims 0-11 (phase one-hot).
        self.has_phase_gates = config.use_phase_gates
        if self.has_phase_gates:
            gate_hidden = max(final_dim // 4, 32)
            self.phase_gates = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(final_dim, gate_hidden),
                    nn.GELU(),
                    nn.Linear(gate_hidden, config.action_dim),
                )
                for _ in range(config.num_phase_groups)
            ])
            # Small init so gates start near-zero (don't disrupt base actor)
            for gate in self.phase_gates:
                nn.init.zeros_(gate[-1].weight)
                nn.init.zeros_(gate[-1].bias)

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

    @staticmethod
    def _extract_phase_group(state: torch.Tensor) -> torch.Tensor:
        """R68: Extract phase group index from state tensor.

        State dims 0-11 encode the attack phase as one-hot:
          0-1 → RECON/ENUMERATION       → group 0 (recon)
          2-3 → EXPLOITATION/PRIV_ESC    → group 1 (exploit)
          4+  → LATERAL/POST/EXFIL/CLOSE → group 2 (post-exploit)

        Args:
            state: (..., state_dim) tensor.

        Returns:
            Long tensor of shape (...) with values in {0, 1, 2}.
        """
        phase_one_hot = state[..., :12]
        phase_idx = phase_one_hot.argmax(dim=-1)
        group = torch.where(
            phase_idx <= 1, torch.zeros_like(phase_idx),
            torch.where(phase_idx <= 3, torch.ones_like(phase_idx),
                        torch.full_like(phase_idx, 2))
        )
        return group

    def forward(
        self, state: torch.Tensor, phase_group: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action logits and state value.

        Args:
            state: (B, state_dim) tensor.
            phase_group: Optional (B,) long tensor with phase group ids.
                If None and use_phase_gates=True, extracted from state.

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

        # R68: Phase-gated modulation
        if self.has_phase_gates:
            if phase_group is None:
                phase_group = self._extract_phase_group(state)
            # Apply per-sample phase gate
            if phase_group.dim() == 0:
                phase_group = phase_group.unsqueeze(0)
            gate_logits = torch.zeros_like(logits)
            for g in range(len(self.phase_gates)):
                mask_g = (phase_group == g)
                if mask_g.any():
                    gate_logits[mask_g] = self.phase_gates[g](x[mask_g])
            logits = logits + gate_logits

        value = self.critic(x)
        return logits, value

    def get_action_and_value(
        self, state: torch.Tensor, action: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
        logit_bias: Optional[torch.Tensor] = None,
        phase_group: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log_prob, entropy, and value.

        If *action* is None, sample from the policy. Otherwise compute
        log_prob and entropy for the given action (used during PPO update).

        Args:
            state: (B, state_dim) tensor.
            action: If provided, compute log_prob for this action.
            action_mask: Optional (B, action_dim) or (action_dim,) bool tensor.
                True = valid action, False = masked out (logit → -inf).
            logit_bias: Optional (action_dim,) float tensor. Added to logits
                before softmax. R67: Used for soft-penalizing repeated commands.
            phase_group: Optional (B,) long tensor. R68: Phase group for
                phase-gated actor heads. Auto-extracted from state if None.
        """
        logits, value = self.forward(state, phase_group=phase_group)

        # R67: Apply logit bias (soft penalty for used commands)
        if logit_bias is not None:
            bias = logit_bias.to(logits.device)
            if bias.dim() == 1:
                bias = bias.unsqueeze(0).expand_as(logits)
            logits = logits + bias

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


class SILBuffer:
    """R70: Self-Imitation Learning buffer.

    Stores (state, action, return) tuples from above-average episodes.
    During PPO updates, these golden transitions reinforce the best
    behaviors seen across the entire training run, not just the current
    rollout. Only stores transitions with POSITIVE advantage (returns
    above the running mean), ensuring we never imitate bad behavior.

    Reference: Oh et al., "Self-Imitation Learning" (ICML 2018).
    """

    def __init__(self, capacity: int = 500):
        self.capacity = capacity
        self.states: List[torch.Tensor] = []
        self.actions: List[int] = []
        self.returns: List[float] = []
        self._return_baseline = 0.0
        self._return_count = 0

    def _update_baseline(self, ep_return: float):
        """Update running baseline (EMA of episode returns)."""
        self._return_count += 1
        alpha = min(0.1, 1.0 / self._return_count)
        self._return_baseline = (1 - alpha) * self._return_baseline + alpha * ep_return

    def add_episode(
        self,
        states: List[torch.Tensor],
        actions: List[int],
        rewards: List[float],
        gamma: float = 0.99,
    ):
        """Store transitions from an above-average episode.

        Only stores transitions whose discounted return exceeds the
        running baseline (positive advantage only — core SIL principle).

        Args:
            states: List of state tensors from the episode.
            actions: List of action indices.
            rewards: List of per-step rewards.
            gamma: Discount factor for computing returns.
        """
        ep_return = sum(rewards)
        self._update_baseline(ep_return)

        # Only store if episode is above average
        if ep_return <= self._return_baseline:
            return 0

        # Compute discounted returns for each step
        n = len(rewards)
        returns = [0.0] * n
        running = 0.0
        for t in reversed(range(n)):
            running = rewards[t] + gamma * running
            returns[t] = running

        # Store transitions with positive advantage
        added = 0
        for t in range(n):
            adv = returns[t] - self._return_baseline
            if adv > 0:
                self.states.append(states[t].detach().cpu())
                self.actions.append(actions[t])
                self.returns.append(returns[t])
                added += 1

        # Evict oldest if over capacity
        while len(self.states) > self.capacity:
            self.states.pop(0)
            self.actions.pop(0)
            self.returns.pop(0)

        return added

    def sample(
        self, batch_size: int, device: torch.device,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Sample a minibatch for SIL update.

        Returns:
            Dict with states, actions, returns tensors, or None if empty.
        """
        if len(self.states) < 4:
            return None

        n = len(self.states)
        idx = np.random.choice(n, size=min(batch_size, n), replace=False)

        return {
            "states": torch.stack([self.states[i] for i in idx]).to(device),
            "actions": torch.tensor([self.actions[i] for i in idx], dtype=torch.long).to(device),
            "returns": torch.tensor([self.returns[i] for i in idx], dtype=torch.float32).to(device),
        }

    def __len__(self):
        return len(self.states)


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

        # ── R58 Layer 2c: Adaptive entropy ───────────────────────────
        # Multiplier on base entropy coefficient. Decays on success streaks
        # (exploit more), grows on failure streaks (explore more).
        self._entropy_adaptive_multiplier: float = 1.0
        self._consecutive_closeouts: int = 0
        self._consecutive_failures: int = 0

        # ── R70: Self-Imitation Learning buffer ──────────────────────
        self.sil_buffer = SILBuffer(capacity=self.config.sil_buffer_size)

    # ── Action Selection ─────────────────────────────────────────────

    @torch.no_grad()
    def select_action(
        self, state_tensor: torch.Tensor, training: bool = True,
        action_mask: Optional[torch.Tensor] = None,
        logit_bias: Optional[torch.Tensor] = None,
        phase_group: Optional[int] = None,
    ) -> Tuple[int, float, float]:
        """Select an action from the current policy with optional masking.

        Args:
            state_tensor: (state_dim,) tensor from the state encoder.
            training: If True, sample from the distribution.
                      If False, take the greedy (argmax) action.
            action_mask: Optional (action_dim,) bool tensor.
                True = valid, False = masked (logit → -inf).
            logit_bias: Optional (action_dim,) float tensor. R67: Added to
                logits before softmax for soft-penalizing repeated commands.
            phase_group: Optional int (0=recon, 1=exploit, 2=post). R68:
                Phase group for phase-gated heads. Auto-extracted if None.

        Returns:
            (action_index, log_probability, state_value)
        """
        self.network.eval()
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)

        state_tensor = state_tensor.to(self.device)

        # R68: Build phase_group tensor if provided as int
        _pg_tensor = None
        if phase_group is not None:
            _pg_tensor = torch.tensor([phase_group], dtype=torch.long, device=self.device)

        if not training:
            # Greedy for evaluation
            logits, v = self.network(state_tensor, phase_group=_pg_tensor)
            if logit_bias is not None:
                bias = logit_bias.to(logits.device)
                if bias.dim() == 1:
                    bias = bias.unsqueeze(0).expand_as(logits)
                logits = logits + bias
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
                state_tensor, action_mask=action_mask, logit_bias=logit_bias,
                phase_group=_pg_tensor,
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

    # ── R58 Layer 2c: Adaptive Entropy Signaling ─────────────────

    def signal_episode_outcome(self, reached_closeout: bool):
        """Signal episode success/failure for adaptive entropy annealing.

        R58 Layer 2c: Adjusts ``_entropy_adaptive_multiplier`` based on
        consecutive outcome streaks:
          - 3+ consecutive CLOSEOUTs → decay multiplier (exploit more)
          - 2+ consecutive failures → boost multiplier (explore more)

        Called by SmartCoach.end_episode_ppo() after each episode.

        Args:
            reached_closeout: Whether CLOSEOUT phase was reached.
        """
        if reached_closeout:
            self._consecutive_closeouts += 1
            self._consecutive_failures = 0
            # Accelerate exploitation on success streaks
            if self._consecutive_closeouts >= 3:
                self._entropy_adaptive_multiplier = max(
                    0.5, self._entropy_adaptive_multiplier * 0.85
                )
        else:
            self._consecutive_failures += 1
            self._consecutive_closeouts = 0
            # Boost exploration on failure streaks
            if self._consecutive_failures >= 2:
                self._entropy_adaptive_multiplier = min(
                    1.5, self._entropy_adaptive_multiplier * 1.3
                )

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

        # R70: Clip normalized advantages to prevent gradient explosion
        # from extreme outlier steps (e.g., root_shell + flag combo = +655 raw)
        if self.config.advantage_clip > 0:
            advantages = torch.clamp(
                advantages, -self.config.advantage_clip, self.config.advantage_clip
            )

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
        # R58 Layer 2c: Apply adaptive multiplier on top of base annealing
        progress = min(self.total_steps / self.config.total_timesteps, 1.0)
        base_entropy = self.config.entropy_coef + progress * (
            self.config.entropy_coef_min - self.config.entropy_coef
        )
        self.entropy_coef = base_entropy * self._entropy_adaptive_multiplier

        # ── R70: Self-Imitation Learning update ──────────────────────
        # After regular PPO update, do 1 epoch of SIL on golden transitions.
        # SIL loss = -log_prob(action) * clamp(return - baseline, 0, inf)
        # Only reinforces above-average behavior (positive advantage only).
        sil_loss_val = 0.0
        if len(self.sil_buffer) >= 4 and self.config.sil_coef > 0:
            for _ in range(self.config.sil_epochs):
                sil_batch = self.sil_buffer.sample(
                    self.config.minibatch_size, self.device
                )
                if sil_batch is not None:
                    _, sil_log_probs, sil_entropy, sil_values = (
                        self.network.get_action_and_value(
                            sil_batch["states"], sil_batch["actions"]
                        )
                    )
                    # SIL advantage: return - value (only positive)
                    sil_adv = (sil_batch["returns"] - sil_values.detach()).clamp(min=0)
                    sil_policy_loss = -(sil_log_probs * sil_adv).mean()
                    sil_value_loss = F.huber_loss(
                        sil_values, sil_batch["returns"], delta=10.0
                    )
                    sil_loss = self.config.sil_coef * (
                        sil_policy_loss + 0.5 * sil_value_loss
                    )
                    self.optimizer.zero_grad()
                    sil_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.network.parameters(), self.config.max_grad_norm
                    )
                    self.optimizer.step()
                    sil_loss_val = sil_loss.item()
        metrics["sil_loss"] = sil_loss_val

        # Store metrics
        for k, v in metrics.items():
            if k in self.training_metrics:
                self.training_metrics[k].append(v)
                if len(self.training_metrics[k]) > 500:
                    self.training_metrics[k].pop(0)

        # Clear buffer for next rollout
        self.buffer.reset()

        return metrics

    # ── R70: SIL Episode Storage ─────────────────────────────────────

    def store_sil_episode(
        self,
        states: List[torch.Tensor],
        actions: List[int],
        rewards: List[float],
    ) -> int:
        """Store an episode's transitions in the SIL buffer if above average.

        Called by SmartCoach.end_episode_ppo() after each episode.

        Args:
            states: List of state tensors from the episode.
            actions: List of action indices.
            rewards: List of per-step rewards.

        Returns:
            Number of transitions stored (0 if below average).
        """
        return self.sil_buffer.add_episode(
            states, actions, rewards, gamma=self.config.gamma
        )

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
                # R58 Layer 2c: adaptive entropy state
                "adaptive_entropy": {
                    "multiplier": self._entropy_adaptive_multiplier,
                    "consecutive_closeouts": self._consecutive_closeouts,
                    "consecutive_failures": self._consecutive_failures,
                },
                # R68: phase gates config
                "has_phase_gates": getattr(self.network, 'has_phase_gates', False),
                # R70: SIL buffer baseline
                "sil_baseline": self.sil_buffer._return_baseline,
                "sil_count": self.sil_buffer._return_count,
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
        # R58 Layer 2c: Restore adaptive entropy state
        if "adaptive_entropy" in ckpt:
            ae = ckpt["adaptive_entropy"]
            self._entropy_adaptive_multiplier = ae.get("multiplier", 1.0)
            self._consecutive_closeouts = ae.get("consecutive_closeouts", 0)
            self._consecutive_failures = ae.get("consecutive_failures", 0)
        # R70: Restore SIL buffer baseline
        if "sil_baseline" in ckpt:
            self.sil_buffer._return_baseline = ckpt["sil_baseline"]
            self.sil_buffer._return_count = ckpt.get("sil_count", 0)

    # ── Diagnostics ──────────────────────────────────────────────────

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return current training diagnostics."""
        lr = self.optimizer.param_groups[0]["lr"]
        return {
            "total_steps": self.total_steps,
            "updates_done": self.updates_done,
            "learning_rate": lr,
            "entropy_coef": self.entropy_coef,
            "entropy_adaptive_multiplier": self._entropy_adaptive_multiplier,
            "buffer_size": len(self.buffer),
            "has_phase_gates": getattr(self.network, 'has_phase_gates', False),
            "sil_buffer_size": len(self.sil_buffer),
            "sil_baseline": round(self.sil_buffer._return_baseline, 2),
            "reward_norm": {
                "mean": round(self._reward_mean, 4),
                "std": round(math.sqrt(abs(self._reward_var)), 4),
                "count": self._reward_count,
            },
            "latest_metrics": {
                k: v[-1] if v else None for k, v in self.training_metrics.items()
            },
        }
