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
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import math

logger = logging.getLogger("ariaska.ppo_agent")


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
    sil_coef: float = 0.25             # Phase 13.0: +150% (was 0.1) — stronger imitation from high-reward episodes
    sil_epochs: int = 2                 # Phase 13.0: +100% (was 1) — deeper SIL gradient absorption
    # R71: Symlog value compression + cosine entropy schedule
    use_symlog: bool = True             # Compress rewards via sign(x)*log(1+|x|) (DreamerV3)
    use_cosine_entropy: bool = True     # Cosine entropy annealing (stays high longer)
    # R72: Prioritized advantage sampling + return-variance entropy coupling
    use_prioritized_advantages: bool = True  # Sample PPO minibatches by |advantage|
    return_variance_entropy: bool = True     # Couple entropy decay to return variance
    # R73: Gradient accumulation + policy ramp-up
    grad_accum_steps: int = 2            # Accumulate gradients over N mini-batches (effective batch 2×)
    policy_warmup_updates: int = 3       # Ramp up policy loss over first N PPO updates
    # R74: KL-adaptive LR + per-phase advantage whitening
    target_kl: float = 0.01              # Target KL for adaptive LR (OpenAI standard)
    use_kl_adaptive_lr: bool = True      # Adjust LR based on recent KL divergence
    use_phase_advantage_whitening: bool = True  # Normalize advantages per phase group
    # R75: EMA target network + value-surprise intrinsic bonus
    use_ema_anchor: bool = True          # Polyak-averaged target network for stable bootstrapping
    ema_tau: float = 0.995               # Polyak coefficient: ema = τ*ema + (1-τ)*online
    value_surprise_coef: float = 0.3     # Intrinsic bonus for value disagreement (capped at 2.0)
    # R76: Dual-horizon GAE blending
    use_dual_horizon_gae: bool = True    # Blend long-horizon (λ=0.95) + short-horizon GAE
    gae_lambda_short: float = 0.70       # Short-horizon GAE lambda for faster credit assignment
    dual_horizon_blend: float = 0.65     # Weight for long-horizon (0.35 for short)
    # R77: Spectral normalization on critic + soft advantage clipping
    use_spectral_norm_critic: bool = True  # Constrain critic Lipschitz constant
    use_soft_advantage_clip: bool = True   # Tanh-based smooth clip (gradients flow at boundary)
    # R79: Auxiliary phase prediction head
    use_phase_prediction_aux: bool = True  # Auxiliary loss predicting attack phase group
    phase_prediction_coef: float = 0.1     # Auxiliary loss weight (forces phase-aware representations)
    # R80: Adaptive clip scheduling + entropy rebound
    use_adaptive_clip: bool = True       # Auto-adjust clip_epsilon from rolling clip_fraction
    clip_epsilon_min: float = 0.15       # Minimum clip (too narrow → conservative)
    clip_epsilon_max: float = 0.25       # Maximum clip (too wide → unstable)
    use_entropy_rebound: bool = True     # Boost entropy if it drops too low for 3+ updates
    entropy_rebound_threshold: float = 0.5  # Minimum acceptable entropy level

    # Phase 14.0: Behavioral Cloning loss (from TeacherTrace → BCBuffer)
    use_bc_loss: bool = False             # Feature-flag gated
    bc_loss_coef: float = 0.1            # BC loss weight (kept low to not dominate PPO)
    bc_buffer: Any = None                # Reference to BCBuffer (set externally)

    # ── Phase 37: Level 5 GPT ↔ RL Integration ──────────────────────
    # LLM feature injection — additional dims concatenated with state
    llm_feature_dim: int = 0              # 0 = disabled, 256 = Level 5 (768 total input)
    # Action prior injection — LLM preference vector added to logits
    use_llm_prior: bool = False           # Enable LLM prior injection into policy logits
    prior_alpha_init: float = 0.50        # Initial prior weight (decays via anneal)
    # KL teacher distillation loss
    use_kl_teacher_loss: bool = False     # KL(policy || teacher_distribution)
    kl_teacher_coef: float = 0.15        # KL loss weight (decays with anneal)
    # Ranking margin loss — teacher action ranks higher than non-teacher
    use_ranking_loss: bool = False        # Margin-based ranking loss
    ranking_loss_coef: float = 0.05      # Ranking loss weight
    ranking_margin: float = 1.0          # Margin between teacher vs non-teacher logits
    # Value regularization — align critic with LLM value estimates
    use_value_reg_loss: bool = False      # MSE(value, llm_value_estimate)
    value_reg_coef: float = 0.10         # Value reg weight (decays with anneal)

    # ── Phase 42: Contrastive state representation ───────────────
    use_contrastive_loss: bool = False    # NT-Xent on phase-grouped states (model config, not FF)
    contrastive_coef: float = 0.05       # Contrastive loss weight


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
        # Phase 37: input_dim includes base state + optional LLM features
        effective_input_dim = config.state_dim + config.llm_feature_dim
        self.input_proj = nn.Linear(effective_input_dim, dims[0])
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

        # ── R77: Spectral normalization on critic ────────────────────
        # Constrains the Lipschitz constant of the value network,
        # preventing dramatic value predictions that destabilize PPO.
        # Applied to linear layers only (GELU/Dropout are unaffected).
        # NOTE: Applied AFTER _init_weights so orthogonal init works on raw weights,
        # then spectral norm wraps the initialized weights.
        self.use_spectral_norm_critic = config.use_spectral_norm_critic

        # ── R79: Auxiliary phase prediction head ─────────────────────
        # Predicts current attack phase group (recon=0, exploit=1, post=2)
        # from shared backbone features. Forces the representation to
        # encode phase information, improving actor and critic quality.
        self.has_phase_prediction = config.use_phase_prediction_aux
        if self.has_phase_prediction:
            self.phase_predictor = nn.Sequential(
                nn.Linear(final_dim, 32),
                nn.GELU(),
                nn.Linear(32, config.num_phase_groups),
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

        # R77: Apply spectral norm AFTER orthogonal init
        if self.use_spectral_norm_critic:
            for i, layer in enumerate(self.critic):
                if isinstance(layer, nn.Linear):
                    self.critic[i] = nn.utils.spectral_norm(layer)

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

        # R79: Cache backbone features for auxiliary heads (phase prediction)
        self._cached_features = x

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

    def get_logits(self, state: torch.Tensor) -> torch.Tensor:
        """Return raw action logits for BC loss computation (Phase 14.0)."""
        logits, _ = self.forward(state)
        return logits

    def get_backbone_features(self, state: torch.Tensor) -> torch.Tensor:
        """Return backbone features before actor/critic heads (Phase 42).

        Runs forward pass if needed and returns cached features.

        Args:
            state: (B, state_dim) tensor.

        Returns:
            (B, hidden_dims[-1]) backbone feature tensor.
        """
        self.forward(state)
        return self._cached_features

    def get_action_and_value(
        self, state: torch.Tensor, action: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
        logit_bias: Optional[torch.Tensor] = None,
        phase_group: Optional[torch.Tensor] = None,
        llm_prior: Optional[torch.Tensor] = None,
        prior_alpha: float = 0.0,
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
            llm_prior: Optional (action_dim,) float tensor. Phase 37: LLM
                action preference vector, added to logits weighted by prior_alpha.
            prior_alpha: Weight for LLM prior injection (decays via anneal).
        """
        logits, value = self.forward(state, phase_group=phase_group)

        # R67: Apply logit bias (soft penalty for used commands)
        if logit_bias is not None:
            bias = logit_bias.to(logits.device)
            if bias.dim() == 1:
                bias = bias.unsqueeze(0).expand_as(logits)
            logits = logits + bias

        # Phase 37: Apply LLM action prior — weighted preference injection
        # prior_alpha controls influence strength (decays via teacher anneal)
        if llm_prior is not None and prior_alpha > 0.0:
            lp = llm_prior.to(logits.device)
            if lp.dim() == 1:
                lp = lp.unsqueeze(0).expand_as(logits)
            logits = logits + prior_alpha * lp

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

    def predict_phase(self, state: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """R79: Predict phase group from cached backbone features.

        Returns logits (B, num_phase_groups) for cross-entropy loss.
        Uses features cached by the most recent forward() call, so
        gradients flow back through the shared backbone — the auxiliary
        loss improves representation quality for both actor and critic.

        Args:
            state: Optional fallback. If no cached features, runs forward().

        Returns:
            Phase prediction logits, or None if phase prediction disabled.
        """
        if not self.has_phase_prediction:
            return None
        features = getattr(self, '_cached_features', None)
        if features is None:
            if state is not None:
                self.forward(state)
                features = self._cached_features
            else:
                return None
        return self.phase_predictor(features)


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
        # Phase 37: Teacher distributions for KL loss + teacher actions for ranking
        self.teacher_distributions: List[Optional[torch.Tensor]] = []
        self.teacher_actions: List[Optional[int]] = []
        self.size = 0

    def add(
        self,
        state: torch.Tensor,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
        teacher_distribution: Optional[torch.Tensor] = None,
        teacher_action: Optional[int] = None,
    ):
        """Add a single transition."""
        self.states.append(state.detach().cpu())
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        # Phase 37: Store teacher data for auxiliary losses
        self.teacher_distributions.append(
            teacher_distribution.detach().cpu() if teacher_distribution is not None else None
        )
        self.teacher_actions.append(teacher_action)
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
        prioritized: bool = False,
    ):
        """Yield minibatches for PPO epochs.

        R72: When ``prioritized=True``, samples transitions proportional
        to ``|advantage|`` so the most informative transitions are seen
        more often. Standard PPO uses uniform sampling; this is a soft
        prioritization that doesn't change the loss, only the sampling
        distribution within each epoch.

        Yields:
            dict with keys: states, actions, old_log_probs, returns, advantages, values
        """
        n = self.size

        if prioritized and n > minibatch_size:
            # R72: Priority sampling by |advantage|
            adv_abs = np.abs(advantages.numpy()) + 1e-6
            probs = adv_abs / adv_abs.sum()
            indices = np.random.choice(n, size=n, replace=True, p=probs)
        else:
            indices = np.arange(n)
            np.random.shuffle(indices)

        states_t = torch.stack(self.states, dim=0)
        actions_t = torch.tensor(self.actions, dtype=torch.long)
        old_log_probs_t = torch.tensor(self.log_probs, dtype=torch.float32)
        values_t = torch.tensor(self.values, dtype=torch.float32)

        # Phase 37: Pre-stack teacher distributions (None → zeros)
        action_dim = 5  # default
        has_any_teacher = any(td is not None for td in self.teacher_distributions)
        if has_any_teacher:
            teacher_dists_t = torch.stack([
                td if td is not None else torch.zeros(action_dim)
                for td in self.teacher_distributions
            ])
            teacher_has_mask = torch.tensor([
                td is not None for td in self.teacher_distributions
            ], dtype=torch.bool)
        else:
            teacher_dists_t = None
            teacher_has_mask = None

        has_any_teacher_action = any(ta is not None for ta in self.teacher_actions)
        if has_any_teacher_action:
            teacher_actions_t = torch.tensor([
                ta if ta is not None else -1
                for ta in self.teacher_actions
            ], dtype=torch.long)
        else:
            teacher_actions_t = None

        for start in range(0, n, minibatch_size):
            end = min(start + minibatch_size, n)
            mb_idx = indices[start:end]

            batch = {
                "states": states_t[mb_idx].to(device),
                "actions": actions_t[mb_idx].to(device),
                "old_log_probs": old_log_probs_t[mb_idx].to(device),
                "returns": returns[mb_idx].to(device),
                "advantages": advantages[mb_idx].to(device),
                "values": values_t[mb_idx].to(device),
            }
            # Phase 37: Include teacher data if available
            if teacher_dists_t is not None:
                batch["teacher_distributions"] = teacher_dists_t[mb_idx].to(device)
                batch["teacher_has_mask"] = teacher_has_mask[mb_idx].to(device)
            if teacher_actions_t is not None:
                batch["teacher_actions"] = teacher_actions_t[mb_idx].to(device)

            yield batch

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

        # Phase 13.0: Lowered threshold — store ANY episode with positive return.
        # Original SIL required ep_return > baseline, which filters too aggressively
        # in early learning when baseline is low. A positive return means the agent
        # did SOMETHING right — capture those transitions for imitation.
        if ep_return <= 0:
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

        # ── R75: EMA target network ──────────────────────────────────
        # Polyak-averaged copy for stable value bootstrapping + surprise
        self.ema_network = None
        if self.config.use_ema_anchor:
            import copy
            self.ema_network = copy.deepcopy(self.network)
            self.ema_network.eval()
            for p in self.ema_network.parameters():
                p.requires_grad = False

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

        # ── R80: Adaptive clip + entropy rebound tracking ────────────
        self._clip_fraction_history: List[float] = []
        self._entropy_below_count: int = 0

        # ── R70: Self-Imitation Learning buffer ──────────────────────
        self.sil_buffer = SILBuffer(capacity=self.config.sil_buffer_size)

        # ── Phase 42: Contrastive state representation ───────────────
        self._contrastive_loss: Optional[object] = None
        if self.config.use_contrastive_loss:
            self._ensure_contrastive_loss()

    # ── Phase 42: Contrastive loss init ──────────────────────────────

    def _ensure_contrastive_loss(self) -> None:
        """Lazy-init ContrastiveLoss module."""
        if self._contrastive_loss is not None:
            return
        try:
            from core.feature_flags import get_feature_flags
            if not get_feature_flags().contrastive_ppo:
                return
            from core.algorithms.contrastive_state import (
                ContrastiveLoss, ContrastiveConfig,
            )
            cfg = ContrastiveConfig(
                enabled=True,
                coef=self.config.contrastive_coef,
                feature_dim=self.config.hidden_dims[-1],
            )
            self._contrastive_loss = ContrastiveLoss(cfg).to(self.device)
            # Add to optimizer
            self.optimizer.add_param_group(
                {"params": self._contrastive_loss.parameters(), "lr": self.config.learning_rate}
            )
            logger.info("Phase 42: ContrastiveLoss wired into PPO")
        except Exception as exc:
            logger.warning("Phase 42: ContrastiveLoss init failed: %s", exc)
            self._contrastive_loss = None

    # ── Action Selection ─────────────────────────────────────────────

    @torch.no_grad()
    def select_action(
        self, state_tensor: torch.Tensor, training: bool = True,
        action_mask: Optional[torch.Tensor] = None,
        logit_bias: Optional[torch.Tensor] = None,
        phase_group: Optional[int] = None,
        llm_prior: Optional[torch.Tensor] = None,
        prior_alpha: float = 0.0,
    ) -> Tuple[int, float, float]:
        """Select an action from the current policy with optional masking.

        Args:
            state_tensor: (state_dim,) tensor from the state encoder.
                Phase 37: May be (state_dim + llm_feature_dim,) for enhanced state.
            training: If True, sample from the distribution.
                      If False, take the greedy (argmax) action.
            action_mask: Optional (action_dim,) bool tensor.
                True = valid, False = masked (logit → -inf).
            logit_bias: Optional (action_dim,) float tensor. R67: Added to
                logits before softmax for soft-penalizing repeated commands.
            phase_group: Optional int (0=recon, 1=exploit, 2=post). R68:
                Phase group for phase-gated heads. Auto-extracted if None.
            llm_prior: Optional (action_dim,) float tensor. Phase 37: LLM
                action preference vector, injected into logits before softmax.
            prior_alpha: Weight for the LLM prior (decays via teacher anneal).

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
            # Phase 37: LLM prior injection in greedy mode
            if llm_prior is not None and prior_alpha > 0.0:
                lp = llm_prior.to(logits.device)
                if lp.dim() == 1:
                    lp = lp.unsqueeze(0).expand_as(logits)
                logits = logits + prior_alpha * lp
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
                llm_prior=llm_prior, prior_alpha=prior_alpha,
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

    @staticmethod
    def _symlog(x: float) -> float:
        """R71: Symmetric logarithm — compresses extreme values.

        Applies sign(x) * log(1 + |x|) which maps (-∞,+∞) to a compressed
        range while preserving sign and ordering. Used by DreamerV3
        (Hafner et al., 2023) to stabilize value learning across reward
        scales that vary by orders of magnitude.

        Our rewards range from -5 to ~230 per step. Symlog compresses:
          -5   → -1.79
          +3   → +1.39
          +75  → +4.33
          +230 → +5.44
        This makes the value function's job much easier.
        """
        return math.copysign(math.log1p(abs(x)), x)

    @torch.no_grad()
    def _get_value_surprise(self, state_tensor: torch.Tensor) -> float:
        """R75: Compute value disagreement between online and EMA networks.

        High disagreement indicates an uncertain/novel state worth exploring.
        Returns min(2.0, |V_online - V_ema|) as intrinsic exploration bonus.
        The cap prevents extreme bonuses from destabilizing reward normalization.
        """
        if self.ema_network is None:
            return 0.0
        s = state_tensor
        if s.dim() == 1:
            s = s.unsqueeze(0)
        s = s.to(self.device)
        _, v_online = self.network(s)
        _, v_ema = self.ema_network(s)
        surprise = abs(v_online.item() - v_ema.item())
        return min(2.0, surprise)

    def normalize_reward(self, reward: float, state_tensor: Optional[torch.Tensor] = None) -> float:
        """Normalize reward using running statistics.

        R71: Optionally applies symlog compression BEFORE normalization.
        R75: Adds value-surprise intrinsic bonus from EMA network disagreement.

        This multi-stage pipeline (compress → surprise bonus → normalize)
        prevents extreme rewards from dominating the value function while
        encouraging exploration of uncertain states.
        """
        # R71: Symlog compression (DreamerV3)
        r = self._symlog(reward) if self.config.use_symlog else reward

        # R75: Value-surprise intrinsic bonus
        if self.config.use_ema_anchor and state_tensor is not None and self.ema_network is not None:
            surprise = self._get_value_surprise(state_tensor)
            r += surprise * self.config.value_surprise_coef

        self._update_reward_stats(r)
        std = max(math.sqrt(abs(self._reward_var)), 1e-4)
        return (r - self._reward_mean) / std

    def store_transition(
        self,
        state: torch.Tensor,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
        teacher_distribution: Optional[torch.Tensor] = None,
        teacher_action: Optional[int] = None,
    ):
        """Store a single transition in the rollout buffer.

        Phase 6: Rewards are normalized using running statistics before
        storage, which stabilizes the value function and prevents the
        ~20,000 value loss we saw with raw rewards.
        R75: State tensor passed for value-surprise intrinsic bonus.
        Phase 37: Optionally stores teacher distribution + action for
        KL distillation and ranking loss during PPO update.
        """
        normalized_reward = self.normalize_reward(reward, state_tensor=state)
        self.buffer.add(
            state, action, log_prob, normalized_reward, value, done,
            teacher_distribution=teacher_distribution,
            teacher_action=teacher_action,
        )
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

        # R76: Dual-horizon GAE blending
        # Blend long-horizon (λ=0.95, captures delayed rewards) with
        # short-horizon (λ=0.70, faster credit assignment) advantages.
        # Short-horizon helps PPO learn immediate cause-effect (discovery
        # commands → reward), while long-horizon preserves multi-step
        # planning signal (recon → exploit → root shell chain).
        if self.config.use_dual_horizon_gae:
            returns_s, advantages_s = self.buffer.compute_returns_and_advantages(
                last_value=last_value,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda_short,
            )
            blend = self.config.dual_horizon_blend
            advantages = blend * advantages + (1 - blend) * advantages_s
            returns = blend * returns + (1 - blend) * returns_s

        # Track return statistics for monitoring
        for r in returns.numpy():
            self._update_return_stats(float(r))

        # Normalise advantages (critical for PPO stability)
        adv_std = advantages.std()
        if adv_std > 1e-8:
            advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

        # R70: Clip normalized advantages to prevent gradient explosion
        # R77: Soft tanh clip — smooth gradients near boundary (vs hard clamp)
        if self.config.advantage_clip > 0:
            if self.config.use_soft_advantage_clip:
                # tanh(x/c)*c maps to (-c, c) with smooth gradients everywhere
                _clip = self.config.advantage_clip
                advantages = _clip * torch.tanh(advantages / _clip)
            else:
                advantages = torch.clamp(
                    advantages, -self.config.advantage_clip, self.config.advantage_clip
                )

        # R74: Per-phase-group advantage whitening
        # Normalize advantages separately within recon/exploit/post-exploit
        # groups so exploitation steps don't dominate learning signal.
        if self.config.use_phase_advantage_whitening and self.buffer.size > 3:
            try:
                states_t = torch.stack(self.buffer.states, dim=0)
                phase_groups = PPOActorCritic._extract_phase_group(states_t)
                for g in range(3):
                    mask_g = (phase_groups == g)
                    if mask_g.sum() > 1:
                        adv_g = advantages[mask_g]
                        adv_g_std = adv_g.std()
                        if adv_g_std > 1e-8:
                            advantages[mask_g] = (adv_g - adv_g.mean()) / (adv_g_std + 1e-8)
            except Exception:
                pass  # Fall back to global normalization

        # ── Multi-Epoch PPO ──────────────────────────────────────────
        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            # Phase 37 Level 5 auxiliary losses
            "kl_teacher_loss": 0.0,
            "ranking_loss": 0.0,
            "value_reg_loss": 0.0,
        }
        num_batches = 0

        # R73: Gradient accumulation — effective batch = minibatch_size × grad_accum_steps
        _grad_accum = max(1, self.config.grad_accum_steps)

        for _epoch in range(self.config.epochs_per_update):
            self.optimizer.zero_grad()  # R73: Zero once at epoch start
            _accum_idx = 0

            for batch in self.buffer.get_batches(
                returns, advantages, self.config.minibatch_size, self.device,
                prioritized=self.config.use_prioritized_advantages,
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
                # R73: Policy ramp-up — let critic calibrate first
                _policy_ramp = min(1.0, (self.updates_done + 1) / max(self.config.policy_warmup_updates, 1))
                loss = (
                    _policy_ramp * policy_loss
                    + self.config.value_loss_coef * value_loss
                    - self.entropy_coef * entropy_loss
                )

                # ── R79: Phase prediction auxiliary loss ─────────────
                # Predicts phase group from shared features. Gradients
                # flow back through backbone → better representations.
                if self.config.use_phase_prediction_aux and self.network.has_phase_prediction:
                    phase_pred = self.network.predict_phase()
                    if phase_pred is not None:
                        phase_targets = PPOActorCritic._extract_phase_group(batch["states"])
                        phase_aux_loss = F.cross_entropy(phase_pred, phase_targets)
                        loss = loss + self.config.phase_prediction_coef * phase_aux_loss

                # ── Phase 14.0: Behavioral Cloning loss ──────────────
                # Distills mentor demonstrations into PPO policy.
                # Samples from BCBuffer and adds cross-entropy loss
                # between policy output and teacher actions.
                if self.config.use_bc_loss and self.config.bc_buffer is not None:
                    try:
                        bc_buf = self.config.bc_buffer
                        _bc_batch_size = min(8, bc_buf.get_stats()["size"])
                        if _bc_batch_size >= 2:
                            bc_samples = bc_buf.sample(_bc_batch_size)
                            bc_states = torch.stack([s.state for s in bc_samples]).to(self.device)
                            bc_teacher_actions = torch.tensor(
                                [s.teacher_action for s in bc_samples],
                                dtype=torch.long, device=self.device
                            )
                            bc_weights = torch.tensor(
                                [s.weight for s in bc_samples],
                                dtype=torch.float32, device=self.device
                            )
                            # Get policy logits for BC states
                            bc_logits = self.network.get_logits(bc_states)
                            bc_log_probs = F.log_softmax(bc_logits, dim=-1)
                            # Weighted negative log-likelihood
                            bc_nll = F.nll_loss(bc_log_probs, bc_teacher_actions, reduction="none")
                            bc_loss = (bc_nll * bc_weights).mean()
                            loss = loss + self.config.bc_loss_coef * bc_loss
                    except Exception:
                        pass  # BC loss is non-critical; skip on error

                # ── Phase 37 Level 5: KL Teacher Distillation Loss ───
                # KL(policy || teacher) — forces policy to stay close to
                # the teacher's action preference distribution. Weight
                # decays via teacher anneal schedule, allowing the policy
                # to internalize knowledge and gradually detach.
                _kl_teacher_val = 0.0
                if self.config.use_kl_teacher_loss:
                    teacher_dists = batch.get("teacher_distributions")
                    teacher_mask = batch.get("teacher_has_mask")
                    if teacher_dists is not None and teacher_mask is not None:
                        if teacher_mask.any():
                            # Get current policy distribution
                            logits_for_kl, _ = self.network.forward(
                                batch["states"][teacher_mask]
                            )
                            policy_log_probs = F.log_softmax(logits_for_kl, dim=-1)
                            teacher_probs = teacher_dists[teacher_mask]
                            # KL(teacher || policy) — teacher is the reference
                            kl_loss = F.kl_div(
                                policy_log_probs, teacher_probs,
                                reduction="batchmean", log_target=False,
                            )
                            loss = loss + self.config.kl_teacher_coef * kl_loss
                            _kl_teacher_val = kl_loss.item()

                # ── Phase 37 Level 5: Ranking Margin Loss ────────────
                # Ensures the teacher's preferred action has higher logit
                # than non-preferred actions by at least a margin.
                # loss = max(0, margin - (logit[teacher] - max(logit[others])))
                _ranking_val = 0.0
                if self.config.use_ranking_loss:
                    teacher_acts = batch.get("teacher_actions")
                    if teacher_acts is not None:
                        valid_teacher = teacher_acts >= 0
                        if valid_teacher.any():
                            rank_logits, _ = self.network.forward(
                                batch["states"][valid_teacher]
                            )
                            t_acts = teacher_acts[valid_teacher]
                            # Gather teacher action logits
                            teacher_logit = rank_logits.gather(
                                1, t_acts.unsqueeze(1)
                            ).squeeze(1)
                            # Max logit among non-teacher actions
                            mask_others = torch.ones_like(rank_logits, dtype=torch.bool)
                            for i, ta in enumerate(t_acts):
                                mask_others[i, ta] = False
                            other_logits = rank_logits.clone()
                            other_logits[~mask_others] = float("-inf")
                            max_other = other_logits.max(dim=1).values
                            # Margin loss: hinge
                            margin = self.config.ranking_margin
                            rank_loss = torch.clamp(
                                margin - (teacher_logit - max_other), min=0.0
                            ).mean()
                            loss = loss + self.config.ranking_loss_coef * rank_loss
                            _ranking_val = rank_loss.item()

                # ── Phase 37 Level 5: Value Regularization Loss ──────
                # Soft-aligns the critic's value estimate with the teacher's
                # confidence signal, expressed as a scaled return estimate.
                # This accelerates value calibration in early training.
                _value_reg_val = 0.0
                if self.config.use_value_reg_loss:
                    teacher_dists = batch.get("teacher_distributions")
                    teacher_mask = batch.get("teacher_has_mask")
                    if teacher_dists is not None and teacher_mask is not None:
                        if teacher_mask.any():
                            # Teacher confidence → approximate value target
                            # Higher teacher confidence → value should be higher
                            t_conf = teacher_dists[teacher_mask].max(dim=-1).values
                            # Scale to match return magnitude (~normalized)
                            llm_value_target = t_conf * 2.0 - 1.0  # map [0,1] to [-1,1]
                            _, v_pred = self.network.forward(
                                batch["states"][teacher_mask]
                            )
                            v_pred = v_pred.squeeze(-1)
                            vreg_loss = F.mse_loss(v_pred, llm_value_target)
                            loss = loss + self.config.value_reg_coef * vreg_loss
                            _value_reg_val = vreg_loss.item()

                # ── Phase 42: Contrastive state loss ─────────────────
                _contrastive_val = 0.0
                if (
                    self.config.use_contrastive_loss
                    and self._contrastive_loss is not None
                ):
                    try:
                        backbone_feats = self.network.get_backbone_features(
                            batch["states"]
                        )
                        phase_labels = PPOActorCritic._extract_phase_group(
                            batch["states"]
                        )
                        c_loss = self._contrastive_loss.compute_loss(
                            backbone_feats, phase_labels
                        )
                        if c_loss.item() > 0:
                            loss = loss + self.config.contrastive_coef * c_loss
                            _contrastive_val = c_loss.item()
                    except Exception:
                        pass  # Contrastive loss is non-critical

                # ── Backward (R73: gradient accumulation) ────────────
                (loss / _grad_accum).backward()
                _accum_idx += 1

                if _accum_idx % _grad_accum == 0:
                    nn.utils.clip_grad_norm_(
                        self.network.parameters(), self.config.max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # R75: Polyak-update EMA target network
                    if self.ema_network is not None:
                        _tau = self.config.ema_tau
                        for p_online, p_ema in zip(
                            self.network.parameters(), self.ema_network.parameters()
                        ):
                            p_ema.data.mul_(_tau).add_(p_online.data, alpha=1 - _tau)

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
                # Phase 37 Level 5 metrics
                metrics["kl_teacher_loss"] += _kl_teacher_val
                metrics["ranking_loss"] += _ranking_val
                metrics["value_reg_loss"] += _value_reg_val
                metrics["contrastive_loss"] = metrics.get("contrastive_loss", 0.0) + _contrastive_val
                num_batches += 1

            # R73: Step on any remaining accumulated gradients
            if _accum_idx % _grad_accum != 0:
                nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()

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

        # R74: KL-adaptive learning rate
        # If KL is too low → policy barely changed → increase LR to learn faster.
        # If KL is too high → policy changed too much → decrease LR for stability.
        # Standard technique from OpenAI's PPO implementation.
        if self.config.use_kl_adaptive_lr and num_batches > 0:
            recent_kl = metrics["approx_kl"]
            _initial_lr = self.config.learning_rate
            for pg in self.optimizer.param_groups:
                if recent_kl < 0.5 * self.config.target_kl:
                    pg["lr"] = min(pg["lr"] * 1.5, _initial_lr * 2.0)
                elif recent_kl > 2.0 * self.config.target_kl:
                    pg["lr"] = max(pg["lr"] * 0.5, self.config.lr_min)

        # R80: Adaptive clip scheduling
        # Tracks rolling clip_fraction to auto-adjust clip_epsilon:
        # - <10% clipping → too conservative → widen clip
        # - >30% clipping → too aggressive → narrow clip
        # This keeps the policy update magnitude in the sweet spot.
        if self.config.use_adaptive_clip and num_batches > 0:
            self._clip_fraction_history.append(metrics["clip_fraction"])
            if len(self._clip_fraction_history) > 5:
                self._clip_fraction_history.pop(0)
            rolling_clip = sum(self._clip_fraction_history) / len(self._clip_fraction_history)
            if rolling_clip < 0.1:
                self.config.clip_epsilon = min(
                    self.config.clip_epsilon * 1.05, self.config.clip_epsilon_max
                )
            elif rolling_clip > 0.3:
                self.config.clip_epsilon = max(
                    self.config.clip_epsilon * 0.95, self.config.clip_epsilon_min
                )

        # Anneal entropy coefficient
        # R58 Layer 2c: Apply adaptive multiplier on top of base annealing
        # R71: Cosine schedule stays high longer then decays smoothly,
        # providing more exploration in early training vs linear decay.
        progress = min(self.total_steps / self.config.total_timesteps, 1.0)
        if self.config.use_cosine_entropy:
            # Cosine: starts at entropy_coef, smoothly decays to entropy_coef_min
            # cos(0) = 1 → full exploration, cos(π) = -1 → minimum exploration
            base_entropy = self.config.entropy_coef_min + 0.5 * (
                self.config.entropy_coef - self.config.entropy_coef_min
            ) * (1.0 + math.cos(math.pi * progress))
        else:
            # Original linear decay
            base_entropy = self.config.entropy_coef + progress * (
                self.config.entropy_coef_min - self.config.entropy_coef
            )
        self.entropy_coef = base_entropy * self._entropy_adaptive_multiplier

        # R72: Return-variance entropy coupling
        # If return variance is low AND mean return is high, the agent has
        # found a stable good strategy — decay entropy faster. If variance
        # is high, keep entropy elevated to encourage further exploration.
        if self.config.return_variance_entropy and self._return_count > 10:
            return_std = math.sqrt(abs(self._return_var))
            return_cv = return_std / max(abs(self._return_mean), 1e-4)
            if return_cv < 0.3 and self._return_mean > 0:
                # Stable positive returns — reduce entropy by up to 30%
                stability_factor = max(0.7, 1.0 - (0.3 - return_cv))
                self.entropy_coef *= stability_factor
            elif return_cv > 1.0:
                # Highly variable — boost entropy by up to 20%
                self.entropy_coef *= min(1.2, 1.0 + (return_cv - 1.0) * 0.1)

        # R80: Entropy rebound
        # If entropy drops below threshold for 3+ consecutive updates,
        # the policy has collapsed to near-deterministic — boost entropy
        # to prevent premature convergence and re-enable exploration.
        if self.config.use_entropy_rebound and num_batches > 0:
            if metrics["entropy"] < self.config.entropy_rebound_threshold:
                self._entropy_below_count += 1
                if self._entropy_below_count >= 3:
                    self.entropy_coef = max(
                        self.entropy_coef, self.config.entropy_coef * 0.7
                    )
                    self._entropy_below_count = 0
            else:
                self._entropy_below_count = 0

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
                # R71: symlog + cosine entropy config
                "use_symlog": self.config.use_symlog,
                "use_cosine_entropy": self.config.use_cosine_entropy,
                # R75: EMA network state
                "ema_network_state": self.ema_network.state_dict() if self.ema_network else None,
                # R80: Adaptive clip + entropy rebound state
                "clip_epsilon_current": self.config.clip_epsilon,
                "clip_fraction_history": self._clip_fraction_history,
                "entropy_below_count": self._entropy_below_count,
            },
            path,
        )

    def load(self, path: str):
        """Load PPO checkpoint with architecture migration support."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.load_from_state_dict(ckpt)

    def load_from_state_dict(self, ckpt: Dict[str, Any]) -> None:
        """Load from an in-memory state dict (same format as ``save()``).

        Core restore logic shared by ``load()`` (from file) and
        ``UnifiedCheckpoint.apply_ppo()`` (from memory).  Handles
        architecture migration (spectral norm, phase predictor) and
        restores all R58-R80 adaptive state.
        """
        try:
            self.network.load_state_dict(ckpt["network_state_dict"])
        except RuntimeError:
            logger.warning("Architecture mismatch — loading with strict=False")
            self.network.load_state_dict(ckpt["network_state_dict"], strict=False)
        try:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except (RuntimeError, ValueError):
            logger.warning("Optimizer state mismatch — reinitializing optimizer")
            self.optimizer = torch.optim.Adam(
                self.network.parameters(), lr=self.config.learning_rate
            )
        if "scheduler_state_dict" in ckpt:
            self.lr_scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.total_steps = ckpt.get("total_steps", 0)
        self.updates_done = ckpt.get("updates_done", 0)
        self.entropy_coef = ckpt.get("entropy_coef", self.config.entropy_coef)
        if "reward_norm" in ckpt:
            rn = ckpt["reward_norm"]
            self._reward_mean = rn.get("mean", 0.0)
            self._reward_var = rn.get("var", 1.0)
            self._reward_count = rn.get("count", 0)
            self._return_mean = rn.get("return_mean", 0.0)
            self._return_var = rn.get("return_var", 1.0)
            self._return_count = rn.get("return_count", 0)
        if "adaptive_entropy" in ckpt:
            ae = ckpt["adaptive_entropy"]
            self._entropy_adaptive_multiplier = ae.get("multiplier", 1.0)
            self._consecutive_closeouts = ae.get("consecutive_closeouts", 0)
            self._consecutive_failures = ae.get("consecutive_failures", 0)
        if "sil_baseline" in ckpt:
            self.sil_buffer._return_baseline = ckpt["sil_baseline"]
            self.sil_buffer._return_count = ckpt.get("sil_count", 0)
        if "ema_network_state" in ckpt and self.ema_network is not None:
            try:
                self.ema_network.load_state_dict(ckpt["ema_network_state"])
            except RuntimeError:
                import copy
                self.ema_network = copy.deepcopy(self.network)
                for p in self.ema_network.parameters():
                    p.requires_grad = False
        if "clip_epsilon_current" in ckpt:
            self.config.clip_epsilon = ckpt["clip_epsilon_current"]
        if "clip_fraction_history" in ckpt:
            self._clip_fraction_history = ckpt["clip_fraction_history"]
        if "entropy_below_count" in ckpt:
            self._entropy_below_count = ckpt["entropy_below_count"]

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
            "symlog_enabled": self.config.use_symlog,
            "cosine_entropy": self.config.use_cosine_entropy,
            "ema_enabled": self.config.use_ema_anchor,
            "current_clip_epsilon": round(self.config.clip_epsilon, 4),
            "entropy_rebound_count": self._entropy_below_count,
            "dual_horizon_gae": self.config.use_dual_horizon_gae,
            "spectral_norm_critic": self.config.use_spectral_norm_critic,
            "phase_prediction_aux": self.config.use_phase_prediction_aux,
            # Phase 37 Level 5 GPT ↔ RL integration diagnostics
            "llm_feature_dim": self.config.llm_feature_dim,
            "use_llm_prior": self.config.use_llm_prior,
            "use_kl_teacher_loss": self.config.use_kl_teacher_loss,
            "use_ranking_loss": self.config.use_ranking_loss,
            "use_value_reg_loss": self.config.use_value_reg_loss,
            "kl_teacher_coef": self.config.kl_teacher_coef,
            "ranking_loss_coef": self.config.ranking_loss_coef,
            "value_reg_coef": self.config.value_reg_coef,
            "reward_norm": {
                "mean": round(self._reward_mean, 4),
                "std": round(math.sqrt(abs(self._reward_var)), 4),
                "count": self._reward_count,
            },
            "latest_metrics": {
                k: v[-1] if v else None for k, v in self.training_metrics.items()
            },
        }

    # ── Phase 15.0: Neuromodulation Hook ─────────────────────────────

    def apply_neuromodulation(
        self,
        entropy_coef_mult: float = 1.0,
        lr_mult: float = 1.0,
        bc_weight_mult: float = 1.0,
    ) -> None:
        """
        Apply neuromodulator-driven adjustments to PPO hyperparameters.

        Feature-flag gated: FF_NEUROMODULATORS (caller checks).
        All multipliers are bounded to prevent instability:
          - entropy_coef: ×0.5 to ×1.5
          - learning_rate: ×0.5 to ×2.0 (never exceeds base)
          - bc_loss_coef: ×0.5 to ×1.5

        Args:
            entropy_coef_mult: Multiplier for entropy coefficient.
            lr_mult: Multiplier for learning rate.
            bc_weight_mult: Multiplier for behavioral cloning weight.
        """
        # Bound entropy coef multiplier
        ent_mult = max(0.5, min(1.5, entropy_coef_mult))
        base_entropy = self.config.entropy_coef
        self.entropy_coef = max(
            self.config.entropy_coef_min,
            base_entropy * ent_mult * self._entropy_adaptive_multiplier,
        )

        # Bound LR multiplier — never exceed base config LR
        lr_mult_clamped = max(0.5, min(2.0, lr_mult))
        target_lr = self.config.learning_rate * lr_mult_clamped
        target_lr = max(self.config.lr_min, min(self.config.learning_rate, target_lr))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = target_lr

        # Bound BC weight multiplier
        if self.config.use_bc_loss:
            bc_mult = max(0.5, min(1.5, bc_weight_mult))
            self.config.bc_loss_coef = 0.1 * bc_mult  # base is 0.1

        logger.debug(
            f"[P15] Neuromod applied: ent={self.entropy_coef:.5f} "
            f"lr={target_lr:.2e} bc_coef={self.config.bc_loss_coef:.3f}"
        )
