"""
Soft Actor-Critic (SAC) Agent — Entropy-regularized reinforcement learning.

SAC is an off-policy algorithm that optimizes a stochastic policy with entropy
regularization for maximum expected reward + entropy. It provides:
- Better exploration via entropy bonus (avoids premature convergence)
- Sample efficiency through replay buffer (off-policy)
- Automatic temperature tuning (α auto-adjustment)

Architecture:
    - Actor: Stochastic policy π(a|s) with log_prob computation
    - Twin Critics: Q1(s,a), Q2(s,a) — min(Q1,Q2) reduces overestimation
    - Target Critics: Soft-updated copies for stable TD targets
    - Auto-α: Learnable temperature for entropy-reward tradeoff

Integration with Ariaska:
    SmartCoach pipeline:  Playbook → PPO/SAC → Registry → Mentor
    SAC maps discrete actions (action_dim=5) to commands via CommandActionMapper
    SAC's entropy bonus naturally encourages command diversity (reducing anti-repeat)

References:
    Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL" (2018)
    Haarnoja et al., "Soft Actor-Critic Algorithms and Applications" (2019)
"""

import logging
import math
import numpy as np
from collections import deque
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("ariaska.sac_agent")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available — SAC agent will be disabled")


@dataclass
class SACConfig:
    """Configuration for Soft Actor-Critic agent."""
    
    # Dimensions
    state_dim: int = 512          # Must match state_encoder output
    action_dim: int = 5           # Discrete actions mapped via CommandActionMapper
    hidden_dims: List[int] = field(default_factory=lambda: [512, 512, 256])
    
    # Core SAC hyperparameters
    gamma: float = 0.99           # Discount factor
    tau: float = 0.005            # Soft update rate for target networks
    alpha: float = 0.2            # Initial entropy temperature
    auto_alpha: bool = True       # Auto-tune alpha
    
    # Learning rates
    actor_lr: float = 3e-4        # Actor network learning rate
    critic_lr: float = 3e-4       # Critic network learning rate
    alpha_lr: float = 3e-4        # Alpha (temperature) learning rate
    lr_min: float = 1e-5          # Minimum LR after annealing
    
    # Replay buffer
    buffer_size: int = 50000      # Replay buffer capacity
    batch_size: int = 128         # Mini-batch size for updates
    min_buffer_size: int = 256    # Minimum buffer size before training
    
    # Training
    updates_per_step: int = 1     # Gradient updates per environment step
    target_entropy_ratio: float = 0.6  # Target entropy as ratio of max (-log(1/|A|))
    max_grad_norm: float = 1.0    # Gradient clipping
    
    # Exploration
    warmup_steps: int = 100       # Random actions before policy kicks in
    
    def __post_init__(self):
        if isinstance(self.hidden_dims, tuple):
            self.hidden_dims = list(self.hidden_dims)


if TORCH_AVAILABLE:
    
    class DiscreteActor(nn.Module):
        """
        Stochastic actor for discrete action spaces.
        
        Outputs categorical probabilities over actions with log_prob computation
        for the SAC entropy objective.
        """
        
        def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
            super().__init__()
            
            layers = []
            prev_dim = state_dim
            for h_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, h_dim),
                    nn.LayerNorm(h_dim),
                    nn.ReLU(),
                    nn.Dropout(0.05),
                ])
                prev_dim = h_dim
            
            self.backbone = nn.Sequential(*layers)
            self.action_head = nn.Linear(prev_dim, action_dim)
        
        def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Forward pass: compute action probabilities and log probabilities.
            
            Returns:
                probs: Action probabilities [batch, action_dim]
                log_probs: Log probabilities [batch, action_dim]
            """
            features = self.backbone(state)
            logits = self.action_head(features)
            
            # Stable softmax with temperature
            probs = F.softmax(logits, dim=-1)
            # Avoid log(0) with clamping
            log_probs = torch.log(probs.clamp(min=1e-8))
            
            return probs, log_probs
        
        def get_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
            """
            Sample action from policy.
            
            Returns:
                action: Sampled action index
                log_prob: Log probability of the action
                probs: Full probability distribution
            """
            probs, log_probs = self.forward(state)
            
            # Sample from categorical distribution
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            
            # Get log prob of sampled action
            log_prob = log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1)
            
            return action.item(), log_prob, probs
    
    
    class TwinCritic(nn.Module):
        """
        Twin Q-networks for discrete SAC.
        
        Two independent Q-networks to reduce overestimation bias.
        Each outputs Q(s,a) for all actions simultaneously.
        """
        
        def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
            super().__init__()
            
            # Q1 network
            q1_layers = []
            prev_dim = state_dim
            for h_dim in hidden_dims:
                q1_layers.extend([
                    nn.Linear(prev_dim, h_dim),
                    nn.LayerNorm(h_dim),
                    nn.ReLU(),
                ])
                prev_dim = h_dim
            q1_layers.append(nn.Linear(prev_dim, action_dim))
            self.q1 = nn.Sequential(*q1_layers)
            
            # Q2 network (independent)
            q2_layers = []
            prev_dim = state_dim
            for h_dim in hidden_dims:
                q2_layers.extend([
                    nn.Linear(prev_dim, h_dim),
                    nn.LayerNorm(h_dim),
                    nn.ReLU(),
                ])
                prev_dim = h_dim
            q2_layers.append(nn.Linear(prev_dim, action_dim))
            self.q2 = nn.Sequential(*q2_layers)
        
        def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Forward pass: compute Q-values from both critics.
            
            Returns:
                q1_values: Q-values from critic 1 [batch, action_dim]
                q2_values: Q-values from critic 2 [batch, action_dim]
            """
            return self.q1(state), self.q2(state)
    
    
    class SACReplayBuffer:
        """Simple replay buffer for SAC transitions."""
        
        def __init__(self, capacity: int = 50000):
            self.buffer = deque(maxlen=capacity)
        
        def push(self, state, action, reward, next_state, done):
            """Store transition."""
            self.buffer.append((state, action, reward, next_state, done))
        
        def sample(self, batch_size: int) -> Tuple:
            """Sample a batch of transitions."""
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            batch = [self.buffer[i] for i in indices]
            
            states = torch.stack([b[0] for b in batch])
            actions = torch.tensor([b[1] for b in batch], dtype=torch.long)
            rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32)
            next_states = torch.stack([b[3] for b in batch])
            dones = torch.tensor([b[4] for b in batch], dtype=torch.float32)
            
            return states, actions, rewards, next_states, dones
        
        def __len__(self):
            return len(self.buffer)


class SACAgent:
    """
    Soft Actor-Critic agent for discrete action spaces.
    
    Implements SAC with:
    - Discrete action categorical policy (actor)
    - Twin Q-critics with target networks
    - Automatic entropy temperature (alpha) tuning
    - Replay buffer for off-policy learning
    
    Usage:
        config = SACConfig(state_dim=512, action_dim=5)
        sac = SACAgent(config)
        
        # During training:
        action, log_prob = sac.select_action(state_tensor)
        sac.store_transition(state, action, reward, next_state, done)
        metrics = sac.update()  # Returns loss metrics
    """
    
    def __init__(self, config: Optional[SACConfig] = None):
        """Initialize SAC agent with given config."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for SACAgent. Install with: pip install torch")
        
        self.config = config or SACConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.actor = DiscreteActor(
            self.config.state_dim, self.config.action_dim, self.config.hidden_dims
        ).to(self.device)
        
        self.critic = TwinCritic(
            self.config.state_dim, self.config.action_dim, self.config.hidden_dims
        ).to(self.device)
        
        self.target_critic = TwinCritic(
            self.config.state_dim, self.config.action_dim, self.config.hidden_dims
        ).to(self.device)
        
        # Copy parameters to target
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config.critic_lr)
        
        # Auto-alpha (entropy temperature)
        self.target_entropy = -self.config.target_entropy_ratio * math.log(1.0 / self.config.action_dim)
        if self.config.auto_alpha:
            self.log_alpha = torch.tensor(
                math.log(self.config.alpha), dtype=torch.float32, requires_grad=True, device=self.device
            )
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config.alpha_lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = self.config.alpha
            self.log_alpha = None
            self.alpha_optimizer = None
        
        # Replay buffer
        self.replay_buffer = SACReplayBuffer(self.config.buffer_size)
        
        # Training stats
        self._step_count = 0
        self._update_count = 0
        self.metrics = {
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "alpha": self.alpha,
            "alpha_loss": 0.0,
            "entropy": 0.0,
            "q_mean": 0.0,
        }
        
        logger.info(
            f"SACAgent initialized | state_dim={self.config.state_dim} | "
            f"action_dim={self.config.action_dim} | α={self.alpha:.3f} | "
            f"device={self.device}"
        )
    
    def select_action(
        self, state: torch.Tensor, evaluate: bool = False
    ) -> Tuple[int, torch.Tensor]:
        """
        Select action given current state.
        
        Args:
            state: State tensor [state_dim] or [1, state_dim]
            evaluate: If True, use greedy (argmax) instead of sampling
            
        Returns:
            action: Selected action index
            log_prob: Log probability of the action
        """
        self._step_count += 1
        
        # Warmup: random actions
        if self._step_count < self.config.warmup_steps and not evaluate:
            action = np.random.randint(0, self.config.action_dim)
            return action, torch.tensor(0.0)
        
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            state = state.to(self.device)
            
            probs, log_probs = self.actor(state)
            
            if evaluate:
                action = probs.argmax(dim=-1).item()
                log_prob = log_probs.gather(1, torch.tensor([[action]]).to(self.device)).squeeze()
            else:
                dist = torch.distributions.Categorical(probs)
                action_tensor = dist.sample()
                action = action_tensor.item()
                log_prob = log_probs.gather(1, action_tensor.unsqueeze(-1)).squeeze()
        
        return action, log_prob
    
    def store_transition(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        """Store transition in replay buffer."""
        if state.dim() > 1:
            state = state.squeeze(0)
        if next_state.dim() > 1:
            next_state = next_state.squeeze(0)
        
        self.replay_buffer.push(
            state.detach().cpu(),
            action,
            reward,
            next_state.detach().cpu(),
            float(done),
        )
    
    def update(self) -> Dict[str, float]:
        """
        Perform SAC update step.
        
        Returns:
            Dict with loss metrics (actor_loss, critic_loss, alpha, entropy, etc.)
        """
        if len(self.replay_buffer) < self.config.min_buffer_size:
            return self.metrics
        
        for _ in range(self.config.updates_per_step):
            # Sample batch
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(
                min(self.config.batch_size, len(self.replay_buffer))
            )
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            next_states = next_states.to(self.device)
            dones = dones.to(self.device)
            
            # ====== Update Critics ======
            with torch.no_grad():
                # Get next action probabilities
                next_probs, next_log_probs = self.actor(next_states)
                
                # Target Q-values
                next_q1, next_q2 = self.target_critic(next_states)
                next_q_min = torch.min(next_q1, next_q2)
                
                # Soft target value: V(s') = E_a[Q(s',a) - α * log π(a|s')]
                next_v = (next_probs * (next_q_min - self.alpha * next_log_probs)).sum(dim=-1)
                
                # TD target
                target_q = rewards + (1.0 - dones) * self.config.gamma * next_v
            
            # Current Q-values
            q1_all, q2_all = self.critic(states)
            q1 = q1_all.gather(1, actions.unsqueeze(-1)).squeeze(-1)
            q2 = q2_all.gather(1, actions.unsqueeze(-1)).squeeze(-1)
            
            # Critic loss (MSE)
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
            self.critic_optimizer.step()
            
            # ====== Update Actor ======
            probs, log_probs = self.actor(states)
            q1_pi, q2_pi = self.critic(states)
            q_min_pi = torch.min(q1_pi, q2_pi)
            
            # Actor loss: minimize α*log_π - Q
            # For discrete: sum over actions weighted by policy probs
            actor_loss = (probs * (self.alpha * log_probs - q_min_pi)).sum(dim=-1).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
            self.actor_optimizer.step()
            
            # ====== Update Alpha (entropy temperature) ======
            alpha_loss = torch.tensor(0.0)
            if self.config.auto_alpha and self.log_alpha is not None:
                # Entropy of current policy
                entropy = -(probs.detach() * log_probs.detach()).sum(dim=-1).mean()
                
                # Alpha loss: minimize α * (entropy - target_entropy)
                alpha_loss = -(self.log_alpha * (entropy - self.target_entropy).detach()).mean()
                
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                
                self.alpha = self.log_alpha.exp().item()
            else:
                entropy = -(probs.detach() * log_probs.detach()).sum(dim=-1).mean()
            
            # ====== Soft Update Target Networks ======
            self._soft_update_target()
            
            self._update_count += 1
            
            # Update metrics
            self.metrics = {
                "actor_loss": actor_loss.item(),
                "critic_loss": critic_loss.item(),
                "alpha": self.alpha,
                "alpha_loss": alpha_loss.item() if isinstance(alpha_loss, torch.Tensor) else 0.0,
                "entropy": entropy.item(),
                "q_mean": q_min_pi.mean().item(),
            }
        
        return self.metrics
    
    def _soft_update_target(self) -> None:
        """Soft update target critic networks."""
        tau = self.config.tau
        for param, target_param in zip(
            self.critic.parameters(), self.target_critic.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    
    def get_action_probs(self, state: torch.Tensor) -> torch.Tensor:
        """Get action probabilities for a state (for compatibility with SmartCoach)."""
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            state = state.to(self.device)
            probs, _ = self.actor(state)
            return probs.squeeze(0)
    
    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.item() if self.log_alpha is not None else None,
            "alpha": self.alpha,
            "step_count": self._step_count,
            "update_count": self._update_count,
            "config": {
                "state_dim": self.config.state_dim,
                "action_dim": self.config.action_dim,
                "hidden_dims": self.config.hidden_dims,
            },
        }, path)
        logger.info(f"SAC model saved to {path}")
    
    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.target_critic.load_state_dict(checkpoint["target_critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        if checkpoint.get("log_alpha") is not None and self.log_alpha is not None:
            self.log_alpha.data = torch.tensor(checkpoint["log_alpha"])
            self.alpha = self.log_alpha.exp().item()
        self._step_count = checkpoint.get("step_count", 0)
        self._update_count = checkpoint.get("update_count", 0)
        logger.info(f"SAC model loaded from {path} (step={self._step_count}, updates={self._update_count})")
    
    def reset(self) -> None:
        """Reset episode-level counters (not weights)."""
        pass  # SAC is off-policy, no episode-level reset needed
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "algorithm": "SAC",
            "step_count": self._step_count,
            "update_count": self._update_count,
            "buffer_size": len(self.replay_buffer),
            "alpha": self.alpha,
            "metrics": dict(self.metrics),
            "device": str(self.device),
        }
