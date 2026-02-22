"""core/algorithms/world_model.py — Phase 49: World Model (DreamerV3-inspired).

Learned environment dynamics model for model-based RL planning.
Predicts next state, reward, and done from current state + action,
enabling N-step look-ahead without real execution.

Architecture (simplified RSSM):
  - Deterministic path: GRU over (state, action) sequences.
  - Stochastic path: posterior from observation, prior from GRU hidden.
  - Reward predictor: hidden → scalar reward.
  - Continue predictor: hidden → P(not done).
  - Decoder: hidden → reconstructed observation (optional).

Integration:
  - SmartCoach can query `imagine(state, action_seq)` for look-ahead.
  - PPO can use imagined rollouts for additional training data.
  - Compatible with STATE_DIM=512 and ACTION_DIM=5.

Reference: Hafner et al. "Mastering Diverse Domains through World Models" 2023.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("ariaska.algorithms.world_model")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class WorldModelConfig:
    """Configuration for the world model.

    Args:
        state_dim: Observation / state dimension.
        action_dim: Number of discrete actions.
        hidden_dim: GRU hidden and latent representation dim.
        stoch_dim: Stochastic latent dimension.
        lr: Learning rate.
        imagination_horizon: N-step look-ahead depth.
        kl_balance: Balance between prior and posterior KL (0.5=equal).
        kl_free: Free nats for KL (below this, no gradient).
        symlog: Use symlog encoding for targets (DreamerV3).
        device: Torch device.
    """
    state_dim: int = 512
    action_dim: int = 5
    hidden_dim: int = 256
    stoch_dim: int = 32
    lr: float = 3e-4
    imagination_horizon: int = 5
    kl_balance: float = 0.8
    kl_free: float = 1.0
    symlog: bool = True
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric logarithmic compression (DreamerV3)."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of symlog."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


# ---------------------------------------------------------------------------
# RSSM components
# ---------------------------------------------------------------------------


class GRUCell(nn.Module):
    """Deterministic state transition via GRU."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRUCell(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, input_dim) input features.
            h: (B, hidden_dim) previous hidden state.

        Returns:
            (B, hidden_dim) next hidden state.
        """
        return self.norm(self.gru(x, h))


class StochasticSSM(nn.Module):
    """Stochastic state space model (prior + posterior)."""

    def __init__(self, hidden_dim: int, stoch_dim: int, obs_dim: int) -> None:
        super().__init__()
        # Prior: predict stochastic from deterministic hidden
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, stoch_dim * 2),  # mean + logvar
        )
        # Posterior: refine with actual observation
        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_dim + obs_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, stoch_dim * 2),
        )

    def prior(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute prior distribution parameters.

        Returns:
            (mean, logvar) each (B, stoch_dim).
        """
        out = self.prior_net(h)
        mean, logvar = out.chunk(2, dim=-1)
        return mean, logvar

    def posterior(
        self,
        h: torch.Tensor,
        obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute posterior distribution parameters.

        Returns:
            (mean, logvar) each (B, stoch_dim).
        """
        out = self.posterior_net(torch.cat([h, obs], dim=-1))
        mean, logvar = out.chunk(2, dim=-1)
        return mean, logvar

    @staticmethod
    def sample(
        mean: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Reparameterised sample from N(mean, exp(logvar))."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std


# ---------------------------------------------------------------------------
# Predictor heads
# ---------------------------------------------------------------------------


class RewardPredictor(nn.Module):
    """Predict scalar reward from latent state."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict reward.

        Args:
            x: (B, input_dim) latent state.

        Returns:
            (B,) predicted reward.
        """
        return self.net(x).squeeze(-1)


class ContinuePredictor(nn.Module):
    """Predict continuation probability P(not done)."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict P(continue).

        Args:
            x: (B, input_dim) latent state.

        Returns:
            (B,) continuation logits.
        """
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# World Model
# ---------------------------------------------------------------------------


class WorldModel(nn.Module):
    """DreamerV3-inspired world model for Ariaska.

    Learns environment dynamics in latent space and enables
    imagination-based planning (N-step look-ahead).
    """

    def __init__(self, config: Optional[WorldModelConfig] = None) -> None:
        super().__init__()
        self.config = config or WorldModelConfig()
        c = self.config

        latent_dim = c.hidden_dim + c.stoch_dim

        # Action embedding (discrete → dense)
        self.action_embed = nn.Embedding(c.action_dim, c.hidden_dim)

        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(c.state_dim, c.hidden_dim),
            nn.ELU(),
            nn.Linear(c.hidden_dim, c.hidden_dim),
        )

        # Deterministic transition
        self.gru = GRUCell(c.hidden_dim + c.stoch_dim, c.hidden_dim)

        # Stochastic SSM
        self.ssm = StochasticSSM(c.hidden_dim, c.stoch_dim, c.hidden_dim)

        # Predictor heads
        self.reward_pred = RewardPredictor(latent_dim)
        self.continue_pred = ContinuePredictor(latent_dim)

        # Observation decoder (optional reconstruction loss)
        self.obs_decoder = nn.Sequential(
            nn.Linear(latent_dim, c.hidden_dim),
            nn.ELU(),
            nn.Linear(c.hidden_dim, c.state_dim),
        )

        # Optimiser
        self.optimizer = torch.optim.Adam(self.parameters(), lr=c.lr)

        self._step_count: int = 0
        logger.info(
            f"WorldModel initialised: state={c.state_dim}, "
            f"hidden={c.hidden_dim}, stoch={c.stoch_dim}"
        )

    def _get_latent(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate deterministic and stochastic state."""
        return torch.cat([h, z], dim=-1)

    def observe(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Process a sequence of observation-action pairs.

        Args:
            observations: (B, T, state_dim) observation sequence.
            actions: (B, T) action sequence (long).

        Returns:
            Dict with prior/posterior stats, hidden states, latent states.
        """
        B, T = observations.shape[:2]
        c = self.config
        device = observations.device

        h = torch.zeros(B, c.hidden_dim, device=device)
        z = torch.zeros(B, c.stoch_dim, device=device)

        priors: List[Tuple[torch.Tensor, torch.Tensor]] = []
        posteriors: List[Tuple[torch.Tensor, torch.Tensor]] = []
        hiddens: List[torch.Tensor] = []
        latents: List[torch.Tensor] = []

        for t in range(T):
            obs_t = observations[:, t]
            act_t = actions[:, t]

            # Encode action
            act_emb = self.action_embed(act_t)  # (B, hidden_dim)

            # Deterministic transition
            x = torch.cat([act_emb, z], dim=-1)
            h = self.gru(x, h)

            # Stochastic
            prior_mean, prior_logvar = self.ssm.prior(h)
            obs_enc = self.obs_encoder(obs_t)
            post_mean, post_logvar = self.ssm.posterior(h, obs_enc)
            z = self.ssm.sample(post_mean, post_logvar)

            priors.append((prior_mean, prior_logvar))
            posteriors.append((post_mean, post_logvar))
            hiddens.append(h)
            latents.append(self._get_latent(h, z))

        return {
            "hiddens": torch.stack(hiddens, dim=1),
            "latents": torch.stack(latents, dim=1),
            "priors": priors,
            "posteriors": posteriors,
        }

    def imagine(
        self,
        initial_state: torch.Tensor,
        action_sequence: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Imagine future states without real observations (planning).

        Args:
            initial_state: (state_dim,) or (B, state_dim) current state.
            action_sequence: (H,) or (B, H) planned action sequence.

        Returns:
            Dict with predicted rewards, continues, and latent states.
        """
        if initial_state.dim() == 1:
            initial_state = initial_state.unsqueeze(0)
        if action_sequence.dim() == 1:
            action_sequence = action_sequence.unsqueeze(0)

        B, H = action_sequence.shape
        c = self.config
        device = initial_state.device

        # Initialise from observation
        obs_enc = self.obs_encoder(initial_state)
        h = obs_enc  # Use encoded obs as initial hidden
        if h.shape[-1] != c.hidden_dim:
            h = h[:, :c.hidden_dim]  # Truncate if needed
        z = torch.zeros(B, c.stoch_dim, device=device)

        pred_rewards: List[torch.Tensor] = []
        pred_continues: List[torch.Tensor] = []
        pred_states: List[torch.Tensor] = []

        for t in range(H):
            act_t = action_sequence[:, t]
            act_emb = self.action_embed(act_t)

            x = torch.cat([act_emb, z], dim=-1)
            h = self.gru(x, h)

            prior_mean, prior_logvar = self.ssm.prior(h)
            z = self.ssm.sample(prior_mean, prior_logvar)

            latent = self._get_latent(h, z)
            pred_r = self.reward_pred(latent)
            pred_c = torch.sigmoid(self.continue_pred(latent))
            pred_s = self.obs_decoder(latent)

            pred_rewards.append(pred_r)
            pred_continues.append(pred_c)
            pred_states.append(pred_s)

        return {
            "rewards": torch.stack(pred_rewards, dim=1),       # (B, H)
            "continues": torch.stack(pred_continues, dim=1),   # (B, H)
            "states": torch.stack(pred_states, dim=1),         # (B, H, state_dim)
        }

    def compute_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute world model training loss.

        Args:
            observations: (B, T, state_dim).
            actions: (B, T) long.
            rewards: (B, T) float.
            dones: (B, T) float (1.0 = done).

        Returns:
            Dict with total loss and component losses.
        """
        c = self.config
        result = self.observe(observations, actions)
        latents = result["latents"]  # (B, T, latent_dim)

        B, T = latents.shape[:2]

        # Target encoding (symlog for DreamerV3)
        if c.symlog:
            reward_target = symlog(rewards)
        else:
            reward_target = rewards

        # Reward loss
        pred_rewards = self.reward_pred(latents.reshape(B * T, -1))
        reward_loss = F.mse_loss(pred_rewards, reward_target.reshape(B * T))

        # Continue loss (binary cross-entropy)
        pred_continues = self.continue_pred(latents.reshape(B * T, -1))
        continue_loss = F.binary_cross_entropy_with_logits(
            pred_continues,
            1.0 - dones.reshape(B * T),
        )

        # Reconstruction loss
        pred_obs = self.obs_decoder(latents.reshape(B * T, -1))
        recon_loss = F.mse_loss(pred_obs, observations.reshape(B * T, -1))

        # KL loss (balancing prior and posterior)
        kl_loss = torch.tensor(0.0, device=latents.device)
        for t in range(T):
            prior_m, prior_lv = result["priors"][t]
            post_m, post_lv = result["posteriors"][t]

            kl_forward = self._kl_divergence(post_m, post_lv, prior_m, prior_lv)
            kl_reverse = self._kl_divergence(prior_m, prior_lv, post_m, post_lv)

            kl_t = (
                c.kl_balance * torch.clamp(kl_forward - c.kl_free, min=0.0).mean()
                + (1 - c.kl_balance)
                * torch.clamp(kl_reverse - c.kl_free, min=0.0).mean()
            )
            kl_loss = kl_loss + kl_t

        kl_loss = kl_loss / max(T, 1)

        total_loss = reward_loss + continue_loss + 0.5 * recon_loss + kl_loss

        return {
            "total": total_loss,
            "reward": reward_loss,
            "continue": continue_loss,
            "recon": recon_loss,
            "kl": kl_loss,
        }

    @staticmethod
    def _kl_divergence(
        mu1: torch.Tensor,
        lv1: torch.Tensor,
        mu2: torch.Tensor,
        lv2: torch.Tensor,
    ) -> torch.Tensor:
        """KL(N(mu1,sigma1) || N(mu2,sigma2)) per element."""
        var1 = torch.exp(lv1)
        var2 = torch.exp(lv2)
        kl = 0.5 * (lv2 - lv1 + var1 / var2 + (mu1 - mu2) ** 2 / var2 - 1)
        return kl.sum(dim=-1)

    def train_step(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> Dict[str, float]:
        """Single training step.

        Args:
            observations: (B, T, state_dim).
            actions: (B, T) long.
            rewards: (B, T) float.
            dones: (B, T) float.

        Returns:
            Dict of loss values (float).
        """
        self.train()
        losses = self.compute_loss(observations, actions, rewards, dones)

        self.optimizer.zero_grad()
        losses["total"].backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=100.0)
        self.optimizer.step()

        self._step_count += 1
        return {k: v.item() for k, v in losses.items()}

    def evaluate_action_sequence(
        self,
        state: torch.Tensor,
        candidates: List[List[int]],
    ) -> List[float]:
        """Score multiple candidate action sequences via imagination.

        Args:
            state: (state_dim,) current state.
            candidates: List of action sequences (each a list of ints).

        Returns:
            List of cumulative predicted rewards for each candidate.
        """
        self.eval()
        scores: List[float] = []

        with torch.no_grad():
            for actions in candidates:
                act_tensor = torch.tensor(
                    actions, dtype=torch.long, device=state.device
                )
                result = self.imagine(state, act_tensor)
                cum_reward = result["rewards"].sum().item()
                scores.append(cum_reward)

        return scores

    def get_stats(self) -> Dict[str, Any]:
        """Return model statistics."""
        return {
            "train_steps": self._step_count,
            "hidden_dim": self.config.hidden_dim,
            "stoch_dim": self.config.stoch_dim,
            "imagination_horizon": self.config.imagination_horizon,
            "n_params": sum(p.numel() for p in self.parameters()),
        }
