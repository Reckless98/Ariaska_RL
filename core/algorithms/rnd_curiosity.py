"""
Random Network Distillation (RND) curiosity module for intrinsic reward.

R66: Provides exploration pressure via prediction-error-based intrinsic reward.
- Fixed random target network (never trained)
- Trainable predictor network (learns to match target)
- Intrinsic reward = MSE between predictor and target outputs
- Higher on novel states, lower on familiar ones
- Weight multiplier per target (MS3 > MS2) since MS3 has fewer services
- Decays near CLOSEOUT to avoid interfering with completion incentives
- Capped to prevent overwhelming extrinsic rewards

Usage:
    from core.algorithms.rnd_curiosity import RNDCuriosity
    rnd = RNDCuriosity(state_dim=512)
    intrinsic_reward = rnd.compute_intrinsic_reward(state_tensor)
    rnd.update(state_tensor)  # Train predictor
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger("ariaska.algorithms.rnd")


class RNDNetwork(nn.Module):
    """Simple MLP used for both target and predictor networks."""

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RNDCuriosity:
    """Lightweight RND curiosity module.
    
    Attributes:
        target_net:    Fixed random network (never trained).
        predictor_net: Trainable network that learns to match target.
        intrinsic_reward = ||predictor(s) - target(s)||^2
        
    Design choices:
        - Small networks (512→256→128) — minimal compute overhead
        - Running normalization of intrinsic rewards for stability
        - Phase-aware decay: reward diminishes as agent approaches CLOSEOUT
        - Target-aware scaling: MS3 gets 1.5× weight (fewer services, needs more exploration)
    """

    # Phase ordinal for decay computation
    PHASE_ORDINALS = {
        "RECON": 0, "ENUMERATION": 1, "EXPLOITATION": 2,
        "PRIVILEGE_ESCALATION": 3, "LATERAL_MOVEMENT": 4,
        "POST_EXPLOITATION": 5, "EXFILTRATION": 6, "CLOSEOUT": 7,
    }

    def __init__(
        self,
        state_dim: int = 512,
        hidden_dim: int = 256,
        output_dim: int = 128,
        learning_rate: float = 1e-3,
        reward_scale: float = 1.0,
        reward_cap: float = 5.0,
        ms3_multiplier: float = 1.5,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cpu")
        self.reward_scale = reward_scale
        self.reward_cap = reward_cap
        self.ms3_multiplier = ms3_multiplier
        self._is_ms3 = False

        # Fixed target network — random initialization, never updated
        self.target_net = RNDNetwork(state_dim, hidden_dim, output_dim).to(self.device)
        for p in self.target_net.parameters():
            p.requires_grad = False

        # Trainable predictor network
        self.predictor_net = RNDNetwork(state_dim, hidden_dim, output_dim).to(self.device)
        self.optimizer = optim.Adam(self.predictor_net.parameters(), lr=learning_rate)

        # Running stats for reward normalization
        self._reward_mean = 0.0
        self._reward_var = 1.0
        self._reward_count = 0
        self._update_count = 0

        logger.info(
            f"RND initialized: state_dim={state_dim}, hidden={hidden_dim}, "
            f"output={output_dim}, scale={reward_scale}, cap={reward_cap}"
        )

    def set_target_mode(self, env_name: str) -> None:
        """Set target-aware scaling. MS3 gets higher intrinsic reward."""
        self._is_ms3 = "ms3" in env_name.lower() or "172.28.0.11" in env_name
        logger.debug(f"RND target mode: {'MS3' if self._is_ms3 else 'MS2'}")

    def compute_intrinsic_reward(
        self,
        state: torch.Tensor,
        phase: str = "RECON",
    ) -> float:
        """Compute intrinsic curiosity reward for a state.
        
        Args:
            state: (state_dim,) state tensor
            phase: Current attack phase name (for decay near CLOSEOUT)
            
        Returns:
            Intrinsic reward (float), capped and phase-decayed.
        """
        if state is None:
            return 0.0

        with torch.no_grad():
            s = state.unsqueeze(0).to(self.device) if state.dim() == 1 else state.to(self.device)
            target_out = self.target_net(s)
            predictor_out = self.predictor_net(s)
            mse = ((target_out - predictor_out) ** 2).mean().item()

        # Normalize using running stats
        self._update_running_stats(mse)
        normalized = (mse - self._reward_mean) / max(self._reward_var ** 0.5, 1e-6)
        raw_reward = max(0.0, normalized) * self.reward_scale

        # Phase decay: reduce near CLOSEOUT to not interfere with completion
        phase_ord = self.PHASE_ORDINALS.get(phase, 0)
        if phase_ord >= 5:  # POST_EXPLOITATION or later
            decay = max(0.1, 1.0 - (phase_ord - 4) * 0.3)  # 0.7, 0.4, 0.1
            raw_reward *= decay

        # Target-aware scaling
        if self._is_ms3:
            raw_reward *= self.ms3_multiplier

        # Cap
        return min(raw_reward, self.reward_cap)

    def update(self, state: torch.Tensor) -> float:
        """Train predictor to match target on observed state.
        
        Returns:
            Prediction loss (float).
        """
        if state is None:
            return 0.0

        s = state.unsqueeze(0).to(self.device) if state.dim() == 1 else state.to(self.device)
        target_out = self.target_net(s).detach()
        predictor_out = self.predictor_net(s)
        loss = ((target_out - predictor_out) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.predictor_net.parameters(), 1.0)
        self.optimizer.step()

        self._update_count += 1
        return loss.item()

    def _update_running_stats(self, x: float) -> None:
        """Welford's online algorithm for running mean/variance."""
        self._reward_count += 1
        delta = x - self._reward_mean
        self._reward_mean += delta / self._reward_count
        delta2 = x - self._reward_mean
        self._reward_var += (delta * delta2 - self._reward_var) / self._reward_count

    def reset_episode(self) -> None:
        """Reset per-episode tracking (running stats persist across episodes)."""
        pass  # Running stats intentionally persist

    def get_stats(self) -> dict:
        return {
            "rnd_updates": self._update_count,
            "rnd_reward_mean": round(self._reward_mean, 4),
            "rnd_reward_std": round(self._reward_var ** 0.5, 4),
            "rnd_is_ms3": self._is_ms3,
        }
