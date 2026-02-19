"""
core/algorithms/nstep_returns.py — Phase 41: N-Step Return Computation

Provides n-step return targets that can be blended with GAE-lambda
returns in the PPO update.  Longer n-step horizons reduce bias at the
cost of higher variance; this module lets PPO mix both signals.

Usage:
    from core.algorithms.nstep_returns import NStepConfig, compute_nstep_returns
    cfg = NStepConfig(n=5, gamma=0.99, blend_alpha=0.3)
    nstep = compute_nstep_returns(rewards, values, dones, cfg)
    # Blend: targets = (1 - alpha) * gae_targets + alpha * nstep
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger("ariaska.nstep_returns")

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False


def _env_bool(key: str, default: bool = False) -> bool:
    import os
    return os.environ.get(key, str(default)).lower() in ("1", "true", "yes")


@dataclass
class NStepConfig:
    """Configuration for n-step return computation."""
    n: int = 5
    gamma: float = 0.99
    blend_alpha: float = 0.3  # weight for n-step vs GAE blend
    enabled: bool = field(default_factory=lambda: _env_bool("FF_NSTEP", True))


def compute_nstep_returns(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    config: Optional[NStepConfig] = None,
) -> List[float]:
    """
    Compute n-step return targets from a trajectory.

    For each timestep t:
        G_t^n = r_t + gamma*r_{t+1} + ... + gamma^{n-1}*r_{t+n-1}
              + gamma^n * V(s_{t+n})

    If a done signal is encountered before n steps, the return is
    truncated at the terminal state.

    Args:
        rewards: Per-step rewards [T].
        values: Value estimates [T] (can include bootstrap at end).
        dones: Episode termination flags [T].
        config: NStepConfig instance.

    Returns:
        List of n-step return targets [T].
    """
    cfg = config or NStepConfig()
    T = len(rewards)
    if T == 0:
        return []

    nstep_returns: List[float] = [0.0] * T
    n = cfg.n
    gamma = cfg.gamma

    for t in range(T):
        g = 0.0
        discount = 1.0
        steps_used = 0
        for k in range(n):
            idx = t + k
            if idx >= T:
                break
            g += discount * rewards[idx]
            discount *= gamma
            steps_used = k + 1
            if dones[idx]:
                break  # Terminal — no bootstrap
        else:
            # No terminal encountered, bootstrap with V(s_{t+n})
            bootstrap_idx = t + n
            if bootstrap_idx < len(values) and not (
                steps_used > 0 and t + steps_used - 1 < T and dones[t + steps_used - 1]
            ):
                g += discount * values[bootstrap_idx]

        # If terminal hit before n steps, check if we should still bootstrap
        if steps_used > 0 and steps_used < n:
            last_idx = t + steps_used - 1
            if last_idx < T and not dones[last_idx]:
                bootstrap_idx = t + steps_used
                if bootstrap_idx < len(values):
                    g += discount * values[bootstrap_idx]

        nstep_returns[t] = g

    return nstep_returns


def blend_returns(
    gae_targets: List[float],
    nstep_targets: List[float],
    alpha: float = 0.3,
) -> List[float]:
    """
    Linearly blend GAE and n-step return targets.

    result = (1 - alpha) * gae + alpha * nstep

    Args:
        gae_targets: GAE-lambda return targets.
        nstep_targets: N-step return targets.
        alpha: Blend weight for n-step (0 = pure GAE, 1 = pure n-step).

    Returns:
        Blended targets.
    """
    if len(gae_targets) != len(nstep_targets):
        raise ValueError(
            f"Length mismatch: gae={len(gae_targets)} vs nstep={len(nstep_targets)}"
        )
    return [
        (1 - alpha) * g + alpha * n
        for g, n in zip(gae_targets, nstep_targets)
    ]


if _HAS_TORCH:
    def compute_nstep_returns_tensor(
        rewards: "torch.Tensor",
        values: "torch.Tensor",
        dones: "torch.Tensor",
        config: Optional[NStepConfig] = None,
    ) -> "torch.Tensor":
        """Tensor-based n-step returns (for GPU-accelerated training)."""
        cfg = config or NStepConfig()
        r_list = rewards.detach().cpu().tolist()
        v_list = values.detach().cpu().tolist()
        d_list = [bool(d) for d in dones.detach().cpu().tolist()]
        result = compute_nstep_returns(r_list, v_list, d_list, cfg)
        return torch.tensor(result, dtype=rewards.dtype, device=rewards.device)
