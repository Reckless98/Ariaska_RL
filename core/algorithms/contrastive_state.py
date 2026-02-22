"""core/algorithms/contrastive_state.py — Phase 41+49: Contrastive state representation.

Phase 41: NT-Xent contrastive loss that groups same-phase states as positives
and different-phase states as negatives, producing better representations.

Phase 49 enhancements:
  - StateAugmentor: SimCLR-style noise/dropout/masking augmentation
  - Trajectory contrast: success vs failure trajectory separation
  - Temporal consistency: nearby states should have similar representations
  - Hard negative mining: focus on cross-phase states with similar embeddings
  - Momentum encoder: EMA target encoder for stable contrastive targets
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger("ariaska.algorithms.contrastive_state")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


@dataclass
class ContrastiveConfig:
    """Configuration for contrastive learning.

    Args:
        enabled: Master switch.
        temperature: NT-Xent temperature (lower = sharper).
        projection_dim: Output dimension of projection head.
        feature_dim: Input feature dimension.
        coef: Loss coefficient when added to total loss.
        augment_noise: Gaussian noise std for augmentation.
        augment_dropout: Feature dropout rate for augmentation.
        augment_mask_frac: Fraction of features to zero-mask.
        momentum: EMA coefficient for momentum encoder.
        hard_negative_frac: Fraction of hardest negatives to emphasize.
        temporal_window: Window size for temporal consistency loss.
        trajectory_margin: Margin for success vs failure trajectory contrast.
    """
    enabled: bool = False
    temperature: float = 0.1
    projection_dim: int = 64
    feature_dim: int = 256
    coef: float = 0.05
    augment_noise: float = 0.05
    augment_dropout: float = 0.1
    augment_mask_frac: float = 0.1
    momentum: float = 0.99
    hard_negative_frac: float = 0.2
    temporal_window: int = 3
    trajectory_margin: float = 1.0


# ──────────────────────────────────────────────────────────────
# State Augmentor
# ──────────────────────────────────────────────────────────────


class StateAugmentor:
    """SimCLR-style state augmentation for contrastive pairs.

    Applies noise injection, feature dropout, and random masking
    to produce augmented views of a state tensor.
    """

    def __init__(self, config: Optional[ContrastiveConfig] = None) -> None:
        self.config = config or ContrastiveConfig()

    def augment(self, x: "torch.Tensor") -> "torch.Tensor":
        """Produce an augmented view of state tensor.

        Args:
            x: (*, feature_dim) tensor.

        Returns:
            Augmented tensor of same shape.
        """
        if not _HAS_TORCH:
            raise RuntimeError("torch required")

        aug = x.clone()

        # Gaussian noise
        if self.config.augment_noise > 0:
            noise = torch.randn_like(aug) * self.config.augment_noise
            aug = aug + noise

        # Feature dropout
        if self.config.augment_dropout > 0:
            mask = torch.bernoulli(
                torch.full_like(aug, 1.0 - self.config.augment_dropout)
            )
            aug = aug * mask

        # Random masking
        if self.config.augment_mask_frac > 0:
            flat = aug.reshape(-1)
            n_mask = max(1, int(flat.numel() * self.config.augment_mask_frac))
            indices = torch.randperm(flat.numel())[:n_mask]
            flat[indices] = 0.0
            aug = flat.reshape(aug.shape)

        return aug


# ──────────────────────────────────────────────────────────────
# Projection Head
# ──────────────────────────────────────────────────────────────


class _ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning."""

    def __init__(self, feature_dim: int, projection_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────────────────────
# Contrastive Loss (Phase 41 + Phase 49)
# ──────────────────────────────────────────────────────────────


class ContrastiveLoss(nn.Module if _HAS_TORCH else object):
    """NT-Xent contrastive loss with Phase 49 enhancements.

    Phase 41: Positive pairs = same attack phase, negative = different phase.
    Phase 49 additions:
      - Trajectory contrast (success vs failure)
      - Temporal consistency (nearby steps should be similar)
      - Hard negative mining (focus on hardest cross-phase negatives)
      - Momentum encoder (EMA target for stable targets)
      - StateAugmentor integration
      - get_representations() for downstream use
    """

    def __init__(self, config: Optional[ContrastiveConfig] = None) -> None:
        if _HAS_TORCH:
            super().__init__()
        self.config = config or ContrastiveConfig()
        self._steps = 0

        if _HAS_TORCH:
            self.projector = _ProjectionHead(
                self.config.feature_dim, self.config.projection_dim
            )
            # Momentum encoder
            self.momentum_projector = copy.deepcopy(self.projector)
            for p in self.momentum_projector.parameters():
                p.requires_grad = False

            self.augmentor = StateAugmentor(self.config)

    # ── Core NT-Xent Loss (Phase 41) ──

    def compute_loss(
        self,
        backbone_features: "torch.Tensor",
        phase_labels: "torch.Tensor",
        discovery_counts: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        """Compute NT-Xent contrastive loss.

        Args:
            backbone_features: (B, feature_dim) from PPO backbone.
            phase_labels: (B,) integer phase indices.
            discovery_counts: (B,) optional weighting.

        Returns:
            Scalar loss tensor.
        """
        if not _HAS_TORCH:
            raise RuntimeError("torch required for contrastive loss")

        batch_size = backbone_features.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=backbone_features.device)

        unique_phases = phase_labels.unique()
        if unique_phases.numel() <= 1:
            return torch.tensor(0.0, device=backbone_features.device)

        # Project
        z = self.projector(backbone_features)
        z = F.normalize(z, dim=1)

        # Similarity matrix
        sim = torch.mm(z, z.t()) / self.config.temperature

        # Mask: positive = same phase, negative = different
        labels_eq = phase_labels.unsqueeze(0) == phase_labels.unsqueeze(1)
        diag_mask = torch.eye(
            batch_size, dtype=torch.bool, device=backbone_features.device
        )
        mask_pos = labels_eq.float() * (~diag_mask).float()

        if mask_pos.sum() == 0:
            return torch.tensor(0.0, device=backbone_features.device)

        # NT-Xent
        exp_sim = torch.exp(sim) * (~diag_mask).float()

        # Hard negative mining: upweight hardest negatives
        if self.config.hard_negative_frac > 0:
            mask_neg = (~labels_eq).float() * (~diag_mask).float()
            neg_sim = exp_sim * mask_neg
            k = max(1, int(batch_size * self.config.hard_negative_frac))
            # Boost top-k negative similarities
            topk_vals, _ = neg_sim.topk(k, dim=1)
            min_topk = topk_vals[:, -1:].detach()
            hard_boost = (neg_sim >= min_topk).float() * 2.0
            hard_boost = torch.clamp(hard_boost, min=1.0)
            exp_sim = exp_sim * (mask_pos + mask_neg * hard_boost)

        pos_sum = (exp_sim * mask_pos).sum(dim=1)
        all_sum = exp_sim.sum(dim=1)

        loss = -torch.log(pos_sum / (all_sum + 1e-8) + 1e-8)

        self._steps += 1
        self._update_momentum()
        return loss.mean()

    # ── Trajectory Contrast (Phase 49) ──

    def compute_trajectory_loss(
        self,
        success_features: "torch.Tensor",
        failure_features: "torch.Tensor",
    ) -> "torch.Tensor":
        """Contrastive loss between success and failure trajectories.

        Pushes success trajectories apart from failure trajectories
        in the representation space.

        Args:
            success_features: (N, feature_dim) features from successful episodes.
            failure_features: (M, feature_dim) features from failed episodes.

        Returns:
            Scalar loss.
        """
        if not _HAS_TORCH:
            raise RuntimeError("torch required")

        if success_features.size(0) == 0 or failure_features.size(0) == 0:
            return torch.tensor(0.0)

        z_s = F.normalize(self.projector(success_features), dim=1)
        z_f = F.normalize(self.projector(failure_features), dim=1)

        # Mean embeddings
        mu_s = z_s.mean(dim=0, keepdim=True)
        mu_f = z_f.mean(dim=0, keepdim=True)

        # Triplet-style: success centroids should be far from failure centroids
        dist = F.pairwise_distance(mu_s, mu_f)
        loss = F.relu(self.config.trajectory_margin - dist)
        return loss.mean()

    # ── Temporal Consistency (Phase 49) ──

    def compute_temporal_loss(
        self,
        trajectory_features: "torch.Tensor",
    ) -> "torch.Tensor":
        """Temporal consistency: nearby steps should have similar representations.

        Args:
            trajectory_features: (T, feature_dim) time-ordered features.

        Returns:
            Scalar loss.
        """
        if not _HAS_TORCH:
            raise RuntimeError("torch required")

        T = trajectory_features.size(0)
        w = self.config.temporal_window
        if T < 2:
            return torch.tensor(0.0, device=trajectory_features.device)

        z = F.normalize(self.projector(trajectory_features), dim=1)

        # Compute similarity between each step and its window neighbors
        losses = []
        for t in range(T):
            start = max(0, t - w)
            end = min(T, t + w + 1)
            neighbors = list(range(start, end))
            neighbors = [n for n in neighbors if n != t]
            if not neighbors:
                continue

            anchor = z[t : t + 1]  # (1, D)
            pos = z[neighbors]  # (K, D)

            # Positive similarity should be high
            sim = torch.mm(anchor, pos.t()).squeeze(0)  # (K,)
            loss = -sim.mean()
            losses.append(loss)

        if not losses:
            return torch.tensor(0.0, device=trajectory_features.device)

        return torch.stack(losses).mean()

    # ── Representations ──

    def get_representations(
        self, features: "torch.Tensor"
    ) -> "torch.Tensor":
        """Get L2-normalized projected representations.

        Args:
            features: (B, feature_dim) raw features.

        Returns:
            (B, projection_dim) normalized embeddings.
        """
        if not _HAS_TORCH:
            raise RuntimeError("torch required")
        z = self.projector(features)
        return F.normalize(z, dim=1)

    # ── Momentum Encoder ──

    def _update_momentum(self) -> None:
        """EMA update of momentum encoder."""
        m = self.config.momentum
        for p_online, p_target in zip(
            self.projector.parameters(),
            self.momentum_projector.parameters(),
        ):
            p_target.data.copy_(
                m * p_target.data + (1.0 - m) * p_online.data
            )

    # ── Stats ──

    def get_stats(self) -> Dict[str, Any]:
        """Return contrastive learning statistics."""
        return {
            "steps": self._steps,
            "temperature": self.config.temperature,
            "projection_dim": self.config.projection_dim,
            "feature_dim": self.config.feature_dim,
            "momentum": self.config.momentum,
            "hard_negative_frac": self.config.hard_negative_frac,
        }
