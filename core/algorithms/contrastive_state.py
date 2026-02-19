"""core/algorithms/contrastive_state.py — Phase 41: Contrastive state representation.

NT-Xent contrastive loss that groups same-phase states as positives
and different-phase states as negatives, producing better representations.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

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
    """Configuration for contrastive learning."""
    enabled: bool = False
    temperature: float = 0.1
    projection_dim: int = 64
    feature_dim: int = 256
    coef: float = 0.05


class ContrastiveLoss(nn.Module if _HAS_TORCH else object):
    """NT-Xent contrastive loss on phase-grouped state representations.

    Positive pairs: states from the same attack phase.
    Negative pairs: states from different phases.
    """

    def __init__(self, config: Optional[ContrastiveConfig] = None) -> None:
        if _HAS_TORCH:
            super().__init__()
        self.config = config or ContrastiveConfig()
        if _HAS_TORCH:
            self.projector = nn.Sequential(
                nn.Linear(self.config.feature_dim, 128),
                nn.ReLU(),
                nn.Linear(128, self.config.projection_dim),
            )

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

        # Check if all same phase
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
        diag_mask = torch.eye(batch_size, dtype=torch.bool, device=backbone_features.device)
        mask_pos = labels_eq.float() * (~diag_mask).float()  # exclude self

        # If no positive pairs at all, return zero
        if mask_pos.sum() == 0:
            return torch.tensor(0.0, device=backbone_features.device)

        # NT-Xent: for each anchor, loss = -log(sum_pos / sum_all)
        exp_sim = torch.exp(sim) * (~diag_mask).float()  # zero diagonal without inplace

        pos_sum = (exp_sim * mask_pos).sum(dim=1)
        all_sum = exp_sim.sum(dim=1)

        # Guard log(0) 
        loss = -torch.log(pos_sum / (all_sum + 1e-8) + 1e-8)
        return loss.mean()
