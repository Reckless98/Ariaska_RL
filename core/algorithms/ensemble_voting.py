"""core/algorithms/ensemble_voting.py — Phase 49: Ensemble PPO with Voting.

Run N independent PPO policy heads (sharing backbone), vote on action.
Reduces variance and catches edge cases that single heads miss.

Architecture:
    Shared Backbone (512→[512,512,256]) → N Actor Heads → Voting → Action
                                        → Single Critic Head → Value

Voting strategies:
  - majority: most-picked action wins
  - weighted: weighted by head confidence (softmax entropy)
  - ranked: Borda count over top-3 per head
  - boltzmann: sample from averaged logit distribution
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("ariaska.algorithms.ensemble_voting")


@dataclass
class EnsembleConfig:
    """Configuration for ensemble PPO voting.

    Args:
        enabled: Master switch.
        n_heads: Number of independent actor heads.
        state_dim: Input state dimension.
        hidden_dim: Backbone output dimension.
        action_dim: Output action dimension.
        voting: Voting strategy — majority | weighted | ranked | boltzmann.
        temperature: Temperature for Boltzmann voting.
        dropout: Dropout rate per head.
        disagreement_bonus: Intrinsic reward for head disagreement.
        head_diversity_coef: Diversity regularization coefficient.
    """
    enabled: bool = True
    n_heads: int = 5
    state_dim: int = 512
    hidden_dim: int = 256
    action_dim: int = 5
    voting: str = "weighted"
    temperature: float = 1.0
    dropout: float = 0.1
    disagreement_bonus: float = 0.1
    head_diversity_coef: float = 0.01


class ActorHead(nn.Module):
    """Single actor head for the ensemble."""

    def __init__(self, input_dim: int, action_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, action_dim),
        )
        # Small init for diversity
        last_layer = self.net[-1]
        assert isinstance(last_layer, nn.Linear)
        nn.init.orthogonal_(last_layer.weight, gain=0.01)
        nn.init.zeros_(last_layer.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Returns raw logits (B, action_dim)."""
        return self.net(features)


class EnsembleVoter(nn.Module):
    """Ensemble of N actor heads with voting for action selection.

    Usage::

        voter = EnsembleVoter(config)
        action, info = voter.vote(backbone_features)
        # info contains {head_logits, agreement, entropy, chosen_by}
    """

    def __init__(self, config: Optional[EnsembleConfig] = None) -> None:
        super().__init__()
        self.config = config or EnsembleConfig()

        self.heads = nn.ModuleList([
            ActorHead(
                self.config.hidden_dim,
                self.config.action_dim,
                self.config.dropout,
            )
            for _ in range(self.config.n_heads)
        ])

        self._vote_counts: Dict[str, int] = {}
        self._total_votes: int = 0

    def forward(self, features: torch.Tensor) -> List[torch.Tensor]:
        """Get logits from all heads.

        Args:
            features: (B, hidden_dim) backbone features.

        Returns:
            List of N tensors, each (B, action_dim).
        """
        return [head(features) for head in self.heads]

    def vote(
        self,
        features: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Select action via ensemble voting.

        Args:
            features: (B, hidden_dim) backbone features.
            deterministic: If True, always pick argmax.

        Returns:
            (actions, info) — actions: (B,) long tensor, info: metadata dict.
        """
        all_logits = self.forward(features)  # N × (B, A)
        B = features.shape[0]
        A = self.config.action_dim

        strategy = self.config.voting

        if strategy == "majority":
            actions, info = self._majority_vote(all_logits, deterministic)
        elif strategy == "weighted":
            actions, info = self._weighted_vote(all_logits, deterministic)
        elif strategy == "ranked":
            actions, info = self._ranked_vote(all_logits, deterministic)
        elif strategy == "boltzmann":
            actions, info = self._boltzmann_vote(all_logits, deterministic)
        else:
            actions, info = self._weighted_vote(all_logits, deterministic)

        # Compute agreement and disagreement bonus
        head_actions = torch.stack([logits.argmax(dim=-1) for logits in all_logits])  # (N, B)
        mode_action = actions.unsqueeze(0).expand_as(head_actions)
        agreement = (head_actions == mode_action).float().mean(dim=0)  # (B,)

        info["agreement"] = agreement.mean().item()
        info["disagreement_bonus"] = (1.0 - agreement.mean().item()) * self.config.disagreement_bonus
        info["head_logits"] = all_logits

        self._total_votes += 1
        return actions, info

    def _majority_vote(
        self, all_logits: List[torch.Tensor], deterministic: bool,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Each head picks its best action, majority wins."""
        B = all_logits[0].shape[0]
        A = self.config.action_dim
        head_choices = torch.stack([l.argmax(dim=-1) for l in all_logits])  # (N, B)

        # Count votes per action
        votes = torch.zeros(B, A, device=all_logits[0].device)
        for h in range(len(all_logits)):
            one_hot = F.one_hot(head_choices[h], A).float()
            votes += one_hot

        actions = votes.argmax(dim=-1)  # (B,)
        return actions, {"strategy": "majority"}

    def _weighted_vote(
        self, all_logits: List[torch.Tensor], deterministic: bool,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Weight each head's vote by its confidence (inverse entropy)."""
        B = all_logits[0].shape[0]
        A = self.config.action_dim

        weighted_logits = torch.zeros(B, A, device=all_logits[0].device)
        for logits in all_logits:
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1, keepdim=True)
            max_entropy = math.log(A)
            confidence = 1.0 - entropy / max_entropy  # (B, 1) in [0, 1]
            weighted_logits += logits * confidence

        if deterministic:
            actions = weighted_logits.argmax(dim=-1)
        else:
            probs = F.softmax(weighted_logits / self.config.temperature, dim=-1)
            actions = torch.multinomial(probs, 1).squeeze(-1)

        return actions, {"strategy": "weighted"}

    def _ranked_vote(
        self, all_logits: List[torch.Tensor], deterministic: bool,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Borda count: each head ranks actions, points by rank position."""
        B = all_logits[0].shape[0]
        A = self.config.action_dim

        scores = torch.zeros(B, A, device=all_logits[0].device)
        for logits in all_logits:
            ranks = logits.argsort(dim=-1, descending=True)  # (B, A)
            for rank_pos in range(A):
                action_at_rank = ranks[:, rank_pos]
                scores.scatter_add_(1, action_at_rank.unsqueeze(1),
                                    torch.full((B, 1), float(A - rank_pos),
                                               device=logits.device))

        actions = scores.argmax(dim=-1)
        return actions, {"strategy": "ranked"}

    def _boltzmann_vote(
        self, all_logits: List[torch.Tensor], deterministic: bool,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Average logits across heads, sample from Boltzmann distribution."""
        avg_logits = torch.stack(all_logits).mean(dim=0)  # (B, A)

        if deterministic:
            actions = avg_logits.argmax(dim=-1)
        else:
            probs = F.softmax(avg_logits / self.config.temperature, dim=-1)
            actions = torch.multinomial(probs, 1).squeeze(-1)

        return actions, {"strategy": "boltzmann"}

    def compute_diversity_loss(self, features: torch.Tensor) -> torch.Tensor:
        """Regularization loss to encourage head diversity.

        Penalizes heads that produce similar probability distributions.
        Uses pairwise KL divergence between head outputs.
        """
        all_logits = self.forward(features)
        if len(all_logits) < 2:
            return torch.tensor(0.0, device=features.device)

        all_probs = [F.softmax(l, dim=-1) for l in all_logits]
        all_log_probs = [F.log_softmax(l, dim=-1) for l in all_logits]

        diversity_loss = torch.tensor(0.0, device=features.device)
        n_pairs = 0
        for i in range(len(all_probs)):
            for j in range(i + 1, len(all_probs)):
                # KL(p_i || p_j)
                kl = (all_probs[i] * (all_log_probs[i] - all_log_probs[j])).sum(dim=-1)
                diversity_loss -= kl.mean()  # Negative: we WANT high KL (diversity)
                n_pairs += 1

        return diversity_loss * self.config.head_diversity_coef / max(n_pairs, 1)

    def get_stats(self) -> Dict[str, Any]:
        """Return ensemble statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        return {
            "n_heads": self.config.n_heads,
            "voting_strategy": self.config.voting,
            "total_params": total_params,
            "total_votes": self._total_votes,
        }
