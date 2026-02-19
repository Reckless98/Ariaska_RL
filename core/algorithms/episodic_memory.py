"""core/algorithms/episodic_memory.py — Phase 41: Episodic Memory with k-NN retrieval.

Stores compressed state representations from past episodes and retrieves
similar experiences via cosine similarity for retrieval-augmented RL.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ariaska.algorithms.episodic_memory")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


@dataclass
class EpisodicMemoryConfig:
    """Configuration for episodic memory."""
    enabled: bool = True
    capacity: int = 5000
    embedding_dim: int = 128
    state_dim: int = 512
    k_neighbors: int = 5


@dataclass
class MemoryEntry:
    """A single episodic memory entry."""
    compressed_state: Any  # torch.Tensor (embedding_dim,)
    action: int = 0
    reward: float = 0.0
    phase: str = "RECON"
    outcome: str = "neutral"


class EpisodicMemory:
    """k-NN episodic memory for retrieval-augmented RL.

    Stores compressed state representations and retrieves the *k* most
    similar past experiences via cosine similarity.
    """

    def __init__(self, config: Optional[EpisodicMemoryConfig] = None) -> None:
        self.config = config or EpisodicMemoryConfig()
        self._entries: List[MemoryEntry] = []

        if not _HAS_TORCH:
            self._encoder: Optional[Any] = None
            logger.warning("torch not available, episodic memory disabled")
            return

        self._encoder = nn.Linear(self.config.state_dim, self.config.embedding_dim)
        self._encoder.eval()

    @property
    def size(self) -> int:
        """Number of entries in memory."""
        return len(self._entries)

    def store(
        self,
        state_tensor: Any,
        action_idx: int,
        reward: float,
        phase: str = "RECON",
        outcome: str = "neutral",
    ) -> None:
        """Store a state in episodic memory.

        Args:
            state_tensor: Raw state tensor (state_dim,).
            action_idx: Action taken.
            reward: Reward received.
            phase: Attack phase at time of storage.
            outcome: Outcome label (success/fail/neutral).
        """
        if not _HAS_TORCH or self._encoder is None:
            return

        with torch.no_grad():
            if not isinstance(state_tensor, torch.Tensor):
                state_tensor = torch.tensor(state_tensor, dtype=torch.float32)
            if state_tensor.dim() == 0:
                return
            if state_tensor.dim() > 1:
                state_tensor = state_tensor.flatten()[:self.config.state_dim]
            compressed = self._encoder(state_tensor)
            norm = compressed.norm()
            if norm > 0:
                compressed = compressed / norm

        entry = MemoryEntry(
            compressed_state=compressed.detach().clone(),
            action=action_idx,
            reward=reward,
            phase=phase,
            outcome=outcome,
        )

        if len(self._entries) >= self.config.capacity:
            self._entries.pop(0)
        self._entries.append(entry)

    def retrieve(
        self, query_state: Any, k: Optional[int] = None
    ) -> List[MemoryEntry]:
        """Retrieve k most similar past experiences.

        Args:
            query_state: Query state tensor (state_dim,).
            k: Number of neighbors (defaults to config.k_neighbors).

        Returns:
            List of most similar MemoryEntry objects.
        """
        if not _HAS_TORCH or self._encoder is None or not self._entries:
            return []

        k = k or self.config.k_neighbors
        k = min(k, len(self._entries))

        with torch.no_grad():
            if not isinstance(query_state, torch.Tensor):
                query_state = torch.tensor(query_state, dtype=torch.float32)
            if query_state.dim() > 1:
                query_state = query_state.flatten()[:self.config.state_dim]
            query_compressed = self._encoder(query_state)
            q_norm = query_compressed.norm()
            if q_norm > 0:
                query_compressed = query_compressed / q_norm

            stored = torch.stack([e.compressed_state for e in self._entries])
            similarities = F.cosine_similarity(
                query_compressed.unsqueeze(0), stored, dim=1
            )
            topk = similarities.topk(k)

        return [self._entries[i] for i in topk.indices.tolist()]

    def format_for_injection(self, entries: List[MemoryEntry]) -> Optional[Any]:
        """Stack retrieved memory entries into a tensor for injection.

        Args:
            entries: Retrieved memory entries.

        Returns:
            Tensor of shape (len(entries), embedding_dim) or None.
        """
        if not _HAS_TORCH or not entries:
            return None
        return torch.stack([e.compressed_state for e in entries])

    def clear(self) -> None:
        """Clear all stored memories."""
        self._entries.clear()
