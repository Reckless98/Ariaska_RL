#!/usr/bin/env python3
"""
core/memory/semantic_index.py — Phase 15.0: Semantic Index

Bounded semantic search over past commands and discoveries within an episode.
Uses simple term-frequency scoring (no external ML dependencies) with cosine
similarity over character n-gram vectors. Fast, deterministic, bounded.

Feature-flag gated: FF_SEMANTIC_INDEX.

Max entries: 256 (ring buffer).
Max query results: 8.
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ariaska.memory.semantic_index")

_MAX_ENTRIES = 256
_MAX_QUERY_RESULTS = 8
_NGRAM_SIZE = 3
_MAX_TEXT_LEN = 200


# ── Schemas ─────────────────────────────────────────────────────────────────

@dataclass
class SemanticEntry:
    """A single entry in the semantic index."""
    text: str = ""              # bounded to 200 chars
    entry_type: str = ""        # "command", "discovery", "finding"
    step: int = 0
    reward: float = 0.0
    phase: str = ""
    tags: List[str] = field(default_factory=list)
    _ngrams: Dict[str, int] = field(default_factory=dict, repr=False)
    _norm: float = 0.0

    def content_hash(self) -> str:
        return hashlib.sha256(
            self.text.encode("utf-8", errors="replace")
        ).hexdigest()[:10]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "entry_type": self.entry_type,
            "step": self.step,
            "reward": self.reward,
            "phase": self.phase,
            "tags": self.tags,
        }


@dataclass
class SemanticQueryResult:
    """Result of a semantic query."""
    entry: SemanticEntry
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = self.entry.to_dict()
        d["score"] = round(self.score, 4)
        return d


# ── Helpers ─────────────────────────────────────────────────────────────────

def _char_ngrams(text: str, n: int = _NGRAM_SIZE) -> Dict[str, int]:
    """Extract character n-gram frequency vector."""
    text = text.lower().strip()[:_MAX_TEXT_LEN]
    counts: Counter[str] = Counter()
    for i in range(max(0, len(text) - n + 1)):
        counts[text[i:i + n]] += 1
    return dict(counts)


def _vector_norm(vec: Dict[str, int]) -> float:
    """L2 norm of a sparse vector."""
    return math.sqrt(sum(v * v for v in vec.values())) if vec else 0.0


def _cosine_similarity(a: Dict[str, int], a_norm: float,
                       b: Dict[str, int], b_norm: float) -> float:
    """Cosine similarity between two sparse n-gram vectors."""
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    # Dot product over shared keys
    dot = sum(a[k] * b[k] for k in a if k in b)
    return dot / (a_norm * b_norm)


# ── Semantic Index ──────────────────────────────────────────────────────────

class SemanticIndex:
    """
    Bounded semantic search index for commands and discoveries.

    - Ring buffer of up to 256 entries
    - Character n-gram cosine similarity
    - No external ML dependencies
    - Bounded query results (max 8)
    """

    def __init__(
        self,
        max_entries: int = _MAX_ENTRIES,
        max_results: int = _MAX_QUERY_RESULTS,
    ) -> None:
        self._max_entries = max(16, min(max_entries, 1024))  # hard cap
        self._max_results = max(1, min(max_results, 16))
        self._entries: List[SemanticEntry] = []
        self._seen_hashes: set = set()
        self._write_idx = 0  # ring buffer pointer

    def add(
        self,
        text: str,
        entry_type: str = "command",
        step: int = 0,
        reward: float = 0.0,
        phase: str = "",
        tags: Optional[List[str]] = None,
    ) -> bool:
        """
        Add an entry to the index. Returns True if added (not duplicate).
        Text is bounded to 200 chars.
        """
        text = text[:_MAX_TEXT_LEN]
        if not text.strip():
            return False

        ngrams = _char_ngrams(text)
        norm = _vector_norm(ngrams)

        entry = SemanticEntry(
            text=text,
            entry_type=entry_type,
            step=step,
            reward=reward,
            phase=phase,
            tags=(tags or [])[:8],
            _ngrams=ngrams,
            _norm=norm,
        )

        # Dedup by content hash
        h = entry.content_hash()
        if h in self._seen_hashes:
            return False
        self._seen_hashes.add(h)

        if len(self._entries) < self._max_entries:
            self._entries.append(entry)
        else:
            # Ring buffer overwrite
            old = self._entries[self._write_idx]
            old_h = old.content_hash()
            self._seen_hashes.discard(old_h)
            self._entries[self._write_idx] = entry
            self._write_idx = (self._write_idx + 1) % self._max_entries

        return True

    def query(
        self,
        text: str,
        top_k: Optional[int] = None,
        min_score: float = 0.1,
        entry_type: Optional[str] = None,
    ) -> List[SemanticQueryResult]:
        """
        Query the index for similar entries.

        Args:
            text: Query text
            top_k: Max results (default: max_results)
            min_score: Minimum cosine similarity threshold
            entry_type: Optional filter by entry type

        Returns:
            List of SemanticQueryResult sorted by descending score.
        """
        if not self._entries or not text.strip():
            return []

        k = min(top_k or self._max_results, self._max_results)
        q_ngrams = _char_ngrams(text)
        q_norm = _vector_norm(q_ngrams)

        if q_norm == 0.0:
            return []

        results: List[SemanticQueryResult] = []
        for entry in self._entries:
            if entry_type and entry.entry_type != entry_type:
                continue
            score = _cosine_similarity(q_ngrams, q_norm, entry._ngrams, entry._norm)
            if score >= min_score:
                results.append(SemanticQueryResult(entry=entry, score=score))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]

    def clear(self) -> None:
        """Clear all entries."""
        self._entries.clear()
        self._seen_hashes.clear()
        self._write_idx = 0

    def __len__(self) -> int:
        return len(self._entries)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "entries": len(self._entries),
            "max_entries": self._max_entries,
            "unique_hashes": len(self._seen_hashes),
        }
