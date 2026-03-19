#!/usr/bin/env python3
"""Ariaska RAG Retriever — Phase-aware retrieval-augmented reasoning for Qwen.

Indexes real traces, postmortems, and high-quality memories into a vector store.
At runtime, retrieves the most relevant prior experience for context injection
into Qwen prompts, enabling retrieval-aware reasoning.

Uses FAISS for vector search + sentence-transformers for embeddings.
Falls back to keyword matching if FAISS is unavailable.

Architecture:
  index_all() → builds FAISS index from traces/postmortems/memories
  retrieve() → returns top-k relevant chunks for a query
  build_rag_context() → formats retrieved chunks for prompt injection
"""

from __future__ import annotations

import json
import hashlib
import logging
import os
import pickle
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("ariaska.rag")

# ── Paths ──────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRACE_DIR = _PROJECT_ROOT / "traces"
POSTMORTEM_DIR = _PROJECT_ROOT / "postmortems"
INDEX_DIR = _PROJECT_ROOT / "ariaska_ai" / "memory_index"
DATASET_V2_DIR = _PROJECT_ROOT / "ariaska_ai" / "dataset" / "v2"
DATASET_V3_DIR = _PROJECT_ROOT / "ariaska_ai" / "dataset" / "v3"

# ── Config ─────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = os.getenv("ARIASKA_EMBED_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = 512  # characters per chunk
CHUNK_OVERLAP = 64
MAX_CHUNKS = 50000  # safety cap
TOP_K = 5
MAX_CONTEXT_CHARS = 1500  # max RAG context injected into prompt


@dataclass
class MemoryChunk:
    """A single chunk of memory for retrieval."""
    text: str
    metadata: dict = field(default_factory=dict)
    chunk_id: str = ""
    
    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = hashlib.md5(self.text.encode()).hexdigest()[:12]


@dataclass
class RetrievalResult:
    """Result of a retrieval query."""
    chunks: list[MemoryChunk]
    scores: list[float]
    query: str
    latency_ms: float = 0.0
    
    def to_context_string(self, max_chars: int = MAX_CONTEXT_CHARS) -> str:
        """Format retrieved chunks as compact context for prompt injection."""
        if not self.chunks:
            return ""
        parts = []
        total = 0
        for chunk, score in zip(self.chunks, self.scores):
            entry = f"[{chunk.metadata.get('source', 'memory')}|{chunk.metadata.get('phase', '?')}|score:{score:.2f}] {chunk.text}"
            if total + len(entry) > max_chars:
                break
            parts.append(entry)
            total += len(entry)
        return "\n".join(parts)


class AriaskaRetriever:
    """FAISS-based retriever for Ariaska memory and traces.
    
    Falls back to keyword matching if FAISS/sentence-transformers unavailable.
    """
    
    def __init__(self, index_dir: Path = INDEX_DIR):
        self._index_dir = index_dir
        self._chunks: list[MemoryChunk] = []
        self._faiss_index = None
        self._embed_model = None
        self._use_faiss = False
        self._loaded = False
    
    def _load_embedder(self) -> bool:
        """Try to load sentence-transformers embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            self._embed_model = SentenceTransformer(EMBEDDING_MODEL)
            return True
        except ImportError:
            logger.warning("sentence-transformers not available — using keyword fallback")
            return False
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}")
            return False
    
    def _embed_texts(self, texts: list[str]) -> Any:
        """Embed texts using sentence-transformers."""
        import numpy as np
        if self._embed_model is None:
            raise RuntimeError("Embedding model not loaded")
        embeddings = self._embed_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return (embeddings / norms).astype("float32")
    
    def index_all(self, force_rebuild: bool = False) -> int:
        """Build or load the FAISS index from all data sources.
        
        Returns number of indexed chunks.
        """
        index_file = self._index_dir / "faiss.index"
        chunks_file = self._index_dir / "chunks.pkl"
        
        # Try loading existing index
        if not force_rebuild and index_file.exists() and chunks_file.exists():
            return self._load_index()
        
        # Build from scratch
        self._index_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Building RAG index from traces, postmortems, and dataset...")
        
        chunks = []
        chunks.extend(self._chunk_traces())
        chunks.extend(self._chunk_postmortems())
        chunks.extend(self._chunk_dataset())
        
        # Deduplicate
        seen = set()
        unique = []
        for c in chunks:
            if c.chunk_id not in seen:
                seen.add(c.chunk_id)
                unique.append(c)
        chunks = unique[:MAX_CHUNKS]
        self._chunks = chunks
        
        logger.info(f"Total unique chunks: {len(chunks)}")
        
        # Build FAISS index if available
        if self._load_embedder():
            self._build_faiss_index(chunks)
            self._use_faiss = True
        else:
            self._use_faiss = False
        
        # Save
        with open(chunks_file, "wb") as f:
            pickle.dump(chunks, f)
        
        self._loaded = True
        return len(chunks)
    
    def _build_faiss_index(self, chunks: list[MemoryChunk]) -> None:
        """Build FAISS index from chunks."""
        try:
            import faiss
            import numpy as np
        except ImportError:
            logger.warning("FAISS not available — using keyword fallback")
            self._use_faiss = False
            return
        
        texts = [c.text for c in chunks]
        logger.info(f"Embedding {len(texts)} chunks...")
        
        # Batch embedding to avoid OOM
        batch_size = 256
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            emb = self._embed_texts(batch)
            all_embeddings.append(emb)
        
        import numpy as np
        embeddings = np.vstack(all_embeddings)
        dim = embeddings.shape[1]
        
        # Use IndexFlatIP (inner product = cosine similarity on normalized vectors)
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        
        self._faiss_index = index
        
        # Save index
        faiss.write_index(index, str(self._index_dir / "faiss.index"))
        logger.info(f"FAISS index built: {index.ntotal} vectors, dim={dim}")
    
    def _load_index(self) -> int:
        """Load existing index from disk."""
        chunks_file = self._index_dir / "chunks.pkl"
        index_file = self._index_dir / "faiss.index"
        
        with open(chunks_file, "rb") as f:
            self._chunks = pickle.load(f)
        
        if index_file.exists():
            try:
                import faiss
                self._faiss_index = faiss.read_index(str(index_file))
                if self._load_embedder():
                    self._use_faiss = True
            except (ImportError, Exception) as e:
                logger.warning(f"FAISS load failed, using keyword fallback: {e}")
                self._use_faiss = False
        
        self._loaded = True
        logger.info(f"Loaded RAG index: {len(self._chunks)} chunks, faiss={self._use_faiss}")
        return len(self._chunks)
    
    def retrieve(self, query: str, top_k: int = TOP_K,
                 phase_filter: str | None = None,
                 source_filter: str | None = None) -> RetrievalResult:
        """Retrieve top-k relevant chunks for a query.
        
        Args:
            query: Natural language query or state description
            top_k: Number of results to return
            phase_filter: Optional phase to filter by (e.g. "RECON")
            source_filter: Optional source type filter (e.g. "trace", "postmortem")
        """
        if not self._loaded:
            self.index_all()
        
        t0 = time.time()
        
        if self._use_faiss and self._faiss_index is not None:
            result = self._faiss_retrieve(query, top_k, phase_filter, source_filter)
        else:
            result = self._keyword_retrieve(query, top_k, phase_filter, source_filter)
        
        result.latency_ms = (time.time() - t0) * 1000
        return result
    
    def _faiss_retrieve(self, query: str, top_k: int,
                        phase_filter: str | None,
                        source_filter: str | None) -> RetrievalResult:
        """FAISS-based semantic retrieval."""
        import numpy as np
        
        # Apply filters first
        candidates = self._filter_chunks(phase_filter, source_filter)
        if not candidates:
            return RetrievalResult(chunks=[], scores=[], query=query)
        
        # Embed query
        q_emb = self._embed_texts([query])
        
        # If filtered, we need to search subset
        if len(candidates) < len(self._chunks):
            # Re-embed filtered subset
            texts = [self._chunks[i].text for i in candidates]
            c_emb = self._embed_texts(texts)
            scores = (q_emb @ c_emb.T).flatten()
            top_indices = np.argsort(scores)[::-1][:top_k]
            results_chunks = [self._chunks[candidates[i]] for i in top_indices]
            results_scores = [float(scores[i]) for i in top_indices]
        else:
            # Search full index
            scores, indices = self._faiss_index.search(q_emb, min(top_k, len(self._chunks)))
            results_chunks = [self._chunks[i] for i in indices[0] if i >= 0]
            results_scores = [float(s) for s in scores[0][:len(results_chunks)]]
        
        return RetrievalResult(chunks=results_chunks, scores=results_scores, query=query)
    
    def _keyword_retrieve(self, query: str, top_k: int,
                          phase_filter: str | None,
                          source_filter: str | None) -> RetrievalResult:
        """Keyword-based fallback retrieval."""
        candidates = self._filter_chunks(phase_filter, source_filter)
        query_lower = query.lower()
        query_words = set(re.findall(r'\w+', query_lower))
        
        scored = []
        for idx in candidates:
            chunk = self._chunks[idx]
            text_lower = chunk.text.lower()
            text_words = set(re.findall(r'\w+', text_lower))
            
            # Jaccard-like overlap score
            overlap = len(query_words & text_words)
            if overlap == 0:
                continue
            score = overlap / max(1, len(query_words | text_words))
            
            # Boost for exact phrase matches
            for word in query_words:
                if len(word) > 4 and word in text_lower:
                    score += 0.1
            
            scored.append((idx, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]
        
        return RetrievalResult(
            chunks=[self._chunks[i] for i, _ in top],
            scores=[s for _, s in top],
            query=query,
        )
    
    def _filter_chunks(self, phase_filter: str | None,
                       source_filter: str | None) -> list[int]:
        """Filter chunk indices by metadata."""
        candidates = list(range(len(self._chunks)))
        if phase_filter:
            candidates = [i for i in candidates
                         if self._chunks[i].metadata.get("phase", "").upper() == phase_filter.upper()
                         or not self._chunks[i].metadata.get("phase")]
        if source_filter:
            candidates = [i for i in candidates
                         if source_filter.lower() in self._chunks[i].metadata.get("source", "").lower()]
        return candidates
    
    # ── Chunking ───────────────────────────────────────────────────────────
    
    def _chunk_traces(self) -> list[MemoryChunk]:
        """Extract memory chunks from trace event files (old + new format)."""
        chunks = []

        # Old flat format: events_*.jsonl with kind=step + agent_records
        for ef in sorted(TRACE_DIR.glob("events_*.jsonl"))[-200:]:
            try:
                with open(ef) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        d = json.loads(line)
                        if d.get("kind") != "step":
                            continue
                        for rec in d.get("agent_records", []):
                            chunk = self._trace_record_to_chunk(
                                rec,
                                phase=d.get("phase_before", "RECON"),
                                target=d.get("target_ip", ""),
                                source_file=ef.name,
                            )
                            if chunk:
                                chunks.append(chunk)
            except (json.JSONDecodeError, OSError):
                continue

        # New run_*/steps.jsonl format: one record per line
        for sf in sorted(TRACE_DIR.glob("run_*/steps.jsonl"))[-200:]:
            try:
                with open(sf) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        d = json.loads(line)
                        chunk = self._trace_record_to_chunk(
                            d,
                            phase=d.get("phase", "RECON"),
                            target="",
                            source_file=sf.parent.name,
                        )
                        if chunk:
                            chunks.append(chunk)
            except (json.JSONDecodeError, OSError):
                continue

        logger.info(f"Chunked {len(chunks)} trace memories")
        return chunks

    @staticmethod
    def _trace_record_to_chunk(
        rec: dict, phase: str, target: str, source_file: str,
    ) -> MemoryChunk | None:
        """Convert a single agent trace record to a MemoryChunk."""
        cmd = rec.get("command") or rec.get("chosen_action", "")
        reward = rec.get("reward", 0)
        if not cmd or reward <= 0:
            return None

        text_parts = [f"Phase: {phase.upper()}"]
        agent = rec.get("agent", "")
        if agent:
            text_parts.append(f"Agent: {agent}")
        if target:
            text_parts.append(f"Target: {target}")
        text_parts.append(f"Command: {cmd[:200]}")
        text_parts.append(f"Reward: {reward:.1f}")

        discoveries = rec.get("discoveries", [])
        if discoveries:
            text_parts.append(f"Found: {', '.join(str(d) for d in discoveries[:5])}")
        stdout = rec.get("stdout_snippet", "")
        if stdout:
            text_parts.append(f"Output: {stdout[:200]}")

        text = "\n".join(text_parts)
        return MemoryChunk(
            text=text[:CHUNK_SIZE],
            metadata={
                "source": "trace",
                "phase": phase.upper(),
                "reward": reward,
                "target": target,
                "file": source_file,
            },
        )
    
    def _chunk_postmortems(self) -> list[MemoryChunk]:
        """Extract memory chunks from postmortems."""
        chunks = []
        for pf in sorted(POSTMORTEM_DIR.glob("postmortem_*.json"))[-500:]:
            try:
                with open(pf) as f:
                    pm = json.load(f)
                
                if pm.get("model_used") == "offline":
                    continue
                
                outcomes = pm.get("key_outcomes", {})
                summary = outcomes.get("summary", "")
                wins = outcomes.get("wins", [])
                fails = outcomes.get("fails", [])
                skills = pm.get("skill_cards", [])
                
                if not summary:
                    continue
                
                # Build postmortem chunk
                parts = [f"Postmortem: {pm.get('run_id', 'unknown')}"]
                parts.append(f"Summary: {summary[:200]}")
                if wins:
                    parts.append(f"Wins: {'; '.join(wins[:3])}")
                if fails:
                    parts.append(f"Fails: {'; '.join(fails[:3])}")
                for s in skills[:2]:
                    if s.get("if_condition") and s.get("then_action"):
                        parts.append(f"Pattern: IF {s['if_condition']} THEN {s['then_action']}")
                
                text = "\n".join(parts)
                chunks.append(MemoryChunk(
                    text=text[:CHUNK_SIZE],
                    metadata={
                        "source": "postmortem",
                        "run_id": pm.get("run_id", ""),
                        "file": pf.name,
                    },
                ))
            except (json.JSONDecodeError, OSError):
                continue
        
        logger.info(f"Chunked {len(chunks)} postmortem memories")
        return chunks
    
    def _chunk_dataset(self) -> list[MemoryChunk]:
        """Extract high-quality chunks from v2 and v3 datasets (retrieval-only)."""
        chunks = []
        dataset_files = []
        for d in (DATASET_V2_DIR, DATASET_V3_DIR):
            for name in ("train.jsonl", "sft_train.jsonl"):
                candidate = d / name
                if candidate.exists():
                    dataset_files.append(candidate)

        for dataset_file in dataset_files:
            try:
                with open(dataset_file) as f:
                    for line in f:
                        data = json.loads(line.strip())
                        quality = data.get("quality_score", 0)
                        if quality < 0.8:
                            continue

                        # Use the assistant response as a high-quality memory
                        messages = data.get("messages", [])
                        if len(messages) >= 3:
                            user_msg = messages[1].get("content", "")
                            asst_msg = messages[2].get("content", "")
                            task = data.get("task_family", "")

                            text = f"[{task}] Q: {user_msg[:150]} A: {asst_msg[:300]}"
                            chunks.append(MemoryChunk(
                                text=text[:CHUNK_SIZE],
                                metadata={
                                    "source": f"dataset-{dataset_file.parent.name}",
                                    "task": task,
                                    "quality": quality,
                                    "phase": data.get("metadata", {}).get("phase", ""),
                                },
                            ))
            except (json.JSONDecodeError, OSError):
                continue

        logger.info(f"Chunked {len(chunks)} dataset memories from {len(dataset_files)} files")
        return chunks
    
    def get_stats(self) -> dict:
        """Return index statistics."""
        if not self._loaded:
            return {"loaded": False}
        
        source_counts: dict[str, int] = {}
        phase_counts: dict[str, int] = {}
        for c in self._chunks:
            src = c.metadata.get("source", "unknown")
            source_counts[src] = source_counts.get(src, 0) + 1
            phase = c.metadata.get("phase", "unknown")
            if phase:
                phase_counts[phase] = phase_counts.get(phase, 0) + 1
        
        return {
            "loaded": True,
            "total_chunks": len(self._chunks),
            "faiss_enabled": self._use_faiss,
            "by_source": source_counts,
            "by_phase": phase_counts,
        }


# ── Singleton ──────────────────────────────────────────────────────────────

_retriever: AriaskaRetriever | None = None


def get_retriever() -> AriaskaRetriever:
    """Get or create the singleton retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = AriaskaRetriever()
    return _retriever


def build_rag_context(
    phase: str,
    evidence: dict[str, Any],
    recent_commands: list[str] | None = None,
    query_override: str | None = None,
    top_k: int = TOP_K,
) -> str:
    """Build RAG context for injection into Qwen prompts.
    
    This is the main runtime entry point. Called by GPTManager/SmartCoach
    before assembling the Qwen prompt.
    
    Args:
        phase: Current attack phase
        evidence: Current evidence snapshot (ports, services, etc.)
        recent_commands: Last N commands executed
        query_override: Custom query instead of auto-building one
        top_k: Number of chunks to retrieve
    
    Returns:
        Formatted context string for prompt injection, or empty string.
    """
    retriever = get_retriever()
    if not retriever._loaded:
        try:
            retriever.index_all()
        except Exception as e:
            logger.warning(f"RAG index build failed: {e}")
            return ""
    
    # Build query from current state
    if query_override:
        query = query_override
    else:
        parts = [f"Phase: {phase}"]
        if evidence:
            for k, v in evidence.items():
                if v:
                    if isinstance(v, (list, set)):
                        parts.append(f"{k}: {', '.join(str(x) for x in list(v)[:5])}")
                    else:
                        parts.append(f"{k}: {v}")
        if recent_commands:
            parts.append(f"Recent: {'; '.join(recent_commands[-3:])}")
        query = "\n".join(parts)
    
    result = retriever.retrieve(query, top_k=top_k, phase_filter=phase)
    return result.to_context_string()


# ── CLI Entry ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="Build or query Ariaska RAG index")
    parser.add_argument("--build", action="store_true", help="Build index from scratch")
    parser.add_argument("--query", type=str, help="Test query")
    parser.add_argument("--stats", action="store_true", help="Print index stats")
    args = parser.parse_args()
    
    retriever = get_retriever()
    
    if args.build:
        n = retriever.index_all(force_rebuild=True)
        print(f"Indexed {n} chunks")
    
    if args.stats:
        print(json.dumps(retriever.get_stats(), indent=2))
    
    if args.query:
        result = retriever.retrieve(args.query)
        print(f"\nQuery: {args.query}")
        print(f"Results ({result.latency_ms:.1f}ms):")
        for chunk, score in zip(result.chunks, result.scores):
            print(f"  [{score:.3f}] {chunk.text[:150]}...")
