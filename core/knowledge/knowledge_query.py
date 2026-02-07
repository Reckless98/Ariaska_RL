#!/usr/bin/env python3
"""
core/knowledge/knowledge_query.py — ARIASKA Knowledge Query v1.0
🔍 RAG-powered knowledge retrieval for SmartMentor context enrichment.

Connects to the ChromaDB knowledge base (populated by data/knowledge_loader.py)
and provides fast semantic search for attack context enrichment.

Usage:
    from core.knowledge.knowledge_query import query_knowledge
    results = query_knowledge("vsftpd 2.3.4 exploit", top_k=3)
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("ariaska.knowledge_query")

# Lazy-loaded singletons
_chroma_client = None
_kb_collection = None
_embedder = None
_initialized = False
_init_failed = False


def _ensure_initialized() -> bool:
    """Lazy-initialize ChromaDB and embedder (once)."""
    global _chroma_client, _kb_collection, _embedder, _initialized, _init_failed

    if _initialized:
        return _kb_collection is not None
    if _init_failed:
        return False

    try:
        import chromadb
        from sentence_transformers import SentenceTransformer

        _chroma_client = chromadb.PersistentClient(path="./chroma_storage")
        _kb_collection = _chroma_client.get_or_create_collection("ariaska_kb")
        _embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        _initialized = True

        count = _kb_collection.count()
        logger.info(f"Knowledge base initialized: {count} documents in ariaska_kb")
        return True

    except ImportError:
        logger.debug("chromadb or sentence-transformers not installed; RAG disabled")
        _init_failed = True
        return False
    except Exception as e:
        logger.warning(f"Knowledge base init failed: {e}")
        _init_failed = True
        return False


def query_knowledge(
    query: str,
    top_k: int = 3,
    phase_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Query the knowledge base for relevant documents.

    Args:
        query: Natural language query (e.g. "exploit vsftpd 2.3.4").
        top_k: Number of results to return.
        phase_filter: Optional attack phase filter (e.g. "Exploitation").

    Returns:
        List of dicts with keys: document, description, phase, distance.
        Empty list if knowledge base is unavailable.
    """
    if not _ensure_initialized():
        return []

    try:
        # Build query embedding
        query_embedding = _embedder.encode([query]).tolist()

        # Optional phase filter
        where_filter = None
        if phase_filter:
            where_filter = {"phase": phase_filter}

        results = _kb_collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=where_filter,
        )

        if not results or not results.get("documents"):
            return []

        docs = results["documents"][0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        output = []
        for i, doc in enumerate(docs):
            meta = metadatas[i] if i < len(metadatas) else {}
            dist = distances[i] if i < len(distances) else 1.0
            output.append({
                "document": doc,
                "description": meta.get("description", ""),
                "phase": meta.get("phase", ""),
                "distance": dist,
            })

        return output

    except Exception as e:
        logger.debug(f"Knowledge query failed: {e}")
        return []


def build_rag_context(
    phase: str,
    discoveries: Dict[str, Any],
    target: str = "",
    max_snippets: int = 3,
) -> str:
    """Build RAG context string from knowledge base for mentor prompts.

    Creates a natural language query from the current attack state,
    retrieves relevant knowledge, and formats it for LLM consumption.

    Args:
        phase: Current attack phase name.
        discoveries: Current discoveries dict.
        target: Target IP/hostname.
        max_snippets: Maximum knowledge snippets to include.

    Returns:
        Formatted context string, or "" if no relevant knowledge found.
    """
    # Build query from attack state
    query_parts = [phase]

    # Add discovered services/ports for context
    for disc_type in ["services", "ports", "vulns"]:
        vals = discoveries.get(disc_type, set())
        if isinstance(vals, (set, list)):
            for v in list(vals)[:5]:
                query_parts.append(str(v))

    query = " ".join(query_parts)
    results = query_knowledge(query, top_k=max_snippets, phase_filter=None)

    if not results:
        return ""

    lines = ["=== RELEVANT KNOWLEDGE FROM DATABASE ==="]
    for r in results:
        if r["distance"] < 1.5:  # Only include reasonably close matches
            desc = r.get("description", "")
            doc = r.get("document", "")
            if desc and desc != "N/A":
                lines.append(f"• {doc} — {desc}")
            else:
                lines.append(f"• {doc}")

    if len(lines) <= 1:
        return ""  # No close-enough matches

    lines.append("")
    return "\n".join(lines)
