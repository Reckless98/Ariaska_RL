# core/vector_search.py — Tactical Memory Module for RL Agent

import json
import os
from typing import Dict, Any, List

import re
import hashlib


def generate_embedding(text: str) -> List[float]:
    """
    Placeholder for a future model-based encoder.
    Here: simple hashed token scoring for simulation.
    """
    tokens = re.findall(r"\w+", text.lower())
    seed = sum([hashlib.sha256(t.encode()).digest()[0] for t in tokens])
    random_vector = [(seed * (i + 1) % 97) / 100.0 for i in range(512)]
    return random_vector


class VectorSearch:
    def __init__(self, storage_path: str = 'data/knowledge_sources/initial.json', cache_size: int = 50):
        self.storage_path = storage_path
        self.cache_size = cache_size
        self.cache = {} if cache_size > 0 else None

        self.database = self.load_database()
        print(f"[VectorSearch] Initialized with {len(self.database)} entries")

    def load_database(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.storage_path):
            return []

        try:
            with open(self.storage_path, 'r') as file:
                data = json.load(file)
                if isinstance(data, list):
                    return data
        except Exception as e:
            print(f"[VectorSearch] Failed to load DB: {e}")
        return []

    def save_database(self):
        try:
            with open(self.storage_path, 'w') as file:
                json.dump(self.database, file, indent=2)
        except Exception as e:
            print(f"[VectorSearch] Error saving DB: {e}")

    def add_memory_entry(self, text: str, phase: str = None, success: bool = False, tags: List[str] = None):
        entry = {
            "text": text,
            "embedding": generate_embedding(text),
            "phase": phase,
            "success": success,
            "tags": tags or [],
        }
        self.database.append(entry)
        self.save_database()

    def query(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        if self.cache and query_text in self.cache:
            return {'query': query_text, 'results': self.cache[query_text]}

        query_vec = generate_embedding(query_text)
        scored = self.score_entries(query_vec)

        top = sorted(scored, key=lambda x: x['score'], reverse=True)[:top_k]
        results = [r['entry'] for r in top]

        if self.cache is not None:
            if len(self.cache) >= self.cache_size:
                self.cache.pop(next(iter(self.cache)))
            self.cache[query_text] = results

        return {'query': query_text, 'results': results}

    def score_entries(self, query_vec: List[float]) -> List[Dict[str, Any]]:
        scores = []
        for entry in self.database:
            entry_vec = entry.get('embedding', [0.0] * len(query_vec))
            score = self.cosine_similarity(query_vec, entry_vec)
            scores.append({"entry": entry, "score": score})
        return scores

    @staticmethod
    def cosine_similarity(v1: List[float], v2: List[float]) -> float:
        try:
            dot = sum(a * b for a, b in zip(v1, v2))
            norm1 = sum(a ** 2 for a in v1) ** 0.5
            norm2 = sum(b ** 2 for b in v2) ** 0.5
            return dot / (norm1 * norm2) if norm1 and norm2 else 0.0
        except:
            return 0.0

    def query_contextual_memory(self, phase: str, context: str = "", top_k: int = 5) -> List[Dict[str, Any]]:
        results = []
        query = f"{phase} {context}".strip()
        vector = generate_embedding(query)

        for entry in self.database:
            if entry.get("phase") != phase:
                continue
            entry_vec = entry.get("embedding", [0.0] * len(vector))
            score = self.cosine_similarity(vector, entry_vec)
            if score > 0.4:
                results.append({"command": entry["text"], "score": round(score, 3)})

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]

    def reinforce_command_pattern(self, command: str, phase: str = None, success: bool = True, tags: List[str] = None):
        exists = any(e for e in self.database if e["text"] == command)
        if not exists:
            self.add_memory_entry(command, phase=phase, success=success, tags=tags or [])

    def update_database(self, new_entries: List[Dict[str, Any]]) -> None:
        if not isinstance(new_entries, list):
            print("[VectorSearch] Skipped invalid entry list.")
            return

        for entry in new_entries:
            if "text" in entry:
                entry["embedding"] = generate_embedding(entry["text"])
                self.database.append(entry)

        self.save_database()

    def get_agent_recommendations(self, agent_id: str, phase: str, context: str, top_k: int = 5) -> Dict[str, Any]:
        query_text = f"[Agent {agent_id}] Phase: {phase} Context: {context}"
        return self.query(query_text, top_k=top_k)
