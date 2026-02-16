# core/models/gpt_context_encoder.py — ARIASKA GPT Context Encoder v2.0 APEX
# 🧠 Token-Efficient Vectorization | ⚡ Strategic GPT Triggers | 🔄 Smart Caching Core

import os
import json
import hashlib
import logging
import subprocess
from typing import List
from rich.console import Console
from core.gpt_manager import GPTManager

console = Console()
logger = logging.getLogger("ariaska.context_encoder")

class GPTContextEncoder:
    """
    Encodes cybersecurity context into dense vector representations using GPTManager for LLM-powered embeddings.
    Delegates all GPT calls to GPTManager for token tracking, caching, and fallback. Maintains a local cache for efficiency.
    """
    def __init__(self, cache_dir="core/memories/shared/gpt_vectors"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, "context_vectors.json")
        self.cache = self._load_cache()
        self.gpt_manager = GPTManager.get_instance()
        logger.debug(f"ContextEncoder Ready — Cached: {len(self.cache)}")

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as f:
                    return json.load(f)
            except:
                console.print("[yellow]⚠ Corrupt cache. Starting fresh.[/yellow]")
        return {}

    def _save_cache(self):
        try:
            with open(self.cache_file, "w") as f:
                json.dump(self.cache, f)
        except Exception as e:
            console.print(f"[red]❌ Cache save error: {e}[/red]")

    def _hash_context(self, context: str) -> str:
        return hashlib.sha256(context.encode()).hexdigest()

    def encode(self, context_text: str) -> List[float]:
        key = self._hash_context(context_text)
        if (key in self.cache):
            return self.cache[key]
        if len(context_text) < 50:
            vector = self._stub_vector(context_text)
        else:
            vector = self._gpt_vectorize(context_text)
        self.cache[key] = vector
        self._save_cache()
        return vector

    def _stub_vector(self, context: str) -> List[float]:
        seed = sum(ord(c) for c in context) % 97
        return [((seed * (i + 7)) % 200) / 100.0 - 1.0 for i in range(16)]

    def _gpt_vectorize(self, context_text: str) -> List[float]:
        """
        Use GPTManager to vectorize context text. Falls back to stub if LLM fails.
        """
        prompt = f"Vectorize this cybersecurity context into 16 floats between -1 and 1:\n{context_text}\nRespond ONLY with JSON array."
        try:
            response = self.gpt_manager.gpt_request(prompt, task_type="embedding")
            vector = json.loads(response)
            if isinstance(vector, list) and len(vector) == 16:
                console.print(f"[cyan]🧠 GPT vector generated.[/cyan]")
                return vector
            raise ValueError("Malformed GPT response.")
        except Exception as e:
            console.print(f"[yellow]⚠ GPT failed: {e} — Using stub[/yellow]")
            return self._stub_vector(context_text)

# ─────────────────────────────────────────────
# 🚀 Diagnostic Mode
# ─────────────────────────────────────────────
if __name__ == "__main__":
    encoder = GPTContextEncoder()
    sample_context = (
        "Privilege escalation detected on Linux target with weak sudo permissions."
    )
    vector = encoder.encode(sample_context)
    console.print(vector)
