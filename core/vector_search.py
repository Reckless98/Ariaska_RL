# core/vector_search.py — ARIASKA Vector Intelligence v12.0 APEX
# 🧠 Hybrid GPT-Vector Memory | ⚡ Context-Aware Suggestions | 🌐 Orion-Synced Knowledge Core

import json
import os
import re
import hashlib
import subprocess
from typing import Dict, Any, List
from rich.console import Console
from core.gpt_manager import GPTManager

console = Console()


def generate_embedding(text: str) -> List[float]:
    tokens = re.findall(r"\w+", text.lower())
    seed = sum([hashlib.sha256(t.encode()).digest()[0] for t in tokens])
    vector = [(seed * (i + 11) % 101) / 100.0 for i in range(256)]
    return vector


class VectorSearch:
    """
    Hybrid GPT-Vector search and memory system for ARIASKA. Supports context-aware suggestions, Orion-synced knowledge,
    and GPT-powered fallback for low-confidence queries. All GPT calls are routed through GPTManager for orchestration,
    caching, and token tracking.
    """
    def __init__(
        self, storage_path="data/knowledge/ariaska_vectors.json", cache_size=100
    ):
        self.storage_path = storage_path
        self.cache_size = cache_size
        self.cache = {} if cache_size > 0 else None
        self.gpt_trigger_threshold = 0.4
        self.gpt_model = "gpt-4o-mini"
        self.database = self._load_database()
        self.gpt_manager = GPTManager()
        console.print(
            f"[green]✔ VectorSearch v12.0 initialized with {len(self.database)} entries[/green]"
        )

    def _load_database(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.storage_path):
            return []
        try:
            with open(self.storage_path, "r") as file:
                data = json.load(file)
                return data if isinstance(data, list) else []
        except Exception as e:
            console.print(f"[red]⚠ Failed to load vector DB: {e}[/red]")
            return []

    def save_database(self):
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, "w") as file:
                json.dump(self.database, file, indent=2)
            console.print(f"[cyan]💾 Vector DB saved[/cyan]")
        except Exception as e:
            console.print(f"[red]❌ Error saving Vector DB: {e}[/red]")

    def add_entry(self, text: str, phase: str = "unknown", tags: List[str] = None):
        entry = {
            "text": text,
            "embedding": generate_embedding(text),
            "phase": phase,
            "tags": tags or [],
            "usage_count": 1,
        }
        self.database.append(entry)
        self.save_database()
        console.print(f"[blue]➕ New vector entry added:[/blue] {text}")

    def query(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        if self.cache and query_text in self.cache:
            console.print(f"[yellow]⚡ Cache hit for query:[/yellow] {query_text}")
            return {"query": query_text, "results": self.cache[query_text]}

        query_vec = generate_embedding(query_text)
        scored = self._score_entries(query_vec)
        top_results = sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]
        results = [r["entry"] for r in top_results]

        avg_score = sum(r["score"] for r in top_results) / top_k if top_results else 0
        if avg_score < self.gpt_trigger_threshold:
            gpt_suggestion = self._gpt_suggest(query_text)
            if gpt_suggestion:
                results.insert(0, {"text": gpt_suggestion, "source": "GPT-Fallback"})

        if self.cache is not None:
            if len(self.cache) >= self.cache_size:
                self.cache.pop(next(iter(self.cache)))
            self.cache[query_text] = results

        return {"query": query_text, "results": results}

    def _gpt_suggest(self, query_text: str) -> str:
        """
        Use GPTManager to suggest a tactical command for a query. Fallback to empty string on failure.
        """
        prompt = f"Suggest a tactical cybersecurity command for: {query_text}\nRespond ONLY with the command."
        try:
            suggestion = self.gpt_manager.gpt_request(prompt, model=self.gpt_model)
            suggestion = suggestion.strip().splitlines()[0]
            console.print(f"[magenta]🎯 GPT Suggestion:[/magenta] {suggestion}")
            return suggestion
        except Exception as e:
            console.print(f"[red]⚠ GPT suggestion failed: {e}[/red]")
            return ""

    def _score_entries(self, query_vec: List[float]) -> List[Dict[str, Any]]:
        scores = []
        for entry in self.database:
            entry_vec = entry.get("embedding", [0.0] * len(query_vec))
            score = self._cosine_similarity(query_vec, entry_vec)
            scores.append({"entry": entry, "score": round(score, 4)})
        return scores

    @staticmethod
    def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
        try:
            dot = sum(a * b for a, b in zip(v1, v2))
            norm1 = sum(a**2 for a in v1) ** 0.5
            norm2 = sum(b**2 for b in v2) ** 0.5
            return dot / (norm1 * norm2) if norm1 and norm2 else 0.0
        except:
            return 0.0

    # ─────────────────────────────────────────────
    # 📊 Analytics & Diagnostics
    # ─────────────────────────────────────────────
    def display_top_patterns(self, top_n=10):
        pattern_counts = sorted(
            self.database, key=lambda e: e.get("usage_count", 0), reverse=True
        )[:top_n]
        from rich.table import Table

        table = Table(title=f"🔍 Top {top_n} Reinforced Patterns")
        table.add_column("Command / Pattern", style="cyan")
        table.add_column("Usage Count", style="magenta")
        for entry in pattern_counts:
            table.add_row(entry.get("text", "N/A"), str(entry.get("usage_count", 1)))
        console.print(table)

    def gpt_enhance_database(self):
        """
        Enhance database entries with GPT-powered tactical insights using GPTManager.
        """
        console.print(
            "[blue]⚡ Running GPT enhancement across database entries...[/blue]"
        )
        enhanced = 0
        for entry in self.database:
            if "gpt_insight" not in entry:
                prompt = f"Provide a brief tactical insight for this cybersecurity command: {entry['text']}"
                try:
                    insight = self.gpt_manager.gpt_request(prompt, model=self.gpt_model)
                    entry["gpt_insight"] = insight.strip()
                    enhanced += 1
                except Exception as e:
                    console.print(
                        f"[yellow]⚠ GPT enhancement failed for {entry['text']}: {e}[/yellow]"
                    )
        self.save_database()
        console.print(f"[green]✔ Enhanced {enhanced} entries with GPT insights[/green]")


# ─────────────────────────────────────────────
# 🚀 Diagnostic Mode
# ─────────────────────────────────────────────
if __name__ == "__main__":
    vs = VectorSearch()
    test_query = "lateral movement windows smb"
    results = vs.query(test_query)
    console.print(results)
    vs.display_top_patterns()
    vs.gpt_enhance_database()
