# core/semantic_memory/chromadb_store.py — ARIASKA Vector Memory v1.0
# 🧠 Semantic Memory Storage | 🔍 Vector Search | 🔄 Memory Retrieval and Storage

import os
import json
import time
import uuid
import threading
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from rich.console import Console

console = Console()

# Conditional import to handle ChromaDB availability
try:
    import chromadb
    from chromadb import Settings
    CHROMADB_AVAILABLE = True
    console.print("[green]✓ ChromaDB available - using persistent vector store[/green]")
except ImportError:
    console.print("[yellow]⚠ ChromaDB not available, falling back to in-memory vector store[/yellow]")
    CHROMADB_AVAILABLE = False

class ChromaDB:
    """
    ChromaDB integration for ARIASKA_RL.
    Provides vector database capabilities with fallback to in-memory when ChromaDB is not available.
    """
    def __init__(
        self,
        collection_name: str = "ariaska_memory",
        persist_directory: str = "data/chromadb",
        embedding_function: Optional[Any] = None,
        gpt_manager: Optional[Any] = None
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.gpt_manager = gpt_manager
        self.client = None
        self.collection = None
        self.fallback_memory = {}
        self.lock = threading.Lock()
        self._initialize_storage(embedding_function)
        console.print(f"[green]✓ ChromaDB initialized for collection '{collection_name}'[/green]")

    def _initialize_storage(self, embedding_function: Optional[Any] = None):
        if CHROMADB_AVAILABLE:
            try:
                os.makedirs(self.persist_directory, exist_ok=True)
                self.client = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
                try:
                    self.collection = self.client.get_collection(name=self.collection_name)
                    console.print(f"[green]✓ Connected to existing collection '{self.collection_name}'[/green]")
                except Exception:
                    self.collection = self.client.create_collection(
                        name=self.collection_name,
                        embedding_function=embedding_function,
                        metadata={"created_at": time.time()}
                    )
                    console.print(f"[green]✓ Created new collection '{self.collection_name}'[/green]")
            except Exception as e:
                console.print(f"[yellow]⚠ ChromaDB initialization failed: {e}. Using in-memory fallback.[/yellow]")
                self.client = None
                self.collection = None
        else:
            console.print("[yellow]⚠ Using in-memory vector store (ChromaDB not available)[/yellow]")
            self.client = None
            self.collection = None

    def add_memory(
        self,
        text: str,
        metadata: Dict[str, Any],
        embedding: Optional[List[float]] = None,
        id: Optional[str] = None
    ) -> str:
        memory_id = id or str(uuid.uuid4())
        with self.lock:
            if self.collection:
                try:
                    self.collection.add(
                        documents=[text],
                        metadatas=[metadata],
                        ids=[memory_id]
                    )
                    return memory_id
                except Exception as e:
                    console.print(f"[yellow]⚠ ChromaDB add failed: {e}. Using in-memory fallback.[/yellow]")
            if embedding is None and self.gpt_manager:
                embedding = self._generate_embedding(text)
            self.fallback_memory[memory_id] = {
                "text": text,
                "metadata": metadata,
                "embedding": embedding,
                "timestamp": time.time()
            }
            return memory_id

    def search_similar(
        self,
        query: str,
        n_results: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        with self.lock:
            if self.collection:
                try:
                    filter_dict = metadata_filter or {}
                    results = self.collection.query(
                        query_texts=[query],
                        n_results=n_results,
                        where=filter_dict
                    )
                    formatted_results = []
                    if results and "documents" in results and results["documents"]:
                        documents = results["documents"][0]
                        metadatas = results["metadatas"][0]
                        distances = results["distances"][0] if "distances" in results else [0.0] * len(documents)
                        ids = results["ids"][0]
                        for i, (doc, meta, dist, id) in enumerate(zip(documents, metadatas, distances, ids)):
                            similarity = 1.0 - min(1.0, max(0.0, dist))
                            if similarity >= threshold:
                                formatted_results.append({
                                    "id": id,
                                    "text": doc,
                                    "metadata": meta,
                                    "similarity": similarity
                                })
                    return formatted_results
                except Exception as e:
                    console.print(f"[yellow]⚠ ChromaDB query failed: {e}. Using in-memory fallback.[/yellow]")
            if not self.fallback_memory:
                return []
            query_embedding = self._generate_embedding(query)
            if not query_embedding:
                return []
            results = []
            for memory_id, memory in self.fallback_memory.items():
                if not memory.get("embedding"):
                    continue
                if metadata_filter and not self._matches_filter(memory["metadata"], metadata_filter):
                    continue
                similarity = self._calculate_similarity(query_embedding, memory["embedding"])
                if similarity >= threshold:
                    results.append({
                        "id": memory_id,
                        "text": memory["text"],
                        "metadata": memory["metadata"],
                        "similarity": similarity
                    })
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:n_results]

    def get_memory_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            if self.collection:
                try:
                    result = self.collection.get(ids=[memory_id])
                    if result and "documents" in result and result["documents"]:
                        return {
                            "id": memory_id,
                            "text": result["documents"][0],
                            "metadata": result["metadatas"][0] if "metadatas" in result else {}
                        }
                except Exception as e:
                    console.print(f"[yellow]⚠ ChromaDB get failed: {e}. Using in-memory fallback.[/yellow]")
            if memory_id in self.fallback_memory:
                memory = self.fallback_memory[memory_id]
                return {
                    "id": memory_id,
                    "text": memory["text"],
                    "metadata": memory["metadata"]
                }
            return None

    def delete_memory(self, memory_id: str) -> bool:
        with self.lock:
            if self.collection:
                try:
                    self.collection.delete(ids=[memory_id])
                    return True
                except Exception as e:
                    console.print(f"[yellow]⚠ ChromaDB delete failed: {e}. Using in-memory fallback.[/yellow]")
            if memory_id in self.fallback_memory:
                del self.fallback_memory[memory_id]
                return True
            return False

    def update_memory(
        self,
        memory_id: str,
        text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        existing = self.get_memory_by_id(memory_id)
        if not existing:
            return False
        updated_text = text if text is not None else existing.get("text", "")
        updated_metadata = {**existing.get("metadata", {}), **(metadata or {})}
        self.delete_memory(memory_id)
        self.add_memory(
            text=updated_text,
            metadata=updated_metadata,
            id=memory_id
        )
        return True

    def _generate_embedding(self, text: str) -> List[float]:
        if self.gpt_manager:
            try:
                response = self.gpt_manager.embedding_request(text)
                if isinstance(response, list) and len(response) > 0:
                    return response
            except Exception as e:
                console.print(f"[yellow]⚠ GPTManager embedding failed: {e}. Using fallback.[/yellow]")
        import hashlib
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        rng = np.random.RandomState(hash_val)
        return rng.rand(16).tolist()

    def _calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        if not vec1 or not vec2:
            return 0.0
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def _matches_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        for key, value in filter_dict.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True

    def count_memories(self) -> int:
        with self.lock:
            if self.collection:
                try:
                    return self.collection.count()
                except Exception:
                    pass
            return len(self.fallback_memory)

    def backup(self, backup_path: Optional[str] = None) -> str:
        timestamp = int(time.time())
        backup_path = backup_path or f"data/backups/memory_backup_{timestamp}.json"
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        memories = []
        with self.lock:
            if self.collection:
                try:
                    all_data = self.collection.get()
                    if all_data and "documents" in all_data and all_data["documents"]:
                        for i, doc in enumerate(all_data["documents"]):
                            memory_id = all_data["ids"][i] if "ids" in all_data and i < len(all_data["ids"]) else f"memory_{i}"
                            metadata = all_data["metadatas"][i] if "metadatas" in all_data and i < len(all_data["metadatas"]) else {}
                            memories.append({
                                "id": memory_id,
                                "text": doc,
                                "metadata": metadata
                            })
                except Exception as e:
                    console.print(f"[yellow]⚠ ChromaDB export failed: {e}. Using in-memory fallback.[/yellow]")
            if not memories or not self.collection:
                for memory_id, memory in self.fallback_memory.items():
                    memories.append({
                        "id": memory_id,
                        "text": memory["text"],
                        "metadata": memory["metadata"]
                    })
        try:
            with open(backup_path, "w") as f:
                json.dump({
                    "timestamp": timestamp,
                    "collection": self.collection_name,
                    "memories": memories
                }, f, indent=2)
            console.print(f"[green]✓ Memory backup created at {backup_path}[/green]")
            return backup_path
        except Exception as e:
            console.print(f"[red]❌ Memory backup failed: {e}[/red]")
            return ""

    def restore(self, backup_path: str) -> bool:
        try:
            with open(backup_path, "r") as f:
                backup_data = json.load(f)
            self.reset()
            memories = backup_data.get("memories", [])
            for memory in memories:
                self.add_memory(
                    text=memory["text"],
                    metadata=memory["metadata"],
                    id=memory.get("id")
                )
            console.print(f"[green]✓ Restored {len(memories)} memories from backup[/green]")
            return True
        except Exception as e:
            console.print(f"[red]❌ Memory restore failed: {e}[/red]")
            return False

    def reset(self):
        with self.lock:
            if self.collection:
                try:
                    if hasattr(self.collection, "delete") and callable(self.collection.delete):
                        all_data = self.collection.get()
                        if all_data and "ids" in all_data and all_data["ids"]:
                            self.collection.delete(ids=all_data["ids"])
                    else:
                        self.client.delete_collection(self.collection_name)
                        self.collection = self.client.create_collection(
                            name=self.collection_name,
                            metadata={"created_at": time.time()}
                        )
                except Exception as e:
                    console.print(f"[yellow]⚠ ChromaDB reset failed: {e}[/yellow]")
            self.fallback_memory.clear()
            console.print("[green]✓ Memory store reset complete[/green]")

    def get_most_relevant(
        self,
        agent_id: str,
        current_state: Dict[str, Any],
        n_results: int = 3
    ) -> List[Dict[str, Any]]:
        query = f"Agent: {agent_id}, "
        if isinstance(current_state, dict):
            phase = current_state.get("phase", "unknown")
            query += f"Phase: {phase}, "
            for key in ["privilege_level", "blue_team_alert", "detection_risk"]:
                if key in current_state:
                    query += f"{key}: {current_state[key]}, "
            if "open_ports" in current_state:
                ports = current_state["open_ports"]
                if len(ports) > 0:
                    query += f"Ports: {', '.join(map(str, ports[:5]))}"
        else:
            query += str(current_state)
        return self.search_similar(
            query=query,
            n_results=n_results,
            metadata_filter={"agent_id": agent_id},
            threshold=0.6
        )

class SemanticMemoryStore(ChromaDB):
    """
    Vector database integration for ARIASKA_RL using ChromaDB.
    Provides semantic search capabilities for agent memories.
    """
    pass

if __name__ == "__main__":
    console.print("[bold]Testing SemanticMemoryStore[/bold]")
    store = SemanticMemoryStore()
    memory1_id = store.add_memory(
        text="nmap -sV -sC 10.10.10.10 found open ports 22, 80",
        metadata={
            "agent_id": "RedAgent",
            "phase": "recon",
            "reward": 10.0,
            "timestamp": time.time()
        }
    )
    memory2_id = store.add_memory(
        text="gobuster found /admin directory on the web server",
        metadata={
            "agent_id": "RedAgent",
            "phase": "enumeration",
            "reward": 15.0,
            "timestamp": time.time()
        }
    )
    results = store.search_similar("web enumeration with gobuster")
    console.print(f"Found {len(results)} similar memories:")
    for result in results:
        console.print(f"- {result['text']} (similarity: {result['similarity']:.2f})")
    backup_path = store.backup()
    if backup_path:
        store.reset()
        console.print(f"After reset: {store.count_memories()} memories")
        store.restore(backup_path)
        console.print(f"After restore: {store.count_memories()} memories")
    console.print("[green]✓ Test complete[/green]")
