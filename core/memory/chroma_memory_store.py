# core/memory/chroma_memory_store.py — ARIASKA ChromaDB Vector Memory v1.0
# 🧠 Semantic Memory | 🔍 Vector Search | 📊 Command Embedding | 🔄 Multi-Agent Memory Integration

import os
import json
import uuid
import time
from typing import Dict, List, Any, Union, Optional
import numpy as np
from rich.console import Console

console = Console()

class ChromaMemoryStore:
    """
    ChromaDB-backed semantic memory store for ARIASKA_RL.
    Provides vector-based storage and retrieval of agent experiences, commands, and observations.
    Features:
    - Semantic search for similar commands/experiences
    - Agent-specific collections with centralized querying
    - Command and state vectorization
    - Prioritized memory retrieval
    - Deduplication via vector similarity
    """

    def __init__(
        self, 
        persist_directory: str = "data/vector_memory",
        embedding_function=None,
        collection_name: str = "ariaska_memory",
        gpt_manager=None
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.gpt_manager = gpt_manager
        self.client = None
        self.collection = None
        self.embedding_function = embedding_function
        self._initialize_chroma()

    def _initialize_chroma(self):
        """Initialize ChromaDB client and collection."""
        try:
            import chromadb
            # Import Settings directly from chromadb for newer versions
            from chromadb import Settings
            
            # Create directory if it doesn't exist
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # Initialize client
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Try to get collection or create if it doesn't exist
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                console.print(f"[green]✓ ChromaDB collection '{self.collection_name}' loaded[/green]")
                console.print(f"[cyan]📊 Collection count: {self.collection.count()}[/cyan]")
            except ValueError:
                # Create embedding function if not provided
                if self.embedding_function is None:
                    if chromadb.utils.embedding_functions.OpenAIEmbeddingFunction is not None:
                        # Try to get API key from environment
                        api_key = os.environ.get("OPENAI_API_KEY")
                        if api_key:
                            try:
                                self.embedding_function = chromadb.utils.embedding_functions.OpenAIEmbeddingFunction(
                                    api_key=api_key,
                                    model_name="text-embedding-ada-002"
                                )
                            except Exception as e:
                                console.print(f"[yellow]⚠ Could not initialize OpenAI embedding: {e}[/yellow]")
                                self.embedding_function = None
                
                # Create collection with or without embedding function
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function,
                    metadata={"hnsw:space": "cosine"}
                )
                console.print(f"[green]✓ ChromaDB collection '{self.collection_name}' created[/green]")
        except ImportError:
            console.print("[yellow]⚠ ChromaDB not installed. Using fallback memory mechanism.[/yellow]")
            console.print("[yellow]To enable vector memory: pip install chromadb[/yellow]")
            self._setup_fallback()
        except Exception as e:
            console.print(f"[red]❌ ChromaDB initialization error: {e}. Using fallback.[/red]")
            self._setup_fallback()

    def _setup_fallback(self):
        """Set up fallback memory if ChromaDB isn't available."""
        self.using_fallback = True
        self.fallback_memory = {}
        
    def _get_embedding_from_gpt(self, text: str) -> List[float]:
        """Get embedding using GPTManager if available."""
        if self.gpt_manager is None:
            # Fallback to simple hash-based embedding
            import hashlib
            hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
            np.random.seed(hash_val)
            return list(np.random.normal(0, 1, 384))  # 384-dim random vector
            
        try:
            # Use GPTManager to get embedding
            prompt = f"Encode this text as a semantic vector: {text}"
            response = self.gpt_manager.gpt_request(
                prompt=prompt,
                task_type="embedding",
                model="gpt-5-nano"
            )
            
            try:
                # Try to parse as JSON
                import json
                vector = json.loads(response)
                if isinstance(vector, list) and len(vector) > 0:
                    return vector
            except:
                pass
                
            # Fallback: hash-based embedding
            import hashlib
            hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
            np.random.seed(hash_val)
            return list(np.random.normal(0, 1, 384))
        except Exception as e:
            console.print(f"[yellow]⚠ Error generating embedding: {e}[/yellow]")
            # Fallback embedding
            import hashlib
            hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
            np.random.seed(hash_val)
            return list(np.random.normal(0, 1, 384))

    def add_experience(
        self,
        agent_id: str,
        command: str,
        state: Dict[str, Any],
        reward: float,
        output: str,
        metadata: Optional[Dict[str, Any]] = None,
        embeddings: Optional[List[float]] = None
    ) -> str:
        """
        Add an agent experience to the vector store.
        
        Args:
            agent_id: The ID of the agent
            command: The command executed
            state: The environment state
            reward: The reward received
            output: Command output
            metadata: Additional metadata
            embeddings: Optional pre-computed embeddings
            
        Returns:
            The ID of the stored experience
        """
        if hasattr(self, 'using_fallback') and self.using_fallback:
            return self._fallback_add_experience(agent_id, command, state, reward, output, metadata)
            
        try:
            # Generate ID for the experience
            doc_id = f"{agent_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            
            # Prepare document
            document = f"Command: {command}\nOutput: {output}\nReward: {reward}"
            if isinstance(state, dict):
                state_str = "\n".join([f"{k}: {v}" for k, v in state.items() if k in ["phase", "privilege_level", "open_ports", "blue_team_alert"]])
                document += f"\nState: {state_str}"
                
            # Prepare metadata
            if metadata is None:
                metadata = {}
            
            full_metadata = {
                "agent_id": agent_id,
                "command": command,
                "reward": float(reward),
                "timestamp": time.time(),
                "phase": state.get("phase", "unknown") if isinstance(state, dict) else "unknown",
                **metadata
            }
            
            # Add to collection
            self.collection.add(
                ids=[doc_id],
                documents=[document],
                metadatas=[full_metadata],
                embeddings=[embeddings] if embeddings is not None else None
            )
            
            return doc_id
        except Exception as e:
            console.print(f"[red]❌ Error adding experience: {e}[/red]")
            return self._fallback_add_experience(agent_id, command, state, reward, output, metadata)
            
    def _fallback_add_experience(self, agent_id, command, state, reward, output, metadata):
        """Fallback method for adding experiences when ChromaDB is unavailable."""
        doc_id = f"{agent_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        self.fallback_memory[doc_id] = {
            "agent_id": agent_id,
            "command": command,
            "state": state,
            "reward": reward,
            "output": output,
            "metadata": metadata,
            "timestamp": time.time()
        }
        return doc_id
            
    def search_similar_commands(
        self,
        query_command: str,
        agent_id: Optional[str] = None,
        n_results: int = 5,
        min_similarity: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search for similar commands in the vector store.
        
        Args:
            query_command: The command to search for
            agent_id: Optional filter by agent ID
            n_results: Number of results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of similar command records
        """
        if hasattr(self, 'using_fallback') and self.using_fallback:
            return self._fallback_search_commands(query_command, agent_id, n_results)
            
        try:
            # Prepare metadata filter
            where = {"command": {"$exists": True}}
            if agent_id:
                where["agent_id"] = agent_id
                
            # Search collection
            results = self.collection.query(
                query_texts=[query_command],
                n_results=n_results,
                where=where
            )
            
            # Format results
            formatted_results = []
            for i, doc_id in enumerate(results.get('ids', [[]])[0]):
                if i >= len(results.get('distances', [[]]))[0]:
                    continue
                    
                distance = results['distances'][0][i]
                similarity = 1.0 - distance  # Convert distance to similarity
                
                if similarity < min_similarity:
                    continue
                    
                metadata = results['metadatas'][0][i] if results.get('metadatas') else {}
                document = results['documents'][0][i] if results.get('documents') else ""
                
                formatted_results.append({
                    "id": doc_id,
                    "command": metadata.get("command", ""),
                    "similarity": similarity,
                    "reward": metadata.get("reward", 0.0),
                    "agent_id": metadata.get("agent_id", ""),
                    "phase": metadata.get("phase", ""),
                    "document": document
                })
                
            return formatted_results
        except Exception as e:
            console.print(f"[red]❌ Error searching commands: {e}[/red]")
            return self._fallback_search_commands(query_command, agent_id, n_results)
            
    def _fallback_search_commands(self, query_command, agent_id, n_results):
        """Fallback search when ChromaDB is unavailable."""
        results = []
        
        # Basic string similarity for fallback
        def string_similarity(a, b):
            from difflib import SequenceMatcher
            return SequenceMatcher(None, a, b).ratio()
        
        for doc_id, data in self.fallback_memory.items():
            if agent_id and data.get("agent_id") != agent_id:
                continue
                
            command = data.get("command", "")
            if not command:
                continue
                
            similarity = string_similarity(query_command, command)
            
            results.append({
                "id": doc_id,
                "command": command,
                "similarity": similarity,
                "reward": data.get("reward", 0.0),
                "agent_id": data.get("agent_id", ""),
                "phase": data.get("state", {}).get("phase", "")
            })
        
        # Sort by similarity descending
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:n_results]
            
    def get_high_reward_experiences(
        self,
        agent_id: Optional[str] = None,
        phase: Optional[str] = None,
        n_results: int = 10,
        min_reward: float = 5.0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve high-reward experiences from the vector store.
        
        Args:
            agent_id: Optional filter by agent ID
            phase: Optional filter by phase
            n_results: Number of results to return
            min_reward: Minimum reward threshold
            
        Returns:
            List of high-reward experiences
        """
        if hasattr(self, 'using_fallback') and self.using_fallback:
            return self._fallback_get_high_reward(agent_id, phase, n_results, min_reward)
            
        try:
            # Prepare metadata filter
            where = {"reward": {"$gte": min_reward}}
            if agent_id:
                where["agent_id"] = agent_id
            if phase:
                where["phase"] = phase
                
            # Query collection with ordering
            results = self.collection.query(
                query_texts=None,
                n_results=n_results,
                where=where
            )
            
            # Format results
            formatted_results = []
            for i, doc_id in enumerate(results.get('ids', [[]])[0]):
                metadata = results['metadatas'][0][i] if results.get('metadatas') else {}
                document = results['documents'][0][i] if results.get('documents') else ""
                
                formatted_results.append({
                    "id": doc_id,
                    "command": metadata.get("command", ""),
                    "reward": metadata.get("reward", 0.0),
                    "agent_id": metadata.get("agent_id", ""),
                    "phase": metadata.get("phase", ""),
                    "document": document
                })
                
            # Sort by reward descending
            formatted_results.sort(key=lambda x: x["reward"], reverse=True)
            return formatted_results
        except Exception as e:
            console.print(f"[red]❌ Error retrieving high-reward experiences: {e}[/red]")
            return self._fallback_get_high_reward(agent_id, phase, n_results, min_reward)
            
    def _fallback_get_high_reward(self, agent_id, phase, n_results, min_reward):
        """Fallback high-reward retrieval when ChromaDB is unavailable."""
        results = []
        
        for doc_id, data in self.fallback_memory.items():
            reward = data.get("reward", 0.0)
            if reward < min_reward:
                continue
                
            if agent_id and data.get("agent_id") != agent_id:
                continue
                
            if phase and data.get("state", {}).get("phase") != phase:
                continue
                
            results.append({
                "id": doc_id,
                "command": data.get("command", ""),
                "reward": reward,
                "agent_id": data.get("agent_id", ""),
                "phase": data.get("state", {}).get("phase", ""),
                "output": data.get("output", "")
            })
        
        # Sort by reward descending
        results.sort(key=lambda x: x["reward"], reverse=True)
        return results[:n_results]
            
    def check_command_redundancy(
        self,
        command: str,
        agent_id: Optional[str] = None,
        threshold: float = 0.85
    ) -> bool:
        """
        Check if a command is redundant based on similarity to existing commands.
        
        Args:
            command: The command to check
            agent_id: Optional filter by agent ID
            threshold: Similarity threshold for redundancy
            
        Returns:
            True if redundant, False otherwise
        """
        similar_commands = self.search_similar_commands(
            query_command=command,
            agent_id=agent_id,
            n_results=1,
            min_similarity=threshold
        )
        
        return len(similar_commands) > 0
            
    def get_agent_summary(self, agent_id: str) -> Dict[str, Any]:
        """
        Generate a summary of agent's experiences.
        
        Args:
            agent_id: The agent ID
            
        Returns:
            Summary statistics dictionary
        """
        if hasattr(self, 'using_fallback') and self.using_fallback:
            return self._fallback_agent_summary(agent_id)
            
        try:
            # Count agent's experiences
            where = {"agent_id": agent_id}
            count = self.collection.count(where=where)
            
            # Get average reward
            results = self.collection.query(
                query_texts=None,
                n_results=count,
                where=where
            )
            
            rewards = [metadata.get("reward", 0.0) for metadata in results.get('metadatas', [[]])[0]]
            avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
            
            # Count by phase
            phases = {}
            for metadata in results.get('metadatas', [[]])[0]:
                phase = metadata.get("phase", "unknown")
                phases[phase] = phases.get(phase, 0) + 1
                
            return {
                "agent_id": agent_id,
                "total_experiences": count,
                "average_reward": avg_reward,
                "phases": phases,
                "total_positive_rewards": sum(1 for r in rewards if r > 0),
                "total_negative_rewards": sum(1 for r in rewards if r < 0)
            }
        except Exception as e:
            console.print(f"[red]❌ Error generating agent summary: {e}[/red]")
            return self._fallback_agent_summary(agent_id)
            
    def _fallback_agent_summary(self, agent_id):
        """Fallback agent summary when ChromaDB is unavailable."""
        agent_data = [data for doc_id, data in self.fallback_memory.items() 
                     if data.get("agent_id") == agent_id]
        
        rewards = [data.get("reward", 0.0) for data in agent_data]
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        
        phases = {}
        for data in agent_data:
            phase = data.get("state", {}).get("phase", "unknown")
            phases[phase] = phases.get(phase, 0) + 1
            
        return {
            "agent_id": agent_id,
            "total_experiences": len(agent_data),
            "average_reward": avg_reward,
            "phases": phases,
            "total_positive_rewards": sum(1 for r in rewards if r > 0),
            "total_negative_rewards": sum(1 for r in rewards if r < 0)
        }
            
    def export_to_jsonl(self, output_path: str, agent_id: Optional[str] = None):
        """
        Export vector store contents to JSONL file.
        
        Args:
            output_path: Path to output file
            agent_id: Optional filter by agent ID
        """
        if hasattr(self, 'using_fallback') and self.using_fallback:
            return self._fallback_export(output_path, agent_id)
            
        try:
            # Prepare query
            where = {}
            if agent_id:
                where["agent_id"] = agent_id
                
            # Get all matching records
            results = self.collection.query(
                query_texts=None,
                n_results=self.collection.count(where=where),
                where=where
            )
            
            # Write to JSONL
            with open(output_path, 'w') as f:
                for i, doc_id in enumerate(results.get('ids', [[]])[0]):
                    metadata = results['metadatas'][0][i] if results.get('metadatas') else {}
                    document = results['documents'][0][i] if results.get('documents') else ""
                    
                    record = {
                        "id": doc_id,
                        "document": document,
                        **metadata
                    }
                    
                    f.write(json.dumps(record) + "\n")
                    
            console.print(f"[green]✓ Exported vector store to {output_path}[/green]")
        except Exception as e:
            console.print(f"[red]❌ Error exporting to JSONL: {e}[/red]")
            self._fallback_export(output_path, agent_id)
            
    def _fallback_export(self, output_path, agent_id):
        """Fallback export when ChromaDB is unavailable."""
        try:
            with open(output_path, 'w') as f:
                for doc_id, data in self.fallback_memory.items():
                    if agent_id and data.get("agent_id") != agent_id:
                        continue
                    record = {"id": doc_id, **data}
                    f.write(json.dumps(record) + "\n")
            console.print(f"[green]✓ Exported fallback memory to {output_path}[/green]")
        except Exception as e:
            console.print(f"[red]❌ Error exporting fallback memory: {e}[/red]")

# For CLI testing
if __name__ == "__main__":
    console.print("[bold blue]Testing ChromaMemoryStore...[/bold blue]")
    
    # Import GPTManager for embedding
    from core.gpt_manager import GPTManager
    gpt_manager = GPTManager.get_instance()
    
    store = ChromaMemoryStore(gpt_manager=gpt_manager)
    
    # Test adding experience
    console.print("[yellow]Adding test experience...[/yellow]")
    doc_id = store.add_experience(
        agent_id="RedAgent",
        command="nmap -sT -sV 10.10.10.10",
        state={"phase": "recon", "privilege_level": "none", "open_ports": [22, 80]},
        reward=10.0,
        output="22/tcp open ssh\n80/tcp open http"
    )
    
    console.print(f"[green]Added document with ID: {doc_id}[/green]")
    
    # Test searching similar commands
    console.print("[yellow]Testing similarity search...[/yellow]")
    similar = store.search_similar_commands("nmap -sV 10.10.10.10")
    console.print(f"[green]Found {len(similar)} similar commands[/green]")
    
    # Test redundancy check
    is_redundant = store.check_command_redundancy("nmap -sT -sV 10.10.10.10")
    console.print(f"[green]Redundancy check: {is_redundant}[/green]")
    
    # Test high-reward experiences
    high_reward = store.get_high_reward_experiences(min_reward=5.0)
    console.print(f"[green]Found {len(high_reward)} high-reward experiences[/green]")
    
    # Test agent summary
    summary = store.get_agent_summary("RedAgent")
    console.print(f"[green]Agent summary: {summary}[/green]")
