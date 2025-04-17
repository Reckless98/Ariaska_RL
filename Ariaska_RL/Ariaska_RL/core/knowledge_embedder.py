# 📁 /core/knowledge_embedder.py

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from rich.console import Console

console = Console()

class KnowledgeEmbedder:
    def __init__(self, embed_dim=384, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        console.print(f"[cyan]🔧 Initializing Knowledge Embedder on {self.device}...[/cyan]")
        
        try:
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=self.device)
            console.print("[green]✔ SentenceTransformer loaded for KnowledgeEmbedder.[/green]")
        except Exception as e:
            console.print(f"[red]❌ Failed to load SentenceTransformer: {e}[/red]")
            raise e

        self.index = faiss.IndexFlatL2(embed_dim)
        self.texts = []
        self.embed_dim = embed_dim

    def add_documents(self, docs):
        if not docs:
            console.print("[yellow]⚠ No documents provided to add_documents.[/yellow]")
            return
        
        console.print(f"[cyan]➕ Adding {len(docs)} documents to Knowledge Embedder...[/cyan]")

        embeddings = self.model.encode(docs, convert_to_numpy=True)

        if embeddings.shape[1] != self.embed_dim:
            raise ValueError(f"Embedding dimension mismatch! Expected {self.embed_dim}, got {embeddings.shape[1]}.")

        self.index.add(embeddings)
        self.texts.extend(docs)

        console.print(f"[green]✔ {len(docs)} documents embedded and indexed.[/green]")

    def query(self, query_text, top_k=5):
        if not self.texts:
            console.print("[red]❌ No documents indexed yet in Knowledge Embedder![/red]")
            return []
        
        console.print(f"[cyan]🔎 Querying for: '{query_text}'[/cyan]")

        query_embedding = self.model.encode([query_text], convert_to_numpy=True)

        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.texts):
                results.append(self.texts[idx])

        console.print(f"[green]✔ Query returned {len(results)} results.[/green]")
        return results

    def save_index(self, index_file="./faiss_knowledge.index", meta_file="./faiss_knowledge.json"):
        faiss.write_index(self.index, index_file)

        with open(meta_file, 'w') as f:
            json.dump(self.texts, f, indent=4)

        console.print(f"[green]✔ Knowledge index and metadata saved.[/green]")

    def load_index(self, index_file="./faiss_knowledge.index", meta_file="./faiss_knowledge.json"):
        if os.path.exists(index_file):
            self.index = faiss.read_index(index_file)
            console.print("[green]✔ FAISS index loaded from disk.[/green]")
        else:
            console.print("[red]❌ Index file not found![/red]")

        if os.path.exists(meta_file):
            with open(meta_file, 'r') as f:
                self.texts = json.load(f)
            console.print(f"[green]✔ Metadata loaded. {len(self.texts)} documents available.[/green]")
        else:
            console.print("[red]❌ Metadata file not found![/red]")
