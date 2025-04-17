import os
import json
import asyncio
import glob
import csv
from rich.console import Console
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from core.teach import TeachModule

console = Console()

############################################
# Paths & Global Setup
############################################
KNOWLEDGE_SOURCES_FOLDER = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "knowledge_sources")
)

# SentenceTransformer Embedder (Consistent Model)
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ChromaDB PersistentClient Initialization
try:
    chroma_client = chromadb.PersistentClient(path="./chroma_storage")
    kb_collection = chroma_client.get_or_create_collection("ariaska_kb")
    console.print("[green]✔ ChromaDB PersistentClient initialized![/green]")
except Exception as e:
    console.print(f"[red]❌ ChromaDB PersistentClient failed: {e}[/red]")
    chroma_client = None
    kb_collection = None

# Initialize TeachModule (Handles memory.json + actions/scenarios)
teach = TeachModule()

############################################
# Parser Functions for Different File Types
############################################
async def parse_json(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)

    actions = data.get("actions", [])
    console.print(f"[cyan]✔ Parsed {len(actions)} actions from {os.path.basename(filepath)}[/cyan]")
    return actions

async def parse_pdf(filepath):
    try:
        reader = PdfReader(filepath)
        text = "\n".join([
            page.extract_text() for page in reader.pages if page.extract_text()
        ])
    except Exception as e:
        console.print(f"[red]❌ PDF Parse Error ({filepath}): {e}[/red]")
        return []

    commands = []
    for line in text.split("\n"):
        command = line.strip()
        if not command:
            continue

        commands.append({
            "command": command.split()[0] if command.split() else command,
            "full_command": command,
            "description": "Imported from PDF - needs review",
            "tools": [command.split()[0]] if command.split() else [],
            "parameters": [],
            "param_descriptions": [],
            "when": "Manual review pending",
            "why": "Imported via PDF",
            "phase": "Recon",
            "reward": 50
        })

    console.print(f"[cyan]✔ Extracted {len(commands)} commands from PDF {os.path.basename(filepath)}[/cyan]")
    return commands

async def parse_txt(filepath):
    commands = []
    with open(filepath, "r") as f:
        for line in f:
            command = line.strip()
            if not command:
                continue

            commands.append({
                "command": command.split()[0] if command.split() else command,
                "full_command": command,
                "description": "Imported from TXT - needs review",
                "tools": [command.split()[0]] if command.split() else [],
                "parameters": [],
                "param_descriptions": [],
                "when": "Manual review pending",
                "why": "Imported via TXT",
                "phase": "Recon",
                "reward": 50
            })

    console.print(f"[cyan]✔ Extracted {len(commands)} commands from TXT {os.path.basename(filepath)}[/cyan]")
    return commands

async def parse_csv(filepath):
    commands = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            command = row.get("command", "").strip()
            if not command:
                continue

            commands.append({
                "command": command.split()[0] if command.split() else command,
                "full_command": command,
                "description": row.get("description", "Imported from CSV - needs review"),
                "tools": [command.split()[0]] if command.split() else [],
                "parameters": [],
                "param_descriptions": [],
                "when": row.get("when", "Manual review pending"),
                "why": row.get("why", "Imported via CSV"),
                "phase": row.get("phase", "Recon"),
                "reward": int(row.get("reward", 50))
            })

    console.print(f"[cyan]✔ Extracted {len(commands)} commands from CSV {os.path.basename(filepath)}[/cyan]")
    return commands

############################################
# MAIN Async Function to Run Import Process
############################################
async def main():
    console.print("[bold magenta]🚀 Starting Knowledge Import...[/bold magenta]")

    if not os.path.exists(KNOWLEDGE_SOURCES_FOLDER):
        console.print(f"[red]❌ Folder not found: {KNOWLEDGE_SOURCES_FOLDER}[/red]")
        return

    files = glob.glob(os.path.join(KNOWLEDGE_SOURCES_FOLDER, "*"))
    if not files:
        console.print(f"[yellow]⚠ No files found in {KNOWLEDGE_SOURCES_FOLDER}[/yellow]")
        return

    total_actions = 0

    for file in files:
        extension = os.path.splitext(file)[-1].lower()

        if extension == ".json":
            console.print(f"→ Parsing JSON: {file}")
            actions = await parse_json(file)
        elif extension == ".pdf":
            console.print(f"→ Parsing PDF: {file}")
            actions = await parse_pdf(file)
        elif extension == ".txt":
            console.print(f"→ Parsing TXT: {file}")
            actions = await parse_txt(file)
        elif extension == ".csv":
            console.print(f"→ Parsing CSV: {file}")
            actions = await parse_csv(file)
        else:
            console.print(f"[red]✖ Unsupported file type: {file}[/red]")
            continue

        if actions:
            total_actions += len(actions)

            # TeachModule loads it to memory.json & FAISS
            teach.bulk_add_actions(actions)

            # Add to Chroma Vector DB (Batch Add)
            try:
                vectors = embedder.encode(
                    [a["full_command"] for a in actions]
                ).tolist()

                kb_collection.add(
                    documents=[a["full_command"] for a in actions],
                    embeddings=vectors,
                    metadatas=[{
                        "description": a.get("description", "N/A"),
                        "phase": a.get("phase", "Recon")
                    } for a in actions]
                )

                console.print(f"[green]✔ {len(actions)} actions added to ChromaDB.[/green]")
            except Exception as e:
                console.print(f"[red]❌ ChromaDB Bulk Add Failed: {e}[/red]")

    if total_actions == 0:
        console.print("[yellow]⚠ No new actions were added.[/yellow]")
    else:
        console.print(f"[bold green]✔ Knowledge Import Complete. {total_actions} actions processed![/bold green]")

    # Persist Chroma Storage
    if chroma_client:
        try:
            chroma_client.persist()
            console.print("[green]✔ ChromaDB storage persisted.[/green]")
        except Exception as e:
            console.print(f"[red]❌ ChromaDB Persist Failed: {e}[/red]")

############################################
# Script Entry Point
############################################
if __name__ == "__main__":
    asyncio.run(main())
