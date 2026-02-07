# core/logic/chainbuilder.py — ARIASKA Chain Builder v12.0 APEX
# ⚡ Attack Chain Construction | 🔗 Cross-Agent Coordination | 🧠 AI-Driven Chain Synthesis

import os
import json
import asyncio
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from core.utils.llm_router import LLMRouter
from core.vector_search import VectorSearch

console = Console()

CHAIN_CACHE_DIR = "data/chains"
os.makedirs(CHAIN_CACHE_DIR, exist_ok=True)

class ChainCache:
    """Persistent cache for generated chains, keyed by agent and memory hash."""
    def __init__(self, cache_dir=CHAIN_CACHE_DIR):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_index = self._load_index()

    def _index_path(self):
        return os.path.join(self.cache_dir, "chain_index.json")

    def _load_index(self):
        path = self._index_path()
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_index(self):
        with open(self._index_path(), "w") as f:
            json.dump(self.cache_index, f, indent=2)

    def get(self, agent_name, memory_hash):
        key = f"{agent_name}_{memory_hash}"
        fname = self.cache_index.get(key)
        if fname and os.path.exists(os.path.join(self.cache_dir, fname)):
            with open(os.path.join(self.cache_dir, fname), "r") as f:
                return json.load(f)
        return None

    def set(self, agent_name, memory_hash, chain_data):
        key = f"{agent_name}_{memory_hash}"
        fname = f"chain_{agent_name}_{memory_hash[:8]}.json"
        with open(os.path.join(self.cache_dir, fname), "w") as f:
            json.dump(chain_data, f, indent=2)
        self.cache_index[key] = fname
        self._save_index()

class ChainGenerator:
    """
    Core chain generation service.
    Consumes agent memory/actions and produces a chain (list of commands) using LLMRouter.
    Handles caching, retries, and fallback logic.
    """
    def __init__(self, llm_router=None, vector_search=None, cache=None, verbosity="standard", memory_router=None):
        self.llm_router = llm_router or LLMRouter()
        self.vector_search = vector_search or VectorSearch()
        self.cache = cache or ChainCache()
        self.verbosity = verbosity
        self.memory_router = memory_router  # Store memory_router for future use

    def _memory_hash(self, actions: List[Dict[str, Any]]) -> str:
        # Hash the top-N actions (by reward) for cache key
        actions_str = json.dumps(sorted(actions, key=lambda x: -x.get("reward", 0))[:12], sort_keys=True)
        return hashlib.sha256(actions_str.encode()).hexdigest()

    async def generate_chain(self, agent_memory: Dict[str, Any], agent_name: str, top_n: int = 12, use_orion_strategy: bool = False) -> List[str]:
        """
        Generate a chain for the agent using LLMRouter, with persistent caching and fallback.
        Returns a list of commands.
        """
        actions = agent_memory.get("actions", [])
        if not actions:
            console.print(f"[red]⚠ {agent_name}: No actions in memory. Skipping chain generation.[/red]")
            return []
        sorted_actions = sorted(actions, key=lambda x: x.get("reward", 0), reverse=True)[:top_n]
        memory_hash = self._memory_hash(sorted_actions)
        cached = self.cache.get(agent_name, memory_hash)
        if cached:
            if self.verbosity != "silent":
                console.print(f"[yellow]⚡ Cached chain loaded for {agent_name}[/yellow]")
            return cached.get("chain", [])
        # VectorSearch: try to find similar context and reuse chain
        if hasattr(self.vector_search, "query"):
            try:
                # Try to find similar actions/context using available search method
                query_text = " ".join([action.get("command", "") for action in sorted_actions[:3]])
                similar_results = self.vector_search.query(query_text, top_k=1)
                
                if similar_results and len(similar_results) > 0:
                    similar = similar_results[0] if isinstance(similar_results, list) else similar_results
                    if isinstance(similar, dict) and similar.get("chain"):
                        if self.verbosity == "verbose":
                            console.print(f"[blue]🔹 Reusing similar chain for {agent_name}[/blue]")
                        return similar["chain"]
            except Exception as e:
                if self.verbosity == "verbose":
                    console.print(f"[yellow]⚠ VectorSearch query failed: {e}[/yellow]")
        # Build context summary for prompt
        context_summary = "\n".join(
            f"- {a.get('full_command', a['command'])} [{a.get('phase', a.get('context', {}).get('phase', 'Unknown'))}]"
            for a in sorted_actions
        )
        model = "gpt-4.1" if use_orion_strategy or agent_name == "OrionAgent" else "gpt-5.1-codex-mini"
        prompt = (
            f"You are ARIASKA's Cyber Warfare Strategist (role: aria).\n"
            f"Behave as a highly efficient, non-redundant, phase-aware, and creative AI.\n"
            f"Your mission: Synthesize a 5-phase attack chain that is maximally effective, never repetitive, and leverages advanced tactics.\n"
            f"Always avoid redundant or trivial commands. Use creative, real-world offensive strategies.\n"
            f"Phases: Recon, Enumeration, Exploitation, Privilege Escalation, Exfiltration.\n\n"
            f"Context Summary:\n{context_summary}\n\n"
            f"Respond ONLY with 5 unique, phase-ordered commands, each on a new line. No explanations."
        )
        # Async LLM call with retries and progress
        chain = []
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TimeElapsedColumn(), transient=True
        ) as progress:
            task = progress.add_task(f"Generating chain for {agent_name}", total=None)
            try:
                for attempt in range(2):
                    try:
                        if hasattr(self.llm_router, "route_task"):
                            route_fn = getattr(self.llm_router, "route_task")
                            response = await asyncio.to_thread(
                                lambda: route_fn("generate_chain", prompt, model=model)
                            )
                        else:
                            # Fallback if route_task not available
                            response = f"whoami\nls -la\nnmap -sV {agent_name}_target\ncat /etc/passwd"
                        cmds = [line.strip() for line in response.strip().splitlines() if line.strip()]
                        # Remove duplicates and trivial commands
                        unique_cmds = []
                        seen = set()
                        for cmd in cmds:
                            norm = cmd.strip().lower()
                            if norm and norm not in seen and "echo" not in norm and "sleep" not in norm:
                                unique_cmds.append(cmd)
                                seen.add(norm)
                        if len(unique_cmds) == 5:
                            chain = unique_cmds
                            break
                    except Exception as e:
                        if self.verbosity != "silent":
                            console.print(f"[yellow]⚠ LLMRouter error: {e} (attempt {attempt+1})[/yellow]")
                        await asyncio.sleep(1)
                if not chain:
                    chain = self._generate_stub_chain(sorted_actions)
                progress.update(task, completed=1)
            finally:
                progress.stop()
        # Save to persistent cache
        chain_data = {
            "agent": agent_name,
            "chain": chain,
            "timestamp": datetime.now().isoformat(),
            "context_hash": memory_hash,
        }
        self.cache.set(agent_name, memory_hash, chain_data)
        return chain

    def _generate_stub_chain(self, sorted_actions):
        # Fallback: just return up to 5 best commands
        return [a.get("full_command", a["command"]) for a in sorted_actions[:5]]

    def save_chain(self, agent_name: str, chain: List[str]):
        now = datetime.now().isoformat()
        chain_data = [
            {
                "phase": ["recon", "enumeration", "exploit", "privesc", "exfiltrate"][i] if i < 5 else "unknown",
                "command": cmd,
                "confidence_score": 0.95,
                "trigger_reason": "Strategic chain generation",
                "timestamp": now,
                "agent": agent_name
            }
            for i, cmd in enumerate(chain)
        ]
        path = os.path.join(CHAIN_CACHE_DIR, f"chain_{agent_name.lower()}_{now.replace(':','-')}.json")
        try:
            with open(path, "w") as f:
                json.dump({"agent": agent_name, "chain": chain_data}, f, indent=2)
            console.print(f"[green][Chain] {agent_name}: Chain saved to {path}[/green]")
        except Exception as e:
            console.print(f"[red]❌ Failed to save chain for {agent_name}: {e}[/red]")

    def diagnostic_summary(self):
        """
        Display a summary of all generated chains.
        """
        console.rule("[bold green]📊 ChainBuilder Diagnostic Summary")
        chains = [f for f in os.listdir(self.cache.cache_dir) if f.startswith("chain_")]
        for chain_file in chains:
            path = os.path.join(self.cache.cache_dir, chain_file)
            with open(path, "r") as f:
                data = json.load(f)
                agent = data.get("agent", "Unknown")
                cmds = data.get("chain", [])
                console.print(
                    f"[cyan]{agent}:[/cyan] {len(cmds)} commands synthesized."
                )
        console.print(
            f"[blue]Total Chains:[/blue] {len(chains)} | [yellow]Location:[/yellow] {self.cache.cache_dir}"
        )
    
    def build_and_store_chain_multiagent(self, agent_manager):
        """Build and store chains for all agents using multi-agent approach."""
        try:
            console.print("[magenta]🔗 Generating attack chains for all agents...[/magenta]")
            
            # Get all agents from agent manager
            agents = agent_manager.all_agents() if hasattr(agent_manager, 'all_agents') else []
            
            if not agents:
                console.print("[yellow]⚠ No agents available for chain generation[/yellow]")
                return
            
            # Generate chains for each agent
            for agent in agents:
                try:
                    # Get agent memory
                    memory = getattr(agent, 'get_memory_for_chain', lambda: {"actions": []})()
                    
                    # Use asyncio to generate chain
                    import asyncio
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # If loop is already running, use create_task
                        chain = []
                    else:
                        chain = loop.run_until_complete(
                            self.generate_chain(memory, agent.agent_id, use_orion_strategy=(agent.agent_id == "OrionAgent"))
                        )
                    
                    # Save the chain
                    if chain:
                        self.save_chain(agent.agent_id, chain)
                        console.print(f"[green]✓ Chain generated for {agent.agent_id}: {len(chain)} commands[/green]")
                    else:
                        console.print(f"[yellow]⚠ No chain generated for {agent.agent_id}[/yellow]")
                        
                except Exception as e:
                    console.print(f"[red]❌ Chain generation failed for {getattr(agent, 'agent_id', 'Unknown')}: {e}[/red]")
                    
        except Exception as e:
            console.print(f"[red]❌ Multi-agent chain generation failed: {e}[/red]")

# === Integration API ===

async def generate_and_save_chain(agent_memory, agent_name, top_n=12, use_orion_strategy=False, verbosity="standard"):
    """
    High-level API: Generate a chain and persist it, returning the chain.
    """
    generator = ChainGenerator(verbosity=verbosity)
    chain = await generator.generate_chain(agent_memory, agent_name, top_n, use_orion_strategy)
    generator.save_chain(agent_name, chain)
    return chain

def build_and_store_chain_multiagent(agent_manager, dry_run=False, verbosity="standard"):
    """
    Build and store chains for all agents in agent_manager.
    Can be called from the trainer or CLI.
    """
    generator = ChainGenerator(verbosity=verbosity)
    agents = agent_manager.all_agents() if hasattr(agent_manager, "all_agents") else []
    loop = asyncio.get_event_loop()
    tasks = []
    for agent in agents:
        mem = agent.memory_manager.memory if hasattr(agent, "memory_manager") else {}
        use_orion = agent.agent_id == "OrionAgent"
        tasks.append(generator.generate_chain(mem, agent.agent_id, use_orion_strategy=use_orion))
    chains = loop.run_until_complete(asyncio.gather(*tasks))
    for agent, chain in zip(agents, chains):
        if not dry_run:
            generator.save_chain(agent.agent_id, chain)
        if verbosity != "silent":
            summary = ", ".join(chain)
            console.print(f"[cyan]{agent.agent_id} Chain:[/cyan] {summary}")
    if verbosity != "silent":
        console.print("[blue]👁 OrionAgent: Chains synchronized. Strategic layer updated.[/blue]")

# === CLI/Standalone Diagnostic ===
if __name__ == "__main__":
    from core.multiagent.agent_manager import AgentManager
    from core.multiagent.memory_router import MemoryRouter
    console.rule("[bold magenta]⚡ ARIASKA ChainBuilder v12.0 — Standalone Mode")
    manager = AgentManager()
    memory_router = MemoryRouter()
    build_and_store_chain_multiagent(manager)
    ChainGenerator().diagnostic_summary()
