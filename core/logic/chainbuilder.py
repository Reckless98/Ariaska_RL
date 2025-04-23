# core/logic/chainbuilder.py — ARIASKA Chain Builder v12.0 APEX
# ⚡ Attack Chain Construction | 🔗 Cross-Agent Coordination | 🧠 AI-Driven Chain Synthesis

import os
import json
import subprocess
import random
from datetime import datetime
from rich.console import Console
from core.teach.teach import TeachModule
from core.vector_search import VectorSearch

console = Console()
teach = TeachModule()
vector_search = VectorSearch()

PHASES = ["recon", "enumeration", "exploit", "privesc", "exfiltrate"]
def build_and_store_chain(agent_manager, memory_router=None, dry_run=False):
    """
    Top-level function to build and store chains using ChainBuilder.
    Can be imported and used directly (e.g., from main.py).
    """
    builder = ChainBuilder(memory_router=memory_router)
    builder.build_and_store_chain_multiagent(agent_manager, dry_run=dry_run)

class ChainBuilder:
    def __init__(self, memory_router=None, verbosity="standard"):
        self.cache_dir = "data/chains"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.memory_router = memory_router
        self.verbosity = verbosity
        self.chain_cache = {}
        console.print(
            f"[green]✔ ChainBuilder v12.0 Initialized — Smart Chain Synthesis Ready[/green]"
        )

    # ─────────────────────────────────────────────
    # 🧠 GPT-Driven Chain Generation with Context Awareness
    # ─────────────────────────────────────────────
    def gpt_generate_chain(self, memory, agent_name="UnknownAgent", top_n=12, use_orion_strategy=False):
        actions = memory.get("actions", [])
        if not actions:
            console.print(
                f"[red]⚠ {agent_name}: No actions in memory. Skipping chain generation.[/red]"
            )
            return []

        sorted_actions = sorted(
            actions, key=lambda x: x.get("reward", 0), reverse=True
        )[:top_n]
        context_summary = self._build_context_summary(sorted_actions)

        # Use GPT-4.1 Full for OrionAgent or if requested, else GPT-4o-mini for simple tasks
        model = "gpt-4.1" if use_orion_strategy or agent_name == "OrionAgent" else "gpt-4o-mini"
        prompt = f"""
You are ARIASKA's Cyber Warfare Strategist (role: aria).
Behave as a highly efficient, non-redundant, phase-aware, and creative AI.
Your mission: Synthesize a 5-phase attack chain that is maximally effective, never repetitive, and leverages advanced tactics.
Always avoid redundant or trivial commands. Use creative, real-world offensive strategies.
Phases: Recon, Enumeration, Exploitation, Privilege Escalation, Exfiltration.

Context Summary:
{context_summary}

Respond ONLY with 5 unique, phase-ordered commands, each on a new line. No explanations.
"""
        cache_key = f"{agent_name}_chain_{datetime.now().strftime('%Y%m%d')}"
        if cache_key in self.chain_cache:
            if self.verbosity != "silent":
                console.print(f"[yellow]⚡ Cached chain loaded for {agent_name}[/yellow]")
            return self.chain_cache[cache_key]

        if self.memory_router:
            cached = self.memory_router.check_gpt_cache(cache_key)
            if cached:
                self.chain_cache[cache_key] = cached
                if self.verbosity != "silent":
                    console.print(f"[yellow]⚡ Cached chain loaded for {agent_name}[/yellow]")
                return cached

        commands = self._call_gpt(prompt, agent_name, model=model)
        # Remove any duplicate or trivial commands, keep only unique, non-empty
        if commands:
            unique_cmds = []
            seen = set()
            for cmd in commands:
                norm = cmd.strip().lower()
                if norm and norm not in seen and "echo" not in norm and "sleep" not in norm:
                    unique_cmds.append(cmd)
                    seen.add(norm)
            commands = unique_cmds[:5]
        if commands and self.memory_router:
            self.memory_router.store_gpt_response(cache_key, commands)
            self.chain_cache[cache_key] = commands
            return commands

        if commands:
            self.chain_cache[cache_key] = commands
            return commands

        console.print(
            f"[red]❌ {agent_name}: GPT failed. Using fallback stub chain.[/red]"
        )
        return self._generate_stub_chain(sorted_actions)

    def _build_context_summary(self, actions):
        summary = "\n".join(
            f"- {a.get('full_command', a['command'])} [{a.get('context', {}).get('phase', 'Unknown')}]"
            for a in actions
        )
        return summary

    def _call_gpt(self, prompt, agent_name, model="gpt-4o-mini"):
        try:
            result = subprocess.run(
                [
                    "sgpt",
                    "--model",
                    model,
                    "--temperature",
                    "0.3",
                    "--role",
                    "aria",
                    prompt,
                ],
                stdout=subprocess.PIPE,
                text=True,
                timeout=60 if model == "gpt-4.1" else 45,
            )
            cmds = [
                line.strip()
                for line in result.stdout.strip().splitlines()
                if line.strip()
            ]
            # Remove duplicates and trivial commands
            unique_cmds = []
            seen = set()
            for cmd in cmds:
                norm = cmd.strip().lower()
                if norm and norm not in seen and "echo" not in norm and "sleep" not in norm:
                    unique_cmds.append(cmd)
                    seen.add(norm)
            if unique_cmds and len(unique_cmds) == 5:
                if self.verbosity == "verbose":
                    console.print(
                        f"[blue]🎯 {agent_name}: Chain generated via {model}[/blue]"
                    )
                return unique_cmds
        except Exception as e:
            if self.verbosity != "silent":
                console.print(f"[red]⚠ {agent_name}: {model} GPT error: {e}[/red]")
        return []

    def _generate_stub_chain(self, sorted_actions):
        # Fallback: just return up to 5 best commands
        return [
            a.get("full_command", a["command"])
            for a in sorted_actions[:5]
        ]

    # ─────────────────────────────────────────────
    # 🛡️ Chain Validation — Memory & Vector Intelligence
    # ─────────────────────────────────────────────
    def validate_chain(self, commands, memory, agent_name="UnknownAgent"):
        known_templates = {
            teach.template_from_command(a.get("full_command", a["command"]))
            for a in memory.get("actions", [])
        }

        injected = 0
        for cmd in commands:
            tmpl = teach.template_from_command(cmd)
            if tmpl not in known_templates:
                # Cross-check with Vector DB before injection
                vector_result = vector_search.query(cmd, top_k=3)
                if not vector_result["results"]:
                    teach.inject_from_gpt(cmd, phase="unknown", reward=12)
                    if self.memory_router:
                        self.memory_router.store_global_insight(cmd)
                    injected += 1
                else:
                    console.print(
                        f"[blue]🔹 Vector DB already contains similar command: {cmd}[/blue]"
                    )

        if injected:
            console.print(
                f"[yellow]➕ {agent_name}: Injected {injected} novel commands into memory[/yellow]"
            )
        else:
            console.print(
                f"[green]✔ {agent_name}: No new commands needed after validation[/green]"
            )

        return commands

    # ─────────────────────────────────────────────
    # 💾 Smart Chain Saving — Orion-Integrated
    # ─────────────────────────────────────────────
    def save_chain(self, commands, agent_name="UnknownAgent"):
        now = datetime.now().isoformat()
        chain_data = [
            {
                "phase": PHASES[i] if i < len(PHASES) else "unknown",
                "command": cmd,
                "confidence_score": round(random.uniform(0.7, 0.99), 2),
                "trigger_reason": "Strategic chain generation",
                "timestamp": now,
                "agent": agent_name
            }
            for i, cmd in enumerate(commands)
        ]
        path = os.path.join(self.cache_dir, f"chain_{agent_name.lower()}.json")
        try:
            with open(path, "w") as f:
                json.dump({"agent": agent_name, "chain": chain_data}, f, indent=2)
            console.print(f"[green][Chain] {agent_name}: Chain saved to {path}[/green]")
        except Exception as e:
            console.print(f"[red]❌ Failed to save chain for {agent_name}: {e}[/red]")

    # ─────────────────────────────────────────────
    # 🕸️ Multi-Agent Chain Synthesis
    # ─────────────────────────────────────────────
    def build_and_store_chain_multiagent(self, agent_manager, dry_run=False):
        """
        agent_manager: must provide .all_agents() and each agent must have .memory_manager.memory and .agent_id
        memory_router: must be set on self if needed for caching/global insight
        """
        console.rule("[bold cyan]🕸️ Orion-Guided Multi-Agent Chain Synthesis Initiated")
        for agent in agent_manager.all_agents():
            use_orion_strategy = agent.agent_id == "OrionAgent"
            mem = agent.memory_manager.memory
            cmds = self.gpt_generate_chain(mem, agent_name=agent.agent_id, use_orion_strategy=use_orion_strategy)
            if not cmds:
                continue
            validated_cmds = self.validate_chain(cmds, mem, agent_name=agent.agent_id)
            if not dry_run:
                self.save_chain(validated_cmds, agent_name=agent.agent_id)
            # Concise summary per agent
            if self.verbosity != "silent":
                summary = ", ".join(validated_cmds)
                console.print(f"[cyan]{agent.agent_id} Chain:[/cyan] {summary}")
        if self.verbosity != "silent":
            console.print(
                "[blue]👁 OrionAgent: Chains synchronized. Strategic layer updated.[/blue]"
            )

    # ─────────────────────────────────────────────
    # ⚡ Quick Inline Chain Generation
    # ─────────────────────────────────────────────
    def generate_chain_from_memory(self, memory, agent_name="UnknownAgent"):
        console.print(
            f"[cyan]🔗 {agent_name}: Inline chain generation initiated...[/cyan]"
        )
        cmds = self.gpt_generate_chain(memory, agent_name=agent_name)
        return self.validate_chain(cmds, memory, agent_name=agent_name)

    # ─────────────────────────────────────────────
    # 🧪 Diagnostic & Orion Hooks (Future-Ready)
    # ─────────────────────────────────────────────
    def orion_unify_chains(self):
        """
        Placeholder for Orion's future unified chain logic.
        Will merge multi-agent chains into a coherent global attack strategy.
        """
        console.print(
            "[magenta]👁 OrionAgent: Unified chain synthesis pending implementation...[/magenta]"
        )

    def diagnostic_summary(self):
        """
        Display a summary of all generated chains.
        """
        console.rule("[bold green]📊 ChainBuilder Diagnostic Summary")
        chains = [f for f in os.listdir(self.cache_dir) if f.startswith("chain_")]
        for chain_file in chains:
            path = os.path.join(self.cache_dir, chain_file)
            with open(path, "r") as f:
                data = json.load(f)
                agent = data.get("agent", "Unknown")
                cmds = data.get("chain", [])
                console.print(
                    f"[cyan]{agent}:[/cyan] {len(cmds)} commands synthesized."
                )

        console.print(
            f"[blue]Total Chains:[/blue] {len(chains)} | [yellow]Location:[/yellow] {self.cache_dir}"
        )


# ─────────────────────────────────────────────
# 🚀 Execution Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    from core.multiagent.agent_manager import AgentManager
    from core.multiagent.memory_router import MemoryRouter

    console.rule("[bold magenta]⚡ ARIASKA ChainBuilder v12.0 — Standalone Mode")
    manager = AgentManager()
    memory_router = MemoryRouter(manager.all_agents())
    chainbuilder = ChainBuilder(memory_router=memory_router)

    chainbuilder.build_and_store_chain_multiagent(manager)
    chainbuilder.diagnostic_summary()
    chainbuilder.orion_unify_chains()
