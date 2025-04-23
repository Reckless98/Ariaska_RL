# core/logic/rule_engine.py — ARIASKA Rule Engine v12.0 APEX STRATEGIST
# 🧠 GPT-Dominant Decision Engine | ⚡ Entropy-Weighted Intelligence | 👁 Orion-Integrated Reasoning | ♻️ Redundancy Control

import math
import json
import subprocess
from typing import Dict, List, Any
from rich.console import Console
from rich.table import Table

from core.teach.teach import TeachModule

console = Console()
teach = TeachModule()

# ─────────────────────────────────────────────
# 🔢 Utility Scoring & Entropy Functions
# ─────────────────────────────────────────────
def normalize(val, max_val=10):
    return min(1.0, max(0.0, val / max_val))


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def entropy_of_probs(probs: List[float]) -> float:
    probs = [max(p, 1e-6) for p in probs]
    return -sum(p * math.log(p) for p in probs)


# ─────────────────────────────────────────────
# 🎯 Core Decision Pipeline — GPT First
# ─────────────────────────────────────────────
def rule_based_selection(
    memory: Dict[str, Any], state: Dict[str, Any], agent_id="RedAgent", memory_router=None, shared_context=None
) -> Dict[str, Any]:
    """
    Hybrid selection: Prefers GPT-optimized decisions, with rule-based fallback.
    Now considers shared_context for multi-agent coordination.
    """
    console.rule(f"[bold cyan]🎯 {agent_id}: Initiating Decision Pipeline[/bold cyan]")

    # Use shared_context for context-aware decision making
    if shared_context:
        # Example: avoid redundant commands if another agent just did it
        recent_phases = [v for k, v in shared_context.items() if k.endswith("_phase")]
        if recent_phases and state.get("phase") in recent_phases:
            console.print(f"[yellow]⚡ {agent_id}: Phase {state.get('phase')} already active by another agent.[/yellow]")
            # Optionally, suggest a different phase or command

    decision = gpt_decision_suggest(state, agent_id, memory_router=memory_router)
    if not decision or len(str(decision).split()) < 2:
        console.print(
            f"[yellow]⚠ {agent_id}: GPT suggestion weak, using rule fallback.[/yellow]"
        )
        decision = fallback_command(state)

    # Integrate redundancy/inefficiency detection
    if memory and "actions" in memory:
        from core.logic.redundancy_detector import detect_redundancy
        recent_cmds = [a.get("command") for a in memory["actions"][-5:]]
        if detect_redundancy(recent_cmds, state.get("last_command", "")):
            console.print(f"[yellow]♻ {agent_id}: Detected redundancy in recent commands.[/yellow]")
            # Optionally, trigger GPT for a novel command

    from core.logic.redundancy_detector import detect_redundancy
    redundancy_flag = detect_redundancy(state.get("history", []), decision)
    if redundancy_flag:
        decision = gpt_redundancy_recover(state, agent_id, memory_router=memory_router)

    reasoning = explain_command_reasoning(decision, state, agent_id, memory_router=memory_router)
    console.print(f"[green]✔ Final Decision:[/green] {decision}")
    console.print(f"[blue]🧠 Reasoning:[/blue] {reasoning}")

    return {
        "command": decision,
        "source": "gpt" if not redundancy_flag else "gpt-redundancy",
        "reasoning": reasoning,
    }
# 🚨 Redundancy Detection & GPT Recovery
# ─────────────────────────────────────────────
def detect_redundancy(command_history: List[str], new_command: str) -> bool:
    if new_command in command_history[-5:]:
        console.print(f"[yellow]♻ Redundancy Detected: {new_command}[/yellow]")
        return True
    return False


def gpt_redundancy_recover(state: Dict[str, Any], agent_id="RedAgent", memory_router=None) -> str:
    prompt = f"""
Detected command redundancy in recent actions.
Current Phase: {state.get('phase')}
Last Command: {state.get('history', [])[-1] if state.get('history') else 'N/A'}

Suggest a novel, non-redundant command for phase '{state.get('phase')}'.
Respond ONLY with the command.
"""
    return _call_gpt_with_fallback(prompt, agent_id, memory_router=memory_router)


# ─────────────────────────────────────────────
# 🧠 GPT Decision Suggestion Engine
# ─────────────────────────────────────────────
gpt_decision_cache = {}
gpt_reasoning_cache = {}

def gpt_decision_suggest(
    state: Dict[str, Any], agent_id="RedAgent", memory_router=None
) -> str:
    cache_key = f"{agent_id}_decision_{state.get('phase', '')}"
    if cache_key in gpt_decision_cache:
        return gpt_decision_cache[cache_key]
    if memory_router is None:
        # Import locally to avoid circular import
        from core.multiagent.memory_router import MemoryRouter
        memory_router = MemoryRouter([])
    cache_key = f"{agent_id}_decision_{state.get('phase', '')}"
    cached = memory_router.check_gpt_cache(cache_key)
    if cached:
        if isinstance(cached, dict) and "response" in cached:
            return cached["response"]
        return cached

    prompt = f"""
You are {agent_id}'s tactical strategist AI.
Mission Context:
- Phase: {state.get('phase')}
- Target IP: {state.get('target_ip')}
- Open Ports: {state.get('open_ports')}
- Blue Team Alert Level: {state.get('blue_team_alert')}
- Privilege Level: {state.get('privilege_level')}
- Detection Risk: {state.get('detection_risk')}

Suggest ONE optimal command for this phase. Be concise and effective.
Respond ONLY with the command.
"""
    decision = _call_gpt_with_fallback(prompt, agent_id, memory_router=memory_router)
    memory_router.store_gpt_response(cache_key, decision)
    gpt_decision_cache[cache_key] = decision
    return decision


def _call_gpt_with_fallback(prompt: str, agent_id: str, memory_router=None) -> str:
    for model in ["gpt-4o-mini", "gpt-4.1-nano"]:
        try:
            result = subprocess.run(
                [
                    "sgpt",
                    "--model",
                    model,
                    "--temperature",
                    "0.35",
                    "--role",
                    "aria",
                    prompt,
                ],
                stdout=subprocess.PIPE,
                text=True,
                timeout=25,
            )
            output = result.stdout.strip()
            if output and len(output.split()) > 1:
                console.print(
                    f"[blue]🎯 {agent_id}: GPT({model}) Decision → {output}[/blue]"
                )
                return output
        except Exception as e:
            console.print(f"[red]⚠ {agent_id}: {model} failed: {e}[/red]")
    return "echo 'Fallback_Command'"


def orion_override_decision(
    current_command: str, state: Dict[str, Any], agent_id="RedAgent"
) -> str:
    """
    Placeholder for future Orion overrides.
    """
    console.print(f"[dim]👁 Orion review pending for {agent_id}...[/dim]")
    return current_command


def explain_command_reasoning(
    cmd: str, state: Dict[str, Any], agent_id="RedAgent", memory_router=None
) -> str:
    cache_key = f"{agent_id}_reason_{cmd}"
    if cache_key in gpt_reasoning_cache:
        return gpt_reasoning_cache[cache_key]
    if memory_router is None:
        from core.multiagent.memory_router import MemoryRouter
        memory_router = MemoryRouter([])

    cache_key = f"{agent_id}_reason_{cmd}"
    cached = memory_router.check_gpt_cache(cache_key)
    if cached:
        if isinstance(cached, dict) and "response" in cached:
            return cached["response"]
        return cached

    prompt = f"""
You are {agent_id}'s tactical analyst.
Explain in 2 sentences why the following command is optimal:
- Command: {cmd}
- Phase: {state.get('phase')}
- Blue Team Alert: {state.get('blue_team_alert')}
- Detection Risk: {state.get('detection_risk')}

Focus on stealth, efficiency, and strategic fit.
"""
    try:
        result = subprocess.run(
            [
                "sgpt",
                "--model",
                "gpt-4o-mini",
                "--temperature",
                "0.3",
                "--role",
                "aria",
                prompt,
            ],
            stdout=subprocess.PIPE,
            text=True,
            timeout=15,
        )
        reasoning = result.stdout.strip()
        memory_router.store_gpt_response(cache_key, reasoning)
        gpt_reasoning_cache[cache_key] = reasoning
        return reasoning
    except Exception as e:
        console.print(f"[yellow]⚠ Reasoning GPT failed: {e}[/yellow]")
        return "Strategic reasoning unavailable."


def summarize_rule_stats(agent):
    memory = agent.memory_manager.memory
    phase_counts = phase_command_distribution(memory)
    table = Table(title=f"📊 {agent.agent_id} Phase Command Distribution")
    table.add_column("Phase", style="cyan")
    table.add_column("Command Count", style="magenta")
    for phase, count in sorted(phase_counts.items()):
        table.add_row(phase, str(count))
    console.print(table)


def phase_command_distribution(memory: Dict[str, Any]) -> Dict[str, int]:
    counts = {}
    for action in memory.get("actions", []):
        phase = action.get("context", {}).get("phase", "unknown")
        counts[phase] = counts.get(phase, 0) + 1
    return counts

def phase_distribution_for_visualization(memory: Dict[str, Any]) -> Dict[str, int]:
    """Return phase distribution for visualization."""
    counts = {}
    for action in memory.get("actions", []):
        phase = action.get("phase", "unknown")
        counts[phase] = counts.get(phase, 0) + 1
    return counts

# ─────────────────────────────────────────────
# 🧪 Diagnostic & Test Hooks
# ─────────────────────────────────────────────
def test_rule_engine(agent, memory_router):
    dummy_state = {
        "phase": "recon",
        "open_ports": [22, 80, 443],
        "privilege_level": "none",
        "blue_team_alert": 3.5,
        "detection_risk": 0.4,
        "target_ip": "192.168.1.100",
    }
    decision = rule_based_selection(
        agent.memory_manager.memory, dummy_state, agent_id=agent.agent_id, memory_router=memory_router
    )
    reason = explain_command_reasoning(
        decision["command"], dummy_state, agent_id=agent.agent_id, memory_router=memory_router
    )

    console.print(f"[bold green]🎯 Decision:[/bold green] {decision['command']}")
    console.print(f"[bold blue]🧠 Reasoning:[/bold blue] {reason}")


def fallback_command(state: Dict[str, Any]) -> str:
    # Simple fallback logic for demonstration
    phase = state.get("phase", "")
    if phase == "recon":
        return "nmap -sV " + state.get("target_ip", "")
    elif phase == "exploit":
        return "exploit-db --search " + state.get("target_ip", "")
    else:
        return "echo 'No valid command found'"


# ─────────────────────────────────────────────
# 🚀 CLI Execution
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Import agent-related modules only in main guard to avoid circular imports
    from core.multiagent.agent_manager import AgentManager
    from core.multiagent.memory_router import MemoryRouter

    manager = AgentManager()
    agents = manager.all_agents()
    memory_router = MemoryRouter(agents)

    for agent in agents:
        test_rule_engine(agent, memory_router)
        summarize_rule_stats(agent)
