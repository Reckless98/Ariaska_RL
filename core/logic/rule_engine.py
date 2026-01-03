# core/logic/rule_engine.py — ARIASKA Rule Engine v12.0 APEX STRATEGIST
# 🧠 GPT-Dominant Decision Engine | ⚡ Entropy-Weighted Intelligence | 👁 Orion-Integrated Reasoning | ♻️ Redundancy Control

import math
import json
import subprocess
import random
from typing import Dict, List, Any, Optional
from rich.console import Console
from rich.table import Table

from core.teach.teach import TeachModule
from core.gpt_manager import GPTManager

console = Console()

# Lazy TeachModule: initialized on first use to avoid import-time LLM init
_teach_module: Optional[TeachModule] = None


def get_teach_module() -> TeachModule:
    """Lazy getter for TeachModule. Avoids import-time LLM initialization."""
    global _teach_module
    if _teach_module is None:
        _teach_module = TeachModule()
    return _teach_module


# Lazy GPTManager: initialized on first use, respects runtime_flags
_gpt_manager: Optional[GPTManager] = None


def get_gpt_manager() -> GPTManager:
    """Lazy getter for GPTManager. Ensures runtime_flags are set before init."""
    global _gpt_manager
    if _gpt_manager is None:
        _gpt_manager = GPTManager()
    return _gpt_manager

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
# 🎯 Core Decision Pipeline — Dynamic, Adaptive Rules
# ─────────────────────────────────────────────
def rule_based_selection(
    memory: Dict[str, Any], state: Dict[str, Any], agent_id="RedAgent", memory_router=None, shared_context=None
) -> Dict[str, Any]:
    """
    Hybrid selection: Prefers GPT-optimized decisions, with dynamic rule-based fallback.
    Rules adapt based on training feedback and environment state.
    """
    console.rule(f"[bold cyan]🎯 {agent_id}: Initiating Decision Pipeline[/bold cyan]")

    # Use shared_context for context-aware decision making
    if shared_context:
        recent_phases = [v for k, v in shared_context.items() if k.endswith("_phase")]
        if recent_phases and state.get("phase") in recent_phases:
            console.print(f"[yellow]⚡ {agent_id}: Phase {state.get('phase')} already active by another agent.[/yellow]")

    # Dynamic rule adaptation: adjust rules based on environment and agent feedback
    dynamic_rules = get_dynamic_rules(state, memory)
    if dynamic_rules:
        for rule in dynamic_rules:
            if rule["condition"](state, memory):
                console.print(f"[blue]🧠 Dynamic Rule Triggered: {rule['description']}[/blue]")
                decision = rule["action"](state, memory)
                reasoning = rule.get("reasoning", lambda s, m: "Dynamic rule applied.")(state, memory)
                return {
                    "command": decision,
                    "source": "dynamic-rule",
                    "reasoning": reasoning,
                }

    # Fallback to GPT or static rules
    decision = gpt_decision_suggest(state, agent_id, memory_router=memory_router)
    if not decision or len(str(decision).split()) < 2:
        console.print(
            f"[yellow]⚠ {agent_id}: GPT suggestion weak, using static fallback.[/yellow]"
        )
        decision = fallback_command(state)

    # Redundancy/inefficiency detection
    if memory and "actions" in memory:
        from core.logic.redundancy_detector import detect_redundancy
        recent_cmds = [a.get("command") for a in memory["actions"][-5:]]
        if detect_redundancy(recent_cmds, state.get("last_command", "")):
            console.print(f"[yellow]♻ {agent_id}: Detected redundancy in recent commands.[/yellow]")

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

# --- Dynamic Rule System ---
def get_dynamic_rules(state, memory):
    """
    Return a list of dynamic rules (dicts) that adapt based on state/memory.
    Each rule: {"condition": fn, "action": fn, "description": str, "reasoning": fn}
    """
    rules = []

    # Example: If alert/risk is high, enforce stealth
    def high_alert_condition(state, memory):
        return state.get("blue_team_alert", 0) > 8 or state.get("detection_risk", 0) > 7

    def high_alert_action(state, memory):
        return "sleep 2 && nmap -T1 --top-ports 10 10.10.10.10"

    rules.append({
        "condition": high_alert_condition,
        "action": high_alert_action,
        "description": "High alert/risk: enforce stealthy command.",
        "reasoning": lambda s, m: "Stealth enforced due to high alert/risk."
    })

    # Example: If agent is stuck in phase, suggest phase shift
    def phase_stuck_condition(state, memory):
        phase_hist = state.get("phase_history", [])
        return len(phase_hist) >= 4 and all(p == phase_hist[-1] for p in phase_hist[-4:])

    def phase_stuck_action(state, memory):
        alt_phases = ["recon", "enumeration", "exploit", "privesc", "exfiltrate"]
        current = state.get("phase", "recon")
        alt = [p for p in alt_phases if p != current]
        return f"echo 'Switching phase to {random.choice(alt)}'"

    rules.append({
        "condition": phase_stuck_condition,
        "action": phase_stuck_action,
        "description": "Phase stuck: suggest phase shift.",
        "reasoning": lambda s, m: "Phase repeated, shifting for diversity."
    })

    # Add more dynamic rules as needed, possibly loaded from agent feedback or environment logs

    return rules

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
    gpt = get_gpt_manager()
    response = gpt.gpt_request(prompt, task_type="reasoning", agent_id=agent_id)
    return gpt._sanitize_output(response)


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
        return "echo 'No memory_router provided'"
    cached = None
    if hasattr(memory_router, 'check_gpt_cache'):
        cached = memory_router.check_gpt_cache(cache_key)
    if cached:
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
    gpt = get_gpt_manager()
    decision = gpt.gpt_request(prompt, task_type="decision", agent_id=agent_id)
    decision = gpt._sanitize_output(decision)
    memory_router.store_gpt_response(cache_key, decision)
    gpt_decision_cache[cache_key] = decision
    return decision


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
        memory_router = MemoryRouter()

    cache_key = f"{agent_id}_reason_{cmd}"
    cached = None
    if hasattr(memory_router, "check_gpt_cache"):
        cached = memory_router.check_gpt_cache(cache_key)
    if cached:
        if isinstance(cached, dict) and "response" in cached:
            return str(cached["response"])
        return str(cached) if cached else ""

    prompt = f"""
You are {agent_id}'s tactical analyst.
Explain in 2 sentences why the following command is optimal:
- Command: {cmd}
- Phase: {state.get('phase')}
- Blue Team Alert: {state.get('blue_team_alert')}
- Detection Risk: {state.get('detection_risk')}

Focus on stealth, efficiency, and strategic fit.
"""
    reasoning = get_gpt_manager().gpt_request(prompt, task_type="reasoning")
    if hasattr(memory_router, "store_gpt_response"):
        memory_router.store_gpt_response(cache_key, reasoning)
    gpt_reasoning_cache[cache_key] = reasoning
    return reasoning


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
    memory_router = MemoryRouter()

    for agent in agents:
        test_rule_engine(agent, memory_router)
        summarize_rule_stats(agent)
