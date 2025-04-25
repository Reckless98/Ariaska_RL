from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.live import Live
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, SpinnerColumn
from rich.columns import Columns
from rich import box
import os

# Add prompt_toolkit-based prompt session for CLI input
def create_prompt_session(prompt_text="> ", completer=None, lexer=None, style=None):
    """
    Create a Rich-compatible prompt session for CLI input.
    Optionally accepts a prompt_toolkit Completer, Lexer, and Style.
    """
    try:
        from prompt_toolkit import PromptSession
    except ImportError:
        raise ImportError("prompt_toolkit is required for create_prompt_session.")
    session_kwargs = {}
    if completer:
        session_kwargs["completer"] = completer
    if lexer:
        session_kwargs["lexer"] = lexer
    if style:
        session_kwargs["style"] = style
    return PromptSession(prompt_text, **session_kwargs)

console = Console()

def display_output(output, title="Output", style="cyan"):
    """
    Display formatted output in a Rich panel.
    """
    from rich.panel import Panel
    from rich.syntax import Syntax

    # If output is code-like, use Syntax highlighting, else plain text
    if isinstance(output, str) and ("\n" in output or len(output) > 80):
        try:
            # Try to auto-detect code blocks
            panel_content = Syntax(output, "bash", theme="ansi_dark", line_numbers=False)
        except Exception:
            panel_content = output
    else:
        panel_content = output

    console.print(Panel(panel_content, title=title, border_style=style))

def display_ai_hint_table(phase=None, recommendations=None):
    """
    Display AI-suggested commands and recommendations in a Rich table.
    
    Args:
        phase (str, optional): Current phase of attack. Defaults to None.
        recommendations (list, optional): List of command recommendations. Defaults to None.
    """
    from rich.table import Table

    if not recommendations:
        return

    table = Table(title=f"🧠 AI-Suggested Commands {f'for {phase}' if phase else ''}", box=box.ROUNDED)
    table.add_column("Command", style="cyan")
    table.add_column("Params", style="magenta")
    table.add_column("Description", style="green")

    for rec in recommendations:
        # Handle different recommendation formats
        if isinstance(rec, dict):
            command = rec.get("command", "N/A")
            params = rec.get("params", "")
            desc = rec.get("why", "")
            table.add_row(command, params, desc)
        elif isinstance(rec, str):
            # Simple string recommendation
            table.add_row(rec, "", "")

    console.print(table)

def get_action_description(action_index):
    action_map = {
        0: "nmap -sS -sV [target]",
        1: "gobuster dir -u [url] -w [wordlist]",
        2: "hydra -l admin -P [wordlist] ssh://[target]",
        3: "find / -perm -u=s -type f",
        4: "zip -r /tmp/data.zip /etc/passwd",
    }
    return action_map.get(action_index, f"Custom/Unknown ({action_index})")

def display_redagent_learning_dashboard(redagent=None, memory_router=None, redagent_brain=None):
    # Accept either direct agent or memory_router/brain
    if redagent is not None:
        memory_router = getattr(redagent, "memory_router", None)
        redagent_brain = getattr(redagent, "redagent_brain", None)
    if memory_router is None or redagent_brain is None:
        console.print("[yellow]RedAgent dashboard: missing memory_router or redagent_brain[/yellow]")
        return

    # Timeline panel: last 15 steps, color-coded by success
    steps = redagent_brain.load_recent_steps(n=15)
    timeline = Table(title="RedAgent Timeline", box=box.ROUNDED)
    timeline.add_column("Step", style="dim")
    timeline.add_column("Intent", style="cyan")
    timeline.add_column("Command", style="magenta")
    timeline.add_column("Reward", style="green")
    timeline.add_column("Success", style="yellow")
    timeline.add_column("Model", style="blue")
    for s in steps:
        reward = s.get("reward", 0)
        reward_str = f"[green]{reward:.2f}[/green]" if reward > 0 else f"[red]{reward:.2f}[/red]"
        success = s.get("success", False)
        success_str = "[bold green]✔[/bold green]" if success else "[red]✗[/red]"
        action = s.get("command", "-")
        if isinstance(action, int):
            action = get_action_description(action)
        timeline.add_row(
            str(s.get("step", "-")),
            str(s.get("state", {}).get("phase", "-")),
            str(action),
            reward_str,
            success_str,
            str(s.get("model", "-"))
        )

    # GPT call log: last 5 GPT feedbacks
    gpt_feedbacks = redagent_brain.load_recent_gpt_feedback(n=5)
    gpt_table = Table(title="SGPT Call Log", box=box.ROUNDED)
    gpt_table.add_column("Episode", style="dim")
    gpt_table.add_column("Prompt", style="cyan")
    gpt_table.add_column("Summary", style="magenta")
    gpt_table.add_column("Feedback", style="yellow")
    for f in gpt_feedbacks:
        gpt_table.add_row(
            str(f.get("episode", "-")),
            (f.get("prompt", "")[:30] + "...") if f.get("prompt") else "",
            (f.get("summary", "")[:30] + "...") if f.get("summary") else "",
            (f.get("gpt_feedback", "")[:40] + "...") if f.get("gpt_feedback") else ""
        )

    # Rolling reward graph (ASCII bar)
    rewards = [s.get("reward", 0) for s in steps]
    reward_bar = "".join(
        "[green]█[/green]" if r > 0 else "[red]█[/red]" for r in rewards
    )
    reward_panel = Panel(reward_bar, title="Rolling Reward", border_style="green")

    # Memory snapshot: show last 5 commands and their success
    mem_snapshot = Table(title="Memory Snapshot", box=box.ROUNDED)
    mem_snapshot.add_column("Command", style="cyan")
    mem_snapshot.add_column("Success", style="yellow")
    for s in steps[-5:]:
        mem_snapshot.add_row(
            str(s.get("command", "-")),
            "[green]✔[/green]" if s.get("success", False) else "[red]✗[/red]"
        )

    # Evolution stats: top 3 commands by success
    stats = memory_router.get_evolution_stats() if hasattr(memory_router, "get_evolution_stats") else {}
    top_cmds = []
    for intent_hash, stat in stats.items():
        for cmd, count in stat.get("commands", {}).items():
            top_cmds.append((cmd, count, stat.get("success", 0)))
    top_cmds = sorted(top_cmds, key=lambda x: -x[1])[:3]
    stats_table = Table(title="Top Commands", box=box.ROUNDED)
    stats_table.add_column("Command", style="magenta")
    stats_table.add_column("Count", style="cyan")
    stats_table.add_column("Success", style="green")
    for cmd, count, succ in top_cmds:
        stats_table.add_row(str(cmd), str(count), str(succ))

    # Layout
    layout = Layout()
    layout.split_column(
        Layout(Columns([timeline, gpt_table]), name="top", ratio=2),
        Layout(Columns([reward_panel, mem_snapshot, stats_table]), name="bottom", ratio=1)
    )
    console.print(layout)

def display_phase_tables(agent=None, memory=None, phase_counts=None, title="Phase Distribution"):
    """
    Display phase distribution or phase-related tables for an agent using Rich.
    Args:
        agent: Agent instance (optional, for title/context).
        memory: Memory dict with 'actions' (optional).
        phase_counts: Dict of phase -> count (optional, overrides memory).
        title: Table title.
    """
    from rich.table import Table
    from rich.panel import Panel

    # Compute phase counts if not provided
    if phase_counts is None and memory is not None:
        phase_counts = {}
        for action in memory.get("actions", []):
            phase = action.get("phase") or action.get("context", {}).get("phase", "unknown")
            phase_counts[phase] = phase_counts.get(phase, 0) + 1

    if not phase_counts:
        console.print("[yellow]No phase data available to display.[/yellow]")
        return

    table = Table(title=title, box=box.ROUNDED)
    table.add_column("Phase", style="cyan")
    table.add_column("Count", style="magenta")

    for phase, count in sorted(phase_counts.items()):
        table.add_row(str(phase), str(count))

    agent_title = f" for {getattr(agent, 'agent_id', '')}" if agent else ""
    console.print(Panel(table, title=f"📊 Phase Table{agent_title}", border_style="blue"))

def display_status_bar(agent_id, episode, step, reward=None, phase=None, info=None):
    """
    Display a compact status bar for an agent's current step, with color-coded phase, risk, and GPT usage.
    """
    bar = f"[bold cyan]{agent_id}[/bold cyan] | Ep [green]{episode}[/green] | Step [yellow]{step}[/yellow]"
    if phase:
        phase_colors = {
            "recon": "blue",
            "enumeration": "cyan",
            "exploit": "yellow",
            "privesc": "magenta",
            "exfiltrate": "red"
        }
        color = phase_colors.get(str(phase).lower(), "white")
        bar += f" | Phase: [{color}]{phase}[/{color}]"
    if reward is not None:
        color = "green" if reward > 0 else "red" if reward < 0 else "yellow"
        bar += f" | Reward: [{color}]{reward:.2f}[/{color}]"
    if info:
        if "risk" in info:
            risk = info["risk"]
            risk_color = "green" if risk < 3 else "yellow" if risk < 6 else "red"
            bar += f" | Risk: [{risk_color}]{risk:.2f}[/{risk_color}]"
        if "gpt_calls" in info:
            gpt_calls = info["gpt_calls"]
            bar += f" | GPT: [magenta]{gpt_calls}[/magenta]"
        for k, v in info.items():
            if k not in ("risk", "gpt_calls"):
                bar += f" | {k}: {v}"
    console.print(bar, highlight=False)

def display_agent_panel(agent, info=None):
    """
    Display a Rich panel with agent info and optional step info.
    """
    table = Table(title=f"{getattr(agent, 'agent_id', 'Agent')} Status", box=box.ROUNDED)
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Role", getattr(agent, "role", "N/A"))
    table.add_row("Episode", str(getattr(agent, "total_episodes", "N/A")))
    table.add_row("Step", str(getattr(agent, "total_steps", "N/A")))
    table.add_row("Epsilon", f"{getattr(agent, 'epsilon', 0):.3f}")
    table.add_row("Entropy", f"{getattr(agent, 'entropy_beta', 0):.3f}")
    if info:
        for k, v in info.items():
            table.add_row(str(k), str(v))
    console.print(Panel(table, border_style="blue"))

def display_training_progress(current, total, agent_id=None):
    """
    Display a Rich progress bar for training steps.
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30),
        TextColumn("[cyan]{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(f"{agent_id or 'Agent'} Training", total=total)
        progress.update(task, completed=current)

def display_llm_usage_summary(agent_manager, gpt_manager):
    """
    [Dashboard] Show LLM usage summary for all agents.
    """
    table = Table(title="🧠 LLM Usage Summary", box=None)
    table.add_column("Agent", style="cyan")
    table.add_column("Seneca", style="blue")
    table.add_column("Lily", style="magenta")
    table.add_column("GPT", style="yellow")
    for agent in agent_manager.all_agents():
        agent_id = getattr(agent, "agent_id", "N/A")
        # TODO: Track Seneca/Lily calls if available
        gpt_calls = gpt_manager.get_token_usage(agent_id)
        table.add_row(agent_id, "-", "-", str(gpt_calls))
    console.print(Panel(table, title="LLM Calls Per Agent", border_style="yellow"))

def display_agent_health_panel(agent_manager):
    """
    [Dashboard] Show agent health (exploration/exploitation, stagnation, etc).
    """
    table = Table(title="Agent Health", box=None)
    table.add_column("Agent", style="cyan")
    table.add_column("Phase", style="magenta")
    table.add_column("Health", style="green")
    for agent in agent_manager.all_agents():
        phase = getattr(agent, "current_mode", "N/A")
        # Health: green/yellow/red based on recent rewards (stub)
        rewards = getattr(agent.stats_monitor, "agent_stats", {}).get(agent.agent_id, {}).get("rewards", [])
        health = "[green]●[/green]"
        if rewards and sum(rewards[-5:]) < 0:
            health = "[red]●[/red]"
        elif rewards and sum(rewards[-5:]) < 2.5:
            health = "[yellow]●[/yellow]"
        table.add_row(agent.agent_id, str(phase), health)
    console.print(Panel(table, title="Agent Health", border_style="green"))
