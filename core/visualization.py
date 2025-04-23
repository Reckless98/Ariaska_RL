from rich.console import Console
from rich.panel import Panel
from rich.table import Table, box
import torch

console = Console()

# === Banner Display ===
def display_banner():
    banner = """
    ╭──────────────────────────────────────────────────────────────╮
    │   █████╗ ██████╗ ██╗ █████╗ ███████╗██╗  ██╗ █████╗         │
    │  ██╔══██╗██╔══██╗██║██╔══██╗██╔════╝██║ ██╔╝██╔══██╗        │
    │  ███████║██████╔╝██║███████║███████╗█████╔╝ ███████║        │
    │  ██╔══██║██╔═══╝ ██║██╔══██║╚════██║██╔═██╗ ██╔══██║        │
    │  ██║  ██║██║     ██║██║  ██║███████║██║  ██╗██║  ██║        │
    │  ╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝        │
    │        ARIASKA_RL | Hybrid Offensive RL AI ⚔               │
    ╰──────────────────────────────────────────────────────────────╯
    """
    console.print(Panel.fit(banner, style="bold magenta"))

# === GPU Status ===
def display_gpu_status():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        details = "\n".join([
            f"GPU {i}: {torch.cuda.get_device_name(i)}"
            for i in range(num_gpus)
        ])
        console.print(Panel.fit(details, title="🚀 GPU Status", style="bold green"))
    else:
        console.print("[red]🚨 No GPUs detected! Running on CPU.[/red]")

# === Knowledge Base Summary ===
def display_knowledge_base_stats(memory):
    actions = len(memory.get("actions", []))
    scenarios = len(memory.get("scenarios", []))
    history = len(memory.get("history", []))

    table = Table(
        title="📚 Knowledge Base Overview",
        header_style="bold magenta",
        box=box.ROUNDED,
        expand=True
    )
    table.add_column("Metric", style="cyan", width=30)
    table.add_column("Count", justify="center", style="green", width=10)

    table.add_row("Actions Known", str(actions))
    table.add_row("Scenarios Loaded", str(scenarios))
    table.add_row("History Entries", str(history))

    console.print(table)

# === Phase-Based Initial Recommendations (Capped at 5 for clarity) ===
def show_initial_recommendations(memory, limit=5):
    console.print("\n[bold cyan]📚 Initial Tool Recommendations[/bold cyan]\n")

    phases = {
        "Recon": [],
        "Exploitation": [],
        "Privilege Escalation": [],
        "Post-Exploitation": []
    }

    valid_actions = [a for a in memory.get("actions", []) if isinstance(a, dict)]

    for action in valid_actions:
        phase = action.get("phase", "Other")
        if phase in phases:
            phases[phase].append(action)

    for phase, actions in phases.items():
        if not actions:
            continue

        console.rule(f"[bold green]{phase} Phase Recommendations[/bold green]")

        table = Table(
            title=f"{phase} Tools & Commands (Top {min(len(actions), limit)})",
            show_header=True,
            header_style="bold magenta",
            box=box.ROUNDED,
            show_lines=True,
            expand=True,
            row_styles=["dim", ""]
        )

        table.add_column("Command", style="cyan", no_wrap=True, width=20)
        table.add_column("Param Descriptions", style="yellow", width=35)
        table.add_column("When / Why", style="green", width=40)
        table.add_column("Example Command", style="bold blue", overflow="fold", width=40)

        for act in actions[:limit]:
            tool = act.get("command", "-")
            params_list = act.get("parameters", [])
            param_desc_list = act.get("param_descriptions", [])
            when = act.get("when", "No context provided.")
            why = act.get("why", "No reasoning provided.")
            full_cmd = act.get("full_command", tool)

            param_desc_string = "\n".join([
                f"[bold yellow]{param}[/bold yellow]: [bright_cyan]{desc}[/bright_cyan]"
                for param, desc in zip(params_list, param_desc_list)
            ]) if params_list and param_desc_list else "-"

            combo = f"[bold yellow]When:[/bold yellow] {when}\n[bold yellow]Why:[/bold yellow] {why}"

            table.add_row(tool, param_desc_string, combo, full_cmd)

        console.print(table)

# === Show ALL Tools in Knowledge Base ===
def show_all_tools(memory):
    console.print("\n[bold cyan]📚 Full Arsenal: All Known Tools[/bold cyan]\n")

    actions = [a for a in memory.get("actions", []) if isinstance(a, dict)]
    if not actions:
        console.print("[red]⚠ No actions found in Knowledge Base![/red]")
        return

    table = Table(
        title=f"🗡 ALL TOOLS KNOWN ({len(actions)} Tools)",
        header_style="bold magenta",
        box=box.ROUNDED,
        show_lines=True,
        expand=True,
        row_styles=["dim", ""]
    )

    table.add_column("#", style="cyan", justify="center", width=4)
    table.add_column("Tool", style="green", no_wrap=True, width=20)
    table.add_column("Phase", style="yellow", width=20)
    table.add_column("Description", style="bold blue", width=50)

    for idx, action in enumerate(actions, 1):
        cmd = action.get("command", "-")
        phase = action.get("phase", "Unknown")
        description = action.get("description", "No description available.")

        table.add_row(str(idx), cmd, phase, description)

    console.print(table)

# === AI Recommendations ===
def show_ai_recommendations(commands, explanations, memory):
    if not commands:
        console.print("[red]⚠ No AI recommendations available yet![/red]")
        return

    console.rule("[bold magenta]⚔ AI Recommendations ⚔[/bold magenta]")

    table = Table(
        header_style="bold magenta",
        box=box.ROUNDED,
        show_lines=True,
        expand=True,
        row_styles=["dim", ""]
    )

    table.add_column("#", style="cyan", justify="center", width=4)
    table.add_column("Command", style="green", no_wrap=True, width=20)
    table.add_column("Param Descriptions", style="yellow", width=35)
    table.add_column("When / Why", style="green", width=40)
    table.add_column("Example Command", style="bold blue", overflow="fold", width=40)

    for idx, (cmd, expl) in enumerate(zip(commands, explanations), 1):
        base_cmd = cmd.split()[0]

        action_data = next(
            (a for a in memory.get("actions", [])
             if isinstance(a, dict) and a["command"] == base_cmd),
            None
        )

        if action_data:
            params_list = action_data.get("parameters", [])
            param_desc_list = action_data.get("param_descriptions", [])
            when = action_data.get("when", "No context provided.")
            why = action_data.get("why", expl)

            param_desc_string = "\n".join([
                f"[bold yellow]{param}[/bold yellow]: [bright_cyan]{desc}[/bright_cyan]"
                for param, desc in zip(params_list, param_desc_list)
            ]) if params_list and param_desc_list else "-"
        else:
            param_desc_string = "-"
            when = "Unknown"
            why = expl

        when_why = f"[bold yellow]When:[/bold yellow] {when}\n[bold yellow]Why:[/bold yellow] {why}"

        table.add_row(
            str(idx),
            base_cmd,
            param_desc_string,
            when_why,
            cmd
        )

    console.print(table)

# === AI Hint Panel ===
def display_ai_hint(hint):
    panel = Panel.fit(f"💡 {hint}", title="AI Hint", style="bold green")
    console.print(panel)
