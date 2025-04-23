from prompt_toolkit.lexers import Lexer
from prompt_toolkit.formatted_text import FormattedText, HTML
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.styles import Style
from prompt_toolkit.shortcuts import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggest, Suggestion

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.align import Align
from rich.layout import Layout
from rich.live import Live
from rich.progress import Progress, BarColumn, TextColumn
from rich import box
from shutil import get_terminal_size

import re
import time
from core.vector_search import VectorSearch

console = Console()

# 🚀 Vector Worker Init (Singleton)
vector_worker = VectorSearch(cache_size=75)
console.print(
    "[cyan]✔ VectorSearch initialized — AI-powered suggestions active.[/cyan]"
)
# 🎨 Custom Syntax Highlighting Lexer
class CustomLexer(Lexer):
    def lex_document(self, document):
        text = document.text

        def get_line(lineno):
            tokens = []
            for word in text.split():
                if re.match(
                    r"\b(sudo|nmap|hydra|msfconsole|sqlmap|ffuf|gobuster|linpeas|winpeas|evil-winrm|masscan|amass|crackmapexec|enum4linux|pspy)\b",
                    word,
                ):
                    tokens.append(("class:command", word + " "))
                elif word.startswith("-"):
                    tokens.append(("class:param", word + " "))
                else:
                    tokens.append(("", word + " "))
            return FormattedText(tokens)

        return get_line
# 👻 Ghost Text AutoSuggest (Vector AI)
class VectorAutoSuggest(AutoSuggest):
    def get_suggestion(self, buffer, document):
        # Disable vector ghost suggestions to avoid interruptions
        return None
# ⌨️ GhostText Tab Completion (Context-Aware)
class VectorCompleter(Completer):
    def __init__(self, base_commands, top_k=5):
        self.commands = base_commands
        self.top_k = top_k

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor.strip()
        # Only suggest base commands, no vector/AI suggestions
        for cmd in self.commands:
            if cmd.startswith(text):
                yield Completion(cmd, start_position=-len(text), style="fg:#FFB86C bold")
        # Do NOT yield vector/AI completions or use display=HTML(...)
# 🎨 Syntax Highlighter for Output (Rich Style)
def syntax_highlight(text):
    rules = [
        (r"\bsudo\b", "[bold red]sudo[/bold red]"),
        (r"\bnmap\b", "[bold green]nmap[/bold green]"),
        (r"\bgobuster\b", "[bold cyan]gobuster[/bold cyan]"),
        (r"\bffuf\b", "[bold cyan]ffuf[/bold cyan]"),
        (r"\bhydra\b", "[bold yellow]hydra[/bold yellow]"),
        (r"\bmsfconsole\b", "[bold magenta]msfconsole[/bold magenta]"),
        (r"\bsqlmap\b", "[bold yellow]sqlmap[/bold yellow]"),
        (r"\bevil-winrm\b", "[bold red]evil-winrm[/bold red]"),
        (r"\blinpeas\b", "[bold cyan]linpeas[/bold cyan]"),
        (r"\bwinpeas\b", "[bold cyan]winpeas[/bold cyan]"),
        (r"\bmasscan\b", "[bold cyan]masscan[/bold cyan]"),
        (r"\bamass\b", "[bold green]amass[/bold green]"),
        (r"\b(enum4linux|pspy|crackmapexec)\b", "[bold magenta]\\1[/bold magenta]"),
        (r"\s(-{1,2}[a-zA-Z0-9]+)", r" [bold blue]\1[/bold blue]"),
        (r"\b\d{1,3}(\.\d{1,3}){3}\b", "[bold green]\\g<0>"),
        (r"\b([0-9]{2,5})/(tcp|udp)\b", "[bold cyan]\\1/\\2"),
        (r"\bhttps?\b", "[bold magenta]\\g<0>"),
        (r"(flag\{[^\}]+\})", "[bold red]\\1[/bold red]"),
    ]

    for pattern, replacement in rules:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text
# 📤 Display: Command Output Panel
def display_output(output, title="Command Output", style="bold blue"):
    if not output:
        output = "[yellow]No output returned.[/yellow]"
    colorized_output = syntax_highlight(output)
    panel = Panel.fit(
        colorized_output, title=title, style=style, padding=(1, 2), box=box.ROUNDED
    )
    console.print(panel)
# 💡 Display: AI Hint Panel
def display_ai_hint(hint):
    panel = Panel.fit(
        f"💡 {hint}", title="AI Hint", style="bold green", box=box.ROUNDED
    )
    console.print(panel)
# 📊 Display: AI Recommendations Table
def display_ai_hint_table(hint, recommendations):
    term_width = get_terminal_size().columns
    console.rule("[bold green]🤖 AI Recommendations[/bold green]")

    if hint:
        display_ai_hint(
            hint
            if isinstance(hint, str)
            else hint.get("command", "💡 No hint available.")
        )
    else:
        console.print(
            Panel("💡 No AI hint available.", style="yellow", box=box.ROUNDED)
        )

    if not recommendations:
        console.print("[yellow]⚠ No AI Recommendations Available.[/yellow]")
        return

    table = Table(show_header=True, show_lines=True, expand=True, box=box.ROUNDED)
    table.add_column("Command", style="cyan", justify="center")
    table.add_column("Params", style="yellow", justify="center")
    table.add_column("Why", style="green", justify="center")
    table.add_column("Full Command", style="magenta", justify="center")

    for rec in recommendations:
        table.add_row(
            rec.get("command", "N/A"),
            rec.get("params", "N/A"),
            rec.get("why", "N/A"),
            rec.get("command", "N/A"),
        )

    aligned_table = Align.center(table, vertical="middle", width=term_width)
    console.print(aligned_table)
# 📊 Display: Agent Status Bar
def display_status_bar(agent, episode, step):
    console.rule(
        f"[bold cyan]Agent: {agent} | Episode: {episode} | Step: {step}[/bold cyan]",
        style="cyan",
    )
# 🧭 Display: Phase Recommendations Tables
def display_phase_tables():
    phases = {
        "🔎 Recon": [
            (
                "nmap",
                "-p- -sC -sV",
                "Initial port & service scan",
                "nmap -p- -sC -sV 10.10.10.10",
            ),
            (
                "masscan",
                "-p1-65535 --rate=10000",
                "High-speed scan",
                "masscan -p1-65535 10.10.10.10 --rate=10000",
            ),
        ],
        "💥 Exploit": [
            (
                "hydra",
                "-L users.txt -P passwords.txt ssh://TARGET",
                "Brute-force login",
                "hydra -L users.txt -P passwords.txt ssh://10.10.10.10",
            ),
            (
                "sqlmap",
                "-u URL --cookie=session",
                "Automate SQLi",
                "sqlmap -u http://10.10.10.10/login.php --cookie=PHPSESSID=123 --dbs",
            ),
        ],
        "🛡 PrivEsc": [
            (
                "linpeas.sh",
                "",
                "Linux privilege escalation enumeration",
                "bash linpeas.sh",
            ),
            ("winpeas.exe", "", "Windows privilege escalation", "winpeas.exe"),
        ],
        "🎯 Post-Exploitation": [
            (
                "crackmapexec",
                "smb --shares",
                "Enumerate SMB shares",
                "crackmapexec smb 10.10.10.10 -u user -p pass --shares",
            ),
            (
                "evil-winrm",
                "-i IP -u user -p pass",
                "WinRM shell access",
                "evil-winrm -i 10.10.10.10 -u Admin -p Pass123!",
            ),
        ],
    }

    for phase, commands in phases.items():
        table = Table(
            title=f"{phase} Phase Recommendations",
            show_lines=True,
            box=box.ROUNDED,
            expand=True,
        )
        table.add_column("Command", style="cyan", justify="center")
        table.add_column("Params / Description", style="yellow", justify="center")
        table.add_column("When / Why", style="green", justify="center")
        table.add_column("Example", style="magenta", justify="center")

        for cmd, params, why, example in commands:
            table.add_row(cmd, params, why, example)
        console.print(table)

# 🧠 Display: Output Intelligence Analysis
def display_output_analysis(parsed_result):
    phase = parsed_result.get("phase", "unknown")
    success = parsed_result.get("success", False)
    artifacts = parsed_result.get("artifacts", [])
    hints = parsed_result.get("hints", [])
    entities = parsed_result.get("entities", {})
    excerpt = parsed_result.get("output_excerpt", "")
    risk = parsed_result.get("risk_score", 0.0)
    stealth = parsed_result.get("stealth_score", 1.0)

    console.rule("[bold blue]📡 Output Intelligence Summary[/bold blue]")

    status = "[green]✅ SUCCESS[/green]" if success else "[red]❌ FAILURE[/red]"
    console.print(f"[cyan]Phase:[/cyan] {phase} • {status}")
    console.print(f"[red]Risk:[/red] {risk} | [green]Stealth:[/green] {stealth}")

    if artifacts:
        console.print(f"[bold green]Artifacts:[/bold green] {', '.join(artifacts)}")
    if hints:
        for hint in hints:
            console.print(f"[yellow]💡 Hint:[/yellow] {hint}")

    if entities:
        ent_table = Table(title="📦 Entities Detected", show_lines=True, box=box.SIMPLE)
        ent_table.add_column("Type", style="cyan", justify="right")
        ent_table.add_column("Values", style="white", overflow="fold")
        for k, v in entities.items():
            if v:
                val_str = ", ".join(str(i) for i in v[:5]) + (
                    "..." if len(v) > 5 else ""
                )
                ent_table.add_row(k, val_str)
        console.print(ent_table)

    if excerpt:
        console.print(
            Panel.fit(
                syntax_highlight(excerpt), title="🔎 Output Snapshot", style="dim"
            )
        )
# 🚀 Create Prompt Session (Vector + GhostText)
# 🖌️ CLI Style Definition
cli_style = Style.from_dict({
    "command": "bold cyan",
    "param": "bold yellow",
    "": "",  # default
})

def create_prompt_session():
    return PromptSession(
        lexer=CustomLexer(),
        completer=VectorCompleter(base_commands=["nmap", "hydra", "msfconsole", "sqlmap", "ffuf", "gobuster", "linpeas", "winpeas", "evil-winrm", "masscan", "amass", "crackmapexec", "enum4linux", "pspy"]),
        auto_suggest=VectorAutoSuggest(),
        style=cli_style,
        # Use a visible arrow in your prompt, e.g. in main.py:
        # prompt_text = HTML('<ansicyan>zer0</ansicyan><ansimagenta>@ARIASKA</ansimagenta><ansibright_white> > </ansibright_white>')
    )
