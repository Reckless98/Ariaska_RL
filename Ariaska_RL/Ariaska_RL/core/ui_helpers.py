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
from rich import box
from shutil import get_terminal_size

import re
from core.vector_search import VectorSearch

console = Console()

############################################
# Vector Worker Init (Singleton)
############################################
vector_worker = VectorSearch(cache_size=50)
print("[VectorSearch] Worker initialized (start() not required).")

############################################
# Custom Syntax Highlighting Lexer
############################################
class CustomLexer(Lexer):
    def lex_document(self, document):
        text = document.text

        def get_line(lineno):
            tokens = []
            for word in text.split():
                if re.match(r'\b(sudo|nmap|hydra|msfconsole|sqlmap|ffuf|gobuster|linpeas|winpeas|evil-winrm|masscan|amass)\b', word):
                    tokens.append(('class:command', word + ' '))
                elif word.startswith('-'):
                    tokens.append(('class:param', word + ' '))
                else:
                    tokens.append(('', word + ' '))
            return FormattedText(tokens)

        return get_line

############################################
# Ghost Text AutoSuggest (ZSH-style)
############################################
class VectorAutoSuggest(AutoSuggest):
    def get_suggestion(self, buffer, document):
        text = document.text.strip()

        if not text or len(text) < 1:
            return None

        query_result = vector_worker.query(text)
        suggestions = query_result.get('results', [])

        if suggestions:
            suggestion = suggestions[0].get('text', '')
            if suggestion and suggestion != text:
                return Suggestion(suggestion[len(text):])

        return None

############################################
# GhostText Tab Completion (Suggestions)
############################################
class GhostTextCompleter(Completer):
    def get_completions(self, document, complete_event):
        text = document.text_before_cursor.strip()

        if not text:
            return

        query_result = vector_worker.query(text)
        suggestions = query_result.get('results', [])

        for suggestion in suggestions:
            suggestion_text = suggestion.get('text', '')
            if suggestion_text and suggestion_text != text:
                yield Completion(
                    suggestion_text,
                    start_position=-len(text),
                    display=HTML(f'<style bg="gray"> {suggestion_text} </style>')
                )

############################################
# Syntax Highlighter for Output (Rich Style)
############################################
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
        (r"\s(-{1,2}[a-zA-Z0-9]+)", r" [bold blue]\1[/bold blue]"),
        (r"\b\d{1,3}(\.\d{1,3}){3}\b", "[bold green]\\g<0>"),
        (r"\b([0-9]{2,5})/tcp\b", "[bold cyan]\\1[/bold cyan]/tcp"),
        (r"\b([0-9]{2,5})/udp\b", "[bold cyan]\\1[/bold cyan]/udp"),
        (r"\bhttps\b", "[bold magenta]https[/bold magenta]"),
        (r"\bhttp\b", "[bold blue]http[/bold blue]"),
        (r"\bdomain\b", "[bold yellow]domain[/bold yellow]"),
        (r"(flag\{[^\}]+\})", "[bold red]\\g<0>")
    ]

    for pattern, replacement in rules:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text

############################################
# Display: Command Output
############################################
def display_output(output, title="Command Output", style="bold blue"):
    if not output:
        output = "[yellow]No output returned.[/yellow]"

    colorized_output = syntax_highlight(output)
    panel = Panel.fit(colorized_output, title=title, style=style, padding=(1, 2), box=box.ROUNDED)
    console.print(panel)
############################################
# Display: AI Hint Panel
############################################
def display_ai_hint(hint):
    panel = Panel.fit(f"💡 {hint}", title="AI Hint", style="bold green", box=box.ROUNDED)
    console.print(panel)

############################################
# Display: AI Recommendations Table (Full Width, Dynamic)
############################################
def display_ai_hint_table(hint, recommendations):
    term_width = get_terminal_size().columns
    console.rule("[bold green]🤖 AI Recommendations[/bold green]")

    if hint:
        display_ai_hint(hint if isinstance(hint, str) else hint.get("command", "💡 No hint available."))
    else:
        console.print(Panel("💡 No AI hint available.", style="yellow", box=box.ROUNDED))

    if not recommendations:
        console.print("[yellow]⚠ No AI Recommendations Available.[/yellow]")
        return

    table = Table(
        show_header=True,
        show_lines=True,
        expand=True,
        box=box.ROUNDED,
        padding=(0, 1)
    )

    table.add_column("Command", style="bold cyan", justify="center", ratio=30)
    table.add_column("Params", style="bold yellow", justify="center", ratio=20)
    table.add_column("Why", style="bold green", justify="center", ratio=50)
    table.add_column("Full Command", style="bold magenta", justify="center", ratio=80)

    for rec in recommendations:
        command = rec.get("command", "N/A")
        params = rec.get("params", "N/A")
        why = rec.get("why", "N/A")
        full_command = command
        table.add_row(command, params, why, full_command)

    aligned_table = Align.center(table, vertical="middle", width=term_width)
    console.print(aligned_table)

############################################
# Optional: Display Status Bar (Agent Status)
############################################
def display_status_bar(agent, episode, step):
    console.rule(f"[bold cyan]Agent: {agent} | Episode: {episode} | Step: {step}[/bold cyan]", style="cyan")

############################################
# Display: Phase Recommendations (Static Tables)
############################################
def display_phase_tables():
    recon_table = Table(title="🔎 Recon Phase Recommendations", show_lines=True, box=box.ROUNDED, expand=True)
    recon_table.add_column("Command", style="cyan", justify="center")
    recon_table.add_column("Param Descriptions", style="yellow", justify="center")
    recon_table.add_column("When / Why", style="green", justify="center")
    recon_table.add_column("Example Command", style="magenta", justify="center")

    recon_table.add_row("nmap", "-p- : all ports\n-sC : scripts\n-sV : version",
                        "Initial recon", "nmap -p- -sC -sV 10.10.10.10")
    recon_table.add_row("masscan", "-p1-65535 : all ports\n--rate : speed",
                        "Large subnet scan", "masscan -p1-65535 10.10.10.10 --rate=10000")

    exploit_table = Table(title="💥 Exploitation Phase Recommendations", show_lines=True, box=box.ROUNDED, expand=True)
    exploit_table.add_column("Command", style="cyan", justify="center")
    exploit_table.add_column("Param Descriptions", style="yellow", justify="center")
    exploit_table.add_column("When / Why", style="green", justify="center")
    exploit_table.add_column("Example Command", style="magenta", justify="center")

    exploit_table.add_row("hydra", "-L : user list\n-P : pass list\nssh:// : target",
                          "Brute-force creds", "hydra -L users.txt -P passwords.txt ssh://10.10.10.10")
    exploit_table.add_row("sqlmap", "-u : target URL\n--cookie : session",
                          "SQLi automation", "sqlmap -u http://10.10.10.10/login.php --cookie=PHPSESSID=123 --dbs")

    privesc_table = Table(title="🛡 Privilege Escalation Recommendations", show_lines=True, box=box.ROUNDED, expand=True)
    privesc_table.add_column("Command", style="cyan", justify="center")
    privesc_table.add_column("Param Descriptions", style="yellow", justify="center")
    privesc_table.add_column("When / Why", style="green", justify="center")
    privesc_table.add_column("Example Command", style="magenta", justify="center")

    privesc_table.add_row("linpeas.sh", "Auto enum privesc vectors", "Linux privesc", "bash linpeas.sh")
    privesc_table.add_row("winpeas.exe", "Auto enum privesc vectors", "Windows privesc", "winpeas.exe")

    post_table = Table(title="🎯 Post-Exploitation Recommendations", show_lines=True, box=box.ROUNDED, expand=True)
    post_table.add_column("Command", style="cyan", justify="center")
    post_table.add_column("Param Descriptions", style="yellow", justify="center")
    post_table.add_column("When / Why", style="green", justify="center")
    post_table.add_column("Example Command", style="magenta", justify="center")

    post_table.add_row("crackmapexec", "smb : module\n--shares : list shares",
                       "Validate creds / map shares", "crackmapexec smb 10.10.10.10 -u user -p pass --shares")
    post_table.add_row("evil-winrm", "-i : IP\n-u : user\n-p : password",
                       "Admin creds on Win", "evil-winrm -i 10.10.10.10 -u Admin -p Pass123!")

    console.print(recon_table)
    console.print(exploit_table)
    console.print(privesc_table)
    console.print(post_table)

############################################
# Display: Output Intelligence Analysis
############################################
def display_output_analysis(parsed_result):
    phase = parsed_result.get("phase", "unknown")
    success = parsed_result.get("success", False)
    artifacts = parsed_result.get("artifacts", [])
    hints = parsed_result.get("hints", [])
    entities = parsed_result.get("entities", {})
    excerpt = parsed_result.get("output_excerpt", "")
    risk = parsed_result.get("risk_score", 0.0)
    stealth = parsed_result.get("stealth_score", 1.0)

    panel_content = []

    header = f"[bold cyan]📡 Phase Detected:[/bold cyan] [bold magenta]{phase.upper()}[/bold magenta]"
    success_line = "[green]✅ SUCCESS[/green]" if success else "[red]❌ FAILURE[/red]"
    panel_content.append(f"{header} • {success_line}")
    panel_content.append(f"[bold red]🔺 Risk:[/bold red] {risk} | [bold green]🫥 Stealth:[/bold green] {stealth}")

    if artifacts:
        panel_content.append(f"[bold green]🧠 Artifacts:[/bold green] {', '.join(artifacts)}")
    if hints:
        for hint in hints:
            panel_content.append(f"[yellow]💡 {hint}[/yellow]")

    if entities:
        ent_table = Table(title="📦 Parsed Entities", show_lines=True, box=box.MINIMAL, padding=(0,1))
        ent_table.add_column("Type", style="bold cyan", justify="right")
        ent_table.add_column("Values", style="white", overflow="fold")
        for k, v in entities.items():
            if isinstance(v, list) and v:
                val_str = ", ".join(str(i) for i in v[:5]) + ("..." if len(v) > 5 else "")
                ent_table.add_row(k, val_str)
        panel_content.append(ent_table)

    if excerpt:
        panel_content.append(Panel.fit(syntax_highlight(excerpt), title="🔎 Output Snapshot", style="dim"))

    for item in panel_content:
        console.print(item)

############################################
# CLI Style (PromptToolkit)
############################################
cli_style = Style.from_dict({
    'prompt': 'bold #00FF00',
    'command': 'bold #FFB86C',
    'param': 'bold #8BE9FD',
    '': '#FFFFFF'
})

############################################
# Create the Prompt Session (Ghost Text + Completer)
############################################
def create_prompt_session():
    session = PromptSession(
        lexer=CustomLexer(),
        completer=GhostTextCompleter(),
        auto_suggest=VectorAutoSuggest(),
        style=cli_style
    )
    return session
