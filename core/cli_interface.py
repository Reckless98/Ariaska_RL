import re
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.lexers import Lexer
from rich.console import Console

from core.vector_search import VectorSearch

console = Console()

# === CLI Color Style ===
cli_style = Style.from_dict({
    'prompt': 'bold #00FF00',
    'command': 'bold #FFB86C',
    'param': 'bold #8BE9FD',
    'ip': 'bold #FF5555',
    'path': 'bold #FF79C6',
    'flag': 'bold #50FA7B',
    '': '#FFFFFF'
})

# === Regex Patterns for Syntax Highlighting ===
COMMAND_PATTERN = r'^\s*(\w+)'
PARAM_PATTERN = r'(\s+-{1,2}[a-zA-Z0-9\-_]+)'
IP_PATTERN = r'(\b\d{1,3}(\.\d{1,3}){3}\b)'
FILE_PATTERN = r'(\s+\/\S+)'

# === Custom Lexer for Inline Syntax Highlight ===
class CustomLexer(Lexer):
    def lex_document(self, document):
        text = document.text

        def get_line(lineno):
            tokens = []
            pos = 0

            # Command Highlight
            match = re.match(COMMAND_PATTERN, text)
            if match:
                start, end = match.span(1)
                tokens.append(('class:command', text[start:end]))
                pos = end

            # Params Highlight
            for match in re.finditer(PARAM_PATTERN, text):
                start, end = match.span(1)
                tokens.append(('class:param', text[start:end]))

            # IPs Highlight
            for match in re.finditer(IP_PATTERN, text):
                start, end = match.span(1)
                tokens.append(('class:ip', text[start:end]))

            # File Paths Highlight
            for match in re.finditer(FILE_PATTERN, text):
                start, end = match.span(1)
                tokens.append(('class:path', text[start:end]))

            if not tokens:
                tokens.append(('', text))

            return tokens

        return get_line

# === Vector-Powered Autocomplete ===
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

# === Setup CLI Prompt Session ===
def setup_prompt(rl_agent):
    """
    rl_agent: RLAgent instance, used for retrieving known commands.
    """
    base_commands = rl_agent.get_base_commands()

    # Colorize zer0@ARIASKA prompt
    prompt_text = HTML('<ansicyan>zer0</ansicyan><ansimagenta>@ARIASKA</ansimagenta><ansiblack> > </ansiblack>')

    session = PromptSession(
        lexer=CustomLexer(),
        completer=VectorCompleter(base_commands),
        style=cli_style,
        # Set default prompt to colored HTML
        default_buffer_name="DEFAULT_BUFFER",
        # Use ghost text if you want (prompt_toolkit 3.0+)
        # See https://python-prompt-toolkit.readthedocs.io/en/master/pages/reference.html#prompt_toolkit.shortcuts.prompt.PromptSession
    )

    # To use the colored prompt in your main loop:
    # user_input = session.prompt(prompt_text)

    return session

# === OPTIONAL CLI Standalone Debug/Test ===
if __name__ == "__main__":
    console.print("[bold magenta]🚀 Launching Ariaska CLI Interface Test Mode[/bold magenta]")

    # Import or define RedAgent before using it
    try:
        from core.red_agent import RedAgent  # Adjust the import path as needed
    except ImportError:
        console.print("[red]Error: Could not import 'core.red_agent.RedAgent'. Please ensure the module exists and is in the PYTHONPATH.[/red]")
        exit(1)
    rl_agent = RedAgent()  # If you want to test standalone
    session = setup_prompt(rl_agent)
    console.print("[green]Successfully imported RedAgent![/green]")
    try:
        while True:
            user_input = session.prompt("zer0@Ariaska_CLI> ")
            console.print(f"[bold cyan]Input Received:[/bold cyan] {user_input}")

    except KeyboardInterrupt:
        console.print("\n[red]Exiting Test Mode[/red]")
