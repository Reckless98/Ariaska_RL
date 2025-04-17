import re
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.lexers import Lexer
from rich.console import Console

from core.vector_search import query_vector_database
from core.rl_agent import RLAgent  # Only if needed in the future

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

        # === Local Base Commands ===
        for cmd in self.commands:
            if cmd.startswith(text):
                yield Completion(cmd, start_position=-len(text), style="fg:#FFB86C bold")

        # === Vector Search Suggestions ===
        if text and len(text) > 2:
            try:
                vector_results = query_vector_database(text, top_k=self.top_k)
                for res in vector_results:
                    suggestion = res if isinstance(res, str) else res.get('full_command', '')
                    if suggestion:
                        yield Completion(suggestion, start_position=-len(text), style="fg:#50FA7B italic")
            except Exception as e:
                console.print(f"[red]❌ Vector completer error:[/red] {e}")

# === Setup CLI Prompt Session ===
def setup_prompt(rl_agent):
    """
    rl_agent: RLAgent instance, used for retrieving known commands.
    """
    base_commands = rl_agent.get_base_commands()

    session = PromptSession(
        lexer=CustomLexer(),
        completer=VectorCompleter(base_commands),
        style=cli_style
    )

    return session

# === OPTIONAL CLI Standalone Debug/Test ===
if __name__ == "__main__":
    console.print("[bold magenta]🚀 Launching Ariaska CLI Interface Test Mode[/bold magenta]")

    rl_agent = RLAgent()  # If you want to test standalone
    session = setup_prompt(rl_agent)

    try:
        while True:
            user_input = session.prompt("zer0@Ariaska_CLI> ")
            console.print(f"[bold cyan]Input Received:[/bold cyan] {user_input}")

    except KeyboardInterrupt:
        console.print("\n[red]Exiting Test Mode[/red]")
