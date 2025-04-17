import os
import json
from rich.console import Console

console = Console()

MEMORY_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/memory.json'))

class TeachModule:
    def __init__(self):
        self.memory = self.load_memory()

    def load_memory(self):
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, 'r') as f:
                    memory = json.load(f)
                console.print(f"[cyan]✔ Loaded memory.json ({len(memory.get('actions', []))} actions)[/cyan]")
                return memory
            except Exception as e:
                console.print(f"[red]❌ Failed to load memory.json: {e}[/red]")
        else:
            console.print("[yellow]⚠ No memory.json found. Starting fresh.[/yellow]")

        return {
            "actions": [],
            "rewards": {},
            "history": [],
            "scenarios": []
        }

    def save_memory(self):
        try:
            with open(MEMORY_FILE, 'w') as f:
                json.dump(self.memory, f, indent=4)
            console.print("[green]✔ Knowledge base memory.json updated.[/green]")
        except Exception as e:
            console.print(f"[red]❌ Failed to save memory.json: {e}[/red]")

    def add_action(self, command, description="", phase="Recon", reward=10, parameters=None, param_descriptions=None, when="", why=""):
        """
        Adds a single action to memory.json knowledge base.
        """
        if not command.strip():
            console.print("[red]❌ Invalid command. Cannot add empty command.[/red]")
            return

        parameters = parameters or []
        param_descriptions = param_descriptions or []

        base_command = command.split()[0]

        action_entry = {
            "command": base_command,
            "full_command": command,
            "description": description or "No description provided.",
            "tools": [base_command],
            "parameters": parameters,
            "param_descriptions": param_descriptions,
            "when": when or "Unknown execution context.",
            "why": why or "No reasoning provided.",
            "phase": phase,
            "reward": reward
        }

        existing_cmds = [a.get("command") for a in self.memory.get("actions", []) if isinstance(a, dict)]

        if base_command in existing_cmds:
            console.print(f"[yellow]⚠ Action already exists in KB: {command}[/yellow]")
            return

        self.memory["actions"].append(action_entry)
        self.memory["rewards"][command] = reward
        console.print(f"[cyan]➕ Added action:[/cyan] {command}")

        self.save_memory()

    def bulk_add_actions(self, actions):
        """
        Bulk add multiple actions (used by knowledge_loader).
        """
        new_count = 0

        for action in actions:
            cmd = action.get("full_command") or action.get("command")
            if not cmd:
                continue

            existing_cmds = [a.get("command") for a in self.memory.get("actions", []) if isinstance(a, dict)]
            base_command = cmd.split()[0]

            if base_command in existing_cmds:
                continue  # Skip duplicates

            # Clean action data
            action_entry = {
                "command": base_command,
                "full_command": cmd,
                "description": action.get("description", "No description provided."),
                "tools": action.get("tools", [base_command]),
                "parameters": action.get("parameters", []),
                "param_descriptions": action.get("param_descriptions", []),
                "when": action.get("when", "Unknown execution context."),
                "why": action.get("why", "No reasoning provided."),
                "phase": action.get("phase", "Recon"),
                "reward": action.get("reward", 50)
            }

            self.memory["actions"].append(action_entry)
            self.memory["rewards"][cmd] = action_entry["reward"]
            new_count += 1

        if new_count:
            console.print(f"[cyan]➕ Bulk added {new_count} new actions.[/cyan]")
            self.save_memory()
        else:
            console.print("[yellow]⚠ No new actions were added to memory.[/yellow]")

    def add_scenario(self, name, description=""):
        """
        Add a scenario to the KB memory.
        """
        if not name.strip():
            console.print("[red]❌ Invalid scenario name.[/red]")
            return

        existing_scenarios = [s.get("name") for s in self.memory.get("scenarios", [])]
        if name in existing_scenarios:
            console.print(f"[yellow]⚠ Scenario already exists: {name}[/yellow]")
            return

        scenario_entry = {
            "name": name,
            "description": description or "No description provided."
        }

        self.memory["scenarios"].append(scenario_entry)
        console.print(f"[cyan]➕ Added scenario:[/cyan] {name}")

        self.save_memory()

############################################
# Optional Test
############################################
if __name__ == "__main__":
    teach = TeachModule()
    teach.add_action(
        command="nmap -p- 10.10.10.10",
        description="Port scan all TCP ports",
        phase="Recon",
        reward=100,
        parameters=["-p-"],
        param_descriptions=["Scan all ports"],
        when="Initial recon",
        why="Identify open ports"
    )
