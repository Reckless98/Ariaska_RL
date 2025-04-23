# core/teach/teach.py — ARIASKA TeachModule v12.0 Distillation Nexus
# 🎓 Curriculum-Aware | 🧠 Deep GPT Distillation | 🌐 Global Knowledge Orchestrator | 📊 Advanced Teaching Analytics

import os
import json
import subprocess
import re
import time
from rich.console import Console
from core.utils.memory_manager import MemoryManager

console = Console()


class TeachModule:
    def __init__(self, agent_name="red_agent"):
        self.agent_name = agent_name
        self.memory_manager = MemoryManager(agent_name=agent_name)
        self.memory = self.memory_manager.memory
        self.template_cache = set(self._load_existing_templates())
        self.gpt_calls = 0
        self.gpt_call_limit = 100  # Expanded for smarter sessions
        self.teach_log_path = os.path.join("logs", f"{agent_name}_teach_log.jsonl")
        os.makedirs("logs", exist_ok=True)
        self.gpt_cache = {}
        console.print(
            f"[green]🎓 TeachModule v12.0 Initialized for {agent_name}[/green]"
        )

    def _load_existing_templates(self):
        return {
            self.template_from_command(
                action.get("full_command", action.get("command", ""))
            )
            for action in self.memory.get("actions", [])
        }

    def template_from_command(self, command):
        patterns = [
            (re.compile(r"(\b(?:\d{1,3}\.){3}\d{1,3}\b)"), "{IP}"),
            (re.compile(r"\b\d{2,5}\b"), "{PORT}"),
            (re.compile(r"\b[a-f0-9]{32,64}\b"), "{HASH}"),
            (re.compile(r"/[^\s]*"), "{PATH}"),
        ]
        tokens = command.strip().split()
        rebuilt = []
        for tok in tokens:
            replaced = False
            for pattern, placeholder in patterns:
                if pattern.search(tok):
                    rebuilt.append(placeholder)
                    replaced = True
                    break
            if not replaced:
                rebuilt.append(tok)
        return " ".join(rebuilt)

    def add_action(
        self,
        command,
        description="",
        phase="Recon",
        reward=10,
        parameters=None,
        param_descriptions=None,
        when="",
        why="",
        tags=None,
    ):
        if not command or not command.strip():
            console.print("[red]❌ Invalid command (empty).[/red]")
            return

        base_command = command.strip().split()[0]
        parameters = parameters or []
        param_descriptions = param_descriptions or []
        tags = tags or []

        template = self.template_from_command(command)
        is_new_template = template not in self.template_cache
        if is_new_template:
            self.template_cache.add(template)

        existing_cmds = [a.get("command") for a in self.memory["actions"]]
        if base_command in existing_cmds and not is_new_template:
            console.print(f"[yellow]⚠ Duplicate detected in KB: {command}[/yellow]")
            return

        entry = {
            "command": base_command,
            "full_command": command,
            "template": template,
            "description": description or "No description provided.",
            "tools": [base_command],
            "parameters": parameters,
            "param_descriptions": param_descriptions,
            "when": when or "Unknown context.",
            "why": why or "No reasoning given.",
            "phase": phase,
            "reward": reward,
            "tags": tags,
            "is_novel": is_new_template,
            "distill_ready": reward >= 50,
        }

        self.memory["actions"].append(entry)
        self.memory["rewards"][command] = reward

        if reward >= 60:
            shared = self.memory_manager.load_shared_knowledge()
            shared["insights"].append(entry)
            self.memory_manager.save_shared_knowledge(shared)

        self._log_teach_event(entry)
        console.print(
            f"[cyan]➕ Action Learned:[/cyan] {command} ({'🆕' if is_new_template else '↻'})"
        )
        self.memory_manager.save_memory()

    def inject_from_gpt(self, command, phase, reward=10):
        if self.gpt_calls >= self.gpt_call_limit:
            console.print("[yellow]⚠ GPT session limit reached.[/yellow]")
            return
        prompt = self._build_prompt(command, phase)
        return self._call_gpt_with_fallbacks(prompt, command, phase, reward)

    def _build_prompt(self, command, phase):
        return f"""
You are an elite cyber instructor AI.
Analyze this command for deep learning purposes.

Command: {command}
Phase: {phase}

Return STRICT JSON:
- description
- when
- why
- parameters (array)
- param_descriptions (array)
- tags (array of relevant keywords)
"""

    def _call_gpt_with_fallbacks(self, prompt, command, phase, reward):
        models = ["gpt-4o-mini", "gpt-4.1-nano"]
        for model in models:
            try:
                self.gpt_calls += 1
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
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=30,
                )
                raw = result.stdout.strip()
                if raw.startswith("{"):
                    gpt_data = json.loads(raw)
                    return self._inject_parsed_action(gpt_data, command, phase, reward)
            except Exception as e:
                console.print(f"[yellow]⚠ {model} failed: {e}[/yellow]")
        return self.inject_from_openai(command, phase, reward)

    def inject_from_openai(self, command, phase, reward=10):
        try:
            import openai

            openai.api_key = os.getenv("OPENAI_API_KEY", "")
            if not openai.api_key:
                raise RuntimeError("OPENAI_API_KEY not set")
            self.gpt_calls += 1
            response = openai.ChatCompletion.create(
                model="gpt-4.1-mini",
                temperature=0.35,
                max_tokens=250,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a cyber instructor AI. Reply with structured JSON only.",
                    },
                    {"role": "user", "content": self._build_prompt(command, phase)},
                ],
            )
            raw = response.choices[0].message.content.strip()
            json_str = re.search(r"\{.*\}", raw, re.DOTALL)
            gpt_data = json.loads(json_str.group(0) if json_str else raw)
            return self._inject_parsed_action(gpt_data, command, phase, reward)
        except Exception as e:
            console.print(f"[red]⚠ OpenAI fallback failed: {e} — using stub[/red]")
            return self.add_action(
                command=command,
                description="Fallback: GPT unavailable",
                when="Auto-injected context",
                why="GPT parsing failure",
                parameters=[],
                param_descriptions=[],
                phase=phase,
                reward=reward,
                tags=["fallback"],
            )

    def _inject_parsed_action(self, gpt_data, command, phase, reward):
        self.add_action(
            command=command,
            description=gpt_data.get("description", ""),
            when=gpt_data.get("when", ""),
            why=gpt_data.get("why", ""),
            parameters=gpt_data.get("parameters", []),
            param_descriptions=gpt_data.get("param_descriptions", []),
            tags=gpt_data.get("tags", []),
            phase=phase,
            reward=reward,
        )

    def bulk_add_actions(self, actions):
        added = 0
        for a in actions:
            cmd = a.get("full_command") or a.get("command")
            if not cmd:
                continue
            template = self.template_from_command(cmd)
            if template in self.template_cache:
                continue
            self.template_cache.add(template)

            base = cmd.strip().split()[0]
            entry = {
                "command": base,
                "full_command": cmd,
                "template": template,
                "description": a.get("description", "No description."),
                "tools": a.get("tools", [base]),
                "parameters": a.get("parameters", []),
                "param_descriptions": a.get("param_descriptions", []),
                "when": a.get("when", "Unknown"),
                "why": a.get("why", "No reasoning."),
                "phase": a.get("phase", "Recon"),
                "reward": a.get("reward", 50),
                "tags": a.get("tags", []),
                "is_novel": True,
                "distill_ready": True,
            }
            self.memory["actions"].append(entry)
            self.memory["rewards"][cmd] = entry["reward"]
            added += 1
        if added:
            console.print(f"[cyan]➕ Bulk imported {added} new actions.[/cyan]")
            self.memory_manager.save_memory()
        else:
            console.print(
                "[yellow]⚠ No novel actions detected in bulk import.[/yellow]"
            )

    def add_scenario(self, name, description=""):
        if not name.strip():
            console.print("[red]❌ Invalid scenario name[/red]")
            return
        existing = [s.get("name") for s in self.memory.get("scenarios", [])]
        if name in existing:
            console.print(f"[yellow]⚠ Scenario already exists: {name}[/yellow]")
            return
        self.memory["scenarios"].append(
            {"name": name.strip(), "description": description or "No description."}
        )
        console.print(f"[cyan]➕ Scenario registered:[/cyan] {name}")
        self.memory_manager.save_memory()

    def _log_teach_event(self, entry):
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "agent": self.agent_name,
            "command": entry["full_command"],
            "reward": entry["reward"],
            "phase": entry["phase"],
            "is_novel": entry["is_novel"],
            "tags": entry.get("tags", []),
        }
        with open(self.teach_log_path, "a") as logf:
            logf.write(json.dumps(log_entry) + "\n")
        console.print(f"[dim]📝 Teach log updated.[/dim]")

    def ask_gpt(self, prompt):
        if prompt in self.gpt_cache:
            return self.gpt_cache[prompt]
        # ...existing code...
        response = ... # result from GPT
        self.gpt_cache[prompt] = response
        return response
