# core/gpt_manager.py — ARIASKA GPTManager v2.0 DUAL-LLM
# Centralized LLM Gateway: SenecaLLM + GPT Orchestration, Caching, Fallback, Logging

import os
import time
import hashlib
import subprocess
import threading
import re
from typing import Optional, Dict, Any
from rich.console import Console
from core.utils.local_llm_manager import LocalLLMManager

console = Console()


def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


class GPTManager:
    """
    Centralized manager for all GPT/LLM calls in ARIASKA_RL.
    Now enhanced with Dual-LLM: SenecaLLM (local) + GPT (cloud).
    Handles routing, caching, fallback, logging, and cost tracking.
    """

    def __init__(
        self,
        primary_model="gpt-4.1",
        fallback_model="gpt-4o-mini",
        embedding_model="gpt-4.1-nano",
        token_limit: int = None,
        cache_size: int = 5000,
        log_path: str = None,
    ):
        self.primary_model = os.getenv("GPT_PRIMARY_MODEL", primary_model)
        self.fallback_model = os.getenv("GPT_FALLBACK_MODEL", fallback_model)
        self.embedding_model = os.getenv("GPT_EMBEDDING_MODEL", embedding_model)
        self.token_limit = int(os.getenv("GPT_TOKEN_LIMIT", token_limit or 8000))
        self.cache_size = int(os.getenv("GPT_CACHE_SIZE", cache_size))
        self.prompt_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_lock = threading.Lock()
        self.token_usage = {}  # agent_id -> int
        self.log_path = log_path or os.getenv("GPT_LOG_PATH", "logs/gpt_manager.log")
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # Initialize Local LLM (SenecaLLM via Ollama)
        self.local_llm = LocalLLMManager()
        self.lily_llm = LocalLLMManager(model_name="QuantFactory/Lily-Cybersecurity-7B-v0.2-GGUF:Q8_0")

    # === Dual-LLM Smart Decision Logic ===
    def smart_decision(
        self, task_type: str, task_description: str, agent_id: Optional[str] = None
    ) -> str:
        """
        Dual-LLM decision flow:
        1. For tactical tasks, try LilyLLM first.
        2. If not valid, fallback to SenecaLLM + GPT review.
        """
        try:
            if task_type == "tactical":
                lily_suggestion = self.lily_llm.query(task_description)
                console.print(f"[cyan]🌸 LilyLLM Suggestion:[/cyan] {lily_suggestion}")
                if self._is_simple_command(lily_suggestion):
                    return lily_suggestion
            # ...existing SenecaLLM + GPT review logic...
            seneca_suggestion = self.local_llm.query(task_description)
            console.print(f"[blue]⚡ SenecaLLM Suggestion:[/blue] {seneca_suggestion}")

            if self._is_simple_command(seneca_suggestion):
                return seneca_suggestion

            review_prompt = (
                f"As a cybersecurity strategist, review the AI's suggested command:\n\n"
                f"Task: {task_description}\n"
                f"Suggestion: {seneca_suggestion}\n\n"
                f"Do you approve this command? If not, refine it. Respond ONLY with the final Linux command."
            )
            final_command = self.gpt_request(
                review_prompt,
                task_type="reasoning",
                agent_id=agent_id,
                model="gpt-4o-mini",
            )
            sanitized = self._sanitize_output(final_command)
            console.print(f"[green]🎯 Final Command (GPT Refined):[/green] {sanitized}")
            return sanitized

        except Exception as e:
            console.print(f"[red]❌ smart_decision error: {e}[/red]")
            return "LLM unavailable"

    def _is_simple_command(self, command: str) -> bool:
        if not command:
            return False
        lines = command.strip().split("\n")
        if len(lines) == 1 and "<" not in lines[0] and "[" not in lines[0]:
            return True
        return False

    # === Core GPT Request Handling ===
    def gpt_request(
        self,
        prompt,
        task_type="reasoning",
        agent_id=None,
        cache_key=None,
        model=None,
        action_index=None,
    ):
        """
        Centralized GPT/LLM request handler with caching, fallback, and token tracking.
        All GPT calls must use this method.
        """
        key = cache_key or f"{agent_id or 'global'}|{task_type}|{prompt[:80]}"

        if action_index is not None:
            from core.ui_helpers import get_action_description

            prompt = f"{prompt}\nAction Details: {get_action_description(action_index)}"

        if key in self.prompt_cache:
            console.print(f"[dim cyan]🧠 Cache hit: {key}[/dim cyan]")
            self._log_token_usage(agent_id, 0)
            return self.prompt_cache[key]

        model = model or self._select_model(task_type)
        try:
            # Prefer local LLMs for tactical/planner/embedding
            if model in ["wahidmounir/SenecaLLM_x_Qwen2.5-7B-CyberSecurity-Q8_0-GGUF", "QuantFactory/Lily-Cybersecurity-7B-v0.2-GGUF:Q8_0"]:
                if "lily" in model.lower():
                    response = self.lily_llm.query(prompt)
                else:
                    response = self.local_llm.query(prompt)
                tokens = len(prompt.split()) + len(str(response).split())
                self.prompt_cache[key] = response
                self._log_token_usage(agent_id, tokens)
                return response
            # ...existing code for GPT API...
            response, tokens = self._call_gpt_api(prompt, model)
            self.prompt_cache[key] = response
            self._log_token_usage(agent_id, tokens)
            console.print(f"[green]🧠 Cached: {key} | Tokens: {tokens}[/green]")
            return response
        except Exception as e:
            fallback_model = self._select_fallback_model(model)
            console.print(
                f"[yellow]⚠ Primary failed, fallback to: {fallback_model}[/yellow]"
            )
            try:
                # Only use GPT-4o-mini or similar for fallback/high-level
                if fallback_model in ["gpt-4o-mini", "gpt-4.1-nano"]:
                    response, tokens = self._call_gpt_api(prompt, fallback_model)
                    self.prompt_cache[key] = response
                    self._log_token_usage(agent_id, tokens)
                    console.print(
                        f"[green]🧠 Fallback cached: {key} | Tokens: {tokens}[/green]"
                    )
                    return response
                else:
                    # Fallback to local LLM if possible
                    response = self.local_llm.query(prompt)
                    tokens = len(prompt.split()) + len(str(response).split())
                    self.prompt_cache[key] = response
                    self._log_token_usage(agent_id, tokens)
                    return response
            except Exception as e2:
                self._log_token_usage(agent_id, 0)
                console.print(f"[red]❌ All models failed: {e2}[/red]")
                return f"GPT unavailable: {e2}"

    def _call_gpt_api(self, prompt, model):
        """
        Call GPT via sgpt subprocess and return (response, token_count).
        """
        result = subprocess.run(
            ["sgpt", "--model", model, "--role", "aria", prompt],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20,
            text=True,
        )
        response = result.stdout.strip()
        tokens = len(re.findall(r"\w+", prompt)) + len(re.findall(r"\w+", response))
        return response, tokens

    def _select_model(self, task_type):
        if task_type == "reasoning":
            return self.primary_model
        elif task_type == "embedding":
            return self.embedding_model
        elif task_type == "analysis":
            return self.fallback_model
        return self.fallback_model

    def _select_fallback_model(self, model):
        if model == self.primary_model:
            return self.fallback_model
        return self.embedding_model

    # === Token Usage Tracking ===
    def _log_token_usage(self, agent_id, tokens):
        if not isinstance(self.token_usage, dict):
            self.token_usage = {}
        if agent_id:
            self.token_usage.setdefault(agent_id, 0)
            self.token_usage[agent_id] += tokens
        self.token_usage.setdefault("global", 0)
        self.token_usage["global"] += tokens

    def get_token_usage(self, agent_id=None):
        if not isinstance(self.token_usage, dict):
            self.token_usage = {}
        if agent_id:
            return self.token_usage.get(agent_id, 0)
        return self.token_usage.get("global", 0)

    # === Output Sanitization ===
    def _sanitize_output(self, output: str) -> str:
        if not output:
            return ""
        forbidden = [
            "import os",
            "import sys",
            "subprocess",
            "eval(",
            "exec(",
            "open(",
            "os.system",
            "__import__",
            "pickle.load",
            "base64.b64decode",
        ]
        for f in forbidden:
            output = output.replace(f, "[REDACTED]")
        output = output.replace("`", "'")
        if len(output) > 1000:
            output = output[:1000] + "..."
        return output.strip()

    # === Embedding Request ===
    def embedding_request(self, prompt: str) -> str:
        return self.gpt_request(prompt, task_type="embedding")

    # === Cache Management ===
    def clear_cache(self):
        with self.cache_lock:
            self.prompt_cache.clear()

    def set_token_limit(self, limit: int):
        self.token_limit = limit

    def set_cache_size(self, size: int):
        self.cache_size = size

    def get_cache_stats(self) -> Dict[str, Any]:
        with self.cache_lock:
            return {"size": len(self.prompt_cache), "max_size": self.cache_size}

    # === Context Sync ===
    def sync_context(self, context_data: dict):
        try:
            self.prompt_cache["global_context"] = context_data
            console.print("[green]🧠 Context synchronized successfully.[/green]")
            return True
        except Exception as e:
            console.print(f"[red]❌ Context sync failed: {e}[/red]")
            return False


# === Example Usage ===
if __name__ == "__main__":
    gpt_manager = GPTManager()
    task = "Scan 10.10.10.5 for open ports"
    final_cmd = gpt_manager.smart_decision("recon", task)
    print(f"Final Command: {final_cmd}")
