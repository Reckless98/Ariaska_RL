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

    def _lily_prompt(self, task_description: str) -> str:
        """
        LilyLLM prompt template: enforce concise, tactical, non-self-referential output.
        """
        return (
            "Provide a concise tactical recommendation in one sentence. "
            "Avoid any self-referential commentary. "
            "Be direct and actionable.\n"
            f"{task_description.strip()}"
        )

    def _postprocess_lily_output(self, output: str) -> str:
        """
        Remove verbose/AI disclaimers from LilyLLM output.
        """
        if not output:
            return ""
        # Remove common disclaimers
        patterns = [
            r"(?i)^as an ai( language)? model[,. ]*",
            r"(?i)^as a (cybersecurity )?ai( assistant)?[,. ]*",
            r"(?i)^i am (an|a) (ai|language model)[,. ]*",
            r"(?i)^note:.*",
            r"(?i)^please note.*",
        ]
        for pat in patterns:
            output = re.sub(pat, "", output).strip()
        # Remove trailing generic sentences
        output = re.sub(r"(?i)for more information.*$", "", output).strip()
        return output

    # === Dual-LLM Smart Decision Logic ===
    def smart_decision(
        self, task_type: str, task_description: str, agent_id: Optional[str] = None, use_gpt: bool = False
    ) -> str:
        """
        Dual-LLM decision flow:
        1. For tactical/planning tasks, use local LLMs (SenecaLLM/LilyLLM) first.
        2. Only use GPT if use_gpt=True AND local LLM fails or returns poor quality.
        
        Args:
            task_type: Type of task ('tactical', 'planning', etc.)
            task_description: Description of the task/query
            agent_id: ID of the agent making the request (for token tracking)
            use_gpt: Whether to allow fallback to GPT if local LLM succeeds
            
        Returns:
            str: The suggested command or response
        """
        console.print(f"[dim]🧠 Processing {task_type} task with use_gpt={use_gpt}[/dim]")
        
        try:
            # Track attempts and results
            attempts = {
                "lily": {"tried": False, "success": False, "result": None},
                "seneca": {"tried": False, "success": False, "result": None},
                "gpt": {"tried": False, "success": False, "result": None}
            }
            
            # Step 1: For tactical tasks, try Lily first (specialized for concise commands)
            if task_type in ("tactical", "recon"):
                attempts["lily"]["tried"] = True
                try:
                    lily_prompt = self._lily_prompt(task_description)
                    lily_suggestion = self.lily_llm.query(lily_prompt)
                    lily_suggestion = self._postprocess_lily_output(lily_suggestion)
                    console.print(f"[cyan]🌸 LilyLLM Suggestion:[/cyan] {lily_suggestion}")
                    
                    # Check if Lily returned a valid command
                    if self._is_valid_command(lily_suggestion):
                        attempts["lily"]["success"] = True
                        attempts["lily"]["result"] = lily_suggestion
                        
                        # If it's a valid command and we're not forcing GPT, return it directly
                        if not use_gpt:
                            console.print("[green]✓ Using LilyLLM suggestion (no GPT refinement needed)[/green]")
                            return lily_suggestion
                except Exception as e:
                    console.print(f"[yellow]⚠️ LilyLLM error: {e}[/yellow]")
            
            # Step 2: Try Seneca for all task types
            attempts["seneca"]["tried"] = True
            try:
                # Add task type hint to help Seneca generate better response
                enhanced_prompt = f"Task type: {task_type}\n{task_description}"
                seneca_suggestion = self.local_llm.query(enhanced_prompt)
                console.print(f"[blue]⚡ SenecaLLM Suggestion:[/blue] {seneca_suggestion}")
                
                # Check if Seneca returned a valid command
                if self._is_valid_command(seneca_suggestion):
                    attempts["seneca"]["success"] = True
                    attempts["seneca"]["result"] = seneca_suggestion
                    
                    # If it's a valid command and we're not forcing GPT, return it directly
                    if not use_gpt:
                        console.print("[green]✓ Using SenecaLLM suggestion (no GPT refinement needed)[/green]")
                        return seneca_suggestion
            except Exception as e:
                console.print(f"[yellow]⚠️ SenecaLLM error: {e}[/yellow]")
            
            # Step 3: Only use GPT in specific situations:
            # - If explicitly requested (use_gpt=True)
            # - OR if both local LLMs failed to produce valid outputs
            use_gpt_now = use_gpt or (
                (attempts["lily"]["tried"] and not attempts["lily"]["success"]) and
                (attempts["seneca"]["tried"] and not attempts["seneca"]["success"])
            )
            
            if use_gpt_now:
                attempts["gpt"]["tried"] = True
                
                # Prioritize the best local LLM result to send to GPT
                best_local_result = None
                if attempts["lily"]["success"]:
                    best_local_result = attempts["lily"]["result"]
                elif attempts["seneca"]["success"]:
                    best_local_result = attempts["seneca"]["result"]
                
                if best_local_result:
                    # If we have a local result, ask GPT to review/refine it
                    review_prompt = (
                        f"As a cybersecurity strategist, review this command for the task:\n\n"
                        f"Task: {task_description}\n"
                        f"Suggested Command: {best_local_result}\n\n"
                        f"If the command is correct, respond with EXACTLY that command. "
                        f"If incorrect, provide the proper command. "
                        f"Respond ONLY with the final command, no explanations."
                    )
                else:
                    # If no local results, ask GPT directly
                    review_prompt = (
                        f"As a cybersecurity expert, provide the exact command for this task:\n\n"
                        f"{task_description}\n\n"
                        f"Respond ONLY with the command, no explanations."
                    )
                
                final_command = self.gpt_request(
                    review_prompt,
                    task_type="reasoning",
                    agent_id=agent_id,
                    model="gpt-4o-mini",
                    use_gpt=True,  # This is an explicit GPT call
                )
                
                sanitized = self._sanitize_output(final_command)
                if sanitized and self._is_valid_command(sanitized):
                    attempts["gpt"]["success"] = True
                    attempts["gpt"]["result"] = sanitized
                    console.print(f"[green]🎯 Final Command (GPT):[/green] {sanitized}")
                    return sanitized
            
            # Step 4: Return best available result, with fallback priority:
            # Local LLMs are preferred if they succeeded but GPT refinement failed or wasn't used
            for source in ["lily", "seneca", "gpt"]:
                if attempts[source]["success"]:
                    console.print(f"[cyan]🔄 Returning best available result from {source}[/cyan]")
                    return attempts[source]["result"]
            
            # Final fallback if everything failed
            console.print("[red]❌ All LLM attempts failed[/red]")
            return "LLM unavailable"

        except Exception as e:
            console.print(f"[red]❌ smart_decision error: {e}[/red]")
            return "LLM unavailable"
    
    def _is_valid_command(self, command: str) -> bool:
        """
        Enhanced validation for commands to determine if they're usable.
        
        Args:
            command: Command string to validate
            
        Returns:
            bool: Whether the command is valid and usable
        """
        if not command or not isinstance(command, str):
            return False
            
        command = command.strip()
        if not command:
            return False
            
        # Valid single-word commands
        single_word_commands = {"ls", "pwd", "whoami", "id", "ps", "top", "help", "ifconfig", "ip"}
        if command.lower() in single_word_commands:
            return True
        
        # Reject excessively long outputs that are likely explanations, not commands
        if len(command) > 300 or command.count('\n') > 3:
            return False
            
        # Reject obvious explanation patterns
        explanation_patterns = [
            r'^I would use',
            r'^You can use',
            r'^The command to',
            r'^This command will',
            r'^To accomplish this'
        ]
        for pattern in explanation_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return False
        
        # Check if it looks like a shell command (word followed by parameters or flags)
        lines = command.splitlines()
        for line in lines:
            cleaned = line.strip()
            if cleaned and re.match(r'^[a-z]\w+(\s+(-{1,2}[a-zA-Z0-9]+|\S+))+$', cleaned, re.IGNORECASE):
                return True
                
        # Simple heuristic for short commands that start with common command tools
        common_command_prefixes = [
            "nmap", "ssh", "nc", "curl", "wget", "python", "perl", "bash",
            "cat", "echo", "grep", "find", "awk", "sed"
        ]
        for prefix in common_command_prefixes:
            if command.lower().startswith(f"{prefix} "):
                return True
                
        # If it doesn't match common patterns but is short and has a space, 
        # it might still be a valid command
        return len(command) < 80 and ' ' in command

    def dual_llm_feedback(
        self, task_description: str, agent_id: Optional[str] = None
    ) -> str:
        """
        Get Seneca and Lily outputs, then ask GPT to critique/consolidate.
        Returns the optimized command.
        """
        seneca_output = self.local_llm.query(task_description)
        lily_prompt = self._lily_prompt(task_description)
        lily_output = self.lily_llm.query(lily_prompt)
        lily_output = self._postprocess_lily_output(lily_output)
        feedback_prompt = (
            f"Seneca plan: {seneca_output}\n"
            f"Lily advice: {lily_output}\n"
            "As a cybersecurity strategist, critique and consolidate these into an optimized command. "
            "Respond only with the final command."
        )
        result = self.gpt_request(
            feedback_prompt,
            task_type="reasoning",
            agent_id=agent_id,
            model="gpt-4o-mini",
        )
        result = self._sanitize_output(result)
        # TODO: integrate dual-LLM critique loop with agent feedback/memory
        console.print(f"[magenta]🤖 Dual-LLM Feedback Result:[/magenta] {result}")
        return result

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
        use_gpt: bool = False,
        allow_fallback: bool = True,
    ):
        """
        Centralized GPT/LLM request handler with caching, fallback, and token tracking.
        For tactical/planning tasks, only use GPT if use_gpt=True.
        allow_fallback: If False, do not call local_llm.query() with fallback enabled (prevents recursion).
        """
        # --- Ensure cache key includes model name ---
        key = cache_key or f"{model}|{prompt.strip()[:120]}"
        if action_index is not None:
            from core.ui_helpers import get_action_description
            prompt = f"{prompt}\nAction Details: {get_action_description(action_index)}"
        if key in self.prompt_cache:
            console.print(f"[dim cyan]🧠 Cache hit: {key}[/dim cyan]")
            self._log_token_usage(agent_id, 0)
            return self.prompt_cache[key]
        model = model or self._select_model(task_type)
        # Lily pre-prompt for conciseness
        if "lily" in model.lower():
            prompt = "You are a concise assistant: answer succinctly.\n" + prompt
        # --- Local LLM priority for tactical/planning ---
        if task_type in ("tactical", "planning", "planner", "recon", "strategy") and not use_gpt:
            try:
                if "lily" in model.lower():
                    response = self.lily_llm.query(prompt, allow_fallback=allow_fallback)
                else:
                    response = self.local_llm.query(prompt, allow_fallback=allow_fallback)
                tokens = len(prompt.split()) + len(str(response).split())
                self.prompt_cache[key] = response
                self._log_token_usage(agent_id, tokens)
                return self._sanitize_output(response)
            except Exception as e:
                console.print(f"[yellow]⚠ LocalLLMManager failed: {e}[/yellow]")
        try:
            # Try local LLM first if available
            if hasattr(self, "local_llm") and self.local_llm:
                try:
                    response = self.local_llm.query(prompt, allow_fallback=allow_fallback)
                    if not response or len(response) > 800 or response.lower().startswith("error") or response.count(" ") < 2:
                        raise ValueError("Unusable output from local LLM")
                    self.prompt_cache[key] = response
                    return self._sanitize_output(response)
                except Exception as e:
                    console.print(f"[yellow]⚠ LocalLLMManager failed: {e}[/yellow]")
            # Try subprocess call with timeout
            try:
                result = subprocess.run(
                    ["sgpt", "--model", model, "--temperature", "0.4", "--role", "aria", prompt],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=12
                )
                output = result.stdout.strip()
                if output and len(output) < 800:
                    self.prompt_cache[key] = output
                    return self._sanitize_output(output)
            except subprocess.TimeoutExpired:
                console.print(f"[yellow]⚠ GPT subprocess timed out. Trying fallback model.[/yellow]")
            except Exception as e:
                console.print(f"[yellow]⚠ GPT subprocess error: {e}[/yellow]")
            # Fallback to alternate model
            if model != "gpt-3.5-turbo":
                try:
                    result = subprocess.run(
                        ["sgpt", "--model", "gpt-3.5-turbo", "--temperature", "0.4", "--role", "aria", prompt],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=8
                    )
                    output = result.stdout.strip()
                    if output and len(output) < 800:
                        self.prompt_cache[key] = output
                        return self._sanitize_output(output)
                except Exception as e:
                    console.print(f"[yellow]⚠ Fallback GPT-3.5 failed: {e}[/yellow]")
            # Fallback to cached answer if available
            if key in getattr(self, "prompt_cache", {}):
                console.print(f"[yellow]⚠ Using cached GPT response for {key}[/yellow]")
                return self._sanitize_output(self.prompt_cache[key])
            # Final fallback: short default response
            return self._sanitize_output("GPT unavailable. Please try again later.")
        except Exception as e:
            console.print(f"[red]❌ GPTManager.gpt_request failed: {e}[/red]")
            return self._sanitize_output("GPT error.")

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
        """
        Smart model selection: lightweight/embedding → nano/Lily, reasoning/critical → GPT-4o.
        """
        if task_type in ("embedding", "vectorize", "lightweight"):
            return "gpt-4.1-nano"
        if task_type in ("tactical", "tactic", "simple", "yesno"):
            return "QuantFactory/Lily-Cybersecurity-7B-v0.2-GGUF:Q8_0"
        if task_type in ("planner", "planning", "analysis"):
            return "wahidmounir/SenecaLLM_x_Qwen2.5-7B-CyberSecurity-Q8_0-GGUF"
        # Default: escalate to GPT-4o-mini for complex/critical
        return "gpt-4o-mini"

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
        # Enforce usage cap per agent
        if agent_id and self.token_limit and self.token_usage.get(agent_id, 0) > self.token_limit:
            console.print(f"[yellow]⚠ Agent {agent_id} exceeded GPT token cap ({self.token_limit})[/yellow]")

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
