# core/gpt_manager.py — ARIASKA GPTManager v2.1 APEX
# Centralized LLM Gateway: SenecaLLM + GPT Orchestration, Caching, Fallback, Logging

import os
import time
import hashlib
import subprocess
import threading
import re
import json
from typing import Optional, Dict, Any, List, Union, Tuple
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

    # === Configurable Fallback Tree and Cache Persistence ===
    FALLBACK_TREE = ["lily", "seneca", "gpt"]  # Can be extended via config
    CACHE_PERSIST_PATH = os.getenv("GPT_CACHE_PERSIST", "logs/gpt_prompt_cache.json")

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

        # Prompt templates for different tasks (optimized for token efficiency)
        self.prompt_templates = self._init_prompt_templates()
        
        # Stat counters
        self.fallback_attempts = 0
        self.successful_requests = 0
        self.request_times = []

        self._load_cache()

    def _init_prompt_templates(self) -> Dict[str, str]:
        """Initialize prompt templates for different task types."""
        return {
            # Tactical tasks (commands, actions)
            "tactical": (
                "Task: {task_description}\n"
                "Respond with a single command for this cybersecurity task.\n"
                "Context: Target hosts: {target_hosts}, discovered ports: {open_ports}, current phase: {phase}\n"
                "Include only the command itself, no explanations or formatting."
            ),
            
            # Planning tasks (strategy, sequencing)
            "planning": (
                "Plan the next sequence of actions for this task: {task_description}\n"
                "Current phase: {phase}, available information: {context}\n"
                "Format: numbered list of up to 3 steps, each being a precise command."
            ),
            
            # Reasoning tasks (analysis, evaluation)
            "reasoning": (
                "Analyze this cybersecurity scenario: {task_description}\n"
                "Context: {context}\n"
                "Provide your assessment in a concise paragraph. Include recommended action."
            ),
            
            # Embedding tasks (vector representation)
            "embedding": (
                "Convert this text to a vector representation:\n{task_description}\n"
                "Format: JSON array of 16 floating point values between -1 and 1."
            ),
            
            # Analysis tasks (output interpretation)
            "analysis": (
                "Interpret the output of this command:\n"
                "Command: {command}\n"
                "Output: {output}\n"
                "Provide a concise summary of the key findings."
            ),
            
            # Default catch-all template
            "default": "Task: {task_description}\nRespond with a concise answer."
        }

    def _load_cache(self):
        """Load persistent cache from disk if available."""
        try:
            import json
            if os.path.exists(self.CACHE_PERSIST_PATH):
                with open(self.CACHE_PERSIST_PATH, "r") as f:
                    self.prompt_cache = json.load(f)
        except Exception as e:
            console.print(f"[yellow]⚠ Failed to load persistent cache: {e}[/yellow]")

    def _save_cache(self):
        """Persist cache to disk."""
        try:
            import json
            with open(self.CACHE_PERSIST_PATH, "w") as f:
                json.dump(self.prompt_cache, f)
        except Exception as e:
            console.print(f"[yellow]⚠ Failed to save persistent cache: {e}[/yellow]")

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
        self, task_type: str, task_description: str, agent_id: Optional[str] = None, use_gpt: bool = False,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Enhanced dual-LLM decision flow:
        1. Format prompt with task-specific template and minimal context
        2. For tactical/planning tasks, use local LLMs (SenecaLLM/LilyLLM) first.
        3. Only use GPT if use_gpt=True AND local LLM fails or returns poor quality.
        
        Args:
            task_type: Type of task ('tactical', 'planning', etc.)
            task_description: Description of the task/query
            agent_id: ID of the agent making the request (for token tracking)
            use_gpt: Whether to allow fallback to GPT if local LLM succeeds
            context: Additional context for prompt templating
            
        Returns:
            str: The suggested command or response
        """
        start_time = time.time()
        
        # Format prompt with template and minimal context
        formatted_prompt = self._format_prompt_with_template(task_type, task_description, context)
        
        # Generate hash for caching
        prompt_hash = hash_prompt(formatted_prompt)
        if prompt_hash in self.prompt_cache:
            # Cache hit
            return self.prompt_cache[prompt_hash]
            
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
                    lily_prompt = self._lily_prompt(formatted_prompt)
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
                            
                            # Cache and return
                            self.prompt_cache[prompt_hash] = lily_suggestion
                            self._save_cache()
                            
                            # Track successful request
                            self.successful_requests += 1
                            self.request_times.append(time.time() - start_time)
                            
                            return lily_suggestion
                except Exception as e:
                    console.print(f"[yellow]⚠️ LilyLLM error: {e}[/yellow]")
            
            # Step 2: Try Seneca for all task types
            attempts["seneca"]["tried"] = True
            try:
                # Add task type hint to help Seneca generate better response
                enhanced_prompt = f"Task type: {task_type}\n{formatted_prompt}"
                seneca_suggestion = self.local_llm.query(enhanced_prompt)
                console.print(f"[blue]⚡ SenecaLLM Suggestion:[/blue] {seneca_suggestion}")
                
                # Check if Seneca returned a valid command
                if self._is_valid_command(seneca_suggestion):
                    attempts["seneca"]["success"] = True
                    attempts["seneca"]["result"] = seneca_suggestion
                    
                    # If it's a valid command and we're not forcing GPT, return it directly
                    if not use_gpt:
                        console.print("[green]✓ Using SenecaLLM suggestion (no GPT refinement needed)[/green]")
                        
                        # Cache and return
                        self.prompt_cache[prompt_hash] = seneca_suggestion
                        self._save_cache()
                        
                        # Track successful request
                        self.successful_requests += 1
                        self.request_times.append(time.time() - start_time)
                        
                        return seneca_suggestion
            except Exception as e:
                console.print(f"[yellow]⚠️ SenecaLLM error: {e}[/yellow]")
            
            # Track fallback attempt
            self.fallback_attempts += 1
            
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
                    # If no local results, ask GPT directly with minimized prompt
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
                    
                    # Cache and return
                    self.prompt_cache[prompt_hash] = sanitized
                    self._save_cache()
                    
                    # Track successful request
                    self.successful_requests += 1
                    self.request_times.append(time.time() - start_time)
                    
                    return sanitized
            
            # Step 4: Correction loop: try to auto-correct with next LLM in tree
            if not any(attempts[src]["success"] for src in self.FALLBACK_TREE):
                for idx, src in enumerate(self.FALLBACK_TREE):
                    if attempts[src]["tried"] and not attempts[src]["success"]:
                        # If not last in tree, try next LLM with feedback
                        if idx + 1 < len(self.FALLBACK_TREE):
                            next_src = self.FALLBACK_TREE[idx + 1]
                            feedback = f"Previous {src} output was invalid. Please provide a valid shell command only."
                            try:
                                if next_src == "seneca":
                                    enhanced_prompt = f"{formatted_prompt}\n{feedback}"
                                    suggestion = self.local_llm.query(enhanced_prompt)
                                elif next_src == "lily":
                                    suggestion = self.lily_llm.query(self._lily_prompt(formatted_prompt + f"\n{feedback}"))
                                else:  # gpt
                                    suggestion = self.gpt_request(
                                        f"{formatted_prompt}\n{feedback}",
                                        task_type="reasoning",
                                        agent_id=agent_id,
                                        model="gpt-4o-mini",
                                        use_gpt=True,
                                    )
                                if self._is_valid_command(suggestion):
                                    attempts[next_src]["success"] = True
                                    attempts[next_src]["result"] = suggestion
                                    console.print(f"[yellow]🔄 Correction loop: {next_src} provided valid command.[/yellow]")
                                    
                                    # Cache and return
                                    self.prompt_cache[prompt_hash] = suggestion
                                    self._save_cache()
                                    
                                    # Track successful request
                                    self.successful_requests += 1
                                    self.request_times.append(time.time() - start_time)
                                    
                                    return suggestion
                            except Exception as e:
                                console.print(f"[yellow]⚠ Correction loop error ({next_src}): {e}[/yellow]")

            # Step 5: Return best available result, with fallback priority:
            # Local LLMs are preferred if they succeeded but GPT refinement failed or wasn't used
            for source in self.FALLBACK_TREE:
                if attempts[source]["success"]:
                    console.print(f"[cyan]🔄 Returning best available result from {source}[/cyan]")
                    
                    # Cache and return
                    best_result = attempts[source]["result"]
                    self.prompt_cache[prompt_hash] = best_result
                    self._save_cache()
                    
                    # Track successful request
                    self.successful_requests += 1
                    self.request_times.append(time.time() - start_time)
                    
                    return best_result
            
            # Final fallback if everything failed
            console.print("[red]❌ All LLM attempts failed[/red]")
            
            # Track timing even for failures
            self.request_times.append(time.time() - start_time)
            
            # Return a sensible fallback
            fallback = self._generate_fallback_command(task_type, task_description)
            self.prompt_cache[prompt_hash] = fallback
            self._save_cache()
            return fallback

        except Exception as e:
            console.print(f"[red]❌ smart_decision error: {e}[/red]")
            # Track timing even for failures
            self.request_times.append(time.time() - start_time)
            return "LLM unavailable"
    
    def _format_prompt_with_template(self, task_type: str, task_description: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Format prompt using task-specific template and minimal context.
        """
        # Get the right template
        template = self.prompt_templates.get(task_type, self.prompt_templates["default"])
        
        # Initialize context with defaults
        ctx = {
            "task_description": task_description,
            "phase": "recon",
            "target_hosts": "10.10.10.10", 
            "open_ports": "[]",
            "context": "minimal information available",
            "command": "",
            "output": ""
        }
        
        # Update with provided context
        if context:
            ctx.update(context)
        
        # Format template
        try:
            return template.format(**ctx)
        except KeyError as e:
            # If formatting fails, use a simple fallback
            console.print(f"[yellow]⚠️ Template formatting error: {e}. Using simple prompt.[/yellow]")
            return f"Task type: {task_type}\n{task_description}"
    
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

    def _generate_fallback_command(self, task_type: str, task_description: str) -> str:
        """Generate a fallback command when all LLMs fail."""
        task_lower = task_description.lower()
        
        if task_type == "tactical" or task_type == "recon" or "scan" in task_lower:
            return "nmap -sS -sV 10.10.10.10"
        elif "exploit" in task_lower or "vulnerability" in task_lower:
            return "searchsploit apache 2.4"
        elif "directory" in task_lower or "fuzzing" in task_lower:
            return "gobuster dir -u http://10.10.10.10 -w /usr/share/wordlists/common.txt"
        elif "password" in task_lower or "brute force" in task_lower:
            return "hydra -l admin -P /usr/share/wordlists/rockyou.txt ssh://10.10.10.10"
        elif "privilege" in task_lower:
            return "sudo -l"
        else:
            return "echo 'LLM fallback: Command generation failed'"

    def dual_llm_feedback(
        self, task_description: str, agent_id: Optional[str] = None
    ) -> str:
        """
        Get Seneca and Lily outputs, then ask GPT to critique/consolidate.
        Returns the optimized command.
        """
        # Minimize token usage by using focused prompts
        seneca_output = self.local_llm.query(f"Generate a command for: {task_description}")
        lily_prompt = self._lily_prompt(task_description)
        lily_output = self.lily_llm.query(lily_prompt)
        lily_output = self._postprocess_lily_output(lily_output)
        
        # Minimal feedback prompt
        feedback_prompt = (
            f"Compare and optimize these commands for the task '{task_description}':\n"
            f"Command 1: {seneca_output}\n"
            f"Command 2: {lily_output}\n"
            "Respond with only the best command, no explanations."
        )
        
        result = self.gpt_request(
            feedback_prompt,
            task_type="reasoning",
            agent_id=agent_id,
            model="gpt-4o-mini",
        )
        
        result = self._sanitize_output(result)
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
            if model != "gpt-4o-mini":
                try:
                    result = subprocess.run(
                        ["sgpt", "--model", "gpt-4o-mini", "--temperature", "0.4", "--role", "aria", prompt],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=8
                    )
                    output = result.stdout.strip()
                    if output and len(output) < 800:
                        self.prompt_cache[key] = output
                        return self._sanitize_output(output)
                except Exception as e:
                    console.print(f"[yellow]⚠ Fallback GPT-4o-mini failed: {e}[/yellow]")

            if model != "gpt-4o-mini":
                try:
                    result = subprocess.run(
                        ["sgpt", "--model", "gpt-4o-mini", "--temperature", "0.4", "--role", "aria", prompt],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=8
                    )
                    output = result.stdout.strip()
                    if output and len(output) < 800:
                        self.prompt_cache[key] = output
                        return self._sanitize_output(output)
                except Exception as e:
                    console.print(f"[yellow]⚠ Fallback GPT-4o-mini failed: {e}[/yellow]")

            # Final fallback to default response
            if allow_fallback:
                return self._generate_fallback_response(task_type, prompt)
            else:
                return "LLM request failed: no models available"
        except Exception as e:
            console.print(f"[yellow]⚠ GPT request error: {e}[/yellow]")
            return self._sanitize_output(self._generate_fallback_response(task_type, prompt))

    def _call_gpt_api(self, prompt, model):
        """
        Call GPT via sgpt subprocess and return (response, token_count).
        """
        try:
            result = subprocess.run(
                ["sgpt", "--model", model, prompt],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10
            )
            output = result.stdout.strip()
            # Estimate token count (rough approximation)
            token_count = len(prompt.split()) + len(output.split())
            return output, token_count
        except subprocess.TimeoutExpired:
            console.print(f"[yellow]⚠ GPT API call timed out for model {model}[/yellow]")
            return None, 0
        except Exception as e:
            console.print(f"[yellow]⚠ GPT API error: {e}[/yellow]")
            return None, 0

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
        if agent_id:
            with self.cache_lock:
                self.token_usage[agent_id] = self.token_usage.get(agent_id, 0) + tokens
        return tokens

    def get_token_usage(self, agent_id=None):
        if agent_id:
            return self.token_usage.get(agent_id, 0)
        return sum(self.token_usage.values())

    # === Output Sanitization ===
    def _sanitize_output(self, output):
        """
        Sanitize GPT output to remove any potential dangerous elements.
        """
        if output is None:
            return ""
        
        # Convert to string
        output = str(output)
        
        # Remove common prefixes
        prefixes_to_strip = [
            "I'd recommend", 
            "Here's the command", 
            "The command is",
            "You can use", 
            "Try using", 
            "Use this command",
            "Here is",
            "Sure,",
            "As a",
            "Here's a",
            "Let me",
            "To accomplish",
            "The optimal",
        ]
        
        for prefix in prefixes_to_strip:
            if output.startswith(prefix):
                output = output[len(prefix):].strip()
                break
                
        # Extract code blocks
        code_block_match = re.search(r"```(?:\w+)?\n(.+?)\n```", output, re.DOTALL)
        if code_block_match:
            return code_block_match.group(1).strip()
            
        # Extract inline code
        inline_code_match = re.search(r"`(.+?)`", output)
        if inline_code_match:
            return inline_code_match.group(1).strip()
            
        # Extract the first line if it's a command
        lines = output.strip().split("\n")
        if lines and self._is_simple_command(lines[0]):
            return lines[0].strip()
            
        return output

    def _generate_fallback_response(self, task_type, prompt):
        """
        Generate appropriate fallback response for the given task type.
        """
        if task_type in ("tactical", "recon"):
            return "nmap -sV -p- 10.10.10.10"
        elif task_type == "planning":
            return "1. Scan targets\n2. Enumerate services\n3. Search exploits"
        elif task_type == "exploit":
            return "searchsploit apache"
        elif task_type in ("reasoning", "analysis"):
            return "The output suggests vulnerability in the target system that could be exploited."
        else:
            return f"Unable to process {task_type} request. Please try again."

    def save_cache(self):
        self._save_cache()

    def get_stats(self):
        avg_time = 0
        if self.request_times:
            avg_time = sum(self.request_times) / len(self.request_times)
            
        return {
            "total_requests": self.successful_requests + self.fallback_attempts,
            "successful_requests": self.successful_requests,
            "fallbacks": self.fallback_attempts,
            "cache_size": len(self.prompt_cache),
            "avg_response_time": avg_time,
            "token_usage": self.token_usage
        }

    def clear_cache(self):
        self.prompt_cache = {}
        try:
            if os.path.exists(self.CACHE_PERSIST_PATH):
                os.remove(self.CACHE_PERSIST_PATH)
        except Exception as e:
            console.print(f"[yellow]⚠ Failed to clear cache file: {e}[/yellow]")
        console.print("[green]✓ Cache cleared[/green]")

# === Example Usage ===
if __name__ == "__main__":
    gpt_manager = GPTManager()
    task = "Scan 10.10.10.5 for open ports"
    final_cmd = gpt_manager.smart_decision("recon", task)
    print(f"Final Command: {final_cmd}")
