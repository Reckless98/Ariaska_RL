import os
import json
import requests
import subprocess
import time
import shutil
import re
from pathlib import Path
from rich.console import Console

console = Console()

def _get_env_or_default(var, default):
    return os.environ.get(var, default)

class LocalLLMManager:
    """
    Local LLM manager for SenecaLLM or LilyLLM via Ollama or compatible API.
    Checks Ollama server/model availability and provides robust fallback.
    
    Attributes:
        model_name (str): Name of the model to use
        host (str): Ollama server host URL
        cache_path (str): Path to cache file
        max_retries (int): Maximum number of retries before fallback
        token_usage (dict): Tracks token usage
        cache (dict): In-memory cache of queries
    """
    def __init__(self, model_name=None, host=None):
        # Allow model name and host to be set via env/config
        self.model_name = model_name or _get_env_or_default("ARIASKA_LOCAL_LLM_MODEL", "wahidmounir/SenecaLLM_x_Qwen2.5-7B-CyberSecurity-Q8_0-GGUF")
        self.host = host or _get_env_or_default("ARIASKA_OLLAMA_HOST", "http://localhost:11434")
        
        # Create directory structure if it doesn't exist
        cache_dir = Path("core/memories")
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = str(cache_dir / "local_llm_cache.json")
        
        # Enhanced configuration
        self.max_retries = 3
        self.timeout = 30  # Default timeout
        self.token_usage = {"total": 0, "seneca": 0, "lily": 0}
        self.last_error = None
        
        # Load cache and check model availability
        self._load_cache()
        self._ensure_ollama_available()
        
    def _load_cache(self):
        """Load query cache from disk with robust error handling."""
        try:
            if os.path.exists(self.cache_path):
                with open(self.cache_path, "r") as f:
                    self.cache = json.load(f)
                console.print(f"[cyan]📂 Loaded LLM cache with {len(self.cache)} entries[/cyan]")
            else:
                self.cache = {}
        except Exception as e:
            console.print(f"[yellow]⚠️ Could not load LLM cache: {e}. Using empty cache.[/yellow]")
            self.cache = {}

    def _save_cache(self):
        """Save query cache to disk with error handling and directory creation."""
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            
            # Write cache to file
            with open(self.cache_path, "w") as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            console.print(f"[yellow]⚠️ Could not save LLM cache: {e}[/yellow]")

    def _ensure_ollama_available(self):
        """Check if Ollama is installed and running, with setup guidance."""
        self.ollama_available = False
        
        # First check if Ollama is installed
        if not shutil.which("ollama"):
            console.print("""[yellow]⚠️ Ollama not found in PATH. To install:
            curl -fsSL https://ollama.com/install.sh | sh
            or visit https://ollama.com for installation instructions.[/yellow]""")
            return
            
        # Then check if Ollama server is running
        try:
            self._check_ollama_server()
            self._check_model_loaded()
            self.ollama_available = True
        except Exception as e:
            self.last_error = str(e)
            console.print(f"[yellow]⚠️ Ollama setup issue: {e}[/yellow]")

    def _check_ollama_server(self):
        """Check if Ollama server is running with enhanced error reporting."""
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            if resp.status_code != 200:
                raise Exception(f"Ollama server not healthy: {resp.status_code}")
                
            # Server is running, log success
            console.print("[green]✅ Ollama server active[/green]")
            
        except requests.exceptions.ConnectionError:
            # Most common error - server not running
            console.print(f"""[red]❌ Ollama server not running at {self.host}
            Please start it with 'ollama serve' in a separate terminal.[/red]""")
            raise RuntimeError(f"Ollama server not running at {self.host}. Start with 'ollama serve'")
            
        except Exception as e:
            # Other network or API errors
            console.print(f"[red]❌ Ollama server error: {e}[/red]")
            raise RuntimeError(f"Ollama server error: {e}")

    def _check_model_loaded(self):
        """Check if the specified model is available in Ollama and pull if missing."""
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            tags = resp.json().get("models", [])
            
            # Extract the base model name (without tags)
            base_model = self.model_name.split(":")[0]
            available = any(base_model in m.get("name", "") for m in tags)
            
            if available:
                console.print(f"[green]✅ Model '{self.model_name}' available[/green]")
            else:
                console.print(f"[yellow]⚠️ Model '{self.model_name}' not loaded. Attempting to pull...[/yellow]")
                
                # More informative pull process
                console.print(f"[cyan]📥 Pulling '{self.model_name}'. This may take several minutes for the first download...[/cyan]")
                result = subprocess.run(["ollama", "pull", self.model_name], 
                                      stdout=subprocess.PIPE, 
                                      stderr=subprocess.PIPE, 
                                      text=True, 
                                      timeout=600)  # 10-minute timeout for large models
                
                # Check pull success
                if result.returncode == 0:
                    console.print(f"[green]✅ Model '{self.model_name}' successfully pulled[/green]")
                else:
                    console.print(f"[red]❌ Failed to pull model: {result.stderr}[/red]")
                    # List available models as fallback suggestion
                    try:
                        available_models = [m.get("name", "") for m in tags if m.get("name")]
                        if available_models:
                            console.print(f"[cyan]ℹ️ Available models: {', '.join(available_models[:5])}{' and more' if len(available_models) > 5 else ''}[/cyan]")
                    except Exception:
                        pass
                    
                    raise RuntimeError(f"Failed to pull model '{self.model_name}': {result.stderr}")
                    
        except Exception as e:
            if "timeout" in str(e).lower():
                console.print(f"[yellow]⚠️ Model check timed out. If this is first run, model may still be downloading.[/yellow]")
            else:
                console.print(f"[red]❌ Model check failed: {e}[/red]")
            raise

    def _extract_command(self, output: str):
        """
        Extract a valid command from LLM output with improved parsing for cybersecurity tools.
        
        Uses multiple strategies in priority order:
        1. Commands in backticks (markdown code format)
        2. Common command prefix patterns 
        3. General command syntax patterns
        4. Single-word valid commands
        
        Args:
            output (str): Raw LLM output text
            
        Returns:
            str: Extracted command
            
        Raises:
            ValueError: If no valid command can be extracted
        """
        if output is None:
            raise ValueError("Empty or invalid LLM output")
            
        # Ensure string format
        if not isinstance(output, str):
            output = str(output)
        
        # Return immediately if output is completely empty after stripping
        output = output.strip()
        if not output:
            raise ValueError("Empty or invalid LLM output")
            
        # 1. First, try to find commands wrapped in backticks (most reliable)
        backtick_matches = re.findall(r'`([^`]+)`', output)
        if backtick_matches:
            # Filter out any that are just explanatory text, not commands
            commands = [cmd.strip() for cmd in backtick_matches 
                      if not re.search(r'^(example|output|this is|result)', cmd.strip(), re.IGNORECASE)]
            if commands:
                return commands[0]  # Return the first valid command
            
        # 2. Look for "The command is" pattern (common in LLM responses)
        cmd_patterns = [
            # Different ways an LLM might introduce a command
            r'(?:The command is|You can use|Try|Run|Execute|Use)(?:\:)?\s*(?:`)?([^`\n\.]+)(?:`)?',
            r'(?:command|syntax)(?:\:)\s*(?:`)?([^`\n\.]+)(?:`)?',
            r'(?:\$|#)\s*([^\n]+)'  # Shell prompt pattern
        ]
        
        for pattern in cmd_patterns:
            cmd_match = re.search(pattern, output, re.IGNORECASE)
            if cmd_match:
                return cmd_match.group(1).strip()
            
        # 3. Look for common cybersecurity tool patterns (comprehensive list)
        command_starts = (
            # Network discovery & scanning
            "nmap", "masscan", "zmap", "rustscan", "autoscan", 
            # Web tools
            "gobuster", "dirb", "nikto", "whatweb", "wfuzz", "ffuf",
            # Authentication attacks
            "hydra", "medusa", "crowbar", "patator", "john", "hashcat",
            # Exploitation
            "msfconsole", "msfvenom", "metasploit", "searchsploit", "exploitdb",
            # Post-exploitation
            "linpeas", "winpeas", "linenum", "pspy", "mimikatz", 
            # SMB/Windows tools
            "smbclient", "smbmap", "enum4linux", "crackmapexec", "evil-winrm", "impacket",
            # Common Unix commands  
            "sudo", "ls", "cat", "grep", "find", "curl", "wget", "ssh", "nc",
            # File operations
            "zip", "tar", "gzip", "cp", "mv", "rm",
            # Network tools
            "dig", "host", "ping", "traceroute", "tcpdump", "wireshark", "tshark"
        )
        
        lines = output.splitlines()
        commands = []
        for line in lines:
            stripped = line.strip()
            if stripped and any(stripped.lower().startswith(cmd.lower()) for cmd in command_starts):
                commands.append(stripped)
                
        if commands:
            return commands[0]  # Return the first command found
            
        # 4. Try to find any line that looks like a command (starts with word + parameters)
        for line in lines:
            stripped = line.strip()
            # Match lines that look like commands (word followed by parameters with flags)
            if re.match(r'^[a-z]\w+\s+(?:(?:-{1,2}\w+(?:=\S+)?)|(?:\S+))+', stripped, re.IGNORECASE):
                return stripped
            
            # Match simpler command patterns too (just word + something)
            if re.match(r'^[a-z]\w+\s+\S+', stripped, re.IGNORECASE):
                return stripped
                
        # 5. Handle single-word valid commands (special case for commands like "ls", "pwd")
        single_word_commands = {"ls", "pwd", "whoami", "id", "ps", "top", "help"}
        for line in lines:
            stripped = line.strip()
            if stripped.lower() in single_word_commands:
                return stripped
                
        # 6. If nothing found, try non-explanatory text as last resort
        for line in lines:
            stripped = line.strip()
            if (stripped and len(stripped) > 3 and 
                not re.search(r'^(i |as an ai|i\'m |i am |i apologize|i cannot|sorry)', stripped.lower())):
                # Exclude very long lines that look like explanations
                if len(stripped) < 100:  # Avoid multi-sentence explanations
                    return stripped
                
        # If we get here, we couldn't find a valid command
        raise ValueError("No actionable command found in LLM output.")

    def _log(self, msg, level="info"):
        """Enhanced logger with color-coding and error level handling."""
        color_map = {
            "info": "cyan",
            "warning": "yellow",
            "error": "red",
            "success": "green"
        }
        color = color_map.get(level, "white")
        console.print(f"[{color}]{msg}[/{color}]")

    def _create_effective_prompt(self, task_type, prompt):
        """
        Create a task-optimized prompt for specific LLM models to improve command extraction.
        
        Args:
            task_type (str): Type of task (recon, exploit, etc.)
            prompt (str): Original prompt
            
        Returns:
            str: Optimized prompt with proper instructions
        """
        # Base prefix that works well with cybersecurity LLMs
        base_prefix = "You are an expert cybersecurity assistant that provides concise, direct commands without explanation. "
        
        # Model-specific handling
        if "lily" in self.model_name.lower():
            # Lily works best with very terse, direct instructions
            prefix = "Return ONLY the exact command for this task. No explanations. Command must be in backticks. "
            
        elif "seneca" in self.model_name.lower():
            # Seneca needs more specific cybersecurity context
            prefix = (f"You are SenecaLLM, a cybersecurity expert. Respond ONLY with the appropriate {task_type} "
                    "command in backtick format. No explanations or commentary. ")
            
        else:
            # Generic instruction for other Ollama models
            prefix = "Respond ONLY with the exact command needed, wrapped in backticks. No explanations. "
        
        # Add task-specific guidance
        task_guidance = {
            "recon": "Provide a reconnaissance command. ",
            "exploit": "Provide an exploit command for the vulnerability. ",
            "enumeration": "Provide an enumeration command to gather more information. ",
            "privesc": "Provide a privilege escalation command for this scenario. ",
            "exfiltrate": "Provide a data exfiltration command. "
        }
        
        task_prefix = task_guidance.get(task_type.lower(), "")
        
        # Construct the final prompt
        return f"{base_prefix}{prefix}{task_prefix}{prompt}"
        
    def query(self, prompt, model=None, task_type="general", timeout=None, allow_fallback=True, retry_with_different_prompt=True):
        """
        Query the local LLM with comprehensive error handling and intelligent fallbacks.
        
        Args:
            prompt (str): Prompt to send to the LLM
            model (str, optional): Model to use (defaults to self.model_name)
            task_type (str): Type of task (recon, exploit, etc.) for prompt optimization
            timeout (int, optional): Query timeout in seconds
            allow_fallback (bool): Whether to allow fallback to GPT
            retry_with_different_prompt (bool): Try different prompting strategies before fallback
            
        Returns:
            str: Parsed command from LLM response
            
        Raises:
            RuntimeError: If query fails and fallback is disabled
        """
        # Use provided model or default
        model = model or self.model_name
        timeout = timeout or self.timeout
        
        # Create cache key from model and prompt
        cache_key = f"{model}|{task_type}|{prompt.strip()[:120]}"
        
        # Return cached response if available
        if cache_key in self.cache:
            self._log(f"🔄 Using cached response for: {prompt[:50]}...", "info")
            return self.cache[cache_key]
            
        # Create an effective prompt with proper instructions for the model
        effective_prompt = self._create_effective_prompt(task_type, prompt)
            
        # Try querying with retries
        tries = 0
        max_tries = self.max_retries
        last_error = None
        prompt_variations = [
            effective_prompt,
            f"Return ONLY the command in backticks: {prompt}",
            f"What's the exact command for: {prompt}"
        ]
        
        while tries < max_tries:
            try:
                # Select the prompt variation based on retry count
                current_prompt = prompt_variations[min(tries, len(prompt_variations)-1)]
                
                # Log the attempt
                self._log(f"🔄 Attempt {tries+1}/{max_tries}: Querying {model}", "info")
                
                # Choose the invocation method based on model type
                if model in ["seneca", "lily"]:
                    # Direct invocation of Seneca/Lily
                    result = subprocess.run(
                        [model], 
                        input=current_prompt, 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE, 
                        text=True, 
                        timeout=timeout
                    )
                else:
                    # Ollama API invocation
                    result = subprocess.run(
                        ["ollama", "run", model], 
                        input=current_prompt, 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE, 
                        text=True, 
                        timeout=timeout
                    )
                
                # Process the output
                raw_output = result.stdout.strip()
                
                # Check if we got any meaningful output
                if not raw_output:
                    raise ValueError(f"Empty output from {model}")
                    
                # Log raw output with length control to avoid console spam
                output_preview = raw_output[:200] + ('...' if len(raw_output) > 200 else '')
                self._log(f"⚡ Raw LLM Output: {output_preview}", "info")
                
                # Extract command using improved parsing
                parsed_command = self._extract_command(raw_output)
                self._log(f"🎯 Parsed Command: {parsed_command}", "success")
                
                # Handle multiple commands returned
                if isinstance(parsed_command, list):
                    parsed_command = parsed_command[0]
                    
                # Validate the command
                if not parsed_command:
                    raise ValueError("Extracted command is empty")
                    
                # More lenient validation - accept commands of any length
                # The old check was: if len(parsed_command.split()) < 2
                # This would reject valid single-word commands like 'ls'
                
                # Cache the successful result
                self.cache[cache_key] = parsed_command
                self._save_cache()
                
                # Track token usage (approximate)
                self.token_usage["total"] += len(effective_prompt.split())
                if "seneca" in model.lower():
                    self.token_usage["seneca"] += 1
                elif "lily" in model.lower():
                    self.token_usage["lily"] += 1
                    
                return parsed_command
                
            except Exception as e:
                last_error = str(e)
                self._log(f"⚠️ Attempt {tries+1} failed: {last_error}", "warning")
                tries += 1
                # Short delay before retry
                time.sleep(0.5)
                
        # If we reach here, all attempts failed
        self._log(f"❌ All {max_tries} attempts failed: {last_error}", "error")
        
        # Fall back to GPT if allowed
        if allow_fallback:
            self._log("🔄 Falling back to GPTManager", "warning")
            try:
                from core.gpt_manager import GPTManager
                # Use a more direct prompt for GPT
                gpt_prompt = f"Provide ONLY the exact command for this cybersecurity task. Command MUST be in backticks: {prompt}"
                response = GPTManager().gpt_request(gpt_prompt, model="gpt-4o-mini", allow_fallback=False)
                
                # Cache the GPT response
                self.cache[cache_key] = response
                self._save_cache()
                return response
                
            except Exception as gpt_error:
                self._log(f"❌ GPT fallback also failed: {gpt_error}", "error")
                raise RuntimeError(f"Local LLM failed: {last_error}, and GPT fallback failed: {gpt_error}")
        else:
            error_msg = f"Local LLM query failed after {max_tries} attempts: {last_error}"
            self._log(f"❌ {error_msg}", "error")
            raise RuntimeError(error_msg)

    def get_token_usage(self):
        """Get token usage statistics."""
        return self.token_usage
        
    def reset_token_usage(self):
        """Reset token usage statistics."""
        self.token_usage = {"total": 0, "seneca": 0, "lily": 0}

class LocalLilyLLMManager(LocalLLMManager):
    """
    LilyLLM manager. Uses same API as LocalLLMManager, but with Lily model.
    Specialized for tactical, concise command generation.
    """
    def __init__(self, host=None):
        super().__init__(model_name=_get_env_or_default("ARIASKA_LILY_MODEL", "QuantFactory/Lily-Cybersecurity-7B-v0.2-GGUF:Q8_0"), host=host)
        # Lily-specific optimizations
        self.timeout = 15  # Lily tends to respond faster

class LocalSenecaLLMManager(LocalLLMManager):
    """
    SenecaLLM manager. Uses same API as LocalLLMManager, but with Seneca model.
    Specialized for strategic reasoning and complex attacks.
    """
    def __init__(self, host=None):
        super().__init__(model_name=_get_env_or_default("ARIASKA_SENECA_MODEL", "wahidmounir/SenecaLLM_x_Qwen2.5-7B-CyberSecurity-Q8_0-GGUF"), host=host)
        # Seneca-specific optimizations
        self.timeout = 25  # Seneca needs more thinking time
