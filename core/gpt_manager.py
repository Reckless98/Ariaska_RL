# core/gpt_manager.py — ARIASKA GPTManager v5.0 APEX (GPT-5-mini + Multi-Model Routing)
# Centralized LLM Gateway: Role-Based Routing, Cross-Platform, Learning-Enhanced
# Models: GPT-5-mini (primary), GPT-4o-mini (fallback), GPT-5-nano (lightweight), GPT-5.2 (postmortem)

import os
import logging
from typing import Dict, Any, Optional, List
import time
import json
import platform
import subprocess
import shlex
import hashlib
import threading
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If python-dotenv is not available, try to load manually
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                if '=' in line and not line.strip().startswith('#'):
                    key, value = line.strip().split('=', 1)
                    value = value.strip('"\'')
                    os.environ[key] = value

try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI library not available. Install with: pip install openai")
    OPENAI_AVAILABLE = False

try:
    from rich.console import Console
    console = Console()
except ImportError:
    console = None

logger = logging.getLogger(__name__)

class PlatformUtils:
    """Cross-platform utilities for Windows and Linux compatibility"""
    
    @staticmethod
    def is_windows() -> bool:
        return platform.system().lower() == "windows"
    
    @staticmethod
    def is_linux() -> bool:
        return platform.system().lower() == "linux"
    
    @staticmethod
    def translate_command(command: str) -> str:
        """Translate Linux commands to Windows equivalents when needed"""
        if not PlatformUtils.is_windows():
            return command
        
        # Command translation mappings
        translations = {
            # Network commands
            "netstat -tulnp": "netstat -an",
            "ss -tulnp": "netstat -an",
            "ifconfig": "ipconfig",
            "ip addr": "ipconfig /all",
            
            # Process commands  
            "ps aux": "tasklist",
            "ps -ef": "tasklist /v",
            "kill -9": "taskkill /F /PID",
            "killall": "taskkill /F /IM",
            
            # File commands
            "ls -la": "dir",
            "ls": "dir",
            "cat": "type",
            "grep": "findstr",
            "which": "where",
            "chmod": "attrib",
            
            # Network tools
            "wget": "curl",
            
            # Service commands
            "systemctl": "sc",
            "service": "net",
        }
        
        # Apply translations
        for linux_cmd, windows_cmd in translations.items():
            if command.startswith(linux_cmd):
                translated = command.replace(linux_cmd, windows_cmd, 1)
                logger.debug(f"Translated command: {command} -> {translated}")
                return translated
        
        return command
    
    @staticmethod
    def execute_command(command: str, timeout: int = 30, 
                       working_dir: Optional[str] = None) -> tuple:
        """Execute command with platform-specific handling"""
        
        # Translate command if needed
        translated_command = PlatformUtils.translate_command(command)
        
        try:
            # Use shell=True for Windows compatibility
            shell = True
            
            # On Unix systems, properly split the command for security
            if not PlatformUtils.is_windows():
                try:
                    # Try to use shlex for proper splitting
                    cmd_args = shlex.split(translated_command)
                    shell = False
                except ValueError:
                    # Fall back to shell=True if shlex fails
                    cmd_args = translated_command
                    shell = True
            else:
                cmd_args = translated_command
            
            result = subprocess.run(
                cmd_args,
                shell=shell,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=working_dir,
                encoding='utf-8',
                errors='replace'  # Handle encoding issues gracefully
            )
            
            return result.stdout, result.stderr, result.returncode
            
        except subprocess.TimeoutExpired:
            return "", f"Command timed out after {timeout} seconds", 124
        except FileNotFoundError as e:
            return "", f"Command not found: {e}", 127
        except Exception as e:
            return "", f"Execution error: {e}", 1

class GPTManager:
    """
    Centralized LLM manager for all agents with role-based model routing.
    
    Model Routing:
    - GPT-5-mini: Primary model for Red/Orion agents (strategy/tactics)
    - GPT-4o-mini: Fallback model when GPT-5-mini unavailable
    - GPT-5-nano: Lightweight model for Scout/Shadow/Blue (classification/rewrites)
    - GPT-5.2: Deep reasoning for end-of-run postmortem (feature-flagged)
    
    Features:
    - Role-based automatic routing
    - Response caching by state fingerprint
    - Token tracking and limits
    - Cross-platform command translation
    """
    
    # Model configuration
    MODEL_MAP = {
        # Primary models by role
        "red": "gpt-5-mini",
        "orion": "gpt-5-mini", 
        "scout": "gpt-5-nano",
        "shadow": "gpt-5-nano",
        "blue": "gpt-5-nano",
        # Task-based routing
        "tactical": "gpt-5-mini",
        "strategic": "gpt-5-mini",
        "reasoning": "gpt-5-mini",
        "analysis": "gpt-5-nano",
        "classification": "gpt-5-nano",
        "embedding": "gpt-5-nano",
        "postmortem": "gpt-5.2",  # Feature-flagged deep reasoning
        # Fallbacks
        "general": "gpt-5-mini",
        "default": "gpt-5-mini",
    }
    
    FALLBACK_MODEL = "gpt-4o-mini"  # Universal fallback
    
    def __init__(self):
        # Lazy init: don't require API key or openai package at construction
        self.api_key = os.getenv("OPENAI_API_KEY")
        self._client = None  # Lazy-initialized OpenAI client
        
        # Model configuration from environment or defaults
        self.primary_model = os.getenv("GPT_PRIMARY_MODEL", "gpt-5-mini")
        self.fallback_model = os.getenv("GPT_FALLBACK_MODEL", "gpt-4o-mini")
        self.nano_model = os.getenv("GPT_NANO_MODEL", "gpt-5-nano")
        self.postmortem_model = os.getenv("GPT_POSTMORTEM_MODEL", "gpt-5.2")
        
        # Feature flags
        self.enable_postmortem_5_2 = os.getenv("ENABLE_GPT_5_2_POSTMORTEM", "false").lower() == "true"
        
        self.token_limit = int(os.getenv("TOKEN_LIMIT_PER_EPISODE", "3000"))
        self.tokens_used = 0
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
        # Cross-platform detection
        self.is_windows = platform.system().lower() == "windows"
        
        # Cache for responses
        self.cache_dir = Path("core/memories")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "gpt_cache.json"
        self.cache = {}
        self.cache_lock = threading.Lock()
        
        # Learning storage for agent feedback
        self.learning_feedback = {}
        self.command_history = {}
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "failures": 0,
            "tokens_used_total": 0
        }
        
        # Load existing cache
        self._load_cache()
        
        logger.info(f"GPTManager initialized with primary model: {self.primary_model}")
        logger.info(f"Fallback model: {self.fallback_model}")
        logger.info(f"Platform detected: {platform.system()}")
        if self.is_configured():
            if console:
                console.print(f"[green]GPTManager v5.0 initialized | Primary: {self.primary_model} | Fallback: {self.fallback_model}[/green]")
        else:
            logger.warning("GPTManager: OPENAI_API_KEY not set. LLM calls disabled until configured.")
    
    def is_configured(self) -> bool:
        """Check if API key is available for LLM calls."""
        return bool(self.api_key)
    
    @property
    def client(self):
        """Lazy-initialize OpenAI client on first use."""
        if self._client is None:
            if not OPENAI_AVAILABLE or OpenAI is None:
                raise RuntimeError(
                    "OpenAI library not installed. Install with: pip install openai"
                )
            if not self.api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY not set. Set the environment variable or use offline mode."
                )
            self._client = OpenAI(api_key=self.api_key)
        return self._client
    
    def get_model_for_role(self, agent_id: str = None, task_type: str = None) -> str:
        """
        Get appropriate model based on agent role and task type.
        
        Role-based routing:
        - Red/Orion agents -> GPT-5-mini (tactical/strategic)
        - Scout/Shadow/Blue -> GPT-5-nano (lightweight classification)
        - Postmortem tasks -> GPT-5.2 (deep reasoning, if enabled)
        
        Args:
            agent_id: Agent identifier (e.g., "RedAgent", "ScoutAgent")
            task_type: Type of task (e.g., "tactical", "analysis", "postmortem")
            
        Returns:
            str: Model name to use
        """
        # Extract role from agent_id
        role = None
        if agent_id:
            agent_lower = agent_id.lower()
            if "red" in agent_lower:
                role = "red"
            elif "orion" in agent_lower:
                role = "orion"
            elif "scout" in agent_lower:
                role = "scout"
            elif "shadow" in agent_lower:
                role = "shadow"
            elif "blue" in agent_lower:
                role = "blue"
        
        # Special handling for postmortem
        if task_type == "postmortem":
            if self.enable_postmortem_5_2:
                return self.postmortem_model
            else:
                return self.primary_model  # Fall back to primary
        
        # Role-based selection
        if role:
            model = self.MODEL_MAP.get(role, self.primary_model)
            # Map to actual configured models
            if model == "gpt-5-mini":
                return self.primary_model
            elif model == "gpt-5-nano":
                return self.nano_model
            elif model == "gpt-5.2":
                return self.postmortem_model
        
        # Task-type based selection
        if task_type:
            model = self.MODEL_MAP.get(task_type, self.primary_model)
            if model == "gpt-5-mini":
                return self.primary_model
            elif model == "gpt-5-nano":
                return self.nano_model
        
        return self.primary_model
    
    def reset_token_count(self):
        """Reset token count for new episode"""
        self.tokens_used = 0
    
    def can_make_request(self) -> bool:
        """Check if we can make another request within token limits"""
        return self.tokens_used < self.token_limit
    
    def _load_cache(self):
        """Load response cache from disk"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
                logger.info(f"Loaded {len(self.cache)} cached responses")
        except Exception as e:
            logger.warning(f"Could not load cache: {e}")
            self.cache = {}
    
    def _save_cache(self):
        """Save response cache to disk"""
        try:
            with self.cache_lock:
                with open(self.cache_file, 'w') as f:
                    json.dump(self.cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save cache: {e}")
    
    def _sanitize_command(self, command: str) -> str:
        """Sanitize GPT output for security"""
        if not command or not isinstance(command, str):
            return ""
        
        # Remove dangerous patterns
        dangerous_patterns = [
            "rm -rf", "del /s", "format c:", "shutdown", "reboot",
            "dd if=", "mkfs", "fdisk", "> /dev/", ":(){ :|:& };:",
            "sudo rm", "rm -r", "del /q"
        ]
        
        for pattern in dangerous_patterns:
            if pattern.lower() in command.lower():
                logger.warning(f"Blocked dangerous command pattern: {pattern}")
                return "echo 'Command blocked for safety'"
        
        # Extract command from backticks or quotes
        import re
        
        # Try to extract from backticks first
        backtick_match = re.search(r'`([^`]+)`', command)
        if backtick_match:
            command = backtick_match.group(1)
        
        # Try to extract from quotes
        quote_match = re.search(r'"([^"]+)"', command)
        if quote_match and not backtick_match:
            command = quote_match.group(1)
        
        return command.strip()
    
    def _create_cache_key(self, prompt: str, task_type: str, agent_id: str) -> str:
        """Create a cache key for the request"""
        content = f"{task_type}|{agent_id}|{prompt[:100]}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def gpt_request(self, prompt: str, task_type: str = "general", 
                   agent_id: str = "unknown", max_tokens: int = 150,
                   model: Optional[str] = None, allow_fallback: bool = True) -> str:
        """
        Make a request to GPT with role-based model routing.
        
        Args:
            prompt: The prompt to send
            task_type: Type of task (tactical, analysis, postmortem, etc.)
            agent_id: Agent making the request (used for model routing)
            max_tokens: Maximum tokens in response
            model: Optional explicit model override
            allow_fallback: Whether to allow fallback to backup model
            
        Returns:
            str: The model's response
        """
        
        if not self.can_make_request():
            logger.warning(f"Token limit reached for episode ({self.tokens_used}/{self.token_limit})")
            return "echo 'Token limit reached'"
        
        # Role-based model selection (unless explicitly overridden)
        if model is None:
            model = self.get_model_for_role(agent_id=agent_id, task_type=task_type)
        
        # Log model selection for debugging
        logger.debug(f"Model selected: {model} for agent={agent_id}, task={task_type}")
        
        # Create cache key
        cache_key = self._create_cache_key(prompt, task_type, agent_id)
        
        # Check cache first
        with self.cache_lock:
            if cache_key in self.cache:
                self.stats["cache_hits"] += 1
                return self.cache[cache_key]["response"]
        
        # Rate limiting
        current_time = time.time()
        if current_time - self.last_request_time < self.min_request_interval:
            time.sleep(self.min_request_interval)
        
        try:
            # Enhanced system prompts based on task type
            system_prompts = {
                "tactical": "You are a cybersecurity expert. Provide a single, safe Linux command for penetration testing. No explanations, just the command.",
                "defensive": "You are a blue team expert. Provide a single, safe defensive command for system monitoring. No explanations, just the command.",
                "reconnaissance": "You are a reconnaissance expert. Provide a single, safe information gathering command. No explanations, just the command.",
                "analysis": "You are a security analyst. Provide brief analysis in 2-3 sentences.",
                "general": "You are a cybersecurity AI assistant. Be concise and helpful.",
                "diversify": "You are an expert at creating alternative cybersecurity commands. Provide only the command, no explanations.",
                "reasoning": "You are a strategic cybersecurity analyst. Provide clear, actionable reasoning."
            }
            
            system_prompt = system_prompts.get(task_type, system_prompts["general"])
            
            # Use threading for cross-platform timeout handling with aggressive fallback
            import concurrent.futures
            import signal
            
            def make_gpt_request():
                # Use existing client with shorter timeout
                return self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=0.7 if task_type == "diversify" else 0.3,
                    timeout=5.0  # 5 second client timeout
                )
            
            # Execute with aggressive timeout using ThreadPoolExecutor
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(make_gpt_request)
                    try:
                        response = future.result(timeout=8)  # 8 second overall timeout
                    except concurrent.futures.TimeoutError:
                        logger.warning(f"GPT request timed out after 8 seconds for {agent_id}, using fallback")
                        # Return immediate fallback command based on task type
                        fallback_commands = {
                            "tactical": "nmap -sV 10.10.10.10",
                            "defensive": "netstat -an",
                            "reconnaissance": "ping 10.10.10.10",
                            "diversify": "nslookup 10.10.10.10",
                            "general": "echo 'GPT timeout - using fallback'"
                        }
                        return fallback_commands.get(task_type, "echo 'GPT timeout'")
            except Exception as e:
                logger.warning(f"GPT request failed with exception: {e}, using immediate fallback")
                # Immediate fallback for any network issues
                fallback_commands = {
                    "tactical": "nmap -sV 10.10.10.10",
                    "defensive": "netstat -an", 
                    "reconnaissance": "ping 10.10.10.10",
                    "diversify": "nslookup 10.10.10.10",
                    "general": "echo 'GPT error - using fallback'"
                }
                return fallback_commands.get(task_type, "echo 'GPT error'")
            
            self.last_request_time = time.time()
            self.stats["total_requests"] += 1
            
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                if content:
                    content = content.strip()
                else:
                    content = ""
                
                # Track tokens
                if hasattr(response, 'usage') and response.usage:
                    tokens_used = response.usage.total_tokens
                    self.tokens_used += tokens_used
                    self.stats["tokens_used_total"] += tokens_used
                
                # Sanitize if it's a command
                if task_type in ["tactical", "defensive", "reconnaissance", "diversify"]:
                    content = self._sanitize_command(content)
                
                # Cache the response
                with self.cache_lock:
                    self.cache[cache_key] = {
                        "response": content,
                        "timestamp": time.time(),
                        "agent_id": agent_id,
                        "task_type": task_type
                    }
                    
                    # Save cache periodically
                    if len(self.cache) % 10 == 0:
                        threading.Thread(target=self._save_cache).start()
                
                logger.debug(f"GPT response for {agent_id}: {content[:50]}...")
                return content
            else:
                logger.error("Empty response from GPT")
                return "echo 'No response from GPT'"
                
        except Exception as e:
            logger.error(f"GPT request failed for {agent_id}: {e}")
            self.stats["failures"] += 1
            return f"echo 'GPT error: {str(e)[:50]}'"
    
    def smart_decision(self, task_type: str, task_description: str, 
                      agent_id: str = "unknown", use_gpt: bool = True) -> str:
        """Enhanced decision making with context awareness"""
        
        # Build enhanced prompt with context
        enhanced_prompt = f"""
        Task Type: {task_type}
        Description: {task_description}
        
        Provide the most appropriate cybersecurity command for this situation.
        Consider the task type and provide only the command, no explanations.
        """
        
        return self.gpt_request(
            enhanced_prompt,
            task_type=task_type,
            agent_id=agent_id,
            max_tokens=100
        )
    
    def get_learning_feedback(self, command: str, result: str, reward: float, 
                            agent_id: str) -> str:
        """Get learning feedback to help agents improve"""
        prompt = f"""
        Command executed: {command}
        Result: {result[:200]}
        Reward received: {reward}
        
        Provide 1-2 sentences on what went well or what could be improved.
        Focus on tactical improvements for future commands.
        """
        
        return self.gpt_request(prompt, "analysis", agent_id, max_tokens=100)
    
    def get_training_hint(self, phase: str, previous_commands: list, 
                         agent_id: str) -> str:
        """Get training hints for agents during learning"""
        recent_commands = previous_commands[-3:] if previous_commands else []
        
        prompt = f"""
        Current phase: {phase}
        Recent commands: {recent_commands}
        
        Suggest ONE cybersecurity command for this phase. Only the command, no explanation.
        Consider what commands were recently used and suggest something different but appropriate.
        """
        
        return self.gpt_request(prompt, "tactical", agent_id, max_tokens=50)
    
    def dual_llm_feedback(self, prompt: str, agent_id: str = "unknown", 
                         task_type: str = "tactical") -> str:
        """
        Simplified dual feedback - just use GPT-4o-mini for everything
        This maintains API compatibility with old dual_llm_feedback calls
        """
        return self.gpt_request(prompt, task_type, agent_id)
    
    def get_token_usage(self) -> int:
        """Get current token usage for this episode"""
        return self.tokens_used
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            **self.stats,
            "tokens_used_current_episode": self.tokens_used,
            "cache_size": len(self.cache)
        }
    
    def store_learning_feedback(self, agent_id: str, command: str, 
                              feedback: str, reward: float):
        """Store learning feedback for future reference"""
        if agent_id not in self.learning_feedback:
            self.learning_feedback[agent_id] = []
        
        self.learning_feedback[agent_id].append({
            "command": command,
            "feedback": feedback,
            "reward": reward,
            "timestamp": time.time()
        })
        
        # Keep only recent feedback
        if len(self.learning_feedback[agent_id]) > 50:
            self.learning_feedback[agent_id] = self.learning_feedback[agent_id][-50:]
    
    def get_learning_context(self, agent_id: str) -> str:
        """Get learning context for an agent"""
        if agent_id not in self.learning_feedback:
            return "No previous learning feedback available."
        
        recent_feedback = self.learning_feedback[agent_id][-3:]
        if not recent_feedback:
            return "No recent learning feedback available."
        
        context_parts = []
        for fb in recent_feedback:
            context_parts.append(f"Command: {fb['command']}, Result: {fb['feedback']}")
        
        return "Recent learning: " + " | ".join(context_parts)
    
    def _is_simple_command(self, command: str) -> bool:
        """Check if command is simple enough to use directly"""
        if not command or len(command.split()) < 2:
            return False
        
        # Check if it starts with known command words
        known_commands = [
            "nmap", "nc", "ssh", "telnet", "ftp", "curl", "wget",
            "gobuster", "dirb", "nikto", "hydra", "john", "hashcat",
            "msfconsole", "searchsploit", "ls", "cat", "grep", "find"
        ]
        
        first_word = command.split()[0].lower()
        return first_word in known_commands
    
    def _sanitize_output(self, output: str) -> str:
        """Sanitize output for safety"""
        return self._sanitize_command(output)
    
    def test_connectivity(self) -> dict:
        """Test GPT connectivity and return status"""
        try:
            test_prompt = "Respond with 'ARIASKA GPT-4o-mini is operational'"
            response = self.gpt_request(test_prompt, "general", "test", max_tokens=20)
            
            return {
                "status": "success",
                "response": response,
                "model": self.primary_model,
                "platform": platform.system()
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "model": self.primary_model,
                "platform": platform.system()
            }
    
    def cleanup(self):
        """Clean shutdown - save cache and stats"""
        self._save_cache()
        logger.info("GPTManager cleaned up successfully")

# Singleton instance
_gpt_manager_instance = None

def get_gpt_manager() -> GPTManager:
    """Get singleton GPTManager instance"""
    global _gpt_manager_instance
    if _gpt_manager_instance is None:
        _gpt_manager_instance = GPTManager()
    return _gpt_manager_instance
