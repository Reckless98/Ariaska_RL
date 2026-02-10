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
    - Strict mode: fail fast if API key missing
    """
    
    # Model configuration
    MODEL_MAP = {
        # Primary models by role — all agents use codex-mini for normal ops
        "red": "gpt-5.1-codex-mini",
        "orion": "gpt-5.1-codex-mini",
        "scout": "gpt-5.1-codex-mini",
        "shadow": "gpt-5.1-codex-mini",
        "blue": "gpt-5.1-codex-mini",
        # Task-based routing
        "tactical": "gpt-5.1-codex-mini",
        "strategic": "gpt-5.2-codex",        # Orion planning + postmortem (shared deep model)
        "reasoning": "gpt-5.1-codex-mini",
        "analysis": "gpt-5.1-codex-mini",
        "classification": "gpt-5.1-codex-mini",
        "embedding": "gpt-5.1-codex-mini",
        "postmortem": "gpt-5.2-codex",       # Deep reasoning for end-of-run analysis
        # Fallbacks
        "general": "gpt-5.1-codex-mini",
        "default": "gpt-5.1-codex-mini",
    }
    
    FALLBACK_MODEL = "gpt-4o-mini"  # Universal fallback

    # Cost per 1K tokens (USD) — approximate, input+output blended average
    COST_PER_1K_TOKENS: Dict[str, float] = {
        "gpt-5-nano": 0.00010,
        "gpt-5-mini": 0.00040,
        "gpt-5.1-codex-mini": 0.00150,
        "gpt-5.1-codex": 0.00600,
        "gpt-5.2-codex": 0.01000,
        "gpt-5.2": 0.01000,
        "gpt-4o-mini": 0.00015,
        "gpt-4o": 0.00250,
    }
    
    def __init__(self, enable_llm: bool = None, require_llm: bool = None, 
                 offline: bool = None):
        """
        Initialize GPTManager.
        
        Args:
            enable_llm: Whether LLM calls are enabled at all (default from runtime_flags)
            require_llm: If True and enable_llm, raise RuntimeError if no API key (default from runtime_flags)
            offline: Force offline mode (no LLM calls, use placeholders) (default from runtime_flags)
        """
        # Import runtime flags for defaults
        from core.runtime_flags import get_runtime_flags
        flags = get_runtime_flags()
        
        # Use explicit args if provided, otherwise fall back to runtime flags
        if offline is None:
            offline = flags.offline if flags.initialized else False
        if enable_llm is None:
            enable_llm = flags.enable_llm if flags.initialized else True
        if require_llm is None:
            require_llm = flags.require_llm if flags.initialized else False  # Default False for backwards compat
        
        self._enable_llm = enable_llm and not offline
        self._require_llm = require_llm
        self._offline = offline
        
        # Lazy init: don't require API key or openai package at construction
        self.api_key = os.getenv("OPENAI_API_KEY")
        self._client = None  # Lazy-initialized OpenAI sync client
        self._async_client = None  # Lazy-initialized OpenAI async client
        
        # Venice AI integration
        self.venice_api_key = os.getenv("VENICE_API_KEY")
        self.venice_base_url = "https://api.venice.ai/api/v1"
        self._venice_client = None  # Lazy-initialized Venice sync client
        self._venice_async_client = None  # Lazy-initialized Venice async client
        self.venice_model = os.getenv("VENICE_MODEL", "venice-uncensored")
        
        # Dual-mentor strategy settings
        self.enable_dual_mentor = os.getenv("ENABLE_DUAL_MENTOR", "true").lower() == "true"
        self.mentor_strategy = os.getenv("MENTOR_STRATEGY", "gpt_first")  # gpt_first, venice_first, round_robin, parallel
        self._mentor_call_count = 0  # For round-robin
        
        # Strict mode validation
        if self._enable_llm and self._require_llm and not self.api_key:
            raise RuntimeError(
                "GPTManager: require_llm=True but OPENAI_API_KEY not set. "
                "Set the environment variable or use offline mode."
            )
        
        # Model configuration from environment or defaults
        self.primary_model = os.getenv("GPT_PRIMARY_MODEL", "gpt-5.1-codex-mini")
        self.fallback_model = os.getenv("GPT_FALLBACK_MODEL", "gpt-4o-mini")
        self.nano_model = os.getenv("GPT_NANO_MODEL", "gpt-5.1-codex-mini")
        self.postmortem_model = os.getenv("GPT_POSTMORTEM_MODEL", "gpt-5.2-codex")
        self.strategic_model = os.getenv("GPT_STRATEGIC_MODEL", "gpt-5.2-codex")
        
        # Feature flags
        self.enable_postmortem_5_2 = True  # Always use deep model for postmortem
        
        # Venice integration stats
        self.venice_enabled = bool(self.venice_api_key) and self.enable_dual_mentor
        self.stats_venice = {
            "total_requests": 0,
            "successes": 0,
            "failures": 0,
            "tokens_used": 0
        }
        
        # Token budgeting - per episode and per agent
        self.token_limit = int(os.getenv("TOKEN_LIMIT_PER_EPISODE", "8000"))
        self.token_limit_per_agent = int(os.getenv("TOKEN_LIMIT_PER_AGENT", "2500"))
        self.tokens_used = 0
        self.tokens_by_agent: Dict[str, int] = {}
        self.current_episode_id: Optional[str] = None
        
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

        # Phase 6.3: Per-model token tracking and cost estimation
        self.tokens_by_model: Dict[str, int] = {}       # model_name → total tokens
        self.requests_by_model: Dict[str, int] = {}     # model_name → request count
        self._cumulative_cost_usd: float = 0.0           # running total $
        self._episode_cost_usd: float = 0.0              # per-episode $
        
        # Load existing cache
        self._load_cache()
        
        logger.debug(f"GPTManager initialized with primary model: {self.primary_model}")
        logger.debug(f"Fallback model: {self.fallback_model}")
        logger.debug(f"Platform detected: {platform.system()}")
        if self.is_configured():
            logger.debug(f"Venice AI enabled: {self.venice_enabled}, Model: {self.venice_model if self.venice_enabled else 'N/A'}")
        else:
            logger.warning("GPTManager: OPENAI_API_KEY not set. LLM calls disabled until configured.")
    
    def is_configured(self) -> bool:
        """Check if API key is available for LLM calls."""
        return bool(self.api_key) and self._enable_llm and not self._offline
    
    def is_offline(self) -> bool:
        """Check if running in offline mode."""
        return self._offline or not self._enable_llm or not self.api_key
    
    def request(
        self,
        role: str,
        task_type: str,
        prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        max_tokens: int = 150
    ) -> Dict[str, Any]:
        """
        Unified request method with role-based routing and token budget tracking.
        
        Args:
            role: Agent role (e.g., "RedAgent", "OrionAgent", "ScoutAgent")
            task_type: Type of task (e.g., "tactical", "reasoning", "postmortem")
            prompt: The prompt to send
            schema: Optional JSON schema for structured output (not yet implemented)
            max_tokens: Maximum tokens in response
            
        Returns:
            Dict with keys:
                - success: bool
                - response: str (the model's response)
                - model_used: str
                - offline: bool
                - tokens: int (estimated tokens used)
                - error: str (if failed)
        """
        # Offline mode returns placeholder (no logging, no API access)
        if self.is_offline() or getattr(self, '_quota_exhausted', False):
            return {
                "success": True,
                "response": self._get_offline_placeholder(task_type),
                "model_used": "offline",
                "offline": True,
                "tokens": 0
            }
        
        # Check budget before making request
        if not self.can_make_request(agent_name=role):
            budget = self.get_budget_status(agent_name=role)
            error_msg = f"Token budget exceeded: total={budget['total_used']}/{budget['total_limit']}"
            if role and budget.get("agent_over_budget"):
                error_msg = f"Agent {role} token budget exceeded: {budget['agent_used']}/{budget['agent_limit']}"
            logger.warning(error_msg)
            return {
                "success": False,
                "response": self._get_offline_placeholder(task_type),
                "model_used": "budget_exceeded",
                "offline": True,
                "tokens": 0,
                "error": error_msg
            }
        
        # Determine model from role and task_type
        model = self.get_model_for_role(agent_id=role, task_type=task_type)
        
        start_time = time.time()
        try:
            # Call gpt_request which handles caching, rate limiting, and API call
            response = self.gpt_request(
                prompt=prompt,
                task_type=task_type,
                agent_id=role,
                max_tokens=max_tokens,
                model=model
            )
            
            # Estimate tokens (rough: prompt words + response words) 
            prompt_tokens = len(prompt.split()) + 50  # ~50 for system prompt
            completion_tokens = len(response.split()) if response else 0
            total_tokens = prompt_tokens + completion_tokens
            
            # Update token counters
            self.tokens_used += total_tokens
            if role:
                self.tokens_by_agent[role] = self.tokens_by_agent.get(role, 0) + total_tokens
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            return {
                "success": True,
                "response": response,
                "model_used": model,
                "offline": False,
                "tokens": total_tokens,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "latency_ms": latency_ms
            }
        except Exception as e:
            logger.error(f"GPT request failed: {e}")
            return {
                "success": False,
                "response": self._get_offline_placeholder(task_type),
                "model_used": "offline",
                "offline": True,
                "tokens": 0,
                "error": str(e)
            }
    
    def _get_offline_placeholder(self, task_type: str) -> str:
        """Get a deterministic placeholder response for offline mode."""
        placeholders = {
            "tactical": "nmap -sV 10.10.10.10",
            "defensive": "netstat -an",
            "reconnaissance": "ping 10.10.10.10",
            "diversify": "nslookup 10.10.10.10",
            "analysis": "Offline mode: analysis unavailable.",
            "reasoning": "Offline mode: reasoning unavailable.",
            "postmortem": "Offline mode: postmortem analysis unavailable.",
            "general": "Offline mode: LLM unavailable."
        }
        return placeholders.get(task_type, placeholders["general"])

    @property
    def client(self):
        """Lazy-initialize OpenAI sync client on first use."""
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

    @property
    def async_client(self):
        """Lazy-initialize OpenAI async client on first use."""
        if self._async_client is None:
            if not OPENAI_AVAILABLE:
                raise RuntimeError(
                    "OpenAI library not installed. Install with: pip install openai"
                )
            if not self.api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY not set. Set the environment variable or use offline mode."
                )
            from openai import AsyncOpenAI
            self._async_client = AsyncOpenAI(api_key=self.api_key)
        return self._async_client

    @property
    def venice_client(self):
        """Lazy-initialize Venice AI sync client on first use."""
        if self._venice_client is None:
            if not OPENAI_AVAILABLE or OpenAI is None:
                raise RuntimeError(
                    "OpenAI library not installed. Venice uses OpenAI-compatible API. Install with: pip install openai"
                )
            if not self.venice_api_key:
                raise RuntimeError(
                    "VENICE_API_KEY not set. Set the environment variable to use Venice AI."
                )
            self._venice_client = OpenAI(
                api_key=self.venice_api_key,
                base_url=self.venice_base_url
            )
            logger.info(f"Venice AI client initialized | Model: {self.venice_model}")
        return self._venice_client

    @property
    def venice_async_client(self):
        """Lazy-initialize Venice AI async client on first use."""
        if self._venice_async_client is None:
            if not OPENAI_AVAILABLE:
                raise RuntimeError(
                    "OpenAI library not installed. Venice uses OpenAI-compatible API. Install with: pip install openai"
                )
            if not self.venice_api_key:
                raise RuntimeError(
                    "VENICE_API_KEY not set. Set the environment variable to use Venice AI."
                )
            from openai import AsyncOpenAI
            self._venice_async_client = AsyncOpenAI(
                api_key=self.venice_api_key,
                base_url=self.venice_base_url
            )
            logger.info(f"Venice AI async client initialized | Model: {self.venice_model}")
        return self._venice_async_client

    def has_venice(self) -> bool:
        """Check if Venice AI is available."""
        return bool(self.venice_api_key) and self.venice_enabled

    def get_next_mentor_provider(self) -> str:
        """
        Get the next mentor provider based on strategy.
        
        Strategies:
        - gpt_first: Always GPT, Venice as fallback
        - venice_first: Always Venice, GPT as fallback
        - round_robin: Alternate between GPT and Venice
        - parallel: Use both (caller handles)
        
        Returns:
            str: "gpt" or "venice"
        """
        if not self.has_venice():
            return "gpt"
        
        if self.mentor_strategy == "venice_first":
            return "venice"
        elif self.mentor_strategy == "round_robin":
            self._mentor_call_count += 1
            return "venice" if self._mentor_call_count % 2 == 0 else "gpt"
        elif self.mentor_strategy == "parallel":
            return "both"
        else:  # gpt_first (default)
            return "gpt"

    def get_mentor_clients(self) -> Dict[str, Any]:
        """
        Get mentor clients based on availability.
        
        Returns:
            Dict with 'primary', 'secondary', and their async variants
        """
        result = {
            "primary": None,
            "primary_async": None,
            "primary_model": None,
            "secondary": None,
            "secondary_async": None,
            "secondary_model": None,
            "strategy": self.mentor_strategy
        }
        
        # Primary is always GPT (if available)
        if self.api_key and self._enable_llm:
            try:
                result["primary"] = self.client
                result["primary_async"] = self.async_client
                result["primary_model"] = self.primary_model
            except RuntimeError:
                pass
        
        # Secondary is Venice (if available)
        if self.has_venice():
            try:
                result["secondary"] = self.venice_client
                result["secondary_async"] = self.venice_async_client
                result["secondary_model"] = self.venice_model
            except RuntimeError:
                pass
        
        return result
    
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
            return self.postmortem_model
        
        # Strategic: Orion planning + deep reasoning (gpt-5.2-codex, shared with postmortem)
        if task_type == "strategic":
            return self.strategic_model
        
        # Everything else uses primary (codex-mini)
        return self.primary_model
    
    def get_model_for_task(self, task_type: str) -> str:
        """
        Get appropriate model for a task type.
        Alias for get_model_for_role with task_type mapping.
        
        Args:
            task_type: Type of task (e.g., "tactical", "reasoning", "postmortem")
            
        Returns:
            str: Model name to use
        """
        return self.get_model_for_role(agent_id=None, task_type=task_type)
    
    def reset_episode(self, episode_id: Optional[int] = None, agent_name: Optional[str] = None) -> None:
        """
        Reset token counters for a new episode.
        
        Args:
            episode_id: Episode identifier (for logging)
            agent_name: If provided, reset only that agent's counter. Otherwise reset all.
        """
        if agent_name:
            self.tokens_by_agent[agent_name] = 0
            logger.debug(f"Reset token count for {agent_name} (episode {episode_id})")
        else:
            self.tokens_used = 0
            self.tokens_by_agent = {}
            self.current_episode_id = str(episode_id) if episode_id is not None else None
            logger.debug(f"Reset all token counts for episode {episode_id}")
    
    def reset_token_count(self):
        """Reset token count for new episode (legacy method, use reset_episode)"""
        self.reset_episode()
    
    def can_make_request(self, agent_name: Optional[str] = None) -> bool:
        """
        Check if we can make another request within token limits.
        
        Args:
            agent_name: If provided, check per-agent limit too.
            
        Returns:
            bool: True if request is within budget.
        """
        if self.tokens_used >= self.token_limit:
            return False
        if agent_name and self.tokens_by_agent.get(agent_name, 0) >= self.token_limit_per_agent:
            return False
        return True
    
    def get_token_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive token usage statistics.
        
        Returns dict with:
            total_used, total_limit, by_agent, stats
        """
        return {
            "total_used": self.tokens_used,
            "total_limit": self.token_limit,
            "remaining": max(0, self.token_limit - self.tokens_used),
            "by_agent": dict(self.tokens_by_agent),
            "stats": {
                "total_requests": self.stats.get("total_requests", 0),
                "tokens_used_total": self.stats.get("tokens_used_total", 0),
            }
        }
    
    def reset_episode_tokens(self):
        """Reset per-episode token counters (call at episode start)."""
        self.tokens_used = 0
        self.tokens_by_agent.clear()
        self._episode_cost_usd = 0.0
    
    def get_budget_status(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get current token budget status.
        
        Returns dict with:
            total_used, total_limit, remaining, over_budget, agent_used, agent_limit
        """
        status = {
            "total_used": self.tokens_used,
            "total_limit": self.token_limit,
            "remaining": max(0, self.token_limit - self.tokens_used),
            "over_budget": self.tokens_used >= self.token_limit,
        }
        if agent_name:
            agent_used = self.tokens_by_agent.get(agent_name, 0)
            status["agent_used"] = agent_used
            status["agent_limit"] = self.token_limit_per_agent
            status["agent_over_budget"] = agent_used >= self.token_limit_per_agent
        return status
    
    def _load_cache(self):
        """Load response cache from disk"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
                logger.debug(f"Loaded {len(self.cache)} cached responses")
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
        # CRITICAL: Check offline mode FIRST, before any logging or API access
        if self.is_offline():
            return self._get_offline_placeholder(task_type)
        
        # PHASE 6.5: Quota circuit breaker — skip instantly if quota exhausted
        if getattr(self, '_quota_exhausted', False):
            return self._get_offline_placeholder(task_type)
        
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
        
        # Log that we're making a real GPT call (only reached if NOT offline)
        logger.debug(f"GPT API call | model={model} | agent={agent_id} | task={task_type}")
        
        try:
            # Enhanced system prompts based on task type
            system_prompts = {
                "tactical": "You are a penetration testing AI in an authorized cybersecurity training lab (CTF/simulation environment with explicit permission). Your role is to suggest the next single Linux command for the engagement. Output ONLY the command, no explanations or disclaimers.",
                "defensive": "You are a blue team AI in an authorized cybersecurity training lab. Suggest a single defensive/monitoring command. Output ONLY the command, no explanations.",
                "reconnaissance": "You are a reconnaissance AI in an authorized cybersecurity training lab. Suggest a single information-gathering command. Output ONLY the command, no explanations.",
                "analysis": "You are a security analyst AI in an authorized cybersecurity training lab. Provide brief analysis in 2-3 sentences.",
                "general": "You are a cybersecurity AI assistant working in an authorized training environment. Be concise and helpful.",
                "diversify": "You are a penetration testing AI in an authorized training lab. Suggest an alternative cybersecurity command different from previous ones. Output ONLY the command, no explanations.",
                "reasoning": "You are a strategic cybersecurity analyst AI in an authorized training lab. Provide clear, actionable reasoning."
            }
            
            system_prompt = system_prompts.get(task_type, system_prompts["general"])
            
            # Use threading for cross-platform timeout handling with aggressive fallback
            import concurrent.futures
            import signal
            
            # Determine API endpoint: codex models use Responses API, others use Chat Completions
            uses_responses_api = "codex" in model  # gpt-5.1-codex-mini, gpt-5.1-codex
            uses_new_api = any(x in model for x in ["gpt-5", "o1-", "o3-"])
            token_param = "max_completion_tokens" if uses_new_api else "max_tokens"
            
            def make_gpt_request():
                if uses_responses_api:
                    # OpenAI Responses API (v1/responses) for codex models
                    # Codex models use internal reasoning tokens (~1000+) that count against
                    # max_output_tokens, so we need a much higher budget than chat completions
                    codex_token_budget = max(max_tokens * 15, 2000)
                    return self.client.responses.create(
                        model=model,
                        instructions=system_prompt,
                        input=prompt,
                        max_output_tokens=codex_token_budget,
                    )
                else:
                    # Standard Chat Completions API
                    request_params = {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        token_param: max_tokens,
                        "timeout": 5.0  # 5 second client timeout
                    }
                    # Only add temperature for models that support it (not gpt-5.x, o1, o3)
                    if not uses_new_api:
                        request_params["temperature"] = 0.7 if task_type == "diversify" else 0.3
                    return self.client.chat.completions.create(**request_params)
            
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
                # PHASE 6.5: Activate circuit breaker on quota exhaustion
                error_str = str(e).lower()
                if "insufficient_quota" in error_str or "exceeded your current quota" in error_str:
                    self._quota_exhausted = True
                    logger.warning("⚡ GPT quota exhausted — circuit breaker activated, all future LLM calls skipped this session")
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
            
            # Extract content — Responses API uses .output_text, Chat Completions uses .choices[]
            if uses_responses_api:
                content = getattr(response, 'output_text', '') or ''
                content = content.strip()
            elif response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                if content:
                    content = content.strip()
                else:
                    content = ""
            else:
                content = ""
                
            if content:
                # Track tokens
                if hasattr(response, 'usage') and response.usage:
                    tokens_used = response.usage.total_tokens
                    self.tokens_used += tokens_used
                    self.stats["tokens_used_total"] += tokens_used
                    # Phase 6.3: per-model tracking + cost
                    self.tokens_by_model[model] = self.tokens_by_model.get(model, 0) + tokens_used
                    self.requests_by_model[model] = self.requests_by_model.get(model, 0) + 1
                    cost_rate = self.COST_PER_1K_TOKENS.get(model, 0.001)
                    step_cost = (tokens_used / 1000.0) * cost_rate
                    self._cumulative_cost_usd += step_cost
                    self._episode_cost_usd += step_cost
                
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
        """Get usage statistics including cost tracking."""
        return {
            **self.stats,
            "tokens_used_current_episode": self.tokens_used,
            "cache_size": len(self.cache),
            "cumulative_cost_usd": round(self._cumulative_cost_usd, 6),
            "episode_cost_usd": round(self._episode_cost_usd, 6),
            "tokens_by_model": dict(self.tokens_by_model),
            "requests_by_model": dict(self.requests_by_model),
        }

    def get_cost_summary(self) -> Dict[str, Any]:
        """Get detailed cost breakdown by model.
        
        Returns:
            Dict with cumulative_usd, episode_usd, and per-model breakdown.
        """
        breakdown = {}
        for model_name, tok_count in self.tokens_by_model.items():
            rate = self.COST_PER_1K_TOKENS.get(model_name, 0.001)
            cost = (tok_count / 1000.0) * rate
            breakdown[model_name] = {
                "tokens": tok_count,
                "requests": self.requests_by_model.get(model_name, 0),
                "cost_usd": round(cost, 6),
                "rate_per_1k": rate,
            }
        return {
            "cumulative_usd": round(self._cumulative_cost_usd, 6),
            "episode_usd": round(self._episode_cost_usd, 6),
            "models": breakdown,
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
