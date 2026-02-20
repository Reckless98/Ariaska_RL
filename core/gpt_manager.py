# core/gpt_manager.py — ARIASKA GPTManager v6.0 (GPT-5.2-codex + Mini + Nano)
# Centralized LLM Gateway: Role-Based Routing, Cross-Platform, Learning-Enhanced
# Models: gpt-5.2-codex (primary), gpt-5.2-mini (parsing/fallback), gpt-5-nano (lightweight)

import os
import logging
from collections import deque
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
    
    Model Routing (Phase 36 — codex-primary):
    - gpt-5.2-codex: All reasoning — strategic, tactical, analysis, defensive, recon, postmortem
    - gpt-5.2-mini: Parsing, output interpretation, playbook selection, structured extraction
    - gpt-5-nano: Lightweight classification, reformatting, boolean checks
    
    Features:
    - Role-based automatic routing
    - Response caching by state fingerprint
    - Token tracking and limits
    - Cross-platform command translation
    - Strict mode: fail fast if API key missing
    """
    
    # Model configuration — Phase 12.1: All reasoning tasks → gpt-5.2-codex
    # ── Model routing constants (Phase 38: centralized, no hardcoded strings) ──
    MODEL_CODEX = "gpt-5.2-codex"       # Primary decision model
    MODEL_MINI  = "gpt-5.2-mini"        # Free-text / fallback
    MODEL_NANO  = "gpt-5-nano"          # Classification / lightweight parse

    # This map is kept for test compatibility. Actual routing is in get_model_for_role().
    MODEL_MAP = {
        # All agents use gpt-5.2-codex for reasoning/mentor tasks
        "red": "gpt-5.2-codex",
        "orion": "gpt-5.2-codex",
        "scout": "gpt-5.2-codex",
        "shadow": "gpt-5.2-codex",
        "blue": "gpt-5.2-codex",
        # Task-based routing
        "tactical": "gpt-5.2-codex",
        "strategic": "gpt-5.2-codex",
        "reasoning": "gpt-5.2-codex",
        "analysis": "gpt-5.2-codex",
        "classification": "gpt-5.2-codex",   # Phase 38: upgraded from codex-mini
        "embedding": "gpt-5.2-codex",         # Phase 38: upgraded from codex-mini
        "postmortem": "gpt-5.2-codex",
        # Fallbacks
        "general": "gpt-5.2-mini",
        "default": "gpt-5.2-mini",
    }
    
    FALLBACK_MODEL = "gpt-5.2-mini"  # Universal fallback (Phase 23: no gpt-4)

    # Cost per 1K tokens (USD) — approximate, input+output blended average
    COST_PER_1K_TOKENS: Dict[str, float] = {
        "gpt-5-nano": 0.00010,
        "gpt-5-mini": 0.00040,
        "gpt-5.2-mini": 0.00060,
        "gpt-5.1-codex-mini": 0.00150,   # Legacy — kept for cost tracking
        "gpt-5.1-codex": 0.00600,
        "gpt-5.2-codex": 0.01000,
        "gpt-5.2": 0.01000,
        # Venice AI models
        "qwen3-coder-480b-a35b-instruct": 0.000315,
    }
    
    def __init__(self, enable_llm: Optional[bool] = None, require_llm: Optional[bool] = None, 
                 offline: Optional[bool] = None):
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
        
        # ── Phase 43: Local LLM integration (replaces Venice) ────────
        self._local_llm_provider = None  # Lazy-initialized
        self._local_client = None        # OpenAI-compatible client for local model
        self._model_router = None        # Lazy-initialized ModelRouter
        
        # Legacy Venice compat (now maps to local LLM)
        self.venice_api_key = os.getenv("VENICE_API_KEY")  # Legacy — ignored if local LLM active
        self.venice_base_url = "https://api.venice.ai/api/v1"  # Legacy
        self._venice_client = None
        self._venice_async_client = None
        self.venice_model = os.getenv("VENICE_MODEL", "qwen3-coder-480b-a35b-instruct")
        
        # Dual-mentor strategy settings
        self.enable_dual_mentor = os.getenv("ENABLE_DUAL_MENTOR", "true").lower() == "true"
        self.mentor_strategy = os.getenv("MENTOR_STRATEGY", "gpt_first")  # gpt_first, local_first, round_robin, parallel
        self._mentor_call_count = 0  # For round-robin
        
        # Strict mode validation
        if self._enable_llm and self._require_llm and not self.api_key:
            raise RuntimeError(
                "GPTManager: require_llm=True but OPENAI_API_KEY not set. "
                "Set the environment variable or use offline mode."
            )
        
        # Model configuration from environment or defaults
        self.primary_model = os.getenv("GPT_PRIMARY_MODEL", "gpt-5.2-codex")
        self.fallback_model = os.getenv("GPT_FALLBACK_MODEL", "gpt-5.2-mini")
        self.nano_model = os.getenv("GPT_NANO_MODEL", "gpt-5-nano")
        self.postmortem_model = os.getenv("GPT_POSTMORTEM_MODEL", "gpt-5.2-codex")
        self.strategic_model = os.getenv("GPT_STRATEGIC_MODEL", "gpt-5.2-codex")
        
        # Feature flags
        self.enable_postmortem_5_2 = True  # Always use deep model for postmortem
        
        # Venice (secondary provider)
        self.venice_enabled = bool(self.venice_api_key) and self.enable_dual_mentor
        self.stats_venice = {
            "total_requests": 0,
            "successes": 0,
            "failures": 0,
            "tokens_used": 0
        }
        
        # Phase 43: Local LLM (opt-in via FF_LOCAL_LLM=1 + GPU model present)
        self._local_llm_enabled = False
        self._detect_local_llm()  # Sets _local_llm_enabled=True only when FF_LOCAL_LLM=1 + server live
        self.stats_local_llm = {
            "total_requests": 0,
            "successes": 0,
            "failures": 0,
            "tokens_used": 0
        }
        
        # Token budgeting - per episode and per agent
        # Phase 13.0: +100% for ultra-accelerated autonomous learning pipeline
        # Agents need maximum token headroom to learn from GPT reasoning chains,
        # build internal world models, and develop autonomous decision-making
        self.token_limit = int(os.getenv("TOKEN_LIMIT_PER_EPISODE", "720000"))  # Phase 36: +23% (was 585K) — codex-primary routing needs more headroom
        self.token_limit_per_agent = int(os.getenv("TOKEN_LIMIT_PER_AGENT", "230000"))  # Phase 36: +23% (was 187.2K) — per-agent capacity for codex reasoning
        # Phase 13.0: Reasoning tasks get 5.5× multiplier for deep multi-step chains
        # Supports: exploit reasoning, pwn trajectory analysis, reflective meta-learning,
        # strategic planning, autonomous decision justification, and output interpretation
        self.reasoning_task_types = {"reasoning", "tactical", "strategic", "analysis", "learning", "postmortem", "defensive", "reconnaissance"}
        self.reasoning_token_multiplier = 5.50  # Phase 13.0: +34% (was 4.10) — deeper reasoning chains for autonomous learning
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

        # ── Phase 23: GPT Call Log for dashboard visibility ──────────
        # Ring buffer of recent API calls so the Rich dashboard can show
        # exactly what prompts were sent, what came back, tokens & cost.
        self._call_log: deque = deque(maxlen=200)  # full session log
        self._step_calls: List[Dict[str, Any]] = []  # per-step buffer (cleared each step)
        
        # ── Phase 15.0: BudgetManagerV2 (flag-gated) ────────────────
        self._budget_manager_v2 = None
        try:
            from core.feature_flags import get_feature_flags
            if get_feature_flags().budget_manager_v2:
                from core.llm.budget_manager import BudgetManagerV2
                self._budget_manager_v2 = BudgetManagerV2()
                logger.debug("GPTManager: BudgetManagerV2 initialized")
        except Exception:
            pass
        
        # Load existing cache
        self._load_cache()
        
        logger.debug(f"GPTManager initialized with primary model: {self.primary_model}")
        logger.debug(f"Fallback model: {self.fallback_model}")
        logger.debug(f"Platform detected: {platform.system()}")
        if self.is_configured():
            if self._local_llm_enabled:
                logger.info(f"Local LLM enabled | nano+mini → local GPU")
            elif self.venice_enabled:
                logger.debug(f"Venice AI enabled: {self.venice_enabled}, Model: {self.venice_model if self.venice_enabled else 'N/A'}")
        else:
            logger.warning("GPTManager: OPENAI_API_KEY not set. LLM calls disabled until configured.")
    
    def is_configured(self) -> bool:
        """Check if API key is available for LLM calls."""
        return bool(self.api_key) and self._enable_llm and not self._offline
    
    def _detect_local_llm(self) -> bool:
        """Check if local LLM should be activated (Phase 43).
        
        Returns True if local LLM is available and feature flag is on.
        """
        try:
            from core.feature_flags import get_feature_flags
            ff = get_feature_flags()
            if not getattr(ff, 'local_llm', False):
                return False
        except Exception:
            if os.getenv("FF_LOCAL_LLM", "0").lower() not in ("1", "true", "yes", "on"):
                return False
        
        try:
            from core.llm.local_llm_provider import get_local_llm_provider
            provider = get_local_llm_provider()
            if provider.is_available():
                self._local_llm_provider = provider
                self._local_llm_enabled = True
                return True
        except Exception as e:
            logger.debug(f"Local LLM detection failed: {e}")
        
        return False
    
    def _get_model_router(self):
        """Get or create the model router (lazy init)."""
        if self._model_router is None:
            from core.llm.model_router import ModelRouter
            self._model_router = ModelRouter(
                local_available=self._local_llm_enabled,
                local_model_name=(
                    self._local_llm_provider.get_model_name()
                    if self._local_llm_provider else ""
                ),
            )
        return self._model_router
    
    @property
    def local_client(self):
        """Get the local LLM client (OpenAI-compatible)."""
        if self._local_client is None and self._local_llm_provider:
            self._local_client = self._local_llm_provider.get_client()
        return self._local_client
    
    def has_local_llm(self) -> bool:
        """Check if local LLM is available."""
        return self._local_llm_enabled and self._local_llm_provider is not None
    
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
    
    # Sentinel prefix for offline placeholder responses — callers should check
    # for this prefix to avoid treating placeholders as executable commands.
    OFFLINE_SENTINEL = "[OFFLINE]"

    # Schema-compliant fallback for analysis/phase-guide callers.
    # Returned when the real API is unavailable so downstream JSON parsers
    # always receive valid JSON instead of echo/nmap command strings.
    _ANALYSIS_FALLBACK_JSON = json.dumps({
        "phase_decision": {
            "chosen_phase": "RECON", "phase_confidence": 0.3,
            "phase_goal": "fallback", "stay_conditions": [],
            "move_on_conditions": [], "contradictions": [],
            "phase_tag": "P34",
        },
        "anomalies": [],
        "candidates": [{
            "template_name": "nmap_basic", "family": "nmap",
            "why": "Fallback — API unavailable", "expected_outcome": "port discovery",
            "stop_condition": "any output", "confidence": 0.3,
            "risk": "low", "tags": ["fallback"],
        }],
        "selection": {
            "best_template_name": "nmap_basic",
            "runner_up_template_name": "",
            "selection_reason": "LLM fallback",
            "should_escalate_to_codex": True,
            "escalation_reason": "API unavailable",
        },
        "distillation_packet": {
            "observation": "API fallback", "reasoning": "LLM unavailable",
            "action_target": {"template_name": "nmap_basic", "why": "fallback"},
            "expected_outcome": "port discovery", "phase_target": "RECON",
            "confidence_target": 0.3,
            "gating_notes": {"expected_gate_result": "PASS", "reasons": ["fallback"]},
            "phase_tag": "P34",
        },
    })

    def _get_offline_placeholder(self, task_type: str) -> str:
        """Get a deterministic placeholder response for offline mode.
        
        All non-command placeholders are prefixed with OFFLINE_SENTINEL so
        downstream consumers can distinguish them from real commands.
        """
        placeholders = {
            "tactical": "nmap -sV 10.10.10.10",
            "defensive": "netstat -an",
            "reconnaissance": "ping 10.10.10.10",
            "diversify": "nslookup 10.10.10.10",
            "analysis": self._ANALYSIS_FALLBACK_JSON,
            "reasoning": f"{self.OFFLINE_SENTINEL} reasoning unavailable.",
            "postmortem": f"{self.OFFLINE_SENTINEL} postmortem analysis unavailable.",
            "general": f"{self.OFFLINE_SENTINEL} LLM unavailable."
        }
        return placeholders.get(task_type, placeholders["general"])

    @staticmethod
    def is_offline_placeholder(response: str) -> bool:
        """Check if a response is an offline placeholder (not a real command)."""
        if not response or not isinstance(response, str):
            return True
        return response.startswith(GPTManager.OFFLINE_SENTINEL) or response.startswith("Offline mode:")

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
        """Check if Venice AI is available as secondary provider."""
        return bool(self.venice_api_key) and self.venice_enabled

    def has_secondary_provider(self) -> bool:
        """Check if any secondary LLM provider is available (local or Venice)."""
        return self.has_local_llm() or self.has_venice()

    def get_next_mentor_provider(self) -> str:
        """
        Get the next mentor provider based on strategy.
        
        Phase 43: local LLM replaces Venice as secondary provider.
        
        Strategies:
        - gpt_first: Always GPT, local/Venice as fallback
        - local_first: Local LLM first, GPT as fallback
        - round_robin: Alternate between GPT and local
        - parallel: Use both (caller handles)
        
        Returns:
            str: "gpt", "local", or "both"
        """
        has_secondary = self.has_secondary_provider()
        if not has_secondary:
            return "gpt"
        
        # Phase 43: "local_first" and legacy "venice_first" both map to local
        if self.mentor_strategy in ("local_first", "venice_first"):
            return "local" if self.has_local_llm() else "venice"
        elif self.mentor_strategy == "round_robin":
            self._mentor_call_count += 1
            if self._mentor_call_count % 2 == 0:
                return "local" if self.has_local_llm() else "venice"
            return "gpt"
        elif self.mentor_strategy == "parallel":
            return "both"
        else:  # gpt_first (default)
            return "gpt"

    def get_mentor_clients(self) -> Dict[str, Any]:
        """
        Get mentor clients based on availability.
        
        Phase 43: Local LLM preferred as secondary over Venice.
        
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
        
        # Secondary: prefer local LLM, fall back to Venice
        if self.has_local_llm():
            try:
                local_cl = self.local_client
                result["secondary"] = local_cl
                result["secondary_async"] = local_cl  # llama-cpp sync is fine
                result["secondary_model"] = self._local_llm_provider.get_model_name()
            except Exception:
                pass
        elif self.has_venice() and not self.has_local_llm():
            try:
                result["secondary"] = self.venice_client
                result["secondary_async"] = self.venice_async_client
                result["secondary_model"] = self.venice_model
            except RuntimeError:
                pass
        
        return result
    
    def get_model_for_role(self, agent_id: Optional[str] = None, task_type: Optional[str] = None) -> str:
        """
        Get appropriate model based on 3-tier cost-optimized routing.
        
        Tier 1 — NANO (gpt-5-nano, $0.0001/1K tokens):
            Simple classification, output reformatting, cache key generation,
            boolean checks, template selection.
            
        Tier 2 — MINI (gpt-5.2-mini, $0.0006/1K tokens):
            Command parsing, tactical decisions, playbook selection,
            defensive analysis, reconnaissance planning, output interpretation.
            
        Tier 3 — CODEX (gpt-5.2-codex, $0.01/1K tokens):
            Strategic reasoning, exploit chain planning, postmortem analysis,
            deep learning/teaching, diversification.
        
        Args:
            agent_id: Agent identifier (e.g., "RedAgent", "ScoutAgent")
            task_type: Type of task (e.g., "tactical", "analysis", "postmortem")
            
        Returns:
            str: Model name to use
        """
        # Tier 3 — CODEX: Deep reasoning, strategic, tactical, analysis, exploit planning
        # Phase 36: Promoted tactical+analysis to codex for smarter command selection
        # Phase 36.1: Moved defensive+reconnaissance back to mini (cost optimization)
        _codex_tasks = {
            "strategic", "postmortem", "reasoning", "learning",
            "diversify", "exploit_chain",
            "tactical", "analysis",  # Core reasoning stays at codex
        }
        if task_type in _codex_tasks:
            return self.strategic_model  # gpt-5.2-codex
        
        # Tier 2 — MINI: Parsing, output interpretation, playbook selection, recon, defense
        _mini_tasks = {
            "playbook", "parsing", "command_selection", "output_parse",
            "defensive", "reconnaissance",  # Phase 36.1: moved from codex (cost-efficient)
        }
        if task_type in _mini_tasks:
            return self.fallback_model  # gpt-5.2-mini — structured extraction tier
        
        # Tier 1 — NANO: Simple classification, reformatting, checks
        # "general", "classification", "reformat", "cache", None, or unknown
        return self.nano_model  # gpt-5-nano
    
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
            # Phase 15.0: Reset BudgetManagerV2 per episode
            if self._budget_manager_v2 is not None:
                self._budget_manager_v2.reset_episode(
                    episode_id=str(episode_id) if episode_id is not None else ""
                )
            logger.debug(f"Reset all token counts for episode {episode_id}")
    
    def reset_token_count(self):
        """Reset token count for new episode (legacy method, use reset_episode)"""
        self.reset_episode()
    
    def can_make_request(self, agent_name: Optional[str] = None,
                         task_type: Optional[str] = None) -> bool:
        """
        Check if we can make another request within token limits.
        
        Phase 11.2: Reasoning/memory/learning tasks get 40% more headroom
        to prevent cutting off deep exploit reasoning mid-thought.
        Confidence-gated via task_type — only reasoning tasks get the boost.
        
        Args:
            agent_name: If provided, check per-agent limit too.
            task_type: If reasoning/tactical/strategic, allow 40% more tokens.
            
        Returns:
            bool: True if request is within budget.
        """
        # Phase 11.2: Reasoning tasks get expanded ceiling to avoid token bonfire
        # on non-reasoning tasks while allowing deep reasoning to breathe
        _multiplier = 1.0
        if task_type and task_type in self.reasoning_task_types:
            _multiplier = self.reasoning_token_multiplier
        
        _effective_limit = int(self.token_limit * _multiplier)
        _effective_agent_limit = int(self.token_limit_per_agent * _multiplier)
        
        if self.tokens_used >= _effective_limit:
            return False
        if agent_name and self.tokens_by_agent.get(agent_name, 0) >= _effective_agent_limit:
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
        """Create a cache key for the request.
        
        Phase 38: Use full prompt hash to prevent stale cache hits when
        discovery state changes (e.g., ports/services differ between steps).
        """
        content = f"{task_type}|{agent_id}|{prompt}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def gpt_request(self, prompt: str, task_type: str = "general", 
                   agent_id: str = "unknown", max_tokens: int = 150,
                   model: Optional[str] = None, allow_fallback: bool = True,
                   timeout: Optional[int] = None,
                   system_prompt: Optional[str] = None,
                   response_format: Optional[str] = None) -> str:
        """
        Make a request to GPT with role-based model routing.
        
        Args:
            prompt: The prompt to send
            task_type: Type of task (tactical, analysis, postmortem, etc.)
            agent_id: Agent making the request (used for model routing)
            max_tokens: Maximum tokens in response
            model: Optional explicit model override
            allow_fallback: Whether to allow fallback to backup model
            timeout: Optional request timeout in seconds (default: 8 for agents, higher for analysis)
            system_prompt: Optional custom system prompt override. When provided,
                replaces the internal task_type-based system prompt. Used by
                ReflectiveCortex and other callers needing specialized prompts.
            response_format: Optional response format hint. "json_object" forces
                the API to return valid JSON (Chat Completions only, not codex/Responses API).
            
        Returns:
            str: The model's response
        """
        # CRITICAL: Check offline mode FIRST, before any logging or API access
        if self.is_offline():
            return self._get_offline_placeholder(task_type)
        
        # PHASE 6.5: Quota circuit breaker — skip instantly if quota exhausted
        if getattr(self, '_quota_exhausted', False):
            return self._get_offline_placeholder(task_type)
        
        if not self.can_make_request(task_type=task_type):
            logger.warning(f"Token limit reached for episode ({self.tokens_used}/{self.token_limit}, task={task_type})")
            if task_type == "analysis":
                return self._ANALYSIS_FALLBACK_JSON
            return "echo 'Token limit reached'"
        
        # Role-based model selection (unless explicitly overridden)
        if model is None:
            model = self.get_model_for_role(agent_id=agent_id, task_type=task_type)
        
        # ── Phase 43: Route to local LLM if applicable ──────────────
        _use_local = False
        _routing_decision = None
        if self._local_llm_enabled:
            router = self._get_model_router()
            _routing_decision = router.route(model, task_type)
            if _routing_decision.provider == "local":
                _use_local = True
        
        # ── Phase 32: Per-tier soft token ceilings ───────────────────
        # Clamp max_tokens to tier-appropriate limits to prevent waste.
        _TIER_OUTPUT_CAPS: Dict[str, int] = {
            "nano": 450,
            "mini": 1500,   # Phase 36: +67% for richer parsing output
            "codex": 2000,  # Phase 36: +67% for deeper reasoning chains
            "full": 1200,   # Phase 36: +33%
        }
        from core.llm.budget_manager import _MODEL_TIER as _tier_map
        _tier = _tier_map.get(model, "mini")
        _cap = _TIER_OUTPUT_CAPS.get(_tier, 900)
        if max_tokens > _cap:
            max_tokens = _cap

        # Phase 36: Soft input-token ceilings (approx 4 chars/token)
        _TIER_INPUT_CAPS: Dict[str, int] = {
            "nano": 2_000,
            "mini": 10_000,  # Phase 36: +67% for richer context
            "codex": 20_000, # Phase 36: +67% for full attack context
            "full": 10_000,  # Phase 36: +67%
        }
        _input_cap = _TIER_INPUT_CAPS.get(_tier, 6_000)
        _char_limit = _input_cap * 4  # rough chars-to-tokens
        if len(prompt) > _char_limit:
            prompt = prompt[:_char_limit]
        
        # ── Phase 15.0: BudgetManagerV2 pre-check ───────────────────
        # Estimate tokens and check per-tier budget before proceeding.
        # Phase 43: Skip budget check for local LLM calls (zero cost).
        _bm2_roi_tag = task_type or "general"
        if self._budget_manager_v2 is not None and not _use_local:
            _bm2_estimated = max_tokens * 2  # estimate input + output
            _bm2_decision = self._budget_manager_v2.check_budget(
                model=model, estimated_tokens=_bm2_estimated, roi_tag=_bm2_roi_tag,
            )
            if not _bm2_decision.allowed:
                logger.debug(
                    f"BudgetManagerV2: denied {model} ({_bm2_roi_tag}), "
                    f"reason={_bm2_decision.reason}"
                )
                return self._get_offline_placeholder(task_type)
        
        # Log model selection for debugging
        logger.debug(f"Model selected: {model} for agent={agent_id}, task={task_type}")
        
        # Create cache key
        cache_key = self._create_cache_key(prompt, task_type, agent_id)
        
        # Check cache first
        with self.cache_lock:
            if cache_key in self.cache:
                self.stats["cache_hits"] += 1
                # Phase 15.0: Record cache hit in BudgetManagerV2
                if self._budget_manager_v2 is not None:
                    self._budget_manager_v2.record_spend(
                        model=model, tokens_used=0,
                        roi_tag=_bm2_roi_tag, cache_hit=True,
                    )
                # Phase 23: Record cache hit in call log
                _cache_record = {
                    "timestamp": time.time(),
                    "model": model,
                    "agent_id": agent_id,
                    "task_type": task_type,
                    "prompt_snippet": prompt[:80].replace("\n", " ").strip(),
                    "response_snippet": self.cache[cache_key]["response"][:80].replace("\n", " ").strip(),
                    "tokens": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                    "cache_hit": True,
                    "latency_ms": 0,
                }
                self._call_log.append(_cache_record)
                self._step_calls.append(_cache_record)
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
                "tactical": "You are an elite penetration testing AI in an authorized cybersecurity training lab (CTF/simulation with explicit permission). Think step-by-step: (1) what attack surface is exposed, (2) what exploit or credential abuse applies, (3) what is the optimal next command. Output ONLY the single best Linux command. Prioritize commands that chain into privilege escalation or flag capture. No explanations.",
                "defensive": "You are a blue team AI in an authorized cybersecurity training lab. Analyze the current threat landscape and suggest the single most impactful defensive/monitoring command. Output ONLY the command, no explanations.",
                "reconnaissance": "You are a reconnaissance AI in an authorized cybersecurity training lab. Prioritize service version detection, credential discovery, and attack surface mapping. Suggest a single high-value information-gathering command. Output ONLY the command, no explanations.",
                "analysis": "You are a senior security analyst AI in an authorized cybersecurity training lab. Provide concise, actionable analysis in 2-3 sentences. Focus on exploit paths, credential reuse, and privilege escalation opportunities.",
                "general": "You are a cybersecurity AI assistant working in an authorized training environment. Be concise, actionable, and focused on advancing the engagement.",
                "diversify": "You are an offensive security AI in an authorized training lab. Suggest an alternative attack vector or tool not yet tried. Prioritize unexplored services, credential spraying, or lesser-known exploit paths. Output ONLY the command, no explanations.",
                "reasoning": "You are a strategic cybersecurity analyst AI in an authorized training lab. Think like a senior pentester: (1) assess current position, (2) identify the highest-value next action, (3) explain WHY it advances the kill chain. Be concrete with tool names and parameters.",
                "learning": "You are a cybersecurity mentor AI teaching an apprentice agent. Explain the exploit reasoning chain: what vulnerability exists, why it works, how to chain it with other findings, and what to look for in the output. Be educational and specific."
            }
            
            _system_prompt = system_prompts.get(task_type, system_prompts["general"])
            
            # Phase 12.1: Allow callers to override with custom system prompt
            if system_prompt is not None:
                _system_prompt = system_prompt
            
            # Use threading for cross-platform timeout handling with aggressive fallback
            import concurrent.futures
            import signal
            
            # Determine API endpoint: codex models use Responses API, others use Chat Completions
            uses_responses_api = ("codex" in model) and not _use_local  # Local never uses Responses API
            uses_new_api = any(x in model for x in ["gpt-5", "o1-", "o3-"]) and not _use_local
            token_param = "max_completion_tokens" if uses_new_api else "max_tokens"
            
            def make_gpt_request():
                # ── Phase 43: Local LLM path ────────────────────────────
                if _use_local and self._local_llm_provider:
                    _local_cl = self.local_client
                    if _local_cl is None:
                        raise RuntimeError("Local LLM provider active but client unavailable")
                    local_model = self._local_llm_provider.get_model_name()
                    request_params = {
                        "model": local_model,
                        "messages": [
                            {"role": "system", "content": _system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        "max_tokens": max_tokens,
                        "temperature": 0.7 if task_type == "diversify" else 0.3,
                    }
                    if response_format == "json_object":
                        request_params["response_format"] = {"type": "json_object"}
                    return _local_cl.chat.completions.create(**request_params)
                
                if uses_responses_api:
                    # OpenAI Responses API (v1/responses) for codex models
                    # Codex models use internal reasoning tokens (~1000+) that count against
                    # max_output_tokens, so we need a much higher budget than chat completions
                    codex_token_budget = max(max_tokens * 15, 2000)
                    return self.client.responses.create(
                        model=model,
                        instructions=_system_prompt,
                        input=prompt,
                        max_output_tokens=codex_token_budget,
                    )
                else:
                    # Standard Chat Completions API
                    request_params = {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": _system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        token_param: max_tokens,
                        "timeout": min(max(5.0, (timeout or 8) - 2), 120.0),  # Client timeout: adapt to request timeout
                    }
                    # Only add temperature for models that support it (not gpt-5.x, o1, o3)
                    if not uses_new_api:
                        request_params["temperature"] = 0.7 if task_type == "diversify" else 0.3
                    # Phase 38: JSON mode — force structured output when requested
                    if response_format == "json_object" and not uses_responses_api:
                        request_params["response_format"] = {"type": "json_object"}
                    return self.client.chat.completions.create(**request_params)
            
            # Execute with aggressive timeout using ThreadPoolExecutor
            # CRITICAL: Do NOT use `with` context manager — its __exit__ calls
            # shutdown(wait=True) which blocks until the thread completes, even
            # after TimeoutError. For slow models (gpt-5.2-codex postmortem with
            # 30K tokens), this would hang for minutes.
            _request_timeout = timeout if timeout is not None else 8
            try:
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                future = executor.submit(make_gpt_request)
                try:
                    response = future.result(timeout=_request_timeout)
                    executor.shutdown(wait=False)
                except concurrent.futures.TimeoutError:
                    logger.warning(f"GPT request timed out after {_request_timeout} seconds for {agent_id}, using fallback")
                    future.cancel()
                    executor.shutdown(wait=False, cancel_futures=True)
                    # Return immediate fallback command based on task type
                    fallback_commands = {
                        "tactical": "nmap -sV 10.10.10.10",
                        "defensive": "netstat -an",
                        "reconnaissance": "ping 10.10.10.10",
                        "diversify": "nslookup 10.10.10.10",
                        "analysis": self._ANALYSIS_FALLBACK_JSON,
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
                    "analysis": self._ANALYSIS_FALLBACK_JSON,
                    "general": "echo 'GPT error - using fallback'"
                }
                return fallback_commands.get(task_type, "echo 'GPT error'")
            
            self.last_request_time = time.time()
            self.stats["total_requests"] += 1
            
            # Extract content — Responses API uses .output_text, Chat Completions uses .choices[]
            if uses_responses_api:
                content = getattr(response, 'output_text', '') or ''
                content = content.strip()
                
                # Fallback: if output_text is empty, try parsing response.output directly
                if not content and hasattr(response, 'output') and response.output:
                    for item in response.output:
                        # ResponseOutputMessage items have .content list
                        if hasattr(item, 'content') and item.content:  # type: ignore[union-attr]
                            for part in item.content:  # type: ignore[union-attr]
                                if hasattr(part, 'text') and part.text:  # type: ignore[union-attr]
                                    content = part.text.strip()  # type: ignore[union-attr]
                                    if content:
                                        break
                        if content:
                            break
                
                # Diagnostic logging when response is truly empty
                if not content:
                    resp_status = getattr(response, 'status', 'unknown')
                    resp_output_len = len(response.output) if hasattr(response, 'output') and response.output else 0
                    output_types = []
                    if hasattr(response, 'output') and response.output:
                        output_types = [type(item).__name__ for item in response.output]
                    logger.warning(
                        f"Empty Responses API output | agent={agent_id} | task={task_type} | "
                        f"status={resp_status} | output_items={resp_output_len} | "
                        f"types={output_types}"
                    )
            elif hasattr(response, 'choices') and response.choices and len(response.choices) > 0:  # type: ignore[union-attr]
                content = response.choices[0].message.content  # type: ignore[union-attr]
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
                    # Phase 43: Track local LLM separately
                    _track_model = model
                    if _use_local:
                        _track_model = f"local:{self._local_llm_provider.get_model_name()}" if self._local_llm_provider else "local:unknown"
                        self.stats_local_llm["total_requests"] = self.stats_local_llm.get("total_requests", 0) + 1
                        self.stats_local_llm["successes"] = self.stats_local_llm.get("successes", 0) + 1
                        self.stats_local_llm["tokens_used"] = self.stats_local_llm.get("tokens_used", 0) + tokens_used
                    # Phase 6.3: per-model tracking + cost
                    self.tokens_by_model[_track_model] = self.tokens_by_model.get(_track_model, 0) + tokens_used
                    self.requests_by_model[_track_model] = self.requests_by_model.get(_track_model, 0) + 1
                    cost_rate = 0.0 if _use_local else self.COST_PER_1K_TOKENS.get(model, 0.001)
                    step_cost = (tokens_used / 1000.0) * cost_rate
                    self._cumulative_cost_usd += step_cost
                    self._episode_cost_usd += step_cost
                    # Phase 15.0: Record spend in BudgetManagerV2
                    # Phase 43: Skip for local LLM (zero cost, avoid depleting OpenAI tiers)
                    if self._budget_manager_v2 is not None and not _use_local:
                        self._budget_manager_v2.record_spend(
                            model=model, tokens_used=tokens_used,
                            roi_tag=_bm2_roi_tag,
                        )

                    # Phase 23: Record call in visibility log for dashboard
                    _call_record = {
                        "timestamp": time.time(),
                        "model": model,
                        "agent_id": agent_id,
                        "task_type": task_type,
                        "prompt_snippet": prompt[:120].replace("\n", " ").strip(),
                        "response_snippet": content[:120].replace("\n", " ").strip(),
                        "tokens": tokens_used,
                        "input_tokens": getattr(response.usage, 'input_tokens', 0) or getattr(response.usage, 'prompt_tokens', 0) or 0,
                        "output_tokens": getattr(response.usage, 'output_tokens', 0) or getattr(response.usage, 'completion_tokens', 0) or 0,
                        "cost_usd": step_cost,
                        "cache_hit": False,
                        "latency_ms": int((time.time() - self.last_request_time) * 1000) if self.last_request_time else 0,
                    }
                    self._call_log.append(_call_record)
                    self._step_calls.append(_call_record)
                
                # Sanitize if it's a command
                if task_type in ["tactical", "defensive", "reconnaissance", "diversify"]:
                    content = self._sanitize_command(content)
                    if not content:
                        logger.warning(f"Sanitizer emptied GPT response for {agent_id}/{task_type}")
                        content = "nmap -sV {target}" if task_type == "tactical" else "netstat -an"
                
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
                logger.warning(f"Empty response from GPT for {agent_id}/{task_type}, using task-based fallback")
                fallback_commands = {
                    "tactical": "nmap -sV -p- --min-rate=1000 {target}",
                    "defensive": "netstat -tlnp",
                    "reconnaissance": "nmap -sC -sV {target}",
                    "diversify": "nikto -h {target}",
                    "analysis": self._ANALYSIS_FALLBACK_JSON,
                    "general": "echo 'Empty GPT response — fallback'"
                }
                return fallback_commands.get(task_type, "echo 'Empty GPT response'")
                
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
        Simplified dual feedback — routes through standard model routing.
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

    # ── Phase 23: GPT Call Visibility for Dashboard ──────────────────

    def get_step_calls(self) -> List[Dict[str, Any]]:
        """Return GPT calls made since last clear (i.e. this step).
        
        Each entry: {timestamp, model, agent_id, task_type, prompt_snippet,
                     response_snippet, tokens, input_tokens, output_tokens,
                     cost_usd, cache_hit, latency_ms}
        """
        return list(self._step_calls)

    def clear_step_calls(self) -> None:
        """Clear the per-step call buffer. Call at the START of each step."""
        self._step_calls.clear()

    def get_recent_calls(self, n: int = 10) -> List[Dict[str, Any]]:
        """Return the last N GPT calls from the session ring buffer."""
        calls = list(self._call_log)
        return calls[-n:] if len(calls) > n else calls

    def get_gpt_activity_snapshot(self) -> Dict[str, Any]:
        """Full GPT activity snapshot for dashboard rendering.
        
        Returns:
            Dict with cumulative stats, per-step calls, model breakdown,
            and Phase 32 tier-level cost summary.
        """
        step_calls = self.get_step_calls()
        step_tokens = sum(c.get("tokens", 0) for c in step_calls)
        step_cost = sum(c.get("cost_usd", 0.0) for c in step_calls)
        step_api_calls = sum(1 for c in step_calls if not c.get("cache_hit"))
        step_cache_hits = sum(1 for c in step_calls if c.get("cache_hit"))

        # Phase 32: episode-level tier cost summary from BudgetManagerV2
        tier_cost_summary: Dict[str, Any] = {}
        if self._budget_manager_v2 is not None:
            tier_cost_summary = self._budget_manager_v2.get_episode_cost_summary()
        
        return {
            "cumulative_cost_usd": round(self._cumulative_cost_usd, 4),
            "episode_cost_usd": round(self._episode_cost_usd, 4),
            "total_requests": self.stats.get("total_requests", 0),
            "total_tokens": self.stats.get("tokens_used_total", 0),
            "cache_hits": self.stats.get("cache_hits", 0),
            "step_calls": step_calls,
            "step_tokens": step_tokens,
            "step_cost_usd": round(step_cost, 6),
            "step_api_calls": step_api_calls,
            "step_cache_hits": step_cache_hits,
            "models": {
                model: {
                    "tokens": self.tokens_by_model.get(model, 0),
                    "requests": self.requests_by_model.get(model, 0),
                    "cost_usd": round((self.tokens_by_model.get(model, 0) / 1000.0) * self.COST_PER_1K_TOKENS.get(model, 0.001), 4),
                }
                for model in self.tokens_by_model
            },
            # Phase 32: per-tier cost summary
            "tier_cost_summary": tier_cost_summary,
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
            test_prompt = "Respond with 'ARIASKA GPT is operational'"
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
    
    # ─── Singleton pattern ────────────────────────────────────────────
    _instance: Optional["GPTManager"] = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls, **kwargs) -> "GPTManager":
        """Get or create the singleton GPTManager instance.
        
        Thread-safe. First call creates the instance with any provided kwargs.
        Subsequent calls return the same instance (kwargs ignored).
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(**kwargs)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton — ONLY for tests."""
        with cls._lock:
            cls._instance = None

    def cleanup(self):
        """Clean shutdown - save cache and stats"""
        self._save_cache()
        logger.info("GPTManager cleaned up successfully")

# Singleton instance
_gpt_manager_instance = None

def get_gpt_manager() -> GPTManager:
    """Get singleton GPTManager instance (legacy compat — prefers classmethod)."""
    return GPTManager.get_instance()
