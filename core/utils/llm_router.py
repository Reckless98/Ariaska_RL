#!/usr/bin/env python3
# core/utils/llm_router.py — ARIASKA LLM Router v1.0
# 🧠 Centralized LLM Orchestration | 🔄 Fallback Chains | 📊 Token Optimization

import os
import time
import json
import threading
import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from enum import Enum
from pathlib import Path
from rich.console import Console
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger("ariaska.llm_router")
console = Console()

# Define router response structure
@dataclass
class RouterResponse:
    """Structured response from the LLM Router."""
    content: str
    model_used: str
    tokens: Dict[str, int]
    latency: float
    success: bool
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class ModelTier(Enum):
    """Model tier classification for routing decisions."""
    LOCAL = "local"        # Local Ollama models
    TACTICAL = "tactical"  # Fast, small, cost-effective models
    STRATEGIC = "strategic" # Balanced models
    ADVANCED = "advanced"   # Most powerful models for complex tasks

class LLMRouter:
    """
    Central router for all LLM interactions in ARIASKA_RL.
    
    Key features:
    - Unified interface for all LLM types (Local, GPT, Azure, etc.)
    - Intelligent routing based on task complexity and model capabilities
    - Systematic fallback chains
    - Centralized caching and token optimization
    - Comprehensive error handling and logging
    - Thread safety for concurrent usage
    
    Usage:
        router = LLMRouter()
        response = router.request(
            prompt="Generate a command for port scanning 10.0.0.1",
            role="tactical",
            require_json=False
        )
    """
    
    _instance = None
    _lock = threading.Lock()
    
    # Singleton pattern to ensure only one router instance
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(LLMRouter, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, cache_path=None, config_path=None):
        """Initialize the LLM Router with configuration and cache."""
        # Skip initialization if already done (singleton pattern)
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        # Basic setup
        self.cache_path = cache_path or os.path.join("core", "memories", "llm_cache", "router_cache.json")
        self.config_path = config_path or os.path.join("config", "llm_router.json")
        
        # Initialize core components
        self.backends = {}  # LLM backend instances
        self.cache = {}     # In-memory response cache
        self.cache_lock = threading.Lock()
        
        # Stats tracking
        self.stats = {
            "requests": 0,
            "cache_hits": 0,
            "backend_usage": {},
            "errors": 0,
            "tokens": {
                "total": 0,
                "per_backend": {}
            },
            "avg_latency": 0,
            "response_times": []
        }
        
        # Load configuration
        self._load_config()
        
        # Load cache
        self._load_cache()
        
        # Initialize backends based on config
        self._initialize_backends()
        
        # Mark initialization complete
        self._initialized = True
        logger.info("🧠 LLM Router initialized")
    
    def _load_config(self):
        """Load router configuration from file."""
        default_config = {
            "fallback_chains": {
                "tactical": ["LocalLilyLLM", "GPT4o-mini", "GPT4o"],
                "strategic": ["LocalSenecaLLM", "LocalWhiteRabbitNeo", "GPT4o"],
                "advanced": ["GPT4o", "GPT4-turbo"]
            },
            "max_cache_size": 1000,
            "cache_ttl_hours": 24,
            "default_timeout": 30,
            "token_budget": {
                "default": 500,
                "tactical": 300,
                "strategic": 800,
                "advanced": 2000
            }
        }
        
        try:
            # Ensure config directory exists
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.config = {**default_config, **json.load(f)}
                logger.debug(f"Loaded router config from {self.config_path}")
            else:
                # Create default config if it doesn't exist
                self.config = default_config
                with open(self.config_path, 'w') as f:
                    json.dump(default_config, f, indent=2)
                logger.info(f"Created default router config at {self.config_path}")
        except Exception as e:
            logger.warning(f"Failed to load router config: {e}, using defaults")
            self.config = default_config
    
    def _load_cache(self):
        """Load the response cache from disk."""
        try:
            # Ensure cache directory exists
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            
            if os.path.exists(self.cache_path):
                with open(self.cache_path, 'r') as f:
                    cache_data = json.load(f)
                    
                # Filter out expired cache entries
                now = time.time()
                ttl_seconds = self.config["cache_ttl_hours"] * 3600
                
                self.cache = {
                    k: v for k, v in cache_data.items()
                    if not v.get("timestamp") or now - v["timestamp"] < ttl_seconds
                }
                
                # Log cache status
                logger.info(f"Loaded LLM router cache with {len(self.cache)} valid entries")
            else:
                self.cache = {}
                
        except Exception as e:
            logger.warning(f"Failed to load router cache: {e}, using empty cache")
            self.cache = {}
    
    def _save_cache(self):
        """Save the response cache to disk."""
        try:
            with self.cache_lock:
                # Prune cache if it exceeds max size
                if len(self.cache) > self.config["max_cache_size"]:
                    # Sort by timestamp and keep only newest entries
                    sorted_cache = sorted(
                        self.cache.items(),
                        key=lambda x: x[1].get("timestamp", 0)
                    )
                    self.cache = dict(sorted_cache[-self.config["max_cache_size"]:])
                
                # Write cache to disk
                with open(self.cache_path, 'w') as f:
                    json.dump(self.cache, f)
                    
        except Exception as e:
            logger.warning(f"Failed to save router cache: {e}")
    
    def _initialize_backends(self):
        """Initialize LLM backends lazily to prevent early imports."""
        # We're not initializing backends immediately to avoid circular imports
        # They will be initialized on first use
        self.backends = {}
        self._backend_initializers = {
            # Local LLM backends
            "LocalLilyLLM": self._init_local_lily,
            "LocalSenecaLLM": self._init_local_seneca,
            "LocalWhiteRabbitNeo": self._init_local_whiterabbit,
            "LocalPsyfighter": self._init_local_psyfighter,
            
            # OpenAI/Azure backends
            "GPT4o-mini": self._init_gpt4o_mini,
            "GPT4o": self._init_gpt4o,
            "GPT4-turbo": self._init_gpt4_turbo
        }
    
    def _get_backend(self, backend_name: str):
        """
        Get or initialize an LLM backend by name.
        
        Uses lazy initialization to prevent circular imports.
        """
        # Return already initialized backend if available
        if backend_name in self.backends:
            return self.backends[backend_name]
        
        # Try to initialize backend if we have a registered initializer
        if backend_name in self._backend_initializers:
            try:
                # Call the appropriate initializer
                self.backends[backend_name] = self._backend_initializers[backend_name]()
                return self.backends[backend_name]
            except Exception as e:
                logger.error(f"Failed to initialize {backend_name}: {e}")
                return None
        
        # Backend not recognized
        logger.warning(f"Unknown LLM backend: {backend_name}")
        return None
    
    def _init_local_lily(self):
        """Initialize LocalLilyLLM backend."""
        try:
            from core.utils.local_llm_manager import LocalLilyLLMManager
            return LocalLilyLLMManager()
        except Exception as e:
            logger.error(f"Failed to initialize LocalLilyLLM: {e}")
            return None
    
    def _init_local_seneca(self):
        """Initialize LocalSenecaLLM backend."""
        try:
            from core.utils.local_llm_manager import LocalSenecaLLMManager
            return LocalSenecaLLMManager()
        except Exception as e:
            logger.error(f"Failed to initialize LocalSenecaLLM: {e}")
            return None
    
    def _init_local_whiterabbit(self):
        """Initialize LocalWhiteRabbitNeo backend."""
        try:
            from core.utils.local_llm_manager import LocalWhiteRabbitNeoManager
            return LocalWhiteRabbitNeoManager()
        except Exception as e:
            logger.error(f"Failed to initialize LocalWhiteRabbitNeo: {e}")
            return None
    
    def _init_local_psyfighter(self):
        """Initialize LocalPsyfighter backend."""
        try:
            from core.utils.local_llm_manager import LocalPsyfighterManager
            return LocalPsyfighterManager()
        except Exception as e:
            logger.error(f"Failed to initialize LocalPsyfighter: {e}")
            return None
    
    def _init_gpt4o_mini(self):
        """Initialize GPT-4o-mini backend via GPTManager."""
        try:
            # Import here to avoid circular imports
            from core.gpt_manager import GPTManager
            gpt = GPTManager()
            # Return a wrapper function to use proper model
            return lambda prompt, **kwargs: gpt.gpt_request(prompt, model="gpt-4o-mini", **kwargs)
        except Exception as e:
            logger.error(f"Failed to initialize GPT4o-mini: {e}")
            return None
    
    def _init_gpt4o(self):
        """Initialize GPT-4o backend via GPTManager."""
        try:
            from core.gpt_manager import GPTManager
            gpt = GPTManager()
            return lambda prompt, **kwargs: gpt.gpt_request(prompt, model="gpt-4o", **kwargs)
        except Exception as e:
            logger.error(f"Failed to initialize GPT4o: {e}")
            return None
    
    def _init_gpt4_turbo(self):
        """Initialize GPT-4 Turbo backend via GPTManager."""
        try:
            from core.gpt_manager import GPTManager
            gpt = GPTManager()
            return lambda prompt, **kwargs: gpt.gpt_request(prompt, model="gpt-4-turbo", **kwargs)
        except Exception as e:
            logger.error(f"Failed to initialize GPT4-turbo: {e}")
            return None
    
    def _get_fallback_chain(self, role: str) -> List[str]:
        """Get the appropriate fallback chain for a given role."""
        fallback_chains = self.config.get("fallback_chains", {})
        
        # Return specific role chain if it exists
        if role in fallback_chains:
            return fallback_chains[role]
        
        # Default fallback chain combines local and cloud options
        default_chain = ["LocalLilyLLM", "GPT4o-mini", "GPT4o"]
        return fallback_chains.get("default", default_chain)
    
    @lru_cache(maxsize=100)
    def _classify_task_complexity(self, prompt: str) -> str:
        """
        Classify task complexity to determine appropriate model tier.
        Uses simple heuristics to avoid circular LLM calls.
        
        Args:
            prompt: The LLM prompt to classify
            
        Returns:
            str: Complexity classification ("tactical", "strategic", or "advanced")
        """
        # Simple heuristic classification based on prompt length and keywords
        prompt_lower = prompt.lower()
        
        # Check for advanced keywords suggesting complex reasoning
        advanced_keywords = [
            "sophisticated", "comprehensive", "analyze in depth",
            "full analysis", "complex", "elaborate", "detailed explanation",
            "compare and contrast", "implications", "novel approach",
            "synthesize", "strategic assessment"
        ]
        if any(keyword in prompt_lower for keyword in advanced_keywords) or len(prompt) > 500:
            return "advanced"
        
        # Check for strategic keywords suggesting medium complexity
        strategic_keywords = [
            "explain", "strategy", "approach", "methodology", "recommend",
            "suggest", "develop", "design", "framework", "why", "how", "evaluate"
        ]
        if any(keyword in prompt_lower for keyword in strategic_keywords) or len(prompt) > 250:
            return "strategic"
        
        # Default to tactical for simpler, direct questions or commands
        return "tactical"
    
    def request(
        self,
        prompt: str,
        role: str = None,
        model: str = None,
        require_json: bool = False,
        require_validation: bool = False,
        json_schema: Dict = None,
        task_type: str = "general",
        timeout: int = None,
        allow_fallback: bool = True
    ) -> RouterResponse:
        """
        Primary method to request a response from the LLM ecosystem.
        
        Args:
            prompt: The prompt to send to the LLM
            role: Role-based routing ("tactical", "strategic", "advanced")
            model: Specific model to use (overrides role-based routing)
            require_json: Whether the response should be valid JSON
            require_validation: Whether to validate the response against expectations
            json_schema: JSON schema for validation if require_json is True
            task_type: Task classification for specialized LLM prompting
            timeout: Request timeout in seconds
            allow_fallback: Whether to allow fallbacks if primary model fails
            
        Returns:
            RouterResponse: Structured response with content and metadata
        """
        start_time = time.time()
        self.stats["requests"] += 1
        
        # Create cache key from role/model and prompt
        model_info = model if model else f"role:{role or 'auto'}"
        json_flag = "-json" if require_json else ""
        cache_key = f"{model_info}{json_flag}|{task_type}|{prompt[:100]}"
        
        # Return cached response if available
        with self.cache_lock:
            if cache_key in self.cache:
                self.stats["cache_hits"] += 1
                cached = self.cache[cache_key]
                
                # Create RouterResponse from cached data
                response = RouterResponse(
                    content=cached.get("content", ""),
                    model_used=cached.get("model", "cache"),
                    tokens=cached.get("tokens", {"total": 0}),
                    latency=0.001,  # Negligible latency for cache hits
                    success=True,
                    metadata=cached.get("metadata", {})
                )
                
                # Log cache hit
                logger.debug(f"Cache hit for: {prompt[:50]}...")
                return response
        
        # If no model is specified, determine routing based on role or auto-classify
        if not model:
            if not role:
                # Automatically classify task complexity to determine role
                role = self._classify_task_complexity(prompt)
            
            # Get the fallback chain for this role
            fallback_chain = self._get_fallback_chain(role)
        else:
            # Use only the specified model
            fallback_chain = [model]
            
        # Use fallback chain if fallbacks are allowed, otherwise just use the first model
        models_to_try = fallback_chain if allow_fallback else [fallback_chain[0]]
        
        # Try each model in the chain
        last_error = None
        for model_name in models_to_try:
            try:
                # Get the backend for this model
                backend = self._get_backend(model_name)
                if not backend:
                    logger.warning(f"Backend {model_name} could not be initialized, trying next")
                    continue
                
                # Track backend usage
                self.stats["backend_usage"][model_name] = self.stats["backend_usage"].get(model_name, 0) + 1
                
                # Execute prompt against backend with appropriate method
                if require_json:
                    # Handle specialized JSON request
                    if hasattr(backend, "query_json"):
                        # Native JSON support (like LocalLLMManager)
                        success, json_result = backend.query_json(
                            prompt=prompt,
                            task_type=task_type,
                            schema=json_schema,
                            timeout=timeout
                        )
                        content = json_result
                        success = success and isinstance(content, dict)
                    else:
                        # Standard function, we'll need to parse JSON ourselves
                        raw_result = backend(prompt, timeout=timeout) if callable(backend) else backend.query(prompt, timeout=timeout)
                        try:
                            # Try to parse JSON from result
                            import json, re
                            # Extract JSON block if in markdown format
                            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```|(\{.*?\})', raw_result, re.DOTALL)
                            if json_match:
                                content = json.loads(json_match.group(1) or json_match.group(2))
                                success = True
                            else:
                                # Try parsing whole response
                                content = json.loads(raw_result)
                                success = True
                        except Exception:
                            content = {"error": "Failed to parse JSON", "raw": raw_result}
                            success = False
                else:
                    # Standard text request
                    if callable(backend):
                        # Function-style backend (like GPTManager wrapper)
                        content = backend(prompt, timeout=timeout)
                    else:
                        # Object-style backend (like LocalLLMManager)
                        content = backend.query(
                            prompt=prompt,
                            task_type=task_type,
                            timeout=timeout,
                            allow_fallback=False  # We handle fallbacks at this level
                        )
                    success = bool(content)
                
                # Validate response if required
                if require_validation and success:
                    if require_json and json_schema:
                        # Validate against JSON schema
                        for key in json_schema:
                            if key not in content:
                                success = False
                                last_error = f"Response missing required field: {key}"
                                break
                    elif not require_json and len(str(content).strip()) < 2:
                        # Basic length validation for non-JSON responses
                        success = False
                        last_error = "Response too short"
                
                # If we have a valid response, cache it and return
                if success:
                    # Estimate token usage if not available from backend
                    tokens = {"total": 0}
                    if hasattr(backend, "get_token_usage"):
                        tokens = backend.get_token_usage()
                    else:
                        # Simple estimation (~4 chars per token)
                        prompt_tokens = max(1, len(prompt) // 4)
                        response_tokens = max(1, len(str(content)) // 4)
                        tokens = {
                            "total": prompt_tokens + response_tokens,
                            "prompt": prompt_tokens,
                            "completion": response_tokens
                        }
                    
                    # Update token statistics
                    self.stats["tokens"]["total"] += tokens.get("total", 0)
                    if model_name not in self.stats["tokens"]["per_backend"]:
                        self.stats["tokens"]["per_backend"][model_name] = 0
                    self.stats["tokens"]["per_backend"][model_name] += tokens.get("total", 0)
                    
                    # Calculate latency
                    elapsed = time.time() - start_time
                    
                    # Update latency stats
                    self.stats["response_times"].append(elapsed)
                    if len(self.stats["response_times"]) > 100:
                        self.stats["response_times"] = self.stats["response_times"][-100:]
                    self.stats["avg_latency"] = sum(self.stats["response_times"]) / len(self.stats["response_times"])
                    
                    # Prepare response metadata
                    metadata = {
                        "role": role,
                        "task_type": task_type,
                        "timestamp": time.time()
                    }
                    
                    # Create RouterResponse object
                    response = RouterResponse(
                        content=content,
                        model_used=model_name,
                        tokens=tokens,
                        latency=elapsed,
                        success=True,
                        metadata=metadata
                    )
                    
                    # Cache the successful response
                    with self.cache_lock:
                        self.cache[cache_key] = {
                            "content": content,
                            "model": model_name,
                            "tokens": tokens,
                            "timestamp": time.time(),
                            "metadata": metadata
                        }
                        
                        # Periodically save cache in a background thread
                        if self.stats["requests"] % 10 == 0:
                            threading.Thread(target=self._save_cache).start()
                    
                    # Log success
                    logger.info(f"✅ LLM request successful using {model_name} ({elapsed:.2f}s)")
                    return response
                    
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Model {model_name} failed: {last_error}")
                continue
        
        # If we reach here, all models failed
        self.stats["errors"] += 1
        error_message = last_error or "All LLM models failed to generate a valid response"
        logger.error(f"❌ All LLM models failed: {error_message}")
        
        # Return error response
        return RouterResponse(
            content=f"Error: {error_message}",
            model_used="none",
            tokens={"total": 0},
            latency=time.time() - start_time,
            success=False,
            metadata={"error": error_message}
        )
    
    def generate_command(
        self,
        task_description: str,
        task_type: str = "general",
        timeout: int = None,
        allow_fallback: bool = True
    ) -> Tuple[bool, str]:
        """
        Specialized method to generate a command for a given task.
        
        Args:
            task_description: Description of the task to generate a command for
            task_type: Type of task (recon, exploit, etc.)
            timeout: Request timeout in seconds
            allow_fallback: Whether to allow fallbacks if primary model fails
            
        Returns:
            Tuple[bool, str]: (success, command)
        """
        # Generate command using tactical role which is optimized for command generation
        response = self.request(
            prompt=task_description,
            role="tactical", 
            task_type=task_type,
            timeout=timeout,
            allow_fallback=allow_fallback
        )
        
        if response.success:
            # Extract command if needed
            command = response.content
            if isinstance(command, dict) and "command" in command:
                command = command["command"]
                
            # Ensure we have a string command
            command = str(command).strip()
            
            # Simple validation
            if command and len(command.split()) >= 1:
                return True, command
            else:
                return False, "Generated command was empty or invalid"
        else:
            return False, f"Command generation failed: {response.content}"
    
    def strategic_analysis(
        self,
        context: str,
        question: str,
        timeout: int = 60,
        allow_fallback: bool = True
    ) -> str:
        """
        Specialized method for strategic analysis using the most capable models.
        
        Args:
            context: Background context for the analysis
            question: Specific question to analyze
            timeout: Request timeout in seconds
            allow_fallback: Whether to allow fallbacks if primary model fails
            
        Returns:
            str: Strategic analysis
        """
        # Use strategic role which uses more powerful models for reasoning
        prompt = f"Context:\n{context}\n\nQuestion for analysis:\n{question}\n\nProvide a concise strategic analysis."
        
        response = self.request(
            prompt=prompt,
            role="strategic",
            task_type="reasoning",
            timeout=timeout,
            allow_fallback=allow_fallback
        )
        
        return response.content
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive usage statistics."""
        return dict(self.stats)
    
    def clear_cache(self):
        """Clear the response cache."""
        with self.cache_lock:
            self.cache = {}
            try:
                if os.path.exists(self.cache_path):
                    os.remove(self.cache_path)
            except Exception as e:
                logger.warning(f"Failed to remove cache file: {e}")
        logger.info("Cache cleared")
        
    def reset_stats(self):
        """Reset usage statistics."""
        self.stats = {
            "requests": 0,
            "cache_hits": 0,
            "backend_usage": {},
            "errors": 0,
            "tokens": {
                "total": 0,
                "per_backend": {}
            },
            "avg_latency": 0,
            "response_times": []
        }
        logger.info("Stats reset")

# Helper function to get or create a router instance
_router_instance = None

def get_router() -> LLMRouter:
    """Get the global router instance."""
    global _router_instance
    if _router_instance is None:
        _router_instance = LLMRouter()
    return _router_instance

# Module-level convenience functions
def query(
    prompt: str,
    role: str = None,
    model: str = None,
    require_json: bool = False,
    task_type: str = "general"
) -> Union[str, Dict]:
    """
    Convenience function to query the LLM ecosystem.
    
    Args:
        prompt: The prompt to send
        role: Role-based routing ("tactical", "strategic", "advanced")
        model: Specific model to use (overrides role-based routing)
        require_json: Whether the response should be JSON
        task_type: Task classification for specialized prompting
        
    Returns:
        Union[str, Dict]: Response content (string or dict for JSON)
    """
    router = get_router()
    response = router.request(
        prompt=prompt,
        role=role,
        model=model,
        require_json=require_json,
        task_type=task_type
    )
    return response.content

def generate_command(task_description: str, task_type: str = "general") -> str:
    """
    Generate a command for a specific task.
    
    Args:
        task_description: Description of what needs to be done
        task_type: Type of task (recon, exploit, etc.)
        
    Returns:
        str: The generated command
    """
    router = get_router()
    success, command = router.generate_command(
        task_description=task_description,
        task_type=task_type
    )
    if success:
        return command
    else:
        logger.warning(f"Command generation failed: {command}")
        return ""

# Testing module functionality
if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(level=logging.INFO)
    
    router = LLMRouter()
    
    # Test basic query
    print("\nTesting basic query...")
    response = router.request(
        prompt="List files in the current directory",
        role="tactical"
    )
    print(f"Response: {response.content}")
    print(f"Model used: {response.model_used}")
    print(f"Tokens: {response.tokens}")
    
    # Test command generation
    print("\nTesting command generation...")
    success, command = router.generate_command(
        task_description="Scan open ports on 192.168.1.1",
        task_type="recon"
    )
    print(f"Success: {success}")
    print(f"Command: {command}")
    
    # Test error handling
    print("\nTesting error handling with invalid model...")
    response = router.request(
        prompt="Test prompt",
        model="non_existent_model",
        allow_fallback=False
    )
    print(f"Success: {response.success}")
    print(f"Error: {response.content}")
    
    # Test stats
    print("\nRouter stats:")
    for key, value in router.get_stats().items():
        print(f"{key}: {value}")