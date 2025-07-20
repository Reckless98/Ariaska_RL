#!/usr/bin/env python3
# llm_orchestrator.py — Central LLM management system for ARIASKA_RL
# Eliminates circular dependencies between gpt_manager and local_llm_manager

import os
import json
import time
import random
import hashlib
import logging
import asyncio
import requests
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ariaska.llm_orchestrator")

try:
    from rich.console import Console
    console = Console()
except ImportError:
    # Fallback for environments without rich
    import sys
    class SimpleConsole:
        def print(self, *args, **kwargs):
            # Strip rich formatting if present
            text = args[0]
            if isinstance(text, str):
                # Basic removal of rich formatting tags
                import re
                text = re.sub(r'\[.*?\]', '', text)
            print(text)
    console = SimpleConsole()

class LLMOrchestrator:
    """
    Central orchestration system for all LLM interactions in ARIASKA_RL.
    
    Features:
    - Single point of access for all LLM models (local and remote)
    - Unified caching system
    - Fallback chain handling
    - Retry logic with exponential backoff
    - Health monitoring and reporting
    - Token usage tracking
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern to ensure only one orchestrator instance exists"""
        if cls._instance is None:
            cls._instance = super(LLMOrchestrator, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config=None):
        """Initialize the LLM orchestrator with configuration"""
        # Skip re-initialization if already initialized (singleton pattern)
        if self._initialized:
            return
        
        self._initialized = True
        self.config = config or {}
        
        # LLM server configurations
        self.ollama_host = os.environ.get("ARIASKA_OLLAMA_HOST", "http://localhost:11434")
        self.ollama_timeout = int(os.environ.get("ARIASKA_OLLAMA_TIMEOUT", 30))
        
        # Default models
        self.default_models = {
            "tactical": "wahidmounir/SenecaLLM_x_Qwen2.5-7B-CyberSecurity-Q8_0-GGUF",
            "planning": "TheBloke/WhiteRabbitNeo-13B-GGUF:Q6_K",
            "reasoning": "TheBloke/Psyfighter-13B-GGUF:Q6_K",
            "fallback": "QuantFactory/Lily-Cybersecurity-7B-v0.2-GGUF:Q8_0"
        }
        
        # Override with config if provided
        if self.config.get("models"):
            for role, model in self.config["models"].items():
                self.default_models[role] = model
        
        # Initialize caches
        self.cache_dir = Path("core/memories/llm_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_file = self.cache_dir / "unified_llm_cache.json"
        self.token_usage_file = Path("logs/token_usage.json")
        
        # Load cache if exists
        self.response_cache = {}
        self.load_cache()
        
        # Initialize token usage tracking
        self.token_usage = self._load_token_usage()
        
        # Server health status
        self.server_health = {
            "ollama": {"status": False, "last_checked": None},
            "openai": {"status": False, "last_checked": None}
        }
        
        # Check initial server health
        self._check_server_health()
        
        logger.info(f"LLM Orchestrator initialized with models: {self.default_models}")
    
    def load_cache(self):
        """Load the response cache from disk"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    self.response_cache = json.load(f)
                logger.info(f"Loaded {len(self.response_cache)} cached responses")
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON in cache file {self.cache_file}")
                self.response_cache = {}
            except Exception as e:
                logger.error(f"Error loading cache: {e}")
                self.response_cache = {}
    
    def save_cache(self):
        """Save the response cache to disk"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.response_cache, f, indent=2)
            logger.info(f"Saved {len(self.response_cache)} responses to cache")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def _load_token_usage(self):
        """Load token usage statistics from disk"""
        if self.token_usage_file.exists():
            try:
                with open(self.token_usage_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON in token usage file {self.token_usage_file}")
                return self._initialize_token_usage()
            except Exception as e:
                logger.error(f"Error loading token usage: {e}")
                return self._initialize_token_usage()
        else:
            return self._initialize_token_usage()
    
    def _initialize_token_usage(self):
        """Initialize token usage tracking structure"""
        return {
            "ollama": {model: {"prompts": 0, "completions": 0} for model in self.default_models.values()},
            "openai": {
                "gpt-4o": {"prompts": 0, "completions": 0, "total_tokens": 0, "cost": 0.0},
                "gpt-4o-mini": {"prompts": 0, "completions": 0, "total_tokens": 0, "cost": 0.0},
                "gpt-3.5-turbo": {"prompts": 0, "completions": 0, "total_tokens": 0, "cost": 0.0}
            },
            "last_updated": datetime.now().isoformat()
        }
    
    def save_token_usage(self):
        """Save token usage statistics to disk"""
        self.token_usage["last_updated"] = datetime.now().isoformat()
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.token_usage_file), exist_ok=True)
            
            with open(self.token_usage_file, 'w') as f:
                json.dump(self.token_usage, f, indent=2)
            logger.info(f"Saved token usage statistics")
        except Exception as e:
            logger.error(f"Error saving token usage: {e}")
    
    def _update_token_usage(self, provider, model, prompt_tokens=0, completion_tokens=0, total_tokens=0, cost=0.0):
        """Update token usage statistics"""
        if provider == "ollama":
            # For Ollama, we don't have exact token counts, so just increment request counts
            if model not in self.token_usage["ollama"]:
                self.token_usage["ollama"][model] = {"prompts": 0, "completions": 0}
            
            if prompt_tokens > 0:
                self.token_usage["ollama"][model]["prompts"] += 1
            if completion_tokens > 0:
                self.token_usage["ollama"][model]["completions"] += 1
                
        elif provider == "openai":
            if model not in self.token_usage["openai"]:
                self.token_usage["openai"][model] = {"prompts": 0, "completions": 0, "total_tokens": 0, "cost": 0.0}
            
            if prompt_tokens > 0:
                self.token_usage["openai"][model]["prompts"] += prompt_tokens
            if completion_tokens > 0:
                self.token_usage["openai"][model]["completions"] += completion_tokens
            if total_tokens > 0:
                self.token_usage["openai"][model]["total_tokens"] += total_tokens
                self.token_usage["openai"][model]["cost"] += cost
        
        # Save token usage periodically (every 10 updates)
        if random.random() < 0.1:
            self.save_token_usage()
    
    def _get_cache_key(self, model: str, prompt: str, system_prompt: str = None) -> str:
        """Generate a cache key from model name and prompt"""
        combined_input = f"{model}_{prompt}"
        if system_prompt:
            combined_input = f"{combined_input}_{system_prompt}"
        return hashlib.md5(combined_input.encode()).hexdigest()
    
    def _check_server_health(self):
        """Check the health of LLM servers"""
        # Check Ollama server
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if response.status_code == 200:
                self.server_health["ollama"] = {"status": True, "last_checked": datetime.now().isoformat()}
                
                # Also list available models
                models = [model["name"] for model in response.json().get("models", [])]
                self.server_health["ollama"]["available_models"] = models
                logger.info(f"Ollama server is healthy. Available models: {models}")
            else:
                self.server_health["ollama"] = {
                    "status": False, 
                    "last_checked": datetime.now().isoformat(),
                    "error": f"HTTP {response.status_code}"
                }
                logger.warning(f"Ollama server returned status code {response.status_code}")
        except requests.exceptions.ConnectionError:
            self.server_health["ollama"] = {
                "status": False, 
                "last_checked": datetime.now().isoformat(),
                "error": "Connection refused"
            }
            logger.error(f"Could not connect to Ollama server at {self.ollama_host}")
        except Exception as e:
            self.server_health["ollama"] = {
                "status": False, 
                "last_checked": datetime.now().isoformat(),
                "error": str(e)
            }
            logger.error(f"Error checking Ollama server: {e}")
        
        # Check OpenAI (via a simple environment variable test for now)
        self.server_health["openai"] = {
            "status": os.environ.get("OPENAI_API_KEY") is not None,
            "last_checked": datetime.now().isoformat()
        }
        
        if self.server_health["openai"]["status"]:
            logger.info("OpenAI API key found")
        else:
            logger.warning("OpenAI API key not found in environment")
        
        return self.server_health
    
    def get_server_health(self, force_check=False):
        """Get server health status, optionally forcing a new check"""
        if force_check:
            return self._check_server_health()
        return self.server_health
    
    def get_completion(self, prompt: str, model: str = None, role: str = None, 
                      system_prompt: str = None, temperature: float = 0.7,
                      max_tokens: int = 2048, use_cache: bool = True,
                      use_fallback: bool = True) -> Dict[str, Any]:
        """
        Get a completion from an LLM model with automatic fallback
        
        Args:
            prompt (str): The prompt to send to the model
            model (str, optional): Specific model to use
            role (str, optional): Role-based model selection (tactical, planning, reasoning)
            system_prompt (str, optional): System prompt for chat models
            temperature (float, optional): Sampling temperature
            max_tokens (int, optional): Maximum tokens to generate
            use_cache (bool, optional): Whether to use cached responses
            use_fallback (bool, optional): Whether to use fallback models on failure
            
        Returns:
            dict: Response with text, model used, and metadata
        """
        # Select model based on role if no specific model provided
        if not model and role and role in self.default_models:
            model = self.default_models[role]
        elif not model:
            # Default to tactical model
            model = self.default_models["tactical"]
        
        # Try to get from cache first if enabled
        if use_cache:
            cache_key = self._get_cache_key(model, prompt, system_prompt)
            if cache_key in self.response_cache:
                logger.info(f"Cache hit for model {model}")
                return {
                    "text": self.response_cache[cache_key]["text"],
                    "model": model,
                    "from_cache": True,
                    "created_at": self.response_cache[cache_key].get("created_at", datetime.now().isoformat())
                }
        
        # Determine if this is an Ollama model or OpenAI model
        is_ollama = "GGUF" in model or "/" in model or ":" in model
        
        try:
            if is_ollama:
                # Check if Ollama server is available
                if not self.server_health["ollama"]["status"]:
                    self._check_server_health()  # Try to refresh status
                    if not self.server_health["ollama"]["status"] and use_fallback:
                        # Fallback to OpenAI if Ollama not available
                        logger.warning(f"Ollama server unavailable, falling back to OpenAI")
                        return self._get_openai_completion(prompt, system_prompt, temperature, max_tokens, use_cache)
                
                # Try to get completion from Ollama
                response = self._get_ollama_completion(model, prompt, system_prompt, temperature, max_tokens)
            else:
                # Use OpenAI for non-Ollama models
                response = self._get_openai_completion(prompt, system_prompt, temperature, max_tokens, use_cache, model=model)
            
            # Cache the response if successful
            if use_cache and response and "text" in response:
                cache_key = self._get_cache_key(model, prompt, system_prompt)
                self.response_cache[cache_key] = {
                    "text": response["text"],
                    "model": response["model"],
                    "created_at": datetime.now().isoformat()
                }
                
                # Save cache periodically (10% chance each call)
                if random.random() < 0.1:
                    self.save_cache()
            
            return response
            
        except Exception as e:
            logger.error(f"Error getting completion from {model}: {e}")
            
            if use_fallback:
                # Try fallback model
                fallback_model = self.default_models["fallback"]
                logger.warning(f"Falling back to {fallback_model}")
                
                try:
                    # Determine if fallback is Ollama or OpenAI
                    if "GGUF" in fallback_model or "/" in fallback_model:
                        response = self._get_ollama_completion(
                            fallback_model, prompt, system_prompt, temperature, max_tokens
                        )
                    else:
                        response = self._get_openai_completion(
                            prompt, system_prompt, temperature, max_tokens, use_cache
                        )
                    return response
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
                    
                    # If both primary and fallback failed, try one last emergency fallback to OpenAI
                    try:
                        logger.warning("Attempting emergency fallback to OpenAI gpt-3.5-turbo")
                        emergency_response = self._get_openai_completion(
                            prompt, system_prompt, temperature, max_tokens, use_cache,
                            model="gpt-3.5-turbo"
                        )
                        return emergency_response
                    except Exception as emergency_error:
                        logger.error(f"Emergency fallback also failed: {emergency_error}")
                        return {
                            "text": f"Error: All LLM calls failed. Please check LLM server availability and API keys.",
                            "model": "error",
                            "error": f"Primary: {e}, Fallback: {fallback_error}, Emergency: {emergency_error}"
                        }
            
            return {"text": f"Error: {e}", "model": model, "error": str(e)}
    
    def _get_ollama_completion(self, model: str, prompt: str, 
                             system_prompt: str = None, temperature: float = 0.7,
                             max_tokens: int = 2048) -> Dict[str, Any]:
        """Get completion from Ollama"""
        request_data = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "num_predict": max_tokens,
            "stream": False
        }
        
        # Add system prompt if provided
        if system_prompt:
            request_data["system"] = system_prompt
        
        # Use exponential backoff for retries
        max_retries = 3
        base_delay = 1  # Start with 1 second delay
        
        for retry in range(max_retries):
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.ollama_host}/api/generate",
                    json=request_data,
                    timeout=self.ollama_timeout
                )
                latency = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    # Update token usage stats
                    self._update_token_usage(
                        provider="ollama", 
                        model=model, 
                        prompt_tokens=1,  # We don't have actual token counts, so just increment
                        completion_tokens=1
                    )
                    
                    return {
                        "text": result.get("response", ""),
                        "model": model,
                        "latency": latency,
                        "from_cache": False
                    }
                # Handle specific error codes
                elif response.status_code == 404:
                    error_msg = f"Model {model} not found in Ollama"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                elif response.status_code == 500:
                    error_msg = f"Ollama server error: {response.text}"
                    logger.error(error_msg)
                    # For server errors, we should retry
                    delay = base_delay * (2 ** retry)
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue
                else:
                    error_msg = f"Ollama API error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    raise Exception(error_msg)
            except requests.exceptions.Timeout:
                logger.warning(f"Ollama request timed out (retry {retry+1}/{max_retries})")
                # If we've tried enough times, rethrow the exception
                if retry == max_retries - 1:
                    raise
                # Otherwise, wait and retry
                delay = base_delay * (2 ** retry)
                time.sleep(delay)
            except requests.exceptions.ConnectionError as e:
                logger.error(f"Connection error to Ollama server: {e}")
                # Connection errors are likely not transient during a single request
                # Rethrow immediately
                raise
        
        # If we get here, we've exhausted our retries
        raise Exception(f"Failed to get response from Ollama after {max_retries} retries")
    
    def _get_openai_completion(self, prompt: str, system_prompt: str = None,
                             temperature: float = 0.7, max_tokens: int = 2048,
                             use_cache: bool = True, model: str = "gpt-4o-mini") -> Dict[str, Any]:
        """
        Get completion from OpenAI API
        
        This function uses the sgpt CLI tool for simplicity, but could be
        modified to use the OpenAI Python library directly.
        """
        # Validate API key
        if not os.environ.get("OPENAI_API_KEY"):
            raise Exception("OpenAI API key not found in environment")
        
        import subprocess
        
        # Construct command
        cmd = ["sgpt", "--model", model]
        if system_prompt:
            cmd.extend(["--system", system_prompt])
        cmd.extend(["--temperature", str(temperature)])
        cmd.extend(["--max-tokens", str(max_tokens)])
        cmd.append(prompt)
        
        # Try a few times with exponential backoff
        max_retries = 3
        base_delay = 2  # Start with 2 second delay
        
        for retry in range(max_retries):
            try:
                start_time = time.time()
                result = subprocess.run(
                    cmd,
                    capture_output=True, 
                    text=True, 
                    timeout=60
                )
                latency = time.time() - start_time
                
                if result.returncode == 0:
                    # Estimate token counts based on character lengths
                    # This is a rough approximation
                    prompt_tokens = len(prompt) // 4
                    completion_tokens = len(result.stdout) // 4
                    total_tokens = prompt_tokens + completion_tokens
                    
                    # Calculate approximate cost based on model
                    # https://openai.com/pricing
                    cost_per_1k_input = 0.0
                    cost_per_1k_output = 0.0
                    
                    if model == "gpt-4o":
                        cost_per_1k_input = 0.01
                        cost_per_1k_output = 0.03
                    elif model == "gpt-4o-mini":
                        cost_per_1k_input = 0.00
                        cost_per_1k_output = 0.00
                    elif model == "gpt-3.5-turbo":
                        cost_per_1k_input = 0.0005
                        cost_per_1k_output = 0.0015
                    
                    cost = (prompt_tokens / 1000 * cost_per_1k_input) + (completion_tokens / 1000 * cost_per_1k_output)
                    
                    # Update token usage stats
                    self._update_token_usage(
                        provider="openai", 
                        model=model, 
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        cost=cost
                    )
                    
                    return {
                        "text": result.stdout.strip(),
                        "model": model,
                        "latency": latency,
                        "from_cache": False,
                        "tokens": {
                            "prompt": prompt_tokens,
                            "completion": completion_tokens,
                            "total": total_tokens
                        }
                    }
                else:
                    error_msg = f"OpenAI CLI error: {result.stderr}"
                    logger.error(error_msg)
                    
                    # Check for rate limit errors in stderr
                    if "rate limit" in result.stderr.lower():
                        if retry < max_retries - 1:
                            delay = base_delay * (2 ** retry)
                            logger.info(f"Rate limited. Retrying in {delay} seconds...")
                            time.sleep(delay)
                            continue
                    
                    raise Exception(error_msg)
            except subprocess.TimeoutExpired:
                logger.warning(f"OpenAI request timed out (retry {retry+1}/{max_retries})")
                if retry == max_retries - 1:
                    raise
                delay = base_delay * (2 ** retry)
                time.sleep(delay)
        
        # If we get here, we've exhausted our retries
        raise Exception(f"Failed to get response from OpenAI after {max_retries} retries")
    
    def get_best_model_for_task(self, task_description: str) -> str:
        """
        Determine the best model to use for a given task based on its description
        """
        # Simple keyword-based matching
        tactical_keywords = ["scan", "exploit", "command", "payload", "inject", "script", "check"]
        planning_keywords = ["plan", "strategy", "approach", "steps", "methodology", "procedure"]
        reasoning_keywords = ["analyze", "understand", "explain", "interpret", "evaluate", "assess"]
        
        # Count matches
        tactical_count = sum(1 for kw in tactical_keywords if kw in task_description.lower())
        planning_count = sum(1 for kw in planning_keywords if kw in task_description.lower())
        reasoning_count = sum(1 for kw in reasoning_keywords if kw in task_description.lower())
        
        # Find highest match
        counts = [
            (tactical_count, "tactical"),
            (planning_count, "planning"),
            (reasoning_count, "reasoning")
        ]
        
        _, best_role = max(counts, key=lambda x: x[0])
        return self.default_models[best_role]
    
    async def get_completion_async(self, prompt: str, model: str = None, role: str = None,
                                 system_prompt: str = None, temperature: float = 0.7,
                                 max_tokens: int = 2048, use_cache: bool = True,
                                 use_fallback: bool = True) -> Dict[str, Any]:
        """
        Async version of get_completion
        """
        # Create a function that runs the synchronous method in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: self.get_completion(
                prompt, model, role, system_prompt, temperature, 
                max_tokens, use_cache, use_fallback
            )
        )
    
    async def get_parallel_completions(self, prompts: List[str], models: List[str] = None,
                                     roles: List[str] = None, system_prompts: List[str] = None,
                                     temperature: float = 0.7, max_tokens: int = 2048,
                                     use_cache: bool = True, use_fallback: bool = True) -> List[Dict[str, Any]]:
        """
        Get completions for multiple prompts in parallel
        """
        # Prepare arguments for each call
        tasks = []
        
        for i, prompt in enumerate(prompts):
            model = models[i] if models and i < len(models) else None
            role = roles[i] if roles and i < len(roles) else None
            system_prompt = system_prompts[i] if system_prompts and i < len(system_prompts) else None
            
            tasks.append(
                self.get_completion_async(
                    prompt, model, role, system_prompt, 
                    temperature, max_tokens, use_cache, use_fallback
                )
            )
        
        # Execute all tasks concurrently and return results
        return await asyncio.gather(*tasks)
    
    def get_token_usage_report(self) -> Dict[str, Any]:
        """Generate a report of token usage"""
        return {
            "usage": self.token_usage,
            "generated_at": datetime.now().isoformat()
        }
    
    def clear_cache(self):
        """Clear the response cache"""
        self.response_cache = {}
        if self.cache_file.exists():
            try:
                self.cache_file.unlink()
                logger.info("Response cache file deleted")
            except Exception as e:
                logger.error(f"Failed to delete cache file: {e}")
        logger.info("Response cache cleared")
        
    def get_diagnostics(self, include_models: bool = True) -> Dict[str, Any]:
        """
        Generate a comprehensive diagnostics report for the LLM Orchestrator
        
        Args:
            include_models (bool): Whether to include loaded model information
            
        Returns:
            dict: Comprehensive diagnostics information
        """
        # Get server health with a fresh check
        server_health = self._check_server_health()
        
        # Get cache stats
        cache_size = len(self.response_cache)
        cache_models = {}
        
        # Calculate cache usage per model
        for key, value in self.response_cache.items():
            model = value.get("model", "unknown")
            if model not in cache_models:
                cache_models[model] = 0
            cache_models[model] += 1
            
        # Get token usage statistics
        token_report = self.get_token_usage_report()
        
        # Calculate latency metrics if we have cached responses
        latency_metrics = {"min": 0, "max": 0, "avg": 0, "count": 0}
        latency_values = []
        
        for key, value in self.response_cache.items():
            if "latency" in value:
                latency_values.append(value["latency"])
                
        if latency_values:
            latency_metrics = {
                "min": min(latency_values),
                "max": max(latency_values),
                "avg": sum(latency_values) / len(latency_values),
                "count": len(latency_values)
            }
            
        # Build diagnostics report
        report = {
            "server_health": server_health,
            "cache": {
                "size": cache_size,
                "models": cache_models,
                "path": str(self.cache_file)
            },
            "tokens": token_report,
            "latency": latency_metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        # Include model information if requested
        if include_models:
            try:
                if self.server_health["ollama"]["status"]:
                    response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
                    if response.status_code == 200:
                        report["models"] = {
                            "available": response.json(),
                            "default_models": self.default_models
                        }
                    else:
                        report["models"] = {
                            "available": "Failed to retrieve model list",
                            "default_models": self.default_models
                        }
                else:
                    report["models"] = {
                        "available": "Ollama server unavailable",
                        "default_models": self.default_models
                    }
            except Exception as e:
                report["models"] = {
                    "error": str(e),
                    "default_models": self.default_models
                }
                
        return report
    
    def get_latency_report(self, last_n: int = 100) -> Dict[str, Any]:
        """
        Generate a latency report for recent LLM calls
        
        Args:
            last_n (int): Number of recent calls to include in report
            
        Returns:
            dict: Latency metrics by model and overall
        """
        # Extract latency values from cache with timestamps
        latency_data = []
        
        for key, value in self.response_cache.items():
            if "latency" in value and "created_at" in value and "model" in value:
                latency_data.append({
                    "model": value["model"],
                    "latency": value["latency"],
                    "timestamp": value["created_at"]
                })
                
        # Sort by timestamp (newest first) and take last_n entries
        latency_data.sort(key=lambda x: x["timestamp"], reverse=True)
        latency_data = latency_data[:last_n]
        
        # Calculate overall metrics
        if latency_data:
            latency_values = [entry["latency"] for entry in latency_data]
            overall = {
                "min": min(latency_values),
                "max": max(latency_values),
                "avg": sum(latency_values) / len(latency_values),
                "count": len(latency_values),
                "p50": sorted(latency_values)[len(latency_values) // 2],
                "p90": sorted(latency_values)[int(len(latency_values) * 0.9)]
            }
        else:
            overall = {
                "min": 0,
                "max": 0,
                "avg": 0,
                "count": 0,
                "p50": 0,
                "p90": 0
            }
            
        # Calculate per-model metrics
        models = {}
        for entry in latency_data:
            model = entry["model"]
            if model not in models:
                models[model] = []
            models[model].append(entry["latency"])
            
        model_metrics = {}
        for model, values in models.items():
            model_metrics[model] = {
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "count": len(values),
                "p50": sorted(values)[len(values) // 2],
                "p90": sorted(values)[int(len(values) * 0.9)] if len(values) >= 10 else max(values)
            }
            
        return {
            "overall": overall,
            "models": model_metrics,
            "sample_size": len(latency_data),
            "generated_at": datetime.now().isoformat()
        }
    
    def health_check(self, test_model: str = None) -> Dict[str, Any]:
        """
        Run a complete health check on the LLM system
        
        Args:
            test_model (str, optional): Model to test with a simple prompt
            
        Returns:
            dict: Health check results
        """
        results = {
            "server_status": self._check_server_health(),
            "timestamp": datetime.now().isoformat()
        }
        
        # Test a model if specified
        if test_model:
            try:
                start_time = time.time()
                response = self.get_completion(
                    prompt="Hello, this is a test prompt. Please respond with 'OK'.",
                    model=test_model,
                    use_cache=False  # Don't use cache for health check
                )
                latency = time.time() - start_time
                
                results["model_test"] = {
                    "model": test_model,
                    "success": True,
                    "latency": latency,
                    "response_length": len(response.get("text", ""))
                }
            except Exception as e:
                results["model_test"] = {
                    "model": test_model,
                    "success": False,
                    "error": str(e)
                }
        
        # Check disk space for cache directory
        try:
            import shutil
            cache_usage = shutil.disk_usage(self.cache_dir)
            results["disk"] = {
                "total": cache_usage.total,
                "used": cache_usage.used,
                "free": cache_usage.free,
                "percent_used": cache_usage.used / cache_usage.total * 100
            }
        except Exception as e:
            results["disk"] = {"error": str(e)}
            
        # Check cache integrity
        try:
            cache_count = len(self.response_cache)
            results["cache_check"] = {
                "entries": cache_count,
                "status": "healthy" if cache_count >= 0 else "error"
            }
        except Exception as e:
            results["cache_check"] = {"error": str(e)}
            
        return results
            
    def run_latency_tests(self) -> Dict[str, Any]:
        """Run latency tests on all available models"""
        results = {}
        test_prompt = "Respond with a single word for testing purposes."
        
        # Get available models from Ollama
        ollama_models = []
        try:
            if self.server_health["ollama"]["status"]:
                response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
                if response.status_code == 200:
                    ollama_models = [model["name"] for model in response.json().get("models", [])]
        except Exception:
            pass  # Silently continue if we can't get models
            
        # Test each default model
        for role, model in self.default_models.items():
            # Skip if it's not available in Ollama
            if "GGUF" in model or "/" in model or ":" in model:
                if model not in ollama_models and not any(m.startswith(model.split(":")[0]) for m in ollama_models):
                    results[role] = {
                        "model": model,
                        "status": "unavailable",
                        "error": "Model not found in Ollama"
                    }
                    continue
            
            try:
                start_time = time.time()
                response = self.get_completion(
                    prompt=test_prompt,
                    model=model,
                    max_tokens=20,  # Small response for faster testing
                    use_cache=False  # Force actual API call
                )
                latency = time.time() - start_time
                
                results[role] = {
                    "model": model,
                    "status": "success",
                    "latency": latency,
                    "response": response.get("text", "")[:50] + ("..." if len(response.get("text", "")) > 50 else "")
                }
            except Exception as e:
                results[role] = {
                    "model": model, 
                    "status": "error",
                    "error": str(e)
                }
                
        # Test OpenAI fallback if configured
        if self.server_health["openai"]["status"]:
            try:
                start_time = time.time()
                response = self._get_openai_completion(
                    prompt=test_prompt,
                    max_tokens=20,
                    model="gpt-3.5-turbo"
                )
                latency = time.time() - start_time
                
                results["openai_fallback"] = {
                    "model": "gpt-3.5-turbo",
                    "status": "success",
                    "latency": latency,
                    "response": response.get("text", "")[:50] + ("..." if len(response.get("text", "")) > 50 else "")
                }
            except Exception as e:
                results["openai_fallback"] = {
                    "model": "gpt-3.5-turbo", 
                    "status": "error",
                    "error": str(e)
                }
                
        return results

# Singleton accessor - recommended way to get the LLMOrchestrator instance
def get_orchestrator() -> LLMOrchestrator:
    """Get the singleton LLMOrchestrator instance"""
    return LLMOrchestrator()

# Direct test function for CLI usage
def test_orchestrator():
    """Run a simple test of the LLM Orchestrator"""
    orchestrator = LLMOrchestrator()
    console.print("[bold]Testing LLM Orchestrator[/bold]")
    
    # Check server health
    health = orchestrator.get_server_health(force_check=True)
    console.print(f"Ollama server status: [{'green' if health['ollama']['status'] else 'red'}]"
                 f"{'ONLINE' if health['ollama']['status'] else 'OFFLINE'}[/]")
    console.print(f"OpenAI status: [{'green' if health['openai']['status'] else 'red'}]"
                 f"{'CONFIGURED' if health['openai']['status'] else 'NOT CONFIGURED'}[/]")
    
    # Test completion if servers available
    if health['ollama']['status']:
        try:
            console.print("\n[bold]Testing completion with local model...[/bold]")
            response = orchestrator.get_completion(
                "What are the three most important tools for network reconnaissance?", 
                role="tactical"
            )
            console.print(f"[green]Response:[/green] {response.get('text', '')}")
            console.print(f"[blue]Model used:[/blue] {response.get('model', 'unknown')}")
        except Exception as e:
            console.print(f"[red]Error testing completion:[/red] {e}")
    
    # Show cache stats
    console.print(f"\n[bold]Cache status:[/bold] {len(orchestrator.response_cache)} entries")
    
    # Show token usage
    usage = orchestrator.get_token_usage_report()
    console.print("\n[bold]Token usage:[/bold]")
    console.print(usage)
    
# Command line interface - enables calling from terminal
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            test_orchestrator()
        elif sys.argv[1] == "health":
            orchestrator = LLMOrchestrator()
            health = orchestrator.health_check()
            console.print(health)
        elif sys.argv[1] == "clear-cache":
            orchestrator = LLMOrchestrator()
            orchestrator.clear_cache()
            console.print("[green]Cache cleared.[/green]")
        elif sys.argv[1] == "run":
            if len(sys.argv) > 2:
                orchestrator = LLMOrchestrator()
                response = orchestrator.get_completion(sys.argv[2])
                console.print(response.get("text", ""))
        else:
            console.print("[yellow]Unknown command. Available commands: test, health, clear-cache, run[/yellow]")
    else:
        test_orchestrator()