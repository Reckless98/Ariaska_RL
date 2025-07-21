#!/usr/bin/env python3
# core/utils/llm_router.py — ARIASKA LLM Router v2.0 SIMPLIFIED
# 🧠 GPT-4o-mini Only | 🔄 Simplified Architecture | 📊 Token Optimization

import os
import time
import json
import threading
import logging
from typing import Any, Dict, Optional, Tuple
from pathlib import Path
from rich.console import Console

# Configure logging
logger = logging.getLogger("ariaska.llm_router")
console = Console()

class LLMRouter:
    """
    Simplified router for GPT-4o-mini only.
    
    Key features:
    - Single GPT-4o-mini backend via GPTManager
    - Intelligent caching for performance
    - Comprehensive error handling
    - Thread safety for concurrent usage
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(LLMRouter, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, cache_path=None):
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self.cache_path = cache_path or os.path.join("core", "memories", "llm_cache", "router_cache.json")
        self.cache = {}
        self.cache_lock = threading.Lock()
        self.gpt_manager = None
        
        # Stats tracking
        self.stats = {
            "requests": 0,
            "cache_hits": 0,
            "errors": 0,
            "tokens": {"total": 0},
            "avg_latency": 0,
            "response_times": []
        }
        
        # Load cache
        self._load_cache()
        
        # Initialize GPT backend
        self._init_gpt_backend()
        
        self._initialized = True
        logger.info("Brain: Simplified LLM Router (GPT-4o-mini only) initialized")
        
    def _init_gpt_backend(self):
        """Initialize GPT-4o-mini backend."""
        try:
            from core.gpt_manager import GPTManager
            self.gpt_manager = GPTManager()
            logger.info("Checkmark: GPT-4o-mini backend initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize GPT-4o-mini: {e}")
            self.gpt_manager = None
    
    def _load_cache(self):
        """Load the response cache from disk."""
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            
            if os.path.exists(self.cache_path):
                with open(self.cache_path, 'r') as f:
                    cache_data = json.load(f)
                    
                # Filter out expired cache entries (24 hours)
                now = time.time()
                ttl_seconds = 24 * 3600
                
                self.cache = {
                    k: v for k, v in cache_data.items()
                    if not v.get("timestamp") or now - v["timestamp"] < ttl_seconds
                }
                
                logger.info(f"Loaded cache with {len(self.cache)} valid entries")
            else:
                self.cache = {}
                
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            self.cache = {}
    
    def _save_cache(self):
        """Save the response cache to disk."""
        try:
            with self.cache_lock:
                # Prune cache if too large (keep 1000 entries)
                if len(self.cache) > 1000:
                    sorted_cache = sorted(
                        self.cache.items(),
                        key=lambda x: x[1].get("timestamp", 0)
                    )
                    self.cache = dict(sorted_cache[-1000:])
                
                with open(self.cache_path, 'w') as f:
                    json.dump(self.cache, f)
                    
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def request(
        self,
        prompt: str,
        task_type: str = "general",
        agent_id: str = "router",
        require_json: bool = False,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Send request to GPT-4o-mini.
        
        Args:
            prompt: The prompt to send
            task_type: Task classification for specialized prompting
            agent_id: Agent making the request
            require_json: Whether the response should be JSON
            timeout: Request timeout in seconds
            
        Returns:
            Dict: Response with content, success, tokens, latency, etc.
        """
        start_time = time.time()
        self.stats["requests"] += 1
        
        # Create cache key
        json_flag = "-json" if require_json else ""
        cache_key = f"{task_type}{json_flag}|{prompt[:100]}"
        
        # Check cache first
        with self.cache_lock:
            if cache_key in self.cache:
                self.stats["cache_hits"] += 1
                cached = self.cache[cache_key]
                logger.debug(f"Cache hit for: {prompt[:50]}...")
                return {
                    "content": cached.get("content", ""),
                    "success": True,
                    "model_used": "cache",
                    "tokens": cached.get("tokens", {"total": 0}),
                    "latency": 0.001,
                    "metadata": cached.get("metadata", {})
                }
        
        # Check if GPT backend is available
        if not self.gpt_manager:
            self.stats["errors"] += 1
            return {
                "content": "Error: GPT-4o-mini backend not available",
                "success": False,
                "model_used": "none",
                "tokens": {"total": 0},
                "latency": time.time() - start_time,
                "metadata": {"error": "Backend unavailable"}
            }
        
        try:
            # Make request to GPT-4o-mini
            if require_json:
                # For JSON requests, add instruction to prompt
                json_prompt = f"{prompt}\n\nPlease respond with valid JSON only."
                content = self.gpt_manager.gpt_request(
                    json_prompt,
                    task_type=task_type,
                    agent_id=agent_id,
                    max_tokens=800
                )
                
                # Try to parse as JSON
                try:
                    import re
                    # Extract JSON block if in markdown format
                    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```|(\{.*?\})', str(content), re.DOTALL)
                    if json_match:
                        content = json.loads(json_match.group(1) or json_match.group(2))
                    else:
                        content = json.loads(str(content))
                    success = True
                except Exception:
                    content = {"error": "Failed to parse JSON", "raw": str(content)}
                    success = False
            else:
                # Standard text request
                content = self.gpt_manager.gpt_request(
                    prompt,
                    task_type=task_type,
                    agent_id=agent_id,
                    max_tokens=500
                )
                success = bool(content and len(str(content).strip()) > 0)
            
            if success:
                # Estimate token usage
                prompt_tokens = max(1, len(prompt) // 4)
                response_tokens = max(1, len(str(content)) // 4)
                tokens = {
                    "total": prompt_tokens + response_tokens,
                    "prompt": prompt_tokens,
                    "completion": response_tokens
                }
                
                # Update stats
                self.stats["tokens"]["total"] += tokens["total"]
                elapsed = time.time() - start_time
                self.stats["response_times"].append(elapsed)
                if len(self.stats["response_times"]) > 100:
                    self.stats["response_times"] = self.stats["response_times"][-100:]
                self.stats["avg_latency"] = sum(self.stats["response_times"]) / len(self.stats["response_times"])
                
                # Cache successful response
                with self.cache_lock:
                    self.cache[cache_key] = {
                        "content": content,
                        "tokens": tokens,
                        "timestamp": time.time(),
                        "metadata": {"task_type": task_type}
                    }
                    
                    # Save cache periodically
                    if self.stats["requests"] % 10 == 0:
                        threading.Thread(target=self._save_cache).start()
                
                logger.info(f"✅ GPT-4o-mini request successful ({elapsed:.2f}s)")
                return {
                    "content": content,
                    "success": True,
                    "model_used": "gpt-4o-mini",
                    "tokens": tokens,
                    "latency": elapsed,
                    "metadata": {"task_type": task_type}
                }
            else:
                raise Exception("Empty or invalid response")
                
        except Exception as e:
            self.stats["errors"] += 1
            error_message = str(e)
            logger.error(f"❌ GPT-4o-mini request failed: {error_message}")
            
            return {
                "content": f"Error: {error_message}",
                "success": False,
                "model_used": "none",
                "tokens": {"total": 0},
                "latency": time.time() - start_time,
                "metadata": {"error": error_message}
            }
    
    def generate_command(
        self,
        task_description: str,
        task_type: str = "general",
        timeout: Optional[int] = None
    ) -> Tuple[bool, str]:
        """Generate a command for a given task."""
        response = self.request(
            prompt=f"Generate a cybersecurity command for: {task_description}",
            task_type=task_type,
            timeout=timeout
        )
        
        if response["success"]:
            command = str(response["content"]).strip()
            if command and len(command.split()) >= 1:
                return True, command
            else:
                return False, "Generated command was empty"
        else:
            return False, f"Command generation failed: {response['content']}"
    
    def strategic_analysis(
        self,
        context: str,
        question: str,
        timeout: Optional[int] = None
    ) -> str:
        """Perform strategic analysis."""
        prompt = f"Context:\n{context}\n\nQuestion:\n{question}\n\nProvide strategic analysis:"
        
        response = self.request(
            prompt=prompt,
            task_type="analysis",
            timeout=timeout
        )
        
        return str(response["content"])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
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

# Global router instance
_router_instance = None

def get_router() -> LLMRouter:
    """Get the global router instance."""
    global _router_instance
    if _router_instance is None:
        _router_instance = LLMRouter()
    return _router_instance

# Convenience functions
def query(
    prompt: str,
    task_type: str = "general",
    require_json: bool = False
) -> str:
    """Simple query function."""
    router = get_router()
    response = router.request(
        prompt=prompt,
        task_type=task_type,
        require_json=require_json
    )
    return str(response["content"])

def generate_command(task_description: str, task_type: str = "general") -> str:
    """Generate a command for a specific task."""
    router = get_router()
    success, command = router.generate_command(
        task_description=task_description,
        task_type=task_type
    )
    return command if success else ""

# Testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    router = LLMRouter()
    
    # Test basic query
    print("\nTesting basic query...")
    response = router.request(
        prompt="List files in current directory",
        task_type="tactical"
    )
    print(f"Response: {response['content']}")
    print(f"Success: {response['success']}")
    
    # Test stats
    print("\nRouter stats:")
    for key, value in router.get_stats().items():
        print(f"{key}: {value}")
