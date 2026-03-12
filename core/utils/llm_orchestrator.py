#!/usr/bin/env python3
# core/utils/llm_orchestrator.py — Simplified LLM Orchestrator v2.0
# GPT-4o-mini Only | Simplified Architecture | No Local LLMs

import os
import json
import time
import logging
from typing import Dict, Any, Optional
from pathlib import Path

# Configure logging
logger = logging.getLogger("ariaska.llm_orchestrator")

try:
    from rich.console import Console
    console = Console()
except ImportError:
    class SimpleConsole:
        def print(self, *args, **kwargs):
            print(*args)
    console = SimpleConsole()

class LLMOrchestrator:
    """
    Simplified orchestration system using only GPT-4o-mini.
    
    Features:
    - Single GPT-4o-mini backend
    - Unified caching system
    - Simple retry logic
    - Performance tracking
    """
    
    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir or "core/memory/llm_cache"
        self.cache_path = os.path.join(self.cache_dir, "orchestrator_cache.json")
        self.cache = {}
        self.gpt_manager = None
        
        # Stats tracking
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "gpt_requests": 0,
            "errors": 0,
            "avg_response_time": 0.0,
            "total_tokens": 0
        }
        
        # Initialize
        self._load_cache()
        self._init_gpt_manager()
        
        logger.debug("LLM Orchestrator (local-llm) initialized")
    
    def _init_gpt_manager(self):
        """Initialize GPT-4o-mini manager."""
        try:
            from core.gpt_manager import GPTManager
            self.gpt_manager = GPTManager.get_instance()
            logger.debug("local-llm manager initialized")
        except Exception as e:
            logger.error(f"X Failed to initialize GPT manager: {e}")
            self.gpt_manager = None
    
    def _load_cache(self):
        """Load cache from disk."""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            if os.path.exists(self.cache_path):
                with open(self.cache_path, 'r') as f:
                    self.cache = json.load(f)
                logger.info(f"Loaded {len(self.cache)} cached responses")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            self.cache = {}
    
    def _save_cache(self):
        """Save cache to disk."""
        try:
            with open(self.cache_path, 'w') as f:
                json.dump(self.cache, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _get_cache_key(self, task_type: str, prompt: str, agent_id: str) -> str:
        """Generate cache key for request."""
        return f"{task_type}|{agent_id}|{prompt[:100]}"
    
    def route_task(
        self,
        task_type: str,
        prompt: str,
        agent_id: str = "system",
        max_tokens: int = 500,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Route task to GPT-4o-mini.
        
        Args:
            task_type: Type of task (tactical, strategic, analysis, etc.)
            prompt: The prompt to send
            agent_id: ID of requesting agent
            max_tokens: Maximum tokens to generate
            use_cache: Whether to use caching
            
        Returns:
            Dict: Response with content, success, metadata, etc.
        """
        start_time = time.time()
        self.stats["total_requests"] += 1
        
        # Check cache first
        if use_cache:
            cache_key = self._get_cache_key(task_type, prompt, agent_id)
            if cache_key in self.cache:
                self.stats["cache_hits"] += 1
                logger.debug(f"Cache hit for {agent_id}: {prompt[:50]}...")
                return {
                    "content": self.cache[cache_key]["content"],
                    "success": True,
                    "model": "cache",
                    "tokens": self.cache[cache_key].get("tokens", 0),
                    "response_time": 0.001,
                    "cached": True
                }
        
        # Check if GPT manager is available
        if not self.gpt_manager:
            self.stats["errors"] += 1
            return {
                "content": "Error: local-llm not available",
                "success": False,
                "model": "none",
                "tokens": 0,
                "response_time": time.time() - start_time,
                "cached": False
            }
        
        try:
            # Make request to GPT-4o-mini
            self.stats["gpt_requests"] += 1
            
            response = self.gpt_manager.gpt_request(
                prompt,
                task_type=task_type,
                agent_id=agent_id,
                max_tokens=max_tokens
            )
            
            if response and len(str(response).strip()) > 0:
                response_time = time.time() - start_time
                
                # Estimate tokens
                tokens = max(1, (len(prompt) + len(str(response))) // 4)
                self.stats["total_tokens"] += tokens
                
                # Update response time average
                if self.stats["total_requests"] > 0:
                    self.stats["avg_response_time"] = (
                        (self.stats["avg_response_time"] * (self.stats["total_requests"] - 1) + response_time) 
                        / self.stats["total_requests"]
                    )
                
                # Cache successful response
                if use_cache:
                    cache_key = self._get_cache_key(task_type, prompt, agent_id)
                    self.cache[cache_key] = {
                        "content": response,
                        "tokens": tokens,
                        "timestamp": time.time()
                    }
                    
                    # Prune cache if too large
                    if len(self.cache) > 1000:
                        # Keep newest 800 entries
                        sorted_cache = sorted(
                            self.cache.items(),
                            key=lambda x: x[1].get("timestamp", 0)
                        )
                        self.cache = dict(sorted_cache[-800:])
                    
                    # Save cache periodically
                    if self.stats["total_requests"] % 20 == 0:
                        self._save_cache()
                
                logger.debug(f"✅ GPT response for {agent_id}: {str(response)[:100]}...")
                
                return {
                    "content": response,
                    "success": True,
                    "model": "local-llm",
                    "tokens": tokens,
                    "response_time": response_time,
                    "cached": False
                }
            else:
                raise Exception("Empty response from GPT")
                
        except Exception as e:
            self.stats["errors"] += 1
            error_msg = str(e)
            logger.error(f"❌ GPT request failed for {agent_id}: {error_msg}")
            
            return {
                "content": f"Error: {error_msg}",
                "success": False,
                "model": "none",
                "tokens": 0,
                "response_time": time.time() - start_time,
                "cached": False
            }
    
    def request_strategy(
        self,
        context: str,
        objective: str,
        agent_id: str = "system"
    ) -> str:
        """Request strategic advice."""
        prompt = f"Context: {context}\nObjective: {objective}\nProvide strategic cybersecurity advice:"
        
        response = self.route_task(
            task_type="strategic",
            prompt=prompt,
            agent_id=agent_id,
            max_tokens=400
        )
        
        return str(response["content"]) if response["success"] else "Strategy request failed"
    
    def dual_llm_feedback(
        self,
        prompt: str,
        agent_id: str = "system"
    ) -> Dict[str, Any]:
        """
        Simplified dual feedback using GPT-4o-mini twice with different approaches.
        """
        # First response - direct approach
        response1 = self.route_task(
            task_type="analysis",
            prompt=f"Direct analysis: {prompt}",
            agent_id=agent_id,
            use_cache=False
        )
        
        # Second response - critical review
        response2 = self.route_task(
            task_type="analysis", 
            prompt=f"Critical review and alternative perspective: {prompt}",
            agent_id=agent_id,
            use_cache=False
        )
        
        return {
            "primary_response": response1["content"] if response1["success"] else "Primary analysis failed",
            "secondary_response": response2["content"] if response2["success"] else "Secondary analysis failed",
            "success": response1["success"] and response2["success"],
            "metadata": {
                "primary_tokens": response1.get("tokens", 0),
                "secondary_tokens": response2.get("tokens", 0),
                "total_time": response1.get("response_time", 0) + response2.get("response_time", 0)
            }
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Check orchestrator health."""
        gpt_available = self.gpt_manager is not None
        
        if gpt_available:
            try:
                test_response = self.route_task(
                    task_type="test",
                    prompt="Health check: respond with 'OK'",
                    agent_id="health_check",
                    max_tokens=10,
                    use_cache=False
                )
                gpt_working = test_response["success"]
            except Exception:
                gpt_working = False
        else:
            gpt_working = False
        
        return {
            "gpt_manager_available": gpt_available,
            "gpt_working": gpt_working,
            "cache_size": len(self.cache),
            "stats": dict(self.stats)
        }
    
    def display_stats(self):
        """Display orchestrator statistics."""
        console.print("[bold cyan]LLM Orchestrator Stats[/bold cyan]")
        console.print(f"Total Requests: {self.stats['total_requests']}")
        console.print(f"Cache Hits: {self.stats['cache_hits']} ({self.stats['cache_hits']/max(1,self.stats['total_requests'])*100:.1f}%)")
        console.print(f"GPT Requests: {self.stats['gpt_requests']}")
        console.print(f"Errors: {self.stats['errors']}")
        console.print(f"Total Tokens: {self.stats['total_tokens']}")
        console.print(f"Avg Response Time: {self.stats['avg_response_time']:.3f}s")
        console.print(f"Cache Size: {len(self.cache)} entries")
    
    def clear_cache(self):
        """Clear the response cache."""
        self.cache = {}
        try:
            if os.path.exists(self.cache_path):
                os.remove(self.cache_path)
        except Exception as e:
            logger.warning(f"Failed to remove cache file: {e}")
        logger.info("Cache cleared")

# Testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    orchestrator = LLMOrchestrator()
    
    # Test basic functionality
    console.print("[bold]Testing LLM Orchestrator[/bold]")
    
    response = orchestrator.route_task(
        task_type="tactical",
        prompt="Generate command to scan ports on 192.168.1.1",
        agent_id="TestAgent"
    )
    
    console.print(f"Response: {response['content']}")
    console.print(f"Success: {response['success']}")
    console.print(f"Model: {response['model']}")
    
    # Display stats
    orchestrator.display_stats()
    
    # Health check
    health = orchestrator.health_check()
    console.print(f"Health: {health}")
