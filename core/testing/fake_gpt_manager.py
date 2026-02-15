#!/usr/bin/env python3
"""
core/testing/fake_gpt_manager.py — Mock GPT client for testing

Provides deterministic, seeded responses for unit tests without requiring
an actual OpenAI API key.

Usage:
    from core.testing import FakeGPTManager
    
    gpt = FakeGPTManager(seed=42)
    result = gpt.request("RedAgent", "tactical", "Get a command for recon")
    assert result["offline"] == False  # It's a fake, but acts like online
    assert result["response"] == "nmap -sV 10.10.10.10"  # Deterministic
"""

import hashlib
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class FakeGPTManager:
    """
    Mock GPTManager for testing without API calls.
    
    Features:
    - Deterministic responses based on seed
    - Role-based routing (mimics real manager)
    - No network calls
    - Tracks all requests for assertions
    """
    
    # Deterministic responses by task type
    RESPONSES = {
        "tactical": [
            "nmap -sV 10.10.10.10",
            "gobuster dir -u http://10.10.10.10 -w /usr/share/dirb/wordlists/common.txt",
            "ssh admin@10.10.10.10",
            "curl http://10.10.10.10/robots.txt",
            "nc -nv 10.10.10.10 22",
        ],
        "defensive": [
            "netstat -tulnp",
            "ss -tulnp",
            "iptables -L -n",
            "fail2ban-client status",
            "last -a",
        ],
        "reconnaissance": [
            "ping -c 4 10.10.10.10",
            "nslookup 10.10.10.10",
            "whois 10.10.10.10",
            "traceroute 10.10.10.10",
            "dig 10.10.10.10",
        ],
        "analysis": [
            "Target shows open ports 22, 80, 443.",
            "Service enumeration complete. SSH and HTTP detected.",
            "Vulnerability scan indicates outdated OpenSSH version.",
            "Web server running Apache 2.4 with default configuration.",
            "No critical vulnerabilities detected in initial scan.",
        ],
        "reasoning": [
            "Based on open ports, prioritize HTTP enumeration before SSH brute force.",
            "The service versions suggest this is a legacy system with known CVEs.",
            "Recommend stealth approach due to IDS indicators.",
            "Attack surface analysis complete. Web vector most promising.",
            "Lateral movement potential high given credential reuse patterns.",
        ],
        "postmortem": [
            '{"skill_cards": [], "failure_patterns": [], "next_experiments": []}',
            '{"skill_cards": [{"id": "skill_001", "if_condition": "port 22 open", "then_action": "ssh enumeration"}], "failure_patterns": [], "next_experiments": []}',
        ],
        "diversify": [
            "masscan -p1-65535 10.10.10.10",
            "rustscan -a 10.10.10.10",
            "unicornscan 10.10.10.10",
        ],
        "general": [
            "Command executed successfully.",
            "Operation complete.",
            "Task finished.",
        ],
    }
    
    # Model routing (mirrors real GPTManager)
    MODEL_MAP = {
        "red": "gpt-5.1-codex-mini",
        "orion": "gpt-5.1-codex-mini",
        "scout": "gpt-5.1-codex-mini",
        "shadow": "gpt-5.1-codex-mini",
        "blue": "gpt-5.1-codex-mini",
        "tactical": "gpt-5.1-codex-mini",
        "strategic": "gpt-5.1-codex-mini",
        "reasoning": "gpt-5.1-codex-mini",
        "analysis": "gpt-5.1-codex-mini",
        "postmortem": "gpt-5.1-codex",
    }
    
    def __init__(self, seed: int = 42):
        """
        Initialize FakeGPTManager.
        
        Args:
            seed: Random seed for deterministic responses
        """
        self.seed = seed
        self._request_count = 0
        self._requests: List[Dict[str, Any]] = []
        
        # Mimic real GPTManager interface
        self.primary_model = "gpt-5.1-codex-mini"
        self.fallback_model = "gpt-4o-mini"
        self.nano_model = "gpt-5.1-codex-mini"
        self.postmortem_model = "gpt-5.1-codex"
        self.tokens_used = 0
        self.token_limit = 3000
        
        # Stats tracking
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "failures": 0,
            "tokens_used_total": 0
        }
        
        # Attributes that real GPTManager exposes
        self.tokens_by_agent: Dict[str, int] = {}
        self.current_episode_id: Optional[str] = None
        self.token_limit_per_agent = 1000
        self._episode_cost_usd = 0.0
        
        logger.info(f"FakeGPTManager initialized with seed={seed}")
    
    def is_configured(self) -> bool:
        """Always returns True for fake manager."""
        return True
    
    def is_offline(self) -> bool:
        """Fake manager is never truly offline."""
        return False
    
    def request(
        self,
        role: str,
        task_type: str,
        prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        max_tokens: int = 150
    ) -> Dict[str, Any]:
        """
        Get a deterministic response based on seed and request count.
        
        Args:
            role: Agent role
            task_type: Type of task
            prompt: The prompt (used for determinism)
            schema: Optional schema (ignored in fake)
            max_tokens: Max tokens (ignored in fake)
            
        Returns:
            Dict matching real GPTManager.request() format
        """
        self._request_count += 1
        self.stats["total_requests"] += 1
        
        # Track request for assertions
        self._requests.append({
            "role": role,
            "task_type": task_type,
            "prompt": prompt,
            "count": self._request_count
        })
        
        # Get model for role
        model = self.get_model_for_role(role, task_type)
        
        # Get deterministic response
        response = self._get_deterministic_response(task_type, prompt)
        
        # Simulate token usage
        self.tokens_used += len(response.split()) * 2
        self.stats["tokens_used_total"] += len(response.split()) * 2
        
        return {
            "success": True,
            "response": response,
            "model_used": model,
            "offline": False
        }
    
    def gpt_request(
        self,
        prompt: str,
        task_type: str = "general",
        agent_id: str = "unknown",
        max_tokens: int = 150,
        model: Optional[str] = None,
        allow_fallback: bool = True
    ) -> str:
        """
        Compatibility method matching real GPTManager.gpt_request().
        
        Returns just the response string.
        """
        result = self.request(
            role=agent_id,
            task_type=task_type,
            prompt=prompt,
            max_tokens=max_tokens
        )
        return result["response"]
    
    def get_model_for_role(self, agent_id: str = None, task_type: str = None) -> str:
        """Get model based on role/task (deterministic)."""
        if agent_id:
            agent_lower = agent_id.lower()
            for key in ["red", "orion", "scout", "shadow", "blue"]:
                if key in agent_lower:
                    return self.MODEL_MAP.get(key, self.primary_model)
        
        if task_type:
            return self.MODEL_MAP.get(task_type, self.primary_model)
        
        return self.primary_model
    
    def _get_deterministic_response(self, task_type: str, prompt: str) -> str:
        """Get a deterministic response based on seed, task, and prompt."""
        responses = self.RESPONSES.get(task_type, self.RESPONSES["general"])
        
        # Hash seed + request count + prompt prefix for determinism
        hash_input = f"{self.seed}:{self._request_count}:{prompt[:50]}"
        hash_val = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        
        index = hash_val % len(responses)
        return responses[index]
    
    def reset_token_count(self):
        """Reset token count for new episode."""
        self.tokens_used = 0
    
    def can_make_request(self) -> bool:
        """Check if within token limits."""
        return self.tokens_used < self.token_limit
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            **self.stats,
            "tokens_used_current_episode": self.tokens_used,
            "cache_size": 0
        }
    
    def get_requests(self) -> List[Dict[str, Any]]:
        """Get all recorded requests for test assertions."""
        return self._requests.copy()
    
    def clear_requests(self):
        """Clear recorded requests."""
        self._requests.clear()
        self._request_count = 0
    
    # Compatibility methods
    def dual_llm_feedback(self, prompt: str, agent_id: str = "unknown",
                         task_type: str = "tactical") -> str:
        """Compatibility with old dual_llm_feedback calls."""
        return self.gpt_request(prompt, task_type, agent_id)
    
    def get_token_usage(self) -> int:
        """Get current token usage."""
        return self.tokens_used
    
    def smart_decision(self, task_type: str, task_description: str,
                      agent_id: str = "unknown", use_gpt: bool = True) -> str:
        """Compatibility with smart_decision calls."""
        return self.gpt_request(task_description, task_type, agent_id)
    
    def get_learning_feedback(self, command: str, result: str, reward: float,
                             agent_id: str) -> str:
        """Get learning feedback (deterministic)."""
        return "Good progress. Consider varying your approach."
    
    def get_training_hint(self, phase: str, previous_commands: list,
                         agent_id: str) -> str:
        """Get training hint (deterministic)."""
        return self.gpt_request(f"Hint for {phase}", "tactical", agent_id)
    
    def reset_episode(self, episode_id: Optional[int] = None, agent_name: Optional[str] = None) -> None:
        """Reset token counters for a new episode (mirrors real GPTManager)."""
        if agent_name:
            self.tokens_by_agent[agent_name] = 0
        else:
            self.tokens_used = 0
            self.tokens_by_agent = {}
            self.current_episode_id = str(episode_id) if episode_id is not None else None
    
    def reset_episode_tokens(self) -> None:
        """Reset per-episode token counters."""
        self.tokens_used = 0
        self.tokens_by_agent.clear()
        self._episode_cost_usd = 0.0
    
    def has_venice(self) -> bool:
        """Fake manager has no Venice integration."""
        return False
    
    def get_budget_status(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """Get budget status (always within budget for fake)."""
        return {
            "within_budget": True,
            "tokens_used": self.tokens_used,
            "tokens_remaining": max(0, self.token_limit - self.tokens_used),
        }
    
    def cleanup(self):
        """Clean shutdown (no-op for fake)."""
        pass
