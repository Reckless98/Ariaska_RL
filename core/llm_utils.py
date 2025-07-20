#!/usr/bin/env python3
# llm_utils.py - Utility functions for GPT-4o-mini only components
# This file provides centralized access to GPTManager (no local LLMs)

import os
import logging
import importlib
from typing import Any, Dict, Optional, Union, List, Tuple
from rich.console import Console

console = Console()
logger = logging.getLogger("ariaska.llm_utils")

# Singleton instances storage for lazy loading
_instances = {}

def get_gpt_manager():
    """Get singleton instance of GPTManager using lazy loading"""
    if "gpt_manager" not in _instances:
        try:
            from core.gpt_manager import GPTManager
            _instances["gpt_manager"] = GPTManager()
            logger.info("GPT Manager loaded")
        except ImportError as e:
            logger.error(f"Failed to import GPTManager: {e}")
            console.print(f"[red]❌ Failed to import GPTManager: {e}[/red]")
            return None
    return _instances.get("gpt_manager")

def get_completion(prompt: str, task_type: str = "general", 
                 agent_id: str = "unknown", max_tokens: int = 150) -> str:
    """
    Get completion using GPT-4o-mini only
    
    Args:
        prompt: The prompt text
        task_type: Type of task (tactical, defensive, reconnaissance, analysis, planning, general)
        agent_id: ID of the requesting agent
        max_tokens: Maximum tokens in response
        
    Returns:
        Response string from GPT-4o-mini
    """
    gpt_manager = get_gpt_manager()
    if not gpt_manager:
        return "echo 'GPT Manager unavailable'"
    
    return gpt_manager.gpt_request(prompt, task_type, agent_id, max_tokens)

def get_learning_feedback(command: str, result: str, reward: float, agent_id: str) -> str:
    """Get learning feedback for agents"""
    gpt_manager = get_gpt_manager()
    if not gpt_manager:
        return "No feedback available"
    
    return gpt_manager.get_learning_feedback(command, result, reward, agent_id)

def get_training_hint(phase: str, previous_commands: list, agent_id: str) -> str:
    """Get training hints for agents"""
    gpt_manager = get_gpt_manager()
    if not gpt_manager:
        return "echo 'No hints available'"
    
    return gpt_manager.get_training_hint(phase, previous_commands, agent_id)

def get_strategic_insight(context: dict, agent_id: str) -> str:
    """Get strategic insights for agents"""
    gpt_manager = get_gpt_manager()
    if not gpt_manager:
        return "No strategic insight available"
    
    return gpt_manager.get_strategic_insight(context, agent_id)

def analyze_command_output(command: str, output: str, agent_id: str) -> str:
    """Analyze command output"""
    gpt_manager = get_gpt_manager()
    if not gpt_manager:
        return "No analysis available"
    
    return gpt_manager.analyze_command_output(command, output, agent_id)

def test_gpt_connectivity() -> dict:
    """Test GPT connectivity"""
    gpt_manager = get_gpt_manager()
    if not gpt_manager:
        return {"status": "failed", "error": "GPT Manager unavailable"}
    
    return gpt_manager.test_connectivity()

def get_gpt_stats() -> dict:
    """Get GPT usage statistics"""
    gpt_manager = get_gpt_manager()
    if not gpt_manager:
        return {"error": "GPT Manager unavailable"}
    
    return gpt_manager.get_global_stats()

def reset_episode_tokens():
    """Reset token count for new episode"""
    gpt_manager = get_gpt_manager()
    if gpt_manager:
        gpt_manager.reset_token_count()

# For backwards compatibility - deprecated functions that previously used LocalLLM
def get_local_llm_manager(model_name: Optional[str] = None):
    """DEPRECATED: Local LLMs removed. Use get_gpt_manager() instead."""
    logger.warning("get_local_llm_manager is deprecated. All LLM requests now use GPT-4o-mini.")
    return get_gpt_manager()

def get_llm_orchestrator():
    """DEPRECATED: Use get_gpt_manager() instead."""
    logger.warning("get_llm_orchestrator is deprecated. Use get_gpt_manager() instead.")
    return get_gpt_manager()
    """
    Get a completion from an LLM model with automatic fallback
    
    This function is the main entry point for all LLM completions in the system
    and will use the LLMOrchestrator to get the response.
    """
    orchestrator = get_llm_orchestrator()
    if orchestrator is None:
        logger.error("LLM Orchestrator not available")
        return {"text": "Error: LLM Orchestrator not available", "error": True}
        
    return orchestrator.get_completion(
        prompt=prompt,
        model=model,
        role=role,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        use_cache=use_cache,
        use_fallback=use_fallback
    )

def get_best_model_for_task(task_description: str) -> str:
    """
    Get the best model for a given task description
    """
    orchestrator = get_llm_orchestrator()
    if orchestrator is None:
        logger.error("LLM Orchestrator not available")
        return "fallback"
        
    return orchestrator.get_best_model_for_task(task_description)

def clear_llm_cache():
    """Clear all LLM caches in the system"""
    orchestrator = get_llm_orchestrator()
    if orchestrator:
        orchestrator.clear_cache()
        
    gpt = get_gpt_manager()
    if gpt:
        gpt.clear_cache()
        
    return {"status": "Cache cleared"}

def get_server_health():
    """Check LLM server health"""
    orchestrator = get_llm_orchestrator()
    if orchestrator:
        return orchestrator.get_server_health(force_check=True)
    return {"status": "Orchestrator not available"}

def run_llm_health_check(detailed: bool = False):
    """Run a comprehensive health check of all LLM components"""
    orchestrator = get_llm_orchestrator()
    if orchestrator:
        return orchestrator.health_check(detailed=detailed)
    return {"status": "critical", "error": "Orchestrator not available"}

# Initialize singletons (can be commented out to use pure lazy loading)
get_llm_orchestrator()
get_gpt_manager()