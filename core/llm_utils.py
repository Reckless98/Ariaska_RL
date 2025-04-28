#!/usr/bin/env python3
# llm_utils.py - Utility functions and imports for LLM components
# This file helps resolve circular imports between LLM components

import os
import logging
import importlib
from typing import Any, Dict, Optional, Union, List, Tuple
from rich.console import Console

console = Console()
logger = logging.getLogger("ariaska.llm_utils")

# Singleton instances storage for lazy loading
_instances = {}

def get_llm_orchestrator():
    """Get singleton instance of LLMOrchestrator using lazy loading"""
    if "orchestrator" not in _instances:
        try:
            from Ariaska_RL.core.llm_orchestrator import LLMOrchestrator
            _instances["orchestrator"] = LLMOrchestrator()
            logger.info("LLM Orchestrator loaded")
        except ImportError:
            try:
                # Alternative import path
                from core.llm_orchestrator import LLMOrchestrator
                _instances["orchestrator"] = LLMOrchestrator()
                logger.info("LLM Orchestrator loaded (alternative path)")
            except ImportError as e:
                logger.error(f"Failed to import LLMOrchestrator: {e}")
                console.print(f"[red]❌ Failed to import LLMOrchestrator: {e}[/red]")
                return None
    return _instances.get("orchestrator")

def get_gpt_manager():
    """Get singleton instance of GPTManager using lazy loading"""
    if "gpt_manager" not in _instances:
        try:
            from Ariaska_RL.core.gpt_manager import GPTManager
            _instances["gpt_manager"] = GPTManager()
            logger.info("GPT Manager loaded")
        except ImportError:
            try:
                # Alternative import path
                from core.gpt_manager import GPTManager
                _instances["gpt_manager"] = GPTManager()
                logger.info("GPT Manager loaded (alternative path)")
            except ImportError as e:
                logger.error(f"Failed to import GPTManager: {e}")
                console.print(f"[red]❌ Failed to import GPTManager: {e}[/red]")
                return None
    return _instances.get("gpt_manager")

def get_local_llm_manager(model_name: Optional[str] = None):
    """Get instance of LocalLLMManager with specific model"""
    key = f"local_llm_{model_name}" if model_name else "local_llm"
    if key not in _instances:
        try:
            from Ariaska_RL.core.utils.local_llm_manager import LocalLLMManager
            _instances[key] = LocalLLMManager(model_name=model_name)
            logger.info(f"Local LLM Manager loaded for model: {model_name or 'default'}")
        except ImportError:
            try:
                # Alternative import path
                from core.utils.local_llm_manager import LocalLLMManager
                _instances[key] = LocalLLMManager(model_name=model_name)
                logger.info(f"Local LLM Manager loaded for model: {model_name or 'default'} (alternative path)")
            except ImportError as e:
                logger.error(f"Failed to import LocalLLMManager: {e}")
                console.print(f"[red]❌ Failed to import LocalLLMManager: {e}[/red]")
                return None
    return _instances.get(key)

def get_completion(prompt: str, model: str = None, role: str = None, 
                 system_prompt: str = None, temperature: float = 0.7,
                 max_tokens: int = 2048, use_cache: bool = True,
                 use_fallback: bool = True):
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