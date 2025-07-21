#!/usr/bin/env python3
# core/llm_utils.py — ARIASKA LLM Utilities v2.0 SIMPLIFIED  
# This file provides centralized access to GPTManager (GPT-4o-mini only)

import logging
from typing import Optional, Dict, Any, Union
from rich.console import Console

logger = logging.getLogger("ariaska.llm_utils")
console = Console()

# Global GPT manager instance
_gpt_manager = None

def get_gpt_manager():
    """
    Get the global GPTManager instance.
    
    Returns:
        GPTManager: Initialized GPT manager using GPT-4o-mini
    """
    global _gpt_manager
    if _gpt_manager is None:
        try:
            from core.gpt_manager import GPTManager
            _gpt_manager = GPTManager()
            logger.info("✅ GPTManager (GPT-4o-mini) initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize GPTManager: {e}")
            _gpt_manager = None
    return _gpt_manager

def get_llm_router():
    """
    Get the LLM router instance.
    
    Returns:
        LLMRouter: Simplified router for GPT-4o-mini
    """
    try:
        from core.utils.llm_router import get_router
        return get_router()
    except Exception as e:
        logger.error(f"Failed to get LLM router: {e}")
        return None

def gpt_request(
    prompt: str,
    task_type: str = "general",
    agent_id: str = "system",
    max_tokens: int = 500,
    require_json: bool = False
) -> str:
    """
    Convenience function for GPT requests.
    
    Args:
        prompt: The prompt to send
        task_type: Type of task (tactical, analysis, etc.)
        agent_id: ID of the requesting agent
        max_tokens: Maximum tokens to generate
        require_json: Whether response should be JSON
        
    Returns:
        str: The GPT response
    """
    gpt = get_gpt_manager()
    if not gpt:
        return "Error: GPT manager not available"
    
    try:
        return gpt.gpt_request(
            prompt=prompt,
            task_type=task_type,
            agent_id=agent_id,
            max_tokens=max_tokens
        )
    except Exception as e:
        logger.error(f"GPT request failed: {e}")
        return f"Error: {str(e)}"

def generate_tactical_command(
    task_description: str,
    context: Optional[Dict[str, Any]] = None,
    agent_id: str = "system"
) -> str:
    """
    Generate a tactical cybersecurity command.
    
    Args:
        task_description: Description of what needs to be done
        context: Optional context information
        agent_id: ID of the requesting agent
        
    Returns:
        str: Generated command
    """
    context_str = ""
    if context:
        context_str = f" Context: {context}"
    
    prompt = f"Generate a cybersecurity command for: {task_description}{context_str}"
    
    return gpt_request(
        prompt=prompt,
        task_type="tactical",
        agent_id=agent_id,
        max_tokens=100
    )

def analyze_output(
    command: str,
    output: str,
    agent_id: str = "system"
) -> str:
    """
    Analyze command output using GPT.
    
    Args:
        command: The command that was executed
        output: The output from the command
        agent_id: ID of the requesting agent
        
    Returns:
        str: Analysis of the output
    """
    prompt = f"Analyze this cybersecurity command output:\nCommand: {command}\nOutput: {output}\nProvide insights:"
    
    return gpt_request(
        prompt=prompt,
        task_type="analysis",
        agent_id=agent_id,
        max_tokens=300
    )

def get_strategic_advice(
    situation: str,
    objective: str,
    agent_id: str = "system"
) -> str:
    """
    Get strategic advice for cybersecurity scenarios.
    
    Args:
        situation: Current situation description
        objective: What needs to be achieved
        agent_id: ID of the requesting agent
        
    Returns:
        str: Strategic advice
    """
    prompt = f"Situation: {situation}\nObjective: {objective}\nProvide strategic cybersecurity advice:"
    
    return gpt_request(
        prompt=prompt,
        task_type="strategic",
        agent_id=agent_id,
        max_tokens=400
    )

# For backwards compatibility - deprecated functions that previously used LocalLLM
def get_local_llm_manager(model_name: Optional[str] = None):
    """DEPRECATED: Local LLMs removed. Use get_gpt_manager() instead."""
    logger.warning("get_local_llm_manager is deprecated. All LLM requests now use GPT-4o-mini.")
    return get_gpt_manager()

def query_lily_llm(prompt: str, **kwargs) -> str:
    """DEPRECATED: Lily LLM removed. Redirected to GPT-4o-mini."""
    logger.warning("query_lily_llm is deprecated. Using GPT-4o-mini instead.")
    return gpt_request(prompt, task_type="tactical", **kwargs)

def query_seneca_llm(prompt: str, **kwargs) -> str:
    """DEPRECATED: Seneca LLM removed. Redirected to GPT-4o-mini."""
    logger.warning("query_seneca_llm is deprecated. Using GPT-4o-mini instead.")
    return gpt_request(prompt, task_type="tactical", **kwargs)

# Test function
def test_llm_utils():
    """Test the LLM utilities."""
    console.print("[bold cyan]Testing LLM Utils (GPT-4o-mini only)[/bold cyan]")
    
    # Test GPT manager
    gpt = get_gpt_manager()
    if gpt:
        console.print("[green]✓ GPT Manager initialized[/green]")
        
        # Test basic request
        try:
            response = gpt_request("Say hello", task_type="test")
            console.print(f"[green]✓ GPT Response:[/green] {response[:100]}...")
        except Exception as e:
            console.print(f"[red]✗ GPT Request failed: {e}[/red]")
    else:
        console.print("[red]✗ GPT Manager failed to initialize[/red]")
    
    # Test router
    router = get_llm_router()
    if router:
        console.print("[green]✓ LLM Router available[/green]")
    else:
        console.print("[red]✗ LLM Router unavailable[/red]")

if __name__ == "__main__":
    test_llm_utils()
