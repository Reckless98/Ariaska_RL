# core/utils/llm_integration.py — ARIASKA LLM Integration Helper v1.0
# Simplified API for agents to interact with the LLM orchestration system

import os
from typing import Dict, Any, Optional, List, Union
from rich.console import Console

from core.utils.llm_router import LLMRouter, LLMResponse
from core.utils.context_encoder import ContextEncoder

# Singleton pattern to ensure one router instance
_router_instance = None

console = Console()

def get_router() -> LLMRouter:
    """
    Get or create the LLM router singleton instance.
    
    Returns:
        LLMRouter instance
    """
    global _router_instance
    if _router_instance is None:
        try:
            _router_instance = LLMRouter()
        except Exception as e:
            console.print(f"[red]❌ Failed to initialize LLM Router: {e}[/red]")
            raise
    return _router_instance

def request_tactical_command(
    prompt: str, 
    agent_id: str, 
    context: Dict[str, Any] = None
) -> str:
    """
    Request a tactical command from the LLM system.
    
    Args:
        prompt: The command prompt
        agent_id: Agent making the request
        context: Command context
        
    Returns:
        Command string
    """
    # Get the router
    router = get_router()
    
    # Optimize context for token efficiency
    if context:
        optimized_context = context.copy()  # Make a copy to avoid modifying original
    else:
        optimized_context = {}
        
    # Request a command
    response = router.request(
        prompt=prompt,
        role="tactical",
        agent_id=agent_id,
        context=optimized_context,
        require_validation=False  # Simple command doesn't need schema validation
    )
    
    return response.content

def request_recon_command(
    prompt: str,
    agent_id: str,
    targets: List[str],
    context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Request a reconnaissance command from the LLM system.
    
    Args:
        prompt: The recon prompt
        agent_id: Agent making the request
        targets: List of targets
        context: Additional context
        
    Returns:
        Dict containing command and metadata
    """
    # Get the router
    router = get_router()
    
    # Build context with targets
    if context:
        optimized_context = context.copy()
    else:
        optimized_context = {}
        
    optimized_context["targets"] = targets
    
    # Request a structured command
    response = router.request(
        prompt=prompt,
        role="recon",
        agent_id=agent_id,
        context=optimized_context,
        require_validation=True  # Use schema validation
    )
    
    # Check if we have structured data
    if response.parsed:
        return response.parsed
    else:
        # Return basic structure with just the command
        return {
            "command": response.content,
            "targets": targets,
            "confidence": 0.6
        }

def request_exploit_command(
    prompt: str,
    agent_id: str,
    target: str,
    service: str,
    context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Request an exploit command from the LLM system.
    
    Args:
        prompt: The exploit prompt
        agent_id: Agent making the request
        target: Target to exploit
        service: Service to exploit
        context: Additional context
        
    Returns:
        Dict containing command and metadata
    """
    # Get the router
    router = get_router()
    
    # Build context
    if context:
        optimized_context = context.copy() 
    else:
        optimized_context = {}
        
    optimized_context["target"] = target
    optimized_context["service"] = service
    
    # Request a structured command
    response = router.request(
        prompt=prompt,
        role="exploit",
        agent_id=agent_id,
        context=optimized_context,
        require_validation=True  # Use schema validation
    )
    
    # Check if we have structured data
    if response.parsed:
        return response.parsed
    else:
        # Return basic structure with just the command
        return {
            "command": response.content,
            "target": target,
            "service": service,
            "exploit_type": "unknown",
            "confidence": 0.5
        }

def request_strategy(
    prompt: str,
    agent_id: str,
    context: Dict[str, Any]
) -> str:
    """
    Request a high-level strategy from the LLM system.
    
    Args:
        prompt: The strategy prompt
        agent_id: Agent making the request 
        context: Mission context
        
    Returns:
        Strategy string
    """
    # Get the router
    router = get_router()
    
    # For strategy requests, optimize context aggressively to save tokens
    optimized_context = {}
    if context:
        # Use the ContextEncoder for maximum efficiency
        context_str = ContextEncoder.optimize_for_llm_prompt(context, max_chars=800)
        optimized_context["state"] = context_str
    
    # Strategy needs more context, so use a larger model
    response = router.request(
        prompt=prompt,
        role="strategic",
        agent_id=agent_id,
        context=optimized_context,
        # No validation needed for strategy text
    )
    
    return response.content

def analyze_output(
    command: str,
    output: str,
    agent_id: str
) -> Dict[str, Any]:
    """
    Analyze command output using the LLM system.
    
    Args:
        command: The command that was executed
        output: The command's output
        agent_id: Agent making the request
        
    Returns:
        Analysis dictionary
    """
    # Get the router
    router = get_router()
    
    # Build analysis context
    context = {
        "command": command,
        "output": output[:500]  # Limit output size
    }
    
    # Analysis prompt
    prompt = f"Analyze the output of this command: {command}"
    
    response = router.request(
        prompt=prompt,
        role="analysis",
        agent_id=agent_id,
        context=context,
    )
    
    # Return both raw text and structured info if available
    result = {"analysis": response.content}
    if response.parsed:
        result.update(response.parsed)
    return result

def get_token_usage(agent_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Get token usage statistics.
    
    Args:
        agent_id: Optional agent ID to filter by
        
    Returns:
        Token usage statistics
    """
    router = get_router()
    return router.get_stats()

def display_token_usage():
    """Display token usage statistics in the console."""
    router = get_router()
    router.display_stats()

# CLI test
if __name__ == "__main__":
    # Test the integration
    console.print("[bold magenta]🚀 Testing LLM Integration[/bold magenta]")
    
    # Test tactical command
    command = request_tactical_command(
        "Scan the target host for open ports",
        "TestAgent",
        {"targets": ["10.10.10.10"]}
    )
    console.print(f"[green]Tactical Command:[/green] {command}")
    
    # Test recon command
    result = request_recon_command(
        "Find all WordPress installations on the target network",
        "ScoutAgent",
        ["10.10.10.0/24"],
        {"phase": "recon", "discovered_ports": [80, 443, 8080]}
    )
    console.print(f"[blue]Recon Command:[/blue] {result}")
    
    # Test strategy 
    strategy = request_strategy(
        "Develop a strategy for lateral movement",
        "OrionAgent",
        {
            "phase": "post_exploitation",
            "privilege_level": "user",
            "discovered_hosts": ["10.10.10.10", "10.10.10.15", "10.10.10.20"],
            "exploited_hosts": ["10.10.10.10"]
        }
    )
    console.print(f"[yellow]Strategy:[/yellow] {strategy}")
    
    # Display token usage
    display_token_usage()