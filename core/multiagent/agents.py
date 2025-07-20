# core/multiagent/agents.py — ARIASKA Multi-Agent Factory v1.0
# Factory method for creating all agents in the correct order to avoid circular imports

from rich.console import Console
from typing import Dict, Any, List

console = Console()

def get_all_agents(agent_manager=None, memory_router=None):
    """
    Create all agents in the correct order to avoid circular imports.
    
    Args:
        agent_manager: Optional agent manager instance
        memory_router: Optional memory router instance
        
    Returns:
        Dict of agent instances keyed by agent_id
    """
    agents = {}
    
    # Create RedAgent
    try:
        from core.agents.red_agent import RedAgent
        agents["RedAgent"] = RedAgent(agent_manager=agent_manager, memory_router=memory_router)
        console.print("[green]✓ RedAgent created successfully[/green]")
    except Exception as e:
        console.print(f"[red]❌ Error creating RedAgent: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        agents["RedAgent"] = None
    
    # Create BlueAgent
    try:
        from core.agents.blue_agent import BlueAgent
        agents["BlueAgent"] = BlueAgent(agent_manager=agent_manager, memory_router=memory_router)
        console.print("[green]✓ BlueAgent created successfully[/green]")
    except Exception as e:
        console.print(f"[red]❌ Error creating BlueAgent: {e}[/red]")
        agents["BlueAgent"] = None
    
    # Create ScoutAgent
    try:
        from core.agents.scout_agent import ScoutAgent
        agents["ScoutAgent"] = ScoutAgent(agent_manager=agent_manager, memory_router=memory_router)
        console.print("[green]✓ ScoutAgent created successfully[/green]")
    except Exception as e:
        console.print(f"[red]❌ Error creating ScoutAgent: {e}[/red]")
        agents["ScoutAgent"] = None
    
    # Create ShadowAgent
    try:
        from core.agents.shadow_agent import ShadowAgent
        agents["ShadowAgent"] = ShadowAgent(agent_manager=agent_manager, memory_router=memory_router)
        console.print("[green]✓ ShadowAgent created successfully[/green]")
    except Exception as e:
        console.print(f"[red]❌ Error creating ShadowAgent: {e}[/red]")
        agents["ShadowAgent"] = None
    
    # Create OrionAgent (must be last)
    try:
        from core.agents.orion_agent import OrionAgent
        agents["OrionAgent"] = OrionAgent(agent_manager=agent_manager, memory_router=memory_router)
        console.print("[green]✓ OrionAgent created successfully[/green]")
    except Exception as e:
        console.print(f"[red]❌ Error creating OrionAgent: {e}[/red]")
        agents["OrionAgent"] = None
    
    return agents

def create_agent(agent_type, agent_manager=None, memory_router=None, memory_manager=None, verbosity="standard", **kwargs):
    """
    Create a single agent of the specified type.
    
    Args:
        agent_type: Type of agent to create ("RedAgent", "BlueAgent", etc.)
        agent_manager: Optional AgentManager instance for agent coordination
        memory_router: Optional MemoryRouter instance for memory sharing
        memory_manager: Optional MemoryManager instance for memory persistence
        verbosity: Verbosity level for the agent
        **kwargs: Additional parameters to pass to the agent constructor
        
    Returns:
        Agent instance
    """
    agent_classes = {
        "RedAgent": "core.agents.red_agent.RedAgent",
        "BlueAgent": "core.agents.blue_agent.BlueAgent",
        "ScoutAgent": "core.agents.scout_agent.ScoutAgent",
        "ShadowAgent": "core.agents.shadow_agent.ShadowAgent",
        "OrionAgent": "core.agents.orion_agent.OrionAgent"
    }
    
    if agent_type not in agent_classes:
        console.print(f"[red]❌ Unknown agent type: {agent_type}[/red]")
        return None
        
    # Import the agent class dynamically
    import importlib
    module_path, class_name = agent_classes[agent_type].rsplit('.', 1)
    module = importlib.import_module(module_path)
    agent_class = getattr(module, class_name)
    
    # Create memory manager if needed
    if not memory_manager:
        from core.utils.memory_manager import MemoryManager
        memory_manager = MemoryManager(agent_id=agent_type.lower())
    
    # Initialize the agent
    agent = agent_class(
        agent_id=agent_type,
        agent_manager=agent_manager,
        memory_router=memory_router,
        memory_manager=memory_manager,
        verbosity=verbosity,
        **kwargs
    )
    
    return agent

def configure_agent_strategy(agent, strategy):
    """
    Configure agent parameters based on a named strategy.
    
    Args:
        agent: Agent to configure
        strategy: Strategy name ('aggressive', 'defensive', 'balanced', etc.)
    """
    if not agent:
        return
        
    strategies = {
        "aggressive": {
            "epsilon": 0.8,
            "entropy_beta": 0.02,
            "risk_tolerance": 0.7
        },
        "defensive": {
            "epsilon": 0.3,
            "entropy_beta": 0.005,
            "risk_tolerance": 0.3
        },
        "balanced": {
            "epsilon": 0.5,
            "entropy_beta": 0.01,
            "risk_tolerance": 0.5
        },
        "exploratory": {
            "epsilon": 0.9,
            "entropy_beta": 0.03,
            "risk_tolerance": 0.6
        },
        "conservative": {
            "epsilon": 0.2,
            "entropy_beta": 0.003,
            "risk_tolerance": 0.2
        }
    }
    
    if strategy not in strategies:
        console.print(f"[yellow]⚠ Unknown strategy: {strategy}. Using balanced.[/yellow]")
        strategy = "balanced"
        
    params = strategies[strategy]
    
    # Apply parameters if agent has the attributes
    for param, value in params.items():
        if hasattr(agent, param):
            setattr(agent, param, value)
            
    console.print(f"[green]✓ Applied {strategy} strategy to {agent.agent_id}[/green]")
