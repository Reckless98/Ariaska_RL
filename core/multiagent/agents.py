# core/multiagent/agents.py — ARIASKA Agent Factory v1.0
# 🧩 Centralized Agent Creation | 🔄 Import Resolution | 🧠 Agent Configuration

"""
Centralized module for agent creation and management.
Helps resolve circular imports by dynamically importing agents only when needed.
"""

from rich.console import Console
from typing import Dict, Any, List

console = Console()

def get_all_agents(agent_manager=None, memory_router=None, memory_manager=None, verbosity="standard"):
    """
    Create all agents with appropriate dependencies.
    This function centralizes agent creation to avoid circular imports.
    
    Args:
        agent_manager: Optional AgentManager instance for agent coordination
        memory_router: Optional MemoryRouter instance for memory sharing
        memory_manager: Optional MemoryManager instance for memory persistence
        verbosity: Verbosity level for agents
        
    Returns:
        dict: Dictionary with agent instances keyed by agent_id
    """
    from core.agents.red_agent import RedAgent
    from core.agents.blue_agent import BlueAgent
    from core.agents.scout_agent import ScoutAgent
    from core.agents.shadow_agent import ShadowAgent
    from core.agents.orion_agent import OrionAgent
    from core.utils.memory_manager import MemoryManager

    # Create memory managers for each agent if not provided
    if not memory_manager:
        red_memory_manager = MemoryManager(agent_id="red_agent")
        blue_memory_manager = MemoryManager(agent_id="blue_agent")
        scout_memory_manager = MemoryManager(agent_id="scout_agent")
        shadow_memory_manager = MemoryManager(agent_id="shadow_agent")
        orion_memory_manager = MemoryManager(agent_id="orion_agent")
    else:
        # Use the provided memory manager
        red_memory_manager = blue_memory_manager = scout_memory_manager = shadow_memory_manager = orion_memory_manager = memory_manager

    # Create agents
    red_agent = RedAgent(
        agent_id="RedAgent",
        role="CyberOffense",
        agent_manager=agent_manager,
        memory_router=memory_router,
        memory_manager=red_memory_manager,
        verbosity=verbosity
    )
    
    blue_agent = BlueAgent(
        agent_id="BlueAgent",
        role="CyberDefense",
        agent_manager=agent_manager,
        memory_router=memory_router,
        memory_manager=blue_memory_manager,
        verbosity=verbosity
    )
    
    scout_agent = ScoutAgent(
        agent_id="ScoutAgent",
        role="PhaseNavigator",
        agent_manager=agent_manager,
        memory_router=memory_router,
        memory_manager=scout_memory_manager,
        verbosity=verbosity
    )
    
    shadow_agent = ShadowAgent(
        agent_manager=agent_manager,
        memory_router=memory_router,
        verbosity=verbosity
    )
    
    orion_agent = OrionAgent(
        agent_id="OrionAgent",
        role="StrategicOverseer",
        agent_manager=agent_manager,
        memory_router=memory_router,
        memory_manager=orion_memory_manager,
        verbosity=verbosity
    )
    
    return {
        "RedAgent": red_agent,
        "BlueAgent": blue_agent,
        "ScoutAgent": scout_agent,
        "ShadowAgent": shadow_agent,
        "OrionAgent": orion_agent
    }

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
