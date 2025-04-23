# core/multiagent/agents.py — ARIASKA Agent Registry v1.0

from core.agents.red_agent import RedAgent
from core.agents.blue_agent import BlueAgent
from core.agents.scout_agent import ScoutAgent
from core.agents.shadow_agent import ShadowAgent
from core.agents.orion_agent import OrionAgent
from core.utils.memory_manager import MemoryManager

def get_all_agents(agent_manager=None):
    """
    Centralized agent factory function that returns all agent instances.
    """
    # Pass agent_manager to all agents to avoid recursion
    red_agent = RedAgent(agent_id="RedAgent", role="Offense", agent_manager=agent_manager, memory_manager=MemoryManager("red_agent"))
    blue_agent = BlueAgent(agent_id="BlueAgent", role="Defense", agent_manager=agent_manager, memory_manager=MemoryManager("blue_agent"))
    scout_agent = ScoutAgent(agent_id="ScoutAgent", memory_manager=MemoryManager("scout_agent"), agent_manager=agent_manager, memory_router=None)
    shadow_agent = ShadowAgent(agent_manager=agent_manager, memory_router=None, verbosity="standard")
    orion_agent = OrionAgent(agent_manager=agent_manager, memory_router=None)

    return {
        "RedAgent": red_agent,
        "BlueAgent": blue_agent,
        "ScoutAgent": scout_agent,
        "ShadowAgent": shadow_agent,
        "OrionAgent": orion_agent,
    }
