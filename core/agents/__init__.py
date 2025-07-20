"""
Core agents package init - Provides proper import functionality
"""

# Import locally within functions that need them to prevent circular imports
def get_red_agent():
    from .red_agent import RedAgent
    return RedAgent

def get_blue_agent():
    from .blue_agent import BlueAgent
    return BlueAgent

def get_scout_agent():
    from .scout_agent import ScoutAgent
    return ScoutAgent

def get_shadow_agent():
    from .shadow_agent import ShadowAgent
    return ShadowAgent

def get_orion_agent():
    from .orion_agent import OrionAgent
    return OrionAgent

# Export classes for direct access from core.agents
__all__ = ['RedAgent', 'BlueAgent', 'ScoutAgent', 'ShadowAgent', 'OrionAgent']
