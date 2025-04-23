#!/bin/bash

# Define a function to replace AgentManager import in agent files
fix_agent_file() {
    local agent_file=$1
    local agent_name=$2

    # Replace the AgentManager import with a passed argument
    sed -i "s/from core.multiagent.agent_manager import AgentManager/# from core.multiagent.agent_manager import AgentManager/g" "$agent_file"
    
    # Modify the constructor to accept agent_manager as a parameter
    sed -i "s/self.agent_manager = AgentManager()/self.agent_manager = agent_manager/g" "$agent_file"

    # Add agent_manager argument to the __init__ function of the agent class
    sed -i "s/def __init__(self)/def __init__(self, agent_manager, /g" "$agent_file"

    echo "Fixed $agent_name agent: $agent_file"
}

# Apply fixes to all agent files
fix_agent_file "core/agents/red_agent.py" "RedAgent"
fix_agent_file "core/agents/blue_agent.py" "BlueAgent"
fix_agent_file "core/agents/scout_agent.py" "ScoutAgent"
fix_agent_file "core/agents/shadow_agent.py" "ShadowAgent"
fix_agent_file "core/agents/orion_agent.py" "OrionAgent"

# Modify the main file to pass the AgentManager to the agents
sed -i "/# Initialize agent manager/,/agent_manager.get_agent('RedAgent')/s/agent_manager.get_agent/agent_manager.get_agent(agent_manager)/g" "main.py"
sed -i "/# Initialize agent manager/,/manager.get_agent('RedAgent')/s/agent_manager/agent_manager=AgentManager()/g" "main.py"

echo "Main file updated with AgentManager injection."

# Add missing imports in core/multiagent/agent_manager.py
sed -i '1s/^/from core.agents.red_agent import RedAgent\nfrom core.agents.blue_agent import BlueAgent\nfrom core.agents.scout_agent import ScoutAgent\nfrom core.agents.shadow_agent import ShadowAgent\nfrom core.agents.orion_agent import OrionAgent\n/' "core/multiagent/agent_manager.py"

echo "Circular imports should now be fixed and AgentManager passed properly."

# Done
echo "Fixes complete. Now you can run the project again."
