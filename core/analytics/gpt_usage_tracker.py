class GPTUsageTracker:
    """
    Tracks GPT token usage per agent for monitoring and optimization.
    """
    def __init__(self):
        self.usage = {}

    def log(self, agent_id, tokens):
        """
        Log token usage for a given agent.
        Args:
            agent_id (str): The agent identifier.
            tokens (int): Number of tokens used.
        """
        self.usage.setdefault(agent_id, 0)
        self.usage[agent_id] += tokens

    def get_usage(self, agent_id):
        """
        Get total token usage for an agent.
        Args:
            agent_id (str): The agent identifier.
        Returns:
            int: Total tokens used by the agent.
        """
        return self.usage.get(agent_id, 0)
