class GPTUsageTracker:
    def __init__(self):
        self.usage = {}

    def log(self, agent_id, tokens):
        self.usage.setdefault(agent_id, 0)
        self.usage[agent_id] += tokens

    def get_usage(self, agent_id):
        return self.usage.get(agent_id, 0)
