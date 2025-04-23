class AgentInterface:
    def act(self, state):
        raise NotImplementedError

    def learn(self):
        raise NotImplementedError

    def sync_memory(self):
        raise NotImplementedError
