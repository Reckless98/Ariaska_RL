import random

class ReplayBuffer:
    def __init__(self, capacity=2000, alpha=0.6):
        self.buffer = []
        self.capacity = capacity
        self.alpha = alpha  # Prioritization factor

    def add(self, experience):
        priority = abs(experience.get('reward', 0)) + 0.01
        self.buffer.append((priority, experience))
        self.buffer = sorted(self.buffer, key=lambda x: x[0], reverse=True)
        if len(self.buffer) > self.capacity:
            self.buffer.pop()

    def sample(self, batch_size):
        return [exp for _, exp in self.buffer[:batch_size]]

    def prune_redundancy(self, redundancy_detector):
        # Remove redundant experiences using provided detector
        commands = [exp['command'] for _, exp in self.buffer if 'command' in exp]
        redundant_idxs = redundancy_detector(commands)
        self.buffer = [item for idx, item in enumerate(self.buffer) if idx not in redundant_idxs]
