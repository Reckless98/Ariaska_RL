import gym
from gym import spaces
import numpy as np

class PenTestEnv(gym.Env):
    def __init__(self):
        super(PenTestEnv, self).__init__()
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        self.state = self.reset()

    def reset(self):
        self.state = np.zeros(10)
        return self.state

    def step(self, action):
        reward = self.evaluate_action(action)
        done = self.check_done()
        return self.state, reward, done, {}

    def evaluate_action(self, action):
        return np.random.choice([10, 20, 50])

    def check_done(self):
        return np.random.choice([False, True], p=[0.9, 0.1])
