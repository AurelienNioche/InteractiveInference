import numpy as np


class Random:
    def __init__(self, env):
        self.env = env

    def act(self, obs):
        return np.random.choice(self.env.n_item)
