import numpy as np
import gym
from gym import spaces


class Assistant(gym.Env):

    def __init__(self, n_targets, t_threshold=50, n_dim=1):
        super().__init__()
        self.n_targets = n_targets
        self.n_dim = n_dim

        self.t_threshold = t_threshold

        self.goal = None  # index of preferred target, assigned during reset()
        self.pos = None
        self.t = None
        self.phase = None
        self.amp = None
        self.freq = None

        self.mu_dist = None

        self.selected = None

    def step(self, action: np.ndarray):

        alpha = 1 / self.t

        old_pos = self.pos.copy()

        self.pos[:] = self.amp * np.sin(self.freq * self.t + self.phase)

        a = action

        dist = np.sum((a - old_pos) ** 2, axis=1)

        self.mu_dist = self.mu_dist * (1 - alpha) + dist * alpha

        if self.t > self.t_threshold:
            self.selected = np.argmin(self.mu_dist)

        if self.selected is not None:
            done = True
        else:
            done = False

        self.t += 1
        observation = self.pos

        reward, info = None, None
        return observation, reward, done, info

    def reset(self):

        self.selected = None

        self.t = 0

        self.phase = np.random.random((self.n_targets, self.n_dim))
        self.amp = np.ones_like(self.phase)
        self.freq = np.ones_like(self.phase)

        self.pos = self.amp * np.sin(self.freq * self.t + self.phase)

        self.mu_dist = 0

        self.t += 1

        observation = self.pos
        return observation  # reward, done, info can't be included

    def render(self, mode='human'):
        pass

    def close(self):
        pass