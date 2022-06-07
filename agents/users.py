import numpy as np
import gym


class User(gym.Env):

    def __init__(self, n_targets, beta=1.0, debug=False):
        super().__init__()

        self.n_targets = n_targets

        self.beta = beta

        self.goal = None  # index of preferred target, assigned during reset()
        self.mu_hat = None
        self.t = None

        self.debug = debug

    def step(self, action: np.ndarray):

        alpha = 1 / self.t

        a = action[self.goal]

        self.mu_hat = self.mu_hat * (1 - alpha) + a * alpha

        user_action = self.beta * (self.mu_hat - a)

        self.t += 1

        obs = user_action
        reward, done, info = None, None, None
        return obs, reward, done, info

    def reset(self):

        if not self.debug:
            self.goal = np.random.randint(self.n_targets)
        else:
            self.goal = 0

        self.t = 1

        self.mu_hat = 0  # Could be whathever

        # We don't consume observation after the reset
        observation = None
        return observation  # reward, done, info can't be included

    def render(self, mode='human'):
        pass

    def close(self):
        pass
