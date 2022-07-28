import gym
import numpy as np
import torch


class User(gym.Env):

    def __init__(self, n_targets, goal, parameters=None):
        super().__init__()

        assert goal < n_targets, "Revise your logic!"

        self.n_targets = n_targets

        if parameters is None:
            parameters = (8.0, 0.5)
        self.parameters = parameters

        self.goal = goal  # index of preferred target

        self.a = None
        self.t = None

    @property
    def action(self):
        return self.a.item()

    def _logit_p(self, x):
        prm = self.parameters
        return prm[0]*(- prm[1] + x)

    def complain_prob(self, x):
        return torch.sigmoid(self._logit_p(x))

    def complain_log_prob(self, x):
        return torch.nn.functional.logsigmoid(self._logit_p(x))

    def step(self, action: np.ndarray):

        # Targets position is the output from the assistant
        position = action

        # User action is 1 ("complain") or 0 ("accept")
        p_complain = self.complain_prob(position[self.goal])
        self.a = (torch.rand(1) < p_complain).long()

        self.t += 1

        obs = self.a
        reward, done, info = None, None, None
        return obs, reward, done, info

    def reset(self):

        self.a = None
        self.t = 1

        # We don't consume observation after the reset
        observation = None
        return observation  # reward, done, info can't be included

    def render(self, mode='human'):
        pass

    def close(self):
        pass
