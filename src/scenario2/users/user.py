import gym
import numpy as np


class User(gym.Env):

    def __init__(self, n_targets, goal, parameters=None):
        super().__init__()

        assert goal < n_targets, "Revise your logic!"

        self.n_targets = n_targets

        if parameters is None:
            parameters = (8.0, 0.5)
        self.parameters = parameters

        self.goal = goal  # index of preferred target

        self.t = None

    def conditional_probability_action(self, x):

        prm = self.parameters
        return 1 / (1 + np.exp(-prm[0]*(- prm[1] + x)))

    def step(self, action: np.ndarray):

        # Targets position is the output from the assistant
        position = action

        # User action is 1 ("complain") or 0 ("accept")
        p_complain = self.conditional_probability_action(position[self.goal])
        user_action = np.random.random() < p_complain

        self.t += 1

        obs = user_action
        reward, done, info = None, None, None
        return obs, reward, done, info

    def reset(self):

        self.t = 1

        # We don't consume observation after the reset
        observation = None
        return observation  # reward, done, info can't be included

    def render(self, mode='human'):
        pass

    def close(self):
        pass
