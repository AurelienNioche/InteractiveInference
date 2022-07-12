import gym
import numpy as np


class User(gym.Env):

    def __init__(self, n_targets, parameters=None, debug=False):
        super().__init__()

        self.n_targets = n_targets

        self.parameters = parameters

        self.goal = None  # index of preferred target, assigned during reset()
        self.t = None

        self.debug = debug

    def conditional_probability_action(self, x):

        prm = self.parameters
        if prm is None:
            prm = 8, 0.5

        return 1 / (1 + np.exp(-prm[0]*(- prm[1] + x)))

    def step(self, action: np.ndarray):

        # Position is the action from the assistant
        position = action

        # User action is 1 ("complain") or 0 ("accept")
        # np.tanh(position[self.goal]*self.beta)
        p_complain = self.conditional_probability_action(position[self.goal])

        # print("goal", position[self.goal])
        # print("p complain", p_complain)
        user_action = np.random.random() < p_complain

        self.t += 1

        if self.debug:
            print("x target", position[self.goal])
            print("p complain", p_complain)

        obs = user_action
        reward, done, info = None, None, None
        return obs, reward, done, info

    def reset(self):

        if not self.debug:
            self.goal = np.random.randint(self.n_targets)
        else:
            self.goal = 0

        self.t = 1

        # We don't consume observation after the reset
        observation = None
        return observation  # reward, done, info can't be included

    def render(self, mode='human'):
        pass

    def close(self):
        pass
