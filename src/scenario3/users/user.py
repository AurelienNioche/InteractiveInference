import gym
import torch


class User(gym.Env):

    def __init__(self, goal, parameters=None):
        super().__init__()

        self.n_targets = len(goal)

        if parameters is None:
            parameters = (8.0, 0.5)
        self.parameters = parameters

        self._goal = goal  # Desired positions of targets

        self._action = None
        self.t = None

    @property
    def goal(self):
        return self._goal.numpy()

    @property
    def action(self):
        return self._action.item()

    def conditional_probability_action(self, mean_dist, log=False):

        prm = self.parameters
        x = prm[0]*(- prm[1] + mean_dist)
        if log:
            return torch.nn.functional.logsigmoid(x)
        else:
            return torch.sigmoid(x)

    def step(self, action: torch.Tensor):

        # Targets position is the output from the assistant
        position = action

        mean_dist = torch.abs(position - self._goal).mean()

        # User action is 1 ("complain") or 0 ("accept")
        p_complain = self.conditional_probability_action(mean_dist)
        self._action = torch.rand(1) < p_complain
        self.t += 1

        obs = self._action
        reward, done, info = None, None, None
        return obs, reward, done, info

    def reset(self):

        self._action = None
        self.t = 1

        # We don't consume observation after the reset
        observation = None
        return observation  # reward, done, info can't be included

    def render(self, mode='human'):
        pass

    def close(self):
        pass
