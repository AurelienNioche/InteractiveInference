import numpy as np
import torch
from scipy import stats


class User:
    DUMMY_VALUE = - 1.0

    def __init__(self, n_target, goal, sigma=5, seed=123):

        self.n_target = n_target
        self.goal = goal
        self.rng = np.random.default_rng(seed=seed)
        self.sigma = sigma

        self.target_prev_pos = None
        self.action = None

    def update_historic_and_compute_delta(self, target_position, target_prev_pos):

        if target_prev_pos is None or np.all(np.isclose(target_prev_pos, self.DUMMY_VALUE)):
            target_prev_pos = target_position.copy()

        delta = target_position - target_prev_pos
        return delta, target_position.copy()

    def act(self, positions, goal=None):

        if goal is None:
            goal = self.goal

        delta, self.target_prev_pos = self.update_historic_and_compute_delta(
            target_position=positions[goal],
            target_prev_pos=self.target_prev_pos)
        noise = self.rng.normal(0, self.sigma, size=2)
        self.action = delta + noise
        return self.action

    def reset(self):
        pass


class UserModel(User):

    def __init__(self, n_target, sigma):
        super().__init__(n_target=n_target, sigma=sigma, goal=None)
        self.target_prev_pos = torch.full((n_target, 2), self.DUMMY_VALUE)

    def update_historic_and_compute_delta(self, target_position, target_prev_pos):

        if target_prev_pos is None or torch.all(torch.isclose(target_prev_pos, torch.full_like(target_prev_pos, self.DUMMY_VALUE))):
            target_prev_pos = target_position # .clone()

        delta = target_position - target_prev_pos
        return delta, target_position.clone()

    def logp_action(self, positions, action):
        delta = torch.zeros_like(self.target_prev_pos)
        logp = torch.zeros(self.n_target)
        for target in range(self.n_target):
            delta[target], self.target_prev_pos[target] = self.update_historic_and_compute_delta(
                target_position=positions[target],
                target_prev_pos=self.target_prev_pos[target])
            for coord in range(2):
                logp_coord = torch.distributions.Normal(delta[target, coord], self.sigma).log_prob(action[coord])
                logp[target] += logp_coord
        return logp

    def sim_act(self, goal, positions, prev_positions):

        delta, _ = self.update_historic_and_compute_delta(
            target_position=positions[goal],
            target_prev_pos=prev_positions[goal])
        noise = torch.randn(2) * self.sigma
        action = delta + noise
        return action
