import numpy as np
import torch
import math


class Fish:

    def __init__(self, goal, seed, sigma, position, movement_amplitude):

        self.goal = goal
        self.rng = np.random.default_rng(seed=seed)
        self.sigma = sigma
        self.position = position

        self.movement_amplitude = movement_amplitude

        self.action = None
        self.position = None

    def act(self, positions, goal=None):

        if goal is None:
            goal = self.goal

        x0, y0, x1, y1 = positions[goal]
        own_x, own_y = self.position
        if x0 <= own_x <= x1 and y0 <= own_y <= y1:
            mu = np.zeros(2)
        else:
            x_center, y_center = (x0 + x1) / 2, (y0 + y1) / 2
            opp = y_center - own_y
            adj = x_center - own_x
            target_angle = np.degrees(np.arctan2(opp, adj))

            x_prime = 1.0
            if 90 < target_angle <= 270:
                x_prime *= -1

            y_prime = np.tan(np.radians(target_angle)) * x_prime

            norm = self.movement_amplitude / torch.sqrt(y_prime ** 2 + x_prime ** 2)
            mu = np.array([x_prime, y_prime]) * norm

        noise = self.rng.normal(0, self.sigma, size=2)
        self.action = mu + noise
        return self.action

    def reset(self):
        pass


class FishModel(Fish):

    def __init__(self, n_target, sigma):
        super().__init__(n_target=n_target, sigma=sigma, goal=None)
        self.target_prev_pos = torch.full((n_target, 2), self.DUMMY_VALUE)

    def update_historic_and_compute_delta(self, target_position, target_prev_pos):

        if target_prev_pos is None or torch.all(torch.isclose(target_prev_pos, torch.full_like(target_prev_pos, self.DUMMY_VALUE))):
            target_prev_pos = target_position  # .clone()

        delta = target_position - target_prev_pos
        return delta, target_position.clone()

    def logp_action(self, positions, action, prev_positions=None, update=False):
        if prev_positions is None:
            prev_positions = self.target_prev_pos

        delta = torch.zeros_like(prev_positions)
        logp = torch.zeros(self.n_target)
        for target in range(self.n_target):
            delta[target], new_pos_target = self.update_historic_and_compute_delta(
                target_position=positions[target],
                target_prev_pos=prev_positions[target])

            if update:
                self.target_prev_pos[target] = new_pos_target

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
