import numpy as np
import torch


class Fish:

    def __init__(self, environment, goal, seed, sigma, init_position, movement_amplitude):

        self.environment = environment
        self.goal = goal
        self.rng = np.random.default_rng(seed=seed)
        self.sigma = sigma

        self.movement_amplitude = movement_amplitude

        self.action = init_position

    def act(self, *args, **kwargs):

        noise = self.rng.normal(0, self.sigma, size=2)
        self.action = self.mu(*args, **kwargs) + noise
        return self.action

    def mu(self, positions, goal=None, own_position=None):

        if goal is None:
            assert self.goal is not None
            goal = self.goal

        if own_position is None:
            assert self.action is not None
            own_position = self.action

        own_x, own_y = own_position
        if self.environment.fish_in(target=goal, fish_position=own_position, target_positions=positions):
            mu = np.zeros(2)
        else:
            x_center, y_center = self.environment.target_center(target=goal)
            opp = y_center - own_y
            adj = x_center - own_x
            target_angle = np.degrees(np.arctan2(opp, adj))

            x_prime = 1.0
            if 90 < target_angle <= 270:
                x_prime *= -1

            y_prime = np.tan(np.radians(target_angle)) * x_prime

            norm = self.movement_amplitude / torch.sqrt(y_prime ** 2 + x_prime ** 2)
            mu = np.array([x_prime, y_prime]) * norm
        return mu

    def reset(self):
        pass


class FishModel(Fish):

    def __init__(self, environment, n_target, movement_amplitude, sigma, seed):

        self.n_target = n_target
        super().__init__(environment=environment, seed=seed, sigma=sigma, movement_amplitude=movement_amplitude,
                         init_position=None, goal=None)

    def logp_action(self, positions, action, own_position):

        logp = np.zeros(self.n_target)
        for goal in range(self.n_target):
            mu = self.mu(positions=positions, goal=goal, own_position=own_position)
            for coord in range(2):
                a_coord = action[coord]
                border = self.environment.size(0)
                dist = torch.distributions.Normal(mu[coord], self.sigma)
                if a_coord == 0:
                    logp_coord = (1 - dist.cdf(a_coord)).log()
                elif a_coord == border:
                    logp_coord = dist.cdf(a_coord).log()
                else:
                    logp_coord = dist.log_prob(a_coord)
                logp[goal] += logp_coord
        return logp
