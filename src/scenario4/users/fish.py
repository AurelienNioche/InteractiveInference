import numpy as np
import scipy
from typing import Union


class Fish:

    def __init__(self, environment, goal: Union[None, int] = 0, sigma=1.,
                 movement_amplitude=3,
                 seed=123):

        self.env = environment
        self.goal = goal
        self.rng = np.random.default_rng(seed=seed)
        self.sigma = sigma

        self.movement_amplitude = movement_amplitude
        self.action = None

    @property
    def position(self):
        return self.env.fish.position

    def act(self, own_position=None, *args, **kwargs):

        if own_position is None:
            own_position = self.position

        noise = self.rng.normal(0, self.sigma, size=2)
        mvt = self.mu(own_position=own_position, *args, **kwargs) + noise

        new_pos = own_position + mvt
        for coord in range(2):
            new_pos[coord] = np.clip(new_pos[coord], a_min=0, a_max=self.env.size(coord))

        self.action = new_pos - own_position
        return self.action

    def mu(self, target_positions=None, goal=None, own_position=None):

        if goal is None:
            assert self.goal is not None
            goal = self.goal

        if own_position is None:
            own_position = self.position

        own_x, own_y = own_position
        if self.env.fish_is_in(target=goal, fish_position=own_position, target_positions=target_positions):
            mu = np.zeros(2)
        else:
            x_center, y_center = self.env.fish_aim(target=goal, fish_position=own_position)
            opp = y_center - own_y
            adj = x_center - own_x
            target_angle = np.degrees(np.arctan2(opp, adj))

            x_prime = 1.0
            if 90 < target_angle <= 270:
                x_prime *= -1

            y_prime = np.tan(np.radians(target_angle)) * x_prime

            norm = self.movement_amplitude / np.sqrt(y_prime ** 2 + x_prime ** 2)
            mu = np.array([x_prime, y_prime]) * norm
        return mu

    def reset(self):
        pass


class FishModel(Fish):

    def __init__(self, environment, movement_amplitude, sigma, seed=12345):

        self.n_target = environment.n_target
        super().__init__(environment=environment, seed=seed, sigma=sigma, movement_amplitude=movement_amplitude,
                         goal=None)

    def logp_action(self, target_positions, action, own_position):

        logp = np.zeros(self.n_target)
        for goal in range(self.n_target):
            mu = self.mu(target_positions=target_positions, goal=goal, own_position=own_position)
            for coord in range(2):
                a_coord = action[coord]
                border = self.env.size(0)
                dist = scipy.stats.norm(mu[coord], self.sigma)
                if a_coord == 0:
                    logp_coord = np.log((1 - dist.cdf(a_coord)))
                elif a_coord == border:
                    logp_coord = np.log(dist.cdf(a_coord))
                else:
                    logp_coord = np.log(dist.pdf(a_coord))
                logp[goal] += logp_coord
        return logp
