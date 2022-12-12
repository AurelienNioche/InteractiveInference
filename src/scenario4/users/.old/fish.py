import torch
from typing import Union


class Fish:

    def __init__(self, environment, goal: Union[None, int] = 0,
                 sigma=1.,
                 movement_amplitude=3):

        self.env = environment
        self.goal = goal
        self.sigma = sigma

        self.movement_amplitude = movement_amplitude
        self.action = None

    @property
    def position(self):
        return self.env.fish_position

    def act(self, fish_position, target_positions, goal):

        noise = torch.randn(2) * self.sigma
        mvt = self.mu(fish_position=fish_position, target_positions=target_positions, goal=goal) \
            + noise
        # if target_positions is not None:
        #     print("target positions here", target_positions.requires_grad)
        #     print("mu", mvt.requires_grad)

        new_pos = fish_position + mvt
        for coord in range(2):
            if new_pos[coord] < 0:
                new_pos[coord] = 0
            elif new_pos[coord] > self.env.size(coord):
                new_pos[coord] = self.env.size(coord)
            # new_pos[coord] = torch.clip(new_pos[coord].clone(), min=0, max=self.env.size(coord))

        self.action = new_pos - fish_position
        return self.action

    def mu(self, target_positions, goal, fish_position):

        # if target_positions is not None:
        #     print("target_positions mu", target_positions.requires_grad)
        own_x, own_y = fish_position
        if self.env.fish_is_in(target=goal, fish_position=fish_position,
                               target_positions=target_positions):
            mu = torch.zeros(2, requires_grad=True)
            # if target_positions is not None:
            #     print("mu fish in mu", target_positions.requires_grad)
        else:
            x_center, y_center = self.env.fish_aim(target=goal,
                                                   fish_position=fish_position,
                                                   target_positions=target_positions)
            opp = y_center - own_y
            adj = x_center - own_x
            target_angle = torch.rad2deg(torch.arctan2(opp, adj))

            x_prime = torch.ones(1)
            if 90 < target_angle <= 270:
                x_prime *= -1

            y_prime = torch.tan(torch.deg2rad(target_angle)) * x_prime

            norm = self.movement_amplitude / torch.sqrt(y_prime ** 2 + x_prime ** 2)
            mu = torch.concatenate([x_prime, y_prime]) * norm
        return mu

    def reset(self):
        pass


class FishModel(Fish):

    def __init__(self, environment, movement_amplitude, sigma):

        self.n_target = environment.n_target
        super().__init__(environment=environment, sigma=sigma,
                         movement_amplitude=movement_amplitude,
                         goal=None)

    def logp_action(self, target_positions, fish_jump, fish_initial_position):

        logp = torch.zeros(self.n_target)
        for goal in range(self.n_target):
            mu = self.mu(target_positions=target_positions, goal=goal,
                         fish_position=fish_initial_position)

            for coord in range(2):
                a_coord = fish_jump[coord]
                border = self.env.size(0)
                dist = torch.distributions.Normal(mu[coord], self.sigma)
                if a_coord == 0:
                    logp_coord = torch.log((1 - dist.cdf(a_coord)))
                elif a_coord == border:
                    logp_coord = torch.log(dist.cdf(a_coord))
                else:
                    logp_coord = dist.log_prob(a_coord)

                print("logp_coord", logp_coord.requires_grad)
                logp[goal] += logp_coord
        return logp
