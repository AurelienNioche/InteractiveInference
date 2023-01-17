from typing import Union
import numpy as np
from scipy import stats


class Fish:

    def __init__(self,
                 goal: Union[None, int] = 0,
                 sigma=1.,
                 jump_size=3):

        self.goal = goal
        self.sigma = sigma

        self.jump_size = jump_size

    def act(self, fish_position, target_positions, goal, screen_size):

        fish_noise = np.random.normal(size=2) * self.sigma
        fish_mu = Fish.compute_mu(
            fish_position=fish_position,
            target_positions=target_positions,
            goal=goal,
            jump_size=self.jump_size)
        unbounded_fish_jump = fish_mu + fish_noise

        new_pos = Fish.update_fish_position(
            fish_position=fish_position,
            fish_jump=unbounded_fish_jump,
            screen_size=screen_size)

        fish_jump = new_pos - fish_position
        return fish_jump

    @staticmethod
    def update_fish_position(fish_position, fish_jump, screen_size):
        new_pos = fish_position + fish_jump
        for coord in range(2):
            new_pos[coord] = np.clip(new_pos[coord], a_min=0., a_max=screen_size[coord])
        return new_pos

    @staticmethod
    def compute_mu(target_positions, goal, fish_position, jump_size):

        if Fish.fish_is_in(target_positions=target_positions, fish_position=fish_position, goal=goal):
            mu = np.zeros(2)
        else:
            mu = Fish.compute_aim(
                target_positions=target_positions,
                goal=goal,
                fish_position=fish_position,
                jump_size=jump_size)

        return mu

    @staticmethod
    def fish_is_in(fish_position, target_positions, goal):
        x, first_width, second_width, screen_height = target_positions[goal]
        fish_x = fish_position[0]
        return x <= fish_x <= x + first_width or (second_width > 0. and 0. <= fish_x <= second_width)

    @staticmethod
    def compute_aim(target_positions, goal, fish_position, jump_size):

        def compare_with(position):
            first_diff = (x - x_fish) ** 2
            second_diff = (position - x_fish) ** 2
            return np.where(first_diff < second_diff, x, position)

        x_fish = fish_position[0]
        x = target_positions[goal, 0]
        first_width = target_positions[goal, 1]
        second_width = target_positions[goal, 2]
        if second_width > 0:
            x_center = compare_with(second_width)
        else:
            x_center = compare_with(x + first_width)

        if x_center > x_fish:
            x_mvt = jump_size
        else:
            x_mvt = - jump_size

        aim = np.array([x_mvt, 0.])
        return aim


class FishModel(Fish):

    def __init__(self, n_target, jump_size, sigma):

        self.n_target = n_target
        super().__init__(sigma=sigma,
                         jump_size=jump_size,
                         goal=None)

    def logp_action(self, target_positions, fish_jump, fish_initial_position, screen_size):

        jump_size = self.jump_size
        sigma = self.sigma
        n_target = target_positions.shape[0]
        logp = np.zeros(n_target)
        for goal in range(n_target):
            mu = Fish.compute_mu(
                target_positions=target_positions,
                goal=goal,
                fish_position=fish_initial_position,
                jump_size=jump_size)
            final_position = fish_initial_position + fish_jump
            mu_abs = fish_initial_position+mu
            logp_goal = 0
            for coord in range(2):
                mu_abs_coord = mu_abs[coord]
                final_position_coord = final_position[coord]
                border = screen_size[coord]
                dist = stats.norm(mu_abs_coord, sigma)
                if 0 < final_position_coord < border:
                    logp_goal += np.log(dist.pdf(final_position_coord))
                elif final_position_coord <= 0:
                    logp_goal += np.log(1 - dist.cdf(final_position_coord))
                elif final_position_coord >= border:
                    logp_goal += np.log(dist.cdf(final_position_coord))
                else:
                    raise ValueError
            logp[goal] = logp_goal
        return logp
