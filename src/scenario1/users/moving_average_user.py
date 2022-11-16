import numpy as np
from scipy import stats


class User:

    DUMMY_VALUE = - 1.0

    def __init__(self, n_target, goal, alpha, beta, sigma, seed=123):

        self.n_target = n_target

        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma

        self.goal = goal
        self.moving_average = None

        self.rng = np.random.default_rng(seed=seed)

        self.action = None

    def update_moving_average_and_compute_delta(self, target_moving_average, target_position, alpha, beta):

        delta = np.zeros(2)

        if target_moving_average is None or np.all(np.isclose(target_moving_average, self.DUMMY_VALUE)):
            target_moving_average = target_position.copy()
            return delta, target_moving_average

        prev_moving_avg = target_moving_average.copy()

        for coord in range(2):
                target_moving_average[coord] = \
                    target_moving_average[coord] * (1 - alpha) \
                    + target_position[coord] * alpha
                delta[coord] = - beta * (target_moving_average[coord] - prev_moving_avg[coord])
        return delta, target_moving_average

    def act(self, positions):

        delta, self.moving_average = self.update_moving_average_and_compute_delta(
            target_position=positions[self.goal],
            target_moving_average=self.moving_average,
            alpha=self.alpha,
            beta=self.beta)
        noise = self.rng.normal(0, self.sigma)
        self.action = delta + noise
        return self.action

    def reset(self):
        pass


class UserModel(User):

    def __init__(self, n_target, alpha, beta, sigma):
        super().__init__(n_target=n_target, alpha=alpha, beta=beta, sigma=sigma, goal=None)
        self.moving_average = np.full((n_target, 2), self.DUMMY_VALUE)

    def p_action(self, positions, action):

        delta = np.zeros_like(self.moving_average)
        p = np.zeros(self.n_target)
        for target in range(self.n_target):
            delta[target], self.moving_average[target] = self.update_moving_average_and_compute_delta(
                target_position=positions[target],
                target_moving_average=self.moving_average[target],
                alpha=self.alpha,
                beta=self.beta)
            p[target] = np.sum([np.log(stats.norm(delta[target, coord], self.sigma).pdf(action[coord]))
                                for coord in range(2)])
        return p
