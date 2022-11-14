import numpy as np


class User:

    def __init__(self, n_target, goal, alpha, sigma, beta=1.0, seed=123):
        super().__init__()

        self.n_target = n_target

        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma

        self.goal = goal
        self.mu_hat = None
        self.t = None

        self.rng = np.random.default_rng(seed=seed)

    def act(self, positions):

        p_star = positions[self.goal]

        prev_mu_hat = self.mu_hat.copy()

        delta = np.zeros(2)
        for coord in range(2):
            self.mu_hat[coord] = self.mu_hat[coord] * (1 - self.alpha) + p_star[coord] * self.alpha
            delta[coord] = - self.beta * (self.mu_hat[coord] - prev_mu_hat[coord])

        self.t += 1

        noise = self.rng.normal(0, self.sigma)
        user_control = delta + noise
        return user_control

    def p_action(self):

        ...
        p = np.random.random(self.n_target)
        return p

    def reset(self):

        self.t = 1
        self.mu_hat = np.zeros(2)  # Could be whatever

