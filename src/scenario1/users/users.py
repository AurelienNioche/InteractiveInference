import numpy as np
import gym


class User(gym.Env):

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

    def step(self, action: np.ndarray):

        # Position is the action from the assistant
        position = action

        p_star = position[self.goal]

        prev_mu_hat = self.mu_hat.copy()

        delta = np.zeros(2)

        for coord in range(2):
            self.mu_hat[coord] = self.mu_hat[coord] * (1 - self.alpha) + p_star[coord] * self.alpha

            delta[coord] = - self.beta * (self.mu_hat[coord] - prev_mu_hat[coord])

        self.t += 1

        noise = self.rng.normal(0, self.sigma)
        obs = delta + noise
        reward, done, info = None, None, None
        return obs, reward, done, info

    def reset(self):

        # if not self.debug:
        #     self.goal = np.random.randint(self.n_target)
        # else:
        #     self.goal = 0

        self.t = 1

        self.mu_hat = np.zeros(2)  # Could be whatever

        # We don't consume observation after the reset
        observation = None
        return observation  # reward, done, info can't be included

    def render(self, mode='human'):
        pass

    def close(self):
        pass
