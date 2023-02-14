from typing import Union

import numpy as np


class Teaching:

    def __init__(
            self,
            initial_forget_rates: np.ndarray,
            repetition_rates: np.ndarray,
            tau: float = 0.9,
            n_item: int = 30,                       # 500
            n_session: int = 6,                     # 6
            n_iter_per_session: int = 100,          # 100
            break_length: Union[float, int] = 10,   # 24*60**2
            time_per_iter: Union[float, int] = 1,  # 4
    ):
        super().__init__()

        self.n_item = n_item
        self.tau = tau
        self.log_tau = np.log(tau)
        self.initial_forget_rates = initial_forget_rates
        self.repetition_rates = repetition_rates
        self.n_session = n_session
        self.n_iter_per_session = n_iter_per_session
        self.time_per_iter = time_per_iter
        self.break_length = break_length

        self.t_max = n_session * n_iter_per_session

        self.current_iter = 0
        self.current_ss = 0
        self.n_pres = np.zeros(n_item)
        self.delta = np.zeros(n_item)

    def reset(self):

        self.current_iter = 0
        self.current_ss = 0
        self.n_pres = np.zeros(self.n_item)
        self.delta = np.zeros(self.n_item)

    def step(self, action):
        done = False
        self.current_iter += 1
        if self.current_iter >= self.n_iter_per_session:
            self.current_iter = 0
            self.current_ss += 1
            time_elapsed_since_last_iter = self.break_length
        else:
            time_elapsed_since_last_iter = self.time_per_iter

        if self.current_ss >= self.n_session:
            done = True

        # increase delta for all items
        self.delta += time_elapsed_since_last_iter
        # ...except for item shown
        self.delta[action] = time_elapsed_since_last_iter
        # increment number of presentation
        self.n_pres[action] += 1

        seen = self.n_pres > 0
        delta = self.delta[seen]
        rep = self.n_pres[seen] - 1.

        fr = self.initial_forget_rates[seen]
        rep_eff = self.repetition_rates[seen]

        forget_rate = fr * (1 - rep_eff) ** rep
        logp_recall = - forget_rate * delta
        reward = np.sum(logp_recall > self.log_tau) / self.n_item

        info = {}
        return None, reward, done, info

    @property
    def t(self):
        return (self.current_ss*self.n_iter_per_session) + self.current_iter
