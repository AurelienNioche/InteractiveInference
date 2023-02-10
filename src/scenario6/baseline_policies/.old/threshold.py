import numpy as np


class Threshold:
    def __init__(self, env):
        self.env = env

    @staticmethod
    def extract_p_rec(env):

        # If environment is ContinuousTeaching
        view = env.state[:, 1] > 0
        delta = env.state[view, 0]  # only consider already seen items
        rep = env.state[view, 1] - 1.  # only consider already seen items

        forget_rate = env.initial_forget_rates[view] * \
                      (1 - env.initial_repetition_rates[view]) ** rep
        p_recs = np.zeros(shape=(env.state.shape[0]))
        p_recs[view] = np.exp(
                -forget_rate *
                (delta + env.time_per_iter)
            )
        return p_recs

    def act(self, obs):

        p_rec = self.extract_p_rec(self.env)

        view_under_thr = (0 < p_rec) * (p_rec <= self.env.tau)
        if np.count_nonzero(view_under_thr) > 0:
            items = np.arange(self.env.n_item)
            selection = items[view_under_thr]
            action = selection[np.argmin(p_rec[view_under_thr])]
        else:
            n_seen = np.count_nonzero(p_rec)
            max_item = self.env.n_item - 1
            action = np.min((n_seen, max_item))

        return action
