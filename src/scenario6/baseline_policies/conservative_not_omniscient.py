import numpy as np
from scipy.special import logsumexp

from baseline_policies.conservative import Conservative

EPS = np.finfo(np.float32).eps


class PsyGrid:

    LIN = 'lin'
    GEO = 'geo'

    METHODS = {LIN: np.linspace, GEO: np.geomspace}

    def __init__(self,
                 env,
                 bounds=np.asarray([[2e-07, 0.025], [0.0001, 0.9999]]),
                 grid_size=100,
                 grid_methods=('geo', 'lin'),
                 true_param=None):

        self.is_item_specific = len(np.unique(env.initial_forget_rates)) > 1
        self.n_item = env.n_item

        self.omniscient = true_param is not None
        if self.omniscient:
            self.est_param = true_param
        else:
            self.bounds = np.asarray(bounds)
            self.methods = np.asarray([self.METHODS[k] for k in grid_methods])
            self.grid_param = self.cp_grid_param(grid_size=grid_size)

            n_param_set, n_param = self.grid_param.shape

            lp = np.ones(n_param_set)
            lp -= logsumexp(lp)

            ep = np.dot(np.exp(lp), self.grid_param)

            if self.is_item_specific:
                log_post = np.zeros((self.n_item, n_param_set))
                log_post[:] = lp

                est_param = np.zeros((self.n_item, n_param))
                est_param[:] = ep

            else:
                log_post = lp
                est_param = ep

            self.log_post = log_post
            self.est_param = est_param

            self.n_param = n_param

            self.check_bounds(bounds, env)

            self.inferred_parameters = None

    def check_bounds(self, bounds, env):

        for item in range(self.n_item):
            assert bounds[0, 0] <= env.initial_forget_rates[item] < bounds[0, 1]
            assert bounds[1, 0] <= env.repetition_rates[item] < bounds[1, 1]

    @staticmethod
    def cartesian_product(*arrays):

        la = len(arrays)
        dtype = np.result_type(*arrays)
        arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
        for i, a in enumerate(np.ix_(*arrays)):
            arr[..., i] = a
        return arr.reshape(-1, la)

    def cp_grid_param(self, grid_size):

        diff = self.bounds[:, 1] - self.bounds[:, 0] > 0
        not_diff = np.invert(diff)

        values = np.atleast_2d(
            [m(*b, num=grid_size) for (b, m) in
             zip(self.bounds[diff], self.methods[diff])])

        var = self.cartesian_product(*values)
        grid = np.zeros((max(1, len(var)), len(self.bounds)))
        if np.sum(diff):
            grid[:, diff] = var
        if np.sum(not_diff):
            grid[:, not_diff] = self.bounds[not_diff, 0]

        return grid

    def update(self, item, success, delta, n_pres):

        if self.omniscient:
            return
        elif n_pres[item] == 0:
            return

        log_lik = self.log_lik_grid(
            item=item,
            grid_param=self.grid_param,
            success=success,
            delta=delta,
            n_pres=n_pres)

        if self.is_item_specific:
            lp = self.log_post[item]
        else:
            lp = self.log_post

        lp += log_lik
        lp -= logsumexp(lp)
        est_param = np.dot(np.exp(lp), self.grid_param)

        if self.is_item_specific:
            self.log_post[item] = lp
            self.est_param[item] = est_param
        else:
            self.log_post = lp
            self.est_param = est_param

    def inferred_learner_param(self, n_pres):

        if self.omniscient:
            return self.est_param.T
        elif not self.is_item_specific:
            est_param = np.zeros((self.n_item, self.n_param))
            for i in range(self.n_param):
                est_param[:, i] = self.est_param[i]
            return est_param.T  # Same shape as for item specific

        # Prior over unseen items is expectation when considering seen items

        is_rep = n_pres > 1
        not_is_rep = np.invert(is_rep)

        if np.sum(is_rep) == self.n_item or np.sum(not_is_rep) == self.n_item:
            return self.est_param.T

        lp_to_consider = self.log_post[is_rep]
        lp = logsumexp(lp_to_consider, axis=0) - np.log(lp_to_consider.shape[0])

        self.log_post[not_is_rep] = lp
        self.est_param[not_is_rep] = np.dot(np.exp(lp), self.grid_param)
        return self.est_param.T

    @staticmethod
    def log_lik_grid(
            item,
            delta,
            n_pres,
            grid_param,
            success):

        fr = grid_param[:, 0] \
             * (1 - grid_param[:, 1]) ** (n_pres[item] - 1)
        p_success = np.exp(- fr * delta[item])
        p = p_success if success else 1-p_success
        log_lik = np.log(p + EPS)
        return log_lik


class ConservativeNotOmniscient(Conservative):

    """
    Works only with the discontinuous environment
    """

    def __init__(self, env, *args, **kwargs):
        super().__init__(env=env)
        self.psy = PsyGrid(env=env, *args, **kwargs)

    def act(self, obs):

        env = self.env
        param = self.psy.inferred_learner_param(n_pres=env.n_pres)
        item = self.find_first_feasible_item(initial_forget_rates=param[0], repetition_rates=param[1])

        self.inferred_parameters = param

        # Simulate answer and update psychologist's beliefs ---
        if env.n_pres[item]:
            init_forget = env.initial_forget_rates[item]
            rep_effect = env.repetition_rates[item]

            rep = env.n_pres[item] - 1
            delta = env.delta[item]

            forget_rate = init_forget * \
                (1 - rep_effect) ** rep
            p = np.exp(- forget_rate * delta)
            success = p > np.random.random()

            self.psy.update(item=item, success=success, delta=env.delta, n_pres=env.n_pres)

        return item
