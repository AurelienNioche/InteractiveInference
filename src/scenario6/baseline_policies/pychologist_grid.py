import numpy as np
from scipy.special import logsumexp

EPS = np.finfo(np.float).eps


class Exponential:

    DUMMY_VALUE = -1

    def __init__(self, n_item, n_iter):

        self.n_item = n_item

        self.seen = np.zeros(n_item, dtype=bool)
        self.ts = np.full(n_iter, self.DUMMY_VALUE, dtype=float)
        self.hist = np.full(n_iter, self.DUMMY_VALUE, dtype=int)
        self.seen_item = None
        self.n_seen = 0
        self.i = 0

        self.n_pres = np.zeros(n_item, dtype=int)
        self.last_pres = np.zeros(n_item, dtype=float)

    def p_seen(self, param, is_item_specific, now, cst_time):

        seen = self.n_pres >= 1
        if np.sum(seen) == 0:
            return np.array([]), seen

        if is_item_specific:
            init_forget = param[seen, 0]
            rep_effect = param[seen, 1]
        else:
            init_forget, rep_effect = param

        fr = init_forget * (1 - rep_effect) ** (self.n_pres[seen] - 1)

        last_pres = self.last_pres[seen]
        delta = now - last_pres

        delta *= cst_time
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            p = np.exp(-fr * delta)
        return p, seen

    @staticmethod
    def p_seen_spec_hist(param, now, hist, ts, seen, is_item_specific,
                         cst_time):

        if is_item_specific:
            init_forget = param[seen, 0]
            rep_effect = param[seen, 1]
        else:
            init_forget, rep_effect = param

        seen_item = np.flatnonzero(seen)

        n_seen = np.sum(seen)
        n_pres = np.zeros(n_seen)
        last_pres = np.zeros(n_seen)
        for i, item in enumerate(seen_item):
            is_item = hist == item
            n_pres[i] = np.sum(is_item)
            last_pres[i] = np.max(ts[is_item])

        fr = init_forget * (1-rep_effect) ** (n_pres - 1)

        delta = now - last_pres
        delta *= cst_time
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            p = np.exp(-fr * delta)
        return p, seen

    def log_lik_grid(self, item, grid_param, response, timestamp,
                     cst_time):

        fr = grid_param[:, 0] \
             * (1 - grid_param[:, 1]) ** (self.n_pres[item] - 1)

        delta = timestamp - self.last_pres[item]

        delta *= cst_time
        p_success = np.exp(- fr * delta)

        p = p_success if response else 1-p_success

        log_lik = np.log(p + EPS)
        return log_lik

    def p(self, item, param, now, is_item_specific, cst_time):

        if is_item_specific:
            init_forget = param[item, 0]
            rep_effect = param[item, 1]
        else:
            init_forget, rep_effect = param

        fr = init_forget * (1 - rep_effect) ** (self.n_pres[item] - 1)

        delta = now - self.last_pres[item]

        delta *= cst_time
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            p = np.exp(- fr * delta)
        return p

    def update(self, item, timestamp):

        self.last_pres[item] = timestamp
        self.n_pres[item] += 1

        self.hist[self.i] = item
        self.ts[self.i] = timestamp

        self.seen[item] = True

        self.seen_item = np.flatnonzero(self.seen)
        self.n_seen = np.sum(self.seen)

        self.i += 1


class PsyGrid:

    LIN = 'lin'
    GEO = 'geo'

    METHODS = {LIN: np.linspace, GEO: np.geomspace}

    def __init__(self, n_item, is_item_specific, n_iter,
                 bounds, grid_size, grid_methods, cst_time, true_param=None):

        self.omniscient = true_param is not None
        if not self.omniscient:
            self.bounds = np.asarray(bounds)
            self.methods = np.asarray([self.METHODS[k] for k in grid_methods])
            self.grid_param = self.cp_grid_param(grid_size=grid_size)

            n_param_set, n_param = self.grid_param.shape

            lp = np.ones(n_param_set)
            lp -= logsumexp(lp)

            ep = np.dot(np.exp(lp), self.grid_param)

            if is_item_specific:
                log_post = np.zeros((n_item, n_param_set))
                log_post[:] = lp

                est_param = np.zeros((n_item, n_param))
                est_param[:] = ep

            else:
                log_post = lp
                est_param = ep

            self.log_post = log_post
            self.est_param = est_param

            self.n_param = n_param

            self.n_pres = np.zeros(n_item, dtype=int)
            self.n_item = n_item

        else:
            self.est_param = true_param

        self.is_item_specific = is_item_specific
        self.cst_time = cst_time
        self.learner = Exponential(n_item=n_item, n_iter=n_iter)

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

    def update(self, item, response, timestamp):

        if not self.omniscient:
            if self.n_pres[item] == 0:
                pass
            else:
                log_lik = self.learner.log_lik_grid(
                    item=item,
                    grid_param=self.grid_param,
                    response=response,
                    timestamp=timestamp,
                    cst_time=self.cst_time)

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

            self.n_pres[item] += 1

        self.learner.update(timestamp=timestamp, item=item)

    def p_seen(self, now, param=None):
        if param is None:
            param = self.est_param

        return self.learner.p_seen(
            param=param,
            is_item_specific=self.is_item_specific,
            cst_time=self.cst_time,
            now=now)

    def inferred_learner_param(self):

        if self.omniscient or not self.is_item_specific:
            return self.est_param

        is_rep = self.n_pres > 1
        not_is_rep = np.invert(is_rep)

        if np.sum(is_rep) == self.n_item or np.sum(not_is_rep) == self.n_item:
            return self.est_param

        lp_to_consider = self.log_post[is_rep]
        lp = logsumexp(lp_to_consider, axis=0) - np.log(lp_to_consider.shape[0])

        self.log_post[not_is_rep] = lp
        self.est_param[not_is_rep] = np.dot(np.exp(lp), self.grid_param)

        return self.est_param

    def p(self, param, item, now):
        return self.learner.p(
            item=item,
            is_item_specific=self.is_item_specific,
            cst_time=self.cst_time,
            param=param,
            now=now)
