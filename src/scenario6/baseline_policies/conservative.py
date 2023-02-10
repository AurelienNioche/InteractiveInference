import numpy as np


class Conservative:

    """
    Works only with the discontinuous environment
    """

    def __init__(self, env):
        self.env = env

    def _threshold_select(
            self, n_pres,
            initial_forget_rates,
            repetition_rates, n_item,
            delta):

        if np.max(n_pres) == 0:
            item = 0
        else:
            seen = n_pres > 0

            log_p_seen = self._cp_log_p_seen(
                n_pres=n_pres, delta=delta,
                initial_forget_rates=initial_forget_rates,
                repetition_rates=repetition_rates)

            if np.sum(seen) == n_item \
                    or np.min(log_p_seen) <= self.env.log_tau:

                item = np.flatnonzero(seen)[np.argmin(log_p_seen)]
            else:
                item = np.argmin(seen)

        return item

    @staticmethod
    def _cp_log_p_seen(
            n_pres,
            delta,
            initial_forget_rates,
            repetition_rates):

        view = n_pres > 0
        rep = n_pres[view] - 1.
        delta = delta[view]

        init_fr = initial_forget_rates[np.nonzero(view)]
        rep_eff = repetition_rates[np.nonzero(view)]

        forget_rate = init_fr * (1 - rep_eff) ** rep
        logp_recall = - forget_rate * delta
        return logp_recall

    def step(
            self,
            item,
            n_pres,
            delta,
            current_iter,
            current_ss,
    ):

        done = False

        # update progression within session, and between session
        # - which iteration the learner is at?
        # - which session the learner is at?
        current_iter += 1
        if current_iter >= self.env.n_iter_per_session:
            current_iter = 0
            current_ss += 1
            time_elapsed = self.env.break_length
        else:
            time_elapsed = self.env.time_per_iter

        if current_ss >= self.env.n_session:
            done = True

        # increase delta
        delta += time_elapsed
        # ...specific for item shown
        delta[item] = time_elapsed
        # increment number of presentation
        n_pres[item] += 1

        return n_pres, delta, current_iter, current_ss, done

    def find_first_feasible_item(self, initial_forget_rates, repetition_rates):

        current_iter = self.env.current_iter
        current_ss = self.env.current_ss

        n_item = self.env.n_item

        delta_current = self.env.delta
        n_pres_current = self.env.n_pres

        # Reduce the number of item to learn
        # until every item presented is learnable
        while True:

            n_pres = n_pres_current[:n_item]
            delta = delta_current[:n_item]

            first_item = self._threshold_select(
                n_pres=n_pres, delta=delta,
                initial_forget_rates=initial_forget_rates,
                repetition_rates=repetition_rates,
                n_item=n_item)

            n_item = first_item + 1

            n_pres = n_pres_current[:n_item].copy()
            delta = delta_current[:n_item].copy()

            n_pres, delta, current_iter, current_ss, done = \
                self.step(
                    item=first_item,
                    n_pres=n_pres,
                    delta=delta,
                    current_iter=current_iter,
                    current_ss=current_ss)

            # Do rollouts...
            while not done:
                item = self._threshold_select(
                    n_pres=n_pres, delta=delta,
                    initial_forget_rates=initial_forget_rates,
                    repetition_rates=repetition_rates,
                    n_item=n_item)

                n_pres, delta, current_iter, current_ss, done = \
                    self.step(item, n_pres, delta, current_iter,
                              current_ss)

            log_p_seen = self._cp_log_p_seen(
                n_pres=n_pres, delta=delta,
                initial_forget_rates=initial_forget_rates,
                repetition_rates=repetition_rates)

            n_learnt = np.sum(log_p_seen > self.env.log_tau)
            if n_learnt == n_item:
                break

            n_item = first_item
            if n_item <= 1:
                return 0

        return first_item

    def act(self, obs):

        initial_forget_rates = self.env.initial_forget_rates
        repetition_rates = self.env.repetition_rates

        return self.find_first_feasible_item(initial_forget_rates=initial_forget_rates,
                                             repetition_rates=repetition_rates)
