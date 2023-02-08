import numpy as np


DUMMY_VALUE = -1


class Conservative:

    # TODO: Adapt the code

    def __init__(self, n_item, learnt_threshold, time_per_iter,
                 n_ss, ss_n_iter, time_between_ss):

        self.n_item = n_item
        self.log_thr = np.log(learnt_threshold)

        self.eval_ts = n_ss * time_between_ss
        self.review_ts = np.hstack(
            [
                np.arange(x, x + (ss_n_iter * time_per_iter), time_per_iter)
                for x in np.arange(0, time_between_ss * n_ss, time_between_ss)
            ])

    def _threshold_select(self, n_pres, param, n_item, is_item_specific,
                          ts, last_pres):

        if np.max(n_pres) == 0:
            item = 0
        else:
            seen = n_pres > 0

            log_p_seen = self._cp_log_p_seen(
                seen=seen,
                n_pres=n_pres,
                param=param,
                n_item=n_item,
                is_item_specific=is_item_specific,
                last_pres=last_pres,
                ts=ts)

            if np.sum(seen) == n_item or np.min(log_p_seen) <= self.log_thr:
                item = np.flatnonzero(seen)[np.argmin(log_p_seen)]
            else:
                item = np.argmin(seen)

        return item

    @staticmethod
    def _cp_log_p_seen(seen, n_pres, param, n_item, is_item_specific,
                       last_pres, ts):

        if is_item_specific:
            init_forget = param[:n_item][seen, 0]
            rep_effect = param[:n_item][seen, 1]
        else:
            init_forget, rep_effect = param

        return \
            -init_forget \
            * (1 - rep_effect) ** (n_pres[seen] - 1) \
            * (ts - last_pres[seen])

    def _recursive_exp_decay(self, n_pres, last_pres,
                             future_ts, param, eval_ts,
                             is_item_specific):

        n_item = self.n_item

        now = future_ts[0]
        future = future_ts[1:]

        n_pres_current = n_pres
        last_pres_current = last_pres

        while True:

            n_pres = n_pres_current[:n_item]
            last_pres = last_pres_current[:n_item]

            first_item = self._threshold_select(
                n_pres=n_pres,
                param=param,
                n_item=n_item,
                is_item_specific=is_item_specific,
                ts=now, last_pres=last_pres)

            n_item = first_item + 1

            n_pres = n_pres_current[:n_item].copy()
            last_pres = last_pres_current[:n_item].copy()

            n_pres[first_item] += 1
            last_pres[first_item] = now

            for ts in future:

                item = self._threshold_select(
                    n_pres=n_pres,
                    param=param,
                    n_item=n_item,
                    is_item_specific=is_item_specific,
                    ts=ts, last_pres=last_pres)

                n_pres[item] += 1
                last_pres[item] = ts

            seen = n_pres > 0
            log_p_seen = self._cp_log_p_seen(
                seen=seen,
                n_pres=n_pres,
                param=param,
                n_item=n_item,
                is_item_specific=is_item_specific,
                last_pres=last_pres,
                ts=eval_ts)

            n_learnt = np.sum(log_p_seen > self.log_thr)
            if n_learnt == n_item:
                break

            n_item = first_item
            if n_item <= 1:
                break

        return first_item

    def ask(self, psy):

        param = psy.inferred_learner_param()
        hist = psy.learner.hist.copy()
        is_item_specific = psy.is_item_specific

        no_dummy = hist != DUMMY_VALUE
        current_step = np.sum(no_dummy)
        current_seen_item = np.unique(hist[no_dummy])

        future_ts = self.review_ts[current_step:]

        n_pres = np.zeros(self.n_item)
        last_pres = np.zeros(self.n_item)
        for i, item in enumerate(current_seen_item):
            is_item = hist == item
            n_pres[item] = np.sum(is_item)
            last_pres[item] = np.max(self.review_ts[is_item])

        item = self._recursive_exp_decay(
            is_item_specific=is_item_specific,
            future_ts=future_ts,
            eval_ts=self.eval_ts,
            param=param,
            n_pres=n_pres,
            last_pres=last_pres)

        return item
