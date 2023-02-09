import numpy as np


class Leitner:

    def __init__(self, env,  delay_factor=2, delay_min=4):

        self.n_item = env.n_item

        self.delay_factor = delay_factor
        self.delay_min = delay_min

        self.box = np.full(self.n_item, -1)
        self.due = np.full(self.n_item, -1)

        # Adaptation for RL env
        self._env = env
        self._last_was_success = None
        self._last_time_reply = None
        self._idx_last_q = None
        self._now = 0
        self._current_iter = 0
        self._current_ss = 0

    def update_box_and_due_time(self, last_idx,
                                last_was_success, last_time_reply):

        if last_was_success:
            self.box[last_idx] += 1
        else:
            self.box[last_idx] = \
                max(0, self.box[last_idx] - 1)

        delay = self.delay_factor ** self.box[last_idx]
        # If delay_factor =2,
        # then delay for each box is 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 ...
        self.due[last_idx] = \
            last_time_reply + self.delay_min * delay

    def _pickup_item(self, now):

        seen = np.argwhere(np.asarray(self.box) >= 0).flatten()
        n_seen = len(seen)

        if n_seen == self.n_item:
            return np.argmin(self.due)

        else:
            seen__due = np.asarray(self.due)[seen]
            seen__is_due = np.asarray(seen__due) <= now
            if np.sum(seen__is_due):
                seen_and_is_due__due = seen__due[seen__is_due]

                return seen[seen__is_due][np.argmin(seen_and_is_due__due)]
            else:
                return self._pickup_new()

    def _pickup_new(self):
        return np.argmin(self.box)

    def ask(self, now, last_was_success, last_time_reply, idx_last_q):

        if idx_last_q is None:
            item_idx = self._pickup_new()

        else:

            self.update_box_and_due_time(
                last_idx=idx_last_q,
                last_was_success=last_was_success,
                last_time_reply=last_time_reply)
            item_idx = self._pickup_item(now)

        return item_idx

    def _step(self):

        # update progression within session, and between session
        # - which iteration the learner is at?
        # - which session the learner is at?
        self._current_iter += 1
        if self._current_iter >= self._env.n_iter_per_session:
            self._current_iter = 0
            self._current_ss += 1
            time_elapsed = self._env.break_length
        else:
            time_elapsed = self._env.time_per_iter

        self._last_time_reply = self._now
        self._now += time_elapsed

    def act(self, obs):

        """
        Adaptation for RL env
        """

        item = self.ask(
            now=self._now,
            last_was_success=self._last_was_success,
            last_time_reply=self._last_time_reply,
            idx_last_q=self._idx_last_q)

        delta_current = self._env.delta
        n_pres_current = self._env.n_pres

        if n_pres_current[item] == 0:
            success = False

        else:
            init_forget = self._env.initial_forget_rates[item]
            rep_effect = self._env.repetition_rates[item]

            rep = n_pres_current[item] - 1
            delta = delta_current[item]

            forget_rate = init_forget * \
                (1 - rep_effect) ** rep
            p = np.exp(- forget_rate * delta)
            success = p > np.random.random()

        self._step()
        self._idx_last_q = item
        self._last_was_success = success
        return item
