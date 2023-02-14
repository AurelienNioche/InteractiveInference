import torch
from collections import namedtuple
from tqdm import tqdm
import numpy as np
import scipy

from baseline_policies.conservative_not_omniscient import PsyGrid

EPS = np.finfo(np.float32).eps


class ActiveEpistemicOnly:

    def __init__(self,
                 env,
                 prior=None,
                 param_kwargs=None,
                 *args, **kwargs):
        super().__init__()

        self.env = env
        self.psy = PsyGrid(env=env, *args, **kwargs)

        Param = namedtuple("Param", ["learning_rate", "max_epochs", "n_sample"],
                           defaults=[0.2, 400, 5])
        if param_kwargs is None:
            param_kwargs = {}
        self.param = Param(**param_kwargs)
        self.policy = None
        self.prior = prior

    def train(self):

        env = self.env
        t, t_max, n_item = env.t, env.t_max, env.n_item

        t_remaining = t_max - t

        n_epochs = self.param.max_epochs
        lr = self.param.learning_rate

        if self.prior is None:
            logits = torch.ones((t_remaining, n_item))
        else:
            logits = torch.logit(torch.clamp(self.prior, 1e-3, 1 - 1e-3))

        b = torch.nn.Parameter(logits)

        opt = torch.optim.Adam([b, ], lr=lr)

        with tqdm(total=n_epochs, leave=False, position=1) as pbar:
            for epoch in range(n_epochs):

                old_b = b.clone()
                opt.zero_grad()

                loss = 0
                ig = 0
                dist = torch.distributions.Categorical(logits=b)

                for _ in range(self.param.n_sample):

                    n_pres = env.n_pres.copy()
                    delta = env.delta.copy()

                    current_iter, current_ss = env.current_iter, env.current_ss

                    pre = self.psy.log_post.copy()

                    smp = dist.sample()

                    for idx_t_sim, t_sim in enumerate(range(t_remaining)):

                        item = smp[idx_t_sim]

                        if self.psy.is_item_specific:
                            pre_item = pre[item]
                        else:
                            pre_item = pre
                        init_fr, rep_effect = np.dot(np.exp(pre_item), self.psy.grid_param)
                        p_success = np.exp(-init_fr*(1-rep_effect)**(n_pres[item]-1)*delta[item])
                        success = p_success > np.random.random()
                        if success:
                            p = p_success
                        else:
                            p = 1 - p_success

                        log_lik = np.log(p + EPS)

                        post_item = pre_item + log_lik
                        post_item -= scipy.special.logsumexp(post_item)

                        # Compute information gain / relative entropy

                        ig += scipy.stats.entropy(np.exp(post_item), np.exp(pre_item))

                        # Update
                        n_pres, delta, current_iter, current_ss = self.step(
                            n_pres=n_pres, delta=delta,
                            current_iter=current_iter, current_ss=current_ss, item=item)
                        if self.psy.is_item_specific:
                            pre[item] = post_item
                            # Implement tweak for items not seen
                        else:
                            pre = post_item

                    logp_traj = dist.log_prob(smp).sum()

                    loss -= logp_traj

                ig /= (self.param.n_sample*t_remaining)
                loss -= ig

                # --------------------- #

                loss.backward()
                opt.step()

                pbar.update()
                pbar.set_postfix({f"loss": f"{loss.item():.2f}"})

                # if torch.isclose(old_b, b).all():
                #     break

        self.policy = torch.distributions.Categorical(logits=b)

    @staticmethod
    def _cp_log_p_seen(
            n_pres,
            delta,
            initial_forget_rates,
            repetition_rates):

        view = n_pres > 0
        rep = n_pres[view] - 1.
        delta = delta[view]

        init_fr = initial_forget_rates[view]
        rep_eff = repetition_rates[view]

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
        # done = False

        env = self.env

        # update progression within session, and between session
        # - which iteration the learner is at?
        # - which session the learner is at?
        current_iter += 1
        if current_iter >= env.n_iter_per_session:
            current_iter = 0
            current_ss += 1
            time_elapsed = env.break_length
        else:
            time_elapsed = env.time_per_iter

        if current_ss >= env.n_session:
            pass
            # done = True

        # increase delta
        delta += time_elapsed
        # ...specific for item shown
        delta[item] = time_elapsed
        # increment number of presentation
        n_pres[item] += 1

        return n_pres, delta, current_iter, current_ss  # , done

    def act(self, obs):

        env = self.env

        if not np.sum(env.n_pres):
            return 0

        self.train()
        item = self.policy.sample()[0]

        # Simulate answer and update psychologist's beliefs ---
        if env.n_pres[item]:
            success = self.psy.generate_response(item=item)
            self.psy.update(item=item, success=success, delta=env.delta, n_pres=env.n_pres)

        return item
