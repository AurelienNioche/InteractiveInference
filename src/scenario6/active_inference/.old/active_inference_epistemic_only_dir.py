import torch
from collections import namedtuple
from tqdm import tqdm
import numpy as np
import scipy

from scipy.special import logsumexp
from scipy.stats import entropy

from baseline_policies.conservative_not_omniscient import PsyGrid

EPS = np.finfo(np.float32).eps


class ActiveEpistemicOnlyDir:

    def __init__(self,
                 env,
                 param_kwargs=None,
                 *args, **kwargs):
        super().__init__()

        self.env = env
        self.psy = PsyGrid(env=env, *args, **kwargs)

        Param = namedtuple("Param", ["learning_rate", "max_epochs",
                                     "n_sample_cat", "n_sample_dir"],
                           defaults=[0.9, 500, 10, 10])
        if param_kwargs is None:
            param_kwargs = {}
        self.param = Param(**param_kwargs)
        self.policy = None

    def train(self):

        env = self.env
        t, t_max, n_item = env.t, env.t_max, env.n_item

        t_remaining = t_max - t

        n_epochs = self.param.max_epochs
        lr = self.param.learning_rate

        grid = self.psy.grid_param

        log_alpha = torch.rand((t_remaining, n_item))

        log_alpha = torch.nn.Parameter(log_alpha)

        opt = torch.optim.Adam([log_alpha, ], lr=lr)

        with tqdm(total=n_epochs, leave=False, position=1) as pbar:
            for epoch in range(n_epochs):

                # old_b = b.clone()
                opt.zero_grad()

                total_ig = 0
                dir_dist = torch.distributions.Dirichlet(log_alpha.exp())
                dir_samples = dir_dist.sample((self.param.n_sample_dir, ))

                logp_traj = 0

                for idx_dir, dir_smp in enumerate(dir_samples):

                    cat_dist = torch.distributions.Categorical(probs=dir_smp)
                    cat_samples = cat_dist.sample((self.param.n_sample_cat, ))
                    for idx_cat, cat_smp in enumerate(cat_samples):

                        n_pres = env.n_pres.copy()
                        delta = env.delta.copy()

                        current_iter, current_ss = env.current_iter, env.current_ss

                        prior_all = self.psy.log_post.copy()

                        for idx_t_sim, t_sim in enumerate(range(t_remaining)):

                            item = cat_smp[idx_t_sim]

                            if n_pres[item] == 0:
                                ig = 0

                            else:
                                if self.psy.is_item_specific:
                                    prior = prior_all[item]
                                else:
                                    prior = prior_all

                                init_fr, rep_effect = grid.T

                                logp_success = -init_fr * (1 - rep_effect) ** (n_pres[item] - 1) * delta[item]

                                post_success = prior + logp_success
                                post_success -= logsumexp(post_success)
                                ig_success = entropy(np.exp(post_success), np.exp(prior))

                                post_failure = prior + np.log(1 - np.exp(logp_success))
                                post_failure -= logsumexp(post_failure)
                                ig_failure = entropy(np.exp(post_failure), np.exp(prior))

                                marg_p_success = np.sum(np.exp(prior + logp_success))

                                # Expected info gain
                                ig = marg_p_success*ig_success + (1 - marg_p_success)*ig_failure

                                # Generate answer

                                idx_param = np.random.choice(np.arange(len(grid)), p=np.exp(prior))
                                success = np.exp(logp_success[idx_param]) > np.random.random()
                                if success:
                                    post = post_success
                                else:
                                    post = post_failure

                                # Make the post the prior

                                if self.psy.is_item_specific:
                                    prior_all[item] = post
                                    # Implement tweak for items not seen
                                else:
                                    prior_all = post

                            total_ig += ig

                            # Update
                            n_pres, delta, current_iter, current_ss = self.step(
                                n_pres=n_pres, delta=delta,
                                current_iter=current_iter, current_ss=current_ss, item=item)

                        logp_traj += cat_dist.log_prob(cat_smp).sum()

                    logp_traj += dir_dist.log_prob(dir_smp).sum()

                ig /= t_remaining
                loss = - (ig + logp_traj)

                # --------------------- #

                loss.backward()
                opt.step()

                pbar.update()
                pbar.set_postfix({f"loss": f"{loss.item():.2f}"})

                # if torch.isclose(old_b, b).all():
                #     break
        return torch.argmax(log_alpha[0])

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

        if env.t < 2:
            return 0  # First item doesn't matter

        item = self.train()

        # Simulate answer and update psychologist's beliefs ---
        if env.n_pres[item]:
            success = self.psy.generate_response(item=item)
            self.psy.update(item=item, success=success, delta=env.delta, n_pres=env.n_pres)

        return item
