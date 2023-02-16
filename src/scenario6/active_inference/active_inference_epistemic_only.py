import torch
from collections import namedtuple
from tqdm import tqdm
import numpy as np

from scipy.special import logsumexp
from scipy.stats import entropy
from torch import distributions as dist


from baseline_policies.conservative_not_omniscient import PsyGrid

EPS = np.finfo(np.float32).eps


class ActiveEpistemic:

    def __init__(self,
                 env,
                 param_kwargs=None,
                 *args, **kwargs):
        super().__init__()

        self.env = env
        self.psy = PsyGrid(env=env, *args, **kwargs)

        Param = namedtuple("Param", ["learning_rate", "max_epochs", "n_sample"],
                           defaults=[0.1, 100, 10])
        if param_kwargs is None:
            param_kwargs = {}
        self.param = Param(**param_kwargs)

    def train(self):

        env = self.env
        t, t_max, n_item = env.t, env.t_max, env.n_item

        t_remaining = t_max - t

        n_epochs = self.param.max_epochs
        lr = self.param.learning_rate
        n_sample = self.param.n_sample

        logits_action = torch.nn.Parameter(torch.ones((t_remaining, n_item)))

        opt = torch.optim.Adam([logits_action, ], lr=lr)

        prior_action = dist.Categorical(logits=torch.ones(n_item))

        prior = self.psy.log_post.copy()
        grid = self.psy.grid_param

        with tqdm(total=n_epochs, position=1, leave=False) as pbar:
            for epoch in range(n_epochs):

                opt.zero_grad()

                for smp in range(n_sample):

                    n_pres = env.n_pres.copy()
                    delta = env.delta.copy()

                    current_iter = env.current_iter
                    current_ss = env.current_ss

                    sum_terms = 0
                    sum_kl_div_prior = 0

                    for t in range(t_remaining):

                        q = dist.Categorical(logits=logits_action[t])
                        item = q.sample()

                        if n_pres[item] == 0:
                            expected_ig = 0.0

                        else:
                            init_fr, rep_effect = grid.T
                            logp_success = -init_fr * (1 - rep_effect) ** (n_pres[item] - 1) * delta[item]

                            post_success = prior + logp_success
                            post_success -= logsumexp(post_success)
                            ig_success = entropy(np.exp(post_success), np.exp(prior))

                            post_failure = prior + np.log(1 - np.exp(logp_success))
                            post_failure -= logsumexp(post_failure)
                            ig_failure = entropy(np.exp(post_failure), np.exp(prior))

                            marg_p_success = np.sum(np.exp(prior + logp_success))

                            expected_ig = marg_p_success * ig_success + (1 - marg_p_success) * ig_failure

                        sum_terms += q.log_prob(item).exp() * expected_ig

                        n_pres, delta, current_iter, current_ss, _ = self.env.update_state(
                            n_pres=n_pres, delta=delta,
                            current_iter=current_iter, current_ss=current_ss, item=item)

                        sum_kl_div_prior += torch.distributions.kl_divergence(q, prior_action)

                    loss = - torch.log(sum_terms+1e-16)

                loss.backward()
                opt.step()

                pbar.set_postfix({"loss": loss.item()})
                pbar.update()

        return np.argmax(logits_action.detach().numpy()[0])

    def act(self, obs):

        env = self.env

        if not np.sum(env.n_pres):
            return 0

        item = self.train()

        # Simulate answer and update psychologist's beliefs ---
        if env.n_pres[item]:
            success = self.psy.generate_response(item=item)
            self.psy.update(item=item, success=success, delta=env.delta, n_pres=env.n_pres)

        return item
