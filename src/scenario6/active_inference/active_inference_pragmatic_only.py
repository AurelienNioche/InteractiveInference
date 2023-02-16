import torch
from collections import namedtuple
from tqdm import tqdm
import numpy as np

from torch import distributions as dist
from scipy.special import logsumexp, expit
from scipy.stats import entropy
from copy import deepcopy

from baseline_policies.conservative import Conservative
from run import run


class ActivePragmatic:

    def __init__(self,
                 env,
                 param_kwargs=None):
        super().__init__()

        self.env = env

        Param = namedtuple("Param", ["learning_rate", "max_epochs",
                                     "n_sample", "inv_temp",
                                     "optimizer", "optimizer_kwargs"],
                           defaults=[0.7, 10000, 200, 1.0, "RMSprop",
                                     dict()])
        if param_kwargs is None:
            param_kwargs = {}
        self.param = Param(**param_kwargs)

    # def build_prior(self):
    #
    #     actions = []
    #
    #     env = deepcopy(self.env)
    #
    #     obs = env.reset()
    #
    #     policy = Conservative(env=env)
    #
    #     with tqdm(total=env.t_max, position=0, desc="Building prior") as pb:
    #         done = False
    #         while not done:
    #             action = policy.act(obs)
    #             obs, reward, done, _ = env.step(action)
    #             actions.append(action)
    #             pb.update()
    #
    #     p = torch.zeros((env.t_max, env.n_item))
    #     for t in range(env.t_max):
    #         p[t] = torch.from_numpy(np.eye(env.n_item)[actions[t]])
    #
    #     glitch = 0.4  # 1e-1
    #     logits = torch.logit(torch.clamp(p, glitch/(env.n_item-1), 1 - glitch))
    #     self.prior = logits

    def train(self):

        env = self.env
        threshold = env.tau
        t, t_max, n_item = env.t, env.t_max, env.n_item

        t_remaining = t_max - t

        n_epochs = self.param.max_epochs
        lr = self.param.learning_rate
        n_sample = self.param.n_sample

        logits_action = torch.nn.Parameter(torch.ones((t_remaining, n_item)))

        opt = torch.optim.Adam([logits_action, ], lr=lr)

        prior_action = dist.Categorical(logits=torch.ones(n_item))

        delays = np.tile(
            [env.time_per_iter for _ in range(env.n_iter_per_session - 1)] + [env.break_length, ],
            env.n_session)[t:]

        n_step = t_remaining

        with tqdm(total=n_epochs, position=1, leave=False) as pbar:
            for epoch in range(n_epochs):

                opt.zero_grad()

                q = dist.Categorical(logits_action)
                trajectories = q.sample((n_sample, ))

                loss = 0

                for trajectory in trajectories:
                    traj = trajectory.numpy()
                    n_pres = env.n_pres.copy()
                    delta = env.delta.copy()

                    for item in range(n_item):
                        item_pres = traj == item
                        n_pres_traj = np.sum(item_pres)
                        n_pres[item] += n_pres_traj
                        if n_pres_traj == 0:
                            delta[item] += np.sum(delays)
                        else:
                            idx_last_pres = np.arange(n_step)[item_pres][-1]
                            delta[item] = np.sum(delays[idx_last_pres:])

                    p = np.zeros(env.n_item)

                    view = n_pres > 0
                    rep = n_pres[view] - 1.
                    delta = delta[view]

                    init_fr = env.initial_forget_rates[view]
                    rep_eff = env.repetition_rates[view]

                    forget_rate = init_fr * (1 - rep_eff) ** rep
                    logp_recall = - forget_rate * delta

                    p[n_pres > 0] = np.exp(logp_recall)

                    # print("n learnt", torch.sum(p > env.tau), "n_item", len(p))
                    learning_reward = expit(self.param.inv_temp * (p - threshold)).mean()

                    loss -= q.log_prob(trajectory).sum().exp() * learning_reward

                    for t in range(n_step):
                        q = dist.Categorical(logits_action[t])
                        loss += torch.distributions.kl_divergence(q, prior_action)

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
        return item
