import torch
from collections import namedtuple
from tqdm import tqdm
import numpy as np
from copy import deepcopy

from baseline_policies.conservative import Conservative
from run import run


class ActivePragmaticOnly:

    def __init__(self,
                 env,
                 prior=None,
                 param_kwargs=None):
        super().__init__()

        self.env = env

        Param = namedtuple("Param", ["learning_rate", "max_epochs", "n_sample", "inv_temp",
                                     "optimizer", "optimizer_kwargs"],
                           defaults=[0.7, 10000, 200, 1.0, "RMSprop",
                                     dict()])
        if param_kwargs is None:
            param_kwargs = {}
        self.param = Param(**param_kwargs)
        self.policy = None
        self.prior = prior

    def build_prior(self):

        actions = []

        env = deepcopy(self.env)

        obs = env.reset()

        policy = Conservative(env=env)

        with tqdm(total=env.t_max, position=0, desc="Building prior") as pb:
            done = False
            while not done:
                action = policy.act(obs)
                obs, reward, done, _ = env.step(action)
                actions.append(action)
                pb.update()

        p = torch.zeros((env.t_max, env.n_item))
        for t in range(env.t_max):
            p[t] = torch.from_numpy(np.eye(env.n_item)[actions[t]])

        glitch = 0.4  # 1e-1
        logits = torch.logit(torch.clamp(p, glitch/(env.n_item-1), 1 - glitch))
        self.prior = logits

        # logits = torch.ones((t_max, n_item))

    def train(self):

        if self.prior is None:
            self.prior = torch.ones((self.env.t_max, self.env.n_item))
            # self.build_prior()

        env = self.env
        t_max, n_item = env.t_max, env.n_item
        init_forget_rates = torch.from_numpy(env.initial_forget_rates)
        rep_rates = torch.from_numpy(env.repetition_rates)
        init_forget_rates.requires_grad = True
        rep_rates.requires_grad = True
        threshold = np.exp(self.env.log_tau)

        n_epochs = self.param.max_epochs
        lr = self.param.learning_rate

        b = torch.nn.Parameter(self.prior)

        prior_dist = torch.distributions.Categorical(logits=self.prior)

        opt = getattr(torch.optim, self.param.optimizer)([b, ], lr=lr, **self.param.optimizer_kwargs)

        with tqdm(total=n_epochs, leave=True, position=0) as pbar:
            for epoch in range(n_epochs):

                old_b = b.clone()
                opt.zero_grad()

                loss = 0

                hist_n_learnt = []
                q_dist = torch.distributions.Categorical(logits=b - b.max())

                for _ in range(self.param.n_sample):
                    env.reset()

                    n_pres = torch.zeros(env.n_item)
                    delta = torch.zeros(env.n_item)

                    current_iter, current_ss = 0, 0

                    smp = q_dist.sample()

                    for t in range(t_max):

                        item = smp[t]
                        n_pres, delta, current_iter, current_ss = self.step(
                            n_pres=n_pres, delta=delta,
                            current_iter=current_iter, current_ss=current_ss, item=item)

                    log_p_seen = self._cp_log_p_seen(
                        n_pres=n_pres, delta=delta,
                        initial_forget_rates=init_forget_rates,
                        repetition_rates=rep_rates)

                    p = torch.zeros(env.n_item)
                    p[n_pres > 0] = log_p_seen.exp().float()

                    # print("n learnt", torch.sum(p > env.tau), "n_item", len(p))
                    learning_reward = torch.mean(torch.sigmoid(self.param.inv_temp*(p - threshold)))
                        # torch.log()

                    logp_traj = q_dist.log_prob(smp).sum()
                    pragmatic_value = torch.log(learning_reward) + logp_traj
                    loss -= pragmatic_value

                    # Just for display/debug
                    n_learnt = torch.sum(log_p_seen > self.env.log_tau)
                    hist_n_learnt.append(n_learnt.item())

                loss /= self.param.n_sample

                # Compute entropy ----------------------------------------------
                # ent = 0
                # for t in range(t_max):
                #     ent += torch.distributions.Categorical(logits=b[t]).entropy().exp() / n_item
                # ent /= t_max
                # loss += torch.log(ent)

                # --------------------------------------------------------------------------

                # ---------------------- #

                # loss += torch.distributions.kl_divergence(q_dist, prior_dist).mean()

                # --------------------- #

                loss.backward()
                opt.step()

                # if epoch == 0 and n_learnt == 0:
                #     raise Exception("Something is going wrong here")

                pbar.update()
                pbar.set_postfix({f"loss": f"{loss.item():.2f}, n_learnt={np.mean(hist_n_learnt)}"})

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

        if self.policy is None:
            self.train()

        return self.policy.sample()[self.env.t]
