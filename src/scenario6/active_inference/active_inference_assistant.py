import torch
from collections import namedtuple
from tqdm import tqdm
import numpy as np


class ActiveTeacher:

    """
    x: target positions
    psi: latent state; preferences of the user for each target
    actions: moving one target closer, and moving away all the other targets
    b_star: preferences of the assistant; preferences relate to the distance to the preferred target
    """

    def __init__(self,
                 env,
                 prior=None,
                 param_kwargs=None):
        super().__init__()

        self.env = env

        Param = namedtuple("Param", ["learning_rate", "max_epochs", "n_sample"],
                           defaults=[0.1, 400, 50])
        if param_kwargs is None:
            param_kwargs = {}
        self.param = Param(**param_kwargs)
        self.policy = None
        self.prior = prior

    def train(self):

        env = self.env
        t_max, n_item = env.t_max, env.n_item
        init_forget_rates = torch.from_numpy(env.initial_forget_rates)
        rep_rates = torch.from_numpy(env.repetition_rates)
        init_forget_rates.requires_grad = True
        rep_rates.requires_grad = True
        threshold = np.exp(self.env.log_tau)

        n_epochs = self.param.max_epochs
        lr = self.param.learning_rate

        if self.prior is None:
            logits = torch.ones((t_max, n_item))
        else:
            logits = torch.logit(torch.clamp(self.prior, 1e-1, 1 - 1e-1))

        b = torch.nn.Parameter(logits)

        opt = torch.optim.Adam([b, ], lr=lr)

        with tqdm(total=n_epochs, leave=True, position=0) as pbar:
            for epoch in range(n_epochs):

                old_b = b.clone()
                opt.zero_grad()

                loss = 0

                # Compute entropy
                ent = 0
                for t in range(t_max):
                    ent += torch.distributions.Categorical(logits=b[t]).entropy().exp() / n_item
                ent /= t_max
                log_ent = torch.log(ent)
                # ----------------

                hist_n_learnt = []
                dist = torch.distributions.Categorical(logits=b)

                for _ in range(self.param.n_sample):
                    env.reset()

                    n_pres = torch.zeros(env.n_item)
                    delta = torch.zeros(env.n_item)

                    current_iter, current_ss = 0, 0

                    smp = dist.sample()

                    for t in range(t_max):

                        item = smp[t]
                        n_pres, delta, current_iter, current_ss = self.step(
                            n_pres=n_pres, delta=delta,
                            current_iter=current_iter, current_ss=current_ss, item=item)
                        # traj.append(item.item())
                        # logp_traj += dist.log_prob(item)

                    log_p_seen = self._cp_log_p_seen(
                        n_pres=n_pres, delta=delta,
                        initial_forget_rates=init_forget_rates,
                        repetition_rates=rep_rates)

                    learning_reward = torch.mean(torch.sigmoid(3e2 * (log_p_seen.exp() - threshold)))

                    logp_traj = dist.log_prob(smp).sum()

                    loss -= torch.log(learning_reward+1e-16) + logp_traj

                    # Just for display/debug
                    n_learnt = torch.sum(log_p_seen > self.env.log_tau)
                    hist_n_learnt.append(n_learnt.item())

                loss /= self.param.n_sample

                loss += log_ent

                # exit(0)

                loss.backward()
                opt.step()

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
