from abc import ABC

import numpy as np
import torch
import gym


class AiAssistant(gym.Env, ABC):

    """
    x: target positions
    psi: latent state; preferences of the user for each target
    actions: moving one target closer, and moving away all the other targets
    b_star: preferences of the assistant; preferences relate to the distance to the preferred target
    """

    def __init__(self, n_targets,
                 user_model,
                 starting_x=0.5,
                 target_closer_step_size=0.1,
                 target_further_step_size=None,
                 min_x=0.0,
                 max_x=1.0,
                 inference_learning_rate=0.1,
                 inference_max_epochs=500,
                 decision_rule='active_inference',
                 decision_rule_parameters=None):

        super().__init__()

        self.n_targets = n_targets

        self.target_closer_step_size = target_closer_step_size

        if target_further_step_size is None:
            target_further_step_size = target_closer_step_size / (n_targets - 1)
        self.target_further_step_size = target_further_step_size

        self.user_model = user_model

        # Used for updating beliefs
        self.inference_max_epochs = inference_max_epochs
        self.inference_learning_rate = inference_learning_rate

        self.actions = np.arange(n_targets)  # With target to move

        self.min_x = min_x
        self.max_x = max_x

        self.starting_x = starting_x

        self.decision_rule = decision_rule
        if decision_rule_parameters is None:
            if decision_rule == "active_inference":
                decision_rule_parameters = dict(
                    decay_factor=0.9,
                    n_rollout=5,
                    n_step_per_rollout=2)
            elif decision_rule == "epsilon_rule":
                decision_rule_parameters = dict(
                    epsilon=0.1
                )
            elif decision_rule == "softmax":
                decision_rule_parameters = dict(
                    temperature=100
                )
            elif decision_rule == "random":
                decision_rule_parameters = dict()
            else:
                raise ValueError("Decision rule not recognized")

        self.decision_rule_parameters = decision_rule_parameters

        self.a = None  # Current action
        self.b = None  # Beliefs over user preferences
        self.t = None  # Iteration counter
        self.x = None  # Positions of the targets

    @property
    def belief(self):
        return torch.softmax(self.b, dim=0).detach().numpy().copy()

    @property
    def targets_position(self):
        return self.x.detach().numpy().copy()

    @property
    def action(self):
        return self.a

    def reset(self):

        self.a = None
        self.b = torch.ones(self.n_targets)
        self.x = torch.ones(self.n_targets) * self.starting_x
        observation = self.x  # np.random.choice(self.actions)
        return observation  # reward, done, info can't be included

    def step(self, action: np.ndarray):

        y = action  # sensory state is action of other user
        self.b, _ = self.revise_belief(y=y, b=self.b, x=self.x)
        self.act()

        observation = self.x
        done = False

        reward, info = None, None
        return observation, reward, done, info

    def revise_belief(self, y, b, x):
        """
        Update the belief based on a new observation
        """
        # Alternative solution but less efficient computationally
        # p = self.user_model.complain_prob(x)
        # p_y = p ** y * (1 - p) ** (1 - y)
        # log_p_y_under_q = (p_y.log() * q_prime).sum()
        #
        # log_q = torch.log_softmax(b - b.max(), dim=0)
        #
        # kl_div_q_p = torch.sum(q_prime * (q_prime.log() - log_q))
        # fe = - log_p_y_under_q + kl_div_q_p

        p = self.user_model.complain_prob(x)
        p_y = p ** y * (1 - p) ** (1 - y)

        q = torch.softmax(b - b.max(), dim=0)
        logp_yq = (p_y * q).log()
        logp_yq.requires_grad = True

        # Start by creating a new belief `b_prime` the previous belief `b`
        b_prime = torch.nn.Parameter(b.clone())

        loss = None

        opt = torch.optim.Adam([b_prime, ], lr=self.inference_learning_rate)

        # Minimise free energy
        for step in range(self.inference_max_epochs):

            old_b_prime = b_prime.clone()

            opt.zero_grad()

            q_prime = torch.softmax(b_prime - b_prime.max(), dim=0)
            loss = torch.sum(q_prime * (q_prime.log() - logp_yq))

            loss.backward()
            opt.step()

            if torch.isclose(old_b_prime, b_prime).all():
                # print(f"converged at step {step}")
                break

        return b_prime.detach(), loss.item()

    def update_target_positions(self, x, a):

        mask = np.ones(self.n_targets, dtype=bool)
        mask[a] = False

        x[mask] += self.target_further_step_size
        x[a] -= self.target_closer_step_size
        x[x > self.max_x] = self.max_x
        x[x < self.min_x] = self.min_x

        return x

    def act(self):

        i = getattr(self, f'_act_{self.decision_rule}')(**self.decision_rule_parameters)
        self.a = self.actions[i]
        self.update_target_positions(x=self.x, a=self.a)

    def _act_random(self):
        return np.random.choice(np.arange(self.n_targets))

    def _act_epsilon_rule(self, epsilon):
        rd = np.random.random()
        if rd < epsilon:
            i = np.random.randint(self.n_targets)
        else:
            i = np.random.choice(np.nonzero(self.b == torch.max(self.b))[0])

        return i

    def _act_softmax(self, temperature):
        p = torch.nn.functional.softmax(temperature * self.b, dim=0).numpy()
        i = np.random.choice(np.arange(self.n_targets), p=p)
        return i

    def _act_active_inference(self,
                              decay_factor,
                              n_rollout,
                              n_step_per_rollout, efe='efe_on_obs'):

        efe = getattr(self, efe)

        kl_div = np.asarray([self.rollout(
            action=a,
            n_rollout=n_rollout,
            n_step_per_rollout=n_step_per_rollout,
            decay_factor=decay_factor,
            efe=efe,
        ) for a in self.actions])
        i = np.random.choice(np.nonzero(kl_div == np.min(kl_div))[0])

        # i = np.random.choice(np.arange(len(self.actions)),
        #                      p=torch.softmax(kl_div, dim=0).numpy())
        return i

    def rollout(self, action, n_rollout, n_step_per_rollout, decay_factor, efe):

        total_efe = 0
        for _ in range(n_rollout):
            efe_rollout = 0

            x = self.x.clone()
            b = self.b.clone()

            action_plan = np.zeros(n_step_per_rollout, dtype=int)
            action_plan[0] = action
            if n_step_per_rollout > 1:
                action_plan[1:] = np.random.choice(self.actions, size=n_step_per_rollout-1)

            for step, action in enumerate(action_plan):

                self.update_target_positions(x=x, a=action)

                efe_step, p_y_under_q, b_new = efe(x=x, b=b)

                efe_rollout += decay_factor**step * efe_step

                if n_step_per_rollout > 1:
                    y = torch.bernoulli(p_y_under_q[1]).long()
                    b = b_new[y]

            total_efe += efe_rollout.item()

        total_efe /= n_rollout
        return total_efe

    def efe_on_obs(self, x, b):

        epistemic_value, p_y_under_q, b_new = self.epistemic_value(b, x)

        extrinsic_value = p_y_under_q[1] * torch.tensor([1e-8]).log()

        efe_step = - extrinsic_value - epistemic_value

        return efe_step, p_y_under_q, b_new

    def efe_on_latent(self, x, b,
                      scale_objective=0.05):

        # --- Compute epistemic value
        epistemic_value, p_y_under_q, b_new = self.epistemic_value(b, x)

        # --- Compute extrinsic value
        objective_dist = torch.distributions.HalfNormal(scale_objective)
        log_p = (objective_dist.log_prob(x) * b).sum()
        extrinsic_value = log_p

        efe_step = - extrinsic_value - epistemic_value

        return efe_step, p_y_under_q, b_new

    def epistemic_value(self, b, x):

        ep_value = torch.zeros(2)
        b_new = torch.zeros((2, len(self.b)))

        y = torch.arange(2)
        p = self.user_model.complain_prob(x)
        q = torch.softmax(b - b.max(), dim=0)
        p_y = p ** y.unsqueeze(dim=1) * (1 - p) ** (1 - y.unsqueeze(dim=1))
        p_y_under_q = (p_y * q).sum(1)

        for i in y:
            b_new_i, kl_div_i = self.revise_belief(y=i, b=b, x=x)
            b_new[i] = b_new_i
            ep_value[i] = kl_div_i

        epistemic_value = (p_y_under_q * ep_value).sum()
        return epistemic_value, p_y_under_q, b_new
