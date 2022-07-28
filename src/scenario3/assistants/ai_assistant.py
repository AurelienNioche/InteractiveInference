from abc import ABC

import numpy as np
import torch
import gym
from tqdm import tqdm


class AiAssistant(gym.Env, ABC):

    """
    x: position of the targets
    psi: latent state; preferences of the user for each target
    actions: moving one target closer, and moving away all the other targets
    b_star: preferences of the assistant; preferences relate to the distance to the preferred target
    """

    def __init__(self,
                 user_model,
                 starting_belief=0.0,
                 starting_target_positions=0.5,
                 inference_learning_rate=0.1,
                 inference_max_epochs=500,
                 inference_n_sample=1000,
                 decision_rule='active_inference',
                 decision_rule_parameters=None):

        super().__init__()

        self.n_targets = user_model.n_targets

        self.user_model = user_model

        # Used for updating beliefs
        self.inference_max_epochs = inference_max_epochs
        self.inference_learning_rate = inference_learning_rate
        self.inference_n_sample = inference_n_sample

        # self.actions = np.arange(n_targets)  # With target to move

        # self.min_x = min_x
        # self.max_x = max_x

        self.starting_target_positions = starting_target_positions
        self.starting_belief = starting_belief

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

        self.a = None  # Current action / Positions of the target
        self.b = None  # Beliefs over user preferences (distributions parameterization)
        self.t = None  # Iteration counter

    @property
    def belief(self):
        return self.b.detach().numpy()

    @property
    def action(self):
        return self.a.detach().numpy()

    def reset(self):

        self.b = torch.ones((self.n_targets, 2)) * self.starting_belief
        self.a = torch.ones(self.n_targets) * self.starting_target_positions
        observation = self.a
        return observation  # reward, done, info can't be included

    def step(self, action: np.ndarray):

        y = action  # sensory state is action of other user

        self.b = self.revise_belief(y=y, b=self.b, a=self.a,
                                    conditional_probability_action=self.user_model.conditional_probability_action,
                                    lr=self.inference_learning_rate,
                                    max_n_epochs=self.inference_max_epochs)
        self.a = getattr(self, f'_act_{self.decision_rule}')(**self.decision_rule_parameters)

        observation = self.a
        done = False

        reward, info = None, None
        return observation, reward, done, info

    @staticmethod
    def revise_belief(b, a, y, conditional_probability_action,
                      lr=0.1,
                      max_n_epochs=500,
                      n_sample=1000):

        n_targets = len(a)

        # Start by creating a new belief `b_prime` the previous belief `b`
        b_prime = torch.nn.Parameter(b.clone())

        prior_mu, prior_log_var = b.T
        prior_std = (0.5 * prior_log_var).exp()

        p_dist = torch.distributions.MultivariateNormal(prior_mu, scale_tril=torch.diag(prior_std))

        opt = torch.optim.Adam([b_prime, ], lr=lr)

        # Minimise free energy
        for step in range(max_n_epochs):

            old_b_prime = b_prime.clone()

            opt.zero_grad()

            mu, log_var = b_prime.T
            std = (0.5 * log_var).exp()
            eps = torch.randn((n_sample, n_targets))

            z = eps * std + mu
            z_scaled = torch.sigmoid(z)

            distance = torch.absolute(z_scaled - a)
            average_distance = distance.mean(axis=-1)

            p = conditional_probability_action(average_distance)
            accuracy = (p ** y * (1 - p) ** (1 - y) + 1e-8).log().mean()

            q_dist = torch.distributions.MultivariateNormal(mu, scale_tril=torch.diag(std))

            complexity = torch.distributions.kl.kl_divergence(q_dist, p_dist)
            loss = - accuracy + complexity

            loss.backward()
            opt.step()

            if torch.isclose(old_b_prime, b_prime).all():
                # print(f"converged at step {step}")
                break

        return b_prime.detach()

    def _act_epsilon_rule(self, epsilon):
        """
        Expectation maximization + epsilon rule
        :param epsilon:
        :return:
        """
        rd = torch.rand(1)
        if rd[0] < epsilon:
            pos = torch.rand(self.n_targets)
        else:
            pos = torch.zeros(self.n_targets)
            for t in range(self.n_targets):
                mu, _ = self.b[t]
                mu_scaled = torch.sigmoid(mu)
                pos[t] = mu_scaled
        return pos

    def _act_random(self):

        return torch.rand(self.n_targets)

    def _act_active_inference(self, action_max_epochs,
                              action_learning_rate, *args, **kwargs):

        a = torch.nn.Parameter(self.a.clone())
        # b.requires_grad = True

        opt = torch.optim.Adam([a, ], lr=action_learning_rate)

        # Minimise free energy
        for _ in tqdm(range(action_max_epochs), leave=False):

            old_a = a.clone()

            opt.zero_grad()

            # torch.autograd.set_detect_anomaly(True)

            loss = self.expected_free_energy_action(a=a, *args, **kwargs)
            loss.backward()
            opt.step()

            if torch.isclose(old_a, a).all():
                break

        return a.detach()

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
